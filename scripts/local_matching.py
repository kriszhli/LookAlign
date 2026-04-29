"""LookAlign V0.3.6 local diffuse mood matching.

This stage is intentionally deterministic and lightweight. It approximates a
colorful diffuse-shading transfer by operating on smooth CIE Lab maps while
leaving base-intermediate detail in place.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from scripts.global_matching import (
    Tensor,
    image_stats_from_lab,
    lab_to_rgb,
    save_gray,
    save_rgb,
    soft_gamut_compress,
    to_hwc_np,
)


ALIGNMENT_KEYS = (
    "ref_residual_confidence",
    "specular_confidence",
    "shadow_confidence",
    "hue_stability_confidence",
    "delta_sanity_confidence",
    "luma_confidence",
    "hue_confidence",
    "chroma_confidence",
    "combined_confidence",
    "ref_residual_rejection",
)


@dataclass
class LocalMatchingConfig:
    enable_lightglue: bool = True
    match_long_edge: int = 512
    max_keypoints: int = 1024
    min_match_score: float = 0.20
    ransac_reproj_threshold: float = 8.0
    min_valid_matches: int = 24
    sparse_map_long_edge: int = 384
    fallback_global_blend: float = 0.20
    sparse_sigma: float = 0.035
    sparse_density_blend_low: float = 0.005
    sparse_density_blend_high: float = 0.060
    map_smoothing_passes: int = 1
    map_smoothing_strength: float = 0.35
    diffuse_proxy_long_edge: int = 320
    diffuse_blur_passes: int = 4
    max_diffuse_luma_delta: float = 10.0
    max_diffuse_hue_delta: float = 0.35
    min_diffuse_chroma_scale: float = 0.80
    max_diffuse_chroma_scale: float = 1.25
    low_chroma_start: float = 1.5
    low_chroma_end: float = 6.0
    clipped_start: float = 0.965
    clipped_end: float = 0.995
    shadow_start: float = 2.0
    shadow_end: float = 12.0
    delta_sanity_luma: float = 18.0
    delta_sanity_hue: float = 0.700
    delta_sanity_chroma_scale: float = 0.600
    confidence_blur_passes: int = 6


def smoothstep(edge0: float, edge1: float, x: Tensor) -> Tensor:
    t = ((x - float(edge0)) / max(float(edge1) - float(edge0), 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def resize_long_edge_unclamped(img: Tensor, long_edge: int) -> Tensor:
    _, _, height, width = img.shape
    scale = min(1.0, float(max(16, int(long_edge))) / float(max(height, width)))
    if scale >= 0.999:
        return img.clone()
    target_h = max(8, int(round(height * scale)))
    target_w = max(8, int(round(width * scale)))
    return F.interpolate(img, size=(target_h, target_w), mode="bicubic", align_corners=False)


def resize_to_hw_unclamped(img: Tensor, height: int, width: int) -> Tensor:
    return F.interpolate(img, size=(int(height), int(width)), mode="bicubic", align_corners=False)


def resize_long_edge_with_scale(img: Tensor, long_edge: int) -> tuple[Tensor, float]:
    _, _, height, width = img.shape
    scale = min(1.0, float(max(16, int(long_edge))) / float(max(height, width)))
    if scale >= 0.999:
        return img.clone(), 1.0
    target_h = max(8, int(round(height * scale)))
    target_w = max(8, int(round(width * scale)))
    return F.interpolate(img, size=(target_h, target_w), mode="bicubic", align_corners=False).clamp(0.0, 1.0), scale


def low_frequency_proxy_lab(lab: Tensor, cfg: LocalMatchingConfig) -> Tensor:
    _, _, height, width = lab.shape
    proxy = resize_long_edge_unclamped(lab, cfg.diffuse_proxy_long_edge)
    for _ in range(max(1, int(cfg.diffuse_blur_passes))):
        proxy = F.avg_pool2d(F.pad(proxy, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
    return resize_to_hw_unclamped(proxy, height, width)


def low_frequency_proxy_oklab(lab: Tensor, cfg: LocalMatchingConfig) -> Tensor:
    return low_frequency_proxy_lab(lab, cfg)


def chroma_and_hue(lab: Tensor) -> tuple[Tensor, Tensor]:
    ab = lab[:, 1:3]
    chroma = ab.norm(dim=1, keepdim=True).clamp_min(1e-6)
    hue = torch.atan2(ab[:, 1:2], ab[:, 0:1])
    return chroma, hue


def shortest_angle_delta(target: Tensor, source: Tensor) -> Tensor:
    return torch.atan2(torch.sin(target - source), torch.cos(target - source))


def signed_map_visual(x: Tensor, limit: float) -> np.ndarray:
    return ((x / max(float(limit), 1e-6)) * 0.5 + 0.5).detach().cpu().numpy()[0, 0].astype(np.float32)


def range_map_visual(x: Tensor, lo: float, hi: float) -> np.ndarray:
    return ((x - float(lo)) / max(float(hi) - float(lo), 1e-6)).clamp(0.0, 1.0).detach().cpu().numpy()[0, 0].astype(np.float32)


def gray_visual(x: Tensor) -> np.ndarray:
    return x.clamp(0.0, 1.0).detach().cpu().numpy()[0, 0].astype(np.float32)


def percentile_threshold(x: Tensor, q: float) -> Tensor:
    return torch.quantile(x.flatten(), float(q) / 100.0)


def smooth_confidence_from_range(x: Tensor, low: Tensor | float, high: Tensor | float) -> Tensor:
    low_t = torch.as_tensor(low, device=x.device, dtype=x.dtype)
    high_t = torch.as_tensor(high, device=x.device, dtype=x.dtype)
    t = ((x - low_t) / (high_t - low_t).clamp_min(1e-6)).clamp(0.0, 1.0)
    return (1.0 - (t * t * (3.0 - 2.0 * t))).clamp(0.0, 1.0)


def smooth_confidence_map(confidence: Tensor, cfg: LocalMatchingConfig) -> Tensor:
    _, _, height, width = confidence.shape
    x = resize_long_edge_unclamped(confidence, cfg.diffuse_proxy_long_edge)
    for _ in range(max(1, int(cfg.confidence_blur_passes))):
        x = F.avg_pool2d(F.pad(x, (3, 3, 3, 3), mode="replicate"), kernel_size=7, stride=1)
    return resize_to_hw_unclamped(x, height, width).clamp(0.0, 1.0)


def reference_specular_confidence(ref_lab: Tensor, ref_rgb: Tensor, cfg: LocalMatchingConfig) -> Tensor:
    ref_chroma, _ = chroma_and_hue(ref_lab)
    max_rgb = ref_rgb.max(dim=1, keepdim=True).values
    min_rgb = ref_rgb.min(dim=1, keepdim=True).values
    saturation = (max_rgb - min_rgb) / max_rgb.clamp_min(1e-4)
    bright = smoothstep(cfg.clipped_start, cfg.clipped_end, max_rgb)
    low_saturation = 1.0 - smoothstep(0.08, 0.22, saturation)
    low_chroma = 1.0 - smoothstep(4.0, 12.0, ref_chroma)
    specular = (bright * torch.maximum(low_saturation, low_chroma)).clamp(0.0, 1.0)
    return 1.0 - specular


def shadow_confidence(ref_proxy: Tensor, cfg: LocalMatchingConfig) -> Tensor:
    return smoothstep(cfg.shadow_start, cfg.shadow_end, ref_proxy[:, 0:1])


def hue_stability_confidence(base_proxy: Tensor, ref_proxy: Tensor, cfg: LocalMatchingConfig) -> Tensor:
    base_chroma, _ = chroma_and_hue(base_proxy)
    ref_chroma, _ = chroma_and_hue(ref_proxy)
    return smoothstep(cfg.low_chroma_start, cfg.low_chroma_end, torch.minimum(base_chroma, ref_chroma))


@lru_cache(maxsize=2)
def get_lightglue_models(device_name: str, max_keypoints: int) -> tuple[Any, Any, Any, str]:
    try:
        from lightglue import ALIKED, LightGlue
        from lightglue.utils import rbd
    except ImportError as exc:
        raise RuntimeError(
            "LightGlue is required for V0.3.6 correspondence-guided local matching. "
            "Install it with `python3 -m pip install git+https://github.com/cvg/LightGlue.git`, "
            "or run with LocalMatchingConfig(enable_lightglue=False) for the conservative fallback."
        ) from exc

    device = torch.device(device_name)
    extractor = ALIKED(max_num_keypoints=int(max_keypoints)).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
    return extractor, matcher, rbd, device_name


def extract_lightglue_features(extractor: Any, image: Tensor) -> Dict[str, Tensor]:
    try:
        return extractor.extract(image, resize=None)
    except TypeError:
        return extractor.extract(image)


def run_lightglue_matches_on_device(base_rgb: Tensor, ref_rgb: Tensor, cfg: LocalMatchingConfig, device_name: str) -> Dict[str, Any]:
    extractor, matcher, rbd, actual_device = get_lightglue_models(device_name, int(cfg.max_keypoints))
    base_small, base_scale = resize_long_edge_with_scale(base_rgb, cfg.match_long_edge)
    ref_small, ref_scale = resize_long_edge_with_scale(ref_rgb, cfg.match_long_edge)
    image0 = base_small[0].to(actual_device)
    image1 = ref_small[0].to(actual_device)

    with torch.no_grad():
        feats0 = extract_lightglue_features(extractor, image0)
        feats1 = extract_lightglue_features(extractor, image1)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

    matches = matches01["matches"]
    if matches.numel() == 0:
        empty = torch.empty((0, 2), device=base_rgb.device, dtype=base_rgb.dtype)
        return {
            "base_points": empty,
            "ref_points": empty,
            "scores": torch.empty((0,), device=base_rgb.device, dtype=base_rgb.dtype),
            "raw_count": 0,
            "score_filtered_count": 0,
            "inlier_count": 0,
            "device": actual_device,
            "homography": None,
        }

    points0 = feats0["keypoints"][matches[:, 0]]
    points1 = feats1["keypoints"][matches[:, 1]]
    if "scores" in matches01:
        scores = matches01["scores"]
    elif "matching_scores0" in matches01:
        scores = matches01["matching_scores0"][matches[:, 0]]
    else:
        scores = torch.ones((matches.shape[0],), device=points0.device, dtype=points0.dtype)

    raw_count = int(matches.shape[0])
    keep = scores >= float(cfg.min_match_score)
    points0 = points0[keep]
    points1 = points1[keep]
    scores = scores[keep]
    score_filtered_count = int(points0.shape[0])

    if score_filtered_count >= 4:
        import cv2

        p0_np = (points0.detach().cpu().numpy() / max(base_scale, 1e-6)).astype(np.float32)
        p1_np = (points1.detach().cpu().numpy() / max(ref_scale, 1e-6)).astype(np.float32)
        homography, mask = cv2.findHomography(p0_np, p1_np, cv2.RANSAC, float(cfg.ransac_reproj_threshold))
        if mask is None:
            inlier_mask = np.ones((score_filtered_count,), dtype=bool)
        else:
            inlier_mask = mask.reshape(-1).astype(bool)
        inlier_t = torch.from_numpy(inlier_mask).to(device=points0.device, dtype=torch.bool)
        points0 = points0[inlier_t]
        points1 = points1[inlier_t]
        scores = scores[inlier_t]

    base_points = (points0 / max(base_scale, 1e-6)).to(base_rgb.device, dtype=base_rgb.dtype)
    ref_points = (points1 / max(ref_scale, 1e-6)).to(base_rgb.device, dtype=base_rgb.dtype)
    return {
        "base_points": base_points,
        "ref_points": ref_points,
        "scores": scores.to(base_rgb.device, dtype=base_rgb.dtype),
        "raw_count": raw_count,
        "score_filtered_count": score_filtered_count,
        "inlier_count": int(base_points.shape[0]),
        "device": actual_device,
        "homography": homography if score_filtered_count >= 4 else None,
    }


def run_lightglue_matches(base_rgb: Tensor, ref_rgb: Tensor, cfg: LocalMatchingConfig) -> Dict[str, Any]:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            return run_lightglue_matches_on_device(base_rgb, ref_rgb, cfg, "mps")
        except RuntimeError as exc:
            if "LightGlue is required" in str(exc):
                raise
    return run_lightglue_matches_on_device(base_rgb, ref_rgb, cfg, "cpu")


def sample_points(img: Tensor, points_xy: Tensor) -> Tensor:
    if points_xy.numel() == 0:
        return torch.empty((0, img.shape[1]), device=img.device, dtype=img.dtype)
    _, _, height, width = img.shape
    px = points_xy[:, 0].clamp(0.0, float(max(width - 1, 0)))
    py = points_xy[:, 1].clamp(0.0, float(max(height - 1, 0)))
    x = px / max(width - 1, 1) * 2.0 - 1.0
    y = py / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack((x, y), dim=1).view(1, -1, 1, 2)
    values = F.grid_sample(img, grid, mode="bilinear", align_corners=True)
    return values[0, :, :, 0].transpose(0, 1)


def proxy_delta_maps(base_proxy: Tensor, ref_proxy: Tensor, cfg: LocalMatchingConfig) -> Dict[str, Tensor]:
    base_chroma, base_hue = chroma_and_hue(base_proxy)
    ref_chroma, ref_hue = chroma_and_hue(ref_proxy)
    luma_delta = (ref_proxy[:, 0:1] - base_proxy[:, 0:1]).clamp(
        -float(cfg.max_diffuse_luma_delta), float(cfg.max_diffuse_luma_delta)
    )
    hue_delta = shortest_angle_delta(ref_hue, base_hue).clamp(
        -float(cfg.max_diffuse_hue_delta), float(cfg.max_diffuse_hue_delta)
    )
    chroma_scale = (ref_chroma / base_chroma).clamp(
        float(cfg.min_diffuse_chroma_scale), float(cfg.max_diffuse_chroma_scale)
    )
    return {"luma_delta": luma_delta, "hue_delta": hue_delta, "chroma_scale": chroma_scale}


def warp_reference_to_base(img: Tensor, homography: Any, height: int, width: int) -> tuple[Tensor, Tensor]:
    if homography is None:
        valid = torch.zeros((1, 1, height, width), device=img.device, dtype=img.dtype)
        return resize_to_hw_unclamped(img, height, width), valid
    h = torch.as_tensor(homography, device=img.device, dtype=img.dtype)
    _, _, ref_h, ref_w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=img.device, dtype=img.dtype),
        torch.arange(width, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xx)
    base = torch.stack((xx.reshape(-1), yy.reshape(-1), ones.reshape(-1)), dim=0)
    ref = h @ base
    ref_x = ref[0] / ref[2].clamp_min(1e-6)
    ref_y = ref[1] / ref[2].clamp_min(1e-6)
    valid = ((ref_x >= 0.0) & (ref_x <= float(ref_w - 1)) & (ref_y >= 0.0) & (ref_y <= float(ref_h - 1))).to(img.dtype)
    grid_x = ref_x / max(ref_w - 1, 1) * 2.0 - 1.0
    grid_y = ref_y / max(ref_h - 1, 1) * 2.0 - 1.0
    grid = torch.stack((grid_x, grid_y), dim=1).view(1, height, width, 2)
    warped = F.grid_sample(img, grid, mode="bilinear", align_corners=True)
    return warped, valid.view(1, 1, height, width)


def confidence_weighted_mean(value: Tensor, confidence: Tensor, default: float) -> Tensor:
    weight = confidence.clamp(0.0, 1.0)
    total = weight.sum().clamp_min(1e-6)
    return (value * weight).sum() / total if bool((weight.sum() > 1e-6).detach().cpu()) else torch.tensor(default, device=value.device, dtype=value.dtype)


def global_fallback_maps(base_proxy: Tensor, ref_proxy: Tensor, confidence: Dict[str, Tensor], cfg: LocalMatchingConfig) -> Dict[str, Tensor]:
    deltas = proxy_delta_maps(base_proxy, ref_proxy, cfg)
    luma = confidence_weighted_mean(deltas["luma_delta"], confidence["luma_confidence"], 0.0)
    hue = confidence_weighted_mean(deltas["hue_delta"], confidence["hue_confidence"], 0.0)
    chroma = confidence_weighted_mean(deltas["chroma_scale"], confidence["chroma_confidence"], 1.0)
    return {
        "luma_delta": torch.ones_like(deltas["luma_delta"]) * luma,
        "hue_delta": torch.ones_like(deltas["hue_delta"]) * hue,
        "chroma_scale": torch.ones_like(deltas["chroma_scale"]) * chroma,
        "match_density": torch.zeros_like(deltas["luma_delta"]),
    }


def edge_aware_lowres_smooth(maps: Dict[str, Tensor], base_proxy: Tensor, cfg: LocalMatchingConfig) -> Dict[str, Tensor]:
    base_l = resize_to_hw_unclamped(base_proxy[:, 0:1], maps["luma_delta"].shape[2], maps["luma_delta"].shape[3])
    grad_x = F.pad((base_l[:, :, :, 1:] - base_l[:, :, :, :-1]).abs(), (0, 1, 0, 0), mode="replicate")
    grad_y = F.pad((base_l[:, :, 1:, :] - base_l[:, :, :-1, :]).abs(), (0, 0, 0, 1), mode="replicate")
    edge = grad_x + grad_y
    lo = percentile_threshold(edge, 75.0)
    hi = torch.maximum(percentile_threshold(edge, 95.0), lo + 1e-5)
    smooth_mix = smooth_confidence_from_range(edge, lo, hi)
    out = {}
    for key in ("luma_delta", "hue_delta", "chroma_scale"):
        blurred = maps[key]
        for _ in range(max(0, int(cfg.map_smoothing_passes))):
            blurred = F.avg_pool2d(F.pad(blurred, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        local_mix = smooth_mix * float(np.clip(cfg.map_smoothing_strength, 0.0, 1.0))
        out[key] = maps[key] * (1.0 - local_mix) + blurred * local_mix
    out["match_density"] = maps["match_density"]
    return out


def estimate_alignment(base_rgb: Tensor, reference_rgb: Tensor, cfg: LocalMatchingConfig) -> Dict[str, Any]:
    if not cfg.enable_lightglue:
        empty_points = torch.empty((0, 2), device=base_rgb.device, dtype=base_rgb.dtype)
        return {
            "base_points": empty_points,
            "ref_points": empty_points,
            "scores": torch.empty((0,), device=base_rgb.device, dtype=base_rgb.dtype),
            "raw_count": 0,
            "score_filtered_count": 0,
            "inlier_count": 0,
            "device": str(base_rgb.device),
            "homography": None,
            "valid": False,
        }
    matches = run_lightglue_matches(base_rgb, reference_rgb, cfg)
    matches["valid"] = matches.get("homography") is not None and int(matches["inlier_count"]) >= int(cfg.min_valid_matches)
    return matches


def build_aligned_reference_fields(
    base_lab: Tensor,
    reference_lab: Tensor,
    reference_rgb: Tensor,
    alignment: Dict[str, Any],
    cfg: LocalMatchingConfig,
) -> Dict[str, Any]:
    base_proxy = low_frequency_proxy_lab(base_lab, cfg)
    if not alignment["valid"]:
        ref_lab = reference_lab
        ref_rgb = reference_rgb
        valid = torch.ones_like(base_lab[:, 0:1])
        mode = "fallback_global"
    else:
        ref_lab, valid = warp_reference_to_base(reference_lab, alignment["homography"], base_lab.shape[2], base_lab.shape[3])
        ref_rgb, _ = warp_reference_to_base(reference_rgb, alignment["homography"], base_lab.shape[2], base_lab.shape[3])
        mode = "lightglue_aligned_proxy"
    ref_proxy = low_frequency_proxy_lab(ref_lab, cfg)
    confidence = confidence_maps(base_lab, base_proxy, ref_lab, ref_proxy, lab_to_rgb(base_lab), ref_rgb, cfg)
    confidence["combined_confidence"] = (confidence["combined_confidence"] * valid).clamp(0.0, 1.0)
    confidence["ref_residual_rejection"] = 1.0 - (confidence["ref_residual_confidence"] * valid).clamp(0.0, 1.0)
    for key in ALIGNMENT_KEYS:
        if key in confidence:
            confidence[key] = (confidence[key] * valid).clamp(0.0, 1.0)
    return {
        "base_proxy": base_proxy,
        "reference_lab": ref_lab,
        "reference_rgb": ref_rgb,
        "reference_proxy": ref_proxy,
        "warp_validity": valid,
        "confidence": confidence,
        "mode": mode,
    }


def compute_dense_deltas_and_apply(
    base_lab: Tensor,
    aligned_fields: Dict[str, Any],
    reference_lab_unaligned: Tensor,
    base_rgb: Tensor,
    alignment: Dict[str, Any],
    cfg: LocalMatchingConfig,
) -> tuple[Tensor, Dict[str, Tensor], Dict[str, Any], Dict[str, Tensor]]:
    base_proxy = aligned_fields["base_proxy"]
    ref_proxy = aligned_fields["reference_proxy"]
    confidence = aligned_fields["confidence"]
    valid = aligned_fields["warp_validity"]
    fallback = global_fallback_maps(base_proxy, reference_lab_unaligned if reference_lab_unaligned.shape == ref_proxy.shape else ref_proxy, confidence, cfg)
    dense = proxy_delta_maps(base_proxy, ref_proxy, cfg)
    dense["match_density"] = (valid * confidence["ref_residual_confidence"]).clamp(0.0, 1.0)
    low_maps = edge_aware_lowres_smooth(dense, base_proxy, cfg)
    dense["luma_delta"] = resize_to_hw_unclamped(low_maps["luma_delta"], base_lab.shape[2], base_lab.shape[3]).clamp(
        -float(cfg.max_diffuse_luma_delta), float(cfg.max_diffuse_luma_delta)
    )
    dense["hue_delta"] = resize_to_hw_unclamped(low_maps["hue_delta"], base_lab.shape[2], base_lab.shape[3]).clamp(
        -float(cfg.max_diffuse_hue_delta), float(cfg.max_diffuse_hue_delta)
    )
    dense["chroma_scale"] = resize_to_hw_unclamped(low_maps["chroma_scale"], base_lab.shape[2], base_lab.shape[3]).clamp(
        float(cfg.min_diffuse_chroma_scale), float(cfg.max_diffuse_chroma_scale)
    )
    dense["match_density"] = resize_to_hw_unclamped(low_maps["match_density"], base_lab.shape[2], base_lab.shape[3]).clamp(0.0, 1.0)
    blend = smoothstep(cfg.sparse_density_blend_low, cfg.sparse_density_blend_high, dense["match_density"])
    dense["luma_delta"] = dense["luma_delta"] * blend + fallback["luma_delta"] * (1.0 - blend)
    dense["hue_delta"] = dense["hue_delta"] * blend + fallback["hue_delta"] * (1.0 - blend)
    dense["chroma_scale"] = dense["chroma_scale"] * blend + fallback["chroma_scale"] * (1.0 - blend)

    final_lab, maps = apply_local_maps(
        base_lab,
        dense,
        confidence["luma_confidence"],
        confidence["hue_confidence"],
        confidence["chroma_confidence"],
        cfg,
    )
    maps["match_density"] = dense["match_density"]
    maps["filtered_match_confidence"] = confidence["combined_confidence"]
    stats = {
        "sparse_sample_count": int(alignment["base_points"].shape[0]),
        "match_density_mean": float(dense["match_density"].mean().detach().cpu()),
        "fallback_ratio": float((dense["match_density"] < float(cfg.sparse_density_blend_high)).float().mean().detach().cpu()),
    }

    sparse_debug = {
        "match_confidence": confidence["combined_confidence"],
        "luma_delta": dense["luma_delta"],
        "hue_delta": dense["hue_delta"],
        "chroma_scale": dense["chroma_scale"],
    }
    return final_lab, maps, stats, sparse_debug


def tensor_rgb_to_uint8(img: Tensor) -> np.ndarray:
    return (to_hwc_np(img) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def save_match_visual(path: str | Path, base_rgb: Tensor, ref_rgb: Tensor, matches: Dict[str, Any], max_lines: int = 160) -> None:
    base_np = tensor_rgb_to_uint8(base_rgb)
    ref_np = tensor_rgb_to_uint8(ref_rgb)
    height = max(base_np.shape[0], ref_np.shape[0])
    width = base_np.shape[1] + ref_np.shape[1]
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(Image.fromarray(base_np), (0, 0))
    canvas.paste(Image.fromarray(ref_np), (base_np.shape[1], 0))
    draw = ImageDraw.Draw(canvas)
    base_points = matches.get("base_points", torch.empty((0, 2))).detach().cpu().numpy()
    ref_points = matches.get("ref_points", torch.empty((0, 2))).detach().cpu().numpy()
    scores = matches.get("scores", torch.empty((0,))).detach().cpu().numpy()
    if base_points.shape[0] > 0:
        order = np.argsort(-scores)[:max_lines]
        for idx in order:
            x0, y0 = base_points[idx]
            x1, y1 = ref_points[idx]
            x1 = x1 + base_np.shape[1]
            score = float(scores[idx]) if scores.size else 0.5
            color = (int(255 * (1.0 - score)), int(255 * score), 80)
            draw.line((float(x0), float(y0), float(x1), float(y1)), fill=color, width=1)
            r = 2
            draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), outline=color)
            draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=color)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_sparse_delta_visual(path: str | Path, points: Tensor, values: Tensor, height: int, width: int, mode: str, cfg: LocalMatchingConfig) -> None:
    img = Image.new("L", (width, height), 128 if mode != "chroma" else 0)
    draw = ImageDraw.Draw(img)
    if points.numel() > 0:
        pts = points.detach().cpu().numpy()
        vals = values.detach().cpu().numpy()
        for (x, y), val in zip(pts, vals):
            if mode == "luma":
                gray = int(np.clip((val / max(cfg.max_diffuse_luma_delta, 1e-6)) * 0.5 + 0.5, 0.0, 1.0) * 255)
            elif mode == "hue":
                gray = int(np.clip((val / max(cfg.max_diffuse_hue_delta, 1e-6)) * 0.5 + 0.5, 0.0, 1.0) * 255)
            elif mode == "chroma":
                gray = int(np.clip((val - cfg.min_diffuse_chroma_scale) / max(cfg.max_diffuse_chroma_scale - cfg.min_diffuse_chroma_scale, 1e-6), 0.0, 1.0) * 255)
            else:
                gray = int(np.clip(val, 0.0, 1.0) * 255)
            r = 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill=gray)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def dense_sample_debug_map(points: Tensor, values: Tensor, height: int, width: int, default: float, cfg: LocalMatchingConfig) -> Tensor:
    device = points.device if points.numel() > 0 else values.device
    dtype = values.dtype
    fallback = torch.full((1, 1, height, width), float(default), device=device, dtype=dtype)
    weights = torch.ones_like(values)
    side_scale = min(1.0, float(max(16, int(cfg.sparse_map_long_edge))) / float(max(height, width)))
    low_h = max(8, int(round(height * side_scale)))
    low_w = max(8, int(round(width * side_scale)))
    low, _ = interpolate_sparse_channel(points, values, weights, low_h, low_w, fallback, cfg)
    return resize_to_hw_unclamped(low, height, width)


def sparse_debug_maps(samples: Dict[str, Tensor], matches: Dict[str, Any], height: int, width: int, cfg: LocalMatchingConfig) -> Dict[str, Tensor]:
    if matches["scores"].numel() > 0:
        match_conf = dense_sample_debug_map(matches["base_points"], matches["scores"].clamp(0.0, 1.0), height, width, 0.0, cfg).clamp(0.0, 1.0)
    else:
        match_conf = torch.zeros((1, 1, height, width), device=samples["base_points"].device, dtype=samples["base_points"].dtype)

    if samples["base_points"].numel() > 0:
        luma = dense_sample_debug_map(samples["base_points"], samples["luma_delta"], height, width, 0.0, cfg).clamp(
            -float(cfg.max_diffuse_luma_delta), float(cfg.max_diffuse_luma_delta)
        )
        hue = dense_sample_debug_map(samples["base_points"], samples["hue_delta"], height, width, 0.0, cfg).clamp(
            -float(cfg.max_diffuse_hue_delta), float(cfg.max_diffuse_hue_delta)
        )
        chroma = dense_sample_debug_map(samples["base_points"], samples["chroma_scale"], height, width, 1.0, cfg).clamp(
            float(cfg.min_diffuse_chroma_scale), float(cfg.max_diffuse_chroma_scale)
        )
    else:
        luma = torch.zeros_like(match_conf)
        hue = torch.zeros_like(match_conf)
        chroma = torch.ones_like(match_conf)

    return {"match_confidence": match_conf, "luma_delta": luma, "hue_delta": hue, "chroma_scale": chroma}


def confidence_maps(
    base_lab: Tensor,
    base_proxy: Tensor,
    ref_lab: Tensor,
    ref_proxy: Tensor,
    base_rgb: Tensor,
    ref_rgb: Tensor,
    cfg: LocalMatchingConfig,
) -> Dict[str, Tensor]:
    ref_residual = (ref_lab - ref_proxy).norm(dim=1, keepdim=True)
    residual_low = percentile_threshold(ref_residual, 65.0)
    residual_high = torch.maximum(percentile_threshold(ref_residual, 92.0), residual_low + 1e-4)
    ref_residual_confidence = smooth_confidence_from_range(ref_residual, residual_low, residual_high)
    specular_conf = reference_specular_confidence(ref_lab, ref_rgb, cfg)
    shadow_conf = shadow_confidence(ref_proxy, cfg)
    hue_stability_conf = hue_stability_confidence(base_proxy, ref_proxy, cfg)

    raw_luma_delta = (ref_proxy[:, 0:1] - base_proxy[:, 0:1]).abs()
    base_proxy_chroma, _ = chroma_and_hue(base_proxy)
    ref_proxy_chroma, ref_proxy_hue = chroma_and_hue(ref_proxy)
    _, base_proxy_hue = chroma_and_hue(base_proxy)
    raw_hue_delta = shortest_angle_delta(ref_proxy_hue, base_proxy_hue).abs()
    raw_chroma_delta = ((ref_proxy_chroma / base_proxy_chroma) - 1.0).abs()

    delta_sanity_luma = smooth_confidence_from_range(raw_luma_delta, cfg.max_diffuse_luma_delta, cfg.delta_sanity_luma)
    delta_sanity_hue = smooth_confidence_from_range(raw_hue_delta, cfg.max_diffuse_hue_delta, cfg.delta_sanity_hue)
    delta_sanity_chroma = smooth_confidence_from_range(
        raw_chroma_delta,
        max(1.0 - cfg.min_diffuse_chroma_scale, cfg.max_diffuse_chroma_scale - 1.0),
        cfg.delta_sanity_chroma_scale,
    )
    delta_sanity_conf = torch.minimum(delta_sanity_luma, torch.minimum(delta_sanity_hue, delta_sanity_chroma))

    luma_confidence = ref_residual_confidence * specular_conf * shadow_conf * delta_sanity_luma
    hue_confidence = luma_confidence * hue_stability_conf * delta_sanity_hue
    chroma_confidence = luma_confidence * hue_stability_conf * specular_conf * delta_sanity_chroma

    ref_residual_confidence = smooth_confidence_map(ref_residual_confidence, cfg)
    specular_conf = smooth_confidence_map(specular_conf, cfg)
    shadow_conf = smooth_confidence_map(shadow_conf, cfg)
    hue_stability_conf = smooth_confidence_map(hue_stability_conf, cfg)
    delta_sanity_conf = smooth_confidence_map(delta_sanity_conf, cfg)
    luma_confidence = smooth_confidence_map(luma_confidence, cfg)
    hue_confidence = smooth_confidence_map(hue_confidence, cfg)
    chroma_confidence = smooth_confidence_map(chroma_confidence, cfg)

    return {
        "ref_residual_confidence": ref_residual_confidence,
        "specular_confidence": specular_conf,
        "shadow_confidence": shadow_conf,
        "hue_stability_confidence": hue_stability_conf,
        "delta_sanity_confidence": delta_sanity_conf,
        "luma_confidence": luma_confidence,
        "hue_confidence": hue_confidence,
        "chroma_confidence": chroma_confidence,
        "combined_confidence": torch.maximum(luma_confidence, torch.maximum(hue_confidence, chroma_confidence)),
        "ref_residual_rejection": 1.0 - ref_residual_confidence,
    }


def apply_local_maps(
    base_lab: Tensor,
    delta_maps: Dict[str, Tensor],
    luma_confidence: Tensor,
    hue_confidence: Tensor,
    chroma_confidence: Tensor,
    cfg: LocalMatchingConfig,
) -> tuple[Tensor, Dict[str, Tensor]]:
    luma_delta = delta_maps["luma_delta"]
    hue_delta = delta_maps["hue_delta"]
    chroma_scale = delta_maps["chroma_scale"]

    source_chroma, source_hue = chroma_and_hue(base_lab)
    final_l = (base_lab[:, 0:1] + luma_confidence * luma_delta).clamp(0.0, 100.0)
    final_hue = source_hue + hue_confidence * hue_delta
    final_chroma = source_chroma * (1.0 + chroma_confidence * (chroma_scale - 1.0))
    final_ab = torch.cat((torch.cos(final_hue), torch.sin(final_hue)), dim=1) * final_chroma
    final_lab = torch.cat((final_l, final_ab), dim=1)

    maps = {
        "luma_delta": luma_delta,
        "hue_delta": hue_delta,
        "chroma_scale": chroma_scale,
        "confidence": torch.maximum(luma_confidence, torch.maximum(hue_confidence, chroma_confidence)),
        "luma_confidence": luma_confidence,
        "hue_confidence": hue_confidence,
        "chroma_confidence": chroma_confidence,
    }
    return final_lab, maps


def run_local_diffuse_matching(
    *,
    base_intermediate_lab: Tensor,
    base_intermediate_rgb: Tensor,
    reference_resized_lab: Tensor,
    reference_resized_rgb: Tensor,
    source_lab: Tensor,
    source_rgb: Tensor,
    output_dir: str | Path,
    global_metrics: Dict[str, Any],
    config: Optional[LocalMatchingConfig | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config if isinstance(config, LocalMatchingConfig) else LocalMatchingConfig(**(config or {}))
    output_dir = Path(output_dir)
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    base_proxy = low_frequency_proxy_lab(base_intermediate_lab, cfg)
    ref_proxy = low_frequency_proxy_lab(reference_resized_lab, cfg)
    timings["diffuse_proxy"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    global_confidence = confidence_maps(
        base_intermediate_lab, base_proxy, reference_resized_lab, ref_proxy, base_intermediate_rgb, reference_resized_rgb, cfg
    )
    timings["confidence_maps"] = time.perf_counter() - t1

    t_match = time.perf_counter()
    matches = estimate_alignment(base_intermediate_rgb, reference_resized_rgb, cfg)
    timings["lightglue_match"] = time.perf_counter() - t_match

    t_align = time.perf_counter()
    aligned_fields = build_aligned_reference_fields(
        base_intermediate_lab,
        reference_resized_lab,
        reference_resized_rgb,
        matches,
        cfg,
    )
    timings["aligned_reference_fields"] = time.perf_counter() - t_align

    t_apply = time.perf_counter()
    final_lab, maps, sparse_stats, sparse_debug = compute_dense_deltas_and_apply(
        base_intermediate_lab,
        aligned_fields,
        ref_proxy,
        base_intermediate_rgb,
        matches,
        cfg,
    )
    applied_confidence = aligned_fields["confidence"]
    local_mode = aligned_fields["mode"]
    final_rgb = soft_gamut_compress(lab_to_rgb(final_lab))
    timings["local_apply"] = time.perf_counter() - t_apply

    paths = dict(global_metrics["paths"])
    paths.update(
        {
            "diffuse_luma_delta": str(output_dir / "diffuse_luma_delta.png"),
            "diffuse_hue_delta": str(output_dir / "diffuse_hue_delta.png"),
            "diffuse_chroma_scale": str(output_dir / "diffuse_chroma_scale.png"),
            "diffuse_confidence": str(output_dir / "diffuse_confidence.png"),
            "diffuse_residual_rejection": str(output_dir / "diffuse_residual_rejection.png"),
            "ref_residual_confidence": str(output_dir / "ref_residual_confidence.png"),
            "specular_confidence": str(output_dir / "specular_confidence.png"),
            "shadow_confidence": str(output_dir / "shadow_confidence.png"),
            "hue_stability_confidence": str(output_dir / "hue_stability_confidence.png"),
            "delta_sanity_confidence": str(output_dir / "delta_sanity_confidence.png"),
            "luma_confidence": str(output_dir / "luma_confidence.png"),
            "hue_confidence": str(output_dir / "hue_confidence.png"),
            "chroma_confidence": str(output_dir / "chroma_confidence.png"),
            "lightglue_matches": str(output_dir / "lightglue_matches.png"),
            "filtered_match_confidence": str(output_dir / "filtered_match_confidence.png"),
            "sparse_luma_delta": str(output_dir / "sparse_luma_delta.png"),
            "sparse_hue_delta": str(output_dir / "sparse_hue_delta.png"),
            "sparse_chroma_scale": str(output_dir / "sparse_chroma_scale.png"),
            "match_density": str(output_dir / "match_density.png"),
            "final_output": str(output_dir / "final_output.png"),
        }
    )

    t2 = time.perf_counter()
    save_gray(paths["diffuse_luma_delta"], signed_map_visual(maps["luma_delta"], cfg.max_diffuse_luma_delta))
    save_gray(paths["diffuse_hue_delta"], signed_map_visual(maps["hue_delta"], cfg.max_diffuse_hue_delta))
    save_gray(
        paths["diffuse_chroma_scale"],
        range_map_visual(maps["chroma_scale"], cfg.min_diffuse_chroma_scale, cfg.max_diffuse_chroma_scale),
    )
    save_gray(paths["diffuse_confidence"], gray_visual(maps["confidence"]))
    save_gray(paths["diffuse_residual_rejection"], gray_visual(applied_confidence["ref_residual_rejection"]))
    save_gray(paths["ref_residual_confidence"], gray_visual(applied_confidence["ref_residual_confidence"]))
    save_gray(paths["specular_confidence"], gray_visual(applied_confidence["specular_confidence"]))
    save_gray(paths["shadow_confidence"], gray_visual(applied_confidence["shadow_confidence"]))
    save_gray(paths["hue_stability_confidence"], gray_visual(applied_confidence["hue_stability_confidence"]))
    save_gray(paths["delta_sanity_confidence"], gray_visual(applied_confidence["delta_sanity_confidence"]))
    save_gray(paths["luma_confidence"], gray_visual(applied_confidence["luma_confidence"]))
    save_gray(paths["hue_confidence"], gray_visual(applied_confidence["hue_confidence"]))
    save_gray(paths["chroma_confidence"], gray_visual(applied_confidence["chroma_confidence"]))
    save_match_visual(paths["lightglue_matches"], base_intermediate_rgb, reference_resized_rgb, matches)
    save_gray(paths["filtered_match_confidence"], gray_visual(sparse_debug["match_confidence"]))
    save_gray(paths["sparse_luma_delta"], signed_map_visual(sparse_debug["luma_delta"], cfg.max_diffuse_luma_delta))
    save_gray(paths["sparse_hue_delta"], signed_map_visual(sparse_debug["hue_delta"], cfg.max_diffuse_hue_delta))
    save_gray(paths["sparse_chroma_scale"], range_map_visual(sparse_debug["chroma_scale"], cfg.min_diffuse_chroma_scale, cfg.max_diffuse_chroma_scale))
    save_gray(paths["match_density"], gray_visual(maps["match_density"]))
    save_rgb(paths["final_output"], to_hwc_np(final_rgb))
    timings["local_debug_saves"] = time.perf_counter() - t2

    serial_global = {key: value for key, value in global_metrics.items() if key != "tensors"}
    metrics: Dict[str, Any] = {
        **serial_global,
        "pipeline_version": "v0.3.6-linear-lut-lightglue-local-diffuse",
        "global_config": serial_global.get("config", {}),
        "local_config": asdict(cfg),
        "global_timings": serial_global.get("timings", {}),
        "local_timings": timings,
        "mean_local_confidence": float(maps["confidence"].mean().detach().cpu()),
        "mean_ref_residual_confidence": float(applied_confidence["ref_residual_confidence"].mean().detach().cpu()),
        "mean_specular_confidence": float(applied_confidence["specular_confidence"].mean().detach().cpu()),
        "mean_shadow_confidence": float(applied_confidence["shadow_confidence"].mean().detach().cpu()),
        "mean_hue_stability_confidence": float(applied_confidence["hue_stability_confidence"].mean().detach().cpu()),
        "mean_delta_sanity_confidence": float(applied_confidence["delta_sanity_confidence"].mean().detach().cpu()),
        "mean_local_luma_confidence": float(applied_confidence["luma_confidence"].mean().detach().cpu()),
        "mean_local_hue_confidence": float(applied_confidence["hue_confidence"].mean().detach().cpu()),
        "mean_local_chroma_confidence": float(applied_confidence["chroma_confidence"].mean().detach().cpu()),
        "mean_local_residual_rejection": float(applied_confidence["ref_residual_rejection"].mean().detach().cpu()),
        "lightglue_device": matches["device"],
        "lightglue_extractor": "aliked",
        "lightglue_mode": local_mode,
        "raw_match_count": int(matches["raw_count"]),
        "confidence_filtered_match_count": int(matches["score_filtered_count"]),
        "ransac_inlier_count": int(matches["inlier_count"]),
        "sparse_sample_count": int(sparse_stats["sparse_sample_count"]),
        "match_density_mean": float(sparse_stats["match_density_mean"]),
        "fallback_ratio": float(sparse_stats["fallback_ratio"]),
        "final_output_stats": image_stats_from_lab(source_lab, final_lab, final_rgb),
        "paths": paths,
    }
    metrics.pop("config", None)
    metrics.pop("timings", None)
    Path(paths["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["tensors"] = {"final_output_rgb": final_rgb, "final_output_lab": final_lab}
    return metrics
