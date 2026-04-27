"""LookAlign V0.3.5 local diffuse mood matching.

This stage is intentionally deterministic and lightweight. It approximates a
colorful diffuse-shading transfer by operating on smooth OKLab maps while
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

from scripts.ot_svd_lut_mps import (
    Tensor,
    image_stats_from_lab,
    oklab_to_rgb,
    save_gray,
    save_rgb,
    soft_gamut_compress,
    to_hwc_np,
)


@dataclass
class LocalMatchingConfig:
    enable_lightglue: bool = True
    match_long_edge: int = 512
    max_keypoints: int = 1024
    min_match_score: float = 0.20
    ransac_reproj_threshold: float = 8.0
    min_valid_matches: int = 24
    sparse_map_long_edge: int = 192
    fallback_global_blend: float = 0.20
    sparse_sigma: float = 0.075
    diffuse_proxy_long_edge: int = 320
    diffuse_blur_passes: int = 4
    max_diffuse_luma_delta: float = 0.10
    max_diffuse_hue_delta: float = 0.35
    min_diffuse_chroma_scale: float = 0.80
    max_diffuse_chroma_scale: float = 1.25
    low_chroma_start: float = 0.006
    low_chroma_end: float = 0.025
    clipped_start: float = 0.965
    clipped_end: float = 0.995
    shadow_start: float = 0.020
    shadow_end: float = 0.120
    delta_sanity_luma: float = 0.180
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


def low_frequency_proxy_oklab(lab: Tensor, cfg: LocalMatchingConfig) -> Tensor:
    _, _, height, width = lab.shape
    proxy = resize_long_edge_unclamped(lab, cfg.diffuse_proxy_long_edge)
    for _ in range(max(1, int(cfg.diffuse_blur_passes))):
        proxy = F.avg_pool2d(F.pad(proxy, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
    return resize_to_hw_unclamped(proxy, height, width)


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
    low_chroma = 1.0 - smoothstep(0.015, 0.050, ref_chroma)
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
            "LightGlue is required for V0.3.5 correspondence-guided local matching. "
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
        _, mask = cv2.findHomography(p0_np, p1_np, cv2.RANSAC, float(cfg.ransac_reproj_threshold))
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


def sample_delta_candidates(
    base_proxy: Tensor,
    ref_proxy: Tensor,
    confidence: Dict[str, Tensor],
    matches: Dict[str, Any],
    cfg: LocalMatchingConfig,
) -> Dict[str, Tensor]:
    base_points = matches["base_points"]
    ref_points = matches["ref_points"]
    if base_points.numel() == 0:
        empty = torch.empty((0,), device=base_proxy.device, dtype=base_proxy.dtype)
        return {"base_points": base_points, "luma_delta": empty, "hue_delta": empty, "chroma_scale": empty, "luma_weight": empty, "hue_weight": empty, "chroma_weight": empty}

    base_samples = sample_points(base_proxy, base_points)
    ref_samples = sample_points(ref_proxy, ref_points)
    base_chroma = base_samples[:, 1:3].norm(dim=1).clamp_min(1e-6)
    ref_chroma = ref_samples[:, 1:3].norm(dim=1).clamp_min(1e-6)
    base_hue = torch.atan2(base_samples[:, 2], base_samples[:, 1])
    ref_hue = torch.atan2(ref_samples[:, 2], ref_samples[:, 1])

    raw_luma = ref_samples[:, 0] - base_samples[:, 0]
    raw_hue = torch.atan2(torch.sin(ref_hue - base_hue), torch.cos(ref_hue - base_hue))
    raw_chroma = ref_chroma / base_chroma

    luma_delta = raw_luma.clamp(-float(cfg.max_diffuse_luma_delta), float(cfg.max_diffuse_luma_delta))
    hue_delta = raw_hue.clamp(-float(cfg.max_diffuse_hue_delta), float(cfg.max_diffuse_hue_delta))
    chroma_scale = raw_chroma.clamp(float(cfg.min_diffuse_chroma_scale), float(cfg.max_diffuse_chroma_scale))

    ref_residual = sample_points(confidence["ref_residual_confidence"], ref_points)[:, 0]
    specular = sample_points(confidence["specular_confidence"], ref_points)[:, 0]
    shadow = sample_points(confidence["shadow_confidence"], ref_points)[:, 0]
    hue_stability = smoothstep(cfg.low_chroma_start, cfg.low_chroma_end, torch.minimum(base_chroma, ref_chroma))
    delta_luma = smooth_confidence_from_range(raw_luma.abs(), cfg.max_diffuse_luma_delta, cfg.delta_sanity_luma)
    delta_hue = smooth_confidence_from_range(raw_hue.abs(), cfg.max_diffuse_hue_delta, cfg.delta_sanity_hue)
    delta_chroma = smooth_confidence_from_range(
        (raw_chroma - 1.0).abs(),
        max(1.0 - cfg.min_diffuse_chroma_scale, cfg.max_diffuse_chroma_scale - 1.0),
        cfg.delta_sanity_chroma_scale,
    )
    match_scores = matches["scores"].clamp(0.0, 1.0)

    luma_weight = match_scores * ref_residual * specular * shadow * delta_luma
    hue_weight = luma_weight * hue_stability * delta_hue
    chroma_weight = luma_weight * hue_stability * specular * delta_chroma
    keep = (luma_weight > 0.03) | (hue_weight > 0.03) | (chroma_weight > 0.03)
    return {
        "base_points": base_points[keep],
        "luma_delta": luma_delta[keep],
        "hue_delta": hue_delta[keep],
        "chroma_scale": chroma_scale[keep],
        "luma_weight": luma_weight[keep],
        "hue_weight": hue_weight[keep],
        "chroma_weight": chroma_weight[keep],
    }


def interpolate_sparse_channel(
    points_xy: Tensor,
    values: Tensor,
    weights: Tensor,
    height: int,
    width: int,
    fallback: Tensor,
    cfg: LocalMatchingConfig,
) -> tuple[Tensor, Tensor]:
    if points_xy.shape[0] == 0:
        return fallback.expand(1, 1, height, width).clone(), torch.zeros((1, 1, height, width), device=fallback.device, dtype=fallback.dtype)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=points_xy.device, dtype=points_xy.dtype),
        torch.linspace(0.0, 1.0, width, device=points_xy.device, dtype=points_xy.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    _, _, full_h, full_w = fallback.shape
    norm_points = torch.stack(
        (
            points_xy[:, 0] / max(full_w - 1, 1),
            points_xy[:, 1] / max(full_h - 1, 1),
        ),
        dim=1,
    )
    out_num = torch.zeros((grid.shape[0],), device=points_xy.device, dtype=points_xy.dtype)
    out_den = torch.zeros_like(out_num)
    sigma2 = max(float(cfg.sparse_sigma), 1e-4) ** 2
    for start in range(0, int(grid.shape[0]), 4096):
        chunk = grid[start : start + 4096]
        dist2 = ((chunk[:, None, :] - norm_points[None, :, :]) ** 2).sum(dim=2)
        w = torch.exp(-dist2 / (2.0 * sigma2)) * weights.unsqueeze(0)
        den = w.sum(dim=1)
        out_num[start : start + chunk.shape[0]] = (w * values.unsqueeze(0)).sum(dim=1)
        out_den[start : start + chunk.shape[0]] = den
    dense = out_num / out_den.clamp_min(1e-6)
    density = (out_den / out_den.max().clamp_min(1e-6)).clamp(0.0, 1.0)
    fallback_small = resize_to_hw_unclamped(fallback, height, width)[0, 0].reshape(-1)
    blend = smoothstep(0.02, 0.18, density)
    dense = dense * blend + fallback_small * (1.0 - blend)
    return dense.view(1, 1, height, width), density.view(1, 1, height, width)


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
        for _ in range(2):
            blurred = F.avg_pool2d(F.pad(blurred, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
        out[key] = maps[key] * (1.0 - smooth_mix) + blurred * smooth_mix
    out["match_density"] = maps["match_density"]
    return out


def lightglue_delta_maps(
    base_proxy: Tensor,
    ref_proxy: Tensor,
    confidence: Dict[str, Tensor],
    matches: Dict[str, Any],
    cfg: LocalMatchingConfig,
) -> tuple[Dict[str, Tensor], Dict[str, Any], Dict[str, Tensor]]:
    _, _, height, width = base_proxy.shape
    fallback = global_fallback_maps(base_proxy, ref_proxy, confidence, cfg)
    samples = sample_delta_candidates(base_proxy, ref_proxy, confidence, matches, cfg)
    side_scale = min(1.0, float(max(16, int(cfg.sparse_map_long_edge))) / float(max(height, width)))
    low_h = max(8, int(round(height * side_scale)))
    low_w = max(8, int(round(width * side_scale)))

    fallback_luma = fallback["luma_delta"]
    fallback_hue = fallback["hue_delta"]
    fallback_chroma = fallback["chroma_scale"]
    luma_low, density_l = interpolate_sparse_channel(samples["base_points"], samples["luma_delta"], samples["luma_weight"], low_h, low_w, fallback_luma, cfg)
    hue_low, density_h = interpolate_sparse_channel(samples["base_points"], samples["hue_delta"], samples["hue_weight"], low_h, low_w, fallback_hue, cfg)
    chroma_low, density_c = interpolate_sparse_channel(samples["base_points"], samples["chroma_scale"], samples["chroma_weight"], low_h, low_w, fallback_chroma, cfg)
    density = torch.maximum(density_l, torch.maximum(density_h, density_c))
    low_maps = edge_aware_lowres_smooth(
        {"luma_delta": luma_low, "hue_delta": hue_low, "chroma_scale": chroma_low, "match_density": density},
        base_proxy,
        cfg,
    )
    dense = {
        "luma_delta": resize_to_hw_unclamped(low_maps["luma_delta"], height, width).clamp(-float(cfg.max_diffuse_luma_delta), float(cfg.max_diffuse_luma_delta)),
        "hue_delta": resize_to_hw_unclamped(low_maps["hue_delta"], height, width).clamp(-float(cfg.max_diffuse_hue_delta), float(cfg.max_diffuse_hue_delta)),
        "chroma_scale": resize_to_hw_unclamped(low_maps["chroma_scale"], height, width).clamp(float(cfg.min_diffuse_chroma_scale), float(cfg.max_diffuse_chroma_scale)),
        "match_density": resize_to_hw_unclamped(low_maps["match_density"], height, width).clamp(0.0, 1.0),
    }
    fallback_ratio = float((dense["match_density"] < 0.18).float().mean().detach().cpu())
    stats = {
        "sparse_sample_count": int(samples["base_points"].shape[0]),
        "match_density_mean": float(dense["match_density"].mean().detach().cpu()),
        "fallback_ratio": fallback_ratio,
    }
    return dense, stats, samples


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
    final_l = (base_lab[:, 0:1] + luma_confidence * luma_delta).clamp(0.0, 1.0)
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
    base_proxy = low_frequency_proxy_oklab(base_intermediate_lab, cfg)
    ref_proxy = low_frequency_proxy_oklab(reference_resized_lab, cfg)
    timings["diffuse_proxy"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    confidence = confidence_maps(
        base_intermediate_lab,
        base_proxy,
        reference_resized_lab,
        ref_proxy,
        base_intermediate_rgb,
        reference_resized_rgb,
        cfg,
    )
    timings["confidence_maps"] = time.perf_counter() - t1

    t_match = time.perf_counter()
    match_stats: Dict[str, Any]
    if cfg.enable_lightglue:
        matches = run_lightglue_matches(base_intermediate_rgb, reference_resized_rgb, cfg)
    else:
        empty_points = torch.empty((0, 2), device=base_intermediate_lab.device, dtype=base_intermediate_lab.dtype)
        matches = {
            "base_points": empty_points,
            "ref_points": empty_points,
            "scores": torch.empty((0,), device=base_intermediate_lab.device, dtype=base_intermediate_lab.dtype),
            "raw_count": 0,
            "score_filtered_count": 0,
            "inlier_count": 0,
            "device": str(base_intermediate_lab.device),
        }
    timings["lightglue_match"] = time.perf_counter() - t_match

    t_interp = time.perf_counter()
    if int(matches["inlier_count"]) >= int(cfg.min_valid_matches):
        delta_maps, sparse_stats, sparse_samples = lightglue_delta_maps(base_proxy, ref_proxy, confidence, matches, cfg)
        local_mode = "lightglue"
    else:
        delta_maps = global_fallback_maps(base_proxy, ref_proxy, confidence, cfg)
        sparse_samples = {
            "base_points": torch.empty((0, 2), device=base_intermediate_lab.device, dtype=base_intermediate_lab.dtype),
            "luma_delta": torch.empty((0,), device=base_intermediate_lab.device, dtype=base_intermediate_lab.dtype),
            "hue_delta": torch.empty((0,), device=base_intermediate_lab.device, dtype=base_intermediate_lab.dtype),
            "chroma_scale": torch.empty((0,), device=base_intermediate_lab.device, dtype=base_intermediate_lab.dtype),
        }
        sparse_stats = {
            "sparse_sample_count": 0,
            "match_density_mean": 0.0,
            "fallback_ratio": 1.0,
        }
        local_mode = "fallback_global"
    timings["sparse_interpolation"] = time.perf_counter() - t_interp

    t_apply = time.perf_counter()
    final_lab, maps = apply_local_maps(
        base_intermediate_lab,
        delta_maps,
        confidence["luma_confidence"],
        confidence["hue_confidence"],
        confidence["chroma_confidence"],
        cfg,
    )
    maps["match_density"] = delta_maps["match_density"]
    final_rgb = soft_gamut_compress(oklab_to_rgb(final_lab))
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
    save_gray(paths["diffuse_residual_rejection"], gray_visual(confidence["ref_residual_rejection"]))
    save_gray(paths["ref_residual_confidence"], gray_visual(confidence["ref_residual_confidence"]))
    save_gray(paths["specular_confidence"], gray_visual(confidence["specular_confidence"]))
    save_gray(paths["shadow_confidence"], gray_visual(confidence["shadow_confidence"]))
    save_gray(paths["hue_stability_confidence"], gray_visual(confidence["hue_stability_confidence"]))
    save_gray(paths["delta_sanity_confidence"], gray_visual(confidence["delta_sanity_confidence"]))
    save_gray(paths["luma_confidence"], gray_visual(confidence["luma_confidence"]))
    save_gray(paths["hue_confidence"], gray_visual(confidence["hue_confidence"]))
    save_gray(paths["chroma_confidence"], gray_visual(confidence["chroma_confidence"]))
    save_match_visual(paths["lightglue_matches"], base_intermediate_rgb, reference_resized_rgb, matches)
    save_sparse_delta_visual(paths["filtered_match_confidence"], matches["base_points"], matches["scores"], base_intermediate_rgb.shape[2], base_intermediate_rgb.shape[3], "confidence", cfg)
    save_sparse_delta_visual(paths["sparse_luma_delta"], sparse_samples["base_points"], sparse_samples["luma_delta"], base_intermediate_rgb.shape[2], base_intermediate_rgb.shape[3], "luma", cfg)
    save_sparse_delta_visual(paths["sparse_hue_delta"], sparse_samples["base_points"], sparse_samples["hue_delta"], base_intermediate_rgb.shape[2], base_intermediate_rgb.shape[3], "hue", cfg)
    save_sparse_delta_visual(paths["sparse_chroma_scale"], sparse_samples["base_points"], sparse_samples["chroma_scale"], base_intermediate_rgb.shape[2], base_intermediate_rgb.shape[3], "chroma", cfg)
    save_gray(paths["match_density"], gray_visual(maps["match_density"]))
    save_rgb(paths["final_output"], to_hwc_np(final_rgb))
    timings["local_debug_saves"] = time.perf_counter() - t2

    serial_global = {key: value for key, value in global_metrics.items() if key != "tensors"}
    metrics: Dict[str, Any] = {
        **serial_global,
        "pipeline_version": "v0.3.5-mps-global-ot-local-diffuse",
        "global_config": serial_global.get("config", {}),
        "local_config": asdict(cfg),
        "global_timings": serial_global.get("timings", {}),
        "local_timings": timings,
        "mean_local_confidence": float(maps["confidence"].mean().detach().cpu()),
        "mean_ref_residual_confidence": float(confidence["ref_residual_confidence"].mean().detach().cpu()),
        "mean_specular_confidence": float(confidence["specular_confidence"].mean().detach().cpu()),
        "mean_shadow_confidence": float(confidence["shadow_confidence"].mean().detach().cpu()),
        "mean_hue_stability_confidence": float(confidence["hue_stability_confidence"].mean().detach().cpu()),
        "mean_delta_sanity_confidence": float(confidence["delta_sanity_confidence"].mean().detach().cpu()),
        "mean_local_luma_confidence": float(confidence["luma_confidence"].mean().detach().cpu()),
        "mean_local_hue_confidence": float(confidence["hue_confidence"].mean().detach().cpu()),
        "mean_local_chroma_confidence": float(confidence["chroma_confidence"].mean().detach().cpu()),
        "mean_local_residual_rejection": float(confidence["ref_residual_rejection"].mean().detach().cpu()),
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
