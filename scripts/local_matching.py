"""LookAlign V0.3.5 local diffuse mood matching.

This stage is intentionally deterministic and lightweight. It approximates a
colorful diffuse-shading transfer by operating on smooth OKLab maps while
leaving base-intermediate detail in place.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

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
    diffuse_proxy_long_edge: int = 320
    diffuse_blur_passes: int = 4
    max_diffuse_luma_delta: float = 0.10
    max_diffuse_hue_delta: float = 0.35
    min_diffuse_chroma_scale: float = 0.80
    max_diffuse_chroma_scale: float = 1.25
    residual_low: float = 0.035
    residual_high: float = 0.115
    texture_low: float = 0.018
    texture_high: float = 0.070
    low_chroma_start: float = 0.018
    low_chroma_end: float = 0.055
    clipped_start: float = 0.965
    clipped_end: float = 0.995


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


def local_texture_energy(lab: Tensor) -> Tensor:
    local_mean = F.avg_pool2d(lab, kernel_size=3, stride=1, padding=1)
    return (lab - local_mean).norm(dim=1, keepdim=True)


def clipped_highlight_rejection(rgb: Tensor, cfg: LocalMatchingConfig) -> Tensor:
    max_rgb = rgb.max(dim=1, keepdim=True).values
    sat_channels = (rgb > float(cfg.clipped_start)).float().mean(dim=1, keepdim=True)
    bright_reject = smoothstep(cfg.clipped_start, cfg.clipped_end, max_rgb)
    return (bright_reject * sat_channels).clamp(0.0, 1.0)


def residual_rejection(
    base_lab: Tensor,
    base_proxy: Tensor,
    ref_lab: Tensor,
    ref_proxy: Tensor,
    base_rgb: Tensor,
    ref_rgb: Tensor,
    cfg: LocalMatchingConfig,
) -> Tensor:
    base_residual = (base_lab - base_proxy).norm(dim=1, keepdim=True)
    ref_residual = (ref_lab - ref_proxy).norm(dim=1, keepdim=True)
    residual = torch.maximum(base_residual, ref_residual)
    residual_reject = smoothstep(cfg.residual_low, cfg.residual_high, residual)

    texture = torch.maximum(local_texture_energy(base_lab), local_texture_energy(ref_lab))
    texture_reject = smoothstep(cfg.texture_low, cfg.texture_high, texture)

    clipped = torch.maximum(clipped_highlight_rejection(base_rgb, cfg), clipped_highlight_rejection(ref_rgb, cfg))

    base_proxy_chroma, _ = chroma_and_hue(base_proxy)
    ref_proxy_chroma, _ = chroma_and_hue(ref_proxy)
    min_chroma = torch.minimum(base_proxy_chroma, ref_proxy_chroma)
    unstable_hue = 1.0 - smoothstep(cfg.low_chroma_start, cfg.low_chroma_end, min_chroma)

    return torch.maximum(torch.maximum(residual_reject, texture_reject), torch.maximum(clipped, unstable_hue)).clamp(0.0, 1.0)


def apply_local_maps(
    base_lab: Tensor,
    base_proxy: Tensor,
    ref_proxy: Tensor,
    confidence: Tensor,
    cfg: LocalMatchingConfig,
) -> tuple[Tensor, Dict[str, Tensor]]:
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

    source_chroma, source_hue = chroma_and_hue(base_lab)
    final_l = (base_lab[:, 0:1] + confidence * luma_delta).clamp(0.0, 1.0)
    final_hue = source_hue + confidence * hue_delta
    final_chroma = source_chroma * (1.0 + confidence * (chroma_scale - 1.0))
    final_ab = torch.cat((torch.cos(final_hue), torch.sin(final_hue)), dim=1) * final_chroma
    final_lab = torch.cat((final_l, final_ab), dim=1)

    maps = {
        "luma_delta": luma_delta,
        "hue_delta": hue_delta,
        "chroma_scale": chroma_scale,
        "confidence": confidence,
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
    rejection = residual_rejection(
        base_intermediate_lab,
        base_proxy,
        reference_resized_lab,
        ref_proxy,
        base_intermediate_rgb,
        reference_resized_rgb,
        cfg,
    )
    confidence = 1.0 - rejection
    final_lab, maps = apply_local_maps(base_intermediate_lab, base_proxy, ref_proxy, confidence, cfg)
    final_rgb = soft_gamut_compress(oklab_to_rgb(final_lab))
    timings["local_apply"] = time.perf_counter() - t1

    paths = dict(global_metrics["paths"])
    paths.update(
        {
            "diffuse_luma_delta": str(output_dir / "diffuse_luma_delta.png"),
            "diffuse_hue_delta": str(output_dir / "diffuse_hue_delta.png"),
            "diffuse_chroma_scale": str(output_dir / "diffuse_chroma_scale.png"),
            "diffuse_confidence": str(output_dir / "diffuse_confidence.png"),
            "diffuse_residual_rejection": str(output_dir / "diffuse_residual_rejection.png"),
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
    save_gray(paths["diffuse_residual_rejection"], gray_visual(rejection))
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
        "mean_local_confidence": float(confidence.mean().detach().cpu()),
        "mean_local_residual_rejection": float(rejection.mean().detach().cpu()),
        "final_output_stats": image_stats_from_lab(source_lab, final_lab, final_rgb),
        "paths": paths,
    }
    metrics.pop("config", None)
    metrics.pop("timings", None)
    Path(paths["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["tensors"] = {"final_output_rgb": final_rgb, "final_output_lab": final_lab}
    return metrics
