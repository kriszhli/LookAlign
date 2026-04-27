"""LookAlign V0.3.6 global matching: simple linear-RGB 3D LUT fitting."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import colour
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


Tensor = torch.Tensor
GLOBAL_STAGE_VERSION = "v0.3.6-global-linear-lut"
PIPELINE_VERSION = "v0.3.6-linear-lut-lightglue-local-diffuse"


@dataclass
class GlobalMatchingConfig:
    fit_long_edge: int = 768
    lut_size: int = 33
    weak_cell_mix: float = 0.65
    neutral_protection: float = 0.15
    max_rgb_delta: float = 0.45
    apply_chunk_pixels: int = 262144


def require_mps() -> torch.device:
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise RuntimeError(
            "LookAlign V0.3.6 requires PyTorch MPS in this interpreter. "
            "Run `python3 -c \"import torch; print(torch.backends.mps.is_available())\"` "
            "with the same Python used by the app."
        )
    return torch.device("mps")


def require_colour() -> None:
    if not hasattr(colour, "LUT3D"):
        raise RuntimeError("colour-science is required for LookAlign V0.3.6 global matching.")


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def save_rgb(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def save_gray(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def to_nchw_mps(img: np.ndarray, device: torch.device) -> Tensor:
    arr = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def to_hwc_np(img: Tensor) -> np.ndarray:
    return img.detach().clamp(0.0, 1.0).cpu().numpy()[0].transpose(1, 2, 0).astype(np.float32)


def resize_to_hw(img: Tensor, height: int, width: int) -> Tensor:
    return F.interpolate(img, size=(int(height), int(width)), mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def resize_long_edge(img: Tensor, long_edge: int) -> Tensor:
    _, _, height, width = img.shape
    scale = min(1.0, float(max(16, int(long_edge))) / float(max(height, width)))
    if scale >= 0.999:
        return img.clone()
    return resize_to_hw(img, max(8, int(round(height * scale))), max(8, int(round(width * scale))))


def srgb_to_linear(rgb: Tensor) -> Tensor:
    return torch.where(rgb <= 0.04045, rgb / 12.92, torch.pow((rgb + 0.055) / 1.055, 2.4))


def linear_to_srgb(rgb: Tensor) -> Tensor:
    rgb = rgb.clamp_min(0.0)
    return torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * torch.pow(rgb, 1.0 / 2.4) - 0.055)


def rgb_to_oklab(rgb: Tensor) -> Tensor:
    linear = srgb_to_linear(rgb)
    flat = linear.permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()
    xyz = colour.RGB_to_XYZ(
        flat,
        colour.models.RGB_COLOURSPACE_sRGB,
        apply_cctf_decoding=False,
    )
    lab = colour.XYZ_to_Oklab(xyz).astype(np.float32)
    return torch.from_numpy(lab).to(device=rgb.device, dtype=rgb.dtype).reshape(rgb.shape[0], rgb.shape[2], rgb.shape[3], 3).permute(0, 3, 1, 2)


def oklab_to_rgb(lab: Tensor) -> Tensor:
    flat = lab.permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()
    xyz = colour.Oklab_to_XYZ(flat)
    linear = colour.XYZ_to_RGB(
        xyz,
        colour.models.RGB_COLOURSPACE_sRGB,
        apply_cctf_encoding=False,
    ).astype(np.float32)
    linear_t = torch.from_numpy(linear).to(device=lab.device, dtype=lab.dtype).reshape(lab.shape[0], lab.shape[2], lab.shape[3], 3).permute(0, 3, 1, 2)
    return linear_to_srgb(linear_t)


def soft_gamut_compress(rgb: Tensor) -> Tensor:
    return rgb.clamp(0.0, 1.0)


def flatten_rgb(img: Tensor) -> Tensor:
    return img.permute(0, 2, 3, 1).reshape(-1, 3)


def identity_lut_linear(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    axis = torch.linspace(0.0, 1.0, int(size), device=device, dtype=dtype)
    rr, gg, bb = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((rr, gg, bb), dim=0)


def fit_linear_rgb_lut(source_linear: Tensor, reference_linear: Tensor, cfg: GlobalMatchingConfig) -> Tuple[Tensor, Tensor]:
    size = int(cfg.lut_size)
    src = flatten_rgb(source_linear)
    ref = flatten_rgb(reference_linear)
    pos = src.clamp(0.0, 1.0) * float(size - 1)
    base = torch.floor(pos).long().clamp(0, size - 1)
    frac = pos - base.to(src.dtype)
    upper = (base + 1).clamp(0, size - 1)

    value_sum = torch.zeros((3, size, size, size), device=src.device, dtype=src.dtype)
    weight_sum = torch.zeros((size, size, size), device=src.device, dtype=src.dtype)

    for ri, rw in ((base[:, 0], 1.0 - frac[:, 0]), (upper[:, 0], frac[:, 0])):
        for gi, gw in ((base[:, 1], 1.0 - frac[:, 1]), (upper[:, 1], frac[:, 1])):
            for bi, bw in ((base[:, 2], 1.0 - frac[:, 2]), (upper[:, 2], frac[:, 2])):
                weight = (rw * gw * bw).clamp_min(0.0)
                flat_idx = (ri * size + gi) * size + bi
                weight_sum.reshape(-1).index_add_(0, flat_idx, weight)
                for channel in range(3):
                    value_sum[channel].reshape(-1).index_add_(0, flat_idx, ref[:, channel] * weight)

    support = weight_sum
    mean_lut = value_sum / support.clamp_min(1e-6).unsqueeze(0)
    filled = smooth_lut_values(mean_lut, support, cfg)
    protected = protect_neutral_lut(identity_lut_linear(size, src.device, src.dtype), filled, cfg)
    _ = colour.LUT3D(protected.permute(1, 2, 3, 0).detach().cpu().numpy(), size=size, name="LookAlign V0.3.6")
    return protected, support


def smooth_lut_values(values: Tensor, support: Tensor, cfg: GlobalMatchingConfig) -> Tensor:
    x = values
    for _ in range(3):
        x = (
            x
            + torch.roll(x, shifts=1, dims=1)
            + torch.roll(x, shifts=-1, dims=1)
            + torch.roll(x, shifts=1, dims=2)
            + torch.roll(x, shifts=-1, dims=2)
            + torch.roll(x, shifts=1, dims=3)
            + torch.roll(x, shifts=-1, dims=3)
        ) / 7.0
    support_norm = support / support.max().clamp_min(1e-6)
    mix = support_norm.unsqueeze(0).clamp(0.0, 1.0)
    weak = float(np.clip(cfg.weak_cell_mix, 0.0, 1.0))
    return values * mix + x * (1.0 - mix) * weak


def protect_neutral_lut(identity_linear: Tensor, lut_linear: Tensor, cfg: GlobalMatchingConfig) -> Tensor:
    gray = identity_linear.mean(dim=0, keepdim=True)
    chroma = (identity_linear - gray).norm(dim=0, keepdim=True)
    neutral = (1.0 - (chroma / 0.10).clamp(0.0, 1.0)) * float(np.clip(cfg.neutral_protection, 0.0, 1.0))
    max_delta = float(cfg.max_rgb_delta)
    delta = (lut_linear - identity_linear).clamp(-max_delta, max_delta)
    return (identity_linear + delta * (1.0 - neutral)).clamp(0.0, 1.0)


def sample_lut_flat(lut: Tensor, rgb: Tensor) -> Tensor:
    _, size, _, _ = lut.shape
    pos = rgb.clamp(0.0, 1.0) * float(size - 1)
    base = torch.floor(pos).long().clamp(0, size - 1)
    frac = pos - base.to(rgb.dtype)
    upper = (base + 1).clamp(0, size - 1)
    flat_lut = lut.reshape(3, -1)
    out = torch.zeros((rgb.shape[0], 3), device=rgb.device, dtype=rgb.dtype)
    for ri, rw in ((base[:, 0], 1.0 - frac[:, 0]), (upper[:, 0], frac[:, 0])):
        for gi, gw in ((base[:, 1], 1.0 - frac[:, 1]), (upper[:, 1], frac[:, 1])):
            for bi, bw in ((base[:, 2], 1.0 - frac[:, 2]), (upper[:, 2], frac[:, 2])):
                idx = (ri * size + gi) * size + bi
                out = out + flat_lut[:, idx].transpose(0, 1) * (rw * gw * bw).unsqueeze(1)
    return out


def apply_lut_linear(lut: Tensor, img_linear: Tensor, chunk_pixels: int) -> Tensor:
    flat = flatten_rgb(img_linear)
    chunks = []
    for start in range(0, int(flat.shape[0]), int(chunk_pixels)):
        chunks.append(sample_lut_flat(lut, flat[start : start + int(chunk_pixels)]))
    out = torch.cat(chunks, dim=0).reshape(img_linear.shape[0], img_linear.shape[2], img_linear.shape[3], 3)
    return out.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)


def support_visual(support: Tensor) -> np.ndarray:
    support = support / support.max().clamp_min(1e-6)
    return support.max(dim=2).values.detach().cpu().numpy().astype(np.float32)


def image_stats_from_lab(src_lab: Tensor, out_lab: Tensor, out_rgb: Tensor) -> Dict[str, float]:
    delta = out_lab - src_lab
    chroma_src = src_lab[:, 1:3].norm(dim=1)
    chroma_out = out_lab[:, 1:3].norm(dim=1)
    compressed = ((out_rgb <= 0.001) | (out_rgb >= 0.999)).float().mean()
    return {
        "mean_luma_delta": float(delta[:, 0].mean().detach().cpu()),
        "mean_abs_luma_delta": float(delta[:, 0].abs().mean().detach().cpu()),
        "mean_chroma_delta": float((chroma_out - chroma_src).mean().detach().cpu()),
        "mean_abs_chroma_delta": float((chroma_out - chroma_src).abs().mean().detach().cpu()),
        "compressed_pixel_ratio": float(compressed.detach().cpu()),
    }


def run_global_matching(
    source_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    config: Optional[GlobalMatchingConfig | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    require_colour()
    cfg = config if isinstance(config, GlobalMatchingConfig) else GlobalMatchingConfig(**(config or {}))
    device = require_mps()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    source_np = load_rgb(source_path)
    reference_np = load_rgb(reference_path)
    source = to_nchw_mps(source_np, device)
    reference = to_nchw_mps(reference_np, device)
    reference_resized = resize_to_hw(reference, source.shape[2], source.shape[3])
    source_fit = resize_long_edge(source, cfg.fit_long_edge)
    reference_fit = resize_to_hw(reference_resized, source_fit.shape[2], source_fit.shape[3])
    source_linear = srgb_to_linear(source)
    reference_linear = srgb_to_linear(reference_resized)
    source_fit_linear = srgb_to_linear(source_fit)
    reference_fit_linear = srgb_to_linear(reference_fit)
    timings["load_and_resize"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    lut_linear, support = fit_linear_rgb_lut(source_fit_linear, reference_fit_linear, cfg)
    timings["fit_lut"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    base_linear = apply_lut_linear(lut_linear, source_linear, cfg.apply_chunk_pixels)
    base_rgb = soft_gamut_compress(linear_to_srgb(base_linear))
    base_lab = rgb_to_oklab(base_rgb)
    source_lab = rgb_to_oklab(source)
    reference_resized_lab = rgb_to_oklab(reference_resized)
    timings["apply_lut"] = time.perf_counter() - t2

    paths = {
        "base_intermediate": str(output_dir / "base_intermediate.png"),
        "reference_resized": str(output_dir / "reference_resized.png"),
        "source_fit": str(output_dir / "source_fit.png"),
        "lut_support": str(output_dir / "lut_support.png"),
        "metrics": str(output_dir / "metrics.json"),
    }
    save_rgb(paths["base_intermediate"], to_hwc_np(base_rgb))
    save_rgb(paths["reference_resized"], to_hwc_np(reference_resized))
    save_rgb(paths["source_fit"], to_hwc_np(source_fit))
    save_gray(paths["lut_support"], support_visual(support))

    stats = image_stats_from_lab(source_lab, base_lab, base_rgb)
    metrics: Dict[str, Any] = {
        "pipeline_version": GLOBAL_STAGE_VERSION,
        "global_stage_version": GLOBAL_STAGE_VERSION,
        "device": str(device),
        "mps_available": True,
        "source_shape": list(source_np.shape),
        "reference_shape": list(reference_np.shape),
        "fit_shape": [int(source_fit.shape[2]), int(source_fit.shape[3]), 3],
        "config": asdict(cfg),
        "lut_support_coverage": float((support > 0).float().mean().detach().cpu()),
        "timings": timings,
        **stats,
        "paths": paths,
    }
    Path(paths["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["tensors"] = {
        "base_intermediate_lab": base_lab,
        "base_intermediate_rgb": base_rgb,
        "reference_resized_lab": reference_resized_lab,
        "reference_resized_rgb": reference_resized,
        "source_lab": source_lab,
        "source_rgb": source,
    }
    return metrics
