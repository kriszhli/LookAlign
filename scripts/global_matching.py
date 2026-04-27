"""LookAlign V0.3.6 global matching: Lab-marginal 3D LUT."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import colour
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from colour import LUT3D
from colour.models import RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_Lab, XYZ_to_RGB


Tensor = torch.Tensor
GLOBAL_STAGE_VERSION = "v0.3.6-global-lab-lut"
PIPELINE_VERSION = "v0.3.6-lab-lut-lightglue-local-diffuse"


@dataclass
class GlobalMatchingConfig:
    fit_long_edge: int = 768
    lut_size: int = 33


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
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


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
    r, g, b = linear.unbind(dim=1)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = torch.sign(l) * torch.abs(l).pow(1.0 / 3.0), torch.sign(m) * torch.abs(m).pow(1.0 / 3.0), torch.sign(s) * torch.abs(s).pow(1.0 / 3.0)
    return torch.stack(
        (
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ),
        dim=1,
    )


def oklab_to_rgb(lab: Tensor) -> Tensor:
    l, a, b = lab.unbind(dim=1)
    l_ = l + 0.3963377774 * a + 0.2158037573 * b
    m_ = l - 0.1055613458 * a - 0.0638541728 * b
    s_ = l - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_.pow(3.0), m_.pow(3.0), s_.pow(3.0)
    linear = torch.stack(
        (
            4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
            -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
            -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3,
        ),
        dim=1,
    )
    return linear_to_srgb(linear)


def soft_gamut_compress(rgb: Tensor) -> Tensor:
    return rgb.clamp(0.0, 1.0)


def rgb_to_lab_np(rgb: np.ndarray) -> np.ndarray:
    cs = RGB_COLOURSPACES["sRGB"]
    xyz = RGB_to_XYZ(rgb, cs, apply_cctf_decoding=True)
    return XYZ_to_Lab(xyz, cs.whitepoint)


def lab_to_rgb_np(lab: np.ndarray) -> np.ndarray:
    cs = RGB_COLOURSPACES["sRGB"]
    xyz = colour.Lab_to_XYZ(lab, cs.whitepoint)
    rgb = XYZ_to_RGB(xyz, cs, apply_cctf_encoding=True)
    return np.clip(rgb, 0.0, 1.0)


def matched_channel(values: np.ndarray, source_samples: np.ndarray, reference_samples: np.ndarray) -> np.ndarray:
    source_sorted = np.sort(np.asarray(source_samples, dtype=np.float64))
    reference_sorted = np.sort(np.asarray(reference_samples, dtype=np.float64))
    source_percentiles = np.linspace(0.0, 1.0, source_sorted.size)
    reference_percentiles = np.linspace(0.0, 1.0, reference_sorted.size)
    source_cdf = np.interp(values, source_sorted, source_percentiles, left=0.0, right=1.0)
    return np.interp(source_cdf, reference_percentiles, reference_sorted)


def build_lab_lut(source_rgb: np.ndarray, reference_rgb: np.ndarray, size: int) -> LUT3D:
    source_lab = rgb_to_lab_np(source_rgb.reshape(-1, 3))
    reference_lab = rgb_to_lab_np(reference_rgb.reshape(-1, 3))
    grid = np.linspace(0.0, 1.0, int(size), dtype=np.float64)
    rr, gg, bb = np.meshgrid(grid, grid, grid, indexing="ij")
    rgb_grid = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)
    lab_grid = rgb_to_lab_np(rgb_grid)
    matched_lab = np.empty_like(lab_grid)
    for channel in range(3):
        matched_lab[:, channel] = matched_channel(lab_grid[:, channel], source_lab[:, channel], reference_lab[:, channel])
    lut_rgb = lab_to_rgb_np(matched_lab).reshape(int(size), int(size), int(size), 3)
    return LUT3D(table=lut_rgb, domain=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64), name="LookAlign V0.3.6")


def apply_lut_np(image_rgb: np.ndarray, lut: LUT3D) -> np.ndarray:
    return np.clip(lut.apply(image_rgb.astype(np.float64)), 0.0, 1.0).astype(np.float32)


def support_visual(size: int) -> np.ndarray:
    return np.ones((size, size), dtype=np.float32)


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
    source_fit_np = to_hwc_np(source_fit)
    reference_fit_np = to_hwc_np(reference_fit)
    timings["load_and_resize"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    lut = build_lab_lut(source_fit_np, reference_fit_np, cfg.lut_size)
    timings["fit_lut"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    base_np = apply_lut_np(source_np, lut)
    base_rgb = to_nchw_mps(base_np, device)
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
    save_rgb(paths["base_intermediate"], base_np)
    save_rgb(paths["reference_resized"], to_hwc_np(reference_resized))
    save_rgb(paths["source_fit"], source_fit_np)
    save_gray(paths["lut_support"], support_visual(cfg.lut_size))

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
        "lut_support_coverage": 1.0,
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
