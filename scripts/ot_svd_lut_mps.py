"""LookAlign V0.3.1: MPS-native sliced OT distilled into an SVD-smoothed LUT."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


Tensor = torch.Tensor
PIPELINE_VERSION = "v0.3.1-mps-ot-svd-lut"


@dataclass
class OTSVDLUTConfig:
    fit_long_edge: int = 768
    sample_count: int = 131072
    ot_iterations: int = 64
    partial_ratio: float = 0.85
    lut_size: int = 33
    svd_rank: int = 8
    svd_smoothing: float = 0.35
    max_luma_delta: float = 0.18
    max_chroma_scale: float = 1.25
    neutral_protection: float = 0.85
    weak_cell_mix: float = 0.85
    apply_chunk_pixels: int = 262144
    seed: int = 1234


def require_mps() -> torch.device:
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise RuntimeError(
            "LookAlign V0.3.1 requires PyTorch MPS in this interpreter. "
            "Run `python3 -c \"import torch; print(torch.backends.mps.is_available())\"` "
            "with the same Python used by the app."
        )
    return torch.device("mps")


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
    r, g, b = srgb_to_linear(rgb).unbind(dim=1)
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


def flatten_rgb(img: Tensor) -> Tensor:
    return img.permute(0, 2, 3, 1).reshape(-1, 3)


def stratified_sample_colors(rgb: Tensor, sample_count: int) -> Tensor:
    flat = flatten_rgb(rgb)
    count = min(int(sample_count), int(flat.shape[0]))
    luma = flat @ torch.tensor([0.2126, 0.7152, 0.0722], device=flat.device, dtype=flat.dtype)
    order = torch.argsort(luma)
    positions = torch.linspace(0, int(order.shape[0]) - 1, count, device=flat.device).round().long()
    return flat[order[positions]]


def build_ot_directions(iterations: int, device: torch.device, dtype: torch.dtype, seed: int) -> Tensor:
    torch.manual_seed(int(seed))
    dirs = torch.randn((max(1, int(iterations)), 3), device=device, dtype=dtype)
    return dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-6)


def sliced_partial_ot_delta(src_lab: Tensor, ref_lab: Tensor, cfg: OTSVDLUTConfig) -> Tensor:
    count = min(int(src_lab.shape[0]), int(ref_lab.shape[0]))
    src_lab = src_lab[:count]
    ref_lab = ref_lab[:count]
    partial = float(np.clip(cfg.partial_ratio, 0.05, 1.0))
    keep = max(16, int(round(count * partial)))
    start = max(0, (count - keep) // 2)
    stop = min(count, start + keep)
    directions = build_ot_directions(cfg.ot_iterations, src_lab.device, src_lab.dtype, cfg.seed)
    delta_sum = torch.zeros_like(src_lab)
    count_sum = torch.zeros((count, 1), device=src_lab.device, dtype=src_lab.dtype)

    for direction in directions:
        src_proj = src_lab @ direction
        ref_proj = ref_lab @ direction
        src_order = torch.argsort(src_proj)
        ref_order = torch.argsort(ref_proj)
        src_idx = src_order[start:stop]
        ref_idx = ref_order[start:stop]
        delta = ref_lab[ref_idx] - src_lab[src_idx]
        delta[:, 0] = delta[:, 0].clamp(-float(cfg.max_luma_delta), float(cfg.max_luma_delta))
        src_chroma = src_lab[src_idx, 1:3].norm(dim=1, keepdim=True).clamp_min(1e-5)
        new_chroma = (src_lab[src_idx, 1:3] + delta[:, 1:3]).norm(dim=1, keepdim=True)
        max_chroma = src_chroma * float(max(1.0, cfg.max_chroma_scale))
        scale = torch.minimum(torch.ones_like(new_chroma), max_chroma / new_chroma.clamp_min(1e-5))
        delta[:, 1:3] = (src_lab[src_idx, 1:3] + delta[:, 1:3]) * scale - src_lab[src_idx, 1:3]
        delta_sum.index_add_(0, src_idx, delta)
        count_sum.index_add_(0, src_idx, torch.ones_like(delta[:, :1]))

    return delta_sum / count_sum.clamp_min(1.0)


def identity_lut_rgb(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    axis = torch.linspace(0.0, 1.0, int(size), device=device, dtype=dtype)
    rr, gg, bb = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((rr, gg, bb), dim=0)


def splat_lut_delta(src_rgb: Tensor, delta_lab: Tensor, cfg: OTSVDLUTConfig) -> Tuple[Tensor, Tensor]:
    size = int(cfg.lut_size)
    device, dtype = src_rgb.device, src_rgb.dtype
    pos = src_rgb.clamp(0.0, 1.0) * float(size - 1)
    base = torch.floor(pos).long().clamp(0, size - 1)
    frac = pos - base.to(dtype)
    upper = (base + 1).clamp(0, size - 1)
    delta_sum = torch.zeros((3, size, size, size), device=device, dtype=dtype)
    weight_sum = torch.zeros((size, size, size), device=device, dtype=dtype)

    for ri, rw in ((base[:, 0], 1.0 - frac[:, 0]), (upper[:, 0], frac[:, 0])):
        for gi, gw in ((base[:, 1], 1.0 - frac[:, 1]), (upper[:, 1], frac[:, 1])):
            for bi, bw in ((base[:, 2], 1.0 - frac[:, 2]), (upper[:, 2], frac[:, 2])):
                weight = (rw * gw * bw).clamp_min(0.0)
                flat_idx = (ri * size + gi) * size + bi
                weight_sum.reshape(-1).index_add_(0, flat_idx, weight)
                for channel in range(3):
                    delta_sum[channel].reshape(-1).index_add_(0, flat_idx, delta_lab[:, channel] * weight)

    return delta_sum / weight_sum.clamp_min(1e-6).unsqueeze(0), weight_sum


def smooth_lut_delta(delta: Tensor, support: Tensor, weak_cell_mix: float) -> Tensor:
    x = delta
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
    smoothed = x
    support_norm = support / support.max().clamp_min(1e-6)
    support_mix = support_norm.clamp(0.0, 1.0).unsqueeze(0)
    weak = float(np.clip(weak_cell_mix, 0.0, 1.0))
    return delta * support_mix + smoothed * (1.0 - support_mix) * weak


def low_rank_axis_smooth(channel: Tensor, rank: int, axis: int) -> Tensor:
    size = channel.shape[0]
    moved = channel.movedim(axis, 0).reshape(size, -1)
    q_rank = max(1, min(int(rank), min(moved.shape)))
    basis = torch.randn((moved.shape[1], q_rank), device=channel.device, dtype=channel.dtype)
    basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)
    for _ in range(5):
        left = moved @ basis
        left = left / left.norm(dim=0, keepdim=True).clamp_min(1e-6)
        basis = moved.transpose(0, 1) @ left
        basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)
    coeff = moved @ basis
    approx = coeff @ basis.transpose(0, 1)
    return approx.reshape([size] + [channel.shape[i] for i in range(3) if i != axis]).movedim(0, axis)


def svd_style_smooth(delta: Tensor, rank: int, smoothing: float) -> Tensor:
    mix = float(np.clip(smoothing, 0.0, 1.0))
    if mix <= 0.0:
        return delta
    channels = []
    for channel in delta:
        approx = torch.zeros_like(channel)
        for axis in range(3):
            approx = approx + low_rank_axis_smooth(channel, rank, axis)
        channels.append(channel * (1.0 - mix) + (approx / 3.0) * mix)
    return torch.stack(channels, dim=0)


def protect_neutral_lut(identity_rgb: Tensor, delta_lab: Tensor, strength: float) -> Tensor:
    flat_rgb = identity_rgb.reshape(3, -1).transpose(0, 1)
    gray = flat_rgb.mean(dim=1, keepdim=True)
    chroma = (flat_rgb - gray).norm(dim=1).reshape(identity_rgb.shape[1:])
    neutral = (1.0 - (chroma / 0.12).clamp(0.0, 1.0)).unsqueeze(0)
    protected = delta_lab.clone()
    protected[1:3] = protected[1:3] * (1.0 - neutral * float(np.clip(strength, 0.0, 1.0)))
    return protected


def make_rgb_lut(delta_lab: Tensor, cfg: OTSVDLUTConfig) -> Tensor:
    identity_rgb = identity_lut_rgb(cfg.lut_size, delta_lab.device, delta_lab.dtype)
    identity_lab = rgb_to_oklab(identity_rgb.unsqueeze(0))[0]
    delta_lab = protect_neutral_lut(identity_rgb, delta_lab, cfg.neutral_protection)
    mapped_lab = identity_lab + delta_lab
    mapped_lab[0] = mapped_lab[0].clamp(identity_lab[0] - float(cfg.max_luma_delta), identity_lab[0] + float(cfg.max_luma_delta))
    mapped_rgb = oklab_to_rgb(mapped_lab.unsqueeze(0))[0]
    return soft_gamut_compress(mapped_rgb)


def soft_gamut_compress(rgb: Tensor) -> Tensor:
    below = F.softplus(-rgb * 8.0) / 8.0
    above = F.softplus((rgb - 1.0) * 8.0) / 8.0
    return (rgb + below - above).clamp(0.0, 1.0)


def apply_lut(lut: Tensor, img: Tensor, chunk_pixels: int) -> Tensor:
    flat = flatten_rgb(img)
    chunks = []
    for start in range(0, int(flat.shape[0]), int(chunk_pixels)):
        chunks.append(sample_lut_flat(lut, flat[start : start + int(chunk_pixels)]))
    out = torch.cat(chunks, dim=0).reshape(img.shape[0], img.shape[2], img.shape[3], 3)
    return out.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)


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


def support_visual(support: Tensor) -> np.ndarray:
    support = support / support.max().clamp_min(1e-6)
    return support.max(dim=2).values.detach().cpu().numpy().astype(np.float32)


def image_stats(src: Tensor, out: Tensor) -> Dict[str, float]:
    src_lab = rgb_to_oklab(src)
    out_lab = rgb_to_oklab(out)
    delta = out_lab - src_lab
    chroma_src = src_lab[:, 1:3].norm(dim=1)
    chroma_out = out_lab[:, 1:3].norm(dim=1)
    compressed = ((out <= 0.001) | (out >= 0.999)).float().mean()
    return {
        "mean_luma_delta": float(delta[:, 0].mean().detach().cpu()),
        "mean_abs_luma_delta": float(delta[:, 0].abs().mean().detach().cpu()),
        "mean_chroma_delta": float((chroma_out - chroma_src).mean().detach().cpu()),
        "mean_abs_chroma_delta": float((chroma_out - chroma_src).abs().mean().detach().cpu()),
        "compressed_pixel_ratio": float(compressed.detach().cpu()),
    }


def run_ot_svd_lut_mps(
    source_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    config: Optional[OTSVDLUTConfig | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config if isinstance(config, OTSVDLUTConfig) else OTSVDLUTConfig(**(config or {}))
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
    timings["load_and_resize"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    src_samples_rgb = stratified_sample_colors(source_fit, cfg.sample_count)
    ref_samples_rgb = stratified_sample_colors(reference_fit, cfg.sample_count)
    src_samples_lab = rgb_to_oklab(src_samples_rgb.transpose(0, 1).reshape(1, 3, -1, 1)).reshape(3, -1).transpose(0, 1)
    ref_samples_lab = rgb_to_oklab(ref_samples_rgb.transpose(0, 1).reshape(1, 3, -1, 1)).reshape(3, -1).transpose(0, 1)
    ot_delta = sliced_partial_ot_delta(src_samples_lab, ref_samples_lab, cfg)
    ot_target_lab = src_samples_lab[: ot_delta.shape[0]] + ot_delta
    ot_preview_rgb = oklab_to_rgb(ot_target_lab.transpose(0, 1).reshape(1, 3, -1, 1)).reshape(3, -1).transpose(0, 1)
    timings["sliced_partial_ot"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    dense_delta, support = splat_lut_delta(src_samples_rgb[: ot_delta.shape[0]], ot_delta, cfg)
    dense_delta = smooth_lut_delta(dense_delta, support, cfg.weak_cell_mix)
    dense_delta = svd_style_smooth(dense_delta, cfg.svd_rank, cfg.svd_smoothing)
    lut_rgb = make_rgb_lut(dense_delta, cfg)
    timings["lut_fit_and_smooth"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    base = apply_lut(lut_rgb, source, cfg.apply_chunk_pixels)
    ot_preview = src_samples_rgb[: ot_preview_rgb.shape[0]].clone()
    preview_count = int(math.sqrt(float(ot_preview.shape[0]))) ** 2
    preview_side = int(math.sqrt(float(preview_count)))
    ot_preview_img = ot_preview_rgb[:preview_count].reshape(preview_side, preview_side, 3).permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
    timings["full_res_apply"] = time.perf_counter() - t3

    paths = {
        "base_intermediate": str(output_dir / "base_intermediate.png"),
        "reference_resized": str(output_dir / "reference_resized.png"),
        "source_fit": str(output_dir / "source_fit.png"),
        "ot_preview": str(output_dir / "ot_preview.png"),
        "lut_support": str(output_dir / "lut_support.png"),
        "metrics": str(output_dir / "metrics.json"),
    }
    save_rgb(paths["base_intermediate"], to_hwc_np(base))
    save_rgb(paths["reference_resized"], to_hwc_np(reference_resized))
    save_rgb(paths["source_fit"], to_hwc_np(source_fit))
    save_rgb(paths["ot_preview"], to_hwc_np(ot_preview_img))
    save_gray(paths["lut_support"], support_visual(support))

    stats = image_stats(source, base)
    metrics: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "device": str(device),
        "mps_available": True,
        "source_shape": list(source_np.shape),
        "reference_shape": list(reference_np.shape),
        "fit_shape": [int(source_fit.shape[2]), int(source_fit.shape[3]), 3],
        "config": asdict(cfg),
        "sample_count_used": int(src_samples_rgb.shape[0]),
        "lut_support_coverage": float((support > 0).float().mean().detach().cpu()),
        "timings": timings,
        **stats,
        "paths": paths,
    }
    Path(paths["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
