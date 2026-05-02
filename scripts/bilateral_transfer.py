"""LookAlign V0.4 bilateral-grid local affine transfer.

Replaces the low-frequency proxy delta approach with a bilateral-grid
local affine color model.  Edge-awareness is built into the grid structure
(luminance dimension), and misalignment tolerance comes from statistics-based
cell fitting rather than pixel-level correspondences.
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
from PIL import Image, ImageDraw

from scripts.global_matching import (
    Tensor,
    image_stats_from_lab,
    lab_to_rgb,
    save_rgb,
    soft_gamut_compress,
    to_hwc_np,
)

PIPELINE_VERSION = "v0.4.5-edge-aware-bilateral-affine"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BilateralTransferConfig:
    fit_long_edge: int = 768
    spatial_bins: int = 48          # grid cells along the long edge
    luma_bins: int = 40             # luminance bins
    ref_denoise_sigma: float = 0.5  # Gaussian blur σ on reference before splatting;
                                    # suppresses film grain that biases cell means
    affine_regularization: float = 0.05  # pull toward identity
    min_samples_per_cell: int = 2
    max_offset: float = 24.0            # clamp affine offset (Lab a*/b* units)
    max_luma_offset: float = 16.0       # slightly looser clamp for Lab L* offsets
    max_scale_delta: float = 0.35       # clamp per-channel scale around identity
    max_luma_scale_delta: float = 0.22  # slightly looser clamp for Lab L* scale
    coeff_smooth_iterations: int = 2
    coeff_smooth_spatial_sigma: float = 0.06
    coeff_smooth_luma_sigma: float = 0.05
    coeff_smooth_confidence_power: float = 1.5
    coeff_smooth_scale_blend: float = 0.15
    coeff_smooth_offset_blend: float = 0.55
    detail_sigma: float = 2.0
    detail_max_boost: float = 8.0
    detail_positive_bias: float = 0.75
    detail_negative_bias: float = 0.75
    detail_edge_sigma: float = 0.08
    detail_strength: float = 0.85
    detail_negative_strength: float = 0.65
    guided_filter_radius: int = 0       # 0 = disabled; GF destroys source detail
    guided_filter_eps: float = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_long_edge(img: Tensor, long_edge: int, clamp_01: bool = True) -> Tensor:
    _, _, h, w = img.shape
    scale = min(1.0, float(max(16, long_edge)) / float(max(h, w)))
    if scale >= 0.999:
        return img.clone()
    th = max(8, round(h * scale))
    tw = max(8, round(w * scale))
    out = F.interpolate(img, size=(th, tw), mode="bicubic", align_corners=False)
    return out.clamp(0, 1) if clamp_01 else out


def _resize_to_hw(img: Tensor, h: int, w: int) -> Tensor:
    return F.interpolate(img, size=(int(h), int(w)), mode="bicubic", align_corners=False)


def _spatial_gaussian_blur(img: Tensor, sigma: float) -> Tensor:
    """Per-channel 2-D Gaussian blur on (1, C, H, W) tensor."""
    if sigma < 0.1:
        return img
    ks = max(3, int(sigma * 4) | 1)
    pad = ks // 2
    x = torch.arange(ks, device=img.device, dtype=img.dtype) - pad
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()
    # Separable: blur H then W
    kh = k1d.view(1, 1, -1, 1)
    kw = k1d.view(1, 1, 1, -1)
    channels = []
    for c in range(img.shape[1]):
        ch = img[:, c:c+1]
        ch = F.pad(ch, (0, 0, pad, pad), "replicate")
        ch = F.conv2d(ch, kh)
        ch = F.pad(ch, (pad, pad, 0, 0), "replicate")
        ch = F.conv2d(ch, kw)
        channels.append(ch)
    return torch.cat(channels, dim=1)


def _grid_dims(h: int, w: int, spatial_bins: int) -> tuple[int, int]:
    """Grid spatial dimensions preserving aspect ratio."""
    if h >= w:
        gh = spatial_bins
        gw = max(4, round(spatial_bins * w / h))
    else:
        gw = spatial_bins
        gh = max(4, round(spatial_bins * h / w))
    return gh, gw


# ---------------------------------------------------------------------------
# Core: splat → solve → edge-aware smooth → slice
# ---------------------------------------------------------------------------

def _identity_affine(device: torch.device, dtype: torch.dtype) -> Tensor:
    """3×4 identity affine as a flat (12,) tensor."""
    A = torch.zeros(3, 4, device=device, dtype=dtype)
    A[0, 0] = A[1, 1] = A[2, 2] = 1.0
    return A.reshape(12)


def splat_statistics(
    base_lab: Tensor,
    ref_lab: Tensor,
    gh: int, gw: int, gl: int,
) -> Dict[str, Tensor]:
    """Accumulate per-channel statistics into bilateral grid cells.

    Returns dicts of (gh, gw, gl, 3) tensors for sum, sum-of-squares, and
    (gh, gw, gl) for counts.
    """
    _, _, H, W = base_lab.shape
    device, dtype = base_lab.device, base_lab.dtype

    base_flat = base_lab[0].reshape(3, -1).T  # (N, 3)
    ref_flat = ref_lab[0].reshape(3, -1).T

    # Pixel → bilateral grid coordinates (soft trilinear splat)
    yy = torch.arange(H, device=device, dtype=dtype).view(-1, 1).expand(H, W).reshape(-1)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, -1).expand(H, W).reshape(-1)
    luma = (base_flat[:, 0] / 100.0).clamp(0, 1)

    total = gh * gw * gl
    count = torch.zeros(total, device=device, dtype=dtype)
    sum_base = torch.zeros(total, 3, device=device, dtype=dtype)
    sum_ref = torch.zeros(total, 3, device=device, dtype=dtype)
    ssq_base = torch.zeros(total, 3, device=device, dtype=dtype)
    ssq_ref = torch.zeros(total, 3, device=device, dtype=dtype)

    gy = yy / max(H - 1, 1) * (gh - 1)
    gx = xx / max(W - 1, 1) * (gw - 1)
    gz = luma * (gl - 1)

    iy0 = gy.floor().long().clamp(0, gh - 1)
    ix0 = gx.floor().long().clamp(0, gw - 1)
    il0 = gz.floor().long().clamp(0, gl - 1)
    iy1 = (iy0 + 1).clamp(0, gh - 1)
    ix1 = (ix0 + 1).clamp(0, gw - 1)
    il1 = (il0 + 1).clamp(0, gl - 1)

    fy = (gy - iy0.to(dtype)).clamp(0, 1)
    fx = (gx - ix0.to(dtype)).clamp(0, 1)
    fl = (gz - il0.to(dtype)).clamp(0, 1)

    weights = (
        ((iy0, ix0, il0), (1 - fy) * (1 - fx) * (1 - fl)),
        ((iy0, ix0, il1), (1 - fy) * (1 - fx) * fl),
        ((iy0, ix1, il0), (1 - fy) * fx * (1 - fl)),
        ((iy0, ix1, il1), (1 - fy) * fx * fl),
        ((iy1, ix0, il0), fy * (1 - fx) * (1 - fl)),
        ((iy1, ix0, il1), fy * (1 - fx) * fl),
        ((iy1, ix1, il0), fy * fx * (1 - fl)),
        ((iy1, ix1, il1), fy * fx * fl),
    )

    for (iy, ix, il), w in weights:
        cell_idx = iy * (gw * gl) + ix * gl + il
        idx3 = cell_idx.unsqueeze(1).expand(-1, 3)
        count.scatter_add_(0, cell_idx, w)
        w3 = w.unsqueeze(1)
        sum_base.scatter_add_(0, idx3, base_flat * w3)
        sum_ref.scatter_add_(0, idx3, ref_flat * w3)
        ssq_base.scatter_add_(0, idx3, base_flat.square() * w3)
        ssq_ref.scatter_add_(0, idx3, ref_flat.square() * w3)

    reshape = lambda t: t.reshape(gh, gw, gl) if t.ndim == 1 else t.reshape(gh, gw, gl, 3)
    return {
        "count": reshape(count),
        "sum_base": reshape(sum_base),
        "sum_ref": reshape(sum_ref),
        "ssq_base": reshape(ssq_base),
        "ssq_ref": reshape(ssq_ref),
    }


def solve_diagonal_affine(
    stats: Dict[str, Tensor],
    cfg: BilateralTransferConfig,
) -> Tensor:
    """Solve per-cell diagonal affine transfer from local statistics.

    All Lab channels, including L*, are corrected here so the low-frequency
    tone and color structure can converge toward the reference. High-frequency
    detail still comes from the source because the transfer is estimated on
    local cell statistics rather than per-pixel matches.

    Returns (gh, gw, gl, 12) grid of flattened 3×4 diagonal affine matrices.
    """
    device, dtype = stats["count"].device, stats["count"].dtype
    gh, gw, gl = stats["count"].shape

    count = stats["count"].clamp_min(1)
    mu_b = stats["sum_base"] / count.unsqueeze(-1)  # (gh, gw, gl, 3)
    mu_r = stats["sum_ref"] / count.unsqueeze(-1)
    var_b = (stats["ssq_base"] / count.unsqueeze(-1) - mu_b.square()).clamp_min(1e-6)
    var_r = (stats["ssq_ref"] / count.unsqueeze(-1) - mu_r.square()).clamp_min(1e-6)

    raw_scale = torch.sqrt(var_r / var_b)
    scale_min = torch.tensor(
        [1.0 - cfg.max_luma_scale_delta, 1.0 - cfg.max_scale_delta, 1.0 - cfg.max_scale_delta],
        device=device,
        dtype=dtype,
    ).view(1, 1, 1, 3)
    scale_max = torch.tensor(
        [1.0 + cfg.max_luma_scale_delta, 1.0 + cfg.max_scale_delta, 1.0 + cfg.max_scale_delta],
        device=device,
        dtype=dtype,
    ).view(1, 1, 1, 3)
    raw_scale = raw_scale.clamp(scale_min, scale_max)
    raw_offset = mu_r - raw_scale * mu_b
    offset_min = torch.tensor(
        [-cfg.max_luma_offset, -cfg.max_offset, -cfg.max_offset],
        device=device,
        dtype=dtype,
    ).view(1, 1, 1, 3)
    offset_max = torch.tensor(
        [cfg.max_luma_offset, cfg.max_offset, cfg.max_offset],
        device=device,
        dtype=dtype,
    ).view(1, 1, 1, 3)
    raw_offset = raw_offset.clamp(offset_min, offset_max)

    lam = cfg.affine_regularization
    confidence = (stats["count"] / max(float(cfg.min_samples_per_cell), 1.0)).clamp(0.0, 1.0).unsqueeze(-1)
    scale = 1.0 + (raw_scale - 1.0) * confidence * (1.0 - lam)
    offset = raw_offset * confidence * (1.0 - lam)

    grid = torch.zeros(gh, gw, gl, 12, device=device, dtype=dtype)
    grid[..., 0] = scale[..., 0]
    grid[..., 3] = offset[..., 0]
    grid[..., 5] = scale[..., 1]
    grid[..., 7] = offset[..., 1]
    grid[..., 10] = scale[..., 2]
    grid[..., 11] = offset[..., 2]
    return grid


def _grid_cell_guidance(base_lab: Tensor, gh: int, gw: int, gl: int) -> Tensor:
    """Per-cell mean guide luminance in normalized bilateral space."""
    device, dtype = base_lab.device, base_lab.dtype

    guide = (base_lab[0, 0] / 100.0).clamp(0, 1)
    low = F.interpolate(
        guide.unsqueeze(0).unsqueeze(0),
        size=(gh, gw),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    l_centers = torch.linspace(0.0, 1.0, steps=gl, device=device, dtype=dtype)
    return low.unsqueeze(-1).expand(gh, gw, gl) - l_centers.view(1, 1, gl)


def smooth_affine_grid(
    grid: Tensor,
    count: Tensor,
    guide_delta: Tensor,
    cfg: BilateralTransferConfig,
) -> Tensor:
    """Diffuse affine coefficients across low-barrier neighbors only."""
    if cfg.coeff_smooth_iterations <= 0:
        return grid

    confidence = (count / max(float(cfg.min_samples_per_cell), 1.0)).clamp(0.0, 1.0)
    confidence = confidence.pow(cfg.coeff_smooth_confidence_power).unsqueeze(-1)
    out = grid.clone()

    spatial_sigma = max(cfg.coeff_smooth_spatial_sigma, 1e-6)
    luma_sigma = max(cfg.coeff_smooth_luma_sigma, 1e-6)

    neighbor_specs = (
        (0, 1, spatial_sigma),
        (1, 1, spatial_sigma),
        (2, 1, luma_sigma),
    )

    raw = out.clone()

    for _ in range(cfg.coeff_smooth_iterations):
        numer = confidence * out
        denom = confidence.clone()

        for dim, direction, sigma in neighbor_specs:
            src = [slice(None)] * 4
            dst = [slice(None)] * 4
            src[dim] = slice(0, -direction)
            dst[dim] = slice(direction, None)

            delta = guide_delta[tuple(dst[:3])] - guide_delta[tuple(src[:3])]
            weight = torch.exp(-0.5 * (delta / sigma).square()).unsqueeze(-1)

            src_count = confidence[tuple(src)]
            dst_count = confidence[tuple(dst)]
            src_vals = out[tuple(src)]
            dst_vals = out[tuple(dst)]

            w_src_to_dst = weight * src_count
            w_dst_to_src = weight * dst_count

            numer[tuple(dst)] += w_src_to_dst * src_vals
            denom[tuple(dst)] += w_src_to_dst
            numer[tuple(src)] += w_dst_to_src * dst_vals
            denom[tuple(src)] += w_dst_to_src

        smoothed = numer / denom.clamp_min(1e-6)
        out = smoothed

    out_final = raw.clone()

    scale_mix = cfg.coeff_smooth_scale_blend
    offset_mix = cfg.coeff_smooth_offset_blend
    conf_keep = confidence.clamp(0.0, 1.0)
    scale_weight = scale_mix * (1.0 - conf_keep)
    offset_weight = offset_mix * (1.0 - conf_keep)

    for idx in (0, 5, 10):
        out_final[..., idx] = raw[..., idx] * (1.0 - scale_weight[..., 0]) + out[..., idx] * scale_weight[..., 0]
    for idx in (3, 7, 11):
        out_final[..., idx] = raw[..., idx] * (1.0 - offset_weight[..., 0]) + out[..., idx] * offset_weight[..., 0]

    return out_final


def bilateral_slice(
    base_int_lab: Tensor,
    grid: Tensor,
) -> Tensor:
    """Apply bilateral grid to full-res image.

    Guide = base-intermediate luminance so the slice follows the actual tone
    structure being transferred, especially in the highlights.
    Input = base-intermediate (LUT-corrected) colors.
    """
    _, _, H, W = base_int_lab.shape
    gh, gw, gl, _ = grid.shape
    device, dtype = base_int_lab.device, base_int_lab.dtype

    # Guide coordinates: base-intermediate luminance
    guide_L = (base_int_lab[0, 0] / 100.0).clamp(0, 1)  # (H, W)

    # Continuous grid coordinates
    gy = torch.arange(H, device=device, dtype=dtype).view(-1, 1) / max(H - 1, 1) * (gh - 1)
    gx = torch.arange(W, device=device, dtype=dtype).view(1, -1) / max(W - 1, 1) * (gw - 1)
    gz = guide_L * (gl - 1)

    # Trilinear interpolation using grid_sample
    # Reshape grid to (1, 12, gh, gw*gl) and sample with 2D grid_sample
    # OR: use manual trilinear interpolation for clarity

    # Manual trilinear: floor/ceil indices + fractional weights
    iy0 = gy.long().clamp(0, gh - 2).expand(H, W)
    ix0 = gx.long().clamp(0, gw - 2).expand(H, W)
    il0 = gz.long().clamp(0, gl - 2)

    iy1, ix1, il1 = iy0 + 1, ix0 + 1, il0 + 1
    fy = (gy.expand(H, W) - iy0.float()).clamp(0, 1)
    fx = (gx.expand(H, W) - ix0.float()).clamp(0, 1)
    fl = (gz - il0.float()).clamp(0, 1)

    # Sample 8 corners and interpolate
    def sample(iy: Tensor, ix: Tensor, il: Tensor) -> Tensor:
        return grid[iy.reshape(-1), ix.reshape(-1), il.reshape(-1)].reshape(H, W, 12)

    c000 = sample(iy0, ix0, il0)
    c001 = sample(iy0, ix0, il1)
    c010 = sample(iy0, ix1, il0)
    c011 = sample(iy0, ix1, il1)
    c100 = sample(iy1, ix0, il0)
    c101 = sample(iy1, ix0, il1)
    c110 = sample(iy1, ix1, il0)
    c111 = sample(iy1, ix1, il1)

    fl_ = fl.unsqueeze(-1)
    fx_ = fx.unsqueeze(-1)
    fy_ = fy.unsqueeze(-1)

    c00 = c000 * (1 - fl_) + c001 * fl_
    c01 = c010 * (1 - fl_) + c011 * fl_
    c10 = c100 * (1 - fl_) + c101 * fl_
    c11 = c110 * (1 - fl_) + c111 * fl_
    c0 = c00 * (1 - fx_) + c01 * fx_
    c1 = c10 * (1 - fx_) + c11 * fx_
    A_flat = c0 * (1 - fy_) + c1 * fy_  # (H, W, 12)

    # Apply 3×4 affine to base-intermediate colors
    A = A_flat.reshape(H, W, 3, 4)  # (H, W, 3, 4)
    bi = base_int_lab[0].permute(1, 2, 0)  # (H, W, 3)
    ones = torch.ones(H, W, 1, device=device, dtype=dtype)
    bi_aug = torch.cat([bi, ones], dim=-1)  # (H, W, 4)
    out = torch.einsum("hwij,hwj->hwi", A, bi_aug)  # (H, W, 3)
    return out.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


def _clamp_lab_l(lab: Tensor) -> Tensor:
    out = lab.clone()
    out[:, 0:1] = out[:, 0:1].clamp(0.0, 100.0)
    return out


def apply_luma_detail_residual(
    base_lab: Tensor,
    ref_lab: Tensor,
    output_lab: Tensor,
    cfg: BilateralTransferConfig,
) -> tuple[Tensor, Tensor]:
    """Restore fine luminance detail after coarse bilateral transfer.

    The bilateral grid gets the low-frequency tone structure close to the
    reference, but it can soften local luminance contrast. Restore signed
    residual detail so both highlight and shadow contrast can come back.
    """
    if cfg.detail_strength <= 1e-4 or cfg.detail_sigma <= 0.1:
        zero = torch.zeros_like(output_lab[:, 0:1])
        return output_lab, zero

    base_l = base_lab[:, 0:1]
    ref_l = ref_lab[:, 0:1]
    out_l = output_lab[:, 0:1]

    base_low = _spatial_gaussian_blur(base_l, cfg.detail_sigma)
    ref_low = _spatial_gaussian_blur(ref_l, cfg.detail_sigma)
    out_low = _spatial_gaussian_blur(out_l, cfg.detail_sigma)

    base_detail = base_l - base_low
    ref_detail = ref_l - ref_low
    out_detail = out_l - out_low

    positive_gate = torch.sigmoid((ref_detail - cfg.detail_positive_bias * base_detail) / 1.5)
    negative_gate = torch.sigmoid(((-ref_detail) - cfg.detail_negative_bias * (-base_detail)) / 1.5)
    detail_delta = ref_detail - out_detail

    positive_residual = detail_delta.clamp_min(0.0) * positive_gate * cfg.detail_strength
    negative_residual = detail_delta.clamp_max(0.0) * negative_gate * cfg.detail_negative_strength

    guide = (base_l / 100.0).clamp(0.0, 1.0)
    guide_low = _spatial_gaussian_blur(guide, cfg.detail_sigma)
    edge_barrier = torch.exp(-((guide - guide_low).abs() / max(cfg.detail_edge_sigma, 1e-6)).square())

    residual = (positive_residual + negative_residual) * edge_barrier
    residual = residual.clamp(-cfg.detail_max_boost, cfg.detail_max_boost)

    output = output_lab.clone()
    output[:, 0:1] = (output[:, 0:1] + residual).clamp(0.0, 100.0)
    return output, residual


# ---------------------------------------------------------------------------
# Guided filter (He et al.) for optional post-processing
# ---------------------------------------------------------------------------

def _box_filter(x: Tensor, r: int) -> Tensor:
    """Box filter via avg_pool2d."""
    ks = 2 * r + 1
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), ks, stride=1)


def guided_filter(guide: Tensor, src: Tensor, r: int, eps: float) -> Tensor:
    """Edge-preserving guided filter.  guide & src are (1, C, H, W)."""
    g = guide[:, 0:1]  # single-channel guide (luminance)
    mean_g = _box_filter(g, r)
    mean_src = _box_filter(src, r)
    corr_gg = _box_filter(g * g, r)
    corr_gs = _box_filter(g * src, r)
    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_src
    a = cov_gs / (var_g + eps)
    b = mean_src - a * mean_g
    return _box_filter(a, r) * g + _box_filter(b, r)


# ---------------------------------------------------------------------------
# Debug visualizations
# ---------------------------------------------------------------------------

def _grid_to_vis(grid: Tensor, channel: int, size: int = 0) -> np.ndarray:
    """Mean-project the grid along the luminance axis for a single channel.

    If *size* > 0, upscale the tiny grid image to *size* × *size* using
    nearest-neighbour so individual cells are clearly visible.
    """
    v = grid[:, :, :, channel].mean(dim=2)  # (gh, gw)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    arr = v.detach().cpu().numpy().astype(np.float32)
    if size > 0:
        from PIL import Image as _PILImage
        img = _PILImage.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        arr = np.array(img.resize((size, size), _PILImage.NEAREST), dtype=np.float32) / 255.0
    return arr


def _grid_viewport(grid: Tensor, size: int = 256) -> np.ndarray:
    """Render a compact viewport of key bilateral-grid coefficient channels."""
    panels = [
        ("L scale", _grid_to_vis(grid, 0, size=size)),
        ("L offset", _grid_to_vis(grid, 3, size=size)),
        ("a scale", _grid_to_vis(grid, 5, size=size)),
        ("a offset", _grid_to_vis(grid, 7, size=size)),
        ("b scale", _grid_to_vis(grid, 10, size=size)),
        ("b offset", _grid_to_vis(grid, 11, size=size)),
    ]

    tile_w = size
    tile_h = size + 18
    cols = 3
    rows = 2
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)

    for idx, (label, panel) in enumerate(panels):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        panel_rgb = Image.fromarray((np.clip(panel, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")
        canvas.paste(panel_rgb, (x, y + 18))
        draw.rectangle((x, y, x + tile_w - 1, y + 17), fill=(0, 0, 0))
        draw.text((x + 6, y + 2), label, fill=(255, 255, 255))

    return np.asarray(canvas).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_bilateral_transfer(
    *,
    base_intermediate_lab: Tensor,
    base_intermediate_rgb: Tensor,
    reference_resized_lab: Tensor,
    reference_resized_rgb: Tensor,
    source_lab: Tensor,
    source_rgb: Tensor,
    output_dir: str | Path,
    global_metrics: Dict[str, Any],
    config: Optional[BilateralTransferConfig | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config if isinstance(config, BilateralTransferConfig) else BilateralTransferConfig(**(config or {}))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    _, _, full_h, full_w = source_lab.shape

    # ---- Stage 2a: downsample to working resolution ----
    t0 = time.perf_counter()
    # CIE Lab channels are not bounded to [0, 1], so do not clamp
    base_low = _resize_long_edge(base_intermediate_lab, cfg.fit_long_edge, clamp_01=False)
    ref_low = _resize_long_edge(reference_resized_lab, cfg.fit_long_edge, clamp_01=False)
    # Ensure same spatial size
    _, _, lh, lw = base_low.shape
    ref_low = _resize_to_hw(ref_low, lh, lw)
    timings["downsample"] = time.perf_counter() - t0

    # ---- Stage 2a+: denoise reference before splatting ----
    # AI-generated film-style references contain grain noise that biases
    # per-cell means (especially lifting shadows).  A light spatial Gaussian
    # blur suppresses grain while preserving the broad color patterns that
    # cell-level statistics are meant to capture.
    t_dn = time.perf_counter()
    if cfg.ref_denoise_sigma > 0.1:
        ref_low = _spatial_gaussian_blur(ref_low, cfg.ref_denoise_sigma)
    timings["ref_denoise"] = time.perf_counter() - t_dn

    # ---- Stage 2b: splat statistics into bilateral grid ----
    t1 = time.perf_counter()
    gh, gw = _grid_dims(lh, lw, cfg.spatial_bins)
    gl = cfg.luma_bins
    stats = splat_statistics(base_low, ref_low, gh, gw, gl)
    timings["splat"] = time.perf_counter() - t1

    # ---- Stage 2c: solve per-cell affine from raw local statistics ----
    t2 = time.perf_counter()
    grid = solve_diagonal_affine(stats, cfg)
    timings["solve_affine"] = time.perf_counter() - t2

    # ---- Stage 2d: confidence-weighted edge-aware coefficient smoothing ----
    t3 = time.perf_counter()
    guide_delta = _grid_cell_guidance(base_low, gh, gw, gl)
    grid = smooth_affine_grid(grid, stats["count"], guide_delta, cfg)
    timings["smooth_affine"] = time.perf_counter() - t3

    # ---- Stage 3: bilateral slice at full resolution ----
    t4 = time.perf_counter()
    output_lab = bilateral_slice(base_intermediate_lab, grid)
    output_lab = _clamp_lab_l(output_lab)
    timings["bilateral_slice"] = time.perf_counter() - t4

    # ---- Stage 3b: restore fine luminance detail from the reference ----
    t4b = time.perf_counter()
    output_lab, detail_residual_l = apply_luma_detail_residual(
        base_intermediate_lab,
        reference_resized_lab,
        output_lab,
        cfg,
    )
    timings["detail_residual"] = time.perf_counter() - t4b

    # ---- Optional guided-filter post-pass ----
    t5 = time.perf_counter()
    if cfg.guided_filter_radius > 0:
        output_lab = guided_filter(
            source_lab[:, 0:1],  # guide = source luminance
            output_lab,
            cfg.guided_filter_radius,
            cfg.guided_filter_eps,
        )
        output_lab = _clamp_lab_l(output_lab)
    timings["guided_filter"] = time.perf_counter() - t5

    t_rgb = time.perf_counter()
    output_rgb = soft_gamut_compress(lab_to_rgb(output_lab))
    timings["output_rgb"] = time.perf_counter() - t_rgb

    # Difference map uses the already aligned reference from global_matching.
    t_debug = time.perf_counter()
    out_chroma = output_lab[:, 1:3].norm(dim=1)
    ref_chroma = reference_resized_lab[:, 1:3].norm(dim=1)
    delta_l = (output_lab[:, 0] - reference_resized_lab[:, 0]) / 100.0
    delta_c = (out_chroma - ref_chroma) / 100.0
    signed_score = (0.65 * delta_l + 0.35 * delta_c).squeeze(0).detach().cpu().numpy()
    magnitude = (0.85 * np.abs(delta_l.squeeze(0).detach().cpu().numpy()) + 0.15 * np.abs(delta_c.squeeze(0).detach().cpu().numpy()))
    strength = np.clip(magnitude * 4.0, 0.0, 1.0)

    base_gray = 128.0
    red = base_gray + 127.0 * np.clip(signed_score, 0.0, 1.0) * strength
    blue = base_gray + 127.0 * np.clip(-signed_score, 0.0, 1.0) * strength
    gb = base_gray - 96.0 * strength
    diff_heatmap = np.stack(
        [
            np.clip(red, 0, 255),
            np.clip(gb, 0, 255),
            np.clip(blue, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)

    # Bilateral transfer edit map: dark background, showing the actual RGB
    # delta introduced by the bilateral stage relative to the base intermediate.
    base_chroma = base_intermediate_lab[:, 1:3].norm(dim=1)
    edit_delta_l = ((output_lab[:, 0] - base_intermediate_lab[:, 0]) / 100.0).squeeze(0).detach().cpu().numpy()
    edit_delta_c = ((out_chroma - base_chroma) / 100.0).squeeze(0).detach().cpu().numpy()
    detail_residual_np = (detail_residual_l[:, 0] / 100.0).squeeze(0).detach().cpu().numpy()
    edit_strength = np.clip(
        (0.55 * np.abs(edit_delta_l) + 0.15 * np.abs(edit_delta_c) + 0.30 * np.abs(detail_residual_np)) * 6.0,
        0.0,
        1.0,
    )
    edit_mask = (edit_strength > 0.02).astype(np.float32)[..., None]
    output_rgb_np = to_hwc_np(output_rgb)
    base_rgb_np = to_hwc_np(base_intermediate_rgb)
    edit_map = np.clip(np.abs(output_rgb_np - base_rgb_np) * edit_mask, 0.0, 1.0).astype(np.float32)
    timings["debug_visualization"] = time.perf_counter() - t_debug

    # ---- Save outputs and debug images ----
    t6_grid = time.perf_counter()
    paths = dict(global_metrics.get("paths", {}))
    paths.update({
        "grid_viewport": str(output_dir / "grid_viewport.png"),
        "final_output": str(output_dir / "final_output.png"),
        "edit_map": str(output_dir / "edit_map.png"),
        "diff_map": str(output_dir / "diff_map.png"),
        "metrics": str(output_dir / "metrics.json"),
    })

    save_rgb(paths["grid_viewport"], _grid_viewport(grid))
    timings["save_grid_debug"] = time.perf_counter() - t6_grid

    t6_final = time.perf_counter()
    save_rgb(paths["final_output"], to_hwc_np(output_rgb))
    timings["save_final_output"] = time.perf_counter() - t6_final
    
    from PIL import Image as _PILImage
    t6_debug = time.perf_counter()
    save_rgb(paths["edit_map"], edit_map)
    _PILImage.fromarray(diff_heatmap).save(paths["diff_map"])
    timings["save_aux_debug"] = time.perf_counter() - t6_debug

    final_stats = image_stats_from_lab(source_lab, output_lab, output_rgb)

    serial_global = {k: v for k, v in global_metrics.items() if k != "tensors"}
    metrics: Dict[str, Any] = {
        **serial_global,
        "pipeline_version": PIPELINE_VERSION,
        "bilateral_config": asdict(cfg),
        "grid_shape": [gh, gw, gl],
        "fit_resolution": [lh, lw],
        "timings": {**serial_global.get("timings", {}), **timings},
        "final_output_stats": final_stats,
        "paths": paths,
    }
    Path(paths["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["tensors"] = {"final_output_rgb": output_rgb, "final_output_lab": output_lab}
    return metrics
