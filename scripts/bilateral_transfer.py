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

from scripts.global_matching import (
    Tensor,
    image_stats_from_lab,
    lab_to_rgb,
    save_rgb,
    soft_gamut_compress,
    to_hwc_np,
)

PIPELINE_VERSION = "v0.4.0-bilateral-grid-affine"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BilateralTransferConfig:
    fit_long_edge: int = 512
    spatial_bins: int = 32          # grid cells along the long edge
    luma_bins: int = 32             # luminance bins
    ref_denoise_sigma: float = 1.0  # Gaussian blur σ on reference before splatting;
                                    # suppresses film grain that biases cell means
    stats_blur_sigma_xy: float = 1.0
    stats_blur_sigma_l: float = 0.8
    affine_regularization: float = 0.05  # pull toward identity
    min_samples_per_cell: int = 4
    max_offset: float = 24.0            # clamp affine offset (Lab a*/b* units)
    max_luma_offset: float = 12.0       # tighter clamp for Lab L* offsets
    max_scale_delta: float = 0.35       # clamp per-channel scale around identity
    max_luma_scale_delta: float = 0.15  # tighter clamp for Lab L* scale
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
# 1-D separable Gaussian helpers
# ---------------------------------------------------------------------------

def _gauss_kernel_1d(sigma: float, max_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if sigma < 0.1:
        return torch.ones(1, device=device, dtype=dtype)
    ks = min(max(3, int(sigma * 4) | 1), max_size)
    if ks % 2 == 0:
        ks += 1
    x = torch.arange(ks, device=device, dtype=dtype) - ks // 2
    k = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    return k / k.sum()


def _blur_dim(grid: Tensor, dim: int, sigma: float) -> Tensor:
    """Gaussian blur along one dimension of an arbitrary-rank tensor."""
    if sigma < 0.1:
        return grid
    size = grid.shape[dim]
    k = _gauss_kernel_1d(sigma, size, grid.device, grid.dtype)
    pad_n = len(k) // 2
    # Move target dim to last, flatten everything else into batch
    perm = list(range(grid.ndim))
    perm.append(perm.pop(dim))
    g = grid.permute(*perm)
    shape_rest = g.shape[:-1]
    g = g.reshape(-1, 1, size)
    g = F.pad(g, (pad_n, pad_n), mode="replicate")
    g = F.conv1d(g, k.view(1, 1, -1))
    g = g.reshape(*shape_rest, size)
    inv = list(range(grid.ndim))
    inv.insert(dim, inv.pop(-1))
    return g.permute(*inv)


def _blur_grid(grid: Tensor, sigma_xy: float, sigma_l: float) -> Tensor:
    """Separable 3-D blur on grid shaped (gh, gw, gl, C)."""
    g = _blur_dim(grid, 0, sigma_xy)   # height
    g = _blur_dim(g, 1, sigma_xy)      # width
    g = _blur_dim(g, 2, sigma_l)       # luminance
    return g


# ---------------------------------------------------------------------------
# Core: splat → blur → solve → slice
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

    # Pixel → grid cell indices (nearest)
    yy = torch.arange(H, device=device, dtype=dtype).view(-1, 1).expand(H, W).reshape(-1)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, -1).expand(H, W).reshape(-1)
    luma = (base_flat[:, 0] / 100.0).clamp(0, 1)

    iy = (yy / max(H - 1, 1) * (gh - 1)).round().long().clamp(0, gh - 1)
    ix = (xx / max(W - 1, 1) * (gw - 1)).round().long().clamp(0, gw - 1)
    il = (luma * (gl - 1)).round().long().clamp(0, gl - 1)

    cell_idx = iy * (gw * gl) + ix * gl + il  # (N,)

    total = gh * gw * gl
    count = torch.zeros(total, device=device, dtype=dtype)
    sum_base = torch.zeros(total, 3, device=device, dtype=dtype)
    sum_ref = torch.zeros(total, 3, device=device, dtype=dtype)
    ssq_base = torch.zeros(total, 3, device=device, dtype=dtype)
    ssq_ref = torch.zeros(total, 3, device=device, dtype=dtype)

    idx3 = cell_idx.unsqueeze(1).expand(-1, 3)
    count.scatter_add_(0, cell_idx, torch.ones_like(cell_idx, dtype=dtype))
    sum_base.scatter_add_(0, idx3, base_flat)
    sum_ref.scatter_add_(0, idx3, ref_flat)
    ssq_base.scatter_add_(0, idx3, base_flat ** 2)
    ssq_ref.scatter_add_(0, idx3, ref_flat ** 2)

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
    """Solve per-cell diagonal affine transfer from blurred local statistics.

    All Lab channels, including L*, are corrected here so the low-frequency
    tone and color structure can converge toward the reference. High-frequency
    detail still comes from the source because the transfer is estimated on the
    bilateral grid's smoothed cell statistics rather than per-pixel matches.

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
    sparse = (stats["count"] < cfg.min_samples_per_cell).float().unsqueeze(-1)
    scale = 1.0 + (raw_scale - 1.0) * (1.0 - sparse) * (1.0 - lam)
    offset = raw_offset * (1.0 - sparse) * (1.0 - lam)

    grid = torch.zeros(gh, gw, gl, 12, device=device, dtype=dtype)
    grid[..., 0] = scale[..., 0]
    grid[..., 3] = offset[..., 0]
    grid[..., 5] = scale[..., 1]
    grid[..., 7] = offset[..., 1]
    grid[..., 10] = scale[..., 2]
    grid[..., 11] = offset[..., 2]
    return grid


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

    # ---- Stage 2c: blur statistics ----
    t2 = time.perf_counter()
    for key in ("count", "sum_base", "sum_ref", "ssq_base", "ssq_ref"):
        s = stats[key]
        if s.ndim == 3:
            s = s.unsqueeze(-1)
        s = _blur_grid(s, cfg.stats_blur_sigma_xy, cfg.stats_blur_sigma_l)
        stats[key] = s.squeeze(-1) if key == "count" else s
    timings["blur_stats"] = time.perf_counter() - t2

    # ---- Stage 2d: solve per-cell affine ----
    t3 = time.perf_counter()
    grid = solve_diagonal_affine(stats, cfg)
    timings["solve_affine"] = time.perf_counter() - t3

    # ---- Stage 3: bilateral slice at full resolution ----
    t4 = time.perf_counter()
    output_lab = bilateral_slice(base_intermediate_lab, grid)
    output_lab = _clamp_lab_l(output_lab)
    timings["bilateral_slice"] = time.perf_counter() - t4



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

    output_rgb = soft_gamut_compress(lab_to_rgb(output_lab))

    # Compute aligned difference map for UI visualization ONLY
    ref_rgb = global_metrics["tensors"]["reference_resized_rgb"]
    import cv2
    ref_L_full = ((reference_resized_lab[0, 0].detach().cpu().numpy() / 100.0) * 255).clip(0, 255).astype(np.uint8)
    out_L_full = ((output_lab[0, 0].detach().cpu().numpy() / 100.0) * 255).clip(0, 255).astype(np.uint8)
    inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_full = inst.calc(ref_L_full, out_L_full, None)
    
    fh, fw = out_L_full.shape
    map_x_f = np.tile(np.arange(fw), (fh, 1)).astype(np.float32) + flow_full[..., 0]
    map_y_f = np.repeat(np.arange(fh), fw).reshape(fh, fw).astype(np.float32) + flow_full[..., 1]
    
    ref_rgb_np = ref_rgb[0].permute(1, 2, 0).detach().cpu().numpy()
    ref_rgb_warped_np = cv2.remap(ref_rgb_np, map_x_f, map_y_f, cv2.INTER_LINEAR)
    ref_rgb_warped = torch.from_numpy(ref_rgb_warped_np).permute(2, 0, 1).unsqueeze(0).to(ref_rgb.device)
    
    diff_map = (output_rgb - ref_rgb_warped).abs().mean(dim=1).squeeze(0).detach().cpu().numpy()
    diff_norm = (np.clip(diff_map * 2.5, 0, 1) * 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)
    diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)

    # ---- Save outputs and debug images ----
    t6 = time.perf_counter()
    paths = dict(global_metrics.get("paths", {}))
    paths.update({
        "final_output": str(output_dir / "final_output.png"),
        "diff_map": str(output_dir / "diff_map.png"),
        "metrics": str(output_dir / "metrics.json"),
    })

    save_rgb(paths["final_output"], to_hwc_np(output_rgb))
    
    from PIL import Image as _PILImage
    _PILImage.fromarray(diff_heatmap).save(paths["diff_map"])

    timings["save_debug"] = time.perf_counter() - t6

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
