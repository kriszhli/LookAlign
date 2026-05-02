"""XFeat*-based pre-alignment for LookAlign V0.4.5."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


class InterpolateSparse2d(nn.Module):
    """Interpolate a feature map at sparse 2D positions."""

    def __init__(self, mode: str = "bicubic") -> None:
        super().__init__()
        self.mode = mode

    def normgrid(self, x: Tensor, height: int, width: int) -> Tensor:
        scale = torch.tensor([max(width - 1, 1), max(height - 1, 1)], device=x.device, dtype=x.dtype)
        return 2.0 * (x / scale) - 1.0

    def forward(self, x: Tensor, pos: Tensor, height: int, width: int) -> Tensor:
        grid = self.normgrid(pos, height, width).unsqueeze(-2).to(x.dtype)
        sampled = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return sampled.permute(0, 2, 3, 1).squeeze(-2)


class BasicLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class XFeatModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        self.skip1 = nn.Sequential(nn.AvgPool2d(4, stride=4), nn.Conv2d(1, 24, 1, stride=1, padding=0))
        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )
        self.block2 = nn.Sequential(BasicLayer(24, 24, stride=1), BasicLayer(24, 24, stride=1))
        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )
        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )
        self.block_fusion = nn.Sequential(BasicLayer(64, 64, stride=1), BasicLayer(64, 64, stride=1), nn.Conv2d(64, 64, 1, padding=0))
        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )
        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )
        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def _unfold2d(self, x: Tensor, ws: int = 2) -> Tensor:
        batch, channels, height, width = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(batch, channels, height // ws, width // ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(batch, -1, height // ws, width // ws)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        feats = self.block_fusion(x3 + x4 + x5)

        heatmap = self.heatmap_head(feats)
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))
        return feats, keypoints, heatmap


class XFeat(nn.Module):
    def __init__(self, weights: str | Path, top_k: int = 1024, detection_threshold: float = 0.05, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = XFeatModel().to(self.dev).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold
        state = torch.load(weights, map_location=self.dev, weights_only=True)
        self.net.load_state_dict(state)
        self.interpolator = InterpolateSparse2d("bicubic")
        self.nearest = InterpolateSparse2d("nearest")
        self.bilinear = InterpolateSparse2d("bilinear")

    @torch.inference_mode()
    def match_xfeat_star(
        self,
        im_set1: np.ndarray | Tensor,
        im_set2: np.ndarray | Tensor,
        top_k: Optional[int] = None,
        multiscale: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if top_k is None:
            top_k = self.top_k
        im_set1 = self.parse_input(im_set1)
        im_set2 = self.parse_input(im_set2)

        out1 = self.detect_and_compute_dense(im_set1, top_k=top_k, multiscale=multiscale)
        out2 = self.detect_and_compute_dense(im_set2, top_k=top_k, multiscale=multiscale)
        matches = self.batch_match(out1["descriptors"], out2["descriptors"])
        refined = self.refine_matches(out1, out2, matches=matches, batch_idx=0)
        return refined[:, :2].cpu().numpy(), refined[:, 2:].cpu().numpy()

    def parse_input(self, x: np.ndarray | Tensor) -> Tensor:
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                x = torch.from_numpy(x).permute(2, 0, 1)[None]
            elif x.ndim == 4:
                x = torch.from_numpy(x).permute(0, 3, 1, 2)
            else:
                raise RuntimeError("Expected image array with shape (H, W, C) or (B, H, W, C).")
        elif x.ndim == 3:
            x = x[None]
        return x.to(self.dev).float()

    def preprocess_tensor(self, x: Tensor) -> tuple[Tensor, float, float]:
        if x.ndim != 4:
            raise RuntimeError("Input tensor needs to be in (B, C, H, W) format.")
        height, width = x.shape[-2:]
        aligned_h = max((height // 32) * 32, 32)
        aligned_w = max((width // 32) * 32, 32)
        rh = height / aligned_h
        rw = width / aligned_w
        x = F.interpolate(x, (aligned_h, aligned_w), mode="bilinear", align_corners=False)
        return x, rh, rw

    def get_kpts_heatmap(self, kpts: Tensor, softmax_temp: float = 1.0) -> Tensor:
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        batch, _, height, width = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(batch, height, width, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(batch, 1, height * 8, width * 8)
        return heatmap

    def nms(self, x: Tensor, threshold: float = 0.05, kernel_size: int = 5) -> Tensor:
        batch, _, _, _ = x.shape
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]
        pad_val = max(len(points) for points in pos_batched)
        out = torch.zeros((batch, pad_val, 2), dtype=torch.long, device=x.device)
        for idx, points in enumerate(pos_batched):
            out[idx, : len(points), :] = points
        return out

    def create_xy(self, height: int, width: int, dev: torch.device) -> Tensor:
        yy, xx = torch.meshgrid(torch.arange(height, device=dev), torch.arange(width, device=dev), indexing="ij")
        return torch.cat([xx[..., None], yy[..., None]], -1).reshape(-1, 2)

    def extract_dense(self, x: Tensor, top_k: int = 8000) -> tuple[Tensor, Tensor]:
        if top_k < 1:
            top_k = 100_000_000
        x, rh1, rw1 = self.preprocess_tensor(x)
        feats_map, _, heatmap = self.net(x)
        batch, channels, height, width = feats_map.shape
        xy = (self.create_xy(height, width, feats_map.device) * 8).expand(batch, -1, -1)
        feats = feats_map.permute(0, 2, 3, 1).reshape(batch, -1, channels)
        rel = heatmap.permute(0, 2, 3, 1).reshape(batch, -1)
        _, top_idx = torch.topk(rel, k=min(len(rel[0]), top_k), dim=-1)
        feats = torch.gather(feats, 1, top_idx[..., None].expand(-1, -1, channels))
        mkpts = torch.gather(xy, 1, top_idx[..., None].expand(-1, -1, 2))
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, -1)
        return mkpts, feats

    def extract_dualscale(self, x: Tensor, top_k: int, s1: float = 0.6, s2: float = 1.3) -> tuple[Tensor, Tensor, Tensor]:
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode="bilinear")
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode="bilinear")
        mkpts1, feats1 = self.extract_dense(x1, int(top_k * 0.20))
        mkpts2, feats2 = self.extract_dense(x2, int(top_k * 0.80))
        mkpts = torch.cat([mkpts1 / s1, mkpts2 / s2], dim=1)
        sc1 = torch.ones(mkpts1.shape[:2], device=mkpts1.device) * (1 / s1)
        sc2 = torch.ones(mkpts2.shape[:2], device=mkpts2.device) * (1 / s2)
        scales = torch.cat([sc1, sc2], dim=1)
        feats = torch.cat([feats1, feats2], dim=1)
        return mkpts, scales, feats

    def detect_and_compute_dense(self, x: Tensor, top_k: Optional[int] = None, multiscale: bool = True) -> Dict[str, Tensor]:
        if top_k is None:
            top_k = self.top_k
        if multiscale:
            mkpts, scales, feats = self.extract_dualscale(x, top_k)
        else:
            mkpts, feats = self.extract_dense(x, top_k)
            scales = torch.ones(mkpts.shape[:2], device=mkpts.device)
        return {"keypoints": mkpts, "descriptors": feats, "scales": scales}

    @torch.inference_mode()
    def batch_match(self, feats1: Tensor, feats2: Tensor, min_cossim: float = -1) -> list[tuple[Tensor, Tensor]]:
        cossim = torch.bmm(feats1, feats2.permute(0, 2, 1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0, 2, 1), dim=-1)
        idx0 = torch.arange(len(match12[0]), device=match12.device)
        batched_matches: list[tuple[Tensor, Tensor]] = []
        for batch_idx in range(len(feats1)):
            mutual = match21[batch_idx][match12[batch_idx]] == idx0
            if min_cossim > 0:
                cossim_max, _ = cossim[batch_idx].max(dim=1)
                good = cossim_max > min_cossim
                idx0_b = idx0[mutual & good]
                idx1_b = match12[batch_idx][mutual & good]
            else:
                idx0_b = idx0[mutual]
                idx1_b = match12[batch_idx][mutual]
            batched_matches.append((idx0_b, idx1_b))
        return batched_matches

    def subpix_softmax2d(self, heatmaps: Tensor, temp: float = 3.0) -> Tensor:
        count, height, width = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, height * width), -1).view(-1, height, width)
        xx, yy = torch.meshgrid(torch.arange(width, device=heatmaps.device), torch.arange(height, device=heatmaps.device), indexing="xy")
        xx = xx - (width // 2)
        yy = yy - (height // 2)
        coords_x = xx[None, ...] * heatmaps
        coords_y = yy[None, ...] * heatmaps
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(count, height * width, 2)
        return coords.sum(1)

    def refine_matches(self, d0: Dict[str, Tensor], d1: Dict[str, Tensor], matches: list[tuple[Tensor, Tensor]], batch_idx: int, fine_conf: float = 0.25) -> Tensor:
        idx0, idx1 = matches[batch_idx]
        feats1 = d0["descriptors"][batch_idx][idx0]
        feats2 = d1["descriptors"][batch_idx][idx1]
        mkpts0 = d0["keypoints"][batch_idx][idx0]
        mkpts1 = d1["keypoints"][batch_idx][idx1]
        scales0 = d0["scales"][batch_idx][idx0]
        offsets = self.net.fine_matcher(torch.cat([feats1, feats2], dim=-1))
        conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))
        mkpts0 = mkpts0 + offsets * scales0[:, None]
        mask_good = conf > fine_conf
        mkpts0 = mkpts0[mask_good]
        mkpts1 = mkpts1[mask_good]
        return torch.cat([mkpts0, mkpts1], dim=-1)


@dataclass
class XFeatAlignmentConfig:
    max_long_edge: int = 1536
    top_k: int = 4096
    detection_threshold: float = 0.05
    multiscale: bool = True
    min_inlier_count: int = 24
    ransac_threshold: float = 3.0
    max_matches_drawn: int = 1000
    max_mean_reprojection_error: float = 4.0
    min_overlap_ratio: float = 0.55
    max_corner_shift_ratio: float = 0.35
    mesh_grid_size: int = 128
    mesh_sigma: float = 0.18
    mesh_strength: float = 1.0
    mesh_max_offset_ratio: float = 0.06
    device: str = "auto"
    weights_path: str = str(Path(__file__).resolve().parents[1] / "ckpts" / "xfeat.pt")


def _select_device(device_hint: str) -> torch.device:
    if device_hint == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_hint)


def _load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _file_fingerprint(path: str | Path) -> str:
    p = Path(path)
    stat = p.stat()
    payload = f"{p.resolve()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _cache_file_path(source_path: str | Path, reference_path: str | Path, cfg: XFeatAlignmentConfig) -> Path:
    source_fp = _file_fingerprint(source_path)
    reference_fp = _file_fingerprint(reference_path)
    weights_fp = _file_fingerprint(cfg.weights_path)
    key = {
        "version": 7,
        "source_fp": source_fp,
        "reference_fp": reference_fp,
        "weights_fp": weights_fp,
        "max_long_edge": int(cfg.max_long_edge),
        "top_k": int(cfg.top_k),
        "detection_threshold": float(cfg.detection_threshold),
        "multiscale": bool(cfg.multiscale),
        "ransac_threshold": float(cfg.ransac_threshold),
        "mesh_grid_size": int(cfg.mesh_grid_size),
        "mesh_sigma": float(cfg.mesh_sigma),
        "mesh_strength": float(cfg.mesh_strength),
        "mesh_max_offset_ratio": float(cfg.mesh_max_offset_ratio),
    }
    digest = hashlib.sha1(repr(sorted(key.items())).encode("utf-8")).hexdigest()
    root = Path(__file__).resolve().parents[1]
    cache_dir = root / "outputs" / "_xfeat_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{digest}.npz"


def _save_rgb(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _resize_long_edge_np(img: np.ndarray, long_edge: int) -> tuple[np.ndarray, float]:
    height, width = img.shape[:2]
    scale = min(1.0, float(max(16, int(long_edge))) / float(max(height, width)))
    if scale >= 0.999:
        return img.copy(), 1.0
    target_h = max(8, int(round(height * scale)))
    target_w = max(8, int(round(width * scale)))
    out = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return out, scale


def _draw_match_overlay(
    src_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inliers: np.ndarray,
    max_matches: int,
    fallback_text: Optional[str] = None,
) -> np.ndarray:
    src = (np.clip(src_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    ref = (np.clip(ref_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    target_h = max(64, min(max(src.shape[0], ref.shape[0]), 720))
    src_scale = target_h / max(src.shape[0], 1)
    ref_scale = target_h / max(ref.shape[0], 1)
    src_disp = cv2.resize(src, (max(1, int(round(src.shape[1] * src_scale))), target_h), interpolation=cv2.INTER_AREA)
    ref_disp = cv2.resize(ref, (max(1, int(round(ref.shape[1] * ref_scale))), target_h), interpolation=cv2.INTER_AREA)
    height = target_h
    width = src_disp.shape[1] + ref_disp.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, : src_disp.shape[1]] = src_disp
    canvas[:, src_disp.shape[1] :] = ref_disp
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)

    if fallback_text:
        draw.rectangle((0, 0, width, 26), fill=(0, 0, 0))
        draw.text((8, 6), fallback_text, fill=(255, 80, 80))
        return np.asarray(img).astype(np.float32) / 255.0

    if src_pts.size == 0 or ref_pts.size == 0:
        draw.rectangle((0, 0, width, 26), fill=(0, 0, 0))
        draw.text((8, 6), "No XFeat* matches", fill=(255, 80, 80))
        return np.asarray(img).astype(np.float32) / 255.0

    order = np.flatnonzero(inliers.astype(bool))
    if order.size == 0:
        order = np.arange(len(src_pts))
    if order.size > max_matches:
        order = order[np.linspace(0, order.size - 1, num=max_matches, dtype=int)]
    offset_x = src_disp.shape[1]
    overlay_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_layer)
    for idx in order:
        p0_xy = np.round(src_pts[idx] * src_scale).astype(int)
        p1_xy = np.round(ref_pts[idx] * ref_scale).astype(int) + np.array([offset_x, 0])
        p0 = tuple(p0_xy.tolist())
        p1 = tuple(p1_xy.tolist())
        overlay_draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(80, 255, 120, 80), width=1)
        overlay_draw.ellipse((p0[0] - 1, p0[1] - 1, p0[0] + 1, p0[1] + 1), fill=(255, 255, 0, 150))
        overlay_draw.ellipse((p1[0] - 1, p1[1] - 1, p1[0] + 1, p1[1] + 1), fill=(255, 255, 0, 150))
    
    img = Image.alpha_composite(img.convert("RGBA"), overlay_layer).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def _draw_warp_field(field_x: np.ndarray, field_y: np.ndarray, bg_rgb: Optional[np.ndarray] = None, step: int = 32) -> np.ndarray:
    h, w = field_x.shape
    if bg_rgb is not None:
        bg_gray = cv2.cvtColor((np.clip(bg_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2RGB)
        img = (img * 0.4 + 153).astype(np.uint8)
    else:
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx = field_x[y, x]
            dy = field_y[y, x]
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                cv2.circle(img, (x, y), 1, (200, 200, 200), -1)
                continue
            
            end = (int(x + dx * 6.0 ), int(y + dy * 6.0 ))
            cv2.arrowedLine(img, (x, y), end, (255, 50, 50), 1, tipLength=0.3)
            
    mesh_offset = np.sqrt(field_x**2 + field_y**2)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle((0, 0, 400, 26), fill=(0, 0, 0))
    max_val = mesh_offset.max() if mesh_offset.size > 0 else 0.0
    draw.text((8, 6), f"Displacement Quiver (Max: {max_val:.1f}px, Arrow Scale: 6x)", fill=(255, 255, 255))
    return np.asarray(pil_img).astype(np.float32) / 255.0


def _draw_aligned_stack(source_rgb: np.ndarray, reference_rgb: np.ndarray, target_h: int = 720) -> np.ndarray:
    src = (np.clip(source_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    ref = (np.clip(reference_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    height, width = source_rgb.shape[:2]
    if reference_rgb.shape[:2] != (height, width):
        ref = cv2.resize(ref, (width, height), interpolation=cv2.INTER_LINEAR)

    scale = target_h / max(height, 1)
    disp_h = max(64, min(int(round(height * scale)), target_h))
    disp_w = max(1, int(round(width * (disp_h / max(height, 1)))))
    src_disp = cv2.resize(src, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    ref_disp = cv2.resize(ref, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    overlay = np.empty_like(src_disp)
    overlay[..., 0] = src_disp[..., 0]
    overlay[..., 1] = ref_disp[..., 1]
    overlay[..., 2] = ref_disp[..., 2]

    canvas = np.zeros((disp_h + 26, disp_w, 3), dtype=np.uint8)
    canvas[26:] = overlay
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, disp_w, 26), fill=(0, 0, 0))
    draw.text((8, 6), "Spatial stack: source (R) + aligned reference (GB)", fill=(255, 255, 255))
    return np.asarray(img).astype(np.float32) / 255.0


def _largest_valid_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    height, width = mask.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = (0, 0, width, height)
    for y in range(height):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[int] = []
        x = 0
        while x <= width:
            curr = heights[x] if x < width else 0
            if not stack or curr >= heights[stack[-1]]:
                stack.append(x)
                x += 1
                continue
            top = stack.pop()
            rect_w = x if not stack else x - stack[-1] - 1
            area = int(heights[top] * rect_w)
            if area > best_area and heights[top] > 0 and rect_w > 0:
                best_area = area
                x1 = x - rect_w
                y1 = y - heights[top] + 1
                best = (x1, y1, rect_w, int(heights[top]))
    return best


def _fallback_alignment(source_rgb: np.ndarray, reference_rgb: np.ndarray, match_path: Path, reason: str) -> Dict[str, Any]:
    overlay = _draw_match_overlay(source_rgb, reference_rgb, np.empty((0, 2)), np.empty((0, 2)), np.zeros(0), 0, fallback_text=reason)
    aligned_stack = _draw_aligned_stack(source_rgb, reference_rgb)
    aligned_path = match_path.with_name("xfeat_aligned_stack.png")
    warp_path = match_path.with_name("xfeat_warp_field.png")
    zero_field = np.zeros(source_rgb.shape[:2], dtype=np.float32)
    warp_vis = _draw_warp_field(zero_field, zero_field, bg_rgb=source_rgb)
    _save_rgb(match_path, overlay)
    _save_rgb(aligned_path, aligned_stack)
    _save_rgb(warp_path, warp_vis)
    return {
        "source_rgb": source_rgb,
        "reference_rgb": reference_rgb,
        "paths": {
            "xfeat_matches": str(match_path),
            "xfeat_aligned_stack": str(aligned_path),
            "xfeat_warp_field": str(warp_path),
        },
        "metrics": {
            "status": "fallback",
            "reason": reason,
            "match_count": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "crop_bounds": [0, 0, int(source_rgb.shape[1]), int(source_rgb.shape[0])],
            "aligned_shape": [int(source_rgb.shape[0]), int(source_rgb.shape[1]), 3],
        },
    }


def _global_warp_is_usable(
    H: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: np.ndarray,
    source_shape: tuple[int, int, int],
    reference_shape: tuple[int, int, int],
    src_scale: float,
    cfg: XFeatAlignmentConfig,
) -> tuple[bool, Dict[str, float]]:
    inlier_src = src_pts[inlier_mask]
    inlier_ref = ref_pts[inlier_mask]
    if inlier_src.shape[0] == 0:
        return False, {"mean_reprojection_error": float("inf"), "overlap_ratio": 0.0, "corner_shift_ratio": 1.0}

    projected = cv2.perspectiveTransform(inlier_ref.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    reproj = np.linalg.norm(projected - inlier_src, axis=1)
    mean_reproj = float(reproj.mean())

    src_h, src_w = source_shape[:2]
    ref_h, ref_w = reference_shape[:2]
    ref_corners = np.array([[0, 0], [0, ref_h - 1], [ref_w - 1, ref_h - 1], [ref_w - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(ref_corners, H).reshape(-1, 2)
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    inter_w = max(0.0, min(float(src_w), float(x1)) - max(0.0, float(x0)))
    inter_h = max(0.0, min(float(src_h), float(y1)) - max(0.0, float(y0)))
    overlap_ratio = float((inter_w * inter_h) / max(float(src_w * src_h), 1.0))

    src_corners = np.array([[0, 0], [0, src_h - 1], [src_w - 1, src_h - 1], [src_w - 1, 0]], dtype=np.float32)
    corner_shift = np.linalg.norm(warped - src_corners, axis=1).mean()
    corner_shift_ratio = float(corner_shift / max(float((src_w**2 + src_h**2) ** 0.5), 1.0))

    usable = (
        (mean_reproj * src_scale) <= cfg.max_mean_reprojection_error
        and overlap_ratio >= cfg.min_overlap_ratio
        and corner_shift_ratio <= cfg.max_corner_shift_ratio
    )
    return usable, {
        "mean_reprojection_error": mean_reproj,
        "overlap_ratio": overlap_ratio,
        "corner_shift_ratio": corner_shift_ratio,
    }


def _build_mesh_residual_field(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: np.ndarray,
    H: np.ndarray,
    out_h: int,
    out_w: int,
    cfg: XFeatAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    inlier_src = src_pts[inlier_mask].astype(np.float32)
    inlier_ref = ref_pts[inlier_mask].astype(np.float32)
    if inlier_src.shape[0] == 0:
        zero = np.zeros((out_h, out_w), dtype=np.float32)
        return zero, zero

    projected = cv2.perspectiveTransform(inlier_ref.reshape(-1, 1, 2), H).reshape(-1, 2)
    residual = (inlier_src - projected).astype(np.float32)
    diag = max(float((out_w**2 + out_h**2) ** 0.5), 1.0)
    max_offset = float(cfg.mesh_max_offset_ratio) * diag
    residual = np.clip(residual, -max_offset, max_offset) * float(cfg.mesh_strength)

    grid_size = max(8, int(cfg.mesh_grid_size))
    yy = np.linspace(0.0, out_h - 1.0, grid_size, dtype=np.float32)
    xx = np.linspace(0.0, out_w - 1.0, grid_size, dtype=np.float32)
    mesh_x, mesh_y = np.meshgrid(xx, yy)
    query = np.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=1)

    sigma_px = max(float(cfg.mesh_sigma) * diag, 1.0)
    diff = query[:, None, :] - inlier_src[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    gauss_weights = np.exp(-0.5 * dist2 / (sigma_px * sigma_px)).astype(np.float32)
    
    idw_weights = gauss_weights / np.maximum(dist2, 1e-4)
    weights_sum = np.maximum(idw_weights.sum(axis=1, keepdims=True), 1e-8)
    mesh_residual = (idw_weights @ residual) / weights_sum
    
    envelope = np.clip(gauss_weights.max(axis=1, keepdims=True) * 5.0, 0.0, 1.0)
    mesh_residual = mesh_residual * envelope

    field_x = cv2.resize(mesh_residual[:, 0].reshape(grid_size, grid_size), (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    field_y = cv2.resize(mesh_residual[:, 1].reshape(grid_size, grid_size), (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    return field_x.astype(np.float32), field_y.astype(np.float32)


def _warp_reference_locally(
    reference_rgb: np.ndarray,
    H: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: np.ndarray,
    out_h: int,
    out_w: int,
    cfg: XFeatAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float], np.ndarray, np.ndarray]:
    global_ref = cv2.warpPerspective(reference_rgb.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    global_mask = cv2.warpPerspective(
        np.ones(reference_rgb.shape[:2], dtype=np.uint8),
        H,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
    ).astype(np.float32)

    field_x, field_y = _build_mesh_residual_field(src_pts, ref_pts, inlier_mask, H, out_h, out_w, cfg)
    yy, xx = np.meshgrid(np.arange(out_h, dtype=np.float32), np.arange(out_w, dtype=np.float32), indexing="ij")
    remap_x = xx - field_x
    remap_y = yy - field_y

    warped_ref = cv2.remap(global_ref, remap_x, remap_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.remap(global_mask, remap_x, remap_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    mesh_offset = np.sqrt(field_x * field_x + field_y * field_y)
    mesh_stats = {
        "mesh_mean_offset": float(mesh_offset.mean()),
        "mesh_max_offset": float(mesh_offset.max()),
    }
    return warped_ref, warped_mask, mesh_stats, field_x, field_y


@lru_cache(maxsize=4)
def _get_model(device_str: str, top_k: int, detection_threshold: float, weights_path: str) -> XFeat:
    device = torch.device(device_str)
    return XFeat(weights=weights_path, top_k=top_k, detection_threshold=detection_threshold, device=device)


@torch.no_grad()
def run_xfeat_alignment(
    source_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    config: Optional[XFeatAlignmentConfig | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config if isinstance(config, XFeatAlignmentConfig) else XFeatAlignmentConfig(**(config or {}))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    match_path = output_dir / "xfeat_matches.png"
    aligned_path = output_dir / "xfeat_aligned_stack.png"
    warp_path = output_dir / "xfeat_warp_field.png"
    cache_path = _cache_file_path(source_path, reference_path, cfg)

    source_fp = _file_fingerprint(source_path)
    reference_fp = _file_fingerprint(reference_path)
    weights_fp = _file_fingerprint(cfg.weights_path)
    cache_key = {
        "version": 7,
        "source_fp": source_fp,
        "reference_fp": reference_fp,
        "weights_fp": weights_fp,
        "max_long_edge": int(cfg.max_long_edge),
        "top_k": int(cfg.top_k),
        "detection_threshold": float(cfg.detection_threshold),
        "multiscale": bool(cfg.multiscale),
        "ransac_threshold": float(cfg.ransac_threshold),
        "mesh_grid_size": int(cfg.mesh_grid_size),
        "mesh_sigma": float(cfg.mesh_sigma),
        "mesh_strength": float(cfg.mesh_strength),
        "mesh_max_offset_ratio": float(cfg.mesh_max_offset_ratio),
    }

    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)
            meta = cached["meta"].item()
            if all(meta.get(k) == v for k, v in cache_key.items()):
                overlay_key = "overlay_u8" if "overlay_u8" in cached.files else "overlay"
                overlay_raw = cached[overlay_key]
                if overlay_raw.dtype == np.uint8:
                    overlay = overlay_raw.astype(np.float32) / 255.0
                else:
                    overlay = overlay_raw.astype(np.float32)
                _save_rgb(match_path, overlay)
                aligned_stack = _draw_aligned_stack(
                    cached["source_rgb"].astype(np.float32),
                    cached["reference_rgb"].astype(np.float32),
                )
                _save_rgb(aligned_path, aligned_stack)
                if "field_x" in cached.files and "field_y" in cached.files:
                    warp_vis = _draw_warp_field(cached["field_x"], cached["field_y"], bg_rgb=cached["source_rgb"])
                    _save_rgb(warp_path, warp_vis)
                return {
                    "source_rgb": cached["source_rgb"].astype(np.float32),
                    "reference_rgb": cached["reference_rgb"].astype(np.float32),
                    "paths": {
                        "xfeat_matches": str(match_path),
                        "xfeat_aligned_stack": str(aligned_path),
                        "xfeat_warp_field": str(warp_path) if "field_x" in cached.files else "",
                    },
                    "metrics": {**cached["metrics"].item(), "cache_hit": True},
                }
        except Exception:
            pass

    source_rgb = _load_rgb(source_path)
    reference_rgb = _load_rgb(reference_path)

    device = _select_device(cfg.device)
    src_small, src_scale = _resize_long_edge_np(source_rgb, cfg.max_long_edge)
    ref_small, ref_scale = _resize_long_edge_np(reference_rgb, cfg.max_long_edge)

    model = _get_model(str(device), int(cfg.top_k), float(cfg.detection_threshold), str(cfg.weights_path))
    src_pts_small, ref_pts_small = model.match_xfeat_star(
        src_small,
        ref_small,
        top_k=int(cfg.top_k),
        multiscale=bool(cfg.multiscale),
    )
    if src_pts_small.shape[0] < cfg.min_inlier_count:
        result = _fallback_alignment(source_rgb, reference_rgb, match_path, "XFeat* fallback: insufficient matches")
        result["metrics"]["cache_hit"] = False
        return result

    src_pts = src_pts_small / max(src_scale, 1e-8)
    ref_pts = ref_pts_small / max(ref_scale, 1e-8)

    actual_ransac_thresh = cfg.ransac_threshold / max(src_scale, 1e-8)
    H, inlier_mask = cv2.findHomography(ref_pts.astype(np.float32), src_pts.astype(np.float32), cv2.RANSAC, actual_ransac_thresh)
    if H is None or inlier_mask is None:
        result = _fallback_alignment(source_rgb, reference_rgb, match_path, "XFeat* fallback: homography failed")
        result["metrics"]["cache_hit"] = False
        return result

    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < cfg.min_inlier_count:
        result = _fallback_alignment(source_rgb, reference_rgb, match_path, "XFeat* fallback: insufficient inliers")
        result["metrics"]["cache_hit"] = False
        return result

    usable, quality = _global_warp_is_usable(H, src_pts, ref_pts, inlier_mask, source_rgb.shape, reference_rgb.shape, src_scale, cfg)
    if not usable:
        result = _fallback_alignment(source_rgb, reference_rgb, match_path, "XFeat* fallback: unstable homography")
        result["metrics"]["cache_hit"] = False
        return result

    F_mat, inlier_mask_F = cv2.findFundamentalMat(ref_pts.astype(np.float32), src_pts.astype(np.float32), cv2.FM_RANSAC, actual_ransac_thresh)
    if F_mat is not None and inlier_mask_F is not None:
        mesh_inlier_mask = inlier_mask_F.reshape(-1).astype(bool)
    else:
        mesh_inlier_mask = inlier_mask.reshape(-1).astype(bool)

    src_h, src_w = source_rgb.shape[:2]
    warped_ref, valid_mask, mesh_stats, field_x, field_y = _warp_reference_locally(
        reference_rgb,
        H,
        src_pts,
        ref_pts,
        mesh_inlier_mask,
        src_h,
        src_w,
        cfg,
    )

    x, y, w, h = _largest_valid_rect(valid_mask)
    if w < 16 or h < 16:
        result = _fallback_alignment(source_rgb, reference_rgb, match_path, "XFeat* fallback: overlap crop too small")
        result["metrics"]["cache_hit"] = False
        return result

    source_crop = source_rgb[y : y + h, x : x + w].copy()
    ref_crop = warped_ref[y : y + h, x : x + w].copy()
    overlay = _draw_match_overlay(source_rgb, reference_rgb, src_pts, ref_pts, mesh_inlier_mask, cfg.max_matches_drawn)
    aligned_stack = _draw_aligned_stack(source_crop, ref_crop)
    fx_crop = field_x[y : y + h, x : x + w]
    fy_crop = field_y[y : y + h, x : x + w]
    warp_vis = _draw_warp_field(fx_crop, fy_crop, bg_rgb=source_crop)
    _save_rgb(match_path, overlay)
    _save_rgb(aligned_path, aligned_stack)
    _save_rgb(warp_path, warp_vis)

    result = {
        "source_rgb": source_crop,
        "reference_rgb": ref_crop,
        "paths": {
            "xfeat_matches": str(match_path),
            "xfeat_aligned_stack": str(aligned_path),
            "xfeat_warp_field": str(warp_path),
        },
        "metrics": {
            "status": "ok",
            "reason": "",
            "match_count": int(src_pts.shape[0]),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_count / max(int(src_pts.shape[0]), 1)),
            "crop_bounds": [int(x), int(y), int(w), int(h)],
            "aligned_shape": [int(h), int(w), 3],
            "homography": H.tolist(),
            "warp_mode": "mesh_local_warp",
            "cache_hit": False,
            **quality,
            **mesh_stats,
        },
    }
    np.savez(
        cache_path,
        meta=cache_key,
        source_rgb=source_crop.astype(np.float16),
        reference_rgb=ref_crop.astype(np.float16),
        overlay_u8=(overlay * 255.0 + 0.5).astype(np.uint8),
        field_x=fx_crop,
        field_y=fy_crop,
        metrics=result["metrics"],
    )
    return result
