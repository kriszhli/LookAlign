"""LightGlue-based pre-alignment for LookAlign V0.4.5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue
from PIL import Image, ImageDraw


Tensor = torch.Tensor


@dataclass
class LightGlueAlignmentConfig:
    max_long_edge: int = 1536
    min_inlier_count: int = 24
    ransac_threshold: float = 3.0
    max_matches_drawn: int = 80
    overlay_height: int = 720
    max_mean_reprojection_error: float = 4.0
    min_overlap_ratio: float = 0.55
    max_corner_shift_ratio: float = 0.35
    device: str = "auto"


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


def _save_rgb(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _to_nchw(img: np.ndarray, device: torch.device) -> Tensor:
    arr = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def _resize_long_edge_np(img: np.ndarray, long_edge: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, float(max(16, int(long_edge))) / float(max(h, w)))
    if scale >= 0.999:
        return img.copy(), 1.0
    th = max(8, int(round(h * scale)))
    tw = max(8, int(round(w * scale)))
    out = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
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
    h = target_h
    w = src_disp.shape[1] + ref_disp.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, : src_disp.shape[1]] = src_disp
    canvas[:, src_disp.shape[1] :] = ref_disp
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)

    if fallback_text:
        draw.rectangle((0, 0, w, 26), fill=(0, 0, 0))
        draw.text((8, 6), fallback_text, fill=(255, 80, 80))
        return np.asarray(img).astype(np.float32) / 255.0

    if src_pts.size == 0 or ref_pts.size == 0:
        draw.rectangle((0, 0, w, 26), fill=(0, 0, 0))
        draw.text((8, 6), "No LightGlue matches", fill=(255, 80, 80))
        return np.asarray(img).astype(np.float32) / 255.0

    order = np.flatnonzero(inliers.astype(bool))
    if order.size == 0:
        order = np.arange(len(src_pts))
    if order.size > max_matches:
        order = order[np.linspace(0, order.size - 1, num=max_matches, dtype=int)]
    offset_x = src_disp.shape[1]
    for idx in order:
        p0_xy = np.round(src_pts[idx] * src_scale).astype(int)
        p1_xy = np.round(ref_pts[idx] * ref_scale).astype(int) + np.array([offset_x, 0])
        p0 = tuple(p0_xy.tolist())
        p1 = tuple(p1_xy.tolist())
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(80, 255, 120), width=1)
        draw.ellipse((p0[0] - 2, p0[1] - 2, p0[0] + 2, p0[1] + 2), fill=(255, 255, 0))
        draw.ellipse((p1[0] - 2, p1[1] - 2, p1[0] + 2, p1[1] + 2), fill=(255, 255, 0))
    return np.asarray(img).astype(np.float32) / 255.0


def _largest_valid_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best = (0, 0, w, h)
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[int] = []
        x = 0
        while x <= w:
            curr = heights[x] if x < w else 0
            if not stack or curr >= heights[stack[-1]]:
                stack.append(x)
                x += 1
                continue
            top = stack.pop()
            width = x if not stack else x - stack[-1] - 1
            area = int(heights[top] * width)
            if area > best_area and heights[top] > 0 and width > 0:
                best_area = area
                x1 = x - width
                y1 = y - heights[top] + 1
                best = (x1, y1, width, int(heights[top]))
        # continue row
    return best


def _fallback_alignment(
    source_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    match_path: Path,
    reason: str,
) -> Dict[str, Any]:
    overlay = _draw_match_overlay(source_rgb, reference_rgb, np.empty((0, 2)), np.empty((0, 2)), np.zeros(0), 0, fallback_text=reason)
    _save_rgb(match_path, overlay)
    return {
        "source_rgb": source_rgb,
        "reference_rgb": reference_rgb,
        "paths": {"lightglue_matches": str(match_path)},
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


def _homography_is_usable(
    H: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: np.ndarray,
    source_shape: tuple[int, int, int],
    cfg: LightGlueAlignmentConfig,
) -> tuple[bool, Dict[str, float]]:
    inlier_src = src_pts[inlier_mask]
    inlier_ref = ref_pts[inlier_mask]
    if inlier_src.shape[0] == 0:
        return False, {"mean_reprojection_error": float("inf"), "overlap_ratio": 0.0, "corner_shift_ratio": 1.0}

    projected = cv2.perspectiveTransform(inlier_ref.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    reproj = np.linalg.norm(projected - inlier_src, axis=1)
    mean_reproj = float(reproj.mean())

    src_h, src_w = source_shape[:2]
    ref_corners = np.array([[0, 0], [0, src_h - 1], [src_w - 1, src_h - 1], [src_w - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(ref_corners, H).reshape(-1, 2)
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    inter_w = max(0.0, min(float(src_w), float(x1)) - max(0.0, float(x0)))
    inter_h = max(0.0, min(float(src_h), float(y1)) - max(0.0, float(y0)))
    overlap_ratio = float((inter_w * inter_h) / max(float(src_w * src_h), 1.0))

    src_corners = ref_corners.reshape(-1, 2)
    corner_shift = np.linalg.norm(warped - src_corners, axis=1).mean()
    corner_shift_ratio = float(corner_shift / max(float((src_w**2 + src_h**2) ** 0.5), 1.0))

    usable = (
        mean_reproj <= cfg.max_mean_reprojection_error
        and overlap_ratio >= cfg.min_overlap_ratio
        and corner_shift_ratio <= cfg.max_corner_shift_ratio
    )
    return usable, {
        "mean_reprojection_error": mean_reproj,
        "overlap_ratio": overlap_ratio,
        "corner_shift_ratio": corner_shift_ratio,
    }


@torch.no_grad()
def run_lightglue_alignment(
    source_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    config: Optional[LightGlueAlignmentConfig | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config if isinstance(config, LightGlueAlignmentConfig) else LightGlueAlignmentConfig(**(config or {}))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    match_path = output_dir / "lightglue_matches.png"

    source_rgb = _load_rgb(source_path)
    reference_rgb = _load_rgb(reference_path)

    device = _select_device(cfg.device)
    src_small, src_scale = _resize_long_edge_np(source_rgb, cfg.max_long_edge)
    ref_small, ref_scale = _resize_long_edge_np(reference_rgb, cfg.max_long_edge)

    extractor = ALIKED().eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)

    src_feats = extractor.extract(_to_nchw(src_small, device))
    ref_feats = extractor.extract(_to_nchw(ref_small, device))
    matches = matcher({"image0": src_feats, "image1": ref_feats})

    pair_idx = matches["matches"][0].detach().cpu().numpy()
    if pair_idx.shape[0] < cfg.min_inlier_count:
        return _fallback_alignment(source_rgb, reference_rgb, match_path, "LightGlue fallback: insufficient matches")

    src_kp = src_feats["keypoints"][0].detach().cpu().numpy()[pair_idx[:, 0]]
    ref_kp = ref_feats["keypoints"][0].detach().cpu().numpy()[pair_idx[:, 1]]
    src_pts = src_kp / max(src_scale, 1e-8)
    ref_pts = ref_kp / max(ref_scale, 1e-8)

    H, inlier_mask = cv2.findHomography(ref_pts.astype(np.float32), src_pts.astype(np.float32), cv2.RANSAC, cfg.ransac_threshold)
    if H is None or inlier_mask is None:
        return _fallback_alignment(source_rgb, reference_rgb, match_path, "LightGlue fallback: homography failed")

    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < cfg.min_inlier_count:
        return _fallback_alignment(source_rgb, reference_rgb, match_path, "LightGlue fallback: insufficient inliers")

    usable, quality = _homography_is_usable(H, src_pts, ref_pts, inlier_mask, source_rgb.shape, cfg)
    if not usable:
        return _fallback_alignment(source_rgb, reference_rgb, match_path, "LightGlue fallback: unstable homography")

    src_h, src_w = source_rgb.shape[:2]
    warped_ref = cv2.warpPerspective(reference_rgb.astype(np.float32), H, (src_w, src_h), flags=cv2.INTER_LINEAR)
    valid_mask = cv2.warpPerspective(
        np.ones(reference_rgb.shape[:2], dtype=np.uint8),
        H,
        (src_w, src_h),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)

    x, y, w, h = _largest_valid_rect(valid_mask)
    if w < 16 or h < 16:
        return _fallback_alignment(source_rgb, reference_rgb, match_path, "LightGlue fallback: overlap crop too small")

    source_crop = source_rgb[y : y + h, x : x + w].copy()
    ref_crop = warped_ref[y : y + h, x : x + w].copy()

    overlay = _draw_match_overlay(source_rgb, reference_rgb, src_pts, ref_pts, inlier_mask, cfg.max_matches_drawn)
    _save_rgb(match_path, overlay)

    return {
        "source_rgb": source_crop,
        "reference_rgb": ref_crop,
        "paths": {"lightglue_matches": str(match_path)},
        "metrics": {
            "status": "ok",
            "reason": "",
            "match_count": int(pair_idx.shape[0]),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_count / max(int(pair_idx.shape[0]), 1)),
            "crop_bounds": [int(x), int(y), int(w), int(h)],
            "aligned_shape": [int(h), int(w), 3],
            "homography": H.tolist(),
            **quality,
        },
    }
