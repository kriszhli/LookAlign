#!/usr/bin/env python3
"""LookAlign MVP: deterministic low-frequency look transfer from reference to source."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only when OpenCV is absent.
    cv2 = None

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - SciPy is expected in the local MVP env.
    ndimage = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - PyTorch is optional for legacy fallback.
    torch = None


EPS = 1e-6
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def finite01(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if not np.isfinite(arr).all():
        raise ValueError(f"Image contains non-finite values after loading: {path}")
    return finite01(arr)


def save_rgb(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = finite01(img)
    h, w = arr.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.int32)
    ordered = ((xx * 13 + yy * 17 + (xx ^ yy) * 3) & 255).astype(np.float32)
    dither = (ordered / 255.0 - 0.5) / 255.0
    out = np.clip(arr + dither[..., None], 0.0, 1.0)
    out = (out * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(path)


def save_gray(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (np.clip(np.nan_to_num(img), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(rgb[..., :3], LUMA, axes=([-1], [0])).astype(np.float32)


def resize_long_edge(gray: np.ndarray, max_edge: int = 900) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    scale = min(1.0, float(max_edge) / float(max(h, w)))
    if scale >= 0.999:
        return gray.astype(np.float32, copy=False), 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if cv2 is not None:
        out = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        out = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return out.astype(np.float32), scale


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr.astype(np.float32, copy=True)
    if cv2 is not None:
        if arr.ndim == 2:
            return cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
        channels = [
            cv2.GaussianBlur(arr[..., c].astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
            for c in range(arr.shape[-1])
        ]
        return np.stack(channels, axis=-1).astype(np.float32)
    if ndimage is None:
        raise RuntimeError("Gaussian blur requires OpenCV or SciPy.")
    if arr.ndim == 2:
        return ndimage.gaussian_filter(arr.astype(np.float32), sigma=sigma, mode="reflect")
    return np.stack(
        [ndimage.gaussian_filter(arr[..., c].astype(np.float32), sigma=sigma, mode="reflect") for c in range(arr.shape[-1])],
        axis=-1,
    ).astype(np.float32)


def mask_aware_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    mask = mask.astype(np.float32)
    den = gaussian_blur(mask, sigma)
    if img.ndim == 2:
        num = gaussian_blur(img.astype(np.float32) * mask, sigma)
        return (num / np.maximum(den, EPS)).astype(np.float32)
    num = gaussian_blur(img.astype(np.float32) * mask[..., None], sigma)
    return (num / np.maximum(den[..., None], EPS)).astype(np.float32)


def sobel_xy(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cv2 is not None:
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return gx, gy
    if ndimage is None:
        raise RuntimeError("Sobel gradients require OpenCV or SciPy.")
    return ndimage.sobel(gray, axis=1, mode="reflect"), ndimage.sobel(gray, axis=0, mode="reflect")


def laplacian(gray: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    if ndimage is None:
        raise RuntimeError("Laplacian requires OpenCV or SciPy.")
    return ndimage.laplace(gray.astype(np.float32), mode="reflect")


def transform_corners(w: int, h: int, matrix: np.ndarray) -> np.ndarray:
    pts = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)
    homo = np.concatenate([pts, np.ones((4, 1), dtype=np.float32)], axis=1)
    return homo @ matrix.T


def rect_intersection_area(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    wh = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    return float(wh[0] * wh[1])


def similarity_scale_rotation(matrix: np.ndarray) -> Tuple[float, float]:
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    sx = math.sqrt(a * a + c * c)
    sy = math.sqrt(b * b + d * d)
    scale = 0.5 * (sx + sy)
    rotation = math.degrees(math.atan2(c, a))
    return scale, rotation


def validate_transform(
    matrix: np.ndarray,
    ref_shape: Tuple[int, int, int],
    src_shape: Tuple[int, int, int],
    scale_min: float,
    scale_max: float,
    min_overlap: float = 0.05,
) -> Tuple[bool, str, float]:
    scale, rotation = similarity_scale_rotation(matrix)
    if not np.isfinite(matrix).all():
        return False, "non-finite alignment transform", 0.0
    if scale < scale_min or scale > scale_max:
        return False, f"alignment scale {scale:.3f} outside [{scale_min:.3f}, {scale_max:.3f}]", 0.0
    if abs(rotation) > 60.0:
        return False, f"alignment rotation {rotation:.1f} degrees is too large", 0.0
    hs, ws = src_shape[:2]
    hr, wr = ref_shape[:2]
    corners = transform_corners(wr, hr, matrix)
    r_min = corners.min(axis=0)
    r_max = corners.max(axis=0)
    area = rect_intersection_area(r_min, r_max, np.array([0.0, 0.0]), np.array([float(ws), float(hs)]))
    overlap_ratio = area / max(1.0, float(ws * hs))
    if overlap_ratio < min_overlap:
        return False, f"alignment overlap ratio {overlap_ratio:.3f} is too small", overlap_ratio
    return True, "", overlap_ratio


def center_transform(src_shape: Tuple[int, int, int], ref_shape: Tuple[int, int, int], scale: float = 1.0) -> np.ndarray:
    hs, ws = src_shape[:2]
    hr, wr = ref_shape[:2]
    tx = 0.5 * (ws - wr * scale)
    ty = 0.5 * (hs - hr * scale)
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)


def scale_fit_transform(src_shape: Tuple[int, int, int], ref_shape: Tuple[int, int, int]) -> np.ndarray:
    hs, ws = src_shape[:2]
    hr, wr = ref_shape[:2]
    scale = min(float(ws) / float(wr), float(hs) / float(hr))
    return center_transform(src_shape, ref_shape, scale=scale)


def estimate_feature_alignment(
    src: np.ndarray,
    ref: np.ndarray,
    scale_min: float,
    scale_max: float,
    warnings: List[str],
) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
    if cv2 is None:
        warnings.append("OpenCV unavailable; feature alignment skipped.")
        return None

    src_l = luminance(src)
    ref_l = luminance(ref)
    src_small, src_factor = resize_long_edge(src_l, 900)
    ref_small, ref_factor = resize_long_edge(ref_l, 900)
    src_u8 = np.clip(src_small * 255.0, 0, 255).astype(np.uint8)
    ref_u8 = np.clip(ref_small * 255.0, 0, 255).astype(np.uint8)

    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8, edgeThreshold=19, patchSize=31, fastThreshold=12)
    kp_s, des_s = orb.detectAndCompute(src_u8, None)
    kp_r, des_r = orb.detectAndCompute(ref_u8, None)
    if des_s is None or des_r is None or len(kp_s) < 12 or len(kp_r) < 12:
        warnings.append("Feature alignment skipped: insufficient ORB features.")
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_r, des_s, k=2)
    matches = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance <= 0.78 * n.distance:
            matches.append(m)
    matches = sorted(matches, key=lambda m: (m.distance, m.queryIdx, m.trainIdx))[:500]
    if len(matches) < 12:
        warnings.append("Feature alignment failed: too few descriptor matches.")
        return None

    pts_r = np.float32([kp_r[m.queryIdx].pt for m in matches])
    pts_s = np.float32([kp_s[m.trainIdx].pt for m in matches])
    mat_small, inlier_mask = cv2.estimateAffinePartial2D(
        pts_r,
        pts_s,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    if mat_small is None or inlier_mask is None:
        warnings.append("Feature alignment failed: RANSAC did not estimate a transform.")
        return None

    inliers = int(inlier_mask.ravel().sum())
    ratio = inliers / max(1, len(matches))
    if inliers < 10 or ratio < 0.18:
        warnings.append(f"Feature alignment rejected: {inliers} inliers, ratio {ratio:.3f}.")
        return None

    mat = mat_small.astype(np.float32).copy()
    mat[:, :2] *= ref_factor / src_factor
    mat[:, 2] /= src_factor
    ok, reason, overlap = validate_transform(mat, ref.shape, src.shape, scale_min, scale_max)
    if not ok:
        warnings.append(f"Feature alignment rejected: {reason}.")
        return None

    confidence = float(np.clip(0.35 + 0.45 * ratio + 0.20 * min(1.0, inliers / 60.0), 0.0, 1.0))
    info = {"feature_matches": len(matches), "feature_inliers": inliers, "feature_inlier_ratio": ratio, "feature_overlap_ratio": overlap}
    return mat, confidence, info


def estimate_phase_translation(src: np.ndarray, ref: np.ndarray, matrix: np.ndarray, warnings: List[str]) -> Tuple[np.ndarray, float]:
    if cv2 is None:
        warnings.append("Phase correlation skipped: OpenCV unavailable.")
        return matrix, 0.0
    hs, ws = src.shape[:2]
    ref_warp, ref_mask = warp_to_shape(ref, np.ones(ref.shape[:2], dtype=np.float32), matrix, (hs, ws))
    if ref_mask.mean() < 0.03:
        warnings.append("Phase correlation skipped: scaled reference overlap is too small.")
        return matrix, 0.0
    src_l = gaussian_blur(luminance(src), 10.0)
    ref_l = mask_aware_blur(luminance(ref_warp), ref_mask, 10.0)
    src_l = (src_l - float(src_l.mean())).astype(np.float32)
    ref_l = ((ref_l - float(ref_l[ref_mask > 0.2].mean())) * ref_mask).astype(np.float32)
    try:
        shift, response = cv2.phaseCorrelate(ref_l, src_l)
    except Exception as exc:
        warnings.append(f"Phase correlation failed: {exc}.")
        return matrix, 0.0
    dx, dy = float(shift[0]), float(shift[1])
    max_shift = 0.35 * max(ws, hs)
    if not np.isfinite([dx, dy, response]).all() or abs(dx) > max_shift or abs(dy) > max_shift or response < 0.04:
        warnings.append(f"Phase correlation rejected: shift=({dx:.1f},{dy:.1f}), response={response:.3f}.")
        return matrix, 0.0
    out = matrix.copy()
    out[0, 2] += dx
    out[1, 2] += dy
    return out.astype(np.float32), float(np.clip(response, 0.0, 1.0))


def rescale_transform_about_ref_center(matrix: np.ndarray, ref_shape: Tuple[int, int, int], scale_multiplier: float) -> np.ndarray:
    hr, wr = ref_shape[:2]
    center = np.array([0.5 * wr, 0.5 * hr], dtype=np.float32)
    mapped_center = matrix[:, :2] @ center + matrix[:, 2]
    out = matrix.astype(np.float32).copy()
    out[:, :2] *= float(scale_multiplier)
    out[:, 2] = mapped_center - out[:, :2] @ center
    return out.astype(np.float32)


def masked_alignment_score(src: np.ndarray, ref: np.ndarray, matrix: np.ndarray) -> float:
    hs, ws = src.shape[:2]
    ref_warp, ref_mask = warp_to_shape(ref, np.ones(ref.shape[:2], dtype=np.float32), matrix, (hs, ws))
    overlap = ref_mask > 0.2
    if int(overlap.sum()) < max(1000, int(0.01 * hs * ws)):
        return -1.0
    src_l = gaussian_blur(luminance(src), 6.0)
    ref_l = gaussian_blur(luminance(ref_warp), 6.0)
    src_vals = src_l[overlap].astype(np.float64)
    ref_vals = ref_l[overlap].astype(np.float64)
    src_vals -= src_vals.mean()
    ref_vals -= ref_vals.mean()
    den = float(np.linalg.norm(src_vals) * np.linalg.norm(ref_vals))
    if den <= EPS:
        return -1.0
    return float(np.dot(src_vals, ref_vals) / den)


def refine_alignment_scale(src: np.ndarray, ref: np.ndarray, matrix: np.ndarray, warnings: List[str]) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    if cv2 is None:
        return matrix, 0.0, {}
    base_score = masked_alignment_score(src, ref, matrix)
    if base_score < -0.5:
        return matrix, 0.0, {}

    best_matrix = matrix.astype(np.float32)
    best_score = base_score
    best_multiplier = 1.0
    candidates = [0.94, 0.97, 0.985, 1.0, 1.015, 1.03, 1.06]
    for multiplier in candidates:
        candidate = rescale_transform_about_ref_center(matrix, ref.shape, multiplier)
        candidate, _ = estimate_phase_translation(src, ref, candidate, warnings=[])
        score = masked_alignment_score(src, ref, candidate)
        if score > best_score + 0.002:
            best_matrix = candidate
            best_score = score
            best_multiplier = float(multiplier)

    if best_multiplier != 1.0:
        warnings.append(f"Alignment scale refined by {best_multiplier:.4f}x.")
    return best_matrix.astype(np.float32), float(np.clip(best_score - base_score, 0.0, 1.0)), {
        "scale_refine_multiplier": best_multiplier,
        "scale_refine_score_before": float(base_score),
        "scale_refine_score_after": float(best_score),
    }


def estimate_alignment(args: argparse.Namespace, src: np.ndarray, ref: np.ndarray, warnings: List[str]) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    if args.align == "identity":
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), "identity", 1.0, {}
    if args.align == "center":
        return center_transform(src.shape, ref.shape), "center", 0.55, {}

    feature = estimate_feature_alignment(src, ref, args.align_scale_min, args.align_scale_max, warnings)
    if feature is not None:
        mat, confidence, info = feature
        mat, refine_gain, refine_info = refine_alignment_scale(src, ref, mat, warnings)
        info.update(refine_info)
        return mat, "auto_feature", float(np.clip(confidence + 0.10 * refine_gain, 0.0, 1.0)), info

    fit = scale_fit_transform(src.shape, ref.shape)
    fit, phase_conf = estimate_phase_translation(src, ref, fit, warnings)
    fit, refine_gain, refine_info = refine_alignment_scale(src, ref, fit, warnings)
    ok, reason, overlap = validate_transform(fit, ref.shape, src.shape, args.align_scale_min, args.align_scale_max, min_overlap=0.03)
    if ok:
        conf = 0.30 + 0.35 * phase_conf + 0.10 * refine_gain
        info = {"fallback_overlap_ratio": overlap, "phase_confidence": phase_conf}
        info.update(refine_info)
        return fit, "auto_scale_fit_phase" if phase_conf > 0 else "auto_scale_fit", conf, info

    warnings.append(f"Scale-fit fallback rejected: {reason}; using centered native-size alignment.")
    return center_transform(src.shape, ref.shape), "auto_center_fallback", 0.15, {}


def warp_to_shape(img: np.ndarray, mask: np.ndarray, matrix: np.ndarray, out_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = out_shape
    if cv2 is not None:
        warped_img = cv2.warpAffine(
            img.astype(np.float32),
            matrix.astype(np.float32),
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            mask.astype(np.float32),
            matrix.astype(np.float32),
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    elif ndimage is not None:
        inv = np.linalg.inv(np.vstack([matrix, [0.0, 0.0, 1.0]]))[:2]
        warped_img = np.zeros((h, w, img.shape[2]), dtype=np.float32)
        for c in range(img.shape[2]):
            warped_img[..., c] = ndimage.affine_transform(
                img[..., c],
                inv[:, :2],
                offset=inv[:, 2],
                output_shape=(h, w),
                order=1,
                mode="constant",
                cval=0.0,
            )
        warped_mask = ndimage.affine_transform(mask, inv[:, :2], offset=inv[:, 2], output_shape=(h, w), order=1, mode="constant", cval=0.0)
    else:
        raise RuntimeError("Warping requires OpenCV or SciPy.")
    return finite01(warped_img), np.clip(warped_mask.astype(np.float32), 0.0, 1.0)


def build_union_canvas(src: np.ndarray, ref: np.ndarray, matrix: np.ndarray) -> Dict[str, Any]:
    hs, ws = src.shape[:2]
    hr, wr = ref.shape[:2]
    ref_corners = transform_corners(wr, hr, matrix)
    all_pts = np.vstack([ref_corners, np.array([[0.0, 0.0], [ws, 0.0], [ws, hs], [0.0, hs]], dtype=np.float32)])
    min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32)
    offset = np.maximum(-min_xy, 0).astype(np.int32)
    width = int(max_xy[0] - min_xy[0])
    height = int(max_xy[1] - min_xy[1])
    width = max(width, ws + int(offset[0]))
    height = max(height, hs + int(offset[1]))

    source_canvas = np.zeros((height, width, 3), dtype=np.float32)
    source_mask = np.zeros((height, width), dtype=np.float32)
    ox, oy = int(offset[0]), int(offset[1])
    source_canvas[oy : oy + hs, ox : ox + ws] = src
    source_mask[oy : oy + hs, ox : ox + ws] = 1.0

    matrix_union = matrix.astype(np.float32).copy()
    matrix_union[0, 2] += ox
    matrix_union[1, 2] += oy
    reference_canvas, reference_mask = warp_to_shape(ref, np.ones((hr, wr), dtype=np.float32), matrix_union, (height, width))
    overlap = ((source_mask > 0.5) & (reference_mask > 0.5)).astype(np.float32)
    return {
        "source_canvas": source_canvas,
        "source_mask": source_mask,
        "reference_canvas": reference_canvas,
        "reference_mask": reference_mask,
        "overlap_mask": overlap,
        "offset": (ox, oy),
        "matrix_union": matrix_union,
    }


def local_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    ma = mask_aware_blur(a, mask, sigma)
    mb = mask_aware_blur(b, mask, sigma)
    va = mask_aware_blur(a * a, mask, sigma) - ma * ma
    vb = mask_aware_blur(b * b, mask, sigma) - mb * mb
    cab = mask_aware_blur(a * b, mask, sigma) - ma * mb
    corr = cab / np.sqrt(np.maximum(va, EPS) * np.maximum(vb, EPS))
    return np.clip((corr + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


def compute_trust(src: np.ndarray, ref: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    src_l = mask_aware_blur(luminance(src), overlap, 1.2)
    ref_l = mask_aware_blur(luminance(ref), overlap, 1.2)
    sx, sy = sobel_xy(src_l)
    rx, ry = sobel_xy(ref_l)
    smag = np.sqrt(sx * sx + sy * sy)
    rmag = np.sqrt(rx * rx + ry * ry)
    dot = sx * rx + sy * ry
    orient = np.clip((dot / np.maximum(smag * rmag, EPS) + 1.0) * 0.5, 0.0, 1.0)
    mag_compat = np.exp(-np.abs(smag - rmag) / (np.maximum(smag, rmag) + 0.035))
    slap = laplacian(src_l)
    rlap = laplacian(ref_l)
    lap_compat = np.exp(-np.abs(slap - rlap) / (np.abs(slap) + np.abs(rlap) + 0.035))
    src_hp = src_l - mask_aware_blur(src_l, overlap, 5.0)
    ref_hp = ref_l - mask_aware_blur(ref_l, overlap, 5.0)
    hp_corr = local_correlation(src_hp, ref_hp, overlap, 7.0)
    lum_compat = np.exp(-np.abs(mask_aware_blur(src_l, overlap, 12.0) - mask_aware_blur(ref_l, overlap, 12.0)) / 0.50)

    edge_ref = float(np.percentile((smag + rmag)[overlap > 0.5], 95)) if np.any(overlap > 0.5) else 0.0
    edge_weight = np.clip((smag + rmag) / max(edge_ref, 0.03), 0.0, 1.0)
    structural = 0.40 * orient + 0.28 * mag_compat + 0.22 * lap_compat + 0.10 * hp_corr
    flat = 0.65 * hp_corr + 0.25 * structural + 0.10 * lum_compat
    trust = ((1.0 - edge_weight) * flat + edge_weight * structural) * (0.88 + 0.12 * lum_compat)
    trust *= overlap
    trust = gaussian_blur(trust.astype(np.float32), 1.5)
    trust *= overlap
    return np.clip(trust, 0.0, 1.0).astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> Tuple[np.ndarray, str]:
    if cv2 is None:
        return rgb.astype(np.float32, copy=True), "rgb"
    lab = cv2.cvtColor(finite01(rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] /= 100.0
    lab[..., 1] = (lab[..., 1] + 128.0) / 255.0
    lab[..., 2] = (lab[..., 2] + 128.0) / 255.0
    return lab, "lab"


def lab_to_rgb(lab_norm: np.ndarray, space: str) -> np.ndarray:
    if space == "rgb" or cv2 is None:
        return finite01(lab_norm)
    lab = lab_norm.astype(np.float32).copy()
    lab[..., 0] *= 100.0
    lab[..., 1] = lab[..., 1] * 255.0 - 128.0
    lab[..., 2] = lab[..., 2] * 255.0 - 128.0
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return finite01(rgb)


def chroma_delta_metrics(src_lab: np.ndarray, out_lab: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    valid = mask > 0.5
    if not np.any(valid):
        return {
            "mean_delta_lab": [0.0, 0.0, 0.0],
            "max_delta_lab": [0.0, 0.0, 0.0],
            "neutral_region_avg_ab_shift": [0.0, 0.0],
        }
    delta = out_lab - src_lab
    mean_delta = [float(delta[..., c][valid].mean()) for c in range(3)]
    max_delta = [float(np.max(np.abs(delta[..., c][valid]))) for c in range(3)]
    src_sat = src_lab[..., 1:3] - 0.5
    neutral = valid & (np.linalg.norm(src_sat, axis=-1) < 0.055)
    if np.any(neutral):
        neutral_shift = [float(delta[..., 1][neutral].mean()), float(delta[..., 2][neutral].mean())]
    else:
        neutral_shift = [0.0, 0.0]
    return {
        "mean_delta_lab": mean_delta,
        "max_delta_lab": max_delta,
        "neutral_region_avg_ab_shift": neutral_shift,
    }


def choose_sa_lut_device(prefer_mps: bool = True) -> Tuple[str, Dict[str, Any]]:
    if torch is None:
        return "none", {"torch_available": False, "mps_built": False, "mps_available": False}
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    device = "mps" if prefer_mps and mps_available else "cpu"
    return device, {"torch_available": True, "mps_built": mps_built, "mps_available": mps_available}


def resolve_render_backend(render_backend: str, warnings: List[str]) -> Tuple[str, Optional[str]]:
    requested = str(render_backend or "auto").lower()
    if requested in ("numpy", "opencv", "torch", "pytorch"):
        return "pytorch", None
    if requested in ("coreimage", "core_image", "accelerate"):
        try:
            __import__("objc")
            __import__("CoreImage")
            return requested, None
        except Exception:
            reason = "Core Image/Accelerate Python bindings are unavailable; using PyTorch/NumPy rendering."
            warnings.append(reason)
            return "pytorch", reason
    if requested == "auto":
        try:
            __import__("objc")
            __import__("CoreImage")
            return "coreimage", None
        except Exception:
            return "pytorch", "Core Image/Accelerate Python bindings are unavailable; using PyTorch/NumPy rendering."
    reason = f"Unknown render_backend '{render_backend}'; using PyTorch/NumPy rendering."
    warnings.append(reason)
    return "pytorch", reason


def identity_sa_lut(size: int, context_bins: int) -> np.ndarray:
    return np.zeros((int(context_bins), int(size), int(size), int(size), 3), dtype=np.float32)


def make_edge_aware_base(
    img: np.ndarray,
    mask: np.ndarray,
    radius: float,
    eps: float,
    base_filter: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mode = str(base_filter or "guided_auto").lower()
    radius = float(max(radius, 0.0))
    if radius <= 0:
        return finite01(img * mask[..., None]), {"backend": "identity", "radius": radius}

    if cv2 is not None and mode in ("guided_auto", "bilateral", "edge_aware"):
        sigma_space = max(1.0, radius)
        sigma_color = float(np.clip(math.sqrt(max(eps, EPS)) * 4.0, 0.04, 0.35))
        masked = finite01(img) * mask[..., None]
        den = cv2.bilateralFilter(mask.astype(np.float32), 0, sigmaColor=1.0, sigmaSpace=sigma_space)
        channels = []
        for c in range(3):
            num = cv2.bilateralFilter(masked[..., c].astype(np.float32), 0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            channels.append(num / np.maximum(den, EPS))
        return finite01(np.stack(channels, axis=-1)), {
            "backend": "opencv_bilateral",
            "radius": radius,
            "eps": float(eps),
            "sigma_color": sigma_color,
        }

    return mask_aware_blur(img, mask, radius), {"backend": "mask_aware_gaussian", "radius": radius, "eps": float(eps)}


def generate_sa_lut_context_map(
    src_low: np.ndarray,
    ref_low: np.ndarray,
    weights: np.ndarray,
    source_mask: np.ndarray,
    device_name: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if torch is None or device_name == "none":
        luma = luminance(src_low)
        context = (luma - float(luma[source_mask > 0.5].min())) / max(float(np.ptp(luma[source_mask > 0.5])), EPS) if np.any(source_mask > 0.5) else luma
        return np.clip(context, 0.0, 1.0).astype(np.float32), {"device": "numpy_fallback"}

    device = torch.device(device_name)
    src_t = torch.from_numpy(finite01(src_low)).to(device=device, dtype=torch.float32)
    ref_t = torch.from_numpy(finite01(ref_low)).to(device=device, dtype=torch.float32)
    weights_t = torch.from_numpy(np.clip(weights, 0.0, 1.0).astype(np.float32)).to(device=device)
    mask_t = torch.from_numpy(np.clip(source_mask, 0.0, 1.0).astype(np.float32)).to(device=device)
    luma_w = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=torch.float32)
    src_l = (src_t * luma_w).sum(dim=-1)
    ref_l = (ref_t * luma_w).sum(dim=-1)
    src_sat = src_t.max(dim=-1).values - src_t.min(dim=-1).values
    ref_sat = ref_t.max(dim=-1).values - ref_t.min(dim=-1).values
    valid = mask_t > 0.5
    if bool(valid.any().item()):
        src_l_norm = (src_l - src_l[valid].min()) / torch.clamp(src_l[valid].max() - src_l[valid].min(), min=EPS)
        sat_norm = (src_sat - src_sat[valid].min()) / torch.clamp(src_sat[valid].max() - src_sat[valid].min(), min=EPS)
    else:
        src_l_norm = src_l
        sat_norm = src_sat
    look_delta = torch.clamp(torch.abs(ref_l - src_l) * 1.7 + torch.abs(ref_sat - src_sat) * 0.8, 0.0, 1.0)
    context = torch.clamp(0.55 * src_l_norm + 0.25 * sat_norm + 0.20 * look_delta, 0.0, 1.0)
    context = context * torch.clamp(0.65 + 0.35 * weights_t, 0.0, 1.0)
    return context.detach().cpu().numpy().astype(np.float32), {"device": device_name}


def fit_sa_lut(
    src_low: np.ndarray,
    ref_low: np.ndarray,
    context_map: np.ndarray,
    weights: np.ndarray,
    size: int,
    context_bins: int,
    max_samples: int,
    ridge: float,
    smooth: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    size = max(2, int(size))
    context_bins = max(2, int(context_bins))
    identity = identity_sa_lut(size, context_bins)
    valid = np.flatnonzero((weights.reshape(-1) > 0.0) & np.isfinite(src_low.reshape(-1, 3)).all(axis=1) & np.isfinite(ref_low.reshape(-1, 3)).all(axis=1))
    if valid.size < 100:
        return identity, {"fit_sample_count": int(valid.size), "enabled": False, "fallback_reason": "insufficient trusted samples"}

    if max_samples > 0 and valid.size > int(max_samples):
        order = np.argsort(weights.reshape(-1)[valid])
        valid = valid[order[-int(max_samples) :]]

    src_flat = finite01(src_low.reshape(-1, 3)[valid])
    src_lab_full, _ = rgb_to_lab(src_low)
    ref_lab_full, _ = rgb_to_lab(ref_low)
    src_lab_flat = src_lab_full.reshape(-1, 3)[valid]
    ref_lab_flat = ref_lab_full.reshape(-1, 3)[valid]
    ctx_flat = np.clip(context_map.reshape(-1)[valid], 0.0, 1.0)
    w_flat = np.clip(weights.reshape(-1)[valid], 0.0, 1.0).astype(np.float32)
    delta_lab = (ref_lab_flat - src_lab_flat).astype(np.float32)
    delta_lab[..., 0] = np.clip(delta_lab[..., 0], -0.22, 0.22)
    delta_lab[..., 1:] = np.clip(delta_lab[..., 1:], -0.22, 0.22)

    delta_sum = np.zeros((context_bins, size, size, size, 3), dtype=np.float32)
    weight_sum = np.zeros((context_bins, size, size, size), dtype=np.float32)
    color_pos = src_flat * float(size - 1)
    c0 = np.floor(color_pos).astype(np.int32)
    cf = (color_pos - c0).astype(np.float32)
    c0 = np.clip(c0, 0, size - 1)
    c1 = np.clip(c0 + 1, 0, size - 1)
    ctx_pos = ctx_flat * float(context_bins - 1)
    k0 = np.floor(ctx_pos).astype(np.int32)
    kf = (ctx_pos - k0).astype(np.float32)
    k0 = np.clip(k0, 0, context_bins - 1)
    k1 = np.clip(k0 + 1, 0, context_bins - 1)

    for kr, kw in ((k0, 1.0 - kf), (k1, kf)):
        for ri, rw in ((c0[:, 0], 1.0 - cf[:, 0]), (c1[:, 0], cf[:, 0])):
            for gi, gw in ((c0[:, 1], 1.0 - cf[:, 1]), (c1[:, 1], cf[:, 1])):
                for bi, bw in ((c0[:, 2], 1.0 - cf[:, 2]), (c1[:, 2], cf[:, 2])):
                    ww = (w_flat * kw * rw * gw * bw).astype(np.float32)
                    np.add.at(delta_sum, (kr, ri, gi, bi, slice(None)), delta_lab * ww[:, None])
                    np.add.at(weight_sum, (kr, ri, gi, bi), ww)

    denom = weight_sum[..., None] + float(max(ridge, EPS))
    fitted_delta_lab = delta_sum / denom
    smooth_w = weight_sum.copy()
    if smooth > 0.0:
        if ndimage is not None:
            sigma = (0.0, float(smooth), float(smooth), float(smooth), 0.0)
            smooth_w = ndimage.gaussian_filter(weight_sum, sigma=(0.0, float(smooth), float(smooth), float(smooth)), mode="nearest")
            smooth_delta = ndimage.gaussian_filter(fitted_delta_lab * weight_sum[..., None], sigma=sigma, mode="nearest")
            smooth_delta = smooth_delta / np.maximum(smooth_w[..., None], float(max(ridge, EPS)))
        elif cv2 is not None:
            smooth_delta = fitted_delta_lab.copy()
            smooth_w = weight_sum.copy()
            for k in range(context_bins):
                smooth_w[k] = cv2.GaussianBlur(weight_sum[k].astype(np.float32), (0, 0), sigmaX=float(smooth))
                for c in range(3):
                    for z in range(size):
                        smooth_num = cv2.GaussianBlur((fitted_delta_lab[k, z, ..., c] * weight_sum[k, z]).astype(np.float32), (0, 0), sigmaX=float(smooth))
                        smooth_delta[k, z, ..., c] = smooth_num / np.maximum(smooth_w[k, z], float(max(ridge, EPS)))
        else:
            smooth_delta = fitted_delta_lab
    else:
        smooth_delta = fitted_delta_lab

    support_ref = max(float(np.percentile(smooth_w[smooth_w > 0], 75)) if np.any(smooth_w > 0) else 1.0, EPS)
    fill_mix = np.clip(smooth_w / support_ref, 0.0, 1.0)[..., None]
    lut_delta_lab = fitted_delta_lab * fill_mix + smooth_delta * (1.0 - fill_mix)
    lut_delta_lab[..., 0] = np.clip(lut_delta_lab[..., 0], -0.22, 0.22)
    lut_delta_lab[..., 1:] = np.clip(lut_delta_lab[..., 1:], -0.22, 0.22)
    confidence = np.clip(weight_sum[..., None] / max(float(np.percentile(weight_sum[weight_sum > 0], 75)) if np.any(weight_sum > 0) else 1.0, EPS), 0.0, 1.0)
    return lut_delta_lab.astype(np.float32), {
        "fit_sample_count": int(valid.size),
        "enabled": True,
        "filled_cell_count": int(np.count_nonzero(weight_sum > 0.0)),
        "active_context_min": float(ctx_flat.min()),
        "active_context_max": float(ctx_flat.max()),
        "lut_payload": "lab_delta",
        "lut_encoding": "identity_rgb",
    }


def apply_sa_lut_torch(src_canvas: np.ndarray, context_map: np.ndarray, lut: np.ndarray, source_mask: np.ndarray, device_name: str) -> np.ndarray:
    if torch is None or device_name == "none":
        return src_canvas.copy()
    device = torch.device(device_name)
    rgb = torch.from_numpy(finite01(src_canvas)).to(device=device, dtype=torch.float32)
    context = torch.from_numpy(np.clip(context_map, 0.0, 1.0).astype(np.float32)).to(device=device)
    mask = torch.from_numpy(np.clip(source_mask, 0.0, 1.0).astype(np.float32)).to(device=device)
    lut_t = torch.from_numpy(lut.astype(np.float32)).to(device=device, dtype=torch.float32)
    context_bins, size = int(lut_t.shape[0]), int(lut_t.shape[1])

    flat = rgb.reshape(-1, 3)
    ctx = context.reshape(-1)
    pos = flat * float(size - 1)
    p0 = torch.floor(pos).to(torch.long)
    pf = pos - p0.to(torch.float32)
    p0 = torch.clamp(p0, 0, size - 1)
    p1 = torch.clamp(p0 + 1, 0, size - 1)
    cpos = ctx * float(context_bins - 1)
    k0 = torch.clamp(torch.floor(cpos).to(torch.long), 0, context_bins - 1)
    k1 = torch.clamp(k0 + 1, 0, context_bins - 1)
    kf = cpos - k0.to(torch.float32)

    def interp_context(k: Any) -> Any:
        out = torch.zeros((flat.shape[0], 3), device=device, dtype=torch.float32)
        for ri, rw in ((p0[:, 0], 1.0 - pf[:, 0]), (p1[:, 0], pf[:, 0])):
            for gi, gw in ((p0[:, 1], 1.0 - pf[:, 1]), (p1[:, 1], pf[:, 1])):
                for bi, bw in ((p0[:, 2], 1.0 - pf[:, 2]), (p1[:, 2], pf[:, 2])):
                    out = out + lut_t[k, ri, gi, bi] * (rw * gw * bw).unsqueeze(-1)
        return out

    out0 = interp_context(k0)
    out1 = interp_context(k1)
    delta_lab = (out0 * (1.0 - kf).unsqueeze(-1) + out1 * kf.unsqueeze(-1)).reshape(src_canvas.shape)
    src_lab = torch_rgb_to_lab_norm(rgb)
    out_lab = src_lab + delta_lab
    out_lab[..., 0] = torch.clamp(out_lab[..., 0], 0.0, 1.0)
    out_lab[..., 1:3] = torch.clamp(out_lab[..., 1:3], 0.0, 1.0)
    out = torch_lab_norm_to_rgb(out_lab)
    out = out * mask.unsqueeze(-1) + rgb * (1.0 - mask.unsqueeze(-1))
    return finite01(out.detach().cpu().numpy())


def sa_lut_global_color_transfer(
    src_canvas: np.ndarray,
    src_base: np.ndarray,
    ref_base: np.ndarray,
    source_mask: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
    warnings: List[str],
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    device_name, device_info = choose_sa_lut_device(prefer_mps=True)
    selected_backend, backend_reason = resolve_render_backend(getattr(args, "render_backend", "auto"), warnings)
    if torch is None:
        warnings.append("PyTorch unavailable; SA-LUT global base uses source-preserving identity.")
        params = {
            "method": "sa_lut",
            "enabled": False,
            "fallback_reason": "torch unavailable",
            "selected_device": "none",
            "device_info": device_info,
            "render_backend": selected_backend,
            "render_backend_fallback_reason": backend_reason,
        }
        return src_canvas.copy(), params, np.zeros(source_mask.shape, dtype=np.float32), src_base.copy()

    try:
        context_map, context_info = generate_sa_lut_context_map(src_base, ref_base, weights, source_mask, device_name)
    except Exception as exc:
        if device_name != "mps":
            raise
        warnings.append(f"SA-LUT MPS context generation failed ({exc}); retrying on CPU.")
        device_name = "cpu"
        context_map, context_info = generate_sa_lut_context_map(src_base, ref_base, weights, source_mask, device_name)
    lut, fit_info = fit_sa_lut(
        src_base,
        ref_base,
        context_map,
        weights,
        getattr(args, "sa_lut_size", 33),
        getattr(args, "sa_lut_context_bins", 2),
        getattr(args, "sa_lut_fit_max_samples", 250000),
        getattr(args, "sa_lut_ridge", 0.035),
        getattr(args, "sa_lut_smooth", 0.75),
    )
    if not fit_info.get("enabled", False):
        reason = fit_info.get("fallback_reason", "insufficient trusted samples")
        warnings.append(f"SA-LUT fitting skipped ({reason}); using source-preserving identity.")
        params = {
            "method": "sa_lut",
            "enabled": False,
            "fallback_reason": reason,
            "selected_device": device_name,
            "device_info": device_info,
            "render_backend": selected_backend,
            "render_backend_fallback_reason": backend_reason,
            "lut_size": int(getattr(args, "sa_lut_size", 33)),
            "context_bins": int(getattr(args, "sa_lut_context_bins", 2)),
            "fit_sample_count": fit_info.get("fit_sample_count", 0),
            "context_generator": context_info,
        }
        return src_canvas.copy(), params, context_map, src_base.copy()

    try:
        result = apply_sa_lut_torch(src_canvas, context_map, lut, source_mask, device_name)
    except Exception as exc:
        if device_name != "mps":
            raise
        warnings.append(f"SA-LUT MPS LUT application failed ({exc}); retrying on CPU.")
        device_name = "cpu"
        result = apply_sa_lut_torch(src_canvas, context_map, lut, source_mask, device_name)
    base_result = apply_sa_lut_torch(src_base, context_map, lut, source_mask, device_name)
    params = {
        "method": "sa_lut",
        "enabled": True,
        "selected_device": device_name,
        "device_info": device_info,
        "render_backend": selected_backend,
        "render_backend_fallback_reason": backend_reason,
        "lut_size": int(lut.shape[1]),
        "context_bins": int(lut.shape[0]),
        "fit_sample_count": fit_info["fit_sample_count"],
        "filled_cell_count": fit_info.get("filled_cell_count", 0),
        "active_context_min": fit_info.get("active_context_min", float(context_map.min())),
        "active_context_max": fit_info.get("active_context_max", float(context_map.max())),
        "context_generator": context_info,
        "lut_payload": fit_info.get("lut_payload", "lab_delta"),
        "lut_encoding": fit_info.get("lut_encoding", "identity_rgb"),
    }
    return finite01(result), params, context_map, finite01(base_result)


def smooth_weighted_grid(values: np.ndarray, confidence: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.astype(np.float32, copy=True)
    if ndimage is not None:
        den = ndimage.gaussian_filter(confidence.astype(np.float32), sigma=sigma, mode="nearest")
        if values.ndim == 2:
            num = ndimage.gaussian_filter(values.astype(np.float32) * confidence, sigma=sigma, mode="nearest")
            return (num / np.maximum(den, EPS)).astype(np.float32)
        out = np.zeros_like(values, dtype=np.float32)
        for c in range(values.shape[-1]):
            num = ndimage.gaussian_filter(values[..., c].astype(np.float32) * confidence, sigma=sigma, mode="nearest")
            out[..., c] = num / np.maximum(den, EPS)
        return out
    if cv2 is not None:
        den = cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), sigmaX=sigma)
        if values.ndim == 2:
            num = cv2.GaussianBlur(values.astype(np.float32) * confidence, (0, 0), sigmaX=sigma)
            return (num / np.maximum(den, EPS)).astype(np.float32)
        num = cv2.GaussianBlur(values.astype(np.float32) * confidence[..., None], (0, 0), sigmaX=sigma)
        return (num / np.maximum(den[..., None], EPS)).astype(np.float32)
    return values.astype(np.float32, copy=True)


def resize_map(arr: np.ndarray, shape: Tuple[int, int], interpolation: int = 3) -> np.ndarray:
    h, w = shape
    if cv2 is not None:
        interp = cv2.INTER_LINEAR if interpolation == 1 else cv2.INTER_CUBIC
        return cv2.resize(arr.astype(np.float32), (w, h), interpolation=interp).astype(np.float32)
    if ndimage is None:
        raise RuntimeError("Map resizing requires OpenCV or SciPy.")
    zoom_y = h / arr.shape[0]
    zoom_x = w / arr.shape[1]
    if arr.ndim == 2:
        return ndimage.zoom(arr.astype(np.float32), (zoom_y, zoom_x), order=interpolation)[:h, :w].astype(np.float32)
    return ndimage.zoom(arr.astype(np.float32), (zoom_y, zoom_x, 1), order=interpolation)[:h, :w].astype(np.float32)


def scalar_map_visual(value: np.ndarray, center: float, span: float) -> np.ndarray:
    return np.clip(0.5 + (value.astype(np.float32) - center) / max(span, EPS), 0.0, 1.0).astype(np.float32)


def guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: float, eps: float, mask: np.ndarray) -> np.ndarray:
    if cv2 is None or radius <= 0:
        return src.astype(np.float32, copy=True)
    r = max(1, int(round(radius)))
    ksize = (2 * r + 1, 2 * r + 1)
    guide = np.clip(guide.astype(np.float32), 0.0, 1.0)
    src = src.astype(np.float32)
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x.astype(np.float32), -1, ksize, normalize=False, borderType=cv2.BORDER_REFLECT101)

    den = np.maximum(box(mask), EPS)
    mean_i = box(guide * mask) / den
    mean_p = box(src * mask) / den
    corr_i = box(guide * guide * mask) / den
    corr_ip = box(guide * src * mask) / den
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(max(eps, EPS)))
    b = mean_p - a * mean_i
    mean_a = box(a * mask) / den
    mean_b = box(b * mask) / den
    return (mean_a * guide + mean_b).astype(np.float32)


def compute_light_map(
    sa_lut_base_result: np.ndarray,
    sa_lut_result: np.ndarray,
    reference_base: np.ndarray,
    source_mask: np.ndarray,
    trusted_weights: np.ndarray,
    light_map_grid: int,
    light_map_smooth: float,
    light_map_radius: float,
    max_exposure_delta: float,
    max_contrast_gain: float,
    guide_eps: float,
    upsample_mode: str,
    warnings: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    h, w = source_mask.shape
    if trusted_weights.sum() < 100.0:
        warnings.append("Light map skipped: insufficient trusted support.")
        maps = {
            "light_map": np.zeros((h, w), dtype=np.float32),
            "coarse_light_map": np.zeros((1, 1), dtype=np.float32),
            "contrast_gain": np.ones((h, w), dtype=np.float32),
            "confidence": np.zeros((h, w), dtype=np.float32),
            "coarse_confidence": np.zeros((1, 1), dtype=np.float32),
            "edge_gate": np.ones((h, w), dtype=np.float32),
        }
        return maps, {"enabled": False, "reason": "insufficient trusted support"}

    base_lab, space = rgb_to_lab(sa_lut_base_result)
    ref_lab, _ = rgb_to_lab(reference_base)
    gy = max(2, int(light_map_grid))
    gx = max(2, int(round(light_map_grid * w / max(h, 1))))
    light_grid = np.zeros((gy, gx), dtype=np.float32)
    contrast_grid = np.ones((gy, gx), dtype=np.float32)
    conf_grid = np.zeros((gy, gx), dtype=np.float32)

    for iy in range(gy):
        y0 = int(round(iy * h / gy))
        y1 = int(round((iy + 1) * h / gy))
        for ix in range(gx):
            x0 = int(round(ix * w / gx))
            x1 = int(round((ix + 1) * w / gx))
            ww = trusted_weights[y0:y1, x0:x1]
            total = float(ww.sum())
            if total < 12.0:
                continue
            src_l = base_lab[y0:y1, x0:x1, 0].astype(np.float64)
            ref_l = ref_lab[y0:y1, x0:x1, 0].astype(np.float64)
            w64 = ww.astype(np.float64)
            mean_src = float((src_l * w64).sum() / max(total, EPS))
            mean_ref = float((ref_l * w64).sum() / max(total, EPS))
            var_src = float(((src_l - mean_src) ** 2 * w64).sum() / max(total, EPS))
            cov = float(((src_l - mean_src) * (ref_l - mean_ref) * w64).sum() / max(total, EPS))
            regression_gain = cov / max(var_src, 2e-4)
            mean_gain = mean_ref / max(mean_src, 0.03)
            gain = float(np.clip(0.35 * regression_gain + 0.65 * mean_gain, 1.0 / max_contrast_gain, max_contrast_gain))
            target_mean = 0.5 + (mean_src - 0.5) * gain
            exposure_delta = float(np.clip(mean_ref - target_mean, -max_exposure_delta, max_exposure_delta))
            light_grid[iy, ix] = exposure_delta
            contrast_grid[iy, ix] = gain
            conf_grid[iy, ix] = min(1.0, total / max(50.0, float((y1 - y0) * (x1 - x0)) * 0.08))

    if float(conf_grid.sum()) < 1.0:
        warnings.append("Light map skipped: no reliable grid cells.")
        maps = {
            "light_map": np.zeros((h, w), dtype=np.float32),
            "coarse_light_map": light_grid,
            "contrast_gain": np.ones((h, w), dtype=np.float32),
            "confidence": np.zeros((h, w), dtype=np.float32),
            "coarse_confidence": conf_grid,
            "edge_gate": np.ones((h, w), dtype=np.float32),
        }
        return maps, {"enabled": False, "reason": "no reliable grid cells"}

    smooth_sigma = max(0.0, float(light_map_smooth))
    light_grid = np.clip(smooth_weighted_grid(light_grid, conf_grid, smooth_sigma), -max_exposure_delta, max_exposure_delta)
    contrast_grid = np.clip(smooth_weighted_grid(contrast_grid, conf_grid, smooth_sigma), 1.0 / max_contrast_gain, max_contrast_gain)
    conf_grid = np.clip(smooth_weighted_grid(conf_grid, conf_grid, smooth_sigma), 0.0, 1.0)

    light_linear = np.clip(resize_map(light_grid, (h, w), interpolation=1), -max_exposure_delta, max_exposure_delta)
    contrast_linear = np.clip(resize_map(contrast_grid, (h, w), interpolation=1), 1.0 / max_contrast_gain, max_contrast_gain)
    confidence = np.clip(resize_map(conf_grid, (h, w), interpolation=1), 0.0, 1.0)
    if str(upsample_mode or "source_guided") == "source_guided":
        guide = luminance(sa_lut_base_result)
        guided_mask = np.clip(source_mask * np.maximum(confidence, 0.15), 0.0, 1.0)
        light_map = guided_filter_gray(guide, light_linear, light_map_radius, guide_eps, guided_mask)
        contrast_gain = guided_filter_gray(guide, contrast_linear, light_map_radius, guide_eps, guided_mask)
    else:
        light_map = light_linear
        contrast_gain = contrast_linear
    light_map = np.clip(light_map, -max_exposure_delta, max_exposure_delta).astype(np.float32)
    contrast_gain = np.clip(contrast_gain, 1.0 / max_contrast_gain, max_contrast_gain).astype(np.float32)

    lum = luminance(sa_lut_base_result)
    sx, sy = sobel_xy(mask_aware_blur(lum, source_mask, 1.0))
    edge = np.sqrt(sx * sx + sy * sy)
    edge_ref = float(np.percentile(edge[source_mask > 0.5], 95)) if np.any(source_mask > 0.5) else 0.0
    edge_gate = 1.0 - 0.25 * np.clip(edge / max(edge_ref, 0.03), 0.0, 1.0)
    confidence = confidence * edge_gate * source_mask

    maps = {
        "light_map": light_map.astype(np.float32),
        "light_map_linear": light_linear.astype(np.float32),
        "coarse_light_map": light_grid.astype(np.float32),
        "contrast_gain": contrast_gain.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "coarse_confidence": conf_grid.astype(np.float32),
        "edge_gate": edge_gate.astype(np.float32),
    }
    stats = {
        "enabled": True,
        "space": space,
        "grid": [gy, gx],
        "upsample": str(upsample_mode or "source_guided"),
        "light_map_radius": float(light_map_radius),
        "confidence_mean": float(confidence[source_mask > 0.5].mean()) if np.any(source_mask > 0.5) else 0.0,
        "confidence_max": float(confidence.max()),
        "light_map_min": float(light_map.min()),
        "light_map_max": float(light_map.max()),
        "contrast_gain_min": float(contrast_gain.min()),
        "contrast_gain_max": float(contrast_gain.max()),
    }
    return maps, stats


def torch_rgb_to_lab_norm(rgb: Any) -> Any:
    rgb = torch.clamp(rgb, 0.0, 1.0)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, torch.pow((rgb + 0.055) / 1.055, 2.4))
    x = linear[..., 0] * 0.4124564 + linear[..., 1] * 0.3575761 + linear[..., 2] * 0.1804375
    y = linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522 + linear[..., 2] * 0.0721750
    z = linear[..., 0] * 0.0193339 + linear[..., 1] * 0.1191920 + linear[..., 2] * 0.9503041
    xyz = torch.stack([x / 0.95047, y, z / 1.08883], dim=-1)
    delta = 6.0 / 29.0
    f = torch.where(xyz > delta**3, torch.pow(torch.clamp(xyz, min=EPS), 1.0 / 3.0), xyz / (3.0 * delta * delta) + 4.0 / 29.0)
    l = (116.0 * f[..., 1] - 16.0) / 100.0
    a = (500.0 * (f[..., 0] - f[..., 1]) + 128.0) / 255.0
    b = (200.0 * (f[..., 1] - f[..., 2]) + 128.0) / 255.0
    return torch.stack([l, a, b], dim=-1)


def torch_lab_norm_to_rgb(lab_norm: Any) -> Any:
    l = lab_norm[..., 0] * 100.0
    a = lab_norm[..., 1] * 255.0 - 128.0
    b = lab_norm[..., 2] * 255.0 - 128.0
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def finv(t: Any) -> Any:
        return torch.where(t > delta, t * t * t, 3.0 * delta * delta * (t - 4.0 / 29.0))

    x = finv(fx) * 0.95047
    y = finv(fy)
    z = finv(fz) * 1.08883
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    bl = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
    linear = torch.stack([r, g, bl], dim=-1)
    srgb = torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * torch.pow(torch.clamp(linear, min=0.0), 1.0 / 2.4) - 0.055)
    return torch.clamp(srgb, 0.0, 1.0)


def apply_light_map_torch(
    sa_lut_result: np.ndarray,
    source_mask: np.ndarray,
    maps: Dict[str, np.ndarray],
    local_strength: float,
    local_luma_strength: float,
    device_name: str,
) -> Tuple[np.ndarray, str]:
    if torch is None:
        return sa_lut_result.copy(), "none"
    if device_name == "none":
        device_name = "cpu"
    device = torch.device(device_name)
    rgb = torch.from_numpy(finite01(sa_lut_result)).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.clip(source_mask, 0.0, 1.0).astype(np.float32)).to(device=device)
    light_map = torch.from_numpy(maps["light_map"].astype(np.float32)).to(device=device)
    contrast_gain = torch.from_numpy(maps["contrast_gain"].astype(np.float32)).to(device=device)
    conf = torch.from_numpy(np.clip(maps["confidence"], 0.0, 1.0).astype(np.float32)).to(device=device)
    blend = torch.clamp(conf * float(np.clip(local_strength, 0.0, 1.0)) * mask, 0.0, 1.0)
    luma_strength = float(np.clip(local_luma_strength, 0.0, 1.0))

    lab = torch_rgb_to_lab_norm(rgb)
    adjusted = lab.clone()
    centered_l = lab[..., 0] - 0.5
    adjusted[..., 0] = torch.clamp(0.5 + centered_l * (1.0 + (contrast_gain - 1.0) * luma_strength) + light_map * luma_strength, 0.0, 1.0)
    out_lab = lab * (1.0 - blend.unsqueeze(-1)) + adjusted * blend.unsqueeze(-1)
    out = torch_lab_norm_to_rgb(out_lab)
    out = out * mask.unsqueeze(-1) + rgb * (1.0 - mask.unsqueeze(-1))
    return finite01(out.detach().cpu().numpy()), device_name


def apply_detail_preservation(src: np.ndarray, transferred: np.ndarray, mask: np.ndarray, base_radius: float, detail_strength: float) -> np.ndarray:
    src_l = luminance(src)
    tr_l = luminance(transferred)
    detail_sigma = max(1.0, base_radius * 0.33)
    src_base = mask_aware_blur(src_l, mask, detail_sigma)
    tr_base = mask_aware_blur(tr_l, mask, detail_sigma)
    source_detail_ratio = np.clip((src_l + 0.025) / (src_base + 0.025), 0.55, 1.80)
    detail_ratio = 1.0 + float(detail_strength) * (source_detail_ratio - 1.0)
    detail_ratio = np.clip(detail_ratio, 0.45, 2.10)
    final_l = np.clip(tr_base * detail_ratio, 0.0, 1.0)
    ratio = np.clip((final_l + 0.015) / (tr_l + 0.015), 0.40, 2.00)
    out = transferred * ratio[..., None]
    return finite01(out * mask[..., None] + transferred * (1.0 - mask[..., None]))


def protect_output(src: np.ndarray, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
    src_lab, space = rgb_to_lab(src)
    out_lab, _ = rgb_to_lab(out)
    src_l = luminance(src)
    out_l = luminance(out)
    src_sat = src.max(axis=-1) - src.min(axis=-1)
    out_sat = out.max(axis=-1) - out.min(axis=-1)
    out_gray = np.repeat(out_l[..., None], 3, axis=-1)

    neutral_region = np.clip((0.12 - src_sat) / 0.12, 0.0, 1.0)
    tonal_guard = np.clip(np.abs(src_l - 0.5) / 0.5, 0.0, 1.0)
    neutral_guard = np.clip(neutral_region + 0.50 * tonal_guard * neutral_region, 0.0, 1.0)
    out_lab[..., 1:3] = out_lab[..., 1:3] * (1.0 - 0.85 * neutral_guard[..., None]) + src_lab[..., 1:3] * (0.85 * neutral_guard[..., None])
    out = lab_to_rgb(out_lab, space)
    out_l = luminance(out)
    out_sat = out.max(axis=-1) - out.min(axis=-1)
    out_gray = np.repeat(out_l[..., None], 3, axis=-1)

    neutral = np.clip((0.13 - src_sat) / 0.13, 0.0, 1.0) * np.clip((out_sat - src_sat - 0.03) / 0.35, 0.0, 1.0)
    out = out * (1.0 - 0.55 * neutral[..., None]) + out_gray * (0.55 * neutral[..., None])

    highlight = np.clip((src_l - 0.84) / 0.16, 0.0, 1.0) * np.clip((0.28 - src_sat) / 0.28, 0.0, 1.0)
    out_l = luminance(out)
    out_gray = np.repeat(out_l[..., None], 3, axis=-1)
    out = out * (1.0 - 0.65 * highlight[..., None]) + out_gray * (0.65 * highlight[..., None])

    out_l = luminance(out)
    gray = np.repeat(out_l[..., None], 3, axis=-1)
    chroma = out - gray
    sat = out.max(axis=-1) - out.min(axis=-1)
    cap = np.clip(0.18 + src_sat * 2.6, 0.32, 0.78)
    scale = np.minimum(1.0, cap / np.maximum(sat, EPS))
    out = gray + chroma * scale[..., None]
    return finite01(out * mask[..., None] + src * (1.0 - mask[..., None]))


def masked_std(values: np.ndarray, mask: np.ndarray) -> float:
    vals = values[mask > 0.5].astype(np.float64)
    if vals.size == 0:
        return 0.0
    return float(vals.std())


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    vals = values[mask > 0.5].astype(np.float64)
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def anti_fade_guard(
    src: np.ndarray,
    out: np.ndarray,
    mask: np.ndarray,
    min_luma_std_ratio: float,
    min_saturation_ratio: float,
    strength: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    guard_strength = float(np.clip(strength, 0.0, 1.0))
    src_l = luminance(src)
    out_l = luminance(out)
    src_l_std = masked_std(src_l, mask)
    out_l_std = masked_std(out_l, mask)
    target_l_std = src_l_std * float(max(0.0, min_luma_std_ratio)) * 1.045
    luma_boost = 1.0
    guarded = out.copy()
    if guard_strength > 0.0 and src_l_std > EPS and out_l_std + EPS < target_l_std:
        luma_boost = float(np.clip(target_l_std / max(out_l_std, 0.01), 1.0, 1.85))
        mean_l = masked_mean(out_l, mask)
        restored_l = np.clip(mean_l + (out_l - mean_l) * luma_boost, 0.0, 1.0)
        restored_l = out_l * (1.0 - guard_strength) + restored_l * guard_strength
        ratio = np.clip((restored_l + 0.015) / (out_l + 0.015), 0.50, 1.90)
        guarded = finite01(guarded * ratio[..., None])

    src_sat = src.max(axis=-1) - src.min(axis=-1)
    guarded_l = luminance(guarded)
    guarded_sat = guarded.max(axis=-1) - guarded.min(axis=-1)
    src_sat_mean = masked_mean(src_sat, mask)
    out_sat_mean = masked_mean(guarded_sat, mask)
    target_sat = src_sat_mean * float(max(0.0, min_saturation_ratio))
    sat_boost = 1.0
    if guard_strength > 0.0 and src_sat_mean > EPS and out_sat_mean + EPS < target_sat:
        sat_boost = float(np.clip(target_sat / max(out_sat_mean, 0.01), 1.0, 1.85))
        gray = np.repeat(guarded_l[..., None], 3, axis=-1)
        chroma = guarded - gray
        boosted = finite01(gray + chroma * sat_boost)
        guarded = guarded * (1.0 - guard_strength) + boosted * guard_strength

    guarded = finite01(guarded * mask[..., None] + src * (1.0 - mask[..., None]))
    final_l = luminance(guarded)
    final_sat = guarded.max(axis=-1) - guarded.min(axis=-1)
    metrics = {
        "strength": guard_strength,
        "min_luma_std_ratio": float(min_luma_std_ratio),
        "min_saturation_ratio": float(min_saturation_ratio),
        "source_luma_std": src_l_std,
        "pre_guard_luma_std": out_l_std,
        "post_guard_luma_std": masked_std(final_l, mask),
        "source_saturation_mean": src_sat_mean,
        "pre_guard_saturation_mean": out_sat_mean,
        "post_guard_saturation_mean": masked_mean(final_sat, mask),
        "luma_boost": luma_boost,
        "saturation_boost": sat_boost,
    }
    return guarded, metrics


def crop_source(canvas: np.ndarray, offset: Tuple[int, int], src_shape: Tuple[int, int, int]) -> np.ndarray:
    ox, oy = offset
    h, w = src_shape[:2]
    return canvas[oy : oy + h, ox : ox + w].copy()


def write_debug(
    debug_dir: Optional[str],
    data: Dict[str, Any],
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
) -> None:
    if not debug_dir:
        return
    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        save_rgb(d / f"{name}.png", img)
    for name, mask in masks.items():
        save_gray(d / f"{name}.png", mask)
    with open(d / "debug.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def normalize_debug_dir(debug_dir: Optional[str]) -> Optional[Path]:
    if not debug_dir:
        return None
    d = Path(debug_dir)
    if d.is_absolute():
        return d
    if d.parts and d.parts[0] == "outputs":
        return d
    return Path("outputs") / d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic low-frequency look transfer while preserving source detail.")
    parser.add_argument("--source", required=True, help="Source image path.")
    parser.add_argument("--reference", required=True, help="Stylized look reference image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--debug-dir", default=None, help="Optional directory for debug images and JSON.")
    parser.add_argument("--align", choices=["auto", "identity", "center"], default="auto", help="Alignment mode.")
    parser.add_argument("--align-scale-min", type=float, default=0.25, help="Minimum accepted similarity scale.")
    parser.add_argument("--align-scale-max", type=float, default=8.0, help="Maximum accepted similarity scale.")
    parser.add_argument("--light-map-grid", type=int, default=20, help="Approximate light map grid rows.")
    parser.add_argument("--base-radius", type=float, default=24.0, help="Edge-aware base smoothing radius.")
    parser.add_argument("--base-filter", choices=["guided_auto", "bilateral", "gaussian"], default="guided_auto", help="Low-frequency base filter.")
    parser.add_argument("--base-filter-eps", type=float, default=0.01, help="Edge-aware base filter range parameter.")
    parser.add_argument("--local-strength", type=float, default=0.65, help="Local correction blend strength.")
    parser.add_argument("--local-luma-strength", type=float, default=0.25, help="Strength for local luminance correction; chroma local correction is unaffected.")
    parser.add_argument("--detail-strength", type=float, default=1.0, help="Source detail restoration strength.")
    parser.add_argument("--trust-threshold", type=float, default=0.15, help="Minimum trust used for fitting.")
    parser.add_argument("--min-luma-std-ratio", type=float, default=0.85, help="Minimum output/source luminance std ratio enforced by anti-fade guard.")
    parser.add_argument("--min-saturation-ratio", type=float, default=0.80, help="Minimum output/source saturation ratio enforced by anti-fade guard.")
    parser.add_argument("--anti-fade-strength", type=float, default=1.0, help="Strength for final contrast and saturation anti-fade guard.")
    parser.add_argument("--pipeline-version", choices=["v0_2_6"], default="v0_2_6", help="LookAlign pipeline version.")
    parser.add_argument("--sa-lut-size", type=int, default=33, help="SA-LUT RGB lattice resolution.")
    parser.add_argument("--sa-lut-context-bins", type=int, default=2, help="SA-LUT context bin count.")
    parser.add_argument("--sa-lut-fit-max-samples", type=int, default=250000, help="Maximum trusted samples used for SA-LUT fitting.")
    parser.add_argument("--sa-lut-ridge", type=float, default=0.035, help="Ridge term for sparse SA-LUT cells.")
    parser.add_argument("--sa-lut-smooth", type=float, default=0.75, help="Gaussian smoothing sigma for fitted SA-LUT residuals.")
    parser.add_argument("--light-map-smooth", type=float, default=0.35, help="Gaussian smoothing sigma for coarse light map fitting.")
    parser.add_argument("--light-map-upsample", choices=["source_guided", "linear"], default="source_guided", help="Light map upsampling mode.")
    parser.add_argument("--light-map-radius", type=float, default=8.0, help="Source-guided light map filter radius.")
    parser.add_argument("--max-exposure-delta", type=float, default=0.18, help="Maximum local exposure delta in normalized Lab luminance.")
    parser.add_argument("--max-contrast-gain", type=float, default=1.35, help="Maximum local luminance contrast gain.")
    parser.add_argument("--render-backend", choices=["auto", "pytorch", "numpy", "opencv", "coreimage", "accelerate"], default="auto", help="Rendering backend preference.")
    return parser.parse_args()


def lookalign_defaults() -> Dict[str, Any]:
    return {
        "debug_dir": None,
        "align": "auto",
        "align_scale_min": 0.25,
        "align_scale_max": 8.0,
        "pipeline_version": "v0_2_6",
        "light_map_grid": 20,
        "base_radius": 24.0,
        "base_filter": "guided_auto",
        "base_filter_eps": 0.01,
        "local_strength": 0.65,
        "local_luma_strength": 0.25,
        "detail_strength": 1.0,
        "trust_threshold": 0.15,
        "min_luma_std_ratio": 0.85,
        "min_saturation_ratio": 0.80,
        "anti_fade_strength": 1.0,
        "sa_lut_size": 33,
        "sa_lut_context_bins": 2,
        "sa_lut_fit_max_samples": 250000,
        "sa_lut_ridge": 0.035,
        "sa_lut_smooth": 0.75,
        "light_map_smooth": 0.35,
        "light_map_upsample": "source_guided",
        "light_map_radius": 8.0,
        "max_exposure_delta": 0.18,
        "max_contrast_gain": 1.35,
        "render_backend": "auto",
    }


def normalize_cli_paths(output: str, debug_dir: Optional[str]) -> Tuple[str, Optional[str]]:
    output_path = Path(output)
    if output_path.parent.name != "outputs" or output_path.name != "output.png":
        return str(output_path), debug_dir
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = output_path.parent / run_id
    resolved_output = run_dir / "lookalign_output.png"
    resolved_debug = debug_dir if debug_dir is not None else str(run_dir / "debug")
    return str(resolved_output), resolved_debug


def _config_to_namespace(config: Optional[Dict[str, Any] | argparse.Namespace]) -> argparse.Namespace:
    values = lookalign_defaults()
    if config is not None:
        raw_config = vars(config) if isinstance(config, argparse.Namespace) else dict(config)
        if "light_map_grid" not in raw_config and "residual_grid" in raw_config:
            raw_config["light_map_grid"] = raw_config["residual_grid"]
        if isinstance(config, argparse.Namespace):
            values.update(raw_config)
        else:
            values.update(raw_config)
    return argparse.Namespace(**values)


def run_lookalign(
    source_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    config: Optional[Dict[str, Any] | argparse.Namespace] = None,
) -> Dict[str, Any]:
    args = _config_to_namespace(config)
    args.source = str(source_path)
    args.reference = str(reference_path)
    args.output = str(output_path)
    args.debug_dir = str(normalize_debug_dir(args.debug_dir)) if args.debug_dir else None

    warnings: List[str] = []
    src = load_rgb(args.source)
    ref = load_rgb(args.reference)

    matrix, mode_used, align_conf, align_info = estimate_alignment(args, src, ref, warnings)
    canvas = build_union_canvas(src, ref, matrix)
    source_canvas = canvas["source_canvas"]
    source_mask = canvas["source_mask"]
    reference_canvas = canvas["reference_canvas"]
    reference_mask = canvas["reference_mask"]
    overlap = canvas["overlap_mask"]
    offset = canvas["offset"]

    overlap_ratio = float(overlap.sum() / max(1.0, source_mask.sum()))
    if overlap_ratio < 0.02:
        warnings.append("Reference overlap with source is tiny; output will stay close to source.")

    source_base, source_base_info = make_edge_aware_base(source_canvas, source_mask, args.base_radius, args.base_filter_eps, args.base_filter)
    reference_base, reference_base_info = make_edge_aware_base(reference_canvas, reference_mask, args.base_radius, args.base_filter_eps, args.base_filter)
    trust = compute_trust(source_canvas, reference_canvas, overlap) if overlap.sum() > 0 else np.zeros_like(overlap)
    mean_trust = float(trust[overlap > 0.5].mean()) if np.any(overlap > 0.5) else 0.0
    trusted = trust * overlap
    trusted = np.where(trusted >= float(args.trust_threshold), trusted, 0.0).astype(np.float32)
    trusted_count = int(np.count_nonzero(trusted > 0.0))

    if trusted_count < max(250, int(0.001 * source_mask.sum())):
        warnings.append("Trusted overlap is small; using conservative overlap-weighted global fitting and reducing local correction.")
        conservative = np.clip(trust, 0.0, 0.35) * overlap
        if conservative.sum() < 50.0:
            conservative = overlap * 0.10
        fit_weights = conservative.astype(np.float32)
        local_strength = min(float(args.local_strength), 0.20)
    else:
        fit_weights = trusted
        local_strength = float(args.local_strength)

    if align_conf < 0.25:
        local_strength = min(local_strength, 0.25)

    sa_lut_base_result, global_params, sa_lut_context_map, sa_lut_base_low = sa_lut_global_color_transfer(
        source_canvas,
        source_base,
        reference_base,
        source_mask,
        fit_weights,
        args,
        warnings,
    )

    light_maps, light_stats = compute_light_map(
        sa_lut_base_low,
        sa_lut_base_result,
        reference_base,
        source_mask,
        fit_weights,
        args.light_map_grid,
        args.light_map_smooth,
        args.light_map_radius,
        args.max_exposure_delta,
        args.max_contrast_gain,
        args.base_filter_eps,
        args.light_map_upsample,
        warnings,
    )
    light_map_device = str(global_params.get("selected_device", "cpu"))
    try:
        local_result, light_map_device = apply_light_map_torch(
            sa_lut_base_result,
            source_mask,
            light_maps,
            local_strength if light_stats.get("enabled", False) else 0.0,
            args.local_luma_strength,
            light_map_device,
        )
    except Exception as exc:
        if light_map_device != "mps":
            raise
        warnings.append(f"Light map MPS application failed ({exc}); retrying on CPU.")
        local_result, light_map_device = apply_light_map_torch(
            sa_lut_base_result,
            source_mask,
            light_maps,
            local_strength if light_stats.get("enabled", False) else 0.0,
            args.local_luma_strength,
            "cpu",
        )

    detailed = apply_detail_preservation(source_canvas, local_result, source_mask, args.base_radius, args.detail_strength)
    protected = protect_output(source_canvas, detailed, source_mask)
    guarded, guard_metrics = anti_fade_guard(
        source_canvas,
        protected,
        source_mask,
        args.min_luma_std_ratio,
        args.min_saturation_ratio,
        args.anti_fade_strength,
    )
    src_lab_full, _ = rgb_to_lab(source_canvas)
    guarded_lab, _ = rgb_to_lab(guarded)
    chroma_metrics = chroma_delta_metrics(src_lab_full, guarded_lab, source_mask)
    output = crop_source(guarded, offset, src.shape)
    output = finite01(output)
    save_rgb(args.output, output)

    aligned_ref_vis = reference_canvas * reference_mask[..., None] + (1.0 - reference_mask[..., None])
    debug_data = {
        "pipeline_version": "v0_2_6",
        "source_shape": list(src.shape),
        "reference_shape": list(ref.shape),
        "estimated_transform": matrix.astype(float).tolist(),
        "estimated_transform_union": canvas["matrix_union"].astype(float).tolist(),
        "alignment_mode_used": mode_used,
        "alignment_confidence": float(align_conf),
        "alignment_info": align_info,
        "overlap_ratio": overlap_ratio,
        "mean_trust": mean_trust,
        "trusted_pixel_count": trusted_count,
        "global_fit_parameters": global_params,
        "base_filter": {"source": source_base_info, "reference": reference_base_info},
        "light_map_parameters": light_stats,
        "light_map_device": light_map_device,
        "chroma_transfer_metrics": chroma_metrics,
        "local_strength_used": float(local_strength if light_stats.get("enabled", False) else 0.0),
        "local_luma_strength": float(args.local_luma_strength),
        "anti_fade_guard": guard_metrics,
        "warnings": warnings,
    }
    debug_images = {
        "aligned_reference": aligned_ref_vis,
        "source_base": finite01(source_base),
        "reference_base": finite01(reference_base),
        "sa_lut_base_result": finite01(sa_lut_base_result),
        "light_map_result": finite01(local_result),
        "final_output": output,
    }
    debug_masks = {
        "reference_valid_mask": reference_mask,
        "overlap_mask": overlap,
        "trust_map": trust,
        "sa_lut_context_map": np.clip(sa_lut_context_map, 0.0, 1.0).astype(np.float32),
        "light_map_coarse": scalar_map_visual(resize_map(light_maps["coarse_light_map"], source_mask.shape, interpolation=1), 0.0, args.max_exposure_delta * 2.0),
        "light_map_guided": scalar_map_visual(light_maps["light_map"], 0.0, args.max_exposure_delta * 2.0),
        "light_map_confidence": np.clip(light_maps["confidence"], 0.0, 1.0).astype(np.float32),
        "light_map_edge_gate": np.clip(light_maps["edge_gate"], 0.0, 1.0).astype(np.float32),
    }

    write_debug(
        args.debug_dir,
        debug_data,
        images=debug_images,
        masks=debug_masks,
    )

    debug_paths: Dict[str, str] = {}
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_names = [
            "aligned_reference",
            "source_base",
            "reference_base",
            "sa_lut_base_result",
            "light_map_result",
            "final_output",
            "reference_valid_mask",
            "overlap_mask",
            "trust_map",
            "sa_lut_context_map",
            "light_map_coarse",
            "light_map_guided",
            "light_map_confidence",
            "light_map_edge_gate",
        ]
        for name in debug_names:
            debug_paths[name] = str(debug_dir / f"{name}.png")
        debug_paths["debug_json"] = str(debug_dir / "debug.json")

    return {
        "output_path": str(args.output),
        "debug_dir": args.debug_dir,
        "debug_paths": debug_paths,
        "metadata": debug_data,
        "warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    args.output, args.debug_dir = normalize_cli_paths(args.output, args.debug_dir)
    result = run_lookalign(args.source, args.reference, args.output, args)

    if result["warnings"]:
        for msg in result["warnings"]:
            print(f"warning: {msg}")
    print(f"wrote {result['output_path']}")
    if result["debug_dir"]:
        print(f"wrote debug data to {result['debug_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
