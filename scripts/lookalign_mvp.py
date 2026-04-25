#!/usr/bin/env python3
"""LookAlign MVP: deterministic low-frequency look transfer from reference to source."""

from __future__ import annotations

import argparse
import json
import math
import os
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
    out = (finite01(img) * 255.0 + 0.5).astype(np.uint8)
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


def estimate_alignment(args: argparse.Namespace, src: np.ndarray, ref: np.ndarray, warnings: List[str]) -> Tuple[np.ndarray, str, float, Dict[str, Any]]:
    if args.align == "identity":
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), "identity", 1.0, {}
    if args.align == "center":
        return center_transform(src.shape, ref.shape), "center", 0.55, {}

    feature = estimate_feature_alignment(src, ref, args.align_scale_min, args.align_scale_max, warnings)
    if feature is not None:
        mat, confidence, info = feature
        return mat, "auto_feature", confidence, info

    fit = scale_fit_transform(src.shape, ref.shape)
    fit, phase_conf = estimate_phase_translation(src, ref, fit, warnings)
    ok, reason, overlap = validate_transform(fit, ref.shape, src.shape, args.align_scale_min, args.align_scale_max, min_overlap=0.03)
    if ok:
        conf = 0.30 + 0.35 * phase_conf
        return fit, "auto_scale_fit_phase" if phase_conf > 0 else "auto_scale_fit", conf, {"fallback_overlap_ratio": overlap, "phase_confidence": phase_conf}

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
    lum_compat = np.exp(-np.abs(mask_aware_blur(src_l, overlap, 8.0) - mask_aware_blur(ref_l, overlap, 8.0)) / 0.35)
    corr = local_correlation(src_l, ref_l, overlap, 9.0)

    edge_ref = float(np.percentile((smag + rmag)[overlap > 0.5], 95)) if np.any(overlap > 0.5) else 0.0
    edge_weight = np.clip((smag + rmag) / max(edge_ref, 0.03), 0.0, 1.0)
    structural = 0.42 * orient + 0.33 * mag_compat + 0.25 * lap_compat
    flat = 0.65 * lum_compat + 0.35 * corr
    trust = ((1.0 - edge_weight) * flat + edge_weight * structural) * (0.70 + 0.30 * corr)
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


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = values.reshape(-1, values.shape[-1]).astype(np.float64)
    w = weights.reshape(-1).astype(np.float64)
    total = max(float(w.sum()), EPS)
    mean = (v * w[:, None]).sum(axis=0) / total
    var = ((v - mean) ** 2 * w[:, None]).sum(axis=0) / total
    return mean.astype(np.float32), np.sqrt(np.maximum(var, EPS)).astype(np.float32)


def global_color_transfer(
    src_canvas: np.ndarray,
    ref_canvas: np.ndarray,
    source_mask: np.ndarray,
    weights: np.ndarray,
    blur_sigma: float,
    mean_trust: float,
    warnings: List[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    src_low = mask_aware_blur(src_canvas, source_mask, blur_sigma)
    ref_low = mask_aware_blur(ref_canvas, (weights > 0).astype(np.float32), blur_sigma)
    src_fit, space = rgb_to_lab(src_low)
    ref_fit, _ = rgb_to_lab(ref_low)
    wsum = float(weights.sum())
    if wsum < 100.0:
        warnings.append("Trusted overlap is tiny; global correction is disabled.")
        params = {
            "space": space,
            "enabled": False,
            "strength": 0.0,
            "src_mean": [0.0, 0.0, 0.0],
            "ref_mean": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "offset": [0.0, 0.0, 0.0],
        }
        return src_canvas.copy(), params

    src_mean, src_std = weighted_mean_std(src_fit, weights)
    ref_mean, ref_std = weighted_mean_std(ref_fit, weights)
    scale = np.clip(ref_std / np.maximum(src_std, 0.015), 0.55, 1.85)
    offset = ref_mean - src_mean * scale
    offset = np.clip(offset, -0.22, 0.22)
    strength = 1.0 if mean_trust >= 0.22 else float(np.clip(0.35 + mean_trust, 0.25, 0.65))

    src_full, _ = rgb_to_lab(src_canvas)
    transformed = src_full * scale.reshape(1, 1, 3) + offset.reshape(1, 1, 3)
    transformed = src_full * (1.0 - strength) + transformed * strength
    result = lab_to_rgb(transformed, space)
    result = result * source_mask[..., None] + src_canvas * (1.0 - source_mask[..., None])
    params = {
        "space": space,
        "enabled": True,
        "strength": strength,
        "src_mean": src_mean.tolist(),
        "ref_mean": ref_mean.tolist(),
        "src_std": src_std.tolist(),
        "ref_std": ref_std.tolist(),
        "scale": scale.tolist(),
        "offset": offset.tolist(),
    }
    return finite01(result), params


def compute_local_field(
    global_result: np.ndarray,
    ref_canvas: np.ndarray,
    source_mask: np.ndarray,
    trusted_weights: np.ndarray,
    grid: int,
    blur_sigma: float,
    warnings: List[str],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    h, w = source_mask.shape
    if trusted_weights.sum() < 100.0:
        warnings.append("Local correction skipped: insufficient trusted support.")
        return np.zeros_like(global_result), global_result.copy(), False

    global_low = mask_aware_blur(global_result, source_mask, blur_sigma)
    ref_low = mask_aware_blur(ref_canvas, (trusted_weights > 0).astype(np.float32), blur_sigma)
    delta = np.clip(ref_low - global_low, -0.22, 0.22)
    gy = max(2, int(grid))
    gx = max(2, int(round(grid * w / max(h, 1))))
    field_grid = np.zeros((gy, gx, 3), dtype=np.float32)
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
            dd = delta[y0:y1, x0:x1]
            field_grid[iy, ix] = (dd * ww[..., None]).sum(axis=(0, 1)) / max(total, EPS)
            conf_grid[iy, ix] = min(1.0, total / max(50.0, float((y1 - y0) * (x1 - x0)) * 0.08))

    if float(conf_grid.sum()) < 1.0:
        warnings.append("Local correction skipped: no reliable grid cells.")
        return np.zeros_like(global_result), global_result.copy(), False

    if ndimage is not None:
        smooth_sigma = max(0.65, grid / 20.0)
        den = ndimage.gaussian_filter(conf_grid, sigma=smooth_sigma, mode="nearest")
        smoothed = np.zeros_like(field_grid)
        for c in range(3):
            num = ndimage.gaussian_filter(field_grid[..., c] * conf_grid, sigma=smooth_sigma, mode="nearest")
            smoothed[..., c] = num / np.maximum(den, EPS)
        field_grid = smoothed
    elif cv2 is not None:
        conf3 = conf_grid[..., None]
        num = cv2.GaussianBlur(field_grid * conf3, (0, 0), sigmaX=1.0)
        den = cv2.GaussianBlur(conf_grid, (0, 0), sigmaX=1.0)
        field_grid = num / np.maximum(den[..., None], EPS)

    if cv2 is not None:
        field = cv2.resize(field_grid, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    else:
        zoom_y = h / field_grid.shape[0]
        zoom_x = w / field_grid.shape[1]
        field = ndimage.zoom(field_grid, (zoom_y, zoom_x, 1), order=3)[:h, :w].astype(np.float32)

    lum = luminance(global_result)
    sx, sy = sobel_xy(mask_aware_blur(lum, source_mask, 1.0))
    edge = np.sqrt(sx * sx + sy * sy)
    edge_ref = float(np.percentile(edge[source_mask > 0.5], 95)) if np.any(source_mask > 0.5) else 0.0
    edge_gate = 1.0 - 0.55 * np.clip(edge / max(edge_ref, 0.03), 0.0, 1.0)
    field = np.clip(field * edge_gate[..., None] * source_mask[..., None], -0.18, 0.18)
    local_result = finite01(global_result + field)
    local_result = local_result * source_mask[..., None] + global_result * (1.0 - source_mask[..., None])
    return field, local_result, True


def apply_detail_preservation(src: np.ndarray, transferred: np.ndarray, mask: np.ndarray, blur_sigma: float, detail_strength: float) -> np.ndarray:
    src_l = luminance(src)
    tr_l = luminance(transferred)
    detail_sigma = max(1.0, blur_sigma * 0.33)
    src_base = mask_aware_blur(src_l, mask, detail_sigma)
    tr_base = mask_aware_blur(tr_l, mask, detail_sigma)
    source_detail = np.clip(src_l - src_base, -0.35, 0.35)
    final_l = np.clip(tr_base + float(detail_strength) * source_detail, 0.0, 1.0)
    ratio = np.clip((final_l + 0.015) / (tr_l + 0.015), 0.45, 1.75)
    out = transferred * ratio[..., None]
    return finite01(out * mask[..., None] + transferred * (1.0 - mask[..., None]))


def protect_output(src: np.ndarray, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
    src_l = luminance(src)
    out_l = luminance(out)
    src_sat = src.max(axis=-1) - src.min(axis=-1)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic low-frequency look transfer while preserving source detail.")
    parser.add_argument("--source", required=True, help="Source image path.")
    parser.add_argument("--reference", required=True, help="Stylized look reference image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--debug-dir", default=None, help="Optional directory for debug images and JSON.")
    parser.add_argument("--align", choices=["auto", "identity", "center"], default="auto", help="Alignment mode.")
    parser.add_argument("--align-scale-min", type=float, default=0.25, help="Minimum accepted similarity scale.")
    parser.add_argument("--align-scale-max", type=float, default=8.0, help="Maximum accepted similarity scale.")
    parser.add_argument("--grid", type=int, default=16, help="Approximate local correction grid rows.")
    parser.add_argument("--blur-sigma", type=float, default=24.0, help="Low-frequency Gaussian sigma.")
    parser.add_argument("--local-strength", type=float, default=0.65, help="Local correction blend strength.")
    parser.add_argument("--detail-strength", type=float, default=1.0, help="Source detail restoration strength.")
    parser.add_argument("--trust-threshold", type=float, default=0.15, help="Minimum trust used for fitting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    source_low = mask_aware_blur(source_canvas, source_mask, args.blur_sigma)
    reference_low = mask_aware_blur(reference_canvas, reference_mask, args.blur_sigma)
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

    global_result, global_params = global_color_transfer(
        source_canvas,
        reference_canvas,
        source_mask,
        fit_weights,
        args.blur_sigma,
        mean_trust,
        warnings,
    )

    field, local_unblended, local_enabled = compute_local_field(
        global_result,
        reference_canvas,
        source_mask,
        fit_weights,
        args.grid,
        args.blur_sigma,
        warnings,
    )
    local_result = finite01(global_result * (1.0 - local_strength) + local_unblended * local_strength)
    if not local_enabled:
        local_result = global_result

    detailed = apply_detail_preservation(source_canvas, local_result, source_mask, args.blur_sigma, args.detail_strength)
    protected = protect_output(source_canvas, detailed, source_mask)
    output = crop_source(protected, offset, src.shape)
    output = finite01(output)
    save_rgb(args.output, output)

    field_vis = np.clip(0.5 + field / 0.36, 0.0, 1.0)
    aligned_ref_vis = reference_canvas * reference_mask[..., None] + (1.0 - reference_mask[..., None])
    debug_data = {
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
        "local_strength_used": float(local_strength if local_enabled else 0.0),
        "warnings": warnings,
    }
    write_debug(
        args.debug_dir,
        debug_data,
        images={
            "aligned_reference": aligned_ref_vis,
            "source_lowfreq": finite01(source_low),
            "reference_lowfreq": finite01(reference_low),
            "global_result": finite01(global_result),
            "local_field_visualization": field_vis,
            "local_result": finite01(local_result),
            "final_output": output,
        },
        masks={
            "reference_valid_mask": reference_mask,
            "overlap_mask": overlap,
            "trust_map": trust,
        },
    )

    if warnings:
        for msg in warnings:
            print(f"warning: {msg}")
    print(f"wrote {args.output}")
    if args.debug_dir:
        print(f"wrote debug data to {args.debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
