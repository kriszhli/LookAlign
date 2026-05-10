from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from .kMeans import OVERLAY_ALPHA, overlay_clusters, render_cluster_map
except ImportError:
    from kMeans import OVERLAY_ALPHA, overlay_clusters, render_cluster_map


DEFAULT_ERODE_RADIUS = 10
DEFAULT_MODAL_RADIUS = 6


def _resize_label_grid(labels: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    label_image = Image.fromarray(labels.astype(np.uint8), mode="L")
    resized = label_image.resize(output_size, Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.int32)


def _build_markers(labels: np.ndarray, cluster_count: int, erode_radius: int) -> np.ndarray:
    height, width = labels.shape
    markers = np.zeros((height, width), dtype=np.int32)
    if erode_radius > 0:
        kernel_size = 2 * erode_radius + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    else:
        kernel = None

    for label_idx in range(cluster_count):
        mask = (labels == label_idx).astype(np.uint8)
        if mask.sum() == 0:
            continue
        if kernel is not None:
            seed = cv2.erode(mask, kernel, iterations=1)
            if seed.sum() == 0:
                seed = mask
        else:
            seed = mask
        markers[seed > 0] = label_idx + 1
    return markers


def _modal_value(window: np.ndarray) -> int:
    values = window.astype(np.int32)
    bincount = np.bincount(values)
    return int(np.argmax(bincount))


def _apply_modal_filter(labels: np.ndarray, modal_radius: int) -> np.ndarray:
    if modal_radius <= 0:
        return labels
    kernel_size = 2 * modal_radius + 1
    filtered = ndimage.generic_filter(
        labels.astype(np.int32),
        function=_modal_value,
        size=(kernel_size, kernel_size),
        mode="nearest",
    )
    return filtered.astype(np.int32)


def watershed_refine_labels(
    cluster_labels: np.ndarray,
    cluster_count: int,
    source_image: Image.Image,
    output_size: Tuple[int, int],
    erode_radius: int = DEFAULT_ERODE_RADIUS,
    modal_radius: int = DEFAULT_MODAL_RADIUS,
) -> np.ndarray:
    resized_labels = _resize_label_grid(cluster_labels, output_size)
    markers = _build_markers(resized_labels, cluster_count, max(0, int(erode_radius)))
    image_bgr = cv2.cvtColor(
        np.asarray(source_image.convert("RGB").resize(output_size, Image.Resampling.BICUBIC), dtype=np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    watershed_markers = cv2.watershed(image_bgr, markers.copy())
    refined = watershed_markers.copy()
    unresolved = refined <= 0
    refined[unresolved] = resized_labels[unresolved] + 1
    refined_labels = refined - 1
    return _apply_modal_filter(refined_labels, max(0, int(modal_radius)))


def build_watershed_overlay_from_labels(
    cluster_labels: np.ndarray,
    cluster_count: int,
    source_image: Image.Image,
    output_size: Tuple[int, int],
    erode_radius: int = DEFAULT_ERODE_RADIUS,
    modal_radius: int = DEFAULT_MODAL_RADIUS,
) -> tuple[Image.Image, Image.Image]:
    refined_labels = watershed_refine_labels(
        cluster_labels=cluster_labels,
        cluster_count=cluster_count,
        source_image=source_image,
        output_size=output_size,
        erode_radius=erode_radius,
        modal_radius=modal_radius,
    )
    target_map = render_cluster_map(refined_labels, refined_labels.shape[0], refined_labels.shape[1], source_image.size)
    return overlay_clusters(source_image, target_map, OVERLAY_ALPHA), target_map
