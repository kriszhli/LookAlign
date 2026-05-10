from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


OVERLAY_ALPHA = 0.45
CLUSTER_PALETTE = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [174, 199, 232],
        [255, 187, 120],
    ],
    dtype=np.uint8,
)


def cluster_patch_features(
    patch_features: np.ndarray,
    grid_h: int,
    grid_w: int,
    cluster_count: int,
) -> np.ndarray:
    estimator = KMeans(n_clusters=cluster_count, n_init=10, random_state=0)
    labels = estimator.fit_predict(patch_features)
    return labels.reshape(grid_h, grid_w)


def make_cluster_colors(labels: np.ndarray, cluster_count: int) -> np.ndarray:
    if cluster_count <= len(CLUSTER_PALETTE):
        return CLUSTER_PALETTE[labels]
    repeats = int(np.ceil(cluster_count / len(CLUSTER_PALETTE)))
    extended = np.tile(CLUSTER_PALETTE, (repeats, 1))[:cluster_count]
    return extended[labels]


def render_cluster_map(labels: np.ndarray, grid_h: int, grid_w: int, output_size: Tuple[int, int]) -> Image.Image:
    colors = make_cluster_colors(labels.reshape(-1), int(labels.max()) + 1)
    cluster_rgb = colors.reshape(grid_h, grid_w, 3)
    cluster_image = Image.fromarray(cluster_rgb, mode="RGB")
    return cluster_image.resize(output_size, Image.Resampling.NEAREST)


def overlay_clusters(base_image: Image.Image, cluster_image: Image.Image, alpha: float = OVERLAY_ALPHA) -> Image.Image:
    base = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    overlay = np.asarray(cluster_image.convert("RGB"), dtype=np.float32)
    blended = np.clip(base * (1.0 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def build_overlay_from_labels(
    cluster_labels: np.ndarray,
    base_image: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    grid_h, grid_w = cluster_labels.shape
    cluster_map = render_cluster_map(cluster_labels, grid_h, grid_w, base_image.size)
    return overlay_clusters(base_image, cluster_map), cluster_map


def build_overlay_from_features(
    patch_features: np.ndarray,
    grid_h: int,
    grid_w: int,
    cluster_count: int,
    base_image: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    cluster_labels = cluster_patch_features(
        patch_features=patch_features,
        grid_h=grid_h,
        grid_w=grid_w,
        cluster_count=cluster_count,
    )
    return build_overlay_from_labels(cluster_labels, base_image)
