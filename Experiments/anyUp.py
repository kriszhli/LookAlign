from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

try:
    from .kMeans import CLUSTER_PALETTE, OVERLAY_ALPHA, overlay_clusters
except ImportError:
    from kMeans import CLUSTER_PALETTE, OVERLAY_ALPHA, overlay_clusters


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ANYUP_MODEL_DIR = SCRIPT_DIR / "models"
ANYUP_CHECKPOINT_NAME = "anyup_multi_backbone.pth"

try:
    from .anyup_model import AnyUp
except ImportError:
    from anyup_model import AnyUp


DEFAULT_PCA_DIMS = 32
DEFAULT_Q_CHUNK_SIZE = 256
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _project_features(
    patch_features: np.ndarray,
    pca_dims: int,
) -> Tuple[np.ndarray, int]:
    feature_dim = patch_features.shape[1]
    projected_dim = max(1, min(int(pca_dims), feature_dim, patch_features.shape[0]))
    if projected_dim >= feature_dim:
        return patch_features.astype(np.float32), feature_dim

    pca = PCA(n_components=projected_dim, whiten=False, random_state=0)
    projected = pca.fit_transform(patch_features).astype(np.float32)
    return projected, projected_dim


def _image_tensor(image: Image.Image, output_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    resized = image.convert("RGB").resize(output_size, Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (tensor.to(device) - mean) / std


def _feature_grid(projected_features: np.ndarray, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
    grid = projected_features.reshape(grid_h, grid_w, projected_features.shape[1])
    return torch.from_numpy(grid).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, a_min=1e-6, a_max=None)


def cluster_dense_features(
    dense_features: np.ndarray,
    fit_features: np.ndarray,
    output_size: Tuple[int, int],
    cluster_count: int,
) -> np.ndarray:
    estimator = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=8192,
        n_init=3,
        random_state=0,
    )
    estimator.fit(fit_features)
    labels = estimator.predict(dense_features)
    return labels.reshape(output_size[1], output_size[0])


def render_dense_cluster_map(labels: np.ndarray, cluster_count: int) -> Image.Image:
    if cluster_count <= len(CLUSTER_PALETTE):
        colors = CLUSTER_PALETTE[labels.reshape(-1)]
    else:
        repeats = int(np.ceil(cluster_count / len(CLUSTER_PALETTE)))
        palette = np.tile(CLUSTER_PALETTE, (repeats, 1))[:cluster_count]
        colors = palette[labels.reshape(-1)]
    cluster_rgb = colors.reshape(labels.shape[0], labels.shape[1], 3)
    return Image.fromarray(cluster_rgb.astype(np.uint8), mode="RGB")


class AnyUpUpsampler:
    def __init__(self) -> None:
        self._model: AnyUp | None = None
        self._device_type: str | None = None

    def _load_model(self, device: torch.device) -> AnyUp:
        if self._model is not None and self._device_type == device.type:
            return self._model

        ANYUP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ANYUP_MODEL_DIR / ANYUP_CHECKPOINT_NAME
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"AnyUp checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        model = AnyUp().to(device).eval()
        model.load_state_dict(state_dict)
        self._model = model
        self._device_type = device.type
        return model

    def upsample(
        self,
        image: Image.Image,
        projected_features: np.ndarray,
        grid_h: int,
        grid_w: int,
        output_size: Tuple[int, int],
        device: torch.device,
        q_chunk_size: int,
    ) -> np.ndarray:
        model = self._load_model(device)
        image_tensor = _image_tensor(image, output_size, device)
        low_res_tensor = _feature_grid(projected_features, grid_h, grid_w, device)
        with torch.inference_mode():
            upsampled = model(image_tensor, low_res_tensor, q_chunk_size=q_chunk_size)
        dense_features = upsampled.squeeze(0).permute(1, 2, 0).reshape(-1, projected_features.shape[1])
        return dense_features.float().cpu().numpy().astype(np.float32)


def build_anyup_overlay_from_features(
    patch_features: np.ndarray,
    grid_h: int,
    grid_w: int,
    cluster_count: int,
    source_image: Image.Image,
    output_size: Tuple[int, int],
    device: torch.device,
    upsampler: AnyUpUpsampler,
    pca_dims: int = DEFAULT_PCA_DIMS,
    q_chunk_size: int = DEFAULT_Q_CHUNK_SIZE,
) -> Tuple[Image.Image, Image.Image, int]:
    projected_features, projected_dim = _project_features(patch_features, pca_dims)
    dense_features = upsampler.upsample(
        image=source_image,
        projected_features=projected_features,
        grid_h=grid_h,
        grid_w=grid_w,
        output_size=output_size,
        device=device,
        q_chunk_size=q_chunk_size,
    )
    normalized_low_res = _normalize_rows(projected_features)
    normalized_dense = _normalize_rows(dense_features)
    dense_labels = cluster_dense_features(
        dense_features=normalized_dense,
        fit_features=normalized_low_res,
        output_size=output_size,
        cluster_count=cluster_count,
    )
    dense_map = render_dense_cluster_map(dense_labels, cluster_count)
    base_image = source_image.convert("RGB").resize(output_size, Image.Resampling.BICUBIC)
    return overlay_clusters(base_image, dense_map, OVERLAY_ALPHA), dense_map, projected_dim
