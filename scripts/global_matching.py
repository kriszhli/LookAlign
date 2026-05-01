"""LookAlign global matching: Neural Preset DNCM color style transfer."""

from __future__ import annotations

import json
import sys
import time
import types
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from efficientnet_pytorch import EfficientNet
except ImportError as exc:  # pragma: no cover - exercised only on missing local dependency.
    EfficientNet = None  # type: ignore[assignment]
    _EFFICIENTNET_IMPORT_ERROR = exc
else:
    _EFFICIENTNET_IMPORT_ERROR = None


Tensor = torch.Tensor
ROOT = Path(__file__).resolve().parents[1]
GLOBAL_STAGE_VERSION = "v0.4.5-global-neural-preset-dncm"
PIPELINE_VERSION = "v0.4.5-neural-preset-bilateral-lab"
DEFAULT_NEURAL_PRESET_CKPT = "ckpts/neural_preset/best.ckpt"
NEURAL_PRESET_CKPT_SOURCE = "https://drive.google.com/open?id=1TZRVwIlzBBewwzgjrScrVzeynhBSLmm0&usp=drive_fs"


@dataclass
class GlobalMatchingConfig:
    fit_long_edge: int = 768
    thumbnail_size: int = 256
    k: int = 16
    checkpoint_path: str = DEFAULT_NEURAL_PRESET_CKPT
    fullres_chunk_pixels: int = 262144


def select_torch_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


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


def to_nchw(img: np.ndarray, device: torch.device) -> Tensor:
    arr = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def to_nchw_mps(img: np.ndarray, device: torch.device) -> Tensor:
    return to_nchw(img, device)


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


def rgb_to_lab(rgb: Tensor) -> Tensor:
    linear = srgb_to_linear(rgb)
    r, g, b = linear.unbind(dim=1)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    xn, yn, zn = 0.95047, 1.0, 1.08883
    delta = 6.0 / 29.0
    delta3 = delta ** 3

    def f(t: Tensor) -> Tensor:
        return torch.where(t > delta3, t.pow(1.0 / 3.0), t / (3.0 * delta * delta) + 4.0 / 29.0)

    fx = f(x / xn)
    fy = f(y / yn)
    fz = f(z / zn)
    return torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), dim=1)


def lab_to_rgb(lab: Tensor) -> Tensor:
    l, a, b = lab.unbind(dim=1)
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    def finv(t: Tensor) -> Tensor:
        return torch.where(t > delta, t.pow(3.0), 3.0 * delta * delta * (t - 4.0 / 29.0))

    xn, yn, zn = 0.95047, 1.0, 1.08883
    x = xn * finv(fx)
    y = yn * finv(fy)
    z = zn * finv(fz)

    linear = torch.stack(
        (
            3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
            -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
            0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
        ),
        dim=1,
    )
    return linear_to_srgb(linear)

def soft_gamut_compress(rgb: Tensor) -> Tensor:
    return rgb.clamp(0.0, 1.0)


def image_stats_from_lab(src_lab: Tensor, out_lab: Tensor, out_rgb: Tensor) -> Dict[str, float]:
    delta = out_lab - src_lab
    chroma_src = src_lab[:, 1:3].norm(dim=1)
    chroma_out = out_lab[:, 1:3].norm(dim=1)
    compressed = ((out_rgb <= 0.001) | (out_rgb >= 0.999)).float().mean()
    return {
        "mean_luma_delta": float(delta[:, 0].mean().detach().cpu()),
        "mean_abs_luma_delta": float(delta[:, 0].abs().mean().detach().cpu()),
        "mean_chroma_delta": float((chroma_out - chroma_src).mean().detach().cpu()),
        "mean_abs_chroma_delta": float((chroma_out - chroma_src).abs().mean().detach().cpu()),
        "compressed_pixel_ratio": float(compressed.detach().cpu()),
    }


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def _require_file(path: str | Path, label: str) -> Path:
    p = _resolve_project_path(path)
    if not p.exists():
        raise RuntimeError(
            f"Missing Neural Preset {label}: {p}\n"
            f"Download the checkpoint from {NEURAL_PRESET_CKPT_SOURCE} and place it at:\n"
            f"  {ROOT / DEFAULT_NEURAL_PRESET_CKPT}"
        )
    return p


def _install_omegaconf_pickle_stubs() -> None:
    """Allow loading the selected Lightning checkpoint without importing OmegaConf."""
    if "omegaconf" in sys.modules:
        return
    module_names = [
        "omegaconf",
        "omegaconf.listconfig",
        "omegaconf.dictconfig",
        "omegaconf.base",
        "omegaconf.nodes",
    ]
    modules = {name: types.ModuleType(name) for name in module_names}
    for name, module in modules.items():
        sys.modules[name] = module

    class ListConfig(list):
        pass

    class DictConfig(dict):
        pass

    class Container:
        pass

    class ContainerMetadata:
        pass

    class Metadata:
        pass

    class AnyNode:
        pass

    modules["omegaconf"].listconfig = modules["omegaconf.listconfig"]  # type: ignore[attr-defined]
    modules["omegaconf"].dictconfig = modules["omegaconf.dictconfig"]  # type: ignore[attr-defined]
    modules["omegaconf"].base = modules["omegaconf.base"]  # type: ignore[attr-defined]
    modules["omegaconf"].nodes = modules["omegaconf.nodes"]  # type: ignore[attr-defined]
    modules["omegaconf.listconfig"].ListConfig = ListConfig  # type: ignore[attr-defined]
    modules["omegaconf.dictconfig"].DictConfig = DictConfig  # type: ignore[attr-defined]
    modules["omegaconf.base"].Container = Container  # type: ignore[attr-defined]
    modules["omegaconf.base"].ContainerMetadata = ContainerMetadata  # type: ignore[attr-defined]
    modules["omegaconf.base"].Metadata = Metadata  # type: ignore[attr-defined]
    modules["omegaconf.nodes"].AnyNode = AnyNode  # type: ignore[attr-defined]


class NeuralPresetDNCM(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        if EfficientNet is None:
            raise RuntimeError(
                "efficientnet_pytorch is required for Neural Preset global matching. "
                "Install it with `python3 -m pip install efficientnet_pytorch`."
            ) from _EFFICIENTNET_IMPORT_ERROR
        self.k = int(k)
        self.style_encoder = EfficientNet.from_name("efficientnet-b0", num_classes=(self.k**2) * 2)
        self.transform_p = nn.Parameter(torch.rand(3, self.k))
        self.transform_q = nn.Parameter(torch.rand(self.k, 3))

    def get_r_and_d(self, thumbnail: Tensor) -> tuple[Tensor, Tensor]:
        params = self.style_encoder(thumbnail)
        r = params[:, : self.k**2].reshape(-1, self.k, self.k)
        d = params[:, self.k**2 :].reshape(-1, self.k, self.k)
        return r, d

    def dncm_matrix(self, params: Tensor) -> Tensor:
        p = self.transform_p.unsqueeze(0).expand(params.shape[0], -1, -1)
        q = self.transform_q.unsqueeze(0).expand(params.shape[0], -1, -1)
        return torch.bmm(torch.bmm(p, params), q)

    def fullres_dncm(self, image: Tensor, matrix: Tensor, chunk_pixels: int) -> Tensor:
        batch, channels, height, width = image.shape
        pixels = image.reshape(batch, channels, height * width)
        outputs = []
        chunk = max(4096, int(chunk_pixels))
        for start in range(0, height * width, chunk):
            outputs.append(torch.bmm(matrix, pixels[:, :, start : start + chunk]))
        return torch.cat(outputs, dim=2).reshape(batch, channels, height, width)

    def forward(
        self,
        content: Tensor,
        content_thumbnail: Tensor,
        style_thumbnail: Tensor,
        chunk_pixels: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _, d_c = self.get_r_and_d(content_thumbnail)
        r_s, _ = self.get_r_and_d(style_thumbnail)
        normalize_matrix = self.dncm_matrix(d_c)
        stylize_matrix = self.dncm_matrix(r_s)
        normalized = self.fullres_dncm(content, normalize_matrix, chunk_pixels)
        stylized = self.fullres_dncm(normalized, stylize_matrix, chunk_pixels)
        return normalized, stylized, normalize_matrix, stylize_matrix


def _extract_state_dict(checkpoint: Any) -> Dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint
    raise RuntimeError("Neural Preset checkpoint does not contain a PyTorch state dict.")


@lru_cache(maxsize=2)
def _load_neural_preset_model(
    device_name: str,
    k: int,
    checkpoint_path: str,
) -> tuple[NeuralPresetDNCM, Dict[str, Any]]:
    ckpt = _require_file(checkpoint_path, "checkpoint")
    device = torch.device(device_name)
    model = NeuralPresetDNCM(k=int(k))
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        _install_omegaconf_pickle_stubs()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)

    state_dict = _extract_state_dict(state)
    prefixes = ("net.", "model.", "module.")
    filtered: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        filtered[clean_key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Neural Preset checkpoint is incompatible with the DNCM model.\n"
            f"Missing keys: {missing[:12]}{'...' if len(missing) > 12 else ''}\n"
            f"Unexpected keys: {unexpected[:12]}{'...' if len(unexpected) > 12 else ''}"
        )
    model.to(device).eval()
    info = {
        "method": "neural_preset_dncm",
        "checkpoint_path": str(ckpt),
        "checkpoint_source": NEURAL_PRESET_CKPT_SOURCE,
        "k": int(k),
    }
    return model, info


@torch.no_grad()
def run_neural_preset_inference(
    source: Tensor,
    reference: Tensor,
    cfg: GlobalMatchingConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor, Dict[str, Any]]:
    if int(cfg.k) != 16:
        raise ValueError("The selected Neural Preset checkpoint is trained for k=16.")
    model, model_info = _load_neural_preset_model(str(device), int(cfg.k), str(cfg.checkpoint_path))
    thumb_size = max(32, int(cfg.thumbnail_size))
    content_thumbnail = F.interpolate(source, size=(thumb_size, thumb_size), mode="bilinear", align_corners=False)
    style_thumbnail = F.interpolate(reference, size=(thumb_size, thumb_size), mode="bilinear", align_corners=False)
    normalized, stylized, normalize_matrix, stylize_matrix = model(
        source,
        content_thumbnail,
        style_thumbnail,
        int(cfg.fullres_chunk_pixels),
    )
    model_info = {
        **model_info,
        "thumbnail_size": thumb_size,
        "normalize_matrix_det": float(torch.det(normalize_matrix[0]).detach().cpu()),
        "stylize_matrix_det": float(torch.det(stylize_matrix[0]).detach().cpu()),
    }
    return soft_gamut_compress(stylized), soft_gamut_compress(normalized), model_info


def mapping_visual(model: NeuralPresetDNCM, cfg: GlobalMatchingConfig, device: torch.device, size: int = 512) -> np.ndarray:
    grid = torch.linspace(0.0, 1.0, steps=size, device=device)
    rr, gg = torch.meshgrid(grid, grid, indexing="xy")
    bb = torch.full_like(rr, 0.5)
    probe = torch.stack([rr, gg, bb], dim=0).unsqueeze(0)
    identity_thumb = F.interpolate(probe, size=(int(cfg.thumbnail_size), int(cfg.thumbnail_size)), mode="bilinear", align_corners=False)
    _, mapped, _, _ = model(probe, identity_thumb, identity_thumb, int(cfg.fullres_chunk_pixels))
    delta = (mapped - probe).abs().mean(dim=1, keepdim=True)
    delta = delta / delta.amax().clamp_min(1e-6)
    return delta.detach().cpu().numpy()[0, 0].astype(np.float32)


def run_global_matching(
    source_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    config: Optional[GlobalMatchingConfig | Dict[str, Any]] = None,
    *,
    source_rgb_np: Optional[np.ndarray] = None,
    reference_rgb_np: Optional[np.ndarray] = None,
    extra_paths: Optional[Dict[str, str]] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config if isinstance(config, GlobalMatchingConfig) else GlobalMatchingConfig(**(config or {}))
    device = select_torch_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    source_np = source_rgb_np if source_rgb_np is not None else load_rgb(source_path)
    reference_np = reference_rgb_np if reference_rgb_np is not None else load_rgb(reference_path)
    source = to_nchw(source_np, device)
    reference = to_nchw(reference_np, device)
    reference_resized = resize_to_hw(reference, source.shape[2], source.shape[3])
    source_fit = resize_long_edge(source, cfg.fit_long_edge)
    timings["load_and_resize"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    base_rgb, normalized_rgb, model_info = run_neural_preset_inference(source, reference, cfg, device)
    timings["neural_preset_inference"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    base_lab = rgb_to_lab(base_rgb)
    source_lab = rgb_to_lab(source)
    reference_resized_lab = rgb_to_lab(reference_resized)
    timings["lab_tensors"] = time.perf_counter() - t2

    paths = {
        "base_intermediate": str(output_dir / "base_intermediate.png"),
        "reference_resized": str(output_dir / "reference_resized.png"),
        "source_fit": str(output_dir / "source_fit.png"),
        "lut_support": str(output_dir / "lut_support.png"),
        "metrics": str(output_dir / "metrics.json"),
    }
    if extra_paths:
        paths.update(extra_paths)
    t3 = time.perf_counter()
    save_rgb(paths["base_intermediate"], to_hwc_np(base_rgb))
    save_rgb(paths["reference_resized"], to_hwc_np(reference_resized))
    save_rgb(paths["source_fit"], to_hwc_np(source_fit))
    model, _ = _load_neural_preset_model(str(device), int(cfg.k), str(cfg.checkpoint_path))
    save_gray(paths["lut_support"], mapping_visual(model, cfg, device))
    timings["save_outputs"] = time.perf_counter() - t3

    stats = image_stats_from_lab(source_lab, base_lab, base_rgb)
    metrics: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "global_stage_version": GLOBAL_STAGE_VERSION,
        "device": str(device),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "source_shape": list(source_np.shape),
        "reference_shape": list(reference_np.shape),
        "fit_shape": [int(source_fit.shape[2]), int(source_fit.shape[3]), 3],
        "config": asdict(cfg),
        "neural_preset_model": model_info,
        "lut_support_coverage": float((base_rgb - source).abs().mean().detach().cpu()),
        "normalized_mean": float(normalized_rgb.mean().detach().cpu()),
        "timings": timings,
        **stats,
        "paths": paths,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    Path(paths["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["tensors"] = {
        "base_intermediate_lab": base_lab,
        "base_intermediate_rgb": base_rgb,
        "reference_resized_lab": reference_resized_lab,
        "reference_resized_rgb": reference_resized,
        "source_lab": source_lab,
        "source_rgb": source,
    }
    return metrics
