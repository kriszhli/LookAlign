import gc
import resource
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import timm
import torch
import torch.nn.functional as F

try:
    from .EoMT import EomtSegmenter
except ImportError:
    from EoMT import EomtSegmenter


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_IMAGE_PATH = ROOT_DIR / "inputs" / "reference2.png"

PATCH_SIZE = 16
TARGET_SHORT_SIDE = 1200
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
MANUAL_CLUSTER_MAX = 40
VRAM_LIMIT_GIB = 8.0
VRAM_LIMIT_BYTES = int(VRAM_LIMIT_GIB * (1024 ** 3))

METHOD_EOMT = "eomt"
METHOD_KMEANS = "kmeans"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_name: str
    checkpoint_path: Path
    target_short_side: int


@dataclass
class DeviceSelection:
    device: torch.device | None
    status_lines: List[str]
    requires_cpu_confirmation: bool
    used_cpu_confirmation: bool


@dataclass
class RunArtifacts:
    status: str
    overlay_map: Image.Image
    elapsed: float
    requires_cpu_confirmation: bool = False
    device_type: str = "unknown"


MODEL_SPEC = ModelSpec(
    key="large",
    label="ViT-L/16",
    model_name="vit_large_patch16_dinov3",
    checkpoint_path=ROOT_DIR / "ckpts" / "dinov3_vitl16_pretrain_lvd1689m.pth",
    target_short_side=640,
)


def normalize_method(method: str) -> str:
    normalized = (method or METHOD_EOMT).strip().lower()
    if normalized not in {METHOD_EOMT, METHOD_KMEANS}:
        raise ValueError(f"Unsupported method: {method}")
    return normalized


def select_device(allow_cpu: bool) -> DeviceSelection:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return DeviceSelection(
            device=torch.device("mps"),
            status_lines=["Device: MPS"],
            requires_cpu_confirmation=False,
            used_cpu_confirmation=False,
        )
    if torch.cuda.is_available():
        return DeviceSelection(
            device=torch.device("cuda"),
            status_lines=["Device: CUDA"],
            requires_cpu_confirmation=False,
            used_cpu_confirmation=False,
        )
    if allow_cpu:
        return DeviceSelection(
            device=torch.device("cpu"),
            status_lines=[
                "Device: CPU",
                "CPU fallback approved by user because MPS/CUDA is unavailable.",
            ],
            requires_cpu_confirmation=False,
            used_cpu_confirmation=True,
        )
    return DeviceSelection(
        device=None,
        status_lines=[
            "Device: unavailable",
            "MPS and CUDA are unavailable.",
            "Confirm CPU execution and run again to continue.",
        ],
        requires_cpu_confirmation=True,
        used_cpu_confirmation=False,
    )


def format_gib(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def process_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def collect_memory_stats(device: torch.device) -> dict:
    stats = {"process_peak_rss": process_peak_rss_bytes()}
    if device.type == "cuda":
        stats["device_allocated"] = int(torch.cuda.memory_allocated())
        stats["device_reserved"] = int(torch.cuda.memory_reserved())
        stats["device_peak_allocated"] = int(torch.cuda.max_memory_allocated())
    elif device.type == "mps":
        stats["device_allocated"] = int(torch.mps.current_allocated_memory())
        stats["device_driver_allocated"] = int(torch.mps.driver_allocated_memory())
        stats["device_recommended_max"] = int(torch.mps.recommended_max_memory())
    return stats


def format_memory_lines(before: dict, after: dict, device: torch.device) -> List[str]:
    lines = [f"Process peak RSS: {format_gib(after['process_peak_rss'])}"]
    if device.type == "cuda":
        lines.append(
            "CUDA allocated/reserved/peak: "
            f"{format_gib(after['device_allocated'])} / "
            f"{format_gib(after['device_reserved'])} / "
            f"{format_gib(after['device_peak_allocated'])}"
        )
        lines.append(
            "CUDA allocated delta: "
            f"{format_gib(after['device_allocated'] - before.get('device_allocated', 0))}"
        )
    elif device.type == "mps":
        lines.append(
            "MPS current/driver allocated: "
            f"{format_gib(after['device_allocated'])} / "
            f"{format_gib(after['device_driver_allocated'])}"
        )
        lines.append(
            "MPS allocated delta: "
            f"{format_gib(after['device_allocated'] - before.get('device_allocated', 0))}"
        )
        lines.append(f"MPS recommended max: {format_gib(after['device_recommended_max'])}")
    return lines


def make_error_image(size: Tuple[int, int]) -> Image.Image:
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[..., 0] = 96
    canvas[..., 1] = 16
    canvas[..., 2] = 16
    return Image.fromarray(canvas, mode="RGB")


def resize_for_model(image: Image.Image, target_short_side: int = TARGET_SHORT_SIDE) -> Image.Image:
    width, height = image.size
    scale = target_short_side / min(width, height)
    resized_w = max(PATCH_SIZE, int(round(width * scale)))
    resized_h = max(PATCH_SIZE, int(round(height * scale)))
    if resized_w <= resized_h:
        resized_w = target_short_side
        resized_h = max(PATCH_SIZE, int(round(height * (target_short_side / width))))
    else:
        resized_h = target_short_side
        resized_w = max(PATCH_SIZE, int(round(width * (target_short_side / height))))
    resized_w = max(PATCH_SIZE, int(round(resized_w / PATCH_SIZE)) * PATCH_SIZE)
    resized_h = max(PATCH_SIZE, int(round(resized_h / PATCH_SIZE)) * PATCH_SIZE)
    return image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)


def preprocess_image(resized: Image.Image, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor - mean) / std


def remap_checkpoint_state(raw_state: dict) -> dict:
    state = {}
    for key, value in raw_state.items():
        if key == "storage_tokens":
            state["reg_token"] = value
        elif key.endswith(".ls1.gamma"):
            state[key.replace(".ls1.gamma", ".gamma_1")] = value
        elif key.endswith(".ls2.gamma"):
            state[key.replace(".ls2.gamma", ".gamma_2")] = value
        elif key in {"mask_token", "rope_embed.periods"}:
            continue
        elif key.endswith(".attn.qkv.bias") or key.endswith(".attn.qkv.bias_mask"):
            continue
        else:
            state[key] = value
    return state


def load_model(spec: ModelSpec) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor, int]:
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint_path}")

    model = timm.create_model(spec.model_name, pretrained=False)
    raw_state = torch.load(spec.checkpoint_path, map_location="cpu")
    if not isinstance(raw_state, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(raw_state).__name__}")

    state = remap_checkpoint_state(raw_state)
    model.load_state_dict(state, strict=False)

    data_cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(data_cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg["std"], dtype=torch.float32).view(1, 3, 1, 1)
    return model.eval(), mean, std, model.num_prefix_tokens


def extract_patch_features(
    spec: ModelSpec,
    image: Image.Image,
    device: torch.device,
    manual_clusters: int,
) -> Tuple[torch.Tensor, np.ndarray, Image.Image, int, int, int, str]:
    resized = resize_for_model(image.convert("RGB"), target_short_side=spec.target_short_side)
    model = None
    try:
        model, mean, std, num_prefix_tokens = load_model(spec)
        tensor = preprocess_image(resized, mean, std).to(device)
        model = model.to(device)
        with torch.inference_mode():
            tokens = model.forward_features(tensor)

        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError(f"Unexpected forward_features output: {type(tokens).__name__}")

        grid_h = resized.size[1] // PATCH_SIZE
        grid_w = resized.size[0] // PATCH_SIZE
        patch_tokens = tokens[0, num_prefix_tokens:, :]
        expected_patch_tokens = grid_h * grid_w
        if patch_tokens.shape[0] != expected_patch_tokens:
            raise RuntimeError(
                f"Patch/token mismatch: got {patch_tokens.shape[0]} patch tokens, expected {expected_patch_tokens}"
            )

        patch_features = F.normalize(patch_tokens.float(), dim=1).cpu().numpy()
        cluster_count, cluster_mode = resolve_cluster_count(patch_features.shape[0], manual_clusters)
        return tokens, patch_features, resized, grid_h, grid_w, cluster_count, cluster_mode
    finally:
        del model
        empty_device_cache(device)
        gc.collect()


def choose_cluster_count(num_patches: int) -> int:
    heuristic = int(round(np.sqrt(num_patches) / 3.0))
    return max(MIN_CLUSTERS, min(MAX_CLUSTERS, heuristic))


def resolve_cluster_count(num_patches: int, manual_clusters: int) -> Tuple[int, str]:
    if manual_clusters > 0:
        cluster_count = min(MANUAL_CLUSTER_MAX, max(1, int(manual_clusters)))
        return min(cluster_count, num_patches), "manual"
    return choose_cluster_count(num_patches), "automatic"


class DinoRunner:
    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec
        self.eomt = EomtSegmenter()

    def _empty_outputs(self, size: Tuple[int, int]) -> Tuple[Image.Image, ...]:
        return (make_error_image(size),)

    def _configure_device_limit(self, device: torch.device) -> List[str]:
        lines = [f"VRAM budget: {VRAM_LIMIT_GIB:.2f} GiB"]
        if device.type == "mps":
            recommended = int(torch.mps.recommended_max_memory())
            fraction = min(1.0, VRAM_LIMIT_BYTES / max(1, recommended))
            torch.mps.set_per_process_memory_fraction(fraction)
            lines.append(f"MPS recommended max: {format_gib(recommended)}")
            lines.append(f"MPS per-process limit fraction: {fraction:.3f}")
        elif device.type == "cuda":
            lines.append("CUDA hard VRAM cap is not configured; runtime will report failures if the model exceeds budget.")
        else:
            lines.append("VRAM cap is only relevant on accelerator devices.")
        return lines

    def _is_memory_budget_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return "out of memory" in text or "oom" in text or "mps backend out of memory" in text

    def run(
        self,
        image: Image.Image | None,
        manual_clusters: int = 0,
        method: str = METHOD_EOMT,
        allow_cpu: bool = False,
    ) -> RunArtifacts:
        if image is None:
            image = Image.open(DEFAULT_IMAGE_PATH).convert("RGB")
        else:
            image = image.convert("RGB")

        method = normalize_method(method)
        display_image = resize_for_model(image, target_short_side=self.model_spec.target_short_side)
        selection = select_device(allow_cpu=allow_cpu)
        if selection.device is None:
            return RunArtifacts(
                status="\n".join(selection.status_lines),
                overlay_map=display_image,
                elapsed=0.0,
                requires_cpu_confirmation=True,
                device_type="unavailable",
            )

        return self._run_model(
            image=image,
            display_image=display_image,
            manual_clusters=manual_clusters,
            method=method,
            device=selection.device,
            status_prefix=selection.status_lines,
            used_cpu_confirmation=selection.used_cpu_confirmation,
        )

    def _run_model(
        self,
        image: Image.Image,
        display_image: Image.Image,
        manual_clusters: int,
        method: str,
        device: torch.device,
        status_prefix: List[str],
        used_cpu_confirmation: bool,
    ) -> RunArtifacts:
        limit_lines = self._configure_device_limit(device)
        memory_before = collect_memory_stats(device)
        start_time = time.perf_counter()
        try:
            if method == METHOD_KMEANS:
                method_lines, overlay_map = self._run_kmeans(
                    image=display_image,
                    device=device,
                    manual_clusters=manual_clusters,
                )
            else:
                method_lines, overlay_map = self._run_eomt(
                    image=image,
                    display_size=display_image.size,
                    device=device,
                )

            synchronize_device(device)
            elapsed = time.perf_counter() - start_time
            method_lines.append(f"Time: {elapsed:.2f}s")
            artifact = RunArtifacts(
                status="\n".join(method_lines),
                overlay_map=overlay_map,
                elapsed=elapsed,
                requires_cpu_confirmation=False,
                device_type=device.type,
            )
        except Exception as error:
            if self._is_memory_budget_error(error):
                message = (
                    f"{method.upper()}: skipped because it exceeded the {VRAM_LIMIT_GIB:.2f} GiB device budget."
                )
            else:
                message = f"{method.upper()}: failed: {''.join(traceback.format_exception_only(type(error), error)).strip()}"
            artifact = RunArtifacts(
                status=message,
                overlay_map=make_error_image(display_image.size),
                elapsed=0.0,
                requires_cpu_confirmation=False,
                device_type=device.type,
            )

        memory_after = collect_memory_stats(device)
        summary_lines = list(status_prefix)
        summary_lines.append(f"Method: {method.upper()}")
        if used_cpu_confirmation:
            summary_lines.append("CPU confirmation: approved")
        summary_lines.extend(line for line in limit_lines if "fraction" in line)
        summary_lines.append(artifact.status)
        memory_lines = format_memory_lines(memory_before, memory_after, device)
        if memory_lines:
            summary_lines.append(memory_lines[0])
        if device.type == "mps":
            summary_lines.extend(line for line in memory_lines[1:] if "current/driver" in line)
        return RunArtifacts(
            status="\n".join(message for message in summary_lines if message),
            overlay_map=artifact.overlay_map,
            elapsed=artifact.elapsed,
            requires_cpu_confirmation=artifact.requires_cpu_confirmation,
            device_type=artifact.device_type,
        )

    def _run_kmeans(
        self,
        image: Image.Image,
        device: torch.device,
        manual_clusters: int,
    ) -> Tuple[List[str], Image.Image]:
        from kMeans import build_overlay_from_features

        tokens, patch_features, resized, grid_h, grid_w, cluster_count, cluster_mode = extract_patch_features(
            self.model_spec,
            image,
            device,
            manual_clusters,
        )
        overlay_map = build_overlay_from_features(
            patch_features=patch_features,
            grid_h=grid_h,
            grid_w=grid_w,
            cluster_count=cluster_count,
            base_image=resized,
        )
        lines = [
            f"Model: {self.model_spec.label}",
            f"Size: {resized.size[0]}x{resized.size[1]}",
            f"Grid: {grid_h}x{grid_w}",
            f"Tokens: {tokens.shape[1]}",
            f"Clusters: {cluster_count} ({cluster_mode})",
        ]
        return lines, overlay_map

    def _run_eomt(
        self,
        image: Image.Image,
        display_size: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[List[str], Image.Image]:
        artifacts = self.eomt.segment(image=image, output_size=display_size, device=device)
        label_count = int(np.unique(artifacts.label_map).size)
        lines = [
            "Model: EoMT-DINOv3 Large",
            f"Input: {image.size[0]}x{image.size[1]}",
            f"Display: {display_size[0]}x{display_size[1]}",
            "Processor: 512 split-window ADE semantic map",
            artifacts.source_label,
            f"Predicted classes: {label_count}",
            artifacts.label_summary,
        ]
        return lines, artifacts.overlay_map
