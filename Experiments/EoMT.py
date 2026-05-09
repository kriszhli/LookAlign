import gc
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from transformers import EomtDinov3ForUniversalSegmentation
from transformers.models.eomt.image_processing_eomt import EomtImageProcessor


OVERLAY_ALPHA = 0.45
MODEL_ID = "tue-mps/eomt-dinov3-ade-semantic-large-512"
PROCESSOR_SIZE = {"shortest_edge": 512, "longest_edge": None}

ADE20K_LABELS = [
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
]


@dataclass
class EomtArtifacts:
    overlay_map: Image.Image
    label_map: np.ndarray
    label_summary: str
    source_label: str


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _build_palette(num_labels: int) -> np.ndarray:
    indices = np.arange(num_labels, dtype=np.uint32)
    palette = np.stack(
        (
            (indices * 37 + 23) % 256,
            (indices * 67 + 59) % 256,
            (indices * 97 + 101) % 256,
        ),
        axis=1,
    )
    return palette.astype(np.uint8)


ADE20K_PALETTE = _build_palette(len(ADE20K_LABELS))


def _format_label_summary(label_map: np.ndarray, top_k: int = 6) -> str:
    counts = np.bincount(label_map.reshape(-1), minlength=len(ADE20K_LABELS))
    total_pixels = int(counts.sum())
    if total_pixels <= 0:
        return "Labels: none"

    ranked = [(index, int(count)) for index, count in enumerate(counts) if count > 0]
    ranked.sort(key=lambda item: item[1], reverse=True)

    lines = ["Labels:"]
    for index, count in ranked[:top_k]:
        percentage = 100.0 * count / total_pixels
        lines.append(f"{ADE20K_LABELS[index].strip()}: {percentage:.1f}%")
    return "\n".join(lines)


def _render_label_map(label_map: np.ndarray, output_size: Tuple[int, int]) -> Image.Image:
    colors = ADE20K_PALETTE[label_map]
    semantic_image = Image.fromarray(colors.astype(np.uint8), mode="RGB")
    return semantic_image.resize(output_size, Image.Resampling.NEAREST)


def _overlay_semantics(base_image: Image.Image, semantic_image: Image.Image, alpha: float = OVERLAY_ALPHA) -> Image.Image:
    base = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    overlay = np.asarray(semantic_image.convert("RGB"), dtype=np.float32)
    blended = np.clip(base * (1.0 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


class EomtSegmenter:
    def __init__(self) -> None:
        self.processor = EomtImageProcessor(size=PROCESSOR_SIZE, do_pad=False, do_split_image=True)
        self._model: EomtDinov3ForUniversalSegmentation | None = None
        self._device_type: str | None = None

    def _reset_loaded_state(self) -> None:
        if self._model is None:
            return
        old_device = torch.device(self._device_type or "cpu")
        del self._model
        self._model = None
        self._device_type = None
        gc.collect()
        _empty_device_cache(old_device)

    def _load_model(self, device: torch.device) -> EomtDinov3ForUniversalSegmentation:
        if self._model is not None and self._device_type == device.type:
            return self._model

        self._reset_loaded_state()
        self._model = EomtDinov3ForUniversalSegmentation.from_pretrained(MODEL_ID, local_files_only=True).eval().to(device)
        self._device_type = device.type
        return self._model

    def segment(self, image: Image.Image, output_size: Tuple[int, int], device: torch.device) -> EomtArtifacts:
        model = self._load_model(device)
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)

        predictions = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(image.height, image.width)],
            size=PROCESSOR_SIZE,
        )[0]

        if torch.is_tensor(predictions):
            label_map = predictions.to("cpu").numpy().astype(np.int32)
        else:
            label_map = np.asarray(predictions, dtype=np.int32)

        base_image = image.convert("RGB").resize(output_size, Image.Resampling.BICUBIC)
        label_image = _render_label_map(label_map, output_size)
        overlay_map = _overlay_semantics(base_image, label_image)
        label_summary = _format_label_summary(label_map)
        return EomtArtifacts(
            overlay_map=overlay_map,
            label_map=label_map,
            label_summary=label_summary,
            source_label=f"Checkpoint: {MODEL_ID}",
        )
