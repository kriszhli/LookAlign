#!/usr/bin/env python3
"""Local Gradio UI for the LookAlign MVP."""

from __future__ import annotations

import base64
import html
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import gradio as gr

from scripts.lookalign_mvp import run_lookalign


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "inputs" / "source.png"
DEFAULT_REFERENCE = ROOT / "inputs" / "reference.png"
OUTPUTS_DIR = ROOT / "outputs"

IMAGE_OPTIONS = ["Source", "Reference", "LookAlign Output"]


def existing_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def choose_input(path: Optional[str], default: Path, label: str) -> str:
    if path:
        return path
    if default.exists():
        return str(default)
    raise gr.Error(f"Please upload a {label} image.")


def image_data_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_aspect_ratio(path: Optional[str]) -> float:
    if not path:
        return 4.0 / 3.0
    p = Path(path)
    if not p.exists():
        return 4.0 / 3.0
    try:
        from PIL import Image

        with Image.open(p) as img:
            w, h = img.size
        if w > 0 and h > 0:
            return float(w) / float(h)
    except Exception:
        pass
    return 4.0 / 3.0


def build_comparison_html(paths: Dict[str, Optional[str]], left_name: str, right_name: str, aspect_ratio: float) -> str:
    left_url = image_data_url(paths.get(left_name))
    right_url = image_data_url(paths.get(right_name))
    if not left_url or not right_url:
        return (
            '<div class="lookalign-empty">'
            "Run LookAlign to compare Source, Reference, and LookAlign Output."
            "</div>"
        )

    slider_id = f"lookalign-compare-{uuid4().hex}"
    left_label = html.escape(left_name)
    right_label = html.escape(right_name)
    return f"""
<div id="{slider_id}" class="lookalign-compare" style="--pos: 50%; --aspect: {aspect_ratio:.6f};">
  <div class="lookalign-compare-stage">
    <img class="lookalign-compare-img" src="{right_url}" alt="{right_label}">
    <div class="lookalign-compare-before">
      <img class="lookalign-compare-img" src="{left_url}" alt="{left_label}">
    </div>
    <div class="lookalign-compare-handle" aria-hidden="true"></div>
    <div class="lookalign-compare-label lookalign-compare-left">{left_label}</div>
    <div class="lookalign-compare-label lookalign-compare-right">{right_label}</div>
    <input
      class="lookalign-compare-range"
      type="range"
      min="0"
      max="100"
      value="50"
      aria-label="Before after comparison slider"
    >
  </div>
</div>
"""


def comparison_from_state(paths: Dict[str, Optional[str]], left_name: str, right_name: str) -> str:
    paths = paths or {}
    return build_comparison_html(paths, left_name, right_name, image_aspect_ratio(paths.get(left_name) or paths.get(right_name)))


def run_ui(
    source_path: Optional[str],
    reference_path: Optional[str],
    align_mode: str,
    local_strength: float,
    local_luma_strength: float,
    detail_strength: float,
    blur_sigma: float,
    grid: int,
    trust_threshold: float,
    min_luma_std_ratio: float,
    min_saturation_ratio: float,
    anti_fade_strength: float,
    compare_a: str,
    compare_b: str,
) -> tuple[Any, ...]:
    source = choose_input(source_path, DEFAULT_SOURCE, "source")
    reference = choose_input(reference_path, DEFAULT_REFERENCE, "reference")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = OUTPUTS_DIR / run_id
    output_path = run_dir / "lookalign_output.png"
    debug_dir = run_dir / "debug"
    config = {
        "debug_dir": str(debug_dir),
        "align": align_mode,
        "local_strength": float(local_strength),
        "local_luma_strength": float(local_luma_strength),
        "detail_strength": float(detail_strength),
        "blur_sigma": float(blur_sigma),
        "grid": int(grid),
        "trust_threshold": float(trust_threshold),
        "min_luma_std_ratio": float(min_luma_std_ratio),
        "min_saturation_ratio": float(min_saturation_ratio),
        "anti_fade_strength": float(anti_fade_strength),
    }

    result = run_lookalign(source, reference, output_path, config)
    debug_paths = result["debug_paths"]
    paths = {
        "Source": source,
        "Reference": reference,
        "LookAlign Output": result["output_path"],
    }
    aspect_ratio = image_aspect_ratio(source)

    return (
        build_comparison_html(paths, compare_a, compare_b, aspect_ratio),
        debug_paths.get("aligned_reference"),
        debug_paths.get("trust_map"),
        debug_paths.get("overlap_mask"),
        result["output_path"],
        paths,
    )


def initial_paths() -> Dict[str, Optional[str]]:
    source = existing_path(DEFAULT_SOURCE)
    reference = existing_path(DEFAULT_REFERENCE)
    output = existing_path(OUTPUTS_DIR / "output.png")
    return {
        "Source": source,
        "Reference": reference,
        "LookAlign Output": output,
    }


CSS = """
.lookalign-compare-stage {
  position: relative;
  overflow: hidden;
  width: 100%;
  aspect-ratio: var(--aspect, 4 / 3);
  max-height: 80vh;
  background: #111827;
  border: 1px solid #d1d5db;
  border-radius: 8px;
}
.lookalign-compare-img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #111827;
}
.lookalign-compare-before {
  position: absolute;
  inset: 0;
  clip-path: inset(0 calc(100% - var(--pos)) 0 0);
  border-right: 2px solid #ffffff;
}
.lookalign-compare-handle {
  position: absolute;
  top: 0;
  bottom: 0;
  left: var(--pos);
  width: 2px;
  background: #ffffff;
  box-shadow: 0 0 0 1px rgba(0,0,0,0.25);
}
.lookalign-compare-handle::after {
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  width: 20px;
  height: 20px;
  transform: translate(-50%, -50%);
  border: 2px solid #ffffff;
  border-radius: 999px;
  background: rgba(17, 24, 39, 0.7);
}
.lookalign-compare-range {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: ew-resize;
}
.lookalign-compare-label {
  position: absolute;
  top: 12px;
  padding: 4px 8px;
  border-radius: 6px;
  background: rgba(17, 24, 39, 0.72);
  color: #ffffff;
  font-size: 12px;
  line-height: 1.2;
  pointer-events: none;
}
.lookalign-compare-left {
  left: 12px;
}
.lookalign-compare-right {
  right: 12px;
}
.lookalign-empty {
  padding: 18px;
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
  color: #475569;
  background: #f8fafc;
}
"""

COMPARISON_HEAD = """
<script>
document.addEventListener("input", function (event) {
  const target = event.target;
  if (!target || !target.classList || !target.classList.contains("lookalign-compare-range")) {
    return;
  }
  const wrapper = target.closest(".lookalign-compare");
  if (wrapper) {
    wrapper.style.setProperty("--pos", target.value + "%");
  }
});
</script>
"""


def build_app() -> gr.Blocks:
    defaults = initial_paths()
    with gr.Blocks(title="LookAlign MVP") as demo:
        gr.Markdown("# LookAlign MVP")
        state = gr.State(defaults)

        with gr.Row():
            source_image = gr.Image(
                label="Source image",
                type="filepath",
                value=defaults["Source"],
                interactive=True,
            )
            reference_image = gr.Image(
                label="Reference image",
                type="filepath",
                value=defaults["Reference"],
                interactive=True,
            )

        with gr.Row():
            align_mode = gr.Dropdown(
                label="Align mode",
                choices=["auto", "identity", "center"],
                value="auto",
            )
            local_strength = gr.Slider(
                0.0,
                1.0,
                value=0.8,
                step=0.01,
                label="local_strength",
                info="How strongly local color corrections are blended in.",
            )
            local_luma_strength = gr.Slider(
                0.0,
                1.0,
                value=0.25,
                step=0.01,
                label="local_luma_strength",
                info="How strongly local corrections may change luminance.",
            )
            detail_strength = gr.Slider(
                0.0,
                2.0,
                value=1.0,
                step=0.01,
                label="detail_strength",
                info="How much source texture and luminance detail is restored.",
            )

        with gr.Row():
            blur_sigma = gr.Slider(
                0.0,
                80.0,
                value=24.0,
                step=1.0,
                label="blur_sigma",
                info="Blur radius used to compare broad color and lighting.",
            )
            grid = gr.Slider(
                2,
                64,
                value=16,
                step=1,
                label="grid",
                info="Number of rows used for local correction patches.",
            )
            trust_threshold = gr.Slider(
                0.0,
                1.0,
                value=0.15,
                step=0.01,
                label="trust_threshold",
                info="Minimum alignment trust needed before pixels affect fitting.",
            )

        with gr.Row():
            min_luma_std_ratio = gr.Slider(
                0.0,
                1.25,
                value=0.85,
                step=0.01,
                label="min_luma_std_ratio",
                info="Minimum output/source luminance contrast ratio.",
            )
            min_saturation_ratio = gr.Slider(
                0.0,
                1.25,
                value=0.80,
                step=0.01,
                label="min_saturation_ratio",
                info="Minimum output/source saturation ratio.",
            )
            anti_fade_strength = gr.Slider(
                0.0,
                1.0,
                value=1.0,
                step=0.01,
                label="anti_fade_strength",
                info="Strength of final contrast and saturation guard.",
            )

        run_button = gr.Button("Run LookAlign", variant="primary")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    compare_a = gr.Dropdown(IMAGE_OPTIONS, value="Source", label="Comparison A")
                    compare_b = gr.Dropdown(IMAGE_OPTIONS, value="LookAlign Output", label="Comparison B")
                comparison = gr.HTML(value=build_comparison_html(defaults, "Source", "LookAlign Output", image_aspect_ratio(defaults["Source"])))

        with gr.Row():
            aligned_reference = gr.Image(label="Aligned reference image", type="filepath")
            trust_map = gr.Image(label="Trust map", type="filepath")
            overlap_mask = gr.Image(label="Overlap mask", type="filepath")

        download = gr.File(label="Download output PNG")

        run_button.click(
            fn=run_ui,
            inputs=[
                source_image,
                reference_image,
                align_mode,
                local_strength,
                local_luma_strength,
                detail_strength,
                blur_sigma,
                grid,
                trust_threshold,
                min_luma_std_ratio,
                min_saturation_ratio,
                anti_fade_strength,
                compare_a,
                compare_b,
            ],
            outputs=[
                comparison,
                aligned_reference,
                trust_map,
                overlap_mask,
                download,
                state,
            ],
            queue=False,
        )
        compare_a.change(comparison_from_state, inputs=[state, compare_a, compare_b], outputs=comparison, queue=False)
        compare_b.change(comparison_from_state, inputs=[state, compare_a, compare_b], outputs=comparison, queue=False)

    return demo


if __name__ == "__main__":
    build_app().launch(css=CSS, head=COMPARISON_HEAD)
