#!/usr/bin/env python3
"""Local Gradio UI for LookAlign V0.2.5."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr

from scripts.lookalign_mvp import run_lookalign


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "inputs" / "source.png"
DEFAULT_REFERENCE = ROOT / "inputs" / "reference.png"
OUTPUTS_DIR = ROOT / "outputs"

IMAGE_OPTIONS = ["Source", "Aligned Reference", "SA-LUT Base", "Light Map Result", "LookAlign Output"]


def existing_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def choose_input(path: Optional[str], default: Path, label: str) -> str:
    if path:
        return path
    if default.exists():
        return str(default)
    raise gr.Error(f"Please upload a {label} image.")


def build_comparison_value(paths: Dict[str, Optional[str]], left_name: str, right_name: str) -> tuple[Optional[str], Optional[str]]:
    paths = paths or {}
    return paths.get(left_name), paths.get(right_name)


def run_ui(
    source_path: Optional[str],
    reference_path: Optional[str],
    align_mode: str,
    local_strength: float,
    local_luma_strength: float,
    detail_strength: float,
    base_radius: float,
    light_map_grid: int,
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
        "base_radius": float(base_radius),
        "light_map_grid": int(light_map_grid),
        "trust_threshold": float(trust_threshold),
        "min_luma_std_ratio": float(min_luma_std_ratio),
        "min_saturation_ratio": float(min_saturation_ratio),
        "anti_fade_strength": float(anti_fade_strength),
    }

    result = run_lookalign(source, reference, output_path, config)
    debug_paths = result["debug_paths"]
    paths = {
        "Source": source,
        "Aligned Reference": debug_paths.get("aligned_reference"),
        "SA-LUT Base": debug_paths.get("sa_lut_base_result"),
        "Light Map Result": debug_paths.get("light_map_result"),
        "LookAlign Output": result["output_path"],
    }

    return (
        build_comparison_value(paths, compare_a, compare_b),
        debug_paths.get("aligned_reference"),
        debug_paths.get("source_base"),
        debug_paths.get("reference_base"),
        debug_paths.get("sa_lut_base_result"),
        debug_paths.get("light_map_coarse"),
        debug_paths.get("light_map_guided"),
        debug_paths.get("light_map_confidence"),
        debug_paths.get("light_map_edge_gate"),
        result["output_path"],
        paths,
    )


def initial_paths() -> Dict[str, Optional[str]]:
    source = existing_path(DEFAULT_SOURCE)
    reference = existing_path(DEFAULT_REFERENCE)
    return {
        "Source": source,
        "Reference": reference,
        "LookAlign Output": None,
    }


def build_app() -> gr.Blocks:
    defaults = initial_paths()
    with gr.Blocks(title="LookAlign V0.2.5") as demo:
        gr.Markdown("# LookAlign V0.2.5")
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
                info="How strongly the source-guided light map is blended in.",
            )
            local_luma_strength = gr.Slider(
                0.0,
                1.0,
                value=0.25,
                step=0.01,
                label="local_luma_strength",
                info="How strongly the light map may change luminance.",
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
            base_radius = gr.Slider(
                0.0,
                80.0,
                value=24.0,
                step=1.0,
                label="base_radius",
                info="Edge-aware base smoothing radius.",
            )
            light_map_grid = gr.Slider(
                2,
                64,
                value=20,
                step=1,
                label="light_map_grid",
                info="Number of rows used for the coarse light/exposure map.",
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
                    compare_a = gr.Dropdown(IMAGE_OPTIONS, value="SA-LUT Base", label="Comparison A")
                    compare_b = gr.Dropdown(IMAGE_OPTIONS, value="LookAlign Output", label="Comparison B")
                comparison = gr.ImageSlider(
                    value=build_comparison_value({"Source": defaults["Source"], "Aligned Reference": defaults["Reference"], "SA-LUT Base": defaults["Reference"], "Light Map Result": defaults["Reference"], "LookAlign Output": defaults["Reference"]}, "SA-LUT Base", "LookAlign Output"),
                    type="filepath",
                    label="Before / after comparison",
                    interactive=False,
                    max_height=700,
                )

        with gr.Row():
            aligned_reference = gr.Image(label="Aligned reference image", type="filepath")
            source_base = gr.Image(label="Source base", type="filepath")
            reference_base = gr.Image(label="Reference base", type="filepath")
            sa_lut_base = gr.Image(label="SA-LUT base result", type="filepath")

        with gr.Row():
            light_map_coarse = gr.Image(label="Coarse light map", type="filepath")
            light_map_guided = gr.Image(label="Guided light map", type="filepath")
            light_map_confidence = gr.Image(label="Light map confidence", type="filepath")
            light_map_edge_gate = gr.Image(label="Light map edge gate", type="filepath")

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
                base_radius,
                light_map_grid,
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
                source_base,
                reference_base,
                sa_lut_base,
                light_map_coarse,
                light_map_guided,
                light_map_confidence,
                light_map_edge_gate,
                download,
                state,
            ],
            queue=False,
        )
        compare_a.change(build_comparison_value, inputs=[state, compare_a, compare_b], outputs=comparison, queue=False)
        compare_b.change(build_comparison_value, inputs=[state, compare_a, compare_b], outputs=comparison, queue=False)

    return demo


if __name__ == "__main__":
    build_app().launch()
