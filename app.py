#!/usr/bin/env python3
"""Local Gradio UI for LookAlign V0.3.6."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from scripts.global_matching import GlobalMatchingConfig, run_global_matching
from scripts.local_matching import LocalMatchingConfig, run_local_diffuse_matching


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "inputs" / "source.png"
DEFAULT_REFERENCE = ROOT / "inputs" / "reference.png"
OUTPUTS_DIR = ROOT / "outputs"


def existing_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def choose_input(path: Optional[str], default: Path, label: str) -> str:
    if path:
        return path
    if default.exists():
        return str(default)
    raise gr.Error(f"Please upload a {label} image.")


def run_ui(source_path: Optional[str], reference_path: Optional[str]) -> Tuple[str, ...]:
    source = choose_input(source_path, DEFAULT_SOURCE, "source")
    reference = choose_input(reference_path, DEFAULT_REFERENCE, "reference")
    run_dir = OUTPUTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # V0.3.6 uses locked global and local matching parameters.
    # The UI intentionally exposes no tuning sliders while the stages stabilize.
    global_metrics = run_global_matching(source, reference, run_dir, GlobalMatchingConfig())
    tensors = global_metrics["tensors"]
    metrics = run_local_diffuse_matching(
        base_intermediate_lab=tensors["base_intermediate_lab"],
        base_intermediate_rgb=tensors["base_intermediate_rgb"],
        reference_resized_lab=tensors["reference_resized_lab"],
        reference_resized_rgb=tensors["reference_resized_rgb"],
        source_lab=tensors["source_lab"],
        source_rgb=tensors["source_rgb"],
        output_dir=run_dir,
        global_metrics=global_metrics,
        config=LocalMatchingConfig(),
    )
    paths = metrics["paths"]
    return (
        paths["base_intermediate"],
        paths["final_output"],
        paths["reference_resized"],
        paths["lightglue_matches"],
        paths["filtered_match_confidence"],
        paths["diffuse_luma_delta"],
        paths["diffuse_hue_delta"],
        paths["diffuse_chroma_scale"],
        paths["diffuse_confidence"],
        paths["match_density"],
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LookAlign V0.3.6") as demo:
        gr.Markdown("# LookAlign V0.3.6")

        with gr.Row():
            source_image = gr.Image(label="Source image", type="filepath", value=existing_path(DEFAULT_SOURCE), interactive=True)
            reference_image = gr.Image(label="Reference image", type="filepath", value=existing_path(DEFAULT_REFERENCE), interactive=True)

        run_button = gr.Button("Run V0.3.6", variant="primary")

        gr.Markdown("## Outputs")
        with gr.Row():
            base_intermediate = gr.Image(label="Base intermediate (after 3D LUT)", type="filepath")
            final_output = gr.Image(label="Final output", type="filepath")

        gr.Markdown("## Reference And Alignment")
        with gr.Row():
            reference_resized = gr.Image(label="Reference resized to base geometry", type="filepath")
            lightglue_matches = gr.Image(label="LightGlue matches", type="filepath")
            filtered_match_confidence = gr.Image(label="Aligned match confidence", type="filepath")

        gr.Markdown("## Applied Local Maps")
        with gr.Row():
            diffuse_luma_delta = gr.Image(label="Diffuse luma delta", type="filepath")
            diffuse_hue_delta = gr.Image(label="Diffuse hue delta", type="filepath")
            diffuse_chroma_scale = gr.Image(label="Diffuse chroma scale", type="filepath")
        with gr.Row():
            diffuse_confidence = gr.Image(label="Diffuse confidence", type="filepath")
            match_density = gr.Image(label="Match density", type="filepath")

        run_button.click(
            fn=run_ui,
            inputs=[source_image, reference_image],
            outputs=[
                base_intermediate,
                final_output,
                reference_resized,
                lightglue_matches,
                filtered_match_confidence,
                diffuse_luma_delta,
                diffuse_hue_delta,
                diffuse_chroma_scale,
                diffuse_confidence,
                match_density,
            ],
            queue=True,
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
