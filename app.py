#!/usr/bin/env python3
"""Local Gradio UI for LookAlign V0.3.5."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from scripts.local_matching import LocalMatchingConfig, run_local_diffuse_matching
from scripts.ot_svd_lut_mps import OTSVDLUTConfig, run_ot_svd_lut_mps


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

    # V0.3.5 uses locked global and local matching parameters.
    # The UI intentionally exposes no tuning sliders while the stages stabilize.
    global_metrics = run_ot_svd_lut_mps(source, reference, run_dir, OTSVDLUTConfig())
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
        paths["final_output"],
        paths["diffuse_luma_delta"],
        paths["diffuse_hue_delta"],
        paths["diffuse_chroma_scale"],
        paths["diffuse_confidence"],
        paths["diffuse_residual_rejection"],
        paths["ref_residual_confidence"],
        paths["specular_confidence"],
        paths["shadow_confidence"],
        paths["hue_stability_confidence"],
        paths["delta_sanity_confidence"],
        paths["luma_confidence"],
        paths["hue_confidence"],
        paths["chroma_confidence"],
        paths["lightglue_matches"],
        paths["filtered_match_confidence"],
        paths["sparse_luma_delta"],
        paths["sparse_hue_delta"],
        paths["sparse_chroma_scale"],
        paths["match_density"],
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LookAlign V0.3.5") as demo:
        gr.Markdown("# LookAlign V0.3.5")

        with gr.Row():
            source_image = gr.Image(label="Source image", type="filepath", value=existing_path(DEFAULT_SOURCE), interactive=True)
            reference_image = gr.Image(label="Reference image", type="filepath", value=existing_path(DEFAULT_REFERENCE), interactive=True)

        run_button = gr.Button("Run V0.3.5", variant="primary")

        final_output = gr.Image(label="Final output", type="filepath")

        gr.Markdown("## Local matching debug maps")
        with gr.Row():
            diffuse_luma_delta = gr.Image(label="Diffuse luma delta", type="filepath")
            diffuse_hue_delta = gr.Image(label="Diffuse hue delta", type="filepath")
            diffuse_chroma_scale = gr.Image(label="Diffuse chroma scale", type="filepath")
        with gr.Row():
            diffuse_confidence = gr.Image(label="Diffuse confidence", type="filepath")
            diffuse_residual_rejection = gr.Image(label="Diffuse residual rejection", type="filepath")
        with gr.Row():
            ref_residual_confidence = gr.Image(label="Reference residual confidence", type="filepath")
            specular_confidence = gr.Image(label="Specular confidence", type="filepath")
            shadow_confidence = gr.Image(label="Shadow confidence", type="filepath")
        with gr.Row():
            hue_stability_confidence = gr.Image(label="Hue stability confidence", type="filepath")
            delta_sanity_confidence = gr.Image(label="Delta sanity confidence", type="filepath")
        with gr.Row():
            luma_confidence = gr.Image(label="Luma confidence", type="filepath")
            hue_confidence = gr.Image(label="Hue confidence", type="filepath")
            chroma_confidence = gr.Image(label="Chroma confidence", type="filepath")
        with gr.Row():
            lightglue_matches = gr.Image(label="LightGlue matches", type="filepath")
            filtered_match_confidence = gr.Image(label="Filtered match confidence", type="filepath")
        with gr.Row():
            sparse_luma_delta = gr.Image(label="Sparse luma delta samples", type="filepath")
            sparse_hue_delta = gr.Image(label="Sparse hue delta samples", type="filepath")
            sparse_chroma_scale = gr.Image(label="Sparse chroma scale samples", type="filepath")
        with gr.Row():
            match_density = gr.Image(label="Match density", type="filepath")

        run_button.click(
            fn=run_ui,
            inputs=[source_image, reference_image],
            outputs=[
                final_output,
                diffuse_luma_delta,
                diffuse_hue_delta,
                diffuse_chroma_scale,
                diffuse_confidence,
                diffuse_residual_rejection,
                ref_residual_confidence,
                specular_confidence,
                shadow_confidence,
                hue_stability_confidence,
                delta_sanity_confidence,
                luma_confidence,
                hue_confidence,
                chroma_confidence,
                lightglue_matches,
                filtered_match_confidence,
                sparse_luma_delta,
                sparse_hue_delta,
                sparse_chroma_scale,
                match_density,
            ],
            queue=True,
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
