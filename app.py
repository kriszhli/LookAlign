#!/usr/bin/env python3
"""Local Gradio UI for LookAlign V0.3.5."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

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


def run_ui(source_path: Optional[str], reference_path: Optional[str]) -> str:
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
    return metrics["paths"]["final_output"]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LookAlign V0.3.5") as demo:
        gr.Markdown("# LookAlign V0.3.5")

        with gr.Row():
            source_image = gr.Image(label="Source image", type="filepath", value=existing_path(DEFAULT_SOURCE), interactive=True)
            reference_image = gr.Image(label="Reference image", type="filepath", value=existing_path(DEFAULT_REFERENCE), interactive=True)

        run_button = gr.Button("Run V0.3.5", variant="primary")

        final_output = gr.Image(label="Final output", type="filepath")

        run_button.click(
            fn=run_ui,
            inputs=[source_image, reference_image],
            outputs=final_output,
            queue=True,
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
