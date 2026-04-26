#!/usr/bin/env python3
"""Local Gradio UI for LookAlign V0.3.1."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr

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

    # V0.3.1 uses locked global-match parameters from OTSVDLUTConfig.
    # The UI intentionally exposes no tuning sliders for this stage.
    metrics = run_ot_svd_lut_mps(source, reference, run_dir, OTSVDLUTConfig())
    return metrics["paths"]["base_intermediate"]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LookAlign V0.3.1") as demo:
        gr.Markdown("# LookAlign V0.3.1")

        with gr.Row():
            source_image = gr.Image(label="Source image", type="filepath", value=existing_path(DEFAULT_SOURCE), interactive=True)
            reference_image = gr.Image(label="Reference image", type="filepath", value=existing_path(DEFAULT_REFERENCE), interactive=True)

        run_button = gr.Button("Run V0.3.1 Base", variant="primary")

        base_intermediate = gr.Image(label="Base intermediate", type="filepath")

        run_button.click(
            fn=run_ui,
            inputs=[source_image, reference_image],
            outputs=base_intermediate,
            queue=True,
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
