#!/usr/bin/env python3
"""Local Gradio UI for LookAlign V0.3.1."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr

from scripts.ot_svd_lut_mps import OTSVDLUTConfig, run_ot_svd_lut_mps


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "inputs" / "source.png"
DEFAULT_REFERENCE = ROOT / "inputs" / "reference.png"
OUTPUTS_DIR = ROOT / "outputs"
IMAGE_OPTIONS = ["Source", "Reference resized", "OT preview", "Base intermediate"]


def existing_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def choose_input(path: Optional[str], default: Path, label: str) -> str:
    if path:
        return path
    if default.exists():
        return str(default)
    raise gr.Error(f"Please upload a {label} image.")


def comparison_value(paths: Dict[str, Optional[str]], left_name: str, right_name: str) -> tuple[Optional[str], Optional[str]]:
    paths = paths or {}
    return paths.get(left_name), paths.get(right_name)


def run_ui(
    source_path: Optional[str],
    reference_path: Optional[str],
    fit_long_edge: int,
    sample_count: int,
    ot_iterations: int,
    partial_ratio: float,
    lut_size: int,
    svd_rank: int,
    svd_smoothing: float,
    max_luma_delta: float,
    max_chroma_scale: float,
    neutral_protection: float,
    compare_a: str,
    compare_b: str,
) -> tuple[Any, ...]:
    source = choose_input(source_path, DEFAULT_SOURCE, "source")
    reference = choose_input(reference_path, DEFAULT_REFERENCE, "reference")
    run_dir = OUTPUTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    config = OTSVDLUTConfig(
        fit_long_edge=int(fit_long_edge),
        sample_count=int(sample_count),
        ot_iterations=int(ot_iterations),
        partial_ratio=float(partial_ratio),
        lut_size=int(lut_size),
        svd_rank=int(svd_rank),
        svd_smoothing=float(svd_smoothing),
        max_luma_delta=float(max_luma_delta),
        max_chroma_scale=float(max_chroma_scale),
        neutral_protection=float(neutral_protection),
    )
    metrics = run_ot_svd_lut_mps(source, reference, run_dir, config)
    paths = {
        "Source": source,
        "Reference resized": metrics["paths"]["reference_resized"],
        "OT preview": metrics["paths"]["ot_preview"],
        "Base intermediate": metrics["paths"]["base_intermediate"],
    }
    summary = {
        "pipeline_version": metrics["pipeline_version"],
        "device": metrics["device"],
        "source_shape": metrics["source_shape"],
        "reference_shape": metrics["reference_shape"],
        "fit_shape": metrics["fit_shape"],
        "sample_count_used": metrics["sample_count_used"],
        "lut_support_coverage": metrics["lut_support_coverage"],
        "compressed_pixel_ratio": metrics["compressed_pixel_ratio"],
        "mean_luma_delta": metrics["mean_luma_delta"],
        "mean_abs_luma_delta": metrics["mean_abs_luma_delta"],
        "mean_chroma_delta": metrics["mean_chroma_delta"],
        "mean_abs_chroma_delta": metrics["mean_abs_chroma_delta"],
        "timings": metrics["timings"],
        "output_dir": str(run_dir),
    }
    return (
        comparison_value(paths, compare_a, compare_b),
        metrics["paths"]["base_intermediate"],
        metrics["paths"]["reference_resized"],
        metrics["paths"]["source_fit"],
        metrics["paths"]["ot_preview"],
        metrics["paths"]["lut_support"],
        metrics["paths"]["metrics"],
        summary,
        paths,
    )


def build_app() -> gr.Blocks:
    defaults = {
        "Source": existing_path(DEFAULT_SOURCE),
        "Reference resized": None,
        "OT preview": None,
        "Base intermediate": None,
    }
    with gr.Blocks(title="LookAlign V0.3.1") as demo:
        gr.Markdown("# LookAlign V0.3.1")
        state = gr.State(defaults)

        with gr.Row():
            source_image = gr.Image(label="Source image", type="filepath", value=defaults["Source"], interactive=True)
            reference_image = gr.Image(label="Reference image", type="filepath", value=existing_path(DEFAULT_REFERENCE), interactive=True)

        with gr.Row():
            fit_long_edge = gr.Slider(128, 1024, value=768, step=64, label="Fit long edge")
            sample_count = gr.Slider(4096, 262144, value=131072, step=4096, label="Sample count")
            ot_iterations = gr.Slider(4, 128, value=64, step=4, label="OT iterations")
            partial_ratio = gr.Slider(0.25, 1.0, value=0.85, step=0.01, label="Partial ratio")

        with gr.Row():
            lut_size = gr.Slider(9, 49, value=33, step=2, label="LUT size")
            svd_rank = gr.Slider(1, 16, value=8, step=1, label="SVD rank")
            svd_smoothing = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="SVD smoothing")

        with gr.Row():
            max_luma_delta = gr.Slider(0.02, 0.35, value=0.18, step=0.01, label="Max luma delta")
            max_chroma_scale = gr.Slider(1.0, 2.0, value=1.25, step=0.01, label="Max chroma scale")
            neutral_protection = gr.Slider(0.0, 1.0, value=0.85, step=0.01, label="Neutral protection")

        with gr.Row():
            compare_a = gr.Dropdown(IMAGE_OPTIONS, value="Source", label="Comparison A")
            compare_b = gr.Dropdown(IMAGE_OPTIONS, value="Base intermediate", label="Comparison B")
            run_button = gr.Button("Run V0.3.1 Base", variant="primary")

        comparison = gr.ImageSlider(
            value=comparison_value(defaults, "Source", "Base intermediate"),
            type="filepath",
            label="Base intermediate comparison",
            interactive=False,
            max_height=760,
        )

        with gr.Row():
            base_intermediate = gr.Image(label="Base intermediate", type="filepath")
            reference_resized = gr.Image(label="Reference resized", type="filepath")
            source_fit = gr.Image(label="Source fit image", type="filepath")

        with gr.Row():
            ot_preview = gr.Image(label="OT preview", type="filepath")
            lut_support = gr.Image(label="LUT support", type="filepath")

        with gr.Row():
            metrics_file = gr.File(label="Metrics JSON")
            summary = gr.JSON(label="Run summary")

        run_button.click(
            fn=run_ui,
            inputs=[
                source_image,
                reference_image,
                fit_long_edge,
                sample_count,
                ot_iterations,
                partial_ratio,
                lut_size,
                svd_rank,
                svd_smoothing,
                max_luma_delta,
                max_chroma_scale,
                neutral_protection,
                compare_a,
                compare_b,
            ],
            outputs=[
                comparison,
                base_intermediate,
                reference_resized,
                source_fit,
                ot_preview,
                lut_support,
                metrics_file,
                summary,
                state,
            ],
            queue=True,
        )
        compare_a.change(comparison_value, inputs=[state, compare_a, compare_b], outputs=comparison, queue=False)
        compare_b.change(comparison_value, inputs=[state, compare_a, compare_b], outputs=comparison, queue=False)

    return demo


if __name__ == "__main__":
    build_app().launch()
