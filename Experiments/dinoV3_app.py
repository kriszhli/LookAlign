import argparse

import gradio as gr
from PIL import Image

try:
    from .dinoV3 import DEFAULT_IMAGE_PATH, DinoRunner, MODEL_SPEC
except ImportError:
    from dinoV3 import DEFAULT_IMAGE_PATH, DinoRunner, MODEL_SPEC


def build_interface(runner: DinoRunner) -> gr.Blocks:
    css = """
    .app-shell {
        max-width: 1500px;
        margin: 0 auto;
    }
    .app-header {
        padding: 10px 0 4px 0;
    }
    .app-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-top: -6px;
    }
    .surface {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .controls-row {
        align-items: end;
        gap: 16px;
    }
    .run-wrap {
        justify-content: center;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .run-wrap button {
        min-width: 220px;
    }
    .debug-box textarea,
    .debug-box pre {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size: 12px !important;
        line-height: 1.45 !important;
    }
    """

    def run_analysis(
        image: Image.Image,
        cluster_count: float,
        pca_dims: float,
        anyup_target_px: float,
        allow_cpu: bool,
    ):
        artifacts = runner.run(
            image=image,
            manual_clusters=int(cluster_count),
            allow_cpu=bool(allow_cpu),
            pca_dims=int(pca_dims),
            anyup_target_short_side=int(anyup_target_px),
        )
        cpu_visible = artifacts.requires_cpu_confirmation or artifacts.device_type == "cpu"
        cpu_value = bool(allow_cpu) if cpu_visible else False
        return (
            artifacts.status,
            (artifacts.kmeans_overlay, artifacts.kmeans_mask),
            (artifacts.anyup_overlay, artifacts.anyup_mask),
            gr.update(visible=cpu_visible, value=cpu_value),
        )

    def reset_cpu_confirmation():
        return gr.update(visible=False, value=False)

    with gr.Blocks(title="DINOv3 AnyUp Viewer") as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(f"<style>{css}</style>")
            gr.Markdown(
                """
                <div class="app-header">
                  <h2>DINOv3 AnyUp Viewer</h2>
                  <div class="app-subtitle">DINOv3 patch KMeans baseline beside official AnyUp feature upsampling.</div>
                </div>
                """
            )
            with gr.Row(equal_height=True):
                input_image = gr.Image(
                    label="Input",
                    type="pil",
                    value=str(DEFAULT_IMAGE_PATH),
                    height=560,
                    elem_classes="surface",
                    show_label=True,
                    sources=["upload", "clipboard"],
                )
                kmeans_slider = gr.ImageSlider(
                    label="KMeans Baseline: Overlay / Mask",
                    height=560,
                    elem_classes="surface",
                    show_label=True,
                    type="pil",
                    slider_position=50,
                )
                anyup_slider = gr.ImageSlider(
                    label="AnyUp: Overlay / Mask",
                    height=560,
                    elem_classes="surface",
                    show_label=True,
                    type="pil",
                    slider_position=50,
                )
            with gr.Row(elem_classes="controls-row"):
                cluster_slider = gr.Slider(
                    label="Clusters",
                    info="0 uses the automatic heuristic.",
                    minimum=0,
                    maximum=40,
                    value=0,
                    step=1,
                )
                pca_slider = gr.Slider(
                    label="AnyUp feature dimensions",
                    info="PCA-compressed DINOv3 feature channels passed into AnyUp.",
                    minimum=8,
                    maximum=128,
                    value=16,
                    step=8,
                )
                anyup_target_slider = gr.Slider(
                    label="AnyUp target px",
                    info="Short-side target for the AnyUp output map.",
                    minimum=64,
                    maximum=1024,
                    value=128,
                    step=16,
                )
            cpu_confirm = gr.Checkbox(
                label="MPS/CUDA unavailable. Check this box, then run again to allow CPU execution for the current input.",
                value=False,
                visible=False,
            )
            with gr.Row(elem_classes="run-wrap"):
                run_button = gr.Button("Run Analysis", variant="primary", size="lg")
            status = gr.Textbox(
                label="Runtime",
                lines=12,
                max_lines=16,
                interactive=False,
                elem_classes="debug-box",
            )

        run_button.click(
            fn=run_analysis,
            inputs=[input_image, cluster_slider, pca_slider, anyup_target_slider, cpu_confirm],
            outputs=[status, kmeans_slider, anyup_slider, cpu_confirm],
            show_progress="minimal",
        )
        input_image.change(
            fn=reset_cpu_confirmation,
            inputs=[],
            outputs=[cpu_confirm],
            show_progress="hidden",
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv3 AnyUp viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for Gradio")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share links")
    parser.add_argument("--smoke-test", action="store_true", help="Run one pass and exit")
    parser.add_argument("--clusters", type=int, default=0, help="Cluster count; 0 uses the automatic heuristic")
    parser.add_argument("--pca-dims", type=int, default=16, help="PCA dimensions for upsampled DINOv3 features")
    parser.add_argument("--anyup-target-px", type=int, default=128, help="Short-side target for the AnyUp output map")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU inference when MPS/CUDA is unavailable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = DinoRunner(MODEL_SPEC)
    if args.smoke_test:
        image = Image.open(DEFAULT_IMAGE_PATH).convert("RGB")
        artifacts = runner.run(
            image=image,
            manual_clusters=args.clusters,
            allow_cpu=args.allow_cpu,
            pca_dims=args.pca_dims,
            anyup_target_short_side=args.anyup_target_px,
        )
        print(artifacts.status)
        if artifacts.requires_cpu_confirmation:
            raise SystemExit(2)
        return

    demo = build_interface(runner)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
