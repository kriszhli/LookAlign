#!/usr/bin/env python3
"""Local Gradio UI for LookAlign V0.4.3."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from PIL import Image
import base64
from io import BytesIO

from scripts.global_matching import GlobalMatchingConfig, run_global_matching
from scripts.bilateral_transfer import BilateralTransferConfig, run_bilateral_transfer


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "inputs" / "source.png"
DEFAULT_REFERENCE = ROOT / "inputs" / "reference.png"
OUTPUTS_DIR = ROOT / "outputs"


def existing_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None

def get_example_pairs() -> list[Tuple[Path, Path]]:
    inputs_dir = ROOT / "inputs"
    if not inputs_dir.exists():
        return []
    
    sources = list(inputs_dir.glob("source*.*"))
    pairs = []
    for src in sorted(sources):
        if not src.is_file():
            continue
        name = src.stem
        if name.startswith("source"):
            suffix = name[len("source"):]
            refs = [r for r in inputs_dir.glob(f"reference{suffix}.*") if r.is_file()]
            if refs:
                pairs.append((src, refs[0]))
    return pairs

def make_composite(src_path: Path, ref_path: Path) -> Image.Image:
    w, h = 800, 600
    try:
        src_img = Image.open(src_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")
    except Exception:
        return Image.new("RGB", (w, h))
    
    def resize_crop(img: Image.Image, tw: int, th: int) -> Image.Image:
        img_ratio = img.width / img.height
        target_ratio = tw / th
        if img_ratio > target_ratio:
            new_w = int(img.height * target_ratio)
            left = (img.width - new_w) // 2
            img = img.crop((left, 0, left + new_w, img.height))
        else:
            new_h = int(img.width / target_ratio)
            top = (img.height - new_h) // 2
            img = img.crop((0, top, img.width, top + new_h))
        return img.resize((tw, th), Image.Resampling.LANCZOS)
    
    src_half = resize_crop(src_img, w // 2, h)
    ref_half = resize_crop(ref_img, w // 2, h)
    
    comp = Image.new("RGB", (w, h))
    comp.paste(src_half, (0, 0))
    comp.paste(ref_half, (w // 2, 0))
    return comp

def pil_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

class ExamplesGallery(gr.HTML):
    def __init__(self, **kwargs):
        self.pairs = get_example_pairs()
        images_html = []
        for i, (src, ref) in enumerate(self.pairs):
            comp = make_composite(src, ref)
            b64 = pil_to_base64(comp)
            images_html.append(f'<img src="{b64}" data-idx="{i}" class="example-thumb" />')
            
        html_template = f'<div class="custom-gallery">{"".join(images_html)}</div>'
        
        css_template = """
        .custom-gallery { display: flex; flex-wrap: wrap; gap: 15px; justify-content: flex-start; align-items: flex-start; }
        .example-thumb { width: calc(18vw - 15px); aspect-ratio: 4/3; object-fit: cover; cursor: pointer; border-radius: 8px; transition: transform 0.2s; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .example-thumb:hover { transform: scale(1.02); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
        """
        
        js_on_load = """
            const imgs = element.querySelectorAll('.example-thumb');
            imgs.forEach(img => {
                img.addEventListener('click', () => {
                    props.value = parseInt(img.getAttribute('data-idx'));
                    trigger('change');
                });
            });
        """
        super().__init__(value=-1, html_template=html_template, css_template=css_template, js_on_load=js_on_load, **kwargs)

    def get_pair(self, idx: int) -> Tuple[Optional[str], Optional[str]]:
        try:
            idx = int(idx)
            if 0 <= idx < len(self.pairs):
                return str(self.pairs[idx][0]), str(self.pairs[idx][1])
        except (ValueError, TypeError):
            pass
        return None, None

def choose_input(path: Optional[str], default: Path, label: str) -> str:
    if path:
        return path
    if default.exists():
        return str(default)
    raise gr.Error(f"Please upload a {label} image.")


def run_v040(source_path: Optional[str], reference_path: Optional[str]) -> Tuple[str, ...]:
    source = choose_input(source_path, DEFAULT_SOURCE, "source")
    reference = choose_input(reference_path, DEFAULT_REFERENCE, "reference")
    run_dir = OUTPUTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f") / "v040"

    global_metrics = run_global_matching(source, reference, run_dir, GlobalMatchingConfig())
    tensors = global_metrics["tensors"]
    metrics = run_bilateral_transfer(
        base_intermediate_lab=tensors["base_intermediate_lab"],
        base_intermediate_rgb=tensors["base_intermediate_rgb"],
        reference_resized_lab=tensors["reference_resized_lab"],
        reference_resized_rgb=tensors["reference_resized_rgb"],
        source_lab=tensors["source_lab"],
        source_rgb=tensors["source_rgb"],
        output_dir=run_dir,
        global_metrics=global_metrics,
        config=BilateralTransferConfig(),
    )
    paths = metrics["paths"]
    return (
        paths["base_intermediate"],
        paths["final_output"],
        paths.get("reference_resized", ""),
        paths.get("diff_map", ""),
    )


css = """
/* No global CSS needed for gallery, handled by ExamplesGallery custom component */
"""

def build_app() -> gr.Blocks:
    with gr.Blocks(title="LookAlign V0.4.3") as demo:
        gr.Markdown("# LookAlign V0.4.3 — Neural Preset + Bilateral Transfer")

        gr.Markdown("### Examples")
        examples_gallery = ExamplesGallery()

        with gr.Row():
            source_image = gr.Image(label="Source image", type="filepath", value=existing_path(DEFAULT_SOURCE), interactive=True)
            reference_image = gr.Image(label="Reference image", type="filepath", value=existing_path(DEFAULT_REFERENCE), interactive=True)

        examples_gallery.change(
            fn=examples_gallery.get_pair,
            inputs=[examples_gallery],
            outputs=[source_image, reference_image]
        )

        run_v040_btn = gr.Button("Run V0.4.3", variant="primary")

        gr.Markdown("## Outputs")
        with gr.Row():
            v4_base = gr.Image(label="Base intermediate (after Neural Preset)", type="filepath")
            v4_final = gr.Image(label="V0.4.3 Final output", type="filepath")

        gr.Markdown("## Alignment and Difference")
        with gr.Row():
            v4_ref = gr.Image(label="Reference resized", type="filepath")
            v4_diff = gr.Image(label="Difference Map (Output vs Aligned Reference)", type="filepath")

        run_v040_btn.click(
            fn=run_v040,
            inputs=[source_image, reference_image],
            outputs=[v4_base, v4_final, v4_ref, v4_diff],
            queue=True,
        )

    return demo


if __name__ == "__main__":
    build_app().launch(css=css)
