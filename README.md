# LookAlign V0.2.5

## Versions

- `V0.2.5`: makes SA-LUT the only global base match and replaces legacy local grid correction with residual luminance/color maps fitted after SA-LUT. The Gradio UI now shows V0.2.5 base, residual, and confidence debug views.
- `V0.2`: replaces the global match with a locally fitted SA-LUT-style 4D LUT (`context + RGB`) applied through PyTorch MPS when available, with CPU fallback. Alignment, local correction, reconstruction, and anti-fade guards remain the same.
- `V0.1`: `low-frequency transfer -> trust map -> reconstruction`, with anti-fade guards to keep contrast and saturation usable.
