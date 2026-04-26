# LookAlign V0.2.6

## Versions

- `V0.2.6`: updates SA-LUT to use identity RGB lookup coordinates and fit Lab deltas instead of absolute RGB outputs, increases the default LUT size to `33`, fixes neutral detection around centered Lab `a/b`, and adds deterministic save-time dithering. The Gradio UI now shows V0.2.6 base and light-map debug views.
- `V0.2.5`: makes SA-LUT the only global base match and replaces legacy local grid correction with residual luminance/color maps fitted after SA-LUT.
- `V0.2`: replaces the global match with a locally fitted SA-LUT-style 4D LUT (`context + RGB`) applied through PyTorch MPS when available, with CPU fallback. Alignment, local correction, reconstruction, and anti-fade guards remain the same.
- `V0.1`: `low-frequency transfer -> trust map -> reconstruction`, with anti-fade guards to keep contrast and saturation usable.
