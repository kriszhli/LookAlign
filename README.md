# LookAlign V0.3.1

## Versions

- `V0.3.1`: replaces the rejected V0.3 global-LUT/BGrid prototype with an MPS-required sliced partial OT color transport stage distilled into an SVD-smoothed LUT. The output is a conservative base intermediate for a future CNN local affine field.
- `V0.3.0`: a complete remake of LookAlign. The pipeline now uses ML models to learn the appearance transform directly from source/reference pairs instead of relying on manual calculations or SA-LUT-style lookup fitting. This release shifts the core workflow from hand-designed correction logic to a learned global LUT fitting approach, with the UI and metrics updated to match the new model-driven pipeline.
- `V0.2.6`: updates SA-LUT to use identity RGB lookup coordinates and fit Lab deltas instead of absolute RGB outputs, increases the default LUT size to `33`, fixes neutral detection around centered Lab `a/b`, and adds deterministic save-time dithering. The Gradio UI now shows V0.2.6 base and light-map debug views.
- `V0.2.5`: makes SA-LUT the only global base match and replaces legacy local grid correction with residual luminance/color maps fitted after SA-LUT.
- `V0.2`: replaces the global match with a locally fitted SA-LUT-style 4D LUT (`context + RGB`) applied through PyTorch MPS when available, with CPU fallback. Alignment, local correction, reconstruction, and anti-fade guards remain the same.
- `V0.1`: `low-frequency transfer -> trust map -> reconstruction`, with anti-fade guards to keep contrast and saturation usable.
