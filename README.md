# LookAlign V0.1


## Versions

- `V0.1`: `low-frequency transfer -> trust map -> reconstruction`, with anti-fade guards to keep contrast and saturation usable.
- `V0.2`: replaces the global match with a locally fitted SA-LUT-style 4D LUT (`context + RGB`) applied through PyTorch MPS when available, with CPU fallback. Alignment, local correction, reconstruction, and anti-fade guards remain the same.
