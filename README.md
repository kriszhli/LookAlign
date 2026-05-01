# LookAlign V0.4.5

LookAlign is a high-performance color-matching system that transfers the visual style of an AI-generated reference onto a source image while preserving all original details and structure. 

It combines **spatially-aware matching** with **deterministic color transforms** to achieve accurate results even when the reference contains extreme AI artifacts. 

Designed for near real-time performance, LookAlign is built for modern image and video workflows where precision and speed both matter.

## Versions

- `V0.4.5` - `XFeat`* pre-alignment before the global/bilateral stages; source/reference features are matched, a homography is estimated, and the valid overlap is cropped before Neural Preset + bilateral transfer.
- `V0.4.4` - `Neural Preset / DNCM` global stage in `sRGB`, then `CIE Lab` bilateral-grid affine transfer with edge-aware coefficient smoothing and a debug edit map.
- `V0.4.3` - `Neural Preset / DNCM` global stage in `sRGB`, followed by `CIE Lab` bilateral-grid affine transfer back to `sRGB`.
- `V0.4.2` - Restricted the bilateral grid to `chrominance (a/b)` only and removed global Reinhard std-match to stop tonal compression.
- `V0.4.1` - Added reference denoising before bilateral splatting; `ref_denoise_sigma=1.0` suppresses grain while keeping broad color patterns.
- `V0.4.0` - Replaced low-frequency proxy deltas with **bilateral-grid local affine** transfer, cutting `LightGlue`, reducing hyperparameters, and running fast on `MPS`.
- `V0.3.5` - Global `OT`/`SVD` base with `LightGlue`+`ALIKED`-guided sparse sampling into smooth `MPS` local OKLab correction maps.
- `V0.3.1` - Replaced global LUT/`BGrid` with sliced partial `OT` -> `SVD`-smoothed LUT, producing a conservative base intermediate.
- `V0.3.0` - Full redesign to ML-driven appearance transform from source/reference pairs with learned global LUT fitting.
- `V0.2.6` - Improved `SA-LUT` (`Lab` deltas, `33^3` LUT, neutral fix, dithering) with updated debug views.
- `V0.2.5` - Simplified pipeline: `SA-LUT` as sole global stage plus residual luminance/color local maps.
- `V0.2` - Introduced `SA-LUT`-style 4D LUT (`context + RGB`) with `MPS` acceleration and unchanged downstream pipeline.
- `V0.1` - Initial pipeline: low-frequency transfer -> trust map -> reconstruction with anti-fade safeguards.
