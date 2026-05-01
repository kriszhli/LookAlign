# LookAlign V0.4.5

LookAlign is a high-performance color-matching system that transfers the visual style of an AI-generated reference onto a source image while preserving all original details and structure. 

It combines **spatially-aware matching** with **deterministic color transforms** to achieve accurate results even when the reference contains extreme AI artifacts. 

Designed for near real-time performance, LookAlign is built for modern image and video workflows where precision and speed both matter.

## Versions

- `V0.4.5` - Reintroduces **LightGlue** pre-alignment before the global and bilateral stages: matches source/reference features, estimates a homography, crops to the valid overlap, and exposes a LightGlue debug viewport in the UI. The downstream pipeline then runs Neural Preset global matching followed by bilateral-grid affine transfer on the aligned pair.
- `V0.4.4` - Uses **Neural Preset / DNCM** as the global stage in `sRGB`, then converts `base_intermediate_rgb` and the resized reference to **CIE Lab** for the bilateral-grid affine transfer before converting back to `sRGB` for gamut compression and saving. The bilateral transfer now uses edge-aware coefficient smoothing and a debug edit map to preserve boundaries without over-blurring edits within protected regions.
- `V0.4.3` - Uses **Neural Preset / DNCM** as the global stage in `sRGB`, then converts `base_intermediate_rgb` and the resized reference to **CIE Lab** for the bilateral-grid affine transfer before converting back to `sRGB` for gamut compression and saving.
- `V0.4.2` - Fixed "washed out" issue by restricting the bilateral grid to correct **chrominance (a/b) only**. The global 3D LUT already perfectly matches the L percentiles, and applying per-cell L offsets was compressing the tonal range. Removed the global Reinhard standard deviation matching which was increasing pixel-wise error.
- `V0.4.1` - Reference denoising before bilateral grid splatting. AI-generated film-style references contain grain noise that biases per-cell means (especially lifting shadows). A configurable Gaussian blur (`ref_denoise_sigma=1.0`) suppresses grain while preserving broad color patterns.
- `V0.4.0` - Replaced low-frequency proxy deltas with **bilateral-grid local affine** transfer. Edge-awareness built into the grid structure (luminance-binned cells); misalignment tolerance via statistics-based cell fitting. Mean-shift-only per cell + global Reinhard std-match. Eliminates `LightGlue` dependency, reduces hyperparameters to 9, runs in ~0.3s on MPS.
- `V0.3.5` - Global `OT`/`SVD` base with `LightGlue`+`ALIKED`-guided sparse sampling into smooth `MPS` local OKLab correction maps + confidence and fallback.
- `V0.3.1` - Replaced global LUT/`BGrid` with sliced partial `OT` -> `SVD`-smoothed LUT, producing a conservative base intermediate.
- `V0.3.0` - Full redesign to ML-driven appearance transform from source/reference pairs with learned global LUT fitting.
- `V0.2.6` - Improved `SA-LUT` (`Lab` deltas, `33^3` LUT, neutral fix, dithering) with updated debug views.
- `V0.2.5` - Simplified pipeline: `SA-LUT` as sole global stage + residual luminance/color local maps.
- `V0.2` - Introduced `SA-LUT`-style 4D LUT (`context + RGB`) with `MPS` acceleration and unchanged downstream pipeline.
- `V0.1` - Initial pipeline: low-frequency transfer -> trust map -> reconstruction with anti-fade safeguards.
