# LookAlign V0.4

LookAlign is a high-performance color-matching system that transfers the visual style of an AI-generated reference onto a source image while preserving all original details and structure. 

It combines **spatially-aware matching** with **deterministic color transforms** to achieve accurate results even when the reference contains extreme AI artifacts. 

Designed for near real-time performance, LookAlign is built for modern image and video workflows where precision and speed both matter.

## Versions

- `V0.4.3` - Addressed unaligned reference color-washing by using **DIS Optical Flow** to align the reference to the source *before* computing bilateral statistics. Ensures saturated, local color features fall into correct spatial bins. Replaced debug grids in the UI with an aligned difference map heat-viz.
- `V0.4.2` - Fixed "washed out" issue by restricting the bilateral grid to correct **chrominance (a/b) only**. The global 3D LUT already perfectly matches the L percentiles, and applying per-cell L offsets was compressing the tonal range. Removed the global Reinhard standard deviation matching which was increasing pixel-wise error.
- `V0.4.1` - Reference denoising before bilateral grid splatting. AI-generated film-style references contain grain noise that biases per-cell means (especially lifting shadows). A configurable Gaussian blur (`ref_denoise_sigma=1.0`) suppresses grain while preserving broad color patterns.
- `V0.4.0` - Replaced low-frequency proxy deltas with **bilateral-grid local affine** transfer. Edge-awareness built into the grid structure (luminance-binned cells); misalignment tolerance via statistics-based cell fitting. Mean-shift-only per cell + global Reinhard std-match. Eliminates `LightGlue` dependency, reduces hyperparameters to 9, runs in ~0.3s on MPS.
- `V0.3.6` - Switched to linear-RGB 3D LUT global fit; `LightGlue`+`ALIKED`-first local alignment with warped reference and dense OKLab diffuse corrections + confidence.
- `V0.3.5` - Global `OT`/`SVD` base with `LightGlue`+`ALIKED`-guided sparse sampling into smooth `MPS` local OKLab correction maps + confidence and fallback.
- `V0.3.1` - Replaced global LUT/`BGrid` with sliced partial `OT` -> `SVD`-smoothed LUT, producing a conservative base intermediate.
- `V0.3.0` - Full redesign to ML-driven appearance transform from source/reference pairs with learned global LUT fitting.
- `V0.2.6` - Improved `SA-LUT` (`Lab` deltas, `33^3` LUT, neutral fix, dithering) with updated debug views.
- `V0.2.5` - Simplified pipeline: `SA-LUT` as sole global stage + residual luminance/color local maps.
- `V0.2` - Introduced `SA-LUT`-style 4D LUT (`context + RGB`) with `MPS` acceleration and unchanged downstream pipeline.
- `V0.1` - Initial pipeline: low-frequency transfer -> trust map -> reconstruction with anti-fade safeguards.
