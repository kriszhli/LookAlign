# LookAlign V0.1

Architectural log for the current V0.1 pipeline.

## Core Flow

1. `low-frequency transfer`
   - Compare source and reference after broad blur.
   - Transfer coarse color / lighting structure without chasing fine detail.

2. `trust map`
   - Estimate where the reference is reliable enough to influence the fit.
   - Downweight weak, noisy, or ambiguous regions.

3. `reconstruction`
   - Rebuild the final image from local corrections and source detail.
   - Apply anti-fade guards so the output keeps usable contrast and saturation.

V0.1 is a minimal log of this architecture. A much larger upgrade is next.
