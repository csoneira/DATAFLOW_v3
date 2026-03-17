---
name: physics
description: Detector physics domain knowledge — RPC efficiency, streamers, noise signatures
type: project
---

## RPC efficiency and streamers

- Real data shows efficiency dip-then-recovery in threshold scan; simulation (no streamers) shows monotonic decrease only.
- **Hypothesis 1 (plane asymmetry):** Some planes may have systematically more streamers due to construction differences (gas flow, HV, gap geometry). This would show up as different streamer fractions per plane.
- **Hypothesis 2 (streamer independence):** Streamers in one plane should NOT make streamers more likely in another plane — they are local gas-gap phenomena. If the contagion matrix shows significant off-diagonal correlation, that would be surprising and suggest a shared cause (e.g., high-ionization muon event triggering multiple gaps).
- **Hypothesis 3 (streamers = big avalanches, not noise):** Streamers may simply be avalanches that grew larger — same physical signal, just with saturated/distorted charge. Under this hypothesis, streamers carry real muon information and are not noise. The efficiency comparison plot (all vs streamer-free) tests whether removing streamers makes real data match simulation behavior.

**Why:** These hypotheses guide interpretation of the streamer investigation plots added to Task 3.
**How to apply:** When analyzing streamer plot outputs, frame results against these three hypotheses rather than assuming streamers = noise.
