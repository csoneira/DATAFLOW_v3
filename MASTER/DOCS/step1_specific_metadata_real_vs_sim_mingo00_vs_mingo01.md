---
title: STEP_1 Specific Metadata Real vs Sim Study
description: Task 1-4 comparison between simulated MINGO00 metadata and real MINGO01 metadata, focused on correlations and likely noise locations.
last_updated: 2026-03-13
status: draft
---

# STEP_1 Specific Metadata Real vs Sim Study

## Scope

This note compares the STEP_1 `task_*_metadata_specific.csv` products for:

- simulated branch: `STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_{1..4}/METADATA/task_{task}_metadata_specific.csv`
- real branch: `STATIONS/MINGO01/STAGE_1/EVENT_DATA/STEP_1/TASK_{1..4}/METADATA/task_{task}_metadata_specific.csv`

The goal is not only to check whether real and simulated metadata differ, but to identify where the differences look most compatible with hidden noise, accidental activity, crosstalk, or fit-background contamination.

## Method

- One row per `filename_base` was kept by taking the latest available `execution_timestamp` in each task CSV.
- Comparison was done on `MINGO00` vs `MINGO01` after that latest-per-basename aggregation.
- I used task-specific derived metrics instead of only raw columns, so the report is about occupancy, retention, topology, and fit behavior rather than thousands of individual metadata fields.
- For each derived metric I compared real and simulated medians. When simulation is identically zero, the result should be interpreted as categorical presence/absence evidence, not as a literal effect size.
- I also compared selected correlation structures between real and simulated derived metrics to check whether the real data behaves like a noisy deformation of the simulated pipeline or like a completely different regime.

## Sample Sizes

| Task | Simulated `MINGO00` | Real `MINGO01` |
|---|---:|---:|
| 1 | 11,535 files | 129 files |
| 2 | 11,039 files | 129 files |
| 3 | 10,972 files | 129 files |
| 4 | 10,970 files | 129 files |

## Executive Summary

The strongest and cleanest noise signature is in `TASK_3`. The simulation has essentially zero non-adjacent strip patterns and zero triple-or-more active-strip fractions, while the real data has both at clearly nonzero levels in all planes. This is the most direct topological evidence that the real station is not behaving like a noiseless detector chain.

`TASK_1` and `TASK_2` already show the same story earlier in the chain. Real data has nonzero zeroed percentages, lower retention from original to final entries, larger front/back asymmetry, larger `CRT_avg`, and much larger `CRT_std`. These effects concentrate in a few edge strips and outer planes, but they are not limited to one single bad channel.

`TASK_4` shows the fit-level consequence of the same problem. Background terms increase in every trigger-combination family, non-adjacent pairs become almost as background-heavy as adjacent pairs, and fourfold external residual tails explode in real data. This suggests that the hidden contamination is not only strip-local noise; part of it survives into the reconstruction tail population.

## Task 1: Raw-to-Clean Symptoms

### Main shifts

| Metric | Sim median | Real median | Reading |
|---|---:|---:|---|
| `zeroed_mean` | `0.000` | `4.421` | Zeroed channels are absent in sim and persistent in real. |
| `time_retention` | `0.991` | `0.852` | Real loses a substantial fraction of time entries. |
| `charge_retention` | `1.000` | `0.961` | Charge also loses entries, but less severely than time. |
| `fb_asym_final` | `0.0050` | `0.0162` | Real front/back imbalance is about three times larger. |
| `time_original_total_rate` | `66.69 Hz` | `146.88 Hz` | Real channel activity is much higher overall. |

### Plane structure

Median mean zeroed percentage by plane in `MINGO01`:

| Plane | Real median zeroed mean |
|---|---:|
| `P1` | `3.67` |
| `P2` | `5.01` |
| `P3` | `4.76` |
| `P4` | `3.28` |

The largest raw entry-rate excesses appear in channels around plane 4 strip 4 and plane 2 strip 4. In other words, the problem is not uniformly spread over the detector; edge strips are among the most affected.

### Correlations

- `corr(zeroed_mean, time_retention)` is undefined in sim because `zeroed_mean` is identically zero there, but in real data it is `-0.614`.
- `corr(zeroed_mean, charge_retention)` is also only informative in real data and equals `-0.435`.
- `corr(time_retention, fb_asym_final)` changes from `-0.256` in sim to `-0.517` in real.

### Interpretation

`TASK_1` already contains a concrete noise or instability signature. Real files with more zeroed content lose more time and charge entries, and they also become more asymmetric between front and back sides. That is compatible with:

- intermittent channel dropout or zero-padding,
- high raw occupancy pushing unstable channels into cleaning losses,
- electronics-side imbalance that is not represented in the simulation.

## Task 2: Calibration and Strip Retention

### Main shifts

| Metric | Sim median | Real median | Reading |
|---|---:|---:|---|
| `CRT_avg` | `351.40` | `415.03` | Real timing-related average is shifted upward. |
| `CRT_std` | `15.75` | `45.76` | Real timing spread is roughly three times broader. |
| `strip_retention` | `0.977` | `0.946` | More strip-level losses survive calibration in real. |
| `strip_imbalance_original` | `0.153` | `0.116` | Real occupancy is not dominated by one strip; it is more common-mode. |
| `strip_imbalance_final` | `0.166` | `0.127` | The same common-mode pattern remains after final selection. |

### Plane and strip structure

Median per-plane strip-retention:

| Plane | Sim median | Real median |
|---|---:|---:|
| `P1` | `0.9786` | `0.9606` |
| `P2` | `0.9867` | `0.9450` |
| `P3` | `0.9746` | `0.9691` |
| `P4` | `0.9658` | `0.9100` |

The worst strip-specific retention deficits are:

- `retention_P2_s4`: `0.990 -> 0.847`
- `retention_P1_s4`: `0.997 -> 0.912`
- `retention_P4_s2`: `0.998 -> 0.971`
- `retention_P4_s1`: `0.999 -> 0.978`

This points to a mixed picture: plane 4 is globally the weakest plane, while strip 4 in planes 1 and 2 also stands out.

### Correlations

- `corr(CRT_std, strip_retention)` changes from `-0.002` in sim to `+0.697` in real.
- `corr(CRT_std, strip_imbalance_final)` changes from `+0.316` in sim to `-0.675` in real.
- `corr(CRT_avg, strip_retention)` changes from `+0.090` in sim to `-0.611` in real.

### Interpretation

This is not the signature of one isolated hot strip. In real data, the broad-`CRT` regime is associated with lower retention but also with more uniform strip occupancy. That is more compatible with common-mode contamination, station state changes, or broad event-class mixing than with a single pathological strip. Still, the strongest absolute losses are concentrated in a few edge strips and in plane 4.

## Task 3: Active-Strip Topology

This is the clearest detector-noise discriminator in the whole study.

### Main shifts

| Metric | Sim median | Real median | Reading |
|---|---:|---:|---|
| `active_rate_mean` | `6.55 Hz` | `12.20 Hz` | Real planes are more active overall. |
| `adj_double_frac_mean` | `0.458` | `0.116` | Real topology is much less dominated by adjacent doubles. |
| `nonadj_double_frac_mean` | `0.000` | `0.0157` | Non-adjacent doubles are absent in sim and persistent in real. |
| `tripleplus_frac_mean` | `0.000` | `0.0167` | Triple-or-more patterns are also absent in sim and persistent in real. |
| `nonadj_over_adj_mean` | `0.000` | `0.135` | Non-adjacent content is no longer negligible in real data. |
| `nonadj_double_rate_mean` | `0.000 Hz` | `0.192 Hz` | Direct excess rate scale for non-adjacent patterns. |
| `tripleplus_rate_mean` | `0.000 Hz` | `0.204 Hz` | Direct excess rate scale for triple-or-more patterns. |

### Per-plane medians

| Plane | Sim non-adjacent frac | Real non-adjacent frac | Sim triple+ frac | Real triple+ frac |
|---|---:|---:|---:|---:|
| `P1` | `0.0000` | `0.0121` | `0.0000` | `0.0118` |
| `P2` | `0.0000` | `0.0193` | `0.0000` | `0.0190` |
| `P3` | `0.0000` | `0.0192` | `0.0000` | `0.0190` |
| `P4` | `0.0000` | `0.0117` | `0.0000` | `0.0166` |

Inner planes `P2` and `P3` are the strongest non-adjacent offenders on average.

### Specific patterns that appear in real but not in sim

- `frac_P4_1001`: real median `0.0090`, sim median `0.0000`
- `frac_P4_1010`: real median `0.0018`, sim median `0.0000`
- `frac_P1_0101`: real median `0.0020`, sim median `0.0000`
- `frac_P3_1011`: real median `0.0048`, sim median `0.0000`

Patterns like `1001`, `1010`, and `0101` are especially suspicious because they activate separated strips. In a noiseless detector they should be rare to absent.

### Correlations

- `corr(nonadj_double_frac_mean, tripleplus_frac_mean)` changes from `0.012` in sim to `0.908` in real.
- `corr(nonadj_over_adj_mean, active_rate_mean)` changes from `-0.015` in sim to `0.791` in real.
- `corr(nonadj_double_rate_mean, tripleplus_rate_mean)` changes from `0.015` in sim to `0.913` in real.

### Interpretation

This is the strongest evidence that the hidden contamination is real detector/electronics behavior and not only a fit artifact. In real data, once non-adjacent strip activity appears, triple-or-more activity rises with it and both scale with total activity. That is exactly what one expects from a mixture of:

- strip crosstalk,
- accidental extra strip firing,
- cluster splitting or charge-sharing tails that are broader than the simulation,
- pickup or noisy periods that activate disconnected strips.

## Task 4: Fit Backgrounds and Residual Tails

### Main shifts

| Metric | Sim median | Real median | Reading |
|---|---:|---:|---|
| `background_adj_pair_mean` | `0.00460` | `0.00966` | Adjacent-pair backgrounds roughly double in real. |
| `background_nonadj_pair_mean` | `0.00171` | `0.00887` | Non-adjacent-pair backgrounds increase by much more. |
| `background_triplet_mean` | `0.00566` | `0.01069` | Triplet backgrounds also roughly double. |
| `background_fourfold_mean` | `0.00647` | `0.01095` | Fourfold background is also inflated. |
| `background_nonadj_over_adj` | `0.364` | `0.957` | In real, non-adjacent pairs become nearly as background-heavy as adjacent pairs. |
| `sigmoid_width_adj_pair_mean` | `0.3818` | `0.4766` | Real fitted edges are broader. |
| `sigmoid_width_nonadj_pair_mean` | `0.3970` | `0.5509` | Broadening is even larger on non-adjacent pairs. |
| `res_ystr_triplet_mean` | `1.540` | `3.050` | Triplet spatial residuals increase strongly. |
| `res_ystr_fourfold_mean` | `1.576` | `2.852` | Fourfold spatial residuals also broaden. |
| `ext_res_ystr_fourfold_mean` | `1.388` | `24.773` | Fourfold external tails explode in real data. |
| `res_tdif_triplet_mean` | `0.0103` | `0.0170` | Triplet timing-difference residuals broaden. |
| `ext_res_tdif_fourfold_mean` | `0.0132` | `0.2119` | Fourfold external timing tails grow by more than an order of magnitude. |

### Background-by-combination medians

| Combo | Sim median | Real median |
|---|---:|---:|
| `12` | `0.00421` | `0.01110` |
| `23` | `0.00453` | `0.00931` |
| `34` | `0.00506` | `0.00817` |
| `13` | `0.00289` | `0.01093` |
| `24` | `0.00139` | `0.00855` |
| `14` | `0.00000` | `0.00750` |
| `1234` | `0.00647` | `0.01095` |

The `14` channel is especially telling: it is almost background-free in simulation and clearly nonzero in real data.

### Correlations

- `corr(background_adj_pair_mean, res_ystr_triplet_mean)` increases from `0.269` to `0.373`.
- `corr(background_nonadj_pair_mean, ext_res_ystr_fourfold_mean)` stays weak, from `-0.005` to `-0.099`.
- `corr(background_triplet_mean, res_tdif_triplet_mean)` stays modest, from `0.165` to `0.098`.

### Interpretation

The fit stage confirms the presence of contamination, but it does not look like one single scalar background parameter explains everything. The largest anomalies are in the fourfold external tails, which suggests a mixed population: many files still look reasonable, but a smaller subset generates very bad outlier residuals. That is consistent with intermittent noisy periods or combinatorial contamination surviving into the fit stage.

## Where The Noise Could Be Hiding

The metadata points to more than one hiding place.

1. Strip-topology level.
   `TASK_3` shows the clearest hidden signal: non-adjacent and triple-plus active-strip patterns are systematically present in real data and essentially absent in the noiseless simulation.

2. Channel and retention level.
   `TASK_1` and `TASK_2` show nonzero zeroed percentages, entry loss, and front/back asymmetry. This suggests that part of the contamination is already present before track fitting and is not created by the reconstruction stage.

3. Edge and outer-plane sensitivity.
   In `TASK_1` and `TASK_2`, several of the worst channels are edge strips, especially strip 4 and plane 4. That points to geometry-dependent vulnerability, thresholding issues, or channel-specific instability.

4. Reconstruction-tail population.
   `TASK_4` shows that some contamination survives into the fit tails, especially in fourfold external residuals. This is likely not pure random noise only; it looks like a mixture of strip-level contamination and event-class mis-reconstruction.

## Main Working Hypothesis

The most defensible working hypothesis from these four tasks is:

- the real station has a common-mode excess activity regime that is absent in the simulation,
- that regime produces disconnected strip patterns and triple-plus strip activity,
- some of that activity is strongest in inner planes `P2` and `P3`,
- some losses are concentrated in edge strips and plane 4,
- and the surviving contaminated subset broadens fit backgrounds and creates heavy residual tails.

This is more compatible with a mixture of crosstalk, accidental strip activity, unstable electronics states, and imperfect cleaning than with one single source of noise.

## Limitations

- This is a station-to-station comparison: `MINGO00` simulation branch versus `MINGO01` real branch. It is not a per-file matched study.
- The real sample here is the available 129-file `MINGO01` window, while the simulation baseline is much larger.
- Some reported effect sizes are enormous because the simulation median and IQR are exactly zero for those quantities. Those should be read as presence-versus-absence evidence, not as literal calibrated sigma values.

## Best Next Checks

1. Join these `TASK_3` per-plane non-adjacent pattern metrics with `TASK_5` trigger-type excesses (`13`, `24`, `14`, `1234`) file by file.
2. Split the `MINGO01` sample by day or run block and check whether the non-adjacent-strip and fourfold-tail signatures turn on and off coherently.
3. Inspect raw channel behavior for the strips that recur here: plane 4, strip 4; plane 2, strip 4; and the plane-4 side channels highlighted in `TASK_1`.

