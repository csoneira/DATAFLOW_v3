---
title: STEP_1 Pattern Metadata Noise Report
description: Early comparison of new TASK_1 and TASK_3 full-pattern metadata between MINGO00 simulation and MINGO01 real data.
last_updated: 2026-03-13
status: draft
---

# STEP_1 Pattern Metadata Noise Report

## Scope

This note analyzes the new full-pattern metadata added to the STEP_1 specific metadata CSVs:

- `TASK_1`: `raw_channel_pattern_<32 bits>_rate_hz` and `clean_channel_pattern_<32 bits>_rate_hz`
- `TASK_3`: `cal_strip_pattern_<16 bits>_rate_hz` and `list_strip_pattern_<16 bits>_rate_hz`

Comparison is between:

- simulation: `STATIONS/MINGO00/.../task_{1,3}_metadata_specific.csv`
- real data: `STATIONS/MINGO01/.../task_{1,3}_metadata_specific.csv`

The main goal is to identify recurring pattern families that look more compatible with detector/electronics noise than with ordinary geometry or angular-spectrum effects.

## Data Slice And Method

This report uses only rows that already contain populated pattern-rate fields, i.e. post-change rows.
For each CSV, I kept the latest row per `filename_base`.

Current populated-row counts at report time:

| Task | MINGO00 | MINGO01 |
|---|---:|---:|
| `TASK_1` | `13` | `28` |
| `TASK_3` | `13` | `28` |

Method summary:

- Pattern rates are read directly from `*_rate_hz` metadata columns.
- For `TASK_1`, “unusual” patterns are ranked by how much rate is removed by `raw -> clean`.
- For `TASK_3`, I compared both exact 16-bit full patterns and per-plane 4-bit strip motifs.
- For `TASK_3`, the most relevant class for noise search is the set of irregular plane motifs:
  - non-adjacent doubles: `1001`, `1010`, `0101`
  - triple/quad patterns: `1110`, `1101`, `1011`, `0111`, `1111`

Interpretive note:

- Exact full-pattern differences can be driven by geometry or flux-spectrum differences.
- Per-plane irregular motifs are more diagnostic of noise/crosstalk-like behavior because they marginalize over the rest of the event topology.

## Validation

Before doing the station comparison, I checked that the new metadata is being written correctly in both stations.
For recent `MINGO00` and `MINGO01` basenames, pattern-rate columns matched exact recomputation from the underlying `.dat` and parquet files with zero numerical difference.

## Task 1: Common Channel-Noise Families In Real Data

`TASK_1` is the clearest place where recurrent noisy channel behavior appears.

### Main qualitative result

The removed population is not dominated by random, unique 32-bit patterns.
It is dominated by repeated one-sided channel families concentrated in a small channel subset, especially around:

- `P3 S1`
- `P3 S4`
- `P4 S3`
- `P2 S3`
- `P2 S4`
- `P1 S4`

This is much more consistent with persistent electronics/channel asymmetry or pickup than with normal detector occupancy.

### Most common removed exact full patterns in `MINGO01`

These are ranked by support first and removed Hz second.

| Support | Removed Hz | Raw Hz | Clean Hz | Pattern | Decoded |
|---:|---:|---:|---:|---|---|
| `27` | `0.187021` | `0.190161` | `0.003140` | `00000011110111110000001100000010` | `P1[S4:FB] P2[S1:FB,S2:B,S3:FB,S4:FB] P3[S4:FB] P4[S4:F]` |
| `26` | `15.461066` | `15.630045` | `0.168979` | `00000011110001110000001100000011` | `P1[S4:FB] P2[S1:FB,S3:B,S4:FB] P3[S4:FB] P4[S4:FB]` |
| `26` | `8.926686` | `9.297928` | `0.371242` | `00000001110001110000001100000011` | `P1[S4:B] P2[S1:FB,S3:B,S4:FB] P3[S4:FB] P4[S4:FB]` |
| `26` | `7.418213` | `9.769358` | `2.351145` | `00000001110000110000001100000011` | `P1[S4:B] P2[S1:FB,S4:FB] P3[S4:FB] P4[S4:FB]` |
| `26` | `4.234013` | `7.038983` | `2.804970` | `00000001100000110000001100000011` | `P1[S4:B] P2[S1:F,S4:FB] P3[S4:FB] P4[S4:FB]` |
| `26` | `1.262395` | `2.391926` | `1.129531` | `00000001000000110000001100000011` | `P1[S4:B] P2[S4:FB] P3[S4:FB] P4[S4:FB]` |

These are not arbitrary. They are one recurring family centered on strip-4 occupancy and one-sided activity in a few specific channels.

### Most common removed asymmetric channel motifs in `MINGO01`

Here I decomposed the 32-bit patterns into front/back channel states per strip and ranked the one-sided motifs.

| Motif | Support | Removed Hz |
|---|---:|---:|
| `P3 S1 B_only` | `28` | `58.731730` |
| `P3 S4 F_only` | `28` | `45.328530` |
| `P4 S3 F_only` | `28` | `37.359362` |
| `P2 S3 B_only` | `28` | `28.871633` |
| `P1 S4 B_only` | `28` | `25.362288` |
| `P2 S4 F_only` | `28` | `23.180264` |
| `P2 S4 B_only` | `28` | `8.895085` |
| `P2 S1 F_only` | `28` | `6.250640` |
| `P3 S4 B_only` | `28` | `4.584851` |
| `P2 S1 B_only` | `28` | `2.572108` |

This is the strongest common-noise result in the current dataset.

## Task 3: Deeper Real-vs-Sim Pattern Study

## High-level summary

The strongest exact full-pattern differences are not the best noise indicators.
They are dominated by simple, regular strip patterns that can plausibly come from geometry and directional spectrum differences.

The actual noise discriminator is the irregular per-plane motif content.
In real data:

- irregular motifs appear in every populated row,
- their rates are stable from `cal` to `list`,
- and they are effectively absent in simulation.

### Global `TASK_3` summary

| Prefix | Station | Rows | Mean total rate (Hz) | Mean unique full patterns | Mean weighted popcount | Mean irregular full-pattern rate (Hz) | Mean irregular full-pattern fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| `cal_strip_pattern` | `MINGO00` | `13` | `20.262822` | `811.54` | `4.128` | `0.000011` | `0.000001` |
| `cal_strip_pattern` | `MINGO01` | `28` | `15.827515` | `2780.82` | `3.367` | `1.523701` | `0.090894` |
| `list_strip_pattern` | `MINGO00` | `13` | `20.260748` | `810.15` | `4.128` | `0.000011` | `0.000001` |
| `list_strip_pattern` | `MINGO01` | `28` | `15.778856` | `2762.89` | `3.382` | `1.522993` | `0.091057` |

Important reading:

- Real data has fewer total `TASK_3` pattern-rate Hz than sim in this sample, but far more unique full patterns.
- About `9.1%` of the full-pattern rate in real data contains at least one irregular plane motif.
- In simulation, the same quantity is essentially zero.
- That fraction is unchanged from `cal` to `list`, so these patterns are not being strongly filtered away later.

## Exact full-pattern differences: mostly regular geometry families

Top `MINGO01 - MINGO00` exact full-pattern deltas at `cal` stage:

| Pattern | Real mean Hz | Sim mean Hz | Delta Hz |
|---|---:|---:|---:|
| `0010010001001000` | `0.169344` | `0.041156` | `0.128188` |
| `0100100000000000` | `0.278553` | `0.157003` | `0.121550` |
| `0001001000100100` | `0.161887` | `0.040865` | `0.121021` |
| `0010001000000000` | `0.141712` | `0.030090` | `0.111622` |
| `0010010000000000` | `0.135779` | `0.024523` | `0.111256` |
| `0010010000100100` | `0.109010` | `0.000000` | `0.109010` |

These are mostly regular one-strip patterns distributed across planes.
They are interesting, but by themselves they are not enough to claim noise.

## Irregular plane motifs: strong real-only signature

This is the most useful table in the report.
The numbers below are mean Hz per file after marginalizing over the rest of the 16-bit topology.

Top irregular plane-motif deltas at `cal` stage:

| Plane motif | Real mean Hz | Sim mean Hz | Delta Hz | Real support | Sim support |
|---|---:|---:|---:|---:|---:|
| `P2 1001` | `0.235803` | `0.000000` | `0.235803` | `28` | `0` |
| `P3 1001` | `0.214085` | `0.000000` | `0.214085` | `28` | `0` |
| `P1 1001` | `0.097768` | `0.000000` | `0.097768` | `28` | `0` |
| `P4 1001` | `0.091579` | `0.000000` | `0.091579` | `28` | `0` |
| `P3 1111` | `0.085212` | `0.000000` | `0.085212` | `28` | `0` |
| `P2 1111` | `0.080781` | `0.000000` | `0.080781` | `28` | `0` |
| `P2 1101` | `0.074688` | `0.000000` | `0.074688` | `28` | `0` |
| `P3 1011` | `0.070233` | `0.000000` | `0.070233` | `28` | `0` |
| `P4 1110` | `0.064713` | `0.000000` | `0.064713` | `28` | `0` |
| `P4 1111` | `0.055826` | `0.000000` | `0.055826` | `28` | `0` |

The same list is essentially unchanged at `list` stage.

### Why this matters

`1001` is a non-adjacent double-strip pattern.
In a clean strip topology model, it should be much rarer than adjacent doubles or singles.
Here it is:

- present in every populated real file,
- absent in every populated simulated file,
- stable across `cal -> list`,
- and visible in all four planes.

That is a strong candidate for noise, pickup, or crosstalk-like contamination.

## Are these irregular motifs actually prominent?

Yes. They are not the dominant motifs, but they are not negligible.

Top regular plane motifs in real data at `cal` stage, for scale:

| Plane motif | Real mean Hz |
|---|---:|
| `P3 0001` | `3.594389` |
| `P2 1000` | `3.560015` |
| `P1 0001` | `2.966002` |
| `P4 1000` | `2.834832` |
| `P2 0100` | `2.391434` |
| `P3 0010` | `2.365530` |
| `P3 0100` | `2.291556` |
| `P2 0010` | `2.290742` |

The top irregular motifs are lower than the dominant regular singles, but still operationally prominent:

- `P2 1001`: `0.236 Hz`
- `P3 1001`: `0.214 Hz`
- `P1 1001`: `0.098 Hz`
- `P4 1001`: `0.092 Hz`
- `P3 1111`: `0.085 Hz`
- `P2 1111`: `0.081 Hz`

These irregular motifs rank around positions `29` to `34` among all active real plane motifs.
That is too high to dismiss as negligible tails, especially because simulation gives zero support for them.

## Top irregular exact full patterns in real data

These are exact 16-bit full patterns containing at least one irregular plane motif, ranked by `MINGO01 - MINGO00` delta at `cal` stage.

| Pattern | Real mean Hz | Sim mean Hz | Real support | Decoded |
|---|---:|---:|---:|---|
| `0001100100000000` | `0.040383` | `0.000000` | `28` | `P1:0001 P2:1001` |
| `0000000010010001` | `0.033452` | `0.000000` | `28` | `P3:1001 P4:0001` |
| `1000100100000000` | `0.026067` | `0.000000` | `28` | `P1:1000 P2:1001` |
| `0000000010011000` | `0.026027` | `0.000000` | `28` | `P3:1001 P4:1000` |
| `0001000110010000` | `0.023277` | `0.000000` | `28` | `P1:0001 P2:0001 P3:1001` |
| `0000100110001000` | `0.016705` | `0.000000` | `27` | `P2:1001 P3:1000 P4:1000` |
| `0001000100011001` | `0.014100` | `0.000000` | `28` | `P1:0001 P2:0001 P3:0001 P4:1001` |
| `0000000000011001` | `0.012259` | `0.000000` | `28` | `P3:0001 P4:1001` |

Again, the same structures remain at `list` level.

## Interpretation

### Task 1

`TASK_1` already exposes a repeated channel-side asymmetry pattern:

- strong one-sided activity on a small set of channels,
- repeated strip-4-heavy families,
- and large raw-to-clean suppression concentrated on those same channels.

This is hard to explain with pure physical occupancy.
It looks like a channel/electronics signature.

### Task 3

The strongest exact full-pattern deltas are not necessarily noise.
Many are regular one-strip geometries and could reflect real-vs-sim track-distribution differences.

The real noise evidence is instead:

1. Irregular plane motifs (`1001`, `1111`, `1101`, `1011`, `1110`, etc.) are present in every real file.
2. The same motifs are absent or effectively zero in every simulated file.
3. Their rates are not tiny: the leading `1001` motifs are around `0.09` to `0.24 Hz` per plane.
4. The full-pattern rate containing at least one irregular plane motif is about `1.52 Hz`, or about `9.1%` of the real `TASK_3` pattern rate.
5. This does not collapse from `cal` to `list`, so the effect survives later filtering.

That combination is strongly compatible with a real detector/electronics effect rather than only a simulation-spectrum mismatch.

## Direct same-pattern agreement test

This section uses a later and fuller snapshot than the earlier exploratory tables above:

- `MINGO00 TASK_1`: `12267` populated rows
- `MINGO01 TASK_1`: `35` populated rows
- `MINGO00 TASK_3`: `11013` populated rows
- `MINGO01 TASK_3`: `34` populated rows

The test here is different from the anomaly ranking above.
Instead of asking which patterns are strange, it asks whether the same pattern IDs carry comparable rates in simulation and in real data.

Method:

1. compute the mean rate of each pattern over all currently populated rows,
2. compare real and simulated rates for the exact same pattern IDs,
3. separate:
   - rate carried by common patterns,
   - rate carried only by real patterns,
   - rate carried only by simulated patterns,
4. evaluate correlation only on the common support.

### Summary table

| Comparison | Real common frac | Real-only frac | Sim common frac | Sim-only frac | Pearson on common | Spearman on common | Reading |
|---|---:|---:|---:|---:|---:|---:|---|
| `TASK_1 raw exact` | `0.3471` | `0.6529` | `0.9971` | `0.0029` | `0.6963` | `0.4290` | Sim support is almost a subset of real support, but it misses most of the real rate mass. |
| `TASK_1 clean exact` | `0.4069` | `0.5931` | `0.9970` | `0.0030` | `0.6986` | `0.4267` | Cleaning helps only slightly; the structural miss remains. |
| `TASK_3 cal exact` | `0.8555` | `0.1445` | `0.99998` | `0.00002` | `0.6872` | `0.5837` | Exact-pattern comparison is usable, but clearly biased. |
| `TASK_3 list exact` | `0.8595` | `0.1405` | `0.99998` | `0.00002` | `0.6868` | `0.5770` | Same conclusion after later filtering. |
| `TASK_3 cal plane motifs` | `0.9633` | `0.0367` | `1.0000` | `0.0000` | `0.3735` | `0.2479` | The motif space overlaps, but the relative motif balance is wrong. |
| `TASK_3 cal plane motifs, regular-only` | `1.0000` | `0.0000` | `1.0000` | `0.0000` | `0.2337` | `0.0750` | Removing irregular motifs does not restore agreement. |

### What this means for Task 1

`TASK_1` is not a good global same-pattern validation test.

The reason is not only correlation.
The decisive point is support mismatch:

- about `65.3%` of the real raw pattern rate is carried by patterns that simulation never produces,
- even after cleaning, about `59.3%` of the real clean pattern rate still sits in patterns with no simulated counterpart,
- while almost all simulated rate lies inside the common support.

So the simulation is not producing a different weighting of the same `TASK_1` patterns.
It is producing a much smaller subset of the real pattern space.

The common subset is not random noise: its common-pattern Pearson correlation is about `0.70`.
But that is not enough to rescue the test, because most of the real pattern mass is outside that common subset.

In short:

- `TASK_1` same-pattern comparison is structurally weak,
- useful for identifying channel-side anomalies,
- not reliable as a full pattern-rate closure test.

### What this means for Task 3 exact full patterns

`TASK_3` behaves very differently.

At exact 16-bit strip-pattern level:

- about `85.6%` to `86.0%` of the real rate lies on exact patterns that simulation also produces,
- effectively all simulated rate lies on patterns that also exist in real data,
- correlations on the common support are moderate but clearly non-zero:
  - Pearson about `0.687`,
  - Spearman about `0.58`.

So the exact same-pattern comparison for `TASK_3` is not terrible.
It is actually usable as a validation test.

But it is not good enough to claim closure.
There is a systematic redistribution inside the common support.

Top common exact patterns that are higher in real than in simulation at `cal` stage:

| Pattern | Real mean Hz | Sim mean Hz | Real/Sim | Decoded |
|---|---:|---:|---:|---|
| `0010010001001000` | `0.170328` | `0.040309` | `4.23` | `P1:0010 P2:0100 P3:0100 P4:1000` |
| `0100100000000000` | `0.280199` | `0.151759` | `1.85` | `P1:0100 P2:1000` |
| `0001001000100100` | `0.162657` | `0.040404` | `4.03` | `P1:0001 P2:0010 P3:0010 P4:0100` |
| `0010001000000000` | `0.141675` | `0.028837` | `4.91` | `P1:0010 P2:0010` |
| `0010010000000000` | `0.135690` | `0.023184` | `5.85` | `P1:0010 P2:0100` |

Top common exact patterns that are higher in simulation than in real:

| Pattern | Real mean Hz | Sim mean Hz | Real/Sim | Decoded |
|---|---:|---:|---:|---|
| `1100100000000000` | `0.074135` | `0.391556` | `0.19` | `P1:1100 P2:1000` |
| `0000000000010011` | `0.083087` | `0.394148` | `0.21` | `P3:0001 P4:0011` |
| `0011001100010000` | `0.009704` | `0.184388` | `0.05` | `P1:0011 P2:0011 P3:0001` |
| `0000100011001100` | `0.014077` | `0.178754` | `0.08` | `P2:1000 P3:1100 P4:1100` |
| `1100100011001000` | `0.014943` | `0.174881` | `0.09` | `P1:1100 P2:1000 P3:1100 P4:1000` |

This is already informative:

- real data overproduces several single-strip chain patterns,
- simulation overproduces several adjacent-double-loaded patterns,
- so even within the shared exact-pattern support, the occupancy balance is different.

### What this means for Task 3 per-plane motifs

The per-plane motif view is harsher than the exact full-pattern view.

All simulated plane motifs are present in real data, and only about `3.7%` of the real motif rate sits in real-only motifs.
So the support overlap is almost complete.

But the relative motif weights are badly mismodelled:

- common-motif Pearson is only about `0.37`,
- common-motif Spearman is only about `0.25`,
- restricting to regular motifs only makes it worse rather than better.

The dominant common shifts are:

| Motif | Real minus sim Hz | Direction |
|---|---:|---|
| `P2 0100` | `+1.196` | real higher |
| `P1 0010` | `+1.113` | real higher |
| `P3 0010` | `+1.111` | real higher |
| `P3 0100` | `+1.094` | real higher |
| `P2 0010` | `+1.084` | real higher |
| `P2 0110` | `-1.910` | sim higher |
| `P3 0110` | `-1.887` | sim higher |
| `P2 1100` | `-1.840` | sim higher |
| `P3 1100` | `-1.782` | sim higher |
| `P3 0011` | `-1.764` | sim higher |

So the issue is not only the presence of real-only irregular motifs such as `1001`.
Even inside the regular motif family, the simulation favors adjacent doubles far more than the real data does, while the real data favors single-strip motifs more strongly.

### Bottom line of the same-pattern test

The two tasks do not fail in the same way.

- `TASK_1`: same-pattern comparison is largely broken as a global closure test, because most real rate lives outside the simulated pattern support.
- `TASK_3 exact`: same-pattern comparison holds well enough to be useful, because most real rate is on patterns shared with simulation.
- `TASK_3 motifs`: the comparison exposes a second problem beyond noise-only tails, namely a systematic redistribution from simulated adjacent doubles to real singles.

So the new pattern metadata is doing two useful jobs at once:

1. it isolates real-only noisy structures,
2. it also shows where the simulated occupancy balance is wrong even on patterns that both datasets share.

## Working Hypothesis

At the moment, the most defensible hypothesis is:

- `TASK_1` is showing persistent one-sided channel activity on a small, repeated subset of channels.
- `TASK_3` is showing the strip-level consequence of that same problem as non-adjacent doubles and triple/quad-like occupancy that are effectively absent in simulation.

In other words, the new metadata is starting to isolate a likely noise/crosstalk family rather than only showing a generic real-vs-sim mismatch.

## Next Useful Follow-up

The next high-value step would be to decode the suspicious patterns against hardware topology:

1. map the repeated `TASK_1` channel motifs to exact front/back electronics chains,
2. check whether the strongest `TASK_3` irregular motifs align with the same planes/strips,
3. compare those channels against HV groups, front-end grouping, or known lab-log anomalies.
