# Stage 1 product tests

These scripts inspect Stage 1 products without changing production data. Their
temporary outputs live below each station's `STAGE_1_PRODUCTS_TESTS` directory.

## Test 1: Task 2 calibration offsets

Edit `config_test_1_calibration_offsets.yaml`, then run:

```bash
python3 test_1_plot_calibration_offsets.py
```

To use another config:

```bash
python3 test_1_plot_calibration_offsets.py --config /path/to/config.yaml
```

The interval refers to acquisition time (`YYDDDHHMMSS` in `filename_base`),
not the later metadata execution timestamp. If Task 2 contains repeated rows
for one acquisition, the latest execution is retained.

Outputs are written to:

```text
MINGO_ANALYSIS_STATIONS/MINGO0X/STAGE_1_PRODUCTS_TESTS/
  TEST_1_CALIBRATION_OFFSETS/
```

The test creates charge (`Q_F`, `Q_B`, and applied `Q_sum`), `T_dif`, `T_sum`,
and per-channel slewing-model PNG plots. A calibration-availability plot shows
how many of the 16 channels have a nonzero `T_sum` offset and a fitted slewing
model for every acquisition. The selected CSV includes the source metadata and
these two availability counts.


## Test 2: consecutive-file event distributions

Edit `config_test_2_event_distributions.yaml`, then run:

```bash
python3 test_2_plot_event_distributions.py
```

The script searches the selected station parquet lake within the acquisition
interval, sorts files chronologically, and selects up to `max_datafiles` as one
contiguous block. If more files are available, it chooses the block with the
smallest span between its first and last acquisition starts.

Every column name and parquet type present in the selected files is printed at
startup. `histogram_columns` accepts exact names or shell-style patterns.
`histogram_grids` creates detector-layout figures with planes as rows and
strips as columns. `scatter_pairs` accepts explicit `x`/`y` names, while
`per_strip_scatter_pairs` creates one detector-layout 4x4 scatter figure for
each suffix pair, with planes as rows and strips as columns. Axes are shared
across all 16 panels in each figure.

`plane_pair_matrix` adds a separate 4x10 figure built only from final
per-plane columns. Its rows are planes 1--4 and its columns are all unique
pairs among `Q_sum`, `Q_dif`, `T_sum`, `X`, and `Y`. Here `X` is derived from
the pipeline-stored `p#_xpos` and `Y` is `p#_ypos`; no per-strip column enters
this matrix.

The optional `rate_timeseries` plot overlays event rates before and after
the configured filter flags. Events are counted first in each exact timestamp
second, then accumulated over `accumulation_timespan`. Both curves divide by
the number of seconds actually represented in the input window, never blindly
by 60; the output unit is always Hz. Its CSV records the observed-second
denominator, event totals, and both rates for every window.

Before plotting, `calibrated_value_limits` applies inclusive bounds to every
matching calibrated plane/strip column. `per_plane_value_limits` does the
same for final plane observables such as `p#_xpos`, `p#_ypos`, and
`p#_qsum`. Nonzero values outside a configured range are replaced by zero.
Exact zero values are excluded from all histogram
inputs (joined and per-file overlays), and scatter rows are omitted whenever
either plotted value is zero.

The default config creates four 4x4 grids for calibrated `Q_sum`, `Q_dif`,
`T_sum`, and `T_dif`, plus a standalone event-slowness (`event_s`) histogram.
It creates 4x4 grids for `T_sum` versus `T_dif`, `T_dif` versus `Q_sum`,
`Q_sum` versus `Q_dif`, and `T_dif` versus `Q_dif`, plus
`event_s`-versus-`event_charge` and `event_phi`-versus-`event_theta` standalone
scatters. Histograms show the joined
distribution plus one outline per source file; scatter points are colored by
source file.

Outputs and a `selected_files.csv` manifest are written below:

```text
MINGO_ANALYSIS_STATIONS/MINGO0X/STAGE_1_PRODUCTS_TESTS/
  TEST_2_EVENT_DISTRIBUTIONS/<interval_and_selected_group>/
```


## Test 3: configurable binary event gates

Edit `config_test_3_event_gates.yaml`, then run:

```bash
python3 test_3_configurable_event_gates.py
```

Test 3 uses the same station, acquisition interval, and tightest consecutive
file-block selection as Test 2. Each configured gate has a distinct binary
power-of-two code (`1`, `10`, `100`, `1000`, ...). An event satisfying several
gates receives their sum as an exact combined binary code: gates `1000`, `10`,
and `1`, for example, produce `1011`.

Gate conditions are nested YAML trees using `all`, `any`, and `not`. Leaves
name a product or derived column and an operator. The supported operators are
`lt`, `le`, `gt`, `ge`, `between`, `eq`, `ne`, `in`, `not_in`, `is_finite`,
`is_zero`, `nonzero`, `isna`, `notna`, and full-regex `matches`. Conditions are
interpreted by the script; arbitrary Python expressions are never executed.

For strip-cluster studies the script derives `p#_cluster_size` and
`p#_strip_topology` from the four configured calibrated strip-charge columns.
For example, `p2_strip_topology == "1001"` means strips 1 and 4 are active in
plane 2. Always quote topology values so leading zeros are preserved.

The default config defines the three starter gates: all four plane charges
below 80; cluster size 0 or 1 in every plane; and at least one plane with topology
`1001` whose strip-1 and strip-4 calibrated T_dif values are both in inclusive
`[-1, 1]`, while every other plane has cluster size 0 or 1. Multiple valid
`1001` planes are accepted. Outputs include individual-gate and exact
combined-code rate time series, a shared two-panel theta density comparison
(individual gates together and exact combined codes together), its numeric CSV,
a 4x4 gate-100 scatter matrix with planes as columns and calibrated Q_sum,
T_dif, Q_dif, and T_sum strip-1-versus-strip-4 comparisons as rows for
per-plane `1001` patterns, a gate population summary, and the selected-file
manifest. Rate denominators
use timestamp-seconds actually observed in each accumulation window.

Outputs are written below:

```text
MINGO_ANALYSIS_STATIONS/MINGO0X/STAGE_1_PRODUCTS_TESTS/
  TEST_3_CONFIGURABLE_GATES/<interval_and_selected_group>/
```
