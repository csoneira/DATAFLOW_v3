You are working locally inside:

`/home/mingo/DATAFLOW_v3`

Do not clone, reset, stash, checkout, or discard local changes. Work only on the current local tree.

Task: implement the column-renaming and cleanup guide pasted below.

Scope:

* `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_0/`
* `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_1/`
* `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_2/`
* `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_3/`
* `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_4/`
* `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_5/`
* relevant config files under `MASTER/CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1/`
* relevant shared/helpers under `MASTER/common/`
* relevant follow-column/observability utilities under `OPERATIONS/OBSERVABILITY/FOLLOW_COLUMNS/`

Instructions:

1. Read the full guide below before editing.
2. Treat the guide as the source of truth for column names.
3. Apply rename rules globally once a column family is introduced. Do not reintroduce old aliases downstream.
4. Keep `acquisition_type`, `datetime`, `event_id`, and `param_hash` untouched.
5. If old parquet compatibility is needed, add one small input-normalization mapping near file loading. Do not spread legacy names through the code.
6. Remove code only where the guide explicitly says `REMOVE`, `STOP USING`, or equivalent.
7. Do not refactor unrelated logic.
8. Prefer generated mappings/helper functions for repeated detector-column patterns.
9. Keep behavior unchanged except for the requested renames, removals, and topology-column additions.

Important topology requirements:

* Create `topology_task1_channel` in Task 1 context, using the 32 raw end-level charge channels.
* Create `topology_task2_strip` already in Task 2, using the 16 strip `qsum` columns.
* Create `topology_task3_plane` in Task 3, using the 4 plane `qsum` columns.
* Replace or avoid `active_strips_P{plane}` where the new topology columns provide the same information.

Before editing, run:

```bash
git diff --stat
git diff --name-only
```

After editing, run:

```bash
python -m py_compile MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_0/script_0_acq_to_raw.py
python -m py_compile MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_1/script_1_raw_to_clean.py
python -m py_compile MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_2/script_2_clean_to_cal.py
python -m py_compile MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_3/script_3_cal_to_list.py
python -m py_compile MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_4/script_4_list_to_fit.py
python -m py_compile MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_5/script_5_fit_to_post.py
```

Then search for remaining legacy names:

```bash
rg -n "Q[1-4]_[FB]_[1-4]|T[1-4]_[FB]_[1-4]" MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1 MASTER/CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1 MASTER/common OPERATIONS/OBSERVABILITY/FOLLOW_COLUMNS

rg -n "Q[1-4]_Q_(sum|dif)_[1-4]|T[1-4]_T_(sum|dif)_[1-4]" MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1 MASTER/CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1 MASTER/common OPERATIONS/OBSERVABILITY/FOLLOW_COLUMNS

rg -n "P[1-4]_(Q|T)_(sum|dif)_final|P[1-4]_Y_final|P[1-4]_Q_total_uncal" MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1 MASTER/CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1 MASTER/common OPERATIONS/OBSERVABILITY/FOLLOW_COLUMNS

rg -n "column_6|Time|Phi_pred|Theta_pred|region|adj_dis|extension_tt|processed_tt|tracking_tt|with_crstlk|no_crstlk|Q_P[1-4]s[1-4]" MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1 MASTER/CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1 MASTER/common OPERATIONS/OBSERVABILITY/FOLLOW_COLUMNS
```

Any remaining match must be reported as one of:

* deliberate backward-compatibility normalization;
* migration documentation;
* false positive;
* still-to-fix reference.

Final response must include:

* files changed;
* main rename families implemented;
* columns/code removed;
* topology columns added and where;
* validation command results;
* remaining old-name matches, if any;
* `git diff --stat`;
* remaining uncertainty.

Now implement the guide below:


# DATAFLOW_v3 column-renaming and cleanup guide

## General naming rules

Use lowercase, underscore-separated column names.

Detector coordinates must appear before the physical quantity:

```text
p{plane}_s{strip}_{quantity}
p{plane}_{quantity}
event_{quantity}
````

Coordinate convention:

```text
p{plane} = detector plane, usually 1, 2, 3, or 4
s{strip} = strip number, usually 1, 2, 3, or 4
ef       = front readout end
eb       = back readout end
q        = charge
t        = time
```

Examples:

```text
p1_s1_ef_q
p1_s1_eb_q
p1_s1_ef_t
p1_s1_eb_t
p1_s1_qsum
p1_s1_tsum
p1_qsum
p1_tsum
```

## Scope rule

When a column is renamed in one task, the new canonical name must be preserved in all downstream tasks.

Do not reintroduce legacy aliases in later tasks. If an old name appears again downstream, replace it with the canonical name or remove it if it is obsolete.

---

# Task 0, Task 1, and Task 2

## Raw end-level channel columns

These are the first detector channel columns produced by the pipeline.

```text
Q{plane}_{F/B}_{strip} --> p{plane}_s{strip}_e{f/b}_q
T{plane}_{F/B}_{strip} --> p{plane}_s{strip}_e{f/b}_t
```

Examples:

```text
Q1_F_1 --> p1_s1_ef_q
Q1_B_1 --> p1_s1_eb_q
T1_F_1 --> p1_s1_ef_t
T1_B_1 --> p1_s1_eb_t
T2_B_4 --> p2_s4_eb_t
```

## Acquisition-type column

```text
column_6 --> acquisition_type
```

`column_6` should not survive as a public pipeline column. It should already be renamed to `acquisition_type` in Task 0.

---

# Task 0: acquisition to raw

## Task-time columns

```text
acq_tt --> tt_task0_acq
raw_tt --> tt_task0_raw
```

Create the Task 0 transfer-time column:

```text
CREATE: transferred_task0_acq_to_raw
```

## Channel topology

Create a new Task 1 input/topology helper column:

```text
topology_task1_channel
```

Definition:

```text
A 32-bit value encoding whether each raw end-level charge channel is zero or nonzero.
There are 4 planes × 4 strips × 2 ends = 32 channels.
Use the canonical raw charge columns:
p{plane}_s{strip}_ef_q
p{plane}_s{strip}_eb_q
```

---

# Task 1: raw to clean

## Task-time columns

```text
clean_tt --> tt_task1_clean
raw_to_clean_tt --> transferred_task1_raw_to_clean
```

## Problematic-channel filter columns

```text
task1_problematic_channel_count --> filter_task1_problematic_channel_count
task1_problematic_channel_resolution_exact --> filter_task1_problematic_channel_exact
```

---

# Task 2 and Task 3

## Strip-level calibrated charge and time columns

These are the calibrated per-strip semisum and semidifference quantities.

```text
Q{plane}_Q_dif_{strip} --> p{plane}_s{strip}_qdif
Q{plane}_Q_sum_{strip} --> p{plane}_s{strip}_qsum
T{plane}_T_dif_{strip} --> p{plane}_s{strip}_tdif
T{plane}_T_sum_{strip} --> p{plane}_s{strip}_tsum
```

Examples:

```text
Q1_Q_dif_4 --> p1_s4_qdif
Q1_Q_sum_1 --> p1_s1_qsum
T1_T_dif_4 --> p1_s4_tdif
T1_T_sum_1 --> p1_s1_tsum
```

## Uncalibrated plane charge

```text
P{plane}_Q_total_uncal --> p{plane}_qsum_uncalibrated
```

Examples:

```text
P1_Q_total_uncal --> p1_qsum_uncalibrated
P4_Q_total_uncal --> p4_qsum_uncalibrated
```

---

# Task 2: clean to calibrated

## Task-time columns

```text
cal_tt --> tt_task2_cal
clean_to_cal_tt --> transferred_task2_clean_to_cal
```

## Problematic-strip filter columns

```text
task2_problematic_strip_count --> filter_task2_problematic_strip_count
task2_problematic_strip_resolution_exact --> filter_task2_problematic_strip_exact
```

## Strip topology

Rename and define the strip-level topology column already in Task 2:

```text
plane_charge_topology_code --> topology_task2_strip
```

Definition:

```text
A 16-bit value encoding whether each strip charge is zero or nonzero.
There are 4 planes × 4 strips = 16 strips.
Use the canonical strip charge columns:
p{plane}_s{strip}_qsum
```

Important:

```text
Define topology_task2_strip in Task 2.
Do not wait until Task 3 to create it.
```

---

# Task 3 and Task 4

## Plane-level final quantities

These are the final plane-level quantities selected for downstream list/fit stages.

```text
P{plane}_Q_dif_final --> p{plane}_qdif
P{plane}_Q_sum_final --> p{plane}_qsum
P{plane}_T_dif_final --> p{plane}_tdif
P{plane}_T_sum_final --> p{plane}_tsum
P{plane}_Y_final --> p{plane}_ypos
```

Examples:

```text
P1_Q_dif_final --> p1_qdif
P1_Q_sum_final --> p1_qsum
P1_T_dif_final --> p1_tdif
P1_T_sum_final --> p1_tsum
P1_Y_final --> p1_ypos
```

## Chi-square threshold columns

```text
th_chi_{ndf} --> th_chisq_df_{ndf}
```

Examples:

```text
th_chi_0 --> th_chisq_df_0
th_chi_3 --> th_chisq_df_3
th_chi_6 --> th_chisq_df_6
```

## Column to remove

Remove `adj_dis` and all code dedicated exclusively to producing or using it:

```text
adj_dis --> REMOVE COMPLETELY
```

---

# Task 3: calibrated to list

## Task-time columns

```text
list_tt --> tt_task3_list
cal_to_list_tt --> transferred_task3_cal_to_list
```

## Problematic-plane filter columns

```text
task3_problematic_plane_count --> filter_task3_problematic_plane_count
task3_problematic_plane_resolution_exact --> filter_task3_problematic_plane_exact
```

## Plane topology

Create a new plane-level topology column:

```text
topology_task3_plane
```

Definition:

```text
A 4-bit value encoding whether each plane charge is zero or nonzero.
There are 4 planes.
Use the canonical plane charge columns:
p{plane}_qsum
```

## Active strip columns

Avoid using:

```text
active_strips_P{plane}
```

Reason:

```text
The strip-activity information should already be encoded in topology_task2_strip.
```

If possible, replace logic based on `active_strips_P{plane}` with logic based on `topology_task2_strip`.

---

# Task 4 and Task 5

## Tracking residuals

```text
ext_res_tdif_{plane} --> p{plane}_tdif_res_ext
res_tdif_{plane} --> p{plane}_tdif_res

ext_res_tsum_{plane} --> p{plane}_tsum_res_ext
res_tsum_{plane} --> p{plane}_tsum_res

ext_res_ystr_{plane} --> p{plane}_ystr_res_ext
res_ystr_{plane} --> p{plane}_ystr_res
```

## Tracking residual uncertainties

```text
ext_res_tdif_{plane}_err --> p{plane}_tdif_res_ext_err
res_tdif_{plane}_err --> p{plane}_tdif_res_err

ext_res_tsum_{plane}_err --> p{plane}_tsum_res_ext_err
res_tsum_{plane}_err --> p{plane}_tsum_res_err

ext_res_ystr_{plane}_err --> p{plane}_ystr_res_ext_err
res_ystr_{plane}_err --> p{plane}_ystr_res_err
```

## Event fit quantities

```text
x --> event_x
y --> event_y
xp --> event_xp
yp --> event_yp
s --> event_s
t0 --> event_t0
theta --> event_theta
phi --> event_phi
```

## Event fit uncertainties

```text
x_err --> event_x_err
y_err --> event_y_err
s_err --> event_s_err
t0_err --> event_t0_err
theta_err --> event_theta_err
phi_err --> event_phi_err
```

`delta_s` represents the same quantity as `s_err`. Do not keep a separate `delta_s` column.

```text
delta_s --> event_s_err
```

## Detector- and timing-prefixed fit quantities

Apply these mechanically unless a more specific rule exists:

```text
det_{variable} --> event_det_{variable}
tim_{variable} --> event_tim_{variable}
```

Examples:

```text
det_x --> event_det_x
det_theta --> event_det_theta
det_res_tsum_1 --> event_det_res_tsum_1

tim_x --> event_tim_x
tim_theta --> event_tim_theta
tim_res_tsum_1 --> event_tim_res_tsum_1
```

## Timestamp column

Keep the timestamp column as:

```text
datetime
```

Do not rename `datetime` to `Time`.

If Task 5 or any downstream logic creates `Time`, remove that rename and keep `datetime` throughout.

```text
Time --> datetime
```

## Obsolete charge aliases

Do not keep these aliases:

```text
Q_P{plane}s{strip}
```

Use the canonical strip charge instead:

```text
p{plane}_s{strip}_qsum
```

Examples:

```text
Q_P1s1 --> p1_s1_qsum
Q_P4s3 --> p4_s3_qsum
```

## Obsolete plane charge aliases

Do not keep these plane charge aliases:

```text
charge_1 --> REMOVE; use p1_qsum
charge_2 --> REMOVE; use p2_qsum
charge_3 --> REMOVE; use p3_qsum
charge_4 --> REMOVE; use p4_qsum
```

Event charge should remain as a renamed event-level quantity:

```text
charge_event --> event_charge
```

## Tracking convergence metadata

```text
conv_distance --> timtrack_conv_distance
converged --> timtrack_converged
iterations --> timtrack_iterations
```

---

# Task 4: list to fit

## Task-time columns

```text
fit_tt --> tt_task4_fit
```

## Plane z-position columns

```text
z_P{plane} --> z_p{plane}
```

Examples:

```text
z_P1 --> z_p1
z_P4 --> z_p4
```

---

# Task 5: fit to post

## Task-time columns

```text
post_tt --> tt_task5_post
fit_to_post_tt --> transferred_task5_fit_to_post
```

## Columns to remove in Task 5

Remove these columns and all code dedicated exclusively to producing or using them:

```text
Phi_pred --> REMOVE COMPLETELY
Theta_pred --> REMOVE COMPLETELY
region --> REMOVE COMPLETELY
```

---

# Cross-task task-time columns

Use these canonical task-time names throughout the full pipeline:

```text
acq_tt   --> tt_task0_acq
raw_tt   --> tt_task0_raw
clean_tt --> tt_task1_clean
cal_tt   --> tt_task2_cal
list_tt  --> tt_task3_list
fit_tt   --> tt_task4_fit
post_tt  --> tt_task5_post
```

Use these canonical transfer-time names throughout the full pipeline:

```text
CREATE: transferred_task0_acq_to_raw

raw_to_clean_tt --> transferred_task1_raw_to_clean
clean_to_cal_tt --> transferred_task2_clean_to_cal
cal_to_list_tt  --> transferred_task3_cal_to_list
fit_to_post_tt  --> transferred_task5_fit_to_post
event_projected_tt --> tt_task4_projected
```

Stop using these obsolete timing aliases:

```text
extension_tt --> REMOVE / STOP USING
processed_tt --> REMOVE / STOP USING
tracking_tt  --> REMOVE / STOP USING
```

Reason:

```text
These aliases are not needed if the canonical task-time columns are carried:
tt_task4_fit
tt_task5_post
```

---

# Cross-task filter summary columns

```text
task1_problematic_channel_count --> filter_task1_problematic_channel_count
task1_problematic_channel_resolution_exact --> filter_task1_problematic_channel_exact

task2_problematic_strip_count --> filter_task2_problematic_strip_count
task2_problematic_strip_resolution_exact --> filter_task2_problematic_strip_exact

task3_problematic_plane_count --> filter_task3_problematic_plane_count
task3_problematic_plane_resolution_exact --> filter_task3_problematic_plane_exact

total_problematic_offender_count --> filter_total_problematic_offender_count
```

---

# Cross-task topology columns

## Task 1 channel topology

```text
CREATE: topology_task1_channel
```

Definition:

```text
A 32-bit value encoding whether each raw end-level charge channel is zero or nonzero.
There are 4 planes × 4 strips × 2 ends = 32 channels.
Use:
p{plane}_s{strip}_ef_q
p{plane}_s{strip}_eb_q
```

## Task 2 strip topology

```text
plane_charge_topology_code --> topology_task2_strip
```

Definition:

```text
A 16-bit value encoding whether each strip charge is zero or nonzero.
There are 4 planes × 4 strips = 16 strips.
Use:
p{plane}_s{strip}_qsum
```

This column must be defined in Task 2, not delayed until Task 3.

## Task 3 plane topology

```text
CREATE: topology_task3_plane
```

Definition:

```text
A 4-bit value encoding whether each plane charge is zero or nonzero.
There are 4 planes.
Use:
p{plane}_qsum
```

## Obsolete active-strip helper columns

Avoid using:

```text
active_strips_P{plane}
```

Reason:

```text
The equivalent information should already be available through topology_task2_strip.
```

---

# Cross-task cleanup rules

## Crosstalk-specific charge columns

Remove all crosstalk-specific columns and all code dedicated exclusively to producing or using them:

```text
*_crstlk --> REMOVE COMPLETELY
```

This includes, for example:

```text
Q{plane}_Q_sum_{strip}_no_crstlk
Q{plane}_Q_sum_{strip}_with_crstlk
Q_P{plane}s{strip}_with_crstlk
```

## Fully obsolete columns

Remove these columns and all code dedicated exclusively to producing or using them:

```text
adj_dis
Phi_pred
Theta_pred
region
extension_tt
processed_tt
tracking_tt
```

---

# Columns to keep untouched

Do not rename these columns:

```text
acquisition_type
datetime
event_id
param_hash
```
