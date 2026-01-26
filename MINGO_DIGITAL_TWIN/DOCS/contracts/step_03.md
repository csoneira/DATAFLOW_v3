# STEP 03 Interface Contract (Crossing -> Avalanche)

## Purpose
Apply gas-gap efficiency and avalanche modeling to crossing events.

## Required inputs
- Input data (from STEP 02):
  - `event_id`.
  - `X_gen_i`, `Y_gen_i`, `T_sum_i_ns` for planes i = 1..4.
  - `T_thick_s` (optional, preserved).
- Config inputs:
  - `efficiencies` or `random` (from param mesh).
  - `avalanche_gap_mm`, `townsend_alpha_per_mm`, `avalanche_electron_sigma`.
  - `avalanche_gain` (used for internal qsum; not persisted).

## Output schema
Outputs:
- `INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/step_3.(pkl|csv|chunks.json)`

Columns:
- `event_id`, `T_thick_s`.
- `T_sum_i_ns` (ns) for i = 1..4.
- Per-plane:
  - `avalanche_ion_i` (count).
  - `avalanche_exists_i` (bool).
  - `avalanche_x_i`, `avalanche_y_i` (mm).
  - `avalanche_size_electrons_i` (electrons).
- `tt_avalanche` (string): planes with avalanche.

## Behavior
- Ionizations follow `Poisson(lambda)` with `lambda = -ln(1 - eff_i)`.
- Avalanche size is `ions * exp(alpha * gap_mm) * LogNormal(0, electron_sigma)`.

## Metadata
- Common fields plus `param_set_id`, `param_date`, `param_row_id`.
- `step_1_id`, `step_2_id`, `step_3_id` (mesh-driven).
- `param_mesh_path`.

## Failure modes
- Efficiency values outside (0,1] raise `ValueError`.
- Missing mesh rows for `random` efficiencies raise `ValueError`.
