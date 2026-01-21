# STEP 03 Interface Contract (Crossing -> Avalanche)

## Purpose
Apply gas-gap efficiency and avalanche modeling to crossing events, producing per-plane avalanche properties.

## Required inputs
- Input data (from STEP 02):
  - `event_id` (int)
  - `X_gen_i`, `Y_gen_i`, `T_sum_i_ns` for planes i = 1..4.
  - `tt_crossing` (optional but used for summary plots).
- Config inputs:
  - `efficiencies` (per plane), `gain`, `townsend_alpha`, `gap_mm`, `electron_sigma`.
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
Retained columns:
- `event_id` (int)
- `T_thick_s` (s) if present upstream
- `T_sum_i_ns` (ns) for planes i = 1..4

Per-plane avalanche columns (i = 1..4):
- `avalanche_ion_i` (count): number of primary ionizations.
- `avalanche_exists_i` (bool): True if at least one ionization occurred.
- `avalanche_x_i` (mm): avalanche centroid x (NaN if no avalanche).
- `avalanche_y_i` (mm): avalanche centroid y (NaN if no avalanche).
- `avalanche_size_electrons_i` (electrons): avalanche size after gain + smearing.
- `tt_avalanche` (string): concatenation of planes with avalanche_exists_i == True.

Time reference: no new time columns are introduced; upstream `T_sum_i_ns` are preserved.

## Invariants & checks
- If `avalanche_exists_i` is False, then `avalanche_x_i` and `avalanche_y_i` are NaN.
- `avalanche_ion_i` is an integer >= 0.
- `tt_avalanche` contains only digits in {1,2,3,4}.

## Failure modes & validation behavior
- Invalid `efficiencies` vectors or values outside (0,1) raise `ValueError`.
- Missing input columns for a plane cause that plane to be skipped (no new columns for that plane).
