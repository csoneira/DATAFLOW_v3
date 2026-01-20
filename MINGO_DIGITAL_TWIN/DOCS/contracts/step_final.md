# STEP FINAL Interface Contract (DAQ -> Station .dat)

## Purpose
Format STEP 10 DAQ-like rows into station-style `.dat` files with timestamps and registry entries.

## Required inputs
- Input data (from STEP 10):
  - `event_id` (int) is preserved in input tables but not emitted in `.dat`.
  - `T_front_i_sj`, `T_back_i_sj`, `Q_front_i_sj`, `Q_back_i_sj` for planes i = 1..4, strips j = 1..4.
  - `T_thick_s` (optional; used for event timing if present).
- Config inputs:
  - station geometry map, output selection criteria, and date range.
- Required metadata: geometry_id and sim-run metadata (provided by upstream steps).

## Schema (guaranteed outputs)
- Output files:
  - `SIMULATED_DATA/mi0XYYDDDHHMMSS.dat` (ASCII text).
  - `SIMULATED_DATA/step_13_output_registry.json` (registry of emitted files).
- Each `.dat` line contains (per event):
  - 64 formatted values for planes ordered [4,3,2,1], fields ordered [T_front, T_back, Q_front, Q_back], strips ordered [1..4].
  - Optional `T_thick_s` appended if present in inputs.

Time reference: uses STEP 10 times; optional `T_thick_s` drives timestamp offsets when present.

## Invariants & checks
- Output line ordering follows plane order [4,3,2,1] and strip order [1,2,3,4].
- Non-finite values are rendered as 0.0 in the output format.

## Failure modes & validation behavior
- Missing geometry/station metadata raises `ValueError` or `FileNotFoundError`.
- Inconsistent presence of `T_thick_s` across input files raises `ValueError`.
