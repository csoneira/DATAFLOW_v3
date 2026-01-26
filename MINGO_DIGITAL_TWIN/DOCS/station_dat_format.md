# Station .dat Output Format

This document describes the ASCII .dat format emitted by STEP FINAL.

## File naming
- Simulation outputs use `mi00YYDDDHHMMSS.dat` where:
  - `YY` = year (2-digit),
  - `DDD` = day of year,
  - `HHMMSS` = time of day.
- Names are generated to be unique within the output directory.

## Line structure
Each line corresponds to one event and has the following layout:

1) Timestamp header (7 fields):
```
YYYY MM DD HH MM SS 1
```
The final "1" is a fixed flag used by the output format.

2) Channel payload (64 fields):
- Plane order: 4, 3, 2, 1
- Field order: T_front, T_back, Q_front, Q_back
- Strip order: 1, 2, 3, 4

Total fields per line: 7 + 64 = 71.

## Value formatting
- Non-finite values are written as 0.0.
- Positive values are formatted with zero padding and 4 decimals: width 9, `0000.0000`.
- Negative values are formatted with 4 decimals and a leading minus sign (no zero padding).

## Timestamp generation
- If `T_thick_s` is present, it is used as a per-event offset from a base date.
- Otherwise, event times are spaced by an exponential inter-arrival distribution
  with mean `1 / rate_hz`.

## Registries
STEP FINAL also writes:
- `SIMULATED_DATA/step_final_output_registry.json`: metadata for each emitted file.
- `SIMULATED_DATA/step_final_simulation_params.csv`: parameter summary for each file.
