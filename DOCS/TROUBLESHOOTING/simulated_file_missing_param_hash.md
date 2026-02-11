# Simulated File Missing `param_hash` — Default `z_positions` Used

## Symptom

The STAGE 1 / STEP 1 log shows:

```
Warning: Simulated file missing param_hash; using default z_positions.
```

This means `resolve_simulated_z_positions()` (in `MASTER/common/simulated_data_utils.py`)
could not find the file's `param_hash` in `step_final_simulation_params.csv`, so the
analysis falls back to default `z_positions = [0, 150, 300, 450]` mm — which are almost
certainly wrong for the actual detector geometry used in the simulation.

The warning appears in every TASK script under `MASTER/STAGE_1/EVENT_DATA/STEP_1/`
(`script_1_raw_to_clean.py`, `script_2_clean_to_cal.py`, etc.).

## Root Cause

Two independent bugs cause orphan files whose hash is not recorded in the CSV.

### Bug 1 — Float/Int Hash Normalisation Mismatch

**Affected files:** those whose `file_name` IS in the CSV but with a *different*
`param_hash`.

In `_normalize_hash_value()` inside
`MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py`,
integer-valued fields (`requested_rows`, `sample_start_index`) were hashed as Python
`int` when the `.dat` file was first written.  However, when the CSV is later read by
pandas, those columns become `float64` (e.g. `100000` → `100000.0`) because other rows
in the column may contain `NaN`.  When `ensure_param_hash_column()` or any downstream
code recomputes the hash from CSV data, the JSON serialisation differs:

```python
json.dumps(100000)    # → "100000"    ← original hash input
json.dumps(100000.0)  # → "100000.0"  ← after CSV round-trip
```

This produces a **different SHA-256 hash** for identical physics parameters.

### Bug 2 — Non-Atomic / Batch-Only CSV Write

**Affected files:** those whose `file_name` is NOT in the CSV at all.

`step_final_daq_to_station_dat.py` accumulated all new CSV rows in memory and wrote
them to disk **once at the end of each parameter-set batch** using a plain
`df.to_csv()` (not atomic).  If the process was interrupted (crash, OOM, timeout,
Ctrl-C, cron kill) before that final write:

- The `.dat` files already existed on disk with valid `# param_hash=…` headers.
- The corresponding CSV rows were lost — they were only in memory.
- `ensure_sim_hashes.py` then marks these files for deletion because their hash is
  "not in table".

## Diagnosis

### Quick Check

```bash
python3 MINGO_DIGITAL_TWIN/ANCILLARY/ensure_sim_hashes.py
```

Typical output when the bug is present:

```
WOULD_DELETE .../mi0008115022514.dat hash_not_in_table=773c6fd7…
...
Mode: DRY_RUN
Checked: 7828 | Has hash: 7720 | Added: 0 | Deleted: 0
       | Deleted (hash mismatch): 108 | ...
```

### Detailed Diagnosis

```bash
python3 MINGO_DIGITAL_TWIN/ANCILLARY/repair_orphan_hashes.py
```

This prints every orphan file, how it will be repaired, and a summary:

```
Scanned: 8056 | OK: 7937 | Fixed hash: 5 | Added missing: 114
       | Unrecoverable: 0 | No hash header: 0
```

## Fix Applied

### 1. `_normalize_hash_value()` — integer normalisation

**File:** `MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py`

Floats that are exact integers are now normalised to `int` before hashing, making the
hash stable across CSV round-trips:

```python
if isinstance(value, (np.floating, float)):
    if not np.isfinite(value):
        return None
    # Normalize exact-integer floats to int so that hashes are
    # stable across CSV round-trips (pandas reads int columns as
    # float64 when NaN values are present in the same column).
    if float(value) == int(value):
        return int(value)
    return float(value)
```

### 2. Atomic + incremental CSV write

**File:** `MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py`

- `df.to_csv()` replaced with `write_csv_atomic()` (temp file + `os.replace`).
- CSV is now written **after each individual file** instead of once per batch, so an
  interruption loses at most 1 row.

### 3. Repair script for existing data

**File:** `MINGO_DIGITAL_TWIN/ANCILLARY/repair_orphan_hashes.py`

Scans all `mi00*.dat` files, finds orphans, and reconstructs their CSV entries:

- **Hash mismatch** (Bug 1): adds an alias row with the file's actual hash and matched
  physics parameters.
- **Missing entirely** (Bug 2): matches to the nearest `param_set_id` by date
  proximity (filename encodes `YYDDDHHMMSS`) and adds a new row.

## Recovery Procedure

### Step 1 — Dry-run (verify what will be repaired)

```bash
python3 MINGO_DIGITAL_TWIN/ANCILLARY/repair_orphan_hashes.py
```

Review the output.  Every line should be `WOULD_ADD_ROW`, `WOULD_FIX_HASH`, or
`WOULD_ADD_ALIAS`.  Check that the assigned `param_set_id` and `z_positions` are
plausible (the date in the filename should be close to the `param_date` of the matched
set).

### Step 2 — Apply the CSV repair

```bash
python3 MINGO_DIGITAL_TWIN/ANCILLARY/repair_orphan_hashes.py --apply
```

This atomically rewrites `step_final_simulation_params.csv` with the missing/corrected
rows.

### Step 3 — Verify with `ensure_sim_hashes`

```bash
python3 MINGO_DIGITAL_TWIN/ANCILLARY/ensure_sim_hashes.py
```

Should now report `Deleted (hash mismatch): 0`.

### Step 4 — Re-run the affected analysis

Reprocess the previously-broken files through STAGE 1 / STEP 1.  They will now
resolve the correct `z_positions` from the CSV instead of using the default
`[0, 150, 300, 450]`.

## Prevention

The code fixes (normalisation + atomic/incremental write) prevent this from recurring.
If you need to verify integrity at any time, run:

```bash
python3 MINGO_DIGITAL_TWIN/ANCILLARY/repair_orphan_hashes.py
```

A healthy system should report `Fixed hash: 0 | Added missing: 0`.
