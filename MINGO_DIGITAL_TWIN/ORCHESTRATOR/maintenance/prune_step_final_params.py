#!/usr/bin/env python3
"""
Prune step_final_simulation_params.csv using real on-disk file presence.

Rows are kept only when their ``file_name`` currently exists in at least one of
the known simulation/downstream locations:
  - MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES
  - MINGO_DIGITAL_TWIN/SIMULATED_DATA (legacy root .dat)
  - STATIONS/MINGO00/STAGE_0_to_1
  - STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_*/INPUT_FILES/*/*

This keeps the catalogue dynamic (manual file deletions are reflected) while
remaining robust to normal file movement between pipeline directories.

The script also removes malformed rows (empty file_name) and duplicate rows by
file_name (keeping the latest occurrence).

Writes atomically: temp file alongside the target, then os.replace().

Usage:
    python3 prune_step_final_params.py              # live run
    python3 prune_step_final_params.py --dry-run    # report only, no write
"""
from __future__ import annotations

import csv
import datetime
import pathlib
import sys
from collections import OrderedDict

ROOT = pathlib.Path.home() / "DATAFLOW_v3"
STEP_FINAL_CSV = ROOT / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA" / "step_final_simulation_params.csv"
SIM_DATA_DIR = ROOT / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA"
SIM_DATA_FILES_DIR = SIM_DATA_DIR / "FILES"
STAGE0_TO_1_DIR = ROOT / "STATIONS" / "MINGO00" / "STAGE_0_to_1"
STAGE1_STEP1_DIR = ROOT / "STATIONS" / "MINGO00" / "STAGE_1" / "EVENT_DATA" / "STEP_1"


def ts() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_present_file_names() -> tuple[set[str], dict[str, int]]:
    """
    Collect live .dat file names from known simulation/downstream directories.

    Returns:
      - Set of lowercase file names currently present or referenced by
        execution metadata.
      - Per-source counters for observability.
    """
    present: set[str] = set()
    counters: dict[str, int] = {
        "sim_files_dir": 0,
        "sim_root_legacy": 0,
        "stage0_to_1": 0,
        "stage1_inputs": 0,
        "stage1_execution_refs": 0,
    }

    for path in SIM_DATA_FILES_DIR.glob("mi00*.dat"):
        if path.is_file():
            counters["sim_files_dir"] += 1
            present.add(path.name.strip().lower())

    for path in SIM_DATA_DIR.glob("mi00*.dat"):
        if path.is_file():
            counters["sim_root_legacy"] += 1
            present.add(path.name.strip().lower())

    for path in STAGE0_TO_1_DIR.glob("mi00*.dat"):
        if path.is_file():
            counters["stage0_to_1"] += 1
            present.add(path.name.strip().lower())

    for path in STAGE1_STEP1_DIR.glob("TASK_*/INPUT_FILES/*/*"):
        if not path.is_file():
            continue
        name = path.name.strip()
        if not name.lower().startswith("mi00") or not name.lower().endswith(".dat"):
            continue
        counters["stage1_inputs"] += 1
        present.add(name.lower())

    # Keep rows that are referenced by execution metadata even when the .dat is
    # no longer physically present in the watched directories.
    for meta_csv in STAGE1_STEP1_DIR.glob("TASK_*/METADATA/task_*_metadata_execution.csv"):
        try:
            with meta_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    base = (row.get("filename_base") or "").strip().lower()
                    if not base or not base.startswith("mi00"):
                        continue
                    name = f"{base}.dat" if not base.endswith(".dat") else base
                    counters["stage1_execution_refs"] += 1
                    present.add(name)
        except OSError as exc:
            print(
                f"{ts()} [PRUNE_FINAL] warn=read_error file={meta_csv} err={exc}",
                file=sys.stderr,
            )

    return present, counters


def prune(dry_run: bool = False) -> None:
    if not STEP_FINAL_CSV.exists():
        print(f"{ts()} [PRUNE_FINAL] status=no_csv path={STEP_FINAL_CSV}")
        return

    present_names, present_counters = collect_present_file_names()
    print(
        f"{ts()} [PRUNE_FINAL] present_files_unique={len(present_names)} "
        f"sim_files_dir={present_counters['sim_files_dir']} "
        f"sim_root_legacy={present_counters['sim_root_legacy']} "
        f"stage0_to_1={present_counters['stage0_to_1']} "
        f"stage1_inputs={present_counters['stage1_inputs']} "
        f"stage1_execution_refs={present_counters['stage1_execution_refs']}"
    )

    rows_total = 0
    rows_missing_name = 0
    rows_not_found_on_disk = 0
    fieldnames: list[str] = []
    by_name: OrderedDict[str, dict] = OrderedDict()

    with STEP_FINAL_CSV.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows_total += 1
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                rows_missing_name += 1
                continue
            key = file_name.lower()
            if key in by_name:
                by_name.move_to_end(key)
            by_name[key] = row

    kept = list(by_name.values())
    rows_with_name = rows_total - rows_missing_name
    duplicate_rows_removed = rows_with_name - len(kept)
    kept_existing: list[dict] = []
    for row in kept:
        file_name = (row.get("file_name") or "").strip().lower()
        if file_name in present_names:
            kept_existing.append(row)
        else:
            rows_not_found_on_disk += 1
    kept = kept_existing
    rows_removed = rows_total - len(kept)

    if not dry_run and rows_removed > 0:
        # Atomic write: sibling temp file then rename.
        tmp = STEP_FINAL_CSV.with_name(STEP_FINAL_CSV.name + ".tmp")
        try:
            with tmp.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept)
            tmp.replace(STEP_FINAL_CSV)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    print(
        f"{ts()} [PRUNE_FINAL] status={'dry_run' if dry_run else 'done'}"
        f" total={rows_total} kept={len(kept)} removed={rows_removed}"
        f" removed_missing_name={rows_missing_name}"
        f" removed_duplicates={duplicate_rows_removed}"
        f" removed_not_found_on_disk={rows_not_found_on_disk}"
    )


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    prune(dry_run=dry_run)
