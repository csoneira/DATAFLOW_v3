#!/usr/bin/env python3
"""
Prune step_final_simulation_params.csv to only keep rows for .dat files that are
either:
  - In process:  present under TASK_*/INPUT_FILES/*/* (any INPUT_FILES subdir)
  - Processed:   listed in TASK_*/METADATA/task_*_metadata_execution.csv

Any row whose file_name is found in neither location is removed.  This prevents
the registry from growing indefinitely as simulations complete and files move
through the downstream pipeline.

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

ROOT = pathlib.Path.home() / "DATAFLOW_v3"
STEP_FINAL_CSV = ROOT / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA" / "step_final_simulation_params.csv"
STATIONS_STEP1 = ROOT / "STATIONS" / "MINGO00" / "STAGE_1" / "EVENT_DATA" / "STEP_1"


def ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_active_file_names() -> set[str]:
    """Return the set of .dat file names that are active in the downstream pipeline."""
    active: set[str] = set()

    # In process: every file present anywhere under TASK_*/INPUT_FILES/*/*
    for path in STATIONS_STEP1.glob("TASK_*/INPUT_FILES/*/*"):
        if path.is_file():
            active.add(path.name)

    # Processed: filename_base column in task_*_metadata_execution.csv (+ ".dat")
    for meta_csv in STATIONS_STEP1.glob("TASK_*/METADATA/task_*_metadata_execution.csv"):
        try:
            with meta_csv.open(newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    basename = (row.get("filename_base") or "").strip()
                    if basename:
                        active.add(basename + ".dat")
        except OSError as exc:
            print(
                f"{ts()} [PRUNE_FINAL] warn=read_error file={meta_csv} err={exc}",
                file=sys.stderr,
            )

    return active


def prune(dry_run: bool = False) -> None:
    if not STEP_FINAL_CSV.exists():
        print(f"{ts()} [PRUNE_FINAL] status=no_csv path={STEP_FINAL_CSV}")
        return

    active = collect_active_file_names()
    print(f"{ts()} [PRUNE_FINAL] active_downstream_files={len(active)}")

    rows_total = 0
    kept: list[dict] = []
    fieldnames: list[str] = []

    with STEP_FINAL_CSV.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows_total += 1
            file_name = (row.get("file_name") or "").strip()
            if file_name in active:
                kept.append(row)

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
    )


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    prune(dry_run=dry_run)
