#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

DEFAULT_SIM_SOURCE_DIR = "~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES"
REGISTRY_FIELDS = ["basename", "execution_timestamp"]


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_imported_basenames(registry_path: Path) -> set[str]:
    imported: set[str] = set()
    if not registry_path.exists():
        return imported
    with registry_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basename = (row.get("basename") or "").strip()
            if basename:
                imported.add(basename)
    return imported


def normalize_registry_schema(registry_path: Path) -> int:
    if not registry_path.exists() or registry_path.stat().st_size == 0:
        return 0

    placeholder_timestamp = now_timestamp()
    normalized_rows: list[dict[str, str]] = []
    needs_rewrite = False

    with registry_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        has_basename = "basename" in fieldnames
        has_execution_timestamp = "execution_timestamp" in fieldnames
        if not has_basename or not has_execution_timestamp:
            needs_rewrite = True

        for row in reader:
            basename = (row.get("basename") or "").strip()
            if not basename:
                continue
            execution_timestamp = (row.get("execution_timestamp") or "").strip() if has_execution_timestamp else ""
            if not execution_timestamp:
                execution_timestamp = placeholder_timestamp
                needs_rewrite = True
            normalized_rows.append(
                {
                    "basename": basename,
                    "execution_timestamp": execution_timestamp,
                }
            )

    if not needs_rewrite:
        return 0

    tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_rows)
    tmp_path.replace(registry_path)
    return len(normalized_rows)


def load_simulation_param_basenames(sim_params_path: Path) -> set[str]:
    basenames: set[str] = set()
    if not sim_params_path.exists():
        return basenames
    with sim_params_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        file_name_column = "file_name" if "file_name" in fieldnames else None
        basename_column = "basename" if "basename" in fieldnames else None
        for row in reader:
            raw_name = (row.get(file_name_column) or "").strip() if file_name_column else ""
            if raw_name:
                basenames.add(Path(raw_name).stem)
                continue
            raw_basename = (row.get(basename_column) or "").strip() if basename_column else ""
            if raw_basename:
                basenames.add(Path(raw_basename).stem if raw_basename.endswith(".dat") else raw_basename)
    return basenames


def append_registry_basenames(registry_path: Path, basenames: list[str]) -> None:
    if not basenames:
        return
    write_header = (not registry_path.exists()) or registry_path.stat().st_size == 0
    execution_timestamp = now_timestamp()
    with registry_path.open("a", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(
            {
                "basename": basename,
                "execution_timestamp": execution_timestamp,
            }
            for basename in basenames
        )


def find_ground_truth_basenames(station_root: Path) -> set[str]:
    """Return basenames found in the station downstream file structure.

    Searches the two fixed locations described in the prompt and returns
    a set of filename stems that start with ``mi00``. The caller is
    responsible for filtering the prefix and for any further use.
    """
    truth: set[str] = set()

    # first location: STAGE_0_to_1/**/*. any file under that tree
    stage01 = station_root / "STAGE_0_to_1"
    for p in stage01.rglob("*"):
        if p.is_file():
            truth.add(p.stem)

    # second location: STAGE_1/EVENT_DATA/STEP_1/TASK_*/INPUT_FILES/*/*
    step1 = station_root / "STAGE_1" / "EVENT_DATA" / "STEP_1"
    for task in step1.glob("TASK_*"):
        input_base = task / "INPUT_FILES"
        for subdir in input_base.glob("*"):
            if not subdir.is_dir():
                continue
            for f in subdir.glob("*"):
                if f.is_file():
                    truth.add(f.stem)

    # filter by prefix
    return {b for b in truth if b.startswith("mi00")}


def sync_registry_with_ground_truth(registry_path: Path, station_root: Path) -> tuple[int, int]:
    """Ensure ``registry_path`` matches the ground truth set.

    Returns a pair ``(added, removed)`` counts.  The CSV is rewritten
    atomically; existing ``execution_timestamp`` values are preserved for
    basenames that remain, and a fresh timestamp is assigned to new entries.
    """
    truth = find_ground_truth_basenames(station_root)

    existing: dict[str, str] = {}
    if registry_path.exists():
        with registry_path.open("r", encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                b = (row.get("basename") or "").strip()
                if not b:
                    continue
                ts = (row.get("execution_timestamp") or "").strip()
                existing[b] = ts or now_timestamp()

    truth_set = set(truth)
    existing_set = set(existing.keys())

    to_add = sorted(truth_set - existing_set)
    to_remove = sorted(existing_set - truth_set)

    if to_add or to_remove:
        # build new row list preserving timestamps where available
        new_rows: list[dict[str, str]] = []
        timestamp_now = now_timestamp()
        for b in sorted(truth_set):
            new_rows.append(
                {
                    "basename": b,
                    "execution_timestamp": existing.get(b, timestamp_now),
                }
            )
        tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="ascii", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(new_rows)
        tmp_path.replace(registry_path)

    return len(to_add), len(to_remove)


def relocate_legacy_root_dat_files(source_dir: Path) -> int:
    if source_dir.name != "FILES":
        return 0
    legacy_root = source_dir.parent
    if legacy_root == source_dir:
        return 0
    moved = 0
    for legacy_dat in sorted(legacy_root.glob("mi0*.dat")):
        if not legacy_dat.is_file():
            continue
        destination = source_dir / legacy_dat.name
        if destination.exists():
            continue
        shutil.move(legacy_dat, destination)
        moved += 1
    return moved



def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 simulation: move .dat files into station buffers.")
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SIM_SOURCE_DIR,
        help="Directory containing simulated .dat files.",
    )
    args = parser.parse_args()

    current_path = Path(__file__).resolve()
    master_dir = next((p for p in current_path.parents if p.name == "MASTER"), None)
    if master_dir is None:
        raise RuntimeError(f"Unable to resolve MASTER directory from {current_path}")
    repo_root = master_dir.parent
    source_dir = Path(args.source_dir).expanduser().resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")
    relocated = relocate_legacy_root_dat_files(source_dir)

    station_root = repo_root / "STATIONS" / "MINGO00"
    stage0_sim_dir = station_root / "STAGE_0" / "SIMULATION"
    stage01_dir = station_root / "STAGE_0_to_1"
    stage0_sim_dir.mkdir(parents=True, exist_ok=True)
    stage01_dir.mkdir(parents=True, exist_ok=True)

    registry_path = stage0_sim_dir / "imported_basenames.csv"
    sim_params_path = repo_root / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA" / "step_final_simulation_params.csv"
    normalized_registry_rows = normalize_registry_schema(registry_path)
    imported = load_imported_basenames(registry_path)

    moved = 0
    new_registry_basenames = []
    for dat_file in sorted(source_dir.glob("*.dat")):
        name = dat_file.name
        if not name.startswith("mi00"):
            continue
        basename = dat_file.stem
        if basename in imported:
            continue
        dest_path = stage01_dir / dat_file.name
        shutil.move(dat_file, dest_path)
        imported.add(basename)
        new_registry_basenames.append(basename)
        moved += 1

    # Keep station registry aligned with the simulation params catalogue.
    sim_param_basenames = load_simulation_param_basenames(sim_params_path)
    missing_from_registry = sorted(sim_param_basenames - imported)
    imported.update(missing_from_registry)
    new_registry_basenames.extend(missing_from_registry)

    append_registry_basenames(registry_path, new_registry_basenames)

    # final sync against the downstream filesystem; this will remove any
    # stale basenames and add any that slipped through the other mechanisms.
    added_count, removed_count = sync_registry_with_ground_truth(registry_path, station_root)

    print(
        f"Moved {moved} .dat files from {source_dir} into {stage01_dir}; "
        f"relocated {relocated} legacy root .dat files into {source_dir}; "
        f"normalized {normalized_registry_rows} existing registry rows with execution_timestamp; "
        f"backfilled {len(missing_from_registry)} basenames into {registry_path}; "
        f"sync added {added_count}, removed {removed_count}, final count={len(find_ground_truth_basenames(station_root))}"
    )


if __name__ == "__main__":
    main()
