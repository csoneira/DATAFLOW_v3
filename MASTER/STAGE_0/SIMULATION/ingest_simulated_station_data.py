#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path



def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 simulation: move .dat files into station buffers.")
    parser.add_argument(
        "--source-dir",
        default="~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA",
        help="Directory containing simulated .dat files.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    station_root = repo_root / "STATIONS" / "MINGO00"
    stage0_sim_dir = station_root / "STAGE_0" / "SIMULATION"
    stage01_dir = station_root / "STAGE_0_to_1"
    stage0_sim_dir.mkdir(parents=True, exist_ok=True)
    stage01_dir.mkdir(parents=True, exist_ok=True)

    registry_path = stage0_sim_dir / "imported_basenames.csv"
    imported = set()
    if registry_path.exists():
        with registry_path.open("r", encoding="ascii") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("basename"):
                    imported.add(row["basename"])

    moved = 0
    new_registry_rows = []
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
        new_registry_rows.append({"basename": basename})
        moved += 1


    if new_registry_rows:
        write_header = not registry_path.exists()
        with registry_path.open("a", encoding="ascii", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["basename"])
            if write_header:
                writer.writeheader()
            writer.writerows(new_registry_rows)

    print(f"Moved {moved} .dat files from {source_dir} into {stage01_dir}")


if __name__ == "__main__":
    main()
