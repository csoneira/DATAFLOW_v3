#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 simulation setup for simulated stations.")
    parser.add_argument("--station", type=int, required=True, help="Destination station number (e.g. 5).")
    parser.add_argument("--source-station", type=int, default=1, help="Source station number for config.")
    parser.add_argument("--only-station", type=int, default=None, help="Only copy files for this station id.")
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

    config_src = (
        repo_root
        / "MASTER"
        / "CONFIG_FILES"
        / "ONLINE_RUN_DICTIONARY"
        / f"STATION_{args.source_station}"
        / f"input_file_mingo{args.source_station:02d}.csv"
    )
    config_dst_dir = (
        repo_root
        / "MASTER"
        / "CONFIG_FILES"
        / "ONLINE_RUN_DICTIONARY"
        / f"STATION_{args.station}"
    )
    config_dst = config_dst_dir / f"input_file_mingo{args.station:02d}.csv"
    config_dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_src, config_dst)

    for dat_file in sorted(source_dir.glob("*.dat")):
        name = dat_file.name
        if not name.startswith("mi0"):
            continue
        station_code = name[2:4]
        if not station_code.isdigit():
            continue
        station_id = int(station_code)
        if station_id <= 0:
            continue
        if args.only_station is not None and station_id != args.only_station:
            continue
        station_dir = repo_root / "STATIONS" / f"MINGO{station_id:02d}" / "STAGE_0_to_1"
        station_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dat_file, station_dir / dat_file.name)

    print(f"Copied config to {config_dst}")
    print(f"Copied .dat files from {source_dir} into station STAGE_0_to_1 directories")


if __name__ == "__main__":
    main()
