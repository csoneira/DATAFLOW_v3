#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        station_dir = repo_root / "STATIONS" / f"MINGO{station_id:02d}" / "STAGE_0_to_1"
        station_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(dat_file, station_dir / dat_file.name)

    print(f"Moved .dat files from {source_dir} into station STAGE_0_to_1 directories")


if __name__ == "__main__":
    main()
