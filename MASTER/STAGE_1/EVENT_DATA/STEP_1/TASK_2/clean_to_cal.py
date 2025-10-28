#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STEP_ROOT = SCRIPT_DIR.parent
if str(STEP_ROOT) not in sys.path:
    sys.path.append(str(STEP_ROOT))

REPO_ROOT = next((p for p in STEP_ROOT.parents if (p / "MASTER").is_dir()), STEP_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from common.pipeline_common import (  # noqa: E402
    acquire_file,
    complete_file,
    ensure_task_layout,
    fail_file,
    load_config,
    read_metadata,
    resolve_station_paths,
    save_dataframe_h5,
    sync_sources_to_queue,
    write_metadata,
    load_dataframe_h5,
)
from common.pipeline_core import StageMetadata, stage2_clean_to_cal  # noqa: E402
from MASTER.common.execution_logger import set_station, start_timer  # noqa: E402
from MASTER.common.status_csv import append_status_row, mark_status_complete  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STEP 1 · TASK 2 — calibrate cleaned data frames."
    )
    parser.add_argument("station", help="Station number (1-4)")
    parser.add_argument(
        "--strategy",
        choices=("oldest", "random"),
        default="oldest",
        help="File selection strategy when multiple inputs are pending.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional override to process a specific cleaned artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    station = args.station
    set_station(station)
    start_timer(__file__)

    config = load_config()
    station_paths = resolve_station_paths(station)
    task_root = station_paths.task_dirs["TASK_2"]
    work_dirs = ensure_task_layout(task_root)
    status_csv = (task_root / "LOGS" / "clean_to_cal_status.csv")
    status_timestamp = append_status_row(status_csv)

    # Ensure downstream queue exists for TASK_3
    stage3_dirs = ensure_task_layout(station_paths.task_dirs["TASK_3"])

    upstream_dirs = ensure_task_layout(station_paths.task_dirs["TASK_1"])
    sync_sources_to_queue(upstream_dirs["completed"], work_dirs["unprocessed"])

    if args.source:
        processing_target = work_dirs["processing"] / args.source.name
        args.source.rename(processing_target)
        candidate = processing_target
        borrowed_source = True
    else:
        borrowed_source = False
        with acquire_file(
            work_dirs["unprocessed"],
            work_dirs["processing"],
            strategy=args.strategy,
        ) as candidate_path:
            if candidate_path is None:
                print("[clean_to_cal] No cleaned artifacts to process.")
                mark_status_complete(status_csv, status_timestamp)
                return
            candidate = candidate_path

    assert isinstance(candidate, Path)
    processing_path = candidate
    metadata_path = processing_path.with_suffix(".json")
    if not metadata_path.exists():
        metadata = {}
    else:
        metadata = read_metadata(metadata_path)

    print(f"[clean_to_cal] Processing {processing_path.name}")
    start_wall = time.time()

    try:
        df = load_dataframe_h5(processing_path)

        default_now = datetime.now(timezone.utc)
        stage_metadata = StageMetadata(
            station=station,
            start_time=pd.Timestamp(metadata.get("start_time", default_now)),
            end_time=pd.Timestamp(metadata.get("end_time", default_now)),
            execution_time=metadata.get("execution_time", ""),
            date_execution=metadata.get(
                "date_execution",
                default_now.strftime("%y-%m-%d_%H.%M.%S"),
            ),
        )

        calibrated_df = stage2_clean_to_cal(df, config, stage_metadata)

        output_name = processing_path.stem.replace("cleaned_", "calibrated_")
        output_path = stage3_dirs["unprocessed"] / f"{output_name}.h5"
        save_dataframe_h5(calibrated_df, output_path)

        metadata.update(
            {
                "stage": "clean_to_cal",
                "calibrated_rows": int(len(calibrated_df)),
                "wall_time_sec": round(time.time() - start_wall, 2),
            }
        )
        write_metadata(metadata, output_path.with_suffix(".json"))

        complete_file(processing_path, work_dirs["completed"])
        if metadata_path.exists():
            metadata_path.unlink()
        mark_status_complete(status_csv, status_timestamp)
        print(f"[clean_to_cal] Produced {output_path.name}")
    except Exception as exc:
        fail_file(processing_path, work_dirs["error"])
        print(f"[clean_to_cal] Failed to process {processing_path.name}: {exc}")
        raise
    finally:
        if borrowed_source and processing_path.exists():
            processing_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
