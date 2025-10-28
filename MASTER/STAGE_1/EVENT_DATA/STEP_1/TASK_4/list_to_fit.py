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
    sync_sources_to_queue,
    write_metadata,
    load_dataframe_h5,
)
from common.pipeline_core import StageMetadata, stage4_list_to_fit  # noqa: E402
from MASTER.common.execution_logger import set_station, start_timer  # noqa: E402
from MASTER.common.status_csv import append_status_row, mark_status_complete  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STEP 1 · TASK 4 — produce fitted outputs ready for STEP_2."
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
        help="Optional override to process a specific listed artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    station = args.station
    set_station(station)
    start_timer(__file__)

    config = load_config()
    station_paths = resolve_station_paths(station)
    task_root = station_paths.task_dirs["TASK_4"]
    work_dirs = ensure_task_layout(task_root)
    status_csv = (task_root / "LOGS" / "list_to_fit_status.csv")
    status_timestamp = append_status_row(status_csv)

    final_output_dir = station_paths.base / "STEP_1" / "TASK_4" / "OUTPUT"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    upstream_dirs = ensure_task_layout(station_paths.task_dirs["TASK_3"])
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
                print("[list_to_fit] No listed artifacts to process.")
                mark_status_complete(status_csv, status_timestamp)
                return
            candidate = candidate_path

    assert isinstance(candidate, Path)
    processing_path = candidate
    metadata_path = processing_path.with_suffix(".json")
    metadata = read_metadata(metadata_path) if metadata_path.exists() else {}

    print(f"[list_to_fit] Processing {processing_path.name}")
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

        fitted_df = stage4_list_to_fit(df, config, stage_metadata)

        output_base = processing_path.stem.replace("listed_", "fitted_")
        output_csv = final_output_dir / f"{output_base}.csv"
        fitted_df.to_csv(output_csv, index=False)

        metadata.update(
            {
                "stage": "list_to_fit",
                "fitted_rows": int(len(fitted_df)),
                "wall_time_sec": round(time.time() - start_wall, 2),
                "output_csv": str(output_csv),
            }
        )
        write_metadata(metadata, output_csv.with_suffix(".json"))

        complete_file(processing_path, work_dirs["completed"])
        if metadata_path.exists():
            metadata_path.unlink()
        mark_status_complete(status_csv, status_timestamp)
        print(f"[list_to_fit] Produced {output_csv.name}")
    except Exception as exc:
        fail_file(processing_path, work_dirs["error"])
        print(f"[list_to_fit] Failed to process {processing_path.name}: {exc}")
        raise
    finally:
        if borrowed_source and processing_path.exists():
            processing_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
