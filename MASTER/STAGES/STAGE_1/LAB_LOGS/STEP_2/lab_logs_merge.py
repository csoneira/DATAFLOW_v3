#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/STAGES/STAGE_1/LAB_LOGS/STEP_2/lab_logs_merge.py
Purpose: !/usr/bin/env python3.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MASTER/STAGES/STAGE_1/LAB_LOGS/STEP_2/lab_logs_merge.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

CURRENT_PATH = Path(__file__).resolve()
REPO_ROOT = None
for parent in CURRENT_PATH.parents:
    if parent.name == "MASTER":
        REPO_ROOT = parent.parent
        break
if REPO_ROOT is None:
    REPO_ROOT = CURRENT_PATH.parents[-1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.execution_logger import set_station, start_timer
from MASTER.common.path_config import (
    get_master_config_root,
    resolve_home_path_from_config,
)
from MASTER.common.selection_config import (
    effective_date_ranges_for_station,
    load_selection_for_paths,
    station_is_selected,
)
from MASTER.common.status_csv import append_status_row, mark_status_complete

CREATE_NEW_CSV = True


@dataclass(frozen=True)
class LogSpec:
    name: str
    prefixes: Sequence[str]
    columns: Sequence[str]


LOG_SPECS: Sequence[LogSpec] = (
    LogSpec(
        name="hv",
        prefixes=("hv0_",),
        columns=(
            "Date",
            "Hour",
            "Unused1",
            "Unused2",
            "Unused3",
            "Unused4",
            "Unused5",
            "Unused6",
            "CurrentNeg",
            "CurrentPos",
            "HVneg",
            "HVpos",
            "Unused7",
            "Unused8",
            "Unused9",
            "Unused10",
            "Unused11",
            "Unused12",
            "Unused13",
            "Unused14",
            "Unused15",
        ),
    ),
    LogSpec(
        name="rates",
        prefixes=("rates_",),
        columns=(
            "Date",
            "Hour",
            "Asserted",
            "Edge",
            "Accepted",
            "Multiplexer1",
            "M2",
            "M3",
            "M4",
            "CM1",
            "CM2",
            "CM3",
            "CM4",
        ),
    ),
    LogSpec(
        name="sensors_ext",
        prefixes=("sensors_bus0_",),
        columns=(
            "Date",
            "Hour",
            "Unused1",
            "Unused2",
            "Unused3",
            "Unused4",
            "Temperature_ext",
            "RH_ext",
            "Pressure_ext",
        ),
    ),
    LogSpec(
        name="sensors_int",
        prefixes=("sensors_bus1_",),
        columns=(
            "Date",
            "Hour",
            "Unused1",
            "Unused2",
            "Unused3",
            "Unused4",
            "Temperature_int",
            "RH_int",
            "Pressure_int",
        ),
    ),
    LogSpec(
        name="odroid",
        prefixes=("Odroid_",),
        columns=(
            "Date",
            "Hour",
            "DiskFill1",
            "DiskFill2",
            "DiskFillX",
        ),
    ),
    LogSpec(
        name="flow",
        prefixes=("Flow",),
        columns=("Date", "Hour", "FlowRate1", "FlowRate2", "FlowRate3", "FlowRate4"),
    ),
)

MASTER_CONFIG_ROOT = get_master_config_root()
LAB_LOGS_CONFIG_SHARED = MASTER_CONFIG_ROOT / "STAGE_1" / "LAB_LOGS" / "config_lab_logs.yaml"
LAB_LOGS_CONFIG_STEP2 = MASTER_CONFIG_ROOT / "STAGE_1" / "LAB_LOGS" / "STEP_2" / "config_step_2.yaml"
LAB_LOGS_OUTLIER_CSV_STEP2 = MASTER_CONFIG_ROOT / "STAGE_1" / "LAB_LOGS" / "STEP_2" / "config_step_2.csv"
FILENAME_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _load_yaml_mapping(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _parse_date_value(value: object) -> Optional[date]:
    if value in ("", None):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z") and "T" in text:
        text = text[:-1] + "+00:00"
    parsed: Optional[datetime] = None
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            if fmt is None:
                parsed = datetime.fromisoformat(text)
            else:
                parsed = datetime.strptime(text, fmt)
            break
        except Exception:
            parsed = None
    if parsed is None:
        return None
    return parsed.date()


def _collect_date_ranges(node: Mapping[str, object], ranges: List[Tuple[Optional[date], Optional[date]]]) -> None:
    if not isinstance(node, dict):
        return
    legacy = node.get("date_range")
    if isinstance(legacy, dict):
        start_val = _parse_date_value(legacy.get("start"))
        end_val = _parse_date_value(legacy.get("end"))
        if start_val is not None or end_val is not None:
            ranges.append((start_val, end_val))
    range_list = node.get("date_ranges")
    if isinstance(range_list, list):
        for item in range_list:
            if not isinstance(item, dict):
                continue
            start_val = _parse_date_value(item.get("start"))
            end_val = _parse_date_value(item.get("end"))
            if start_val is None and end_val is None:
                continue
            ranges.append((start_val, end_val))
    nested = node.get("lab_logs_date_selection")
    if isinstance(nested, dict):
        _collect_date_ranges(nested, ranges)


def load_lab_logs_date_ranges(
    station: Optional[int | str] = None,
) -> List[Tuple[Optional[date], Optional[date]]]:
    selection = load_selection_for_paths(
        [LAB_LOGS_CONFIG_SHARED, LAB_LOGS_CONFIG_STEP2],
        master_config_root=MASTER_CONFIG_ROOT,
    )
    if station is None:
        return list(selection.date_ranges)
    return list(effective_date_ranges_for_station(station, selection.date_ranges))


def extract_date_from_filename(path: Path) -> Optional[date]:
    match = FILENAME_DATE_RE.search(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def date_in_ranges(day_value: date, ranges: Sequence[Tuple[Optional[date], Optional[date]]]) -> bool:
    for start_day, end_day in ranges:
        if start_day is not None and day_value < start_day:
            continue
        if end_day is not None and day_value > end_day:
            continue
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate cleaned lab log files into daily outputs."
    )
    parser.add_argument("station", help="Station identifier (e.g. 1, 2, 3, 4)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without moving or writing files.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Rebuild outputs using files already archived in COMPLETED.",
    )
    return parser.parse_args()


def load_outlier_limits() -> Dict[str, Sequence[float]]:
    config_file_path = LAB_LOGS_OUTLIER_CSV_STEP2
    try:
        config_df = pd.read_csv(config_file_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Outlier limits CSV not found: {config_file_path}"
        ) from exc

    if config_df.empty:
        return {}

    normalized_columns = {str(col).strip().lower(): col for col in config_df.columns}
    required = ("column", "lower", "upper")
    if not all(name in normalized_columns for name in required):
        raise ValueError(
            f"{config_file_path} must include the columns: column, lower, upper"
        )

    column_col = normalized_columns["column"]
    lower_col = normalized_columns["lower"]
    upper_col = normalized_columns["upper"]

    limits: Dict[str, Sequence[float]] = {}
    for index, row in config_df.iterrows():
        column_name = str(row[column_col]).strip()
        if not column_name or column_name.lower() == "nan":
            continue
        try:
            lower = float(row[lower_col])
            upper = float(row[upper_col])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric bounds in {config_file_path} at row {index + 2}"
            ) from exc
        limits[column_name] = (lower, upper)

    return limits


def move_step1_outputs_to_unprocessed(
    step1_output_dir: Path,
    unprocessed_root: Path,
    dry_run: bool,
) -> Dict[Path, Path]:
    planned_moves: Dict[Path, Path] = {}
    if not step1_output_dir.exists():
        return planned_moves

    files = sorted(path for path in step1_output_dir.glob("**/*") if path.is_file())
    if not dry_run:
        unprocessed_root.mkdir(parents=True, exist_ok=True)

    for source in files:
        relative = source.relative_to(step1_output_dir)
        destination = unprocessed_root / relative

        if dry_run:
            print(f"  [dry-run] move {source} -> {destination}")
            planned_moves[destination] = source
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_file():
                destination.unlink()
            else:
                shutil.rmtree(destination)
        shutil.move(str(source), str(destination))
        planned_moves[destination] = destination

    if not dry_run:
        _cleanup_empty_directories(step1_output_dir)

    return planned_moves


def _cleanup_empty_directories(root: Path) -> None:
    for directory in sorted(
        (path for path in root.glob("**/*") if path.is_dir()), reverse=True
    ):
        try:
            directory.rmdir()
        except OSError:
            break


def match_spec(file_name: str) -> LogSpec | None:
    for spec in LOG_SPECS:
        if any(file_name.startswith(prefix) for prefix in spec.prefixes):
            return spec
    return None


def read_clean_log(path: Path, spec: LogSpec) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python",
            on_bad_lines="skip",
        )
    except FileNotFoundError:
        return pd.DataFrame()

    if df.empty:
        return df

    if len(df.columns) > len(spec.columns):
        df = df.iloc[:, : len(spec.columns)]
    elif len(df.columns) < len(spec.columns):
        for _ in range(len(spec.columns) - len(df.columns)):
            df[len(df.columns)] = pd.NA

    df.columns = spec.columns
    drop_columns = [col for col in df.columns if col.lower().startswith("unused")]
    df = df.drop(columns=drop_columns, errors="ignore")

    if "Date" in df.columns and "Hour" in df.columns:
        df["Time"] = pd.to_datetime(
            df["Date"].astype(str) + "T" + df["Hour"].astype(str), errors="coerce"
        )
        df = df.drop(columns=["Date", "Hour"])

    if "Time" not in df.columns:
        return pd.DataFrame()

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])

    value_columns = [col for col in df.columns if col != "Time"]
    if not value_columns:
        return pd.DataFrame()

    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.set_index("Time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns=lambda col: f"{spec.name}_{col}")

    return df


def apply_outlier_filters(df: pd.DataFrame, limits: Mapping[str, Sequence[float]]) -> None:
    for column, (lower, upper) in limits.items():
        if column in df.columns:
            df[column] = df[column].where(
                (df[column] >= lower) & (df[column] <= upper), np.nan
            )


def archive_processed_files(
    processed_files: Iterable[Path],
    unprocessed_root: Path,
    completed_root: Path,
    dry_run: bool,
) -> None:
    for source in processed_files:
        try:
            relative = source.relative_to(unprocessed_root)
        except ValueError:
            continue

        destination = completed_root / relative
        if dry_run:
            print(f"  [dry-run] move {source} -> {destination}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_file():
                destination.unlink()
            else:
                shutil.rmtree(destination)
        shutil.move(str(source), str(destination))

    if not dry_run:
        _cleanup_empty_directories(unprocessed_root)


def rebuild_big_csv(
    output_root: Path,
    big_csv_path: Path,
    dry_run: bool,
) -> None:
    output_files = sorted(
        path for path in output_root.rglob("lab_logs_*.csv") if path.is_file()
    )
    if dry_run:
        if output_files:
            print(f"  [dry-run] would rebuild {big_csv_path}")
        return

    if not output_files:
        if big_csv_path.exists():
            big_csv_path.unlink()
        return

    frames: List[pd.DataFrame] = []
    for file_path in output_files:
        try:
            df = pd.read_csv(file_path, parse_dates=["Time"])
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"Warning: unable to load {file_path}: {exc}")
            continue
        frames.append(df)

    if not frames:
        return

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="Time", keep="last")
        .sort_values("Time")
    )
    combined.to_csv(big_csv_path, index=False, float_format="%.5g")


def resolve_actual_path(path: Path, moves: Mapping[Path, Path]) -> Path:
    return moves.get(path, path)


def main() -> int:
    args = parse_args()

    station = args.station
    selection = load_selection_for_paths(
        [LAB_LOGS_CONFIG_SHARED, LAB_LOGS_CONFIG_STEP2],
        master_config_root=MASTER_CONFIG_ROOT,
    )
    if not station_is_selected(station, selection.stations):
        print(f"Station {station} skipped by selection.stations.")
        return 0

    set_station(station)
    start_timer(__file__)

    base_path = resolve_home_path_from_config()
    station_dir = base_path / "DATAFLOW_v3" / "STATIONS" / f"MINGO0{station}"
    lab_logs_root = station_dir / "STAGE_1" / "LAB_LOGS"

    step1_root = lab_logs_root / "STEP_1"
    step1_output_dir = step1_root / "OUTPUT_FILES"

    step2_root = lab_logs_root / "STEP_2"
    input_root = step2_root / "INPUT_FILES"
    unprocessed_root = input_root / "UNPROCESSED"
    completed_root = input_root / "COMPLETED"
    output_root = step2_root / "OUTPUT_FILES"

    # status_csv_path = step2_root / "log_aggregate_and_join_status.csv"
    # status_timestamp = append_status_row(status_csv_path)

    outlier_limits = load_outlier_limits()
    date_ranges = list(effective_date_ranges_for_station(station, selection.date_ranges))
    if date_ranges:
        human_ranges = []
        for start_day, end_day in date_ranges:
            start_text = start_day.isoformat() if start_day is not None else "-inf"
            end_text = end_day.isoformat() if end_day is not None else "+inf"
            human_ranges.append(f"{start_text} to {end_text}")
        print(
            "Date range filtering enabled for LAB_LOGS STEP_2: "
            + "; ".join(human_ranges)
        )

    if not args.dry_run:
        unprocessed_root.mkdir(parents=True, exist_ok=True)
        completed_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

    planned_moves = move_step1_outputs_to_unprocessed(
        step1_output_dir,
        unprocessed_root,
        args.dry_run,
    )

    candidate_files = sorted(
        path for path in unprocessed_root.glob("**/*") if path.is_file()
    )

    if args.all and completed_root.exists():
        candidate_files.extend(
            sorted(path for path in completed_root.glob("**/*") if path.is_file())
        )

    if date_ranges:
        filtered_candidates: List[Path] = []
        for file_path in candidate_files:
            day_value = extract_date_from_filename(file_path)
            if day_value is None:
                continue
            if date_in_ranges(day_value, date_ranges):
                filtered_candidates.append(file_path)
        candidate_files = filtered_candidates

    if not candidate_files:
        print("No cleaned lab log files found to process.")
        # rebuild_big_csv(output_root, lab_logs_root / "big_log_lab_data.csv", args.dry_run)
        # mark_status_complete(status_csv_path, status_timestamp)
        return 0

    day_data: Dict[date, Dict[str, pd.DataFrame]] = defaultdict(dict)
    processed_sources: List[Path] = []

    for file_path in candidate_files:
        spec = match_spec(file_path.name)
        if spec is None:
            print(f"Skipping {file_path.name}: unknown log prefix.")
            continue

        actual_path = resolve_actual_path(file_path, planned_moves)
        df = read_clean_log(actual_path, spec)
        if df.empty:
            continue

        file_had_data = False

        for day_key, day_frame in df.groupby(df.index.date):
            if day_frame.empty:
                continue
            if date_ranges and not date_in_ranges(day_key, date_ranges):
                continue
            resampled = day_frame.resample("1min").mean()
            if resampled.empty:
                continue

            existing = day_data[day_key].get(spec.name)
            if existing is not None:
                combined = (
                    pd.concat([existing, resampled])
                    .groupby(level=0)
                    .last()
                    .sort_index()
                )
                day_data[day_key][spec.name] = combined
            else:
                day_data[day_key][spec.name] = resampled.sort_index()

            file_had_data = True

        if file_had_data and file_path.is_relative_to(unprocessed_root):
            processed_sources.append(file_path)

    if not day_data:
        print("No log data found to merge.")
        # rebuild_big_csv(output_root, lab_logs_root / "big_log_lab_data.csv", args.dry_run)
        # mark_status_complete(status_csv_path, status_timestamp)
        return 0

    for day_key in sorted(day_data.keys()):
        frames = [frame for frame in day_data[day_key].values()]
        merged = pd.concat(frames, axis=1).sort_index()

        apply_outlier_filters(merged, outlier_limits)

        output_parent = output_root / f"{day_key:%Y}" / f"{day_key:%m}"
        output_path = output_parent / f"lab_logs_{day_key:%Y_%m_%d}.csv"

        if output_path.exists():
            try:
                existing = pd.read_csv(output_path, parse_dates=["Time"]).set_index(
                    "Time"
                )
            except Exception as exc:  # pragma: no cover - best effort logging
                print(f"Warning: unable to load existing {output_path}: {exc}")
                existing = pd.DataFrame()
            merged = (
                pd.concat([existing, merged])
                .groupby(level=0)
                .last()
                .sort_index()
            )

        if args.dry_run:
            print(f"  [dry-run] would write {output_path}")
        elif CREATE_NEW_CSV:
            output_parent.mkdir(parents=True, exist_ok=True)
            merged.reset_index().to_csv(output_path, index=False, float_format="%.5g")
            print(f"  Wrote {output_path}")

    archive_processed_files(
        processed_sources,
        unprocessed_root,
        completed_root,
        args.dry_run,
    )

    # rebuild_big_csv(output_root, lab_logs_root / "big_log_lab_data.csv", args.dry_run)

    # mark_status_complete(status_csv_path, status_timestamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
