#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


COMMENT_PREFIX = "#"


def extract_basenames_from_metadata(metadata: Dict[str, str], fallback: str) -> Set[str]:
    raw_value = metadata.get("source_basenames")
    if not raw_value:
        return {fallback}
    parsed = {name.strip() for name in raw_value.split(",") if name.strip()}
    return parsed or {fallback}


def collect_day_basenames(csv_files: Sequence[Path]) -> Set[str]:
    basenames: Set[str] = set()
    for csv_file in csv_files:
        metadata = parse_metadata(csv_file)
        basenames.update(extract_basenames_from_metadata(metadata, csv_file.stem))
    return basenames


def existing_output_basenames(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    metadata = parse_metadata(output_path)
    return extract_basenames_from_metadata(metadata, output_path.stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge STEP_3/TASK_1_TO_2 daily accumulated CSV files into consolidated "
            "event_data_YYYY_MM_DD.csv outputs for TASK_2."
        )
    )
    parser.add_argument("station", help="Station identifier (e.g. 1, 2, 3, 4)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without writing joined CSV files.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Rebuild outputs for all days even if the existing file is up to date.",
    )
    return parser.parse_args()


def parse_metadata(path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith(COMMENT_PREFIX):
                content = line[len(COMMENT_PREFIX) :].strip()
                if "=" in content:
                    key, value = content.split("=", 1)
                    metadata[key.strip()] = value.strip()
            else:
                # rewind to the start of the first non-comment line
                handle.seek(position)
                break
    return metadata


def read_with_metadata(path: Path) -> pd.DataFrame:
    metadata = parse_metadata(path)
    dataframe = pd.read_csv(path, comment=COMMENT_PREFIX)
    if dataframe.empty:
        return dataframe

    if "Time" not in dataframe.columns:
        raise ValueError(f"Column 'Time' not present in {path}.")

    dataframe["Time"] = pd.to_datetime(dataframe["Time"], errors="coerce")
    dataframe = dataframe.dropna(subset=["Time"]).reset_index(drop=True)

    raw_basenames = metadata.get("source_basenames")
    if raw_basenames:
        basenames = tuple(
            name.strip() for name in raw_basenames.split(",") if name.strip()
        )
    else:
        basenames = (path.stem,)

    exec_str = metadata.get("execution_date")
    execution_dt = pd.NaT
    if exec_str:
        exec_values = [value.strip() for value in exec_str.split(",") if value.strip()]
        parsed_exec_dates: List[pd.Timestamp] = []
        for value in exec_values:
            parsed = pd.to_datetime(value, errors="coerce")
            if isinstance(parsed, pd.Timestamp) and pd.notna(parsed):
                if parsed.tzinfo is not None:
                    parsed = parsed.tz_convert("UTC").tz_localize(None)
                parsed_exec_dates.append(parsed)
        if parsed_exec_dates:
            execution_dt = max(parsed_exec_dates)

    dataframe["source_basenames"] = [basenames] * len(dataframe)
    dataframe["execution_date"] = [execution_dt] * len(dataframe)

    value_columns = [
        column
        for column in dataframe.columns
        if column not in {"Time", "source_basenames", "execution_date"}
    ]
    dataframe[value_columns] = dataframe[value_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    return dataframe


def iter_day_directories(root: Path) -> List[Tuple[date, Path]]:
    day_directories: List[Tuple[date, Path]] = []
    for potential in sorted(root.rglob("*")):
        if not potential.is_dir():
            continue

        csv_files = list(potential.glob("*.csv"))
        if not csv_files:
            continue

        try:
            relative = potential.relative_to(root)
        except ValueError:
            continue

        parts = relative.parts
        if len(parts) < 3:
            continue

        candidate = parts[-3:]
        try:
            day = datetime.strptime("-".join(candidate), "%Y-%m-%d").date()
        except ValueError:
            continue

        day_directories.append((day, potential))

    day_directories.sort(key=lambda item: item[0])
    return day_directories


def resolve_time_group(group: pd.DataFrame, value_columns: Sequence[str]) -> List[pd.Series]:
    if len(group) == 1:
        return [group.iloc[0].copy()]

    basename_sets: List[Set[str]] = [set(entry) for entry in group["source_basenames"]]
    adjacency = {index: set() for index in range(len(group))}

    for idx in range(len(group)):
        for jdx in range(idx + 1, len(group)):
            if basename_sets[idx].intersection(basename_sets[jdx]):
                adjacency[idx].add(jdx)
                adjacency[jdx].add(idx)

    components: List[List[int]] = []
    visited: Set[int] = set()

    for idx in range(len(group)):
        if idx in visited:
            continue
        stack = [idx]
        component: List[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        components.append(component)

    if all(len(component) == 1 for component in components) and len(components) > 1:
        numeric = (
            group.loc[:, value_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum()
        )
        aggregate = group.iloc[0].copy()
        for column, value in numeric.items():
            aggregate[column] = value
        aggregate["source_basenames"] = tuple(
            sorted({name for names in basename_sets for name in names})
        )
        exec_dates = group["execution_date"]
        if exec_dates.notna().any():
            aggregate["execution_date"] = exec_dates.max()
        return [aggregate]

    resolved_rows: List[pd.Series] = []
    for component in components:
        if len(component) == 1:
            resolved_rows.append(group.iloc[component[0]].copy())
            continue

        component_frame = group.iloc[component].copy()
        component_frame = component_frame.sort_values(
            "execution_date", ascending=False, na_position="last"
        )
        resolved_rows.append(component_frame.iloc[0].copy())

    return resolved_rows


def merge_day_files(csv_files: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_file in sorted(csv_files):
        dataframe = read_with_metadata(csv_file)
        if dataframe.empty:
            continue
        frames.append(dataframe)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Time"]).sort_values("Time").reset_index(
        drop=True
    )

    value_columns = [
        column
        for column in combined.columns
        if column not in {"Time", "source_basenames", "execution_date"}
    ]
    combined[value_columns] = combined[value_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    merged_rows: List[pd.Series] = []
    for _, group in combined.groupby("Time", sort=True):
        merged_rows.extend(resolve_time_group(group, value_columns))

    if not merged_rows:
        return pd.DataFrame(columns=["Time", *value_columns])

    merged = pd.DataFrame(merged_rows)

    for column in value_columns:
        if column not in merged.columns:
            merged[column] = 0

    merged = merged[
        ["Time", *value_columns, "source_basenames", "execution_date"]
    ].sort_values("Time")

    merged.reset_index(drop=True, inplace=True)
    return merged


def format_header_values(values: Iterable[str]) -> str:
    return ",".join(sorted(dict.fromkeys(values)))


def main() -> int:
    args = parse_args()
    station = args.station

    station_dir = Path.home() / "DATAFLOW_v3" / "STATIONS" / f"MINGO0{station}"
    stage1_event_data = station_dir / "STAGE_1" / "EVENT_DATA" / "STEP_3"
    task1_to_2_root = stage1_event_data / "TASK_1_TO_2"
    task2_output_root = stage1_event_data / "TASK_2" / "OUTPUT_FILES"

    if not task1_to_2_root.exists():
        print(f"No input directory found: {task1_to_2_root}")
        return 0

    day_directories = iter_day_directories(task1_to_2_root)
    if not day_directories:
        print("No daily directories with CSV files found to merge.")
        return 0

    if not args.dry_run:
        task2_output_root.mkdir -p(parents=True, exist_ok=True)

    for day, day_dir in day_directories:
        csv_files = sorted(day_dir.glob("*.csv"))
        if not csv_files:
            continue

        year = f"{day:%Y}"
        month = f"{day:%m}"

        output_filename = f"event_data_{day:%Y_%m_%d}.csv"
        output_parent = task2_output_root / year / month
        output_path = output_parent / output_filename
        day_basenames = collect_day_basenames(csv_files)
        existing_basenames = existing_output_basenames(output_path)

        needs_merge = args.all or not output_path.exists() or day_basenames != existing_basenames

        if not needs_merge:
            print(f"Skipping {day:%Y-%m-%d}: output already up to date ({output_path}).")
            continue

        print(
            f"Processing {day_dir} ({day:%Y-%m-%d}) with {len(csv_files)} file(s)..."
        )

        merged = merge_day_files(csv_files)

        if merged.empty:
            print("  ! Skipped: nothing to merge.")
            continue

        basenames: List[str] = []
        exec_dates: List[str] = []

        for entry in merged["source_basenames"]:
            basenames.extend(entry)

        for exec_dt in merged["execution_date"]:
            if pd.isna(exec_dt):
                continue
            if isinstance(exec_dt, pd.Timestamp):
                exec_dates.append(exec_dt.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                exec_dates.append(str(exec_dt))

        output_dataframe = merged.drop(columns=["source_basenames", "execution_date"])
        output_dataframe = output_dataframe.copy()
        output_dataframe["Time"] = output_dataframe["Time"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        basename_header = format_header_values(basenames)
        exec_date_header = format_header_values(exec_dates)

        if args.dry_run:
            print(f"  [dry-run] would write {output_path}")
            print(f"           # source_basenames={basename_header}")
            print(f"           # execution_date={exec_date_header}")
            continue

        output_parent.mkdir -p(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(f"# source_basenames={basename_header}\n")
            handle.write(f"# execution_date={exec_date_header}\n")
            output_dataframe.to_csv(handle, index=False)

        print(f"  Wrote {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
