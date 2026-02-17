#!/usr/bin/env python3
"""
Scan Stage 1 metadata CSVs and remove rows containing NaN values, invalid basenames,
or oversized fields that would trip the default csv.field_size_limit.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

CSV_FIELD_LIMIT = 131072

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def detect_repo_root(start: Path) -> Path:
    """Ascend upwards until a directory containing STATIONS/ is found."""
    for parent in [start] + list(start.parents):
        if (parent / "STATIONS").is_dir():
            return parent
    return start


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = detect_repo_root(script_dir)
    parser = argparse.ArgumentParser(
        description=(
            "Clean STAGE_1 EVENT_DATA metadata CSV files by removing rows that "
            "contain 'nan' values or have basenames that do not start with 'mi'."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Repository root containing STATIONS/ (default: {default_root})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many rows would be removed without modifying files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file details even when no rows are removed.",
    )
    return parser.parse_args()


def has_nan_value(row: Iterable[str]) -> bool:
    for value in row:
        if value is None:
            continue
        if str(value).strip().lower() == "nan":
            return True
    return False


def has_oversized_field(row: Iterable[str]) -> bool:
    for value in row:
        if value is None:
            continue
        if len(str(value)) >= CSV_FIELD_LIMIT:
            return True
    return False


def basename_invalid(row: List[str], basename_idx: int | None) -> bool:
    if basename_idx is None or basename_idx >= len(row):
        return False
    value = row[basename_idx].strip()
    if not value:
        return True
    return not value.lower().startswith("mi")


def load_rows(path: Path) -> List[List[str]]:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        return [row for row in reader]


def write_rows(path: Path, rows: List[List[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def process_file(path: Path, dry_run: bool) -> Tuple[int, int, int, int, int]:
    rows = load_rows(path)
    if not rows:
        return 0, 0, 0, 0, 0

    header = rows[0]
    try:
        basename_idx = header.index("basename")
    except ValueError:
        basename_idx = None

    kept_rows = [header]
    removed = 0
    nan_hits = 0
    basename_hits = 0
    oversized_hits = 0

    for row in rows[1:]:
        remove = False
        if has_nan_value(row):
            remove = True
            nan_hits += 1
        if basename_invalid(row, basename_idx):
            remove = True
            basename_hits += 1
        if has_oversized_field(row):
            remove = True
            oversized_hits += 1
        if remove:
            removed += 1
        else:
            kept_rows.append(row)

    if removed > 0 and not dry_run:
        write_rows(path, kept_rows)

    return len(rows) - 1, removed, nan_hits, basename_hits, oversized_hits


def collect_metadata_files(root: Path) -> List[Path]:
    patterns = [
        "STATIONS/MINGO0*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/METADATA/task_*_metadata_execution.csv",
        "STATIONS/MINGO0*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/METADATA/task_*_metadata_specific.csv",
    ]
    results = set()
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if candidate.is_file():
                results.add(candidate)
    return sorted(results)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        print(f"Root directory {root} does not exist or is not a directory.", file=sys.stderr)
        return 1

    files = collect_metadata_files(root)
    if not files:
        print(f"No metadata CSV files matched under {root}.")
        return 0

    total_rows = 0
    total_removed = 0
    total_nan = 0
    total_basename = 0
    total_oversized = 0

    for csv_path in files:
        rows, removed, nan_hits, basename_hits, oversized_hits = process_file(csv_path, args.dry_run)
        total_rows += rows
        total_removed += removed
        total_nan += nan_hits
        total_basename += basename_hits
        total_oversized += oversized_hits
        if removed > 0 or args.verbose:
            action = "would remove" if args.dry_run else "removed"
            print(
                f"{csv_path}: {action} {removed} row(s) "
                f"(nan={nan_hits}, invalid_basename={basename_hits}, oversized_field={oversized_hits})"
            )

    summary_action = "Would remove" if args.dry_run else "Removed"
    print(
        f"{summary_action} {total_removed} of {total_rows} row(s) "
        f"across {len(files)} file(s). "
        f"(nan={total_nan}, invalid_basenames={total_basename}, oversized_field={total_oversized})"
    )
    if args.dry_run:
        print("Dry-run complete. Re-run without --dry-run to apply fixes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
