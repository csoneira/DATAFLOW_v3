"""Utility helpers for tracking script execution status via CSV files.

Each status CSV stores rows in the form ``timestamp,status`` where ``status``
is ``"0"`` for a pending execution and ``"1"`` for a completed run.  This
module exposes helpers for both direct import and a tiny CLI so shell scripts
can reuse the same logic.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.execution_logger import start_timer

start_timer(__file__)


def append_status_row(status_csv_path: Path | str) -> str:
    """Append a new row marking the start of an execution.

    Returns the timestamp string written to the CSV so the caller can later
    mark this particular run as complete.
    """

    path = Path(status_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["timestamp", "status"])
        writer.writerow([timestamp, "0"])

    return timestamp


def mark_status_complete(status_csv_path: Path | str, timestamp: str) -> bool:
    """Mark the row created for *timestamp* as completed.

    Returns ``True`` if a matching pending row was updated, ``False``
    otherwise (for example if the CSV was deleted in the meantime).
    """

    path = Path(status_csv_path)
    if not path.exists():
        return False

    rows = []
    updated = False

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if (
                row
                and row[0] == timestamp
                and len(row) > 1
                and row[1] == "0"
                and not updated
            ):
                row[1] = "1"
                updated = True
            rows.append(row)

    if not updated:
        return False

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    return True


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage status CSV files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    append_parser = subparsers.add_parser("append", help="append a pending row")
    append_parser.add_argument("status_csv", type=Path)

    complete_parser = subparsers.add_parser(
        "complete", help="mark a previously appended row as complete"
    )
    complete_parser.add_argument("status_csv", type=Path)
    complete_parser.add_argument("timestamp")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.command == "append":
        timestamp = append_status_row(args.status_csv)
        print(timestamp)
        return 0

    if args.command == "complete":
        if not mark_status_complete(args.status_csv, args.timestamp):
            print(
                "Warning: could not update status row; entry not found or already marked.",
                file=sys.stderr,
            )
            return 1
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
