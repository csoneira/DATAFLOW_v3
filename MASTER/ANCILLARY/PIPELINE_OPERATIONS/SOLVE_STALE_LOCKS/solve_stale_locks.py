#!/usr/bin/env python3
"""Check for stale STEP_1 and unpacker lock files.

This helper is meant to be executed from cron.  It scans the lock directory
(`~/DATAFLOW_v3/EXECUTION_LOGS/LOCKS`) and tries to determine whether a lock
file is stale.  A lock is considered stale when both of the following are true:

* the file is older than the configured threshold, and
* there is no `guide_raw_to_corrected.sh <station>` process still running for
  the station encoded in the file name.

The routine also keeps an eye on the STEP_0 unpacker shared lock
(`.unpack_shared.lock`). That lock is considered stale only when it
exceeds the age threshold *and* there is no
``unpack_reprocessing_files.sh`` process still holding it.

When a stale lock is detected, the script removes it and records the action in
`~/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS/solve_stale_locks.log` so operators can
audit what happened.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_LOCK_DIR = Path.home() / "DATAFLOW_v3" / "EXECUTION_LOGS" / "LOCKS"
UNPACK_LOCK_NAME = ".unpack_shared.lock"
DEFAULT_LOG_FILE = (
    Path.home() / "DATAFLOW_v3" / "EXECUTION_LOGS" / "CRON_LOGS" / "solve_stale_locks.log"
)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect STEP_1 station locks that look stale and delete them so"
            " the cron-driven pipelines can resume."
        )
    )
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=DEFAULT_LOCK_DIR,
        help=f"Directory that stores lock files (default: {DEFAULT_LOCK_DIR})",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=int,
        default=15,
        help="Minimum age (in minutes) before a lock can be considered stale",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Where to append audit messages (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be cleaned up but do not remove files",
    )
    return parser.parse_args(argv)


def log_message(message: str, *, logfile: Path) -> None:
    timestamp = _now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as handle:
        handle.write(formatted + "\n")


def station_from_lock_name(path: Path) -> str | None:
    match = re.fullmatch(r"step1_station_(\d+)\.lock", path.name)
    if match:
        return match.group(1)
    return None


def pipeline_running(station: str) -> bool:
    pattern = f"guide_raw_to_corrected.sh {station}"
    try:
        result = subprocess.run(
            ["pgrep", "-af", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # Fall back to assuming a process is running to avoid false positives.
        return True
    return result.returncode == 0 and result.stdout.strip() != ""


def unpacker_running() -> bool:
    pattern = "unpack_reprocessing_files.sh"
    try:
        result = subprocess.run(
            ["pgrep", "-af", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return True
    return result.returncode == 0 and result.stdout.strip() != ""


def lock_is_stale(lock_path: Path, *, max_age: dt.timedelta) -> tuple[bool, str]:
    try:
        mtime = dt.datetime.fromtimestamp(lock_path.stat().st_mtime, tz=dt.timezone.utc)
    except FileNotFoundError:
        return False, "lock disappeared before inspection"

    age = _now() - mtime
    if age < max_age:
        return False, f"age {age} is newer than threshold"

    station = station_from_lock_name(lock_path)
    if station and pipeline_running(station):
        return False, f"station {station} pipeline still running"

    if lock_path.name == UNPACK_LOCK_NAME:
        if unpacker_running():
            return False, "unpacker still running"
        details = f"age {age} exceeds threshold; no unpacker process found"
        return True, details

    details = f"age {age} exceeds threshold"
    if station:
        details += f"; no pipeline process found for station {station}"
    else:
        details += "; station unknown"
    return True, details


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    threshold = dt.timedelta(minutes=args.max_age_minutes)
    lock_dir = args.lock_dir
    log_path = args.log_file

    if not lock_dir.exists():
        log_message(f"Lock directory {lock_dir} does not exist; nothing to do", logfile=log_path)
        return 0

    stale_found = False
    for lock_file in sorted(lock_dir.glob("*.lock")):
        is_stale, reason = lock_is_stale(lock_file, max_age=threshold)
        if not is_stale:
            continue
        stale_found = True
        action = "DRY-RUN would delete" if args.dry_run else "Deleting"
        log_message(f"{action} stale lock {lock_file}: {reason}", logfile=log_path)
        if args.dry_run:
            continue
        try:
            lock_file.unlink()
        except FileNotFoundError:
            log_message(f"Lock {lock_file} vanished before removal", logfile=log_path)
        except OSError as exc:
            log_message(f"Failed to delete {lock_file}: {exc}", logfile=log_path)

    if not stale_found:
        log_message("No stale locks detected", logfile=log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
