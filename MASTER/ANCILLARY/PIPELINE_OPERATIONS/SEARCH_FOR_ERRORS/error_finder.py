#!/usr/bin/env python3
"""
Scan cron log files for Tracebacks and store the stack traces in a single report.

The script remembers which portion of every cron log was already checked by adding
markers inside the log files themselves. Only Tracebacks found after the most
recent marker are inspected, which keeps subsequent runs fast. Unique Tracebacks
are appended to the report alongside the execution timestamp and the source log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOG_DIR = PROJECT_ROOT / "EXECUTION_LOGS" / "CRON_LOGS"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "ERROR_SUMMARIES"
OUTPUT_FILE = OUTPUT_DIR / "cron_tracebacks.log"
STATE_FILE = OUTPUT_DIR / "error_finder_state.json"
TRACEBACK_CATALOG_FILE = OUTPUT_DIR / "traceback_catalog.json"

ENTRY_DIVIDER = "# --------------------"
MARKER_PREFIX = "### REVISED BY ERROR_FINDER AT "
TRACEBACK_TRIGGER = "Traceback (most recent call last):"
TIMESTAMP_PATTERNS = [
    re.compile(r"^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"),
    re.compile(r"^[A-Z][a-z]{2} [A-Z][a-z]{2}\s+\d{1,2} \d{2}:\d{2}:\d{2}"),
]


def is_timestamp_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in TIMESTAMP_PATTERNS)


def load_known_hashes(state_path: Path) -> Set[str]:
    if not state_path.exists():
        return set()
    try:
        with state_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        return set()
    hashes = data.get("known_hashes", [])
    return set(hashes)


def save_known_hashes(state_path: Path, hashes: Set[str]) -> None:
    state_path.write_text(
        json.dumps({"known_hashes": sorted(hashes)}, indent=2),
        encoding="utf-8",
    )


def load_traceback_catalog(catalog_path: Path) -> Dict[str, Dict[str, str]]:
    if not catalog_path.exists():
        return {}
    try:
        with catalog_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def save_traceback_catalog(catalog_path: Path, catalog: Dict[str, Dict[str, str]]) -> None:
    catalog_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_lines_after_last_marker(log_path: Path, *, full_scan: bool = False) -> Tuple[List[str], bool]:
    try:
        raw_text = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return [], False
    lines = raw_text.splitlines()
    if full_scan:
        return lines, False
    last_marker_index = -1
    for idx, line in enumerate(lines):
        if line.startswith(MARKER_PREFIX):
            last_marker_index = idx
    start_index = last_marker_index + 1
    if start_index >= len(lines):
        return [], False
    return lines[start_index:], True


def extract_tracebacks(lines: Sequence[str]) -> List[str]:
    tracebacks: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if TRACEBACK_TRIGGER in line:
            block = [line]
            j = i + 1
            seen_exception_line = False
            while j < len(lines):
                current = lines[j]
                if current.startswith(MARKER_PREFIX):
                    break
                if TRACEBACK_TRIGGER in current:
                    break
                if is_timestamp_line(current):
                    break
                if not current.strip():
                    block.append(current)
                    j += 1
                    break
                is_indented = current.startswith((" ", "\t"))
                if not is_indented and seen_exception_line:
                    break
                block.append(current)
                if not is_indented:
                    seen_exception_line = True
                j += 1
            tracebacks.append("\n".join(block).strip())
            i = j
        else:
            i += 1
    return tracebacks


def append_marker(log_path: Path, timestamp: str) -> None:
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(f"\n{MARKER_PREFIX}{timestamp} ###\n")


def append_entries(entries: Iterable[Tuple[str, str, str, str]]) -> None:
    OUTPUT_DIR.mkdir -p(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("a", encoding="utf-8") as fp:
        for timestamp, log_name, digest, traceback_text in entries:
            fp.write(f"{ENTRY_DIVIDER}\n")
            fp.write(f"{timestamp} | {log_name} | HASH {digest}\n")
            fp.write(f"{traceback_text}\n\n")


def append_no_error_entry(timestamp: str) -> None:
    OUTPUT_DIR.mkdir -p(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("a", encoding="utf-8") as fp:
        fp.write(f"{ENTRY_DIVIDER}\n")
        fp.write(f"{timestamp} | NO ERRORS FOUND\n\n")


def append_pending_entry(timestamp: str, pending_counts: Counter) -> None:
    OUTPUT_DIR.mkdir -p(parents=True, exist_ok=True)
    total_occurrences = sum(pending_counts.values())
    unique = len(pending_counts)
    with OUTPUT_FILE.open("a", encoding="utf-8") as fp:
        fp.write(f"{ENTRY_DIVIDER}\n")
        fp.write(
            f"{timestamp} | NO NEW ERRORS, BUT STILL THESE ERRORS PENDING: "
            f"{total_occurrences} occurrences across {unique} tracebacks\n"
        )
        for digest, count in pending_counts.most_common():
            fp.write(f"  - {digest}: {count} occurrences (see {TRACEBACK_CATALOG_FILE.name})\n")
        fp.write("\n")


def process_log(log_path: Path, *, full_scan: bool = False) -> Tuple[List[str], bool]:
    lines, has_new_section = read_lines_after_last_marker(log_path, full_scan=full_scan)
    if not lines:
        return [], has_new_section
    tracebacks = extract_tracebacks(lines)
    return tracebacks, has_new_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Tracebacks from cron logs into a single summary file.",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Re-scan the complete cron logs, ignoring previous markers and state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir -p(parents=True, exist_ok=True)
    if not LOG_DIR.exists():
        raise SystemExit(f"Cron log directory not found: {LOG_DIR}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    known_hashes = set() if args.full_scan else load_known_hashes(STATE_FILE)
    catalog = load_traceback_catalog(TRACEBACK_CATALOG_FILE)
    catalog_modified = False
    entries_to_write: List[Tuple[str, str, str, str]] = []
    pending_counts: Counter = Counter()
    encountered_hashes: Set[str] = set()
    state_modified = args.full_scan

    for log_path in sorted(LOG_DIR.glob("*.log")):
        tracebacks, has_new_content = process_log(log_path, full_scan=args.full_scan)
        for tb in tracebacks:
            digest = hashlib.sha1(tb.encode("utf-8")).hexdigest()
            encountered_hashes.add(digest)
            existing_entry = catalog.get(digest)
            if existing_entry:
                if existing_entry.get("traceback") != tb:
                    existing_entry["traceback"] = tb
                    catalog_modified = True
            else:
                catalog[digest] = {
                    "traceback": tb,
                    "first_seen": now,
                    "source_log": log_path.name,
                }
                catalog_modified = True
            if digest in known_hashes:
                pending_counts[digest] += 1
                continue
            known_hashes.add(digest)
            state_modified = True
            entries_to_write.append((now, log_path.name, digest, tb))
        if has_new_content and not args.full_scan:
            append_marker(log_path, now)

    # Only treat hashes as resolved when performing a full scan. During the normal
    # incremental runs we only inspect the portion of the log written after our
    # previous marker, so the absence of a digest does not mean the error stopped
    # occurring. Deleting hashes in that situation caused the same traceback to be
    # reported repeatedly.
    if args.full_scan:
        resolved_hashes = {digest for digest in known_hashes if digest not in encountered_hashes}
        if resolved_hashes:
            known_hashes.difference_update(resolved_hashes)
            state_modified = True
            for digest in resolved_hashes:
                if digest in catalog:
                    del catalog[digest]
                    catalog_modified = True

    if entries_to_write:
        append_entries(entries_to_write)
    else:
        if pending_counts:
            append_pending_entry(now, pending_counts)
        else:
            append_no_error_entry(now)
    if state_modified:
        save_known_hashes(STATE_FILE, known_hashes)
    if catalog_modified:
        save_traceback_catalog(TRACEBACK_CATALOG_FILE, catalog)


if __name__ == "__main__":
    main()
