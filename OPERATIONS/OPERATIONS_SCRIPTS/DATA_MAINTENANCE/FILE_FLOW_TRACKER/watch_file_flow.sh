#!/usr/bin/env bash

# Direct execution:
#   watch -n 1 -c -t \
#     /home/mingo/DATAFLOW_v3/OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/FILE_FLOW_TRACKER/watch_file_flow.sh
#
# Select one station:
#   watch -n 1 -c -t \
#     /home/mingo/DATAFLOW_v3/OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/FILE_FLOW_TRACKER/watch_file_flow.sh 2
#
# Using the ~/.bashrc function, which you can add like:
#   watch_file_flow() { watch -n 1 -c -t /home/mingo/DATAFLOW_v3/OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/FILE_FLOW_TRACKER/watch_file_flow.sh "$@"; }
#
# And use it like:
#   watch_file_flow       # all stations
#   watch_file_flow 2     # MINGO02

set -euo pipefail

TRACKER_DIR="$HOME/DATAFLOW_v3/OPERATIONS/OPERATIONS_RUNTIME/FILE_FLOW_TRACKER"
ROWS="${ROWS:-25}"

usage() {
    cat <<EOF
Usage:
    $(basename "$0")              Show all stations
    $(basename "$0") NUMBER       Show one station

Examples:
    $(basename "$0")
    $(basename "$0") 1
    $(basename "$0") 2

Environment variables:
    ROWS=40 $(basename "$0")      Show 40 rows
EOF
}

# ---------------------------------------------------------------------------
# Select input CSV files
# ---------------------------------------------------------------------------

CSV_FILES=()
DISPLAY_SCOPE="ALL STATIONS"

case "$#" in
    0)
        shopt -s nullglob
        CSV_FILES=(
            # "$TRACKER_DIR"/MINGO*_file_flow_latest.csv
            "$TRACKER_DIR"/MINGO*_file_flow_realtime.csv
        )
        shopt -u nullglob

        if (( ${#CSV_FILES[@]} == 0 )); then
            printf \
                '\033[31mERROR: no station CSV files found in %s\033[0m\n' \
                "$TRACKER_DIR"
            exit 1
        fi
        ;;

    1)
        if [[ "$1" == "-h" || "$1" == "--help" ]]; then
            usage
            exit 0
        fi

        if [[ ! "$1" =~ ^[0-9]+$ ]]; then
            printf \
                '\033[31mERROR: station must be given as a number.\033[0m\n' \
                >&2
            usage >&2
            exit 2
        fi

        # 10# prevents values such as 08 or 09 from being interpreted as octal.
        STATION_NUMBER=$((10#$1))
        printf -v STATION_ID 'MINGO%02d' "$STATION_NUMBER"

        # CSV="$TRACKER_DIR/${STATION_ID}_file_flow_latest.csv"
        CSV="$TRACKER_DIR/${STATION_ID}_file_flow_realtime.csv"

        if [[ ! -r "$CSV" ]]; then
            printf \
                '\033[31mERROR: cannot read %s\033[0m\n' \
                "$CSV"
            exit 1
        fi

        CSV_FILES=("$CSV")
        DISPLAY_SCOPE="$STATION_ID"
        ;;

    *)
        printf \
            '\033[31mERROR: too many arguments.\033[0m\n' \
            >&2
        usage >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Generate console display
# ---------------------------------------------------------------------------

python3 - "$ROWS" "$DISPLAY_SCOPE" "${CSV_FILES[@]}" <<'PY'
import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

max_rows = int(sys.argv[1])
display_scope = sys.argv[2]
csv_paths = [Path(path) for path in sys.argv[3:]]

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def true_value(value):
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def shorten(value, width):
    value = str(value or "")

    if len(value) <= width:
        return value

    return value[: width - 1] + "…"


def parse_time(value):
    """
    Convert an ISO-8601 timestamp to a Unix timestamp.

    Empty or malformed values receive negative infinity, which places them
    at the bottom when sorting in descending order.
    """
    value = str(value or "").strip()

    if not value:
        return float("-inf")

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.timestamp()

    except (
        ValueError,
        TypeError,
        AttributeError,
        OverflowError,
    ):
        return float("-inf")


def display_time(value):
    """
    Format a timestamp compactly for display.
    """
    value = str(value or "").strip()

    if not value:
        return "-"

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        return parsed.strftime("%Y-%m-%d %H:%M:%S")

    except (
        ValueError,
        TypeError,
        AttributeError,
    ):
        return value


def task_symbol(value):
    state = str(value or "").strip().lower()

    if not state or state in {
        "not_reached",
        "none",
        "n/a",
    }:
        return f"{DIM}·{RESET}"

    if any(
        marker in state
        for marker in (
            "fail",
            "error",
            "invalid",
            "corrupt",
            "manual_review",
            "blocked",
        )
    ):
        return f"{RED}!{RESET}"

    if any(
        marker in state
        for marker in (
            "running",
            "processing",
            "active",
            "in_progress",
        )
    ):
        return f"{CYAN}▶{RESET}"

    if any(
        marker in state
        for marker in (
            "complete",
            "completed",
            "done",
            "success",
            "present",
            "current",
            "up_to_date",
            "archived",
            "valid",
        )
    ):
        return f"{GREEN}✓{RESET}"

    if any(
        marker in state
        for marker in (
            "out_of_date",
            "pending",
            "queued",
            "waiting",
            "reprocess",
        )
    ):
        return f"{YELLOW}○{RESET}"

    if state in {
        "missing",
        "absent",
        "false",
        "0",
    }:
        return f"{DIM}−{RESET}"

    return f"{MAGENTA}?{RESET}"


def row_priority(row):
    """
    Sort primarily by newest artifact modification time.

    The remaining criteria are descending-order tie-breakers.
    """
    return (
        parse_time(
            row.get(
                "newest_artifact_mtime_utc",
                "",
            )
        ),
        true_value(
            row.get(
                "active_reprocessing",
                "",
            )
        ),
        true_value(
            row.get(
                "in_selected_range",
                "",
            )
        ),
        true_value(
            row.get(
                "needs_reprocessing",
                "",
            )
        ),
        bool(
            row.get(
                "anomaly_codes",
                "",
            ).strip()
        ),
        not true_value(
            row.get(
                "pipeline_consistent",
                "",
            )
        ),
        parse_time(
            row.get(
                "event_time_utc",
                "",
            )
        ),
    )


# ---------------------------------------------------------------------------
# Read and combine every selected CSV
# ---------------------------------------------------------------------------

rows = []
read_errors = []

for csv_path in csv_paths:
    try:
        with csv_path.open(
            newline="",
            encoding="utf-8",
        ) as file_handle:
            reader = csv.DictReader(file_handle)

            for row in reader:
                # Preserve the source file for diagnostics, without changing
                # existing tracker columns.
                row["_source_csv"] = csv_path.name
                rows.append(row)

    except (OSError, csv.Error) as exc:
        read_errors.append(
            f"{csv_path}: {exc}"
        )

if read_errors:
    for error in read_errors:
        print(
            f"{RED}WARNING: {error}{RESET}",
            file=sys.stderr,
        )

if not rows:
    print(
        f"{YELLOW}"
        "No tracker records were found in the selected CSV files."
        f"{RESET}"
    )
    raise SystemExit(0)

rows.sort(
    key=row_priority,
    reverse=True,
)

# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

stations = sorted(
    {
        row.get("station", "").strip()
        for row in rows
        if row.get("station", "").strip()
    }
)

observed_values = [
    row.get("observed_at_utc", "")
    for row in rows
    if row.get("observed_at_utc", "")
]

if observed_values:
    observed = max(
        observed_values,
        key=parse_time,
    )
else:
    observed = "unknown"

total = len(rows)

selected = sum(
    true_value(
        row.get(
            "in_selected_range",
            "",
        )
    )
    for row in rows
)

consistent = sum(
    true_value(
        row.get(
            "pipeline_consistent",
            "",
        )
    )
    for row in rows
)

reprocess = sum(
    true_value(
        row.get(
            "needs_reprocessing",
            "",
        )
    )
    for row in rows
)

active = sum(
    true_value(
        row.get(
            "active_reprocessing",
            "",
        )
    )
    for row in rows
)

anomalous = sum(
    bool(
        row.get(
            "anomaly_codes",
            "",
        ).strip()
    )
    for row in rows
)

lifecycle_counts = Counter(
    row.get(
        "lifecycle_state",
        "",
    ).strip()
    or "(empty)"
    for row in rows
)

action_counts = Counter(
    row.get(
        "recommended_action",
        "",
    ).strip()
    or "(none)"
    for row in rows
)

station_counts = Counter(
    row.get(
        "station",
        "",
    ).strip()
    or "(unknown)"
    for row in rows
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print(
    f"{BOLD}{CYAN}"
    f"FILE FLOW TRACKER — {display_scope}"
    f"{RESET}"
)

print(
    f"{DIM}"
    f"Current:  {display_time(datetime.now())} UTC\n"
    f"Snapshot: {display_time(observed)} UTC"
    f"{RESET}"
)

print(
    f"CSV files: {BOLD}{len(csv_paths)}{RESET}   "
    f"Stations: {BOLD}{len(stations)}{RESET}   "
    f"Files: {BOLD}{total}{RESET}   "
    f"In range: {BLUE}{selected}{RESET}   "
    f"Consistent: {GREEN}{consistent}{RESET}   "
    f"Reprocess: {YELLOW}{reprocess}{RESET}   "
    f"Active: {CYAN}{active}{RESET}   "
    f"Anomalies: {RED}{anomalous}{RESET}"
)

if len(stations) > 1:
    top_stations = "  ".join(
        f"{name}={count}"
        for name, count in station_counts.most_common()
    )

    print(
        f"{BOLD}Stations: {RESET}"
        f"{top_stations}"
    )

top_lifecycle = "  ".join(
    f"{name}={count}"
    for name, count in lifecycle_counts.most_common(5)
)

top_actions = "  ".join(
    f"{name}={count}"
    for name, count in action_counts.most_common(4)
)

print(
    f"{BOLD}Lifecycle:{RESET} "
    f"{top_lifecycle}"
)

print(
    f"{BOLD}Actions:  {RESET} "
    f"{top_actions}"
)

print()

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

print(
    f"{BOLD}"
    f"{'STATION':<8} "
    f"{'FILE':<20} "
    f"{'LAST MODIFIED':<19} "
    f"{'EVENT TIME':<19} "
    f"{'R':<1} "
    f"{'LIFECYCLE':<25} "
    f"{'HIGH':>4} "
    f"{'TASK FLOW':<23} "
    f"{'ACTION / ANOMALY'}"
    f"{RESET}"
)

print("─" * 158)

for row in rows[:max_rows]:
    selected_mark = (
        f"{BLUE}●{RESET}"
        if true_value(
            row.get(
                "in_selected_range",
                "",
            )
        )
        else f"{DIM}·{RESET}"
    )

    flow = " ".join(
        f"{task_number}:"
        f"{task_symbol(row.get(f'task_{task_number}_state'))}"
        for task_number in range(6)
    )

    lifecycle = row.get(
        "lifecycle_state",
        "",
    )

    if "manual_review" in lifecycle.lower():
        lifecycle_display = (
            f"{RED}"
            f"{shorten(lifecycle, 25):<25}"
            f"{RESET}"
        )

    elif true_value(
        row.get(
            "active_reprocessing",
            "",
        )
    ):
        lifecycle_display = (
            f"{CYAN}"
            f"{shorten(lifecycle, 25):<25}"
            f"{RESET}"
        )

    elif true_value(
        row.get(
            "needs_reprocessing",
            "",
        )
    ):
        lifecycle_display = (
            f"{YELLOW}"
            f"{shorten(lifecycle, 25):<25}"
            f"{RESET}"
        )

    else:
        lifecycle_display = (
            f"{shorten(lifecycle, 25):<25}"
        )

    action = row.get(
        "recommended_action",
        "",
    ).strip()

    anomaly = row.get(
        "anomaly_codes",
        "",
    ).strip()

    if anomaly:
        if action:
            detail = f"{action}; {anomaly}"
        else:
            detail = anomaly

        detail = (
            f"{RED}"
            f"{shorten(detail, 35)}"
            f"{RESET}"
        )

    else:
        detail = shorten(
            action,
            35,
        )

    station = row.get(
        "station",
        "?",
    )

    filename = row.get(
        "filename_base",
        "",
    )

    modified_time = display_time(
        row.get(
            "newest_artifact_mtime_utc",
            "",
        )
    )

    event_time = display_time(
        row.get(
            "event_time_utc",
            "",
        )
    )

    highest_task = row.get(
        "highest_task_reached",
        "",
    )

    print(
        f"{shorten(station, 8):<8} "
        f"{shorten(filename, 20):<20} "
        f"{shorten(modified_time, 19):<19} "
        f"{shorten(event_time, 19):<19} "
        f"{selected_mark} "
        f"{lifecycle_display} "
        f"{shorten(highest_task, 4):>4} "
        f"{flow:<68} "
        f"{detail}"
    )

print()

print(
    f"{DIM}"
    "Sorted by LAST MODIFIED, newest first."
    f"{RESET}"
)

print(
    f"{DIM}Legend: "
    f"{GREEN}✓ complete{RESET}{DIM}, "
    f"{CYAN}▶ active{RESET}{DIM}, "
    f"{YELLOW}○ pending/outdated{RESET}{DIM}, "
    f"{RED}! error/review{RESET}{DIM}, "
    f"· not reached, "
    f"{BLUE}● selected range{RESET}"
)
PY