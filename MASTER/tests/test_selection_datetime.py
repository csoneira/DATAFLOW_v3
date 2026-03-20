from __future__ import annotations

from datetime import date, datetime, time, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.file_selection import file_name_in_any_date_range
from MASTER.common.selection_config import (
    date_in_ranges,
    extract_selection,
    serialize_date_ranges_for_shell,
)


def test_extract_selection_preserves_subday_precision_and_day_bounds() -> None:
    selection = extract_selection(
        {
            "selection": {
                "date_ranges": [
                    {
                        "start": "2024-09-27 12:00:00",
                        "end": "2024-09-28 12:34:56",
                    },
                    {
                        "start": "2024-09-29",
                        "end": "2024-09-30",
                    },
                ]
            }
        }
    )

    assert selection.date_ranges[0] == (
        datetime(2024, 9, 27, 12, 0, 0),
        datetime(2024, 9, 28, 12, 34, 56),
    )
    assert selection.date_ranges[1] == (
        datetime(2024, 9, 29, 0, 0, 0),
        datetime.combine(date(2024, 9, 30), time.max),
    )


def test_file_selection_uses_exact_timestamp_bounds() -> None:
    date_ranges = [
        (
            datetime(2024, 9, 27, 12, 0, 0),
            datetime(2024, 9, 28, 12, 0, 0),
        )
    ]

    assert not file_name_in_any_date_range(
        "cleaned_mi0124271115959.parquet",
        date_ranges,
    )
    assert file_name_in_any_date_range(
        "cleaned_mi0124271120000.parquet",
        date_ranges,
    )
    assert not file_name_in_any_date_range(
        "cleaned_mi0124272120001.parquet",
        date_ranges,
    )


def test_day_overlap_and_shell_serialization_support_partial_days() -> None:
    date_ranges = [
        (
            datetime(2024, 9, 28, 0, 0, 0),
            datetime(2024, 9, 28, 8, 0, 0),
        )
    ]

    assert date_in_ranges(date(2024, 9, 28), date_ranges)
    assert not date_in_ranges(date(2024, 9, 27), date_ranges)

    epochs, labels = serialize_date_ranges_for_shell(date_ranges)

    expected_start = int(datetime(2024, 9, 28, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    expected_end = int(datetime(2024, 9, 28, 8, 0, 0, tzinfo=timezone.utc).timestamp())

    assert epochs == f"{expected_start},{expected_end}"
    assert labels == "2024-09-28|2024-09-28 08:00:00"
