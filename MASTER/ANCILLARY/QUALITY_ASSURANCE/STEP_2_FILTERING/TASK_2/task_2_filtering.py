#!/usr/bin/env python3
"""TASK_2 filtering QA wrapper."""

from __future__ import annotations

from pathlib import Path
import sys

TASK_DIR = Path(__file__).resolve().parent
STEP_DIR = TASK_DIR.parent
QA_ROOT = STEP_DIR.parent
REPO_ROOT = QA_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.time_series_qa import run_time_series_task


def main() -> int:
    return run_time_series_task(
        TASK_DIR,
        task_id=2,
        metadata_suffix="filter",
        metadata_type="filtering",
        default_pass_column="task_2_filtering_pass",
    )


if __name__ == "__main__":
    raise SystemExit(main())
