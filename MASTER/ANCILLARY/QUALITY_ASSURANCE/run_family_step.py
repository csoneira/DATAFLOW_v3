#!/usr/bin/env python3
"""Run one family-style QUALITY_ASSURANCE step from its step-level config."""

from __future__ import annotations

from pathlib import Path
import sys

QA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = QA_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.task_setup import load_step_configs
from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.time_series_qa import run_time_series_family_step


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: run_family_step.py /abs/path/to/STEP_DIR")
        return 2

    step_dir = Path(argv[0]).resolve()
    config = load_step_configs(step_dir)

    task_ids = [int(value) for value in config.get("task_ids", [])]
    metadata_suffix = str(config.get("metadata_suffix", "")).strip()
    metadata_type = str(config.get("metadata_type", "")).strip()
    pass_column_template = str(config.get("pass_column_template", "")).strip()
    allow_migration_matrix = bool(config.get("allow_migration_matrix", False))

    if not task_ids:
        raise ValueError(f"{step_dir} is missing non-empty 'task_ids'.")
    if not metadata_suffix:
        raise ValueError(f"{step_dir} is missing 'metadata_suffix'.")
    if not metadata_type:
        raise ValueError(f"{step_dir} is missing 'metadata_type'.")
    if not pass_column_template:
        raise ValueError(f"{step_dir} is missing 'pass_column_template'.")

    return run_time_series_family_step(
        step_dir,
        task_ids=task_ids,
        metadata_suffix=metadata_suffix,
        metadata_type=metadata_type,
        pass_column_template=pass_column_template,
        allow_migration_matrix=allow_migration_matrix,
    )


if __name__ == "__main__":
    raise SystemExit(main())
