#!/usr/bin/env python3
"""Build param_metadata_dictionary.csv files without running plots or chisq steps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config/pipeline_config.json"


def _parse_task_ids(raw: str) -> list[int]:
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    task_ids: list[int] = []
    for value in values:
        task_ids.append(int(value))
    return task_ids


def _load_task_ids(config_path: Path) -> list[int]:
    cfg = json.loads(config_path.read_text())
    task_ids = cfg.get("task_ids") or cfg.get("scatter_tasks")
    if not task_ids:
        task_id = cfg.get("task_id", 1)
        return [int(task_id)]
    return [int(task_id) for task_id in task_ids]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build dictionary CSVs only (no chisq, scatter, or plots)."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task IDs to build (overrides config).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.task_ids:
        task_ids = _parse_task_ids(args.task_ids)
    else:
        task_ids = _load_task_ids(config_path)

    for task_id in task_ids:
        cmd = [
            sys.executable,
            str(BASE_DIR / "STEP_1_BUILD/build_param_metadata_dictionary.py"),
            "--config",
            str(config_path),
            "--task-id",
            str(task_id),
        ]
        print("\n== BUILD DICTIONARY (TASK {:02d}) ==".format(task_id))
        print(" ".join(cmd))
        subprocess.check_call(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
