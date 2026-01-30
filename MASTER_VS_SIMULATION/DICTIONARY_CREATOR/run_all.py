#!/usr/bin/env python3
"""Run DICTIONARY_CREATOR pipeline steps in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config/pipeline_config.json"


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n== {label} ==")
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DICTIONARY_CREATOR pipeline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-clean", action="store_true", help="Do not delete outputs first.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-chisq", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-scatter", action="store_true")
    args = parser.parse_args()

    config = str(args.config)

    import json

    cfg = json.loads(Path(config).read_text())
    task_ids = cfg.get("task_ids") or cfg.get("scatter_tasks") or [1]
    scatter_out_dir = Path(cfg.get("scatter_out_dir", str(BASE_DIR / "STEP_4_SCATTERS/output")))
    filtered_scatter_out = scatter_out_dir / "filtered"

    if not args.no_clean:
        run_step(
            "CLEAN outputs",
            [
                sys.executable,
                str(BASE_DIR / "clean_outputs.py"),
                "--all",
            ],
        )

    if not args.skip_build:
        for task_id in task_ids:
            run_step(
                f"STEP 1: build dictionary (TASK {int(task_id):02d})",
                [
                    sys.executable,
                    str(BASE_DIR / "STEP_1_BUILD/build_param_metadata_dictionary.py"),
                    "--config",
                    config,
                    "--task-id",
                    str(task_id),
                ],
            )

    for task_id in task_ids:
        task_label = f"TASK {int(task_id):02d}"
        chisq_csv = (
            Path(cfg.get("chisq_out_dir", str(BASE_DIR / "STEP_2_CHISQ/output")))
            / f"task_{int(task_id):02d}"
            / "chisq_results.csv"
        )

        if not args.skip_scatter:
            run_step(
                f"STEP 2: scatter (pre-chisq) ({task_label})",
                [
                    sys.executable,
                    str(BASE_DIR / "STEP_4_SCATTERS/scatter_median_vs_reference.py"),
                    "--config",
                    config,
                    "--tasks",
                    str(task_id),
                ],
            )

        if not args.skip_chisq:
            run_step(
                f"STEP 3: compute chisq ({task_label})",
                [
                    sys.executable,
                    str(BASE_DIR / "STEP_2_CHISQ/compute_chisq_filter.py"),
                    "--config",
                    config,
                    "--task-id",
                    str(task_id),
                ],
            )

        if not args.skip_scatter and not args.skip_chisq:
            run_step(
                f"STEP 4A: scatter (post-chisq) ({task_label})",
                [
                    sys.executable,
                    str(BASE_DIR / "STEP_4_SCATTERS/scatter_median_vs_reference.py"),
                    "--config",
                    config,
                    "--tasks",
                    str(task_id),
                    "--chisq-csv",
                    str(chisq_csv),
                    "--out-dir",
                    str(filtered_scatter_out),
                ],
            )

        if not args.skip_plots:
            run_step(
                f"STEP 4B: plots ({task_label})",
                [
                    sys.executable,
                    str(BASE_DIR / "STEP_3_PLOTS/plot_station_metadata_vs_dictionary.py"),
                    "--config",
                    config,
                    "--task-id",
                    str(task_id),
                    "--chisq-csv",
                    str(chisq_csv),
                ],
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
