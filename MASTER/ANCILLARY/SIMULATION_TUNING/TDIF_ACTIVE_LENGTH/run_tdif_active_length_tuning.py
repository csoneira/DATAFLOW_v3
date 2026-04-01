#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    config_path = Path(__file__).resolve().parents[1] / "config_simulation_tuning.yaml"
    output_dir = Path(__file__).resolve().parent / "OUTPUTS"
    core_script = (
        repo_root
        / "MASTER"
        / "ANCILLARY"
        / "TDIF_SPAN_COMPARISON"
        / "compare_task2_tdif_span.py"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(core_script),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
