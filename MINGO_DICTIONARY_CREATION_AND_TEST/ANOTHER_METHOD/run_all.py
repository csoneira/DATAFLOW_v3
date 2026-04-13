#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from common import DEFAULT_CONFIG_PATH, load_config
from step_1_prepare_data import run as run_step_1
from step_2_build_lut import run as run_step_2
from step_3_apply_lut import run as run_step_3
from step_4_study_lut import run as run_step_4
from step_5_apply_lut_to_real_data import run as run_step_5


def run_collect_data() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    collect_data_script = (
        project_dir
        / "STEPS"
        / "STEP_1_SETUP"
        / "STEP_1_1_COLLECT_DATA"
        / "collect_data.py"
    )
    print(f"Running upstream data collection: {collect_data_script}")
    subprocess.run([sys.executable, str(collect_data_script)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five-step LUT workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    run_collect_data()
    config = load_config(config_path)
    run_step_1(config_path)
    run_step_2(config_path)
    run_step_3(config_path)
    if bool(config.get("step4", {}).get("enabled", True)):
        run_step_4(config_path)
    if bool(config.get("step5", {}).get("enabled", True)):
        run_step_5(config_path)


if __name__ == "__main__":
    main()
