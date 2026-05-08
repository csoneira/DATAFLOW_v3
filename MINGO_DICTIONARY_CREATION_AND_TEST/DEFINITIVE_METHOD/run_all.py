#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from simple_common import DEFAULT_CONFIG_PATH
from step_0_load_inputs import run as run_step_0
from step_1_build_lut import run as run_step_1
from step_2_apply_lut import run as run_step_2
from step_3_rate_to_flux import run as run_step_3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four-step definitive LUT workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    run_step_0(config_path)
    run_step_1(config_path)
    run_step_2(config_path)
    run_step_3(config_path)


if __name__ == "__main__":
    main()
