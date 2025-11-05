#!/usr/bin/env python3
"""Persist snapshots of the main config files when they change."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import os

import json
import yaml
import csv

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.execution_logger import start_timer

start_timer(__file__)

# Dynamically get the home directory using the environment variable
home_directory = Path(os.environ.get("HOME", os.path.expanduser("~")))

# Define paths relative to the home directory
# /home/mingo/DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml
CONFIG_PATH = home_directory / "DATAFLOW_v3" / "MASTER" / "CONFIG_FILES" / "config_global.yaml"
SNAPSHOT_DIR = home_directory / "DATAFLOW_v3" / "EXECUTION_LOGS" / "CONFIG_FILES" / "GLOBAL"

CONFIG_PATH_PARAM = home_directory / "DATAFLOW_v3" / "MASTER" / "CONFIG_FILES" / "config_parameters.csv"
SNAPSHOT_DIR_PARAM = home_directory / "DATAFLOW_v3" / "EXECUTION_LOGS" / "CONFIG_FILES" / "PARAMETERS"

# Test the paths
print(f"Config Path: {CONFIG_PATH}")
print(f"Snapshot Directory: {SNAPSHOT_DIR}")
print(f"Parameters Config Path: {CONFIG_PATH_PARAM}")
print(f"Parameters Snapshot Directory: {SNAPSHOT_DIR_PARAM}")


def extract_json_payload(snapshot_path: Path) -> str:
    """Return the JSON payload stored in *snapshot_path* (skip header)."""
    lines = snapshot_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines) and lines[index].lstrip().startswith("#"):
        index += 1
    return "\n".join(lines[index:]).strip()


def latest_snapshot_payload(directory: Path, name_suffix: str) -> Optional[str]:
    """Return the JSON payload from the most recent snapshot with the given suffix, if any."""
    snapshots = sorted(directory.glob(f"*_{name_suffix}"))
    if not snapshots:
        return None
    return extract_json_payload(snapshots[-1])


def load_config_as_json(config_path: Path) -> str:
    """Load YAML config and return a stable JSON string representation."""
    try:
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {config_path}") from exc

    return json.dumps(config_data, indent=2, sort_keys=True)


def load_csv_as_json(config_path: Path) -> str:
    """Load CSV config and return a stable JSON string representation."""
    try:
        with config_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {config_path}") from exc

    return json.dumps(rows, indent=2, sort_keys=True)


def snapshot_if_changed(
    config_path: Path,
    snapshot_dir: Path,
    payload_loader,
    name_suffix: str,
    label: str,
) -> bool:
    """Create a snapshot when payload differs from the latest stored version."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    current_payload = payload_loader(config_path)
    previous_payload = latest_snapshot_payload(snapshot_dir, name_suffix)

    if previous_payload is not None and previous_payload == current_payload:
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    snapshot_path = snapshot_dir / f"{timestamp}_{name_suffix}"
    header = f"# Snapshot generated on {timestamp}\n"
    snapshot_path.write_text(f"{header}{current_payload}\n", encoding="utf-8")

    print(f"Saved new {label} snapshot: {snapshot_path}")
    return True


def main() -> int:
    snapshot_if_changed(
        CONFIG_PATH,
        SNAPSHOT_DIR,
        load_config_as_json,
        "config.json",
        "config",
    )

    snapshot_if_changed(
        CONFIG_PATH_PARAM,
        SNAPSHOT_DIR_PARAM,
        load_csv_as_json,
        "parameters.json",
        "parameter config",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
