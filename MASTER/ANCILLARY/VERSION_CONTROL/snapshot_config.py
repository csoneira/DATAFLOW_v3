#!/usr/bin/env python3
"""Persist snapshots of the main config.yaml when it changes."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import json
import yaml

# Dynamically get the home directory using the environment variable
home_directory = Path(os.environ.get("HOME", os.path.expanduser("~")))

# Define paths relative to the home directory
CONFIG_PATH = home_directory / "DATAFLOW_v3" / "MASTER" / "config.yaml"
SNAPSHOT_DIR = home_directory / "DATAFLOW_v3" / "EXECUTION_LOGS" / "CONFIG_FILES"

# Test the paths
print(f"Config Path: {CONFIG_PATH}")
print(f"Snapshot Directory: {SNAPSHOT_DIR}")


def extract_json_payload(snapshot_path: Path) -> str:
    """Return the JSON payload stored in *snapshot_path* (skip header)."""
    lines = snapshot_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines) and lines[index].lstrip().startswith("#"):
        index += 1
    return "\n".join(lines[index:]).strip()


def latest_snapshot_payload(directory: Path) -> Optional[str]:
    """Return the JSON payload from the most recent snapshot, if any."""
    snapshots = sorted(directory.glob("*_config.json"))
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


def main() -> int:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    current_payload = load_config_as_json(CONFIG_PATH)
    previous_payload = latest_snapshot_payload(SNAPSHOT_DIR)

    if previous_payload is not None and previous_payload == current_payload:
        return 0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    snapshot_path = SNAPSHOT_DIR / f"{timestamp}_config.json"
    header = f"# Snapshot generated on {timestamp}\n"

    snapshot_path.write_text(f"{header}{current_payload}\n", encoding="utf-8")
    print(f"Saved new config snapshot: {snapshot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
