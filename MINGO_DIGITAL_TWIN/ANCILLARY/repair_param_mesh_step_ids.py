#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

# Backward-compatible wrapper. Canonical location:
#   ORCHESTRATOR/maintenance/repair_param_mesh_step_ids.py
TARGET = Path(__file__).resolve().parents[1] / "ORCHESTRATOR/maintenance/repair_param_mesh_step_ids.py"

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
