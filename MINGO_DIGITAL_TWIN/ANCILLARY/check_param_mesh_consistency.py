#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

# Backward-compatible wrapper. Canonical location:
#   ORCHESTRATOR/helpers/check_param_mesh_consistency.py
TARGET = Path(__file__).resolve().parents[1] / "ORCHESTRATOR/helpers/check_param_mesh_consistency.py"

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
