#!/usr/bin/env python3
"""Wrapper to run the TASK_3 analysis script but pointing to TASK_2 activation metadata.

Creates OUTPUTS_TASK2 under the same directory and runs the analysis using the
activation matrix families found in task 2 metadata.
"""
from __future__ import annotations

import glob
import subprocess
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

pattern = str(
    REPO_ROOT
    / "STATIONS"
    / "MINGO0*"
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_2"
    / "METADATA"
    / "task_2_metadata_activation.csv"
)

outdir = SCRIPT_DIR / "OUTPUTS_TASK2"
outdir.mkdir(parents=True, exist_ok=True)

# Look for matching files first to avoid running the analysis with an empty pattern
matches = sorted(glob.glob(pattern, recursive=True))
if not matches:
    print(f"No TASK_2 activation metadata files found with pattern: {pattern}")
    print("Created output directory:", outdir)
    sys.exit(0)

cmd = [
    sys.executable,
    str(SCRIPT_DIR / "task3_noise_baseline_analysis.py"),
    "--pattern",
    pattern,
    "--outdir",
    str(outdir),
]
print("Running:", " ".join(cmd))
ret = subprocess.run(cmd)
if ret.returncode != 0:
    raise SystemExit(ret.returncode)
print("TASK_2 analysis complete. Outputs in:", outdir)
