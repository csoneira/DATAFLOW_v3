#!/usr/bin/env python3
"""Remove pipeline outputs so runs start from scratch."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DIRS = [
    BASE_DIR / "STEP_1_BUILD/output",
    BASE_DIR / "STEP_2_SCATTERS/output",
    BASE_DIR / "STEP_2_CHISQ/output",
    BASE_DIR / "STEP_3_PLOTS/output",
    BASE_DIR / "STEP_4_SCATTERS/output",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete DICTIONARY_CREATOR outputs.")
    parser.add_argument("--all", action="store_true", help="Also delete STEP_1_BUILD/output")
    args = parser.parse_args()

    dirs = list(DEFAULT_DIRS)
    if not args.all:
        # keep step 1 unless explicitly asked
        dirs = [d for d in dirs if d.parent.name != "STEP_1_BUILD"]

    for path in dirs:
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed {path}")
        else:
            print(f"Skip (not found): {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
