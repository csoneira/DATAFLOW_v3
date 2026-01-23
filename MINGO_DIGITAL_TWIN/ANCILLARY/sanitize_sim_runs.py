#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from STEP_SHARED.sim_utils import extract_step_id_chain


def parse_step_index(step_dir: Path) -> int | None:
    match = re.match(r"STEP_(\d+)_TO_", step_dir.name)
    if not match:
        return None
    return int(match.group(1))


def normalize_step_id(value: object) -> str:
    try:
        num = int(float(value))
        return f"{num:03d}"
    except (TypeError, ValueError):
        return str(value)


def combo_done_in_mesh(mesh: pd.DataFrame, step_ids: list[str], step_index: int) -> bool | None:
    if "done" not in mesh.columns:
        return None
    if step_index <= 0:
        return None
    if len(step_ids) < step_index:
        return None
    step_cols = [f"step_{idx}_id" for idx in range(1, step_index + 1)]
    for col in step_cols:
        if col not in mesh.columns:
            return None
    mask = pd.Series(True, index=mesh.index)
    normalized_mesh = {col: mesh[col].map(normalize_step_id) for col in step_cols}
    for col, value in zip(step_cols, step_ids[:step_index]):
        mask &= normalized_mesh[col] == normalize_step_id(value)
    matched = mesh[mask]
    if matched.empty:
        return None
    done_flags = matched["done"].fillna(0).astype(int)
    return bool((done_flags == 1).all())


def find_metadata(sim_run_dir: Path) -> dict | None:
    preferred = sorted(sim_run_dir.glob("step_*_chunks.chunks.json"))
    candidates = preferred or sorted(sim_run_dir.glob("*.chunks.json"))
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        meta = data.get("metadata")
        if isinstance(meta, dict):
            return meta
    return None


def iter_step_dirs(intersteps_dir: Path) -> list[Path]:
    step_dirs = sorted(intersteps_dir.glob("STEP_*_TO_*"))
    return [path for path in step_dirs if path.name != "STEP_0_TO_1"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove SIM_RUN directories when all mesh rows sharing the same step-id combination "
            "for that step are marked done in param_mesh.csv."
        )
    )
    parser.add_argument(
        "--intersteps-dir",
        default="~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS",
        help="Base INTERSTEPS directory.",
    )
    parser.add_argument(
        "--param-mesh",
        default="~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1/param_mesh.csv",
        help="Path to param_mesh.csv.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete directories (default is dry-run).",
    )
    args = parser.parse_args()

    intersteps_dir = Path(args.intersteps_dir).expanduser().resolve()
    mesh_path = Path(args.param_mesh).expanduser().resolve()
    if not intersteps_dir.exists():
        raise FileNotFoundError(f"INTERSTEPS dir not found: {intersteps_dir}")
    if not mesh_path.exists():
        raise FileNotFoundError(f"param_mesh.csv not found: {mesh_path}")

    # Recommended workflow:
    # 1) Generate a batch of mesh rows (STEP_0) with shared columns.
    # 2) Run STEP_1/2 once, then reuse those runs for STEP_3+.
    # 3) Mark mesh rows done in STEP_FINAL.
    # 4) Run this script to remove SIM_RUNs whose step IDs are no longer present in active rows.
    mesh = pd.read_csv(mesh_path)
    if "done" not in mesh.columns:
        print("No done column found; nothing to delete.")
        return

    removed = 0
    skipped = 0
    unknown = 0
    for step_dir in iter_step_dirs(intersteps_dir):
        step_index = parse_step_index(step_dir)
        if step_index is None:
            unknown += 1
            print(f"SKIP (unrecognized step dir): {step_dir}")
            continue
        for sim_run_dir in sorted(step_dir.glob("SIM_RUN_*")):
            meta = find_metadata(sim_run_dir)
            if not meta:
                unknown += 1
                print(f"SKIP (no metadata): {sim_run_dir}")
                continue
            step_chain = extract_step_id_chain(meta)
            if not step_chain:
                unknown += 1
                print(f"SKIP (no step ids): {sim_run_dir}")
                continue
            combo_done = combo_done_in_mesh(mesh, step_chain, step_index)
            if combo_done is None:
                unknown += 1
                print(f"SKIP (no mesh match): {sim_run_dir}")
                continue
            if not combo_done:
                skipped += 1
                continue
            if args.apply:
                shutil.rmtree(sim_run_dir)
                removed += 1
                print(f"DELETED: {sim_run_dir}")
            else:
                print(f"DRY-RUN DELETE: {sim_run_dir}")

    print(f"Summary: removed={removed}, skipped={skipped}, unknown={unknown}")


if __name__ == "__main__":
    main()
