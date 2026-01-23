#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_mesh(mesh_path: Path, include_done: bool) -> pd.DataFrame:
    mesh = pd.read_csv(mesh_path)
    if "done" in mesh.columns and not include_done:
        mesh = mesh[mesh["done"].fillna(0).astype(int) != 1]
    return mesh


def normalize_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    try:
        return f"{int(float(value)):03d}"
    except (TypeError, ValueError):
        return str(value)


def expected_counts_from_mesh(mesh: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in range(1, 11):
        cols = [f"step_{idx}_id" for idx in range(1, step + 1)]
        if not all(col in mesh.columns for col in cols):
            counts[f"STEP_{step}_TO_{step + 1}"] = 0
            continue
        combos = mesh[cols].apply(lambda col: col.map(normalize_id)).drop_duplicates()
        counts[f"STEP_{step}_TO_{step + 1}"] = len(combos)
    return counts


def available_inputs_for_step(step: int, intersteps_dir: Path, expected: dict[str, int]) -> int:
    if step == 1:
        return expected.get("STEP_1_TO_2", 0)
    input_dir = intersteps_dir / f"STEP_{step - 1}_TO_{step}"
    if not input_dir.exists():
        return 0
    return len(list(input_dir.glob("SIM_RUN_*")))


def count_dirs_and_size(step_dir: Path) -> tuple[int, int]:
    dirs = list(step_dir.glob("SIM_RUN_*"))
    total_bytes = 0
    for dir_path in dirs:
        for item in dir_path.rglob("*"):
            if item.is_file():
                total_bytes += item.stat().st_size
    return len(dirs), total_bytes


def fmt_gb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined size/expected SIM_RUN report.")
    parser.add_argument(
        "--intersteps-dir",
        default="~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS",
        help="INTERSTEPS base directory.",
    )
    parser.add_argument(
        "--param-mesh",
        default="~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1/param_mesh.csv",
        help="Path to param_mesh.csv.",
    )
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Include done rows in expected counts.",
    )
    args = parser.parse_args()

    intersteps_dir = Path(args.intersteps_dir).expanduser().resolve()
    mesh_path = Path(args.param_mesh).expanduser().resolve()
    if not intersteps_dir.exists():
        raise FileNotFoundError(f"INTERSTEPS dir not found: {intersteps_dir}")
    if not mesh_path.exists():
        raise FileNotFoundError(f"param_mesh.csv not found: {mesh_path}")

    mesh = load_mesh(mesh_path, args.include_done)
    expected = expected_counts_from_mesh(mesh)
    step_dirs = sorted(intersteps_dir.glob("STEP_*_TO_*"))

    rows = []
    total_dirs = 0
    total_expected_dirs = 0
    total_inputs = 0
    total_bytes = 0
    total_expected_bytes = 0

    for step_dir in step_dirs:
        step_name = step_dir.name

        # print(step_name)
        if "0_TO_1" in step_name or "10_TO_FINAL" in step_name:
            continue

        dirs_now, size_now = count_dirs_and_size(step_dir)
        step_num = int(step_name.split("_")[1])
        exp_dirs = expected.get(step_name, 0)
        available_inputs = available_inputs_for_step(step_num, intersteps_dir, expected)
        if dirs_now > 0:
            avg_bytes = size_now / dirs_now
            size_expected = int(avg_bytes * exp_dirs)
        else:
            size_expected = 0
        pct = (dirs_now / exp_dirs * 100.0) if exp_dirs else 0.0
        rows.append((step_name, dirs_now, exp_dirs, available_inputs, size_now, size_expected, pct))

        total_dirs += dirs_now
        total_expected_dirs += exp_dirs
        total_inputs += available_inputs
        total_bytes += size_now
        total_expected_bytes += size_expected

    width_step = max(len(row[0]) for row in rows + [("TOTAL", 0, 0, 0, 0, 0, 0.0)])
    width_dirs = max(len(f"{row[1]}/{row[2]}") for row in rows + [("TOTAL", total_dirs, total_expected_dirs, 0, 0, 0, 0.0)])
    width_inputs = max(len(str(row[3])) for row in rows + [("TOTAL", 0, 0, total_inputs, 0, 0, 0.0)])
    width_size = max(len(f"{fmt_gb(row[4])}/{fmt_gb(row[5])}") for row in rows + [("TOTAL", 0, 0, 0, total_bytes, total_expected_bytes, 0.0)])
    width_pct = 7

    header = (
        f"{'STEP':<{width_step}} | {'DIRS NOW/EXP':>{width_dirs}} | {'INPUTS':>{width_inputs}} | "
        f"{'SIZE GB NOW/EXP':>{width_size}} | {'% DONE':>{width_pct}}"
    )
    sep = (
        f"{'-' * width_step}-+-{'-' * width_dirs}-+-{'-' * width_inputs}-+-{'-' * width_size}-+-{'-' * width_pct}"
    )

    # print("SIM_RUN utilization summary")
    print(sep)
    print(header)
    print(sep)
    for step_name, dirs_now, exp_dirs, available_inputs, size_now, size_expected, pct in rows:
        dirs_cell = f"{dirs_now}/{exp_dirs}"
        size_cell = f"{fmt_gb(size_now)}/{fmt_gb(size_expected)}"
        print(
            f"{step_name:<{width_step}} | {dirs_cell:>{width_dirs}} | {available_inputs:>{width_inputs}} | "
            f"{size_cell:>{width_size}} | {pct:>{width_pct}.1f}%"
        )
    total_pct = (total_dirs / total_expected_dirs * 100.0) if total_expected_dirs else 0.0
    total_dirs_cell = f"{total_dirs}/{total_expected_dirs}"
    total_size_cell = f"{fmt_gb(total_bytes)}/{fmt_gb(total_expected_bytes)}"
    print(sep)
    print(
        f"{'TOTAL':<{width_step}} | {total_dirs_cell:>{width_dirs}} | {total_inputs:>{width_inputs}} | "
        f"{total_size_cell:>{width_size}} | {total_pct:>{width_pct}.1f}%"
    )
    print(sep)


if __name__ == "__main__":
    main()
