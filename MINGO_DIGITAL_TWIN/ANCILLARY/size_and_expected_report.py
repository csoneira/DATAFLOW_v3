#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import time

import numpy as np
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


def estimate_one_line_bytes(
    dirs_now: int,
    size_now: int,
    exp_dirs: int,
    size_expected: int,
) -> int:
    if dirs_now > 0:
        return int(size_now / dirs_now)
    if exp_dirs > 0 and size_expected > 0:
        return int(size_expected / exp_dirs)
    return 0


def format_age(seconds: int) -> str:
    seconds = max(0, int(seconds))
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m{sec:02d}s"
    if mins > 0:
        return f"{mins}m{sec:02d}s"
    return f"{sec}s"


def newest_age_seconds(paths: Iterable[Path], now_ts: float) -> int | None:
    mtimes: list[float] = []
    for path in paths:
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    if not mtimes:
        return None
    newest = max(mtimes)
    return int(max(0.0, now_ts - newest))


def tail_lines(path: Path, max_lines: int) -> list[str]:
    if not path.exists() or max_lines <= 0:
        return []
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = data.splitlines()
    if max_lines >= len(lines):
        return lines
    return lines[-max_lines:]


def health_snapshot(
    dataflow_root: Path,
    now_ts: float,
    tail_n: int,
) -> tuple[list[str], int, str | None]:
    logs_root = dataflow_root / "OPERATIONS_RUNTIME" / "CRON_LOGS"
    step1_logs = sorted(
        (logs_root / "MAIN_ANALYSIS" / "STAGE_1" / "EVENT_DATA" / "STEP_1").glob("guide_raw_to_corrected_*.log")
    )
    # stale_after_seconds is tuned to each job cadence.
    log_groups: list[tuple[str, list[Path], int]] = [
        ("sim_cycle", [logs_root / "SIMULATION" / "RUN" / "sim_main_pipeline_cycle.log"], 180),
        ("step1_all", step1_logs, 180),
        ("step2_all", [logs_root / "MAIN_ANALYSIS" / "STAGE_1" / "EVENT_DATA" / "STEP_2" / "guide_corrected_to_accumulated_all.log"], 180),
        ("step3_all", [logs_root / "MAIN_ANALYSIS" / "STAGE_1" / "EVENT_DATA" / "STEP_3" / "guide_accumulated_to_joined_all.log"], 180),
        ("stale_locks", [logs_root / "ANCILLARY" / "OPERATIONS" / "SOLVE_STALE_LOCKS" / "solve_stale_locks.cron.log"], 420),
        ("watchdog", [logs_root / "ANCILLARY" / "OPERATIONS" / "WATCHDOG_PROCESS_COUNTS" / "watchdog_process_counts.log"], 420),
    ]

    lines: list[str] = []
    for label, paths, stale_after in log_groups:
        age = newest_age_seconds(paths, now_ts)
        if age is None:
            lines.append(f"  {label:<11}: missing")
            continue
        status = "OK" if age < stale_after else "STALE"
        lines.append(f"  {label:<11}: age={format_age(age):>8} status={status}")

    err_patterns = (
        "traceback",
        "exception",
        "error",
        "failed",
        "filenotfounderror",
        "no such file or directory",
    )
    error_hits = 0
    latest_error: str | None = None
    scan_paths: list[Path] = []
    for _, paths, _ in log_groups:
        scan_paths.extend(paths)
    seen: set[Path] = set()
    for path in scan_paths:
        if path in seen:
            continue
        seen.add(path)
        for raw in reversed(tail_lines(path, tail_n)):
            lowered = raw.lower()
            if any(token in lowered for token in err_patterns):
                error_hits += 1
                if latest_error is None:
                    latest_error = f"{path.name}: {raw.strip()}"

    return lines, error_hits, latest_error


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
    parser.add_argument(
        "--z-positions",
        default=None,
        help=(
            "Comma-separated z positions to filter param_mesh (matches z_p1..z_pN). "
            "Provide 1–4 values; only the first N z_p columns are matched."
        ),
    )
    parser.add_argument(
        "--z-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance when matching z positions (used with --z-positions).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress and per-step timings.",
    )
    parser.add_argument(
        "--health-always",
        action="store_true",
        help="Print the log health snapshot on every run.",
    )
    parser.add_argument(
        "--health-interval-seconds",
        type=int,
        default=30,
        help=(
            "Print health snapshot only when epoch time is a multiple of this interval. "
            "Set 0 to disable periodic snapshots."
        ),
    )
    parser.add_argument(
        "--health-tail-lines",
        type=int,
        default=120,
        help="Number of tail lines per log to scan for recent error-like messages.",
    )
    args = parser.parse_args()

    intersteps_dir = Path(args.intersteps_dir).expanduser().resolve()
    mesh_path = Path(args.param_mesh).expanduser().resolve()
    if not intersteps_dir.exists():
        raise FileNotFoundError(f"INTERSTEPS dir not found: {intersteps_dir}")
    if not mesh_path.exists():
        raise FileNotFoundError(f"param_mesh.csv not found: {mesh_path}")

    mesh = load_mesh(mesh_path, args.include_done)

    # load full param_mesh (always use for computing expected counts when z filter is used)
    full_mesh = pd.read_csv(mesh_path)

    # optional filter by z positions (matches z_p1..z_pN)
    if args.z_positions is not None:
        z_vals = [v.strip() for v in str(args.z_positions).split(",") if v.strip() != ""]
        try:
            z_vals = [float(v) for v in z_vals]
        except ValueError:
            raise ValueError("--z-positions must be comma-separated numeric values")
        z_cols = [c for c in ("z_p1", "z_p2", "z_p3", "z_p4") if c in full_mesh.columns]
        if len(z_vals) > len(z_cols):
            raise ValueError("More z values provided than z_p columns available in param_mesh.csv")

        def _z_mask(df: pd.DataFrame) -> pd.Series:
            m = pd.Series(True, index=df.index)
            for i, z in enumerate(z_vals):
                col = z_cols[i]
                m &= np.isclose(pd.to_numeric(df[col], errors="coerce").astype(float), float(z), atol=args.z_tol, rtol=0)
            return m

        full_mask = _z_mask(full_mesh)
        if args.verbose:
            print(f"Filtered full param_mesh by z positions ({', '.join(map(str, z_vals))}) -> {full_mask.sum()} rows match", flush=True)
        # expected counts should be computed from the full param_mesh subset (so %done is correct for that z set)
        expected = expected_counts_from_mesh(full_mesh[full_mask].copy())

        # also restrict the working 'mesh' (which may already be filtered by done) to the same z subset
        mesh = mesh[_z_mask(mesh)].copy()
    else:
        # attempt simple auto-detection: if pending rows (not done) all share a single
        # fully-specified z_p1..z_pN combination, treat report as filtered for that z set
        expected = None
        z_cols = [c for c in ("z_p1", "z_p2", "z_p3", "z_p4") if c in mesh.columns]
        pending = mesh[mesh.get("done", pd.Series(0, index=mesh.index)) != 1]
        if not pending.empty and z_cols:
            # consider only rows with all z columns non-null
            pending_full_z = pending.dropna(subset=z_cols)
            if not pending_full_z.empty:
                unique_z = pending_full_z[z_cols].drop_duplicates()
                if len(unique_z) == 1:
                    # auto-apply this z selection to compute expected counts
                    z_vals = [float(unique_z.iloc[0][c]) for c in z_cols]
                    full_mask = pd.Series(True, index=full_mesh.index)
                    for i, z in enumerate(z_vals):
                        col = z_cols[i]
                        full_mask &= np.isclose(pd.to_numeric(full_mesh[col], errors="coerce").astype(float), z, atol=args.z_tol, rtol=0)
                    if args.verbose:
                        print(f"Auto-detected single active z configuration: {z_vals[:len(z_cols)]}; using it for expected counts", flush=True)
                    expected = expected_counts_from_mesh(full_mesh[full_mask].copy())
                    # restrict working mesh as well to same z subset
                    mesh_mask = full_mask.reindex(mesh.index, fill_value=False)
                    mesh = mesh[mesh_mask].copy()
        if expected is None:
            expected = expected_counts_from_mesh(mesh)

    step_dirs = sorted(intersteps_dir.glob("STEP_*_TO_*"))

    rows = []
    total_dirs = 0
    total_expected_dirs = 0
    total_inputs = 0
    total_bytes = 0
    total_expected_bytes = 0
    total_one_line_bytes = 0

    for step_dir in step_dirs:
        step_name = step_dir.name

        # print(step_name)
        if "0_TO_1" in step_name or "10_TO_FINAL" in step_name:
            continue

        if args.verbose:
            print(f"Scanning {step_name} ...", flush=True)
            t0 = time.perf_counter()
        dirs_now, size_now = count_dirs_and_size(step_dir)
        if args.verbose:
            elapsed = time.perf_counter() - t0
            print(f"  {step_name}: scanned {dirs_now} dirs, {fmt_gb(size_now)} GB in {elapsed:.2f}s", flush=True)
        step_num = int(step_name.split("_")[1])
        exp_dirs = expected.get(step_name, 0)
        available_inputs = available_inputs_for_step(step_num, intersteps_dir, expected)
        if dirs_now > 0:
            avg_bytes = size_now / dirs_now
            size_expected = int(avg_bytes * exp_dirs)
        else:
            size_expected = 0
        one_line_bytes = estimate_one_line_bytes(dirs_now, size_now, exp_dirs, size_expected)
        pct = (dirs_now / exp_dirs * 100.0) if exp_dirs else 0.0
        rows.append(
            (
                step_name,
                dirs_now,
                exp_dirs,
                available_inputs,
                size_now,
                size_expected,
                one_line_bytes,
                pct,
            )
        )

        total_dirs += dirs_now
        total_expected_dirs += exp_dirs
        total_inputs += available_inputs
        total_bytes += size_now
        total_expected_bytes += size_expected
        total_one_line_bytes += one_line_bytes

    width_step = max(len(row[0]) for row in rows + [("TOTAL", 0, 0, 0, 0, 0, 0, 0.0)])
    width_dirs = max(
        len(f"{row[1]}/{row[2]}")
        for row in rows + [("TOTAL", total_dirs, total_expected_dirs, 0, 0, 0, 0, 0.0)]
    )
    width_inputs = max(
        len(str(row[3]))
        for row in rows + [("TOTAL", 0, 0, total_inputs, 0, 0, 0, 0.0)]
    )
    width_size = max(
        len(f"{fmt_gb(row[4])}/{fmt_gb(row[5])}")
        for row in rows + [("TOTAL", 0, 0, 0, total_bytes, total_expected_bytes, 0, 0.0)]
    )
    width_line = max(
        len(fmt_gb(row[6]))
        for row in rows + [("TOTAL", 0, 0, 0, 0, 0, total_one_line_bytes, 0.0)]
    )
    width_step = max(width_step, len("STEP"))
    width_dirs = max(width_dirs, len("DIRS NOW/EXP"))
    width_inputs = max(width_inputs, len("INPUTS"))
    width_size = max(width_size, len("SIZE GB NOW/EXP"))
    width_line = max(width_line, len("ONE LINE GB"))
    width_pct = max(7, len("% DONE"))

    header = (
        f"{'STEP':<{width_step}} | {'DIRS NOW/EXP':>{width_dirs}} | {'INPUTS':>{width_inputs}} | "
        f"{'SIZE GB NOW/EXP':>{width_size}} | {'ONE LINE GB':>{width_line}} | {'% DONE':>{width_pct}}"
    )
    sep = (
        f"{'-' * width_step}-+-{'-' * width_dirs}-+-{'-' * width_inputs}-+-{'-' * width_size}-+-{'-' * width_line}-+-{'-' * width_pct}"
    )

    # print("SIM_RUN utilization summary")
    print(sep)
    print(header)
    print(sep)
    for step_name, dirs_now, exp_dirs, available_inputs, size_now, size_expected, one_line_bytes, pct in rows:
        dirs_cell = f"{dirs_now}/{exp_dirs}"
        size_cell = f"{fmt_gb(size_now)}/{fmt_gb(size_expected)}"
        one_line_cell = fmt_gb(one_line_bytes)
        print(
            f"{step_name:<{width_step}} | {dirs_cell:>{width_dirs}} | {available_inputs:>{width_inputs}} | "
            f"{size_cell:>{width_size}} | {one_line_cell:>{width_line}} | {pct:>{width_pct}.1f}%"
        )
    total_pct = (total_dirs / total_expected_dirs * 100.0) if total_expected_dirs else 0.0
    total_dirs_cell = f"{total_dirs}/{total_expected_dirs}"
    total_size_cell = f"{fmt_gb(total_bytes)}/{fmt_gb(total_expected_bytes)}"
    total_line_cell = fmt_gb(total_one_line_bytes)
    print(sep)
    print(
        f"{'TOTAL':<{width_step}} | {total_dirs_cell:>{width_dirs}} | {total_inputs:>{width_inputs}} | "
        f"{total_size_cell:>{width_size}} | {total_line_cell:>{width_line}} | {total_pct:>{width_pct}.1f}%"
    )
    print(sep)

    now_ts = time.time()
    show_health = args.health_always
    if args.health_interval_seconds > 0 and int(now_ts) % args.health_interval_seconds == 0:
        show_health = True
    if show_health:
        dataflow_root = intersteps_dir.parent.parent
        health_lines, error_hits, latest_error = health_snapshot(
            dataflow_root=dataflow_root,
            now_ts=now_ts,
            tail_n=args.health_tail_lines,
        )
        print()
        print(f"HEALTH SNAPSHOT @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts))}")
        for line in health_lines:
            print(line)
        print(f"  recent_error_hits_in_tails={error_hits}")
        if latest_error:
            print(f"  latest_error_line={latest_error}")


if __name__ == "__main__":
    main()
