#!/usr/bin/env python3
"""STEP_1: Build param_metadata_dictionary.csv for a task."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config" / "pipeline_config.json"
DEFAULT_OUT = BASE_DIR / "output"


def _plot_histogram(df: pd.DataFrame, column: str, plot_path: Path) -> None:
    series = pd.to_numeric(df.get(column), errors="coerce").dropna()
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(series, bins=40, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {column}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, plot_path: Path) -> None:
    x = pd.to_numeric(df.get(x_col), errors="coerce")
    y = pd.to_numeric(df.get(y_col), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[mask], y[mask], s=10, alpha=0.5, color="#F58518")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dictionary CSV for a task.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip quick sanity plots for the output CSV.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"task_{args.task_id:02d}" / "param_metadata_dictionary.csv"
    builder_script = BASE_DIR / "STEP_1_BUILD" / "build_param_metadata_dictionary.py"
    if not builder_script.exists():
        raise FileNotFoundError(
            "Could not find dictionary builder script at "
            "'MASTER_VS_SIMULATION/STEP_1_DICTIONARY/STEP_1_BUILD/'."
        )

    cmd = [
        sys.executable,
        str(builder_script),
        "--config",
        str(args.config),
        "--task-id",
        str(args.task_id),
        "--out",
        str(out_csv),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)

    if not args.no_plots and out_csv.exists():
        df = pd.read_csv(out_csv, low_memory=False)
        plot_dir = out_csv.parent / "plots"
        if plot_dir.exists():
            for path in plot_dir.glob("*.png"):
                path.unlink()
        plot_dir.mkdir(parents=True, exist_ok=True)
        for col in ("flux_cm2_min", "cos_n", "z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"):
            _plot_histogram(df, col, plot_dir / f"hist_{col}.png")
        _plot_scatter(df, "flux_cm2_min", "cos_n", plot_dir / "scatter_flux_vs_cos_n.png")
        _plot_scatter(df, "flux_cm2_min", "z_plane_1", plot_dir / "scatter_flux_vs_z1.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
