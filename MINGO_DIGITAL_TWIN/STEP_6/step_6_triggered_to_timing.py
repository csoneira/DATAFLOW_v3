#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from STEP_SHARED.sim_utils import (
    ensure_dir,
    latest_sim_run,
    load_with_metadata,
    now_iso,
    resolve_sim_run,
    reset_dir,
    save_with_metadata,
)


def compute_front_back(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            tsum_col = f"T_sum_meas_{plane_idx}_s{strip_idx}"
            tdiff_col = f"T_diff_{plane_idx}_s{strip_idx}"
            qsum_col = f"Y_mea_{plane_idx}_s{strip_idx}"
            qdiff_col = f"q_diff_{plane_idx}_s{strip_idx}"

            if tsum_col not in out.columns or tdiff_col not in out.columns:
                continue
            if qsum_col not in out.columns or qdiff_col not in out.columns:
                continue

            tsum = out[tsum_col].to_numpy(dtype=float)
            tdiff = out[tdiff_col].to_numpy(dtype=float)
            qsum = out[qsum_col].to_numpy(dtype=float)
            qdiff = out[qdiff_col].to_numpy(dtype=float)

            out[f"T_front_{plane_idx}_s{strip_idx}"] = tsum - tdiff
            out[f"T_back_{plane_idx}_s{strip_idx}"] = tsum + tdiff
            out[f"Q_front_{plane_idx}_s{strip_idx}"] = qsum - qdiff
            out[f"Q_back_{plane_idx}_s{strip_idx}"] = qsum + qdiff

    return out


def plot_frontback_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        tfront_cols = [c for c in df.columns if c.startswith("T_front_")]
        tfront_vals = df[tfront_cols].to_numpy(dtype=float).ravel() if tfront_cols else np.array([])
        tfront_vals = tfront_vals[~np.isnan(tfront_vals)]
        axes[0].hist(tfront_vals, bins=60, color="steelblue", alpha=0.8)
        axes[0].set_title("T_front (all strips)")
        axes[0].set_xlabel("T_front (ns)")

        tback_cols = [c for c in df.columns if c.startswith("T_back_")]
        tback_vals = df[tback_cols].to_numpy(dtype=float).ravel() if tback_cols else np.array([])
        tback_vals = tback_vals[~np.isnan(tback_vals)]
        axes[1].hist(tback_vals, bins=60, color="seagreen", alpha=0.8)
        axes[1].set_title("T_back (all strips)")
        axes[1].set_xlabel("T_back (ns)")

        qfront_cols = [c for c in df.columns if c.startswith("Q_front_")]
        qfront_vals = df[qfront_cols].to_numpy(dtype=float).ravel() if qfront_cols else np.array([])
        qfront_vals = qfront_vals[qfront_vals != 0]
        axes[2].hist(qfront_vals, bins=60, color="darkorange", alpha=0.8)
        axes[2].set_title("Q_front (all strips)")
        axes[2].set_xlabel("Q_front")

        qback_cols = [c for c in df.columns if c.startswith("Q_back_")]
        qback_vals = df[qback_cols].to_numpy(dtype=float).ravel() if qback_cols else np.array([])
        qback_vals = qback_vals[qback_vals != 0]
        axes[3].hist(qback_vals, bins=60, color="slateblue", alpha=0.8)
        axes[3].set_title("Q_back (all strips)")
        axes[3].set_xlabel("Q_back")

        for ax in axes:
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: compute front/back times and charges.")
    parser.add_argument("--config", default="config_step_6.yaml", help="Path to step config YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    with config_path.open("r") as handle:
        cfg = yaml.safe_load(handle)

    input_dir = Path(cfg["input_dir"])
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parent / input_dir
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    output_format = str(cfg.get("output_format", "pkl")).lower()

    input_glob = cfg.get("input_glob", "**/geom_*_signal.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is None:
        raise ValueError("config geometry_id is required for Step 6.")
    geometry_id = int(geometry_id)
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 6 starting...")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")

    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)

    input_run_dir = input_dir / str(input_sim_run)
    if "**" in input_glob:
        input_paths = sorted(input_run_dir.rglob(input_glob.replace("**/", "")))
    else:
        input_paths = sorted(input_run_dir.glob(input_glob))
    geom_key = f"geom_{geometry_id}"
    input_paths = [p for p in input_paths if p.stem == f"{geom_key}_signal"]
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    print(f"Processing: {input_path}")
    df, upstream_meta = load_with_metadata(input_path)
    out = compute_front_back(df)

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_6", config_path, cfg, upstream_meta
    )
    reset_dir(sim_run_dir)

    out_name = input_path.stem.replace("_signal", "") + f"_frontback.{output_format}"
    out_path = sim_run_dir / out_name
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_6",
        "config": cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
    }
    save_with_metadata(out, out_path, metadata, output_format)
    plot_path = sim_run_dir / f"{out_path.stem}_summary.pdf"
    plot_frontback_summary(out, plot_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
