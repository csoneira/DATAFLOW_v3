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


def apply_jitter(df: pd.DataFrame, jitter_width_ns: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    jitter = rng.uniform(-jitter_width_ns / 2, jitter_width_ns / 2, size=n)

    active_mask = np.zeros(n, dtype=bool)
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            qf_col = f"Q_front_{plane_idx}_s{strip_idx}"
            qb_col = f"Q_back_{plane_idx}_s{strip_idx}"
            if qf_col in out.columns:
                active_mask |= out[qf_col].to_numpy(dtype=float) > 0
            if qb_col in out.columns:
                active_mask |= out[qb_col].to_numpy(dtype=float) > 0

    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            for prefix in ("T_front", "T_back"):
                col = f"{prefix}_{plane_idx}_s{strip_idx}"
                if col not in out.columns:
                    continue
                vals = out[col].to_numpy(dtype=float)
                mask = active_mask & ~np.isnan(vals)
                vals[mask] = vals[mask] + jitter[mask]
                out[col] = vals

    out["daq_jitter_ns"] = np.where(active_mask, jitter, 0.0)
    return out


def plot_jitter_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        jitter_vals = df["daq_jitter_ns"].to_numpy(dtype=float) if "daq_jitter_ns" in df.columns else np.array([])
        axes[0].hist(jitter_vals, bins=60, color="steelblue", alpha=0.8)
        axes[0].set_title("DAQ jitter")
        axes[0].set_xlabel("daq_jitter_ns")

        tfront_cols = [c for c in df.columns if c.startswith("T_front_")]
        tback_cols = [c for c in df.columns if c.startswith("T_back_")]
        tfront_vals = df[tfront_cols].to_numpy(dtype=float).ravel() if tfront_cols else np.array([])
        tback_vals = df[tback_cols].to_numpy(dtype=float).ravel() if tback_cols else np.array([])
        tfront_vals = tfront_vals[~np.isnan(tfront_vals)]
        tback_vals = tback_vals[~np.isnan(tback_vals)]
        axes[1].hist(tfront_vals, bins=60, color="seagreen", alpha=0.6, label="T_front")
        axes[1].hist(tback_vals, bins=60, color="darkorange", alpha=0.6, label="T_back")
        axes[1].set_title("T_front / T_back (jittered)")
        axes[1].set_xlabel("time (ns)")
        axes[1].legend()

        for ax in axes:
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10: apply DAQ jitter to T_front/T_back.")
    parser.add_argument("--config", default="config_step_10.yaml", help="Path to step config YAML")
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
    jitter_width_ns = float(cfg.get("jitter_width_ns", 10.0))
    rng = np.random.default_rng(cfg.get("seed"))

    input_glob = cfg.get("input_glob", "**/geom_*_triggered.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is None:
        raise ValueError("config geometry_id is required for Step 10.")
    geometry_id = int(geometry_id)
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 10 starting...")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"jitter_width_ns: {jitter_width_ns}")

    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)

    input_run_dir = input_dir / str(input_sim_run)
    if "**" in input_glob:
        input_paths = sorted(input_run_dir.rglob(input_glob.replace("**/", "")))
    else:
        input_paths = sorted(input_run_dir.glob(input_glob))
    geom_key = f"geom_{geometry_id}"
    input_paths = [p for p in input_paths if p.stem == f"{geom_key}_triggered"]
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    print(f"Processing: {input_path}")
    df, upstream_meta = load_with_metadata(input_path)
    out = apply_jitter(df, jitter_width_ns, rng)

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_10", config_path, cfg, upstream_meta
    )
    reset_dir(sim_run_dir)

    out_name = input_path.stem.replace("_triggered", "") + f"_daq.{output_format}"
    out_path = sim_run_dir / out_name
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_10",
        "config": cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
    }
    save_with_metadata(out, out_path, metadata, output_format)
    plot_path = sim_run_dir / f"{out_path.stem}_summary.pdf"
    plot_jitter_summary(out, plot_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
