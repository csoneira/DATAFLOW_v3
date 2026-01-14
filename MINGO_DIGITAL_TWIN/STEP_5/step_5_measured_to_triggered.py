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


def compute_tdiff_qdiff(df: pd.DataFrame, c_mm_per_ns: float, qdiff_frac: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    x_to_time_factor = 3.0 / (2.0 * c_mm_per_ns)

    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            x_col = f"X_mea_{plane_idx}_s{strip_idx}"
            qsum_col = f"Y_mea_{plane_idx}_s{strip_idx}"
            tsum_col = f"T_sum_meas_{plane_idx}_s{strip_idx}"
            if x_col not in out.columns or qsum_col not in out.columns:
                continue

            x_vals = out[x_col].to_numpy(dtype=float)
            qsum_vals = out[qsum_col].to_numpy(dtype=float)
            tdiff = x_vals * x_to_time_factor
            qdiff = np.zeros(len(out), dtype=float)
            hit_mask = qsum_vals > 0
            if hit_mask.any():
                qdiff[hit_mask] = rng.normal(0.0, qdiff_frac * qsum_vals[hit_mask], size=hit_mask.sum())

            out[f"T_diff_{plane_idx}_s{strip_idx}"] = tdiff
            out[f"q_diff_{plane_idx}_s{strip_idx}"] = qdiff

            if tsum_col in out.columns:
                out[tsum_col] = out[tsum_col]

    return out


def plot_signal_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        qsum_cols = [c for c in df.columns if c.startswith("Y_mea_") and "_s" in c]
        qsum_vals = df[qsum_cols].to_numpy(dtype=float).ravel() if qsum_cols else np.array([])
        qsum_vals = qsum_vals[qsum_vals > 0]
        axes[0].hist(qsum_vals, bins=60, color="steelblue", alpha=0.8)
        axes[0].set_title("qsum (all strips)")
        axes[0].set_xlabel("qsum")

        qdiff_cols = [c for c in df.columns if c.startswith("q_diff_")]
        qdiff_vals = df[qdiff_cols].to_numpy(dtype=float).ravel() if qdiff_cols else np.array([])
        qdiff_vals = qdiff_vals[qdiff_vals != 0]
        axes[1].hist(qdiff_vals, bins=60, color="seagreen", alpha=0.8)
        axes[1].set_title("q_diff (all strips)")
        axes[1].set_xlabel("q_diff")

        tdiff_cols = [c for c in df.columns if c.startswith("T_diff_")]
        tdiff_vals = df[tdiff_cols].to_numpy(dtype=float).ravel() if tdiff_cols else np.array([])
        tdiff_vals = tdiff_vals[~np.isnan(tdiff_vals)]
        axes[2].hist(tdiff_vals, bins=60, color="darkorange", alpha=0.8)
        axes[2].set_title("T_diff (all strips)")
        axes[2].set_xlabel("T_diff (ns)")

        tsum_cols = [c for c in df.columns if c.startswith("T_sum_meas_")]
        tsum_vals = df[tsum_cols].to_numpy(dtype=float).ravel() if tsum_cols else np.array([])
        tsum_vals = tsum_vals[~np.isnan(tsum_vals)]
        axes[3].hist(tsum_vals, bins=60, color="slateblue", alpha=0.8)
        axes[3].set_title("T_sum_meas (all strips)")
        axes[3].set_xlabel("T_sum_meas (ns)")

        for ax in axes:
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                col = f"T_diff_{plane_idx}_s{strip_idx}"
                if col not in df.columns:
                    ax.axis("off")
                    continue
                vals = df[col].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                ax.hist(vals, bins=80, color="darkorange", alpha=0.8)
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("T_diff (ns)")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                col = f"T_sum_meas_{plane_idx}_s{strip_idx}"
                if col not in df.columns:
                    ax.axis("off")
                    continue
                vals = df[col].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                ax.hist(vals, bins=80, color="slateblue", alpha=0.8)
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("T_sum_meas (ns)")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                col = f"Y_mea_{plane_idx}_s{strip_idx}"
                if col not in df.columns:
                    ax.axis("off")
                    continue
                vals = df[col].to_numpy(dtype=float)
                vals = vals[vals > 0]
                ax.hist(vals, bins=80, color="seagreen", alpha=0.8)
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("qsum")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                col = f"q_diff_{plane_idx}_s{strip_idx}"
                if col not in df.columns:
                    ax.axis("off")
                    continue
                vals = df[col].to_numpy(dtype=float)
                vals = vals[vals != 0]
                ax.hist(vals, bins=80, color="steelblue", alpha=0.8)
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("q_diff")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: compute T_diff and q_diff.")
    parser.add_argument("--config", default="config_step_5.yaml", help="Path to step config YAML")
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
    c_mm_per_ns = float(cfg.get("c_mm_per_ns", 299.792458))
    qdiff_frac = float(cfg.get("qdiff_frac", 0.01))
    rng = np.random.default_rng(cfg.get("seed"))

    input_glob = cfg.get("input_glob", "**/geom_*_hit.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is None:
        raise ValueError("config geometry_id is required for Step 5.")
    geometry_id = int(geometry_id)
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 5 starting...")
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
    input_paths = [p for p in input_paths if p.stem == f"{geom_key}_hit"]
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    print(f"Processing: {input_path}")
    df, upstream_meta = load_with_metadata(input_path)
    out = compute_tdiff_qdiff(df, c_mm_per_ns, qdiff_frac, rng)

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_5", config_path, cfg, upstream_meta
    )
    reset_dir(sim_run_dir)

    out_name = input_path.stem.replace("_hit", "") + f"_signal.{output_format}"
    out_path = sim_run_dir / out_name
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_5",
        "config": cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
    }
    save_with_metadata(out, out_path, metadata, output_format)
    plot_path = sim_run_dir / f"{out_path.stem}_summary.pdf"
    plot_signal_summary(out, plot_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
