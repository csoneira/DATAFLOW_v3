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


def apply_fee(
    df: pd.DataFrame,
    t_fee_sigma_ns: float,
    q_to_time_factor: float,
    qfront_offsets: list[list[float]],
    qback_offsets: list[list[float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            tf_col = f"T_front_{plane_idx}_s{strip_idx}"
            tb_col = f"T_back_{plane_idx}_s{strip_idx}"
            qf_col = f"Q_front_{plane_idx}_s{strip_idx}"
            qb_col = f"Q_back_{plane_idx}_s{strip_idx}"

            if tf_col in out.columns:
                vals = out[tf_col].to_numpy(dtype=float)
                mask = ~np.isnan(vals)
                if mask.any():
                    vals[mask] = vals[mask] + rng.normal(0.0, t_fee_sigma_ns, mask.sum())
                out[tf_col] = vals
            if tb_col in out.columns:
                vals = out[tb_col].to_numpy(dtype=float)
                mask = ~np.isnan(vals)
                if mask.any():
                    vals[mask] = vals[mask] + rng.normal(0.0, t_fee_sigma_ns, mask.sum())
                out[tb_col] = vals

            if qf_col in out.columns:
                vals = out[qf_col].to_numpy(dtype=float)
                mask = vals != 0
                if mask.any():
                    vals[mask] = vals[mask] * q_to_time_factor + float(
                        qfront_offsets[plane_idx - 1][strip_idx - 1]
                    )
                out[qf_col] = vals
            if qb_col in out.columns:
                vals = out[qb_col].to_numpy(dtype=float)
                mask = vals != 0
                if mask.any():
                    vals[mask] = vals[mask] * q_to_time_factor + float(
                        qback_offsets[plane_idx - 1][strip_idx - 1]
                    )
                out[qb_col] = vals
    return out


def apply_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            for prefix in ("Q_front", "Q_back"):
                col = f"{prefix}_{plane_idx}_s{strip_idx}"
                if col not in out.columns:
                    continue
                vals = out[col].to_numpy(dtype=float)
                vals[vals < threshold] = 0.0
                out[col] = vals
    return out


def plot_threshold_summary(
    df: pd.DataFrame,
    output_path: Path,
    threshold: float,
    qfront_offsets: list[list[float]],
    qback_offsets: list[list[float]],
) -> None:
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                qf_col = f"Q_front_{plane_idx}_s{strip_idx}"
                qb_col = f"Q_back_{plane_idx}_s{strip_idx}"
                if qf_col not in df.columns and qb_col not in df.columns:
                    ax.axis("off")
                    continue
                if qf_col in df.columns:
                    vals = df[qf_col].to_numpy(dtype=float)
                    mask = vals != 0
                    if mask.any():
                        vals = vals[mask] - float(qfront_offsets[plane_idx - 1][strip_idx - 1])
                        ax.hist(vals, bins=80, color="steelblue", alpha=0.6, label="front")
                if qb_col in df.columns:
                    vals = df[qb_col].to_numpy(dtype=float)
                    mask = vals != 0
                    if mask.any():
                        vals = vals[mask] - float(qback_offsets[plane_idx - 1][strip_idx - 1])
                        ax.hist(vals, bins=80, color="darkorange", alpha=0.6, label="back")
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("a*x only")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        qfront_cols = [c for c in df.columns if c.startswith("Q_front_")]
        qfront_vals = df[qfront_cols].to_numpy(dtype=float).ravel() if qfront_cols else np.array([])
        axes[0].hist(qfront_vals, bins=60, color="steelblue", alpha=0.8)
        axes[0].axvline(threshold, color="red", linestyle="--", linewidth=1)
        axes[0].set_title("Q_front (thresholded)")
        axes[0].set_xlabel("Q_front")

        qback_cols = [c for c in df.columns if c.startswith("Q_back_")]
        qback_vals = df[qback_cols].to_numpy(dtype=float).ravel() if qback_cols else np.array([])
        axes[1].hist(qback_vals, bins=60, color="seagreen", alpha=0.8)
        axes[1].axvline(threshold, color="red", linestyle="--", linewidth=1)
        axes[1].set_title("Q_back (thresholded)")
        axes[1].set_xlabel("Q_back")

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
                tf_col = f"T_front_{plane_idx}_s{strip_idx}"
                tb_col = f"T_back_{plane_idx}_s{strip_idx}"
                if tf_col not in df.columns and tb_col not in df.columns:
                    ax.axis("off")
                    continue
                if tf_col in df.columns:
                    vals = df[tf_col].to_numpy(dtype=float)
                    vals = vals[(~np.isnan(vals)) & (vals != 0)]
                    ax.hist(vals, bins=80, color="steelblue", alpha=0.6, label="front")
                if tb_col in df.columns:
                    vals = df[tb_col].to_numpy(dtype=float)
                    vals = vals[(~np.isnan(vals)) & (vals != 0)]
                    ax.hist(vals, bins=80, color="darkorange", alpha=0.6, label="back")
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("time (ns)")
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
                qf_col = f"Q_front_{plane_idx}_s{strip_idx}"
                qb_col = f"Q_back_{plane_idx}_s{strip_idx}"
                if qf_col not in df.columns and qb_col not in df.columns:
                    ax.axis("off")
                    continue
                if qf_col in df.columns:
                    vals = df[qf_col].to_numpy(dtype=float)
                    vals = vals[vals != 0]
                    ax.hist(vals, bins=80, color="steelblue", alpha=0.6, label="front")
                if qb_col in df.columns:
                    vals = df[qb_col].to_numpy(dtype=float)
                    vals = vals[vals != 0]
                    ax.hist(vals, bins=80, color="darkorange", alpha=0.6, label="back")
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("charge")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8: apply charge threshold.")
    parser.add_argument("--config", default="config_step_8.yaml", help="Path to step config YAML")
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
    threshold = float(cfg.get("charge_threshold", 0.01))
    t_fee_sigma_ns = float(cfg.get("t_fee_sigma_ns", 0.01))
    q_to_time_factor = float(cfg.get("q_to_time_factor", 1.0e-5))
    qfront_offsets = cfg.get("qfront_offsets", [[0, 0, 0, 0]] * 4)
    qback_offsets = cfg.get("qback_offsets", [[0, 0, 0, 0]] * 4)
    rng = np.random.default_rng(cfg.get("seed"))

    input_glob = cfg.get("input_glob", "**/geom_*_calibrated.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is None:
        raise ValueError("config geometry_id is required for Step 8.")
    geometry_id = int(geometry_id)
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 8 starting...")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"charge_threshold: {threshold}")

    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)

    input_run_dir = input_dir / str(input_sim_run)
    if "**" in input_glob:
        input_paths = sorted(input_run_dir.rglob(input_glob.replace("**/", "")))
    else:
        input_paths = sorted(input_run_dir.glob(input_glob))
    geom_key = f"geom_{geometry_id}"
    input_paths = [p for p in input_paths if p.stem == f"{geom_key}_calibrated"]
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    print(f"Processing: {input_path}")
    df, upstream_meta = load_with_metadata(input_path)
    out = apply_threshold(df, threshold)
    out = apply_fee(out, t_fee_sigma_ns, q_to_time_factor, qfront_offsets, qback_offsets, rng)

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_8", config_path, cfg, upstream_meta
    )
    reset_dir(sim_run_dir)

    out_name = input_path.stem.replace("_calibrated", "") + f"_threshold.{output_format}"
    out_path = sim_run_dir / out_name
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_8",
        "config": cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
    }
    save_with_metadata(out, out_path, metadata, output_format)
    plot_path = sim_run_dir / f"{out_path.stem}_summary.pdf"
    plot_threshold_summary(out, plot_path, threshold, qfront_offsets, qback_offsets)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
