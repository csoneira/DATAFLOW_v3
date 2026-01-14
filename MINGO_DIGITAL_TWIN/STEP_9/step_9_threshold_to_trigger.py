#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

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


def normalize_tt(series: pd.Series) -> pd.Series:
    tt = series.astype("string").fillna("")
    tt = tt.str.strip()
    tt = tt.str.replace(r"\.0$", "", regex=True)
    tt = tt.replace({"0": "", "0.0": "", "nan": "", "<NA>": ""})
    return tt


def passes_trigger(tt_value: str, triggers: List[str]) -> bool:
    for trig in triggers:
        if all(ch in tt_value for ch in trig):
            return True
    return False


def apply_trigger(df: pd.DataFrame, triggers: List[str]) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    tt_array = ["" for _ in range(n)]

    for plane_idx in range(1, 5):
        plane_active = pd.Series(False, index=out.index)
        for strip_idx in range(1, 5):
            qf = out.get(f"Q_front_{plane_idx}_s{strip_idx}")
            qb = out.get(f"Q_back_{plane_idx}_s{strip_idx}")
            if qf is None and qb is None:
                continue
            active = pd.Series(False, index=out.index)
            if qf is not None:
                active |= qf.to_numpy(dtype=float) > 0
            if qb is not None:
                active |= qb.to_numpy(dtype=float) > 0
            plane_active |= active
        tt_array = [tt + str(plane_idx) if active else tt for tt, active in zip(tt_array, plane_active)]

    tt_series = normalize_tt(pd.Series(tt_array))
    keep_mask = tt_series.apply(lambda val: passes_trigger(val, triggers))
    filtered = out[keep_mask].copy()
    filtered["tt_trigger"] = tt_series[keep_mask].values
    return filtered


def plot_trigger_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = normalize_tt(df["tt_trigger"]).value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color="steelblue", alpha=0.8)
        for patch in bars:
            patch.set_rasterized(True)
        ax.set_title("tt_trigger counts")
        ax.set_xlabel("tt_trigger")
        ax.set_ylabel("Counts")
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
    parser = argparse.ArgumentParser(description="Step 9: apply trigger combinations based on channel activity.")
    parser.add_argument("--config", default="config_step_9.yaml", help="Path to step config YAML")
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
    triggers = [str(t) for t in cfg.get("trigger_combinations", [])]

    input_glob = cfg.get("input_glob", "**/geom_*_threshold.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is None:
        raise ValueError("config geometry_id is required for Step 9.")
    geometry_id = int(geometry_id)
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 9 starting...")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Triggers: {triggers}")

    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)

    input_run_dir = input_dir / str(input_sim_run)
    if "**" in input_glob:
        input_paths = sorted(input_run_dir.rglob(input_glob.replace("**/", "")))
    else:
        input_paths = sorted(input_run_dir.glob(input_glob))
    geom_key = f"geom_{geometry_id}"
    input_paths = [p for p in input_paths if p.stem == f"{geom_key}_threshold"]
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    print(f"Processing: {input_path}")
    df, upstream_meta = load_with_metadata(input_path)
    filtered = apply_trigger(df, triggers)

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_9", config_path, cfg, upstream_meta
    )
    reset_dir(sim_run_dir)

    out_name = input_path.stem.replace("_threshold", "") + f"_triggered.{output_format}"
    out_path = sim_run_dir / out_name
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_9",
        "config": cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
    }
    save_with_metadata(filtered, out_path, metadata, output_format)
    plot_path = sim_run_dir / f"{out_path.stem}_summary.pdf"
    plot_trigger_summary(filtered, plot_path)
    print(f"Saved {out_path} (kept {len(filtered):,}/{len(df):,})")


if __name__ == "__main__":
    main()
