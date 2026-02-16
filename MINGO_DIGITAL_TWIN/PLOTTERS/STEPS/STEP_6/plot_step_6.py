#!/usr/bin/env python3
"""Plots for STEP 6 — adapted from MASTER_STEPS/STEP_6/step_6_triggered_to_timing.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_frontback_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        tt_col = None
        for c in ("tt_trigger", "tt_hit", "tt_avalanche", "tt_crossing"):
            if c in df.columns:
                tt_col = c
                break
        if tt_col:
            fig, ax = plt.subplots(figsize=(8, 6))
            counts = df[tt_col].astype("string").fillna("").value_counts().sort_index()
            bars = ax.bar(counts.index.astype(str), counts.values, color="slateblue", alpha=0.8)
            for patch in bars:
                patch.set_rasterized(True)
            ax.set_title(f"{tt_col} counts")
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                tf_col = f"T_front_{plane_idx}_s{strip_idx}"
                tb_col = f"T_back_{plane_idx}_s{strip_idx}"
                if tf_col in df.columns:
                    vals = df[tf_col].to_numpy(dtype=float)
                    vals = vals[~np.isnan(vals)]
                    ax.hist(vals, bins=80, color="steelblue", alpha=0.6)
                if tb_col in df.columns:
                    vals = df[tb_col].to_numpy(dtype=float)
                    vals = vals[~np.isnan(vals)]
                    ax.hist(vals, bins=80, color="darkorange", alpha=0.6)
                ax.set_title(f"P{plane_idx} S{strip_idx}")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # --- Add per-plane 4x4 charge histograms (Q_front / Q_back) copied from MASTER ---
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


def find_any_chunk_for_step(step: int) -> Path | None:
    root = Path(__file__).resolve().parents[3] / "INTERSTEPS"
    parts = sorted(root.glob(f"**/step_{step}_chunks/part_*.pkl"))
    if parts:
        return parts[0]
    manifests = sorted(root.rglob(f"*step_{step}_chunks.chunks.json"))
    if manifests:
        try:
            j = json.loads(manifests[0].read_text())
            parts_list = [p for p in j.get("parts", []) if p.endswith(".pkl")]
            if parts_list:
                return Path(parts_list[0])
        except Exception:
            return manifests[0]
    return None


def load_df(path: Path) -> pd.DataFrame:
    if path.name.endswith(".chunks.json"):
        j = json.loads(path.read_text())
        parts = [p for p in j.get("parts", []) if p.endswith(".pkl")]
        if parts:
            path = Path(parts[0])
        else:
            raise FileNotFoundError("Manifest contains no .pkl parts")
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError("Unsupported chunk type")


def main() -> None:
    step = 6
    sample = find_any_chunk_for_step(step)
    if sample is None:
        print("No sample chunk found for STEP 6; exiting.")
        return
    print(f"Using sample: {sample}")
    df = load_df(sample)
    out_dir = Path(__file__).resolve().parent / "PLOTS"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"step_{step}_sample_plots.pdf"
    plot_frontback_summary(df, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
