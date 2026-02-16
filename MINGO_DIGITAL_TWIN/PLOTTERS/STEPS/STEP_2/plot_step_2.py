#!/usr/bin/env python3
"""Plots for STEP 2 — adapted from MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py
"""
from __future__ import annotations

from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_geometry_summary(df: pd.DataFrame, pdf: PdfPages, title: str = "Geometry") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plane_cols = [(f"X_gen_{i}", f"Y_gen_{i}") for i in range(1, 5)]
    for ax, (x_col, y_col) in zip(axes.flatten(), plane_cols):
        if x_col not in df or y_col not in df:
            ax.axis("off")
            continue
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        ax.scatter(x[mask], y[mask], s=1, alpha=0.2, rasterized=True)
        ax.set_title(f"{x_col} vs {y_col}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    fig.suptitle(title)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # Per-`tt_crossing` diagnostic pages (copied from MASTER STEP_2)
    # Only plot per-tt Theta/Phi pages when the angle columns are present.
    if "tt_crossing" in df.columns and "Theta_gen" in df.columns and "Phi_gen" in df.columns:
        crossing_values = pd.Series(df["tt_crossing"]).dropna().astype(str)
        for ct in sorted(crossing_values.unique()):
            ct_df = df[df["tt_crossing"] == ct]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(ct_df["Theta_gen"], ct_df["Phi_gen"], s=6, alpha=0.25, rasterized=True)
            axes[0].set_title(f"Theta vs Phi (tt_crossing={ct})")
            axes[0].set_xlabel("Theta (rad)")
            axes[0].set_ylabel("Phi (rad)")
            axes[0].set_xlim(0, np.pi / 2)
            axes[0].set_ylim(-np.pi, np.pi)

            axes[1].hist2d(ct_df["Theta_gen"], ct_df["Phi_gen"], bins=60, cmap="magma")
            axes[1].set_title(f"Theta vs Phi density (tt_crossing={ct})")
            axes[1].set_xlabel("Theta (rad)")
            axes[1].set_ylabel("Phi (rad)")
            axes[1].set_xlim(0, np.pi / 2)
            axes[1].set_ylim(-np.pi, np.pi)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    else:
        # angles not available in this chunk — skip per-tt Theta/Phi pages
        pass

def plot_step2_summary(df: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    x_cols = [c for c in df.columns if c.startswith("X_gen_")]
    x_vals = df[x_cols].to_numpy(dtype=float).ravel() if x_cols else np.array([])
    x_vals = x_vals[~np.isnan(x_vals)]
    axes[0].hist(x_vals, bins=80, color="steelblue", alpha=0.8)
    axes[0].set_title("X_gen_i")

    y_cols = [c for c in df.columns if c.startswith("Y_gen_")]
    y_vals = df[y_cols].to_numpy(dtype=float).ravel() if y_cols else np.array([])
    y_vals = y_vals[~np.isnan(y_vals)]
    axes[1].hist(y_vals, bins=80, color="seagreen", alpha=0.8)
    axes[1].set_title("Y_gen_i")

    t_cols = [c for c in df.columns if c.startswith("T_sum_") and c.endswith("_ns")]
    t_vals = df[t_cols].to_numpy(dtype=float).ravel() if t_cols else np.array([])
    t_vals = t_vals[~np.isnan(t_vals)]
    t_vals = t_vals[t_vals != 0]
    axes[2].hist(t_vals, bins=80, color="darkorange", alpha=0.8)
    axes[2].set_title("T_sum_i_ns")

    counts = df.get("tt_crossing")
    if counts is not None:
        counts = pd.Series(counts).astype("string").fillna("")
        vc = counts.value_counts().sort_index()
        bars = axes[3].bar(vc.index.astype(str), vc.values, color="slateblue", alpha=0.8)
        for patch in bars:
            patch.set_rasterized(True)
    else:
        axes[3].axis("off")

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# helpers to find/load a chunk from INTERSTEPS ---------------------------------

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
        try:
            return pd.read_pickle(path)
        except Exception:
            with open(path, "rb") as fh:
                return pickle.load(fh)
    raise ValueError("Unsupported chunk type")


def main() -> None:
    step = 2
    sample = find_any_chunk_for_step(step)
    if sample is None:
        print("No sample chunk found for STEP 2; exiting.")
        return
    print(f"Using sample: {sample}")
    df = load_df(sample)
    out_dir = Path(__file__).resolve().parent / "PLOTS"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"step_{step}_sample_plots.pdf"
    with PdfPages(out_path) as pdf:
        plot_geometry_summary(df, pdf, title=sample.stem)
        plot_step2_summary(df, pdf)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
