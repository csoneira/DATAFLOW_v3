#!/usr/bin/env python3
"""Plots for STEP 3 — adapted from MASTER_STEPS/STEP_3/step_3_crossing_to_hit.py
"""
from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def normalize_tt_series(series: pd.Series) -> pd.Series:
    """Normalize tt-style series to string-like tokens (copied from MASTER_STEP_3).

    Converts numeric-like values to strings, strips trailing ".0", and maps
    '0', '0.0', 'nan', '<NA>' to empty string so value_counts() behaves.
    """
    tt = series.astype("string").fillna("")
    tt = tt.str.strip()
    tt = tt.str.replace(r"\.0$", "", regex=True)
    tt = tt.replace({"0": "", "0.0": "", "nan": "", "<NA>": ""})
    return tt


def plot_avalanche_summary(df: pd.DataFrame, pdf: PdfPages) -> None:
    # tt_avalanche counts
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = normalize_tt_series(df.get("tt_avalanche", pd.Series(dtype="string"))).value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values, color="steelblue", alpha=0.8)
    for patch in bars:
        patch.set_rasterized(True)
    ax.set_title("tt_avalanche counts")
    ax.set_xlabel("tt_avalanche")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # Avalanche size per plane (log scale)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        col = f"avalanche_size_electrons_{plane_idx}"
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        ax.hist(vals, bins=120, color="darkorange", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} avalanche size")
        ax.set_xlim(left=0)
        ax.set_yscale("log")
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.suptitle("Avalanche size per plane (log scale)")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # Ionizations per plane (log scale)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        col = f"avalanche_ion_{plane_idx}"
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        ax.hist(vals, bins=80, color="slateblue", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} avalanche_ion")
        ax.set_xlim(left=0)
        ax.set_yscale("log")
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.suptitle("Ionizations per plane (log scale)")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # Avalanche center positions (2D density)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        x_col = f"avalanche_x_{plane_idx}"
        y_col = f"avalanche_y_{plane_idx}"
        if x_col not in df.columns or y_col not in df.columns:
            ax.axis("off")
            continue
        x_vals = df[x_col].to_numpy(dtype=float)
        y_vals = df[y_col].to_numpy(dtype=float)
        mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        ax.hist2d(x_vals[mask], y_vals[mask], bins=60, cmap="viridis")
        ax.set_title(f"Plane {plane_idx} avalanche center")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    fig.suptitle("Avalanche center positions")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # Avalanche size vs ionizations scatter
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        size_col = f"avalanche_size_electrons_{plane_idx}"
        ion_col = f"avalanche_ion_{plane_idx}"
        if size_col not in df.columns or ion_col not in df.columns:
            ax.axis("off")
            continue
        size_vals = df[size_col].to_numpy(dtype=float)
        ion_vals = df[ion_col].to_numpy(dtype=float)
        mask = (size_vals > 0) & (ion_vals > 0)
        ax.scatter(ion_vals[mask], size_vals[mask], s=2, alpha=0.2, rasterized=True)
        ax.set_title(f"Plane {plane_idx} size vs ion")
        ax.set_xlabel("ionizations")
        ax.set_ylabel("avalanche size")
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    fig.suptitle("Avalanche size vs ionizations")
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
        try:
            return pd.read_pickle(path)
        except Exception:
            with open(path, "rb") as fh:
                import pickle
                return pickle.load(fh)
    raise ValueError("Unsupported chunk type")


def main() -> None:
    step = 3
    sample = find_any_chunk_for_step(step)
    if sample is None:
        print("No sample chunk found for STEP 3; exiting.")
        return
    print(f"Using sample: {sample}")
    df = load_df(sample)
    out_dir = Path(__file__).resolve().parent / "PLOTS"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"step_{step}_sample_plots.pdf"
    with PdfPages(out_path) as pdf:
        plot_avalanche_summary(df, pdf)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
