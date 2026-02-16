#!/usr/bin/env python3
"""Plots for STEP 1 — adapted from MASTER_STEPS/STEP_1/step_1_blank_to_generated.py

Loads any available `step_1` chunk from INTERSTEPS and produces a small PDF with
representative histograms/scatter plots (copies plotting functions from the master
step file).
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# --- copied/adapted plotting functions (full parity with MASTER) -----------------

def plot_muon_sample(df: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _, _, patches = axes[0, 0].hist(df["X_gen"], bins=100, color="steelblue", alpha=0.8)
    for patch in patches:
        patch.set_rasterized(True)
    axes[0, 0].set_title("X_gen distribution")
    axes[0, 0].set_xlabel("X (mm)")
    axes[0, 0].set_ylabel("Counts")

    _, _, patches = axes[0, 1].hist(df["Y_gen"], bins=100, color="seagreen", alpha=0.8)
    for patch in patches:
        patch.set_rasterized(True)
    axes[0, 1].set_title("Y_gen distribution")
    axes[0, 1].set_xlabel("Y (mm)")

    _, _, patches = axes[1, 0].hist(df["Theta_gen"], bins=100, color="darkorange", alpha=0.8)
    for patch in patches:
        patch.set_rasterized(True)
    axes[1, 0].set_title("Theta_gen distribution")
    axes[1, 0].set_xlabel("Theta (rad)")

    _, _, patches = axes[1, 1].hist(df["Phi_gen"], bins=100, color="slateblue", alpha=0.8)
    for patch in patches:
        patch.set_rasterized(True)
    axes[1, 1].set_title("Phi_gen distribution")
    axes[1, 1].set_xlabel("Phi (rad)")

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["Theta_gen"], df["Phi_gen"], s=1, alpha=0.2, rasterized=True)
    ax.set_title("Theta vs Phi")
    ax.set_xlabel("Theta (rad)")
    ax.set_ylabel("Phi (rad)")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def plot_step1_summary(df: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist2d(df["X_gen"], df["Y_gen"], bins=60, cmap="viridis")
    axes[0, 0].set_title("X_gen vs Y_gen (density)")
    axes[0, 0].set_xlabel("X (mm)")
    axes[0, 0].set_ylabel("Y (mm)")

    h = axes[0, 1].hist2d(df["Theta_gen"], df["Phi_gen"], bins=60, cmap="magma")
    axes[0, 1].set_title("Theta_gen vs Phi_gen (density)")
    axes[0, 1].set_xlabel("Theta (rad)")
    axes[0, 1].set_ylabel("Phi (rad)")
    for quad in h[3].get_paths():
        quad._interpolation_steps = 1

    cos_theta = np.cos(df["Theta_gen"].to_numpy(dtype=float))
    axes[1, 0].hist(cos_theta, bins=80, color="darkorange", alpha=0.8)
    axes[1, 0].set_title("cos(Theta_gen)")
    axes[1, 0].set_xlabel("cos(theta)")

    r = np.hypot(df["X_gen"].to_numpy(dtype=float), df["Y_gen"].to_numpy(dtype=float))
    axes[1, 1].hist(r, bins=80, color="slateblue", alpha=0.8)
    axes[1, 1].set_title("Radial distance")
    axes[1, 1].set_xlabel("sqrt(X^2+Y^2) (mm)")

    for ax in axes.flatten():
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def plot_thick_time_summary(df: pd.DataFrame, pdf: PdfPages, rate_hz: float | None) -> bool:
    if "T_thick_s" not in df.columns:
        return False
    t0_s = df["T_thick_s"].to_numpy(dtype=float)
    t0_s = t0_s[np.isfinite(t0_s)]
    if t0_s.size < 2:
        return False
    t0_s = np.sort(t0_s.astype(int))
    if t0_s.size == 0:
        return False
    sec_min = int(t0_s[0])
    sec_max = int(t0_s[-1])
    counts = np.bincount(t0_s - sec_min)
    seconds = np.arange(sec_min, sec_min + len(counts))
    if len(counts) > 1:
        counts = counts[1:]
        seconds = seconds[1:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(seconds, counts, linewidth=1.0, color="slateblue")
    axes[0].set_title("Counts per second")
    axes[0].set_xlabel("Second")
    axes[0].set_ylabel("Events")

    axes[1].hist(counts, bins=60, color="teal", alpha=0.8)
    axes[1].set_title("Histogram of counts per second")
    axes[1].set_xlabel("Events per second")
    axes[1].set_ylabel("Counts")
    if rate_hz and rate_hz > 0:
        axes[1].axvline(rate_hz, color="black", linestyle="--", linewidth=1.0)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return True


# --- helpers: locate + load a chunk/sample from INTERSTEPS -----------------------

def find_any_chunk_for_step(step: int) -> Path | None:
    root = Path(__file__).resolve().parents[3] / "INTERSTEPS"

    # 1) prefer explicit step_N_chunks/part_*.pkl
    parts = sorted(root.glob(f"**/step_{step}_chunks/part_*.pkl"))
    if parts:
        return parts[0]

    # 2) accept `muon_sample_<N>/chunks/part_*.pkl` (STEP 1 commonly stored this way)
    parts = sorted(root.glob(f"**/muon_sample_*/chunks/part_*.pkl"))
    if parts:
        return parts[0]

    # 3) accept standalone muon_sample_*.pkl files
    parts = sorted(root.glob(f"**/muon_sample_*.pkl"))
    if parts:
        return parts[0]

    # 4) fall back to manifests for either form (step_*_chunks or muon_sample_*.chunks.json)
    manifests = sorted(root.rglob(f"*step_{step}_chunks.chunks.json")) + sorted(root.rglob("**/muon_sample_*.chunks.json"))
    for manifest in manifests:
        try:
            j = json.loads(manifest.read_text())
            parts_list = [p for p in j.get("parts", []) if p.endswith(".pkl")]
            if parts_list:
                return Path(parts_list[0])
        except Exception:
            return manifest
    return None


def load_df(path: Path) -> pd.DataFrame:
    if path is None:
        raise FileNotFoundError("No chunk file found for requested step.")
    if path.name.endswith(".chunks.json"):
        j = json.loads(path.read_text())
        parts = [p for p in j.get("parts", []) if p.endswith(".pkl")]
        if parts:
            path = Path(parts[0])
        else:
            raise FileNotFoundError(f"Manifest {path} contains no .pkl parts")
    if path.suffix == ".pkl":
        try:
            return pd.read_pickle(path)
        except Exception:
            with open(path, "rb") as fh:
                return pickle.load(fh)
    raise ValueError(f"Unsupported chunk file type: {path}")


def main() -> None:
    step = 1
    sample_path = find_any_chunk_for_step(step)
    if sample_path is None:
        print("No sample chunk found for STEP 1 under INTERSTEPS; exiting.")
        return
    print(f"Using sample: {sample_path}")
    try:
        df = load_df(sample_path)
    except Exception as exc:
        print(f"Failed to load sample chunk: {exc}")
        return

    out_dir = Path(__file__).resolve().parent / "PLOTS"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"step_{step}_sample_plots.pdf"
    with PdfPages(out_path) as pdf:
        plot_muon_sample(df, pdf)
        plot_step1_summary(df, pdf)
    print(f"Saved plots to {out_path}")


if __name__ == "__main__":
    main()
