#!/usr/bin/env python3
"""Plots for STEP 8 — adapted from MASTER_STEPS/STEP_8/step_8_uncalibrated_to_threshold.py
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


def plot_threshold_summary(
    df: pd.DataFrame,
    output_path: Path,
    threshold: float = 0.0,
    qfront_offsets: list[list[float]] | None = None,
    qback_offsets: list[list[float]] | None = None,
) -> None:
    if qfront_offsets is None:
        qfront_offsets = [[0.0] * 4 for _ in range(4)]
    if qback_offsets is None:
        qback_offsets = [[0.0] * 4 for _ in range(4)]

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
            ax.set_xlabel(tt_col)
            ax.set_ylabel("Counts")
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        qfront_cols = [c for c in df.columns if c.startswith("Q_front_")]
        qfront_vals = df[qfront_cols].to_numpy(dtype=float).ravel() if qfront_cols else np.array([])
        axes[0].hist(qfront_vals, bins=60, color="steelblue", alpha=0.8)
        axes[0].axvline(threshold, color="red", linestyle="--", linewidth=1)
        axes[0].set_title("Q_front (thresholded)")

        qback_cols = [c for c in df.columns if c.startswith("Q_back_")]
        qback_vals = df[qback_cols].to_numpy(dtype=float).ravel() if qback_cols else np.array([])
        axes[1].hist(qback_vals, bins=60, color="seagreen", alpha=0.8)
        axes[1].axvline(threshold, color="red", linestyle="--", linewidth=1)
        axes[1].set_title("Q_back (thresholded)")

        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # --- Add per-plane 4x4 Q grids with offsets (copied from MASTER) ---
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

        # Per-strip charge grids
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


def find_step_manifest_for_sample(sample_path: Path, step: int) -> Path | None:
    manifest_name = f"step_{step}_chunks.chunks.json"
    if sample_path.name == manifest_name:
        return sample_path
    if sample_path.parent.name == f"step_{step}_chunks":
        candidate = sample_path.parent.parent / manifest_name
        if candidate.exists():
            return candidate
    candidate = sample_path.parent / manifest_name
    if candidate.exists():
        return candidate
    return None


def load_step8_config(sample_path: Path) -> dict:
    manifest = find_step_manifest_for_sample(sample_path, step=8)
    if manifest is None or not manifest.exists():
        return {}
    try:
        j = json.loads(manifest.read_text())
    except Exception:
        return {}
    metadata = j.get("metadata", {}) if isinstance(j, dict) else {}
    cfg = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def find_any_chunk_for_step(step: int) -> Path | None:
    root = Path(__file__).resolve().parents[3] / "INTERSTEPS"
    parts = sorted(root.glob(f"**/step_{step}_chunks/part_*.pkl"))
    if parts:
        return parts[0]
    manifests = sorted(root.rglob(f"*step_{step}_chunks.chunks.json"))
    if manifests:
        try:
            j = json.loads(manifests[0].read_text())
            parts_list = [p for p in (j.get("parts") or j.get("chunks") or []) if str(p).endswith(".pkl")]
            if parts_list:
                return Path(parts_list[0])
        except Exception:
            return manifests[0]
    return None


def load_df(path: Path) -> pd.DataFrame:
    if path.name.endswith(".chunks.json"):
        j = json.loads(path.read_text())
        parts = [p for p in (j.get("parts") or j.get("chunks") or []) if str(p).endswith(".pkl")]
        if parts:
            path = Path(parts[0])
        else:
            raise FileNotFoundError("Manifest contains no .pkl parts")
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError("Unsupported chunk type")


def main() -> None:
    step = 8
    sample = find_any_chunk_for_step(step)
    if sample is None:
        print("No sample chunk found for STEP 8; exiting.")
        return
    print(f"Using sample: {sample}")
    df = load_df(sample)
    cfg = load_step8_config(sample)
    thresh = float(cfg.get("threshold", 0.0))
    qfront_offsets = cfg.get("qfront_offsets", [[0.0] * 4 for _ in range(4)])
    qback_offsets = cfg.get("qback_offsets", [[0.0] * 4 for _ in range(4)])

    out_dir = Path(__file__).resolve().parent / "PLOTS"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"step_{step}_sample_plots.pdf"
    plot_threshold_summary(
        df,
        out_path,
        threshold=thresh,
        qfront_offsets=qfront_offsets,
        qback_offsets=qback_offsets,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
