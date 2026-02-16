#!/usr/bin/env python3
"""Plots for STEP 4 — adapted from MASTER_STEPS/STEP_4/step_4_hit_to_measured.py
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


def normalize_tt(series: pd.Series) -> pd.Series:
    tt = series.astype("string").fillna("")
    tt = tt.str.strip()
    tt = tt.str.replace(r"\.0$", "", regex=True)
    tt = tt.replace({"0": "", "0.0": "", "nan": "", "<NA>": ""})
    return tt


def plot_hit_summary(df: pd.DataFrame, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if "tt_hit" in df.columns:
        counts = df["tt_hit"].astype("string").fillna("").value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color="steelblue", alpha=0.8)
        for patch in bars:
            patch.set_rasterized(True)
    ax.set_title("tt_hit counts")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def plot_step4_summary(
    df: pd.DataFrame,
    pdf: PdfPages,
    include_thrown_points: bool = True,
    points_per_plane: int = 2000,
    point_seed: int | None = None,
    thrown_points: dict | None = None,
    examples_df: pd.DataFrame | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    counts = normalize_tt(df.get("tt_hit", pd.Series(dtype="string"))).value_counts().sort_index()
    bars = axes[0].bar(counts.index, counts.values, color="steelblue", alpha=0.8)
    for patch in bars:
        patch.set_rasterized(True)
    axes[0].set_title("tt_hit")

    qsum_cols = [c for c in df.columns if c.startswith("Y_mea_") and "_s" in c]
    qsum_vals = df[qsum_cols].to_numpy(dtype=float).ravel() if qsum_cols else np.array([])
    qsum_vals = qsum_vals[qsum_vals > 0]
    axes[1].hist(qsum_vals, bins=60, color="seagreen", alpha=0.8)
    axes[1].set_title("qsum (all strips)")

    x_cols = [c for c in df.columns if c.startswith("X_mea_") and "_s" in c]
    x_vals = df[x_cols].to_numpy(dtype=float).ravel() if x_cols else np.array([])
    x_vals = x_vals[~np.isnan(x_vals)]
    axes[2].hist(x_vals, bins=60, color="darkorange", alpha=0.8)
    axes[2].set_title("X_mea (all strips)")

    t_cols = [c for c in df.columns if c.startswith("T_sum_meas_") and "_s" in c]
    t_vals = df[t_cols].to_numpy(dtype=float).ravel() if t_cols else np.array([])
    t_vals = t_vals[~np.isnan(t_vals)]
    axes[3].hist(t_vals, bins=60, color="slateblue", alpha=0.8)
    axes[3].set_title("T_sum_meas (all strips)")

    for ax in axes:
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # Per-tt example pages (use df itself as examples source if available)
    if "Theta_gen" in df.columns and "Phi_gen" in df.columns and "tt_hit" in df.columns:
        tt_series = normalize_tt(df["tt_hit"]).dropna().astype(str)
        for tt_value in sorted(tt_series.unique()):
            tt_df = df[df["tt_hit"] == tt_value]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(
                tt_df["Theta_gen"],
                tt_df["Phi_gen"],
                s=6,
                alpha=0.25,
                rasterized=True,
            )
            axes[0].set_title(f"Theta vs Phi (tt_hit={tt_value})")
            axes[0].set_xlabel("Theta (rad)")
            axes[0].set_ylabel("Phi (rad)")
            axes[0].set_xlim(0, np.pi / 2)
            axes[0].set_ylim(-np.pi, np.pi)

            axes[1].hist2d(tt_df["Theta_gen"], tt_df["Phi_gen"], bins=60, cmap="magma")
            axes[1].set_title(f"Theta vs Phi density (tt_hit={tt_value})")
            axes[1].set_xlabel("Theta (rad)")
            axes[1].set_ylabel("Phi (rad)")
            axes[1].set_xlim(0, np.pi / 2)
            axes[1].set_ylim(-np.pi, np.pi)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    strip_cols_all = [col for col in df.columns if col.startswith("Y_mea_") and "_s" in col]
    examples_source = examples_df if examples_df is not None else df
    if strip_cols_all and examples_source is not None and not examples_source.empty:
        rng = np.random.default_rng(0)
        fig, axes = plt.subplots(5, 3, figsize=(12, 14), sharex=True, sharey=False)
        strip_positions = np.arange(1, 5)
        for ax in axes.flatten():
            plane = int(rng.integers(1, 5))
            strip_cols = [f"Y_mea_{plane}_s{s}" for s in range(1, 5)]
            if not all(col in examples_source.columns for col in strip_cols):
                ax.axis("off")
                continue
            qsum_matrix = examples_source[strip_cols].to_numpy(dtype=float)
            aval_col = f"avalanche_size_electrons_{plane}"
            if aval_col in examples_source.columns:
                aval_mask = examples_source[aval_col].to_numpy(dtype=float) >= 1.0
            else:
                aval_mask = np.ones(len(examples_source), dtype=bool)
            if "tt_hit" in examples_source.columns:
                tt_series = normalize_tt(examples_source["tt_hit"]).astype(str)
                tt_mask = tt_series == "1234"
            else:
                tt_series = None
                tt_mask = np.ones(len(examples_source), dtype=bool)
            hit_mask = (qsum_matrix > 0).any(axis=1) & aval_mask & tt_mask
            hit_indices = np.where(hit_mask)[0]
            if len(hit_indices) == 0:
                ax.axis("off")
                continue
            idx = int(rng.choice(hit_indices))
            vals = examples_source.loc[idx, strip_cols].to_numpy(dtype=float)
            bars = ax.bar(strip_positions, vals, width=0.6, alpha=0.8)
            for patch in bars:
                patch.set_rasterized(True)
            tt_label = tt_series.iloc[idx] if tt_series is not None else ""
            ax.set_title(f"P{plane} row {idx} tt={tt_label}")
            ax.set_xticks(strip_positions)
            ax.set_xlabel("Strip")
        axes[0, 0].set_ylabel("qsum")
        fig.suptitle("Charge sharing examples (single plane per event)")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    # Single-event thrown-point cloud (MASTER_STEPS-style)
    # - pick one event that has at least one avalanche center and draw many sampled points
    examples_source = examples_df if examples_df is not None else df
    if examples_source is not None and not examples_source.empty:
        # candidate events must have at least one plane with finite avalanche center
        center_mask = np.zeros(len(examples_source), dtype=bool)
        for p in range(1, 5):
            ax_col = f"avalanche_x_{p}"
            ay_col = f"avalanche_y_{p}"
            if ax_col in examples_source.columns and ay_col in examples_source.columns:
                xv = examples_source[ax_col].to_numpy(dtype=float)
                yv = examples_source[ay_col].to_numpy(dtype=float)
                center_mask |= (~np.isnan(xv)) & (~np.isnan(yv))
        if center_mask.any():
            rng_single = np.random.default_rng(point_seed if point_seed is not None else 1)
            sel_idx = int(rng_single.choice(np.where(center_mask)[0]))
            n_points = int(points_per_plane) if points_per_plane and points_per_plane > 0 else 2000

            fig, axes = plt.subplots(2, 2, figsize=(10, 9))
            for plane_idx in range(1, 5):
                ax = axes[(plane_idx - 1) // 2, (plane_idx - 1) % 2]
                ax_col = f"avalanche_x_{plane_idx}"
                ay_col = f"avalanche_y_{plane_idx}"
                if ax_col not in examples_source.columns or ay_col not in examples_source.columns:
                    ax.axis("off")
                    continue
                center_x = examples_source.iloc[sel_idx][ax_col]
                center_y = examples_source.iloc[sel_idx][ay_col]
                if np.isnan(center_x) or np.isnan(center_y):
                    ax.axis("off")
                    continue

                # determine sampling sigma: prefer scaled width, fall back to sensible default
                scaled_col = f"avalanche_scaled_width_{plane_idx}"
                if scaled_col in examples_source.columns:
                    try:
                        sigma = float(examples_source.iloc[sel_idx][scaled_col])
                    except Exception:
                        sigma = 40.0
                else:
                    # fallback: try avalanche size -> crude proxy, else use 40 mm
                    aval_col = f"avalanche_size_electrons_{plane_idx}"
                    if aval_col in examples_source.columns:
                        try:
                            aval_val = float(examples_source.iloc[sel_idx][aval_col])
                            sigma = float(max(1.0, min(80.0, np.cbrt(aval_val))))
                        except Exception:
                            sigma = 40.0
                    else:
                        sigma = 40.0

                points_rng = np.random.default_rng(point_seed if point_seed is not None else 1)
                x_pts = center_x + points_rng.normal(0.0, sigma, n_points)
                y_pts = center_y + points_rng.normal(0.0, sigma, n_points)

                ax.scatter(x_pts, y_pts, s=4, alpha=0.35, rasterized=True)
                ax.scatter([center_x], [center_y], s=40, color="black", marker="x")

                title = f"Plane {plane_idx} thrown points"
                aval_col = f"avalanche_size_electrons_{plane_idx}"
                if aval_col in examples_source.columns:
                    try:
                        aval_val = float(examples_source.iloc[sel_idx][aval_col])
                        title += f" (avalanche={aval_val:,.0f} e-)"
                    except Exception:
                        pass
                ax.set_title(title)
                ax.set_xlabel("X (mm)")
                ax.set_ylabel("Y (mm)")

                lim = max(150.0, 4.0 * sigma)
                ax.set_xlim(center_x - lim, center_x + lim)
                ax.set_ylim(center_y - lim, center_y + lim)

            tt_label = ""
            if "tt_hit" in examples_source.columns:
                try:
                    tt_label = normalize_tt(examples_source["tt_hit"]).iloc[sel_idx]
                except Exception:
                    tt_label = ""
            fig.suptitle(f"Avalanche thrown-point cloud — single example (row={sel_idx} tt={tt_label})")
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for plane_idx, ax in enumerate(axes, start=1):
        aval_col = f"avalanche_size_electrons_{plane_idx}"
        strip_cols = [c for c in df.columns if c.startswith(f"Y_mea_{plane_idx}_s")]
        if aval_col not in df.columns or not strip_cols:
            ax.axis("off")
            continue
        aval_vals = df[aval_col].to_numpy(dtype=float)
        strip_vals = df[strip_cols].to_numpy(dtype=float)
        qsum_total = strip_vals.sum(axis=1)
        mask = (aval_vals > 0) & (qsum_total > 0)
        ratios = np.zeros_like(aval_vals)
        ratios[mask] = qsum_total[mask] / aval_vals[mask]
        ratios = ratios[ratios > 0]
        if len(ratios) == 0:
            ax.axis("off")
            continue
        ax.hist(ratios, bins=80, color="steelblue", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} qsum / avalanche size")
        ax.set_xlim(left=0)
        for patch in ax.patches:
            patch.set_rasterized(True)
    axes[-1].set_xlabel("qsum / avalanche size")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    if include_thrown_points and thrown_points:
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for plane_idx in range(1, 5):
            ax = axes[(plane_idx - 1) // 2, (plane_idx - 1) % 2]
            points = thrown_points.get(plane_idx)
            if not points:
                ax.axis("off")
                continue
            x_pts = points["x"]
            y_pts = points["y"]
            charge = points.get("charge")
            if charge is None:
                sc = ax.scatter(x_pts, y_pts, s=6, alpha=0.6)
                ax.set_title(f"Plane {plane_idx} thrown points")
            else:
                sc = ax.scatter(x_pts, y_pts, c=charge, s=8, cmap="viridis")
                ax.set_title(f"Plane {plane_idx} thrown points (Q={charge:,.0f})")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

# helpers to find/load chunk ---------------------------------------------------

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
    step = 4
    sample = find_any_chunk_for_step(step)
    if sample is None:
        print("No sample chunk found for STEP 4; exiting.")
        return
    print(f"Using sample: {sample}")
    df = load_df(sample)
    out_dir = Path(__file__).resolve().parent / "PLOTS"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"step_{step}_sample_plots.pdf"
    with PdfPages(out_path) as pdf:
        plot_hit_summary(df, pdf)
        plot_step4_summary(df, pdf)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
