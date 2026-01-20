#!/usr/bin/env python3
"""Step 4: induce strip signals and measured hit quantities.

Inputs: geom_<G>_avalanche from Step 3.
Outputs: geom_<G>_hit.(pkl|csv) with per-strip charges/positions/times and metadata.
"""

from __future__ import annotations

import argparse
import gc
import json
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

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from STEP_SHARED.sim_utils import (
    ensure_dir,
    find_latest_data_path,
    find_sim_run_dir,
    get_strip_geometry,
    iter_input_frames,
    latest_sim_run,
    load_step_configs,
    load_with_metadata,
    now_iso,
    find_sim_run,
    random_sim_run,
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


def compute_strip_signals(
    y_center: np.ndarray,
    lower_edges: np.ndarray,
    upper_edges: np.ndarray,
    width: float,
) -> np.ndarray:
    n_events = len(y_center)
    n_strips = len(lower_edges)
    signals = np.zeros((n_events, n_strips), dtype=float)

    valid = ~np.isnan(y_center)
    if not np.any(valid):
        return signals

    y_valid = y_center[valid]
    y_low = y_valid - width / 2
    y_high = y_valid + width / 2

    for idx in range(n_strips):
        overlap = np.minimum(y_high, upper_edges[idx]) - np.maximum(y_low, lower_edges[idx])
        overlap = np.maximum(overlap, 0.0)
        signals[valid, idx] = overlap / width

    return signals


def compute_strip_signals_variable_width(
    y_center: np.ndarray,
    widths: np.ndarray,
    lower_edges: np.ndarray,
    upper_edges: np.ndarray,
) -> np.ndarray:
    n_events = len(y_center)
    n_strips = len(lower_edges)
    signals = np.zeros((n_events, n_strips), dtype=float)

    valid = ~np.isnan(y_center)
    if not np.any(valid):
        return signals

    y_valid = y_center[valid]
    widths_valid = widths[valid]
    width_safe = np.where(widths_valid > 0, widths_valid, np.nan)
    y_low = y_valid - width_safe / 2
    y_high = y_valid + width_safe / 2

    for idx in range(n_strips):
        overlap = np.minimum(y_high, upper_edges[idx]) - np.maximum(y_low, lower_edges[idx])
        overlap = np.maximum(overlap, 0.0)
        frac = overlap / width_safe
        frac = np.where(np.isfinite(frac), frac, 0.0)
        signals[valid, idx] = frac

    return signals


def circle_area_below(y: np.ndarray, radius: np.ndarray, area_total: np.ndarray) -> np.ndarray:
    y_clipped = np.clip(y, -radius, radius)
    term = y_clipped / radius
    area = (radius ** 2) * (np.arcsin(term) + term * np.sqrt(1.0 - term ** 2)) + 0.5 * area_total
    area = np.where(y <= -radius, 0.0, area)
    area = np.where(y >= radius, area_total, area)
    return area


def induce_signal(
    df: pd.DataFrame,
    x_noise: float,
    time_sigma_ns: float,
    avalanche_width: float,
    charge_share_points: int,
    width_scale_exponent: float,
    width_scale_max: float,
    rng: np.random.Generator,
    debug_event_index: int | None,
    debug_points: dict | None,
    debug_rng: np.random.Generator | None,
) -> pd.DataFrame:
    out = df.copy()
    n = len(df)
    tt_array = np.full(n, "", dtype=object)

    for plane_idx in range(1, 5):
        aval_q_col = f"avalanche_size_electrons_{plane_idx}"
        aval_x_col = f"avalanche_x_{plane_idx}"
        aval_y_col = f"avalanche_y_{plane_idx}"
        aval_q = df.get(aval_q_col, pd.Series(np.zeros(n))).to_numpy(dtype=float)
        aval_x = df.get(aval_x_col, pd.Series(np.full(n, np.nan))).to_numpy(dtype=float)
        aval_y = df.get(aval_y_col, pd.Series(np.full(n, np.nan))).to_numpy(dtype=float)
        t_sum_col = f"T_sum_{plane_idx}_ns"
        t_sum_vals = df[t_sum_col].to_numpy(dtype=float) if t_sum_col in df.columns else None

        y_width, _, lower_edges, upper_edges = get_strip_geometry(plane_idx)
        positive_mask = aval_q > 0
        if np.any(positive_mask):
            hist_vals = aval_q[positive_mask]
            counts, edges = np.histogram(hist_vals, bins="auto")
            mode_idx = int(np.argmax(counts)) if len(counts) else 0
            mode_charge = 0.5 * (edges[mode_idx] + edges[mode_idx + 1]) if len(edges) > 1 else 1.0
        else:
            mode_charge = 1.0
        if mode_charge <= 0:
            mode_charge = 1.0

        base_scale = np.where(aval_q > 0, aval_q / mode_charge, 0.0)
        width_scale = np.power(base_scale, width_scale_exponent, where=base_scale > 0, out=np.zeros_like(base_scale))
        width_scale = np.minimum(width_scale, width_scale_max)
        scaled_width = avalanche_width * width_scale
        valid = ~np.isnan(aval_y) & (scaled_width > 0)
        y_valid = aval_y[valid].astype(np.float64, copy=False)
        radius_valid = (scaled_width[valid] / 2.0).astype(np.float64, copy=False)
        area_total = np.pi * radius_valid * radius_valid
        out[f"avalanche_size_electrons_{plane_idx}"] = aval_q.astype(np.float32, copy=False)
        out[f"avalanche_width_scale_{plane_idx}"] = width_scale.astype(np.float32, copy=False)
        out[f"avalanche_scaled_width_{plane_idx}"] = scaled_width.astype(np.float32, copy=False)

        plane_detected = np.zeros(n, dtype=bool)
        remaining = np.full(n, charge_share_points, dtype=np.int32)
        cumulative_p = np.zeros(n, dtype=np.float32)
        debug_counts = None
        if (
            debug_event_index is not None
            and 0 <= debug_event_index < n
            and valid[debug_event_index]
        ):
            sigma = float(scaled_width[debug_event_index])
            center_x = float(aval_x[debug_event_index])
            center_y = float(aval_y[debug_event_index])
            points_rng = debug_rng or rng
            x_pts = center_x + points_rng.normal(0.0, sigma, int(charge_share_points))
            y_pts = center_y + points_rng.normal(0.0, sigma, int(charge_share_points))
            if debug_points is not None:
                debug_points[plane_idx] = {
                    "x": x_pts,
                    "y": y_pts,
                    "charge": float(aval_q[debug_event_index]),
                }
            debug_counts = []
            for strip_idx in range(len(lower_edges)):
                in_strip = (y_pts >= lower_edges[strip_idx]) & (y_pts < upper_edges[strip_idx])
                debug_counts.append(int(in_strip.sum()))

        for strip_idx in range(len(lower_edges)):
            frac = np.zeros(n, dtype=np.float32)
            if valid.any():
                y_low_rel = lower_edges[strip_idx] - y_valid
                y_high_rel = upper_edges[strip_idx] - y_valid
                area_low = circle_area_below(y_low_rel, radius_valid, area_total)
                area_high = circle_area_below(y_high_rel, radius_valid, area_total)
                area_strip = np.maximum(area_high - area_low, 0.0)
                p_strip = np.zeros(n, dtype=np.float32)
                p_strip[valid] = np.where(area_total > 0, area_strip / area_total, 0.0)
                denom = 1.0 - cumulative_p
                adj = np.zeros(n, dtype=np.float32)
                denom_mask = denom > 0
                adj[denom_mask] = p_strip[denom_mask] / denom[denom_mask]
                counts = rng.binomial(remaining, np.clip(adj, 0.0, 1.0))
                if debug_counts is not None:
                    counts[debug_event_index] = debug_counts[strip_idx]
                remaining -= counts
                cumulative_p += p_strip
                frac = counts.astype(np.float32, copy=False) / float(charge_share_points)
            qsum = (frac * aval_q).astype(np.float32, copy=False)
            out[f"Y_mea_{plane_idx}_s{strip_idx + 1}"] = qsum
            hit_mask = qsum > 0
            plane_detected |= hit_mask

            x_strip = np.full(n, np.nan, dtype=np.float32)
            if hit_mask.any():
                x_strip[hit_mask] = (
                    aval_x[hit_mask] + rng.normal(0.0, x_noise, hit_mask.sum())
                ).astype(np.float32, copy=False)
            out[f"X_mea_{plane_idx}_s{strip_idx + 1}"] = x_strip

            if t_sum_vals is not None:
                t_strip = np.full(n, np.nan, dtype=np.float32)
                t_valid = hit_mask & ~np.isnan(t_sum_vals)
                if t_valid.any():
                    t_strip[t_valid] = (
                        t_sum_vals[t_valid] + rng.normal(0.0, time_sigma_ns, t_valid.sum())
                    ).astype(np.float32, copy=False)
                out[f"T_sum_meas_{plane_idx}_s{strip_idx + 1}"] = t_strip
        tt_array[plane_detected] = tt_array[plane_detected] + str(plane_idx)

    out["tt_hit"] = pd.Series(tt_array, dtype="string").replace("", pd.NA)
    return out


def plot_hit_summary(df: pd.DataFrame, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = normalize_tt(df["tt_hit"]).value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color="steelblue", alpha=0.8)
    for patch in bars:
        patch.set_rasterized(True)
    ax.set_title("tt_hit counts")
    ax.set_xlabel("tt_hit")
    ax.set_ylabel("Counts")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def plot_step4_summary(
    df: pd.DataFrame,
    pdf: PdfPages,
    include_thrown_points: bool,
    points_per_plane: int,
    point_seed: int | None,
    thrown_points: dict | None,
    examples_df: pd.DataFrame | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    counts = normalize_tt(df["tt_hit"]).value_counts().sort_index()
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
            ax.scatter(x_pts, y_pts, s=8, alpha=0.5, rasterized=True)
            ax.scatter([np.mean(x_pts)], [np.mean(y_pts)], s=40, color="black", marker="x")
            _, _, lower_edges, upper_edges = get_strip_geometry(plane_idx)
            for edge in np.concatenate([lower_edges, upper_edges]):
                ax.axhline(edge, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            if charge is not None:
                ax.set_title(f"Plane {plane_idx} thrown points (Q={charge:,.0f})")
            else:
                ax.set_title(f"Plane {plane_idx} thrown points")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)
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
        qsum_total = qsum_total[mask]
        aval_vals = aval_vals[mask]
        ax.hist(qsum_total, bins=120, color="seagreen", alpha=0.8, label="qsum total")
        ax.hist(aval_vals, bins=120, color="darkorange", alpha=0.5, label="avalanche size")
        ax.set_title(f"Plane {plane_idx} qsum total vs avalanche size")
        ax.set_xlim(left=0)
        ax.legend()
        for patch in ax.patches:
            patch.set_rasterized(True)
    axes[-1].set_xlabel("electrons")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for plane_idx, ax in enumerate(axes, start=1):
        strip_cols = [c for c in df.columns if c.startswith(f"Y_mea_{plane_idx}_s")]
        if not strip_cols:
            ax.axis("off")
            continue
        strip_vals = df[strip_cols].to_numpy(dtype=float)
        qsum_total = strip_vals.sum(axis=1)
        qsum_total = qsum_total[qsum_total > 0]
        if len(qsum_total) == 0:
            ax.axis("off")
            continue
        median_val = np.median(qsum_total)
        zoom_vals = qsum_total[qsum_total <= median_val]
        ax.hist(zoom_vals, bins=80, color="teal", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} qsum total (0 to median)")
        ax.set_xlim(0, median_val)
        for patch in ax.patches:
            patch.set_rasterized(True)
    axes[-1].set_xlabel("electrons")
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
        aval_vals = aval_vals[mask]
        qsum_total = qsum_total[mask]
        if len(aval_vals) == 0:
            ax.axis("off")
            continue
        ax.scatter(
            aval_vals,
            qsum_total,
            s=2,
            alpha=0.2,
            rasterized=True,
        )
        ax.set_title(f"Plane {plane_idx} qsum total vs avalanche size")
        ax.set_xlabel("avalanche size")
        ax.set_ylabel("qsum total")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for plane_idx, ax in enumerate(axes, start=1):
        col = f"avalanche_size_electrons_{plane_idx}"
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        ax.hist(vals, bins=80, color="darkorange", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} avalanche size")
        ax.set_xlim(left=0)
        for patch in ax.patches:
            patch.set_rasterized(True)
    axes[-1].set_xlabel("avalanche size (electrons)")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for plane_idx, ax in enumerate(axes, start=1):
        col = f"avalanche_width_scale_{plane_idx}"
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        ax.hist(vals, bins=80, color="seagreen", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} width scale")
        ax.set_xlim(left=0)
        for patch in ax.patches:
            patch.set_rasterized(True)
    axes[-1].set_xlabel("width scale (charge / mode charge)")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for plane_idx, ax in enumerate(axes, start=1):
        col = f"avalanche_scaled_width_{plane_idx}"
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        ax.hist(vals, bins=80, color="slateblue", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} scaled width")
        ax.set_xlim(left=0)
        for patch in ax.patches:
            patch.set_rasterized(True)
    axes[-1].set_xlabel("scaled width (mm)")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        strip_cols = [c for c in df.columns if c.startswith(f"Y_mea_{plane_idx}_s")]
        if not strip_cols:
            ax.axis("off")
            continue
        vals = df[strip_cols].to_numpy(dtype=float).sum(axis=1)
        vals = vals[vals > 0]
        ax.hist(vals, bins=120, color="seagreen", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} qsum total")
        ax.set_xlabel("qsum total")
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

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        strip_cols = [c for c in df.columns if c.startswith(f"X_mea_{plane_idx}_s")]
        if not strip_cols:
            ax.axis("off")
            continue
        vals = df[strip_cols].to_numpy(dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=80, color="darkorange", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} X_mea")
        ax.set_xlabel("X_mea (mm)")
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
            col = f"X_mea_{plane_idx}_s{strip_idx}"
            if col not in df.columns:
                ax.axis("off")
                continue
            vals = df[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            ax.hist(vals, bins=80, color="darkorange", alpha=0.8)
            ax.set_title(f"P{plane_idx} S{strip_idx}")
            ax.set_xlabel("X_mea (mm)")
    for ax in axes.flatten():
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        strip_cols = [c for c in df.columns if c.startswith(f"T_sum_meas_{plane_idx}_s")]
        if not strip_cols:
            ax.axis("off")
            continue
        vals = df[strip_cols].to_numpy(dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=80, color="slateblue", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} T_sum_meas")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: avalanche -> induced signal (hit vectors).")
    parser.add_argument("--config", default="config_step_4_physics.yaml", help="Path to step physics config YAML")
    parser.add_argument(
        "--runtime-config",
        default=None,
        help="Path to step runtime config YAML (defaults to *_runtime.yaml)",
    )
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing outputs")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--force", action="store_true", help="Recompute even if sim_run exists")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    runtime_path = Path(args.runtime_config) if args.runtime_config else None
    if runtime_path is not None and not runtime_path.is_absolute():
        runtime_path = Path(__file__).resolve().parent / runtime_path

    physics_cfg, runtime_cfg, cfg, runtime_path = load_step_configs(config_path, runtime_path)

    input_dir = Path(cfg["input_dir"])
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parent / input_dir
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    output_format = str(cfg.get("output_format", "pkl")).lower()
    chunk_rows = cfg.get("chunk_rows")
    plot_sample_rows = cfg.get("plot_sample_rows")
    x_noise = float(cfg.get("x_noise_mm", 0.0))
    time_sigma_ns = float(cfg.get("time_sigma_ns", 0.0))
    avalanche_width = float(cfg.get("avalanche_width_mm", 40.0))
    charge_share_points = int(cfg.get("charge_share_points", 2000))
    width_scale_exponent = float(cfg.get("width_scale_exponent", 0.5))
    width_scale_max = float(cfg.get("width_scale_max", 2.0))
    rng = np.random.default_rng(cfg.get("seed"))

    input_glob = cfg.get("input_glob", "**/geom_*_avalanche.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is not None and str(geometry_id).lower() != "auto":
        geometry_id = int(geometry_id)
    else:
        geometry_id = None
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 4 starting...")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"geometry_id: {geometry_id}")
    print(f"input_sim_run: {input_sim_run}")

    if args.plot_only:
        if args.no_plots:
            print("Plot-only requested with --no-plots; skipping plots.")
            return
        latest_path = find_latest_data_path(output_dir)
        if latest_path is None:
            raise FileNotFoundError(f"No existing outputs found in {output_dir} for plot-only.")
        df, _ = load_with_metadata(latest_path)
        sim_run_dir = find_sim_run_dir(latest_path)
        plot_dir = (sim_run_dir or latest_path.parent) / "PLOTS"
        ensure_dir(plot_dir)
        plot_path = plot_dir / f"{latest_path.stem}_plots.pdf"
        with PdfPages(plot_path) as pdf:
            plot_hit_summary(df, pdf)
            plot_step4_summary(
                df,
                pdf,
                include_thrown_points=False,
                points_per_plane=charge_share_points,
                point_seed=cfg.get("seed"),
                thrown_points=None,
                examples_df=df,
            )
        print(f"Saved {plot_path}")
        return

    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)
    elif input_sim_run == "random":
        input_sim_run = random_sim_run(input_dir, cfg.get("seed"))

    input_run_dir = input_dir / str(input_sim_run)
    if "**" in input_glob:
        input_paths = sorted(input_run_dir.rglob(input_glob.replace("**/", "")))
    else:
        input_paths = sorted(input_run_dir.glob(input_glob))
    def normalize_stem(path: Path) -> str:
        name = path.name
        if name.endswith(".chunks.json"):
            name = name[: -len(".chunks.json")]
        stem = Path(name).stem
        return stem.replace(".chunks", "")

    if geometry_id is not None:
        geom_key = f"geom_{geometry_id}"
        input_paths = [
            p for p in input_paths if normalize_stem(p).startswith(f"{geom_key}_avalanche")
        ]
        if not input_paths:
            fallback_path = input_run_dir / f"{geom_key}_avalanche.chunks.json"
            if fallback_path.exists():
                input_paths = [fallback_path]
    elif not input_paths:
        input_paths = sorted(input_run_dir.glob("geom_*_avalanche.chunks.json"))
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    normalized_stem = normalize_stem(input_path)
    if geometry_id is None:
        parts = normalized_stem.split("_")
        if len(parts) < 2 or parts[0] != "geom":
            raise ValueError(f"Unable to infer geometry_id from {input_path.stem}")
        geometry_id = int(parts[1])
    print(f"Processing: {input_path}")
    input_iter, upstream_meta, chunked_input = iter_input_frames(input_path, chunk_rows)
    if not args.force:
        existing = find_sim_run(output_dir, physics_cfg, upstream_meta)
        if existing:
            print(f"SIM_RUN {existing} already exists; skipping (use --force to regenerate).")
            return
    print("Inducing strip signals...")

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_4", config_path, physics_cfg, upstream_meta
    )
    print(f"Resolved output sim_run: {sim_run}")
    reset_dir(sim_run_dir)
    print(f"Output dir reset: {sim_run_dir}")

    out_stem = normalized_stem.replace("_avalanche", "") + "_hit"
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_4",
        "config": physics_cfg,
        "runtime_config": runtime_cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
    }
    def select_debug_event(frame: pd.DataFrame, chooser: np.random.Generator) -> int | None:
        required_cols = [f"avalanche_size_electrons_{i}" for i in range(1, 5)]
        for col in required_cols:
            if col not in frame.columns:
                return None
        mask = np.ones(len(frame), dtype=bool)
        for col in required_cols:
            mask &= frame[col].to_numpy(dtype=float) > 0
        if not mask.any():
            return None
        indices = np.where(mask)[0]
        return int(chooser.choice(indices))

    def prepare_chunk(
        chunk: pd.DataFrame,
        debug_state: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        needed_cols = {"event_id", "X_gen", "Y_gen", "Theta_gen", "Phi_gen", "T0_ns", "T_thick_s"}
        for plane_idx in range(1, 5):
            needed_cols.update(
                {
                    f"avalanche_size_electrons_{plane_idx}",
                    f"avalanche_x_{plane_idx}",
                    f"avalanche_y_{plane_idx}",
                    f"T_sum_{plane_idx}_ns",
                }
            )
        keep_cols = [col for col in chunk.columns if col in needed_cols]
        chunk = chunk[keep_cols]
        debug_event_index = None
        if not debug_state["captured"]:
            debug_event_index = select_debug_event(chunk, debug_rng)
        out_full = induce_signal(
            chunk,
            x_noise,
            time_sigma_ns,
            avalanche_width,
            charge_share_points,
            width_scale_exponent,
            width_scale_max,
            rng,
            debug_event_index,
            debug_state["points"] if debug_event_index is not None else None,
            debug_rng if debug_event_index is not None else None,
        )
        plot_cols = [
            col
            for col in out_full.columns
            if col == "tt_hit"
            or col.startswith(("Y_mea_", "X_mea_", "T_sum_meas_"))
            or col.startswith(
                (
                    "avalanche_size_electrons_",
                    "avalanche_width_scale_",
                    "avalanche_scaled_width_",
                    "avalanche_x_",
                    "avalanche_y_",
                )
            )
        ]
        plot_df = out_full[plot_cols]
        if debug_event_index is not None and debug_state["points"]:
            debug_state["captured"] = True
            debug_state["plot_df"] = plot_df
        return out_full, plot_df

    debug_state = {"captured": False, "points": {}, "plot_df": None}
    debug_rng = np.random.default_rng()

    if chunk_rows:
        chunks_dir = sim_run_dir / f"{out_stem}_chunks"
        ensure_dir(chunks_dir)
        chunk_paths = []
        buffer_full = []
        buffer_plot = []
        buffered_rows = 0
        full_chunks = 0
        last_plot_df = None

        def flush_chunk(out_df: pd.DataFrame, plot_df: pd.DataFrame) -> None:
            nonlocal full_chunks, last_plot_df
            chunk_path = chunks_dir / f"part_{full_chunks:04d}.{output_format}"
            if output_format == "csv":
                out_df.to_csv(chunk_path, index=False)
            elif output_format == "pkl":
                out_df.to_pickle(chunk_path)
            else:
                raise ValueError(f"Unsupported output_format: {output_format}")
            chunk_paths.append(str(chunk_path))
            full_chunks += 1
            last_plot_df = plot_df

        def maybe_flush_buffer() -> None:
            nonlocal buffer_full, buffer_plot, buffered_rows
            while buffered_rows >= int(chunk_rows):
                full_df = pd.concat(buffer_full, ignore_index=True)
                plot_full = pd.concat(buffer_plot, ignore_index=True)
                out_df = full_df.iloc[: int(chunk_rows)].copy()
                plot_df = plot_full.iloc[: int(chunk_rows)].copy()
                remainder_full = full_df.iloc[int(chunk_rows):].copy()
                remainder_plot = plot_full.iloc[int(chunk_rows):].copy()
                flush_chunk(out_df, plot_df)
                buffer_full = [remainder_full] if not remainder_full.empty else []
                buffer_plot = [remainder_plot] if not remainder_plot.empty else []
                buffered_rows = len(remainder_full)

        total_rows = 0
        for chunk in input_iter:
            out_chunk, plot_chunk = prepare_chunk(chunk, debug_state)
            if out_chunk.empty:
                continue
            total_rows += len(out_chunk)
            buffer_full.append(out_chunk)
            buffer_plot.append(plot_chunk)
            buffered_rows += len(out_chunk)
            maybe_flush_buffer()

        if full_chunks == 0 and buffered_rows > 0:
            full_df = pd.concat(buffer_full, ignore_index=True)
            plot_full = pd.concat(buffer_plot, ignore_index=True)
            flush_chunk(full_df, plot_full)
            buffered_rows = 0
        else:
            buffered_rows = 0

        row_count = full_chunks * int(chunk_rows) + buffered_rows
        manifest = {
            "version": 1,
            "chunks": chunk_paths,
            "row_count": row_count,
            "metadata": metadata,
        }
        manifest_path = sim_run_dir / f"{out_stem}.chunks.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print("Signal induction complete.")
        print(f"Saved data: {manifest_path}")

        plot_df = debug_state["plot_df"] if debug_state["plot_df"] is not None else last_plot_df
        plot_df_examples = plot_df
        if plot_sample_rows and plot_df is not None:
            sample_n = len(plot_df) if plot_sample_rows is True else int(plot_sample_rows)
            sample_n = min(sample_n, len(plot_df))
            plot_df = plot_df.sample(n=sample_n, random_state=cfg.get("seed"))

        if not args.no_plots and plot_df is not None:
            plot_dir = sim_run_dir / "PLOTS"
            ensure_dir(plot_dir)
            plot_path = plot_dir / f"{out_stem}_plots.pdf"
            print("Plotting plots...")
            with PdfPages(plot_path) as pdf:
                plot_hit_summary(plot_df, pdf)
                plot_step4_summary(
                    plot_df,
                    pdf,
                    include_thrown_points=True,
                    points_per_plane=charge_share_points,
                    point_seed=cfg.get("seed"),
                    thrown_points=debug_state["points"],
                    examples_df=plot_df_examples,
                )
            print(f"Saved plots: {plot_path}")
    else:
        df, upstream_meta = load_with_metadata(input_path)
        print(f"Loaded {len(df):,} rows from {input_path.name}")
        needed_cols = {"event_id", "X_gen", "Y_gen", "Theta_gen", "Phi_gen"}
        for plane_idx in range(1, 5):
            needed_cols.update(
                {
                    f"avalanche_size_electrons_{plane_idx}",
                    f"avalanche_x_{plane_idx}",
                    f"avalanche_y_{plane_idx}",
                    f"T_sum_{plane_idx}_ns",
                }
            )
        keep_cols = [col for col in df.columns if col in needed_cols]
        df = df[keep_cols]
        debug_event_index = select_debug_event(df, debug_rng)
        out = induce_signal(
            df,
            x_noise,
            time_sigma_ns,
            avalanche_width,
            charge_share_points,
            width_scale_exponent,
            width_scale_max,
            rng,
            debug_event_index,
            debug_state["points"] if debug_event_index is not None else None,
            debug_rng if debug_event_index is not None else None,
        )
        print("Signal induction complete.")

        out_path = sim_run_dir / f"{out_stem}.{output_format}"
        plot_cols = [
            col
            for col in out.columns
            if col == "tt_hit"
            or col.startswith(("Y_mea_", "X_mea_", "T_sum_meas_"))
            or col.startswith(
                (
                    "avalanche_size_electrons_",
                    "avalanche_width_scale_",
                    "avalanche_scaled_width_",
                    "avalanche_x_",
                    "avalanche_y_",
                )
            )
        ]
        plot_df = out[plot_cols]
        plot_df_examples = plot_df
        plot_sample_size = cfg.get("plot_sample_size", 200000)
        if plot_sample_size:
            plot_sample_size = int(plot_sample_size)
            if 0 < plot_sample_size < len(plot_df):
                plot_df = plot_df.sample(n=plot_sample_size, random_state=cfg.get("seed"))
                print(f"Plotting with sample size: {len(plot_df):,}")

        save_with_metadata(out, out_path, metadata, output_format)
        print(f"Saved data: {out_path}")
        del df
        del out
        gc.collect()

        if not args.no_plots:
            plot_dir = sim_run_dir / "PLOTS"
            ensure_dir(plot_dir)
            plot_path = plot_dir / f"{out_path.stem}_plots.pdf"
            print("Plotting plots...")
            with PdfPages(plot_path) as pdf:
                plot_hit_summary(plot_df, pdf)
                plot_step4_summary(
                    plot_df,
                    pdf,
                    include_thrown_points=True,
                    points_per_plane=charge_share_points,
                    point_seed=cfg.get("seed"),
                    thrown_points=debug_state["points"],
                    examples_df=plot_df_examples,
                )
            print(f"Saved plots: {plot_path}")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
