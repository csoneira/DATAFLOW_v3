#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
    get_strip_geometry,
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
) -> pd.DataFrame:
    base_cols = [c for c in ("X_gen", "Y_gen", "Theta_gen", "Phi_gen") if c in df.columns]
    out = df[base_cols].copy()
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

        drop_cols = [c for c in (aval_q_col, aval_x_col, aval_y_col, t_sum_col) if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

    out["tt_hit"] = pd.Series(tt_array, dtype="string").replace("", pd.NA)
    return out


def plot_hit_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
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


def plot_step4_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
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
            ax.hist(ratios, bins=80, color="steelblue", alpha=0.8)
            ax.set_title(f"Plane {plane_idx} qsum / avalanche size")
            ax.set_xlim(left=0)
            for patch in ax.patches:
                patch.set_rasterized(True)
        axes[-1].set_xlabel("qsum / avalanche size")
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
            if len(aval_vals) > 0:
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
            vals = df[strip_cols].to_numpy(dtype=float).ravel()
            vals = vals[vals > 0]
            ax.hist(vals, bins=120, color="seagreen", alpha=0.8)
            ax.set_title(f"Plane {plane_idx} qsum")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: avalanche -> induced signal (hit vectors).")
    parser.add_argument("--config", default="config_step_4.yaml", help="Path to step config YAML")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing outputs")
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
    x_noise = float(cfg.get("x_noise_mm", 0.0))
    time_sigma_ns = float(cfg.get("time_sigma_ns", 0.0))
    avalanche_width = float(cfg.get("avalanche_width_mm", 40.0))
    charge_share_points = int(cfg.get("charge_share_points", 2000))
    width_scale_exponent = float(cfg.get("width_scale_exponent", 0.5))
    width_scale_max = float(cfg.get("width_scale_max", 2.0))
    rng = np.random.default_rng(cfg.get("seed"))

    input_glob = cfg.get("input_glob", "**/geom_*_avalanche.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is None:
        raise ValueError("config geometry_id is required for Step 4.")
    geometry_id = int(geometry_id)
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 4 starting...")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"geometry_id: {geometry_id}")
    print(f"input_sim_run: {input_sim_run}")

    if args.plot_only:
        for out_file in sorted(output_dir.rglob(f"SIM_RUN_*/geom_*_hit.{output_format}")):
            df, _ = load_with_metadata(out_file)
            plot_path = out_file.with_name(f"{out_file.stem}_summary.pdf")
            plot_hit_summary(df, plot_path)
            print(f"Saved {plot_path}")
        return

    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)

    input_run_dir = input_dir / str(input_sim_run)
    if "**" in input_glob:
        input_paths = sorted(input_run_dir.rglob(input_glob.replace("**/", "")))
    else:
        input_paths = sorted(input_run_dir.glob(input_glob))
    geom_key = f"geom_{geometry_id}"
    input_paths = [p for p in input_paths if p.stem.startswith(f"{geom_key}_avalanche")]
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    print(f"Processing: {input_path}")
    df, upstream_meta = load_with_metadata(input_path)
    print(f"Loaded {len(df):,} rows from {input_path.name}")
    print("Inducing strip signals...")
    needed_cols = {"X_gen", "Y_gen", "Theta_gen", "Phi_gen"}
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
    out = induce_signal(
        df,
        x_noise,
        time_sigma_ns,
        avalanche_width,
        charge_share_points,
        width_scale_exponent,
        width_scale_max,
        rng,
    )
    print("Signal induction complete.")

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_4", config_path, cfg, upstream_meta
    )
    print(f"Resolved output sim_run: {sim_run}")
    reset_dir(sim_run_dir)
    print(f"Output dir reset: {sim_run_dir}")

    out_name = input_path.stem.replace("_avalanche", "") + f"_hit.{output_format}"
    out_path = sim_run_dir / out_name
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_4",
        "config": cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
    }
    plot_cols = [
        col
        for col in out.columns
        if col == "tt_hit"
        or col.startswith(("Y_mea_", "X_mea_", "T_sum_meas_"))
        or col.startswith(("avalanche_size_electrons_", "avalanche_width_scale_", "avalanche_scaled_width_"))
    ]
    plot_df = out[plot_cols]
    plot_sample_size = cfg.get("plot_sample_size", 200000)
    if plot_sample_size:
        plot_sample_size = int(plot_sample_size)
        if 0 < plot_sample_size < len(plot_df):
            plot_df = plot_df.sample(n=plot_sample_size, random_state=cfg.get("seed"))
            print(f"Plotting with sample size: {len(plot_df):,}")

    keep_cols = {
        "X_gen",
        "Y_gen",
        "Theta_gen",
        "Phi_gen",
        "tt_hit",
    }
    for col in out.columns:
        if col.startswith(("Y_mea_", "X_mea_", "T_sum_meas_")):
            keep_cols.add(col)
    drop_cols = [col for col in out.columns if col not in keep_cols]
    out.drop(columns=drop_cols, inplace=True)

    save_with_metadata(out, out_path, metadata, output_format)
    print(f"Saved data: {out_path}")
    del df
    del out
    gc.collect()

    plot_path = sim_run_dir / f"{out_path.stem}_summary.pdf"
    print("Plotting summary...")
    plot_hit_summary(plot_df, plot_path)
    print(f"Saved summary plot: {plot_path}")
    single_path = sim_run_dir / f"{out_path.stem}_single.pdf"
    print("Plotting detailed summary...")
    plot_step4_summary(plot_df, single_path)
    print(f"Saved single plot: {single_path}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
