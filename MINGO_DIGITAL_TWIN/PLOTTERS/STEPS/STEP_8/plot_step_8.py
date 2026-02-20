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
from matplotlib.colors import LogNorm

# --- Read XY histogram config ---
import yaml

def load_xy_hist_config():
    config_path = Path(__file__).resolve().parent / "xy_hist_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        lims = cfg.get("xy_hist_limits", {})
        x_min = lims.get("x_min", -1000)
        x_max = lims.get("x_max", 1000)
        y_min = lims.get("y_min", -1000)
        y_max = lims.get("y_max", 1000)
        return x_min, x_max, y_min, y_max
    return -1000, 1000, -1000, 1000


def plot_threshold_summary(
    df: pd.DataFrame,
    output_path: Path,
    threshold: float = 0.0,
    qfront_offsets: list[list[float]] | None = None,
    qback_offsets: list[list[float]] | None = None,
    sample_path: Path | None = None,
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

        # --- 2D X_gen / Y_gen histograms per tt_* (identical behaviour to STEP_10) ---
        plot_tt_xy_histograms(df, pdf)

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

        # --- Add muon differential flux vs zenith-angle (compute locally like STEP_1) ---
        try:
            added = _plot_muon_differential_flux_vs_angle_local(df, pdf, sample_path=sample_path)
            if not added:
                print('Step 8 plots: skipped muon differential flux plot (missing gen/time columns).')
        except Exception:
            print('Step 8 plots: muon differential flux page failed to generate (exception).')


def _plot_muon_differential_flux_vs_angle_local(df: pd.DataFrame, pdf: PdfPages, sample_path: Path | None = None) -> bool:
    pass


def plot_tt_xy_histograms(df: pd.DataFrame, pdf: PdfPages) -> None:
    # Find all columns starting with 'tt_'
    tt_cols = [col for col in df.columns if col.startswith("tt_")]
    if "X_gen" not in df.columns or "Y_gen" not in df.columns:
        print("STEP_8: X_gen/Y_gen missing — skipping TT XY histograms.")
        return
    x_min, x_max, y_min, y_max = load_xy_hist_config()

    # Fallback: if there are no tt_* columns, produce a single overall 2D histogram
    if not tt_cols:
        print("STEP_8: no tt_* columns found — creating overall X_gen/Y_gen 2D histogram.")
        fig, ax = plt.subplots(figsize=(6, 6))
        x = df["X_gen"].to_numpy(dtype=float)
        y = df["Y_gen"].to_numpy(dtype=float)
        h = ax.hist2d(x, y, bins=80, range=[[x_min, x_max], [y_min, y_max]], cmap="turbo", norm=LogNorm(vmin=1))
        ax.set_title("2D X_gen/Y_gen (all events — no tt_* columns present)")
        ax.set_xlabel("X_gen")
        ax.set_ylabel("Y_gen")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        fig.colorbar(h[3], ax=ax)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        return

    n = len(tt_cols)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for idx, tt_col in enumerate(tt_cols):
        ax = axes[idx // ncols][idx % ncols]
        mask = df[tt_col].notna() & (df[tt_col] != "")
        x = df.loc[mask, "X_gen"].to_numpy(dtype=float)
        y = df.loc[mask, "Y_gen"].to_numpy(dtype=float)
        h = ax.hist2d(x, y, bins=60, range=[[x_min, x_max], [y_min, y_max]], cmap="turbo", norm=LogNorm(vmin=1))
        ax.set_title(f"2D X_gen/Y_gen for {tt_col}")
        ax.set_xlabel("X_gen")
        ax.set_ylabel("Y_gen")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        fig.colorbar(h[3], ax=ax)
    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    """Self-contained muon differential flux plot (same logic as STEP_1).

    - uses inner-second trimming
    - computes area from X_gen/Y_gen when available
    - divides by per-bin solid angle (ΔΩ)
    - overlays theoretical model using `param_mesh.csv` and `sample_path`
    """
    if "Theta_gen" not in df.columns or "T_thick_s" not in df.columns:
        return False
    if "X_gen" not in df.columns or "Y_gen" not in df.columns:
        return False

    # inner-second trimming
    t0_all = df["T_thick_s"].to_numpy(dtype=float)
    if not np.isfinite(t0_all).any():
        return False
    sec_min = int(np.floor(np.nanmin(t0_all)))
    sec_max = int(np.floor(np.nanmax(t0_all)))
    if sec_max <= sec_min + 1:
        return False
    inner_mask = (np.floor(t0_all).astype(int) > sec_min) & (np.floor(t0_all).astype(int) < sec_max) & np.isfinite(t0_all)

    theta_all = df["Theta_gen"].to_numpy(dtype=float)
    theta = theta_all[np.isfinite(theta_all) & (theta_all >= 0.0) & (theta_all <= np.pi / 2.0) & inner_mask]
    if theta.size < 20:
        return False

    # duration
    sec_min_inner = sec_min + 1
    sec_max_inner = sec_max - 1
    duration_s = float(sec_max_inner - sec_min_inner + 1)
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        return False
    duration_min = duration_s / 60.0

    # compute area from X_gen/Y_gen spread (mm -> cm^2)
    x_all = df["X_gen"].to_numpy(dtype=float)
    y_all = df["Y_gen"].to_numpy(dtype=float)
    x_vals = x_all[np.isfinite(x_all) & inner_mask]
    y_vals = y_all[np.isfinite(y_all) & inner_mask]
    if x_vals.size < 2 or y_vals.size < 2:
        return False
    x_spread_mm = float(np.max(x_vals) - np.min(x_vals))
    y_spread_mm = float(np.max(y_vals) - np.min(y_vals))
    area_mm2 = x_spread_mm * y_spread_mm
    area_cm2 = area_mm2 / 100.0
    if not np.isfinite(area_cm2) or area_cm2 <= 0.0:
        return False

    # histogram + solid-angle per bin
    theta_edges = np.linspace(0.0, np.pi / 2.0, 31)
    theta_centers = 0.5 * (theta_edges[1:] + theta_edges[:-1])
    counts, _ = np.histogram(theta, bins=theta_edges)
    delta_omega = 2.0 * np.pi * (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))
    valid = (counts > 0) & np.isfinite(delta_omega) & (delta_omega > 0.0)
    if not np.any(valid):
        return False

    theta_deg = np.degrees(theta_centers[valid])
    delta_omega_valid = delta_omega[valid]
    counts_valid = counts[valid].astype(float)
    dndt_dadomega = counts_valid / (duration_min * area_cm2 * delta_omega_valid)
    dndt_dadomega_err = np.sqrt(counts_valid) / (duration_min * area_cm2 * delta_omega_valid)

    # plot (same layout as STEP_1/STEP_10)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.5, 8.8), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.1]})
    ax_top.errorbar(theta_deg, dndt_dadomega, yerr=dndt_dadomega_err, fmt="o", markersize=4, linewidth=1.0, color="black", ecolor="gray", capsize=2,
                    label=(f"Sample (N={theta.size}, T={duration_min:.3f} min, Axy={area_cm2:.2f} cm^2)"))

    # overlay theory from param_mesh.csv (use sample_path to select row)
    try:
        param_path = Path(__file__).resolve().parents[3] / "INTERSTEPS" / "STEP_0_TO_1" / "param_mesh.csv"
        cos_n = 2.0
        flux_F = None
        if param_path.exists():
            pm = pd.read_csv(param_path)
            step1_id = None
            if sample_path is not None:
                try:
                    if sample_path.name.endswith('.chunks.json'):
                        meta = json.loads(sample_path.read_text()).get('metadata', {})
                    else:
                        meta_path = sample_path.with_suffix(sample_path.suffix + '.meta.json')
                        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                except Exception:
                    meta = {}
                cfg = meta.get('config', {}) if isinstance(meta, dict) else {}
                step1_id = cfg.get('step_1_id') or meta.get('step_1_id')
            rows = pm[pm['step_1_id'].astype(str) == str(step1_id)] if step1_id is not None else pm
            if rows.empty:
                rows = pm
            row = rows.iloc[0]
            cos_n = float(row.get('cos_n', 2.0))
            flux_F = float(row.get('flux_cm2_min', float('nan')))
        if flux_F is None or not np.isfinite(flux_F):
            flux_F = None
        if flux_F is not None:
            n_fixed = float(cos_n)
            i0_fixed = float(flux_F * (n_fixed + 1.0) / (2.0 * math.pi))
            theta_curve = np.linspace(0.0, np.pi / 2.0 - 1e-4, 400)
            cos_curve = np.clip(np.cos(theta_curve), 1e-12, 1.0)
            model_curve = i0_fixed * np.power(cos_curve, n_fixed)
            fit_label = (r"Theory: $I(\\theta)=I_0\\cdot\\cos^{n}(\\theta)$" + "\\n" + rf"$I_0={i0_fixed:.4g}$ min$^{{-1}}$ cm$^{{-2}}$ sr$^{{-1}}$, $n={n_fixed:.2f}$")
            ax_top.plot(np.degrees(theta_curve), model_curve, color="crimson", linewidth=1.6, label=fit_label)
    except Exception:
        pass

    cumulative_flux = np.cumsum(dndt_dadomega * delta_omega_valid)
    cumulative_flux_err = np.sqrt(np.cumsum((dndt_dadomega_err * delta_omega_valid) ** 2))
    theta_upper_deg = np.degrees(theta_edges[1:])[valid]
    ax_bottom.step(theta_upper_deg, cumulative_flux, where="post", color="navy", linewidth=1.4, label=r"Data cumulative: $\int_0^\theta I(\theta')\,d\Omega$")
    ax_bottom.fill_between(theta_upper_deg, np.maximum(0.0, cumulative_flux - cumulative_flux_err), cumulative_flux + cumulative_flux_err, step="post", color="steelblue", alpha=0.25, label="Data 1-sigma band")

    # dashed horizontal y-line for flux parameter (if available)
    if 'flux_F' in locals() and flux_F is not None:
        try:
            flux_param = float(flux_F)
        except Exception:
            flux_param = None
        if flux_param is not None and np.isfinite(flux_param):
            ax_bottom.axhline(flux_param, color="crimson", linestyle="--", linewidth=1.2, label=f"F (flux_cm2_min) = {flux_param:.4g} min^-1 cm^-2")

    # plot theory cumulative if theory known
    try:
        if 'i0_fixed' in locals() and 'n_fixed' in locals():
            theta_curve = np.linspace(0.0, np.pi / 2.0 - 1e-4, 400)
            cum_model = (2.0 * np.pi * i0_fixed / (n_fixed + 1.0) * (1.0 - np.power(np.clip(np.cos(theta_curve), 0.0, 1.0), n_fixed + 1.0)))
            ax_bottom.plot(np.degrees(theta_curve), cum_model, color="crimson", linewidth=1.5, label="Theory cumulative from I0*cos^n(theta)")
    except Exception:
        pass

    ax_top.set_title("Muon differential flux vs zenith angle (area-normalized)")
    ax_top.set_ylabel("I(theta) [min^-1 cm^-2 sr^-1]")
    ax_top.set_xlim(0.0, 90.0)
    ax_top.set_ylim(bottom=0.0)
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc="upper right")

    ax_bottom.set_xlabel("Zenith angle (deg)")
    ax_bottom.set_ylabel("Accumulated flux\n[min^-1 cm^-2]")
    ax_bottom.set_xlim(0.0, 90.0)
    ax_bottom.set_ylim(bottom=0.0)
    ax_bottom.grid(alpha=0.3)
    ax_bottom.legend(loc="upper left")

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return True


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
        sample_path=sample,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
