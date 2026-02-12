#!/usr/bin/env python3
"""Infer flux and efficiency for real detector data using the simulation
dictionary and uncertainty LUT.

This script:
  1. Loads real data from a station's metadata CSV (e.g. MINGO01 TASK_1).
  2. Loads the simulation dictionary built in INFERENCE_DICTIONARY_VALIDATION.
  3. For each data row, finds the best-matching dictionary entry via L2
     distance on the raw trigger-topology counts (raw_tt_*_count).
  4. Reports the inferred flux, efficiency, and uncertainty per data file.
  5. Produces summary plots (flux/efficiency time series, match quality).

Usage:
    python infer_from_dictionary.py [--config config.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODULES_DIR = REPO_ROOT / "INFERENCE_DICTIONARY_VALIDATION" / "MODULES"

if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

from uncertainty_lut import UncertaintyLUT  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


# ── helpers ─────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    """Read JSON config."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_efficiencies(value: object) -> list[float]:
    """Parse a stringified [e1, e2, e3, e4] into floats."""
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(v) for v in value[:4]]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [np.nan] * 4
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
            return [float(v) for v in parsed[:4]]
    return [np.nan] * 4


def select_matching_columns(
    df: pd.DataFrame, prefix: str, suffix: str, exclude: list[str]
) -> list[str]:
    """Select columns matching prefix and suffix, excluding specific ones."""
    cols = [
        c for c in df.columns
        if c.startswith(prefix) and c.endswith(suffix) and c not in exclude
    ]
    return sorted(cols)


def l2_score(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors, skipping NaN."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.inf
    return float(np.sqrt(np.sum((a[mask] - b[mask]) ** 2)))


def chi2_score(a: np.ndarray, b: np.ndarray) -> float:
    """Chi-squared score weighted by max(a, 1)."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.inf
    a_m, b_m = a[mask], b[mask]
    sigma2 = np.maximum(a_m, 1.0)
    return float(np.sum((a_m - b_m) ** 2 / sigma2))


SCORE_FNS = {"l2": l2_score, "chi2": chi2_score}


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector by its sum (fraction representation)."""
    total = np.nansum(v)
    if total > 0:
        return v / total
    return v


def filter_by_z_planes(
    df: pd.DataFrame, z_planes: list[float], tol: float
) -> pd.DataFrame:
    """Keep only dictionary rows whose z_plane_1..4 match the given values."""
    mask = pd.Series(True, index=df.index)
    for i, z in enumerate(z_planes, start=1):
        col = f"z_plane_{i}"
        if col in df.columns:
            mask &= (df[col] - z).abs() <= tol
        else:
            print(f"  WARNING: column {col} not found in dictionary")
    return df[mask].copy()


def find_best_matches(
    data_vec: np.ndarray,
    dict_vecs: np.ndarray,
    score_fn,
    normalize: bool,
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the k best-matching dictionary entries for a single data row.

    Returns (indices, scores) of the top k matches.
    Vectors are assumed to already be scaled (zscore etc.) if needed.
    """
    if normalize:
        data_vec = normalize_vector(data_vec.copy())

    scores = np.full(len(dict_vecs), np.inf)
    for i, dv in enumerate(dict_vecs):
        cand = normalize_vector(dv.copy()) if normalize else dv
        scores[i] = score_fn(data_vec, cand)

    top_k = np.argsort(scores)[:k]
    return top_k, scores[top_k]


def zscore_scale(
    dict_vecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardise columns using dictionary statistics.

    Returns (scaled_dict_vecs, means, stds).
    """
    means = np.nanmean(dict_vecs, axis=0)
    stds = np.nanstd(dict_vecs, axis=0, ddof=1)
    stds[stds == 0] = np.nan
    scaled = (dict_vecs - means) / stds
    return scaled, means, stds


def zscore_apply(vec: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Apply pre-computed z-score scaling to a single vector."""
    return (vec - means) / stds


def idw_estimate(values: np.ndarray, scores: np.ndarray) -> float:
    """Inverse-distance-weighted average from L2 scores."""
    if len(values) == 1:
        return float(values[0])
    # Avoid division by zero
    safe_scores = np.maximum(scores, 1e-12)
    weights = 1.0 / safe_scores ** 2
    weights /= weights.sum()
    return float(np.nansum(weights * values))


# ── main inference ──────────────────────────────────────────────────────────

def run_inference(cfg: dict) -> pd.DataFrame:
    """Run dictionary matching on real station data."""

    # ── load data ───────────────────────────────────────────────────────
    data_path = Path(cfg["data_csv"])
    print(f"Loading data from: {data_path}")
    data_df = pd.read_csv(data_path, low_memory=False)
    print(f"  Data rows: {len(data_df)}")

    # ── load dictionary ─────────────────────────────────────────────────
    dict_path = Path(cfg["dictionary_csv"])
    print(f"Loading dictionary from: {dict_path}")
    dict_df = pd.read_csv(dict_path, low_memory=False)
    print(f"  Dictionary rows (total): {len(dict_df)}")

    # ── filter dictionary by z-planes ───────────────────────────────────
    z_planes = cfg["z_planes"]
    z_tol = cfg.get("z_match_tolerance", 1e-6)
    dict_z = filter_by_z_planes(dict_df, z_planes, z_tol)
    print(f"  Dictionary rows (z-plane matched): {len(dict_z)}")
    if len(dict_z) == 0:
        print("ERROR: No dictionary rows match the specified z-planes.")
        print(f"  Requested: {z_planes}")
        print("  Available z combos in dictionary:")
        zc = dict_df[["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]].drop_duplicates()
        print(zc.to_string(index=False))
        sys.exit(1)

    # ── extract efficiencies from dictionary ────────────────────────────
    effs = dict_z["efficiencies"].apply(parse_efficiencies)
    for i in range(1, 5):
        dict_z[f"eff_{i}"] = effs.apply(lambda x, idx=i - 1: x[idx])

    # ── select matching columns ─────────────────────────────────────────
    prefix = cfg.get("matching_columns_prefix", "raw_tt_")
    suffix = cfg.get("matching_columns_suffix", "_rate_hz")
    exclude = cfg.get("exclude_columns", [])
    match_cols = select_matching_columns(data_df, prefix, suffix, exclude)

    # Ensure columns exist in both
    match_cols = [c for c in match_cols if c in dict_z.columns]

    # ── optionally include global rate ──────────────────────────────────
    global_rate_col = cfg.get("global_rate_col", "")
    include_global = cfg.get("include_global_rate", False)
    if include_global and global_rate_col:
        if global_rate_col in data_df.columns and global_rate_col in dict_z.columns:
            if global_rate_col not in match_cols:
                match_cols.append(global_rate_col)
            print(f"  Including global rate column: {global_rate_col}")
        else:
            print(f"  WARNING: global_rate_col '{global_rate_col}' not found "
                  f"in both data and dictionary")

    print(f"  Matching columns ({len(match_cols)}): {match_cols}")
    print(f"  Common columns: {len(match_cols)}")
    if len(match_cols) == 0:
        print("ERROR: No matching columns found in both data and dictionary.")
        sys.exit(1)

    # ── coerce to numeric ───────────────────────────────────────────────
    for c in match_cols:
        data_df[c] = pd.to_numeric(data_df[c], errors="coerce")
        dict_z[c] = pd.to_numeric(dict_z[c], errors="coerce")

    # Pre-extract dictionary vectors
    dict_vecs_raw = dict_z[match_cols].to_numpy(dtype=float)
    dict_flux = dict_z["flux_cm2_min"].to_numpy(dtype=float)
    dict_eff1 = dict_z["eff_1"].to_numpy(dtype=float)
    dict_eff2 = dict_z["eff_2"].to_numpy(dtype=float)
    dict_eff3 = dict_z["eff_3"].to_numpy(dtype=float)
    dict_eff4 = dict_z["eff_4"].to_numpy(dtype=float)
    dict_cosn = dict_z["cos_n"].to_numpy(dtype=float)

    score_metric = cfg.get("score_metric", "l2")
    score_fn = SCORE_FNS[score_metric]
    normalize = cfg.get("normalize_counts", False)
    interp_k = cfg.get("interpolation_k", 5)
    metric_scale = cfg.get("metric_scale", "none")

    # ── z-score scaling (computed on dictionary, applied to both) ──────
    zs_means, zs_stds = None, None
    if metric_scale == "zscore":
        dict_vecs, zs_means, zs_stds = zscore_scale(dict_vecs_raw)
        print(f"  Z-score scaling applied (computed on {len(dict_vecs)} "
              f"dictionary rows)")
    else:
        dict_vecs = dict_vecs_raw

    # ── load uncertainty LUT ────────────────────────────────────────────
    lut_dir = cfg.get("lut_dir")
    lut = None
    if lut_dir and Path(lut_dir).exists():
        try:
            lut = UncertaintyLUT.load(lut_dir)
            print(f"  Uncertainty LUT loaded: {lut}")
        except Exception as e:
            print(f"  WARNING: Could not load LUT: {e}")

    # ── run matching ────────────────────────────────────────────────────
    print(f"\nRunning inference (metric={score_metric}, k={interp_k}, "
          f"normalize={normalize}, scale={metric_scale})...")

    results = []
    for row_idx in range(len(data_df)):
        row = data_df.iloc[row_idx]
        data_vec_raw = row[match_cols].to_numpy(dtype=float)

        # Skip rows where all matching values are NaN
        if np.all(~np.isfinite(data_vec_raw)):
            continue

        # Apply z-score scaling using dictionary statistics
        if metric_scale == "zscore" and zs_means is not None:
            data_vec = zscore_apply(data_vec_raw, zs_means, zs_stds)
        else:
            data_vec = data_vec_raw

        top_k_idx, top_k_scores = find_best_matches(
            data_vec, dict_vecs, score_fn, normalize, k=interp_k
        )

        # IDW-interpolated estimates
        est_flux = idw_estimate(dict_flux[top_k_idx], top_k_scores)
        est_eff1 = idw_estimate(dict_eff1[top_k_idx], top_k_scores)
        est_eff2 = idw_estimate(dict_eff2[top_k_idx], top_k_scores)
        est_eff3 = idw_estimate(dict_eff3[top_k_idx], top_k_scores)
        est_eff4 = idw_estimate(dict_eff4[top_k_idx], top_k_scores)
        est_cosn = idw_estimate(dict_cosn[top_k_idx], top_k_scores)

        best_score = float(top_k_scores[0])

        # Event count for LUT query: use raw_tt count columns if available,
        # otherwise fall back to summing rate values
        n_events = np.nan
        tt_count_cols = [c for c in data_df.columns
                         if c.startswith("raw_tt_") and c.endswith("_count")]
        if tt_count_cols:
            counts = pd.to_numeric(row[tt_count_cols], errors="coerce")
            n_events = float(counts.sum())
        elif global_rate_col and global_rate_col in data_df.columns:
            # Rough estimate from global rate × measurement time
            rate = pd.to_numeric(row.get(global_rate_col), errors="coerce")
            denom_col = "count_rate_denominator_seconds"
            if denom_col in data_df.columns:
                secs = pd.to_numeric(row.get(denom_col), errors="coerce")
                if np.isfinite(rate) and np.isfinite(secs):
                    n_events = float(rate * secs)
        if not np.isfinite(n_events):
            n_events = float(np.nansum(data_vec_raw))

        # Uncertainty from LUT
        sigma_flux_pct = np.nan
        sigma_eff_pct = np.nan
        if lut is not None:
            sigma_flux_pct, sigma_eff_pct = lut.query(est_flux, est_eff1, n_events)

        rec = {
            "data_row_index": row_idx,
            "filename_base": row.get("filename_base", f"row_{row_idx}"),
            "execution_timestamp": row.get("execution_timestamp", ""),
            "n_events": n_events,
            "best_score": best_score,
            "estimated_flux_cm2_min": est_flux,
            "estimated_cos_n": est_cosn,
            "estimated_eff_1": est_eff1,
            "estimated_eff_2": est_eff2,
            "estimated_eff_3": est_eff3,
            "estimated_eff_4": est_eff4,
            "sigma_flux_pct": sigma_flux_pct,
            "sigma_eff_pct": sigma_eff_pct,
            "sigma_flux_abs": abs(est_flux) * sigma_flux_pct / 100.0
            if np.isfinite(sigma_flux_pct)
            else np.nan,
            "sigma_eff_abs": abs(est_eff1) * sigma_eff_pct / 100.0
            if np.isfinite(sigma_eff_pct)
            else np.nan,
        }
        results.append(rec)

    results_df = pd.DataFrame(results)
    print(f"\nInference complete: {len(results_df)} rows processed.")
    return results_df


# ── plotting ────────────────────────────────────────────────────────────────

def plot_time_series(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot flux and efficiency estimates as time series."""
    if results_df.empty:
        return

    x = np.arange(len(results_df))
    labels = results_df["filename_base"].tolist()
    tick_step = max(1, len(x) // 10)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ── Flux ────────────────────────────────────────────────────────────
    ax = axes[0]
    flux = results_df["estimated_flux_cm2_min"]
    sigma_flux = results_df["sigma_flux_abs"]
    ax.plot(x, flux, "o-", ms=4, color="#4C78A8", label="Estimated flux")
    if sigma_flux.notna().any():
        ax.fill_between(
            x,
            flux - sigma_flux,
            flux + sigma_flux,
            alpha=0.2,
            color="#4C78A8",
            label="±1σ uncertainty",
        )
    ax.set_ylabel("Flux [cm⁻² min⁻¹]")
    ax.set_title("Inferred Cosmic-Ray Flux (MINGO01, TASK_1)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Efficiency ──────────────────────────────────────────────────────
    ax = axes[1]
    markers = ["o", "s", "^", "D"]
    colors = ["#E45756", "#72B7B2", "#54A24B", "#EECA3B"]
    for plane in range(1, 5):
        col = f"estimated_eff_{plane}"
        ax.plot(
            x,
            results_df[col],
            marker=markers[plane - 1],
            ms=3,
            color=colors[plane - 1],
            label=f"Plane {plane}",
            linewidth=1,
        )
    ax.set_ylabel("Efficiency")
    ax.set_title("Inferred Detector Efficiency per Plane")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Match quality ───────────────────────────────────────────────────
    ax = axes[2]
    ax.semilogy(x, results_df["best_score"], "o-", ms=4, color="#B279A2")
    ax.set_ylabel(f"Best match score (lower = better)")
    ax.set_xlabel("Data file index")
    ax.set_title("Dictionary Match Quality")
    ax.grid(True, alpha=0.3)

    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(
        [labels[i] for i in range(0, len(labels), tick_step)],
        rotation=45,
        ha="right",
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig(out_dir / "inference_time_series.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_dir / 'inference_time_series.png'}")


def plot_flux_histogram(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of inferred flux values."""
    if results_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        results_df["estimated_flux_cm2_min"].dropna(),
        bins=20,
        color="#4C78A8",
        alpha=0.85,
        edgecolor="white",
    )
    mean_f = results_df["estimated_flux_cm2_min"].mean()
    std_f = results_df["estimated_flux_cm2_min"].std()
    ax.axvline(mean_f, color="red", ls="--", lw=1.5,
               label=f"Mean = {mean_f:.4f}")
    ax.set_xlabel("Flux [cm⁻² min⁻¹]")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Inferred Flux (μ={mean_f:.4f}, σ={std_f:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "inference_flux_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_dir / 'inference_flux_histogram.png'}")


def plot_summary_dashboard(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Compact 2x2 dashboard with key results."""
    if results_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # [0,0] Flux time series
    ax = axes[0, 0]
    x = np.arange(len(results_df))
    flux = results_df["estimated_flux_cm2_min"]
    sigma = results_df["sigma_flux_abs"]
    ax.plot(x, flux, "o-", ms=3, color="#4C78A8")
    if sigma.notna().any():
        ax.fill_between(x, flux - sigma, flux + sigma, alpha=0.2, color="#4C78A8")
    ax.set_ylabel("Flux [cm⁻² min⁻¹]")
    ax.set_title("Inferred Flux Time Series")
    ax.grid(True, alpha=0.3)

    # [0,1] Flux histogram
    ax = axes[0, 1]
    ax.hist(flux.dropna(), bins=20, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.axvline(flux.mean(), color="red", ls="--", lw=1.5,
               label=f"μ={flux.mean():.4f}")
    ax.set_xlabel("Flux")
    ax.set_title("Flux Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # [1,0] Efficiency per plane
    ax = axes[1, 0]
    colors = ["#E45756", "#72B7B2", "#54A24B", "#EECA3B"]
    for plane in range(1, 5):
        col = f"estimated_eff_{plane}"
        ax.plot(x, results_df[col], "o-", ms=3, color=colors[plane - 1],
                label=f"P{plane}")
    ax.set_ylabel("Efficiency")
    ax.set_xlabel("File index")
    ax.set_title("Efficiency per Plane")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # [1,1] Match quality
    ax = axes[1, 1]
    ax.semilogy(x, results_df["best_score"], "o-", ms=3, color="#B279A2")
    ax.set_ylabel("Score")
    ax.set_xlabel("File index")
    ax.set_title("Match Quality (L2 score)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("MINGO01 — Dictionary Inference Summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "inference_dashboard.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_dir / 'inference_dashboard.png'}")


# ── eff vs flux L2 landscape ────────────────────────────────────────────────

def plot_eff_vs_flux_l2(
    results_df: pd.DataFrame, cfg: dict, out_dir: Path
) -> None:
    """Plot dictionary entries in (flux, eff) space, coloured by L2 score
    to the median data fingerprint.  Overlay estimated data points with
    LUT uncertainty ellipses."""
    from matplotlib.patches import Ellipse
    from matplotlib.collections import PatchCollection

    if results_df.empty:
        return

    # ── reload dictionary (z-filtered) ──────────────────────────────────
    dict_df = pd.read_csv(cfg["dictionary_csv"], low_memory=False)
    dict_z = filter_by_z_planes(
        dict_df, cfg["z_planes"], cfg.get("z_match_tolerance", 1e-6)
    )
    effs = dict_z["efficiencies"].apply(parse_efficiencies)
    for i in range(1, 5):
        dict_z[f"eff_{i}"] = effs.apply(lambda x, idx=i - 1: x[idx])

    # ── reload data ─────────────────────────────────────────────────────
    data_df = pd.read_csv(cfg["data_csv"], low_memory=False)
    prefix = cfg.get("matching_columns_prefix", "raw_tt_")
    suffix = cfg.get("matching_columns_suffix", "_count")
    exclude = cfg.get("exclude_columns", [])
    match_cols = select_matching_columns(data_df, prefix, suffix, exclude)
    match_cols = [c for c in match_cols if c in dict_z.columns]

    for c in match_cols:
        data_df[c] = pd.to_numeric(data_df[c], errors="coerce")
        dict_z[c] = pd.to_numeric(dict_z[c], errors="coerce")

    normalize = cfg.get("normalize_counts", True)
    score_fn = SCORE_FNS[cfg.get("score_metric", "l2")]

    # Compute median data fingerprint
    data_vecs = data_df[match_cols].to_numpy(dtype=float)
    median_vec = np.nanmedian(data_vecs, axis=0)
    if normalize:
        median_vec = normalize_vector(median_vec.copy())

    dict_vecs = dict_z[match_cols].to_numpy(dtype=float)
    dict_flux = dict_z["flux_cm2_min"].to_numpy(dtype=float)
    dict_eff1 = dict_z["eff_1"].to_numpy(dtype=float)

    # L2 of each dictionary entry to the median data fingerprint
    l2_scores = np.array([
        score_fn(
            median_vec,
            normalize_vector(dv.copy()) if normalize else dv,
        )
        for dv in dict_vecs
    ])

    # ── plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))

    # Dictionary entries coloured by log10(L2)
    finite_mask = np.isfinite(l2_scores) & (l2_scores > 0)
    log_scores = np.full_like(l2_scores, np.nan)
    log_scores[finite_mask] = np.log10(l2_scores[finite_mask])

    sc = ax.scatter(
        dict_flux[finite_mask],
        dict_eff1[finite_mask],
        c=log_scores[finite_mask],
        cmap="viridis_r",
        s=30,
        alpha=0.6,
        edgecolors="none",
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("log₁₀(L2 score to median data)", fontsize=10)

    # Estimated data points with uncertainty ellipses
    est_flux = results_df["estimated_flux_cm2_min"].to_numpy()
    est_eff  = results_df["estimated_eff_1"].to_numpy()
    sig_flux = results_df["sigma_flux_abs"].to_numpy()
    sig_eff  = results_df["sigma_eff_abs"].to_numpy()

    # 1σ and 2σ ellipses
    ellipses_2s = []
    ellipses_1s = []
    for i in range(len(results_df)):
        sf = sig_flux[i] if np.isfinite(sig_flux[i]) else 0
        se = sig_eff[i]  if np.isfinite(sig_eff[i])  else 0
        if sf > 0 and se > 0:
            ellipses_2s.append(
                Ellipse((est_flux[i], est_eff[i]), 4 * sf, 4 * se)
            )
            ellipses_1s.append(
                Ellipse((est_flux[i], est_eff[i]), 2 * sf, 2 * se)
            )

    if ellipses_2s:
        pc2 = PatchCollection(
            ellipses_2s, facecolor="red", alpha=0.08,
            edgecolor="red", linewidth=0.5, zorder=3,
        )
        ax.add_collection(pc2)
    if ellipses_1s:
        pc1 = PatchCollection(
            ellipses_1s, facecolor="red", alpha=0.15,
            edgecolor="red", linewidth=0.8, zorder=4,
        )
        ax.add_collection(pc1)

    # Data points on top
    ax.scatter(
        est_flux, est_eff,
        c="red", s=50, marker="*", edgecolors="darkred",
        linewidths=0.5, zorder=5, label="MINGO01 estimated",
    )

    # Mark the best-match region
    ax.axvline(
        np.mean(est_flux), color="red", ls=":", lw=0.8, alpha=0.5,
    )
    ax.axhline(
        np.mean(est_eff), color="red", ls=":", lw=0.8, alpha=0.5,
    )

    ax.set_xlabel("Flux [cm⁻² min⁻¹]", fontsize=11)
    ax.set_ylabel("Efficiency (plane 1)", fontsize=11)
    ax.set_title(
        "Dictionary L2 Landscape  —  MINGO01 TASK_1 Estimates"
        "\n(ellipses: 1σ/2σ from uncertainty LUT)",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "inference_eff_vs_flux_l2.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_dir / 'inference_eff_vs_flux_l2.png'}")


# ── diagonal data-vs-dictionary comparison ──────────────────────────────────

def plot_diagonal_comparison(
    results_df: pd.DataFrame, cfg: dict, out_dir: Path
) -> None:
    """Diagonal scatter: MINGO01 data values vs best-match dictionary values
    for each matching column.  Uses the same z-score + metric as inference."""

    if results_df.empty:
        return

    # ── reload dictionary (z-filtered) & data ───────────────────────────
    dict_df = pd.read_csv(cfg["dictionary_csv"], low_memory=False)
    dict_z = filter_by_z_planes(
        dict_df, cfg["z_planes"], cfg.get("z_match_tolerance", 1e-6)
    )

    data_df = pd.read_csv(cfg["data_csv"], low_memory=False)
    prefix = cfg.get("matching_columns_prefix", "raw_tt_")
    suffix = cfg.get("matching_columns_suffix", "_rate_hz")
    exclude = cfg.get("exclude_columns", [])
    match_cols = select_matching_columns(data_df, prefix, suffix, exclude)
    match_cols = [c for c in match_cols if c in dict_z.columns]

    # Include global rate if configured
    global_rate_col = cfg.get("global_rate_col", "")
    include_global = cfg.get("include_global_rate", False)
    if include_global and global_rate_col:
        if (global_rate_col in data_df.columns
                and global_rate_col in dict_z.columns
                and global_rate_col not in match_cols):
            match_cols.append(global_rate_col)

    for c in match_cols:
        data_df[c] = pd.to_numeric(data_df[c], errors="coerce")
        dict_z[c] = pd.to_numeric(dict_z[c], errors="coerce")

    normalize = cfg.get("normalize_counts", False)
    score_fn = SCORE_FNS[cfg.get("score_metric", "l2")]
    metric_scale = cfg.get("metric_scale", "none")

    dict_vecs_raw = dict_z[match_cols].to_numpy(dtype=float)
    dict_z_reset = dict_z.reset_index(drop=True)

    # z-score scaling (same as inference)
    zs_means, zs_stds = None, None
    if metric_scale == "zscore":
        dict_vecs_scaled, zs_means, zs_stds = zscore_scale(dict_vecs_raw)
    else:
        dict_vecs_scaled = dict_vecs_raw

    # ── find best match for each data row (using same method as inference)
    data_vals_all = []   # raw values per row
    dict_vals_all = []   # raw values of best-match dict row
    for row_idx in range(len(data_df)):
        data_vec_raw = data_df.iloc[row_idx][match_cols].to_numpy(dtype=float)
        if np.all(~np.isfinite(data_vec_raw)):
            continue

        if metric_scale == "zscore" and zs_means is not None:
            data_vec = zscore_apply(data_vec_raw, zs_means, zs_stds)
        else:
            data_vec = data_vec_raw

        best_idx, _ = find_best_matches(
            data_vec, dict_vecs_scaled, score_fn, normalize, k=1
        )
        best_dict_row = dict_z_reset.iloc[int(best_idx[0])]
        data_vals_all.append(data_vec_raw)
        dict_vals_all.append(best_dict_row[match_cols].to_numpy(dtype=float))

    data_arr = np.array(data_vals_all)
    dict_arr = np.array(dict_vals_all)

    # Separate tt columns from global rate for labelling
    tt_cols = [c for c in match_cols if c != global_rate_col]
    all_cols_for_plot = match_cols  # includes global rate

    n_cols = len(all_cols_for_plot)
    ncols_grid = min(4, n_cols)
    nrows_grid = int(np.ceil(n_cols / ncols_grid))

    # ── helper: nice short label ────────────────────────────────────────
    def _short_label(col: str) -> str:
        return (col.replace("raw_tt_", "")
                   .replace("_rate_hz", "")
                   .replace("_count", "")
                   .replace("events_per_second_", ""))

    # ── figure 1: individual panels (rate values) ───────────────────────
    fig, axes = plt.subplots(
        nrows_grid, ncols_grid, figsize=(5 * ncols_grid, 4.5 * nrows_grid)
    )
    if nrows_grid == 1 and ncols_grid == 1:
        axes = np.array([[axes]])
    elif nrows_grid == 1 or ncols_grid == 1:
        axes = np.atleast_2d(axes)

    for idx, col in enumerate(all_cols_for_plot):
        row_i, col_i = divmod(idx, ncols_grid)
        ax = axes[row_i, col_i]

        d = data_arr[:, idx]
        m = dict_arr[:, idx]
        mask = np.isfinite(d) & np.isfinite(m)
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        ax.scatter(d[mask], m[mask], s=35, alpha=0.7, color="#4C78A8",
                   edgecolors="white", linewidths=0.4, zorder=3)

        # Diagonal y=x
        all_vals = np.concatenate([d[mask], m[mask]])
        margin = (all_vals.max() - all_vals.min()) * 0.08
        lo = all_vals.min() - margin
        hi = all_vals.max() + margin
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, alpha=0.7, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Stats
        if mask.sum() > 2:
            corr = np.corrcoef(d[mask], m[mask])[0, 1]
            resid = m[mask] - d[mask]
            rel_bias = np.mean(resid) / np.mean(d[mask]) * 100
            ax.text(
                0.05, 0.95,
                f"r = {corr:.3f}\nrel. bias = {rel_bias:+.1f}%",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        short = _short_label(col)
        ax.set_xlabel("MINGO01 data [Hz]", fontsize=9)
        ax.set_ylabel("Dict. best match [Hz]", fontsize=9)
        ax.set_title(short, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

    for idx in range(n_cols, nrows_grid * ncols_grid):
        row_i, col_i = divmod(idx, ncols_grid)
        axes[row_i, col_i].set_visible(False)

    fig.suptitle(
        "Diagonal Comparison — MINGO01 Data vs Best-Match Dictionary (rates)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "inference_diagonal_rates.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_dir / 'inference_diagonal_rates.png'}")

    # ── figure 2: normalised fractions (tt cols only) ───────────────────
    tt_indices = [i for i, c in enumerate(all_cols_for_plot)
                  if c != global_rate_col]
    if tt_indices:
        data_tt = data_arr[:, tt_indices]
        dict_tt = dict_arr[:, tt_indices]
        data_frac = np.array([normalize_vector(row.copy()) for row in data_tt])
        dict_frac = np.array([normalize_vector(row.copy()) for row in dict_tt])

        n_tt = len(tt_indices)
        nc_frac = min(3, n_tt)
        nr_frac = int(np.ceil(n_tt / nc_frac))

        fig2, axes2 = plt.subplots(
            nr_frac, nc_frac, figsize=(5.5 * nc_frac, 4.5 * nr_frac)
        )
        if nr_frac == 1 and nc_frac == 1:
            axes2 = np.array([[axes2]])
        elif nr_frac == 1 or nc_frac == 1:
            axes2 = np.atleast_2d(axes2)

        for idx_i, col_idx in enumerate(tt_indices):
            col = all_cols_for_plot[col_idx]
            row_i, col_i = divmod(idx_i, nc_frac)
            ax = axes2[row_i, col_i]

            d = data_frac[:, idx_i]
            m = dict_frac[:, idx_i]
            mask = np.isfinite(d) & np.isfinite(m)
            if mask.sum() == 0:
                ax.set_visible(False)
                continue

            ax.scatter(d[mask], m[mask], s=35, alpha=0.7, color="#54A24B",
                       edgecolors="white", linewidths=0.4, zorder=3)

            all_vals = np.concatenate([d[mask], m[mask]])
            margin = (all_vals.max() - all_vals.min()) * 0.08
            lo = all_vals.min() - margin
            hi = all_vals.max() + margin
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, alpha=0.7, zorder=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            if mask.sum() > 2:
                corr = np.corrcoef(d[mask], m[mask])[0, 1]
                resid = m[mask] - d[mask]
                rms = np.sqrt(np.mean(resid ** 2))
                ax.text(
                    0.05, 0.95,
                    f"r = {corr:.4f}\nRMS = {rms:.5f}",
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

            short = _short_label(col)
            ax.set_xlabel("MINGO01 (fraction)", fontsize=9)
            ax.set_ylabel("Dict. match (fraction)", fontsize=9)
            ax.set_title(short, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.2)

        for idx_i in range(n_tt, nr_frac * nc_frac):
            row_i, col_i = divmod(idx_i, nc_frac)
            axes2[row_i, col_i].set_visible(False)

        fig2.suptitle(
            "Diagonal Comparison — Data vs Dictionary (rate fractions)",
            fontsize=13,
        )
        fig2.tight_layout()
        fig2.savefig(out_dir / "inference_diagonal_fractions.png", dpi=150)
        plt.close(fig2)
        print(f"  Plot saved: {out_dir / 'inference_diagonal_fractions.png'}")

    # ── figure 3: all tt columns combined in one diagonal ───────────────
    if tt_indices:
        fig3, ax3 = plt.subplots(figsize=(8, 7))
        colors_list = plt.cm.tab10(np.linspace(0, 1, n_tt))

        for idx_i, col_idx in enumerate(tt_indices):
            col = all_cols_for_plot[col_idx]
            d = data_frac[:, idx_i]
            m = dict_frac[:, idx_i]
            mask = np.isfinite(d) & np.isfinite(m)
            short = _short_label(col)
            ax3.scatter(
                d[mask], m[mask], s=30, alpha=0.65, color=colors_list[idx_i],
                label=short, edgecolors="white", linewidths=0.3, zorder=3,
            )

        all_d = data_frac.flatten()
        all_m = dict_frac.flatten()
        fmask = np.isfinite(all_d) & np.isfinite(all_m)
        margin = (max(all_d[fmask].max(), all_m[fmask].max())
                  - min(all_d[fmask].min(), all_m[fmask].min())) * 0.08
        lo = min(all_d[fmask].min(), all_m[fmask].min()) - margin
        hi = max(all_d[fmask].max(), all_m[fmask].max()) + margin
        ax3.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6, label="y = x",
                 zorder=2)
        ax3.set_xlim(lo, hi)
        ax3.set_ylim(lo, hi)

        corr_all = np.corrcoef(all_d[fmask], all_m[fmask])[0, 1]
        ax3.text(
            0.05, 0.95,
            f"Global r = {corr_all:.4f}",
            transform=ax3.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

        ax3.set_xlabel("MINGO01 data (rate fraction)", fontsize=11)
        ax3.set_ylabel("Dictionary best match (rate fraction)", fontsize=11)
        ax3.set_title(
            "All Trigger Topologies — Data vs Dictionary", fontsize=12
        )
        ax3.legend(fontsize=8, ncol=3, loc="lower right")
        ax3.grid(True, alpha=0.2)

        fig3.tight_layout()
        fig3.savefig(out_dir / "inference_diagonal_all.png", dpi=150)
        plt.close(fig3)
        print(f"  Plot saved: {out_dir / 'inference_diagonal_all.png'}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Infer flux/efficiency from real data using the "
                    "simulation dictionary."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to JSON config file."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Run inference
    results_df = run_inference(cfg)

    # Save results
    out_dir = Path(cfg.get("output_dir", SCRIPT_DIR / "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "inference_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    if not results_df.empty:
        flux = results_df["estimated_flux_cm2_min"]
        print(f"  Files processed:    {len(results_df)}")
        print(f"  Flux mean ± std:    {flux.mean():.4f} ± {flux.std():.4f} cm⁻² min⁻¹")
        for p in range(1, 5):
            e = results_df[f"estimated_eff_{p}"]
            print(f"  Eff plane {p}:        {e.mean():.4f} ± {e.std():.4f}")
        if results_df["sigma_flux_pct"].notna().any():
            print(f"  LUT σ_flux (avg):   {results_df['sigma_flux_pct'].mean():.2f}%")
            print(f"  LUT σ_eff  (avg):   {results_df['sigma_eff_pct'].mean():.2f}%")
    print("=" * 60)

    # Generate plots
    print("\nGenerating plots...")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_time_series(results_df, plots_dir)
    plot_flux_histogram(results_df, plots_dir)
    plot_summary_dashboard(results_df, plots_dir)
    plot_eff_vs_flux_l2(results_df, cfg, plots_dir)
    plot_diagonal_comparison(results_df, cfg, plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
