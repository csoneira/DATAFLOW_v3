#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_2_INFERENCE/STEP_2_1_ESTIMATE_PARAMS/estimate_and_plot.py
Purpose: STEP 2.1 — Inverse problem: estimate physics parameters from feature vectors.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-11
Runtime: python3
Usage: python3 .../estimate_and_plot.py [--config CONFIG] [--dictionary-csv PATH] [--dataset-csv PATH]
Inputs: dictionary.csv and dataset.csv from STEP 1.4, selected_feature_columns.json.
Outputs: estimated_params.csv, estimation_summary.json, diagnostic plots.
Notes: Simplified single-stage kNN with IDW in z-scored feature space.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = STEP_DIR.parent
PIPELINE_DIR = INFERENCE_DIR.parent
PROJECT_DIR = PIPELINE_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "config_method.json"

DEFAULT_DICTIONARY = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dictionary.csv"
)
DEFAULT_DATASET = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dataset.csv"
)
DEFAULT_FEATURE_COLUMNS = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY"
    / "OUTPUTS" / "FILES" / "selected_feature_columns.json"
)
DEFAULT_DISTANCE_DEFINITION = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_5_TUNE_DISTANCE_DEFINITION"
    / "OUTPUTS" / "FILES" / "distance_definition.json"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PARAM_COLUMNS = ["flux_cm2_min", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"]

logging.basicConfig(
    format="[%(levelname)s] STEP_2.1 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_2.1")

_FIGURE_COUNTER = 0
FIGURE_STEP_PREFIX = "2_1"

_PLOT_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".eps",
    ".tif", ".tiff", ".bmp", ".gif", ".webp",
}


def _save_figure(fig: plt.Figure, path: Path, **kwargs) -> None:
    global _FIGURE_COUNTER
    _FIGURE_COUNTER += 1
    out_path = path.with_name(f"{FIGURE_STEP_PREFIX}_{_FIGURE_COUNTER}_{path.name}")
    fig.savefig(out_path, **kwargs)


def _clear_plots_dir() -> None:
    removed = 0
    for f in PLOTS_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in _PLOT_EXTENSIONS:
            f.unlink()
            removed += 1
    log.info("Cleared %d old plot(s) from %s", removed, PLOTS_DIR)


# =====================================================================
# Core estimation
# =====================================================================

def _load_feature_columns(path: Path) -> list[str]:
    """Load feature column list from the step 1.4 artifact."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cols = data.get("selected_feature_columns", [])
    if not cols:
        raise ValueError(f"No feature columns in {path}")
    return cols


def _load_config(path: Path) -> dict:
    """Load and merge method + plots + runtime configs."""
    cfg: dict = {}
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    for extra_name in ("config_plots.json", "config_runtime.json"):
        extra = path.with_name(extra_name)
        if extra.exists() and extra != path:
            extra_cfg = json.loads(extra.read_text(encoding="utf-8"))
            cfg.update(extra_cfg)
    return cfg


def estimate(
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    feature_cols: list[str],
    param_cols: list[str],
    *,
    k: int = 10,
    idw_power: float = 2.0,
    ridge_lambda: float = 1e6,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    p_norm: float = 2.0,
) -> pd.DataFrame:
    """
    Single-stage kNN estimation with locally weighted linear regression
    (ridge) in weighted Lp feature space.

    When ridge_lambda >= 1e5, falls back to pure IDW² weighted mean.
    When ridge_lambda is smaller, fits a local hyperplane at each query
    point using the k nearest dictionary entries, with IDW² observation
    weights and Tikhonov regularization.

    Distance definition (center, scale, weights, p_norm) comes from the
    distance_definition.json artifact produced by step 1.5.
    """
    # Validate columns
    missing_feat_dict = [c for c in feature_cols if c not in dict_df.columns]
    missing_feat_data = [c for c in feature_cols if c not in data_df.columns]
    if missing_feat_dict:
        raise ValueError(f"Feature columns missing in dictionary: {missing_feat_dict}")
    if missing_feat_data:
        raise ValueError(f"Feature columns missing in dataset: {missing_feat_data}")

    available_param_cols = [c for c in param_cols if c in dict_df.columns]
    if not available_param_cols:
        raise ValueError(f"No parameter columns found in dictionary. Looked for: {param_cols}")

    # Extract feature matrices
    dict_feat = dict_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    data_feat = data_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Normalize using distance definition from step 1.4
    if center is not None and scale is not None:
        dict_z = (dict_feat - center) / scale
        data_z = (data_feat - center) / scale
    else:
        # Fallback: standard z-score from dictionary
        means = np.nanmean(dict_feat, axis=0)
        stds = np.nanstd(dict_feat, axis=0)
        stds[stds < 1e-15] = np.nan
        dict_z = (dict_feat - means) / stds
        data_z = (data_feat - means) / stds

    # Per-feature weights (default uniform)
    feat_w = weights if weights is not None else np.ones(len(feature_cols), dtype=float)
    use_local_linear = ridge_lambda < 1e5
    n_feat = len(feature_cols)

    # Parameter arrays
    dict_params = {pc: pd.to_numeric(dict_df[pc], errors="coerce").to_numpy(dtype=float)
                   for pc in available_param_cols}

    # param_set_id for exclusion
    has_param_set_id = "param_set_id" in dict_df.columns and "param_set_id" in data_df.columns
    if has_param_set_id:
        dict_psid = dict_df["param_set_id"].to_numpy()
        data_psid = data_df["param_set_id"].to_numpy()
    else:
        log.warning("No param_set_id column — no same-parameter-set exclusion applied.")

    n_dict = len(dict_df)
    n_data = len(data_df)
    results = []

    for i in range(n_data):
        row = {"dataset_index": i}

        # Build candidate mask: exclude same param_set_id
        mask = np.ones(n_dict, dtype=bool)
        if has_param_set_id:
            mask &= dict_psid != data_psid[i]

        cand_idx = np.where(mask)[0]
        row["n_candidates"] = len(cand_idx)

        if len(cand_idx) == 0:
            for pc in available_param_cols:
                row[f"est_{pc}"] = np.nan
            row["best_distance"] = np.nan
            results.append(row)
            continue

        # Distance in weighted Lp feature space
        sample = data_z[i]
        candidates = dict_z[cand_idx]
        valid = np.isfinite(sample)[None, :] & np.isfinite(candidates)
        diff_abs = np.where(valid, np.abs(candidates - sample[None, :]), 0.0)
        if p_norm == 1.0:
            distances = np.sum(feat_w * diff_abs, axis=1)
        elif p_norm == 2.0:
            distances = np.sqrt(np.sum(feat_w * diff_abs * diff_abs, axis=1))
        else:
            distances = np.power(
                np.sum(feat_w * np.power(diff_abs, p_norm), axis=1), 1.0 / p_norm
            )
        n_valid_dims = np.sum(valid, axis=1)
        distances[n_valid_dims < 2] = np.nan

        finite = np.isfinite(distances)
        if not np.any(finite):
            for pc in available_param_cols:
                row[f"est_{pc}"] = np.nan
            row["best_distance"] = np.nan
            results.append(row)
            continue

        # Select k nearest
        finite_idx = np.where(finite)[0]
        finite_dist = distances[finite_idx]
        order = np.argsort(finite_dist)
        k_use = min(k, len(order))
        top_local = order[:k_use]
        top_idx = cand_idx[finite_idx[top_local]]
        top_dist = finite_dist[top_local]

        row["best_distance"] = float(top_dist[0])
        row["n_neighbors_used"] = k_use

        # IDW weights
        eps = 1e-12
        idw_w = 1.0 / np.power(np.maximum(top_dist, eps), idw_power)
        idw_w /= np.sum(idw_w)

        # Estimation: local-linear ridge or pure IDW weighted mean
        if use_local_linear:
            # Build design matrix [1, x - x₀] once for all parameters
            xi = dict_z[top_idx] - sample[None, :]            # (k_use, n_feat)
            Z = np.column_stack([np.ones(k_use), xi])          # (k_use, n_feat+1)
            reg = np.zeros(n_feat + 1)
            reg[1:] = ridge_lambda
            ZtW = Z.T * idw_w[np.newaxis, :]                  # (n_feat+1, k_use)
            A = ZtW @ Z + np.diag(reg)                        # (n_feat+1, n_feat+1)

        for pc in available_param_cols:
            vals = dict_params[pc][top_idx]
            fin = np.isfinite(vals)
            if not np.any(fin):
                row[f"est_{pc}"] = np.nan
                continue

            if use_local_linear and np.all(fin):
                b_vec = ZtW @ vals                             # (n_feat+1,)
                try:
                    theta = np.linalg.solve(A, b_vec)
                    row[f"est_{pc}"] = float(theta[0])         # intercept
                except np.linalg.LinAlgError:
                    row[f"est_{pc}"] = float(np.dot(idw_w, vals))
            else:
                w = idw_w.copy()
                w[~fin] = 0.0
                ws = np.sum(w)
                if ws > 0:
                    w /= ws
                row[f"est_{pc}"] = float(np.sum(w * np.nan_to_num(vals)))

        if (i + 1) % 200 == 0 or i == n_data - 1:
            log.info("  Estimated %d / %d", i + 1, n_data)

        results.append(row)

    return pd.DataFrame(results)


# =====================================================================
# Diagnostic plots
# =====================================================================

def _plot_true_vs_estimated(result_df: pd.DataFrame, param_cols: list[str]) -> None:
    """Scatter plot of true vs estimated for each parameter, in a single column."""
    plot_cols = [pc for pc in param_cols
                 if f"est_{pc}" in result_df.columns and f"true_{pc}" in result_df.columns]
    if not plot_cols:
        return

    n = len(plot_cols)
    fig, axes = plt.subplots(n, 1, figsize=(5, 4.5 * n), squeeze=False)

    for i, pc in enumerate(plot_cols):
        ax = axes[i, 0]
        t = pd.to_numeric(result_df[f"true_{pc}"], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(result_df[f"est_{pc}"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(t) & np.isfinite(e)
        if not np.any(m):
            ax.set_title(pc, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.scatter(t[m], e[m], s=8, alpha=0.4, edgecolors="none")
        lo = min(float(np.nanmin(t[m])), float(np.nanmin(e[m])))
        hi = max(float(np.nanmax(t[m])), float(np.nanmax(e[m])))
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"True {pc}", fontsize=8)
        ax.set_ylabel(f"Estimated {pc}", fontsize=8)
        ax.set_title(pc, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15)

        # Compute and show RMSE and MAE
        rmse = float(np.sqrt(np.mean((t[m] - e[m]) ** 2)))
        mae = float(np.mean(np.abs(t[m] - e[m])))
        ax.text(0.03, 0.97, f"RMSE={rmse:.4g}\nMAE={mae:.4g}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("True vs Estimated parameters", fontsize=11, y=1.0)
    fig.tight_layout()
    _save_figure(fig, PLOTS_DIR / "true_vs_estimated.png", dpi=150)
    plt.close(fig)


def _plot_distance_distribution(result_df: pd.DataFrame) -> None:
    """Histogram of best-match distances."""
    d = pd.to_numeric(result_df["best_distance"], errors="coerce").to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d, bins=50, color="#4C78A8", edgecolor="none", alpha=0.7)
    ax.set_xlabel("Best-match distance (normalized feature space)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Distribution of best-match distances", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15)

    med = float(np.median(d))
    ax.axvline(med, color="red", ls="--", lw=1, label=f"Median = {med:.3g}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save_figure(fig, PLOTS_DIR / "distance_distribution.png", dpi=150)
    plt.close(fig)


def _plot_error_vs_distance(result_df: pd.DataFrame, param_cols: list[str]) -> None:
    """Scatter of estimation error vs best distance for each parameter."""
    plot_cols = [pc for pc in param_cols
                 if f"est_{pc}" in result_df.columns and f"true_{pc}" in result_df.columns]
    if not plot_cols:
        return

    d = pd.to_numeric(result_df["best_distance"], errors="coerce").to_numpy(dtype=float)
    n = len(plot_cols)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3.5 * n), squeeze=False)

    for i, pc in enumerate(plot_cols):
        ax = axes[i, 0]
        t = pd.to_numeric(result_df[f"true_{pc}"], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(result_df[f"est_{pc}"], errors="coerce").to_numpy(dtype=float)
        err = np.abs(t - e)
        m = np.isfinite(d) & np.isfinite(err)
        if not np.any(m):
            ax.set_title(pc, fontsize=9)
            continue

        ax.scatter(d[m], err[m], s=8, alpha=0.4, edgecolors="none")
        ax.set_xlabel("Best-match distance", fontsize=8)
        ax.set_ylabel(f"|Error| in {pc}", fontsize=8)
        ax.set_title(pc, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)

    fig.suptitle("Estimation error vs best-match distance", fontsize=10, y=1.0)
    fig.tight_layout()
    _save_figure(fig, PLOTS_DIR / "error_vs_distance.png", dpi=150)
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 2.1: Estimate parameters using dictionary matching."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--dataset-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    _clear_plots_dir()

    cfg_21 = config.get("step_2_1", {}) or {}

    # Resolve paths
    dict_path = Path(args.dictionary_csv) if args.dictionary_csv else DEFAULT_DICTIONARY
    data_path = Path(args.dataset_csv) if args.dataset_csv else DEFAULT_DATASET

    if not dict_path.exists():
        log.error("Dictionary not found: %s", dict_path)
        return 1
    if not data_path.exists():
        log.error("Dataset not found: %s", data_path)
        return 1

    # Load data
    dict_df = pd.read_csv(dict_path, low_memory=False)
    data_df = pd.read_csv(data_path, low_memory=False)

    if dict_df.empty:
        log.error("Dictionary is empty: %s", dict_path)
        return 1
    if data_df.empty:
        log.error("Dataset is empty: %s", data_path)
        return 1

    # Load feature columns from step 1.4 artifact
    if not DEFAULT_FEATURE_COLUMNS.exists():
        log.error("Feature columns artifact not found: %s", DEFAULT_FEATURE_COLUMNS)
        return 1
    feature_cols = _load_feature_columns(DEFAULT_FEATURE_COLUMNS)

    # Filter to columns present in both dictionary and dataset
    feat_common = [c for c in feature_cols if c in dict_df.columns and c in data_df.columns]
    dropped = [c for c in feature_cols if c not in feat_common]
    if dropped:
        log.warning("Dropped %d feature column(s) not in both dict/dataset: %s", len(dropped), dropped)
    if not feat_common:
        log.error("No common feature columns between dictionary and dataset.")
        return 1
    feature_cols = feat_common

    # Detect parameter columns
    param_cols = [c for c in PARAM_COLUMNS if c in dict_df.columns]
    if not param_cols:
        log.error("No parameter columns found in dictionary.")
        return 1

    # Config knobs (with sensible defaults)
    k = int(cfg_21.get("inverse_mapping", {}).get("neighbor_count", 10))
    idw_power = float(cfg_21.get("inverse_mapping", {}).get("inverse_distance_power", 2.0))
    ridge_lambda = 1e6  # default: pure IDW (no local-linear)

    # Load distance definition from step 1.5 artifact
    dist_center: np.ndarray | None = None
    dist_scale: np.ndarray | None = None
    dist_weights: np.ndarray | None = None
    dist_p_norm: float = 2.0
    dist_mode_name: str = "l2_standard_zscore_fallback"
    if DEFAULT_DISTANCE_DEFINITION.exists():
        dist_def = json.loads(DEFAULT_DISTANCE_DEFINITION.read_text(encoding="utf-8"))
        dd_cols = dist_def.get("feature_columns", [])
        dd_center = np.asarray(dist_def["center"], dtype=float)
        dd_scale = np.asarray(dist_def["scale"], dtype=float)
        dd_weights = np.asarray(dist_def.get("weights", [1.0] * len(dd_cols)), dtype=float)
        if list(dd_cols) == list(feature_cols) and len(dd_center) == len(feature_cols):
            dist_center = dd_center
            dist_scale = dd_scale
            dist_weights = dd_weights
            dist_p_norm = float(dist_def.get("p_norm", 2.0))
            dist_mode_name = dist_def.get("selected_mode", "unknown")
            n_active = int(np.sum(dd_weights > 0))
            # Override k and lambda from artifact if available
            if "optimal_k" in dist_def:
                k = int(dist_def["optimal_k"])
            if "optimal_lambda" in dist_def:
                ridge_lambda = float(dist_def["optimal_lambda"])
            regression_mode = "local-linear ridge" if ridge_lambda < 1e5 else "IDW²"
            log.info(
                "Loaded distance definition from step 1.5: %s (p=%.1f, k=%d, λ=%.0e [%s], %d/%d active features)",
                dist_mode_name, dist_p_norm, k, ridge_lambda, regression_mode, n_active, len(feature_cols),
            )
        else:
            log.warning(
                "Distance definition feature columns don't match (%d vs %d); falling back to standard z-score.",
                len(dd_cols), len(feature_cols),
            )
    else:
        log.warning("Distance definition artifact not found (%s); falling back to standard z-score.", DEFAULT_DISTANCE_DEFINITION)

    log.info("Dictionary: %s (%d rows)", dict_path, len(dict_df))
    log.info("Dataset:    %s (%d rows)", data_path, len(data_df))
    log.info("Feature columns: %d", len(feature_cols))
    log.info("Parameter columns: %s", param_cols)
    log.info("k=%d, IDW power=%.1f, ridge λ=%.0e", k, idw_power, ridge_lambda)

    # ── Run estimation ─────────────────────────────────────────
    result_df = estimate(
        dict_df, data_df, feature_cols, param_cols,
        k=k, idw_power=idw_power, ridge_lambda=ridge_lambda,
        center=dist_center, scale=dist_scale, weights=dist_weights, p_norm=dist_p_norm,
    )

    # Attach truth columns for validation
    truth_cols = [
        "flux_cm2_min", "cos_n",
        "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
        "n_events", "is_dictionary_entry",
        "param_hash_x", "param_set_id",
    ]
    for col in truth_cols:
        if col in data_df.columns:
            result_df[f"true_{col}"] = data_df[col].values[:len(result_df)]

    # Attach filename_base if available
    if "filename_base" in data_df.columns:
        result_df["filename_base"] = data_df["filename_base"].values[:len(result_df)]

    # ── Save ─────────────────────────────────────────────────────────
    out_path = FILES_DIR / "estimated_params.csv"
    result_df.to_csv(out_path, index=False)
    log.info("Wrote: %s (%d rows)", out_path, len(result_df))

    n_ok = int(result_df["best_distance"].notna().sum())
    n_fail = int(result_df["best_distance"].isna().sum())

    summary = {
        "dictionary": str(dict_path),
        "dataset": str(data_path),
        "distance_mode": dist_mode_name,
        "distance_p_norm": dist_p_norm,
        "distance_from_step_1_4": dist_center is not None,
        "k": k,
        "idw_power": idw_power,
        "feature_columns": feature_cols,
        "feature_columns_count": len(feature_cols),
        "parameter_columns": param_cols,
        "total_points": len(result_df),
        "successful_estimates": n_ok,
        "failed_estimates": n_fail,
    }
    with open(FILES_DIR / "estimation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ────────────────────────────────────────────────────────
    _plot_true_vs_estimated(result_df, param_cols)
    _plot_distance_distribution(result_df)
    _plot_error_vs_distance(result_df, param_cols)

    log.info("Done. %d OK, %d failed.", n_ok, n_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
