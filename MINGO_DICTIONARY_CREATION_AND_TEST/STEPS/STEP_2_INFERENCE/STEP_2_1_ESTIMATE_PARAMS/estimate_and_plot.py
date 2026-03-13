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
from scipy.spatial import Delaunay

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

# ── Import shared estimation engine ──────────────────────────────────
sys.path.insert(0, str(INFERENCE_DIR))
try:
    from estimate_parameters import (
        estimate_from_dataframes,
        load_distance_definition,
    )
except Exception as exc:
    log.error("Failed to import estimate_parameters from %s: %s", INFERENCE_DIR, exc)
    raise

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


# =====================================================================
# Diagnostic plots
# =====================================================================

def _plot_true_vs_estimated(result_df: pd.DataFrame, param_cols: list[str]) -> None:
    """Scatter plot of true vs estimated for each parameter, coloured by hull membership."""
    plot_cols = [pc for pc in param_cols
                 if f"est_{pc}" in result_df.columns and f"true_{pc}" in result_df.columns]
    if not plot_cols:
        return

    has_hull = "in_coverage" in result_df.columns
    if has_hull:
        in_hull = result_df["in_coverage"].astype(bool).values

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

        if has_hull:
            m_in = m & in_hull
            m_out = m & ~in_hull
            if np.any(m_in):
                ax.scatter(t[m_in], e[m_in], s=8, alpha=0.4, color="#59A14F",
                           edgecolors="none", label="In hull", zorder=1)
            if np.any(m_out):
                ax.scatter(t[m_out], e[m_out], s=12, alpha=0.7, color="#E15759",
                           marker="x", linewidths=0.8, label="Outside hull", zorder=2)
        else:
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
        if has_hull and i == 0:
            ax.legend(fontsize=7, loc="lower right")

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


def _plot_hull_coverage(
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    result_df: pd.DataFrame,
    param_cols: list[str],
) -> None:
    """Lower-triangular pairwise scatter of parameters showing dictionary,
    dataset-inside-hull, and dataset-outside-hull."""
    if "in_coverage" not in result_df.columns:
        return

    in_mask = result_df["in_coverage"].astype(bool).values
    out_mask = ~in_mask

    # Use one row per param_set_id for dataset to avoid overplotting
    data_plot = data_df.copy()
    data_plot["_in_cov"] = in_mask
    if "param_set_id" in data_plot.columns:
        data_plot = data_plot.groupby("param_set_id").first().reset_index()
        in_mask_plot = data_plot["_in_cov"].values
        out_mask_plot = ~in_mask_plot
    else:
        in_mask_plot = in_mask
        out_mask_plot = out_mask

    n = len(param_cols)
    fig, axes = plt.subplots(n - 1, n - 1, figsize=(3.2 * (n - 1), 3.2 * (n - 1)),
                             squeeze=False)

    # Hide upper triangle and diagonal
    for r in range(n - 1):
        for c in range(n - 1):
            if c >= r + 1:
                axes[r, c].set_visible(False)

    for r in range(1, n):
        for c in range(r):
            ax = axes[r - 1, c]
            yp = param_cols[r]
            xp = param_cols[c]
            # Dictionary
            dx = pd.to_numeric(dict_df[xp], errors="coerce").values
            dy = pd.to_numeric(dict_df[yp], errors="coerce").values
            ax.scatter(dx, dy, s=18, alpha=0.5, color="#4C78A8",
                       edgecolors="none", label="Dictionary", zorder=2)
            # Dataset inside hull
            sx_in = pd.to_numeric(data_plot.loc[in_mask_plot, xp], errors="coerce").values
            sy_in = pd.to_numeric(data_plot.loc[in_mask_plot, yp], errors="coerce").values
            ax.scatter(sx_in, sy_in, s=12, alpha=0.4, color="#59A14F",
                       edgecolors="none", label="Dataset (in hull)", zorder=1)
            # Dataset outside hull
            sx_out = pd.to_numeric(data_plot.loc[out_mask_plot, xp], errors="coerce").values
            sy_out = pd.to_numeric(data_plot.loc[out_mask_plot, yp], errors="coerce").values
            ax.scatter(sx_out, sy_out, s=20, alpha=0.8, color="#E15759",
                       marker="x", linewidths=1.0, label="Dataset (outside hull)", zorder=3)

            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.12)
            if r == n - 1:
                ax.set_xlabel(xp.replace("_", " "), fontsize=7)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(yp.replace("_", " "), fontsize=7)
            else:
                ax.set_yticklabels([])

    # Single legend
    handles, labels = axes[n - 2, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)

    n_in = int(in_mask_plot.sum()) if "param_set_id" in data_df.columns else int(in_mask.sum())
    n_out = int(out_mask_plot.sum()) if "param_set_id" in data_df.columns else int(out_mask.sum())
    fig.suptitle(
        f"Parameter-space convex hull coverage\n"
        f"Dictionary: {len(dict_df)}, Dataset inside: {n_in}, outside: {n_out}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, PLOTS_DIR / "hull_coverage_params.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_error_vs_distance(result_df: pd.DataFrame, param_cols: list[str]) -> None:
    """Scatter of estimation error vs best distance for each parameter, coloured by hull membership."""
    plot_cols = [pc for pc in param_cols
                 if f"est_{pc}" in result_df.columns and f"true_{pc}" in result_df.columns]
    if not plot_cols:
        return

    has_hull = "in_coverage" in result_df.columns
    if has_hull:
        in_hull = result_df["in_coverage"].astype(bool).values

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

        if has_hull:
            m_in = m & in_hull
            m_out = m & ~in_hull
            if np.any(m_in):
                ax.scatter(d[m_in], err[m_in], s=8, alpha=0.4, color="#59A14F",
                           edgecolors="none", label="In hull", zorder=1)
            if np.any(m_out):
                ax.scatter(d[m_out], err[m_out], s=12, alpha=0.7, color="#E15759",
                           marker="x", linewidths=0.8, label="Outside hull", zorder=2)
        else:
            ax.scatter(d[m], err[m], s=8, alpha=0.4, edgecolors="none")

        ax.set_xlabel("Best-match distance", fontsize=8)
        ax.set_ylabel(f"|Error| in {pc}", fontsize=8)
        ax.set_title(pc, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)
        if has_hull and i == 0:
            ax.legend(fontsize=7, loc="upper right")

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
    inv_cfg = cfg_21.get("inverse_mapping", {})
    k = int(inv_cfg.get("neighbor_count", 10))
    idw_power = float(inv_cfg.get("inverse_distance_power", 2.0))
    ridge_lambda = 1e6  # default: pure IDW (no local-linear)

    # Load distance definition from step 1.5 artifact (shared function)
    dd = load_distance_definition(feature_cols, path=DEFAULT_DISTANCE_DEFINITION)
    dist_mode_name: str = "l2_standard_zscore_fallback"

    if dd["available"]:
        dist_mode_name = dd.get("selected_mode", "unknown")
        # Override k and lambda from artifact if available
        if "optimal_k" in dd:
            k = int(dd["optimal_k"])
        if "optimal_lambda" in dd:
            ridge_lambda = float(dd["optimal_lambda"])
        n_active = int(np.sum(dd["weights"] > 0))
        regression_mode = "local-linear ridge" if ridge_lambda < 1e5 else "IDW²"
        log.info(
            "Loaded distance definition from step 1.5: %s (p=%.1f, k=%d, λ=%.0e [%s], %d/%d active features)",
            dist_mode_name, dd["p_norm"], k, ridge_lambda, regression_mode, n_active, len(feature_cols),
        )
    else:
        log.warning("Distance definition not available: %s; falling back to standard z-score.", dd.get("reason"))
        dd = None

    # Determine aggregation mode from ridge_lambda
    aggregation = "local_linear" if ridge_lambda < 1e5 else "weighted_mean"

    log.info("Dictionary: %s (%d rows)", dict_path, len(dict_df))
    log.info("Dataset:    %s (%d rows)", data_path, len(data_df))
    log.info("Feature columns: %d", len(feature_cols))
    log.info("Parameter columns: %s", param_cols)
    log.info("k=%d, IDW power=%.1f, ridge λ=%.0e, aggregation=%s", k, idw_power, ridge_lambda, aggregation)

    # ── Run estimation using the shared engine ─────────────────
    result_df = estimate_from_dataframes(
        dict_df=dict_df,
        data_df=data_df,
        feature_columns=feature_cols,
        distance_metric="l2_zscore",
        param_columns=param_cols,
        exclude_same_file=False,
        shared_parameter_exclusion_mode="full",
        shared_parameter_exclusion_columns=["param_set_id"],
        shared_parameter_exclusion_ignore=(),
        density_weighting_cfg=None,
        include_global_rate=False,
        inverse_mapping_cfg={
            "neighbor_selection": "knn",
            "neighbor_count": k,
            "weighting": "inverse_distance",
            "inverse_distance_power": idw_power,
            "aggregation": aggregation,
        },
        distance_definition=dd,
    )

    # Flag entries outside dictionary convex hull in parameter space
    dict_params_hull = dict_df[param_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    data_params_hull = data_df[param_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    try:
        hull_delaunay = Delaunay(dict_params_hull)
        simplex_ids = hull_delaunay.find_simplex(data_params_hull)
        result_df["in_coverage"] = simplex_ids >= 0
        n_out = int((simplex_ids < 0).sum())
        if n_out > 0:
            log.info(
                "Coverage flag: %d / %d dataset entries are outside the dictionary convex hull in parameter space.",
                n_out, len(result_df),
            )
        else:
            log.info("All %d dataset entries are inside the dictionary convex hull.", len(result_df))
    except Exception as exc:
        log.warning("Convex hull computation failed (%s); skipping coverage flag.", exc)
        hull_delaunay = None

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
        "distance_definition_used": dd is not None,
        "distance_definition_mode": dd.get("selected_mode") if dd is not None else None,
        "k": k,
        "idw_power": idw_power,
        "aggregation": aggregation,
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
    _plot_hull_coverage(dict_df, data_df, result_df, param_cols)

    log.info("Done. %d OK, %d failed.", n_ok, n_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
