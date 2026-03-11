#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_5_TUNE_DISTANCE_DEFINITION/tune_distance_definition.py
Purpose: STEP 1.5 — Auto-tune the feature-space distance definition for downstream estimation.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-11
Runtime: python3
Usage: python3 .../tune_distance_definition.py [--config CONFIG]
Inputs: dictionary.csv and selected_feature_columns.json from STEP 1.4.
Outputs: distance_definition.json artifact for STEP 2.1.
Notes: Trains a weighted Lp distance via LOO on dictionary entries
       (grid search over normalization × p, then coordinate descent on per-feature weights).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re

# ── Paths ────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
if STEP_DIR.parents[1].name == "STEPS":
    PIPELINE_DIR = STEP_DIR.parents[2]
else:
    PIPELINE_DIR = STEP_DIR.parents[1]

DEFAULT_CONFIG = PIPELINE_DIR / "config_method.json"
DEFAULT_DICTIONARY = (
    STEP_DIR.parent / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dictionary.csv"
)
DEFAULT_SELECTED_FEATURES = (
    STEP_DIR.parent / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY"
    / "OUTPUTS" / "FILES" / "selected_feature_columns.json"
)
DEFAULT_PARAMETER_SPACE = (
    STEP_DIR.parent / "STEP_1_1_COLLECT_DATA"
    / "OUTPUTS" / "FILES" / "parameter_space_columns.json"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_STEP_PREFIX = "1_5"
_FIGURE_COUNTER = 0

logging.basicConfig(format="[%(levelname)s] STEP_1.5 — %(message)s", level=logging.INFO)
log = logging.getLogger("STEP_1.5")

# ── Tuning constants ─────────────────────────────────────────────────
MIN_FEATURE_NON_NULL_FRACTION = 0.50

_NORM_GRID = ["robust_zscore", "standard_zscore", "none"]
_P_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]
_WEIGHT_CANDIDATES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
_MAX_WEIGHT_ROUNDS = 3
_K_GRID = [3, 5, 7, 10, 15, 20, 30, 50, 80, 120]
_LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1e6]  # 1e6 ≈ pure IDW fallback

# Legacy name → (p, normalization) for backward compat with forced modes.
_LEGACY_MODES: dict[str, tuple[float, str]] = {
    "l2_robust_zscore": (2.0, "robust_zscore"),
    "l2_standard_zscore": (2.0, "standard_zscore"),
    "l2_raw": (2.0, "none"),
    "l1_robust_zscore": (1.0, "robust_zscore"),
    "l1_standard_zscore": (1.0, "standard_zscore"),
    "l1_raw": (1.0, "none"),
    "l2_zscore": (2.0, "robust_zscore"),
    "l1_zscore": (1.0, "robust_zscore"),
}


# ── Helpers ──────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    cfg: dict = {}
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        log.warning("Config file not found: %s", path)
    runtime_path = path.with_name("config_runtime.json")
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        cfg.update(runtime)
        log.info("Loaded runtime overrides: %s", runtime_path)
    return cfg


def _load_selected_features(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    cols = payload.get("selected_feature_columns", []) if isinstance(payload, dict) else []
    return [c for c in cols if isinstance(c, str) and c.strip()]


def _load_parameter_space_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    candidates = [
        payload.get("parameter_space_columns_downstream_preferred"),
        payload.get("selected_parameter_space_columns_downstream_preferred"),
        payload.get("selected_parameter_space_columns"),
        payload.get("parameter_space_columns"),
    ]
    for raw in candidates:
        if isinstance(raw, list):
            cols = [c.strip() for c in raw if isinstance(c, str) and c.strip()]
            if cols:
                return cols
    return []


# ── Tuning functions ─────────────────────────────────────────────────

def _compute_normalization_params(
    x_raw: np.ndarray, normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, scale) arrays for a normalization mode."""
    if normalization == "robust_zscore":
        center = np.nanmedian(x_raw, axis=0)
        mad = np.nanmedian(np.abs(x_raw - center), axis=0)
        scale = np.where(mad > 1e-12, 1.4826 * mad, np.nanstd(x_raw, axis=0))
        scale = np.where(scale > 1e-12, scale, 1.0)
    elif normalization == "standard_zscore":
        center = np.nanmean(x_raw, axis=0)
        scale = np.nanstd(x_raw, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
    else:  # "none"
        center = np.zeros(x_raw.shape[1], dtype=float)
        scale = np.ones(x_raw.shape[1], dtype=float)
    return center.astype(float), scale.astype(float)


def _weighted_lp_pairwise(
    z: np.ndarray, p: float, weights: np.ndarray,
) -> np.ndarray:
    """Pairwise weighted Lp distance matrix.

    d(i,k) = (sum_j w_j |z_i_j - z_k_j|^p )^(1/p)
    """
    diff = np.abs(z[:, np.newaxis, :] - z[np.newaxis, :, :])
    if p == 1.0:
        comp = diff
    elif p == 2.0:
        comp = diff * diff
    else:
        comp = np.power(diff, p)
    dist_p = np.einsum("ijk,k->ij", comp, weights)
    if p == 1.0:
        return dist_p
    if p == 2.0:
        return np.sqrt(dist_p)
    return np.power(np.maximum(dist_p, 0.0), 1.0 / p)


def _loo_knn_error(
    dist_mat: np.ndarray, y_param: np.ndarray, k: int,
    z_feat: np.ndarray | None = None, ridge_lambda: float = 1e6,
) -> float:
    """LOO kNN estimation error from a precomputed distance matrix.

    When z_feat is provided and ridge_lambda < 1e5, uses locally weighted
    linear regression with ridge on slopes (free intercept).
    Otherwise falls back to IDW² weighted mean.

    Returns mean across parameters of median |relative error| %.
    """
    n = dist_mat.shape[0]
    n_params = y_param.shape[1]
    dm = dist_mat.copy()
    np.fill_diagonal(dm, np.inf)
    nn_idx = np.argpartition(dm, k, axis=1)[:, :k]
    nn_dist = np.take_along_axis(dm, nn_idx, axis=1)

    eps = 1e-12
    w = 1.0 / np.maximum(nn_dist, eps) ** 2  # (n, k)

    use_local_linear = (z_feat is not None) and (ridge_lambda < 1e5)

    if not use_local_linear:
        # Vectorized IDW² weighted mean
        w_norm = w / np.maximum(w.sum(axis=1, keepdims=True), eps)
        all_estimated = np.stack(
            [np.sum(w_norm * y_param[nn_idx, j], axis=1) for j in range(n_params)],
            axis=1,
        )
    else:
        # Local-linear ridge: design matrix [1, x-x₀], ridge only on slopes
        d = z_feat.shape[1]
        all_estimated = np.empty((n, n_params), dtype=float)
        reg = np.zeros(d + 1)
        reg[1:] = ridge_lambda
        reg_diag = np.diag(reg)
        all_neigh_params = y_param[nn_idx]  # (n, k, n_params)

        for i in range(n):
            wi = w[i]                                     # (k,)
            xi = z_feat[nn_idx[i]] - z_feat[i]            # (k, d)
            yi = all_neigh_params[i]                       # (k, n_params)
            Z = np.column_stack([np.ones(k), xi])          # (k, d+1)
            ZtW = Z.T * wi[np.newaxis, :]                  # (d+1, k)
            A = ZtW @ Z + reg_diag                         # (d+1, d+1)
            B = ZtW @ yi                                   # (d+1, n_params)
            try:
                Theta = np.linalg.solve(A, B)              # (d+1, n_params)
                all_estimated[i] = Theta[0]                # intercept row
            except np.linalg.LinAlgError:
                w_sum = max(float(np.sum(wi)), eps)
                all_estimated[i] = np.dot(wi, yi) / w_sum

    param_errs: list[float] = []
    for j in range(n_params):
        true_vals = y_param[:, j]
        estimated = all_estimated[:, j]
        abs_err = np.abs(estimated - true_vals)
        prange = max(float(np.ptp(true_vals)), 1e-12)
        denom = np.where(np.abs(true_vals) > 1e-12, np.abs(true_vals), prange)
        rel_pct = abs_err / denom * 100.0
        param_errs.append(float(np.nanmedian(rel_pct)))
    return float(np.mean(param_errs))


def _loo_knn_predictions(
    dist_mat: np.ndarray, y_param: np.ndarray, k: int,
    z_feat: np.ndarray | None = None, ridge_lambda: float = 1e6,
) -> np.ndarray:
    """Compute LOO kNN predictions for all dictionary entries.

    Returns (n, n_params) array of estimated values.
    Same logic as _loo_knn_error but returns estimates instead of scalar.
    """
    n = dist_mat.shape[0]
    n_params = y_param.shape[1]
    dm = dist_mat.copy()
    np.fill_diagonal(dm, np.inf)
    nn_idx = np.argpartition(dm, k, axis=1)[:, :k]
    nn_dist = np.take_along_axis(dm, nn_idx, axis=1)

    eps = 1e-12
    w = 1.0 / np.maximum(nn_dist, eps) ** 2

    use_local_linear = (z_feat is not None) and (ridge_lambda < 1e5)

    if not use_local_linear:
        w_norm = w / np.maximum(w.sum(axis=1, keepdims=True), eps)
        return np.stack(
            [np.sum(w_norm * y_param[nn_idx, j], axis=1) for j in range(n_params)],
            axis=1,
        )

    d = z_feat.shape[1]
    all_estimated = np.empty((n, n_params), dtype=float)
    reg = np.zeros(d + 1)
    reg[1:] = ridge_lambda
    reg_diag = np.diag(reg)
    all_neigh_params = y_param[nn_idx]

    for i in range(n):
        wi = w[i]
        xi = z_feat[nn_idx[i]] - z_feat[i]
        yi = all_neigh_params[i]
        Z = np.column_stack([np.ones(k), xi])
        ZtW = Z.T * wi[np.newaxis, :]
        A = ZtW @ Z + reg_diag
        B = ZtW @ yi
        try:
            Theta = np.linalg.solve(A, B)
            all_estimated[i] = Theta[0]
        except np.linalg.LinAlgError:
            w_sum = max(float(np.sum(wi)), eps)
            all_estimated[i] = np.dot(wi, yi) / w_sum
    return all_estimated


def _plot_loo_true_vs_estimated(
    y_true: np.ndarray,
    y_est: np.ndarray,
    param_cols: list[str],
) -> None:
    """Plot LOO true vs estimated for each parameter (step 1.5 validation)."""
    n_params = y_true.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(5, 4.5 * n_params), squeeze=False)

    for j, pc in enumerate(param_cols):
        ax = axes[j, 0]
        t = y_true[:, j]
        e = y_est[:, j]
        m = np.isfinite(t) & np.isfinite(e)
        if not np.any(m):
            ax.set_title(pc, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.scatter(t[m], e[m], s=14, alpha=0.6, edgecolors="none", color="#4C78A8")
        lo = min(float(np.nanmin(t[m])), float(np.nanmin(e[m])))
        hi = max(float(np.nanmax(t[m])), float(np.nanmax(e[m])))
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"True {pc}", fontsize=8)
        ax.set_ylabel(f"LOO Estimated {pc}", fontsize=8)
        ax.set_title(pc, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15)

        rmse = float(np.sqrt(np.mean((t[m] - e[m]) ** 2)))
        mae = float(np.mean(np.abs(t[m] - e[m])))
        denom = np.where(np.abs(t[m]) > 1e-12, np.abs(t[m]),
                         max(float(np.ptp(t[m])), 1e-12))
        med_rel = float(np.median(np.abs(t[m] - e[m]) / denom * 100.0))
        ax.text(0.03, 0.97,
                f"RMSE={rmse:.4g}\nMAE={mae:.4g}\nMed.rel.err={med_rel:.2f}%",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("LOO validation: True vs Estimated (dictionary entries)", fontsize=11, y=1.0)
    fig.tight_layout()
    path = PLOTS_DIR / "loo_true_vs_estimated.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved LOO true-vs-estimated plot: %s", path)


def _plot_loo_relative_error_histograms(
    y_true: np.ndarray,
    y_est: np.ndarray,
    param_cols: list[str],
) -> None:
    """Histogram of LOO relative errors for each parameter."""
    n_params = y_true.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(5, 3.5 * n_params), squeeze=False)

    for j, pc in enumerate(param_cols):
        ax = axes[j, 0]
        t = y_true[:, j]
        e = y_est[:, j]
        m = np.isfinite(t) & np.isfinite(e)
        if not np.any(m):
            ax.set_title(pc, fontsize=9)
            continue

        denom = np.where(np.abs(t[m]) > 1e-12, np.abs(t[m]),
                         max(float(np.ptp(t[m])), 1e-12))
        rel_pct = (e[m] - t[m]) / denom * 100.0

        ax.hist(rel_pct, bins=40, histtype="stepfilled",
                color="#4C78A8", alpha=0.6, edgecolor="#2a5080")
        ax.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
        med = float(np.median(rel_pct))
        ax.axvline(med, color="red", ls="-", lw=0.9, alpha=0.7,
                   label=f"Median = {med:.2f}%")
        ax.set_xlabel(f"Rel. error {pc} [%]", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(pc, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=7)

    fig.suptitle("LOO validation: relative error distributions (dictionary)", fontsize=11, y=1.0)
    fig.tight_layout()
    path = PLOTS_DIR / "loo_relative_error_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved LOO relative-error histogram plot: %s", path)


def _auto_tune_distance(
    x_raw: np.ndarray,
    y_param: np.ndarray,
    param_cols: list[str],
    feature_cols: list[str],
    *,
    forced_mode: str | None = None,
) -> dict:
    """Auto-tune the feature-space distance definition on dictionary entries.

    Phase 1 — joint grid search over normalization × p × k × λ with uniform weights.
    Phase 2 — coordinate descent on per-feature weights at the best (norm, p, k, λ).
    """
    n = x_raw.shape[0]
    d = x_raw.shape[1]
    k_candidates = [k for k in _K_GRID if k < n]

    # ── Phase 1: joint grid search  normalization × p × k ────────────
    if forced_mode is not None and forced_mode in _LEGACY_MODES:
        forced_p, forced_norm = _LEGACY_MODES[forced_mode]
        norms_to_try = [forced_norm]
        ps_to_try = [forced_p]
        log.info("  Forced mode '%s' → norm=%s, p=%.1f (skipping grid, tuning weights only)",
                 forced_mode, forced_norm, forced_p)
    elif forced_mode is not None:
        log.warning("Unknown forced mode '%s'; falling back to full auto-tune.", forced_mode)
        forced_mode = None
        norms_to_try = _NORM_GRID
        ps_to_try = _P_GRID
    else:
        norms_to_try = _NORM_GRID
        ps_to_try = _P_GRID

    uniform_w = np.ones(d, dtype=float)
    grid_results: dict[str, float] = {}
    best_score = np.inf
    best_norm = norms_to_try[0]
    best_p = ps_to_try[0]
    best_k = k_candidates[0]
    best_lambda = _LAMBDA_GRID[-1]  # default to high λ (≈ IDW)

    for norm in norms_to_try:
        center, scale = _compute_normalization_params(x_raw, norm)
        z = (x_raw - center) / scale
        for p in ps_to_try:
            dm = _weighted_lp_pairwise(z, p, uniform_w)
            # Sweep k × λ on the same distance matrix
            for k in k_candidates:
                for lam in _LAMBDA_GRID:
                    score = _loo_knn_error(dm, y_param, k, z_feat=z, ridge_lambda=lam)
                    label = f"p={p:.1f}_{norm}_k={k}_lam={lam:.0e}"
                    grid_results[label] = round(score, 4)
                    if score < best_score:
                        best_score = score
                        best_norm = norm
                        best_p = p
                        best_k = k
                        best_lambda = lam
            # Log the best (k, λ) for this (norm, p) combo for readability
            combo_keys = [lbl for lbl in grid_results if lbl.startswith(f"p={p:.1f}_{norm}_k=")]
            if combo_keys:
                best_combo_key = min(combo_keys, key=lambda lbl: grid_results[lbl])
                log.info("  Grid  %-25s best: %-30s → %7.3f %%",
                         f"p={p:.1f}_{norm}", best_combo_key.split(f"{norm}_")[1], grid_results[best_combo_key])

    log.info("  Best grid: norm=%s, p=%.1f, k=%d, λ=%.0e (error %.3f %%)",
             best_norm, best_p, best_k, best_lambda, best_score)

    # ── Phase 2: coordinate descent on per-feature weights (at best k) ──
    center, scale = _compute_normalization_params(x_raw, best_norm)
    z = (x_raw - center) / scale

    weights = np.ones(d, dtype=float)
    current_score = best_score

    for rnd in range(1, _MAX_WEIGHT_ROUNDS + 1):
        improved_count = 0
        for j in range(d):
            w_orig = weights[j]
            best_wj = w_orig
            best_wj_score = current_score
            for wc in _WEIGHT_CANDIDATES:
                if wc == w_orig:
                    continue
                weights[j] = wc
                if np.sum(weights > 0) < 1:
                    weights[j] = w_orig
                    continue
                dm = _weighted_lp_pairwise(z, best_p, weights)
                s = _loo_knn_error(dm, y_param, best_k, z_feat=z, ridge_lambda=best_lambda)
                if s < best_wj_score:
                    best_wj_score = s
                    best_wj = wc
            weights[j] = best_wj
            if best_wj != w_orig:
                current_score = best_wj_score
                improved_count += 1
        # Normalize weights so mean of nonzero weights = 1
        nz = weights[weights > 0]
        if len(nz) > 0:
            weights = weights / float(np.mean(nz))
        n_active = int(np.sum(weights > 0))
        n_zero = d - n_active
        log.info(
            "  Weight round %d: error %.3f %%, changed %d/%d features, %d active, %d zeroed",
            rnd, current_score, improved_count, d, n_active, n_zero,
        )
        if improved_count == 0:
            break

    weight_tuned_score = current_score

    mode_label = f"p{best_p:.1f}_{best_norm}"
    log.info(
        "Selected distance: %s, k=%d, λ=%.0e, weights tuned (grid %.3f %% → tuned %.3f %%)",
        mode_label, best_k, best_lambda, best_score, weight_tuned_score,
    )

    return {
        "selected_mode": mode_label,
        "p_norm": float(best_p),
        "normalization": best_norm,
        "optimal_k": best_k,
        "optimal_lambda": best_lambda,
        "feature_columns": list(feature_cols),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "grid_results": grid_results,
        "grid_best_score": round(best_score, 4),
        "weight_tuned_score": round(weight_tuned_score, 4),
        "n_active_features": int(np.sum(weights > 0)),
        "n_zeroed_features": int(np.sum(weights == 0)),
        "tuning_metric": "median_relative_error_pct",
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="STEP 1.5: Auto-tune feature-space distance definition for downstream estimation.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dictionary-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_15 = config.get("step_1_5", {}) or {}

    # ── Locate inputs ────────────────────────────────────────────────
    dict_cfg = cfg_15.get("dictionary_csv")
    if args.dictionary_csv:
        dictionary_path = Path(args.dictionary_csv).expanduser()
    elif dict_cfg not in (None, "", "null", "None"):
        dictionary_path = Path(str(dict_cfg)).expanduser()
    else:
        dictionary_path = DEFAULT_DICTIONARY

    if not dictionary_path.exists():
        log.error("Dictionary CSV not found: %s", dictionary_path)
        return 1

    # Load dictionary
    dictionary = pd.read_csv(dictionary_path, low_memory=False)
    if dictionary.empty:
        log.error("Input dictionary is empty: %s", dictionary_path)
        return 1
    log.info("Loaded dictionary: %s (%d rows)", dictionary_path, len(dictionary))

    # Feature columns
    feat_path = cfg_15.get("selected_feature_columns_json")
    if feat_path and Path(feat_path).exists():
        selected_features_path = Path(feat_path)
    else:
        selected_features_path = DEFAULT_SELECTED_FEATURES

    feature_cols = _load_selected_features(selected_features_path)
    if not feature_cols:
        log.error("No feature columns loaded from %s", selected_features_path)
        return 1
    feature_cols = [c for c in feature_cols if c in dictionary.columns]
    if not feature_cols:
        log.error("No feature columns found in dictionary.")
        return 1

    # Parameter columns
    param_cols_loaded = _load_parameter_space_columns(DEFAULT_PARAMETER_SPACE)
    param_cols = param_cols_loaded or ["flux_cm2_min", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"]
    param_cols = [c for c in param_cols if c in dictionary.columns]
    if not param_cols:
        log.error("No parameter columns found in dictionary.")
        return 1

    log.info("Feature columns: %d", len(feature_cols))
    log.info("Parameter columns: %s", param_cols)

    # ── Prepare numeric matrices ─────────────────────────────────────
    feat_num = dictionary[feature_cols].apply(pd.to_numeric, errors="coerce")
    non_null_frac = feat_num.notna().mean(axis=0)
    kept_cols = [c for c in feature_cols if float(non_null_frac.get(c, 0.0)) >= MIN_FEATURE_NON_NULL_FRACTION]
    if not kept_cols:
        log.error("No feature columns survived non-null filter.")
        return 1
    feature_cols = kept_cols

    feat_num = dictionary[feature_cols].apply(pd.to_numeric, errors="coerce")
    feat_num = feat_num.fillna(feat_num.median(numeric_only=True))

    par_num = dictionary[param_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = par_num.notna().all(axis=1)
    valid_idx = np.flatnonzero(valid_mask.to_numpy())
    if len(valid_idx) < 3:
        log.error("Not enough valid rows for tuning (need >=3, got %d).", len(valid_idx))
        return 1

    x_raw = feat_num.iloc[valid_idx].to_numpy(dtype=float)
    y_param = par_num.iloc[valid_idx].to_numpy(dtype=float)
    log.info("Valid dictionary rows for tuning: %d, feature dims: %d", len(valid_idx), x_raw.shape[1])

    # ── Run distance tuning ──────────────────────────────────────────
    cfg_distance = str(cfg_15.get("feature_distance_definition", "auto")).strip().lower()
    forced_mode = None if cfg_distance == "auto" else cfg_distance
    log.info("Distance-mode selection: %s", "auto-tune" if forced_mode is None else f"forced={forced_mode}")

    tune_result = _auto_tune_distance(
        x_raw, y_param, param_cols, feature_cols,
        forced_mode=forced_mode,
    )

    # ── Save artifact ────────────────────────────────────────────────
    out_path = FILES_DIR / "distance_definition.json"
    out_path.write_text(json.dumps(tune_result, indent=2), encoding="utf-8")
    log.info("Wrote distance definition artifact: %s", out_path)

    # ── LOO validation plots ─────────────────────────────────────────
    center = np.asarray(tune_result["center"])
    scale = np.asarray(tune_result["scale"])
    weights_arr = np.asarray(tune_result["weights"])
    z_tuned = (x_raw - center) / scale
    dm_final = _weighted_lp_pairwise(z_tuned, tune_result["p_norm"], weights_arr)
    y_loo = _loo_knn_predictions(
        dm_final, y_param, tune_result["optimal_k"],
        z_feat=z_tuned, ridge_lambda=tune_result["optimal_lambda"],
    )
    _plot_loo_true_vs_estimated(y_param, y_loo, param_cols)
    _plot_loo_relative_error_histograms(y_param, y_loo, param_cols)
    log.info(
        "Result: %s (p=%.1f, norm=%s, k=%d, λ=%.0e, %d/%d active features, "
        "grid %.3f%% → tuned %.3f%%)",
        tune_result["selected_mode"],
        tune_result["p_norm"],
        tune_result["normalization"],
        tune_result["optimal_k"],
        tune_result["optimal_lambda"],
        tune_result["n_active_features"],
        len(feature_cols),
        tune_result["grid_best_score"],
        tune_result["weight_tuned_score"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
