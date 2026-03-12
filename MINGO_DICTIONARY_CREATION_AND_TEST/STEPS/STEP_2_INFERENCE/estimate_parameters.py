#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_2_INFERENCE/estimate_parameters.py
Purpose: Self-contained dictionary-based parameter estimation module.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_2_INFERENCE/estimate_parameters.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

from itertools import combinations
import json
import logging
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from feature_columns_config import parse_explicit_feature_columns

log = logging.getLogger("estimate_parameters")

# Default location of the STEP 1.5 distance-definition artifact,
# relative to the STEP_2_INFERENCE directory (i.e. this file's parent).
_INFERENCE_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _INFERENCE_DIR.parent
DEFAULT_DISTANCE_DEFINITION_PATH = (
    _PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_5_TUNE_DISTANCE_DEFINITION"
    / "OUTPUTS" / "FILES" / "distance_definition.json"
)


def load_distance_definition(
    feature_columns: list[str],
    *,
    path: Path | str | None = None,
) -> dict:
    """Load the STEP 1.5 distance-definition artifact and validate against *feature_columns*.

    Returns a dict with keys:
      available       – bool
      center          – np.ndarray (only when available)
      scale           – np.ndarray
      weights         – np.ndarray
      p_norm          – float
      optimal_k       – int
      optimal_lambda  – float
      selected_mode   – str
      reason          – str (only when *not* available)
    """
    dd_path = Path(path) if path is not None else DEFAULT_DISTANCE_DEFINITION_PATH
    if not dd_path.exists():
        return {"available": False, "reason": f"file_not_found ({dd_path})"}

    dist_def = json.loads(dd_path.read_text(encoding="utf-8"))
    dd_cols = dist_def.get("feature_columns", [])
    dd_center = np.asarray(dist_def["center"], dtype=float)
    dd_scale = np.asarray(dist_def["scale"], dtype=float)
    dd_weights = np.asarray(
        dist_def.get("weights", [1.0] * len(dd_cols)), dtype=float
    )

    if list(dd_cols) != list(feature_columns) or len(dd_center) != len(feature_columns):
        return {
            "available": False,
            "reason": (
                f"feature_mismatch: artifact has {len(dd_cols)} columns "
                f"vs requested {len(feature_columns)}"
            ),
        }

    return {
        "available": True,
        "center": dd_center,
        "scale": dd_scale,
        "weights": dd_weights,
        "p_norm": float(dist_def.get("p_norm", 2.0)),
        "optimal_k": int(dist_def.get("optimal_k", 5)),
        "optimal_lambda": float(dist_def.get("optimal_lambda", 1e6)),
        "selected_mode": dist_def.get("selected_mode", "unknown"),
    }


# =====================================================================
# Distance functions
# =====================================================================

def _l2(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.sum(diff ** 2)))


def _chi2(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    y_t, y_p = a[mask], b[mask]
    # Poisson variance estimate: max(|observed|, 1) prevents division by zero
    variance_floor = np.maximum(np.abs(y_t), 1.0)
    return float(np.sum((y_t - y_p) ** 2 / variance_floor))


def _poisson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    y_t = np.maximum(a[mask], 0.0)
    y_p = np.maximum(b[mask], 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y_t > 0, y_t / y_p, 1.0)
        term = np.where(y_t > 0, y_t * np.log(ratio) - (y_t - y_p), y_p)
    return float(2.0 * np.sum(term))


def _l2_many(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_valid_dims: int = 2,
) -> np.ndarray:
    """Vectorized L2 distance from one sample vector to many candidate vectors."""
    mask = np.isfinite(a)[None, :] & np.isfinite(b)
    valid_counts = np.sum(mask, axis=1)
    diff = np.where(mask, b - a[None, :], 0.0)
    out = np.sqrt(np.sum(diff * diff, axis=1))
    out = out.astype(float, copy=False)
    min_dims = max(int(min_valid_dims), 1)
    out[valid_counts < min_dims] = np.nan
    return out


def _chi2_many(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_valid_dims: int = 2,
) -> np.ndarray:
    """Vectorized chi2-like distance from one sample to many candidates."""
    mask = np.isfinite(a)[None, :] & np.isfinite(b)
    valid_counts = np.sum(mask, axis=1)
    y_t = np.where(mask, a[None, :], 0.0)
    y_p = np.where(mask, b, 0.0)
    # Poisson variance estimate: max(|observed|, 1) prevents division by zero
    variance_floor = np.maximum(np.abs(y_t), 1.0)
    out = np.sum((y_t - y_p) ** 2 / variance_floor, axis=1).astype(float, copy=False)
    min_dims = max(int(min_valid_dims), 1)
    out[valid_counts < min_dims] = np.nan
    return out


def _poisson_many(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_valid_dims: int = 2,
) -> np.ndarray:
    """Vectorized Poisson deviance-like distance from one sample to many candidates."""
    mask = np.isfinite(a)[None, :] & np.isfinite(b)
    valid_counts = np.sum(mask, axis=1)
    y_t = np.where(mask, np.maximum(a[None, :], 0.0), 0.0)
    y_p = np.where(mask, np.maximum(b, 1e-12), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y_t > 0.0, y_t / y_p, 1.0)
        term = np.where(y_t > 0.0, y_t * np.log(ratio) - (y_t - y_p), y_p)
    out = 2.0 * np.sum(np.where(mask, term, 0.0), axis=1)
    out = out.astype(float, copy=False)
    min_dims = max(int(min_valid_dims), 1)
    out[valid_counts < min_dims] = np.nan
    return out


DISTANCE_FNS = {
    "l2": _l2,
    "l2_zscore": _l2,
    "l2_raw": _l2,
    "chi2": _chi2,
    "poisson": _poisson,
}

DISTANCE_FNS_MANY = {
    "l2": _l2_many,
    "l2_zscore": _l2_many,
    "l2_raw": _l2_many,
    "chi2": _chi2_many,
    "poisson": _poisson_many,
}


def _weighted_lp_many(
    sample: np.ndarray,
    candidates: np.ndarray,
    *,
    weights: np.ndarray,
    p_norm: float,
    min_valid_dims: int = 2,
) -> np.ndarray:
    """Weighted Lp distance from one sample to many candidates.

    ``weights`` and ``p_norm`` come from the STEP 1.5 distance-definition
    artifact.  The formula is:

        d = (Σ_i  w_i · |x_i − y_i|^p )^(1/p)

    Features with zero weight are excluded.
    """
    mask = np.isfinite(sample)[None, :] & np.isfinite(candidates)
    w = np.asarray(weights, dtype=float)
    # Ignore features with zero weight
    mask &= (w > 0.0)[None, :]
    valid_counts = np.sum(mask, axis=1)
    diff_abs = np.where(mask, np.abs(candidates - sample[None, :]), 0.0)
    if p_norm == 1.0:
        out = np.sum(w[None, :] * diff_abs, axis=1)
    elif p_norm == 2.0:
        out = np.sqrt(np.sum(w[None, :] * diff_abs * diff_abs, axis=1))
    else:
        out = np.power(
            np.sum(w[None, :] * np.power(diff_abs, p_norm), axis=1),
            1.0 / p_norm,
        )
    out = out.astype(float, copy=False)
    min_dims = max(int(min_valid_dims), 1)
    out[valid_counts < min_dims] = np.nan
    return out


RATE_HISTOGRAM_BIN_RE = re.compile(r"^events_per_second_(?P<bin>\d+)_rate_hz")


def _histogram_feature_indices(feature_cols: Sequence[str]) -> list[int]:
    indexed: list[tuple[int, int]] = []
    for idx, col in enumerate(feature_cols):
        match = RATE_HISTOGRAM_BIN_RE.match(str(col))
        if match is None:
            continue
        indexed.append((int(match.group("bin")), idx))
    indexed.sort(key=lambda x: x[0])
    return [idx for _, idx in indexed]


def _shared_histogram_feature_columns(
    dict_cols: Sequence[object],
    data_cols: Sequence[object],
) -> list[str]:
    """Resolve shared histogram-bin rate columns ordered by bin index."""
    data_col_set = {str(c) for c in data_cols}
    by_bin: dict[int, str] = {}
    for col in dict_cols:
        name = str(col)
        match = RATE_HISTOGRAM_BIN_RE.match(name)
        if match is None or name not in data_col_set:
            continue
        by_bin.setdefault(int(match.group("bin")), name)
    return [by_bin[k] for k in sorted(by_bin)]


def _histogram_emd_many(sample_hist: np.ndarray, candidates_hist: np.ndarray) -> np.ndarray:
    """
    Compute normalized Wasserstein-1 (EMD) distance from one histogram row to many.

    Inputs can contain NaN and non-positive entries. Distances are computed only on
    bins finite in both sample and candidate, with row-wise normalization to unit sum.
    Output is in [0, 1] when valid (NaN otherwise).
    """
    sample = np.asarray(sample_hist, dtype=float)
    candidates = np.asarray(candidates_hist, dtype=float)
    if sample.ndim != 1 or candidates.ndim != 2 or candidates.shape[1] != sample.size:
        return np.full(candidates.shape[0] if candidates.ndim == 2 else 0, np.nan, dtype=float)

    n_candidates = candidates.shape[0]
    out = np.full(n_candidates, np.nan, dtype=float)
    if sample.size < 2 or n_candidates == 0:
        return out

    valid_mask = np.isfinite(sample)[None, :] & np.isfinite(candidates)
    valid_counts = np.sum(valid_mask, axis=1)

    sample_clipped = np.clip(sample, 0.0, None)[None, :]
    candidates_clipped = np.clip(candidates, 0.0, None)
    p = np.where(valid_mask, sample_clipped, 0.0)
    q = np.where(valid_mask, candidates_clipped, 0.0)

    sum_p = np.sum(p, axis=1)
    sum_q = np.sum(q, axis=1)
    ok = (valid_counts >= 2) & (sum_p > 0.0) & (sum_q > 0.0)
    if not np.any(ok):
        return out

    p_ok = p[ok] / sum_p[ok, None]
    q_ok = q[ok] / sum_q[ok, None]
    cdf_diff = np.abs(np.cumsum(p_ok, axis=1) - np.cumsum(q_ok, axis=1))
    out[ok] = np.mean(cdf_diff, axis=1)
    return out


def _distance_many(
    sample_vec: np.ndarray | None,
    cand_mat: np.ndarray | None,
    *,
    distance_metric: str,
    min_valid_dims: int = 2,
) -> np.ndarray | None:
    if sample_vec is None or cand_mat is None:
        return None
    sample = np.asarray(sample_vec, dtype=float)
    candidates = np.asarray(cand_mat, dtype=float)
    if candidates.ndim != 2 or sample.ndim != 1:
        return None
    if candidates.shape[0] == 0:
        return np.asarray([], dtype=float)
    dist_many_fn = DISTANCE_FNS_MANY.get(distance_metric, None)
    if dist_many_fn is not None:
        return np.asarray(
            dist_many_fn(sample, candidates, min_valid_dims=max(int(min_valid_dims), 1)),
            dtype=float,
        )
    dist_fn = DISTANCE_FNS.get(distance_metric, _l2)
    return np.asarray([dist_fn(sample, candidates[j]) for j in range(candidates.shape[0])], dtype=float)


def _robust_scale(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    med = float(np.median(np.abs(vals)))
    if np.isfinite(med) and med > 1e-15:
        return med
    vmax = float(np.max(np.abs(vals)))
    if np.isfinite(vmax) and vmax > 1e-15:
        return vmax
    return 1.0


def _effective_atol(base_atol: float, value: float) -> float:
    """Absolute tolerance that scales with magnitude for |value| > 1."""
    return max(base_atol, base_atol * max(1.0, abs(value)))


def _combine_base_and_hist_distances(
    *,
    base_distances: np.ndarray | None,
    hist_distances: np.ndarray | None,
    hist_weight: float,
    blend_mode: str,
) -> np.ndarray:
    base = None if base_distances is None else np.asarray(base_distances, dtype=float)
    hist = None if hist_distances is None else np.asarray(hist_distances, dtype=float)

    if base is None and hist is None:
        return np.asarray([], dtype=float)
    if base is None:
        if blend_mode == "normalized":
            return float(hist_weight) * (hist / _robust_scale(hist))
        return float(hist_weight) * hist
    if hist is None:
        return base

    if base.shape != hist.shape:
        return np.full(base.shape[0], np.nan, dtype=float)

    if blend_mode == "normalized":
        base_term = base / _robust_scale(base)
        hist_term = hist / _robust_scale(hist)
    else:
        base_term = base
        hist_term = hist

    out = np.asarray(base_term, dtype=float).copy()
    base_finite = np.isfinite(out)
    hist_finite = np.isfinite(hist_term)
    both_finite = base_finite & hist_finite
    hist_only = (~base_finite) & hist_finite
    out[both_finite] = out[both_finite] + float(hist_weight) * hist_term[both_finite]
    out[hist_only] = float(hist_weight) * hist_term[hist_only]
    return out


def compute_candidate_distances(
    *,
    distance_metric: str,
    sample_scaled_non_hist: np.ndarray | None,
    candidates_scaled_non_hist: np.ndarray | None,
    sample_hist_raw: np.ndarray | None,
    candidates_hist_raw: np.ndarray | None,
    histogram_distance_weight: float,
    histogram_distance_blend_mode: str,
    min_valid_non_hist_dims: int = 2,
    dd_weights: np.ndarray | None = None,
    dd_p_norm: float | None = None,
) -> np.ndarray:
    """
    Compute candidate distances with the same feature+histogram composition used
    by the estimator.

    When *dd_weights* and *dd_p_norm* are supplied (from the STEP 1.5
    distance-definition artifact), the weighted Lp metric is used for the
    non-histogram features instead of the default ``distance_metric`` function.
    """
    if (
        dd_weights is not None
        and dd_p_norm is not None
        and sample_scaled_non_hist is not None
        and candidates_scaled_non_hist is not None
    ):
        base = _weighted_lp_many(
            sample_scaled_non_hist,
            candidates_scaled_non_hist,
            weights=dd_weights,
            p_norm=dd_p_norm,
            min_valid_dims=max(int(min_valid_non_hist_dims), 1),
        )
    else:
        base = _distance_many(
            sample_scaled_non_hist,
            candidates_scaled_non_hist,
            distance_metric=distance_metric,
            min_valid_dims=max(int(min_valid_non_hist_dims), 1),
        )
    hist = None
    if sample_hist_raw is not None and candidates_hist_raw is not None:
        hist = _histogram_emd_many(
            np.asarray(sample_hist_raw, dtype=float),
            np.asarray(candidates_hist_raw, dtype=float),
        )
    return _combine_base_and_hist_distances(
        base_distances=base,
        hist_distances=hist,
        hist_weight=max(float(histogram_distance_weight), 0.0),
        blend_mode=_normalize_hist_blend_mode(histogram_distance_blend_mode),
    )


def _build_neighbor_weights(
    distances: np.ndarray,
    *,
    weighting_mode: str,
    idw_power: float,
    softmax_temperature: float,
    distance_floor: float,
) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    n = d.size
    if n == 0:
        return np.asarray([], dtype=float)
    mode = _normalize_weighting_mode(weighting_mode)
    finite = np.isfinite(d)
    out = np.zeros(n, dtype=float)
    if not np.any(finite):
        return out

    d_eff = np.clip(d[finite], 0.0, None)
    if mode == "uniform":
        out[finite] = 1.0
    elif mode == "softmax":
        temp = max(float(softmax_temperature), 1e-12)
        d_ref = d_eff - float(np.min(d_eff))
        out[finite] = np.exp(-d_ref / temp)
    else:
        power = max(float(idw_power), 0.0)
        eps = max(float(distance_floor), 1e-15)
        out[finite] = 1.0 / np.power(np.maximum(d_eff, eps), power)

    out = np.where(np.isfinite(out), out, 0.0)
    s = float(np.sum(out))
    if s > 0.0:
        out /= s
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not np.any(mask):
        return np.nan
    vals = vals[mask]
    w = w[mask]
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        return np.nan
    cdf = np.cumsum(w) / s
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(max(idx, 0), len(vals) - 1)
    return float(vals[idx])


def _local_linear_estimate(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    sample_features: np.ndarray | None,
    neighbor_features: np.ndarray | None,
    parameter_name: str | None = None,
    ridge_lambda: float = 1e-2,
) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if sample_features is None or neighbor_features is None:
        return float(np.sum(w * np.nan_to_num(vals)))

    x0 = np.asarray(sample_features, dtype=float)
    X = np.asarray(neighbor_features, dtype=float)
    if X.ndim != 2 or x0.ndim != 1 or X.shape[0] != vals.size:
        return float(np.sum(w * np.nan_to_num(vals)))
    if X.shape[1] == 0:
        return float(np.sum(w * np.nan_to_num(vals)))

    feat_mask = np.isfinite(x0)
    if not np.any(feat_mask):
        return float(np.sum(w * np.nan_to_num(vals)))

    X = X[:, feat_mask]
    x0 = x0[feat_mask]
    row_mask = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(vals)
        & np.isfinite(w)
        & (w > 0.0)
    )
    if int(np.sum(row_mask)) < 3:
        return float(np.sum(w * np.nan_to_num(vals)))

    X_use = X[row_mask]
    y_use = vals[row_mask]
    w_use = w[row_mask]
    w_sum = float(np.sum(w_use))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return float(np.sum(w * np.nan_to_num(vals)))
    w_use = w_use / w_sum

    centered = X_use - x0[None, :]
    A = np.hstack([np.ones((centered.shape[0], 1), dtype=float), centered])
    sqrt_w = np.sqrt(np.clip(w_use, 1e-16, None))
    is_eff_param = False
    if parameter_name is not None:
        pname = str(parameter_name).strip().lower()
        is_eff_param = pname.startswith("eff_") or ("_eff_" in pname)

    y_fit = np.asarray(y_use, dtype=float)
    if is_eff_param:
        # Efficiency-like parameters are physically bounded in [0,1].
        # Fit local linear model in logit space to avoid hard clipping to
        # local neighbor ranges, which can cause artificial saturation.
        eps = 1e-6
        y_fit = np.clip(y_fit, eps, 1.0 - eps)
        y_fit = np.log(y_fit / (1.0 - y_fit))

    Aw = A * sqrt_w[:, None]
    yw = y_fit * sqrt_w

    try:
        # Use unbiased weighted least squares by default. Ridge penalties can
        # shrink slopes and introduce systematic intercept offsets.
        beta = np.linalg.lstsq(Aw, yw, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Fallback: try a tiny ridge regularization only for ill-conditioned rows.
        n_coef = A.shape[1]
        reg = np.zeros((n_coef, n_coef), dtype=float)
        if n_coef > 1:
            reg[1:, 1:] = max(float(ridge_lambda), 0.0) * np.eye(n_coef - 1, dtype=float)
        try:
            beta = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ yw)
        except np.linalg.LinAlgError:
            pred_fallback = float(np.sum(w_use * y_use))
            if is_eff_param and np.isfinite(pred_fallback):
                return float(np.clip(pred_fallback, 1e-6, 1.0 - 1e-6))
            return pred_fallback

    pred = float(beta[0])
    if not np.isfinite(pred):
        return float(np.sum(w_use * y_use))
    if is_eff_param:
        # inverse-logit with numerical stability
        pred = float(1.0 / (1.0 + np.exp(-np.clip(pred, -40.0, 40.0))))
        pred = float(np.clip(pred, 1e-6, 1.0 - 1e-6))
    else:
        y_min = float(np.nanmin(y_use))
        y_max = float(np.nanmax(y_use))
        if np.isfinite(y_min) and np.isfinite(y_max):
            pred = float(np.clip(pred, y_min, y_max))
    return pred


# =====================================================================
# Feature selection helpers
# =====================================================================

TT_RATE_COLUMN_RE = re.compile(r"^(?P<prefix>.+?)_tt_(?P<label>[^_]+)_rate_hz$")
AUTO_TT_PREFIX_PRIORITY = [
    "post",
    "fit_to_post",
    "fit",
    "list_to_fit",
    "list",
    "cal",
    "clean",
    "raw_to_clean",
    "raw",
    "corr",
    "task5_to_corr",
    "fit_to_corr",
    "definitive",
]


def _normalize_tt_label(label: object) -> str:
    text = str(label).strip()
    if not text:
        return ""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(value):
        return ""
    if float(value).is_integer():
        return str(int(value))
    return text


def _is_multi_plane_tt_label(label: object) -> bool:
    norm = _normalize_tt_label(label)
    if len(norm) < 2:
        return False
    if not all(ch in {"1", "2", "3", "4"} for ch in norm):
        return False
    return len(set(norm)) == len(norm)


def _prefix_rank(prefix: str) -> int:
    try:
        return AUTO_TT_PREFIX_PRIORITY.index(prefix)
    except ValueError:
        return len(AUTO_TT_PREFIX_PRIORITY)


def _auto_feature_columns(
    df: pd.DataFrame,
    include_global_rate: bool = True,
    global_rate_col: str = "events_per_second_global_rate",
) -> list[str]:
    """Select TT-rate features from the most advanced available task prefix."""
    by_prefix: dict[str, list[str]] = {}
    fallback_non_tt_rate_cols: list[str] = []

    for col in df.columns:
        text = str(col)
        if not text.endswith("_rate_hz"):
            continue
        match = TT_RATE_COLUMN_RE.match(text)
        if match is None:
            fallback_non_tt_rate_cols.append(text)
            continue
        prefix = str(match.group("prefix")).strip()
        label = match.group("label")
        if not _is_multi_plane_tt_label(label):
            continue
        by_prefix.setdefault(prefix, []).append(text)

    cols: list[str] = []
    if by_prefix:
        selected_prefix = min(
            by_prefix.keys(),
            key=lambda p: (_prefix_rank(p), -len(by_prefix[p]), p),
        )
        cols = sorted(set(by_prefix[selected_prefix]))
    elif fallback_non_tt_rate_cols:
        cols = sorted(set(fallback_non_tt_rate_cols))

    return cols


# ── Derived feature mode ─────────────────────────────────────────────
# Empirical efficiencies separate the efficiency signal from the flux
# signal, breaking the degeneracy that afflicts raw TT rates (which are
# all proportional to flux × f(efficiency)).  The global rate carries
# the overall flux scale once efficiencies are controlled for.
DERIVED_EFFICIENCY_COLUMNS = [
    "eff_empirical_1",
    "eff_empirical_2",
    "eff_empirical_3",
    "eff_empirical_4",
]
DERIVED_RATE_COLUMN = "events_per_second_global_rate"
DERIVED_TT_GLOBAL_RATE_COL = "__derived_tt_global_rate_hz"
DERIVED_EFF_PRODUCT_COL = "__derived_eff_emp_product"
DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL = "__derived_log_tt_rate_over_eff_product"
DERIVED_LOG_EFF_PAIR_SUM_COL = "__derived_log_eff_pair_sum"
DERIVED_LOG_EFF_TRIPLET_SUM_COL = "__derived_log_eff_triplet_sum"

_DERIVED_PHYSICS_FEATURE_ALIASES = {
    "eff_product": "eff_product",
    "emp_eff_product": "eff_product",
    "eff_emp_product": "eff_product",
    "eff_coincidence_moments": "eff_coincidence_moments",
    "eff_pair_triplet_products": "eff_coincidence_moments",
    "pair_triplet_eff_products": "eff_coincidence_moments",
    "eff_pair_triplet_sums": "eff_coincidence_moments",
    "coincidence_eff_moments": "eff_coincidence_moments",
    "log_rate_over_eff_product": "log_rate_over_eff_product",
    "log_tt_rate_over_eff_product": "log_rate_over_eff_product",
    "log_rate_div_eff_product": "log_rate_over_eff_product",
    "log_rate_over_effprod": "log_rate_over_eff_product",
}


def _normalize_prefix_selector(value: object) -> str:
    text = str(value).strip().lower()
    return text if text else "last"


def _normalize_trigger_type_allowlist(value: Sequence[str] | str | None) -> list[str]:
    raw = parse_explicit_feature_columns(value)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        norm = _normalize_tt_label(item)
        if not norm or not _is_multi_plane_tt_label(norm):
            continue
        if norm in seen:
            continue
        out.append(norm)
        seen.add(norm)
    return out


def _select_tt_rate_columns_for_prefix(
    df: pd.DataFrame,
    *,
    prefix_selector: str = "last",
    trigger_type_allowlist: Sequence[str] | str | None = None,
    include_to_tt_rate_hz: bool = False,
) -> list[str]:
    by_prefix: dict[str, list[str]] = {}
    allowlist = set(_normalize_trigger_type_allowlist(trigger_type_allowlist))
    for col in df.columns:
        text = str(col).strip()
        match = TT_RATE_COLUMN_RE.match(text)
        if match is None:
            continue
        label = _normalize_tt_label(match.group("label"))
        if not _is_multi_plane_tt_label(label):
            continue
        if allowlist and label not in allowlist:
            continue
        prefix = str(match.group("prefix")).strip()
        if not include_to_tt_rate_hz and "_to_" in prefix:
            continue
        by_prefix.setdefault(prefix, []).append(text)

    if not by_prefix:
        return []

    prefix_mode = _normalize_prefix_selector(prefix_selector)
    if prefix_mode == "last":
        selected_prefix = min(
            by_prefix.keys(),
            key=lambda p: (_prefix_rank(p), -len(by_prefix[p]), p),
        )
        return sorted(set(by_prefix[selected_prefix]))
    return sorted(set(by_prefix.get(prefix_mode, [])))


def _common_tt_rate_columns(
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    *,
    prefix_selector: str = "last",
    trigger_type_allowlist: Sequence[str] | str | None = None,
    include_to_tt_rate_hz: bool = False,
) -> list[str]:
    from_dict = _select_tt_rate_columns_for_prefix(
        dict_df,
        prefix_selector=prefix_selector,
        trigger_type_allowlist=trigger_type_allowlist,
        include_to_tt_rate_hz=include_to_tt_rate_hz,
    )
    from_data = _select_tt_rate_columns_for_prefix(
        data_df,
        prefix_selector=prefix_selector,
        trigger_type_allowlist=trigger_type_allowlist,
        include_to_tt_rate_hz=include_to_tt_rate_hz,
    )
    return sorted(set(from_dict) & set(from_data))


def _append_derived_tt_global_rate_column(
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    *,
    prefix_selector: str = "last",
    trigger_type_allowlist: Sequence[str] | str | None = None,
    include_to_tt_rate_hz: bool = False,
    output_column: str = DERIVED_TT_GLOBAL_RATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None, list[str]]:
    """
    Build a derived global-rate feature from trigger-type sums.

    The derived column is added only when a common TT column set exists
    in dictionary and dataset.
    """
    common_tt = _common_tt_rate_columns(
        dict_df,
        data_df,
        prefix_selector=prefix_selector,
        trigger_type_allowlist=trigger_type_allowlist,
        include_to_tt_rate_hz=include_to_tt_rate_hz,
    )
    if not common_tt:
        return dict_df, data_df, None, []

    dict_out = dict_df.copy()
    data_out = data_df.copy()
    dict_out[output_column] = dict_out[common_tt].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    data_out[output_column] = data_out[common_tt].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    return dict_out, data_out, output_column, common_tt


def _rate_histogram_columns(df: pd.DataFrame) -> list[str]:
    cols: list[tuple[int, str]] = []
    for col in df.columns:
        text = str(col).strip()
        match = RATE_HISTOGRAM_BIN_RE.match(text)
        if match is None:
            continue
        cols.append((int(match.group("bin")), text))
    cols.sort(key=lambda x: x[0])
    return [name for _, name in cols]


def _normalize_derived_physics_features(value: Sequence[str] | str | bool | None) -> list[str]:
    if isinstance(value, bool):
        return ["log_rate_over_eff_product"] if value else []
    raw = parse_explicit_feature_columns(value)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = str(item).strip().lower()
        if not key:
            continue
        normalized = _DERIVED_PHYSICS_FEATURE_ALIASES.get(key)
        if normalized is None or normalized in seen:
            continue
        out.append(normalized)
        seen.add(normalized)
    return out


def _append_derived_physics_feature_columns(
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    *,
    rate_column: str,
    physics_features: Sequence[str] | str | bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    selected = _normalize_derived_physics_features(physics_features)
    if not selected:
        return dict_df, data_df, []
    if rate_column not in dict_df.columns or rate_column not in data_df.columns:
        return dict_df, data_df, []

    eff_cols = [c for c in DERIVED_EFFICIENCY_COLUMNS if c in dict_df.columns and c in data_df.columns]
    if not eff_cols:
        return dict_df, data_df, []

    dict_out = dict_df.copy()
    data_out = data_df.copy()

    dict_eff = dict_out[eff_cols].apply(pd.to_numeric, errors="coerce")
    data_eff = data_out[eff_cols].apply(pd.to_numeric, errors="coerce")
    min_count = len(eff_cols)
    dict_eff_prod = dict_eff.prod(axis=1, min_count=min_count)
    data_eff_prod = data_eff.prod(axis=1, min_count=min_count)

    def _sum_eff_products(eff_frame: pd.DataFrame, order: int) -> pd.Series:
        arr = eff_frame.to_numpy(dtype=float)
        n_rows, n_eff = arr.shape
        out = np.full(n_rows, np.nan, dtype=float)
        if n_eff < order:
            return pd.Series(out, index=eff_frame.index)
        finite_all = np.isfinite(arr).all(axis=1)
        if not np.any(finite_all):
            return pd.Series(out, index=eff_frame.index)
        total = np.zeros(n_rows, dtype=float)
        for idxs in combinations(range(n_eff), order):
            term = np.ones(n_rows, dtype=float)
            for idx in idxs:
                term *= arr[:, idx]
            total += term
        out[finite_all] = total[finite_all]
        return pd.Series(out, index=eff_frame.index)

    added: list[str] = []
    if "eff_product" in selected:
        dict_out[DERIVED_EFF_PRODUCT_COL] = dict_eff_prod
        data_out[DERIVED_EFF_PRODUCT_COL] = data_eff_prod
        added.append(DERIVED_EFF_PRODUCT_COL)

    if "eff_coincidence_moments" in selected:
        dict_pair_sum = _sum_eff_products(dict_eff, order=2)
        data_pair_sum = _sum_eff_products(data_eff, order=2)
        dict_triplet_sum = _sum_eff_products(dict_eff, order=3)
        data_triplet_sum = _sum_eff_products(data_eff, order=3)

        dict_out[DERIVED_LOG_EFF_PAIR_SUM_COL] = np.log(
            np.where(dict_pair_sum > 1e-12, dict_pair_sum, np.nan)
        )
        data_out[DERIVED_LOG_EFF_PAIR_SUM_COL] = np.log(
            np.where(data_pair_sum > 1e-12, data_pair_sum, np.nan)
        )
        dict_out[DERIVED_LOG_EFF_TRIPLET_SUM_COL] = np.log(
            np.where(dict_triplet_sum > 1e-12, dict_triplet_sum, np.nan)
        )
        data_out[DERIVED_LOG_EFF_TRIPLET_SUM_COL] = np.log(
            np.where(data_triplet_sum > 1e-12, data_triplet_sum, np.nan)
        )
        added.extend([DERIVED_LOG_EFF_PAIR_SUM_COL, DERIVED_LOG_EFF_TRIPLET_SUM_COL])

    if "log_rate_over_eff_product" in selected:
        dict_rate = pd.to_numeric(dict_out[rate_column], errors="coerce")
        data_rate = pd.to_numeric(data_out[rate_column], errors="coerce")

        dict_den = dict_eff_prod.where(dict_eff_prod > 1e-8, np.nan)
        data_den = data_eff_prod.where(data_eff_prod > 1e-8, np.nan)

        dict_ratio = (dict_rate / dict_den).where(dict_rate > 0.0, np.nan)
        data_ratio = (data_rate / data_den).where(data_rate > 0.0, np.nan)

        dict_out[DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL] = np.log(
            np.where(np.isfinite(dict_ratio), dict_ratio, np.nan)
        )
        data_out[DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL] = np.log(
            np.where(np.isfinite(data_ratio), data_ratio, np.nan)
        )
        added.append(DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL)

    return dict_out, data_out, added


def _derived_feature_columns(
    df: pd.DataFrame,
    *,
    rate_column: str = DERIVED_TT_GLOBAL_RATE_COL,
    trigger_type_rate_columns: Sequence[str] | None = None,
    include_rate_histogram: bool = False,
    physics_feature_columns: Sequence[str] | None = None,
) -> list[str]:
    """Select derived features: empirical efficiencies + global-rate proxy.

    The preferred global-rate proxy is the derived TT trigger sum.
    `events_per_second_global_rate` remains supported as a fallback.
    """
    cols: list[str] = []
    for c in DERIVED_EFFICIENCY_COLUMNS:
        if c in df.columns:
            cols.append(c)
    if rate_column in df.columns:
        cols.append(rate_column)
    elif rate_column != DERIVED_RATE_COLUMN and DERIVED_RATE_COLUMN in df.columns:
        cols.append(DERIVED_RATE_COLUMN)
    if trigger_type_rate_columns:
        seen = set(cols)
        for c in trigger_type_rate_columns:
            name = str(c).strip()
            if not name or name in seen:
                continue
            if name in df.columns:
                cols.append(name)
                seen.add(name)
    if include_rate_histogram:
        cols.extend(_rate_histogram_columns(df))
    if physics_feature_columns:
        for c in physics_feature_columns:
            name = str(c).strip()
            if name and name in df.columns:
                cols.append(name)
    return cols


def _append_tt_global_sum_feature(
    dict_features: pd.DataFrame,
    data_features: pd.DataFrame,
    feature_cols: Sequence[str],
    include_global_rate: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str | None, list[str]]:
    """
    Optionally append a derived global-rate-like feature as the sum of
    selected multi-plane TT rate columns.

    This intentionally does not rely on a literal `global_rate_col`.
    """
    out_cols = [str(c) for c in feature_cols]
    if not include_global_rate:
        return dict_features, data_features, out_cols, None, []

    tt_cols: list[str] = []
    for col in out_cols:
        match = TT_RATE_COLUMN_RE.match(str(col))
        if match is None:
            continue
        if not _is_multi_plane_tt_label(match.group("label")):
            continue
        tt_cols.append(str(col))

    if not tt_cols:
        return dict_features, data_features, out_cols, None, []

    alias = DERIVED_TT_GLOBAL_RATE_COL
    idx = 2
    while alias in dict_features.columns or alias in data_features.columns or alias in out_cols:
        alias = f"{DERIVED_TT_GLOBAL_RATE_COL}_{idx}"
        idx += 1

    dict_out = dict_features.copy()
    data_out = data_features.copy()
    dict_out[alias] = dict_out[tt_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    data_out[alias] = data_out[tt_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    out_cols.append(alias)
    return dict_out, data_out, out_cols, alias, tt_cols


def _mask_out_of_support_efficiency_features(
    *,
    dict_features: pd.DataFrame,
    data_features: pd.DataFrame,
    feature_cols: Sequence[str],
    non_hist_feature_idx: Sequence[int],
    base_eff_feature_idx: np.ndarray,
    enabled: bool,
) -> tuple[pd.DataFrame, np.ndarray | None, dict]:
    """
    Mask empirical-efficiency feature values outside dictionary support.

    For inverse mapping, distances are only meaningful inside the dictionary
    feature support. Per-row masking avoids forcing boundary matches when real
    data falls outside support in some efficiency planes.
    """
    info: dict = {
        "enabled": bool(enabled),
        "n_eff_features_considered": int(base_eff_feature_idx.size),
        "n_rows_any_masked": 0,
        "fraction_rows_any_masked": 0.0,
        "mean_masked_eff_features_per_row": 0.0,
        "per_feature": {},
    }
    if not enabled or base_eff_feature_idx.size == 0:
        return data_features, None, info

    data_out = data_features.copy()
    n_data = len(data_out)
    if n_data == 0:
        return data_out, np.zeros((0, len(non_hist_feature_idx)), dtype=bool), info

    masked_non_hist = np.zeros((n_data, len(non_hist_feature_idx)), dtype=bool)
    for j_non_hist in base_eff_feature_idx:
        if j_non_hist < 0 or j_non_hist >= len(non_hist_feature_idx):
            continue
        full_idx = int(non_hist_feature_idx[int(j_non_hist)])
        if full_idx < 0 or full_idx >= len(feature_cols):
            continue
        col = str(feature_cols[full_idx])

        dvals = pd.to_numeric(dict_features[col], errors="coerce")
        finite_d = np.isfinite(dvals.to_numpy(dtype=float))
        if not np.any(finite_d):
            info["per_feature"][col] = {
                "dict_min": None,
                "dict_max": None,
                "rows_masked": 0,
                "fraction_rows_masked": 0.0,
                "reason": "no_finite_dictionary_values",
            }
            continue

        dict_min = float(np.nanmin(dvals.to_numpy(dtype=float)))
        dict_max = float(np.nanmax(dvals.to_numpy(dtype=float)))
        tol = max(1e-12, 1e-12 * max(1.0, abs(dict_min), abs(dict_max)))
        x = pd.to_numeric(data_out[col], errors="coerce")
        outside = x.notna() & ((x < (dict_min - tol)) | (x > (dict_max + tol)))
        n_mask = int(outside.sum())
        if n_mask > 0:
            data_out.loc[outside, col] = np.nan
            masked_non_hist[outside.to_numpy(dtype=bool), int(j_non_hist)] = True

        info["per_feature"][col] = {
            "dict_min": dict_min,
            "dict_max": dict_max,
            "rows_masked": n_mask,
            "fraction_rows_masked": float(n_mask / n_data),
        }

    if masked_non_hist.size:
        row_counts = np.sum(masked_non_hist, axis=1)
        rows_any = row_counts > 0
        info["n_rows_any_masked"] = int(np.sum(rows_any))
        info["fraction_rows_any_masked"] = float(np.mean(rows_any))
        info["mean_masked_eff_features_per_row"] = float(np.mean(row_counts))

    return data_out, masked_non_hist, info


# =====================================================================
# Density-aware interpolation helpers
# =====================================================================

_DENSITY_CFG_DEFAULTS = {
    "density_correction_enabled": True,
    "density_correction_k_neighbors": 10,
}
_DENSITY_CORRECTION_EXPONENT = 1.0
_DENSITY_CORRECTION_CLIP_MIN = 0.25
_DENSITY_CORRECTION_CLIP_MAX = 4.0

_INVERSE_MAPPING_DEFAULTS = {
    "estimation_mode": "single_stage",
    "neighbor_selection": "knn",
    "neighbor_count": 5,
    "weighting": "inverse_distance",
    "inverse_distance_power": 2.0,
    "softmax_temperature": 1.0,
    "distance_floor": 1e-12,
    "aggregation": "weighted_mean",
    "stage2_efficiency_conditioning_weight": 1.0,
    "stage2_efficiency_gate_max": 0.20,
    "stage2_efficiency_gate_min_candidates": 24,
    "stage2_use_rate_histogram": True,
    "stage2_histogram_distance_weight": 1.0,
    "histogram_distance_weight": 1.0,
    "histogram_distance_blend_mode": "normalized",
    # When real-data empirical efficiencies lie outside dictionary support,
    # distance in those dimensions is not meaningful. Mask them per-row.
    "mask_out_of_support_eff_features": False,
}


def _safe_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isfinite(out):
        return out
    return float(default)


def _safe_int(value: object, default: int, minimum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if minimum is not None:
        out = max(int(minimum), out)
    return out


def _safe_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _safe_str(value: object, default: str) -> str:
    if value is None:
        return str(default)
    text = str(value).strip()
    return text if text else str(default)


def _normalize_column_spec(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [x.strip() for x in text.split(",") if x.strip()]
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _normalize_neighbor_selection(value: object) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"nearest", "nn", "1nn", "single"}:
        return "nearest"
    if text in {"all", "full", "global"}:
        return "all"
    if text in {"knn", "k", "k_nearest", "k-nearest"}:
        return "knn"
    return "knn"


def _normalize_weighting_mode(value: object) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"uniform", "flat", "mean"}:
        return "uniform"
    if text in {"softmax", "exp", "exponential"}:
        return "softmax"
    if text in {"inverse_distance", "idw", "inverse", "invdist"}:
        return "inverse_distance"
    return "inverse_distance"


def _normalize_aggregation_mode(value: object) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"weighted_median", "median"}:
        return "weighted_median"
    if text in {"local_linear", "local_linear_ridge", "llr"}:
        return "local_linear"
    return "weighted_mean"


def _normalize_hist_blend_mode(value: object) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"raw", "add_raw", "legacy"}:
        return "raw"
    return "normalized"


def _normalize_estimation_mode(value: object) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in {
        "two_stage_eff_address_then_flux",
        "two_stage_address",
        "two_stage_address_then_flux",
        "eff_address_then_flux",
        "2stage_address",
    }:
        return "two_stage_eff_address_then_flux"
    if text in {"two_stage", "two_stage_eff_then_flux", "eff_then_flux", "2stage"}:
        return "two_stage_eff_then_flux"
    return "single_stage"


def _legacy_neighbor_selection_from_interpolation_k(
    interpolation_k: int | None,
) -> tuple[str, int | None]:
    if interpolation_k is None:
        return ("all", None)
    k = max(1, int(interpolation_k))
    if k <= 1:
        return ("nearest", 1)
    return ("knn", k)


def resolve_inverse_mapping_cfg(
    *,
    inverse_mapping_cfg: Mapping[str, object] | None,
    interpolation_k: int | None = 5,
    histogram_distance_weight: float = 1.0,
    histogram_distance_blend_mode: str = "normalized",
) -> dict:
    """
    Resolve inverse-mapping behavior into a normalized configuration dict.

    Keeps backward compatibility with `interpolation_k` and legacy histogram
    distance arguments while allowing explicit control in `inverse_mapping_cfg`.
    """
    legacy_neighbor_selection, legacy_neighbor_count = _legacy_neighbor_selection_from_interpolation_k(
        interpolation_k
    )
    cfg = dict(_INVERSE_MAPPING_DEFAULTS)
    cfg["neighbor_selection"] = legacy_neighbor_selection
    cfg["neighbor_count"] = legacy_neighbor_count
    cfg["histogram_distance_weight"] = max(float(histogram_distance_weight), 0.0)
    cfg["histogram_distance_blend_mode"] = _normalize_hist_blend_mode(histogram_distance_blend_mode)

    if isinstance(inverse_mapping_cfg, Mapping):
        for key in (
            "estimation_mode",
            "neighbor_selection",
            "neighbor_count",
            "weighting",
            "inverse_distance_power",
            "softmax_temperature",
            "distance_floor",
            "aggregation",
            "stage2_efficiency_conditioning_weight",
            "stage2_efficiency_gate_max",
            "stage2_efficiency_gate_min_candidates",
            "stage2_use_rate_histogram",
            "stage2_histogram_distance_weight",
            "histogram_distance_weight",
            "histogram_distance_blend_mode",
            "mask_out_of_support_eff_features",
        ):
            if key in inverse_mapping_cfg and inverse_mapping_cfg.get(key) is not None:
                cfg[key] = inverse_mapping_cfg.get(key)

    cfg["estimation_mode"] = _normalize_estimation_mode(cfg.get("estimation_mode"))
    cfg["neighbor_selection"] = _normalize_neighbor_selection(cfg.get("neighbor_selection"))
    if cfg["neighbor_selection"] == "nearest":
        cfg["neighbor_count"] = 1
    elif cfg["neighbor_selection"] == "all":
        cfg["neighbor_count"] = None
    else:
        if cfg.get("neighbor_count") is None:
            cfg["neighbor_count"] = max(
                1, int(legacy_neighbor_count if legacy_neighbor_count is not None else _INVERSE_MAPPING_DEFAULTS["neighbor_count"])
            )
        else:
            cfg["neighbor_count"] = max(1, _safe_int(cfg.get("neighbor_count"), int(_INVERSE_MAPPING_DEFAULTS["neighbor_count"]), minimum=1))

    cfg["weighting"] = _normalize_weighting_mode(cfg.get("weighting"))
    cfg["inverse_distance_power"] = max(
        _safe_float(cfg.get("inverse_distance_power"), float(_INVERSE_MAPPING_DEFAULTS["inverse_distance_power"])),
        0.0,
    )
    cfg["softmax_temperature"] = max(
        _safe_float(cfg.get("softmax_temperature"), float(_INVERSE_MAPPING_DEFAULTS["softmax_temperature"])),
        1e-12,
    )
    cfg["distance_floor"] = max(
        _safe_float(cfg.get("distance_floor"), float(_INVERSE_MAPPING_DEFAULTS["distance_floor"])),
        1e-15,
    )
    cfg["aggregation"] = _normalize_aggregation_mode(cfg.get("aggregation"))
    cfg["stage2_efficiency_conditioning_weight"] = max(
        _safe_float(
            cfg.get("stage2_efficiency_conditioning_weight"),
            float(_INVERSE_MAPPING_DEFAULTS["stage2_efficiency_conditioning_weight"]),
        ),
        0.0,
    )
    cfg["stage2_efficiency_gate_max"] = max(
        _safe_float(
            cfg.get("stage2_efficiency_gate_max"),
            float(_INVERSE_MAPPING_DEFAULTS["stage2_efficiency_gate_max"]),
        ),
        0.0,
    )
    cfg["stage2_efficiency_gate_min_candidates"] = max(
        1,
        _safe_int(
            cfg.get("stage2_efficiency_gate_min_candidates"),
            int(_INVERSE_MAPPING_DEFAULTS["stage2_efficiency_gate_min_candidates"]),
            minimum=1,
        ),
    )
    cfg["stage2_use_rate_histogram"] = _safe_bool(
        cfg.get("stage2_use_rate_histogram"),
        bool(_INVERSE_MAPPING_DEFAULTS["stage2_use_rate_histogram"]),
    )
    cfg["stage2_histogram_distance_weight"] = max(
        _safe_float(
            cfg.get("stage2_histogram_distance_weight"),
            float(_INVERSE_MAPPING_DEFAULTS["stage2_histogram_distance_weight"]),
        ),
        0.0,
    )
    cfg["histogram_distance_weight"] = max(
        _safe_float(cfg.get("histogram_distance_weight"), float(_INVERSE_MAPPING_DEFAULTS["histogram_distance_weight"])),
        0.0,
    )
    cfg["histogram_distance_blend_mode"] = _normalize_hist_blend_mode(
        cfg.get("histogram_distance_blend_mode")
    )
    cfg["mask_out_of_support_eff_features"] = _safe_bool(
        cfg.get("mask_out_of_support_eff_features"),
        bool(_INVERSE_MAPPING_DEFAULTS["mask_out_of_support_eff_features"]),
    )
    return cfg


def _resolve_shared_parameter_exclusion_mode(
    mode_value: object | None,
    legacy_flag: bool = False,
) -> str:
    """
    Normalize shared-parameter exclusion mode.

    Returns one of: "off", "full", "partial".
    """
    if mode_value is None:
        mode_value = legacy_flag

    if isinstance(mode_value, bool):
        return "partial" if mode_value else "off"

    text = str(mode_value).strip().lower()
    if text in {"", "off", "none", "false", "0", "no", "n"}:
        return "off"
    if text in {"full", "all", "exact"}:
        return "full"
    if text in {"partial", "any", "true", "1", "yes", "y", "on"}:
        return "partial"
    # Safe fallback keeps exclusion active when user provides an unknown
    # truthy-like value.
    return "partial"


def _default_parameter_columns_for_exclusion(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in [
        "flux_cm2_min",
        "cos_n",
        "eff_sim_1",
        "eff_sim_2",
        "eff_sim_3",
        "eff_sim_4",
        "eff_empirical_1",
        "eff_empirical_2",
        "eff_empirical_3",
        "eff_empirical_4",
    ]:
        if c in df.columns:
            cols.append(c)
    return cols


def build_shared_parameter_exclusion_mask(
    *,
    dict_df: pd.DataFrame,
    sample_row: pd.Series,
    initial_mask: np.ndarray | None = None,
    param_columns: Sequence[str] | None = None,
    shared_parameter_exclusion_mode: str | None = None,
    shared_parameter_exclusion_columns: Sequence[str] | str | None = None,
    shared_parameter_exclusion_ignore: Sequence[str] | str | None = ("cos_n",),
    shared_parameter_match_atol: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """
    Apply STEP 2 shared-parameter exclusion logic for one sample row.

    Returns:
      keep_mask (np.ndarray[bool]),
      info dict with counts/columns/mode.
    """
    n_rows = len(dict_df)
    if initial_mask is None:
        keep_mask = np.ones(n_rows, dtype=bool)
    else:
        keep_mask = np.asarray(initial_mask, dtype=bool).copy()
        if keep_mask.shape[0] != n_rows:
            keep_mask = np.ones(n_rows, dtype=bool)

    mode = _resolve_shared_parameter_exclusion_mode(shared_parameter_exclusion_mode, legacy_flag=False)
    info = {
        "mode": mode,
        "enabled": mode in {"partial", "full"},
        "columns": [],
        "n_before": int(np.sum(keep_mask)),
        "n_removed": 0,
        "reason": "disabled",
    }
    if not info["enabled"] or info["n_before"] <= 0:
        return keep_mask, info

    candidate_param_cols = list(param_columns) if param_columns is not None else _default_parameter_columns_for_exclusion(dict_df)
    raw_cols = _normalize_column_spec(shared_parameter_exclusion_columns)
    if not raw_cols or any(str(c).strip().lower() == "auto" for c in raw_cols):
        raw_cols = [str(c) for c in candidate_param_cols if not str(c).startswith("eff_empirical_")]

    ignore_cols = set(_normalize_column_spec(shared_parameter_exclusion_ignore))
    shared_cols: list[str] = []
    seen_cols: set[str] = set()
    for col in raw_cols:
        name = str(col).strip()
        if not name or name in seen_cols or name in ignore_cols:
            continue
        if name not in dict_df.columns or name not in sample_row.index:
            continue
        shared_cols.append(name)
        seen_cols.add(name)
    info["columns"] = shared_cols

    shared_match_atol = max(float(shared_parameter_match_atol), 0.0)
    remove_any = np.zeros(n_rows, dtype=bool)
    remove_full = np.ones(n_rows, dtype=bool)
    n_compared_cols = 0

    for col in shared_cols:
        sample_val = float(pd.to_numeric(pd.Series([sample_row.get(col)]), errors="coerce").iloc[0])
        if not np.isfinite(sample_val):
            continue
        n_compared_cols += 1
        dict_vals = pd.to_numeric(dict_df[col], errors="coerce").to_numpy(dtype=float)
        atol_eff = _effective_atol(shared_match_atol, sample_val)
        same_mask = np.isfinite(dict_vals) & np.isclose(
            dict_vals,
            sample_val,
            rtol=0.0,
            atol=atol_eff,
        )
        if np.any(same_mask):
            remove_any |= same_mask
        remove_full &= same_mask

    if mode == "full":
        remove_basis = remove_full if n_compared_cols > 0 else np.zeros(n_rows, dtype=bool)
    else:
        remove_basis = remove_any

    if (
        "param_set_id" in dict_df.columns
        and "param_set_id" in sample_row.index
        and "param_set_id" not in ignore_cols
    ):
        dict_param_set = pd.to_numeric(dict_df["param_set_id"], errors="coerce").to_numpy(dtype=float)
        sample_param_set = float(pd.to_numeric(pd.Series([sample_row.get("param_set_id")]), errors="coerce").iloc[0])
        if np.isfinite(sample_param_set):
            atol_eff = _effective_atol(shared_match_atol, sample_param_set)
            same_param_set = np.isfinite(dict_param_set) & np.isclose(
                dict_param_set,
                sample_param_set,
                rtol=0.0,
                atol=atol_eff,
            )
            if np.any(same_param_set):
                remove_basis |= same_param_set

    remove_in_candidates = keep_mask & remove_basis
    n_removed = int(np.sum(remove_in_candidates))
    if n_removed > 0:
        keep_mask[remove_in_candidates] = False
        info["reason"] = "removed"
    else:
        info["reason"] = "no_matches"
    info["n_removed"] = n_removed
    info["n_after"] = int(np.sum(keep_mask))
    return keep_mask, info


def _resolve_density_cfg(
    density_weighting_cfg: Mapping[str, object] | None,
) -> dict:
    if not isinstance(density_weighting_cfg, Mapping):
        return {
            **_DENSITY_CFG_DEFAULTS,
            "density_correction_enabled": False,
            "_reason": "not_configured",
        }

    merged = dict(_DENSITY_CFG_DEFAULTS)
    for key in _DENSITY_CFG_DEFAULTS:
        if key in density_weighting_cfg and density_weighting_cfg.get(key) is not None:
            merged[key] = density_weighting_cfg.get(key)
    for legacy_key in (
        "flux_column",
        "eff_column",
        "density_correction_space",
        "density_correction_exponent",
        "density_correction_clip_min",
        "density_correction_clip_max",
    ):
        if density_weighting_cfg.get(legacy_key) is not None:
            log.warning(
                "Deprecated density config key '%s' detected; ignored. "
                "Feature-space density correction now uses fixed exponent/clip constants.",
                legacy_key,
            )

    merged["density_correction_enabled"] = _safe_bool(
        merged.get("density_correction_enabled"), True
    )
    merged["density_correction_k_neighbors"] = _safe_int(
        merged.get("density_correction_k_neighbors"), 10, minimum=1
    )
    return merged


def _scale_from_local_radius(
    local_radius: np.ndarray,
    *,
    exponent: float,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, dict]:
    rr = np.asarray(local_radius, dtype=float)
    n = len(rr)
    scale = np.ones(n, dtype=float)
    info = {"n_basis": int(n), "n_finite_radius": int(np.isfinite(rr).sum())}
    finite = np.isfinite(rr)
    if not np.any(finite):
        info["reason"] = "no_finite_radius"
        return scale, info

    exp = max(float(exponent), 0.0)
    rr_f = np.clip(rr[finite], 1e-12, None)
    scale_f = np.power(rr_f, exp)
    med = float(np.median(scale_f[np.isfinite(scale_f)])) if np.isfinite(scale_f).any() else 1.0
    if med <= 0.0 or not np.isfinite(med):
        med = 1.0
    scale_f = scale_f / med

    lo = max(float(clip_min), 1e-6)
    hi = max(float(clip_max), lo)
    scale_f = np.clip(scale_f, lo, hi)
    scale[finite] = scale_f

    info["scale_min"] = float(np.min(scale_f))
    info["scale_median"] = float(np.median(scale_f))
    info["scale_max"] = float(np.max(scale_f))
    return scale, info


def _z_signature_key(row: np.ndarray) -> tuple[object, ...]:
    vals = np.asarray(row, dtype=float).tolist()
    key: list[object] = []
    for val in vals:
        v = float(val)
        if np.isnan(v):
            key.append(("nan",))
        elif np.isposinf(v):
            key.append(("inf",))
        elif np.isneginf(v):
            key.append(("-inf",))
        else:
            key.append(round(v, 12))
    return tuple(key)


def _group_indices_by_z_signature(
    dict_z: np.ndarray | None,
    n_rows: int,
) -> list[np.ndarray]:
    if n_rows <= 0:
        return []
    if dict_z is None or len(dict_z) != n_rows:
        return [np.arange(n_rows, dtype=int)]

    groups: dict[tuple[object, ...], list[int]] = {}
    for idx in range(n_rows):
        key = _z_signature_key(dict_z[idx])
        groups.setdefault(key, []).append(idx)

    out = [np.asarray(v, dtype=int) for v in groups.values() if len(v) > 0]
    out.sort(key=lambda arr: int(arr[0]))
    return out


def _compute_inverse_density_scaling_feature_space(
    dict_feature_matrix: np.ndarray,
    *,
    distance_metric: str,
    z_groups: list[np.ndarray],
    k_neighbors: int,
    exponent: float,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, dict]:
    X = np.asarray(dict_feature_matrix, dtype=float)
    n = len(X)
    info = {
        "enabled": True,
        "space": "feature",
        "distance_metric": str(distance_metric),
        "k_neighbors": int(k_neighbors),
        "exponent": float(exponent),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "n_basis": int(n),
        "n_z_groups": int(len(z_groups)),
    }
    if n == 0:
        return np.array([], dtype=float), info
    if n == 1:
        return np.ones(1, dtype=float), info

    dist_many_fn = DISTANCE_FNS_MANY.get(distance_metric, None)
    dist_fn = DISTANCE_FNS.get(distance_metric, _l2)
    local_radius = np.full(n, np.nan, dtype=float)
    eff_k_values: list[int] = []
    groups_with_neighbors = 0

    for grp in z_groups:
        if len(grp) <= 1:
            continue

        groups_with_neighbors += 1
        k_eff = min(max(1, int(k_neighbors)), len(grp) - 1)
        eff_k_values.append(int(k_eff))
        Xg = X[grp]

        for local_i, global_i in enumerate(grp):
            if dist_many_fn is not None:
                d = np.asarray(dist_many_fn(Xg[local_i], Xg), dtype=float)
            else:
                d = np.asarray([dist_fn(Xg[local_i], Xg[j]) for j in range(len(grp))], dtype=float)

            if d.shape[0] != len(grp):
                continue

            d[local_i] = np.nan
            finite = np.isfinite(d)
            if not np.any(finite):
                continue
            vals = np.clip(d[finite], 0.0, None)
            if vals.size >= k_eff:
                local_radius[global_i] = float(np.partition(vals, k_eff - 1)[k_eff - 1])
            else:
                local_radius[global_i] = float(np.max(vals))

    scale, scale_info = _scale_from_local_radius(
        local_radius,
        exponent=float(exponent),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
    )
    info.update(scale_info)
    info["n_groups_with_neighbors"] = int(groups_with_neighbors)
    if eff_k_values:
        info["effective_k_neighbors_min"] = int(np.min(eff_k_values))
        info["effective_k_neighbors_max"] = int(np.max(eff_k_values))
    finite_radius = np.isfinite(local_radius)
    if np.any(finite_radius):
        info["radius_min"] = float(np.min(local_radius[finite_radius]))
        info["radius_median"] = float(np.median(local_radius[finite_radius]))
        info["radius_max"] = float(np.max(local_radius[finite_radius]))
    return scale, info


def _build_density_scaling_for_dictionary(
    *,
    dict_df: pd.DataFrame,
    dict_feature_matrix: np.ndarray,
    distance_metric: str,
    dict_z: np.ndarray | None,
    density_weighting_cfg: Mapping[str, object] | None,
) -> tuple[np.ndarray | None, dict]:
    cfg = _resolve_density_cfg(density_weighting_cfg)
    if not cfg.get("density_correction_enabled", False):
        return None, {"enabled": False, "reason": str(cfg.get("_reason", "disabled"))}

    X = np.asarray(dict_feature_matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] != len(dict_df):
        return None, {
            "enabled": False,
            "space": "feature",
            "reason": "invalid_feature_matrix",
            "n_rows": int(len(dict_df)),
            "matrix_shape": tuple(X.shape),
        }

    z_groups = _group_indices_by_z_signature(dict_z, len(dict_df))
    scaling, info = _compute_inverse_density_scaling_feature_space(
        dict_feature_matrix=X,
        distance_metric=distance_metric,
        z_groups=z_groups,
        k_neighbors=int(cfg["density_correction_k_neighbors"]),
        exponent=float(_DENSITY_CORRECTION_EXPONENT),
        clip_min=float(_DENSITY_CORRECTION_CLIP_MIN),
        clip_max=float(_DENSITY_CORRECTION_CLIP_MAX),
    )
    if info.get("n_finite_radius", 0) <= 0:
        return None, {
            "enabled": False,
            "space": "feature",
            "reason": "insufficient_valid_points",
            "n_basis": int(info.get("n_basis", len(dict_df))),
        }
    return scaling, info


# =====================================================================
# Main estimation function
# =====================================================================

def estimate_parameters(
    dictionary_path: str | Path,
    dataset_path: str | Path,
    *,
    feature_columns: str | list[str] = "auto",
    distance_metric: str = "l2_zscore",
    interpolation_k: int | None = 5,
    z_tol: float = 1e-6,
    include_global_rate: bool = True,
    global_rate_col: str = "events_per_second_global_rate",
    param_columns: list[str] | None = None,
    exclude_same_file: bool = True,
    exclude_candidates_sharing_parameter_values: bool = False,
    shared_parameter_exclusion_mode: str | None = None,
    shared_parameter_exclusion_columns: Sequence[str] | str | None = None,
    shared_parameter_exclusion_ignore: Sequence[str] | str | None = ("cos_n",),
    shared_parameter_match_atol: float = 1e-12,
    density_weighting_cfg: Mapping[str, object] | None = None,
    inverse_mapping_cfg: Mapping[str, object] | None = None,
    derived_tt_prefix: str = "last",
    derived_trigger_types: Sequence[str] | str | None = None,
    derived_include_to_tt_rate_hz: bool = False,
    derived_include_trigger_type_rates: bool = False,
    derived_include_rate_histogram: bool = False,
    derived_physics_features: Sequence[str] | str | bool | None = None,
    histogram_distance_weight: float = 1.0,
    histogram_distance_blend_mode: str = "normalized",
) -> pd.DataFrame:
    """Estimate physical parameters for each dataset entry using dictionary matching.

    Parameters
    ----------
    dictionary_path : path
        CSV with dictionary entries (must have z_plane_1..4, feature columns,
        and parameter columns like flux_cm2_min, cos_n, etc.).
    dataset_path : path
        CSV with dataset entries to estimate parameters for.
    feature_columns : "auto" or list of str
        Columns used to build the fingerprint vector. If "auto", uses
        the most advanced available *_tt_*_rate_hz prefix and only
        multi-plane topologies (2-, 3-, and 4-plane combinations).
    distance_metric : str
        One of "l2_zscore", "l2_raw", "chi2", "poisson".
    interpolation_k : int or None
        Number of nearest neighbouring dictionary entries for IDW
        interpolation (1 = nearest-only). If None, use all valid
        candidates.
    z_tol : float
        Tolerance for z-plane position matching.
    include_global_rate : bool
        If True, append a derived global-rate-like feature computed as
        the row-wise sum of selected multi-plane TT rate features.
    global_rate_col : str
        Legacy compatibility argument (not required for matching when
        include_global_rate=True).
    param_columns : list of str or None
        Parameter columns to estimate. If None, auto-detected from
        dictionary (flux_cm2_min, cos_n, eff_sim_1..4).
    exclude_same_file : bool
        If True, exclude dictionary entries with the same filename_base
        as the dataset row being estimated.
    exclude_candidates_sharing_parameter_values : bool
        If True, remove candidate dictionary rows that share any configured
        parameter value with the current dataset row (after same-file and
        z-compatibility filtering).
    shared_parameter_exclusion_mode : str | None
        Shared-value exclusion mode: "partial" removes candidates sharing
        any selected parameter; "full" removes only full coincidences
        across all selected parameters. None falls back to
        exclude_candidates_sharing_parameter_values.
    shared_parameter_exclusion_columns : list[str] | str | None
        Parameter columns used for shared-value exclusion. `None` or `"auto"`
        uses estimated parameter columns.
    shared_parameter_exclusion_ignore : list[str] | str | None
        Columns excluded from shared-value matching (default: cos_n).
    shared_parameter_match_atol : float
        Absolute tolerance for numeric equality checks in shared-parameter
        exclusion.
    density_weighting_cfg : dict or None
        Optional density-correction config. When provided and enabled,
        IDW interpolation weights are multiplied by inverse local-density
        scaling in feature space (same feature columns + distance metric
        as matching). Only `density_correction_enabled` and
        `density_correction_k_neighbors` are configurable.
    inverse_mapping_cfg : dict or None
        Optional controls for inverse mapping from feature-space neighbors
        to parameter estimates. Supported keys include:
        neighbor_selection (nearest|knn|all), neighbor_count, weighting
        (uniform|inverse_distance|softmax), inverse_distance_power,
        softmax_temperature, distance_floor, aggregation
        (weighted_mean|weighted_median|local_linear), histogram_distance_weight,
        histogram_distance_blend_mode (normalized|raw),
        mask_out_of_support_eff_features (bool; default True).
    derived_tt_prefix : str
        Prefix selector for derived mode TT-sum global rate:
        "last" (recommended) or explicit prefix name.
    derived_trigger_types : list[str] | str | None
        Optional TT label allowlist used in derived mode trigger-sum
        construction (e.g. "1234,123,234,124,134"). Empty/None uses all
        available multi-plane trigger types.
    derived_include_to_tt_rate_hz : bool
        If True, allow prefixes containing "_to_" when selecting TT
        columns for derived trigger-sum global rate.
    derived_include_trigger_type_rates : bool
        If True, append selected trigger-type rate columns to derived-mode
        features (in addition to empirical efficiencies and derived TT-sum
        global rate).
    derived_include_rate_histogram : bool
        If True, append `events_per_second_<bin>_rate_hz` histogram bins
        to derived-mode features.
    derived_physics_features : list[str] | str | bool | None
        Optional derived-mode physics transforms to append as features.
        Supported values: "log_rate_over_eff_product", "eff_product",
        "eff_coincidence_moments" (log-sum of pair/triplet efficiency products).
        A boolean True enables "log_rate_over_eff_product".
    histogram_distance_weight : float
        Legacy alias for `inverse_mapping_cfg.histogram_distance_weight`.
    histogram_distance_blend_mode : str
        Legacy alias for `inverse_mapping_cfg.histogram_distance_blend_mode`.

    Returns
    -------
    pd.DataFrame
        One row per dataset entry with columns:
        - dataset_index, filename_base
        - est_<param> for each estimated parameter
        - n_candidates, best_distance
        - all original dataset columns prefixed with data_
    """
    dict_df = pd.read_csv(dictionary_path, low_memory=False)
    data_df = pd.read_csv(dataset_path, low_memory=False)

    return estimate_from_dataframes(
        dict_df=dict_df,
        data_df=data_df,
        feature_columns=feature_columns,
        distance_metric=distance_metric,
        interpolation_k=interpolation_k,
        z_tol=z_tol,
        include_global_rate=include_global_rate,
        global_rate_col=global_rate_col,
        param_columns=param_columns,
        exclude_same_file=exclude_same_file,
        exclude_candidates_sharing_parameter_values=exclude_candidates_sharing_parameter_values,
        shared_parameter_exclusion_mode=shared_parameter_exclusion_mode,
        shared_parameter_exclusion_columns=shared_parameter_exclusion_columns,
        shared_parameter_exclusion_ignore=shared_parameter_exclusion_ignore,
        shared_parameter_match_atol=shared_parameter_match_atol,
        density_weighting_cfg=density_weighting_cfg,
        inverse_mapping_cfg=inverse_mapping_cfg,
        derived_tt_prefix=derived_tt_prefix,
        derived_trigger_types=derived_trigger_types,
        derived_include_to_tt_rate_hz=derived_include_to_tt_rate_hz,
        derived_include_trigger_type_rates=derived_include_trigger_type_rates,
        derived_include_rate_histogram=derived_include_rate_histogram,
        derived_physics_features=derived_physics_features,
        histogram_distance_weight=histogram_distance_weight,
        histogram_distance_blend_mode=histogram_distance_blend_mode,
    )


def estimate_from_dataframes(
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    *,
    feature_columns: str | list[str] = "auto",
    distance_metric: str = "l2_zscore",
    interpolation_k: int | None = 5,
    z_tol: float = 1e-6,
    include_global_rate: bool = True,
    global_rate_col: str = "events_per_second_global_rate",
    param_columns: list[str] | None = None,
    exclude_same_file: bool = True,
    exclude_candidates_sharing_parameter_values: bool = False,
    shared_parameter_exclusion_mode: str | None = None,
    shared_parameter_exclusion_columns: Sequence[str] | str | None = None,
    shared_parameter_exclusion_ignore: Sequence[str] | str | None = ("cos_n",),
    shared_parameter_match_atol: float = 1e-12,
    density_weighting_cfg: Mapping[str, object] | None = None,
    inverse_mapping_cfg: Mapping[str, object] | None = None,
    derived_tt_prefix: str = "last",
    derived_trigger_types: Sequence[str] | str | None = None,
    derived_include_to_tt_rate_hz: bool = False,
    derived_include_trigger_type_rates: bool = False,
    derived_include_rate_histogram: bool = False,
    derived_physics_features: Sequence[str] | str | bool | None = None,
    histogram_distance_weight: float = 1.0,
    histogram_distance_blend_mode: str = "normalized",
    distance_definition: dict | None = None,
) -> pd.DataFrame:
    """Same as estimate_parameters but accepts DataFrames directly."""

    z_cols = [c for c in ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]
              if c in dict_df.columns and c in data_df.columns]

    # ── Auto-detect feature columns ──────────────────────────────────
    _feature_mode = (
        feature_columns.strip().lower()
        if isinstance(feature_columns, str) else ""
    )
    dict_feature_source = dict_df
    data_feature_source = data_df
    derived_rate_col_used: str | None = None
    derived_rate_sources: list[str] = []
    derived_physics_cols_used: list[str] = []
    if _feature_mode == "auto":
        feat_from_dict = _auto_feature_columns(dict_df, include_global_rate, global_rate_col)
        feat_from_data = _auto_feature_columns(data_df, include_global_rate, global_rate_col)
        feature_cols = sorted(set(feat_from_dict) & set(feat_from_data))
        if not feature_cols:
            raise ValueError("No common feature columns found between dictionary and dataset.")
    elif _feature_mode == "derived":
        (
            dict_feature_source,
            data_feature_source,
            derived_rate_col_used,
            derived_rate_sources,
        ) = _append_derived_tt_global_rate_column(
            dict_df=dict_df,
            data_df=data_df,
            prefix_selector=str(derived_tt_prefix),
            trigger_type_allowlist=derived_trigger_types,
            include_to_tt_rate_hz=_safe_bool(derived_include_to_tt_rate_hz, False),
            output_column=DERIVED_TT_GLOBAL_RATE_COL,
        )
        if derived_rate_col_used is None and global_rate_col in dict_df.columns and global_rate_col in data_df.columns:
            derived_rate_col_used = str(global_rate_col)
        if derived_rate_col_used is None:
            raise ValueError(
                "No derived global-rate feature available for derived mode. "
                "Could not build TT-trigger sum and fallback global-rate column is missing."
            )
        (
            dict_feature_source,
            data_feature_source,
            derived_physics_cols_used,
        ) = _append_derived_physics_feature_columns(
            dict_df=dict_feature_source,
            data_df=data_feature_source,
            rate_column=derived_rate_col_used,
            physics_features=derived_physics_features,
        )
        feat_from_dict = _derived_feature_columns(
            dict_feature_source,
            rate_column=derived_rate_col_used,
            trigger_type_rate_columns=(
                derived_rate_sources
                if _safe_bool(derived_include_trigger_type_rates, False)
                else None
            ),
            include_rate_histogram=_safe_bool(derived_include_rate_histogram, False),
            physics_feature_columns=derived_physics_cols_used,
        )
        feat_from_data = _derived_feature_columns(
            data_feature_source,
            rate_column=derived_rate_col_used,
            trigger_type_rate_columns=(
                derived_rate_sources
                if _safe_bool(derived_include_trigger_type_rates, False)
                else None
            ),
            include_rate_histogram=_safe_bool(derived_include_rate_histogram, False),
            physics_feature_columns=derived_physics_cols_used,
        )
        feature_cols = sorted(set(feat_from_dict) & set(feat_from_data))
        if not feature_cols:
            raise ValueError(
                "No derived feature columns found in both dictionary and dataset. "
                "Expected eff_empirical_1..4 and a global-rate feature."
            )
        if derived_rate_col_used == DERIVED_TT_GLOBAL_RATE_COL:
            log.info(
                "Derived feature mode: using TT-sum global rate from %d trigger-rate column(s): %s",
                len(derived_rate_sources),
                derived_rate_sources,
            )
        else:
            log.info(
                "Derived feature mode: TT-sum unavailable, using fallback rate column '%s'.",
                derived_rate_col_used,
            )
        if _safe_bool(derived_include_rate_histogram, False):
            n_hist = sum(1 for c in feature_cols if RATE_HISTOGRAM_BIN_RE.match(str(c)))
            log.info("Derived feature mode: including %d rate-histogram bin feature(s).", n_hist)
        if _safe_bool(derived_include_trigger_type_rates, False):
            n_tt = sum(1 for c in feature_cols if TT_RATE_COLUMN_RE.match(str(c)))
            log.info("Derived feature mode: including %d trigger-type rate feature(s).", n_tt)
        if derived_physics_cols_used:
            log.info(
                "Derived feature mode: including physics transform feature(s): %s",
                derived_physics_cols_used,
            )
    else:
        requested_feature_cols = parse_explicit_feature_columns(feature_columns)
        feature_cols = [
            c for c in requested_feature_cols
            if c in dict_df.columns and c in data_df.columns
        ]
        if not feature_cols:
            raise ValueError(
                "No explicit feature columns found in both dictionary and dataset. "
                f"Requested={requested_feature_cols!r}"
            )
        missing = [c for c in requested_feature_cols if c not in feature_cols]
        if missing:
            log.warning(
                "Ignoring %d explicit feature column(s) missing in dictionary/dataset intersection: %s",
                len(missing),
                missing,
            )

    # ── Auto-detect parameter columns ────────────────────────────────
    feature_col_set = set(feature_cols)
    if param_columns is None:
        param_columns = []
        for c in ["flux_cm2_min", "cos_n",
                   "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
                   "eff_empirical_1", "eff_empirical_2", "eff_empirical_3", "eff_empirical_4"]:
            if c in dict_df.columns and c not in feature_col_set:
                param_columns.append(c)
        if not param_columns:
            raise ValueError("No parameter columns found in dictionary.")
    log.info("Parameter columns to estimate: %s", param_columns)

    # ── Coerce feature columns to numeric ────────────────────────────
    dict_features = dict_feature_source[feature_cols].apply(pd.to_numeric, errors="coerce")
    data_features = data_feature_source[feature_cols].apply(pd.to_numeric, errors="coerce")
    (
        dict_features,
        data_features,
        feature_cols,
        derived_global_col,
        derived_global_sources,
    ) = _append_tt_global_sum_feature(
        dict_features=dict_features,
        data_features=data_features,
        feature_cols=feature_cols,
        include_global_rate=_safe_bool(include_global_rate, True),
    )
    if derived_global_col is not None:
        log.info(
            "Feature columns (%d): %s + derived global '%s' as sum(%d TT cols)",
            len(feature_cols),
            [c for c in feature_cols if c != derived_global_col],
            derived_global_col,
            len(derived_global_sources),
        )
    else:
        log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    hist_feature_idx = _histogram_feature_indices(feature_cols)
    hist_feature_set = set(hist_feature_idx)
    non_hist_feature_idx = [idx for idx in range(len(feature_cols)) if idx not in hist_feature_set]
    non_hist_feature_cols = [str(feature_cols[idx]) for idx in non_hist_feature_idx]

    def _is_raw_tt_rate_feature(name: str) -> bool:
        text = str(name).strip()
        if text.startswith(DERIVED_TT_GLOBAL_RATE_COL):
            return False
        return TT_RATE_COLUMN_RE.match(text) is not None

    base_eff_feature_idx = np.asarray(
        [
            j
            for j, name in enumerate(non_hist_feature_cols)
            if str(name).startswith("eff_empirical_")
        ],
        dtype=int,
    )
    base_tt_feature_idx = np.asarray(
        [j for j, name in enumerate(non_hist_feature_cols) if _is_raw_tt_rate_feature(name)],
        dtype=int,
    )
    base_other_feature_idx = np.asarray(
        [
            j
            for j in range(len(non_hist_feature_cols))
            if j not in set(base_eff_feature_idx.tolist() + base_tt_feature_idx.tolist())
        ],
        dtype=int,
    )
    inverse_cfg = resolve_inverse_mapping_cfg(
        inverse_mapping_cfg=inverse_mapping_cfg,
        interpolation_k=interpolation_k,
        histogram_distance_weight=histogram_distance_weight,
        histogram_distance_blend_mode=histogram_distance_blend_mode,
    )
    estimation_mode = str(inverse_cfg.get("estimation_mode", "single_stage"))
    neighbor_selection = str(inverse_cfg["neighbor_selection"])
    neighbor_count = inverse_cfg.get("neighbor_count")
    neighbor_weighting = str(inverse_cfg["weighting"])
    inverse_distance_power = float(inverse_cfg["inverse_distance_power"])
    softmax_temperature = float(inverse_cfg["softmax_temperature"])
    distance_floor = float(inverse_cfg["distance_floor"])
    aggregation_mode = str(inverse_cfg["aggregation"])
    stage2_eff_conditioning_weight = float(
        inverse_cfg.get("stage2_efficiency_conditioning_weight", 1.0)
    )
    stage2_eff_gate_max = float(
        inverse_cfg.get(
            "stage2_efficiency_gate_max",
            _INVERSE_MAPPING_DEFAULTS["stage2_efficiency_gate_max"],
        )
    )
    stage2_eff_gate_min_candidates = int(
        inverse_cfg.get(
            "stage2_efficiency_gate_min_candidates",
            _INVERSE_MAPPING_DEFAULTS["stage2_efficiency_gate_min_candidates"],
        )
    )
    stage2_use_rate_histogram = bool(
        inverse_cfg.get(
            "stage2_use_rate_histogram",
            _INVERSE_MAPPING_DEFAULTS["stage2_use_rate_histogram"],
        )
    )
    stage2_hist_distance_weight = float(
        inverse_cfg.get(
            "stage2_histogram_distance_weight",
            _INVERSE_MAPPING_DEFAULTS["stage2_histogram_distance_weight"],
        )
    )
    hist_distance_weight = float(inverse_cfg["histogram_distance_weight"])
    hist_distance_blend_mode = str(inverse_cfg["histogram_distance_blend_mode"])
    mask_out_of_support_eff_features = bool(
        inverse_cfg.get("mask_out_of_support_eff_features", True)
    )
    if hist_feature_idx and not non_hist_feature_idx and hist_distance_weight <= 0.0:
        # Histogram-only feature sets need non-zero distance contribution.
        hist_distance_weight = 1.0
    if hist_feature_idx:
        first_hist = feature_cols[hist_feature_idx[0]]
        last_hist = feature_cols[hist_feature_idx[-1]]
        log.info(
            "Histogram feature group detected: %d bins (%s .. %s), weight=%.3g blend=%s.",
            len(hist_feature_idx),
            first_hist,
            last_hist,
            hist_distance_weight,
            hist_distance_blend_mode,
        )

    log.info(
        "Inverse mapping: mode=%s selection=%s k=%s weighting=%s aggregation=%s",
        estimation_mode,
        neighbor_selection,
        ("all" if neighbor_count is None else str(neighbor_count)),
        neighbor_weighting,
        aggregation_mode,
    )
    if estimation_mode == "two_stage_eff_address_then_flux":
        log.info(
            "Two-stage address mode: gate_max=%.3g gate_min_candidates=%d use_rate_histogram=%s stage2_hist_weight=%.3g",
            stage2_eff_gate_max,
            stage2_eff_gate_min_candidates,
            str(stage2_use_rate_histogram).lower(),
            stage2_hist_distance_weight,
        )

    data_features, out_of_support_eff_mask_non_hist, out_of_support_eff_info = (
        _mask_out_of_support_efficiency_features(
            dict_features=dict_features,
            data_features=data_features,
            feature_cols=feature_cols,
            non_hist_feature_idx=non_hist_feature_idx,
            base_eff_feature_idx=base_eff_feature_idx,
            enabled=mask_out_of_support_eff_features,
        )
    )
    if (
        out_of_support_eff_info.get("enabled")
        and out_of_support_eff_info.get("n_eff_features_considered", 0) > 0
    ):
        n_rows_any = int(out_of_support_eff_info.get("n_rows_any_masked", 0))
        frac_rows_any = float(out_of_support_eff_info.get("fraction_rows_any_masked", 0.0))
        mean_masked = float(
            out_of_support_eff_info.get("mean_masked_eff_features_per_row", 0.0)
        )
        per_feature = out_of_support_eff_info.get("per_feature", {})
        if n_rows_any > 0:
            details = []
            if isinstance(per_feature, Mapping):
                for name, stats in per_feature.items():
                    if not isinstance(stats, Mapping):
                        continue
                    frac = float(stats.get("fraction_rows_masked", 0.0))
                    if frac <= 0.0:
                        continue
                    details.append(f"{name}:{100.0 * frac:.1f}%")
            if details:
                log.warning(
                    "Out-of-support empirical-efficiency masking applied: rows_any=%d/%d (%.1f%%), mean_masked_per_row=%.3f; per-feature=%s",
                    n_rows_any,
                    len(data_features),
                    100.0 * frac_rows_any,
                    mean_masked,
                    ", ".join(details),
                )
            else:
                log.warning(
                    "Out-of-support empirical-efficiency masking applied: rows_any=%d/%d (%.1f%%), mean_masked_per_row=%.3f.",
                    n_rows_any,
                    len(data_features),
                    100.0 * frac_rows_any,
                    mean_masked,
                )
        else:
            log.info(
                "Out-of-support empirical-efficiency masking enabled: no rows outside support."
            )

    dict_feature_raw_np = dict_features.to_numpy(dtype=float)
    data_feature_raw_np = data_features.to_numpy(dtype=float)

    # ── Distance-definition artifact (STEP 1.5) ─────────────────────
    dd_center: np.ndarray | None = None
    dd_scale: np.ndarray | None = None
    dd_weights: np.ndarray | None = None
    dd_p_norm: float | None = None
    _dd_non_hist_weights: np.ndarray | None = None  # weights restricted to non-hist features

    if distance_definition is not None and distance_definition.get("available"):
        dd_center = np.asarray(distance_definition["center"], dtype=float)
        dd_scale = np.asarray(distance_definition["scale"], dtype=float)
        dd_weights = np.asarray(distance_definition["weights"], dtype=float)
        dd_p_norm = float(distance_definition["p_norm"])
        n_active = int(np.sum(dd_weights > 0))
        mode_name = distance_definition.get("selected_mode", "unknown")
        log.info(
            "Using STEP 1.5 distance definition: %s (p=%.1f, %d/%d active features)",
            mode_name, dd_p_norm, n_active, len(feature_cols),
        )

    # ── Scaling ──────────────────────────────────────────────────────
    if dd_center is not None and dd_scale is not None:
        # Use STEP 1.5 center/scale instead of auto z-score
        safe_scale = np.where(np.abs(dd_scale) > 1e-15, dd_scale, np.nan)
        dict_scaled = ((dict_feature_raw_np - dd_center) / safe_scale)
        data_scaled = ((data_feature_raw_np - dd_center) / safe_scale)
    elif distance_metric == "l2_zscore":
        # Compute means and stds from the DICTIONARY (reference distribution)
        means = dict_features.mean(axis=0, skipna=True)
        stds = dict_features.std(axis=0, skipna=True).replace({0.0: np.nan})
        zero_var_cols = list(stds.index[stds.isna()])
        if zero_var_cols:
            log.warning(
                "z-score scaling: %d zero-variance feature(s) will be masked "
                "out of distance computation: %s",
                len(zero_var_cols),
                ", ".join(str(c) for c in zero_var_cols),
            )
        dict_scaled = ((dict_features - means) / stds).to_numpy(dtype=float)
        data_scaled = ((data_features - means) / stds).to_numpy(dtype=float)
    else:
        dict_scaled = dict_features.to_numpy(dtype=float)
        data_scaled = data_features.to_numpy(dtype=float)

    if non_hist_feature_idx:
        dict_scaled_base = dict_scaled[:, non_hist_feature_idx]
        data_scaled_base = data_scaled[:, non_hist_feature_idx]
        if dd_weights is not None:
            _dd_non_hist_weights = dd_weights[non_hist_feature_idx]
    else:
        dict_scaled_base = np.empty((len(dict_df), 0), dtype=float)
        data_scaled_base = np.empty((len(data_df), 0), dtype=float)

    if hist_feature_idx:
        dict_hist_matrix = dict_feature_raw_np[:, hist_feature_idx]
        data_hist_matrix = data_feature_raw_np[:, hist_feature_idx]
    else:
        dict_hist_matrix = None
        data_hist_matrix = None

    stage2_hist_cols: list[str] = []
    stage2_hist_matrix_dict: np.ndarray | None = None
    stage2_hist_matrix_data: np.ndarray | None = None
    if stage2_use_rate_histogram:
        stage2_hist_cols = _shared_histogram_feature_columns(dict_df.columns, data_df.columns)
        if stage2_hist_cols:
            stage2_hist_matrix_dict = (
                dict_df[stage2_hist_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
            stage2_hist_matrix_data = (
                data_df[stage2_hist_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
        elif hist_feature_idx and dict_hist_matrix is not None and data_hist_matrix is not None:
            # Fallback to selected histogram features when explicit bins are not available.
            stage2_hist_matrix_dict = dict_hist_matrix
            stage2_hist_matrix_data = data_hist_matrix
    if estimation_mode == "two_stage_eff_address_then_flux":
        log.info(
            "Two-stage address mode histogram bins: %d",
            int(len(stage2_hist_cols))
            if stage2_hist_cols
            else (int(dict_hist_matrix.shape[1]) if dict_hist_matrix is not None else 0),
        )
        if stage2_hist_matrix_dict is None:
            log.warning(
                "Two-stage address mode: no rate-histogram bins resolved; stage 2 will use non-hist rate features only."
            )

    # ── z-position arrays for fast matching ──────────────────────────
    dict_z = dict_df[z_cols].to_numpy(dtype=float) if z_cols else None
    data_z = data_df[z_cols].to_numpy(dtype=float) if z_cols else None

    # ── File IDs for self-exclusion ──────────────────────────────────
    join_col = None
    for candidate in ("filename_base", "file_name"):
        if candidate in dict_df.columns and candidate in data_df.columns:
            join_col = candidate
            break
    dict_ids = dict_df[join_col].astype(str).tolist() if join_col else None
    data_ids = data_df[join_col].astype(str).tolist() if join_col else None
    dict_ids_arr = np.asarray(dict_ids, dtype=object) if dict_ids else None

    # ── Dict parameter values as arrays ──────────────────────────────
    dict_param_arrays = {}
    for pc in param_columns:
        dict_param_arrays[pc] = pd.to_numeric(
            dict_df[pc], errors="coerce"
        ).to_numpy(dtype=float)

    # ── Optional strict shared-parameter exclusion ──────────────────
    shared_param_cols: list[str] = []
    shared_param_ignore = set(_normalize_column_spec(shared_parameter_exclusion_ignore))
    shared_exclusion_mode = _resolve_shared_parameter_exclusion_mode(
        shared_parameter_exclusion_mode,
        legacy_flag=bool(exclude_candidates_sharing_parameter_values),
    )
    shared_exclusion_enabled = shared_exclusion_mode in {"partial", "full"}
    if shared_exclusion_enabled:
        raw_cols = _normalize_column_spec(shared_parameter_exclusion_columns)
        if not raw_cols or any(str(c).strip().lower() == "auto" for c in raw_cols):
            raw_cols = [
                str(c)
                for c in param_columns
                if not str(c).startswith("eff_empirical_")
            ]

        seen_cols: set[str] = set()
        for col in raw_cols:
            name = str(col).strip()
            if not name or name in seen_cols or name in shared_param_ignore:
                continue
            if name not in dict_df.columns or name not in data_df.columns:
                continue
            shared_param_cols.append(name)
            seen_cols.add(name)

        if shared_param_cols:
            log.info(
                "Shared-parameter exclusion enabled: mode=%s columns=%s (ignore=%s, atol=%.3g)",
                shared_exclusion_mode,
                shared_param_cols,
                sorted(shared_param_ignore),
                float(shared_parameter_match_atol),
            )
        else:
            log.warning(
                "Shared-parameter exclusion requested but no valid columns resolved; exclusion disabled."
            )
    dict_shared_param_arrays = {
        col: pd.to_numeric(dict_df[col], errors="coerce").to_numpy(dtype=float)
        for col in shared_param_cols
    }
    data_shared_param_arrays = {
        col: pd.to_numeric(data_df[col], errors="coerce").to_numpy(dtype=float)
        for col in shared_param_cols
    }
    param_set_guard_enabled = (
        shared_exclusion_enabled
        and
        "param_set_id" in dict_df.columns
        and "param_set_id" in data_df.columns
        and "param_set_id" not in shared_param_ignore
    )
    dict_param_set_array = (
        pd.to_numeric(dict_df["param_set_id"], errors="coerce").to_numpy(dtype=float)
        if param_set_guard_enabled else None
    )
    data_param_set_array = (
        pd.to_numeric(data_df["param_set_id"], errors="coerce").to_numpy(dtype=float)
        if param_set_guard_enabled else None
    )
    if param_set_guard_enabled:
        log.info("Shared-parameter exclusion guard enabled for 'param_set_id'.")

    density_scaling_all, density_info = _build_density_scaling_for_dictionary(
        dict_df=dict_df,
        dict_feature_matrix=dict_scaled,
        distance_metric=distance_metric,
        dict_z=dict_z,
        density_weighting_cfg=density_weighting_cfg,
    )
    if density_scaling_all is not None:
        density_space = str(density_info.get("space", "?"))
        log.info(
            "Density correction enabled: space=%s k=%d exp=%.3g clip=[%.3g, %.3g]",
            density_space,
            int(
                density_info.get(
                    "effective_k_neighbors",
                    density_info.get("effective_k_neighbors_max", density_info.get("k_neighbors", 0)),
                )
            ),
            float(density_info.get("exponent", np.nan)),
            float(density_info.get("clip_min", np.nan)),
            float(density_info.get("clip_max", np.nan)),
        )
        if density_space == "feature":
            log.info(
                "Density feature-space details: metric=%s z_groups=%s finite_radius=%s",
                density_info.get("distance_metric", distance_metric),
                density_info.get("n_z_groups", "?"),
                density_info.get("n_finite_radius", "?"),
            )
        else:
            log.info(
                "Density details: n_basis=%s finite_radius=%s",
                density_info.get("n_basis", "?"),
                density_info.get("n_finite_radius", "?"),
            )
    elif isinstance(density_weighting_cfg, Mapping):
        log.info("Density correction disabled for this run (%s).", density_info.get("reason", "disabled"))

    eff_sim_param_columns = [pc for pc in param_columns if str(pc).startswith("eff_sim_")]
    flux_param_columns = [pc for pc in param_columns if "flux" in str(pc).lower()]
    param_span_by_name: dict[str, float] = {}
    for pc in param_columns:
        arr = np.asarray(dict_param_arrays[pc], dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            param_span_by_name[pc] = 1.0
            continue
        span = float(np.max(finite) - np.min(finite))
        param_span_by_name[pc] = span if np.isfinite(span) and span > 1e-12 else 1.0

    stage2_rate_feature_idx = np.asarray(
        [
            j
            for j, name in enumerate(non_hist_feature_cols)
            if (
                ("rate" in str(name).lower())
                and (not str(name).startswith("eff_empirical_"))
            )
        ],
        dtype=int,
    )
    if stage2_rate_feature_idx.size == 0:
        stage2_rate_feature_idx = np.asarray(base_other_feature_idx, dtype=int)
    is_two_stage_soft_mode = estimation_mode == "two_stage_eff_then_flux"
    is_two_stage_address_mode = estimation_mode == "two_stage_eff_address_then_flux"
    is_two_stage_mode = is_two_stage_soft_mode or is_two_stage_address_mode

    def _resolve_k_use(n_available: int) -> int:
        if n_available <= 0:
            return 0
        if neighbor_selection == "nearest":
            return 1
        if neighbor_selection == "all":
            return int(n_available)
        k_eff = int(neighbor_count) if neighbor_count is not None else int(n_available)
        return min(max(1, k_eff), int(n_available))

    def _select_top_neighbors(
        valid_indices_local: np.ndarray,
        valid_distances_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(valid_indices_local) == 0:
            return (
                np.asarray([], dtype=int),
                np.asarray([], dtype=float),
            )
        order_local = np.argsort(valid_distances_local)
        sorted_idx = np.asarray(valid_indices_local[order_local], dtype=int)
        sorted_dist = np.asarray(valid_distances_local[order_local], dtype=float)
        k_use_local = _resolve_k_use(len(sorted_idx))
        return sorted_idx[:k_use_local], sorted_dist[:k_use_local]

    def _build_weights_for_top(
        top_indices_local: np.ndarray,
        top_distances_local: np.ndarray,
    ) -> np.ndarray:
        if len(top_indices_local) == 0:
            return np.asarray([], dtype=float)
        weights_local = _build_neighbor_weights(
            top_distances_local,
            weighting_mode=neighbor_weighting,
            idw_power=inverse_distance_power,
            softmax_temperature=softmax_temperature,
            distance_floor=distance_floor,
        )
        if density_scaling_all is not None:
            density_scaling_top = np.asarray(
                density_scaling_all[top_indices_local], dtype=float
            )
            density_scaling_top = np.where(
                np.isfinite(density_scaling_top), density_scaling_top, 0.0
            )
            weights_local = weights_local * density_scaling_top
        weights_local = np.where(np.isfinite(weights_local), weights_local, 0.0)
        sum_weights = float(np.sum(weights_local))
        if sum_weights > 0.0:
            weights_local /= sum_weights
        return weights_local

    def _aggregate_param_from_top(
        *,
        param_name: str,
        top_indices_local: np.ndarray,
        weights_local: np.ndarray,
        sample_features_local: np.ndarray | None,
        neighbor_features_local: np.ndarray | None,
    ) -> float:
        vals_local = np.asarray(dict_param_arrays[param_name][top_indices_local], dtype=float)
        finite_local = np.isfinite(vals_local)
        if np.sum(finite_local) == 0:
            return np.nan
        w_local = np.asarray(weights_local, dtype=float).copy()
        w_local[~finite_local] = 0.0
        w_sum_local = float(np.sum(w_local))
        if w_sum_local <= 0.0:
            return np.nan
        w_local /= w_sum_local
        if aggregation_mode == "weighted_median":
            return float(_weighted_median(vals_local, w_local))
        if aggregation_mode == "local_linear":
            return float(
                _local_linear_estimate(
                    vals_local,
                    w_local,
                    sample_features=sample_features_local,
                    neighbor_features=neighbor_features_local,
                    parameter_name=param_name,
                )
            )
        return float(np.sum(w_local * np.nan_to_num(vals_local)))

    def _compute_candidate_distances_subset(
        *,
        row_idx: int,
        cand_indices_local: np.ndarray,
        base_feature_idx_local: np.ndarray,
        include_hist: bool,
    ) -> np.ndarray:
        if len(cand_indices_local) == 0:
            return np.asarray([], dtype=float)
        sample_scaled_non_hist_local = (
            data_scaled_base[row_idx, base_feature_idx_local]
            if base_feature_idx_local.size > 0
            else None
        )
        candidates_scaled_non_hist_local = (
            dict_scaled_base[cand_indices_local][:, base_feature_idx_local]
            if base_feature_idx_local.size > 0
            else None
        )
        sample_hist_local = (
            data_hist_matrix[row_idx]
            if (
                include_hist
                and hist_feature_idx
                and data_hist_matrix is not None
            )
            else None
        )
        candidates_hist_local = (
            dict_hist_matrix[cand_indices_local]
            if (
                include_hist
                and hist_feature_idx
                and dict_hist_matrix is not None
            )
            else None
        )
        # Subset dd weights when using STEP 1.5 artifact
        local_dd_weights = None
        local_dd_p_norm = dd_p_norm
        if _dd_non_hist_weights is not None and base_feature_idx_local.size > 0:
            local_dd_weights = _dd_non_hist_weights[base_feature_idx_local]
        return compute_candidate_distances(
            distance_metric=distance_metric,
            sample_scaled_non_hist=sample_scaled_non_hist_local,
            candidates_scaled_non_hist=candidates_scaled_non_hist_local,
            sample_hist_raw=sample_hist_local,
            candidates_hist_raw=candidates_hist_local,
            histogram_distance_weight=hist_distance_weight,
            histogram_distance_blend_mode=hist_distance_blend_mode,
            min_valid_non_hist_dims=1,
            dd_weights=local_dd_weights,
            dd_p_norm=local_dd_p_norm,
        )

    def _compute_stage2_candidate_distances(
        *,
        row_idx: int,
        cand_indices_local: np.ndarray,
    ) -> np.ndarray:
        sample_scaled_non_hist_local = (
            data_scaled_base[row_idx, stage2_rate_feature_idx]
            if stage2_rate_feature_idx.size > 0
            else None
        )
        candidates_scaled_non_hist_local = (
            dict_scaled_base[cand_indices_local][:, stage2_rate_feature_idx]
            if stage2_rate_feature_idx.size > 0
            else None
        )

        sample_hist_local = None
        candidates_hist_local = None
        hist_weight_local = float(hist_distance_weight)
        if (
            stage2_use_rate_histogram
            and stage2_hist_matrix_dict is not None
            and stage2_hist_matrix_data is not None
        ):
            sample_hist_local = stage2_hist_matrix_data[row_idx]
            candidates_hist_local = stage2_hist_matrix_dict[cand_indices_local]
            hist_weight_local = float(stage2_hist_distance_weight)
        elif dict_hist_matrix is not None and data_hist_matrix is not None:
            sample_hist_local = data_hist_matrix[row_idx]
            candidates_hist_local = dict_hist_matrix[cand_indices_local]
            hist_weight_local = float(stage2_hist_distance_weight)

        if stage2_rate_feature_idx.size == 0 and sample_hist_local is None:
            return np.full(len(cand_indices_local), np.nan, dtype=float)

        # Stage 2 uses rate-feature weights from STEP 1.5 if available
        local_dd_weights_s2 = None
        local_dd_p_norm_s2 = dd_p_norm
        if _dd_non_hist_weights is not None and stage2_rate_feature_idx.size > 0:
            local_dd_weights_s2 = _dd_non_hist_weights[stage2_rate_feature_idx]
        return compute_candidate_distances(
            distance_metric=distance_metric,
            sample_scaled_non_hist=sample_scaled_non_hist_local,
            candidates_scaled_non_hist=candidates_scaled_non_hist_local,
            sample_hist_raw=sample_hist_local,
            candidates_hist_raw=candidates_hist_local,
            histogram_distance_weight=max(hist_weight_local, 0.0),
            histogram_distance_blend_mode=hist_distance_blend_mode,
            min_valid_non_hist_dims=1,
            dd_weights=local_dd_weights_s2,
            dd_p_norm=local_dd_p_norm_s2,
        )

    # ── Estimate for each dataset entry ──────────────────────────────
    results = []
    n_data = len(data_df)
    shared_excl_rows_affected = 0
    shared_excl_removed_total = 0
    shared_excl_rows_empty = 0
    shared_match_atol = max(float(shared_parameter_match_atol), 0.0)

    for i in range(n_data):
        row_result: dict = {"dataset_index": i}
        if join_col and data_ids:
            row_result["filename_base"] = data_ids[i]
        row_result["estimation_mode_used"] = "single_stage"
        row_result["stage1_best_distance_eff"] = np.nan
        row_result["stage2_best_distance_rate"] = np.nan
        row_result["stage2_efficiency_conditioning_penalty"] = np.nan
        row_result["n_neighbors_stage1"] = 0
        row_result["n_neighbors_stage2"] = 0
        row_result["stage2_candidates_before_gate"] = 0
        row_result["stage2_candidates_after_gate"] = 0
        row_result["stage2_efficiency_gate_threshold"] = np.nan
        row_result["stage2_efficiency_gate_fallback"] = False
        row_result["stage2_histogram_bins_used"] = int(
            stage2_hist_matrix_dict.shape[1]
            if stage2_hist_matrix_dict is not None
            else (dict_hist_matrix.shape[1] if dict_hist_matrix is not None else 0)
        )
        n_eff_masked = 0
        if out_of_support_eff_mask_non_hist is not None and i < out_of_support_eff_mask_non_hist.shape[0]:
            n_eff_masked = int(np.sum(out_of_support_eff_mask_non_hist[i]))
        row_result["n_eff_features_masked_out_of_support"] = n_eff_masked
        row_result["any_eff_feature_masked_out_of_support"] = bool(n_eff_masked > 0)

        # Find z-compatible candidates
        if dict_z is not None and data_z is not None:
            z_mask = np.all(
                np.abs(dict_z - data_z[i]) < z_tol, axis=1
            )
        else:
            z_mask = np.ones(len(dict_df), dtype=bool)

        # Exclude same file
        if exclude_same_file and join_col and dict_ids_arr is not None and data_ids:
            same_file = dict_ids_arr == data_ids[i]
            z_mask &= ~same_file

        # Exclude dictionary candidates sharing any selected physical parameter value.
        n_before_shared_excl = int(z_mask.sum())
        n_removed_shared_excl = 0
        if (shared_param_cols or param_set_guard_enabled) and n_before_shared_excl > 0:
            remove_any = np.zeros(len(dict_df), dtype=bool)
            remove_full = np.ones(len(dict_df), dtype=bool)
            n_compared_cols = 0
            for col in shared_param_cols:
                sample_val = data_shared_param_arrays[col][i]
                if not np.isfinite(sample_val):
                    continue
                n_compared_cols += 1
                dict_vals = dict_shared_param_arrays[col]
                atol_eff = _effective_atol(shared_match_atol, float(sample_val))
                same_mask = np.isfinite(dict_vals) & np.isclose(
                    dict_vals,
                    float(sample_val),
                    rtol=0.0,
                    atol=atol_eff,
                )
                if np.any(same_mask):
                    remove_any |= same_mask
                remove_full &= same_mask

            if shared_exclusion_mode == "full":
                remove_basis = remove_full if n_compared_cols > 0 else np.zeros(len(dict_df), dtype=bool)
            else:
                remove_basis = remove_any

            if param_set_guard_enabled and dict_param_set_array is not None and data_param_set_array is not None:
                sample_param_set = data_param_set_array[i]
                if np.isfinite(sample_param_set):
                    atol_eff = _effective_atol(shared_match_atol, float(sample_param_set))
                    same_param_set_mask = np.isfinite(dict_param_set_array) & np.isclose(
                        dict_param_set_array,
                        float(sample_param_set),
                        rtol=0.0,
                        atol=atol_eff,
                    )
                    if np.any(same_param_set_mask):
                        remove_basis |= same_param_set_mask

            if np.any(remove_basis):
                remove_in_candidates = z_mask & remove_basis
                n_removed_shared_excl = int(np.sum(remove_in_candidates))
                if n_removed_shared_excl > 0:
                    z_mask[remove_in_candidates] = False
                    shared_excl_rows_affected += 1
                    shared_excl_removed_total += n_removed_shared_excl
                    if not np.any(z_mask):
                        shared_excl_rows_empty += 1

        if shared_param_cols or param_set_guard_enabled:
            row_result["n_candidates_before_shared_param_exclusion"] = n_before_shared_excl
            row_result["n_candidates_removed_shared_param_exclusion"] = n_removed_shared_excl

        cand_indices = np.where(z_mask)[0]
        row_result["n_candidates"] = len(cand_indices)

        if len(cand_indices) == 0:
            # No candidates — set all estimates to NaN
            for pc in param_columns:
                row_result[f"est_{pc}"] = np.nan
            row_result["best_distance"] = np.nan
            results.append(row_result)
            continue

        # Compute feature-space distances (base features + optional histogram shape).
        sample_scaled_non_hist = data_scaled_base[i] if non_hist_feature_idx else None
        cand_scaled_non_hist = dict_scaled_base[cand_indices] if non_hist_feature_idx else None
        sample_hist_raw = (
            data_hist_matrix[i] if (hist_feature_idx and data_hist_matrix is not None) else None
        )
        cand_hist_raw = (
            dict_hist_matrix[cand_indices] if (hist_feature_idx and dict_hist_matrix is not None) else None
        )
        distances = compute_candidate_distances(
            distance_metric=distance_metric,
            sample_scaled_non_hist=sample_scaled_non_hist,
            candidates_scaled_non_hist=cand_scaled_non_hist,
            sample_hist_raw=sample_hist_raw,
            candidates_hist_raw=cand_hist_raw,
            histogram_distance_weight=hist_distance_weight,
            histogram_distance_blend_mode=hist_distance_blend_mode,
        )

        # Handle NaN distances
        valid = np.isfinite(distances)
        if valid.sum() == 0:
            for pc in param_columns:
                row_result[f"est_{pc}"] = np.nan
            row_result["best_distance"] = np.nan
            results.append(row_result)
            continue

        valid_indices = cand_indices[valid]
        valid_distances = distances[valid]

        # Sort by distance (ascending = best first)
        order = np.argsort(valid_distances)
        valid_indices = valid_indices[order]
        valid_distances = valid_distances[order]

        best_j = int(valid_indices[0])
        row_result["best_distance"] = float(valid_distances[0])

        # Feature-group diagnostics at the best dictionary match.
        row_result["best_distance_base_l2"] = np.nan
        row_result["best_distance_base_l2_eff_empirical"] = np.nan
        row_result["best_distance_base_l2_tt_rates"] = np.nan
        row_result["best_distance_base_l2_other"] = np.nan
        row_result["best_distance_base_share_eff_empirical"] = np.nan
        row_result["best_distance_base_share_tt_rates"] = np.nan
        row_result["best_distance_base_share_other"] = np.nan
        row_result["best_distance_non_hist_finite_dims"] = 0
        row_result["best_distance_hist_emd"] = np.nan

        if non_hist_feature_idx:
            sample_base_vec = np.asarray(data_scaled_base[i], dtype=float)
            best_base_vec = np.asarray(dict_scaled_base[best_j], dtype=float)
            finite_base = np.isfinite(sample_base_vec) & np.isfinite(best_base_vec)
            row_result["best_distance_non_hist_finite_dims"] = int(np.sum(finite_base))
            if np.any(finite_base):
                delta = np.zeros_like(sample_base_vec, dtype=float)
                delta[finite_base] = sample_base_vec[finite_base] - best_base_vec[finite_base]

                def _group_l2(group_idx: np.ndarray) -> float:
                    if group_idx.size == 0:
                        return 0.0
                    use = group_idx[finite_base[group_idx]]
                    if use.size == 0:
                        return np.nan
                    vals = delta[use]
                    return float(np.sqrt(np.sum(vals * vals)))

                base_l2 = float(np.sqrt(np.sum(delta[finite_base] * delta[finite_base])))
                eff_l2 = _group_l2(base_eff_feature_idx)
                tt_l2 = _group_l2(base_tt_feature_idx)
                other_l2 = _group_l2(base_other_feature_idx)

                row_result["best_distance_base_l2"] = base_l2
                row_result["best_distance_base_l2_eff_empirical"] = eff_l2
                row_result["best_distance_base_l2_tt_rates"] = tt_l2
                row_result["best_distance_base_l2_other"] = other_l2

                denom = base_l2 * base_l2
                if np.isfinite(denom) and denom > 0.0:
                    row_result["best_distance_base_share_eff_empirical"] = (
                        float(eff_l2 * eff_l2 / denom) if np.isfinite(eff_l2) else np.nan
                    )
                    row_result["best_distance_base_share_tt_rates"] = (
                        float(tt_l2 * tt_l2 / denom) if np.isfinite(tt_l2) else np.nan
                    )
                    row_result["best_distance_base_share_other"] = (
                        float(other_l2 * other_l2 / denom) if np.isfinite(other_l2) else np.nan
                    )

        if hist_feature_idx and data_hist_matrix is not None and dict_hist_matrix is not None:
            hist_pair = _histogram_emd_many(
                np.asarray(data_hist_matrix[i], dtype=float),
                np.asarray(dict_hist_matrix[best_j : best_j + 1], dtype=float),
            )
            if hist_pair.size:
                row_result["best_distance_hist_emd"] = (
                    float(hist_pair[0]) if np.isfinite(hist_pair[0]) else np.nan
                )

        # Select neighborhood according to inverse-mapping strategy.
        top_indices, top_distances = _select_top_neighbors(valid_indices, valid_distances)
        row_result["n_neighbors_used"] = int(len(top_indices))

        if len(top_indices) == 0:
            for pc in param_columns:
                row_result[f"est_{pc}"] = np.nan
        elif len(top_indices) == 1 or top_distances[0] <= distance_floor:
            # Nearest-only (or exact match)
            best_j = top_indices[0]
            for pc in param_columns:
                row_result[f"est_{pc}"] = float(dict_param_arrays[pc][best_j])
        else:
            weights = _build_weights_for_top(top_indices, top_distances)
            sum_weights = float(np.sum(weights))
            if sum_weights <= 0.0:
                best_j = top_indices[0]
                for pc in param_columns:
                    row_result[f"est_{pc}"] = float(dict_param_arrays[pc][best_j])
                results.append(row_result)
                if (i + 1) % 200 == 0 or i == n_data - 1:
                    log.info("  Estimated %d / %d", i + 1, n_data)
                continue
            sample_base_for_local = data_scaled_base[i] if non_hist_feature_idx else None
            top_base_for_local = (
                dict_scaled_base[top_indices] if non_hist_feature_idx else None
            )

            for pc in param_columns:
                row_result[f"est_{pc}"] = _aggregate_param_from_top(
                    param_name=pc,
                    top_indices_local=top_indices,
                    weights_local=weights,
                    sample_features_local=sample_base_for_local,
                    neighbor_features_local=top_base_for_local,
                )

        if (
            is_two_stage_mode
            and len(eff_sim_param_columns) > 0
            and len(flux_param_columns) > 0
            and base_eff_feature_idx.size > 0
        ):
            stage1_eff_estimates: dict[str, float] = {}
            stage1_distances = _compute_candidate_distances_subset(
                row_idx=i,
                cand_indices_local=cand_indices,
                base_feature_idx_local=np.asarray(base_eff_feature_idx, dtype=int),
                include_hist=False,
            )
            valid_stage1 = np.isfinite(stage1_distances)
            if np.any(valid_stage1):
                stage1_indices = cand_indices[valid_stage1]
                stage1_dist = stage1_distances[valid_stage1]
                top1_indices, top1_distances = _select_top_neighbors(stage1_indices, stage1_dist)
                if len(top1_indices) > 0:
                    row_result["n_neighbors_stage1"] = int(len(top1_indices))
                    row_result["stage1_best_distance_eff"] = float(np.min(top1_distances))
                    if len(top1_indices) == 1 or top1_distances[0] <= distance_floor:
                        idx1 = int(top1_indices[0])
                        for pc in eff_sim_param_columns:
                            est_val = float(dict_param_arrays[pc][idx1])
                            row_result[f"est_{pc}"] = est_val
                            stage1_eff_estimates[pc] = est_val
                    else:
                        w1 = _build_weights_for_top(top1_indices, top1_distances)
                        if float(np.sum(w1)) > 0.0:
                            sample_eff_local = data_scaled_base[i, base_eff_feature_idx]
                            top_eff_local = dict_scaled_base[top1_indices][:, base_eff_feature_idx]
                            for pc in eff_sim_param_columns:
                                est_val = _aggregate_param_from_top(
                                    param_name=pc,
                                    top_indices_local=top1_indices,
                                    weights_local=w1,
                                    sample_features_local=sample_eff_local,
                                    neighbor_features_local=top_eff_local,
                                )
                                row_result[f"est_{pc}"] = est_val
                                if np.isfinite(est_val):
                                    stage1_eff_estimates[pc] = float(est_val)

            if stage1_eff_estimates:
                stage2_distances = _compute_stage2_candidate_distances(
                    row_idx=i,
                    cand_indices_local=cand_indices,
                )
                valid_stage2 = np.isfinite(stage2_distances)
                if np.any(valid_stage2):
                    stage2_indices = cand_indices[valid_stage2]
                    stage2_dist = stage2_distances[valid_stage2]
                    row_result["stage2_candidates_before_gate"] = int(len(stage2_indices))
                    stage2_penalty = np.zeros(len(stage2_indices), dtype=float)
                    n_eff_used = 0
                    for pc, est_val in stage1_eff_estimates.items():
                        if not np.isfinite(est_val):
                            continue
                        span = float(param_span_by_name.get(pc, 1.0))
                        span = span if np.isfinite(span) and span > 1e-12 else 1.0
                        cand_vals = np.asarray(dict_param_arrays[pc][stage2_indices], dtype=float)
                        finite_eff = np.isfinite(cand_vals)
                        if not np.any(finite_eff):
                            continue
                        comp = np.zeros(len(stage2_indices), dtype=float)
                        comp[finite_eff] = ((cand_vals[finite_eff] - float(est_val)) / span) ** 2
                        stage2_penalty += comp
                        n_eff_used += 1
                    if n_eff_used > 0:
                        stage2_penalty = np.sqrt(stage2_penalty / float(n_eff_used))
                    else:
                        stage2_penalty[:] = 0.0
                    row_result["stage2_efficiency_gate_threshold"] = (
                        float(stage2_eff_gate_max) if is_two_stage_address_mode else np.nan
                    )

                    if is_two_stage_address_mode and len(stage2_indices) > 0 and n_eff_used > 0:
                        gate_mask = np.isfinite(stage2_penalty) & (
                            stage2_penalty <= float(stage2_eff_gate_max)
                        )
                        min_keep = min(
                            max(1, int(stage2_eff_gate_min_candidates)),
                            int(len(stage2_indices)),
                        )
                        if int(np.sum(gate_mask)) < min_keep:
                            finite_pen = np.isfinite(stage2_penalty)
                            if np.any(finite_pen):
                                finite_pos = np.where(finite_pen)[0]
                                order_gate = np.argsort(stage2_penalty[finite_pos])
                                keep_pos = finite_pos[order_gate][:min_keep]
                            else:
                                keep_pos = np.arange(min_keep, dtype=int)
                            gate_mask = np.zeros(len(stage2_indices), dtype=bool)
                            gate_mask[keep_pos] = True
                            row_result["stage2_efficiency_gate_fallback"] = True
                        stage2_indices = stage2_indices[gate_mask]
                        stage2_dist = stage2_dist[gate_mask]
                        stage2_penalty = stage2_penalty[gate_mask]

                    row_result["stage2_candidates_after_gate"] = int(len(stage2_indices))
                    if len(stage2_indices) > 0:
                        stage2_dist_combined = stage2_dist + float(stage2_eff_conditioning_weight) * stage2_penalty
                        order2 = np.argsort(stage2_dist_combined)
                        stage2_indices = stage2_indices[order2]
                        stage2_dist_combined = stage2_dist_combined[order2]
                        stage2_penalty = stage2_penalty[order2]
                        top2_indices, top2_distances = _select_top_neighbors(stage2_indices, stage2_dist_combined)
                        if len(top2_indices) > 0:
                            row_result["estimation_mode_used"] = (
                                "two_stage_eff_address_then_flux"
                                if is_two_stage_address_mode
                                else "two_stage_eff_then_flux"
                            )
                            row_result["n_neighbors_stage2"] = int(len(top2_indices))
                            row_result["stage2_best_distance_rate"] = float(np.min(top2_distances))
                            row_result["stage2_efficiency_conditioning_penalty"] = float(stage2_penalty[0])
                            if len(top2_indices) == 1 or top2_distances[0] <= distance_floor:
                                idx2 = int(top2_indices[0])
                                for pc in flux_param_columns:
                                    row_result[f"est_{pc}"] = float(dict_param_arrays[pc][idx2])
                            else:
                                w2 = _build_weights_for_top(top2_indices, top2_distances)
                                if float(np.sum(w2)) > 0.0:
                                    sample_rate_local = (
                                        data_scaled_base[i, stage2_rate_feature_idx]
                                        if stage2_rate_feature_idx.size > 0
                                        else None
                                    )
                                    top_rate_local = (
                                        dict_scaled_base[top2_indices][:, stage2_rate_feature_idx]
                                        if stage2_rate_feature_idx.size > 0
                                        else None
                                    )
                                    for pc in flux_param_columns:
                                        row_result[f"est_{pc}"] = _aggregate_param_from_top(
                                            param_name=pc,
                                            top_indices_local=top2_indices,
                                            weights_local=w2,
                                            sample_features_local=sample_rate_local,
                                            neighbor_features_local=top_rate_local,
                                        )

        results.append(row_result)

        if (i + 1) % 200 == 0 or i == n_data - 1:
            log.info("  Estimated %d / %d", i + 1, n_data)

    if shared_param_cols or param_set_guard_enabled:
        log.info(
            "Shared-parameter exclusion summary: mode=%s rows_affected=%d/%d, removed_candidates_total=%d, rows_with_zero_candidates_after_exclusion=%d",
            shared_exclusion_mode,
            shared_excl_rows_affected,
            n_data,
            shared_excl_removed_total,
            shared_excl_rows_empty,
        )

    result_df = pd.DataFrame(results)
    result_df.attrs["efficiency_feature_out_of_support_masking"] = out_of_support_eff_info

    if "best_distance_base_share_tt_rates" in result_df.columns:
        tt_vals = pd.to_numeric(
            result_df["best_distance_base_share_tt_rates"], errors="coerce"
        ).to_numpy(dtype=float)
        eff_vals = pd.to_numeric(
            result_df["best_distance_base_share_eff_empirical"], errors="coerce"
        ).to_numpy(dtype=float)
        other_vals = pd.to_numeric(
            result_df["best_distance_base_share_other"], errors="coerce"
        ).to_numpy(dtype=float)
        finite_any = np.isfinite(tt_vals) | np.isfinite(eff_vals) | np.isfinite(other_vals)
        if np.any(finite_any):
            tt_med = float(np.nanmedian(tt_vals))
            eff_med = float(np.nanmedian(eff_vals))
            other_med = float(np.nanmedian(other_vals))
            log.info(
                "Best-match non-hist distance share medians: eff=%.3f tt=%.3f other=%.3f",
                eff_med,
                tt_med,
                other_med,
            )
            if tt_med > 0.60:
                log.warning(
                    "TT-rate features dominate non-hist best-match distance (median share=%.3f). "
                    "This can re-couple flux and efficiency in inverse mapping.",
                    tt_med,
                )

    return result_df
