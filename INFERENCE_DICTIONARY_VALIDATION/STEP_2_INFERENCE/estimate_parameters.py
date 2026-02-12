#!/usr/bin/env python3
"""Self-contained dictionary-based parameter estimation module.

This module provides a single primary function:

    estimate_parameters(dictionary_path, dataset_path, **kwargs)
        → DataFrame with estimated parameters for every row in the dataset.

It is designed to be completely self-contained so it can be imported and
used with ANY dictionary and ANY dataset (simulated or real) for the
inverse-problem solution: given observed rate fingerprints, find the
physical parameters (flux, cos_n, efficiencies) that best match.

The algorithm:
1. Load dictionary and dataset CSVs.
2. For each point in the dataset, restrict candidate dictionary entries
   to those sharing the same z-plane geometry.
3. Build feature vectors from selected columns (e.g. raw_tt_*_rate_hz).
4. Score each candidate using a chosen distance metric on scaled features.
5. Estimate physical parameters via IDW interpolation over the K nearest
   candidates.

Distance metrics
----------------
- ``l2_zscore``  — Euclidean distance on z-score-scaled features (default).
- ``l2_raw``     — Euclidean distance on raw feature values.
- ``chi2``       — Chi-squared-like distance (weighted by observed value).
- ``poisson``    — Poisson deviance–like distance.

Usage example
-------------
    from estimate_parameters import estimate_parameters

    results = estimate_parameters(
        dictionary_path="dictionary.csv",
        dataset_path="dataset.csv",
        feature_columns="auto",       # or explicit list
        distance_metric="l2_zscore",
        interpolation_k=5,
    )
    results.to_csv("estimated_params.csv", index=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("estimate_parameters")


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
    sigma2 = np.maximum(np.abs(y_t), 1.0)
    return float(np.sum((y_t - y_p) ** 2 / sigma2))


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


DISTANCE_FNS = {
    "l2": _l2,
    "l2_zscore": _l2,
    "l2_raw": _l2,
    "chi2": _chi2,
    "poisson": _poisson,
}


# =====================================================================
# Feature selection helpers
# =====================================================================

def _auto_feature_columns(
    df: pd.DataFrame,
    include_global_rate: bool = True,
    global_rate_col: str = "events_per_second_global_rate",
) -> list[str]:
    """Automatically detect rate columns to build the feature vector."""
    cols = sorted([
        c for c in df.columns
        if c.startswith("raw_tt_") and c.endswith("_rate_hz")
    ])
    if not cols:
        # Fallback: try clean_tt_ or any _rate_hz
        cols = sorted([c for c in df.columns if c.endswith("_rate_hz")])
    if include_global_rate and global_rate_col in df.columns:
        if global_rate_col not in cols:
            cols.append(global_rate_col)
    return cols


# =====================================================================
# Main estimation function
# =====================================================================

def estimate_parameters(
    dictionary_path: str | Path,
    dataset_path: str | Path,
    *,
    feature_columns: str | list[str] = "auto",
    distance_metric: str = "l2_zscore",
    interpolation_k: int = 5,
    z_tol: float = 1e-6,
    include_global_rate: bool = True,
    global_rate_col: str = "events_per_second_global_rate",
    param_columns: list[str] | None = None,
    exclude_same_file: bool = True,
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
        raw_tt_*_rate_hz columns.
    distance_metric : str
        One of "l2_zscore", "l2_raw", "chi2", "poisson".
    interpolation_k : int
        Number of nearest neighbouring dictionary entries for IDW
        interpolation (1 = nearest-only).
    z_tol : float
        Tolerance for z-plane position matching.
    include_global_rate : bool
        Whether to include global rate in auto feature selection.
    global_rate_col : str
        Name of the global rate column.
    param_columns : list of str or None
        Parameter columns to estimate. If None, auto-detected from
        dictionary (flux_cm2_min, cos_n, eff_sim_1..4).
    exclude_same_file : bool
        If True, exclude dictionary entries with the same filename_base
        as the dataset row being estimated.

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
    )


def estimate_from_dataframes(
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    *,
    feature_columns: str | list[str] = "auto",
    distance_metric: str = "l2_zscore",
    interpolation_k: int = 5,
    z_tol: float = 1e-6,
    include_global_rate: bool = True,
    global_rate_col: str = "events_per_second_global_rate",
    param_columns: list[str] | None = None,
    exclude_same_file: bool = True,
) -> pd.DataFrame:
    """Same as estimate_parameters but accepts DataFrames directly."""

    z_cols = [c for c in ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]
              if c in dict_df.columns and c in data_df.columns]

    # ── Auto-detect feature columns ──────────────────────────────────
    if isinstance(feature_columns, str) and feature_columns == "auto":
        # Use the union of columns present in both
        feat_from_dict = _auto_feature_columns(dict_df, include_global_rate, global_rate_col)
        feat_from_data = _auto_feature_columns(data_df, include_global_rate, global_rate_col)
        feature_cols = sorted(set(feat_from_dict) & set(feat_from_data))
        if not feature_cols:
            raise ValueError("No common feature columns found between dictionary and dataset.")
    else:
        feature_cols = list(feature_columns)

    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # ── Auto-detect parameter columns ────────────────────────────────
    if param_columns is None:
        param_columns = []
        for c in ["flux_cm2_min", "cos_n",
                   "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
                   "eff_empirical_1", "eff_empirical_2", "eff_empirical_3", "eff_empirical_4"]:
            if c in dict_df.columns:
                param_columns.append(c)
        if not param_columns:
            raise ValueError("No parameter columns found in dictionary.")
    log.info("Parameter columns to estimate: %s", param_columns)

    # ── Coerce feature columns to numeric ────────────────────────────
    dict_features = dict_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    data_features = data_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # ── Scaling ──────────────────────────────────────────────────────
    use_zscore = distance_metric == "l2_zscore"
    if use_zscore:
        # Compute means and stds from the DICTIONARY (reference distribution)
        means = dict_features.mean(axis=0, skipna=True)
        stds = dict_features.std(axis=0, skipna=True).replace({0.0: np.nan})
        dict_scaled = ((dict_features - means) / stds).to_numpy(dtype=float)
        data_scaled = ((data_features - means) / stds).to_numpy(dtype=float)
    else:
        dict_scaled = dict_features.to_numpy(dtype=float)
        data_scaled = data_features.to_numpy(dtype=float)

    dist_fn = DISTANCE_FNS.get(distance_metric, _l2)

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

    # ── Dict parameter values as arrays ──────────────────────────────
    dict_param_arrays = {}
    for pc in param_columns:
        dict_param_arrays[pc] = pd.to_numeric(
            dict_df[pc], errors="coerce"
        ).to_numpy(dtype=float)

    # ── Estimate for each dataset entry ──────────────────────────────
    results = []
    n_data = len(data_df)

    for i in range(n_data):
        row_result: dict = {"dataset_index": i}
        if join_col and data_ids:
            row_result["filename_base"] = data_ids[i]

        # Find z-compatible candidates
        if dict_z is not None and data_z is not None:
            z_mask = np.all(
                np.abs(dict_z - data_z[i]) < z_tol, axis=1
            )
        else:
            z_mask = np.ones(len(dict_df), dtype=bool)

        # Exclude same file
        if exclude_same_file and join_col and dict_ids and data_ids:
            same_file = np.array([did == data_ids[i] for did in dict_ids])
            z_mask &= ~same_file

        cand_indices = np.where(z_mask)[0]
        row_result["n_candidates"] = len(cand_indices)

        if len(cand_indices) == 0:
            # No candidates — set all estimates to NaN
            for pc in param_columns:
                row_result[f"est_{pc}"] = np.nan
            row_result["best_distance"] = np.nan
            results.append(row_result)
            continue

        # Compute distances
        sample_vec = data_scaled[i]
        distances = np.array([
            dist_fn(sample_vec, dict_scaled[j]) for j in cand_indices
        ])

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

        row_result["best_distance"] = float(valid_distances[0])

        # IDW interpolation over K nearest
        k_use = min(interpolation_k, len(valid_indices))
        top_indices = valid_indices[:k_use]
        top_distances = valid_distances[:k_use]

        if k_use == 1 or top_distances[0] < 1e-15:
            # Nearest-only (or exact match)
            best_j = top_indices[0]
            for pc in param_columns:
                row_result[f"est_{pc}"] = float(dict_param_arrays[pc][best_j])
        else:
            # IDW weights: w_i = 1/d_i^2, normalised
            weights = 1.0 / (top_distances ** 2)
            weights /= weights.sum()

            for pc in param_columns:
                vals = dict_param_arrays[pc][top_indices]
                finite = np.isfinite(vals)
                if finite.sum() == 0:
                    row_result[f"est_{pc}"] = np.nan
                else:
                    w = weights.copy()
                    w[~finite] = 0.0
                    if w.sum() > 0:
                        w /= w.sum()
                    row_result[f"est_{pc}"] = float(np.sum(w * np.nan_to_num(vals)))

        results.append(row_result)

        if (i + 1) % 200 == 0 or i == n_data - 1:
            log.info("  Estimated %d / %d", i + 1, n_data)

    result_df = pd.DataFrame(results)
    return result_df
