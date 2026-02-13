#!/usr/bin/env python3
"""STEP 3.1 — Uncertainty assessment.

Given the validation results (estimated vs simulated parameters, number of
events), this step:

1. Bins the data along each estimated parameter AND number of events.
2. For each bin, computes the histogram of relative errors.
3. Identifies and flags outliers using an IQR-based rule.
4. Calculates quantiles (p50, p68, p90, p95) from the non-outlier
   distribution.
5. Assigns an uncertainty per bin centre → builds a Look-Up Table (LUT).
6. Saves the LUT as a CSV (with a header comment describing the
   dictionary that produced it).
7. Provides an interpolation function: given an estimated parameter set
   and event count, retrieves the corresponding uncertainty from the LUT.
8. Plots per-bin histograms, outlier-marked, and the resulting LUT slices.

Output
------
OUTPUTS/FILES/uncertainty_lut.csv         — the LUT
OUTPUTS/FILES/uncertainty_lut_meta.json   — metadata about the LUT
OUTPUTS/FILES/uncertainty_summary.json
OUTPUTS/PLOTS/                            — diagnostic plots
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
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = STEP_DIR.parent
PIPELINE_DIR = INFERENCE_DIR.parent
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"

DEFAULT_VALIDATION = (
    INFERENCE_DIR / "STEP_2_2_VALIDATION"
    / "OUTPUTS" / "FILES" / "validation_results.csv"
)
DEFAULT_DICTIONARY = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dictionary.csv"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="[%(levelname)s] STEP_2.3 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_2.3")


def _load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


# =====================================================================
# Binning helpers
# =====================================================================

def _as_bool(value: object) -> bool:
    """Parse config-like truthy values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _parse_relerr_filter(cfg_value: object) -> tuple[float, float]:
    """Return a sane signed relative-error window [min, max] in percent."""
    default = (-5.0, 5.0)
    if not isinstance(cfg_value, (list, tuple)) or len(cfg_value) != 2:
        return default
    try:
        lo = float(cfg_value[0])
        hi = float(cfg_value[1])
    except (TypeError, ValueError):
        return default
    if not np.isfinite(lo) or not np.isfinite(hi):
        return default
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return (lo, hi)


def _parse_bins_per_param(
    cfg_value: object,
    param_names: list[str],
    default_bins: int = 5,
) -> dict[str, int]:
    """Parse per-parameter bin settings from int or dict config."""
    bins_map: dict[str, int] = {}

    if isinstance(cfg_value, dict):
        fallback = cfg_value.get(
            "__default__",
            cfg_value.get(
                "default",
                default_bins,
            ),
        )
        try:
            fallback_n = max(1, int(fallback))
        except (TypeError, ValueError):
            fallback_n = max(1, int(default_bins))
        for pname in param_names:
            raw = cfg_value.get(pname, fallback_n)
            try:
                bins_map[pname] = max(1, int(raw))
            except (TypeError, ValueError):
                bins_map[pname] = fallback_n
        return bins_map

    try:
        n = max(1, int(cfg_value))
    except (TypeError, ValueError):
        n = max(1, int(default_bins))
    for pname in param_names:
        bins_map[pname] = n
    return bins_map


def _quantile_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    """Create exactly n_bins bins from quantiles (fallback to linear)."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return np.array([], dtype=float)
    n_bins = max(1, int(n_bins))
    if n_bins == 1:
        return _single_bin_edges(series)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.asarray(series.quantile(quantiles).values, dtype=float)
    if np.any(~np.isfinite(edges)) or np.unique(edges).size != edges.size:
        vmin, vmax = float(series.min()), float(series.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return np.array([], dtype=float)
        if vmax <= vmin:
            pad = max(1e-6, abs(vmin) * 1e-6)
            vmin -= pad
            vmax += pad
        edges = np.linspace(vmin, vmax, n_bins + 1, dtype=float)
    # pd.cut requires strictly increasing edges.
    for i in range(1, len(edges)):
        if not edges[i] > edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def _single_bin_edges(series: pd.Series) -> np.ndarray:
    """Build exactly one valid bin from a numeric series."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return np.array([], dtype=float)
    vmin = float(series.min())
    vmax = float(series.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([], dtype=float)
    if vmax <= vmin:
        pad = max(1e-6, abs(vmin) * 1e-6)
        return np.array([vmin - pad, vmin + pad], dtype=float)
    return np.array([vmin, vmax], dtype=float)


def _bin_centres(edges: np.ndarray) -> np.ndarray:
    """Midpoints of bins defined by edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def _build_dimension_edges(
    val_df: pd.DataFrame,
    param_names: list[str],
    param_bin_map: dict[str, int],
    n_bins_events: int,
) -> dict[str, np.ndarray]:
    """Build bin edges for each LUT dimension."""
    all_edges: dict[str, np.ndarray] = {}
    dim_names = [f"est_{p}" for p in param_names] + ["n_events"]

    for dim in dim_names:
        if dim not in val_df.columns:
            log.warning("Column %s not found; skipping dimension.", dim)
            continue
        series = pd.to_numeric(val_df[dim], errors="coerce").dropna()
        if series.empty:
            log.warning("Column %s has no finite values; skipping dimension.", dim)
            continue
        if dim == "n_events":
            requested_bins = max(1, int(n_bins_events))
        else:
            pname = dim[len("est_"):]
            requested_bins = max(1, int(param_bin_map.get(pname, 1)))

        edges = _quantile_edges(series, requested_bins)
        if edges.size < 2:
            log.warning("Could not build edges for %s; skipping dimension.", dim)
            continue
        all_edges[dim] = edges

    return all_edges


def _assign_multidim_bins(
    val_df: pd.DataFrame,
    all_edges: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Assign integer bin indices for each dimension to each row."""
    work = val_df.copy()
    for dim, edges in all_edges.items():
        work[f"_bin_{dim}"] = pd.cut(
            pd.to_numeric(work[dim], errors="coerce"),
            bins=edges,
            include_lowest=True,
            labels=False,
        )
    bin_cols = [f"_bin_{dim}" for dim in all_edges]
    work = work.dropna(subset=bin_cols).copy()
    for bc in bin_cols:
        work[bc] = work[bc].astype(int)
    return work


def _format_sector_bin_lines(
    dim_order: list[str],
    sector_key: tuple[int, ...],
    all_edges: dict[str, np.ndarray],
    parts_per_line: int = 3,
) -> list[str]:
    """Format human-readable bin-value ranges for a sector."""
    parts: list[str] = []
    for dim, bidx in zip(dim_order, sector_key):
        edges = all_edges.get(dim)
        if edges is None or len(edges) < 2:
            continue
        i = int(bidx)
        if i < 0 or i + 1 >= len(edges):
            continue
        lo = float(edges[i])
        hi = float(edges[i + 1])
        label = dim[len("est_"):] if dim.startswith("est_") else dim
        parts.append(f"{label}=[{lo:.4g},{hi:.4g}]")

    if not parts:
        return []
    lines: list[str] = []
    for i in range(0, len(parts), max(1, int(parts_per_line))):
        lines.append(" | ".join(parts[i:i + parts_per_line]))
    return lines


def _iqr_outlier_mask(
    values: np.ndarray, factor: float = 2.5
) -> np.ndarray:
    """Return boolean mask: True = outlier."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=bool)

    finite = np.isfinite(values)
    if finite.sum() < 4:
        # Too few points for robust IQR; keep finite values as non-outliers.
        out = ~finite
        return out

    core = values[finite]
    q1 = np.percentile(core, 25)
    q3 = np.percentile(core, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    out = (values < lower) | (values > upper)
    out |= ~finite
    return out


def _safe_percentile(values: np.ndarray, pct: float) -> float:
    """Percentile over finite values, returning NaN if no valid data."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, pct))


def _safe_median(values: np.ndarray) -> float:
    """Median over finite values, returning NaN if no valid data."""
    return _safe_percentile(values, 50.0)


# =====================================================================
# LUT builder
# =====================================================================

def build_uncertainty_lut(
    val_df: pd.DataFrame,
    param_names: list[str],
    n_bins_param: int | dict[str, int] = 5,
    n_bins_events: int = 5,
    min_bin_count: int = 10,
    quantiles: list[float] | None = None,
    iqr_factor: float = 2.5,
    relerr_filter_pct: tuple[float, float] = (-5.0, 5.0),
    all_edges: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build a multi-dimensional uncertainty LUT.

    For each combination of bins in (estimated parameters, n_events),
    compute error quantiles from the non-outlier relative-error distribution.

    Parameters
    ----------
    val_df : DataFrame
        Must contain columns: est_<param>, true_<param>, n_events,
        relerr_<param>_pct for each param in param_names.
    param_names : list of str
        Parameter names (e.g. ["flux_cm2_min", "eff_sim_2"]).
    n_bins_param : int or dict
        Number of bins per parameter axis; if dict, keys are parameter names.
    n_bins_events : int
        Number of quantile bins for the event-count axis.
    min_bin_count : int
        Minimum non-outlier samples per cell to include in LUT.
    quantiles : list of float
        Error quantiles to compute (default: [0.50, 0.68, 0.90, 0.95]).
    iqr_factor : float
        IQR multiplier for outlier detection.
    relerr_filter_pct : tuple(float, float)
        Signed relative-error filter window [min, max] applied before
        outlier rejection and quantile/statistics extraction.
    all_edges : dict, optional
        Precomputed edges per dimension. If omitted they are built here.

    Returns
    -------
    DataFrame with columns: bin centres for each param + n_events,
    then sigma_<param>_p<q> / sigma_<param>_std, plus sample counters.
    """
    if quantiles is None:
        quantiles = [0.50, 0.68, 0.90, 0.95]
    relerr_min, relerr_max = relerr_filter_pct
    if not np.isfinite(relerr_min) or not np.isfinite(relerr_max):
        relerr_min, relerr_max = (-5.0, 5.0)
    if relerr_min > relerr_max:
        relerr_min, relerr_max = relerr_max, relerr_min

    param_bin_map = _parse_bins_per_param(n_bins_param, param_names, default_bins=5)
    if all_edges is None:
        all_edges = _build_dimension_edges(
            val_df=val_df,
            param_names=param_names,
            param_bin_map=param_bin_map,
            n_bins_events=n_bins_events,
        )

    if not all_edges:
        log.error("No valid dimensions for LUT binning.")
        return pd.DataFrame()

    # Assign bin labels to each row and index observed groups.
    work = _assign_multidim_bins(val_df, all_edges)
    dim_order = list(all_edges.keys())
    bin_cols = [f"_bin_{dim}" for dim in dim_order]
    grouped_lookup: dict[tuple[int, ...], pd.DataFrame] = {}
    for key, group in work.groupby(bin_cols, observed=True):
        if not isinstance(key, tuple):
            key = (int(key),)
        grouped_lookup[tuple(int(k) for k in key)] = group

    # Build full Cartesian sector grid to preserve empty sectors as NaN rows.
    n_bins_by_dim = [max(1, len(all_edges[dim]) - 1) for dim in dim_order]
    rows: list[dict] = []
    for bin_key in np.ndindex(*n_bins_by_dim):
        group = grouped_lookup.get(tuple(int(b) for b in bin_key))

        row: dict = {}
        # Record bin centres
        for dim_name, edges, bk in zip(dim_order, all_edges.values(), bin_key):
            centres = _bin_centres(edges)
            bk_int = int(bk)
            if bk_int < len(centres):
                row[dim_name + "_centre"] = float(centres[bk_int])
            else:
                row[dim_name + "_centre"] = float(edges[-1])
            row[dim_name + "_bin_idx"] = bk_int

        # For each parameter, compute error quantiles
        for pname in param_names:
            err_col = f"relerr_{pname}_pct"
            for q in quantiles:
                row[f"sigma_{pname}_p{int(q * 100)}"] = np.nan
            row[f"sigma_{pname}_std"] = np.nan
            row[f"n_samples_raw_{pname}"] = 0
            row[f"n_samples_{pname}"] = 0
            row[f"n_outliers_{pname}"] = 0
            if group is None or err_col not in group.columns:
                continue

            raw_errs = pd.to_numeric(group[err_col], errors="coerce").to_numpy(dtype=float)
            raw_errs = raw_errs[np.isfinite(raw_errs)]
            row[f"n_samples_raw_{pname}"] = int(raw_errs.size)
            if raw_errs.size == 0:
                continue

            in_window = raw_errs[(raw_errs >= relerr_min) & (raw_errs <= relerr_max)]
            row[f"n_samples_{pname}"] = int(in_window.size)
            if in_window.size == 0:
                continue

            outlier_mask = _iqr_outlier_mask(in_window, iqr_factor)
            clean = in_window[~outlier_mask]
            row[f"n_outliers_{pname}"] = int(outlier_mask.sum())
            if len(clean) < min_bin_count:
                continue

            row[f"sigma_{pname}_std"] = (
                float(np.std(clean, ddof=1)) if len(clean) > 1 else np.nan
            )
            for q in quantiles:
                q_label = str(int(q * 100))
                row[f"sigma_{pname}_p{q_label}"] = _safe_percentile(clean, q * 100)

        rows.append(row)

    lut_df = pd.DataFrame(rows)
    if lut_df.empty:
        log.warning("LUT is empty.")
    else:
        log.info("Built LUT with %d sectors (including empty sectors).", len(lut_df))

    return lut_df


# =====================================================================
# LUT interpolation
# =====================================================================

def interpolate_uncertainty(
    lut_df: pd.DataFrame,
    est_values: dict[str, float],
    param_names: list[str],
    quantile: float = 0.68,
) -> dict[str, float]:
    """Interpolate uncertainty from a LUT for a given set of estimates.

    Uses nearest-neighbour interpolation: finds the LUT cell whose
    centre is closest to the query point and returns its quantile.

    Parameters
    ----------
    lut_df : DataFrame
        The LUT from build_uncertainty_lut.
    est_values : dict
        Estimated parameter values: {"est_flux_cm2_min": ..., "n_events": ...}
    param_names : list of str
        Parameter names in the LUT.
    quantile : float
        Which quantile to return (e.g. 0.68 → p68).

    Returns
    -------
    dict mapping param_name → uncertainty_pct.
    """
    q_label = str(int(quantile * 100))

    # Build multi-dim distance
    centre_cols = [c for c in lut_df.columns if c.endswith("_centre")]
    if not centre_cols:
        return {p: np.nan for p in param_names}

    lut_centres = lut_df[centre_cols].to_numpy(dtype=float)
    # Normalise each dimension by its range
    ranges = lut_centres.max(axis=0) - lut_centres.min(axis=0)
    ranges[ranges == 0] = 1.0

    query = []
    for cc in centre_cols:
        dim = cc.replace("_centre", "")
        query.append(est_values.get(dim, 0.0))
    query_arr = np.array(query, dtype=float)

    distances = np.sqrt(np.sum(((lut_centres - query_arr) / ranges) ** 2, axis=1))
    nearest_idx = int(np.nanargmin(distances))

    result = {}
    for pname in param_names:
        col = f"sigma_{pname}_p{q_label}"
        if col in lut_df.columns:
            result[pname] = float(lut_df.iloc[nearest_idx][col])
        else:
            result[pname] = np.nan

    return result


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 3.1: Build uncertainty LUT from validation results."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--validation-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_31 = config.get("step_2_3", config.get("step_3_1", {}))

    val_path = Path(args.validation_csv) if args.validation_csv else DEFAULT_VALIDATION
    dict_path = Path(args.dictionary_csv) if args.dictionary_csv else DEFAULT_DICTIONARY

    n_bins_param_cfg = cfg_31.get("n_bins_per_param", 5)
    n_bins_events = int(cfg_31.get("n_bins_events", 5))
    min_bin_count = int(cfg_31.get("min_bin_count", 10))
    quantiles = cfg_31.get("quantiles", [0.50, 0.68, 0.90, 0.95])
    q_clean: list[float] = []
    for q in quantiles:
        try:
            qv = float(q)
        except (TypeError, ValueError):
            continue
        if np.isfinite(qv) and 0.0 <= qv <= 1.0:
            q_clean.append(qv)
    quantiles = q_clean if q_clean else [0.50, 0.68, 0.90, 0.95]
    iqr_factor = float(cfg_31.get("outlier_iqr_factor", 2.5))
    relerr_filter_pct = _parse_relerr_filter(cfg_31.get("relerr_filter_pct", [-5.0, 5.0]))
    exclude_dictionary_entries = _as_bool(cfg_31.get("exclude_dictionary_entries", True))

    # Plot-relevant parameters from config:
    # prefer step_2_3, fallback to step_2_2, then step_1_2.
    plot_params = cfg_31.get("plot_parameters", None)
    if plot_params is None:
        plot_params = config.get("step_2_2", {}).get("plot_parameters", None)
    if plot_params is None:
        plot_params = config.get("step_1_2", {}).get("plot_parameters", None)

    # Ellipse-plot configuration (simple, optional).
    # n_points limits only the number of validation/tested points drawn.
    ellipse_cfg = {
        "params": cfg_31.get("ellipse_params", None),  # e.g. ["flux_cm2_min", "eff_sim_2"]
        "n_points": int(cfg_31.get("ellipse_n_points", 10)),
        "quantile": float(cfg_31.get("ellipse_quantile", 0.68)),
    }

    if not val_path.exists():
        log.error("Validation CSV not found: %s", val_path)
        return 1

    log.info("Loading validation results: %s", val_path)
    val_df = pd.read_csv(val_path, low_memory=False)
    log.info("  Rows: %d", len(val_df))

    # Detect which parameters we can assess
    param_names = []
    for col in val_df.columns:
        if col.startswith("relerr_") and col.endswith("_pct"):
            pname = col[len("relerr_"):-len("_pct")]
            est_col = f"est_{pname}"
            if est_col in val_df.columns:
                param_names.append(pname)
    param_names = list(dict.fromkeys(param_names))
    log.info("Parameters for uncertainty: %s", param_names)

    if not param_names:
        log.error("No estimated parameter columns found.")
        return 1

    # Load dictionary (used in plotting).
    dict_df: pd.DataFrame | None = None
    if dict_path.exists():
        try:
            dict_df = pd.read_csv(dict_path, low_memory=False)
        except Exception as exc:
            log.warning("Could not load dictionary for plotting: %s", exc)
            dict_df = None

    # Exclude dictionary points from uncertainty calculations if requested.
    excluded_dict_entries = 0
    if exclude_dictionary_entries and "true_is_dictionary_entry" in val_df.columns:
        dict_mask = val_df["true_is_dictionary_entry"].map(_as_bool).fillna(False)
        excluded_dict_entries = int(dict_mask.sum())
        val_df = val_df.loc[~dict_mask].copy()
        log.info(
            "Excluded %d dictionary-entry validation rows; using %d rows for uncertainty.",
            excluded_dict_entries,
            len(val_df),
        )
    elif exclude_dictionary_entries:
        log.warning(
            "Config requested dictionary-entry exclusion, but column true_is_dictionary_entry is missing."
        )

    param_bin_map = _parse_bins_per_param(n_bins_param_cfg, param_names, default_bins=5)
    all_edges = _build_dimension_edges(
        val_df=val_df,
        param_names=param_names,
        param_bin_map=param_bin_map,
        n_bins_events=n_bins_events,
    )
    if not all_edges:
        log.error("No bin edges could be built for uncertainty LUT.")
        return 1

    # ── Build LUT ────────────────────────────────────────────────────
    lut_df = build_uncertainty_lut(
        val_df,
        param_names=param_names,
        n_bins_param=param_bin_map,
        n_bins_events=n_bins_events,
        min_bin_count=min_bin_count,
        quantiles=quantiles,
        iqr_factor=iqr_factor,
        relerr_filter_pct=relerr_filter_pct,
        all_edges=all_edges,
    )

    # ── Save LUT ─────────────────────────────────────────────────────
    lut_path = FILES_DIR / "uncertainty_lut.csv"

    # Build a header comment with dictionary info
    dict_info = ""
    if dict_path.exists():
        dict_info = f"Dictionary: {dict_path}"
        try:
            d = pd.read_csv(dict_path, nrows=0)
            dict_info += f", columns: {len(d.columns)}"
        except Exception:
            pass

    header_comment = (
        f"# Uncertainty LUT — generated from validation of dictionary-based estimation\n"
        f"# {dict_info}\n"
        f"# Parameters: {param_names}\n"
        f"# Quantiles: {quantiles}\n"
        f"# Bins per param: {param_bin_map}, events bins: {n_bins_events}\n"
        f"# relerr filter [%]: [{relerr_filter_pct[0]}, {relerr_filter_pct[1]}]\n"
        f"# Exclude dictionary entries: {exclude_dictionary_entries}\n"
        f"# Min bin count: {min_bin_count}, IQR factor: {iqr_factor}\n"
    )

    with open(lut_path, "w", encoding="utf-8") as f:
        f.write(header_comment)
        lut_df.to_csv(f, index=False)
    log.info("Wrote LUT: %s (%d cells)", lut_path, len(lut_df))

    # Metadata JSON
    meta = {
        "dictionary": str(dict_path),
        "param_names": param_names,
        "quantiles": quantiles,
        "n_bins_param": param_bin_map,
        "n_bins_events": n_bins_events,
        "min_bin_count": min_bin_count,
        "iqr_factor": iqr_factor,
        "relerr_filter_pct": [relerr_filter_pct[0], relerr_filter_pct[1]],
        "exclude_dictionary_entries": bool(exclude_dictionary_entries),
        "excluded_dictionary_rows": int(excluded_dict_entries),
        "dimension_bins": {dim: int(len(edges) - 1) for dim, edges in all_edges.items()},
        "lut_cells": len(lut_df),
    }
    with open(FILES_DIR / "uncertainty_lut_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    summary: dict = {
        "total_validation_points": len(val_df),
        "excluded_dictionary_rows": int(excluded_dict_entries),
        "lut_cells": len(lut_df),
        "param_names": param_names,
        "relerr_filter_pct": [relerr_filter_pct[0], relerr_filter_pct[1]],
    }
    for pname in param_names:
        s68_col = f"sigma_{pname}_p68"
        if s68_col in lut_df.columns:
            s = lut_df[s68_col].dropna()
            if not s.empty:
                summary[f"median_sigma_{pname}_p68"] = float(s.median())
                summary[f"min_sigma_{pname}_p68"] = float(s.min())
                summary[f"max_sigma_{pname}_p68"] = float(s.max())
    with open(FILES_DIR / "uncertainty_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ────────────────────────────────────────────────────────
    _make_plots(
        val_df,
        lut_df,
        param_names,
        quantiles,
        iqr_factor,
        param_bin_map,
        n_bins_events,
        relerr_filter_pct,
        all_edges,
        plot_params,
        dict_df,
        ellipse_cfg,
    )

    log.info("Done.")
    return 0


def _make_plots(
    val_df: pd.DataFrame,
    lut_df: pd.DataFrame,
    param_names: list[str],
    quantiles: list[float],
    iqr_factor: float,
    n_bins_param: int | dict[str, int],
    n_bins_events: int,
    relerr_filter_pct: tuple[float, float],
    all_edges: dict[str, np.ndarray],
    plot_params: list[str] | None = None,
    dict_df: pd.DataFrame | None = None,
    ellipse_cfg: dict | None = None,
) -> None:
    """Produce diagnostic plots for the uncertainty assessment.

    Parameters
    ----------
    dict_df : DataFrame or None
        Optional dictionary DataFrame for plotting dictionary entries in the
        ellipse scatter plot.
    ellipse_cfg : dict or None
        Configuration for ellipse plotting. Keys: 'params', 'n_points',
        'quantile'. The dictionary overlay always uses all available points.
    """
    # Decide which params to plot
    plot_pnames = [p for p in param_names if p in plot_params] if plot_params else param_names
    if plot_params:
        missing = [p for p in plot_params if p not in param_names]
        if missing:
            log.warning(
                "Ignoring plot parameters not present in uncertainty inputs: %s",
                missing,
            )
    if not plot_pnames:
        plot_pnames = param_names
        log.warning("No valid plot parameters selected; plotting all parameters.")

    relerr_min, relerr_max = relerr_filter_pct

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Remove stale per-parameter files so outputs reflect current config only.
    for pattern in (
        "error_hist_by_events_*.png",
        "error_hist_by_sector_*.png",
        "error_histograms_parameter_event_space.png",
        "error_histograms_parameter_event_space_*.png",
        "lut_heatmap_*.png",
        "uncertainty_vs_events_*.png",
    ):
        for old_plot in PLOTS_DIR.glob(pattern):
            try:
                old_plot.unlink()
            except OSError as exc:
                log.warning("Could not remove old plot %s: %s", old_plot, exc)

    # ── 1. Per-sector histograms (columns=n_events bins, rows=parameter combos) ───
    if all_edges:
        val_work = _assign_multidim_bins(val_df, all_edges)
        dim_order = list(all_edges.keys())
        has_events_dim = "n_events" in dim_order and f"_bin_n_events" in val_work.columns
        event_dim = "n_events" if has_events_dim else None
        row_dims = [d for d in dim_order if d != event_dim] if event_dim is not None else dim_order.copy()
        event_bins = max(1, len(all_edges[event_dim]) - 1) if event_dim is not None else 1
        row_shape = [max(1, len(all_edges[d]) - 1) for d in row_dims]
        row_keys = list(np.ndindex(*row_shape)) if row_shape else [tuple()]
        # Default histogram edges (n_edges points => n_edges-1 bins). For two
        # frequently-inspected parameters use a finer binning to show detail.
        hist_n_edges_default = 33
        _hist_n_edges_special = {"eff_sim_1": 101, "flux_cm2_min": 101}

        for pname in plot_pnames:
            # per-parameter histogram edges (increase bins for requested params)
            n_edges = _hist_n_edges_special.get(pname, hist_n_edges_default)
            hist_edges = np.linspace(relerr_min, relerr_max, int(n_edges))

            err_col = f"relerr_{pname}_pct"
            if err_col not in val_work.columns:
                continue

            payload_rows: list[dict[str, object]] = []
            global_ymax = 0

            for row_key in row_keys:
                row_selector = np.ones(len(val_work), dtype=bool)
                for dim, bidx in zip(row_dims, row_key):
                    row_selector &= (val_work[f"_bin_{dim}"] == int(bidx)).to_numpy()

                row_panels: list[dict[str, object]] = []
                row_has_data = False
                for ev_bin in range(event_bins):
                    selector = row_selector.copy()
                    if event_dim is not None:
                        selector &= (val_work[f"_bin_{event_dim}"] == int(ev_bin)).to_numpy()
                    group = val_work.loc[selector]

                    raw_errs = pd.to_numeric(group.get(err_col), errors="coerce").to_numpy(dtype=float)
                    total = raw_errs[np.isfinite(raw_errs)]
                    in_window = total[(total >= relerr_min) & (total <= relerr_max)]
                    if in_window.size > 0:
                        outlier_mask = _iqr_outlier_mask(in_window, iqr_factor)
                        filtered = in_window[~outlier_mask]
                    else:
                        filtered = np.array([], dtype=float)

                    total_in_range = total[(total >= relerr_min) & (total <= relerr_max)]
                    h_total, _ = np.histogram(total_in_range, bins=hist_edges)
                    h_filtered, _ = np.histogram(filtered, bins=hist_edges)
                    panel_ymax = max(int(np.max(h_total)) if h_total.size else 0, int(np.max(h_filtered)) if h_filtered.size else 0)
                    global_ymax = max(global_ymax, panel_ymax)
                    row_has_data = row_has_data or (len(total) > 0)

                    row_panels.append(
                        {
                            "total_in_range": total_in_range,
                            "filtered": filtered,
                            "n_total": int(len(total)),
                            "n_in_window": int(len(in_window)),
                            "n_filtered": int(len(filtered)),
                        }
                    )

                if row_has_data:
                    payload_rows.append(
                        {
                            "row_key": tuple(int(x) for x in row_key),
                            "row_lines": _format_sector_bin_lines(row_dims, tuple(int(x) for x in row_key), all_edges, parts_per_line=2),
                            "panels": row_panels,
                        }
                    )

            if not payload_rows:
                log.info("No non-empty sector histograms to plot for %s.", pname)
                continue

            n_rows = len(payload_rows)
            n_cols = event_bins
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.3 * n_cols, 2.8 * n_rows),
                squeeze=False,
                constrained_layout=True,
            )

            y_upper = max(float(global_ymax) * 1.30, 2.0)
            for r, row_payload in enumerate(payload_rows):
                row_lines = row_payload.get("row_lines", [])
                row_text = "\n".join(row_lines) if row_lines else "all parameter bins"
                for c in range(n_cols):
                    ax = axes[r, c]
                    panel = row_payload["panels"][c]
                    total_in_range = np.asarray(panel["total_in_range"], dtype=float)
                    filtered = np.asarray(panel["filtered"], dtype=float)
                    n_total = int(panel["n_total"])
                    n_in_window = int(panel["n_in_window"])
                    n_filtered = int(panel["n_filtered"])
                    n_outside = n_total - n_in_window

                    if total_in_range.size > 0:
                        ax.hist(
                            total_in_range,
                            bins=hist_edges,
                            histtype="step",
                            color="#4D4D4D",
                            linewidth=1.2,
                            alpha=0.95,
                            label="Total (in range)",
                        )
                    if filtered.size > 0:
                        ax.hist(
                            filtered,
                            bins=hist_edges,
                            color="#4C78A8",
                            alpha=0.65,
                            edgecolor="white",
                            linewidth=0.35,
                            label="Filtered (used)",
                        )

                    for q in quantiles:
                        qv = _safe_percentile(filtered, q * 100.0)
                        if np.isfinite(qv):
                            ax.axvline(
                                qv,
                                linestyle="--",
                                linewidth=0.8,
                                color="#303030",
                                alpha=0.85,
                            )

                    ax.set_xlim(relerr_min, relerr_max)
                    ax.set_yscale("log")
                    ax.set_ylim(0.8, y_upper)
                    ax.grid(True, which="both", alpha=0.26)
                    ax.tick_params(labelsize=6)

                    if r == 0:
                        if event_dim is not None:
                            lo = float(all_edges[event_dim][c])
                            hi = float(all_edges[event_dim][c + 1])
                            ax.set_title(f"n_events [{lo:.0f}, {hi:.0f}]", fontsize=8)
                        else:
                            ax.set_title("all n_events", fontsize=8)

                    if c == 0:
                        ax.set_ylabel(f"Count (log)\n{row_text}", fontsize=6.4)
                    if r == n_rows - 1:
                        ax.set_xlabel(f"Rel. error {pname} [%]", fontsize=7)

                    ax.text(
                        0.02,
                        0.98,
                        f"Ntot={n_total}\nNfilt={n_filtered}\nout={n_outside}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6.3,
                        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", alpha=0.65, edgecolor="#C0C0C0"),
                    )
                    if r == 0 and c == 0:
                        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85)

            fig.suptitle(
                f"{pname}: histogram by parameter-bin rows and n_events-bin columns",
                fontsize=10.5,
                y=1.01,
            )
            fig.savefig(PLOTS_DIR / f"error_hist_by_events_{pname}.png", dpi=220)
            plt.close(fig)
            log.info(
                "Histogram grid for %s (rows=%d parameter-combo bins, cols=%d event bins).",
                pname,
                n_rows,
                n_cols,
            )

    # ── 2. LUT heatmap in flux-eff plane by event-count bins ─────────
    # Rows: selected parameters (flux row first if present).
    # Cols: n_events bins (left=low counts, right=high counts for 2 bins).
    if not lut_df.empty:
        flux_plane_param = "flux_cm2_min" if f"est_flux_cm2_min" in all_edges else None
        eff_plane_param = None
        for candidate in plot_pnames + param_names:
            if candidate.startswith("eff_") and f"est_{candidate}" in all_edges:
                eff_plane_param = candidate
                break

        # Fallbacks if canonical flux/eff are not available.
        if flux_plane_param is None:
            for candidate in plot_pnames + param_names:
                if f"est_{candidate}" in all_edges:
                    flux_plane_param = candidate
                    break
        if eff_plane_param is None:
            for candidate in plot_pnames + param_names:
                if candidate != flux_plane_param and f"est_{candidate}" in all_edges:
                    eff_plane_param = candidate
                    break

        x_edges = all_edges.get(f"est_{flux_plane_param}") if flux_plane_param else None
        y_edges = all_edges.get(f"est_{eff_plane_param}") if eff_plane_param else None
        ne_edges = all_edges.get("n_events")
        if (
            flux_plane_param is not None
            and eff_plane_param is not None
            and x_edges is not None
            and y_edges is not None
            and ne_edges is not None
            and len(x_edges) >= 2
            and len(y_edges) >= 2
            and len(ne_edges) >= 2
            and f"est_{flux_plane_param}_bin_idx" in lut_df.columns
            and f"est_{eff_plane_param}_bin_idx" in lut_df.columns
            and "n_events_bin_idx" in lut_df.columns
        ):
            panel_params = [p for p in plot_pnames if p in param_names]
            if "flux_cm2_min" in panel_params:
                panel_params = ["flux_cm2_min"] + [p for p in panel_params if p != "flux_cm2_min"]
            if eff_plane_param in panel_params and "flux_cm2_min" in panel_params:
                panel_params = (
                    ["flux_cm2_min", eff_plane_param]
                    + [p for p in panel_params if p not in {"flux_cm2_min", eff_plane_param}]
                )

            n_rows = max(1, len(panel_params))
            n_cols = len(ne_edges) - 1
            n_x = len(x_edges) - 1
            n_y = len(y_edges) - 1
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.6 * n_cols, 3.7 * n_rows),
                squeeze=False,
                constrained_layout=True,
            )

            x_bin_all = pd.to_numeric(lut_df[f"est_{flux_plane_param}_bin_idx"], errors="coerce")
            y_bin_all = pd.to_numeric(lut_df[f"est_{eff_plane_param}_bin_idx"], errors="coerce")
            ne_bin_all = pd.to_numeric(lut_df["n_events_bin_idx"], errors="coerce")

            # Overlay dictionary points only inside the plotted LUT plane.
            dict_fx = None
            dict_ef = None
            dict_mask_in_plane = None
            dict_in_plane = 0
            dict_out_of_plane = 0
            x_min, x_max = float(x_edges[0]), float(x_edges[-1])
            y_min, y_max = float(y_edges[0]), float(y_edges[-1])
            if dict_df is not None and flux_plane_param in dict_df.columns and eff_plane_param in dict_df.columns:
                dict_fx = pd.to_numeric(dict_df[flux_plane_param], errors="coerce")
                dict_ef = pd.to_numeric(dict_df[eff_plane_param], errors="coerce")
                dict_mask_all = dict_fx.notna() & dict_ef.notna()
                dict_mask_in_plane = (
                    dict_mask_all
                    & (dict_fx >= x_min)
                    & (dict_fx <= x_max)
                    & (dict_ef >= y_min)
                    & (dict_ef <= y_max)
                )
                dict_in_plane = int(dict_mask_in_plane.sum())
                dict_out_of_plane = int(dict_mask_all.sum() - dict_in_plane)

            for r, pname in enumerate(panel_params):
                sig_col = f"sigma_{pname}_std"
                sig_label = f"σ_{pname} std [%]"
                if sig_col not in lut_df.columns:
                    sig_col = f"sigma_{pname}_p68"
                    sig_label = f"σ_{pname} p68 [%]"
                if sig_col not in lut_df.columns:
                    for c in range(n_cols):
                        axes[r, c].axis("off")
                    continue

                sigma_all = pd.to_numeric(lut_df[sig_col], errors="coerce")
                valid_row = x_bin_all.notna() & y_bin_all.notna() & ne_bin_all.notna() & sigma_all.notna()

                # Shared row color scale across event-bin columns.
                row_vals = sigma_all[valid_row].to_numpy(dtype=float)
                if row_vals.size == 0:
                    vmin, vmax = 0.0, 1.0
                else:
                    vmin = float(np.nanpercentile(row_vals, 5))
                    vmax = float(np.nanpercentile(row_vals, 95))
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                        vmin = float(np.nanmin(row_vals))
                        vmax = float(np.nanmax(row_vals))
                        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                            vmin, vmax = 0.0, 1.0

                row_mesh = None
                for c in range(n_cols):
                    ax = axes[r, c]
                    use = valid_row & (ne_bin_all.astype("Int64") == c)
                    z_grid = np.full((n_y, n_x), np.nan, dtype=float)
                    if use.any():
                        xb = x_bin_all[use].astype(int)
                        yb = y_bin_all[use].astype(int)
                        zv = sigma_all[use]
                        grouped = (
                            pd.DataFrame({"xb": xb, "yb": yb, "z": zv})
                            .groupby(["yb", "xb"], observed=True)["z"]
                            .median()
                        )
                        for (yb_i, xb_i), zmed in grouped.items():
                            if 0 <= int(xb_i) < n_x and 0 <= int(yb_i) < n_y:
                                z_grid[int(yb_i), int(xb_i)] = float(zmed)

                    z_masked = np.ma.masked_invalid(z_grid)
                    row_mesh = ax.pcolormesh(
                        x_edges,
                        y_edges,
                        z_masked,
                        cmap="YlOrRd",
                        shading="flat",
                        alpha=0.45,
                        vmin=vmin,
                        vmax=vmax,
                        zorder=1,
                    )

                    if dict_fx is not None and dict_ef is not None and dict_mask_in_plane is not None:
                        if dict_mask_in_plane.any():
                            ax.scatter(
                                dict_fx[dict_mask_in_plane],
                                dict_ef[dict_mask_in_plane],
                                s=18,
                                marker="o",
                                color="#1f4e79",
                                alpha=0.62,
                                edgecolors="white",
                                linewidths=0.30,
                                zorder=3,
                            )

                    lo_ev = float(ne_edges[c])
                    hi_ev = float(ne_edges[c + 1])
                    ax.set_title(f"{pname} | n_events [{lo_ev:.0f}, {hi_ev:.0f}]")
                    if r == n_rows - 1:
                        ax.set_xlabel(f"{flux_plane_param}")
                    if c == 0:
                        ax.set_ylabel(f"{eff_plane_param}")
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.text(
                        0.02,
                        0.02,
                        f"dict in-plane: {dict_in_plane} | out: {dict_out_of_plane}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=6.3,
                        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", alpha=0.65, edgecolor="#C0C0C0"),
                        zorder=5,
                    )
                    ax.grid(True, alpha=0.18)

                if row_mesh is not None:
                    fig.colorbar(
                        row_mesh,
                        ax=list(axes[r, :]),
                        label=sig_label,
                        shrink=0.92,
                        pad=0.035,
                        fraction=0.045,
                        aspect=24,
                    )

            fig.suptitle(
                f"LUT uncertainty in {flux_plane_param} vs {eff_plane_param} plane by event-count bins",
                fontsize=11,
            )
            fig.savefig(PLOTS_DIR / "lut_heatmap_eff_flux_plane.png", dpi=170)
            plt.close(fig)

    # ── 3. Ellipse-plane plot: show simulated (true) pairs, dictionary, and
    #      uncertainty ellipses centred on estimated pairs. (Simple, configurable)
    if ellipse_cfg is None:
        ellipse_cfg = {"params": None, "n_points": 0, "quantile": 0.68}

    # Decide which two parameters to use
    ppair = None
    if ellipse_cfg.get("params") and isinstance(ellipse_cfg.get("params"), (list, tuple)):
        a, b = ellipse_cfg.get("params")[:2]
        if a in param_names and b in param_names:
            ppair = (a, b)

    # Prefer the flux vs efficiency plane if both are present: (flux_cm2_min, eff_sim_2)
    if ppair is None:
        preferred = ("flux_cm2_min", "eff_sim_2")
        if preferred[0] in param_names and preferred[1] in param_names:
            ppair = preferred
        elif len(param_names) >= 2:
            ppair = (param_names[0], param_names[1])

    # Disable a specific undesired plot: flux_cm2_min vs cos_n

    if ppair is not None and ellipse_cfg.get("n_points", 0) > 0:
        x_p, y_p = ppair
        est_x_col = f"est_{x_p}"
        est_y_col = f"est_{y_p}"
        true_x_col = f"true_{x_p}"
        true_y_col = f"true_{y_p}"

        if {est_x_col, est_y_col}.issubset(set(val_df.columns)):
            # Prepare sample of validation points
            vv = val_df.dropna(subset=[est_x_col, est_y_col, "n_events"]).reset_index(drop=True)
            n_points = int(min(len(vv), max(0, int(ellipse_cfg.get("n_points", 0)))))
            if n_points > 0 and len(vv) > 0:
                if len(vv) <= n_points:
                    sel_idx = np.arange(len(vv))
                else:
                    sel_idx = np.linspace(0, len(vv) - 1, n_points, dtype=int)
                sel = vv.iloc[sel_idx]

                fig, ax = plt.subplots(figsize=(8.2, 6.4))
                ax.set_facecolor("#FCFCFD")
                ax.set_xlabel(x_p)
                ax.set_ylabel(y_p)
                ax.set_title(f"Validation: {y_p} vs {x_p} — ellipses (p{int(ellipse_cfg.get('quantile', 0.68)*100)})")
                ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

                # Plot all available validation truths as a light background cloud.
                n_truth_bg = 0
                if {true_x_col, true_y_col}.issubset(set(vv.columns)):
                    vtruth = vv.dropna(subset=[true_x_col, true_y_col])
                    n_truth_bg = len(vtruth)
                    if n_truth_bg > 0:
                        ax.scatter(
                            vtruth[true_x_col],
                            vtruth[true_y_col],
                            marker="o",
                            s=22,
                            color="#666666",
                            alpha=0.34,
                            edgecolors="white",
                            linewidths=0.22,
                            zorder=1,
                            label=f"Validation truths (all: {n_truth_bg})",
                        )

                # Plot all dictionary entries for context (no subsampling).
                n_dict = 0
                if dict_df is not None and {x_p, y_p}.issubset(set(dict_df.columns)):
                    ddf = dict_df.dropna(subset=[x_p, y_p])
                    n_dict = len(ddf)
                    if n_dict > 0:
                        ax.scatter(
                            ddf[x_p],
                            ddf[y_p],
                            marker="s",
                            facecolors="none",
                            edgecolors="#2F4B7C",
                            s=30,
                            alpha=0.86,
                            linewidths=0.75,
                            zorder=2,
                            label=f"Dictionary (all: {n_dict})",
                        )

                # Plot selected points: ellipses centred at estimates and markers for true values
                x_series = pd.to_numeric(vv[est_x_col], errors="coerce").to_numpy(dtype=float)
                y_series = pd.to_numeric(vv[est_y_col], errors="coerce").to_numpy(dtype=float)
                if {true_x_col, true_y_col}.issubset(set(vv.columns)):
                    x_true_series = pd.to_numeric(vv[true_x_col], errors="coerce").to_numpy(dtype=float)
                    y_true_series = pd.to_numeric(vv[true_y_col], errors="coerce").to_numpy(dtype=float)
                    x_series = np.concatenate([x_series, x_true_series[np.isfinite(x_true_series)]])
                    y_series = np.concatenate([y_series, y_true_series[np.isfinite(y_true_series)]])
                x_series = x_series[np.isfinite(x_series)]
                y_series = y_series[np.isfinite(y_series)]
                x_span = float(np.nanmax(x_series) - np.nanmin(x_series)) if x_series.size else 1.0
                y_span = float(np.nanmax(y_series) - np.nanmin(y_series)) if y_series.size else 1.0
                if not np.isfinite(x_span) or x_span <= 0:
                    x_span = 1.0
                if not np.isfinite(y_span) or y_span <= 0:
                    y_span = 1.0

                arrow_lengths: list[float] = []
                for _, row in sel.iterrows():
                    true_x = float(row[true_x_col]) if true_x_col in row and not pd.isna(row[true_x_col]) else np.nan
                    true_y = float(row[true_y_col]) if true_y_col in row and not pd.isna(row[true_y_col]) else np.nan
                    est_x = float(row[est_x_col]) if est_x_col in row and not pd.isna(row[est_x_col]) else np.nan
                    est_y = float(row[est_y_col]) if est_y_col in row and not pd.isna(row[est_y_col]) else np.nan
                    if np.isfinite(est_x) and np.isfinite(est_y) and np.isfinite(true_x) and np.isfinite(true_y):
                        dlen = float(np.hypot((true_x - est_x) / x_span, (true_y - est_y) / y_span))
                        arrow_lengths.append(dlen)

                arrow_norm = None
                arrow_cmap = None
                if arrow_lengths:
                    lmin = float(np.nanmin(arrow_lengths))
                    lmax = float(np.nanmax(arrow_lengths))
                    if not np.isfinite(lmin) or not np.isfinite(lmax):
                        lmin, lmax = 0.0, 1.0
                    if lmax <= lmin:
                        lmax = lmin + 1e-12
                    arrow_norm = Normalize(vmin=lmin, vmax=lmax)
                    arrow_cmap = plt.get_cmap("viridis")

                ell_label_shown = False
                est_label_shown = False
                true_label_shown = False
                arrows_drawn = 0
                for _, row in sel.iterrows():
                    est_x = float(row[est_x_col])
                    est_y = float(row[est_y_col])
                    ne = float(row.get("n_events", np.nan))
                    true_x = float(row[true_x_col]) if true_x_col in row and not pd.isna(row[true_x_col]) else None
                    true_y = float(row[true_y_col]) if true_y_col in row and not pd.isna(row[true_y_col]) else None

                    # Interpolate uncertainties from LUT
                    try:
                        q = float(ellipse_cfg.get("quantile", 0.68))
                        est_values = {f"est_{x_p}": est_x, f"est_{y_p}": est_y, "n_events": ne}
                        sigs = interpolate_uncertainty(lut_df, est_values, param_names, quantile=q)
                        sigma_x_pct = sigs.get(x_p, np.nan)
                        sigma_y_pct = sigs.get(y_p, np.nan)
                    except Exception:
                        sigma_x_pct = np.nan
                        sigma_y_pct = np.nan

                    if np.isfinite(sigma_x_pct) and np.isfinite(sigma_y_pct) and sigma_x_pct > 0 and sigma_y_pct > 0:
                        sigma_x = sigma_x_pct / 100.0 * abs(est_x)
                        sigma_y = sigma_y_pct / 100.0 * abs(est_y)
                        # width/height are diameters
                        ell = Ellipse((est_x, est_y), width=2*sigma_x, height=2*sigma_y,
                                      edgecolor=(31/255, 119/255, 180/255, 0.90),
                                      facecolor=(31/255, 119/255, 180/255, 0.22),
                                      linewidth=1.25)
                        ax.add_patch(ell)
                        if not ell_label_shown:
                            ax.plot([], [], color="#1f77b4", linewidth=1.25, label="Uncertainty ellipse")
                            ell_label_shown = True
                        ax.scatter(
                            [est_x], [est_y],
                            marker="x",
                            color="#1f77b4",
                            s=48,
                            zorder=4,
                            label="Estimate" if not est_label_shown else "",
                        )
                        est_label_shown = True
                    else:
                        ax.scatter(
                            [est_x], [est_y],
                            marker="x",
                            color="#1f77b4",
                            s=48,
                            zorder=4,
                            label="Estimate" if not est_label_shown else "",
                        )
                        est_label_shown = True

                    if true_x is not None and true_y is not None:
                        ax.scatter(
                            [true_x], [true_y],
                            marker="o",
                            color="#d62728",
                            s=34,
                            zorder=5,
                            label="True" if not true_label_shown else "",
                        )
                        true_label_shown = True
                        # Draw arrow from estimate → true.
                        if np.isfinite(true_x) and np.isfinite(true_y):
                            dlen = float(np.hypot((true_x - est_x) / x_span, (true_y - est_y) / y_span))
                            if arrow_norm is not None and arrow_cmap is not None:
                                arrow_color = arrow_cmap(float(arrow_norm(dlen)))
                            else:
                                arrow_color = "#6E6E6E"
                            # Thick translucent trail to make direction/error magnitude visually clear.
                            ax.plot(
                                [est_x, true_x],
                                [est_y, true_y],
                                color=arrow_color,
                                linewidth=4.2,
                                alpha=0.32,
                                solid_capstyle="round",
                                zorder=3,
                            )
                            ax.annotate(
                                "",
                                xy=(true_x, true_y),
                                xytext=(est_x, est_y),
                                arrowprops={
                                    "arrowstyle": "-|>",
                                    "color": arrow_color,
                                    "linewidth": 1.35,
                                    "alpha": 0.96,
                                    "mutation_scale": 11,
                                    "shrinkA": 2,
                                    "shrinkB": 2,
                                },
                                zorder=4,
                            )
                            arrows_drawn += 1

                if arrows_drawn > 0:
                    ax.plot([], [], color="#6E6E6E", linewidth=1.35, label="Estimate -> True")
                    if arrow_norm is not None and arrow_cmap is not None:
                        sm = cm.ScalarMappable(norm=arrow_norm, cmap=arrow_cmap)
                        sm.set_array([])
                        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
                        cbar.set_label("Estimate -> true distance (norm.)", fontsize=8)
                        cbar.ax.tick_params(labelsize=7)

                ax.text(
                    0.015,
                    0.985,
                    f"Dictionary: {n_dict}\nValidation truths: {n_truth_bg}\nSelected tests: {len(sel)}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.28",
                        "facecolor": "white",
                        "edgecolor": "#B7B7B7",
                        "alpha": 0.88,
                    },
                )

                ax.legend(fontsize=8, framealpha=0.95)
                fig.tight_layout()
                outname = PLOTS_DIR / f"lut_ellipse_{x_p}_{y_p}.png"
                fig.savefig(outname)
                plt.close(fig)
                log.info("Wrote ellipse scatter plot: %s", outname)


if __name__ == "__main__":
    raise SystemExit(main())
