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

    # Ellipse-plot configuration (simple, optional)
    ellipse_cfg = {
        "params": cfg_31.get("ellipse_params", None),  # e.g. ["flux_cm2_min", "eff_sim_2"]
        "n_points": int(cfg_31.get("ellipse_n_points", 10)),
        "quantile": float(cfg_31.get("ellipse_quantile", 0.68)),
        "show_dictionary": bool(cfg_31.get("ellipse_show_dictionary", True)),
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
        'quantile', 'show_dictionary'.
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

    # ── 1. Per-sector histograms in full multidimensional bin space ───
    if all_edges:
        val_work = _assign_multidim_bins(val_df, all_edges)
        dim_order = list(all_edges.keys())
        n_bins_by_dim = [max(1, len(all_edges[d]) - 1) for d in dim_order]
        for pname in plot_pnames:
            err_col = f"relerr_{pname}_pct"
            if err_col not in val_work.columns:
                continue

            all_hist_payload: list[dict[str, object]] = []
            for sector_key in np.ndindex(*n_bins_by_dim):
                selector = np.ones(len(val_work), dtype=bool)
                for dim, bidx in zip(dim_order, sector_key):
                    selector &= (val_work[f"_bin_{dim}"] == int(bidx)).to_numpy()
                group = val_work.loc[selector]

                raw_errs = pd.to_numeric(group.get(err_col), errors="coerce").to_numpy(dtype=float)
                raw_errs = raw_errs[np.isfinite(raw_errs)]
                errs = raw_errs[(raw_errs >= relerr_min) & (raw_errs <= relerr_max)]
                if len(errs) == 0:
                    continue

                outlier_mask = _iqr_outlier_mask(errs, iqr_factor)
                clean = errs[~outlier_mask]
                outliers = errs[outlier_mask]
                all_hist_payload.append(
                    {
                        "sector": tuple(int(x) for x in sector_key),
                        "clean": clean,
                        "outliers": outliers,
                    }
                )

            n_hist = len(all_hist_payload)
            if n_hist > 0:
                n_cols = min(10, max(1, int(np.ceil(np.sqrt(n_hist)))))
                n_rows = int(np.ceil(n_hist / n_cols))
                fig, axes = plt.subplots(
                    n_rows, n_cols,
                    figsize=(2.8 * n_cols, 2.2 * n_rows),
                    squeeze=False,
                )
                for idx, payload in enumerate(all_hist_payload):
                    ax = axes[idx // n_cols, idx % n_cols]
                    sector_key = payload["sector"]
                    clean = np.asarray(payload["clean"], dtype=float)
                    outliers = np.asarray(payload["outliers"], dtype=float)

                    if len(clean) > 0:
                        _, _, patches = ax.hist(
                            clean, bins=18, alpha=0.75, color="#4C78A8",
                            edgecolor="white", rasterized=True,
                        )
                        for patch in patches:
                            patch.set_rasterized(True)
                    if len(outliers) > 0:
                        _, _, patches = ax.hist(
                            outliers, bins=10, alpha=0.55, color="#E45756",
                            edgecolor="white", rasterized=True,
                        )
                        for patch in patches:
                            patch.set_rasterized(True)
                    for q in quantiles:
                        pval = _safe_percentile(clean, q * 100.0)
                        if np.isfinite(pval):
                            line = ax.axvline(
                                pval, linestyle="--", linewidth=0.7, color="#333333",
                                alpha=0.8,
                            )
                            line.set_rasterized(True)

                    ax.set_xlim(relerr_min, relerr_max)
                    ax.tick_params(labelsize=5)
                    bin_lines = _format_sector_bin_lines(dim_order, sector_key, all_edges, parts_per_line=2)
                    title_lines = [f"{pname} | idx=" + "_".join(str(x) for x in sector_key)]
                    title_lines.extend(bin_lines)
                    ax.set_title("\n".join(title_lines), fontsize=5.5)

                for idx in range(n_hist, n_rows * n_cols):
                    axes[idx // n_cols, idx % n_cols].axis("off")

                fig.suptitle(
                    f"Relative-error histograms by multidimensional parameter+events sectors: {pname}",
                    fontsize=10,
                )
                fig.tight_layout()
                fig.savefig(PLOTS_DIR / f"error_hist_by_events_{pname}.png", dpi=220)
                plt.close(fig)
                log.info(
                    "Histogram mosaic panels for %s (non-empty sectors): %d",
                    pname,
                    n_hist,
                )
            else:
                log.info("No non-empty sector histograms to plot for %s.", pname)

    # ── 2. LUT heatmap: uncertainty vs (param, n_events) ─────────────
    if not lut_df.empty:
        for pname in plot_pnames:
            est_centre = f"est_{pname}_centre"
            ne_centre = "n_events_centre"
            sig_col = f"sigma_{pname}_std"
            sig_label = f"σ_{pname} std [%]"
            if sig_col not in lut_df.columns:
                sig_col = f"sigma_{pname}_p68"
                sig_label = f"σ_{pname} p68 [%]"

            if not all(c in lut_df.columns for c in [est_centre, ne_centre, sig_col]):
                continue

            x = pd.to_numeric(lut_df[est_centre], errors="coerce")
            y = pd.to_numeric(lut_df[ne_centre], errors="coerce")
            z = pd.to_numeric(lut_df[sig_col], errors="coerce")
            m = x.notna() & y.notna() & z.notna()
            if m.sum() < 3:
                continue

            fig, ax = plt.subplots(figsize=(7, 5.5))
            sc = ax.scatter(x[m], y[m], c=z[m], cmap="YlOrRd", s=80,
                            alpha=0.9, edgecolors="black", linewidths=0.4)
            fig.colorbar(sc, ax=ax, label=sig_label, shrink=0.85)
            ax.set_xlabel(f"Estimated {pname}")
            ax.set_ylabel("Number of events")
            ax.set_title(f"Uncertainty LUT: {sig_label}")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"lut_heatmap_{pname}.png")
            plt.close(fig)

    # ── 3. Uncertainty vs n_events (show parameter-set dependence) ───
    ne_centre = "n_events_centre"
    for pname in plot_pnames:
        cols_q = [f"sigma_{pname}_std"] + [f"sigma_{pname}_p{int(q*100)}" for q in quantiles]
        available = [c for c in cols_q if c in lut_df.columns]

        fig, ax = plt.subplots(figsize=(7, 5))
        drew_from_lut = False
        guide_x: list[float] = []
        guide_y: list[float] = []

        # Try using the LUT first (median over parameter-bin cells at each events bin),
        # while also showing all individual LUT cells as a cloud.
        if not lut_df.empty and ne_centre in lut_df.columns and available:
            sig_ref_col = f"sigma_{pname}_std"
            if sig_ref_col not in lut_df.columns:
                sig_ref_col = f"sigma_{pname}_p68"
            if sig_ref_col not in lut_df.columns:
                sig_ref_col = available[min(len(available) // 2, len(available) - 1)]

            ne_vals = pd.to_numeric(lut_df[ne_centre], errors="coerce")
            sig_ref = pd.to_numeric(lut_df[sig_ref_col], errors="coerce")
            m_cells = ne_vals.notna() & sig_ref.notna() & (ne_vals > 0)
            if m_cells.sum() > 0:
                ax.scatter(
                    ne_vals[m_cells], sig_ref[m_cells],
                    s=28, alpha=0.28, color="#808080",
                    label="LUT cells (parameter-bin combos)",
                    zorder=1,
                )

            colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2"]
            for i, q_col in enumerate(available):
                xs: list[float] = []
                ys: list[float] = []
                for ne_val in sorted(ne_vals.dropna().unique()):
                    subset = lut_df.loc[np.isclose(ne_vals, ne_val, atol=1e-6), q_col]
                    vals = pd.to_numeric(subset, errors="coerce").dropna().to_numpy(dtype=float)
                    if len(vals) == 0:
                        continue
                    med = _safe_median(vals)
                    if np.isfinite(med):
                        xs.append(float(ne_val))
                        ys.append(med)
                if len(xs) >= 2:
                    q_label = "std" if q_col.endswith("_std") else q_col.split("_")[-1]
                    ax.plot(
                        xs, ys, "o-",
                        color=colors[i % len(colors)],
                        markersize=5, linewidth=1.6,
                        label=f"{q_label} median",
                        zorder=3,
                    )
                    if q_col == sig_ref_col:
                        guide_x.extend(xs)
                        guide_y.extend(ys)
                    drew_from_lut = True

        if not drew_from_lut:
            plt.close(fig)
            continue

        # 1/sqrt(N) guide scaled to the median uncertainty trend.
        gx = np.asarray(guide_x, dtype=float)
        gy = np.asarray(guide_y, dtype=float)
        finite = np.isfinite(gx) & np.isfinite(gy) & (gx > 0) & (gy >= 0)
        if finite.sum() >= 2:
            med_sigma = _safe_median(gy[finite])
            med_ne = _safe_median(gx[finite])
            if med_sigma > 0 and med_ne > 0:
                scale = med_sigma * np.sqrt(med_ne)
                x_guide = np.linspace(float(np.nanmin(gx[finite])), float(np.nanmax(gx[finite])), 200)
                ax.plot(
                    x_guide, scale / np.sqrt(x_guide), "k--",
                    linewidth=1, alpha=0.55, label=r"$\propto 1/\sqrt{N}$",
                    zorder=2,
                )

        ax.set_xlabel("Number of events")
        ax.set_ylabel(f"σ_{pname} [%]")
        ax.set_title(f"Uncertainty vs events: {pname}")
        ax.set_xscale("log")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"uncertainty_vs_events_{pname}.png")
        plt.close(fig)

    # ── 4. Ellipse-plane plot: show simulated (true) pairs, dictionary, and
    #      uncertainty ellipses centred on estimated pairs. (Simple, configurable)
    if ellipse_cfg is None:
        ellipse_cfg = {"params": None, "n_points": 0, "quantile": 0.68, "show_dictionary": True}

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

                fig, ax = plt.subplots(figsize=(7, 6))
                ax.set_xlabel(x_p)
                ax.set_ylabel(y_p)
                ax.set_title(f"Validation: {y_p} vs {x_p} — ellipses (p{int(ellipse_cfg.get('quantile', 0.68)*100)})")

                # Plot dictionary entries if requested and available
                if ellipse_cfg.get("show_dictionary") and dict_df is not None and {x_p, y_p}.issubset(set(dict_df.columns)):
                    ddf = dict_df.dropna(subset=[x_p, y_p])
                    n_dict_plot = min(len(ddf), ellipse_cfg.get("n_points", 10))
                    if len(ddf) > 0 and n_dict_plot > 0:
                        # sample evenly for reproducibility
                        if len(ddf) <= n_dict_plot:
                            dsel = ddf
                        else:
                            didx = np.linspace(0, len(ddf) - 1, n_dict_plot, dtype=int)
                            dsel = ddf.iloc[didx]
                        ax.scatter(dsel[x_p], dsel[y_p], marker="^", color="#2ca02c", s=60, label="Dictionary samples")

                # Plot selected points: ellipses centred at estimates and markers for true values
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
                                      edgecolor="#1f77b4", facecolor="none", alpha=0.6, linewidth=1.2)
                        ax.add_patch(ell)
                        ax.scatter([est_x], [est_y], marker="x", color="#1f77b4", s=40, label="Estimate" if _ == sel.index[0] else "")
                    else:
                        ax.scatter([est_x], [est_y], marker="x", color="#1f77b4", s=40, label="Estimate" if _ == sel.index[0] else "")

                    if true_x is not None and true_y is not None:
                        ax.scatter([true_x], [true_y], marker="o", color="#d62728", s=30, label="True" if _ == sel.index[0] else "")
                        # draw line estimate → true
                        if np.isfinite(true_x) and np.isfinite(true_y):
                            ax.plot([est_x, true_x], [est_y, true_y], color="#888888", linewidth=0.7, alpha=0.7)

                ax.legend(fontsize=8)
                fig.tight_layout()
                outname = PLOTS_DIR / f"lut_ellipse_{x_p}_{y_p}.png"
                fig.savefig(outname)
                plt.close(fig)
                log.info("Wrote ellipse scatter plot: %s", outname)


if __name__ == "__main__":
    raise SystemExit(main())
