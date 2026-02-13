#!/usr/bin/env python3

"""STEP 3.3 — Correction by inference on synthetic dataset.

This step applies the dictionary-based inference from STEP 2 to the synthetic
dataset built in STEP 3.2, then propagates uncertainties using the LUT created
in STEP 2.3.

Main actions
------------
1. Run inverse estimation (`estimate_parameters`) on STEP 3.2 synthetic table.
2. Interpolate uncertainty per row from the STEP 2.3 LUT.
3. Build a corrected/diagnostic table with true vs estimated parameters.
4. Produce diagnostic plots:
   - Flux true vs estimated (y=x).
   - Efficiency true vs estimated (y=x).
   - Time-series comparison with uncertainty bands.
   - Simulated and estimated flux overlaid with global rate.
   - A 2x2 summary plot combining all above.

Output
------
OUTPUTS/FILES/corrected_by_inference.csv
    Synthetic rows enriched with estimated parameters and uncertainties.
OUTPUTS/FILES/correction_summary.json
    Summary statistics and run configuration.
OUTPUTS/PLOTS/correction_overview_2x2.png
OUTPUTS/PLOTS/synthetic_time_series_overview_with_estimated.png
OUTPUTS/PLOTS/flux_recovery_vs_global_rate.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
SYNTHETIC_DIR = STEP_DIR.parent
PIPELINE_DIR = SYNTHETIC_DIR.parent
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"

DEFAULT_SYNTHETIC_DATASET = (
    SYNTHETIC_DIR / "STEP_3_2_SYNTHETIC_TIME_SERIES" / "OUTPUTS" / "FILES" / "synthetic_dataset.csv"
)
DEFAULT_TIME_SERIES = (
    SYNTHETIC_DIR / "STEP_3_1_TIME_SERIES_CREATION" / "OUTPUTS" / "FILES" / "time_series.csv"
)
DEFAULT_COMPLETE_CURVE = (
    SYNTHETIC_DIR / "STEP_3_1_TIME_SERIES_CREATION" / "OUTPUTS" / "FILES" / "complete_curve_time_series.csv"
)
DEFAULT_DICTIONARY = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY" / "OUTPUTS" / "FILES" / "dictionary.csv"
)
DEFAULT_DATASET_TEMPLATE = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY" / "OUTPUTS" / "FILES" / "dataset.csv"
)
DEFAULT_LUT = (
    PIPELINE_DIR / "STEP_2_INFERENCE" / "STEP_2_3_UNCERTAINTY" / "OUTPUTS" / "FILES" / "uncertainty_lut.csv"
)
DEFAULT_LUT_META = (
    PIPELINE_DIR / "STEP_2_INFERENCE" / "STEP_2_3_UNCERTAINTY" / "OUTPUTS" / "FILES" / "uncertainty_lut_meta.json"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Import estimation function from STEP 2 module.
INFERENCE_DIR = PIPELINE_DIR / "STEP_2_INFERENCE"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
from estimate_parameters import estimate_parameters  # noqa: E402

logging.basicConfig(
    format="[%(levelname)s] STEP_3.3 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_3.3")


def _load_config(path: Path) -> dict:
    """Load JSON config if it exists."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _safe_float(value: object, default: float) -> float:
    """Convert value to float with fallback."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isfinite(out):
        return out
    return float(default)


def _safe_bool(value: object, default: bool) -> bool:
    """Convert common truthy/falsy representations to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _resolve_input_path(path_like: str | Path) -> Path:
    """Resolve path relative to pipeline when not absolute."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    candidate_pipeline = PIPELINE_DIR / p
    if candidate_pipeline.exists():
        return candidate_pipeline
    candidate_step = STEP_DIR / p
    if candidate_step.exists():
        return candidate_step
    return candidate_pipeline


def _choose_eff_column(df: pd.DataFrame, preferred: str) -> str:
    """Select an efficiency column from dataframe with fallback candidates."""
    if preferred in df.columns:
        return preferred
    for candidate in (
        "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
        "eff_empirical_1", "eff_empirical_2", "eff_empirical_3", "eff_empirical_4",
    ):
        if candidate in df.columns:
            return candidate
    raise KeyError("No efficiency column found in dataframe.")


def _load_lut(lut_path: Path) -> pd.DataFrame:
    """Load LUT CSV allowing comment-prefixed metadata header."""
    return pd.read_csv(lut_path, comment="#", low_memory=False)


def _lut_param_names(
    lut_df: pd.DataFrame,
    lut_meta_path: Path | None = None,
) -> list[str]:
    """Extract LUT parameter names from metadata JSON or LUT columns."""
    if lut_meta_path is not None and lut_meta_path.exists():
        try:
            meta = json.loads(lut_meta_path.read_text(encoding="utf-8"))
            params = meta.get("param_names", [])
            if isinstance(params, list):
                cleaned = [str(p) for p in params if str(p)]
                if cleaned:
                    return cleaned
        except Exception:
            pass

    params: list[str] = []
    for c in lut_df.columns:
        if not c.startswith("sigma_"):
            continue
        if "_p" in c:
            pname = c[len("sigma_"):].split("_p", 1)[0]
        elif c.endswith("_std"):
            pname = c[len("sigma_"):-len("_std")]
        else:
            continue
        if pname and pname not in params:
            params.append(pname)
    return params


def _interpolate_uncertainties(
    query_df: pd.DataFrame,
    lut_df: pd.DataFrame,
    param_names: list[str],
    quantile: float,
) -> pd.DataFrame:
    """Nearest-centre LUT interpolation with finite-value fallback per parameter.

    For each parameter and query row, select the nearest LUT row with finite
    sigma value for that parameter. If no finite sigma exists, return NaN.
    """
    if lut_df.empty or query_df.empty:
        return pd.DataFrame(index=query_df.index)

    q_label = str(int(round(float(quantile) * 100.0)))
    centre_cols = [c for c in lut_df.columns if c.endswith("_centre")]
    if not centre_cols:
        return pd.DataFrame(index=query_df.index)

    lut_centres_df = lut_df[centre_cols].apply(pd.to_numeric, errors="coerce")
    lut_centres = lut_centres_df.to_numpy(dtype=float)
    valid_centres = np.all(np.isfinite(lut_centres), axis=1)
    if not np.any(valid_centres):
        return pd.DataFrame(index=query_df.index)

    # Dimension scales for normalized distance.
    mins = np.nanmin(lut_centres[valid_centres], axis=0)
    maxs = np.nanmax(lut_centres[valid_centres], axis=0)
    ranges = maxs - mins
    ranges[~np.isfinite(ranges) | (ranges <= 0.0)] = 1.0
    dim_fallbacks = np.nanmedian(lut_centres[valid_centres], axis=0)

    n_rows = len(query_df)
    n_dims = len(centre_cols)
    query_vals = np.zeros((n_rows, n_dims), dtype=float)
    for j, cc in enumerate(centre_cols):
        dim = cc.replace("_centre", "")
        if dim in query_df.columns:
            qv = pd.to_numeric(query_df[dim], errors="coerce").to_numpy(dtype=float)
        elif dim == "n_events":
            if "n_events" in query_df.columns:
                qv = pd.to_numeric(query_df["n_events"], errors="coerce").to_numpy(dtype=float)
            elif "true_n_events" in query_df.columns:
                qv = pd.to_numeric(query_df["true_n_events"], errors="coerce").to_numpy(dtype=float)
            else:
                qv = np.full(n_rows, np.nan, dtype=float)
        else:
            qv = np.full(n_rows, np.nan, dtype=float)
        qv = np.where(np.isfinite(qv), qv, dim_fallbacks[j])
        query_vals[:, j] = qv

    d = (lut_centres[np.newaxis, :, :] - query_vals[:, np.newaxis, :]) / ranges[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    dist = np.where(valid_centres[np.newaxis, :], dist, np.inf)

    out = pd.DataFrame(index=query_df.index)
    for pname in param_names:
        pref_col = f"sigma_{pname}_p{q_label}"
        sigma_col = pref_col if pref_col in lut_df.columns else None
        if sigma_col is None:
            alt = f"sigma_{pname}_std"
            sigma_col = alt if alt in lut_df.columns else None
        if sigma_col is None:
            out[f"unc_{pname}_pct_raw"] = np.nan
            out[f"unc_{pname}_pct"] = np.nan
            continue

        sigma_vals = pd.to_numeric(lut_df[sigma_col], errors="coerce").to_numpy(dtype=float)
        valid_sigma = valid_centres & np.isfinite(sigma_vals)
        if not np.any(valid_sigma):
            out[f"unc_{pname}_pct_raw"] = np.nan
            out[f"unc_{pname}_pct"] = np.nan
            continue

        masked_dist = np.where(valid_sigma[np.newaxis, :], dist, np.inf)
        nearest_idx = np.argmin(masked_dist, axis=1)
        nearest_dist = masked_dist[np.arange(n_rows), nearest_idx]
        raw = sigma_vals[nearest_idx]
        raw = np.where(np.isfinite(nearest_dist), raw, np.nan)

        # Fallback to median finite sigma if a row has no finite nearest.
        sigma_median = float(np.nanmedian(sigma_vals[valid_sigma]))
        raw = np.where(np.isfinite(raw), raw, sigma_median)
        out[f"unc_{pname}_pct_raw"] = raw
        out[f"unc_{pname}_pct"] = np.abs(raw)
    return out


def _load_step32_module():
    """Dynamically load STEP 3.2 module to reuse weighting helpers."""
    step32_path = SYNTHETIC_DIR / "STEP_3_2_SYNTHETIC_TIME_SERIES" / "synthetic_time_series.py"
    if not step32_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("step32_synthetic_time_series", str(step32_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        log.warning("Could not import STEP 3.2 module: %s", exc)
        return None


def _compute_density_center_series(
    *,
    config: dict,
    time_df: pd.DataFrame,
    flux_col: str,
    eff_pref: str,
    dictionary_path: Path,
    dataset_template_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Recompute STEP 3.2 density-modulated weighted center for diagnostics."""
    step32 = _load_step32_module()
    if step32 is None or time_df.empty:
        return None, None, None

    cfg_32 = config.get("step_3_2", {})
    basis_source = str(cfg_32.get("basis_source", "dataset")).strip().lower()

    if not dictionary_path.exists():
        return None, None, None
    dictionary_df = pd.read_csv(dictionary_path, low_memory=False)

    template_df = pd.DataFrame()
    if dataset_template_path.exists():
        template_df = pd.read_csv(dataset_template_path, low_memory=False)

    if basis_source == "dictionary":
        basis_input_df = dictionary_df
    else:
        basis_input_df = template_df
    if basis_input_df.empty:
        return None, None, None

    try:
        eff_col_time = step32._choose_eff_column(time_df, eff_pref)
        eff_col_basis = step32._choose_eff_column(basis_input_df, eff_pref)
    except Exception:
        return None, None, None
    if flux_col not in time_df.columns or flux_col not in basis_input_df.columns:
        return None, None, None

    time_events_col = str(cfg_32.get("time_n_events_column", "n_events"))
    basis_events_col = str(cfg_32.get("basis_n_events_column", "n_events"))
    basis_events_tol_pct = _safe_float(
        cfg_32.get("basis_n_events_tolerance_pct", cfg_32.get("basis_n_events_tolerance", 30.0)),
        30.0,
    )
    basis_min_rows = max(1, int(_safe_float(cfg_32.get("basis_min_rows", 1), 1)))
    basis_parameter_set_col_cfg = cfg_32.get("basis_parameter_set_column", None)

    basis_flux_all = pd.to_numeric(basis_input_df.get(flux_col), errors="coerce").to_numpy(dtype=float)
    basis_eff_all = pd.to_numeric(basis_input_df.get(eff_col_basis), errors="coerce").to_numpy(dtype=float)
    basis_events_all = None
    if basis_events_col in basis_input_df.columns:
        basis_events_all = pd.to_numeric(basis_input_df[basis_events_col], errors="coerce").to_numpy(dtype=float)

    target_flux = pd.to_numeric(time_df.get(flux_col), errors="coerce").to_numpy(dtype=float)
    target_eff = pd.to_numeric(time_df.get(eff_col_time), errors="coerce").to_numpy(dtype=float)
    target_events = (
        pd.to_numeric(time_df.get(time_events_col), errors="coerce").to_numpy(dtype=float)
        if time_events_col in time_df.columns
        else None
    )
    if len(target_flux) == 0 or len(target_eff) == 0:
        return None, None, eff_col_time

    valid_basis = np.isfinite(basis_flux_all) & np.isfinite(basis_eff_all)
    if not np.any(valid_basis):
        return None, None, eff_col_time
    dictionary_work = basis_input_df.loc[valid_basis].reset_index(drop=True)
    basis_flux = basis_flux_all[valid_basis]
    basis_eff = basis_eff_all[valid_basis]
    basis_events = None if basis_events_all is None else basis_events_all[valid_basis]

    basis_parameter_set_col = step32._select_parameter_set_column(dictionary_work, basis_parameter_set_col_cfg)
    if basis_parameter_set_col is None:
        parameter_set_values = np.asarray([f"row_{i}" for i in range(len(dictionary_work))], dtype=object)
    else:
        parameter_set_values = dictionary_work[basis_parameter_set_col].astype(str).to_numpy(dtype=object)

    one_per_set_mask, _ = step32._build_one_per_parameter_set_mask(
        parameter_set_values=parameter_set_values,
        basis_events=basis_events,
        target_events=target_events,
        basis_flux=basis_flux,
        basis_eff=basis_eff,
        target_flux=target_flux,
        target_eff=target_eff,
    )
    event_mask_extra, _ = step32._build_event_mask(
        basis_events=basis_events,
        target_events=target_events,
        tolerance_pct=basis_events_tol_pct,
        min_rows=basis_min_rows,
    )
    event_mask = one_per_set_mask if event_mask_extra is None else (one_per_set_mask & event_mask_extra)

    flux_span = max(float(np.nanmax(basis_flux) - np.nanmin(basis_flux)), 1e-9)
    eff_span = max(float(np.nanmax(basis_eff) - np.nanmin(basis_eff)), 1e-9)
    sigma_flux = _safe_float(
        cfg_32.get("distance_sigma_flux_abs"),
        _safe_float(cfg_32.get("distance_sigma_flux_fraction", 0.10), 0.10) * flux_span,
    )
    sigma_eff = _safe_float(
        cfg_32.get("distance_sigma_eff_abs"),
        _safe_float(cfg_32.get("distance_sigma_eff_fraction", 0.10), 0.10) * eff_span,
    )
    method = str(cfg_32.get("weighting_method", "gaussian"))
    top_k_raw = cfg_32.get("top_k", None)
    top_k = None if top_k_raw in (None, "", 0) else max(1, int(_safe_float(top_k_raw, 8)))
    distance_hardness = _safe_float(cfg_32.get("distance_hardness", 1.0), 1.0)

    density_enabled = _safe_bool(cfg_32.get("density_correction_enabled", True), True)
    density_scaling = None
    if density_enabled:
        density_k = max(1, int(_safe_float(cfg_32.get("density_correction_k_neighbors", 10), 10)))
        density_exp = _safe_float(cfg_32.get("density_correction_exponent", 1.0), 1.0)
        density_clip_min = _safe_float(cfg_32.get("density_correction_clip_min", 0.25), 0.25)
        density_clip_max = _safe_float(cfg_32.get("density_correction_clip_max", 4.0), 4.0)
        if density_clip_max < density_clip_min:
            density_clip_max = density_clip_min
        density_scaling, _ = step32._compute_inverse_density_scaling(
            basis_flux=basis_flux,
            basis_eff=basis_eff,
            k_neighbors=density_k,
            exponent=density_exp,
            clip_min=density_clip_min,
            clip_max=density_clip_max,
        )

    weights = step32._build_weights(
        dict_flux=basis_flux,
        dict_eff=basis_eff,
        target_flux=target_flux,
        target_eff=target_eff,
        method=method,
        sigma_flux=sigma_flux,
        sigma_eff=sigma_eff,
        top_k=top_k,
        distance_hardness=distance_hardness,
        density_scaling=density_scaling,
        event_mask=event_mask,
    )
    centers = step32._weighted_numeric_columns(
        weights=weights,
        dict_df=dictionary_work,
        columns=[flux_col, eff_col_basis],
    )
    c_flux = np.asarray(centers.get(flux_col), dtype=float) if flux_col in centers else None
    c_eff = np.asarray(centers.get(eff_col_basis), dtype=float) if eff_col_basis in centers else None
    return c_flux, c_eff, eff_col_time


def _time_axis(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return numeric time axis values and axis label."""
    if "elapsed_hours" in df.columns:
        x = pd.to_numeric(df["elapsed_hours"], errors="coerce").to_numpy(dtype=float)
        return x, "Elapsed time [hours]"
    if "file_index" in df.columns:
        x = pd.to_numeric(df["file_index"], errors="coerce").to_numpy(dtype=float)
        return x, "File index"
    return np.arange(len(df), dtype=float), "Index"


def _apply_mean_striped_background(ax: plt.Axes, y_vals: np.ndarray) -> None:
    """Apply stripes at 1%-of-mean increments, uniformly across the y-axis."""
    y_min, y_max = ax.get_ylim()
    if not (np.isfinite(y_min) and np.isfinite(y_max)):
        return
    span = y_max - y_min
    if span <= 0.0:
        return

    y_arr = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(y_arr)
    if not np.any(valid):
        return
    mean_val = float(np.mean(y_arr[valid]))

    band = abs(mean_val) * 0.01
    if not np.isfinite(band) or band <= 0.0:
        band = span * 0.01
    if band <= 0.0:
        return

    ax.set_facecolor("#FFFFFF")
    idx = int(np.floor((y_min - mean_val) / band))
    y0 = mean_val + idx * band
    while y0 < y_max:
        y1 = y0 + band
        lo = max(y0, y_min)
        hi = min(y1, y_max)
        color = "#FFFFFF" if (idx % 2 == 0) else "#D8DDE4"
        if hi > lo:
            ax.axhspan(lo, hi, facecolor=color, alpha=1.0, linewidth=0.0, zorder=0)
        y0 = y1
        idx += 1
    ax.set_ylim(y_min, y_max)


def _plot_series_panel(
    ax: plt.Axes,
    x: np.ndarray,
    true_vals: np.ndarray,
    est_vals: np.ndarray,
    unc_abs: np.ndarray | None,
    ylabel: str,
    title: str,
) -> None:
    """Plot one time-series comparison panel with optional uncertainty band."""
    m = np.isfinite(x) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if not np.any(m):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        return

    xo = x[m]
    yt = true_vals[m]
    ye = est_vals[m]
    order = np.argsort(xo)
    xo = xo[order]
    yt = yt[order]
    ye = ye[order]

    if unc_abs is not None and len(unc_abs) == len(x):
        yu = np.asarray(unc_abs, dtype=float)[m][order]
        yu = np.where(np.isfinite(yu), np.abs(yu), np.nan)
        valid_band = np.isfinite(yu)
        if np.any(valid_band):
            ax.fill_between(
                xo[valid_band],
                ye[valid_band] - yu[valid_band],
                ye[valid_band] + yu[valid_band],
                color="#FF7F0E",
                alpha=0.20,
                linewidth=0.0,
                label="Estimated ± uncertainty",
            )

    ax.scatter(xo, yt, s=18, facecolors="white", edgecolors="#1F77B4", linewidths=0.8, label="Simulated (true)")
    ax.scatter(xo, ye, s=18, color="#D62728", alpha=0.9, linewidths=0.0, label="Estimated")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _apply_mean_striped_background(ax, np.concatenate([yt, ye]))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _plot_diag_panel(
    ax: plt.Axes,
    true_vals: np.ndarray,
    est_vals: np.ndarray,
    unc_abs: np.ndarray | None,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot true-vs-estimated diagonal panel with optional vertical uncertainty."""
    m = np.isfinite(true_vals) & np.isfinite(est_vals)
    if not np.any(m):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        return

    xt = true_vals[m]
    ye = est_vals[m]
    if unc_abs is not None and len(unc_abs) == len(true_vals):
        yu = np.asarray(unc_abs, dtype=float)[m]
        yu = np.where(np.isfinite(yu), np.abs(yu), np.nan)
        valid_yu = np.isfinite(yu)
        if np.any(valid_yu):
            ax.errorbar(
                xt[valid_yu],
                ye[valid_yu],
                yerr=yu[valid_yu],
                fmt="none",
                ecolor="#D62728",
                alpha=0.20,
                elinewidth=0.8,
                capsize=0,
                zorder=1,
            )

    ax.scatter(xt, ye, s=22, color="#D62728", alpha=0.85, linewidths=0.0, zorder=2)
    lo = float(np.nanmin(np.concatenate([xt, ye])))
    hi = float(np.nanmax(np.concatenate([xt, ye])))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1e-6)
    else:
        pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    _apply_mean_striped_background(ax, np.concatenate([xt, ye]))
    ax.grid(True, alpha=0.25)


def _plot_step32_style_overlay(
    *,
    complete_df: pd.DataFrame | None,
    time_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    flux_col: str,
    eff_col_time: str,
    est_flux_col: str,
    est_eff_col: str,
    flux_unc_abs_col: str | None,
    eff_unc_abs_col: str | None,
    center_flux: np.ndarray | None,
    center_eff: np.ndarray | None,
    path: Path,
) -> None:
    """Plot complete+discretized+density-center+estimated without global rate."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)

    # Discretized STEP 3.1 trajectory
    x_disc, x_label = _time_axis(time_df)
    y_flux_disc = pd.to_numeric(time_df.get(flux_col), errors="coerce").to_numpy(dtype=float)
    y_eff_disc = pd.to_numeric(time_df.get(eff_col_time), errors="coerce").to_numpy(dtype=float)
    flux_stripe_vals = [y_flux_disc]
    eff_stripe_vals = [y_eff_disc]

    # Complete trajectory
    if complete_df is not None and not complete_df.empty:
        x_comp, _ = _time_axis(complete_df)
        y_flux_comp = pd.to_numeric(complete_df.get(flux_col), errors="coerce").to_numpy(dtype=float)
        y_eff_comp = pd.to_numeric(complete_df.get(eff_col_time), errors="coerce").to_numpy(dtype=float)
        flux_stripe_vals.append(y_flux_comp)
        eff_stripe_vals.append(y_eff_comp)
        m0c = np.isfinite(x_comp) & np.isfinite(y_flux_comp)
        m1c = np.isfinite(x_comp) & np.isfinite(y_eff_comp)
        if np.any(m0c):
            axes[0].scatter(x_comp[m0c], y_flux_comp[m0c], s=7, color="#1F77B4", alpha=0.55, linewidths=0.0, label="Complete")
        if np.any(m1c):
            axes[1].scatter(x_comp[m1c], y_eff_comp[m1c], s=7, color="#FF7F0E", alpha=0.55, linewidths=0.0, label="Complete")

    m0d = np.isfinite(x_disc) & np.isfinite(y_flux_disc)
    m1d = np.isfinite(x_disc) & np.isfinite(y_eff_disc)
    if np.any(m0d):
        axes[0].scatter(x_disc[m0d], y_flux_disc[m0d], s=18, facecolors="white", edgecolors="#1F77B4", linewidths=0.8, label="Discretized")
    if np.any(m1d):
        axes[1].scatter(x_disc[m1d], y_eff_disc[m1d], s=18, facecolors="white", edgecolors="#FF7F0E", linewidths=0.8, label="Discretized")

    # Density-modulated weighted center (diagnostic from STEP 3.2 logic)
    if center_flux is not None and len(center_flux) == len(x_disc):
        c_flux = np.asarray(center_flux, dtype=float)
        flux_stripe_vals.append(c_flux)
        mc = np.isfinite(x_disc) & np.isfinite(c_flux)
        if np.any(mc):
            axes[0].plot(
                x_disc[mc], c_flux[mc],
                color="#17BECF", linewidth=1.0, linestyle="-.", marker="s", markersize=2.8,
                markerfacecolor="#17BECF", markeredgewidth=0.0, alpha=0.9,
                label="Density-modulated center",
            )
    if center_eff is not None and len(center_eff) == len(x_disc):
        c_eff = np.asarray(center_eff, dtype=float)
        eff_stripe_vals.append(c_eff)
        mc = np.isfinite(x_disc) & np.isfinite(c_eff)
        if np.any(mc):
            axes[1].plot(
                x_disc[mc], c_eff[mc],
                color="#BCBD22", linewidth=1.0, linestyle="-.", marker="s", markersize=2.8,
                markerfacecolor="#BCBD22", markeredgewidth=0.0, alpha=0.9,
                label="Density-modulated center",
            )

    # Estimated series (+ uncertainty)
    x_est, _ = _time_axis(merged_df)
    y_flux_est = pd.to_numeric(merged_df.get(est_flux_col), errors="coerce").to_numpy(dtype=float)
    y_eff_est = pd.to_numeric(merged_df.get(est_eff_col), errors="coerce").to_numpy(dtype=float)
    flux_stripe_vals.append(y_flux_est)
    eff_stripe_vals.append(y_eff_est)
    u_flux = (
        pd.to_numeric(merged_df.get(flux_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if flux_unc_abs_col and flux_unc_abs_col in merged_df.columns
        else None
    )
    u_eff = (
        pd.to_numeric(merged_df.get(eff_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if eff_unc_abs_col and eff_unc_abs_col in merged_df.columns
        else None
    )

    m0e = np.isfinite(x_est) & np.isfinite(y_flux_est)
    if np.any(m0e):
        order = np.argsort(x_est[m0e])
        xe = x_est[m0e][order]
        ye = y_flux_est[m0e][order]
        axes[0].plot(
            xe, ye,
            color="#D62728", linewidth=1.0, linestyle="-", marker="o", markersize=3.0,
            markerfacecolor="#D62728", markeredgewidth=0.0, alpha=0.9,
            label="Estimated",
        )
        if u_flux is not None and len(u_flux) == len(x_est):
            ue = np.abs(u_flux[m0e][order])
            valid_u = np.isfinite(ue)
            if np.any(valid_u):
                axes[0].fill_between(
                    xe[valid_u], ye[valid_u] - ue[valid_u], ye[valid_u] + ue[valid_u],
                    color="#D62728", alpha=0.16, linewidth=0.0, label="Estimated ± uncertainty",
                )

    m1e = np.isfinite(x_est) & np.isfinite(y_eff_est)
    if np.any(m1e):
        order = np.argsort(x_est[m1e])
        xe = x_est[m1e][order]
        ye = y_eff_est[m1e][order]
        axes[1].plot(
            xe, ye,
            color="#8C564B", linewidth=1.0, linestyle="-", marker="o", markersize=3.0,
            markerfacecolor="#8C564B", markeredgewidth=0.0, alpha=0.9,
            label="Estimated",
        )
        if u_eff is not None and len(u_eff) == len(x_est):
            ue = np.abs(u_eff[m1e][order])
            valid_u = np.isfinite(ue)
            if np.any(valid_u):
                axes[1].fill_between(
                    xe[valid_u], ye[valid_u] - ue[valid_u], ye[valid_u] + ue[valid_u],
                    color="#8C564B", alpha=0.16, linewidth=0.0, label="Estimated ± uncertainty",
                )

    axes[0].set_ylabel("flux_cm2_min")
    axes[0].set_title("Flux: complete + discretized + density center + estimated")
    _apply_mean_striped_background(axes[0], np.concatenate(flux_stripe_vals))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_ylabel("eff")
    axes[1].set_xlabel(x_label)
    axes[1].set_title("Efficiency: complete + discretized + density center + estimated")
    _apply_mean_striped_background(axes[1], np.concatenate(eff_stripe_vals))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_flux_recovery_vs_global_rate(
    *,
    df: pd.DataFrame,
    complete_df: pd.DataFrame | None,
    flux_complete_col: str,
    eff_complete_col: str,
    flux_true_col: str,
    eff_true_col: str,
    flux_est_col: str,
    flux_unc_abs_col: str | None,
    global_rate_col: str,
    path: Path,
) -> None:
    """Plot a 4-panel story using complete curve as simulated reference when available."""
    def _apply_striped_background(ax: plt.Axes, y_vals: np.ndarray) -> None:
        """Apply stripes at 1%-of-mean increments, uniformly across the y-axis."""
        y_min, y_max = ax.get_ylim()
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            return
        span = y_max - y_min
        if span <= 0.0:
            return

        valid = np.isfinite(y_vals)
        if not np.any(valid):
            return
        mean_val = float(np.mean(y_vals[valid]))

        band = abs(mean_val) * 0.01
        if not np.isfinite(band) or band <= 0.0:
            band = span * 0.01
        if band <= 0.0:
            return

        ax.set_facecolor("#FFFFFF")
        idx = int(np.floor((y_min - mean_val) / band))
        y0 = mean_val + idx * band
        while y0 < y_max:
            y1 = y0 + band
            lo = max(y0, y_min)
            hi = min(y1, y_max)
            color = "#FFFFFF" if (idx % 2 == 0) else "#D8DDE4"
            if hi > lo:
                ax.axhspan(lo, hi, facecolor=color, alpha=1.0, linewidth=0.0, zorder=0)
            y0 = y1
            idx += 1
        ax.set_ylim(y_min, y_max)

    x, x_label = _time_axis(df)
    true_flux = pd.to_numeric(df.get(flux_true_col), errors="coerce").to_numpy(dtype=float)
    true_eff = pd.to_numeric(df.get(eff_true_col), errors="coerce").to_numpy(dtype=float)
    est_flux = pd.to_numeric(df.get(flux_est_col), errors="coerce").to_numpy(dtype=float)
    rate_vals = pd.to_numeric(df.get(global_rate_col), errors="coerce").to_numpy(dtype=float)
    unc_flux = (
        pd.to_numeric(df.get(flux_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if flux_unc_abs_col and flux_unc_abs_col in df.columns
        else None
    )

    # Prefer complete-curve references for simulated flux/efficiency.
    x_ref_flux = x
    y_ref_flux = true_flux
    x_ref_eff = x
    y_ref_eff = true_eff
    ref_flux_label = "Simulated flux (discretized)"
    ref_eff_label = "Simulated efficiency (discretized)"
    if complete_df is not None and not complete_df.empty:
        if flux_complete_col in complete_df.columns:
            xc, _ = _time_axis(complete_df)
            yc = pd.to_numeric(complete_df.get(flux_complete_col), errors="coerce").to_numpy(dtype=float)
            if len(xc) == len(yc):
                x_ref_flux = xc
                y_ref_flux = yc
                ref_flux_label = "Simulated flux (complete)"
        if eff_complete_col in complete_df.columns:
            xc, _ = _time_axis(complete_df)
            yc = pd.to_numeric(complete_df.get(eff_complete_col), errors="coerce").to_numpy(dtype=float)
            if len(xc) == len(yc):
                x_ref_eff = xc
                y_ref_eff = yc
                ref_eff_label = "Simulated efficiency (complete)"

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.6, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.15]},
    )
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, alpha=0.24)

    # 1) Simulated flux.
    m_true_flux = np.isfinite(x_ref_flux) & np.isfinite(y_ref_flux)
    if np.any(m_true_flux):
        order = np.argsort(x_ref_flux[m_true_flux])
        xs = x_ref_flux[m_true_flux][order]
        ys = y_ref_flux[m_true_flux][order]
        axes[0].plot(
            xs,
            ys,
            color="#1F77B4",
            linewidth=1.2,
            marker="o",
            markersize=3.1,
            markerfacecolor="white",
            markeredgewidth=0.7,
            alpha=0.92,
            label=ref_flux_label,
        )
    axes[0].set_ylabel("Sim. flux")
    _apply_striped_background(axes[0], y_ref_flux)
    axes[0].legend(loc="best", fontsize=8)

    # 2) Simulated efficiency.
    m_true_eff = np.isfinite(x_ref_eff) & np.isfinite(y_ref_eff)
    if np.any(m_true_eff):
        order = np.argsort(x_ref_eff[m_true_eff])
        xs = x_ref_eff[m_true_eff][order]
        ys = y_ref_eff[m_true_eff][order]
        axes[1].plot(
            xs,
            ys,
            color="#FF7F0E",
            linewidth=1.15,
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.65,
            alpha=0.90,
            label=ref_eff_label,
        )
    axes[1].set_ylabel("Sim. eff")
    _apply_striped_background(axes[1], y_ref_eff)
    axes[1].legend(loc="best", fontsize=8)

    # 3) Global rate.
    m_rate = np.isfinite(x) & np.isfinite(rate_vals)
    if np.any(m_rate):
        order = np.argsort(x[m_rate])
        xr = x[m_rate][order]
        yr = rate_vals[m_rate][order]
        axes[2].plot(
            xr,
            yr,
            color="#2E8B57",
            linewidth=2.4,
            alpha=0.46,
            solid_capstyle="round",
            label=f"Global rate ({global_rate_col})",
        )
    axes[2].set_ylabel("Global rate")
    _apply_striped_background(axes[2], rate_vals)
    axes[2].legend(loc="best", fontsize=8)

    # 4) Estimated reconstructed flux (+ uncertainty), with true flux reference.
    m_est_flux = np.isfinite(x) & np.isfinite(est_flux)
    if np.any(m_est_flux):
        order = np.argsort(x[m_est_flux])
        xe = x[m_est_flux][order]
        ye = est_flux[m_est_flux][order]
        axes[3].plot(
            xe,
            ye,
            color="#D62728",
            linewidth=1.3,
            marker="o",
            markersize=3.0,
            markerfacecolor="#D62728",
            markeredgewidth=0.0,
            alpha=0.88,
            label="Estimated reconstructed flux",
            zorder=3,
        )
        if unc_flux is not None and len(unc_flux) == len(x):
            ue = np.abs(np.asarray(unc_flux, dtype=float)[m_est_flux][order])
            valid_ue = np.isfinite(ue)
            if np.any(valid_ue):
                axes[3].fill_between(
                    xe[valid_ue],
                    ye[valid_ue] - ue[valid_ue],
                    ye[valid_ue] + ue[valid_ue],
                    color="#D62728",
                    alpha=0.16,
                    linewidth=0.0,
                    label="Estimated ± uncertainty",
                    zorder=2,
                )
    if np.any(m_true_flux):
        order = np.argsort(x_ref_flux[m_true_flux])
        xs = x_ref_flux[m_true_flux][order]
        ys = y_ref_flux[m_true_flux][order]
        axes[3].plot(
            xs,
            ys,
            color="#1F77B4",
            linewidth=1.0,
            linestyle="--",
            alpha=0.60,
            label=f"{ref_flux_label} reference",
            zorder=1,
        )
    axes[3].set_ylabel("Estimated flux")
    axes[3].set_xlabel(x_label)
    _apply_striped_background(axes[3], est_flux)
    axes[3].legend(loc="best", fontsize=8)

    fig.suptitle(
        "Flux-recovery story: simulated flux/efficiency -> global-rate response -> reconstructed flux",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_plots(
    df: pd.DataFrame,
    *,
    flux_true_col: str,
    flux_est_col: str,
    flux_unc_abs_col: str | None,
    eff_true_col: str,
    eff_est_col: str,
    eff_unc_abs_col: str | None,
) -> None:
    """Generate requested 2x2 diagnostic plot (transposed layout)."""
    x, x_label = _time_axis(df)
    true_flux = pd.to_numeric(df.get(flux_true_col), errors="coerce").to_numpy(dtype=float)
    est_flux = pd.to_numeric(df.get(flux_est_col), errors="coerce").to_numpy(dtype=float)
    true_eff = pd.to_numeric(df.get(eff_true_col), errors="coerce").to_numpy(dtype=float)
    est_eff = pd.to_numeric(df.get(eff_est_col), errors="coerce").to_numpy(dtype=float)
    unc_flux_abs = (
        pd.to_numeric(df.get(flux_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if flux_unc_abs_col and flux_unc_abs_col in df.columns
        else None
    )
    unc_eff_abs = (
        pd.to_numeric(df.get(eff_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if eff_unc_abs_col and eff_unc_abs_col in df.columns
        else None
    )

    # Main 2x2 diagnostic plot (transposed w.r.t. previous layout).
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    _plot_series_panel(
        axes[0, 0],
        x=x,
        true_vals=true_flux,
        est_vals=est_flux,
        unc_abs=unc_flux_abs,
        ylabel="flux_cm2_min",
        title="Flux time series",
    )
    _plot_diag_panel(
        axes[0, 1],
        true_vals=true_flux,
        est_vals=est_flux,
        unc_abs=unc_flux_abs,
        xlabel="Simulated flux",
        ylabel="Estimated flux",
        title="Flux diagonal (y = x)",
    )
    _plot_series_panel(
        axes[1, 0],
        x=x,
        true_vals=true_eff,
        est_vals=est_eff,
        unc_abs=unc_eff_abs,
        ylabel="eff",
        title="Efficiency time series",
    )
    axes[1, 0].set_xlabel(x_label)
    _plot_diag_panel(
        axes[1, 1],
        true_vals=true_eff,
        est_vals=est_eff,
        unc_abs=unc_eff_abs,
        xlabel="Simulated efficiency",
        ylabel="Estimated efficiency",
        title="Efficiency diagonal (y = x)",
    )
    fig.suptitle("STEP 3.3 correction diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correction_overview_2x2.png", dpi=160)
    plt.close(fig)


def main() -> int:
    """Run STEP 3.3 correction workflow."""
    parser = argparse.ArgumentParser(
        description="Step 3.3: Apply dictionary inference + LUT uncertainty to synthetic dataset."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--synthetic-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--lut-csv", default=None)
    parser.add_argument("--lut-meta-json", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_21 = config.get("step_2_1", {})
    cfg_32 = config.get("step_3_2", {})
    cfg_33 = config.get("step_3_3", {})

    synthetic_csv_cfg = cfg_33.get("synthetic_dataset_csv", None)
    time_series_csv_cfg = cfg_33.get("time_series_csv", cfg_32.get("time_series_csv", None))
    complete_curve_csv_cfg = cfg_33.get("complete_curve_csv", cfg_32.get("complete_curve_csv", None))
    dictionary_csv_cfg = cfg_33.get("dictionary_csv", None)
    dataset_template_csv_cfg = cfg_33.get("dataset_template_csv", cfg_32.get("dataset_template_csv", None))
    lut_csv_cfg = cfg_33.get("uncertainty_lut_csv", None)
    lut_meta_cfg = cfg_33.get("uncertainty_lut_meta_json", None)

    synthetic_path = _resolve_input_path(args.synthetic_csv or synthetic_csv_cfg or DEFAULT_SYNTHETIC_DATASET)
    time_series_path = _resolve_input_path(time_series_csv_cfg or DEFAULT_TIME_SERIES)
    complete_curve_path = _resolve_input_path(complete_curve_csv_cfg or DEFAULT_COMPLETE_CURVE)
    dictionary_path = _resolve_input_path(args.dictionary_csv or dictionary_csv_cfg or DEFAULT_DICTIONARY)
    dataset_template_path = _resolve_input_path(dataset_template_csv_cfg or DEFAULT_DATASET_TEMPLATE)
    lut_path = _resolve_input_path(args.lut_csv or lut_csv_cfg or DEFAULT_LUT)
    lut_meta_path = _resolve_input_path(args.lut_meta_json or lut_meta_cfg or DEFAULT_LUT_META)

    for label, p in (
        ("Synthetic dataset", synthetic_path),
        ("Dictionary", dictionary_path),
        ("Uncertainty LUT", lut_path),
    ):
        if not p.exists():
            log.error("%s CSV not found: %s", label, p)
            return 1

    feature_columns = cfg_21.get("feature_columns", "auto")
    distance_metric = str(cfg_21.get("distance_metric", "l2_zscore"))
    interpolation_k = int(_safe_float(cfg_21.get("interpolation_k", 5), 5))
    include_global_rate = _safe_bool(cfg_21.get("include_global_rate", True), True)
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate"))
    exclude_same_file = _safe_bool(cfg_33.get("exclude_same_file", True), True)
    uncertainty_quantile = _safe_float(cfg_33.get("uncertainty_quantile", 0.68), 0.68)
    uncertainty_quantile = float(np.clip(uncertainty_quantile, 0.0, 1.0))

    flux_col = str(cfg_32.get("flux_column", config.get("step_3_1", {}).get("flux_column", "flux_cm2_min")))
    eff_pref = str(cfg_32.get("eff_column", config.get("step_3_1", {}).get("eff_column", "eff_sim_1")))

    log.info("Synthetic dataset: %s", synthetic_path)
    log.info("Time series:      %s", time_series_path)
    log.info("Complete curve:   %s", complete_curve_path)
    log.info("Dictionary:       %s", dictionary_path)
    log.info("LUT:              %s", lut_path)
    log.info("Metric=%s, k=%d, uncertainty_quantile=%.3f", distance_metric, interpolation_k, uncertainty_quantile)

    synthetic_df = pd.read_csv(synthetic_path, low_memory=False)
    if synthetic_df.empty:
        log.error("Synthetic dataset is empty: %s", synthetic_path)
        return 1

    time_df = pd.DataFrame()
    if time_series_path.exists():
        time_df = pd.read_csv(time_series_path, low_memory=False)
    if time_df.empty:
        # Fallback to synthetic table for discretized trajectory.
        fallback_cols = [c for c in ("file_index", "elapsed_hours", "n_events") if c in synthetic_df.columns]
        time_df = synthetic_df[fallback_cols].copy()

    complete_df = None
    if complete_curve_path.exists():
        tmp_complete = pd.read_csv(complete_curve_path, low_memory=False)
        complete_df = tmp_complete if not tmp_complete.empty else None

    try:
        eff_col = _choose_eff_column(synthetic_df, eff_pref)
    except KeyError as exc:
        log.error("%s", exc)
        return 1
    if flux_col not in synthetic_df.columns:
        log.error("Flux column '%s' not found in synthetic dataset.", flux_col)
        return 1
    if flux_col not in time_df.columns:
        time_df[flux_col] = pd.to_numeric(synthetic_df.get(flux_col), errors="coerce")
    try:
        eff_col_time = _choose_eff_column(time_df, eff_pref)
    except KeyError:
        eff_col_time = eff_col
        time_df[eff_col_time] = pd.to_numeric(synthetic_df.get(eff_col), errors="coerce")

    # ── 1) Inference over synthetic dataset ─────────────────────────
    est_df = estimate_parameters(
        dictionary_path=str(dictionary_path),
        dataset_path=str(synthetic_path),
        feature_columns=feature_columns,
        distance_metric=distance_metric,
        interpolation_k=interpolation_k,
        include_global_rate=include_global_rate,
        global_rate_col=global_rate_col,
        exclude_same_file=exclude_same_file,
    )

    # Merge with synthetic rows for time axes and true values.
    syn_with_idx = synthetic_df.copy()
    syn_with_idx["dataset_index"] = np.arange(len(syn_with_idx), dtype=int)
    merged = pd.merge(est_df, syn_with_idx, on="dataset_index", how="left", suffixes=("", "_synthetic"))

    # Ensure key true columns exist with explicit names.
    merged[f"true_{flux_col}"] = pd.to_numeric(merged.get(flux_col), errors="coerce")
    merged[f"true_{eff_col}"] = pd.to_numeric(merged.get(eff_col), errors="coerce")
    if "n_events" in merged.columns:
        merged["n_events"] = pd.to_numeric(merged["n_events"], errors="coerce")
    elif "true_n_events" in merged.columns:
        merged["n_events"] = pd.to_numeric(merged["true_n_events"], errors="coerce")

    # ── 2) LUT uncertainty interpolation ────────────────────────────
    lut_df = _load_lut(lut_path)
    lut_params = _lut_param_names(lut_df, lut_meta_path if lut_meta_path.exists() else None)
    lut_params = [p for p in lut_params if f"est_{p}" in merged.columns]
    if not lut_params:
        log.warning("No LUT parameter matches found in estimation output. Uncertainty columns will be NaN.")

    unc_df = _interpolate_uncertainties(
        query_df=merged,
        lut_df=lut_df,
        param_names=lut_params,
        quantile=uncertainty_quantile,
    )
    merged = pd.concat([merged, unc_df], axis=1)

    for pname in lut_params:
        est_col = f"est_{pname}"
        unc_pct_col = f"unc_{pname}_pct"
        unc_abs_col = f"unc_{pname}_abs"
        if est_col in merged.columns and unc_pct_col in merged.columns:
            est_v = pd.to_numeric(merged[est_col], errors="coerce").to_numpy(dtype=float)
            up = pd.to_numeric(merged[unc_pct_col], errors="coerce").to_numpy(dtype=float)
            merged[unc_abs_col] = np.abs(est_v) * np.abs(up) / 100.0
        else:
            merged[unc_abs_col] = np.nan

    center_flux, center_eff, center_eff_col = _compute_density_center_series(
        config=config,
        time_df=time_df,
        flux_col=flux_col,
        eff_pref=eff_pref,
        dictionary_path=dictionary_path,
        dataset_template_path=dataset_template_path,
    )
    if center_eff_col is not None:
        eff_col_time = center_eff_col

    # Primary diagnostic parameter columns.
    est_flux_col = f"est_{flux_col}" if f"est_{flux_col}" in merged.columns else "est_flux_cm2_min"
    est_eff_col = f"est_{eff_col}" if f"est_{eff_col}" in merged.columns else f"est_{eff_pref}"
    if est_eff_col not in merged.columns:
        for c in ("est_eff_sim_1", "est_eff_sim_2", "est_eff_sim_3", "est_eff_sim_4"):
            if c in merged.columns:
                est_eff_col = c
                break

    true_flux_col = f"true_{flux_col}"
    true_eff_col = f"true_{eff_col}"
    flux_param = est_flux_col.replace("est_", "", 1) if est_flux_col.startswith("est_") else flux_col
    eff_param = est_eff_col.replace("est_", "", 1) if est_eff_col.startswith("est_") else eff_col
    flux_unc_abs_col = f"unc_{flux_param}_abs" if f"unc_{flux_param}_abs" in merged.columns else None
    eff_unc_abs_col = f"unc_{eff_param}_abs" if f"unc_{eff_param}_abs" in merged.columns else None

    # Error columns for primary diagnostics.
    t_flux = pd.to_numeric(merged.get(true_flux_col), errors="coerce")
    e_flux = pd.to_numeric(merged.get(est_flux_col), errors="coerce")
    t_eff = pd.to_numeric(merged.get(true_eff_col), errors="coerce")
    e_eff = pd.to_numeric(merged.get(est_eff_col), errors="coerce")
    merged["error_flux"] = e_flux - t_flux
    merged["error_eff"] = e_eff - t_eff
    merged["relerr_flux_pct"] = (e_flux - t_flux) / t_flux.replace({0.0: np.nan}) * 100.0
    merged["relerr_eff_pct"] = (e_eff - t_eff) / t_eff.replace({0.0: np.nan}) * 100.0
    merged["abs_relerr_flux_pct"] = merged["relerr_flux_pct"].abs()
    merged["abs_relerr_eff_pct"] = merged["relerr_eff_pct"].abs()

    # ── Save outputs ────────────────────────────────────────────────
    out_csv = FILES_DIR / "corrected_by_inference.csv"
    merged.to_csv(out_csv, index=False)
    log.info("Wrote corrected table: %s (%d rows)", out_csv, len(merged))

    summary = {
        "synthetic_dataset_csv": str(synthetic_path),
        "dictionary_csv": str(dictionary_path),
        "uncertainty_lut_csv": str(lut_path),
        "distance_metric": distance_metric,
        "interpolation_k": interpolation_k,
        "uncertainty_quantile": uncertainty_quantile,
        "n_rows": int(len(merged)),
        "n_successful_flux_estimates": int(pd.to_numeric(merged.get(est_flux_col), errors="coerce").notna().sum()),
        "n_successful_eff_estimates": int(pd.to_numeric(merged.get(est_eff_col), errors="coerce").notna().sum()),
        "flux_true_col": true_flux_col,
        "flux_est_col": est_flux_col,
        "eff_true_col": true_eff_col,
        "eff_est_col": est_eff_col,
        "flux_unc_abs_col": flux_unc_abs_col,
        "eff_unc_abs_col": eff_unc_abs_col,
        "median_abs_relerr_flux_pct": float(pd.to_numeric(merged["abs_relerr_flux_pct"], errors="coerce").median()),
        "median_abs_relerr_eff_pct": float(pd.to_numeric(merged["abs_relerr_eff_pct"], errors="coerce").median()),
        "median_unc_flux_pct": (
            float(pd.to_numeric(merged.get(f"unc_{flux_param}_pct"), errors="coerce").median())
            if f"unc_{flux_param}_pct" in merged.columns else None
        ),
        "median_unc_eff_pct": (
            float(pd.to_numeric(merged.get(f"unc_{eff_param}_pct"), errors="coerce").median())
            if f"unc_{eff_param}_pct" in merged.columns else None
        ),
        "lut_param_names_used": lut_params,
        "density_center_available": bool(center_flux is not None and center_eff is not None),
    }
    out_summary = FILES_DIR / "correction_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary: %s", out_summary)

    # ── Plots ───────────────────────────────────────────────────────
    for stale in (
        "diag_flux_true_vs_est.png",
        "diag_eff_true_vs_est.png",
        "estimated_vs_true_time_series.png",
    ):
        p = PLOTS_DIR / stale
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    _make_plots(
        merged,
        flux_true_col=true_flux_col,
        flux_est_col=est_flux_col,
        flux_unc_abs_col=flux_unc_abs_col,
        eff_true_col=true_eff_col,
        eff_est_col=est_eff_col,
        eff_unc_abs_col=eff_unc_abs_col,
    )
    out_overlay = PLOTS_DIR / "synthetic_time_series_overview_with_estimated.png"
    _plot_step32_style_overlay(
        complete_df=complete_df,
        time_df=time_df,
        merged_df=merged,
        flux_col=flux_col,
        eff_col_time=eff_col_time,
        est_flux_col=est_flux_col,
        est_eff_col=est_eff_col,
        flux_unc_abs_col=flux_unc_abs_col,
        eff_unc_abs_col=eff_unc_abs_col,
        center_flux=center_flux,
        center_eff=center_eff,
        path=out_overlay,
    )

    # Simulated flux + estimated flux + global-rate context.
    global_rate_plot_col = None
    for c in (
        global_rate_col,
        f"true_{global_rate_col}",
        "events_per_second_global_rate",
        "global_rate_hz_mean",
        "true_events_per_second_global_rate",
    ):
        if c in merged.columns:
            global_rate_plot_col = c
            break
    if global_rate_plot_col is None:
        log.warning(
            "Could not produce flux/global-rate plot: no global-rate column found (preferred: %s).",
            global_rate_col,
        )
    else:
        out_flux_rate = PLOTS_DIR / "flux_recovery_vs_global_rate.png"
        _plot_flux_recovery_vs_global_rate(
            df=merged,
            complete_df=complete_df,
            flux_complete_col=flux_col,
            eff_complete_col=eff_col_time,
            flux_true_col=true_flux_col,
            eff_true_col=true_eff_col,
            flux_est_col=est_flux_col,
            flux_unc_abs_col=flux_unc_abs_col,
            global_rate_col=global_rate_plot_col,
            path=out_flux_rate,
        )

    log.info("Wrote plots in: %s", PLOTS_DIR)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
