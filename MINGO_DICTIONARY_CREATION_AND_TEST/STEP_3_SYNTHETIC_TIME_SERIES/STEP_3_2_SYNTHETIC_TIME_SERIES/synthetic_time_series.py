#!/usr/bin/env python3
"""STEP 3.2 — Build synthetic dataset from basis table + complete curve.

For each discretized point from STEP 3.1 in the `(flux, eff)` plane, this step:

1. Computes basis-row proximity weights in `(flux, eff)`.
2. Builds a synthetic row by weighted combination of numeric basis columns.
3. Keeps non-numeric morphology from the dominant contributor.
4. Overrides key truth/time columns with STEP 3.1 values.
5. Constrains basis rows per time point using that point's `n_events`.

The output table keeps STEP 1.2 dataset morphology (same base columns) and adds
time/traceability columns.

Output
------
OUTPUTS/FILES/synthetic_dataset.csv
    Synthetic table for STEP 3.3 inference/correction.
OUTPUTS/FILES/synthetic_generation_summary.json
    Summary of weighting settings and generated table statistics.
OUTPUTS/FILES/highlight_point_contributions.csv
    Contribution percentages for one highlighted curve point.
OUTPUTS/PLOTS/dictionary_contributions_highlight.png
    Single combined plot: dictionary contributions + complete/discretized curve.
OUTPUTS/PLOTS/synthetic_time_series_overview.png
    Complete/discretized overlays for flux/eff and global-rate comparison.
OUTPUTS/PLOTS/events_count_dataset_vs_basis_subset.png
    Histogram of `n_events` for full dataset and selected basis subset (same bins).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = STEP_DIR.parents[1]  # INFERENCE_DICTIONARY_VALIDATION
SYNTHETIC_DIR = STEP_DIR.parent      # STEP_3_SYNTHETIC_TIME_SERIES
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"

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

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="[%(levelname)s] STEP_3.2 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_3.2")


def _load_config(path: Path) -> dict:
    """Load JSON config if it exists."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _safe_float(value: object, default: float) -> float:
    """Convert to float with fallback."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isfinite(out):
        return out
    return float(default)


def _safe_int(value: object, default: int, minimum: int | None = None) -> int:
    """Convert to int with fallback and optional lower bound."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if minimum is not None:
        out = max(int(minimum), out)
    return out


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
    """Return efficiency column from table, with fallback candidates."""
    if preferred in df.columns:
        return preferred
    for candidate in (
        "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
        "eff_empirical_1", "eff_empirical_2", "eff_empirical_3", "eff_empirical_4",
    ):
        if candidate in df.columns:
            return candidate
    raise KeyError("No efficiency column found in table.")


def _build_event_mask(
    basis_events: np.ndarray | None,
    target_events: np.ndarray | None,
    *,
    tolerance_pct: float,
    min_rows: int,
) -> tuple[np.ndarray | None, dict]:
    """Build per-target basis mask from event-count proximity."""
    info = {
        "mode": "disabled",
        "tolerance_pct": float(tolerance_pct),
        "min_rows_fallback": int(max(1, int(min_rows))),
    }
    if basis_events is None or target_events is None:
        return None, info

    b = np.asarray(basis_events, dtype=float)
    t = np.asarray(target_events, dtype=float)
    n_targets = len(t)
    n_basis = len(b)
    if n_targets == 0 or n_basis == 0:
        return None, info

    finite_basis = np.isfinite(b)
    finite_targets = np.isfinite(t)
    info["n_basis_rows"] = int(n_basis)
    info["n_basis_rows_with_finite_events"] = int(np.sum(finite_basis))
    info["n_targets"] = int(n_targets)
    info["n_targets_with_finite_events"] = int(np.sum(finite_targets))
    if not np.any(finite_basis):
        info["mode"] = "disabled_no_finite_basis_events"
        return None, info

    tol_pct = max(float(tolerance_pct), 0.0)
    keep_n = max(1, int(min_rows))
    mask = np.zeros((n_targets, n_basis), dtype=bool)
    counts = np.zeros(n_targets, dtype=int)
    fallback_points = 0
    finite_idx = np.where(finite_basis)[0]

    for i in range(n_targets):
        if not finite_targets[i]:
            m = finite_basis.copy()
        else:
            tol_abs = abs(float(t[i])) * tol_pct / 100.0
            m = finite_basis & (np.abs(b - float(t[i])) <= tol_abs)
            if not np.any(m):
                take = min(keep_n, len(finite_idx))
                nearest = finite_idx[np.argsort(np.abs(b[finite_idx] - float(t[i])))[:take]]
                m = np.zeros(n_basis, dtype=bool)
                m[nearest] = True
                fallback_points += 1
        mask[i] = m
        counts[i] = int(np.sum(m))

    info["mode"] = "enabled"
    info["fallback_points_count"] = int(fallback_points)
    info["allowed_rows_per_target_min"] = int(np.min(counts))
    info["allowed_rows_per_target_median"] = float(np.median(counts))
    info["allowed_rows_per_target_max"] = int(np.max(counts))
    return mask, info


def _select_parameter_set_column(df: pd.DataFrame, configured: str | None) -> str | None:
    """Resolve parameter-set identifier column for basis deduplication."""
    if configured is not None:
        col = str(configured).strip()
        if col and col in df.columns:
            return col
    for col in ("param_hash_x", "param_hash_y", "param_set_id"):
        if col in df.columns:
            return col
    return None


def _build_one_per_parameter_set_mask(
    *,
    parameter_set_values: np.ndarray,
    basis_events: np.ndarray | None,
    target_events: np.ndarray | None,
    basis_flux: np.ndarray,
    basis_eff: np.ndarray,
    target_flux: np.ndarray,
    target_eff: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Select exactly one basis row per parameter set and target point."""
    group_vals = np.asarray(parameter_set_values, dtype=object)
    n_basis = len(group_vals)
    n_targets = len(target_flux)
    if n_basis == 0 or n_targets == 0:
        return np.zeros((n_targets, n_basis), dtype=bool), {
            "mode": "empty",
            "n_parameter_sets": 0,
            "allowed_rows_per_target_min": 0,
            "allowed_rows_per_target_median": 0.0,
            "allowed_rows_per_target_max": 0,
        }

    # Stable per-group indices preserving first-seen group order.
    groups: dict[object, list[int]] = {}
    for j, g in enumerate(group_vals):
        groups.setdefault(g, []).append(j)
    group_index_arrays = [np.asarray(idxs, dtype=int) for idxs in groups.values()]

    be = None if basis_events is None else np.asarray(basis_events, dtype=float)
    te = None if target_events is None else np.asarray(target_events, dtype=float)
    bf = np.asarray(basis_flux, dtype=float)
    beff = np.asarray(basis_eff, dtype=float)
    tf = np.asarray(target_flux, dtype=float)
    teff = np.asarray(target_eff, dtype=float)

    mask = np.zeros((n_targets, n_basis), dtype=bool)
    for i in range(n_targets):
        t_ev = np.nan if te is None or i >= len(te) else float(te[i])
        t_fx = float(tf[i]) if i < len(tf) else np.nan
        t_ef = float(teff[i]) if i < len(teff) else np.nan

        for idxs in group_index_arrays:
            chosen = int(idxs[0])
            if be is not None:
                ev = be[idxs]
                finite = np.isfinite(ev)
                if np.any(finite) and np.isfinite(t_ev):
                    cand = idxs[finite]
                    dev = np.abs(be[cand] - t_ev)
                    best = np.flatnonzero(dev == np.min(dev))
                    cand_best = cand[best]
                    if len(cand_best) > 1 and np.isfinite(t_fx) and np.isfinite(t_ef):
                        d = (bf[cand_best] - t_fx) ** 2 + (beff[cand_best] - t_ef) ** 2
                        chosen = int(cand_best[int(np.argmin(d))])
                    else:
                        chosen = int(cand_best[0])
                elif np.any(finite):
                    cand = idxs[finite]
                    if np.isfinite(t_fx) and np.isfinite(t_ef):
                        d = (bf[cand] - t_fx) ** 2 + (beff[cand] - t_ef) ** 2
                        chosen = int(cand[int(np.argmin(d))])
                    else:
                        chosen = int(cand[0])
                elif np.isfinite(t_fx) and np.isfinite(t_ef):
                    d = (bf[idxs] - t_fx) ** 2 + (beff[idxs] - t_ef) ** 2
                    chosen = int(idxs[int(np.argmin(d))])
            else:
                if np.isfinite(t_fx) and np.isfinite(t_ef):
                    d = (bf[idxs] - t_fx) ** 2 + (beff[idxs] - t_ef) ** 2
                    chosen = int(idxs[int(np.argmin(d))])
            mask[i, chosen] = True

    counts = np.sum(mask, axis=1)
    info = {
        "mode": "one_row_per_parameter_set",
        "n_parameter_sets": int(len(group_index_arrays)),
        "allowed_rows_per_target_min": int(np.min(counts)) if len(counts) else 0,
        "allowed_rows_per_target_median": float(np.median(counts)) if len(counts) else 0.0,
        "allowed_rows_per_target_max": int(np.max(counts)) if len(counts) else 0,
    }
    if be is not None:
        info["basis_rows_with_finite_events"] = int(np.sum(np.isfinite(be)))
    return mask, info


def _build_linear_distance_center_weights(
    basis_flux: np.ndarray,
    basis_eff: np.ndarray,
    target_flux: np.ndarray,
    target_eff: np.ndarray,
    *,
    event_mask: np.ndarray | None,
    top_k: int | None,
    hardness: float = 1.0,
) -> np.ndarray:
    """Build linear distance-based weights for center estimation in (flux, eff)."""
    bf = np.asarray(basis_flux, dtype=float)
    be = np.asarray(basis_eff, dtype=float)
    tf = np.asarray(target_flux, dtype=float)
    te = np.asarray(target_eff, dtype=float)
    if bf.ndim != 1 or be.ndim != 1 or len(bf) != len(be):
        raise ValueError("basis_flux and basis_eff must be 1D with same length")
    if tf.ndim != 1 or te.ndim != 1 or len(tf) != len(te):
        raise ValueError("target_flux and target_eff must be 1D with same length")

    flux_span = max(float(np.nanmax(bf) - np.nanmin(bf)), 1e-12)
    eff_span = max(float(np.nanmax(be) - np.nanmin(be)), 1e-12)
    dx = (tf[:, None] - bf[None, :]) / flux_span
    dy = (te[:, None] - be[None, :]) / eff_span
    d_full = np.sqrt(dx * dx + dy * dy)
    d = d_full.copy()

    n_targets, n_basis = d.shape
    if event_mask is not None:
        m = np.asarray(event_mask, dtype=bool)
        if m.shape != d.shape:
            raise ValueError("event_mask shape must match (n_targets, n_basis)")
        d = np.where(m, d, np.inf)

    k = None
    if top_k is not None:
        k = max(1, int(top_k))

    hard = max(float(hardness), 1e-6)
    w = np.zeros_like(d, dtype=float)
    for i in range(n_targets):
        row = d[i]
        finite_idx = np.where(np.isfinite(row))[0]
        if finite_idx.size == 0:
            # Fallback: nearest overall in flux-eff space.
            j = int(np.argmin(d_full[i]))
            w[i, j] = 1.0
            continue

        selected_idx = finite_idx
        if k is not None and k < finite_idx.size:
            order = finite_idx[np.argsort(row[finite_idx])]
            selected_idx = order[:k]

        sel_d = row[selected_idx]
        dmin = float(np.min(sel_d))
        dmax = float(np.max(sel_d))
        if dmax <= dmin + 1e-12:
            w[i, selected_idx] = 1.0 / float(len(selected_idx))
            continue

        # Linear profile: closest gets highest weight, farthest gets zero.
        lin = 1.0 - ((sel_d - dmin) / (dmax - dmin))
        lin = np.clip(lin, 0.0, None)
        if hard != 1.0:
            lin = np.power(lin, hard)
        s = float(np.sum(lin))
        if s <= 0.0:
            w[i, selected_idx] = 1.0 / float(len(selected_idx))
        else:
            w[i, selected_idx] = lin / s
    return w


def _compute_inverse_density_scaling(
    basis_flux: np.ndarray,
    basis_eff: np.ndarray,
    *,
    k_neighbors: int,
    exponent: float,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, dict]:
    """Compute per-basis scaling to reduce dense-region dominance."""
    bf = np.asarray(basis_flux, dtype=float)
    be = np.asarray(basis_eff, dtype=float)
    n = len(bf)
    info = {
        "enabled": True,
        "k_neighbors": int(k_neighbors),
        "exponent": float(exponent),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "n_basis": int(n),
    }
    if n == 0:
        return np.array([], dtype=float), info
    if n == 1:
        return np.ones(1, dtype=float), info

    sx = max(float(np.nanmax(bf) - np.nanmin(bf)), 1e-12)
    sy = max(float(np.nanmax(be) - np.nanmin(be)), 1e-12)
    x = (bf - np.nanmin(bf)) / sx
    y = (be - np.nanmin(be)) / sy
    coords = np.column_stack([x, y])

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)

    k = max(1, int(k_neighbors))
    k_eff = min(k, n - 1)
    kth = np.partition(dist, k_eff - 1, axis=1)[:, k_eff - 1]

    exp = max(float(exponent), 0.0)
    scale = np.power(np.clip(kth, 1e-12, None), exp)
    med = float(np.median(scale[np.isfinite(scale)])) if np.isfinite(scale).any() else 1.0
    if med <= 0.0 or not np.isfinite(med):
        med = 1.0
    scale = scale / med

    lo = max(float(clip_min), 1e-6)
    hi = max(float(clip_max), lo)
    scale = np.clip(scale, lo, hi)

    info["effective_k_neighbors"] = int(k_eff)
    info["scale_min"] = float(np.min(scale))
    info["scale_median"] = float(np.median(scale))
    info["scale_max"] = float(np.max(scale))
    return scale, info


def _build_weights(
    dict_flux: np.ndarray,
    dict_eff: np.ndarray,
    target_flux: np.ndarray,
    target_eff: np.ndarray,
    *,
    method: str,
    sigma_flux: float,
    sigma_eff: float,
    top_k: int | None,
    distance_hardness: float = 1.0,
    density_scaling: np.ndarray | None = None,
    event_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute normalized basis weights for each target point."""
    sigma_flux = max(float(sigma_flux), 1e-12)
    sigma_eff = max(float(sigma_eff), 1e-12)
    dx = (target_flux[:, None] - dict_flux[None, :]) / sigma_flux
    dy = (target_eff[:, None] - dict_eff[None, :]) / sigma_eff
    d2 = dx * dx + dy * dy
    if event_mask is not None and event_mask.shape != d2.shape:
        raise ValueError("event_mask shape must match (n_targets, n_basis)")

    method_key = str(method).strip().lower()
    if method_key == "nearest":
        w = np.zeros_like(d2, dtype=float)
        for i in range(d2.shape[0]):
            row_d2 = d2[i]
            if event_mask is not None:
                row_mask = event_mask[i]
                if np.any(row_mask):
                    row_d2 = np.where(row_mask, row_d2, np.inf)
            j = int(np.argmin(row_d2))
            w[i, j] = 1.0
        return w

    # Default gaussian
    hard = max(float(distance_hardness), 1e-6)
    w = np.exp(-0.5 * hard * d2)
    if event_mask is not None:
        w = np.where(event_mask, w, 0.0)
    if density_scaling is not None:
        ds = np.asarray(density_scaling, dtype=float)
        if ds.ndim != 1 or ds.shape[0] != w.shape[1]:
            raise ValueError("density_scaling must be 1D with length n_basis")
        w = w * ds[None, :]

    k = None
    if top_k is not None:
        k = max(1, int(top_k))
    if k is not None and k < w.shape[1]:
        keep = np.zeros_like(w, dtype=bool)
        idx_part = np.argpartition(w, -k, axis=1)[:, -k:]
        rows = np.arange(w.shape[0])[:, None]
        keep[rows, idx_part] = True
        w = np.where(keep, w, 0.0)

    # Row-wise normalization with nearest fallback on empty rows.
    row_sum = w.sum(axis=1, keepdims=True)
    empty = (row_sum[:, 0] <= 0.0)
    if np.any(empty):
        for i in np.where(empty)[0]:
            row_d2 = d2[i]
            if event_mask is not None:
                row_mask = event_mask[i]
                if np.any(row_mask):
                    row_d2 = np.where(row_mask, row_d2, np.inf)
            j = int(np.argmin(row_d2))
            w[i] = 0.0
            w[i, j] = 1.0
        row_sum = w.sum(axis=1, keepdims=True)
    return w / row_sum


def _weighted_numeric_columns(weights: np.ndarray, dict_df: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    """Weighted averages for numeric dictionary columns, NaN-aware."""
    out: dict[str, np.ndarray] = {}
    n_rows = weights.shape[0]
    for col in columns:
        values = pd.to_numeric(dict_df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not np.any(valid):
            out[col] = np.full(n_rows, np.nan, dtype=float)
            continue
        w = weights[:, valid]
        v = values[valid]
        num = w @ v
        den = w.sum(axis=1)
        out[col] = np.divide(num, den, out=np.full(n_rows, np.nan), where=den > 0)
    return out


def _round_count_like_columns(df: pd.DataFrame) -> None:
    """Round count-like columns in place to non-negative integers."""
    count_keywords = ("_count", "_entries_", "n_events", "selected_rows", "requested_rows", "generated_events_count")
    for col in df.columns:
        if not any(k in col for k in count_keywords):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            rounded = np.rint(np.clip(s.to_numpy(dtype=float), 0.0, None))
            df[col] = pd.Series(rounded, index=df.index).astype("Int64")


def _rebuild_efficiencies_string(df: pd.DataFrame) -> None:
    """Rebuild efficiencies string column when simulation efficiencies exist."""
    if "efficiencies" not in df.columns:
        return
    needed = [f"eff_sim_{i}" for i in range(1, 5)]
    if not all(c in df.columns for c in needed):
        return

    def _fmt(row: pd.Series) -> str:
        vals = []
        for c in needed:
            v = pd.to_numeric(row[c], errors="coerce")
            vals.append(float(v) if pd.notna(v) else np.nan)
        return str(vals)

    df["efficiencies"] = df[needed].apply(_fmt, axis=1)


def _make_synthetic_dataset(
    dictionary_df: pd.DataFrame,
    template_df: pd.DataFrame,
    time_df: pd.DataFrame,
    weights: np.ndarray,
    *,
    flux_col: str,
    eff_col: str,
    time_rate_col: str,
    time_events_col: str,
    time_duration_col: str,
    flux_output_values: np.ndarray | None = None,
    eff_output_values: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create synthetic dataset preserving template morphology."""
    n_targets = len(time_df)
    template_cols = list(template_df.columns)
    common_cols = [c for c in template_cols if c in dictionary_df.columns]
    out = pd.DataFrame(index=np.arange(n_targets), columns=template_cols)

    # Dominant row gives categorical/non-numeric morphology.
    dominant_idx = np.argmax(weights, axis=1)
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(dictionary_df[col]):
            continue
        out[col] = dictionary_df[col].to_numpy()[dominant_idx]

    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(dictionary_df[c])]
    numeric_values = _weighted_numeric_columns(weights, dictionary_df, numeric_cols)
    for col, values in numeric_values.items():
        out[col] = values

    # Target truth overrides from STEP 3.1.
    target_flux = pd.to_numeric(time_df[flux_col], errors="coerce")
    target_eff = pd.to_numeric(time_df[eff_col], errors="coerce")
    target_rate = pd.to_numeric(time_df[time_rate_col], errors="coerce")
    target_events = pd.to_numeric(time_df[time_events_col], errors="coerce")
    target_duration = pd.to_numeric(time_df[time_duration_col], errors="coerce")
    output_flux = target_flux
    output_eff = target_eff
    if flux_output_values is not None and len(flux_output_values) == n_targets:
        output_flux = pd.Series(np.asarray(flux_output_values, dtype=float), index=time_df.index)
    if eff_output_values is not None and len(eff_output_values) == n_targets:
        output_eff = pd.Series(np.asarray(eff_output_values, dtype=float), index=time_df.index)

    if "flux_cm2_min" in out.columns:
        out["flux_cm2_min"] = output_flux
    if "flux" in out.columns:
        out["flux"] = output_flux
    if "eff_sim_1" in out.columns:
        out["eff_sim_1"] = output_eff
    if "eff" in out.columns:
        out["eff"] = output_eff
    if "events_per_second_global_rate" in out.columns:
        out["events_per_second_global_rate"] = target_rate
    if "n_events" in out.columns:
        out["n_events"] = target_events
    for c in ("selected_rows", "requested_rows", "generated_events_count"):
        if c in out.columns:
            out[c] = target_events
    for c in ("count_rate_denominator_seconds", "events_per_second_total_seconds"):
        if c in out.columns:
            out[c] = target_duration
    if "is_dictionary_entry" in out.columns:
        out["is_dictionary_entry"] = False

    # Keep synthetic identifiers consistent and non-colliding.
    synthetic_ids = np.array([f"synthetic_{i:06d}" for i in range(1, n_targets + 1)], dtype=object)
    if "filename_base" in out.columns:
        out["filename_base"] = synthetic_ids
    for c in ("param_hash_x", "param_hash_y"):
        if c in out.columns:
            out[c] = synthetic_ids

    _rebuild_efficiencies_string(out)
    _round_count_like_columns(out)

    # Add time/traceability columns in one concat to avoid dataframe fragmentation.
    extras = pd.DataFrame({
        "file_index": pd.to_numeric(time_df.get("file_index"), errors="coerce").astype("Int64"),
        "time_start_utc": time_df.get("time_start_utc"),
        "time_end_utc": time_df.get("time_end_utc"),
        "time_utc": time_df.get("time_utc"),
        "elapsed_hours_start": pd.to_numeric(time_df.get("elapsed_hours_start"), errors="coerce"),
        "elapsed_hours_end": pd.to_numeric(time_df.get("elapsed_hours_end"), errors="coerce"),
        "elapsed_hours": pd.to_numeric(time_df.get("elapsed_hours"), errors="coerce"),
        "duration_seconds": target_duration,
        "target_events_per_file": pd.to_numeric(time_df.get("target_events_per_file"), errors="coerce").astype("Int64"),
        "n_events_expected": pd.to_numeric(time_df.get("n_events_expected"), errors="coerce"),
        "global_rate_hz_source": target_rate,
        "dominant_dictionary_index": dominant_idx,
    }, index=out.index)
    if "filename_base" in dictionary_df.columns:
        extras["dominant_dictionary_filename_base"] = dictionary_df["filename_base"].astype(str).to_numpy()[dominant_idx]

    out = pd.concat([out, extras], axis=1).copy()
    return out, dominant_idx


def _plot_highlight_contributions(
    complete_df: pd.DataFrame | None,
    time_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
    weights: np.ndarray,
    flux_col: str,
    eff_col_time: str,
    eff_col_dict: str,
    basis_label: str,
    highlight_idx: int,
    path: Path,
) -> None:
    """Plot highlighted point and dictionary contribution percentages."""
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.2))

    # Base layer: dictionary contribution cloud for selected target point.
    dx_s = pd.to_numeric(dictionary_df.get(flux_col), errors="coerce")
    dy_s = pd.to_numeric(dictionary_df.get(eff_col_dict), errors="coerce")
    contrib_all = weights[highlight_idx] * 100.0
    m_dict = dx_s.notna() & dy_s.notna() & np.isfinite(contrib_all)
    if m_dict.any():
        m_nonzero = m_dict & (contrib_all > 0.0)
        m_zero = m_dict & ~m_nonzero
        if m_zero.any():
            ax.scatter(
                dx_s[m_zero],
                dy_s[m_zero],
                s=14,
                color="lightgray",
                alpha=0.80,
                linewidths=0.0,
                label=f"{basis_label} (excluded by event constraint)",
                zorder=0,
            )
        if m_nonzero.any():
            cvals = contrib_all[np.asarray(m_nonzero)]
            sc = ax.scatter(
                dx_s[m_nonzero],
                dy_s[m_nonzero],
                c=cvals,
                cmap="viridis",
                s=28 + 250 * (cvals / max(float(np.max(cvals)), 1e-12)),
                alpha=0.85,
                edgecolors="black",
                linewidths=0.25,
                label=f"{basis_label} (weighted)",
                zorder=1,
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("Contribution [%]")

    if complete_df is not None and not complete_df.empty:
        cx = pd.to_numeric(complete_df.get(flux_col), errors="coerce")
        cy = pd.to_numeric(complete_df.get(eff_col_time), errors="coerce")
        m = cx.notna() & cy.notna()
        if m.any():
            ax.plot(cx[m], cy[m], color="#1f77b4", linewidth=1.5, alpha=0.9, label="Complete curve", zorder=1)

    tx = pd.to_numeric(time_df.get(flux_col), errors="coerce")
    ty = pd.to_numeric(time_df.get(eff_col_time), errors="coerce")
    m = tx.notna() & ty.notna()
    if m.any():
        ax.scatter(
            tx[m],
            ty[m],
            s=20,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            label="Discretized",
            zorder=2,
        )
    hx = float(tx.iloc[highlight_idx])
    hy = float(ty.iloc[highlight_idx])
    ax.scatter([hx], [hy], s=95, color="#D62728", marker="X", label=f"Highlight idx {highlight_idx}", zorder=3)
    ax.set_xlabel("flux_cm2_min")
    ax.set_ylabel("eff")
    ax.set_title(f"{basis_label} contributions with complete/discretized curve")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    # Shared limits over all plotted points.
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    if m_dict.any():
        x_parts.append(dx_s[m_dict].to_numpy(dtype=float))
        y_parts.append(dy_s[m_dict].to_numpy(dtype=float))
    if complete_df is not None and not complete_df.empty:
        cx = pd.to_numeric(complete_df.get(flux_col), errors="coerce")
        cy = pd.to_numeric(complete_df.get(eff_col_time), errors="coerce")
        m_comp = cx.notna() & cy.notna()
        if m_comp.any():
            x_parts.append(cx[m_comp].to_numpy(dtype=float))
            y_parts.append(cy[m_comp].to_numpy(dtype=float))
    if m.any():
        x_parts.append(tx[m].to_numpy(dtype=float))
        y_parts.append(ty[m].to_numpy(dtype=float))
    x_parts.append(np.array([hx], dtype=float))
    y_parts.append(np.array([hy], dtype=float))

    x_all = np.concatenate(x_parts) if x_parts else np.array([], dtype=float)
    y_all = np.concatenate(y_parts) if y_parts else np.array([], dtype=float)
    x_all = x_all[np.isfinite(x_all)]
    y_all = y_all[np.isfinite(y_all)]
    if x_all.size and y_all.size:
        x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        if x_max <= x_min:
            x_pad = max(abs(x_min) * 0.05, 1e-6)
            x_min -= x_pad
            x_max += x_pad
        else:
            x_pad = 0.03 * (x_max - x_min)
            x_min -= x_pad
            x_max += x_pad
        if y_max <= y_min:
            y_pad = max(abs(y_min) * 0.05, 1e-6)
            y_min -= y_pad
            y_max += y_pad
        else:
            y_pad = 0.03 * (y_max - y_min)
            y_min -= y_pad
            y_max += y_pad
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_time_series_overview(
    complete_df: pd.DataFrame | None,
    time_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    flux_col: str,
    eff_col: str,
    time_rate_col: str,
    interpolated_flux: np.ndarray | None,
    interpolated_eff: np.ndarray | None,
    interpolated_label: str,
    path: Path,
) -> None:
    """Plot complete/discretized flux-eff and global-rate comparison."""
    def _apply_striped_background(ax: plt.Axes, y_vals: np.ndarray) -> None:
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

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)

    x_disc = pd.to_numeric(time_df.get("elapsed_hours"), errors="coerce")
    y_flux_disc = pd.to_numeric(time_df.get(flux_col), errors="coerce")
    y_eff_disc = pd.to_numeric(time_df.get(eff_col), errors="coerce")
    flux_stripe_vals = [y_flux_disc.to_numpy(dtype=float)]
    eff_stripe_vals = [y_eff_disc.to_numpy(dtype=float)]
    rate_stripe_vals: list[np.ndarray] = []

    if complete_df is not None and not complete_df.empty:
        x_comp = pd.to_numeric(complete_df.get("elapsed_hours"), errors="coerce")
        y_flux_comp = pd.to_numeric(complete_df.get(flux_col), errors="coerce")
        y_eff_comp = pd.to_numeric(complete_df.get(eff_col), errors="coerce")
        flux_stripe_vals.append(y_flux_comp.to_numpy(dtype=float))
        eff_stripe_vals.append(y_eff_comp.to_numpy(dtype=float))
        m0 = x_comp.notna() & y_flux_comp.notna()
        m1 = x_comp.notna() & y_eff_comp.notna()
        if m0.any():
            axes[0].scatter(
                x_comp[m0],
                y_flux_comp[m0],
                s=7,
                color="#1f77b4",
                alpha=0.55,
                linewidths=0.0,
                label="Complete curve",
            )
        if m1.any():
            axes[1].scatter(
                x_comp[m1],
                y_eff_comp[m1],
                s=7,
                color="#FF7F0E",
                alpha=0.55,
                linewidths=0.0,
                label="Complete curve",
            )

    m0d = x_disc.notna() & y_flux_disc.notna()
    m1d = x_disc.notna() & y_eff_disc.notna()
    if m0d.any():
        axes[0].scatter(
            x_disc[m0d],
            y_flux_disc[m0d],
            s=18,
            facecolors="white",
            edgecolors="#1f77b4",
            linewidths=0.8,
            label="Discretized",
        )
    if m1d.any():
        axes[1].scatter(
            x_disc[m1d],
            y_eff_disc[m1d],
            s=18,
            facecolors="white",
            edgecolors="#FF7F0E",
            linewidths=0.8,
            label="Discretized",
        )

    has_interp_flux = False
    if interpolated_flux is not None:
        y_flux_interp = pd.to_numeric(pd.Series(interpolated_flux), errors="coerce")
        flux_stripe_vals.append(y_flux_interp.to_numpy(dtype=float))
        m0i = x_disc.notna() & y_flux_interp.notna()
        if m0i.any():
            has_interp_flux = True
            axes[0].plot(
                x_disc[m0i],
                y_flux_interp[m0i],
                color="#17BECF",
                linewidth=1.0,
                linestyle="-.",
                marker="s",
                markersize=2.9,
                markerfacecolor="#17BECF",
                markeredgewidth=0.0,
                alpha=0.9,
                label=interpolated_label,
            )
    has_interp_eff = False
    if interpolated_eff is not None:
        y_eff_interp = pd.to_numeric(pd.Series(interpolated_eff), errors="coerce")
        eff_stripe_vals.append(y_eff_interp.to_numpy(dtype=float))
        m1i = x_disc.notna() & y_eff_interp.notna()
        if m1i.any():
            has_interp_eff = True
            axes[1].plot(
                x_disc[m1i],
                y_eff_interp[m1i],
                color="#BCBD22",
                linewidth=1.0,
                linestyle="-.",
                marker="s",
                markersize=2.9,
                markerfacecolor="#BCBD22",
                markeredgewidth=0.0,
                alpha=0.9,
                label=interpolated_label,
            )

    axes[0].set_ylabel("flux_cm2_min")
    if has_interp_flux:
        axes[0].set_title("Flux: complete + discretized + derived center")
    else:
        axes[0].set_title("Flux: complete + discretized (used for synthetic output)")
    _apply_striped_background(axes[0], np.concatenate(flux_stripe_vals))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_ylabel("eff")
    if has_interp_eff:
        axes[1].set_title("Efficiency: complete + discretized + derived center")
    else:
        axes[1].set_title("Efficiency: complete + discretized (used for synthetic output)")
    _apply_striped_background(axes[1], np.concatenate(eff_stripe_vals))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    # Global rate overlays: complete, discretized target, synthetic newly calculated.
    if complete_df is not None and not complete_df.empty:
        comp_rate_col = None
        for c in ("global_rate_hz", "global_rate_hz_mean", "events_per_second_global_rate"):
            if c in complete_df.columns:
                comp_rate_col = c
                break
        if comp_rate_col is not None:
            x_comp = pd.to_numeric(complete_df.get("elapsed_hours"), errors="coerce")
            y_comp_rate = pd.to_numeric(complete_df.get(comp_rate_col), errors="coerce")
            rate_stripe_vals.append(y_comp_rate.to_numpy(dtype=float))
            m2c = x_comp.notna() & y_comp_rate.notna()
            if m2c.any():
                axes[2].scatter(
                    x_comp[m2c],
                    y_comp_rate[m2c],
                    s=7,
                    color="#6F3CC3",
                    alpha=0.55,
                    linewidths=0.0,
                    label="Complete curve",
                )

    y_disc_rate = pd.to_numeric(time_df.get(time_rate_col), errors="coerce")
    rate_stripe_vals.append(y_disc_rate.to_numpy(dtype=float))
    m2d = x_disc.notna() & y_disc_rate.notna()
    if m2d.any():
        axes[2].scatter(
            x_disc[m2d],
            y_disc_rate[m2d],
            s=18,
            facecolors="white",
            edgecolors="#2CA02C",
            linewidths=0.8,
            label="Discretized",
        )

    x_syn = pd.to_numeric(synthetic_df.get("elapsed_hours"), errors="coerce")
    syn_events = pd.to_numeric(synthetic_df.get("n_events"), errors="coerce")
    syn_dur = pd.to_numeric(synthetic_df.get("duration_seconds"), errors="coerce")
    y_syn_rate_new = np.divide(
        syn_events.to_numpy(dtype=float),
        syn_dur.to_numpy(dtype=float),
        out=np.full(len(synthetic_df), np.nan, dtype=float),
        where=np.isfinite(syn_dur.to_numpy(dtype=float)) & (syn_dur.to_numpy(dtype=float) > 0),
    )
    rate_stripe_vals.append(np.asarray(y_syn_rate_new, dtype=float))
    m2s = x_syn.notna() & np.isfinite(y_syn_rate_new)
    if m2s.any():
        axes[2].plot(
            x_syn[m2s],
            y_syn_rate_new[m2s],
            color="#D62728",
            linewidth=1.0,
            linestyle="-.",
            marker="s",
            markersize=2.9,
            markerfacecolor="#D62728",
            markeredgewidth=0.0,
            alpha=0.9,
            label="Synthetic (newly calculated)",
        )

    axes[2].set_xlabel("Elapsed time [hours]")
    axes[2].set_ylabel("global rate [Hz]")
    axes[2].set_title("Global rate: complete + discretized + synthetic (new)")
    _apply_striped_background(axes[2], np.concatenate(rate_stripe_vals))
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_events_count_histogram(
    dataset_events: np.ndarray,
    basis_subset_events: np.ndarray,
    path: Path,
) -> None:
    """Plot `n_events` histograms for full dataset vs selected basis subset."""
    ds = np.asarray(dataset_events, dtype=float)
    bs = np.asarray(basis_subset_events, dtype=float)
    ds = ds[np.isfinite(ds)]
    bs = bs[np.isfinite(bs)]
    if ds.size == 0:
        return

    n_bins = int(np.clip(np.sqrt(ds.size), 18, 65))
    lo = float(np.min(ds))
    hi = float(np.max(ds))
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.8))
    ax.hist(
        ds,
        bins=bins,
        color="#9AA0A6",
        alpha=0.45,
        edgecolor="white",
        linewidth=0.6,
        label=f"Dataset (n={len(ds)})",
    )
    if bs.size:
        ax.hist(
            bs,
            bins=bins,
            color="#D95F02",
            alpha=0.55,
            edgecolor="white",
            linewidth=0.6,
            label=f"Basis subset (n={len(bs)})",
        )
        ax.hist(
            bs,
            bins=bins,
            histtype="step",
            color="#8C2D04",
            linewidth=1.2,
        )

    ax.set_xlabel("n_events")
    ax.set_ylabel("Count")
    ax.set_title("Event-count distribution: dataset vs selected basis subset")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    """Run STEP 3.2 synthetic dataset creation."""
    parser = argparse.ArgumentParser(
        description="Step 3.2: Build synthetic dataset from dictionary and time series."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--time-series-csv", default=None)
    parser.add_argument("--complete-curve-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--dataset-template-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_32 = config.get("step_3_2", {})
    basis_source_cfg = str(cfg_32.get("basis_source", "dataset")).strip().lower()

    # Input paths
    if args.time_series_csv:
        time_series_path = _resolve_input_path(args.time_series_csv)
    elif cfg_32.get("time_series_csv"):
        time_series_path = _resolve_input_path(str(cfg_32.get("time_series_csv")))
    else:
        time_series_path = DEFAULT_TIME_SERIES

    if args.complete_curve_csv:
        complete_curve_path = _resolve_input_path(args.complete_curve_csv)
    elif cfg_32.get("complete_curve_csv"):
        complete_curve_path = _resolve_input_path(str(cfg_32.get("complete_curve_csv")))
    else:
        complete_curve_path = DEFAULT_COMPLETE_CURVE

    if args.dictionary_csv:
        dictionary_path = _resolve_input_path(args.dictionary_csv)
    elif cfg_32.get("dictionary_csv"):
        dictionary_path = _resolve_input_path(str(cfg_32.get("dictionary_csv")))
    else:
        dictionary_path = DEFAULT_DICTIONARY

    if args.dataset_template_csv:
        template_path = _resolve_input_path(args.dataset_template_csv)
    elif cfg_32.get("dataset_template_csv"):
        template_path = _resolve_input_path(str(cfg_32.get("dataset_template_csv")))
    else:
        template_path = DEFAULT_DATASET_TEMPLATE

    required_paths = [
        ("Time series", time_series_path),
        ("Dataset template", template_path),
    ]
    if basis_source_cfg == "dictionary":
        required_paths.append(("Dictionary", dictionary_path))
    for label, p in required_paths:
        if not p.exists():
            log.error("%s CSV not found: %s", label, p)
            return 1

    time_df = pd.read_csv(time_series_path, low_memory=False)
    dictionary_df = pd.DataFrame()
    if dictionary_path.exists():
        dictionary_df = pd.read_csv(dictionary_path, low_memory=False)
    template_df = pd.read_csv(template_path, low_memory=False)
    complete_df = None
    if complete_curve_path.exists():
        complete_df = pd.read_csv(complete_curve_path, low_memory=False)

    if time_df.empty:
        log.error("Time series CSV is empty: %s", time_series_path)
        return 1
    if basis_source_cfg == "dictionary" and dictionary_df.empty:
        log.error("Dictionary CSV is empty: %s", dictionary_path)
        return 1
    if template_df.empty:
        log.error("Dataset template CSV is empty: %s", template_path)
        return 1

    flux_col = str(cfg_32.get("flux_column", "flux_cm2_min"))
    eff_pref = str(cfg_32.get("eff_column", "eff_sim_1"))
    basis_source = str(cfg_32.get("basis_source", "dataset")).strip().lower()
    if basis_source == "dictionary":
        basis_input_df = dictionary_df
        basis_label = "Dictionary"
        basis_path = dictionary_path
    elif basis_source == "dataset":
        basis_input_df = template_df
        basis_label = "Dataset"
        basis_path = template_path
    else:
        log.error("Invalid step_3_2.basis_source='%s'. Use 'dataset' or 'dictionary'.", basis_source)
        return 1

    if flux_col not in time_df.columns or flux_col not in basis_input_df.columns:
        log.error("Flux column '%s' must exist in time series and selected basis (%s).", flux_col, basis_source)
        return 1
    try:
        eff_col_time = _choose_eff_column(time_df, eff_pref)
        eff_col_basis = _choose_eff_column(basis_input_df, eff_pref)
    except KeyError as exc:
        log.error("%s", exc)
        return 1
    if eff_col_time != eff_col_basis:
        log.warning("Using %s for time series and %s for basis.", eff_col_time, eff_col_basis)

    time_events_col = str(cfg_32.get("time_n_events_column", "n_events"))
    if time_events_col not in time_df.columns:
        log.error("Time series column '%s' not found.", time_events_col)
        return 1
    time_rate_col = str(cfg_32.get("time_rate_column", "global_rate_hz_mean"))
    if time_rate_col not in time_df.columns:
        # Fallback to midpoint rate if mean is not present.
        fallback = "global_rate_hz_mid"
        if fallback in time_df.columns:
            time_rate_col = fallback
        else:
            log.error("No time-series rate column found (%s / %s).", time_rate_col, fallback)
            return 1
    time_duration_col = str(cfg_32.get("time_duration_column", "duration_seconds"))
    if time_duration_col not in time_df.columns:
        log.error("Time series duration column '%s' not found.", time_duration_col)
        return 1

    basis_events_col = str(cfg_32.get("basis_n_events_column", "n_events"))
    basis_events_tol_pct = _safe_float(
        cfg_32.get("basis_n_events_tolerance_pct", cfg_32.get("basis_n_events_tolerance", 30.0)),
        30.0,
    )
    basis_parameter_set_col_cfg = cfg_32.get("basis_parameter_set_column", None)
    basis_min_rows = _safe_int(cfg_32.get("basis_min_rows", 200), 200, minimum=1)

    basis_flux_all = pd.to_numeric(basis_input_df[flux_col], errors="coerce").to_numpy(dtype=float)
    basis_eff_all = pd.to_numeric(basis_input_df[eff_col_basis], errors="coerce").to_numpy(dtype=float)
    basis_events_all = None
    if basis_events_col in basis_input_df.columns:
        basis_events_all = pd.to_numeric(basis_input_df[basis_events_col], errors="coerce").to_numpy(dtype=float)
    else:
        log.warning("Basis events column '%s' not found in selected basis; event filtering disabled.", basis_events_col)

    target_flux = pd.to_numeric(time_df[flux_col], errors="coerce").to_numpy(dtype=float)
    target_eff = pd.to_numeric(time_df[eff_col_time], errors="coerce").to_numpy(dtype=float)
    target_events = pd.to_numeric(time_df[time_events_col], errors="coerce").to_numpy(dtype=float)

    valid_basis = np.isfinite(basis_flux_all) & np.isfinite(basis_eff_all)
    valid_target = np.isfinite(target_flux) & np.isfinite(target_eff)
    if not np.any(valid_basis):
        log.error("No valid basis points in (%s, %s).", flux_col, eff_col_basis)
        return 1
    if not np.all(valid_target):
        log.error("Time series has invalid target points in (%s, %s).", flux_col, eff_col_time)
        return 1

    dictionary_work = basis_input_df.loc[valid_basis].reset_index(drop=True)
    basis_flux = basis_flux_all[valid_basis]
    basis_eff = basis_eff_all[valid_basis]
    basis_events = None if basis_events_all is None else basis_events_all[valid_basis]
    basis_parameter_set_col = _select_parameter_set_column(dictionary_work, basis_parameter_set_col_cfg)
    if basis_parameter_set_col is None:
        log.warning(
            "No parameter-set column found in basis; each row will be treated as its own parameter set."
        )
        parameter_set_values = np.asarray([f"row_{i}" for i in range(len(dictionary_work))], dtype=object)
    else:
        parameter_set_values = dictionary_work[basis_parameter_set_col].astype(str).to_numpy(dtype=object)

    one_per_set_mask, one_per_set_info = _build_one_per_parameter_set_mask(
        parameter_set_values=parameter_set_values,
        basis_events=basis_events,
        target_events=target_events,
        basis_flux=basis_flux,
        basis_eff=basis_eff,
        target_flux=target_flux,
        target_eff=target_eff,
    )

    # Enforce one-row-per-parameter-set, then apply optional event-count tolerance.
    event_mask_extra, extra_info = _build_event_mask(
        basis_events=basis_events,
        target_events=target_events,
        tolerance_pct=basis_events_tol_pct,
        min_rows=basis_min_rows,
    )
    if event_mask_extra is None:
        event_mask = one_per_set_mask
    else:
        event_mask = one_per_set_mask & event_mask_extra

    sel_counts = np.sum(event_mask, axis=1) if event_mask.size else np.array([], dtype=int)
    basis_filter_info = {
        "mode": "one_row_per_parameter_set_with_optional_event_tolerance",
        "parameter_set_mode": one_per_set_info.get("mode"),
        "n_parameter_sets": int(one_per_set_info.get("n_parameter_sets", 0)),
        "parameter_set_column": basis_parameter_set_col if basis_parameter_set_col is not None else "__row_index__",
        "basis_n_events_tolerance_pct_config": float(basis_events_tol_pct),
        "basis_min_rows_config": int(basis_min_rows),
        "event_tolerance_filter_info": extra_info,
        "allowed_rows_per_target_min": int(np.min(sel_counts)) if sel_counts.size else 0,
        "allowed_rows_per_target_median": float(np.median(sel_counts)) if sel_counts.size else 0.0,
        "allowed_rows_per_target_max": int(np.max(sel_counts)) if sel_counts.size else 0,
        "targets_with_zero_selected_rows": int(np.sum(sel_counts <= 0)) if sel_counts.size else 0,
    }
    log.info(
        "Basis source=%s rows=%d (valid_flux_eff=%d), one-per-set=%s(%s), tol_pct=%.3f -> selected/point min=%.0f med=%.1f max=%.0f (zero=%d).",
        basis_source,
        int(len(basis_input_df)),
        int(len(dictionary_work)),
        str(one_per_set_info.get("mode", "n/a")),
        str(basis_filter_info.get("parameter_set_column")),
        float(basis_events_tol_pct),
        float(basis_filter_info.get("allowed_rows_per_target_min", len(dictionary_work))),
        float(basis_filter_info.get("allowed_rows_per_target_median", len(dictionary_work))),
        float(basis_filter_info.get("allowed_rows_per_target_max", len(dictionary_work))),
        int(basis_filter_info.get("targets_with_zero_selected_rows", 0)),
    )

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
    top_k = None if top_k_raw in (None, "", 0) else _safe_int(top_k_raw, 8, minimum=1)
    distance_hardness = _safe_float(cfg_32.get("distance_hardness", 1.0), 1.0)
    density_enabled = _safe_bool(cfg_32.get("density_correction_enabled", True), True)
    density_k = _safe_int(cfg_32.get("density_correction_k_neighbors", 10), 10, minimum=1)
    density_exp = _safe_float(cfg_32.get("density_correction_exponent", 1.0), 1.0)
    density_clip_min = _safe_float(cfg_32.get("density_correction_clip_min", 0.25), 0.25)
    density_clip_max = _safe_float(cfg_32.get("density_correction_clip_max", 4.0), 4.0)
    if density_clip_max < density_clip_min:
        density_clip_max = density_clip_min
    density_scaling = None
    density_info = {"enabled": bool(density_enabled)}
    if density_enabled:
        density_scaling, density_info = _compute_inverse_density_scaling(
            basis_flux=basis_flux,
            basis_eff=basis_eff,
            k_neighbors=density_k,
            exponent=density_exp,
            clip_min=density_clip_min,
            clip_max=density_clip_max,
        )

    weights = _build_weights(
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
    # Diagnostic center in (flux, eff): weighted by the same basis weights used
    # for synthetic-column generation (includes density modulation when enabled).
    diagnostic_center_values = _weighted_numeric_columns(
        weights=weights,
        dict_df=dictionary_work,
        columns=[flux_col, eff_col_basis],
    )
    diagnostic_flux = np.asarray(
        diagnostic_center_values.get(flux_col, np.full(len(time_df), np.nan, dtype=float)),
        dtype=float,
    )
    diagnostic_eff = np.asarray(
        diagnostic_center_values.get(eff_col_basis, np.full(len(time_df), np.nan, dtype=float)),
        dtype=float,
    )
    diagnostic_center_label = (
        "Density-modulated weighted center (diagnostic)"
        if density_enabled
        else "Weighted center (diagnostic)"
    )
    m_diag_flux = np.isfinite(diagnostic_flux) & np.isfinite(target_flux)
    m_diag_eff = np.isfinite(diagnostic_eff) & np.isfinite(target_eff)
    diagnostic_flux_mae = (
        float(np.mean(np.abs(diagnostic_flux[m_diag_flux] - target_flux[m_diag_flux])))
        if np.any(m_diag_flux)
        else None
    )
    diagnostic_eff_mae = (
        float(np.mean(np.abs(diagnostic_eff[m_diag_eff] - target_eff[m_diag_eff])))
        if np.any(m_diag_eff)
        else None
    )
    log.info(
        "Diagnostic center check: flux MAE=%.6g, eff MAE=%.6g.",
        float(diagnostic_flux_mae) if diagnostic_flux_mae is not None else float("nan"),
        float(diagnostic_eff_mae) if diagnostic_eff_mae is not None else float("nan"),
    )

    # Flux/eff assigned to synthetic rows: fixed to STEP 3.1 discretized target.
    flux_linear = target_flux.copy()
    eff_linear = target_eff.copy()

    synthetic_df, dominant_idx = _make_synthetic_dataset(
        dictionary_df=dictionary_work,
        template_df=template_df,
        time_df=time_df,
        weights=weights,
        flux_col=flux_col,
        eff_col=eff_col_time,
        time_rate_col=time_rate_col,
        time_events_col=time_events_col,
        time_duration_col=time_duration_col,
        flux_output_values=flux_linear,
        eff_output_values=eff_linear,
    )

    # Output files
    out_synth = FILES_DIR / "synthetic_dataset.csv"
    synthetic_df.to_csv(out_synth, index=False)
    log.info("Wrote synthetic dataset: %s (%d rows)", out_synth, len(synthetic_df))

    # Highlight point for contribution diagnostics
    seed_cfg = cfg_32.get("random_seed", None)
    if seed_cfg in (None, "", "null", "None"):
        rng = np.random.default_rng()
        seed_used: int | None = None
    else:
        seed_used = _safe_int(seed_cfg, 0)
        rng = np.random.default_rng(seed_used)
    highlight_cfg = cfg_32.get("highlight_point_index", None)
    if highlight_cfg is None:
        highlight_idx = int(rng.integers(0, len(time_df)))
    else:
        highlight_idx = int(np.clip(_safe_int(highlight_cfg, 0), 0, len(time_df) - 1))

    contrib = weights[highlight_idx]
    contrib_df = pd.DataFrame({
        "rank": np.arange(1, len(contrib) + 1),
        "basis_index": np.arange(len(contrib)),
        "dictionary_index": np.arange(len(contrib)),
        "weight": contrib,
        "weight_pct": contrib * 100.0,
        "is_event_allowed": contrib > 0.0,
        flux_col: basis_flux,
        eff_col_basis: basis_eff,
        "basis_source": basis_source,
    })
    if basis_parameter_set_col is not None and basis_parameter_set_col in dictionary_work.columns:
        contrib_df["basis_parameter_set_id"] = dictionary_work[basis_parameter_set_col].astype(str)
    if "filename_base" in dictionary_work.columns:
        contrib_df["filename_base"] = dictionary_work["filename_base"].astype(str)
    if "events_per_second_global_rate" in dictionary_work.columns:
        contrib_df["events_per_second_global_rate"] = pd.to_numeric(
            dictionary_work["events_per_second_global_rate"], errors="coerce"
        )
    contrib_df = contrib_df.sort_values("weight", ascending=False).reset_index(drop=True)
    contrib_df["rank"] = np.arange(1, len(contrib_df) + 1)

    out_contrib = FILES_DIR / "highlight_point_contributions.csv"
    contrib_df.to_csv(out_contrib, index=False)
    log.info("Wrote highlight contributions: %s", out_contrib)

    # Plots
    out_plot_contrib = PLOTS_DIR / "dictionary_contributions_highlight.png"
    _plot_highlight_contributions(
        complete_df=complete_df,
        time_df=time_df,
        dictionary_df=dictionary_work,
        weights=weights,
        flux_col=flux_col,
        eff_col_time=eff_col_time,
        eff_col_dict=eff_col_basis,
        basis_label=basis_label,
        highlight_idx=highlight_idx,
        path=out_plot_contrib,
    )
    log.info("Wrote plot: %s", out_plot_contrib)

    out_plot_series = PLOTS_DIR / "synthetic_time_series_overview.png"
    _plot_time_series_overview(
        complete_df=complete_df,
        time_df=time_df,
        synthetic_df=synthetic_df,
        flux_col=flux_col,
        eff_col=eff_col_time,
        time_rate_col=time_rate_col,
        interpolated_flux=diagnostic_flux,
        interpolated_eff=diagnostic_eff,
        interpolated_label=diagnostic_center_label,
        path=out_plot_series,
    )
    log.info("Wrote plot: %s", out_plot_series)

    # Histogram: full dataset events vs selected basis subset events (same bins).
    if basis_events_col in template_df.columns:
        dataset_events_all = pd.to_numeric(template_df[basis_events_col], errors="coerce").to_numpy(dtype=float)
    else:
        dataset_events_all = np.array([], dtype=float)
    basis_selected_any = np.any(event_mask, axis=0)
    basis_subset_events = (
        np.asarray(basis_events, dtype=float)[basis_selected_any]
        if basis_events is not None and len(basis_events) == event_mask.shape[1]
        else np.array([], dtype=float)
    )
    basis_subset_events_finite = basis_subset_events[np.isfinite(basis_subset_events)]
    out_plot_events_hist = PLOTS_DIR / "events_count_dataset_vs_basis_subset.png"
    _plot_events_count_histogram(
        dataset_events=dataset_events_all,
        basis_subset_events=basis_subset_events,
        path=out_plot_events_hist,
    )
    log.info("Wrote plot: %s", out_plot_events_hist)

    # Summary
    effective_n = 1.0 / np.sum(weights * weights, axis=1)
    summary = {
        "time_series_csv": str(time_series_path),
        "complete_curve_csv": str(complete_curve_path if complete_curve_path.exists() else ""),
        "dictionary_csv": str(dictionary_path),
        "dataset_template_csv": str(template_path),
        "basis_source": basis_source,
        "basis_csv": str(basis_path),
        "basis_events_filter": basis_filter_info,
        "flux_eff_assignment_method": "target_discretized_from_step_3_1",
        "diagnostic_flux_eff_center_method": "weighted_basis_center_not_used_for_output",
        "diagnostic_flux_eff_center_label": diagnostic_center_label,
        "diagnostic_flux_center_mae_vs_target": diagnostic_flux_mae,
        "diagnostic_eff_center_mae_vs_target": diagnostic_eff_mae,
        "diagnostic_flux_center_range": [
            float(np.nanmin(diagnostic_flux)) if np.isfinite(diagnostic_flux).any() else None,
            float(np.nanmax(diagnostic_flux)) if np.isfinite(diagnostic_flux).any() else None,
        ],
        "diagnostic_eff_center_range": [
            float(np.nanmin(diagnostic_eff)) if np.isfinite(diagnostic_eff).any() else None,
            float(np.nanmax(diagnostic_eff)) if np.isfinite(diagnostic_eff).any() else None,
        ],
        "density_correction": density_info,
        "basis_subset_unique_rows_count": int(np.sum(basis_selected_any)),
        "basis_subset_unique_events_min": float(np.min(basis_subset_events_finite)) if basis_subset_events_finite.size else None,
        "basis_subset_unique_events_max": float(np.max(basis_subset_events_finite)) if basis_subset_events_finite.size else None,
        "basis_subset_unique_events_median": float(np.median(basis_subset_events_finite)) if basis_subset_events_finite.size else None,
        "n_time_points": int(len(time_df)),
        "n_basis_points": int(len(dictionary_work)),
        "n_dictionary_points": int(len(dictionary_work)),
        "n_synthetic_rows": int(len(synthetic_df)),
        "weighting_method": method,
        "sigma_flux": float(sigma_flux),
        "sigma_eff": float(sigma_eff),
        "distance_hardness": float(distance_hardness),
        "top_k": int(top_k) if top_k is not None else None,
        "highlight_point_index": int(highlight_idx),
        "highlight_random_seed": seed_used,
        "median_effective_contributors": float(np.nanmedian(effective_n)),
        "min_effective_contributors": float(np.nanmin(effective_n)),
        "max_effective_contributors": float(np.nanmax(effective_n)),
        "dominant_basis_unique_count": int(len(np.unique(dominant_idx))),
        "dominant_dictionary_unique_count": int(len(np.unique(dominant_idx))),
        "synthetic_flux_range": [
            float(pd.to_numeric(synthetic_df.get("flux_cm2_min"), errors="coerce").min()),
            float(pd.to_numeric(synthetic_df.get("flux_cm2_min"), errors="coerce").max()),
        ],
        "synthetic_eff_range": [
            float(pd.to_numeric(synthetic_df.get("eff_sim_1"), errors="coerce").min()),
            float(pd.to_numeric(synthetic_df.get("eff_sim_1"), errors="coerce").max()),
        ],
        "synthetic_global_rate_range_hz": [
            float(pd.to_numeric(synthetic_df.get("events_per_second_global_rate"), errors="coerce").min()),
            float(pd.to_numeric(synthetic_df.get("events_per_second_global_rate"), errors="coerce").max()),
        ],
    }
    out_summary = FILES_DIR / "synthetic_generation_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary: %s", out_summary)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
