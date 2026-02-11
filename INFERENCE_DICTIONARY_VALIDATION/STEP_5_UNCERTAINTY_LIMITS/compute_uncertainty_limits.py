#!/usr/bin/env python3
"""STEP_4: Calibrate uncertainty limits and dictionary coverage.

This script consumes STEP_3 all-mode outputs and produces:
1) uncertainty curves/limits vs sample event count
2) dictionary completeness diagnostics in (flux, eff_1) space
3) filling/coverage statistics for method-validity boundaries
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from msv_utils import (  # noqa: E402
    apply_clean_style,
    build_uncertainty_table,
    coerce_bool_series,
    convex_hull,
    load_config,
    maybe_log_x,
    min_distance_to_points,
    nearest_neighbor_distances,
    parse_list,
    polygon_area,
    resolve_param,
    setup_logger,
    setup_output_dirs,
)

log = setup_logger("STEP_5")
apply_clean_style()

DEFAULT_ALL_RESULTS = REPO_ROOT / "STEP_4_SELF_CONSISTENCY" / "OUTPUTS" / "FILES" / "all_samples_results.csv"
DEFAULT_DICT = REPO_ROOT / "STEP_1_BUILD_DICTIONARY" / "OUTPUTS" / "FILES" / "task_01" / "param_metadata_dictionary.csv"
DEFAULT_OUT = STEP_DIR
DEFAULT_CONFIG = STEP_DIR / "config.json"


# _load_config — replaced by msv_utils.load_config
# _parse_list  — replaced by msv_utils.parse_list

def _as_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _parse_eff_1(value: object) -> float:
    if isinstance(value, (list, tuple)) and len(value) >= 1:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return np.nan
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return np.nan
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 1:
            try:
                return float(parsed[0])
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def _prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"] == "ok"].copy()

    numeric_cols = [
        "sample_events_count",
        "flux_rel_error_pct",
        "eff_rel_error_pct",
        "abs_flux_rel_error_pct",
        "abs_eff_rel_error_pct",
        "true_flux_cm2_min",
        "true_eff_1",
        "estimated_flux_cm2_min",
        "estimated_eff_1",
    ]
    out = _as_numeric(out, numeric_cols)

    if "abs_flux_rel_error_pct" not in out.columns and "flux_rel_error_pct" in out.columns:
        out["abs_flux_rel_error_pct"] = out["flux_rel_error_pct"].abs()
    if "abs_eff_rel_error_pct" not in out.columns and "eff_rel_error_pct" in out.columns:
        out["abs_eff_rel_error_pct"] = out["eff_rel_error_pct"].abs()

    required = ["sample_events_count", "abs_flux_rel_error_pct", "abs_eff_rel_error_pct"]
    for col in required:
        if col not in out.columns:
            raise KeyError(f"Required column missing in all-results table: {col}")

    out = out.dropna(subset=required).copy()
    out = out[out["sample_events_count"] > 0].copy()
    return out


# _build_uncertainty_table — replaced by msv_utils.build_uncertainty_table

def _find_required_events(
    events: pd.Series,
    errors: pd.Series,
    *,
    target_error_pct: float,
    quantile: float,
    min_points: int,
) -> float | None:
    e = pd.to_numeric(events, errors="coerce")
    err = pd.to_numeric(errors, errors="coerce")
    mask = e.notna() & err.notna() & e.gt(0)
    e = e[mask]
    err = err[mask]
    if len(e) < min_points:
        return None

    candidates = np.unique(e.quantile(np.linspace(0.0, 0.95, 60)).to_numpy(dtype=float))
    candidates = candidates[np.isfinite(candidates)]
    if len(candidates) == 0:
        return None

    for threshold in np.sort(candidates):
        sel = err[e >= threshold]
        if len(sel) < min_points:
            continue
        q_val = float(np.nanquantile(sel.to_numpy(dtype=float), quantile))
        if q_val <= target_error_pct:
            return float(threshold)
    return None


# _convex_hull, _polygon_area, _nearest_neighbor_distances, _min_distance_to_points
# — replaced by msv_utils equivalents

def _prepare_dictionary_points(dictionary_df: pd.DataFrame) -> pd.DataFrame:
    out = dictionary_df.copy()
    if "flux_cm2_min" not in out.columns:
        raise KeyError("Dictionary CSV missing flux_cm2_min column.")

    if "eff_1" not in out.columns:
        if "efficiencies" not in out.columns:
            raise KeyError("Dictionary CSV missing both eff_1 and efficiencies columns.")
        out["eff_1"] = out["efficiencies"].apply(_parse_eff_1)

    out = _as_numeric(out, ["flux_cm2_min", "eff_1"])
    out = out.dropna(subset=["flux_cm2_min", "eff_1"]).copy()
    return out


def _compute_dictionary_coverage(
    points_df: pd.DataFrame,
    *,
    grid_size: int,
    coverage_radii: list[float],
    mc_points: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    flux = points_df["flux_cm2_min"].to_numpy(dtype=float)
    eff = points_df["eff_1"].to_numpy(dtype=float)

    flux_min, flux_max = float(np.min(flux)), float(np.max(flux))
    eff_min, eff_max = float(np.min(eff)), float(np.max(eff))
    flux_span = flux_max - flux_min
    eff_span = eff_max - eff_min

    if flux_span <= 0 or eff_span <= 0:
        raise ValueError("Dictionary flux/eff range is degenerate; cannot compute coverage metrics.")

    xy_norm = np.column_stack([
        (flux - flux_min) / flux_span,
        (eff - eff_min) / eff_span,
    ])

    unique_xy = np.unique(xy_norm, axis=0)
    n_unique = int(len(unique_xy))

    grid_n = max(2, int(grid_size))
    ix = np.floor(unique_xy[:, 0] * grid_n).astype(int)
    iy = np.floor(unique_xy[:, 1] * grid_n).astype(int)
    ix = np.clip(ix, 0, grid_n - 1)
    iy = np.clip(iy, 0, grid_n - 1)
    occupied = np.unique(ix * grid_n + iy)
    grid_fill_pct = float(100.0 * len(occupied) / (grid_n * grid_n))

    hull = convex_hull(unique_xy)
    hull_area_pct_bbox = float(100.0 * polygon_area(hull))

    nn_d = nearest_neighbor_distances(unique_xy)
    nn_valid = nn_d[np.isfinite(nn_d)]
    nn_stats = {
        "nn_dist_norm_median": float(np.nanmedian(nn_valid)) if len(nn_valid) else np.nan,
        "nn_dist_norm_p68": float(np.nanpercentile(nn_valid, 68)) if len(nn_valid) else np.nan,
        "nn_dist_norm_p95": float(np.nanpercentile(nn_valid, 95)) if len(nn_valid) else np.nan,
        "nn_dist_norm_mean": float(np.nanmean(nn_valid)) if len(nn_valid) else np.nan,
    }

    rng = np.random.default_rng(seed)
    q = rng.random((max(1000, int(mc_points)), 2), dtype=float)
    q_min_d = min_distance_to_points(q, unique_xy)

    radii = sorted(r for r in coverage_radii if np.isfinite(r) and r > 0)
    if not radii:
        radii = [0.01, 0.02, 0.03, 0.05]

    radius_rows = []
    for r in radii:
        covered_pct = float(100.0 * np.mean(q_min_d <= r))
        radius_rows.append(
            {
                "radius_norm": float(r),
                "covered_fraction_pct": covered_pct,
            }
        )
    radius_df = pd.DataFrame(radius_rows)

    coverage_metrics = {
        "n_points_total": int(len(points_df)),
        "n_points_unique_flux_eff": n_unique,
        "flux_min": flux_min,
        "flux_max": flux_max,
        "eff_min": eff_min,
        "eff_max": eff_max,
        "grid_size": grid_n,
        "grid_fill_pct": grid_fill_pct,
        "convex_hull_area_pct_of_bbox": hull_area_pct_bbox,
        **nn_stats,
    }
    return coverage_metrics, radius_df, nn_valid


# _maybe_log_x — replaced by msv_utils.maybe_log_x


def _plot_uncertainty_bands(unc_df: pd.DataFrame, raw_df: pd.DataFrame, path: Path) -> None:
    """Percentile bands with raw scatter underlay from *raw_df*."""
    if unc_df.empty:
        return
    x_bands = pd.to_numeric(unc_df["events_median"], errors="coerce")
    raw_x = pd.to_numeric(raw_df["sample_events_count"], errors="coerce") if raw_df is not None else pd.Series(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for idx, (tag, col_prefix, ylabel) in enumerate([
        ("Flux", "flux", "|Flux relative error| [%]"),
        ("Efficiency", "eff", "|Efficiency relative error| [%]"),
    ]):
        ax = axes[idx]
        raw_y_col = f"abs_{col_prefix}_rel_error_pct"
        raw_y = pd.to_numeric(raw_df.get(raw_y_col), errors="coerce") if raw_df is not None else pd.Series(dtype=float)
        mask = raw_x.notna() & raw_y.notna()
        if mask.sum() > 0:
            ax.scatter(raw_x[mask], raw_y[mask], s=10, alpha=0.20, color="#AAAAAA", zorder=1,
                       label="individual samples")

        ax.plot(x_bands, unc_df[f"{col_prefix}_abs_rel_err_pct_p50"], "o-", label="p50", color="#4C78A8", zorder=2)
        ax.plot(x_bands, unc_df[f"{col_prefix}_abs_rel_err_pct_p68"], "o-", label="p68", color="#F58518", zorder=2)
        ax.plot(x_bands, unc_df[f"{col_prefix}_abs_rel_err_pct_p95"], "o-", label="p95", color="#E45756", zorder=2)
        ax.set_title(f"{tag} uncertainty vs sample size")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Median events in bin")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        maybe_log_x(ax, x_bands)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dictionary_overview(df: pd.DataFrame, scatter_title: str, path: Path) -> None:
    """Side-by-side scatter + hexbin of dictionary points in flux-eff space."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(df["flux_cm2_min"], df["eff_1"], s=10, alpha=0.45, color="#54A24B")
    axes[0].set_xlabel("Flux [cm^-2 min^-1]")
    axes[0].set_ylabel("eff_1")
    axes[0].set_title(scatter_title)
    axes[0].grid(True, alpha=0.2)

    hb = axes[1].hexbin(
        df["flux_cm2_min"], df["eff_1"],
        gridsize=40, mincnt=1, cmap="viridis",
    )
    fig.colorbar(hb, ax=axes[1], label="Points per hex")
    axes[1].set_xlabel("Flux [cm^-2 min^-1]")
    axes[1].set_ylabel("eff_1")
    axes[1].set_title("Dictionary density in flux-eff space")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dictionary_coverage(nn_dist: np.ndarray, radius_df: pd.DataFrame, path: Path) -> None:
    """Side-by-side NN distance histogram + coverage-vs-radius curve."""
    vals = pd.Series(nn_dist).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if not vals.empty:
        axes[0].hist(vals, bins=50, color="#F58518", alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Nearest-neighbor distance (normalized flux-eff space)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Dictionary nearest-neighbor spacing")
    axes[0].grid(True, alpha=0.2)

    if not radius_df.empty:
        axes[1].plot(
            radius_df["radius_norm"],
            radius_df["covered_fraction_pct"],
            "o-", color="#4C78A8",
        )
    axes[1].set_xlabel("Coverage radius (normalized units)")
    axes[1].set_ylabel("Covered random points [%]")
    axes[1].set_title("Dictionary filling by distance-based coverage")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _event_filter(df: pd.DataFrame, threshold: float, mode: str) -> pd.DataFrame:
    if "sample_events_count" not in df.columns:
        return df.iloc[0:0].copy()
    events = pd.to_numeric(df["sample_events_count"], errors="coerce")
    if mode == "ge":
        return df.loc[events.ge(threshold)].copy()
    if mode == "lt":
        return df.loc[events.lt(threshold)].copy()
    raise ValueError(f"Unsupported event-filter mode: {mode}")


def _compute_plane_mean(
    df: pd.DataFrame,
    *,
    value_col: str,
    flux_col: str = "true_flux_cm2_min",
    eff_col: str = "true_eff_1",
    flux_bins: int = 24,
    eff_bins: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    work = df[[flux_col, eff_col, value_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if work.empty:
        return np.array([]), np.array([]), np.array([[]]), np.array([[]])

    x = work[flux_col].to_numpy(dtype=float)
    y = work[eff_col].to_numpy(dtype=float)
    z = work[value_col].to_numpy(dtype=float)

    x_edges = np.linspace(float(np.min(x)), float(np.max(x)), max(2, flux_bins) + 1)
    y_edges = np.linspace(float(np.min(y)), float(np.max(y)), max(2, eff_bins) + 1)

    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    z_sum, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=z)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_mean = z_sum / count
    z_mean[count == 0] = np.nan
    return x_edges, y_edges, z_mean, count


def _plot_plane_mean_error_combined(
    df: pd.DataFrame,
    *,
    title: str,
    path: Path,
    flux_bins: int,
    eff_bins: int,
) -> None:
    """1×2 figure: left = mean |flux error| heatmap, right = mean |eff error| heatmap."""
    pairs = [
        ("abs_flux_rel_error_pct", "Mean |flux rel. error| [%]"),
        ("abs_eff_rel_error_pct", "Mean |eff rel. error| [%]"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (value_col, cbar_label) in enumerate(pairs):
        ax = axes[idx]
        x_edges, y_edges, z_mean, counts = _compute_plane_mean(
            df,
            value_col=value_col,
            flux_bins=flux_bins,
            eff_bins=eff_bins,
        )
        if z_mean.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        finite = pd.Series(z_mean.ravel()).dropna()
        if finite.empty:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        vmax = float(np.nanpercentile(finite.to_numpy(dtype=float), 95))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = None

        mesh = ax.pcolormesh(
            x_edges, y_edges, z_mean.T, shading="auto", cmap="viridis",
            vmin=0, vmax=vmax,
        )
        fig.colorbar(mesh, ax=ax, label=cbar_label)
        ax.set_xlabel("True flux [cm^-2 min^-1]")
        ax.set_ylabel("True eff_1")
        ax.set_title(cbar_label)
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _build_fixed_bins_table(df: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    events = pd.to_numeric(df["sample_events_count"], errors="coerce")
    if events.dropna().empty:
        return pd.DataFrame()

    clean_edges = sorted(set(float(v) for v in edges if np.isfinite(v)))
    if len(clean_edges) < 2:
        return pd.DataFrame()

    bins = pd.cut(events, bins=clean_edges, include_lowest=True, duplicates="drop")
    rows = []
    categories = bins.cat.categories if hasattr(bins, "cat") else []
    for iv in categories:
        part = df.loc[bins == iv]
        flux = pd.to_numeric(part.get("abs_flux_rel_error_pct"), errors="coerce").dropna()
        eff = pd.to_numeric(part.get("abs_eff_rel_error_pct"), errors="coerce").dropna()
        rows.append(
            {
                "events_bin": str(iv),
                "n_samples": int(len(part)),
                "events_min": float(iv.left),
                "events_max": float(iv.right),
                "flux_abs_rel_err_pct_p68": float(np.nanpercentile(flux, 68)) if len(flux) else np.nan,
                "flux_abs_rel_err_pct_p95": float(np.nanpercentile(flux, 95)) if len(flux) else np.nan,
                "eff_abs_rel_err_pct_p68": float(np.nanpercentile(eff, 68)) if len(eff) else np.nan,
                "eff_abs_rel_err_pct_p95": float(np.nanpercentile(eff, 95)) if len(eff) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# _coerce_bool_series — replaced by msv_utils.coerce_bool_series


def _build_membership_uncertainty_table(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_in_dictionary" not in df.columns:
        return pd.DataFrame()
    membership = coerce_bool_series(df["sample_in_dictionary"])
    rows: list[dict[str, object]] = []
    for label, flag in (("in_dictionary", True), ("out_dictionary", False)):
        part = df.loc[membership == flag]
        flux = pd.to_numeric(part.get("abs_flux_rel_error_pct"), errors="coerce").dropna()
        eff = pd.to_numeric(part.get("abs_eff_rel_error_pct"), errors="coerce").dropna()
        rows.append(
            {
                "subset": label,
                "n_samples": int(len(part)),
                "flux_abs_rel_err_pct_p50": float(np.nanpercentile(flux, 50)) if len(flux) else np.nan,
                "flux_abs_rel_err_pct_p68": float(np.nanpercentile(flux, 68)) if len(flux) else np.nan,
                "flux_abs_rel_err_pct_p95": float(np.nanpercentile(flux, 95)) if len(flux) else np.nan,
                "eff_abs_rel_err_pct_p50": float(np.nanpercentile(eff, 50)) if len(eff) else np.nan,
                "eff_abs_rel_err_pct_p68": float(np.nanpercentile(eff, 68)) if len(eff) else np.nan,
                "eff_abs_rel_err_pct_p95": float(np.nanpercentile(eff, 95)) if len(eff) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_threshold_sweep(df: pd.DataFrame, n_points: int, min_count: int) -> pd.DataFrame:
    events = pd.to_numeric(df["sample_events_count"], errors="coerce")
    events = events.dropna()
    events = events[events > 0]
    if len(events) < max(5, min_count):
        return pd.DataFrame()

    qs = np.linspace(0.0, 0.95, max(5, int(n_points)))
    thresholds = np.unique(events.quantile(qs).to_numpy(dtype=float))
    rows = []
    for thr in np.sort(thresholds):
        part = df.loc[pd.to_numeric(df["sample_events_count"], errors="coerce").ge(thr)].copy()
        flux = pd.to_numeric(part.get("abs_flux_rel_error_pct"), errors="coerce").dropna()
        eff = pd.to_numeric(part.get("abs_eff_rel_error_pct"), errors="coerce").dropna()
        if len(part) < min_count or len(flux) < min_count or len(eff) < min_count:
            continue
        rows.append(
            {
                "threshold_events": float(thr),
                "n_samples": int(len(part)),
                "flux_abs_rel_err_pct_p68": float(np.nanpercentile(flux, 68)),
                "flux_abs_rel_err_pct_p95": float(np.nanpercentile(flux, 95)),
                "eff_abs_rel_err_pct_p68": float(np.nanpercentile(eff, 68)),
                "eff_abs_rel_err_pct_p95": float(np.nanpercentile(eff, 95)),
            }
        )
    return pd.DataFrame(rows)


def _plot_threshold_sweep(sweep_df: pd.DataFrame, path: Path) -> None:
    if sweep_df.empty:
        return
    x = pd.to_numeric(sweep_df["threshold_events"], errors="coerce")
    n = pd.to_numeric(sweep_df["n_samples"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    axes[0].plot(x, sweep_df["flux_abs_rel_err_pct_p68"], "o-", color="#4C78A8", label="p68")
    axes[0].plot(x, sweep_df["flux_abs_rel_err_pct_p95"], "o-", color="#E45756", label="p95")
    axes[0].set_title("Flux error vs min-events threshold")
    axes[0].set_ylabel("|Flux relative error| [%]")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    axes[1].plot(x, sweep_df["eff_abs_rel_err_pct_p68"], "o-", color="#4C78A8", label="p68")
    axes[1].plot(x, sweep_df["eff_abs_rel_err_pct_p95"], "o-", color="#E45756", label="p95")
    axes[1].set_title("Efficiency error vs min-events threshold")
    axes[1].set_ylabel("|Efficiency relative error| [%]")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(x, n, "o-", color="#54A24B")
    axes[2].set_title("Remaining samples vs threshold")
    axes[2].set_ylabel("n samples")
    axes[2].grid(True, alpha=0.2)

    for ax in axes:
        ax.set_xlabel("Minimum sample events threshold")
        maybe_log_x(ax, x)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_sample_size_distribution(df: pd.DataFrame, path: Path, threshold: float) -> None:
    events = pd.to_numeric(df["sample_events_count"], errors="coerce").dropna()
    events = events[events > 0]
    if events.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(events, bins=50, color="#4C78A8", alpha=0.85, edgecolor="white")
    axes[0].axvline(threshold, color="#E45756", linestyle="--", linewidth=1.2, label=f"threshold={threshold:g}")
    axes[0].set_xlabel("Sample events count")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sample-size distribution")
    axes[0].grid(True, alpha=0.2)
    maybe_log_x(axes[0], events)
    axes[0].legend()

    s = np.sort(events.to_numpy(dtype=float))
    cdf = np.arange(1, len(s) + 1) / len(s)
    axes[1].step(s, cdf, where="post", color="#F58518", linewidth=1.5)
    axes[1].axvline(threshold, color="#E45756", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Sample events count")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Sample-size cumulative distribution")
    axes[1].grid(True, alpha=0.2)
    maybe_log_x(axes[1], events)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_residual_overlay(
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    *,
    error_col: str,
    xlabel: str,
    title: str,
    path: Path,
    threshold: float,
) -> None:
    """Overlay high-stat and low-stat residual distributions in one figure."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for subset_df, label, color in (
        (high_df, f"events >= {threshold:g}", "#4C78A8"),
        (low_df, f"events < {threshold:g}", "#E45756"),
    ):
        vals = pd.to_numeric(subset_df.get(error_col), errors="coerce").dropna()
        if vals.empty:
            continue
        # Clip to 1st–99th percentile for readable axis range
        q01, q99 = float(vals.quantile(0.01)), float(vals.quantile(0.99))
        span = max(abs(q01), abs(q99))
        clipped = vals[(vals >= -span) & (vals <= span)]
        ax.hist(
            clipped, bins=60, density=True, alpha=0.55,
            color=color, edgecolor="white", linewidth=0.4,
            label=f"{label} (n={len(vals)})",
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Coverage-error correlation (to_do.md §5.3)
# ---------------------------------------------------------------------------

def _compute_coverage_error_correlation(df: pd.DataFrame) -> dict:
    """Pearson and Spearman correlation between distance-to-dict and errors."""
    from scipy import stats as _stats

    dist = pd.to_numeric(df.get("sample_to_dict_dist_norm"), errors="coerce")
    results: dict[str, object] = {}
    for error_col in ("abs_flux_rel_error_pct", "abs_eff_rel_error_pct"):
        err = pd.to_numeric(df.get(error_col), errors="coerce")
        mask = dist.notna() & err.notna()
        if mask.sum() < 10:
            results[error_col] = {"n": int(mask.sum()), "note": "too few points"}
            continue
        d = dist[mask].to_numpy(dtype=float)
        e = err[mask].to_numpy(dtype=float)
        pr, pp = float(np.corrcoef(d, e)[0, 1]), np.nan
        try:
            sr = _stats.spearmanr(d, e)
            results[error_col] = {
                "n": int(mask.sum()),
                "pearson_r": pr,
                "spearman_r": float(sr.correlation),
                "spearman_p": float(sr.pvalue),
            }
        except Exception:
            results[error_col] = {"n": int(mask.sum()), "pearson_r": pr}
    return results


# ---------------------------------------------------------------------------
# Monotonicity check (to_do.md §5.2)
# ---------------------------------------------------------------------------

def _check_monotonicity(unc_df: pd.DataFrame) -> dict:
    """Check that error quantiles generally decrease with event count."""
    if unc_df.empty:
        return {"checked": False}
    x = pd.to_numeric(unc_df.get("events_median"), errors="coerce")
    results: dict[str, object] = {"checked": True}
    for col in ("flux_abs_rel_err_pct_p68", "flux_abs_rel_err_pct_p95",
                "eff_abs_rel_err_pct_p68", "eff_abs_rel_err_pct_p95"):
        if col not in unc_df.columns:
            continue
        y = pd.to_numeric(unc_df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        xv = x[mask].to_numpy(dtype=float)
        yv = y[mask].to_numpy(dtype=float)
        order = np.argsort(xv)
        yv_sorted = yv[order]
        diffs = np.diff(yv_sorted)
        n_increasing = int((diffs > 0).sum())
        n_decreasing = int((diffs < 0).sum())
        is_monotonic = n_increasing == 0
        # Detect error floor: last 3 bins have similar values
        floor_val = None
        if len(yv_sorted) >= 3:
            tail = yv_sorted[-3:]
            if np.std(tail) < 0.1 * np.mean(tail):
                floor_val = float(np.mean(tail))
        results[col] = {
            "n_bins": int(mask.sum()),
            "n_increasing_steps": n_increasing,
            "n_decreasing_steps": n_decreasing,
            "is_monotonically_decreasing": bool(is_monotonic),
            "error_floor": floor_val,
        }
    return results


# ---------------------------------------------------------------------------
# Validity mask (to_do.md §6.3)
# ---------------------------------------------------------------------------

def _build_validity_mask(
    df: pd.DataFrame,
    limits_df: pd.DataFrame,
    *,
    target_error_pct: float,
    target_quantile: float,
    max_dict_dist: float | None = None,
) -> pd.Series:
    """Compose a trust-region mask from event thresholds + distance limit."""
    mask = pd.Series(True, index=df.index)

    # Event threshold from limits table
    match = limits_df[
        (limits_df["target_error_pct"] == target_error_pct)
        & (limits_df["quantile"] == target_quantile)
    ]
    if not match.empty:
        req_flux = match.iloc[0].get("required_events_flux")
        req_eff = match.iloc[0].get("required_events_eff")
        events = pd.to_numeric(df.get("sample_events_count"), errors="coerce")
        threshold = max(
            float(req_flux) if pd.notna(req_flux) else 0,
            float(req_eff) if pd.notna(req_eff) else 0,
        )
        if threshold > 0:
            mask = mask & events.ge(threshold)

    # Distance constraint
    if max_dict_dist is not None:
        dist = pd.to_numeric(df.get("sample_to_dict_dist_norm"), errors="coerce")
        mask = mask & (dist.le(max_dict_dist) | dist.isna())

    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP_4 uncertainty and coverage diagnostics.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--all-results-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--events-bins", type=int, default=None)
    parser.add_argument("--min-bin-count", type=int, default=None)
    parser.add_argument("--target-error-pct", default=None,
                        help="Comma-separated target relative errors in percent.")
    parser.add_argument("--target-quantiles", default=None,
                        help="Comma-separated quantiles in [0,1], e.g. 0.68,0.95.")
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--coverage-radii", default=None,
                        help="Comma-separated normalized radii for coverage filling metric.")
    parser.add_argument("--mc-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plane-min-events", type=float, default=None,
                        help="Event threshold for high-stat flux-eff error maps (default 40000).")
    parser.add_argument("--plane-flux-bins", type=int, default=None)
    parser.add_argument("--plane-eff-bins", type=int, default=None)
    parser.add_argument("--events-fixed-edges", default=None,
                        help="Comma-separated fixed event-bin edges for uncertainty table.")
    parser.add_argument("--sweep-points", type=int, default=None,
                        help="Number of threshold points for min-event sweep plot/table.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    def _rp(cli, key, default):
        return resolve_param(cli, config, key, default)

    all_results_csv = Path(_rp(args.all_results_csv, "all_results_csv", str(DEFAULT_ALL_RESULTS)))
    dictionary_csv = Path(_rp(args.dictionary_csv, "dictionary_csv", str(DEFAULT_DICT)))
    out_base = Path(_rp(args.out_dir, "out_dir", str(DEFAULT_OUT)))
    events_bins = int(_rp(args.events_bins, "events_bins", 10))
    min_bin_count = int(_rp(args.min_bin_count, "min_bin_count", 30))
    target_error_pct = parse_list(
        _rp(args.target_error_pct, "target_error_pct", [1.0, 2.0, 5.0]),
        cast=float,
    )
    target_quantiles = parse_list(
        _rp(args.target_quantiles, "target_quantiles", [0.68, 0.95]),
        cast=float,
    )
    grid_size = int(_rp(args.grid_size, "grid_size", 40))
    coverage_radii = parse_list(
        _rp(args.coverage_radii, "coverage_radii", [0.01, 0.02, 0.03, 0.05]),
        cast=float,
    )
    mc_points = int(_rp(args.mc_points, "mc_points", 20000))
    seed = int(_rp(args.seed, "seed", 123))
    plane_min_events = float(_rp(args.plane_min_events, "plane_min_events", 40000))
    plane_flux_bins = int(_rp(args.plane_flux_bins, "plane_flux_bins", 24))
    plane_eff_bins = int(_rp(args.plane_eff_bins, "plane_eff_bins", 24))
    fixed_edges = parse_list(
        _rp(args.events_fixed_edges, "events_fixed_edges",
            [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 70000, 100000]),
        cast=float,
    )
    sweep_points = int(_rp(args.sweep_points, "sweep_points", 40))

    if not all_results_csv.exists():
        raise FileNotFoundError(f"All-results CSV not found: {all_results_csv}")
    if not dictionary_csv.exists():
        raise FileNotFoundError(f"Dictionary CSV not found: {dictionary_csv}")

    files_dir, plots_dir = setup_output_dirs(out_base)
    out_dir = files_dir  # CSVs/JSONs go into OUTPUTS/FILES

    log.info("Loading all-mode results: %s", all_results_csv)
    results_df = pd.read_csv(all_results_csv, low_memory=False)
    results_ok = _prepare_results(results_df)
    if results_ok.empty:
        raise ValueError("No valid rows found in all-results table after filtering.")

    log.info("Valid rows for uncertainty calibration: %d", len(results_ok))
    uncertainty_df = build_uncertainty_table(results_ok, n_bins=events_bins, min_bin_count=min_bin_count)
    uncertainty_csv = out_dir / "uncertainty_by_events.csv"
    uncertainty_df.to_csv(uncertainty_csv, index=False)
    fixed_bins_df = _build_fixed_bins_table(results_ok, fixed_edges)
    fixed_bins_csv = out_dir / "uncertainty_fixed_event_bins.csv"
    fixed_bins_df.to_csv(fixed_bins_csv, index=False)
    membership_unc_df = _build_membership_uncertainty_table(results_ok)
    membership_unc_csv = out_dir / "uncertainty_by_dictionary_membership.csv"
    membership_unc_df.to_csv(membership_unc_csv, index=False)
    sweep_df = _build_threshold_sweep(results_ok, n_points=sweep_points, min_count=min_bin_count)
    sweep_csv = out_dir / "threshold_sweep_min_events.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    limits_rows = []
    for q in target_quantiles:
        if not (0 < q < 1):
            continue
        for target in target_error_pct:
            if target <= 0:
                continue
            flux_ev = _find_required_events(
                results_ok["sample_events_count"],
                results_ok["abs_flux_rel_error_pct"],
                target_error_pct=target,
                quantile=float(q),
                min_points=min_bin_count,
            )
            eff_ev = _find_required_events(
                results_ok["sample_events_count"],
                results_ok["abs_eff_rel_error_pct"],
                target_error_pct=target,
                quantile=float(q),
                min_points=min_bin_count,
            )
            limits_rows.append(
                {
                    "quantile": float(q),
                    "target_error_pct": float(target),
                    "required_events_flux": flux_ev,
                    "required_events_eff": eff_ev,
                }
            )

    limits_df = pd.DataFrame(limits_rows)
    limits_csv = out_dir / "validity_limits_by_target.csv"
    limits_df.to_csv(limits_csv, index=False)

    log.info("Loading dictionary: %s", dictionary_csv)
    dict_df = pd.read_csv(dictionary_csv, low_memory=False)
    dict_points = _prepare_dictionary_points(dict_df)

    coverage_metrics, radius_df, nn_dist = _compute_dictionary_coverage(
        dict_points,
        grid_size=grid_size,
        coverage_radii=coverage_radii,
        mc_points=mc_points,
        seed=seed,
    )

    radius_csv = out_dir / "dictionary_coverage_by_radius.csv"
    radius_df.to_csv(radius_csv, index=False)

    results_plus = results_ok.copy()
    dict_flux = dict_points["flux_cm2_min"].to_numpy(dtype=float)
    dict_eff = dict_points["eff_1"].to_numpy(dtype=float)
    flux_min = float(coverage_metrics["flux_min"])
    flux_max = float(coverage_metrics["flux_max"])
    eff_min = float(coverage_metrics["eff_min"])
    eff_max = float(coverage_metrics["eff_max"])
    flux_span = flux_max - flux_min
    eff_span = eff_max - eff_min
    dict_xy_norm = np.unique(
        np.column_stack([
            (dict_flux - flux_min) / flux_span,
            (dict_eff - eff_min) / eff_span,
        ]),
        axis=0,
    )
    results_plus["sample_to_dict_dist_norm"] = np.nan
    if {"true_flux_cm2_min", "true_eff_1"}.issubset(results_plus.columns):
        sample_xy = results_plus[["true_flux_cm2_min", "true_eff_1"]].apply(pd.to_numeric, errors="coerce")
        valid_mask = sample_xy.notna().all(axis=1)
        if valid_mask.any():
            sample_xy_norm = np.column_stack([
                (sample_xy.loc[valid_mask, "true_flux_cm2_min"].to_numpy(dtype=float) - flux_min) / flux_span,
                (sample_xy.loc[valid_mask, "true_eff_1"].to_numpy(dtype=float) - eff_min) / eff_span,
            ])
            sample_dist = min_distance_to_points(sample_xy_norm, dict_xy_norm)
            results_plus.loc[valid_mask, "sample_to_dict_dist_norm"] = sample_dist
    dist_series = pd.to_numeric(results_plus["sample_to_dict_dist_norm"], errors="coerce").dropna()
    if not dist_series.empty and float(dist_series.max()) <= 1e-12:
        log.warning(
            "All evaluated samples lie on dictionary support points "
            "(distance-to-dictionary ~ 0). STEP_4 limits may be optimistic for out-of-dictionary inputs."
        )
    results_plus_csv = out_dir / "all_samples_success_with_distance.csv"
    results_plus.to_csv(results_plus_csv, index=False)

    events_series = pd.to_numeric(results_ok["sample_events_count"], errors="coerce").dropna()
    n_low = int((events_series < plane_min_events).sum()) if not events_series.empty else 0
    n_high = int((events_series >= plane_min_events).sum()) if not events_series.empty else 0
    if n_low == 0:
        log.warning(
            "No low-stat samples under %g events in input table. "
            "To validate low-stat behavior, re-run STEP_3 --all with a lower-stat reference dataset.",
            plane_min_events,
        )
    membership = coerce_bool_series(results_ok["sample_in_dictionary"]) \
        if "sample_in_dictionary" in results_ok.columns else pd.Series(np.nan, index=results_ok.index)
    n_in_dict = int((membership == True).sum())
    n_out_dict = int((membership == False).sum())
    if "sample_in_dictionary" in results_ok.columns and n_out_dict == 0:
        log.warning(
            "STEP_3 all-mode input contains no out-of-dictionary samples. "
            "Validation may still be optimistic outside dictionary support."
        )

    summary = {
        "input_all_results_csv": str(all_results_csv),
        "input_dictionary_csv": str(dictionary_csv),
        "n_all_results_rows": int(len(results_df)),
        "n_valid_uncertainty_rows": int(len(results_ok)),
        "n_samples_in_dictionary": n_in_dict if "sample_in_dictionary" in results_ok.columns else None,
        "n_samples_out_dictionary": n_out_dict if "sample_in_dictionary" in results_ok.columns else None,
        "sample_events_min": float(events_series.min()) if not events_series.empty else None,
        "sample_events_p05": float(events_series.quantile(0.05)) if not events_series.empty else None,
        "sample_events_p50": float(events_series.quantile(0.50)) if not events_series.empty else None,
        "sample_events_p95": float(events_series.quantile(0.95)) if not events_series.empty else None,
        "sample_events_max": float(events_series.max()) if not events_series.empty else None,
        "plane_min_events_threshold": float(plane_min_events),
        "n_samples_lt_plane_threshold": n_low,
        "n_samples_ge_plane_threshold": n_high,
        "events_bins": int(events_bins),
        "min_bin_count": int(min_bin_count),
        "target_error_pct": [float(v) for v in target_error_pct],
        "target_quantiles": [float(v) for v in target_quantiles],
        "events_fixed_edges": [float(v) for v in fixed_edges],
        "sweep_points": int(sweep_points),
        "uncertainty_global": {
            "flux_abs_rel_err_pct_p50": float(np.nanpercentile(results_ok["abs_flux_rel_error_pct"], 50)),
            "flux_abs_rel_err_pct_p68": float(np.nanpercentile(results_ok["abs_flux_rel_error_pct"], 68)),
            "flux_abs_rel_err_pct_p95": float(np.nanpercentile(results_ok["abs_flux_rel_error_pct"], 95)),
            "eff_abs_rel_err_pct_p50": float(np.nanpercentile(results_ok["abs_eff_rel_error_pct"], 50)),
            "eff_abs_rel_err_pct_p68": float(np.nanpercentile(results_ok["abs_eff_rel_error_pct"], 68)),
            "eff_abs_rel_err_pct_p95": float(np.nanpercentile(results_ok["abs_eff_rel_error_pct"], 95)),
        },
        "distance_to_dictionary_global": {
            "dist_norm_p50": float(np.nanpercentile(results_plus["sample_to_dict_dist_norm"], 50))
                if results_plus["sample_to_dict_dist_norm"].notna().any() else None,
            "dist_norm_p95": float(np.nanpercentile(results_plus["sample_to_dict_dist_norm"], 95))
                if results_plus["sample_to_dict_dist_norm"].notna().any() else None,
        },
        "dictionary_coverage": coverage_metrics,
    }

    # ---- Coverage-error correlation (to_do.md §5.3) ----
    try:
        corr_info = _compute_coverage_error_correlation(results_plus)
        summary["coverage_error_correlation"] = corr_info
        log.info("Coverage-error correlation: %s", corr_info)
    except Exception as exc:
        log.warning("Coverage-error correlation skipped: %s", exc)
        summary["coverage_error_correlation"] = {"error": str(exc)}

    # ---- Monotonicity check (to_do.md §5.2) ----
    mono_info = _check_monotonicity(uncertainty_df)
    summary["monotonicity_check"] = mono_info
    if mono_info.get("checked"):
        for col_name, col_data in mono_info.items():
            if isinstance(col_data, dict) and col_data.get("error_floor") is not None:
                log.info("Error floor detected in %s: %.2f%%", col_name, col_data["error_floor"])
            if isinstance(col_data, dict) and not col_data.get("is_monotonically_decreasing", True):
                log.warning("Non-monotonic error curve: %s (%d increasing steps)",
                            col_name, col_data.get("n_increasing_steps", 0))

    # ---- Validity mask (to_do.md §6.3) ----
    validity_masks: list[dict] = []
    for q in target_quantiles:
        for t in target_error_pct:
            mask_series = _build_validity_mask(
                results_plus, limits_df,
                target_error_pct=t, target_quantile=q,
                max_dict_dist=float(np.nanpercentile(
                    results_plus["sample_to_dict_dist_norm"].dropna(), 95
                )) if results_plus["sample_to_dict_dist_norm"].notna().any() else None,
            )
            validity_masks.append({
                "quantile": float(q),
                "target_error_pct": float(t),
                "n_pass": int(mask_series.sum()),
                "n_total": int(len(mask_series)),
                "pass_fraction": float(mask_series.mean()),
            })
    summary["validity_masks"] = validity_masks
    validity_mask_df = pd.DataFrame(validity_masks)
    validity_mask_csv = out_dir / "validity_mask_summary.csv"
    validity_mask_df.to_csv(validity_mask_csv, index=False)

    summary_json = out_dir / "step4_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    coverage_json = out_dir / "dictionary_coverage_metrics.json"
    coverage_json.write_text(json.dumps(coverage_metrics, indent=2), encoding="utf-8")

    if not args.no_plots:
        _plot_sample_size_distribution(
            results_ok,
            plots_dir / "sample_size_distribution.png",
            threshold=plane_min_events,
        )
        _plot_uncertainty_bands(
            uncertainty_df, results_ok,
            plots_dir / "uncertainty_bands_by_events.png",
        )
        _plot_threshold_sweep(sweep_df, plots_dir / "threshold_sweep_min_events.png")

        scatter_title = (
            "Dictionary points in flux-eff space "
            f"(n={coverage_metrics['n_points_unique_flux_eff']}, "
            f"grid_fill={coverage_metrics['grid_fill_pct']:.2f}%)"
        )
        _plot_dictionary_overview(dict_points, scatter_title, plots_dir / "dictionary_flux_eff.png")
        _plot_dictionary_coverage(nn_dist, radius_df, plots_dir / "dictionary_coverage.png")

        high_df = _event_filter(results_plus, plane_min_events, "ge")
        low_df = _event_filter(results_plus, plane_min_events, "lt")

        _plot_plane_mean_error_combined(
            high_df,
            title=f"Mean absolute relative error in (flux, eff) plane — events >= {plane_min_events:g}",
            path=plots_dir / f"plane_mean_error_ge_{int(plane_min_events)}.png",
            flux_bins=plane_flux_bins,
            eff_bins=plane_eff_bins,
        )
        _plot_plane_mean_error_combined(
            low_df,
            title=f"Mean absolute relative error in (flux, eff) plane — events < {plane_min_events:g}",
            path=plots_dir / f"plane_mean_error_lt_{int(plane_min_events)}.png",
            flux_bins=plane_flux_bins,
            eff_bins=plane_eff_bins,
        )

        _plot_residual_overlay(
            high_df, low_df,
            error_col="flux_rel_error_pct",
            xlabel="Flux relative error [%]",
            title="Flux residual distribution: high vs low statistics",
            path=plots_dir / "residual_overlay_flux.png",
            threshold=plane_min_events,
        )
        _plot_residual_overlay(
            high_df, low_df,
            error_col="eff_rel_error_pct",
            xlabel="Efficiency relative error [%]",
            title="Efficiency residual distribution: high vs low statistics",
            path=plots_dir / "residual_overlay_eff.png",
            threshold=plane_min_events,
        )

    log.info("Wrote uncertainty table: %s", uncertainty_csv)
    log.info("Wrote fixed-bin uncertainty table: %s", fixed_bins_csv)
    log.info("Wrote dictionary-membership uncertainty table: %s", membership_unc_csv)
    log.info("Wrote threshold-sweep table: %s", sweep_csv)
    log.info("Wrote limits table: %s", limits_csv)
    log.info("Wrote coverage-by-radius table: %s", radius_csv)
    log.info("Wrote enriched results with distance: %s", results_plus_csv)
    log.info("Wrote validity-mask summary: %s", validity_mask_csv)
    log.info("Wrote summary: %s", summary_json)
    log.info("Wrote coverage metrics: %s", coverage_json)
    log.info("Wrote STEP_4 outputs to: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
