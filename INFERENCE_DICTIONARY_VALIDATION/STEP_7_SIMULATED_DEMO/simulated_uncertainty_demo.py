#!/usr/bin/env python3
"""STEP_7: Simulated-data demo for point estimates + uncertainty coverage.

This script consumes:
  - STEP_4_SELF_CONSISTENCY all-sample results
  - STEP_6_UNCERTAINTY_LUT outputs

and produces:
  1) per-sample table with LUT-predicted uncertainties
  2) empirical coverage checks (1σ / 2σ / 3σ)
  3) a compact set of representative demo points
  4) a markdown summary ready for presentation
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msv_utils import (  # noqa: E402
    coerce_bool_series,
    load_config,
    maybe_log_x,
    resolve_param,
    setup_logger,
)
from uncertainty_lut import UncertaintyLUT  # noqa: E402

log = setup_logger("STEP_7_demo")

DEFAULT_CONFIG = STEP_DIR / "config.json"
DEFAULT_ALL_RESULTS = REPO_ROOT / "STEP_4_SELF_CONSISTENCY" / "output" / "all_samples_results.csv"
DEFAULT_LUT_DIR = REPO_ROOT / "STEP_6_UNCERTAINTY_LUT" / "output" / "lut"
DEFAULT_OUT_DIR = STEP_DIR / "output"


def _load_all_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    needed_numeric = [
        "sample_events_count",
        "true_flux_cm2_min",
        "true_eff_1",
        "estimated_flux_cm2_min",
        "estimated_eff_1",
        "abs_flux_rel_error_pct",
        "abs_eff_rel_error_pct",
        "flux_rel_error_pct",
        "eff_rel_error_pct",
    ]
    for col in needed_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [
        "sample_events_count",
        "true_flux_cm2_min",
        "true_eff_1",
        "estimated_flux_cm2_min",
        "estimated_eff_1",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in all-samples CSV: {missing}")

    df = df.dropna(subset=required).copy()
    df = df[df["sample_events_count"] > 0].copy()
    return df.reset_index(drop=True)


def _exact_self_match_mask(df: pd.DataFrame) -> pd.Series:
    """Detect exact self-matches (sample present in dictionary + zero error)."""
    if "sample_in_dictionary" in df.columns:
        in_dict = coerce_bool_series(df["sample_in_dictionary"]) == True
    else:
        in_dict = pd.Series(False, index=df.index)

    if "abs_flux_rel_error_pct" in df.columns:
        exact_flux = pd.to_numeric(df["abs_flux_rel_error_pct"], errors="coerce").fillna(np.inf) == 0.0
    else:
        exact_flux = (df["estimated_flux_cm2_min"] - df["true_flux_cm2_min"]).abs() == 0.0

    if "abs_eff_rel_error_pct" in df.columns:
        exact_eff = pd.to_numeric(df["abs_eff_rel_error_pct"], errors="coerce").fillna(np.inf) == 0.0
    else:
        exact_eff = (df["estimated_eff_1"] - df["true_eff_1"]).abs() == 0.0

    return in_dict & exact_flux & exact_eff


def _attach_lut_uncertainties(df: pd.DataFrame, lut: UncertaintyLUT) -> pd.DataFrame:
    out = df.copy()
    sigma_flux_pct, sigma_eff_pct = lut.query_batch(
        out["estimated_flux_cm2_min"].to_numpy(dtype=float),
        out["estimated_eff_1"].to_numpy(dtype=float),
        out["sample_events_count"].to_numpy(dtype=float),
    )
    out["sigma_flux_pct"] = sigma_flux_pct
    out["sigma_eff_pct"] = sigma_eff_pct

    out["abs_flux_err"] = (out["estimated_flux_cm2_min"] - out["true_flux_cm2_min"]).abs()
    out["abs_eff_err"] = (out["estimated_eff_1"] - out["true_eff_1"]).abs()
    out["sigma_flux_abs"] = out["estimated_flux_cm2_min"].abs() * out["sigma_flux_pct"] / 100.0
    out["sigma_eff_abs"] = out["estimated_eff_1"].abs() * out["sigma_eff_pct"] / 100.0

    out["error_over_sigma_flux"] = np.where(
        out["sigma_flux_abs"] > 0, out["abs_flux_err"] / out["sigma_flux_abs"], np.nan
    )
    out["error_over_sigma_eff"] = np.where(
        out["sigma_eff_abs"] > 0, out["abs_eff_err"] / out["sigma_eff_abs"], np.nan
    )

    for k in (1, 2, 3):
        out[f"inside_flux_{k}sigma"] = out["abs_flux_err"] <= k * out["sigma_flux_abs"]
        out[f"inside_eff_{k}sigma"] = out["abs_eff_err"] <= k * out["sigma_eff_abs"]
        out[f"inside_joint_{k}sigma"] = out[f"inside_flux_{k}sigma"] & out[f"inside_eff_{k}sigma"]

    out["flux_1sigma_lo"] = out["estimated_flux_cm2_min"] - out["sigma_flux_abs"]
    out["flux_1sigma_hi"] = out["estimated_flux_cm2_min"] + out["sigma_flux_abs"]
    out["eff_1sigma_lo"] = out["estimated_eff_1"] - out["sigma_eff_abs"]
    out["eff_1sigma_hi"] = out["estimated_eff_1"] + out["sigma_eff_abs"]
    return out


def _coverage_stats(df: pd.DataFrame, *, suffix: str = "") -> dict[str, object]:
    n = int(len(df))
    if n == 0:
        return {"n_samples": 0}

    stats: dict[str, object] = {"n_samples": n}
    for k in (1, 2, 3):
        stats[f"coverage_flux_{k}sigma"] = float(df[f"inside_flux_{k}sigma{suffix}"].mean())
        stats[f"coverage_eff_{k}sigma"] = float(df[f"inside_eff_{k}sigma{suffix}"].mean())
        stats[f"coverage_joint_{k}sigma"] = float(df[f"inside_joint_{k}sigma{suffix}"].mean())

    stats["median_abs_flux_rel_error_pct"] = float(
        pd.to_numeric(df.get("abs_flux_rel_error_pct"), errors="coerce").median()
    )
    stats["median_abs_eff_rel_error_pct"] = float(
        pd.to_numeric(df.get("abs_eff_rel_error_pct"), errors="coerce").median()
    )
    stats["p68_abs_flux_rel_error_pct"] = float(
        pd.to_numeric(df.get("abs_flux_rel_error_pct"), errors="coerce").quantile(0.68)
    )
    stats["p68_abs_eff_rel_error_pct"] = float(
        pd.to_numeric(df.get("abs_eff_rel_error_pct"), errors="coerce").quantile(0.68)
    )
    stats["median_error_over_sigma_flux"] = float(
        pd.to_numeric(df.get("error_over_sigma_flux"), errors="coerce").median()
    )
    stats["median_error_over_sigma_eff"] = float(
        pd.to_numeric(df.get("error_over_sigma_eff"), errors="coerce").median()
    )
    return stats


def _apply_sigma_calibration(
    df: pd.DataFrame,
    *,
    target_coverage: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Scale sigma values so empirical 1σ coverage matches target."""
    out = df.copy()
    zf = pd.to_numeric(out["error_over_sigma_flux"], errors="coerce").dropna()
    ze = pd.to_numeric(out["error_over_sigma_eff"], errors="coerce").dropna()

    if zf.empty or ze.empty:
        scale_flux = 1.0
        scale_eff = 1.0
    else:
        scale_flux = float(np.nanquantile(zf.to_numpy(dtype=float), target_coverage))
        scale_eff = float(np.nanquantile(ze.to_numpy(dtype=float), target_coverage))
        if not np.isfinite(scale_flux) or scale_flux <= 0:
            scale_flux = 1.0
        if not np.isfinite(scale_eff) or scale_eff <= 0:
            scale_eff = 1.0

    out["sigma_flux_pct_cal"] = out["sigma_flux_pct"] * scale_flux
    out["sigma_eff_pct_cal"] = out["sigma_eff_pct"] * scale_eff
    out["sigma_flux_abs_cal"] = out["sigma_flux_abs"] * scale_flux
    out["sigma_eff_abs_cal"] = out["sigma_eff_abs"] * scale_eff

    for k in (1, 2, 3):
        out[f"inside_flux_{k}sigma_cal"] = out["abs_flux_err"] <= k * out["sigma_flux_abs_cal"]
        out[f"inside_eff_{k}sigma_cal"] = out["abs_eff_err"] <= k * out["sigma_eff_abs_cal"]
        out[f"inside_joint_{k}sigma_cal"] = (
            out[f"inside_flux_{k}sigma_cal"] & out[f"inside_eff_{k}sigma_cal"]
        )

    meta = {
        "target_coverage": float(target_coverage),
        "scale_flux": float(scale_flux),
        "scale_eff": float(scale_eff),
    }
    return out, meta


def _coverage_by_event_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    work = df.copy()
    if len(work) == 0:
        return pd.DataFrame()
    n_bins = max(2, int(n_bins))
    work["events_bin"] = pd.qcut(
        work["sample_events_count"], q=n_bins, duplicates="drop"
    )
    rows: list[dict[str, object]] = []
    for key, part in work.groupby("events_bin", observed=True):
        rows.append(
            {
                "events_bin": str(key),
                "n_samples": int(len(part)),
                "events_min": float(part["sample_events_count"].min()),
                "events_max": float(part["sample_events_count"].max()),
                "events_median": float(part["sample_events_count"].median()),
                "coverage_flux_1sigma": float(part["inside_flux_1sigma"].mean()),
                "coverage_eff_1sigma": float(part["inside_eff_1sigma"].mean()),
                "coverage_joint_1sigma": float(part["inside_joint_1sigma"].mean()),
                "median_abs_flux_rel_error_pct": float(
                    pd.to_numeric(part["abs_flux_rel_error_pct"], errors="coerce").median()
                ),
                "median_abs_eff_rel_error_pct": float(
                    pd.to_numeric(part["abs_eff_rel_error_pct"], errors="coerce").median()
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("events_min").reset_index(drop=True)


def _farthest_point_indices(coords: np.ndarray, n: int, seed: int) -> list[int]:
    if len(coords) == 0 or n <= 0:
        return []
    n = min(n, len(coords))
    rng = np.random.default_rng(seed)
    chosen: list[int] = [int(rng.integers(0, len(coords)))]
    while len(chosen) < n:
        d_min = np.full(len(coords), np.inf, dtype=float)
        for idx in chosen:
            d = np.linalg.norm(coords - coords[idx], axis=1)
            d_min = np.minimum(d_min, d)
        d_min[chosen] = -1.0
        nxt = int(np.argmax(d_min))
        if d_min[nxt] < 0:
            break
        chosen.append(nxt)
    return chosen


def _pick_demo_points(df: pd.DataFrame, n_points: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    flux = df["estimated_flux_cm2_min"].to_numpy(dtype=float)
    eff = df["estimated_eff_1"].to_numpy(dtype=float)
    ev = df["sample_events_count"].to_numpy(dtype=float)

    def _norm(x: np.ndarray) -> np.ndarray:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        span = hi - lo
        if span <= 0:
            return np.zeros_like(x)
        return (x - lo) / span

    coords = np.column_stack([_norm(flux), _norm(eff), _norm(np.log10(ev))])
    idx = _farthest_point_indices(coords, n_points, seed)
    demo = df.iloc[idx].copy()
    cols = [
        "sample_index",
        "sample_file",
        "sample_events_count",
        "true_flux_cm2_min",
        "estimated_flux_cm2_min",
        "sigma_flux_pct",
        "sigma_flux_abs",
        "sigma_flux_pct_cal",
        "sigma_flux_abs_cal",
        "flux_1sigma_lo",
        "flux_1sigma_hi",
        "inside_flux_1sigma",
        "inside_flux_1sigma_cal",
        "abs_flux_rel_error_pct",
        "true_eff_1",
        "estimated_eff_1",
        "sigma_eff_pct",
        "sigma_eff_abs",
        "sigma_eff_pct_cal",
        "sigma_eff_abs_cal",
        "eff_1sigma_lo",
        "eff_1sigma_hi",
        "inside_eff_1sigma",
        "inside_eff_1sigma_cal",
        "abs_eff_rel_error_pct",
        "error_over_sigma_flux",
        "error_over_sigma_eff",
    ]
    existing_cols = [c for c in cols if c in demo.columns]
    demo = demo[existing_cols].sort_values("sample_events_count")
    return demo.reset_index(drop=True)


def _plot_nsigma_coverage(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    k_vals = np.array([1, 2, 3], dtype=float)
    flux_cov = np.array([df[f"inside_flux_{k}sigma"].mean() for k in (1, 2, 3)], dtype=float)
    eff_cov = np.array([df[f"inside_eff_{k}sigma"].mean() for k in (1, 2, 3)], dtype=float)
    ideal = np.array([math.erf(k / math.sqrt(2.0)) for k in k_vals], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, ideal, "k--", linewidth=1.2, label="Ideal Gaussian CDF")
    ax.plot(k_vals, flux_cov, "o-", color="#4C78A8", label="Flux empirical")
    ax.plot(k_vals, eff_cov, "o-", color="#F58518", label="Efficiency empirical")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Sigma multiplier k")
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title("Empirical uncertainty coverage")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_error_over_sigma_hist(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    zf = pd.to_numeric(df["error_over_sigma_flux"], errors="coerce").dropna()
    ze = pd.to_numeric(df["error_over_sigma_eff"], errors="coerce").dropna()
    if zf.empty and ze.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if not zf.empty:
        ax.hist(zf, bins=40, alpha=0.65, density=True, label="|flux error| / sigma_flux")
    if not ze.empty:
        ax.hist(ze, bins=40, alpha=0.65, density=True, label="|eff error| / sigma_eff")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="1σ")
    ax.set_xlabel("Error / predicted sigma")
    ax.set_ylabel("Density")
    ax.set_title("Calibration diagnostic: normalized error")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_demo_points(demo: pd.DataFrame, path: Path) -> None:
    if demo.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].errorbar(
        demo["true_flux_cm2_min"],
        demo["estimated_flux_cm2_min"],
        yerr=demo["sigma_flux_abs"],
        fmt="o",
        alpha=0.9,
        color="#4C78A8",
        ecolor="#4C78A8",
        elinewidth=1,
        capsize=2,
    )
    lo0 = float(min(demo["true_flux_cm2_min"].min(), demo["estimated_flux_cm2_min"].min()))
    hi0 = float(max(demo["true_flux_cm2_min"].max(), demo["estimated_flux_cm2_min"].max()))
    pad0 = 0.02 * (hi0 - lo0) if hi0 > lo0 else 0.05
    axes[0].plot([lo0 - pad0, hi0 + pad0], [lo0 - pad0, hi0 + pad0], "k--", linewidth=1)
    axes[0].set_xlabel("True flux")
    axes[0].set_ylabel("Estimated flux ± σ")
    axes[0].set_title("Demo points: flux")
    axes[0].grid(True, alpha=0.25)

    axes[1].errorbar(
        demo["true_eff_1"],
        demo["estimated_eff_1"],
        yerr=demo["sigma_eff_abs"],
        fmt="o",
        alpha=0.9,
        color="#F58518",
        ecolor="#F58518",
        elinewidth=1,
        capsize=2,
    )
    lo1 = float(min(demo["true_eff_1"].min(), demo["estimated_eff_1"].min()))
    hi1 = float(max(demo["true_eff_1"].max(), demo["estimated_eff_1"].max()))
    pad1 = 0.02 * (hi1 - lo1) if hi1 > lo1 else 0.05
    axes[1].plot([lo1 - pad1, hi1 + pad1], [lo1 - pad1, hi1 + pad1], "k--", linewidth=1)
    axes[1].set_xlabel("True eff_1")
    axes[1].set_ylabel("Estimated eff_1 ± σ")
    axes[1].set_title("Demo points: efficiency")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_markdown_report(
    out_path: Path,
    *,
    input_results: Path,
    input_lut: Path,
    stats_all: dict[str, object],
    stats_non_exact: dict[str, object],
    stats_non_exact_cal: dict[str, object],
    calibration: dict[str, float],
    n_exact: int,
    demo_points_csv: Path,
) -> None:
    lines = [
        "# Simulated Validation Demo",
        "",
        "This report demonstrates dictionary-based point estimation with uncertainty",
        "using simulated data only (self-consistency validation).",
        "",
        f"- Input all-samples results: `{input_results}`",
        f"- Input LUT directory: `{input_lut}`",
        f"- Exact self-matches removed for conservative uncertainty check: `{n_exact}` rows",
        "",
        "## Coverage Summary (All Successful Samples)",
        "",
        f"- n samples: **{stats_all.get('n_samples', 0)}**",
        f"- Flux coverage @1σ: **{100.0 * float(stats_all.get('coverage_flux_1sigma', 0.0)):.1f}%**",
        f"- Efficiency coverage @1σ: **{100.0 * float(stats_all.get('coverage_eff_1sigma', 0.0)):.1f}%**",
        f"- Joint coverage @1σ: **{100.0 * float(stats_all.get('coverage_joint_1sigma', 0.0)):.1f}%**",
        "",
        "## Coverage Summary (Conservative: Excluding Exact Self-Matches)",
        "",
        f"- n samples: **{stats_non_exact.get('n_samples', 0)}**",
        f"- Flux coverage @1σ: **{100.0 * float(stats_non_exact.get('coverage_flux_1sigma', 0.0)):.1f}%**",
        f"- Efficiency coverage @1σ: **{100.0 * float(stats_non_exact.get('coverage_eff_1sigma', 0.0)):.1f}%**",
        f"- Joint coverage @1σ: **{100.0 * float(stats_non_exact.get('coverage_joint_1sigma', 0.0)):.1f}%**",
        f"- Flux coverage @2σ: **{100.0 * float(stats_non_exact.get('coverage_flux_2sigma', 0.0)):.1f}%**",
        f"- Efficiency coverage @2σ: **{100.0 * float(stats_non_exact.get('coverage_eff_2sigma', 0.0)):.1f}%**",
        "",
        "## Coverage After Sigma Calibration (Conservative Subset)",
        "",
        f"- Target 1σ coverage: **{100.0 * float(calibration.get('target_coverage', 0.68)):.1f}%**",
        f"- Applied scale factors: flux **x{float(calibration.get('scale_flux', 1.0)):.3f}**, "
        f"efficiency **x{float(calibration.get('scale_eff', 1.0)):.3f}**",
        f"- Flux coverage @1σ (calibrated): **{100.0 * float(stats_non_exact_cal.get('coverage_flux_1sigma', 0.0)):.1f}%**",
        f"- Efficiency coverage @1σ (calibrated): **{100.0 * float(stats_non_exact_cal.get('coverage_eff_1sigma', 0.0)):.1f}%**",
        "",
        "## Demo Points",
        "",
        f"Representative points with estimates and uncertainty intervals are in: `{demo_points_csv}`",
        "",
        "Use these rows directly in slides as concrete examples of:",
        "- point estimate (estimated flux / estimated efficiency)",
        "- uncertainty interval (±σ from LUT)",
        "- truth-in-interval check (`inside_*_1sigma`)",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate simulated demo artifacts for point estimates and uncertainty calibration."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--all-results-csv", default=None)
    parser.add_argument("--lut-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-demo-points", type=int, default=None)
    parser.add_argument("--events-bins", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-coverage", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    all_results_csv = Path(
        resolve_param(args.all_results_csv, config, "all_results_csv", str(DEFAULT_ALL_RESULTS))
    )
    lut_dir = Path(resolve_param(args.lut_dir, config, "lut_dir", str(DEFAULT_LUT_DIR)))
    out_dir = Path(resolve_param(args.out_dir, config, "out_dir", str(DEFAULT_OUT_DIR)))
    n_demo_points = int(resolve_param(args.n_demo_points, config, "n_demo_points", 12, int))
    events_bins = int(resolve_param(args.events_bins, config, "events_bins", 4, int))
    seed = int(resolve_param(args.seed, config, "seed", 123, int))
    target_coverage = float(
        resolve_param(args.target_coverage, config, "target_coverage", 0.68, float)
    )

    if not all_results_csv.exists():
        raise FileNotFoundError(f"All-results CSV not found: {all_results_csv}")
    if not lut_dir.exists():
        raise FileNotFoundError(f"LUT directory not found: {lut_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        for p in out_dir.glob("*.png"):
            p.unlink()

    log.info("Loading all-sample results: %s", all_results_csv)
    df = _load_all_results(all_results_csv)
    log.info("Rows available for evaluation: %d", len(df))

    log.info("Loading uncertainty LUT: %s", lut_dir)
    lut = UncertaintyLUT.load(lut_dir)
    eval_df = _attach_lut_uncertainties(df, lut)

    exact_mask = _exact_self_match_mask(eval_df)
    eval_df["is_exact_self_match"] = exact_mask
    non_exact_df = eval_df.loc[~exact_mask].copy()
    non_exact_df_cal, calibration = _apply_sigma_calibration(
        non_exact_df, target_coverage=target_coverage
    )
    eval_df = eval_df.merge(
        non_exact_df_cal[
            [
                "sample_index",
                "sigma_flux_pct_cal",
                "sigma_eff_pct_cal",
                "sigma_flux_abs_cal",
                "sigma_eff_abs_cal",
                "inside_flux_1sigma_cal",
                "inside_eff_1sigma_cal",
                "inside_joint_1sigma_cal",
                "inside_flux_2sigma_cal",
                "inside_eff_2sigma_cal",
                "inside_joint_2sigma_cal",
                "inside_flux_3sigma_cal",
                "inside_eff_3sigma_cal",
                "inside_joint_3sigma_cal",
            ]
        ],
        on="sample_index",
        how="left",
    )

    all_table_csv = out_dir / "all_samples_with_lut_uncertainty.csv"
    eval_df.to_csv(all_table_csv, index=False)

    stats_all = _coverage_stats(eval_df, suffix="")
    stats_non_exact = _coverage_stats(non_exact_df_cal, suffix="")
    stats_non_exact_cal = _coverage_stats(non_exact_df_cal, suffix="_cal")

    by_events = _coverage_by_event_bins(non_exact_df_cal, n_bins=events_bins)
    by_events_csv = out_dir / "coverage_by_events.csv"
    by_events.to_csv(by_events_csv, index=False)

    demo_df = _pick_demo_points(non_exact_df_cal, n_points=n_demo_points, seed=seed)
    demo_points_csv = out_dir / "demo_points_with_uncertainty.csv"
    demo_df.to_csv(demo_points_csv, index=False)

    summary = {
        "input_all_results_csv": str(all_results_csv),
        "input_lut_dir": str(lut_dir),
        "n_success_rows": int(len(eval_df)),
        "n_exact_self_matches": int(exact_mask.sum()),
        "coverage_all_rows": stats_all,
        "coverage_without_exact_self_matches": stats_non_exact,
        "coverage_without_exact_self_matches_calibrated": stats_non_exact_cal,
        "sigma_calibration": calibration,
    }
    summary_json = out_dir / "demo_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_md = out_dir / "demo_report.md"
    _write_markdown_report(
        report_md,
        input_results=all_results_csv,
        input_lut=lut_dir,
        stats_all=stats_all,
        stats_non_exact=stats_non_exact,
        stats_non_exact_cal=stats_non_exact_cal,
        calibration=calibration,
        n_exact=int(exact_mask.sum()),
        demo_points_csv=demo_points_csv,
    )

    if not args.no_plots:
        _plot_nsigma_coverage(non_exact_df, out_dir / "coverage_nsigma.png")
        _plot_error_over_sigma_hist(non_exact_df, out_dir / "error_over_sigma_hist.png")
        _plot_demo_points(demo_df, out_dir / "demo_points_true_vs_est.png")

        # Simple scatter to show sample-size regime used for coverage
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(non_exact_df["sample_events_count"], bins=40, alpha=0.8, color="#4C78A8", edgecolor="white")
        ax.set_xlabel("Sample events count")
        ax.set_ylabel("Count")
        ax.set_title("Sample-size distribution (non-exact subset)")
        ax.grid(True, alpha=0.25)
        maybe_log_x(ax, non_exact_df["sample_events_count"])
        fig.tight_layout()
        fig.savefig(out_dir / "events_distribution_non_exact.png", dpi=150)
        plt.close(fig)

    log.info("Wrote summary JSON: %s", summary_json)
    log.info("Wrote report: %s", report_md)
    log.info("Wrote demo points table: %s", demo_points_csv)
    log.info("Done. Artifacts in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
