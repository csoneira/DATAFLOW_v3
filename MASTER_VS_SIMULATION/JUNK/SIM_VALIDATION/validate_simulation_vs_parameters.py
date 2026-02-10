#!/usr/bin/env python3
"""Validate simulation inputs against rates/efficiencies measured in STEP_1.

This script consumes the dictionary CSV built by DICTIONARY_CREATOR and produces:
  1) an enriched per-file table with simulated and estimated quantities
  2) summary metrics
  3) scatter plots for quick validation
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config_validation.json"
DEFAULT_DICT = (
    BASE_DIR.parent
    / "DICTIONARY_CREATOR"
    / "STEP_1_BUILD"
    / "output"
    / "task_01"
    / "param_metadata_dictionary.csv"
)
DEFAULT_PARAMS_CSV = (
    BASE_DIR.parent.parent
    / "MINGO_DIGITAL_TWIN"
    / "SIMULATED_DATA"
    / "step_final_simulation_params.csv"
)
DEFAULT_OUT_DIR = BASE_DIR / "output"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _parse_efficiencies(value: object) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [np.nan, np.nan, np.nan, np.nan]
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
            return [float(parsed[0]), float(parsed[1]), float(parsed[2]), float(parsed[3])]
    return [np.nan, np.nan, np.nan, np.nan]


def _safe_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _compute_efficiency(
    n_four: pd.Series,
    n_three_missing: pd.Series,
    method: str,
) -> pd.Series:
    if method == "four_over_three_plus_four":
        return n_four / (n_four + n_three_missing)
    if method == "one_minus_three_over_four":
        return 1.0 - (n_three_missing / n_four.replace({0: np.nan}))
    raise ValueError(f"Unsupported efficiency method: {method}")


def build_validation_table(df: pd.DataFrame, prefix: str, eff_method: str) -> pd.DataFrame:
    four_col = f"{prefix}_tt_1234_count"
    miss_cols = {
        1: f"{prefix}_tt_234_count",
        2: f"{prefix}_tt_134_count",
        3: f"{prefix}_tt_124_count",
        4: f"{prefix}_tt_123_count",
    }
    rate_cols = [col for col in df.columns if col.startswith(f"{prefix}_tt_") and col.endswith("_rate_hz")]

    needed = [four_col, *miss_cols.values(), "flux_cm2_min", "cos_n", "efficiencies"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise KeyError(
            "Dictionary CSV is missing required columns for validation: "
            + ", ".join(missing)
        )

    out = df.copy()
    event_count_candidates = [
        "generated_events_count",
        "total_events_generated",
        "n_events_generated",
        "num_events_generated",
        "event_count",
        "num_events",
        "selected_rows",
    ]
    source_event_col = next((c for c in event_count_candidates if c in out.columns), None)
    if source_event_col is not None and source_event_col != "generated_events_count":
        out["generated_events_count"] = out[source_event_col]
    numeric_cols = [four_col, *miss_cols.values(), "flux_cm2_min", "cos_n", *rate_cols, "generated_events_count"]
    out = _safe_numeric(out, numeric_cols)

    sim_eff = out["efficiencies"].apply(_parse_efficiencies).apply(pd.Series)
    sim_eff.columns = [f"eff_sim_p{i}" for i in range(1, 5)]
    out = pd.concat([out, sim_eff], axis=1)

    for plane in range(1, 5):
        miss_col = miss_cols[plane]
        est_col = f"eff_est_p{plane}"
        sim_col = f"eff_sim_p{plane}"
        resid_col = f"eff_resid_p{plane}"
        rel_col = f"eff_rel_err_p{plane}"
        out[est_col] = _compute_efficiency(out[four_col], out[miss_col], eff_method)
        out[resid_col] = out[est_col] - out[sim_col]
        out[f"eff_abs_err_p{plane}"] = out[resid_col].abs()
        out[rel_col] = out[resid_col] / out[sim_col].replace({0: np.nan})

    if rate_cols:
        out["global_trigger_rate_hz"] = out[rate_cols].sum(axis=1, min_count=1)
    else:
        out["global_trigger_rate_hz"] = np.nan

    keep_cols = [
        "file_name",
        "filename_base",
        "param_set_id",
        "flux_cm2_min",
        "cos_n",
        "generated_events_count",
        four_col,
        *miss_cols.values(),
        "global_trigger_rate_hz",
    ]
    keep_cols += [f"eff_sim_p{i}" for i in range(1, 5)]
    keep_cols += [f"eff_est_p{i}" for i in range(1, 5)]
    keep_cols += [f"eff_resid_p{i}" for i in range(1, 5)]
    keep_cols += [f"eff_rel_err_p{i}" for i in range(1, 5)]
    keep_cols += [f"eff_abs_err_p{i}" for i in range(1, 5)]

    for col in keep_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep_cols].copy()


def _calc_metrics(sim: pd.Series, est: pd.Series) -> dict[str, float]:
    mask = sim.notna() & est.notna()
    if not mask.any():
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    s = sim[mask]
    e = est[mask]
    err = e - s
    corr = s.corr(e)
    return {
        "n": int(mask.sum()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "corr": float(corr) if pd.notna(corr) else np.nan,
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for plane in range(1, 5):
        metrics = _calc_metrics(df[f"eff_sim_p{plane}"], df[f"eff_est_p{plane}"])
        metrics["metric"] = f"plane_{plane}_efficiency"
        rows.append(metrics)

    mask = df["flux_cm2_min"].notna() & df["global_trigger_rate_hz"].notna()
    corr_flux = (
        float(df.loc[mask, "flux_cm2_min"].corr(df.loc[mask, "global_trigger_rate_hz"]))
        if mask.any()
        else np.nan
    )
    rows.append(
        {
            "metric": "flux_vs_global_trigger_rate",
            "n": int(mask.sum()),
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "corr": corr_flux,
        }
    )
    return pd.DataFrame(rows)


def plot_flux_vs_rate(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["flux_cm2_min"],
        df["global_trigger_rate_hz"],
        c=df["cos_n"],
        s=20,
        alpha=0.75,
        cmap="viridis",
    )
    ax.set_xlabel("Simulated Flux [cm^-2 min^-1]")
    ax.set_ylabel("Measured Global Trigger Rate [Hz]")
    ax.set_title("Simulation Flux vs Measured Trigger Rate (MINGO00, TASK_1)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("cos_n")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_efficiency_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for plane in range(1, 5):
        ax = axes[(plane - 1) // 2, (plane - 1) % 2]
        sim_col = f"eff_sim_p{plane}"
        est_col = f"eff_est_p{plane}"
        ax.scatter(df[sim_col], df[est_col], s=18, alpha=0.7)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1)
        ax.set_title(f"Plane {plane}")
        ax.set_xlabel("Simulated efficiency")
        ax.set_ylabel("Estimated efficiency")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Simulated vs Estimated Efficiency per Plane", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_contour_eff_vs_flux(
    df: pd.DataFrame,
    out_path: Path,
    *,
    cos_n_target: float,
    cos_n_tolerance: float,
    equal_eff_tolerance: float,
    value_col: str,
    levels: int,
) -> bool:
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in validation table.")

    subset = df.copy()
    subset = subset[
        (subset["cos_n"] - cos_n_target).abs() <= cos_n_tolerance
    ].copy()
    if subset.empty:
        return False

    # Keep only rows where all 4 simulated plane efficiencies are effectively equal.
    e1 = subset["eff_sim_p1"]
    equal_mask = (
        (subset["eff_sim_p2"] - e1).abs() <= equal_eff_tolerance
    ) & (
        (subset["eff_sim_p3"] - e1).abs() <= equal_eff_tolerance
    ) & (
        (subset["eff_sim_p4"] - e1).abs() <= equal_eff_tolerance
    )
    subset = subset[equal_mask].copy()
    if subset.empty:
        return False

    subset["eff_common"] = subset["eff_sim_p1"]
    subset = subset[["flux_cm2_min", "eff_common", value_col]].dropna()
    if subset.empty:
        return False

    # Collapse duplicate (flux,eff) points to one value so triangulation is stable.
    points = (
        subset.groupby(["flux_cm2_min", "eff_common"], as_index=False)[value_col]
        .mean()
    )
    if len(points) < 2:
        return False

    x = points["flux_cm2_min"].to_numpy()
    y = points["eff_common"].to_numpy()
    z = points[value_col].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    # Prefer a cleaner colored-point map over dense contour fills.
    sc = ax.scatter(
        x,
        y,
        c=z,
        s=65,
        cmap="viridis",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.35,
    )
    fig.colorbar(sc, ax=ax, label=value_col)

    # Optional light contour lines when enough unique points are available.
    if len(points) >= 12:
        try:
            ax.tricontour(x, y, z, levels=min(levels, 10), colors="k", linewidths=0.25, alpha=0.25)
        except Exception:
            pass

    ax.set_xlabel("Flux [cm^-2 min^-1]")
    ax.set_ylabel("Common plane efficiency")
    ax.set_title(
        f"Colored points: {value_col} in (eff, flux) plane\n"
        f"cos_n={cos_n_target:g}, eff1≈eff2≈eff3≈eff4 (tol={equal_eff_tolerance:g})"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _maybe_log_x(ax: plt.Axes, values: pd.Series) -> None:
    vals = values.dropna()
    if vals.empty:
        return
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmin > 0 and vmax / vmin >= 100.0:
        ax.set_xscale("log")


def plot_residuals_planes_2_3(
    df: pd.DataFrame,
    out_path: Path,
    *,
    relerr_max_abs: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col")
    planes = [2, 3]
    for row, plane in enumerate(planes):
        sim_col = f"eff_sim_p{plane}"
        resid_col = f"eff_resid_p{plane}"
        rel_col = f"eff_rel_err_p{plane}"
        ev_col = "generated_events_count"

        mask_common = df[sim_col].notna() & df[resid_col].notna() & df[rel_col].notna()
        mask_common = mask_common & (df[rel_col].abs() <= relerr_max_abs)

        ax_res = axes[row, 0]
        ax_res.scatter(df.loc[mask_common, sim_col], df.loc[mask_common, resid_col], s=18, alpha=0.7)
        ax_res.axhline(0.0, color="red", linestyle="--", linewidth=1)
        ax_res.set_title(f"Plane {plane}: residual")
        ax_res.set_ylabel("Estimated - Simulated")
        ax_res.grid(True, alpha=0.3)

        ax_rel = axes[row, 1]
        ax_rel.scatter(
            df.loc[mask_common, sim_col],
            100.0 * df.loc[mask_common, rel_col],
            s=18,
            alpha=0.7,
        )
        ax_rel.axhline(0.0, color="red", linestyle="--", linewidth=1)
        ax_rel.set_title(f"Plane {plane}: relative error (|err| <= {100.0 * relerr_max_abs:g}%)")
        ax_rel.set_ylabel("100 * (Estimated - Simulated) / Simulated [%]")
        ax_rel.grid(True, alpha=0.3)

        ax_ev = axes[row, 2]
        mask_ev = mask_common & df[ev_col].notna()
        ax_ev.scatter(
            df.loc[mask_ev, ev_col],
            100.0 * df.loc[mask_ev, rel_col],
            s=18,
            alpha=0.7,
        )
        ax_ev.axhline(0.0, color="red", linestyle="--", linewidth=1)
        ax_ev.set_title(f"Plane {plane}: rel. error vs generated events")
        ax_ev.set_ylabel("Relative error [%]")
        ax_ev.grid(True, alpha=0.3)
        _maybe_log_x(ax_ev, df.loc[mask_ev, ev_col])

    axes[1, 0].set_xlabel("Simulated efficiency")
    axes[1, 1].set_xlabel("Simulated efficiency")
    axes[1, 2].set_xlabel("Generated events in file")
    fig.suptitle("Residual and Relative Error for Plane 2 and Plane 3 (Filtered)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_colored_sim_eff_relerr_events_planes_2_3(
    df: pd.DataFrame,
    out_path: Path,
    *,
    relerr_max_abs: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ev_col = "generated_events_count"
    planes = [2, 3]
    for idx, plane in enumerate(planes):
        ax = axes[idx]
        sim_col = f"eff_sim_p{plane}"
        rel_col = f"eff_rel_err_p{plane}"
        mask = df[sim_col].notna() & df[rel_col].notna() & df[ev_col].notna()
        mask = mask & (df[rel_col].abs() <= relerr_max_abs)
        if not mask.any():
            ax.set_title(f"Plane {plane}: no points after filter")
            ax.grid(True, alpha=0.3)
            continue

        rel_pct = 100.0 * df.loc[mask, rel_col]
        counts = df.loc[mask, ev_col]
        cmax = max(1e-6, float(np.nanmax(np.abs(rel_pct))))
        sizes = 20.0 + 90.0 * (counts / counts.max())
        sc = ax.scatter(
            df.loc[mask, sim_col],
            rel_pct,
            s=sizes,
            c=rel_pct,
            cmap="coolwarm",
            vmin=-cmax,
            vmax=cmax,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.25,
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Plane {plane}")
        ax.set_xlabel("Simulated efficiency")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Relative error [%]")

    axes[0].set_ylabel("Relative error [%]")
    fig.suptitle(
        "Simulated Efficiency vs Relative Error (size ~ generated events, color = relative error)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate simulation parameters against STEP_1 outputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument(
        "--params-csv",
        default=None,
        help="Source-of-truth params CSV used to filter dictionary rows.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prefix", default=None, help="Trigger prefix (default from config or 'raw').")
    parser.add_argument(
        "--eff-method",
        default=None,
        choices=["four_over_three_plus_four", "one_minus_three_over_four"],
        help=(
            "Efficiency estimator: "
            "four_over_three_plus_four = N4 / (N4 + N3_missing), "
            "one_minus_three_over_four = 1 - N3_missing / N4."
        ),
    )
    parser.add_argument("--contour-cos-n", type=float, default=None)
    parser.add_argument("--contour-cos-tol", type=float, default=None)
    parser.add_argument("--contour-equal-eff-tol", type=float, default=None)
    parser.add_argument("--contour-value-col", default=None)
    parser.add_argument("--contour-levels", type=int, default=None)
    parser.add_argument("--relerr-max-abs", type=float, default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    dictionary_csv = Path(args.dictionary_csv or config.get("dictionary_csv", str(DEFAULT_DICT)))
    params_csv = Path(args.params_csv or config.get("params_csv", str(DEFAULT_PARAMS_CSV)))
    out_dir = Path(args.out_dir or config.get("out_dir", str(DEFAULT_OUT_DIR)))
    prefix = str(args.prefix or config.get("prefix", "raw"))
    eff_method = str(args.eff_method or config.get("eff_method", "four_over_three_plus_four"))
    contour_cos_n = float(args.contour_cos_n or config.get("contour_cos_n_target", 2.0))
    contour_cos_tol = float(args.contour_cos_tol or config.get("contour_cos_n_tolerance", 1e-9))
    contour_equal_eff_tol = float(
        args.contour_equal_eff_tol or config.get("contour_equal_eff_tolerance", 1e-9)
    )
    contour_value_col = str(
        args.contour_value_col or config.get("contour_value_column", "global_trigger_rate_hz")
    )
    contour_levels = int(args.contour_levels or config.get("contour_levels", 18))
    relerr_max_abs = float(args.relerr_max_abs or config.get("relerr_max_abs", 0.05))

    if not dictionary_csv.exists():
        raise FileNotFoundError(f"Dictionary CSV not found: {dictionary_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dictionary_csv, low_memory=False)
    if params_csv.exists():
        params_df = pd.read_csv(params_csv, usecols=["file_name"])
        params_files = params_df["file_name"].dropna().astype(str).str.strip()
        params_set = set(params_files)
        if "file_name" in df.columns:
            before = len(df)
            df = df[df["file_name"].astype(str).str.strip().isin(params_set)].copy()
            print(f"Filtered dictionary rows by {params_csv}: {before} -> {len(df)}")
        elif "filename_base" in df.columns:
            before = len(df)
            params_stems = {Path(name).stem for name in params_set}
            df = df[df["filename_base"].astype(str).str.strip().isin(params_stems)].copy()
            print(f"Filtered dictionary rows by {params_csv}: {before} -> {len(df)}")
        else:
            print("Warning: dictionary CSV has no file_name/filename_base; skipping params filter.")
    else:
        print(f"Warning: params CSV not found: {params_csv}; skipping params filter.")
    validation = build_validation_table(df, prefix=prefix, eff_method=eff_method)
    summary = build_summary(validation)

    validation_csv = out_dir / "validation_table.csv"
    summary_csv = out_dir / "summary_metrics.csv"
    flux_plot = out_dir / "scatter_flux_vs_global_trigger_rate.png"
    eff_plot = out_dir / "scatter_eff_sim_vs_estimated.png"
    contour_plot = out_dir / "contour_eff_vs_flux_cosn2_identical_eff.png"
    residual_plot = out_dir / "scatter_residual_relative_error_planes_2_3.png"
    colored_plot = out_dir / "scatter_colored_sim_eff_relerr_eventcount_planes_2_3.png"

    validation.to_csv(validation_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    plot_flux_vs_rate(validation, flux_plot)
    plot_efficiency_scatter(validation, eff_plot)
    plot_residuals_planes_2_3(validation, residual_plot, relerr_max_abs=relerr_max_abs)
    plot_colored_sim_eff_relerr_events_planes_2_3(
        validation,
        colored_plot,
        relerr_max_abs=relerr_max_abs,
    )
    contour_ok = plot_contour_eff_vs_flux(
        validation,
        contour_plot,
        cos_n_target=contour_cos_n,
        cos_n_tolerance=contour_cos_tol,
        equal_eff_tolerance=contour_equal_eff_tol,
        value_col=contour_value_col,
        levels=contour_levels,
    )

    print(f"Wrote validation table: {validation_csv}")
    print(f"Wrote summary metrics: {summary_csv}")
    print(f"Wrote plot: {flux_plot}")
    print(f"Wrote plot: {eff_plot}")
    print(f"Wrote plot: {residual_plot}")
    print(f"Wrote plot: {colored_plot}")
    if contour_ok:
        print(f"Wrote plot: {contour_plot}")
    else:
        print("Contour plot skipped: not enough matching points after filters.")
    print(f"Rows used: {len(validation)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
