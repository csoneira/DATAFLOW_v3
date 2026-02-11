#!/usr/bin/env python3
"""Validate simulation inputs against rates/efficiencies measured in STEP_1.

This script consumes the dictionary CSV built by STEP_1 and produces:
  1) an enriched per-file table with simulated and estimated quantities
  2) summary metrics
  3) scatter plots for quick validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Shared utilities --------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msv_utils import (  # noqa: E402
    apply_clean_style,
    build_validation_table,
    compute_efficiency,
    load_config,
    maybe_log_x,
    parse_efficiencies,
    safe_numeric,
    setup_logger,
    setup_output_dirs,
)

log = setup_logger("STEP_2_validate")
apply_clean_style()

DEFAULT_CONFIG = BASE_DIR / "config_validation.json"
DEFAULT_DICT = (
    REPO_ROOT / "STEP_1_BUILD_DICTIONARY" / "OUTPUTS" / "FILES" / "task_01"
    / "param_metadata_dictionary.csv"
)
DEFAULT_PARAMS_CSV = (
    REPO_ROOT.parent / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA"
    / "step_final_simulation_params.csv"
)
DEFAULT_OUT_DIR = BASE_DIR


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


def plot_residuals_all_planes(
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
def plot_residuals_all_planes(
    df: pd.DataFrame,
    out_path: Path,
    *,
    relerr_max_abs: float,
) -> None:
    """Residual and relative-error scatter for all four planes.

    Planes 1 & 4 carry an acceptance-factor bias so their residuals vs
    simulation show a systematic offset (marked with *).
    """
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    planes = [1, 2, 3, 4]
    for row, plane in enumerate(planes):
        sim_col = f"eff_sim_p{plane}"
        resid_col = f"eff_resid_p{plane}"
        rel_col = f"eff_rel_err_p{plane}"

        if sim_col not in df.columns or resid_col not in df.columns:
            for c in range(2):
                axes[row, c].set_visible(False)
            continue

        mask = df[sim_col].notna() & df[resid_col].notna() & df[rel_col].notna()
        mask = mask & (df[rel_col].abs() <= relerr_max_abs)
        p_note = " *" if plane in (1, 4) else ""

        ax = axes[row, 0]
        ax.scatter(df.loc[mask, sim_col], df.loc[mask, resid_col], s=10, alpha=0.5)
        ax.axhline(0.0, color="red", ls="--", lw=0.8)
        ax.set_title(f"Plane {plane}: residual{p_note}")
        ax.set_ylabel("Est − Sim")

        ax = axes[row, 1]
        ax.scatter(df.loc[mask, sim_col], 100.0 * df.loc[mask, rel_col], s=10, alpha=0.5)
        ax.axhline(0.0, color="red", ls="--", lw=0.8)
        ax.set_title(f"Plane {plane}: rel. error{p_note}")
        ax.set_ylabel("Rel. error [%]")

    axes[-1, 0].set_xlabel("Simulated efficiency")
    axes[-1, 1].set_xlabel("Simulated efficiency")
    fig.suptitle("Residual & Relative Error — All Planes\n"
                 "(* planes 1 & 4: acceptance bias expected)", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_vs_event_count_all_planes(
    df: pd.DataFrame,
    out_path: Path,
    *,
    relerr_max_abs: float,
) -> None:
    """Error vs event count for all planes — 1/sqrt(N) overlay (to_do.md §3.1).

    The red dashed curve shows the expected purely-statistical scaling
    ``error ∝ 1/√N``.  If the data follow this curve the deviations are
    consistent with sampling noise.  Note that for planes 1 & 4 the
    residual is affected by an acceptance factor that does not cancel in
    the coincidence-ratio estimator, so a systematic offset above the
    guide curve is expected and does NOT indicate a problem.
    """
    ev_col = "generated_events_count"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    for idx, plane in enumerate([1, 2, 3, 4]):
        ax = axes[idx // 2, idx % 2]
        rel_col = f"eff_rel_err_p{plane}"
        if rel_col not in df.columns or ev_col not in df.columns:
            ax.set_title(f"Plane {plane}: no data")
            continue
        mask = df[rel_col].notna() & df[ev_col].notna()
        mask = mask & df[rel_col].abs().le(relerr_max_abs)
        if mask.sum() < 3:
            ax.set_title(f"Plane {plane}: < 3 points")
            continue
        events = df.loc[mask, ev_col].to_numpy(dtype=float)
        err = (100.0 * df.loc[mask, rel_col].abs()).to_numpy(dtype=float)
        ax.scatter(events, err, s=12, alpha=0.5, color="#4C78A8")
        # 1/sqrt(N) guide curve
        ev_sort = np.sort(events)
        if ev_sort[0] > 0:
            guide = err[np.argsort(events)]
            median_err = float(np.nanmedian(err))
            median_ev = float(np.nanmedian(events))
            if median_ev > 0:
                scale = median_err * np.sqrt(median_ev)
                ev_line = np.linspace(ev_sort[0], ev_sort[-1], 200)
                ax.plot(ev_line, scale / np.sqrt(ev_line), "r--",
                        linewidth=1, label=r"$\propto 1/\sqrt{N}$")
                ax.legend(fontsize=8)
        p_star = " *" if plane in (1, 4) else ""
        ax.set_title(f"Plane {plane}: |rel. error| vs events{p_star}")
        ax.set_ylabel("|Relative error| [%]")
        ax.grid(True, alpha=0.3)
        maybe_log_x(ax, pd.Series(events))
    axes[-1, 0].set_xlabel("Generated events")
    axes[-1, 1].set_xlabel("Generated events")
    fig.suptitle("Error vs Event Count — All Planes (variance scaling check §3.1)", fontsize=11)
    fig.text(
        0.5, 0.005,
        "* Planes 1 & 4: acceptance factor NOT cancelled — offset above 1/√N expected",
        ha="center", fontsize=9, style="italic", color="grey",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
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
    parser.add_argument("--relerr-max-abs", type=float, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    dictionary_csv = Path(args.dictionary_csv or config.get("dictionary_csv", str(DEFAULT_DICT)))
    params_csv = Path(args.params_csv or config.get("params_csv", str(DEFAULT_PARAMS_CSV)))
    out_base = Path(args.out_dir or config.get("out_dir", str(DEFAULT_OUT_DIR)))
    prefix = str(args.prefix or config.get("prefix", "raw"))
    eff_method = str(args.eff_method or config.get("eff_method", "four_over_three_plus_four"))
    relerr_max_abs = float(args.relerr_max_abs or config.get("relerr_max_abs", 0.05))

    if not dictionary_csv.exists():
        raise FileNotFoundError(f"Dictionary CSV not found: {dictionary_csv}")

    files_dir, plots_dir = setup_output_dirs(out_base)
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

    validation_csv = files_dir / "validation_table.csv"
    summary_csv = files_dir / "summary_metrics.csv"
    flux_plot = plots_dir / "scatter_flux_vs_global_trigger_rate.png"
    eff_plot = plots_dir / "scatter_eff_sim_vs_estimated.png"
    residual_all_plot = plots_dir / "scatter_residual_all_planes.png"
    error_vs_events_plot = plots_dir / "scatter_error_vs_events_all_planes.png"

    validation.to_csv(validation_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    plot_flux_vs_rate(validation, flux_plot)
    plot_efficiency_scatter(validation, eff_plot)
    plot_residuals_all_planes(validation, residual_all_plot, relerr_max_abs=relerr_max_abs)
    plot_error_vs_event_count_all_planes(validation, error_vs_events_plot, relerr_max_abs=relerr_max_abs)

    print(f"Wrote validation table: {validation_csv}")
    print(f"Wrote summary metrics: {summary_csv}")
    print(f"Wrote plot: {flux_plot}")
    print(f"Wrote plot: {eff_plot}")
    print(f"Wrote plot: {residual_all_plot}")
    print(f"Wrote plot: {error_vs_events_plot}")
    print(f"Rows used: {len(validation)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
