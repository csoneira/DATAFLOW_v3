#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CANONICAL_EFF_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    cfg_path,
    ensure_output_dirs,
    load_config,
    resolve_path,
    write_json,
)

log = logging.getLogger("even_easier_advanced.step3")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_3 - %(message)s", level=logging.INFO, force=True)


def _path_from_config_or_default(config: dict[str, Any], path_key: str, default_relative_path: str) -> Path:
    paths = config.get("paths", {})
    if isinstance(paths, dict) and path_key in paths:
        return cfg_path(config, "paths", path_key)
    return resolve_path(config, default_relative_path)


def _pick_first_existing_column(columns: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError("Missing required column. Tried: " + ", ".join(candidates))


def _resolve_reference_columns(config: dict[str, Any], available_columns: list[str]) -> list[str]:
    step2_config = config.get("step2", {})
    if not isinstance(step2_config, dict):
        step2_config = {}

    raw_planes = step2_config.get("efficiency_reference_planes", [1, 2, 3, 4])
    if isinstance(raw_planes, (int, float)) and not isinstance(raw_planes, bool):
        raw_planes = [int(raw_planes)]
    if not isinstance(raw_planes, list):
        raise ValueError("step2.efficiency_reference_planes must be a list of plane indices.")

    columns: list[str] = []
    for value in raw_planes:
        plane = int(value)
        if plane < 1 or plane > 4:
            raise ValueError("step2.efficiency_reference_planes must contain values in [1, 4].")
        column = f"eff_empirical_{plane}"
        if column in available_columns:
            columns.append(column)

    if columns:
        return columns

    fallback = [column for column in CANONICAL_EFF_COLUMNS if column in available_columns]
    if not fallback:
        raise ValueError("No empirical-efficiency columns were found in the Step 0 training dataframe.")
    return fallback


def _resolve_reference_mode(config: dict[str, Any]) -> str:
    step2_config = config.get("step2", {})
    if not isinstance(step2_config, dict):
        step2_config = {}

    raw_mode = str(step2_config.get("efficiency_reference_mode", "product")).strip().lower()
    aliases = {
        "prod": "product",
        "product": "product",
        "mean": "mean_power4",
        "avg": "mean_power4",
        "average": "mean_power4",
        "mean_power4": "mean_power4",
    }
    mode = aliases.get(raw_mode, raw_mode)
    if mode not in {"product", "mean_power4"}:
        raise ValueError("step2.efficiency_reference_mode must be 'product' or 'mean_power4'.")
    return mode


def _compute_reference_and_corrected_rate(
    dataframe: pd.DataFrame,
    *,
    reference_columns: list[str],
    reference_mode: str,
    rate_column: str,
) -> pd.DataFrame:
    work = dataframe.copy()
    reference_frame = work[reference_columns].apply(pd.to_numeric, errors="coerce")
    rate_numeric = pd.to_numeric(work[rate_column], errors="coerce")

    if reference_mode == "product":
        work["eff_reference_step3"] = reference_frame.prod(axis=1, min_count=len(reference_columns))
        work["scale_factor_step3"] = 1.0 / work["eff_reference_step3"]
    else:
        work["eff_reference_step3"] = reference_frame.mean(axis=1)
        work["scale_factor_step3"] = 1.0 / (work["eff_reference_step3"] ** 4)

    work["corrected_rate_hz_step3"] = rate_numeric * pd.to_numeric(work["scale_factor_step3"], errors="coerce")
    work["eff_mean_step3"] = reference_frame.mean(axis=1)
    return work


def _fit_line(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float, float | None, str]:
    if len(x_values) == 0:
        return np.nan, np.nan, None, "empty"
    if len(x_values) == 1 or np.unique(np.round(x_values, 12)).size < 2:
        return 0.0, float(np.nanmean(y_values)), None, "constant"

    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    y_fit = slope * x_values + intercept
    ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
    ss_res = float(np.sum((y_values - y_fit) ** 2))
    r_squared = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), None if not np.isfinite(r_squared) else float(r_squared), "linear"


def _build_efficiency_case_fits(
    training_df: pd.DataFrame,
    *,
    efficiency_case_bin_width: float,
    min_points_per_case: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = training_df.copy()
    work["flux_step3"] = pd.to_numeric(work["flux_step3"], errors="coerce")
    work["corrected_rate_hz_step3"] = pd.to_numeric(work["corrected_rate_hz_step3"], errors="coerce")
    work["eff_mean_step3"] = pd.to_numeric(work["eff_mean_step3"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["flux_step3", "corrected_rate_hz_step3", "eff_mean_step3"]).copy()
    work = work.loc[work["corrected_rate_hz_step3"] > 0.0].copy()
    work = work.loc[work["flux_step3"] > 0.0].copy()
    if work.empty:
        raise ValueError("No valid rows are available to build Step 3 efficiency-case fits.")

    width = float(efficiency_case_bin_width)
    work["eff_case_bin_step3"] = np.clip(
        np.round(work["eff_mean_step3"].to_numpy(dtype=float) / width) * width,
        0.0,
        1.0,
    )
    work["eff_case_bin_step3"] = np.round(work["eff_case_bin_step3"], 6)

    rows: list[dict[str, Any]] = []
    for eff_case_bin, subset in work.groupby("eff_case_bin_step3", sort=True, dropna=False):
        subset = subset.copy()
        n_points = int(len(subset))
        if n_points < int(min_points_per_case):
            continue

        x = subset["corrected_rate_hz_step3"].to_numpy(dtype=float)
        y = subset["flux_step3"].to_numpy(dtype=float)
        slope, intercept, r_squared, fit_method = _fit_line(x, y)
        rows.append(
            {
                "eff_case_bin_step3": float(eff_case_bin),
                "eff_mean_step3": float(np.nanmedian(subset["eff_mean_step3"])),
                "n_points": n_points,
                "slope_flux_per_hz": slope,
                "intercept_flux": intercept,
                "r_squared": r_squared,
                "fit_method": fit_method,
                "corrected_rate_min_hz": float(np.nanmin(x)),
                "corrected_rate_max_hz": float(np.nanmax(x)),
                "flux_min_cm2_min": float(np.nanmin(y)),
                "flux_max_cm2_min": float(np.nanmax(y)),
            }
        )

    fit_table = pd.DataFrame(rows)
    if fit_table.empty:
        raise ValueError("No efficiency case has enough support to fit a Step 3 rate-to-flux line.")
    fit_table = fit_table.sort_values("eff_case_bin_step3").reset_index(drop=True)
    return fit_table, work


def _select_display_case_bins(fit_table: pd.DataFrame, max_curves: int) -> list[float]:
    bins = fit_table["eff_case_bin_step3"].astype(float).tolist()
    if len(bins) <= int(max_curves):
        return bins
    positions = np.linspace(0, len(bins) - 1, num=int(max_curves), dtype=int)
    selected = [bins[index] for index in np.unique(positions)]
    return [float(value) for value in selected]


def _filter_fit_cases_by_efficiency_window(
    fit_table: pd.DataFrame,
    *,
    efficiency_case_min: float,
    efficiency_case_max: float,
) -> pd.DataFrame:
    filtered = fit_table.loc[
        (fit_table["eff_mean_step3"].astype(float) >= float(efficiency_case_min))
        & (fit_table["eff_mean_step3"].astype(float) <= float(efficiency_case_max))
    ].copy()
    if filtered.empty:
        raise ValueError(
            "No Step 3 efficiency cases remain after applying efficiency_case_min/efficiency_case_max."
        )
    return filtered.sort_values("eff_case_bin_step3").reset_index(drop=True)


def _asymptotic_reference_line(
    fit_table: pd.DataFrame,
    *,
    top_k_cases: int,
) -> dict[str, Any]:
    ranked = fit_table.sort_values("eff_mean_step3", ascending=False).reset_index(drop=True)
    subset = ranked.head(max(int(top_k_cases), 1)).copy()
    weights = subset["n_points"].astype(float).to_numpy(dtype=float)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(subset), dtype=float)

    slope_ref = float(np.average(subset["slope_flux_per_hz"].astype(float), weights=weights))
    intercept_ref = float(np.average(subset["intercept_flux"].astype(float), weights=weights))
    return {
        "slope_flux_per_hz": slope_ref,
        "intercept_flux": intercept_ref,
        "n_cases_used": int(len(subset)),
        "eff_mean_range_used": [
            float(subset["eff_mean_step3"].min()),
            float(subset["eff_mean_step3"].max()),
        ],
        "case_bins_used": [float(value) for value in subset["eff_case_bin_step3"].tolist()],
    }


def _plot_step3_calibration_figure(
    prepared_training_df: pd.DataFrame,
    fit_table: pd.DataFrame,
    *,
    reference_line: dict[str, Any],
    output_path: Path,
    rate_column_label: str,
) -> None:
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.1], height_ratios=[1.0, 1.0], wspace=0.26, hspace=0.28)
    ax_main = fig.add_subplot(grid[:, 0])
    ax_slope = fig.add_subplot(grid[0, 1])
    ax_intercept = fig.add_subplot(grid[1, 1])

    scatter = ax_main.scatter(
        prepared_training_df["corrected_rate_hz_step3"],
        prepared_training_df["flux_step3"],
        c=prepared_training_df["eff_mean_step3"],
        cmap="viridis",
        s=14,
        alpha=0.16,
        edgecolors="none",
    )

    all_fits = fit_table.sort_values("eff_mean_step3").reset_index(drop=True)
    cmap = plt.get_cmap("viridis")
    for idx, row in all_fits.iterrows():
        color = cmap(idx / max(len(all_fits) - 1, 1))
        x_line = np.linspace(float(row["corrected_rate_min_hz"]), float(row["corrected_rate_max_hz"]), 160)
        y_line = float(row["slope_flux_per_hz"]) * x_line + float(row["intercept_flux"])
        ax_main.plot(
            x_line,
            y_line,
            color=color,
            linewidth=2.2,
            label=f"mean eff ~ {float(row['eff_mean_step3']):.2f}",
        )

    x_global = np.linspace(
        float(np.nanmin(prepared_training_df["corrected_rate_hz_step3"])),
        float(np.nanmax(prepared_training_df["corrected_rate_hz_step3"])),
        220,
    )
    y_ref = float(reference_line["slope_flux_per_hz"]) * x_global + float(reference_line["intercept_flux"])
    ax_main.plot(
        x_global,
        y_ref,
        color="black",
        linestyle="--",
        linewidth=2.4,
        label=(
            "asymptotic reference: "
            f"flux = {float(reference_line['slope_flux_per_hz']):.4f} * rate "
            f"{float(reference_line['intercept_flux']):+.4f}"
        ),
    )
    ax_main.set_xlabel(f"Corrected rate from {rate_column_label} [Hz]")
    ax_main.set_ylabel("Simulated flux [cm^-2 min^-1]")
    ax_main.set_title("Step 3: corrected-rate to simulated-flux calibration by efficiency case")
    ax_main.grid(alpha=0.25)
    ax_main.legend(fontsize=8, ncol=1, loc="best")
    cbar = fig.colorbar(scatter, ax=ax_main)
    cbar.set_label("Mean empirical efficiency (selected planes)")

    ax_slope.plot(
        all_fits["eff_mean_step3"],
        all_fits["slope_flux_per_hz"],
        marker="o",
        linewidth=1.7,
        color="#1f77b4",
    )
    ax_slope.axhline(
        float(reference_line["slope_flux_per_hz"]),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="asymptotic slope",
    )
    ax_slope.set_xlabel("Mean efficiency")
    ax_slope.set_ylabel("Slope [flux/Hz]")
    ax_slope.set_title("Slope vs efficiency mean")
    ax_slope.grid(alpha=0.25)
    ax_slope.legend(fontsize=8)

    ax_intercept.plot(
        all_fits["eff_mean_step3"],
        all_fits["intercept_flux"],
        marker="o",
        linewidth=1.7,
        color="#d62728",
    )
    ax_intercept.axhline(
        float(reference_line["intercept_flux"]),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="asymptotic intercept",
    )
    ax_intercept.set_xlabel("Mean efficiency")
    ax_intercept.set_ylabel("Intercept [flux]")
    ax_intercept.set_title("Y-intercept vs efficiency mean")
    ax_intercept.grid(alpha=0.25)
    ax_intercept.legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _resolve_time_axis(dataframe: pd.DataFrame) -> tuple[pd.Series, str]:
    if "file_timestamp_utc" in dataframe.columns:
        parsed = pd.to_datetime(dataframe["file_timestamp_utc"], errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed, "file_timestamp_utc"
    if "execution_timestamp_utc" in dataframe.columns:
        parsed = pd.to_datetime(dataframe["execution_timestamp_utc"], errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed, "execution_timestamp_utc"
    return pd.Series(range(len(dataframe)), dtype=float), "row_index"


def _plot_real_flux_time_series(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    flux_column: str,
) -> None:
    plot_frame = dataframe.copy()
    x_values, x_label = _resolve_time_axis(plot_frame)
    if x_label != "row_index":
        plot_frame["__plot_time_sort_key"] = x_values
        plot_frame = (
            plot_frame.sort_values("__plot_time_sort_key", na_position="last")
            .drop(columns="__plot_time_sort_key")
            .reset_index(drop=True)
        )
        x_values, x_label = _resolve_time_axis(plot_frame)

    flux_values = pd.to_numeric(plot_frame[flux_column], errors="coerce")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        x_values,
        flux_values,
        marker="o",
        markersize=3,
        linewidth=1.4,
        color="#1f77b4",
        label="Calibrated flux",
    )
    ax.set_title("Step 3 calibrated flux time series")
    ax.set_xlabel(x_label.replace("_", " "))
    ax.set_ylabel("Flux [cm^-2 min^-1]")
    ax.grid(alpha=0.25)
    ax.legend()
    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    step0_training_path = cfg_path(config, "paths", "step0_training_csv")
    step2_scaled_path = cfg_path(config, "paths", "step2_scaled_output_csv")
    lines_csv_path = _path_from_config_or_default(
        config,
        "step3_efficiency_case_lines_csv",
        "OUTPUTS/FILES/step3_efficiency_case_rate_to_flux_lines.csv",
    )
    reference_csv_path = _path_from_config_or_default(
        config,
        "step3_reference_line_csv",
        "OUTPUTS/FILES/step3_reference_rate_to_flux_line.csv",
    )
    real_flux_csv_path = _path_from_config_or_default(
        config,
        "step3_real_flux_output_csv",
        "OUTPUTS/FILES/step3_real_data_with_flux.csv",
    )
    meta_path = _path_from_config_or_default(config, "step3_meta_json", "OUTPUTS/FILES/step3_meta.json")
    plot_path = _path_from_config_or_default(
        config,
        "step3_calibration_plot",
        str(PLOTS_DIR / "step3_01_rate_to_flux_calibration.png"),
    )
    real_flux_plot_path = _path_from_config_or_default(
        config,
        "step3_real_flux_plot",
        str(PLOTS_DIR / "step3_02_real_flux_time_series.png"),
    )

    step3_config = config.get("step3", {})
    if not isinstance(step3_config, dict):
        step3_config = {}
    efficiency_case_bin_width = float(step3_config.get("efficiency_case_bin_width", 0.05))
    min_points_per_case = int(step3_config.get("min_points_per_efficiency_case", 20))
    display_case_count = int(step3_config.get("display_efficiency_cases", step3_config.get("display_max_efficiency_cases", 6)))
    efficiency_case_min = float(step3_config.get("efficiency_case_min", 0.0))
    efficiency_case_max = float(step3_config.get("efficiency_case_max", 1.0))
    asymptote_top_k_cases = int(step3_config.get("asymptote_top_k_cases", 5))

    if efficiency_case_bin_width <= 0.0:
        raise ValueError("step3.efficiency_case_bin_width must be > 0.")
    if min_points_per_case < 2:
        raise ValueError("step3.min_points_per_efficiency_case must be >= 2.")
    if display_case_count < 1:
        raise ValueError("step3.display_efficiency_cases must be >= 1.")
    if efficiency_case_min < 0.0 or efficiency_case_max > 1.0 or efficiency_case_min > efficiency_case_max:
        raise ValueError("step3 efficiency case window must satisfy 0 <= efficiency_case_min <= efficiency_case_max <= 1.")
    if asymptote_top_k_cases < 1:
        raise ValueError("step3.asymptote_top_k_cases must be >= 1.")

    training_df = pd.read_csv(step0_training_path, low_memory=False)
    flux_column = _pick_first_existing_column(
        training_df.columns.tolist(),
        ["sim_flux_cm2_min", "flux_cm2_min"],
    )
    rate_column = _pick_first_existing_column(
        training_df.columns.tolist(),
        ["selected_rate_hz", "rate_hz", "rate_1234_hz", "four_plane_rate_hz", "four_plane_robust_hz"],
    )
    reference_columns = _resolve_reference_columns(config, training_df.columns.tolist())
    reference_mode = _resolve_reference_mode(config)

    prepared_training_df = _compute_reference_and_corrected_rate(
        training_df,
        reference_columns=reference_columns,
        reference_mode=reference_mode,
        rate_column=rate_column,
    )
    prepared_training_df["flux_step3"] = pd.to_numeric(prepared_training_df[flux_column], errors="coerce")

    fit_table, fit_input = _build_efficiency_case_fits(
        prepared_training_df,
        efficiency_case_bin_width=efficiency_case_bin_width,
        min_points_per_case=min_points_per_case,
    )
    fit_table_window = _filter_fit_cases_by_efficiency_window(
        fit_table,
        efficiency_case_min=efficiency_case_min,
        efficiency_case_max=efficiency_case_max,
    )
    selected_case_bins = _select_display_case_bins(fit_table_window, display_case_count)
    fit_table_selected = fit_table_window.loc[
        fit_table_window["eff_case_bin_step3"].isin(selected_case_bins)
    ].copy()
    fit_table_selected = fit_table_selected.sort_values("eff_mean_step3").reset_index(drop=True)
    if fit_table_selected.empty:
        raise ValueError("No Step 3 efficiency cases were selected for plotting/calibration.")
    reference_line = _asymptotic_reference_line(
        fit_table_selected,
        top_k_cases=min(asymptote_top_k_cases, len(fit_table_selected)),
    )

    _plot_step3_calibration_figure(
        fit_input,
        fit_table_selected,
        reference_line=reference_line,
        output_path=plot_path,
        rate_column_label=rate_column,
    )

    lines_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fit_table_selected.to_csv(lines_csv_path, index=False)

    reference_row = {
        "reference_slope_flux_per_hz": float(reference_line["slope_flux_per_hz"]),
        "reference_intercept_flux": float(reference_line["intercept_flux"]),
        "reference_n_cases_used": int(reference_line["n_cases_used"]),
        "reference_eff_mean_min": float(reference_line["eff_mean_range_used"][0]),
        "reference_eff_mean_max": float(reference_line["eff_mean_range_used"][1]),
    }
    pd.DataFrame([reference_row]).to_csv(reference_csv_path, index=False)

    real_df = pd.read_csv(step2_scaled_path, low_memory=False)
    real_rate_column = (
        "corrected_rate_hz"
        if "corrected_rate_hz" in real_df.columns
        else _pick_first_existing_column(real_df.columns.tolist(), ["selected_rate_hz", "rate_hz"])
    )
    real_df["flux_from_step3_reference_cm2_min"] = (
        float(reference_line["slope_flux_per_hz"]) * pd.to_numeric(real_df[real_rate_column], errors="coerce")
        + float(reference_line["intercept_flux"])
    )
    real_flux_csv_path.parent.mkdir(parents=True, exist_ok=True)
    real_df.to_csv(real_flux_csv_path, index=False)
    _plot_real_flux_time_series(
        real_df,
        real_flux_plot_path,
        flux_column="flux_from_step3_reference_cm2_min",
    )

    metadata = {
        "step0_training_csv": str(step0_training_path),
        "step2_scaled_output_csv": str(step2_scaled_path),
        "flux_input_column": flux_column,
        "rate_input_column": rate_column,
        "real_rate_input_column": real_rate_column,
        "reference_efficiency_columns": reference_columns,
        "reference_mode": reference_mode,
        "efficiency_case_bin_width": efficiency_case_bin_width,
        "efficiency_case_min": efficiency_case_min,
        "efficiency_case_max": efficiency_case_max,
        "min_points_per_efficiency_case": min_points_per_case,
        "display_efficiency_cases": display_case_count,
        "asymptote_top_k_cases": asymptote_top_k_cases,
        "n_training_rows": int(len(training_df)),
        "n_fit_rows_used": int(len(fit_input)),
        "n_efficiency_cases_fitted_total": int(len(fit_table)),
        "n_efficiency_cases_in_window": int(len(fit_table_window)),
        "n_efficiency_cases_selected": int(len(fit_table_selected)),
        "selected_case_bins_plotted": [float(value) for value in fit_table_selected["eff_case_bin_step3"].astype(float).tolist()],
        "reference_rate_to_flux_line": reference_line,
        "outputs": {
            "step3_efficiency_case_lines_csv": str(lines_csv_path),
            "step3_reference_line_csv": str(reference_csv_path),
            "step3_real_flux_output_csv": str(real_flux_csv_path),
            "step3_real_flux_plot": str(real_flux_plot_path),
            "step3_meta_json": str(meta_path),
            "step3_calibration_plot": str(plot_path),
        },
    }
    write_json(meta_path, metadata)

    log.info("Wrote Step 3 efficiency-case lines to %s", lines_csv_path)
    log.info(
        "Reference flux calibration line: flux = %.6f * rate + %.6f",
        float(reference_line["slope_flux_per_hz"]),
        float(reference_line["intercept_flux"]),
    )
    log.info("Wrote Step 3 calibration figure to %s", plot_path)
    log.info("Wrote Step 3 real-data flux output to %s", real_flux_csv_path)
    log.info("Wrote Step 3 calibrated flux time-series plot to %s", real_flux_plot_path)
    return real_flux_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Step 3 rate-to-flux calibration from MINGO00 training data and "
            "apply the asymptotic reference line to Step 2 corrected rates."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
