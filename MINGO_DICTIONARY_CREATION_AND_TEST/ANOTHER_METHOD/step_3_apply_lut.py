#!/usr/bin/env python3
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

from common import (
    CANONICAL_EFF_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    apply_lut_fallback_matches,
    cfg_path,
    ensure_output_dirs,
    get_rate_column_name,
    load_config,
    quantize_efficiency_series,
    read_ascii_lut,
    write_json,
)

log = logging.getLogger("another_method.step3")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_3 - %(message)s", level=logging.INFO, force=True)


def _rename_relevant_columns(dataframe: pd.DataFrame, config: dict) -> pd.DataFrame:
    eff_columns = list(config["columns"]["efficiencies"])
    z_columns = list(config["columns"]["z_positions"])
    rate_column = get_rate_column_name(config)
    rename_map = {
        eff_columns[0]: CANONICAL_EFF_COLUMNS[0],
        eff_columns[1]: CANONICAL_EFF_COLUMNS[1],
        eff_columns[2]: CANONICAL_EFF_COLUMNS[2],
        eff_columns[3]: CANONICAL_EFF_COLUMNS[3],
        z_columns[0]: "z_pos_1",
        z_columns[1]: "z_pos_2",
        z_columns[2]: "z_pos_3",
        z_columns[3]: "z_pos_4",
        rate_column: "rate_hz",
        str(config["columns"]["simulated_flux"]): "sim_flux_cm2_min",
    }
    return dataframe.rename(columns=rename_map)


def _resolve_lut_match_settings(config: dict) -> tuple[str, int | None, float]:
    step3_config = config.get("step3", {})
    if not isinstance(step3_config, dict):
        step3_config = {}

    raw_mode = step3_config.get("lut_match_mode")
    if raw_mode in (None, "", "null", "None"):
        allow_nearest = bool(step3_config.get("allow_nearest_lut_match", True))
        match_mode = "nearest" if allow_nearest else "exact"
    else:
        normalized = str(raw_mode).strip().lower()
        match_mode = {
            "idw": "interpolate",
            "interpolated": "interpolate",
            "interpolation": "interpolate",
        }.get(normalized, normalized)

    interpolation_k_raw = step3_config.get("lut_interpolation_k", 8)
    interpolation_k = None if interpolation_k_raw in (None, "", "null", "None") else int(interpolation_k_raw)

    interpolation_power_raw = step3_config.get("lut_interpolation_power", 2.0)
    interpolation_power = (
        2.0 if interpolation_power_raw in (None, "", "null", "None") else float(interpolation_power_raw)
    )
    return match_mode, interpolation_k, interpolation_power


def _resolve_time_axis(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    for candidate in ("time_utc", "time_start_utc", "time_end_utc"):
        if candidate not in dataframe.columns:
            continue
        parsed = pd.to_datetime(dataframe[candidate], errors="coerce", utc=True)
        if parsed.notna().any():
            order = np.argsort(parsed.fillna(parsed.min()))
            ordered = dataframe.iloc[order].reset_index(drop=True)
            ordered_time = parsed.iloc[order].reset_index(drop=True)
            return ordered, ordered_time, candidate

    ordered = dataframe.reset_index(drop=True).copy()
    return ordered, pd.Series(np.arange(len(ordered)), dtype=float), "synthetic_dataset_row"


def _plot_time_series_with_flux(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    ordered, x_values, x_label = _resolve_time_axis(dataframe)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        height_ratios=[2.0, 1.0, 1.2],
    )

    axes[0].plot(x_values, ordered["rate_hz"], marker="o", linewidth=1.6, label="Observed rate")
    axes[0].plot(
        x_values,
        ordered["corrected_rate_to_perfect_hz"],
        marker="o",
        linewidth=1.6,
        label="LUT-corrected rate",
    )
    axes[0].set_ylabel("Rate [Hz]")
    axes[0].set_title(
        "Observed and corrected rate with flux time series\n"
        f"rate column: {rate_column_name}"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        x_values,
        ordered["sim_flux_cm2_min"],
        marker="o",
        linewidth=1.6,
        color="#B22222",
        label="Simulated flux",
    )
    axes[1].set_xlabel(x_label.replace("_", " "))
    axes[1].set_ylabel("Flux [cm^-2 min^-1]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plane_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, column in enumerate(CANONICAL_EFF_COLUMNS):
        axes[2].plot(
            x_values,
            ordered[column],
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=plane_colors[idx],
            label=f"Plane {idx + 1} eff",
        )
    axes[2].set_xlabel(x_label.replace("_", " "))
    axes[2].set_ylabel("Empirical efficiency")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(alpha=0.25)
    axes[2].legend(ncol=2)

    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_flux_rate_comparison(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    ordered, _, _ = _resolve_time_axis(dataframe)
    sequence = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
    scatter_left = axes[0].scatter(
        ordered["sim_flux_cm2_min"],
        ordered["rate_hz"],
        c=sequence,
        cmap="viridis",
        s=44,
        alpha=0.85,
    )
    axes[0].set_title(f"Flux vs observed rate\nrate column: {rate_column_name}")
    axes[0].set_xlabel("Flux [cm^-2 min^-1]")
    axes[0].set_ylabel(f"Observed rate [Hz]\n({rate_column_name})")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        ordered["sim_flux_cm2_min"],
        ordered["corrected_rate_to_perfect_hz"],
        c=sequence,
        cmap="viridis",
        s=44,
        alpha=0.85,
    )
    axes[1].set_title(f"Flux vs corrected rate\nrate column: {rate_column_name}")
    axes[1].set_xlabel("Flux [cm^-2 min^-1]")
    axes[1].set_ylabel(f"Corrected rate [Hz]\n(from {rate_column_name})")
    axes[1].grid(alpha=0.25)

    cbar = fig.colorbar(scatter_left, ax=axes, shrink=0.95)
    cbar.set_label("Time-series order")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    synthetic_input_path = cfg_path(config, "paths", "synthetic_dataset_csv")
    lut_path = cfg_path(config, "paths", "step2_lut_ascii")
    lut_meta_path = cfg_path(config, "paths", "step2_meta_json")
    output_path = cfg_path(config, "paths", "step3_output_csv")
    meta_path = cfg_path(config, "paths", "step3_meta_json")

    synthetic_dataframe = pd.read_csv(synthetic_input_path)
    rate_column_name = get_rate_column_name(config)
    required_columns = [rate_column_name, str(config["columns"]["simulated_flux"])]
    missing_source_columns = [column for column in required_columns if column not in synthetic_dataframe.columns]
    if missing_source_columns:
        raise ValueError(
            "Synthetic dataset is missing required columns: " + ", ".join(missing_source_columns)
        )
    work = _rename_relevant_columns(synthetic_dataframe.copy(), config)
    lut_dataframe, lut_comments = read_ascii_lut(lut_path)

    lut_meta = {}
    if lut_meta_path.exists():
        lut_meta = json.loads(lut_meta_path.read_text(encoding="utf-8"))
    efficiency_bin_width = float(
        lut_meta.get("efficiency_bin_width", config.get("step2", {}).get("efficiency_bin_width", 0.02))
    )

    query_columns: list[str] = []
    for column in CANONICAL_EFF_COLUMNS:
        query_column = f"query_{column}"
        work[query_column] = quantize_efficiency_series(work[column], efficiency_bin_width)
        query_columns.append(query_column)

    lut_lookup = lut_dataframe.rename(columns={column: f"lut_{column}" for column in CANONICAL_EFF_COLUMNS})
    merged = work.merge(
        lut_lookup,
        how="left",
        left_on=query_columns,
        right_on=[f"lut_{column}" for column in CANONICAL_EFF_COLUMNS],
    )
    merged = merged.rename(columns={"scale_factor": "lut_scale_factor"})
    merged["lut_match_method"] = np.where(merged["lut_scale_factor"].notna(), "exact", pd.NA)
    merged["lut_match_distance"] = np.where(merged["lut_scale_factor"].notna(), 0.0, np.nan)

    match_mode, interpolation_k, interpolation_power = _resolve_lut_match_settings(config)
    merged = apply_lut_fallback_matches(
        merged,
        lut_dataframe,
        query_columns=query_columns,
        raw_columns=CANONICAL_EFF_COLUMNS,
        match_mode=match_mode,
        interpolation_k=interpolation_k,
        interpolation_power=interpolation_power,
    )

    merged["corrected_rate_to_perfect_hz"] = merged["rate_hz"] * merged["lut_scale_factor"]
    merged["selected_z_vector_match"] = True
    selected_z_positions = lut_meta.get("selected_z_positions")
    if selected_z_positions:
        z_match = np.ones(len(merged), dtype=bool)
        for idx, z_value in enumerate(selected_z_positions, start=1):
            z_match &= np.isclose(merged[f"z_pos_{idx}"].astype(float), float(z_value))
        merged["selected_z_vector_match"] = z_match

    output_dataframe = synthetic_dataframe.copy()
    for column in query_columns:
        output_dataframe[column] = merged[column]
    for column in CANONICAL_EFF_COLUMNS:
        output_dataframe[f"lut_{column}"] = merged[f"lut_{column}"]
    output_dataframe["lut_scale_factor"] = merged["lut_scale_factor"]
    output_dataframe["lut_match_method"] = merged["lut_match_method"]
    output_dataframe["lut_match_distance"] = merged["lut_match_distance"]
    output_dataframe["lut_neighbor_count"] = merged["lut_neighbor_count"]
    output_dataframe["lut_neighbor_min_distance"] = merged["lut_neighbor_min_distance"]
    output_dataframe["lut_neighbor_max_distance"] = merged["lut_neighbor_max_distance"]
    output_dataframe["corrected_rate_to_perfect_hz"] = merged["corrected_rate_to_perfect_hz"]
    output_dataframe["selected_z_vector_match"] = merged["selected_z_vector_match"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(output_path, index=False)
    _plot_time_series_with_flux(
        merged,
        PLOTS_DIR / "step3_rate_correction.png",
        rate_column_name=rate_column_name,
    )
    _plot_flux_rate_comparison(
        merged,
        PLOTS_DIR / "step3_flux_rate_comparison.png",
        rate_column_name=rate_column_name,
    )

    metadata = {
        "source_file": str(synthetic_input_path),
        "lut_file": str(lut_path),
        "lut_comments": lut_comments,
        "efficiency_bin_width": efficiency_bin_width,
        "lut_match_mode_requested": match_mode,
        "lut_interpolation_k": interpolation_k,
        "lut_interpolation_power": interpolation_power,
        "row_count": int(len(output_dataframe)),
        "exact_matches": int((output_dataframe["lut_match_method"] == "exact").sum()),
        "nearest_matches": int((output_dataframe["lut_match_method"] == "nearest").sum()),
        "interpolated_matches": int((output_dataframe["lut_match_method"] == "interpolated").sum()),
        "unmatched_rows": int(output_dataframe["lut_scale_factor"].isna().sum()),
        "rows_matching_selected_z_vector": int(output_dataframe["selected_z_vector_match"].sum()),
        "rate_input_column": rate_column_name,
        "time_axis_column_used_for_plots": _resolve_time_axis(merged)[2],
    }
    write_json(meta_path, metadata)

    log.info("Wrote Step 3 output with LUT corrections to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the scale-factor LUT to the synthetic dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
