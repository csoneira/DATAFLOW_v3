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
    CANONICAL_Z_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    STEP1_OUTPUT_COLUMNS,
    canonicalize_trigger_value,
    cfg_path,
    choose_z_vector,
    ensure_output_dirs,
    filter_by_z_vector,
    get_rate_column_name,
    load_config,
    write_json,
)

log = logging.getLogger("another_method.step1")

TRIGGER_COLUMN = "trigger_combinations"
STEP1_PLOT_PATH = PLOTS_DIR / "step1_parameter_space_overview.png"


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_1 - %(message)s", level=logging.INFO, force=True)


def _display_label(column: str) -> str:
    labels = {
        "sim_flux_cm2_min": "Flux [cm^-2 min^-1]",
        "emp_eff_1": "Plane 1 eff",
        "emp_eff_2": "Plane 2 eff",
        "emp_eff_3": "Plane 3 eff",
        "emp_eff_4": "Plane 4 eff",
    }
    return labels.get(column, column.replace("_", " "))


def _compute_axis_limits(series: pd.Series) -> tuple[float, float] | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    lower = float(numeric.min())
    upper = float(numeric.max())
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if lower == upper:
        pad = max(abs(lower) * 0.05, 0.05)
        return lower - pad, upper + pad
    pad = (upper - lower) * 0.04
    return lower - pad, upper + pad


def _write_parameter_space_plot(dataframe: pd.DataFrame, selected_z_vector: tuple[float, float, float, float]) -> Path | None:
    plot_columns = ["sim_flux_cm2_min"] + CANONICAL_EFF_COLUMNS
    plot_df = dataframe.loc[:, plot_columns].copy()
    for column in plot_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(how="all", subset=plot_columns).reset_index(drop=True)
    if plot_df.empty:
        return None

    axis_limits = {column: _compute_axis_limits(plot_df[column]) for column in plot_columns}
    plot_columns = [column for column in plot_columns if axis_limits.get(column) is not None]
    if not plot_columns:
        return None

    n = len(plot_columns)
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(2.45 * n + 1.2, 2.45 * n + 1.4),
        dpi=130,
    )
    axes_arr = np.asarray(axes).reshape(n, n)
    color = "#4C78A8"

    for row_idx in range(n):
        for col_idx in range(n):
            ax = axes_arr[row_idx, col_idx]
            x_col = plot_columns[col_idx]
            y_col = plot_columns[row_idx]

            ax.grid(True, linestyle=":", linewidth=0.35, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            ax.set_box_aspect(1.0)

            if row_idx == col_idx:
                values = plot_df[y_col].dropna()
                if not values.empty:
                    ax.hist(
                        values,
                        bins=18,
                        color=color,
                        alpha=0.55,
                        edgecolor="white",
                        linewidth=0.45,
                    )
                ax.set_title(_display_label(y_col), fontsize=9, pad=4)
                ax.set_xlim(axis_limits[y_col])
                ax.autoscale_view(scalex=False, scaley=True)
            elif row_idx > col_idx:
                subset = plot_df[[x_col, y_col]].dropna()
                if not subset.empty:
                    ax.scatter(
                        subset[x_col],
                        subset[y_col],
                        s=16.0,
                        alpha=0.68,
                        color=color,
                        edgecolors="none",
                        rasterized=True,
                    )
                ax.set_xlim(axis_limits[x_col])
                ax.set_ylim(axis_limits[y_col])
            else:
                ax.axis("off")

            if row_idx < n - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(_display_label(x_col), fontsize=8)
            if col_idx > 0:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_ylabel(_display_label(y_col), fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7, width=0.75, length=2.6)

    z_text = ", ".join(f"{float(value):g}" for value in selected_z_vector)
    fig.suptitle(
        "Step 1 parameter-space overview\n"
        f"selected z = [{z_text}] | rows = {len(plot_df)}",
        fontsize=11,
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.9, wspace=0.05, hspace=0.05)

    STEP1_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEP1_PLOT_PATH, dpi=170)
    plt.close(fig)
    return STEP1_PLOT_PATH


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    input_path = cfg_path(config, "paths", "collected_data_csv")
    output_path = cfg_path(config, "paths", "step1_filtered_csv")
    meta_path = cfg_path(config, "paths", "step1_meta_json")

    eff_columns = list(config["columns"]["efficiencies"])
    z_columns = list(config["columns"]["z_positions"])
    rate_column = get_rate_column_name(config)
    flux_column = str(config["columns"]["simulated_flux"])
    num_events_column = str(config["columns"]["num_events"])
    min_events = int(config.get("step1", {}).get("min_events", 70000))
    configured_z_vector = config.get("step1", {}).get("z_position_vector")

    required_columns = eff_columns + z_columns + [rate_column, flux_column, num_events_column]
    candidate_columns = required_columns + [TRIGGER_COLUMN]
    available_columns = set(pd.read_csv(input_path, nrows=0).columns)
    missing_required = [column for column in required_columns if column not in available_columns]
    if missing_required:
        raise ValueError(
            "Step 1 input is missing required columns: " + ", ".join(missing_required)
        )
    usecols = [column for column in candidate_columns if column in available_columns]
    dataframe = pd.read_csv(input_path, usecols=usecols)
    initial_rows = int(len(dataframe))

    selected_z_vector = choose_z_vector(dataframe, z_columns, configured_z_vector)
    dataframe = filter_by_z_vector(dataframe, z_columns, selected_z_vector)
    rows_after_z_filter = int(len(dataframe))
    dataframe = dataframe[dataframe[num_events_column].astype(float) >= min_events].copy()
    rows_after_event_filter = int(len(dataframe))

    selected_trigger = None
    trigger_values: list[object] = []
    if TRIGGER_COLUMN in dataframe.columns:
        seen: set[str] = set()
        for raw_value in dataframe[TRIGGER_COLUMN]:
            parsed_value = canonicalize_trigger_value(raw_value)
            if parsed_value is None:
                continue
            key = json.dumps(parsed_value, sort_keys=True) if not isinstance(parsed_value, str) else parsed_value
            if key in seen:
                continue
            seen.add(key)
            trigger_values.append(parsed_value)
        if trigger_values:
            selected_trigger = trigger_values[0]

    rename_map = {
        z_columns[0]: CANONICAL_Z_COLUMNS[0],
        z_columns[1]: CANONICAL_Z_COLUMNS[1],
        z_columns[2]: CANONICAL_Z_COLUMNS[2],
        z_columns[3]: CANONICAL_Z_COLUMNS[3],
        eff_columns[0]: CANONICAL_EFF_COLUMNS[0],
        eff_columns[1]: CANONICAL_EFF_COLUMNS[1],
        eff_columns[2]: CANONICAL_EFF_COLUMNS[2],
        eff_columns[3]: CANONICAL_EFF_COLUMNS[3],
        rate_column: "rate_hz",
        flux_column: "sim_flux_cm2_min",
        num_events_column: "num_events",
    }
    dataframe = dataframe.rename(columns=rename_map)[STEP1_OUTPUT_COLUMNS]
    dataframe = dataframe.sort_values(CANONICAL_EFF_COLUMNS + ["sim_flux_cm2_min"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    plot_path = _write_parameter_space_plot(dataframe, selected_z_vector)

    metadata = {
        "selected_z_positions": list(selected_z_vector),
        "selected_trigger": selected_trigger,
        "trigger_values_in_filtered_data": trigger_values,
        "parameter_space_plot": None if plot_path is None else str(plot_path),
        "source_file": str(input_path),
        "row_counts": {
            "initial": initial_rows,
            "after_z_filter": rows_after_z_filter,
            "after_min_events_filter": rows_after_event_filter,
        },
        "min_events": min_events,
        "input_columns": {
            "efficiencies": eff_columns,
            "z_positions": z_columns,
            "rate": rate_column,
            "simulated_flux": flux_column,
            "num_events": num_events_column,
        },
        "configured_z_position_vector": configured_z_vector,
    }
    write_json(meta_path, metadata)

    log.info("Wrote %d filtered rows to %s", len(dataframe), output_path)
    log.info("Selected z-position vector: %s", selected_z_vector)
    if plot_path is not None:
        log.info("Wrote plot: %s", plot_path)
    if selected_trigger is not None:
        log.info("Selected trigger: %s", selected_trigger)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter the collected data for the LUT workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
