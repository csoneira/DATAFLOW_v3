#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from common import (
    CANONICAL_Z_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    cfg_path,
    ensure_output_dirs,
    get_trigger_type_selection,
    load_config,
    ordered_plot_filename,
    write_json,
)
from multi_z_support import (
    apply_rate_to_flux_lines,
    build_rate_to_flux_lines,
    load_rate_to_flux_lines,
    load_reference_curve_table,
    unique_z_vectors,
)
from step_5_apply_lut_to_real_data import (
    _load_event_markers_for_station,
    _parse_station_id,
    _plot_corrected_flux_from_rate,
    _resolve_plot_moving_average,
)

log = logging.getLogger("another_method.step6")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_6 - %(message)s", level=logging.INFO, force=True)


def _ensure_canonical_z_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    work = dataframe.copy()
    for idx, target_column in enumerate(CANONICAL_Z_COLUMNS, start=1):
        if target_column in work.columns:
            continue
        online_column = f"online_z_plane_{idx}"
        specific_column = f"z_P{idx}"
        if online_column in work.columns:
            work[target_column] = pd.to_numeric(work[online_column], errors="coerce")
        elif specific_column in work.columns:
            work[target_column] = pd.to_numeric(work[specific_column], errors="coerce")
    return work


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    step5_output_path = cfg_path(config, "paths", "step5_output_csv")
    output_path = cfg_path(config, "paths", "step6_output_csv")
    meta_path = cfg_path(config, "paths", "step6_meta_json")
    flux_cells_path = cfg_path(config, "paths", "step2_flux_cells_csv")
    rate_to_flux_lines_path = cfg_path(config, "paths", "step2_rate_to_flux_lines_csv")

    dataframe = _ensure_canonical_z_columns(pd.read_csv(step5_output_path, low_memory=False))
    if "corrected_rate_to_perfect_hz" not in dataframe.columns:
        raise ValueError(
            "Step 6 requires corrected_rate_to_perfect_hz in the Step 5 output. "
            "Rerun Step 5 with the current pipeline first."
        )

    try:
        line_table = load_rate_to_flux_lines(rate_to_flux_lines_path)
        line_source = str(rate_to_flux_lines_path)
    except FileNotFoundError:
        reference_table = load_reference_curve_table(flux_cells_path)
        line_table = build_rate_to_flux_lines(reference_table)
        line_source = f"rebuilt_from:{flux_cells_path}"

    row_z_frame = None
    if all(column in line_table.columns for column in CANONICAL_Z_COLUMNS):
        missing_z_columns = [column for column in CANONICAL_Z_COLUMNS if column not in dataframe.columns]
        if missing_z_columns:
            raise ValueError(
                "Step 6 needs z-position columns to map rate to flux with the selected LUT lines, but these are missing: "
                + ", ".join(missing_z_columns)
            )
        row_z_frame = dataframe[CANONICAL_Z_COLUMNS]

    output_dataframe = dataframe.copy()
    (
        output_dataframe["corrected_flux_cm2_min"],
        output_dataframe["corrected_flux_assignment_method"],
    ) = apply_rate_to_flux_lines(
        pd.to_numeric(output_dataframe["corrected_rate_to_perfect_hz"], errors="coerce"),
        row_z_frame=row_z_frame,
        line_table=line_table,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(output_path, index=False)

    step5_config = config.get("step5", {})
    station_id = _parse_station_id(step5_config.get("station", "MINGO01"))
    event_markers = _load_event_markers_for_station(station_id)
    apply_plot_moving_average, moving_average_kernel = _resolve_plot_moving_average(config)
    trigger_selection = get_trigger_type_selection(config)
    rate_column_name = str(trigger_selection["selected_source_rate_column"])
    _plot_corrected_flux_from_rate(
        output_dataframe,
        line_table,
        PLOTS_DIR / ordered_plot_filename(6, 1, "corrected_flux_from_rate"),
        rate_column_name=rate_column_name,
        event_markers=event_markers,
        apply_moving_average=apply_plot_moving_average,
        moving_average_kernel=moving_average_kernel,
    )

    metadata = {
        "source_step5_file": str(step5_output_path),
        "output_file": str(output_path),
        "rate_to_flux_lines_source": line_source,
        "row_count": int(len(output_dataframe)),
        "station_id": int(station_id),
        "plot_apply_moving_average": apply_plot_moving_average,
        "plot_moving_average_kernel": moving_average_kernel,
        "line_table_rows": int(len(line_table)),
        "line_table_z_configurations": [
            [float(value) for value in z_vector]
            for z_vector in unique_z_vectors(line_table, z_columns=CANONICAL_Z_COLUMNS)
        ] if all(column in line_table.columns for column in CANONICAL_Z_COLUMNS) else [],
        "flux_assignment_method_counts": {
            str(key): int(value)
            for key, value in output_dataframe["corrected_flux_assignment_method"].value_counts(dropna=False).items()
        },
    }
    write_json(meta_path, metadata)

    log.info("Wrote Step 6 flux-converted real-data output to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Step 5 corrected rates into corrected flux values.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
