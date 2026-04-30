#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from common import DEFAULT_CONFIG_PATH, PLOTS_DIR, cfg_path, ensure_output_dirs, load_config, write_json
from multi_z_support import (
    add_z_config_columns,
    format_z_vector,
    json_clone,
    resolve_active_z_vectors_from_config,
    z_vector_to_id,
)
from step_1_prepare_data_single_z import run as run_single_z

log = logging.getLogger("another_method_advanced.step1")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_1 - %(message)s", level=logging.INFO, force=True)


def _per_z_dir(output_path: Path, z_config_id: str) -> Path:
    return output_path.parent / "multi_z" / z_config_id / "step1"


def _move_plot_if_exists(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    return str(destination)


def _build_subconfig(
    config: dict,
    *,
    z_vector: tuple[float, float, float, float],
    step_dir: Path,
) -> Path:
    subconfig = json_clone(config)
    subconfig.setdefault("step1", {})
    subconfig["step1"]["z_position_vector"] = list(z_vector)
    subconfig["step1"]["z_selection_mode"] = "configured_vector"
    subconfig["paths"]["step1_filtered_csv"] = str(step_dir / "step1_filtered_data.csv")
    subconfig["paths"]["step1_meta_json"] = str(step_dir / "step1_selection_meta.json")
    config_path = step_dir / "step1_single_z_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(subconfig, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    output_path = cfg_path(config, "paths", "step1_filtered_csv")
    meta_path = cfg_path(config, "paths", "step1_meta_json")

    requested_z_vectors, selection_meta = resolve_active_z_vectors_from_config(config)
    if not requested_z_vectors:
        raise ValueError(
            "No active z configurations were found for the requested station/date window. "
            "Set step5.station/date_from/date_to or force step1.z_selection_mode='configured_vector'."
        )

    combined_frames: list[pd.DataFrame] = []
    used_configs: list[dict[str, object]] = []
    skipped_configs: list[dict[str, object]] = []
    row_counts_by_z: dict[str, int] = {}

    for z_vector in requested_z_vectors:
        z_config_id = z_vector_to_id(z_vector)
        step_dir = _per_z_dir(output_path, z_config_id)
        subconfig_path = _build_subconfig(config, z_vector=z_vector, step_dir=step_dir)

        try:
            run_single_z(subconfig_path)
        except Exception as exc:  # pragma: no cover - exercised through orchestration integration
            message = str(exc)
            skipped_configs.append(
                {
                    "z_config_id": z_config_id,
                    "z_positions": [float(value) for value in z_vector],
                    "reason": message,
                }
            )
            log.warning("Skipping z configuration %s because Step 1 failed: %s", format_z_vector(z_vector), message)
            continue

        filtered_path = step_dir / "step1_filtered_data.csv"
        submeta_path = step_dir / "step1_selection_meta.json"
        if not filtered_path.exists():
            skipped_configs.append(
                {
                    "z_config_id": z_config_id,
                    "z_positions": [float(value) for value in z_vector],
                    "reason": f"Expected Step 1 output not found: {filtered_path}",
                }
            )
            log.warning("Skipping z configuration %s because no Step 1 CSV was produced.", format_z_vector(z_vector))
            continue

        dataframe = add_z_config_columns(pd.read_csv(filtered_path, low_memory=False))
        combined_frames.append(dataframe)
        row_counts_by_z[z_config_id] = int(len(dataframe))

        plot_path = _move_plot_if_exists(
            PLOTS_DIR / "step1_parameter_space_overview.png",
            PLOTS_DIR / f"step1_parameter_space_overview__{z_config_id}.png",
        )
        submeta = {}
        if submeta_path.exists():
            submeta = json.loads(submeta_path.read_text(encoding="utf-8"))
        used_configs.append(
            {
                "z_config_id": z_config_id,
                "z_positions": [float(value) for value in z_vector],
                "row_count": int(len(dataframe)),
                "step1_filtered_csv": str(filtered_path),
                "step1_meta_json": str(submeta_path),
                "parameter_space_plot": plot_path,
                "single_z_metadata": submeta,
            }
        )
        log.info("Prepared Step 1 data for z = %s with %d rows", format_z_vector(z_vector), len(dataframe))

    if not combined_frames:
        raise ValueError("Step 1 could not prepare any z configuration in the requested window.")

    combined = pd.concat(combined_frames, ignore_index=True)
    sort_columns = ["z_config_id", "emp_eff_1", "emp_eff_2", "emp_eff_3", "emp_eff_4", "sim_flux_cm2_min"]
    available_sort_columns = [column for column in sort_columns if column in combined.columns]
    if available_sort_columns:
        combined = combined.sort_values(available_sort_columns, kind="mergesort").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    metadata = {
        **selection_meta,
        "selected_z_positions": [
            [float(value) for value in z_vector]
            for z_vector in requested_z_vectors
        ],
        "z_configuration_count_requested": int(len(requested_z_vectors)),
        "z_configuration_count_used": int(len(used_configs)),
        "z_configuration_count_skipped": int(len(skipped_configs)),
        "z_configurations_used": used_configs,
        "z_configurations_skipped": skipped_configs,
        "row_count": int(len(combined)),
        "row_counts_by_z_configuration": row_counts_by_z,
        "combined_output_csv": str(output_path),
    }
    write_json(meta_path, metadata)

    log.info(
        "Wrote %d filtered rows across %d z configurations to %s",
        len(combined),
        len(used_configs),
        output_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-z training data for the advanced LUT workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
