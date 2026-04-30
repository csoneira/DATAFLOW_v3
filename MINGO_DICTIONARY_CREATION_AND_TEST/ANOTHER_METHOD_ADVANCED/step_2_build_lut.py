#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from common import CANONICAL_EFF_COLUMNS, CANONICAL_Z_COLUMNS, DEFAULT_CONFIG_PATH, PLOTS_DIR, cfg_path, ensure_output_dirs, load_config, read_ascii_lut, write_json
from multi_z_support import build_rate_to_flux_lines, json_clone, write_ascii_table_with_comments, z_vector_to_id
from step_2_build_lut_single_z import run as run_single_z

log = logging.getLogger("another_method_advanced.step2")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_2 - %(message)s", level=logging.INFO, force=True)


def _build_lut_ascii_export(lut: pd.DataFrame) -> pd.DataFrame:
    required_columns = [*CANONICAL_EFF_COLUMNS, "scale_factor"]
    missing = [column for column in required_columns if column not in lut.columns]
    if missing:
        raise ValueError("LUT export is missing required columns: " + ", ".join(missing))
    z_columns = [column for column in CANONICAL_Z_COLUMNS if column in lut.columns]
    export_columns = [*z_columns, *CANONICAL_EFF_COLUMNS, "scale_factor"]
    return lut[export_columns].copy()


def _step2_dir(step1_output_path: Path, z_config_id: str) -> Path:
    return step1_output_path.parent / "multi_z" / z_config_id / "step2"


def _load_step1_configs(step1_meta_path: Path, step1_output_path: Path) -> list[dict[str, object]]:
    if step1_meta_path.exists():
        metadata = json.loads(step1_meta_path.read_text(encoding="utf-8"))
        used = metadata.get("z_configurations_used")
        if isinstance(used, list):
            return [item for item in used if isinstance(item, dict)]

    dataframe = pd.read_csv(step1_output_path, low_memory=False)
    configs: list[dict[str, object]] = []
    for z_config_id, subset in dataframe.groupby("z_config_id", dropna=False, sort=True):
        if not isinstance(z_config_id, str):
            continue
        row = subset.iloc[0]
        z_positions = [float(row[column]) for column in CANONICAL_Z_COLUMNS]
        step1_dir = step1_output_path.parent / "multi_z" / z_config_id / "step1"
        configs.append(
            {
                "z_config_id": z_config_id,
                "z_positions": z_positions,
                "step1_filtered_csv": str(step1_dir / "step1_filtered_data.csv"),
                "step1_meta_json": str(step1_dir / "step1_selection_meta.json"),
            }
        )
    return configs


def _move_plot_if_exists(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    return str(destination)


def _build_subconfig(
    config: dict,
    *,
    step1_filtered_csv: str,
    step1_meta_json: str,
    z_positions: list[float],
    step2_dir: Path,
) -> Path:
    subconfig = json_clone(config)
    subconfig.setdefault("step1", {})
    subconfig["step1"]["z_position_vector"] = list(z_positions)
    subconfig["step1"]["z_selection_mode"] = "configured_vector"
    subconfig["paths"]["step1_filtered_csv"] = str(step1_filtered_csv)
    subconfig["paths"]["step1_meta_json"] = str(step1_meta_json)
    subconfig["paths"]["step2_flux_cells_csv"] = str(step2_dir / "step2_flux_binned_cells.csv")
    subconfig["paths"]["step2_lut_diagnostics_csv"] = str(step2_dir / "step2_lut_diagnostics.csv")
    subconfig["paths"]["step2_lut_ascii"] = str(step2_dir / "step2_scale_factor_lut.txt")
    subconfig["paths"]["step2_meta_json"] = str(step2_dir / "step2_lut_meta.json")
    config_path = step2_dir / "step2_single_z_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(subconfig, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    step1_output_path = cfg_path(config, "paths", "step1_filtered_csv")
    step1_meta_path = cfg_path(config, "paths", "step1_meta_json")
    flux_cells_path = cfg_path(config, "paths", "step2_flux_cells_csv")
    lut_diag_path = cfg_path(config, "paths", "step2_lut_diagnostics_csv")
    lut_ascii_path = cfg_path(config, "paths", "step2_lut_ascii")
    rate_to_flux_lines_path = cfg_path(config, "paths", "step2_rate_to_flux_lines_csv")
    meta_path = cfg_path(config, "paths", "step2_meta_json")

    step1_configs = _load_step1_configs(step1_meta_path, step1_output_path)
    if not step1_configs:
        raise ValueError("Step 2 could not find any prepared Step 1 z-configuration outputs.")

    combined_flux_cells: list[pd.DataFrame] = []
    combined_lut_diagnostics: list[pd.DataFrame] = []
    combined_lut_rows: list[pd.DataFrame] = []
    used_configs: list[dict[str, object]] = []
    skipped_configs: list[dict[str, object]] = []

    step2_meta_comments: list[str] = []
    rate_column_name: str | None = None
    selected_trigger: object = None

    for item in step1_configs:
        z_config_id = str(item["z_config_id"])
        z_positions = [float(value) for value in item["z_positions"]]
        step2_dir = _step2_dir(step1_output_path, z_config_id)
        subconfig_path = _build_subconfig(
            config,
            step1_filtered_csv=str(item["step1_filtered_csv"]),
            step1_meta_json=str(item["step1_meta_json"]),
            z_positions=z_positions,
            step2_dir=step2_dir,
        )

        try:
            run_single_z(subconfig_path)
        except Exception as exc:  # pragma: no cover - exercised through orchestration integration
            skipped_configs.append(
                {
                    "z_config_id": z_config_id,
                    "z_positions": z_positions,
                    "reason": str(exc),
                }
            )
            log.warning("Skipping z configuration %s because Step 2 failed: %s", z_positions, exc)
            continue

        sub_flux_path = step2_dir / "step2_flux_binned_cells.csv"
        sub_lut_diag_path = step2_dir / "step2_lut_diagnostics.csv"
        sub_lut_ascii_path = step2_dir / "step2_scale_factor_lut.txt"
        sub_meta_path = step2_dir / "step2_lut_meta.json"
        if not (sub_flux_path.exists() and sub_lut_diag_path.exists() and sub_lut_ascii_path.exists()):
            skipped_configs.append(
                {
                    "z_config_id": z_config_id,
                    "z_positions": z_positions,
                    "reason": "Expected Step 2 outputs were not produced.",
                }
            )
            log.warning("Skipping z configuration %s because Step 2 outputs are incomplete.", z_positions)
            continue

        flux_cells = pd.read_csv(sub_flux_path, low_memory=False)
        lut_diagnostics = pd.read_csv(sub_lut_diag_path, low_memory=False)
        lut_ascii, comments = read_ascii_lut(sub_lut_ascii_path)
        submeta = json.loads(sub_meta_path.read_text(encoding="utf-8")) if sub_meta_path.exists() else {}

        for frame in (flux_cells, lut_diagnostics, lut_ascii):
            for column, value in zip(CANONICAL_Z_COLUMNS, z_positions):
                frame[column] = float(value)
            frame["z_config_id"] = z_config_id

        combined_flux_cells.append(flux_cells)
        combined_lut_diagnostics.append(lut_diagnostics)
        combined_lut_rows.append(lut_ascii)

        if not step2_meta_comments:
            step2_meta_comments = comments
        if rate_column_name is None:
            rate_column_name = submeta.get("rate_input_column")
        if selected_trigger is None:
            selected_trigger = submeta.get("selected_trigger")

        used_configs.append(
            {
                "z_config_id": z_config_id,
                "z_positions": z_positions,
                "lut_rows": int(len(lut_ascii)),
                "flux_cell_rows": int(len(flux_cells)),
                "lut_diagnostics_csv": str(sub_lut_diag_path),
                "flux_cells_csv": str(sub_flux_path),
                "lut_ascii": str(sub_lut_ascii_path),
                "step2_meta_json": str(sub_meta_path),
                "plot_rate_vs_flux": _move_plot_if_exists(
                    PLOTS_DIR / "step2_rate_vs_flux.png",
                    PLOTS_DIR / f"step2_rate_vs_flux__{z_config_id}.png",
                ),
                "plot_scale_factor_vs_diagonal_eff": _move_plot_if_exists(
                    PLOTS_DIR / "step2_scale_factor_vs_diagonal_eff.png",
                    PLOTS_DIR / f"step2_scale_factor_vs_diagonal_eff__{z_config_id}.png",
                ),
                "plot_scale_factor_vs_flux": _move_plot_if_exists(
                    PLOTS_DIR / "step2_scale_factor_vs_flux.png",
                    PLOTS_DIR / f"step2_scale_factor_vs_flux__{z_config_id}.png",
                ),
                "single_z_metadata": submeta,
            }
        )
        log.info("Built LUT for z = %s with %d rows", z_positions, len(lut_ascii))

    if not combined_lut_rows:
        raise ValueError("Step 2 could not build any LUT for the requested z configurations.")

    combined_flux = pd.concat(combined_flux_cells, ignore_index=True)
    combined_lut_diag = pd.concat(combined_lut_diagnostics, ignore_index=True)
    combined_lut = pd.concat(combined_lut_rows, ignore_index=True)

    flux_cells_path.parent.mkdir(parents=True, exist_ok=True)
    combined_flux.to_csv(flux_cells_path, index=False)
    combined_lut_diag.to_csv(lut_diag_path, index=False)
    rate_to_flux_lines = build_rate_to_flux_lines(combined_flux)
    rate_to_flux_lines.to_csv(rate_to_flux_lines_path, index=False)

    lut_export = _build_lut_ascii_export(combined_lut)
    comments = [
        "# z_positions: multi",
        f"# z_configuration_count: {len(used_configs)}",
        "# z_config_ids: " + json.dumps([item["z_config_id"] for item in used_configs]),
        "# trigger: " + ("null" if selected_trigger is None else json.dumps(selected_trigger)),
        "# rate_column: " + ("null" if rate_column_name in (None, "", "null", "None") else str(rate_column_name)),
    ]
    write_ascii_table_with_comments(lut_ascii_path, lut_export, comments=comments)

    metadata = {
        "selected_z_positions": [item["z_positions"] for item in used_configs],
        "z_configuration_count_used": int(len(used_configs)),
        "z_configuration_count_skipped": int(len(skipped_configs)),
        "z_configurations_used": used_configs,
        "z_configurations_skipped": skipped_configs,
        "lut_rows": int(len(combined_lut_diag)),
        "flux_cell_rows": int(len(combined_flux)),
        "lut_file": str(lut_ascii_path),
        "flux_cells_file": str(flux_cells_path),
        "lut_diagnostics_file": str(lut_diag_path),
        "rate_to_flux_lines_file": str(rate_to_flux_lines_path),
        "rate_to_flux_lines": rate_to_flux_lines.to_dict(orient="records"),
        "single_z_comment_template": step2_meta_comments,
    }
    write_json(meta_path, metadata)

    log.info(
        "Wrote combined LUT with %d rows across %d z configurations to %s",
        len(combined_lut_diag),
        len(used_configs),
        lut_ascii_path,
    )
    return lut_ascii_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-z LUT for the advanced workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
