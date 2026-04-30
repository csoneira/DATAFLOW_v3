#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from common import DEFAULT_CONFIG_PATH, PLOTS_DIR, cfg_path, ensure_output_dirs, load_config, write_json
from multi_z_support import json_clone
from step_4_study_lut_single_z import run as run_single_z

log = logging.getLogger("another_method_advanced.step4")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_4 - %(message)s", level=logging.INFO, force=True)


def _load_step2_configs(step2_meta_path: Path) -> list[dict[str, object]]:
    if not step2_meta_path.exists():
        return []
    metadata = json.loads(step2_meta_path.read_text(encoding="utf-8"))
    used = metadata.get("z_configurations_used")
    if not isinstance(used, list):
        return []
    return [item for item in used if isinstance(item, dict)]


def _move_plot_if_exists(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    return str(destination)


def _build_subconfig(
    config: dict,
    *,
    lut_diag_csv: str,
    z_positions: list[float],
    step4_dir: Path,
) -> Path:
    subconfig = json_clone(config)
    subconfig.setdefault("step1", {})
    subconfig["step1"]["z_position_vector"] = list(z_positions)
    subconfig["step1"]["z_selection_mode"] = "configured_vector"
    subconfig["paths"]["step2_lut_diagnostics_csv"] = str(lut_diag_csv)
    subconfig["paths"]["step4_axis_slice_csv"] = str(step4_dir / "step4_axis_slice_study.csv")
    subconfig["paths"]["step4_pair_slice_csv"] = str(step4_dir / "step4_pair_slice_study.csv")
    subconfig["paths"]["step4_meta_json"] = str(step4_dir / "step4_study_meta.json")
    config_path = step4_dir / "step4_single_z_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(subconfig, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    step2_meta_path = cfg_path(config, "paths", "step2_meta_json")
    axis_csv_path = cfg_path(config, "paths", "step4_axis_slice_csv")
    pair_csv_path = cfg_path(config, "paths", "step4_pair_slice_csv")
    meta_path = cfg_path(config, "paths", "step4_meta_json")

    step2_configs = _load_step2_configs(step2_meta_path)
    if not step2_configs:
        raise ValueError("Step 4 could not find any Step 2 z-configuration outputs to study.")

    combined_axis: list[pd.DataFrame] = []
    combined_pair: list[pd.DataFrame] = []
    used_configs: list[dict[str, object]] = []
    skipped_configs: list[dict[str, object]] = []

    for item in step2_configs:
        z_config_id = str(item["z_config_id"])
        z_positions = [float(value) for value in item["z_positions"]]
        lut_diag_csv = str(item["lut_diagnostics_csv"])
        step4_dir = axis_csv_path.parent / "multi_z" / z_config_id / "step4"
        subconfig_path = _build_subconfig(
            config,
            lut_diag_csv=lut_diag_csv,
            z_positions=z_positions,
            step4_dir=step4_dir,
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
            log.warning("Skipping z configuration %s because Step 4 failed: %s", z_positions, exc)
            continue

        sub_axis_path = step4_dir / "step4_axis_slice_study.csv"
        sub_pair_path = step4_dir / "step4_pair_slice_study.csv"
        sub_meta_path = step4_dir / "step4_study_meta.json"
        if not (sub_axis_path.exists() and sub_pair_path.exists()):
            skipped_configs.append(
                {
                    "z_config_id": z_config_id,
                    "z_positions": z_positions,
                    "reason": "Expected Step 4 outputs were not produced.",
                }
            )
            log.warning("Skipping z configuration %s because Step 4 outputs are incomplete.", z_positions)
            continue

        axis_df = pd.read_csv(sub_axis_path, low_memory=False)
        pair_df = pd.read_csv(sub_pair_path, low_memory=False)
        for frame in (axis_df, pair_df):
            frame["z_config_id"] = z_config_id
            for idx, value in enumerate(z_positions, start=1):
                frame[f"z_pos_{idx}"] = float(value)

        combined_axis.append(axis_df)
        combined_pair.append(pair_df)

        used_configs.append(
            {
                "z_config_id": z_config_id,
                "z_positions": z_positions,
                "axis_slice_csv": str(sub_axis_path),
                "pair_slice_csv": str(sub_pair_path),
                "step4_meta_json": str(sub_meta_path),
                "plot_axis_slice_scale_factor": _move_plot_if_exists(
                    PLOTS_DIR / "step4_axis_slice_scale_factor.png",
                    PLOTS_DIR / f"step4_axis_slice_scale_factor__{z_config_id}.png",
                ),
                "plot_axis_slice_relative_rate": _move_plot_if_exists(
                    PLOTS_DIR / "step4_axis_slice_relative_rate.png",
                    PLOTS_DIR / f"step4_axis_slice_relative_rate__{z_config_id}.png",
                ),
                "plot_pair_slice_surface": _move_plot_if_exists(
                    PLOTS_DIR / "step4_pair_slice_surface.png",
                    PLOTS_DIR / f"step4_pair_slice_surface__{z_config_id}.png",
                ),
                "plot_pair_slice_quality": _move_plot_if_exists(
                    PLOTS_DIR / "step4_pair_slice_quality.png",
                    PLOTS_DIR / f"step4_pair_slice_quality__{z_config_id}.png",
                ),
            }
        )
        log.info("Studied LUT slices for z = %s", z_positions)

    if not combined_axis:
        raise ValueError("Step 4 could not produce any slice diagnostics.")

    axis_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(combined_axis, ignore_index=True).to_csv(axis_csv_path, index=False)
    pd.concat(combined_pair, ignore_index=True).to_csv(pair_csv_path, index=False)

    metadata = {
        "z_configuration_count_used": int(len(used_configs)),
        "z_configuration_count_skipped": int(len(skipped_configs)),
        "z_configurations_used": used_configs,
        "z_configurations_skipped": skipped_configs,
        "axis_slice_csv": str(axis_csv_path),
        "pair_slice_csv": str(pair_csv_path),
    }
    write_json(meta_path, metadata)

    log.info(
        "Wrote combined Step 4 diagnostics across %d z configurations to %s",
        len(used_configs),
        axis_csv_path,
    )
    return axis_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Study multi-z LUT slices for the advanced workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
