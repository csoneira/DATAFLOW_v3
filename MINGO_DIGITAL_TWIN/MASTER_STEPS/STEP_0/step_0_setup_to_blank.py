#!/usr/bin/env python3
"""STEP_0: build geometry registry and station-to-geometry map.

Inputs: station configuration CSVs in ONLINE_RUN_DICTIONARY.
Outputs: geometry_registry.(csv|json) and geometry_map_all.(csv|json) in STEP_0_TO_1.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from STEP_SHARED.sim_utils import (
    build_global_geometry_registry,
    ensure_dir,
    list_station_config_files,
    load_step_configs,
    map_station_to_geometry,
    now_iso,
    read_station_config,
    reset_dir,
    resolve_sim_run,
)


def _is_points_key(key: str) -> bool:
    return key.endswith("_points")


def _build_axis(min_max: object, points: int, name: str) -> list[float]:
    if not isinstance(min_max, list) or len(min_max) != 2:
        raise ValueError(f"{name} must be a 2-value list [min, max].")
    if not all(isinstance(v, (int, float)) for v in min_max):
        raise ValueError(f"{name} must contain numeric values.")
    if points <= 0:
        raise ValueError(f"{name}_points must be a positive integer.")
    return [float(v) for v in np.linspace(float(min_max[0]), float(min_max[1]), int(points))]


def build_param_mesh(physics_cfg: dict) -> tuple[pd.DataFrame, dict]:
    axes: list[tuple[str, list[float]]] = []
    efficiencies_identical = bool(physics_cfg.get("efficiencies_identical", False))

    key_aliases = {
        "flux": "flux_cm2_min",
        "cos_n": "cos_n",
        "efficiencies": "efficiencies",
    }
    for key, value in physics_cfg.items():
        if not _is_points_key(key):
            continue
        param = key[: -len("_points")]
        param = key_aliases.get(param, param)
        points = int(value)
        if param not in physics_cfg:
            raise ValueError(f"{key} is set but {param} is missing.")
        values = _build_axis(physics_cfg[param], points, param)
        if param == "efficiencies":
            if efficiencies_identical:
                axes.append(("efficiency_base", values))
            else:
                for plane_idx in range(1, 5):
                    axes.append((f"eff_p{plane_idx}", values))
        else:
            axes.append((param, values))

    if not axes:
        raise ValueError("No mesh axes defined. Add *_points keys to config_step_0_physics.yaml.")

    axis_names = [name for name, _ in axes]
    axis_values = [vals for _, vals in axes]
    rows: list[dict] = []
    start_date = date(1970, 1, 1)
    for idx, combo in enumerate(product(*axis_values), start=1):
        row: dict = {}
        for name, val in zip(axis_names, combo):
            if name == "efficiency_base":
                row["eff_p1"] = float(val)
                row["eff_p2"] = float(val)
                row["eff_p3"] = float(val)
                row["eff_p4"] = float(val)
            else:
                row[name] = float(val)
        row["param_set_id"] = idx
        row["param_date"] = (start_date + timedelta(days=idx - 1)).isoformat()
        rows.append(row)

    mesh = pd.DataFrame(rows)
    meta = {
        "created_at": now_iso(),
        "mesh_axes": axis_names,
        "efficiencies_identical": efficiencies_identical,
        "row_count": len(mesh),
        "start_date": start_date.isoformat(),
        "step": "STEP_0",
    }
    return mesh, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="STEP_0: build geometry registry/map.")
    parser.add_argument(
        "--config",
        default="config_step_0_physics.yaml",
        help="Path to step physics config YAML",
    )
    parser.add_argument(
        "--runtime-config",
        default=None,
        help="Path to step runtime config YAML (defaults to *_runtime.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    runtime_path = Path(args.runtime_config) if args.runtime_config else None
    if runtime_path is not None and not runtime_path.is_absolute():
        runtime_path = Path(__file__).resolve().parent / runtime_path

    physics_cfg, runtime_cfg, cfg, runtime_path = load_step_configs(config_path, runtime_path)

    station_root = Path(
        cfg.get(
            "station_config_root",
            "/home/mingo/DATAFLOW_v3/MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY",
        )
    ).expanduser()
    if not station_root.is_absolute():
        station_root = Path(__file__).resolve().parent / station_root
    if not station_root.exists():
        raise FileNotFoundError(f"station_config_root not found: {station_root}")

    output_dir = Path(cfg.get("output_dir", "../../INTERSTEPS/STEP_0_TO_1"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    station_files = list_station_config_files(station_root)
    if not station_files:
        raise FileNotFoundError(f"No station config CSVs found under {station_root}")

    station_dfs = [read_station_config(path) for path in station_files.values()]
    registry = build_global_geometry_registry(station_dfs)
    geometry_map = pd.concat(
        [map_station_to_geometry(df, registry) for df in station_dfs],
        ignore_index=True,
    )

    sim_run, sim_run_dir, config_hash, _, _ = resolve_sim_run(
        output_dir, "STEP_0", config_path, physics_cfg, None
    )
    reset_dir(sim_run_dir)

    registry_path = sim_run_dir / "geometry_registry.csv"
    registry.to_csv(registry_path, index=False)
    geom_map_path = sim_run_dir / "geometry_map_all.csv"
    geometry_map.to_csv(geom_map_path, index=False)

    mesh, mesh_meta = build_param_mesh(physics_cfg)
    mesh_path = sim_run_dir / "param_mesh.csv"
    mesh.to_csv(mesh_path, index=False)
    mesh_meta.update(
        {
            "config": physics_cfg,
            "config_hash": config_hash,
            "sim_run": sim_run,
            "output_path": str(mesh_path),
        }
    )
    (sim_run_dir / "param_mesh_metadata.json").write_text(json.dumps(mesh_meta, indent=2))

    meta = {
        "created_at": now_iso(),
        "step": "STEP_0",
        "config": physics_cfg,
        "runtime_config": runtime_cfg,
        "config_hash": config_hash,
        "sim_run": sim_run,
        "station_config_root": str(station_root),
    }
    (sim_run_dir / "geometry_map_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved geometry registry/map and param mesh to {sim_run_dir}")


if __name__ == "__main__":
    main()
