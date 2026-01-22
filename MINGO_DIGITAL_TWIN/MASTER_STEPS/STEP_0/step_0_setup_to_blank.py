#!/usr/bin/env python3
"""STEP_0: append one parameter row with sampled z positions.

Inputs: station configuration CSVs in ONLINE_RUN_DICTIONARY.
Outputs: param_mesh.csv in STEP_0_TO_1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from STEP_SHARED.sim_utils import (
    ensure_dir,
    list_station_config_files,
    load_step_configs,
    now_iso,
    read_station_config,
)


def _sample_range(rng: np.random.Generator, value: object, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        return float(rng.uniform(lo, hi))
    raise ValueError(f"{name} must be a number or a 2-value list [min, max].")


def _collect_z_positions(station_files: dict[int, Path]) -> pd.DataFrame:
    station_dfs = [read_station_config(path) for path in station_files.values()]
    geom_cols = ["P1", "P2", "P3", "P4"]
    all_geoms = pd.concat([df[geom_cols] for df in station_dfs], ignore_index=True)
    unique_geoms = all_geoms.dropna().drop_duplicates().reset_index(drop=True)
    return unique_geoms


def _append_param_row(
    mesh_path: Path,
    meta_path: Path,
    physics_cfg: dict,
    rng: np.random.Generator,
    z_positions: pd.DataFrame,
) -> None:
    efficiencies_identical = bool(physics_cfg.get("efficiencies_identical", False))
    cos_n = _sample_range(rng, physics_cfg.get("cos_n"), "cos_n")
    flux_cm2_min = _sample_range(rng, physics_cfg.get("flux_cm2_min"), "flux_cm2_min")
    eff_range = physics_cfg.get("efficiencies")
    if eff_range is None:
        raise ValueError("efficiencies must be set in config_step_0_physics.yaml.")
    eff_base = _sample_range(rng, eff_range, "efficiencies")
    if efficiencies_identical:
        effs = [eff_base] * 4
    else:
        if not isinstance(eff_range, list) or len(eff_range) != 2:
            raise ValueError("efficiencies must be a 2-value list [min, max] when not identical.")
        effs = [float(rng.uniform(float(eff_range[0]), float(eff_range[1]))) for _ in range(4)]

    if mesh_path.exists():
        mesh = pd.read_csv(mesh_path)
        if "done" not in mesh.columns:
            mesh["done"] = 0
    else:
        mesh = pd.DataFrame()
        mesh["done"] = []

    if z_positions.empty:
        raise ValueError("No z positions found in station configs; cannot select z positions.")

    for col in ("z_p1", "z_p2", "z_p3", "z_p4"):
        if col not in mesh.columns:
            mesh[col] = np.nan

    if not mesh.empty:
        missing_mask = mesh[["z_p1", "z_p2", "z_p3", "z_p4"]].isna().any(axis=1)
        if missing_mask.any():
            for idx in mesh.index[missing_mask]:
                geom_row = z_positions.sample(
                    n=1, random_state=rng.integers(0, 2**32 - 1)
                ).iloc[0]
                mesh.at[idx, "z_p1"] = float(geom_row["P1"])
                mesh.at[idx, "z_p2"] = float(geom_row["P2"])
                mesh.at[idx, "z_p3"] = float(geom_row["P3"])
                mesh.at[idx, "z_p4"] = float(geom_row["P4"])

    geom_row = z_positions.sample(n=1, random_state=rng.integers(0, 2**32 - 1)).iloc[0]

    new_row = {
        "done": 0,
        "cos_n": float(cos_n),
        "flux_cm2_min": float(flux_cm2_min),
        "eff_p1": float(effs[0]),
        "eff_p2": float(effs[1]),
        "eff_p3": float(effs[2]),
        "eff_p4": float(effs[3]),
        "z_p1": float(geom_row["P1"]),
        "z_p2": float(geom_row["P2"]),
        "z_p3": float(geom_row["P3"]),
        "z_p4": float(geom_row["P4"]),
    }
    mesh = pd.concat([mesh, pd.DataFrame([new_row])], ignore_index=True)
    if "param_set_id" in mesh.columns:
        mesh = mesh.sort_values("param_set_id").reset_index(drop=True)
    z_cols = ["z_p1", "z_p2", "z_p3", "z_p4"]
    head_cols = ["done", "param_set_id", "param_date"]
    ordered_cols = [c for c in head_cols if c in mesh.columns] + [
        c for c in mesh.columns if c not in head_cols and c not in z_cols
    ] + [c for c in z_cols if c in mesh.columns]
    mesh = mesh[ordered_cols]
    mesh.to_csv(mesh_path, index=False)

    meta = {
        "updated_at": now_iso(),
        "row_count": int(len(mesh)),
        "efficiencies_identical": efficiencies_identical,
        "step": "STEP_0",
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="STEP_0: append one parameter row and update mesh.")
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

    z_positions = _collect_z_positions(station_files)

    rng = np.random.default_rng(cfg.get("seed"))
    mesh_path = output_dir / "param_mesh.csv"
    mesh_meta_path = output_dir / "param_mesh_metadata.json"
    _append_param_row(mesh_path, mesh_meta_path, physics_cfg, rng, z_positions)

    meta = {
        "created_at": now_iso(),
        "step": "STEP_0",
        "config": physics_cfg,
        "runtime_config": runtime_cfg,
        "station_config_root": str(station_root),
    }
    mesh_meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Updated param mesh in {output_dir}")


if __name__ == "__main__":
    main()
