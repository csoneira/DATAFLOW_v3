#!/usr/bin/env python3
"""STEP_0: append one parameter row with sampled z positions.

Inputs: station configuration CSVs in ONLINE_RUN_DICTIONARY.
Outputs: param_mesh.csv in STEP_0_TO_1.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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


def _assign_step_ids(mesh: pd.DataFrame) -> pd.DataFrame:
    mesh = mesh.copy()
    def _ensure_column(name: str) -> None:
        if name not in mesh.columns:
            mesh[name] = pd.NA
        mesh[name] = mesh[name].astype("string")

    def _parse_step_id(value: object) -> int | None:
        if value is None or pd.isna(value):
            return None
        try:
            return int(str(value))
        except ValueError:
            try:
                return int(float(str(value)))
            except ValueError:
                return None

    def _normalize_step_id_column(col: str) -> None:
        def _format(value: object) -> object:
            parsed = _parse_step_id(value)
            if parsed is None:
                return pd.NA
            return f"{parsed:03d}"
        mesh[col] = mesh[col].apply(_format).astype("string")

    for idx in range(1, 11):
        _ensure_column(f"step_{idx}_id")

    def assign_ids(cols: list[str], id_col: str) -> None:
        existing = {}
        if mesh[id_col].notna().any():
            for _, row in mesh[mesh[id_col].notna()].iterrows():
                key = tuple(row[col] for col in cols)
                parsed = _parse_step_id(row[id_col])
                if parsed is not None:
                    existing.setdefault(key, parsed)
        next_id = max(existing.values(), default=0) + 1
        for i, row in mesh.iterrows():
            if pd.notna(row[id_col]):
                continue
            key = tuple(row[col] for col in cols)
            if key not in existing:
                existing[key] = next_id
                next_id += 1
            mesh.at[i, id_col] = f"{existing[key]:03d}"
        _normalize_step_id_column(id_col)

    assign_ids(["cos_n", "flux_cm2_min"], "step_1_id")
    assign_ids(["z_p1", "z_p2", "z_p3", "z_p4"], "step_2_id")
    assign_ids(["eff_p1", "eff_p2", "eff_p3", "eff_p4"], "step_3_id")
    for idx in range(4, 11):
        col = f"step_{idx}_id"
        mesh[col] = mesh[col].fillna("001").astype("string")
        _normalize_step_id_column(col)
    return mesh


def _sample_range(rng: np.random.Generator, value: object, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        return float(rng.uniform(lo, hi))
    raise ValueError(f"{name} must be a number or a 2-value list [min, max].")


def _sample_efficiencies(
    rng: np.random.Generator,
    eff_range: object,
    efficiencies_identical: bool,
) -> list[float]:
    eff_base = _sample_range(rng, eff_range, "efficiencies")
    if efficiencies_identical:
        return [eff_base] * 4
    if not isinstance(eff_range, list) or len(eff_range) != 2:
        raise ValueError("efficiencies must be a 2-value list [min, max] when not identical.")
    min_val, max_val = float(eff_range[0]), float(eff_range[1])
    # Keep per-plane efficiencies within +/-0.05 of a base value (max spread 0.10).
    lo = max(min_val, eff_base - 0.05)
    hi = min(max_val, eff_base + 0.05)
    return [float(rng.uniform(lo, hi)) for _ in range(4)]


def _collect_z_positions(station_files: dict[int, Path]) -> pd.DataFrame:
    station_dfs = [read_station_config(path) for path in station_files.values()]
    geom_cols = ["P1", "P2", "P3", "P4"]
    all_geoms = pd.concat([df[geom_cols] for df in station_dfs], ignore_index=True)
    unique_geoms = all_geoms.dropna().drop_duplicates().reset_index(drop=True)
    non_zero = (unique_geoms[geom_cols] != 0).any(axis=1)
    unique_geoms = unique_geoms[non_zero].reset_index(drop=True)
    return unique_geoms


def _expected_counts_from_mesh(mesh: pd.DataFrame, include_done: bool) -> dict[str, int]:
    if "done" in mesh.columns and not include_done:
        mesh = mesh[mesh["done"].fillna(0).astype(int) != 1]
    step1 = mesh["step_1_id"].astype(str).nunique() if "step_1_id" in mesh.columns else 0
    step2 = mesh["step_2_id"].astype(str).nunique() if "step_2_id" in mesh.columns else 0
    step3 = mesh["step_3_id"].astype(str).nunique() if "step_3_id" in mesh.columns else 0
    counts = {
        "STEP_1_TO_2": step1,
        "STEP_2_TO_3": step1 * step2,
        "STEP_3_TO_4": step1 * step2 * step3,
    }
    total = counts["STEP_3_TO_4"]
    for step in range(4, 11):
        counts[f"STEP_{step}_TO_{step + 1}"] = total
    return counts


def _total_generation_pct(intersteps_dir: Path, mesh: pd.DataFrame) -> tuple[float, int, int]:
    expected = _expected_counts_from_mesh(mesh, include_done=True)
    step_dirs = sorted(intersteps_dir.glob("STEP_*_TO_*"))
    total_dirs = 0
    total_expected_dirs = 0
    for step_dir in step_dirs:
        step_name = step_dir.name
        if "0_TO_1" in step_name or "10_TO_FINAL" in step_name:
            continue
        total_dirs += len(list(step_dir.glob("SIM_RUN_*")))
        total_expected_dirs += expected.get(step_name, 0)
    if total_expected_dirs == 0:
        total_pct = 0.0 if not mesh.empty else 100.0
    else:
        total_pct = total_dirs / total_expected_dirs * 100.0
    return total_pct, total_dirs, total_expected_dirs


def _append_param_row(
    mesh_path: Path,
    meta_path: Path,
    physics_cfg: dict,
    rng: np.random.Generator,
    z_positions: pd.DataFrame,
) -> None:
    efficiencies_identical = bool(physics_cfg.get("efficiencies_identical", False))
    eff_range = physics_cfg.get("efficiencies")
    if eff_range is None:
        raise ValueError("efficiencies must be set in config_step_0_physics.yaml.")
    repeat_samples = int(physics_cfg.get("repeat_samples", 1))
    if repeat_samples < 1:
        raise ValueError("repeat_samples must be >= 1.")
    shared_columns = physics_cfg.get("shared_columns", [])
    if isinstance(shared_columns, str):
        shared_columns = [shared_columns]
    shared = {str(col) for col in shared_columns or []}
    if "efficiencies" in shared:
        shared.update({"eff_p1", "eff_p2", "eff_p3", "eff_p4"})
    if "z_positions" in shared:
        shared.update({"z_p1", "z_p2", "z_p3", "z_p4"})

    if mesh_path.exists():
        mesh = pd.read_csv(mesh_path)
        if "done" not in mesh.columns:
            mesh["done"] = 0
        dup_cols = [col for col in mesh.columns if col.endswith(".1")]
        for col in dup_cols:
            base = col[:-2]
            if base in mesh.columns:
                mesh = mesh.drop(columns=[col])
        if mesh.columns.duplicated().any():
            mesh = mesh.loc[:, ~mesh.columns.duplicated()]
    else:
        mesh = pd.DataFrame()
        mesh["done"] = []
    mesh["done"] = mesh["done"].fillna(0).astype(int)

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

    shared_values: dict[str, float] = {}
    expand_z_positions = bool(physics_cfg.get("expand_z_positions", False))
    shared_geom_row = None
    if not expand_z_positions and {"z_p1", "z_p2", "z_p3", "z_p4"} & shared:
        shared_geom_row = z_positions.sample(
            n=1, random_state=rng.integers(0, 2**32 - 1)
        ).iloc[0]
        shared_values.update(
            {
                "z_p1": float(shared_geom_row["P1"]),
                "z_p2": float(shared_geom_row["P2"]),
                "z_p3": float(shared_geom_row["P3"]),
                "z_p4": float(shared_geom_row["P4"]),
            }
        )
    if "cos_n" in shared:
        shared_values["cos_n"] = _sample_range(rng, physics_cfg.get("cos_n"), "cos_n")
    if "flux_cm2_min" in shared:
        shared_values["flux_cm2_min"] = _sample_range(
            rng, physics_cfg.get("flux_cm2_min"), "flux_cm2_min"
        )
    if {"eff_p1", "eff_p2", "eff_p3", "eff_p4"} & shared:
        effs_shared = _sample_efficiencies(rng, eff_range, efficiencies_identical)
        shared_values.update(
            {
                "eff_p1": float(effs_shared[0]),
                "eff_p2": float(effs_shared[1]),
                "eff_p3": float(effs_shared[2]),
                "eff_p4": float(effs_shared[3]),
            }
        )

    new_rows = []
    for _ in range(repeat_samples):
        cos_n = shared_values.get("cos_n", _sample_range(rng, physics_cfg.get("cos_n"), "cos_n"))
        flux_cm2_min = shared_values.get(
            "flux_cm2_min", _sample_range(rng, physics_cfg.get("flux_cm2_min"), "flux_cm2_min")
        )
        effs = _sample_efficiencies(rng, eff_range, efficiencies_identical)
        for key in ("eff_p1", "eff_p2", "eff_p3", "eff_p4"):
            if key in shared_values:
                idx = int(key.split("_p")[-1]) - 1
                effs[idx] = float(shared_values[key])
        if shared_geom_row is not None:
            geom_rows = [shared_geom_row]
        elif expand_z_positions:
            geom_rows = [row for _, row in z_positions.iterrows()]
        else:
            geom_rows = [
                z_positions.sample(n=1, random_state=rng.integers(0, 2**32 - 1)).iloc[0]
            ]
        for geom_row in geom_rows:
            new_rows.append(
                {
                "done": 0,
                    "cos_n": float(cos_n),
                    "flux_cm2_min": float(flux_cm2_min),
                    "z_p1": float(geom_row["P1"]),
                    "z_p2": float(geom_row["P2"]),
                    "z_p3": float(geom_row["P3"]),
                    "z_p4": float(geom_row["P4"]),
                    "eff_p1": float(effs[0]),
                    "eff_p2": float(effs[1]),
                    "eff_p3": float(effs[2]),
                    "eff_p4": float(effs[3]),
                }
            )
    mesh = pd.concat([mesh, pd.DataFrame(new_rows)], ignore_index=True)
    mesh = _assign_step_ids(mesh)
    if "param_set_id" in mesh.columns:
        mesh = mesh.sort_values("param_set_id").reset_index(drop=True)
    z_cols = ["z_p1", "z_p2", "z_p3", "z_p4"]
    head_cols = ["done", "param_set_id", "param_date"]
    step_id_cols = [f"step_{idx}_id" for idx in range(1, 11)]
    front_cols = ["cos_n", "flux_cm2_min"] + z_cols
    ordered_cols = (
        [c for c in head_cols if c in mesh.columns]
        + [c for c in step_id_cols if c in mesh.columns]
        + [c for c in front_cols if c in mesh.columns]
        + [
            c
            for c in mesh.columns
            if c not in head_cols and c not in front_cols and c not in step_id_cols
        ]
    )
    mesh = mesh[ordered_cols]
    mesh["done"] = mesh["done"].fillna(0).astype(int)
    mesh.to_csv(mesh_path, index=False)

    meta = {
        "updated_at": now_iso(),
        "row_count": int(len(mesh)),
        "efficiencies_identical": efficiencies_identical,
        "repeat_samples": repeat_samples,
        "expand_z_positions": expand_z_positions,
        "shared_columns": sorted(shared),
        "step": "STEP_0",
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def _log_info(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [STEP_0] {message}")


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Append even if total completion is below 100%.",
    )
    args = parser.parse_args()

    _log_info("STEP_0 setup started")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    runtime_path = Path(args.runtime_config) if args.runtime_config else None
    if runtime_path is not None and not runtime_path.is_absolute():
        runtime_path = Path(__file__).resolve().parent / runtime_path

    physics_cfg, runtime_cfg, cfg, runtime_path = load_step_configs(config_path, runtime_path)

    _log_info(
        "Loaded configs: "
        f"physics={config_path}, "
        f"runtime={runtime_path if runtime_path else 'auto'}"
    )

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
    _log_info(f"Output directory: {output_dir}")

    station_files = list_station_config_files(station_root)
    if not station_files:
        raise FileNotFoundError(f"No station config CSVs found under {station_root}")
    _log_info(f"Station configs found: {len(station_files)}")

    z_positions = _collect_z_positions(station_files)
    _log_info(f"Unique z-position rows: {len(z_positions)}")

    rng = np.random.default_rng(cfg.get("seed"))
    mesh_path = output_dir / "param_mesh.csv"
    mesh_meta_path = output_dir / "param_mesh_metadata.json"

    if mesh_path.exists() and not args.force:
        mesh = pd.read_csv(mesh_path)
        if "done" not in mesh.columns:
            mesh["done"] = 0
        done_series = mesh["done"].fillna(0).astype(int)
        total_rows = len(mesh)
        done_rows = int((done_series == 1).sum())
        if total_rows > 0 and done_rows < total_rows:
            done_pct = done_rows / total_rows * 100.0
            print(
                "Skipping append: mesh completion "
                f"{done_pct:.1f}% ({done_rows}/{total_rows} rows done) is below 100%."
            )
            _log_info("Skip append: mesh not fully done")
            return

    _append_param_row(mesh_path, mesh_meta_path, physics_cfg, rng, z_positions)

    try:
        mesh = pd.read_csv(mesh_path)
        _log_info(f"Mesh rows after append: {len(mesh)}")
    except (OSError, pd.errors.ParserError):
        _log_info("Mesh updated (row count unavailable)")

    meta = {
        "created_at": now_iso(),
        "step": "STEP_0",
        "config": physics_cfg,
        "runtime_config": runtime_cfg,
        "station_config_root": str(station_root),
    }
    mesh_meta_path.write_text(json.dumps(meta, indent=2))

    _log_info(f"Updated param mesh in {output_dir}")


if __name__ == "__main__":
    main()
