#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from STEP_SHARED.sim_utils import param_mesh_lock, write_csv_atomic, write_text_atomic


MESH_NUMERIC_COLS = [
    "cos_n",
    "flux_cm2_min",
    "eff_p1",
    "eff_p2",
    "eff_p3",
    "eff_p4",
    "z_p1",
    "z_p2",
    "z_p3",
    "z_p4",
]


STEP_ID_SPECS = [
    ("step_1_id", ["cos_n", "flux_cm2_min"]),
    ("step_2_id", ["z_p1", "z_p2", "z_p3", "z_p4"]),
    ("step_3_id", ["eff_p1", "eff_p2", "eff_p3", "eff_p4"]),
]


def _parse_efficiencies(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        effs = list(value)
    else:
        effs = ast.literal_eval(str(value))
    if not isinstance(effs, list) or len(effs) != 4:
        raise ValueError(f"efficiencies must be a 4-value list, got: {value!r}")
    return [float(x) for x in effs]


def _normalize_step_id(value: object) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _build_param_key(row: pd.Series, cols: list[str]) -> tuple[float, ...]:
    return tuple(round(float(row[col]), 12) for col in cols)


def _assign_step_ids(mesh: pd.DataFrame, existing_mesh: pd.DataFrame | None = None) -> pd.DataFrame:
    mesh = mesh.copy()
    existing = existing_mesh.copy() if existing_mesh is not None else pd.DataFrame()
    if not existing.empty:
        for col in MESH_NUMERIC_COLS:
            if col in existing.columns:
                existing[col] = pd.to_numeric(existing[col], errors="coerce")
        existing = existing.dropna(subset=[col for col in MESH_NUMERIC_COLS if col in existing.columns])

    def assign_ids(cols: list[str], id_col: str) -> None:
        mapping: dict[tuple[float, ...], int] = {}
        used_ids: set[int] = set()

        if not existing.empty and id_col in existing.columns and all(col in existing.columns for col in cols):
            for _, row in existing.iterrows():
                step_id = _normalize_step_id(row.get(id_col))
                if step_id is None:
                    continue
                key = _build_param_key(row, cols)
                if key not in mapping or step_id < mapping[key]:
                    mapping[key] = step_id
                used_ids.add(step_id)

        next_id = max(used_ids, default=0) + 1
        ids: list[str] = []
        for _, row in mesh.iterrows():
            key = _build_param_key(row, cols)
            if key not in mapping:
                mapping[key] = next_id
                used_ids.add(next_id)
                next_id += 1
            ids.append(f"{mapping[key]:03d}")
        mesh[id_col] = pd.Series(ids, dtype="string")

    for id_col, cols in STEP_ID_SPECS:
        assign_ids(cols, id_col)
    for idx in range(4, 11):
        mesh[f"step_{idx}_id"] = "001"
    return mesh


def _build_mesh(sim_params: pd.DataFrame) -> pd.DataFrame:
    required = [
        "cos_n",
        "flux_cm2_min",
        "z_plane_1",
        "z_plane_2",
        "z_plane_3",
        "z_plane_4",
        "efficiencies",
    ]
    missing = [col for col in required if col not in sim_params.columns]
    if missing:
        raise ValueError(f"Missing columns in step_final_simulation_params.csv: {missing}")

    sim_params = sim_params.copy()
    param_cols = [
        "cos_n",
        "flux_cm2_min",
        "z_plane_1",
        "z_plane_2",
        "z_plane_3",
        "z_plane_4",
        "efficiencies",
    ]
    if sim_params.duplicated(subset=param_cols).any():
        sim_params = sim_params.drop_duplicates(subset=param_cols, keep="first")

    efficiencies = sim_params["efficiencies"].apply(_parse_efficiencies)
    eff_cols = pd.DataFrame(
        efficiencies.tolist(),
        columns=["eff_p1", "eff_p2", "eff_p3", "eff_p4"],
        index=sim_params.index,
    )

    mesh = pd.DataFrame(
        {
            "done": 1,
            "cos_n": pd.to_numeric(sim_params["cos_n"], errors="coerce"),
            "flux_cm2_min": pd.to_numeric(sim_params["flux_cm2_min"], errors="coerce"),
            "z_p1": pd.to_numeric(sim_params["z_plane_1"], errors="coerce"),
            "z_p2": pd.to_numeric(sim_params["z_plane_2"], errors="coerce"),
            "z_p3": pd.to_numeric(sim_params["z_plane_3"], errors="coerce"),
            "z_p4": pd.to_numeric(sim_params["z_plane_4"], errors="coerce"),
        }
    )
    if mesh.isna().any().any():
        raise ValueError("One or more required numeric values are missing or invalid.")
    mesh = pd.concat([mesh, eff_cols], axis=1)

    mesh = mesh.sort_values(["cos_n", "flux_cm2_min", "z_p1", "z_p2", "z_p3", "z_p4"]).reset_index(drop=True)
    return mesh


def _write_mesh(mesh: pd.DataFrame, mesh_path: Path) -> None:
    step_id_cols = [f"step_{idx}_id" for idx in range(1, 11)]
    ordered_cols = ["done"] + step_id_cols + [
        "cos_n",
        "flux_cm2_min",
        "eff_p1",
        "eff_p2",
        "eff_p3",
        "eff_p4",
        "z_p1",
        "z_p2",
        "z_p3",
        "z_p4",
    ]
    mesh = mesh[ordered_cols]
    mesh["done"] = mesh["done"].astype(int)
    write_csv_atomic(mesh, mesh_path, index=False)


def _load_pending_rows(mesh_path: Path) -> pd.DataFrame:
    if not mesh_path.exists():
        return pd.DataFrame(columns=["done"] + MESH_NUMERIC_COLS)

    try:
        existing = pd.read_csv(mesh_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["done"] + MESH_NUMERIC_COLS)

    if "done" not in existing.columns:
        return pd.DataFrame(columns=["done"] + MESH_NUMERIC_COLS)

    pending = existing[existing["done"].fillna(0).astype(int) != 1].copy()
    if pending.empty:
        return pd.DataFrame(columns=["done"] + MESH_NUMERIC_COLS)

    missing_cols = [col for col in MESH_NUMERIC_COLS if col not in pending.columns]
    if missing_cols:
        raise ValueError(f"Existing pending rows are missing required columns: {missing_cols}")

    pending["done"] = 0
    for col in MESH_NUMERIC_COLS:
        pending[col] = pd.to_numeric(pending[col], errors="coerce")
    if pending[MESH_NUMERIC_COLS].isna().any().any():
        raise ValueError("Existing pending rows contain invalid numeric values.")

    return pending[["done"] + MESH_NUMERIC_COLS].reset_index(drop=True)


def _load_existing_mesh(mesh_path: Path) -> pd.DataFrame:
    if not mesh_path.exists():
        return pd.DataFrame(columns=["done"] + MESH_NUMERIC_COLS)

    try:
        existing = pd.read_csv(mesh_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["done"] + MESH_NUMERIC_COLS)

    missing_cols = [col for col in MESH_NUMERIC_COLS if col not in existing.columns]
    if missing_cols:
        return pd.DataFrame(columns=["done"] + MESH_NUMERIC_COLS)

    for col in MESH_NUMERIC_COLS:
        existing[col] = pd.to_numeric(existing[col], errors="coerce")
    existing = existing.dropna(subset=MESH_NUMERIC_COLS).reset_index(drop=True)
    return existing


def _cleanup_sim_runs(intersteps_dir: Path, apply: bool) -> None:
    for step_dir in sorted(intersteps_dir.glob("STEP_*_TO_*")):
        if step_dir.name == "STEP_0_TO_1":
            continue
        for sim_run_dir in sorted(step_dir.glob("SIM_RUN_*")):
            if apply:
                shutil.rmtree(sim_run_dir)
                print(f"DELETED: {sim_run_dir}")
            else:
                print(f"DRY-RUN DELETE: {sim_run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild param_mesh.csv from step_final_simulation_params.csv and "
            "optionally clean intermediate SIM_RUN directories."
        )
    )
    parser.add_argument(
        "--sim-params",
        default="~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA/step_final_simulation_params.csv",
        help="Path to step_final_simulation_params.csv",
    )
    parser.add_argument(
        "--param-mesh",
        default="~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1/param_mesh.csv",
        help="Path to param_mesh.csv",
    )
    parser.add_argument(
        "--intersteps-dir",
        default="~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS",
        help="Base INTERSTEPS directory",
    )
    parser.add_argument(
        "--skip-delete",
        action="store_true",
        help="Skip deleting SIM_RUN directories (kept for backwards compatibility).",
    )
    parser.add_argument(
        "--delete-sim-runs",
        action="store_true",
        help="Enable cleanup of intermediate SIM_RUN directories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply destructive cleanup when used with --delete-sim-runs (otherwise cleanup is dry-run).",
    )
    args = parser.parse_args()

    sim_params_path = Path(args.sim_params).expanduser().resolve()
    mesh_path = Path(args.param_mesh).expanduser().resolve()
    intersteps_dir = Path(args.intersteps_dir).expanduser().resolve()

    if not sim_params_path.exists():
        raise FileNotFoundError(f"step_final_simulation_params.csv not found: {sim_params_path}")
    if not intersteps_dir.exists():
        raise FileNotFoundError(f"INTERSTEPS dir not found: {intersteps_dir}")

    sim_params = pd.read_csv(sim_params_path)
    mesh_done = _build_mesh(sim_params)
    with param_mesh_lock(mesh_path):
        existing_mesh = _load_existing_mesh(mesh_path)
        pending_rows = _load_pending_rows(mesh_path)
        if not pending_rows.empty:
            print(f"Keeping {len(pending_rows)} pending rows (done=0) from existing param mesh.")
            mesh = pd.concat([mesh_done, pending_rows], ignore_index=True)
            sort_cols = [
                "cos_n",
                "flux_cm2_min",
                "z_p1",
                "z_p2",
                "z_p3",
                "z_p4",
                "eff_p1",
                "eff_p2",
                "eff_p3",
                "eff_p4",
                "done",
            ]
            mesh = mesh.sort_values(sort_cols, ascending=[True] * 10 + [False]).reset_index(drop=True)
        else:
            mesh = mesh_done
        mesh = _assign_step_ids(mesh, existing_mesh=existing_mesh)
        _write_mesh(mesh, mesh_path)

        meta_path = mesh_path.with_name("param_mesh_metadata.json")
        meta = {
            "source": str(sim_params_path),
            "row_count": int(len(mesh)),
            "pending_rows_kept": int(len(pending_rows)),
            "note": (
                "Rebuilt from step_final_simulation_params.csv while preserving existing pending rows "
                "(done=0). param_set_id/param_date excluded."
            ),
        }
        write_text_atomic(meta_path, json.dumps(meta, indent=2))

    cleanup_enabled = bool(args.delete_sim_runs) and not bool(args.skip_delete)
    if cleanup_enabled:
        _cleanup_sim_runs(intersteps_dir, args.apply)
    else:
        print("SIM_RUN cleanup skipped (pass --delete-sim-runs to enable).")

    print(f"Wrote param mesh to {mesh_path}")


if __name__ == "__main__":
    main()
