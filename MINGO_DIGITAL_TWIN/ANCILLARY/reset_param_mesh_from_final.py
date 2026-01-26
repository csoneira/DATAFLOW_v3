#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path

import pandas as pd


def _parse_efficiencies(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        effs = list(value)
    else:
        effs = ast.literal_eval(str(value))
    if not isinstance(effs, list) or len(effs) != 4:
        raise ValueError(f"efficiencies must be a 4-value list, got: {value!r}")
    return [float(x) for x in effs]


def _assign_step_ids(mesh: pd.DataFrame) -> pd.DataFrame:
    mesh = mesh.copy()

    def assign_ids(cols: list[str], id_col: str) -> None:
        mapping: dict[tuple[object, ...], int] = {}
        next_id = 1
        ids: list[str] = []
        for _, row in mesh.iterrows():
            key = tuple(row[col] for col in cols)
            if key not in mapping:
                mapping[key] = next_id
                next_id += 1
            ids.append(f"{mapping[key]:03d}")
        mesh[id_col] = pd.Series(ids, dtype="string")

    assign_ids(["cos_n", "flux_cm2_min"], "step_1_id")
    assign_ids(["z_p1", "z_p2", "z_p3", "z_p4"], "step_2_id")
    assign_ids(["eff_p1", "eff_p2", "eff_p3", "eff_p4"], "step_3_id")
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
    mesh = _assign_step_ids(mesh)
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
    mesh.to_csv(mesh_path, index=False)


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
            "delete intermediate SIM_RUN directories."
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
        help="Skip deleting SIM_RUN directories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete directories (default is dry-run).",
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
    mesh = _build_mesh(sim_params)
    _write_mesh(mesh, mesh_path)

    meta_path = mesh_path.with_name("param_mesh_metadata.json")
    meta = {
        "source": str(sim_params_path),
        "row_count": int(len(mesh)),
        "note": "Rebuilt from step_final_simulation_params.csv (param_set_id/param_date excluded).",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    if not args.skip_delete:
        _cleanup_sim_runs(intersteps_dir, args.apply)

    print(f"Wrote param mesh to {mesh_path}")


if __name__ == "__main__":
    main()
