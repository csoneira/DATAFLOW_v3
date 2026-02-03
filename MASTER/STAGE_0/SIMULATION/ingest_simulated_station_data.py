#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 simulation: move .dat files into station buffers.")
    parser.add_argument(
        "--source-dir",
        default="~/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA",
        help="Directory containing simulated .dat files.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    station_root = repo_root / "STATIONS" / "MINGO00"
    stage0_dir = station_root / "STAGE_0"
    stage01_dir = station_root / "STAGE_0_to_1"
    stage0_dir.mkdir -p(parents=True, exist_ok=True)
    stage01_dir.mkdir -p(parents=True, exist_ok=True)

    registry_path = stage0_dir / "imported_basenames.csv"
    imported = set()
    if registry_path.exists():
        with registry_path.open("r", encoding="ascii") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("basename"):
                    imported.add(row["basename"])

    sim_params_path = repo_root / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA" / "step_final_simulation_params.csv"
    sim_params = None
    if sim_params_path.exists():
        sim_params = pd.read_csv(sim_params_path)

    moved = 0
    new_registry_rows = []
    config_rows = []
    for dat_file in sorted(source_dir.glob("*.dat")):
        name = dat_file.name
        if not name.startswith("mi00"):
            continue
        basename = dat_file.stem
        if basename in imported:
            continue
        dest_path = stage01_dir / dat_file.name
        shutil.move(dat_file, dest_path)
        imported.add(basename)
        new_registry_rows.append({"basename": basename})
        moved += 1

        if sim_params is not None:
            row_match = sim_params[sim_params["file_name"] == name]
            if not row_match.empty:
                row = row_match.iloc[0]
                trig = []
                try:
                    trig = json.loads(str(row.get("trigger_combinations", "[]")))
                except json.JSONDecodeError:
                    trig = []
                config_rows.append(
                    {
                        "station": 0,
                        "conf": int(row.get("param_set_id", 0) or 0),
                        "start": str(row.get("param_date", "")),
                        "end": str(row.get("param_date", "")),
                        "over_P1": 0,
                        "P1-P2": 0,
                        "P2-P3": 0,
                        "P3-P4": 0,
                        "P1": float(row.get("z_plane_1", 0.0)),
                        "P2": float(row.get("z_plane_2", 0.0)),
                        "P3": float(row.get("z_plane_3", 0.0)),
                        "P4": float(row.get("z_plane_4", 0.0)),
                        "C1": 0,
                        "C2": 0,
                        "C3": 0,
                        "C4": 0,
                        "phi_north": 0,
                        "city": "SIMULATION",
                        "comment": "",
                    }
                )

    if new_registry_rows:
        write_header = not registry_path.exists()
        with registry_path.open("a", encoding="ascii", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["basename"])
            if write_header:
                writer.writeheader()
            writer.writerows(new_registry_rows)

    config_path = repo_root / "MASTER" / "CONFIG_FILES" / "ONLINE_RUN_DICTIONARY" / "STATION_0" / "input_file_mingo00.csv"
    if config_rows:
        config_path.parent.mkdir -p(parents=True, exist_ok=True)
        existing = set()
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            if len(lines) >= 2:
                df_existing = pd.read_csv(config_path, header=1, decimal=",", dtype=str)
                for _, row in df_existing.iterrows():
                    key = (row.get("start"), row.get("P1"), row.get("P2"), row.get("P3"), row.get("P4"))
                    existing.add(key)
        pending = []
        for row in config_rows:
            key = (row["start"], str(row["P1"]), str(row["P2"]), str(row["P3"]), str(row["P4"]))
            if key not in existing:
                pending.append(row)
        if pending:
            header_line = (
                "Detector,Conf number,Date,,Lead (mm),,,,Z distance (mm),,,,Trigger,,,,Azimuth to North (º),"
                "Location,Comments"
            )
            columns_line = (
                "station,conf,start,end,over_P1,P1-P2,P2-P3,P3-P4,P1,P2,P3,P4,C1,C2,C3,C4,phi_north,city,comment"
            )
            write_header = not config_path.exists()
            with config_path.open("a", encoding="utf-8", newline="") as handle:
                if write_header:
                    handle.write(header_line + "\n")
                    handle.write(columns_line + "\n")
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "station",
                        "conf",
                        "start",
                        "end",
                        "over_P1",
                        "P1-P2",
                        "P2-P3",
                        "P3-P4",
                        "P1",
                        "P2",
                        "P3",
                        "P4",
                        "C1",
                        "C2",
                        "C3",
                        "C4",
                        "phi_north",
                        "city",
                        "comment",
                    ],
                )
                writer.writerows(pending)

    stage0_config_path = stage0_dir / "input_file_mingo00.csv"
    if sim_params is not None and not sim_params.empty:
        stage0_rows = []
        for _, row in sim_params.iterrows():
            stage0_rows.append(
                {
                    "station": 0,
                    "conf": int(row.get("param_set_id", 0) or 0),
                    "start": str(row.get("param_date", "")),
                    "end": str(row.get("param_date", "")),
                    "over_P1": 0,
                    "P1-P2": 0,
                    "P2-P3": 0,
                    "P3-P4": 0,
                    "P1": float(row.get("z_plane_1", 0.0)),
                    "P2": float(row.get("z_plane_2", 0.0)),
                    "P3": float(row.get("z_plane_3", 0.0)),
                    "P4": float(row.get("z_plane_4", 0.0)),
                    "C1": 0,
                    "C2": 0,
                    "C3": 0,
                    "C4": 0,
                    "phi_north": 0,
                    "city": "SIMULATION",
                    "comment": "",
                }
            )
        df_stage0 = pd.DataFrame(stage0_rows)
        df_stage0 = df_stage0.drop_duplicates(subset=["start", "conf"], keep="last")
        df_stage0 = df_stage0.sort_values(["start", "conf"]).reset_index(drop=True)
        header_line = (
            "Detector,Conf number,Date,,Lead (mm),,,,Z distance (mm),,,,Trigger,,,,Azimuth to North (º),"
            "Location,Comments"
        )
        columns_line = (
            "station,conf,start,end,over_P1,P1-P2,P2-P3,P3-P4,P1,P2,P3,P4,C1,C2,C3,C4,phi_north,city,comment"
        )
        with stage0_config_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(header_line + "\n")
            handle.write(columns_line + "\n")
            df_stage0.to_csv(handle, index=False, header=False)
        print(f"Updated stage-0 config: {stage0_config_path}")
        config_path.parent.mkdir -p(parents=True, exist_ok=True)
        shutil.copyfile(stage0_config_path, config_path)
        print(f"Synced online config: {config_path}")

    print(f"Moved {moved} .dat files from {source_dir} into {stage01_dir}")


if __name__ == "__main__":
    main()
