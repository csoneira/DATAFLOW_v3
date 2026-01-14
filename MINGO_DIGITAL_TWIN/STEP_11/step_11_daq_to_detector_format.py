#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from STEP_SHARED.sim_utils import ensure_dir, latest_sim_run, load_with_metadata, now_iso, resolve_sim_run, reset_dir


def format_value(val: float) -> str:
    if not np.isfinite(val):
        val = 0.0
    if val < 0:
        return f"{val:.4f}"
    return f"{val:09.4f}"


def extract_timestamp(cfg: dict) -> tuple[int, int, int, int, int, int]:
    mode = str(cfg.get("date_mode", "now")).lower()
    if mode == "fixed":
        return (
            int(cfg["year"]),
            int(cfg["month"]),
            int(cfg["day"]),
            int(cfg["hour"]),
            int(cfg["minute"]),
            int(cfg["second"]),
        )
    now = datetime.now()
    return now.year, now.month, now.day, now.hour, now.minute, now.second


def build_output_name(cfg: dict) -> str:
    year, month, day, hour, minute, second = extract_timestamp(cfg)
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    return f"mi00{year % 100:02d}{day_of_year:03d}{hour:02d}{minute:02d}{second:02d}.dat"


def write_event_rows(df: pd.DataFrame, cfg: dict, output_path: Path) -> None:
    year, month, day, hour, minute, second = extract_timestamp(cfg)
    event_type = int(cfg.get("event_type", 1))

    plane_order = [4, 3, 2, 1]
    field_order = [
        ("T_front", "T_F"),
        ("T_back", "T_B"),
        ("Q_front", "Q_F"),
        ("Q_back", "Q_B"),
    ]
    strip_order = [1, 2, 3, 4]

    with output_path.open("w", encoding="ascii") as handle:
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            parts = [
                "0000",
                "00",
                "00",
                "00",
                "00",
                "00",
                str(event_type),
            ]

            for plane_idx in plane_order:
                for prefix, _ in field_order:
                    for strip_idx in strip_order:
                        col = f"{prefix}_{plane_idx}_s{strip_idx}"
                        val = float(row_dict.get(col, 0.0))
                        parts.append(format_value(val))

            handle.write(" ".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11: format DAQ output as detector text data.")
    parser.add_argument("--config", default="config_step_11.yaml", help="Path to step config YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    with config_path.open("r") as handle:
        cfg = yaml.safe_load(handle)

    input_dir = Path(cfg["input_dir"])
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parent / input_dir
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    input_glob = cfg.get("input_glob", "**/geom_*_daq.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is None:
        raise ValueError("config geometry_id is required for Step 11.")
    geometry_id = int(geometry_id)
    input_sim_run = cfg.get("input_sim_run", "latest")
    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)

    input_run_dir = input_dir / str(input_sim_run)
    if "**" in input_glob:
        input_paths = sorted(input_run_dir.rglob(input_glob.replace("**/", "")))
    else:
        input_paths = sorted(input_run_dir.glob(input_glob))
    geom_key = f"geom_{geometry_id}"
    input_paths = [p for p in input_paths if p.stem == f"{geom_key}_daq"]
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    print(f"Processing: {input_path}")
    df, upstream_meta = load_with_metadata(input_path)

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_11", config_path, cfg, upstream_meta
    )
    reset_dir(sim_run_dir)

    out_name = build_output_name(cfg)
    out_path = sim_run_dir / out_name
    write_event_rows(df, cfg, out_path)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_11",
        "config": cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
