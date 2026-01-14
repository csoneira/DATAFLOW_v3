#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from STEP_SHARED.sim_utils import ensure_dir, latest_sim_run, now_iso, resolve_sim_run, reset_dir


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def select_station_conf(geom_map: pd.DataFrame, geometry_id: int, rng: np.random.Generator) -> dict:
    subset = geom_map[geom_map["geometry_id"] == geometry_id]
    if subset.empty:
        raise ValueError(f"geometry_id {geometry_id} not found in geometry_map_all.")
    row = subset.sample(n=1, random_state=rng.integers(0, 2**32 - 1)).iloc[0]
    return row.to_dict()


def random_start_datetime(start_date: str, end_date: str, rng: np.random.Generator) -> datetime:
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date) + timedelta(hours=23, minutes=59, seconds=59)
    if end_dt <= start_dt:
        return start_dt
    span_seconds = (end_dt - start_dt).total_seconds()
    offset = rng.uniform(0.0, span_seconds)
    return start_dt + timedelta(seconds=offset)


def build_filename(station_id: int, timestamp: datetime) -> str:
    day_of_year = timestamp.timetuple().tm_yday
    return f"mi0{station_id}{timestamp.year % 100:02d}{day_of_year:03d}{timestamp:%H%M%S}.dat"


def write_with_timestamps(
    input_path: Path,
    output_path: Path,
    start_time: datetime,
    rate_hz: float,
    rng: np.random.Generator,
) -> None:
    current_time = start_time
    with input_path.open("r", encoding="ascii") as src, output_path.open("w", encoding="ascii") as dst:
        for line in src:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            year = current_time.year
            month = current_time.month
            day = current_time.day
            hour = current_time.hour
            minute = current_time.minute
            second = current_time.second
            header = [
                f"{year:04d}",
                f"{month:02d}",
                f"{day:02d}",
                f"{hour:02d}",
                f"{minute:02d}",
                f"{second:02d}",
                "1",
            ]
            dst.write(" ".join(header + parts[7:]) + "\n")
            if rate_hz > 0:
                delta = rng.exponential(1.0 / rate_hz)
                current_time = current_time + timedelta(seconds=float(delta))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 12: assign station/conf and timestamps, emit station .dat.")
    parser.add_argument("--config", default="config_step_12.yaml", help="Path to step config YAML")
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

    geometry_id = int(cfg["geometry_id"])
    rate_hz = float(cfg.get("rate_hz", 1.0))

    input_sim_run = cfg.get("input_sim_run", "latest")
    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)

    input_run_dir = input_dir / str(input_sim_run)
    input_glob = cfg.get("input_glob", "*.dat")
    input_paths = sorted(input_run_dir.glob(input_glob))
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input .dat in {input_run_dir}, found {len(input_paths)}.")
    input_path = input_paths[0]

    map_dir = Path(cfg["geometry_map_dir"])
    if not map_dir.is_absolute():
        map_dir = Path(__file__).resolve().parent / map_dir
    map_sim_run = cfg.get("geometry_map_sim_run", "latest")
    if map_sim_run == "latest":
        map_sim_run = latest_sim_run(map_dir)
    geom_map_path = map_dir / str(map_sim_run) / "geometry_map_all.csv"
    geom_map = pd.read_csv(geom_map_path)

    rng = np.random.default_rng(cfg.get("seed"))
    selection = select_station_conf(geom_map, geometry_id, rng)
    start_date = selection.get("start")
    end_date = selection.get("end")
    if not start_date or not end_date:
        raise ValueError("start/end date columns are required in geometry_map_all.csv.")

    start_time = random_start_datetime(str(start_date), str(end_date), rng)
    station_id = int(selection.get("station", 0))

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_12", config_path, cfg, {"source_dataset": str(input_path)}
    )
    reset_dir(sim_run_dir)

    out_name = build_filename(station_id, start_time)
    out_path = sim_run_dir / out_name
    write_with_timestamps(input_path, out_path, start_time, rate_hz, rng)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_12",
        "config": cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "geometry_map": str(geom_map_path),
        "station_selection": selection,
        "start_time": start_time.isoformat(),
        "rate_hz": rate_hz,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
