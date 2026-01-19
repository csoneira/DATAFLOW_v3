#!/usr/bin/env python3
"""
STEP_FINAL: format DAQ data and emit station .dat files with assigned date ranges.

Inputs: geom_<G>_daq from Step 10.
Outputs: SIMULATED_DATA/mi0XYYDDDHHMMSS.dat + step_13_output_registry.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from STEP_SHARED.sim_utils import (
    ensure_dir,
    iter_input_frames,
    latest_sim_run,
    load_sim_run_registry,
    load_step_configs,
    load_with_metadata,
    now_iso,
    random_sim_run,
)


def format_value(val: float) -> str:
    if not np.isfinite(val):
        val = 0.0
    if val < 0:
        return f"{val:.4f}"
    return f"{val:09.4f}"


def format_time_s(val: float) -> str:
    if not np.isfinite(val):
        val = 0.0
    return f"{val:.6f}"


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def select_station_conf(geom_map: pd.DataFrame, geometry_id: int, rng: np.random.Generator) -> dict:
    subset = geom_map[geom_map["geometry_id"] == geometry_id].copy()
    if subset.empty:
        raise ValueError(f"geometry_id {geometry_id} not found in geometry_map_all.")
    if "start" in subset.columns and "end" in subset.columns:
        subset = subset[subset["start"].notna() & subset["end"].notna()]
        subset = subset[(subset["start"].astype(str) != "nan") & (subset["end"].astype(str) != "nan")]
    if subset.empty:
        raise ValueError(f"geometry_id {geometry_id} has no valid start/end dates in geometry_map_all.")
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
    station_id = station_id + 4
    day_of_year = timestamp.timetuple().tm_yday
    return f"mi0{station_id}{timestamp.year % 100:02d}{day_of_year:03d}{timestamp:%H%M%S}.dat"


def load_output_registry(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "files": []}


def save_output_registry(path: Path, registry: dict) -> None:
    path.write_text(json.dumps(registry, indent=2))


def collect_sim_run_configs(base_dir: Path) -> list[dict]:
    registry_path = base_dir / "sim_run_registry.json"
    if not registry_path.exists():
        return []
    registry = load_sim_run_registry(base_dir)
    return registry.get("runs", [])


def normalize_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".chunks.json"):
        name = name[: -len(".chunks.json")]
    stem = Path(name).stem
    return stem.replace(".chunks", "")


def load_input_meta(path: Path) -> dict:
    if path.name.endswith(".chunks.json"):
        manifest = json.loads(path.read_text())
        return manifest.get("metadata", {})
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    _, meta = load_with_metadata(path)
    return meta


def parse_geometry_id(meta: dict) -> int | None:
    if "geometry_id" in meta:
        try:
            return int(meta["geometry_id"])
        except (TypeError, ValueError):
            return None
    meta_cfg = meta.get("config", {})
    geom_val = meta_cfg.get("geometry_id")
    if geom_val is None or str(geom_val).lower() == "auto":
        return None
    try:
        return int(geom_val)
    except (TypeError, ValueError):
        return None


def list_paths(run_dir: Path, input_glob: str) -> list[Path]:
    if "**" in input_glob:
        return sorted(run_dir.rglob(input_glob.replace("**/", "")))
    return sorted(run_dir.glob(input_glob))


def select_input_paths(
    input_dir: Path,
    input_sim_run: str,
    geometry_id: int | None,
    input_glob: str,
    input_collect: str,
) -> tuple[list[Path], dict]:
    input_run_dir = input_dir / str(input_sim_run)
    baseline_paths = list_paths(input_run_dir, input_glob)
    if geometry_id is not None:
        geom_key = f"geom_{geometry_id}"
        baseline_paths = [
            p for p in baseline_paths if normalize_stem(p) == f"{geom_key}_daq"
        ]
        if not baseline_paths:
            fallback_path = input_run_dir / f"{geom_key}_daq.chunks.json"
            if fallback_path.exists():
                baseline_paths = [fallback_path]
    elif not baseline_paths:
        baseline_paths = sorted(input_run_dir.glob("geom_*_daq.chunks.json"))
    if not baseline_paths:
        raise FileNotFoundError(f"Expected at least 1 input in {input_run_dir}, found 0.")

    baseline_meta = load_input_meta(baseline_paths[0])
    if input_collect == "baseline_only":
        return baseline_paths, baseline_meta

    baseline_config_hash = baseline_meta.get("config_hash")
    baseline_upstream_hash = baseline_meta.get("upstream_hash")

    candidates = []
    for sim_run_dir in sorted(input_dir.glob("SIM_RUN_*")):
        candidates.extend(list_paths(sim_run_dir, input_glob))
    selected: list[Path] = []
    for path in candidates:
        meta = load_input_meta(path)
        if geometry_id is not None:
            meta_geom = parse_geometry_id(meta)
            if meta_geom != geometry_id:
                continue
        if input_collect == "matching":
            if meta.get("config_hash") != baseline_config_hash:
                continue
            if meta.get("upstream_hash") != baseline_upstream_hash:
                continue
        selected.append(path)

    if not selected:
        raise FileNotFoundError("No input files matched the selection criteria.")
    return selected, baseline_meta


def build_payload(row_dict: dict, include_thick: bool) -> str:
    plane_order = [4, 3, 2, 1]
    field_order = [
        ("T_front", "T_F"),
        ("T_back", "T_B"),
        ("Q_front", "Q_F"),
        ("Q_back", "Q_B"),
    ]
    strip_order = [1, 2, 3, 4]

    parts: list[str] = []
    for plane_idx in plane_order:
        for prefix, _ in field_order:
            for strip_idx in strip_order:
                col = f"{prefix}_{plane_idx}_s{strip_idx}"
                val = float(row_dict.get(col, 0.0))
                parts.append(format_value(val))
    if include_thick:
        parts.append(format_time_s(float(row_dict.get("T_thick_s", 0.0))))
    return " ".join(parts)


def sample_payloads(
    input_paths: list[Path],
    target_rows: int,
    chunk_rows: int | None,
    rng: np.random.Generator,
) -> tuple[list[str], int, list[float] | None]:
    reservoir: list[str] = []
    offsets: list[float] | None = None
    use_thick: bool | None = None
    seen = 0
    for path in input_paths:
        input_iter, _, _ = iter_input_frames(path, chunk_rows)
        for chunk in input_iter:
            has_thick = "T_thick_s" in chunk.columns
            if use_thick is None:
                use_thick = has_thick
            elif use_thick != has_thick:
                raise ValueError("Inconsistent T_thick_s presence across input chunks.")
            for row in chunk.itertuples(index=False):
                row_dict = row._asdict()
                payload = build_payload(row_dict, include_thick=bool(use_thick))
                thick_time = float(row_dict.get("T_thick_s", 0.0)) if use_thick else None
                seen += 1
                if len(reservoir) < target_rows:
                    reservoir.append(payload)
                    if use_thick:
                        if offsets is None:
                            offsets = []
                        offsets.append(thick_time or 0.0)
                    continue
                pick = int(rng.integers(0, seen))
                if pick < target_rows:
                    reservoir[pick] = payload
                    if use_thick:
                        if offsets is None:
                            offsets = [0.0] * target_rows
                        offsets[pick] = thick_time or 0.0
    return reservoir, seen, offsets


def write_with_timestamps(
    payloads: list[str],
    output_path: Path,
    start_time: datetime,
    rate_hz: float,
    rng: np.random.Generator,
    offsets_s: list[float] | None = None,
) -> None:
    with output_path.open("w", encoding="ascii") as dst:
        current_time = start_time
        for idx, payload in enumerate(payloads):
            if offsets_s is not None:
                current_time = start_time + timedelta(seconds=float(offsets_s[idx]))
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
            dst.write(" ".join(header + payload.split()) + "\n")
            if offsets_s is None and rate_hz > 0:
                delta = rng.exponential(1.0 / rate_hz)
                current_time = current_time + timedelta(seconds=float(delta))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STEP_FINAL: format DAQ output and emit station .dat files."
    )
    parser.add_argument(
        "--config",
        default="config_step_final_physics.yaml",
        help="Path to step physics config YAML",
    )
    parser.add_argument(
        "--runtime-config",
        default=None,
        help="Path to step runtime config YAML (defaults to *_runtime.yaml)",
    )
    parser.add_argument("--no-plots", action="store_true", help="No-op for consistency")
    parser.add_argument("--plot-only", action="store_true", help="No-op for consistency")
    args = parser.parse_args()

    if args.plot_only:
        print("Plot-only requested; STEP_FINAL does not generate plots. Skipping.")
        return

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    runtime_path = Path(args.runtime_config) if args.runtime_config else None
    if runtime_path is not None and not runtime_path.is_absolute():
        runtime_path = Path(__file__).resolve().parent / runtime_path

    physics_cfg, runtime_cfg, cfg, runtime_path = load_step_configs(config_path, runtime_path)

    input_dir = Path(cfg["input_dir"])
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parent / input_dir
    output_dir = Path(cfg.get("output_dir", "../../SIMULATED_DATA"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    geometry_id_cfg = cfg.get("geometry_id")
    if geometry_id_cfg is None or str(geometry_id_cfg).lower() == "auto":
        geometry_id = None
    else:
        geometry_id = int(geometry_id_cfg)
    rate_hz = float(cfg.get("rate_hz", 0.0))
    chunk_rows = cfg.get("chunk_rows")

    input_sim_run = cfg.get("input_sim_run", "latest")
    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)
    elif input_sim_run == "random":
        input_sim_run = random_sim_run(input_dir, cfg.get("seed"))
    input_glob = cfg.get("input_glob", "**/geom_*_daq.pkl")
    input_collect = str(cfg.get("input_collect", "matching")).lower()
    input_paths, baseline_meta = select_input_paths(
        input_dir, str(input_sim_run), geometry_id, input_glob, input_collect
    )

    map_dir = Path(cfg["geometry_map_dir"])
    if not map_dir.is_absolute():
        map_dir = Path(__file__).resolve().parent / map_dir
    map_sim_run = cfg.get("geometry_map_sim_run", "latest")
    if map_sim_run == "latest":
        map_sim_run = latest_sim_run(map_dir)
    elif map_sim_run == "random":
        map_sim_run = random_sim_run(map_dir, cfg.get("seed"))
    geom_map_path = map_dir / str(map_sim_run) / "geometry_map_all.csv"
    geom_map = pd.read_csv(geom_map_path)

    rng = np.random.default_rng(cfg.get("seed"))
    if geometry_id is None:
        meta_geom = baseline_meta.get("geometry_id")
        if meta_geom is None:
            meta_cfg = baseline_meta.get("config", {})
            meta_geom = meta_cfg.get("geometry_id")
        if meta_geom is None:
            raise ValueError("geometry_id not set in config and not found in Step 10 metadata.")
        geometry_id = int(meta_geom)
    selection = select_station_conf(geom_map, geometry_id, rng)
    start_date = selection.get("start")
    end_date = selection.get("end")
    if not start_date or not end_date:
        raise ValueError("start/end date columns are required in geometry_map_all.csv.")

    start_time = random_start_datetime(str(start_date), str(end_date), rng)
    station_id = int(selection.get("station", 0))

    target_rows = int(cfg.get("target_rows", 50000))
    if target_rows <= 0:
        raise ValueError("config target_rows must be a positive integer.")
    payloads, total_rows, thick_offsets = sample_payloads(
        input_paths, target_rows, chunk_rows, rng
    )
    out_name = build_filename(station_id, start_time)
    out_path = output_dir / out_name
    write_with_timestamps(payloads, out_path, start_time, rate_hz, rng, offsets_s=thick_offsets)

    registry_path = output_dir / "step_13_output_registry.json"
    registry = load_output_registry(registry_path)

    registry_entry = {
        "file_name": out_name,
        "created_at": now_iso(),
        "step": "STEP_FINAL",
        "config": physics_cfg,
        "runtime_config": runtime_cfg,
        "source_dataset": [str(path) for path in input_paths],
        "geometry_map": str(geom_map_path),
        "station_selection": selection,
        "start_time": start_time.isoformat(),
        "rate_hz": rate_hz,
        "target_rows": target_rows,
        "total_source_rows": total_rows,
        "input_collect": input_collect,
        "thick_time_mode": "offset" if thick_offsets is not None else "poisson",
        "baseline_meta": baseline_meta,
        "sim_run_configs": {
            "STEP_2_TO_3": collect_sim_run_configs(Path(cfg["geometry_map_dir"])),
            "STEP_10_TO_FINAL": collect_sim_run_configs(input_dir),
        },
    }
    registry["files"].append(registry_entry)
    save_output_registry(registry_path, registry)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
