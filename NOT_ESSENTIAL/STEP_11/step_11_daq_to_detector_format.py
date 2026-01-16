#!/usr/bin/env python3
"""Step 11: serialize DAQ-level data into detector-style text rows.

Inputs: geom_<G>_daq from Step 10.
Outputs: mi00YYDDDHHMMSS.dat (chunked as needed) with metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
    load_with_metadata,
    load_step_configs,
    now_iso,
    resolve_sim_run,
    reset_dir,
)


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


def find_step_meta(meta: dict | None, step_name: str) -> dict | None:
    current = meta
    while isinstance(current, dict):
        if current.get("step") == step_name:
            return current
        current = current.get("upstream")
    return None


def resolve_thick_rate_hz(meta: dict | None) -> float:
    step1_meta = find_step_meta(meta, "STEP_1")
    if not step1_meta:
        return 0.0
    step1_cfg = step1_meta.get("config", {})
    if not isinstance(step1_cfg, dict):
        return 0.0
    flux_cm2_min = float(step1_cfg.get("flux_cm2_min", 1.0))
    xlim_mm = float(step1_cfg.get("xlim_mm", 0.0))
    ylim_mm = float(step1_cfg.get("ylim_mm", 0.0))
    if xlim_mm <= 0 or ylim_mm <= 0:
        return 0.0
    area_cm2 = (2.0 * xlim_mm) * (2.0 * ylim_mm) / 100.0
    rate_per_min = flux_cm2_min * area_cm2
    return rate_per_min / 60.0


class ThickTimeSequencer:
    def __init__(self, rate_hz: float, rng: np.random.Generator) -> None:
        self.rate_hz = rate_hz
        self.rng = rng
        self.current_time = 0.0
        self.first_event = True

    def next(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=float)
        if self.rate_hz <= 0:
            return np.zeros(n, dtype=float)
        intervals = self.rng.exponential(1.0 / self.rate_hz, size=n)
        if self.first_event:
            intervals[0] = 0.0
            self.first_event = False
        times = self.current_time + np.cumsum(intervals)
        self.current_time = float(times[-1])
        return times


def format_time_s(val: float) -> str:
    if not np.isfinite(val):
        val = 0.0
    return f"{val:.6f}"


def write_event_rows(
    df: pd.DataFrame,
    cfg: dict,
    output_path: Path,
    thick_time_col: str | None = None,
) -> None:
    year, month, day, hour, minute, second = extract_timestamp(cfg)
    event_type = 1

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
            if thick_time_col and thick_time_col in row_dict:
                parts.append(format_time_s(float(row_dict.get(thick_time_col, 0.0))))

            handle.write(" ".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11: format DAQ output as detector text data.")
    parser.add_argument("--config", default="config_step_11_physics.yaml", help="Path to step physics config YAML")
    parser.add_argument(
        "--runtime-config",
        default=None,
        help="Path to step runtime config YAML (defaults to *_runtime.yaml)",
    )
    parser.add_argument("--no-plots", action="store_true", help="No-op for consistency")
    parser.add_argument("--plot-only", action="store_true", help="No-op for consistency")
    args = parser.parse_args()

    if args.plot_only:
        print("Plot-only requested; Step 11 does not generate plots. Skipping.")
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
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    input_glob = cfg.get("input_glob", "**/geom_*_daq.pkl")
    chunk_rows = cfg.get("chunk_rows")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is not None and str(geometry_id).lower() != "auto":
        geometry_id = int(geometry_id)
    else:
        geometry_id = None
    input_sim_run = cfg.get("input_sim_run", "latest")
    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)

    input_run_dir = input_dir / str(input_sim_run)
    if "**" in input_glob:
        input_paths = sorted(input_run_dir.rglob(input_glob.replace("**/", "")))
    else:
        input_paths = sorted(input_run_dir.glob(input_glob))
    def normalize_stem(path: Path) -> str:
        name = path.name
        if name.endswith(".chunks.json"):
            name = name[: -len(".chunks.json")]
        stem = Path(name).stem
        return stem.replace(".chunks", "")

    if geometry_id is not None:
        geom_key = f"geom_{geometry_id}"
        input_paths = [
            p for p in input_paths if normalize_stem(p) == f"{geom_key}_daq"
        ]
        if not input_paths:
            fallback_path = input_run_dir / f"{geom_key}_daq.chunks.json"
            if fallback_path.exists():
                input_paths = [fallback_path]
    elif not input_paths:
        input_paths = sorted(input_run_dir.glob("geom_*_daq.chunks.json"))
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    if geometry_id is None:
        normalized_stem = normalize_stem(input_path)
        parts = normalized_stem.split("_")
        if len(parts) < 2 or parts[0] != "geom":
            raise ValueError(f"Unable to infer geometry_id from {input_path.stem}")
        geometry_id = int(parts[1])
    print(f"Processing: {input_path}")
    input_iter, upstream_meta, chunked_input = iter_input_frames(input_path, chunk_rows)
    thick_rate_hz = resolve_thick_rate_hz(upstream_meta)
    thick_time_col = "T_thick_s"
    thick_seq = None
    if thick_rate_hz > 0:
        thick_seq = ThickTimeSequencer(thick_rate_hz, np.random.default_rng())
        print(f"Step 11 thick times: rate_hz={thick_rate_hz}")

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_11", config_path, physics_cfg, upstream_meta
    )
    reset_dir(sim_run_dir)

    out_name = build_output_name(cfg)
    out_stem = out_name.replace(".dat", "")

    if chunk_rows:
        chunk_paths = []
        buffer = []
        buffered_rows = 0
        full_chunks = 0
        metadata = {
            "created_at": now_iso(),
            "step": "STEP_11",
            "config": physics_cfg,
            "runtime_config": runtime_cfg,
            "geometry_id": geometry_id,
            "sim_run": sim_run,
            "config_hash": config_hash,
            "upstream_hash": upstream_hash,
            "source_dataset": str(input_path),
            "upstream": upstream_meta,
        }

        def flush_chunk(out_df: pd.DataFrame, idx: int) -> None:
            chunk_name = f"{out_stem}_part_{idx:04d}.dat"
            chunk_path = sim_run_dir / chunk_name
            if thick_seq:
                out_df = out_df.copy()
                out_df[thick_time_col] = thick_seq.next(len(out_df))
            write_event_rows(out_df, cfg, chunk_path, thick_time_col=thick_time_col)
            meta_path = chunk_path.with_suffix(chunk_path.suffix + ".meta.json")
            meta_path.write_text(json.dumps(metadata, indent=2))
            chunk_paths.append(str(chunk_path))

        for chunk in input_iter:
            if chunk.empty:
                continue
            buffer.append(chunk)
            buffered_rows += len(chunk)
            while buffered_rows >= int(chunk_rows):
                full_df = pd.concat(buffer, ignore_index=True)
                out_df = full_df.iloc[: int(chunk_rows)].copy()
                remainder = full_df.iloc[int(chunk_rows):].copy()
                flush_chunk(out_df, full_chunks)
                full_chunks += 1
                buffer = [remainder] if not remainder.empty else []
                buffered_rows = len(remainder)

        if full_chunks == 0 and buffered_rows > 0:
            full_df = pd.concat(buffer, ignore_index=True)
            flush_chunk(full_df, full_chunks)
            full_chunks += 1
            buffered_rows = 0
            buffer = []
        else:
            buffered_rows = 0
            buffer = []

        manifest = {
            "version": 1,
            "chunks": chunk_paths,
            "row_count": full_chunks * int(chunk_rows) + buffered_rows,
            "metadata": metadata,
        }
        manifest_path = sim_run_dir / f"{out_stem}.chunks.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        out_path = manifest_path
    else:
        df, upstream_meta = load_with_metadata(input_path)
        out_path = sim_run_dir / out_name
        if thick_seq:
            df = df.copy()
            df[thick_time_col] = thick_seq.next(len(df))
        write_event_rows(df, cfg, out_path, thick_time_col=thick_time_col)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    if "metadata" not in locals():
        metadata = {
            "created_at": now_iso(),
            "step": "STEP_11",
            "config": physics_cfg,
            "runtime_config": runtime_cfg,
            "geometry_id": geometry_id,
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
