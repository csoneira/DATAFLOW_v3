#!/usr/bin/env python3
"""
STEP_FINAL: format DAQ data and emit station .dat files with assigned date ranges.

Inputs: Step 10 output (step_10 or step_10_chunks).
Outputs: SIMULATED_DATA/SIM_RUN_<N>/mi00YYDDDHHMMSS.dat
         + step_final_output_registry.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, date, timezone
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
    normalize_param_mesh_ids,
    resolve_param_mesh,
    load_sim_run_registry,
    load_step_configs,
    load_with_metadata,
    now_iso,
    random_sim_run,
)


def _log_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _log_info(message: str) -> None:
    print(f"[{_log_ts()}] [STEP_FINAL] {message}")


def _log_warn(message: str) -> None:
    print(f"[{_log_ts()}] [STEP_FINAL] [WARN] {message}")


def _log_err(message: str) -> None:
    print(f"[{_log_ts()}] [STEP_FINAL] [ERROR] {message}")


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


def parse_param_datetime(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return parse_date(date_str)


PARAM_HASH_FIELDS = (
    "cos_n",
    "flux_cm2_min",
    "z_plane_1",
    "z_plane_2",
    "z_plane_3",
    "z_plane_4",
    "efficiencies",
    "trigger_combinations",
    "requested_rows",
    "sample_start_index",
)


def _normalize_hash_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_normalize_hash_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_hash_value(val) for key, val in value.items()}
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (
            (stripped.startswith("[") and stripped.endswith("]"))
            or (stripped.startswith("{") and stripped.endswith("}"))
        ):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            return _normalize_hash_value(parsed)
        return stripped
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return str(value)


def compute_param_hash(values: dict[str, object]) -> str:
    payload = {key: _normalize_hash_value(values.get(key)) for key in PARAM_HASH_FIELDS}
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def ensure_param_hash_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "param_hash" not in df.columns:
        df["param_hash"] = pd.NA
    missing = df["param_hash"].isna()
    if missing.any():
        df.loc[missing, "param_hash"] = [
            compute_param_hash(row.to_dict()) for _, row in df.loc[missing].iterrows()
        ]
    return df


def build_filename(station_id: int, timestamp: datetime) -> str:
    station_id = station_id + 4
    day_of_year = timestamp.timetuple().tm_yday
    return f"mi0{station_id}{timestamp.year % 100:02d}{day_of_year:03d}{timestamp:%H%M%S}.dat"


def build_sim_filename(timestamp: datetime) -> str:
    day_of_year = timestamp.timetuple().tm_yday
    year = timestamp.year
    return f"mi00{year % 100:02d}{day_of_year:03d}{timestamp:%H%M%S}.dat"


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


def _extract_sim_run(entry: dict) -> str | None:
    sim_run = entry.get("input_sim_run")
    if sim_run:
        return str(sim_run)
    sources = entry.get("source_dataset")
    if isinstance(sources, list):
        for src in sources:
            try:
                parts = Path(str(src)).parts
            except (TypeError, ValueError):
                continue
            for part in parts:
                if part.startswith("SIM_RUN_"):
                    return part
    return None


def _load_existing_outputs(registry: dict) -> tuple[dict, dict]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    dates_by_sim_run_station: dict[tuple[str, int], set] = defaultdict(set)
    for entry in registry.get("files", []):
        if not isinstance(entry, dict):
            continue
        sim_run = _extract_sim_run(entry)
        selection = entry.get("station_selection") or {}
        station = selection.get("station")
        conf = selection.get("conf")
        if sim_run is not None and station is not None and conf is not None:
            counts[(str(sim_run), str(station), str(conf))] += 1

        if sim_run is None or station is None:
            continue
        try:
            station_id = int(station)
        except (TypeError, ValueError):
            continue
        start_time = entry.get("start_time")
        if start_time:
            try:
                dt = datetime.fromisoformat(str(start_time))
            except ValueError:
                continue
            dates_by_sim_run_station[(str(sim_run), station_id)].add(dt.date())
    return counts, dates_by_sim_run_station


def normalize_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".chunks.json"):
        name = name[: -len(".chunks.json")]
    stem = Path(name).stem
    return stem.replace(".chunks", "")


def _load_row_count(path: Path, chunk_rows: int | None) -> int:
    manifest_path = path if path.name.endswith(".chunks.json") else path.with_suffix(".chunks.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        row_count = manifest.get("row_count")
        if row_count is not None:
            return int(row_count)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        row_count = meta.get("row_count")
        if row_count is not None:
            return int(row_count)
    total_rows = 0
    input_iter, _, _ = iter_input_frames(path, chunk_rows)
    for chunk in input_iter:
        total_rows += len(chunk)
    return total_rows


def load_input_meta(path: Path) -> dict:
    if path.name.endswith(".chunks.json"):
        manifest = json.loads(path.read_text())
        return manifest.get("metadata", {})
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    _, meta = load_with_metadata(path)
    return meta


def find_upstream_config(meta: dict, step: str) -> dict | None:
    current = meta
    while isinstance(current, dict):
        if current.get("step") == step:
            cfg = current.get("config")
            if isinstance(cfg, dict):
                return cfg
            return None
        current = current.get("upstream")
    return None


def find_upstream_value(meta: dict, key: str):
    current = meta
    while isinstance(current, dict):
        if key in current:
            return current.get(key)
        current = current.get("upstream")
    return None


def list_paths(run_dir: Path, input_glob: str) -> list[Path]:
    if "**" in input_glob:
        return sorted(run_dir.rglob(input_glob.replace("**/", "")))
    return sorted(run_dir.glob(input_glob))


def select_input_paths(
    input_dir: Path,
    input_sim_run: str,
    input_glob: str,
    input_collect: str,
) -> tuple[list[Path], dict]:
    input_run_dir = input_dir / str(input_sim_run)
    baseline_paths = list_paths(input_run_dir, input_glob)
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
        if input_collect == "matching":
            if meta.get("config_hash") != baseline_config_hash:
                continue
            if meta.get("upstream_hash") != baseline_upstream_hash:
                continue
        selected.append(path)

    if not selected:
        raise FileNotFoundError("No input files matched the selection criteria.")
    return selected, baseline_meta


def resolve_input_sim_runs(
    input_dir: Path,
    input_sim_run: object,
    seed: int | None,
) -> list[str]:
    if isinstance(input_sim_run, (list, tuple)):
        return [str(item) for item in input_sim_run]
    value = str(input_sim_run)
    if value.lower() in {"all", "each", "every"}:
        sim_runs = sorted(path.name for path in input_dir.glob("SIM_RUN_*"))
        if not sim_runs:
            raise FileNotFoundError(f"No SIM_RUN_* directories found in {input_dir}.")
        return sim_runs
    if value == "latest":
        return [latest_sim_run(input_dir)]
    if value == "random":
        return [random_sim_run(input_dir, seed)]
    return [value]


def ensure_unique_out_name(
    station_id: int,
    start_time: datetime,
    output_dir: Path,
    used_names: set[str],
    rng: np.random.Generator,
) -> tuple[str, datetime]:
    out_name = build_filename(station_id, start_time)
    out_path = output_dir / out_name
    if out_name not in used_names and not out_path.exists():
        used_names.add(out_name)
        return out_name, start_time

    day_start = datetime(start_time.year, start_time.month, start_time.day)
    for _ in range(2000):
        candidate = day_start + timedelta(seconds=int(rng.integers(0, 24 * 60 * 60)))
        out_name = build_filename(station_id, candidate)
        out_path = output_dir / out_name
        if out_name not in used_names and not out_path.exists():
            used_names.add(out_name)
            return out_name, candidate

    raise RuntimeError(f"Unable to find unique output name for station {station_id}.")


def ensure_unique_sim_name(
    start_time: datetime,
    output_dir: Path,
    used_names: set[str],
) -> tuple[str, datetime]:
    out_name = build_sim_filename(start_time)
    out_path = output_dir / out_name
    if out_name not in used_names and not out_path.exists():
        used_names.add(out_name)
        return out_name, start_time
    candidate = start_time
    for _ in range(24 * 60 * 60):
        candidate = candidate + timedelta(seconds=1)
        out_name = build_sim_filename(candidate)
        out_path = output_dir / out_name
        if out_name not in used_names and not out_path.exists():
            used_names.add(out_name)
            return out_name, candidate
    raise RuntimeError("Unable to find unique output name for simulation day.")


def parse_subsample_rows(value: object) -> list[int]:
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.lower() in {"none", "null", "[]"}:
            return []
        raw_rows = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw_rows = list(value)
    else:
        raise ValueError("config subsample_rows must be a list of integers or a comma-separated string.")

    rows: list[int] = []
    for row in raw_rows:
        row_i = int(row)
        if row_i <= 0:
            raise ValueError("config subsample_rows values must be positive integers.")
        rows.append(row_i)
    return rows


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def collect_input_row_counts(
    input_paths: list[Path],
    chunk_rows: int | None,
) -> tuple[list[int], int]:
    row_counts: list[int] = []
    for path in input_paths:
        total = 0
        input_iter, _, _ = iter_input_frames(path, chunk_rows)
        for chunk in input_iter:
            total += len(chunk)
        row_counts.append(total)
        print(f"Input rows: {path} -> {total}")
    total_rows = int(sum(row_counts))
    print(f"Total rows across inputs: {total_rows}")
    return row_counts, total_rows


def build_payload(row_dict: dict) -> str:
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
                payload = build_payload(row_dict)
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


def sample_payloads_sequential(
    input_paths: list[Path],
    target_rows: int,
    chunk_rows: int | None,
    rng: np.random.Generator,
    total_rows: int | None = None,
) -> tuple[list[str], int, list[float] | None, int]:
    if total_rows is None:
        _, total_rows = collect_input_row_counts(input_paths, chunk_rows)
    if total_rows <= 0:
        raise ValueError("No rows available in the input data.")
    if target_rows <= 0:
        raise ValueError("target_rows must be positive.")
    if total_rows < target_rows:
        raise ValueError(
            f"Requested {target_rows} rows but only {total_rows} are available for sequential sampling."
        )

    start_idx = int(rng.integers(0, total_rows - target_rows + 1))
    print(f"Sequential sampling start index ({target_rows} rows): {start_idx}")

    remaining_skip = start_idx
    remaining_take = target_rows
    payloads: list[str] = []
    offsets: list[float] | None = None
    use_thick: bool | None = None

    for path in input_paths:
        input_iter, _, _ = iter_input_frames(path, chunk_rows)
        for chunk in input_iter:
            if chunk.empty:
                continue
            has_thick = "T_thick_s" in chunk.columns
            if use_thick is None:
                use_thick = has_thick
            elif use_thick != has_thick:
                raise ValueError("Inconsistent T_thick_s presence across input chunks.")

            chunk_len = len(chunk)
            if remaining_skip >= chunk_len:
                remaining_skip -= chunk_len
                continue

            start = remaining_skip
            remaining_skip = 0
            take = min(chunk_len - start, remaining_take)
            subset = chunk.iloc[start : start + take]
            for row in subset.itertuples(index=False):
                row_dict = row._asdict()
                payloads.append(build_payload(row_dict))
                if use_thick:
                    if offsets is None:
                        offsets = []
                    offsets.append(float(row_dict.get("T_thick_s", 0.0)) or 0.0)
            remaining_take -= take
            if remaining_take == 0:
                break
        if remaining_take == 0:
            break

    if remaining_take != 0:
        raise RuntimeError(
            f"Failed to collect {target_rows} rows; {remaining_take} remaining."
        )

    if offsets is not None and offsets:
        first = offsets[0]
        offsets = [val - first for val in offsets]
    return payloads, total_rows, offsets, start_idx


def write_with_timestamps(
    payloads: list[str],
    output_path: Path,
    start_time: datetime,
    rate_hz: float,
    rng: np.random.Generator,
    offsets_s: list[float] | None = None,
    param_hash: str | None = None,
) -> tuple[int, datetime | None]:
    day_end = datetime(start_time.year, start_time.month, start_time.day) + timedelta(days=1)
    rows_written = 0
    last_written_time: datetime | None = None
    with output_path.open("w", encoding="ascii") as dst:
        if param_hash:
            dst.write(f"# param_hash={param_hash}\n")
        current_time = start_time
        for idx, payload in enumerate(payloads):
            if offsets_s is not None:
                event_time = start_time + timedelta(seconds=float(offsets_s[idx]))
            else:
                event_time = current_time

            if event_time >= day_end:
                break

            year = event_time.year
            month = event_time.month
            day = event_time.day
            hour = event_time.hour
            minute = event_time.minute
            second = event_time.second
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
            rows_written += 1
            last_written_time = event_time
            if offsets_s is None and rate_hz > 0:
                delta = rng.exponential(1.0 / rate_hz)
                current_time = event_time + timedelta(seconds=float(delta))
            elif offsets_s is None:
                current_time = event_time
    return rows_written, last_written_time


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

    _log_info("STEP_FINAL started")

    if args.plot_only:
        _log_warn("Plot-only requested; STEP_FINAL does not generate plots. Skipping.")
        return

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

    input_dir = Path(cfg["input_dir"])
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parent / input_dir
    output_dir = Path(cfg.get("output_dir", "../../SIMULATED_DATA"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)
    _log_info(f"Input dir: {input_dir}")
    _log_info(f"Output dir: {output_dir}")

    rate_hz = float(cfg.get("rate_hz", 0.0))
    chunk_rows = cfg.get("chunk_rows")
    payload_sampling = str(cfg.get("payload_sampling", "random")).lower()
    if payload_sampling not in {"random", "sequential_random_start"}:
        raise ValueError("payload_sampling must be 'random' or 'sequential_random_start'.")

    input_glob = cfg.get("input_glob", "**/step_10_chunks.chunks.json")
    input_collect = str(cfg.get("input_collect", "matching")).lower()
    input_sim_run = cfg.get("input_sim_run", "latest")
    sim_runs = resolve_input_sim_runs(input_dir, input_sim_run, cfg.get("seed"))
    if len(sim_runs) > 1 and input_collect != "baseline_only":
        _log_warn("Multiple SIM_RUNs requested; using baseline_only input collection per SIM_RUN.")
        input_collect = "baseline_only"
    _log_info(f"SIM_RUNs selected: {len(sim_runs)}")

    mesh_dir = Path(cfg.get("param_mesh_dir", "../../INTERSTEPS/STEP_0_TO_1"))
    if not mesh_dir.is_absolute():
        mesh_dir = Path(__file__).resolve().parent / mesh_dir

    requested_rows = int(cfg.get("target_rows", 50000))
    if requested_rows <= 0:
        raise ValueError("config target_rows must be a positive integer.")
    subsample_rows = parse_subsample_rows(cfg.get("subsample_rows", []))
    if subsample_rows and payload_sampling != "sequential_random_start":
        raise ValueError(
            "config subsample_rows requires payload_sampling='sequential_random_start'."
        )

    registry_enabled = parse_bool(cfg.get("registry_enabled"), True)
    registry_path = output_dir / "step_final_output_registry.json"
    registry = load_output_registry(registry_path) if registry_enabled else {"version": 1, "files": []}
    used_output_names: dict[Path, set[str]] = defaultdict(set)
    for existing_file in output_dir.rglob("mi0*.dat"):
        used_output_names[existing_file.parent].add(existing_file.name)
    rng = np.random.default_rng(cfg.get("seed"))
    target_files = int(cfg.get("files_per_station_conf", 1))
    if target_files <= 0:
        raise ValueError("config files_per_station_conf must be a positive integer.")

    for sim_run in sim_runs:
        input_paths, baseline_meta = select_input_paths(
            input_dir, str(sim_run), input_glob, input_collect
        )
        step1_cfg = find_upstream_config(baseline_meta, "STEP_1") or {}
        step3_cfg = find_upstream_config(baseline_meta, "STEP_3") or {}
        z_positions = find_upstream_value(baseline_meta, "z_positions_raw_mm")
        if z_positions is None:
            z_positions = find_upstream_value(baseline_meta, "z_positions_mm")
        if z_positions is None or len(z_positions) != 4:
            raise ValueError("z_positions_mm not found in upstream metadata.")
        effs = step3_cfg.get("efficiencies") or []
        if not isinstance(effs, list) or len(effs) != 4:
            raise ValueError("efficiencies not found in STEP_3 metadata.")
        mesh, mesh_path = resolve_param_mesh(mesh_dir, cfg.get("param_mesh_sim_run", "none"), cfg.get("seed"))
        if "done" not in mesh.columns:
            mesh["done"] = 0
        has_param_set_id = "param_set_id" in mesh.columns
        has_param_date = "param_date" in mesh.columns
        if has_param_set_id:
            pending_mask = mesh["param_set_id"].isna()
        else:
            pending_mask = pd.Series(True, index=mesh.index)
        mask = (mesh["done"] != 1) & pending_mask
        mask &= np.isclose(mesh["cos_n"].astype(float), float(step1_cfg.get("cos_n")))
        mask &= np.isclose(mesh["flux_cm2_min"].astype(float), float(step1_cfg.get("flux_cm2_min")))
        mask &= np.isclose(mesh["eff_p1"].astype(float), float(effs[0]))
        mask &= np.isclose(mesh["eff_p2"].astype(float), float(effs[1]))
        mask &= np.isclose(mesh["eff_p3"].astype(float), float(effs[2]))
        mask &= np.isclose(mesh["eff_p4"].astype(float), float(effs[3]))
        abs_mask = mask.copy()
        abs_mask &= np.isclose(mesh["z_p1"].astype(float), float(z_positions[0]))
        abs_mask &= np.isclose(mesh["z_p2"].astype(float), float(z_positions[1]))
        abs_mask &= np.isclose(mesh["z_p3"].astype(float), float(z_positions[2]))
        abs_mask &= np.isclose(mesh["z_p4"].astype(float), float(z_positions[3]))
        candidates = mesh[abs_mask]
        if candidates.empty:
            base = mesh["z_p1"].astype(float)
            rel_mask = mask.copy()
            rel_mask &= np.isclose(mesh["z_p1"].astype(float) - base, float(z_positions[0]))
            rel_mask &= np.isclose(mesh["z_p2"].astype(float) - base, float(z_positions[1]))
            rel_mask &= np.isclose(mesh["z_p3"].astype(float) - base, float(z_positions[2]))
            rel_mask &= np.isclose(mesh["z_p4"].astype(float) - base, float(z_positions[3]))
            candidates = mesh[rel_mask]
        if candidates.empty:
            full_mask = (
                np.isclose(mesh["cos_n"].astype(float), float(step1_cfg.get("cos_n")))
                & np.isclose(mesh["flux_cm2_min"].astype(float), float(step1_cfg.get("flux_cm2_min")))
                & np.isclose(mesh["eff_p1"].astype(float), float(effs[0]))
                & np.isclose(mesh["eff_p2"].astype(float), float(effs[1]))
                & np.isclose(mesh["eff_p3"].astype(float), float(effs[2]))
                & np.isclose(mesh["eff_p4"].astype(float), float(effs[3]))
            )
            base = mesh["z_p1"].astype(float)
            abs_full = full_mask.copy()
            abs_full &= np.isclose(mesh["z_p1"].astype(float), float(z_positions[0]))
            abs_full &= np.isclose(mesh["z_p2"].astype(float), float(z_positions[1]))
            abs_full &= np.isclose(mesh["z_p3"].astype(float), float(z_positions[2]))
            abs_full &= np.isclose(mesh["z_p4"].astype(float), float(z_positions[3]))
            rel_full = full_mask.copy()
            rel_full &= np.isclose(mesh["z_p1"].astype(float) - base, float(z_positions[0]))
            rel_full &= np.isclose(mesh["z_p2"].astype(float) - base, float(z_positions[1]))
            rel_full &= np.isclose(mesh["z_p3"].astype(float) - base, float(z_positions[2]))
            rel_full &= np.isclose(mesh["z_p4"].astype(float) - base, float(z_positions[3]))
            candidates = mesh[abs_full]
            if candidates.empty:
                candidates = mesh[rel_full]
        if candidates.empty:
            _log_warn("No matching param_mesh row found for this parameter set; skipping.")
            continue
        if len(candidates) > 1:
            raise ValueError("Multiple matching param_mesh rows found; cannot assign param_set_id.")
        param_row = candidates.iloc[0]
        existing_param_set_id = param_row.get("param_set_id") if has_param_set_id else pd.NA
        existing_param_date = param_row.get("param_date") if has_param_date else pd.NA
        sim_params_path = output_dir / "step_final_simulation_params.csv"
        sim_params_df = None
        sim_params_needs_write = False
        if sim_params_path.exists():
            sim_params_df = pd.read_csv(sim_params_path)
            drop_cols = [col for col in ("subfile_kind", "subfile_index") if col in sim_params_df.columns]
            if drop_cols:
                sim_params_df = sim_params_df.drop(columns=drop_cols)
                sim_params_needs_write = True
            if "param_hash" not in sim_params_df.columns or sim_params_df["param_hash"].isna().any():
                sim_params_df = ensure_param_hash_column(sim_params_df)
                sim_params_needs_write = True
        if pd.isna(existing_param_set_id) or pd.isna(existing_param_date):
            if has_param_set_id:
                existing_ids = mesh["param_set_id"].dropna()
            elif sim_params_df is not None and "param_set_id" in sim_params_df.columns:
                existing_ids = sim_params_df["param_set_id"].dropna()
            else:
                existing_ids = pd.Series([], dtype="float64")
            next_id = int(existing_ids.max()) + 1 if not existing_ids.empty else 1
            if sim_params_df is not None and "param_date" in sim_params_df.columns:
                existing_dates = sim_params_df["param_date"].dropna()
            elif has_param_date:
                existing_dates = mesh["param_date"].dropna()
            else:
                existing_dates = pd.Series([], dtype="object")
            if not existing_dates.empty:
                last_date = parse_param_datetime(str(existing_dates.iloc[-1])).date()
                next_date = last_date + timedelta(days=1)
            else:
                next_date = date(2000, 1, 1)
            if has_param_set_id:
                mesh.loc[param_row.name, "param_set_id"] = next_id
            if has_param_date:
                mesh.loc[param_row.name, "param_date"] = next_date.isoformat()
            mesh.loc[param_row.name, "done"] = 1
            param_set_id = next_id
            param_date = next_date.isoformat()
        else:
            param_set_id = int(existing_param_set_id)
            param_date = str(existing_param_date)
        z_cols = ["z_p1", "z_p2", "z_p3", "z_p4"]
        head_cols = ["done", "param_set_id", "param_date"]
        ordered_cols = [c for c in head_cols if c in mesh.columns] + [
            c for c in mesh.columns if c not in head_cols and c not in z_cols
        ] + [c for c in z_cols if c in mesh.columns]
        mesh = mesh[ordered_cols]
        mesh["done"] = mesh["done"].fillna(0).astype(int)
        mesh = normalize_param_mesh_ids(mesh)
        mesh.to_csv(mesh_path, index=False)
        if param_date is None or pd.isna(param_date):
            raise ValueError("param_date is missing; enable param_date column or ensure it is populated.")
        base_time = parse_param_datetime(str(param_date))
        z_vals = [
            float(param_row["z_p1"]),
            float(param_row["z_p2"]),
            float(param_row["z_p3"]),
            float(param_row["z_p4"]),
        ]
        step1_cfg = find_upstream_config(baseline_meta, "STEP_1") or {}
        step3_cfg = find_upstream_config(baseline_meta, "STEP_3") or {}
        step9_cfg = find_upstream_config(baseline_meta, "STEP_9") or {}
        total_rows: int
        if payload_sampling == "sequential_random_start":
            _, total_rows = collect_input_row_counts(input_paths, chunk_rows)
            if total_rows <= 0:
                raise ValueError("No rows available in the input data.")
        else:
            total_rows = -1

        file_plans: list[dict[str, object]] = []
        for _ in range(target_files):
            file_plans.append({"kind": "target", "requested_rows": requested_rows})
        for sub_rows in subsample_rows:
            file_plans.append({"kind": "subsample", "requested_rows": int(sub_rows)})

        existing_for_run = sum(
            1
            for entry in registry.get("files", [])
            if entry.get("param_set_id") == int(param_set_id)
        )
        if existing_for_run >= len(file_plans):
            _log_info(
                "Skipping existing output for "
                f"param_set_id={param_set_id} "
                f"({existing_for_run}/{len(file_plans)})."
            )
            continue

        sim_dir = output_dir
        ensure_dir(sim_dir)
        dir_used_names = used_output_names[sim_dir]
        next_start = datetime.combine(base_time.date(), datetime.min.time())
        new_param_rows: list[dict] = []

        for subfile_index, plan in enumerate(file_plans):
            file_kind = str(plan["kind"])
            requested_for_file = int(plan["requested_rows"])
            desired_rows = requested_for_file
            sample_start_index: int | None = None

            if payload_sampling == "sequential_random_start":
                if total_rows < desired_rows:
                    if file_kind == "subsample":
                        _log_warn(
                            f"Skipping subsample_rows={desired_rows}; only {total_rows} source rows available."
                        )
                        continue
                    _log_warn(
                        f"Requested {desired_rows} rows but only {total_rows} available; using {total_rows}."
                    )
                    desired_rows = total_rows

                payloads, _, thick_offsets, sample_start_index = sample_payloads_sequential(
                    input_paths,
                    desired_rows,
                    chunk_rows,
                    rng,
                    total_rows=total_rows,
                )
            else:
                payloads, total_rows, thick_offsets = sample_payloads(
                    input_paths,
                    desired_rows,
                    chunk_rows,
                    rng,
                )

            if not payloads:
                _log_warn(
                    f"Skipping file generation for param_set_id={param_set_id}: no payload rows available."
                )
                continue

            out_name, start_time = ensure_unique_sim_name(next_start, sim_dir, dir_used_names)
            out_path = sim_dir / out_name
            hash_payload = {
                "cos_n": step1_cfg.get("cos_n"),
                "flux_cm2_min": step1_cfg.get("flux_cm2_min"),
                "z_plane_1": z_vals[0],
                "z_plane_2": z_vals[1],
                "z_plane_3": z_vals[2],
                "z_plane_4": z_vals[3],
                "efficiencies": step3_cfg.get("efficiencies", []),
                "trigger_combinations": step9_cfg.get("trigger_combinations", []),
                "requested_rows": requested_for_file,
                "sample_start_index": sample_start_index,
            }
            param_hash = compute_param_hash(hash_payload)
            rows_written, last_time = write_with_timestamps(
                payloads,
                out_path,
                start_time,
                rate_hz,
                rng,
                offsets_s=thick_offsets,
                param_hash=param_hash,
            )

            if rows_written == 0:
                out_path.unlink(missing_ok=True)
                _log_warn(f"Skipping empty output {out_name}; no rows fit before midnight.")
                continue

            if rows_written < len(payloads):
                _log_warn(
                    f"Cut {out_name} at midnight: wrote {rows_written}/{len(payloads)} rows."
                )
            selected_rows = rows_written
            next_start = (
                last_time + timedelta(seconds=1)
                if last_time is not None
                else start_time + timedelta(seconds=1)
            )

            registry_entry = {
                "file_name": out_name,
                "file_path": str(out_path.relative_to(output_dir)),
                "created_at": now_iso(),
                "step": "STEP_FINAL",
                "config": physics_cfg,
                "runtime_config": runtime_cfg,
                "param_set_id": int(param_set_id),
                "param_date": str(param_date),
                "source_dataset": [str(path) for path in input_paths],
                "param_mesh": str(mesh_path),
                "start_time": start_time.isoformat(),
                "rate_hz": rate_hz,
                "target_rows": requested_for_file,
                "requested_rows": requested_for_file,
                "selected_rows": selected_rows,
                "subfile_kind": file_kind,
                "subfile_index": subfile_index,
                "sample_start_index": sample_start_index,
                "total_source_rows": total_rows,
                "input_collect": input_collect,
                "thick_time_mode": "offset" if thick_offsets is not None else "poisson",
                "baseline_meta": baseline_meta,
                "sim_run_configs": {
                    "STEP_10_TO_FINAL": collect_sim_run_configs(input_dir),
                },
            }
            if registry_enabled:
                registry["files"].append(registry_entry)
                save_output_registry(registry_path, registry)

            row = {
                "file_name": out_name,
                "param_hash": param_hash,
                "param_set_id": int(param_set_id),
                "param_date": str(param_date),
                "cos_n": step1_cfg.get("cos_n"),
                "flux_cm2_min": step1_cfg.get("flux_cm2_min"),
                "z_plane_1": z_vals[0],
                "z_plane_2": z_vals[1],
                "z_plane_3": z_vals[2],
                "z_plane_4": z_vals[3],
                "efficiencies": json.dumps(step3_cfg.get("efficiencies", [])),
                "trigger_combinations": json.dumps(step9_cfg.get("trigger_combinations", [])),
                "requested_rows": requested_for_file,
                "selected_rows": selected_rows,
                "sample_start_index": sample_start_index,
            }
            new_param_rows.append(row)
            _log_info(
                f"Saved {out_path} (param_set_id={param_set_id}, requested_rows={requested_for_file}, selected_rows={selected_rows})"
            )

        if new_param_rows:
            if sim_params_df is not None:
                df_params = sim_params_df
                drop_cols = [
                    col
                    for col in ("file_path", "input_sim_run", "subfile_kind", "subfile_index")
                    if col in df_params.columns
                ]
                if drop_cols:
                    df_params = df_params.drop(columns=drop_cols)
                for new_row in new_param_rows:
                    df_params = df_params[df_params["file_name"] != new_row["file_name"]]
                df_params = pd.concat([df_params, pd.DataFrame(new_param_rows)], ignore_index=True)
            else:
                df_params = pd.DataFrame(new_param_rows)
            df_params.to_csv(sim_params_path, index=False)
        elif sim_params_needs_write and sim_params_df is not None:
            sim_params_df.to_csv(sim_params_path, index=False)


if __name__ == "__main__":
    main()
