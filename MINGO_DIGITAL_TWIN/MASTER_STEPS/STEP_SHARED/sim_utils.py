"""Shared helpers for sim-run bookkeeping, metadata, chunked I/O, and geometry utilities.
from __future__ import annotations
"""

from dataclasses import dataclass
import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import json
from datetime import datetime, timezone


@dataclass(frozen=True)
class DetectorBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


DEFAULT_BOUNDS = DetectorBounds(x_min=-150.0, x_max=150.0, y_min=-143.5, y_max=143.5)

Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]


def load_global_home_path() -> str:
    """Load home_path from config_global.yaml used across DATAFLOW_v3."""
    user_home = Path.home()
    config_file = user_home / "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml"
    with config_file.open("r") as handle:
        config = yaml.safe_load(handle)
    return config["home_path"]


def load_step_configs(
    physics_path: Path,
    runtime_path: Optional[Path] = None,
) -> Tuple[Dict, Dict, Dict, Path]:
    if not physics_path.exists():
        raise FileNotFoundError(f"Physics config not found: {physics_path}")
    if runtime_path is None:
        if physics_path.name.endswith("_physics.yaml"):
            runtime_path = physics_path.with_name(physics_path.name.replace("_physics.yaml", "_runtime.yaml"))
        else:
            raise ValueError("Runtime config path not provided and cannot infer from physics config name.")
    if not runtime_path.exists():
        raise FileNotFoundError(f"Runtime config not found: {runtime_path}")

    with physics_path.open("r") as handle:
        physics_cfg = yaml.safe_load(handle) or {}
    with runtime_path.open("r") as handle:
        runtime_cfg = yaml.safe_load(handle) or {}

    if not isinstance(physics_cfg, dict) or not isinstance(runtime_cfg, dict):
        raise ValueError("Both physics and runtime configs must be YAML mappings.")

    overlap = set(physics_cfg.keys()) & set(runtime_cfg.keys())
    if overlap:
        overlap_list = ", ".join(sorted(overlap))
        raise ValueError(f"Config keys overlap between physics and runtime configs: {overlap_list}")

    merged_cfg = dict(runtime_cfg)
    merged_cfg.update(physics_cfg)
    return physics_cfg, runtime_cfg, merged_cfg, runtime_path


def list_station_config_files(root_dir: Path) -> Dict[int, Path]:
    station_files: Dict[int, Path] = {}
    for station_dir in sorted(root_dir.glob("STATION_*")):
        if not station_dir.is_dir():
            continue
        station_id = int(station_dir.name.split("_")[-1])
        csv_files = list(station_dir.glob("input_file_mingo*.csv"))
        if not csv_files:
            continue
        station_files[station_id] = csv_files[0]
    return station_files


def read_station_config(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=1, decimal=",", dtype=str)
    df.columns = [col.strip() for col in df.columns]
    for col in ["station", "conf", "P1", "P2", "P3", "P4"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_geometry_map(station_df: pd.DataFrame) -> pd.DataFrame:
    geom_cols = ["P1", "P2", "P3", "P4"]
    extra_cols = [col for col in ("start", "end") if col in station_df.columns]
    unique_geoms = (
        station_df[geom_cols + extra_cols]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_geoms["geometry_id"] = np.arange(len(unique_geoms), dtype=int)
    merged = station_df.merge(unique_geoms, on=geom_cols, how="left")
    cols = ["station", "conf", "geometry_id", "P1", "P2", "P3", "P4"] + extra_cols
    return merged[cols]


def build_global_geometry_registry(station_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    geom_cols = ["P1", "P2", "P3", "P4"]
    all_geoms = pd.concat([df[geom_cols] for df in station_dfs], ignore_index=True)
    unique_geoms = all_geoms.dropna().drop_duplicates().reset_index(drop=True)
    unique_geoms["geometry_id"] = np.arange(len(unique_geoms), dtype=int)
    return unique_geoms[["geometry_id", "P1", "P2", "P3", "P4"]]


def map_station_to_geometry(station_df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    geom_cols = ["P1", "P2", "P3", "P4"]
    extra_cols = [col for col in ("start", "end") if col in station_df.columns]
    merged = station_df.merge(registry, on=geom_cols, how="left")
    cols = ["station", "conf", "geometry_id", "P1", "P2", "P3", "P4"] + extra_cols
    return merged[cols]


def iter_geometries(geom_map: pd.DataFrame) -> Iterable[Tuple[int, Tuple[float, float, float, float]]]:
    geom_cols = ["P1", "P2", "P3", "P4"]
    for geometry_id, group in geom_map.dropna(subset=["geometry_id"]).groupby("geometry_id"):
        values = group.iloc[0][geom_cols].to_numpy(dtype=float)
        yield int(geometry_id), (values[0], values[1], values[2], values[3])


def get_strip_geometry(plane_idx: int):
    y_width = Y_WIDTHS[0] if plane_idx in (1, 3) else Y_WIDTHS[1]
    total_width = np.sum(y_width)
    offsets = np.cumsum(np.concatenate(([0], y_width[:-1])))
    lower_edges = -total_width / 2 + offsets
    upper_edges = lower_edges + y_width
    centres = (lower_edges + upper_edges) / 2
    return y_width, centres, lower_edges, upper_edges


def num_strips_for_plane(plane_idx: int) -> int:
    y_width, _, _, _ = get_strip_geometry(plane_idx)
    return len(y_width)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_with_metadata(path: Path) -> Tuple[pd.DataFrame, Dict]:
    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
        meta = df.attrs.get("metadata", {})
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        return df, meta
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return df, meta
    raise ValueError(f"Unsupported input format: {path.suffix}")


def iter_input_frames(path: Path, chunk_rows: Optional[int]) -> Tuple[Iterable[pd.DataFrame], Dict, bool]:
    if path.name.endswith(".chunks.json"):
        manifest_path = path
    else:
        manifest_path = path.with_suffix(".chunks.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        meta = manifest.get("metadata", {})
        chunk_paths = manifest.get("chunks", [])

        def _iter() -> Iterable[pd.DataFrame]:
            for chunk_path in chunk_paths:
                chunk_file = Path(chunk_path)
                if chunk_file.suffix == ".csv":
                    yield pd.read_csv(chunk_file)
                elif chunk_file.suffix == ".pkl":
                    yield pd.read_pickle(chunk_file)
                else:
                    raise ValueError(f"Unsupported chunk format: {chunk_file.suffix}")

        return _iter(), meta, True

    if chunk_rows and path.suffix == ".csv":
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return pd.read_csv(path, chunksize=int(chunk_rows)), meta, True

    df, meta = load_with_metadata(path)
    return [df], meta, False


def find_latest_data_path(root_dir: Path) -> Optional[Path]:
    """Return the most recently modified data file under root_dir."""
    candidates = list(root_dir.rglob("*.pkl"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    candidates = list(root_dir.rglob("*.csv"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def find_sim_run_dir(path: Path) -> Optional[Path]:
    """Return the nearest SIM_RUN_* parent directory for a given path."""
    for parent in path.parents:
        if parent.name.startswith("SIM_RUN_"):
            return parent
    return None


def write_chunked_output(
    df_iter: Iterable[pd.DataFrame],
    output_dir: Path,
    out_stem: str,
    output_format: str,
    chunk_rows: int,
    metadata: Dict,
) -> Tuple[Path, Optional[pd.DataFrame], int]:
    chunks_dir = output_dir / f"{out_stem}_chunks"
    ensure_dir(chunks_dir)

    chunk_paths: List[str] = []
    buffer: List[pd.DataFrame] = []
    buffered_rows = 0
    full_chunks = 0
    last_chunk: Optional[pd.DataFrame] = None

    def flush_chunk(chunk_df: pd.DataFrame) -> None:
        nonlocal full_chunks, last_chunk
        chunk_path = chunks_dir / f"part_{full_chunks:04d}.{output_format}"
        if output_format == "csv":
            chunk_df.to_csv(chunk_path, index=False)
        elif output_format == "pkl":
            chunk_df.to_pickle(chunk_path)
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")
        chunk_paths.append(str(chunk_path))
        full_chunks += 1
        last_chunk = chunk_df

    def maybe_flush_buffer() -> None:
        nonlocal buffer, buffered_rows
        while buffered_rows >= chunk_rows:
            chunk_df = pd.concat(buffer, ignore_index=True)
            out_df = chunk_df.iloc[:chunk_rows].copy()
            remainder = chunk_df.iloc[chunk_rows:].copy()
            flush_chunk(out_df)
            buffer = [remainder] if not remainder.empty else []
            buffered_rows = len(remainder)

    total_rows = 0
    for df in df_iter:
        if df.empty:
            continue
        total_rows += len(df)
        buffer.append(df)
        buffered_rows += len(df)
        maybe_flush_buffer()

    if full_chunks == 0 and buffered_rows > 0:
        flush_chunk(pd.concat(buffer, ignore_index=True))
        buffered_rows = 0
        buffer = []
    else:
        buffered_rows = 0
        buffer = []

    row_count = full_chunks * chunk_rows + buffered_rows
    manifest = {
        "version": 1,
        "chunks": chunk_paths,
        "row_count": row_count,
        "metadata": metadata,
    }
    manifest_path = output_dir / f"{out_stem}.chunks.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path, last_chunk, row_count


def save_with_metadata(df: pd.DataFrame, path: Path, metadata: Dict, output_format: str) -> None:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    if output_format == "pkl":
        df.attrs["metadata"] = metadata
        df.to_pickle(path)
    elif output_format == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")


def _json_fingerprint(payload: Dict) -> str:
    payload_json = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _upstream_fingerprint(upstream_meta: Optional[Dict]) -> Optional[str]:
    if upstream_meta is None:
        return None
    if isinstance(upstream_meta, dict):
        return _json_fingerprint(
            {
                "step": upstream_meta.get("step"),
                "config_hash": upstream_meta.get("config_hash"),
                "upstream_hash": upstream_meta.get("upstream_hash"),
            }
        )
    return _json_fingerprint({"upstream_meta": upstream_meta})


def load_sim_run_registry(output_dir: Path) -> Dict:
    registry_path = output_dir / "sim_run_registry.json"
    if registry_path.exists():
        return json.loads(registry_path.read_text())
    return {"version": 1, "runs": []}


def save_sim_run_registry(output_dir: Path, registry: Dict) -> None:
    registry_path = output_dir / "sim_run_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2))


def resolve_sim_run(
    output_dir: Path,
    step: str,
    config_path: Path,
    cfg: Dict,
    upstream_meta: Optional[Dict],
) -> Tuple[str, Path, str, Optional[str], Dict]:
    config_hash = _json_fingerprint(cfg)
    upstream_hash = _upstream_fingerprint(upstream_meta)
    registry = load_sim_run_registry(output_dir)

    for entry in registry.get("runs", []):
        if entry.get("config_hash") == config_hash and entry.get("upstream_hash") == upstream_hash:
            sim_run = entry["sim_run"]
            return sim_run, output_dir / sim_run, config_hash, upstream_hash, registry

    existing = [entry.get("sim_run", "") for entry in registry.get("runs", [])]
    max_id = 0
    for run_id in existing:
        if run_id.startswith("SIM_RUN_"):
            try:
                max_id = max(max_id, int(run_id.split("_")[-1]))
            except ValueError:
                continue
    next_id = max_id + 1
    sim_run = f"SIM_RUN_{next_id:04d}"
    registry.setdefault("runs", []).append(
        {
            "sim_run": sim_run,
            "created_at": now_iso(),
            "step": step,
            "config_path": str(config_path),
            "config_hash": config_hash,
            "upstream_hash": upstream_hash,
            "config": cfg,
        }
    )
    save_sim_run_registry(output_dir, registry)
    return sim_run, output_dir / sim_run, config_hash, upstream_hash, registry


def find_sim_run(output_dir: Path, cfg: Dict, upstream_meta: Optional[Dict]) -> Optional[str]:
    config_hash = _json_fingerprint(cfg)
    upstream_hash = _upstream_fingerprint(upstream_meta)
    registry = load_sim_run_registry(output_dir)
    for entry in registry.get("runs", []):
        if entry.get("config_hash") == config_hash and entry.get("upstream_hash") == upstream_hash:
            return entry.get("sim_run")
    return None


def latest_sim_run(output_dir: Path) -> str:
    registry_path = output_dir / "sim_run_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("sim_run_registry.json not found in input_dir.")
    registry = json.loads(registry_path.read_text())
    runs = registry.get("runs", [])
    if not runs:
        raise FileNotFoundError("No SIM_RUN entries found in input_dir registry.")
    runs_sorted = sorted(runs, key=lambda r: r.get("created_at", ""))
    return runs_sorted[-1]["sim_run"]


def random_sim_run(output_dir: Path, seed: Optional[int] = None) -> str:
    registry_path = output_dir / "sim_run_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("sim_run_registry.json not found in input_dir.")
    registry = json.loads(registry_path.read_text())
    runs = [entry.get("sim_run") for entry in registry.get("runs", []) if entry.get("sim_run")]
    if not runs:
        raise FileNotFoundError("No SIM_RUN entries found in input_dir registry.")
    runs_sorted = sorted(runs)
    rng = random.Random(seed)
    return rng.choice(runs_sorted)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)
