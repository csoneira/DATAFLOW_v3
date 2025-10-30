#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for STEP_1 RAW→LIST tasks (lightweight, minimal coupling)."""

from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt
import yaml

CONFIG_PATH = Path.home() / "DATAFLOW_v3" / "MASTER" / "config_global.yaml"


@dataclass(frozen=True)
class StationPaths:
    station: str
    base: Path
    raw_dir: Path
    step_dir: Path
    task_dirs: Dict[str, Path]


def load_config() -> dict:
    with CONFIG_PATH.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_station_paths(station: str) -> StationPaths:
    station_id = f"MINGO0{station}"
    base = Path.home() / "DATAFLOW_v3" / "STATIONS" / station_id / "STAGE_1" / "EVENT_DATA"
    step_dir = base / "STEP_1"
    raw_dir = base / "RAW"
    task_dirs = {
        "TASK_1": step_dir / "TASK_1",
        "TASK_2": step_dir / "TASK_2",
        "TASK_3": step_dir / "TASK_3",
        "TASK_4": step_dir / "TASK_4",
    }
    return StationPaths(
        station=station,
        base=base,
        raw_dir=raw_dir,
        step_dir=step_dir,
        task_dirs=task_dirs,
    )


def ensure_task_layout(task_root: Path) -> Dict[str, Path]:
    files = task_root / "FILES"
    plots = task_root / "PLOTS"
    logs = task_root / "LOGS"
    queue_dirs = {
        "unprocessed": files / "UNPROCESSED",
        "processing": files / "PROCESSING",
        "completed": files / "COMPLETED",
        "error": files / "ERROR",
    }
    for path in [task_root, files, plots, logs, *queue_dirs.values()]:
        path.mkdir(parents=True, exist_ok=True)
    return {"plots": plots, "logs": logs, **queue_dirs}


def sync_sources_to_queue(source: Path, queue: Path) -> None:
    if not source.exists():
        return
    filenames = [p.name for p in source.iterdir() if p.is_file()]
    for name in filenames:
        src = source / name
        dst = queue / name
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
        except FileNotFoundError:
            continue


def choose_candidate(queue: Path, strategy: str = "oldest") -> Optional[Path]:
    candidates = [p for p in queue.iterdir() if p.is_file()]
    if not candidates:
        return None
    if strategy == "random":
        return random.choice(candidates)
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[0]


@contextmanager
def acquire_file(source: Path, processing: Path, strategy: str = "oldest") -> Iterator[Optional[Path]]:
    candidate = choose_candidate(source, strategy=strategy)
    if candidate is None:
        yield None
        return
    target = processing / candidate.name
    candidate.rename(target)
    try:
        yield target
    except Exception:
        raise


def complete_file(processed: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / processed.name
    processed.rename(target)
    return target


def fail_file(processed: Path, error_dir: Path) -> Path:
    error_dir.mkdir(parents=True, exist_ok=True)
    target = error_dir / processed.name
    processed.rename(target)
    return target


def save_dataframe_h5(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_for_hdf(data)
    prepared.to_hdf(path, key="data", mode="w")


def load_dataframe_h5(path: Path) -> pd.DataFrame:
    return pd.read_hdf(path, key="data")


def write_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def read_metadata(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def _prepare_for_hdf(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        series = result[col]
        if pdt.is_bool_dtype(series.dtype) or pdt.is_integer_dtype(series.dtype):
            result[col] = series.astype(np.float64)
        elif pdt.is_float_dtype(series.dtype):
            result[col] = series.astype(np.float64)
    return result
