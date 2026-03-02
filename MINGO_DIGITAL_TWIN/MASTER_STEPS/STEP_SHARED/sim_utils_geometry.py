"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_SHARED/sim_utils_geometry.py
Purpose: Geometry helpers shared by simulation steps.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_SHARED/sim_utils_geometry.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectorBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


DEFAULT_BOUNDS = DetectorBounds(x_min=-150.0, x_max=150.0, y_min=-143.5, y_max=143.5)

Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]


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
