#!/usr/bin/env python3
from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "OUTPUTS"
FILES_DIR = OUTPUT_DIR / "FILES"
PLOTS_DIR = OUTPUT_DIR / "PLOTS"
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.json"

CANONICAL_EFF_COLUMNS = ["emp_eff_1", "emp_eff_2", "emp_eff_3", "emp_eff_4"]
CANONICAL_Z_COLUMNS = ["z_pos_1", "z_pos_2", "z_pos_3", "z_pos_4"]
STEP1_OUTPUT_COLUMNS = (
    CANONICAL_Z_COLUMNS
    + CANONICAL_EFF_COLUMNS
    + ["rate_hz", "sim_flux_cm2_min", "num_events"]
)


def ensure_output_dirs() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_path(config: dict[str, Any], raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path(config["_config_dir"]) / path).resolve()


def cfg_path(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        value = value[key]
    return resolve_path(config, value)


def get_rate_column_name(config: dict[str, Any]) -> str:
    columns = config.get("columns", {})
    if not isinstance(columns, dict):
        raise ValueError("Config is missing the 'columns' object.")
    raw_value = columns.get("rate", columns.get("global_rate"))
    if raw_value in (None, "", "null", "None"):
        raise ValueError("Config must define columns.rate or columns.global_rate.")
    return str(raw_value)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonicalize_trigger_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (list, dict, int, float, bool)):
        return value

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = text

    if isinstance(parsed, tuple):
        return list(parsed)
    return parsed


def choose_z_vector(
    dataframe: pd.DataFrame,
    z_columns: list[str],
    configured_vector: list[float] | None,
) -> tuple[float, float, float, float]:
    if configured_vector:
        if len(configured_vector) != len(z_columns):
            raise ValueError(
                f"Configured z_position_vector has length {len(configured_vector)}; "
                f"expected {len(z_columns)}."
            )
        return tuple(float(value) for value in configured_vector)

    counts = (
        dataframe.groupby(z_columns, dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["row_count"] + z_columns, ascending=[False] + [True] * len(z_columns))
        .reset_index(drop=True)
    )
    if counts.empty:
        raise ValueError("Could not determine a z-position vector from an empty dataframe.")
    row = counts.iloc[0]
    return tuple(float(row[col]) for col in z_columns)


def filter_by_z_vector(
    dataframe: pd.DataFrame,
    z_columns: list[str],
    z_vector: tuple[float, float, float, float],
) -> pd.DataFrame:
    mask = np.ones(len(dataframe), dtype=bool)
    for column, value in zip(z_columns, z_vector):
        mask &= np.isclose(dataframe[column].astype(float), float(value))
    return dataframe.loc[mask].copy()


def q25(series: pd.Series) -> float:
    return float(series.quantile(0.25))


def q75(series: pd.Series) -> float:
    return float(series.quantile(0.75))


def quantize_efficiency_series(series: pd.Series, bin_width: float | None) -> pd.Series:
    values = series.astype(float).to_numpy()
    if bin_width is None or float(bin_width) <= 0:
        return pd.Series(np.round(values, 6), index=series.index)
    width = float(bin_width)
    binned = np.round(values / width) * width
    binned = np.clip(binned, 0.0, 1.0)
    return pd.Series(np.round(binned, 6), index=series.index)


def assign_efficiency_bins(
    dataframe: pd.DataFrame,
    eff_columns: list[str],
    bin_width: float | None,
    *,
    suffix: str = "_bin",
) -> pd.DataFrame:
    out = dataframe.copy()
    for column in eff_columns:
        out[f"{column}{suffix}"] = quantize_efficiency_series(out[column], bin_width)
    return out


def assign_flux_bins(series: pd.Series, bin_count: int) -> tuple[pd.Series, np.ndarray]:
    codes, edges = pd.cut(
        series.astype(float),
        bins=int(bin_count),
        labels=False,
        retbins=True,
        include_lowest=True,
        duplicates="drop",
    )
    return pd.Series(codes, index=series.index, dtype="Int64"), np.asarray(edges, dtype=float)


def choose_reference_row(
    dataframe: pd.DataFrame,
    eff_columns: list[str],
) -> dict[str, Any]:
    work = dataframe.copy()
    work["distance_to_perfect"] = np.sqrt(
        np.sum((1.0 - work[eff_columns].astype(float).to_numpy()) ** 2, axis=1)
    )
    work["eff_sum"] = work[eff_columns].astype(float).sum(axis=1)
    work = work.sort_values(
        ["distance_to_perfect", "eff_sum", "n_flux_bins", "support_rows"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    if work.empty:
        raise ValueError("No efficiency bins available for reference selection.")
    return work.iloc[0].to_dict()


def write_ascii_lut(
    path: str | Path,
    z_vector: tuple[float, float, float, float],
    lut_dataframe: pd.DataFrame,
    *,
    trigger: Any = None,
    rate_column_name: str | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# z_positions: "
            + ", ".join(f"{float(value):.6f}" for value in z_vector)
            + "\n"
        )
        trigger_value = canonicalize_trigger_value(trigger)
        if trigger_value is None:
            trigger_text = "null"
        elif isinstance(trigger_value, str):
            trigger_text = trigger_value
        else:
            trigger_text = json.dumps(trigger_value)
        handle.write(f"# trigger: {trigger_text}\n")
        rate_text = "null" if rate_column_name in (None, "", "null", "None") else str(rate_column_name)
        handle.write(f"# rate_column: {rate_text}\n")
        lut_dataframe.to_csv(handle, sep=" ", index=False, float_format="%.6f")


def read_ascii_lut(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    in_path = Path(path)
    comments: list[str] = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            comments.append(line.rstrip("\n"))
    dataframe = pd.read_csv(in_path, comment="#", sep=r"\s+", engine="python")
    return dataframe, comments


def apply_lut_fallback_matches(
    dataframe: pd.DataFrame,
    lut_dataframe: pd.DataFrame,
    *,
    query_columns: list[str],
    raw_columns: list[str] | None = None,
    match_mode: str = "nearest",
    interpolation_k: int | None = None,
    interpolation_power: float = 2.0,
) -> pd.DataFrame:
    work = dataframe.copy()

    for column in CANONICAL_EFF_COLUMNS:
        lut_column = f"lut_{column}"
        if lut_column not in work.columns:
            work[lut_column] = np.nan

    if "lut_neighbor_count" not in work.columns:
        work["lut_neighbor_count"] = np.nan
    if "lut_neighbor_min_distance" not in work.columns:
        work["lut_neighbor_min_distance"] = np.nan
    if "lut_neighbor_max_distance" not in work.columns:
        work["lut_neighbor_max_distance"] = np.nan

    exact_mask = work["lut_scale_factor"].notna()
    work.loc[exact_mask, "lut_neighbor_count"] = 1.0
    work.loc[exact_mask, "lut_neighbor_min_distance"] = 0.0
    work.loc[exact_mask, "lut_neighbor_max_distance"] = 0.0

    missing_mask = work["lut_scale_factor"].isna()
    if not missing_mask.any() or match_mode == "exact":
        return work

    normalized_mode = str(match_mode).strip().lower()
    if normalized_mode not in {"nearest", "interpolate"}:
        raise ValueError(f"Unsupported LUT match mode: {match_mode!r}")

    if normalized_mode == "interpolate":
        if raw_columns is None:
            raise ValueError("raw_columns must be provided when using interpolated LUT matching.")
        distance_columns = list(raw_columns)
    else:
        distance_columns = list(query_columns)

    if normalized_mode == "interpolate" and float(interpolation_power) <= 0.0:
        raise ValueError("interpolation_power must be positive.")

    lut_matrix = lut_dataframe[CANONICAL_EFF_COLUMNS].to_numpy(dtype=float)
    scale_values = lut_dataframe["scale_factor"].to_numpy(dtype=float)
    query_matrix = work.loc[missing_mask, distance_columns].to_numpy(dtype=float)
    distances = np.sqrt(((query_matrix[:, None, :] - lut_matrix[None, :, :]) ** 2).sum(axis=2))

    best_indices = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(len(best_indices)), best_indices]
    anchor_rows = lut_dataframe.iloc[best_indices].reset_index(drop=True)

    if normalized_mode == "nearest":
        work.loc[missing_mask, "lut_scale_factor"] = anchor_rows["scale_factor"].to_numpy(dtype=float)
        work.loc[missing_mask, "lut_match_method"] = "nearest"
        work.loc[missing_mask, "lut_match_distance"] = best_distances
        work.loc[missing_mask, "lut_neighbor_count"] = 1.0
        work.loc[missing_mask, "lut_neighbor_min_distance"] = best_distances
        work.loc[missing_mask, "lut_neighbor_max_distance"] = best_distances
        for column in CANONICAL_EFF_COLUMNS:
            work.loc[missing_mask, f"lut_{column}"] = anchor_rows[column].to_numpy(dtype=float)
        return work

    neighbor_count = int(interpolation_k) if interpolation_k is not None else 8
    neighbor_count = max(1, min(neighbor_count, len(lut_dataframe)))
    if neighbor_count == 1:
        work.loc[missing_mask, "lut_scale_factor"] = anchor_rows["scale_factor"].to_numpy(dtype=float)
        work.loc[missing_mask, "lut_match_method"] = "nearest"
        work.loc[missing_mask, "lut_match_distance"] = best_distances
        work.loc[missing_mask, "lut_neighbor_count"] = 1.0
        work.loc[missing_mask, "lut_neighbor_min_distance"] = best_distances
        work.loc[missing_mask, "lut_neighbor_max_distance"] = best_distances
        for column in CANONICAL_EFF_COLUMNS:
            work.loc[missing_mask, f"lut_{column}"] = anchor_rows[column].to_numpy(dtype=float)
        return work

    neighbor_indices = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
    neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
    ordering = np.argsort(neighbor_distances, axis=1)
    neighbor_indices = np.take_along_axis(neighbor_indices, ordering, axis=1)
    neighbor_distances = np.take_along_axis(neighbor_distances, ordering, axis=1)
    neighbor_scales = scale_values[neighbor_indices]
    weights = 1.0 / np.maximum(neighbor_distances, 1e-9) ** float(interpolation_power)
    interpolated_scale = (weights * neighbor_scales).sum(axis=1) / weights.sum(axis=1)

    work.loc[missing_mask, "lut_scale_factor"] = interpolated_scale
    work.loc[missing_mask, "lut_match_method"] = "interpolated"
    work.loc[missing_mask, "lut_match_distance"] = best_distances
    work.loc[missing_mask, "lut_neighbor_count"] = float(neighbor_count)
    work.loc[missing_mask, "lut_neighbor_min_distance"] = neighbor_distances[:, 0]
    work.loc[missing_mask, "lut_neighbor_max_distance"] = neighbor_distances[:, -1]
    for column in CANONICAL_EFF_COLUMNS:
        work.loc[missing_mask, f"lut_{column}"] = anchor_rows[column].to_numpy(dtype=float)
    return work
