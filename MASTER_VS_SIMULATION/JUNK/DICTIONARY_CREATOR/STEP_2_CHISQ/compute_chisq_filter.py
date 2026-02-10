#!/usr/bin/env python3
"""Compute chi-square scores between station metadata and dictionary rows."""
# See config/COLUMN_REFERENCE_TASKS.md for task-specific column lists by type.

# COLUMN REFERENCE (from TASK_1 metadata headers; expected same structure in TASK_2-5)
# RAW_TT_COUNT:
#   raw_tt_1234_count, raw_tt_123_count, raw_tt_124_count, raw_tt_12_count, raw_tt_134_count,
#   raw_tt_13_count, raw_tt_14_count, raw_tt_1_count, raw_tt_234_count, raw_tt_23_count,
#   raw_tt_24_count, raw_tt_2_count, raw_tt_34_count, raw_tt_3_count, raw_tt_4_count
# CLEAN_TT_COUNT:
#   clean_tt_1234_count, clean_tt_123_count, clean_tt_124_count, clean_tt_12_count,
#   clean_tt_134_count, clean_tt_13_count, clean_tt_14_count, clean_tt_1_count,
#   clean_tt_234_count, clean_tt_23_count, clean_tt_24_count, clean_tt_2_count,
#   clean_tt_34_count, clean_tt_3_count, clean_tt_4_count
# RAW_TO_CLEAN_TT_COUNT:
#   raw_to_clean_tt_1234_1234_count, raw_to_clean_tt_1234_123_count, raw_to_clean_tt_1234_124_count,
#   raw_to_clean_tt_1234_12_count, raw_to_clean_tt_1234_134_count, raw_to_clean_tt_1234_13_count,
#   raw_to_clean_tt_1234_14_count, raw_to_clean_tt_1234_1_count, raw_to_clean_tt_1234_234_count,
#   raw_to_clean_tt_1234_23_count, raw_to_clean_tt_1234_24_count, raw_to_clean_tt_1234_2_count,
#   raw_to_clean_tt_1234_34_count, raw_to_clean_tt_1234_3_count, raw_to_clean_tt_1234_4_count,
#   raw_to_clean_tt_123_123_count, raw_to_clean_tt_123_12_count, raw_to_clean_tt_123_13_count,
#   raw_to_clean_tt_123_1_count, raw_to_clean_tt_123_23_count, raw_to_clean_tt_123_2_count,
#   raw_to_clean_tt_123_3_count, raw_to_clean_tt_124_124_count, raw_to_clean_tt_124_12_count,
#   raw_to_clean_tt_124_14_count, raw_to_clean_tt_124_1_count, raw_to_clean_tt_124_24_count,
#   raw_to_clean_tt_124_2_count, raw_to_clean_tt_124_4_count, raw_to_clean_tt_12_12_count,
#   raw_to_clean_tt_12_1_count, raw_to_clean_tt_12_2_count, raw_to_clean_tt_134_134_count,
#   raw_to_clean_tt_134_13_count, raw_to_clean_tt_134_14_count, raw_to_clean_tt_134_1_count,
#   raw_to_clean_tt_134_34_count, raw_to_clean_tt_134_3_count, raw_to_clean_tt_134_4_count,
#   raw_to_clean_tt_13_13_count, raw_to_clean_tt_13_1_count, raw_to_clean_tt_13_3_count,
#   raw_to_clean_tt_14_14_count, raw_to_clean_tt_14_1_count, raw_to_clean_tt_14_4_count,
#   raw_to_clean_tt_1_1_count, raw_to_clean_tt_234_234_count, raw_to_clean_tt_234_23_count,
#   raw_to_clean_tt_234_24_count, raw_to_clean_tt_234_2_count, raw_to_clean_tt_234_34_count,
#   raw_to_clean_tt_234_3_count, raw_to_clean_tt_234_4_count, raw_to_clean_tt_23_23_count,
#   raw_to_clean_tt_23_2_count, raw_to_clean_tt_23_3_count, raw_to_clean_tt_24_24_count,
#   raw_to_clean_tt_24_2_count, raw_to_clean_tt_24_4_count, raw_to_clean_tt_2_2_count,
#   raw_to_clean_tt_34_34_count, raw_to_clean_tt_34_3_count, raw_to_clean_tt_34_4_count,
#   raw_to_clean_tt_3_3_count
# Q_ENTRIES_ORIGINAL:
#   Q1_B_1_entries_original, Q1_B_2_entries_original, Q1_B_3_entries_original, Q1_B_4_entries_original,
#   Q1_F_1_entries_original, Q1_F_2_entries_original, Q1_F_3_entries_original, Q1_F_4_entries_original,
#   Q2_B_1_entries_original, Q2_B_2_entries_original, Q2_B_3_entries_original, Q2_B_4_entries_original,
#   Q2_F_1_entries_original, Q2_F_2_entries_original, Q2_F_3_entries_original, Q2_F_4_entries_original,
#   Q3_B_1_entries_original, Q3_B_2_entries_original, Q3_B_3_entries_original, Q3_B_4_entries_original,
#   Q3_F_1_entries_original, Q3_F_2_entries_original, Q3_F_3_entries_original, Q3_F_4_entries_original,
#   Q4_B_1_entries_original, Q4_B_2_entries_original, Q4_B_3_entries_original, Q4_B_4_entries_original,
#   Q4_F_1_entries_original, Q4_F_2_entries_original, Q4_F_3_entries_original, Q4_F_4_entries_original
# Q_ENTRIES_FINAL:
#   Q1_B_1_entries_final, Q1_B_2_entries_final, Q1_B_3_entries_final, Q1_B_4_entries_final,
#   Q1_F_1_entries_final, Q1_F_2_entries_final, Q1_F_3_entries_final, Q1_F_4_entries_final,
#   Q2_B_1_entries_final, Q2_B_2_entries_final, Q2_B_3_entries_final, Q2_B_4_entries_final,
#   Q2_F_1_entries_final, Q2_F_2_entries_final, Q2_F_3_entries_final, Q2_F_4_entries_final,
#   Q3_B_1_entries_final, Q3_B_2_entries_final, Q3_B_3_entries_final, Q3_B_4_entries_final,
#   Q3_F_1_entries_final, Q3_F_2_entries_final, Q3_F_3_entries_final, Q3_F_4_entries_final,
#   Q4_B_1_entries_final, Q4_B_2_entries_final, Q4_B_3_entries_final, Q4_B_4_entries_final,
#   Q4_F_1_entries_final, Q4_F_2_entries_final, Q4_F_3_entries_final, Q4_F_4_entries_final
# T_ENTRIES_ORIGINAL:
#   T1_B_1_entries_original, T1_B_2_entries_original, T1_B_3_entries_original, T1_B_4_entries_original,
#   T1_F_1_entries_original, T1_F_2_entries_original, T1_F_3_entries_original, T1_F_4_entries_original,
#   T2_B_1_entries_original, T2_B_2_entries_original, T2_B_3_entries_original, T2_B_4_entries_original,
#   T2_F_1_entries_original, T2_F_2_entries_original, T2_F_3_entries_original, T2_F_4_entries_original,
#   T3_B_1_entries_original, T3_B_2_entries_original, T3_B_3_entries_original, T3_B_4_entries_original,
#   T3_F_1_entries_original, T3_F_2_entries_original, T3_F_3_entries_original, T3_F_4_entries_original,
#   T4_B_1_entries_original, T4_B_2_entries_original, T4_B_3_entries_original, T4_B_4_entries_original,
#   T4_F_1_entries_original, T4_F_2_entries_original, T4_F_3_entries_original, T4_F_4_entries_original
# T_ENTRIES_FINAL:
#   T1_B_1_entries_final, T1_B_2_entries_final, T1_B_3_entries_final, T1_B_4_entries_final,
#   T1_F_1_entries_final, T1_F_2_entries_final, T1_F_3_entries_final, T1_F_4_entries_final,
#   T2_B_1_entries_final, T2_B_2_entries_final, T2_B_3_entries_final, T2_B_4_entries_final,
#   T2_F_1_entries_final, T2_F_2_entries_final, T2_F_3_entries_final, T2_F_4_entries_final,
#   T3_B_1_entries_final, T3_B_2_entries_final, T3_B_3_entries_final, T3_B_4_entries_final,
#   T3_F_1_entries_final, T3_F_2_entries_final, T3_F_3_entries_final, T3_F_4_entries_final,
#   T4_B_1_entries_final, T4_B_2_entries_final, T4_B_3_entries_final, T4_B_4_entries_final,
#   T4_F_1_entries_final, T4_F_2_entries_final, T4_F_3_entries_final, T4_F_4_entries_final
# OTHER:
#   analysis_mode, execution_timestamp, filename_base, valid_lines_in_binary_file_percentage,
#   zeroed_percentage_P1s1, zeroed_percentage_P1s2, zeroed_percentage_P1s3, zeroed_percentage_P1s4,
#   zeroed_percentage_P2s1, zeroed_percentage_P2s2, zeroed_percentage_P2s3, zeroed_percentage_P2s4,
#   zeroed_percentage_P3s1, zeroed_percentage_P3s2, zeroed_percentage_P3s3, zeroed_percentage_P3s4,
#   zeroed_percentage_P4s1, zeroed_percentage_P4s2, zeroed_percentage_P4s3, zeroed_percentage_P4s4

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from STEP_3_PLOTS.plot_station_metadata_vs_dictionary import (  # noqa: E402
    _apply_median_filter,
    _load_config,
    _select_columns_by_group,
    _sort_q_columns,
    _task_metadata_path,
)

DEFAULT_CONFIG = BASE_DIR / "config/pipeline_config.json"
DEFAULT_DICT = BASE_DIR / "STEP_1_BUILD/param_metadata_dictionary.csv"


def _parse_datetime(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, format="%Y-%m-%d_%H.%M.%S", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Could not parse date/time: {value}")
    return ts


def _parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("Range must be in start:end format.")
    return int(parts[0]), int(parts[1])


def _filter_param_ranges(df: pd.DataFrame, ranges: list[list[str]]) -> pd.DataFrame:
    for entry in ranges:
        if len(entry) != 3:
            raise ValueError("--param-range expects: COL MIN MAX")
        col, min_val, max_val = entry
        if col not in df.columns:
            raise ValueError(f"Param column '{col}' not found in dictionary.")
        series = pd.to_numeric(df[col], errors="coerce")
        df = df[series.between(float(min_val), float(max_val))]
    return df


def _load_config_or_fail(path: Path) -> dict:
    try:
        return _load_config(path)
    except Exception as exc:
        raise SystemExit(f"Failed to load config: {exc}")


def _select_chisq_columns(meta_cols: list[str], config: dict) -> list[str]:
    group_names = config.get("chisq_groups")
    if group_names is None:
        group_names = config.get("chisq_group")
    if isinstance(group_names, str):
        group_names = [group_names]
    if not group_names:
        group_names = ["raw_tt"]

    groups = config.get("groups") or []
    selected: list[str] = []
    group_defs = config.get("group_definitions") or {}
    for group_name in group_names:
        group = None
        for entry in groups:
            if entry.get("name") == group_name:
                group = entry
                break
        if group is None:
            if group_name in group_defs:
                group = {"name": group_name, "regex": group_defs[group_name]}
            elif group_name == "Q_entries_original":
                group = {
                    "name": "Q_entries_original",
                    "regex": r"^Q[1-4]_[FB]_[1-4]_entries_original$",
                }
            elif group_name == "Q_entries_final":
                group = {
                    "name": "Q_entries_final",
                    "regex": r"^Q[1-4]_[FB]_[1-4]_entries_final$",
                }
            elif group_name == "T_entries_original":
                group = {
                    "name": "T_entries_original",
                    "regex": r"^T[1-4]_[FB]_[1-4]_entries_original$",
                }
            elif group_name == "T_entries_final":
                group = {
                    "name": "T_entries_final",
                    "regex": r"^T[1-4]_[FB]_[1-4]_entries_final$",
                }
            elif group_name == "clean_tt":
                group = {"name": "clean_tt", "prefix": "clean_tt_"}
            elif group_name == "raw_to_clean_tt":
                group = {"name": "raw_to_clean_tt", "prefix": "raw_to_clean_tt_"}
            elif group_name == "tt_count_any":
                group = {"name": "tt_count_any", "regex": r".*_tt_.*_count$|^fit_tt_.*_count$|^corr_tt_.*_count$"}
            else:
                group = {"name": "raw_tt", "prefix": config.get("column_prefix", "raw_tt_")}

        columns = _select_columns_by_group(
            meta_cols, group, config.get("column_prefix", "raw_tt_")
        )
        if group.get("name") == "Q_entries_original":
            columns = _sort_q_columns(columns)
        selected.extend(columns)

    # de-dup while preserving order
    return list(dict.fromkeys(selected))


def _apply_task_overrides(config: dict, task_id: int) -> dict:
    overrides = (config.get("task_settings") or {}).get(str(task_id))
    if not overrides:
        return config
    merged = dict(config)
    merged.update(overrides)
    return merged


def _compute_loss_pointwise(
    data: pd.DataFrame,
    dict_row: pd.Series,
    columns: list[str],
    loss: str,
    total_series: pd.Series | None,
    total_ref: float | None,
    tt_cols_set: set[str] | None,
) -> tuple[float, int]:
    total = 0.0
    count = 0
    if total_series is not None:
        denom = total_series.where(total_series > 0)
    else:
        denom = None
    for col in columns:
        if col not in data.columns:
            continue
        series = pd.to_numeric(data[col], errors="coerce")
        if series.empty:
            continue
        mu = dict_row.get(col)
        if pd.isna(mu):
            continue
        if (
            denom is not None
            and total_ref is not None
            and total_ref > 0
            and tt_cols_set is not None
            and col in tt_cols_set
        ):
            series = series / denom
            mu = float(mu) / total_ref
        valid = series.notna()
        if not valid.any():
            continue
        diff = series[valid] - float(mu)
        if loss == "mae":
            total += float(np.sum(np.abs(diff)))
        elif loss == "l2":
            total += float(np.sum(diff * diff))
        else:
            sigma_data = float(series.std(ddof=0))
            sigma = np.sqrt(np.square(sigma_data) + np.maximum(series, 0))
            good = valid & np.isfinite(sigma) & (sigma > 0)
            if not good.any():
                continue
            diff = series[good] - float(mu)
            total += float(np.sum((diff / sigma[good]) ** 2))
        count += int(valid.sum())
    return total, count


def _aggregate_columns(
    data: pd.DataFrame, columns: list[str], stat: str
) -> dict[str, float]:
    agg: dict[str, float] = {}
    for col in columns:
        if col not in data.columns:
            continue
        series = pd.to_numeric(data[col], errors="coerce").dropna()
        if series.empty:
            continue
        if stat == "median":
            agg[col] = float(series.median())
        elif stat == "mean":
            agg[col] = float(series.mean())
        else:
            raise ValueError(f"Unknown chisq_stat: {stat}")
    return agg


def _compute_loss_aggregate(
    agg: dict[str, float],
    dict_row: pd.Series,
    columns: list[str],
    loss: str,
    total_ref: float | None,
    agg_normalized: bool,
    tt_cols_set: set[str] | None,
) -> tuple[float, int]:
    total = 0.0
    count = 0
    for col in columns:
        if col not in agg:
            continue
        mu = dict_row.get(col)
        if pd.isna(mu):
            continue
        mu_val = float(mu)
        agg_val = agg[col]
        if (
            total_ref is not None
            and total_ref > 0
            and tt_cols_set is not None
            and col in tt_cols_set
        ):
            mu_val = mu_val / total_ref
            if not agg_normalized:
                agg_val = agg_val / total_ref
        diff = agg_val - mu_val
        if loss == "mae":
            total += abs(diff)
        else:
            total += diff * diff
        count += 1
    return total, count


def _find_tt_columns(columns: list[str]) -> list[str]:
    import re

    pattern = re.compile(
        r".*_tt_.*_count$|^fit_tt_.*_count$|^corr_tt_.*_count$|^cal_tt_.*_count$|^list_tt_.*_count$"
    )
    return [c for c in columns if pattern.match(c)]


def _compute_total_series(
    df: pd.DataFrame, total_events_col: str | None, norm_cols: list[str]
) -> pd.Series | None:
    if total_events_col and total_events_col in df.columns:
        return pd.to_numeric(df[total_events_col], errors="coerce")
    if norm_cols:
        return (
            df[norm_cols]
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1, min_count=1)
        )
    return None


def _compute_total_ref(
    row: pd.Series, total_events_col: str | None, norm_cols: list[str]
) -> float | None:
    if total_events_col and total_events_col in row:
        val = pd.to_numeric(row[total_events_col], errors="coerce")
        return float(val) if not pd.isna(val) else None
    if norm_cols:
        series = pd.to_numeric(row[norm_cols], errors="coerce")
        total = series.sum(min_count=1)
        return float(total) if not pd.isna(total) else None
    return None


def _find_tt_prefixes(columns: list[str]) -> list[str]:
    import re

    prefixes = set()
    for col in columns:
        match = re.match(r"^(.*)_tt_1234_count$", col)
        if match:
            prefixes.add(match.group(1))
    return sorted(prefixes)


def _add_eff_quick_columns(df: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    df = df.copy()
    for prefix in prefixes:
        required = {
            f"eff_quick_{prefix}_1": (f"{prefix}_tt_234_count", f"{prefix}_tt_1234_count"),
            f"eff_quick_{prefix}_2": (f"{prefix}_tt_134_count", f"{prefix}_tt_1234_count"),
            f"eff_quick_{prefix}_3": (f"{prefix}_tt_124_count", f"{prefix}_tt_1234_count"),
            f"eff_quick_{prefix}_4": (f"{prefix}_tt_123_count", f"{prefix}_tt_1234_count"),
        }
        for name, (num_col, den_col) in required.items():
            if num_col in df.columns and den_col in df.columns:
                num = pd.to_numeric(df[num_col], errors="coerce")
                den = pd.to_numeric(df[den_col], errors="coerce")
                ratio = num / den.replace({0: np.nan})
                df[name] = 1 - ratio
    return df


def _aggregate_columns_normalized(
    data: pd.DataFrame,
    columns: list[str],
    stat: str,
    total_series: pd.Series | None,
    tt_cols_set: set[str] | None,
) -> dict[str, float]:
    agg: dict[str, float] = {}
    denom = total_series.where(total_series > 0) if total_series is not None else None
    for col in columns:
        if col not in data.columns:
            continue
        series = pd.to_numeric(data[col], errors="coerce")
        if denom is not None and tt_cols_set is not None and col in tt_cols_set:
            series = series / denom
        series = series.dropna()
        if series.empty:
            continue
        if stat == "median":
            agg[col] = float(series.median())
        elif stat == "mean":
            agg[col] = float(series.mean())
        else:
            raise ValueError(f"Unknown chisq_stat: {stat}")
    return agg


def _load_station_config(station_id: int, config_dir: str) -> pd.DataFrame:
    path = Path(config_dir) / f"STATION_{station_id}" / f"input_file_mingo{station_id:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Station config not found: {path}")
    df = pd.read_csv(path, header=1)
    df.columns = [c.strip() for c in df.columns]
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    return df


def _assign_z_positions(meta_df: pd.DataFrame, cfg_df: pd.DataFrame) -> pd.DataFrame:
    meta_df = meta_df.copy()
    meta_df["z_p1"] = np.nan
    meta_df["z_p2"] = np.nan
    meta_df["z_p3"] = np.nan
    meta_df["z_p4"] = np.nan
    for _, row in cfg_df.iterrows():
        start = row["start"]
        end = row["end"]
        if pd.isna(start):
            continue
        if pd.isna(end):
            mask = meta_df["execution_dt"] >= start
        else:
            mask = (meta_df["execution_dt"] >= start) & (meta_df["execution_dt"] <= end)
        meta_df.loc[mask, "z_p1"] = row.get("P1")
        meta_df.loc[mask, "z_p2"] = row.get("P2")
        meta_df.loc[mask, "z_p3"] = row.get("P3")
        meta_df.loc[mask, "z_p4"] = row.get("P4")
    return meta_df


def _filter_dict_by_z(dict_df: pd.DataFrame, z_tuple: tuple[float, float, float, float], tol: float) -> pd.DataFrame:
    z1, z2, z3, z4 = z_tuple
    mask = (
        np.isclose(dict_df["z_plane_1"].astype(float), z1, atol=tol, rtol=0)
        & np.isclose(dict_df["z_plane_2"].astype(float), z2, atol=tol, rtol=0)
        & np.isclose(dict_df["z_plane_3"].astype(float), z3, atol=tol, rtol=0)
        & np.isclose(dict_df["z_plane_4"].astype(float), z4, atol=tol, rtol=0)
    )
    return dict_df.loc[mask]


def _parse_efficiencies(series: pd.Series) -> pd.DataFrame:
    import ast

    effs = {f"eff{i}": [] for i in range(1, 5)}
    for raw in series:
        values = raw
        if isinstance(raw, str):
            try:
                values = ast.literal_eval(raw)
            except Exception:
                values = None
        if isinstance(values, (list, tuple)) and len(values) >= 4:
            for i in range(4):
                effs[f"eff{i+1}"].append(float(values[i]))
        else:
            for i in range(4):
                effs[f"eff{i+1}"].append(np.nan)
    return pd.DataFrame(effs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute chi-square scores between station data and dictionary rows."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to pipeline_config.json",
    )
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--chisq-threshold", type=float, default=None)
    parser.add_argument("--hist-out", default=None)
    parser.add_argument("--hist-bins", type=int, default=None)
    parser.add_argument(
        "--loss",
        default=None,
        choices=["chisq", "mae", "l2"],
        help="Loss function for comparison (chisq, mae, or l2).",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config_or_fail(config_path)

    task_id = args.task_id or int(config.get("task_id", 1))
    config = _apply_task_overrides(config, task_id)

    station_id = int(config.get("station_id", 1))
    metadata_path = (
        Path(args.metadata_csv)
        if args.metadata_csv
        else _task_metadata_path(station_id, task_id)
    )
    if args.dictionary_csv:
        dict_path = Path(args.dictionary_csv)
    else:
        dict_template = config.get("dictionary_csv", str(DEFAULT_DICT))
        if "{task_id" in dict_template:
            dict_template = dict_template.format(task_id=task_id)
        dict_path = Path(dict_template)

    if not metadata_path.exists():
        print(f"Metadata CSV not found: {metadata_path}", file=sys.stderr)
        return 1
    if not dict_path.exists():
        print(f"Dictionary CSV not found: {dict_path}", file=sys.stderr)
        return 1

    meta_df = pd.read_csv(metadata_path, low_memory=False)
    meta_df["execution_dt"] = pd.to_datetime(
        meta_df["execution_timestamp"],
        format="%Y-%m-%d_%H.%M.%S",
        errors="coerce",
    )
    meta_df = meta_df.dropna(subset=["execution_dt"]).sort_values("execution_dt")

    start_dt = _parse_datetime(config.get("start"))
    end_dt = _parse_datetime(config.get("end"))
    if start_dt is not None:
        meta_df = meta_df[meta_df["execution_dt"] >= start_dt]
    if end_dt is not None:
        meta_df = meta_df[meta_df["execution_dt"] <= end_dt]

    dict_df = pd.read_csv(dict_path, low_memory=False)
    dict_df = dict_df.copy()
    dict_df["param_set_id"] = pd.to_numeric(dict_df["param_set_id"], errors="coerce")
    dict_df = dict_df.dropna(subset=["param_set_id"])
    dict_df["param_set_id"] = dict_df["param_set_id"].astype(int)

    prefixes = sorted(
        set(_find_tt_prefixes(meta_df.columns.tolist()))
        | set(_find_tt_prefixes(dict_df.columns.tolist()))
    )
    if prefixes:
        meta_df = _add_eff_quick_columns(meta_df, prefixes)
        dict_df = _add_eff_quick_columns(dict_df, prefixes)

    if config.get("param_set_ids"):
        ids = _parse_int_list(config.get("param_set_ids"))
        dict_df = dict_df[dict_df["param_set_id"].isin(ids)]

    if config.get("param_set_range"):
        range_vals = _parse_range(config.get("param_set_range"))
        if range_vals:
            start_id, end_id = range_vals
            dict_df = dict_df[
                (dict_df["param_set_id"] >= start_id)
                & (dict_df["param_set_id"] <= end_id)
            ]

    if config.get("param_range"):
        dict_df = _filter_param_ranges(dict_df, config.get("param_range"))

    if dict_df.empty:
        print("No parameter sets selected after filtering.", file=sys.stderr)
        return 1

    prefixes = sorted(
        set(_find_tt_prefixes(meta_df.columns.tolist()))
        | set(_find_tt_prefixes(dict_df.columns.tolist()))
    )
    if prefixes:
        meta_df = _add_eff_quick_columns(meta_df, prefixes)
        dict_df = _add_eff_quick_columns(dict_df, prefixes)

    columns = _select_chisq_columns(meta_df.columns.tolist(), config)
    if not columns:
        print("No columns selected for chi-square.", file=sys.stderr)
        return 1

    median_window = int(config.get("median_window", 11))
    meta_df = _apply_median_filter(meta_df, columns, median_window)

    loss = args.loss or config.get("chisq_loss", "chisq")
    stat = config.get("chisq_stat", "point")
    if stat not in ("point", "median", "mean"):
        raise ValueError(f"Unsupported chisq_stat: {stat}")
    use_z_match = bool(config.get("z_match_enabled", True))
    z_tol = float(config.get("z_match_tolerance", 1e-6))
    normalize = bool(config.get("chisq_normalize", False))
    total_events_col = config.get("total_events_column")
    tt_cols = _find_tt_columns(meta_df.columns.tolist())
    tt_cols_set = set(tt_cols)
    norm_cols = [c for c in columns if c in tt_cols_set]
    use_total_events_col = (
        total_events_col
        if total_events_col
        and total_events_col in meta_df.columns
        and total_events_col in dict_df.columns
        else None
    )
    dict_norm_cols = [c for c in norm_cols if c in dict_df.columns]
    if use_z_match:
        config_dir = config.get(
            "station_config_dir",
            "/home/mingo/DATAFLOW_v3/MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY",
        )
        cfg_df = _load_station_config(station_id, config_dir)
        meta_df = _assign_z_positions(meta_df, cfg_df)
        meta_df = meta_df.dropna(subset=["z_p1", "z_p2", "z_p3", "z_p4"])
        print(
            f"[DEBUG] meta rows after z-assign: {len(meta_df)}; unique z tuples: "
            f"{meta_df[['z_p1','z_p2','z_p3','z_p4']].drop_duplicates().shape[0]}"
        )
        print(
            f"[DEBUG] dict rows: {len(dict_df)}; z_plane_1-4 present: "
            f"{all(c in dict_df.columns for c in ['z_plane_1','z_plane_2','z_plane_3','z_plane_4'])}"
        )
        print(f"[DEBUG] selected columns: {len(columns)}")
        if normalize:
            print(
                "[DEBUG] normalization enabled; "
                f"total_events_column={use_total_events_col}; "
                f"norm_cols={len(norm_cols)}"
            )

    total_loss = np.zeros(len(dict_df), dtype=float)
    total_count = np.zeros(len(dict_df), dtype=int)

    if use_z_match and not meta_df.empty:
        grouped = meta_df.groupby(["z_p1", "z_p2", "z_p3", "z_p4"])
        for (z1, z2, z3, z4), group_df in grouped:
            sub = _filter_dict_by_z(dict_df, (z1, z2, z3, z4), z_tol)
            if sub.empty:
                print(
                    f"[DEBUG] z tuple {z1,z2,z3,z4} has {len(group_df)} rows, "
                    "no matching dict rows"
                )
                continue
            if normalize:
                total_series = _compute_total_series(group_df, use_total_events_col, norm_cols)
            else:
                total_series = None
            if normalize and total_series is not None:
                valid_total_rows = int((total_series > 0).sum())
                print(
                    f"[DEBUG] z tuple {z1,z2,z3,z4} rows={len(group_df)} "
                    f"valid totals={valid_total_rows}"
                )
            if stat != "point":
                if normalize:
                    agg = _aggregate_columns_normalized(
                        group_df, columns, stat, total_series, tt_cols_set
                    )
                else:
                    agg = _aggregate_columns(group_df, columns, stat)
            for idx, row in sub.iterrows():
                if normalize and total_series is not None:
                    total_ref = _compute_total_ref(row, use_total_events_col, dict_norm_cols)
                else:
                    total_ref = None
                if normalize and total_ref is not None and total_ref <= 0:
                    continue
                if stat == "point":
                    loss_sum, cnt = _compute_loss_pointwise(
                        group_df, row, columns, loss, total_series, total_ref, tt_cols_set
                    )
                else:
                    loss_sum, cnt = _compute_loss_aggregate(
                        agg, row, columns, loss, total_ref, normalize, tt_cols_set
                    )
                if cnt == 0:
                    continue
                total_loss[idx] += loss_sum
                total_count[idx] += cnt
    else:
        if normalize:
            total_series = _compute_total_series(meta_df, use_total_events_col, norm_cols)
        else:
            total_series = None
        if stat != "point":
            if normalize:
                agg = _aggregate_columns_normalized(
                    meta_df, columns, stat, total_series, tt_cols_set
                )
            else:
                agg = _aggregate_columns(meta_df, columns, stat)
        for idx, row in dict_df.iterrows():
            if normalize and total_series is not None:
                total_ref = _compute_total_ref(row, use_total_events_col, dict_norm_cols)
            else:
                total_ref = None
            if normalize and total_ref is not None and total_ref <= 0:
                continue
            if stat == "point":
                loss_sum, cnt = _compute_loss_pointwise(
                    meta_df, row, columns, loss, total_series, total_ref, tt_cols_set
                )
            else:
                loss_sum, cnt = _compute_loss_aggregate(
                    agg, row, columns, loss, total_ref, normalize, tt_cols_set
                )
            if cnt == 0:
                continue
            total_loss[idx] += loss_sum
            total_count[idx] += cnt

    valid_counts = int((total_count > 0).sum())
    print(f"[DEBUG] dict rows with cnt>0: {valid_counts}/{len(dict_df)}")
    chisq_values = []
    for loss_sum, cnt in zip(total_loss, total_count):
        if cnt == 0:
            chisq_values.append(float("inf"))
        else:
            chisq_values.append(loss_sum / cnt)

    result = dict_df.copy()
    result["chisq"] = chisq_values
    result = result.sort_values("chisq")

    threshold = args.chisq_threshold
    if threshold is None:
        threshold = config.get("chisq_threshold")
    if threshold is not None:
        threshold = float(threshold)

    base_out_dir = Path(
        config.get("chisq_out_dir", str(BASE_DIR / "STEP_2_CHISQ/output"))
    ) / f"task_{task_id:02d}"
    out_path = Path(args.out) if args.out else Path(
        config.get(
            "chisq_output_csv",
            str(base_out_dir / "chisq_results.csv"),
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    hist_out = args.hist_out or config.get(
        "chisq_histogram_path",
        str(base_out_dir / "chisq_histogram.png"),
    )
    hist_bins = args.hist_bins or int(config.get("chisq_histogram_bins", 150))
    hist_out_path = Path(hist_out)
    hist_out_path.parent.mkdir(parents=True, exist_ok=True)

    finite = result["chisq"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(finite, bins=hist_bins, color="#4c78a8", alpha=0.8)
        if threshold is not None:
            plt.axvline(threshold, color="red", linewidth=1.5, linestyle="--")
        plt.title("Chi-square distribution")
        plt.xlabel("Chi-square")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(hist_out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote chi-square histogram: {hist_out_path}")
    else:
        print("No finite chi-square values to plot histogram.")

    if threshold is not None and not result.empty:
        good_mask = result["chisq"] <= threshold
        bad_mask = ~good_mask

        def hist_two_sets(values_good: pd.Series, values_bad: pd.Series, title: str, out_path: Path) -> None:
            values_good = values_good.replace([np.inf, -np.inf], np.nan).dropna()
            values_bad = values_bad.replace([np.inf, -np.inf], np.nan).dropna()
            if values_good.empty and values_bad.empty:
                return
            plt.figure(figsize=(6, 4))
            if not values_bad.empty:
                plt.hist(values_bad, bins=40, color="#9ecae1", alpha=0.7, label="rest")
            if not values_good.empty:
                plt.hist(values_good, bins=40, color="#f28e2b", alpha=0.8, label="chisq <= threshold")
            plt.title(title)
            plt.xlabel(title)
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()

        base_dir = Path(
            config.get(
                "chisq_param_hist_dir",
                str(base_out_dir / "chisq_param_hists"),
            )
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        hist_two_sets(
            pd.to_numeric(result.loc[good_mask, "cos_n"], errors="coerce"),
            pd.to_numeric(result.loc[bad_mask, "cos_n"], errors="coerce"),
            "cos_n",
            base_dir / "hist_cos_n.png",
        )
        hist_two_sets(
            pd.to_numeric(result.loc[good_mask, "flux_cm2_min"], errors="coerce"),
            pd.to_numeric(result.loc[bad_mask, "flux_cm2_min"], errors="coerce"),
            "flux_cm2_min",
            base_dir / "hist_flux_cm2_min.png",
        )

        eff_df = _parse_efficiencies(result["efficiencies"])
        for i in range(1, 5):
            col = f"eff{i}"
            hist_two_sets(
                eff_df.loc[good_mask, col],
                eff_df.loc[bad_mask, col],
                col,
                base_dir / f"hist_{col}.png",
            )

        print(f"Wrote parameter histograms to {base_dir}")

    print(f"Wrote chi-square results: {out_path} (rows={len(result)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
