#!/usr/bin/env python3
"""Plot station metadata raw_tt_* counts with parameter-set reference lines."""
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
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE_DIR / "config/pipeline_config.json"
DEFAULT_DICT = BASE_DIR / "STEP_1_BUILD/param_metadata_dictionary.csv"


def _task_metadata_path(station_id: int, task_id: int) -> Path:
    station = f"MINGO{station_id:02d}"
    return Path(
        "/home/mingo/DATAFLOW_v3/STATIONS"
        f"/{station}/STAGE_1/EVENT_DATA/STEP_1/TASK_{task_id}/METADATA/"
        f"task_{task_id}_metadata_specific.csv"
    )


def _station_metadata_path(station_id: int) -> Path:
    return _task_metadata_path(station_id, 1)


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


def _load_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object.")
    return data


def _apply_task_overrides(config: dict, task_id: int) -> dict:
    overrides = (config.get("task_settings") or {}).get(str(task_id))
    if not overrides:
        return config
    merged = dict(config)
    merged.update(overrides)
    return merged


def _merge_config(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if key == "param_range":
            if value:
                current = current or []
                current.extend(value)
                setattr(args, key, current)
        elif current is None:
            setattr(args, key, value)
    return args


def _select_columns_by_group(meta_cols: list[str], group: dict, default_prefix: str) -> list[str]:
    if "columns" in group and group["columns"]:
        selected = [c for c in group["columns"] if c in meta_cols]
    elif "prefix" in group and group["prefix"]:
        selected = [c for c in meta_cols if c.startswith(group["prefix"])]
    elif "regex" in group and group["regex"]:
        import re

        pattern = re.compile(group["regex"])
        selected = [c for c in meta_cols if pattern.match(c)]
    elif default_prefix:
        selected = [c for c in meta_cols if c.startswith(default_prefix)]
    else:
        selected = []

    excludes = group.get("exclude") or []
    if excludes:
        selected = [c for c in selected if c not in excludes]
    return selected


def _sort_q_columns(columns: list[str]) -> list[str]:
    import re

    def key(col: str) -> tuple[int, int, int]:
        match = re.match(r"^Q(\\d)_(F|B)_(\\d)_entries_original$", col)
        if not match:
            return (9, 9, 9)
        plane = int(match.group(1))
        side = 0 if match.group(2) == "F" else 1
        strip = int(match.group(3))
        return (plane, side, strip)

    return sorted(columns, key=key)


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


def _apply_median_filter(df: pd.DataFrame, columns: list[str], window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    filtered = df.copy()
    data = (
        filtered[columns]
        .apply(pd.to_numeric, errors="coerce")
        .rolling(window=window, center=True, min_periods=1)
        .median()
    )
    filtered[columns] = data
    return filtered


def _format_legend_label(row: pd.Series, keys: list[str]) -> str:
    import ast

    parts: list[str] = []
    for key in keys:
        if key == "efficiencies":
            raw = row.get("efficiencies")
            if pd.isna(raw):
                continue
            values = raw
            if isinstance(raw, str):
                try:
                    values = ast.literal_eval(raw)
                except Exception:
                    values = None
            if isinstance(values, (list, tuple)) and len(values) >= 4:
                for idx, val in enumerate(values[:4], start=1):
                    parts.append(f"eff{idx}={float(val):.4g}")
            continue

        val = row.get(key)
        if pd.isna(val):
            continue
        if isinstance(val, float):
            parts.append(f"{key}={val:.4g}")
        else:
            parts.append(f"{key}={val}")
    if not parts:
        if "param_set_id" in row:
            return f"param_set={int(row['param_set_id'])}"
        return "param_set"
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot station raw_tt_*_count columns and overlay parameter-set "
            "reference lines from a dictionary CSV."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to a JSON config file.",
    )
    parser.add_argument("--station-id", type=int, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--start", default=None, help="Start time (e.g. 2026-01-01).")
    parser.add_argument("--end", default=None, help="End time (e.g. 2026-01-31).")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated list of raw_tt_* columns to plot.",
    )
    parser.add_argument(
        "--column-prefix",
        default=None,
        help="Prefix to select columns if --columns is not set.",
    )
    parser.add_argument(
        "--param-set-ids",
        default=None,
        help="Comma-separated param_set_id values to include.",
    )
    parser.add_argument(
        "--param-set-range",
        default=None,
        help="Inclusive param_set_id range in start:end format.",
    )
    parser.add_argument(
        "--param-range",
        action="append",
        nargs=3,
        default=[],
        metavar=("COL", "MIN", "MAX"),
        help="Filter param column by range: --param-range COL MIN MAX",
    )
    parser.add_argument(
        "--max-param-sets",
        type=int,
        default=None,
        help="Cap the number of param sets to plot (0 for no cap).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (png). Defaults next to this script.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for output images (used when plotting multiple groups).",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    parser.add_argument("--legend", action="store_true", default=None, help="Show legend.")
    parser.add_argument(
        "--no-legend",
        action="store_false",
        dest="legend",
        help="Disable legend.",
    )
    parser.add_argument(
        "--legend-params",
        default=None,
        help="Comma-separated param columns to show in legend labels.",
    )
    parser.add_argument("--cols-per-row", type=int, default=3)
    parser.add_argument(
        "--median-window",
        type=int,
        default=None,
        help="Rolling median window (rows) for station data; set 1 to disable.",
    )
    parser.add_argument(
        "--chisq-csv",
        default=None,
        help="Optional chi-square results CSV to filter dictionary rows.",
    )

    args = parser.parse_args()

    config_data = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config_data = _load_config(config_path)
        args = _merge_config(args, config_data)

    if args.station_id is None:
        args.station_id = 1
    if args.task_id is None:
        args.task_id = int((config_data or {}).get("task_id", 1))
    config_data = _apply_task_overrides(config_data or {}, args.task_id)
    normalize = bool(config_data.get("chisq_normalize", False))
    total_events_col = config_data.get("total_events_column")
    if args.dictionary_csv is None:
        dict_template = (config_data or {}).get("dictionary_csv", str(DEFAULT_DICT))
        if "{task_id" in dict_template:
            dict_template = dict_template.format(task_id=args.task_id)
        args.dictionary_csv = dict_template
    elif "{task_id" in args.dictionary_csv:
        args.dictionary_csv = args.dictionary_csv.format(task_id=args.task_id)
    if args.column_prefix is None:
        args.column_prefix = "raw_tt_"
    if args.max_param_sets is None:
        args.max_param_sets = 15
    if args.median_window is None:
        args.median_window = 15
    if args.legend_params is None:
        args.legend_params = "param_set_id,cos_n,flux_cm2_min,z_plane_1,z_plane_2,z_plane_3,z_plane_4"
    if args.legend is None:
        args.legend = False
    if args.out_dir is None and config_data:
        args.out_dir = config_data.get("plot_out_dir")

    metadata_path = (
        Path(args.metadata_csv)
        if args.metadata_csv
        else _task_metadata_path(args.station_id, args.task_id)
    )
    dictionary_path = Path(args.dictionary_csv)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Dictionary CSV not found: {dictionary_path}")

    meta_df = pd.read_csv(metadata_path, low_memory=False)
    if "execution_timestamp" not in meta_df.columns:
        raise KeyError("Metadata CSV must include 'execution_timestamp'.")

    meta_df = meta_df.copy()
    meta_df["execution_dt"] = pd.to_datetime(
        meta_df["execution_timestamp"],
        format="%Y-%m-%d_%H.%M.%S",
        errors="coerce",
    )
    meta_df = meta_df.dropna(subset=["execution_dt"]).sort_values("execution_dt")

    start_dt = _parse_datetime(args.start)
    end_dt = _parse_datetime(args.end)
    if start_dt is not None:
        meta_df = meta_df[meta_df["execution_dt"] >= start_dt]
    if end_dt is not None:
        meta_df = meta_df[meta_df["execution_dt"] <= end_dt]

    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        columns = [c for c in meta_df.columns if c.startswith(args.column_prefix)]

    dict_df = pd.read_csv(dictionary_path, low_memory=False)
    if "param_set_id" not in dict_df.columns:
        raise KeyError("Dictionary CSV must include 'param_set_id'.")
    dict_df = dict_df.copy()
    dict_df["param_set_id"] = pd.to_numeric(dict_df["param_set_id"], errors="coerce")
    dict_df = dict_df.dropna(subset=["param_set_id"])
    dict_df["param_set_id"] = dict_df["param_set_id"].astype(int)

    if args.param_set_ids:
        ids = _parse_int_list(args.param_set_ids)
        dict_df = dict_df[dict_df["param_set_id"].isin(ids)]

    if args.param_set_range:
        range_vals = _parse_range(args.param_set_range)
        if range_vals:
            start_id, end_id = range_vals
            dict_df = dict_df[
                (dict_df["param_set_id"] >= start_id)
                & (dict_df["param_set_id"] <= end_id)
            ]

    if args.param_range:
        dict_df = _filter_param_ranges(dict_df, args.param_range)

    dict_df = dict_df.sort_values("param_set_id")

    if args.chisq_csv:
        chisq_path = Path(args.chisq_csv)
        if not chisq_path.exists():
            raise FileNotFoundError(f"Chi-square CSV not found: {chisq_path}")
        chisq_df = pd.read_csv(chisq_path, low_memory=False)
        if "param_set_id" not in chisq_df.columns:
            raise KeyError("Chi-square CSV must include 'param_set_id'.")
        if "chisq" not in chisq_df.columns:
            raise KeyError("Chi-square CSV must include 'chisq'.")
        chisq_df = chisq_df.sort_values("chisq")
        dict_df = dict_df.merge(
            chisq_df[["param_set_id", "chisq"]],
            on="param_set_id",
            how="inner",
        )
        dict_df = dict_df.sort_values("chisq")
    if args.max_param_sets > 0 and len(dict_df) > args.max_param_sets:
        dict_df = dict_df.head(args.max_param_sets)

    if dict_df.empty:
        raise ValueError("No parameter sets selected after filtering.")

    tt_cols_set = set(_find_tt_columns(meta_df.columns.tolist()))
    use_total_events_col = (
        total_events_col
        if total_events_col
        and total_events_col in meta_df.columns
        and total_events_col in dict_df.columns
        else None
    )

    legend_params = [p.strip() for p in args.legend_params.split(",") if p.strip()]
    legend_params = [
        p for p in legend_params if p in dict_df.columns or p == "efficiencies"
    ]

    groups = []
    if config_data:
        config_groups = config_data.get("plot_groups") or config_data.get("groups")
        if isinstance(config_groups, list) and config_groups:
            groups = config_groups

    if not groups:
        groups = [
            {"name": "raw_tt", "prefix": args.column_prefix, "cols_per_row": args.cols_per_row},
            {
                "name": "Q_entries_original",
                "regex": r"^Q[1-4]_[FB]_[1-4]_entries_original$",
                "grid_rows": 4,
                "grid_cols": 4,
            },
        ]

    if args.columns:
        groups = [
            {
                "name": "custom_columns",
                "columns": columns,
                "cols_per_row": args.cols_per_row,
            }
        ]

    colors = plt.cm.tab20(np.linspace(0, 1, len(dict_df)))
    out_dir = Path(args.out_dir) if args.out_dir else None

    for group in groups:
        group_columns = _select_columns_by_group(
            meta_df.columns.tolist(), group, args.column_prefix
        )
        if group.get("name") == "Q_entries_original":
            group_columns = _sort_q_columns(group_columns)

        if not group_columns:
            print(f"Skipping group '{group.get('name', 'unnamed')}', no columns found.")
            continue

        meta_plot_df = _apply_median_filter(meta_df, group_columns, args.median_window)
        dict_plot_df = dict_df
        if normalize:
            norm_cols = [c for c in group_columns if c in tt_cols_set]
            if norm_cols:
                total_series = _compute_total_series(meta_plot_df, use_total_events_col, norm_cols)
                if total_series is not None:
                    denom = total_series.where(total_series > 0)
                    meta_plot_df = meta_plot_df.copy()
                    for col in norm_cols:
                        meta_plot_df[col] = pd.to_numeric(meta_plot_df[col], errors="coerce") / denom
                dict_norm_cols = [c for c in norm_cols if c in dict_df.columns]
                dict_total = _compute_total_series(dict_df, use_total_events_col, dict_norm_cols)
                if dict_total is not None:
                    denom = dict_total.where(dict_total > 0)
                    dict_plot_df = dict_df.copy()
                    for col in dict_norm_cols:
                        dict_plot_df[col] = pd.to_numeric(dict_plot_df[col], errors="coerce") / denom

        grid_rows = group.get("grid_rows")
        grid_cols = group.get("grid_cols")
        if grid_rows and grid_cols:
            rows = int(grid_rows)
            cols = int(grid_cols)
        else:
            cols = max(1, int(group.get("cols_per_row", args.cols_per_row)))
            rows = math.ceil(len(group_columns) / cols)

        per_page = rows * cols
        pages = math.ceil(len(group_columns) / per_page)

        for page_idx in range(pages):
            start_idx = page_idx * per_page
            end_idx = start_idx + per_page
            page_columns = group_columns[start_idx:end_idx]

            page_data = meta_plot_df[page_columns].apply(pd.to_numeric, errors="coerce")
            y_min = page_data.min(numeric_only=True).min()
            y_max = page_data.max(numeric_only=True).max()
            if pd.isna(y_min) or pd.isna(y_max):
                y_min, y_max = None, None
            elif y_min == y_max:
                pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
                y_min -= pad
                y_max += pad

            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(5.5 * cols, 3.5 * rows),
                sharex=True,
                sharey=True,
            )

            if isinstance(axes, np.ndarray):
                axes_list = axes.flatten().tolist()
            else:
                axes_list = [axes]

            legend_handles = []
            legend_labels = []
            for idx, col in enumerate(page_columns):
                ax = axes_list[idx]
                ax.plot(meta_plot_df["execution_dt"], meta_plot_df[col], color="black", linewidth=1)

                for color, (_, row) in zip(colors, dict_plot_df.iterrows()):
                    value = row.get(col)
                    if pd.isna(value):
                        continue
                    line = ax.axhline(
                        value,
                        color=color,
                        linestyle="--",
                        linewidth=1,
                        alpha=0.8,
                    )

                ax.set_title(col)
                ax.grid(True, alpha=0.3)

            for extra_ax in axes_list[len(page_columns) :]:
                extra_ax.set_visible(False)

            if y_min is not None and y_max is not None:
                for ax in axes_list[: len(page_columns)]:
                    ax.set_ylim(y_min, y_max)
                    ax.tick_params(labelleft=True)

            title_name = group.get("title") or group.get("name") or "group"
            page_suffix = f" (page {page_idx + 1}/{pages})" if pages > 1 else ""
            fig.suptitle(
                f"Station {args.station_id:02d} {title_name} vs parameter sets{page_suffix}",
                fontsize=12,
            )
            fig.autofmt_xdate(rotation=30, ha="right")

            if args.legend:
                if not legend_handles:
                    legend_handles = []
                    legend_labels = []
                    for color, (_, row) in zip(colors, dict_df.iterrows()):
                        label = _format_legend_label(row, legend_params)
                        legend_handles.append(
                            Line2D([0], [0], color=color, linestyle="--", linewidth=1)
                        )
                        legend_labels.append(label)
                if legend_handles:
                    fig.legend(
                        legend_handles,
                        legend_labels,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        borderaxespad=0.0,
                        fontsize=8,
                    )

            if args.out:
                base_out = Path(args.out)
                if len(groups) == 1 and pages == 1:
                    out_path = base_out
                else:
                    out_path = base_out.with_name(
                        f"{base_out.stem}_{group.get('name','group')}_p{page_idx + 1}{base_out.suffix}"
                    )
        else:
            base_dir = out_dir or (BASE_DIR / "STEP_3_PLOTS/output")
            base_dir = base_dir / f"task_{args.task_id:02d}"
            out_path = base_dir / (
                    f"station_{args.station_id:02d}_{group.get('name','group')}_p{page_idx + 1}.png"
                    if pages > 1
                    else f"station_{args.station_id:02d}_{group.get('name','group')}.png"
                )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            right_edge = 0.8 if args.legend else 1
            fig.tight_layout(rect=(0.06, 0, right_edge, 0.95))
            fig.subplots_adjust(left=0.08)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {out_path}")

            if args.show:
                plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
