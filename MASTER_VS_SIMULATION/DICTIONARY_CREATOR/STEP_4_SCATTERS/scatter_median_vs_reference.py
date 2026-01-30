#!/usr/bin/env python3
"""Scatter plot of median data values vs dictionary reference values."""
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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from STEP_3_PLOTS.plot_station_metadata_vs_dictionary import (  # noqa: E402
    _load_config,
    _parse_datetime,
    _select_columns_by_group,
    _sort_q_columns,
)

DEFAULT_CONFIG = BASE_DIR / "config/pipeline_config.json"
DEFAULT_DICT = BASE_DIR / "STEP_1_BUILD/param_metadata_dictionary.csv"


def _compute_medians(
    meta_df: pd.DataFrame,
    columns: list[str],
    total_series: pd.Series | None,
    normalize_cols: set[str] | None,
) -> dict[str, float]:
    med = {}
    denom = total_series.where(total_series > 0) if total_series is not None else None
    for col in columns:
        series = pd.to_numeric(meta_df[col], errors="coerce")
        if denom is not None and normalize_cols is not None and col in normalize_cols:
            series = series / denom
        series = series.dropna()
        if series.empty:
            continue
        med[col] = float(series.median())
    return med


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


def _add_eff_columns_from_dict(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "efficiencies" not in df.columns:
        return df
    import ast

    def parse_eff(val):
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            return val[:4]
        if isinstance(val, str):
            try:
                out = ast.literal_eval(val)
                if isinstance(out, (list, tuple)) and len(out) >= 4:
                    return out[:4]
            except Exception:
                return None
        return None

    effs = df["efficiencies"].apply(parse_eff)
    df["eff_1"] = effs.apply(lambda x: x[0] if x else np.nan)
    df["eff_2"] = effs.apply(lambda x: x[1] if x else np.nan)
    df["eff_3"] = effs.apply(lambda x: x[2] if x else np.nan)
    df["eff_4"] = effs.apply(lambda x: x[3] if x else np.nan)
    return df


def _task_metadata_path(station_id: int, task_id: int) -> Path:
    station = f"MINGO{station_id:02d}"
    return Path(
        "/home/mingo/DATAFLOW_v3/STATIONS"
        f"/{station}/STAGE_1/EVENT_DATA/STEP_1/TASK_{task_id}/METADATA/"
        f"task_{task_id}_metadata_specific.csv"
    )


def _load_and_filter_metadata(config: dict, task_id: int) -> pd.DataFrame:
    station_id = int(config.get("station_id", 1))
    meta_path = Path(config.get("metadata_csv") or _task_metadata_path(station_id, task_id))
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
    df = pd.read_csv(meta_path, low_memory=False)
    df["execution_dt"] = pd.to_datetime(
        df["execution_timestamp"],
        format="%Y-%m-%d_%H.%M.%S",
        errors="coerce",
    )
    df = df.dropna(subset=["execution_dt"]).sort_values("execution_dt")
    start_dt = _parse_datetime(config.get("start"))
    end_dt = _parse_datetime(config.get("end"))
    if start_dt is not None:
        df = df[df["execution_dt"] >= start_dt]
    if end_dt is not None:
        df = df[df["execution_dt"] <= end_dt]
    return df


def _load_dictionary(config: dict, task_id: int, chisq_csv: str | None) -> pd.DataFrame:
    dict_template = config.get("dictionary_csv", str(DEFAULT_DICT))
    if "{task_id" in dict_template:
        dict_template = dict_template.format(task_id=task_id)
    dict_path = Path(dict_template)
    if not dict_path.exists():
        raise FileNotFoundError(f"Dictionary CSV not found: {dict_path}")
    df = pd.read_csv(dict_path, low_memory=False)
    df["param_set_id"] = pd.to_numeric(df["param_set_id"], errors="coerce")
    df = df.dropna(subset=["param_set_id"])
    df["param_set_id"] = df["param_set_id"].astype(int)

    if chisq_csv:
        chisq_path = Path(chisq_csv)
        if not chisq_path.exists():
            raise FileNotFoundError(f"Chi-square CSV not found: {chisq_path}")
        chisq_df = pd.read_csv(chisq_path, low_memory=False)
        if "param_set_id" not in chisq_df.columns:
            raise KeyError("Chi-square CSV must include 'param_set_id'.")
        if "chisq" not in chisq_df.columns:
            raise KeyError("Chi-square CSV must include 'chisq'.")
        chisq_df = chisq_df.sort_values("chisq")
        df = df.merge(chisq_df[["param_set_id", "chisq"]], on="param_set_id", how="inner")
        df = df.sort_values("chisq")

    max_sets = int(config.get("max_param_sets", 0) or 0)
    if max_sets > 0 and len(df) > max_sets:
        df = df.head(max_sets)
    return df


def _apply_task_overrides(config: dict, task_id: int) -> dict:
    overrides = (config.get("task_settings") or {}).get(str(task_id))
    if not overrides:
        return config
    merged = dict(config)
    merged.update(overrides)
    return merged


def _plot_group(
    medians: dict[str, float],
    dict_df: pd.DataFrame,
    columns: list[str],
    title: str,
    out_path: Path,
) -> None:
    if not columns:
        return
    dict_columns = [c for c in columns if c in dict_df.columns]
    if not dict_columns:
        return
    x_vals = np.array([medians.get(c, np.nan) for c in dict_columns], dtype=float)
    mask = np.isfinite(x_vals)
    if not mask.any():
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(dict_df), 1)))

    for color, (_, row) in zip(colors, dict_df.iterrows()):
        y_vals = np.array([row.get(c, np.nan) for c in dict_columns], dtype=float)
        ok = mask & np.isfinite(y_vals)
        if not ok.any():
            continue
        ax.scatter(x_vals[ok], y_vals[ok], s=14, alpha=0.7, color=color)

    min_val = np.nanmin(
        [x_vals[mask].min(), dict_df[dict_columns].min(numeric_only=True).min()]
    )
    max_val = np.nanmax(
        [x_vals[mask].max(), dict_df[dict_columns].max(numeric_only=True).max()]
    )
    if np.isfinite(min_val) and np.isfinite(max_val):
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1)

    ax.set_xlabel("Median data value")
    ax.set_ylabel("Reference value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_eff_cal_vs_eff(
    meta_df: pd.DataFrame,
    dict_df: pd.DataFrame,
    out_path: Path,
    title: str,
    prefix: str,
) -> None:
    eff_cols = [
        f"eff_quick_{prefix}_1",
        f"eff_quick_{prefix}_2",
        f"eff_quick_{prefix}_3",
        f"eff_quick_{prefix}_4",
    ]
    dict_eff_cols = [
        f"eff_quick_{prefix}_1",
        f"eff_quick_{prefix}_2",
        f"eff_quick_{prefix}_3",
        f"eff_quick_{prefix}_4",
    ]
    if not all(c in meta_df.columns for c in eff_cols):
        return
    if not all(c in dict_df.columns for c in dict_eff_cols):
        return
    medians = {c: float(pd.to_numeric(meta_df[c], errors="coerce").median()) for c in eff_cols}
    x_vals = np.array([medians[c] for c in eff_cols], dtype=float)
    if not np.isfinite(x_vals).any():
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    plane_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    plane_labels = ["plane 1", "plane 2", "plane 3", "plane 4"]
    for idx, (x, eff_col, color, label) in enumerate(
        zip(x_vals, dict_eff_cols, plane_colors, plane_labels)
    ):
        if not np.isfinite(x):
            continue
        y_vals = pd.to_numeric(dict_df[eff_col], errors="coerce")
        ok = np.isfinite(y_vals)
        if not ok.any():
            continue
        ax.scatter(
            np.full(ok.sum(), x),
            y_vals[ok],
            s=18,
            alpha=0.6,
            color=color,
            label=label,
        )

    min_val = np.nanmin([x_vals[np.isfinite(x_vals)].min(), dict_df[dict_eff_cols].min().min()])
    max_val = np.nanmax([x_vals[np.isfinite(x_vals)].max(), dict_df[dict_eff_cols].max().max()])
    if np.isfinite(min_val) and np.isfinite(max_val):
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1)
    ax.set_xlabel(f"Median eff_quick ({prefix})")
    ax.set_ylabel(f"Dictionary eff_quick ({prefix})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scatter plot of median data values vs dictionary references."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--chisq-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids to process (default: 1,2,3,4,5).",
    )

    args = parser.parse_args()
    config = _load_config(Path(args.config))

    groups = config.get("scatter_groups") or config.get("groups") or [
        {"name": "tt_count", "regex": r".*_tt_.*_count$"},
    ]

    task_ids = config.get("scatter_tasks")
    if args.tasks:
        task_ids = [int(t.strip()) for t in args.tasks.split(",") if t.strip()]
    if not task_ids:
        task_ids = [1, 2, 3, 4, 5]

    out_dir = Path(
        args.out_dir or config.get("scatter_out_dir") or config.get("out_dir") or "/tmp"
    )

    for task_id in task_ids:
        task_cfg = _apply_task_overrides(config, task_id)
        meta_df = _load_and_filter_metadata(task_cfg, task_id)
        dict_df = _load_dictionary(task_cfg, task_id, args.chisq_csv)
        dict_df = _add_eff_columns_from_dict(dict_df)
        normalize = bool(task_cfg.get("chisq_normalize", False))
        total_events_col = task_cfg.get("total_events_column")
        tt_cols_set = set(_find_tt_columns(meta_df.columns.tolist()))
        use_total_events_col = (
            total_events_col
            if total_events_col
            and total_events_col in meta_df.columns
            and total_events_col in dict_df.columns
            else None
        )
        prefixes = sorted(
            set(_find_tt_prefixes(meta_df.columns.tolist()))
            | set(_find_tt_prefixes(dict_df.columns.tolist()))
        )
        if prefixes:
            meta_df = _add_eff_quick_columns(meta_df, prefixes)
            dict_df = _add_eff_quick_columns(dict_df, prefixes)
        task_out_dir = out_dir / f"task_{task_id:02d}"
        groups = task_cfg.get("scatter_groups") or task_cfg.get("groups") or [
            {"name": "tt_count", "regex": r".*_tt_.*_count$"},
        ]
        for group in groups:
            name = group.get("name", "group")
            columns = _select_columns_by_group(meta_df.columns.tolist(), group, "raw_tt_")
            if name == "Q_entries_original":
                columns = _sort_q_columns(columns)
            if not columns:
                continue
            norm_cols = [c for c in columns if c in tt_cols_set]
            total_series = (
                _compute_total_series(meta_df, use_total_events_col, norm_cols)
                if normalize
                else None
            )
            medians = _compute_medians(meta_df, columns, total_series, set(norm_cols))
            dict_plot_df = dict_df
            if normalize and norm_cols:
                dict_norm_cols = [c for c in norm_cols if c in dict_df.columns]
                dict_total = _compute_total_series(dict_df, use_total_events_col, dict_norm_cols)
                if dict_total is not None:
                    denom = dict_total.where(dict_total > 0)
                    dict_plot_df = dict_df.copy()
                    for col in dict_norm_cols:
                        dict_plot_df[col] = pd.to_numeric(dict_plot_df[col], errors="coerce") / denom
            out_path = (
                task_out_dir
                / f"scatter_task_{task_id:02d}_median_vs_reference_{name}.png"
            )
            _plot_group(
                medians,
                dict_plot_df,
                columns,
                f"TASK {task_id}: {name} median vs reference",
                out_path,
            )
            print(f"Saved scatter: {out_path}")

        for prefix in prefixes:
            eff_out = task_out_dir / f"scatter_task_{task_id:02d}_eff_quick_{prefix}_vs_eff.png"
            _plot_eff_cal_vs_eff(
                meta_df,
                dict_df,
                eff_out,
                f"TASK {task_id}: eff_quick ({prefix}) vs efficiencies",
                prefix,
            )
            print(f"Saved scatter: {eff_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
