#!/usr/bin/env python3
"""TASK_2 specific QA: generate custom specific plots and pass placeholders."""

from __future__ import annotations

from datetime import date, datetime
from fnmatch import fnmatch
from pathlib import Path
import math
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

TASK_DIR = Path(__file__).resolve().parent
STEP_DIR = TASK_DIR.parent
QA_ROOT = STEP_DIR.parent
REPO_ROOT = QA_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from MASTER.common.file_selection import extract_run_datetime_from_name
from common.task_setup import bootstrap_task, get_station_date_range, load_task_configs

TASK_ID = 2
METADATA_SUFFIX = "specific"
METADATA_TYPE = "specific"
DEFAULT_PASS_COLUMN = "task_2_specific_pass"
DEFAULT_CONFIG_NAME = "config.yaml"


def _normalize_station_name(station: object) -> str:
    text = str(station).strip().upper()
    if text.startswith("MINGO"):
        suffix = text.removeprefix("MINGO")
        return f"MINGO{int(suffix):02d}" if suffix.isdigit() else text
    return f"MINGO{int(text):02d}" if text.isdigit() else text


def _metadata_path(station_name: str) -> Path:
    return (
        REPO_ROOT
        / "STATIONS"
        / station_name
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{TASK_ID}"
        / "METADATA"
        / f"task_{TASK_ID}_metadata_{METADATA_SUFFIX}.csv"
    )


def _outputs_root(station_name: str) -> Path:
    return TASK_DIR / "STATIONS" / station_name / "OUTPUTS"


def _output_files_dir(station_name: str) -> Path:
    return _outputs_root(station_name) / "FILES"


def _output_plots_dir(station_name: str) -> Path:
    return _outputs_root(station_name) / "PLOTS"


def _build_pass_dataframe(meta_df: pd.DataFrame, pass_column: str, default_pass: float) -> pd.DataFrame:
    if "filename_base" not in meta_df.columns:
        return pd.DataFrame(columns=["filename_base", pass_column])
    out = pd.DataFrame()
    out["filename_base"] = meta_df["filename_base"].astype("string").fillna("").str.strip()
    out = out[out["filename_base"] != ""].drop_duplicates().reset_index(drop=True)
    out[pass_column] = float(default_pass)
    return out


def _load_yaml_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _parse_boundary(value: Any, *, end_of_day_if_date_only: bool) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        boundary = pd.to_datetime(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        boundary = pd.to_datetime(text, errors="coerce")

    if pd.isna(boundary):
        raise ValueError(f"Invalid date boundary '{value}'.")

    if end_of_day_if_date_only and isinstance(value, str) and len(value.strip()) <= 10:
        boundary = boundary + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    return boundary


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    as_text = series.astype("string").fillna("").str.strip()
    parsed = pd.to_datetime(as_text, format="%Y-%m-%d_%H.%M.%S", errors="coerce")
    if parsed.notna().all():
        return parsed

    alt_formats = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
    for fmt in alt_formats:
        remaining = parsed.isna()
        if not remaining.any():
            break
        parsed.loc[remaining] = pd.to_datetime(as_text.loc[remaining], format=fmt, errors="coerce")

    return parsed


def _parse_filename_timestamp_series(series: pd.Series) -> pd.Series:
    as_text = series.astype("string").fillna("").str.strip()
    parsed = as_text.map(lambda value: extract_run_datetime_from_name(value) if value else None)
    return pd.to_datetime(parsed, errors="coerce")


def _x_axis_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("x_axis")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "mode": str(raw.get("mode", "filename_timestamp")).strip().lower(),
        "filename_column": str(raw.get("filename_column", "filename_base")).strip() or "filename_base",
        "column": raw.get("column"),
    }


def _pick_time_column(df: pd.DataFrame, config: dict[str, Any]) -> str:
    priority = config.get("time_columns_priority") or ["execution_timestamp", "datetime"]
    for candidate in priority:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "No preferred timestamp column found. "
        f"Looked for: {priority}. Available columns: {list(df.columns)}"
    )


def _resolve_filter_timestamp(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, str]:
    axis_cfg = _x_axis_config(config)
    filename_column = axis_cfg["filename_column"]
    if filename_column in df.columns:
        from_filename = _parse_filename_timestamp_series(df[filename_column])
        if from_filename.notna().any():
            return from_filename, f"{filename_column} (miXXYYDDDHHMMSS)"

    time_col = _pick_time_column(df, config)
    parsed_time = _parse_timestamp_series(df[time_col])
    if parsed_time.notna().any():
        return parsed_time, time_col

    raise KeyError(
        "Could not resolve timestamp values for filtering. "
        f"Tried filename column '{filename_column}' and priority columns "
        f"{config.get('time_columns_priority') or ['execution_timestamp', 'datetime']}."
    )


def _resolve_plot_x_axis(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, str, bool]:
    axis_cfg = _x_axis_config(config)
    mode = axis_cfg["mode"]

    if mode == "filename_timestamp":
        filename_column = axis_cfg["filename_column"]
        if filename_column in df.columns:
            parsed = _parse_filename_timestamp_series(df[filename_column])
            if parsed.notna().any():
                return parsed, filename_column, True
        return df["__timestamp__"], "__timestamp__", True

    if mode == "column":
        column = axis_cfg.get("column")
        if column is None:
            return df["__timestamp__"], "__timestamp__", True
        column_name = str(column).strip()
        if not column_name:
            return df["__timestamp__"], "__timestamp__", True
        if column_name not in df.columns:
            raise KeyError(f"x_axis.column='{column_name}' not present in metadata.")

        numeric = pd.to_numeric(df[column_name], errors="coerce")
        if numeric.notna().any():
            return numeric, column_name, False

        parsed = _parse_timestamp_series(df[column_name])
        if parsed.notna().any():
            return parsed, column_name, True

        raise ValueError(f"x_axis.column='{column_name}' could not be parsed as numeric or datetime.")

    raise ValueError("Invalid x_axis.mode. Use one of: filename_timestamp, column.")


def _apply_date_filter(df: pd.DataFrame, date_range: dict[str, Any] | None) -> pd.DataFrame:
    if date_range is None:
        return df

    start = _parse_boundary(date_range.get("start"), end_of_day_if_date_only=False)
    end = _parse_boundary(date_range.get("end"), end_of_day_if_date_only=True)

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["__timestamp__"] >= start
    if end is not None:
        mask &= df["__timestamp__"] <= end
    return df.loc[mask].copy()


def _plot_series_or_empty(
    *,
    ax: plt.Axes,
    x_values: pd.Series,
    y_values: pd.Series,
    label: str,
    color: str,
    line_width: float,
    marker_size: float,
) -> bool:
    numeric = pd.to_numeric(y_values, errors="coerce")
    if numeric.notna().any():
        ax.plot(
            x_values,
            numeric,
            linestyle="-",
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            label=label,
            color=color,
        )
        return True

    ax.plot([], [], linestyle="-", linewidth=line_width, marker="o", markersize=marker_size, label=f"{label} (no data)", color=color)
    return False


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    return None


def _plot_crt_band(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    station_name: str,
    plots_dir: Path,
    plot_cfg: dict[str, Any],
) -> Path | None:
    crt_cfg = plot_cfg.get("crt_band", {}) if isinstance(plot_cfg.get("crt_band"), dict) else {}
    avg_col = str(crt_cfg.get("avg_column", "CRT_avg")).strip() or "CRT_avg"
    std_col = str(crt_cfg.get("std_column", "CRT_std")).strip() or "CRT_std"

    if avg_col not in df.columns or std_col not in df.columns:
        return None

    avg = pd.to_numeric(df[avg_col], errors="coerce")
    std = pd.to_numeric(df[std_col], errors="coerce").abs()
    valid = avg.notna()

    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    dpi = int(plot_cfg.get("dpi", 150))
    figsize = tuple(crt_cfg.get("figsize", [14, 5]))

    fig, ax = plt.subplots(figsize=figsize)

    if valid.any():
        x = x_values[valid]
        y = avg[valid]
        s = std[valid].fillna(0)
        ax.plot(x, y, color="#1f77b4", linewidth=1.2, marker="o", markersize=2.2, label=avg_col)
        ax.fill_between(x, y - s, y + s, color="#1f77b4", alpha=0.22, label=f"{std_col} band")
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=10)

    ax.set_title(f"TASK_{TASK_ID} {station_name} - CRT average with uncertainty band")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Rate / value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(crt_cfg.get("filename", "crt_avg_std_band")).strip() or "crt_avg_std_band"
    out_path = plots_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_{base_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_entries_grid(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    station_name: str,
    plots_dir: Path,
    plot_cfg: dict[str, Any],
) -> Path:
    entries_cfg = plot_cfg.get("entries_grid", {}) if isinstance(plot_cfg.get("entries_grid"), dict) else {}
    nrows = int(entries_cfg.get("nrows", 4))
    ncols = int(entries_cfg.get("ncols", 4))
    figsize = tuple(entries_cfg.get("figsize", [18, 12]))
    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    dpi = int(plot_cfg.get("dpi", 150))
    marker_size = float(plot_cfg.get("marker_size", 2.0))
    line_width = float(plot_cfg.get("line_width", 0.9))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    axes_flat = np.ravel(axes)

    for plane in range(1, 5):
        for strip in range(1, 5):
            idx = (plane - 1) * 4 + (strip - 1)
            ax = axes_flat[idx]

            col_original = _first_existing_column(
                df,
                [
                    f"P{plane}_s{strip}_entries_original_rate_hz",
                    f"P{plane}_s{strip}_entries_original",
                ],
            )
            col_final = _first_existing_column(
                df,
                [
                    f"P{plane}_s{strip}_entries_final_rate_hz",
                    f"P{plane}_s{strip}_entries_final",
                ],
            )

            has_any = False
            missing: list[str] = []

            if col_original is not None:
                label = "original" if col_original.endswith("_rate_hz") else "original (count)"
                has_any = _plot_series_or_empty(
                    ax=ax,
                    x_values=x_values,
                    y_values=df[col_original],
                    label=label,
                    color="#1f77b4",
                    line_width=line_width,
                    marker_size=marker_size,
                ) or has_any
            else:
                missing.append("original")

            if col_final is not None:
                label = "final" if col_final.endswith("_rate_hz") else "final (count)"
                has_any = _plot_series_or_empty(
                    ax=ax,
                    x_values=x_values,
                    y_values=df[col_final],
                    label=label,
                    color="#ff7f0e",
                    line_width=line_width,
                    marker_size=marker_size,
                ) or has_any
            else:
                missing.append("final")

            if not has_any and not missing:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=8)
            if missing:
                ax.text(0.5, 0.08, f"missing: {', '.join(missing)}", transform=ax.transAxes, ha="center", va="bottom", color="red", fontsize=7)

            ax.set_title(f"P{plane}_s{strip}", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, loc="best")

    for idx in range(16, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"TASK_{TASK_ID} {station_name} specific - entries original vs final (x={x_label})", fontsize=11)
    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plots_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(entries_cfg.get("filename", "entries_grid")).strip() or "entries_grid"
    out_path = plots_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_{base_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_pair_rates_threshold(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    station_name: str,
    plots_dir: Path,
    plot_cfg: dict[str, Any],
) -> Path | None:
    pair_cfg = plot_cfg.get("pair_rates", {}) if isinstance(plot_cfg.get("pair_rates"), dict) else {}
    if not bool(pair_cfg.get("enabled", True)):
        return None

    pattern = str(pair_cfg.get("pattern", "P*s*_P*s*_*_rate_hz")).strip() or "P*s*_P*s*_*_rate_hz"
    threshold_hz = float(pair_cfg.get("threshold_hz", 0.5))

    candidates = [col for col in df.columns if fnmatch(col, pattern)]
    selected: list[str] = []
    for col in candidates:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any() and float(s.max()) > threshold_hz:
            selected.append(col)

    if not selected:
        return None

    ncols = max(1, int(pair_cfg.get("ncols", 4)))
    nrows = int(math.ceil(len(selected) / ncols))
    figsize_per_col = float(pair_cfg.get("figsize_per_col", 4.2))
    figsize_per_row = float(pair_cfg.get("figsize_per_row", 2.7))
    figsize = (figsize_per_col * ncols, figsize_per_row * nrows)

    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    dpi = int(plot_cfg.get("dpi", 150))
    marker_size = float(plot_cfg.get("marker_size", 2.0))
    line_width = float(plot_cfg.get("line_width", 0.9))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    axes_flat = np.ravel(axes)

    for idx, col in enumerate(selected):
        ax = axes_flat[idx]
        y = pd.to_numeric(df[col], errors="coerce")
        ax.plot(x_values, y, linestyle="-", linewidth=line_width, marker="o", markersize=marker_size, color="#2ca02c")
        ax.set_title(col, fontsize=8)
        ax.grid(True, alpha=0.25)

    for idx in range(len(selected), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        f"TASK_{TASK_ID} {station_name} specific - pair rates > {threshold_hz:g} Hz (x={x_label})",
        fontsize=11,
    )
    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plots_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(pair_cfg.get("filename", "pair_rates_threshold")).strip() or "pair_rates_threshold"
    out_path = plots_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_{base_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _generate_station_plots(station_name: str, config: dict[str, Any], meta_df: pd.DataFrame) -> list[Path]:
    if meta_df.empty:
        return []

    df = meta_df.copy()
    df["__timestamp__"], timestamp_source = _resolve_filter_timestamp(df, config)
    df = df.loc[df["__timestamp__"].notna()].copy()
    if df.empty:
        print(f"TASK_{TASK_ID} {station_name}: no rows with valid timestamp source '{timestamp_source}'.")
        return []

    date_range = get_station_date_range(config=config, station=station_name)
    df = _apply_date_filter(df, date_range)
    if df.empty:
        print(f"TASK_{TASK_ID} {station_name}: all rows filtered out by date_range={date_range}.")
        return []

    x_values, x_label, x_is_datetime = _resolve_plot_x_axis(df, config)
    df["__plot_x__"] = x_values
    df = df.loc[df["__plot_x__"].notna()].copy()
    if df.empty:
        print(f"TASK_{TASK_ID} {station_name}: no rows with valid x-axis values.")
        return []

    sort_key = "__plot_x__" if x_label != "__timestamp__" else "__timestamp__"
    df = df.sort_values(sort_key).reset_index(drop=True)
    x_values = df["__plot_x__"]

    plot_cfg = config.get("plots", {}) if isinstance(config.get("plots"), dict) else {}
    plots_dir = _output_plots_dir(station_name)

    created: list[Path] = []
    crt_path = _plot_crt_band(
        df=df,
        x_values=x_values,
        x_label=x_label,
        x_is_datetime=x_is_datetime,
        station_name=station_name,
        plots_dir=plots_dir,
        plot_cfg=plot_cfg,
    )
    if crt_path is not None:
        created.append(crt_path)

    created.append(
        _plot_entries_grid(
            df=df,
            x_values=x_values,
            x_label=x_label,
            x_is_datetime=x_is_datetime,
            station_name=station_name,
            plots_dir=plots_dir,
            plot_cfg=plot_cfg,
        )
    )

    pair_path = _plot_pair_rates_threshold(
        df=df,
        x_values=x_values,
        x_label=x_label,
        x_is_datetime=x_is_datetime,
        station_name=station_name,
        plots_dir=plots_dir,
        plot_cfg=plot_cfg,
    )
    if pair_path is not None:
        created.append(pair_path)

    return created


def main() -> int:
    config = load_task_configs(TASK_DIR)
    config_path = TASK_DIR / DEFAULT_CONFIG_NAME
    if not config_path.exists():
        config_path = TASK_DIR / "config.yaml"
    config_override = _load_yaml_mapping(config_path)
    if config_override:
        config = _deep_merge(config, config_override)

    bootstrap_task(TASK_DIR)

    stations = config.get("stations", [0, 1, 2, 3, 4])
    pass_column = str(config.get("pass_column_name", DEFAULT_PASS_COLUMN))
    default_pass = float(config.get("pass_default_value", 1.0))

    total_rows = 0
    written = 0
    total_plots = 0

    for station in stations:
        station_name = _normalize_station_name(station)
        meta_path = _metadata_path(station_name)

        if meta_path.exists() and meta_path.stat().st_size > 0:
            meta_df = pd.read_csv(meta_path, low_memory=False)
        else:
            meta_df = pd.DataFrame(columns=["filename_base"])

        pass_df = _build_pass_dataframe(meta_df, pass_column, default_pass)
        total_rows += len(pass_df)

        files_dir = _output_files_dir(station_name)
        files_dir.mkdir(parents=True, exist_ok=True)
        out_path = files_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_pass.csv"
        pass_df.to_csv(out_path, index=False)
        written += 1

        try:
            created_plots = _generate_station_plots(station_name=station_name, config=config, meta_df=meta_df)
            total_plots += len(created_plots)
            if created_plots:
                print(f"TASK_{TASK_ID} {station_name}: generated {len(created_plots)} plot files.")
        except (KeyError, ValueError) as exc:
            print(f"TASK_{TASK_ID} {station_name}: plotting skipped ({exc})")

    print(
        f"TASK_{TASK_ID} {METADATA_TYPE} complete: "
        f"stations={written} total_rows={total_rows} pass_column={pass_column} plots={total_plots}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
