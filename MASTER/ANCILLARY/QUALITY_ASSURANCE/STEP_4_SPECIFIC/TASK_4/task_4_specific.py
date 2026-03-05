#!/usr/bin/env python3
"""TASK_4 specific QA: generate matrix plots and pass placeholders."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

TASK_ID = 4
METADATA_SUFFIX = "specific"
METADATA_TYPE = "specific"
DEFAULT_PASS_COLUMN = "task_4_specific_pass"
DEFAULT_CONFIG_NAME = "config.yaml"

DEFAULT_FAMILIES = ["12", "13", "14", "23", "24", "34", "123", "124", "134", "234", "1234"]
DEFAULT_PAIR_FAMILIES = ["12", "13", "14", "23", "24", "34"]
DEFAULT_TRIPLE_QUAD_FAMILIES = ["123", "124", "134", "234", "1234"]
DEFAULT_SIGMOID_VARIABLES = [
    "sigmoid_width",
    "background_slope",
    "sigmoid_amplitude",
    "sigmoid_center",
    "fit_normalization",
]
DEFAULT_RESIDUAL_METRICS = ["ystr", "tsum", "tdif"]
PLANE_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}


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


def _normalize_list(values: object, fallback: list[str]) -> list[str]:
    if not isinstance(values, list):
        return list(fallback)
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            out.append(text)
    return out or list(fallback)


def _plot_sigmoid_matrix(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    station_name: str,
    plots_dir: Path,
    plot_cfg: dict[str, Any],
) -> Path | None:
    sigmoid_cfg = plot_cfg.get("sigmoid_fit") if isinstance(plot_cfg.get("sigmoid_fit"), dict) else {}
    if sigmoid_cfg.get("enabled", True) is False:
        return None

    families = _normalize_list(sigmoid_cfg.get("families"), DEFAULT_FAMILIES)
    variables = _normalize_list(sigmoid_cfg.get("variables"), DEFAULT_SIGMOID_VARIABLES)

    nrows = len(families)
    ncols = len(variables)
    if nrows <= 0 or ncols <= 0:
        return None

    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    dpi = int(plot_cfg.get("dpi", 150))
    default_size = [max(20, 4 * ncols), max(20, int(2.2 * nrows))]
    figsize = tuple(sigmoid_cfg.get("figsize", default_size))
    marker_size = float(plot_cfg.get("marker_size", 1.8))
    line_width = float(plot_cfg.get("line_width", 0.8))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    axes_2d = np.atleast_2d(axes)

    for row_idx, family in enumerate(families):
        for col_idx, variable in enumerate(variables):
            ax = axes_2d[row_idx, col_idx]
            column = f"{variable}_{family}"

            if column in df.columns:
                y = pd.to_numeric(df[column], errors="coerce")
                if y.notna().any():
                    ax.plot(x_values, y, linestyle="-", linewidth=line_width, marker="o", markersize=marker_size)
                else:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=7)
            else:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", color="red", fontsize=7)

            if row_idx == 0:
                ax.set_title(variable, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(family, fontsize=8)
            ax.grid(True, alpha=0.25)

    fig.suptitle(f"TASK_{TASK_ID} {station_name} specific - sigmoid/fit matrix (x={x_label})", fontsize=12)
    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plots_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(sigmoid_cfg.get("filename", "sigmoid_fit_matrix")).strip() or "sigmoid_fit_matrix"
    out_path = plots_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_{base_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_residual_matrix(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    station_name: str,
    plots_dir: Path,
    plot_cfg: dict[str, Any],
    cfg: dict[str, Any],
    prefix: str,
    title_prefix: str,
) -> Path | None:
    if cfg.get("enabled", True) is False:
        return None

    families = _normalize_list(cfg.get("families"), DEFAULT_TRIPLE_QUAD_FAMILIES)
    metrics = _normalize_list(cfg.get("metrics"), DEFAULT_RESIDUAL_METRICS)
    if not families or not metrics:
        return None

    nrows = len(families)
    ncols = len(metrics)

    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    dpi = int(plot_cfg.get("dpi", 150))
    default_size = [max(14, 4 * ncols), max(10, int(2.0 * nrows))]
    figsize = tuple(cfg.get("figsize", default_size))
    marker_size = float(plot_cfg.get("marker_size", 1.8))
    line_width = float(plot_cfg.get("line_width", 0.8))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    axes_2d = np.atleast_2d(axes)

    for row_idx, family in enumerate(families):
        for col_idx, metric in enumerate(metrics):
            ax = axes_2d[row_idx, col_idx]
            available_planes: list[int] = []
            has_data = False

            for plane in (1, 2, 3, 4):
                column = f"{prefix}res_{metric}_{plane}_{family}_sigma"
                if column not in df.columns:
                    continue
                available_planes.append(plane)
                y = pd.to_numeric(df[column], errors="coerce")
                if y.notna().any():
                    ax.plot(
                        x_values,
                        y,
                        linestyle="-",
                        linewidth=line_width,
                        marker="o",
                        markersize=marker_size,
                        color=PLANE_COLORS[plane],
                    )
                    has_data = True

            if not available_planes:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", color="red", fontsize=7)
            elif not has_data:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=7)

            if row_idx == 0:
                ax.set_title(metric, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(family, fontsize=8)

            # Keep legends compact: only at right column.
            if col_idx == ncols - 1 and available_planes:
                handles = [Line2D([0], [0], color=PLANE_COLORS[p], lw=1.0, label=f"P{p}") for p in available_planes]
                ax.legend(handles=handles, fontsize=6, loc="best")

            ax.grid(True, alpha=0.25)

    fig.suptitle(f"TASK_{TASK_ID} {station_name} specific - {title_prefix} residuals (x={x_label})", fontsize=12)
    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plots_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(cfg.get("filename", f"{prefix}residuals_matrix")).strip() or f"{prefix}residuals_matrix"
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

    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    plots_dir = _output_plots_dir(station_name)

    created: list[Path] = []

    sigmoid_path = _plot_sigmoid_matrix(
        df=df,
        x_values=x_values,
        x_label=x_label,
        x_is_datetime=x_is_datetime,
        station_name=station_name,
        plots_dir=plots_dir,
        plot_cfg=plot_cfg,
    )
    if sigmoid_path is not None:
        created.append(sigmoid_path)

    residual_cfg = plot_cfg.get("residuals") if isinstance(plot_cfg.get("residuals"), dict) else {}
    pairs_cfg = residual_cfg.get("pairs") if isinstance(residual_cfg.get("pairs"), dict) else {
        "families": DEFAULT_PAIR_FAMILIES,
        "metrics": DEFAULT_RESIDUAL_METRICS,
        "filename": "residuals_pairs",
    }
    if "families" not in pairs_cfg:
        pairs_cfg = dict(pairs_cfg)
        pairs_cfg["families"] = DEFAULT_PAIR_FAMILIES
    if "metrics" not in pairs_cfg:
        pairs_cfg = dict(pairs_cfg)
        pairs_cfg["metrics"] = DEFAULT_RESIDUAL_METRICS

    triples_cfg = residual_cfg.get("triples_quads") if isinstance(residual_cfg.get("triples_quads"), dict) else {
        "families": DEFAULT_TRIPLE_QUAD_FAMILIES,
        "metrics": DEFAULT_RESIDUAL_METRICS,
        "filename": "residuals_triples_quads",
    }
    if "families" not in triples_cfg:
        triples_cfg = dict(triples_cfg)
        triples_cfg["families"] = DEFAULT_TRIPLE_QUAD_FAMILIES
    if "metrics" not in triples_cfg:
        triples_cfg = dict(triples_cfg)
        triples_cfg["metrics"] = DEFAULT_RESIDUAL_METRICS

    ext_cfg = residual_cfg.get("ext_triples_quads") if isinstance(residual_cfg.get("ext_triples_quads"), dict) else {
        "families": DEFAULT_TRIPLE_QUAD_FAMILIES,
        "metrics": DEFAULT_RESIDUAL_METRICS,
        "filename": "ext_residuals_triples_quads",
    }
    if "families" not in ext_cfg:
        ext_cfg = dict(ext_cfg)
        ext_cfg["families"] = DEFAULT_TRIPLE_QUAD_FAMILIES
    if "metrics" not in ext_cfg:
        ext_cfg = dict(ext_cfg)
        ext_cfg["metrics"] = DEFAULT_RESIDUAL_METRICS

    pairs_path = _plot_residual_matrix(
        df=df,
        x_values=x_values,
        x_label=x_label,
        x_is_datetime=x_is_datetime,
        station_name=station_name,
        plots_dir=plots_dir,
        plot_cfg=plot_cfg,
        cfg=pairs_cfg,
        prefix="",
        title_prefix="pair-family",
    )
    if pairs_path is not None:
        created.append(pairs_path)

    triples_path = _plot_residual_matrix(
        df=df,
        x_values=x_values,
        x_label=x_label,
        x_is_datetime=x_is_datetime,
        station_name=station_name,
        plots_dir=plots_dir,
        plot_cfg=plot_cfg,
        cfg=triples_cfg,
        prefix="",
        title_prefix="triple/quad-family",
    )
    if triples_path is not None:
        created.append(triples_path)

    ext_path = _plot_residual_matrix(
        df=df,
        x_values=x_values,
        x_label=x_label,
        x_is_datetime=x_is_datetime,
        station_name=station_name,
        plots_dir=plots_dir,
        plot_cfg=plot_cfg,
        cfg=ext_cfg,
        prefix="ext_",
        title_prefix="extrapolated triple/quad-family",
    )
    if ext_path is not None:
        created.append(ext_path)

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
