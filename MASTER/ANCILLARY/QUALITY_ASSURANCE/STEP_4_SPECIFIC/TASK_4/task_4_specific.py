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
EFFICIENCY_MISSING_RATE_COLUMNS = {
    1: "list_tt_234_rate_hz",
    2: "list_tt_134_rate_hz",
    3: "list_tt_124_rate_hz",
    4: "list_tt_123_rate_hz",
}
EFFICIENCY_BASE_RATE_COLUMN = "list_tt_1234_rate_hz"
STREAMER_EVENT_COUNT_COLUMN = "activation_plane_streamer_to_signal_filtered_by_tt_tt1234_event_count"
STREAMER_GIVEN_COUNT_COLUMNS = {
    plane: f"activation_plane_streamer_to_signal_filtered_by_tt_tt1234_given_count_P{plane}"
    for plane in range(1, 5)
}
PURITY_SOURCES = (
    ("TASK 1", 1, "execution"),
    ("TASK 2", 2, "filter"),
    ("TASK 3", 3, "filter"),
    ("TASK 4", 4, "filter"),
)


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


def _task_metadata_path(station_name: str, task_id: int, suffix: str) -> Path:
    return (
        REPO_ROOT
        / "STATIONS"
        / station_name
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{task_id}"
        / "METADATA"
        / f"task_{task_id}_metadata_{suffix}.csv"
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


def _load_latest_metadata_csv(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame(columns=["filename_base"])
    effective_usecols = usecols
    if usecols is not None:
        available_columns = pd.read_csv(path, nrows=0).columns.tolist()
        effective_usecols = [column for column in usecols if column in available_columns]
        if "filename_base" not in effective_usecols and "filename_base" in available_columns:
            effective_usecols = ["filename_base", *effective_usecols]
        if "execution_timestamp" in available_columns and "execution_timestamp" not in effective_usecols:
            effective_usecols.append("execution_timestamp")
    frame = pd.read_csv(path, usecols=effective_usecols, low_memory=False)
    if frame.empty or "filename_base" not in frame.columns:
        return pd.DataFrame(columns=["filename_base"])
    frame = frame.copy()
    frame["filename_base"] = frame["filename_base"].astype("string").fillna("").str.strip()
    frame = frame[frame["filename_base"] != ""].copy()
    if frame.empty:
        return pd.DataFrame(columns=["filename_base"])
    if "execution_timestamp" in frame.columns:
        frame["_exec_ts"] = _parse_timestamp_series(frame["execution_timestamp"])
        frame = frame.sort_values(["filename_base", "_exec_ts"], na_position="last")
        frame = frame.drop_duplicates(subset=["filename_base"], keep="last")
        frame = frame.drop(columns=["_exec_ts"])
    else:
        frame = frame.drop_duplicates(subset=["filename_base"], keep="last")
    return frame.reset_index(drop=True)


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


def _correlation_plot_config(config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("global_x_correlations"), dict) or "global_x_correlations" in config:
        plots_cfg = config
    else:
        plots_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    raw = plots_cfg.get("global_x_correlations")
    if not isinstance(raw, dict):
        raw = {}
    return raw


def _compute_correlation_stats(x: pd.Series, y: pd.Series) -> tuple[int, float | None, float | None]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna()
    if int(mask.sum()) < 2:
        return int(mask.sum()), None, None
    xv = x_num.loc[mask].to_numpy(dtype=float)
    yv = y_num.loc[mask].to_numpy(dtype=float)
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return int(mask.sum()), None, None
    pearson = float(np.corrcoef(xv, yv)[0, 1])
    slope = float(np.polyfit(xv, yv, 1)[0])
    return int(mask.sum()), pearson, slope


def _draw_scatter_with_trend(
    *,
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_values: pd.Series,
    cmap: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> object | None:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        return None

    scatter = ax.scatter(
        x.loc[mask],
        y.loc[mask],
        c=color_values.loc[mask],
        cmap=cmap,
        s=24,
        alpha=0.90,
        edgecolors="none",
    )

    n_points, pearson, slope = _compute_correlation_stats(x, y)
    if n_points >= 2 and pearson is not None:
        xv = x.loc[mask].to_numpy(dtype=float)
        yv = y.loc[mask].to_numpy(dtype=float)
        fit = np.polyfit(xv, yv, 1)
        x_line = np.linspace(float(np.nanmin(xv)), float(np.nanmax(xv)), 100)
        y_line = fit[0] * x_line + fit[1]
        ax.plot(x_line, y_line, color="black", lw=1.0, ls="--")
        title = f"{title}\nr={pearson:.3f}, slope={slope:.3f}, n={n_points}"
    else:
        title = f"{title}\nn={n_points}"

    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return scatter


def _build_global_x_correlation_frame(station_name: str, task4_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "filename_base",
        "execution_timestamp",
        "timtrack_projection_scaling_factor_xproj_global",
    ]
    task4_base = task4_df[base_columns].copy()
    task4_base.rename(
        columns={"timtrack_projection_scaling_factor_xproj_global": "global_x_scale_factor"},
        inplace=True,
    )
    task4_base["global_x_scale_factor"] = pd.to_numeric(
        task4_base["global_x_scale_factor"], errors="coerce"
    )
    task4_base["__timestamp__"] = _parse_filename_timestamp_series(task4_base["filename_base"])
    task4_base = task4_base.sort_values(["__timestamp__", "filename_base"], na_position="last").reset_index(drop=True)
    task4_base["__time_order__"] = np.arange(1, len(task4_base) + 1, dtype=int)

    trigger_cols = ["filename_base", "execution_timestamp", EFFICIENCY_BASE_RATE_COLUMN]
    trigger_cols.extend(EFFICIENCY_MISSING_RATE_COLUMNS.values())
    trigger_df = _load_latest_metadata_csv(
        _task_metadata_path(station_name, 3, "trigger_type"),
        usecols=trigger_cols,
    )
    for plane, missing_col in EFFICIENCY_MISSING_RATE_COLUMNS.items():
        base_rate = pd.to_numeric(trigger_df.get(EFFICIENCY_BASE_RATE_COLUMN), errors="coerce")
        missing_rate = pd.to_numeric(trigger_df.get(missing_col), errors="coerce")
        trigger_df[f"plane_efficiency_p{plane}"] = np.where(
            np.isfinite(base_rate) & np.isfinite(missing_rate) & (base_rate > 0),
            1.0 - (missing_rate / base_rate),
            np.nan,
        )
    keep_trigger_cols = ["filename_base"] + [f"plane_efficiency_p{plane}" for plane in range(1, 5)]
    task4_base = task4_base.merge(trigger_df[keep_trigger_cols], on="filename_base", how="left")

    activation_cols = ["filename_base", "execution_timestamp", STREAMER_EVENT_COUNT_COLUMN]
    activation_cols.extend(STREAMER_GIVEN_COUNT_COLUMNS.values())
    activation_df = _load_latest_metadata_csv(
        _task_metadata_path(station_name, 3, "activation"),
        usecols=activation_cols,
    )
    streamer_event_count = pd.to_numeric(
        activation_df.get(STREAMER_EVENT_COUNT_COLUMN), errors="coerce"
    )
    streamer_pct_cols: list[str] = []
    for plane, given_col in STREAMER_GIVEN_COUNT_COLUMNS.items():
        streamer_col = f"streamer_percentage_p{plane}"
        activation_df[streamer_col] = np.where(
            np.isfinite(streamer_event_count)
            & (streamer_event_count > 0)
            & pd.to_numeric(activation_df.get(given_col), errors="coerce").notna(),
            100.0
            * pd.to_numeric(activation_df.get(given_col), errors="coerce")
            / streamer_event_count,
            np.nan,
        )
        streamer_pct_cols.append(streamer_col)
    activation_df["streamer_percentage_mean"] = activation_df[streamer_pct_cols].mean(axis=1)
    task4_base = task4_base.merge(
        activation_df[["filename_base", "streamer_percentage_mean", *streamer_pct_cols]],
        on="filename_base",
        how="left",
    )

    for label, task_id, suffix in PURITY_SOURCES:
        purity_df = _load_latest_metadata_csv(
            _task_metadata_path(station_name, task_id, suffix),
            usecols=["filename_base", "execution_timestamp", "data_purity_percentage"],
        )
        purity_col = f"task_{task_id}_data_purity_percentage"
        if "data_purity_percentage" in purity_df.columns:
            purity_df = purity_df[["filename_base", "data_purity_percentage"]].rename(
                columns={"data_purity_percentage": purity_col}
            )
            task4_base = task4_base.merge(purity_df, on="filename_base", how="left")
        else:
            task4_base[purity_col] = np.nan

    return task4_base


def _build_global_x_correlation_summary(correlation_df: pd.DataFrame) -> pd.DataFrame:
    specs = [(f"plane_efficiency_p{plane}", f"Plane {plane} efficiency") for plane in range(1, 5)]
    specs.append(("streamer_percentage_mean", "Mean filtered tt1234 streamer percentage"))
    specs.extend(
        (f"task_{task_id}_data_purity_percentage", f"{label} data purity percentage")
        for label, task_id, _ in PURITY_SOURCES
    )
    rows: list[dict[str, Any]] = []
    for column, label in specs:
        if column not in correlation_df.columns:
            continue
        x = pd.to_numeric(correlation_df[column], errors="coerce")
        y = pd.to_numeric(correlation_df["global_x_scale_factor"], errors="coerce")
        mask = x.notna() & y.notna()
        if not mask.any():
            rows.append(
                {
                    "column": column,
                    "label": label,
                    "n_points": 0,
                    "pearson_r": np.nan,
                    "slope": np.nan,
                    "x_min": np.nan,
                    "x_max": np.nan,
                    "y_min": np.nan,
                    "y_max": np.nan,
                }
            )
            continue
        n_points, pearson, slope = _compute_correlation_stats(x, y)
        rows.append(
            {
                "column": column,
                "label": label,
                "n_points": n_points,
                "pearson_r": pearson,
                "slope": slope,
                "x_min": float(np.nanmin(x.loc[mask].to_numpy(dtype=float))),
                "x_max": float(np.nanmax(x.loc[mask].to_numpy(dtype=float))),
                "y_min": float(np.nanmin(y.loc[mask].to_numpy(dtype=float))),
                "y_max": float(np.nanmax(y.loc[mask].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def _plot_global_x_vs_efficiencies(
    *,
    station_name: str,
    plots_dir: Path,
    correlation_df: pd.DataFrame,
    plot_cfg: dict[str, Any],
) -> Path | None:
    corr_cfg = _correlation_plot_config(plot_cfg)
    eff_cfg = corr_cfg.get("efficiency") if isinstance(corr_cfg.get("efficiency"), dict) else {}
    if corr_cfg.get("enabled", True) is False or eff_cfg.get("enabled", True) is False:
        return None

    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    dpi = int(plot_cfg.get("dpi", 150))
    cmap = str(corr_cfg.get("cmap", "viridis")).strip() or "viridis"
    figsize = tuple(eff_cfg.get("figsize", [11, 8]))
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    color_values = correlation_df["__time_order__"]
    scatter_ref = None
    for ax, plane in zip(axes.flat, range(1, 5)):
        scatter = _draw_scatter_with_trend(
            ax=ax,
            df=correlation_df,
            x_col=f"plane_efficiency_p{plane}",
            y_col="global_x_scale_factor",
            color_values=color_values,
            cmap=cmap,
            title=f"Global x scale vs eff P{plane}",
            xlabel=f"Plane {plane} efficiency",
            ylabel="Global x scale factor",
        )
        if scatter_ref is None and scatter is not None:
            scatter_ref = scatter
    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("file order")
    fig.suptitle(f"TASK_{TASK_ID} {station_name} QA - global x scale vs plane efficiencies", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.90, wspace=0.20, hspace=0.30)

    plots_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(eff_cfg.get("filename", "global_x_vs_plane_efficiency")).strip() or "global_x_vs_plane_efficiency"
    out_path = plots_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_{base_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_global_x_vs_streamer(
    *,
    station_name: str,
    plots_dir: Path,
    correlation_df: pd.DataFrame,
    plot_cfg: dict[str, Any],
) -> Path | None:
    corr_cfg = _correlation_plot_config(plot_cfg)
    streamer_cfg = corr_cfg.get("streamer") if isinstance(corr_cfg.get("streamer"), dict) else {}
    if corr_cfg.get("enabled", True) is False or streamer_cfg.get("enabled", True) is False:
        return None

    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    dpi = int(plot_cfg.get("dpi", 150))
    cmap = str(corr_cfg.get("cmap", "viridis")).strip() or "viridis"
    figsize = tuple(streamer_cfg.get("figsize", [7, 5.5]))

    fig, ax = plt.subplots(figsize=figsize)
    scatter = _draw_scatter_with_trend(
        ax=ax,
        df=correlation_df,
        x_col="streamer_percentage_mean",
        y_col="global_x_scale_factor",
        color_values=correlation_df["__time_order__"],
        cmap=cmap,
        title="Global x scale vs streamer %",
        xlabel="Mean filtered tt1234 streamer percentage [%]",
        ylabel="Global x scale factor",
    )
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("file order")
    fig.suptitle(
        f"TASK_{TASK_ID} {station_name} QA - global x scale vs streamer fraction\n"
        "Streamer % = mean per-plane streamer occupancy within filtered tt1234 events",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    plots_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(streamer_cfg.get("filename", "global_x_vs_streamer_pct")).strip() or "global_x_vs_streamer_pct"
    out_path = plots_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_{base_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_global_x_vs_purity(
    *,
    station_name: str,
    plots_dir: Path,
    correlation_df: pd.DataFrame,
    plot_cfg: dict[str, Any],
) -> Path | None:
    corr_cfg = _correlation_plot_config(plot_cfg)
    purity_cfg = corr_cfg.get("purity") if isinstance(corr_cfg.get("purity"), dict) else {}
    if corr_cfg.get("enabled", True) is False or purity_cfg.get("enabled", True) is False:
        return None

    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    dpi = int(plot_cfg.get("dpi", 150))
    cmap = str(corr_cfg.get("cmap", "viridis")).strip() or "viridis"
    figsize = tuple(purity_cfg.get("figsize", [11, 8]))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    color_values = correlation_df["__time_order__"]
    scatter_ref = None
    for ax, (label, task_id, _) in zip(axes.flat, PURITY_SOURCES):
        scatter = _draw_scatter_with_trend(
            ax=ax,
            df=correlation_df,
            x_col=f"task_{task_id}_data_purity_percentage",
            y_col="global_x_scale_factor",
            color_values=color_values,
            cmap=cmap,
            title=f"Global x scale vs {label} purity",
            xlabel=f"{label} data purity [%]",
            ylabel="Global x scale factor",
        )
        if scatter_ref is None and scatter is not None:
            scatter_ref = scatter
    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("file order")
    fig.suptitle(f"TASK_{TASK_ID} {station_name} QA - global x scale vs data purity", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.90, wspace=0.20, hspace=0.30)

    plots_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(purity_cfg.get("filename", "global_x_vs_purity")).strip() or "global_x_vs_purity"
    out_path = plots_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_{base_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


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
    files_dir = _output_files_dir(station_name)

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

    if "timtrack_projection_scaling_factor_xproj_global" in df.columns:
        correlation_df = _build_global_x_correlation_frame(station_name, df)
        if not correlation_df.empty:
            files_dir.mkdir(parents=True, exist_ok=True)
            correlation_df.to_csv(
                files_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_global_x_correlation_inputs.csv",
                index=False,
            )
            summary_df = _build_global_x_correlation_summary(correlation_df)
            summary_df.to_csv(
                files_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_global_x_correlation_summary.csv",
                index=False,
            )

            eff_path = _plot_global_x_vs_efficiencies(
                station_name=station_name,
                plots_dir=plots_dir,
                correlation_df=correlation_df,
                plot_cfg=plot_cfg,
            )
            if eff_path is not None:
                created.append(eff_path)

            streamer_path = _plot_global_x_vs_streamer(
                station_name=station_name,
                plots_dir=plots_dir,
                correlation_df=correlation_df,
                plot_cfg=plot_cfg,
            )
            if streamer_path is not None:
                created.append(streamer_path)

            purity_path = _plot_global_x_vs_purity(
                station_name=station_name,
                plots_dir=plots_dir,
                correlation_df=correlation_df,
                plot_cfg=plot_cfg,
            )
            if purity_path is not None:
                created.append(purity_path)

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
