#!/usr/bin/env python3
"""TASK_2 calibration QA: grouped plots + dataset summary CSV."""

from __future__ import annotations

from ast import literal_eval
from datetime import date, datetime
from fnmatch import fnmatch
from pathlib import Path
import math
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
METADATA_SUFFIX = "calibration"
METADATA_TYPE = "calibration"
DEFAULT_PASS_COLUMN = "task_2_calibration_pass"

METADATA_COLUMNS_TO_SKIP = {"filename_base", "execution_timestamp", "param_hash"}


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


def _unique_preserve(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _has_glob(pattern: str) -> bool:
    return any(token in pattern for token in ("*", "?", "[", "]"))


def _resolve_columns(available_columns: list[str], patterns: list[str]) -> list[str]:
    resolved: list[str] = []
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip()
        if not pattern:
            continue
        if _has_glob(pattern):
            resolved.extend([col for col in available_columns if fnmatch(col, pattern)])
            continue
        if pattern in available_columns:
            resolved.append(pattern)
        else:
            # Keep explicit column names so a subplot still appears as missing.
            resolved.append(pattern)
    return _unique_preserve(resolved)


def _default_plot_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    patterns = config.get("column_patterns")
    if not isinstance(patterns, list):
        patterns = []
    filtered = [
        str(p).strip()
        for p in patterns
        if str(p).strip() and str(p).strip() not in METADATA_COLUMNS_TO_SKIP
    ]
    if not filtered:
        filtered = ["P[1-4]_s[1-4]_*"]
    return [{"name": "calibration", "columns": filtered}]


def _plot_groups_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    groups_cfg = plot_cfg.get("plot_groups")
    if not isinstance(groups_cfg, list) or not groups_cfg:
        return _default_plot_groups(config)

    groups: list[dict[str, Any]] = []
    for idx, item in enumerate(groups_cfg, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"group_{idx}")).strip() or f"group_{idx}"
        columns = item.get("columns")
        if not isinstance(columns, list):
            continue
        patterns = [str(c).strip() for c in columns if str(c).strip()]
        if not patterns:
            continue
        group: dict[str, Any] = {"name": name, "columns": patterns}
        if "ncols" in item:
            try:
                group["ncols"] = int(item.get("ncols"))
            except (TypeError, ValueError):
                pass

        pair_suffixes_raw = item.get("pair_suffixes")
        if isinstance(pair_suffixes_raw, list):
            pair_suffixes = [str(value).strip() for value in pair_suffixes_raw if str(value).strip()]
            if len(pair_suffixes) == 2 and len(set(pair_suffixes)) == 2:
                group["pair_suffixes"] = pair_suffixes
                pair_labels_raw = item.get("pair_labels")
                if isinstance(pair_labels_raw, list):
                    pair_labels = [str(value).strip() for value in pair_labels_raw if str(value).strip()]
                    if len(pair_labels) == 2:
                        group["pair_labels"] = pair_labels
        groups.append(group)
    return groups or _default_plot_groups(config)


def _parse_coeff_list(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    parsed: Any
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = literal_eval(text)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(parsed, (list, tuple, np.ndarray)):
        return None

    numbers: list[float] = []
    for item in parsed:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            numbers.append(np.nan)
    return np.asarray(numbers, dtype=float)


def _iqr_inlier_mask(values: np.ndarray, iqr_multiplier: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros(values.shape, dtype=bool)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.ones(values.shape, dtype=bool)
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    return (values >= lower) & (values <= upper)


def _aggregate_numeric(values: np.ndarray, summary_cfg: dict[str, Any]) -> float:
    finite_mask = np.isfinite(values)
    sample = values[finite_mask]
    if sample.size == 0:
        return math.nan

    method = str(summary_cfg.get("method", "robust_median")).strip().lower()
    outlier_cfg = summary_cfg.get("outlier_filter", {})
    iqr_multiplier = float(outlier_cfg.get("iqr_multiplier", 1.5))
    min_points = int(outlier_cfg.get("min_points", 8))

    if method.startswith("robust") and sample.size >= min_points:
        inlier_mask = _iqr_inlier_mask(sample, iqr_multiplier=iqr_multiplier)
        if inlier_mask.any():
            sample = sample[inlier_mask]

    if method in {"mean", "robust_mean"}:
        return float(np.mean(sample))
    if method in {"median", "robust_median"}:
        return float(np.median(sample))

    raise ValueError(
        "Invalid summary.method. Use one of: mean, median, robust_mean, robust_median."
    )


def _summarize_column(series: pd.Series, summary_cfg: dict[str, Any]) -> float | list[float] | None:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(numeric).any():
        value = _aggregate_numeric(numeric, summary_cfg)
        return None if math.isnan(value) else value

    parsed = [_parse_coeff_list(item) for item in series.tolist()]
    max_len = max((len(arr) for arr in parsed if arr is not None), default=0)
    if max_len == 0:
        return None

    matrix = np.full((len(parsed), max_len), np.nan, dtype=float)
    for row_idx, arr in enumerate(parsed):
        if arr is None:
            continue
        matrix[row_idx, : len(arr)] = arr

    summary_values: list[float] = []
    for coeff_idx in range(max_len):
        value = _aggregate_numeric(matrix[:, coeff_idx], summary_cfg)
        summary_values.append(value)

    if not any(math.isfinite(item) for item in summary_values):
        return None
    return summary_values


def _serialize_summary_value(value: float | list[float] | None) -> Any:
    if value is None:
        return ""

    if isinstance(value, (list, tuple, np.ndarray)):
        parts: list[str] = []
        for item in value:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                parts.append("")
                continue
            parts.append("" if not math.isfinite(numeric) else f"{numeric:.10g}")
        return "[" + ", ".join(parts) + "]"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(numeric) else numeric


def _draw_summary_lines(
    *,
    ax: Any,
    summary_value: float | list[float] | None,
    line_width: float,
) -> None:
    if summary_value is None:
        return

    values: list[float] = []
    if isinstance(summary_value, (list, tuple, np.ndarray)):
        for item in summary_value:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
    else:
        try:
            numeric = float(summary_value)
        except (TypeError, ValueError):
            return
        if math.isfinite(numeric):
            values.append(numeric)

    for value in values:
        ax.axhline(
            value,
            color="red",
            linestyle="--",
            linewidth=max(0.9, line_width),
            alpha=0.85,
            zorder=2,
        )


def _format_scalar_3sig(value: float, *, allow_scientific: bool) -> str:
    if not math.isfinite(value):
        return ""
    if allow_scientific:
        return f"{value:.3g}"

    if value == 0:
        return "0"

    abs_value = abs(value)
    digits_before_decimal = int(math.floor(math.log10(abs_value))) + 1
    decimals = max(0, 3 - digits_before_decimal)
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _format_summary_for_title(summary_value: float | list[float] | None) -> str:
    if summary_value is None:
        return ""

    if isinstance(summary_value, (list, tuple, np.ndarray)):
        parts: list[str] = []
        for item in summary_value:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                parts.append(_format_scalar_3sig(numeric, allow_scientific=True))
        if not parts:
            return ""
        return "[" + ", ".join(parts) + "]"

    try:
        numeric = float(summary_value)
    except (TypeError, ValueError):
        return ""
    return _format_scalar_3sig(numeric, allow_scientific=False)


def _plot_column_series(
    *,
    ax: Any,
    x_values: pd.Series,
    df: pd.DataFrame,
    column_name: str,
    marker_size: float,
    line_width: float,
) -> None:
    if column_name not in df.columns:
        ax.text(0.5, 0.5, "missing column", ha="center", va="center", color="red", fontsize=9)
        return

    series = df[column_name]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        ax.plot(
            x_values,
            numeric,
            linestyle="-",
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            color="#1f77b4",
        )
        return

    parsed = [_parse_coeff_list(item) for item in series.tolist()]
    max_len = max((len(arr) for arr in parsed if arr is not None), default=0)
    if max_len == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", fontsize=9)
        return

    matrix = np.full((len(parsed), max_len), np.nan, dtype=float)
    for row_idx, arr in enumerate(parsed):
        if arr is None:
            continue
        matrix[row_idx, : len(arr)] = arr

    for coeff_idx in range(max_len):
        ax.plot(
            x_values,
            matrix[:, coeff_idx],
            linestyle="-",
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            label=f"c{coeff_idx}",
        )
    if max_len <= 8:
        ax.legend(fontsize=7, ncol=min(4, max_len), loc="best")


def _auto_grid(n_panels: int, ncols_hint: int | None) -> tuple[int, int]:
    if n_panels <= 0:
        return 1, 1
    if ncols_hint is not None and ncols_hint > 0:
        ncols = min(n_panels, ncols_hint)
        nrows = int(math.ceil(n_panels / ncols))
        return nrows, ncols
    ncols = int(math.ceil(math.sqrt(n_panels)))
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def _base_for_suffix_pair(column_name: str, suffix: str) -> str | None:
    token = f"_{suffix}"
    if not column_name.endswith(token):
        return None
    base_name = column_name[: -len(token)]
    return base_name or None


def _plot_pair_series(
    *,
    ax: Any,
    x_values: pd.Series,
    y_values: pd.Series,
    label: str,
    color: str,
    line_width: float,
    marker_size: float,
) -> bool:
    numeric = pd.to_numeric(y_values, errors="coerce")
    valid_mask = numeric.notna()
    if not valid_mask.any():
        return False
    ax.plot(
        x_values.loc[valid_mask],
        numeric.loc[valid_mask],
        linestyle="-",
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        color=color,
        label=label,
    )
    return True


def _plot_group_suffix_pairs(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    columns: list[str],
    station_name: str,
    group_name: str,
    out_dir: Path,
    config: dict[str, Any],
    pair_suffixes: list[str],
    pair_labels: list[str] | None = None,
    group_ncols: int | None = None,
) -> Path | None:
    plot_columns = [col for col in columns if col not in METADATA_COLUMNS_TO_SKIP]
    if not plot_columns:
        return None

    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    ncols_hint = group_ncols if group_ncols is not None else plot_cfg.get("ncols")
    try:
        ncols_hint_int = int(ncols_hint) if ncols_hint is not None else None
    except (TypeError, ValueError):
        ncols_hint_int = None

    suffix_to_label = {
        pair_suffixes[0]: pair_labels[0] if pair_labels is not None else pair_suffixes[0],
        pair_suffixes[1]: pair_labels[1] if pair_labels is not None else pair_suffixes[1],
    }
    suffix_to_color = {pair_suffixes[0]: "#1f77b4", pair_suffixes[1]: "#ff7f0e"}

    ordered_bases: list[str] = []
    pairs_by_base: dict[str, dict[str, str]] = {}
    for column_name in plot_columns:
        matched_suffix: str | None = None
        base_name: str | None = None
        for suffix in pair_suffixes:
            base_name = _base_for_suffix_pair(column_name, suffix)
            if base_name is not None:
                matched_suffix = suffix
                break
        if matched_suffix is None or base_name is None:
            continue
        if base_name not in pairs_by_base:
            pairs_by_base[base_name] = {}
            ordered_bases.append(base_name)
        pairs_by_base[base_name][matched_suffix] = column_name

    if not ordered_bases:
        return None

    nrows, ncols = _auto_grid(len(ordered_bases), ncols_hint_int)
    dpi = int(plot_cfg.get("dpi", 150))
    marker_size = float(plot_cfg.get("marker_size", 2.0))
    line_width = float(plot_cfg.get("line_width", 0.9))
    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    fig_width = float(plot_cfg.get("figsize_per_col", 5.0)) * ncols
    fig_height = float(plot_cfg.get("figsize_per_row", 3.0)) * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), sharex=True)
    axes_flat = np.ravel(axes)

    for idx, base_name in enumerate(ordered_bases):
        ax = axes_flat[idx]
        has_any = False
        missing_labels: list[str] = []
        pair_map = pairs_by_base.get(base_name, {})
        for suffix in pair_suffixes:
            label = suffix_to_label[suffix]
            column_name = pair_map.get(suffix)
            if column_name is None:
                missing_labels.append(label)
                continue
            plotted = _plot_pair_series(
                ax=ax,
                x_values=x_values,
                y_values=df[column_name],
                label=label,
                color=suffix_to_color[suffix],
                line_width=line_width,
                marker_size=marker_size,
            )
            has_any = plotted or has_any

        if not has_any:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", fontsize=9)
        if missing_labels:
            ax.text(
                0.5,
                0.08,
                f"missing: {', '.join(missing_labels)}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                color="red",
                fontsize=7,
            )

        ax.set_title(base_name, fontsize=9)
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="best")

    for idx in range(len(ordered_bases), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"TASK_{TASK_ID} {station_name} {METADATA_TYPE} - {group_name} (x={x_label})", fontsize=11)
    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_{group_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_group(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    columns: list[str],
    station_name: str,
    group_name: str,
    out_dir: Path,
    config: dict[str, Any],
    group_ncols: int | None = None,
) -> Path | None:
    plot_columns = [col for col in columns if col not in METADATA_COLUMNS_TO_SKIP]
    if not plot_columns:
        return None

    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    summary_cfg = config.get("summary") if isinstance(config.get("summary"), dict) else {}
    ncols_hint = group_ncols if group_ncols is not None else plot_cfg.get("ncols")
    try:
        ncols_hint_int = int(ncols_hint) if ncols_hint is not None else None
    except (TypeError, ValueError):
        ncols_hint_int = None
    nrows, ncols = _auto_grid(len(plot_columns), ncols_hint_int)

    dpi = int(plot_cfg.get("dpi", 150))
    marker_size = float(plot_cfg.get("marker_size", 2.0))
    line_width = float(plot_cfg.get("line_width", 0.9))
    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    fig_width = float(plot_cfg.get("figsize_per_col", 5.0)) * ncols
    fig_height = float(plot_cfg.get("figsize_per_row", 3.0)) * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), sharex=True)
    axes_flat = np.ravel(axes)

    for idx, column_name in enumerate(plot_columns):
        ax = axes_flat[idx]
        title_text = column_name
        _plot_column_series(
            ax=ax,
            x_values=x_values,
            df=df,
            column_name=column_name,
            marker_size=marker_size,
            line_width=line_width,
        )
        if column_name in df.columns:
            summary_value = _summarize_column(df[column_name], summary_cfg)
            _draw_summary_lines(ax=ax, summary_value=summary_value, line_width=line_width)
            summary_label = _format_summary_for_title(summary_value)
            if summary_label:
                title_text = f"{column_name}\np={summary_label}"
        ax.set_title(title_text, fontsize=9)
        ax.grid(True, alpha=0.25)

    for idx in range(len(plot_columns), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"TASK_{TASK_ID} {station_name} {METADATA_TYPE} - {group_name} (x={x_label})", fontsize=11)
    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_{group_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _summary_columns_from_groups(
    *,
    groups: list[dict[str, Any]],
    available_columns: list[str],
    summary_cfg: dict[str, Any],
) -> list[str]:
    if isinstance(summary_cfg.get("columns"), list):
        patterns = [str(item).strip() for item in summary_cfg["columns"] if str(item).strip()]
        return _resolve_columns(available_columns, patterns)

    ordered: list[str] = []
    for group in groups:
        ordered.extend(_resolve_columns(available_columns, group.get("columns", [])))
    return _unique_preserve([col for col in ordered if col not in METADATA_COLUMNS_TO_SKIP])


def _write_dataset_summary(
    *,
    station_name: str,
    analyzed_df: pd.DataFrame,
    groups: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path | None:
    if analyzed_df.empty:
        return None

    summary_cfg = config.get("summary") if isinstance(config.get("summary"), dict) else {}
    summary_columns = _summary_columns_from_groups(
        groups=groups,
        available_columns=list(analyzed_df.columns),
        summary_cfg=summary_cfg,
    )
    if not summary_columns:
        return None

    summary_row: dict[str, Any] = {}
    for column_name in summary_columns:
        if column_name not in analyzed_df.columns:
            summary_row[column_name] = ""
            continue
        summary_row[column_name] = _serialize_summary_value(
            _summarize_column(analyzed_df[column_name], summary_cfg)
        )

    output_filename = str(summary_cfg.get("output_filename", "calibrations_task_2.csv")).strip()
    if not output_filename:
        output_filename = "calibrations_task_2.csv"
    out_path = _output_files_dir(station_name) / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary_row], columns=summary_columns).to_csv(out_path, index=False)
    return out_path


def _generate_station_plots(station_name: str, config: dict[str, Any], meta_df: pd.DataFrame) -> tuple[list[Path], pd.DataFrame]:
    if meta_df.empty:
        return [], pd.DataFrame()

    df = meta_df.copy()
    df["__timestamp__"], timestamp_source = _resolve_filter_timestamp(df, config)
    df = df.loc[df["__timestamp__"].notna()].copy()
    if df.empty:
        print(f"TASK_{TASK_ID} {station_name}: no rows with valid timestamp source '{timestamp_source}'.")
        return [], pd.DataFrame()

    date_range = get_station_date_range(config=config, station=station_name)
    df = _apply_date_filter(df, date_range)
    if df.empty:
        print(f"TASK_{TASK_ID} {station_name}: all rows filtered out by date_range={date_range}.")
        return [], pd.DataFrame()

    x_values, x_label, x_is_datetime = _resolve_plot_x_axis(df, config)
    df["__plot_x__"] = x_values
    df = df.loc[df["__plot_x__"].notna()].copy()
    if df.empty:
        print(f"TASK_{TASK_ID} {station_name}: no rows with valid x-axis values.")
        return [], pd.DataFrame()

    sort_key = "__plot_x__" if x_label != "__timestamp__" else "__timestamp__"
    df = df.sort_values(sort_key).reset_index(drop=True)
    x_values = df["__plot_x__"]

    groups = _plot_groups_from_config(config)
    plots_dir = _output_plots_dir(station_name)
    created: list[Path] = []
    for group in groups:
        group_name = group["name"]
        columns = _resolve_columns(list(df.columns), group.get("columns", []))
        pair_suffixes = group.get("pair_suffixes")
        if isinstance(pair_suffixes, list) and len(pair_suffixes) == 2:
            out_path = _plot_group_suffix_pairs(
                df=df,
                x_values=x_values,
                x_label=x_label,
                x_is_datetime=x_is_datetime,
                columns=columns,
                station_name=station_name,
                group_name=group_name,
                out_dir=plots_dir,
                config=config,
                pair_suffixes=pair_suffixes,
                pair_labels=group.get("pair_labels"),
                group_ncols=group.get("ncols"),
            )
        else:
            out_path = _plot_group(
                df=df,
                x_values=x_values,
                x_label=x_label,
                x_is_datetime=x_is_datetime,
                columns=columns,
                station_name=station_name,
                group_name=group_name,
                out_dir=plots_dir,
                config=config,
                group_ncols=group.get("ncols"),
            )
        if out_path is not None:
            created.append(out_path)

    return created, df


def main() -> int:
    config = load_task_configs(TASK_DIR)
    bootstrap_task(TASK_DIR)

    stations = config.get("stations", [0, 1, 2, 3, 4])
    pass_column = str(config.get("pass_column_name", DEFAULT_PASS_COLUMN))
    default_pass = float(config.get("pass_default_value", 1.0))

    total_rows = 0
    written = 0
    total_plots = 0
    total_summaries = 0

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
        pass_path = files_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_pass.csv"
        pass_df.to_csv(pass_path, index=False)
        written += 1

        try:
            created_plots, analyzed_df = _generate_station_plots(
                station_name=station_name,
                config=config,
                meta_df=meta_df,
            )
            total_plots += len(created_plots)

            summary_path = _write_dataset_summary(
                station_name=station_name,
                analyzed_df=analyzed_df,
                groups=_plot_groups_from_config(config),
                config=config,
            )
            if summary_path is not None:
                total_summaries += 1
            if created_plots or summary_path is not None:
                print(
                    f"TASK_{TASK_ID} {station_name}: plots={len(created_plots)} "
                    f"summary={'yes' if summary_path is not None else 'no'}"
                )
        except (KeyError, ValueError) as exc:
            print(f"TASK_{TASK_ID} {station_name}: plotting/summary skipped ({exc})")

    print(
        f"TASK_{TASK_ID} {METADATA_TYPE} complete: "
        f"stations={written} total_rows={total_rows} pass_column={pass_column} "
        f"plots={total_plots} summaries={total_summaries}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
