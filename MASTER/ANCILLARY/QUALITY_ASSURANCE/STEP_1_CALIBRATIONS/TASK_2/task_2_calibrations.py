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
from matplotlib.colors import to_rgba
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
from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.column_rule_table import (
    ColumnRule,
    load_column_rule_table,
    matches_any_pattern,
    resolve_column_rule,
    rule_to_threshold_mapping,
    split_default_rule,
)
from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.epoch_quality import (
    build_epoch_reference_table,
    build_scalar_value_frame,
    evaluate_scalar_frame,
)
from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.epochs import load_online_run_dictionary
from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.thresholds import (
    ThresholdRule,
    compute_bounds,
    select_threshold_rule,
)
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


def _configured_plot_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    groups_cfg = plot_cfg.get("plot_groups")
    if not isinstance(groups_cfg, list) or not groups_cfg:
        return []

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
    return groups


def _column_rule_csv_path(config: dict[str, Any]) -> Path | None:
    raw_path = str(config.get("column_rule_table_csv", "")).strip()
    if not raw_path:
        return None
    csv_path = Path(raw_path)
    if not csv_path.is_absolute():
        csv_path = TASK_DIR / csv_path
    return csv_path


def _column_rule_table(config: dict[str, Any]) -> list[ColumnRule]:
    cached = config.get("__column_rule_table__")
    if isinstance(cached, list):
        return cached

    csv_path = _column_rule_csv_path(config)
    if csv_path is None or not csv_path.exists():
        rules: list[ColumnRule] = []
    else:
        rules = load_column_rule_table(csv_path)
    config["__column_rule_table__"] = rules
    return rules


def _plot_ignore_patterns(config: dict[str, Any]) -> list[str]:
    patterns: list[str] = []
    for key in ("plot_ignore_patterns_common", "plot_ignore_patterns_extra"):
        values = config.get(key)
        if not isinstance(values, list):
            continue
        patterns.extend(str(value).strip() for value in values if str(value).strip())
    return _unique_preserve(patterns)


def _should_ignore_column(config: dict[str, Any], column_name: str) -> bool:
    if column_name in METADATA_COLUMNS_TO_SKIP:
        return True
    return matches_any_pattern(column_name, _plot_ignore_patterns(config))


def _is_plot_candidate_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return True
    return _column_component_count(series) > 0


def _resolve_effective_column_rule(
    config: dict[str, Any],
    column_name: str,
    *,
    source_column: str | None = None,
) -> ColumnRule | None:
    rules = _column_rule_table(config)
    if not rules:
        return None

    direct_rule = resolve_column_rule(column_name, rules)
    if source_column is None or source_column == column_name:
        return direct_rule

    source_rule = resolve_column_rule(source_column, rules)
    if direct_rule is None:
        return source_rule
    if source_rule is None:
        return direct_rule
    if direct_rule.pattern == "*" and source_rule.pattern != "*":
        return source_rule
    return direct_rule


def _quality_rule_for_column(
    config: dict[str, Any],
    evaluation_column: str,
    *,
    source_column: str | None = None,
) -> ThresholdRule:
    resolved_rule = _resolve_effective_column_rule(
        config,
        evaluation_column,
        source_column=source_column,
    )
    if resolved_rule is not None:
        return ThresholdRule.from_mapping(rule_to_threshold_mapping(resolved_rule))

    quality_cfg = config.get("quality") if isinstance(config.get("quality"), dict) else {}
    rules_cfg = quality_cfg.get("rules") if isinstance(quality_cfg.get("rules"), dict) else {}
    return select_threshold_rule(
        evaluation_column,
        defaults=rules_cfg.get("defaults"),
        column_rules=rules_cfg.get("column_rules"),
    )


def _split_chunks(values: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        return [values]
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def _next_group_name(base_name: str, used_names: set[str]) -> str:
    candidate = base_name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    counter = 2
    while True:
        candidate = f"{base_name}_{counter}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def _auto_rule_plot_groups(
    config: dict[str, Any],
    df: pd.DataFrame,
    explicit_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    rules = _column_rule_table(config)
    if not rules:
        return []

    available_columns = list(df.columns)
    covered_columns: set[str] = set()
    for group in explicit_groups:
        covered_columns.update(_resolve_columns(available_columns, group.get("columns", [])))

    figure_columns: dict[str, list[str]] = {}
    unexpected_group_name = str(config.get("unexpected_plot_group_name", "unexpected")).strip() or "unexpected"
    used_group_names = {str(group.get("name", "")).strip() for group in explicit_groups if str(group.get("name", "")).strip()}

    for column_name in available_columns:
        if column_name in covered_columns or _should_ignore_column(config, column_name):
            continue
        if column_name not in df.columns or not _is_plot_candidate_series(df[column_name]):
            continue
        resolved_rule = _resolve_effective_column_rule(config, column_name)
        if resolved_rule is None or not resolved_rule.plot_enabled:
            continue
        figure_name = resolved_rule.figure or unexpected_group_name
        figure_columns.setdefault(figure_name, []).append(column_name)

    groups: list[dict[str, Any]] = []
    unexpected_chunk_size = int(config.get("unexpected_group_max_columns", 12))
    for figure_name, columns in figure_columns.items():
        if figure_name == unexpected_group_name and len(columns) > unexpected_chunk_size:
            for chunk in _split_chunks(columns, unexpected_chunk_size):
                group_name = _next_group_name(figure_name, used_group_names)
                groups.append({"name": group_name, "columns": chunk})
            continue
        group_name = _next_group_name(figure_name, used_group_names)
        groups.append({"name": group_name, "columns": columns})
    return groups


def _runtime_plot_groups(config: dict[str, Any], df: pd.DataFrame) -> list[dict[str, Any]]:
    explicit_groups = _configured_plot_groups(config)
    auto_groups = _auto_rule_plot_groups(config, df, explicit_groups)
    groups = explicit_groups + auto_groups
    if groups:
        return groups
    return _default_plot_groups(config)


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
            zorder=3,
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
            zorder=3,
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


def _plot_palette_color(index: int) -> str:
    palette = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    if not palette:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    return palette[index % len(palette)]


def _epoch_span_data(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_is_datetime: bool,
    reference_df: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if df.empty or "epoch_id" not in df.columns:
        return [], {}

    x_series = pd.Series(x_values, index=df.index)
    epoch_series = df["epoch_id"].astype("string")
    spans: list[dict[str, Any]] = []
    span_lookup: dict[str, dict[str, Any]] = {}

    if (
        x_is_datetime
        and reference_df is not None
        and not reference_df.empty
        and {"epoch_id", "start_date", "end_date"} <= set(reference_df.columns)
    ):
        x_datetime = pd.to_datetime(x_series, errors="coerce")
        x_min = x_datetime.min()
        x_max = x_datetime.max()
        if pd.notna(x_min) and pd.notna(x_max):
            epoch_table = reference_df[["epoch_id", "start_date", "end_date"]].drop_duplicates(subset=["epoch_id"]).copy()
            epoch_table["start_plot"] = pd.to_datetime(epoch_table["start_date"], errors="coerce")
            epoch_table["end_plot"] = (
                pd.to_datetime(epoch_table["end_date"], errors="coerce")
                + pd.Timedelta(days=1)
                - pd.Timedelta(microseconds=1)
            )
            epoch_table = epoch_table[
                epoch_table["start_plot"].notna()
                & epoch_table["end_plot"].notna()
                & (epoch_table["end_plot"] >= x_min)
                & (epoch_table["start_plot"] <= x_max)
            ].sort_values("start_plot")
            for shade_index, row in enumerate(epoch_table.itertuples(index=False), start=0):
                start_plot = max(row.start_plot, x_min)
                end_plot = min(row.end_plot, x_max)
                epoch_id = str(row.epoch_id)
                span = {"epoch_id": epoch_id, "start": start_plot, "end": end_plot, "shade_index": shade_index}
                spans.append(span)
                span_lookup[epoch_id] = span
            if spans:
                return spans, span_lookup

    working = pd.DataFrame({"epoch_id": epoch_series, "plot_x": x_series})
    working = working[working["epoch_id"].notna() & working["plot_x"].notna()].copy()
    if working.empty:
        return [], {}
    working = working.sort_values("plot_x").reset_index(drop=True)
    for shade_index, epoch_id in enumerate(pd.unique(working["epoch_id"]), start=0):
        group = working.loc[working["epoch_id"] == epoch_id, "plot_x"]
        if group.empty:
            continue
        span = {"epoch_id": str(epoch_id), "start": group.iloc[0], "end": group.iloc[-1], "shade_index": shade_index}
        spans.append(span)
        span_lookup[str(epoch_id)] = span
    return spans, span_lookup


def _draw_epoch_background(ax: Any, epoch_spans: list[dict[str, Any]]) -> None:
    if not epoch_spans:
        return
    shade_cycle = ("#fafafa", "#ececec")
    for idx, span in enumerate(epoch_spans):
        ax.axvspan(
            span["start"],
            span["end"],
            facecolor=shade_cycle[idx % len(shade_cycle)],
            alpha=0.55,
            zorder=0,
            linewidth=0,
        )
        if idx > 0:
            ax.axvline(span["start"], color="#d6d6d6", linewidth=0.8, alpha=0.7, zorder=0.2)


def _reference_center_and_band(
    reference_row: pd.Series | dict[str, Any],
    rule: Any,
) -> tuple[float, float, float] | None:
    center_key = "center_mean" if getattr(rule, "center_method", "median") == "mean" else "center_median"
    center_value = float(reference_row[center_key])
    if not math.isfinite(center_value):
        return None

    scale = None
    if getattr(rule, "tolerance_mode", "") == "mad_multiplier":
        scale = float(reference_row.get("scale_mad", math.nan))
    elif getattr(rule, "tolerance_mode", "") == "iqr_multiplier":
        scale = float(reference_row.get("scale_iqr", math.nan))
    elif getattr(rule, "tolerance_mode", "") == "zscore":
        scale = float(reference_row.get("scale_std", math.nan))

    try:
        lower, upper = compute_bounds(center_value, rule, scale=scale)
    except ValueError:
        return None

    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    return center_value, lower, upper


def _draw_reference_overlay(
    *,
    ax: Any,
    reference_df: pd.DataFrame | None,
    epoch_span_lookup: dict[str, dict[str, Any]],
    config: dict[str, Any],
    evaluation_column: str,
    source_column: str | None,
    color: str,
    line_width: float,
) -> None:
    if reference_df is None or reference_df.empty or evaluation_column not in set(reference_df.get("evaluation_column", [])):
        return

    rule = _quality_rule_for_column(config, evaluation_column, source_column=source_column)
    ref_rows = reference_df.loc[reference_df["evaluation_column"] == evaluation_column].copy()
    if "start_date" in ref_rows.columns:
        ref_rows = ref_rows.sort_values("start_date")

    band_color = to_rgba(color, 0.12)
    for row in ref_rows.to_dict("records"):
        span = epoch_span_lookup.get(str(row["epoch_id"]))
        if span is None:
            continue
        center_band = _reference_center_and_band(row, rule)
        if center_band is None:
            continue
        center_value, lower, upper = center_band
        xs = [span["start"], span["end"]]
        ax.fill_between(xs, [lower, lower], [upper, upper], color=band_color, zorder=1)
        ax.plot(xs, [center_value, center_value], linestyle="--", linewidth=max(1.0, line_width), color=color, alpha=0.95, zorder=2)
        ax.plot(xs, [lower, lower], linestyle=":", linewidth=max(0.8, line_width * 0.8), color=color, alpha=0.55, zorder=2)
        ax.plot(xs, [upper, upper], linestyle=":", linewidth=max(0.8, line_width * 0.8), color=color, alpha=0.55, zorder=2)


def _column_component_count(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return 1
    parsed = [_parse_coeff_list(item) for item in series.tolist()]
    return max((len(arr) for arr in parsed if arr is not None), default=0)


def _draw_column_reference_overlays(
    *,
    ax: Any,
    df: pd.DataFrame,
    column_name: str,
    reference_df: pd.DataFrame | None,
    epoch_span_lookup: dict[str, dict[str, Any]],
    config: dict[str, Any],
    line_width: float,
) -> None:
    if column_name not in df.columns:
        return

    component_count = _column_component_count(df[column_name])
    if component_count <= 0:
        return

    if component_count == 1:
        _draw_reference_overlay(
            ax=ax,
            reference_df=reference_df,
            epoch_span_lookup=epoch_span_lookup,
            config=config,
            evaluation_column=column_name,
            source_column=column_name,
            color=_plot_palette_color(0),
            line_width=line_width,
        )
        return

    for component_index in range(component_count):
        _draw_reference_overlay(
            ax=ax,
            reference_df=reference_df,
            epoch_span_lookup=epoch_span_lookup,
            config=config,
            evaluation_column=f"{column_name}__{component_index}",
            source_column=column_name,
            color=_plot_palette_color(component_index),
            line_width=line_width,
        )


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
        zorder=3,
    )
    return True


def _plot_group_suffix_pairs(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    columns: list[str],
    reference_df: pd.DataFrame | None,
    epoch_spans: list[dict[str, Any]],
    epoch_span_lookup: dict[str, dict[str, Any]],
    station_name: str,
    group_name: str,
    out_dir: Path,
    config: dict[str, Any],
    pair_suffixes: list[str],
    pair_labels: list[str] | None = None,
    group_ncols: int | None = None,
) -> Path | None:
    plot_columns = [col for col in columns if not _should_ignore_column(config, col)]
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
        _draw_epoch_background(ax, epoch_spans)
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
            _draw_reference_overlay(
                ax=ax,
                reference_df=reference_df,
                epoch_span_lookup=epoch_span_lookup,
                config=config,
                evaluation_column=column_name,
                source_column=column_name,
                color=suffix_to_color[suffix],
                line_width=line_width,
            )

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
    reference_df: pd.DataFrame | None,
    epoch_spans: list[dict[str, Any]],
    epoch_span_lookup: dict[str, dict[str, Any]],
    station_name: str,
    group_name: str,
    out_dir: Path,
    config: dict[str, Any],
    group_ncols: int | None = None,
) -> Path | None:
    plot_columns = [col for col in columns if not _should_ignore_column(config, col)]
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
        _draw_epoch_background(ax, epoch_spans)
        title_text = column_name
        _plot_column_series(
            ax=ax,
            x_values=x_values,
            df=df,
            column_name=column_name,
            marker_size=marker_size,
            line_width=line_width,
        )
        _draw_column_reference_overlays(
            ax=ax,
            df=df,
            column_name=column_name,
            reference_df=reference_df,
            epoch_span_lookup=epoch_span_lookup,
            config=config,
            line_width=line_width,
        )
        if column_name in df.columns:
            summary_value = _summarize_column(df[column_name], summary_cfg)
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
    config: dict[str, Any],
) -> list[str]:
    if isinstance(summary_cfg.get("columns"), list):
        patterns = [str(item).strip() for item in summary_cfg["columns"] if str(item).strip()]
        resolved = _resolve_columns(available_columns, patterns)
        return _unique_preserve(
            [col for col in resolved if col in available_columns and not _should_ignore_column(config, col)]
        )

    ordered: list[str] = []
    for group in groups:
        ordered.extend(_resolve_columns(available_columns, group.get("columns", [])))
    return _unique_preserve(
        [
            col
            for col in ordered
            if col in available_columns and not _should_ignore_column(config, col)
        ]
    )


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
        config=config,
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


def _load_metadata_frame(meta_path: Path) -> pd.DataFrame:
    if meta_path.exists() and meta_path.stat().st_size > 0:
        return pd.read_csv(meta_path, low_memory=False)
    return pd.DataFrame(columns=["filename_base"])


def _deduplicate_metadata_rows(meta_df: pd.DataFrame) -> pd.DataFrame:
    if meta_df.empty or "filename_base" not in meta_df.columns:
        return pd.DataFrame(columns=["filename_base"])

    df = meta_df.copy()
    df["filename_base"] = df["filename_base"].astype("string").fillna("").str.strip()
    df = df[df["filename_base"] != ""].copy()
    if df.empty:
        return pd.DataFrame(columns=["filename_base"])

    if "execution_timestamp" in df.columns:
        df["_exec_ts"] = _parse_timestamp_series(df["execution_timestamp"])
        df = df.sort_values(["filename_base", "_exec_ts"], na_position="last")
        df = df.drop_duplicates(subset=["filename_base"], keep="last")
        df = df.drop(columns=["_exec_ts"])
    else:
        df = df.drop_duplicates(subset=["filename_base"], keep="last")

    return df.reset_index(drop=True)


def _prepare_scope_dataframe(station_name: str, config: dict[str, Any], meta_df: pd.DataFrame) -> pd.DataFrame:
    df = _deduplicate_metadata_rows(meta_df)
    if df.empty:
        return df

    try:
        timestamps, timestamp_source = _resolve_filter_timestamp(df, config)
        df["__timestamp__"] = timestamps
        df["qa_timestamp_source"] = timestamp_source
    except (KeyError, ValueError):
        df["__timestamp__"] = pd.NaT
        df["qa_timestamp_source"] = ""

    date_range = get_station_date_range(config=config, station=station_name)
    if date_range is None:
        df["qa_in_scope"] = df["__timestamp__"].notna()
    else:
        in_scope = pd.Series(True, index=df.index)
        start = _parse_boundary(date_range.get("start"), end_of_day_if_date_only=False)
        end = _parse_boundary(date_range.get("end"), end_of_day_if_date_only=True)
        in_scope &= df["__timestamp__"].notna()
        if start is not None:
            in_scope &= df["__timestamp__"] >= start
        if end is not None:
            in_scope &= df["__timestamp__"] <= end
        df["qa_in_scope"] = in_scope
    return df


def _prepare_analysis_dataframe(station_name: str, config: dict[str, Any], meta_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if meta_df.empty:
        return pd.DataFrame(), "empty_metadata"

    scope_df = _prepare_scope_dataframe(station_name=station_name, config=config, meta_df=meta_df)
    if scope_df.empty:
        return pd.DataFrame(), "empty_metadata"

    df = scope_df.loc[scope_df["qa_in_scope"]].copy()
    if df.empty:
        date_range = get_station_date_range(config=config, station=station_name)
        return pd.DataFrame(), f"all rows filtered out by date_range={date_range}"

    x_values, x_label, x_is_datetime = _resolve_plot_x_axis(df, config)
    df["__plot_x__"] = x_values
    df = df.loc[df["__plot_x__"].notna()].copy()
    if df.empty:
        return pd.DataFrame(), "no rows with valid x-axis values"

    sort_key = "__plot_x__" if x_label != "__timestamp__" else "__timestamp__"
    df = df.sort_values(sort_key).reset_index(drop=True)
    return df, None


def _quality_columns_from_config(
    *,
    config: dict[str, Any],
    groups: list[dict[str, Any]],
    available_columns: list[str],
) -> list[str]:
    quality_cfg = config.get("quality") if isinstance(config.get("quality"), dict) else {}
    quality_columns = quality_cfg.get("columns")
    if isinstance(quality_columns, list):
        patterns = [str(item).strip() for item in quality_columns if str(item).strip()]
        return _resolve_columns(available_columns, patterns)
    rules = _column_rule_table(config)
    if rules:
        out: list[str] = []
        for column_name in available_columns:
            if _should_ignore_column(config, column_name):
                continue
            resolved_rule = _resolve_effective_column_rule(config, column_name)
            if resolved_rule is not None and resolved_rule.quality_enabled:
                out.append(column_name)
        return _unique_preserve(out)

    summary_cfg = config.get("summary") if isinstance(config.get("summary"), dict) else {}
    return _summary_columns_from_groups(
        groups=groups,
        available_columns=available_columns,
        summary_cfg=summary_cfg,
        config=config,
    )


def _attach_epoch_metadata(station_name: str, analyzed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if analyzed_df.empty or station_name == "MINGO00":
        df = analyzed_df.copy()
        df["epoch_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
        return df, pd.DataFrame()

    try:
        epochs_df = load_online_run_dictionary(REPO_ROOT, int(station_name.removeprefix("MINGO")))
    except (FileNotFoundError, ValueError):
        df = analyzed_df.copy()
        df["epoch_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
        return df, pd.DataFrame()

    df = analyzed_df.copy()
    epoch_records = []
    for timestamp in df["__timestamp__"]:
        match = epochs_df[
            epochs_df["start_timestamp"].notna()
            & (epochs_df["start_timestamp"] <= timestamp)
            & (epochs_df["end_timestamp"].isna() | (timestamp <= epochs_df["end_timestamp"]))
        ]
        epoch_records.append(None if match.empty else str(match.iloc[-1]["epoch_id"]))
    df["epoch_id"] = pd.array(epoch_records, dtype="string")
    return df, epochs_df


def _enrich_reference_table(reference_df: pd.DataFrame, epochs_df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    if reference_df.empty:
        return reference_df

    epoch_columns = [
        column
        for column in (
            "epoch_id",
            "conf_number",
            "start_date",
            "end_date",
            "location",
            "comment",
            "boundary_overlap",
        )
        if column in epochs_df.columns
    ]
    epoch_lookup = epochs_df[epoch_columns].drop_duplicates(subset=["epoch_id"])
    out = reference_df.merge(epoch_lookup, on="epoch_id", how="left")
    out.insert(0, "station_name", station_name)
    return out


def _write_epoch_reference_table(
    *,
    station_name: str,
    analyzed_df: pd.DataFrame,
    epochs_df: pd.DataFrame,
    config: dict[str, Any],
    groups: list[dict[str, Any]],
) -> tuple[pd.DataFrame, Path | None]:
    quality_cfg = config.get("quality") if isinstance(config.get("quality"), dict) else {}
    if not quality_cfg.get("enabled", False):
        return pd.DataFrame(), None
    if analyzed_df.empty:
        return pd.DataFrame(), None
    if epochs_df.empty:
        return pd.DataFrame(), None

    quality_columns = _quality_columns_from_config(
        config=config,
        groups=groups,
        available_columns=list(analyzed_df.columns),
    )
    value_frame, specs_df = build_scalar_value_frame(analyzed_df, quality_columns)
    reference_df = build_epoch_reference_table(value_frame, specs_df, analyzed_df["epoch_id"])
    if reference_df.empty:
        return pd.DataFrame(), None

    reference_df = _enrich_reference_table(reference_df, epochs_df, station_name)
    output_filename = str(
        quality_cfg.get("output_reference_filename", "calibration_epoch_references.csv")
    ).strip() or "calibration_epoch_references.csv"
    out_path = _output_files_dir(station_name) / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_csv(out_path, index=False)
    return reference_df, out_path


def _quality_threshold_config_for_specs(
    config: dict[str, Any],
    specs_df: pd.DataFrame,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]] | None]:
    rules = _column_rule_table(config)
    if not rules:
        quality_cfg = config.get("quality") if isinstance(config.get("quality"), dict) else {}
        rules_cfg = quality_cfg.get("rules") if isinstance(quality_cfg.get("rules"), dict) else {}
        return rules_cfg.get("defaults"), rules_cfg.get("column_rules")

    default_rule, _ = split_default_rule(rules)
    defaults = rule_to_threshold_mapping(default_rule) if default_rule is not None else None
    column_rules: dict[str, dict[str, Any]] = {}
    for spec in specs_df.to_dict("records"):
        evaluation_column = str(spec["evaluation_column"])
        source_column = str(spec["source_column"])
        resolved_rule = _resolve_effective_column_rule(
            config,
            evaluation_column,
            source_column=source_column,
        )
        if resolved_rule is None:
            continue
        column_rules[evaluation_column] = rule_to_threshold_mapping(resolved_rule)
    return defaults, column_rules or None


def _build_quality_pass_dataframe(
    *,
    station_name: str,
    scope_df: pd.DataFrame,
    analyzed_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    config: dict[str, Any],
    groups: list[dict[str, Any]],
    pass_column: str,
) -> tuple[pd.DataFrame, Path | None]:
    quality_cfg = config.get("quality") if isinstance(config.get("quality"), dict) else {}
    if not quality_cfg.get("enabled", False):
        default_pass = float(config.get("pass_default_value", 1.0))
        return _build_pass_dataframe(scope_df, pass_column, default_pass), None

    if scope_df.empty:
        return pd.DataFrame(columns=["filename_base", pass_column]), None

    file_df = scope_df[["filename_base", "__timestamp__", "qa_timestamp_source", "qa_in_scope"]].copy()
    file_df.rename(columns={"__timestamp__": "qa_timestamp"}, inplace=True)
    file_df["epoch_id"] = pd.Series(pd.NA, index=file_df.index, dtype="string")
    file_df["conf_number"] = pd.Series(pd.NA, index=file_df.index, dtype="Int64")

    if not analyzed_df.empty and "epoch_id" in analyzed_df.columns:
        file_epoch_df = analyzed_df[["filename_base", "epoch_id"]].drop_duplicates(subset=["filename_base"])
        if not reference_df.empty and {"epoch_id", "conf_number"} <= set(reference_df.columns):
            epoch_lookup = reference_df[["epoch_id", "conf_number"]].drop_duplicates(subset=["epoch_id"])
            file_epoch_df = file_epoch_df.merge(epoch_lookup, on="epoch_id", how="left")
        else:
            file_epoch_df["conf_number"] = pd.Series(pd.NA, index=file_epoch_df.index, dtype="Int64")
        file_df = file_df.merge(file_epoch_df, on="filename_base", how="left", suffixes=("", "_eval"))
        file_df["epoch_id"] = file_df["epoch_id_eval"].combine_first(file_df["epoch_id"])
        merged_conf_number = pd.to_numeric(file_df["conf_number_eval"], errors="coerce").astype("Float64")
        base_conf_number = pd.to_numeric(file_df["conf_number"], errors="coerce").astype("Float64")
        file_df["conf_number"] = merged_conf_number.fillna(base_conf_number).astype("Int64")
        file_df = file_df.drop(columns=["epoch_id_eval", "conf_number_eval"])

    column_eval_df = pd.DataFrame()
    if not analyzed_df.empty and not reference_df.empty:
        quality_columns = _quality_columns_from_config(
            config=config,
            groups=groups,
            available_columns=list(analyzed_df.columns),
        )
        value_frame, specs_df = build_scalar_value_frame(analyzed_df, quality_columns)
        defaults, column_rules = _quality_threshold_config_for_specs(config, specs_df)
        column_eval_df = evaluate_scalar_frame(
            analyzed_df,
            value_frame,
            reference_df,
            defaults=defaults,
            column_rules=column_rules,
        )

    if not column_eval_df.empty:
        grouped = column_eval_df.groupby("filename_base", dropna=False)
        agg_df = grouped.agg(
            qa_evaluated_columns=("status", lambda values: int(sum(item in {"pass", "fail"} for item in values))),
            qa_passed_columns=("status", lambda values: int(sum(item == "pass" for item in values))),
            qa_failed_columns=("status", lambda values: int(sum(item == "fail" for item in values))),
            qa_warning_columns=("status", lambda values: int(sum(item not in {"pass", "fail"} for item in values))),
        ).reset_index()
        failed_observables = (
            column_eval_df.loc[column_eval_df["status"] == "fail", ["filename_base", "evaluation_column"]]
            .groupby("filename_base")["evaluation_column"]
            .apply(lambda values: ";".join(sorted(values.astype(str).unique())))
            .rename("qa_failed_observables")
        )
        warning_reasons = (
            column_eval_df.loc[column_eval_df["status"] != "pass", ["filename_base", "reason"]]
            .groupby("filename_base")["reason"]
            .apply(lambda values: ";".join(sorted(values.astype(str).unique())))
            .rename("qa_warning_reasons")
        )
        agg_df = agg_df.merge(failed_observables.reset_index(), on="filename_base", how="left")
        agg_df = agg_df.merge(warning_reasons.reset_index(), on="filename_base", how="left")
        file_df = file_df.merge(agg_df, on="filename_base", how="left")
    else:
        file_df["qa_evaluated_columns"] = 0
        file_df["qa_passed_columns"] = 0
        file_df["qa_failed_columns"] = 0
        file_df["qa_warning_columns"] = 0
        file_df["qa_failed_observables"] = ""
        file_df["qa_warning_reasons"] = ""

    for column_name in (
        "qa_evaluated_columns",
        "qa_passed_columns",
        "qa_failed_columns",
        "qa_warning_columns",
    ):
        file_df[column_name] = pd.to_numeric(file_df[column_name], errors="coerce").fillna(0).astype(int)
    for column_name in ("qa_failed_observables", "qa_warning_reasons"):
        file_df[column_name] = file_df[column_name].fillna("")

    file_df["qa_status"] = "not_evaluated"
    invalid_mask = file_df["qa_timestamp"].isna()
    out_of_scope_mask = (~invalid_mask) & (~file_df["qa_in_scope"].fillna(False))
    no_epoch_mask = file_df["qa_in_scope"].fillna(False) & file_df["epoch_id"].isna()
    fail_mask = file_df["qa_failed_columns"] > 0
    warn_mask = (~fail_mask) & (file_df["qa_warning_columns"] > 0)
    pass_mask = (
        file_df["qa_in_scope"].fillna(False)
        & file_df["epoch_id"].notna()
        & (file_df["qa_evaluated_columns"] > 0)
        & (file_df["qa_failed_columns"] == 0)
        & (file_df["qa_warning_columns"] == 0)
    )

    file_df.loc[invalid_mask, "qa_status"] = "invalid_timestamp"
    file_df.loc[out_of_scope_mask, "qa_status"] = "out_of_scope"
    file_df.loc[no_epoch_mask, "qa_status"] = "no_epoch_match"
    file_df.loc[warn_mask, "qa_status"] = "warn"
    file_df.loc[fail_mask, "qa_status"] = "fail"
    file_df.loc[pass_mask, "qa_status"] = "pass"
    file_df[pass_column] = file_df["qa_status"].eq("pass").astype(float)

    output_filename = str(
        quality_cfg.get("output_column_evaluation_filename", "calibration_column_evaluations.csv")
    ).strip() or "calibration_column_evaluations.csv"
    out_path = None
    if not column_eval_df.empty:
        out_path = _output_files_dir(station_name) / output_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        column_eval_df.to_csv(out_path, index=False)

    ordered_columns = [
        "filename_base",
        pass_column,
        "qa_status",
        "qa_timestamp",
        "qa_timestamp_source",
        "qa_in_scope",
        "epoch_id",
        "conf_number",
        "qa_evaluated_columns",
        "qa_passed_columns",
        "qa_failed_columns",
        "qa_warning_columns",
        "qa_failed_observables",
        "qa_warning_reasons",
    ]
    available_columns = [column for column in ordered_columns if column in file_df.columns]
    return file_df[available_columns].copy(), out_path


def _generate_station_plots(
    station_name: str,
    config: dict[str, Any],
    analyzed_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> list[Path]:
    if analyzed_df.empty:
        return []

    df = analyzed_df.copy()
    x_values, x_label, x_is_datetime = _resolve_plot_x_axis(df, config)
    df["__plot_x__"] = x_values
    sort_key = "__plot_x__" if x_label != "__timestamp__" else "__timestamp__"
    df = df.sort_values(sort_key).reset_index(drop=True)
    x_values = df["__plot_x__"]
    epoch_spans, epoch_span_lookup = _epoch_span_data(
        df=df,
        x_values=x_values,
        x_is_datetime=x_is_datetime,
        reference_df=reference_df,
    )

    plots_dir = _output_plots_dir(station_name)
    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    image_format = str(plot_cfg.get("format", "png")).strip().lower() or "png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in plots_dir.glob(f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_*.{image_format}"):
        stale_path.unlink()

    groups = _runtime_plot_groups(config, df)
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
                reference_df=reference_df,
                epoch_spans=epoch_spans,
                epoch_span_lookup=epoch_span_lookup,
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
                reference_df=reference_df,
                epoch_spans=epoch_spans,
                epoch_span_lookup=epoch_span_lookup,
                station_name=station_name,
                group_name=group_name,
                out_dir=plots_dir,
                config=config,
                group_ncols=group.get("ncols"),
            )
        if out_path is not None:
            created.append(out_path)

    return created


def main() -> int:
    config = load_task_configs(TASK_DIR)
    bootstrap_task(TASK_DIR)

    stations = config.get("stations", [0, 1, 2, 3, 4])
    pass_column = str(config.get("pass_column_name", DEFAULT_PASS_COLUMN))

    total_rows = 0
    written = 0
    total_plots = 0
    total_summaries = 0
    total_references = 0
    total_quality_tables = 0

    for station in stations:
        station_name = _normalize_station_name(station)
        meta_path = _metadata_path(station_name)
        meta_df = _load_metadata_frame(meta_path)
        scope_df = _prepare_scope_dataframe(station_name=station_name, config=config, meta_df=meta_df)
        analyzed_df, analysis_reason = _prepare_analysis_dataframe(
            station_name=station_name,
            config=config,
            meta_df=meta_df,
        )
        analyzed_df, epochs_df = _attach_epoch_metadata(station_name=station_name, analyzed_df=analyzed_df)
        groups = _runtime_plot_groups(config, analyzed_df)

        reference_df, reference_path = _write_epoch_reference_table(
            station_name=station_name,
            analyzed_df=analyzed_df,
            epochs_df=epochs_df,
            config=config,
            groups=groups,
        )
        pass_df, quality_eval_path = _build_quality_pass_dataframe(
            station_name=station_name,
            scope_df=scope_df,
            analyzed_df=analyzed_df,
            reference_df=reference_df,
            config=config,
            groups=groups,
            pass_column=pass_column,
        )
        total_rows += len(pass_df)

        files_dir = _output_files_dir(station_name)
        files_dir.mkdir(parents=True, exist_ok=True)
        pass_path = files_dir / f"{station_name}_task_{TASK_ID}_{METADATA_TYPE}_pass.csv"
        pass_df.to_csv(pass_path, index=False)
        written += 1
        if reference_path is not None:
            total_references += 1
        if quality_eval_path is not None:
            total_quality_tables += 1

        try:
            created_plots = _generate_station_plots(
                station_name=station_name,
                config=config,
                analyzed_df=analyzed_df,
                reference_df=reference_df,
            )
            total_plots += len(created_plots)

            summary_path = _write_dataset_summary(
                station_name=station_name,
                analyzed_df=analyzed_df,
                groups=groups,
                config=config,
            )
            if summary_path is not None:
                total_summaries += 1
            if created_plots or summary_path is not None or reference_path is not None or quality_eval_path is not None:
                print(
                    f"TASK_{TASK_ID} {station_name}: plots={len(created_plots)} "
                    f"summary={'yes' if summary_path is not None else 'no'} "
                    f"epoch_ref={'yes' if reference_path is not None else 'no'} "
                    f"quality={'yes' if quality_eval_path is not None else 'no'}"
                )
            elif analysis_reason is not None:
                print(f"TASK_{TASK_ID} {station_name}: {analysis_reason}.")
        except (KeyError, ValueError) as exc:
            print(f"TASK_{TASK_ID} {station_name}: plotting/summary skipped ({exc})")

    print(
        f"TASK_{TASK_ID} {METADATA_TYPE} complete: "
        f"stations={written} total_rows={total_rows} pass_column={pass_column} "
        f"plots={total_plots} summaries={total_summaries} "
        f"epoch_references={total_references} quality_tables={total_quality_tables}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
