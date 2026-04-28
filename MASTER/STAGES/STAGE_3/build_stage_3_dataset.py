#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/STAGES/STAGE_3/build_stage_3_dataset.py
Purpose: Apply atmospheric corrections to the Stage 2 joined dataset per station and date range.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-03
Runtime: python3
Usage: python3 MASTER/STAGES/STAGE_3/build_stage_3_dataset.py <station> [--config PATH]
Inputs: Stage 2 joined CSV and Stage 3 config.
Outputs: Stage 3 joined CSV plus one coefficient summary CSV.
Notes: Pressure correction is applied first, followed by the higher-order meteorological correction.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
from scipy.interpolate import Akima1DInterpolator


CURRENT_PATH = Path(__file__).resolve()
MASTER_ROOT = None
REPO_ROOT = None
for parent in CURRENT_PATH.parents:
    if parent.name == "MASTER":
        MASTER_ROOT = parent
        REPO_ROOT = parent.parent
        break
if MASTER_ROOT is None:
    MASTER_ROOT = CURRENT_PATH.parents[-1]
if REPO_ROOT is None:
    REPO_ROOT = CURRENT_PATH.parents[-1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.execution_logger import set_station, start_timer
from MASTER.common.path_config import get_master_config_root, get_repo_root
from MASTER.common.selection_config import parse_station_id
from MASTER.STAGES.STAGE_2.plot_joined_stage_groups import generate_joined_groups_plot


CONFIG_PATH = get_master_config_root() / "STAGE_3" / "config_stage_3.yaml"
PLOT_CONFIG_PATH = get_master_config_root() / "STAGE_3" / "plot_joined_stage_3_groups.yaml"
ANGULAR_REGION_LAYOUT: Dict[str, Dict[str, float]] = {
    "High": {"r_inner": 0.0, "r_outer": 1.0, "theta_center_deg": 90.0, "width_deg": 360.0},
    "Mid-N": {"r_inner": 1.0, "r_outer": 2.0, "theta_center_deg": 90.0, "width_deg": 45.0},
    "Mid-NE": {"r_inner": 1.0, "r_outer": 2.0, "theta_center_deg": 45.0, "width_deg": 45.0},
    "Mid-E": {"r_inner": 1.0, "r_outer": 2.0, "theta_center_deg": 0.0, "width_deg": 45.0},
    "Mid-SE": {"r_inner": 1.0, "r_outer": 2.0, "theta_center_deg": 315.0, "width_deg": 45.0},
    "Mid-S": {"r_inner": 1.0, "r_outer": 2.0, "theta_center_deg": 270.0, "width_deg": 45.0},
    "Mid-SW": {"r_inner": 1.0, "r_outer": 2.0, "theta_center_deg": 225.0, "width_deg": 45.0},
    "Mid-W": {"r_inner": 1.0, "r_outer": 2.0, "theta_center_deg": 180.0, "width_deg": 45.0},
    "Mid-NW": {"r_inner": 1.0, "r_outer": 2.0, "theta_center_deg": 135.0, "width_deg": 45.0},
    "Low-N": {"r_inner": 2.0, "r_outer": 3.0, "theta_center_deg": 90.0, "width_deg": 90.0},
    "Low-E": {"r_inner": 2.0, "r_outer": 3.0, "theta_center_deg": 0.0, "width_deg": 90.0},
    "Low-S": {"r_inner": 2.0, "r_outer": 3.0, "theta_center_deg": 270.0, "width_deg": 90.0},
    "Low-W": {"r_inner": 2.0, "r_outer": 3.0, "theta_center_deg": 180.0, "width_deg": 90.0},
}
SUMMARY_FIELDS: Tuple[str, ...] = (
    "pressure_status",
    "pressure_fit_rows",
    "pressure_reference",
    "eta_p",
    "eta_p_uncertainty",
    "eta_p_intercept",
    "high_order_status",
    "high_order_fit_rows",
    "A",
    "B",
    "C",
    "D",
    "Tg0",
    "Th0",
    "H0",
)
HIGH_ORDER_PARAMETER_LABELS: Mapping[str, str] = {
    "A": "Ground Temperature",
    "B": "100 mbar Temperature",
    "C": "100 mbar Height",
}


@dataclass(frozen=True)
class TimeWindow:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]


@dataclass(frozen=True)
class CorrectionRange:
    label: str
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    fit_exclude_ranges: Tuple[TimeWindow, ...]


@dataclass(frozen=True)
class PressureCorrectionResult:
    corrected: pd.Series
    status: str
    eta_p: float
    eta_p_uncertainty: float
    eta_p_intercept: float
    pressure_reference: float
    rows_used: int


@dataclass(frozen=True)
class HighOrderCorrectionResult:
    corrected: pd.Series
    status: str
    a: float
    b: float
    c: float
    d: float
    tg0: float
    th0: float
    h0: float
    rows_used: int


def load_yaml_mapping(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage 3 from the station Stage 2 joined dataset."
    )
    parser.add_argument("station", help="Station identifier (e.g. 1 or MINGO01)")
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help=f"Path to the Stage 3 config file. Default: {CONFIG_PATH}",
    )
    return parser.parse_args()


def station_name_from_arg(raw_station: str) -> Tuple[int, str]:
    station_id = parse_station_id(raw_station)
    if station_id is None:
        raise ValueError(f"Could not parse station identifier from {raw_station!r}.")
    return station_id, f"MINGO{station_id:02d}"


def parse_boundary(value: object) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(parsed)


def parse_bool_value(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_time_windows(raw_ranges: object) -> Tuple[TimeWindow, ...]:
    if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes)):
        return tuple()

    parsed_ranges: List[TimeWindow] = []
    for item in raw_ranges:
        if not isinstance(item, Mapping):
            continue
        start_value = parse_boundary(item.get("start"))
        end_value = parse_boundary(item.get("end"))
        if start_value is not None and end_value is not None and start_value > end_value:
            start_value, end_value = end_value, start_value
        parsed_ranges.append(TimeWindow(start=start_value, end=end_value))
    return tuple(parsed_ranges)


def parse_ranges(
    raw_ranges: object,
    *,
    inherited_exclude_ranges: Sequence[TimeWindow] = (),
) -> List[CorrectionRange]:
    if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes)):
        return []

    parsed_ranges: List[CorrectionRange] = []
    for index, item in enumerate(raw_ranges, start=1):
        if not isinstance(item, Mapping):
            continue
        start_value = parse_boundary(item.get("start"))
        end_value = parse_boundary(item.get("end"))
        label = str(item.get("label") or f"range_{index}").strip() or f"range_{index}"
        if start_value is not None and end_value is not None and start_value > end_value:
            start_value, end_value = end_value, start_value
        local_exclude_ranges = parse_time_windows(item.get("exclude_fit_ranges"))
        parsed_ranges.append(
            CorrectionRange(
                label=label,
                start=start_value,
                end=end_value,
                fit_exclude_ranges=tuple((*inherited_exclude_ranges, *local_exclude_ranges)),
            )
        )
    return parsed_ranges


def resolve_station_ranges(config: Mapping[str, object], station_id: int, station_name: str) -> List[CorrectionRange]:
    stations_node = config.get("stations")
    if not isinstance(stations_node, Mapping):
        raise ValueError("config.stations must be a mapping.")

    candidates = [
        station_name,
        station_name.lower(),
        station_name.upper(),
        str(station_id),
        f"{station_id:02d}",
    ]
    station_config = None
    for key in candidates:
        if key in stations_node:
            station_config = stations_node[key]
            break
    if not isinstance(station_config, Mapping):
        raise ValueError(f"No Stage 3 station config found for {station_name}.")

    station_exclude_ranges = parse_time_windows(station_config.get("exclude_fit_ranges"))
    ranges = parse_ranges(
        station_config.get("correction_ranges"),
        inherited_exclude_ranges=station_exclude_ranges,
    )
    if not ranges:
        raise ValueError(f"No correction ranges configured for {station_name}.")
    return ranges


def build_time_window_mask(
    timestamps: pd.Series,
    time_window: TimeWindow,
) -> pd.Series:
    mask = pd.Series(True, index=timestamps.index)
    if time_window.start is not None:
        mask &= timestamps >= time_window.start
    if time_window.end is not None:
        mask &= timestamps < time_window.end
    return mask


def build_range_mask(
    timestamps: pd.Series,
    correction_range: CorrectionRange,
) -> pd.Series:
    return build_time_window_mask(
        timestamps,
        TimeWindow(start=correction_range.start, end=correction_range.end),
    )


def build_excluded_fit_mask(
    timestamps: pd.Series,
    exclude_ranges: Sequence[TimeWindow],
) -> pd.Series:
    excluded_mask = pd.Series(False, index=timestamps.index)
    for time_window in exclude_ranges:
        excluded_mask |= build_time_window_mask(timestamps, time_window)
    return excluded_mask


def akima_fill(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=True)
    missing_mask = np.isnan(values)
    if not missing_mask.any():
        return pd.Series(values, index=series.index, name=series.name)

    positions = np.arange(len(values), dtype=float)
    valid_positions = positions[~missing_mask]
    valid_values = values[~missing_mask]
    if valid_values.size < 2:
        return pd.Series(values, index=series.index, name=series.name)

    interpolator = Akima1DInterpolator(valid_positions, valid_values)
    values[missing_mask] = interpolator(positions[missing_mask])
    return pd.Series(values, index=series.index, name=series.name)


def quantile_mean(values: pd.Series, lower_quantile: float, upper_quantile: float) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return float("nan")
    q_low, q_high = np.quantile(numeric_values, [lower_quantile, upper_quantile])
    filtered = numeric_values[(numeric_values >= q_low) & (numeric_values <= q_high)]
    if filtered.empty:
        return float("nan")
    return float(filtered.mean())


def infer_target_columns(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    explicit_targets: Sequence[object],
    excluded_targets: Sequence[object],
) -> List[str]:
    if explicit_targets:
        resolved = [str(column).strip() for column in explicit_targets if str(column).strip()]
        return [column for column in resolved if column in dataframe.columns]

    excluded = {time_column}
    excluded.update(str(column).strip() for column in excluded_targets if str(column).strip())

    targets: List[str] = []
    for column in dataframe.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        if numeric.notna().any():
            targets.append(column)
    return targets


def normalize_column_label(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value)).strip("_")


def serialize_time_windows(time_windows: Sequence[TimeWindow]) -> str:
    chunks: List[str] = []
    for time_window in time_windows:
        start_text = "" if time_window.start is None else str(time_window.start)
        end_text = "" if time_window.end is None else str(time_window.end)
        chunks.append(f"{start_text}->{end_text}")
    return ";".join(chunks)


def build_wide_summary(coefficients_frame: pd.DataFrame) -> pd.DataFrame:
    if coefficients_frame.empty:
        return pd.DataFrame(
            columns=["station", "range_label", "range_start", "range_end"]
        )

    summary_rows: List[Dict[str, object]] = []
    group_columns = ["station", "range_label", "range_start", "range_end"]
    for group_values, group_frame in coefficients_frame.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, group_values))
        first_item = group_frame.iloc[0]
        row["rows_in_range"] = first_item.get("rows_in_range")
        row["fit_rows_in_range"] = first_item.get("fit_rows_in_range")
        row["fit_excluded_rows"] = first_item.get("fit_excluded_rows")
        row["fit_exclude_ranges"] = first_item.get("fit_exclude_ranges")
        for _, item in group_frame.iterrows():
            normalized_target = normalize_column_label(str(item["target_column"]))
            for field_name in SUMMARY_FIELDS:
                row[f"{normalized_target}_{field_name}"] = item.get(field_name)
        summary_rows.append(row)

    summary_frame = pd.DataFrame(summary_rows)
    fixed_columns = [column for column in group_columns if column in summary_frame.columns]
    dynamic_columns = sorted(column for column in summary_frame.columns if column not in fixed_columns)
    return summary_frame.loc[:, [*fixed_columns, *dynamic_columns]]


def format_parameter_value(value: object, precision: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{precision}f}"


def draw_sector_parameter_map(
    axis: plt.Axes,
    parameter_by_region: Mapping[str, object],
    *,
    parameter_label: str,
    title: str,
    cmap_name: str = "coolwarm",
) -> None:
    available_values = [
        float(parameter_by_region[region])
        for region in ANGULAR_REGION_LAYOUT
        if region in parameter_by_region
        and pd.notna(parameter_by_region[region])
        and np.isfinite(float(parameter_by_region[region]))
    ]
    if available_values:
        value_max = max(abs(min(available_values)), abs(max(available_values)))
        if value_max == 0:
            value_max = 1.0
        normalizer = Normalize(vmin=-value_max, vmax=value_max)
    else:
        normalizer = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap(cmap_name)

    axis.set_aspect("equal")
    axis.set_xlim(-3.35, 3.35)
    axis.set_ylim(-3.35, 3.35)
    axis.axis("off")
    axis.set_title(title, fontsize=12)

    for region_name, geometry in ANGULAR_REGION_LAYOUT.items():
        value = parameter_by_region.get(region_name, np.nan)
        has_value = pd.notna(value)
        if has_value:
            numeric_value = float(value)
            facecolor = cmap(normalizer(numeric_value))
        else:
            numeric_value = float("nan")
            facecolor = "#d9d9d9"

        if region_name == "High":
            patch = Circle((0.0, 0.0), radius=geometry["r_outer"], facecolor=facecolor, edgecolor="black", linewidth=1.0)
            axis.add_patch(patch)
            axis.text(
                0.0,
                0.0,
                f"{region_name}\n{format_parameter_value(numeric_value)}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
            continue

        theta_center = np.deg2rad(geometry["theta_center_deg"])
        theta1 = geometry["theta_center_deg"] - geometry["width_deg"] / 2.0
        theta2 = geometry["theta_center_deg"] + geometry["width_deg"] / 2.0
        patch = Wedge(
            (0.0, 0.0),
            r=geometry["r_outer"],
            theta1=theta1,
            theta2=theta2,
            width=geometry["r_outer"] - geometry["r_inner"],
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.0,
        )
        axis.add_patch(patch)
        radius = 0.5 * (geometry["r_inner"] + geometry["r_outer"])
        x_position = radius * np.cos(theta_center)
        y_position = radius * np.sin(theta_center)
        axis.text(
            x_position,
            y_position,
            f"{region_name}\n{format_parameter_value(numeric_value)}",
            ha="center",
            va="center",
            fontsize=8,
        )

    colorbar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colorbar = plt.colorbar(colorbar_mappable, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(parameter_label)


def save_range_parameter_plots(
    range_frame: pd.DataFrame,
    *,
    station_name: str,
    range_label: str,
    plots_dir: Path,
    high_order_enabled: bool,
) -> None:
    angular_frame = range_frame[range_frame["target_column"].isin(ANGULAR_REGION_LAYOUT.keys())].copy()
    if angular_frame.empty:
        return

    beta_values = angular_frame.set_index("target_column")["eta_p"].to_dict()
    fig_beta, axis_beta = plt.subplots(figsize=(7, 7))
    draw_sector_parameter_map(
        axis_beta,
        beta_values,
        parameter_label="beta [%/mbar]",
        title=f"{station_name} | {range_label} | Pressure beta",
    )
    fig_beta.tight_layout()
    fig_beta.savefig(
        plots_dir / f"{normalize_column_label(range_label).lower()}_pressure_beta_map.png",
        format="png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig_beta)

    fig_high_order, axes_high_order = plt.subplots(1, 3, figsize=(18, 6))
    for axis, parameter_name in zip(axes_high_order, ("A", "B", "C")):
        parameter_values = angular_frame.set_index("target_column")[parameter_name].to_dict()
        high_order_state = "" if high_order_enabled else " | correction disabled"
        draw_sector_parameter_map(
            axis,
            parameter_values,
            parameter_label=parameter_name,
            title=(
                f"{station_name} | {range_label} | {parameter_name} "
                f"({HIGH_ORDER_PARAMETER_LABELS[parameter_name]}){high_order_state}"
            ),
        )
    fig_high_order.tight_layout()
    fig_high_order.savefig(
        plots_dir / f"{normalize_column_label(range_label).lower()}_high_order_ABC_maps.png",
        format="png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig_high_order)


def add_scatter_with_trend(
    axis: plt.Axes,
    *,
    x_values: pd.Series,
    y_values: pd.Series,
    fit_included_mask: Optional[pd.Series],
    scatter_label: str,
    scatter_color: str,
    line_style: str,
) -> None:
    plot_frame = pd.DataFrame({"x": x_values, "y": y_values}).replace([np.inf, -np.inf], np.nan).dropna()
    if plot_frame.empty:
        return

    axis.scatter(
        plot_frame["x"],
        plot_frame["y"],
        s=7,
        alpha=0.35,
        color=scatter_color,
        label=scatter_label,
    )

    fit_frame = pd.DataFrame({"x": x_values, "y": y_values}).replace([np.inf, -np.inf], np.nan)
    if fit_included_mask is not None:
        aligned_fit_mask = pd.Series(fit_included_mask).reindex(fit_frame.index, fill_value=False)
        fit_frame = fit_frame.loc[aligned_fit_mask]
    fit_frame = fit_frame.dropna()

    if len(fit_frame) < 2 or fit_frame["x"].nunique() < 2:
        return
    slope, intercept = np.polyfit(
        fit_frame["x"].to_numpy(dtype=float),
        fit_frame["y"].to_numpy(dtype=float),
        deg=1,
    )
    x_line = np.linspace(fit_frame["x"].min(), fit_frame["x"].max(), 200)
    y_line = slope * x_line + intercept
    axis.plot(
        x_line,
        y_line,
        color=scatter_color,
        linestyle=line_style,
        linewidth=1.8,
        label=f"{scatter_label} trend: y = {slope:.3g}x + {intercept:.3g}",
    )


def resolve_plot_y_limits(
    *,
    fit_included_mask: Optional[pd.Series],
    y_series_list: Sequence[pd.Series],
    quantile_limit: float,
) -> Optional[Tuple[float, float]]:
    combined_values: List[float] = []
    for y_series in y_series_list:
        numeric = pd.to_numeric(y_series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if fit_included_mask is not None:
            aligned_fit_mask = pd.Series(fit_included_mask).reindex(numeric.index, fill_value=False)
            numeric = numeric.loc[aligned_fit_mask]
        numeric = numeric.dropna()
        if not numeric.empty:
            combined_values.extend(numeric.to_numpy(dtype=float).tolist())

    if len(combined_values) < 2:
        return None

    quantile_limit = float(quantile_limit)
    if not np.isfinite(quantile_limit):
        return None
    quantile_limit = min(max(quantile_limit, 0.0), 0.49)

    lower_quantile, upper_quantile = np.quantile(
        np.asarray(combined_values, dtype=float),
        [quantile_limit, 1.0 - quantile_limit],
    )
    lower = float(lower_quantile * 0.95)
    upper = float(upper_quantile * 1.05)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return None
    return lower, upper


def save_range_decorrelation_plots(
    *,
    station_name: str,
    range_label: str,
    plots_dir: Path,
    range_frame: pd.DataFrame,
    pressure_column: str,
    temp_ground_column: str,
    temp_100mbar_column: str,
    height_100mbar_column: str,
    original_count: pd.Series,
    pressure_corrected_count: pd.Series,
    final_corrected_count: pd.Series,
    fit_included_mask: Optional[pd.Series],
    plot_limit_quantile: float,
    high_order_enabled: bool,
) -> None:
    pressure_y_limits = resolve_plot_y_limits(
        fit_included_mask=fit_included_mask,
        y_series_list=(
            pd.to_numeric(original_count, errors="coerce"),
            pd.to_numeric(pressure_corrected_count, errors="coerce"),
        ),
        quantile_limit=plot_limit_quantile,
    )
    pressure_series = pd.to_numeric(range_frame.get(pressure_column), errors="coerce")
    if pressure_series.notna().any():
        fig_pressure, axes_pressure = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        add_scatter_with_trend(
            axes_pressure[0],
            x_values=pressure_series,
            y_values=pd.to_numeric(original_count, errors="coerce"),
            fit_included_mask=fit_included_mask,
            scatter_label="Original count",
            scatter_color="tab:blue",
            line_style="--",
        )
        add_scatter_with_trend(
            axes_pressure[1],
            x_values=pressure_series,
            y_values=pd.to_numeric(pressure_corrected_count, errors="coerce"),
            fit_included_mask=fit_included_mask,
            scatter_label="Pressure-corrected count",
            scatter_color="tab:orange",
            line_style="-.",
        )
        if pressure_y_limits is not None:
            for axis_pressure in axes_pressure:
                axis_pressure.set_ylim(*pressure_y_limits)
        axes_pressure[0].set_xlabel("pressure_lab [mbar]")
        axes_pressure[1].set_xlabel("pressure_lab [mbar]")
        axes_pressure[0].set_ylabel("Count")
        axes_pressure[0].set_title("Original")
        axes_pressure[1].set_title("Pressure-corrected")
        for axis_pressure in axes_pressure:
            axis_pressure.grid(True, alpha=0.3)
            axis_pressure.legend(fontsize=8)
        fig_pressure.suptitle(f"{station_name} | {range_label} | Pressure decorrelation", y=1.02)
        fig_pressure.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
        fig_pressure.savefig(
            plots_dir / f"{normalize_column_label(range_label).lower()}_pressure_decorrelation_scatter.png",
            format="png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig_pressure)

    high_order_y_limits = resolve_plot_y_limits(
        fit_included_mask=fit_included_mask,
        y_series_list=(
            pd.to_numeric(pressure_corrected_count, errors="coerce"),
            pd.to_numeric(final_corrected_count, errors="coerce"),
        ),
        quantile_limit=plot_limit_quantile,
    )
    high_order_specs = (
        (temp_ground_column, "Ground Temperature", "temp_ground"),
        (temp_100mbar_column, "100 mbar Temperature", "temp_100mbar"),
        (height_100mbar_column, "100 mbar Height", "height_100mbar"),
    )
    fig_high_order, axes_high_order = plt.subplots(3, 2, figsize=(14, 14), sharey=True)
    plotted_any = False
    for row_index, (column_name, title_label, axis_label) in enumerate(high_order_specs):
        x_values = pd.to_numeric(range_frame.get(column_name), errors="coerce")
        left_axis = axes_high_order[row_index, 0]
        right_axis = axes_high_order[row_index, 1]
        add_scatter_with_trend(
            left_axis,
            x_values=x_values,
            y_values=pd.to_numeric(pressure_corrected_count, errors="coerce"),
            fit_included_mask=fit_included_mask,
            scatter_label="Pressure-corrected count",
            scatter_color="tab:blue",
            line_style="--",
        )
        add_scatter_with_trend(
            right_axis,
            x_values=x_values,
            y_values=pd.to_numeric(final_corrected_count, errors="coerce"),
            fit_included_mask=fit_included_mask,
            scatter_label="Final corrected count",
            scatter_color="tab:orange",
            line_style="-.",
        )
        if high_order_y_limits is not None:
            left_axis.set_ylim(*high_order_y_limits)
            right_axis.set_ylim(*high_order_y_limits)
        left_axis.set_xlabel(axis_label)
        right_axis.set_xlabel(axis_label)
        left_axis.set_title(f"{title_label} | Original")
        right_axis.set_title(
            f"{title_label} | {'Decorrelated' if high_order_enabled else 'Not Applied'}"
        )
        left_axis.grid(True, alpha=0.3)
        right_axis.grid(True, alpha=0.3)
        left_axis.legend(fontsize=8)
        right_axis.legend(fontsize=8)
        left_axis.set_ylabel("Count")
        if x_values.notna().any():
            plotted_any = True

    if plotted_any:
        figure_title = (
            f"{station_name} | {range_label} | High-order decorrelation"
            if high_order_enabled
            else f"{station_name} | {range_label} | High-order correction not applied"
        )
        fig_high_order.suptitle(figure_title, y=1.01)
        fig_high_order.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
        fig_high_order.savefig(
            plots_dir / f"{normalize_column_label(range_label).lower()}_high_order_decorrelation_scatter.png",
            format="png",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig_high_order)


def apply_pressure_coefficients(
    series: pd.Series,
    pressure_series: pd.Series,
    fit_result: PressureCorrectionResult,
) -> pd.Series:
    corrected = pd.to_numeric(series, errors="coerce").copy()
    if fit_result.status != "applied" or not np.isfinite(fit_result.eta_p) or not np.isfinite(fit_result.pressure_reference):
        return corrected

    pressure = pd.to_numeric(pressure_series, errors="coerce")
    delta_pressure = pressure - fit_result.pressure_reference
    apply_mask = corrected.notna() & delta_pressure.notna() & np.isfinite(delta_pressure)
    corrected.loc[apply_mask] = corrected.loc[apply_mask] * np.exp(
        -1.0 * fit_result.eta_p / 100.0 * delta_pressure.loc[apply_mask]
    )
    return corrected


def apply_high_order_coefficients(
    dataframe: pd.DataFrame,
    *,
    series: pd.Series,
    temp_ground_column: str,
    temp_100mbar_column: str,
    height_100mbar_column: str,
    fit_result: HighOrderCorrectionResult,
) -> pd.Series:
    corrected = pd.to_numeric(series, errors="coerce").copy()
    if fit_result.status != "applied":
        return corrected
    required_columns = [temp_ground_column, temp_100mbar_column, height_100mbar_column]
    if any(column not in dataframe.columns for column in required_columns):
        return corrected
    if not all(np.isfinite(value) for value in (fit_result.a, fit_result.b, fit_result.c, fit_result.d)):
        return corrected
    if any(value == 0 or not np.isfinite(value) for value in (fit_result.tg0, fit_result.th0, fit_result.h0)):
        return corrected

    temp_ground = pd.to_numeric(dataframe[temp_ground_column], errors="coerce")
    temp_100mbar = pd.to_numeric(dataframe[temp_100mbar_column], errors="coerce")
    height_100mbar = pd.to_numeric(dataframe[height_100mbar_column], errors="coerce")
    term = (
        fit_result.a * (temp_ground - fit_result.tg0) / fit_result.tg0
        + fit_result.b * (temp_100mbar - fit_result.th0) / fit_result.th0
        + fit_result.c * (height_100mbar - fit_result.h0) / fit_result.h0
        + fit_result.d
    )
    apply_mask = corrected.notna() & term.notna()
    corrected.loc[apply_mask] = corrected.loc[apply_mask] * (1.0 - term.loc[apply_mask])
    return corrected


def apply_pressure_correction(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    pressure_column: str,
    lower_quantile: float,
    upper_quantile: float,
    pressure_uncertainty_mbar: float,
) -> PressureCorrectionResult:
    original = pd.to_numeric(dataframe[target_column], errors="coerce")
    corrected = original.copy()

    if pressure_column not in dataframe.columns:
        return PressureCorrectionResult(
            corrected=corrected,
            status="skipped_missing_pressure_column",
            eta_p=float("nan"),
            eta_p_uncertainty=float("nan"),
            eta_p_intercept=float("nan"),
            pressure_reference=float("nan"),
            rows_used=0,
        )

    pressure = pd.to_numeric(dataframe[pressure_column], errors="coerce")
    if pressure.notna().sum() == 0:
        return PressureCorrectionResult(
            corrected=corrected,
            status="skipped_all_pressure_nan",
            eta_p=float("nan"),
            eta_p_uncertainty=float("nan"),
            eta_p_intercept=float("nan"),
            pressure_reference=float("nan"),
            rows_used=0,
        )

    valid_mask = original.notna() & pressure.notna() & np.isfinite(original) & np.isfinite(pressure)
    valid_mask &= original > 0
    if int(valid_mask.sum()) < 3:
        return PressureCorrectionResult(
            corrected=corrected,
            status="skipped_insufficient_pressure_fit_rows",
            eta_p=float("nan"),
            eta_p_uncertainty=float("nan"),
            eta_p_intercept=float("nan"),
            pressure_reference=float("nan"),
            rows_used=int(valid_mask.sum()),
        )

    intensity = original.loc[valid_mask]
    pressure_valid = pressure.loc[valid_mask]
    pressure_reference = float(pressure_valid.mean())
    intensity_reference = quantile_mean(intensity, lower_quantile, upper_quantile)
    if not np.isfinite(intensity_reference) or intensity_reference <= 0:
        return PressureCorrectionResult(
            corrected=corrected,
            status="skipped_invalid_intensity_reference",
            eta_p=float("nan"),
            eta_p_uncertainty=float("nan"),
            eta_p_intercept=float("nan"),
            pressure_reference=pressure_reference,
            rows_used=int(valid_mask.sum()),
        )

    unc_pressure = float(pressure_uncertainty_mbar)
    unc_pressure_reference = unc_pressure / np.sqrt(len(pressure_valid))
    delta_pressure = pressure_valid - pressure_reference
    unc_delta_pressure = np.sqrt(unc_pressure**2 + unc_pressure_reference**2)

    unc_intensity = 1.0
    unc_intensity_reference = unc_intensity / np.sqrt(len(intensity))
    intensity_over_reference = intensity / intensity_reference
    with np.errstate(divide="ignore", invalid="ignore"):
        unc_intensity_over_reference = intensity_over_reference * np.sqrt(
            (unc_intensity / intensity) ** 2
            + (unc_intensity_reference / intensity_reference) ** 2
        )

    fit_mask = (
        intensity_over_reference > 0
    ) & np.isfinite(intensity_over_reference) & np.isfinite(unc_intensity_over_reference)
    fit_mask &= unc_intensity_over_reference > 0
    if int(fit_mask.sum()) < 3:
        return PressureCorrectionResult(
            corrected=corrected,
            status="skipped_insufficient_positive_pressure_fit_rows",
            eta_p=float("nan"),
            eta_p_uncertainty=float("nan"),
            eta_p_intercept=float("nan"),
            pressure_reference=pressure_reference,
            rows_used=int(fit_mask.sum()),
        )

    delta_pressure_fit = delta_pressure.loc[fit_mask].to_numpy(dtype=float)
    log_intensity_ratio = np.log(intensity_over_reference.loc[fit_mask].to_numpy(dtype=float))
    unc_log_ratio = (
        unc_intensity_over_reference.loc[fit_mask].to_numpy(dtype=float)
        / intensity_over_reference.loc[fit_mask].to_numpy(dtype=float)
    )

    try:
        coefficients, covariance = np.polyfit(
            delta_pressure_fit,
            log_intensity_ratio,
            deg=1,
            w=1.0 / unc_log_ratio,
            cov=True,
        )
        slope, eta_p_intercept = (float(value) for value in coefficients)
        eta_p = 100.0 * slope
        eta_p_uncertainty = 100.0 * float(np.sqrt(covariance[0, 0]))
    except Exception:
        return PressureCorrectionResult(
            corrected=corrected,
            status="skipped_pressure_fit_failed",
            eta_p=float("nan"),
            eta_p_uncertainty=float("nan"),
            eta_p_intercept=float("nan"),
            pressure_reference=pressure_reference,
            rows_used=int(fit_mask.sum()),
        )

    delta_pressure_all = pressure - pressure_reference
    apply_mask = original.notna() & delta_pressure_all.notna() & np.isfinite(delta_pressure_all)
    corrected.loc[apply_mask] = original.loc[apply_mask] * np.exp(
        -1.0 * eta_p / 100.0 * delta_pressure_all.loc[apply_mask]
    )

    return PressureCorrectionResult(
        corrected=corrected,
        status="applied",
        eta_p=eta_p,
        eta_p_uncertainty=eta_p_uncertainty,
        eta_p_intercept=eta_p_intercept,
        pressure_reference=pressure_reference,
        rows_used=int(fit_mask.sum()),
    )


def apply_high_order_correction(
    dataframe: pd.DataFrame,
    *,
    series: pd.Series,
    temp_ground_column: str,
    temp_100mbar_column: str,
    height_100mbar_column: str,
) -> HighOrderCorrectionResult:
    corrected = pd.to_numeric(series, errors="coerce").copy()
    required_columns = [temp_ground_column, temp_100mbar_column, height_100mbar_column]
    if any(column not in dataframe.columns for column in required_columns):
        return HighOrderCorrectionResult(
            corrected=corrected,
            status="skipped_missing_environment_column",
            a=float("nan"),
            b=float("nan"),
            c=float("nan"),
            d=float("nan"),
            tg0=float("nan"),
            th0=float("nan"),
            h0=float("nan"),
            rows_used=0,
        )

    temp_ground = pd.to_numeric(dataframe[temp_ground_column], errors="coerce")
    temp_100mbar = pd.to_numeric(dataframe[temp_100mbar_column], errors="coerce")
    height_100mbar = pd.to_numeric(dataframe[height_100mbar_column], errors="coerce")

    tg0 = float(temp_ground.mean())
    th0 = float(temp_100mbar.mean())
    h0 = float(height_100mbar.mean())
    intensity_reference = float(corrected.mean())
    if not np.isfinite(intensity_reference) or intensity_reference == 0:
        return HighOrderCorrectionResult(
            corrected=corrected,
            status="skipped_invalid_high_order_intensity_reference",
            a=float("nan"),
            b=float("nan"),
            c=float("nan"),
            d=float("nan"),
            tg0=tg0,
            th0=th0,
            h0=h0,
            rows_used=0,
        )
    if not all(np.isfinite(value) and value != 0 for value in (tg0, th0, h0)):
        return HighOrderCorrectionResult(
            corrected=corrected,
            status="skipped_invalid_environment_reference",
            a=float("nan"),
            b=float("nan"),
            c=float("nan"),
            d=float("nan"),
            tg0=tg0,
            th0=th0,
            h0=h0,
            rows_used=0,
        )

    delta_tg = temp_ground - tg0
    delta_th = temp_100mbar - th0
    delta_h = height_100mbar - h0
    delta_i_over_i0 = (corrected - intensity_reference) / intensity_reference

    fit_frame = pd.DataFrame(
        {
            "delta_I_over_I0": delta_i_over_i0,
            "delta_Tg_over_Tg0": delta_tg / tg0,
            "delta_Th_over_Th0": delta_th / th0,
            "delta_H_over_H0": delta_h / h0,
        }
    ).dropna()
    if len(fit_frame) < 3:
        return HighOrderCorrectionResult(
            corrected=corrected,
            status="skipped_insufficient_high_order_fit_rows",
            a=float("nan"),
            b=float("nan"),
            c=float("nan"),
            d=float("nan"),
            tg0=tg0,
            th0=th0,
            h0=h0,
            rows_used=len(fit_frame),
        )

    try:
        design_matrix = np.column_stack(
            [
                fit_frame["delta_Tg_over_Tg0"].to_numpy(dtype=float),
                fit_frame["delta_Th_over_Th0"].to_numpy(dtype=float),
                fit_frame["delta_H_over_H0"].to_numpy(dtype=float),
                np.ones(len(fit_frame), dtype=float),
            ]
        )
        response = fit_frame["delta_I_over_I0"].to_numpy(dtype=float)
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, response, rcond=None)
    except Exception:
        return HighOrderCorrectionResult(
            corrected=corrected,
            status="skipped_high_order_fit_failed",
            a=float("nan"),
            b=float("nan"),
            c=float("nan"),
            d=float("nan"),
            tg0=tg0,
            th0=th0,
            h0=h0,
            rows_used=len(fit_frame),
        )

    a, b, c, d = (float(value) for value in coefficients)

    term = a * delta_tg / tg0 + b * delta_th / th0 + c * delta_h / h0 + d
    apply_mask = corrected.notna() & term.notna()
    corrected.loc[apply_mask] = corrected.loc[apply_mask] * (1.0 - term.loc[apply_mask])

    return HighOrderCorrectionResult(
        corrected=corrected,
        status="applied",
        a=a,
        b=b,
        c=c,
        d=d,
        tg0=tg0,
        th0=th0,
        h0=h0,
        rows_used=len(fit_frame),
    )


def build_disabled_high_order_result(
    dataframe: pd.DataFrame,
    *,
    series: pd.Series,
    temp_ground_column: str,
    temp_100mbar_column: str,
    height_100mbar_column: str,
) -> HighOrderCorrectionResult:
    corrected = pd.to_numeric(series, errors="coerce").copy()
    temp_ground = pd.to_numeric(
        dataframe[temp_ground_column] if temp_ground_column in dataframe.columns else pd.Series(np.nan, index=dataframe.index),
        errors="coerce",
    )
    temp_100mbar = pd.to_numeric(
        dataframe[temp_100mbar_column] if temp_100mbar_column in dataframe.columns else pd.Series(np.nan, index=dataframe.index),
        errors="coerce",
    )
    height_100mbar = pd.to_numeric(
        dataframe[height_100mbar_column] if height_100mbar_column in dataframe.columns else pd.Series(np.nan, index=dataframe.index),
        errors="coerce",
    )
    return HighOrderCorrectionResult(
        corrected=corrected,
        status="disabled_by_config",
        a=float("nan"),
        b=float("nan"),
        c=float("nan"),
        d=float("nan"),
        tg0=float(temp_ground.mean()) if not temp_ground.empty else float("nan"),
        th0=float(temp_100mbar.mean()) if not temp_100mbar.empty else float("nan"),
        h0=float(height_100mbar.mean()) if not height_100mbar.empty else float("nan"),
        rows_used=0,
    )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = load_yaml_mapping(config_path)

    station_id, station_name = station_name_from_arg(args.station)
    set_station(station_id)
    start_timer(__file__)

    repo_root = get_repo_root()
    input_config = config.get("input") or {}
    output_config = config.get("output") or {}
    plotting_config = config.get("plotting") or {}
    correction_config = config.get("correction") or {}
    if not isinstance(input_config, Mapping) or not isinstance(output_config, Mapping) or not isinstance(plotting_config, Mapping) or not isinstance(correction_config, Mapping):
        raise ValueError("input, output, plotting, and correction sections must be mappings.")

    input_stage_subdir = str(input_config.get("stage_subdir", "STAGE_2")).strip() or "STAGE_2"
    input_filename = str(input_config.get("input_filename", "joined_stage_2.csv")).strip() or "joined_stage_2.csv"
    output_stage_subdir = str(output_config.get("stage_subdir", "STAGE_3")).strip() or "STAGE_3"
    output_filename = str(output_config.get("output_filename", "joined_stage_3.csv")).strip() or "joined_stage_3.csv"
    coefficients_filename = str(output_config.get("coefficients_filename", "correction_coefficients.csv")).strip() or "correction_coefficients.csv"
    wide_summary_filename = str(output_config.get("wide_summary_filename", "correction_coefficients_wide.csv")).strip() or "correction_coefficients_wide.csv"
    plots_subdir = str(output_config.get("plots_subdir", "PLOTS")).strip() or "PLOTS"
    time_column = str(input_config.get("time_column", "time")).strip() or "time"
    plot_limit_quantile = float(plotting_config.get("decorrelation_plot_limit_quantile", 0.05))

    stage_2_path = repo_root / "STATIONS" / station_name / input_stage_subdir / input_filename
    stage_3_dir = repo_root / "STATIONS" / station_name / output_stage_subdir
    stage_3_output_path = stage_3_dir / output_filename
    coefficients_output_path = stage_3_dir / coefficients_filename
    wide_summary_output_path = stage_3_dir / wide_summary_filename
    plots_output_dir = stage_3_dir / plots_subdir

    if not stage_2_path.exists():
        raise FileNotFoundError(f"Stage 2 input not found: {stage_2_path}")

    dataframe = pd.read_csv(stage_2_path)
    if time_column not in dataframe.columns:
        raise ValueError(f"Time column {time_column!r} not found in {stage_2_path}.")
    dataframe[time_column] = pd.to_datetime(dataframe[time_column], errors="coerce")
    dataframe = dataframe.dropna(subset=[time_column]).reset_index(drop=True)

    ranges = resolve_station_ranges(config, station_id, station_name)

    pressure_column = str(correction_config.get("pressure_column", "pressure_lab")).strip() or "pressure_lab"
    temp_ground_column = str(correction_config.get("temp_ground_column", "temp_ground")).strip() or "temp_ground"
    temp_100mbar_column = str(correction_config.get("temp_100mbar_column", "temp_100mbar")).strip() or "temp_100mbar"
    height_100mbar_column = str(correction_config.get("height_100mbar_column", "height_100mbar")).strip() or "height_100mbar"
    lower_quantile = float(correction_config.get("lower_quantile", 0.01))
    upper_quantile = float(correction_config.get("upper_quantile", 0.99))
    pressure_uncertainty_mbar = float(correction_config.get("pressure_uncertainty_mbar", 1.0))
    apply_high_order_correction_enabled = parse_bool_value(
        correction_config.get("apply_high_order_correction", True),
        default=True,
    )

    environmental_columns = correction_config.get("environmental_columns_to_interpolate") or [
        pressure_column,
        temp_ground_column,
        temp_100mbar_column,
        height_100mbar_column,
    ]
    for column in environmental_columns:
        column_name = str(column).strip()
        if column_name and column_name in dataframe.columns:
            dataframe[column_name] = akima_fill(dataframe[column_name])

    target_columns = infer_target_columns(
        dataframe,
        time_column=time_column,
        explicit_targets=correction_config.get("target_columns") or [],
        excluded_targets=correction_config.get("excluded_target_columns") or [],
    )
    if not target_columns:
        raise ValueError("No target columns resolved for Stage 3 correction.")

    corrected_dataframe = dataframe.copy()
    coefficients_rows: List[Dict[str, object]] = []
    covered_rows = pd.Series(False, index=corrected_dataframe.index)
    decorrelation_plot_inputs: Dict[str, Dict[str, object]] = {}

    for correction_range in ranges:
        range_mask = build_range_mask(corrected_dataframe[time_column], correction_range)
        covered_rows |= range_mask
        range_rows = int(range_mask.sum())
        range_frame = corrected_dataframe.loc[range_mask].copy()
        fit_excluded_mask = build_excluded_fit_mask(
            range_frame[time_column],
            correction_range.fit_exclude_ranges,
        )
        fit_frame = range_frame.loc[~fit_excluded_mask].copy()
        fit_rows_in_range = len(fit_frame)
        fit_excluded_rows = int(fit_excluded_mask.sum())
        fit_exclude_ranges_text = serialize_time_windows(correction_range.fit_exclude_ranges)

        if range_rows == 0:
            for target_column in target_columns:
                coefficients_rows.append(
                    {
                        "station": station_name,
                        "range_label": correction_range.label,
                        "range_start": correction_range.start,
                        "range_end": correction_range.end,
                        "target_column": target_column,
                        "rows_in_range": 0,
                        "fit_rows_in_range": 0,
                        "fit_excluded_rows": 0,
                        "fit_exclude_ranges": fit_exclude_ranges_text,
                        "pressure_status": "skipped_empty_range",
                        "pressure_fit_rows": 0,
                        "pressure_reference": np.nan,
                        "eta_p": np.nan,
                        "eta_p_uncertainty": np.nan,
                        "eta_p_intercept": np.nan,
                        "high_order_status": "skipped_empty_range",
                        "high_order_fit_rows": 0,
                        "A": np.nan,
                        "B": np.nan,
                        "C": np.nan,
                        "D": np.nan,
                        "Tg0": np.nan,
                        "Th0": np.nan,
                        "H0": np.nan,
                    }
                )
            continue

        for target_column in target_columns:
            pressure_result = apply_pressure_correction(
                fit_frame,
                target_column=target_column,
                pressure_column=pressure_column,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                pressure_uncertainty_mbar=pressure_uncertainty_mbar,
            )
            if apply_high_order_correction_enabled:
                high_order_result = apply_high_order_correction(
                    fit_frame,
                    series=pressure_result.corrected,
                    temp_ground_column=temp_ground_column,
                    temp_100mbar_column=temp_100mbar_column,
                    height_100mbar_column=height_100mbar_column,
                )
            else:
                high_order_result = build_disabled_high_order_result(
                    fit_frame,
                    series=pressure_result.corrected,
                    temp_ground_column=temp_ground_column,
                    temp_100mbar_column=temp_100mbar_column,
                    height_100mbar_column=height_100mbar_column,
                )

            pressure_applied_series = apply_pressure_coefficients(
                range_frame[target_column],
                range_frame[pressure_column] if pressure_column in range_frame.columns else pd.Series(np.nan, index=range_frame.index),
                pressure_result,
            )
            final_applied_series = apply_high_order_coefficients(
                range_frame,
                series=pressure_applied_series,
                temp_ground_column=temp_ground_column,
                temp_100mbar_column=temp_100mbar_column,
                height_100mbar_column=height_100mbar_column,
                fit_result=high_order_result,
            )

            corrected_dataframe.loc[range_mask, target_column] = final_applied_series
            if target_column == "count":
                decorrelation_plot_inputs[correction_range.label] = {
                    "range_frame": range_frame.copy(),
                    "original_count": pd.to_numeric(range_frame[target_column], errors="coerce").copy(),
                    "pressure_corrected_count": pd.to_numeric(pressure_applied_series, errors="coerce").copy(),
                    "final_corrected_count": pd.to_numeric(final_applied_series, errors="coerce").copy(),
                    "fit_included_mask": (~fit_excluded_mask).copy(),
                    "high_order_enabled": apply_high_order_correction_enabled,
                }
            coefficients_rows.append(
                {
                    "station": station_name,
                    "range_label": correction_range.label,
                    "range_start": correction_range.start,
                    "range_end": correction_range.end,
                    "target_column": target_column,
                    "rows_in_range": range_rows,
                    "fit_rows_in_range": fit_rows_in_range,
                    "fit_excluded_rows": fit_excluded_rows,
                    "fit_exclude_ranges": fit_exclude_ranges_text,
                    "pressure_status": pressure_result.status,
                    "pressure_fit_rows": pressure_result.rows_used,
                    "pressure_reference": pressure_result.pressure_reference,
                    "eta_p": pressure_result.eta_p,
                    "eta_p_uncertainty": pressure_result.eta_p_uncertainty,
                    "eta_p_intercept": pressure_result.eta_p_intercept,
                    "high_order_status": high_order_result.status,
                    "high_order_fit_rows": high_order_result.rows_used,
                    "A": high_order_result.a,
                    "B": high_order_result.b,
                    "C": high_order_result.c,
                    "D": high_order_result.d,
                    "Tg0": high_order_result.tg0,
                    "Th0": high_order_result.th0,
                    "H0": high_order_result.h0,
                }
            )

    uncovered_rows = int((~covered_rows).sum())
    if uncovered_rows:
        print(
            f"[STAGE_3] Warning: {uncovered_rows} rows were outside the configured correction ranges "
            "and were left unchanged."
        )

    stage_3_dir.mkdir(parents=True, exist_ok=True)
    plots_output_dir.mkdir(parents=True, exist_ok=True)
    corrected_dataframe.to_csv(stage_3_output_path, index=False)
    coefficients_frame = pd.DataFrame(coefficients_rows)
    coefficients_frame.to_csv(coefficients_output_path, index=False)
    build_wide_summary(coefficients_frame).to_csv(wide_summary_output_path, index=False)

    for correction_range in ranges:
        range_frame = coefficients_frame[coefficients_frame["range_label"] == correction_range.label].copy()
        save_range_parameter_plots(
            range_frame,
            station_name=station_name,
            range_label=correction_range.label,
            plots_dir=plots_output_dir,
            high_order_enabled=apply_high_order_correction_enabled,
        )
        decorrelation_inputs = decorrelation_plot_inputs.get(correction_range.label)
        if decorrelation_inputs:
            save_range_decorrelation_plots(
                station_name=station_name,
                range_label=correction_range.label,
                plots_dir=plots_output_dir,
                range_frame=decorrelation_inputs["range_frame"],
                pressure_column=pressure_column,
                temp_ground_column=temp_ground_column,
                temp_100mbar_column=temp_100mbar_column,
                height_100mbar_column=height_100mbar_column,
                original_count=decorrelation_inputs["original_count"],
                pressure_corrected_count=decorrelation_inputs["pressure_corrected_count"],
                final_corrected_count=decorrelation_inputs["final_corrected_count"],
                fit_included_mask=decorrelation_inputs["fit_included_mask"],
                plot_limit_quantile=plot_limit_quantile,
                high_order_enabled=decorrelation_inputs["high_order_enabled"],
            )
    grouped_plot_output_path = stage_3_dir / "joined_stage_3_groups.png"
    try:
        generate_joined_groups_plot(
            stage_3_output_path,
            grouped_plot_output_path,
            PLOT_CONFIG_PATH,
            rate_column="count",
            rate_title="Corrected Count",
            rate_ylabel="Corrected Count",
        )
    except Exception as exc:
        print(f"Warning: unable to generate grouped Stage 3 plot: {exc}")

    print(f"[STAGE_3] Station: {station_name}")
    print(f"[STAGE_3] Input : {stage_2_path}")
    print(f"[STAGE_3] Output: {stage_3_output_path}")
    print(f"[STAGE_3] Coefficients: {coefficients_output_path}")
    print(f"[STAGE_3] Wide summary: {wide_summary_output_path}")
    print(f"[STAGE_3] Plots dir: {plots_output_dir}")
    print(f"[STAGE_3] Grouped plot: {grouped_plot_output_path}")
    print(f"[STAGE_3] Corrected columns: {', '.join(target_columns)}")
    print(f"[STAGE_3] Configured ranges: {len(ranges)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
