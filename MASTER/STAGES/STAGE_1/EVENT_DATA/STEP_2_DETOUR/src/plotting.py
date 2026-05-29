from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rates import (
    EFFICIENCY_TOPOLOGIES,
    RATE_TIME_COLUMN,
    gate_efficiency_column,
    gate_hz_column,
    gate_percent_column,
    gate_topology_hz_column,
)


RATE_AND_EFFICIENCY_FIGURE = "gate_rate_efficiency_comparison.png"
TOPOLOGY_FIGURE = "gate_topology_rate_comparison.png"
RATE_MODE_RATES = "rates"
RATE_MODE_ZSCORES = "zscores"


def plot_diagnostic_table(
    diagnostic_table: pd.DataFrame,
    output_path: Path,
    gate_names: list[str],
    moving_average_minutes: int | None = None,
    rate_mode: str = RATE_MODE_RATES,
    title_context: str | None = None,
) -> list[Path]:
    if RATE_TIME_COLUMN not in diagnostic_table.columns:
        raise ValueError(f"Diagnostic table is missing required column '{RATE_TIME_COLUMN}'.")

    df = diagnostic_table.copy()
    df[RATE_TIME_COLUMN] = pd.to_datetime(df[RATE_TIME_COLUMN], errors="coerce")
    if df[RATE_TIME_COLUMN].isna().any():
        raise ValueError("Diagnostic table contains invalid datetime values.")
    df = df.sort_values(RATE_TIME_COLUMN).reset_index(drop=True)
    normalized_rate_mode = _normalize_rate_mode(rate_mode)
    df = apply_rate_mode(df, rate_mode=normalized_rate_mode)

    if not gate_names:
        raise ValueError("At least one gate is required for plotting.")

    plot_dir = _resolve_plot_directory(output_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    color_map = _build_color_map(gate_names)
    smoothing_label = ""
    if moving_average_minutes is not None and moving_average_minutes > 1:
        smoothing_label = f" | {moving_average_minutes} min moving average"

    rate_unit_label = "Hz" if normalized_rate_mode == RATE_MODE_RATES else "z-score"
    rate_efficiency_path = _plot_metric_figure(
        df=df,
        gate_names=gate_names,
        color_map=color_map,
        metric_specs=[
            (f"Gate rate [{rate_unit_label}]", lambda gate_name: gate_hz_column(gate_name)),
            ("Eff 1", lambda gate_name: gate_efficiency_column(gate_name, 1)),
            ("Eff 2", lambda gate_name: gate_efficiency_column(gate_name, 2)),
            ("Eff 3", lambda gate_name: gate_efficiency_column(gate_name, 3)),
            ("Eff 4", lambda gate_name: gate_efficiency_column(gate_name, 4)),
            ("Gate fraction [%]", lambda gate_name: gate_percent_column(gate_name)),
        ],
        output_path=plot_dir / RATE_AND_EFFICIENCY_FIGURE,
        moving_average_minutes=moving_average_minutes,
        figure_title="Gate rate, empirical efficiency, and fraction comparison",
        title_context=_join_title_parts(title_context, smoothing_label),
    )
    topology_path = _plot_metric_figure(
        df=df,
        gate_names=gate_names,
        color_map=color_map,
        metric_specs=[
            (
                f"post_tt_{topology_code} [{rate_unit_label}]",
                lambda gate_name, code=topology_code: gate_topology_hz_column(gate_name, code),
            )
            for topology_code in EFFICIENCY_TOPOLOGIES
        ],
        output_path=plot_dir / TOPOLOGY_FIGURE,
        moving_average_minutes=moving_average_minutes,
        figure_title="Gate topology rate comparison",
        title_context=_join_title_parts(title_context, smoothing_label),
    )

    return [rate_efficiency_path, topology_path]


def apply_moving_average(series: pd.Series, moving_average_minutes: int | None) -> pd.Series:
    if moving_average_minutes is None or moving_average_minutes <= 1:
        return series.astype("float64").copy()

    return (
        series.astype("float64")
        .rolling(window=moving_average_minutes, center=True, min_periods=1)
        .mean()
    )


def apply_rate_mode(diagnostic_table: pd.DataFrame, rate_mode: str) -> pd.DataFrame:
    normalized_rate_mode = _normalize_rate_mode(rate_mode)
    df = diagnostic_table.copy()
    if normalized_rate_mode == RATE_MODE_RATES:
        return df

    for column in [column_name for column_name in df.columns if column_name.endswith("_hz")]:
        df[column] = _zscore_series(df[column])
    return df


def _plot_metric_figure(
    df: pd.DataFrame,
    gate_names: list[str],
    color_map: dict[str, tuple[float, float, float, float]],
    metric_specs: list[tuple[str, callable]],
    output_path: Path,
    moving_average_minutes: int | None,
    figure_title: str,
    title_context: str | None,
) -> Path:
    axes_count = len(metric_specs)
    fig, axes = plt.subplots(axes_count, 1, sharex=True, figsize=(14, max(3 * axes_count, 10)))
    if axes_count == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for axis, (ylabel, column_factory) in zip(axes, metric_specs, strict=True):
        for gate_name in gate_names:
            column_name = column_factory(gate_name)
            if column_name not in df.columns:
                raise ValueError(f"Diagnostic table is missing required column '{column_name}'.")

            line, = axis.plot(
                df[RATE_TIME_COLUMN],
                apply_moving_average(df[column_name], moving_average_minutes),
                linewidth=1.4,
                label=gate_name,
                color=color_map[gate_name],
            )
            if ylabel == metric_specs[0][0]:
                legend_handles.append(line)
                legend_labels.append(gate_name)

        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")

    title = figure_title
    if title_context:
        title = f"{title}\n{title_context}"
    fig.suptitle(title)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper right", ncol=min(len(gate_names), 4))
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_color_map(gate_names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {gate_name: cmap(index % 10) for index, gate_name in enumerate(gate_names)}


def _resolve_plot_directory(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.parent
    return output_path


def _join_title_parts(title_context: str | None, smoothing_label: str) -> str | None:
    if title_context and smoothing_label:
        return f"{title_context}{smoothing_label}"
    if title_context:
        return title_context
    if smoothing_label:
        return smoothing_label.lstrip(" |")
    return None


def _normalize_rate_mode(rate_mode: str) -> str:
    normalized_rate_mode = str(rate_mode or RATE_MODE_RATES).strip().lower()
    if normalized_rate_mode not in {RATE_MODE_RATES, RATE_MODE_ZSCORES}:
        raise ValueError(
            f"Unsupported rate_mode '{rate_mode}'. Expected '{RATE_MODE_RATES}' or '{RATE_MODE_ZSCORES}'."
        )
    return normalized_rate_mode


def _zscore_series(series: pd.Series) -> pd.Series:
    numeric = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index, dtype="float64")
    finite_mask = numeric.notna() & np.isfinite(numeric)
    valid_values = numeric.loc[finite_mask]

    result = pd.Series(np.nan, index=series.index, dtype="float64")
    if valid_values.empty:
        return result

    mean = valid_values.mean()
    std = valid_values.std(ddof=0)
    if pd.isna(std) or std == 0:
        result.loc[finite_mask] = 0.0
        return result

    result.loc[finite_mask] = (valid_values - mean) / std
    return result
