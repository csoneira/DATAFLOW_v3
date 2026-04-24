#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DEFAULT_CONFIG_PATH, cfg_path, ensure_output_dirs, load_config


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_2 - %(message)s", level=logging.INFO, force=True)


def _resolve_efficiency_reference_columns(config: dict[str, Any]) -> list[str]:
    step2_config = config.get("step2", {})
    if not isinstance(step2_config, dict):
        step2_config = {}

    raw_planes = step2_config.get("efficiency_reference_planes", [2, 3])
    if isinstance(raw_planes, str):
        text = raw_planes.strip()
        if not text:
            raw_planes = [2, 3]
        else:
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = [piece.strip() for piece in text.split(",") if piece.strip()]
            raw_planes = decoded

    if isinstance(raw_planes, (int, float)) and not isinstance(raw_planes, bool):
        raw_planes = [int(raw_planes)]

    if not isinstance(raw_planes, (list, tuple)) or not raw_planes:
        raise ValueError("step2.efficiency_reference_planes must be a non-empty list of plane numbers between 1 and 4.")

    planes: list[int] = []
    for value in raw_planes:
        plane = int(value)
        if plane < 1 or plane > 4:
            raise ValueError(
                "step2.efficiency_reference_planes contains an invalid plane index "
                f"{plane}. Valid values are 1, 2, 3, 4."
            )
        if plane not in planes:
            planes.append(plane)

    return [f"eff_empirical_{plane}" for plane in planes]


def _resolve_efficiency_reference_min(config: dict[str, Any]) -> float | None:
    step2_config = config.get("step2", {})
    if not isinstance(step2_config, dict):
        step2_config = {}

    raw_min = step2_config.get("efficiency_reference_min")
    if raw_min in (None, "", "null", "None"):
        return None

    minimum = float(raw_min)
    if minimum < 0.0:
        raise ValueError("step2.efficiency_reference_min must be >= 0.")
    return minimum


def _validate_columns(dataframe: pd.DataFrame, reference_columns: list[str]) -> None:
    required = ["selected_rate_hz", *reference_columns]
    missing = sorted(column for column in required if column not in dataframe.columns)
    if missing:
        raise ValueError(
            "Input data is missing required columns: " + ", ".join(missing)
        )


def _apply_simplified_scale(dataframe: pd.DataFrame, reference_columns: list[str]) -> pd.DataFrame:
    work = dataframe.copy()
    work["eff_reference"] = work[reference_columns].astype(float).mean(axis=1)
    work["scale_factor"] = 1.0 / (work["eff_reference"] ** 4)
    work["corrected_rate_hz"] = work["selected_rate_hz"].astype(float) * work["scale_factor"].astype(float)
    return work


def _filter_by_eff_reference_min(dataframe: pd.DataFrame, minimum: float | None) -> tuple[pd.DataFrame, int]:
    if minimum is None:
        return dataframe.copy(), 0
    keep_mask = dataframe["eff_reference"].astype(float) >= float(minimum)
    filtered = dataframe.loc[keep_mask].copy()
    removed_rows = int((~keep_mask).sum())
    return filtered, removed_rows


def _resolve_time_axis(dataframe: pd.DataFrame) -> tuple[pd.Series, str]:
    if "file_timestamp_utc" in dataframe.columns:
        parsed = pd.to_datetime(dataframe["file_timestamp_utc"], errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed, "file_timestamp_utc"
    if "execution_timestamp_utc" in dataframe.columns:
        parsed = pd.to_datetime(dataframe["execution_timestamp_utc"], errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed, "execution_timestamp_utc"
    return pd.Series(range(len(dataframe)), dtype=float), "row_index"


def _plot_rate_series(dataframe: pd.DataFrame, output_path: Path) -> None:
    x_values, x_label = _resolve_time_axis(dataframe)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        x_values,
        dataframe["selected_rate_hz"].astype(float),
        marker="o",
        markersize=3,
        linewidth=1.4,
        label="Original selected_rate_hz",
    )
    ax.plot(
        x_values,
        dataframe["corrected_rate_hz"].astype(float),
        marker="o",
        markersize=3,
        linewidth=1.4,
        label="Corrected rate",
    )
    ax.set_title("Original and corrected rate time series")
    ax.set_xlabel(x_label.replace("_", " "))
    ax.set_ylabel("Rate [Hz]")
    ax.grid(alpha=0.25)
    ax.legend()
    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_reference_efficiency_series(
    dataframe: pd.DataFrame,
    output_path: Path,
    reference_columns: list[str],
) -> None:
    x_values, x_label = _resolve_time_axis(dataframe)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, column in enumerate(reference_columns):
        ax.plot(
            x_values,
            dataframe[column].astype(float),
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=column,
            color=colors[idx % len(colors)],
        )
    ax.plot(
        x_values,
        dataframe["eff_reference"].astype(float),
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="eff_reference",
        color="black",
    )
    selected_planes_label = ", ".join(column.replace("eff_empirical_", "P") for column in reference_columns)
    ax.set_title(f"Selected empirical efficiencies over time | reference = mean({selected_planes_label})")
    ax.set_xlabel(x_label.replace("_", " "))
    ax.set_ylabel("Empirical efficiency")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_eff_reference_vs_rates_scatter(dataframe: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    eff_reference = pd.to_numeric(dataframe["eff_reference"], errors="coerce")
    selected_rate = pd.to_numeric(dataframe["selected_rate_hz"], errors="coerce")
    corrected_rate = pd.to_numeric(dataframe["corrected_rate_hz"], errors="coerce")

    selected_color = "#1f77b4"
    corrected_color = "#ff7f0e"

    ax.scatter(
        eff_reference,
        selected_rate,
        s=14,
        alpha=0.65,
        label="selected_rate_hz",
        color=selected_color,
    )
    ax.scatter(
        eff_reference,
        corrected_rate,
        s=14,
        alpha=0.65,
        label="corrected_rate_hz",
        color=corrected_color,
    )

    def _add_linear_trend_line(x_values: pd.Series, y_values: pd.Series, color: str, label: str) -> None:
        valid = x_values.notna() & y_values.notna()
        if int(valid.sum()) < 2:
            return
        x = x_values.loc[valid].to_numpy(dtype=float)
        y = y_values.loc[valid].to_numpy(dtype=float)
        x_min = float(np.nanmin(x))
        x_line_end = 1.01
        x_line_start = min(x_min, x_line_end)
        if x_line_end <= x_line_start:
            return
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x_line_start, x_line_end, 120)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, linestyle="--", linewidth=1.8, color=color, label=label)

    _add_linear_trend_line(eff_reference, selected_rate, selected_color, "selected trend")
    _add_linear_trend_line(eff_reference, corrected_rate, corrected_color, "corrected trend")
    ax.axvline(1.0, linestyle=":", linewidth=1.5, color="black", label="eff_reference = 1")

    ax.set_title("Rate vs eff_reference")
    ax.set_xlabel("eff_reference")
    ax.set_ylabel("Rate [Hz]")
    ax.grid(alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    input_path = cfg_path(config, "paths", "output_csv")
    output_path = cfg_path(config, "paths", "step2_scaled_output_csv")
    reference_columns = _resolve_efficiency_reference_columns(config)
    eff_reference_min = _resolve_efficiency_reference_min(config)

    dataframe = pd.read_csv(input_path, low_memory=False)
    _validate_columns(dataframe, reference_columns)
    scaled = _apply_simplified_scale(dataframe, reference_columns)
    scaled, removed_rows = _filter_by_eff_reference_min(scaled, eff_reference_min)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_csv(output_path, index=False)

    plot_path = output_path.parent.parent / "PLOTS" / "real_data_selected_vs_corrected_rate.png"
    _plot_rate_series(scaled, plot_path)
    eff_plot_path = output_path.parent.parent / "PLOTS" / "real_data_eff2_eff3_series.png"
    _plot_reference_efficiency_series(scaled, eff_plot_path, reference_columns)
    scatter_plot_path = output_path.parent.parent / "PLOTS" / "real_data_eff_reference_vs_rate_scatter.png"
    _plot_eff_reference_vs_rates_scatter(scaled, scatter_plot_path)

    logging.info(
        "Wrote simplified scaled output with %d rows to %s",
        len(scaled),
        output_path,
    )
    logging.info("Efficiency reference columns used: %s", reference_columns)
    if eff_reference_min is not None:
        logging.info(
            "Applied eff_reference minimum %.3f and dropped %d rows below threshold.",
            eff_reference_min,
            removed_rows,
        )
    logging.info("Wrote reference-efficiency time series plot to %s", eff_plot_path)
    logging.info("Wrote original vs corrected time series plot to %s", plot_path)
    logging.info("Wrote eff_reference vs rates scatter plot to %s", scatter_plot_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a simplified scale factor based on mean(selected empirical efficiencies)^4."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
