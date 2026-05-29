from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.gates import GateDefinition, load_gate_config
from src.plotting import plot_diagnostic_table
from src.rates import (
    EFFICIENCY_TOPOLOGIES,
    LIVE_TIME_SECONDS_COLUMN,
    RATE_TIME_COLUMN,
    TOTAL_EVENTS_COUNT_COLUMN,
    TOTAL_EVENTS_HZ_COLUMN,
    gate_count_column,
    gate_efficiency_column,
    gate_hz_column,
    gate_percent_column,
    gate_topology_count_column,
    gate_topology_hz_column,
)
from src.select_files import SelectionConfig, load_selection_config


PLANE_TOPOLOGY_MAP = {
    1: 234,
    2: 134,
    3: 124,
    4: 123,
}


@dataclass(frozen=True)
class DiagnosticResult:
    selection: SelectionConfig
    gates: list[GateDefinition]
    input_rate_path: Path
    diagnostic_output_path: Path
    plot_output_paths: list[Path]


def build_diagnostic_table(
    rate_table: pd.DataFrame,
    gates: list[GateDefinition],
) -> pd.DataFrame:
    if RATE_TIME_COLUMN not in rate_table.columns:
        raise ValueError(f"Rate table is missing required column '{RATE_TIME_COLUMN}'.")

    diagnostic = rate_table.copy()
    diagnostic[RATE_TIME_COLUMN] = pd.to_datetime(diagnostic[RATE_TIME_COLUMN], errors="coerce")
    if diagnostic[RATE_TIME_COLUMN].isna().any():
        raise ValueError("Rate table contains invalid datetime values.")
    diagnostic = diagnostic.sort_values(RATE_TIME_COLUMN).reset_index(drop=True)

    _ensure_total_rate_columns(diagnostic)

    active_gates = _filter_available_gates(rate_table=diagnostic, gates=gates)
    for gate in active_gates:
        _ensure_gate_metric_columns(diagnostic, gate)
        for plane_index, missing_topology in PLANE_TOPOLOGY_MAP.items():
            numerator = pd.to_numeric(
                diagnostic[gate_topology_count_column(gate.name, 1234)],
                errors="coerce",
            )
            denominator = numerator + pd.to_numeric(
                diagnostic[gate_topology_count_column(gate.name, missing_topology)],
                errors="coerce",
            )
            diagnostic[gate_efficiency_column(gate.name, plane_index)] = np.where(
                denominator > 0,
                numerator / denominator,
                np.nan,
            )

    return diagnostic


def run_diagnostics(
    selection_config_path: Path,
    gate_config_path: Path,
    rate_file: Path | None = None,
) -> DiagnosticResult:
    selection = load_selection_config(selection_config_path)
    gates = load_gate_config(gate_config_path)

    resolved_input_rate = _resolve_input_rate_path(selection, rate_file)
    rate_table = _read_table(resolved_input_rate)
    active_gates = _filter_available_gates(rate_table=rate_table, gates=gates)
    if not active_gates:
        raise ValueError("No gate columns in the rate table match the configured gates.")
    diagnostic_table = build_diagnostic_table(rate_table, active_gates)

    diagnostic_output_path = _resolve_diagnostic_output_path(resolved_input_rate)
    write_table(diagnostic_table, diagnostic_output_path)

    plot_output_paths: list[Path] = []
    if selection.plotting.enabled:
        selected_gate_names = _resolve_selected_gate_names(
            active_gates,
            selection.plotting.gate_columns,
            rate_table=diagnostic_table,
        )
        plot_output_paths = plot_diagnostic_table(
            diagnostic_table=diagnostic_table,
            output_path=selection.resolve_output_path(selection.plotting.output_file),
            gate_names=selected_gate_names,
            moving_average_minutes=selection.rate_output.moving_average_minutes,
            rate_mode=selection.plotting.rate_mode,
            title_context=_build_plot_title_context(selection),
        )

    return DiagnosticResult(
        selection=selection,
        gates=active_gates,
        input_rate_path=resolved_input_rate,
        diagnostic_output_path=diagnostic_output_path,
        plot_output_paths=plot_output_paths,
    )


def write_table(dataframe: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        dataframe.to_parquet(output_path, index=False)
        return output_path
    if suffix == ".csv":
        dataframe.to_csv(output_path, index=False)
        return output_path
    raise ValueError("Diagnostic output file must end with .parquet or .csv.")


def _ensure_total_rate_columns(diagnostic: pd.DataFrame) -> None:
    if LIVE_TIME_SECONDS_COLUMN not in diagnostic.columns:
        raise ValueError(f"Rate table is missing required column '{LIVE_TIME_SECONDS_COLUMN}'.")
    if TOTAL_EVENTS_COUNT_COLUMN not in diagnostic.columns:
        raise ValueError(f"Rate table is missing required column '{TOTAL_EVENTS_COUNT_COLUMN}'.")

    if TOTAL_EVENTS_HZ_COLUMN not in diagnostic.columns:
        diagnostic[TOTAL_EVENTS_HZ_COLUMN] = _counts_to_hz(
            diagnostic[TOTAL_EVENTS_COUNT_COLUMN],
            diagnostic[LIVE_TIME_SECONDS_COLUMN],
        )


def _ensure_gate_metric_columns(diagnostic: pd.DataFrame, gate: GateDefinition) -> None:
    count_column = gate_count_column(gate.name)
    if count_column not in diagnostic.columns:
        raise ValueError(f"Rate table is missing required column '{count_column}'.")

    hz_column = gate_hz_column(gate.name)
    if hz_column not in diagnostic.columns:
        diagnostic[hz_column] = _counts_to_hz(
            diagnostic[count_column],
            diagnostic[LIVE_TIME_SECONDS_COLUMN],
        )

    percent_column = gate_percent_column(gate.name)
    if percent_column not in diagnostic.columns:
        total_events = pd.to_numeric(diagnostic[TOTAL_EVENTS_COUNT_COLUMN], errors="coerce")
        gate_counts = pd.to_numeric(diagnostic[count_column], errors="coerce")
        diagnostic[percent_column] = np.where(
            total_events > 0,
            100.0 * gate_counts / total_events,
            np.nan,
        )

    for topology_code in EFFICIENCY_TOPOLOGIES:
        topology_count_column = gate_topology_count_column(gate.name, topology_code)
        if topology_count_column not in diagnostic.columns:
            raise ValueError(f"Rate table is missing required column '{topology_count_column}'.")

        topology_hz_column = gate_topology_hz_column(gate.name, topology_code)
        if topology_hz_column not in diagnostic.columns:
            diagnostic[topology_hz_column] = _counts_to_hz(
                diagnostic[topology_count_column],
                diagnostic[LIVE_TIME_SECONDS_COLUMN],
            )


def _counts_to_hz(counts: pd.Series, live_time_seconds: pd.Series) -> pd.Series:
    count_values = pd.to_numeric(counts, errors="coerce")
    live_time_values = pd.to_numeric(live_time_seconds, errors="coerce")
    return pd.Series(
        np.where(live_time_values > 0, count_values / live_time_values, np.nan),
        index=counts.index,
        dtype="float64",
    )


def _resolve_input_rate_path(selection: SelectionConfig, rate_file: Path | None) -> Path:
    if rate_file is not None:
        if not rate_file.exists():
            raise FileNotFoundError(f"Provided rate file does not exist: {rate_file}")
        return rate_file

    configured_rate_path = selection.resolve_output_path(selection.rate_output.output_file)
    if configured_rate_path.exists():
        return configured_rate_path

    alternate_path = configured_rate_path.with_suffix(".csv" if configured_rate_path.suffix.lower() == ".parquet" else ".parquet")
    if alternate_path.exists():
        return alternate_path

    raise FileNotFoundError(f"Configured rate file does not exist: {configured_rate_path}")


def _resolve_diagnostic_output_path(input_rate_path: Path) -> Path:
    return input_rate_path.with_name(f"{input_rate_path.stem}_diagnostics{input_rate_path.suffix}")


def _resolve_selected_gate_names(
    gates: list[GateDefinition],
    configured_gate_names: list[str],
    rate_table: pd.DataFrame,
) -> list[str]:
    available_gate_names = _infer_available_gate_names(rate_table)
    if not configured_gate_names:
        return [gate.name for gate in gates if gate.name in available_gate_names]

    known_gate_names = {gate.name for gate in gates}
    missing_gate_names = [gate_name for gate_name in configured_gate_names if gate_name not in known_gate_names]
    if missing_gate_names:
        raise ValueError(f"Plotting requested unknown gates: {missing_gate_names}")
    resolved_gate_names = [gate_name for gate_name in configured_gate_names if gate_name in available_gate_names]
    if not resolved_gate_names:
        raise ValueError("None of the requested plotting gates are present in the rate table.")
    return resolved_gate_names


def _build_plot_title_context(selection: SelectionConfig) -> str:
    stations = ", ".join(selection.stations)
    return (
        f"Stations: {stations} | "
        f"{selection.start_datetime.isoformat()} to {selection.end_datetime.isoformat()}"
    )


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported rate file format: {path}")


def _infer_available_gate_names(rate_table: pd.DataFrame) -> list[str]:
    gate_names: list[str] = []
    for column in rate_table.columns:
        if not column.endswith("_count"):
            continue
        if column == TOTAL_EVENTS_COUNT_COLUMN or "_post_tt_" in column:
            continue
        gate_names.append(column[:-len("_count")])
    return gate_names


def _filter_available_gates(
    rate_table: pd.DataFrame,
    gates: list[GateDefinition],
) -> list[GateDefinition]:
    available_gate_names = set(_infer_available_gate_names(rate_table))
    return [gate for gate in gates if gate.name in available_gate_names]
