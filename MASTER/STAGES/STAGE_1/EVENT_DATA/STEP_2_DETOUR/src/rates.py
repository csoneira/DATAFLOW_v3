from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.gates import GateDefinition


RATE_TIME_COLUMN = "datetime_minute"
LIVE_TIME_SECONDS_COLUMN = "live_time_seconds"
TOTAL_EVENTS_COUNT_COLUMN = "total_events_count"
TOTAL_EVENTS_HZ_COLUMN = "total_events_hz"
POST_TT_COLUMN = "post_tt"
EFFICIENCY_TOPOLOGIES = (1234, 123, 124, 134, 234)

_OBSERVED_START_COLUMN = "__observed_start"
_OBSERVED_END_COLUMN = "__observed_end"
_AVAILABLE_SECONDS_COLUMN = "__available_seconds"
_FULL_BIN_COLUMN = "__full_bin"
_INTERNAL_COLUMNS = {
    _OBSERVED_START_COLUMN,
    _OBSERVED_END_COLUMN,
    _AVAILABLE_SECONDS_COLUMN,
    _FULL_BIN_COLUMN,
}


def gate_count_column(gate_name: str) -> str:
    return f"{gate_name}_count"


def gate_hz_column(gate_name: str) -> str:
    return f"{gate_name}_hz"


def gate_fraction_column(gate_name: str) -> str:
    return f"{gate_name}_fraction"


def gate_percent_column(gate_name: str) -> str:
    return f"{gate_name}_percent"


def gate_topology_count_column(gate_name: str, topology_code: int) -> str:
    return f"{gate_name}_post_tt_{topology_code}_count"


def gate_topology_hz_column(gate_name: str, topology_code: int) -> str:
    return f"{gate_name}_post_tt_{topology_code}_hz"


def gate_efficiency_column(gate_name: str, plane_index: int) -> str:
    return f"{gate_name}_eff_{plane_index}"


def build_chunk_rates(
    df: pd.DataFrame,
    gate_mask: np.ndarray,
    gates: list[GateDefinition],
    time_column: str,
    bin_size: str,
    selection_start: datetime,
    selection_end: datetime,
) -> pd.DataFrame:
    if time_column not in df.columns:
        raise ValueError(f"Rate output requires missing time column '{time_column}'.")
    if POST_TT_COLUMN not in df.columns:
        raise ValueError(f"Rate output requires missing topology column '{POST_TT_COLUMN}'.")

    bin_delta = pd.to_timedelta(bin_size)
    if bin_delta <= pd.Timedelta(0):
        raise ValueError(f"Rate output bin_size must be positive, received '{bin_size}'.")

    selection_start_ts = pd.Timestamp(selection_start)
    selection_end_ts = pd.Timestamp(selection_end)
    if selection_start_ts >= selection_end_ts:
        raise ValueError("Rate output requires selection_start to be earlier than selection_end.")

    all_timestamps = pd.to_datetime(df[time_column], errors="coerce")
    if all_timestamps.isna().any():
        raise ValueError(f"Rate output found non-parseable timestamps in '{time_column}'.")

    selection_mask = (all_timestamps >= selection_start_ts) & (all_timestamps < selection_end_ts)
    if not selection_mask.any():
        return _empty_chunk_rate_table(gates)

    selected_timestamps = all_timestamps.loc[selection_mask]
    selected_gate_mask = gate_mask[selection_mask.to_numpy(dtype=bool)]
    selected_topologies = pd.to_numeric(
        df.loc[selection_mask, POST_TT_COLUMN],
        errors="coerce",
    ).astype("Int64")
    time_bins = selected_timestamps.dt.floor(bin_size)
    working = pd.DataFrame(
        {
            RATE_TIME_COLUMN: time_bins,
            "__event_time": selected_timestamps,
            "__post_tt": selected_topologies.to_numpy(),
        }
    )

    chunk_rates = (
        working.groupby(RATE_TIME_COLUMN)
        .agg(
            **{
                TOTAL_EVENTS_COUNT_COLUMN: ("__event_time", "size"),
                _OBSERVED_START_COLUMN: ("__event_time", "min"),
                _OBSERVED_END_COLUMN: ("__event_time", "max"),
            }
        )
        .sort_index()
    )

    for gate in gates:
        in_gate = (selected_gate_mask & gate.bit_value) != 0
        gate_counts = working.loc[in_gate].groupby(RATE_TIME_COLUMN).size()
        chunk_rates[gate_count_column(gate.name)] = gate_counts
        for topology_code in EFFICIENCY_TOPOLOGIES:
            topology_matches = (working["__post_tt"] == topology_code).fillna(False).to_numpy(dtype=bool)
            topology_counts = working.loc[in_gate & topology_matches].groupby(RATE_TIME_COLUMN).size()
            chunk_rates[gate_topology_count_column(gate.name, topology_code)] = topology_counts

    chunk_rates = chunk_rates.fillna(0)
    count_columns = _count_columns(chunk_rates)
    for column in count_columns:
        chunk_rates[column] = chunk_rates[column].astype("uint64")

    file_start_time = all_timestamps.min()
    file_end_time = all_timestamps.max()
    first_selected_bin = time_bins.min()
    last_selected_bin = time_bins.max()

    available_seconds: list[float] = []
    clipped_observed_starts: list[pd.Timestamp] = []
    clipped_observed_ends: list[pd.Timestamp] = []
    full_bin_flags: list[bool] = []

    for minute_start in chunk_rates.index:
        overlap_start, overlap_end = _selection_overlap_bounds(
            minute_start=minute_start,
            bin_delta=bin_delta,
            selection_start=selection_start_ts,
            selection_end=selection_end_ts,
        )
        available = max((overlap_end - overlap_start).total_seconds(), 0.0)

        observed_start = max(chunk_rates.at[minute_start, _OBSERVED_START_COLUMN], overlap_start)
        observed_end = min(chunk_rates.at[minute_start, _OBSERVED_END_COLUMN], overlap_end)
        observed_span = max((observed_end - observed_start).total_seconds(), 0.0)

        starts_covered = (minute_start > first_selected_bin) or (file_start_time <= overlap_start)
        ends_covered = (minute_start < last_selected_bin) or (file_end_time >= overlap_end)
        full_bin = bool(available > 0 and starts_covered and ends_covered)

        # Approximation: if observed timestamps nearly span the full available interval,
        # treat the bin as fully covered even when it lands on a file edge.
        if available > 0 and observed_span >= max(available - 1.0, 0.0):
            full_bin = True

        available_seconds.append(available)
        clipped_observed_starts.append(observed_start)
        clipped_observed_ends.append(observed_end)
        full_bin_flags.append(full_bin)

    chunk_rates[_AVAILABLE_SECONDS_COLUMN] = pd.Series(available_seconds, index=chunk_rates.index, dtype="float64")
    chunk_rates[_OBSERVED_START_COLUMN] = clipped_observed_starts
    chunk_rates[_OBSERVED_END_COLUMN] = clipped_observed_ends
    chunk_rates[_FULL_BIN_COLUMN] = pd.Series(full_bin_flags, index=chunk_rates.index, dtype="bool")
    chunk_rates.index.name = RATE_TIME_COLUMN

    return chunk_rates


def accumulate_rate_tables(
    current_rates: pd.DataFrame | None,
    chunk_rates: pd.DataFrame,
) -> pd.DataFrame:
    if chunk_rates.empty:
        if current_rates is None:
            return chunk_rates.copy()
        return current_rates.copy()

    if current_rates is None or current_rates.empty:
        return chunk_rates.sort_index().copy()

    merged_index = current_rates.index.union(chunk_rates.index).sort_values()
    merged = pd.DataFrame(index=merged_index)

    for column in sorted(set(_count_columns(current_rates)) | set(_count_columns(chunk_rates))):
        merged[column] = (
            _series_or_default(current_rates, column, merged_index, 0)
            .fillna(0)
            .astype("uint64")
            + _series_or_default(chunk_rates, column, merged_index, 0).fillna(0).astype("uint64")
        ).astype("uint64")

    merged[_OBSERVED_START_COLUMN] = pd.concat(
        [
            _series_or_default(current_rates, _OBSERVED_START_COLUMN, merged_index, pd.NaT),
            _series_or_default(chunk_rates, _OBSERVED_START_COLUMN, merged_index, pd.NaT),
        ],
        axis=1,
    ).min(axis=1)
    merged[_OBSERVED_END_COLUMN] = pd.concat(
        [
            _series_or_default(current_rates, _OBSERVED_END_COLUMN, merged_index, pd.NaT),
            _series_or_default(chunk_rates, _OBSERVED_END_COLUMN, merged_index, pd.NaT),
        ],
        axis=1,
    ).max(axis=1)
    merged[_AVAILABLE_SECONDS_COLUMN] = pd.concat(
        [
            _series_or_default(current_rates, _AVAILABLE_SECONDS_COLUMN, merged_index, np.nan),
            _series_or_default(chunk_rates, _AVAILABLE_SECONDS_COLUMN, merged_index, np.nan),
        ],
        axis=1,
    ).max(axis=1)
    merged[_FULL_BIN_COLUMN] = (
        _boolean_series_or_default(current_rates, _FULL_BIN_COLUMN, merged_index)
        | _boolean_series_or_default(chunk_rates, _FULL_BIN_COLUMN, merged_index)
    )
    merged.index.name = RATE_TIME_COLUMN

    return merged.sort_index()


def finalize_rate_table(
    accumulated_rates: pd.DataFrame,
    gates: list[GateDefinition],
) -> pd.DataFrame:
    if accumulated_rates.empty:
        final_columns = [LIVE_TIME_SECONDS_COLUMN, TOTAL_EVENTS_COUNT_COLUMN, TOTAL_EVENTS_HZ_COLUMN]
        for gate in gates:
            final_columns.extend(
                [
                    gate_count_column(gate.name),
                    gate_hz_column(gate.name),
                    gate_fraction_column(gate.name),
                    gate_percent_column(gate.name),
                ]
            )
            for topology_code in EFFICIENCY_TOPOLOGIES:
                final_columns.extend(
                    [
                        gate_topology_count_column(gate.name, topology_code),
                        gate_topology_hz_column(gate.name, topology_code),
                    ]
                )
        return pd.DataFrame(columns=final_columns).rename_axis(RATE_TIME_COLUMN)

    finalized = pd.DataFrame(index=accumulated_rates.sort_index().index)
    finalized.index.name = RATE_TIME_COLUMN

    live_time_seconds = _compute_live_time_seconds(accumulated_rates)
    finalized[LIVE_TIME_SECONDS_COLUMN] = live_time_seconds
    finalized[TOTAL_EVENTS_COUNT_COLUMN] = accumulated_rates[TOTAL_EVENTS_COUNT_COLUMN].astype("uint64")
    finalized[TOTAL_EVENTS_HZ_COLUMN] = _counts_to_hz(
        finalized[TOTAL_EVENTS_COUNT_COLUMN],
        live_time_seconds,
    )

    total_count_float = finalized[TOTAL_EVENTS_COUNT_COLUMN].astype("float64")
    for gate in gates:
        count_column = gate_count_column(gate.name)
        hz_column = gate_hz_column(gate.name)
        fraction_column = gate_fraction_column(gate.name)
        percent_column = gate_percent_column(gate.name)

        finalized[count_column] = accumulated_rates.get(
            count_column,
            pd.Series(0, index=finalized.index, dtype="uint64"),
        ).astype("uint64")
        finalized[hz_column] = _counts_to_hz(finalized[count_column], live_time_seconds)
        finalized[fraction_column] = np.where(
            total_count_float > 0,
            finalized[count_column].astype("float64") / total_count_float,
            np.nan,
        )
        finalized[percent_column] = finalized[fraction_column] * 100.0
        for topology_code in EFFICIENCY_TOPOLOGIES:
            topology_count_column = gate_topology_count_column(gate.name, topology_code)
            topology_hz_column = gate_topology_hz_column(gate.name, topology_code)
            finalized[topology_count_column] = accumulated_rates.get(
                topology_count_column,
                pd.Series(0, index=finalized.index, dtype="uint64"),
            ).astype("uint64")
            finalized[topology_hz_column] = _counts_to_hz(
                finalized[topology_count_column],
                live_time_seconds,
            )

    return finalized


def write_rate_table(rate_table: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame = rate_table.reset_index()

    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        try:
            output_frame.to_parquet(output_path, index=False)
            return output_path
        except Exception:
            fallback_path = output_path.with_suffix(".csv")
            output_frame.to_csv(fallback_path, index=False)
            return fallback_path

    if suffix == ".csv":
        output_frame.to_csv(output_path, index=False)
        return output_path

    raise ValueError("Rate output file must end with .parquet or .csv.")


def _count_columns(rate_table: pd.DataFrame) -> list[str]:
    return [column for column in rate_table.columns if column.endswith("_count")]


def _empty_chunk_rate_table(gates: list[GateDefinition]) -> pd.DataFrame:
    columns = [TOTAL_EVENTS_COUNT_COLUMN]
    for gate in gates:
        columns.append(gate_count_column(gate.name))
        for topology_code in EFFICIENCY_TOPOLOGIES:
            columns.append(gate_topology_count_column(gate.name, topology_code))
    columns.extend(
        [
            _OBSERVED_START_COLUMN,
            _OBSERVED_END_COLUMN,
            _AVAILABLE_SECONDS_COLUMN,
            _FULL_BIN_COLUMN,
        ]
    )
    empty = pd.DataFrame(columns=columns)
    empty.index = pd.DatetimeIndex([], name=RATE_TIME_COLUMN)
    return empty


def _selection_overlap_bounds(
    minute_start: pd.Timestamp,
    bin_delta: pd.Timedelta,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    bin_end = minute_start + bin_delta
    overlap_start = max(minute_start, selection_start)
    overlap_end = min(bin_end, selection_end)
    return overlap_start, overlap_end


def _series_or_default(
    frame: pd.DataFrame,
    column: str,
    index: pd.Index,
    default_value: object,
) -> pd.Series:
    if column in frame.columns:
        return frame[column].reindex(index)
    return pd.Series(default_value, index=index)


def _boolean_series_or_default(
    frame: pd.DataFrame,
    column: str,
    index: pd.Index,
) -> pd.Series:
    if column in frame.columns:
        series = frame[column].reindex(index)
    else:
        series = pd.Series(pd.array([False] * len(index), dtype="boolean"), index=index)
    return series.astype("boolean").fillna(False).astype(bool)


def _compute_live_time_seconds(accumulated_rates: pd.DataFrame) -> pd.Series:
    available_seconds = accumulated_rates[_AVAILABLE_SECONDS_COLUMN].astype("float64")
    observed_span_seconds = (
        accumulated_rates[_OBSERVED_END_COLUMN] - accumulated_rates[_OBSERVED_START_COLUMN]
    ).dt.total_seconds()
    observed_span_seconds = observed_span_seconds.clip(lower=0)

    live_time = observed_span_seconds.where(~accumulated_rates[_FULL_BIN_COLUMN], available_seconds)
    live_time = live_time.clip(upper=available_seconds)
    live_time = live_time.where(live_time > 0, np.nan)
    return live_time.astype("float64")


def _counts_to_hz(counts: pd.Series, live_time_seconds: pd.Series) -> pd.Series:
    live_time = live_time_seconds.astype("float64")
    count_values = counts.astype("float64")
    rates = np.where(live_time > 0, count_values / live_time, np.nan)
    return pd.Series(rates, index=counts.index, dtype="float64")
