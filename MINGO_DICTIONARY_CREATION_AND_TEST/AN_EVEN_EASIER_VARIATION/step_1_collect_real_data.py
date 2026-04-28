#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    CANONICAL_EFF_COLUMNS,
    DEFAULT_CONFIG_PATH,
    ensure_output_dirs,
    get_trigger_type_selection,
    load_config,
    cfg_path,
    derive_trigger_rate_features,
)

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parents[1]
STATIONS_ROOT = REPO_ROOT / "STATIONS"

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FILE_TS_RE = re.compile(r"(\d{11})$")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_1 - %(message)s", level=logging.INFO, force=True)


def _parse_station_id(raw: object) -> int:
    if raw in (None, "", "null", "None"):
        raise ValueError("step5.station must not be empty.")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)
    text = str(raw).strip()
    match = re.fullmatch(r"(?i)MINGO(\d{1,2})", text)
    if match is not None:
        return int(match.group(1))
    return int(text)


def _parse_time_bound(value: object, *, end_of_day: bool) -> pd.Timestamp | None:
    if value in (None, "", "null", "None"):
        return None
    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Could not parse datetime bound: {value!r}")
    if end_of_day and _DATE_ONLY_RE.fullmatch(text):
        parsed = parsed + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return pd.Timestamp(parsed)


def _parse_filename_base_ts(value: object) -> pd.Timestamp:
    text = str(value).strip().lower()
    if text.startswith("mini"):
        text = "mi01" + text[4:]
    match = _FILE_TS_RE.search(text)
    if match is None:
        return pd.NaT
    stamp = match.group(1)
    try:
        year = 2000 + int(stamp[0:2])
        day_of_year = int(stamp[2:5])
        hour = int(stamp[5:7])
        minute = int(stamp[7:9])
        second = int(stamp[9:11])
        dt = datetime(year, 1, 1) + timedelta(
            days=day_of_year - 1,
            hours=hour,
            minutes=minute,
            seconds=second,
        )
    except ValueError:
        return pd.NaT
    return pd.Timestamp(dt, tz="UTC")


def _parse_execution_timestamp(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    parsed = pd.to_datetime(text, format="%Y-%m-%d_%H.%M.%S", errors="coerce", utc=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce", utc=True)
    return parsed


def _aggregate_latest_per_file(dataframe: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    work = dataframe.copy()
    if timestamp_column in work.columns:
        work["_exec_dt"] = _parse_execution_timestamp(work[timestamp_column])
        work = work.sort_values(["filename_base", "_exec_dt"], na_position="last", kind="mergesort")
        work = work.groupby("filename_base").tail(1).drop(columns=["_exec_dt"])
    return work.groupby("filename_base", sort=False).tail(1).reset_index(drop=True)


def _task_metadata_dir(station_id: int, task_id: int) -> Path:
    station = f"MINGO{station_id:02d}"
    return (
        STATIONS_ROOT
        / station
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{task_id}"
        / "METADATA"
    )


def _task_metadata_path(station_id: int, task_id: int, source_name: str) -> Path:
    return _task_metadata_dir(station_id, task_id) / f"task_{task_id}_metadata_{source_name}.csv"


def _load_task_metadata_source_csv(
    *,
    station_id: int,
    task_id: int,
    source_name: str,
    metadata_agg: str,
    timestamp_column: str,
) -> pd.DataFrame:
    source_path = _task_metadata_path(station_id, task_id, source_name)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing required {source_name} metadata for task {task_id}: {source_path}")
    dataframe = pd.read_csv(source_path, low_memory=False)
    if "filename_base" not in dataframe.columns:
        raise KeyError(f"Task {task_id} {source_name} metadata has no 'filename_base' column: {source_path}")
    if str(metadata_agg).strip().lower() == "latest":
        dataframe = _aggregate_latest_per_file(dataframe, timestamp_column)
    return dataframe


def _merge_sources(
    base_source: tuple[str, pd.DataFrame],
    extra_sources: list[tuple[str, pd.DataFrame]],
    *,
    how: str,
) -> pd.DataFrame:
    merged = base_source[1].copy()
    for source_name, source_df in extra_sources:
        overlap = sorted(set(merged.columns).intersection(set(source_df.columns)) - {"filename_base"})
        renamed = source_df.rename(columns={column: f"{source_name}__{column}" for column in overlap})
        merged = merged.merge(renamed, on="filename_base", how=how)
    return merged


def _normalize_task_ids(raw: object, fallback: list[int]) -> list[int]:
    if raw in (None, "", "null", "None"):
        return list(fallback)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return [int(raw)]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return list(fallback)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = [piece.strip() for piece in text.split(",") if piece.strip()]
        raw = decoded
    if isinstance(raw, (list, tuple)):
        parsed = []
        for value in raw:
            try:
                parsed.append(int(value))
            except (TypeError, ValueError):
                continue
        return sorted(set(parsed)) or list(fallback)
    return list(fallback)


def _collect_real_data_slice(
    *,
    config: dict[str, Any],
    station_id: int,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
    min_events: float | None,
    metadata_agg: str,
    timestamp_column: str,
) -> pd.DataFrame:
    trigger_selection = get_trigger_type_selection(config)
    metadata_source = str(trigger_selection.get("metadata_source", "trigger_type"))
    source_name = str(trigger_selection.get("source_name", "trigger_type"))
    if metadata_source == "robust_efficiency":
        task_ids = [int(trigger_selection.get("metadata_task_id", trigger_selection["task_id"]))]
    else:
        task_ids = _normalize_task_ids(
            config.get("step5", {}).get("task_ids"),
            [int(trigger_selection.get("metadata_task_id", trigger_selection["task_id"]))],
        )

    sources: list[tuple[str, pd.DataFrame]] = []
    for task_id in task_ids:
        task_df = _load_task_metadata_source_csv(
            station_id=station_id,
            task_id=task_id,
            source_name=source_name,
            metadata_agg=metadata_agg,
            timestamp_column=timestamp_column,
        )
        sources.append((f"task_{task_id}", task_df))

    merged = sources[0][1].copy()
    if len(sources) > 1:
        merged = _merge_sources(sources[0], sources[1:], how="outer")

    merged, _ = derive_trigger_rate_features(merged, config, allow_plain_fallback=False)
    merged["file_timestamp_utc"] = merged["filename_base"].map(_parse_filename_base_ts)
    if timestamp_column in merged.columns:
        merged["execution_timestamp_utc"] = _parse_execution_timestamp(merged[timestamp_column])
    else:
        merged["execution_timestamp_utc"] = pd.NaT

    keep = merged["file_timestamp_utc"].notna()
    if date_from is not None:
        keep &= merged["file_timestamp_utc"] >= date_from
    if date_to is not None:
        keep &= merged["file_timestamp_utc"] <= date_to
    collected = merged.loc[keep].copy()
    if collected.empty:
        raise ValueError("No real rows were collected for the requested station/date window.")

    event_values = pd.to_numeric(collected.get("selected_rate_count"), errors="coerce")
    if event_values.notna().any():
        collected["n_events"] = event_values
    if min_events is not None and event_values.notna().any():
        collected = collected.loc[event_values >= float(min_events)].copy()
    if collected.empty:
        raise ValueError("No real rows remain after Step 1 event-count filtering.")

    sort_column = "file_timestamp_utc" if "file_timestamp_utc" in collected.columns else "execution_timestamp_utc"
    collected = collected.sort_values(sort_column, kind="mergesort").reset_index(drop=True)

    return collected


def _selected_output_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "filename_base",
        "file_timestamp_utc",
        "execution_timestamp_utc",
        "selected_rate_hz",
        "selected_rate_count",
        "count_rate_denominator_seconds",
        "four_plane_rate_hz",
        "four_plane_count",
        "four_plane_robust_hz",
        "four_plane_robust_count",
        "four_plane_robust_hz_union",
        "four_plane_robust_count_union",
        "four_plane_robust_hz_intersection",
        "four_plane_robust_count_intersection",
        "total_rate_hz",
        "total_count",
        *CANONICAL_EFF_COLUMNS,
    ]
    available = [column for column in columns if column in dataframe.columns]
    return dataframe[available].copy()


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    step5_config = config.get("step5", {})
    if not isinstance(step5_config, dict):
        step5_config = {}

    station_id = _parse_station_id(step5_config.get("station", "MINGO01"))
    date_from = _parse_time_bound(step5_config.get("date_from"), end_of_day=False)
    date_to = _parse_time_bound(step5_config.get("date_to"), end_of_day=True)
    min_events_raw = step5_config.get("min_events")
    min_events = None if min_events_raw in (None, "", "null", "None") else float(min_events_raw)
    metadata_agg = str(step5_config.get("metadata_agg", "latest")).strip().lower()
    timestamp_column = str(step5_config.get("timestamp_column", "execution_timestamp"))

    trigger_selection = get_trigger_type_selection(config)
    output_path = cfg_path(config, "paths", "output_csv")
    collected = _collect_real_data_slice(
        config=config,
        station_id=station_id,
        date_from=date_from,
        date_to=date_to,
        min_events=min_events,
        metadata_agg=metadata_agg,
        timestamp_column=timestamp_column,
    )

    if "selected_rate_hz" not in collected.columns:
        raise ValueError("Collected real data does not contain the selected rate column 'selected_rate_hz'.")

    expected = set(["selected_rate_hz", *CANONICAL_EFF_COLUMNS])
    missing_expected = sorted(column for column in expected if column not in collected.columns)
    if missing_expected:
        raise ValueError(
            "Collected real data is missing required columns: " + ", ".join(missing_expected)
        )

    selected = _selected_output_dataframe(collected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)

    logging.info("Wrote selected real-data file with %d rows to %s", len(selected), output_path)
    logging.info(
        "Selected source column: %s | selected count column: %s",
        trigger_selection.get("selected_source_rate_column"),
        trigger_selection.get("selected_count_column"),
    )
    if str(trigger_selection.get("metadata_source")) == "robust_efficiency":
        logging.info(
            "Robust efficiency variant used for eff_empirical_<plane>: %s",
            trigger_selection.get("robust_efficiency_variant"),
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real station data and write selected rate + empirical efficiencies.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
