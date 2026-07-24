#!/usr/bin/env python3
"""Assign configurable binary gates to consecutive Stage 1 product events."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
import re
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.interpolate import CubicSpline

from calibration_context import generate_calibration_context
from mingo00_product_selection import (
    Mingo00Selection,
    select_mingo00_products,
    validate_close_parameters,
)


ANALYSIS_ROOT = Path(__file__).resolve().parents[3]
STATIONS_ROOT = ANALYSIS_ROOT / "MINGO_ANALYSIS_STATIONS"
DEFAULT_CONFIG = Path(__file__).with_name("config_test_3_event_gates.yaml")
TOT_TO_CHARGE_CALIBRATION = (
    ANALYSIS_ROOT
    / "MINGO_ANALYSIS_SCRIPTS"
    / "ANCILLARY"
    / "CALIBRATIONS_AND_LUTS"
    / "TOT_TO_CHARGE_CAL"
    / "tot_to_charge_calibration.csv"
)
OUTPUT_NAME = "TEST_3_CONFIGURABLE_GATES"
BASENAME_RE = re.compile(r"mi0[0-9]\d{11}")
COMBINATORS = frozenset({"all", "any", "not"})
NUMERIC_OPERATORS = frozenset({"lt", "le", "gt", "ge", "between"})


@dataclass(frozen=True)
class Product:
    path: Path
    basename: str
    acquired: datetime


@dataclass(frozen=True)
class Gate:
    code: str
    bit_value: int
    name: str
    condition: Mapping[str, Any]
    short_label: str = ""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def station_name(value: Any) -> str:
    text = str(value).strip().upper().removeprefix("MINGO")
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid station {value!r}; use 1, 01, or MINGO01") from exc
    if not 0 <= number <= 99:
        raise ValueError(f"Station number outside supported range: {number}")
    return f"MINGO{number:02d}"


def boundary(value: Any, *, is_end: bool) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        parsed = datetime.combine(value, time.min)
        return parsed + timedelta(days=1) - timedelta(microseconds=1) if is_end else parsed
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date/datetime: {value!r}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    if is_end and re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        parsed += timedelta(days=1) - timedelta(microseconds=1)
    return parsed


def acquisition_time(basename: str) -> datetime | None:
    if not BASENAME_RE.fullmatch(basename):
        return None
    stamp = basename[4:]
    try:
        year = 2000 + int(stamp[:2])
        doy = int(stamp[2:5])
        if not 1 <= doy <= 366:
            return None
        parsed = datetime(year, 1, 1) + timedelta(
            days=doy - 1,
            hours=int(stamp[5:7]),
            minutes=int(stamp[7:9]),
            seconds=int(stamp[9:11]),
        )
    except (ValueError, OverflowError):
        return None
    return parsed if parsed.year == year else None


def parquet_basename(path: Path) -> str | None:
    name = path.stem
    for prefix in ("postprocessed_", "fitted_", "listed_", "calibrated_", "cleaned_", "raw_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    return name if BASENAME_RE.fullmatch(name) else None


def load_configuration(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML mapping")
    required = ("station", "max_datafiles", "gates")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Missing config fields: " + ", ".join(missing))
    resolved_station = station_name(config["station"])
    if resolved_station == "MINGO00":
        close_parameters = validate_close_parameters(config.get("mingo00_close_parameters"))
        # MINGO00 acquisition dates are synthetic and must not restrict candidates.
        start = datetime(2000, 1, 1)
        end = datetime(2100, 1, 1) - timedelta(microseconds=1)
    else:
        date_missing = [key for key in ("start_date", "end_date") if key not in config]
        if date_missing:
            raise ValueError("Missing config fields: " + ", ".join(date_missing))
        start = boundary(config["start_date"], is_end=False)
        end = boundary(config["end_date"], is_end=True)
        close_parameters = (
            validate_close_parameters(config["mingo00_close_parameters"])
            if "mingo00_close_parameters" in config
            else ()
        )
    maximum = int(config["max_datafiles"])
    if start > end or maximum < 1:
        raise ValueError("Require start_date <= end_date and max_datafiles >= 1")
    config.update(
        station_name=resolved_station,
        start=start,
        end=end,
        maximum=maximum,
        close_parameters=close_parameters,
    )
    return config


def discover(lake: Path, start: datetime, end: datetime) -> list[Product]:
    if not lake.is_dir():
        raise FileNotFoundError(f"Parquet lake not found: {lake}")
    products: list[Product] = []
    for path in lake.glob("*.parquet"):
        basename = parquet_basename(path)
        acquired = acquisition_time(basename) if basename else None
        if basename and acquired and start <= acquired <= end:
            products.append(Product(path, basename, acquired))
    products.sort(key=lambda item: (item.acquired, item.basename))
    if not products:
        raise ValueError(f"No product files between {start.isoformat()} and {end.isoformat()}")
    return products


def tightest_block(files: list[Product], maximum: int) -> list[Product]:
    """Select the same minimum-span contiguous chronological block as Test 2."""
    count = min(maximum, len(files))
    if count == len(files):
        return files
    index = min(
        range(len(files) - count + 1),
        key=lambda i: (files[i + count - 1].acquired - files[i].acquired, files[i].acquired),
    )
    return files[index:index + count]


def schemas(files: list[Product]) -> tuple[list[str], dict[str, set[str]]]:
    columns: list[str] = []
    types: dict[str, set[str]] = {}
    for product in files:
        for field in pq.read_schema(product.path):
            if field.name not in types:
                columns.append(field.name)
            types.setdefault(field.name, set()).add(str(field.type))
    return columns, types


def show_columns(columns: list[str], types: dict[str, set[str]]) -> None:
    print(f"\nAvailable parquet columns in selected files ({len(columns)}):")
    for index, name in enumerate(columns, 1):
        print(f"  {index:03d}. {name:<52} [{' | '.join(sorted(types[name]))}]")
    print()


def parse_gates(raw_gates: Any) -> list[Gate]:
    if not isinstance(raw_gates, list) or not raw_gates:
        raise ValueError("gates must be a nonempty YAML list")
    gates: list[Gate] = []
    used_codes: set[str] = set()
    used_names: set[str] = set()
    used_labels: set[str] = set()
    for index, raw_gate in enumerate(raw_gates, 1):
        if not isinstance(raw_gate, dict):
            raise ValueError(f"gates item {index} must be a mapping")
        enabled = raw_gate.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"gates item {index} enabled must be true or false")
        code = str(raw_gate.get("code", "")).strip()
        if not re.fullmatch(r"10*", code):
            raise ValueError(
                f"Gate code {code!r} must be a binary power of two: 1, 10, 100, 1000, ..."
            )
        if len(code) > 62:
            raise ValueError("Gate codes are limited to 62 binary digits")
        name = str(raw_gate.get("name", f"gate_{code}")).strip()
        short_label = str(raw_gate.get("short_label", name)).strip()
        if not short_label or len(short_label) > 16:
            raise ValueError(f"Gate {code} short_label must contain 1..16 characters")
        condition = raw_gate.get("condition")
        if code in used_codes or name in used_names or short_label in used_labels:
            raise ValueError(f"Duplicate gate code, name, or short_label at gates item {index}")
        if not isinstance(condition, dict):
            raise ValueError(f"Gate {code} needs a condition mapping")
        used_codes.add(code)
        used_names.add(name)
        used_labels.add(short_label)
        if enabled:
            gates.append(Gate(code, int(code, 2), name, condition, short_label))
        else:
            print(f"Gate {code} ({name}) is disabled")
    if not gates:
        raise ValueError("At least one gate must have enabled: true")
    return gates


def condition_columns(node: Any, *, location: str = "condition") -> set[str]:
    if not isinstance(node, dict):
        raise ValueError(f"{location} must be a mapping")
    combinators = [key for key in COMBINATORS if key in node]
    if combinators:
        if len(combinators) != 1 or len(node) != 1:
            raise ValueError(f"{location} must contain exactly one all/any/not operator")
        operator = combinators[0]
        children = node[operator]
        if operator == "not":
            return condition_columns(children, location=f"{location}.not")
        if not isinstance(children, list) or not children:
            raise ValueError(f"{location}.{operator} must be a nonempty list")
        result: set[str] = set()
        for index, child in enumerate(children):
            result.update(condition_columns(child, location=f"{location}.{operator}[{index}]"))
        return result
    column = str(node.get("column", "")).strip()
    operator = str(node.get("op", node.get("operator", ""))).strip().lower()
    if not column or not operator:
        raise ValueError(f"{location} leaf needs column and op")
    columns = {column}
    if operator in {"abs_diff_lt", "abs_diff_le"}:
        other_column = str(node.get("other_column", "")).strip()
        if not other_column:
            raise ValueError(f"{location} {operator} needs other_column")
        columns.add(other_column)
    return columns


def derived_topology_setting(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("derived_strip_topology", {})
    if raw is False:
        return {"enabled": False, "source_suffix": "qsum_cal", "threshold": 0.0, "active_when": "gt"}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("derived_strip_topology must be a YAML mapping or false")
    active_when = str(raw.get("active_when", "gt")).strip().lower()
    if active_when not in {"gt", "ge"}:
        raise ValueError("derived_strip_topology.active_when must be gt or ge")
    return {
        "enabled": bool(raw.get("enabled", True)),
        "source_suffix": str(raw.get("source_suffix", "qsum_cal")).strip(),
        "threshold": float(raw.get("threshold", 0.0)),
        "active_when": active_when,
    }


def derived_column_names() -> list[str]:
    return [
        name
        for plane in range(1, 5)
        for name in (f"p{plane}_cluster_size", f"p{plane}_strip_topology")
    ]


def topology_source_columns(setting: dict[str, Any]) -> list[str]:
    if not setting["enabled"]:
        return []
    suffix = setting["source_suffix"]
    return [f"p{plane}_s{strip}_{suffix}" for plane in range(1, 5) for strip in range(1, 5)]


def read_events(files: list[Product], columns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for product in files:
        present = set(pq.read_schema(product.path).names)
        frame = pd.read_parquet(product.path, columns=[name for name in columns if name in present])
        for name in columns:
            if name not in frame:
                frame[name] = np.nan
        frame = frame[columns]
        frame["_source_basename"] = product.basename
        frames.append(frame)
        print(f"Loaded {len(frame):,} events from {product.basename}")
    return pd.concat(frames, ignore_index=True)


def add_derived_topology_columns(frame: pd.DataFrame, setting: dict[str, Any]) -> None:
    if not setting["enabled"]:
        return
    suffix = setting["source_suffix"]
    threshold = setting["threshold"]
    for plane in range(1, 5):
        columns = [f"p{plane}_s{strip}_{suffix}" for strip in range(1, 5)]
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(numeric)
        if setting["active_when"] == "ge":
            active = finite & (numeric >= threshold)
        else:
            active = finite & (numeric > threshold)
        frame[f"p{plane}_cluster_size"] = active.sum(axis=1).astype(np.int8)
        frame[f"p{plane}_strip_topology"] = [
            "".join("1" if state else "0" for state in row) for row in active
        ]
    print(
        "Derived p#_cluster_size and p#_strip_topology from "
        f"p#_s#_{suffix} ({setting['active_when']} {threshold:g})."
    )


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def evaluate_leaf(frame: pd.DataFrame, node: Mapping[str, Any], location: str) -> pd.Series:
    column = str(node.get("column", "")).strip()
    operator = str(node.get("op", node.get("operator", ""))).strip().lower()
    if column not in frame.columns:
        raise ValueError(f"{location} references unavailable column {column!r}")
    series = frame[column]
    if operator in {"abs_diff_lt", "abs_diff_le"}:
        other_column = str(node.get("other_column", "")).strip()
        if not other_column:
            raise ValueError(f"{location} {operator} needs other_column")
        if other_column not in frame.columns:
            raise ValueError(
                f"{location} references unavailable column {other_column!r}"
            )
        if "value" not in node:
            raise ValueError(f"{location} {operator} needs value")
        limit = float(node["value"])
        if not np.isfinite(limit) or limit < 0:
            raise ValueError(f"{location} {operator} value must be finite and nonnegative")
        left = pd.to_numeric(series, errors="coerce")
        right = pd.to_numeric(frame[other_column], errors="coerce")
        valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
        difference = (left - right).abs()
        comparison = difference.lt(limit) if operator == "abs_diff_lt" else difference.le(limit)
        return valid & comparison
    if operator in NUMERIC_OPERATORS:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric)
        if operator == "between":
            bounds = node.get("values", node.get("value"))
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"{location} between needs values: [lower, upper]")
            lower, upper = float(bounds[0]), float(bounds[1])
            if lower > upper:
                raise ValueError(f"{location} between lower bound exceeds upper bound")
            return valid & numeric.between(lower, upper, inclusive="both")
        if "value" not in node:
            raise ValueError(f"{location} {operator} needs value")
        value = float(node["value"])
        comparison = {
            "lt": numeric < value,
            "le": numeric <= value,
            "gt": numeric > value,
            "ge": numeric >= value,
        }[operator]
        return valid & comparison
    if operator in {"eq", "ne"}:
        if "value" not in node:
            raise ValueError(f"{location} {operator} needs value")
        value = node["value"]
        if _is_numeric_scalar(value):
            comparable = pd.to_numeric(series, errors="coerce")
            valid = comparable.notna() & np.isfinite(comparable)
            result = comparable.eq(float(value)) if operator == "eq" else comparable.ne(float(value))
        else:
            comparable = series.astype("string")
            valid = comparable.notna()
            result = comparable.eq(str(value)) if operator == "eq" else comparable.ne(str(value))
        return valid & result
    if operator in {"in", "not_in"}:
        values = node.get("values", node.get("value"))
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{location} {operator} needs a nonempty values list")
        if all(_is_numeric_scalar(value) for value in values):
            comparable = pd.to_numeric(series, errors="coerce")
            valid = comparable.notna() & np.isfinite(comparable)
            result = comparable.isin([float(value) for value in values])
        else:
            comparable = series.astype("string")
            valid = comparable.notna()
            result = comparable.isin([str(value) for value in values])
        return valid & (result if operator == "in" else ~result)
    if operator in {"is_finite", "finite"}:
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.notna() & np.isfinite(numeric)
    if operator in {"is_zero", "zero", "nonzero"}:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric)
        return valid & (numeric.ne(0) if operator == "nonzero" else numeric.eq(0))
    if operator in {"isna", "notna"}:
        return series.isna() if operator == "isna" else series.notna()
    if operator in {"matches", "regex"}:
        pattern = str(node.get("value", ""))
        return series.astype("string").str.fullmatch(pattern, na=False)
    raise ValueError(f"Unsupported operator {operator!r} at {location}")


def evaluate_condition(frame: pd.DataFrame, node: Any, *, location: str) -> pd.Series:
    if not isinstance(node, dict):
        raise ValueError(f"{location} must be a mapping")
    combinators = [key for key in COMBINATORS if key in node]
    if not combinators:
        return evaluate_leaf(frame, node, location).astype(bool)
    if len(combinators) != 1 or len(node) != 1:
        raise ValueError(f"{location} must contain exactly one all/any/not operator")
    operator = combinators[0]
    children = node[operator]
    if operator == "not":
        return ~evaluate_condition(frame, children, location=f"{location}.not")
    if not isinstance(children, list) or not children:
        raise ValueError(f"{location}.{operator} must be a nonempty list")
    result = pd.Series(operator == "all", index=frame.index, dtype=bool)
    for index, child in enumerate(children):
        child_mask = evaluate_condition(
            frame, child, location=f"{location}.{operator}[{index}]",
        )
        result = result & child_mask if operator == "all" else result | child_mask
    return result


def assign_gates(frame: pd.DataFrame, gates: list[Gate]) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    combined = np.zeros(len(frame), dtype=np.int64)
    for gate in gates:
        mask = evaluate_condition(frame, gate.condition, location=f"gate[{gate.code}]")
        masks[gate.code] = mask
        combined += np.where(mask.to_numpy(dtype=bool), gate.bit_value, 0).astype(np.int64)
        print(
            f"Gate {gate.code} ({gate.name}): {int(mask.sum()):,}/{len(frame):,} "
            f"events ({float(mask.mean()) * 100:.3f}%)"
        )
    frame["gate_code"] = pd.Series(
        [format(int(value), "b") if value else "0" for value in combined],
        index=frame.index,
        dtype="string",
    )
    return masks


def combined_name(code: str, gates: list[Gate]) -> str:
    value = int(code, 2) if code != "0" else 0
    names = [gate.name for gate in gates if value & gate.bit_value]
    return " + ".join(names) if names else "no configured gate"


def combined_short_label(code: str, gates: list[Gate]) -> str:
    value = int(code, 2) if code != "0" else 0
    labels = [gate.short_label or gate.name for gate in gates if value & gate.bit_value]
    return "+".join(labels) if labels else "None"


def write_gate_summary(
    frame: pd.DataFrame, gates: list[Gate], masks: dict[str, pd.Series], output: Path,
) -> pd.DataFrame:
    total = len(frame)
    rows: list[dict[str, Any]] = []
    for gate in gates:
        count = int(masks[gate.code].sum())
        rows.append({
            "kind": "individual",
            "gate_code": gate.code,
            "gate_name": gate.name,
            "gate_label": gate.short_label or gate.name,
            "events": count,
            "fraction": count / total if total else np.nan,
        })
    combined_counts = frame["gate_code"].value_counts().to_dict()
    for code in sorted(combined_counts, key=lambda value: int(value, 2) if value != "0" else 0):
        count = int(combined_counts[code])
        rows.append({
            "kind": "combined_exact",
            "gate_code": code,
            "gate_name": combined_name(code, gates),
            "gate_label": combined_short_label(code, gates),
            "events": count,
            "fraction": count / total if total else np.nan,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False)
    return summary


def time_series_setting(config: dict[str, Any], available: set[str]) -> dict[str, Any] | None:
    raw = config.get("time_series", {})
    if raw is False:
        return None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("time_series must be a YAML mapping or false")
    if not bool(raw.get("enabled", True)):
        return None
    column = str(raw.get("time_column", "datetime")).strip()
    if column not in available:
        raise ValueError(f"Time-series column absent from schema: {column}")
    try:
        window = pd.Timedelta(str(raw.get("accumulation_timespan", "1min")))
    except ValueError as exc:
        raise ValueError("Invalid time_series.accumulation_timespan") from exc
    if window <= pd.Timedelta(0):
        raise ValueError("time_series.accumulation_timespan must be positive")
    return {"column": column, "window": window}


def plot_rate_lines(
    summary: pd.DataFrame,
    lines: list[tuple[str, str]],
    output: Path,
    title: str,
) -> bool:
    plotted = [(column, label) for column, label in lines if summary[column].gt(0).any()]
    if not plotted:
        print(f"Warning: no populated rate series for {output.name}")
        return False
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    for column, label in plotted:
        axis.plot(summary["window_start"], summary[column], marker=".", markersize=3,
                  linewidth=1.1, label=label)
    axis.set(xlabel="Time", ylabel="Event rate [Hz]", title=title)
    axis.grid(True, alpha=0.25)
    legend_columns = min(4, max(1, (len(plotted) + 5) // 6))
    axis.legend(fontsize=7, ncols=legend_columns)
    fig.autofmt_xdate()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def write_gate_time_series(
    frame: pd.DataFrame,
    gates: list[Gate],
    masks: dict[str, pd.Series],
    setting: dict[str, Any],
    output_dir: Path,
    title: str,
) -> tuple[Path, int]:
    timestamps = pd.to_datetime(frame[setting["column"]], errors="coerce")
    valid = timestamps.notna()
    if not bool(valid.any()):
        raise ValueError(f"{setting['column']} contains no valid timestamps")
    events = pd.DataFrame({"second": timestamps.loc[valid].dt.floor("s")})
    individual_columns: list[tuple[str, str]] = []
    for gate in gates:
        column = f"individual_{gate.code}_events"
        events[column] = masks[gate.code].loc[valid].to_numpy(dtype=np.int8)
        individual_columns.append((column, gate.short_label or gate.name))
    combined_codes = sorted(
        frame.loc[valid, "gate_code"].dropna().astype(str).unique(),
        key=lambda code: int(code, 2) if code != "0" else 0,
    )
    combined_columns: list[tuple[str, str]] = []
    gate_codes = frame.loc[valid, "gate_code"].astype(str)
    for code in combined_codes:
        column = f"combined_{code}_events"
        events[column] = gate_codes.eq(code).to_numpy(dtype=np.int8)
        combined_columns.append((column, combined_short_label(code, gates)))
    events["total_events"] = 1
    per_second = events.groupby("second", sort=True).sum(numeric_only=True)
    per_second["window_start"] = per_second.index.floor(setting["window"])
    value_columns = ["total_events", *(name for name, _ in individual_columns),
                     *(name for name, _ in combined_columns)]
    summary = per_second.groupby("window_start", sort=True)[value_columns].sum()
    summary["observed_seconds"] = per_second.groupby("window_start").size()
    summary["window_end"] = summary.index + setting["window"]
    for column in value_columns:
        summary[column.removesuffix("_events") + "_hz"] = (
            summary[column] / summary["observed_seconds"]
        )
    summary = summary.reset_index()
    ordered = ["window_start", "window_end", "observed_seconds", *value_columns,
               *(column.removesuffix("_events") + "_hz" for column in value_columns)]
    summary = summary[ordered]
    csv_path = output_dir / "gate_rates.csv"
    summary.to_csv(csv_path, index=False)
    individual_hz = [
        (column.removesuffix("_events") + "_hz", label) for column, label in individual_columns
    ]
    combined_hz = [
        (column.removesuffix("_events") + "_hz", label) for column, label in combined_columns
    ]
    written = int(plot_rate_lines(
        summary, individual_hz, output_dir / "individual_gate_rates.png",
        f"{title}\nIndividual gate rates | accumulation={setting['window']} | denominator=observed seconds",
    ))
    written += int(plot_rate_lines(
        summary, combined_hz, output_dir / "combined_exact_gate_rates.png",
        f"{title}\nExact combined-code rates | accumulation={setting['window']} | denominator=observed seconds",
    ))
    return csv_path, written


def efficiency_time_series_setting(
    config: dict[str, Any], gates: list[Gate], available: set[str],
) -> dict[str, Any] | None:
    raw = config.get("efficiency_time_series", {})
    if raw is False:
        return None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("efficiency_time_series must be a YAML mapping or false")
    if not bool(raw.get("enabled", True)):
        return None
    time_column = str(raw.get("time_column", "datetime")).strip()
    topology_column = str(raw.get("topology_column", "tt_task3_list")).strip()
    missing = sorted({time_column, topology_column} - available)
    if missing:
        raise ValueError("Efficiency time-series columns absent from schema: " + ", ".join(missing))
    try:
        window = pd.Timedelta(str(raw.get("accumulation_timespan", "10min")))
    except ValueError as exc:
        raise ValueError("Invalid efficiency_time_series.accumulation_timespan") from exc
    if window <= pd.Timedelta(0):
        raise ValueError("efficiency_time_series.accumulation_timespan must be positive")

    raw_selections = raw.get("selections", "all")
    selections: list[dict[str, str]] | None
    if isinstance(raw_selections, str) and raw_selections.strip().lower() == "all":
        selections = None
    elif isinstance(raw_selections, list) and raw_selections:
        selections = []
        gate_codes = {gate.code for gate in gates}
        configured_bits = sum(gate.bit_value for gate in gates)
        seen: set[tuple[str, str]] = set()
        for index, item in enumerate(raw_selections, 1):
            if not isinstance(item, dict):
                raise ValueError(f"efficiency_time_series.selections item {index} must be a mapping")
            kind = str(item.get("kind", "individual")).strip().lower()
            kind = {"combined": "combined_exact", "exact": "combined_exact"}.get(kind, kind)
            if kind not in {"individual", "combined_exact"}:
                raise ValueError(f"Efficiency selection kind must be individual or combined_exact: {kind}")
            code = str(item.get("code", "")).strip()
            if kind == "individual" and code not in gate_codes:
                raise ValueError(f"Efficiency selection references unknown individual gate {code!r}")
            if kind == "combined_exact":
                if not re.fullmatch(r"[01]+", code):
                    raise ValueError(f"Efficiency combined code must be binary: {code!r}")
                if int(code, 2) & ~configured_bits:
                    raise ValueError(f"Efficiency combined code uses an unconfigured gate bit: {code}")
            key = (kind, code)
            if key in seen:
                raise ValueError(f"Duplicate efficiency selection: {kind} {code}")
            seen.add(key)
            selections.append({
                "kind": kind,
                "code": code,
                "label": str(item.get("label", "")).strip(),
            })
    else:
        raise ValueError("efficiency_time_series.selections must be all or a nonempty list")
    return {
        "time_column": time_column,
        "topology_column": topology_column,
        "window": window,
        "selections": selections,
        "include_combined_zero": bool(raw.get("include_combined_zero", False)),
    }


def efficiency_selections(
    frame: pd.DataFrame,
    gates: list[Gate],
    masks: dict[str, pd.Series],
    setting: dict[str, Any],
) -> list[tuple[str, str, str, pd.Series]]:
    specs = setting["selections"]
    if specs is None:
        specs = [
            {"kind": "individual", "code": gate.code, "label": ""}
            for gate in gates
        ]
        combined_codes = sorted(
            frame["gate_code"].dropna().astype(str).unique(),
            key=lambda code: int(code, 2) if code != "0" else 0,
        )
        specs.extend(
            {"kind": "combined_exact", "code": code, "label": ""}
            for code in combined_codes
            if code != "0" or setting["include_combined_zero"]
        )

    gate_names = {gate.code: gate.name for gate in gates}
    selections: list[tuple[str, str, str, pd.Series]] = []
    for spec in specs:
        kind, code = spec["kind"], spec["code"]
        if kind == "individual":
            default_label = f"{code}: {gate_names[code]}"
            mask = masks[code]
        else:
            default_label = f"{code}: {combined_name(code, gates)}"
            mask = frame["gate_code"].astype(str).eq(code)
        selections.append((kind, code, spec["label"] or default_label, mask.astype(bool)))
    return selections


def plot_efficiency_lines(
    summary: pd.DataFrame, output: Path, title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    plotted = False
    for plane in range(1, 5):
        values = pd.to_numeric(summary[f"plane_{plane}_efficiency"], errors="coerce")
        if bool(values.notna().any()):
            plotted = True
            axis.plot(summary["window_start"], values, marker=".", markersize=4,
                      linewidth=1.2, label=f"Plane {plane}")
    if not plotted:
        axis.text(0.5, 0.5, "No window has a nonzero efficiency denominator",
                  ha="center", va="center", transform=axis.transAxes)
    axis.set(xlabel="Time", ylabel="Efficiency", ylim=(-0.02, 1.02), title=title)
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(ncols=4)
    fig.autofmt_xdate()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_efficiency_time_series(
    frame: pd.DataFrame,
    gates: list[Gate],
    masks: dict[str, pd.Series],
    setting: dict[str, Any],
    output_dir: Path,
    title: str,
) -> tuple[Path, int]:
    time_column = setting["time_column"]
    topology_column = setting["topology_column"]
    window = setting["window"]
    timestamps = pd.to_datetime(frame[time_column], errors="coerce")
    topology_numeric = pd.to_numeric(frame[topology_column], errors="coerce")
    integer_topology = topology_numeric.where(topology_numeric.mod(1).eq(0)).astype("Int64")
    topologies = integer_topology.astype("string")
    valid_time = timestamps.notna()
    if not bool(valid_time.any()):
        raise ValueError(f"{time_column} contains no valid timestamps")
    all_windows = pd.DatetimeIndex(
        timestamps.loc[valid_time].dt.floor(window).drop_duplicates().sort_values(),
        name="window_start",
    )
    missing_topology = {1: "234", 2: "134", 3: "124", 4: "123"}
    counted_topologies = ("123", "124", "134", "234", "1234")
    rows: list[pd.DataFrame] = []
    plot_count = 0
    selections = efficiency_selections(frame, gates, masks, setting)
    for kind, code, label, gate_mask in selections:
        selected = valid_time & gate_mask
        selected_events = pd.DataFrame({
            "window_start": timestamps.loc[selected].dt.floor(window),
            "topology": topologies.loc[selected],
        })
        counts = selected_events.groupby(["window_start", "topology"]).size().unstack(fill_value=0)
        counts = counts.reindex(all_windows, fill_value=0)
        for topology in counted_topologies:
            if topology not in counts:
                counts[topology] = 0
        selection_summary = pd.DataFrame({
            "window_start": all_windows,
            "window_end": all_windows + window,
            "selection_kind": kind,
            "gate_code": code,
            "gate_name": label,
        })
        for topology in counted_topologies:
            selection_summary[f"topology_{topology}_count"] = (
                pd.to_numeric(counts[topology], errors="coerce").fillna(0).astype(np.int64).to_numpy()
            )
        detected = selection_summary["topology_1234_count"].to_numpy(dtype=float)
        for plane, missing_code in missing_topology.items():
            undetected = selection_summary[f"topology_{missing_code}_count"].to_numpy(dtype=float)
            total = detected + undetected
            selection_summary[f"plane_{plane}_undetected_count"] = undetected.astype(np.int64)
            selection_summary[f"plane_{plane}_total_count"] = total.astype(np.int64)
            selection_summary[f"plane_{plane}_efficiency"] = np.divide(
                detected, total, out=np.full(len(total), np.nan), where=total > 0,
            )
        plot_path = output_dir / f"efficiency_{kind}_{code}.png"
        plot_efficiency_lines(
            selection_summary,
            plot_path,
            f"{title} | {kind} gate {label} | accumulation={window} | "
            "efficiency = 1 - N(missing plane) / [N(1234) + N(missing plane)]",
        )
        plot_count += 1
        rows.append(selection_summary)
    summary = pd.concat(rows, ignore_index=True)
    csv_path = output_dir / "gate_plane_efficiencies.csv"
    summary.to_csv(csv_path, index=False)
    print(
        f"Efficiency time series: {len(selections)} gate selection(s), "
        f"{len(all_windows)} window(s), accumulation={window}"
    )
    return csv_path, plot_count


def enabled_gate_comparison_setting(
    config: dict[str, Any], available: set[str],
) -> dict[str, Any] | None:
    raw = config.get("enabled_gate_comparison", {})
    if raw is False:
        return None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("enabled_gate_comparison must be a YAML mapping or false")
    if not bool(raw.get("enabled", True)):
        return None
    time_column = str(raw.get("time_column", "datetime")).strip()
    topology_column = str(raw.get("topology_column", "tt_task3_list")).strip()
    missing = sorted({time_column, topology_column} - available)
    if missing:
        raise ValueError(
            "Enabled-gate comparison columns absent from schema: " + ", ".join(missing)
        )
    try:
        window = pd.Timedelta(str(raw.get("accumulation_timespan", "10min")))
    except ValueError as exc:
        raise ValueError("Invalid enabled_gate_comparison.accumulation_timespan") from exc
    if window <= pd.Timedelta(0):
        raise ValueError("enabled_gate_comparison.accumulation_timespan must be positive")
    return {
        "time_column": time_column,
        "topology_column": topology_column,
        "window": window,
    }


def write_enabled_gate_comparison(
    frame: pd.DataFrame,
    gates: list[Gate],
    masks: dict[str, pd.Series],
    setting: dict[str, Any],
    output_dir: Path,
    title: str,
) -> tuple[Path, Path, Path, int]:
    """Overlay efficiency and rate diagnostics for every enabled individual gate."""
    time_column = setting["time_column"]
    topology_column = setting["topology_column"]
    window = setting["window"]
    timestamps = pd.to_datetime(frame[time_column], errors="coerce")
    topology_numeric = pd.to_numeric(frame[topology_column], errors="coerce")
    topologies = topology_numeric.where(topology_numeric.mod(1).eq(0)).astype("Int64").astype("string")
    valid_time = timestamps.notna()
    if not bool(valid_time.any()):
        raise ValueError(f"{time_column} contains no valid timestamps")
    seconds = timestamps.loc[valid_time].dt.floor("s")
    all_windows = pd.DatetimeIndex(
        timestamps.loc[valid_time].dt.floor(window).drop_duplicates().sort_values(),
        name="window_start",
    )
    exposure = pd.DataFrame({
        "window_start": seconds.dt.floor(window),
        "second": seconds,
    }).groupby("window_start")["second"].nunique().reindex(all_windows, fill_value=0)

    counted_topologies = ("123", "124", "134", "234", "1234")
    missing_topology = {1: "234", 2: "134", 3: "124", 4: "123"}
    rows: list[pd.DataFrame] = []
    for gate in gates:
        selected = valid_time & masks[gate.code]
        selected_windows = timestamps.loc[selected].dt.floor(window)
        gate_event_counts = selected_windows.value_counts(sort=False).reindex(
            all_windows, fill_value=0,
        )
        selected_events = pd.DataFrame({
            "window_start": selected_windows,
            "topology": topologies.loc[selected],
        })
        counts = selected_events.groupby(["window_start", "topology"]).size().unstack(fill_value=0)
        counts = counts.reindex(all_windows, fill_value=0)
        for topology in counted_topologies:
            if topology not in counts:
                counts[topology] = 0
        summary = pd.DataFrame({
            "window_start": all_windows,
            "window_end": all_windows + window,
            "gate_code": gate.code,
            "gate_name": gate.name,
            "gate_label": gate.short_label or gate.name,
            "observed_seconds": exposure.to_numpy(dtype=np.int64),
            "gate_event_count": gate_event_counts.to_numpy(dtype=np.int64),
        })
        for topology in counted_topologies:
            summary[f"topology_{topology}_count"] = (
                pd.to_numeric(counts[topology], errors="coerce")
                .fillna(0).astype(np.int64).to_numpy()
            )
        detected = summary["topology_1234_count"].to_numpy(dtype=float)
        for plane, missing_code in missing_topology.items():
            undetected = summary[f"topology_{missing_code}_count"].to_numpy(dtype=float)
            denominator = detected + undetected
            summary[f"plane_{plane}_efficiency"] = np.divide(
                detected, denominator,
                out=np.full(len(denominator), np.nan), where=denominator > 0,
            )
        efficiency_columns = [f"plane_{plane}_efficiency" for plane in range(1, 5)]
        summary["efficiency_product"] = summary[efficiency_columns].prod(axis=1, min_count=4)
        observed_seconds = summary["observed_seconds"].to_numpy(dtype=float)
        summary["total_gate_rate_hz"] = np.divide(
            summary["gate_event_count"].to_numpy(dtype=float), observed_seconds,
            out=np.full(len(observed_seconds), np.nan), where=observed_seconds > 0,
        )
        summary["topology_1234_rate_hz"] = np.divide(
            detected, observed_seconds,
            out=np.full(len(observed_seconds), np.nan), where=observed_seconds > 0,
        )
        efficiency_product = summary["efficiency_product"].to_numpy(dtype=float)
        rate_1234 = summary["topology_1234_rate_hz"].to_numpy(dtype=float)
        summary["corrected_1234_rate_hz"] = np.divide(
            rate_1234, efficiency_product,
            out=np.full(len(rate_1234), np.nan),
            where=np.isfinite(efficiency_product) & (efficiency_product > 0),
        )
        rows.append(summary)

    comparison = pd.concat(rows, ignore_index=True)
    csv_path = output_dir / "enabled_gate_comparison.csv"
    comparison.to_csv(csv_path, index=False)
    colors = plt.get_cmap("tab10").colors

    efficiency_path = output_dir / "enabled_gate_plane_efficiencies.png"
    efficiency_figure, efficiency_axes = plt.subplots(
        4, 1, figsize=(16, 14), sharex=True, constrained_layout=True,
    )
    for gate_index, gate in enumerate(gates):
        gate_rows = comparison.loc[comparison["gate_code"].eq(gate.code)]
        label = f"{gate.short_label or gate.name} [{gate.code}]"
        for plane, axis in enumerate(efficiency_axes, 1):
            axis.plot(
                gate_rows["window_start"], gate_rows[f"plane_{plane}_efficiency"],
                color=colors[gate_index % len(colors)], marker=".", markersize=3,
                linewidth=1.1, label=label,
            )
    for plane, axis in enumerate(efficiency_axes, 1):
        axis.set(ylabel=f"Plane {plane}\nefficiency", ylim=(-0.02, 1.02))
        axis.grid(True, alpha=0.25)
    efficiency_axes[0].legend(ncols=min(4, len(gates)), fontsize=8)
    efficiency_axes[-1].set_xlabel("Time")
    efficiency_figure.suptitle(
        f"{title}\nPlane efficiency by enabled gate | accumulation={window}",
        fontsize=14,
    )
    efficiency_figure.autofmt_xdate()
    efficiency_figure.savefig(efficiency_path, dpi=160)
    plt.close(efficiency_figure)

    metrics_path = output_dir / "enabled_gate_rate_efficiency_product.png"
    metrics_figure, metrics_axes = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True, constrained_layout=True,
    )
    metric_specs = (
        ("total_gate_rate_hz", "Total gate rate [Hz]", "Total selected-event rate per enabled gate"),
        ("efficiency_product", "Efficiency product", "Product of the four plane efficiencies"),
        (
            "corrected_1234_rate_hz", "Corrected 1234 rate [Hz]",
            "1234 rate × (1 / efficiency product)",
        ),
    )
    for gate_index, gate in enumerate(gates):
        gate_rows = comparison.loc[comparison["gate_code"].eq(gate.code)]
        label = f"{gate.short_label or gate.name} [{gate.code}]"
        for axis, (column, _, _) in zip(metrics_axes, metric_specs, strict=True):
            axis.plot(
                gate_rows["window_start"], gate_rows[column],
                color=colors[gate_index % len(colors)], marker=".", markersize=3,
                linewidth=1.1, label=label,
            )
    for axis, (_, ylabel, axis_title) in zip(metrics_axes, metric_specs, strict=True):
        axis.set(ylabel=ylabel, title=axis_title)
        axis.grid(True, alpha=0.25)
    metrics_axes[0].legend(ncols=min(4, len(gates)), fontsize=8)
    metrics_axes[-1].set_xlabel("Time")
    metrics_figure.suptitle(
        f"{title}\nEnabled-gate comparison | accumulation={window} | "
        "rate denominator=observed timestamp-seconds",
        fontsize=14,
    )
    metrics_figure.autofmt_xdate()
    metrics_figure.savefig(metrics_path, dpi=160)
    plt.close(metrics_figure)
    print(
        f"Enabled-gate comparison: {len(gates)} gate(s), {len(all_windows)} window(s), "
        f"accumulation={window}"
    )
    return efficiency_path, metrics_path, csv_path, 2



def plane_combination_setting(
    config: dict[str, Any], gates: list[Gate], available: set[str],
) -> dict[str, Any] | None:
    raw = config.get("plane_combinations", {})
    if raw is False:
        return None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("plane_combinations must be a YAML mapping or false")
    if not bool(raw.get("enabled", True)):
        return None
    time_column = str(raw.get("time_column", "datetime")).strip()
    topology_column = str(raw.get("topology_column", "tt_task3_list")).strip()
    missing = sorted({time_column, topology_column} - available)
    if missing:
        raise ValueError("Plane-combination columns absent from schema: " + ", ".join(missing))
    try:
        window = pd.Timedelta(str(raw.get("accumulation_timespan", "10min")))
    except ValueError as exc:
        raise ValueError("Invalid plane_combinations.accumulation_timespan") from exc
    if window <= pd.Timedelta(0):
        raise ValueError("plane_combinations.accumulation_timespan must be positive")
    kind = str(raw.get("selection_kind", "individual")).strip().lower()
    kind = {"combined": "combined_exact", "exact": "combined_exact"}.get(kind, kind)
    if kind not in {"individual", "combined_exact"}:
        raise ValueError("plane_combinations.selection_kind must be individual or combined_exact")
    code = str(raw.get("gate_code", gates[-1].code)).strip()
    gate_codes = {gate.code for gate in gates}
    if kind == "individual" and code not in gate_codes:
        raise ValueError(f"Plane combination references unknown gate {code!r}")
    if kind == "combined_exact":
        configured_bits = sum(gate.bit_value for gate in gates)
        if not re.fullmatch(r"[01]+", code) or int(code, 2) & ~configured_bits:
            raise ValueError(f"Plane combination uses invalid combined code {code!r}")
    return {
        "time_column": time_column,
        "topology_column": topology_column,
        "window": window,
        "kind": kind,
        "code": code,
    }


def write_plane_combination_timeseries(
    frame: pd.DataFrame,
    gates: list[Gate],
    masks: dict[str, pd.Series],
    setting: dict[str, Any],
    output_dir: Path,
    title: str,
) -> tuple[Path, Path, int]:
    kind = setting["kind"]
    code = setting["code"]
    window = setting["window"]
    time_column = setting["time_column"]
    topology_column = setting["topology_column"]
    if kind == "individual":
        gate = next(gate for gate in gates if gate.code == code)
        selection_name = gate.name
        gate_mask = masks[code]
    else:
        selection_name = combined_name(code, gates)
        gate_mask = frame["gate_code"].astype(str).eq(code)

    timestamps = pd.to_datetime(frame[time_column], errors="coerce")
    topology_numeric = pd.to_numeric(frame[topology_column], errors="coerce")
    topologies = topology_numeric.where(topology_numeric.mod(1).eq(0)).astype("Int64").astype("string")
    valid_time = timestamps.notna()
    if not bool(valid_time.any()):
        raise ValueError(f"{time_column} contains no valid timestamps")
    seconds = timestamps.loc[valid_time].dt.floor("s")
    all_windows = pd.DatetimeIndex(
        timestamps.loc[valid_time].dt.floor(window).drop_duplicates().sort_values(),
        name="window_start",
    )
    exposure = pd.DataFrame({
        "window_start": seconds.dt.floor(window),
        "second": seconds,
    }).groupby("window_start")["second"].nunique().reindex(all_windows, fill_value=0)

    selected = valid_time & gate_mask
    selected_events = pd.DataFrame({
        "window_start": timestamps.loc[selected].dt.floor(window),
        "topology": topologies.loc[selected],
    })
    counts = selected_events.groupby(["window_start", "topology"]).size().unstack(fill_value=0)
    counts = counts.reindex(all_windows, fill_value=0)
    counted_topologies = ("123", "124", "134", "234", "1234")
    for topology in counted_topologies:
        if topology not in counts:
            counts[topology] = 0

    summary = pd.DataFrame({
        "window_start": all_windows,
        "window_end": all_windows + window,
        "selection_kind": kind,
        "gate_code": code,
        "gate_name": selection_name,
        "observed_seconds": exposure.to_numpy(dtype=np.int64),
    })
    for topology in counted_topologies:
        summary[f"topology_{topology}_count"] = (
            pd.to_numeric(counts[topology], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        )
    detected = summary["topology_1234_count"].to_numpy(dtype=float)
    missing_topology = {1: "234", 2: "134", 3: "124", 4: "123"}
    for plane, missing_code in missing_topology.items():
        undetected = summary[f"topology_{missing_code}_count"].to_numpy(dtype=float)
        denominator = detected + undetected
        summary[f"plane_{plane}_efficiency"] = np.divide(
            detected, denominator,
            out=np.full(len(denominator), np.nan), where=denominator > 0,
        )
    efficiencies = summary[[f"plane_{plane}_efficiency" for plane in range(1, 5)]]
    summary["efficiency_product"] = efficiencies.prod(axis=1, min_count=4)
    observed_seconds = summary["observed_seconds"].to_numpy(dtype=float)
    summary["topology_1234_rate_hz"] = np.divide(
        detected, observed_seconds,
        out=np.full(len(observed_seconds), np.nan), where=observed_seconds > 0,
    )
    efficiency_product = summary["efficiency_product"].to_numpy(dtype=float)
    raw_rate = summary["topology_1234_rate_hz"].to_numpy(dtype=float)
    summary["efficiency_corrected_rate_hz"] = np.divide(
        raw_rate, efficiency_product,
        out=np.full(len(raw_rate), np.nan),
        where=np.isfinite(efficiency_product) & (efficiency_product > 0),
    )

    csv_path = output_dir / f"plane_combinations_{kind}_{code}.csv"
    summary.to_csv(csv_path, index=False)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, constrained_layout=True)
    axes[0].plot(
        summary["window_start"], summary["topology_1234_rate_hz"],
        color="tab:blue", marker=".", markersize=4, linewidth=1.2,
    )
    axes[0].set_ylabel("1234 rate [Hz]")
    axes[0].set_title("Observed four-plane (1234) event rate")
    for plane in range(1, 5):
        axes[1].plot(
            summary["window_start"], summary[f"plane_{plane}_efficiency"],
            marker=".", markersize=3, linewidth=1.0, label=f"Plane {plane}",
        )
    axes[1].plot(
        summary["window_start"], summary["efficiency_product"],
        color="black", linestyle="--", linewidth=1.8, label="efficiency product",
    )
    axes[1].set(ylabel="Efficiency", ylim=(-0.02, 1.02),
                title="Plane efficiencies and four-plane efficiency product")
    axes[1].legend(ncols=5)
    axes[2].plot(
        summary["window_start"], summary["efficiency_corrected_rate_hz"],
        color="tab:red", marker=".", markersize=4, linewidth=1.2,
    )
    axes[2].set(
        xlabel="Time", ylabel="Corrected rate [Hz]",
        title="1234 rate / efficiency product",
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle(
        f"{title}" + chr(10) + f"{kind} gate {code}: {selection_name} | accumulation={window} | "
        "rate denominator=observed timestamp-seconds",
        fontsize=14,
    )
    fig.autofmt_xdate()
    plot_path = output_dir / f"plane_combinations_{kind}_{code}.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(
        f"Plane combinations: {kind} gate {code}, {len(summary)} window(s), "
        f"accumulation={window}"
    )
    return plot_path, csv_path, 1


def angular_histogram_setting(
    config: dict[str, Any], available: set[str],
) -> dict[str, Any] | None:
    raw = config.get("angular_histograms", config.get("theta_histograms", {}))
    if raw is False:
        return None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("angular_histograms must be a YAML mapping or false")
    if not bool(raw.get("enabled", True)):
        return None
    raw_variables = raw.get("variables")
    if raw_variables is None:
        raw_variables = [{
            "column": str(raw.get("column", "event_theta")),
            "label": "Theta",
            "slug": "theta",
            "bins": raw.get("bins", 90),
            "range": raw.get("range", [0, 90]),
        }]
    if not isinstance(raw_variables, list) or not raw_variables:
        raise ValueError("angular_histograms.variables must be a nonempty list")
    variables: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for index, item in enumerate(raw_variables, 1):
        if not isinstance(item, dict):
            raise ValueError(f"angular_histograms.variables item {index} must be a mapping")
        column = str(item.get("column", "")).strip()
        label = str(item.get("label", column)).strip()
        slug = str(item.get("slug", label)).strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
        if not column or not label or not slug:
            raise ValueError(f"angular_histograms.variables item {index} needs column and label")
        if column not in available:
            raise ValueError(f"Angular histogram column absent from schema: {column}")
        if slug in seen_slugs:
            raise ValueError(f"Duplicate angular histogram slug: {slug}")
        bins = int(item.get("bins", raw.get("bins", 90)))
        if bins < 1:
            raise ValueError(f"angular_histograms variable {slug} bins must be positive")
        value_range = item.get("range", raw.get("range", [0, 90]))
        if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
            raise ValueError(f"angular_histograms variable {slug} range needs [min, max]")
        lower, upper = float(value_range[0]), float(value_range[1])
        if lower >= upper:
            raise ValueError(f"angular_histograms variable {slug} range must be increasing")
        variables.append({
            "column": column,
            "label": label,
            "slug": slug,
            "bins": bins,
            "range": (lower, upper),
        })
        seen_slugs.add(slug)
    z_columns = [f"z_p{plane}" for plane in range(1, 5)]
    if not set(z_columns).issubset(available):
        z_columns = []
    return {
        "variables": variables,
        "degrees": bool(raw.get("convert_radians_to_degrees", True)),
        "include_combined_zero": bool(raw.get("include_combined_zero", True)),
        "z_columns": z_columns,
        "z_unit": str(raw.get("plane_z_unit", "mm")).strip(),
    }


def charge_calibration_comparison_setting(
    config: dict[str, Any], available: set[str],
) -> dict[str, Any] | None:
    """Resolve the non-commuting strip-sum/calibration comparison."""
    raw = config.get("charge_calibration_comparison", {})
    if raw is False:
        return None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("charge_calibration_comparison must be a YAML mapping or false")
    if not bool(raw.get("enabled", True)):
        return None
    suffix = str(raw.get("source_suffix", "qsum_cal")).strip()
    columns = [
        f"p{plane}_s{strip}_{suffix}"
        for plane in range(1, 5)
        for strip in range(1, 5)
    ]
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(
            "Charge-calibration comparison columns absent from schema: "
            + ", ".join(missing)
        )
    raw_bins = raw.get("bins", 100)
    if isinstance(raw_bins, str) and raw_bins.strip().lower() == "auto":
        bins: int | str = "auto"
    else:
        bins = int(raw_bins)
        if bins < 1:
            raise ValueError(
                "charge_calibration_comparison.bins must be a positive integer or auto"
            )
    minimum_active = int(raw.get("min_active_strips", 2))
    if not 1 <= minimum_active <= 4:
        raise ValueError(
            "charge_calibration_comparison.min_active_strips must be in 1..4"
        )
    return {
        "suffix": suffix,
        "columns": columns,
        "bins": bins,
        "minimum_active": minimum_active,
    }


def write_per_strip_charge_histograms(
    frame: pd.DataFrame,
    setting: dict[str, Any],
    spline: CubicSpline,
    width_domain: tuple[float, float],
    output_dir: Path,
    title: str,
) -> tuple[Path, Path]:
    """Write matching 4x4 strip-ToT and calibrated-charge histogram grids."""
    source_labels = (
        frame["_source_basename"].astype(str).to_numpy()
        if "_source_basename" in frame.columns
        else np.full(len(frame), "selected data", dtype=object)
    )
    source_order = list(dict.fromkeys(source_labels.tolist()))
    source_colors = plt.get_cmap("tab10").colors
    suffix = setting["suffix"]
    specifications = [
        (
            "00_per_strip_uncalibrated_tot_histograms.png",
            "Uncalibrated ToT input within calibration domain", "ToT (ns)", False,
        ),
        (
            "01_per_strip_calibrated_charge_histograms.png",
            "ToT-to-charge calibrated", "Charge (fC)", True,
        ),
    ]
    written: list[Path] = []
    for filename, diagnostic_label, x_label, apply_spline in specifications:
        figure, axes = plt.subplots(
            4, 4, figsize=(16, 12), constrained_layout=True,
        )
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for plane in range(1, 5):
            for strip in range(1, 5):
                axis = axes[plane - 1, strip - 1]
                column = f"p{plane}_s{strip}_{suffix}"
                widths = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                positive = np.isfinite(widths) & (widths > 0)
                in_domain = (widths >= width_domain[0]) & (widths <= width_domain[1])
                out_of_domain_count = int((positive & ~in_domain).sum())
                valid = positive & in_domain
                parts: list[np.ndarray] = []
                part_labels: list[str] = []
                part_colors: list[Any] = []
                for source_index, source_name in enumerate(source_order):
                    source_mask = valid & (source_labels == source_name)
                    if not np.any(source_mask):
                        continue
                    values = widths[source_mask]
                    parts.append(spline(values) if apply_spline else values)
                    part_labels.append(source_name)
                    part_colors.append(source_colors[source_index % len(source_colors)])
                if parts:
                    edges = np.histogram_bin_edges(
                        np.concatenate(parts), bins=setting["bins"],
                    )
                    for values, part_label, part_color in zip(
                        parts, part_labels, part_colors, strict=True,
                    ):
                        axis.hist(
                            values, bins=edges, histtype="step", linewidth=1.25,
                            color=part_color, label=part_label,
                        )
                    if not legend_handles:
                        legend_handles, legend_labels = axis.get_legend_handles_labels()
                else:
                    axis.text(
                        0.5, 0.5, "No positive values", ha="center", va="center",
                        transform=axis.transAxes,
                    )
                axis.set_title(
                    f"Plane {plane}, strip {strip} "
                    f"(N={sum(len(part) for part in parts):,}, out={out_of_domain_count:,})",
                    fontsize=10,
                )
                axis.grid(True, alpha=0.22)
        figure.supxlabel(x_label)
        figure.supylabel("Count")
        figure.suptitle(
            f"{title.partition("|")[0].strip()} | {diagnostic_label} per strip\n"
            f"source: p#_s#_{suffix}",
            fontsize=15,
        )
        if legend_handles:
            figure.legend(
                legend_handles, legend_labels, loc="upper right",
                fontsize=8, title="Source file",
            )
        destination = output_dir / filename
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        written.append(destination)
        print(f"Per-strip charge diagnostic: {destination}")
    return written[0], written[1]


def write_charge_calibration_comparison(
    frame: pd.DataFrame,
    setting: dict[str, Any],
    output_dir: Path,
    title: str,
) -> tuple[Path, int]:
    """Compare sum(calibrate(strip ToT)) with calibrate(sum(strip ToT))."""
    calibration = pd.read_csv(TOT_TO_CHARGE_CALIBRATION)
    required = {"Width", "Fast_Charge"}
    if not required.issubset(calibration.columns):
        raise ValueError(
            f"Expected {required} in charge calibration {TOT_TO_CHARGE_CALIBRATION}"
        )
    calibration = calibration.loc[:, ["Width", "Fast_Charge"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().sort_values("Width")
    widths_table = calibration["Width"].to_numpy(dtype=float)
    charges_table = calibration["Fast_Charge"].to_numpy(dtype=float)
    if len(widths_table) < 2 or np.any(np.diff(widths_table) <= 0):
        raise ValueError("Charge-calibration widths must be strictly increasing")
    spline = CubicSpline(widths_table, charges_table, bc_type="natural")
    write_per_strip_charge_histograms(
        frame, setting, spline, (float(widths_table[0]), float(widths_table[-1])),
        output_dir, title,
    )

    calibrate_then_sum: list[np.ndarray] = []
    sum_then_calibrate: list[np.ndarray] = []
    source_by_observation: list[np.ndarray] = []
    plane_by_observation: list[np.ndarray] = []
    source_labels = (
        frame["_source_basename"].astype(str).to_numpy()
        if "_source_basename" in frame.columns
        else np.full(len(frame), "selected data", dtype=object)
    )
    suffix = setting["suffix"]
    rejected_outside_domain = 0
    for plane in range(1, 5):
        columns = [f"p{plane}_s{strip}_{suffix}" for strip in range(1, 5)]
        widths = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        active = np.isfinite(widths) & (widths > 0)
        active_count = active.sum(axis=1)
        positive_widths = np.where(active, widths, 0.0)
        width_sum = positive_widths.sum(axis=1)
        in_domain = (
            np.all(~active | (widths <= widths_table[-1]), axis=1)
            & (width_sum >= widths_table[0])
            & (width_sum <= widths_table[-1])
        )
        candidate = active_count >= setting["minimum_active"]
        valid = candidate & in_domain
        rejected_outside_domain += int((candidate & ~in_domain).sum())
        if not np.any(valid):
            continue
        strip_charges = np.where(active[valid], spline(widths[valid]), 0.0)
        calibrate_then_sum.append(strip_charges.sum(axis=1))
        sum_then_calibrate.append(spline(width_sum[valid]))
        source_by_observation.append(source_labels[valid])
        plane_by_observation.append(
            np.full(int(valid.sum()), plane, dtype=np.int8)
        )

    plot_path = output_dir / "strip_charge_calibration_order_comparison.png"
    if not calibrate_then_sum:
        raise ValueError(
            "No valid multi-strip plane events remained for the charge-calibration "
            "order comparison"
        )
    first = np.concatenate(calibrate_then_sum)
    second = np.concatenate(sum_then_calibrate)
    sources = np.concatenate(source_by_observation)
    planes = np.concatenate(plane_by_observation)
    combined = np.concatenate((first, second))
    edges = np.histogram_bin_edges(combined, bins=setting["bins"])
    difference = first - second

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), sharex=True, sharey=True,
        constrained_layout=True,
    )
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for plane in range(1, 5):
        axis = axes.flat[plane - 1]
        plane_mask = planes == plane
        plane_first = first[plane_mask]
        plane_second = second[plane_mask]
        axis.hist(
            plane_first, bins=edges, histtype="stepfilled", alpha=0.30,
            linewidth=1.6, color="tab:blue",
            label="sum of calibrated strip charges",
        )
        axis.hist(
            plane_second, bins=edges, histtype="step", linewidth=2.0,
            color="tab:orange", label="calibration of summed strip ToT",
        )
        axis.set_title(f"Plane {plane}")
        axis.grid(True, alpha=0.25)
        if len(plane_first):
            plane_difference = plane_first - plane_second
            annotation = (
                f"N = {len(plane_first):,}\n"
                f"median difference = {np.median(plane_difference):+.1f} fC"
            )
        else:
            annotation = "No valid plane events"
        axis.text(
            0.98, 0.97, annotation,
            transform=axis.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )
        if not legend_handles:
            legend_handles, legend_labels = axis.get_legend_handles_labels()
    fig.supxlabel("Plane charge (fC)")
    fig.supylabel("Plane-event count")
    fig.suptitle(
        f"{title.partition("|")[0].strip()} | ToT-to-charge calibration order by plane\n"
        f"(at least {setting['minimum_active']} active strips)",
        fontsize=15,
    )
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels, loc="upper right",
            bbox_to_anchor=(0.99, 0.96), fontsize=9,
        )
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    scatter_path = output_dir / "strip_charge_calibration_order_scatter.png"
    scatter_fig, scatter_axes = plt.subplots(
        2, 2, figsize=(11, 10), sharex=True, sharey=True,
        constrained_layout=True,
    )
    source_order = list(dict.fromkeys(sources.tolist()))
    source_colors = plt.get_cmap("tab10").colors
    lower = float(combined.min())
    upper = float(combined.max())
    for plane in range(1, 5):
        scatter_axis = scatter_axes.flat[plane - 1]
        plane_mask = planes == plane
        for source_index, source_name in enumerate(source_order):
            selection = plane_mask & (sources == source_name)
            if not np.any(selection):
                continue
            scatter_axis.scatter(
                second[selection], first[selection], s=7, alpha=0.22,
                color=source_colors[source_index % len(source_colors)],
                edgecolors="none", rasterized=True, label=source_name,
            )
        scatter_axis.plot(
            [lower, upper], [lower, upper], color="black", linestyle="--",
            linewidth=1.2, label="equal charge",
        )
        plane_difference = difference[plane_mask]
        scatter_axis.set(
            xlim=(lower, upper), ylim=(lower, upper),
            title=f"Plane {plane} (N={int(plane_mask.sum()):,})",
        )
        scatter_axis.set_aspect("equal", adjustable="box")
        scatter_axis.grid(True, alpha=0.25)
        scatter_axis.text(
            0.03, 0.97,
            f"median y - x = {np.median(plane_difference):+.1f} fC",
            transform=scatter_axis.transAxes, ha="left", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )
    for scatter_axis in scatter_axes[1, :]:
        scatter_axis.set_xlabel("Calibration of summed strip ToT (fC)")
    for scatter_axis in scatter_axes[:, 0]:
        scatter_axis.set_ylabel("Sum of calibrated strip charges (fC)")
    scatter_axes[1, 1].legend(
        loc="lower right", fontsize=8, title="Source file / reference",
    )
    scatter_fig.suptitle(
        f"{title.partition("|")[0].strip()} | Calibration-order scatter by plane\n"
        f"(at least {setting['minimum_active']} active strips)",
        fontsize=15,
    )
    scatter_fig.savefig(scatter_path, dpi=160)
    plt.close(scatter_fig)
    print(f"Charge-calibration scatter: {scatter_path}")
    print(
        f"Charge-calibration comparison: {len(first):,} plane events; "
        f"excluded {rejected_outside_domain:,} outside the "
        f"[{widths_table[0]:g}, {widths_table[-1]:g}] ns calibration domain."
    )
    return plot_path, len(first)


def write_angular_histogram_comparison(
    frame: pd.DataFrame,
    gates: list[Gate],
    masks: dict[str, pd.Series],
    setting: dict[str, Any],
    variable: dict[str, Any],
    output_dir: Path,
    title: str,
) -> tuple[Path, int]:
    angle_column = variable["column"]
    angle_label = variable["label"]
    angle_slug = variable["slug"]
    angle = pd.to_numeric(frame[angle_column], errors="coerce").to_numpy(dtype=float)
    if setting["degrees"]:
        angle = np.degrees(angle)
        unit = "degrees"
    else:
        unit = "radians"
    finite = np.isfinite(angle)
    selections: list[tuple[str, str, str, np.ndarray]] = []
    for gate in gates:
        selections.append((
            "individual", gate.code, gate.name, masks[gate.code].to_numpy(dtype=bool),
        ))
    combined_codes = sorted(
        frame["gate_code"].dropna().astype(str).unique(),
        key=lambda code: int(code, 2) if code != "0" else 0,
    )
    for code in combined_codes:
        if code == "0" and not setting["include_combined_zero"]:
            continue
        selections.append((
            "combined_exact", code, combined_name(code, gates),
            frame["gate_code"].astype(str).eq(code).to_numpy(dtype=bool),
        ))

    edges = np.linspace(variable["range"][0], variable["range"][1], variable["bins"] + 1)
    bin_widths = np.diff(edges)
    histogram_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(
        2, 1, figsize=(15, 11), constrained_layout=True, sharex=True,
    )
    axes_by_kind = {"individual": axes[0], "combined_exact": axes[1]}
    plotted_by_kind = {"individual": 0, "combined_exact": 0}

    for kind, code, name, gate_mask in selections:
        values = angle[finite & gate_mask]
        counts, _ = np.histogram(values, bins=edges)
        in_range_events = int(counts.sum())
        density = (
            counts.astype(float) / (in_range_events * bin_widths)
            if in_range_events > 0
            else np.zeros_like(bin_widths, dtype=float)
        )
        for index, count in enumerate(counts):
            histogram_rows.append({
                "kind": kind,
                "gate_code": code,
                "gate_name": name,
                "bin_left": edges[index],
                "bin_right": edges[index + 1],
                "events": int(count),
                "density": float(density[index]),
                "finite_events": int(values.size),
                "in_range_events": in_range_events,
            })
        if in_range_events == 0:
            print(f"Warning: no in-range {angle_label} values for {kind} gate {code}")
            continue
        legend_name = (
            next(gate.short_label or gate.name for gate in gates if gate.code == code)
            if kind == "individual" else combined_short_label(code, gates)
        )
        axes_by_kind[kind].stairs(
            density,
            edges,
            linewidth=1.5,
            label=f"{legend_name} (n={in_range_events:,})",
        )
        plotted_by_kind[kind] += 1

    density_unit = f"1/{unit}"
    axes[0].set(
        ylabel=f"Probability density [{density_unit}]",
        title="Individual gates (events may appear in more than one curve)",
    )
    axes[1].set(
        xlabel=f"{angle_column} [{unit}]",
        ylabel=f"Probability density [{density_unit}]",
        title="Exact combined gate codes (mutually exclusive)",
    )
    for kind, axis in axes_by_kind.items():
        axis.grid(True, alpha=0.25)
        if plotted_by_kind[kind]:
            axis.legend(fontsize=7, ncols=2)
        else:
            axis.text(
                0.5, 0.5, "No populated gates", ha="center", va="center",
                transform=axis.transAxes,
            )
    title_parts = [part.strip() for part in title.split("|")]
    compact_title = " | ".join(title_parts[:3])
    z_labels: list[str] = []
    for plane, column in enumerate(setting["z_columns"], 1):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if not len(finite_values):
            label = "n/a"
        else:
            lower_z = float(np.min(finite_values))
            upper_z = float(np.max(finite_values))
            label = f"{lower_z:g}" if np.isclose(lower_z, upper_z) else f"{lower_z:g}–{upper_z:g}"
        z_labels.append(f"P{plane}={label}")
    z_title = (
        f"PLANE Z POSITIONS [{setting['z_unit']}]: " + "  |  ".join(z_labels)
        if z_labels else "PLANE Z POSITIONS: unavailable"
    )
    fig.suptitle(
        f"{compact_title}\n{z_title}\n"
        f"{angle_label} density comparison with shared bins and x-axis",
        fontsize=17,
        fontweight="bold",
    )
    plot_path = output_dir / f"{angle_slug}_gate_density_comparison.png"
    written = int(any(plotted_by_kind.values()))
    if written:
        fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    csv_path = output_dir / f"{angle_slug}_histogram_density.csv"
    pd.DataFrame(histogram_rows).to_csv(csv_path, index=False)
    return csv_path, written


def write_angular_histograms(
    frame: pd.DataFrame,
    gates: list[Gate],
    masks: dict[str, pd.Series],
    setting: dict[str, Any],
    output_dir: Path,
    title: str,
) -> tuple[list[Path], int]:
    results = [
        write_angular_histogram_comparison(
            frame, gates, masks, setting, variable, output_dir, title,
        )
        for variable in setting["variables"]
    ]
    return [csv_path for csv_path, _ in results], sum(written for _, written in results)


def topology_charge_scatter_setting(
    config: dict[str, Any], gates: list[Gate], available: set[str],
) -> dict[str, Any] | None:
    raw = config.get("topology_charge_scatter", {})
    if raw is False:
        return None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("topology_charge_scatter must be a YAML mapping or false")
    if not bool(raw.get("enabled", True)):
        return None
    gate_code = str(raw.get("gate_code", "100")).strip()
    if gate_code not in {gate.code for gate in gates}:
        raise ValueError(f"Topology scatter gate is not configured: {gate_code}")
    pattern = str(raw.get("topology", "1001")).strip()
    if not re.fullmatch(r"[01]{4}", pattern):
        raise ValueError("topology_charge_scatter.topology must be four binary digits")
    x_strip = int(raw.get("x_strip", 1))
    y_strip = int(raw.get("y_strip", 4))
    if x_strip not in range(1, 5) or y_strip not in range(1, 5):
        raise ValueError("topology_charge_scatter strip numbers must be in 1..4")
    raw_variables = raw.get("variables", [
        {"suffix": "qsum_cal", "label": "Calibrated Q_sum"},
        {"suffix": "tdif_cal", "label": "Calibrated T_dif"},
        {"suffix": "qdif_cal", "label": "Calibrated Q_dif"},
        {"suffix": "tsum_cal", "label": "Calibrated T_sum"},
    ])
    if not isinstance(raw_variables, list) or not raw_variables:
        raise ValueError("topology_charge_scatter.variables must be a nonempty list")
    variables: list[dict[str, str]] = []
    for index, item in enumerate(raw_variables, 1):
        if not isinstance(item, dict) or not str(item.get("suffix", "")).strip():
            raise ValueError(f"topology_charge_scatter.variables item {index} needs suffix")
        suffix = str(item["suffix"]).strip()
        label = str(item.get("label", suffix)).strip()
        if suffix in {variable["suffix"] for variable in variables}:
            raise ValueError(f"Duplicate topology scatter variable suffix: {suffix}")
        variables.append({"suffix": suffix, "label": label})
    columns = [
        f"p{plane}_s{strip}_{variable["suffix"]}"
        for variable in variables
        for plane in range(1, 5)
        for strip in (x_strip, y_strip)
    ]
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError("Topology scatter columns absent: " + ", ".join(missing))
    maximum = int(raw.get("max_points_per_file", 25000))
    if maximum < 1:
        raise ValueError("topology_charge_scatter.max_points_per_file must be positive")
    quantiles = raw.get("plot_quantiles", [0.001, 0.999])
    if not isinstance(quantiles, (list, tuple)) or len(quantiles) != 2:
        raise ValueError("topology_charge_scatter.plot_quantiles needs two values")
    low, high = float(quantiles[0]), float(quantiles[1])
    if not 0 <= low < high <= 1:
        raise ValueError("Topology scatter quantiles must satisfy 0 <= low < high <= 1")
    return {
        "gate_code": gate_code,
        "pattern": pattern,
        "x_strip": x_strip,
        "y_strip": y_strip,
        "variables": variables,
        "columns": columns,
        "maximum": maximum,
        "quantiles": (low, high),
    }


def topology_y_position_setting(
    config: dict[str, Any], gates: list[Gate], available: set[str],
) -> dict[str, Any] | None:
    raw = config.get("topology_y_position", {})
    if raw is False:
        return None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("topology_y_position must be a YAML mapping or false")
    if not bool(raw.get("enabled", True)):
        return None
    gate_code = str(raw.get("gate_code", "100")).strip()
    if gate_code not in {gate.code for gate in gates}:
        raise ValueError(f"Y-position plot gate is not configured: {gate_code}")
    pattern = str(raw.get("topology", "1001")).strip()
    if not re.fullmatch(r"[01]{4}", pattern):
        raise ValueError("topology_y_position.topology must be four binary digits")
    time_column = str(raw.get("time_column", "datetime")).strip()
    y_columns = [f"p{plane}_ypos" for plane in range(1, 5)]
    first_strip = int(raw.get("first_strip", 1))
    second_strip = int(raw.get("second_strip", 4))
    if first_strip not in range(1, 5) or second_strip not in range(1, 5) or first_strip == second_strip:
        raise ValueError("topology_y_position strips must be distinct values in 1..4")
    charge_suffix = str(raw.get("charge_suffix", "qsum_cal")).strip()
    charge_columns = [
        f"p{plane}_s{strip}_{charge_suffix}"
        for plane in range(1, 5) for strip in (first_strip, second_strip)
    ]
    columns = [time_column, *y_columns, *charge_columns]
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError("Topology Y-position columns absent: " + ", ".join(missing))
    bins = int(raw.get("bins", 80))
    if bins < 1:
        raise ValueError("topology_y_position.bins must be positive")
    value_range = raw.get("range", [-150, 150])
    if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
        raise ValueError("topology_y_position.range must contain [minimum, maximum]")
    lower, upper = float(value_range[0]), float(value_range[1])
    if lower >= upper:
        raise ValueError("topology_y_position.range must be increasing")
    maximum = int(raw.get("max_scatter_points_per_plane", 100000))
    if maximum < 1:
        raise ValueError("topology_y_position.max_scatter_points_per_plane must be positive")
    return {
        "gate_code": gate_code,
        "pattern": pattern,
        "time_column": time_column,
        "y_columns": y_columns,
        "first_strip": first_strip,
        "second_strip": second_strip,
        "charge_suffix": charge_suffix,
        "charge_columns": charge_columns,
        "columns": columns,
        "bins": bins,
        "range": (lower, upper),
        "maximum": maximum,
        "omit_zero": bool(raw.get("omit_zero", False)),
    }


def expand_topology_diagnostic_settings(
    config: dict[str, Any],
    gates: list[Gate],
    available: set[str],
    charge_base: dict[str, Any] | None,
    y_base: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if charge_base is None and y_base is None:
        return [], []
    base = charge_base if charge_base is not None else y_base
    targets: list[dict[str, Any]] = [{
        "gate_code": base["gate_code"],
        "topology": base["pattern"],
        "first_strip": (
            charge_base["x_strip"] if charge_base is not None else y_base["first_strip"]
        ),
        "second_strip": (
            charge_base["y_strip"] if charge_base is not None else y_base["second_strip"]
        ),
    }]
    additional = config.get("additional_topology_diagnostic_targets", [])
    if additional is None:
        additional = []
    if not isinstance(additional, list):
        raise ValueError("additional_topology_diagnostic_targets must be a list")
    targets.extend(additional)

    gate_codes = {gate.code for gate in gates}
    seen: set[tuple[str, str]] = set()
    charge_settings: list[dict[str, Any]] = []
    y_settings: list[dict[str, Any]] = []
    for index, target in enumerate(targets, 1):
        if not isinstance(target, dict):
            raise ValueError(f"Topology diagnostic target {index} must be a mapping")
        gate_code = str(target.get("gate_code", "")).strip()
        pattern = str(target.get("topology", "")).strip()
        first_strip = int(target.get("first_strip", 0))
        second_strip = int(target.get("second_strip", 0))
        if gate_code not in gate_codes:
            raise ValueError(f"Topology diagnostic target uses unknown gate {gate_code!r}")
        if not re.fullmatch(r"[01]{4}", pattern):
            raise ValueError(f"Topology diagnostic target has invalid topology {pattern!r}")
        if first_strip not in range(1, 5) or second_strip not in range(1, 5) or first_strip == second_strip:
            raise ValueError("Topology diagnostic target strips must be distinct values in 1..4")
        expected_pattern = "".join(
            "1" if strip in {first_strip, second_strip} else "0"
            for strip in range(1, 5)
        )
        if pattern != expected_pattern:
            raise ValueError(
                f"Topology {pattern} does not match active strips {first_strip},{second_strip}"
            )
        key = (gate_code, pattern)
        if key in seen:
            raise ValueError(f"Duplicate topology diagnostic target: gate {gate_code}, {pattern}")
        seen.add(key)

        if charge_base is not None:
            charge_setting = dict(charge_base)
            charge_setting.update(
                gate_code=gate_code, pattern=pattern,
                x_strip=first_strip, y_strip=second_strip,
            )
            charge_columns = [
                f"p{plane}_s{strip}_{variable['suffix']}"
                for variable in charge_setting["variables"]
                for plane in range(1, 5)
                for strip in (first_strip, second_strip)
            ]
            missing = sorted(set(charge_columns) - available)
            if missing:
                raise ValueError("Topology scatter columns absent: " + ", ".join(missing))
            charge_setting["columns"] = charge_columns
            raw_limits = target.get("diagonal_limits", {})
            if raw_limits is None:
                raw_limits = {}
            if not isinstance(raw_limits, dict):
                raise ValueError("Topology diagnostic diagonal_limits must be a mapping")
            known_suffixes = {variable["suffix"] for variable in charge_setting["variables"]}
            unknown_limits = sorted(set(raw_limits) - known_suffixes)
            if unknown_limits:
                raise ValueError(
                    "Topology diagnostic limits use unknown variables: "
                    + ", ".join(unknown_limits)
                )
            diagonal_limits = {key: float(value) for key, value in raw_limits.items()}
            if any(not np.isfinite(value) or value <= 0 for value in diagonal_limits.values()):
                raise ValueError("Topology diagnostic diagonal limits must be finite and positive")
            charge_setting["diagonal_limits"] = diagonal_limits
            charge_settings.append(charge_setting)

        if y_base is not None:
            y_setting = dict(y_base)
            charge_suffix = y_setting["charge_suffix"]
            y_charge_columns = [
                f"p{plane}_s{strip}_{charge_suffix}"
                for plane in range(1, 5)
                for strip in (first_strip, second_strip)
            ]
            y_columns = [
                y_setting["time_column"],
                *y_setting["y_columns"],
                *y_charge_columns,
            ]
            missing = sorted(set(y_columns) - available)
            if missing:
                raise ValueError("Topology Y-position columns absent: " + ", ".join(missing))
            y_setting.update(
                gate_code=gate_code, pattern=pattern,
                first_strip=first_strip, second_strip=second_strip,
                charge_columns=y_charge_columns, columns=y_columns,
            )
            y_settings.append(y_setting)
    return charge_settings, y_settings


def write_topology_charge_scatter(
    frame: pd.DataFrame,
    files: list[Product],
    masks: dict[str, pd.Series],
    setting: dict[str, Any],
    output_dir: Path,
    title: str,
) -> tuple[Path, Path, int]:
    gate_code = setting["gate_code"]
    pattern = setting["pattern"]
    x_strip = setting["x_strip"]
    y_strip = setting["y_strip"]
    variables = setting["variables"]
    maximum = setting["maximum"]
    quantiles = setting["quantiles"]
    diagonal_limits = setting.get("diagonal_limits", {})
    fig, axes = plt.subplots(
        len(variables), 4, figsize=(19, 4.3 * len(variables)),
        constrained_layout=True, squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors
    summary_rows: list[dict[str, Any]] = []
    range_values: dict[str, list[np.ndarray]] = {
        variable["suffix"]: [] for variable in variables
    }
    plotted = 0
    legend_handles = None
    legend_labels = None
    gate_mask = masks[gate_code]

    for row_index, variable in enumerate(variables):
        suffix = variable["suffix"]
        label = variable["label"]
        for plane in range(1, 5):
            axis = axes[row_index, plane - 1]
            topology_mask = frame[f"p{plane}_strip_topology"].astype(str).eq(pattern)
            plane_mask = gate_mask & topology_mask
            x_column = f"p{plane}_s{x_strip}_{suffix}"
            y_column = f"p{plane}_s{y_strip}_{suffix}"
            joined = frame.loc[plane_mask, [x_column, y_column]].apply(
                pd.to_numeric, errors="coerce",
            )
            joined = joined.loc[
                np.isfinite(joined[x_column])
                & np.isfinite(joined[y_column])
                & joined[x_column].ne(0)
                & joined[y_column].ne(0)
            ]
            joined_correlation = (
                joined[x_column].corr(joined[y_column]) if len(joined) > 1 else np.nan
            )
            summary_rows.append({
                "variable": suffix,
                "plane": plane,
                "source": "joined",
                "events": len(joined),
                f"strip_{x_strip}_median": joined[x_column].median() if len(joined) else np.nan,
                f"strip_{y_strip}_median": joined[y_column].median() if len(joined) else np.nan,
                "pearson_correlation": joined_correlation,
            })
            if not joined.empty:
                range_values[suffix].extend([
                    joined[x_column].to_numpy(dtype=float),
                    joined[y_column].to_numpy(dtype=float),
                ])
            panel_points = 0
            for file_index, product in enumerate(files):
                source_mask = plane_mask & frame["_source_basename"].eq(product.basename)
                part = frame.loc[source_mask, [x_column, y_column]].apply(
                    pd.to_numeric, errors="coerce",
                )
                part = part.loc[
                    np.isfinite(part[x_column])
                    & np.isfinite(part[y_column])
                    & part[x_column].ne(0)
                    & part[y_column].ne(0)
                ]
                correlation = part[x_column].corr(part[y_column]) if len(part) > 1 else np.nan
                summary_rows.append({
                    "variable": suffix,
                    "plane": plane,
                    "source": product.basename,
                    "events": len(part),
                    f"strip_{x_strip}_median": part[x_column].median() if len(part) else np.nan,
                    f"strip_{y_strip}_median": part[y_column].median() if len(part) else np.nan,
                    "pearson_correlation": correlation,
                })
                if len(part) > maximum:
                    part = part.sample(
                        maximum,
                        random_state=4100 + 100 * row_index + 10 * plane + file_index,
                    )
                if part.empty:
                    continue
                axis.scatter(
                    part[x_column], part[y_column], s=6, alpha=0.30, linewidths=0,
                    color=colors[file_index % len(colors)], label=product.basename,
                    rasterized=True,
                )
                panel_points += len(part)
                plotted += len(part)
            if row_index == 0:
                axis.set_title(f"Plane {plane}", fontsize=12)
            axis.text(
                0.98, 0.97, f"n={len(joined):,}", ha="right", va="top",
                transform=axis.transAxes, fontsize=8,
            )
            axis.set_xlabel(f"Strip {x_strip}")
            if plane == 1:
                axis.set_ylabel(f"{label}\nStrip {y_strip}")
            axis.grid(True, alpha=0.25)
            if not panel_points:
                axis.text(
                    0.5, 0.5, "No matching pairs", ha="center", va="center",
                    transform=axis.transAxes,
                )
            if legend_handles is None and panel_points:
                legend_handles, legend_labels = axis.get_legend_handles_labels()

    for row_index, variable in enumerate(variables):
        values_for_row = range_values[variable["suffix"]]
        if not values_for_row:
            continue
        values = np.concatenate(values_for_row)
        low, high = np.quantile(values, quantiles)
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = np.nanmin(values), np.nanmax(values)
        if low == high:
            padding = max(abs(float(low)) * 0.01, 1e-6)
            low, high = low - padding, high + padding
        for axis in axes[row_index, :]:
            axis.set_xlim(float(low), float(high))
            axis.set_ylim(float(low), float(high))
            axis.plot(
                [low, high], [low, high], linestyle="--", linewidth=0.8,
                color="black", alpha=0.6,
            )
            limit = diagonal_limits.get(variable["suffix"])
            if limit is not None:
                diagonal = np.asarray([low, high], dtype=float)
                axis.fill_between(
                    diagonal, diagonal - limit, diagonal + limit,
                    color="red", alpha=0.10, zorder=2,
                )
                axis.plot(
                    diagonal, diagonal - limit, color="red", linewidth=1.2,
                    zorder=3,
                )
                axis.plot(
                    diagonal, diagonal + limit, color="red", linewidth=1.2,
                    zorder=3,
                )
                axis.text(
                    0.02, 0.97, f"|Δ| < {limit:g}", color="red",
                    ha="left", va="top", transform=axis.transAxes, fontsize=8,
                )

    fig.suptitle(
        f"{title}\nGate {gate_code}, per-plane topology {pattern}: "
        f"strip {y_strip} versus strip {x_strip}",
        fontsize=14,
    )
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels, loc="upper right",
            bbox_to_anchor=(0.995, 0.995), fontsize=7,
        )
    plot_path = output_dir / (
        f"gate_{gate_code}_topology_{pattern}_"
        f"strip_{y_strip}_vs_{x_strip}_variables.png"
    )
    if plotted:
        fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    summary_path = output_dir / (
        f"gate_{gate_code}_topology_{pattern}_variable_summary.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return plot_path, summary_path, int(plotted > 0)


def write_topology_y_position(
    frame: pd.DataFrame,
    files: list[Product],
    masks: dict[str, pd.Series],
    setting: dict[str, Any],
    output_dir: Path,
    title: str,
) -> tuple[Path, Path, Path, int]:
    gate_code = setting["gate_code"]
    pattern = setting["pattern"]
    time_column = setting["time_column"]
    first_strip = setting["first_strip"]
    second_strip = setting["second_strip"]
    charge_suffix = setting["charge_suffix"]
    lower, upper = setting["range"]
    position_fig, position_axes = plt.subplots(
        2, 4, figsize=(20, 9), constrained_layout=True, squeeze=False,
    )
    event_rows: list[pd.DataFrame] = []
    total_events = 0
    gate_mask = masks[gate_code]
    for plane in range(1, 5):
        topology_mask = frame[f"p{plane}_strip_topology"].astype(str).eq(pattern)
        plane_mask = gate_mask & topology_mask
        y_column = f"p{plane}_ypos"
        first_column = f"p{plane}_s{first_strip}_{charge_suffix}"
        second_column = f"p{plane}_s{second_strip}_{charge_suffix}"
        values = pd.to_numeric(frame.loc[plane_mask, y_column], errors="coerce")
        first_charge = pd.to_numeric(frame.loc[plane_mask, first_column], errors="coerce")
        second_charge = pd.to_numeric(frame.loc[plane_mask, second_column], errors="coerce")
        charge_total = first_charge + second_charge
        times = pd.to_datetime(frame.loc[plane_mask, time_column], errors="coerce")
        sources = frame.loc[plane_mask, "_source_basename"].astype(str)
        valid = (
            values.notna() & np.isfinite(values)
            & first_charge.notna() & np.isfinite(first_charge)
            & second_charge.notna() & np.isfinite(second_charge)
            & charge_total.ne(0) & times.notna()
        )
        if setting["omit_zero"]:
            valid &= values.ne(0)
        data = pd.DataFrame({
            "datetime": times.loc[valid],
            "source_basename": sources.loc[valid],
            "gate_code": gate_code,
            "plane_topology": pattern,
            "plane": plane,
            "y_position": values.loc[valid],
            f"charge_strip_{first_strip}": first_charge.loc[valid],
            f"charge_strip_{second_strip}": second_charge.loc[valid],
            "charge_pair_sum": charge_total.loc[valid],
            f"charge_fraction_strip_{first_strip}": (
                first_charge.loc[valid] / charge_total.loc[valid]
            ),
            "charge_asymmetry": (
                (first_charge.loc[valid] - second_charge.loc[valid])
                / charge_total.loc[valid]
            ),
        }).sort_values("datetime")
        event_rows.append(data)
        total_events += len(data)

        time_axis = position_axes[0, plane - 1]
        histogram_axis = position_axes[1, plane - 1]
        scatter_data = data
        if len(scatter_data) > setting["maximum"]:
            scatter_data = scatter_data.sample(
                setting["maximum"], random_state=8100 + plane,
            ).sort_values("datetime")
        if data.empty:
            time_axis.text(0.5, 0.5, "No matching Y values", ha="center", va="center",
                           transform=time_axis.transAxes)
            histogram_axis.text(0.5, 0.5, "No matching Y values", ha="center", va="center",
                                transform=histogram_axis.transAxes)
        else:
            time_axis.scatter(
                scatter_data["datetime"], scatter_data["y_position"],
                s=5, alpha=0.35, linewidths=0, color=f"C{plane - 1}", rasterized=True,
            )
            histogram_axis.hist(
                data["y_position"], bins=setting["bins"], range=(lower, upper),
                color=f"C{plane - 1}", alpha=0.70, edgecolor="black", linewidth=0.35,
            )
            median = float(data["y_position"].median())
            histogram_axis.axvline(
                median, color="crimson", linestyle="--", linewidth=1.2,
                label=f"median={median:.3g}",
            )
            histogram_axis.legend(fontsize=8)
        time_axis.set_title(f"Plane {plane} | n={len(data):,}")
        time_axis.set_xlabel("Time")
        time_axis.set_ylabel("Selected Y position")
        time_axis.set_ylim(lower, upper)
        time_axis.grid(True, alpha=0.25)
        histogram_axis.set_xlabel("Selected Y position")
        histogram_axis.set_ylabel("Events")
        histogram_axis.set_xlim(lower, upper)
        histogram_axis.grid(True, alpha=0.25)

    position_fig.suptitle(
        f"{title} | Gate {gate_code}, per-plane strip topology {pattern}: selected Y position",
        fontsize=14,
    )
    position_plot_path = (
        output_dir / f"gate_{gate_code}_topology_{pattern}_selected_y_position.png"
    )
    position_fig.savefig(position_plot_path, dpi=160)
    plt.close(position_fig)

    all_data = pd.concat(event_rows, ignore_index=True)
    charge_fig, charge_axes = plt.subplots(
        2, 4, figsize=(20, 10), constrained_layout=True, squeeze=False,
    )
    colored_scatter: Any = None
    fraction_column = f"charge_fraction_strip_{first_strip}"
    for plane in range(1, 5):
        data = all_data.loc[all_data["plane"].eq(plane)]
        fraction_axis = charge_axes[0, plane - 1]
        pair_axis = charge_axes[1, plane - 1]
        scatter_data = data
        if len(scatter_data) > setting["maximum"]:
            scatter_data = scatter_data.sample(
                setting["maximum"], random_state=9100 + plane,
            )
        if data.empty:
            fraction_axis.text(0.5, 0.5, "No matching charge pairs", ha="center", va="center",
                               transform=fraction_axis.transAxes)
            pair_axis.text(0.5, 0.5, "No matching charge pairs", ha="center", va="center",
                           transform=pair_axis.transAxes)
        else:
            fraction_axis.scatter(
                scatter_data[fraction_column], scatter_data["y_position"],
                s=6, alpha=0.30, linewidths=0, color=f"C{plane - 1}", rasterized=True,
            )
            fraction_axis.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
            first_values = scatter_data[f"charge_strip_{first_strip}"]
            second_values = scatter_data[f"charge_strip_{second_strip}"]
            colored_scatter = pair_axis.scatter(
                first_values, second_values, c=scatter_data["y_position"],
                cmap="coolwarm", vmin=lower, vmax=upper, s=7, alpha=0.38,
                linewidths=0, rasterized=True,
            )
            pair_values = np.concatenate([
                data[f"charge_strip_{first_strip}"].to_numpy(dtype=float),
                data[f"charge_strip_{second_strip}"].to_numpy(dtype=float),
            ])
            charge_low, charge_high = np.quantile(pair_values, [0.001, 0.999])
            if charge_low == charge_high:
                padding = max(abs(float(charge_low)) * 0.01, 1e-6)
                charge_low, charge_high = charge_low - padding, charge_high + padding
            pair_axis.set_xlim(float(charge_low), float(charge_high))
            pair_axis.set_ylim(float(charge_low), float(charge_high))
            pair_axis.plot(
                [charge_low, charge_high], [charge_low, charge_high],
                color="black", linestyle="--", linewidth=0.8,
            )
        fraction_axis.set_title(f"Plane {plane} | n={len(data):,}")
        fraction_axis.set_xlabel(
            f"Q(strip {first_strip}) / [Q(strip {first_strip}) + Q(strip {second_strip})]"
        )
        fraction_axis.set_ylabel("Selected Y position")
        fraction_axis.set_xlim(-0.02, 1.02)
        fraction_axis.set_ylim(lower, upper)
        fraction_axis.grid(True, alpha=0.25)
        pair_axis.set_xlabel(f"Q(strip {first_strip})")
        pair_axis.set_ylabel(f"Q(strip {second_strip})")
        pair_axis.grid(True, alpha=0.25)
    if colored_scatter is not None:
        charge_fig.colorbar(
            colored_scatter, ax=charge_axes[1, :].tolist(),
            label="Selected Y position", shrink=0.90,
        )
    charge_fig.suptitle(
        f"{title} | Gate {gate_code}, topology {pattern}: active-strip charges versus selected Y",
        fontsize=14,
    )
    charge_plot_path = (
        output_dir / f"gate_{gate_code}_topology_{pattern}_charge_fraction_vs_y.png"
    )
    charge_fig.savefig(charge_plot_path, dpi=160)
    plt.close(charge_fig)

    csv_path = output_dir / f"gate_{gate_code}_topology_{pattern}_selected_y_position.csv"
    all_data.to_csv(csv_path, index=False)
    print(
        f"Topology Y/charge diagnostic: gate={gate_code}, topology={pattern}, "
        f"values={total_events:,}, files={len(files)}"
    )
    return position_plot_path, charge_plot_path, csv_path, 2


def main() -> int:
    args = arguments()
    config = load_configuration(args.config)
    gates = parse_gates(config["gates"])
    topology_setting = derived_topology_setting(config)
    root = STATIONS_ROOT / config["station_name"]
    lake = root / "STAGE_1_PRODUCTS" / "EVENT_DATA" / "PARQUET_LAKE"
    simulation_selection: Mingo00Selection | None = None
    if config["station_name"] == "MINGO00":
        simulation_selection = select_mingo00_products(
            lake,
            config["maximum"],
            config["close_parameters"],
            product_factory=Product,
            parquet_basename=parquet_basename,
            acquisition_time=acquisition_time,
        )
        files = simulation_selection.products
        print(
            f"Found {simulation_selection.candidate_count} MINGO00 products with simulation "
            f"metadata; selected tightest parameter cluster of {len(files)} using: "
            + ", ".join(simulation_selection.close_parameters)
        )
        print(f"Parameter-cluster center: {simulation_selection.center_basename}")
        for index, product in enumerate(files, 1):
            param_id = simulation_selection.param_set_id_by_basename[product.basename]
            distance = simulation_selection.distance_by_basename[product.basename]
            values = simulation_selection.values_by_basename[product.basename]
            print(
                f"  {index}. param_set_id={param_id} distance={distance:.6g} "
                f"{product.basename}  {values}"
            )
    else:
        candidates = discover(lake, config["start"], config["end"])
        files = tightest_block(candidates, config["maximum"])
        print(f"Found {len(candidates)} files in range; selected tightest consecutive block of {len(files)}:")
        for index, product in enumerate(files, 1):
            print(f"  {index}. {product.acquired}  {product.basename}")
        if len(files) > 1:
            print(f"Acquisition-start span: {files[-1].acquired - files[0].acquired}")
    available, types = schemas(files)
    available_set = set(available)
    show_columns(available, types)

    derived_names = derived_column_names() if topology_setting["enabled"] else []
    if derived_names:
        print("Derived columns available to gate conditions:")
        for name in derived_names:
            print(f"  - {name}")
        print()
    gate_columns: set[str] = set()
    for gate in gates:
        gate_columns.update(condition_columns(gate.condition, location=f"gate[{gate.code}]"))
    allowed_columns = available_set | set(derived_names)
    missing_gate_columns = sorted(gate_columns - allowed_columns)
    if missing_gate_columns:
        raise ValueError("Gate condition columns absent from schema: " + ", ".join(missing_gate_columns))
    source_columns = topology_source_columns(topology_setting)
    missing_sources = sorted(set(source_columns) - available_set)
    if missing_sources:
        raise ValueError("Derived strip-topology source columns absent: " + ", ".join(missing_sources))
    rate_setting = time_series_setting(config, available_set)
    efficiency_setting = efficiency_time_series_setting(config, gates, available_set)
    gate_comparison = enabled_gate_comparison_setting(config, available_set)
    plane_combination = plane_combination_setting(config, gates, available_set)
    histogram_setting = angular_histogram_setting(config, available_set)
    calibration_comparison = charge_calibration_comparison_setting(
        config, available_set,
    )
    charge_scatter_setting = topology_charge_scatter_setting(
        config, gates, available_set,
    )
    y_position_setting = topology_y_position_setting(config, gates, available_set)
    charge_scatter_settings, y_position_settings = expand_topology_diagnostic_settings(
        config, gates, available_set, charge_scatter_setting, y_position_setting,
    )
    needed = list(dict.fromkeys([
        *(column for column in gate_columns if column in available_set),
        *source_columns,
        *([] if rate_setting is None else [rate_setting["column"]]),
        *([] if efficiency_setting is None else [
            efficiency_setting["time_column"], efficiency_setting["topology_column"],
        ]),
        *([] if gate_comparison is None else [
            gate_comparison["time_column"], gate_comparison["topology_column"],
        ]),
        *([] if plane_combination is None else [
            plane_combination["time_column"], plane_combination["topology_column"],
        ]),
        *([] if histogram_setting is None else [
            *(variable["column"] for variable in histogram_setting["variables"]),
            *histogram_setting["z_columns"],
        ]),
        *([] if calibration_comparison is None else calibration_comparison["columns"]),
        *(column for setting in charge_scatter_settings for column in setting["columns"]),
        *(column for setting in y_position_settings for column in setting["columns"]),
    ]))
    frame = read_events(files, needed)
    add_derived_topology_columns(frame, topology_setting)
    masks = assign_gates(frame, gates)

    interval = (
        "PARAMETER_CLOSE"
        if simulation_selection is not None
        else f"{config['start']:%Y%m%d_%H%M%S}_{config['end']:%Y%m%d_%H%M%S}"
    )
    group = f"{files[0].basename}_{files[-1].basename}"
    output = root / "STAGE_1_PRODUCTS_TESTS" / OUTPUT_NAME / f"{interval}_{group}"
    rate_dir = output / "RATE_TIMESERIES"
    efficiency_dir = output / "EFFICIENCY_TIMESERIES"
    gate_comparison_dir = output / "ENABLED_GATE_COMPARISON"
    plane_combination_dir = output / "PLANE_COMBINATIONS"
    angular_dir = output / "ANGULAR_HISTOGRAMS"
    charge_calibration_dir = output / "CHARGE_CALIBRATION_COMPARISON"
    scatter_dir = output / "SCATTERS"
    for directory in (
        output, rate_dir, efficiency_dir, gate_comparison_dir, plane_combination_dir, angular_dir,
        charge_calibration_dir, scatter_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (
        rate_dir, efficiency_dir, gate_comparison_dir, plane_combination_dir, angular_dir,
        charge_calibration_dir, scatter_dir,
    ):
        for pattern in ("*.png", "*.csv"):
            for obsolete in directory.glob(pattern):
                obsolete.unlink()

    manifest_path = output / "selected_files.csv"
    pd.DataFrame([
        {
            "selection_order": index,
            "station": config["station_name"],
            "filename_base": product.basename,
            "acquisition_datetime": product.acquired,
            "rows": pq.ParquetFile(product.path).metadata.num_rows,
            "path": str(product.path),
            **(
                simulation_selection.manifest_fields(product.basename)
                if simulation_selection is not None
                else {"selection_mode": "chronological_minimum_span"}
            ),
        }
        for index, product in enumerate(files, 1)
    ]).to_csv(manifest_path, index=False)
    summary_path = output / "gate_summary.csv"
    summary = write_gate_summary(frame, gates, masks, summary_path)
    print("\nGate population summary:")
    print(summary.to_string(index=False))

    if simulation_selection is not None:
        title = (
            f"{root.name} | {len(files)} simulation-parameter-close files | "
            f"center={simulation_selection.center_basename} | "
            "parameters=" + ",".join(simulation_selection.close_parameters)
        )
    else:
        title = (
            f"{root.name} | {len(files)} consecutive files | "
            f"{files[0].acquired:%Y-%m-%d %H:%M:%S} to "
            f"{files[-1].acquired:%Y-%m-%d %H:%M:%S}"
        )
    selected_start = min(product.acquired for product in files)
    selected_end = max(product.acquired for product in files)
    calibration_paths = generate_calibration_context(
        root,
        selected_start if simulation_selection is not None else config["start"],
        selected_end if simulation_selection is not None else config["end"],
        output / "00_CALIBRATION_CONTEXT",
        title,
        {product.basename for product in files},
        context=config.get("context", "full"),
    )
    rate_csv: Path | None = None
    rate_plots = 0
    if rate_setting is not None:
        rate_csv, rate_plots = write_gate_time_series(
            frame, gates, masks, rate_setting, rate_dir, title,
        )
    efficiency_csv: Path | None = None
    efficiency_plots = 0
    if efficiency_setting is not None:
        efficiency_csv, efficiency_plots = write_efficiency_time_series(
            frame, gates, masks, efficiency_setting, efficiency_dir, title,
        )
    gate_efficiency_plot: Path | None = None
    gate_metrics_plot: Path | None = None
    gate_comparison_csv: Path | None = None
    gate_comparison_plots = 0
    if gate_comparison is not None:
        (
            gate_efficiency_plot,
            gate_metrics_plot,
            gate_comparison_csv,
            gate_comparison_plots,
        ) = write_enabled_gate_comparison(
            frame, gates, masks, gate_comparison, gate_comparison_dir, title,
        )
    plane_combination_plot: Path | None = None
    plane_combination_csv: Path | None = None
    plane_combination_plots = 0
    if plane_combination is not None:
        plane_combination_plot, plane_combination_csv, plane_combination_plots = (
            write_plane_combination_timeseries(
                frame, gates, masks, plane_combination, plane_combination_dir, title,
            )
        )
    angular_csvs: list[Path] = []
    angular_plots = 0
    if histogram_setting is not None:
        angular_csvs, angular_plots = write_angular_histograms(
            frame, gates, masks, histogram_setting, angular_dir, title,
        )
    charge_calibration_plot: Path | None = None
    charge_calibration_values = 0
    if calibration_comparison is not None:
        charge_calibration_plot, charge_calibration_values = (
            write_charge_calibration_comparison(
                frame, calibration_comparison, charge_calibration_dir, title,
            )
        )
    charge_scatter_results = [
        write_topology_charge_scatter(
            frame, files, masks, setting, scatter_dir, title,
        )
        for setting in charge_scatter_settings
    ]
    charge_scatter_plots = sum(result[2] for result in charge_scatter_results)
    y_position_results = [
        write_topology_y_position(
            frame, files, masks, setting, scatter_dir, title,
        )
        for setting in y_position_settings
    ]
    y_position_plots = sum(result[3] for result in y_position_results)
    print(f"\nJoined {len(frame):,} event rows")
    print(
        f"Wrote {len(calibration_paths) - 1} calibration context plot(s), "
        f"{rate_plots} gate-rate plot(s), {efficiency_plots} efficiency plot(s), "
        f"{gate_comparison_plots} enabled-gate comparison plot(s), "
        f"{plane_combination_plots} plane-combination plot(s), "
        f"{angular_plots} angular histogram(s), "
        f"{int(charge_calibration_plot is not None)} charge-calibration comparison "
        f"plot(s) from {charge_calibration_values:,} plane events, "
        f"{charge_scatter_plots} topology-charge scatter plot(s), and "
        f"{y_position_plots} topology-Y plot(s)"
    )
    print(f"Selected-file manifest: {manifest_path}")
    print(f"Gate summary: {summary_path}")
    if rate_csv is not None:
        print(f"Gate rate data: {rate_csv}")
    if efficiency_csv is not None:
        print(f"Efficiency time-series data: {efficiency_csv}")
    if gate_efficiency_plot is not None:
        print(f"Enabled-gate efficiency overlay: {gate_efficiency_plot}")
    if gate_metrics_plot is not None:
        print(f"Enabled-gate rate/efficiency comparison: {gate_metrics_plot}")
    if gate_comparison_csv is not None:
        print(f"Enabled-gate comparison data: {gate_comparison_csv}")
    if plane_combination_plot is not None:
        print(f"Plane-combination plot: {plane_combination_plot}")
    if plane_combination_csv is not None:
        print(f"Plane-combination data: {plane_combination_csv}")
    for angular_csv in angular_csvs:
        print(f"Angular histogram data: {angular_csv}")
    if charge_calibration_plot is not None:
        print(f"Charge-calibration comparison: {charge_calibration_plot}")
    for plot_path, csv_path, plotted in charge_scatter_results:
        if plotted:
            print(f"Topology charge scatter: {plot_path}")
        print(f"Topology charge scatter summary: {csv_path}")
    for position_path, charge_path, csv_path, _ in y_position_results:
        print(f"Topology Y-position plot: {position_path}")
        print(f"Topology Y/charge plot: {charge_path}")
        print(f"Topology Y/charge data: {csv_path}")
    print(f"Outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
