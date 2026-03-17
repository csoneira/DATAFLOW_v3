"""
DATAFLOW_v3 Script Header v1
Script: MASTER/common/step1_shared.py
Purpose: Shared STEP_1 helpers for TASK_1..TASK_5 scripts.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MASTER/common/step1_shared.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import fcntl
import math
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_STATION_CHOICES: Tuple[str, ...] = ("0", "1", "2", "3", "4")
DEFAULT_IMPORTANT_KEYWORDS: Tuple[str, ...] = (
    "error",
    "warning",
    "failed",
    "exception",
    "traceback",
    "usage",
)

EVENTS_PER_SECOND_MAX = 40
EVENTS_PER_SECOND_COLUMNS = [
    *(f"events_per_second_{idx}_count" for idx in range(EVENTS_PER_SECOND_MAX + 1)),
    "events_per_second_total_seconds",
    "events_per_second_global_rate",
]
EVENTS_PER_SECOND_COLUMN_SET = frozenset(EVENTS_PER_SECOND_COLUMNS)
TRACEABILITY_COLUMNS: Tuple[str, ...] = ("filename_base", "execution_timestamp", "param_hash")
CANONICAL_TT_LABELS: frozenset[str] = frozenset(
    {
        "0",
        "1",
        "2",
        "3",
        "4",
        "12",
        "13",
        "14",
        "23",
        "24",
        "34",
        "123",
        "124",
        "134",
        "234",
        "1234",
    }
)
TT_RATE_COLUMN_RE = re.compile(r"^(?P<prefix>.+_tt)_(?P<label>[^_]+)_rate_hz$")


def _metadata_value_is_empty(value: object) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def normalize_tt_label(value: object, default: str = "0") -> str:
    """Return canonical TT label text (e.g. 12.0 -> 12, 12.0_134.0 -> 12_134)."""
    if value is None:
        return default

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return default
        if float(value).is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if not text:
        return default

    if "_" in text:
        return "_".join(normalize_tt_label(part, default=default) for part in text.split("_"))

    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text

    if not np.isfinite(numeric):
        return default
    if numeric.is_integer():
        return str(int(numeric))
    return text


def _normalize_metadata_tt_key(key: str) -> str:
    if "_tt_" not in key:
        return key

    suffix = ""
    for candidate in ("_count_rate_hz", "_count", "_fraction"):
        if key.endswith(candidate):
            suffix = candidate
            key = key[: -len(candidate)]
            break

    prefix, tt_fragment = key.rsplit("_tt_", 1)
    normalized_fragment = normalize_tt_label(tt_fragment, default="0")
    return f"{prefix}_tt_{normalized_fragment}{suffix}"


def _normalize_metadata_fieldnames(fieldnames: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for raw_name in fieldnames:
        if raw_name is None:
            continue
        name = _normalize_metadata_tt_key(str(raw_name))
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _is_entries_metric_key(key: str) -> bool:
    return key.endswith("_entries") or "_entries_" in key


RATE_HISTOGRAM_REQUIRED_COLUMNS = frozenset(
    {
        "count_rate_denominator_seconds",
        "events_per_second_total_seconds",
        "events_per_second_global_rate",
    }
)
EVENTS_PER_SECOND_RATE_HZ_RE = re.compile(r"^events_per_second_(\d+)_rate_hz$")


def is_rate_histogram_metadata_column(column_name: str) -> bool:
    if column_name in RATE_HISTOGRAM_REQUIRED_COLUMNS:
        return True
    match = EVENTS_PER_SECOND_RATE_HZ_RE.match(column_name)
    if match:
        return int(match.group(1)) <= EVENTS_PER_SECOND_MAX
    return False


def is_rate_histogram_file_column(column_name: str) -> bool:
    if column_name in TRACEABILITY_COLUMNS:
        return True
    return is_rate_histogram_metadata_column(column_name)


def is_redundant_count_metadata_column(column_name: str) -> bool:
    if column_name == "count_rate_denominator_seconds":
        return False
    if column_name.endswith("_count"):
        return True
    if column_name.endswith("_fraction"):
        return True
    if _is_entries_metric_key(column_name) and not column_name.endswith("_rate_hz"):
        return True
    return False


def is_specific_metadata_excluded_column(column_name: str) -> bool:
    if column_name.startswith("activation_"):
        return True
    if column_name.startswith("streamer_contagion_"):
        return True
    return is_redundant_count_metadata_column(column_name) or is_rate_histogram_metadata_column(
        column_name
    )


def extract_rate_histogram_metadata(
    metadata: Dict[str, object],
    remove_from_source: bool = True,
) -> Dict[str, object]:
    extracted: Dict[str, object] = {}
    for key in TRACEABILITY_COLUMNS:
        if key in metadata:
            extracted[key] = metadata[key]

    for key in list(metadata.keys()):
        if not isinstance(key, str):
            continue
        if not is_rate_histogram_metadata_column(key):
            continue
        extracted[key] = metadata[key]
        if remove_from_source:
            metadata.pop(key, None)

    return extracted


def is_trigger_type_metadata_column(column_name: str, tt_prefixes: Iterable[str]) -> bool:
    for prefix in tt_prefixes:
        if column_name.startswith(f"{prefix}_"):
            return True
    return False


def is_trigger_type_file_column(column_name: str, tt_prefixes: Iterable[str]) -> bool:
    if column_name in TRACEABILITY_COLUMNS:
        return True
    if column_name == "count_rate_denominator_seconds":
        return True
    if is_trigger_type_metadata_column(column_name, tt_prefixes):
        return column_name.endswith("_rate_hz")
    return False


def extract_trigger_type_metadata(
    metadata: Dict[str, object],
    tt_prefixes: Iterable[str],
    remove_from_source: bool = True,
) -> Dict[str, object]:
    prefixes = tuple(str(prefix) for prefix in tt_prefixes)
    extracted: Dict[str, object] = {}

    for key in TRACEABILITY_COLUMNS:
        if key in metadata:
            extracted[key] = metadata[key]

    for key in list(metadata.keys()):
        if not isinstance(key, str):
            continue
        if not is_trigger_type_metadata_column(key, prefixes):
            continue
        extracted[key] = metadata[key]
        if remove_from_source:
            metadata.pop(key, None)

    return extracted


def prune_redundant_count_metadata(
    metadata: Dict[str, object],
    log_fn: Callable[..., None] = builtins.print,
) -> int:
    dropped: list[str] = []
    for key in list(metadata.keys()):
        if not isinstance(key, str):
            continue
        if key == "count_rate_denominator_seconds":
            continue

        if key.endswith("_count"):
            rate_key = key[: -len("_count")] + "_rate_hz"
            if rate_key in metadata:
                dropped.append(key)
            continue

        if _is_entries_metric_key(key) and not key.endswith("_rate_hz"):
            rate_key = f"{key}_rate_hz"
            if rate_key in metadata:
                dropped.append(key)
            continue

        if key.endswith("_fraction"):
            dropped.append(key)

    for key in dropped:
        metadata.pop(key, None)

    if dropped:
        log_fn(f"[metadata] Dropped {len(dropped)} redundant count/fraction column(s).")
    return len(dropped)


def build_step1_cli_parser(
    description: str,
    station_choices: Sequence[str] = DEFAULT_STATION_CHOICES,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "station",
        nargs="?",
        choices=tuple(station_choices),
        help="Station identifier (0, 1, 2, 3, or 4).",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Optional input file path to process instead of auto-selecting.",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file_flag",
        help="Optional input file path (named form).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose execution logging.",
    )
    return parser


def validate_step1_input_file_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.input_file and args.input_file_flag:
        parser.error("Use either positional input_file or --input-file, not both.")


def build_step1_filtered_print(
    verbose: bool,
    debug_mode_getter: Callable[[], bool],
    raw_print: Callable[..., None] = builtins.print,
    important_keywords: Sequence[str] = DEFAULT_IMPORTANT_KEYWORDS,
) -> Callable[..., None]:
    important = tuple(keyword.lower() for keyword in important_keywords)

    def is_important_message(message: str) -> bool:
        lowered = message.lower()
        if any(keyword in lowered for keyword in important):
            return True
        return "total execution time" in lowered or "data purity" in lowered

    def filtered_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or debug_mode_getter() or verbose:
            raw_print(*args, **kwargs)
            return
        message = " ".join(str(arg) for arg in args)
        if is_important_message(message):
            raw_print(*args, **kwargs)

    return filtered_print


def resolve_step1_station(
    cli_station: str | None,
    run_jupyter_notebook: bool,
    default_station: str,
    parser: argparse.ArgumentParser,
    station_choices: Sequence[str] = DEFAULT_STATION_CHOICES,
    config_hint: str = "TASK_<N>/config_task_<N>.yaml",
) -> str:
    if cli_station is not None:
        station = str(cli_station)
    elif run_jupyter_notebook:
        station = str(default_station)
    else:
        parser.error(
            f"No station provided. Pass <station> or enable run_jupyter_notebook in {config_hint}."
        )

    if station not in station_choices:
        parser.error("Invalid station. Choose one of: 0, 1, 2, 3, 4.")
    return station


def apply_step1_task_parameter_overrides(
    config_obj: dict,
    station_id: str,
    task_parameter_path: str | Path,
    fallback_parameter_path: str | Path,
    task_number: int,
    update_fn: Callable[[dict, Path | str, str], dict],
    log_fn: Callable[..., None] = builtins.print,
) -> dict:
    task_path = Path(task_parameter_path)
    if task_path.exists():
        config_obj = update_fn(config_obj, task_path, station_id)
        log_fn(f"Warning: Loaded task parameters from {task_path}")
        return config_obj

    fallback_path = Path(fallback_parameter_path)
    if fallback_path.exists():
        log_fn(f"Warning: Task parameters file not found; falling back to {fallback_path}")
        config_obj = update_fn(config_obj, fallback_path, station_id)
    else:
        log_fn(f"Warning: No parameters file found for task {task_number}")
    return config_obj


def normalize_analysis_mode_value(value: object) -> str:
    """Normalise analysis_mode to a non-negative integer string.

    analysis_mode is a prime-product composite that encodes per-calibration
    modes (charge_side × charge_front_back × time_calibration ×
    time_dif_calibration).  Any positive integer is valid; empty string is
    used when the value is absent.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    # Strip trailing .0 produced by float-to-string conversion
    if text.endswith(".0"):
        text = text[:-2]
    # Accept any non-negative integer string
    if text.lstrip("-").isdigit() and not text.startswith("-"):
        return text
    return ""


def _normalize_plot_mode(raw_value: object) -> str:
    if raw_value is None:
        return "none"
    if isinstance(raw_value, bool):
        return "all" if raw_value else "none"

    mode = str(raw_value).strip().lower()
    if mode in {"", "none", "null", "false", "0", "off"}:
        return "none"
    if mode in {"debug"}:
        return "debug"
    if mode in {"usual", "standard", "normal"}:
        return "usual"
    if mode in {"essential"}:
        return "essential"
    if mode in {"all", "true", "1", "on"}:
        return "all"

    raise ValueError(
        "Invalid create_plots value. Use one of: null/none, debug, usual, essential, all."
    )


def resolve_step1_plot_options(
    config_obj: Dict[str, object],
) -> Tuple[str, bool, bool, bool, bool, bool, bool]:
    """Return normalized plotting mode and derived plotting toggles.

    Output tuple:
    (plot_mode, create_plots, create_essential_plots, save_plots,
     create_pdf, show_plots, create_debug_plots)
    """
    plot_mode = _normalize_plot_mode(config_obj.get("create_plots", None))
    create_plots = plot_mode in {"usual", "all"}
    create_debug_plots = plot_mode in {"debug", "all"}
    create_essential_plots = plot_mode in {"usual", "essential", "all"}
    save_plots = plot_mode != "none"
    create_pdf = save_plots
    show_plots = False
    return (
        plot_mode,
        create_plots,
        create_essential_plots,
        save_plots,
        create_pdf,
        show_plots,
        create_debug_plots,
    )


STEP1_PLOT_STATUSES: Tuple[str, ...] = ("none", "debug", "usual", "essential")


def _normalize_step1_plot_catalog_status(raw_status: object) -> str:
    if raw_status is None:
        return "none"
    if isinstance(raw_status, bool):
        return "usual" if raw_status else "none"

    status = str(raw_status).strip().lower()
    if status in {"", "none", "null", "false", "0", "off"}:
        return "none"
    if status in {"true", "1", "on"}:
        return "usual"
    if status in {"debug", "usual", "essential"}:
        return status
    return status


def load_step1_task_plot_catalog(
    catalog_path: Path,
    plot_aliases: Sequence[str],
    task_label: str,
    log_fn: Callable[..., None] = builtins.print,
) -> Dict[str, str]:
    """Load and validate a STEP_1 task plot catalog YAML.

    Expected YAML shape:
      plots:
        alias_name: null|debug|usual|essential
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"{task_label} plot catalog not found: {catalog_path}")

    import yaml

    with catalog_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, Mapping):
        raise ValueError(f"{task_label} plot catalog must be a mapping: {catalog_path}")

    raw_plots = loaded.get("plots", {})
    if not isinstance(raw_plots, Mapping):
        raise ValueError(f"'plots' entry in {catalog_path} must be a mapping.")

    aliases_set = set(plot_aliases)
    catalog: Dict[str, str] = {}
    for alias, raw_status in raw_plots.items():
        alias_name = str(alias)
        status = _normalize_step1_plot_catalog_status(raw_status)
        if status not in STEP1_PLOT_STATUSES:
            raise ValueError(
                f"Invalid status {raw_status!r} for {task_label} plot alias {alias_name!r} in {catalog_path}."
            )
        if alias_name not in aliases_set:
            log_fn(
                f"{task_label} plot catalog entry {alias_name!r} is unknown and will be ignored.",
            )
            continue
        catalog[alias_name] = status

    for alias in plot_aliases:
        if alias not in catalog:
            log_fn(
                f"{task_label} plot alias {alias!r} missing in {catalog_path.name}; defaulting to 'usual'."
            )
            catalog[alias] = "usual"

    return catalog


def step1_task_plot_enabled(
    alias: str,
    status_by_alias: Mapping[str, str],
    plot_mode: str,
) -> bool:
    """Return whether a task plot alias is enabled for the current mode."""
    if alias not in status_by_alias:
        raise KeyError(f"Unknown STEP_1 task plot alias: {alias}")

    status = status_by_alias[alias]

    if plot_mode == "none":
        return False
    if status == "none":
        return False
    if plot_mode == "all":
        return True
    if plot_mode == "debug":
        return status == "debug"
    if plot_mode == "usual":
        return status in {"usual", "essential"}
    if plot_mode == "essential":
        return status == "essential"
    return False


def sanitize_analysis_mode_rows(rows: List[Dict[str, object]]) -> int:
    fixed = 0
    for row in rows:
        if "analysis_mode" not in row:
            continue
        clean_value = normalize_analysis_mode_value(row.get("analysis_mode"))
        if row.get("analysis_mode") != clean_value:
            row["analysis_mode"] = clean_value
            fixed += 1
    return fixed


def repair_metadata_file(metadata_path: Path) -> int:
    original_limit = csv.field_size_limit()
    csv.field_size_limit(2**31 - 1)
    try:
        with metadata_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(existing) for existing in reader]
    finally:
        csv.field_size_limit(original_limit)

    if not fieldnames:
        return 0

    fixed = sanitize_analysis_mode_rows(rows)
    if fixed:
        with metadata_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return fixed


def _normalize_metadata_row(raw: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in raw.items():
        if key is None:
            continue
        out_key = _normalize_metadata_tt_key(str(key)) if isinstance(key, str) else key
        if out_key in normalized:
            if _metadata_value_is_empty(normalized[out_key]) and not _metadata_value_is_empty(value):
                normalized[out_key] = value
            continue
        normalized[out_key] = value
    return normalized


def _load_existing_rows(metadata_path: Path) -> Tuple[List[str], List[Dict[str, object]]]:
    with metadata_path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        existing_fields = _normalize_metadata_fieldnames(list(reader.fieldnames or []))
        existing_rows = [_normalize_metadata_row(dict(existing)) for existing in reader]
    return existing_fields, existing_rows


def _apply_preferred_field_order(
    fieldnames: List[str],
    preferred_fieldnames: Iterable[str] | None,
) -> List[str]:
    if preferred_fieldnames:
        seen = set(fieldnames)
        preferred = [name for name in preferred_fieldnames if name in seen]
        remainder = [name for name in fieldnames if name not in preferred]
        fieldnames = preferred + remainder
    identity_prefix = [name for name in TRACEABILITY_COLUMNS if name in fieldnames]
    if identity_prefix:
        fieldnames = identity_prefix + [name for name in fieldnames if name not in identity_prefix]
    if "data_purity_percentage" in fieldnames:
        fieldnames = [name for name in fieldnames if name != "data_purity_percentage"] + [
            "data_purity_percentage"
        ]
    return fieldnames


def _format_metadata_row(item: Dict[str, object], fieldnames: List[str]) -> Dict[str, object]:
    formatted: Dict[str, object] = {}
    for key in fieldnames:
        if (
            key in EVENTS_PER_SECOND_COLUMN_SET
            or key.startswith("events_per_second_")
            or key == "analysis_mode"
            or key.endswith("_count")
            or key.endswith("_rate_hz")
            or key.endswith("_fraction")
        ):
            value = item.get(key, 0)
            if _metadata_value_is_empty(value):
                formatted[key] = 0
            else:
                formatted[key] = value
            continue
        value = item.get(key, "")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            formatted[key] = ""
        elif isinstance(value, (list, dict, np.ndarray)):
            formatted[key] = str(value)
        else:
            formatted[key] = value
    return formatted


def _save_metadata_rewrite(
    metadata_path: Path,
    row: Dict[str, object],
    preferred_fieldnames: Iterable[str] | None,
    log_fn: Callable[..., None],
    drop_field_predicate: Callable[[str], bool] | None = None,
) -> None:
    rows: List[Dict[str, object]] = []
    fieldnames: List[str] = []
    row_filtered = {
        key: value
        for key, value in row.items()
        if not (
            drop_field_predicate
            and isinstance(key, str)
            and drop_field_predicate(key)
        )
    }

    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        try:
            fieldnames, existing_rows = _load_existing_rows(metadata_path)
        except csv.Error as exc:
            if "field larger than field limit" in str(exc).lower():
                fixed = repair_metadata_file(metadata_path)
                log_fn(
                    f"Detected oversized analysis_mode entries in {metadata_path}; normalized {fixed} row(s)."
                )
                try:
                    fieldnames, existing_rows = _load_existing_rows(metadata_path)
                except csv.Error as err:
                    raise RuntimeError(
                        f"Failed to repair metadata file {metadata_path} after detecting oversized fields."
                    ) from err
            else:
                raise
        if drop_field_predicate:
            existing_rows = [
                {
                    key: value
                    for key, value in existing.items()
                    if not (
                        isinstance(key, str)
                        and drop_field_predicate(key)
                    )
                }
                for existing in existing_rows
            ]
            fieldnames = [
                key
                for key in fieldnames
                if not (
                    isinstance(key, str)
                    and drop_field_predicate(key)
                )
            ]
        rows.extend(existing_rows)

    rows.append(row_filtered)
    fixed_during_rewrite = sanitize_analysis_mode_rows(rows)
    if fixed_during_rewrite:
        log_fn(f"Normalised analysis_mode in {fixed_during_rewrite} metadata row(s).")

    seen = set(fieldnames)
    for item in rows:
        for key in item.keys():
            if (
                drop_field_predicate
                and isinstance(key, str)
                and drop_field_predicate(key)
            ):
                continue
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    fieldnames = _apply_preferred_field_order(fieldnames, preferred_fieldnames)

    with metadata_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow(_format_metadata_row(item, fieldnames))


def _metadata_operation_dir(metadata_path: Path) -> Path:
    return metadata_path.parent / "OPERATION"


def _metadata_lock_path(metadata_path: Path) -> Path:
    return _metadata_operation_dir(metadata_path) / f"{metadata_path.name}.lock"


def _legacy_filename_base_index_path(metadata_path: Path) -> Path:
    return metadata_path.with_suffix(metadata_path.suffix + ".filename_base.index")


def _filename_base_index_path(metadata_path: Path) -> Path:
    return _metadata_operation_dir(metadata_path) / f"{metadata_path.name}.filename_base.index"


def _load_filename_base_index(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    values: set[str] = set()
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                values.add(value)
    return values


def _write_filename_base_index(index_path: Path, values: Iterable[str]) -> None:
    unique_values = sorted({value.strip() for value in values if str(value).strip()})
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        for value in unique_values:
            handle.write(f"{value}\n")


def _sync_filename_base_index_from_csv(
    metadata_path: Path,
    log_fn: Callable[..., None],
) -> set[str]:
    index_path = _filename_base_index_path(metadata_path)
    legacy_index_path = _legacy_filename_base_index_path(metadata_path)
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        for candidate in (index_path, legacy_index_path):
            if candidate.exists():
                candidate.unlink()
        return set()

    try:
        with metadata_path.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = list(reader.fieldnames or [])
            if "filename_base" not in fieldnames:
                for candidate in (index_path, legacy_index_path):
                    if candidate.exists():
                        candidate.unlink()
                return set()
            values = {
                str(raw).strip()
                for raw in (row.get("filename_base") for row in reader)
                if str(raw).strip()
            }
    except csv.Error as exc:
        if "field larger than field limit" in str(exc).lower():
            fixed = repair_metadata_file(metadata_path)
            log_fn(
                f"Detected oversized analysis_mode entries in {metadata_path}; normalized {fixed} row(s)."
            )
            return _sync_filename_base_index_from_csv(metadata_path, log_fn)
        raise

    _write_filename_base_index(index_path, values)
    return values


def _load_or_build_filename_base_index(
    metadata_path: Path,
    log_fn: Callable[..., None],
) -> Tuple[Path, set[str]]:
    index_path = _filename_base_index_path(metadata_path)
    legacy_index_path = _legacy_filename_base_index_path(metadata_path)
    if index_path.exists():
        try:
            return index_path, _load_filename_base_index(index_path)
        except OSError:
            # Fall back to rebuilding from the CSV below.
            pass

    if legacy_index_path.exists():
        try:
            legacy_values = _load_filename_base_index(legacy_index_path)
            _write_filename_base_index(index_path, legacy_values)
            try:
                legacy_index_path.unlink()
            except OSError:
                pass
            return index_path, legacy_values
        except OSError:
            # Fall back to rebuilding from CSV below.
            pass

    return index_path, _sync_filename_base_index_from_csv(metadata_path, log_fn)


def _compose_drop_field_predicate(
    metadata_path: Path,
    drop_field_predicate: Callable[[str], bool] | None,
) -> Callable[[str], bool] | None:
    """Return effective drop predicate, enforcing per-file schema constraints."""
    if metadata_path.name.endswith("_metadata_rate_histogram.csv"):
        def _rate_histogram_drop(column_name: str) -> bool:
            return not is_rate_histogram_file_column(column_name)

        if drop_field_predicate is None:
            return _rate_histogram_drop

        return lambda column_name: (
            drop_field_predicate(column_name) or _rate_histogram_drop(column_name)
        )

    return drop_field_predicate


def save_metadata(
    metadata_path: str | Path,
    row: Dict[str, object],
    preferred_fieldnames: Iterable[str] | None = None,
    log_fn: Callable[..., None] = builtins.print,
    drop_field_predicate: Callable[[str], bool] | None = None,
) -> Path:
    metadata_path = Path(metadata_path)
    effective_drop_field_predicate = _compose_drop_field_predicate(
        metadata_path,
        drop_field_predicate,
    )
    row_data = _normalize_metadata_row(dict(row))
    if effective_drop_field_predicate:
        row_data = {
            key: value
            for key, value in row_data.items()
            if not (
                isinstance(key, str)
                and effective_drop_field_predicate(key)
            )
        }
    fixed_new_row = sanitize_analysis_mode_rows([row_data])
    if fixed_new_row:
        log_fn(f"Normalised analysis_mode in {fixed_new_row} metadata row(s).")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _metadata_lock_path(metadata_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

        if not metadata_path.exists() or metadata_path.stat().st_size == 0:
            initial_fields = _apply_preferred_field_order(list(row_data.keys()), preferred_fieldnames)
            with metadata_path.open("w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=initial_fields)
                writer.writeheader()
                writer.writerow(_format_metadata_row(row_data, initial_fields))
            _sync_filename_base_index_from_csv(metadata_path, log_fn)
            return metadata_path

        try:
            with metadata_path.open("r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                raw_existing_fields = list(reader.fieldnames or [])
                existing_fields = _normalize_metadata_fieldnames(raw_existing_fields)
        except csv.Error as exc:
            if "field larger than field limit" in str(exc).lower():
                fixed = repair_metadata_file(metadata_path)
                log_fn(
                    f"Detected oversized analysis_mode entries in {metadata_path}; normalized {fixed} row(s)."
                )
                _save_metadata_rewrite(
                    metadata_path,
                    row_data,
                    preferred_fieldnames,
                    log_fn,
                    drop_field_predicate=effective_drop_field_predicate,
                )
                _sync_filename_base_index_from_csv(metadata_path, log_fn)
                return metadata_path
            raise

        if not existing_fields:
            _save_metadata_rewrite(
                metadata_path,
                row_data,
                preferred_fieldnames,
                log_fn,
                drop_field_predicate=effective_drop_field_predicate,
            )
            _sync_filename_base_index_from_csv(metadata_path, log_fn)
            return metadata_path

        if existing_fields != raw_existing_fields:
            _save_metadata_rewrite(
                metadata_path,
                row_data,
                preferred_fieldnames,
                log_fn,
                drop_field_predicate=effective_drop_field_predicate,
            )
            _sync_filename_base_index_from_csv(metadata_path, log_fn)
            return metadata_path

        if effective_drop_field_predicate and any(
            isinstance(name, str) and effective_drop_field_predicate(name) for name in existing_fields
        ):
            _save_metadata_rewrite(
                metadata_path,
                row_data,
                preferred_fieldnames,
                log_fn,
                drop_field_predicate=effective_drop_field_predicate,
            )
            _sync_filename_base_index_from_csv(metadata_path, log_fn)
            return metadata_path

        existing_field_set = set(existing_fields)
        if any(key not in existing_field_set for key in row_data.keys()):
            _save_metadata_rewrite(
                metadata_path,
                row_data,
                preferred_fieldnames,
                log_fn,
                drop_field_predicate=effective_drop_field_predicate,
            )
            _sync_filename_base_index_from_csv(metadata_path, log_fn)
            return metadata_path

        basename = str(row_data.get("filename_base", "")).strip()
        index_path: Path | None = None
        index_values: set[str] = set()
        duplicate_basename = False
        if basename:
            index_path, index_values = _load_or_build_filename_base_index(metadata_path, log_fn)
            duplicate_basename = basename in index_values
            if duplicate_basename:
                log_fn(
                    f"[metadata] filename_base={basename} already present in {metadata_path.name}; appending new row."
                )

        with metadata_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=existing_fields)
            writer.writerow(_format_metadata_row(row_data, existing_fields))

        if basename and index_path is not None and not duplicate_basename:
            with index_path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(f"{basename}\n")

    return metadata_path


def build_events_per_second_metadata(
    df: pd.DataFrame,
    time_columns: Tuple[str, ...] = ("datetime", "Time"),
) -> Dict[str, object]:
    metadata = {column: 0 for column in EVENTS_PER_SECOND_COLUMNS}
    if df is None or df.empty:
        return metadata

    time_col = next((col for col in time_columns if col in df.columns), None)
    if time_col is not None:
        times = pd.to_datetime(df[time_col], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        times = pd.Series(df.index)
    else:
        return metadata

    times = pd.Series(times).dropna()
    if times.empty:
        return metadata

    times = times.dt.floor("s")
    start_time = times.min()
    end_time = times.max()
    if pd.isna(start_time) or pd.isna(end_time):
        return metadata

    full_range = pd.date_range(start=start_time, end=end_time, freq="s")
    events_per_second = times.value_counts().reindex(full_range, fill_value=0).sort_index()

    total_seconds = int(events_per_second.size)
    total_events = int(events_per_second.sum())
    metadata["events_per_second_total_seconds"] = total_seconds
    metadata["events_per_second_global_rate"] = round(total_events / total_seconds, 6) if total_seconds > 0 else 0

    hist_counts = events_per_second.value_counts()
    for events_count, seconds_count in hist_counts.items():
        events_count_int = int(events_count)
        if 0 <= events_count_int <= EVENTS_PER_SECOND_MAX:
            metadata[f"events_per_second_{events_count_int}_count"] = int(seconds_count)

    return metadata


def add_normalized_count_metadata(
    metadata: Dict[str, object],
    denominator_seconds: object,
    log_fn: Callable[..., None] = builtins.print,
) -> None:
    try:
        denom = float(denominator_seconds) if denominator_seconds is not None else 0.0
    except (TypeError, ValueError):
        denom = 0.0

    metadata["count_rate_denominator_seconds"] = int(denom) if denom > 0 else 0

    if denom <= 0:
        log_fn("[count-rates] Denominator seconds is 0; skipping normalized count columns.")
        return

    for key, value in list(metadata.items()):
        if not isinstance(key, str):
            continue

        is_count = key.endswith("_count")
        is_entries = _is_entries_metric_key(key) and not key.endswith("_rate_hz")
        if not (is_count or is_entries):
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(num):
            continue

        if is_count:
            out_key = key[: -len("_count")] + "_rate_hz"
        else:
            out_key = key + "_rate_hz"

        metadata[out_key] = round(num / denom, 6)


def _normalize_tt_prefix(prefix: object) -> str:
    text = str(prefix).strip()
    if text.endswith("_"):
        text = text[:-1]
    return text


def _extract_tt_rate_components(column_name: str) -> tuple[str, str] | None:
    match = TT_RATE_COLUMN_RE.match(column_name)
    if match is None:
        return None
    prefix = _normalize_tt_prefix(match.group("prefix"))
    label = normalize_tt_label(match.group("label"), default="")
    if not prefix or not label:
        return None
    return prefix, label


def compute_global_rate_from_tt_rates(
    metadata: Dict[str, object],
    preferred_prefixes: Iterable[str] | None = None,
) -> tuple[float | None, str | None]:
    by_prefix: dict[str, dict[str, float]] = {}

    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        parsed = _extract_tt_rate_components(key)
        if parsed is None:
            continue
        prefix, tt_label = parsed
        if tt_label not in CANONICAL_TT_LABELS:
            # Ignore transition-matrix keys like clean_to_cal_tt_12_34_rate_hz.
            continue
        try:
            rate_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rate_value):
            continue
        by_prefix.setdefault(prefix, {})[tt_label] = rate_value

    if not by_prefix:
        return None, None

    selected_prefix: str | None = None
    if preferred_prefixes is not None:
        for raw_prefix in preferred_prefixes:
            normalized_prefix = _normalize_tt_prefix(raw_prefix)
            if normalized_prefix in by_prefix and by_prefix[normalized_prefix]:
                selected_prefix = normalized_prefix
                break

    if selected_prefix is None:
        selected_prefix = min(
            by_prefix.keys(),
            key=lambda prefix: (-len(by_prefix[prefix]), prefix),
        )

    total_rate = float(sum(by_prefix[selected_prefix].values()))
    return round(total_rate, 6), selected_prefix


def set_global_rate_from_tt_rates(
    metadata: Dict[str, object],
    preferred_prefixes: Iterable[str] | None = None,
    global_rate_key: str = "events_per_second_global_rate",
    log_fn: Callable[..., None] | None = None,
) -> bool:
    total_rate, selected_prefix = compute_global_rate_from_tt_rates(
        metadata,
        preferred_prefixes=preferred_prefixes,
    )
    if total_rate is None or selected_prefix is None:
        return False

    metadata[global_rate_key] = round(float(total_rate), 6)
    if log_fn is not None:
        log_fn(
            f"[global-rate] {global_rate_key} set as sum of {selected_prefix}_*_rate_hz."
        )
    return True


def load_itineraries_from_file(
    file_path: Path,
    required: bool = True,
    log_fn: Callable[..., None] = builtins.print,
    echo_entries: bool = True,
    header_suffix: str = ":",
) -> list[list[str]]:
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"Cannot find itineraries file: {file_path}")
        return []

    itineraries: list[list[str]] = []
    with file_path.open("r", encoding="utf-8") as itinerary_file:
        if log_fn is not None:
            log_fn(f"Loading itineraries from {file_path}{header_suffix}")
        for raw_line in itinerary_file:
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            segments = [segment.strip() for segment in stripped_line.split(",") if segment.strip()]
            if segments:
                itineraries.append(segments)
                if log_fn is not None and echo_entries:
                    log_fn(segments)

    if not itineraries and required:
        raise ValueError(f"Itineraries file {file_path} is empty.")
    return itineraries


def write_itineraries_to_file(file_path: Path, itineraries: Iterable[Iterable[str]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    unique_itineraries: dict[tuple[str, ...], None] = {}

    for itinerary in itineraries:
        itinerary_tuple = tuple(itinerary)
        if not itinerary_tuple:
            continue
        unique_itineraries.setdefault(itinerary_tuple, None)

    with file_path.open("w", encoding="utf-8") as itinerary_file:
        for itinerary_tuple in unique_itineraries:
            itinerary_file.write(",".join(itinerary_tuple) + "\n")


def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2


def collect_columns(columns: Iterable[str], pattern) -> list[str]:
    return [column for column in columns if pattern.match(column)]
