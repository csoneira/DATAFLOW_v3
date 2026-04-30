#!/usr/bin/env python3
from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "OUTPUTS"
FILES_DIR = OUTPUT_DIR / "FILES"
PLOTS_DIR = OUTPUT_DIR / "PLOTS"
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.json"

CANONICAL_EFF_COLUMNS = ["emp_eff_1", "emp_eff_2", "emp_eff_3", "emp_eff_4"]
CANONICAL_Z_COLUMNS = ["z_pos_1", "z_pos_2", "z_pos_3", "z_pos_4"]
TT_TWO_PLANE_LABELS = ["12", "13", "14", "23", "24", "34"]
TT_THREE_PLANE_LABELS = ["123", "124", "134", "234"]
TT_FOUR_PLANE_LABEL = "1234"
TT_RATE_LABELS = TT_TWO_PLANE_LABELS + TT_THREE_PLANE_LABELS + [TT_FOUR_PLANE_LABEL]
TT_THREE_TO_FOUR_MISSING_BY_PLANE = {1: "234", 2: "134", 3: "124", 4: "123"}
TASK_FINAL_STAGE_PREFIX = {1: "clean_tt", 2: "cal_tt", 3: "list_tt", 4: "fit_tt", 5: "post_tt"}
TRIGGER_METADATA_SOURCE_ALIASES = {
    "trigger": "trigger_type",
    "trigger_type": "trigger_type",
    "robust": "robust_efficiency",
    "robust_eff": "robust_efficiency",
    "robust_efficiency": "robust_efficiency",
}
TRIGGER_RATE_FAMILY_TO_COLUMN = {
    "total": "total_rate_hz",
    "four_plane": "four_plane_rate_hz",
    "three_plane": "three_plane_rate_hz",
    "two_plane": "two_plane_rate_hz",
    "three_and_four_plane": "three_and_four_plane_rate_hz",
    "two_and_three_plane": "two_and_three_plane_rate_hz",
}
TRIGGER_RATE_FAMILY_ALIASES = {
    "global": "total",
    "all": "total",
    "total_rate": "total",
    "rate_total": "total",
    "total_rate_hz": "total",
    "rate_total_hz": "total",
    "1234": "four_plane",
    "4": "four_plane",
    "four_plane_rate": "four_plane",
    "rate_1234": "four_plane",
    "four_plane_rate_hz": "four_plane",
    "rate_1234_hz": "four_plane",
    "3": "three_plane",
    "three_plane_rate": "three_plane",
    "three_plane_rate_hz": "three_plane",
    "2": "two_plane",
    "two_plane_rate": "two_plane",
    "two_plane_rate_hz": "two_plane",
    "34": "three_and_four_plane",
    "three_and_four_plane_rate": "three_and_four_plane",
    "three_and_four_plane_rate_hz": "three_and_four_plane",
    "23": "two_and_three_plane",
    "two_and_three_plane_rate": "two_and_three_plane",
    "two_and_three_plane_rate_hz": "two_and_three_plane",
}
ROBUST_RATE_FAMILY_TO_COLUMN = {
    "total": "total_rate_hz",
    "four_plane": "four_plane_rate_hz",
    "four_plane_robust_hz": "four_plane_robust_hz",
}
ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN = {
    "total": "rate_total_hz",
    "four_plane": "rate_1234_hz",
    "four_plane_robust_hz": "four_plane_robust_hz",
}
ROBUST_RATE_FAMILY_TO_COUNT_COLUMN = {
    "total": "total_count",
    "four_plane": "four_plane_count",
    "four_plane_robust_hz": "four_plane_robust_count",
}
ROBUST_RATE_FAMILY_ALIASES = {
    "global": "total",
    "all": "total",
    "total_rate": "total",
    "rate_total": "total",
    "total_rate_hz": "total",
    "rate_total_hz": "total",
    "1234": "four_plane",
    "4": "four_plane",
    "four_plane_rate": "four_plane",
    "rate_1234": "four_plane",
    "four_plane_rate_hz": "four_plane",
    "rate_1234_hz": "four_plane",
    "four_plane_robust": "four_plane_robust_hz",
    "four_plane_robust_rate": "four_plane_robust_hz",
    "four_plane_robust_rate_hz": "four_plane_robust_hz",
    "four_plane_robust_hz": "four_plane_robust_hz",
}
ROBUST_OPTIONAL_COUNT_COLUMNS = {
    "total": ["total_count", "count_total", "rate_total_count"],
    "four_plane": ["four_plane_count", "count_1234", "rate_1234_count"],
    "four_plane_robust_hz": ["four_plane_robust_count", "four_plane_robust_count_union", "count_four_plane_robust"],
}
ROBUST_DIAGNOSTIC_COLUMNS = [
    "four_plane_robust_count_union",
    "four_plane_robust_count_intersection",
    "four_plane_robust_hz_union",
    "four_plane_robust_hz_intersection",
]
ROBUST_EFFICIENCY_VARIANT_TO_SUFFIX = {
    "default": "",
    "plateau": "_plateau",
    "overall": "_overall",
    "median_x": "_median_x",
}
ROBUST_EFFICIENCY_VARIANT_ALIASES = {
    "": "default",
    "default": "default",
    "base": "default",
    "plain": "default",
    "nominal": "default",
    "eff": "default",
    "plateau": "plateau",
    "overall": "overall",
    "median_x": "median_x",
    "median": "median_x",
    "x_median": "median_x",
}
DEFAULT_ROBUST_EFFICIENCY_TASK_ID = 4
DERIVED_RATE_COLUMNS = [
    "two_plane_rate_hz",
    "three_plane_rate_hz",
    "four_plane_rate_hz",
    "four_plane_robust_hz",
    "three_and_four_plane_rate_hz",
    "two_and_three_plane_rate_hz",
    "total_rate_hz",
]
DERIVED_COUNT_COLUMNS = [
    "two_plane_count",
    "three_plane_count",
    "four_plane_count",
    "four_plane_robust_count",
    "three_and_four_plane_count",
    "two_and_three_plane_count",
    "total_count",
]
STEP1_OUTPUT_COLUMNS = (
    CANONICAL_Z_COLUMNS
    + CANONICAL_EFF_COLUMNS
    + [f"eff_empirical_{idx}" for idx in range(1, 5)]
    + DERIVED_RATE_COLUMNS
    + DERIVED_COUNT_COLUMNS
    + ["count_rate_denominator_seconds", "selected_rate_hz", "selected_rate_count", "rate_hz", "sim_flux_cm2_min", "num_events"]
)


def ensure_output_dirs() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_path(config: dict[str, Any], raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path(config["_config_dir"]) / path).resolve()


def cfg_path(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        value = value[key]
    return resolve_path(config, value)


def get_rate_column_name(config: dict[str, Any]) -> str:
    trigger_config = config.get("trigger_type_selection", {})
    if isinstance(trigger_config, dict) and trigger_config:
        selection = get_trigger_type_selection(config)
        return str(selection["selected_source_rate_column"])
    columns = config.get("columns", {})
    if not isinstance(columns, dict):
        raise ValueError("Config is missing the 'columns' object.")
    raw_value = columns.get("rate", columns.get("global_rate"))
    if raw_value in (None, "", "null", "None"):
        raise ValueError("Config must define columns.rate or columns.global_rate.")
    return str(raw_value)


def _normalize_optional_int(value: Any) -> int | None:
    if value in (None, "", "null", "None"):
        return None
    return int(value)


def get_trigger_type_selection(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("trigger_type_selection", {})
    if not isinstance(raw, dict):
        raw = {}

    metadata_source_raw = str(raw.get("metadata_source", "trigger_type")).strip().lower()
    metadata_source = TRIGGER_METADATA_SOURCE_ALIASES.get(metadata_source_raw, metadata_source_raw)
    if metadata_source == "robust_efficiency":
        efficiency_variant_raw = str(raw.get("robust_efficiency_variant", "default")).strip().lower()
        efficiency_variant = ROBUST_EFFICIENCY_VARIANT_ALIASES.get(efficiency_variant_raw, efficiency_variant_raw)
        if efficiency_variant not in ROBUST_EFFICIENCY_VARIANT_TO_SUFFIX:
            raise ValueError(
                "Unsupported trigger_type_selection.robust_efficiency_variant: "
                f"{raw.get('robust_efficiency_variant')!r}. Supported values are: "
                + ", ".join(sorted(ROBUST_EFFICIENCY_VARIANT_TO_SUFFIX))
            )
        rate_family_raw = str(raw.get("rate_family", "1234")).strip().lower()
        rate_family = ROBUST_RATE_FAMILY_ALIASES.get(rate_family_raw, rate_family_raw)
        if rate_family not in ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN:
            raise ValueError(
                "Unsupported trigger_type_selection.rate_family for robust_efficiency metadata: "
                f"{raw.get('rate_family')!r}. Supported values are: "
                + ", ".join(sorted(ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN))
            )
        selected_source_rate_column = ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN[rate_family]
        return {
            "metadata_source": metadata_source,
            "source_name": "robust_efficiency",
            "task_id": DEFAULT_ROBUST_EFFICIENCY_TASK_ID,
            "metadata_task_id": DEFAULT_ROBUST_EFFICIENCY_TASK_ID,
            "stage_prefix": None,
            "offender_threshold": None,
            "rate_family": rate_family,
            "rate_family_column": ROBUST_RATE_FAMILY_TO_COLUMN[rate_family],
            "selected_source_rate_column": selected_source_rate_column,
            "robust_efficiency_variant": efficiency_variant,
        }
    if metadata_source != "trigger_type":
        raise ValueError(
            "Unsupported trigger_type_selection.metadata_source: "
            f"{raw.get('metadata_source')!r}. Supported values are: trigger_type, robust_efficiency"
        )

    task_id = int(raw.get("task_id", 5))
    stage_prefix_raw = raw.get("stage_prefix")
    if stage_prefix_raw in (None, "", "null", "None"):
        stage_prefix = TASK_FINAL_STAGE_PREFIX.get(task_id)
        if stage_prefix is None:
            raise ValueError(f"Unsupported trigger_type_selection.task_id: {task_id}")
    else:
        stage_prefix = str(stage_prefix_raw).strip()

    offender_threshold = _normalize_optional_int(raw.get("offender_threshold"))
    rate_family_raw = str(raw.get("rate_family", "total")).strip().lower()
    rate_family = TRIGGER_RATE_FAMILY_ALIASES.get(rate_family_raw, rate_family_raw)
    if rate_family not in TRIGGER_RATE_FAMILY_TO_COLUMN:
        raise ValueError(
            "Unsupported trigger_type_selection.rate_family: "
            f"{raw.get('rate_family')!r}. Supported values are: "
            + ", ".join(sorted(TRIGGER_RATE_FAMILY_TO_COLUMN))
        )

    selected_source_rate_column = format_selected_rate_name(
        stage_prefix=stage_prefix,
        rate_family_column=TRIGGER_RATE_FAMILY_TO_COLUMN[rate_family],
        offender_threshold=offender_threshold,
    )
    return {
        "metadata_source": metadata_source,
        "source_name": "trigger_type",
        "task_id": task_id,
        "metadata_task_id": task_id,
        "stage_prefix": stage_prefix,
        "offender_threshold": offender_threshold,
        "rate_family": rate_family,
        "rate_family_column": TRIGGER_RATE_FAMILY_TO_COLUMN[rate_family],
        "selected_source_rate_column": selected_source_rate_column,
    }


def format_selected_rate_name(
    *,
    stage_prefix: str | None,
    rate_family_column: str,
    offender_threshold: int | None,
    metadata_source: str = "trigger_type",
) -> str:
    if str(metadata_source).strip().lower() == "robust_efficiency":
        return str(rate_family_column)
    if stage_prefix in (None, "", "null", "None"):
        return str(rate_family_column)
    if offender_threshold is None:
        return f"{stage_prefix}_{rate_family_column}"
    return f"{stage_prefix}_total_offenders_le_{int(offender_threshold)}_{rate_family_column}"


def trigger_rate_source_column(stage_prefix: str, tt_label: str, offender_threshold: int | None) -> str:
    if offender_threshold is None:
        return f"{stage_prefix}_{tt_label}_rate_hz"
    return f"{stage_prefix}_total_offenders_le_{int(offender_threshold)}_{tt_label}_rate_hz"


def trigger_count_source_column(stage_prefix: str, tt_label: str, offender_threshold: int | None) -> str | None:
    if offender_threshold is None:
        return None
    return f"{stage_prefix}_total_offenders_le_{int(offender_threshold)}_{tt_label}_count"


def required_trigger_rate_columns(stage_prefix: str, offender_threshold: int | None) -> list[str]:
    return [trigger_rate_source_column(stage_prefix, label, offender_threshold) for label in TT_RATE_LABELS]


def _safe_sum_series(dataframe: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=dataframe.index, dtype=float)
    numeric = dataframe[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=1, min_count=1)


def _resolved_count_series(
    dataframe: pd.DataFrame,
    *,
    rate_column: str,
    count_column: str | None,
    denominator_column: str | None = "count_rate_denominator_seconds",
) -> pd.Series:
    if count_column and count_column in dataframe.columns:
        return pd.to_numeric(dataframe[count_column], errors="coerce")
    rate_series = pd.to_numeric(dataframe.get(rate_column), errors="coerce")
    if denominator_column is None or denominator_column not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype=float)
    denominator = pd.to_numeric(dataframe[denominator_column], errors="coerce")
    return rate_series * denominator


def _missing_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in dataframe.columns]


def _optional_numeric_series(dataframe: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce")


def _first_present_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def _derive_robust_efficiency_features(
    dataframe: pd.DataFrame,
    selection: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    efficiency_variant = str(selection.get("robust_efficiency_variant", "default"))
    efficiency_suffix = ROBUST_EFFICIENCY_VARIANT_TO_SUFFIX[efficiency_variant]
    required_columns = [
        f"eff{idx}{efficiency_suffix}" for idx in range(1, 5)
    ] + [
        ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["total"],
        ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["four_plane"],
    ]
    selected_source_rate_column = str(selection["selected_source_rate_column"])
    if selected_source_rate_column not in required_columns:
        required_columns.append(selected_source_rate_column)
    missing = _missing_columns(dataframe, required_columns)
    if missing:
        raise KeyError(
            "Missing robust-efficiency columns required by trigger_type_selection: "
            + ", ".join(sorted(missing))
        )

    total_count_source = _first_present_column(dataframe, ROBUST_OPTIONAL_COUNT_COLUMNS["total"])
    four_plane_count_source = _first_present_column(dataframe, ROBUST_OPTIONAL_COUNT_COLUMNS["four_plane"])
    four_plane_robust_count_source = _first_present_column(
        dataframe,
        ROBUST_OPTIONAL_COUNT_COLUMNS["four_plane_robust_hz"],
    )

    out = dataframe.copy()
    out["count_rate_denominator_seconds"] = (
        pd.to_numeric(out["count_rate_denominator_seconds"], errors="coerce")
        if "count_rate_denominator_seconds" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    out["two_plane_rate_hz"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["three_plane_rate_hz"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["four_plane_rate_hz"] = pd.to_numeric(out[ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["four_plane"]], errors="coerce")
    out["four_plane_robust_hz"] = (
        pd.to_numeric(out[ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["four_plane_robust_hz"]], errors="coerce")
        if ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["four_plane_robust_hz"] in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    out["three_and_four_plane_rate_hz"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["two_and_three_plane_rate_hz"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["total_rate_hz"] = pd.to_numeric(out[ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["total"]], errors="coerce")

    out["two_plane_count"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["three_plane_count"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["four_plane_count"] = _resolved_count_series(
        out,
        rate_column=ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["four_plane"],
        count_column=four_plane_count_source,
    )
    out["four_plane_robust_count"] = _resolved_count_series(
        out,
        rate_column=ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["four_plane_robust_hz"],
        count_column=four_plane_robust_count_source,
    )
    out["three_and_four_plane_count"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["two_and_three_plane_count"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["total_count"] = _resolved_count_series(
        out,
        rate_column=ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["total"],
        count_column=total_count_source,
    )
    for column_name in ROBUST_DIAGNOSTIC_COLUMNS:
        if column_name in out.columns:
            out[column_name] = pd.to_numeric(out[column_name], errors="coerce")

    efficiency_source_columns: dict[str, str] = {}
    for plane_idx in range(1, 5):
        source_column = f"eff{plane_idx}{efficiency_suffix}"
        out[f"eff_empirical_{plane_idx}"] = pd.to_numeric(out[source_column], errors="coerce")
        efficiency_source_columns[f"plane_{plane_idx}"] = source_column

    selected_rate_column = str(selection["rate_family_column"])
    selected_count_column = ROBUST_RATE_FAMILY_TO_COUNT_COLUMN[str(selection["rate_family"])]
    out["selected_rate_hz"] = pd.to_numeric(out[selected_rate_column], errors="coerce")
    out["selected_rate_count"] = pd.to_numeric(out[selected_count_column], errors="coerce")
    out["rate_hz"] = out["selected_rate_hz"]

    metadata = {
        "metadata_source": selection["metadata_source"],
        "source_name": selection["source_name"],
        "task_id": int(selection["task_id"]),
        "stage_prefix": None,
        "requested_stage_prefix": None,
        "used_stage_prefix": None,
        "stage_prefix_fallback_used": False,
        "requested_offender_threshold": None,
        "used_offender_threshold": None,
        "rate_family": selection["rate_family"],
        "rate_family_column": selected_rate_column,
        "selected_source_rate_column": selection["selected_source_rate_column"],
        "robust_efficiency_variant": efficiency_variant,
        "robust_efficiency_source_columns": efficiency_source_columns,
        "plain_column_fallback_used": False,
        "source_rate_columns": {
            "four_plane": ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["four_plane"],
            "four_plane_robust_hz": ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["four_plane_robust_hz"],
            "total": ROBUST_RATE_FAMILY_TO_SOURCE_COLUMN["total"],
        },
        "source_count_columns": {
            "four_plane": four_plane_count_source,
            "four_plane_robust_hz": four_plane_robust_count_source,
            "total": total_count_source,
        },
    }
    return out, metadata


def derive_trigger_rate_features(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    *,
    allow_plain_fallback: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection = get_trigger_type_selection(config)
    if str(selection.get("metadata_source", "trigger_type")) == "robust_efficiency":
        return _derive_robust_efficiency_features(dataframe, selection)

    requested_stage_prefix = str(selection["stage_prefix"])
    requested_threshold = selection["offender_threshold"]

    def _build_source_columns(stage_prefix: str, offender_threshold: int | None) -> dict[str, str]:
        return {
            label: trigger_rate_source_column(stage_prefix, label, offender_threshold)
            for label in TT_RATE_LABELS
        }

    def _missing_source_columns(source_columns: dict[str, str]) -> list[str]:
        return [column for column in source_columns.values() if column not in dataframe.columns]

    used_stage_prefix = requested_stage_prefix
    used_threshold = requested_threshold
    source_columns = _build_source_columns(used_stage_prefix, used_threshold)
    missing = _missing_source_columns(source_columns)

    if missing and allow_plain_fallback:
        fallback_stage_candidates = [
            TASK_FINAL_STAGE_PREFIX[task_id]
            for task_id in sorted(TASK_FINAL_STAGE_PREFIX.keys(), reverse=True)
            if TASK_FINAL_STAGE_PREFIX[task_id] != requested_stage_prefix
        ]

        # Fallback priority: plain columns on requested stage, then alternate stages
        # (prefer later pipeline stages first), then plain columns on alternates.
        fallback_candidates: list[tuple[str, int | None]] = []
        if requested_threshold is not None:
            fallback_candidates.append((requested_stage_prefix, None))
        fallback_candidates.extend((stage_prefix, requested_threshold) for stage_prefix in fallback_stage_candidates)
        if requested_threshold is not None:
            fallback_candidates.extend((stage_prefix, None) for stage_prefix in fallback_stage_candidates)

        for candidate_stage_prefix, candidate_threshold in fallback_candidates:
            candidate_columns = _build_source_columns(candidate_stage_prefix, candidate_threshold)
            candidate_missing = _missing_source_columns(candidate_columns)
            if candidate_missing:
                continue
            source_columns = candidate_columns
            used_stage_prefix = candidate_stage_prefix
            used_threshold = candidate_threshold
            missing = []
            break

    if missing:
        available_stage_prefixes = sorted(
            {
                column_name[: -len(f"_{tt_label}_rate_hz")]
                for column_name in dataframe.columns.astype(str)
                for tt_label in TT_RATE_LABELS
                if column_name.endswith(f"_{tt_label}_rate_hz")
            }
        )
        available_stage_hint = (
            "none detected"
            if not available_stage_prefixes
            else ", ".join(available_stage_prefixes)
        )
        raise KeyError(
            "Missing trigger-type rate columns required by trigger_type_selection: "
            + ", ".join(sorted(missing))
            + f". Requested stage_prefix={requested_stage_prefix!r}, "
            + f"offender_threshold={requested_threshold!r}. "
            + f"Available stage prefixes in dataframe: {available_stage_hint}."
        )

    count_columns = {
        label: trigger_count_source_column(used_stage_prefix, label, used_threshold)
        for label in TT_RATE_LABELS
    }

    out = dataframe.copy()
    out["count_rate_denominator_seconds"] = (
        pd.to_numeric(out["count_rate_denominator_seconds"], errors="coerce")
        if "count_rate_denominator_seconds" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    component_rates = {
        label: pd.to_numeric(out[source_columns[label]], errors="coerce")
        for label in TT_RATE_LABELS
    }
    component_counts = {
        label: _resolved_count_series(
            out,
            rate_column=source_columns[label],
            count_column=count_columns[label],
        )
        for label in TT_RATE_LABELS
    }

    out["two_plane_rate_hz"] = _safe_sum_series(out.assign(**{f"__{k}": v for k, v in component_rates.items()}), [f"__{label}" for label in TT_TWO_PLANE_LABELS])
    out["three_plane_rate_hz"] = _safe_sum_series(out.assign(**{f"__{k}": v for k, v in component_rates.items()}), [f"__{label}" for label in TT_THREE_PLANE_LABELS])
    out["four_plane_rate_hz"] = component_rates[TT_FOUR_PLANE_LABEL]
    out["four_plane_robust_hz"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["three_and_four_plane_rate_hz"] = out["three_plane_rate_hz"] + out["four_plane_rate_hz"]
    out["two_and_three_plane_rate_hz"] = out["two_plane_rate_hz"] + out["three_plane_rate_hz"]
    out["total_rate_hz"] = out["two_plane_rate_hz"] + out["three_plane_rate_hz"] + out["four_plane_rate_hz"]

    out["two_plane_count"] = _safe_sum_series(out.assign(**{f"__{k}": v for k, v in component_counts.items()}), [f"__{label}" for label in TT_TWO_PLANE_LABELS])
    out["three_plane_count"] = _safe_sum_series(out.assign(**{f"__{k}": v for k, v in component_counts.items()}), [f"__{label}" for label in TT_THREE_PLANE_LABELS])
    out["four_plane_count"] = component_counts[TT_FOUR_PLANE_LABEL]
    out["four_plane_robust_count"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["three_and_four_plane_count"] = out["three_plane_count"] + out["four_plane_count"]
    out["two_and_three_plane_count"] = out["two_plane_count"] + out["three_plane_count"]
    out["total_count"] = out["two_plane_count"] + out["three_plane_count"] + out["four_plane_count"]

    valid_denominator = out["four_plane_rate_hz"].where(pd.to_numeric(out["four_plane_rate_hz"], errors="coerce") > 0.0)
    for plane_idx in range(1, 5):
        numerator_label = TT_THREE_TO_FOUR_MISSING_BY_PLANE[plane_idx]
        out[f"eff_empirical_{plane_idx}"] = 1 - component_rates[numerator_label] / valid_denominator

    selected_rate_column = selection["rate_family_column"]
    selected_count_column = selected_rate_column.replace("_rate_hz", "_count")
    out["selected_rate_hz"] = out[selected_rate_column]
    out["selected_rate_count"] = out[selected_count_column]
    out["rate_hz"] = out["selected_rate_hz"]

    metadata = {
        "metadata_source": selection["metadata_source"],
        "source_name": selection["source_name"],
        "task_id": int(selection["task_id"]),
        "stage_prefix": used_stage_prefix,
        "requested_stage_prefix": requested_stage_prefix,
        "used_stage_prefix": used_stage_prefix,
        "stage_prefix_fallback_used": bool(used_stage_prefix != requested_stage_prefix),
        "requested_offender_threshold": requested_threshold,
        "used_offender_threshold": used_threshold,
        "rate_family": selection["rate_family"],
        "rate_family_column": selected_rate_column,
        "selected_source_rate_column": selection["selected_source_rate_column"],
        "plain_column_fallback_used": bool(requested_threshold is not None and used_threshold is None),
        "source_rate_columns": source_columns,
        "source_count_columns": count_columns,
    }
    return out, metadata


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonicalize_trigger_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (list, dict, int, float, bool)):
        return value

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = text

    if isinstance(parsed, tuple):
        return list(parsed)
    return parsed


def choose_z_vector(
    dataframe: pd.DataFrame,
    z_columns: list[str],
    configured_vector: list[float] | None,
) -> tuple[float, float, float, float]:
    if configured_vector:
        if len(configured_vector) != len(z_columns):
            raise ValueError(
                f"Configured z_position_vector has length {len(configured_vector)}; "
                f"expected {len(z_columns)}."
            )
        return tuple(float(value) for value in configured_vector)

    counts = (
        dataframe.groupby(z_columns, dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["row_count"] + z_columns, ascending=[False] + [True] * len(z_columns))
        .reset_index(drop=True)
    )
    if counts.empty:
        raise ValueError("Could not determine a z-position vector from an empty dataframe.")
    row = counts.iloc[0]
    return tuple(float(row[col]) for col in z_columns)


def filter_by_z_vector(
    dataframe: pd.DataFrame,
    z_columns: list[str],
    z_vector: tuple[float, float, float, float],
) -> pd.DataFrame:
    mask = np.ones(len(dataframe), dtype=bool)
    for column, value in zip(z_columns, z_vector):
        mask &= np.isclose(dataframe[column].astype(float), float(value))
    return dataframe.loc[mask].copy()


def q25(series: pd.Series) -> float:
    return float(series.quantile(0.25))


def q75(series: pd.Series) -> float:
    return float(series.quantile(0.75))


def quantize_efficiency_series(series: pd.Series, bin_width: float | None) -> pd.Series:
    values = series.astype(float).to_numpy()
    if bin_width is None or float(bin_width) <= 0:
        return pd.Series(np.round(values, 6), index=series.index)
    width = float(bin_width)
    binned = np.round(values / width) * width
    binned = np.clip(binned, 0.0, 1.0)
    return pd.Series(np.round(binned, 6), index=series.index)


def assign_efficiency_bins(
    dataframe: pd.DataFrame,
    eff_columns: list[str],
    bin_width: float | None,
    *,
    suffix: str = "_bin",
) -> pd.DataFrame:
    out = dataframe.copy()
    for column in eff_columns:
        out[f"{column}{suffix}"] = quantize_efficiency_series(out[column], bin_width)
    return out


def assign_flux_bins(series: pd.Series, bin_count: int) -> tuple[pd.Series, np.ndarray]:
    codes, edges = pd.cut(
        series.astype(float),
        bins=int(bin_count),
        labels=False,
        retbins=True,
        include_lowest=True,
        duplicates="drop",
    )
    return pd.Series(codes, index=series.index, dtype="Int64"), np.asarray(edges, dtype=float)


def choose_reference_row(
    dataframe: pd.DataFrame,
    eff_columns: list[str],
) -> dict[str, Any]:
    work = dataframe.copy()
    work["distance_to_perfect"] = np.sqrt(
        np.sum((1.0 - work[eff_columns].astype(float).to_numpy()) ** 2, axis=1)
    )
    work["eff_sum"] = work[eff_columns].astype(float).sum(axis=1)
    work = work.sort_values(
        ["distance_to_perfect", "eff_sum", "n_flux_bins", "support_rows"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    if work.empty:
        raise ValueError("No efficiency bins available for reference selection.")
    return work.iloc[0].to_dict()


def write_ascii_lut(
    path: str | Path,
    z_vector: tuple[float, float, float, float],
    lut_dataframe: pd.DataFrame,
    *,
    trigger: Any = None,
    rate_column_name: str | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# z_positions: "
            + ", ".join(f"{float(value):.6f}" for value in z_vector)
            + "\n"
        )
        trigger_value = canonicalize_trigger_value(trigger)
        if trigger_value is None:
            trigger_text = "null"
        elif isinstance(trigger_value, str):
            trigger_text = trigger_value
        else:
            trigger_text = json.dumps(trigger_value)
        handle.write(f"# trigger: {trigger_text}\n")
        rate_text = "null" if rate_column_name in (None, "", "null", "None") else str(rate_column_name)
        handle.write(f"# rate_column: {rate_text}\n")
        lut_dataframe.to_csv(handle, sep=" ", index=False, float_format="%.6f")


def read_ascii_lut(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    in_path = Path(path)
    comments: list[str] = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            comments.append(line.rstrip("\n"))
    dataframe = pd.read_csv(in_path, comment="#", sep=r"\s+", engine="python")
    return dataframe, comments


def apply_lut_fallback_matches(
    dataframe: pd.DataFrame,
    lut_dataframe: pd.DataFrame,
    *,
    query_columns: list[str],
    raw_columns: list[str] | None = None,
    match_mode: str = "nearest",
    interpolation_k: int | None = None,
    interpolation_power: float = 2.0,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    if lut_dataframe.empty:
        raise ValueError(
            "The LUT is empty. Step 2 did not produce any usable support rows for the selected "
            "task/offender/rate configuration."
        )

    work = dataframe.copy()

    for column in CANONICAL_EFF_COLUMNS:
        lut_column = f"lut_{column}"
        if lut_column not in work.columns:
            work[lut_column] = np.nan

    if "lut_neighbor_count" not in work.columns:
        work["lut_neighbor_count"] = np.nan
    if "lut_neighbor_min_distance" not in work.columns:
        work["lut_neighbor_min_distance"] = np.nan
    if "lut_neighbor_max_distance" not in work.columns:
        work["lut_neighbor_max_distance"] = np.nan

    exact_mask = work["lut_scale_factor"].notna()
    work.loc[exact_mask, "lut_neighbor_count"] = 1.0
    work.loc[exact_mask, "lut_neighbor_min_distance"] = 0.0
    work.loc[exact_mask, "lut_neighbor_max_distance"] = 0.0

    missing_mask = work["lut_scale_factor"].isna()
    if not missing_mask.any() or match_mode == "exact":
        return work

    normalized_mode = str(match_mode).strip().lower()
    if normalized_mode not in {"nearest", "interpolate"}:
        raise ValueError(f"Unsupported LUT match mode: {match_mode!r}")

    if normalized_mode == "interpolate":
        if raw_columns is None:
            raise ValueError("raw_columns must be provided when using interpolated LUT matching.")
        distance_columns = list(raw_columns)
    else:
        distance_columns = list(query_columns)

    if normalized_mode == "interpolate" and float(interpolation_power) <= 0.0:
        raise ValueError("interpolation_power must be positive.")

    def _fill_subset(row_mask: pd.Series, lut_subset: pd.DataFrame) -> None:
        if not row_mask.any() or lut_subset.empty:
            return

        lut_matrix = lut_subset[CANONICAL_EFF_COLUMNS].to_numpy(dtype=float)
        scale_values = lut_subset["scale_factor"].to_numpy(dtype=float)
        query_matrix = work.loc[row_mask, distance_columns].to_numpy(dtype=float)
        distances = np.sqrt(((query_matrix[:, None, :] - lut_matrix[None, :, :]) ** 2).sum(axis=2))

        best_indices = np.argmin(distances, axis=1)
        best_distances = distances[np.arange(len(best_indices)), best_indices]
        anchor_rows = lut_subset.iloc[best_indices].reset_index(drop=True)

        if normalized_mode == "nearest":
            work.loc[row_mask, "lut_scale_factor"] = anchor_rows["scale_factor"].to_numpy(dtype=float)
            work.loc[row_mask, "lut_match_method"] = "nearest"
            work.loc[row_mask, "lut_match_distance"] = best_distances
            work.loc[row_mask, "lut_neighbor_count"] = 1.0
            work.loc[row_mask, "lut_neighbor_min_distance"] = best_distances
            work.loc[row_mask, "lut_neighbor_max_distance"] = best_distances
            for column in CANONICAL_EFF_COLUMNS:
                work.loc[row_mask, f"lut_{column}"] = anchor_rows[column].to_numpy(dtype=float)
            return

        neighbor_count = int(interpolation_k) if interpolation_k is not None else 8
        neighbor_count = max(1, min(neighbor_count, len(lut_subset)))
        if neighbor_count == 1:
            work.loc[row_mask, "lut_scale_factor"] = anchor_rows["scale_factor"].to_numpy(dtype=float)
            work.loc[row_mask, "lut_match_method"] = "nearest"
            work.loc[row_mask, "lut_match_distance"] = best_distances
            work.loc[row_mask, "lut_neighbor_count"] = 1.0
            work.loc[row_mask, "lut_neighbor_min_distance"] = best_distances
            work.loc[row_mask, "lut_neighbor_max_distance"] = best_distances
            for column in CANONICAL_EFF_COLUMNS:
                work.loc[row_mask, f"lut_{column}"] = anchor_rows[column].to_numpy(dtype=float)
            return

        neighbor_indices = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        ordering = np.argsort(neighbor_distances, axis=1)
        neighbor_indices = np.take_along_axis(neighbor_indices, ordering, axis=1)
        neighbor_distances = np.take_along_axis(neighbor_distances, ordering, axis=1)
        neighbor_scales = scale_values[neighbor_indices]
        weights = 1.0 / np.maximum(neighbor_distances, 1e-9) ** float(interpolation_power)
        interpolated_scale = (weights * neighbor_scales).sum(axis=1) / weights.sum(axis=1)

        work.loc[row_mask, "lut_scale_factor"] = interpolated_scale
        work.loc[row_mask, "lut_match_method"] = "interpolated"
        work.loc[row_mask, "lut_match_distance"] = best_distances
        work.loc[row_mask, "lut_neighbor_count"] = float(neighbor_count)
        work.loc[row_mask, "lut_neighbor_min_distance"] = neighbor_distances[:, 0]
        work.loc[row_mask, "lut_neighbor_max_distance"] = neighbor_distances[:, -1]
        for column in CANONICAL_EFF_COLUMNS:
            work.loc[row_mask, f"lut_{column}"] = anchor_rows[column].to_numpy(dtype=float)

    effective_group_columns = [
        column
        for column in (group_columns or [])
        if column in work.columns and column in lut_dataframe.columns
    ]
    if not effective_group_columns:
        _fill_subset(missing_mask, lut_dataframe)
        return work

    unmatched_mask = missing_mask.copy()
    candidate_groups = (
        work.loc[missing_mask, effective_group_columns]
        .dropna()
        .drop_duplicates()
    )
    for row in candidate_groups.itertuples(index=False, name=None):
        row_mask = missing_mask.copy()
        lut_mask = np.ones(len(lut_dataframe), dtype=bool)
        for column, value in zip(effective_group_columns, row):
            row_mask &= np.isclose(pd.to_numeric(work[column], errors="coerce"), float(value), equal_nan=False)
            lut_mask &= np.isclose(pd.to_numeric(lut_dataframe[column], errors="coerce"), float(value), equal_nan=False)
        if not row_mask.any():
            continue
        unmatched_mask &= ~row_mask
        _fill_subset(row_mask, lut_dataframe.loc[lut_mask].reset_index(drop=True))

    if unmatched_mask.any():
        _fill_subset(unmatched_mask, lut_dataframe)
    return work
