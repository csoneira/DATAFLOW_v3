#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CANONICAL_EFF_COLUMNS,
    CANONICAL_Z_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    ROBUST_EFFICIENCY_VARIANT_TO_SUFFIX,
    STEP1_OUTPUT_COLUMNS,
    canonicalize_trigger_value,
    cfg_path,
    choose_z_vector,
    derive_trigger_rate_features,
    ensure_output_dirs,
    filter_by_z_vector,
    get_trigger_type_selection,
    load_config,
    ordered_plot_filename,
    required_trigger_rate_columns,
    trigger_count_source_column,
    write_json,
)

log = logging.getLogger("another_method.step1")

TRIGGER_COLUMN = "trigger_combinations"
STEP1_PLOT_PATH = PLOTS_DIR / ordered_plot_filename(1, 1, "parameter_space_overview")
SIMULATED_EFF_SOURCE_COLUMNS = ["eff_p1", "eff_p2", "eff_p3", "eff_p4"]
EMPIRICAL_EFF_SOURCE_COLUMNS = [f"eff_empirical_{idx}" for idx in range(1, 5)]
DEFAULT_SIM_PARAMS_CSV = (
    Path(__file__).resolve().parents[2]
    / "MINGO_DIGITAL_TWIN"
    / "SIMULATED_DATA"
    / "step_final_simulation_params.csv"
)
DEFAULT_MINGO00_METADATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "STATIONS"
    / "MINGO00"
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
)


def _resolve_observed_efficiency_limits(config: dict, *, config_key: str) -> dict[int, float]:
    step1_config = config.get("step1", {})
    if not isinstance(step1_config, dict):
        step1_config = {}

    raw = step1_config.get(config_key, {})
    if raw in (None, "", "null", "None"):
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"step1.{config_key} must be a JSON object keyed by plane index.")

    limits: dict[int, float] = {}
    for key, value in raw.items():
        text = str(key).strip().lower()
        if text.startswith("plane_"):
            text = text[6:]
        plane_idx = int(text)
        if plane_idx < 1 or plane_idx > 4:
            raise ValueError(f"Invalid plane index in step1.{config_key}: {key!r}")
        if value in (None, "", "null", "None"):
            continue
        limits[plane_idx] = float(value)
    return limits


def _resolve_observed_efficiency_upper_limits(config: dict) -> dict[int, float]:
    return _resolve_observed_efficiency_limits(config, config_key="observed_efficiency_upper_limits")


def _resolve_observed_efficiency_lower_limits(config: dict) -> dict[int, float]:
    return _resolve_observed_efficiency_limits(config, config_key="observed_efficiency_lower_limits")


def _drop_rows_outside_observed_efficiency_limits(
    dataframe: pd.DataFrame,
    upper_limits: dict[int, float],
    lower_limits: dict[int, float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    row_count_before = int(len(dataframe))
    if not upper_limits and not lower_limits:
        return dataframe.copy(), {
            "mode": "drop",
            "limits_by_plane": {},
            "upper_limits_by_plane": {},
            "lower_limits_by_plane": {},
            "affected_rows_by_plane": {},
            "affected_rows_above_upper_by_plane": {},
            "affected_rows_below_lower_by_plane": {},
            "affected_rows_total": 0,
            "row_count_before": row_count_before,
            "row_count_after": row_count_before,
        }

    work = dataframe.copy()
    counts_by_plane: dict[str, int] = {}
    counts_above_by_plane: dict[str, int] = {}
    counts_below_by_plane: dict[str, int] = {}
    outside_union_mask = pd.Series(False, index=work.index, dtype=bool)
    for plane_idx in sorted(set(upper_limits).union(lower_limits)):
        column = f"eff_empirical_{plane_idx}"
        if column not in work.columns:
            continue
        numeric = pd.to_numeric(work[column], errors="coerce")
        lower_limit = lower_limits.get(plane_idx, None)
        upper_limit = upper_limits.get(plane_idx, None)
        below_mask = pd.Series(False, index=work.index, dtype=bool)
        above_mask = pd.Series(False, index=work.index, dtype=bool)
        if lower_limit is not None:
            below_mask = numeric.notna() & (numeric < float(lower_limit))
            counts_below_by_plane[str(plane_idx)] = int(below_mask.sum())
        if upper_limit is not None:
            above_mask = numeric.notna() & (numeric > float(upper_limit))
            counts_above_by_plane[str(plane_idx)] = int(above_mask.sum())
        outside_mask = below_mask | above_mask
        counts_by_plane[str(plane_idx)] = int(outside_mask.sum())
        outside_union_mask = outside_union_mask | outside_mask
        work[column] = numeric

    if bool(outside_union_mask.any()):
        work = work.loc[~outside_union_mask].copy()

    return work, {
        "mode": "drop",
        "limits_by_plane": {
            str(plane_idx): {
                "lower": (float(lower_limits[plane_idx]) if plane_idx in lower_limits else None),
                "upper": (float(upper_limits[plane_idx]) if plane_idx in upper_limits else None),
            }
            for plane_idx in sorted(set(upper_limits).union(lower_limits))
        },
        "upper_limits_by_plane": {str(key): float(value) for key, value in sorted(upper_limits.items())},
        "lower_limits_by_plane": {str(key): float(value) for key, value in sorted(lower_limits.items())},
        "affected_rows_by_plane": counts_by_plane,
        "affected_rows_above_upper_by_plane": counts_above_by_plane,
        "affected_rows_below_lower_by_plane": counts_below_by_plane,
        "affected_rows_total": int(outside_union_mask.sum()),
        "row_count_before": row_count_before,
        "row_count_after": int(len(work)),
    }


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_1 - %(message)s", level=logging.INFO, force=True)


def _display_label(column: str) -> str:
    labels = {
        "sim_flux_cm2_min": "Flux [cm^-2 min^-1]",
        "emp_eff_1": "Plane 1 sim eff",
        "emp_eff_2": "Plane 2 sim eff",
        "emp_eff_3": "Plane 3 sim eff",
        "emp_eff_4": "Plane 4 sim eff",
    }
    return labels.get(column, column.replace("_", " "))


def _compute_axis_limits(series: pd.Series) -> tuple[float, float] | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    lower = float(numeric.min())
    upper = float(numeric.max())
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if lower == upper:
        pad = max(abs(lower) * 0.05, 0.05)
        return lower - pad, upper + pad
    pad = (upper - lower) * 0.04
    return lower - pad, upper + pad


def _write_parameter_space_plot(dataframe: pd.DataFrame, selected_z_vector: tuple[float, float, float, float]) -> Path | None:
    plot_columns = ["sim_flux_cm2_min"] + CANONICAL_EFF_COLUMNS
    plot_df = dataframe.loc[:, plot_columns].copy()
    for column in plot_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(how="all", subset=plot_columns).reset_index(drop=True)
    if plot_df.empty:
        return None

    axis_limits = {column: _compute_axis_limits(plot_df[column]) for column in plot_columns}
    plot_columns = [column for column in plot_columns if axis_limits.get(column) is not None]
    if not plot_columns:
        return None

    n = len(plot_columns)
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(2.45 * n + 1.2, 2.45 * n + 1.4),
        dpi=130,
    )
    axes_arr = np.asarray(axes).reshape(n, n)
    color = "#4C78A8"

    for row_idx in range(n):
        for col_idx in range(n):
            ax = axes_arr[row_idx, col_idx]
            x_col = plot_columns[col_idx]
            y_col = plot_columns[row_idx]

            ax.grid(True, linestyle=":", linewidth=0.35, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            ax.set_box_aspect(1.0)

            if row_idx == col_idx:
                values = plot_df[y_col].dropna()
                if not values.empty:
                    ax.hist(
                        values,
                        bins=18,
                        color=color,
                        alpha=0.55,
                        edgecolor="white",
                        linewidth=0.45,
                    )
                ax.set_title(_display_label(y_col), fontsize=9, pad=4)
                ax.set_xlim(axis_limits[y_col])
                ax.autoscale_view(scalex=False, scaley=True)
            elif row_idx > col_idx:
                subset = plot_df[[x_col, y_col]].dropna()
                if not subset.empty:
                    ax.scatter(
                        subset[x_col],
                        subset[y_col],
                        s=16.0,
                        alpha=0.68,
                        color=color,
                        edgecolors="none",
                        rasterized=True,
                    )
                ax.set_xlim(axis_limits[x_col])
                ax.set_ylim(axis_limits[y_col])
            else:
                ax.axis("off")

            if row_idx < n - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(_display_label(x_col), fontsize=8)
            if col_idx > 0:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_ylabel(_display_label(y_col), fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7, width=0.75, length=2.6)

    z_text = ", ".join(f"{float(value):g}" for value in selected_z_vector)
    fig.suptitle(
        "Step 1 parameter-space overview\n"
        f"selected z = [{z_text}] | rows = {len(plot_df)}",
        fontsize=11,
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.9, wspace=0.05, hspace=0.05)

    STEP1_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEP1_PLOT_PATH, dpi=170)
    plt.close(fig)
    return STEP1_PLOT_PATH


def _metadata_csv_path(metadata_root: Path, task_id: int, source_name: str) -> Path:
    return metadata_root / f"TASK_{int(task_id)}" / "METADATA" / f"task_{int(task_id)}_metadata_{source_name}.csv"


def _load_param_hash_metadata(path: Path, needed_columns: list[str]) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    usecols = [column for column in dict.fromkeys(needed_columns) if column in header]
    dataframe = pd.read_csv(path, usecols=usecols, low_memory=False)
    if "execution_timestamp" in dataframe.columns:
        dataframe["_exec_ts"] = pd.to_datetime(
            dataframe["execution_timestamp"],
            errors="coerce",
            format="%Y-%m-%d_%H.%M.%S",
        )
        missing_ts = dataframe["_exec_ts"].isna()
        if missing_ts.any():
            dataframe.loc[missing_ts, "_exec_ts"] = pd.to_datetime(
                dataframe.loc[missing_ts, "execution_timestamp"],
                errors="coerce",
            )
        dataframe = (
            dataframe.sort_values(["param_hash", "_exec_ts"], na_position="last")
            .groupby("param_hash")
            .tail(1)
            .drop(columns=["_exec_ts"], errors="ignore")
        )
    return dataframe


def _resolve_step1_input_dataframe(config: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    step1_config = config.get("step1", {})
    if not isinstance(step1_config, dict):
        step1_config = {}

    use_trigger_type_param_hash_source = bool(step1_config.get("use_mingo00_param_hash_source", True))
    if not use_trigger_type_param_hash_source:
        input_path = cfg_path(config, "paths", "collected_data_csv")
        dataframe = pd.read_csv(input_path)
        return dataframe, {
            "input_mode": "legacy_collected_data_csv",
            "source_file": str(input_path),
        }

    trigger_selection = get_trigger_type_selection(config)
    sim_params_path = Path(step1_config.get("simulation_params_csv", str(DEFAULT_SIM_PARAMS_CSV))).expanduser()
    if not sim_params_path.is_absolute():
        sim_params_path = (Path(config.get("_config_dir", ".")) / sim_params_path).resolve()
    if not sim_params_path.exists():
        raise FileNotFoundError(f"Simulation-params CSV does not exist: {sim_params_path}")

    params_df = pd.read_csv(sim_params_path, low_memory=False)
    if "param_hash" not in params_df.columns:
        raise ValueError("Simulation-params CSV must contain param_hash.")

    if "efficiencies" in params_df.columns and not all(c in params_df.columns for c in SIMULATED_EFF_SOURCE_COLUMNS):
        parsed = params_df["efficiencies"].astype(str).str.strip()
        parsed = parsed.str.replace("(", "[", regex=False).str.replace(")", "]", regex=False)
        eff_vectors = parsed.map(lambda text: json.loads(text) if text.startswith("[") and text.endswith("]") else None)
        for idx, column in enumerate(SIMULATED_EFF_SOURCE_COLUMNS):
            params_df[column] = eff_vectors.map(lambda vec, i=idx: np.nan if not isinstance(vec, list) or len(vec) != 4 else float(vec[i]))
    missing_sim_eff_columns = [column for column in SIMULATED_EFF_SOURCE_COLUMNS if column not in params_df.columns]
    if missing_sim_eff_columns:
        raise ValueError(
            "Simulation-params CSV must contain simulated efficiencies either as "
            "eff_p1..eff_p4 columns or a parseable efficiencies vector. Missing: "
            + ", ".join(missing_sim_eff_columns)
        )
    for column in SIMULATED_EFF_SOURCE_COLUMNS:
        params_df[column] = pd.to_numeric(params_df[column], errors="coerce")

    metadata_root = Path(
        step1_config.get(
            "metadata_root",
            step1_config.get(
                "trigger_type_metadata_root",
                step1_config.get("mingo00_metadata_root", str(DEFAULT_MINGO00_METADATA_ROOT)),
            ),
        )
    ).expanduser()
    if not metadata_root.is_absolute():
        metadata_root = (Path(config.get("_config_dir", ".")) / metadata_root).resolve()
    metadata_source = str(trigger_selection.get("metadata_source", "trigger_type"))
    source_name = str(trigger_selection.get("source_name", "trigger_type"))
    task_id = int(trigger_selection.get("metadata_task_id", trigger_selection["task_id"]))
    rate_source_name = str(trigger_selection.get("rate_source_name", source_name))
    rate_task_id = int(trigger_selection.get("rate_metadata_task_id", task_id))
    trigger_path = _metadata_csv_path(metadata_root, task_id, source_name)
    if not trigger_path.exists():
        raise FileNotFoundError(f"Missing required {source_name} metadata file: {trigger_path}")
    rate_path = _metadata_csv_path(metadata_root, rate_task_id, rate_source_name)
    rate_source_is_primary = trigger_path == rate_path

    header = set(pd.read_csv(trigger_path, nrows=0).columns)
    stage_prefix = trigger_selection.get("stage_prefix")
    offender_threshold = trigger_selection.get("offender_threshold")
    if metadata_source == "robust_efficiency":
        required_rate_columns = ["rate_1234_hz", "rate_total_hz"]
        selected_source_rate_column = str(trigger_selection.get("selected_source_rate_column", "rate_1234_hz"))
        if rate_source_is_primary and selected_source_rate_column not in required_rate_columns:
            required_rate_columns.append(selected_source_rate_column)
        selected_count_column = trigger_selection.get("selected_count_column")
        required_count_columns = [
            column
            for column in (
                selected_count_column,
                "four_plane_count",
                "count_1234",
                "rate_1234_count",
                "four_plane_robust_count",
                "four_plane_robust_count_union",
                "four_plane_robust_count_intersection",
                "count_four_plane_robust",
                "total_count",
                "count_total",
                "rate_total_count",
            )
            if column in header
        ]
        efficiency_variant = str(trigger_selection.get("robust_efficiency_variant", "default"))
        efficiency_suffix = ROBUST_EFFICIENCY_VARIANT_TO_SUFFIX[efficiency_variant]
        required_efficiency_columns = [f"eff{idx}{efficiency_suffix}" for idx in range(1, 5)]
        needed = ["param_hash"] + required_efficiency_columns + required_rate_columns + required_count_columns
        missing_required_columns = [
            column for column in ["param_hash"] + required_efficiency_columns + required_rate_columns if column not in header
        ]
    else:
        stage_prefix = str(stage_prefix)
        required_rate_columns = required_trigger_rate_columns(stage_prefix, offender_threshold)
        selected_source_rate_column = str(trigger_selection.get("selected_source_rate_column", ""))
        if rate_source_is_primary and selected_source_rate_column and selected_source_rate_column not in required_rate_columns:
            required_rate_columns.append(selected_source_rate_column)
        required_count_columns = [
            column
            for column in (
                trigger_count_source_column(stage_prefix, label, offender_threshold)
                for label in ("12", "13", "14", "23", "24", "34", "123", "124", "134", "234", "1234")
            )
            if column is not None
        ]
        needed = ["param_hash"] + [column for column in required_rate_columns + required_count_columns if column in header]
        missing_required_columns = [column for column in ["param_hash"] + required_rate_columns if column not in header]
    if "count_rate_denominator_seconds" in header:
        needed.append("count_rate_denominator_seconds")
    if "execution_timestamp" in header:
        needed.append("execution_timestamp")
    needed = list(dict.fromkeys(needed))
    if missing_required_columns:
        raise ValueError(
            f"Selected {source_name} metadata is missing required columns: "
            + ", ".join(missing_required_columns)
        )

    trigger_df = _load_param_hash_metadata(trigger_path, needed)
    metadata_files_used = [str(trigger_path)]
    if not rate_source_is_primary:
        if not rate_path.exists():
            raise FileNotFoundError(f"Missing required rate-source metadata file: {rate_path}")
        rate_header = set(pd.read_csv(rate_path, nrows=0).columns)
        selected_count_column = trigger_selection.get("selected_count_column")
        rate_needed = ["param_hash", selected_source_rate_column]
        if selected_count_column not in (None, "", "null", "None"):
            rate_needed.append(str(selected_count_column))
        if "count_rate_denominator_seconds" not in trigger_df.columns:
            rate_needed.append("count_rate_denominator_seconds")
        missing_rate_columns = [column for column in ["param_hash", selected_source_rate_column] if column not in rate_header]
        if missing_rate_columns:
            raise ValueError(
                f"Selected rate-source metadata is missing required columns: "
                + ", ".join(missing_rate_columns)
            )
        rate_df = _load_param_hash_metadata(rate_path, rate_needed)
        rename_map = {column: f"__rate_source__{column}" for column in rate_df.columns if column != "param_hash"}
        merged_rate = trigger_df.merge(rate_df.rename(columns=rename_map), on="param_hash", how="inner")
        for column, temp_column in rename_map.items():
            merged_rate[column] = merged_rate[temp_column]
        trigger_df = merged_rate.drop(columns=list(rename_map.values()), errors="ignore")
        metadata_files_used.append(str(rate_path))

    merged = params_df.merge(trigger_df, on="param_hash", how="inner")
    if merged.empty:
        raise ValueError("Param-hash merge produced no rows between sim params and MINGO00 metadata.")

    return merged, {
        "input_mode": f"param_hash_merge_{source_name}",
        "metadata_source": metadata_source,
        "simulation_params_csv": str(sim_params_path),
        "simulation_efficiency_columns": SIMULATED_EFF_SOURCE_COLUMNS,
        "metadata_files_used": metadata_files_used,
        "metadata_root": str(metadata_root),
        "trigger_type_metadata_root": str(metadata_root),
        "source_name": source_name,
        "task_id": task_id,
        "rate_task_id": rate_task_id,
        "rate_source_name": rate_source_name,
        "stage_prefix": stage_prefix,
        "offender_threshold": offender_threshold,
        "required_rate_columns": required_rate_columns,
        "required_count_columns": required_count_columns,
        "rows_sim_params": int(len(params_df)),
        "rows_rate_by_hash": int(len(trigger_df)),
        "rows_after_param_hash_merge": int(len(merged)),
    }


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    output_path = cfg_path(config, "paths", "step1_filtered_csv")
    meta_path = cfg_path(config, "paths", "step1_meta_json")

    z_columns = list(config["columns"]["z_positions"])
    flux_column = str(config["columns"]["simulated_flux"])
    num_events_column = str(config["columns"]["num_events"])
    min_events = int(config.get("step1", {}).get("min_events", 70000))
    configured_z_vector = config.get("step1", {}).get("z_position_vector")
    step1_config = config.get("step1", {})

    dataframe, source_info = _resolve_step1_input_dataframe(config)
    dataframe, trigger_feature_info = derive_trigger_rate_features(
        dataframe,
        config,
        allow_plain_fallback=False,
    )
    required_columns = EMPIRICAL_EFF_SOURCE_COLUMNS + SIMULATED_EFF_SOURCE_COLUMNS + z_columns + ["rate_hz", flux_column, num_events_column]
    missing_required = [column for column in required_columns if column not in dataframe.columns]
    if missing_required:
        raise ValueError("Step 1 resolved input is missing required columns: " + ", ".join(missing_required))
    initial_rows = int(len(dataframe))

    rate_values = pd.to_numeric(dataframe["rate_hz"], errors="coerce")
    four_plane_values = pd.to_numeric(dataframe.get("four_plane_rate_hz"), errors="coerce")
    emp_eff_frame = dataframe[EMPIRICAL_EFF_SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    sim_eff_frame = dataframe[SIMULATED_EFF_SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_trigger_mask = np.isfinite(rate_values) & (rate_values > 0.0)
    if "four_plane_rate_hz" in dataframe.columns:
        valid_trigger_mask &= np.isfinite(four_plane_values) & (four_plane_values > 0.0)
    valid_trigger_mask &= np.isfinite(emp_eff_frame.to_numpy()).all(axis=1)
    valid_trigger_mask &= np.isfinite(sim_eff_frame.to_numpy()).all(axis=1)
    dataframe = dataframe.loc[valid_trigger_mask].copy()
    rows_after_trigger_validity_filter = int(len(dataframe))
    if dataframe.empty:
        selected_rate_name = str(trigger_feature_info.get("selected_source_rate_column", trigger_feature_info["rate_family_column"]))
        raise ValueError(
            "Step 1 found no usable rows after deriving rate-source features for "
            f"{selected_rate_name}. The selected metadata columns currently yield no positive "
            "four-plane support with finite empirical and simulated efficiencies."
        )

    observed_efficiency_upper_limits = _resolve_observed_efficiency_upper_limits(config)
    observed_efficiency_lower_limits = _resolve_observed_efficiency_lower_limits(config)
    dataframe, observed_efficiency_limit_filter = _drop_rows_outside_observed_efficiency_limits(
        dataframe,
        upper_limits=observed_efficiency_upper_limits,
        lower_limits=observed_efficiency_lower_limits,
    )
    rows_after_observed_efficiency_limit_filter = int(len(dataframe))
    if dataframe.empty:
        raise ValueError(
            "No Step 1 rows remain after applying observed_efficiency bounds."
        )

    selected_z_vector = choose_z_vector(dataframe, z_columns, configured_z_vector)
    dataframe = filter_by_z_vector(dataframe, z_columns, selected_z_vector)
    rows_after_z_filter = int(len(dataframe))
    dataframe = dataframe[dataframe[num_events_column].astype(float) >= min_events].copy()
    rows_after_event_filter = int(len(dataframe))
    if dataframe.empty:
        raise ValueError(
            "No Step 1 rows remain after z-position and minimum-event filtering for the selected "
            "rate-source configuration."
        )

    selected_trigger = None
    trigger_values: list[object] = []
    if TRIGGER_COLUMN in dataframe.columns:
        seen: set[str] = set()
        for raw_value in dataframe[TRIGGER_COLUMN]:
            parsed_value = canonicalize_trigger_value(raw_value)
            if parsed_value is None:
                continue
            key = json.dumps(parsed_value, sort_keys=True) if not isinstance(parsed_value, str) else parsed_value
            if key in seen:
                continue
            seen.add(key)
            trigger_values.append(parsed_value)
        if trigger_values:
            selected_trigger = trigger_values[0]

    rename_map = {
        z_columns[0]: CANONICAL_Z_COLUMNS[0],
        z_columns[1]: CANONICAL_Z_COLUMNS[1],
        z_columns[2]: CANONICAL_Z_COLUMNS[2],
        z_columns[3]: CANONICAL_Z_COLUMNS[3],
        "eff_p1": CANONICAL_EFF_COLUMNS[0],
        "eff_p2": CANONICAL_EFF_COLUMNS[1],
        "eff_p3": CANONICAL_EFF_COLUMNS[2],
        "eff_p4": CANONICAL_EFF_COLUMNS[3],
        flux_column: "sim_flux_cm2_min",
        num_events_column: "num_events",
    }
    dataframe = dataframe.rename(columns=rename_map)
    trigger_passthrough = [
        column
        for column in (
            list(trigger_feature_info.get("source_rate_columns", {}).values())
            + [col for col in trigger_feature_info.get("source_count_columns", {}).values() if col]
            + list(trigger_feature_info.get("robust_efficiency_source_columns", {}).values())
        )
        if column in dataframe.columns and column not in STEP1_OUTPUT_COLUMNS
    ]
    trigger_passthrough = list(dict.fromkeys(trigger_passthrough))
    dataframe = dataframe[STEP1_OUTPUT_COLUMNS + trigger_passthrough]
    dataframe = dataframe.sort_values(CANONICAL_EFF_COLUMNS + ["sim_flux_cm2_min"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    plot_path = _write_parameter_space_plot(dataframe, selected_z_vector)

    metadata = {
        "selected_z_positions": list(selected_z_vector),
        "selected_trigger": selected_trigger,
        "trigger_values_in_filtered_data": trigger_values,
        "parameter_space_plot": None if plot_path is None else str(plot_path),
        "source": source_info,
        "trigger_rate_selection": trigger_feature_info,
        "trigger_type_selection": trigger_feature_info,
        "row_counts": {
            "initial": initial_rows,
            "after_trigger_validity_filter": rows_after_trigger_validity_filter,
            "after_observed_efficiency_limit_filter": rows_after_observed_efficiency_limit_filter,
            "after_observed_efficiency_upper_limit_filter": rows_after_observed_efficiency_limit_filter,
            "after_z_filter": rows_after_z_filter,
            "after_min_events_filter": rows_after_event_filter,
        },
        "observed_efficiency_limit_filter": observed_efficiency_limit_filter,
        "observed_efficiency_upper_limit_filter": observed_efficiency_limit_filter,
        "min_events": min_events,
        "input_columns": {
            "z_positions": z_columns,
            "efficiencies": SIMULATED_EFF_SOURCE_COLUMNS,
            "empirical_efficiencies_for_trigger_validation": EMPIRICAL_EFF_SOURCE_COLUMNS,
            "rate": str(
                trigger_feature_info.get(
                    "selected_display_label",
                    trigger_feature_info.get("selected_source_rate_column", trigger_feature_info["rate_family_column"]),
                )
            ),
            "rate_source_column": str(
                trigger_feature_info.get("selected_source_rate_column", trigger_feature_info["rate_family_column"])
            ),
            "rate_canonical_column": str(trigger_feature_info["rate_family_column"]),
            "selected_empirical_efficiency_source_columns": trigger_feature_info.get(
                "robust_efficiency_source_columns",
                {f"plane_{idx}": f"eff_empirical_{idx}" for idx in range(1, 5)},
            ),
            "simulated_flux": flux_column,
            "num_events": num_events_column,
            "passthrough_trigger_columns": trigger_passthrough,
        },
        "configured_z_position_vector": configured_z_vector,
    }
    write_json(meta_path, metadata)

    log.info("Wrote %d filtered rows to %s", len(dataframe), output_path)
    log.info("Selected z-position vector: %s", selected_z_vector)
    if observed_efficiency_limit_filter["affected_rows_total"] > 0:
        log.info(
            "Dropped rows outside observed-efficiency bounds %s. Affected rows by plane: %s",
            observed_efficiency_limit_filter["limits_by_plane"],
            observed_efficiency_limit_filter["affected_rows_by_plane"],
        )
    if plot_path is not None:
        log.info("Wrote plot: %s", plot_path)
    if selected_trigger is not None:
        log.info("Selected trigger: %s", selected_trigger)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter the collected data for the LUT workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
