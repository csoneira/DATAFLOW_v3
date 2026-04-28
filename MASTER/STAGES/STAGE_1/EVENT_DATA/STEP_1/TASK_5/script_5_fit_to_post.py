#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_5/script_5_fit_to_post.py
Purpose: !/usr/bin/env python3.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_5/script_5_fit_to_post.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

#%%

from __future__ import annotations

"""
Stage 1 Task 5 (FIT-->CORR) finalisation stage.

Consumes the fit outputs from Task 4, applies the derived corrections to the
event lists, validates the corrected distributions, and emits the Stage 1
deliverables that feed Stage 2. The script oversees QA plotting, execution
metadata tracking, and file lifecycle management so the pipeline finishes with
a coherent, traceable set of corrected datasets per station.
"""
# Standard Library
from ast import literal_eval
import argparse
import atexit
import builtins
import csv
from datetime import datetime, timedelta
import gc
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import warnings
from collections import defaultdict
from functools import reduce
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple, Union

# Scientific Computing
import numpy as np
import pandas as pd
import scipy.linalg as linalg
from scipy.constants import c
from scipy.interpolate import CubicSpline, interp1d, RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, curve_fit, minimize, minimize_scalar, nnls
from scipy.special import erf, gamma
from scipy.sparse import csc_matrix, load_npz
from scipy.stats import norm, poisson, linregress, median_abs_deviation, skew

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Image Processing
from PIL import Image

# Progress Bar
from tqdm import tqdm

import yaml

# Resolve repo root for local imports
CURRENT_PATH = Path(__file__).resolve()
REPO_ROOT = None
for parent in CURRENT_PATH.parents:
    if parent.name == "MASTER":
        REPO_ROOT = parent.parent
        break
if REPO_ROOT is None:
    REPO_ROOT = CURRENT_PATH.parents[-1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.config_loader import update_config_with_parameters
from MASTER.common.debug_plots import plot_debug_histograms
from MASTER.common.execution_logger import set_station, start_timer
from MASTER.common.file_selection import (
    filter_expected_artifact_names,
    file_name_in_any_date_range,
    load_date_ranges_from_config,
    select_latest_candidate,
    sync_unprocessed_with_date_range,
)
from MASTER.common.input_file_config import select_input_file_configuration
from MASTER.common.path_config import (
    get_master_config_root,
    get_repo_root,
    resolve_home_path_from_config,
    resolve_master_config_root_from_config,
)
from MASTER.common.plot_utils import (
    collect_saved_plot_paths,
    ensure_plot_state,
    pdf_save_rasterized_page,
)
from MASTER.common.selection_config import load_selection_for_paths, station_is_selected
from MASTER.common.status_csv import initialize_status_row, update_status_progress
from MASTER.common.reprocessing_utils import (
    QA_REPROCESSING_METADATA_KEYS,
    apply_qa_reprocessing_context,
    canonical_processing_basename,
    filter_filenames_by_qa_retry_basenames,
    get_reprocessing_value,
    load_active_qa_retry_basenames,
    load_qa_reprocessing_context_for_file,
)
from MASTER.common.simulated_data_utils import resolve_simulated_z_positions
from MASTER.common.step1_shared import (
    add_normalized_count_metadata,
    add_trigger_type_total_offender_threshold_metadata,
    apply_step1_master_overrides,
    apply_step1_task_parameter_overrides,
    build_events_per_second_metadata,
    build_step1_cli_parser,
    build_step1_filtered_print,
    collect_columns,
    extract_rate_histogram_metadata,
    extract_trigger_type_metadata,
    is_trigger_type_file_column,
    is_trigger_type_metadata_column,
    is_specific_metadata_excluded_column,
    load_itineraries_from_file,
    normalize_tt_label,
    prune_redundant_count_metadata,
    save_metadata,
    set_global_rate_from_tt_rates,
    load_step1_task_plot_catalog,
    resolve_step1_plot_options,
    step1_task_plot_enabled,
    validate_step1_input_file_args,
    y_pos,
)

task_number = 5
TASK4_CHARGE_SUM_COLUMNS: tuple[str, ...] = (
    "P1_Q_sum_final",
    "P2_Q_sum_final",
    "P3_Q_sum_final",
    "P4_Q_sum_final",
)
TASK4_LISTED_FALLBACK_SUBDIRS: tuple[str, ...] = (
    "PROCESSING_DIRECTORY",
    "COMPLETED_DIRECTORY",
    "UNPROCESSED_DIRECTORY",
    "OUT_OF_DATE_DIRECTORY",
)
TASK5_STRIP_Q_COLUMN_RE = re.compile(r"^Q_P[1-4]s[1-4]$")
TASK5_PLANE_Q_SUM_COLUMN_RE = re.compile(r"^P[1-4]_Q_sum_final$")

try:
    import pyarrow as pa
except Exception:  # pragma: no cover - pyarrow is already required for parquet IO here.
    pa = None


def _preferred_parquet_compression() -> str:
    if pa is not None:
        try:
            if pa.Codec.is_available("snappy"):
                return "snappy"
        except Exception:
            pass
    return "zstd"


def _charge_series_is_usable(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return False
    return bool((numeric > 0).any())


def _derive_charge_event_from_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series | None, str | None]:
    for column_name in ("charge_event", "tim_charge_event"):
        if column_name not in dataframe.columns:
            continue
        series = pd.to_numeric(dataframe[column_name], errors="coerce")
        if _charge_series_is_usable(series):
            return series.astype(float), column_name

    plane_sum_columns = [
        column_name
        for column_name in dataframe.columns
        if TASK5_PLANE_Q_SUM_COLUMN_RE.fullmatch(str(column_name))
    ]
    if plane_sum_columns:
        plane_sum_series = (
            dataframe.loc[:, plane_sum_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
            .astype(float)
        )
        if _charge_series_is_usable(plane_sum_series):
            return plane_sum_series, "+".join(plane_sum_columns)

    strip_q_columns = [
        column_name
        for column_name in dataframe.columns
        if TASK5_STRIP_Q_COLUMN_RE.fullmatch(str(column_name))
    ]
    if strip_q_columns:
        strip_q_series = (
            dataframe.loc[:, strip_q_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
            .astype(float)
        )
        if _charge_series_is_usable(strip_q_series):
            return strip_q_series, "+".join(strip_q_columns)

    return None, None


def _load_charge_event_from_task4_listed(
    station_root: str | Path,
    basename_no_ext: str,
    event_ids: pd.Series,
) -> tuple[pd.Series | None, str | None]:
    if event_ids.empty:
        return None, None

    task4_input_root = (
        Path(station_root)
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_4"
        / "INPUT_FILES"
    )

    for subdir_name in TASK4_LISTED_FALLBACK_SUBDIRS:
        candidate_path = task4_input_root / subdir_name / f"listed_{basename_no_ext}.parquet"
        if not candidate_path.exists():
            continue
        try:
            task4_df = pd.read_parquet(
                candidate_path,
                columns=["event_id", *TASK4_CHARGE_SUM_COLUMNS],
            )
        except Exception as exc:
            print(f"Warning: could not load Task 4 charge fallback {candidate_path}: {exc}")
            continue
        if "event_id" not in task4_df.columns:
            continue

        present_charge_columns = [
            column_name
            for column_name in TASK4_CHARGE_SUM_COLUMNS
            if column_name in task4_df.columns
        ]
        if not present_charge_columns:
            continue

        charge_lookup = task4_df.loc[:, ["event_id", *present_charge_columns]].copy()
        charge_lookup["charge_event"] = (
            charge_lookup.loc[:, present_charge_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
            .astype(float)
        )
        charge_lookup = charge_lookup.loc[:, ["event_id", "charge_event"]]
        charge_lookup = charge_lookup.drop_duplicates(subset=["event_id"], keep="last")

        aligned = pd.DataFrame({"event_id": event_ids}).merge(
            charge_lookup,
            on="event_id",
            how="left",
        )["charge_event"]
        if _charge_series_is_usable(aligned):
            return aligned.astype(float), str(candidate_path)

    return None, None


def resolve_charge_event_series(
    dataframe: pd.DataFrame,
    *,
    station_root: str | Path,
    basename_no_ext: str,
) -> tuple[pd.Series | None, str | None]:
    series, source = _derive_charge_event_from_dataframe(dataframe)
    if series is not None:
        return series, source

    if "event_id" in dataframe.columns:
        return _load_charge_event_from_task4_listed(
            station_root,
            basename_no_ext,
            pd.to_numeric(dataframe["event_id"], errors="coerce"),
        )

    return None, None

# I want to chrono the execution time of the script
start_execution_time_counting = datetime.now()
_prof_t0 = time.perf_counter()
_prof = {}

STATION_CHOICES = ("0", "1", "2", "3", "4")
TASK5_PLOT_ALIASES: tuple[str, ...] = (
    "debug_suite",
    "usual_suite",
    "essential_suite",
    "charge_fit_tt_1234_per_plane_and_total",
    "theta_phi_definitive_tt_2d",
    "polar_theta_phi_definitive_tt_2d_detail_pre",
    "polar_theta_phi_definitive_tt_2d_detail_angle_correction",
    "polar_theta_phi_definitive_tt_2d_detail_final",
    "theta_efficiency_simple_3v4",
    "channel_combination_filter_by_tt",
)
task5_plot_status_by_alias: dict[str, str] = {}


def task5_plot_enabled(alias: str) -> bool:
    if not task5_plot_status_by_alias:
        return True
    return step1_task_plot_enabled(alias, task5_plot_status_by_alias, plot_mode)


def resolve_task5_plot_alias(save_path: str, alias: str | None = None) -> str | None:
    if alias is not None:
        return alias if alias in TASK5_PLOT_ALIASES else None

    stem = os.path.splitext(os.path.basename(str(save_path)))[0].lower()
    stem = re.sub(r"^\d+_", "", stem)
    if stem in TASK5_PLOT_ALIASES:
        return stem
    return None


def apply_task5_plot_catalog_modes() -> None:
    global create_plots, create_essential_plots, create_debug_plots, save_plots, create_pdf
    if not task5_plot_status_by_alias:
        return
    create_plots = create_plots and task5_plot_enabled("usual_suite")
    create_essential_plots = create_essential_plots and task5_plot_enabled("essential_suite")
    create_debug_plots = create_debug_plots and task5_plot_enabled("debug_suite")
    save_plots = bool(create_plots or create_essential_plots or create_debug_plots)
    create_pdf = save_plots


def _coerce_config_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _task5_config_float(
    config_obj: dict[str, object],
    primary_key: str,
    *alias_keys: str,
    default: float,
) -> float:
    for key in (primary_key, *alias_keys):
        raw_value = config_obj.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str) and raw_value.strip().lower() in {"", "none", "null"}:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric configuration value for '{key}': {raw_value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"Non-finite numeric configuration value for '{key}': {raw_value!r}")
        return value
    return float(default)

CLI_PARSER = build_step1_cli_parser("Run Stage 1 STEP_1 TASK_5 (FIT->CORR).", STATION_CHOICES)
CLI_ARGS = CLI_PARSER.parse_args()
validate_step1_input_file_args(CLI_PARSER, CLI_ARGS)

VERBOSE = bool(os.environ.get("DATAFLOW_VERBOSE")) or CLI_ARGS.verbose
print = build_step1_filtered_print(
    verbose=VERBOSE,
    debug_mode_getter=lambda: bool(globals().get("debug_mode", False)),
    raw_print=builtins.print,
)

def safe_move(source_path: str, dest_path: str) -> str:
    """Move *source_path* to *dest_path* with explicit diagnostics on failure."""
    try:
        return shutil.move(source_path, dest_path)
    except OSError as exc:
        print(f"Error moving '{source_path}' to '{dest_path}': {exc}")
        raise

MAX_OPEN_FIGURES = 16
_OPEN_FIGURE_IDS: list[int] = []
_ORIGINAL_PLT_FIGURE = plt.figure
_ORIGINAL_PLT_SUBPLOTS = plt.subplots
_ORIGINAL_PLT_CLOSE = plt.close

def _track_figure(fig: mpl.figure.Figure) -> mpl.figure.Figure:
    fig_number = getattr(fig, "number", None)
    if fig_number is None:
        return fig
    if fig_number not in _OPEN_FIGURE_IDS:
        _OPEN_FIGURE_IDS.append(fig_number)
    while len(_OPEN_FIGURE_IDS) > MAX_OPEN_FIGURES:
        stale_fig_number = _OPEN_FIGURE_IDS.pop(0)
        _ORIGINAL_PLT_CLOSE(stale_fig_number)
    return fig

def _guarded_figure(*args, **kwargs):
    return _track_figure(_ORIGINAL_PLT_FIGURE(*args, **kwargs))

def _guarded_subplots(*args, **kwargs):
    fig, axes = _ORIGINAL_PLT_SUBPLOTS(*args, **kwargs)
    _track_figure(fig)
    return fig, axes

def _guarded_close(*args, **kwargs):
    target = args[0] if args else kwargs.get("fig", None)
    result = _ORIGINAL_PLT_CLOSE(*args, **kwargs)
    if target in (None, "all"):
        _OPEN_FIGURE_IDS.clear()
    elif isinstance(target, mpl.figure.Figure):
        fig_number = getattr(target, "number", None)
        if fig_number in _OPEN_FIGURE_IDS:
            _OPEN_FIGURE_IDS.remove(fig_number)
    elif isinstance(target, int) and target in _OPEN_FIGURE_IDS:
        _OPEN_FIGURE_IDS.remove(target)
    return result

plt.figure = _guarded_figure
plt.subplots = _guarded_subplots
plt.close = _guarded_close

_direct_pdf_pages: PdfPages | None = None
_direct_pdf_page_count = 0
_direct_pdf_target_path: str | None = None
_direct_pdf_temp_path: str | None = None


def _build_temp_pdf_path(target_path: str) -> str:
    """Return a non-colliding temporary PDF path near the final target."""
    base = f"{target_path}.tmp.{os.getpid()}"
    candidate = base
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{base}.{counter}"
        counter += 1
    return candidate

def save_plot_figure(
    save_path: str,
    fig: mpl.figure.Figure | None = None,
    alias: str | None = None,
    **savefig_kwargs,
) -> None:
    """Save a figure to PNG or directly append it to the task PDF."""
    global _direct_pdf_pages, _direct_pdf_page_count, _direct_pdf_target_path, _direct_pdf_temp_path
    plot_alias = resolve_task5_plot_alias(save_path, alias=alias)
    if plot_alias is not None and not task5_plot_enabled(plot_alias):
        return
    target_fig = fig if fig is not None else plt.gcf()
    direct_pdf_path = globals().get("save_pdf_path")
    if globals().get("create_pdf", False) and direct_pdf_path:
        if _direct_pdf_pages is None:
            _direct_pdf_target_path = str(direct_pdf_path)
            _direct_pdf_temp_path = _build_temp_pdf_path(_direct_pdf_target_path)
            _direct_pdf_pages = PdfPages(_direct_pdf_temp_path)
        pdf_kwargs = dict(savefig_kwargs)
        dpi = int(pdf_kwargs.pop("dpi", 150))
        pdf_kwargs.pop("format", None)
        pdf_save_rasterized_page(_direct_pdf_pages, target_fig, dpi=dpi, **pdf_kwargs)
        _direct_pdf_page_count += 1
        return
    target_fig.savefig(save_path, **savefig_kwargs)

def close_direct_pdf_writer() -> None:
    global _direct_pdf_pages, _direct_pdf_page_count, _direct_pdf_target_path, _direct_pdf_temp_path
    if _direct_pdf_pages is not None:
        _direct_pdf_pages.close()
        _direct_pdf_pages = None

    if _direct_pdf_temp_path and os.path.exists(_direct_pdf_temp_path):
        if _direct_pdf_page_count > 0 and _direct_pdf_target_path:
            os.replace(_direct_pdf_temp_path, _direct_pdf_target_path)
        else:
            os.remove(_direct_pdf_temp_path)

    _direct_pdf_page_count = 0
    _direct_pdf_target_path = None
    _direct_pdf_temp_path = None


atexit.register(close_direct_pdf_writer)

# Warning Filters
warnings.filterwarnings("ignore", message=".*Data has no positive values, and therefore cannot be log-scaled.*")

start_timer(__file__)
config_root = get_master_config_root()
config_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_5"
    / "config_task_5.yaml"
)
plot_catalog_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_5"
    / "config_plots_task_5.yaml"
)
parameter_config_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_5"
    / "config_parameters_task_5.csv"
)
fallback_parameter_config_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "config_parameters.csv"
)
print(f"Using config file: {config_file_path}")
print(f"Using plot catalog file: {plot_catalog_file_path}")
with config_file_path.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
task5_plot_status_by_alias = load_step1_task_plot_catalog(
    plot_catalog_file_path,
    TASK5_PLOT_ALIASES,
    "Task 5",
    log_fn=print,
)
debug_mode = False

home_path = str(resolve_home_path_from_config(config))
REFERENCE_TABLES_DIR = Path(home_path) / "DATAFLOW_v3" / "MASTER" / "CONFIG_FILES" / "METADATA_REPRISE" / "REFERENCE_TABLES"

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

if CLI_ARGS.station is None:
    CLI_PARSER.error("No station provided. Pass <station>.")
station = str(CLI_ARGS.station)
set_station(station)

config = apply_step1_task_parameter_overrides(
    config_obj=config,
    station_id=station,
    task_parameter_path=str(parameter_config_file_path),
    fallback_parameter_path=str(fallback_parameter_config_file_path),
    task_number=task_number,
    update_fn=update_config_with_parameters,
    log_fn=print,
)
config = apply_step1_master_overrides(
    config_obj=config,
    master_config_root=config_root,
    log_fn=print,
)
process_only_qa_retry_files = bool(config.get("process_only_qa_retry_files", False))
if process_only_qa_retry_files:
    print(
        "[QA_ONLY] Enabled by STEP_1 shared config: only files present in the active QA retry list will be processed."
    )

selection_config = load_selection_for_paths(
    [config_file_path],
    master_config_root=config_root,
)
if not station_is_selected(station, selection_config.stations):
    print(f"Station {station} skipped by selection.stations.")
    sys.exit(0)

# Cron job switch that decides if completed files can be revisited.
complete_reanalysis = config.get("complete_reanalysis", False)

def _coerce_numeric_sequence(raw_value, caster):
    """Return a list of numbers parsed from *raw_value*."""
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        result: List[float] = []
        for item in raw_value:
            result.extend(_coerce_numeric_sequence(item, caster))
        return result
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if not cleaned:
            return []
        try:
            parsed = literal_eval(cleaned)
        except (ValueError, SyntaxError):
            cleaned = cleaned.replace("[", " ").replace("]", " ")
            tokens = [tok for tok in re.split(r"[;,\\s]+", cleaned) if tok]
            result = []
            for tok in tokens:
                try:
                    result.append(caster(tok))
                except (ValueError, TypeError):
                    continue
            return result
        else:
            return _coerce_numeric_sequence(parsed, caster)
    if np.isscalar(raw_value):
        try:
            return [caster(raw_value)]
        except (ValueError, TypeError):
            return []
    return []

selected_input_file = CLI_ARGS.input_file_flag or CLI_ARGS.input_file
if selected_input_file:
    user_file_path = selected_input_file
    user_file_selection = True
    print("User provided file path:", user_file_path)
else:
    user_file_selection = False

(
    plot_mode,
    create_plots,
    create_essential_plots,
    save_plots,
    create_pdf,
    show_plots,
    create_debug_plots,
) = resolve_step1_plot_options(config)
apply_task5_plot_catalog_modes()

print("Creating the necessary directories...")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

# Define base working directory
repo_root = get_repo_root()
home_directory = str(repo_root.parent)
station_directory = str(repo_root / "STATIONS" / f"MINGO0{station}")
config_file_directory = str(
    config_root
    / "STAGE_0"
    / "ONLINE_RUN_DICTIONARY"
    / f"STATION_{station}"
)
base_directory = str(repo_root / "STATIONS" / f"MINGO0{station}" / "STAGE_1" / "EVENT_DATA")
raw_to_list_working_directory = os.path.join(base_directory, f"STEP_1/TASK_{task_number}")

metadata_directory = os.path.join(raw_to_list_working_directory, "METADATA")

if task_number == 1:
    raw_directory = "STAGE_0_to_1"
    raw_working_directory = os.path.join(station_directory, raw_directory)
    
else:
    raw_directory = f"STEP_1/TASK_{task_number - 1}/OUTPUT_FILES"
    raw_working_directory = os.path.join(base_directory, raw_directory)

if task_number == 5:
    output_location = os.path.join(base_directory, "STEP_1_TO_2_OUTPUT")
else:
    output_location = os.path.join(raw_to_list_working_directory, "OUTPUT_FILES")

# Define directory paths relative to base_directory
base_directories = {
    "stratos_list_events_directory": os.path.join(home_directory, "STRATOS_XY_DIRECTORY"),
    
    "base_plots_directory": os.path.join(raw_to_list_working_directory, "PLOTS"),
    
    "pdf_directory": os.path.join(raw_to_list_working_directory, "PLOTS/PDF_DIRECTORY"),
    "base_figure_directory": os.path.join(raw_to_list_working_directory, "PLOTS/FIGURE_DIRECTORY"),
    "figure_directory": os.path.join(raw_to_list_working_directory, f"PLOTS/FIGURE_DIRECTORY/FIGURES_EXEC_ON_{date_execution}"),
    
    "ancillary_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY"),
    
    "empty_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/EMPTY_FILES"),
    "rejected_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/REJECTED_FILES"),
    "temp_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/TEMP_FILES"),
    
    "unprocessed_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/UNPROCESSED_DIRECTORY"),
    "out_of_date_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/OUT_OF_DATE_DIRECTORY"),
    "error_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/ERROR_DIRECTORY"),
    "processing_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/PROCESSING_DIRECTORY"),
    "completed_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/COMPLETED_DIRECTORY"),
    
    "output_directory": output_location,

    "raw_directory": os.path.join(raw_working_directory, "."),
    
    "metadata_directory": metadata_directory,
}

# Create ALL directories if they don't already exist

for directory in base_directories.values():
    # Skip figure directories at startup; create lazily after selecting a file.
    if directory in (base_directories["base_figure_directory"], base_directories["figure_directory"]):
        continue
    os.makedirs(directory, exist_ok=True)

debug_plot_directory = os.path.join(
    base_directories["base_plots_directory"],
    "DEBUG_PLOTS",
    f"FIGURES_EXEC_ON_{date_execution}",
)

EXPECTED_INPUT_PREFIX = "fitted_"
EXPECTED_INPUT_EXTENSION = ".parquet"


def _expected_input_files(file_names):
    return sorted(
        filter_expected_artifact_names(
            file_names,
            prefix=EXPECTED_INPUT_PREFIX,
            extension=EXPECTED_INPUT_EXTENSION,
        )
    )
debug_fig_idx = 1

csv_path = os.path.join(metadata_directory, f"task_{task_number}_metadata_execution.csv")
csv_path_specific = os.path.join(metadata_directory, f"task_{task_number}_metadata_specific.csv")
csv_path_rate_histogram = os.path.join(
    metadata_directory,
    f"task_{task_number}_metadata_rate_histogram.csv",
)
csv_path_trigger_type = os.path.join(
    metadata_directory,
    f"task_{task_number}_metadata_trigger_type.csv",
)
csv_path_filter = os.path.join(metadata_directory, f"task_{task_number}_metadata_filter.csv")
csv_path_deep_fiter = os.path.join(
    metadata_directory,
    f"task_{task_number}_metadata_deep_fiter.csv",
)
csv_path_status = os.path.join(metadata_directory, f"task_{task_number}_metadata_status.csv")
csv_path_profiling = os.path.join(metadata_directory, f"task_{task_number}_metadata_profiling.csv")
status_filename_base = ""
status_execution_date = None

# Move files from STAGE_0_to_1 to STAGE_0_to_1_TO_LIST/STAGE_0_to_1_TO_LIST_FILES/UNPROCESSED,
# ensuring that only files not already in UNPROCESSED, PROCESSING,
# or COMPLETED are moved:

raw_directory = base_directories["raw_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
error_directory = base_directories["error_directory"]
stratos_list_events_directory = base_directories["stratos_list_events_directory"]
processing_directory = base_directories["processing_directory"]
completed_directory = base_directories["completed_directory"]
output_directory = base_directories["output_directory"]

empty_files_directory = base_directories["empty_files_directory"]
rejected_files_directory = base_directories["rejected_files_directory"]
temp_files_directory = base_directories["temp_files_directory"]

raw_files = set(_expected_input_files(os.listdir(raw_directory)))
unprocessed_files = set(_expected_input_files(os.listdir(unprocessed_directory)))
processing_files = set(_expected_input_files(os.listdir(processing_directory)))
completed_files = set(_expected_input_files(os.listdir(completed_directory)))

_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------- Data reading and preprocessing ---------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Get lists of files in the directories
unprocessed_files = _expected_input_files(os.listdir(base_directories["unprocessed_directory"]))
processing_files = _expected_input_files(os.listdir(base_directories["processing_directory"]))
completed_files = _expected_input_files(os.listdir(base_directories["completed_directory"]))

def process_file(source_path, dest_path):
    print("Source path:", source_path)
    print("Destination path:", dest_path)
    
    if source_path == dest_path:
        return True
    
    if os.path.exists(dest_path):
        print(f"File already exists at destination (removing...)")
        os.remove(dest_path)
        # return False
    
    print("**********************************************************************")
    print(f"Moving\n'{source_path}'\nto\n'{dest_path}'...")
    print("**********************************************************************")
    
    safe_move(source_path, dest_path)
    now = time.time()
    os.utime(dest_path, (now, now))
    return True

def get_file_path(directory, file_name):
    return os.path.join(directory, file_name)

# Create ALL directories if they don't already exist

for directory in base_directories.values():
    # Skip figure directories at startup; create lazily after selecting a file.
    if directory in (base_directories["base_figure_directory"], base_directories["figure_directory"]):
        continue
    os.makedirs(directory, exist_ok=True)

# status_csv_path = os.path.join(base_directory, "raw_to_list_status.csv")
# status_timestamp = append_status_row(status_csv_path)

# Move files from STAGE_0_to_1 to STAGE_0_to_1_TO_LIST/STAGE_0_to_1_TO_LIST_FILES/UNPROCESSED,
# ensuring that only files not already in UNPROCESSED, PROCESSING,
# or COMPLETED are moved:

raw_directory = base_directories["raw_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
error_directory = base_directories["error_directory"]
stratos_list_events_directory = base_directories["stratos_list_events_directory"]
processing_directory = base_directories["processing_directory"]
completed_directory = base_directories["completed_directory"]
output_directory = base_directories["output_directory"]

empty_files_directory = base_directories["empty_files_directory"]
rejected_files_directory = base_directories["rejected_files_directory"]
temp_files_directory = base_directories["temp_files_directory"]

raw_files = set(_expected_input_files(os.listdir(raw_directory)))
unprocessed_files = set(_expected_input_files(os.listdir(unprocessed_directory)))
processing_files = set(_expected_input_files(os.listdir(processing_directory)))
completed_files = set(_expected_input_files(os.listdir(completed_directory)))

# Ordered list from highest to lowest priority
LEVELS = [
    completed_directory,
    processing_directory,
    unprocessed_directory,
    raw_directory,
]

station_re = re.compile(r'^mi0(\d).*\.dat$', re.IGNORECASE)

seen = set()
for d in LEVELS:
    d = Path(d)
    if not d.exists():
        continue

    current_files = {p.name for p in d.iterdir() if p.is_file()}

    # ────────────────────────────────────────────────────────────────
    # Remove .dat files whose prefix “mi0X” does not match `station`
    # ────────────────────────────────────────────────────────────────
    mismatched = {
        fname for fname in current_files
        if (m := station_re.match(fname)) and int(m.group(1)) != int(station)
    }
    for fname in mismatched:
        fp = d / fname
        try:
            fp.unlink()
            print(f"Removed wrong-station file: {fp}")
        except FileNotFoundError:
            pass

    current_files -= mismatched

    # ────────────────────────────────────────────────────────────────
    # Remove duplicates lower in the hierarchy
    # ────────────────────────────────────────────────────────────────
    duplicates = current_files & seen
    for fname in duplicates:
        fp = d / fname
        try:
            fp.unlink()
            print(f"Removed duplicate: {fp}")
        except FileNotFoundError:
            pass

    seen |= (current_files - duplicates)

# Search in all this directories for empty files and move them to the empty_files_directory
for directory in [raw_directory, unprocessed_directory, processing_directory, completed_directory]:
    files = os.listdir(directory)
    for file in files:
        file_empty = os.path.join(directory, file)
        if os.path.getsize(file_empty) == 0:
            # Ensure the empty files directory exists
            os.makedirs(empty_files_directory, exist_ok=True)
            
            # Define the destination path for the file
            empty_destination_path = os.path.join(empty_files_directory, file)
            
            # Remove the destination file if it already exists
            if os.path.exists(empty_destination_path):
                os.remove(empty_destination_path)
            
            print("Moving empty file:", file)
            safe_move(file_empty, empty_destination_path)
            now = time.time()
            os.utime(empty_destination_path, (now, now))

# Files to move: in STAGE_0_to_1 but not in UNPROCESSED, PROCESSING, or COMPLETED
raw_files = set(_expected_input_files(os.listdir(raw_directory)))
unprocessed_files = set(_expected_input_files(os.listdir(unprocessed_directory)))
processing_files = set(_expected_input_files(os.listdir(processing_directory)))
completed_files = set(_expected_input_files(os.listdir(completed_directory)))

files_to_move = raw_files - unprocessed_files - processing_files - completed_files

# Move files to UNPROCESSED ---------------------------------------------------------------
for file_name in files_to_move:
    src_path = os.path.join(raw_directory, file_name)
    dest_path = os.path.join(unprocessed_directory, file_name)
    try:
        safe_move(src_path, dest_path)
        now = time.time()
        os.utime(dest_path, (now, now))
        print(f"Move {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to move {file_name} to UNPROCESSED: {e}")
        error_dest_path = os.path.join(error_directory, file_name)
        try:
            safe_move(src_path, error_dest_path)
            now = time.time()
            os.utime(error_dest_path, (now, now))
            print(f"Moved {file_name} to ERROR directory after UNPROCESSED move failure.")
        except Exception as error_move_exc:
            raise RuntimeError(
                f"Unable to move {file_name} to either UNPROCESSED or ERROR directory."
            ) from error_move_exc

# Erase all files in the figure_directory -------------------------------------------------
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory) if os.path.exists(figure_directory) else []

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))

# Define input file path ------------------------------------------------------------------
input_file_config_path = os.path.join(config_file_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    print("Searching input configuration file:", input_file_config_path)
    
    # It is a csv
    try:
        input_file = pd.read_csv(input_file_config_path, skiprows=1)
    except pd.errors.EmptyDataError:
        input_file = pd.DataFrame()
        print("Input configuration file is empty.")

    if not input_file.empty:
        print("Input configuration file found and is not empty.")
        exists_input_file = True
    else:
        print("Input configuration file is empty.")
        exists_input_file = False
    
    # Print the head
    # print(input_file.head())
    
else:
    exists_input_file = False
    print("Input configuration file does not exist.")

sync_unprocessed_with_date_range(
    config=config,
    unprocessed_directory=base_directories["unprocessed_directory"],
    out_of_date_directory=base_directories["out_of_date_directory"],
    log_fn=print,
    station_id=station,
    master_config_root=config_root,
)
unprocessed_files = _expected_input_files(os.listdir(base_directories["unprocessed_directory"]))
processing_files = _expected_input_files(os.listdir(base_directories["processing_directory"]))
completed_files = _expected_input_files(os.listdir(base_directories["completed_directory"]))
date_ranges = load_date_ranges_from_config(
    config,
    station_id=station,
    master_config_root=config_root,
)
if date_ranges:
    processing_before = len(processing_files)
    completed_before = len(completed_files)
    processing_files = [
        name
        for name in processing_files
        if file_name_in_any_date_range(name, date_ranges)
    ]
    completed_files = [
        name
        for name in completed_files
        if file_name_in_any_date_range(name, date_ranges)
    ]
    skipped_processing = processing_before - len(processing_files)
    skipped_completed = completed_before - len(completed_files)
    if skipped_processing > 0:
        print(
            f"[DATE_RANGE] Ignoring {skipped_processing} out-of-range file(s) in PROCESSING_DIRECTORY.",
            force=True,
        )
    if skipped_completed > 0:
        print(
            f"[DATE_RANGE] Ignoring {skipped_completed} out-of-range file(s) in COMPLETED_DIRECTORY.",
            force=True,
        )
active_qa_retry_basenames: set[str] = set()
if process_only_qa_retry_files:
    active_qa_retry_basenames = load_active_qa_retry_basenames(
        station,
        repo_root=repo_root,
    )
    unprocessed_files = filter_filenames_by_qa_retry_basenames(
        unprocessed_files,
        active_qa_retry_basenames,
    )
    processing_files = filter_filenames_by_qa_retry_basenames(
        processing_files,
        active_qa_retry_basenames,
    )
    completed_files = filter_filenames_by_qa_retry_basenames(
        completed_files,
        active_qa_retry_basenames,
    )
    print(
        "[QA_ONLY] Active basenames="
        f"{len(active_qa_retry_basenames)} eligible files: "
        f"UNPROCESSED={len(unprocessed_files)} "
        f"PROCESSING={len(processing_files)} "
        f"COMPLETED={len(completed_files)}"
    )
last_file_test = bool(config.get("last_file_test", False))
keep_all_columns_output = _coerce_config_bool(
    config.get("keep_all_columns_output", False),
    default=False,
)

if user_file_selection:
    processing_file_path = user_file_path
    file_name = os.path.basename(user_file_path)
    if (
        process_only_qa_retry_files
        and canonical_processing_basename(file_name) not in active_qa_retry_basenames
    ):
        sys.exit(
            "[QA_ONLY] The selected input file is not present in the active QA retry list: "
            f"{canonical_processing_basename(file_name)}"
        )
else:
    if last_file_test:
        latest_unprocessed = select_latest_candidate(unprocessed_files, station)
        if latest_unprocessed:
            file_name = latest_unprocessed
            unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

            print(f"Processing the newest file in UNPROCESSED: {unprocessed_file_path}")
            print(f"Moving '{file_name}' to PROCESSING...")
            safe_move(unprocessed_file_path, processing_file_path)
            print(f"File moved to PROCESSING: {processing_file_path}")

        else:
            latest_processing = select_latest_candidate(processing_files, station)
            if latest_processing:
                file_name = latest_processing
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

                print(f"Processing the newest file already in PROCESSING:\n    {processing_file_path}")
                error_file_path = os.path.join(base_directories["error_directory"], file_name)
                print(f"File '{processing_file_path}' is already in PROCESSING. Moving it temporarily to ERROR for analysis...")
                safe_move(processing_file_path, error_file_path)
                processing_file_path = error_file_path
                print(f"File moved to ERROR: {processing_file_path}")

            elif complete_reanalysis and completed_files:
                latest_completed = select_latest_candidate(completed_files, station)
                if latest_completed:
                    file_name = latest_completed
                    processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                    completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

                    print(f"Reprocessing the newest file in COMPLETED: {completed_file_path}")
                    print(f"Moving '{completed_file_path}' to PROCESSING...")
                    safe_move(completed_file_path, processing_file_path)
                    print(f"File moved to PROCESSING: {processing_file_path}")
                else:
                    print("No files to process in COMPLETED after normalization.")
                    sys.exit(0)
            else:
                print("No files to process in UNPROCESSED, PROCESSING and decided to not reanalyze COMPLETED.")
                sys.exit(0)

    else:
        if unprocessed_files:
            print("Selecting a random file in UNPROCESSED...")
            file_name = random.choice(unprocessed_files)
            unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

            print(f"Moving '{file_name}' to PROCESSING...")
            safe_move(unprocessed_file_path, processing_file_path)
            print(f"File moved to PROCESSING: {processing_file_path}")

        elif processing_files:
            print("Selecting a random file in PROCESSING...")
            file_name = random.choice(processing_files)
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

            print(f"Processing the last file in PROCESSING: {processing_file_path}")
            error_file_path = os.path.join(base_directories["error_directory"], file_name)
            print(f"File '{processing_file_path}' is already in PROCESSING. Moving it temporarily to ERROR for analysis...")
            safe_move(processing_file_path, error_file_path)
            processing_file_path = error_file_path
            print(f"File moved to ERROR: {processing_file_path}")

        elif completed_files:
            if complete_reanalysis:
                print("Selecting a random file in COMPLETED...")
                file_name = random.choice(completed_files)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)

                print(f"Moving '{file_name}' to PROCESSING...")
                safe_move(completed_file_path, processing_file_path)
                print(f"File moved to PROCESSING: {processing_file_path}")
            else:
                print("No files to process in UNPROCESSED, PROCESSING and decided to not reanalyze COMPLETED.")
                sys.exit(0)

        else:
            print("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")
            sys.exit(0)

# This is for all cases
file_path = processing_file_path
if save_plots:
    os.makedirs(base_directories["figure_directory"], exist_ok=True)

the_filename = os.path.basename(file_path)
print(f"File to process: {the_filename}")

basename_no_ext, file_extension = os.path.splitext(the_filename)
# Take basename of IN_PATH without extension and witouth the 'listed_' prefix
basename_no_ext = the_filename.replace("fitted_", "").replace(".parquet", "")

print(f"File basename (no extension): {basename_no_ext}")
status_filename_base = basename_no_ext
status_execution_date = initialize_status_row(
    csv_path_status,
    filename_base=status_filename_base,
    completion_fraction=0.0,
)

simulated_z_positions, simulated_param_hash = resolve_simulated_z_positions(
    basename_no_ext,
    Path(base_directory),
    parquet_path=Path(file_path),
)

analysis_date = datetime.now().strftime("%Y-%m-%d")
print(f"Analysis date and time: {analysis_date}")

# Modify the time of the processing file to the current time so it looks fresh
now = time.time()
if os.path.exists(processing_file_path):
    os.utime(processing_file_path, (now, now))
else:
    print(
        f"Warning: processing file path not found for timestamp refresh: {processing_file_path}"
    )

# Check the station number in the datafile
try:
    file_station_number = int(basename_no_ext[3])  # 4th character (index 3)
    if file_station_number != int(station):
        print(f'File station number is: {file_station_number}, it does not match.')
        # Move the file to the ERROR directory
        error_file_path = os.path.join(base_directories["error_directory"], file_name)
        print(f"Moving file '{file_name}' to ERROR directory: {error_file_path}")
        process_file(file_path, error_file_path)
        sys.exit(f"File '{file_name}' does not belong to station {station}. Exiting.")
except ValueError:
    sys.exit(f"Invalid station number in file '{file_name}'. Exiting.")

if status_execution_date is not None:
    update_status_progress(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        completion_fraction=0.25,
        param_hash=str(simulated_param_hash) if simulated_param_hash else "",
    )

left_limit_time = pd.to_datetime("1-1-2000", format='%d-%m-%Y')
right_limit_time = pd.to_datetime("1-1-2100", format='%d-%m-%Y')

# if limit:
#     print(f'Taking the first {limit_number} rows.')

# Read the data file into a DataFrame
KEY = "df"

# Load dataframe
working_df = pd.read_parquet(file_path, engine="pyarrow")
working_df = working_df.rename(columns=lambda col: col.replace("_diff_", "_dif_"))
if "event_id" not in working_df.columns:
    print("Warning: 'event_id' missing in Task 5 input; reconstructing from current row order.")
    working_df.insert(0, "event_id", np.arange(len(working_df), dtype=np.int64))
charge_event_series, charge_event_source = resolve_charge_event_series(
    working_df,
    station_root=station_directory,
    basename_no_ext=basename_no_ext,
)
if charge_event_series is not None:
    working_df["charge_event"] = charge_event_series
    print(f"Recovered charge_event for TASK_5 using: {charge_event_source}")
else:
    print("[WARN] charge_event could not be reconstructed in TASK_5.")
print(f"Listed dataframe reloaded from: {file_path}")
# Ensure param_hash is persisted for downstream tasks.
if "param_hash" not in working_df.columns:
    working_df["param_hash"] = str(simulated_param_hash) if simulated_param_hash else ""
elif simulated_param_hash:
    param_series = working_df["param_hash"]
    missing_hash = param_series.isna()
    try:
        missing_hash |= param_series.astype(str).str.strip().eq("")
    except Exception as exc:
        print(f"Warning: param_hash validation fallback used due to error: {exc}")
    if missing_hash.any():
        working_df.loc[missing_hash, "param_hash"] = str(simulated_param_hash)
# print("Columns loaded from parquet:")
# for col in working_df.columns:
#     print(f" - {col}")

if create_debug_plots:
    main_cols: list[str] = []
    for i_plane in range(1, 5):
        main_cols.extend(
            [
                f"P{i_plane}_T_sum_final",
                f"P{i_plane}_T_dif_final",
                f"P{i_plane}_Q_sum_final",
                f"P{i_plane}_Q_dif_final",
                f"P{i_plane}_Y_final",
            ]
        )
    main_cols.extend(
        [
            col
            for col in ("raw_tt", "clean_tt", "cal_tt", "list_tt", "tracking_tt", "definitive_tt")
            if col in working_df.columns
        ]
    )
    main_cols = [col for col in main_cols if col in working_df.columns]
    if main_cols:
        debug_fig_idx = plot_debug_histograms(
            working_df,
            main_cols,
            thresholds=None,
            title=f"Task 5 incoming parquet: main columns [NON-TUNABLE] (station {station})",
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
            max_cols_per_fig=20,
        )

# Helper: compute trigger types based on non-zero charge columns
def compute_tt(df: pd.DataFrame, column_name: str, columns_map: dict[int, list[str]] | None = None) -> pd.DataFrame:
    """Compute trigger type based on planes with non-zero charge."""
    tt_str = pd.Series("", index=df.index, dtype="object")
    for plane in range(1, 5):
        if columns_map:
            charge_columns = [col for col in columns_map.get(plane, []) if col in df.columns]
        else:
            charge_columns = [
                col
                for col in [
                    f"Q{plane}_F_1",
                    f"Q{plane}_F_2",
                    f"Q{plane}_F_3",
                    f"Q{plane}_F_4",
                    f"Q{plane}_B_1",
                    f"Q{plane}_B_2",
                    f"Q{plane}_B_3",
                    f"Q{plane}_B_4",
                ]
                if col in df.columns
            ]
        if charge_columns:
            has_charge = df.loc[:, charge_columns].ne(0).any(axis=1)
            tt_str = tt_str.where(~has_charge, tt_str + str(plane))
    df.loc[:, column_name] = tt_str.replace("", "0").astype(int)
    return df

fit_tt_columns = {
    i_plane: [
        f"P{i_plane}_T_sum_final",
        f"P{i_plane}_T_dif_final",
        f"P{i_plane}_Q_sum_final",
        f"P{i_plane}_Q_dif_final",
        f"P{i_plane}_Y_final",
    ]
    for i_plane in range(1, 5)
}

# Prefer the corr_tt already provided by upstream steps.
CORR_TT_COLUMN = "corr_tt"
POST_TT_COLUMN = "post_tt"
global_variables = {}

TT_COUNT_VALUES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 12, 13, 14, 23, 24, 34, 123, 124, 134, 234, 1234
)

TASK5_CHANNEL_COMBINATION_RELATION_TYPES: tuple[str, ...] = (
    "self",
    "same_strip",
    "same_plane",
    "any",
)
TASK5_CHANNEL_COMBINATION_OBSERVABLES: tuple[tuple[str, str], ...] = (
    ("q_sum", "Q semisum"),
    ("q_dif", "Q semidifference"),
    ("t_sum", "T semisum"),
    ("t_dif", "T semidifference"),
)
TASK5_CHANNEL_COMBINATION_SCATTER_MAX_POINTS = 5000

def ensure_global_count_keys(prefixes: Iterable[str]) -> None:
    for prefix in prefixes:
        for tt_value in TT_COUNT_VALUES:
            global_variables.setdefault(f"{prefix}_{tt_value}_count", 0)


def _task5_channel_order_key(channel_key: tuple[int, str, int]) -> tuple[int, int, int]:
    plane, side, strip = channel_key
    return (plane, 0 if side == "F" else 1, strip)


def _task5_channel_relation_type(
    channel_a: tuple[int, str, int],
    channel_b: tuple[int, str, int],
) -> str:
    if channel_a == channel_b:
        return "self"
    if channel_a[0] == channel_b[0] and channel_a[2] == channel_b[2]:
        return "same_strip"
    if channel_a[0] == channel_b[0]:
        return "same_plane"
    return "any"


def collect_task5_channel_qt_map(
    df_input: pd.DataFrame,
) -> dict[tuple[int, str, int], dict[str, str]]:
    channel_map: dict[tuple[int, str, int], dict[str, str]] = {}
    for plane in range(1, 5):
        for side in ("F", "B"):
            for strip in range(1, 5):
                q_col = f"Q{plane}_{side}_{strip}"
                t_col = f"T{plane}_{side}_{strip}"
                if q_col in df_input.columns and t_col in df_input.columns:
                    channel_map[(plane, side, strip)] = {"Q": q_col, "T": t_col}
    return channel_map


def _iter_task5_channel_relation_pairs(
    channel_map: dict[tuple[int, str, int], dict[str, str]],
) -> Iterable[tuple[tuple[int, str, int], tuple[int, str, int], str]]:
    ordered_channels = sorted(channel_map, key=_task5_channel_order_key)
    for idx, channel_a in enumerate(ordered_channels):
        yield channel_a, channel_a, "self"
        for channel_b in ordered_channels[idx + 1 :]:
            yield channel_a, channel_b, _task5_channel_relation_type(channel_a, channel_b)


def _task5_channel_hist_range(
    values: np.ndarray,
    limits: tuple[float | None, float | None],
) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    lower_limit, upper_limit = limits
    if finite_values.size:
        low = float(np.nanpercentile(finite_values, 1))
        high = float(np.nanpercentile(finite_values, 99))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low = float(np.nanmin(finite_values))
            high = float(np.nanmax(finite_values))
    else:
        low, high = -1.0, 1.0
    if lower_limit is not None:
        low = min(low, float(lower_limit))
    if upper_limit is not None:
        high = max(high, float(upper_limit))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        center = float(low if np.isfinite(low) else 0.0)
        low, high = center - 1.0, center + 1.0
    span = high - low
    padding = 0.08 * span if span > 0 else 1.0
    return low - padding, high + padding


def _task5_load_channel_combination_relation_limits(
    task1_config: dict,
) -> dict[str, dict[str, tuple[float | None, float | None]]]:
    q_side_left_default = float(task1_config.get("Q_side_left_pre_cal_default", 0.0))
    q_side_right_default = float(task1_config.get("Q_side_right_pre_cal_default", 0.0))
    t_side_left_default = float(task1_config.get("T_side_left_pre_cal_default", 0.0))
    t_side_right_default = float(task1_config.get("T_side_right_pre_cal_default", 0.0))
    t_dif_threshold = float(task1_config.get("T_dif_pre_cal_threshold", 20.0))

    base_q_sum_left = float(
        task1_config.get(
            "channel_combination_q_sum_left",
            task1_config.get("plane_combination_q_sum_left", q_side_left_default),
        )
    )
    base_q_sum_right = float(
        task1_config.get(
            "channel_combination_q_sum_right",
            task1_config.get("plane_combination_q_sum_right", q_side_right_default),
        )
    )
    base_q_dif_threshold = float(
        task1_config.get(
            "channel_combination_q_dif_threshold",
            task1_config.get("plane_combination_q_dif_threshold", 200.0),
        )
    )
    base_t_sum_left = float(
        task1_config.get(
            "channel_combination_t_sum_left",
            task1_config.get("plane_combination_t_sum_left", t_side_left_default),
        )
    )
    base_t_sum_right = float(
        task1_config.get(
            "channel_combination_t_sum_right",
            task1_config.get("plane_combination_t_sum_right", t_side_right_default),
        )
    )
    base_t_dif_threshold = float(
        task1_config.get(
            "channel_combination_t_dif_threshold",
            task1_config.get("plane_combination_t_dif_threshold", t_dif_threshold),
        )
    )

    relation_limits: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    for relation_type in TASK5_CHANNEL_COMBINATION_RELATION_TYPES:
        q_sum_left = float(
            task1_config.get(f"channel_combination_{relation_type}_q_sum_left", base_q_sum_left)
        )
        q_sum_right = float(
            task1_config.get(f"channel_combination_{relation_type}_q_sum_right", base_q_sum_right)
        )
        q_dif_abs = float(
            task1_config.get(
                f"channel_combination_{relation_type}_q_dif_threshold",
                base_q_dif_threshold,
            )
        )
        t_sum_left = float(
            task1_config.get(f"channel_combination_{relation_type}_t_sum_left", base_t_sum_left)
        )
        t_sum_right = float(
            task1_config.get(f"channel_combination_{relation_type}_t_sum_right", base_t_sum_right)
        )
        t_dif_abs = float(
            task1_config.get(
                f"channel_combination_{relation_type}_t_dif_threshold",
                base_t_dif_threshold,
            )
        )
        relation_limits[relation_type] = {
            "q_sum": (q_sum_left, q_sum_right),
            "q_dif": (-q_dif_abs, q_dif_abs),
            "t_sum": (t_sum_left, t_sum_right),
            "t_dif": (-t_dif_abs, t_dif_abs),
        }
    return relation_limits


def _task5_parse_optional_top_n(raw_value: object, default: int | None) -> int | None:
    if raw_value in (None, "", "null", "None"):
        return None
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return default


def _task5_load_task1_channel_combination_settings() -> tuple[
    dict[str, dict[str, tuple[float | None, float | None]]],
    dict[str, int | None],
]:
    try:
        task1_config_path = (
            config_root
            / "STAGE_1"
            / "EVENT_DATA"
            / "STEP_1"
            / "TASK_1"
            / "config_task_1.yaml"
        )
        task1_parameter_config_path = (
            config_root
            / "STAGE_1"
            / "EVENT_DATA"
            / "STEP_1"
            / "TASK_1"
            / "config_parameters_task_1.csv"
        )
        task1_fallback_parameter_config_path = (
            config_root
            / "STAGE_1"
            / "EVENT_DATA"
            / "STEP_1"
            / "config_parameters.csv"
        )
        with task1_config_path.open("r", encoding="utf-8") as task1_config_file:
            task1_config = yaml.safe_load(task1_config_file) or {}
        task1_filter_parameter_config_path = task1_config_path.with_name(
            str(task1_config.get("filter_parameter_config_csv", "config_filter_parameters_task_1.csv"))
        )
        if task1_filter_parameter_config_path.exists():
            task1_config = update_config_with_parameters(
                task1_config,
                task1_filter_parameter_config_path,
                station,
            )
        task1_config = apply_step1_task_parameter_overrides(
            config_obj=task1_config,
            station_id=station,
            task_parameter_path=str(task1_parameter_config_path),
            fallback_parameter_path=str(task1_fallback_parameter_config_path),
            task_number=1,
            update_fn=update_config_with_parameters,
            log_fn=lambda *_args, **_kwargs: None,
        )
        base_top_n = _task5_parse_optional_top_n(
            task1_config.get("channel_combination_plot_top_n", 10),
            10,
        )
        top_n_by_relation = {
            relation_type: _task5_parse_optional_top_n(
                task1_config.get(f"channel_combination_plot_top_n_{relation_type}", base_top_n),
                base_top_n,
            )
            for relation_type in TASK5_CHANNEL_COMBINATION_RELATION_TYPES
        }
        return _task5_load_channel_combination_relation_limits(task1_config), top_n_by_relation
    except Exception as exc:
        print(
            "Warning: failed to load Task 1 channel-combination limits for Task 5 audit: "
            f"{exc}"
        )
        return (
            {
                relation_type: {
                    observable: (None, None)
                    for observable, _ in TASK5_CHANNEL_COMBINATION_OBSERVABLES
                }
                for relation_type in TASK5_CHANNEL_COMBINATION_RELATION_TYPES
            },
            {relation_type: 10 for relation_type in TASK5_CHANNEL_COMBINATION_RELATION_TYPES},
        )


def _task5_resolve_audit_tt_column(df_input: pd.DataFrame) -> str | None:
    for candidate in ("raw_tt", "clean_tt", "cal_tt", "list_tt", CORR_TT_COLUMN, "definitive_tt"):
        if candidate in df_input.columns:
            return candidate
    return None


def collect_task5_channel_combination_payload(
    df_input: pd.DataFrame,
    tt_column: str,
) -> pd.DataFrame:
    if tt_column not in df_input.columns:
        return pd.DataFrame(
            columns=["tt", "combo", "relation_type", "q_sum", "q_dif", "t_sum", "t_dif"]
        )

    channel_map = collect_task5_channel_qt_map(df_input)
    if not channel_map:
        return pd.DataFrame(
            columns=["tt", "combo", "relation_type", "q_sum", "q_dif", "t_sum", "t_dif"]
        )

    tt_series = df_input[tt_column].apply(normalize_tt_label).astype(str)
    payload_rows: list[pd.DataFrame] = []
    for channel_a, channel_b, relation_type in _iter_task5_channel_relation_pairs(channel_map):
        cols_a = channel_map[channel_a]
        cols_b = channel_map[channel_b]
        q_a = pd.to_numeric(df_input[cols_a["Q"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_b = pd.to_numeric(df_input[cols_b["Q"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_a = pd.to_numeric(df_input[cols_a["T"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_b = pd.to_numeric(df_input[cols_b["T"]], errors="coerce").fillna(0).to_numpy(dtype=float)

        if relation_type == "self":
            valid_mask = np.isfinite(q_a) & np.isfinite(t_a) & ((q_a != 0) | (t_a != 0))
            q_sum = q_a
            q_dif = np.zeros_like(q_a, dtype=float)
            t_sum = t_a
            t_dif = np.zeros_like(t_a, dtype=float)
        else:
            valid_mask = (
                np.isfinite(q_a)
                & np.isfinite(q_b)
                & np.isfinite(t_a)
                & np.isfinite(t_b)
                & (q_a != 0)
                & (q_b != 0)
                & (t_a != 0)
                & (t_b != 0)
            )
            q_sum = 0.5 * (q_a + q_b)
            q_dif = 0.5 * (q_a - q_b)
            t_sum = 0.5 * (t_a + t_b)
            t_dif = 0.5 * (t_a - t_b)

        if not np.any(valid_mask):
            continue

        valid_index = df_input.index[valid_mask]
        payload_rows.append(
            pd.DataFrame(
                {
                    "tt": tt_series.loc[valid_index].to_numpy(dtype=str),
                    "combo": (
                        f"P{channel_a[0]}S{channel_a[2]}{channel_a[1]}-"
                        f"P{channel_b[0]}S{channel_b[2]}{channel_b[1]}"
                    ),
                    "relation_type": relation_type,
                    "q_sum": q_sum[valid_mask],
                    "q_dif": q_dif[valid_mask],
                    "t_sum": t_sum[valid_mask],
                    "t_dif": t_dif[valid_mask],
                }
            )
        )

    if not payload_rows:
        return pd.DataFrame(
            columns=["tt", "combo", "relation_type", "q_sum", "q_dif", "t_sum", "t_dif"]
        )
    return pd.concat(payload_rows, ignore_index=True)


def plot_task5_channel_combination_filter_by_tt(
    df_input: pd.DataFrame,
    basename_no_ext_value: str,
    fig_idx_value: int,
    *,
    show_plots: bool,
    save_plots: bool,
    plot_list: list[str] | None,
) -> int:
    channel_map = collect_task5_channel_qt_map(df_input)
    if not channel_map:
        print(
            "Warning: Task 5 channel_combination_filter_by_tt requires the surviving "
            "channel Q*/T* columns to still be present. Re-run with "
            "keep_all_columns_output=true from Task 2 onward."
        )
        return fig_idx_value

    tt_column = _task5_resolve_audit_tt_column(df_input)
    if tt_column is None:
        print("Warning: Task 5 channel_combination_filter_by_tt found no TT column to group by.")
        return fig_idx_value

    relation_limits, top_n_by_relation = _task5_load_task1_channel_combination_settings()
    relation_label_map = {
        "self": "self",
        "same_strip": "same strip",
        "same_plane": "same plane",
        "any": "any",
    }
    observable_labels = {
        observable: label for observable, label in TASK5_CHANNEL_COMBINATION_OBSERVABLES
    }
    payload = collect_task5_channel_combination_payload(df_input, tt_column)
    tt_counts = (
        pd.to_numeric(df_input[tt_column], errors="coerce").fillna(0).astype(int).value_counts()
    )
    ordered_tts = [
        normalize_tt_label(tt_value)
        for tt_value in TT_COUNT_VALUES
        if tt_value >= 10 and int(tt_counts.get(int(tt_value), 0)) > 0
    ]
    if not ordered_tts:
        print(f"Warning: no {tt_column}>=10 populations available for Task 5 channel audit plot.")
        return fig_idx_value

    rng = np.random.default_rng(0)
    for tt_label in ordered_tts:
        for relation_type in TASK5_CHANNEL_COMBINATION_RELATION_TYPES:
            current_tt_all = payload.loc[
                (payload["tt"] == tt_label) & (payload["relation_type"] == relation_type)
            ].copy()
            if current_tt_all.empty:
                continue

            top_n = top_n_by_relation.get(relation_type)
            combo_counts = current_tt_all["combo"].value_counts()
            combo_labels = combo_counts.index.tolist() if top_n is None else combo_counts.index[:top_n].tolist()
            current_tt = current_tt_all.loc[current_tt_all["combo"].isin(combo_labels)].copy()
            if current_tt.empty:
                continue

            n_obs = len(TASK5_CHANNEL_COMBINATION_OBSERVABLES)
            fig, axes = plt.subplots(
                n_obs,
                n_obs,
                figsize=(2.4 * n_obs, 2.4 * n_obs),
                constrained_layout=True,
            )
            combo_color_map = {
                label: plt.get_cmap("turbo")(
                    0.08 + 0.84 * idx / max(1, len(combo_labels) - 1)
                )
                for idx, label in enumerate(combo_labels)
            }
            axis_ranges: dict[str, tuple[float, float]] = {}
            for observable_name, _observable_label in TASK5_CHANNEL_COMBINATION_OBSERVABLES:
                values = pd.to_numeric(
                    current_tt_all.get(observable_name, pd.Series(dtype=float)),
                    errors="coerce",
                ).to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                axis_ranges[observable_name] = _task5_channel_hist_range(
                    values,
                    relation_limits.get(relation_type, {}).get(observable_name, (None, None)),
                )

            any_panel_data = False
            for row_idx, (y_name, y_label) in enumerate(TASK5_CHANNEL_COMBINATION_OBSERVABLES):
                for col_idx, (x_name, x_label) in enumerate(TASK5_CHANNEL_COMBINATION_OBSERVABLES):
                    ax = axes[row_idx, col_idx]
                    if col_idx > row_idx:
                        ax.set_axis_off()
                        continue

                    x_low, x_high = axis_ranges[x_name]
                    if row_idx == col_idx:
                        bins = np.linspace(x_low, x_high, 60)
                        panel_has_data = False
                        for combo_label in combo_labels:
                            values = pd.to_numeric(
                                current_tt.loc[current_tt["combo"] == combo_label, x_name],
                                errors="coerce",
                            ).to_numpy(dtype=float)
                            values = values[np.isfinite(values)]
                            if not values.size:
                                continue
                            panel_has_data = True
                            ax.hist(
                                values,
                                bins=bins,
                                histtype="step",
                                linewidth=1.3,
                                alpha=0.95,
                                color=combo_color_map[combo_label],
                            )
                        if not panel_has_data:
                            ax.set_axis_off()
                            continue
                        any_panel_data = True
                        lower_limit, upper_limit = relation_limits.get(relation_type, {}).get(
                            x_name,
                            (None, None),
                        )
                        if lower_limit is not None:
                            ax.axvline(float(lower_limit), color="red", linestyle="--", linewidth=1.0)
                        if upper_limit is not None:
                            ax.axvline(float(upper_limit), color="red", linestyle="--", linewidth=1.0)
                        ax.set_xlim(x_low, x_high)
                        ax.set_yscale("log", nonpositive="clip")
                        ax.set_ylabel("Counts", fontsize=8)
                        ax.set_title(x_label, fontsize=9)
                    else:
                        y_low, y_high = axis_ranges[y_name]
                        panel_has_data = False
                        for combo_label in combo_labels:
                            combo_df = current_tt.loc[current_tt["combo"] == combo_label]
                            x_values = pd.to_numeric(combo_df[x_name], errors="coerce").to_numpy(dtype=float)
                            y_values = pd.to_numeric(combo_df[y_name], errors="coerce").to_numpy(dtype=float)
                            mask = np.isfinite(x_values) & np.isfinite(y_values)
                            if not np.any(mask):
                                continue
                            x_values = x_values[mask]
                            y_values = y_values[mask]
                            if x_values.size > TASK5_CHANNEL_COMBINATION_SCATTER_MAX_POINTS:
                                selection = rng.choice(
                                    x_values.size,
                                    size=TASK5_CHANNEL_COMBINATION_SCATTER_MAX_POINTS,
                                    replace=False,
                                )
                                x_values = x_values[selection]
                                y_values = y_values[selection]
                            panel_has_data = True
                            ax.scatter(
                                x_values,
                                y_values,
                                s=6,
                                alpha=0.18,
                                color=combo_color_map[combo_label],
                                edgecolors="none",
                                rasterized=True,
                            )
                        if not panel_has_data:
                            ax.set_axis_off()
                            continue
                        any_panel_data = True
                        x_limits = relation_limits.get(relation_type, {}).get(x_name, (None, None))
                        y_limits = relation_limits.get(relation_type, {}).get(y_name, (None, None))
                        if x_limits[0] is not None:
                            ax.axvline(float(x_limits[0]), color="red", linestyle="--", linewidth=0.9)
                        if x_limits[1] is not None:
                            ax.axvline(float(x_limits[1]), color="red", linestyle="--", linewidth=0.9)
                        if y_limits[0] is not None:
                            ax.axhline(float(y_limits[0]), color="red", linestyle="--", linewidth=0.9)
                        if y_limits[1] is not None:
                            ax.axhline(float(y_limits[1]), color="red", linestyle="--", linewidth=0.9)
                        ax.set_xlim(x_low, x_high)
                        ax.set_ylim(y_low, y_high)

                    if row_idx == n_obs - 1:
                        ax.set_xlabel(x_label, fontsize=8)
                    else:
                        ax.set_xticklabels([])
                    if col_idx == 0 and row_idx != col_idx:
                        ax.set_ylabel(y_label, fontsize=8)
                    elif col_idx != 0:
                        ax.set_yticklabels([])
                    ax.grid(alpha=0.12)
                    ax.tick_params(labelsize=6)

            if not any_panel_data:
                plt.close(fig)
                continue

            style_handles = [
                mpl.lines.Line2D(
                    [0],
                    [0],
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    label="Task 1 configured limits",
                )
            ]
            combo_handles = [
                mpl.lines.Line2D(
                    [0],
                    [0],
                    color=combo_color_map[label],
                    linestyle="-",
                    linewidth=1.6,
                    label=f"{label} (N={int(combo_counts.get(label, 0))})",
                )
                for label in combo_labels
            ]
            fig.legend(
                handles=style_handles + combo_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=min(5, max(2, len(combo_handles) + 1)),
                fontsize=6,
                frameon=False,
                handlelength=1.8,
                columnspacing=0.8,
            )
            combo_mode_label = "all pairs" if top_n is None else f"top {int(top_n)} pairs"
            fig.suptitle(
                f"Task 5 channel-combination state by {tt_column} {tt_label}\n"
                f"{basename_no_ext_value} · {relation_label_map[relation_type]} · "
                f"{combo_mode_label} · post-filter surviving values only",
                fontsize=11,
                y=0.94,
            )
            if save_plots:
                final_filename = (
                    f"{fig_idx_value}_channel_combination_filter_by_tt_"
                    f"{relation_type}_TT_{tt_label}.png"
                )
                fig_idx_value += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                if plot_list is not None:
                    plot_list.append(save_fig_path)
                save_plot_figure(
                    save_fig_path,
                    fig=fig,
                    format="png",
                    dpi=150,
                    alias="channel_combination_filter_by_tt",
                )
            if show_plots:
                plt.show()
            plt.close(fig)

    return fig_idx_value

if simulated_param_hash:
    global_variables["param_hash"] = simulated_param_hash

FILTER_METRIC_NAMES: tuple[str, ...] = (
    "total_rows_removed_pct",
    "data_purity_percentage",
    "all_components_zero_rows_removed_pct",
    "corr_tt_lt_10_rows_removed_pct",
)

filter_metrics: dict[str, float] = {}

def record_filter_metric(name: str, removed: float, total: float) -> None:
    """Record percentage removed for a filter."""
    pct = 0.0 if total == 0 else 100.0 * float(removed) / float(total)
    filter_metrics[name] = round(pct, 4)
    print(f"[filter-metrics] {name}: removed {removed} of {total} ({pct:.2f}%)")

# Keep fit_tt from Task 4 when present; compute only if missing.
if "fit_tt" not in working_df.columns:
    working_df = compute_tt(working_df, "fit_tt", fit_tt_columns)
else:
    working_df.loc[:, "fit_tt"] = (
        pd.to_numeric(working_df["fit_tt"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

# Prefer corr_tt from upstream; otherwise fall back to definitive_tt/list_tt.
if CORR_TT_COLUMN in working_df.columns:
    working_df.loc[:, CORR_TT_COLUMN] = (
        pd.to_numeric(working_df[CORR_TT_COLUMN], errors="coerce")
        .fillna(0)
        .astype(int)
    )
elif "definitive_tt" in working_df.columns:
    print("Warning: corr_tt missing; using definitive_tt for filtering.")
    working_df.loc[:, CORR_TT_COLUMN] = (
        pd.to_numeric(working_df["definitive_tt"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
elif "list_tt" in working_df.columns:
    print("Warning: corr_tt missing; using list_tt for filtering.")
    working_df.loc[:, CORR_TT_COLUMN] = (
        pd.to_numeric(working_df["list_tt"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
else:
    corr_tt_columns = {
        i_plane: [
            f"P{i_plane}_T_sum_final",
            f"P{i_plane}_T_dif_final",
            f"P{i_plane}_Q_sum_final",
            f"P{i_plane}_Q_dif_final",
            f"P{i_plane}_Y_final",
        ]
        for i_plane in range(1, 5)
    }
    print("Warning: corr_tt missing; computing from available charge columns.")
    working_df = compute_tt(working_df, CORR_TT_COLUMN, corr_tt_columns)

fit_tt_counts_initial = working_df["fit_tt"].value_counts()
for tt_value, count in fit_tt_counts_initial.items():
    tt_label = normalize_tt_label(tt_value)
    global_variables[f"fit_tt_{tt_label}_count"] = int(count)

working_df.loc[:, POST_TT_COLUMN] = (
    pd.to_numeric(working_df[CORR_TT_COLUMN], errors="coerce")
    .fillna(0)
    .astype(int)
)

post_tt_counts_initial = working_df[POST_TT_COLUMN].value_counts()
for tt_value, count in post_tt_counts_initial.items():
    tt_label = normalize_tt_label(tt_value)
    global_variables[f"{POST_TT_COLUMN}_{tt_label}_count"] = int(count)

fig_idx, plot_list = ensure_plot_state(globals())
if task5_plot_enabled("channel_combination_filter_by_tt"):
    print("----------- Task 1 channel-combination audit in Task 5 ------------")
    fig_idx = plot_task5_channel_combination_filter_by_tt(
        working_df,
        basename_no_ext,
        fig_idx,
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list,
    )

# Change 'Time' column to 'datetime' ------------------------------------------
if 'Time' in working_df.columns:
    working_df.rename(columns={'Time': 'datetime'}, inplace=True)
else:
    print("Column 'datetime' not found in DataFrame!")

# Original number of events
original_number_of_events = len(working_df)
if status_execution_date is not None:
    update_status_progress(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        completion_fraction=0.5,
        param_hash=str(global_variables.get("param_hash", "")),
    )

# Round execution time to seconds and format it in YYYY-MM-DD_HH.MM.SS
execution_time = str(start_execution_time_counting).split('.')[0]  # Remove microseconds
print("Execution time is:", execution_time)

ITINERARY_FILE_PATH = Path(
    f"{home_path}/DATAFLOW_v3/MASTER/CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1/TASK_2/TIME_CALIBRATION_ITINERARIES/itineraries.csv"
)

fast_mode = False
debug_mode = False
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
(
    plot_mode,
    create_plots,
    create_essential_plots,
    save_plots,
    create_pdf,
    show_plots,
    create_debug_plots,
) = resolve_step1_plot_options(config)
apply_task5_plot_catalog_modes()

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering

# Time calibration

# Y position

# RPC variables

# Alternative

# TimTrack

# Validation

# Pre-cal Front & Back
T_side_left_pre_cal_debug = config.get("T_side_left_pre_cal_debug", -500)
T_side_right_pre_cal_debug = config.get("T_side_right_pre_cal_debug", 500)

T_side_left_pre_cal_default = config.get("T_side_left_pre_cal_default", -200)
T_side_right_pre_cal_default = config.get("T_side_right_pre_cal_default", -100)

T_side_left_pre_cal_ST = config.get("T_side_left_pre_cal_ST", -200)
T_side_right_pre_cal_ST = config.get("T_side_right_pre_cal_ST", -50)

# Pre-cal Sum & Diff

# Post-calibration

# Once calculated the RPC variables
T_sum_RPC_left = _task5_config_float(
    config,
    "T_sum_RPC_left",
    "plane_combination_plane_t_sum_sum_left",
    "plane_combination_same_plane_t_sum_sum_left",
    "plane_combination_self_t_sum_sum_left",
    default=-25.0,
)
T_sum_RPC_right = _task5_config_float(
    config,
    "T_sum_RPC_right",
    "plane_combination_plane_t_sum_sum_right",
    "plane_combination_same_plane_t_sum_sum_right",
    "plane_combination_self_t_sum_sum_right",
    default=25.0,
)

# Alternative fitter filter
det_pos_filter = config["det_pos_filter"]
det_theta_left_filter = config["det_theta_left_filter"]
det_theta_right_filter = config["det_theta_right_filter"]
det_phi_filter_abs = abs(float(config.get("det_phi_filter_abs", config.get("det_phi_right_filter", 3.141592))))
det_phi_right_filter = det_phi_filter_abs
det_phi_left_filter = -det_phi_filter_abs
det_slowness_filter_left = config["det_slowness_filter_left"]
det_slowness_filter_right = config["det_slowness_filter_right"]

# TimTrack filter

# Fitting comparison

# Calibrations

# Pedestal charge calibration

# Front-back charge

# Variables to modify
beta = config.get("beta", 1)
strip_speed_factor_of_c = config.get("strip_speed_factor_of_c", 0.666666667)

# X
strip_length = config.get("strip_length", 300)
narrow_strip = config.get("narrow_strip", 63)
wide_strip = config.get("wide_strip", 98)

# Timtrack parameters
anc_std = config.get("anc_std", 0.075)

n_planes_timtrack = config.get("n_planes_timtrack", 4)

# Plotting options
T_clip_min_debug = config.get("T_clip_min_debug", -500)
T_clip_max_debug = config.get("T_clip_max_debug", 500)
Q_clip_min_debug = config.get("Q_clip_min_debug", -500)
Q_clip_max_debug = config.get("Q_clip_max_debug", 500)
num_bins_debug = config.get("num_bins_debug", 100)

T_clip_min_default = config.get("T_clip_min_default", -300)
T_clip_max_default = config.get("T_clip_max_default", 100)
Q_clip_min_default = config.get("Q_clip_min_default", 0)
Q_clip_max_default = config.get("Q_clip_max_default", 500)
num_bins_default = config.get("num_bins_default", 100)

T_clip_min_ST = config.get("T_clip_min_ST", -300)
T_clip_max_ST = config.get("T_clip_max_ST", 100)
Q_clip_min_ST = config.get("Q_clip_min_ST", 0)
Q_clip_max_ST = config.get("Q_clip_max_ST", 500)

# -----------------------------------------------------------------------------
# Some variables that define the analysis, define a dictionary with the variables:
# 'purity_of_data', etc.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Variables to not touch unless necessary -------------------------------------
# -----------------------------------------------------------------------------
Q_sum_color = 'orange'
Q_dif_color = 'red'
T_sum_color = 'blue'
T_dif_color = 'green'

pos_filter = det_pos_filter
t0_left_filter = T_sum_RPC_left
t0_right_filter = T_sum_RPC_right
slowness_filter_left = det_slowness_filter_left
slowness_filter_right = det_slowness_filter_right

theta_left_filter = det_theta_left_filter
theta_right_filter = det_theta_right_filter
phi_left_filter = det_phi_left_filter
phi_right_filter = det_phi_right_filter

if create_debug_plots:
    def _emit_param_debug(param_label, columns, thresholds, y_scale="log"):
        cols_present = [col for col in columns if col in working_df.columns]
        if not cols_present:
            return
        debug_thresholds = {col: thresholds for col in cols_present}
        cols_present.sort()
        global debug_fig_idx
        debug_fig_idx = plot_debug_histograms(
            working_df,
            cols_present,
            debug_thresholds,
            title=f"Task 5 pre-filter: {param_label} [tunable] (station {station})",
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
            y_scale=y_scale,
        )

    _emit_param_debug(
        "det_pos_filter",
        ["det_x", "det_y", "x", "y"],
        [-det_pos_filter, det_pos_filter],
    )
    _emit_param_debug(
        "slowness_filter_left/right",
        ["det_s", "s"],
        [slowness_filter_left, slowness_filter_right],
    )
    _emit_param_debug(
        "theta_left/right_filter",
        ["det_theta", "theta"],
        [theta_left_filter, theta_right_filter],
        y_scale="linear",
    )
    _emit_param_debug(
        "phi_left/right_filter",
        ["det_phi", "phi"],
        [phi_left_filter, phi_right_filter],
    )
    _emit_param_debug(
        "t0_left/right_filter",
        ["t0", "det_t0", "tim_t0"],
        [t0_left_filter, t0_right_filter],
    )

fig_idx, plot_list = ensure_plot_state(globals())

if False:
    print('Working in fast mode.')

if False:
    print('Working in debug mode.')

if False:
    T_F_left_pre_cal = T_side_left_pre_cal_debug
    T_F_right_pre_cal = T_side_right_pre_cal_debug

    T_B_left_pre_cal = T_side_left_pre_cal_debug
    T_B_right_pre_cal = T_side_right_pre_cal_debug

else:
    T_F_left_pre_cal = T_side_left_pre_cal_default  #-130
    T_F_right_pre_cal = T_side_right_pre_cal_default

    T_B_left_pre_cal = T_side_left_pre_cal_default
    T_B_right_pre_cal = T_side_right_pre_cal_default

T_F_left_pre_cal_ST = T_side_left_pre_cal_ST  #-115
T_F_right_pre_cal_ST = T_side_right_pre_cal_ST
T_B_left_pre_cal_ST = T_side_left_pre_cal_ST
T_B_right_pre_cal_ST = T_side_right_pre_cal_ST

# Y ---------------------------------------------------------------------------
y_widths = [np.array([wide_strip, wide_strip, wide_strip, narrow_strip]), 
            np.array([narrow_strip, wide_strip, wide_strip, wide_strip])]

y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_P1_and_P3 = y_widths[0]
y_width_P2_and_P4 = y_widths[1]
y_pos_P1_and_P3 = y_pos(y_width_P1_and_P3)
y_pos_P2_and_P4 = y_pos(y_width_P2_and_P4)
total_width = np.sum(y_width_P1_and_P3)

c_mm_ns = c/1000000
print(c_mm_ns)

# Miscelanous ----------------------------
muon_speed = beta * c_mm_ns
strip_speed = strip_speed_factor_of_c * c_mm_ns # 200 mm/ns
tdiff_to_x = strip_speed # Factor to transform t_diff to X

# Not-Hardcoded
vc    = beta * c_mm_ns # mm/ns
sc    = 1/vc
ss    = 1/strip_speed # slowness of the signal in the strip
nplan = n_planes_timtrack
lenx  = strip_length
anc_sx = tdiff_to_x * anc_std # 2 cm

if False:
    T_clip_min = T_clip_min_debug
    T_clip_max = T_clip_max_debug
    Q_clip_min = Q_clip_min_debug
    Q_clip_max = Q_clip_max_debug
    num_bins = num_bins_debug
else:
    T_clip_min = T_clip_min_default
    T_clip_max = T_clip_max_default
    Q_clip_min = Q_clip_min_default
    Q_clip_max = Q_clip_max_default
    num_bins = num_bins_default

T_clip_min_ST = T_clip_min_ST
T_clip_max_ST = T_clip_max_ST
Q_clip_min_ST = Q_clip_min_ST
Q_clip_max_ST = Q_clip_max_ST

self_trigger = bool(config.get("self_trigger", False))

fast_mode = False
debug_mode = False
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
(
    plot_mode,
    create_plots,
    create_essential_plots,
    save_plots,
    create_pdf,
    show_plots,
    create_debug_plots,
) = resolve_step1_plot_options(config)
apply_task5_plot_catalog_modes()

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering

# Time calibration

# Y position

# RPC variables

# Alternative

# TimTrack

# Validation

# Pre-cal Front & Back
T_side_left_pre_cal_debug = config.get("T_side_left_pre_cal_debug", -500)
T_side_right_pre_cal_debug = config.get("T_side_right_pre_cal_debug", 500)

T_side_left_pre_cal_default = config.get("T_side_left_pre_cal_default", -200)
T_side_right_pre_cal_default = config.get("T_side_right_pre_cal_default", -100)

T_side_left_pre_cal_ST = config.get("T_side_left_pre_cal_ST", -200)
T_side_right_pre_cal_ST = config.get("T_side_right_pre_cal_ST", -50)

# Pre-cal Sum & Diff

# Post-calibration

# Once calculated the RPC variables
T_sum_RPC_left = _task5_config_float(
    config,
    "T_sum_RPC_left",
    "plane_combination_plane_t_sum_sum_left",
    "plane_combination_same_plane_t_sum_sum_left",
    "plane_combination_self_t_sum_sum_left",
    default=-25.0,
)
T_sum_RPC_right = _task5_config_float(
    config,
    "T_sum_RPC_right",
    "plane_combination_plane_t_sum_sum_right",
    "plane_combination_same_plane_t_sum_sum_right",
    "plane_combination_self_t_sum_sum_right",
    default=25.0,
)

# Alternative fitter filter
det_pos_filter = config["det_pos_filter"]
det_theta_left_filter = config["det_theta_left_filter"]
det_theta_right_filter = config["det_theta_right_filter"]
det_phi_filter_abs = abs(float(config.get("det_phi_filter_abs", config.get("det_phi_right_filter", 3.141592))))
det_phi_right_filter = det_phi_filter_abs
det_phi_left_filter = -det_phi_filter_abs
det_slowness_filter_left = config["det_slowness_filter_left"]
det_slowness_filter_right = config["det_slowness_filter_right"]

# TimTrack filter

# Fitting comparison

# Calibrations

# Pedestal charge calibration

# Front-back charge

# Variables to modify
beta = config.get("beta", 1)
strip_speed_factor_of_c = config.get("strip_speed_factor_of_c", 0.666666667)

# X
strip_length = config.get("strip_length", 300)
narrow_strip = config.get("narrow_strip", 63)
wide_strip = config.get("wide_strip", 98)

# Timtrack parameters
anc_std = config.get("anc_std", 0.075)

n_planes_timtrack = config.get("n_planes_timtrack", 4)

# Plotting options
T_clip_min_debug = config.get("T_clip_min_debug", -500)
T_clip_max_debug = config.get("T_clip_max_debug", 500)
Q_clip_min_debug = config.get("Q_clip_min_debug", -500)
Q_clip_max_debug = config.get("Q_clip_max_debug", 500)
num_bins_debug = config.get("num_bins_debug", 100)

T_clip_min_default = config.get("T_clip_min_default", -300)
T_clip_max_default = config.get("T_clip_max_default", 100)
Q_clip_min_default = config.get("Q_clip_min_default", 0)
Q_clip_max_default = config.get("Q_clip_max_default", 500)
num_bins_default = config.get("num_bins_default", 100)

T_clip_min_ST = config.get("T_clip_min_ST", -300)
T_clip_max_ST = config.get("T_clip_max_ST", 100)
Q_clip_min_ST = config.get("Q_clip_min_ST", 0)
Q_clip_max_ST = config.get("Q_clip_max_ST", 500)

# -----------------------------------------------------------------------------
# Some variables that define the analysis, define a dictionary with the variables:
# 'purity_of_data', etc.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Variables to not touch unless necessary -------------------------------------
# -----------------------------------------------------------------------------
Q_sum_color = 'orange'
Q_dif_color = 'red'
T_sum_color = 'blue'
T_dif_color = 'green'

pos_filter = det_pos_filter
t0_left_filter = T_sum_RPC_left
t0_right_filter = T_sum_RPC_right
slowness_filter_left = det_slowness_filter_left
slowness_filter_right = det_slowness_filter_right

theta_left_filter = det_theta_left_filter
theta_right_filter = det_theta_right_filter
phi_left_filter = det_phi_left_filter
phi_right_filter = det_phi_right_filter

fig_idx, plot_list = ensure_plot_state(globals())

if False:
    print('Working in fast mode.')

if False:
    print('Working in debug mode.')

if False:
    T_F_left_pre_cal = T_side_left_pre_cal_debug
    T_F_right_pre_cal = T_side_right_pre_cal_debug

    T_B_left_pre_cal = T_side_left_pre_cal_debug
    T_B_right_pre_cal = T_side_right_pre_cal_debug

else:
    T_F_left_pre_cal = T_side_left_pre_cal_default  #-130
    T_F_right_pre_cal = T_side_right_pre_cal_default

    T_B_left_pre_cal = T_side_left_pre_cal_default
    T_B_right_pre_cal = T_side_right_pre_cal_default

T_F_left_pre_cal_ST = T_side_left_pre_cal_ST  #-115
T_F_right_pre_cal_ST = T_side_right_pre_cal_ST
T_B_left_pre_cal_ST = T_side_left_pre_cal_ST
T_B_right_pre_cal_ST = T_side_right_pre_cal_ST

# Y ---------------------------------------------------------------------------
y_widths = [np.array([wide_strip, wide_strip, wide_strip, narrow_strip]), 
            np.array([narrow_strip, wide_strip, wide_strip, wide_strip])]

y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_P1_and_P3 = y_widths[0]
y_width_P2_and_P4 = y_widths[1]
y_pos_P1_and_P3 = y_pos(y_width_P1_and_P3)
y_pos_P2_and_P4 = y_pos(y_width_P2_and_P4)
total_width = np.sum(y_width_P1_and_P3)

c_mm_ns = c/1000000
print(c_mm_ns)

# Miscelanous ----------------------------
muon_speed = beta * c_mm_ns
strip_speed = strip_speed_factor_of_c * c_mm_ns # 200 mm/ns
tdiff_to_x = strip_speed # Factor to transform t_diff to X

# Not-Hardcoded
vc    = beta * c_mm_ns # mm/ns
sc    = 1/vc
ss    = 1/strip_speed # slowness of the signal in the strip
nplan = n_planes_timtrack
lenx  = strip_length
anc_sx = tdiff_to_x * anc_std # 2 cm

if False:
    T_clip_min = T_clip_min_debug
    T_clip_max = T_clip_max_debug
    Q_clip_min = Q_clip_min_debug
    Q_clip_max = Q_clip_max_debug
    num_bins = num_bins_debug
else:
    T_clip_min = T_clip_min_default
    T_clip_max = T_clip_max_default
    Q_clip_min = Q_clip_min_default
    Q_clip_max = Q_clip_max_default
    num_bins = num_bins_default

T_clip_min_ST = T_clip_min_ST
T_clip_max_ST = T_clip_max_ST
Q_clip_min_ST = Q_clip_min_ST
Q_clip_max_ST = Q_clip_max_ST

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

# Note that the middle between start and end time could also be taken. This is for calibration storage.
if "datetime" in working_df.columns:
    datetime_series = pd.to_datetime(working_df["datetime"], errors="coerce").dropna()
else:
    datetime_series = pd.Series(dtype="datetime64[ns]")
if datetime_series.empty:
    print(
        f"Warning: No valid datetime rows found in {the_filename}; moving file to ERROR and skipping."
    )
    if not user_file_selection:
        error_file_path = os.path.join(base_directories["error_directory"], the_filename)
        print(f"Moving file '{the_filename}' to ERROR directory: {error_file_path}")
        process_file(file_path, error_file_path)
    if status_execution_date is not None:
        update_status_progress(
            csv_path_status,
            filename_base=status_filename_base,
            execution_date=status_execution_date,
            completion_fraction=1.0,
            param_hash=str(global_variables.get("param_hash", "")),
        )
    sys.exit(0)

datetime_value = datetime_series.iloc[0]
end_datetime_value = datetime_series.iloc[-1]

if self_trigger:
    print(self_trigger_df)
    if "datetime" in self_trigger_df.columns:
        datetime_series_st = pd.to_datetime(self_trigger_df["datetime"], errors="coerce").dropna()
    else:
        datetime_series_st = pd.Series(dtype="datetime64[ns]")
    if datetime_series_st.empty:
        print("Warning: Self-trigger dataframe has no valid datetime values; skipping self-trigger timestamp suffix.")
    else:
        datetime_value_st = datetime_series_st.iloc[0]
        end_datetime_value_st = datetime_series_st.iloc[-1]
        datetime_str_st = str(datetime_value_st)
        save_filename_suffix_st = datetime_str_st.replace(' ', "_").replace(':', ".").replace('-', ".")

start_time = datetime_value
end_time = end_datetime_value
datetime_str = str(datetime_value)
save_filename_suffix = datetime_str.replace(' ', "_").replace(':', ".").replace('-', ".")

_prof["s_data_read_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print(f"------------- Starting date is {save_filename_suffix} -------------------") # This is longer so it displays nicely
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Defining the directories that will store the data
save_full_filename = f"full_list_events_{save_filename_suffix}.txt"
save_filename = f"list_events_{save_filename_suffix}.txt"
save_pdf_filename = f"mingo{str(station).zfill(2)}_task5_{save_filename_suffix}.pdf"

save_pdf_path = os.path.join(base_directories["pdf_directory"], save_pdf_filename)

fig_idx, plot_list = ensure_plot_state(globals())
if (
    (create_essential_plots or create_plots)
    and task5_plot_enabled("charge_fit_tt_1234_per_plane_and_total")
):
    plane_charge_df = pd.DataFrame(index=working_df.index)
    plane_charge_columns: list[str] = []
    plane_charge_titles: list[str] = []
    fallback_notes: list[str] = []

    charge_candidates = [
        col
        for col in working_df.columns
        if re.search(r"^(P[1-4]_Q_sum(?:_.*)?|Q_P[1-4]s[1-4])$", col)
    ]
    if charge_candidates:
        print(
            "Task 5: detected charge-related columns in input dataframe: "
            + ", ".join(charge_candidates)
        )
    else:
        possible_charge_cols = [
            col
            for col in working_df.columns
            if re.search(r"Q|q", col)
            and ("sum" in col.lower() or "charge" in col.lower() or "q_" in col.lower())
        ]
        print(
            "Task 5: no exact charge-summary columns found; possible charge-like columns: "
            + ", ".join(possible_charge_cols)
        )

    for i_plane in range(1, 5):
        sum_col = f"P{i_plane}_Q_sum_final"
        if sum_col in working_df.columns:
            plane_charge_df.loc[:, sum_col] = pd.to_numeric(
                working_df[sum_col], errors="coerce"
            )
            plane_charge_columns.append(sum_col)
            plane_charge_titles.append(f"Plane {i_plane} Q_sum_final")
            continue

        alt_sum_cols = [
            col
            for col in working_df.columns
            if re.fullmatch(fr"P{i_plane}_Q_sum(?:_.*)?", col)
        ]
        if alt_sum_cols:
            alt_sum_cols = [c for c in alt_sum_cols if c != sum_col]
            best_alt = [c for c in alt_sum_cols if c.endswith("_final")] or alt_sum_cols
            alt_col = best_alt[0]
            plane_charge_df.loc[:, alt_col] = pd.to_numeric(
                working_df[alt_col], errors="coerce"
            )
            plane_charge_columns.append(alt_col)
            plane_charge_titles.append(f"Plane {i_plane} {alt_col}")
            fallback_notes.append(f"{sum_col} from {alt_col}")
            continue

        strip_cols = [
            f"Q_P{i_plane}s{i_strip}"
            for i_strip in range(1, 5)
            if f"Q_P{i_plane}s{i_strip}" in working_df.columns
        ]
        if not strip_cols:
            continue

        fallback_col = f"P{i_plane}_Q_sum_from_strips"
        strip_values = working_df[strip_cols].apply(pd.to_numeric, errors="coerce")
        # Keep all-NaN rows as NaN so missing charge data is not misrepresented as 0.
        plane_charge_df.loc[:, fallback_col] = strip_values.sum(axis=1, min_count=1)
        plane_charge_columns.append(fallback_col)
        plane_charge_titles.append(f"Plane {i_plane} Q sum from strips")
        fallback_notes.append(f"P{i_plane}={' + '.join(strip_cols)}")

    if plane_charge_columns:
        if fallback_notes:
            print(
                "Warning: Task 5 charge summary is using fallback charge columns: "
                + ", ".join(fallback_notes)
            )

        plane_charge_df["detector_total_charge"] = plane_charge_df[plane_charge_columns].sum(
            axis=1,
            min_count=1,
        )

        fit_tt_values = pd.to_numeric(working_df["fit_tt"], errors="coerce")
        df_charge_1234 = plane_charge_df.loc[
            fit_tt_values == 1234,
            plane_charge_columns + ["detector_total_charge"],
        ].copy()
        print(
            f"Task 5: preparing charge summary for fit_tt=1234 with {len(df_charge_1234)} matching rows"
        )

        if not df_charge_1234.empty:
            for column_name in plane_charge_columns + ["detector_total_charge"]:
                df_charge_1234.loc[:, column_name] = pd.to_numeric(
                    df_charge_1234[column_name], errors="coerce"
                )
            df_charge_1234 = df_charge_1234.replace([np.inf, -np.inf], np.nan)
            df_charge_1234["event_total_charge"] = df_charge_1234[
                plane_charge_columns
            ].sum(axis=1, min_count=1)
        else:
            df_charge_1234 = pd.DataFrame(
                columns=plane_charge_columns + ["detector_total_charge", "event_total_charge"]
            )

        fig, axes = plt.subplots(
            5,
            2,
            figsize=(14, 18),
            constrained_layout=True,
            sharex="col",
        )
        plot_columns = plane_charge_columns + ["event_total_charge"]
        plot_titles = plane_charge_titles + ["Total event charge"]

        for row_idx, (column_name, panel_title) in enumerate(zip(plot_columns, plot_titles)):
            zoom_axis = axes[row_idx, 0]
            full_axis = axes[row_idx, 1]
            for axis, zoomed, title_suffix in [
                (zoom_axis, True, " (0-40 zoom)"),
                (full_axis, False, ""),
            ]:
                if df_charge_1234.empty:
                    axis.text(
                        0.5,
                        0.5,
                        "No fit_tt=1234 data available",
                        ha="center",
                        va="center",
                    )
                    axis.set_title(panel_title + title_suffix)
                    axis.grid(alpha=0.25, which="both")
                    axis.set_yscale("log")
                    if zoomed:
                        axis.set_xlim(0, 40)
                    continue

                values = pd.to_numeric(df_charge_1234[column_name], errors="coerce")
                values = values[np.isfinite(values)]
                if values.empty:
                    axis.text(0.5, 0.5, "No finite values", ha="center", va="center")
                    axis.set_title(panel_title + title_suffix)
                    axis.grid(alpha=0.25, which="both")
                    axis.set_yscale("log")
                    if zoomed:
                        axis.set_xlim(0, 40)
                    continue

                if zoomed:
                    zoom_values = values[(values >= 0) & (values <= 40)]
                    axis.hist(
                        zoom_values,
                        bins=np.linspace(0, 40, 121),
                        color="tab:orange",
                        alpha=0.75,
                    )
                else:
                    axis.hist(values, bins=120, color="tab:orange", alpha=0.75)
                axis.axvline(float(np.median(values)), color="black", ls="--", lw=1.0, alpha=0.8)
                axis.set_title(panel_title + title_suffix)
                axis.set_ylabel("Events")
                axis.set_yscale("log")
                axis.grid(alpha=0.25, which="both")
                if zoomed:
                    axis.set_xlim(0, 40)
                if row_idx == len(plot_columns) - 1:
                    axis.set_xlabel("Charge")

        for row_idx in range(len(plot_columns), axes.shape[0]):
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].axis("off")

        fig.suptitle(
            "Task 5 charge summary for fit_tt = 1234\n"
            "Per-plane charge and total event charge\n"
            f"Events: {len(df_charge_1234)} | Columns: {', '.join(plot_columns)}",
            fontsize=12,
        )

        if save_plots:
            final_filename = f"{fig_idx}_charge_fit_tt_1234_per_plane_and_total.png"
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(
                save_fig_path,
                fig=fig,
                format="png",
                dpi=150,
                alias="charge_fit_tt_1234_per_plane_and_total",
            )
        if show_plots:
            plt.show()
        plt.close(fig)
    else:
        print(
            "Warning: Task 5 charge summary plot skipped because no per-plane charge "
            "columns were found (neither P#_Q_sum_final nor Q_P#s# columns)."
        )

reprocessing_parameters = pd.DataFrame()

def load_reprocessing_parameters_for_file(station_id: str, task_id: str, basename: str) -> pd.DataFrame:
    """Return matching reprocessing parameters for *basename* or an empty frame."""
    station_str = str(station_id).zfill(2)
    table_path = REFERENCE_TABLES_DIR / f"reprocess_files_station_{station_str}_task_{task_id}.csv"
    if not table_path.exists():
        return pd.DataFrame()
    try:
        table_df = pd.read_csv(table_path)
    except Exception as exc:
        print(f"Warning: unable to read reprocessing table {table_path}: {exc}")
        return pd.DataFrame()
    if "filename_base" not in table_df.columns:
        return pd.DataFrame()
    matches = table_df[table_df["filename_base"] == basename]
    return matches.reset_index(drop=True)

# -------------------------------------------------------------------------------
# ------------ Input file and data managing to select configuration -------------
# -------------------------------------------------------------------------------

is_simulated_file = basename_no_ext.startswith("mi00")
used_input_file = False

if simulated_z_positions is not None:
    z_positions = np.array(simulated_z_positions, dtype=float)
    found_matching_conf = True
    print(f"Using simulated z_positions from param_hash={simulated_param_hash}")
elif is_simulated_file:
    print("Warning: Simulated file missing param_hash; using default z_positions.")
    found_matching_conf = False
    z_positions = np.array([0, 150, 300, 450])  # In mm
elif exists_input_file:
    used_input_file = True
    selection_result = select_input_file_configuration(
        input_file,
        start_time=start_time,
        end_time=end_time,
    )
    print(selection_result.matching_confs)

    if selection_result.selected_conf is not None:
        if selection_result.reason == "exact_overlap_latest_start":
            print(
                "Warning:\n"
                "Multiple configurations match the date range\n"
                f"{start_time} to {end_time}.\n"
                "Selecting the matching configuration with the most recent start date."
            )
        elif selection_result.reason != "exact":
            print("Warning: No matching configuration for the date range; selecting closest configuration.")
        selected_conf = selection_result.selected_conf
        print(f"Selected configuration: {selected_conf['conf']}")
        z_positions = np.array([selected_conf.get(f"P{i}", np.nan) for i in range(1, 5)])
        found_matching_conf = True
        print(selected_conf['conf'])
    else:
        print("Warning: Input configuration file has no valid selectable rows. Using default z_positions.")
        z_positions = np.array([0, 150, 300, 450])  # In mm
else:
    print("Error: No input file. Using default z_positions.")
    z_positions = np.array([0, 150, 300, 450])  # In mm

# If any of the z_positions is NaN, use default values
if np.isnan(z_positions).any():
    print("Error: Incomplete z_positions in the selected configuration. Using default z_positions.")
    z_positions = np.array([0, 150, 300, 450])  # In mm

# Print the resulting z_positions
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")

# Save the z_positions in the metadata file
global_variables['z_P1'] =  z_positions[0]
global_variables['z_P2'] =  z_positions[1]
global_variables['z_P3'] =  z_positions[2]
global_variables['z_P4'] =  z_positions[3]

fast_mode = False
debug_mode = False
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
(
    plot_mode,
    create_plots,
    create_essential_plots,
    save_plots,
    create_pdf,
    show_plots,
    create_debug_plots,
) = resolve_step1_plot_options(config)
apply_task5_plot_catalog_modes()

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering

# Time calibration

# Y position

# RPC variables

# Alternative

# TimTrack

# Validation

# Pre-cal Front & Back
T_side_left_pre_cal_debug = config.get("T_side_left_pre_cal_debug", -500)
T_side_right_pre_cal_debug = config.get("T_side_right_pre_cal_debug", 500)

T_side_left_pre_cal_default = config.get("T_side_left_pre_cal_default", -200)
T_side_right_pre_cal_default = config.get("T_side_right_pre_cal_default", -100)

T_side_left_pre_cal_ST = config.get("T_side_left_pre_cal_ST", -200)
T_side_right_pre_cal_ST = config.get("T_side_right_pre_cal_ST", -50)

# Pre-cal Sum & Diff

# Post-calibration

# Once calculated the RPC variables
T_sum_RPC_left = _task5_config_float(
    config,
    "T_sum_RPC_left",
    "plane_combination_plane_t_sum_sum_left",
    "plane_combination_same_plane_t_sum_sum_left",
    "plane_combination_self_t_sum_sum_left",
    default=-25.0,
)
T_sum_RPC_right = _task5_config_float(
    config,
    "T_sum_RPC_right",
    "plane_combination_plane_t_sum_sum_right",
    "plane_combination_same_plane_t_sum_sum_right",
    "plane_combination_self_t_sum_sum_right",
    default=25.0,
)

# Alternative fitter filter
det_pos_filter = config["det_pos_filter"]
det_theta_left_filter = config["det_theta_left_filter"]
det_theta_right_filter = config["det_theta_right_filter"]
det_phi_filter_abs = abs(float(config.get("det_phi_filter_abs", config.get("det_phi_right_filter", 3.141592))))
det_phi_right_filter = det_phi_filter_abs
det_phi_left_filter = -det_phi_filter_abs
det_slowness_filter_left = config["det_slowness_filter_left"]
det_slowness_filter_right = config["det_slowness_filter_right"]

# TimTrack filter

# Fitting comparison

# Calibrations

# Pedestal charge calibration

# Front-back charge

# Variables to modify
beta = config.get("beta", 1)
strip_speed_factor_of_c = config.get("strip_speed_factor_of_c", 0.666666667)

# X
strip_length = config.get("strip_length", 300)
narrow_strip = config.get("narrow_strip", 63)
wide_strip = config.get("wide_strip", 98)

# Timtrack parameters
anc_std = config.get("anc_std", 0.075)

n_planes_timtrack = config.get("n_planes_timtrack", 4)

# Plotting options
T_clip_min_debug = config.get("T_clip_min_debug", -500)
T_clip_max_debug = config.get("T_clip_max_debug", 500)
Q_clip_min_debug = config.get("Q_clip_min_debug", -500)
Q_clip_max_debug = config.get("Q_clip_max_debug", 500)
num_bins_debug = config.get("num_bins_debug", 100)

T_clip_min_default = config.get("T_clip_min_default", -300)
T_clip_max_default = config.get("T_clip_max_default", 100)
Q_clip_min_default = config.get("Q_clip_min_default", 0)
Q_clip_max_default = config.get("Q_clip_max_default", 500)
num_bins_default = config.get("num_bins_default", 100)

T_clip_min_ST = config.get("T_clip_min_ST", -300)
T_clip_max_ST = config.get("T_clip_max_ST", 100)
Q_clip_min_ST = config.get("Q_clip_min_ST", 0)
Q_clip_max_ST = config.get("Q_clip_max_ST", 500)

config_files_directory = str(resolve_master_config_root_from_config(config))

angular_corr_directory = (
    config_files_directory + "/STAGE_1/EVENT_DATA/STEP_1/TASK_5/ANGULAR_CORRECTION"
)

# -----------------------------------------------------------------------------
# Some variables that define the analysis, define a dictionary with the variables:
# 'purity_of_data', etc.
if False:
    print('Working in fast mode.')

if False:
    print('Working in debug mode.')

if False:
    T_F_left_pre_cal = T_side_left_pre_cal_debug
    T_F_right_pre_cal = T_side_right_pre_cal_debug

    T_B_left_pre_cal = T_side_left_pre_cal_debug
    T_B_right_pre_cal = T_side_right_pre_cal_debug

else:
    T_F_left_pre_cal = T_side_left_pre_cal_default  #-130
    T_F_right_pre_cal = T_side_right_pre_cal_default

    T_B_left_pre_cal = T_side_left_pre_cal_default
    T_B_right_pre_cal = T_side_right_pre_cal_default

T_F_left_pre_cal_ST = T_side_left_pre_cal_ST  #-115
T_F_right_pre_cal_ST = T_side_right_pre_cal_ST
T_B_left_pre_cal_ST = T_side_left_pre_cal_ST
T_B_right_pre_cal_ST = T_side_right_pre_cal_ST

# Y ---------------------------------------------------------------------------
y_widths = [np.array([wide_strip, wide_strip, wide_strip, narrow_strip]), 
            np.array([narrow_strip, wide_strip, wide_strip, wide_strip])]

y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_P1_and_P3 = y_widths[0]
y_width_P2_and_P4 = y_widths[1]
y_pos_P1_and_P3 = y_pos(y_width_P1_and_P3)
y_pos_P2_and_P4 = y_pos(y_width_P2_and_P4)
total_width = np.sum(y_width_P1_and_P3)

c_mm_ns = c/1000000
print(c_mm_ns)

# Miscelanous ----------------------------
muon_speed = beta * c_mm_ns
strip_speed = strip_speed_factor_of_c * c_mm_ns # 200 mm/ns
tdiff_to_x = strip_speed # Factor to transform t_diff to X

# Not-Hardcoded
vc    = beta * c_mm_ns # mm/ns
sc    = 1/vc
ss    = 1/strip_speed # slowness of the signal in the strip
nplan = n_planes_timtrack
lenx  = strip_length
anc_sx = tdiff_to_x * anc_std # 2 cm

if False:
    T_clip_min = T_clip_min_debug
    T_clip_max = T_clip_max_debug
    Q_clip_min = Q_clip_min_debug
    Q_clip_max = Q_clip_max_debug
    num_bins = num_bins_debug
else:
    T_clip_min = T_clip_min_default
    T_clip_max = T_clip_max_default
    Q_clip_min = Q_clip_min_default
    Q_clip_max = Q_clip_max_default
    num_bins = num_bins_default

T_clip_min_ST = T_clip_min_ST
T_clip_max_ST = T_clip_max_ST
Q_clip_min_ST = Q_clip_min_ST
Q_clip_max_ST = Q_clip_max_ST

reprocessing_values: dict[str, object] = {}
qa_reprocessing_context = load_qa_reprocessing_context_for_file(
    station,
    basename_no_ext,
    repo_root=repo_root,
)
apply_qa_reprocessing_context(global_variables, qa_reprocessing_context)
if int(qa_reprocessing_context.get("qa_reprocessing_mode", 0) or 0) == 1:
    print("Active QA reprocessing state found for this file.")
    print(
        "QA selectors: "
        f"{qa_reprocessing_context.get('qa_reprocessing_selector_ids', '') or 'none'}"
    )

reprocessing_parameters = load_reprocessing_parameters_for_file(station, str(task_number), basename_no_ext)
if not reprocessing_parameters.empty:
    if int(global_variables.get("qa_reprocessing_mode", 0) or 0) == 1:
        print("Reprocessing parameters found for this file. qa_reprocessing_mode=1.")
    else:
        print("Reprocessing parameters found for this file.")
    # Print only non-NaN entries from the reprocessing table
    non_nan = reprocessing_parameters.dropna(how="all").dropna(axis=1, how="all")
    if non_nan.empty:
        print("Reprocessing parameters found but all values are NaN.")
        columns_with_values: list[str] = []
    else:
        print(non_nan.to_string(index=False))
        columns_with_values = list(non_nan.columns)

    reprocessing_values = {
        column: get_reprocessing_value(reprocessing_parameters, column)
        for column in columns_with_values
    }
    reprocessing_values = {
        key: value for key, value in reprocessing_values.items() if value is not None
    }

# -------------------------------------------------------------------------------
# ------------ Input file and data managing to select configuration -------------
# -------------------------------------------------------------------------------

z_source = "unset"

if simulated_z_positions is not None:
    z_positions = np.array(simulated_z_positions, dtype=float)
    found_matching_conf = True
    print(f"Using simulated z_positions from param_hash={simulated_param_hash}")
    z_source = "simulated_param_hash"
elif is_simulated_file:
    print("Warning: Simulated file missing param_hash; using default z_positions.")
    found_matching_conf = False
    z_positions = np.array([0, 150, 300, 450])  # In mm
    z_source = "simulated_default_missing_param_hash"
elif exists_input_file:
    selection_result = select_input_file_configuration(
        input_file,
        start_time=start_time,
        end_time=end_time,
    )
    print(selection_result.matching_confs)

    if selection_result.selected_conf is not None:
        if selection_result.reason == "exact_overlap_latest_start":
            print(
                "Warning:\n"
                "Multiple configurations match the date range\n"
                f"{start_time} to {end_time}.\n"
                "Selecting the matching configuration with the most recent start date."
            )
        elif selection_result.reason != "exact":
            print("Warning: No matching configuration for the date range; selecting closest configuration.")
        selected_conf = selection_result.selected_conf
        print(f"Selected configuration: {selected_conf['conf']}")
        z_positions = np.array([selected_conf.get(f"P{i}", np.nan) for i in range(1, 5)])
        found_matching_conf = True
        print(selected_conf['conf'])
        z_source = f"input_file_conf_{selected_conf.get('conf')}"
    else:
        print("Warning: Input configuration file has no valid selectable rows. Using default z_positions.")
        z_positions = np.array([0, 150, 300, 450])  # In mm
        z_source = "default_invalid_input_file"
else:
    print("Error: No input file. Using default z_positions.")
    z_positions = np.array([0, 150, 300, 450])  # In mm
    z_source = "default_no_input_file"

# Print the resulting z_positions
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")
z_vector_mm = [round(float(value), 3) for value in z_positions.tolist()]
print(
    f"[Z_TRACE] file={basename_no_ext} source={z_source} "
    f"param_hash={simulated_param_hash or 'NA'} z_vector_mm={z_vector_mm}",
    force=True,
)

# Save the z_positions in the metadata file
global_variables['z_P1'] =  z_positions[0]
global_variables['z_P2'] =  z_positions[1]
global_variables['z_P3'] =  z_positions[2]
global_variables['z_P4'] =  z_positions[3]

raw_data_len = len(working_df)
if raw_data_len == 0 and not self_trigger:
    print("No coincidence nor self-trigger events.")
    sys.exit(1)

print("TASK_5 angular-region segmentation disabled.")

correct_angle = bool(config.get("correct_angle", False))
global_variables['correct_angle'] = correct_angle

df = working_df.copy()
df['Theta_fit'] = df['theta']
df['Phi_fit'] = df['phi']

if task5_plot_enabled("theta_phi_definitive_tt_2d"):

    print("-------------------------- Angular plots -----------------------------")

    df_filtered = df
    # tt_values = [13, 12, 23, 34, 123, 124, 134, 234, 1234]
    tt_values = [23, 123, 234, 1234]
    
    n_tt = len(tt_values)
    ncols = 2
    nrows = (n_tt + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        
    nbins = 50
    theta_bins = np.linspace(0, np.pi/2, nbins)
    phi_bins = np.linspace(-np.pi, np.pi, nbins)
    colors = plt.cm.viridis

    for idx, tt_val in enumerate(tt_values):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]
            
        df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val]
        theta_vals = df_tt['theta'].dropna()
        phi_vals = df_tt['phi'].dropna()

        if len(theta_vals) < 10 or len(phi_vals) < 10:
            ax.set_visible(False)
            continue
        
        h = ax.hist2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins], cmap='viridis', norm=None, cmin=0, cmax=None)
        ax.set_title(f'definitive_tt = {tt_val}')
        ax.set_xlabel(r'$\theta$ [rad]')
        ax.set_ylabel(r'$\phi$ [rad]')
        ax.grid(True)
        # Put the background color to the darkest in the colormap
        ax.set_facecolor(colors(0.0))  # darkest background in colormap

        fig.colorbar(h[3], ax=ax, label='Counts')

    plt.suptitle(r'2D Histogram of $\theta$ vs. $\phi$ for each definitive_tt Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plots:
        final_filename = f'{fig_idx}_theta_phi_definitive_tt_2D.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png', alias="theta_phi_definitive_tt_2d")

    if show_plots:
        plt.show()
    plt.close()



if task5_plot_enabled("polar_theta_phi_definitive_tt_2d_detail_pre"):

    theta_left_filter = 0
    theta_right_filter = np.pi / 2.5
        
    phi_left_filter = -np.pi
    phi_right_filter = np.pi
        
    df_filtered = df
    # tt_values = sorted(df_filtered['definitive_tt'].dropna().unique(), key=lambda x: int(x))

    # tt_values = [13, 12, 23, 34, 123, 124, 134, 234, 1234]
    tt_values = [23, 123, 234, 1234]

    n_tt = len(tt_values)
    ncols = 2
    nrows = (n_tt + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    phi_nbins = 70
    # theta_nbins = int(round(phi_nbins / 2) + 1)
    theta_nbins = 40
    theta_bins = np.linspace(theta_left_filter, theta_right_filter, theta_nbins )
    phi_bins = np.linspace(phi_left_filter, phi_right_filter, phi_nbins)
    colors = plt.cm.turbo

    # Select theta/phi range (optional filtering)
    theta_min, theta_max = theta_left_filter, theta_right_filter    # adjust as needed
    phi_min, phi_max     = phi_left_filter, phi_right_filter        # adjust as needed
    
    vmax_global = df_filtered.groupby('definitive_tt').apply(lambda df: np.histogram2d(df['theta'], df['phi'], bins=[theta_bins, phi_bins])[0].max()).max()
    
    for idx, tt_val in enumerate(tt_values):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]

        df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val]
        theta_vals = df_tt['theta'].dropna()
        phi_vals = df_tt['phi'].dropna()

        # Apply range filtering
        df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val].copy()
        mask = (
            (df_tt['theta'] >= theta_min) & (df_tt['theta'] <= theta_max) &
            (df_tt['phi'] >= phi_min) & (df_tt['phi'] <= phi_max)
        )
        df_tt = df_tt[mask]

        theta_vals = df_tt['theta']
        phi_vals   = df_tt['phi']

        if len(theta_vals) < 10 or len(phi_vals) < 10:
            ax.set_visible(False)
            continue

        # Polar plot settings
        fig.delaxes(axes[row_idx][col_idx])  # remove the original non-polar Axes
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)  # add a polar Axes
        axes[row_idx][col_idx] = ax  # update reference for consistency

        ax.set_facecolor(colors(0.0))  # darkest background in colormap
        ax.set_title(f'definitive_tt = {tt_val}', fontsize=14)
            
        # Limit in radius in theta_right_filter
        ax.set_ylim(0, theta_right_filter)
            
        # 2D histogram: use phi as angle, theta as radius
        h, r_edges, phi_edges = np.histogram2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins])
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        # R, PHI = np.meshgrid(r_centers, phi_centers, indexing='ij')
        R, PHI = np.meshgrid(r_edges, phi_edges, indexing='ij')
        c = ax.pcolormesh(PHI, R, h, cmap='viridis', vmin=0, vmax=vmax_global)
        local_max = h.max()
        cb = fig.colorbar(c, ax=ax, pad=0.1)
        cb.ax.hlines(local_max, *cb.ax.get_xlim(), colors='white', linewidth=2, linestyles='dashed')

    plt.suptitle(r'PRE-CORRECTION. 2D Histogram of $\theta$ vs. $\phi$ for each definitive_tt Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_polar_theta_phi_definitive_tt_2D_detail.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(
            save_fig_path,
            format='png',
            alias="polar_theta_phi_definitive_tt_2d_detail_pre",
        )
    if show_plots:
        plt.show()
    plt.close()

# ---------------------------------------------------------------
# 1. Build absolute path and sanity-check
# ---------------------------------------------------------------
hdf_path = os.path.join(angular_corr_directory, "likelihood_matrices.parquet")
if not os.path.isfile(hdf_path):
    print(f"HDF5 file not found: {hdf_path}")
    correct_angle = bool(config.get("correct_angle", False))

_prof["s_post_read_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
if correct_angle:
    print("----------------------------------------------------------------------")
    print("-------- 1. Correction of the fitted angle --> predicted angle -------")
    print("----------------------------------------------------------------------")

    # ---------------------------------------------------------------
    # 2. Load all matrices into memory
    # ---------------------------------------------------------------
    matrices = {}
    n_bins = None

    with pd.HDFStore(hdf_path, mode='r') as store:
        keys = store.keys()
        if not keys:
            raise ValueError(f"{hdf_path} contains no datasets.")

        for key in keys:                     # keys like '/P1', '/P2', …
            ttype = key.strip('/')           # remove leading slash
            # df_M = store.get(key)
            
            # Reduce the precision to float32 to not kill RAM
            df_M = store.get(key).astype(np.float16)
            
            matrices[ttype] = df_M

            # set n_bins once, based on the first matrix's shape
            if n_bins is None:
                size = df_M.shape[0]
                n_bins = int(np.sqrt(size))
                if n_bins * n_bins != size:
                    raise ValueError(f"Matrix size {size} is not a perfect square.")

            print(f"Loaded matrix for {ttype}: shape {df_M.shape}")

    print(f"n_bins detected: {n_bins}")

    # Helpers
    def flat(u_idx, v_idx, n_bins):
        return u_idx * n_bins + v_idx

    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    #%%

    with pd.HDFStore(hdf_path, 'r') as store:
        print("HDF5 keys:", store.keys())

    def sample_true_angles_nearest(
        df_fit: pd.DataFrame,
        matrices: Optional[Dict[str, pd.DataFrame]],
        n_bins: int,
        rng: Optional[np.random.Generator] = None,
        show_progress: bool = True,
        print_every: int = 10_000
        ) -> pd.DataFrame:
        
        if rng is None:
            rng = np.random.default_rng()

        matrix_cache = {t: df_m.to_numpy() for t, df_m in matrices.items()}
        
        u_edges = np.linspace(-1.0, 1.0, n_bins + 1)
        v_edges = np.linspace(-1.0, 1.0, n_bins + 1)

        u_fit = np.sin(df_fit["Theta_fit"].values) * np.sin(df_fit["Phi_fit"].values)
        v_fit = np.sin(df_fit["Theta_fit"].values) * np.cos(df_fit["Phi_fit"].values)

        iu = np.clip(np.digitize(u_fit, u_edges) - 1, 0, n_bins - 2)
        iv = np.clip(np.digitize(v_fit, v_edges) - 1, 0, n_bins - 2)

        iu += (u_fit - u_edges[iu]) > (u_edges[iu + 1] - u_fit)
        iv += (v_fit - v_edges[iv]) > (v_edges[iv + 1] - v_fit)

        flat_idx = lambda u, v: u * n_bins + v
        unflat = lambda k: divmod(k, n_bins)

        N = len(df_fit)
        theta_pred = np.empty(N, dtype=np.float32)
        phi_pred = np.empty(N, dtype=np.float32)

        iterator = tqdm(range(N), desc="Sampling true angles (nearest-bin)", unit="evt") if show_progress else range(N)

        for n in iterator:
            t_type = str(df_fit["definitive_tt"].iat[n])   # ensure string

            if t_type not in matrix_cache:
                raise ValueError(f"LUT not found for type: {t_type}")
            M = matrix_cache[t_type]

            col_idx = flat_idx(iu[n], iv[n])
            p = M[:, col_idx]
            s = p.sum()

            if s == 0:
                p = np.full_like(p, 1.0 / len(p))
            else:
                p /= s

            gen_idx = rng.choice(len(p), p=p)
            g_u_idx, g_v_idx = unflat(gen_idx)

            u_pred = rng.uniform(u_edges[g_u_idx], u_edges[g_u_idx + 1])
            v_pred = rng.uniform(v_edges[g_v_idx], v_edges[g_v_idx + 1])

            sin_theta = min(np.hypot(u_pred, v_pred), 1.0)
            theta_pred[n] = math.asin(sin_theta)
            phi_pred[n] = wrap_to_pi(math.atan2(u_pred, v_pred))

        df_out = df_fit.copy()
        df_out["Theta_pred"] = theta_pred
        df_out["Phi_pred"] = phi_pred
        return df_out

    if VERBOSE:
        print(df.columns.to_list())

    #%%

    df = sample_true_angles_nearest(
        df_fit=df,
        matrices=matrices,
        n_bins=n_bins,
        rng=np.random.default_rng(),
        show_progress=True,
    )
    
    df['theta'] = df['Theta_pred']
    df['phi'] = df['Phi_pred']
    
    # Plotting corrected vs measured angles
    if task5_plot_enabled("polar_theta_phi_definitive_tt_2d_detail_angle_correction"):
        VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']
        tt_lists = [ VALID_MEASURED_TYPES ]
        
        for tt_list in tt_lists:
            fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex='row')

            # Fourth column: Measured (θ_fit, ϕ_fit)
            axes[0, 0].hist(df['Theta_fit'], bins=theta_bins, histtype='step', color='black', label='All')
            axes[1, 0].hist(df['Phi_fit'], bins=phi_bins, histtype='step', color='black', label='All')
            for tt in tt_list:
                    sel = (df['definitive_tt'] == int(tt))
                    axes[0, 0].hist(df.loc[sel, 'Theta_fit'], bins=theta_bins, histtype='step', label=tt)
                    axes[1, 0].hist(df.loc[sel, 'Phi_fit'], bins=phi_bins, histtype='step', label=tt)
                    axes[0, 0].set_title("Measured tracks θ_fit")
                    axes[1, 0].set_title("Measured tracks ϕ_fit")
        
            # Fourth column: Measured (θ_fit, ϕ_fit)
            axes[0, 1].hist(df['Theta_pred'], bins=theta_bins, histtype='step', color='black', label='All')
            axes[1, 1].hist(df['Phi_pred'], bins=phi_bins, histtype='step', color='black', label='All')
            for tt in tt_list:
                    sel = (df['definitive_tt'] == int(tt))
                    axes[0, 1].hist(df.loc[sel, 'Theta_pred'], bins=theta_bins, histtype='step', label=tt)
                    axes[1, 1].hist(df.loc[sel, 'Phi_pred'], bins=phi_bins, histtype='step', label=tt)
                    axes[0, 1].set_title("Corrected tracks θ_fit")
                    axes[1, 1].set_title("Corrected tracks ϕ_fit")

            # Common settings
            for ax in axes.flat:
                    ax.legend(fontsize='x-small')
                    ax.grid(True)

            axes[1, 0].set_xlabel(r'$\phi$ [rad]')
            axes[0, 0].set_ylabel('Counts')
            axes[1, 0].set_ylabel('Counts')
            axes[0, 1].set_xlim(0, np.pi / 2)
            axes[1, 1].set_xlim(-np.pi, np.pi)

            fig.tight_layout()
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save_plots:
                final_filename = f'{fig_idx}_polar_theta_phi_definitive_tt_2D_detail.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(
                    save_fig_path,
                    format='png',
                    alias="polar_theta_phi_definitive_tt_2d_detail_angle_correction",
                )
            if show_plots:
                plt.show()
            plt.close()

else:
    print("Angle correction is disabled.")
    df['Theta_pred'] = df['Theta_fit']
    df['Phi_pred'] = df['Phi_fit']
    
    df['theta'] = df['Theta_pred']
    df['phi'] = df['Phi_pred']
    df.drop(columns=["Theta_fit", "Phi_fit"], inplace=True, errors="ignore")

gc.collect()

if task5_plot_enabled("polar_theta_phi_definitive_tt_2d_detail_final"):
    
    theta_left_filter = 0
    theta_right_filter = np.pi / 2.5
        
    phi_left_filter = -np.pi
    phi_right_filter = np.pi
        
    df_filtered = df
    # tt_values = sorted(df_filtered['definitive_tt'].dropna().unique(), key=lambda x: int(x))

    # tt_values = [13, 12, 23, 34, 123, 124, 134, 234, 1234]
    tt_values = [23, 123, 234, 1234]

    n_tt = len(tt_values)
    ncols = 2
    nrows = (n_tt + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    phi_nbins = 70
    # theta_nbins = int(round(phi_nbins / 2) + 1)
    theta_nbins = 40
    theta_bins = np.linspace(theta_left_filter, theta_right_filter, theta_nbins )
    phi_bins = np.linspace(phi_left_filter, phi_right_filter, phi_nbins)
    colors = plt.cm.turbo

    # Select theta/phi range (optional filtering)
    theta_min, theta_max = theta_left_filter, theta_right_filter    # adjust as needed
    phi_min, phi_max     = phi_left_filter, phi_right_filter        # adjust as needed
    
    vmax_global = df_filtered.groupby('definitive_tt').apply(lambda df: np.histogram2d(df['theta'], df['phi'], bins=[theta_bins, phi_bins])[0].max()).max()
    
    for idx, tt_val in enumerate(tt_values):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]

        df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val]
        theta_vals = df_tt['theta'].dropna()
        phi_vals = df_tt['phi'].dropna()

        # Apply range filtering
        # Apply range filtering
        df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val].copy()
        mask = (
            (df_tt['theta'] >= theta_min) & (df_tt['theta'] <= theta_max) &
            (df_tt['phi'] >= phi_min) & (df_tt['phi'] <= phi_max)
        )
        df_tt = df_tt[mask]

        theta_vals = df_tt['theta']
        phi_vals   = df_tt['phi']

        if len(theta_vals) < 10 or len(phi_vals) < 10:
            ax.set_visible(False)
            continue

        # Polar plot settings
        fig.delaxes(axes[row_idx][col_idx])  # remove the original non-polar Axes
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)  # add a polar Axes
        axes[row_idx][col_idx] = ax  # update reference for consistency

        ax.set_facecolor(colors(0.0))  # darkest background in colormap
        ax.set_title(f'definitive_tt = {tt_val}', fontsize=14)
            
        # Limit in radius in theta_right_filter
        ax.set_ylim(0, theta_right_filter)
            
        # 2D histogram: use phi as angle, theta as radius
        h, r_edges, phi_edges = np.histogram2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins])
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        # R, PHI = np.meshgrid(r_centers, phi_centers, indexing='ij')
        R, PHI = np.meshgrid(r_edges, phi_edges, indexing='ij')
        c = ax.pcolormesh(PHI, R, h, cmap='viridis', vmin=0, vmax=vmax_global)
        local_max = h.max()
        cb = fig.colorbar(c, ax=ax, pad=0.1)
        cb.ax.hlines(local_max, *cb.ax.get_xlim(), colors='white', linewidth=2, linestyles='dashed')

    plt.suptitle(rf'FINAL. Correction = {correct_angle}. 2D Histogram of $\theta$ vs. $\phi$ for each definitive_tt Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_polar_theta_phi_definitive_tt_2D_detail.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(
            save_fig_path,
            format='png',
            alias="polar_theta_phi_definitive_tt_2d_detail_final",
        )
    if show_plots:
        plt.show()
    plt.close()

_prof["s_angle_correction_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
if task5_plot_enabled("theta_efficiency_simple_3v4"):
    print("--------------- Simple efficiency vs theta (3v4 planes) -------------")

    if CORR_TT_COLUMN not in df.columns:
        print(f"Warning: {CORR_TT_COLUMN} column missing, skipping simple efficiency-vs-theta plot.")
    else:
        if "theta" not in df.columns:
            print("Warning: theta column missing, skipping simple efficiency-vs-theta plot.")
            theta_vals_all = None
            corr_tt_vals = None
        else:
            theta_vals_all = pd.to_numeric(df["theta"], errors="coerce")
            corr_tt_vals = pd.to_numeric(df[CORR_TT_COLUMN], errors="coerce").fillna(0).astype(int)

        if theta_vals_all is not None and corr_tt_vals is not None:
            right_theta_eff = float(pd.to_numeric(det_theta_right_filter, errors="coerce"))
            if not np.isfinite(right_theta_eff) or right_theta_eff <= 0:
                right_theta_eff = np.pi / 2.5
            n_theta_bins_eff = 10
            bins_theta_eff = np.linspace(0, right_theta_eff, n_theta_bins_eff + 1)
            theta_centers_eff = 0.5 * (bins_theta_eff[:-1] + bins_theta_eff[1:])

            theta_np = theta_vals_all.to_numpy(dtype=float, copy=False)
            corr_np = corr_tt_vals.to_numpy(dtype=int, copy=False)
            valid = (
                np.isfinite(theta_np)
                & (theta_np >= 0)
                & (theta_np <= right_theta_eff)
                & np.isin(corr_np, [1234, 134, 124])
            )
            theta_vals_eff = theta_np[valid]
            corr_tt_eff = corr_np[valid]

            counts_1234, _ = np.histogram(theta_vals_eff[corr_tt_eff == 1234], bins=bins_theta_eff)
            counts_134, _ = np.histogram(theta_vals_eff[corr_tt_eff == 134], bins=bins_theta_eff)
            counts_124, _ = np.histogram(theta_vals_eff[corr_tt_eff == 124], bins=bins_theta_eff)

            if counts_1234.sum() == 0:
                print("Warning: No corr_tt==1234 events in theta range, skipping simple efficiency-vs-theta plot.")
            else:
                eff_2 = np.full_like(theta_centers_eff, np.nan, dtype=float)
                eff_3 = np.full_like(theta_centers_eff, np.nan, dtype=float)
                err_2 = np.full_like(theta_centers_eff, np.nan, dtype=float)
                err_3 = np.full_like(theta_centers_eff, np.nan, dtype=float)
                nonzero_den = counts_1234 > 0
                n4 = counts_1234.astype(float)
                n134 = counts_134.astype(float)
                n124 = counts_124.astype(float)

                # Efficiencies requested by user:
                #   eff_2 = 1 - N134 / N1234
                #   eff_3 = 1 - N124 / N1234
                eff_2[nonzero_den] = 1.0 - (n134[nonzero_den] / n4[nonzero_den])
                eff_3[nonzero_den] = 1.0 - (n124[nonzero_den] / n4[nonzero_den])

                # Poisson error propagation for ratio N3/N4 (independent counts):
                # sigma_r^2 = (sigma_N3/N4)^2 + (N3*sigma_N4/N4^2)^2, sigma_N = sqrt(N)
                with np.errstate(divide="ignore", invalid="ignore"):
                    err_2[nonzero_den] = np.sqrt(
                        (np.sqrt(n134[nonzero_den]) / n4[nonzero_den]) ** 2
                        + ((n134[nonzero_den] * np.sqrt(n4[nonzero_den])) / (n4[nonzero_den] ** 2)) ** 2
                    )
                    err_3[nonzero_den] = np.sqrt(
                        (np.sqrt(n124[nonzero_den]) / n4[nonzero_den]) ** 2
                        + ((n124[nonzero_den] * np.sqrt(n4[nonzero_den])) / (n4[nonzero_den] ** 2)) ** 2
                    )

                valid_w2 = nonzero_den & np.isfinite(eff_2) & np.isfinite(err_2) & (err_2 > 0)
                valid_w3 = nonzero_den & np.isfinite(eff_3) & np.isfinite(err_3) & (err_3 > 0)
                if np.any(valid_w2):
                    weights_2 = 1.0 / np.square(err_2[valid_w2])
                    mean_eff_2 = float(np.sum(weights_2 * eff_2[valid_w2]) / np.sum(weights_2))
                else:
                    mean_eff_2 = float(np.nanmean(eff_2[nonzero_den])) if np.any(nonzero_den) else np.nan
                if np.any(valid_w3):
                    weights_3 = 1.0 / np.square(err_3[valid_w3])
                    mean_eff_3 = float(np.sum(weights_3 * eff_3[valid_w3]) / np.sum(weights_3))
                else:
                    mean_eff_3 = float(np.nanmean(eff_3[nonzero_den])) if np.any(nonzero_den) else np.nan

                theta_centers_eff_deg = np.degrees(theta_centers_eff)
                right_theta_eff_deg = np.degrees(right_theta_eff)
                cos_theta_centers = np.cos(theta_centers_eff)
                cos_theta_safe = np.clip(np.abs(cos_theta_centers), 1e-6, None)
                sec_theta_centers = 1.0 / cos_theta_safe
                valid_eff_bins = nonzero_den & (np.isfinite(eff_2) | np.isfinite(eff_3))

                def _xlimits_from_values(values: np.ndarray, default_limits: tuple[float, float]) -> tuple[float, float]:
                    vals = np.asarray(values, dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        return default_limits
                    vmin = float(np.min(vals))
                    vmax = float(np.max(vals))
                    if vals.size > 1:
                        unique_vals = np.unique(np.sort(vals))
                        if unique_vals.size > 1:
                            step = float(np.median(np.diff(unique_vals)))
                            if np.isfinite(step) and step > 0:
                                pad = 0.5 * step
                            else:
                                pad = 0.02 * (vmax - vmin if vmax > vmin else 1.0)
                        else:
                            pad = 0.05 * max(abs(vmin), 1.0)
                    else:
                        pad = 0.05 * max(abs(vmin), 1.0)
                    if vmax <= vmin:
                        pad = max(pad, 0.05 * max(abs(vmin), 1.0))
                    return vmin - pad, vmax + pad

                theta_xlim = _xlimits_from_values(
                    theta_centers_eff_deg[valid_eff_bins],
                    (0.0, right_theta_eff_deg),
                )
                sec_xlim = _xlimits_from_values(
                    sec_theta_centers[valid_eff_bins],
                    (float(np.nanmin(sec_theta_centers)), float(np.nanmax(sec_theta_centers))),
                )

                fig_eff, (ax_eff, ax_eff_cos) = plt.subplots(
                    nrows=2,
                    ncols=1,
                    figsize=(7, 10),
                )

                ax_eff.plot(
                    theta_centers_eff_deg,
                    eff_2,
                    marker="o",
                    linewidth=1.8,
                    label=r"$\epsilon_2 = 1 - N_{134}/N_{1234}$",
                    color="tab:blue",
                )
                ax_eff.plot(
                    theta_centers_eff_deg,
                    eff_3,
                    marker="s",
                    linewidth=1.8,
                    label=r"$\epsilon_3 = 1 - N_{124}/N_{1234}$",
                    color="tab:green",
                )
                ax_eff.fill_between(
                    theta_centers_eff_deg,
                    np.clip(eff_2 - err_2, 0.0, 1.05),
                    np.clip(eff_2 + err_2, 0.0, 1.05),
                    alpha=0.2,
                    color="tab:blue",
                )
                ax_eff.fill_between(
                    theta_centers_eff_deg,
                    np.clip(eff_3 - err_3, 0.0, 1.05),
                    np.clip(eff_3 + err_3, 0.0, 1.05),
                    alpha=0.2,
                    color="tab:green",
                )
                if np.isfinite(mean_eff_2):
                    ax_eff.axhline(
                        mean_eff_2,
                        linestyle="--",
                        linewidth=1.4,
                        alpha=0.35,
                        color="tab:blue",
                        label=rf"w-mean $\epsilon_2$ = {mean_eff_2:.3f}",
                    )
                if np.isfinite(mean_eff_3):
                    ax_eff.axhline(
                        mean_eff_3,
                        linestyle="--",
                        linewidth=1.4,
                        alpha=0.35,
                        color="tab:green",
                        label=rf"w-mean $\epsilon_3$ = {mean_eff_3:.3f}",
                    )
                ax_eff.set_xlim(*theta_xlim)
                ax_eff.set_ylim(0.5, 1.05)
                ax_eff.set_ylabel("Efficiency")
                ax_eff.set_title("Simple Angular Efficiencies (3-plane vs 4-plane)")
                ax_eff.grid(True, alpha=0.3)
                ax_eff.legend(fontsize="small")

                ax_eff_cos.plot(
                    sec_theta_centers,
                    eff_2,
                    marker="o",
                    linewidth=1.8,
                    label=r"$\epsilon_2$ vs $\sec(\theta_{\mathrm{bin}})$",
                    color="tab:blue",
                )
                ax_eff_cos.plot(
                    sec_theta_centers,
                    eff_3,
                    marker="s",
                    linewidth=1.8,
                    label=r"$\epsilon_3$ vs $\sec(\theta_{\mathrm{bin}})$",
                    color="tab:green",
                )
                ax_eff_cos.fill_between(
                    sec_theta_centers,
                    np.clip(eff_2 - err_2, 0.0, 1.05),
                    np.clip(eff_2 + err_2, 0.0, 1.05),
                    alpha=0.2,
                    color="tab:blue",
                )
                ax_eff_cos.fill_between(
                    sec_theta_centers,
                    np.clip(eff_3 - err_3, 0.0, 1.05),
                    np.clip(eff_3 + err_3, 0.0, 1.05),
                    alpha=0.2,
                    color="tab:green",
                )
                ax_eff_cos.set_xlim(*sec_xlim)
                ax_eff_cos.set_ylim(0.5, 1.05)
                ax_eff_cos.set_xlabel(r"$\sec(\theta_{\mathrm{bin}}) = 1/\cos(\theta_{\mathrm{bin}})$")
                ax_eff_cos.set_ylabel("Efficiency")
                ax_eff_cos.set_title(r"Efficiency vs $\sec(\theta_{\mathrm{bin}})$")
                ax_eff_cos.grid(True, alpha=0.3)
                ax_eff_cos.legend(fontsize="small")
                plt.tight_layout()

                if save_plots:
                    final_filename = f"{fig_idx}_theta_efficiency_simple_3v4.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, format="png", alias="theta_efficiency_simple_3v4")
                if show_plots:
                    plt.show()
                plt.close()

df['region'] = "all"
print(df['region'].value_counts())

working_df = df

# -----------------------------------------------------------------------------
# Create and save the PDF -----------------------------------------------------
# -----------------------------------------------------------------------------

if create_pdf:
    print(f"Creating PDF with all plots in {save_pdf_path}")
    existing_pngs = collect_saved_plot_paths(plot_list, base_directories["figure_directory"])

    if _direct_pdf_pages is not None:
        # Direct PDF mode already wrote each page incrementally via save_plot_figure.
        close_direct_pdf_writer()
    elif existing_pngs:
        temp_pdf_path = _build_temp_pdf_path(save_pdf_path)
        try:
            with PdfPages(temp_pdf_path) as pdf:
                for png in existing_pngs:
                    img = Image.open(png)
                    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
                    ax.imshow(img)
                    ax.axis('off')
                    pdf_save_rasterized_page(pdf, fig, bbox_inches='tight')
                    plt.close(fig)
            os.replace(temp_pdf_path, save_pdf_path)
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    # Remove PNG files after creating the PDF (or after direct PDF append path).
    for png in existing_pngs:
        try:
            os.remove(png)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
    
    # Remove run-specific figure directory if all PNGs were deleted.
    figure_directory = base_directories["figure_directory"]
    if os.path.exists(figure_directory):
        if not os.listdir(figure_directory):
            os.rmdir(figure_directory)
        else:
            print(f"Figure directory not empty, skipping removal: {figure_directory}")

# Path to save the cleaned dataframe
# Create output directory if it does not exist.
os.makedirs(f"{output_directory}", exist_ok=True)
OUT_PATH = f"{output_directory}/corrected_{basename_no_ext}.parquet"
KEY = "df"  # HDF5 key name

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# --- Example: your cleaned DataFrame is called working_df ---
# (Here, you would have your data cleaning code before saving)
# working_df = ...

# Detailed column dumps are only useful while debugging.
if VERBOSE:
    print("Columns in the cleaned dataframe:")
    for col in working_df.columns:
        print(f" - {col}")

# Remove the columns in the form "T*_T_sum_*", "T*_T_dif_*", "Q*_Q_sum_*", "Q*_Q_dif_*", do a loop from 1 to 4
cols_to_remove = []
for i_plane in range(1, 5):
    for strip in range(1, 5):
        cols_to_remove.append(f'T{i_plane}_T_sum_{strip}')
        cols_to_remove.append(f'T{i_plane}_T_dif_{strip}')
        cols_to_remove.append(f'Q{i_plane}_Q_sum_{strip}')
        cols_to_remove.append(f'Q{i_plane}_Q_dif_{strip}')
if keep_all_columns_output:
    print(
        "Task 5 keep_all_columns_output enabled: "
        f"retaining {len(cols_to_remove)} strip-summary columns."
    )
else:
    working_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

if VERBOSE:
    print("Columns in the final dataframe:")
    for col in working_df.columns:
        print(f" - {col}")
    
    

component_cols = []
for i_plane in range(1, 5):
    component_cols.extend(
        [
            f"P{i_plane}_T_sum_final",
            f"P{i_plane}_T_dif_final",
            f"P{i_plane}_Q_sum_final",
            f"P{i_plane}_Q_dif_final",
            f"P{i_plane}_Y_final",
        ]
    )
component_cols = [col for col in component_cols if col in working_df.columns]
if component_cols:
    component_data = working_df[component_cols].fillna(0)
    all_zero_mask = (component_data == 0).all(axis=1)
    removed_all_zero = int(all_zero_mask.sum())
    if removed_all_zero > 0:
        working_df = working_df.loc[~all_zero_mask].copy()
    record_filter_metric(
        "all_components_zero_rows_removed_pct",
        removed_all_zero,
        len(working_df) + removed_all_zero if (len(working_df) + removed_all_zero) else 0,
    )

print(f"Original number of events in the dataframe: {original_number_of_events}")
if create_debug_plots and POST_TT_COLUMN in working_df.columns:
    debug_fig_idx = plot_debug_histograms(
        working_df,
        [POST_TT_COLUMN],
        {POST_TT_COLUMN: [10]},
        title=f"Task 5 pre-filter: {POST_TT_COLUMN} >= 10 [NON-TUNABLE] (station {station})",
        out_dir=debug_plot_directory,
        fig_idx=debug_fig_idx,
    )
post_tt_total = len(working_df)
post_tt_mask = working_df[POST_TT_COLUMN].notna() & (working_df[POST_TT_COLUMN] >= 10)
working_df = working_df.loc[post_tt_mask].copy()
record_filter_metric(
    "corr_tt_lt_10_rows_removed_pct",
    post_tt_total - int(post_tt_mask.sum()),
    post_tt_total if post_tt_total else 0,
)

working_df.loc[:, "fit_to_post_tt"] = (
    pd.to_numeric(working_df["fit_tt"], errors="coerce").fillna(0).astype(int).astype(str)
    + "_"
    + pd.to_numeric(working_df[POST_TT_COLUMN], errors="coerce").fillna(0).astype(int).astype(str)
)

post_tt_counts = working_df[POST_TT_COLUMN].value_counts()
for tt_value, count in post_tt_counts.items():
    tt_label = normalize_tt_label(tt_value)
    global_variables[f"{POST_TT_COLUMN}_{tt_label}_count"] = int(count)

fit_to_post_counts = working_df["fit_to_post_tt"].value_counts()
for combo_value, count in fit_to_post_counts.items():
    combo_label = normalize_tt_label(combo_value)
    global_variables[f"fit_to_post_tt_{combo_label}_count"] = int(count)

# Final number of events
final_number_of_events = len(working_df)
print(f"Final number of events in the dataframe: {final_number_of_events}")
record_filter_metric(
    "total_rows_removed_pct",
    original_number_of_events - final_number_of_events,
    original_number_of_events if original_number_of_events else 0,
)

print(
    f"Writing corrected parquet: rows={len(working_df)} cols={len(working_df.columns)} -> {OUT_PATH}"
)
if VERBOSE:
    print("Columns before saving fit->corr parquet:")
    for col in working_df.columns:
        print(col)

# Data purity
data_purity = final_number_of_events / original_number_of_events * 100
global_variables['purity_of_data_percentage'] = data_purity

# Change 'datetime' column to 'Time' ------------------------------------------
if 'datetime' in working_df.columns:
    working_df.rename(columns={'datetime': 'Time'}, inplace=True)
else:
    print("Column 'datetime' not found in DataFrame!")

# End of the execution time
_prof["s_efficiency_s"] = round(time.perf_counter() - _t_sec, 2)
end_time_execution = datetime.now()
execution_time = end_time_execution - start_execution_time_counting
# In minutes
execution_time_minutes = execution_time.total_seconds() / 60

# To save as metadata
filename_base = basename_no_ext
execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
data_purity_percentage = data_purity
total_execution_time_minutes = execution_time_minutes
param_hash_value = str(simulated_param_hash) if simulated_param_hash else str(global_variables.get("param_hash", ""))
global_variables["param_hash"] = param_hash_value

# -------------------------------------------------------------------------------
# Filter metadata (ancillary) ---------------------------------------------------
# -------------------------------------------------------------------------------
if status_execution_date is not None:
    update_status_progress(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        completion_fraction=0.75,
        param_hash=str(global_variables.get("param_hash", "")),
    )

filter_metrics["data_purity_percentage"] = round(float(data_purity_percentage), 4)
filter_row = {
    "filename_base": filename_base,
    "execution_timestamp": execution_timestamp,
    "param_hash": param_hash_value,
}
for name in FILTER_METRIC_NAMES:
    filter_row[name] = filter_metrics.get(name, "")

metadata_filter_csv_path = save_metadata(
    csv_path_filter,
    filter_row,
    preferred_fieldnames=("filename_base", "execution_timestamp", "param_hash", *FILTER_METRIC_NAMES),
)
print(f"Metadata (filter) CSV updated at: {metadata_filter_csv_path}")

deep_fiter_row = {
    "filename_base": filename_base,
    "execution_timestamp": execution_timestamp,
    "param_hash": param_hash_value,
}
for name in sorted(filter_metrics):
    deep_fiter_row[name] = filter_metrics.get(name, "")

metadata_deep_fiter_csv_path = save_metadata(
    csv_path_deep_fiter,
    deep_fiter_row,
    preferred_fieldnames=(
        "filename_base",
        "execution_timestamp",
        "param_hash",
        *sorted(filter_metrics),
    ),
    replace_existing_basename=True,
)
print(f"Metadata (deep_fiter) CSV updated at: {metadata_deep_fiter_csv_path}")

# -------------------------------------------------------------------------------
# Execution metadata ------------------------------------------------------------
# -------------------------------------------------------------------------------

print("----------\nExecution metadata to be saved:")
print(f"Filename base: {filename_base}")
print(f"Execution timestamp: {execution_timestamp}")
print(f"Data purity percentage: {data_purity_percentage:.2f}%")
print(f"Total execution time: {total_execution_time_minutes:.2f} minutes")

execution_metadata_row = {
    "filename_base": filename_base,
    "execution_timestamp": execution_timestamp,
    "param_hash": param_hash_value,
    "data_purity_percentage": round(float(data_purity_percentage), 4),
    "total_execution_time_minutes": round(float(total_execution_time_minutes), 4),
}
apply_qa_reprocessing_context(execution_metadata_row, qa_reprocessing_context)

metadata_execution_csv_path = save_metadata(
    csv_path,
    execution_metadata_row,
    preferred_fieldnames=(
        "filename_base",
        "execution_timestamp",
        "param_hash",
        "data_purity_percentage",
        "total_execution_time_minutes",
        *QA_REPROCESSING_METADATA_KEYS,
    ),
)
print(f"Metadata (execution) CSV updated at: {metadata_execution_csv_path}")

# -------------------------------------------------------------------------------
# Specific metadata ------------------------------------------------------------
# -------------------------------------------------------------------------------

global_variables.update(build_events_per_second_metadata(working_df))
ensure_global_count_keys(("fit_tt", "post_tt", "fit_to_post_tt"))
add_normalized_count_metadata(
    global_variables,
    global_variables.get("events_per_second_total_seconds", 0),
)
set_global_rate_from_tt_rates(
    global_variables,
    preferred_prefixes=("post_tt", "fit_tt"),
    log_fn=print,
)
global_variables["filename_base"] = filename_base
global_variables["execution_timestamp"] = execution_timestamp
global_variables["param_hash"] = param_hash_value

rate_histogram_variables = extract_rate_histogram_metadata(global_variables, remove_from_source=True)
metadata_rate_histogram_csv_path = save_metadata(
    csv_path_rate_histogram,
    rate_histogram_variables,
)
print(f"Metadata (rate_histogram) CSV updated at: {metadata_rate_histogram_csv_path}")

prune_redundant_count_metadata(global_variables, log_fn=print)
trigger_type_prefixes = ("fit_tt", "post_tt", "fit_to_post_tt")
legacy_trigger_type_prefixes = ("corr_tt", "task5_to_corr_tt", "fit_to_corr_tt")
trigger_type_variables = extract_trigger_type_metadata(
    global_variables,
    trigger_type_prefixes,
    remove_from_source=True,
)
# Keep the denominator in trigger_type so rate_hz values can be converted back to counts.
trigger_type_variables["count_rate_denominator_seconds"] = rate_histogram_variables.get(
    "count_rate_denominator_seconds",
    0,
)
add_trigger_type_total_offender_threshold_metadata(
    trigger_type_variables,
    working_df,
    stage_tt_columns=("fit_tt", "post_tt"),
    denominator_seconds=trigger_type_variables["count_rate_denominator_seconds"],
)
metadata_trigger_type_csv_path = save_metadata(
    csv_path_trigger_type,
    trigger_type_variables,
    drop_field_predicate=lambda column_name: (
        (not is_trigger_type_file_column(column_name, trigger_type_prefixes))
        or is_trigger_type_metadata_column(
            column_name,
            legacy_trigger_type_prefixes,
        )
    ),
)
print(f"Metadata (trigger_type) CSV updated at: {metadata_trigger_type_csv_path}")

print(f"Specific metadata keys to be saved: {len(global_variables)}")
if VERBOSE:
    print("----------\nAll global variables to be saved:")
    for key, value in global_variables.items():
        print(f"{key}: {value}")
    print("----------\n")

print("----------\nSpecific metadata to be saved:")
print(f"Filename base: {filename_base}")
print(f"Execution timestamp: {execution_timestamp}")
print(f"------------- Any other variable interesting -------------")
print("\n----------")
print(
    f"[Z_TRACE] metadata_append filename_base={filename_base} "
    f"param_hash={simulated_param_hash or 'NA'} z_vector_mm={z_vector_mm}",
    force=True,
)

metadata_specific_csv_path = save_metadata(
    csv_path_specific,
    global_variables,
    drop_field_predicate=lambda column_name: (
        is_specific_metadata_excluded_column(column_name)
        or is_trigger_type_metadata_column(
            column_name,
            trigger_type_prefixes + legacy_trigger_type_prefixes,
        )
    ),
)
print(f"Metadata (specific) CSV updated at: {metadata_specific_csv_path}")

# Ensure no figure handles remain open before persistence/final move.
plt.close("all")

# Save to HDF5 file
working_df.to_parquet(
    OUT_PATH,
    engine="pyarrow",
    compression=_preferred_parquet_compression(),
    index=False,
)
# working_df.to_csv(OUT_PATH.replace('.h5', '.csv'), index=False)
print(f"Listed dataframe saved to: {OUT_PATH}")

# Move the original datafile to COMPLETED -------------------------------------
print("Moving file to COMPLETED directory...")

if user_file_selection == False:
    safe_move(file_path, completed_file_path)
    now = time.time()
    os.utime(completed_file_path, (now, now))
    print("************************************************************")
    print(f"File moved from\n{file_path}\nto:\n{completed_file_path}")
    print("************************************************************")

if status_execution_date is not None:
    update_status_progress(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        completion_fraction=1.0,
        param_hash=str(global_variables.get("param_hash", "")),
    )

_prof["filename_base"] = filename_base
_prof["execution_timestamp"] = execution_timestamp
_prof["param_hash"] = param_hash_value
_prof["total_s"] = round(time.perf_counter() - _prof_t0, 2)
save_metadata(csv_path_profiling, _prof)
