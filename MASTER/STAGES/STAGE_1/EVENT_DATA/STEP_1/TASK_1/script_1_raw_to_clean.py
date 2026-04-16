#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_1/script_1_raw_to_clean.py
Purpose: !/usr/bin/env python3.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_1/script_1_raw_to_clean.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

#%%

from __future__ import annotations

"""
Stage 1 Task 1 (RAW-->CLEAN) driver.

The script pulls the next available STAGE_0_to_1 acquisition, applies the full
raw cleaning chain (baseline removal, channel checks, derived quantities, and
pre-selection cuts), and persists the cleaned dataframe for downstream tasks.
Along the way it maintains the station staging directories, generates QA plots,
tracks execution metadata, and emits run-level summaries consumed by the rest
of the Stage 1 event workflow.
"""
# Standard Library
import argparse
import atexit
from array import array
import builtins
import csv
from datetime import datetime, timedelta
import gc
import hashlib
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
from typing import Dict, Iterable, List, Mapping, Tuple

# Scientific Computing
import numpy as np
import pandas as pd
import scipy.linalg as linalg
from scipy.constants import c
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, curve_fit, minimize_scalar
from scipy.special import erf
from scipy.stats import norm, poisson, linregress, median_abs_deviation, skew

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
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
    file_name_in_any_date_range,
    load_date_ranges_from_config,
    newest_order_key,
    select_latest_candidate,
    sync_unprocessed_with_date_range,
)
from MASTER.common.input_file_config import select_input_file_configuration
from MASTER.common.path_config import (
    get_master_config_root,
    get_repo_root,
    resolve_home_path_from_config,
)
from MASTER.common.plot_utils import (
    collect_saved_plot_paths,
    ensure_plot_state,
    pdf_save_rasterized_page,
)
from MASTER.common.selection_config import load_selection_for_paths, station_is_selected
from MASTER.common.status_csv import initialize_status_row, update_status_progress
from MASTER.common.reprocessing_utils import get_reprocessing_value
from MASTER.common.simulated_data_utils import SIM_PARAMS_DEFAULT, resolve_simulated_z_positions
from MASTER.common.step1_activation import (
    compute_conditional_matrices_by_tt,
    store_activation_matrices_by_tt_metadata,
    store_activation_matrix_metadata,
)
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
    select_exact_minimum_vertex_cover,
    set_global_rate_from_tt_rates,
    load_step1_task_plot_catalog,
    resolve_step1_plot_options,
    step1_task_plot_enabled,
    validate_step1_input_file_args,
    y_pos,
)

task_number = 1

# I want to chrono the execution time of the script
start_execution_time_counting = datetime.now()
_prof_t0 = time.perf_counter()
_prof = {}

STATION_CHOICES = ("0", "1", "2", "3", "4")
TASK1_PLOT_ALIASES: tuple[str, ...] = (
    "debug_suite",
    "usual_suite",
    "essential_suite",
    "event_total_charge_raw",
    "raw_tt_overview",
    "channel_histograms_raw",
    "tq_scatter_raw",
    "channel_histograms_filtered",
    "tq_scatter_filtered",
    "tsum_spread_diagnostics",
    "channel_contamination_matrix_32",
    "channel_contagion_by_tt",
    "channel_contamination_matrix_32_raw",
    "channel_contagion_by_tt_raw",
    "channel_histograms_self_trigger",
    "tsum_spread_histograms_filtered_og",
    "tq_scatter_raw_by_tt",
    "tq_scatter_filtered_by_tt",
    "channel_tq_matrix_by_planepair",
    "channel_combination_filter_by_tt",
)
task1_plot_status_by_alias: dict[str, str] = {}


def task1_plot_enabled(alias: str) -> bool:
    if not task1_plot_status_by_alias:
        return True
    return step1_task_plot_enabled(alias, task1_plot_status_by_alias, plot_mode)


def apply_task1_plot_catalog_modes() -> None:
    global create_plots, create_essential_plots, create_debug_plots, save_plots, create_pdf
    if not task1_plot_status_by_alias:
        return
    create_plots = create_plots and task1_plot_enabled("usual_suite")
    create_essential_plots = create_essential_plots and task1_plot_enabled("essential_suite")
    create_debug_plots = create_debug_plots and task1_plot_enabled("debug_suite")
    save_plots = bool(create_plots or create_essential_plots or create_debug_plots)
    create_pdf = save_plots


def _coerce_config_bool(value: object, default: bool = False) -> bool:
    """Interpret common config boolean encodings without changing existing defaults."""
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


def _coerce_config_value(value: object, cast_fn, default):
    try:
        return cast_fn(value)
    except (TypeError, ValueError):
        return default


def _task1_load_channel_combination_relation_limits(
    config_dict: dict,
    *,
    base_q_sum_left: float,
    base_q_sum_right: float,
    base_q_dif_threshold: float,
    base_t_sum_left: float,
    base_t_sum_right: float,
    base_t_dif_threshold: float,
) -> dict[str, dict[str, float]]:
    base_limits = {
        "q_sum_left": float(base_q_sum_left),
        "q_sum_right": float(base_q_sum_right),
        "q_dif_threshold": float(base_q_dif_threshold),
        "t_sum_left": float(base_t_sum_left),
        "t_sum_right": float(base_t_sum_right),
        "t_dif_threshold": float(base_t_dif_threshold),
    }
    relation_limits: dict[str, dict[str, float]] = {}
    relation_types = ("self", "same_strip", "same_plane", "any")
    bound_suffixes = (
        "q_sum_left",
        "q_sum_right",
        "q_dif_threshold",
        "t_sum_left",
        "t_sum_right",
        "t_dif_threshold",
    )
    for relation_type in relation_types:
        relation_limits[relation_type] = {
            suffix: float(
                config_dict.get(
                    f"channel_combination_{relation_type}_{suffix}",
                    base_limits[suffix],
                )
            )
            for suffix in bound_suffixes
        }
    return relation_limits

CLI_PARSER = build_step1_cli_parser("Run Stage 1 STEP_1 TASK_1 (RAW->CLEAN).", STATION_CHOICES)
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

def save_plot_figure(save_path: str, fig: mpl.figure.Figure | None = None, **savefig_kwargs) -> None:
    """Save a figure to PNG or directly append it to the task PDF."""
    global _direct_pdf_pages, _direct_pdf_page_count, _direct_pdf_target_path, _direct_pdf_temp_path
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


def finalize_saved_plots_to_pdf() -> None:
    if not create_pdf:
        return

    print(f"Creating PDF with all plots in {save_pdf_path}")
    existing_pngs = collect_saved_plot_paths(plot_list, base_directories["figure_directory"])

    if _direct_pdf_pages is not None:
        for png in existing_pngs:
            img = Image.open(png)
            fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
            ax.imshow(img)
            ax.axis('off')
            pdf_save_rasterized_page(_direct_pdf_pages, fig, bbox_inches='tight')
            plt.close(fig)
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

    for png in existing_pngs:
        try:
            os.remove(png)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")

    figure_directory = base_directories["figure_directory"]
    if os.path.exists(figure_directory):
        if not os.listdir(figure_directory):
            os.rmdir(figure_directory)
        else:
            print(f"Figure directory not empty, skipping removal: {figure_directory}")


atexit.register(close_direct_pdf_writer)

# Warning Filters
warnings.filterwarnings("ignore", message=".*Data has no positive values, and therefore cannot be log-scaled.*")

start_timer(__file__)
config_root = get_master_config_root()
config_file_path = config_root / "STAGE_1" / "EVENT_DATA" / "STEP_1" / "TASK_1" / "config_task_1.yaml"
plot_catalog_file_path = (
    config_root / "STAGE_1" / "EVENT_DATA" / "STEP_1" / "TASK_1" / "config_plots_task_1.yaml"
)
parameter_config_file_path = (
    config_root / "STAGE_1" / "EVENT_DATA" / "STEP_1" / "TASK_1" / "config_parameters_task_1.csv"
)
filter_parameter_config_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_1"
    / "config_filter_parameters_task_1.csv"
)
fallback_parameter_config_file_path = (
    config_root / "STAGE_1" / "EVENT_DATA" / "STEP_1" / "config_parameters.csv"
)
print(f"Using config file: {config_file_path}")
print(f"Using plot catalog file: {plot_catalog_file_path}")
with config_file_path.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
filter_parameter_config_file_path = filter_parameter_config_file_path.with_name(
    str(config.get("filter_parameter_config_csv", filter_parameter_config_file_path.name))
)
print(f"Using filter parameter config file: {filter_parameter_config_file_path}")
task1_plot_status_by_alias = load_step1_task_plot_catalog(
    plot_catalog_file_path,
    TASK1_PLOT_ALIASES,
    "Task 1",
    log_fn=print,
)
debug_mode = False

home_path = str(resolve_home_path_from_config(config))

# ~/DATAFLOW_v3/MASTER/ANCILLARY/QUALITY_ASSURANCE
REFERENCE_TABLES_DIR = Path(home_path) / "DATAFLOW_v3" / "MASTER" / "ANCILLARY" / "QUALITY_ASSURANCE" / "REFERENCE_TABLES"

# -----------------------------------------------------------------------------

# Round execution time to seconds and format it in YYYY-MM-DD_HH.MM.SS
execution_time = str(start_execution_time_counting).split('.')[0]  # Remove microseconds
print("Execution time is:", execution_time)

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

if CLI_ARGS.station is None:
    CLI_PARSER.error("No station provided. Pass <station>.")
station = str(CLI_ARGS.station)
set_station(station)

if filter_parameter_config_file_path.exists():
    config = update_config_with_parameters(config, filter_parameter_config_file_path, station)
    print(f"Warning: Loaded filter parameters from {filter_parameter_config_file_path}")

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

selection_config = load_selection_for_paths(
    [config_file_path],
    master_config_root=config_root,
)
if not station_is_selected(station, selection_config.stations):
    print(f"Station {station} skipped by selection.stations.", force=True)
    sys.exit(0)

selected_input_file = CLI_ARGS.input_file_flag or CLI_ARGS.input_file
if selected_input_file:
    user_file_path = selected_input_file
    user_file_selection = True
    print("User provided file path:", user_file_path)
else:
    user_file_selection = False

# -----------------------------------------------------------------------------

print("Creating the necessary directories...")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

# Define base working directory
repo_root = get_repo_root()
home_directory = str(repo_root.parent)
station_directory = str(repo_root / "STATIONS" / f"MINGO0{station}")
config_file_directory = str(
    config_root
    / "STAGE_0"
    / "NEW_FILES"
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
    "removed_channel_values_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/REMOVED_CHANNEL_VALUES"),
    "tracking_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/TRACKING"),
    
    "unprocessed_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/UNPROCESSED_DIRECTORY"),
    "out_of_date_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/OUT_OF_DATE_DIRECTORY"),
    "error_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/ERROR_DIRECTORY"),
    "processing_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/PROCESSING_DIRECTORY"),
    "completed_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/COMPLETED_DIRECTORY"),
    
    "output_directory": output_location,

    "raw_directory": os.path.join(raw_working_directory, "."),
    
    "metadata_directory": metadata_directory,
}

(
    plot_mode,
    create_plots,
    create_essential_plots,
    save_plots,
    create_pdf,
    show_plots,
    create_debug_plots,
) = resolve_step1_plot_options(config)
apply_task1_plot_catalog_modes()

debug_plot_directory = os.path.join(
    base_directories["base_plots_directory"],
    "DEBUG_PLOTS",
    f"FIGURES_EXEC_ON_{date_execution}",
)
debug_fig_idx = 1

def _debug_plot_filter_group(
    df: pd.DataFrame,
    columns: Iterable[str],
    thresholds: Iterable[float] | float | None,
    label: str,
    *,
    tag: str = "tunable",
    max_cols_per_fig: int = 16,
) -> None:
    """Emit debug histograms for a filter group with threshold lines."""
    global debug_fig_idx
    if not create_debug_plots:
        return

    cols = [col for col in columns if col in df.columns]
    if not cols:
        return

    if thresholds is None:
        threshold_values: list[float] = []
    elif isinstance(thresholds, (list, tuple, np.ndarray)):
        threshold_values = [float(v) for v in thresholds if v is not None]
    else:
        threshold_values = [float(thresholds)]

    debug_thresholds = {col: threshold_values for col in cols}
    debug_fig_idx = plot_debug_histograms(
        df,
        cols,
        debug_thresholds,
        title=f"Task 1 pre-filter: {label} [{tag}] (station {station})",
        out_dir=debug_plot_directory,
        fig_idx=debug_fig_idx,
        max_cols_per_fig=max_cols_per_fig,
    )

# Create ALL directories if they don't already exist
# Create ALL directories if they don't already exist
for directory in base_directories.values():
    # Skip figure directories at startup; create lazily after selecting a file.
    if directory in (base_directories["base_figure_directory"], base_directories["figure_directory"]):
        continue
    os.makedirs(directory, exist_ok=True)

csv_path = os.path.join(metadata_directory, f"task_{task_number}_metadata_execution.csv")
csv_path_specific = os.path.join(metadata_directory, f"task_{task_number}_metadata_specific.csv")
csv_path_pattern = os.path.join(metadata_directory, f"task_{task_number}_metadata_pattern.csv")
csv_path_rate_histogram = os.path.join(
    metadata_directory,
    f"task_{task_number}_metadata_rate_histogram.csv",
)
csv_path_activation = os.path.join(
    metadata_directory,
    f"task_{task_number}_metadata_activation.csv",
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
csv_path_noise_control = os.path.join(
    metadata_directory,
    f"task_{task_number}_metadata_noise_control.csv",
)
csv_path_status = os.path.join(metadata_directory, f"task_{task_number}_metadata_status.csv")
csv_path_profiling = os.path.join(metadata_directory, f"task_{task_number}_metadata_profiling.csv")
status_filename_base = ""
status_execution_date = None

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

raw_files = set(os.listdir(raw_directory))
unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))

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
raw_files = set(os.listdir(raw_directory))

# Take only in raw_files those in raw_directory that are strictly ending in *.dat
raw_files = {f for f in raw_files if f.lower().endswith('.dat')}

print("dat files are:", raw_files)

unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))

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
print(input_file_config_path)

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
    z_1 = 0
    z_2 = 150
    z_3 = 300
    z_4 = 450

ITINERARY_FILE_PATH = Path(
    f"{home_path}/DATAFLOW_v3/MASTER/CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1/TASK_2/TIME_CALIBRATION_ITINERARIES/itineraries.csv"
)

not_use_q_semisum = False

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
apply_task1_plot_catalog_modes()
limit_number = config.get("limit_number", None)
limit = last_file_test and limit_number is not None
force_replacement = config["force_replacement"]
article_format = config["article_format"]

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering
time_window_filtering = config["time_window_filtering"]

# Time calibration

# Y position

# RPC variables

# Alternative

# TimTrack

# Validation

EXPECTED_COLUMNS_config = config["EXPECTED_COLUMNS_config"]

complete_reanalysis = config["complete_reanalysis"]

limit_number = config.get("limit_number", None)
limit = last_file_test and limit_number is not None

# Pre-cal Front & Back
T_side_left_pre_cal_debug = config["T_side_left_pre_cal_debug"]
T_side_right_pre_cal_debug = config["T_side_right_pre_cal_debug"]
Q_side_left_pre_cal_debug = config["Q_side_left_pre_cal_debug"]
Q_side_right_pre_cal_debug = config["Q_side_right_pre_cal_debug"]

T_side_left_pre_cal_default = config["T_side_left_pre_cal_default"]
T_side_right_pre_cal_default = config["T_side_right_pre_cal_default"]
Q_side_left_pre_cal_default = config["Q_side_left_pre_cal_default"]
Q_side_right_pre_cal_default = config["Q_side_right_pre_cal_default"]

T_side_left_pre_cal_ST = config["T_side_left_pre_cal_ST"]
T_side_right_pre_cal_ST = config["T_side_right_pre_cal_ST"]
Q_side_left_pre_cal_ST = config["Q_side_left_pre_cal_ST"]
Q_side_right_pre_cal_ST = config["Q_side_right_pre_cal_ST"]

# Pre-cal Sum & Diff
T_dif_pre_cal_threshold = config.get("T_dif_pre_cal_threshold", 20)
channel_combination_q_sum_left = config.get(
    "channel_combination_q_sum_left",
    config.get("plane_combination_q_sum_left", Q_side_left_pre_cal_default),
)
channel_combination_q_sum_right = config.get(
    "channel_combination_q_sum_right",
    config.get("plane_combination_q_sum_right", Q_side_right_pre_cal_default),
)
channel_combination_q_dif_threshold = config.get(
    "channel_combination_q_dif_threshold",
    config.get("plane_combination_q_dif_threshold", 200),
)
channel_combination_t_sum_left = config.get(
    "channel_combination_t_sum_left",
    config.get("plane_combination_t_sum_left", T_side_left_pre_cal_default),
)
channel_combination_t_sum_right = config.get(
    "channel_combination_t_sum_right",
    config.get("plane_combination_t_sum_right", T_side_right_pre_cal_default),
)
channel_combination_t_dif_threshold = config.get(
    "channel_combination_t_dif_threshold",
    config.get("plane_combination_t_dif_threshold", T_dif_pre_cal_threshold),
)
task1_channel_combination_limits_by_relation = _task1_load_channel_combination_relation_limits(
    config,
    base_q_sum_left=channel_combination_q_sum_left,
    base_q_sum_right=channel_combination_q_sum_right,
    base_q_dif_threshold=channel_combination_q_dif_threshold,
    base_t_sum_left=channel_combination_t_sum_left,
    base_t_sum_right=channel_combination_t_sum_right,
    base_t_dif_threshold=channel_combination_t_dif_threshold,
)
print(
    "Task 1 channel-combination relations: "
    + ", ".join(
        f"{relation_type}[Qsum={limits['q_sum_left']}..{limits['q_sum_right']}, "
        f"Qdif=+/-{limits['q_dif_threshold']}, "
        f"Tsum={limits['t_sum_left']}..{limits['t_sum_right']}, "
        f"Tdif=+/-{limits['t_dif_threshold']}]"
        for relation_type, limits in task1_channel_combination_limits_by_relation.items()
    )
)
recalculate_noise_charge_limit = _coerce_config_bool(
    config.get("recalculate_noise_charge_limit", True),
    default=True,
)
noise_charge_limit_by_channel_config = config.get("noise_charge_limit_by_channel", {}) or {}
manual_channel_noise_charge_limits: dict[tuple[int, str, int], float] = {}
if isinstance(noise_charge_limit_by_channel_config, dict):
    for channel_label, limit_value in noise_charge_limit_by_channel_config.items():
        match = re.fullmatch(r"P(?P<plane>\d+)S(?P<strip>\d+)(?P<side>[FB])", str(channel_label).strip())
        if match is None:
            continue
        if limit_value in (None, "", "null", "None"):
            continue
        try:
            parsed_limit = float(limit_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(parsed_limit):
            continue
        manual_channel_noise_charge_limits[
            (
                int(match.group("plane")),
                str(match.group("side")),
                int(match.group("strip")),
            )
        ] = parsed_limit
print(
    "Task 1 noise-charge-limit mode: inactive "
    f"(legacy config kept for reference; configured channels: {len(manual_channel_noise_charge_limits)})"
)
channel_noise_limit_hist_bins = max(
    24,
    _coerce_config_value(config.get("channel_noise_limit_hist_bins", 48), int, 48),
)
channel_noise_limit_min_total_values = max(
    20,
    _coerce_config_value(config.get("channel_noise_limit_min_total_values", 80), int, 80),
)
channel_noise_limit_min_removed_values = max(
    5,
    _coerce_config_value(config.get("channel_noise_limit_min_removed_values", 12), int, 12),
)
channel_noise_limit_min_class_count = max(
    5,
    _coerce_config_value(config.get("channel_noise_limit_min_class_count", 10), int, 10),
)
channel_noise_limit_min_removed_left_fraction = min(
    1.0,
    max(
        0.0,
        _coerce_config_value(
            config.get("channel_noise_limit_min_removed_left_fraction", 0.7),
            float,
            0.7,
        ),
    ),
)
channel_noise_limit_min_log_separation = max(
    0.0,
    _coerce_config_value(config.get("channel_noise_limit_min_log_separation", 0.12), float, 0.12),
)

# Post-calibration

# Once calculated the RPC variables
T_sum_RPC_left = config.get("T_sum_RPC_left", -7)
T_sum_RPC_right = config.get("T_sum_RPC_right", 7)

# TimTrack filter
res_ystr_filter = config.get("res_ystr_filter", 500)
res_tsum_filter = config.get("res_tsum_filter", 3.5)
res_tdif_filter = config.get("res_tdif_filter", 1.0)

# Fitting comparison

# Calibrations
CRT_gaussian_fit_quantile = config.get("CRT_gaussian_fit_quantile", 0.03)
coincidence_window_og_ns = config["coincidence_window_og_ns"]
coincidence_window_cal_ns = config.get("coincidence_window_cal_ns", 3)

# Pedestal charge calibration
pedestal_left = config.get("pedestal_left", -1)
pedestal_right = config.get("pedestal_right", 3)

# Front-back charge
distance_sum_charges_left_fit = config.get("distance_sum_charges_left_fit", -5)
distance_sum_charges_right_fit = config.get("distance_sum_charges_right_fit", 200)
distance_dif_charges_up_fit = config.get("distance_dif_charges_up_fit", 10)
distance_dif_charges_low_fit = config.get("distance_dif_charges_low_fit", -10)
distance_sum_charges_plot = config.get("distance_sum_charges_plot", 800)
front_back_fit_threshold = config.get("front_back_fit_threshold", 4)

# Variables to modify
beta = config.get("beta", 1)
strip_speed_factor_of_c = config.get("strip_speed_factor_of_c", 0.666666667)

degree_of_polynomial = config.get("degree_of_polynomial", 4)

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
Q_clip_max_debug = config.get("Q_clip_max_debug", 600)
num_bins_debug = config["num_bins_debug"]

T_clip_min_default = config.get("T_clip_min_default", -300)
T_clip_max_default = config.get("T_clip_max_default", 100)
Q_clip_min_default = config.get("Q_clip_min_default", 0)
Q_clip_max_default = config.get("Q_clip_max_default", 500)
num_bins_default = config["num_bins_default"]

T_clip_min_ST = config.get("T_clip_min_ST", -300)
T_clip_max_ST = config.get("T_clip_max_ST", 100)
Q_clip_min_ST = config.get("Q_clip_min_ST", 0)
Q_clip_max_ST = config.get("Q_clip_max_ST", 500)

log_scale = config["log_scale"]
track_removed_rows = _coerce_config_bool(config.get("track_removed_rows", False), default=False)
keep_all_columns_output = _coerce_config_bool(
    config.get("keep_all_columns_output", False),
    default=False,
)
removed_marker = str(config.get("removed_marker", "x"))
removed_marker_size = _coerce_config_value(config.get("removed_marker_size", 30), int, 30)
removed_marker_alpha = _coerce_config_value(config.get("removed_marker_alpha", 0.9), float, 0.9)
_channel_combination_plot_top_n_raw = config.get("channel_combination_plot_top_n", 10)
if _channel_combination_plot_top_n_raw in (None, "", "null", "None"):
    channel_combination_plot_top_n = None
else:
    channel_combination_plot_top_n = max(
        1,
        _coerce_config_value(_channel_combination_plot_top_n_raw, int, 10),
    )
channel_combination_plot_include_same_strip_pairs = _coerce_config_bool(
    config.get("channel_combination_plot_include_same_strip_pairs", True),
    default=True,
)
channel_combination_plot_top_n_by_relation: dict[str, int | None] = {}
for _relation_type in ("self", "same_strip", "same_plane", "any"):
    _relation_top_n_raw = config.get(
        f"channel_combination_plot_top_n_{_relation_type}",
        channel_combination_plot_top_n,
    )
    if _relation_top_n_raw in (None, "", "null", "None"):
        channel_combination_plot_top_n_by_relation[_relation_type] = None
    else:
        channel_combination_plot_top_n_by_relation[_relation_type] = max(
            1,
            _coerce_config_value(_relation_top_n_raw, int, 10),
        )
_channel_pair_plot_top_n_raw = config.get("channel_pair_plot_top_n", None)
if _channel_pair_plot_top_n_raw in (None, "", "null", "None"):
    channel_pair_plot_top_n = None
else:
    channel_pair_plot_top_n = max(
        1,
        _coerce_config_value(_channel_pair_plot_top_n_raw, int, 10),
    )
calibrate_strip_Q_pedestal_thr_factor = config.get("calibrate_strip_Q_pedestal_thr_factor", 31.62)
calibrate_strip_Q_pedestal_thr_factor_2 = config.get("calibrate_strip_Q_pedestal_thr_factor_2", 1.5)
calibrate_strip_Q_pedestal_translate_charge_cal = config.get("calibrate_strip_Q_pedestal_translate_charge_cal", 0.25)

calibrate_strip_Q_pedestal_percentile = config.get("calibrate_strip_Q_pedestal_percentile", 10)
calibrate_strip_Q_pedestal_rel_th = config.get("calibrate_strip_Q_pedestal_rel_th", 0.015)
calibrate_strip_Q_pedestal_rel_th_cal = config.get("calibrate_strip_Q_pedestal_rel_th_cal", 0.4)
calibrate_strip_Q_pedestal_abs_th = config.get("calibrate_strip_Q_pedestal_abs_th", 3)
calibrate_strip_Q_pedestal_q_quantile = config.get("calibrate_strip_Q_pedestal_q_quantile", 0.4)

scatter_2d_and_fit_new_xlim_left = config.get("scatter_2d_and_fit_new_xlim_left", -5)
scatter_2d_and_fit_new_xlim_right = config.get("scatter_2d_and_fit_new_xlim_right", 200)
scatter_2d_and_fit_new_ylim_abs = abs(float(config.get("scatter_2d_and_fit_new_ylim_abs", config.get("scatter_2d_and_fit_new_ylim_top", 11))))
scatter_2d_and_fit_new_ylim_top = scatter_2d_and_fit_new_ylim_abs
scatter_2d_and_fit_new_ylim_bottom = -scatter_2d_and_fit_new_ylim_abs

calibrate_strip_T_dif_T_rel_th = config.get("calibrate_strip_T_dif_T_rel_th", 0.1)
calibrate_strip_T_dif_T_abs_th = config.get("calibrate_strip_T_dif_T_abs_th", 1)

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

fig_idx, plot_list = ensure_plot_state(globals())

# Time dif calibration (time_dif_reference)
time_dif_distance = 30
time_dif_reference = np.array([
    [-0.0573, 0.031275, 1.033875, 0.761475],
    [-0.914, -0.873975, -0.19815, 0.452025],
    [0.8769, 1.2008, 1.014, 2.43915],
    [1.508825, 2.086375, 1.6876, 3.023575]
])

# Charge sum pedestal (charge_sum_reference)
charge_sum_distance = 30
charge_sum_reference = np.array([
    [89.4319, 98.19605, 95.99055, 91.83875],
    [96.55775, 94.50385, 94.9254, 91.0775],
    [92.12985, 92.23395, 90.60545, 95.5214],
    [93.75635, 93.57425, 93.07055, 89.27305]
])

# Charge dif calibration (charge_dif_reference)
charge_dif_distance = 30
charge_dif_reference = np.array([
    [4.512, 0.58715, 1.3204, -1.3918],
    [-4.50885, 0.918, -3.39445, -0.12325],
    [-3.8931, -3.28515, 3.27295, 1.0554],
    [-2.29505, 0.012, 2.49045, -2.14565]
])

# Time sum calibration (time_sum_reference)
time_sum_distance = 30
time_sum_reference = np.array([
    [0.0, -0.3886308, -0.53020947, 0.33711737],
    [-0.80494094, -0.68836069, -2.01289387, -1.13481931],
    [-0.23899338, -0.51373738, 0.50845317, 0.11685095],
    [0.33586385, 1.08329847, 0.91410244, 0.58815813]
])

if False:
    print('Working in fast mode.')

if False:
    print('Working in debug mode.')

if False:
    T_F_left_pre_cal = T_side_left_pre_cal_debug
    T_F_right_pre_cal = T_side_right_pre_cal_debug

    T_B_left_pre_cal = T_side_left_pre_cal_debug
    T_B_right_pre_cal = T_side_right_pre_cal_debug

    Q_F_left_pre_cal = Q_side_left_pre_cal_debug
    Q_F_right_pre_cal = Q_side_right_pre_cal_debug

    Q_B_left_pre_cal = Q_side_left_pre_cal_debug
    Q_B_right_pre_cal = Q_side_right_pre_cal_debug
else:
    T_F_left_pre_cal = T_side_left_pre_cal_default  #-130
    T_F_right_pre_cal = T_side_right_pre_cal_default

    T_B_left_pre_cal = T_side_left_pre_cal_default
    T_B_right_pre_cal = T_side_right_pre_cal_default

    Q_F_left_pre_cal = Q_side_left_pre_cal_default
    Q_F_right_pre_cal = Q_side_right_pre_cal_default

    Q_B_left_pre_cal = Q_side_left_pre_cal_default
    Q_B_right_pre_cal = Q_side_right_pre_cal_default

T_F_left_pre_cal_ST = T_side_left_pre_cal_ST  #-115
T_F_right_pre_cal_ST = T_side_right_pre_cal_ST
T_B_left_pre_cal_ST = T_side_left_pre_cal_ST
T_B_right_pre_cal_ST = T_side_right_pre_cal_ST
Q_F_left_pre_cal_ST = Q_side_left_pre_cal_ST
Q_F_right_pre_cal_ST = Q_side_right_pre_cal_ST
Q_B_left_pre_cal_ST = Q_side_left_pre_cal_ST
Q_B_right_pre_cal_ST = Q_side_right_pre_cal_ST

Q_left_side = Q_side_left_pre_cal_ST
Q_right_side = Q_side_right_pre_cal_ST

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

# the analysis mode indicates if it is a regular analysis or a repeated, careful analysis
# 0 -> regular analysis
# 1 -> repeated, careful analysis
global_variables = {
    'analysis_mode': 0,
}

CHANNEL_CONTAGION_METADATA_FIELDS: tuple[str, ...] = (
    "mean_off_diagonal_global_raw",
    "max_off_diagonal_global_raw",
    "mean_off_diagonal_interplane_global_raw",
    "mean_off_diagonal_global_clean",
    "max_off_diagonal_global_clean",
    "mean_off_diagonal_interplane_global_clean",
    "mean_off_diagonal_by_tt_raw",
    "max_off_diagonal_by_tt_raw",
    "mean_off_diagonal_interplane_by_tt_raw",
    "mean_off_diagonal_by_tt_clean",
    "max_off_diagonal_by_tt_clean",
    "mean_off_diagonal_interplane_by_tt_clean",
)
channel_contagion_metrics: dict[str, float] = {
    key: float("nan") for key in CHANNEL_CONTAGION_METADATA_FIELDS
}
activation_metadata: dict[str, object] = {}
pattern_metadata: dict[str, object] = {}

TT_COUNT_VALUES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 12, 13, 14, 23, 24, 34, 123, 124, 134, 234, 1234
)
TT_COLOR_LABELS: tuple[str, ...] = tuple(str(tt_value) for tt_value in TT_COUNT_VALUES)
# Colour mapping for trigger-type labels.
# Single-plane labels ('1','2','3','4') are assigned a muted gray so the
# more informative multi-plane combinations receive the vivid palette colors.
TT_COLOR_CMAP = plt.get_cmap("tab10")
_palette = sns.color_palette("tab10", n_colors=10)
TT_COLOR_MAP: dict[str, tuple[float, float, float, float]] = {}
_multi_idx = 0
for tt_label in TT_COLOR_LABELS:
    if len(tt_label) == 1:
        # muted gray for single-plane events
        TT_COLOR_MAP[tt_label] = (0.60, 0.60, 0.60, 1.0)
    else:
        rgb = _palette[_multi_idx % len(_palette)]
        TT_COLOR_MAP[tt_label] = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        _multi_idx += 1

# Fallback default color
TT_COLOR_DEFAULT = (0.45, 0.45, 0.45, 1.0)


def get_tt_color(tt_value: object) -> tuple[float, float, float, float]:
    """Return a stable color for a trigger-type label across all Task 1 plots."""
    return TT_COLOR_MAP.get(normalize_tt_label(tt_value), TT_COLOR_DEFAULT)

def ensure_global_count_keys(prefixes: Iterable[str]) -> None:
    for prefix in prefixes:
        for tt_value in TT_COUNT_VALUES:
            global_variables.setdefault(f"{prefix}_{tt_value}_count", 0)


def refresh_global_count_metadata(df: pd.DataFrame, column_names: Iterable[str]) -> None:
    prefixes = tuple(f"{column_name}_" for column_name in column_names)
    for key in list(global_variables):
        if key.endswith("_count") and any(key.startswith(prefix) for prefix in prefixes):
            global_variables.pop(key, None)
    for column_name in column_names:
        if column_name not in df.columns:
            continue
        value_counts = df[column_name].value_counts()
        for tt_value, count in value_counts.items():
            tt_label = normalize_tt_label(tt_value)
            global_variables[f"{column_name}_{tt_label}_count"] = int(count)

TASK1_CHANNEL_PATTERN_ORDER: tuple[tuple[int, int, str], ...] = tuple(
    (plane, strip, side)
    for plane in range(1, 5)
    for strip in range(1, 5)
    for side in ("F", "B")
)


def _compute_channel_contagion_inputs(
    df: pd.DataFrame,
) -> tuple[list[str], np.ndarray] | None:
    """Return channel labels and active-mask matrix (rows=events, cols=channels)."""
    ch_labels: list[str] = []
    ch_cols: list[str] = []
    for _p, _s, _sd in TASK1_CHANNEL_PATTERN_ORDER:
        col = f"Q{_p}_{_sd}_{_s}"
        if col in df.columns:
            ch_labels.append(f"P{_p}S{_s}{_sd}")
            ch_cols.append(col)
    if not ch_cols:
        return None
    vals = df.loc[:, ch_cols].fillna(0).to_numpy(dtype=np.float32, copy=False)
    active = vals != 0
    return ch_labels, active


def _compute_channel_conditional_matrix_from_active(active: np.ndarray) -> np.ndarray:
    """Compute P(ch_j active | ch_i active) from boolean activity matrix."""
    co = active.T.astype(np.int64) @ active.astype(np.int64)
    diag = np.diag(co).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.divide(
            co.astype(float),
            diag[:, None],
            out=np.full_like(co, np.nan, dtype=float),
            where=diag[:, None] > 0,
        )
    return cond


def _compute_channel_conditional_matrix(df: pd.DataFrame) -> tuple[list[str], np.ndarray] | None:
    """Compute full-table conditional channel activation matrix."""
    inputs = _compute_channel_contagion_inputs(df)
    if inputs is None:
        return None
    ch_labels, active = inputs
    return ch_labels, _compute_channel_conditional_matrix_from_active(active)


def _compute_channel_conditional_matrix_for_tt(
    df: pd.DataFrame,
    tt_column: str,
    tt_value: int,
) -> tuple[list[str], np.ndarray] | None:
    """Compute conditional channel activation matrix for a specific TT value."""
    if tt_column not in df.columns:
        return None
    inputs = _compute_channel_contagion_inputs(df)
    if inputs is None:
        return None
    ch_labels, active = inputs
    tt_s = pd.to_numeric(df[tt_column], errors="coerce").fillna(0).astype(int)
    mask_tt = (tt_s == int(tt_value)).to_numpy(dtype=bool)
    if not np.any(mask_tt):
        return None
    return ch_labels, _compute_channel_conditional_matrix_from_active(active[mask_tt])


def _summarize_channel_conditional_matrix(cond: np.ndarray) -> tuple[float, float, float]:
    """Return mean/max off-diagonal and mean off-diagonal inter-plane values."""
    n_ch = int(cond.shape[0])
    if n_ch <= 1:
        return (float("nan"), float("nan"), float("nan"))

    idx = np.arange(n_ch)
    offdiag_mask = idx[:, None] != idx[None, :]
    offdiag_vals = cond[offdiag_mask]
    offdiag_finite = offdiag_vals[np.isfinite(offdiag_vals)]
    mean_offdiag = float(np.mean(offdiag_finite)) if offdiag_finite.size else float("nan")
    max_offdiag = float(np.max(offdiag_finite)) if offdiag_finite.size else float("nan")

    plane_ids = idx // 8
    interplane_mask = offdiag_mask & (plane_ids[:, None] != plane_ids[None, :])
    inter_vals = cond[interplane_mask]
    inter_finite = inter_vals[np.isfinite(inter_vals)]
    mean_interplane = float(np.mean(inter_finite)) if inter_finite.size else float("nan")
    return (mean_offdiag, max_offdiag, mean_interplane)


def _store_channel_contagion_metrics_variant(
    variant: str,
    matrix_data: tuple[list[str], np.ndarray] | None,
) -> None:
    """Store global/by-TT channel contagion summary metrics for a variant."""
    key_mean = f"mean_off_diagonal_{variant}"
    key_max = f"max_off_diagonal_{variant}"
    key_inter = f"mean_off_diagonal_interplane_{variant}"
    if matrix_data is None:
        channel_contagion_metrics[key_mean] = float("nan")
        channel_contagion_metrics[key_max] = float("nan")
        channel_contagion_metrics[key_inter] = float("nan")
        return
    _, cond = matrix_data
    mean_offdiag, max_offdiag, mean_interplane = _summarize_channel_conditional_matrix(cond)
    channel_contagion_metrics[key_mean] = mean_offdiag
    channel_contagion_metrics[key_max] = max_offdiag
    channel_contagion_metrics[key_inter] = mean_interplane

def _plot_channel_contamination_global(
    df: pd.DataFrame,
    stage_label: str,
    fig_idx_val: int,
    base_dir: str,
    save: bool,
    show: bool,
    plist: list[str],
) -> int:
    """Plot a single 32×32 channel contamination heatmap."""
    matrix_data = _compute_channel_conditional_matrix(df)
    if matrix_data is None:
        return fig_idx_val
    ch_labels, cond = matrix_data
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cond, vmin=0.0, vmax=1.0, cmap="viridis",
                xticklabels=ch_labels, yticklabels=ch_labels, ax=ax,
                cbar_kws={"label": "P(ch j active | ch i active)"})
    ax.set_title(f"Task 1 {stage_label} — 32-channel contamination matrix")
    ax.set_xlabel("Channel j")
    ax.set_ylabel("Channel i")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    fig.tight_layout()
    if save:
        fname = f"{fig_idx_val}_{stage_label.lower().replace(' ', '_')}_channel_contamination_matrix_32.png"
        fig_idx_val += 1
        spath = os.path.join(base_dir, fname)
        plist.append(spath)
        save_plot_figure(spath, fig=fig, format="png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig_idx_val


def _plot_channel_contagion_by_tt(
    df: pd.DataFrame,
    tt_column: str,
    stage_label: str,
    fig_idx_val: int,
    base_dir: str,
    save: bool,
    show: bool,
    plist: list[str],
    min_events: int = 30,
    max_tt_panels: int = 6,
) -> int:
    """Plot 32-channel contagion matrices split by TT."""
    required_tts: tuple[int, int] = (124, 134)
    inputs = _compute_channel_contagion_inputs(df)
    if inputs is None or tt_column not in df.columns:
        return fig_idx_val
    ch_labels_global, active = inputs
    n_ch = len(ch_labels_global)
    ch_labels = [lbl.replace("S", "") for lbl in ch_labels_global]
    tt_s = pd.to_numeric(df[tt_column], errors="coerce").fillna(0).astype(int)
    tt_cts = tt_s.value_counts()
    required_present = [tt for tt in required_tts if int(tt_cts.get(tt, 0)) > 0]
    optional_pool = [
        tt for tt, cnt in tt_cts.items()
        if tt >= 10 and cnt >= min_events and tt not in required_tts
    ]
    # Keep required TTs visible even when panel count is capped.
    sel_tts = required_present + optional_pool
    sel_tts = sel_tts[:max(max_tt_panels, len(required_present))]
    if not sel_tts:
        return fig_idx_val
    ncols_tt = len(sel_tts)
    fig_w = max(7, 6.0 * ncols_tt)
    fig, axes = plt.subplots(1, ncols_tt, figsize=(fig_w, 7.0 + 0.15 * n_ch), squeeze=False)
    _im = None
    for ci, tv in enumerate(sel_tts):
        ax = axes[0, ci]
        mtt = (tt_s == tv).to_numpy(dtype=bool)
        ntt = int(np.sum(mtt))
        cond = _compute_channel_conditional_matrix_from_active(active[mtt])
        _im = ax.imshow(cond, cmap="viridis", vmin=0, aspect="equal")
        ax.set_xticks(range(n_ch))
        ax.set_xticklabels(ch_labels, fontsize=4, rotation=90)
        ax.set_yticks(range(n_ch))
        ax.set_yticklabels(ch_labels, fontsize=4)
        for bnd in range(8, n_ch, 8):
            ax.axhline(bnd - 0.5, color="grey", linewidth=0.7, alpha=0.6)
            ax.axvline(bnd - 0.5, color="grey", linewidth=0.7, alpha=0.6)
        ax.set_title(f"TT {tv} (N={ntt})", fontsize=10)
    if _im is not None:
        fig.colorbar(_im, ax=axes[0, -1], label="P(ch j | ch i)", shrink=0.75, pad=0.02)
    fig.suptitle(
        f"Task 1 {stage_label} — channel contagion by TT\n"
        "P(channel j active | channel i active)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.97, 0.93])
    if save:
        fname = f"{fig_idx_val}_{stage_label.lower().replace(' ', '_')}_channel_contagion_by_tt.png"
        fig_idx_val += 1
        spath = os.path.join(base_dir, fname)
        plist.append(spath)
        save_plot_figure(spath, fig=fig, format="png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig_idx_val


TASK1_SELECTED_OFFENDER_CARDINALITY_VALUES: tuple[int, ...] = tuple(range(0, 33))
TASK1_THREE_TO_FOUR_MISSING_TRIGGER_BY_PLANE: dict[int, int] = {
    1: 234,
    2: 134,
    3: 124,
    4: 123,
}
TASK1_CHANNEL_COMBINATION_RELATION_TYPES: tuple[str, ...] = (
    "self",
    "same_strip",
    "same_plane",
    "any",
)
TASK1_CHANNEL_COMBINATION_BOUND_SUFFIXES: tuple[str, ...] = (
    "q_sum_left",
    "q_sum_right",
    "q_dif_threshold",
    "t_sum_left",
    "t_sum_right",
    "t_dif_threshold",
)

FILTER_METRIC_NAMES: tuple[str, ...] = (
    "plane_combination_filter_rows_affected_pct",
    "plane_combination_filter_rows_with_self_relation_failures_pct",
    "plane_combination_filter_rows_with_same_strip_relation_failures_pct",
    "plane_combination_filter_rows_with_same_plane_relation_failures_pct",
    "plane_combination_filter_rows_with_any_relation_failures_pct",
    "valid_lines_in_binary_file_percentage",
    "data_purity_percentage",
    "clean_tt_lt_10_rows_removed_pct",
)
NOISE_CONTROL_RATE_DENOMINATOR_COLUMN = "count_rate_denominator_seconds"
TASK1_NOISE_CONTROL_METRIC_NAMES: tuple[str, ...] = tuple(
    f"plane_combination_filter_rows_with_{selected_count}_selected_offenders_rate_hz"
    for selected_count in TASK1_SELECTED_OFFENDER_CARDINALITY_VALUES
)
TASK1_NOISE_CONTROL_PERCENT_METRIC_NAMES: tuple[str, ...] = tuple(
    f"plane_combination_filter_rows_with_{selected_count}_selected_offenders_pct"
    for selected_count in TASK1_SELECTED_OFFENDER_CARDINALITY_VALUES
)
FILTER_METADATA_ALLOWED_COLUMNS = {
    "filename_base",
    "execution_timestamp",
    "param_hash",
    *FILTER_METRIC_NAMES,
}

filter_metrics: dict[str, float] = {}

def record_activity_metric(name: str, affected: float, total: float, label: str = "affected") -> None:
    """Record a generic percentage metric."""
    pct = 0.0 if total == 0 else 100.0 * float(affected) / float(total)
    filter_metrics[name] = round(pct, 4)
    print(f"[filter-metrics] {name}: {label} {affected} of {total} ({pct:.2f}%)")

def record_filter_metric(name: str, removed: float, total: float) -> None:
    """Record percentage removed for a filter."""
    record_activity_metric(name, removed, total, label="removed")


def collect_task1_channel_qt_map(df_input: pd.DataFrame) -> dict[tuple[int, str, int], dict[str, str]]:
    """Return the per-channel Q/T column mapping for Task 1."""
    channel_map: dict[tuple[int, str, int], dict[str, str]] = {}
    for plane in range(1, 5):
        for side in ("F", "B"):
            for strip in range(1, 5):
                q_col = f"Q{plane}_{side}_{strip}"
                t_col = f"T{plane}_{side}_{strip}"
                if q_col in df_input.columns and t_col in df_input.columns:
                    channel_map[(plane, side, strip)] = {"Q": q_col, "T": t_col}
    return channel_map


TASK1_CHANNEL_KEYS: tuple[tuple[int, str, int], ...] = tuple(
    (plane, side, strip)
    for plane in range(1, 5)
    for side in ("F", "B")
    for strip in range(1, 5)
)


def _task1_channel_order_key(channel_key: tuple[int, str, int]) -> tuple[int, int, int]:
    plane, side, strip = channel_key
    side_rank = 0 if side == "F" else 1
    return plane, side_rank, strip


def _task1_channel_offender_metric_key(channel_key: tuple[int, str, int]) -> str:
    plane, side, strip = channel_key
    return f"plane_combination_filter_offender_count_P{plane}{side}{strip}"


def _task1_channel_noise_limit_metric_key(channel_key: tuple[int, str, int]) -> str:
    plane, side, strip = channel_key
    return f"plane_combination_noise_limit_P{plane}{side}{strip}"


def _task1_selected_offender_cardinality_metric_key(selected_count: int) -> str:
    return f"plane_combination_filter_rows_with_{selected_count}_selected_offenders"


def _task1_selected_offender_cardinality_rate_metric_key(selected_count: int) -> str:
    return f"{_task1_selected_offender_cardinality_metric_key(selected_count)}_rate_hz"


def _task1_selected_offender_cardinality_percent_metric_key(selected_count: int) -> str:
    return f"{_task1_selected_offender_cardinality_metric_key(selected_count)}_pct"


def _task1_noise_control_efficiency_metric_key(plane: int, selected_count_threshold: int) -> str:
    return (
        f"plane_combination_filter_eff_p{plane}_selected_offenders_le_"
        f"{selected_count_threshold}"
    )


def _task1_noise_control_efficiency_metric_names(
    selected_count_thresholds: Iterable[int],
) -> tuple[str, ...]:
    return tuple(
        _task1_noise_control_efficiency_metric_key(plane, selected_count_threshold)
        for selected_count_threshold in selected_count_thresholds
        for plane in sorted(TASK1_THREE_TO_FOUR_MISSING_TRIGGER_BY_PLANE)
    )


task1_noise_control_efficiency_max_selected_offenders = min(
    TASK1_SELECTED_OFFENDER_CARDINALITY_VALUES[-1],
    max(
        0,
        _coerce_config_value(
            config.get("noise_control_efficiency_max_selected_offenders", 5),
            int,
            5,
        ),
    ),
)
task1_noise_control_efficiency_selected_offender_values: tuple[int, ...] = tuple(
    range(task1_noise_control_efficiency_max_selected_offenders + 1)
)


def _task1_channel_label(channel_key: tuple[int, str, int]) -> str:
    plane, side, strip = channel_key
    return f"P{plane}S{strip}{side}"


def _task1_channel_relation_type(
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


def _iter_task1_channel_relation_pairs(
    channel_map: dict[tuple[int, str, int], dict[str, str]],
) -> Iterable[tuple[tuple[int, str, int], tuple[int, str, int], str]]:
    ordered_channels = sorted(channel_map, key=_task1_channel_order_key)
    for idx, channel_a in enumerate(ordered_channels):
        yield channel_a, channel_a, "self"
        for channel_b in ordered_channels[idx + 1 :]:
            yield channel_a, channel_b, _task1_channel_relation_type(channel_a, channel_b)


def _task1_default_q_left_for_channel(channel_key: tuple[int, str, int]) -> float:
    _, side, _ = channel_key
    return float(Q_F_left_pre_cal if side == "F" else Q_B_left_pre_cal)


def _empty_task1_removed_channel_values_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "row_index",
            "channel",
            "plane",
            "side",
            "strip",
            "Q",
            "T",
            "pass_label",
            "Q_zeroed",
            "T_zeroed",
        ]
    )


def _task1_build_removed_channel_values_frame(
    df_input: pd.DataFrame,
    channel_key: tuple[int, str, int],
    q_values: pd.Series,
    t_values: pd.Series,
    fail_mask: np.ndarray | pd.Series,
    pass_label: str,
) -> pd.DataFrame:
    aligned_mask = pd.Series(fail_mask, index=df_input.index, dtype=bool).reindex(df_input.index, fill_value=False)
    if not aligned_mask.any():
        return _empty_task1_removed_channel_values_frame()

    plane, side, strip = channel_key
    rows = pd.DataFrame(
        {
            "row_index": df_input.index[aligned_mask].to_numpy(),
            "channel": _task1_channel_label(channel_key),
            "plane": plane,
            "side": side,
            "strip": strip,
            "Q": pd.to_numeric(q_values.loc[aligned_mask], errors="coerce").to_numpy(dtype=float),
            "T": pd.to_numeric(t_values.loc[aligned_mask], errors="coerce").to_numpy(dtype=float),
            "pass_label": pass_label,
        }
    )
    return rows.reset_index(drop=True)


def collect_task1_zeroed_channel_values(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    pass_label: str,
) -> pd.DataFrame:
    if reference_df.empty or current_df.empty:
        return _empty_task1_removed_channel_values_frame()

    channel_map = collect_task1_channel_qt_map(reference_df)
    if not channel_map:
        return _empty_task1_removed_channel_values_frame()

    aligned_reference = reference_df.reindex(current_df.index).copy()
    payload_frames: list[pd.DataFrame] = []
    for channel_key, cols in channel_map.items():
        if cols["Q"] not in current_df.columns or cols["T"] not in current_df.columns:
            continue
        ref_q = pd.to_numeric(aligned_reference[cols["Q"]], errors="coerce").fillna(0)
        ref_t = pd.to_numeric(aligned_reference[cols["T"]], errors="coerce").fillna(0)
        cur_q = pd.to_numeric(current_df[cols["Q"]], errors="coerce").fillna(0)
        cur_t = pd.to_numeric(current_df[cols["T"]], errors="coerce").fillna(0)
        q_zeroed = (ref_q != 0) & (cur_q == 0)
        t_zeroed = (ref_t != 0) & (cur_t == 0)
        zeroed_mask = q_zeroed | t_zeroed
        if not zeroed_mask.any():
            continue

        channel_payload = _task1_build_removed_channel_values_frame(
            aligned_reference,
            channel_key,
            ref_q,
            ref_t,
            zeroed_mask.to_numpy(dtype=bool),
            pass_label,
        )
        if channel_payload.empty:
            continue
        channel_payload["Q_zeroed"] = q_zeroed.reindex(channel_payload["row_index"]).to_numpy(dtype=bool)
        channel_payload["T_zeroed"] = t_zeroed.reindex(channel_payload["row_index"]).to_numpy(dtype=bool)
        payload_frames.append(channel_payload)

    if not payload_frames:
        return _empty_task1_removed_channel_values_frame()
    return pd.concat(payload_frames, ignore_index=True)


def _task1_estimate_channel_noise_limit(
    channel_key: tuple[int, str, int],
    q_all: np.ndarray,
    q_removed: np.ndarray,
) -> float | None:
    q_all = q_all[np.isfinite(q_all) & (q_all > 0)]
    q_removed = q_removed[np.isfinite(q_removed) & (q_removed > 0)]
    if (
        q_all.size < channel_noise_limit_min_total_values
        or q_removed.size < channel_noise_limit_min_removed_values
    ):
        return None

    log_q_all = np.log10(q_all)
    low = float(np.nanmin(log_q_all))
    high = float(np.nanmax(log_q_all))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    n_bins = min(channel_noise_limit_hist_bins, max(24, int(np.sqrt(log_q_all.size))))
    hist, edges = np.histogram(log_q_all, bins=n_bins, range=(low, high))
    if hist.sum() == 0 or np.count_nonzero(hist) < 2:
        return None

    mids = 0.5 * (edges[:-1] + edges[1:])
    probs = hist.astype(float) / float(hist.sum())
    omega = np.cumsum(probs)
    mu = np.cumsum(probs * mids)
    mu_total = float(mu[-1])
    sigma_b2 = np.full_like(mids, -np.inf, dtype=float)
    valid = (omega > 0) & (omega < 1)
    sigma_b2[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / (
        omega[valid] * (1.0 - omega[valid])
    )
    if not np.isfinite(sigma_b2).any():
        return None

    threshold_idx = int(np.nanargmax(sigma_b2))
    threshold_log_q = float(edges[threshold_idx + 1])

    left_mask = log_q_all <= threshold_log_q
    right_mask = log_q_all > threshold_log_q
    if (
        int(np.count_nonzero(left_mask)) < channel_noise_limit_min_class_count
        or int(np.count_nonzero(right_mask)) < channel_noise_limit_min_class_count
    ):
        return None

    log_q_removed = np.log10(q_removed)
    removed_left_fraction = float(np.mean(log_q_removed <= threshold_log_q))
    if removed_left_fraction < channel_noise_limit_min_removed_left_fraction:
        return None

    class_separation = float(np.mean(log_q_all[right_mask]) - np.mean(log_q_all[left_mask]))
    if not np.isfinite(class_separation) or class_separation < channel_noise_limit_min_log_separation:
        return None

    learned_limit = float(10 ** threshold_log_q)
    return max(learned_limit, _task1_default_q_left_for_channel(channel_key))


def derive_task1_channel_noise_limits(
    df_input: pd.DataFrame,
    removed_channel_values: pd.DataFrame,
) -> dict[tuple[int, str, int], float]:
    if removed_channel_values.empty:
        return {}

    learned_limits: dict[tuple[int, str, int], float] = {}
    channel_map = collect_task1_channel_qt_map(df_input)
    grouped = removed_channel_values.groupby(["plane", "side", "strip"], sort=False)
    for (plane, side, strip), group in grouped:
        channel_key = (int(plane), str(side), int(strip))
        if channel_key not in channel_map:
            continue
        cols = channel_map[channel_key]
        q_all = pd.to_numeric(df_input[cols["Q"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_removed = pd.to_numeric(group.get("Q"), errors="coerce").dropna().to_numpy(dtype=float)
        learned_limit = _task1_estimate_channel_noise_limit(channel_key, q_all, q_removed)
        if learned_limit is not None and np.isfinite(learned_limit):
            learned_limits[channel_key] = float(learned_limit)
    return learned_limits


def build_task1_effective_q_left_limits(
    learned_limits: dict[tuple[int, str, int], float],
) -> dict[tuple[int, str, int], float]:
    return {
        channel_key: float(learned_limits.get(channel_key, _task1_default_q_left_for_channel(channel_key)))
        for channel_key in TASK1_CHANNEL_KEYS
    }


def apply_task1_channel_noise_limit_filter(
    df_input: pd.DataFrame,
    *,
    learned_q_limits: dict[tuple[int, str, int], float],
    snapshot_originals=None,
    removed_value_pass_label: str | None = None,
) -> dict[str, object]:
    channel_map = collect_task1_channel_qt_map(df_input)
    if len(channel_map) < 1 or not learned_q_limits:
        return {
            "tracked_channel_count": len(channel_map),
            "channels_with_learned_limits": 0,
            "rows_affected": 0,
            "values_zeroed": 0,
            "removed_channel_values": _empty_task1_removed_channel_values_frame(),
        }

    n_rows = len(df_input)
    any_row_affected = np.zeros(n_rows, dtype=bool)
    changed_columns: list[str] = []
    removed_value_frames: list[pd.DataFrame] = []
    values_zeroed = 0
    channels_with_learned_limits = 0

    channel_fail_masks: dict[tuple[int, str, int], np.ndarray] = {}
    for channel_key in sorted(learned_q_limits, key=_task1_channel_order_key):
        if channel_key not in channel_map:
            continue
        limit_value = learned_q_limits.get(channel_key)
        if limit_value is None or not np.isfinite(limit_value):
            continue

        cols = channel_map[channel_key]
        q_values = pd.to_numeric(df_input[cols["Q"]], errors="coerce").fillna(0)
        t_values = pd.to_numeric(df_input[cols["T"]], errors="coerce").fillna(0)
        q_array = q_values.to_numpy(dtype=float)
        fail_mask = (q_array != 0) & (q_array <= float(limit_value))
        if not np.any(fail_mask):
            continue

        channels_with_learned_limits += 1
        channel_fail_masks[channel_key] = fail_mask
        any_row_affected |= fail_mask
        changed_columns.extend((cols["Q"], cols["T"]))
        values_zeroed += int((q_values[fail_mask] != 0).sum())
        values_zeroed += int((t_values[fail_mask] != 0).sum())
        if removed_value_pass_label is not None:
            removed_value_frames.append(
                _task1_build_removed_channel_values_frame(
                    df_input,
                    channel_key,
                    q_values,
                    t_values,
                    fail_mask,
                    removed_value_pass_label,
                )
            )

    if changed_columns and snapshot_originals is not None:
        snapshot_originals(df_input, list(dict.fromkeys(changed_columns)))

    for channel_key, fail_mask in channel_fail_masks.items():
        cols = channel_map[channel_key]
        df_input.loc[fail_mask, [cols["Q"], cols["T"]]] = 0

    removed_values_df = (
        pd.concat(removed_value_frames, ignore_index=True)
        if removed_value_frames
        else _empty_task1_removed_channel_values_frame()
    )
    return {
        "tracked_channel_count": len(channel_map),
        "channels_with_learned_limits": channels_with_learned_limits,
        "rows_affected": int(np.count_nonzero(any_row_affected)),
        "values_zeroed": int(values_zeroed),
        "removed_channel_values": removed_values_df,
    }


def apply_task1_channel_qt_mismatch_filter(
    df_input: pd.DataFrame,
    *,
    snapshot_originals=None,
) -> dict[str, object]:
    task1_channel_pairs: list[tuple[str, str]] = []
    for plane in range(1, 5):
        for side in ("F", "B"):
            for strip in range(1, 5):
                q_col = f"Q{plane}_{side}_{strip}"
                t_col = f"T{plane}_{side}_{strip}"
                if q_col in df_input.columns and t_col in df_input.columns:
                    task1_channel_pairs.append((q_col, t_col))

    row_mismatch_mask = pd.Series(False, index=df_input.index)
    q_only_row_mask = pd.Series(False, index=df_input.index)
    t_only_row_mask = pd.Series(False, index=df_input.index)
    front_row_mask = pd.Series(False, index=df_input.index)
    back_row_mask = pd.Series(False, index=df_input.index)
    channel_qt_mismatch_channels = 0
    channel_qt_values_zeroed = 0

    if task1_channel_pairs and not df_input.empty:
        for q_col, t_col in task1_channel_pairs:
            q_values = pd.to_numeric(df_input[q_col], errors="coerce").fillna(0)
            t_values = pd.to_numeric(df_input[t_col], errors="coerce").fillna(0)
            q_only_mask = (q_values != 0) & (t_values == 0)
            t_only_mask = (q_values == 0) & (t_values != 0)
            mismatch_mask = q_only_mask | t_only_mask
            if not mismatch_mask.any():
                continue
            if snapshot_originals is not None:
                snapshot_originals(df_input, [q_col, t_col])
            row_mismatch_mask |= mismatch_mask
            q_only_row_mask |= q_only_mask
            t_only_row_mask |= t_only_mask
            if "_F_" in q_col:
                front_row_mask |= mismatch_mask
            elif "_B_" in q_col:
                back_row_mask |= mismatch_mask
            channel_qt_mismatch_channels += int(mismatch_mask.sum())
            channel_qt_values_zeroed += int((q_values[mismatch_mask] != 0).sum())
            channel_qt_values_zeroed += int((t_values[mismatch_mask] != 0).sum())
            df_input.loc[mismatch_mask, [q_col, t_col]] = 0

    return {
        "task1_channel_pairs": task1_channel_pairs,
        "row_mismatch_mask": row_mismatch_mask,
        "rows_affected": int(row_mismatch_mask.sum()),
        "values_zeroed": int(channel_qt_values_zeroed),
        "q_only_rows": int(q_only_row_mask.sum()),
        "t_only_rows": int(t_only_row_mask.sum()),
        "front_rows": int(front_row_mask.sum()),
        "back_rows": int(back_row_mask.sum()),
        "mismatch_channels": int(channel_qt_mismatch_channels),
    }


_TASK1_COMBO_LABEL_PATTERN = re.compile(
    r"^P(?P<plane_a>\d+)S(?P<strip_a>\d+)(?P<side_a>[FB])-P(?P<plane_b>\d+)S(?P<strip_b>\d+)(?P<side_b>[FB])$"
)


def _task1_parse_combo_label(
    combo_label: str,
) -> tuple[tuple[int, str, int], tuple[int, str, int]] | None:
    match = _TASK1_COMBO_LABEL_PATTERN.match(str(combo_label))
    if match is None:
        return None
    return (
        (
            int(match.group("plane_a")),
            match.group("side_a"),
            int(match.group("strip_a")),
        ),
        (
            int(match.group("plane_b")),
            match.group("side_b"),
            int(match.group("strip_b")),
        ),
    )


def apply_task1_plane_combination_filter(
    df_input: pd.DataFrame,
    *,
    relation_limits_by_type: Mapping[str, Mapping[str, float]],
    snapshot_originals=None,
    apply_changes: bool = True,
    removed_value_pass_label: str | None = None,
) -> dict[str, object]:
    """
    Apply the Task 1 channel-combination filter directly from channel Q/T values.

    The filter is evaluated in one pass over four mutually-exclusive relation classes:
    ``self`` (channel with itself), ``same_strip`` (front/back of the same strip),
    ``same_plane`` (different strips within the same plane), and ``any`` (different
    planes). Each relation class has its own Q_sum, Q_dif, T_sum, and T_dif bounds.

    Failed self-relations become mandatory offending channels. Failed non-self
    relations are solved per event with the exact minimum-cardinality channel set
    that covers all failed edges.
    """
    channel_map = collect_task1_channel_qt_map(df_input)
    if not channel_map:
        return {
            "input_rows": len(df_input),
            "tracked_channel_count": len(channel_map),
            "valid_pair_observations": 0,
            "failed_pair_any": 0,
            "failed_pair_q_sum": 0,
            "failed_pair_q_dif": 0,
            "failed_pair_t_sum": 0,
            "failed_pair_t_dif": 0,
            "rows_affected": 0,
            "values_zeroed": 0,
            "flagged_rows": 0,
            "selected_offender_channels": 0,
            "rows_with_multiple_offenders": 0,
            "max_failed_pairs_in_row": 0,
            "max_selected_offenders_in_row": 0,
            "valid_pair_observations_by_relation": {
                relation_type: 0
                for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
            },
            "failed_pair_any_by_relation": {
                relation_type: 0
                for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
            },
            "rows_with_relation_failures": {
                relation_type: 0
                for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
            },
            "selected_offender_cardinality_counts": {
                selected_count: 0
                for selected_count in TASK1_SELECTED_OFFENDER_CARDINALITY_VALUES
            },
            "selected_offender_counts": {
                channel_key: 0 for channel_key in TASK1_CHANNEL_KEYS
            },
            "selected_offender_count_by_row": pd.Series(0, index=df_input.index, dtype=int),
            "resolution_exact_by_row": pd.Series(True, index=df_input.index, dtype=bool),
            "removed_channel_values": _empty_task1_removed_channel_values_frame(),
        }

    n_rows = len(df_input)
    summary = {
        "input_rows": n_rows,
        "tracked_channel_count": len(channel_map),
        "valid_pair_observations": 0,
        "failed_pair_any": 0,
        "failed_pair_q_sum": 0,
        "failed_pair_q_sum_low": 0,
        "failed_pair_q_sum_high": 0,
        "failed_pair_q_dif": 0,
        "failed_pair_t_sum": 0,
        "failed_pair_t_sum_low": 0,
        "failed_pair_t_sum_high": 0,
        "failed_pair_t_dif": 0,
        "rows_affected": 0,
        "values_zeroed": 0,
        "flagged_rows": 0,
        "selected_offender_channels": 0,
        "rows_with_multiple_offenders": 0,
        "max_failed_pairs_in_row": 0,
        "max_selected_offenders_in_row": 0,
        "valid_pair_observations_by_relation": {
            relation_type: 0
            for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
        },
        "failed_pair_any_by_relation": {
            relation_type: 0
            for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
        },
        "failed_pair_q_sum_low_by_relation": {
            relation_type: 0
            for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
        },
        "failed_pair_q_sum_high_by_relation": {
            relation_type: 0
            for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
        },
        "failed_pair_t_sum_low_by_relation": {
            relation_type: 0
            for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
        },
        "failed_pair_t_sum_high_by_relation": {
            relation_type: 0
            for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
        },
        "rows_with_relation_failures": {
            relation_type: 0
            for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES
        },
        "selected_offender_cardinality_counts": {
            selected_count: 0
            for selected_count in TASK1_SELECTED_OFFENDER_CARDINALITY_VALUES
        },
    }
    row_failed_edges: dict[int, list[tuple[tuple[int, str, int], tuple[int, str, int], float]]] = {}
    row_forced_channels: dict[int, set[tuple[int, str, int]]] = {}
    row_failure_relations: dict[int, set[str]] = {}
    channel_fail_masks = {
        channel_key: np.zeros(n_rows, dtype=bool)
        for channel_key in channel_map
    }
    offender_hit_counts = {
        channel_key: 0
        for channel_key in TASK1_CHANNEL_KEYS
    }
    selected_offender_count_by_row = np.zeros(n_rows, dtype=int)
    resolution_exact_by_row = np.ones(n_rows, dtype=bool)
    removed_value_frames: list[pd.DataFrame] = []

    for channel_a, channel_b, relation_type in _iter_task1_channel_relation_pairs(channel_map):
        relation_limits = relation_limits_by_type.get(relation_type) or relation_limits_by_type.get("any") or {}
        q_sum_left = float(relation_limits.get("q_sum_left", 0.0))
        q_sum_right = float(relation_limits.get("q_sum_right", 0.0))
        q_dif_threshold = float(relation_limits.get("q_dif_threshold", 0.0))
        t_sum_left = float(relation_limits.get("t_sum_left", 0.0))
        t_sum_right = float(relation_limits.get("t_sum_right", 0.0))
        t_dif_threshold = float(relation_limits.get("t_dif_threshold", 0.0))
        cols_a = channel_map[channel_a]
        cols_b = channel_map[channel_b]
        q_a = pd.to_numeric(df_input[cols_a["Q"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_b = pd.to_numeric(df_input[cols_b["Q"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_a = pd.to_numeric(df_input[cols_a["T"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_b = pd.to_numeric(df_input[cols_b["T"]], errors="coerce").fillna(0).to_numpy(dtype=float)

        if relation_type == "self":
            valid_mask = np.isfinite(q_a) & np.isfinite(t_a)
            pair_q_sum = q_a.copy()
            pair_q_dif = np.zeros_like(q_a, dtype=float)
            pair_t_sum = t_a.copy()
            pair_t_dif = np.zeros_like(t_a, dtype=float)
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
            pair_q_sum = 0.5 * (q_a + q_b)
            pair_q_dif = 0.5 * (q_a - q_b)
            pair_t_sum = 0.5 * (t_a + t_b)
            pair_t_dif = 0.5 * (t_a - t_b)
        if not np.any(valid_mask):
            continue

        valid_count = int(np.count_nonzero(valid_mask))
        summary["valid_pair_observations"] += valid_count
        summary["valid_pair_observations_by_relation"][relation_type] += valid_count

        q_metric_valid_mask = valid_mask & np.isfinite(pair_q_sum) & (pair_q_sum != 0)
        t_metric_valid_mask = valid_mask & np.isfinite(pair_t_sum) & (pair_t_sum != 0)
        q_dif_valid_mask = valid_mask & np.isfinite(pair_q_dif)
        t_dif_valid_mask = valid_mask & np.isfinite(pair_t_dif)
        if relation_type != "self":
            q_dif_valid_mask &= (pair_q_sum != 0)
            t_dif_valid_mask &= (pair_t_sum != 0)

        fail_q_sum_low = q_metric_valid_mask & (pair_q_sum < float(q_sum_left))
        fail_q_sum_high = q_metric_valid_mask & (pair_q_sum > float(q_sum_right))
        fail_q_sum = fail_q_sum_low | fail_q_sum_high
        fail_q_dif = q_dif_valid_mask & (np.abs(pair_q_dif) > abs(float(q_dif_threshold)))
        fail_t_sum_low = t_metric_valid_mask & (pair_t_sum < float(t_sum_left))
        fail_t_sum_high = t_metric_valid_mask & (pair_t_sum > float(t_sum_right))
        fail_t_sum = fail_t_sum_low | fail_t_sum_high
        fail_t_dif = t_dif_valid_mask & (np.abs(pair_t_dif) > abs(float(t_dif_threshold)))
        fail_any = fail_q_sum | fail_q_dif | fail_t_sum | fail_t_dif

        summary["failed_pair_q_sum"] += int(np.count_nonzero(fail_q_sum))
        summary["failed_pair_q_sum_low"] += int(np.count_nonzero(fail_q_sum_low))
        summary["failed_pair_q_sum_high"] += int(np.count_nonzero(fail_q_sum_high))
        summary["failed_pair_q_dif"] += int(np.count_nonzero(fail_q_dif))
        summary["failed_pair_t_sum"] += int(np.count_nonzero(fail_t_sum))
        summary["failed_pair_t_sum_low"] += int(np.count_nonzero(fail_t_sum_low))
        summary["failed_pair_t_sum_high"] += int(np.count_nonzero(fail_t_sum_high))
        summary["failed_pair_t_dif"] += int(np.count_nonzero(fail_t_dif))
        summary["failed_pair_any"] += int(np.count_nonzero(fail_any))
        summary["failed_pair_any_by_relation"][relation_type] += int(np.count_nonzero(fail_any))
        summary["failed_pair_q_sum_low_by_relation"][relation_type] += int(np.count_nonzero(fail_q_sum_low))
        summary["failed_pair_q_sum_high_by_relation"][relation_type] += int(np.count_nonzero(fail_q_sum_high))
        summary["failed_pair_t_sum_low_by_relation"][relation_type] += int(np.count_nonzero(fail_t_sum_low))
        summary["failed_pair_t_sum_high_by_relation"][relation_type] += int(np.count_nonzero(fail_t_sum_high))

        if np.any(fail_any):
            q_sum_width = max(abs(float(q_sum_right) - float(q_sum_left)), 1e-9)
            t_sum_width = max(abs(float(t_sum_right) - float(t_sum_left)), 1e-9)
            q_dif_scale = max(abs(float(q_dif_threshold)), 1e-9)
            t_dif_scale = max(abs(float(t_dif_threshold)), 1e-9)
            q_sum_excess = np.maximum.reduce(
                [
                    np.zeros_like(pair_q_sum, dtype=float),
                    float(q_sum_left) - pair_q_sum,
                    pair_q_sum - float(q_sum_right),
                ]
            ) / q_sum_width
            q_dif_excess = np.maximum(np.abs(pair_q_dif) - abs(float(q_dif_threshold)), 0.0) / q_dif_scale
            t_sum_excess = np.maximum.reduce(
                [
                    np.zeros_like(pair_t_sum, dtype=float),
                    float(t_sum_left) - pair_t_sum,
                    pair_t_sum - float(t_sum_right),
                ]
            ) / t_sum_width
            t_dif_excess = np.maximum(np.abs(pair_t_dif) - abs(float(t_dif_threshold)), 0.0) / t_dif_scale
            pair_severity = q_sum_excess + q_dif_excess + t_sum_excess + t_dif_excess
            pair_severity = np.where(fail_any, pair_severity, 0.0)
            pair_severity = np.where(fail_any & (pair_severity <= 0), 1.0, pair_severity)
            for row_pos in np.flatnonzero(fail_any):
                row_pos_int = int(row_pos)
                row_failure_relations.setdefault(row_pos_int, set()).add(relation_type)
                if relation_type == "self":
                    row_forced_channels.setdefault(row_pos_int, set()).add(channel_a)
                else:
                    row_failed_edges.setdefault(row_pos_int, []).append(
                        (channel_a, channel_b, float(pair_severity[row_pos]))
                    )

    flagged_row_positions = sorted(set(row_failed_edges) | set(row_forced_channels))
    summary["flagged_rows"] = len(flagged_row_positions)
    for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES:
        summary["rows_with_relation_failures"][relation_type] = sum(
            1
            for relation_set in row_failure_relations.values()
            if relation_type in relation_set
        )

    for row_pos in flagged_row_positions:
        forced_channels = sorted(
            row_forced_channels.get(row_pos, set()),
            key=_task1_channel_order_key,
        )
        edge_list = row_failed_edges.get(row_pos, [])
        uncovered_edges = [
            (channel_a, channel_b, severity)
            for channel_a, channel_b, severity in edge_list
            if channel_a not in forced_channels and channel_b not in forced_channels
        ]
        selected_channels = forced_channels + select_exact_minimum_vertex_cover(
            uncovered_edges,
            _task1_channel_order_key,
        )
        selected_channel_set = set(selected_channels)
        unresolved_edges = [
            (channel_a, channel_b, severity)
            for channel_a, channel_b, severity in edge_list
            if channel_a not in selected_channel_set and channel_b not in selected_channel_set
        ]
        resolution_exact_by_row[row_pos] = len(unresolved_edges) == 0

        summary["max_failed_pairs_in_row"] = max(
            summary["max_failed_pairs_in_row"],
            len(edge_list) + len(forced_channels),
        )
        summary["max_selected_offenders_in_row"] = max(
            summary["max_selected_offenders_in_row"],
            len(selected_channels),
        )
        summary["selected_offender_channels"] += len(selected_channels)
        if len(selected_channels) > 1:
            summary["rows_with_multiple_offenders"] += 1
        summary["selected_offender_cardinality_counts"][len(selected_channels)] = (
            int(summary["selected_offender_cardinality_counts"].get(len(selected_channels), 0)) + 1
        )
        selected_offender_count_by_row[row_pos] = len(selected_channels)
        for channel_key in selected_channels:
            channel_fail_masks[channel_key][row_pos] = True
            offender_hit_counts[channel_key] += 1

    changed_columns: list[str] = []
    any_row_affected = np.zeros(n_rows, dtype=bool)
    for channel_key, fail_mask in channel_fail_masks.items():
        if not np.any(fail_mask):
            continue
        cols = channel_map[channel_key]
        q_values = pd.to_numeric(df_input[cols["Q"]], errors="coerce").fillna(0)
        t_values = pd.to_numeric(df_input[cols["T"]], errors="coerce").fillna(0)
        summary["values_zeroed"] += int((q_values[fail_mask] != 0).sum())
        summary["values_zeroed"] += int((t_values[fail_mask] != 0).sum())
        any_row_affected |= fail_mask
        changed_columns.extend((cols["Q"], cols["T"]))
        if removed_value_pass_label is not None:
            removed_value_frames.append(
                _task1_build_removed_channel_values_frame(
                    df_input,
                    channel_key,
                    q_values,
                    t_values,
                    fail_mask,
                    removed_value_pass_label,
                )
            )

    summary["rows_affected"] = int(np.count_nonzero(any_row_affected))
    if apply_changes and changed_columns and snapshot_originals is not None:
        snapshot_originals(df_input, list(dict.fromkeys(changed_columns)))

    if apply_changes:
        for channel_key, fail_mask in channel_fail_masks.items():
            if not np.any(fail_mask):
                continue
            cols = channel_map[channel_key]
            df_input.loc[fail_mask, [cols["Q"], cols["T"]]] = 0

    top_offenders = sorted(
        (
            (channel_key, count)
            for channel_key, count in offender_hit_counts.items()
            if count > 0
        ),
        key=lambda item: (-item[1], *_task1_channel_order_key(item[0])),
    )[:8]
    if top_offenders:
        top_offenders_str = ", ".join(
            f"P{plane}{side}{strip}:{count}"
            for (plane, side, strip), count in top_offenders
        )
        print(
            "[channel-combination-offenders] "
            f"flagged_rows={summary['flagged_rows']} "
            f"selected_offenders={summary['selected_offender_channels']} "
            f"multi_offender_rows={summary['rows_with_multiple_offenders']} "
            f"max_failed_pairs_in_row={summary['max_failed_pairs_in_row']} "
            f"top={top_offenders_str}"
        )

    summary["selected_offender_counts"] = {
        channel_key: int(offender_hit_counts.get(channel_key, 0))
        for channel_key in TASK1_CHANNEL_KEYS
    }
    summary["selected_offender_count_by_row"] = pd.Series(
        selected_offender_count_by_row,
        index=df_input.index,
        dtype=int,
    )
    summary["resolution_exact_by_row"] = pd.Series(
        resolution_exact_by_row,
        index=df_input.index,
        dtype=bool,
    )
    summary["rows_with_inexact_resolution"] = int(np.count_nonzero(~resolution_exact_by_row))
    summary["removed_channel_values"] = (
        pd.concat(removed_value_frames, ignore_index=True)
        if removed_value_frames
        else _empty_task1_removed_channel_values_frame()
    )
    return summary


TASK1_CHANNEL_COMBINATION_OBSERVABLES: tuple[tuple[str, str], ...] = (
    ("q_sum", "Q semisum"),
    ("q_dif", "Q semidifference"),
    ("t_sum", "T semisum"),
    ("t_dif", "T semidifference"),
)


def _task1_filter_tt_series(df_input: pd.DataFrame) -> pd.Series:
    for candidate in ("raw_tt", "clean_tt"):
        if candidate in df_input.columns:
            return df_input[candidate].apply(normalize_tt_label).astype(str)
    return pd.Series(["0"] * len(df_input), index=df_input.index, dtype=str)


def _task1_population_color(label: str) -> tuple[float, float, float, float]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    color_pos = int.from_bytes(digest[:4], "big") / float(2**32 - 1)
    return plt.get_cmap("turbo")(0.08 + 0.84 * color_pos)


def _task1_hist_range(
    before_values: np.ndarray,
    after_values: np.ndarray,
    limits: tuple[float | None, float | None],
) -> tuple[float, float]:
    finite_values = np.concatenate(
        [
            before_values[np.isfinite(before_values)],
            after_values[np.isfinite(after_values)],
        ]
    )
    lower_limit, upper_limit = limits
    if finite_values.size:
        low = float(np.nanpercentile(finite_values, 1))
        high = float(np.nanpercentile(finite_values, 99))
        if not np.isfinite(low) or not np.isfinite(high):
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


def _task1_full_data_range(
    values: np.ndarray,
    limits: tuple[float | None, float | None],
) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    lower_limit, upper_limit = limits
    if finite_values.size:
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
    padding = 0.05 * span if span > 0 else max(1.0, 0.05 * max(abs(low), abs(high), 1.0))
    return low - padding, high + padding


def _task1_capped_plot_range(
    values: np.ndarray,
    limits: tuple[float | None, float | None],
    *,
    boundary_expansion_factor: float = 1.5,
) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    lower_limit, upper_limit = limits
    lower_limit = float(lower_limit) if lower_limit is not None and np.isfinite(lower_limit) else None
    upper_limit = float(upper_limit) if upper_limit is not None and np.isfinite(upper_limit) else None

    if finite_values.size:
        data_low = float(np.nanmin(finite_values))
        data_high = float(np.nanmax(finite_values))
    else:
        data_low, data_high = np.nan, np.nan

    if lower_limit is not None and upper_limit is not None and upper_limit > lower_limit:
        center = 0.5 * (lower_limit + upper_limit)
        half_span = 0.5 * (upper_limit - lower_limit)
        cap_low = center - boundary_expansion_factor * half_span
        cap_high = center + boundary_expansion_factor * half_span
    else:
        return _task1_full_data_range(values, limits)

    low = lower_limit
    high = upper_limit
    if np.isfinite(data_low):
        low = min(low, data_low)
    if np.isfinite(data_high):
        high = max(high, data_high)

    low = max(low, cap_low)
    high = min(high, cap_high)

    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low, high = cap_low, cap_high

    span = high - low
    padding = 0.04 * span if span > 0 else max(0.5, 0.04 * max(abs(low), abs(high), 1.0))
    return low - padding, high + padding


def collect_task1_channel_combination_histogram_payload(
    df_input: pd.DataFrame,
    tt_series: pd.Series,
) -> pd.DataFrame:
    payload_rows: list[pd.DataFrame] = []
    channel_map = collect_task1_channel_qt_map(df_input)
    if not channel_map:
        return pd.DataFrame(
            columns=[
                "row_index",
                "tt",
                "combo",
                "relation_type",
                "same_strip",
                "q_sum",
                "q_dif",
                "t_sum",
                "t_dif",
            ]
        )

    tt_series = tt_series.reindex(df_input.index).fillna("0").astype(str)
    for channel_a, channel_b, relation_type in _iter_task1_channel_relation_pairs(channel_map):
        cols_a = channel_map[channel_a]
        cols_b = channel_map[channel_b]
        q_a = pd.to_numeric(df_input[cols_a["Q"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_b = pd.to_numeric(df_input[cols_b["Q"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_a = pd.to_numeric(df_input[cols_a["T"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_b = pd.to_numeric(df_input[cols_b["T"]], errors="coerce").fillna(0).to_numpy(dtype=float)

        if relation_type == "self":
            valid_mask = (
                np.isfinite(q_a)
                & np.isfinite(t_a)
                & (q_a != 0)
                & (t_a != 0)
            )
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
                    "row_index": valid_index.to_numpy(),
                    "tt": tt_series.loc[valid_index].to_numpy(dtype=str),
                    "combo": f"P{channel_a[0]}S{channel_a[2]}{channel_a[1]}-P{channel_b[0]}S{channel_b[2]}{channel_b[1]}",
                    "relation_type": relation_type,
                    "same_strip": relation_type == "same_strip",
                    "q_sum": q_sum[valid_mask],
                    "q_dif": q_dif[valid_mask],
                    "t_sum": t_sum[valid_mask],
                    "t_dif": t_dif[valid_mask],
                }
            )
        )

    if not payload_rows:
        return pd.DataFrame(
            columns=[
                "row_index",
                "tt",
                "combo",
                "relation_type",
                "same_strip",
                "q_sum",
                "q_dif",
                "t_sum",
                "t_dif",
            ]
        )
    return pd.concat(payload_rows, ignore_index=True)


def subtract_task1_channel_combination_payload(
    left_payload: pd.DataFrame,
    right_payload: pd.DataFrame,
) -> pd.DataFrame:
    if left_payload.empty:
        return left_payload.copy()
    if right_payload.empty or not {"row_index", "combo"}.issubset(left_payload.columns) or not {"row_index", "combo"}.issubset(right_payload.columns):
        return left_payload.copy()

    reduced_right = right_payload.loc[:, ["row_index", "combo"]].drop_duplicates()
    removed_payload = left_payload.merge(
        reduced_right,
        on=["row_index", "combo"],
        how="left",
        indicator=True,
    )
    return removed_payload.loc[removed_payload["_merge"] == "left_only"].drop(columns="_merge")


def plot_task1_channel_combination_filter_by_tt(
    original_payload: pd.DataFrame,
    before_payload: pd.DataFrame,
    after_payload: pd.DataFrame,
    basename_no_ext_value: str,
    fig_idx_value: int,
    *,
    show_plots: bool,
    save_plots: bool,
    plot_list: list[str] | None,
    limits_by_relation: dict[str, dict[str, tuple[float | None, float | None]]],
    q_left_limits_by_channel: dict[tuple[int, str, int], float] | None,
    top_n_by_relation: dict[str, int | None],
    include_same_strip_pairs: bool,
) -> int:
    observable_names = [observable for observable, _ in TASK1_CHANNEL_COMBINATION_OBSERVABLES]
    observable_labels = {observable: label for observable, label in TASK1_CHANNEL_COMBINATION_OBSERVABLES}
    scatter_max_points = 5000
    removed_scatter_marker_size = max(8, int(round(0.6 * removed_marker_size)))
    prefilter_scatter_size = max(3, int(round(0.25 * removed_marker_size)))
    noise_limit_line_kwargs = {
        "color": "red",
        "linestyle": "--",
        "linewidth": 1.4,
        "alpha": 0.85,
        "zorder": 6,
    }
    noise_limit_boundary_kwargs = {
        "color": "red",
        "linestyle": "--",
        "linewidth": 1.2,
        "alpha": 0.65,
        "zorder": 6,
    }
    rng = np.random.default_rng(0)
    q_left_limits_by_channel = q_left_limits_by_channel or {}

    original_payload = original_payload.copy()
    before_payload = before_payload.copy()
    after_payload = after_payload.copy()
    tt_labels = []
    for payload in (original_payload, before_payload, after_payload):
        if "tt" in payload.columns:
            tt_labels.extend(payload["tt"].astype(str).tolist())
    ordered_tts = [tt_label for tt_label in TT_COLOR_LABELS if tt_label != "0" and tt_label in set(tt_labels)]
    relation_label_map = {
        "self": "self",
        "same_strip": "same strip",
        "same_plane": "same plane",
        "any": "any",
    }

    for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES:
        relation_limits = limits_by_relation.get(relation_type, {})
        relation_top_n = top_n_by_relation.get(relation_type)
        for tt_label in ordered_tts:
            original_tt_all = original_payload.loc[
                (original_payload.get("tt", pd.Series(dtype=str)).astype(str) == tt_label)
                & (original_payload.get("relation_type", pd.Series(dtype=str)).astype(str) == relation_type)
            ].copy()
            before_tt_all = before_payload.loc[
                (before_payload.get("tt", pd.Series(dtype=str)).astype(str) == tt_label)
                & (before_payload.get("relation_type", pd.Series(dtype=str)).astype(str) == relation_type)
            ].copy()
            after_tt_all = after_payload.loc[
                (after_payload.get("tt", pd.Series(dtype=str)).astype(str) == tt_label)
                & (after_payload.get("relation_type", pd.Series(dtype=str)).astype(str) == relation_type)
            ].copy()
            if original_tt_all.empty and before_tt_all.empty and after_tt_all.empty:
                continue

            reference_counts = before_tt_all.groupby("combo").size().sort_values(ascending=False)
            if reference_counts.empty:
                reference_counts = original_tt_all.groupby("combo").size().sort_values(ascending=False)
            if reference_counts.empty:
                reference_counts = after_tt_all.groupby("combo").size().sort_values(ascending=False)
            combo_labels = (
                reference_counts.index.tolist()
                if relation_top_n is None
                else reference_counts.head(int(relation_top_n)).index.tolist()
            )
            if not combo_labels:
                continue

            original_tt_full = original_tt_all.loc[original_tt_all["combo"].isin(combo_labels)].copy()
            before_tt_full = before_tt_all.loc[before_tt_all["combo"].isin(combo_labels)].copy()
            after_tt_full = after_tt_all.loc[after_tt_all["combo"].isin(combo_labels)].copy()
            if original_tt_full.empty and before_tt_full.empty and after_tt_full.empty:
                continue

            prefilter_removed_tt_full = subtract_task1_channel_combination_payload(
                original_tt_full,
                before_tt_full,
            )
            removed_tt_full = subtract_task1_channel_combination_payload(
                before_tt_full,
                after_tt_full,
            )

            def _sample_scatter_payload(payload: pd.DataFrame) -> pd.DataFrame:
                if len(payload) <= scatter_max_points:
                    return payload.copy()
                return payload.iloc[
                    rng.choice(len(payload), size=scatter_max_points, replace=False)
                ].copy()

            after_tt = _sample_scatter_payload(after_tt_full)
            prefilter_removed_tt = _sample_scatter_payload(prefilter_removed_tt_full)
            removed_tt = _sample_scatter_payload(removed_tt_full)

            combo_color_map = {label: _task1_population_color(label) for label in combo_labels}
            combo_q_boundaries: dict[str, dict[str, object]] = {}
            for combo_label in combo_labels:
                parsed_combo = _task1_parse_combo_label(combo_label)
                if parsed_combo is None:
                    continue
                channel_a, channel_b = parsed_combo
                q_left_a = q_left_limits_by_channel.get(channel_a)
                q_left_b = q_left_limits_by_channel.get(channel_b)
                combo_q_boundaries[combo_label] = {
                    "channel_a": channel_a,
                    "channel_b": channel_b,
                    "q_left_a": q_left_a,
                    "q_left_b": q_left_b,
                    "q_sum_left": (
                        0.5 * (float(q_left_a) + float(q_left_b))
                        if q_left_a is not None and q_left_b is not None
                        else None
                    ),
                }
            after_combo_data = {
                label: {
                    observable: pd.to_numeric(
                        after_tt_full.loc[after_tt_full["combo"] == label, observable],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    for observable in observable_names
                }
                for label in combo_labels
            }
            removed_combo_data = {
                label: {
                    observable: pd.to_numeric(
                        removed_tt_full.loc[removed_tt_full["combo"] == label, observable],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    for observable in observable_names
                }
                for label in combo_labels
            }
            after_scatter_combo_data = {
                label: {
                    observable: pd.to_numeric(
                        after_tt.loc[after_tt["combo"] == label, observable],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    for observable in observable_names
                }
                for label in combo_labels
            }
            removed_scatter_combo_data = {
                label: {
                    observable: pd.to_numeric(
                        removed_tt.loc[removed_tt["combo"] == label, observable],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    for observable in observable_names
                }
                for label in combo_labels
            }
            n_obs = len(observable_names)
            fig, axes = plt.subplots(n_obs, n_obs, figsize=(2.3 * n_obs, 2.3 * n_obs), constrained_layout=True)
            any_panel_data = False

            axis_ranges: dict[str, tuple[float, float]] = {}
            for observable in observable_names:
                full_values = pd.to_numeric(
                    original_tt_full.get(observable, pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
                axis_ranges[observable] = _task1_capped_plot_range(
                    full_values,
                    relation_limits.get(observable, (None, None)),
                )

            for row_idx, y_name in enumerate(observable_names):
                for col_idx, x_name in enumerate(observable_names):
                    ax = axes[row_idx, col_idx]
                    if col_idx > row_idx:
                        ax.set_axis_off()
                        continue

                    if row_idx == col_idx:
                        before_values = pd.to_numeric(
                            before_tt_full.get(x_name, pd.Series(dtype=float)),
                            errors="coerce",
                        ).dropna().to_numpy(dtype=float)
                        after_values = pd.to_numeric(
                            after_tt_full.get(x_name, pd.Series(dtype=float)),
                            errors="coerce",
                        ).dropna().to_numpy(dtype=float)
                        prefilter_removed_values = pd.to_numeric(
                            prefilter_removed_tt_full.get(x_name, pd.Series(dtype=float)),
                            errors="coerce",
                        ).dropna().to_numpy(dtype=float)
                        prefilter_removed_values = prefilter_removed_values[np.isfinite(prefilter_removed_values)]
                        if (
                            before_values.size == 0
                            and after_values.size == 0
                            and prefilter_removed_values.size == 0
                        ):
                            ax.set_axis_off()
                            continue
                        any_panel_data = True
                        x_low, x_high = axis_ranges[x_name]
                        bins = np.linspace(x_low, x_high, 60)
                        if prefilter_removed_values.size:
                            ax.hist(
                                prefilter_removed_values,
                                bins=bins,
                                histtype="step",
                                linestyle="--",
                                linewidth=1.0,
                                alpha=0.55,
                                color="lightgrey",
                            )
                        for combo_label in combo_labels:
                            combo_before = removed_combo_data[combo_label][x_name]
                            combo_after = after_combo_data[combo_label][x_name]
                            combo_before = combo_before[np.isfinite(combo_before)]
                            combo_after = combo_after[np.isfinite(combo_after)]
                            if combo_before.size:
                                ax.hist(
                                    combo_before,
                                    bins=bins,
                                    histtype="step",
                                    linestyle="--",
                                    linewidth=0.95,
                                    alpha=0.55,
                                    color=combo_color_map[combo_label],
                                )
                            if combo_after.size:
                                ax.hist(
                                    combo_after,
                                    bins=bins,
                                    histtype="step",
                                    linestyle="-",
                                    linewidth=1.35,
                                    alpha=0.95,
                                    color=combo_color_map[combo_label],
                                )
                        lower_limit, upper_limit = relation_limits.get(x_name, (None, None))
                        if lower_limit is not None:
                            ax.axvline(float(lower_limit), color="lightgrey", linestyle="--", linewidth=1.1)
                        if upper_limit is not None:
                            ax.axvline(float(upper_limit), color="lightgrey", linestyle="--", linewidth=1.1)
                        if x_name == "q_sum":
                            for combo_label in combo_labels:
                                q_sum_noise_left = combo_q_boundaries.get(combo_label, {}).get("q_sum_left")
                                if q_sum_noise_left is None or not np.isfinite(q_sum_noise_left):
                                    continue
                                ax.axvline(
                                    float(q_sum_noise_left),
                                    **noise_limit_line_kwargs,
                                )
                        ax.set_xlim(x_low, x_high)
                        ax.set_yscale("log", nonpositive="clip")
                        ax.set_title(observable_labels[x_name], fontsize=9)
                    else:
                        after_x = pd.to_numeric(
                            after_tt.get(x_name, pd.Series(dtype=float)),
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        after_y = pd.to_numeric(
                            after_tt.get(y_name, pd.Series(dtype=float)),
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        prefilter_removed_x = pd.to_numeric(
                            prefilter_removed_tt.get(x_name, pd.Series(dtype=float)),
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        prefilter_removed_y = pd.to_numeric(
                            prefilter_removed_tt.get(y_name, pd.Series(dtype=float)),
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        removed_x = pd.to_numeric(
                            removed_tt.get(x_name, pd.Series(dtype=float)),
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        removed_y = pd.to_numeric(
                            removed_tt.get(y_name, pd.Series(dtype=float)),
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        after_mask = np.isfinite(after_x) & np.isfinite(after_y)
                        prefilter_removed_mask = np.isfinite(prefilter_removed_x) & np.isfinite(prefilter_removed_y)
                        removed_mask = np.isfinite(removed_x) & np.isfinite(removed_y)
                        if not np.any(prefilter_removed_mask) and not np.any(removed_mask) and not np.any(after_mask):
                            ax.set_axis_off()
                            continue
                        any_panel_data = True
                        if np.any(prefilter_removed_mask):
                            ax.scatter(
                                prefilter_removed_x[prefilter_removed_mask],
                                prefilter_removed_y[prefilter_removed_mask],
                                s=prefilter_scatter_size,
                                alpha=0.08,
                                color="grey",
                                edgecolors="none",
                                rasterized=True,
                            )
                        for combo_label in combo_labels:
                            combo_after_x = after_scatter_combo_data[combo_label][x_name]
                            combo_after_y = after_scatter_combo_data[combo_label][y_name]
                            combo_removed_x = removed_scatter_combo_data[combo_label][x_name]
                            combo_removed_y = removed_scatter_combo_data[combo_label][y_name]
                            combo_after_mask = np.isfinite(combo_after_x) & np.isfinite(combo_after_y)
                            combo_removed_mask = np.isfinite(combo_removed_x) & np.isfinite(combo_removed_y)
                            if np.any(combo_after_mask):
                                ax.scatter(
                                    combo_after_x[combo_after_mask],
                                    combo_after_y[combo_after_mask],
                                    s=6,
                                    alpha=0.16,
                                    color=combo_color_map[combo_label],
                                    edgecolors="none",
                                    rasterized=True,
                                )
                            if np.any(combo_removed_mask):
                                ax.scatter(
                                    combo_removed_x[combo_removed_mask],
                                    combo_removed_y[combo_removed_mask],
                                    s=removed_scatter_marker_size,
                                    marker=removed_marker,
                                    alpha=removed_marker_alpha,
                                    linewidths=1.0,
                                    color=combo_color_map[combo_label],
                                    rasterized=True,
                                )
                        x_low, x_high = axis_ranges[x_name]
                        y_low, y_high = axis_ranges[y_name]
                        ax.set_xlim(x_low, x_high)
                        ax.set_ylim(y_low, y_high)
                        x_limits = relation_limits.get(x_name, (None, None))
                        y_limits = relation_limits.get(y_name, (None, None))
                        if x_limits[0] is not None:
                            ax.axvline(float(x_limits[0]), color="lightgrey", linestyle="--", linewidth=0.9)
                        if x_limits[1] is not None:
                            ax.axvline(float(x_limits[1]), color="lightgrey", linestyle="--", linewidth=0.9)
                        if y_limits[0] is not None:
                            ax.axhline(float(y_limits[0]), color="lightgrey", linestyle="--", linewidth=0.9)
                        if y_limits[1] is not None:
                            ax.axhline(float(y_limits[1]), color="lightgrey", linestyle="--", linewidth=0.9)
                        if x_name == "q_sum":
                            for combo_label in combo_labels:
                                q_sum_noise_left = combo_q_boundaries.get(combo_label, {}).get("q_sum_left")
                                if q_sum_noise_left is None or not np.isfinite(q_sum_noise_left):
                                    continue
                                ax.axvline(
                                    float(q_sum_noise_left),
                                    **noise_limit_line_kwargs,
                                )
                        if y_name == "q_sum":
                            for combo_label in combo_labels:
                                q_sum_noise_left = combo_q_boundaries.get(combo_label, {}).get("q_sum_left")
                                if q_sum_noise_left is None or not np.isfinite(q_sum_noise_left):
                                    continue
                                ax.axhline(
                                    float(q_sum_noise_left),
                                    **noise_limit_line_kwargs,
                                )
                        if x_name == "q_sum" and y_name == "q_dif":
                            q_sum_values = np.linspace(x_low, x_high, 200)
                            for combo_label in combo_labels:
                                combo_boundary = combo_q_boundaries.get(combo_label, {})
                                q_left_a = combo_boundary.get("q_left_a")
                                q_left_b = combo_boundary.get("q_left_b")
                                if q_left_a is not None and np.isfinite(q_left_a):
                                    ax.plot(
                                        q_sum_values,
                                        float(q_left_a) - q_sum_values,
                                        **noise_limit_boundary_kwargs,
                                    )
                                if q_left_b is not None and np.isfinite(q_left_b):
                                    ax.plot(
                                        q_sum_values,
                                        q_sum_values - float(q_left_b),
                                        **noise_limit_boundary_kwargs,
                                    )

                    if row_idx == n_obs - 1:
                        ax.set_xlabel(observable_labels[x_name], fontsize=8)
                    else:
                        ax.set_xticklabels([])
                    if col_idx == 0 and row_idx != col_idx:
                        ax.set_ylabel(observable_labels[y_name], fontsize=8)
                    elif col_idx != 0:
                        ax.set_yticklabels([])
                    if row_idx == col_idx:
                        ax.set_ylabel("Counts", fontsize=8)
                    ax.tick_params(labelsize=6)

            if not any_panel_data:
                plt.close(fig)
                continue

            has_noise_limit_boundaries = any(
                combo_q_boundaries.get(label, {}).get("q_sum_left") is not None
                or combo_q_boundaries.get(label, {}).get("q_left_a") is not None
                or combo_q_boundaries.get(label, {}).get("q_left_b") is not None
                for label in combo_labels
            )
            style_handles = [
                mpl.lines.Line2D([0], [0], color="lightgrey", linestyle="--", linewidth=1.0, label="Removed Before Combination"),
                mpl.lines.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="Removed By Combination"),
                mpl.lines.Line2D([0], [0], color="black", linestyle="-", linewidth=1.4, label="Retained"),
                mpl.lines.Line2D(
                    [0],
                    [0],
                    color="black",
                    marker=removed_marker,
                    linestyle="None",
                    markersize=6,
                    markerfacecolor="none",
                    label="Combination Rejection",
                ),
            ]
            if has_noise_limit_boundaries:
                style_handles.append(
                    mpl.lines.Line2D(
                        [0],
                        [0],
                        color="red",
                        linestyle="--",
                        linewidth=1.4,
                        label="Noise Charge Limit",
                    )
                )
            combo_handles = [
                mpl.lines.Line2D([0], [0], color=combo_color_map[label], linestyle="-", linewidth=1.6, label=label)
                for label in combo_labels
            ]
            fig.legend(
                handles=style_handles + combo_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=min(6, max(2, len(combo_handles) + 2)),
                fontsize=6,
                frameon=False,
                handlelength=1.8,
                columnspacing=0.8,
            )
            combo_mode_label = "all pairs" if relation_top_n is None else f"top {int(relation_top_n)} pairs"
            fig.suptitle(
                f"Task 1 channel-combination filter by TT {tt_label}\n"
                f"{basename_no_ext_value} · {relation_label_map.get(relation_type, relation_type)} · {combo_mode_label}",
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
                save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
            if show_plots:
                plt.show()
            plt.close(fig)

    return fig_idx_value


def build_task1_channel_pattern_series(df: pd.DataFrame) -> pd.Series:
    """Encode per-event front/back channel occupancy as a deterministic 32-bit string."""
    pattern_arrays: list[np.ndarray] = []
    for plane, strip, side in TASK1_CHANNEL_PATTERN_ORDER:
        col_name = f"Q{plane}_{side}_{strip}"
        if col_name in df.columns:
            values = df[col_name].fillna(0).to_numpy(copy=False)
            bits = np.where(values != 0, "1", "0")
        else:
            print(f"Warning: missing channel column '{col_name}' while building TASK_1 patterns.")
            bits = np.full(len(df), "0", dtype="<U1")
        pattern_arrays.append(bits)

    if not pattern_arrays:
        return pd.Series(dtype="object", index=df.index)

    full_pattern = pattern_arrays[0].copy()
    for bits in pattern_arrays[1:]:
        full_pattern = np.char.add(full_pattern, bits)
    return pd.Series(full_pattern, index=df.index, dtype="object")


def store_pattern_rates(metadata: dict[str, object], patterns: pd.Series, prefix: str, df: pd.DataFrame) -> None:
    phase_meta = build_events_per_second_metadata(df)
    try:
        denominator = float(phase_meta.get("events_per_second_total_seconds", 0) or 0)
    except (TypeError, ValueError):
        denominator = 0.0

    counts = patterns.value_counts()
    for pattern, count in counts.items():
        rate_hz = round(float(count) / denominator, 6) if denominator > 0 else 0.0
        metadata[f"{prefix}_{pattern}_rate_hz"] = rate_hz

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

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Function definition ---------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def calibrate_strip_T_diff(T_F, T_B, self_trigger_mode = False):
    
    if self_trigger_mode:
        T_left_side = T_F_left_pre_cal_ST
        T_right_side = T_F_right_pre_cal_ST
    else:
        T_left_side = T_F_left_pre_cal
        T_right_side = T_F_right_pre_cal
    
    cond = (T_F != 0) & (T_F > T_left_side) & (T_F < T_right_side) & (T_B != 0) & (T_B > T_left_side) & (T_B < T_right_side)
    
    # Front
    T_F = T_F[cond]
    counts, bin_edges = np.histogram(T_F, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / 10**1.5
    indices_above_threshold = np.where(counts > threshold)[0]
    if indices_above_threshold.size > 0:
        min_bin_edge_F = bin_edges[indices_above_threshold[0]]
        max_bin_edge_F = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
        # print(f"Minimum bin edge: {min_bin_edge_F}")
        # print(f"Maximum bin edge: {max_bin_edge_F}")
    else:
        print("No bins have counts above the threshold, Front.")
        threshold = (min_counts + max_counts) / 2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge_F = bin_edges[indices_above_threshold[0]]
        max_bin_edge_F = bin_edges[indices_above_threshold[-1] + 1]
    
    # Back
    T_B = T_B[cond]
    counts, bin_edges = np.histogram(T_B, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / 10**1.5
    indices_above_threshold = np.where(counts > threshold)[0]
    if indices_above_threshold.size > 0:
        min_bin_edge_B = bin_edges[indices_above_threshold[0]]
        max_bin_edge_B = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
        # print(f"Minimum bin edge: {min_bin_edge_B}")
        # print(f"Maximum bin edge: {max_bin_edge_B}")
    else:
        print("No bins have counts above the threshold, Back.")
        threshold = (min_counts + max_counts) / 2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge_B = bin_edges[indices_above_threshold[0]]
        max_bin_edge_B = bin_edges[indices_above_threshold[-1] + 1]
    
    cond = (T_F > min_bin_edge_F) & (T_F < max_bin_edge_F) & (T_B > min_bin_edge_B) & (T_B < max_bin_edge_B)
            
    T_F = T_F[cond]
    T_B = T_B[cond]
    
    # T_diff = ( T_F - T_B ) / 2
    T_diff = ( T_B - T_F ) / 2
    
    # ------------------------------------------------------------------------------
    
    
    T_rel_th = calibrate_strip_T_dif_T_rel_th
    abs_th = calibrate_strip_T_dif_T_abs_th

    # Apply mask to filter values within the threshold
    mask = (np.abs(T_diff) < T_dif_pre_cal_threshold)
    T_diff = T_diff[mask]
    
    # Remove zero values
    T_diff = T_diff[T_diff != 0]
    
    if T_diff.size == 0:
        return np.nan

    # Calculate histogram
    counts, bin_edges = np.histogram(T_diff, bins='auto')
    
    # Calculate the nunber of counts of the bin that has the most counts
    max_counts = np.max(counts)
    
    # Find bins with at least one count
    th = T_rel_th * max_counts
    if th < abs_th:
        th = abs_th
    non_empty_bins = counts >= th

    # Find the longest contiguous subset of non-empty bins
    max_length = 0
    current_length = 0
    start_index = 0
    temp_start = 0

    for i, is_non_empty in enumerate(non_empty_bins):
        if is_non_empty:
            if current_length == 0:
                temp_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_index = temp_start
                end_index = i
        else:
            current_length = 0

    if max_length == 0:
        return np.nan

    # Reject edge-only/single-bin plateaus: they are unstable calibration anchors.
    if max_length < 2 or start_index == 0 or end_index == (len(non_empty_bins) - 1):
        return np.nan
    
    plateau_left = bin_edges[start_index]
    plateau_right = bin_edges[end_index + 1]
    
    # Calculate the offset using the mean of the filtered values
    offset = ( plateau_left + plateau_right ) / 2
    
    return offset

def calibrate_strip_Q_pedestal(Q_ch, T_ch, Q_other, self_trigger_mode = False):
    
    translate_charge_cal = calibrate_strip_Q_pedestal_translate_charge_cal
    percentile = calibrate_strip_Q_pedestal_percentile
    
    rel_th = calibrate_strip_Q_pedestal_rel_th
    rel_th_cal = calibrate_strip_Q_pedestal_rel_th_cal
    abs_th = calibrate_strip_Q_pedestal_abs_th
    q_quantile = calibrate_strip_Q_pedestal_q_quantile # percentile
    
    # First let's tale good values of Time, we want to avoid outliers that might confuse the charge pedestal calibration
    
    if self_trigger_mode:
        T_left_side = T_F_left_pre_cal_ST
        T_right_side = T_F_right_pre_cal_ST
    else:
        T_left_side = T_F_left_pre_cal
        T_right_side = T_F_right_pre_cal
        
    cond = (T_ch != 0) & (T_ch > T_left_side) & (T_ch < T_right_side)
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]
    Q_other = Q_other[cond]
    
    # Condition based on the charge difference: it cannot be too high
    Q_dif = Q_ch - Q_other
    
    cond = ( Q_dif > np.percentile(Q_dif, percentile) ) & ( Q_dif < np.percentile(Q_dif, 100 - percentile ) )
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]
    
    counts, bin_edges = np.histogram(T_ch, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / calibrate_strip_Q_pedestal_thr_factor
    
    indices_above_threshold = np.where(counts > threshold)[0]

    if indices_above_threshold.size > 0:
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
    else:
        print("No bins have counts above the threshold; Q pedestal calibration.")
        threshold = (min_counts + max_counts) / calibrate_strip_Q_pedestal_thr_factor_2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]
    
    Q_ch = Q_ch[(T_ch > min_bin_edge) & (T_ch < max_bin_edge)]
    
    # First take the values that are not zero
    Q_ch = Q_ch[Q_ch != 0]
    
    # Remove the values that are not in (50,500)
    Q_ch = Q_ch[(Q_ch > Q_left_side) & (Q_ch < Q_right_side)]
    
    # Quantile filtering
    Q_ch = Q_ch[Q_ch > np.percentile(Q_ch, q_quantile)]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(Q_ch, bins='auto')
    
    # Calculate the nunber of counts of the bin that has the most counts
    max_counts = np.max(counts)
    counts = counts[counts < max_counts]
    max_counts = np.max(counts)
    
    # Find bins with at least one count
    th = rel_th * max_counts
    if th < abs_th:
        th = abs_th
    non_empty_bins = counts >= th

    # Find the longest contiguous subset of non-empty bins
    max_length = 0
    current_length = 0
    start_index = 0
    temp_start = 0

    for i, is_non_empty in enumerate(non_empty_bins):
        if is_non_empty:
            if current_length == 0:
                temp_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_index = temp_start
        else:
            current_length = 0

    # Get the first bin edge of the longest subset
    offset = bin_edges[start_index]
    
    # Second part --------------------------------------------------------------
    Q_ch_cal = Q_ch - offset
    
    # Remove values outside the range (-2, 2)
    Q_ch_cal = Q_ch_cal[(Q_ch_cal > pedestal_left) & (Q_ch_cal < pedestal_right)]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(Q_ch_cal, bins='auto')
    
    # Find the bin with the most counts
    max_counts = np.max(counts)
    max_bin_index = np.argmax(counts)
    
    # Calculate the threshold
    threshold = rel_th_cal * max_counts
    
    # Start from the bin with the most counts and move left
    offset_bin_index = max_bin_index
    while offset_bin_index > 0 and counts[offset_bin_index] >= threshold:
        offset_bin_index -= 1
    
    # Determine the X value (left edge) of the bin where the threshold is crossed
    offset_cal = bin_edges[offset_bin_index]
    
    pedestal = offset
    
    if translate_charge_cal:
        pedestal = pedestal - translate_charge_cal
        
    return pedestal

enumerate = builtins.enumerate
def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def scatter_2d_and_fit_new(xdat, ydat, title, x_label, y_label, name_of_file):
    global fig_idx
    
    ydat_translated = ydat

    xdat_plot = xdat[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
    ydat_plot = ydat_translated[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
    xdat_pre_fit = xdat[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_dif_charges_up_fit) & (ydat_translated > distance_dif_charges_low_fit)]
    ydat_pre_fit = ydat_translated[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_dif_charges_up_fit) & (ydat_translated > distance_dif_charges_low_fit)]
    
    # Fit a polynomial of specified degree using curve_fit
    initial_guess = [1] * (degree_of_polynomial + 1)
    coeffs, _ = curve_fit(polynomial, xdat_pre_fit, ydat_pre_fit, p0=initial_guess)
    y_pre_fit = polynomial(xdat_pre_fit, *coeffs)
    
    # Filter data for fitting based on residues
    threshold = front_back_fit_threshold  # Set your desired threshold here
    residues = np.abs(ydat_pre_fit - y_pre_fit)  # Calculate residues
    xdat_fit = xdat_pre_fit[residues < threshold]
    ydat_fit = ydat_pre_fit[residues < threshold]
    
    # Perform fit on filtered data
    coeffs, _ = curve_fit(polynomial, xdat_fit, ydat_fit, p0=initial_guess)
    
    y_mean = np.mean(ydat_fit)
    y_check = polynomial(xdat_fit, *coeffs)
    ss_res = np.sum((ydat_fit - y_check)**2)
    ss_tot = np.sum((ydat_fit - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot)
    if r_squared < 0.5:
        print(f"---> R**2 in {name_of_file[0:4]}: {r_squared:.2g}")
    
    if create_plots:
        x_fit = np.linspace(min(xdat_fit), max(xdat_fit), 100)
        y_fit = polynomial(x_fit, *coeffs)
        x_final = xdat_plot
        y_final = ydat_plot - polynomial(xdat_plot, *coeffs)
        plt.close()
        
        if article_format:
            ww = (10.84, 4) # (16,6) was very nice
        else:
            ww = (13.33, 5)
            
        plt.figure(figsize=ww)  # Use plt.subplots() to create figure and axis    
        plt.scatter(xdat_plot, ydat_plot, s=1, label="Original data points")
        # plt.scatter(xdat_pre_fit, ydat_pre_fit, s=1, color="magenta", label="Points for prefitting")
        plt.scatter(xdat_fit, ydat_fit, s=1, color="orange", label="Points for fitting")
        plt.scatter(x_final, y_final, s=1, color="green", label="Calibrated points")
        plt.plot(x_fit, y_fit, 'r-', label='Polynomial Fit: ' + ' '.join([f'a{i}={coeff:.2g}' for i, coeff in enumerate(coeffs[::-1])]))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([scatter_2d_and_fit_new_xlim_left, scatter_2d_and_fit_new_xlim_right])
        plt.ylim([scatter_2d_and_fit_new_ylim_bottom, scatter_2d_and_fit_new_ylim_top])
        plt.grid()
        plt.legend(markerscale=5)  # Increase marker scale by 5 times
        plt.tight_layout()
        if save_plots:
            name_of_file = 'charge_dif_vs_charge_sum_cal'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()
    return coeffs

def interpolate_fast_charge(width):
        """
        Interpolates the Fast Charge for given Width values using cubic spline interpolation.
        Parameters:
        - width (float or np.ndarray): The Width value(s) to interpolate in ns.
        Returns:
        - float or np.ndarray: The interpolated Fast Charge value(s) in fC.
        """
        width = np.asarray(width)  # Ensure input is a NumPy array
        # Keep zero values unchanged
        result = np.where(width == 0, 0, cs(width))
        return result
    

def summary(vector):
    global coincidence_window_cal_ns
    quantile_left = CRT_gaussian_fit_quantile * 100
    quantile_right = 100 - CRT_gaussian_fit_quantile * 100
    
    vector = np.array(vector)  # Convert list to NumPy array
    cond = (vector > -coincidence_window_cal_ns) & (vector < coincidence_window_cal_ns)  # This should result in a boolean array
    vector = vector[cond]
    
    if len(vector) < 100:
        return np.nan
    try:
        percentile_left = np.percentile(vector, quantile_left)
        percentile_right = np.percentile(vector, quantile_right)
    except IndexError:
        print("Gave issue with:")
        print(vector)
        return np.nan
    vector = [x for x in vector if percentile_left <= x <= percentile_right]
    if len(vector) == 0:
        return np.nan
    mu, std = norm.fit(vector)
    return mu

def hist_1d(vdat, bin_number, title, axis_label, name_of_file):
    global fig_idx, coincidence_window_cal_ns
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    vdat = np.array(vdat)  # Convert list to NumPy array
    cond = (vdat > -coincidence_window_cal_ns) & (vdat < coincidence_window_cal_ns)  # This should result in a boolean array
    vdat = vdat[cond]
    counts, bins, _ = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red", label=f"All hits, {len(vdat)} events", density=False)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    h1_q = CRT_gaussian_fit_quantile
    lower_bound = np.quantile(vdat, h1_q)
    upper_bound = np.quantile(vdat, 1 - h1_q)
    cond = (vdat > lower_bound) & (vdat < upper_bound)  # This should result in a boolean array
    vdat = vdat[cond]
    mu, std = norm.fit(vdat)
    p = norm.pdf(bin_centers, mu, std) * len(vdat) * (bins[1] - bins[0])  # Scale to match histogram
    label_plot = f'Gaussian fit:\n    $\\mu={mu:.2g}$,\n    $\\sigma={std:.2g}$\n    CRT$={std/np.sqrt(2)*1000:.3g}$ ps'
    ax.plot(bin_centers, p, 'k', linewidth=2, label=label_plot)
    ax.legend()
    ax.set_title(title)
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    plt.tight_layout()
    if save_plots:
        name_of_file = 'timing'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots: plt.show()
    plt.close()

def plot_histograms_and_gaussian(df, columns, title, figure_number, quantile=0.99, fit_gaussian=False):
    global fig_idx
    nrows, ncols = (2, 3) if figure_number == 1 else (3, 4)
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), constrained_layout=True)
    axs = axs.flatten()
    def gaussian(x, mu, sigma, amplitude):
        return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Precompute quantiles for faster filtering
    if fit_gaussian:
        quantile_bounds = {}
        for col in columns:
            data = df[col].values
            data = data[data != 0]
            if len(data) > 0:
                quantile_bounds[col] = np.quantile(data, [(1 - quantile), quantile])

    # Plot histograms and fit Gaussian if needed
    for i, col in enumerate(columns):
        
        data = df[col].values
        data = data[data != 0]  # Filter out zero values

        if len(data) == 0:  # Skip if no data
            axs[i].text(0.5, 0.5, "No data", transform=axs[i].transAxes, ha='center', va='center', color='gray')
            continue

        # Example color map per column type
        color_map = {
            "theta": "blue",
            "phi": "green",
            "x": "darkorange",
            "y": "darkorange",
            "det_y": "darkorange",
            "s": "purple",
            "det_s": "purple",
            "th_chi": "red",
            "res_ystr": "teal",
            "res_tsum": "brown",
            "res_tdif": "purple",
            "t0": "black"
        }

        # Set default in case no match is found
        selected_col = 'gray'

        if "theta" in col:
            left, right = theta_left_filter, theta_right_filter
            selected_col = color_map["theta"]

        elif "phi" in col:
            left, right = phi_left_filter, phi_right_filter
            selected_col = color_map["phi"]

        elif col in ["x", "det_x", "y", "det_y"]:
            left, right = -pos_filter, pos_filter
            selected_col = color_map["x"]

        elif col in ["s", "det_s"]:
            left, right = slowness_filter_left, slowness_filter_right
            selected_col = color_map["s"]

        elif "th_chi" in col:
            left, right = 0, 10
            selected_col = color_map["th_chi"]

        elif "res_ystr" in col:
            left, right = -res_ystr_filter, res_ystr_filter
            selected_col = color_map["res_ystr"]

        elif "res_tsum" in col:
            left, right = -res_tsum_filter, res_tsum_filter
            selected_col = color_map["res_tsum"]

        elif "res_tdif" in col:
            left, right = -res_tdif_filter, res_tdif_filter
            selected_col = color_map["res_tdif"]

        elif "t0" in col:
            left, right = t0_left_filter, t0_right_filter
            selected_col = color_map["t0"]

        # Plot histogram
        cond = (data > left) & (data < right)
        hist_data, bin_edges, _ = axs[i].hist(data, bins=50, alpha=0.7, label='Data', color=selected_col)

        axs[i].set_title(col)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        
        axs[i].set_xlim([left, right])

        # Fit Gaussian if enabled and data is sufficient
        if fit_gaussian and len(data) >= 10:
            try:
                # Use precomputed quantile bounds
                if col in quantile_bounds:
                    lower_bound, upper_bound = quantile_bounds[col]
                    filt_data = data[(data >= lower_bound) & (data <= upper_bound)]
                else:
                    lower_bound, upper_bound = left, right
                    filt_data = data[(data >= lower_bound) & (data <= upper_bound)]

                if len(filt_data) < 2:
                    axs[i].text(0.5, 0.5, "Not enough data to fit", transform=axs[i].transAxes, ha='center', va='center', color='gray')
                    continue

                # Fit Gaussian to the histogram data
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                popt, _ = curve_fit(gaussian, bin_centers, hist_data, p0=[np.mean(filt_data), np.std(filt_data), max(hist_data)])
                mu, sigma, amplitude = popt

                # Plot Gaussian fit
                x = np.linspace(lower_bound, upper_bound, 1000)
                axs[i].plot(x, gaussian(x, mu, sigma, amplitude), 'r-', label=f'Gaussian Fit\nμ={mu:.2g}, σ={sigma:.2g}')
                axs[i].legend()
            except (RuntimeError, ValueError):
                axs[i].text(0.5, 0.5, "Fit failed", transform=axs[i].transAxes, ha='center', va='center', color='red')

    # Remove unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(title, fontsize=16)
    if save_plots:
        final_filename = f'{fig_idx}_{title.replace(" ", "_")}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------- Data reading and preprocessing ---------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Get lists of files in the directories
unprocessed_files = sorted(os.listdir(base_directories["unprocessed_directory"]))
processing_files = sorted(os.listdir(base_directories["processing_directory"]))
completed_files = sorted(os.listdir(base_directories["completed_directory"]))

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

def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default

def _coerce_z_vector(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 4:
        return None
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(item) for item in vector):
        return None
    return vector

def _normalize_z_vector(z_vector: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    base = float(z_vector[0])
    return tuple(round(float(value) - base, 6) for value in z_vector)

def _iter_priority_vectors(value: object) -> list[tuple[float, float, float, float]]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        single = _coerce_z_vector(value)
        if single is not None:
            return [single]
        vectors: list[tuple[float, float, float, float]] = []
        for entry in value:
            coerced = _coerce_z_vector(entry)
            if coerced is not None:
                vectors.append(coerced)
        return vectors
    return []

def _load_task_1_z_priority_settings(config_obj: dict) -> dict[str, object]:
    settings: dict[str, object] = {
        "enabled": False,
        "simulated_only": True,
        "match_mode": "absolute",
        "source_csv": SIM_PARAMS_DEFAULT,
        "priority_abs": set(),
        "priority_norm": set(),
    }

    node = config_obj.get("task_1_z_position_priority")
    if not isinstance(node, dict):
        return settings

    settings["enabled"] = _as_bool(node.get("enabled"), False)
    settings["simulated_only"] = _as_bool(node.get("simulated_only"), True)

    match_mode = str(node.get("match_mode", "absolute")).strip().lower()
    if match_mode not in {"absolute", "normalized", "both"}:
        match_mode = "absolute"
    settings["match_mode"] = match_mode

    source_csv = node.get("source_csv")
    if source_csv:
        settings["source_csv"] = Path(os.path.expanduser(str(source_csv)))

    vectors = _iter_priority_vectors(node.get("vectors_mm"))
    settings["priority_abs"] = {tuple(round(value, 6) for value in vector) for vector in vectors}
    settings["priority_norm"] = {_normalize_z_vector(vector) for vector in vectors}

    if settings["enabled"] and not settings["priority_abs"]:
        print(
            "Warning: task_1_z_position_priority.enabled is true but vectors_mm is empty. "
            "Disabling z-priority for this run.",
            force=True,
        )
        settings["enabled"] = False

    return settings

def _load_simulated_z_lookup(
    csv_path: Path,
) -> dict[str, tuple[float, float, float, float]]:
    lookup: dict[str, tuple[float, float, float, float]] = {}
    if not csv_path.exists():
        print(
            f"Warning: z-priority lookup CSV not found: {csv_path}. "
            "Falling back to regular newest-file selection.",
            force=True,
        )
        return lookup

    required_cols = ["file_name", "z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]
    try:
        sim_df = pd.read_csv(csv_path, usecols=required_cols)
    except Exception as exc:
        print(
            f"Warning: failed to load z-priority lookup CSV {csv_path}: {exc}. "
            "Falling back to regular newest-file selection.",
            force=True,
        )
        return lookup

    for file_name, z1, z2, z3, z4 in sim_df.itertuples(index=False, name=None):
        file_key = str(file_name).strip().lower()
        if not file_key:
            continue
        try:
            z_vector = (float(z1), float(z2), float(z3), float(z4))
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(value) for value in z_vector):
            continue
        lookup[file_key] = z_vector

    return lookup

def _z_vector_matches_priority(
    z_vector: tuple[float, float, float, float],
    settings: dict[str, object],
) -> bool:
    abs_vector = tuple(round(float(value), 6) for value in z_vector)
    norm_vector = _normalize_z_vector(abs_vector)
    priority_abs = settings.get("priority_abs", set())
    priority_norm = settings.get("priority_norm", set())
    match_mode = settings.get("match_mode", "absolute")

    if match_mode == "normalized":
        return norm_vector in priority_norm
    if match_mode == "both":
        return abs_vector in priority_abs or norm_vector in priority_norm
    return abs_vector in priority_abs

def _select_latest_candidate_with_z_priority(
    files: Iterable[str],
    station: str,
    settings: dict[str, object],
    z_lookup: dict[str, tuple[float, float, float, float]],
    source_label: str,
    *,
    fallback_to_latest: bool = True,
) -> str | None:
    candidates = [name for name in files if isinstance(name, str) and name]
    if not candidates:
        return None

    if not settings.get("enabled", False):
        return select_latest_candidate(candidates, station) if fallback_to_latest else None

    prioritized: list[str] = []
    for name in candidates:
        key = name.lower()
        if settings.get("simulated_only", True) and not key.startswith("mi00"):
            continue
        z_vector = z_lookup.get(key)
        if z_vector is None:
            continue
        if _z_vector_matches_priority(z_vector, settings):
            prioritized.append(name)

    if prioritized:
        selected = max(prioritized, key=lambda name: newest_order_key(name, station))
        selected_z = z_lookup.get(selected.lower())
        print(
            f"[Z_PRIORITY] Selected prioritized file from {source_label}: {selected} "
            f"z={list(selected_z) if selected_z else 'NA'}",
            force=True,
        )
        return selected

    return select_latest_candidate(candidates, station) if fallback_to_latest else None

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    # Skip figure directories at startup; create lazily after selecting a file.
    if directory in (base_directories["base_figure_directory"], base_directories["figure_directory"]):
        continue
    os.makedirs(directory, exist_ok=True)

sync_unprocessed_with_date_range(
    config=config,
    unprocessed_directory=base_directories["unprocessed_directory"],
    out_of_date_directory=base_directories["out_of_date_directory"],
    log_fn=print,
    station_id=station,
    master_config_root=config_root,
)
unprocessed_files = os.listdir(base_directories["unprocessed_directory"])
processing_files = os.listdir(base_directories["processing_directory"])
completed_files = os.listdir(base_directories["completed_directory"])
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
            f"[DATE_RANGE] Ignoring {skipped_processing} out-of-range file(s) in PROCESSING_DIRECTORY."
        )
    if skipped_completed > 0:
        print(
            f"[DATE_RANGE] Ignoring {skipped_completed} out-of-range file(s) in COMPLETED_DIRECTORY."
        )

task_1_z_priority_settings = _load_task_1_z_priority_settings(config)
simulated_z_lookup: dict[str, tuple[float, float, float, float]] = {}
if task_1_z_priority_settings.get("enabled", False):
    source_csv = Path(task_1_z_priority_settings.get("source_csv", SIM_PARAMS_DEFAULT))
    simulated_z_lookup = _load_simulated_z_lookup(source_csv)
    print(
        f"[Z_PRIORITY] Enabled with {len(task_1_z_priority_settings.get('priority_abs', set()))} "
        f"configured vector(s); lookup entries={len(simulated_z_lookup)}",
        force=True,
    )

if user_file_selection:
    processing_file_path = user_file_path
    file_name = os.path.basename(user_file_path)
else:
    if last_file_test:
        latest_unprocessed = _select_latest_candidate_with_z_priority(
            unprocessed_files,
            station,
            task_1_z_priority_settings,
            simulated_z_lookup,
            "UNPROCESSED",
        )
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
            latest_processing = _select_latest_candidate_with_z_priority(
                processing_files,
                station,
                task_1_z_priority_settings,
                simulated_z_lookup,
                "PROCESSING",
            )
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

            else:
                latest_completed = _select_latest_candidate_with_z_priority(
                    completed_files,
                    station,
                    task_1_z_priority_settings,
                    simulated_z_lookup,
                    "COMPLETED",
                )
                if latest_completed:
                    file_name = latest_completed
                    processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                    completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

                    print(f"Reprocessing the newest file in COMPLETED: {completed_file_path}")
                    print(f"Moving '{completed_file_path}' to PROCESSING...")
                    safe_move(completed_file_path, processing_file_path)
                    print(f"File moved to PROCESSING: {processing_file_path}")
                else:
                    sys.exit("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")

    else:
        if unprocessed_files:
            priority_file = _select_latest_candidate_with_z_priority(
                unprocessed_files,
                station,
                task_1_z_priority_settings,
                simulated_z_lookup,
                "UNPROCESSED",
                fallback_to_latest=False,
            )
            if priority_file is not None:
                file_name = priority_file
                unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
                print(f"Moving prioritized '{file_name}' to PROCESSING...")
                safe_move(unprocessed_file_path, processing_file_path)
                print(f"File moved to PROCESSING: {processing_file_path}")
            else:
                print("Selecting a random file in UNPROCESSED...")
                file_name = random.choice(unprocessed_files)
                unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

                print(f"Moving '{file_name}' to PROCESSING...")
                safe_move(unprocessed_file_path, processing_file_path)
                print(f"File moved to PROCESSING: {processing_file_path}")

        elif processing_files:
            priority_file = _select_latest_candidate_with_z_priority(
                processing_files,
                station,
                task_1_z_priority_settings,
                simulated_z_lookup,
                "PROCESSING",
                fallback_to_latest=False,
            )
            if priority_file is not None:
                file_name = priority_file
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
                print(f"Processing prioritized file in PROCESSING: {processing_file_path}")
                error_file_path = os.path.join(base_directories["error_directory"], file_name)
                print(f"File '{processing_file_path}' is already in PROCESSING. Moving it temporarily to ERROR for analysis...")
                safe_move(processing_file_path, error_file_path)
                processing_file_path = error_file_path
                print(f"File moved to ERROR: {processing_file_path}")
            else:
                print("Selecting a random file in PROCESSING...")
                file_name = random.choice(processing_files)
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

                print(f"Processing file in PROCESSING: {processing_file_path}")
                error_file_path = os.path.join(base_directories["error_directory"], file_name)
                print(f"File '{processing_file_path}' is already in PROCESSING. Moving it temporarily to ERROR for analysis...")
                safe_move(processing_file_path, error_file_path)
                processing_file_path = error_file_path
                print(f"File moved to ERROR: {processing_file_path}")

        elif completed_files:
            if complete_reanalysis:
                priority_file = _select_latest_candidate_with_z_priority(
                    completed_files,
                    station,
                    task_1_z_priority_settings,
                    simulated_z_lookup,
                    "COMPLETED",
                    fallback_to_latest=False,
                )
                if priority_file is not None:
                    file_name = priority_file
                    completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
                    processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                    print(f"Moving prioritized '{file_name}' to PROCESSING...")
                    safe_move(completed_file_path, processing_file_path)
                    print(f"File moved to PROCESSING: {processing_file_path}")
                else:
                    print("Selecting a random file in COMPLETED...")
                    file_name = random.choice(completed_files)
                    completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
                    processing_file_path = os.path.join(base_directories["processing_directory"], file_name)

                    print(f"Moving '{file_name}' to PROCESSING...")
                    safe_move(completed_file_path, processing_file_path)
                    print(f"File moved to PROCESSING: {processing_file_path}")
            else:
                sys.exit("No files to process in UNPROCESSED, PROCESSING and decided to not reanalyze COMPLETED.")

        else:
            sys.exit("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")

# This is for all cases
file_path = processing_file_path
if save_plots:
    os.makedirs(base_directories["figure_directory"], exist_ok=True)
print(f"File to be processed, complete original path: {file_path}")
the_filename = os.path.basename(file_path)
print(f"File to process: {the_filename}")
basename_no_ext, file_extension = os.path.splitext(the_filename)
print(f"File basename (no extension): {basename_no_ext}")
status_filename_base = basename_no_ext
status_execution_date = initialize_status_row(
    csv_path_status,
    filename_base=status_filename_base,
    completion_fraction=0.0,
)

reprocessing_values: dict[str, object] = {}

reprocessing_parameters = load_reprocessing_parameters_for_file(station, str(task_number), basename_no_ext)
if not reprocessing_parameters.empty:
    global_variables["analysis_mode"] = 1
    print("Reprocessing parameters found for this file. Setting analysis_mode to 1.")
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

analysis_date = datetime.now().strftime("%Y-%m-%d")
# print(f"Analysis date and time: {analysis_date}")

# Modify the time of the processing file to the current time so it looks fresh
now = time.time()
os.utime(processing_file_path, (now, now))

# Check the station number in the datafile
# It might be that the data header is, instead of mi01: minI, which is the same, in that
# case consider minI as mi01
try:
    station_label = file_name[3]  # 4th character (index 3)
    print(f'File station number is: {station_label}')
    
    if station_label == "I":
        print("Station label is 'I', interpreting as station 1.")
        station_label = int(1)

    file_station_number = int(station_label)  # 4th character (index 3)
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
        param_hash=str(global_variables.get("param_hash", "")),
    )

left_limit_time = pd.to_datetime("1-1-2000", format='%d-%m-%Y')
right_limit_time = pd.to_datetime("1-1-2100", format='%d-%m-%Y')

if limit:
    print(f'Taking the first {limit_number} rows.')

# ------------------------------------------------------------------------------------------------------

# Move rejected_file to the rejected file folder
rejected_file = os.path.join(base_directories["rejected_files_directory"], f"temp_file_{date_execution}.csv")

print(f"Rejected-row log is {rejected_file}")
EXPECTED_COLUMNS = EXPECTED_COLUMNS_config  # Expected number of columns

ZERO_TOKEN_PATTERN = re.compile(r"0000\.0000")
LEADING_ZERO_PATTERN = re.compile(r"\b0+([0-9]+)")
MULTI_SPACE_PATTERN = re.compile(r" +")
XYEAR_PATTERN = re.compile(r"X(20\d{2})")
NEG_GAP_PATTERN = re.compile(r"(\w)-(\d)")
MALFORMED_NUMBER_PATTERN = re.compile(r"-?\d+\.\d+\.\d+")
VALID_YEARS = set(range(1999, 2100))

T_FRONT_PATTERN = re.compile(r"^T\d+_F_\d+$")
T_BACK_PATTERN = re.compile(r"^T\d+_B_\d+$")
Q_FRONT_PATTERN = re.compile(r"^Q\d+_F_\d+$")
Q_BACK_PATTERN = re.compile(r"^Q\d+_B_\d+$")

def _apply_bounds(
    frame: pd.DataFrame,
    column_names: Iterable[str],
    lower: float,
    upper: float,
    *,
    snapshot_originals=None,
) -> None:
    """Zero out values outside [lower, upper] for the provided columns."""
    cols = tuple(column_names)
    if not cols:
        return
    subset = frame.loc[:, cols]
    in_range_mask = (subset >= lower) & (subset <= upper)
    if snapshot_originals is not None:
        changed_cols = [col for col in cols if (~in_range_mask[col]).any()]
        if changed_cols:
            snapshot_originals(frame, changed_cols)
    frame.loc[:, cols] = subset.where(in_range_mask, 0)

# Function to process each line
def process_line(line):
    line = ZERO_TOKEN_PATTERN.sub('0', line)  # Replace '0000.0000' with '0'
    line = LEADING_ZERO_PATTERN.sub(r'\1', line)  # Remove leading zeros
    line = MULTI_SPACE_PATTERN.sub(',', line.strip())  # Replace multiple spaces with a comma
    line = XYEAR_PATTERN.sub(r'X\n\1', line)  # Replace XYYYY with X\nYYYY
    line = NEG_GAP_PATTERN.sub(r'\1 -\2', line)  # Ensure X-Y is properly spaced
    return line

# Function to check for malformed numbers (e.g., '-120.144.0')
def contains_malformed_numbers(line):
    return bool(MALFORMED_NUMBER_PATTERN.search(line))  # Detects multiple decimal points

# Function to validate year, month, and day
def is_valid_date(values):
    try:
        year, month, day = int(values[0]), int(values[1]), int(values[2])
        if year not in VALID_YEARS:  # Check valid years
            return False
        if not (1 <= month <= 12):  # Check valid month
            return False
        if not (1 <= day <= 31):  # Check valid day
            return False
        return True
    except ValueError:  # In case of non-numeric values
        return False


def _normalize_line_for_numpy(line: str) -> str:
    """Normalize raw text just enough for fast numeric parsing."""
    normalized = line.strip()
    normalized = ZERO_TOKEN_PATTERN.sub("0", normalized)
    normalized = LEADING_ZERO_PATTERN.sub(r"\1", normalized)
    if "X20" in normalized:
        normalized = XYEAR_PATTERN.sub(r"X \1", normalized)
    if "-" in normalized:
        normalized = NEG_GAP_PATTERN.sub(r"\1 -\2", normalized)
    return normalized


def _parse_raw_line_fast(line: str, expected_columns: int) -> np.ndarray | None:
    """Fast path for the common whitespace-only numeric rows."""
    stripped = line.strip()
    values = np.fromstring(stripped, sep=" ", dtype=np.float64)
    if values.size == expected_columns and is_valid_date(values[:3]):
        return values
    normalized = _normalize_line_for_numpy(stripped)
    values = np.fromstring(normalized, sep=" ", dtype=np.float64)
    if values.size == expected_columns and is_valid_date(values[:3]):
        return values
    return None


def _build_task1_input_dataframe(
    source_path: str,
    rejected_path: str,
    expected_columns: int,
    limit_rows: int | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Parse the raw .dat file in one pass and build the typed input dataframe."""
    read_lines = 0
    written_lines = 0
    stored_rows = 0
    flat_values = array("d")
    event_ids: list[int] = []

    with open(source_path, "r") as infile, open(rejected_path, "w") as rejectfile:
        for i, line in enumerate(infile, start=1):
            if line.lstrip().startswith("#"):
                continue
            read_lines += 1

            parsed_values = _parse_raw_line_fast(line, expected_columns)
            if parsed_values is None:
                cleaned_line = process_line(line)
                cleaned_values = cleaned_line.split(",")

                if len(cleaned_values) < 3 or not is_valid_date(cleaned_values[:3]):
                    rejectfile.write(f"Line {i} (Invalid date): {line.strip()}\n")
                    continue

                if contains_malformed_numbers(line):
                    rejectfile.write(f"Line {i} (Malformed number): {line.strip()}\n")
                    continue

                if len(cleaned_values) != expected_columns:
                    rejectfile.write(f"Line {i} (Wrong column count): {line.strip()}\n")
                    continue

                parsed_values = np.fromstring(
                    cleaned_line.replace(",", " "),
                    sep=" ",
                    dtype=np.float64,
                )
                if parsed_values.size != expected_columns:
                    rejectfile.write(f"Line {i} (Wrong column count): {line.strip()}\n")
                    continue

            written_lines += 1
            if limit_rows is None or stored_rows < limit_rows:
                flat_values.extend(parsed_values)
                event_ids.append(i)
                stored_rows += 1

    if stored_rows == 0:
        return pd.DataFrame(), read_lines, written_lines

    raw_matrix = np.frombuffer(flat_values, dtype=np.float64).reshape(stored_rows, expected_columns)
    datetime_components = {
        "year": np.rint(raw_matrix[:, 0]).astype(np.int16, copy=False),
        "month": np.rint(raw_matrix[:, 1]).astype(np.int8, copy=False),
        "day": np.rint(raw_matrix[:, 2]).astype(np.int8, copy=False),
        "hour": np.rint(raw_matrix[:, 3]).astype(np.int8, copy=False),
        "minute": np.rint(raw_matrix[:, 4]).astype(np.int8, copy=False),
        "second": np.rint(raw_matrix[:, 5]).astype(np.int8, copy=False),
    }
    datetime_series = pd.to_datetime(datetime_components)

    value_columns = [f"column_{i}" for i in range(6, expected_columns)]
    value_data = raw_matrix[:, 6:].astype(np.float32, copy=False)
    read_df = pd.DataFrame(value_data, columns=value_columns, copy=False)
    # Stable per-file event identity: raw file line number before any Task 1 filtering.
    read_df.insert(0, "event_id", np.asarray(event_ids, dtype=np.int64))
    read_df.insert(0, "datetime", datetime_series)
    read_df["column_6"] = np.rint(raw_matrix[:, 6]).astype(np.int8, copy=False)
    return read_df, read_lines, written_lines

# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------- 
# -----------------------------------------------------------------------------------------------------------
# TASK 1 start
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# Process the file
read_df, read_lines, written_lines = _build_task1_input_dataframe(
    file_path,
    rejected_file,
    EXPECTED_COLUMNS,
    limit_rows=limit_number if limit else None,
)

if written_lines == 0:
    print("No valid lines found after preprocessing; skipping file.")
    if user_file_selection == False:
        error_file_path = os.path.join(base_directories["error_directory"], file_name)
        process_file(file_path, error_file_path)
    sys.exit("Empty temp file generated; no valid data to process.")

# Print the number of rows in input
print(f"\nOriginal file has {read_lines} lines.")
print(f"Processed file has {written_lines} lines.")
if read_lines > 0:
    valid_lines_in_dat_file = written_lines / read_lines * 100
else:
    valid_lines_in_dat_file = 0.0
    print("Warning: input file has zero lines; setting valid line percentage to 0.0.")
print(f"--> A {valid_lines_in_dat_file:.2f}% of the lines were valid.\n")

global_variables['valid_lines_in_binary_file_percentage'] =  valid_lines_in_dat_file

if task1_plot_enabled("event_total_charge_raw"):
    raw_total_charge_columns = [
        *(f"column_{idx}" for idx in range(15, 23)),
        *(f"column_{idx}" for idx in range(31, 39)),
        *(f"column_{idx}" for idx in range(47, 55)),
        *(f"column_{idx}" for idx in range(63, 71)),
    ]
    raw_total_charge_columns = [col for col in raw_total_charge_columns if col in read_df.columns]
    if raw_total_charge_columns:
        raw_total_charge = read_df.loc[:, raw_total_charge_columns].sum(axis=1, skipna=True)
        raw_total_charge = raw_total_charge[np.isfinite(raw_total_charge.to_numpy())]
        if not raw_total_charge.empty:
            raw_total_charge_plot = raw_total_charge.clip(lower=0, upper=2000)
            plt.figure(figsize=(10, 6))
            plt.hist(raw_total_charge_plot, bins=100, alpha=0.8, color='tab:blue')
            plt.yscale('log')
            plt.xlim(0, 2000)
            plt.title(f"Raw total charge per event, {basename_no_ext}")
            plt.xlabel("Total event charge (sum of raw Q channels)")
            plt.ylabel("Number of events")
            plt.tight_layout()

            if save_plots:
                final_filename = f'{fig_idx}_raw_total_event_charge.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()

_prof["s_data_read_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("-------------------------- Filter 1: by date -------------------------")
print("----------------------------------------------------------------------")

read_df = read_df.loc[read_df["datetime"].between(left_limit_time, right_limit_time)].copy()
gc.collect()
if not isinstance(read_df.set_index('datetime').index, pd.DatetimeIndex):
    raise ValueError("The index is not a DatetimeIndex. Check 'datetime' column formatting.")

# Print the count frequency of the values in column_6
print(read_df['column_6'].value_counts())
# Take only the rows in which column_6 is equal to 1

self_trigger_df = read_df.loc[read_df['column_6'] == 2].copy()
selected_df = read_df.loc[read_df['column_6'] == 1].copy()
del read_df
self_trigger = not self_trigger_df.empty # If self_trigger_df has values, define an indicator as True

raw_data_len = len(selected_df)
if raw_data_len == 0 and not self_trigger:
    print("No coincidence nor self-trigger events.")
    sys.exit(1)

# Note that the middle between start and end time could also be taken. This is for calibration storage.
if "datetime" in selected_df.columns:
    datetime_series = pd.to_datetime(selected_df["datetime"], errors="coerce").dropna()
else:
    datetime_series = pd.Series(dtype="datetime64[ns]")
if datetime_series.empty:
    print(
        f"Warning: No valid datetime rows found in {file_name}; moving file to ERROR and skipping."
    )
    if not user_file_selection:
        error_file_path = os.path.join(base_directories["error_directory"], file_name)
        print(f"Moving file '{file_name}' to ERROR directory: {error_file_path}")
        process_file(file_path, error_file_path)
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

# -------------------------------------------------------------------------------
# ------------ Input file and data managing to select configuration -------------
# -------------------------------------------------------------------------------

conf_value = None
z_source = "unset"
simulated_z_positions, simulated_param_hash = resolve_simulated_z_positions(
    basename_no_ext,
    Path(base_directory),
    dat_path=Path(file_path),
)
if simulated_param_hash:
    global_variables["param_hash"] = simulated_param_hash

is_simulated_file = basename_no_ext.startswith("mi00")
used_input_file = False

if is_simulated_file:
    if simulated_param_hash:
        print(f"Simulated param_hash resolved: {simulated_param_hash}")
    else:
        print("Warning: Simulated param_hash missing; default z_positions will be used.")

if is_simulated_file and simulated_z_positions is None:
    print("Warning: Simulated file missing param_hash; using default z_positions.")
    found_matching_conf = False
    z_positions = np.array([0, 150, 300, 450])  # In mm
    z_source = "simulated_default_missing_param_hash"
elif simulated_z_positions is not None:
    z_positions = np.array(simulated_z_positions, dtype=float)
    found_matching_conf = True
    print(f"Using simulated z_positions from param_hash={simulated_param_hash}")
    z_source = "simulated_param_hash"
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
        try:
            conf_value = float(selected_conf.get("conf"))
        except (TypeError, ValueError):
            conf_value = None
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

def _zpos_from_conf(row):
    return np.array([row.get(f"P{i}", np.nan) for i in range(1, 5)])

# If any z_positions is NaN or all zeros, find the closest non-zero configuration.
if np.isnan(z_positions).any() or np.all(z_positions == 0):
    if used_input_file:
        print("Warning: Invalid z_positions in selected configuration; searching for closest non-zero configuration.")
        valid_rows = input_file.dropna(subset=["start"]).copy()
        valid_rows["has_nonzero_z"] = valid_rows.apply(
            lambda r: np.any(_zpos_from_conf(r) != 0), axis=1
        )
        valid_rows = valid_rows[valid_rows["has_nonzero_z"]]
        if not valid_rows.empty:
            valid_rows["delta"] = (valid_rows["start_day"] - start_day).abs()
            selected_conf = valid_rows.sort_values("delta").iloc[0]
            print(f"Selected non-zero configuration: {selected_conf['conf']}")
            z_positions = _zpos_from_conf(selected_conf)
            try:
                conf_value = float(selected_conf.get("conf"))
            except (TypeError, ValueError):
                conf_value = None
            z_source = f"input_file_nonzero_fallback_conf_{selected_conf.get('conf')}"
        else:
            print("Error: No non-zero z_positions available. Using default z_positions.")
            z_positions = np.array([0, 150, 300, 450])  # In mm
            z_source = "default_no_nonzero_z_available"
    else:
        print("Error: Invalid z_positions without config fallback. Using default z_positions.")
        z_positions = np.array([0, 150, 300, 450])  # In mm
        z_source = "default_invalid_without_input"

# Print the resulting z_positions
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")
z_vector_mm = [round(float(value), 3) for value in z_positions.tolist()]
print(
    f"[Z_TRACE] file={basename_no_ext} source={z_source} "
    f"param_hash={simulated_param_hash or 'NA'} z_vector_mm={z_vector_mm}",
    force=True,
)

print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print(f"------------- Starting date is {save_filename_suffix} -------------------") # This is longer so it displays nicely
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Defining the directories that will store the data
save_full_filename = f"full_list_events_{save_filename_suffix}.txt"
save_filename = f"list_events_{save_filename_suffix}.txt"
save_pdf_filename = f"mingo{str(station).zfill(2)}_task1_{save_filename_suffix}.pdf"

save_pdf_path = os.path.join(base_directories["pdf_directory"], save_pdf_filename)

# Check if the file exists and its size
if os.path.exists(save_filename):
    if os.path.getsize(save_filename) >= 1 * 1024 * 1024: # Bigger than 1MB
        if force_replacement == False:
            print("Datafile found and it looks completed. Exiting...")
            sys.exit()  # Exit the script
        else:
            print("Datafile found and it is not empty, but 'force_replacement' is True, so it creates new datafiles anyway.")
    else:
        print("Datafile found, but empty.")

column_indices = {
    'T1_F': range(55, 59), 'T1_B': range(59, 63), 'Q1_F': range(63, 67), 'Q1_B': range(67, 71),
    'T2_F': range(39, 43), 'T2_B': range(43, 47), 'Q2_F': range(47, 51), 'Q2_B': range(51, 55),
    'T3_F': range(23, 27), 'T3_B': range(27, 31), 'Q3_F': range(31, 35), 'Q3_B': range(35, 39),
    'T4_F': range(7, 11), 'T4_B': range(11, 15), 'Q4_F': range(15, 19), 'Q4_B': range(19, 23)
}

channel_source_columns: list[str] = []
channel_rename_map: dict[str, str] = {}
for key, idx_range in column_indices.items():
    for i, col_idx in enumerate(idx_range, start=1):
        source_column = f"column_{col_idx}"
        target_column = f"{key}_{i}"
        channel_source_columns.append(source_column)
        channel_rename_map[source_column] = target_column

working_df = selected_df.loc[:, ["event_id", "datetime", *channel_source_columns]].copy()
working_df = working_df.rename(columns=channel_rename_map)
# Track rows that later drop out of Task 1 so their original indexed values stay inspectable.
removed_rows_df = working_df.iloc[0:0].copy()
tracking_base_index = working_df.index.copy()
original_columns_store: dict[str, pd.Series] = {}
channel_pair_plot_reference_df = pd.DataFrame(index=working_df.index.copy())


def snapshot_original_columns_once(
    frame: pd.DataFrame,
    column_names: Iterable[str],
) -> None:
    """Store a column once, before its first in-place mutation, to keep original values."""
    if not track_removed_rows or frame is not working_df:
        return
    for col in column_names:
        if col in frame.columns and col not in original_columns_store:
            original_columns_store[col] = frame[col].copy()


def _restore_original_values(rows: pd.DataFrame) -> pd.DataFrame:
    if not track_removed_rows or rows.empty:
        return rows
    restored_rows = rows.copy()
    for col, original_series in original_columns_store.items():
        if col in restored_rows.columns:
            restored_rows.loc[:, col] = original_series.reindex(restored_rows.index)
    return restored_rows


def append_removed_rows(rows: pd.DataFrame) -> None:
    """Append fully removed rows with original index and pre-modification values preserved."""
    global removed_rows_df
    if not track_removed_rows or rows.empty:
        return
    rows_to_add = _restore_original_values(rows)
    if not removed_rows_df.empty:
        rows_to_add = rows_to_add.loc[~rows_to_add.index.isin(removed_rows_df.index)]
        if rows_to_add.empty:
            return
    removed_rows_df = pd.concat([removed_rows_df, rows_to_add], ignore_index=False, sort=False)


def append_removed_rows_from_mask(frame: pd.DataFrame, removed_mask: pd.Series) -> None:
    if not track_removed_rows or frame.empty:
        return
    aligned_mask = removed_mask.reindex(frame.index, fill_value=False)
    if aligned_mask.any():
        append_removed_rows(frame.loc[aligned_mask].copy())


def build_original_columns_frame() -> pd.DataFrame:
    if not track_removed_rows:
        return pd.DataFrame(index=tracking_base_index)
    if not original_columns_store:
        return pd.DataFrame(index=tracking_base_index)
    return pd.DataFrame(
        {col: series.reindex(tracking_base_index) for col, series in original_columns_store.items()},
        index=tracking_base_index,
    )


def build_channel_pair_plot_frames(
    current_df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the retained set plus the reference frame used to show removed points."""
    if current_df.empty:
        return current_df, current_df.iloc[0:0].copy()
    if reference_df is None:
        reference_df = channel_pair_plot_reference_df
    if reference_df.empty:
        return current_df, current_df.copy()
    reference_df = reference_df.reindex(current_df.index).copy()
    for col in current_df.columns:
        if col not in reference_df.columns:
            reference_df[col] = current_df[col]
    return current_df, reference_df


def build_task1_channel_filter_stage_frames(
    df_input: pd.DataFrame,
    *,
    original_reference_df: pd.DataFrame | None,
    relation_limits_by_type: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df_input.empty:
        empty_df = df_input.iloc[0:0].copy()
        return empty_df, empty_df, empty_df

    if original_reference_df is None or original_reference_df.empty:
        original_df = df_input.copy()
    else:
        original_df = original_reference_df.reindex(df_input.index).copy()
        for col in df_input.columns:
            if col not in original_df.columns:
                original_df[col] = df_input[col]

    pre_combination_df = df_input.copy()
    final_df = pre_combination_df.copy()
    apply_task1_plane_combination_filter(
        final_df,
        relation_limits_by_type=relation_limits_by_type,
    )
    return original_df, pre_combination_df, final_df


original_number_of_events = len(working_df)
if status_execution_date is not None:
    update_status_progress(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        completion_fraction=0.5,
        param_hash=str(global_variables.get("param_hash", "")),
    )

snapshot_original_columns_once(working_df, ["datetime"])
working_df["datetime"] = selected_df['datetime']
working_df = working_df.rename(columns=lambda col: col.replace("_diff_", "_dif_"))

# print("Columns right after initial assignment (before raw_tt computation):")
# for col in working_df.columns:
#     print(f" - {col}")

if found_matching_conf and conf_value is not None:
    # --- Conditional swap for station 2, Plane 4: swap channels 2 and 4 ---
    if conf_value < 2:
        if station == "2":
            print("Configuration of the detector is less than 2.")
            print("Swapping channels that give problems in plane 4.")
            plane4_keys = ['T4_F', 'T4_B', 'Q4_F', 'Q4_B']
            for key in plane4_keys:
                col2 = f'{key}_3'
                col4 = f'{key}_4'
                snapshot_original_columns_once(working_df, [col2, col4])
                working_df[[col2, col4]] = working_df[[col4, col2]].values  # swap columns

if self_trigger:
    working_st_df = self_trigger_df.loc[:, ["event_id", "datetime", *channel_source_columns]].copy()
    working_st_df = working_st_df.rename(columns=channel_rename_map)
    working_st_df = working_st_df.rename(columns=lambda col: col.replace("_diff_", "_dif_"))
    
    if found_matching_conf and conf_value is not None:
        # --- Conditional swap for station 2, Plane 4: swap channels 2 and 4 ---
        if conf_value < 2:
            if station == "2":
                print("Configuration of the detector is less than 2.")
                print("Swapping channels that give problems in plane 4.")
                plane4_keys = ['T4_F', 'T4_B', 'Q4_F', 'Q4_B']
                for key in plane4_keys:
                    col2 = f'{key}_3'
                    col4 = f'{key}_4'
                    working_st_df[[col2, col4]] = working_st_df[[col4, col2]].values  # swap columns

del selected_df
if self_trigger:
    del self_trigger_df
gc.collect()

# ----------------------------------------------------------------------------------
# Count the number of non-zero entries per channel in the whole dataframe ----------
# ----------------------------------------------------------------------------------

# Count per each column the number of non-zero entries and save it in a column of
# global_variables called TX_F_Y_entries or TX_B_Y_entries

# Count for main dataframe (non-self-trigger)
for key, idx_range in column_indices.items():
    for i in range(1, len(idx_range) + 1):
        colname = f"{key}_{i}"
        count = (working_df[colname] != 0).sum()
        global_var_name = f"{key}_{i}_entries_original"
        global_variables[global_var_name] = count

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# Trigger type helpers -------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

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

# Apply the function to the DataFrame
working_df = compute_tt(working_df, "raw_tt")

raw_tt_counts = working_df["raw_tt"].value_counts()

raw_channel_patterns = build_task1_channel_pattern_series(working_df)
store_pattern_rates(pattern_metadata, raw_channel_patterns, "raw_channel_pattern", working_df)

# Always compute raw-stage channel contagion metrics for metadata.
raw_channel_matrix_data = _compute_channel_conditional_matrix(working_df)
_store_channel_contagion_metrics_variant(
    "global_raw",
    raw_channel_matrix_data,
)
raw_channel_by_tt_data = _compute_channel_conditional_matrix_for_tt(working_df, "raw_tt", 1234)
_store_channel_contagion_metrics_variant(
    "by_tt_raw",
    raw_channel_by_tt_data,
)

# Print the counts of each raw_tt value and the percentage
total_events = len(working_df)
print("Raw TT counts and percentages:")
for tt_value, count in sorted(raw_tt_counts.items()):
    percentage = (count / total_events) * 100
    print(f"  Raw TT {tt_value}: {count} events ({percentage:.2f}%)")

if self_trigger:
    working_st_df = compute_tt(working_st_df, "raw_tt")

_debug_plot_filter_group(
    working_df,
    [col for col in working_df.columns if re.match(r"^[TQ][1-4]_[FB]_[1-4]$", col)] + ["raw_tt"],
    None,
    "incoming parquet: main channels and raw_tt",
    tag="NON-TUNABLE",
    max_cols_per_fig=20,
)

if task1_plot_enabled("raw_tt_overview"):
    event_counts = working_df['raw_tt'].value_counts()

    plt.figure(figsize=(10, 6))
    event_counts.plot(kind='bar', alpha=0.7)
    plt.title(f'Number of Events per Raw TT Label, {start_time}')
    plt.xlabel('Raw TT Label')
    plt.ylabel('Number of Events')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plots:
        final_filename = f'{fig_idx}_raw_TT.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close()

# --- Pre-filter (raw) channel contamination matrices ---
if task1_plot_enabled("channel_contamination_matrix_32_raw"):
    fig_idx = _plot_channel_contamination_global(
        working_df, "raw", fig_idx, base_directories["figure_directory"],
        save_plots, show_plots, plot_list,
    )

if task1_plot_enabled("channel_contagion_by_tt_raw"):
    fig_idx = _plot_channel_contagion_by_tt(
        working_df, "raw_tt", "raw", fig_idx, base_directories["figure_directory"],
        save_plots, show_plots, plot_list,
    )

if self_trigger:
    if task1_plot_enabled("tsum_spread_diagnostics"):
   
        event_counts = working_st_df['raw_tt'].value_counts()

        plt.figure(figsize=(10, 6))
        event_counts.plot(kind='bar', alpha=0.7)
        plt.title(f'Number of Events per Raw TT Label, {start_time}')
        plt.xlabel('Raw TT Label')
        plt.ylabel('Number of Events')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            final_filename = f'{fig_idx}_raw_TT_ST.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')

        if show_plots: plt.show()
        plt.close()

# -----------------------------------------------------------------------------
# New channel-wise plot -------------------------------------------------------
# -----------------------------------------------------------------------------

if task1_plot_enabled("channel_histograms_raw"):
    # Create the grand figure for T values
    fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_T = axes_T.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'
            col_B = f'{key}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with T-specific clipping and bins
            axes_T[i*4 + j].hist(y_F[(y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_T[i*4 + j].hist(y_B[(y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_T[i*4 + j].axvline(x=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
            axes_T[i*4 + j].axvline(x=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
            axes_T[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_T[i*4 + j].legend()
            
            if log_scale:
                axes_T[i*4 + j].set_yscale('log')  # For T values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for T values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_T.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_T)

    # Create the grand figure for Q values
    fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_Q = axes_Q.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key.replace("T", "Q")}_F_{j+1}'
            col_B = f'{key.replace("T", "Q")}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with Q-specific clipping and bins
            axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_Q[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
            axes_Q[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
            axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_Q[i*4 + j].legend()
            
            if log_scale:
                axes_Q[i*4 + j].set_yscale('log')  # For Q values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for Q values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_Q.png'
        fig_idx += 1
        
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_Q)

if self_trigger:
    if task1_plot_enabled("channel_histograms_self_trigger"):
   
        # Create the grand figure for T values
        fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_T = axes_T.flatten()
        
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = working_st_df[col_F]
                y_B = working_st_df[col_B]
                
                # Plot histograms with T-specific clipping and bins
                axes_T[i*4 + j].hist(y_F[(y_F != 0) & (y_F > T_clip_min_ST) & (y_F < T_clip_max_ST)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_T[i*4 + j].hist(y_B[(y_B != 0) & (y_B > T_clip_min_ST) & (y_B < T_clip_max_ST)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_T[i*4 + j].axvline(x=T_F_left_pre_cal_ST, color='red', linestyle='--', label='T_left_pre_cal_ST')
                axes_T[i*4 + j].axvline(x=T_F_right_pre_cal_ST, color='blue', linestyle='--', label='T_right_pre_cal_ST')
                axes_T[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_T[i*4 + j].legend()
                
                if log_scale:
                    axes_T[i*4 + j].set_yscale('log')  # For T values

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"SELF TRIGGER. Grand Figure for T values, mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_T_ST.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')

        if show_plots: plt.show()
        plt.close(fig_T)

        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key.replace("T", "Q")}_F_{j+1}'
                col_B = f'{key.replace("T", "Q")}_B_{j+1}'
                y_F = working_st_df[col_F]
                y_B = working_st_df[col_B]
                
                # Plot histograms with Q-specific clipping and bins
                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min_ST) & (y_F < Q_clip_max_ST)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min_ST) & (y_B < Q_clip_max_ST)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_Q[i*4 + j].axvline(x=Q_F_left_pre_cal_ST, color='red', linestyle='--', label='Q_left_pre_cal_ST')
                axes_Q[i*4 + j].axvline(x=Q_F_right_pre_cal_ST, color='blue', linestyle='--', label='Q_right_pre_cal_ST')
                axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_Q[i*4 + j].legend()
                
                if log_scale:
                    axes_Q[i*4 + j].set_yscale('log')  # For Q values

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"SELF TRIGGER. Grand Figure for Q values, mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_Q_ST.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close(fig_Q)

# -----------------------------------------------------------------------------------------------

if task1_plot_enabled("tq_scatter_raw"):
    # Initialize figure and axes for scatter plot of Time vs Charge
    fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_TQ = axes_TQ.flatten()

    # Iterate over each module (T1, T2, T3, T4)
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'  # Time F column
            col_B = f'{key}_B_{j+1}'  # Time B column
            
            y_F = working_df[col_F]  # Time values for front
            y_B = working_df[col_B]  # Time values for back
            
            charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'  # Corresponding charge column for front
            charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'  # Corresponding charge column for back
            
            charge_F = working_df[charge_col_F]  # Charge values for front
            charge_B = working_df[charge_col_B]  # Charge values for back
            
            # Apply clipping ranges to the data
            mask_F = (y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max) & (charge_F > Q_clip_min) & (charge_F < Q_clip_max)
            mask_B = (y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max) & (charge_B > Q_clip_min) & (charge_B < Q_clip_max)
            
            # Plot scatter plots for Time F vs Charge F and Time B vs Charge B
            axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
            axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
            
            # Plot threshold lines for time and charge
            axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
            axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
            
            axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_TQ[i*4 + j].legend()

    # Adjust the layout and title
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Scatter Plot for T vs Q values, mingo0{station}\n{start_time}", fontsize=16)

    # Save the plot
    if save_plots:
        final_filename = f'{fig_idx}_scatter_plot_TQ.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    # Show the plot if requested
    if show_plots:
        plt.show()

    # Close the plot to avoid excessive memory usage
    plt.close(fig_TQ)

if task1_plot_enabled("tq_scatter_raw_by_tt"):
    _tt_vals = sorted(working_df["raw_tt"].dropna().unique()) if "raw_tt" in working_df.columns else []
    for _tt_v in _tt_vals:
        _tt_sub = working_df[working_df["raw_tt"] == _tt_v]
        if len(_tt_sub) < 10:
            continue
        _ttl = normalize_tt_label(_tt_v)
        fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))
        axes_TQ = axes_TQ.flatten()
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = _tt_sub[col_F]
                y_B = _tt_sub[col_B]
                charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'
                charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'
                charge_F = _tt_sub[charge_col_F]
                charge_B = _tt_sub[charge_col_B]
                mask_F = (y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max) & (charge_F > Q_clip_min) & (charge_F < Q_clip_max)
                mask_B = (y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max) & (charge_B > Q_clip_min) & (charge_B < Q_clip_max)
                axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
                axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
                axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
                axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
                axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
                axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
                axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_TQ[i*4 + j].legend()
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Scatter T vs Q [raw_tt={_ttl}, N={len(_tt_sub)}], mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_scatter_plot_TQ_tt{_ttl}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close(fig_TQ)

# -----------------------------------------------------------------------------

_prof["s_filter_date_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("------------------ Filter 1.1.1: uncalibrated data -------------------")
print("----------------------------------------------------------------------")

# FILTER 2: TF, TB, QF, QB PRECALIBRATED THRESHOLDS --> 0 if out --------------

snapshot_original_columns_once(
    working_df,
    [col for col in working_df.columns if working_df[col].isna().any()],
)
working_df.fillna(0, inplace=True)
T_F_cols = collect_columns(working_df.columns, T_FRONT_PATTERN)
T_B_cols = collect_columns(working_df.columns, T_BACK_PATTERN)
Q_F_cols = collect_columns(working_df.columns, Q_FRONT_PATTERN)
Q_B_cols = collect_columns(working_df.columns, Q_BACK_PATTERN)
channel_pair_plot_reference_columns = [
    col
    for col in ["raw_tt", *T_F_cols, *T_B_cols, *Q_F_cols, *Q_B_cols]
    if col in working_df.columns
]
channel_pair_plot_reference_df = working_df.loc[:, channel_pair_plot_reference_columns].copy()

_debug_plot_filter_group(
    working_df,
    T_F_cols + T_B_cols,
    [T_F_left_pre_cal, T_F_right_pre_cal],
    "T_side_left/right_pre_cal",
)
_debug_plot_filter_group(
    working_df,
    Q_F_cols + Q_B_cols,
    [Q_F_left_pre_cal, Q_F_right_pre_cal],
    "Q_side_left/right_pre_cal",
)

if self_trigger:
    working_st_df.fillna(0, inplace=True)

    st_T_F_cols = collect_columns(working_st_df.columns, T_FRONT_PATTERN)
    st_T_B_cols = collect_columns(working_st_df.columns, T_BACK_PATTERN)
    st_Q_F_cols = collect_columns(working_st_df.columns, Q_FRONT_PATTERN)
    st_Q_B_cols = collect_columns(working_st_df.columns, Q_BACK_PATTERN)

    _debug_plot_filter_group(
        working_st_df,
        st_T_F_cols + st_T_B_cols,
        [T_F_left_pre_cal_ST, T_F_right_pre_cal_ST],
        "T_side_left/right_pre_cal_ST",
    )
    _debug_plot_filter_group(
        working_st_df,
        st_Q_F_cols + st_Q_B_cols,
        [Q_F_left_pre_cal_ST, Q_F_right_pre_cal_ST],
        "Q_side_left/right_pre_cal_ST",
    )

# -----------------------------------------------------------------------------
# New channel-wise plot -------------------------------------------------------
# -----------------------------------------------------------------------------

if task1_plot_enabled("channel_histograms_filtered"):
    # Create the grand figure for T values
    fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_T = axes_T.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'
            col_B = f'{key}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with T-specific clipping and bins
            axes_T[i*4 + j].hist(y_F[(y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_T[i*4 + j].hist(y_B[(y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_T[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_T[i*4 + j].legend()
            
            if log_scale:
                axes_T[i*4 + j].set_yscale('log')  # For T values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for T values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_T.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_T)

    # Create the grand figure for Q values
    fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_Q = axes_Q.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key.replace("T", "Q")}_F_{j+1}'
            col_B = f'{key.replace("T", "Q")}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with Q-specific clipping and bins
            axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_Q[i*4 + j].legend()
            
            if log_scale:
                axes_Q[i*4 + j].set_yscale('log')  # For Q values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for Q values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_Q.png'
        fig_idx += 1
        
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_Q)

# Per trigger type
create_super_essential_plots = False

if create_plots or create_super_essential_plots:

    for tt_value in sorted(working_df['raw_tt'].unique()):
        filtered_df = working_df[working_df['raw_tt'] == tt_value]

        # Create the grand figure for T values
        fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_T = axes_T.flatten()
        
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = filtered_df[col_F]
                y_B = filtered_df[col_B]
                
                # Plot histograms with T-specific clipping and bins
                axes_T[i*4 + j].hist(y_F[(y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_T[i*4 + j].hist(y_B[(y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_T[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_T[i*4 + j].legend()
                
                if log_scale:
                    axes_T[i*4 + j].set_yscale('log')  # For T values

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for T values, {tt_value}, mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_{tt_value}_grand_figure_T.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')

        if show_plots: plt.show()
        plt.close(fig_T)

        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key.replace("T", "Q")}_F_{j+1}'
                col_B = f'{key.replace("T", "Q")}_B_{j+1}'
                y_F = filtered_df[col_F]
                y_B = filtered_df[col_B]
                
                # Plot histograms with Q-specific clipping and bins

                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_Q[i*4 + j].legend()
                
                # if log_scale:
                #     axes_Q[i*4 + j].set_yscale('log')  # For Q values

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for Q values, {tt_value}, mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_{tt_value}_grand_figure_Q.png'
            fig_idx += 1
            
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')

        if show_plots: plt.show()
        plt.close(fig_Q)

if task1_plot_enabled("tq_scatter_filtered"):
    # Initialize figure and axes for scatter plot of Time vs Charge
    fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_TQ = axes_TQ.flatten()

    # Iterate over each module (T1, T2, T3, T4)
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'  # Time F column
            col_B = f'{key}_B_{j+1}'  # Time B column
            
            y_F = working_df[col_F]  # Time values for front
            y_B = working_df[col_B]  # Time values for back
            
            charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'  # Corresponding charge column for front
            charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'  # Corresponding charge column for back
            
            charge_F = working_df[charge_col_F]  # Charge values for front
            charge_B = working_df[charge_col_B]  # Charge values for back
            
            # Apply clipping ranges to the data
            mask_F = (y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max) & (charge_F > Q_clip_min) & (charge_F < Q_clip_max)
            mask_B = (y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max) & (charge_B > Q_clip_min) & (charge_B < Q_clip_max)
            
            # Plot scatter plots for Time F vs Charge F and Time B vs Charge B
            axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
            axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
            
            # Plot threshold lines for time and charge
            axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
            axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
            
            axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_TQ[i*4 + j].legend()

    # Adjust the layout and title
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Scatter Plot for T vs Q values, mingo0{station}\n{start_time}", fontsize=16)

    # Save the plot
    if save_plots:
        final_filename = f'{fig_idx}_scatter_plot_TQ_filtered.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    # Show the plot if requested
    if show_plots:
        plt.show()

    # Close the plot to avoid excessive memory usage
    plt.close(fig_TQ)

if task1_plot_enabled("tq_scatter_filtered_by_tt"):
    _tt_vals = sorted(working_df["raw_tt"].dropna().unique()) if "raw_tt" in working_df.columns else []
    for _tt_v in _tt_vals:
        _tt_sub = working_df[working_df["raw_tt"] == _tt_v]
        if len(_tt_sub) < 10:
            continue
        _ttl = normalize_tt_label(_tt_v)
        fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))
        axes_TQ = axes_TQ.flatten()
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = _tt_sub[col_F]
                y_B = _tt_sub[col_B]
                charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'
                charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'
                charge_F = _tt_sub[charge_col_F]
                charge_B = _tt_sub[charge_col_B]
                mask_F = (y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max) & (charge_F > Q_clip_min) & (charge_F < Q_clip_max)
                mask_B = (y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max) & (charge_B > Q_clip_min) & (charge_B < Q_clip_max)
                axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
                axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
                axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
                axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
                axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
                axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
                axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_TQ[i*4 + j].legend()
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Scatter T vs Q filtered [raw_tt={_ttl}, N={len(_tt_sub)}], mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_scatter_plot_TQ_filtered_tt{_ttl}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close(fig_TQ)

if self_trigger:
    if task1_plot_enabled("tq_scatter_filtered"):
        # Initialize figure and axes for scatter plot of Time vs Charge
        fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_TQ = axes_TQ.flatten()

        # Iterate over each module (T1, T2, T3, T4)
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'  # Time F column
                col_B = f'{key}_B_{j+1}'  # Time B column
                
                y_F = working_st_df[col_F]  # Time values for front
                y_B = working_st_df[col_B]  # Time values for back
                
                charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'  # Corresponding charge column for front
                charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'  # Corresponding charge column for back
                
                charge_F = working_st_df[charge_col_F]  # Charge values for front
                charge_B = working_st_df[charge_col_B]  # Charge values for back
                
                # Apply clipping ranges to the data
                mask_F = (y_F != 0) & (y_F > T_clip_min_ST) & (y_F < T_clip_max_ST) & (charge_F > Q_clip_min_ST) & (charge_F < Q_clip_max_ST)
                mask_B = (y_B != 0) & (y_B > T_clip_min_ST) & (y_B < T_clip_max_ST) & (charge_B > Q_clip_min_ST) & (charge_B < Q_clip_max_ST)
                
                # Plot scatter plots for Time F vs Charge F and Time B vs Charge B
                axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
                axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
                
                # Plot threshold lines for time and charge
                axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal_ST, color='red', linestyle='--', label='T_left_pre_cal_ST')
                axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal_ST, color='blue', linestyle='--', label='T_right_pre_cal_ST')
                axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal_ST, color='red', linestyle='--', label='Q_left_pre_cal_ST')
                axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal_ST, color='blue', linestyle='--', label='Q_right_pre_cal_ST')
                
                axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_TQ[i*4 + j].legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"SELF TRIGGER. Scatter Plot for T vs Q values, mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_scatter_plot_TQ_filtered_ST.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close(fig_TQ)
    

# -----------------------------------------------------------------------------
# Comprobation of emptiness of the columns
# -----------------------------------------------------------------------------

# Count the number of nonzero values in each column
nonzero_counts = (working_df != 0).sum()

# Identify columns with fewer than 100 nonzero values
low_value_cols = nonzero_counts[nonzero_counts < 100].index.tolist()

if low_value_cols:
    print(f"Warning: The following columns contain fewer than 100 nonzero values and may require review: {low_value_cols}")
    print("Rejecting file due to insufficient data.")

    # Move the file to the error directory
    final_path = os.path.join(base_directories["error_directory"], file_name)
    print(f"Moving {file_path} to the error directory {final_path}...")
    safe_move(file_path, final_path)
    now = time.time()
    os.utime(final_path, (now, now))
    sys.exit(1)

print("Task 1 applies one unified channel-combination filter over self, same-strip, same-plane, and any-channel relations.")

_prof["s_filter_uncal_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
if time_window_filtering:
    print("Task 1 coincidence-window filtering disabled; strip-coupled timing cleanup now belongs to Task 2.")
_prof["s_time_window_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()

# ─────────────────────────────────────────────────────────────────────────────
# 64×64 lower-triangular channel T/Q matrix, one figure per plane pair
# Variables: [T_F, T_B, Q_F, Q_B] × 4 planes × 4 strips = 64 variables
# Diagonal: histogram; lower triangle: scatter; upper triangle: hidden.
# ─────────────────────────────────────────────────────────────────────────────
channel_noise_limit_learning_removed_values = _empty_task1_removed_channel_values_frame()
task1_learned_channel_noise_limits: dict[tuple[int, str, int], float] = {}
task1_effective_channel_q_left_limits: dict[tuple[int, str, int], float] = {}
_plot_channel_tq_matrix_by_planepair = task1_plot_enabled("channel_tq_matrix_by_planepair")
_plot_channel_combination_filter_by_tt = task1_plot_enabled("channel_combination_filter_by_tt")
channel_plot_original_df = pd.DataFrame()
channel_plot_precombination_df = pd.DataFrame()
channel_plot_final_df = pd.DataFrame()
channel_combination_hist_original = pd.DataFrame()
channel_combination_hist_before = pd.DataFrame()
channel_combination_hist_after_preview = pd.DataFrame()
if _plot_channel_tq_matrix_by_planepair or _plot_channel_combination_filter_by_tt:
    (
        channel_plot_original_df,
        channel_plot_precombination_df,
        channel_plot_final_df,
    ) = build_task1_channel_filter_stage_frames(
        working_df,
        original_reference_df=None,
        relation_limits_by_type=task1_channel_combination_limits_by_relation,
    )
if _plot_channel_combination_filter_by_tt:
    channel_combination_hist_original = collect_task1_channel_combination_histogram_payload(
        channel_plot_original_df,
        _task1_filter_tt_series(channel_plot_original_df),
    )
    channel_combination_hist_before = collect_task1_channel_combination_histogram_payload(
        channel_plot_precombination_df,
        _task1_filter_tt_series(channel_plot_precombination_df),
    )
    channel_combination_hist_after_preview = collect_task1_channel_combination_histogram_payload(
        channel_plot_final_df,
        _task1_filter_tt_series(channel_plot_precombination_df),
    )

if _plot_channel_tq_matrix_by_planepair:
    # Sectorized approach per CHANNEL (plane,strip,side): 2×2 (Q,T) matrix per channel-pair
    # Improve image definition and aesthetics: larger DPI, larger markers, color per plane-pair,
    # histogram drawn with histtype='step' and log-counts.
    _MM1 = 1000
    raw_noise_limit_line_kwargs = {
        "color": "red",
        "linestyle": "--",
        "linewidth": 1.4,
        "alpha": 0.9,
        "zorder": 6,
    }
    _self_relation_limits = task1_channel_combination_limits_by_relation.get("self", {})
    raw_var_limits: dict[int, tuple[float | None, float | None]] = {
        0: (
            float(_self_relation_limits.get("q_sum_left", channel_combination_q_sum_left)),
            float(_self_relation_limits.get("q_sum_right", channel_combination_q_sum_right)),
        ),
        1: (
            float(_self_relation_limits.get("t_sum_left", channel_combination_t_sum_left)),
            float(_self_relation_limits.get("t_sum_right", channel_combination_t_sum_right)),
        ),
    }

    pair_list = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    CHANNEL_PAIR_FIGSIZE = (7, 7)
    channel_pair_min_events = int(config.get("channel_pair_min_events", 10))
    channel_pair_sector_payloads: list[dict[str, object]] = []
    channel_pair_candidate_counts: list[tuple[int, int, int, int, int, int]] = []

    for _pi, _pj in pair_list:
        if "raw_tt" in channel_plot_original_df.columns:
            _tts_all = channel_plot_original_df["raw_tt"].apply(normalize_tt_label)
            _pmask_all = _tts_all.str.contains(str(_pi)) & _tts_all.str.contains(str(_pj))
            _df_pp_all = channel_plot_original_df.loc[_pmask_all]
        else:
            _df_pp_all = channel_plot_original_df
        # Previously we gated on the raw number of events for the plane-pair
        # directly from `working_df`. That counted many rows that would later
        # produce blank plots because their channel values were zero. Instead,
        # require that the per-pair threshold is applied to the number of
        # events that actually have non-zero (Q,T) data for at least one
        # channel in the plane-pair. Build the retained preview frame first
        # and then compute the effective count of non-zero channel pairs.
        if "raw_tt" in channel_plot_final_df.columns:
            _tts_retained = channel_plot_final_df["raw_tt"].apply(normalize_tt_label)
            _pmask_retained = _tts_retained.str.contains(str(_pi)) & _tts_retained.str.contains(str(_pj))
            _df_pp1 = channel_plot_final_df.loc[_pmask_retained]
        else:
            _df_pp1 = channel_plot_final_df
        if "raw_tt" in channel_plot_precombination_df.columns:
            _tts_precombination = channel_plot_precombination_df["raw_tt"].apply(normalize_tt_label)
            _pmask_precombination = _tts_precombination.str.contains(str(_pi)) & _tts_precombination.str.contains(str(_pj))
            _df_precombination_pp1 = channel_plot_precombination_df.loc[_pmask_precombination]
        else:
            _df_precombination_pp1 = channel_plot_precombination_df.iloc[0:0].copy()

        # Compute how many events actually contain at least one channel with
        # both Q and T non-zero for this plane-pair. Apply the minimum-event
        # gating to that effective count to avoid producing blank figures.
        channels_check = [(p, s, sd) for p in (_pi, _pj) for s in range(1, 5) for sd in ("F", "B")]
        if not _df_pp_all.empty or not _df_precombination_pp1.empty or not _df_pp1.empty:
            # Vectorized per-column checks are faster than row-wise Python loops.
            any_channel_original_both_nonzero = (
                np.zeros(len(_df_pp_all), dtype=bool)
                if not _df_pp_all.empty
                else np.zeros(0, dtype=bool)
            )
            any_channel_precombination_both_nonzero = (
                np.zeros(len(_df_precombination_pp1), dtype=bool)
                if not _df_precombination_pp1.empty
                else np.zeros(0, dtype=bool)
            )
            any_channel_both_nonzero = np.zeros(len(_df_pp1), dtype=bool) if not _df_pp1.empty else np.zeros(0, dtype=bool)
            for (pl, ss, sd) in channels_check:
                qcol = f"Q{pl}_{sd}_{ss}"
                tcol = f"T{pl}_{sd}_{ss}"
                if qcol in _df_pp_all.columns and tcol in _df_pp_all.columns:
                    q_original_nonzero = _df_pp_all[qcol].fillna(0).to_numpy(dtype=float) != 0
                    t_original_nonzero = _df_pp_all[tcol].fillna(0).to_numpy(dtype=float) != 0
                    any_channel_original_both_nonzero |= (q_original_nonzero & t_original_nonzero)
                if qcol in _df_precombination_pp1.columns and tcol in _df_precombination_pp1.columns:
                    q_precombination_nonzero = _df_precombination_pp1[qcol].fillna(0).to_numpy(dtype=float) != 0
                    t_precombination_nonzero = _df_precombination_pp1[tcol].fillna(0).to_numpy(dtype=float) != 0
                    any_channel_precombination_both_nonzero |= (q_precombination_nonzero & t_precombination_nonzero)
                if qcol in _df_pp1.columns and tcol in _df_pp1.columns:
                    q_nonzero = _df_pp1[qcol].fillna(0).to_numpy(dtype=float) != 0
                    t_nonzero = _df_pp1[tcol].fillna(0).to_numpy(dtype=float) != 0
                    any_channel_both_nonzero |= (q_nonzero & t_nonzero)
            effective_events = max(
                int(any_channel_original_both_nonzero.sum()),
                int(any_channel_precombination_both_nonzero.sum()),
                int(any_channel_both_nonzero.sum()),
            )
        else:
            effective_events = 0

        if effective_events < channel_pair_min_events:
            # Nothing worth plotting for this plane-pair after non-zero filtering
            continue

        if not _df_pp1.empty:
            sampled_index = _df_pp1.sample(n=min(len(_df_pp1), _MM1), random_state=42).index
        elif not _df_precombination_pp1.empty:
            sampled_index = _df_precombination_pp1.sample(n=min(len(_df_precombination_pp1), _MM1), random_state=42).index
        elif not _df_pp_all.empty:
            sampled_index = _df_pp_all.sample(n=min(len(_df_pp_all), _MM1), random_state=42).index
        else:
            sampled_index = _df_pp1.index
        _df_s1 = _df_pp1.reindex(sampled_index).copy()
        _df_precombination_s1 = _df_precombination_pp1.reindex(sampled_index).copy()
        _df_original_s1 = _df_pp_all.reindex(sampled_index).copy()

        # Map event plane-combination (normalized raw_tt) to colors for per-point coloring
        if "raw_tt" in _df_s1.columns:
            _row_tt = _df_s1["raw_tt"].apply(normalize_tt_label).astype(str)
        else:
            _row_tt = pd.Series([f"{_pi}{_pj}"] * len(_df_s1), index=_df_s1.index)
        if "raw_tt" in _df_precombination_s1.columns:
            _precombination_row_tt = _df_precombination_s1["raw_tt"].apply(normalize_tt_label).astype(str)
        else:
            _precombination_row_tt = pd.Series([f"{_pi}{_pj}"] * len(_df_precombination_s1), index=_df_precombination_s1.index)
        unique_tts = sorted(set(_row_tt.unique()).union(set(_precombination_row_tt.unique())))
        if not unique_tts:
            unique_tts = [f"{_pi}{_pj}"]
        tt_color_map: dict[str, tuple[float, float, float, float]] = {
            tt_label: get_tt_color(tt_label) for tt_label in unique_tts
        }
        # Per-row color array aligned with _df_s1
        _row_colors = np.array([tt_color_map[str(lbl)] for lbl in _row_tt], dtype=object)
        _precombination_row_colors = np.array([tt_color_map[str(lbl)] for lbl in _precombination_row_tt], dtype=object)

        # Build channels including side: ('plane', strip, 'F'/'B')
        channels = [(p, s, sd) for p in (_pi, _pj) for s in range(1, 5) for sd in ("F", "B")]

        # Precompute retained and pre-zeroing reference arrays for each (channel_idx, var_idx)
        _chan_var_arr: dict[tuple[int, int], np.ndarray] = {}
        _precombination_chan_var_arr: dict[tuple[int, int], np.ndarray] = {}
        _original_chan_var_arr: dict[tuple[int, int], np.ndarray] = {}
        for ch_idx, (pl, ss, sd) in enumerate(channels):
            q_col = f"Q{pl}_{sd}_{ss}"
            t_col = f"T{pl}_{sd}_{ss}"
            for var_idx, col in enumerate((q_col, t_col)):
                if col in _df_s1.columns:
                    arr = _df_s1[col].fillna(0).to_numpy(dtype=float)
                    _chan_var_arr[(ch_idx, var_idx)] = arr
                else:
                    _chan_var_arr[(ch_idx, var_idx)] = np.zeros(len(_df_s1), dtype=float)
                if col in _df_precombination_s1.columns:
                    arr_precombination = _df_precombination_s1[col].fillna(0).to_numpy(dtype=float)
                    _precombination_chan_var_arr[(ch_idx, var_idx)] = arr_precombination
                else:
                    _precombination_chan_var_arr[(ch_idx, var_idx)] = np.zeros(len(_df_precombination_s1), dtype=float)
                if col in _df_original_s1.columns:
                    arr_original = _df_original_s1[col].fillna(0).to_numpy(dtype=float)
                    _original_chan_var_arr[(ch_idx, var_idx)] = arr_original
                else:
                    _original_chan_var_arr[(ch_idx, var_idx)] = np.zeros(len(_df_original_s1), dtype=float)

        # Compute per-variable ranges from the full retained+removed raw values and
        # ensure they always include the active configured Q/T limits.
        col_ranges: dict[int, tuple[float, float]] = {}
        for var_idx in (0, 1):
            retained_vals = [
                _chan_var_arr.get((ch, var_idx), np.zeros(0, dtype=float))
                for ch in range(len(channels))
            ]
            precombination_vals = [
                _precombination_chan_var_arr.get((ch, var_idx), np.zeros(0, dtype=float))
                for ch in range(len(channels))
            ]
            original_vals = [
                _original_chan_var_arr.get((ch, var_idx), np.zeros(0, dtype=float))
                for ch in range(len(channels))
            ]
            all_vals = np.concatenate([*retained_vals, *precombination_vals, *original_vals])
            nonzero = all_vals[all_vals != 0]
            col_ranges[var_idx] = _task1_capped_plot_range(
                nonzero if nonzero.size else all_vals,
                raw_var_limits.get(var_idx, (None, None)),
            )

        pair_plot_candidates: list[tuple[int, int, int, int]] = []
        for ai in range(len(channels)):
            for bj in range(ai, len(channels)):
                ai_q = _chan_var_arr.get((ai, 0), np.zeros(0, dtype=float))
                ai_t = _chan_var_arr.get((ai, 1), np.zeros(0, dtype=float))
                bj_q = _chan_var_arr.get((bj, 0), np.zeros(0, dtype=float))
                bj_t = _chan_var_arr.get((bj, 1), np.zeros(0, dtype=float))
                ai_q_precombination = _precombination_chan_var_arr.get((ai, 0), np.zeros(0, dtype=float))
                ai_t_precombination = _precombination_chan_var_arr.get((ai, 1), np.zeros(0, dtype=float))
                bj_q_precombination = _precombination_chan_var_arr.get((bj, 0), np.zeros(0, dtype=float))
                bj_t_precombination = _precombination_chan_var_arr.get((bj, 1), np.zeros(0, dtype=float))
                ai_q_original = _original_chan_var_arr.get((ai, 0), np.zeros(0, dtype=float))
                ai_t_original = _original_chan_var_arr.get((ai, 1), np.zeros(0, dtype=float))
                bj_q_original = _original_chan_var_arr.get((bj, 0), np.zeros(0, dtype=float))
                bj_t_original = _original_chan_var_arr.get((bj, 1), np.zeros(0, dtype=float))
                figure_effective_events = max(
                    int((((ai_q_original != 0) & (ai_t_original != 0)) | ((bj_q_original != 0) & (bj_t_original != 0))).sum()),
                    int((((ai_q_precombination != 0) & (ai_t_precombination != 0)) | ((bj_q_precombination != 0) & (bj_t_precombination != 0))).sum()),
                    int((((ai_q != 0) & (ai_t != 0)) | ((bj_q != 0) & (bj_t != 0))).sum()),
                )
                combination_removed_events = int(
                    (
                        (
                            ((ai_q_precombination != 0) & (ai_t_precombination != 0))
                            | ((bj_q_precombination != 0) & (bj_t_precombination != 0))
                        )
                        & ~(
                            ((ai_q != 0) & (ai_t != 0))
                            | ((bj_q != 0) & (bj_t != 0))
                        )
                    ).sum()
                )
                if figure_effective_events < channel_pair_min_events:
                    continue
                pair_plot_candidates.append((ai, bj, figure_effective_events, combination_removed_events))
                channel_pair_candidate_counts.append(
                    (combination_removed_events, figure_effective_events, _pi, _pj, ai, bj)
                )

        if not pair_plot_candidates:
            continue

        channel_pair_sector_payloads.append(
            {
                "plane_a": _pi,
                "plane_b": _pj,
                "plane_pair_event_count": len(_df_pp_all),
                "pair_plot_candidates": pair_plot_candidates,
                "row_tt": _row_tt,
                "precombination_row_tt": _precombination_row_tt,
                "unique_tts": unique_tts,
                "tt_color_map": tt_color_map,
                "row_colors": _row_colors,
                "precombination_row_colors": _precombination_row_colors,
                "channels": channels,
                "chan_var_arr": _chan_var_arr,
                "precombination_chan_var_arr": _precombination_chan_var_arr,
                "original_chan_var_arr": _original_chan_var_arr,
                "col_ranges": col_ranges,
            }
        )

    selected_channel_pair_keys: set[tuple[int, int, int, int]] | None = None
    selected_channel_pair_order: list[tuple[int, int, int, int]] | None = None
    if channel_pair_plot_top_n is not None:
        ranked_channel_pair_candidates = sorted(
            channel_pair_candidate_counts,
            key=lambda item: (-item[0], -item[1], item[2], item[3], item[4], item[5]),
        )
        selected_channel_pair_order = [
            (plane_a, plane_b, ai, bj)
            for _, _, plane_a, plane_b, ai, bj in ranked_channel_pair_candidates[:channel_pair_plot_top_n]
        ]
        selected_channel_pair_keys = {
            (plane_a, plane_b, ai, bj)
            for plane_a, plane_b, ai, bj in selected_channel_pair_order
        }
        print(
            "Task 1 channel T/Q matrix plot cap: "
            f"plotting top {len(selected_channel_pair_keys)} of {len(channel_pair_candidate_counts)} "
            "eligible channel-pair figures ranked by combination-filter removals."
        )
    else:
        print(
            "Task 1 channel T/Q matrix plot cap: plotting all "
            f"{len(channel_pair_candidate_counts)} eligible channel-pair figures."
        )

    sector_payload_lookup = {
        (int(sector_payload["plane_a"]), int(sector_payload["plane_b"])): sector_payload
        for sector_payload in channel_pair_sector_payloads
    }

    if selected_channel_pair_order is not None:
        pair_iteration_order: list[tuple[int, int, int, int, int, int]] = []
        for _pi, _pj, ai, bj in selected_channel_pair_order:
            sector_payload = sector_payload_lookup.get((_pi, _pj))
            if sector_payload is None:
                continue
            for candidate_ai, candidate_bj, figure_effective_events, combination_removed_events in sector_payload["pair_plot_candidates"]:
                if candidate_ai == ai and candidate_bj == bj:
                    pair_iteration_order.append(
                        (_pi, _pj, candidate_ai, candidate_bj, figure_effective_events, combination_removed_events)
                    )
                    break
    else:
        pair_iteration_order = []
        for sector_payload in channel_pair_sector_payloads:
            _pi = int(sector_payload["plane_a"])
            _pj = int(sector_payload["plane_b"])
            for ai, bj, figure_effective_events, combination_removed_events in sector_payload["pair_plot_candidates"]:
                pair_iteration_order.append(
                    (_pi, _pj, ai, bj, figure_effective_events, combination_removed_events)
                )

    for _pi, _pj, ai, bj, figure_effective_events, combination_removed_events in pair_iteration_order:
        sector_payload = sector_payload_lookup.get((_pi, _pj))
        if sector_payload is None:
            continue
        plane_pair_event_count = int(sector_payload["plane_pair_event_count"])
        _row_tt = sector_payload["row_tt"]
        _precombination_row_tt = sector_payload["precombination_row_tt"]
        unique_tts = list(sector_payload["unique_tts"])
        tt_color_map = dict(sector_payload["tt_color_map"])
        _row_colors = sector_payload["row_colors"]
        _precombination_row_colors = sector_payload["precombination_row_colors"]
        channels = list(sector_payload["channels"])
        _chan_var_arr = dict(sector_payload["chan_var_arr"])
        _precombination_chan_var_arr = dict(sector_payload["precombination_chan_var_arr"])
        _original_chan_var_arr = dict(sector_payload["original_chan_var_arr"])
        col_ranges = dict(sector_payload["col_ranges"])

        if (
            selected_channel_pair_keys is not None
            and (_pi, _pj, ai, bj) not in selected_channel_pair_keys
        ):
            continue

        ai_channel_key = (channels[ai][0], channels[ai][2], channels[ai][1])
        bj_channel_key = (channels[bj][0], channels[bj][2], channels[bj][1])
        ai_noise_limit = task1_effective_channel_q_left_limits.get(ai_channel_key)
        bj_noise_limit = task1_effective_channel_q_left_limits.get(bj_channel_key)
        _fig, _axes = plt.subplots(2, 2, figsize=CHANNEL_PAIR_FIGSIZE, squeeze=False, sharex='col', sharey='row')
        same_channel = ai == bj
        for r in range(2):
            for c in range(2):
                ax = _axes[r][c]
                if same_channel and c > r:
                    ax.set_visible(False)
                    continue
                for sp in ax.spines.values():
                    sp.set_linewidth(0.4)

                col_x = _chan_var_arr.get((ai, c), np.zeros(0, dtype=float))
                col_y = _chan_var_arr.get((bj, r), np.zeros(0, dtype=float))
                precombination_col_x = np.clip(
                    _precombination_chan_var_arr.get((ai, c), np.zeros(0, dtype=float)),
                    *col_ranges[c],
                )
                precombination_col_y = np.clip(
                    _precombination_chan_var_arr.get((bj, r), np.zeros(0, dtype=float)),
                    *col_ranges[r],
                )
                original_col_x = np.clip(
                    _original_chan_var_arr.get((ai, c), np.zeros(0, dtype=float)),
                    *col_ranges[c],
                )
                original_col_y = np.clip(
                    _original_chan_var_arr.get((bj, r), np.zeros(0, dtype=float)),
                    *col_ranges[r],
                )

                if same_channel and c == r:
                    vals_all = col_x[col_x != 0]
                    var_name = 'Q' if c == 0 else 'T'
                    prefilter_removed_vals_all = original_col_x[(original_col_x != 0) & (precombination_col_x == 0)]
                    combination_removed_vals_all = precombination_col_x[(precombination_col_x != 0) & (col_x == 0)]
                    if vals_all.size > 1 or prefilter_removed_vals_all.size > 0 or combination_removed_vals_all.size > 0:
                        if prefilter_removed_vals_all.size > 0:
                            hist_kwargs = dict(
                                bins=30,
                                histtype='step',
                                color='lightgrey',
                                linewidth=1.4,
                                linestyle='--',
                                log=True,
                                alpha=0.65,
                                label='Removed before combination',
                            )
                            if var_name == 'T':
                                hist_kwargs["orientation"] = 'horizontal'
                            ax.hist(prefilter_removed_vals_all, **hist_kwargs)
                        for tt_label in unique_tts:
                            precombination_mask_tt = (_precombination_row_tt.values == tt_label)
                            removed_vals_tt = precombination_col_x[precombination_mask_tt]
                            removed_vals_tt = removed_vals_tt[(removed_vals_tt != 0) & (col_x[precombination_mask_tt] == 0)]
                            if removed_vals_tt.size > 1:
                                combo_hist_kwargs = dict(
                                    bins=30,
                                    histtype='step',
                                    color=tt_color_map[tt_label],
                                    linewidth=1.0,
                                    linestyle='--',
                                    log=True,
                                    alpha=0.7,
                                    label=f"TT={tt_label} removed" if len(unique_tts) > 1 else None,
                                )
                                if var_name == 'T':
                                    combo_hist_kwargs["orientation"] = 'horizontal'
                                ax.hist(removed_vals_tt, **combo_hist_kwargs)
                            mask_tt = (_row_tt.values == tt_label)
                            vals_tt = col_x[mask_tt]
                            vals_tt = vals_tt[vals_tt != 0]
                            if vals_tt.size > 1:
                                if var_name == 'T':
                                    ax.hist(
                                        vals_tt,
                                        bins=30,
                                        histtype='step',
                                        color=tt_color_map[tt_label],
                                        linewidth=1.2,
                                        log=True,
                                        orientation='horizontal',
                                        label=f"TT={tt_label}" if len(unique_tts) > 1 else None,
                                    )
                                else:
                                    ax.hist(
                                        vals_tt,
                                        bins=30,
                                        histtype='step',
                                        color=tt_color_map[tt_label],
                                        linewidth=1.2,
                                        log=True,
                                        label=f"TT={tt_label}" if len(unique_tts) > 1 else None,
                                    )
                        if len(unique_tts) > 1 or prefilter_removed_vals_all.size > 0 or combination_removed_vals_all.size > 0:
                            ax.legend(fontsize=6)
                        if var_name == 'T':
                            ax.set_xlabel('Counts (log)', fontsize=7)
                            ax.set_ylabel('T (ns)', fontsize=7)
                            ax.set_ylim(col_ranges[c])
                            try:
                                tl_lo = float(raw_var_limits[1][0])
                                tl_hi = float(raw_var_limits[1][1])
                                ax.axhline(tl_lo, color='lightgrey', linestyle='--', linewidth=0.8)
                                ax.axhline(tl_hi, color='lightgrey', linestyle='--', linewidth=0.8)
                            except Exception:
                                pass
                        else:
                            ax.set_xlabel('Q (ns)', fontsize=7)
                            ax.set_ylabel('Counts (log)', fontsize=7)
                            ax.set_xlim(col_ranges[c])
                            try:
                                ql_lo = float(raw_var_limits[0][0])
                                ql_hi = float(raw_var_limits[0][1])
                                ax.axvline(ql_lo, color='lightgrey', linestyle='--', linewidth=0.8)
                                ax.axvline(ql_hi, color='lightgrey', linestyle='--', linewidth=0.8)
                            except Exception:
                                pass
                            if ai_noise_limit is not None and np.isfinite(ai_noise_limit):
                                ax.axvline(
                                    float(ai_noise_limit),
                                    **raw_noise_limit_line_kwargs,
                                )
                else:
                    prefilter_removed_mask = (
                        (original_col_x != 0)
                        & (original_col_y != 0)
                        & ~((precombination_col_x != 0) & (precombination_col_y != 0))
                    )
                    if np.any(prefilter_removed_mask):
                        ax.scatter(
                            original_col_x[prefilter_removed_mask],
                            original_col_y[prefilter_removed_mask],
                            s=4,
                            alpha=0.08,
                            linewidths=0,
                            color='grey',
                            edgecolors='none',
                            zorder=1,
                        )
                    if col_x.size and col_y.size:
                        mask = (col_x != 0) & (col_y != 0)
                        if np.any(mask):
                            ax.scatter(
                                col_x[mask],
                                col_y[mask],
                                s=12,
                                alpha=0.75,
                                linewidths=0,
                                c=_row_colors[mask].tolist(),
                                edgecolors='none',
                            )
                    if precombination_col_x.size and precombination_col_y.size:
                        removed_mask = (
                            (precombination_col_x != 0)
                            & (precombination_col_y != 0)
                            & ~((col_x != 0) & (col_y != 0))
                        )
                        if np.any(removed_mask):
                            ax.scatter(
                                precombination_col_x[removed_mask],
                                precombination_col_y[removed_mask],
                                s=removed_marker_size,
                                marker=removed_marker,
                                alpha=removed_marker_alpha,
                                linewidths=1.0,
                                c=_precombination_row_colors[removed_mask].tolist(),
                                zorder=3,
                            )
                    ax.set_xlim(col_ranges[c])
                    ax.set_ylim(col_ranges[r])
                    try:
                        ql_lo = float(raw_var_limits[0][0])
                        ql_hi = float(raw_var_limits[0][1])
                        ax.axvline(ql_lo, color='lightgrey', linestyle='--', linewidth=0.8)
                        ax.axvline(ql_hi, color='lightgrey', linestyle='--', linewidth=0.8)
                    except Exception:
                        pass
                    if c == 0 and ai_noise_limit is not None and np.isfinite(ai_noise_limit):
                        ax.axvline(
                            float(ai_noise_limit),
                            **raw_noise_limit_line_kwargs,
                        )
                    try:
                        tl_lo = float(raw_var_limits[1][0])
                        tl_hi = float(raw_var_limits[1][1])
                        ax.axhline(tl_lo, color='lightgrey', linestyle='--', linewidth=0.8)
                        ax.axhline(tl_hi, color='lightgrey', linestyle='--', linewidth=0.8)
                    except Exception:
                        pass
                    if r == 0 and bj_noise_limit is not None and np.isfinite(bj_noise_limit):
                        ax.axhline(
                            float(bj_noise_limit),
                            **raw_noise_limit_line_kwargs,
                        )
                    if r == 1:
                        ax.set_xlabel("Q" if c == 0 else "T", fontsize=7)
                    else:
                        ax.set_xlabel("")
                    if c == 0:
                        ax.set_ylabel("Q" if r == 0 else "T", fontsize=7)
                    else:
                        ax.set_ylabel("")

                    if ax.get_xscale() == "linear":
                        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, min_n_ticks=3))
                        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                    if ax.get_yscale() == "linear":
                        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, min_n_ticks=3))
                        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                    ax.tick_params(
                        axis="x",
                        which="both",
                        labelsize=6,
                        labelbottom=True,
                        bottom=True,
                    )
                    ax.tick_params(
                        axis="y",
                        which="both",
                        labelsize=6,
                        labelleft=True,
                        left=True,
                    )
                    for _tick_label in ax.get_xticklabels():
                        _tick_label.set_visible(True)
                    for _tick_label in ax.get_yticklabels():
                        _tick_label.set_visible(True)

        plt.subplots_adjust(wspace=0.08, hspace=0.08, left=0.08, right=0.98, top=0.95, bottom=0.08)
        a_pl, a_strip, a_side = channels[ai]
        b_pl, b_strip, b_side = channels[bj]
        a_addr = f"P{a_pl}S{a_strip}{a_side}"
        b_addr = f"P{b_pl}S{b_strip}{b_side}"
        _fig.suptitle(
            f"{a_addr} vs {b_addr} — Channel Sector: P{_pi}×P{_pj} "
            f"[{plane_pair_event_count} events, combo-removed={combination_removed_events}] · mingo0{station}",
            fontsize=9,
        )

        if save_plots:
            fname = f"{fig_idx:03d}_channel_pair_P{_pi}P{_pj}_ch{ai+1}_ch{bj+1}.png"
            fig_idx += 1
            fpath = os.path.join(base_directories["figure_directory"], fname)
            plot_list.append(fpath)
            save_plot_figure(fpath, fig=_fig, format="png", dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(_fig)

# Path to save the cleaned dataframe
# Create output directory if it does not exist.
os.makedirs(f"{output_directory}", exist_ok=True)
OUT_PATH = f"{output_directory}/cleaned_{basename_no_ext}.parquet"
KEY = "df"  # HDF5 key name

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# --- Example: your cleaned DataFrame is called working_df ---
# (Here, you would have your data cleaning code before saving)
# working_df = ...

# Task 1 now applies a single unified channel-combination filter. The legacy
# per-channel mismatch and learned/manual low-charge stages remain available in
# the codebase for backwards inspection, but they are inactive in the live path.
task1_channel_pairs = collect_task1_channel_qt_map(working_df)
row_mismatch_mask = pd.Series(False, index=working_df.index, dtype=bool)
channel_qt_rows_affected = 0
channel_qt_values_zeroed = 0
channel_qt_q_only_rows = 0
channel_qt_t_only_rows = 0
channel_qt_front_rows = 0
channel_qt_back_rows = 0
channel_qt_mismatch_channels = 0
channel_qt_removed_clean_tt = 0
channel_qt_retained_final = 0
global_variables["plane_combination_noise_limit_learning_removed_values"] = 0
global_variables["plane_combination_noise_limit_learning_zeroed_values"] = 0
global_variables["plane_combination_noise_limit_learning_channels"] = 0
global_variables["plane_combination_noise_limit_mode"] = "inactive"
global_variables["plane_combination_noise_limit_configured_channels"] = int(
    len(manual_channel_noise_charge_limits)
)
global_variables["plane_combination_noise_limit_rows_affected"] = 0
global_variables["plane_combination_noise_limit_values_zeroed"] = 0
for channel_key in TASK1_CHANNEL_KEYS:
    global_variables[_task1_channel_noise_limit_metric_key(channel_key)] = ""

plane_combination_summary = apply_task1_plane_combination_filter(
    working_df,
    relation_limits_by_type=task1_channel_combination_limits_by_relation,
    snapshot_originals=snapshot_original_columns_once,
    removed_value_pass_label="plane_combination_filter",
)
working_df.loc[:, "task1_problematic_channel_count"] = (
    plane_combination_summary["selected_offender_count_by_row"]
    .reindex(working_df.index, fill_value=0)
    .astype(int)
    .to_numpy()
)
working_df.loc[:, "task1_problematic_channel_resolution_exact"] = (
    plane_combination_summary["resolution_exact_by_row"]
    .reindex(working_df.index, fill_value=True)
    .astype(bool)
    .to_numpy()
)
task1_problematic_channel_count_snapshot = working_df.loc[
    :,
    [col for col in ("datetime", "task1_problematic_channel_count") if col in working_df.columns],
].copy()
task1_problematic_channel_count_counts = (
    pd.to_numeric(
        task1_problematic_channel_count_snapshot.get(
            "task1_problematic_channel_count",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
    .value_counts()
    .to_dict()
)
record_activity_metric(
    "plane_combination_filter_rows_affected_pct",
    plane_combination_summary["rows_affected"],
    len(working_df) if len(working_df) else 0,
    label="rows with channel-combination failures",
)
record_activity_metric(
    "plane_combination_filter_values_zeroed_pct",
    plane_combination_summary["values_zeroed"],
    len(working_df) * (2 * plane_combination_summary["tracked_channel_count"])
    if (len(working_df) and plane_combination_summary["tracked_channel_count"])
    else 0,
    label="channel Q/T values zeroed by plane-combination filter",
)
for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES:
    valid_relation_observations = int(
        plane_combination_summary["valid_pair_observations_by_relation"].get(relation_type, 0)
    )
    failed_relation_observations = int(
        plane_combination_summary["failed_pair_any_by_relation"].get(relation_type, 0)
    )
    relation_failed_rows = int(
        plane_combination_summary["rows_with_relation_failures"].get(relation_type, 0)
    )
    record_activity_metric(
        f"plane_combination_filter_{relation_type}_relation_failed_pct",
        failed_relation_observations,
        valid_relation_observations if valid_relation_observations else 0,
        label=f"{relation_type} relation failures",
    )
    record_activity_metric(
        f"plane_combination_filter_rows_with_{relation_type}_relation_failures_pct",
        relation_failed_rows,
        len(working_df) if len(working_df) else 0,
        label=f"rows with {relation_type} relation failures",
    )
    record_activity_metric(
        f"plane_combination_filter_{relation_type}_q_sum_low_failed_pct",
        plane_combination_summary["failed_pair_q_sum_low_by_relation"].get(relation_type, 0),
        valid_relation_observations if valid_relation_observations else 0,
        label=f"{relation_type} q_sum low-bound failures",
    )
    record_activity_metric(
        f"plane_combination_filter_{relation_type}_q_sum_high_failed_pct",
        plane_combination_summary["failed_pair_q_sum_high_by_relation"].get(relation_type, 0),
        valid_relation_observations if valid_relation_observations else 0,
        label=f"{relation_type} q_sum high-bound failures",
    )
    record_activity_metric(
        f"plane_combination_filter_{relation_type}_t_sum_low_failed_pct",
        plane_combination_summary["failed_pair_t_sum_low_by_relation"].get(relation_type, 0),
        valid_relation_observations if valid_relation_observations else 0,
        label=f"{relation_type} t_sum low-bound failures",
    )
    record_activity_metric(
        f"plane_combination_filter_{relation_type}_t_sum_high_failed_pct",
        plane_combination_summary["failed_pair_t_sum_high_by_relation"].get(relation_type, 0),
        valid_relation_observations if valid_relation_observations else 0,
        label=f"{relation_type} t_sum high-bound failures",
    )
record_activity_metric(
    "plane_combination_filter_any_failed_pct",
    plane_combination_summary["failed_pair_any"],
    plane_combination_summary["valid_pair_observations"],
    label="failed channel plane combinations",
)
record_activity_metric(
    "plane_combination_filter_q_sum_failed_pct",
    plane_combination_summary["failed_pair_q_sum"],
    plane_combination_summary["valid_pair_observations"],
    label="Q_sum failed channel plane combinations",
)
record_activity_metric(
    "plane_combination_filter_q_sum_low_failed_pct",
    plane_combination_summary["failed_pair_q_sum_low"],
    plane_combination_summary["valid_pair_observations"],
    label="Q_sum low-bound failed channel plane combinations",
)
record_activity_metric(
    "plane_combination_filter_q_sum_high_failed_pct",
    plane_combination_summary["failed_pair_q_sum_high"],
    plane_combination_summary["valid_pair_observations"],
    label="Q_sum high-bound failed channel plane combinations",
)
record_activity_metric(
    "plane_combination_filter_q_dif_failed_pct",
    plane_combination_summary["failed_pair_q_dif"],
    plane_combination_summary["valid_pair_observations"],
    label="Q_dif failed channel plane combinations",
)
record_activity_metric(
    "plane_combination_filter_t_sum_failed_pct",
    plane_combination_summary["failed_pair_t_sum"],
    plane_combination_summary["valid_pair_observations"],
    label="T_sum failed channel plane combinations",
)
record_activity_metric(
    "plane_combination_filter_t_sum_low_failed_pct",
    plane_combination_summary["failed_pair_t_sum_low"],
    plane_combination_summary["valid_pair_observations"],
    label="T_sum low-bound failed channel plane combinations",
)
record_activity_metric(
    "plane_combination_filter_t_sum_high_failed_pct",
    plane_combination_summary["failed_pair_t_sum_high"],
    plane_combination_summary["valid_pair_observations"],
    label="T_sum high-bound failed channel plane combinations",
)
record_activity_metric(
    "plane_combination_filter_t_dif_failed_pct",
    plane_combination_summary["failed_pair_t_dif"],
    plane_combination_summary["valid_pair_observations"],
    label="T_dif failed channel plane combinations",
)
record_activity_metric(
    "plane_combination_filter_flagged_rows_pct",
    plane_combination_summary["flagged_rows"],
    len(working_df) if len(working_df) else 0,
    label="rows with at least one failed channel plane combination",
)
record_activity_metric(
    "plane_combination_filter_selected_offender_channels_pct",
    plane_combination_summary["selected_offender_channels"],
    len(working_df) * plane_combination_summary["tracked_channel_count"]
    if (len(working_df) and plane_combination_summary["tracked_channel_count"])
    else 0,
    label="selected offending channels",
)
record_activity_metric(
    "plane_combination_filter_multi_offender_rows_pct",
    plane_combination_summary["rows_with_multiple_offenders"],
    plane_combination_summary["flagged_rows"],
    label="flagged rows needing multiple offending channels",
)
global_variables["plane_combination_filter_flagged_rows"] = int(
    plane_combination_summary["flagged_rows"]
)
global_variables["plane_combination_filter_input_rows"] = int(
    plane_combination_summary.get("input_rows", len(working_df))
)
for relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES:
    global_variables[f"plane_combination_filter_{relation_type}_failed_observations"] = int(
        plane_combination_summary["failed_pair_any_by_relation"].get(relation_type, 0)
    )
    global_variables[f"plane_combination_filter_{relation_type}_q_sum_low_failed_observations"] = int(
        plane_combination_summary["failed_pair_q_sum_low_by_relation"].get(relation_type, 0)
    )
    global_variables[f"plane_combination_filter_{relation_type}_q_sum_high_failed_observations"] = int(
        plane_combination_summary["failed_pair_q_sum_high_by_relation"].get(relation_type, 0)
    )
    global_variables[f"plane_combination_filter_{relation_type}_t_sum_low_failed_observations"] = int(
        plane_combination_summary["failed_pair_t_sum_low_by_relation"].get(relation_type, 0)
    )
    global_variables[f"plane_combination_filter_{relation_type}_t_sum_high_failed_observations"] = int(
        plane_combination_summary["failed_pair_t_sum_high_by_relation"].get(relation_type, 0)
    )
    global_variables[f"plane_combination_filter_rows_with_{relation_type}_relation_failures"] = int(
        plane_combination_summary["rows_with_relation_failures"].get(relation_type, 0)
    )
global_variables["plane_combination_filter_q_sum_low_failed_observations"] = int(
    plane_combination_summary["failed_pair_q_sum_low"]
)
global_variables["plane_combination_filter_q_sum_high_failed_observations"] = int(
    plane_combination_summary["failed_pair_q_sum_high"]
)
global_variables["plane_combination_filter_t_sum_low_failed_observations"] = int(
    plane_combination_summary["failed_pair_t_sum_low"]
)
global_variables["plane_combination_filter_t_sum_high_failed_observations"] = int(
    plane_combination_summary["failed_pair_t_sum_high"]
)
global_variables["plane_combination_filter_selected_offender_channels"] = int(
    plane_combination_summary["selected_offender_channels"]
)
global_variables["plane_combination_filter_rows_with_multiple_offenders"] = int(
    plane_combination_summary["rows_with_multiple_offenders"]
)
global_variables["plane_combination_filter_max_failed_pairs_in_row"] = int(
    plane_combination_summary["max_failed_pairs_in_row"]
)
global_variables["plane_combination_filter_max_selected_offenders_in_row"] = int(
    plane_combination_summary["max_selected_offenders_in_row"]
)
for selected_count in TASK1_SELECTED_OFFENDER_CARDINALITY_VALUES:
    selected_rows = int(task1_problematic_channel_count_counts.get(selected_count, 0))
    global_variables[_task1_selected_offender_cardinality_metric_key(selected_count)] = selected_rows
for channel_key, offender_count in plane_combination_summary["selected_offender_counts"].items():
    global_variables[_task1_channel_offender_metric_key(channel_key)] = int(offender_count)
filter_metrics["plane_combination_filter_mean_failed_pairs_per_flagged_row"] = round(
    float(plane_combination_summary["failed_pair_any"]) / float(plane_combination_summary["flagged_rows"])
    if plane_combination_summary["flagged_rows"]
    else 0.0,
    4,
)
filter_metrics["plane_combination_filter_mean_selected_offenders_per_flagged_row"] = round(
    float(plane_combination_summary["selected_offender_channels"]) / float(plane_combination_summary["flagged_rows"])
    if plane_combination_summary["flagged_rows"]
    else 0.0,
    4,
)
filter_metrics["plane_combination_filter_max_failed_pairs_in_row"] = int(
    plane_combination_summary["max_failed_pairs_in_row"]
)
filter_metrics["plane_combination_filter_max_selected_offenders_in_row"] = int(
    plane_combination_summary["max_selected_offenders_in_row"]
)
task1_removed_channel_values_df = collect_task1_zeroed_channel_values(
    channel_pair_plot_reference_df if not channel_pair_plot_reference_df.empty else working_df,
    working_df,
    pass_label="final_zeroed_any_filter",
)
global_variables["task1_zeroed_channel_values"] = int(len(task1_removed_channel_values_df))
if _plot_channel_combination_filter_by_tt:
    channel_combination_limits_by_relation = {}
    for _relation_type in TASK1_CHANNEL_COMBINATION_RELATION_TYPES:
        _relation_limits = task1_channel_combination_limits_by_relation.get(_relation_type, {})
        channel_combination_limits_by_relation[_relation_type] = {
            "q_sum": (
                float(_relation_limits.get("q_sum_left", channel_combination_q_sum_left)),
                float(_relation_limits.get("q_sum_right", channel_combination_q_sum_right)),
            ),
            "q_dif": (
                -abs(float(_relation_limits.get("q_dif_threshold", channel_combination_q_dif_threshold))),
                abs(float(_relation_limits.get("q_dif_threshold", channel_combination_q_dif_threshold))),
            ),
            "t_sum": (
                float(_relation_limits.get("t_sum_left", channel_combination_t_sum_left)),
                float(_relation_limits.get("t_sum_right", channel_combination_t_sum_right)),
            ),
            "t_dif": (
                -abs(float(_relation_limits.get("t_dif_threshold", channel_combination_t_dif_threshold))),
                abs(float(_relation_limits.get("t_dif_threshold", channel_combination_t_dif_threshold))),
            ),
        }
    fig_idx = plot_task1_channel_combination_filter_by_tt(
        channel_combination_hist_original,
        channel_combination_hist_before,
        channel_combination_hist_after_preview,
        basename_no_ext,
        fig_idx,
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list,
        limits_by_relation=channel_combination_limits_by_relation,
        q_left_limits_by_channel=task1_effective_channel_q_left_limits,
        top_n_by_relation=channel_combination_plot_top_n_by_relation,
        include_same_strip_pairs=channel_combination_plot_include_same_strip_pairs,
    )

print(f"Original number of events in the dataframe: {original_number_of_events}")
# Compute clean trigger types after channel-only regularization, then drop the
# final single-plane / null combinations so Task 1 hands off only multi-plane events.
working_df = compute_tt(working_df, "clean_tt")
clean_tt_total = len(working_df)
clean_tt_mask = working_df["clean_tt"].notna() & (working_df["clean_tt"] >= 10)
channel_qt_removed_clean_tt = int((~clean_tt_mask & row_mismatch_mask.reindex(working_df.index, fill_value=False)).sum())
append_removed_rows_from_mask(working_df, ~clean_tt_mask)
working_df = working_df.loc[clean_tt_mask].copy()
record_filter_metric(
    "clean_tt_lt_10_rows_removed_pct",
    clean_tt_total - int(clean_tt_mask.sum()),
    clean_tt_total if clean_tt_total else 0,
)
channel_qt_retained_final = int(row_mismatch_mask.reindex(working_df.index, fill_value=False).sum())
record_activity_metric(
    "channel_qt_mismatch_rows_removed_clean_tt_of_mismatch_pct",
    channel_qt_removed_clean_tt,
    channel_qt_rows_affected,
    label="mismatch-touched rows later removed by clean_tt filter",
)
record_activity_metric(
    "channel_qt_mismatch_rows_retained_final_of_mismatch_pct",
    channel_qt_retained_final,
    channel_qt_rows_affected,
    label="mismatch-touched rows retained after Task 1",
)
working_df.loc[:, "raw_to_clean_tt"] = (
    pd.to_numeric(working_df["raw_tt"], errors="coerce").fillna(0).astype(int).astype(str)
    + "_"
    + pd.to_numeric(working_df["clean_tt"], errors="coerce").fillna(0).astype(int).astype(str)
)
refresh_global_count_metadata(
    working_df,
    ("raw_tt", "clean_tt", "raw_to_clean_tt"),
)

clean_channel_patterns = build_task1_channel_pattern_series(working_df)
store_pattern_rates(pattern_metadata, clean_channel_patterns, "clean_channel_pattern", working_df)

# Always compute clean-stage channel contagion metrics for metadata.
clean_channel_matrix_data = _compute_channel_conditional_matrix(working_df)
_store_channel_contagion_metrics_variant(
    "global_clean",
    clean_channel_matrix_data,
)
_store_channel_contagion_metrics_variant(
    "by_tt_clean",
    _compute_channel_conditional_matrix_for_tt(working_df, "clean_tt", 1234),
)
clean_channel_inputs = _compute_channel_contagion_inputs(working_df)
if clean_channel_matrix_data is not None and clean_channel_inputs is not None:
    clean_channel_labels, clean_channel_active = clean_channel_inputs
    _, clean_channel_matrix = clean_channel_matrix_data
    clean_channel_given_counts = {
        label: int(clean_channel_active[:, idx].sum())
        for idx, label in enumerate(clean_channel_labels)
    }
    store_activation_matrix_metadata(
        activation_metadata,
        "activation_channel_signal_to_signal_filtered",
        clean_channel_labels,
        clean_channel_matrix,
        clean_channel_given_counts,
        group_ids=[idx // 8 for idx in range(len(clean_channel_labels))],
    )
    clean_selected_tts, clean_tt_matrices, clean_tt_given_counts, clean_tt_event_counts = (
        compute_conditional_matrices_by_tt(
            pd.to_numeric(working_df["clean_tt"], errors="coerce").fillna(0).astype(int),
            clean_channel_labels,
            [clean_channel_active[:, idx] for idx in range(clean_channel_active.shape[1])],
        )
    )
    store_activation_matrices_by_tt_metadata(
        activation_metadata,
        "activation_channel_signal_to_signal_filtered_by_tt",
        clean_channel_labels,
        clean_channel_labels,
        clean_selected_tts,
        clean_tt_matrices,
        clean_tt_given_counts,
        clean_tt_event_counts,
    )
    del clean_channel_active
    del clean_channel_matrix
    del clean_selected_tts, clean_tt_matrices, clean_tt_given_counts, clean_tt_event_counts
    gc.collect()

# --- Post-filter (clean) channel contamination matrices ---
if task1_plot_enabled("channel_contamination_matrix_32"):
    fig_idx = _plot_channel_contamination_global(
        working_df, "clean", fig_idx, base_directories["figure_directory"],
        save_plots, show_plots, plot_list,
    )

if task1_plot_enabled("channel_contagion_by_tt"):
    fig_idx = _plot_channel_contagion_by_tt(
        working_df, "clean_tt", "clean", fig_idx, base_directories["figure_directory"],
        save_plots, show_plots, plot_list,
    )

# -----------------------------------------------------------------------------
# Create and save the PDF (deferred until all plots are generated) ------------
# -----------------------------------------------------------------------------
finalize_saved_plots_to_pdf()

# Final number of events
final_number_of_events = len(working_df)
print(f"Final number of events in the dataframe: {final_number_of_events}")

# ----------------------------------------------------------------------------------
# Count the number of non-zero entries per channel in the whole dataframe ----------
# ----------------------------------------------------------------------------------

# Count per each column the number of non-zero entries and save it in a column of
# global_variables called TX_F_Y_entries or TX_B_Y_entries

# Count for main dataframe (non-self-trigger)
for key, idx_range in column_indices.items():
    for i in range(1, len(idx_range) + 1):
        colname = f"{key}_{i}"
        count = (working_df[colname] != 0).sum()
        global_var_name = f"{key}_{i}_entries_final"
        global_variables[global_var_name] = count

# Data purity
data_purity = final_number_of_events / original_number_of_events * 100

# End of the execution time
_prof["s_finalize_s"] = round(time.perf_counter() - _t_sec, 2)
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
filter_metrics["valid_lines_in_binary_file_percentage"] = round(
    float(global_variables.get("valid_lines_in_binary_file_percentage", 0.0)),
    4,
)
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
    drop_field_predicate=lambda column_name: column_name not in FILTER_METADATA_ALLOWED_COLUMNS,
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
)
print(f"Metadata (deep_fiter) CSV updated at: {metadata_deep_fiter_csv_path}")

noise_control_events_per_second_meta = build_events_per_second_metadata(
    task1_problematic_channel_count_snapshot
)
try:
    noise_control_rate_denominator_seconds = int(
        noise_control_events_per_second_meta.get("events_per_second_total_seconds", 0) or 0
    )
except (TypeError, ValueError):
    noise_control_rate_denominator_seconds = 0
if noise_control_rate_denominator_seconds < 0:
    noise_control_rate_denominator_seconds = 0

noise_control_row = {
    "filename_base": filename_base,
    "execution_timestamp": execution_timestamp,
    "param_hash": param_hash_value,
    NOISE_CONTROL_RATE_DENOMINATOR_COLUMN: noise_control_rate_denominator_seconds,
}
for selected_count in TASK1_SELECTED_OFFENDER_CARDINALITY_VALUES:
    raw_count = task1_problematic_channel_count_counts.get(selected_count, 0)
    try:
        raw_count_value = float(raw_count)
    except (TypeError, ValueError):
        noise_control_row[_task1_selected_offender_cardinality_rate_metric_key(selected_count)] = ""
        noise_control_row[_task1_selected_offender_cardinality_percent_metric_key(selected_count)] = ""
        continue
    rate_value = (
        raw_count_value / noise_control_rate_denominator_seconds
        if noise_control_rate_denominator_seconds > 0
        else 0.0
    )
    noise_control_row[_task1_selected_offender_cardinality_rate_metric_key(selected_count)] = round(
        rate_value,
        6,
    )
    input_rows_value = float(len(task1_problematic_channel_count_snapshot.index))
    pct_value = 100.0 * raw_count_value / input_rows_value if input_rows_value > 0 else 0.0
    noise_control_row[_task1_selected_offender_cardinality_percent_metric_key(selected_count)] = round(
        pct_value,
        4,
    )

task1_noise_control_clean_tt = pd.to_numeric(
    working_df.get("clean_tt", pd.Series(index=working_df.index, dtype=float)),
    errors="coerce",
).fillna(0).astype(int)
task1_noise_control_selected_offenders = pd.to_numeric(
    working_df.get(
        "task1_problematic_channel_count",
        pd.Series(index=working_df.index, dtype=float),
    ),
    errors="coerce",
).fillna(0).astype(int)

for selected_count_threshold in task1_noise_control_efficiency_selected_offender_values:
    threshold_mask = task1_noise_control_selected_offenders <= selected_count_threshold
    threshold_clean_tt = task1_noise_control_clean_tt.loc[threshold_mask]
    four_plane_count = int((threshold_clean_tt == 1234).sum())
    for plane, missing_trigger in TASK1_THREE_TO_FOUR_MISSING_TRIGGER_BY_PLANE.items():
        metric_key = _task1_noise_control_efficiency_metric_key(
            plane,
            selected_count_threshold,
        )
        if four_plane_count <= 0:
            noise_control_row[metric_key] = ""
            continue
        missing_plane_count = int((threshold_clean_tt == missing_trigger).sum())
        noise_control_row[metric_key] = round(
            1.0 - (float(missing_plane_count) / float(four_plane_count)),
            6,
        )

metadata_noise_control_csv_path = save_metadata(
    csv_path_noise_control,
    noise_control_row,
    preferred_fieldnames=(
        "filename_base",
        "execution_timestamp",
        "param_hash",
        NOISE_CONTROL_RATE_DENOMINATOR_COLUMN,
        *TASK1_NOISE_CONTROL_METRIC_NAMES,
        *TASK1_NOISE_CONTROL_PERCENT_METRIC_NAMES,
        *_task1_noise_control_efficiency_metric_names(
            task1_noise_control_efficiency_selected_offender_values
        ),
    ),
)
print(f"Metadata (noise_control) CSV updated at: {metadata_noise_control_csv_path}")

 
# -------------------------------------------------------------------------------
# Execution metadata ------------------------------------------------------------
# -------------------------------------------------------------------------------

print("----------\nExecution metadata to be saved:")
print(f"Filename base: {filename_base}")
print(f"Execution timestamp: {execution_timestamp}")
print(f"Data purity percentage: {data_purity_percentage:.2f}%")
print(f"Total execution time: {total_execution_time_minutes:.2f} minutes")

for _metric_name, _metric_value in channel_contagion_metrics.items():
    global_variables[_metric_name] = _metric_value

metadata_execution_csv_path = save_metadata(
    csv_path,
    {
        "filename_base": filename_base,
        "execution_timestamp": execution_timestamp,
        "param_hash": param_hash_value,
        "data_purity_percentage": round(float(data_purity_percentage), 4),
        "total_execution_time_minutes": round(float(total_execution_time_minutes), 4),
        **channel_contagion_metrics,
    },
    preferred_fieldnames=(
        "filename_base",
        "execution_timestamp",
        "param_hash",
        "data_purity_percentage",
        "total_execution_time_minutes",
        *CHANNEL_CONTAGION_METADATA_FIELDS,
    ),
)
print(f"Metadata (execution) CSV updated at: {metadata_execution_csv_path}")

_prof["filename_base"] = filename_base
_prof["execution_timestamp"] = execution_timestamp
_prof["param_hash"] = param_hash_value
_prof["total_s"] = round(time.perf_counter() - _prof_t0, 2)
save_metadata(csv_path_profiling, _prof)

activation_metadata["filename_base"] = filename_base
activation_metadata["execution_timestamp"] = execution_timestamp
activation_metadata["param_hash"] = param_hash_value
metadata_activation_csv_path = save_metadata(csv_path_activation, activation_metadata)
print(f"Metadata (activation) CSV updated at: {metadata_activation_csv_path}")

pattern_metadata["filename_base"] = filename_base
pattern_metadata["execution_timestamp"] = execution_timestamp
pattern_metadata["param_hash"] = param_hash_value
metadata_pattern_csv_path = save_metadata(
    csv_path_pattern,
    pattern_metadata,
    preferred_fieldnames=("filename_base", "execution_timestamp", "param_hash"),
)
print(f"Metadata (pattern) CSV updated at: {metadata_pattern_csv_path}")

# -------------------------------------------------------------------------------
# Specific metadata ------------------------------------------------------------
# -------------------------------------------------------------------------------

global_variables.update(build_events_per_second_metadata(working_df))
ensure_global_count_keys(("raw_tt", "clean_tt", "raw_to_clean_tt"))
add_normalized_count_metadata(
    global_variables,
    global_variables.get("events_per_second_total_seconds", 0),
)
set_global_rate_from_tt_rates(
    global_variables,
    preferred_prefixes=("raw_tt", "clean_tt"),
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
trigger_type_prefixes = ("raw_tt", "clean_tt", "raw_to_clean_tt")
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
    stage_tt_columns=("raw_tt", "clean_tt"),
    denominator_seconds=trigger_type_variables["count_rate_denominator_seconds"],
)
metadata_trigger_type_csv_path = save_metadata(
    csv_path_trigger_type,
    trigger_type_variables,
    drop_field_predicate=lambda column_name: not is_trigger_type_file_column(
        column_name,
        trigger_type_prefixes,
    ),
)
print(f"Metadata (trigger_type) CSV updated at: {metadata_trigger_type_csv_path}")

print(f"Specific metadata keys to be saved: {len(global_variables)}")
# if VERBOSE:
#     print("----------\nAll global variables to be saved:")
#     for key, value in global_variables.items():
#         print(f"{key}: {value}")
#     print("----------\n")

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
        or column_name.startswith("raw_channel_pattern_")
        or column_name.startswith("clean_channel_pattern_")
        or is_trigger_type_metadata_column(column_name, trigger_type_prefixes)
        or column_name == "valid_lines_in_binary_file_percentage"
    ),
)
print(f"Metadata (specific) CSV updated at: {metadata_specific_csv_path}")

print(
    f"Writing cleaned parquet: rows={len(working_df)} cols={len(working_df.columns)} -> {OUT_PATH}"
)
# if VERBOSE:
#     print("Columns before saving cleaned parquet:")
#     for col in working_df.columns:
#         print(f" - {col}")

# Ensure datetime column is stored with a pandas datetime64 dtype to satisfy pyarrow
if "datetime" in working_df.columns:
    snapshot_original_columns_once(working_df, ["datetime"])
    working_df["datetime"] = pd.to_datetime(working_df["datetime"], errors="coerce")
    if working_df["datetime"].dtype == object:
        working_df["datetime"] = pd.to_datetime(
            working_df["datetime"].astype(str), errors="coerce"
        )

# Persist simulated parameter hash as a constant column (string) for traceability.
if "param_hash" not in working_df.columns:
    working_df["param_hash"] = str(simulated_param_hash) if simulated_param_hash else ""

channel_removed_values_output_directory = base_directories["removed_channel_values_directory"]
os.makedirs(channel_removed_values_output_directory, exist_ok=True)
channel_removed_values_base = os.path.join(
    channel_removed_values_output_directory,
    f"removed_channel_values_{basename_no_ext}",
)
task1_removed_channel_values_df = task1_removed_channel_values_df.copy()
task1_removed_channel_values_df["filename_base"] = filename_base
task1_removed_channel_values_df.to_parquet(
    f"{channel_removed_values_base}.parquet",
    engine="pyarrow",
    compression="zstd",
    index=False,
)
task1_removed_channel_values_df.to_csv(
    f"{channel_removed_values_base}.csv",
    index=False,
)
print(f"Removed channel-value parquet saved to: {channel_removed_values_base}.parquet")
print(f"Removed channel-value CSV saved to: {channel_removed_values_base}.csv")

if track_removed_rows:
    tracking_output_directory = base_directories["tracking_directory"]
    os.makedirs(tracking_output_directory, exist_ok=True)

    removed_rows_base = os.path.join(
        tracking_output_directory,
        f"removed_rows_{basename_no_ext}",
    )
    original_cols_base = os.path.join(
        tracking_output_directory,
        f"original_cols_{basename_no_ext}",
    )
    original_columns_df = build_original_columns_frame()

    removed_rows_df.to_parquet(
        f"{removed_rows_base}.parquet",
        engine="pyarrow",
        compression="zstd",
        index=True,
    )
    removed_rows_df.to_csv(f"{removed_rows_base}.csv", index=True)
    original_columns_df.to_parquet(
        f"{original_cols_base}.parquet",
        engine="pyarrow",
        compression="zstd",
        index=True,
    )
    print(f"Removed-row tracking parquet saved to: {removed_rows_base}.parquet")
    print(f"Removed-row tracking CSV saved to: {removed_rows_base}.csv")
    print(f"Original-column snapshot parquet saved to: {original_cols_base}.parquet")

# Ensure no figure handles remain open before persistence/final move.
plt.close("all")

# Save to HDF5 file
working_df.to_parquet(OUT_PATH, engine="pyarrow", compression="zstd", index=False)
print(f"Cleaned dataframe saved to: {OUT_PATH}")

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
