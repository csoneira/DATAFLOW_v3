#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_3/script_3_cal_to_list.py
Purpose: !/usr/bin/env python3.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_3/script_3_cal_to_list.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

#%%

from __future__ import annotations

"""
Stage 1 Task 3 (CAL-->LIST) transformation.

Takes the calibrated event sample from Task 2, builds the per-hit LIST-level
representation (timing, charge, geometry groupings), applies physics-driven
selection and quality filters, and exports the structured list data required
for the fitting stages. It also manages plotting artefacts, metadata logs, and
file movements so subsequent tasks receive consistent inputs.
"""
# Standard Library
import argparse
import atexit
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
from typing import Dict, Iterable, List, Tuple

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
    select_latest_candidate,
    sync_unprocessed_with_date_range,
)
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
from MASTER.common.simulated_data_utils import resolve_simulated_z_positions
from MASTER.common.step1_activation import (
    ACTIVATION_METADATA_DECIMALS,
    compute_conditional_matrices_by_tt,
    compute_conditional_matrix_from_boolean_arrays,
    store_activation_matrices_by_tt_metadata,
    store_activation_matrix_metadata,
)
from MASTER.common.step1_shared import (
    add_normalized_count_metadata,
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
    validate_step1_input_file_args,
    y_pos,
)

task_number = 3

# I want to chrono the execution time of the script
start_execution_time_counting = datetime.now()
_prof_t0 = time.perf_counter()
_prof = {}
activation_metadata: dict[str, object] = {}
pattern_metadata: dict[str, object] = {}

STATION_CHOICES = ("0", "1", "2", "3", "4")
TASK3_PLOT_STATUSES: tuple[str, ...] = ("none", "debug", "usual", "essential")
TASK3_PLOT_ALIASES: tuple[str, ...] = (
    "incoming_parquet_main_columns_debug",
    "active_strip_patterns_overview",
    "multi_strip_pair_diagnostics",
    "tdiff_pattern_spatial_scatter",
    "tdiff_pattern_charge_scatter",
    "tdiff_pattern_charge_scan_scatter",
    "tdiff_pattern_histograms",
    "tdiff_pattern_charge_slice_fits",
    "tdiff_pattern_sigma_vs_charge",
    "tdiff_pattern_sigma1_charge_surface",
    "tdiff_pattern_sigma2_charge_surface",
    "y_position_by_cal_tt",
    "strip_variable_pairgrid",
    "self_trigger_strip_variable_pairgrid",
    "rpc_variables_hexbin",
    "rpc_variables_hexbin_low_charge",
    "filter6_tsum_debug",
    "filter6_tdif_debug",
    "filter6_qsum_debug",
    "filter6_qdif_debug",
    "filter6_y_debug",
    "filtered_rpc_variables_hexbin",
    "prefilter_qsum_nonzero_debug",
    "prefilter_list_tt_debug",
    "list_tt_qsum_pairgrid",
    "all_events_charge_threshold_population",
    "source_list_tt_charge_threshold_population",
    "list_tt_transition_matrices",
    "list_tt_retention_curves",
    "list_tt_minimum_charge_distributions",
    "list_tt_empirical_efficiency_vs_threshold",
    "full_topology_threshold_retention",
    "full_topology_exact_retention",
    "full_topology_class_fraction",
    "tsum_coincidence_window_histograms",
    "tsum_coincidence_window_vs_threshold",
    "plane_charge_fraction_vs_total_charge_threshold_scan",
    "charge_asymmetry_vs_threshold",
    "interplane_timing_correlation",
    "multiplicity_charge_landscape",
    "streamer_charge_histograms",
    "streamer_prevalence_by_plane",
    "streamer_multiplicity",
    "streamer_contagion_matrix",
    "streamer_contagion_matrix_strip",
    "streamer_efficiency_comparison",
    "fourplane_weakest_charge",
    "fourplane_weakest_plane_identity",
    "efficiency_anatomy_by_charge_band",
    "streamer_contagion_strip_by_tt",
    "signal_contagion_strip_by_tt",
    "streamer_to_signal_contagion_strip_by_tt",
    "streamer_to_highcharge_contagion_strip_by_tt",
    "strip_activation_matrix_before_after",
    "plane_combination_filter_by_tt",
    "charge_by_strip_multiplicity_adj",
    "charge_by_strip_multiplicity_dis",
)

TT_COUNT_VALUES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 12, 13, 14, 23, 24, 34, 123, 124, 134, 234, 1234
)
TT_COLOR_LABELS: tuple[str, ...] = tuple(str(tt_value) for tt_value in TT_COUNT_VALUES)
_task3_tt_palette = sns.color_palette("tab10", n_colors=10)
TT_COLOR_MAP: dict[str, tuple[float, float, float, float]] = {}
_task3_multi_idx = 0
for tt_label in TT_COLOR_LABELS:
    if len(tt_label) == 1:
        TT_COLOR_MAP[tt_label] = (0.60, 0.60, 0.60, 1.0)
    else:
        rgb = _task3_tt_palette[_task3_multi_idx % len(_task3_tt_palette)]
        TT_COLOR_MAP[tt_label] = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        _task3_multi_idx += 1
TT_COLOR_DEFAULT = (0.45, 0.45, 0.45, 1.0)


def get_tt_color(tt_value: object) -> tuple[float, float, float, float]:
    return TT_COLOR_MAP.get(normalize_tt_label(tt_value), TT_COLOR_DEFAULT)

CLI_PARSER = build_step1_cli_parser("Run Stage 1 STEP_1 TASK_3 (CAL->LIST).", STATION_CHOICES)
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


def normalize_task3_plot_mode(raw_value: object) -> str:
    if raw_value is None:
        return "none"
    if isinstance(raw_value, bool):
        return "all" if raw_value else "none"

    mode = str(raw_value).strip().lower()
    if mode in {"", "none", "null", "false", "0", "off"}:
        return "none"
    if mode == "debug":
        return "debug"
    if mode in {"usual", "standard", "normal"}:
        return "usual"
    if mode == "essential":
        return "essential"
    if mode in {"all", "true", "1", "on"}:
        return "all"

    raise ValueError(
        "Invalid create_plots value for Task 3. Use one of: null/none, debug, usual, essential, all."
    )


def normalize_task3_plot_catalog_status(raw_value: object) -> str:
    if raw_value is None:
        return "none"
    if isinstance(raw_value, bool):
        return "usual" if raw_value else "none"

    status = str(raw_value).strip().lower()
    if status in {"", "none", "null", "false", "0", "off"}:
        return "none"
    if status in {"true", "1", "on"}:
        return "usual"
    if status in {"debug", "usual", "essential"}:
        return status
    return status


def resolve_task3_plot_options(config_obj: Dict[str, object]) -> tuple[str, bool, bool, bool, bool, bool, bool]:
    plot_mode = normalize_task3_plot_mode(config_obj.get("create_plots", None))
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


def load_task3_plot_catalog(catalog_path: Path) -> dict[str, str]:
    with catalog_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Task 3 plot catalog must be a mapping: {catalog_path}")

    raw_plots = loaded.get("plots", {})
    if not isinstance(raw_plots, dict):
        raise ValueError(f"'plots' entry in {catalog_path} must be a mapping.")

    catalog: dict[str, str] = {}
    for alias, raw_entry in raw_plots.items():
        if isinstance(raw_entry, dict):
            raw_status = raw_entry.get("status", "")
        else:
            raw_status = raw_entry
        status = normalize_task3_plot_catalog_status(raw_status)
        if status not in TASK3_PLOT_STATUSES:
            raise ValueError(
                f"Invalid status {raw_status!r} for Task 3 plot alias {alias!r} in {catalog_path}."
            )
        catalog[str(alias)] = status

    unknown_aliases = sorted(alias for alias in catalog if alias not in TASK3_PLOT_ALIASES)
    for alias in unknown_aliases:
        warnings.warn(
            f"Task 3 plot catalog entry {alias!r} is not a known plot alias and will be ignored.",
            RuntimeWarning,
        )
        catalog.pop(alias, None)

    missing_aliases = [alias for alias in TASK3_PLOT_ALIASES if alias not in catalog]
    for alias in missing_aliases:
        warnings.warn(
            f"Task 3 plot alias {alias!r} is not listed in {catalog_path.name}; defaulting to 'usual'.",
            RuntimeWarning,
        )
        catalog[alias] = "usual"
    return catalog


task3_plot_status_by_alias: dict[str, str] = {}
plot_mode = "none"


def task3_plot_enabled(alias: str) -> bool:
    if alias not in task3_plot_status_by_alias:
        raise KeyError(f"Unknown Task 3 plot alias: {alias}")

    status = task3_plot_status_by_alias[alias]
    current_mode = str(globals().get("plot_mode", "none"))
    if current_mode == "none":
        return False
    if status == "none":
        return False
    if current_mode == "all":
        return True
    if current_mode == "debug":
        return status == "debug"
    if current_mode == "usual":
        return status in {"usual", "essential"}
    if current_mode == "essential":
        return status == "essential"
    return False


def task3_any_plot_enabled(*aliases: str) -> bool:
    return any(task3_plot_enabled(alias) for alias in aliases)

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
    """Save a figure to disk; the task PDF is assembled later from saved plots."""
    target_fig = fig if fig is not None else plt.gcf()
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


def finalize_saved_plots_to_pdf() -> None:
    if not create_pdf:
        return

    close_direct_pdf_writer()
    existing_pngs = collect_saved_plot_paths(plot_list, base_directories["figure_directory"])
    if not existing_pngs:
        print(
            "Warning: Plotting is enabled for Task 3 but no plot pages were generated "
            f"for {basename_no_ext}; skipping PDF creation: {save_pdf_path}"
        )
        figure_directory = base_directories["figure_directory"]
        if os.path.exists(figure_directory) and not os.listdir(figure_directory):
            os.rmdir(figure_directory)
        return

    print(f"Creating PDF with all plots in {save_pdf_path}")

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


def _task3_quantile_axis_limits(values: pd.Series | np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> tuple[float, float] | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return None

    low = float(np.nanpercentile(numeric, low_q))
    high = float(np.nanpercentile(numeric, high_q))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(np.nanmin(numeric))
        high = float(np.nanmax(numeric))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low >= high:
        center = low
        pad = max(abs(center) * 0.05, 1.0)
        low = center - pad
        high = center + pad
    return low, high


def _task3_plot_quantile_hexbin(
    ax: mpl.axes.Axes,
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    title: str,
    *,
    gridsize: int = 50,
    cmap: str = "turbo",
    low_q: float = 1.0,
    high_q: float = 99.0,
) -> None:
    x_vals = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    ax.set_title(title)

    if not np.any(valid):
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return

    x_valid = x_vals[valid]
    y_valid = y_vals[valid]
    x_limits = _task3_quantile_axis_limits(x_valid, low_q=low_q, high_q=high_q)
    y_limits = _task3_quantile_axis_limits(y_valid, low_q=low_q, high_q=high_q)
    if x_limits is None or y_limits is None:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        return

    x_low, x_high = x_limits
    y_low, y_high = y_limits
    plot_mask = (
        valid
        & (x_vals >= x_low)
        & (x_vals <= x_high)
        & (y_vals >= y_low)
        & (y_vals <= y_high)
    )
    if not np.any(plot_mask):
        plot_mask = valid

    ax.hexbin(
        x_vals[plot_mask],
        y_vals[plot_mask],
        gridsize=gridsize,
        cmap=cmap,
        mincnt=1,
        extent=(x_low, x_high, y_low, y_high),
    )
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)

# Warning Filters
warnings.filterwarnings("ignore", message=".*Data has no positive values, and therefore cannot be log-scaled.*")

start_timer(__file__)
config_root = get_master_config_root()
config_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_3"
    / "config_task_3.yaml"
)
parameter_config_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_3"
    / "config_parameters_task_3.csv"
)
plot_catalog_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_3"
    / "config_plots_task_3.yaml"
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
task3_plot_status_by_alias = load_task3_plot_catalog(plot_catalog_file_path)
debug_mode = False
home_path = str(resolve_home_path_from_config(config))

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
selection_config = load_selection_for_paths(
    [config_file_path],
    master_config_root=config_root,
)
if not station_is_selected(station, selection_config.stations):
    print(f"Station {station} skipped by selection.stations.")
    sys.exit(0)
home_path = str(resolve_home_path_from_config(config))
REFERENCE_TABLES_DIR = Path(home_path) / "DATAFLOW_v3" / "MASTER" / "CONFIG_FILES" / "METADATA_REPRISE" / "REFERENCE_TABLES"

# Round execution time to seconds and format it in YYYY-MM-DD_HH.MM.SS
execution_time = str(start_execution_time_counting).split('.')[0]  # Remove microseconds
print("Execution time is:", execution_time)

ITINERARY_FILE_PATH = Path(
    f"{home_path}/DATAFLOW_v3/MASTER/CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1/TASK_2/TIME_CALIBRATION_ITINERARIES/itineraries.csv"
)

not_use_q_semisum = False

stratos_save = config["stratos_save"]
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
) = resolve_task3_plot_options(config)
limit_number = config.get("limit_number", None)
limit = limit_number is not None

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering

# Time calibration

# Y position

# RPC variables
y_new_method = config["y_new_method"]
streamer_high_charge_factor = float(config.get("streamer_high_charge_factor", 0.2))

# Alternative

# TimTrack

# Validation

complete_reanalysis = config["complete_reanalysis"]

limit_number = config.get("limit_number", None)
limit = limit_number is not None

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
T_sum_RPC_left = config["T_sum_RPC_left"]
T_sum_RPC_right = config["T_sum_RPC_right"]
T_dif_RPC_abs = abs(float(config.get("T_dif_RPC_abs", config.get("T_dif_RPC_right", 0.8))))
T_dif_RPC_right = T_dif_RPC_abs
T_dif_RPC_left = -T_dif_RPC_abs
Q_RPC_left = config["Q_RPC_left"]
Q_RPC_right = config["Q_RPC_right"]
Q_dif_RPC_abs = abs(float(config.get("Q_dif_RPC_abs", config.get("Q_dif_RPC_right", 4))))
Q_dif_RPC_right = Q_dif_RPC_abs
Q_dif_RPC_left = -Q_dif_RPC_abs
Y_RPC_abs = abs(float(config.get("Y_RPC_abs", config.get("Y_RPC_right", 200))))
Y_RPC_right = Y_RPC_abs
Y_RPC_left = -Y_RPC_abs
plane_combination_q_sum_sum_left = float(config.get("plane_combination_q_sum_sum_left", Q_RPC_left))
plane_combination_q_sum_sum_right = float(config.get("plane_combination_q_sum_sum_right", Q_RPC_right))
plane_combination_q_sum_dif_threshold = abs(
    float(config.get("plane_combination_q_sum_dif_threshold", Q_RPC_right))
)
plane_combination_q_dif_sum_threshold = abs(
    float(config.get("plane_combination_q_dif_sum_threshold", Q_dif_RPC_abs))
)
plane_combination_q_dif_dif_threshold = abs(
    float(config.get("plane_combination_q_dif_dif_threshold", Q_dif_RPC_abs))
)
plane_combination_t_sum_sum_left = float(config.get("plane_combination_t_sum_sum_left", T_sum_RPC_left))
plane_combination_t_sum_sum_right = float(config.get("plane_combination_t_sum_sum_right", T_sum_RPC_right))
plane_combination_t_sum_dif_threshold = abs(
    float(config.get("plane_combination_t_sum_dif_threshold", max(abs(T_sum_RPC_left), abs(T_sum_RPC_right))))
)
plane_combination_t_dif_sum_threshold = abs(
    float(config.get("plane_combination_t_dif_sum_threshold", T_dif_RPC_abs))
)
plane_combination_t_dif_dif_threshold = abs(
    float(config.get("plane_combination_t_dif_dif_threshold", T_dif_RPC_abs))
)
plane_combination_y_sum_left = float(config.get("plane_combination_y_sum_left", Y_RPC_left))
plane_combination_y_sum_right = float(config.get("plane_combination_y_sum_right", Y_RPC_right))
plane_combination_y_dif_threshold = abs(
    float(config.get("plane_combination_y_dif_threshold", Y_RPC_abs))
)

# Alternative fitter filter
det_pos_filter = config.get("det_pos_filter", 800)
det_theta_left_filter = config.get("det_theta_left_filter", 0)
det_theta_right_filter = config.get("det_theta_right_filter", 1.5708)
det_phi_filter_abs = abs(float(config.get("det_phi_filter_abs", config.get("det_phi_right_filter", 3.141592))))
det_phi_right_filter = det_phi_filter_abs
det_phi_left_filter = -det_phi_filter_abs
det_slowness_filter_left = config.get("det_slowness_filter_left", -0.02)
det_slowness_filter_right = config.get("det_slowness_filter_right", 0.02)

# TimTrack filter

# Fitting comparison

# Calibrations

# Pedestal charge calibration

# Front-back charge

# Variables to modify
beta = config.get("beta", 1)
strip_speed_factor_of_c = config["strip_speed_factor_of_c"]

# X
strip_length = config.get("strip_length", 300)
narrow_strip = config["narrow_strip"]
wide_strip = config["wide_strip"]

# Timtrack parameters
anc_std = config["anc_std"]

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

TT_COUNT_VALUES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 12, 13, 14, 23, 24, 34, 123, 124, 134, 234, 1234
)

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

TASK3_ACTIVE_STRIP_COLUMNS: tuple[str, ...] = tuple(f"active_strips_P{plane}" for plane in range(1, 5))

FILTER_METRIC_NAMES: tuple[str, ...] = (
    "filter6_new_zero_rows_pct",
    "plane_combination_filter_rows_affected_pct",
    "plane_combination_filter_values_zeroed_pct",
    "plane_combination_filter_any_failed_pct",
    "plane_combination_filter_q_sum_sum_failed_pct",
    "plane_combination_filter_q_sum_dif_failed_pct",
    "plane_combination_filter_q_dif_sum_failed_pct",
    "plane_combination_filter_q_dif_dif_failed_pct",
    "plane_combination_filter_t_sum_sum_failed_pct",
    "plane_combination_filter_t_sum_dif_failed_pct",
    "plane_combination_filter_t_dif_sum_failed_pct",
    "plane_combination_filter_t_dif_dif_failed_pct",
    "plane_combination_filter_y_sum_failed_pct",
    "plane_combination_filter_y_dif_failed_pct",
    "q_sum_all_zero_rows_removed_pct",
    "data_purity_percentage",
    "all_components_zero_rows_removed_pct",
    "list_tt_lt_10_rows_removed_pct",
)
FILTER6_NONZERO_COUNTER_NAMES: tuple[str, ...] = tuple(
    f"P{i_plane}_{label}_nonzero_{tag}"
    for tag in ("before_filter6", "after_filter6")
    for i_plane in range(1, 5)
    for label in ("T_sum", "T_diff", "Q_sum", "Q_diff", "Y")
)
FILTER6_NONZERO_RATE_NAMES: tuple[str, ...] = tuple(
    f"{name}_rate_hz" for name in FILTER6_NONZERO_COUNTER_NAMES
)
FILTER6_RATE_DENOMINATOR_COLUMN = "count_rate_denominator_seconds"

filter_metrics: dict[str, float] = {}

def record_filter_metric(name: str, removed: float, total: float) -> None:
    """Record percentage removed for a filter."""
    pct = 0.0 if total == 0 else 100.0 * float(removed) / float(total)
    filter_metrics[name] = round(pct, 4)
    print(f"[filter-metrics] {name}: removed {removed} of {total} ({pct:.2f}%)")


def record_activity_metric(name: str, affected: float, total: float, label: str = "affected") -> None:
    """Record a generic percentage metric for non-row-removal filter activity."""
    pct = 0.0 if total == 0 else 100.0 * float(affected) / float(total)
    filter_metrics[name] = round(pct, 4)
    print(f"[filter-metrics] {name}: {label} {affected} of {total} ({pct:.2f}%)")


def build_task3_full_strip_pattern_series(df: pd.DataFrame) -> pd.Series:
    """Encode the four per-plane active-strip labels as one deterministic 16-bit string."""
    pattern_arrays: list[np.ndarray] = []
    for col_name in TASK3_ACTIVE_STRIP_COLUMNS:
        if col_name in df.columns:
            values = df[col_name].fillna("0000").astype(str).to_numpy(copy=False)
        else:
            print(f"Warning: missing active-strip column '{col_name}' while building TASK_3 patterns.")
            values = np.full(len(df), "0000", dtype="<U4")
        pattern_arrays.append(values)

    if not pattern_arrays:
        return pd.Series(dtype="object", index=df.index)

    full_pattern = pattern_arrays[0].copy()
    for values in pattern_arrays[1:]:
        full_pattern = np.char.add(full_pattern, values)
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


def compute_strip_activation_conditional_matrix(
    df: pd.DataFrame,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Return strip-level conditional activation matrix P(target strip active | given strip active)."""
    labels: list[str] = []
    strip_active_arrays: list[np.ndarray] = []

    for plane in range(1, 5):
        col_name = f"active_strips_P{plane}"
        if col_name in df.columns:
            strip_values = df[col_name].fillna("0000").astype(str)
            strip_values = strip_values.where(strip_values.str.len() == 4, "0000")
        else:
            strip_values = pd.Series("0000", index=df.index, dtype="object")

        for strip_index in range(4):
            labels.append(f"P{plane}S{strip_index + 1}")
            strip_active_arrays.append((strip_values.str[strip_index] == "1").to_numpy(dtype=bool))

    n_strips = len(labels)
    cond = np.full((n_strips, n_strips), np.nan, dtype=float)
    given_counts: dict[str, int] = {}

    for i in range(n_strips):
        n_i = int(np.sum(strip_active_arrays[i]))
        given_counts[labels[i]] = n_i
        if n_i == 0:
            continue
        for j in range(n_strips):
            if i == j:
                cond[i, j] = 1.0
            else:
                cond[i, j] = float(np.sum(strip_active_arrays[i] & strip_active_arrays[j])) / n_i

    return labels, cond, given_counts


def summarize_strip_activation_matrix(labels: list[str], matrix: np.ndarray) -> dict[str, float | str]:
    """Extract compact scalar summaries from a strip conditional matrix."""
    if matrix.size == 0 or len(labels) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        return {
            "mean_off_diagonal": "",
            "max_off_diagonal": "",
            "mean_off_diagonal_interplane": "",
        }

    finite = np.isfinite(matrix)
    n = matrix.shape[0]
    off_diag_mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(off_diag_mask, False)

    off_diag_values = matrix[finite & off_diag_mask]
    mean_off_diag = float(np.mean(off_diag_values)) if off_diag_values.size else ""
    max_off_diag = float(np.max(off_diag_values)) if off_diag_values.size else ""

    plane_ids = np.array([int(label[1]) for label in labels], dtype=int)
    interplane_mask = off_diag_mask & (plane_ids[:, None] != plane_ids[None, :])
    interplane_values = matrix[finite & interplane_mask]
    mean_off_diag_interplane = (
        float(np.mean(interplane_values)) if interplane_values.size else ""
    )

    return {
        "mean_off_diagonal": (
            round(mean_off_diag, ACTIVATION_METADATA_DECIMALS) if mean_off_diag != "" else ""
        ),
        "max_off_diagonal": (
            round(max_off_diag, ACTIVATION_METADATA_DECIMALS) if max_off_diag != "" else ""
        ),
        "mean_off_diagonal_interplane": (
            round(mean_off_diag_interplane, ACTIVATION_METADATA_DECIMALS)
            if mean_off_diag_interplane != ""
            else ""
        ),
    }


def store_strip_activation_matrix_metadata(
    metadata: dict[str, object],
    prefix: str,
    labels: list[str],
    matrix: np.ndarray,
    given_counts: dict[str, int],
) -> None:
    """Persist strip-activation matrix values and summary metrics into metadata."""
    summary = summarize_strip_activation_matrix(labels, matrix)
    metadata[f"strip_activation_mean_off_diagonal_{prefix}"] = summary["mean_off_diagonal"]
    metadata[f"strip_activation_max_off_diagonal_{prefix}"] = summary["max_off_diagonal"]
    metadata[f"strip_activation_mean_off_diagonal_interplane_{prefix}"] = summary[
        "mean_off_diagonal_interplane"
    ]

    for label in labels:
        metadata[f"strip_activation_given_count_{prefix}_{label}"] = int(given_counts.get(label, 0))

    if matrix.size == 0:
        return

    for i, src_label in enumerate(labels):
        for j, dst_label in enumerate(labels):
            value = matrix[i, j]
            field = f"strip_activation_conditional_{prefix}_{src_label}_to_{dst_label}"
            metadata[field] = (
                round(float(value), ACTIVATION_METADATA_DECIMALS) if np.isfinite(value) else ""
            )


TASK3_PLANE_LABELS: tuple[str, ...] = tuple(f"P{plane}" for plane in range(1, 5))


def _task3_plane_charge_arrays(df: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:
    labels: list[str] = []
    arrays: list[np.ndarray] = []
    for plane in range(1, 5):
        column_name = f"P{plane}_Q_sum_final"
        if column_name in df.columns:
            arr = pd.to_numeric(df[column_name], errors="coerce").to_numpy(dtype=float)
        else:
            arr = np.full(len(df), np.nan, dtype=float)
        labels.append(f"P{plane}")
        arrays.append(arr)
    return labels, arrays


def _format_activation_scalar(value: float | None) -> float | str:
    if value is None:
        return ""
    value_float = float(value)
    if not np.isfinite(value_float):
        return ""
    return round(value_float, ACTIVATION_METADATA_DECIMALS)


def _store_activation_scalar(
    activation_meta: dict[str, object],
    scalar_meta: dict[str, object],
    key: str,
    value: object,
) -> None:
    activation_meta[key] = value
    scalar_meta.pop(key, None)


def store_task3_plane_activation_snapshot(
    *,
    activation_meta: dict[str, object],
    scalar_meta: dict[str, object],
    df: pd.DataFrame,
    snapshot_label: str,
    tt_series: pd.Series,
    streamer_high_charge_factor_value: float,
) -> dict[str, object]:
    labels, charge_arrays = _task3_plane_charge_arrays(df)
    signal_arrays = [np.isfinite(arr) & (arr > 0) for arr in charge_arrays]
    signal_matrix, signal_given_counts = compute_conditional_matrix_from_boolean_arrays(
        labels,
        signal_arrays,
    )
    store_activation_matrix_metadata(
        activation_meta,
        f"activation_plane_signal_to_signal_{snapshot_label}",
        labels,
        signal_matrix,
        signal_given_counts,
        group_ids=[0, 1, 2, 3],
    )
    selected_tts, matrices_by_tt, given_counts_by_tt, event_counts_by_tt = (
        compute_conditional_matrices_by_tt(
            tt_series,
            labels,
            signal_arrays,
        )
    )
    store_activation_matrices_by_tt_metadata(
        activation_meta,
        f"activation_plane_signal_to_signal_{snapshot_label}_by_tt",
        labels,
        labels,
        selected_tts,
        matrices_by_tt,
        given_counts_by_tt,
        event_counts_by_tt,
    )

    q_sum_cols = [f"P{plane}_Q_sum_final" for plane in range(1, 5)]
    streamer_threshold = detect_streamer_threshold(df, q_sum_cols)
    threshold_key = f"streamer_threshold_selected_{snapshot_label}"
    _store_activation_scalar(
        activation_meta,
        scalar_meta,
        threshold_key,
        _format_activation_scalar(streamer_threshold),
    )

    high_charge_threshold = None
    if streamer_threshold is not None:
        high_charge_threshold = float(streamer_threshold) * float(streamer_high_charge_factor_value)
        _store_activation_scalar(
            activation_meta,
            scalar_meta,
            f"streamer_high_charge_threshold_selected_{snapshot_label}",
            _format_activation_scalar(high_charge_threshold),
        )
    else:
        _store_activation_scalar(
            activation_meta,
            scalar_meta,
            f"streamer_high_charge_threshold_selected_{snapshot_label}",
            "",
        )

    if snapshot_label == "filtered":
        _store_activation_scalar(
            activation_meta,
            scalar_meta,
            "streamer_high_charge_factor",
            _format_activation_scalar(streamer_high_charge_factor_value),
        )
        _store_activation_scalar(
            activation_meta,
            scalar_meta,
            "streamer_threshold_selected",
            activation_meta.get(threshold_key, ""),
        )
        _store_activation_scalar(
            activation_meta,
            scalar_meta,
            "streamer_high_charge_threshold_selected",
            activation_meta.get(f"streamer_high_charge_threshold_selected_{snapshot_label}", ""),
        )

    if streamer_threshold is None or high_charge_threshold is None:
        for plane in range(1, 5):
            _store_activation_scalar(
                activation_meta,
                scalar_meta,
                f"streamer_rate_plane_{snapshot_label}_{plane}",
                "",
            )
            if snapshot_label == "filtered":
                _store_activation_scalar(
                    activation_meta,
                    scalar_meta,
                    f"streamer_rate_plane_{plane}",
                    "",
                )
        return {
            "labels": labels,
            "signal_matrix": signal_matrix,
            "streamer_matrix": np.empty((0, 0), dtype=float),
        }

    streamer_arrays = [np.isfinite(arr) & (arr > float(streamer_threshold)) for arr in charge_arrays]
    high_charge_arrays = [np.isfinite(arr) & (arr > float(high_charge_threshold)) for arr in charge_arrays]
    streamer_matrix = np.empty((0, 0), dtype=float)
    for plane, signal_mask, streamer_mask in zip(range(1, 5), signal_arrays, streamer_arrays):
        n_signal = int(np.sum(signal_mask))
        n_streamer = int(np.sum(streamer_mask))
        rate_key = f"streamer_rate_plane_{snapshot_label}_{plane}"
        _store_activation_scalar(
            activation_meta,
            scalar_meta,
            rate_key,
            _format_activation_scalar(float(n_streamer) / float(n_signal)) if n_signal > 0 else "",
        )
        if snapshot_label == "filtered":
            _store_activation_scalar(
                activation_meta,
                scalar_meta,
                f"streamer_rate_plane_{plane}",
                activation_meta.get(rate_key, ""),
            )

    variant_specs = [
        ("streamer_to_streamer", streamer_arrays, streamer_arrays),
        ("streamer_to_signal", streamer_arrays, signal_arrays),
        ("streamer_to_highcharge", streamer_arrays, high_charge_arrays),
    ]
    for variant_name, source_arrays, target_arrays in variant_specs:
        variant_matrix, variant_given_counts = compute_conditional_matrix_from_boolean_arrays(
            labels,
            source_arrays,
            target_labels=labels,
            target_arrays=target_arrays,
        )
        if variant_name == "streamer_to_streamer":
            streamer_matrix = variant_matrix
        store_activation_matrix_metadata(
            activation_meta,
            f"activation_plane_{variant_name}_{snapshot_label}",
            labels,
            variant_matrix,
            variant_given_counts,
            target_labels=labels,
            group_ids=[0, 1, 2, 3],
        )
        selected_tts, matrices_by_tt, given_counts_by_tt, event_counts_by_tt = (
            compute_conditional_matrices_by_tt(
                tt_series,
                labels,
                source_arrays,
                target_labels=labels,
                target_arrays=target_arrays,
            )
        )
        store_activation_matrices_by_tt_metadata(
            activation_meta,
            f"activation_plane_{variant_name}_{snapshot_label}_by_tt",
            labels,
            labels,
            selected_tts,
            matrices_by_tt,
            given_counts_by_tt,
            event_counts_by_tt,
        )

    return {
        "labels": labels,
        "signal_matrix": signal_matrix,
        "streamer_matrix": streamer_matrix,
    }


def plot_task3_plane_activation_before_after(
    initial_snapshot: dict[str, object],
    filtered_snapshot: dict[str, object],
    fig_idx_value: int,
) -> int:
    labels_initial = initial_snapshot.get("labels", [])
    labels_filtered = filtered_snapshot.get("labels", [])
    matrix_initial = initial_snapshot.get("signal_matrix")
    matrix_filtered = filtered_snapshot.get("signal_matrix")
    if (
        not isinstance(labels_initial, list)
        or not isinstance(labels_filtered, list)
        or labels_initial != labels_filtered
        or not isinstance(matrix_initial, np.ndarray)
        or not isinstance(matrix_filtered, np.ndarray)
        or matrix_initial.size == 0
        or matrix_filtered.size == 0
    ):
        return fig_idx_value

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    last_im = None
    for ax, matrix, title_text in zip(
        axes,
        [matrix_initial, matrix_filtered],
        ["Initial plane activation", "Filtered plane activation"],
    ):
        last_im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(len(labels_initial)))
        ax.set_xticklabels(labels_initial, fontsize=9)
        ax.set_yticks(range(len(labels_initial)))
        ax.set_yticklabels(labels_initial, fontsize=9)
        ax.set_title(title_text, fontsize=11)
        ax.set_xlabel("Target plane active")
        ax.set_ylabel("Given plane active")
        for i in range(len(labels_initial)):
            for j in range(len(labels_initial)):
                value = matrix[i, j]
                if np.isfinite(value):
                    text_color = "white" if value > 0.6 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color=text_color)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, label="P(target plane active | given plane active)", shrink=0.85)
    fig.suptitle("Plane activation matrix: initial to filtered", fontsize=13)
    if save_plots:
        final_filename = f"{fig_idx_value}_plane_activation_matrix_before_after.png"
        fig_idx_value += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig_idx_value


def compute_qsum_threshold_tt(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Return TT labels using only planes with P*_Q_sum_final > threshold."""
    tt_str = pd.Series("", index=df.index, dtype="object")
    for plane in range(1, 5):
        col_name = f"P{plane}_Q_sum_final"
        if col_name not in df.columns:
            continue
        charge_vals = pd.to_numeric(df[col_name], errors="coerce")
        plane_active = charge_vals.gt(float(threshold))
        tt_str = tt_str.where(~plane_active, tt_str + str(plane))
    return tt_str.replace("", "0").astype(int)


def compute_qsum_threshold_full_strip_patterns(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Return full 16-bit strip topology after applying per-plane Q_sum threshold."""
    plane_arrays: list[np.ndarray] = []
    for plane in range(1, 5):
        strip_col = f"active_strips_P{plane}"
        qsum_col = f"P{plane}_Q_sum_final"

        if strip_col in df.columns:
            strip_vals = df[strip_col].fillna("0000").astype(str)
            strip_vals = strip_vals.where(strip_vals.str.len() == 4, "0000")
        else:
            strip_vals = pd.Series("0000", index=df.index, dtype="object")

        if qsum_col in df.columns:
            qsum_vals = pd.to_numeric(df[qsum_col], errors="coerce")
            keep_plane = qsum_vals.gt(float(threshold)).to_numpy(dtype=bool)
        else:
            keep_plane = np.zeros(len(df), dtype=bool)

        filtered = np.where(keep_plane, strip_vals.to_numpy(dtype=object), "0000").astype(str)
        plane_arrays.append(filtered)

    if not plane_arrays:
        return pd.Series(dtype="object", index=df.index)

    full_pattern = plane_arrays[0]
    for values in plane_arrays[1:]:
        full_pattern = np.char.add(full_pattern, values)
    return pd.Series(full_pattern, index=df.index, dtype="object")


def compute_qsum_threshold_tsum_window(df: pd.DataFrame, threshold: float) -> tuple[pd.Series, pd.Series]:
    """Compute T_sum coincidence window=max-min across planes with Q_sum > threshold."""
    t_cols = [f"P{plane}_T_sum_final" for plane in range(1, 5)]
    q_cols = [f"P{plane}_Q_sum_final" for plane in range(1, 5)]

    t_vals = np.full((len(df), 4), np.nan, dtype=float)
    q_vals = np.full((len(df), 4), np.nan, dtype=float)
    for idx, (t_col, q_col) in enumerate(zip(t_cols, q_cols)):
        if t_col in df.columns:
            t_vals[:, idx] = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
        if q_col in df.columns:
            q_vals[:, idx] = pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float)

    active = np.isfinite(q_vals) & np.isfinite(t_vals) & (q_vals > float(threshold))
    active_counts = active.sum(axis=1)
    masked_t = np.where(active, t_vals, np.nan)

    with np.errstate(all="ignore"):
        t_max = np.nanmax(masked_t, axis=1)
        t_min = np.nanmin(masked_t, axis=1)
    window = t_max - t_min
    window = np.where(active_counts >= 2, window, np.nan)

    return (
        pd.Series(window, index=df.index, dtype=float),
        pd.Series(active_counts, index=df.index, dtype=int),
    )


def _count_turns(strips: list[int]) -> int:
    if len(strips) < 3:
        return 0
    deltas = [strips[idx + 1] - strips[idx] for idx in range(len(strips) - 1)]
    signs = [int(np.sign(delta)) for delta in deltas if delta != 0]
    return int(sum(1 for idx in range(len(signs) - 1) if signs[idx] != signs[idx + 1]))


def classify_full_strip_topology(full_pattern: str) -> tuple[str, str, str]:
    """Classify full strip topology into active-mask and coarse topology class."""
    if not isinstance(full_pattern, str) or len(full_pattern) != 16:
        return "0000", "invalid", ""

    plane_patterns = [full_pattern[idx : idx + 4] for idx in range(0, 16, 4)]
    active_planes: list[int] = []
    strip_path: list[int] = []
    has_multi_strip = False

    for plane_idx, pattern in enumerate(plane_patterns, start=1):
        if pattern == "0000":
            continue
        active_planes.append(plane_idx)
        if pattern.count("1") == 1:
            strip_path.append(pattern.index("1") + 1)
        else:
            has_multi_strip = True

    active_mask = "".join("1" if plane in active_planes else "0" for plane in range(1, 5))
    if not active_planes:
        return active_mask, "empty", ""
    if has_multi_strip:
        return active_mask, "multi_strip", ""

    path_label = "-".join(str(val) for val in strip_path)
    jumps = [abs(strip_path[idx + 1] - strip_path[idx]) for idx in range(len(strip_path) - 1)]
    max_jump = max(jumps) if jumps else 0
    turns = _count_turns(strip_path)
    if turns >= 1:
        return active_mask, "single_strip_zigzag", path_label
    if max_jump >= 2:
        return active_mask, "single_strip_rough", path_label
    return active_mask, "single_strip_smooth", path_label


def tt_value_to_planes(tt_value: object) -> list[int]:
    label = normalize_tt_label(tt_value, default="0")
    return [int(char) for char in label if char in {"1", "2", "3", "4"}]


def compute_source_tt_min_charge(df: pd.DataFrame, source_tt: object) -> pd.Series:
    planes = tt_value_to_planes(source_tt)
    if not planes:
        return pd.Series(np.nan, index=df.index, dtype=float)

    cols = [f"P{plane}_Q_sum_final" for plane in planes if f"P{plane}_Q_sum_final" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=float)

    charge_df = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return charge_df.min(axis=1, skipna=False)


def plot_population_table(
    counts_df: pd.DataFrame,
    title: str,
    filename_suffix: str,
    fig_idx: int,
    base_dir: str,
    *,
    row_label: str = "Charge-filtered plane combination",
    col_label: str = "Charge threshold",
    colorbar_label: str = "log10(count + 1)",
    show_plots: bool = False,
    save_plots: bool = False,
    plot_list: list[str] | None = None,
) -> int:
    """Render an annotated heatmap-style table for threshold population counts."""
    if counts_df.empty:
        return fig_idx

    display_values = counts_df.to_numpy(dtype=float)
    color_values = np.log10(display_values + 1.0)
    fig_width = max(7.0, 1.35 * counts_df.shape[1] + 2.5)
    fig_height = max(4.0, 0.45 * counts_df.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(color_values, cmap="turbo", aspect="auto")
    ax.set_xticks(np.arange(counts_df.shape[1]))
    ax.set_yticks(np.arange(counts_df.shape[0]))
    ax.set_xticklabels(counts_df.columns)
    ax.set_yticklabels(counts_df.index.astype(str))
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)

    max_color = float(np.nanmax(color_values)) if color_values.size else 0.0
    for i_row in range(counts_df.shape[0]):
        for j_col in range(counts_df.shape[1]):
            value = int(display_values[i_row, j_col])
            if value <= 0:
                continue
            color_value = float(color_values[i_row, j_col])
            text_color = "black" if max_color > 0 and color_value > 0.55 * max_color else "white"
            ax.text(j_col, i_row, f"{value}", ha="center", va="center", color=text_color)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(colorbar_label)
    fig.tight_layout()

    if save_plots:
        final_filename = f"{fig_idx}_{filename_suffix}.png"
        save_fig_path = os.path.join(base_dir, final_filename)
        if plot_list is not None:
            plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig_idx + 1


def compute_empirical_efficiency_from_tt_counts(
    tt_counts: pd.Series,
) -> dict[int, tuple[float, float, int, int]]:
    """Compute per-plane empirical efficiencies 1 - N(others)/N(1234)."""
    n_four = int(tt_counts.get(1234, 0))
    missing_plane_tt = {1: 234, 2: 134, 3: 124, 4: 123}
    results: dict[int, tuple[float, float, int, int]] = {}

    for plane, tt_value in missing_plane_tt.items():
        n_three = int(tt_counts.get(tt_value, 0))
        if n_four <= 0:
            results[plane] = (np.nan, np.nan, n_three, n_four)
            continue

        n_three_float = float(n_three)
        n_four_float = float(n_four)
        efficiency = 1.0 - (n_three_float / n_four_float)
        variance = (n_three_float / (n_four_float ** 2)) + ((n_three_float ** 2) / (n_four_float ** 3))
        error = math.sqrt(max(variance, 0.0))
        results[plane] = (efficiency, error, n_three, n_four)

    return results


def detect_streamer_threshold(
    df: pd.DataFrame,
    q_sum_cols: list[str],
    *,
    sigma: float = 3.0,
    n_bins: int = 300,
    search_start_quantile: float = 60.0,
) -> float | None:
    """Auto-detect the avalanche–streamer valley in the pooled Q_sum spectrum.

    Pools all non-zero Q_sum values across *q_sum_cols*, builds a smoothed
    histogram, and locates the first local minimum after the avalanche peak
    (searched from *search_start_quantile* of the distribution onward).

    Returns the charge value at the valley, or None if detection fails.
    """
    vals_list = []
    for col in q_sum_cols:
        if col not in df.columns:
            continue
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            vals_list.append(arr)
    if not vals_list:
        return None
    pooled = np.concatenate(vals_list)
    if pooled.size < 200:
        return None

    q_low = float(np.nanpercentile(pooled, 1))
    q_high = float(np.nanpercentile(pooled, 99.9))
    if q_high <= q_low:
        return None

    counts, edges = np.histogram(pooled, bins=n_bins, range=(q_low, q_high))
    centres = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)

    # Search for the valley starting from the search_start_quantile
    search_start = float(np.nanpercentile(pooled, search_start_quantile))
    start_idx = int(np.searchsorted(centres, search_start))
    start_idx = max(1, min(start_idx, len(smoothed) - 2))

    # Find first local minimum after avalanche peak
    for idx in range(start_idx, len(smoothed) - 1):
        if smoothed[idx] < smoothed[idx - 1] and smoothed[idx] <= smoothed[idx + 1]:
            return float(centres[idx])

    return None


def build_strip_boolean_arrays(
    df: pd.DataFrame,
    condition_per_plane: dict[int, np.ndarray],
) -> tuple[list[str], list[np.ndarray]]:
    """Build per-strip boolean arrays from a per-plane condition and active_strips columns.

    *condition_per_plane* maps plane number → boolean array (length = len(df)).
    A strip is True when the plane condition is True AND the strip bit is 1.
    Returns (labels, arrays) where labels are like "P1S1", "P1S2", ..., "P4S4".
    """
    labels: list[str] = []
    arrays: list[np.ndarray] = []
    for p in sorted(condition_per_plane.keys()):
        as_col = f"active_strips_P{p}"
        if as_col not in df.columns:
            continue
        pattern = df[as_col].fillna("0000").astype(str)
        pattern = pattern.where(pattern.str.len() == 4, "0000")
        plane_cond = condition_per_plane[p]
        for s in range(4):
            strip_on = (pattern.str[s] == "1").to_numpy(dtype=bool)
            arrays.append(plane_cond & strip_on)
            labels.append(f"P{p}S{s+1}")
    return labels, arrays


def compute_strip_contagion_matrices_by_tt(
    tt_series: pd.Series,
    source_labels: list[str],
    source_arrays: list[np.ndarray],
    target_labels: list[str],
    target_arrays: list[np.ndarray],
    *,
    min_events: int = 30,
    max_tt_panels: int = 6,
) -> tuple[list[int], dict[int, np.ndarray], dict[int, dict[str, int]], dict[int, int]]:
    """Compute per-TT conditional strip contagion matrices P(target | given source)."""
    if not source_labels or not source_arrays or not target_labels or not target_arrays:
        return [], {}, {}, {}

    tt_counts = tt_series.value_counts()
    selected_tts = [
        tt for tt, cnt in tt_counts.items()
        if cnt >= min_events and tt >= 10
    ][:max_tt_panels]
    if not selected_tts:
        return [], {}, {}, {}

    matrices: dict[int, np.ndarray] = {}
    given_counts_by_tt: dict[int, dict[str, int]] = {}
    n_events_by_tt: dict[int, int] = {}

    for tt_val in selected_tts:
        mask_tt = (tt_series == tt_val).to_numpy(dtype=bool)
        n_events_by_tt[tt_val] = int(np.sum(mask_tt))
        cond = np.full((len(source_labels), len(target_labels)), np.nan, dtype=float)
        given_counts: dict[str, int] = {}

        for i, src_label in enumerate(source_labels):
            src_active = source_arrays[i] & mask_tt
            n_given = int(np.sum(src_active))
            given_counts[src_label] = n_given
            if n_given == 0:
                continue
            for j in range(len(target_labels)):
                cond[i, j] = float(np.sum(src_active & target_arrays[j])) / n_given

        matrices[tt_val] = cond
        given_counts_by_tt[tt_val] = given_counts

    return selected_tts, matrices, given_counts_by_tt, n_events_by_tt


def store_strip_contagion_by_tt_metadata(
    metadata: dict[str, object],
    variant_prefix: str,
    source_labels: list[str],
    target_labels: list[str],
    selected_tts: list[int],
    matrices: dict[int, np.ndarray],
    given_counts_by_tt: dict[int, dict[str, int]],
    n_events_by_tt: dict[int, int],
) -> None:
    """Store per-TT strip contagion matrices and source counts in specific metadata."""
    metadata[f"streamer_contagion_{variant_prefix}_selected_tts"] = ",".join(
        str(tt) for tt in selected_tts
    )

    for tt_val in selected_tts:
        metadata[f"streamer_contagion_{variant_prefix}_tt{tt_val}_event_count"] = int(
            n_events_by_tt.get(tt_val, 0)
        )
        given_counts = given_counts_by_tt.get(tt_val, {})
        for src_label in source_labels:
            metadata[
                f"streamer_contagion_{variant_prefix}_tt{tt_val}_given_count_{src_label}"
            ] = int(given_counts.get(src_label, 0))

        matrix = matrices.get(tt_val)
        if matrix is None:
            continue
        for i, src_label in enumerate(source_labels):
            for j, dst_label in enumerate(target_labels):
                value = matrix[i, j]
                field = (
                    f"streamer_contagion_{variant_prefix}_tt{tt_val}_{src_label}_to_{dst_label}"
                )
                metadata[field] = (
                    round(float(value), ACTIVATION_METADATA_DECIMALS) if np.isfinite(value) else ""
                )


def plot_strip_contagion_by_tt(
    df: pd.DataFrame,
    source_labels: list[str],
    source_arrays: list[np.ndarray],
    tt_column: str,
    *,
    title_prefix: str,
    filename_suffix: str,
    fig_idx: int,
    base_dir: str,
    save_plots: bool,
    show_plots: bool,
    plot_list: list[str],
    target_labels: list[str] | None = None,
    target_arrays: list[np.ndarray] | None = None,
    min_events: int = 30,
    max_tt_panels: int = 6,
) -> int:
    """Plot strip-level contagion matrices split by plane combination (TT)."""
    if not source_labels or not source_arrays:
        return fig_idx

    if target_labels is None:
        target_labels = source_labels
    if target_arrays is None:
        target_arrays = source_arrays
    if not target_labels or not target_arrays:
        return fig_idx

    tt_series = pd.to_numeric(df[tt_column], errors="coerce").fillna(0).astype(int)
    selected_tts, matrices, _, n_events_by_tt = compute_strip_contagion_matrices_by_tt(
        tt_series,
        source_labels,
        source_arrays,
        target_labels,
        target_arrays,
        min_events=min_events,
        max_tt_panels=max_tt_panels,
    )
    if not selected_tts:
        return fig_idx

    n_source = len(source_labels)
    n_target = len(target_labels)
    ncols = len(selected_tts)
    fig_w = max(6, 5.5 * ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 5.5 + 0.15 * max(n_source, n_target)), squeeze=False)

    for col_idx, tt_val in enumerate(selected_tts):
        ax = axes[0, col_idx]
        n_tt = int(n_events_by_tt.get(tt_val, 0))
        cond = matrices.get(tt_val)
        if cond is None:
            continue

        im = ax.imshow(cond, cmap="YlOrRd", vmin=0, aspect="equal")
        ax.set_xticks(range(n_target))
        ax.set_xticklabels(target_labels, fontsize=5, rotation=90)
        ax.set_yticks(range(n_source))
        ax.set_yticklabels(source_labels, fontsize=5)
        for i in range(n_source):
            for j in range(n_target):
                v = cond[i, j]
                if np.isfinite(v) and v > 0.005:
                    tc = "white" if v > 0.5 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color=tc)
        for boundary in range(4, max(n_source, n_target), 4):
            ax.axhline(boundary - 0.5, color="grey", linewidth=0.7, alpha=0.6)
            ax.axvline(boundary - 0.5, color="grey", linewidth=0.7, alpha=0.6)
        ax.set_title(f"TT {tt_val} (N={n_tt})", fontsize=10)

    fig.colorbar(im, ax=axes[0, -1], label="P(target | given)", shrink=0.75, pad=0.02)
    fig.suptitle(title_prefix, fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.97, 0.93])
    if save_plots:
        final_filename = f"{fig_idx}_{filename_suffix}.png"
        fig_idx += 1
        save_fig_path = os.path.join(base_dir, final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig)
    return fig_idx


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


def collect_task3_plane_final_map(df: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Return the available plane-level final columns grouped by plane."""
    plane_map: dict[int, dict[str, str]] = {}
    for plane in range(1, 5):
        cols = {
            "T_sum": f"P{plane}_T_sum_final",
            "T_dif": f"P{plane}_T_dif_final",
            "Q_sum": f"P{plane}_Q_sum_final",
            "Q_dif": f"P{plane}_Q_dif_final",
            "Y": f"P{plane}_Y_final",
        }
        if all(col in df.columns for col in cols.values()):
            plane_map[plane] = cols
    return plane_map


TASK3_PLANE_KEYS: tuple[int, ...] = (1, 2, 3, 4)


def _task3_plane_order_key(plane_key: int) -> tuple[int]:
    return (plane_key,)


def _task3_plane_offender_metric_key(plane_key: int) -> str:
    return f"plane_combination_filter_offender_count_P{plane_key}"


def apply_task3_plane_combination_filter(
    df_input: pd.DataFrame,
    *,
    q_sum_sum_left: float,
    q_sum_sum_right: float,
    q_sum_dif_threshold: float,
    q_dif_sum_threshold: float,
    q_dif_dif_threshold: float,
    t_sum_sum_left: float,
    t_sum_sum_right: float,
    t_sum_dif_threshold: float,
    t_dif_sum_threshold: float,
    t_dif_dif_threshold: float,
    y_sum_left: float,
    y_sum_right: float,
    y_dif_threshold: float,
) -> dict[str, int]:
    """
    Apply a final Task 3 plane-combination consistency filter.

    Distinct plane pairs are evaluated only when both planes carry non-zero
    `Q_sum/Q_dif/T_sum/T_dif/Y`. Failed plane pairs are flagged first. Then,
    per event, only the smallest high-impact set of offending planes is zeroed
    so the full set of flagged plane pairs is covered.
    """
    plane_map = collect_task3_plane_final_map(df_input)
    if len(plane_map) < 2:
        return {
            "tracked_plane_count": len(plane_map),
            "valid_pair_observations": 0,
            "failed_pair_any": 0,
            "failed_pair_q_sum_sum": 0,
            "failed_pair_q_sum_dif": 0,
            "failed_pair_q_dif_sum": 0,
            "failed_pair_q_dif_dif": 0,
            "failed_pair_t_sum_sum": 0,
            "failed_pair_t_sum_dif": 0,
            "failed_pair_t_dif_sum": 0,
            "failed_pair_t_dif_dif": 0,
            "failed_pair_y_sum": 0,
            "failed_pair_y_dif": 0,
            "rows_affected": 0,
            "values_zeroed": 0,
            "flagged_rows": 0,
            "selected_offender_planes": 0,
            "rows_with_multiple_offenders": 0,
            "max_failed_pairs_in_row": 0,
            "max_selected_offenders_in_row": 0,
            "selected_offender_counts": {
                plane_key: 0 for plane_key in TASK3_PLANE_KEYS
            },
        }

    plane_fail_masks = {
        plane_key: np.zeros(len(df_input), dtype=bool)
        for plane_key in TASK3_PLANE_KEYS
    }
    summary = {
        "tracked_plane_count": len(plane_map),
        "valid_pair_observations": 0,
        "failed_pair_any": 0,
        "failed_pair_q_sum_sum": 0,
        "failed_pair_q_sum_dif": 0,
        "failed_pair_q_dif_sum": 0,
        "failed_pair_q_dif_dif": 0,
        "failed_pair_t_sum_sum": 0,
        "failed_pair_t_sum_dif": 0,
        "failed_pair_t_dif_sum": 0,
        "failed_pair_t_dif_dif": 0,
        "failed_pair_y_sum": 0,
        "failed_pair_y_dif": 0,
        "rows_affected": 0,
        "values_zeroed": 0,
        "flagged_rows": 0,
        "selected_offender_planes": 0,
        "rows_with_multiple_offenders": 0,
        "max_failed_pairs_in_row": 0,
        "max_selected_offenders_in_row": 0,
    }
    row_failed_edges: dict[int, list[tuple[int, int, float]]] = {}
    offender_hit_counts = {
        plane_key: 0
        for plane_key in TASK3_PLANE_KEYS
    }

    for plane_a, plane_b in combinations(sorted(plane_map), 2):
        cols_a = plane_map[plane_a]
        cols_b = plane_map[plane_b]
        q_sum_a = pd.to_numeric(df_input[cols_a["Q_sum"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_sum_b = pd.to_numeric(df_input[cols_b["Q_sum"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_dif_a = pd.to_numeric(df_input[cols_a["Q_dif"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_dif_b = pd.to_numeric(df_input[cols_b["Q_dif"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_sum_a = pd.to_numeric(df_input[cols_a["T_sum"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_sum_b = pd.to_numeric(df_input[cols_b["T_sum"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_dif_a = pd.to_numeric(df_input[cols_a["T_dif"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_dif_b = pd.to_numeric(df_input[cols_b["T_dif"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        y_a = pd.to_numeric(df_input[cols_a["Y"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        y_b = pd.to_numeric(df_input[cols_b["Y"]], errors="coerce").fillna(0).to_numpy(dtype=float)

        valid_mask = (
            np.isfinite(q_sum_a)
            & np.isfinite(q_sum_b)
            & np.isfinite(q_dif_a)
            & np.isfinite(q_dif_b)
            & np.isfinite(t_sum_a)
            & np.isfinite(t_sum_b)
            & np.isfinite(t_dif_a)
            & np.isfinite(t_dif_b)
            & np.isfinite(y_a)
            & np.isfinite(y_b)
            & (q_sum_a != 0)
            & (q_sum_b != 0)
            & (q_dif_a != 0)
            & (q_dif_b != 0)
            & (t_sum_a != 0)
            & (t_sum_b != 0)
            & (t_dif_a != 0)
            & (t_dif_b != 0)
            & (y_a != 0)
            & (y_b != 0)
        )
        if not np.any(valid_mask):
            continue

        summary["valid_pair_observations"] += int(np.count_nonzero(valid_mask))
        pair_q_sum_sum = 0.5 * (q_sum_a + q_sum_b)
        pair_q_sum_dif = 0.5 * (q_sum_a - q_sum_b)
        pair_q_dif_sum = 0.5 * (q_dif_a + q_dif_b)
        pair_q_dif_dif = 0.5 * (q_dif_a - q_dif_b)
        pair_t_sum_sum = 0.5 * (t_sum_a + t_sum_b)
        pair_t_sum_dif = 0.5 * (t_sum_a - t_sum_b)
        pair_t_dif_sum = 0.5 * (t_dif_a + t_dif_b)
        pair_t_dif_dif = 0.5 * (t_dif_a - t_dif_b)
        pair_y_sum = 0.5 * (y_a + y_b)
        pair_y_dif = 0.5 * (y_a - y_b)

        fail_q_sum_sum = valid_mask & (
            (pair_q_sum_sum < float(q_sum_sum_left))
            | (pair_q_sum_sum > float(q_sum_sum_right))
        )
        fail_q_sum_dif = valid_mask & (np.abs(pair_q_sum_dif) > abs(float(q_sum_dif_threshold)))
        fail_q_dif_sum = valid_mask & (np.abs(pair_q_dif_sum) > abs(float(q_dif_sum_threshold)))
        fail_q_dif_dif = valid_mask & (np.abs(pair_q_dif_dif) > abs(float(q_dif_dif_threshold)))
        fail_t_sum_sum = valid_mask & (
            (pair_t_sum_sum < float(t_sum_sum_left))
            | (pair_t_sum_sum > float(t_sum_sum_right))
        )
        fail_t_sum_dif = valid_mask & (np.abs(pair_t_sum_dif) > abs(float(t_sum_dif_threshold)))
        fail_t_dif_sum = valid_mask & (np.abs(pair_t_dif_sum) > abs(float(t_dif_sum_threshold)))
        fail_t_dif_dif = valid_mask & (np.abs(pair_t_dif_dif) > abs(float(t_dif_dif_threshold)))
        fail_y_sum = valid_mask & (
            (pair_y_sum < float(y_sum_left))
            | (pair_y_sum > float(y_sum_right))
        )
        fail_y_dif = valid_mask & (np.abs(pair_y_dif) > abs(float(y_dif_threshold)))
        fail_any = (
            fail_q_sum_sum
            | fail_q_sum_dif
            | fail_q_dif_sum
            | fail_q_dif_dif
            | fail_t_sum_sum
            | fail_t_sum_dif
            | fail_t_dif_sum
            | fail_t_dif_dif
            | fail_y_sum
            | fail_y_dif
        )

        summary["failed_pair_q_sum_sum"] += int(np.count_nonzero(fail_q_sum_sum))
        summary["failed_pair_q_sum_dif"] += int(np.count_nonzero(fail_q_sum_dif))
        summary["failed_pair_q_dif_sum"] += int(np.count_nonzero(fail_q_dif_sum))
        summary["failed_pair_q_dif_dif"] += int(np.count_nonzero(fail_q_dif_dif))
        summary["failed_pair_t_sum_sum"] += int(np.count_nonzero(fail_t_sum_sum))
        summary["failed_pair_t_sum_dif"] += int(np.count_nonzero(fail_t_sum_dif))
        summary["failed_pair_t_dif_sum"] += int(np.count_nonzero(fail_t_dif_sum))
        summary["failed_pair_t_dif_dif"] += int(np.count_nonzero(fail_t_dif_dif))
        summary["failed_pair_y_sum"] += int(np.count_nonzero(fail_y_sum))
        summary["failed_pair_y_dif"] += int(np.count_nonzero(fail_y_dif))
        summary["failed_pair_any"] += int(np.count_nonzero(fail_any))

        if np.any(fail_any):
            q_sum_sum_width = max(abs(float(q_sum_sum_right) - float(q_sum_sum_left)), 1e-9)
            q_sum_dif_scale = max(abs(float(q_sum_dif_threshold)), 1e-9)
            q_dif_sum_scale = max(abs(float(q_dif_sum_threshold)), 1e-9)
            q_dif_dif_scale = max(abs(float(q_dif_dif_threshold)), 1e-9)
            t_sum_sum_width = max(abs(float(t_sum_sum_right) - float(t_sum_sum_left)), 1e-9)
            t_sum_dif_scale = max(abs(float(t_sum_dif_threshold)), 1e-9)
            t_dif_sum_scale = max(abs(float(t_dif_sum_threshold)), 1e-9)
            t_dif_dif_scale = max(abs(float(t_dif_dif_threshold)), 1e-9)
            y_sum_width = max(abs(float(y_sum_right) - float(y_sum_left)), 1e-9)
            y_dif_scale = max(abs(float(y_dif_threshold)), 1e-9)

            q_sum_sum_excess = np.maximum.reduce(
                [
                    np.zeros_like(pair_q_sum_sum, dtype=float),
                    float(q_sum_sum_left) - pair_q_sum_sum,
                    pair_q_sum_sum - float(q_sum_sum_right),
                ]
            ) / q_sum_sum_width
            q_sum_dif_excess = (
                np.maximum(np.abs(pair_q_sum_dif) - abs(float(q_sum_dif_threshold)), 0.0)
                / q_sum_dif_scale
            )
            q_dif_sum_excess = (
                np.maximum(np.abs(pair_q_dif_sum) - abs(float(q_dif_sum_threshold)), 0.0)
                / q_dif_sum_scale
            )
            q_dif_dif_excess = (
                np.maximum(np.abs(pair_q_dif_dif) - abs(float(q_dif_dif_threshold)), 0.0)
                / q_dif_dif_scale
            )
            t_sum_sum_excess = np.maximum.reduce(
                [
                    np.zeros_like(pair_t_sum_sum, dtype=float),
                    float(t_sum_sum_left) - pair_t_sum_sum,
                    pair_t_sum_sum - float(t_sum_sum_right),
                ]
            ) / t_sum_sum_width
            t_sum_dif_excess = (
                np.maximum(np.abs(pair_t_sum_dif) - abs(float(t_sum_dif_threshold)), 0.0)
                / t_sum_dif_scale
            )
            t_dif_sum_excess = (
                np.maximum(np.abs(pair_t_dif_sum) - abs(float(t_dif_sum_threshold)), 0.0)
                / t_dif_sum_scale
            )
            t_dif_dif_excess = (
                np.maximum(np.abs(pair_t_dif_dif) - abs(float(t_dif_dif_threshold)), 0.0)
                / t_dif_dif_scale
            )
            y_sum_excess = np.maximum.reduce(
                [
                    np.zeros_like(pair_y_sum, dtype=float),
                    float(y_sum_left) - pair_y_sum,
                    pair_y_sum - float(y_sum_right),
                ]
            ) / y_sum_width
            y_dif_excess = (
                np.maximum(np.abs(pair_y_dif) - abs(float(y_dif_threshold)), 0.0)
                / y_dif_scale
            )
            pair_severity = (
                q_sum_sum_excess
                + q_sum_dif_excess
                + q_dif_sum_excess
                + q_dif_dif_excess
                + t_sum_sum_excess
                + t_sum_dif_excess
                + t_dif_sum_excess
                + t_dif_dif_excess
                + y_sum_excess
                + y_dif_excess
            )
            pair_severity = np.where(fail_any, pair_severity, 0.0)
            pair_severity = np.where(fail_any & (pair_severity <= 0), 1.0, pair_severity)
            for row_pos in np.flatnonzero(fail_any):
                row_failed_edges.setdefault(int(row_pos), []).append(
                    (plane_a, plane_b, float(pair_severity[row_pos]))
                )

    any_row_affected = np.zeros(len(df_input), dtype=bool)
    summary["flagged_rows"] = len(row_failed_edges)
    for row_pos, edge_list in row_failed_edges.items():
        uncovered_edges = list(edge_list)
        selected_planes: list[int] = []
        while uncovered_edges:
            candidate_scores: dict[int, tuple[int, float]] = {}
            for plane_a, plane_b, severity in uncovered_edges:
                count_a, score_a = candidate_scores.get(plane_a, (0, 0.0))
                candidate_scores[plane_a] = (count_a + 1, score_a + severity)
                count_b, score_b = candidate_scores.get(plane_b, (0, 0.0))
                candidate_scores[plane_b] = (count_b + 1, score_b + severity)

            best_plane = min(
                candidate_scores,
                key=lambda plane_key: (
                    -candidate_scores[plane_key][0],
                    -candidate_scores[plane_key][1],
                    *_task3_plane_order_key(plane_key),
                ),
            )
            selected_planes.append(best_plane)
            uncovered_edges = [
                edge for edge in uncovered_edges
                if best_plane not in edge[:2]
            ]

        summary["max_failed_pairs_in_row"] = max(
            summary["max_failed_pairs_in_row"],
            len(edge_list),
        )
        summary["max_selected_offenders_in_row"] = max(
            summary["max_selected_offenders_in_row"],
            len(selected_planes),
        )
        summary["selected_offender_planes"] += len(selected_planes)
        if len(selected_planes) > 1:
            summary["rows_with_multiple_offenders"] += 1
        for plane_key in selected_planes:
            plane_fail_masks[plane_key][row_pos] = True
            offender_hit_counts[plane_key] += 1

    for plane_key in plane_map:
        fail_mask = plane_fail_masks[plane_key]
        if not np.any(fail_mask):
            continue
        cols = plane_map[plane_key]
        for variable_name in ("T_sum", "T_dif", "Q_sum", "Q_dif", "Y"):
            values = pd.to_numeric(df_input[cols[variable_name]], errors="coerce").fillna(0)
            summary["values_zeroed"] += int((values[fail_mask] != 0).sum())
        any_row_affected |= fail_mask
        df_input.loc[fail_mask, list(cols.values())] = 0

    summary["rows_affected"] = int(np.count_nonzero(any_row_affected))
    top_offenders = sorted(
        (
            (plane_key, count)
            for plane_key, count in offender_hit_counts.items()
            if count > 0
        ),
        key=lambda item: (-item[1], *_task3_plane_order_key(item[0])),
    )[:8]
    if top_offenders:
        top_offenders_str = ", ".join(
            f"P{plane}:{count}"
            for plane, count in top_offenders
        )
        print(
            "[plane-combination-offenders] "
            f"flagged_rows={summary['flagged_rows']} "
            f"selected_offenders={summary['selected_offender_planes']} "
            f"multi_offender_rows={summary['rows_with_multiple_offenders']} "
            f"max_failed_pairs_in_row={summary['max_failed_pairs_in_row']} "
            f"top={top_offenders_str}"
        )

    summary["selected_offender_counts"] = {
        plane_key: int(offender_hit_counts.get(plane_key, 0))
        for plane_key in TASK3_PLANE_KEYS
    }
    return summary


TASK3_PLANE_COMBINATION_OBSERVABLES: tuple[tuple[str, str], ...] = (
    ("q_sum_sum", "Q_sum semisum"),
    ("q_sum_dif", "Q_sum semidifference"),
    ("q_dif_sum", "Q_dif semisum"),
    ("q_dif_dif", "Q_dif semidifference"),
    ("t_sum_sum", "T_sum semisum"),
    ("t_sum_dif", "T_sum semidifference"),
    ("t_dif_sum", "T_dif semisum"),
    ("t_dif_dif", "T_dif semidifference"),
    ("y_sum", "Y semisum"),
    ("y_dif", "Y semidifference"),
)


def _task3_filter_tt_series(df: pd.DataFrame) -> pd.Series:
    for candidate in ("list_tt", "cal_tt", "clean_tt", "raw_tt"):
        if candidate in df.columns:
            return df[candidate].apply(normalize_tt_label).astype(str)
    return pd.Series(["0"] * len(df), index=df.index, dtype=str)


def collect_task3_plane_combination_histogram_payload(
    df_input: pd.DataFrame,
    tt_series: pd.Series,
) -> pd.DataFrame:
    payload_rows: list[pd.DataFrame] = []
    plane_map = collect_task3_plane_final_map(df_input)
    if len(plane_map) < 2:
        return pd.DataFrame(columns=["tt", *[observable for observable, _ in TASK3_PLANE_COMBINATION_OBSERVABLES]])

    tt_series = tt_series.reindex(df_input.index).fillna("0").astype(str)
    for plane_a, plane_b in combinations(sorted(plane_map), 2):
        combo_label = f"{plane_a}{plane_b}"
        cols_a = plane_map[plane_a]
        cols_b = plane_map[plane_b]
        q_sum_a = pd.to_numeric(df_input[cols_a["Q_sum"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_sum_b = pd.to_numeric(df_input[cols_b["Q_sum"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_dif_a = pd.to_numeric(df_input[cols_a["Q_dif"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        q_dif_b = pd.to_numeric(df_input[cols_b["Q_dif"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_sum_a = pd.to_numeric(df_input[cols_a["T_sum"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_sum_b = pd.to_numeric(df_input[cols_b["T_sum"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_dif_a = pd.to_numeric(df_input[cols_a["T_dif"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        t_dif_b = pd.to_numeric(df_input[cols_b["T_dif"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        y_a = pd.to_numeric(df_input[cols_a["Y"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        y_b = pd.to_numeric(df_input[cols_b["Y"]], errors="coerce").fillna(0).to_numpy(dtype=float)

        valid_mask = (
            np.isfinite(q_sum_a)
            & np.isfinite(q_sum_b)
            & np.isfinite(q_dif_a)
            & np.isfinite(q_dif_b)
            & np.isfinite(t_sum_a)
            & np.isfinite(t_sum_b)
            & np.isfinite(t_dif_a)
            & np.isfinite(t_dif_b)
            & np.isfinite(y_a)
            & np.isfinite(y_b)
            & (q_sum_a != 0)
            & (q_sum_b != 0)
            & (q_dif_a != 0)
            & (q_dif_b != 0)
            & (t_sum_a != 0)
            & (t_sum_b != 0)
            & (t_dif_a != 0)
            & (t_dif_b != 0)
            & (y_a != 0)
            & (y_b != 0)
        )
        if not np.any(valid_mask):
            continue

        valid_index = df_input.index[valid_mask]
        tt_values = tt_series.loc[valid_index].to_numpy(dtype=str)
        derived_values = {
            "q_sum_sum": 0.5 * (q_sum_a + q_sum_b),
            "q_sum_dif": 0.5 * (q_sum_a - q_sum_b),
            "q_dif_sum": 0.5 * (q_dif_a + q_dif_b),
            "q_dif_dif": 0.5 * (q_dif_a - q_dif_b),
            "t_sum_sum": 0.5 * (t_sum_a + t_sum_b),
            "t_sum_dif": 0.5 * (t_sum_a - t_sum_b),
            "t_dif_sum": 0.5 * (t_dif_a + t_dif_b),
            "t_dif_dif": 0.5 * (t_dif_a - t_dif_b),
            "y_sum": 0.5 * (y_a + y_b),
            "y_dif": 0.5 * (y_a - y_b),
        }
        payload_rows.append(
            pd.DataFrame(
                {
                    "tt": tt_values,
                    "combo": combo_label,
                    **{
                        observable: values[valid_mask]
                        for observable, values in derived_values.items()
                    },
                }
            )
        )

    if not payload_rows:
        return pd.DataFrame(columns=["tt", *[observable for observable, _ in TASK3_PLANE_COMBINATION_OBSERVABLES]])
    return pd.concat(payload_rows, ignore_index=True)


def _task3_hist_range(
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


def _task3_population_color(label: str) -> tuple[float, float, float, float]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    color_pos = int.from_bytes(digest[:4], "big") / float(2**32 - 1)
    return plt.get_cmap("turbo")(0.08 + 0.84 * color_pos)


def plot_task3_plane_combination_filter_by_tt(
    before_payload: pd.DataFrame,
    after_payload: pd.DataFrame,
    basename_no_ext_value: str,
    fig_idx_value: int,
    base_dir: str,
    *,
    show_plots: bool,
    save_plots: bool,
    plot_list: list[str] | None,
    limits_by_observable: dict[str, tuple[float | None, float | None]],
) -> int:
    tt_labels = []
    for payload in (before_payload, after_payload):
        if "tt" in payload.columns:
            tt_labels.extend(payload["tt"].astype(str).tolist())
    ordered_tts = [
        tt_label for tt_label in TT_COLOR_LABELS
        if tt_label != "0" and tt_label in set(tt_labels)
    ]
    observable_names = [observable for observable, _ in TASK3_PLANE_COMBINATION_OBSERVABLES]
    observable_labels = {observable: label for observable, label in TASK3_PLANE_COMBINATION_OBSERVABLES}
    scatter_max_points = 5000
    rng = np.random.default_rng(0)

    for tt_label in ordered_tts:
        before_tt = before_payload.loc[before_payload.get("tt", pd.Series(dtype=str)).astype(str) == tt_label].copy()
        after_tt = after_payload.loc[after_payload.get("tt", pd.Series(dtype=str)).astype(str) == tt_label].copy()
        if before_tt.empty and after_tt.empty:
            continue

        if len(before_tt) > scatter_max_points:
            before_tt = before_tt.iloc[rng.choice(len(before_tt), size=scatter_max_points, replace=False)].copy()
        if len(after_tt) > scatter_max_points:
            after_tt = after_tt.iloc[rng.choice(len(after_tt), size=scatter_max_points, replace=False)].copy()

        combo_labels = sorted(
            set(before_tt.get("combo", pd.Series(dtype=str)).astype(str))
            | set(after_tt.get("combo", pd.Series(dtype=str)).astype(str))
        )
        combo_labels = [label for label in combo_labels if label and label != "nan"]
        combo_color_map = {label: _task3_population_color(label) for label in combo_labels}
        before_combo_data = {
            label: {
                observable: pd.to_numeric(
                    before_tt.loc[before_tt["combo"] == label, observable],
                    errors="coerce",
                ).to_numpy(dtype=float)
                for observable in observable_names
            }
            for label in combo_labels
        }
        after_combo_data = {
            label: {
                observable: pd.to_numeric(
                    after_tt.loc[after_tt["combo"] == label, observable],
                    errors="coerce",
                ).to_numpy(dtype=float)
                for observable in observable_names
            }
            for label in combo_labels
        }

        n_obs = len(observable_names)
        fig, axes = plt.subplots(n_obs, n_obs, figsize=(2.15 * n_obs, 2.15 * n_obs), constrained_layout=True)
        any_panel_data = False

        axis_ranges: dict[str, tuple[float, float]] = {}
        for observable in observable_names:
            before_values = pd.to_numeric(before_tt.get(observable, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
            after_values = pd.to_numeric(after_tt.get(observable, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
            axis_ranges[observable] = _task3_hist_range(
                before_values,
                after_values,
                limits_by_observable.get(observable, (None, None)),
            )

        for row_idx, y_name in enumerate(observable_names):
            for col_idx, x_name in enumerate(observable_names):
                ax = axes[row_idx, col_idx]
                if col_idx > row_idx:
                    ax.set_axis_off()
                    continue

                before_x = pd.to_numeric(before_tt.get(x_name, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
                before_y = pd.to_numeric(before_tt.get(y_name, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
                after_x = pd.to_numeric(after_tt.get(x_name, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
                after_y = pd.to_numeric(after_tt.get(y_name, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)

                if row_idx == col_idx:
                    before_values = before_x[np.isfinite(before_x)]
                    after_values = after_x[np.isfinite(after_x)]
                    if before_values.size == 0 and after_values.size == 0:
                        ax.set_axis_off()
                        continue
                    any_panel_data = True
                    x_low, x_high = axis_ranges[x_name]
                    bins = np.linspace(x_low, x_high, 60)
                    for combo_label in combo_labels:
                        combo_before = before_combo_data[combo_label][x_name]
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
                    lower_limit, upper_limit = limits_by_observable.get(x_name, (None, None))
                    if lower_limit is not None:
                        ax.axvline(float(lower_limit), color="lightgrey", linestyle="--", linewidth=1.1)
                    if upper_limit is not None:
                        ax.axvline(float(upper_limit), color="lightgrey", linestyle="--", linewidth=1.1)
                    ax.set_xlim(x_low, x_high)
                    ax.set_yscale("log", nonpositive="clip")
                    ax.set_title(observable_labels[x_name], fontsize=9)
                else:
                    before_mask = np.isfinite(before_x) & np.isfinite(before_y)
                    after_mask = np.isfinite(after_x) & np.isfinite(after_y)
                    if not np.any(before_mask) and not np.any(after_mask):
                        ax.set_axis_off()
                        continue
                    any_panel_data = True
                    for combo_label in combo_labels:
                        combo_before_x = before_combo_data[combo_label][x_name]
                        combo_before_y = before_combo_data[combo_label][y_name]
                        combo_after_x = after_combo_data[combo_label][x_name]
                        combo_after_y = after_combo_data[combo_label][y_name]
                        combo_before_mask = np.isfinite(combo_before_x) & np.isfinite(combo_before_y)
                        combo_after_mask = np.isfinite(combo_after_x) & np.isfinite(combo_after_y)
                        if np.any(combo_before_mask):
                            ax.scatter(
                                combo_before_x[combo_before_mask],
                                combo_before_y[combo_before_mask],
                                s=5,
                                alpha=0.05,
                                color=combo_color_map[combo_label],
                                edgecolors="none",
                                rasterized=True,
                            )
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
                    x_low, x_high = axis_ranges[x_name]
                    y_low, y_high = axis_ranges[y_name]
                    ax.set_xlim(x_low, x_high)
                    ax.set_ylim(y_low, y_high)
                    x_limits = limits_by_observable.get(x_name, (None, None))
                    y_limits = limits_by_observable.get(y_name, (None, None))
                    if x_limits[0] is not None:
                        ax.axvline(float(x_limits[0]), color="lightgrey", linestyle="--", linewidth=0.9)
                    if x_limits[1] is not None:
                        ax.axvline(float(x_limits[1]), color="lightgrey", linestyle="--", linewidth=0.9)
                    if y_limits[0] is not None:
                        ax.axhline(float(y_limits[0]), color="lightgrey", linestyle="--", linewidth=0.9)
                    if y_limits[1] is not None:
                        ax.axhline(float(y_limits[1]), color="lightgrey", linestyle="--", linewidth=0.9)

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

        if combo_labels:
            style_handles = [
                mpl.lines.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="Before"),
                mpl.lines.Line2D([0], [0], color="black", linestyle="-", linewidth=1.4, label="After"),
            ]
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

        fig.suptitle(
            f"Task 3 plane-combination filter by TT {tt_label}\n{basename_no_ext_value}",
            fontsize=11,
            y=0.94,
        )
        if save_plots:
            final_filename = f"{fig_idx_value}_plane_combination_filter_by_tt_TT_{tt_label}.png"
            fig_idx_value += 1
            save_fig_path = os.path.join(base_dir, final_filename)
            if plot_list is not None:
                plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
        if show_plots:
            plt.show()
        plt.close(fig)

    return fig_idx_value

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
# Input selection --------------------------------------------------------------
# -----------------------------------------------------------------------------

selected_input_file = CLI_ARGS.input_file_flag or CLI_ARGS.input_file
if selected_input_file:
    user_file_path = selected_input_file
    user_file_selection = True
    print("User provided file path:", user_file_path)
else:
    user_file_selection = False

repo_root = get_repo_root()
station_directory = str(repo_root / "STATIONS" / f"MINGO0{station}")
config_file_directory = str(
    config_root
    / "STAGE_0"
    / "NEW_FILES"
    / "ONLINE_RUN_DICTIONARY"
    / f"STATION_{station}"
)

# Define input file path ------------------------------------------------------------------
input_file_config_path = os.path.join(config_file_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    print("Searching input configuration file:", input_file_config_path)
    
    # It is a csv
    input_file = pd.read_csv(input_file_config_path, skiprows=1)
    
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
    default_z_positions = config.get("default_z_positions", [0, 150, 300, 450])
    if isinstance(default_z_positions, (list, tuple)) and len(default_z_positions) >= 4:
        z_1, z_2, z_3, z_4 = default_z_positions[:4]
    else:
        z_1, z_2, z_3, z_4 = 0, 150, 300, 450

self_trigger = False

print("Creating the necessary directories...")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

# Define base working directory
home_directory = str(repo_root.parent)
station_directory = str(repo_root / "STATIONS" / f"MINGO0{station}")
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
debug_fig_idx = 1

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

raw_files = set(os.listdir(raw_directory))
unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))

# -----------------------------------------------------------------------------
# Events per second metadata helpers ------------------------------------------
# -----------------------------------------------------------------------------

not_use_q_semisum = False

stratos_save = config["stratos_save"]
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
) = resolve_task3_plot_options(config)
limit_number = config.get("limit_number", None)
limit = limit_number is not None

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering

# Time calibration

# Y position

# RPC variables
y_new_method = config["y_new_method"]

# Alternative

# TimTrack

# Validation

limit_number = config.get("limit_number", None)
limit = limit_number is not None

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
T_sum_RPC_left = config["T_sum_RPC_left"]
T_sum_RPC_right = config["T_sum_RPC_right"]
T_dif_RPC_abs = abs(float(config.get("T_dif_RPC_abs", config.get("T_dif_RPC_right", 0.8))))
T_dif_RPC_right = T_dif_RPC_abs
T_dif_RPC_left = -T_dif_RPC_abs
Q_RPC_left = config["Q_RPC_left"]
Q_RPC_right = config["Q_RPC_right"]
Q_dif_RPC_abs = abs(float(config.get("Q_dif_RPC_abs", config.get("Q_dif_RPC_right", 4))))
Q_dif_RPC_right = Q_dif_RPC_abs
Q_dif_RPC_left = -Q_dif_RPC_abs
Y_RPC_abs = abs(float(config.get("Y_RPC_abs", config.get("Y_RPC_right", 200))))
Y_RPC_right = Y_RPC_abs
Y_RPC_left = -Y_RPC_abs

# Alternative fitter filter
det_pos_filter = config.get("det_pos_filter", 800)
det_theta_left_filter = config.get("det_theta_left_filter", 0)
det_theta_right_filter = config.get("det_theta_right_filter", 1.5708)
det_phi_filter_abs = abs(float(config.get("det_phi_filter_abs", config.get("det_phi_right_filter", 3.141592))))
det_phi_right_filter = det_phi_filter_abs
det_phi_left_filter = -det_phi_filter_abs
det_slowness_filter_left = config.get("det_slowness_filter_left", -0.02)
det_slowness_filter_right = config.get("det_slowness_filter_right", 0.02)

# TimTrack filter

# Fitting comparison

# Calibrations

# Pedestal charge calibration

# Front-back charge

# Variables to modify
beta = config.get("beta", 1)
strip_speed_factor_of_c = config["strip_speed_factor_of_c"]

# X
strip_length = config.get("strip_length", 300)
narrow_strip = config["narrow_strip"]
wide_strip = config["wide_strip"]

# Timtrack parameters
anc_std = config["anc_std"]

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

self_trigger = False

not_use_q_semisum = False

stratos_save = config["stratos_save"]
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
) = resolve_task3_plot_options(config)
limit_number = config.get("limit_number", None)
limit = limit_number is not None

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering

# Time calibration

# Y position

# RPC variables
y_new_method = config["y_new_method"]

# Alternative

# TimTrack

# Validation

limit_number = config.get("limit_number", None)
limit = limit_number is not None

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
T_sum_RPC_left = config["T_sum_RPC_left"]
T_sum_RPC_right = config["T_sum_RPC_right"]
T_dif_RPC_abs = abs(float(config.get("T_dif_RPC_abs", config.get("T_dif_RPC_right", 0.8))))
T_dif_RPC_right = T_dif_RPC_abs
T_dif_RPC_left = -T_dif_RPC_abs
Q_RPC_left = config["Q_RPC_left"]
Q_RPC_right = config["Q_RPC_right"]
Q_dif_RPC_abs = abs(float(config.get("Q_dif_RPC_abs", config.get("Q_dif_RPC_right", 4))))
Q_dif_RPC_right = Q_dif_RPC_abs
Q_dif_RPC_left = -Q_dif_RPC_abs
Y_RPC_abs = abs(float(config.get("Y_RPC_abs", config.get("Y_RPC_right", 200))))
Y_RPC_right = Y_RPC_abs
Y_RPC_left = -Y_RPC_abs

# Alternative fitter filter
det_pos_filter = config.get("det_pos_filter", 800)
det_theta_left_filter = config.get("det_theta_left_filter", 0)
det_theta_right_filter = config.get("det_theta_right_filter", 1.5708)
det_phi_filter_abs = abs(float(config.get("det_phi_filter_abs", config.get("det_phi_right_filter", 3.141592))))
det_phi_right_filter = det_phi_filter_abs
det_phi_left_filter = -det_phi_filter_abs
det_slowness_filter_left = config.get("det_slowness_filter_left", -0.02)
det_slowness_filter_right = config.get("det_slowness_filter_right", 0.02)

# TimTrack filter

# Fitting comparison

# Calibrations

# Pedestal charge calibration

# Front-back charge

# Variables to modify
beta = config.get("beta", 1)
strip_speed_factor_of_c = config["strip_speed_factor_of_c"]

# X
strip_length = config.get("strip_length", 300)
narrow_strip = config["narrow_strip"]
wide_strip = config["wide_strip"]

# Timtrack parameters
anc_std = config["anc_std"]

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

charge_per_strip_plot_threshold = config["charge_per_strip_plot_threshold"]
charge_per_plane_plot_threshold = config["charge_per_plane_plot_threshold"]
list_tt_qsum_pair_plot_mode = str(config.get("list_tt_qsum_pair_plot_mode", "hexbin")).strip().lower()
if list_tt_qsum_pair_plot_mode not in {"hexbin", "scatter"}:
    print(
        "Warning: invalid list_tt_qsum_pair_plot_mode="
        f"{list_tt_qsum_pair_plot_mode!r}; using 'hexbin'."
    )
    list_tt_qsum_pair_plot_mode = "hexbin"
_plot_charge_cap_raw = config.get("list_tt_qsum_pair_plot_charge_cap", None)
try:
    if _plot_charge_cap_raw in (None, ""):
        list_tt_qsum_pair_plot_charge_cap = None
    else:
        list_tt_qsum_pair_plot_charge_cap = float(_plot_charge_cap_raw)
except (TypeError, ValueError):
    print(
        "Warning: invalid list_tt_qsum_pair_plot_charge_cap="
        f"{_plot_charge_cap_raw!r}; using automatic range."
    )
    list_tt_qsum_pair_plot_charge_cap = None
if (
    list_tt_qsum_pair_plot_charge_cap is not None
    and (not np.isfinite(list_tt_qsum_pair_plot_charge_cap) or list_tt_qsum_pair_plot_charge_cap <= 0)
):
    print(
        "Warning: non-positive list_tt_qsum_pair_plot_charge_cap="
        f"{list_tt_qsum_pair_plot_charge_cap!r}; using automatic range."
    )
    list_tt_qsum_pair_plot_charge_cap = None

_global_charge_limit_raw = config.get("list_tt_charge_limit", None)
try:
    if _global_charge_limit_raw in (None, ""):
        list_tt_charge_limit = None
    else:
        list_tt_charge_limit = float(_global_charge_limit_raw)
except (TypeError, ValueError):
    print(
        "Warning: invalid list_tt_charge_limit="
        f"{_global_charge_limit_raw!r}; ignoring global charge limit."
    )
    list_tt_charge_limit = None
if list_tt_charge_limit is not None and (
    not np.isfinite(list_tt_charge_limit) or list_tt_charge_limit <= 0
):
    print(
        "Warning: non-positive list_tt_charge_limit="
        f"{list_tt_charge_limit!r}; ignoring global charge limit."
    )
    list_tt_charge_limit = None

_station_charge_limits_raw = config.get("list_tt_charge_limit_by_station", {})
list_tt_charge_limit_station = None
if isinstance(_station_charge_limits_raw, dict):
    station_norm = str(station).strip()
    station_candidates = {
        station_norm.lower(),
        station_norm.zfill(2).lower(),
        f"mingo0{station_norm}".lower(),
        f"mingo{station_norm.zfill(2)}".lower(),
        f"station_{station_norm}".lower(),
        f"station_{station_norm.zfill(2)}".lower(),
    }
    for raw_key, raw_value in _station_charge_limits_raw.items():
        key_norm = str(raw_key).strip().lower()
        if key_norm not in station_candidates:
            continue
        try:
            parsed_limit = float(raw_value)
        except (TypeError, ValueError):
            print(
                "Warning: invalid station charge limit for "
                f"{raw_key!r}: {raw_value!r}; ignoring this entry."
            )
            continue
        if np.isfinite(parsed_limit) and parsed_limit > 0:
            list_tt_charge_limit_station = parsed_limit
            break
        print(
            "Warning: non-positive station charge limit for "
            f"{raw_key!r}: {raw_value!r}; ignoring this entry."
        )
elif _station_charge_limits_raw not in (None, ""):
    print(
        "Warning: list_tt_charge_limit_by_station must be a mapping; "
        f"got {_station_charge_limits_raw!r}."
    )

list_tt_charge_limit_effective = list_tt_charge_limit_station
if list_tt_charge_limit_effective is None:
    list_tt_charge_limit_effective = list_tt_charge_limit

if list_tt_qsum_pair_plot_charge_cap is not None and list_tt_charge_limit_effective is not None:
    list_tt_qsum_pair_plot_charge_cap = min(
        float(list_tt_qsum_pair_plot_charge_cap),
        float(list_tt_charge_limit_effective),
    )
elif list_tt_qsum_pair_plot_charge_cap is None and list_tt_charge_limit_effective is not None:
    list_tt_qsum_pair_plot_charge_cap = float(list_tt_charge_limit_effective)

if list_tt_charge_limit_effective is not None:
    print(
        "Task 3: effective charge limit cap for station "
        f"{station} set to {float(list_tt_charge_limit_effective):g}."
    )

_list_tt_charge_thresholds_raw = config.get("list_tt_charge_thresholds", [0, 5, 10, 20, 50])
if isinstance(_list_tt_charge_thresholds_raw, (int, float, str)):
    _list_tt_charge_thresholds_raw = [_list_tt_charge_thresholds_raw]
list_tt_charge_thresholds: list[float] = []
for raw_threshold in _list_tt_charge_thresholds_raw:
    try:
        threshold_value = float(raw_threshold)
    except (TypeError, ValueError):
        print(f"Warning: invalid list_tt_charge_thresholds entry {raw_threshold!r}; skipping.")
        continue
    if not np.isfinite(threshold_value) or threshold_value < 0:
        print(f"Warning: non-finite or negative list_tt_charge_thresholds entry {raw_threshold!r}; skipping.")
        continue
    if threshold_value not in list_tt_charge_thresholds:
        list_tt_charge_thresholds.append(threshold_value)
if not list_tt_charge_thresholds:
    list_tt_charge_thresholds = [0.0, 5.0, 10.0, 20.0, 50.0]
manual_list_tt_charge_thresholds = list(list_tt_charge_thresholds)
list_tt_charge_threshold_mode = str(config.get("list_tt_charge_threshold_mode", "manual")).strip().lower()
if list_tt_charge_threshold_mode not in {"manual", "auto", "quantile"}:
    print(
        "Warning: invalid list_tt_charge_threshold_mode="
        f"{list_tt_charge_threshold_mode!r}; using 'manual'."
    )
    list_tt_charge_threshold_mode = "manual"

_auto_thr_quantile_raw = config.get("list_tt_charge_threshold_auto_quantile", 95.0)
try:
    list_tt_charge_threshold_auto_quantile = float(_auto_thr_quantile_raw)
except (TypeError, ValueError):
    list_tt_charge_threshold_auto_quantile = 95.0
if not np.isfinite(list_tt_charge_threshold_auto_quantile):
    list_tt_charge_threshold_auto_quantile = 95.0
list_tt_charge_threshold_auto_quantile = float(
    np.clip(list_tt_charge_threshold_auto_quantile, 50.0, 99.999)
)

_auto_thr_steps_raw = config.get("list_tt_charge_threshold_auto_steps", 10)
try:
    list_tt_charge_threshold_auto_steps = int(_auto_thr_steps_raw)
except (TypeError, ValueError):
    list_tt_charge_threshold_auto_steps = 10
list_tt_charge_threshold_auto_steps = max(2, list_tt_charge_threshold_auto_steps)

full_topology_min_baseline_count = int(config.get("full_topology_min_baseline_count", 30))
full_topology_min_baseline_count = max(1, full_topology_min_baseline_count)
full_topology_max_masks = int(config.get("full_topology_max_masks", 6))
full_topology_max_masks = max(1, full_topology_max_masks)
full_topology_max_patterns_per_mask = int(config.get("full_topology_max_patterns_per_mask", 8))
full_topology_max_patterns_per_mask = max(1, full_topology_max_patterns_per_mask)
tsum_window_hist_bins = int(config.get("tsum_window_hist_bins", 60))
tsum_window_hist_bins = max(20, tsum_window_hist_bins)
_tsum_window_percentiles_raw = config.get("tsum_window_percentiles", [50, 68, 90, 95])
if isinstance(_tsum_window_percentiles_raw, (int, float, str)):
    _tsum_window_percentiles_raw = [_tsum_window_percentiles_raw]
tsum_window_percentiles: list[float] = []
for _raw_pct in _tsum_window_percentiles_raw:
    try:
        _pct = float(_raw_pct)
    except (TypeError, ValueError):
        continue
    if 0 < _pct < 100 and _pct not in tsum_window_percentiles:
        tsum_window_percentiles.append(_pct)
if not tsum_window_percentiles:
    tsum_window_percentiles = [50.0, 68.0, 90.0, 95.0]
_tsum_window_reference_raw = config.get("tsum_window_reference_ns", 4.0)
try:
    tsum_window_reference_ns = float(_tsum_window_reference_raw)
except (TypeError, ValueError):
    tsum_window_reference_ns = 4.0
if not np.isfinite(tsum_window_reference_ns) or tsum_window_reference_ns <= 0:
    tsum_window_reference_ns = 4.0
_tsum_window_cut_values_raw = config.get("tsum_window_cut_values_ns", [2.0, 4.0, 6.0, 8.0])
if isinstance(_tsum_window_cut_values_raw, (int, float, str)):
    _tsum_window_cut_values_raw = [_tsum_window_cut_values_raw]
tsum_window_cut_values_ns: list[float] = []
for _raw_cut in _tsum_window_cut_values_raw:
    try:
        _cut = float(_raw_cut)
    except (TypeError, ValueError):
        continue
    if np.isfinite(_cut) and _cut > 0 and _cut not in tsum_window_cut_values_ns:
        tsum_window_cut_values_ns.append(_cut)
if not tsum_window_cut_values_ns:
    tsum_window_cut_values_ns = [2.0, 4.0, 6.0, 8.0]
tsum_window_max_plane_combinations = int(config.get("tsum_window_max_plane_combinations", 8))
tsum_window_max_plane_combinations = max(1, tsum_window_max_plane_combinations)
tsum_window_combo_min_count = int(config.get("tsum_window_combo_min_count", 120))
tsum_window_combo_min_count = max(10, tsum_window_combo_min_count)
plane_charge_fraction_max_panels = int(config.get("plane_charge_fraction_max_panels", 6))
plane_charge_fraction_max_panels = max(1, plane_charge_fraction_max_panels)
plane_charge_fraction_scatter_max_points = int(config.get("plane_charge_fraction_scatter_max_points", 120000))
plane_charge_fraction_scatter_max_points = max(2000, plane_charge_fraction_scatter_max_points)

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

else:
    T_F_left_pre_cal = T_side_left_pre_cal_default  #-130
    T_F_right_pre_cal = T_side_right_pre_cal_default

    T_B_left_pre_cal = T_side_left_pre_cal_default
    T_B_right_pre_cal = T_side_right_pre_cal_default

T_F_left_pre_cal_ST = T_side_left_pre_cal_ST  #-115
T_F_right_pre_cal_ST = T_side_right_pre_cal_ST
T_B_left_pre_cal_ST = T_side_left_pre_cal_ST
T_B_right_pre_cal_ST = T_side_right_pre_cal_ST

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

# Create ALL directories if they don't already exist
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

if user_file_selection:
    processing_file_path = user_file_path
    file_name = os.path.basename(user_file_path)
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
                    print("Warning: No files to process in COMPLETED after normalization.")
                    sys.exit(0)
            else:
                print(
                    "Warning: No files to process in UNPROCESSED or PROCESSING, and COMPLETED reanalysis is disabled."
                )
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
                print(
                    "Warning: No files to process in UNPROCESSED or PROCESSING, and COMPLETED reanalysis is disabled."
                )
                sys.exit(0)

        else:
            print("Warning: No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")
            sys.exit(0)

# This is for all cases
file_path = processing_file_path
if save_plots:
    os.makedirs(base_directories["figure_directory"], exist_ok=True)

the_filename = os.path.basename(file_path)
print(f"File to process: {the_filename}")

basename_no_ext, file_extension = os.path.splitext(the_filename)
# Take basename of IN_PATH without extension and witouth the 'calibrated_' prefix
basename_no_ext = the_filename.replace("calibrated_", "").replace(".parquet", "")

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

is_simulated_file = basename_no_ext.startswith("mi00")
if is_simulated_file:
    if simulated_param_hash:
        print(f"Simulated param_hash resolved: {simulated_param_hash}")
        global_variables["param_hash"] = simulated_param_hash
    else:
        print("Warning: Simulated param_hash missing; default z_positions will be used.")

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
        param_hash=str(global_variables.get("param_hash", "")),
    )

left_limit_time = pd.to_datetime("1-1-2000", format='%d-%m-%Y')
right_limit_time = pd.to_datetime("1-1-2100", format='%d-%m-%Y')

if limit:
    print(f'Taking the first {limit_number} rows.')

# Read the data file into a DataFrame
KEY = "df"

# Load dataframe
working_df = pd.read_parquet(file_path, engine="pyarrow")
working_df = working_df.rename(columns=lambda col: col.replace("_diff_", "_dif_"))
if "event_id" not in working_df.columns:
    print("Warning: 'event_id' missing in Task 3 input; reconstructing from current row order.")
    working_df.insert(0, "event_id", np.arange(len(working_df), dtype=np.int64))
# Ensure param_hash is persisted for downstream tasks.
if "param_hash" not in working_df.columns:
    working_df["param_hash"] = str(simulated_param_hash) if simulated_param_hash else ""
elif simulated_param_hash:
    _ph_series = working_df["param_hash"]
    _ph_missing = _ph_series.isna()
    try:
        _ph_missing |= _ph_series.astype(str).str.strip().eq("")
    except Exception as exc:
        print(f"Warning: param_hash validation fallback used due to error: {exc}")
    if _ph_missing.any():
        working_df.loc[_ph_missing, "param_hash"] = str(simulated_param_hash)
print(f"Cleaned dataframe reloaded from: {file_path}")
print("Columns loaded from parquet:")
for col in working_df.columns:
    print(f" - {col}")

if task3_plot_enabled("incoming_parquet_main_columns_debug"):
    incoming_patterns = [
        re.compile(r"^T\d+_T_(sum|dif)_\d+$"),
        re.compile(r"^Q\d+_Q_(sum|dif)_\d+$"),
        re.compile(r"^T\d+_[FB]_\d+$"),
        re.compile(r"^Q\d+_[FB]_\d+$"),
    ]
    main_cols = [
        col
        for col in working_df.columns
        if any(pattern.match(col) for pattern in incoming_patterns)
    ]
    main_cols.extend([col for col in ("raw_tt", "clean_tt", "cal_tt") if col in working_df.columns])
    seen = set()
    main_cols = [col for col in main_cols if not (col in seen or seen.add(col))]
    if main_cols:
        debug_fig_idx = plot_debug_histograms(
            working_df,
            main_cols,
            thresholds=None,
            title=f"Task 3 incoming parquet: main columns [NON-TUNABLE] (station {station})",
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
            max_cols_per_fig=20,
        )

cal_tt_columns: dict[int, list[str]] = {}
for plane in range(1, 5):
    cal_tt_columns[plane] = [
        f"T{plane}_T_sum_{strip}" for strip in range(1, 5) if f"T{plane}_T_sum_{strip}" in working_df.columns
    ] + [
        f"T{plane}_T_dif_{strip}" for strip in range(1, 5) if f"T{plane}_T_dif_{strip}" in working_df.columns
    ] + [
        f"Q{plane}_Q_sum_{strip}" for strip in range(1, 5) if f"Q{plane}_Q_sum_{strip}" in working_df.columns
    ] + [
        f"Q{plane}_Q_dif_{strip}" for strip in range(1, 5) if f"Q{plane}_Q_dif_{strip}" in working_df.columns
    ]

# Keep cal_tt from Task 2 when present; compute only if missing.
if "cal_tt" not in working_df.columns:
    working_df = compute_tt(working_df, "cal_tt", cal_tt_columns)
else:
    working_df.loc[:, "cal_tt"] = (
        pd.to_numeric(working_df["cal_tt"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
original_number_of_events = len(working_df)
print(f"Original number of events in the dataframe: {original_number_of_events}")
if status_execution_date is not None:
    update_status_progress(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        completion_fraction=0.5,
        param_hash=str(global_variables.get("param_hash", "")),
    )

# --- Continue your calibration or analysis code here ---
# e.g.:
# run_calibration(working_df)

# Note that the middle between start and end time could also be taken. This is for calibration storage.
if "datetime" in working_df.columns:
    datetime_series = pd.to_datetime(working_df["datetime"], errors="coerce").dropna()
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

print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print(f"------------- Starting date is {save_filename_suffix} -------------------") # This is longer so it displays nicely
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Defining the directories that will store the data
save_full_filename = f"full_list_events_{save_filename_suffix}.txt"
save_filename = f"list_events_{save_filename_suffix}.txt"
save_pdf_filename = f"mingo{str(station).zfill(2)}_task3_{save_filename_suffix}.pdf"

save_pdf_path = os.path.join(base_directories["pdf_directory"], save_pdf_filename)

# -------------------------------------------------------------------------------
# ------------ Input file and data managing to select configuration -------------
# -------------------------------------------------------------------------------

is_simulated_file = basename_no_ext.startswith("mi00")
used_input_file = False
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
    used_input_file = True
    # Ensure `start` and `end` columns are in datetime format
    input_file["start"] = pd.to_datetime(input_file["start"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = pd.to_datetime(input_file["end"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = input_file["end"].fillna(pd.to_datetime('now'))
    start_day = pd.to_datetime(start_time).normalize()
    end_day = pd.to_datetime(end_time).normalize()
    input_file["start_day"] = input_file["start"].dt.normalize()
    input_file["end_day"] = input_file["end"].dt.normalize()
    matching_confs = input_file[(input_file["start_day"] <= start_day) & (input_file["end_day"] >= end_day)]
    print(matching_confs)
    
    if not matching_confs.empty:
        if len(matching_confs) > 1:
            print(f"Warning:\nMultiple configurations match the date range\n{start_time} to {end_time}.\nTaking the first one.")
        selected_conf = matching_confs.iloc[0]
        print(f"Selected configuration: {selected_conf['conf']}")
        z_positions = np.array([selected_conf.get(f"P{i}", np.nan) for i in range(1, 5)])
        found_matching_conf = True
        print(selected_conf['conf'])
        z_source = f"input_file_conf_{selected_conf.get('conf')}"
    else:
        print("Warning: No matching configuration for the date range; selecting closest configuration.")
        before = input_file[input_file["start_day"] <= end_day].sort_values("start_day", ascending=False)
        if not before.empty:
            selected_conf = before.iloc[0]
        else:
            selected_conf = input_file.sort_values("start", ascending=True).iloc[0]
        print(f"Selected configuration: {selected_conf['conf']}")
        z_positions = np.array([selected_conf.get(f"P{i}", np.nan) for i in range(1, 5)])
        found_matching_conf = True
        z_source = f"input_file_closest_conf_{selected_conf.get('conf')}"
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

# Persist plane z-values in the TASK_3 parquet so TASK_4 can reuse them
# without repeating configuration resolution.
for plane_index, z_value in enumerate(z_positions, start=1):
    z_col = f"z_P{plane_index}"
    z_float = float(z_value)
    working_df[z_col] = z_float
    global_variables[z_col] = z_float

_prof["s_data_read_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("---------------- Binary topology of active strips --------------------")
print("----------------------------------------------------------------------")

# Collect new columns in a dict first
active_strip_cols = {}

for plane_id in range(1, 5):
    cols = [f'Q{plane_id}_Q_sum_{i}' for i in range(1, 5)]
    Q_plane = working_df[cols].values  # shape (N, 4)
    active_strips_binary = (Q_plane > 1).astype(int)
    binary_strings = [''.join(map(str, row)) for row in active_strips_binary]
    active_strip_cols[f'active_strips_P{plane_id}'] = binary_strings

# Concatenate all new columns at once (column-wise)
working_df = pd.concat([working_df, pd.DataFrame(active_strip_cols, index=working_df.index)], axis=1)

# Print check
print("Active strips per plane calculated.")
print(working_df[['active_strips_P1', 'active_strips_P2', 'active_strips_P3', 'active_strips_P4']].head())

# Store counts of each active strip pattern (per plane) into metadata
active_patterns = [
    "1000", "0100", "0010", "0001",
    "1100", "0110", "0011", "1010",
    "1001", "0101", "1110", "1011",
    "0111", "1101", "1111",
]
for plane_id in range(1, 5):
    col_name = f"active_strips_P{plane_id}"
    if col_name not in working_df.columns:
        continue
    counts = working_df[col_name].value_counts()
    for pattern in active_patterns:
        global_variables[f"{col_name}_{pattern}_count"] = int(counts.get(pattern, 0))

# ---------------------------------------------------------------------------
# Adj / Dis classification
# A plane pattern is DISPERSED when there is at least one inactive strip (0)
# between two active strips (1), i.e. the active bits are not contiguous.
# Method: strip leading/trailing 0s; if the result still contains a '0', the
# pattern is dispersed.  "0000" and all single-strip patterns are always adj.
# An event is classified "dis" if any plane carries a dispersed pattern;
# otherwise it is "adj".
# ---------------------------------------------------------------------------
_DISPERSED_STRIP_PATTERNS: frozenset[str] = frozenset(
    bits
    for val in range(16)
    for bits in [format(val, "04b")]
    if (
        (fi := bits.find("1")) != -1
        and (li := bits.rfind("1")) > fi
        and "0" in bits[fi : li + 1]
    )
)
# Sanity check (expected: {"0101","1001","1010","1011","1101"})
assert _DISPERSED_STRIP_PATTERNS == {"0101", "1001", "1010", "1011", "1101"}, (
    f"Unexpected dispersed set: {_DISPERSED_STRIP_PATTERNS}"
)

_adj_dis_flags = np.zeros(len(working_df), dtype=bool)  # True = dispersed
for _p in range(1, 5):
    _col = f"active_strips_P{_p}"
    if _col in working_df.columns:
        _adj_dis_flags |= working_df[_col].isin(_DISPERSED_STRIP_PATTERNS).to_numpy()

working_df["adj_dis"] = np.where(_adj_dis_flags, "dis", "adj")

_adj_dis_counts = working_df["adj_dis"].value_counts()
_n_adj = int(_adj_dis_counts.get("adj", 0))
_n_dis = int(_adj_dis_counts.get("dis", 0))
_n_total_adj_dis = _n_adj + _n_dis
global_variables["adj_count"] = _n_adj
global_variables["dis_count"] = _n_dis
global_variables["dis_fraction"] = round(_n_dis / _n_total_adj_dis, 6) if _n_total_adj_dis > 0 else 0.0
print(f"adj_dis column added: adj={_n_adj:,}  dis={_n_dis:,}  dis_fraction={global_variables['dis_fraction']:.4f}")

cal_strip_patterns = build_task3_full_strip_pattern_series(working_df)
store_pattern_rates(pattern_metadata, cal_strip_patterns, "cal_strip_pattern", working_df)

if task3_plot_enabled("active_strip_patterns_overview"):

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True, sharey=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    y_max = 0

    # First pass to determine global y-axis limit
    event_counts_list = []
    for i in [1, 2, 3, 4]:
        counts = working_df[f'active_strips_P{i}'].value_counts()
        counts = counts[counts.index != '0000']
        event_counts_list.append(counts)
        if not counts.empty:
            y_max = max(y_max, counts.max())
    
    # Get global label order from P1 (or any consistent source)
    label_order = working_df['active_strips_P1'].value_counts().drop('0000', errors='ignore').index.tolist()

    # Second pass to plot
    for i, ax in zip([1, 2, 3, 4], axes):
        event_counts_filt = event_counts_list[i - 1]
        event_counts_filt = event_counts_filt.reindex(label_order, fill_value=0)

        # event_counts_filt.plot(kind='bar', ax=ax, color=colors[i - 1], alpha=0.7)
        event_counts_filt.plot(ax=ax, color=colors[i - 1], alpha=0.7)
        ax.set_title(f'Plane {i}', fontsize=12)
        ax.set_ylabel('Counts')
        ax.set_ylim(0, y_max * 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelrotation=45)

    axes[-1].set_xlabel('Active Strip Pattern')
    plt.tight_layout()

    if save_plots:
        final_filename = f'{fig_idx}_filtered_active_strips_all_planes.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()

_prof["s_topology_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("----------------- Some more tests (multi-strip data) -----------------")
print("----------------------------------------------------------------------")

if task3_plot_enabled("multi_strip_pair_diagnostics"):

    # Build shared auto-limits for Q_diff/Q_sum-like panels across all planes/patterns.
    qsum_normdiff_x_all = []
    qsum_normdiff_y_all = []
    for i_plane in range(1, 5):
        active_col = f'active_strips_P{i_plane}'
        q_sum_cols_for_limits = [f'Q{i_plane}_Q_sum_{j+1}' for j in range(4)]
        patterns_for_limits = working_df[active_col].unique()
        multi_patterns_for_limits = [
            p for p in patterns_for_limits if p != '0000' and p.count('1') > 1
        ]

        for pattern in multi_patterns_for_limits:
            active_strips = [idx for idx, c in enumerate(pattern) if c == '1']
            if len(active_strips) != 2:
                continue

            mask = working_df[active_col] == pattern
            if mask.sum() == 0:
                continue

            for i_strip, j_strip in combinations(active_strips, 2):
                xi = pd.to_numeric(
                    working_df.loc[mask, q_sum_cols_for_limits[i_strip]],
                    errors='coerce',
                ).to_numpy(dtype=float)
                yi = pd.to_numeric(
                    working_df.loc[mask, q_sum_cols_for_limits[j_strip]],
                    errors='coerce',
                ).to_numpy(dtype=float)

                denom = (xi + yi) / 2.0
                valid = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(denom) & (denom != 0)
                if not np.any(valid):
                    continue

                x_sum = denom[valid]
                y_norm_diff = (xi[valid] - yi[valid]) / (2.0 * x_sum)
                finite = np.isfinite(x_sum) & np.isfinite(y_norm_diff)
                if np.any(finite):
                    qsum_normdiff_x_all.append(x_sum[finite])
                    qsum_normdiff_y_all.append(y_norm_diff[finite])

    qsum_normdiff_shared_limits = None
    if qsum_normdiff_x_all and qsum_normdiff_y_all:
        x_all = np.concatenate(qsum_normdiff_x_all)
        y_all = np.concatenate(qsum_normdiff_y_all)
        x_left, x_right = np.nanpercentile(x_all, [0.5, 99.5])
        y_bottom, y_top = np.nanpercentile(y_all, [0.5, 99.5])

        if not np.isfinite(x_left) or not np.isfinite(x_right) or x_left == x_right:
            x_min = float(np.nanmin(x_all)) if x_all.size else 0.0
            x_max = float(np.nanmax(x_all)) if x_all.size else 1.0
            x_pad = max(1e-6, 0.05 * max(abs(x_min), abs(x_max), 1.0))
            x_left, x_right = x_min - x_pad, x_max + x_pad
        if not np.isfinite(y_bottom) or not np.isfinite(y_top) or y_bottom == y_top:
            y_min = float(np.nanmin(y_all)) if y_all.size else -1.0
            y_max = float(np.nanmax(y_all)) if y_all.size else 1.0
            y_pad = max(1e-6, 0.10 * max(abs(y_min), abs(y_max), 1.0))
            y_bottom, y_top = y_min - y_pad, y_max + y_pad

        qsum_normdiff_shared_limits = (
            float(x_left),
            float(x_right),
            float(y_bottom),
            float(y_top),
        )

    for i_plane in range(1, 5):
        active_col = f'active_strips_P{i_plane}'
        print(f"\n--- Plane {i_plane} ---")

        # Column names
        T_sum_cols = [f'T{i_plane}_T_sum_{j+1}' for j in range(4)]
        T_dif_cols = [f'T{i_plane}_T_dif_{j+1}' for j in range(4)]
        Q_sum_cols = [f'Q{i_plane}_Q_sum_{j+1}' for j in range(4)]
        Q_dif_cols = [f'Q{i_plane}_Q_dif_{j+1}' for j in range(4)]

        variable_sets = [
            ('T_sum', T_sum_cols),
            ('T_diff', T_dif_cols),
            ('Q_sum', Q_sum_cols),
            ('Q_dif', Q_dif_cols)
        ]

        patterns = working_df[active_col].unique()
        multi_patterns = [p for p in patterns if p != '0000' and p.count('1') > 1]

        for pattern in multi_patterns:
            active_strips = [i for i, c in enumerate(pattern) if c == '1']
            if len(active_strips) != 2:
                continue

            mask = working_df[active_col] == pattern
            n_events = mask.sum()
            if n_events == 0:
                continue

            print(f"Pattern {pattern} ({n_events} events):")

            for i, j in combinations(active_strips, 2):
                fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=False, sharey=False)

                for col_idx, (var_label, cols) in enumerate(variable_sets):
                    xi = working_df.loc[mask, cols[i]].values
                    yi = working_df.loc[mask, cols[j]].values

                    # Row 0: xi vs yi
                    ax = axs[0, col_idx]
                    plot_label = var_label
                    
                    if var_label == "T_sum":
                        lim_left = -125 # -125
                        lim_right = -100 # -100
                    elif var_label == "T_diff":
                        lim_left = -1
                        lim_right = 1
                        
                        error = np.std(yi - xi)
                        plot_label += f', {error:.2f} ns'
                        
                    elif var_label == "Q_sum":
                        lim_left = 0
                        lim_right = 60
                    elif var_label == "Q_dif":
                        lim_left = -1
                        lim_right = 1
                    else:
                        print(f"Unknown variable label: {var_label}")
                        continue
                    
                    ax.scatter(xi, yi, alpha=0.5, s=10, label = plot_label)
                    
                    ax.set_xlim(lim_left, lim_right)
                    ax.set_ylim(lim_left, lim_right)
                    ax.plot([lim_left, lim_right], [lim_left, lim_right], 'k--', lw=1, label='y = x')
                    ax.set_xlabel(f'{var_label} Strip {i+1}')
                    ax.set_ylabel(f'{var_label} Strip {j+1}')
                    ax.set_title(f'{var_label}: Strip {i+1} vs {j+1}')
                    ax.set_aspect('equal', adjustable='box')
                    ax.grid(True)
                    ax.legend()

                    # Row 1: (xi + yi) vs (xi - yi) / (xi + yi)
                    ax = axs[1, col_idx]
                    denom = ( xi + yi ) / 2
                    valid = denom != 0
                    x_sum = denom[valid]
                    y_norm_diff = (xi[valid] - yi[valid]) / x_sum / 2
                    if x_sum.size == 0:
                        continue

                    ax.scatter(x_sum, y_norm_diff, alpha=0.5, s=10)
                    if var_label == "Q_sum" and qsum_normdiff_shared_limits is not None:
                        x_left, x_right, y_bottom, y_top = qsum_normdiff_shared_limits
                        ax.set_xlim(x_left, x_right)
                        ax.set_ylim(y_bottom, y_top)
                    else:
                        ax.set_xlim(lim_left, lim_right)
                        ax.set_ylim(-1, 1)
                    ax.set_xlabel(f'{var_label}$_i$ + {var_label}$_j$ / 2')
                    ax.set_ylabel(f'({var_label}$_i$ - {var_label}$_j$) / ( 2 * sum )')
                    ax.set_title(f'{var_label}: Sum vs Norm. Diff')
                    ax.grid(True)

                fig.suptitle(f'Plane {i_plane}, Pattern {pattern}, Strips {i+1} & {j+1}', fontsize=16)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                if save_plots:
                    name_of_file = f'rpc_variables_2row_P{i_plane}_{pattern}_s{i+1}s{j+1}.png'
                    final_filename = f'{fig_idx}_{name_of_file}'
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, format='png')
                if show_plots:
                    plt.show()
                plt.close()

if task3_plot_enabled("tdiff_pattern_spatial_scatter"):

    patterns_of_interest = ['1100', '0110', '0011', '1001', '1010', '0101']
    fig, axs = plt.subplots(4, len(patterns_of_interest), figsize=(18, 12), sharex=True, sharey=False)

    for i_plane in range(1, 5):
        active_col = f'active_strips_P{i_plane}'
        T_dif_cols = [f'T{i_plane}_T_dif_{j+1}' for j in range(4)]

        for j_pattern, pattern in enumerate(patterns_of_interest):
            ax = axs[i_plane - 1, j_pattern]

            active_strips = [i for i, c in enumerate(pattern) if c == '1']
            if len(active_strips) != 2:
                ax.set_visible(False)
                continue

            i, j = active_strips
            mask = working_df[active_col] == pattern
            if mask.sum() == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            xi = working_df.loc[mask, T_dif_cols[i]].values
            yi = working_df.loc[mask, T_dif_cols[j]].values
            diff = ( yi - xi ) * tdiff_to_x
            semi_suma = ( yi + xi ) / 2 * tdiff_to_x

            # ax.hist(diff, bins=40, color='blue', alpha=0.7)
            ax.scatter(semi_suma, diff, color='blue', alpha=0.6, s = 1)
            # ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlim(-150, 150)
            ax.set_ylim(-2 * tdiff_to_x, 2 * tdiff_to_x)
            ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
            ax.set_xlabel(f'X mean along the strip (mm)')
            ax.set_ylabel(f'X difference (mm)')
            ax.grid(True)

    fig.suptitle("Histograms of T_diff Differences for Different Patterns", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plots:
        name_of_file = 'tdiff_differences_hist_4x3.png'
        final_filename = f'{fig_idx}_{name_of_file}'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

if task3_plot_enabled("tdiff_pattern_charge_scatter"):

    patterns_of_interest = ['1100', '0110', '0011', '1001', '1010', '0101']
    fig, axs = plt.subplots(4, len(patterns_of_interest), figsize=(18, 12), sharex=True, sharey=False)

    for i_plane in range(1, 5):
        active_col = f'active_strips_P{i_plane}'
        T_dif_cols = [f'T{i_plane}_T_dif_{j+1}' for j in range(4)]
        Q_sum_cols = [f'Q{i_plane}_Q_sum_{j+1}' for j in range(4)]

        for j_pattern, pattern in enumerate(patterns_of_interest):
            ax = axs[i_plane - 1, j_pattern]

            active_strips = [i for i, c in enumerate(pattern) if c == '1']
            if len(active_strips) != 2:
                ax.set_visible(False)
                continue

            i, j = active_strips
            mask = working_df[active_col] == pattern
            if mask.sum() == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            xi = working_df.loc[mask, T_dif_cols[i]].values
            yi = working_df.loc[mask, T_dif_cols[j]].values
            diff = ( yi - xi ) * tdiff_to_x
            semi_suma = ( yi + xi ) / 2 * tdiff_to_x
            qi = working_df.loc[mask, Q_sum_cols[i]].values
            qj = working_df.loc[mask, Q_sum_cols[j]].values
            q_semisum = ( qi + qj ) / 2
            q_semidiff = ( qi - qj ) / 2 / q_semisum

            # ax.hist(diff, bins=40, color='blue', alpha=0.7)
            ax.scatter(q_semisum, diff, color='blue', alpha=0.6, s = 1)
            # ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlim(0, 50)
            ax.set_ylim(-2 * tdiff_to_x, 2 * tdiff_to_x)
            ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
            ax.set_xlabel(f'Q mean (ns)')
            ax.set_ylabel(f'X difference (mm)')
            ax.grid(True)

    fig.suptitle("Histograms of T_diff Differences for Different Patterns", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plots:
        name_of_file = 'tdiff_differences_vs_q_sum_4x3.png'
        final_filename = f'{fig_idx}_{name_of_file}'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

if task3_plot_enabled("tdiff_pattern_charge_scan_scatter"):

    for charge_limit in np.linspace(5, 15, 3):

        patterns_of_interest = ['1100', '0110', '0011', '1001', '1010', '0101']
        fig, axs = plt.subplots(4, len(patterns_of_interest), figsize=(18, 12), sharex=True, sharey=False)

        for i_plane in range(1, 5):
            active_col = f'active_strips_P{i_plane}'
            T_dif_cols = [f'T{i_plane}_T_dif_{j+1}' for j in range(4)]
            Q_sum_cols = [f'Q{i_plane}_Q_sum_{j+1}' for j in range(4)]

            for j_pattern, pattern in enumerate(patterns_of_interest):
                ax = axs[i_plane - 1, j_pattern]

                active_strips = [i for i, c in enumerate(pattern) if c == '1']
                if len(active_strips) != 2:
                    ax.set_visible(False)
                    continue

                i, j = active_strips
                mask = working_df[active_col] == pattern
                if mask.sum() == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue

                xi = working_df.loc[mask, T_dif_cols[i]].values
                yi = working_df.loc[mask, T_dif_cols[j]].values
                diff = ( yi - xi ) * tdiff_to_x
                semi_suma = ( yi + xi ) / 2 * tdiff_to_x

                qi = working_df.loc[mask, Q_sum_cols[i]].values
                qj = working_df.loc[mask, Q_sum_cols[j]].values
                charge_condition = (qi > charge_limit) & (qj > charge_limit)

                diff = diff[charge_condition]
                semi_suma = semi_suma[charge_condition]

                # ax.hist(diff, bins=40, color='blue', alpha=0.7)
                ax.scatter(semi_suma, diff, color='blue', alpha=0.6, s = 1)
                # ax.axvline(0, color='black', linestyle='--', linewidth=1)
                ax.set_xlim(-150, 150)
                ax.set_ylim(-2 * tdiff_to_x, 2 * tdiff_to_x)
                ax.set_title(f'Plane {i_plane}, Pattern {pattern}, Charge > {charge_limit} ns')
                ax.set_xlabel(f'X mean along the strip (mm)')
                ax.set_ylabel(f'X difference (mm)')
                ax.grid(True)

        fig.suptitle("Histograms of T_diff Differences for Different Patterns", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_plots:
            name_of_file = f'tdiff_differences_hist_4x3_charge_{charge_limit}.png'
            final_filename = f'{fig_idx}_{name_of_file}'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()

if task3_plot_enabled("tdiff_pattern_histograms"):

    patterns_of_interest = ['1100', '0110', '0011', '1001', '1010', '0101']
    fig, axs = plt.subplots(4, len(patterns_of_interest), figsize=(18, 12), sharex=True, sharey=False)

    for i_plane in range(1, 5):
        active_col = f'active_strips_P{i_plane}'
        T_dif_cols = [f'T{i_plane}_T_dif_{j+1}' for j in range(4)]

        for j_pattern, pattern in enumerate(patterns_of_interest):
            ax = axs[i_plane - 1, j_pattern]

            active_strips = [i for i, c in enumerate(pattern) if c == '1']
            if len(active_strips) != 2:
                ax.set_visible(False)
                continue

            i, j = active_strips
            mask = working_df[active_col] == pattern
            if mask.sum() == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            xi = working_df.loc[mask, T_dif_cols[i]].values
            yi = working_df.loc[mask, T_dif_cols[j]].values
            
            cond = (xi != 0) & (yi != 0) & (abs(xi) < 1) & (abs(yi) < 1)
            xi = xi[cond]
            yi = yi[cond]
            diff = ( yi - xi ) * tdiff_to_x

            ax.hist(diff, bins=40, color='blue', alpha=0.6)
            # ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlim(-2 * tdiff_to_x, 2 * tdiff_to_x)
            ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
            ax.set_xlabel(f'X difference (mm)')
            ax.set_ylabel('Counts')
            ax.grid(True)

    fig.suptitle("Histograms of T_diff Differences for Different Patterns", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plots:
        name_of_file = 'tdiff_differences_hist_4x3_only_adj.png'
        final_filename = f'{fig_idx}_{name_of_file}'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()
    

    patterns_of_interest = ['1100', '0110', '0011']
    fig, axs = plt.subplots(4, len(patterns_of_interest), figsize=(24, 18), sharex=True, sharey=False)
    
    # Double Gaussian model
    def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
        g1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
        g2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
        return g1 + g2
    
    for i_plane in range(1, 5):
        active_col = f'active_strips_P{i_plane}'
        T_dif_cols = [f'T{i_plane}_T_dif_{j+1}' for j in range(4)]

        for j_pattern, pattern in enumerate(patterns_of_interest):
            ax = axs[i_plane - 1, j_pattern]

            active_strips = [i for i, c in enumerate(pattern) if c == '1']
            if len(active_strips) != 2:
                ax.set_visible(False)
                continue

            i, j = active_strips
            mask = working_df[active_col] == pattern
            if mask.sum() == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            xi = working_df.loc[mask, T_dif_cols[i]].values
            yi = working_df.loc[mask, T_dif_cols[j]].values
        
            cond = (xi != 0) & (yi != 0) & (abs(xi) < 1) & (abs(yi) < 1)
            xi = xi[cond]
            yi = yi[cond]
            diff = ( yi - xi ) * tdiff_to_x
            
            cond_new = abs(diff) < 150
            diff = diff[cond_new]
            
            adjacent_nbins = 100
            
            # Histogram
            counts, bin_edges = np.histogram(diff, bins=adjacent_nbins, range=(-150, 150))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Constraint bounds
            tolerance_in_pct = 100  # percent
            
            anc_std_in_mm = anc_std * tdiff_to_x
            
            sigma_small_left = anc_std_in_mm * (1 - tolerance_in_pct/100)
            sigma_small_right = anc_std_in_mm * (1 + tolerance_in_pct/100)
            
            print(f"Left and right limits in sigma: {sigma_small_left:.3f}, {sigma_small_right:.3f} mm")
            
            lower_bound = [0,     -100, sigma_small_left,  0,     -100, 0]
            upper_bound = [np.inf, 100, sigma_small_right, np.inf, 100, 1000]

            # Initial guesses
            p0 = [50, 0, anc_std_in_mm, 50, 0, 20]

            # Fit, if not fit, skip
            try:
                popt, _ = curve_fit(double_gaussian, bin_centers, counts, p0=p0, bounds=(lower_bound, upper_bound))
            except RuntimeError:
                print(f"Fit failed for Plane {i_plane}, Pattern {pattern}. Skipping.")
                ax.text(0.5, 0.5, 'Fit failed', ha='center', va='center', transform=ax.transAxes)
                continue

            # Extract fitted components
            A1, mu1, sigma1, A2, mu2, sigma2 = popt
            fit_x = np.linspace(-150, 150, 500)
            g1 = A1 * np.exp(-0.5 * ((fit_x - mu1) / sigma1)**2)
            g2 = A2 * np.exp(-0.5 * ((fit_x - mu2) / sigma2)**2)
            fit_total = g1 + g2
            
            ax.hist(diff, bins=adjacent_nbins, range=(-150, 150), color='blue', alpha=0.4, label='Data')
            ax.plot(fit_x, g1, '--', label=f'σ={sigma1:.1f}')
            ax.plot(fit_x, g2, '--', label=f'σ={sigma2:.1f}')

            ax.plot(fit_x, fit_total, '-', color='red', label='Total fit')
            
            ax.set_xlim(-150, 150)
            ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
            ax.set_xlabel(f'X difference (mm)')
            ax.set_ylabel('Counts')
            ax.grid(True)
            ax.legend()

    fig.suptitle("Fit to the Histograms of T_diff Differences for Different Patterns", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plots:
        name_of_file = 'tdiff_differences_hist_4x3_fit.png'
        final_filename = f'{fig_idx}_{name_of_file}'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

calculate_sigmas_adjacent = False

if calculate_sigmas_adjacent:

    # Sigmas per charge limit storage, Save in each line the charge limit, the pattern and the sigmas for each plane
    sigma_results = {}
    all_results = {}

    charge_limits_to_test = np.linspace(0, 50, 50)
    for charge_limit in charge_limits_to_test:

        patterns_of_interest = ['1100', '0110', '0011']
        
        # Double Gaussian model
        def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
            g1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
            g2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
            return g1 + g2
        
        for i_plane in range(1, 5):
            active_col = f'active_strips_P{i_plane}'
            T_dif_cols = [f'T{i_plane}_T_dif_{j+1}' for j in range(4)]
            Q_sum_cols = [f'Q{i_plane}_Q_sum_{j+1}' for j in range(4)]

            for j_pattern, pattern in enumerate(patterns_of_interest):

                active_strips = [i for i, c in enumerate(pattern) if c == '1']

                i, j = active_strips
                mask = working_df[active_col] == pattern

                xi = working_df.loc[mask, T_dif_cols[i]].values
                yi = working_df.loc[mask, T_dif_cols[j]].values
                qi = working_df.loc[mask, Q_sum_cols[i]].values
                qj = working_df.loc[mask, Q_sum_cols[j]].values
            
                cond = (xi != 0) & (yi != 0) & (abs(xi) < 1) & (abs(yi) < 1)
                xi = xi[cond]
                yi = yi[cond]
                qi = qi[cond]
                qj = qj[cond]
                diff = ( yi - xi ) * tdiff_to_x

                # charge_condition = (qi > charge_limit) & (qj > charge_limit)
                radius = 2
                charge_condition = ( (qi + qj) > charge_limit - radius) & ( (qi + qj) < charge_limit + radius) & (qi > 1) & (qj > 1)
                # charge_condition = ( (qi + qj) > charge_limit)
                diff = diff[charge_condition]
                
                cond_new = abs(diff) < 150
                diff = diff[cond_new]
                
                adjacent_nbins = 100
                
                # Histogram
                counts, bin_edges = np.histogram(diff, bins=adjacent_nbins, range=(-150, 150))
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                # Constraint bounds
                tolerance_in_pct = 100  # percent
                
                anc_std_in_mm = anc_std * tdiff_to_x
                
                sigma_small_left = anc_std_in_mm * (1 - tolerance_in_pct/100)
                sigma_small_right = anc_std_in_mm * (1 + tolerance_in_pct/100)
                
                # print(f"Left and right limits in sigma_small: {sigma_small_left:.3f}, {sigma_small_right:.3f} mm")
                
                lower_bound = [0,     -10, sigma_small_left,  0,     -10, 0]
                upper_bound = [np.inf, 10, sigma_small_right, np.inf, 10, 1000]

                # Initial guesses
                p0 = [50, 0, anc_std_in_mm, 50, 0, 20]

                # Fit
                try:
                    popt, _ = curve_fit(double_gaussian, bin_centers, counts, p0=p0, bounds=(lower_bound, upper_bound))
                except RuntimeError:
                    print(f"Fit failed for Plane {i_plane}, Pattern {pattern}, Charge limit {charge_limit:.1f}. Skipping.")
                    continue

                # Extract fitted components
                A1, mu1, sigma1, A2, mu2, sigma2 = popt
                fit_x = np.linspace(-150, 150, 500)
                g1 = A1 * np.exp(-0.5 * ((fit_x - mu1) / sigma1)**2)
                g2 = A2 * np.exp(-0.5 * ((fit_x - mu2) / sigma2)**2)
                fit_total = g1 + g2

                # Store sigma1 result
                sigma_results[(f'Charge_{charge_limit:.1f}', f'Plane_{i_plane}', pattern)] = sigma1, sigma2
                all_results[(f'Charge_{charge_limit:.1f}', f'Plane_{i_plane}', pattern)] = popt
    
    
    if task3_plot_enabled("tdiff_pattern_charge_slice_fits"):
        for charge_limit in charge_limits_to_test:

            patterns_of_interest = ['1100', '0110', '0011']
            fig, axs = plt.subplots(4, len(patterns_of_interest), figsize=(24, 18), sharex=True, sharey=True)
            
            for i_plane in range(1, 5):
                active_col = f'active_strips_P{i_plane}'
                T_dif_cols = [f'T{i_plane}_T_dif_{j+1}' for j in range(4)]
                Q_sum_cols = [f'Q{i_plane}_Q_sum_{j+1}' for j in range(4)]

                for j_pattern, pattern in enumerate(patterns_of_interest):
                    ax = axs[i_plane - 1, j_pattern]

                    active_strips = [i for i, c in enumerate(pattern) if c == '1']
                    if len(active_strips) != 2:
                        ax.set_visible(False)
                        continue

                    i, j = active_strips
                    mask = working_df[active_col] == pattern
                    if mask.sum() == 0:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                        continue

                    xi = working_df.loc[mask, T_dif_cols[i]].values
                    yi = working_df.loc[mask, T_dif_cols[j]].values
                    qi = working_df.loc[mask, Q_sum_cols[i]].values
                    qj = working_df.loc[mask, Q_sum_cols[j]].values
                
                    cond = (xi != 0) & (yi != 0) & (abs(xi) < 1) & (abs(yi) < 1)
                    xi = xi[cond]
                    yi = yi[cond]
                    qi = qi[cond]
                    qj = qj[cond]
                    diff = ( yi - xi ) * tdiff_to_x

                    # charge_condition = (qi > charge_limit) & (qj > charge_limit)
                    radius = 5
                    charge_condition = ( (qi + qj) > charge_limit - radius) & ( (qi + qj) < charge_limit + radius) & (qi > 1) & (qj > 1)
                    # charge_condition = (qi + qj > charge_limit)
                    diff = diff[charge_condition]
                    
                    cond_new = abs(diff) < 150
                    diff = diff[cond_new]
                    
                    adjacent_nbins = 100
                    
                    # Histogram
                    counts, bin_edges = np.histogram(diff, bins=adjacent_nbins, range=(-150, 150))
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                    # Extract fitted components
                    popt = all_results.get((f'Charge_{charge_limit:.1f}', f'Plane_{i_plane}', pattern), None)
                    
                    if popt is None:
                        print(f"No fit results for Plane {i_plane}, Pattern {pattern}, Charge limit {charge_limit:.1f}. Skipping.")
                        ax.text(0.5, 0.5, 'No fit results', ha='center', va='center', transform=ax.transAxes)
                        continue

                    A1, mu1, sigma1, A2, mu2, sigma2 = popt
                    fit_x = np.linspace(-150, 150, 500)
                    g1 = A1 * np.exp(-0.5 * ((fit_x - mu1) / sigma1)**2)
                    g2 = A2 * np.exp(-0.5 * ((fit_x - mu2) / sigma2)**2)
                    fit_total = g1 + g2

                    ax.hist(diff, bins=adjacent_nbins, range=(-150, 150), color='blue', alpha=0.4, label='Data')
                    ax.plot(fit_x, g1, '--', label=f'σ={sigma1:.1f}')
                    ax.plot(fit_x, g2, '--', label=f'σ={sigma2:.1f}')

                    ax.plot(fit_x, fit_total, '-', color='red', label='Total fit')
                    
                    ax.set_xlim(-150, 150)
                    ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
                    ax.set_xlabel(f'X difference (mm)')
                    ax.set_ylabel('Counts')
                    ax.grid(True)
                    ax.legend()

            fig.suptitle(f"Fit to the Histograms of T_diff Differences for Different Patterns, Charge limit: {charge_limit:.1f}", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_plots:
                name_of_file = f'tdiff_differences_hist_4x3_fit_{charge_limit:.1f}.png'
                final_filename = f'{fig_idx}_{name_of_file}'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()

    

    if task3_plot_enabled("tdiff_pattern_sigma_vs_charge"):
        # Plot especifically the sigmas vs charge limit for each plane and pattern, one row per plane, one column per pattern
        fig, axs = plt.subplots(4, len(patterns_of_interest), figsize=(18, 24), sharex=True, sharey=True)
        for i_plane in range(1, 5):
            for j_pattern, pattern in enumerate(patterns_of_interest):
                ax = axs[i_plane - 1, j_pattern]

                charge_limits = []
                mu1_values = []
                mu2_values = []
                sigma1_values = []
                sigma2_values = []

                for charge_limit in charge_limits_to_test:
                    key = (f'Charge_{charge_limit:.1f}', f'Plane_{i_plane}', pattern)
                    
                    if key in all_results:
                        A1, mu1, sigma1, A2, mu2, sigma2 = all_results[key]
                        charge_limits.append(charge_limit)
                        mu1_values.append(mu1)
                        sigma1_values.append(sigma1)
                        mu2_values.append(mu2)
                        sigma2_values.append(sigma2)

                ax.plot(charge_limits, mu1_values, marker='o', label='Mu 1')
                ax.fill_between(charge_limits, mu1_values - np.array(sigma1_values), mu1_values + np.array(sigma1_values), alpha=0.2)
                ax.plot(charge_limits, mu2_values, marker='s', label='Mu 2')
                ax.fill_between(charge_limits, mu2_values - np.array(sigma2_values), mu2_values + np.array(sigma2_values), alpha=0.2)

                ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
                ax.set_xlabel('Charge Limit')
                ax.set_ylabel('Sigma (mm)')
                ax.grid(True)
                ax.legend()
        fig.suptitle("Fitted Sigmas vs Charge Limit for Different Patterns", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_plots:
            name_of_file = f'tdiff_fitted_sigmas_vs_charge_limit.png'
            final_filename = f'{fig_idx}_{name_of_file}'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()
    

    loop_adjacent_strip_fit = False

    if loop_adjacent_strip_fit and task3_any_plot_enabled(
        "tdiff_pattern_sigma1_charge_surface",
        "tdiff_pattern_sigma2_charge_surface",
    ):
        # Sigmas per charge limit storage, Save in each line the charge limit, the pattern and the sigmas for each plane
        all_results_loop = {}

        charge_limits_to_loop = np.linspace(0, 10, 10)
        for charge_limit_1 in charge_limits_to_loop:
            for charge_limit_2 in charge_limits_to_loop:

                patterns_of_interest = ['1100', '0110', '0011']
                
                # Double Gaussian model
                def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
                    g1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
                    g2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
                    return g1 + g2
                
                for i_plane in range(1, 5):
                    active_col = f'active_strips_P{i_plane}'
                    T_dif_cols = [f'T{i_plane}_T_dif_{j+1}' for j in range(4)]
                    Q_sum_cols = [f'Q{i_plane}_Q_sum_{j+1}' for j in range(4)]

                    for j_pattern, pattern in enumerate(patterns_of_interest):

                        active_strips = [i for i, c in enumerate(pattern) if c == '1']

                        i, j = active_strips
                        mask = working_df[active_col] == pattern

                        xi = working_df.loc[mask, T_dif_cols[i]].values
                        yi = working_df.loc[mask, T_dif_cols[j]].values
                        qi = working_df.loc[mask, Q_sum_cols[i]].values
                        qj = working_df.loc[mask, Q_sum_cols[j]].values
                    
                        cond = (xi != 0) & (yi != 0) & (abs(xi) < 1) & (abs(yi) < 1)
                        xi = xi[cond]
                        yi = yi[cond]
                        qi = qi[cond]
                        qj = qj[cond]
                        diff = ( yi - xi ) * tdiff_to_x

                        charge_condition = (qi > charge_limit_1) & (qj > charge_limit_2)
                        diff = diff[charge_condition]
                        
                        cond_new = abs(diff) < 150
                        diff = diff[cond_new]
                        
                        adjacent_nbins = 100
                        
                        # Histogram
                        counts, bin_edges = np.histogram(diff, bins=adjacent_nbins, range=(-150, 150))
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                        # Constraint bounds
                        tolerance_in_pct = 100  # percent
                        
                        anc_std_in_mm = anc_std * tdiff_to_x
                        
                        sigma_small_left = anc_std_in_mm * (1 - tolerance_in_pct/100)
                        sigma_small_right = anc_std_in_mm * (1 + tolerance_in_pct/100)
                        
                        # print(f"Left and right limits in sigma: {sigma_small_left:.3f}, {sigma_small_right:.3f} mm")
                        
                        lower_bound = [0,     -10, sigma_small_left,  0,     -10, 0]
                        upper_bound = [np.inf, 10, sigma_small_right, np.inf, 10, 1000]

                        # Initial guesses
                        p0 = [50, 0, anc_std_in_mm, 50, 0, 20]

                        # Fit
                        try:
                            popt, _ = curve_fit(double_gaussian, bin_centers, counts, p0=p0, bounds=(lower_bound, upper_bound))
                        except RuntimeError:
                            print(f"Fit failed for Plane {i_plane}, Pattern {pattern}, Charge limits {charge_limit_1:.1f}, {charge_limit_2:.1f}. Skipping.")
                            continue

                        # Extract fitted components
                        A1, mu1, sigma1, A2, mu2, sigma2 = popt
                        fit_x = np.linspace(-150, 150, 500)
                        g1 = A1 * np.exp(-0.5 * ((fit_x - mu1) / sigma1)**2)
                        g2 = A2 * np.exp(-0.5 * ((fit_x - mu2) / sigma2)**2)
                        fit_total = g1 + g2

                        # Store sigma1 result
                        all_results_loop[(f'Charge_1_{charge_limit_1:.1f}', f'Charge_2_{charge_limit_2:.1f}', f'Plane_{i_plane}', pattern)] = popt

        # Plot especifically the sigmas vs charge limit for each plane and pattern, one row per plane, one column per pattern

        if task3_plot_enabled("tdiff_pattern_sigma1_charge_surface"):
            fig, axs = plt.subplots(4, len(patterns_of_interest), figsize=(18, 24), sharex=True, sharey=True)
            for i_plane in range(1, 5):
                for j_pattern, pattern in enumerate(patterns_of_interest):
                    ax = axs[i_plane - 1, j_pattern]

                    charge_limits_1 = []
                    charge_limits_2 = []
                    mu1_values = []
                    mu2_values = []
                    sigma1_values = []
                    sigma2_values = []

                    for charge_limit_1 in charge_limits_to_loop:
                        for charge_limit_2 in charge_limits_to_loop:
                            key = (f'Charge_1_{charge_limit_1:.1f}', f'Charge_2_{charge_limit_2:.1f}', f'Plane_{i_plane}', pattern)

                            if key in all_results_loop:
                                A1, mu1, sigma1, A2, mu2, sigma2 = all_results_loop[key]
                                charge_limits_1.append(charge_limit_1)
                                charge_limits_2.append(charge_limit_2)
                                mu1_values.append(mu1)
                                sigma1_values.append(sigma1)
                                mu2_values.append(mu2)
                                sigma2_values.append(sigma2)

                    print(sigma1_values)

                    charge1_values = charge_limits_1
                    charge2_values = charge_limits_2

                    unique_c1 = np.array(sorted(set(charge1_values)))
                    unique_c2 = np.array(sorted(set(charge2_values)))
                    C1, C2 = np.meshgrid(unique_c1, unique_c2, indexing='xy')

                    sigma1_grid = np.full(C1.shape, np.nan, dtype=float)
                    sigma2_grid = np.full(C2.shape, np.nan, dtype=float)
                    idx_c1 = {c: j for j, c in enumerate(unique_c1)}
                    idx_c2 = {c: i for i, c in enumerate(unique_c2)}

                    for c1, c2, s1, s2 in zip(charge1_values, charge2_values, sigma1_values, sigma2_values):
                        i = idx_c2[c2]
                        j = idx_c1[c1]
                        sigma1_grid[i, j] = s1
                        sigma2_grid[i, j] = s2

                    Z = sigma1_grid
                    if np.all(np.isnan(Z)):
                        ax.set_title(f'Plane {i_plane}, Pattern {pattern}\n(no valid fits)')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    cs = ax.contourf(C1, C2, Z, levels=20)
                    fig.colorbar(cs, ax=ax)
                    ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
                    ax.set_xlabel('Charge 1')
                    ax.set_ylabel('Charge 2')
                    ax.grid(True)

            fig.suptitle("Fitted Sigma 1 vs Charge Limit for Different Patterns", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_plots:
                name_of_file = 'tdiff_fitted_sigma_1_vs_charge_limit.png'
                final_filename = f'{fig_idx}_{name_of_file}'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()

        if task3_plot_enabled("tdiff_pattern_sigma2_charge_surface"):
            fig, axs = plt.subplots(4, len(patterns_of_interest), figsize=(18, 24), sharex=True, sharey=True)
            for i_plane in range(1, 5):
                for j_pattern, pattern in enumerate(patterns_of_interest):
                    ax = axs[i_plane - 1, j_pattern]

                    charge_limits_1 = []
                    charge_limits_2 = []
                    mu1_values = []
                    mu2_values = []
                    sigma1_values = []
                    sigma2_values = []

                    for charge_limit_1 in charge_limits_to_loop:
                        for charge_limit_2 in charge_limits_to_loop:
                            key = (f'Charge_1_{charge_limit_1:.1f}', f'Charge_2_{charge_limit_2:.1f}', f'Plane_{i_plane}', pattern)

                            if key in all_results_loop:
                                A1, mu1, sigma1, A2, mu2, sigma2 = all_results_loop[key]
                                charge_limits_1.append(charge_limit_1)
                                charge_limits_2.append(charge_limit_2)
                                mu1_values.append(mu1)
                                sigma1_values.append(sigma1)
                                mu2_values.append(mu2)
                                sigma2_values.append(sigma2)

                    print(sigma1_values)

                    charge1_values = charge_limits_1
                    charge2_values = charge_limits_2

                    unique_c1 = np.array(sorted(set(charge1_values)))
                    unique_c2 = np.array(sorted(set(charge2_values)))
                    C1, C2 = np.meshgrid(unique_c1, unique_c2, indexing='xy')

                    sigma1_grid = np.full(C1.shape, np.nan, dtype=float)
                    sigma2_grid = np.full(C2.shape, np.nan, dtype=float)
                    idx_c1 = {c: j for j, c in enumerate(unique_c1)}
                    idx_c2 = {c: i for i, c in enumerate(unique_c2)}

                    for c1, c2, s1, s2 in zip(charge1_values, charge2_values, sigma1_values, sigma2_values):
                        i = idx_c2[c2]
                        j = idx_c1[c1]
                        sigma1_grid[i, j] = s1
                        sigma2_grid[i, j] = s2

                    Z = sigma2_grid
                    if np.all(np.isnan(Z)):
                        ax.set_title(f'Plane {i_plane}, Pattern {pattern}\n(no valid fits)')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    cs = ax.contourf(C1, C2, Z, levels=20)
                    fig.colorbar(cs, ax=ax)
                    ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
                    ax.set_xlabel('Charge 1')
                    ax.set_ylabel('Charge 2')
                    ax.grid(True)

            fig.suptitle("Fitted Sigma 2 vs Charge Limit for Different Patterns", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_plots:
                name_of_file = 'tdiff_fitted_sigma_2_vs_charge_limit.png'
                final_filename = f'{fig_idx}_{name_of_file}'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()

_prof["s_multi_strip_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("----------------------- Y position calculation -----------------------")
print("----------------------------------------------------------------------")

# Y ---------------------------------------------------------------------------
y_widths = [np.array([narrow_strip, narrow_strip, narrow_strip, wide_strip]), 
            np.array([wide_strip, narrow_strip, narrow_strip, narrow_strip])]

y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_P1_and_P3 = y_widths[0]
y_width_P2_and_P4 = y_widths[1]
y_pos_P1_and_P3 = y_pos(y_width_P1_and_P3)
y_pos_P2_and_P4 = y_pos(y_width_P2_and_P4)
total_width = np.sum(y_width_P1_and_P3)

print("Total width:", total_width)

strip_boundaries_1_3 = np.cumsum(np.insert(y_widths[0], 0, 0)) - total_width / 2
strip_boundaries_2_4 = np.cumsum(np.insert(y_widths[1], 0, 0)) - total_width / 2

if y_new_method:
    y_columns = {}

    for plane_id in range(1, 5):
        # Decode binary strip activity per plane into shape (N_events, 4)
        topo_binary = np.array([
            list(map(int, s)) for s in working_df[f'active_strips_P{plane_id}']
        ])
        
        q_plane = working_df[[f'Q{plane_id}_Q_sum_{i}' for i in range(1, 5)]].values  # shape (N, 4)
        
        # Take only active strips' charges
        q_active = topo_binary * q_plane
        
        # y-position vector by plane ID
        y_vec = y_pos_P1_and_P3 if plane_id in [1, 3] else y_pos_P2_and_P4

        # Initial weighted y estimate (default for multi-strip)
        weighted_y = q_active * y_vec
        active_counts = topo_binary.sum(axis=1)
        total_charge = q_active.sum(axis=1)
        total_charge_safe = np.where(total_charge == 0, 1, total_charge)

        y_position = weighted_y.sum(axis=1) / total_charge_safe
        y_position[active_counts == 0] = 0  # zero when no strips active

        # Apply uniform blur only to single-strip cases (vectorized)
        one_strip_mask = active_counts == 1

        if np.any(one_strip_mask):
            # Which row indices are single-strip?
            rows = np.where(one_strip_mask)[0]
            # For those rows, which strip is active? (use topo_binary or q_active)
            cols = topo_binary[one_strip_mask].argmax(axis=1)  # shape (n_single,)

            # Centers and widths for the selected strips
            y_vec = y_pos_P1_and_P3 if plane_id in [1, 3] else y_pos_P2_and_P4
            widths_vec = y_width_P1_and_P3 if plane_id in [1, 3] else y_width_P2_and_P4

            centers = y_vec[cols]
            widths  = widths_vec[cols]

            # Random uniform within the active strip
            y_position[rows] = np.random.uniform(centers - widths/2, centers + widths/2)

        # Store result
        y_columns[f'P{plane_id}_Y_final'] = y_position

    # Insert all new Y_ columns at once
    working_df = pd.concat([working_df, pd.DataFrame(y_columns, index=working_df.index)], axis=1)

if task3_plot_enabled("y_position_by_cal_tt"):

    for cal_tt in [ 12, 23, 34, 1234, 123, 234, 124, 13, 14, 24, 134]:
        mask = working_df['cal_tt'] == cal_tt
        filtered_df = working_df[mask].copy()  # Work on a copy for fitting
    
        plt.figure(figsize=(12, 8))
        for i, plane_id in enumerate(range(1, 5), 1):
            plt.subplot(2, 2, i)
            column_name = f'P{plane_id}_Y_final'
            data = filtered_df[column_name]
            
            plt.hist(data[data != 0], bins=100, histtype='stepfilled', alpha=0.6)
            
            # Plot the strip boundaries
            boundaries = strip_boundaries_1_3 if plane_id in [1, 3] else strip_boundaries_2_4
            for boundary in boundaries:
                plt.axvline(boundary, color='red', linestyle='--', linewidth=1)
            
            # Plot the strip centers
            centers = y_pos_P1_and_P3 if plane_id in [1, 3] else y_pos_P2_and_P4
            for center in centers:
                plt.axvline(center, color='green', linestyle=':', linewidth=0.5)
            
            plt.title(f'Y Position Distribution - Plane {plane_id}')
            plt.xlabel('Y Position (a.u.)')
            plt.ylabel('Counts')
            plt.grid(True)
        
        plt.suptitle(f'Y Position Distribution for cal_tt = {cal_tt}', fontsize=16)
        plt.tight_layout()
        if save_plots:
            name_of_file = f'Y_{cal_tt}'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()

print("Y position calculated.")

_prof["s_y_pos_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("------------ Last comprobation to the per-strip variables ------------")
print("----------------------------------------------------------------------")

if task3_plot_enabled("strip_variable_pairgrid"):

    for i_plane in range(1, 5):
        
        fig, axes = plt.subplots(4, 6, figsize=(30, 20))
        axes = axes.flatten()
        
        for strip in range(1, 5):
            # Column names
            t_sum_col = f'T{i_plane}_T_sum_{strip}'
            t_dif_col = f'T{i_plane}_T_dif_{strip}'
            q_sum_col = f'Q{i_plane}_Q_sum_{strip}'
            q_dif_col = f'Q{i_plane}_Q_dif_{strip}'

            # Filter valid rows (non-zero)
            valid_rows = working_df[[t_sum_col, t_dif_col, q_sum_col, q_dif_col]].replace(0, np.nan).dropna()
            
            # Extract variables and filter low charge
            cond = valid_rows[q_sum_col] < charge_per_strip_plot_threshold
            t_sum  = valid_rows.loc[cond, t_sum_col]
            t_diff = valid_rows.loc[cond, t_dif_col]
            q_sum  = valid_rows.loc[cond, q_sum_col]
            q_diff = valid_rows.loc[cond, q_dif_col]

            base_idx = (strip - 1) * 6

            plot_pairs = [
                (t_sum,  t_diff, f'{t_sum_col} vs {t_dif_col}'),
                (t_sum,  q_sum,  f'{t_sum_col} vs {q_sum_col}'),
                (t_diff, q_sum,  f'{t_dif_col} vs {q_sum_col}'),
                (t_sum,  q_diff, f'{t_sum_col} vs {q_dif_col}'),
                (t_diff, q_diff, f'{t_dif_col} vs {q_dif_col}'),
                (q_sum,  q_diff, f'{q_sum_col} vs {q_dif_col}')
            ]

            for offset, (x, yv, title) in enumerate(plot_pairs):
                ax = axes[base_idx + offset]
                ax.hexbin(x, yv, gridsize=50, cmap='turbo')
                # ax.scatter(x, yv)
                ax.set_title(title)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.suptitle(f'Hexbin Plots for All Variable Combinations by strip for plane {i_plane}', fontsize=18)

        if save_plots:
            name_of_file = f'strip_check_hexbin_combinations_filtered_{i_plane}'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')

        if show_plots: plt.show()
        plt.close()

if self_trigger:
    if task3_plot_enabled("self_trigger_strip_variable_pairgrid"):
        
        for i_plane in range(1, 5):
            
            fig, axes = plt.subplots(4, 6, figsize=(30, 20))
            axes = axes.flatten()
            
            for strip in range(1, 5):
                # Column names
                t_sum_col = f'T{i_plane}_T_sum_{strip}'
                t_dif_col = f'T{i_plane}_T_dif_{strip}'
                q_sum_col = f'Q{i_plane}_Q_sum_{strip}'
                q_dif_col = f'Q{i_plane}_Q_dif_{strip}'

                # Filter valid rows (non-zero)
                valid_rows = working_st_df[[t_sum_col, t_dif_col, q_sum_col, q_dif_col]].replace(0, np.nan).dropna()
                
                # Extract variables and filter low charge
                cond = valid_rows[q_sum_col] < 40
                t_sum  = valid_rows.loc[cond, t_sum_col]
                t_diff = valid_rows.loc[cond, t_dif_col]
                q_sum  = valid_rows.loc[cond, q_sum_col]
                q_diff = valid_rows.loc[cond, q_dif_col]

                base_idx = (strip - 1) * 6

                plot_pairs = [
                    (t_sum,  t_diff, f'{t_sum_col} vs {t_dif_col}'),
                    (t_sum,  q_sum,  f'{t_sum_col} vs {q_sum_col}'),
                    (t_diff, q_sum,  f'{t_dif_col} vs {q_sum_col}'),
                    (t_sum,  q_diff, f'{t_sum_col} vs {q_dif_col}'),
                    (t_diff, q_diff, f'{t_dif_col} vs {q_dif_col}'),
                    (q_sum,  q_diff, f'{q_sum_col} vs {q_dif_col}')
                ]

                for offset, (x, yv, title) in enumerate(plot_pairs):
                    ax = axes[base_idx + offset]
                    ax.hexbin(x, yv, gridsize=50, cmap='turbo')
                    # ax.scatter(x, yv)
                    ax.set_title(title)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.suptitle(f'SELF TRIGGER Hexbin Plots for All Variable Combinations by strip for plane {i_plane}', fontsize=18)

            if save_plots:
                name_of_file = f'strip_check_hexbin_combinations_filtered_{i_plane}_ST'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, format='png')

            if show_plots: plt.show()
            plt.close()

_prof["s_last_comprobation_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("----------------- Setting the variables of each RPC ------------------")
print("----------------------------------------------------------------------")

# Build the raw per-plane observables first. The filtering/zeroing path is
# applied immediately afterwards, one variable family at a time.
plane_raw_values: dict[int, dict[str, np.ndarray]] = {}
final_columns: dict[str, np.ndarray] = {}

for i_plane in range(1, 5):
    t_sum_cols = [f'T{i_plane}_T_sum_{i+1}' for i in range(4)]
    t_dif_cols = [f'T{i_plane}_T_dif_{i+1}' for i in range(4)]
    q_sum_cols = [f'Q{i_plane}_Q_sum_{i+1}' for i in range(4)]
    q_dif_cols = [f'Q{i_plane}_Q_dif_{i+1}' for i in range(4)]

    t_sums = working_df[t_sum_cols].astype(float).fillna(0).to_numpy(copy=False)
    t_difs = working_df[t_dif_cols].astype(float).fillna(0).to_numpy(copy=False)
    q_sums = working_df[q_sum_cols].astype(float).fillna(0).to_numpy(copy=False)
    q_difs = working_df[q_dif_cols].astype(float).fillna(0).to_numpy(copy=False)

    active_mask = np.array(
        [list(map(int, s)) for s in working_df[f'active_strips_P{i_plane}']],
        dtype=float,
    )
    n_active = active_mask.sum(axis=1)
    n_active_safe = np.where(n_active == 0, 1, n_active)

    t_sum_final = (t_sums * active_mask).sum(axis=1) / n_active_safe
    t_dif_final = (t_difs * active_mask).sum(axis=1) / n_active_safe
    q_sum_final = (q_sums * active_mask).sum(axis=1)
    q_dif_final = (q_difs * active_mask).sum(axis=1)

    t_sum_final[n_active == 0] = 0
    t_dif_final[n_active == 0] = 0
    q_sum_final[n_active == 0] = 0
    q_dif_final[n_active == 0] = 0

    plane_raw_values[i_plane] = {
        "T_sum": t_sum_final.copy(),
        "T_dif": t_dif_final.copy(),
        "Q_sum": q_sum_final.copy(),
        "Q_dif": q_dif_final.copy(),
    }
    final_columns[f'P{i_plane}_T_sum_final'] = t_sum_final
    final_columns[f'P{i_plane}_T_dif_final'] = t_dif_final
    final_columns[f'P{i_plane}_Q_sum_final'] = q_sum_final
    final_columns[f'P{i_plane}_Q_dif_final'] = q_dif_final

working_df = pd.concat([working_df, pd.DataFrame(final_columns, index=working_df.index)], axis=1)

if task3_plot_enabled("rpc_variables_hexbin"):
    fig, axes = plt.subplots(4, 10, figsize=(40, 20))  # 10 combinations per plane
    axes = axes.flatten()

    for i_plane in range(1, 5):
        # Column names
        t_sum_col = f'P{i_plane}_T_sum_final'
        t_dif_col = f'P{i_plane}_T_dif_final'
        q_sum_col = f'P{i_plane}_Q_sum_final'
        q_dif_col = f'P{i_plane}_Q_dif_final'
        y_col = f'P{i_plane}_Y_final'

        # Filter valid rows (non-zero)
        valid_rows = working_df[[t_sum_col, t_dif_col, q_sum_col, q_dif_col, y_col]].replace(0, np.nan).dropna()
        
        # Extract variables and filter low charge
        cond = valid_rows[q_sum_col] < charge_per_plane_plot_threshold
        t_sum  = valid_rows.loc[cond, t_sum_col]
        t_diff = valid_rows.loc[cond, t_dif_col]
        q_sum  = valid_rows.loc[cond, q_sum_col]
        q_diff = valid_rows.loc[cond, q_dif_col]
        y      = valid_rows.loc[cond, y_col]

        base_idx = (i_plane - 1) * 10  # 10 plots per plane

        plot_pairs = [
            (t_sum,  t_diff, f'{t_sum_col} vs {t_dif_col}'),
            (t_sum,  q_sum,  f'{t_sum_col} vs {q_sum_col}'),
            (t_sum,  y,      f'{t_sum_col} vs {y_col}'),
            (t_diff, q_sum,  f'{t_dif_col} vs {q_sum_col}'),
            (t_diff, y,      f'{t_dif_col} vs {y_col}'),
            (q_sum,  y,      f'{q_sum_col} vs {y_col}'),
            (t_sum,  q_diff, f'{t_sum_col} vs {q_dif_col}'),
            (t_diff, q_diff, f'{t_dif_col} vs {q_dif_col}'),
            (q_diff, y,      f'{q_dif_col} vs {y_col}'),
            (q_sum,  q_diff, f'{q_sum_col} vs {q_dif_col}')
        ]

        for offset, (x, yv, title) in enumerate(plot_pairs):
            ax = axes[base_idx + offset]
            _task3_plot_quantile_hexbin(ax, x, yv, title, gridsize=50, cmap="turbo")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Hexbin Plots for All Variable Combinations by Plane', fontsize=18)

    if save_plots:
        name_of_file = 'rpc_variables_hexbin_combinations'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close()

if task3_plot_enabled("rpc_variables_hexbin_low_charge"):
    fig, axes = plt.subplots(4, 10, figsize=(40, 20))  # 10 combinations per plane
    axes = axes.flatten()

    for i_plane in range(1, 5):
        # Column names
        t_sum_col = f'P{i_plane}_T_sum_final'
        t_dif_col = f'P{i_plane}_T_dif_final'
        q_sum_col = f'P{i_plane}_Q_sum_final'
        q_dif_col = f'P{i_plane}_Q_dif_final'
        y_col = f'P{i_plane}_Y_final'

        # Filter valid rows (non-zero)
        valid_rows = working_df[[t_sum_col, t_dif_col, q_sum_col, q_dif_col, y_col]].replace(0, np.nan).dropna()
        
        # Extract variables and filter low charge
        cond = valid_rows[q_sum_col] < charge_per_plane_plot_threshold / 4
        t_sum  = valid_rows.loc[cond, t_sum_col]
        t_diff = valid_rows.loc[cond, t_dif_col]
        q_sum  = valid_rows.loc[cond, q_sum_col]
        q_diff = valid_rows.loc[cond, q_dif_col]
        y      = valid_rows.loc[cond, y_col]

        base_idx = (i_plane - 1) * 10  # 10 plots per plane

        plot_pairs = [
            (t_sum,  t_diff, f'{t_sum_col} vs {t_dif_col}'),
            (t_sum,  q_sum,  f'{t_sum_col} vs {q_sum_col}'),
            (t_sum,  y,      f'{t_sum_col} vs {y_col}'),
            (t_diff, q_sum,  f'{t_dif_col} vs {q_sum_col}'),
            (t_diff, y,      f'{t_dif_col} vs {y_col}'),
            (q_sum,  y,      f'{q_sum_col} vs {y_col}'),
            (t_sum,  q_diff, f'{t_sum_col} vs {q_dif_col}'),
            (t_diff, q_diff, f'{t_dif_col} vs {q_dif_col}'),
            (q_diff, y,      f'{q_dif_col} vs {y_col}'),
            (q_sum,  q_diff, f'{q_sum_col} vs {q_dif_col}')
        ]

        for offset, (x, yv, title) in enumerate(plot_pairs):
            ax = axes[base_idx + offset]
            _task3_plot_quantile_hexbin(ax, x, yv, title, gridsize=50, cmap="turbo")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Hexbin Plots for All Variable Combinations by Plane', fontsize=18)

    if save_plots:
        name_of_file = 'rpc_variables_hexbin_combinations'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close()

_prof["s_rpc_vars_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("------ Put Tsum in reference to the first strip that is not zero -----")
print("----------------------------------------------------------------------")

cols = ["P1_T_sum_final", "P2_T_sum_final", "P3_T_sum_final", "P4_T_sum_final"]
vals = working_df[cols].to_numpy()
nonzero_mask = vals != 0
has_signal = nonzero_mask.any(axis=1)
first_nonzero_idx = np.where(has_signal, nonzero_mask.argmax(axis=1), 0)
row_indices = np.arange(len(working_df))
baseline_vals = np.zeros(len(working_df))
baseline_vals[has_signal] = vals[row_indices[has_signal], first_nonzero_idx[has_signal]]

# Normalize only events with signal; keep missing planes and empty events at 0.
vals_normalized = vals.copy()
vals_normalized[has_signal] = vals_normalized[has_signal] - baseline_vals[has_signal, np.newaxis] + 1
vals_normalized[~nonzero_mask] = 0
vals_normalized[~has_signal] = 0
working_df[cols] = vals_normalized
for i_plane in range(1, 5):
    plane_raw_values[i_plane]["T_sum"] = vals_normalized[:, i_plane - 1].copy()

# Legacy metadata key retained for compatibility: it now tracks the new
# sequential per-plane filtering cascade instead of the old all-at-once Filter 6.
def record_filter6_counts(df: pd.DataFrame, tag: str) -> None:
    for i_plane in range(1, 5):
        columns = {
            "T_sum": f"P{i_plane}_T_sum_final",
            "T_diff": f"P{i_plane}_T_dif_final",
            "Q_sum": f"P{i_plane}_Q_sum_final",
            "Q_diff": f"P{i_plane}_Q_dif_final",
            "Y": f"P{i_plane}_Y_final",
        }
        for label, col in columns.items():
            if col in df:
                count = int((df[col] != 0).sum())
                global_variables[f"P{i_plane}_{label}_nonzero_{tag}"] = count

filter6_cols: list[str] = []
for i_plane in range(1, 5):
    filter6_cols.extend([
        f"P{i_plane}_Y_final",
        f"P{i_plane}_T_sum_final",
        f"P{i_plane}_T_dif_final",
        f"P{i_plane}_Q_sum_final",
        f"P{i_plane}_Q_dif_final",
    ])
filter6_cols = [col for col in filter6_cols if col in working_df.columns]
filter6_before_zero_mask = None
if filter6_cols:
    filter6_before_zero_mask = (working_df[filter6_cols] == 0).any(axis=1)

if filter6_cols and task3_any_plot_enabled(
    "filter6_tsum_debug",
    "filter6_tdif_debug",
    "filter6_qsum_debug",
    "filter6_qdif_debug",
    "filter6_y_debug",
):
    t_sum_cols = [col for col in filter6_cols if "T_sum_final" in col]
    if t_sum_cols and task3_plot_enabled("filter6_tsum_debug"):
        debug_thresholds = {col: [T_sum_RPC_left, T_sum_RPC_right] for col in t_sum_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            t_sum_cols,
            debug_thresholds,
            title=(
                f"Task 3 pre-plane T_sum filter: T_sum_RPC_left/right "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

    t_dif_cols = [col for col in filter6_cols if "T_dif_final" in col]
    if t_dif_cols and task3_plot_enabled("filter6_tdif_debug"):
        debug_thresholds = {col: [T_dif_RPC_left, T_dif_RPC_right] for col in t_dif_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            t_dif_cols,
            debug_thresholds,
            title=(
                f"Task 3 pre-plane T_dif filter: T_dif_RPC_left/right "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

    q_sum_cols = [col for col in filter6_cols if "Q_sum_final" in col]
    if q_sum_cols and task3_plot_enabled("filter6_qsum_debug"):
        debug_thresholds = {col: [Q_RPC_left, Q_RPC_right] for col in q_sum_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            q_sum_cols,
            debug_thresholds,
            title=(
                f"Task 3 pre-plane Q_sum filter: Q_RPC_left/right "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

    q_dif_cols = [col for col in filter6_cols if "Q_dif_final" in col]
    if q_dif_cols and task3_plot_enabled("filter6_qdif_debug"):
        debug_thresholds = {col: [Q_dif_RPC_left, Q_dif_RPC_right] for col in q_dif_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            q_dif_cols,
            debug_thresholds,
            title=(
                f"Task 3 pre-plane Q_dif filter: Q_dif_RPC_left/right "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

    y_cols = [col for col in filter6_cols if "Y_final" in col]
    if y_cols and task3_plot_enabled("filter6_y_debug"):
        debug_thresholds = {col: [Y_RPC_left, Y_RPC_right] for col in y_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            y_cols,
            debug_thresholds,
            title=f"Task 3 pre-plane Y filter: Y_RPC_left/right [tunable] (station {station})",
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

record_filter6_counts(working_df, "before_filter6")

_prof["s_tsum_ref_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------- Sequential per-plane variable filtering ------------------")

plane_dead_masks: dict[int, np.ndarray] = {
    i_plane: np.zeros(len(working_df), dtype=bool)
    for i_plane in range(1, 5)
}

def _task3_plane_block_columns(i_plane: int) -> list[str]:
    return [
        col for col in (
            f"P{i_plane}_T_sum_final",
            f"P{i_plane}_T_dif_final",
            f"P{i_plane}_Q_dif_final",
            f"P{i_plane}_Q_sum_final",
            f"P{i_plane}_Y_final",
        )
        if col in working_df.columns
    ]


def _task3_zero_plane_block(i_plane: int) -> None:
    dead_mask = plane_dead_masks[i_plane]
    if not np.any(dead_mask):
        return
    plane_cols = _task3_plane_block_columns(i_plane)
    if plane_cols:
        working_df.loc[dead_mask, plane_cols] = 0


def _task3_apply_plane_stage(
    *,
    raw_key: str | None,
    column_suffix: str,
    left: float,
    right: float,
) -> None:
    total_events = len(working_df)
    for i_plane in range(1, 5):
        col = f"P{i_plane}_{column_suffix}"
        if raw_key is not None:
            values = np.asarray(plane_raw_values[i_plane][raw_key], dtype=float).copy()
        else:
            values = pd.to_numeric(working_df[col], errors="coerce").fillna(0).to_numpy(dtype=float)

        values[plane_dead_masks[i_plane]] = 0
        out_of_range = (values != 0) & ((values < left) | (values > right))
        values[out_of_range] = 0
        working_df.loc[:, col] = values

        plane_dead_masks[i_plane] |= (values == 0)
        affected = int(np.count_nonzero(plane_dead_masks[i_plane]))
        print(
            f"Plane {i_plane}, {column_suffix}: {affected} out of {total_events} "
            f"events affected ({(affected / total_events) * 100:.2f}%)"
        )
        _task3_zero_plane_block(i_plane)


_task3_apply_plane_stage(
    raw_key="T_sum",
    column_suffix="T_sum_final",
    left=T_sum_RPC_left,
    right=T_sum_RPC_right,
)
_task3_apply_plane_stage(
    raw_key="T_dif",
    column_suffix="T_dif_final",
    left=T_dif_RPC_left,
    right=T_dif_RPC_right,
)
_task3_apply_plane_stage(
    raw_key="Q_dif",
    column_suffix="Q_dif_final",
    left=Q_dif_RPC_left,
    right=Q_dif_RPC_right,
)
_task3_apply_plane_stage(
    raw_key="Q_sum",
    column_suffix="Q_sum_final",
    left=Q_RPC_left,
    right=Q_RPC_right,
)
_task3_apply_plane_stage(
    raw_key=None,
    column_suffix="Y_final",
    left=Y_RPC_left,
    right=Y_RPC_right,
)

if filter6_cols and filter6_before_zero_mask is not None:
    filter6_after_zero_mask = (working_df[filter6_cols] == 0).any(axis=1)
    newly_zeroed = int((filter6_after_zero_mask & ~filter6_before_zero_mask).sum())
    record_filter_metric(
        "filter6_new_zero_rows_pct",
        newly_zeroed,
        len(working_df) if len(working_df) else 0,
    )

record_filter6_counts(working_df, "after_filter6")
_prof["s_filter6_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()

print("----------------------------------------------------------------------")
print("-------- Final plane-combination consistency filter -------------------")
print("----------------------------------------------------------------------")

# Final Task 3 filter: evaluate distinct plane pairs only when both plane
# blocks are fully populated, then zero both planes when any derived
# plane-combination observable falls outside the configured limits.
_plot_plane_combination_filter_by_tt = task3_plot_enabled("plane_combination_filter_by_tt")
if _plot_plane_combination_filter_by_tt:
    plane_combination_tt_before = _task3_filter_tt_series(working_df).copy()
    plane_combination_hist_before = collect_task3_plane_combination_histogram_payload(
        working_df,
        plane_combination_tt_before,
    )
else:
    plane_combination_tt_before = pd.Series(dtype=str)
    plane_combination_hist_before = {}

plane_combination_summary = apply_task3_plane_combination_filter(
    working_df,
    q_sum_sum_left=plane_combination_q_sum_sum_left,
    q_sum_sum_right=plane_combination_q_sum_sum_right,
    q_sum_dif_threshold=plane_combination_q_sum_dif_threshold,
    q_dif_sum_threshold=plane_combination_q_dif_sum_threshold,
    q_dif_dif_threshold=plane_combination_q_dif_dif_threshold,
    t_sum_sum_left=plane_combination_t_sum_sum_left,
    t_sum_sum_right=plane_combination_t_sum_sum_right,
    t_sum_dif_threshold=plane_combination_t_sum_dif_threshold,
    t_dif_sum_threshold=plane_combination_t_dif_sum_threshold,
    t_dif_dif_threshold=plane_combination_t_dif_dif_threshold,
    y_sum_left=plane_combination_y_sum_left,
    y_sum_right=plane_combination_y_sum_right,
    y_dif_threshold=plane_combination_y_dif_threshold,
)
record_activity_metric(
    "plane_combination_filter_rows_affected_pct",
    plane_combination_summary["rows_affected"],
    len(working_df) if len(working_df) else 0,
    label="rows with plane-combination failures",
)
record_activity_metric(
    "plane_combination_filter_values_zeroed_pct",
    plane_combination_summary["values_zeroed"],
    len(working_df) * (5 * plane_combination_summary["tracked_plane_count"])
    if (len(working_df) and plane_combination_summary["tracked_plane_count"])
    else 0,
    label="plane values zeroed by plane-combination filter",
)
record_activity_metric(
    "plane_combination_filter_any_failed_pct",
    plane_combination_summary["failed_pair_any"],
    plane_combination_summary["valid_pair_observations"],
    label="failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_q_sum_sum_failed_pct",
    plane_combination_summary["failed_pair_q_sum_sum"],
    plane_combination_summary["valid_pair_observations"],
    label="Q_sum semisum failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_q_sum_dif_failed_pct",
    plane_combination_summary["failed_pair_q_sum_dif"],
    plane_combination_summary["valid_pair_observations"],
    label="Q_sum semidifference failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_q_dif_sum_failed_pct",
    plane_combination_summary["failed_pair_q_dif_sum"],
    plane_combination_summary["valid_pair_observations"],
    label="Q_dif semisum failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_q_dif_dif_failed_pct",
    plane_combination_summary["failed_pair_q_dif_dif"],
    plane_combination_summary["valid_pair_observations"],
    label="Q_dif semidifference failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_t_sum_sum_failed_pct",
    plane_combination_summary["failed_pair_t_sum_sum"],
    plane_combination_summary["valid_pair_observations"],
    label="T_sum semisum failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_t_sum_dif_failed_pct",
    plane_combination_summary["failed_pair_t_sum_dif"],
    plane_combination_summary["valid_pair_observations"],
    label="T_sum semidifference failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_t_dif_sum_failed_pct",
    plane_combination_summary["failed_pair_t_dif_sum"],
    plane_combination_summary["valid_pair_observations"],
    label="T_dif semisum failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_t_dif_dif_failed_pct",
    plane_combination_summary["failed_pair_t_dif_dif"],
    plane_combination_summary["valid_pair_observations"],
    label="T_dif semidifference failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_y_sum_failed_pct",
    plane_combination_summary["failed_pair_y_sum"],
    plane_combination_summary["valid_pair_observations"],
    label="Y semisum failed plane combinations",
)
record_activity_metric(
    "plane_combination_filter_y_dif_failed_pct",
    plane_combination_summary["failed_pair_y_dif"],
    plane_combination_summary["valid_pair_observations"],
    label="Y semidifference failed plane combinations",
)
print(
    "[plane-combination-filter] "
    f"valid_pair_obs={plane_combination_summary['valid_pair_observations']} "
    f"failed_any={plane_combination_summary['failed_pair_any']} "
    f"rows_affected={plane_combination_summary['rows_affected']} "
    f"flagged_rows={plane_combination_summary['flagged_rows']} "
    f"selected_offenders={plane_combination_summary['selected_offender_planes']}"
)
global_variables["plane_combination_filter_flagged_rows"] = int(
    plane_combination_summary["flagged_rows"]
)
global_variables["plane_combination_filter_selected_offender_planes"] = int(
    plane_combination_summary["selected_offender_planes"]
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
for plane_key, offender_count in plane_combination_summary["selected_offender_counts"].items():
    global_variables[_task3_plane_offender_metric_key(plane_key)] = int(offender_count)
if _plot_plane_combination_filter_by_tt:
    plane_combination_hist_after = collect_task3_plane_combination_histogram_payload(
        working_df,
        plane_combination_tt_before,
    )
    plane_combination_limits = {
        "q_sum_sum": (plane_combination_q_sum_sum_left, plane_combination_q_sum_sum_right),
        "q_sum_dif": (-abs(plane_combination_q_sum_dif_threshold), abs(plane_combination_q_sum_dif_threshold)),
        "q_dif_sum": (-abs(plane_combination_q_dif_sum_threshold), abs(plane_combination_q_dif_sum_threshold)),
        "q_dif_dif": (-abs(plane_combination_q_dif_dif_threshold), abs(plane_combination_q_dif_dif_threshold)),
        "t_sum_sum": (plane_combination_t_sum_sum_left, plane_combination_t_sum_sum_right),
        "t_sum_dif": (-abs(plane_combination_t_sum_dif_threshold), abs(plane_combination_t_sum_dif_threshold)),
        "t_dif_sum": (-abs(plane_combination_t_dif_sum_threshold), abs(plane_combination_t_dif_sum_threshold)),
        "t_dif_dif": (-abs(plane_combination_t_dif_dif_threshold), abs(plane_combination_t_dif_dif_threshold)),
        "y_sum": (plane_combination_y_sum_left, plane_combination_y_sum_right),
        "y_dif": (-abs(plane_combination_y_dif_threshold), abs(plane_combination_y_dif_threshold)),
    }
    fig_idx = plot_task3_plane_combination_filter_by_tt(
        plane_combination_hist_before,
        plane_combination_hist_after,
        basename_no_ext,
        fig_idx,
        base_directories["figure_directory"],
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list,
        limits_by_observable=plane_combination_limits,
    )
_prof["s_plane_combination_filter_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()

# ----------------------------------------------------------------------------------------------------------------
# if stratos_save and station == 2:
if stratos_save:
    print("Saving X and Y for stratos.")
    
    # Select columns that start with "Y_" or match "T<number>_T_dif_final"
    filtered_columns = [col for col in working_df.columns if col.startswith("Y_") or "_T_dif_final" in col or 'datetime' in col]

    # Create a new DataFrame with the selected columns; copy is needed because
    # we rename and scale in-place below.
    filtered_stratos_df = working_df[filtered_columns].copy()

    # Rename "T<number>_T_dif_final" to "X_<number>" and multiply by 200
    filtered_stratos_df.rename(columns=lambda col: f'X_{col.split("_")[0][1:]}' if "_T_dif_final" in col else col, inplace=True)
    filtered_stratos_df.loc[:, filtered_stratos_df.columns.str.startswith("X_")] *= 200

    # Define the save path
    save_stratos = os.path.join(stratos_list_events_directory, f'stratos_data_{save_filename_suffix}.csv')

    # Save DataFrame to CSV (correcting the method name)
    filtered_stratos_df.to_csv(save_stratos, index=False, float_format="%.1f")
    del filtered_stratos_df
# ----------------------------------------------------------------------------------------------------------------

# Same for hexbin
if task3_plot_enabled("filtered_rpc_variables_hexbin"):

    fig, axes = plt.subplots(4, 10, figsize=(40, 20))  # 10 combinations per plane
    axes = axes.flatten()

    for i_plane in range(1, 5):
        # Column names
        t_sum_col = f'P{i_plane}_T_sum_final'
        t_dif_col = f'P{i_plane}_T_dif_final'
        q_sum_col = f'P{i_plane}_Q_sum_final'
        q_dif_col = f'P{i_plane}_Q_dif_final'
        y_col = f'P{i_plane}_Y_final'

        # Filter valid rows (non-zero)
        valid_rows = working_df[[t_sum_col, t_dif_col, q_sum_col, q_dif_col, y_col]].replace(0, np.nan).dropna()
        
        # Extract variables and filter low charge
        cond = valid_rows[q_sum_col] < 150
        t_sum  = valid_rows.loc[cond, t_sum_col]
        t_diff = valid_rows.loc[cond, t_dif_col]
        q_sum  = valid_rows.loc[cond, q_sum_col]
        q_diff = valid_rows.loc[cond, q_dif_col]
        y      = valid_rows.loc[cond, y_col]

        base_idx = (i_plane - 1) * 10  # 10 plots per plane

        plot_pairs = [
            (t_sum,  t_diff, f'{t_sum_col} vs {t_dif_col}'),
            (t_sum,  q_sum,  f'{t_sum_col} vs {q_sum_col}'),
            (t_sum,  y,      f'{t_sum_col} vs {y_col}'),
            (t_diff, q_sum,  f'{t_dif_col} vs {q_sum_col}'),
            (t_diff, y,      f'{t_dif_col} vs {y_col}'),
            (q_sum,  y,      f'{q_sum_col} vs {y_col}'),
            (t_sum,  q_diff, f'{t_sum_col} vs {q_dif_col}'),
            (t_diff, q_diff, f'{t_dif_col} vs {q_dif_col}'),
            (q_diff, y,      f'{q_dif_col} vs {y_col}'),
            (q_sum,  q_diff, f'{q_sum_col} vs {q_dif_col}')
        ]

        for offset, (x, yv, title) in enumerate(plot_pairs):
            ax = axes[base_idx + offset]
            _task3_plot_quantile_hexbin(ax, x, yv, title, gridsize=50, cmap="turbo")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Hexbin Plots for All Variable Combinations by Plane, filtered', fontsize=18)
    if save_plots:
        name_of_file = 'rpc_variables_hexbin_combinations_filtered'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots: plt.show()
    plt.close()

# Path to save the cleaned dataframe
# Create output directory if it does not exist.
os.makedirs(f"{output_directory}", exist_ok=True)
OUT_PATH = f"{output_directory}/listed_{basename_no_ext}.parquet"
KEY = "df"  # HDF5 key name

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# --- Example: your cleaned DataFrame is called working_df ---
# (Here, you would have your data cleaning code before saving)
# working_df = ...

# Remove the columns in the form "T*_T_sum_*", "T*_T_dif_*", "Q*_Q_sum_*", "Q*_Q_dif_*", do a loop from 1 to 4
cols_to_remove = []
for i_plane in range(1, 5):
    for strip in range(1, 5):
        cols_to_remove.append(f'T{i_plane}_T_sum_{strip}')
        cols_to_remove.append(f'T{i_plane}_T_dif_{strip}')
        cols_to_remove.append(f'Q{i_plane}_Q_sum_{strip}')
        cols_to_remove.append(f'Q{i_plane}_Q_dif_{strip}')
working_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    
    

# Pattern for P1_Q_sum_*, P2_Q_sum_*, P3_Q_sum_*, P4_Q_sum_*
Q_SUM_PATTERN = re.compile(r'^P[1-4]_Q_sum_.*$')

# If Q*_F_* and Q*_B_* are zero for all cases, remove the row
Q_cols = collect_columns(working_df.columns, Q_SUM_PATTERN)
if Q_cols and task3_plot_enabled("prefilter_qsum_nonzero_debug"):
    debug_thresholds = {col: [0] for col in Q_cols}
    debug_fig_idx = plot_debug_histograms(
        working_df,
        Q_cols,
        debug_thresholds,
        title=f"Task 3 pre-filter: Q_sum nonzero [NON-TUNABLE] (station {station})",
        out_dir=debug_plot_directory,
        fig_idx=debug_fig_idx,
    )
qsum_total = len(working_df)
qsum_mask = (working_df[Q_cols] != 0).any(axis=1)
working_df = working_df.loc[qsum_mask].copy()
record_filter_metric(
    "q_sum_all_zero_rows_removed_pct",
    qsum_total - int(qsum_mask.sum()),
    original_number_of_events if original_number_of_events else 0,
)

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
list_tt_columns = {
    i_plane: [
        f"P{i_plane}_T_sum_final",
        f"P{i_plane}_T_dif_final",
        f"P{i_plane}_Q_sum_final",
        f"P{i_plane}_Q_dif_final",
        f"P{i_plane}_Y_final",
    ]
    for i_plane in range(1, 5)
}
working_df = compute_tt(working_df, "list_tt", list_tt_columns)
task3_plane_activation_initial = store_task3_plane_activation_snapshot(
    activation_meta={},
    scalar_meta={},
    df=working_df,
    snapshot_label="initial",
    tt_series=pd.to_numeric(working_df["list_tt"], errors="coerce"),
    streamer_high_charge_factor_value=streamer_high_charge_factor,
)
list_tt_total = len(working_df)
if "list_tt" in working_df.columns and task3_plot_enabled("prefilter_list_tt_debug"):
    debug_fig_idx = plot_debug_histograms(
        working_df,
        ["list_tt"],
        {"list_tt": [10]},
        title=f"Task 3 pre-filter: list_tt >= 10 [NON-TUNABLE] (station {station})",
        out_dir=debug_plot_directory,
        fig_idx=debug_fig_idx,
    )
list_tt_mask = working_df["list_tt"].notna() & (working_df["list_tt"] >= 10)
working_df = working_df.loc[list_tt_mask].copy()
record_filter_metric(
    "list_tt_lt_10_rows_removed_pct",
    list_tt_total - int(list_tt_mask.sum()),
    list_tt_total if list_tt_total else 0,
)
working_df.loc[:, "cal_to_list_tt"] = (
    pd.to_numeric(working_df["cal_tt"], errors="coerce").fillna(0).astype(int).astype(str)
    + "_"
    + pd.to_numeric(working_df["list_tt"], errors="coerce").fillna(0).astype(int).astype(str)
)
refresh_global_count_metadata(
    working_df,
    ("cal_tt", "list_tt", "cal_to_list_tt"),
)

task3_plane_activation_filtered = store_task3_plane_activation_snapshot(
    activation_meta=activation_metadata,
    scalar_meta=global_variables,
    df=working_df,
    snapshot_label="filtered",
    tt_series=pd.to_numeric(working_df["list_tt"], errors="coerce"),
    streamer_high_charge_factor_value=streamer_high_charge_factor,
)

if "list_tt" in working_df.columns and task3_any_plot_enabled(
    "list_tt_qsum_pairgrid",
    "all_events_charge_threshold_population",
    "source_list_tt_charge_threshold_population",
    "list_tt_transition_matrices",
    "list_tt_retention_curves",
    "list_tt_minimum_charge_distributions",
    "list_tt_empirical_efficiency_vs_threshold",
    "charge_asymmetry_vs_threshold",
    "interplane_timing_correlation",
    "multiplicity_charge_landscape",
    "streamer_charge_histograms",
    "streamer_prevalence_by_plane",
    "streamer_multiplicity",
    "streamer_contagion_matrix",
    "streamer_efficiency_comparison",
    "fourplane_weakest_charge",
    "fourplane_weakest_plane_identity",
    "efficiency_anatomy_by_charge_band",
    "charge_by_strip_multiplicity_adj",
    "charge_by_strip_multiplicity_dis",
):
    q_sum_final_cols = [
        f"P{i_plane}_Q_sum_final" for i_plane in range(1, 5) if f"P{i_plane}_Q_sum_final" in working_df.columns
    ]
    if q_sum_final_cols:
        streamer_thr_auto = detect_streamer_threshold(working_df, q_sum_final_cols)
        list_tt_charge_limit_effective_runtime = list_tt_charge_limit_effective
        if streamer_thr_auto is not None and np.isfinite(streamer_thr_auto) and streamer_thr_auto > 0:
            list_tt_charge_limit_effective_runtime = float(streamer_thr_auto)
            print(
                "Task 3: using auto-detected streamer threshold as charge-threshold cap: "
                f"{float(streamer_thr_auto):.3g}"
            )

        if list_tt_charge_threshold_mode in {"auto", "quantile"}:
            positive_charge_arrays: list[np.ndarray] = []
            for col_name in q_sum_final_cols:
                q_vals = pd.to_numeric(working_df[col_name], errors="coerce").to_numpy(dtype=float)
                finite_positive = q_vals[np.isfinite(q_vals) & (q_vals > 0)]
                if list_tt_charge_limit_effective_runtime is not None:
                    finite_positive = finite_positive[finite_positive <= float(list_tt_charge_limit_effective_runtime)]
                if finite_positive.size:
                    positive_charge_arrays.append(finite_positive)

            if positive_charge_arrays:
                auto_qmax = float(
                    np.nanpercentile(
                        np.concatenate(positive_charge_arrays),
                        list_tt_charge_threshold_auto_quantile,
                    )
                )
                if np.isfinite(auto_qmax) and auto_qmax > 0:
                    if list_tt_charge_limit_effective_runtime is not None:
                        auto_qmax = min(auto_qmax, float(list_tt_charge_limit_effective_runtime))
                    auto_thresholds = np.linspace(0.0, auto_qmax, list_tt_charge_threshold_auto_steps)
                    auto_thresholds = np.unique(np.round(auto_thresholds, 6))
                    list_tt_charge_thresholds = [float(v) for v in auto_thresholds if np.isfinite(v) and v >= 0]
                    if len(list_tt_charge_thresholds) >= 2:
                        print(
                            "Task 3: auto list_tt_charge_thresholds generated from 0 to "
                            f"Q{list_tt_charge_threshold_auto_quantile:g} ({auto_qmax:.3g}) with "
                            f"{len(list_tt_charge_thresholds)} steps."
                        )
                    else:
                        print(
                            "Warning: auto threshold generation produced too few values; "
                            "falling back to manual list_tt_charge_thresholds."
                        )
                        list_tt_charge_thresholds = list(manual_list_tt_charge_thresholds)
                else:
                    print(
                        "Warning: auto threshold quantile is invalid/non-positive; "
                        "using manual list_tt_charge_thresholds."
                    )
                    list_tt_charge_thresholds = list(manual_list_tt_charge_thresholds)
            else:
                print(
                    "Warning: no positive Q_sum_final values found for auto threshold generation; "
                    "using manual list_tt_charge_thresholds."
                )
                list_tt_charge_thresholds = list(manual_list_tt_charge_thresholds)

        list_tt_charge_thresholds = sorted({float(v) for v in list_tt_charge_thresholds if np.isfinite(v) and v >= 0})
        if list_tt_charge_limit_effective_runtime is not None:
            cap_val = float(list_tt_charge_limit_effective_runtime)
            list_tt_charge_thresholds = [thr for thr in list_tt_charge_thresholds if thr <= cap_val]
            if 0.0 not in list_tt_charge_thresholds:
                list_tt_charge_thresholds.insert(0, 0.0)
            if len(list_tt_charge_thresholds) < 2 and cap_val > 0:
                list_tt_charge_thresholds = [0.0, cap_val]
        if not list_tt_charge_thresholds:
            list_tt_charge_thresholds = [0.0, 5.0, 10.0, 20.0, 50.0]

        tt_numeric = pd.to_numeric(working_df["list_tt"], errors="coerce").fillna(0).astype(int)
        present_tt_values = set(tt_numeric.unique().tolist())
        ordered_tt_values = [
            tt_value for tt_value in TT_COUNT_VALUES if tt_value >= 10 and tt_value in present_tt_values
        ]
        ordered_tt_values.extend(
            sorted(tt_value for tt_value in present_tt_values if tt_value >= 10 and tt_value not in TT_COUNT_VALUES)
        )

        if task3_plot_enabled("list_tt_qsum_pairgrid"):
            if list_tt_qsum_pair_plot_charge_cap is not None:
                qsum_plot_max = float(list_tt_qsum_pair_plot_charge_cap)
            else:
                positive_charge_arrays = []
                for col_name in q_sum_final_cols:
                    charge_vals = pd.to_numeric(working_df[col_name], errors="coerce").to_numpy(dtype=float)
                    finite_positive = charge_vals[np.isfinite(charge_vals) & (charge_vals > 0)]
                    if list_tt_charge_limit_effective_runtime is not None:
                        finite_positive = finite_positive[finite_positive <= float(list_tt_charge_limit_effective_runtime)]
                    if finite_positive.size:
                        positive_charge_arrays.append(finite_positive)

                if positive_charge_arrays:
                    qsum_plot_max = float(np.nanpercentile(np.concatenate(positive_charge_arrays), 99.5))
                    if not np.isfinite(qsum_plot_max) or qsum_plot_max <= 0:
                        qsum_plot_max = 10.0
                else:
                    qsum_plot_max = 10.0
                qsum_plot_max = float(max(10.0, math.ceil(qsum_plot_max / 10.0) * 10.0))
                if list_tt_charge_limit_effective_runtime is not None:
                    qsum_plot_max = float(min(qsum_plot_max, float(list_tt_charge_limit_effective_runtime)))

            for tt_value in ordered_tt_values:
                tt_subset = working_df.loc[tt_numeric == tt_value, q_sum_final_cols]
                if tt_subset.empty:
                    continue

                fig, axes = plt.subplots(4, 4, figsize=(14, 14), squeeze=False)
                last_hexbin = None

                for i_plane in range(1, 5):
                    y_col = f"P{i_plane}_Q_sum_final"
                    for j_plane in range(1, 5):
                        x_col = f"P{j_plane}_Q_sum_final"
                        ax = axes[i_plane - 1, j_plane - 1]

                        if j_plane > i_plane:
                            ax.set_axis_off()
                            continue

                        if x_col not in tt_subset.columns or y_col not in tt_subset.columns:
                            ax.set_axis_off()
                            continue

                        x_vals = pd.to_numeric(tt_subset[x_col], errors="coerce").to_numpy(dtype=float)
                        y_vals = pd.to_numeric(tt_subset[y_col], errors="coerce").to_numpy(dtype=float)

                        if i_plane == j_plane:
                            diag_valid = np.isfinite(x_vals) & (x_vals > 0)
                            if list_tt_qsum_pair_plot_charge_cap is not None:
                                diag_valid &= x_vals < list_tt_qsum_pair_plot_charge_cap
                            diag_vals = x_vals[diag_valid]
                            if diag_vals.size:
                                ax.hist(
                                    diag_vals,
                                    bins=50,
                                    range=(0, qsum_plot_max),
                                    color="tab:blue",
                                    alpha=0.75,
                                )
                            else:
                                ax.text(
                                    0.5,
                                    0.5,
                                    "No positive charge",
                                    ha="center",
                                    va="center",
                                    transform=ax.transAxes,
                                )
                            ax.set_xlim(0, qsum_plot_max)
                            ax.set_title(f"P{i_plane}", fontsize=12)
                            ax.grid(True, alpha=0.2)
                            if j_plane == 1:
                                ax.set_ylabel("Count")
                            if i_plane == 4:
                                ax.set_xlabel("Q_sum_final")
                            continue

                        valid = (
                            np.isfinite(x_vals)
                            & np.isfinite(y_vals)
                            & (x_vals > 0)
                            & (y_vals > 0)
                        )
                        if list_tt_qsum_pair_plot_charge_cap is not None:
                            valid &= (x_vals < list_tt_qsum_pair_plot_charge_cap)
                            valid &= (y_vals < list_tt_qsum_pair_plot_charge_cap)
                        if np.any(valid):
                            if list_tt_qsum_pair_plot_mode == "hexbin":
                                last_hexbin = ax.hexbin(
                                    x_vals[valid],
                                    y_vals[valid],
                                    gridsize=40,
                                    extent=(0, qsum_plot_max, 0, qsum_plot_max),
                                    mincnt=1,
                                    cmap="turbo",
                                )
                                diag_color = "white"
                            else:
                                ax.scatter(
                                    x_vals[valid],
                                    y_vals[valid],
                                    s=2,
                                    alpha=0.25,
                                    color="tab:blue",
                                    linewidths=0,
                                    rasterized=True,
                                )
                                diag_color = "black"
                            ax.plot(
                                [0, qsum_plot_max],
                                [0, qsum_plot_max],
                                linestyle="--",
                                linewidth=0.8,
                                color=diag_color,
                                alpha=0.7,
                            )
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "No overlap",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )

                        ax.set_xlim(0, qsum_plot_max)
                        ax.set_ylim(0, qsum_plot_max)
                        ax.grid(True, alpha=0.15)
                        if i_plane == 4:
                            ax.set_xlabel(f"P{j_plane} Q_sum_final")
                        if j_plane == 1:
                            ax.set_ylabel(f"P{i_plane} Q_sum_final")

                cap_label = (
                    f", Q < {list_tt_qsum_pair_plot_charge_cap:g}"
                    if list_tt_qsum_pair_plot_charge_cap is not None
                    else ""
                )
                fig.suptitle(
                    "Task 3 Q_sum lower-triangular plot for "
                    f"list_tt {tt_value} ({list_tt_qsum_pair_plot_mode}, N={len(tt_subset)}{cap_label})",
                    fontsize=16,
                )
                if list_tt_qsum_pair_plot_mode == "hexbin" and last_hexbin is not None:
                    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.07, top=0.93, wspace=0.28, hspace=0.28)
                    cbar_ax = fig.add_axes([0.91, 0.12, 0.018, 0.74])
                    fig.colorbar(last_hexbin, cax=cbar_ax, label="Counts")
                else:
                    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.07, top=0.93, wspace=0.28, hspace=0.28)

                if save_plots:
                    mode_slug = list_tt_qsum_pair_plot_mode
                    cap_slug = (
                        f"_qcap_{list_tt_qsum_pair_plot_charge_cap:g}".replace(".", "p")
                        if list_tt_qsum_pair_plot_charge_cap is not None
                        else ""
                    )
                    name_of_file = f"list_tt_{tt_value}_qsum_lower_triangular_{mode_slug}{cap_slug}.png"
                    final_filename = f"{fig_idx}_{name_of_file}"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)

                if show_plots:
                    plt.show()
                plt.close(fig)

        threshold_labels = [f"Q > {threshold:g}" for threshold in list_tt_charge_thresholds]
        threshold_tt_by_label: dict[str, pd.Series] = {}
        if task3_any_plot_enabled(
            "all_events_charge_threshold_population",
            "list_tt_transition_matrices",
            "list_tt_empirical_efficiency_vs_threshold",
            "full_topology_threshold_retention",
            "tsum_coincidence_window_histograms",
            "tsum_coincidence_window_vs_threshold",
        ):
            for threshold, threshold_label in zip(list_tt_charge_thresholds, threshold_labels):
                threshold_tt_by_label[threshold_label] = compute_qsum_threshold_tt(
                    working_df[q_sum_final_cols], threshold
                )

        threshold_full_patterns_by_label: dict[str, pd.Series] = {}
        if task3_any_plot_enabled(
            "full_topology_threshold_retention",
            "full_topology_exact_retention",
            "full_topology_class_fraction",
        ):
            for threshold, threshold_label in zip(list_tt_charge_thresholds, threshold_labels):
                threshold_full_patterns_by_label[threshold_label] = compute_qsum_threshold_full_strip_patterns(
                    working_df,
                    threshold,
                )

        threshold_tsum_window_by_label: dict[str, pd.Series] = {}
        threshold_tsum_active_counts_by_label: dict[str, pd.Series] = {}
        if task3_any_plot_enabled(
            "tsum_coincidence_window_histograms",
            "tsum_coincidence_window_vs_threshold",
        ):
            for threshold, threshold_label in zip(list_tt_charge_thresholds, threshold_labels):
                window_series, active_counts = compute_qsum_threshold_tsum_window(working_df, threshold)
                threshold_tsum_window_by_label[threshold_label] = window_series
                threshold_tsum_active_counts_by_label[threshold_label] = active_counts

        if task3_plot_enabled("all_events_charge_threshold_population"):
            overall_threshold_counts = {
                threshold_label: threshold_tt_by_label[threshold_label].value_counts()
                for threshold_label in threshold_labels
            }
            overall_counts_df = pd.DataFrame(overall_threshold_counts).fillna(0).astype(int)
            overall_row_order = [
                tt_value for tt_value in TT_COUNT_VALUES if tt_value in overall_counts_df.index
            ]
            overall_row_order.extend(
                sorted(tt_value for tt_value in overall_counts_df.index if tt_value not in TT_COUNT_VALUES)
            )
            if overall_row_order:
                overall_counts_df = overall_counts_df.reindex(index=overall_row_order, fill_value=0)
                overall_counts_df = overall_counts_df.loc[overall_counts_df.sum(axis=1) > 0]
                fig_idx = plot_population_table(
                    overall_counts_df,
                    title=(
                        "All events: charge-filtered plane combinations "
                        "(all planes in row satisfy Q > threshold)"
                    ),
                    filename_suffix="all_events_charge_threshold_combo_population",
                    fig_idx=fig_idx,
                    base_dir=base_directories["figure_directory"],
                    show_plots=show_plots,
                    save_plots=save_plots,
                    plot_list=plot_list,
                )

        if task3_plot_enabled("source_list_tt_charge_threshold_population"):
            for source_tt in ordered_tt_values:
                source_subset = working_df.loc[tt_numeric == source_tt, q_sum_final_cols]
                if source_subset.empty:
                    continue

                source_threshold_counts: dict[str, pd.Series] = {}
                for threshold, threshold_label in zip(list_tt_charge_thresholds, threshold_labels):
                    threshold_tt = compute_qsum_threshold_tt(source_subset, threshold)
                    source_threshold_counts[threshold_label] = threshold_tt.value_counts()

                source_counts_df = pd.DataFrame(source_threshold_counts).fillna(0).astype(int)
                source_row_order = [
                    tt_value for tt_value in TT_COUNT_VALUES if tt_value in source_counts_df.index
                ]
                source_row_order.extend(
                    sorted(tt_value for tt_value in source_counts_df.index if tt_value not in TT_COUNT_VALUES)
                )
                if not source_row_order:
                    continue
                source_counts_df = source_counts_df.reindex(index=source_row_order, fill_value=0)
                source_counts_df = source_counts_df.loc[source_counts_df.sum(axis=1) > 0]
                if source_counts_df.empty:
                    continue

                fig_idx = plot_population_table(
                    source_counts_df,
                    title=(
                        f"Source list_tt {source_tt}: charge-filtered plane combinations "
                        f"(N={len(source_subset)})"
                    ),
                    filename_suffix=f"list_tt_{source_tt}_charge_threshold_combo_population",
                    fig_idx=fig_idx,
                    base_dir=base_directories["figure_directory"],
                    show_plots=show_plots,
                    save_plots=save_plots,
                    plot_list=plot_list,
                )

        if task3_plot_enabled("list_tt_transition_matrices"):
            transition_row_order = ordered_tt_values
            transition_col_order = list(TT_COUNT_VALUES)
            for threshold_label in threshold_labels:
                threshold_tt = threshold_tt_by_label[threshold_label]
                transition_counts = pd.crosstab(tt_numeric, threshold_tt)
                transition_counts = transition_counts.reindex(
                    index=transition_row_order,
                    columns=transition_col_order,
                    fill_value=0,
                )
                transition_counts = transition_counts.loc[:, transition_counts.sum(axis=0) > 0]
                transition_counts = transition_counts.loc[transition_counts.sum(axis=1) > 0]
                if transition_counts.empty:
                    continue

                threshold_value = threshold_label.replace("Q > ", "")
                fig_idx = plot_population_table(
                    transition_counts,
                    title=f"Source list_tt to charge-filtered combination at {threshold_label}",
                    filename_suffix=f"list_tt_transition_matrix_qgt_{threshold_value}".replace(".", "p"),
                    fig_idx=fig_idx,
                    base_dir=base_directories["figure_directory"],
                    row_label="Source list_tt",
                    col_label="Charge-filtered combination",
                    show_plots=show_plots,
                    save_plots=save_plots,
                    plot_list=plot_list,
                )

        min_charge_by_source_tt: dict[int, pd.Series] = {}
        if task3_any_plot_enabled(
            "list_tt_retention_curves",
            "list_tt_minimum_charge_distributions",
        ):
            for source_tt in ordered_tt_values:
                source_mask = tt_numeric == source_tt
                source_subset = working_df.loc[source_mask, q_sum_final_cols]
                if source_subset.empty:
                    continue
                min_charge_by_source_tt[source_tt] = compute_source_tt_min_charge(source_subset, source_tt)

        if task3_plot_enabled("list_tt_retention_curves") and min_charge_by_source_tt:
            fig, ax = plt.subplots(figsize=(11, 6.5))
            cmap = plt.get_cmap("tab20")
            for idx, source_tt in enumerate(ordered_tt_values):
                min_charge = min_charge_by_source_tt.get(source_tt)
                if min_charge is None:
                    continue
                min_charge_vals = pd.to_numeric(min_charge, errors="coerce").to_numpy(dtype=float)
                baseline_mask = np.isfinite(min_charge_vals) & (min_charge_vals > 0)
                baseline_count = int(baseline_mask.sum())
                if baseline_count == 0:
                    continue
                retention = [
                    float(np.count_nonzero(np.isfinite(min_charge_vals) & (min_charge_vals > threshold))) / baseline_count
                    for threshold in list_tt_charge_thresholds
                ]
                ax.plot(
                    list_tt_charge_thresholds,
                    retention,
                    marker="o",
                    linewidth=1.7,
                    alpha=0.9,
                    color=cmap(idx % 20),
                    label=f"{source_tt}",
                )

            ax.set_xlabel("Charge threshold")
            ax.set_ylabel("Retention fraction relative to Q > 0")
            ax.set_title("Retention curves by source list_tt")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.25)
            ax.legend(title="Source list_tt", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
            fig.tight_layout(rect=[0, 0, 0.84, 1])

            if save_plots:
                final_filename = f"{fig_idx}_list_tt_retention_curves.png"
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
            if show_plots:
                plt.show()
            plt.close(fig)

        if task3_plot_enabled("list_tt_minimum_charge_distributions") and min_charge_by_source_tt:
            positive_min_charge_arrays = []
            for min_charge in min_charge_by_source_tt.values():
                min_charge_vals = pd.to_numeric(min_charge, errors="coerce").to_numpy(dtype=float)
                positive_min_charge = min_charge_vals[np.isfinite(min_charge_vals) & (min_charge_vals > 0)]
                if positive_min_charge.size:
                    positive_min_charge_arrays.append(positive_min_charge)

            if positive_min_charge_arrays:
                min_charge_plot_max = float(np.nanpercentile(np.concatenate(positive_min_charge_arrays), 99.5))
                min_charge_plot_max = max(min_charge_plot_max, max(list_tt_charge_thresholds) if list_tt_charge_thresholds else 0)
                if not np.isfinite(min_charge_plot_max) or min_charge_plot_max <= 0:
                    min_charge_plot_max = 10.0
            else:
                min_charge_plot_max = max(max(list_tt_charge_thresholds), 10.0) if list_tt_charge_thresholds else 10.0
            min_charge_plot_max = float(max(10.0, math.ceil(min_charge_plot_max / 10.0) * 10.0))
            if list_tt_charge_limit_effective_runtime is not None:
                min_charge_plot_max = float(min(min_charge_plot_max, float(list_tt_charge_limit_effective_runtime)))

            n_sources = len(min_charge_by_source_tt)
            ncols = 4
            nrows = int(math.ceil(n_sources / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
            axes_flat = axes.flatten()

            for ax, source_tt in zip(axes_flat, ordered_tt_values):
                min_charge = min_charge_by_source_tt.get(source_tt)
                if min_charge is None:
                    ax.set_axis_off()
                    continue
                min_charge_vals = pd.to_numeric(min_charge, errors="coerce").to_numpy(dtype=float)
                hist_vals = min_charge_vals[np.isfinite(min_charge_vals) & (min_charge_vals > 0)]
                if hist_vals.size:
                    ax.hist(
                        hist_vals,
                        bins=50,
                        range=(0, min_charge_plot_max),
                        color="tab:blue",
                        alpha=0.75,
                    )
                    for threshold in list_tt_charge_thresholds:
                        ax.axvline(threshold, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
                else:
                    ax.text(0.5, 0.5, "No positive charge", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, min_charge_plot_max)
                ax.set_title(f"list_tt {source_tt} (N={len(hist_vals)})", fontsize=10)
                ax.grid(True, alpha=0.2)

            for ax in axes_flat[n_sources:]:
                ax.set_axis_off()

            for row_idx in range(nrows):
                axes[row_idx, 0].set_ylabel("Counts")
            for col_idx in range(ncols):
                axes[-1, col_idx].set_xlabel("Minimum active-plane Q_sum_final")

            fig.suptitle("Minimum active-plane charge by source list_tt", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            if save_plots:
                final_filename = f"{fig_idx}_list_tt_minimum_charge_distributions.png"
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
            if show_plots:
                plt.show()
            plt.close(fig)

        if task3_plot_enabled("list_tt_empirical_efficiency_vs_threshold"):
            plane_to_thresholds: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
            plane_to_efficiencies: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
            plane_to_errors: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
            plane_to_counts: dict[int, list[tuple[int, int]]] = {1: [], 2: [], 3: [], 4: []}
            for threshold, threshold_label in zip(list_tt_charge_thresholds, threshold_labels):
                tt_counts = threshold_tt_by_label[threshold_label].value_counts()
                per_plane = compute_empirical_efficiency_from_tt_counts(tt_counts)
                for plane in (1, 2, 3, 4):
                    efficiency, error, n_three, n_four = per_plane[plane]
                    plane_to_thresholds[plane].append(float(threshold))
                    plane_to_efficiencies[plane].append(float(efficiency))
                    plane_to_errors[plane].append(float(error))
                    plane_to_counts[plane].append((n_three, n_four))

            has_any_valid = False
            y_low_candidates: list[float] = []
            y_high_candidates: list[float] = []
            for plane in (1, 2, 3, 4):
                x_vals = np.asarray(plane_to_thresholds[plane], dtype=float)
                y_vals = np.asarray(plane_to_efficiencies[plane], dtype=float)
                y_errs = np.asarray(plane_to_errors[plane], dtype=float)
                valid = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(y_errs)
                if np.any(valid):
                    has_any_valid = True
                    y_low_candidates.append(float(np.nanmin(y_vals[valid] - y_errs[valid])))
                    y_high_candidates.append(float(np.nanmax(y_vals[valid] + y_errs[valid])))

            if has_any_valid:
                fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.0), sharex=True)
                ax_top, ax_bottom = axes
                plane_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
                missing_plane_tt_label = {1: "234", 2: "134", 3: "124", 4: "123"}
                for plane in (1, 2, 3, 4):
                    x_vals = np.asarray(plane_to_thresholds[plane], dtype=float)
                    y_vals = np.asarray(plane_to_efficiencies[plane], dtype=float)
                    y_errs = np.asarray(plane_to_errors[plane], dtype=float)
                    valid = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(y_errs)
                    if not np.any(valid):
                        continue

                    for axis in (ax_top, ax_bottom):
                        axis.errorbar(
                            x_vals[valid],
                            y_vals[valid],
                            yerr=y_errs[valid],
                            fmt="o-",
                            linewidth=1.6,
                            markersize=4.5,
                            capsize=3.5,
                            color=plane_colors[plane],
                            label=f"eff_{plane} = 1 - N({missing_plane_tt_label[plane]})/N(1234)",
                        )

                    valid_indices = [idx for idx, ok in enumerate(valid) if ok]
                    for idx in valid_indices:
                        x_val = x_vals[idx]
                        y_val = y_vals[idx]
                        n_three, n_four = plane_to_counts[plane][idx]
                        for axis in (ax_top, ax_bottom):
                            axis.annotate(
                                f"{n_three}/{n_four}",
                                xy=(x_val, y_val),
                                xytext=(0, 6),
                                textcoords="offset points",
                                ha="center",
                                fontsize=7,
                                alpha=0.8,
                                color=plane_colors[plane],
                            )

                ax_top.set_ylabel("Empirical efficiency")
                ax_top.set_title("Threshold scan: per-plane empirical efficiencies (zoom 0-1)")
                ax_top.grid(True, alpha=0.25)
                ax_top.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
                ax_top.set_ylim(0.0, 1.0)
                ax_top.legend(loc="upper right", fontsize=9)

                ax_bottom.set_xlabel("Charge threshold")
                ax_bottom.set_ylabel("Empirical efficiency")
                ax_bottom.set_title("Threshold scan: per-plane empirical efficiencies (full range)")
                ax_bottom.grid(True, alpha=0.25)
                ax_bottom.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
                ax_bottom.legend(loc="upper right", fontsize=9)

                y_low = float(min(y_low_candidates))
                y_high = float(max(y_high_candidates))
                y_margin = max(0.03, 0.08 * max(y_high - y_low, 0.1))
                ax_bottom.set_ylim(y_low - y_margin, y_high + y_margin)

                fig.tight_layout()

                if save_plots:
                    final_filename = f"{fig_idx}_list_tt_empirical_efficiency_vs_threshold.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig)

        if task3_plot_enabled("tsum_coincidence_window_histograms") and threshold_tsum_window_by_label:
            all_window_vals: list[np.ndarray] = []
            for threshold_label in threshold_labels:
                w_vals = threshold_tsum_window_by_label[threshold_label].to_numpy(dtype=float)
                valid = np.isfinite(w_vals)
                if np.any(valid):
                    all_window_vals.append(w_vals[valid])

            if all_window_vals:
                window_max = float(np.nanpercentile(np.concatenate(all_window_vals), 99.5))
                if not np.isfinite(window_max) or window_max <= 0:
                    window_max = float(max(tsum_window_cut_values_ns) * 1.5)
            else:
                window_max = float(max(tsum_window_cut_values_ns) * 1.5)
            window_max = max(window_max, float(max(tsum_window_cut_values_ns) * 1.1))

            baseline_label = threshold_labels[0]
            combo_counts = threshold_tt_by_label.get(baseline_label, pd.Series(dtype=int)).value_counts().to_dict()
            combo_order = [
                int(tt_value)
                for tt_value, count in sorted(combo_counts.items(), key=lambda item: item[1], reverse=True)
                if len(tt_value_to_planes(tt_value)) >= 2 and int(count) >= tsum_window_combo_min_count
            ][:tsum_window_max_plane_combinations]

            if combo_order:
                nrows = len(combo_order)
                ncols = len(threshold_labels)
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(3.0 * max(1, ncols), 2.2 * max(1, nrows)),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                )
                panel_density_max: list[float] = []

                for i_row, combo_tt in enumerate(combo_order):
                    for j_col, threshold_label in enumerate(threshold_labels):
                        ax = axes[i_row, j_col]
                        tt_vals = threshold_tt_by_label[threshold_label].to_numpy(dtype=int)
                        w_vals = threshold_tsum_window_by_label[threshold_label].to_numpy(dtype=float)
                        valid = np.isfinite(w_vals) & (tt_vals == int(combo_tt))
                        hist_vals = w_vals[valid]

                        if hist_vals.size > 0:
                            hist_n, _, _ = ax.hist(
                                hist_vals,
                                bins=tsum_window_hist_bins,
                                range=(0.0, window_max),
                                color="tab:blue",
                                alpha=0.8,
                                density=True,
                            )
                            if hist_n.size > 0 and np.isfinite(hist_n).any():
                                panel_density_max.append(float(np.nanmax(hist_n)))
                            for cut_val in tsum_window_cut_values_ns:
                                ax.axvline(cut_val, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
                        else:
                            ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes, fontsize=7)

                        ax.set_xlim(0.0, window_max)
                        ax.grid(True, alpha=0.2)

                        if i_row == 0:
                            ax.set_title(f"{threshold_label}", fontsize=9)
                        if j_col == 0:
                            base_count = int(combo_counts.get(combo_tt, 0))
                            ax.set_ylabel(f"TT {combo_tt}\nN0={base_count}\nDensity", fontsize=8)

                # Use a robust shared y-limit so sparse single-bin panels remain readable.
                if panel_density_max:
                    y_ref = float(np.nanpercentile(np.asarray(panel_density_max, dtype=float), 90))
                    if not np.isfinite(y_ref) or y_ref <= 0:
                        y_ref = float(np.nanmax(np.asarray(panel_density_max, dtype=float)))
                    if np.isfinite(y_ref) and y_ref > 0:
                        y_top = max(0.05, 1.2 * y_ref)
                        axes[0, 0].set_ylim(0.0, y_top)

                for j_col in range(ncols):
                    axes[-1, j_col].set_xlabel("T_sum coincidence window [ns]")

                fig.suptitle(
                    "T_sum coincidence-window histograms\n"
                    "Rows: plane combinations (TT), Columns: Q thresholds",
                    fontsize=13,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                if save_plots:
                    final_filename = f"{fig_idx}_tsum_coincidence_window_histograms_by_plane_combination.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig)

        if task3_plot_enabled("tsum_coincidence_window_vs_threshold") and threshold_tsum_window_by_label:
            x_vals = np.asarray(list_tt_charge_thresholds, dtype=float)
            percentile_curves: dict[float, list[float]] = {pct: [] for pct in tsum_window_percentiles}
            acceptance_curves: dict[float, list[float]] = {cut_val: [] for cut_val in tsum_window_cut_values_ns}
            reference_acceptance: list[float] = []
            n_events_per_threshold: list[int] = []

            for threshold_label in threshold_labels:
                w_vals = threshold_tsum_window_by_label[threshold_label].to_numpy(dtype=float)
                valid = np.isfinite(w_vals)
                hist_vals = w_vals[valid]
                n_events_per_threshold.append(int(hist_vals.size))

                if hist_vals.size == 0:
                    for pct in tsum_window_percentiles:
                        percentile_curves[pct].append(np.nan)
                    for cut_val in tsum_window_cut_values_ns:
                        acceptance_curves[cut_val].append(np.nan)
                    reference_acceptance.append(np.nan)
                    continue

                for pct in tsum_window_percentiles:
                    percentile_curves[pct].append(float(np.nanpercentile(hist_vals, pct)))
                for cut_val in tsum_window_cut_values_ns:
                    acceptance_curves[cut_val].append(float(np.mean(hist_vals <= cut_val)))
                reference_acceptance.append(float(np.mean(hist_vals <= tsum_window_reference_ns)))

            fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True)
            ax_top, ax_bottom = axes

            pct_cmap = plt.get_cmap("viridis")
            for idx, pct in enumerate(tsum_window_percentiles):
                y_vals = np.asarray(percentile_curves[pct], dtype=float)
                valid = np.isfinite(y_vals)
                if np.any(valid):
                    ax_top.plot(
                        x_vals[valid],
                        y_vals[valid],
                        marker="o",
                        linewidth=1.9,
                        color=pct_cmap(idx / max(len(tsum_window_percentiles) - 1, 1)),
                        label=f"P{pct:g}",
                    )
            ax_top.set_ylabel("Coincidence window [ns]")
            ax_top.set_title("T_sum coincidence-window quantiles vs Q threshold")
            ax_top.grid(True, alpha=0.25)
            ax_top.legend(loc="upper right", fontsize=9)

            cut_cmap = plt.get_cmap("plasma")
            for idx, cut_val in enumerate(tsum_window_cut_values_ns):
                y_vals = np.asarray(acceptance_curves[cut_val], dtype=float)
                valid = np.isfinite(y_vals)
                if np.any(valid):
                    ax_bottom.plot(
                        x_vals[valid],
                        y_vals[valid],
                        marker="o",
                        linewidth=1.9,
                        color=cut_cmap(idx / max(len(tsum_window_cut_values_ns) - 1, 1)),
                        label=f"window <= {cut_val:g} ns",
                    )

            y_ref = np.asarray(reference_acceptance, dtype=float)
            valid_ref = np.isfinite(y_ref)
            if np.any(valid_ref):
                ax_bottom.plot(
                    x_vals[valid_ref],
                    y_ref[valid_ref],
                    linestyle="--",
                    linewidth=1.5,
                    color="black",
                    alpha=0.75,
                    label=f"window <= {tsum_window_reference_ns:g} ns (reference)",
                )

            ax_bottom.set_xlabel("Charge threshold")
            ax_bottom.set_ylabel("Fraction within window")
            ax_bottom.set_ylim(0.0, 1.05)
            ax_bottom.set_title("Coincidence-window acceptance vs Q threshold")
            ax_bottom.grid(True, alpha=0.25)
            ax_bottom.legend(loc="upper right", fontsize=8)

            # Annotate usable event counts for context.
            for x_val, n_val in zip(x_vals, n_events_per_threshold):
                ax_bottom.annotate(
                    f"N={n_val}",
                    xy=(x_val, 0.02),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    alpha=0.7,
                )

            fig.tight_layout()

            if save_plots:
                final_filename = f"{fig_idx}_tsum_coincidence_window_vs_threshold.png"
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
            if show_plots:
                plt.show()
            plt.close(fig)

            # Same study split by plane combination (thresholded TT code).
            if threshold_tt_by_label:
                baseline_label = threshold_labels[0]
                baseline_tt = threshold_tt_by_label[baseline_label]
                baseline_window = threshold_tsum_window_by_label[baseline_label]
                baseline_valid = np.isfinite(baseline_window.to_numpy(dtype=float))

                combo_counts = (
                    baseline_tt[baseline_valid]
                    .value_counts()
                    .to_dict()
                )
                combo_order = [
                    int(tt_value)
                    for tt_value, count in sorted(combo_counts.items(), key=lambda item: item[1], reverse=True)
                    if len(tt_value_to_planes(tt_value)) >= 2 and int(count) >= tsum_window_combo_min_count
                ][:tsum_window_max_plane_combinations]

                if combo_order:
                    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.6), sharex=True)
                    ax_top, ax_bottom = axes
                    cmap = plt.get_cmap("tab20")

                    for idx, combo_tt in enumerate(combo_order):
                        combo_window_median: list[float] = []
                        combo_window_p90: list[float] = []
                        combo_acceptance_ref: list[float] = []

                        for threshold_label in threshold_labels:
                            tt_series = threshold_tt_by_label[threshold_label]
                            w_series = threshold_tsum_window_by_label[threshold_label]
                            mask = tt_series.eq(combo_tt).to_numpy(dtype=bool)
                            w_vals = w_series.to_numpy(dtype=float)
                            valid = mask & np.isfinite(w_vals)
                            combo_vals = w_vals[valid]
                            if combo_vals.size == 0:
                                combo_window_median.append(np.nan)
                                combo_window_p90.append(np.nan)
                                combo_acceptance_ref.append(np.nan)
                                continue
                            combo_window_median.append(float(np.nanpercentile(combo_vals, 50)))
                            combo_window_p90.append(float(np.nanpercentile(combo_vals, 90)))
                            combo_acceptance_ref.append(float(np.mean(combo_vals <= tsum_window_reference_ns)))

                        color = cmap(idx % 20)
                        median_vals = np.asarray(combo_window_median, dtype=float)
                        p90_vals = np.asarray(combo_window_p90, dtype=float)
                        acc_vals = np.asarray(combo_acceptance_ref, dtype=float)
                        valid_median = np.isfinite(median_vals)
                        valid_p90 = np.isfinite(p90_vals)
                        valid_acc = np.isfinite(acc_vals)

                        base_count = int(combo_counts.get(combo_tt, 0))
                        label_base = f"TT {combo_tt} (N0={base_count})"
                        if np.any(valid_median):
                            ax_top.plot(
                                x_vals[valid_median],
                                median_vals[valid_median],
                                marker="o",
                                linewidth=1.8,
                                color=color,
                                label=f"{label_base} median",
                            )
                        if np.any(valid_p90):
                            ax_top.plot(
                                x_vals[valid_p90],
                                p90_vals[valid_p90],
                                marker=".",
                                linestyle="--",
                                linewidth=1.1,
                                color=color,
                                alpha=0.8,
                                label=f"TT {combo_tt} P90",
                            )
                        if np.any(valid_acc):
                            ax_bottom.plot(
                                x_vals[valid_acc],
                                acc_vals[valid_acc],
                                marker="o",
                                linewidth=1.9,
                                color=color,
                                label=label_base,
                            )

                    ax_top.set_ylabel("Coincidence window [ns]")
                    ax_top.set_title("T_sum window vs Q threshold by plane combination (TT)")
                    ax_top.grid(True, alpha=0.25)
                    ax_top.legend(loc="upper right", fontsize=7, ncol=2)

                    ax_bottom.set_xlabel("Charge threshold")
                    ax_bottom.set_ylabel(f"Fraction with window <= {tsum_window_reference_ns:g} ns")
                    ax_bottom.set_ylim(0.0, 1.05)
                    ax_bottom.set_title("Coincidence-window acceptance by plane combination")
                    ax_bottom.grid(True, alpha=0.25)
                    ax_bottom.legend(loc="upper right", fontsize=8)

                    fig.tight_layout()

                    if save_plots:
                        final_filename = f"{fig_idx}_tsum_coincidence_window_vs_threshold_by_plane_combination.png"
                        fig_idx += 1
                        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                        plot_list.append(save_fig_path)
                        save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
                    if show_plots:
                        plt.show()
                    plt.close(fig)

        if task3_plot_enabled("plane_charge_fraction_vs_total_charge_threshold_scan"):
            q_cols = [f"P{plane}_Q_sum_final" for plane in range(1, 5) if f"P{plane}_Q_sum_final" in working_df.columns]
            if len(q_cols) >= 2:
                q_df = working_df[q_cols].apply(pd.to_numeric, errors="coerce")
                q_df = q_df.where(q_df > 0, 0.0).fillna(0.0)
                q_total = q_df.sum(axis=1)

                scan_thresholds = sorted(list_tt_charge_thresholds)[:plane_charge_fraction_max_panels]
                ncols = len(scan_thresholds)
                nrows = len(q_cols)
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(3.1 * max(1, ncols), 2.5 * max(1, nrows)),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                )

                plane_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
                plane_labels = {1: "Plane 1", 2: "Plane 2", 3: "Plane 3", 4: "Plane 4"}

                q_total_max = float(np.nanpercentile(q_total.to_numpy(dtype=float), 99.5)) if len(q_total) else 0.0
                if not np.isfinite(q_total_max) or q_total_max <= 0:
                    q_total_max = float(max(scan_thresholds) * 5.0 + 50.0)

                threshold_to_idx: dict[float, np.ndarray] = {}
                for thr in scan_thresholds:
                    # Apply threshold plane-by-plane with logical AND, consistent with
                    # the charge-threshold logic used elsewhere in Task 3.
                    mask = pd.Series(True, index=q_df.index, dtype=bool)
                    for q_col in q_cols:
                        mask &= q_df[q_col].gt(float(thr))
                    idx = np.flatnonzero(mask.to_numpy(dtype=bool))
                    if idx.size > plane_charge_fraction_scatter_max_points:
                        keep = np.linspace(0, idx.size - 1, plane_charge_fraction_scatter_max_points, dtype=int)
                        idx = idx[keep]
                    threshold_to_idx[float(thr)] = idx

                for i_row, q_col in enumerate(q_cols):
                    plane = int(q_col.split("_")[0].replace("P", ""))
                    color = plane_colors.get(plane, "black")
                    for j_col, thr in enumerate(scan_thresholds):
                        ax = axes[i_row, j_col]
                        idx = threshold_to_idx.get(float(thr), np.array([], dtype=int))
                        if idx.size == 0:
                            ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                            ax.grid(True, alpha=0.2)
                            continue

                        x_vals = q_total.to_numpy(dtype=float)[idx]
                        denom = np.where(x_vals > 0, x_vals, np.nan)

                        plane_q = q_df[q_col].to_numpy(dtype=float)[idx]
                        frac = np.divide(plane_q, denom, out=np.full_like(plane_q, np.nan), where=np.isfinite(denom))
                        valid = np.isfinite(frac) & np.isfinite(x_vals)
                        if np.any(valid):
                            ax.scatter(
                                x_vals[valid],
                                frac[valid],
                                s=4,
                                alpha=0.25,
                                color=color,
                                linewidths=0,
                                rasterized=True,
                            )

                        ax.set_ylim(0.0, 1.02)
                        ax.set_xlim(0.0, q_total_max)
                        ax.grid(True, alpha=0.2)

                        if i_row == 0:
                            ax.set_title(f"All planes Q > {thr:g}\nN={len(idx)}", fontsize=9)
                        if j_col == 0:
                            ax.set_ylabel(f"{plane_labels.get(plane, f'Plane {plane}')}\nQ_plane / Q_total", fontsize=9)

                for j_col in range(ncols):
                    axes[-1, j_col].set_xlabel("Total event charge Q_total")

                fig.suptitle(
                    "Per-plane charge fraction vs total event charge\n"
                    "Rows: planes, Columns: charge thresholds",
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                if save_plots:
                    final_filename = f"{fig_idx}_plane_charge_fraction_vs_total_charge_threshold_scan.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig)

        if task3_any_plot_enabled(
            "full_topology_threshold_retention",
            "full_topology_exact_retention",
            "full_topology_class_fraction",
        ) and threshold_full_patterns_by_label:
            baseline_label = threshold_labels[0]
            baseline_counts = threshold_full_patterns_by_label[baseline_label].value_counts()
            threshold_counts_by_label = {
                label: threshold_full_patterns_by_label[label].value_counts()
                for label in threshold_labels
            }

            per_mask_class_patterns: dict[str, dict[str, list[str]]] = {}
            per_mask_class_baseline: dict[str, dict[str, int]] = {}
            per_mask_total_baseline: dict[str, int] = {}
            for pattern, count in baseline_counts.items():
                active_mask, topo_class, _ = classify_full_strip_topology(str(pattern))
                if active_mask == "0000" or topo_class in {"empty", "invalid"}:
                    continue
                if active_mask not in per_mask_class_patterns:
                    per_mask_class_patterns[active_mask] = {}
                    per_mask_class_baseline[active_mask] = {}
                    per_mask_total_baseline[active_mask] = 0
                per_mask_class_patterns[active_mask].setdefault(topo_class, []).append(str(pattern))
                per_mask_class_baseline[active_mask][topo_class] = (
                    per_mask_class_baseline[active_mask].get(topo_class, 0) + int(count)
                )
                per_mask_total_baseline[active_mask] += int(count)

            mask_order = [
                mask
                for mask, total in sorted(
                    per_mask_total_baseline.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if total >= full_topology_min_baseline_count
            ][:full_topology_max_masks]

            if mask_order:
                ncols = 2
                nrows = int(math.ceil(len(mask_order) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.1 * nrows), squeeze=False, sharex=True)
                axes_flat = axes.flatten()
                class_colors = {
                    "single_strip_smooth": "tab:blue",
                    "single_strip_rough": "tab:orange",
                    "single_strip_zigzag": "tab:red",
                    "multi_strip": "tab:gray",
                }
                class_labels = {
                    "single_strip_smooth": "single-strip smooth",
                    "single_strip_rough": "single-strip rough",
                    "single_strip_zigzag": "single-strip zigzag",
                    "multi_strip": "multi-strip",
                }
                class_order = [
                    "single_strip_smooth",
                    "single_strip_rough",
                    "single_strip_zigzag",
                    "multi_strip",
                ]

                for ax, mask in zip(axes_flat, mask_order):
                    class_to_patterns = per_mask_class_patterns.get(mask, {})
                    class_to_baseline = per_mask_class_baseline.get(mask, {})
                    drawn = 0
                    for topo_class in class_order:
                        baseline_n = int(class_to_baseline.get(topo_class, 0))
                        if baseline_n <= 0:
                            continue
                        patterns_in_class = class_to_patterns.get(topo_class, [])
                        if not patterns_in_class:
                            continue

                        retention_vals: list[float] = []
                        for threshold_label in threshold_labels:
                            counts = threshold_counts_by_label[threshold_label]
                            surviving = int(sum(int(counts.get(pattern, 0)) for pattern in patterns_in_class))
                            retention_vals.append(float(surviving) / float(baseline_n))

                        ax.plot(
                            list_tt_charge_thresholds,
                            retention_vals,
                            marker="o",
                            linewidth=1.9,
                            markersize=4,
                            color=class_colors[topo_class],
                            label=f"{class_labels[topo_class]} (N0={baseline_n})",
                        )
                        drawn += 1

                    ax.set_title(f"Active mask {mask} (N0={per_mask_total_baseline.get(mask, 0)})")
                    ax.set_ylim(0.0, 1.05)
                    ax.grid(True, alpha=0.25)
                    if drawn > 0:
                        ax.legend(loc="upper right", fontsize=8)
                    else:
                        ax.text(0.5, 0.5, "No classes above baseline threshold", ha="center", va="center", transform=ax.transAxes)

                for ax in axes_flat[len(mask_order):]:
                    ax.set_axis_off()

                for row_idx in range(nrows):
                    axes[row_idx, 0].set_ylabel("Retention vs baseline (Q > first threshold)")
                for col_idx in range(ncols):
                    axes[-1, col_idx].set_xlabel("Charge threshold")

                fig.suptitle(
                    "Full topology threshold study by active-plane combination\n"
                    "(faster zigzag/rough decay may indicate noise-like behavior)",
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                if save_plots:
                    final_filename = f"{fig_idx}_full_topology_threshold_retention.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig)

            if task3_plot_enabled("full_topology_exact_retention") and mask_order:
                ncols = 2
                nrows = int(math.ceil(len(mask_order) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 4.4 * nrows), squeeze=False, sharex=True)
                axes_flat = axes.flatten()
                cmap = plt.get_cmap("tab20")

                for ax, mask in zip(axes_flat, mask_order):
                    baseline_by_pattern = {
                        str(pattern): int(count)
                        for pattern, count in baseline_counts.items()
                        if classify_full_strip_topology(str(pattern))[0] == mask
                    }
                    top_patterns = [
                        pattern
                        for pattern, count in sorted(
                            baseline_by_pattern.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                        if count >= full_topology_min_baseline_count
                    ][:full_topology_max_patterns_per_mask]

                    if not top_patterns:
                        ax.text(0.5, 0.5, "No topologies above baseline threshold", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"Active mask {mask}")
                        ax.grid(True, alpha=0.2)
                        continue

                    for idx, pattern in enumerate(top_patterns):
                        baseline_n = float(baseline_by_pattern[pattern])
                        retention_vals: list[float] = []
                        for threshold_label in threshold_labels:
                            counts = threshold_counts_by_label[threshold_label]
                            surviving = float(int(counts.get(pattern, 0)))
                            retention_vals.append(surviving / baseline_n)

                        _, topo_class, path_label = classify_full_strip_topology(pattern)
                        class_short = {
                            "single_strip_smooth": "smooth",
                            "single_strip_rough": "rough",
                            "single_strip_zigzag": "zigzag",
                            "multi_strip": "multi",
                        }.get(topo_class, topo_class)
                        suffix = f"{class_short}:{path_label}" if path_label else class_short
                        label = f"{pattern[:8]}.. {suffix} (N0={int(baseline_n)})"
                        ax.plot(
                            list_tt_charge_thresholds,
                            retention_vals,
                            marker="o",
                            linewidth=1.4,
                            markersize=3.5,
                            color=cmap(idx % 20),
                            label=label,
                            alpha=0.95,
                        )

                    ax.set_title(f"Active mask {mask}: top exact topologies")
                    ax.set_ylim(0.0, 1.05)
                    ax.grid(True, alpha=0.25)
                    ax.legend(loc="upper right", fontsize=7)

                for ax in axes_flat[len(mask_order):]:
                    ax.set_axis_off()

                for row_idx in range(nrows):
                    axes[row_idx, 0].set_ylabel("Retention vs baseline (Q > first threshold)")
                for col_idx in range(ncols):
                    axes[-1, col_idx].set_xlabel("Charge threshold")

                fig.suptitle(
                    "Exact topology decay by active-plane combination\n"
                    "(which specific patterns disappear faster?)",
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                if save_plots:
                    final_filename = f"{fig_idx}_full_topology_exact_retention.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig)

            if task3_plot_enabled("full_topology_class_fraction") and mask_order:
                ncols = 2
                nrows = int(math.ceil(len(mask_order) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.1 * nrows), squeeze=False, sharex=True)
                axes_flat = axes.flatten()
                class_order = [
                    "single_strip_smooth",
                    "single_strip_rough",
                    "single_strip_zigzag",
                    "multi_strip",
                ]
                class_labels = {
                    "single_strip_smooth": "single-strip smooth",
                    "single_strip_rough": "single-strip rough",
                    "single_strip_zigzag": "single-strip zigzag",
                    "multi_strip": "multi-strip",
                }
                class_colors = {
                    "single_strip_smooth": "tab:blue",
                    "single_strip_rough": "tab:orange",
                    "single_strip_zigzag": "tab:red",
                    "multi_strip": "tab:gray",
                }

                for ax, mask in zip(axes_flat, mask_order):
                    class_fraction_values: dict[str, list[float]] = {key: [] for key in class_order}
                    for threshold_label in threshold_labels:
                        counts = threshold_counts_by_label[threshold_label]
                        class_totals = {key: 0 for key in class_order}
                        mask_total = 0
                        for pattern, count in counts.items():
                            active_mask, topo_class, _ = classify_full_strip_topology(str(pattern))
                            if active_mask != mask:
                                continue
                            if topo_class not in class_totals:
                                continue
                            class_totals[topo_class] += int(count)
                            mask_total += int(count)

                        if mask_total <= 0:
                            for topo_class in class_order:
                                class_fraction_values[topo_class].append(np.nan)
                        else:
                            for topo_class in class_order:
                                class_fraction_values[topo_class].append(
                                    float(class_totals[topo_class]) / float(mask_total)
                                )

                    for topo_class in class_order:
                        y_vals = np.asarray(class_fraction_values[topo_class], dtype=float)
                        valid = np.isfinite(y_vals)
                        if not np.any(valid):
                            continue
                        ax.plot(
                            np.asarray(list_tt_charge_thresholds, dtype=float)[valid],
                            y_vals[valid],
                            marker="o",
                            linewidth=1.9,
                            markersize=4,
                            color=class_colors[topo_class],
                            label=class_labels[topo_class],
                        )

                    ax.set_title(f"Active mask {mask}: class composition")
                    ax.set_ylim(0.0, 1.05)
                    ax.grid(True, alpha=0.25)
                    ax.legend(loc="upper right", fontsize=8)

                for ax in axes_flat[len(mask_order):]:
                    ax.set_axis_off()

                for row_idx in range(nrows):
                    axes[row_idx, 0].set_ylabel("Class fraction within mask")
                for col_idx in range(ncols):
                    axes[-1, col_idx].set_xlabel("Charge threshold")

                fig.suptitle(
                    "Topology class composition vs charge threshold\n"
                    "(within each active-plane combination)",
                    fontsize=14,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                if save_plots:
                    final_filename = f"{fig_idx}_full_topology_class_fraction.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig)

        # -----------------------------------------------------------------
        # Noise-discovery diagnostics block
        # -----------------------------------------------------------------

        # --- 1. Charge asymmetry (Q_diff / Q_sum) vs threshold ---
        if task3_plot_enabled("charge_asymmetry_vs_threshold"):
            q_sum_cols_asym = [f"P{p}_Q_sum_final" for p in range(1, 5) if f"P{p}_Q_sum_final" in working_df.columns]
            q_dif_cols_asym = [f"P{p}_Q_dif_final" for p in range(1, 5) if f"P{p}_Q_dif_final" in working_df.columns]
            paired_planes = [
                (int(qs.split("_")[0].replace("P", "")), qs, qd)
                for qs, qd in zip(q_sum_cols_asym, q_dif_cols_asym)
            ]
            if paired_planes:
                scan_thrs = sorted(list_tt_charge_thresholds)[:plane_charge_fraction_max_panels]

                # Shared x-range for all asymmetry panels (auto, with optional manual override).
                _asym_x_abs_raw = config.get("charge_asymmetry_x_abs_max", None)
                asym_x_abs_max = None
                try:
                    if _asym_x_abs_raw not in (None, ""):
                        asym_x_abs_max = abs(float(_asym_x_abs_raw))
                except (TypeError, ValueError):
                    asym_x_abs_max = None
                if asym_x_abs_max is not None and (not np.isfinite(asym_x_abs_max) or asym_x_abs_max <= 0):
                    asym_x_abs_max = None

                if asym_x_abs_max is None:
                    _asym_q_low_raw = config.get("charge_asymmetry_x_low_quantile", 0.5)
                    _asym_q_high_raw = config.get("charge_asymmetry_x_high_quantile", 99.5)
                    try:
                        asym_q_low = float(_asym_q_low_raw)
                    except (TypeError, ValueError):
                        asym_q_low = 0.5
                    try:
                        asym_q_high = float(_asym_q_high_raw)
                    except (TypeError, ValueError):
                        asym_q_high = 99.5
                    asym_q_low = float(np.clip(asym_q_low, 0.0, 49.9))
                    asym_q_high = float(np.clip(asym_q_high, 50.1, 100.0))
                    if asym_q_low >= asym_q_high:
                        asym_q_low, asym_q_high = 0.5, 99.5

                    asym_pool: list[np.ndarray] = []
                    for _, qs_col, qd_col in paired_planes:
                        qs_arr = pd.to_numeric(working_df[qs_col], errors="coerce").to_numpy(dtype=float)
                        qd_arr = pd.to_numeric(working_df[qd_col], errors="coerce").to_numpy(dtype=float)
                        for thr in scan_thrs:
                            keep = np.isfinite(qs_arr) & np.isfinite(qd_arr) & (qs_arr > float(thr))
                            if not np.any(keep):
                                continue
                            asym_vals = qd_arr[keep] / qs_arr[keep]
                            finite_vals = asym_vals[np.isfinite(asym_vals)]
                            if finite_vals.size:
                                asym_pool.append(finite_vals)

                    if asym_pool:
                        asym_all = np.concatenate(asym_pool)
                        q_lo, q_hi = np.nanpercentile(asym_all, [asym_q_low, asym_q_high])
                        ref_abs = max(abs(float(q_lo)), abs(float(q_hi)))
                        if not np.isfinite(ref_abs) or ref_abs <= 0:
                            ref_abs = 1.0
                        asym_x_abs_max = max(0.05, 1.10 * ref_abs)
                    else:
                        asym_x_abs_max = 1.0

                asym_x_left = -float(asym_x_abs_max)
                asym_x_right = float(asym_x_abs_max)

                nrows_a = len(paired_planes)
                ncols_a = len(scan_thrs)
                fig_a, axes_a = plt.subplots(
                    nrows_a, ncols_a,
                    figsize=(3.1 * max(1, ncols_a), 2.5 * max(1, nrows_a)),
                    squeeze=False, sharex=True, sharey=True,
                )
                plane_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
                for i_row, (plane, qs_col, qd_col) in enumerate(paired_planes):
                    qs_arr = pd.to_numeric(working_df[qs_col], errors="coerce").to_numpy(dtype=float)
                    qd_arr = pd.to_numeric(working_df[qd_col], errors="coerce").to_numpy(dtype=float)
                    for j_col, thr in enumerate(scan_thrs):
                        ax = axes_a[i_row, j_col]
                        keep = np.isfinite(qs_arr) & np.isfinite(qd_arr) & (qs_arr > float(thr))
                        if not np.any(keep):
                            ax.text(0.5, 0.5, "No events", ha="center", va="center",
                                    transform=ax.transAxes, fontsize=8)
                            ax.grid(True, alpha=0.2)
                            continue
                        asym = qd_arr[keep] / qs_arr[keep]
                        asym = asym[np.isfinite(asym)]
                        if asym.size == 0:
                            ax.text(0.5, 0.5, "No finite asymmetry", ha="center", va="center",
                                    transform=ax.transAxes, fontsize=8)
                            ax.grid(True, alpha=0.2)
                            continue
                        ax.hist(asym, bins=60, range=(asym_x_left, asym_x_right),
                                color=plane_colors.get(plane, "grey"), alpha=0.7,
                                edgecolor="none", density=True)
                        ax.axvline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
                        ax.grid(True, alpha=0.2)
                        if i_row == 0:
                            ax.set_title(f"Q > {thr:g}\nN={int(np.sum(keep))}", fontsize=9)
                        if j_col == 0:
                            ax.set_ylabel(f"Plane {plane}\nPDF", fontsize=9)
                axes_a[0, 0].set_xlim(asym_x_left, asym_x_right)
                for j_col in range(ncols_a):
                    axes_a[-1, j_col].set_xlabel("Q_diff / Q_sum")
                fig_a.suptitle(
                    "Charge asymmetry (Q_diff/Q_sum) per plane vs threshold\n"
                    "Noise: flat / bimodal at |A|→1; signal: peaked near 0",
                    fontsize=13,
                )
                fig_a.tight_layout(rect=[0, 0, 1, 0.93])
                if save_plots:
                    final_filename = f"{fig_idx}_charge_asymmetry_vs_threshold.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig_a, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig_a)

        # --- 2. Inter-plane T_sum correlation ---
        if task3_plot_enabled("interplane_timing_correlation"):
            t_sum_corr_cols = {
                p: f"P{p}_T_sum_final"
                for p in range(1, 5)
                if f"P{p}_T_sum_final" in working_df.columns
            }
            plane_pairs = list(combinations(sorted(t_sum_corr_cols.keys()), 2))
            # Use a sparse selection of thresholds to keep the figure readable.
            corr_thrs = sorted(list_tt_charge_thresholds)[::max(1, len(list_tt_charge_thresholds) // 4)][:4]
            if plane_pairs and corr_thrs:
                nrows_t = len(plane_pairs)
                ncols_t = len(corr_thrs)
                fig_t, axes_t = plt.subplots(
                    nrows_t, ncols_t,
                    figsize=(3.5 * max(1, ncols_t), 3.0 * max(1, nrows_t)),
                    squeeze=False,
                )
                max_scatter_pts = int(config.get("interplane_timing_scatter_max_points", 80000))
                max_scatter_pts = max(2000, max_scatter_pts)
                for i_row, (pA, pB) in enumerate(plane_pairs):
                    tA = pd.to_numeric(working_df[t_sum_corr_cols[pA]], errors="coerce").to_numpy(dtype=float)
                    tB = pd.to_numeric(working_df[t_sum_corr_cols[pB]], errors="coerce").to_numpy(dtype=float)
                    qA_col = f"P{pA}_Q_sum_final"
                    qB_col = f"P{pB}_Q_sum_final"
                    qA = pd.to_numeric(working_df.get(qA_col, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
                    qB = pd.to_numeric(working_df.get(qB_col, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
                    for j_col, thr in enumerate(corr_thrs):
                        ax = axes_t[i_row, j_col]
                        keep = (
                            np.isfinite(tA) & np.isfinite(tB) &
                            (tA != 0) & (tB != 0) &
                            np.isfinite(qA) & np.isfinite(qB) &
                            (qA > float(thr)) & (qB > float(thr))
                        )
                        n_keep = int(np.sum(keep))
                        if n_keep < 5:
                            ax.text(0.5, 0.5, "Too few", ha="center", va="center",
                                    transform=ax.transAxes, fontsize=8)
                            ax.grid(True, alpha=0.2)
                            continue
                        xv = tA[keep]
                        yv = tB[keep]
                        if n_keep > max_scatter_pts:
                            sel = np.linspace(0, n_keep - 1, max_scatter_pts, dtype=int)
                            xv, yv = xv[sel], yv[sel]
                        ax.scatter(xv, yv, s=2, alpha=0.15, color="steelblue",
                                   linewidths=0, rasterized=True)
                        # Reference diagonal
                        lo = min(float(np.nanmin(xv)), float(np.nanmin(yv)))
                        hi = max(float(np.nanmax(xv)), float(np.nanmax(yv)))
                        ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.7, alpha=0.5)
                        ax.grid(True, alpha=0.2)
                        if i_row == 0:
                            ax.set_title(f"Q > {thr:g}\nN={n_keep}", fontsize=9)
                        if j_col == 0:
                            ax.set_ylabel(f"P{pA}-P{pB}\nT_sum P{pB}", fontsize=9)
                for j_col in range(ncols_t):
                    axes_t[-1, j_col].set_xlabel("T_sum (earlier plane)")
                fig_t.suptitle(
                    "Inter-plane T_sum correlation vs threshold\n"
                    "Noise: diffuse cloud; signal: tight diagonal band",
                    fontsize=13,
                )
                fig_t.tight_layout(rect=[0, 0, 1, 0.93])
                if save_plots:
                    final_filename = f"{fig_idx}_interplane_timing_correlation.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig_t, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig_t)

        # --- 3. Multiplicity–charge landscape ---
        if task3_plot_enabled("multiplicity_charge_landscape"):
            q_land_cols = [f"P{p}_Q_sum_final" for p in range(1, 5) if f"P{p}_Q_sum_final" in working_df.columns]
            if len(q_land_cols) >= 2:
                q_land = working_df[q_land_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                scan_thrs_m = sorted(list_tt_charge_thresholds)[:plane_charge_fraction_max_panels]
                ncols_m = len(scan_thrs_m)
                fig_m, axes_m = plt.subplots(
                    1, ncols_m,
                    figsize=(3.5 * max(1, ncols_m), 4.0),
                    squeeze=False, sharey=True,
                )
                q_arr = q_land.to_numpy(dtype=float)
                hb = None
                for j_col, thr in enumerate(scan_thrs_m):
                    ax = axes_m[0, j_col]
                    active = q_arr > float(thr)
                    n_active = active.sum(axis=1)
                    # Minimum charge across active planes (nan where none active)
                    masked = np.where(active, q_arr, np.nan)
                    with np.errstate(all="ignore"):
                        min_q = np.nanmin(masked, axis=1)
                    valid = (n_active >= 1) & np.isfinite(min_q)
                    if not np.any(valid):
                        ax.text(0.5, 0.5, "No events", ha="center", va="center",
                                transform=ax.transAxes, fontsize=8)
                        continue
                    # 2D histogram
                    mult_vals = n_active[valid].astype(float)
                    min_q_vals = min_q[valid]
                    q99 = float(np.nanpercentile(min_q_vals, 99))
                    if not np.isfinite(q99) or q99 <= 0:
                        q99 = float(thr * 5 + 50)
                    hb = ax.hexbin(
                        mult_vals, min_q_vals,
                        gridsize=(4, 30), cmap="inferno",
                        mincnt=1, extent=(0.5, 4.5, 0, q99),
                    )
                    ax.set_xticks([1, 2, 3, 4])
                    ax.set_xlabel("Active planes")
                    ax.set_title(f"Q > {thr:g}\nN={int(np.sum(valid))}", fontsize=9)
                    ax.grid(True, alpha=0.2)
                    if j_col == 0:
                        ax.set_ylabel("Min Q_sum across\nactive planes")
                fig_m.suptitle(
                    "Multiplicity–charge landscape vs threshold\n"
                    "Noise: piles up at low multiplicity + low charge",
                    fontsize=13,
                )
                fig_m.tight_layout(rect=[0, 0, 1, 0.92])
                if ncols_m > 0 and hb is not None:
                    fig_m.colorbar(
                        hb, ax=axes_m[0, -1], pad=0.02, label="Counts",
                        shrink=0.85,
                    )
                if save_plots:
                    final_filename = f"{fig_idx}_multiplicity_charge_landscape.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig_m, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig_m)

        # --- Charge by strip multiplicity (adj / dis events) ---
        if task3_any_plot_enabled(
            "charge_by_strip_multiplicity_adj",
            "charge_by_strip_multiplicity_dis",
        ):
            _cbsm_have_adj_dis = "adj_dis" in working_df.columns
            _cbsm_mult_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
            _cbsm_mult_labels = {1: "single", 2: "double", 3: "triple", 4: "quad"}

            for _cbsm_label, _cbsm_alias in [
                ("adj", "charge_by_strip_multiplicity_adj"),
                ("dis", "charge_by_strip_multiplicity_dis"),
            ]:
                if not task3_plot_enabled(_cbsm_alias):
                    continue
                if not _cbsm_have_adj_dis:
                    continue

                _cbsm_df = working_df.loc[working_df["adj_dis"] == _cbsm_label]
                if _cbsm_df.empty:
                    continue

                _cbsm_tt = pd.to_numeric(_cbsm_df["list_tt"], errors="coerce").fillna(0).astype(int)
                _cbsm_present = set(_cbsm_tt.unique().tolist())
                _cbsm_ordered_tt = [tt for tt in ordered_tt_values if tt in _cbsm_present]
                _n_cbsm_cols = len(_cbsm_ordered_tt)
                if _n_cbsm_cols == 0:
                    continue

                # Common x range from 99th percentile across all planes
                _cbsm_all_charges: list[np.ndarray] = []
                for p in range(1, 5):
                    _cbsm_qcol = f"P{p}_Q_sum_final"
                    if _cbsm_qcol in _cbsm_df.columns:
                        _cbsm_qv = pd.to_numeric(_cbsm_df[_cbsm_qcol], errors="coerce").to_numpy(float)
                        _cbsm_pos = _cbsm_qv[np.isfinite(_cbsm_qv) & (_cbsm_qv > 0)]
                        if _cbsm_pos.size > 0:
                            _cbsm_all_charges.append(_cbsm_pos)
                if _cbsm_all_charges:
                    _cbsm_qmax = float(np.nanpercentile(np.concatenate(_cbsm_all_charges), 99.0))
                else:
                    _cbsm_qmax = 100.0
                _cbsm_qmin = max(
                    float(np.nanmin(np.concatenate(_cbsm_all_charges))) if _cbsm_all_charges else 0.1,
                    0.1,
                )
                _cbsm_bins = np.logspace(np.log10(_cbsm_qmin), np.log10(max(_cbsm_qmax, _cbsm_qmin * 10)), 60)

                fig_cbsm, axes_cbsm = plt.subplots(
                    4, _n_cbsm_cols,
                    figsize=(max(2.5 * _n_cbsm_cols, 8), 12),
                    sharex=True, sharey=True,
                    squeeze=False,
                )

                for j_col, tt_val in enumerate(_cbsm_ordered_tt):
                    _cbsm_mask_tt = _cbsm_tt == tt_val
                    _cbsm_sub = _cbsm_df.loc[_cbsm_mask_tt]
                    axes_cbsm[0, j_col].set_title(
                        f"tt={tt_val}\nN={int(_cbsm_mask_tt.sum())}",
                        fontsize=8,
                    )
                    for i_row, p in enumerate(range(1, 5)):
                        ax = axes_cbsm[i_row, j_col]
                        _cbsm_qcol = f"P{p}_Q_sum_final"
                        _cbsm_scol = f"active_strips_P{p}"
                        if _cbsm_qcol not in _cbsm_sub.columns or _cbsm_scol not in _cbsm_sub.columns:
                            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                                    transform=ax.transAxes, fontsize=9, color="gray")
                            ax.set_axis_off()
                            continue
                        _cbsm_q = pd.to_numeric(_cbsm_sub[_cbsm_qcol], errors="coerce").to_numpy(float)
                        _cbsm_s = _cbsm_sub[_cbsm_scol].astype(str).str.count("1").to_numpy(int)
                        _cbsm_valid = np.isfinite(_cbsm_q) & (_cbsm_q > 0)
                        for mult in [1, 2, 3, 4]:
                            _cbsm_m = _cbsm_valid & (_cbsm_s == mult)
                            if _cbsm_m.sum() == 0:
                                continue
                            ax.hist(
                                _cbsm_q[_cbsm_m],
                                bins=_cbsm_bins,
                                histtype="step",
                                color=_cbsm_mult_colors[mult],
                                label=_cbsm_mult_labels[mult],
                                linewidth=1.2,
                            )
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        if j_col == 0:
                            ax.set_ylabel(f"Plane {p}", fontsize=9)
                        if i_row == 3:
                            ax.set_xlabel("Q sum final [a.u.]", fontsize=8)
                        ax.tick_params(labelsize=7)

                _cbsm_legend_handles = [
                    plt.Line2D([0], [0], color=_cbsm_mult_colors[m], linewidth=1.5,
                               label=_cbsm_mult_labels[m])
                    for m in [1, 2, 3, 4]
                ]
                fig_cbsm.legend(
                    handles=_cbsm_legend_handles,
                    loc="upper right", fontsize=8,
                    title="Strip mult.", title_fontsize=8,
                    ncol=2,
                )
                _cbsm_title = "Adjacent" if _cbsm_label == "adj" else "Dispersed"
                fig_cbsm.suptitle(
                    f"{_cbsm_title} events – charge by plane and strip multiplicity\n"
                    f"Station {station}  (N={len(_cbsm_df):,})",
                    fontsize=12,
                )
                fig_cbsm.tight_layout(rect=[0, 0, 1, 0.93])
                if save_plots:
                    final_filename = f"{fig_idx}_{_cbsm_alias}.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig_cbsm, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig_cbsm)

        # -----------------------------------------------------------------
        # Streamer investigation block
        # -----------------------------------------------------------------
        streamer_q_cols = [
            f"P{p}_Q_sum_final" for p in range(1, 5)
            if f"P{p}_Q_sum_final" in working_df.columns
        ]
        streamer_thr = detect_streamer_threshold(working_df, streamer_q_cols)

        plane_q_arr: dict[int, np.ndarray] = {}
        plane_is_active: dict[int, np.ndarray] = {}
        plane_is_streamer: dict[int, np.ndarray] = {}
        any_streamer = np.zeros(len(working_df), dtype=bool)
        n_streamer_planes = np.zeros(len(working_df), dtype=int)
        for p in range(1, 5):
            col = f"P{p}_Q_sum_final"
            if col not in working_df.columns:
                continue

            arr = pd.to_numeric(working_df[col], errors="coerce").to_numpy(dtype=float)
            plane_q_arr[p] = arr
            plane_active = np.isfinite(arr) & (arr > 0)
            plane_is_active[p] = plane_active

            if streamer_thr is None:
                continue

            plane_streamer = plane_active & (arr > streamer_thr)
            plane_is_streamer[p] = plane_streamer
            any_streamer |= plane_streamer
            n_streamer_planes += plane_streamer.astype(int)

        streamer_plots_requested = task3_any_plot_enabled(
            "streamer_charge_histograms",
            "streamer_prevalence_by_plane",
            "streamer_multiplicity",
            "streamer_contagion_matrix",
            "streamer_efficiency_comparison",
        )

        plane_is_high_charge: dict[int, np.ndarray] = {}
        streamer_high_charge_limit = None
        if streamer_thr is not None and plane_is_streamer:
            streamer_high_charge_limit = float(streamer_thr) * float(streamer_high_charge_factor)
            for p, arr in plane_q_arr.items():
                plane_is_high_charge[p] = plane_is_active[p] & (arr > streamer_high_charge_limit)

        if streamer_plots_requested:
            if streamer_thr is not None:
                print(f"Auto-detected streamer threshold: {streamer_thr:.1f}")

                # --- Plot 0: Per-plane charge histogram with streamer line ---
                if task3_plot_enabled("streamer_charge_histograms"):
                    plane_colors_h = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
                    fig_sh, axes_sh = plt.subplots(2, 2, figsize=(12, 9))
                    axes_sh = axes_sh.flatten()
                    for idx_p, p in enumerate(sorted(plane_q_arr.keys())):
                        ax = axes_sh[idx_p]
                        vals = plane_q_arr[p]
                        valid = np.isfinite(vals) & (vals > 0)
                        if not np.any(valid):
                            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                                    transform=ax.transAxes)
                            continue
                        v = vals[valid]
                        q_max = float(np.nanpercentile(v, 99.9))
                        ax.hist(
                            v, bins=150, range=(0, q_max),
                            color=plane_colors_h[p], alpha=0.75,
                            edgecolor="none", log=True,
                        )
                        ax.axvline(
                            streamer_thr, color="red", linestyle="--", linewidth=1.8,
                            label=f"Streamer boundary = {streamer_thr:.1f}",
                        )
                        n_str = int(np.sum(valid & (vals > streamer_thr)))
                        n_tot = int(np.sum(valid))
                        ax.set_xlabel("Q_sum_final")
                        ax.set_ylabel("Counts (log)")
                        ax.set_title(
                            f"Plane {p}  (N={n_tot}, streamers={n_str}, "
                            f"{100*n_str/n_tot:.1f}%)"
                        )
                        ax.legend(fontsize=9)
                        ax.grid(True, alpha=0.2, which="both")
                    fig_sh.suptitle(
                        f"Per-plane charge spectrum with auto-detected streamer boundary",
                        fontsize=14,
                    )
                    fig_sh.tight_layout(rect=[0, 0, 1, 0.95])
                    if save_plots:
                        final_filename = f"{fig_idx}_streamer_charge_histograms.png"
                        fig_idx += 1
                        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                        plot_list.append(save_fig_path)
                        save_plot_figure(save_fig_path, fig=fig_sh, format="png", dpi=150)
                    if show_plots:
                        plt.show()
                    plt.close(fig_sh)

                # --- Plot 1: Streamer prevalence by plane ---
                if task3_plot_enabled("streamer_prevalence_by_plane"):
                    plane_colors_s = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
                    fig_sp, ax_sp = plt.subplots(figsize=(8, 5))
                    for p in sorted(plane_is_streamer.keys()):
                        fracs = []
                        for thr in sorted(list_tt_charge_thresholds):
                            active_mask = plane_q_arr[p] > float(thr)
                            n_act = int(np.sum(active_mask))
                            n_str = int(np.sum(active_mask & plane_is_streamer[p]))
                            fracs.append(n_str / n_act if n_act > 0 else np.nan)
                        ax_sp.plot(
                            sorted(list_tt_charge_thresholds), fracs,
                            "o-", color=plane_colors_s[p], label=f"Plane {p}",
                            markersize=4, linewidth=1.5,
                        )
                    ax_sp.set_xlabel("Lower charge threshold")
                    ax_sp.set_ylabel("Fraction of active hits that are streamers")
                    ax_sp.set_title(
                        f"Streamer prevalence by plane (auto threshold = {streamer_thr:.1f})"
                    )
                    ax_sp.legend()
                    ax_sp.grid(True, alpha=0.25)
                    ax_sp.set_ylim(bottom=0)
                    fig_sp.tight_layout()
                    if save_plots:
                        final_filename = f"{fig_idx}_streamer_prevalence_by_plane.png"
                        fig_idx += 1
                        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                        plot_list.append(save_fig_path)
                        save_plot_figure(save_fig_path, fig=fig_sp, format="png", dpi=150)
                    if show_plots:
                        plt.show()
                    plt.close(fig_sp)

                # --- Plot 2: Streamer multiplicity histogram (excluding 0) ---
                if task3_plot_enabled("streamer_multiplicity"):
                    # Only among events with at least one streamer
                    mult_vals = n_streamer_planes[any_streamer]
                    n_with_streamer = int(np.sum(any_streamer))
                    n_total_events = int(len(working_df))

                    fig_sm, ax_sm = plt.subplots(figsize=(7, 5))
                    bins_m = np.arange(0.5, 5.5, 1)
                    counts_m, _, patches_m = ax_sm.hist(
                        mult_vals, bins=bins_m, color="steelblue",
                        edgecolor="white", alpha=0.85,
                    )
                    for i_b, count_val in enumerate(counts_m):
                        if count_val > 0:
                            frac_pct = 100.0 * count_val / n_total_events if n_total_events > 0 else 0.0
                            ax_sm.text(
                                i_b + 1, count_val, f"{frac_pct:.1f}%",
                                ha="center", va="bottom", fontsize=9,
                            )
                    ax_sm.set_xticks([1, 2, 3, 4])
                    ax_sm.set_xlabel("Number of planes with streamer")
                    ax_sm.set_ylabel("Events")
                    ax_sm.set_title(
                        f"Streamer multiplicity among streamer events "
                        f"(threshold = {streamer_thr:.1f}, N_streamer={n_with_streamer}, N_total={n_total_events})"
                    )
                    ax_sm.grid(True, alpha=0.25, axis="y")
                    fig_sm.tight_layout()
                    if save_plots:
                        final_filename = f"{fig_idx}_streamer_multiplicity.png"
                        fig_idx += 1
                        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                        plot_list.append(save_fig_path)
                        save_plot_figure(save_fig_path, fig=fig_sm, format="png", dpi=150)
                    if show_plots:
                        plt.show()
                    plt.close(fig_sm)

                # --- Plot 3: Streamer contagion matrix ---
                if task3_plot_enabled("streamer_contagion_matrix"):
                    planes_list = sorted(plane_is_streamer.keys())
                    n_planes = len(planes_list)
                    cond_prob = np.full((n_planes, n_planes), np.nan)
                    for i, pi in enumerate(planes_list):
                        mask_i = plane_is_streamer[pi]
                        n_i = int(np.sum(mask_i))
                        if n_i == 0:
                            continue
                        for j, pj in enumerate(planes_list):
                            if i == j:
                                cond_prob[i, j] = 1.0
                            else:
                                cond_prob[i, j] = float(np.sum(mask_i & plane_is_streamer[pj])) / n_i

                    fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
                    im_sc = ax_sc.imshow(cond_prob, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
                    ax_sc.set_xticks(range(n_planes))
                    ax_sc.set_xticklabels([f"P{p}" for p in planes_list])
                    ax_sc.set_yticks(range(n_planes))
                    ax_sc.set_yticklabels([f"P{p}" for p in planes_list])
                    ax_sc.set_xlabel("Target plane (streamer?)")
                    ax_sc.set_ylabel("Given streamer in this plane")
                    for i in range(n_planes):
                        for j in range(n_planes):
                            val = cond_prob[i, j]
                            if np.isfinite(val):
                                text_color = "white" if val > 0.6 else "black"
                                ax_sc.text(
                                    j, i, f"{val:.2f}", ha="center", va="center",
                                    fontsize=11, color=text_color,
                                )
                    fig_sc.colorbar(im_sc, ax=ax_sc, label="P(streamer in target | streamer in given)")
                    ax_sc.set_title(
                        f"Streamer contagion matrix (threshold = {streamer_thr:.1f})"
                    )
                    fig_sc.tight_layout()
                    if save_plots:
                        final_filename = f"{fig_idx}_streamer_contagion_matrix.png"
                        fig_idx += 1
                        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                        plot_list.append(save_fig_path)
                        save_plot_figure(save_fig_path, fig=fig_sc, format="png", dpi=150)
                    if show_plots:
                        plt.show()
                    plt.close(fig_sc)

                # --- Plot 4: Efficiency with vs without streamers ---
                if task3_plot_enabled("streamer_efficiency_comparison"):
                    no_streamer_mask = ~any_streamer
                    plane_colors_e = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
                    missing_plane_tt_label = {1: "234", 2: "134", 3: "124", 4: "123"}

                    fig_se, axes_se = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
                    for ax_idx, (label, mask_sel) in enumerate([
                        ("All events", np.ones(len(working_df), dtype=bool)),
                        ("Streamer-free events", no_streamer_mask),
                    ]):
                        ax = axes_se[ax_idx]
                        subset_df = working_df.loc[mask_sel, streamer_q_cols]
                        for plane in (1, 2, 3, 4):
                            thrs_plot = []
                            effs_plot = []
                            errs_plot = []
                            for thr in sorted(list_tt_charge_thresholds):
                                tt_series = compute_qsum_threshold_tt(subset_df, thr)
                                tt_counts = tt_series.value_counts()
                                per_plane = compute_empirical_efficiency_from_tt_counts(tt_counts)
                                eff, err, _, _ = per_plane[plane]
                                thrs_plot.append(float(thr))
                                effs_plot.append(float(eff))
                                errs_plot.append(float(err))
                            x = np.asarray(thrs_plot)
                            y = np.asarray(effs_plot)
                            ye = np.asarray(errs_plot)
                            valid = np.isfinite(y) & np.isfinite(ye)
                            if np.any(valid):
                                ax.errorbar(
                                    x[valid], y[valid], yerr=ye[valid],
                                    fmt="o-", linewidth=1.5, markersize=4,
                                    capsize=3, color=plane_colors_e[plane],
                                    label=f"eff_{plane} = 1-N({missing_plane_tt_label[plane]})/N(1234)",
                                )
                        ax.set_xlabel("Charge threshold")
                        ax.set_ylabel("Empirical efficiency")
                        n_sel = int(mask_sel.sum())
                        ax.set_title(f"{label} (N={n_sel})")
                        ax.grid(True, alpha=0.25)
                        ax.legend(fontsize=8)
                        ax.set_ylim(0, 1.05)

                    fig_se.suptitle(
                        f"Efficiency threshold scan: all vs streamer-free\n"
                        f"(auto streamer threshold = {streamer_thr:.1f})",
                        fontsize=14,
                    )
                    fig_se.tight_layout(rect=[0, 0, 1, 0.93])
                    if save_plots:
                        final_filename = f"{fig_idx}_streamer_efficiency_comparison.png"
                        fig_idx += 1
                        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                        plot_list.append(save_fig_path)
                        save_plot_figure(save_fig_path, fig=fig_se, format="png", dpi=150)
                    if show_plots:
                        plt.show()
                    plt.close(fig_se)

            else:
                print("Streamer auto-detection: no valley found in Q_sum spectrum. Skipping streamer plots.")

        # -----------------------------------------------------------------
        # Efficiency anatomy block — probing the dip mechanism
        # -----------------------------------------------------------------

        # --- Weakest-plane charge distribution for 4-plane events ---
        if task3_plot_enabled("fourplane_weakest_charge"):
            q_anat_cols = [
                f"P{p}_Q_sum_final" for p in range(1, 5)
                if f"P{p}_Q_sum_final" in working_df.columns
            ]
            if len(q_anat_cols) == 4:
                q_anat = working_df[q_anat_cols].apply(pd.to_numeric, errors="coerce")
                q_arr_a = q_anat.to_numpy(dtype=float)
                # 4-plane events: all planes have Q > 0
                is_4plane = np.all((q_arr_a > 0) & np.isfinite(q_arr_a), axis=1)
                if np.any(is_4plane):
                    q_4p = q_arr_a[is_4plane]
                    q_min_per_event = np.min(q_4p, axis=1)
                    q_max_per_event = np.max(q_4p, axis=1)
                    q_ratio = q_min_per_event / np.where(q_max_per_event > 0, q_max_per_event, np.nan)

                    fig_wc, axes_wc = plt.subplots(1, 3, figsize=(16, 5))

                    # Panel 1: min(Q) histogram
                    ax = axes_wc[0]
                    q99 = float(np.nanpercentile(q_min_per_event, 99))
                    ax.hist(q_min_per_event, bins=100, range=(0, max(q99, 1)),
                            color="steelblue", edgecolor="none", alpha=0.8, log=True)
                    ax.set_xlabel("Weakest-plane Q_sum")
                    ax.set_ylabel("4-plane events (log)")
                    ax.set_title(f"Min charge across 4 planes\n(N={int(np.sum(is_4plane))})")
                    ax.grid(True, alpha=0.2, which="both")

                    # Panel 2: max(Q) histogram for context
                    ax = axes_wc[1]
                    q99h = float(np.nanpercentile(q_max_per_event, 99))
                    ax.hist(q_max_per_event, bins=100, range=(0, max(q99h, 1)),
                            color="coral", edgecolor="none", alpha=0.8, log=True)
                    ax.set_xlabel("Strongest-plane Q_sum")
                    ax.set_ylabel("4-plane events (log)")
                    ax.set_title("Max charge across 4 planes")
                    ax.grid(True, alpha=0.2, which="both")

                    # Panel 3: Q_min / Q_max ratio
                    ax = axes_wc[2]
                    valid_ratio = q_ratio[np.isfinite(q_ratio)]
                    if valid_ratio.size > 0:
                        ax.hist(valid_ratio, bins=80, range=(0, 1),
                                color="mediumpurple", edgecolor="none", alpha=0.8, log=True)
                    ax.set_xlabel("Q_min / Q_max")
                    ax.set_ylabel("4-plane events (log)")
                    ax.set_title("Charge balance (1 = uniform, →0 = one weak plane)")
                    ax.grid(True, alpha=0.2, which="both")

                    fig_wc.suptitle(
                        "4-plane events: weakest-plane charge anatomy\n"
                        "Crosstalk/noise → excess at low Q_min and low Q_min/Q_max",
                        fontsize=13,
                    )
                    fig_wc.tight_layout(rect=[0, 0, 1, 0.92])
                    if save_plots:
                        final_filename = f"{fig_idx}_fourplane_weakest_charge.png"
                        fig_idx += 1
                        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                        plot_list.append(save_fig_path)
                        save_plot_figure(save_fig_path, fig=fig_wc, format="png", dpi=150)
                    if show_plots:
                        plt.show()
                    plt.close(fig_wc)

        # --- Which plane is weakest in 4-plane events vs threshold ---
        if task3_plot_enabled("fourplane_weakest_plane_identity"):
            q_anat_cols = [
                f"P{p}_Q_sum_final" for p in range(1, 5)
                if f"P{p}_Q_sum_final" in working_df.columns
            ]
            if len(q_anat_cols) == 4:
                q_anat = working_df[q_anat_cols].apply(pd.to_numeric, errors="coerce")
                q_arr_a = q_anat.to_numpy(dtype=float)
                plane_colors_w = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}

                scan_thrs = sorted(list_tt_charge_thresholds)
                fig_wp, axes_wp = plt.subplots(1, 2, figsize=(14, 5.5))

                # Left: fraction of 4-plane events where each plane is the weakest
                ax_left = axes_wp[0]
                for p_idx in range(4):
                    fracs = []
                    for thr in scan_thrs:
                        above = (q_arr_a > float(thr)) & np.isfinite(q_arr_a)
                        is_4p = np.all(above, axis=1)
                        n_4p = int(np.sum(is_4p))
                        if n_4p == 0:
                            fracs.append(np.nan)
                            continue
                        argmin_plane = np.argmin(q_arr_a[is_4p], axis=1)
                        fracs.append(float(np.sum(argmin_plane == p_idx)) / n_4p)
                    ax_left.plot(
                        scan_thrs, fracs, "o-",
                        color=plane_colors_w[p_idx + 1],
                        label=f"Plane {p_idx + 1}", markersize=4, linewidth=1.5,
                    )
                ax_left.axhline(0.25, color="grey", linestyle=":", linewidth=1, alpha=0.6,
                                label="Uniform (0.25)")
                ax_left.set_xlabel("Lower charge threshold")
                ax_left.set_ylabel("Fraction of 4-plane events\nwhere this plane is weakest")
                ax_left.set_title("Which plane is weakest?")
                ax_left.legend(fontsize=8)
                ax_left.grid(True, alpha=0.25)
                ax_left.set_ylim(0, 0.65)

                # Right: for events that transition 4-plane→3-plane between
                # consecutive thresholds, which plane dropped out?
                ax_right = axes_wp[1]
                for p_idx in range(4):
                    dropout_fracs = []
                    for k in range(len(scan_thrs) - 1):
                        thr_lo = float(scan_thrs[k])
                        thr_hi = float(scan_thrs[k + 1])
                        above_lo = (q_arr_a > thr_lo) & np.isfinite(q_arr_a)
                        above_hi = (q_arr_a > thr_hi) & np.isfinite(q_arr_a)
                        was_4p = np.all(above_lo, axis=1)
                        now_3p = was_4p & (above_hi.sum(axis=1) == 3)
                        n_trans = int(np.sum(now_3p))
                        if n_trans == 0:
                            dropout_fracs.append(np.nan)
                            continue
                        # Which plane dropped: was active at lo, inactive at hi
                        dropped = above_lo[now_3p] & ~above_hi[now_3p]
                        dropout_fracs.append(float(np.sum(dropped[:, p_idx])) / n_trans)
                    # Plot at midpoint of threshold interval
                    midpoints = [0.5 * (scan_thrs[k] + scan_thrs[k + 1]) for k in range(len(scan_thrs) - 1)]
                    ax_right.plot(
                        midpoints, dropout_fracs, "s-",
                        color=plane_colors_w[p_idx + 1],
                        label=f"Plane {p_idx + 1}", markersize=4, linewidth=1.5,
                    )
                ax_right.axhline(0.25, color="grey", linestyle=":", linewidth=1, alpha=0.6,
                                 label="Uniform (0.25)")
                ax_right.set_xlabel("Threshold midpoint")
                ax_right.set_ylabel("Fraction of 4→3 transitions\nwhere this plane dropped")
                ax_right.set_title("Which plane drops out first?")
                ax_right.legend(fontsize=8)
                ax_right.grid(True, alpha=0.25)
                ax_right.set_ylim(0, 0.65)

                fig_wp.suptitle(
                    "4-plane event anatomy: weakest plane identity\n"
                    "Crosstalk target → one plane disproportionately weak / drops first",
                    fontsize=13,
                )
                fig_wp.tight_layout(rect=[0, 0, 1, 0.91])
                if save_plots:
                    final_filename = f"{fig_idx}_fourplane_weakest_plane_identity.png"
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, fig=fig_wp, format="png", dpi=150)
                if show_plots:
                    plt.show()
                plt.close(fig_wp)

        # --- Efficiency anatomy by charge band ---
        if task3_plot_enabled("efficiency_anatomy_by_charge_band"):
            q_anat_cols = [
                f"P{p}_Q_sum_final" for p in range(1, 5)
                if f"P{p}_Q_sum_final" in working_df.columns
            ]
            if len(q_anat_cols) == 4:
                q_anat = working_df[q_anat_cols].apply(pd.to_numeric, errors="coerce")
                q_arr_a = q_anat.to_numpy(dtype=float)
                # Q_min across active planes (planes with Q > 0)
                q_active = np.where((q_arr_a > 0) & np.isfinite(q_arr_a), q_arr_a, np.nan)
                with np.errstate(all="ignore"):
                    q_min_all = np.nanmin(q_active, axis=1)

                # Define charge bands from data quantiles
                valid_qmin = q_min_all[np.isfinite(q_min_all) & (q_min_all > 0)]
                if valid_qmin.size > 100:
                    band_edges = [0.0] + [
                        float(np.nanpercentile(valid_qmin, p))
                        for p in (25, 50, 75)
                    ] + [float(np.nanmax(valid_qmin)) + 1.0]
                    # Remove duplicate edges
                    band_edges = sorted(set(band_edges))
                    if len(band_edges) < 3:
                        band_edges = [0.0, float(np.nanmedian(valid_qmin)), float(np.nanmax(valid_qmin)) + 1.0]

                    n_bands = len(band_edges) - 1
                    plane_colors_b = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
                    fig_ea, axes_ea = plt.subplots(1, n_bands, figsize=(5.5 * n_bands, 5.5),
                                                   sharey=True, squeeze=False)

                    scan_thrs = sorted(list_tt_charge_thresholds)
                    for b_idx in range(n_bands):
                        ax = axes_ea[0, b_idx]
                        lo = band_edges[b_idx]
                        hi = band_edges[b_idx + 1]
                        # Events whose Q_min (at threshold=0) falls in this band
                        in_band = np.isfinite(q_min_all) & (q_min_all > lo) & (q_min_all <= hi)
                        n_band = int(np.sum(in_band))
                        band_df = working_df.loc[in_band, q_anat_cols]

                        for plane in (1, 2, 3, 4):
                            effs = []
                            for thr in scan_thrs:
                                tt_s = compute_qsum_threshold_tt(band_df, thr)
                                tt_c = tt_s.value_counts()
                                per_p = compute_empirical_efficiency_from_tt_counts(tt_c)
                                eff, err, _, _ = per_p[plane]
                                effs.append(float(eff))
                            ax.plot(
                                scan_thrs, effs, "o-",
                                color=plane_colors_b[plane],
                                label=f"Plane {plane}", markersize=3.5, linewidth=1.3,
                            )
                        ax.set_xlabel("Charge threshold")
                        if b_idx == 0:
                            ax.set_ylabel("Empirical efficiency")
                        ax.set_title(
                            f"Q_min band ({lo:.0f}, {hi:.0f}]\nN = {n_band}",
                            fontsize=10,
                        )
                        ax.legend(fontsize=7)
                        ax.grid(True, alpha=0.25)
                        ax.set_ylim(0, 1.05)

                    fig_ea.suptitle(
                        "Efficiency anatomy: threshold scan split by weakest-plane charge band\n"
                        "Low Q_min → noise/crosstalk candidates; high Q_min → clean signal",
                        fontsize=13,
                    )
                    fig_ea.tight_layout(rect=[0, 0, 1, 0.90])
                    if save_plots:
                        final_filename = f"{fig_idx}_efficiency_anatomy_by_charge_band.png"
                        fig_idx += 1
                        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                        plot_list.append(save_fig_path)
                        save_plot_figure(save_fig_path, fig=fig_ea, format="png", dpi=150)
                    if show_plots:
                        plt.show()
                    plt.close(fig_ea)

list_strip_patterns = build_task3_full_strip_pattern_series(working_df)
store_pattern_rates(pattern_metadata, list_strip_patterns, "list_strip_pattern", working_df)

if task3_plot_enabled("strip_activation_matrix_before_after"):
    fig_idx = plot_task3_plane_activation_before_after(
        task3_plane_activation_initial,
        task3_plane_activation_filtered,
        fig_idx,
    )

finalize_saved_plots_to_pdf()

# Final number of events
final_number_of_events = len(working_df)
print(f"Final number of events in the dataframe: {final_number_of_events}")

print(
    f"Writing list parquet: rows={len(working_df)} cols={len(working_df.columns)} -> {OUT_PATH}"
)
if VERBOSE:
    print("Columns before saving calibrated->list parquet:")
    for col in working_df.columns:
        print(f" - {col}")

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
filter_events_per_second_meta = build_events_per_second_metadata(working_df)
try:
    filter_rate_denominator_seconds = int(
        filter_events_per_second_meta.get("events_per_second_total_seconds", 0) or 0
    )
except (TypeError, ValueError):
    filter_rate_denominator_seconds = 0
if filter_rate_denominator_seconds < 0:
    filter_rate_denominator_seconds = 0

filter_row = {
    "filename_base": filename_base,
    "execution_timestamp": execution_timestamp,
    "param_hash": param_hash_value,
    FILTER6_RATE_DENOMINATOR_COLUMN: filter_rate_denominator_seconds,
}
for name in FILTER_METRIC_NAMES:
    filter_row[name] = filter_metrics.get(name, "")
for name in FILTER6_NONZERO_COUNTER_NAMES:
    raw_count = global_variables.get(name, "")
    rate_key = f"{name}_rate_hz"
    try:
        raw_count_value = float(raw_count)
    except (TypeError, ValueError):
        filter_row[rate_key] = ""
        continue
    if not np.isfinite(raw_count_value):
        filter_row[rate_key] = ""
        continue
    if filter_rate_denominator_seconds > 0:
        filter_row[rate_key] = round(raw_count_value / filter_rate_denominator_seconds, 6)
    else:
        filter_row[rate_key] = 0

metadata_filter_csv_path = save_metadata(
    csv_path_filter,
    filter_row,
    preferred_fieldnames=(
        "filename_base",
        "execution_timestamp",
        "param_hash",
        *FILTER_METRIC_NAMES,
        FILTER6_RATE_DENOMINATOR_COLUMN,
        *FILTER6_NONZERO_RATE_NAMES,
    ),
    drop_field_predicate=lambda column_name: column_name in FILTER6_NONZERO_COUNTER_NAMES,
)
print(f"Metadata (filter) CSV updated at: {metadata_filter_csv_path}")

# -------------------------------------------------------------------------------
# Execution metadata ------------------------------------------------------------
# -------------------------------------------------------------------------------

print("----------\nExecution metadata to be saved:")
print(f"Filename base: {filename_base}")
print(f"Execution timestamp: {execution_timestamp}")
print(f"Data purity percentage: {data_purity_percentage:.2f}%")
print(f"Total execution time: {total_execution_time_minutes:.2f} minutes")

metadata_execution_csv_path = save_metadata(
    csv_path,
    {
        "filename_base": filename_base,
        "execution_timestamp": execution_timestamp,
        "param_hash": param_hash_value,
        "data_purity_percentage": round(float(data_purity_percentage), 4),
        "total_execution_time_minutes": round(float(total_execution_time_minutes), 4),
    },
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
ensure_global_count_keys(("cal_tt", "list_tt", "cal_to_list_tt"))
add_normalized_count_metadata(
    global_variables,
    global_variables.get("events_per_second_total_seconds", 0),
)
set_global_rate_from_tt_rates(
    global_variables,
    preferred_prefixes=("list_tt", "cal_tt"),
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
trigger_type_prefixes = ("cal_tt", "list_tt", "cal_to_list_tt")
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
        or column_name.startswith("cal_strip_pattern_")
        or column_name.startswith("list_strip_pattern_")
        or is_trigger_type_metadata_column(column_name, trigger_type_prefixes)
        or column_name in FILTER6_NONZERO_COUNTER_NAMES
    ),
)
print(f"Metadata (specific) CSV updated at: {metadata_specific_csv_path}")

# Ensure no figure handles remain open before persistence/final move.
plt.close("all")

# Save to HDF5 file
working_df.to_parquet(OUT_PATH, engine="pyarrow", compression="zstd", index=False)
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
