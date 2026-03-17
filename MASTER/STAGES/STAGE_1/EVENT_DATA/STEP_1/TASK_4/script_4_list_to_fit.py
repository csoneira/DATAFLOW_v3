#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_4/script_4_list_to_fit.py
Purpose: !/usr/bin/env python3.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_4/script_4_list_to_fit.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

#%%

from __future__ import annotations

"""
Stage 1 Task 4 (LIST-->FIT) preparation workflow.

Ingests the LIST-format events from Task 3, assembles the inputs used by the
timing/charge fitting routines (feature engineering, binning, sanity checks),
executes the intermediate analyses that characterise detector response, and
writes the fit-ready artefacts. Execution metadata, diagnostic plots, and
directory bookkeeping are kept in sync so the pipeline can hand off cleanly to
the correction stage.
"""
# Standard Library
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
from MASTER.common.plot_utils import pdf_save_rasterized_page
from MASTER.common.selection_config import load_selection_for_paths, station_is_selected
from MASTER.common.status_csv import initialize_status_row, update_status_progress
from MASTER.common.reprocessing_utils import get_reprocessing_value
from MASTER.common.simulated_data_utils import resolve_simulated_z_positions
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
    load_step1_task_plot_catalog,
    resolve_step1_plot_options,
    step1_task_plot_enabled,
    validate_step1_input_file_args,
    y_pos,
)

task_number = 4

# I want to chrono the execution time of the script
start_execution_time_counting = datetime.now()
_prof_t0 = time.perf_counter()
_prof = {}

STATION_CHOICES = ("0", "1", "2", "3", "4")
TASK4_PLOT_ALIASES: tuple[str, ...] = (
    "debug_suite",
    "usual_suite",
    "essential_suite",
    "flat_values_histogram",
    "detached_timeseries_combo",
    "detached_residuals_combo",
    "timtrack_timeseries_combo",
    "timtrack_residuals_combo",
    "combined_timeseries_combo",
    "combined_residuals_combo",
    "hist_core_errs_combined",
    "hist_core_errs_combined_with_fits",
    "stat_window_accumulation",
    "trigger_types_raw_and_list",
    "trigger_types_tracking_and_list",
    "trigger_types_tracking_and_raw",
    "trigger_types_definitive_tt_and_raw",
    "timtrack_results_hexbin_combination_projections",
    "theta_det_theta_zoom_tracking_tt",
    "polar_theta_phi_definitive_tt_2d",
    "polar_theta_phi_tracking_tt_2d",
    "events_per_second_by_plane_cardinality_double_row",
    "timtrack_residuals_gaussian",
    "external_residuals_gaussian",
    "all_channels_charge",
    "event_display_sample",
    "track_consistency_loo_residuals",
    "strip_hit_occupancy",
    "track_based_efficiency",
    "track_based_efficiency_vs_theta",
    "chi2_charge_populations",
)
task4_plot_status_by_alias: dict[str, str] = {}


def task4_plot_enabled(alias: str) -> bool:
    if not task4_plot_status_by_alias:
        return True
    return step1_task_plot_enabled(alias, task4_plot_status_by_alias, plot_mode)


TASK4_PLOT_PREFIX_ALIASES: tuple[tuple[str, str], ...] = (
    ("detached_timeseries_combo_", "detached_timeseries_combo"),
    ("detached_residuals_combo_", "detached_residuals_combo"),
    ("timtrack_timeseries_combo_", "timtrack_timeseries_combo"),
    ("timtrack_residuals_combo_", "timtrack_residuals_combo"),
    ("combined_timeseries_combo_", "combined_timeseries_combo"),
    ("combined_residuals_combo_", "combined_residuals_combo"),
    ("stat_window_accumulation_", "stat_window_accumulation"),
    ("timtrack_residuals_with_gaussian_for_processed_type_", "timtrack_residuals_gaussian"),
    ("external_residuals_with_gaussian_for_processed_type_", "external_residuals_gaussian"),
    ("all_channels_charge_mingo0", "all_channels_charge"),
)


def resolve_task4_plot_alias(save_path: str, alias: str | None = None) -> str | None:
    if alias is not None:
        return alias if alias in TASK4_PLOT_ALIASES else None

    stem = os.path.splitext(os.path.basename(str(save_path)))[0].lower()
    stem = re.sub(r"^\d+_", "", stem)
    if stem in TASK4_PLOT_ALIASES:
        return stem
    for prefix, mapped_alias in TASK4_PLOT_PREFIX_ALIASES:
        if stem.startswith(prefix):
            return mapped_alias
    return None


def apply_task4_plot_catalog_modes() -> None:
    global create_plots, create_essential_plots, create_debug_plots, save_plots, create_pdf
    if not task4_plot_status_by_alias:
        return
    create_plots = create_plots and task4_plot_enabled("usual_suite")
    create_essential_plots = create_essential_plots and task4_plot_enabled("essential_suite")
    create_debug_plots = create_debug_plots and task4_plot_enabled("debug_suite")
    save_plots = bool(create_plots or create_essential_plots or create_debug_plots)
    create_pdf = save_plots

CLI_PARSER = build_step1_cli_parser("Run Stage 1 STEP_1 TASK_4 (LIST->FIT).", STATION_CHOICES)
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
    plot_alias = resolve_task4_plot_alias(save_path, alias=alias)
    if plot_alias is not None and not task4_plot_enabled(plot_alias):
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
    / "TASK_4"
    / "config_task_4.yaml"
)
plot_catalog_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_4"
    / "config_plots_task_4.yaml"
)
parameter_config_file_path = (
    config_root
    / "STAGE_1"
    / "EVENT_DATA"
    / "STEP_1"
    / "TASK_4"
    / "config_parameters_task_4.csv"
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
task4_plot_status_by_alias = load_step1_task_plot_catalog(
    plot_catalog_file_path,
    TASK4_PLOT_ALIASES,
    "Task 4",
    log_fn=print,
)
debug_mode = False

home_path = str(resolve_home_path_from_config(config))
REFERENCE_TABLES_DIR = Path(home_path) / "DATAFLOW_v3" / "MASTER" / "CONFIG_FILES" / "METADATA_REPRISE" / "REFERENCE_TABLES"

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
        safe_title = re.sub(r'[\\\\/]+', '_', title).replace(' ', '_')
        final_filename = f'{fig_idx}_{safe_title}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

# Time series + histogram helper ------------------------------------------------
def _safe_hist_params(series, max_bins=50):
    """Return (bins, range) for hist; avoid zero-range/invalid bins."""
    finite_series = series[np.isfinite(series)]
    if finite_series.empty:
        return None, None
    vmin = finite_series.min()
    vmax = finite_series.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if vmin == vmax:
        pad = 0.5 if vmin == 0 else 0.05 * abs(vmin)
        return 1, (vmin - pad, vmax + pad)
    unique_count = int(finite_series.nunique())
    bins = int(min(max_bins, max(1, unique_count)))
    return bins, (vmin, vmax)

def plot_ts_with_side_hist(df, columns, time_col, title, width_ratios=(3, 1)):
    """Plot time series with side histograms for each column."""
    global fig_idx
    if df.empty:
        return
    n_vars = len(columns)
    fig, axes = plt.subplots(
        n_vars, 2, figsize=(14, 2.2 * n_vars),
        gridspec_kw={"width_ratios": width_ratios},
        sharex="col"
    )
    if n_vars == 1:
        axes = np.array([axes])
    for row_idx, col in enumerate(columns):
        ts_ax, hist_ax = axes[row_idx]
        if col not in df.columns:
            ts_ax.set_visible(False)
            hist_ax.set_visible(False)
            continue
        series = df[col].dropna()
        series = series[np.isfinite(series)]
        if series.empty:
            ts_ax.set_visible(False)
            hist_ax.set_visible(False)
            continue
        ts_ax.plot(df[time_col], df[col], ".", ms=1, alpha=0.8)
        ts_ax.set_ylabel(col)
        ts_ax.grid(True, alpha=0.3)
        bins, hist_range = _safe_hist_params(series, max_bins=50)
        if bins is None:
            hist_ax.set_visible(False)
            continue
        hist_ax.hist(
            series,
            bins=bins,
            range=hist_range,
            orientation="horizontal",
            color="C1",
            alpha=0.8,
        )
        # Let each histogram choose its own x-limits so peaks don't compress other panels
        hist_ax.set_autoscale_on(True)
        hist_ax.autoscale_view()
        hist_ax.set_xlabel("count")
        hist_ax.grid(True, alpha=0.2)
    axes[-1, 0].set_xlabel(time_col)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_plots:
        final_filename = f"{fig_idx}_{title.replace(' ', '_')}.png"
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format="png")
    if show_plots:
        plt.show()
    plt.close()

# Reject-only histograms/scatters ------------------------------------------------
def plot_histograms_grid(df, columns, title_prefix, max_bins=60, cols_per_fig=16):
    """Plot histograms for many columns, chunked into multiple figures."""
    global fig_idx
    if df.empty:
        return
    if not columns:
        return
    cols_per_fig = max(1, int(cols_per_fig))
    ncols = min(4, cols_per_fig)
    for chunk_start in range(0, len(columns), cols_per_fig):
        chunk = columns[chunk_start:chunk_start + cols_per_fig]
        if not chunk:
            continue
        nrows = int(math.ceil(len(chunk) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.0 * ncols, 3.0 * nrows),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()
        last_idx = -1
        for idx, col in enumerate(chunk):
            ax = axes[idx]
            last_idx = idx
            if col not in df.columns:
                ax.set_visible(False)
                continue
            series = df[col].dropna()
            if series.empty:
                ax.set_visible(False)
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                ax.set_visible(False)
                continue
            if pd.api.types.is_bool_dtype(series):
                # Avoid matplotlib warning by coercing bools to ints.
                series = series.astype(np.int8)
                bins, hist_range = 2, (-0.5, 1.5)
            else:
                series = series[np.isfinite(series)]
                if series.empty:
                    ax.set_visible(False)
                    continue
                bins, hist_range = _safe_hist_params(series, max_bins=max_bins)
            if bins is None:
                ax.set_visible(False)
                continue
            ax.hist(series, bins=bins, range=hist_range, alpha=0.75, color="C0")
            ax.set_title(col, fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, alpha=0.2)
        for j in range(last_idx + 1, len(axes)):
            axes[j].set_visible(False)
        title = f"{title_prefix} histograms ({chunk_start + 1}-{chunk_start + len(chunk)})"
        plt.suptitle(title, fontsize=12)
        if save_plots:
            safe_title = re.sub(r"[\\\\/]+", "_", title).replace(" ", "_")
            if len(safe_title) > 180:
                safe_title = safe_title[:180]
            final_filename = f"{fig_idx}_{safe_title}.png"
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format="png")
        if show_plots:
            plt.show()
        plt.close()

def plot_scatter_simple(df, x_col, y_col, title, max_points=200000):
    """Plot a simple scatter with optional downsampling."""
    global fig_idx
    if df.empty:
        return
    if x_col not in df.columns or y_col not in df.columns:
        return
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        return
    sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])]
    if sub.empty:
        return
    if len(sub) > max_points:
        sub = sub.sample(n=max_points, random_state=0)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(
        sub[x_col],
        sub[y_col],
        s=2,
        alpha=0.3,
        color="C0",
        edgecolors="none",
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    if save_plots:
        safe_title = re.sub(r"[\\\\/]+", "_", title).replace(" ", "_")
        if len(safe_title) > 180:
            safe_title = safe_title[:180]
        final_filename = f"{fig_idx}_{safe_title}.png"
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format="png")
    if show_plots:
        plt.show()
    plt.close()

def plot_reject_diagnostics(
    df,
    label,
    basename,
    hist_bins=60,
    cols_per_fig=16,
    scatter_max=200000,
):
    """Create histogram grids and key scatter plots for rejected events."""
    if df.empty:
        return
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        plot_histograms_grid(
            df,
            numeric_cols,
            f"{label}_{basename}",
            max_bins=hist_bins,
            cols_per_fig=cols_per_fig,
        )
    if "theta" in df.columns and "phi" in df.columns:
        plot_scatter_simple(
            df,
            "theta",
            "phi",
            f"{label}_theta_phi_{basename}",
            max_points=scatter_max,
        )
    if "x" in df.columns and "y" in df.columns:
        plot_scatter_simple(
            df,
            "x",
            "y",
            f"{label}_x_y_{basename}",
            max_points=scatter_max,
        )
    elif "det_x" in df.columns and "det_y" in df.columns:
        plot_scatter_simple(
            df,
            "det_x",
            "det_y",
            f"{label}_det_x_det_y_{basename}",
            max_points=scatter_max,
        )

#%%

def plot_residuals_ts_hist(df, prefixes, time_col, title):
    """
    Plot residuals (usual and external variants) per plane with side histograms.
    *prefixes* is a list of (label, usual_prefix, ext_prefix) tuples.
    Each residual type gets its own column; rows are planes.
    """
    global fig_idx
    planes = [1, 2, 3, 4]
    n_cols = len(prefixes)
    fig, axes = plt.subplots(
        len(planes), n_cols * 2,
        figsize=(6 * n_cols, 2.0 * len(planes)),
        gridspec_kw={"width_ratios": sum([[3, 1] for _ in range(n_cols)], [])},
        sharex="col"
    )
    axes = np.atleast_2d(axes)
    axes = axes.reshape(len(planes), n_cols * 2)
    for row_idx, plane in enumerate(planes):
        for col_idx, (label, plane_prefix, ext_prefix) in enumerate(prefixes):
            ts_ax = axes[row_idx, col_idx * 2]
            hist_ax = axes[row_idx, col_idx * 2 + 1]
            col_usual = f"{plane_prefix}{plane}"
            col_ext = f"{ext_prefix}{plane}"
            if col_usual not in df.columns or col_ext not in df.columns:
                ts_ax.set_visible(False)
                hist_ax.set_visible(False)
                continue
            sub = df[[time_col, col_usual, col_ext]].dropna()
            # Drop rows where both residuals are zero (no data)
            sub = sub[(sub[col_usual] != 0) | (sub[col_ext] != 0)]
            if sub.empty:
                ts_ax.set_visible(False)
                hist_ax.set_visible(False)
                continue
            # Plot external first, then usual on top (usual narrower)
            ts_ax.plot(sub[time_col], sub[col_ext], ".", ms=1, label=f"{label}_ext", alpha=0.6)
            ts_ax.plot(sub[time_col], sub[col_usual], ".", ms=1, label=f"{label}", alpha=0.9)
            if col_idx == 0:
                ts_ax.set_ylabel(f"P{plane}")
            ts_ax.grid(True, alpha=0.3)
            ts_ax.legend(fontsize="x-small")
            ext_series = sub[col_ext][np.isfinite(sub[col_ext])]
            usual_series = sub[col_usual][np.isfinite(sub[col_usual])]
            ext_bins, ext_range = _safe_hist_params(ext_series, max_bins=50)
            usual_bins, usual_range = _safe_hist_params(usual_series, max_bins=50)
            if ext_bins is not None:
                hist_ax.hist(
                    ext_series,
                    bins=ext_bins,
                    range=ext_range,
                    orientation="horizontal",
                    alpha=0.5,
                    label=f"{label}_ext",
                )
            if usual_bins is not None:
                hist_ax.hist(
                    usual_series,
                    bins=usual_bins,
                    range=usual_range,
                    orientation="horizontal",
                    alpha=0.8,
                    label=f"{label}",
                )
            hist_ax.set_xlabel("count")
            hist_ax.grid(True, alpha=0.2)
    axes[-1, 0].set_xlabel(time_col)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_plots:
        final_filename = f"{fig_idx}_{title.replace(' ', '_')}.png"
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format="png")
    if show_plots:
        plt.show()
    plt.close()

# Dedicated two-column plot for combined errors
def plot_err_only_ts_hist(df, base_cols, time_col, title):
    """Plot error time series and histograms for given `<col>_err` fields."""
    global fig_idx
    if df.empty:
        return
    n_vars = len(base_cols)
    fig, axes = plt.subplots(
        n_vars, 2, figsize=(14, 2.0 * n_vars),
        gridspec_kw={"width_ratios": [3, 1]},
        sharex="col"
    )
    if n_vars == 1:
        axes = np.array([axes])
    for idx, col in enumerate(base_cols):
        ts_ax, hist_ax = axes[idx]
        err_col = f"{col}_err"
        if err_col not in df.columns:
            ts_ax.set_visible(False)
            hist_ax.set_visible(False)
            continue
        # series = df[err_col].dropna().abs()
        series = df[err_col].dropna()
        if series.empty:
            ts_ax.set_visible(False)
            hist_ax.set_visible(False)
            continue
        ts_ax.plot(df[time_col], series, ".", ms=1, alpha=0.8)
        ts_ax.set_ylabel(f"{col}_err")
        ts_ax.grid(True, alpha=0.3)
        hist_ax.hist(series, bins=50, orientation="horizontal", color="C4", alpha=0.7)
        # Let each histogram scale independently so one peak doesn't compress others
        hist_ax.set_autoscale_on(True)
        hist_ax.autoscale_view()
        hist_ax.set_xlabel("count")
        hist_ax.set_xscale("log")
        hist_ax.grid(True, alpha=0.2)
    axes[-1, 0].set_xlabel(time_col)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_plots:
        final_filename = f"{fig_idx}_{title.replace(' ', '_')}.png"
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format="png")
    if show_plots:
        plt.show()
    plt.close()

#%%

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

selection_config = load_selection_for_paths(
    [config_file_path],
    master_config_root=config_root,
)
if not station_is_selected(station, selection_config.stations):
    print(f"Station {station} skipped by selection.stations.")
    sys.exit(0)

# Cron job switch that decides if completed files can be revisited.
complete_reanalysis = config.get("complete_reanalysis", False)

selected_input_file = CLI_ARGS.input_file_flag or CLI_ARGS.input_file
if selected_input_file:
    user_file_path = selected_input_file
    user_file_selection = True
    print("User provided file path:", user_file_path)
else:
    user_file_selection = False

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
else:
    raw_directory = f"STEP_1/TASK_{task_number - 1}/OUTPUT_FILES"
if task_number == 5:
    output_location = os.path.join(base_directory, "STEP_1_TO_2_OUTPUT")
else:
    output_location = os.path.join(raw_to_list_working_directory, "OUTPUT_FILES")
raw_working_directory = os.path.join(base_directory, raw_directory)

# Example TASK_1 output directory under the current station.
raw_working_directory = os.path.join(base_directory, "STEP_1/TASK_3/OUTPUT_FILES")

raw_to_list_working_directory = os.path.join(base_directory, "STEP_1/TASK_4/")

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

# Accessing all the variables from the configuration
crontab_execution = config["crontab_execution"]
(
    plot_mode,
    create_plots,
    create_essential_plots,
    save_plots,
    create_pdf,
    show_plots,
    create_debug_plots,
) = resolve_step1_plot_options(config)
apply_task4_plot_catalog_modes()
save_rejected_rows = config.get("save_rejected_rows", False)
create_reject_plots = config.get("create_reject_plots", False)
if create_reject_plots:
    save_rejected_rows = True
print(
    f"Warning: reject-plots config -> create_reject_plots={create_reject_plots} "
    f"save_rejected_rows={save_rejected_rows} save_plots={save_plots}"
)
reject_plot_hist_bins = int(config.get("reject_plot_hist_bins", 60))
reject_plot_hist_cols_per_fig = int(config.get("reject_plot_hist_cols_per_fig", 16))
reject_plot_scatter_max_points = int(config.get("reject_plot_scatter_max_points", 200000))

if create_plots:
    # When Task 4 plotting is enabled, force full plotting outputs for this task.
    create_essential_plots = True
    save_plots = True
    create_pdf = True

# Create ALL directories if they don't already exist

for directory in base_directories.values():
    # Skip figure directories at startup; create lazily after selecting a file.
    if directory in (base_directories["base_figure_directory"], base_directories["figure_directory"]):
        print("\n\nSKIPPING THE FIGURE DICTIONARY CREATION\n\n")
        continue
    print(f"Created {directory}")
    os.makedirs(directory, exist_ok=True)

debug_plot_directory = os.path.join(
    base_directories["base_plots_directory"],
    "DEBUG_PLOTS",
    f"FIGURES_EXEC_ON_{date_execution}",
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

last_file_test = bool(config.get("last_file_test", False))

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
                    sys.exit("No files to process in COMPLETED after normalization.")
            else:
                sys.exit("No files to process in UNPROCESSED, PROCESSING and decided to not reanalyze COMPLETED.")

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
                sys.exit("No files to process in UNPROCESSED, PROCESSING and decided to not reanalyze COMPLETED.")

        else:
            sys.exit("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")

# This is for all cases
file_path = processing_file_path
if save_plots:
    os.makedirs(base_directories["figure_directory"], exist_ok=True)

the_filename = os.path.basename(file_path)
print(f"File to process: {the_filename}")

basename_no_ext, file_extension = os.path.splitext(the_filename)
# Take basename of IN_PATH without extension and witouth the 'listed_' prefix
basename_no_ext = the_filename.replace("listed_", "").replace(".parquet", "")

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

global_variables = {}
if simulated_param_hash:
    global_variables["param_hash"] = simulated_param_hash

TT_COUNT_VALUES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 12, 13, 14, 23, 24, 34, 123, 124, 134, 234, 1234
)

def ensure_global_count_keys(prefixes: Iterable[str]) -> None:
    for prefix in prefixes:
        for tt_value in TT_COUNT_VALUES:
            global_variables.setdefault(f"{prefix}_{tt_value}_count", 0)

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

# if limit:
#     print(f'Taking the first {limit_number} rows.')

# Read the data file into a DataFrame
KEY = "df"

# Load dataframe
working_df = pd.read_parquet(file_path, engine="pyarrow")
working_df = working_df.rename(columns=lambda col: col.replace("_diff_", "_dif_"))
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
print(f"Listed dataframe reloaded from: {file_path}")
# print("Columns loaded from parquet:")
# for col in working_df.columns:
#     print(f" - {col}")
# Backward compatibility: if old original_tt exists but raw_tt is missing, reuse it.
if "raw_tt" not in working_df.columns and "original_tt" in working_df.columns:
    working_df = working_df.rename(columns={"original_tt": "raw_tt"})
# Backward compatibility: if clean_tt is missing but preprocessed_tt exists, reuse it.
if "clean_tt" not in working_df.columns and "preprocessed_tt" in working_df.columns:
    working_df = working_df.rename(columns={"preprocessed_tt": "clean_tt"})

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
    main_cols.extend(["raw_tt", "clean_tt", "cal_tt", "list_tt"])
    main_cols = [col for col in main_cols if col in working_df.columns]
    if main_cols:
        debug_fig_idx = plot_debug_histograms(
            working_df,
            main_cols,
            thresholds=None,
            title=f"Task 4 incoming parquet: main columns [NON-TUNABLE] (station {station})",
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
            max_cols_per_fig=20,
        )

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

# the analysis mode indicates if it is a regular analysis or a repeated, careful analysis
# 0 -> regular analysis
# 1 -> repeated, careful analysis

working_df = compute_tt(working_df, "list_tt", list_tt_columns)
list_tt_counts_initial = working_df["list_tt"].value_counts()
for tt_value, count in list_tt_counts_initial.items():
    tt_label = normalize_tt_label(tt_value)
    global_variables[f"list_tt_{tt_label}_count"] = int(count)
working_df["processed_tt"] = working_df["list_tt"].astype(int)

# Ensure cal_tt is present for downstream correlations
if "cal_tt" not in working_df.columns:
    working_df["cal_tt"] = working_df["processed_tt"]

# List all names of columns

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

def load_iteration_settings(cfg):
    number_of_det_executions = max(1, int(cfg.get("number_of_det_executions", 1)))
    number_of_tt_executions = max(1, int(cfg.get("number_of_tt_executions", 1)))
    return (
        number_of_det_executions,
        cfg["fixed_speed"],
        cfg["res_ana_removing_planes"],
        number_of_tt_executions,
        cfg.get("complete_reanalysis", False),
        cfg.get("limit_number", None),
    )

(
    number_of_det_executions,
    fixed_speed,
    res_ana_removing_planes,
    number_of_tt_executions,
    complete_reanalysis,
    limit_number,
) = load_iteration_settings(config)
limit = limit_number is not None

fit_method = str(config.get("fit_method", "both")).strip().lower()
if fit_method not in {"detached", "timtrack", "both"}:
    print(f"Warning: Invalid fit_method='{fit_method}'. Falling back to 'both'.")
    fit_method = "both"
run_detached_fit = fit_method in {"detached", "both"}
run_timtrack_fit = fit_method in {"timtrack", "both"}
print(f"Fitting mode selected: fit_method='{fit_method}'")

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering

# Time calibration

# Y position

# RPC variables

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

# Alternative fitter filter
det_pos_filter = config["det_pos_filter"]
det_theta_left_filter = config["det_theta_left_filter"]
det_theta_right_filter = config["det_theta_right_filter"]
det_phi_filter_abs = abs(float(config.get("det_phi_filter_abs", config.get("det_phi_right_filter", 3.141592))))
det_phi_right_filter = det_phi_filter_abs
det_phi_left_filter = -det_phi_filter_abs
det_slowness_filter_left = config["det_slowness_filter_left"]
det_slowness_filter_right = config["det_slowness_filter_right"]

det_res_ystr_filter = config["det_res_ystr_filter"]
det_res_tsum_filter = config["det_res_tsum_filter"]
det_res_tdif_filter = config["det_res_tdif_filter"]
det_ext_res_ystr_filter = config["det_ext_res_ystr_filter"]
det_ext_res_tsum_filter = config["det_ext_res_tsum_filter"]
det_ext_res_tdif_filter = config["det_ext_res_tdif_filter"]

# TimTrack filter
proj_filter = config["proj_filter"]
res_ystr_filter = config["res_ystr_filter"]
res_tsum_filter = config["res_tsum_filter"]
res_tdif_filter = config["res_tdif_filter"]
ext_res_ystr_filter = config["ext_res_ystr_filter"]
ext_res_tsum_filter = config["ext_res_tsum_filter"]
ext_res_tdif_filter = config["ext_res_tdif_filter"]

# Fitting comparison
delta_s_left = config.get("delta_s_left", -0.0003)
delta_s_right = config.get("delta_s_right", 0.0003)

# Calibrations
coincidence_window_cal_ns = config["coincidence_window_cal_ns"]
coincidence_window_cal_number_of_points = config["coincidence_window_cal_number_of_points"]

# Pedestal charge calibration

# Front-back charge

# Variables to modify
beta = config["beta"]
strip_speed_factor_of_c = config["strip_speed_factor_of_c"]

# X
strip_length = config["strip_length"]
narrow_strip = config["narrow_strip"]
wide_strip = config["wide_strip"]

# Timtrack parameters
d0 = config["d0"]
cocut = config["cocut"]
iter_max = config["iter_max"]
anc_sy = config["anc_sy"]
anc_sts = config["anc_sts"]
anc_std = config["anc_std"]
anc_sz = config["anc_sz"]

n_planes_timtrack = config["n_planes_timtrack"]

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

time_window_fitting = config["time_window_fitting"]

charge_plot_limit_left = config["charge_plot_limit_left"]
charge_plot_limit_right = config["charge_plot_limit_right"]
charge_plot_event_limit_right = config.get("charge_plot_event_limit_right", 400)

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

fig_idx = 1
plot_list = []

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

global_variables['analysis_mode'] = 0
global_variables['unc_y'] = anc_sy
global_variables['unc_tsum'] = anc_sts
global_variables['unc_tdif'] = anc_std

TRACK_COMBINATIONS: tuple[str, ...] = (
    "12", "13", "14", "23", "24", "34",
    "123", "124", "134", "234", "1234",
)

RESIDUAL_SERIES: tuple[str, ...] = (
    "res_ystr",
    "res_tsum",
    "res_tdif",
    "ext_res_ystr",
    "ext_res_tsum",
    "ext_res_tdif",
)

FILTER_METRIC_NAMES: tuple[str, ...] = (
    "det_residual_zeroed_event_pct",
    "det_bounds_zeroed_event_pct",
    "residual_zeroed_event_pct",
    "small_values_zeroed_event_pct",
    "small_values_zeroed_value_pct",
    "definitive_small_values_zeroed_event_pct",
    "definitive_small_values_zeroed_value_pct",
    "low_tt_zeroed_event_pct",
    "definitive_rows_removed_pct",
    "definitive_removed_single_zero_rows_pct",
    "definitive_removed_multi_zero_rows_pct",
    "definitive_removed_primary_x_zero_rows_pct",
    "definitive_removed_primary_y_zero_rows_pct",
    "definitive_removed_primary_s_zero_rows_pct",
    "definitive_removed_primary_t0_zero_rows_pct",
    "definitive_removed_primary_theta_zero_rows_pct",
    "definitive_removed_primary_phi_zero_rows_pct",
    "definitive_x_zero_rows_pct",
    "definitive_y_zero_rows_pct",
    "definitive_s_zero_rows_pct",
    "definitive_t0_zero_rows_pct",
    "definitive_theta_zero_rows_pct",
    "definitive_phi_zero_rows_pct",
    "ancillary_charge_filtered_rows_pct",
    "data_purity_percentage",
    "all_components_zero_rows_removed_pct",
    "fit_tt_lt_10_rows_removed_pct",
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

def _fit_gaussian_sigma(series: pd.Series) -> float:
    """Return Gaussian sigma for *series* ignoring zeros/NaNs; NaN if insufficient data."""
    arr = np.asarray(series)
    arr = arr[np.isfinite(arr) & (arr != 0)]
    if arr.size < 10:
        return np.nan
    try:
        _, sigma = norm.fit(arr)
    except Exception:
        return np.nan
    return float(abs(sigma)) if np.isfinite(sigma) else np.nan

filter_metrics: dict[str, float] = {}

def record_filter_metric(name: str, removed: float, total: float) -> None:
    """Record percentage removed for a filter."""
    pct = 0.0 if total == 0 else 100.0 * float(removed) / float(total)
    filter_metrics[name] = round(pct, 4)
    print(f"[filter-metrics] {name}: removed {removed} of {total} ({pct:.2f}%)")

def record_residual_sigmas(df: pd.DataFrame) -> None:
    """Fit Gaussian sigmas for residual columns per track combination and plane."""
    tt_col = "list_tt" if "list_tt" in df.columns else "processed_tt"
    if tt_col not in df.columns:
        return

    processed = df[tt_col].astype(str)
    for combo in TRACK_COMBINATIONS:
        combo_str = str(combo)
        combo_mask = processed == combo_str
        combo_df = df.loc[combo_mask]
        for plane in range(1, 5):
            if str(plane) not in combo_str:
                continue  # skip planes not present in the trigger combination
            for metric in RESIDUAL_SERIES:
                if metric.startswith("ext_") and len(combo_str) < 3:
                    continue  # external residuals need at least 3 planes
                col = f"{metric}_{plane}"
                key = f"{col}_{combo_str}_sigma"
                if col not in df.columns or combo_df.empty:
                    continue
                sigma = _fit_gaussian_sigma(combo_df[col])
                if np.isnan(sigma):
                    continue  # avoid recording meaningless NaNs
                global_variables[key] = sigma

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

self_trigger = bool(config.get("self_trigger", False))

fast_mode = False
debug_mode = False
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
crontab_execution = config["crontab_execution"]

(
    number_of_det_executions,
    fixed_speed,
    res_ana_removing_planes,
    number_of_tt_executions,
    complete_reanalysis,
    limit_number,
) = load_iteration_settings(config)
limit = limit_number is not None

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering

# Time calibration

# Y position

# RPC variables

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

# Alternative fitter filter
det_pos_filter = config["det_pos_filter"]
det_theta_left_filter = config["det_theta_left_filter"]
det_theta_right_filter = config["det_theta_right_filter"]
det_phi_filter_abs = abs(float(config.get("det_phi_filter_abs", config.get("det_phi_right_filter", 3.141592))))
det_phi_right_filter = det_phi_filter_abs
det_phi_left_filter = -det_phi_filter_abs
det_slowness_filter_left = config["det_slowness_filter_left"]
det_slowness_filter_right = config["det_slowness_filter_right"]

det_res_ystr_filter = config["det_res_ystr_filter"]
det_res_tsum_filter = config["det_res_tsum_filter"]
det_res_tdif_filter = config["det_res_tdif_filter"]

# TimTrack filter
proj_filter = config["proj_filter"]
res_ystr_filter = config["res_ystr_filter"]
res_tsum_filter = config["res_tsum_filter"]
res_tdif_filter = config["res_tdif_filter"]
ext_res_ystr_filter = config["ext_res_ystr_filter"]
ext_res_tsum_filter = config["ext_res_tsum_filter"]
ext_res_tdif_filter = config["ext_res_tdif_filter"]

# Fitting comparison
delta_s_left = config.get("delta_s_left", -0.0003)
delta_s_right = config.get("delta_s_right", 0.0003)

# Calibrations
coincidence_window_cal_ns = config["coincidence_window_cal_ns"]
coincidence_window_cal_number_of_points = config["coincidence_window_cal_number_of_points"]

# Pedestal charge calibration

# Front-back charge

# Variables to modify
beta = config["beta"]
strip_speed_factor_of_c = config["strip_speed_factor_of_c"]

# X
strip_length = config["strip_length"]
narrow_strip = config["narrow_strip"]
wide_strip = config["wide_strip"]

# Timtrack parameters
d0 = config["d0"]
cocut = config["cocut"]
iter_max = config["iter_max"]
anc_sy = config["anc_sy"]
anc_sts = config["anc_sts"]
anc_std = config["anc_std"]
anc_sz = config["anc_sz"]

n_planes_timtrack = config["n_planes_timtrack"]

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

time_window_fitting = config["time_window_fitting"]

charge_plot_limit_left = config["charge_plot_limit_left"]
charge_plot_limit_right = config["charge_plot_limit_right"]
charge_plot_event_limit_right = config.get("charge_plot_event_limit_right", 400)

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

fig_idx = 1
plot_list = []

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

# Note that the middle between start and end time could also be taken. This is for calibration storage.
datetime_value = working_df['datetime'].iloc[0]
end_datetime_value = working_df['datetime'].iloc[-1]

if self_trigger:
    print(self_trigger_df)
    datetime_value_st = self_trigger_df['datetime'].iloc[0]
    end_datetime_value_st = self_trigger_df['datetime'].iloc[-1]
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
save_pdf_filename = f"mingo{str(station).zfill(2)}_task4_{save_filename_suffix}.pdf"

save_pdf_path = os.path.join(base_directories["pdf_directory"], save_pdf_filename)

# -------------------------------------------------------------------------------
# ------------ Input file and data managing to select configuration -------------
# -------------------------------------------------------------------------------

is_simulated_file = basename_no_ext.startswith("mi00")
used_input_file = False
z_source = "unset"
z_positions_from_task3 = None
z_columns = [f"z_P{i}" for i in range(1, 5)]
if all(col in working_df.columns for col in z_columns):
    z_frame = working_df[z_columns].apply(pd.to_numeric, errors="coerce")
    valid_z_rows = z_frame.dropna(how="any")
    if not valid_z_rows.empty:
        z_positions_from_task3 = valid_z_rows.iloc[0].to_numpy(dtype=float)
        if (valid_z_rows.nunique(dropna=True) > 1).any():
            print("Warning: Inconsistent z_P* values in TASK_3 parquet; using first valid row.")
    else:
        print("Warning: TASK_3 parquet has z_P* columns but no fully valid row.")

if z_positions_from_task3 is not None and not np.all(z_positions_from_task3 == 0):
    z_positions = z_positions_from_task3
    found_matching_conf = True
    z_source = "task3_parquet_z_columns"
    print("Using z_positions from TASK_3 parquet columns (z_P1..z_P4).")
elif simulated_z_positions is not None:
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

# Save the z_positions in the metadata file
global_variables['z_P1'] =  z_positions[0]
global_variables['z_P2'] =  z_positions[1]
global_variables['z_P3'] =  z_positions[2]
global_variables['z_P4'] =  z_positions[3]

fast_mode = False
debug_mode = False
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
crontab_execution = config["crontab_execution"]

(
    number_of_det_executions,
    fixed_speed,
    res_ana_removing_planes,
    number_of_tt_executions,
    complete_reanalysis,
    limit_number,
) = load_iteration_settings(config)
limit = limit_number is not None

# Charge calibration to fC

# Charge front-back

# Slewing correction

# Time filtering

# Time calibration

# Y position

# RPC variables

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

# Alternative fitter filter
det_pos_filter = config["det_pos_filter"]
det_theta_left_filter = config["det_theta_left_filter"]
det_theta_right_filter = config["det_theta_right_filter"]
det_phi_filter_abs = abs(float(config.get("det_phi_filter_abs", config.get("det_phi_right_filter", 3.141592))))
det_phi_right_filter = det_phi_filter_abs
det_phi_left_filter = -det_phi_filter_abs
det_slowness_filter_left = config["det_slowness_filter_left"]
det_slowness_filter_right = config["det_slowness_filter_right"]

det_res_ystr_filter = config["det_res_ystr_filter"]
det_res_tsum_filter = config["det_res_tsum_filter"]
det_res_tdif_filter = config["det_res_tdif_filter"]

# TimTrack filter
proj_filter = config["proj_filter"]
res_ystr_filter = config["res_ystr_filter"]
res_tsum_filter = config["res_tsum_filter"]
res_tdif_filter = config["res_tdif_filter"]
ext_res_ystr_filter = config["ext_res_ystr_filter"]
ext_res_tsum_filter = config["ext_res_tsum_filter"]
ext_res_tdif_filter = config["ext_res_tdif_filter"]

# Fitting comparison
delta_s_left = config.get("delta_s_left", -0.0003)
delta_s_right = config.get("delta_s_right", 0.0003)

# Calibrations
coincidence_window_cal_ns = config["coincidence_window_cal_ns"]
coincidence_window_cal_number_of_points = config["coincidence_window_cal_number_of_points"]

# Pedestal charge calibration

# Front-back charge

# Variables to modify
beta = config["beta"]
strip_speed_factor_of_c = config["strip_speed_factor_of_c"]

# X
strip_length = config["strip_length"]
narrow_strip = config["narrow_strip"]
wide_strip = config["wide_strip"]

# Timtrack parameters
d0 = config["d0"]
cocut = config["cocut"]
iter_max = config["iter_max"]
anc_sy = config["anc_sy"]
anc_sts = config["anc_sts"]
anc_std = config["anc_std"]
anc_sz = config["anc_sz"]

n_planes_timtrack = config["n_planes_timtrack"]

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

time_window_fitting = config["time_window_fitting"]

charge_plot_limit_left = config["charge_plot_limit_left"]
charge_plot_limit_right = config["charge_plot_limit_right"]
charge_plot_event_limit_right = config.get("charge_plot_event_limit_right", 400)

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

fig_idx = 1
plot_list = []

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

raw_data_len = len(working_df)
if raw_data_len == 0 and not self_trigger:
    print("No coincidence nor self-trigger events.")
    sys.exit(1)

#%%

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -------- TASK_4: fitting ----------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# # Print the name of the columns
# print("Columns in working_df:", working_df.columns.tolist())

# # Create a 4x4 plot histogram of: Q{plane}_Q_sum_{strip}_with_crstlk between 0 and 80, per each combination in cal_tt

# log_scale = False

# if create_plots:
#     # Detached method plots (per combination)
#     if "list_tt" in working_df.columns and "datetime" in working_df.columns:
#         for combo in TRACK_COMBINATIONS:

#             try:
#                 combo_int = int(combo)
#             except ValueError:
#                 continue
#             subset = working_df[working_df["list_tt"] == combo_int]
#             if subset.empty:
#                 continue

#             fig, axes = plt.subplots(4, 4, figsize=(16, 16))
#             planes = [1, 2, 3, 4]
#             for i, plane_i in enumerate(planes):
#                 for j, plane_j in enumerate(planes):
#                     ax = axes[i, j]
#                     col_name = f'Q{plane_i}_Q_sum_{plane_j}_with_crstlk'
#                     if col_name in working_df.columns:
#                         q_sum_data = subset[col_name].dropna()

#                         # Remove 0s
#                         q_sum_data = q_sum_data[q_sum_data != 0]

#                         if not q_sum_data.empty:
#                             ax.hist(q_sum_data, bins=50, range=(0, 80), color='blue', alpha=0.5)
#                             ax.set_title(f'Plane {plane_i} strip {plane_j} Q_sum')
#                             ax.set_xlabel('Q_sum_final (fC)')
#                             ax.set_ylabel('Counts')

#                             # Log scale if needed
#                             if log_scale:
#                                 ax.set_yscale('log')
#                         else:
#                             ax.set_axis_off()
#                     else:
#                         ax.set_axis_off()
#             plt.suptitle(f'Charge Distributions for Combination {combo_int}')
#             plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#             plt.show()

# #%%

# # Detached processed_tt label for plotting (fallback only if not set by alt loop)
# if "det_processed_tt" not in working_df.columns:
#     working_df["det_processed_tt"] = working_df.get("processed_tt", 0)

# # Select events to plot based on the charge of the planes

# if create_plots:
#     # Detached method plots (per combination)
#     if "list_tt" in working_df.columns and "datetime" in working_df.columns:
#         for combo in TRACK_COMBINATIONS:
#             try:
#                 combo_int = int(combo)
#             except ValueError:
#                 continue
#             subset = working_df[working_df["list_tt"] == combo_int]
#             if subset.empty:
#                 continue

#             print(f"Plotting charge distributions for combination {combo_int} with {len(subset)} events.")

#             # 1x4 grid of histograms (0–80 fC) for each plane's Q_sum_final; leave missing planes blank
#             fig, axes = plt.subplots(1, 4, figsize=(16, 4))
#             planes_in_combo = [int(p) for p in combo]
#             max_height = 0

#             for plane_id, ax in enumerate(axes, start=1):
#                 col_name = f'P{plane_id}_Q_sum_final'
#                 if plane_id in planes_in_combo and col_name in subset.columns:
#                     q_sum_data = subset[col_name].dropna()
#                     if not q_sum_data.empty:
#                         counts, bin_edges = np.histogram(q_sum_data, bins=50, range=(0, 80))
#                         max_height = max(max_height, counts.max(initial=0))
#                         ax.hist(q_sum_data, bins=50, range=(0, 80), color=plane_colors.get(plane_id, "blue"), alpha=0.5, label=f"P{plane_id}")
#                         ax.set_title(f'Plane {plane_id} Q_sum_final')
#                         ax.set_xlabel('Q_sum_final (fC)')
#                         ax.set_ylabel('Counts')
#                         ax.legend()
#                         continue
#                 ax.set_axis_off()
#             if max_height > 0:
#                 for ax in axes:
#                     if ax.has_data():
#                         ax.set_ylim(0, max_height * 1.05)
            

#             # Same y axis for all histograms
#             max_y = 0
#             for plane_id in planes_in_combo:
#                 col_name = f'P{plane_id}_Q_sum_final'
#                 if col_name in subset.columns:
#                     q_sum_data = subset[col_name].dropna()
#                     if not q_sum_data.empty:
#                         counts, _ = np.histogram(q_sum_data, bins=50, range=(0, 80))
#                         max_y = max(max_y, counts.max())
#             for ax in axes:
#                 ax.set_ylim(0, max_y * 1.1)

#             plt.suptitle(f'Charge Distributions for Combination {combo_int}')
#             plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#             plt.show()

# #%%

# # Differences between histograms of Q_sum_final for each pair of planes in the combination

# bin_number = 50
# max_bin = 60
# hist_diff_stats = {}  # {combo_int: { "p1-p2": {"pos": sum_positive, "neg": sum_negative} }}

# if create_plots:
#     # Detached method plots (per combination)
#     if "list_tt" in working_df.columns and "datetime" in working_df.columns:
#         for combo in TRACK_COMBINATIONS:

#             try:
#                 combo_int = int(combo)
#             except ValueError:
#                 continue
#             subset = working_df[working_df["list_tt"] == combo_int]
#             if subset.empty:
#                 continue

#             print(f"Plotting charge distributions for combination {combo} with {len(subset)} events.")

#             planes_in_combo = [int(p) for p in combo]
#             hist_diff_stats.setdefault(combo_int, {})

#             # Now i want a len(planes_in_combo)x len(planes_in_combo) grid
#             if len(planes_in_combo) >= 2:
#                 n_planes = len(planes_in_combo)
#                 fig, axes = plt.subplots(n_planes, n_planes, figsize=(4*n_planes, 4*n_planes))

#                 for i, p1 in enumerate(planes_in_combo):
#                     for j, p2 in enumerate(planes_in_combo):

#                         ax = axes[i, j]

#                         # Plot only the plots below the diagonal
#                         if j > i:
#                             ax.set_axis_off()
#                             continue

#                         if j == i:
#                             # Diagonal: histogram of that plane
#                             col_name = f'P{p1}_Q_sum_final'
#                             if col_name in subset.columns:
#                                 q_sum_data = subset[col_name].dropna()
#                                 if not q_sum_data.empty:
#                                     ax.hist(q_sum_data, bins=bin_number, range=(0, max_bin), color='blue', alpha=0.5)
#                                     ax.set_title(f'Plane {p1} Q_sum_final')
#                                     ax.set_xlabel('Q_sum_final (fC)')
#                                     ax.set_ylabel('Counts')
#                             else:
#                                 ax.set_axis_off()
#                             continue

#                         col_name1 = f'P{p1}_Q_sum_final'
#                         col_name2 = f'P{p2}_Q_sum_final'
#                         if col_name1 in subset.columns and col_name2 in subset.columns:
#                             q_sum_data1 = subset[col_name1].dropna()
#                             q_sum_data2 = subset[col_name2].dropna()
#                             common_indices = q_sum_data1.index.intersection(q_sum_data2.index)
#                             if not common_indices.empty:
#                                 # Calculate the histogram 1d for each case with 50 bins between 0 and 80 and make the difference
#                                 # I want you to plot in x the 0-80 and in y the difference of the histograms. Not scatter.
#                                 hist1, bin_edges = np.histogram(q_sum_data1.loc[common_indices], bins=bin_number, range=(0, max_bin))
#                                 hist2, _ = np.histogram(q_sum_data2.loc[common_indices], bins=bin_number, range=(0, max_bin))
#                                 hist_diff = hist1 - hist2
#                                 pos_bins = int((hist_diff > 0).sum())
#                                 neg_bins = int((hist_diff < 0).sum())
#                                 hist_diff_stats[combo_int][f"{p1}-{p2}"] = {"pos_bins": pos_bins, "neg_bins": neg_bins}
#                                 bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#                                 ax.bar(bin_centers, hist_diff, width=bin_edges[1]-bin_edges[0], alpha=0.5)
#                                 ax.set_xlabel(f'Q_sum_final (fC)')
#                                 ax.set_ylabel(f'Histogram Difference (Plane {p1} - Plane {p2})')
#                                 max_abs = np.abs(hist_diff).max(initial=0)
#                                 if max_abs > 0:
#                                     ax.set_ylim(-max_abs, max_abs)
#                                 # ax.set_xlim(0, 80)
#                                 ax.grid(True)
#                         else:
#                             ax.set_axis_off()
#                 plt.suptitle(f'Differences in histograms of Q_sum_final for Combination {combo}')
#                 plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#                 plt.show()

# #%%

# # Plane color palette for consistent plotting
# plane_colors = {1: "red", 2: "green", 3: "blue", 4: "purple"}

# bin_number = 100
# max_bin = 100
# hist_diff_stats = {}  # {combo_int: { "p1-p2": {"pos": sum_positive, "neg": sum_negative} }}

# # Overlayed histograms and differences on the same axes (one histogram positive, the other negative, plus the difference)
# if create_plots:
#     if "list_tt" in working_df.columns and "datetime" in working_df.columns:
#         for combo in TRACK_COMBINATIONS:

#             try:
#                 combo_int = int(combo)
#             except ValueError:
#                 continue
#             subset = working_df[working_df["cal_tt"] == combo_int]
#             if subset.empty:
#                 continue

#             planes_in_combo = [int(p) for p in combo]

#             if len(planes_in_combo) >= 2:
#                 n_planes = len(planes_in_combo)
#                 grid_size = n_planes - 1  # omit diagonal: rows/cols shifted to skip self-pairs
#                 fig, axes = plt.subplots(grid_size, grid_size, figsize=(4 * grid_size, 4 * grid_size))
#                 axes_array = np.array(axes, ndmin=2, dtype=object)

#                 for i in range(grid_size):
#                     p1 = planes_in_combo[i]
#                     for j in range(grid_size):
#                         p2 = planes_in_combo[j + 1]
#                         ax = axes_array[i, j]

#                         # Only plot when p2 comes after p1 (no diagonal/self)
#                         if p2 <= p1:
#                             ax.set_axis_off()
#                             continue

#                         col_name1 = f'P{p1}_Q_sum_final'
#                         col_name2 = f'P{p2}_Q_sum_final'
#                         if col_name1 in subset.columns and col_name2 in subset.columns:
#                             q_sum_data1 = subset[col_name1].dropna()
#                             q_sum_data2 = subset[col_name2].dropna()
#                             common_indices = q_sum_data1.index.intersection(q_sum_data2.index)
#                             if common_indices.empty:
#                                 ax.set_axis_off()
#                                 continue

#                             hist1, bin_edges = np.histogram(q_sum_data1.loc[common_indices], bins=bin_number, range=(0, max_bin))
#                             hist2, _ = np.histogram(q_sum_data2.loc[common_indices], bins=bin_number, range=(0, max_bin))
#                             hist_diff = hist1 - hist2
#                             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#                             max_abs = max(
#                                 hist1.max(initial=0),
#                                 hist2.max(initial=0),
#                                 np.abs(hist_diff).max(initial=0),
#                             )

#                             # Plot p1 as positive bars, p2 as negative bars, and overlay the difference
#                             color1 = plane_colors.get(p1, "gray")
#                             color2 = plane_colors.get(p2, "orange")
#                             ax.bar(bin_centers, hist1, width=bin_edges[1] - bin_edges[0], color=color1, alpha=0.4, label=f'P{p1}')
#                             ax.bar(bin_centers, -hist2, width=bin_edges[1] - bin_edges[0], color=color2, alpha=0.4, label=f'P{p2}')
#                             ax.plot(bin_centers, hist_diff, color='black', linewidth=1.0, label='diff')
#                             ax.set_xlabel('Q_sum_final (fC)')
#                             ax.set_ylabel(f'Hist +/- and diff (P{p1}-P{p2})')
#                             ax.legend()
#                             if max_abs > 0:
#                                 ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
#                             ax.grid(True)
#                             # if i == 0 and j == 0:
#                             #     ax.legend()
#                         else:
#                             ax.set_axis_off()

#                 plt.suptitle(f'Overlayed histograms and differences for Combination {combo}')
#                 plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#                 plt.show()

# #%%

# # Plane color palette for consistent plotting
# plane_colors = {1: "red", 2: "green", 3: "blue", 4: "purple"}
# hist_store = {}  # {combo_int: {plane_id: (hist, bin_edges)}}
# hist_comparison_stats = []  # records for later dataframe summary

# bin_number = 70
# max_bin = 40

# # Store per-combo, per-plane charge histograms and compare to reference combo
# hist_range = (0, max_bin)
# reference_combo = 1234
# reference_scale_fallback = 1.0  # fallback scalar if scaling cannot be computed
# scale_threshold = 10  # minimum charge (fC) to consider when computing scaling
# ref_scales = {}  # {plane_id: {combo_int: scale}}

# # Collect histograms
# if "list_tt" in working_df.columns and "datetime" in working_df.columns:
#     for combo in TRACK_COMBINATIONS:
#         try:
#             combo_int = int(combo)
#         except ValueError:
#             continue
#         subset = working_df[working_df["list_tt"] == combo_int]
#         if subset.empty:
#             continue

#         planes_in_combo = [int(p) for p in combo]
#         for plane_id in planes_in_combo:
#             col_name = f'P{plane_id}_Q_sum_final'
#             if col_name not in subset.columns:
#                 continue
#             q_sum_data = subset[col_name].dropna()
#             if q_sum_data.empty:
#                 continue
#             hist, bin_edges = np.histogram(q_sum_data, bins=bin_number, range=hist_range)
#             hist_store.setdefault(combo_int, {})[plane_id] = (hist, bin_edges)

# # Compute optimal per-combo scaling against the reference (per plane) using bins above the threshold
# if reference_combo in hist_store:
#     for plane_id in range(1, 5):
#         ref_entry = hist_store.get(reference_combo, {}).get(plane_id)
#         if ref_entry is None:
#             continue
#         ref_hist, ref_edges = ref_entry
#         ref_centers = (ref_edges[:-1] + ref_edges[1:]) / 2
#         mask = ref_centers >= scale_threshold
#         ref_slice = ref_hist[mask]
#         denom = np.dot(ref_slice, ref_slice)

#         for combo_int, plane_dict in hist_store.items():
#             if combo_int == reference_combo:
#                 continue
#             if plane_id not in plane_dict:
#                 continue
#             combo_hist, combo_edges = plane_dict[plane_id]
#             combo_centers = (combo_edges[:-1] + combo_edges[1:]) / 2
#             combo_slice = combo_hist[(combo_centers >= scale_threshold) & (combo_centers <= hist_range[1])]
#             if len(combo_slice) != len(ref_slice):
#                 ref_scales.setdefault(plane_id, {})[combo_int] = reference_scale_fallback
#                 continue
#             if denom == 0:
#                 ref_scales.setdefault(plane_id, {})[combo_int] = reference_scale_fallback
#                 continue
#             scale = float(np.dot(combo_slice, ref_slice) / denom)
#             ref_scales.setdefault(plane_id, {})[combo_int] = scale

# # Plot comparisons to reference per plane, one subplot per comparison combo (in a row).
# # Reference histogram is shown as negative bars (scaled) to mimic earlier overlay style; difference is also shown.
# if create_plots and reference_combo in hist_store:
#     for plane_id in range(1, 5):
#         ref_entry = hist_store.get(reference_combo, {}).get(plane_id)
#         if ref_entry is None:
#             continue
#         ref_hist, ref_edges = ref_entry
#         ref_centers = (ref_edges[:-1] + ref_edges[1:]) / 2
#         # ref_scaled will be combo-dependent below

#         # Gather other combos that include this plane
#         comparison_entries = []
#         for combo_int, plane_dict in hist_store.items():
#             if combo_int == reference_combo:
#                 continue
#             if plane_id in plane_dict:
#                 comparison_entries.append((combo_int, plane_dict[plane_id]))

#         if not comparison_entries:
#             continue

#         comparison_entries = sorted(comparison_entries, key=lambda x: x[0])
#         fig, axes = plt.subplots(1, len(comparison_entries), figsize=(6 * len(comparison_entries), 4))
#         axes_array = np.array(axes, ndmin=1, dtype=object)

#         for idx, (combo_int, (hist_vals, edges_vals)) in enumerate(comparison_entries):
#             ax = axes_array[idx]
#             centers = (edges_vals[:-1] + edges_vals[1:]) / 2
#             scale = ref_scales.get(plane_id, {}).get(combo_int, reference_scale_fallback)
#             ref_scaled = ref_hist * scale
#             diff_vals = hist_vals - ref_scaled

#             color = plane_colors.get(plane_id, "gray")
#             width = edges_vals[1] - edges_vals[0]

#             ax.bar(centers, hist_vals, width=width, color=color, alpha=0.5, label=f'P{plane_id} combo {combo_int}')
#             ax.bar(ref_centers, -ref_scaled, width=width, color=color, alpha=0.2,
#                    label=f'P{plane_id} combo {reference_combo} (scaled {scale:.3f})')
#             ax.plot(centers, diff_vals, color='black', linewidth=1.0, label='diff (combo - ref)')

#             max_abs = max(hist_vals.max(initial=0), ref_scaled.max(initial=0), np.abs(diff_vals).max(initial=0))
#             if max_abs > 0:
#                 ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
#             ax.set_xlabel('Q_sum_final (fC)')
#             ax.set_ylabel('Counts / Diff')
#             ax.set_title(f'Plane {plane_id}: combo {combo_int} vs {reference_combo}')
#             ax.grid(True)
#             ax.legend()

#             # Stats for dataframe summary
#             total_counts = int(hist_vals.sum())
#             mask_low = centers < scale_threshold
#             diff_low_counts = float( np.abs( diff_vals[mask_low].sum() ) ) if mask_low.any() else 0.0
#             diff_low_counts = round(diff_low_counts, 0)
#             noise_pct = (diff_low_counts / total_counts * 100.0) if total_counts > 0 else 0.0
#             hist_comparison_stats.append(
#                 {
#                     "plane": plane_id,
#                     "combo": combo_int,
#                     "reference_combo": reference_combo,
#                     "scale_used": scale,
#                     "scale_threshold": scale_threshold,
#                     "total_counts": total_counts,
#                     "diff_counts_below_threshold": diff_low_counts,
#                     "noise_pct": noise_pct,
#                 }
#             )

#         plt.suptitle(f'Plane {plane_id} histogram comparison vs reference {reference_combo}')
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.show()

# #%%

# # Print nicely in the terminal the hist_comparison_stats noise_pct per plane and combo
# if hist_comparison_stats:
#     print("Histogram Comparison Noise Summary (vs reference combo {}):".format(reference_combo))
#     print("{:<10} {:<10} {:<15} {:<12} {:<18} {:<12}".format(
#         "Plane", "Combo", "Scale Used", "Total Cts", "Diff Cts < Thr", "Noise (%)"
#     ))
#     for stat in hist_comparison_stats:
#         print("{:<10} {:<10} {:<15.4f} {:<12} {:<18.2f} {:<12.2f}".format(
#             stat["plane"],
#             stat["combo"],
#             stat["scale_used"],
#             stat["total_counts"],
#             stat["diff_counts_below_threshold"],
#             stat["noise_pct"]
#         ))

    

# #%%

# # Make a nice 4 row x combo length column python table in the spirit of plot_tt_correlation,
# # where the value shown is noise_pct and the background color is using that value in turbo colormap
# if hist_comparison_stats:
    
#     # Heatmap-like table of noise_pct (planes as rows, combos as columns) with turbo colormap
#     combos_all = sorted({stat["combo"] for stat in hist_comparison_stats})
#     planes_all = [1, 2, 3, 4]
#     data = np.full((len(planes_all), len(combos_all)), np.nan)
#     for stat in hist_comparison_stats:
#         plane_id = stat["plane"]
#         combo_id = stat["combo"]
#         if plane_id in planes_all and combo_id in combos_all:
#             i = planes_all.index(plane_id)
#             j = combos_all.index(combo_id)
#             data[i, j] = stat["noise_pct"]

#     fig, ax = plt.subplots(figsize=(1.1 * len(combos_all), 4))
#     ax.set_xticks(np.arange(len(combos_all)))
#     ax.set_yticks(np.arange(len(planes_all)))
#     ax.set_xticklabels([str(c) for c in combos_all])
#     ax.set_yticklabels([str(p) for p in planes_all])
#     ax.set_xlabel("Track Combination")
#     ax.set_ylabel("Plane")
#     ax.set_title("Histogram Comparison Noise (%) Summary")
#     cmap = plt.get_cmap('viridis')
#     norm = plt.Normalize(0, 100)
#     im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
#     plt.colorbar(im, ax=ax, label="Noise (%)")
#     for i in range(len(planes_all)):
#         for j in range(len(combos_all)):
#             if not np.isnan(data[i, j]):
#                 # If the colour of background is too bright, put the text to black, else white
#                 ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
#                         color="white" if data[i, j] < 50 else "black")
#     plt.tight_layout()
#     plt.show()
    

# #%%

# # Make a nice 4 row x combo length column python table in the spirit of plot_tt_correlation,
# # where the value shown is noise_pct and the background color is using that value in turbo colormap
# if hist_comparison_stats:
    
#     # Heatmap-like table of noise_pct (planes as rows, combos as columns) with turbo colormap
#     combos_all = sorted({stat["combo"] for stat in hist_comparison_stats})
#     planes_all = [1, 2, 3, 4]
#     data = np.full((len(planes_all), len(combos_all)), np.nan)
#     for stat in hist_comparison_stats:
#         plane_id = stat["plane"]
#         combo_id = stat["combo"]
#         if plane_id in planes_all and combo_id in combos_all:
#             i = planes_all.index(plane_id)
#             j = combos_all.index(combo_id)
#             data[i, j] = stat["diff_counts_below_threshold"]

#     fig, ax = plt.subplots(figsize=(1.1 * len(combos_all), 4))
#     ax.set_xticks(np.arange(len(combos_all)))
#     ax.set_yticks(np.arange(len(planes_all)))
#     ax.set_xticklabels([str(c) for c in combos_all])
#     ax.set_yticklabels([str(p) for p in planes_all])
#     ax.set_xlabel("Track Combination")
#     ax.set_ylabel("Plane")
#     ax.set_title("Histogram Comparison Noise COUNTS Summary")
#     cmap = plt.get_cmap('viridis')
#     # norm = plt.Normalize(0, 100)
#     # im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
#     im = ax.imshow(data, cmap=cmap, aspect='auto')
#     plt.colorbar(im, ax=ax, label="Noise COUNTS")
#     for i in range(len(planes_all)):
#         for j in range(len(combos_all)):
#             if not np.isnan(data[i, j]):
#                 # If the colour of background is too bright, put the text to black, else white
#                 ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
#                         color="white" if data[i, j] < np.nanmedian(data) * 2 else "black")
#     plt.tight_layout()
#     plt.show()

# #%%

# if create_plots:
#     # Detached method plots (per combination)
#     if "list_tt" in working_df.columns and "datetime" in working_df.columns:
#         for combo in TRACK_COMBINATIONS:

#             try:
#                 combo_int = int(combo)
#             except ValueError:
#                 continue
#             subset = working_df[working_df["list_tt"] == combo_int]
#             if subset.empty:
#                 continue

#             print(f"Plotting charge distributions for combination {combo} with {len(subset)} events.")

#             planes_in_combo = [int(p) for p in combo]

#             # Now i want a len(planes_in_combo)x len(planes_in_combo) grid of scatter plots
#             if len(planes_in_combo) >= 2:
#                 n_planes = len(planes_in_combo)
#                 fig, axes = plt.subplots(n_planes, n_planes, figsize=(4*n_planes, 4*n_planes))

#                 for i, p1 in enumerate(planes_in_combo):
#                     for j, p2 in enumerate(planes_in_combo):

#                         ax = axes[i, j]

#                         # Plot only the plots below the diagonal
#                         if j > i:
#                             ax.set_axis_off()
#                             continue

#                         if j == i:
#                             # Diagonal: histogram of that plane
#                             col_name = f'P{p1}_Q_sum_final'
#                             if col_name in subset.columns:
#                                 q_sum_data = subset[col_name].dropna()
#                                 if not q_sum_data.empty:
#                                     ax.hist(q_sum_data, bins=50, range=(0, 80), color='blue', alpha=0.5)
#                                     ax.set_title(f'Plane {p1} Q_sum_final')
#                                     ax.set_xlabel('Q_sum_final (fC)')
#                                     ax.set_ylabel('Counts')
#                             else:
#                                 ax.set_axis_off()
#                             continue

#                         col_name1 = f'P{p1}_Q_sum_final'
#                         col_name2 = f'P{p2}_Q_sum_final'
#                         if col_name1 in subset.columns and col_name2 in subset.columns:
#                             q_sum_data1 = subset[col_name1].dropna()
#                             q_sum_data2 = subset[col_name2].dropna()
#                             common_indices = q_sum_data1.index.intersection(q_sum_data2.index)
#                             if not common_indices.empty:
#                                 ax.scatter(q_sum_data1.loc[common_indices], q_sum_data2.loc[common_indices], alpha=0.5, s=1)
#                                 ax.set_xlabel(f'Plane {p1} Q_sum_final (fC)')
#                                 ax.set_ylabel(f'Plane {p2} Q_sum_final (fC)')
#                                 ax.set_xlim(0, 80)
#                                 ax.set_ylim(0, 80)
#                                 ax.grid(True)
#                         else:
#                             ax.set_axis_off()
#                 plt.suptitle(f'Scatter Plots of Q_sum_final for Combination {combo}')
#                 plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#                 plt.show()

# #%%

# # I want a calculation of efficiency which is simply counting how many list_tt 1234 are compared to list_tt 234 for eff 1,
# # 1234 vs 134 for eff 2, 1234 vs 124 for eff 3, 1234 vs 123 for eff 4. But I want to do a loop in charge threshold from 0 to 80
# # for all the planes in steps of 5 fC, and plot the efficiency curves for each plane.

# # HERE
# if create_plots and "list_tt" in working_df.columns:
#     eff_pairs = {1: 234, 2: 134, 3: 124, 4: 123}
#     thresholds = np.arange(1, 40, 0.5)  # 0 to 80 inclusive in 5 fC steps
#     plane_efficiency = {plane: [] for plane in eff_pairs}
#     plane_errors = {plane: [] for plane in eff_pairs}
#     charge_cols = {plane: f"P{plane}_Q_sum_final" for plane in eff_pairs}
#     available_planes = [p for p, col in charge_cols.items() if col in working_df.columns]

#     if available_planes:
#         # Preload charges as array (NaN -> 0 so threshold comparison is False)
#         charges = np.stack([working_df[charge_cols[p]].fillna(0).to_numpy() for p in [1, 2, 3, 4]], axis=1)
#         weights = np.array([1, 2, 4, 8])
#         mask_map = {1: 14, 2: 13, 3: 11, 4: 7}  # combo masks missing each plane

#         for thr in thresholds:
#             presence = charges >= thr  # (n,4)
#             combo_mask = (presence * weights).sum(axis=1)  # bitmask for planes passing thr
#             count_1234 = np.count_nonzero(combo_mask == 15)
#             for plane, missing_mask in mask_map.items():
#                 if plane not in available_planes:
#                     plane_efficiency[plane].append(np.nan)
#                     continue
#                 count_missing = np.count_nonzero(combo_mask == missing_mask)
#                 denom = count_1234 + count_missing
#                 eff = (count_1234 / denom) if denom > 0 else np.nan
#                 plane_efficiency[plane].append(eff)

#         # Error bars: binomial sqrt(p*(1-p)/N) per threshold
#         for plane, missing_mask in mask_map.items():
#             errs = []
#             if plane not in available_planes:
#                 plane_errors[plane] = [np.nan] * len(thresholds)
#                 continue
#             for idx_thr, thr in enumerate(thresholds):
#                 presence = charges >= thr
#                 combo_mask = (presence * weights).sum(axis=1)
#                 count_1234 = np.count_nonzero(combo_mask == 15)
#                 count_missing = np.count_nonzero(combo_mask == missing_mask)
#                 N = count_1234 + count_missing
#                 p = plane_efficiency[plane][idx_thr] if idx_thr < len(plane_efficiency[plane]) else np.nan
#                 if N > 0 and not np.isnan(p):
#                     errs.append(np.sqrt(p * (1 - p) / N))
#                 else:
#                     errs.append(np.nan)
#             plane_errors[plane] = errs

#     plt.figure(figsize=(8, 6))
#     for plane, effs in plane_efficiency.items():
#         color = plane_colors.get(plane, None)
#         errs = plane_errors.get(plane, [np.nan] * len(thresholds))
#         plt.plot(thresholds, effs, marker='o', label=f"Plane {plane}", color=color)
#         effs_arr = np.array(effs, dtype=float)
#         errs_arr = np.array(errs, dtype=float)
#         if np.any(np.isfinite(errs_arr)):
#             upper = effs_arr + errs_arr
#             lower = effs_arr - errs_arr
#             plt.fill_between(thresholds, lower, upper, color=color, alpha=0.15)
#     plt.xlabel("Charge threshold (fC)")
#     plt.ylabel("Efficiency (1234 vs missing-plane combo)")
#     plt.title("Efficiency vs threshold")
#     plt.ylim(0, 1.05)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# #%%

# # Contour plots scanning separate thresholds for 1234 and missing combo (per plane)
# if create_plots and "list_tt" in working_df.columns:
#     charge_cols = {plane: f"P{plane}_Q_sum_final" for plane in [1, 2, 3, 4]}
#     if all(col in working_df.columns for col in charge_cols.values()):
#         thr_vals = thresholds  # reuse same threshold grid
#         charge_arr = {p: working_df[charge_cols[p]].fillna(0).to_numpy() for p in [1, 2, 3, 4]}
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
#         axes = np.array(axes).reshape(-1)
#         im = None
#         for idx, plane in enumerate([1, 2, 3, 4]):
#             ax = axes[idx]
#             others = [p for p in [1, 2, 3, 4] if p != plane]
#             grid = np.full((len(thr_vals), len(thr_vals)), np.nan)
#             for i, thr_ref in enumerate(thr_vals):
#                 ref_mask = np.ones(len(working_df), dtype=bool)
#                 for p in [1, 2, 3, 4]:
#                     ref_mask &= charge_arr[p] >= thr_ref
#                 for j, thr_cmp in enumerate(thr_vals):
#                     cmp_mask = charge_arr[plane] < thr_cmp
#                     for p in others:
#                         cmp_mask &= charge_arr[p] >= thr_cmp
#                     total_1234 = ref_mask.sum()
#                     total_missing = cmp_mask.sum()
#                     denom = total_1234 + total_missing
#                     grid[i, j] = (total_1234 / denom) if denom > 0 else np.nan
#             im = ax.contourf(thr_vals, thr_vals, grid, levels=np.linspace(0, 1, 21), cmap="viridis", vmin=0, vmax=1)
#             ax.set_title(f"Plane {plane}")
#             ax.set_xlabel("Thr 1234 (fC)")
#             if idx in [0, 2]:
#                 ax.set_ylabel("Thr missing (fC)")
#             ax.grid(True, alpha=0.2)
#         if im is not None:
#             # The bar is over the plot, but it should not be
#             cbar = fig.colorbar(im, ax=axes.tolist(), label="Efficiency", shrink=0.95, pad=0.02)
#         fig.suptitle("Efficiency contour: threshold(1234) vs threshold(missing combo)", y=1.01)
#         # fig.tight_layout(rect=[0, 0, 1, 0.98])
#         plt.show()

#%%

_prof["s_data_read_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("-------------- Detached angle and slowness fitting ----------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# ---------------------------------------------------------------------------
# 1. Geometrical line fit (orthogonal-distance regression) ------------------
# ---------------------------------------------------------------------------

def fit_3d_line(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sx: float,
    sy: float,
    sz: float,
    plane_ids: Iterable[int],
    tdiff_to_x: float,
) -> Tuple[float, float, float, float, float,
           Dict[int, float], Dict[int, float]]:
    """
    Returns
    -------
    x_z0, y_z0              : intercept with z = 0
    theta, phi              : zenith (0 = down-coming) and azimuth  [rad]
    chi2                    : χ² of the ODR
    res_td_dict, res_y_dict : residuals per plane (ΔTdiff units, y units)
    """
    pts = np.column_stack((x, y, z))
    c   = pts.mean(axis=0)
    d   = np.linalg.svd(pts - c, full_matrices=False)[2][0]   # principal axis

    if d[2] < 0:                                              # enforce d_z > 0
        d = -d
    d /= np.linalg.norm(d)

    theta = np.arccos(d[2])
    phi   = np.arctan2(d[1], d[0])

    # z = 0 intercept
    t0  = -c[2] / d[2] if d[2] != 0 else np.nan
    xz0 = c[0] + t0 * d[0]
    yz0 = c[1] + t0 * d[1]

    # orthogonal residual vectors
    proj = np.outer((pts - c) @ d, d)                         # (N,3)
    res  = (pts - c) - proj

    res_td = res[:, 0] / tdiff_to_x
    res_y  = res[:, 1]

    chi2 = np.einsum('ij,ij->', res, res) / (sx**2 + sy**2 + sz**2)
    return (xz0, yz0, float(theta), float(phi), float(chi2), dict(zip(plane_ids, res_td)), dict(zip(plane_ids, res_y)))

# ---------------------------------------------------------------------------
# ---------------------------- Loop starts here -----------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Vectorized replacement for the per-event detached fitting loop.
# Groups events by active-plane bitmask; within each group a single batched
# np.linalg.svd call replaces N individual calls. np.polyfit is replaced by
# the closed-form OLS formula so it can be applied across the batch in one go.
# Numerically identical to the original per-event loop.
# ---------------------------------------------------------------------------
def _detached_vectorized(
    det_Q, det_Tdif, det_Y, det_Tsum,
    z_positions, tdiff_to_x,
    anc_sx, anc_sy, anc_sz, anc_sts,
    n, nplan,
    fit_res, slow_res,
    ext_res_ystr, ext_res_tsum, ext_res_tdif,
):
    _sx2sy2sz2 = float(anc_sx**2 + anc_sy**2 + anc_sz**2)

    def _batch_line_fit(x_b, y_b, z_sel):
        """Batched SVD line fit. x_b, y_b: (n_g, n_pl); z_sel: (n_pl,)."""
        n_g, n_pl = x_b.shape
        z_br  = np.broadcast_to(z_sel, (n_g, n_pl))
        pts   = np.stack([x_b, y_b, z_br], axis=2)        # (n_g, n_pl, 3)
        c     = pts.mean(axis=1, keepdims=True)             # (n_g,  1,   3)
        pts_c = pts - c
        _, _, vt = np.linalg.svd(pts_c, full_matrices=False)
        d = vt[:, 0, :].copy()                              # (n_g, 3)
        d[d[:, 2] < 0] *= -1
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        c3  = c[:, 0, :]
        t0  = np.where(d[:, 2] != 0.0, -c3[:, 2] / d[:, 2], np.nan)
        xz0 = c3[:, 0] + t0 * d[:, 0]
        yz0 = c3[:, 1] + t0 * d[:, 1]
        pdot = np.einsum('npi,ni->np', pts_c, d)            # (n_g, n_pl)
        proj = pdot[:, :, np.newaxis] * d[:, np.newaxis, :] # (n_g, n_pl, 3)
        res  = pts_c - proj
        res_td = res[:, :, 0] / tdiff_to_x
        res_y  = res[:, :, 1]
        chi2   = np.einsum('npi,npi->n', res, res) / _sx2sy2sz2
        return d, xz0, yz0, chi2, res_td, res_y

    def _batch_slowness(xz0, yz0, d, z_sel, Tsum_g):
        """Vectorized polyfit(s_rel, t_rel, 1) for a batch of events."""
        n_g, n_pl = Tsum_g.shape
        x_fit = xz0[:, np.newaxis] + d[:, 0:1] * z_sel / d[:, 2:3]
        y_fit = yz0[:, np.newaxis] + d[:, 1:2] * z_sel / d[:, 2:3]
        pos   = np.stack([x_fit, y_fit, np.broadcast_to(z_sel, (n_g, n_pl))], axis=2)
        rdist = np.einsum('npi,ni->np', pos, d)             # (n_g, n_pl)
        s_rel = rdist - rdist[:, 0:1]
        t_rel = Tsum_g - Tsum_g[:, 0:1]
        s_m   = s_rel.mean(axis=1); t_m = t_rel.mean(axis=1)
        ss2   = np.sum(s_rel ** 2, axis=1)
        st    = np.sum(s_rel * t_rel, axis=1)
        denom = ss2 - n_pl * s_m ** 2
        k     = np.where(np.abs(denom) > 0.0, (st - n_pl * s_m * t_m) / denom, 0.0)
        b     = t_m - k * s_m
        t_fit = k[:, np.newaxis] * s_rel + b[:, np.newaxis]
        res   = t_rel - t_fit
        chi2  = np.sum((res / anc_sts) ** 2, axis=1)
        return k, b, chi2, res, rdist

    q_pos   = det_Q > 0                                              # (n, nplan) bool
    bitmask = (q_pos * (1 << np.arange(nplan))).sum(axis=1).astype(np.int32)

    for bm in np.unique(bitmask):
        if bm == 0:
            continue
        active_idx = np.array([p for p in range(nplan) if bm & (1 << p)], dtype=int)
        n_pl       = len(active_idx)
        if n_pl < 2:
            continue
        plane_ids = active_idx + 1                                    # 1-based
        g_idx     = np.where(bitmask == bm)[0]
        n_g       = len(g_idx)
        z_sel     = z_positions[active_idx]                           # (n_pl,)
        Tdif_g    = det_Tdif[np.ix_(g_idx, active_idx)]
        Y_g       = det_Y   [np.ix_(g_idx, active_idx)]
        Tsum_g    = det_Tsum[np.ix_(g_idx, active_idx)]
        x_g       = tdiff_to_x * Tdif_g
        y_g       = Y_g

        # --- Primary fit ---
        d, xz0, yz0, chi2, res_td, res_y = _batch_line_fit(x_g, y_g, z_sel)
        theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
        phi   = np.arctan2(d[:, 1], d[:, 0])
        fit_res['det_x']    [g_idx] = xz0
        fit_res['det_y']    [g_idx] = yz0
        fit_res['det_theta'][g_idx] = theta
        fit_res['det_phi']  [g_idx] = phi
        fit_res['det_chi2'] [g_idx] = chi2
        for k_pl, pid in enumerate(plane_ids):
            fit_res[f'det_res_tdif_{pid}'][g_idx] = res_td[:, k_pl]
            fit_res[f'det_res_ystr_{pid}'][g_idx] = res_y [:, k_pl]

        # --- Slowness ---
        k_slow, b_slow, chi2_slow, res_slow, _ = _batch_slowness(xz0, yz0, d, z_sel, Tsum_g)
        slow_res['det_s']         [g_idx] = k_slow
        slow_res['det_s_ordinate'][g_idx] = b_slow
        slow_res['chi2_tsum_fit'] [g_idx] = chi2_slow
        for k_pl, pid in enumerate(plane_ids):
            slow_res[f'det_res_tsum_{pid}'][g_idx] = res_slow[:, k_pl]

        # --- Leave-one-out (requires ≥3 active planes) ---
        if n_pl < 3:
            continue
        for k_exc in range(n_pl):
            lo_k = [j for j in range(n_pl) if j != k_exc]
            if len(lo_k) < 2:
                continue
            pid_exc = int(plane_ids[k_exc])
            out_p   = pid_exc - 1                                     # 0-based output col
            z_lo    = z_sel[lo_k]
            x_lo    = x_g[:, lo_k]
            y_lo    = y_g[:, lo_k]
            ts_lo   = Tsum_g[:, lo_k]

            d_lo, xz0_lo, yz0_lo, _, _, _ = _batch_line_fit(x_lo, y_lo, z_lo)

            z_p    = float(z_sel[k_exc])
            x_pred = xz0_lo + d_lo[:, 0] * z_p / d_lo[:, 2]
            y_pred = yz0_lo + d_lo[:, 1] * z_p / d_lo[:, 2]

            ext_res_tdif[g_idx, out_p] = (x_g[:, k_exc] - x_pred) / tdiff_to_x
            ext_res_ystr[g_idx, out_p] = (y_g[:, k_exc] - y_pred)

            # Tsum external residual
            k_lo, b_lo, _, _, rdist_lo = _batch_slowness(xz0_lo, yz0_lo, d_lo, z_lo, ts_lo)
            pos_p  = np.stack([x_pred, y_pred, np.full(n_g, z_p)], axis=1)
            s_p    = np.einsum('ni,ni->n', pos_p, d_lo) - rdist_lo[:, 0]
            t_p    = Tsum_g[:, k_exc] - ts_lo[:, 0]
            ext_res_tsum[g_idx, out_p] = t_p - (k_lo * s_p + b_lo)

n = len(working_df)

# Angular definitions
fit_cols = (
    ['det_x', 'det_y', 'det_theta', 'det_phi', 'det_chi2'] +
    [f'det_res_tdif_{p}' for p in range(1, 5)] +
    [f'det_res_ystr_{p}' for p in range(1, 5)]
)

# Slowness definitions
slow_cols = ['det_s', 'det_s_ordinate' , 'chi2_tsum_fit'] + [f'det_res_tsum_{p}' for p in range(1, 5)]

# Pre-extract per-plane columns as contiguous numpy arrays (avoids per-event
# getattr overhead in the detached fitting loop below).
_det_Q    = np.column_stack([working_df[f'P{p}_Q_sum_final'].to_numpy(dtype=float) for p in range(1, nplan + 1)])
_det_Tdif = np.column_stack([working_df[f'P{p}_T_dif_final'].to_numpy(dtype=float) for p in range(1, nplan + 1)])
_det_Y    = np.column_stack([working_df[f'P{p}_Y_final'].to_numpy(dtype=float)     for p in range(1, nplan + 1)])
_det_Tsum = np.column_stack([working_df[f'P{p}_T_sum_final'].to_numpy(dtype=float) for p in range(1, nplan + 1)])

# Alternative analysis starts -----------------------------------------------
if run_detached_fit:
    repeat = number_of_det_executions - 1
    for det_iteration in range(repeat + 1):
        fitted = 0
        if number_of_det_executions > 1:
            print(f"Alternative iteration {det_iteration+1} out of {number_of_det_executions}.")
        
        fit_res = {c: np.zeros(n, dtype=float) for c in fit_cols}
        slow_res  = {c: np.zeros(n, dtype=float) for c in slow_cols}
        det_ext_res_ystr_arr = np.zeros((n, 4), dtype=float)
        det_ext_res_tsum_arr = np.zeros((n, 4), dtype=float)
        det_ext_res_tdif_arr = np.zeros((n, 4), dtype=float)
        det_processed_tt_arr = working_df.get("list_tt", pd.Series([0]*n)).astype(int).to_numpy()
        
        _detached_vectorized(
            _det_Q, _det_Tdif, _det_Y, _det_Tsum,
            z_positions, tdiff_to_x,
            anc_sx, anc_sy, anc_sz, anc_sts,
            n, nplan,
            fit_res, slow_res,
            det_ext_res_ystr_arr, det_ext_res_tsum_arr, det_ext_res_tdif_arr,
        )

        # 4.  Assemble all results and join once
        all_res = {**fit_res, **slow_res}
        all_res['det_ext_res_ystr_1'] = det_ext_res_ystr_arr[:, 0]
        all_res['det_ext_res_ystr_2'] = det_ext_res_ystr_arr[:, 1]
        all_res['det_ext_res_ystr_3'] = det_ext_res_ystr_arr[:, 2]
        all_res['det_ext_res_ystr_4'] = det_ext_res_ystr_arr[:, 3]
        all_res['det_ext_res_tsum_1'] = det_ext_res_tsum_arr[:, 0]
        all_res['det_ext_res_tsum_2'] = det_ext_res_tsum_arr[:, 1]
        all_res['det_ext_res_tsum_3'] = det_ext_res_tsum_arr[:, 2]
        all_res['det_ext_res_tsum_4'] = det_ext_res_tsum_arr[:, 3]
        all_res['det_ext_res_tdif_1'] = det_ext_res_tdif_arr[:, 0]
        all_res['det_ext_res_tdif_2'] = det_ext_res_tdif_arr[:, 1]
        all_res['det_ext_res_tdif_3'] = det_ext_res_tdif_arr[:, 2]
        all_res['det_ext_res_tdif_4'] = det_ext_res_tdif_arr[:, 3]
        all_res['det_th_chi'] = all_res['det_chi2'] + all_res['chi2_tsum_fit']
        all_res['det_processed_tt'] = det_processed_tt_arr

        new_cols = pd.DataFrame(all_res, index=working_df.index)
        dupes = new_cols.columns.intersection(working_df.columns)
        working_df = working_df.drop(columns=dupes, errors='ignore')
        working_df = working_df.join(new_cols)
        working_df = working_df.copy()
else:
    print("Skipping detached fitting (fit_method excludes detached).")

    # # Filter according to residual ------------------------------------------------
    # det_changed_event_count = 0
    # for index, row in working_df.iterrows():
    #     det_changed = False
    #     for i in range(1, 5):
    #         if abs(row[f'det_res_tsum_{i}']) > det_res_tsum_filter or \
    #             abs(row[f'det_res_tdif_{i}']) > det_res_tdif_filter or \
    #             abs(row[f'det_res_ystr_{i}']) > det_res_ystr_filter or \
    #             abs(row[f'det_ext_res_tsum_{i}']) > det_ext_res_tsum_filter or \
    #             abs(row[f'det_ext_res_tdif_{i}']) > det_ext_res_tdif_filter or \
    #             abs(row[f'det_ext_res_ystr_{i}']) > det_ext_res_ystr_filter:
    #             
    #             det_changed = True
    #             working_df.at[index, f'P{i}_Y_final'] = 0
    #             working_df.at[index, f'P{i}_T_sum_final'] = 0
    #             working_df.at[index, f'P{i}_T_dif_final'] = 0
    #             working_df.at[index, f'P{i}_Q_sum_final'] = 0
    #             working_df.at[index, f'P{i}_Q_dif_final'] = 0
    #             # Also clear residuals so they don't appear in other plane combinations
    #             for col in (
    #                 f'det_res_ystr_{i}', f'det_res_tsum_{i}', f'det_res_tdif_{i}',
    #                 f'det_ext_res_ystr_{i}', f'det_ext_res_tsum_{i}', f'det_ext_res_tdif_{i}'
    #             ):
    #                 if col in working_df.columns:
    #                     working_df.at[index, col] = 0
    #     if det_changed:
    #         det_changed_event_count += 1
    # print(f"--> {det_changed_event_count} events were residual filtered.")
    # record_filter_metric(
    #     "det_residual_zeroed_event_pct",
    #     det_changed_event_count,
    #     len(working_df),
    # )
    # 
    # det_iteration += 1

# ---------------------------------------------------------------------------
# Put every value close to 0 to effectively 0 -------------------------------
# ---------------------------------------------------------------------------

# Filter the values inside the machine number window ------------------------
remove_small = bool(config.get("remove_small", False))
remove_small_eps = float(config.get("remove_small_eps", 1e-7))
eps = remove_small_eps  # Threshold
def is_small_nonzero(x):
    return isinstance(x, (int, float)) and x != 0 and abs(x) < eps

if create_plots:

    # Flatten all numeric values except 0
    flat_values = working_df.select_dtypes(include=[np.number]).values.ravel()
    flat_values = flat_values[flat_values != 0]

    cond = abs(flat_values) < eps
    flat_values = flat_values[cond]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flat_values, bins=300, alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.title('Histogram of All Nonzero Values in working_df')
    plt.yscale('log')  # Optional: log scale to reveal structure
    plt.grid(True)
    plt.tight_layout()
    if save_plots:
        name_of_file = 'flat_values_histogram'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

def plot_ts_err_with_hist(df, base_cols, time_col, title):
    """Plot combined data: data TS+hist and error TS+hist (4 columns per variable)."""
    global fig_idx
    if df.empty:
        return
    n_vars = len(base_cols)
    fig, axes = plt.subplots(
        n_vars, 4, figsize=(18, 2.2 * n_vars),
        gridspec_kw={"width_ratios": [3, 1, 3, 1]},
        sharex="col"
    )
    if n_vars == 1:
        axes = np.array([axes])
    for idx, col in enumerate(base_cols):
        ts_ax, hist_ax, ts_err_ax, hist_err_ax = axes[idx]
        if col not in df.columns:
            for ax in (ts_ax, hist_ax, ts_err_ax, hist_err_ax):
                ax.set_visible(False)
            continue
        series = df[col].dropna()
        series = series[np.isfinite(series)]
        if series.empty:
            for ax in (ts_ax, hist_ax, ts_err_ax, hist_err_ax):
                ax.set_visible(False)
            continue
        err_col = f"{col}_err"
        err_series = df[err_col].dropna() if err_col in df.columns else None
        if err_series is not None:
            err_series = err_series[np.isfinite(err_series)]
        yerr = err_series.abs() if err_series is not None else None
        ts_ax.errorbar(df[time_col], df[col], yerr=yerr, fmt=".", ms=1, alpha=0.85)
        ts_ax.set_ylabel(col)
        ts_ax.grid(True, alpha=0.3)
        bins, hist_range = _safe_hist_params(series, max_bins=50)
        if bins is not None:
            hist_ax.hist(
                series,
                bins=bins,
                range=hist_range,
                orientation="horizontal",
                color="C2",
                alpha=0.8,
            )
        hist_ax.set_xlabel("count")
        hist_ax.grid(True, alpha=0.2)
        if err_series is not None and not err_series.empty:
            ts_err_ax.plot(df[time_col], err_series, ".", ms=1, alpha=0.8, label=f"{col}_err")
            ts_err_ax.grid(True, alpha=0.3)
            ts_err_ax.legend(fontsize="x-small")
            err_bins, err_range = _safe_hist_params(err_series, max_bins=50)
            if err_bins is not None:
                hist_err_ax.hist(
                    err_series,
                    bins=err_bins,
                    range=err_range,
                    orientation="horizontal",
                    color="C4",
                    alpha=0.7,
                )
            hist_err_ax.set_autoscale_on(True)
            hist_err_ax.autoscale_view()
            hist_err_ax.set_xscale("log")
            hist_err_ax.set_xlabel("count")
            hist_err_ax.grid(True, alpha=0.2)
        else:
            ts_err_ax.set_visible(False)
            hist_err_ax.set_visible(False)
    axes[-1, 0].set_xlabel(time_col)
    axes[-1, 2].set_xlabel(time_col)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_plots:
        final_filename = f"{fig_idx}_{title.replace(' ', '_')}.png"
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format="png")
    if show_plots:
        plt.show()
    plt.close()

if remove_small:
    # Filter the small values ----------------------------------------------------
    if create_debug_plots:
        numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            debug_thresholds = {col: [-eps, eps] for col in numeric_cols}
            debug_fig_idx = plot_debug_histograms(
                working_df,
                numeric_cols,
                debug_thresholds,
                title=(
                    f"Task 4 pre-filter (intermediate): remove_small_eps={eps:g} "
                    f"[tunable] (station {station})"
                ),
                out_dir=debug_plot_directory,
                fig_idx=debug_fig_idx,
            )
    mask = working_df.map(is_small_nonzero)  # Create mask of small, non-zero numeric values
    nonzero_numeric_mask = working_df.map(lambda x: isinstance(x, (int, float)) and x != 0)  # Count total non-zero numeric entries
    n_events = len(working_df)
    rows_with_small = int(mask.any(axis=1).sum())
    n_total = nonzero_numeric_mask.sum().sum()
    n_small = mask.sum().sum()
    working_df = working_df.mask(mask, 0)  # Apply the replacement
    pct = 100 * n_small / n_total if n_total > 0 else 0
    print(f"{n_small} out of {n_total} non-zero numeric values are below {eps} ({pct:.4f}%)")  # Report
    record_filter_metric(
        "small_values_zeroed_event_pct",
        rows_with_small,
        n_events if n_events else 0,
    )
    record_filter_metric(
        "small_values_zeroed_value_pct",
        n_small,
        n_total if n_total else 0,
    )

# det_bounds_changed = np.zeros(len(working_df), dtype=bool)
# for col in working_df.columns:
#     # Alternative fitting results
#     if 'det_x' == col or 'det_y' == col:
#         cond_bound = (working_df[col] > det_pos_filter) | (working_df[col] < -1*det_pos_filter)
#         cond_zero = (working_df[col] == 0)
#         change_mask = cond_bound | cond_zero
#         det_bounds_changed |= change_mask
#         working_df.loc[:, col] = np.where(change_mask, 0, working_df[col])
#     if 'det_theta' == col:
#         cond_bound = (working_df[col] > det_theta_right_filter) | (working_df[col] < det_theta_left_filter)
#         cond_zero = (working_df[col] == 0)
#         change_mask = cond_bound | cond_zero
#         det_bounds_changed |= change_mask
#         working_df.loc[:, col] = np.where(change_mask, 0, working_df[col])
#     if 'det_phi' == col:
#         cond_bound = (working_df[col] > det_phi_right_filter) | (working_df[col] < det_phi_left_filter)
#         cond_zero = (working_df[col] == 0)
#         change_mask = cond_bound | cond_zero
#         det_bounds_changed |= change_mask
#         working_df.loc[:, col] = np.where(change_mask, 0, working_df[col])
#     if 'det_s' == col:
#         cond_bound = (working_df[col] > det_slowness_filter_right) | (working_df[col] < det_slowness_filter_left)
#         cond_zero = (working_df[col] == 0)
#         working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])

# record_filter_metric(
#     "det_bounds_zeroed_event_pct",
#     float(det_bounds_changed.sum()),
#     float(len(working_df)),
# )

print("Alternative fitting done.")

#%%

# Build detached (independent) variables
working_df["det_x"] = working_df.get("det_x", 0)
working_df["det_y"] = working_df.get("det_y", 0)
working_df["det_theta"] = working_df.get("det_theta", 0)
working_df["det_phi"] = working_df.get("det_phi", 0)
working_df["det_s"] = working_df.get("det_s", 0)
working_df["det_t0"] = working_df.get("det_s_ordinate", 0)

for p in range(1, 5):
    working_df[f"det_res_ystr_{p}"] = working_df.get(f"det_res_ystr_{p}", 0)
    working_df[f"det_res_tsum_{p}"] = working_df.get(f"det_res_tsum_{p}", 0)
    working_df[f"det_res_tdif_{p}"] = working_df.get(f"det_res_tdif_{p}", 0)
    working_df[f"det_ext_res_ystr_{p}"] = working_df.get(f"det_ext_res_ystr_{p}", 0)
    working_df[f"det_ext_res_tsum_{p}"] = working_df.get(f"det_ext_res_tsum_{p}", 0)
    working_df[f"det_ext_res_tdif_{p}"] = working_df.get(f"det_ext_res_tdif_{p}", 0)

#%%

if create_plots:
    # Detached method plots (per combination)
    if "det_processed_tt" in working_df.columns and "datetime" in working_df.columns:
        for combo in TRACK_COMBINATIONS:
            try:
                combo_int = int(combo)
            except ValueError:
                continue
            subset = working_df[working_df["det_processed_tt"] == combo_int]
            if subset.empty:
                continue
            plot_ts_with_side_hist(
                subset,
                ["det_x", "det_y", "det_theta", "det_phi", "det_s", "det_t0"],
                "datetime",
                f"detached_timeseries_combo_{combo}",
            )
            # Only plot residuals when we have 3+ planes
            if len(str(combo_int)) >= 3:
                plot_residuals_ts_hist(
                    subset,
                    prefixes=[
                        ("ystr", "det_res_ystr_", "det_ext_res_ystr_"),
                        ("tsum", "det_res_tsum_", "det_ext_res_tsum_"),
                        ("tdif", "det_res_tdif_", "det_ext_res_tdif_"),
                    ],
                    time_col="datetime",
                    title=f"detached_residuals_combo_{combo}",
                )

#%%

_prof["s_detached_fitting_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("------------------------- TimTrack fitting ---------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

if fixed_speed:
    print("Fixed the slowness to 1 / speed of light.")
    npar = 5
else:
    print("Slowness not fixed.")
    npar = 6

def fmgx(nvar, npar, vs, ss, zi): # G matrix for t measurements in X-axis
    mg = np.zeros([nvar, npar])
    XP = vs[1]; YP = vs[3]
    if fixed_speed:
        S0 = sc
    else:
        S0 = vs[5]
    kz = math.sqrt(1 + XP*XP + YP*YP)
    kzi = 1 / kz
    mg[0,2] = 1
    mg[0,3] = zi
    mg[1,1] = kzi * S0 * XP * zi
    mg[1,3] = kzi * S0 * YP * zi
    mg[1,4] = 1
    if fixed_speed == False: mg[1,5] = kz * zi
    mg[2,0] = ss
    mg[2,1] = ss * zi
    return mg

def fmwx(nvar, vsig): # Weigth matrix 
    sy = vsig[0]; sts = vsig[1]; std = vsig[2]
    mw = np.zeros([nvar, nvar])
    mw[0,0] = 1/(sy*sy)
    mw[1,1] = 1/(sts*sts)
    mw[2,2] = 1/(std*std)
    return mw

def fvmx(nvar, vs, lenx, ss, zi): # Fitting model array with X-strips
    vm = np.zeros(nvar)
    X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]
    if fixed_speed:
        S0 = sc
    else:
        S0 = vs[5]
    kz = np.sqrt(1 + XP*XP + YP*YP)
    xi = X0 + XP * zi
    yi = Y0 + YP * zi
    ti = T0 + kz * S0 * zi
    th = 0.5 * lenx * ss   # tau half
    # lxmn = -lenx/2
    vm[0] = yi
    vm[1] = th + ti
    # vm[2] = ss * (xi-lxmn) - th
    vm[2] = ss * xi
    return vm

def fmkx(nvar, npar, vs, vsig, ss, zi): # K matrix
    mk  = np.zeros([npar,npar])
    mg  = fmgx(nvar, npar, vs, ss, zi)
    mgt = mg.transpose()
    mw  = fmwx(nvar, vsig)
    mk  = mgt @ mw @ mg
    return mk

def fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi): # va vector
    va = np.zeros(npar)
    mw = fmwx(nvar, vsig)
    vm = fvmx(nvar, vs, lenx, ss, zi)
    mg = fmgx(nvar, npar, vs, ss, zi)
    vg = vm - mg @ vs
    vdmg = vdat - vg
    va = mg.transpose() @ mw @ vdmg
    return va

def fmahd(npar, vin1, vin2, merr): # Mahalanobis distance
    merr_diag = np.diag(merr) if merr.ndim > 1 else merr
    acc = 0.0
    for i in range(npar):
        d = vin1[i] - vin2[i]
        m = merr_diag[i]
        if m != 0.0:
            acc += d * d / m
    return math.sqrt(acc)

def solve_only(matrix, rhs):
    """Solve matrix @ x = rhs (fast path for code paths that do not need covariance)."""
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ rhs

def solve_and_covdiag(matrix, rhs):
    """Solve matrix @ x = rhs and return (x, diag(inv(matrix))) with one factorization.

    Fast path computes inverse once and reuses it for both the linear solve and
    covariance-diagonal extraction. Fallback uses pinv once.
    """
    try:
        inv_matrix = np.linalg.inv(matrix)
        sol = inv_matrix @ rhs
        inv_diag = np.diag(inv_matrix)
        return sol, inv_diag
    except np.linalg.LinAlgError:
        matrix_pinv = np.linalg.pinv(matrix)
        return matrix_pinv @ rhs, np.diag(matrix_pinv)

def _accumulate_mk_va(npar, vs, ydat, tsdat, tddat, zi, w_arr, ss, lenx, sc_val, fixed_speed_flag, mk, va):
    """In-place TimTrack accumulator: adds one plane contribution into mk and va.

    Avoids per-plane temporary (npar x npar) and (npar,) allocations from
    return-based helpers in the main fitting hot loop.
    """
    XP = vs[1]
    YP = vs[3]
    S0 = sc_val if fixed_speed_flag else vs[5]
    kz = math.sqrt(1.0 + XP * XP + YP * YP)
    kzi = 1.0 / kz

    w0 = w_arr[0]
    w1 = w_arr[1]
    w2 = w_arr[2]

    zi2 = zi * zi
    sszi = ss * zi
    th = 0.5 * lenx * ss

    # Non-zero entries of the 3 Jacobian rows (g0, g1, g2)
    g0_2 = 1.0
    g0_3 = zi

    g1_1 = kzi * S0 * XP * zi
    g1_3 = kzi * S0 * YP * zi
    g1_4 = 1.0

    g2_0 = ss
    g2_1 = sszi

    # vdmg terms simplify exactly for this linearised model
    vd0 = ydat
    vd1 = tsdat - th
    vd2 = tddat

    # g0 contribution (weight w0)
    mk[2, 2] += w0 * g0_2 * g0_2
    mk[2, 3] += w0 * g0_2 * g0_3
    mk[3, 2] += w0 * g0_3 * g0_2
    mk[3, 3] += w0 * g0_3 * g0_3
    va[2] += w0 * g0_2 * vd0
    va[3] += w0 * g0_3 * vd0

    # g2 contribution (weight w2)
    mk[0, 0] += w2 * g2_0 * g2_0
    mk[0, 1] += w2 * g2_0 * g2_1
    mk[1, 0] += w2 * g2_1 * g2_0
    mk[1, 1] += w2 * g2_1 * g2_1
    va[0] += w2 * g2_0 * vd2
    va[1] += w2 * g2_1 * vd2

    # g1 contribution (weight w1)
    mk[1, 1] += w1 * g1_1 * g1_1
    mk[1, 3] += w1 * g1_1 * g1_3
    mk[3, 1] += w1 * g1_3 * g1_1
    mk[1, 4] += w1 * g1_1 * g1_4
    mk[4, 1] += w1 * g1_4 * g1_1
    mk[3, 3] += w1 * g1_3 * g1_3
    mk[3, 4] += w1 * g1_3 * g1_4
    mk[4, 3] += w1 * g1_4 * g1_3
    mk[4, 4] += w1 * g1_4 * g1_4
    va[1] += w1 * g1_1 * vd1
    va[3] += w1 * g1_3 * vd1
    va[4] += w1 * g1_4 * vd1

    if not fixed_speed_flag:
        g1_5 = kz * zi
        mk[1, 5] += w1 * g1_1 * g1_5
        mk[5, 1] += w1 * g1_5 * g1_1
        mk[3, 5] += w1 * g1_3 * g1_5
        mk[5, 3] += w1 * g1_5 * g1_3
        mk[4, 5] += w1 * g1_4 * g1_5
        mk[5, 4] += w1 * g1_5 * g1_4
        mk[5, 5] += w1 * g1_5 * g1_5
        va[5] += w1 * g1_5 * vd1

def _build_mk_va_base(
    plane_idx_arr,
    y_row,
    td_row,
    z_pos_arr,
    w_arr,
    ss,
    mk_base,
    va_base,
):
    """Build per-event constant TimTrack terms (independent of current vs)."""
    mk_base.fill(0.0)
    va_base.fill(0.0)

    w0 = w_arr[0]
    w2 = w_arr[2]
    ss2 = ss * ss
    w2_ss = w2 * ss
    w2_ss2 = w2 * ss2

    for plane_idx in plane_idx_arr:
        zi = z_pos_arr[plane_idx]
        ydat = y_row[plane_idx]
        tddat = td_row[plane_idx]
        sszi = ss * zi
        zi2 = zi * zi

        # g0 contribution (weight w0)
        mk_base[2, 2] += w0
        mk_base[2, 3] += w0 * zi
        mk_base[3, 2] += w0 * zi
        mk_base[3, 3] += w0 * zi2
        va_base[2] += w0 * ydat
        va_base[3] += w0 * zi * ydat

        # g2 contribution (weight w2)
        mk_base[0, 0] += w2_ss2
        mk_base[0, 1] += w2_ss2 * zi
        mk_base[1, 0] += w2_ss2 * zi
        mk_base[1, 1] += w2 * sszi * sszi
        va_base[0] += w2_ss * tddat
        va_base[1] += w2 * sszi * tddat

def _accumulate_mk_va_dynamic_g1(
    vs,
    plane_idx_arr,
    ts_row,
    z_pos_arr,
    w1,
    half_lenx_ss,
    sc_val,
    fixed_speed_flag,
    mk,
    va,
):
    """Add per-iteration dynamic g1 terms (depend on current vs)."""
    XP = vs[1]
    YP = vs[3]
    S0 = sc_val if fixed_speed_flag else vs[5]
    kz = math.sqrt(1.0 + XP * XP + YP * YP)
    kzi = 1.0 / kz

    for plane_idx in plane_idx_arr:
        zi = z_pos_arr[plane_idx]
        vd1 = ts_row[plane_idx] - half_lenx_ss

        g1_1 = kzi * S0 * XP * zi
        g1_3 = kzi * S0 * YP * zi

        mk[1, 1] += w1 * g1_1 * g1_1
        mk[1, 3] += w1 * g1_1 * g1_3
        mk[3, 1] += w1 * g1_3 * g1_1
        mk[1, 4] += w1 * g1_1
        mk[4, 1] += w1 * g1_1
        mk[3, 3] += w1 * g1_3 * g1_3
        mk[3, 4] += w1 * g1_3
        mk[4, 3] += w1 * g1_3
        mk[4, 4] += w1
        va[1] += w1 * g1_1 * vd1
        va[3] += w1 * g1_3 * vd1
        va[4] += w1 * vd1

        if not fixed_speed_flag:
            g1_5 = kz * zi
            mk[1, 5] += w1 * g1_1 * g1_5
            mk[5, 1] += w1 * g1_5 * g1_1
            mk[3, 5] += w1 * g1_3 * g1_5
            mk[5, 3] += w1 * g1_5 * g1_3
            mk[4, 5] += w1 * g1_5
            mk[5, 4] += w1 * g1_5
            mk[5, 5] += w1 * g1_5 * g1_5
            va[5] += w1 * g1_5 * vd1

def _build_loo_single_step_system(
    vs,
    plane_idx_arr,
    plane_idx_ref,
    y_row,
    ts_row,
    td_row,
    z_pos_arr,
    w_arr,
    ss,
    half_lenx_ss,
    sc_val,
    fixed_speed_flag,
    mk,
    va,
):
    """Build LOO normal equations for one hidden plane in single-step mode.

    This is equivalent to repeated _accumulate_mk_va calls with shifted z, but
    avoids per-plane helper call overhead in the hottest residual-LOO path.
    """
    mk.fill(0.0)
    va.fill(0.0)

    XP = vs[1]
    YP = vs[3]
    S0 = sc_val if fixed_speed_flag else vs[5]
    kz = math.sqrt(1.0 + XP * XP + YP * YP)
    kzi = 1.0 / kz

    a1 = kzi * S0 * XP
    a3 = kzi * S0 * YP

    w0 = w_arr[0]
    w1 = w_arr[1]
    w2 = w_arr[2]
    w2_ss = w2 * ss
    w2_ss2 = w2_ss * ss

    z_ref = z_pos_arr[plane_idx_ref]

    for plane_idx in plane_idx_arr:
        if plane_idx == plane_idx_ref:
            continue

        zi = z_pos_arr[plane_idx] - z_ref
        zi2 = zi * zi
        sszi = ss * zi

        ydat = y_row[plane_idx]
        tsdat = ts_row[plane_idx]
        tddat = td_row[plane_idx]
        vd1 = tsdat - half_lenx_ss

        # g0 contribution (weight w0)
        mk[2, 2] += w0
        mk[2, 3] += w0 * zi
        mk[3, 2] += w0 * zi
        mk[3, 3] += w0 * zi2
        va[2] += w0 * ydat
        va[3] += w0 * zi * ydat

        # g2 contribution (weight w2)
        mk[0, 0] += w2_ss2
        mk[0, 1] += w2_ss2 * zi
        mk[1, 0] += w2_ss2 * zi
        mk[1, 1] += w2 * sszi * sszi
        va[0] += w2_ss * tddat
        va[1] += w2 * sszi * tddat

        # g1 contribution (weight w1)
        g1_1 = a1 * zi
        g1_3 = a3 * zi
        mk[1, 1] += w1 * g1_1 * g1_1
        mk[1, 3] += w1 * g1_1 * g1_3
        mk[3, 1] += w1 * g1_3 * g1_1
        mk[1, 4] += w1 * g1_1
        mk[4, 1] += w1 * g1_1
        mk[3, 3] += w1 * g1_3 * g1_3
        mk[3, 4] += w1 * g1_3
        mk[4, 3] += w1 * g1_3
        mk[4, 4] += w1
        va[1] += w1 * g1_1 * vd1
        va[3] += w1 * g1_3 * vd1
        va[4] += w1 * vd1

        if not fixed_speed_flag:
            g1_5 = kz * zi
            mk[1, 5] += w1 * g1_1 * g1_5
            mk[5, 1] += w1 * g1_5 * g1_1
            mk[3, 5] += w1 * g1_3 * g1_5
            mk[5, 3] += w1 * g1_5 * g1_3
            mk[4, 5] += w1 * g1_5
            mk[5, 4] += w1 * g1_5
            mk[5, 5] += w1 * g1_5 * g1_5
            va[5] += w1 * g1_5 * vd1

def _fres_zi0(vs, ydat, tsdat, tddat, half_lenx_ss, ss):
    """Residuals at zi=0 for LOO hidden plane evaluation."""
    x0 = vs[0]
    y0 = vs[2]
    t0 = vs[4]
    return [y0 - ydat, (t0 + half_lenx_ss) - tsdat, (x0 * ss) - tddat]

def fres(vs, vdat, lenx, ss, zi):  # Residuals array
    X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]
    if fixed_speed:
        S0 = sc
    else:
        S0 = vs[5]
    kz = math.sqrt(1 + XP*XP + YP*YP)
    xfit = X0 + XP * zi
    yfit = Y0 + YP * zi
    tsfit = T0 + S0 * kz * zi + 0.5 * lenx * ss
    tdfit = ss * xfit
    ydat  = vdat[0]
    tsdat = vdat[1]
    tddat = vdat[2]
    return (yfit - ydat, tsfit - tsdat, tdfit - tddat)

def extract_plane_data(pos, iplane):
    zi  = z_positions[iplane - 1]
    yst = _tt_Y[pos, iplane - 1]
    ts  = _tt_Tsum[pos, iplane - 1]
    td  = _tt_Tdif[pos, iplane - 1]
    return [yst, ts, td], _tt_vsig, zi

nvar = 3
i = 0
ntrk  = len(working_df)
if limit and limit_number < ntrk: ntrk = limit_number
print("-----------------------------")
print(f"{ntrk} events to be fitted")

timtrack_results = [
    'tim_x', 'tim_xp', 'tim_y', 'tim_yp', 'tim_t0', 'tim_s',
    'tim_th_chi', 'tim_res_y', 'tim_res_ts', 'tim_res_td', 'tim_list_tt',
    'tim_res_ystr_1', 'tim_res_ystr_2', 'tim_res_ystr_3', 'tim_res_ystr_4',
    'tim_res_tsum_1', 'tim_res_tsum_2', 'tim_res_tsum_3', 'tim_res_tsum_4',
    'tim_res_tdif_1', 'tim_res_tdif_2', 'tim_res_tdif_3', 'tim_res_tdif_4',
    'tim_ext_res_ystr_1', 'tim_ext_res_ystr_2', 'tim_ext_res_ystr_3', 'tim_ext_res_ystr_4',
    'tim_ext_res_tsum_1', 'tim_ext_res_tsum_2', 'tim_ext_res_tsum_3', 'tim_ext_res_tsum_4',
    'tim_ext_res_tdif_1', 'tim_ext_res_tdif_2', 'tim_ext_res_tdif_3', 'tim_ext_res_tdif_4',
    'tim_charge_1', 'tim_charge_2', 'tim_charge_3', 'tim_charge_4', 'tim_charge_event',
    "tim_iterations", "tim_conv_distance", 'tim_converged'
]

missing_tim_cols = {col: 0.0 for col in timtrack_results if col not in working_df.columns}
if missing_tim_cols:
    working_df = pd.concat([working_df, pd.DataFrame(missing_tim_cols, index=working_df.index)], axis=1)

# Pre-extract per-plane columns for TimTrack (avoids per-event getattr overhead).
# _tt_vsig is constant for every plane/event so it is allocated once here.
_tt_Q    = np.column_stack([working_df[f'P{p}_Q_sum_final'].to_numpy(dtype=float) for p in range(1, nplan + 1)])
_tt_Tsum = np.column_stack([working_df[f'P{p}_T_sum_final'].to_numpy(dtype=float) for p in range(1, nplan + 1)])
_tt_Tdif = np.column_stack([working_df[f'P{p}_T_dif_final'].to_numpy(dtype=float) for p in range(1, nplan + 1)])
_tt_Y    = np.column_stack([working_df[f'P{p}_Y_final'].to_numpy(dtype=float)     for p in range(1, nplan + 1)])
_tt_vsig = [anc_sy, anc_sts, anc_std]
# Pre-computed weight array for _fmk_and_va (constant for all events/planes/iterations)
_w_arr  = np.array([1.0 / anc_sy**2, 1.0 / anc_sts**2, 1.0 / anc_std**2], dtype=float)
_sc_val = sc
_z_pos_arr = np.asarray(z_positions, dtype=float)
_half_lenx_ss = 0.5 * lenx * ss

def _cfg_int_or_default(key: str, default_value: int, *, min_value: int = 1) -> int:
    raw = config.get(key, default_value)
    try:
        if raw is None or str(raw).strip() == "":
            return int(default_value)
        value = int(float(raw))
    except (TypeError, ValueError):
        return int(default_value)
    return max(min_value, value)

def _cfg_float_or_default(key: str, default_value: float, *, min_value: float = 0.0) -> float:
    raw = config.get(key, default_value)
    try:
        if raw is None or str(raw).strip() == "":
            return float(default_value)
        value = float(raw)
    except (TypeError, ValueError):
        return float(default_value)
    return max(min_value, value)

tt_loo_every_n = _cfg_int_or_default("tt_loo_every_n", 1, min_value=1)
tt_loo_iter_max = _cfg_int_or_default("tt_loo_iter_max", int(iter_max), min_value=1)
tt_loo_cocut = _cfg_float_or_default("tt_loo_cocut", float(cocut), min_value=0.0)
_tt_loo_mode_raw = str(config.get("tt_loo_mode", "single_step")).strip().lower()
if _tt_loo_mode_raw not in ("single_step", "iterative"):
    print(f"WARNING: unrecognised tt_loo_mode='{_tt_loo_mode_raw}', falling back to 'single_step'")
    _tt_loo_mode_raw = "single_step"
tt_loo_mode = _tt_loo_mode_raw
print(
    "TimTrack LOO settings: "
    f"every_n={tt_loo_every_n}, "
    f"iter_max={tt_loo_iter_max}, "
    f"cocut={tt_loo_cocut:g}, "
    f"mode={tt_loo_mode}"
)

_tt_seed_enabled = False
_tt_seed_slope_abs_max = float(config.get("tt_seed_slope_abs_max", 5.0))
if run_detached_fit and all(
    col in working_df.columns
    for col in ("det_x", "det_y", "det_theta", "det_phi", "det_t0", "det_s")
):
    _seed_x = working_df["det_x"].to_numpy(dtype=float)
    _seed_y = working_df["det_y"].to_numpy(dtype=float)
    _seed_theta = working_df["det_theta"].to_numpy(dtype=float)
    _seed_phi = working_df["det_phi"].to_numpy(dtype=float)
    _seed_t0 = working_df["det_t0"].to_numpy(dtype=float)
    _seed_s = working_df["det_s"].to_numpy(dtype=float)
    _tt_seed_enabled = True
    print("TimTrack warm-start enabled from detached fit columns.")
else:
    print("TimTrack warm-start disabled (detached seed columns unavailable).")

# TimTrack thin profiling breakdown (accumulated across all TT iterations)
_tt_intro_s = 0.0
_tt_main_fit_s = 0.0
_tt_residual_s = 0.0
_tt_residual_loo_s = 0.0
_tt_writeback_s = 0.0

# TimTrack thin profiling counters (for deeper bottleneck diagnosis)
_tt_events_total = 0
_tt_events_with_2plus_planes = 0
_tt_events_loo_eligible = 0
_tt_events_loo_processed = 0
_tt_mainfit_iterations_total = 0
_tt_loo_refits_total = 0
_tt_loo_iterations_total = 0

# TimTrack starts ------------------------------------------------------
if not run_timtrack_fit:
    print("Skipping TimTrack fitting (fit_method excludes timtrack).")
repeat = number_of_tt_executions - 1 if run_timtrack_fit else -1
for iteration in range(repeat + 1):
    fitted = 0
    if number_of_tt_executions > 1:
        print(f"TimTrack iteration {iteration+1} out of {number_of_tt_executions}")
    
    n_rows = len(working_df)
    charge_arr = np.zeros((n_rows, 4), dtype=float)
    res_ystr_arr = np.zeros((n_rows, 4), dtype=float)
    res_tsum_arr = np.zeros((n_rows, 4), dtype=float)
    res_tdif_arr = np.zeros((n_rows, 4), dtype=float)
    ext_res_ystr_arr = np.zeros((n_rows, 4), dtype=float)
    ext_res_tsum_arr = np.zeros((n_rows, 4), dtype=float)
    ext_res_tdif_arr = np.zeros((n_rows, 4), dtype=float)

    charge_event_arr = np.zeros(n_rows, dtype=float)
    iterations_arr = np.zeros(n_rows, dtype=np.int32)
    conv_distance_arr = np.zeros(n_rows, dtype=float)
    converged_arr = np.zeros(n_rows, dtype=np.int8)
    if "list_tt" in working_df.columns:
        processed_tt_arr = pd.to_numeric(
            working_df["list_tt"],
            errors="coerce",
        ).fillna(0).to_numpy(dtype=np.int32)
    else:
        processed_tt_arr = np.zeros(n_rows, dtype=np.int32)

    th_chi_arr = np.zeros(n_rows, dtype=float)
    x_arr = np.zeros(n_rows, dtype=float)
    xp_arr = np.zeros(n_rows, dtype=float)
    y_arr = np.zeros(n_rows, dtype=float)
    yp_arr = np.zeros(n_rows, dtype=float)
    t0_arr = np.zeros(n_rows, dtype=float)
    s_arr = np.zeros(n_rows, dtype=float)

    th_chi_ndf_arrays = {}

    iterator = range(n_rows)
    if not crontab_execution:
        iterator = tqdm(iterator, total=n_rows, desc="Processing events")

    _acc_fn = _accumulate_mk_va
    _build_base_fn = _build_mk_va_base
    _acc_g1_fn = _accumulate_mk_va_dynamic_g1
    _solve_fn = solve_and_covdiag
    _solve_only_fn = solve_only
    _fmahd_fn = fmahd
    _loo_build_single_fn = _build_loo_single_step_system
    _w1 = float(_w_arr[1])
    _iter_max_f = float(iter_max)
    _cocut_f = float(cocut)
    _tt_loo_iter_max_f = float(tt_loo_iter_max)
    _tt_loo_cocut_f = float(tt_loo_cocut)
    _tt_loo_mode_is_single_step = (tt_loo_mode == "single_step")

    _mk_buf = np.zeros([npar, npar])
    _va_buf = np.zeros(npar)
    _mk_base_buf = np.zeros([npar, npar])
    _va_base_buf = np.zeros(npar)
    _mk_loo_buf = np.zeros([npar, npar])
    _va_loo_buf = np.zeros(npar)

    for pos in iterator:
        _tt_events_total += 1
        _tt_t = time.perf_counter()

        # INTRODUCTION ------------------------------------------------------------------
        _q_row  = _tt_Q[pos]
        _ts_row = _tt_Tsum[pos]
        _td_row = _tt_Tdif[pos]
        _y_row  = _tt_Y[pos]
        _valid_plane_mask = (
            (_q_row != 0.0)
            & (_ts_row != 0.0)
            & (_td_row != 0.0)
            & (_y_row != 0.0)
            & np.isfinite(_q_row)
            & np.isfinite(_ts_row)
            & np.isfinite(_td_row)
            & np.isfinite(_y_row)
        )
        plane_idx_arr = np.flatnonzero(_valid_plane_mask)
        if plane_idx_arr.size > 0:
            charge_event = float(np.sum(_q_row[plane_idx_arr], dtype=float))
            plane_idx_4 = plane_idx_arr[plane_idx_arr < 4]
            if plane_idx_4.size > 0:
                charge_arr[pos, plane_idx_4] = _q_row[plane_idx_4]
        else:
            charge_event = 0.0
        _tt_intro_s += time.perf_counter() - _tt_t
        charge_event_arr[pos] = charge_event

        # FITTING -----------------------------------------------------------------------
        if plane_idx_arr.size <= 1:
            x_arr[pos] = np.nan
            xp_arr[pos] = np.nan
            y_arr[pos] = np.nan
            yp_arr[pos] = np.nan
            t0_arr[pos] = np.nan
            s_arr[pos] = np.nan
            conv_distance_arr[pos] = np.nan
            th_chi_arr[pos] = np.nan
            continue
        _tt_events_with_2plus_planes += 1
        _tt_t = time.perf_counter()

        if fixed_speed:
            vs = np.zeros(5, dtype=float)
        else:
            vs = np.zeros(6, dtype=float)
            vs[5] = sc

        if _tt_seed_enabled:
            sx = _seed_x[pos]
            sy = _seed_y[pos]
            stheta = _seed_theta[pos]
            sphi = _seed_phi[pos]
            st0 = _seed_t0[pos]
            if np.isfinite(sx) and np.isfinite(sy) and np.isfinite(stheta) and np.isfinite(sphi) and np.isfinite(st0):
                tan_theta = math.tan(stheta)
                xp_seed = tan_theta * math.cos(sphi)
                yp_seed = tan_theta * math.sin(sphi)
                if (
                    np.isfinite(xp_seed)
                    and np.isfinite(yp_seed)
                    and abs(xp_seed) <= _tt_seed_slope_abs_max
                    and abs(yp_seed) <= _tt_seed_slope_abs_max
                ):
                    vs[0] = sx
                    vs[1] = xp_seed
                    vs[2] = sy
                    vs[3] = yp_seed
                    vs[4] = st0
                    if not fixed_speed:
                        ss_seed = _seed_s[pos]
                        if np.isfinite(ss_seed) and ss_seed > 0:
                            vs[5] = ss_seed

        mk = _mk_buf
        va = _va_buf
        mk_base = _mk_base_buf
        va_base = _va_base_buf
        _build_base_fn(
            plane_idx_arr,
            _y_row,
            _td_row,
            _z_pos_arr,
            _w_arr,
            ss,
            mk_base,
            va_base,
        )
        istp = 0
        dist = d0
        while dist > _cocut_f and istp < _iter_max_f:
            mk[:, :] = mk_base
            va[:] = va_base
            _acc_g1_fn(
                vs,
                plane_idx_arr,
                _ts_row,
                _z_pos_arr,
                _w1,
                _half_lenx_ss,
                _sc_val,
                fixed_speed,
                mk,
                va,
            )
            istp += 1
            vs0 = vs
            vs, merr_diag = _solve_fn(mk, va)
            dist = _fmahd_fn(npar, vs, vs0, merr_diag)
        _tt_mainfit_iterations_total += int(istp)
        _tt_main_fit_s += time.perf_counter() - _tt_t

        if istp >= _iter_max_f or dist >= _cocut_f:
            converged_arr[pos] = 1
        iterations_arr[pos] = istp
        conv_distance_arr[pos] = dist

        vsf = vs
        fitted += 1

        # RESIDUAL ANALYSIS -------------------------------------------------------------
        res_ystr = res_tsum = res_tdif = ndat = 0
        _tt_t = time.perf_counter()
        X0 = vsf[0]
        XP = vsf[1]
        Y0 = vsf[2]
        YP = vsf[3]
        T0 = vsf[4]
        S0 = _sc_val if fixed_speed else vsf[5]
        kz = math.sqrt(1.0 + XP * XP + YP * YP)
        ts_base = T0 + _half_lenx_ss
        for plane_idx in plane_idx_arr:
            ndat += nvar
            zi = _z_pos_arr[plane_idx]
            xfit = X0 + XP * zi
            yr = (Y0 + YP * zi) - _y_row[plane_idx]
            tsr = (ts_base + S0 * kz * zi) - _ts_row[plane_idx]
            tdr = (ss * xfit) - _td_row[plane_idx]

            res_ystr += yr
            res_tsum += tsr
            res_tdif += tdr

            if plane_idx < 4:
                res_ystr_arr[pos, plane_idx] = yr
                res_tsum_arr[pos, plane_idx] = tsr
                res_tdif_arr[pos, plane_idx] = tdr

        ndf = ndat - npar
        chi2 = (res_ystr / anc_sy) ** 2 + (res_tsum / anc_sts) ** 2 + (res_tdif / anc_std) ** 2
        th_chi_arr[pos] = chi2
        th_chi_ndf_arrays.setdefault(ndf, np.zeros(n_rows, dtype=float))[pos] = chi2

        x_arr[pos] = vsf[0]
        xp_arr[pos] = vsf[1]
        y_arr[pos] = vsf[2]
        yp_arr[pos] = vsf[3]
        t0_arr[pos] = vsf[4]
        if fixed_speed:
            s_arr[pos] = sc
        else:
            s_arr[pos] = vsf[5]
        _tt_residual_s += time.perf_counter() - _tt_t

        # ---------------------------------------------------------------------------------------------
        # Residual analysis with 4-plane tracks (hide a plane and fit the 3 remaining planes)
        # ---------------------------------------------------------------------------------------------
        if plane_idx_arr.size >= 3 and res_ana_removing_planes:
            _tt_events_loo_eligible += 1
            if (pos % tt_loo_every_n) == 0:
                _tt_events_loo_processed += 1
                _tt_t = time.perf_counter()
                mk_loo = _mk_loo_buf
                va_loo = _va_loo_buf

                if _tt_loo_mode_is_single_step:
                    # --- Single-step LOO: one Newton-Raphson step from vsf ---
                    for plane_idx_ref in plane_idx_arr:
                        _loo_build_single_fn(
                            vsf,
                            plane_idx_arr,
                            plane_idx_ref,
                            _y_row,
                            _ts_row,
                            _td_row,
                            _z_pos_arr,
                            _w_arr,
                            ss,
                            _half_lenx_ss,
                            _sc_val,
                            fixed_speed,
                            mk_loo,
                            va_loo,
                        )
                        vs_loo = _solve_only_fn(mk_loo, va_loo)
                        _tt_loo_refits_total += 1
                        _tt_loo_iterations_total += 1
                        y_res = vs_loo[2] - _y_row[plane_idx_ref]
                        ts_res = (vs_loo[4] + _half_lenx_ss) - _ts_row[plane_idx_ref]
                        td_res = (vs_loo[0] * ss) - _td_row[plane_idx_ref]

                        if plane_idx_ref < 4:
                            ext_res_ystr_arr[pos, plane_idx_ref] = y_res
                            ext_res_tsum_arr[pos, plane_idx_ref] = ts_res
                            ext_res_tdif_arr[pos, plane_idx_ref] = td_res
                else:
                    # --- Iterative LOO (legacy convergence loop) ---
                    for plane_idx_ref in plane_idx_arr:
                        z_ref = _z_pos_arr[plane_idx_ref]
                        y_ref = _y_row[plane_idx_ref]
                        ts_ref = _ts_row[plane_idx_ref]
                        td_ref = _td_row[plane_idx_ref]

                        vs = vsf
                        istp_loo = 0
                        dist_loo = d0
                        while dist_loo > _tt_loo_cocut_f and istp_loo < _tt_loo_iter_max_f:
                            mk_loo.fill(0.0)
                            va_loo.fill(0.0)
                            for plane_idx in plane_idx_arr:
                                if plane_idx == plane_idx_ref:
                                    continue
                                zi = _z_pos_arr[plane_idx] - z_ref
                                _acc_fn(
                                    npar,
                                    vs,
                                    _y_row[plane_idx],
                                    _ts_row[plane_idx],
                                    _td_row[plane_idx],
                                    zi,
                                    _w_arr,
                                    ss,
                                    lenx,
                                    _sc_val,
                                    fixed_speed,
                                    mk_loo,
                                    va_loo,
                                )
                            istp_loo += 1
                            vs0 = vs
                            vs, merr_diag = _solve_fn(mk_loo, va_loo)
                            dist_loo = _fmahd_fn(npar, vs, vs0, merr_diag)

                        _tt_loo_refits_total += 1
                        _tt_loo_iterations_total += int(istp_loo)
                        y_res = vs[2] - y_ref
                        ts_res = (vs[4] + _half_lenx_ss) - ts_ref
                        td_res = (vs[0] * ss) - td_ref

                        if plane_idx_ref < 4:
                            ext_res_ystr_arr[pos, plane_idx_ref] = y_res
                            ext_res_tsum_arr[pos, plane_idx_ref] = ts_res
                            ext_res_tdif_arr[pos, plane_idx_ref] = td_res
                _tt_residual_loo_s += time.perf_counter() - _tt_t
    
    # Push the accumulated results back to the DataFrame in a single shot ------
    _tt_t = time.perf_counter()
    for plane_idx in range(4):
        col_suffix = plane_idx + 1
        working_df[f'tim_charge_{col_suffix}'] = charge_arr[:, plane_idx]
        working_df[f'tim_res_ystr_{col_suffix}'] = res_ystr_arr[:, plane_idx]
        working_df[f'tim_res_tsum_{col_suffix}'] = res_tsum_arr[:, plane_idx]
        working_df[f'tim_res_tdif_{col_suffix}'] = res_tdif_arr[:, plane_idx]
        working_df[f'tim_ext_res_ystr_{col_suffix}'] = ext_res_ystr_arr[:, plane_idx]
        working_df[f'tim_ext_res_tsum_{col_suffix}'] = ext_res_tsum_arr[:, plane_idx]
        working_df[f'tim_ext_res_tdif_{col_suffix}'] = ext_res_tdif_arr[:, plane_idx]

    working_df['tim_charge_event'] = charge_event_arr
    working_df['tim_iterations'] = iterations_arr
    working_df['tim_conv_distance'] = conv_distance_arr
    working_df['tim_converged'] = converged_arr
    working_df['tim_list_tt'] = processed_tt_arr

    working_df['tim_th_chi'] = th_chi_arr
    working_df['tim_x'] = x_arr
    working_df['tim_xp'] = xp_arr
    working_df['tim_y'] = y_arr
    working_df['tim_yp'] = yp_arr
    working_df['tim_t0'] = t0_arr
    working_df['tim_s'] = s_arr
    working_df[['tim_res_y', 'tim_res_ts', 'tim_res_td']] = 0.0

    possible_ndf = {nvar * planes - npar for planes in range(2, nplan + 1)}
    possible_ndf = {ndf for ndf in possible_ndf if ndf >= 0}
    for ndf in possible_ndf:
        working_df[f'th_chi_{ndf}'] = th_chi_ndf_arrays.get(ndf, np.zeros(n_rows, dtype=float))
    _tt_writeback_s += time.perf_counter() - _tt_t
    
    # # Filter according to residual ------------------------------------------------
    # plane_cols = range(1, 5)
    # res_tsum_abs = np.abs(working_df[[f'res_tsum_{i}' for i in plane_cols]].to_numpy())
    # res_tdif_abs = np.abs(working_df[[f'res_tdif_{i}' for i in plane_cols]].to_numpy())
    # res_ystr_abs = np.abs(working_df[[f'res_ystr_{i}' for i in plane_cols]].to_numpy())
    # ext_res_tsum_abs = np.abs(working_df[[f'ext_res_tsum_{i}' for i in plane_cols]].to_numpy())
    # ext_res_tdif_abs = np.abs(working_df[[f'ext_res_tdif_{i}' for i in plane_cols]].to_numpy())
    # ext_res_ystr_abs = np.abs(working_df[[f'ext_res_ystr_{i}' for i in plane_cols]].to_numpy())
    #
    # plane_rejected = (
    #     (res_tsum_abs > res_tsum_filter) |
    #     (res_tdif_abs > res_tdif_filter) |
    #     (res_ystr_abs > res_ystr_filter) |
    #     (ext_res_tsum_abs > ext_res_tsum_filter) |
    #     (ext_res_tdif_abs > ext_res_tdif_filter) |
    #     (ext_res_ystr_abs > ext_res_ystr_filter)
    # )
    # plane_rejected_df = pd.DataFrame(plane_rejected, index=working_df.index, columns=list(plane_cols))
    #
    # changed_event_mask = plane_rejected_df.any(axis=1)
    # changed_event_count = int(changed_event_mask.sum())
    #
    # for plane_idx in plane_cols:
    #     mask = plane_rejected_df[plane_idx]
    #     if mask.any():
    #         cols_to_zero = [
    #             f'P{plane_idx}_Y_final',
    #             f'P{plane_idx}_T_sum_final',
    #             f'P{plane_idx}_T_dif_final',
    #             f'P{plane_idx}_Q_sum_final',
    #             f'P{plane_idx}_Q_dif_final',
    #             f'res_ystr_{plane_idx}',
    #             f'res_tsum_{plane_idx}',
    #             f'res_tdif_{plane_idx}',
    #             f'ext_res_ystr_{plane_idx}',
    #             f'ext_res_tsum_{plane_idx}',
    #             f'ext_res_tdif_{plane_idx}',
    #             f'det_res_ystr_{plane_idx}',
    #             f'det_res_tsum_{plane_idx}',
    #             f'det_res_tdif_{plane_idx}',
    #         ]
    #         existing = [c for c in cols_to_zero if c in working_df.columns]
    #         working_df.loc[mask, existing] = 0
    #
    # print(f"--> {changed_event_count} events were residual filtered.")
    # record_filter_metric(
    #     "residual_zeroed_event_pct",
    #     changed_event_count,
    #     len(working_df),
    # )
    # 
    # print(f\"{len(working_df[working_df.iterations == iter_max])} reached the maximum number of iterations ({iter_max}).\")
    # print(f\"Percentage of events that did not converge: {len(working_df[working_df.iterations == iter_max]) / len(working_df) * 100:.2f}%\")
    # 
    # # --------------------------------------------------------------------------------
    # iteration += 1

#%%

#%%

# ------------------------------------------------------------------------------------
# End of TimTrack loop ---------------------------------------------------------------
# ------------------------------------------------------------------------------------

# Set the label to integer -----------------------------------------------------------
working_df["processed_tt"] = working_df["processed_tt"].astype(np.int32, copy=False)

# Calculate angles -------------------------------------------------------------------
def calculate_angles(xproj, yproj):
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    return theta, phi

theta_vals, phi_vals = calculate_angles(working_df["tim_xp"], working_df["tim_yp"])

# TimTrack-prefixed variables already exist; just ensure angles are set.
working_df["tim_theta"] = theta_vals
working_df["tim_phi"] = phi_vals

# Preserve slope columns for downstream compatibility
if "xp" not in working_df.columns:
    working_df["xp"] = working_df["tim_xp"]
if "yp" not in working_df.columns:
    working_df["yp"] = working_df["tim_yp"]

# Backward compatibility: expose tim_* results under legacy column names when missing
for p in range(1, 5):
    if f"res_ystr_{p}" not in working_df.columns:
        working_df[f"res_ystr_{p}"] = working_df.get(f"tim_res_ystr_{p}", 0)
    if f"res_tsum_{p}" not in working_df.columns:
        working_df[f"res_tsum_{p}"] = working_df.get(f"tim_res_tsum_{p}", 0)
    if f"res_tdif_{p}" not in working_df.columns:
        working_df[f"res_tdif_{p}"] = working_df.get(f"tim_res_tdif_{p}", 0)
    if f"ext_res_ystr_{p}" not in working_df.columns:
        working_df[f"ext_res_ystr_{p}"] = working_df.get(f"tim_ext_res_ystr_{p}", 0)
    if f"ext_res_tsum_{p}" not in working_df.columns:
        working_df[f"ext_res_tsum_{p}"] = working_df.get(f"tim_ext_res_tsum_{p}", 0)
    if f"ext_res_tdif_{p}" not in working_df.columns:
        working_df[f"ext_res_tdif_{p}"] = working_df.get(f"tim_ext_res_tdif_{p}", 0)
    if f"charge_{p}" not in working_df.columns:
        working_df[f"charge_{p}"] = working_df.get(f"tim_charge_{p}", 0)

if "charge_event" not in working_df.columns:
    working_df["charge_event"] = working_df.get("tim_charge_event", 0)
if "iterations" not in working_df.columns:
    working_df["iterations"] = working_df.get("tim_iterations", 0)
if "conv_distance" not in working_df.columns:
    working_df["conv_distance"] = working_df.get("tim_conv_distance", 0)
if "converged" not in working_df.columns:
    working_df["converged"] = working_df.get("tim_converged", 0)

for p in range(1, 5):
    if f"tim_res_ystr_{p}" not in working_df.columns:
        working_df[f"tim_res_ystr_{p}"] = working_df.get(f"res_ystr_{p}", 0)
    if f"tim_res_tsum_{p}" not in working_df.columns:
        working_df[f"tim_res_tsum_{p}"] = working_df.get(f"res_tsum_{p}", 0)
    if f"tim_res_tdif_{p}" not in working_df.columns:
        working_df[f"tim_res_tdif_{p}"] = working_df.get(f"res_tdif_{p}", 0)
    if f"tim_ext_res_ystr_{p}" not in working_df.columns:
        working_df[f"tim_ext_res_ystr_{p}"] = working_df.get(f"ext_res_ystr_{p}", 0)
    if f"tim_ext_res_tsum_{p}" not in working_df.columns:
        working_df[f"tim_ext_res_tsum_{p}"] = working_df.get(f"ext_res_tsum_{p}", 0)
    if f"tim_ext_res_tdif_{p}" not in working_df.columns:
        working_df[f"tim_ext_res_tdif_{p}"] = working_df.get(f"ext_res_tdif_{p}", 0)

#%%

if create_plots and "processed_tt" in working_df.columns and "datetime" in working_df.columns:
    print("In")
    for combo in TRACK_COMBINATIONS:
        try:
            combo_int = int(combo)
        except ValueError:
            continue
        subset = working_df[working_df["processed_tt"] == combo_int]
        if subset.empty:
            continue
        plot_ts_with_side_hist(
            subset,
            ["tim_x", "tim_y", "tim_theta", "tim_phi", "tim_s", "tim_t0"],
            "datetime",
            f"timtrack_timeseries_combo_{combo}",
        )
        plot_residuals_ts_hist(
            subset,
            prefixes=[
                ("ystr", "tim_res_ystr_", "tim_ext_res_ystr_"),
                ("tsum", "tim_res_tsum_", "tim_ext_res_tsum_"),
                ("tdif", "tim_res_tdif_", "tim_ext_res_tdif_"),
            ],
            time_col="datetime",
            title=f"timtrack_residuals_combo_{combo}",
        )

#%%

# Combine detached and TimTrack estimates ------------------------------------
combined_core_vars = ["x", "y", "theta", "phi", "s", "t0"]
combined_columns = {}
for base in combined_core_vars:
    det_col = f"det_{base}"
    tim_col = f"tim_{base}"
    det_vals = working_df[det_col].to_numpy(copy=False) if det_col in working_df else np.zeros(len(working_df))
    tim_vals = working_df[tim_col].to_numpy(copy=False) if tim_col in working_df else np.zeros(len(working_df))
    if base == "phi":
        # Handle angular wrap: diff in [-pi, pi], average with wrap-aware mid-point
        diff = np.angle(np.exp(1j * (det_vals - tim_vals)))
        avg = tim_vals + diff / 2.0
        avg = np.angle(np.exp(1j * avg))  # wrap back to [-pi, pi]
        combined_columns[base] = avg
        combined_columns[f"{base}_err"] = diff / 2.0
    else:
        combined_columns[base] = 0.5 * (det_vals + tim_vals)
        combined_columns[f"{base}_err"] = 0.5 * (det_vals - tim_vals)

residual_sets = [
    ("res_ystr", "det_res_ystr_", "tim_res_ystr_"),
    ("res_tsum", "det_res_tsum_", "tim_res_tsum_"),
    ("res_tdif", "det_res_tdif_", "tim_res_tdif_"),
    ("ext_res_ystr", "det_ext_res_ystr_", "tim_ext_res_ystr_"),
    ("ext_res_tsum", "det_ext_res_tsum_", "tim_ext_res_tsum_"),
    ("ext_res_tdif", "det_ext_res_tdif_", "tim_ext_res_tdif_"),]
for base, det_prefix, tim_prefix in residual_sets:
    for p in range(1, 5):
        det_col = f"{det_prefix}{p}"
        tim_col = f"{tim_prefix}{p}"
        det_vals = working_df[det_col].to_numpy(copy=False) if det_col in working_df else np.zeros(len(working_df))
        tim_vals = working_df[tim_col].to_numpy(copy=False) if tim_col in working_df else np.zeros(len(working_df))
        combined_columns[f"{base}_{p}"] = 0.5 * (det_vals + tim_vals)
        combined_columns[f"{base}_{p}_err"] = 0.5 * (det_vals - tim_vals)

if combined_columns:
    combined_df = pd.DataFrame(combined_columns, index=working_df.index)
    overlap_cols = [col for col in combined_df.columns if col in working_df.columns]
    if overlap_cols:
        working_df.loc[:, overlap_cols] = combined_df[overlap_cols].to_numpy()
    new_cols = [col for col in combined_df.columns if col not in working_df.columns]
    if new_cols:
        working_df = pd.concat([working_df, combined_df[new_cols]], axis=1)

# Defrag after all combined-column writes are complete
working_df = working_df.copy()

if create_debug_plots:
    def _emit_param_debug(param_label, columns, thresholds, *, tag="tunable"):
        cols_present = [col for col in columns if col in working_df.columns]
        if not cols_present:
            return
        debug_thresholds = {col: thresholds for col in cols_present}
        title = f"Task 4 pre-filter: {param_label} [{tag}] (station {station})"
        cols_present.sort()
        global debug_fig_idx
        debug_fig_idx = plot_debug_histograms(
            working_df,
            cols_present,
            debug_thresholds,
            title=title,
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

    def _plane_cols(prefix):
        return [f"{prefix}{p}" for p in range(1, 5)]

    # Core kinematic filters
    _emit_param_debug(
        "det_pos_filter",
        ["det_x", "det_y", "x", "y"],
        [-det_pos_filter, det_pos_filter],
    )
    _emit_param_debug(
        "proj_filter",
        ["xp", "yp"],
        [-proj_filter, proj_filter],
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

    # Residual filters (timtrack + detached + combined)
    _emit_param_debug(
        "res_ystr_filter",
        _plane_cols("res_ystr_"),
        [-res_ystr_filter, res_ystr_filter],
    )
    _emit_param_debug(
        "res_tsum_filter",
        _plane_cols("res_tsum_"),
        [-res_tsum_filter, res_tsum_filter],
    )
    _emit_param_debug(
        "res_tdif_filter",
        _plane_cols("res_tdif_"),
        [-res_tdif_filter, res_tdif_filter],
    )
    _emit_param_debug(
        "ext_res_ystr_filter",
        _plane_cols("ext_res_ystr_"),
        [-ext_res_ystr_filter, ext_res_ystr_filter],
    )
    _emit_param_debug(
        "ext_res_tsum_filter",
        _plane_cols("ext_res_tsum_"),
        [-ext_res_tsum_filter, ext_res_tsum_filter],
    )
    _emit_param_debug(
        "ext_res_tdif_filter",
        _plane_cols("ext_res_tdif_"),
        [-ext_res_tdif_filter, ext_res_tdif_filter],
    )
    _emit_param_debug(
        "det_res_ystr_filter",
        _plane_cols("det_res_ystr_"),
        [-det_res_ystr_filter, det_res_ystr_filter],
    )
    _emit_param_debug(
        "det_res_tsum_filter",
        _plane_cols("det_res_tsum_"),
        [-det_res_tsum_filter, det_res_tsum_filter],
    )
    _emit_param_debug(
        "det_res_tdif_filter",
        _plane_cols("det_res_tdif_"),
        [-det_res_tdif_filter, det_res_tdif_filter],
    )
    _emit_param_debug(
        "det_ext_res_ystr_filter",
        _plane_cols("det_ext_res_ystr_"),
        [-det_ext_res_ystr_filter, det_ext_res_ystr_filter],
    )
    _emit_param_debug(
        "det_ext_res_tsum_filter",
        _plane_cols("det_ext_res_tsum_"),
        [-det_ext_res_tsum_filter, det_ext_res_tsum_filter],
    )
    _emit_param_debug(
        "det_ext_res_tdif_filter",
        _plane_cols("det_ext_res_tdif_"),
        [-det_ext_res_tdif_filter, det_ext_res_tdif_filter],
    )

# print("----------------------------------------------------------------------")
# print("-------------------------- New definitions ---------------------------")
# print("----------------------------------------------------------------------")

# Derive definitive_tt before any plotting/filters that rely on it.
def compute_definitive_tt(df_input: pd.DataFrame) -> pd.Series:
    """Build definitive_tt from planes with all final reconstructed components."""
    tt_str = pd.Series("", index=df_input.index, dtype="object")
    for plane in range(1, 5):
        plane_columns = [
            f"P{plane}_Y_final",
            f"P{plane}_T_sum_final",
            f"P{plane}_T_dif_final",
            f"P{plane}_Q_sum_final",
            f"P{plane}_Q_dif_final",
        ]
        if not all(col in df_input.columns for col in plane_columns):
            continue
        valid_plane = df_input.loc[:, plane_columns].ne(0).all(axis=1)
        tt_str = tt_str.where(~valid_plane, tt_str + str(plane))
    return tt_str.replace("", "0").astype(int)

working_df["definitive_tt"] = compute_definitive_tt(working_df)

if create_plots and "processed_tt" in working_df.columns and "datetime" in working_df.columns:
    for combo in TRACK_COMBINATIONS:
        try:
            combo_int = int(combo)
        except ValueError:
            continue
        subset = working_df[working_df["processed_tt"] == combo_int]
        if subset.empty:
            continue
        plot_ts_err_with_hist(
            subset,
            ["x", "y", "theta", "phi", "s", "t0"],
            "datetime",
            title=f"combined_timeseries_combo_{combo}",
        )
        plot_residuals_ts_hist(
            subset,
            prefixes=[
                ("ystr", "res_ystr_", "ext_res_ystr_"),
                ("tsum", "res_tsum_", "ext_res_tsum_"),
                ("tdif", "res_tdif_", "ext_res_tdif_"),
            ],
            time_col="datetime",
            title=f"combined_residuals_combo_{combo}",
        )

#%%

# Time series and fittings

timeseries_and_fits = bool(config.get("timeseries_and_fits", False))
if timeseries_and_fits:

    # Time series of core track variables (averaged and errors) ------------------
    if 'datetime' in working_df.columns:
        
        ts_core = working_df.copy()
        ts_core['datetime'] = pd.to_datetime(ts_core['datetime'])
        ts_core = ts_core.sort_values('datetime')

        # Combined error histograms across combinations (one figure) --------------------
        err_vars = ['x_err', 'y_err', 'theta_err', 'phi_err', 's_err', 't0_err']
        combo_subsets = []
        for combo in TRACK_COMBINATIONS:
            try:
                combo_int = int(combo)
            except ValueError:
                continue
            subset = ts_core[ts_core['processed_tt'] == combo_int]
            if subset.empty:
                continue
            combo_subsets.append((combo, subset))
        del ts_core  # free the sorted full-copy; subsets in combo_subsets are independent

        if combo_subsets:
            # Compute global quantile bounds per variable across all combos
            bounds_dict = {}
            for var in err_vars:
                lows = []
                highs = []
                for _, sub in combo_subsets:
                    if var not in sub.columns:
                        continue
                    data = sub[var].dropna()
                    if data.empty:
                        continue
                    q_low, q_high = data.quantile([0.0001, 0.99999])
                    lows.append(q_low)
                    highs.append(q_high)
                if lows and highs:
                    bounds_dict[var] = (min(lows), max(highs))
                else:
                    bounds_dict[var] = (0, 1)

            n_rows = len(combo_subsets)
            n_cols = len(err_vars)

            # Fit a Gaussian per combination/variable and store stats (always, even if plots are off)
            def _gauss(x, amp, mu, sigma):
                return amp * norm.pdf(x, mu, sigma)

            fit_results: dict[tuple[int, str], tuple[float, float, float]] = {}

            for combo, sub in combo_subsets:
                try:
                    combo_int = int(combo)
                except ValueError:
                    continue
                for var in err_vars:
                    if var not in sub.columns:
                        continue
                    series = sub[var].dropna()
                    if series.empty:
                        continue
                    q25, q75 = series.quantile([0.25, 0.75])
                    global_variables[f"{var}_{combo_int}_q25"] = float(q25)
                    global_variables[f"{var}_{combo_int}_q75"] = float(q75)
                    mirror_for_half = var == "theta_err"
                    series_fit = pd.concat([series, -series]) if mirror_for_half else series
                    q_low, q_high = series_fit.quantile([0.0001, 0.99999])
                    bin_edges = np.linspace(q_low, q_high, 161)
                    bin_width = bin_edges[1] - bin_edges[0]
                    counts, edges = np.histogram(series_fit, bins=bin_edges)
                    if not np.any(counts):
                        continue

                    # Suppress central spike: drop bins far above the typical height
                    nonzero = counts[counts > 0]
                    typical = np.median(nonzero) if len(nonzero) else 0
                    mask = counts > 0
                    if typical > 0:
                        mask &= counts < typical * 6  # keep more populated bins as well
                    centers = 0.5 * (edges[1:] + edges[:-1])
                    x_fit = centers[mask]
                    y_fit = counts[mask]
                    if y_fit.size < 5:
                        continue
                    try:
                        p0 = [y_fit.max(), float(series.mean()), float(max(series.std(), 1e-6))]
                        weights = np.maximum(y_fit, 1)
                        popt, _ = curve_fit(
                            _gauss,
                            x_fit,
                            y_fit,
                            p0=p0,
                            sigma=1.0 / weights,  # heavier weight for populous bins
                            absolute_sigma=False,
                            maxfev=4000,
                        )
                    except Exception:
                        continue
                    amp, mu, sigma = popt
                    if mirror_for_half:
                        amp *= 0.5  # adjust amplitude back to one-sided scale
                    fit_results[(combo_int, var)] = (float(amp), float(mu), float(sigma))
                    global_variables[f"{var}_{combo_int}_gauss1_amp"] = float(amp)
                    global_variables[f"{var}_{combo_int}_gauss1_mu"] = float(mu)
                    global_variables[f"{var}_{combo_int}_gauss1_sigma"] = float(sigma)

            # Only plot if requested
            if create_plots or create_essential_plots:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex='col')
                if n_rows == 1:
                    axes = np.array([axes])
                for r, (combo, sub) in enumerate(combo_subsets):
                    for c, var in enumerate(err_vars):
                        ax = axes[r, c]
                        if var not in sub.columns:
                            ax.set_visible(False)
                            continue
                        data = sub[var].dropna()
                        if data.empty:
                            ax.set_visible(False)
                            continue
                        q_low, q_high = bounds_dict.get(var, (data.min(), data.max()))
                        bin_edges = np.linspace(q_low, q_high, 161)
                        ax.hist(data, bins=bin_edges, color='C0', alpha=0.7)
                        ax.set_yscale('log')
                        ax.set_xlim(q_low, q_high)
                        if r == 0:
                            ax.set_title(var)
                        if c == 0:
                            ax.set_ylabel(f'Comb {combo}')
                        ax.grid(True, alpha=0.3)
                plt.suptitle('Error distributions per combination (data)', fontsize=12)
                plt.tight_layout(rect=[0, 0, 1, 0.94])
                if save_plots:
                    final_filename = f'{fig_idx}_hist_core_errs_combined.png'
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, format='png')
                if show_plots:
                    plt.show()
                plt.close()

                # Redraw with overlays
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex='col')
                if n_rows == 1:
                    axes = np.array([axes])
                for r, (combo, sub) in enumerate(combo_subsets):
                    for c, var in enumerate(err_vars):
                        ax = axes[r, c]
                        if var not in sub.columns:
                            ax.set_visible(False)
                            continue
                        data = sub[var].dropna()
                        if data.empty:
                            ax.set_visible(False)
                            continue
                        q_low, q_high = bounds_dict.get(var, (data.min(), data.max()))
                        bin_edges = np.linspace(q_low, q_high, 161)
                        counts, edges, _ = ax.hist(data, bins=bin_edges, color='C0', alpha=0.6, label='data')
                        ax.set_yscale('log')
                        ax.set_xlim(q_low, q_high)
                        y_min, y_max = ax.get_ylim()

                        try:
                            combo_int = int(combo)
                        except ValueError:
                            combo_int = None
                        if combo_int is not None:
                            params = fit_results.get((combo_int, var))
                            if params:
                                amp, mu, sigma = params
                                x_grid = np.linspace(q_low, q_high, 400)
                                g_vals = _gauss(x_grid, amp, mu, sigma)
                                ax.plot(x_grid, g_vals, 'r-', lw=1.0, label='fit')
                        if r == 0:
                            ax.set_title(var)
                        if c == 0:
                            ax.set_ylabel(f'Comb {combo}')
                        ax.set_ylim(y_min, y_max)
                        ax.grid(True, alpha=0.3)
                plt.suptitle('Error distributions per combination (fits)', fontsize=12)
                plt.tight_layout(rect=[0, 0, 1, 0.94])
                if save_plots:
                    final_filename = f'{fig_idx}_hist_core_errs_combined_with_fits.png'
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    save_plot_figure(save_fig_path, format='png')
                if show_plots:
                    plt.show()
                plt.close()
#print("DEBUG EXITING")
#sys.exit()
# print("----------------------------------------------------------------------")
# print("----------------------- Timtrack results filter ----------------------")
# print("----------------------------------------------------------------------")

# for col in working_df.columns:
#     # TimTrack results
#     if 't0' == col:
#         working_df.loc[:, col] = np.where((working_df[col] > t0_right_filter) | (working_df[col] < t0_left_filter), 0, working_df[col])
#     if 'x' == col or 'y' == col:
#         cond_bound = (working_df[col] > pos_filter) | (working_df[col] < -1*pos_filter)
#         cond_zero = (working_df[col] == 0)
#         working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
#     if 'xp' == col or 'yp' == col:
#         cond_bound = (working_df[col] > proj_filter) | (working_df[col] < -1*proj_filter)
#         cond_zero = (working_df[col] == 0)
#         working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
#     if 's' == col:
#         cond_bound = (working_df[col] > slowness_filter_right) | (working_df[col] < slowness_filter_left)
#         cond_zero = (working_df[col] == 0)
#         working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
#     if 'theta' == col:
#         cond_bound = (working_df[col] > theta_right_filter) | (working_df[col] < theta_left_filter)
#         cond_zero = (working_df[col] == 0)
#         working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
#     if 'phi' == col:
#         cond_bound = (working_df[col] > phi_right_filter) | (working_df[col] < phi_left_filter)
#         cond_zero = (working_df[col] == 0)
#         working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])

# print("----------------------------------------------------------------------")
# print("------------------ TimTrack convergence comprobation -----------------")
# print("----------------------------------------------------------------------")

# if create_plots:
#     df_filtered = working_df.copy()
#     colors = plt.cm.tab10.colors
#     tt_values = [12, 23, 34, 13, 124, 134, 123, 234, 1234]
#     n_plots = len(tt_values)
#     ncols = 3
#     nrows = 3

#     fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
#     axes = axes.flatten()  # Flatten for easier indexing
    
#     for i, tt_val in enumerate(tt_values):
#         ax = axes[i]
        
#         df_tt = df_filtered[df_filtered['processed_tt'] == tt_val]
#         x = df_tt['iterations']
#         y = df_tt['conv_distance']
#         # ax.scatter(df_tt['s'], residuals, s=1, color='C0', alpha=0.5)
#         ax.scatter(x, y, s=2, color='C0', alpha=0.5)
#         ax.axvline(x=iter_max, color='r', linestyle='--', linewidth=1.5, label = "Iteration limit set")
#         ax.axhline(y=cocut, color='g', linestyle='--', linewidth=1.5, label = "Convergence cut set")
#         ax.set_title(f'TT {tt_val}', fontsize=10)
#         # ax.set_xlim(slowness_filter_left, slowness_filter_right)
#         ax.set_ylim(0, cocut * 1.05)
#         # ax.set_xlim(-1, 5)
#         # ax.set_ylim(-0.15, 0.15)
#         # ax.set_ylim(slowness_filter_left / 10, slowness_filter_right / 20)
#         ax.grid(True)
#         ax.legend()

#         if i % ncols == 0:
#             ax.set_ylabel(r'Iterations vs cocut')
#         if i // ncols == nrows - 1:
#             ax.set_xlabel(r'$Iterations$')
#     for j in range(i + 1, len(axes)):
#         axes[j].set_visible(False)
#     plt.suptitle(r'Iteration vs distance cut in convergence per processed_tt case', fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.tight_layout()
#     if save_plots:
#         filename = f'{fig_idx}_iterations_vs_cocut.png'
#         fig_idx += 1
#         save_fig_path = os.path.join(base_directories["figure_directory"], filename)
#         plot_list.append(save_fig_path)
#         save_plot_figure(save_fig_path, format='png')
#     if show_plots:
#         plt.show()
#     plt.close()

# print("----------------------------------------------------------------------")
# print("------------------ Slowness residual comprobation ---------------------")
# print("----------------------------------------------------------------------")

# working_df['delta_s'] = working_df['det_s'] - working_df['s']  # Calculate the difference from the speed of light

# if create_plots:
#     print("Plotting residuals of det_s - s for each original_tt to processed_tt case...")
    
#     df_filtered = working_df.copy()
#     bins = np.linspace(delta_s_left, delta_s_right, 100)  # Adjust range and bin size as needed
#     colors = plt.cm.tab10.colors

#     tt_values = [12, 23, 34, 13, 124, 134, 123, 234, 1234]
    
#     # Layout configuration
#     n_plots = len(tt_values)
#     ncols = 3
#     nrows = 3

#     fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
#     axes = axes.flatten()  # Flatten for easier indexing
    
#     for i, tt_val in enumerate(tt_values):
#         ax = axes[i]

#         df_tt = df_filtered[df_filtered['processed_tt'] == tt_val]
#         residuals = df_tt['delta_s']  # Calculate the residuals
#         # residuals = 2 * ( df_tt['det_s'] - df_tt['s'] ) / ( df_tt['det_s'] + df_tt['s'] )  # Calculate the residuals
#         # rel_sum = ( df_tt['det_s'] + df_tt['s'] ) / 2
#         rel_sum = df_tt['s']
        
#         if len(residuals) < 10:
#             ax.set_visible(False)
#             continue

#         # ax.scatter(df_tt['s'], residuals, s=1, color='C0', alpha=0.5)
#         ax.scatter(rel_sum, residuals, s=0.8, color='C0', alpha=0.1)
#         ax.axvline(x=sc, color='r', linestyle='--', linewidth=1.5, label = "$\\beta = 1$")  # Vertical line at x=0
#         ax.axvline(x=0, color='g', linestyle='--', linewidth=1.5, label = "Zero")  # Vertical line at x=0
#         ax.set_title(f'TT {tt_val}', fontsize=10)
#         ax.set_xlim(slowness_filter_left, slowness_filter_right)
#         # ax.set_ylim(-0.001, 0.001)
#         # ax.set_xlim(-1, 5)
#         # ax.set_ylim(-0.15, 0.15)
#         ax.set_ylim(delta_s_left, delta_s_right)
#         ax.grid(True)
#         ax.legend()

#         if i % ncols == 0:
#             ax.set_ylabel(r'$det_s - s$')
#         if i // ncols == nrows - 1:
#             ax.set_xlabel(r'$s$')

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         axes[j].set_visible(False)

#     plt.suptitle(r'Residuals: $det_s - s$ per processed_tt case', fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])

#     # Save or show the plot
#     plt.tight_layout()
#     if save_plots:
#         filename = f'{fig_idx}_residuals_det_s_minus_s_processed_tt.png'
#         fig_idx += 1
#         save_fig_path = os.path.join(base_directories["figure_directory"], filename)
#         plot_list.append(save_fig_path)
#         save_plot_figure(save_fig_path, format='png')
#     if show_plots:
#         plt.show()
#     plt.close()

# print("----------------------------------------------------------------------")
# print("--------------------- Comparison results filter ----------------------")
# print("----------------------------------------------------------------------")

# for col in working_df.columns:
#     # TimTrack results
#     if 'delta_s' == col:
#         working_df.loc[:, col] = np.where((working_df[col] > delta_s_right) | (working_df[col] < delta_s_left), 0, working_df[col])

# working_df['x'] = ( working_df['x'] + working_df['det_x'] ) / 2
# working_df['y'] = ( working_df['y'] + working_df['det_y'] ) / 2
# working_df['theta'] = ( working_df['theta'] + working_df['det_theta'] ) / 2
# working_df['phi'] = ( working_df['phi'] + working_df['det_phi'] ) / 2
# working_df['s'] = ( working_df['s'] + working_df['det_s'] ) / 2

# working_df['x_err'] = ( working_df['x'] - working_df['det_x'] ) / 2
# working_df['y_err'] = ( working_df['y'] - working_df['det_y'] ) / 2
# working_df['theta_err'] = ( working_df['theta'] - working_df['det_theta'] ) / 2
# phi_diff = working_df['phi'] - working_df['det_phi']
# # Keep the smallest angular separation considering 2π periodicity
# phi_err_abs = np.minimum.reduce([
#     np.abs(phi_diff),
#     np.abs(phi_diff + 2*np.pi),
#     np.abs(phi_diff - 2*np.pi)
# ])
# working_df['phi_err'] = phi_err_abs / 2
# working_df['s_err'] = ( working_df['s'] - working_df['det_s'] ) / 2

# Mean/error residuals between TimTrack and alternative methods ---------------
# for i in range(1, 5):
#     working_df[f'res_ystr_mean_{i}'] = (working_df.get(f'res_ystr_{i}', 0) + working_df.get(f'det_res_ystr_{i}', 0)) / 2
#     working_df[f'res_tsum_mean_{i}'] = (working_df.get(f'res_tsum_{i}', 0) + working_df.get(f'det_res_tsum_{i}', 0)) / 2
#     working_df[f'res_tdif_mean_{i}'] = (working_df.get(f'res_tdif_{i}', 0) + working_df.get(f'det_res_tdif_{i}', 0)) / 2

#     working_df[f'res_ystr_err_{i}'] = (working_df.get(f'res_ystr_{i}', 0) - working_df.get(f'det_res_ystr_{i}', 0)) / 2
#     working_df[f'res_tsum_err_{i}'] = (working_df.get(f'res_tsum_{i}', 0) - working_df.get(f'det_res_tsum_{i}', 0)) / 2
#     working_df[f'res_tdif_err_{i}'] = (working_df.get(f'res_tdif_{i}', 0) - working_df.get(f'det_res_tdif_{i}', 0)) / 2

#     working_df[f'ext_res_ystr_mean_{i}'] = (working_df.get(f'ext_res_ystr_{i}', 0) + working_df.get(f'det_ext_res_ystr_{i}', 0)) / 2
#     working_df[f'ext_res_tsum_mean_{i}'] = (working_df.get(f'ext_res_tsum_{i}', 0) + working_df.get(f'det_ext_res_tsum_{i}', 0)) / 2
#     working_df[f'ext_res_tdif_mean_{i}'] = (working_df.get(f'ext_res_tdif_{i}', 0) + working_df.get(f'det_ext_res_tdif_{i}', 0)) / 2

#     working_df[f'ext_res_ystr_err_{i}'] = (working_df.get(f'ext_res_ystr_{i}', 0) - working_df.get(f'det_ext_res_ystr_{i}', 0)) / 2
#     working_df[f'ext_res_tsum_err_{i}'] = (working_df.get(f'ext_res_tsum_{i}', 0) - working_df.get(f'det_ext_res_tsum_{i}', 0)) / 2
#     working_df[f'ext_res_tdif_err_{i}'] = (working_df.get(f'ext_res_tdif_{i}', 0) - working_df.get(f'det_ext_res_tdif_{i}', 0)) / 2

# working_df['chi_timtrack'] = working_df['th_chi']
# working_df['chi_alternative'] = working_df['det_th_chi']

_prof["s_timtrack_fitting_s"] = round(time.perf_counter() - _t_sec, 2)
_prof["s_tt_intro_s"] = round(_tt_intro_s, 2)
_prof["s_tt_main_fit_s"] = round(_tt_main_fit_s, 2)
_prof["s_tt_residual_s"] = round(_tt_residual_s, 2)
_prof["s_tt_residual_loo_s"] = round(_tt_residual_loo_s, 2)
_prof["s_tt_writeback_s"] = round(_tt_writeback_s, 2)
_prof["tt_events_total_n"] = int(_tt_events_total)
_prof["tt_events_with_2plus_planes_n"] = int(_tt_events_with_2plus_planes)
_prof["tt_events_loo_eligible_n"] = int(_tt_events_loo_eligible)
_prof["tt_events_loo_processed_n"] = int(_tt_events_loo_processed)
_prof["tt_mainfit_iterations_total_n"] = int(_tt_mainfit_iterations_total)
_prof["tt_loo_refits_total_n"] = int(_tt_loo_refits_total)
_prof["tt_loo_iterations_total_n"] = int(_tt_loo_iterations_total)
_prof["tt_mainfit_iter_per_event"] = (
    float(_tt_mainfit_iterations_total) / float(_tt_events_with_2plus_planes)
    if _tt_events_with_2plus_planes > 0
    else np.nan
)
_prof["tt_loo_iter_per_refit"] = (
    float(_tt_loo_iterations_total) / float(_tt_loo_refits_total)
    if _tt_loo_refits_total > 0
    else np.nan
)
_prof["tt_loo_every_n"] = int(tt_loo_every_n)
_prof["tt_loo_iter_max"] = int(tt_loo_iter_max)
_prof["tt_loo_cocut"] = float(tt_loo_cocut)
_prof["tt_loo_mode"] = str(tt_loo_mode)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("-------------------- Real tracking trigger type ----------------------")
print("----------------------------------------------------------------------")

# Required constants supplied by the DAQ geometry
strip_half  = strip_length / 2.0         # x acceptance  : [-strip_half , +strip_half ]
width_half  = total_width / 2.0          # y acceptance  : [-width_half , +width_half ]
z_planes    = np.asarray(z_positions)    # shape (nplan,)

# Precompute averages of the two independent fits --------------------------
# New fitting track columns combining timtrack and the alternative method
x0_avg = pd.to_numeric(working_df["x"], errors="coerce").to_numpy(dtype=float)
y0_avg = pd.to_numeric(working_df["y"], errors="coerce").to_numpy(dtype=float)
theta_av = pd.to_numeric(working_df["theta"], errors="coerce").to_numpy(dtype=float)
phi_av = pd.to_numeric(working_df["phi"], errors="coerce").to_numpy(dtype=float)

vx = np.sin(theta_av) * np.cos(phi_av)                   # direction cosines
vy = np.sin(theta_av) * np.sin(phi_av)
vz = np.cos(theta_av)

tracking_vals = np.zeros(len(working_df), dtype=np.int32)

valid_track = (
    np.isfinite(x0_avg)
    & np.isfinite(y0_avg)
    & np.isfinite(theta_av)
    & np.isfinite(phi_av)
    & np.isfinite(vx)
    & np.isfinite(vy)
    & np.isfinite(vz)
    & (vz > 0.0)                                          # upward track => no planes
    & (x0_avg != 0.0)
    & (y0_avg != 0.0)
    & (theta_av != 0.0)
    & (phi_av != 0.0)
)

if np.any(valid_track):
    n_planes = int(len(z_planes))
    hits = np.zeros((len(working_df), n_planes), dtype=bool)
    for j, z_p in enumerate(z_planes):
        t = np.divide(
            z_p,
            vz,
            out=np.full(vz.shape, np.nan, dtype=float),
            where=vz != 0.0,
        )
        x_i = x0_avg + vx * t
        y_i = y0_avg + vy * t
        hits[:, j] = (
            np.isfinite(x_i)
            & np.isfinite(y_i)
            & (-strip_half <= x_i)
            & (x_i <= strip_half)
            & (-width_half <= y_i)
            & (y_i <= width_half)
        )
    hits &= valid_track[:, np.newaxis]

    if n_planes == 4:
        # Map 4-plane hit bitmask to concatenated plane code (e.g., 13, 234, 1234).
        bit_weights = np.array([1, 2, 4, 8], dtype=np.uint8)
        bitmask = (hits.astype(np.uint8) * bit_weights).sum(axis=1)
        bitmask_to_tt = np.array(
            [0, 1, 2, 12, 3, 13, 23, 123, 4, 14, 24, 124, 34, 134, 234, 1234],
            dtype=np.int32,
        )
        tracking_vals = bitmask_to_tt[bitmask]
    else:
        # Generic fallback for unexpected plane counts.
        for idx in np.flatnonzero(valid_track):
            planes_hit = [str(p + 1) for p in range(n_planes) if hits[idx, p]]
            if planes_hit:
                tracking_vals[idx] = int("".join(planes_hit))

tracking_df = pd.DataFrame({'tracking_tt': tracking_vals}, index=working_df.index)
working_df = working_df.drop(columns=tracking_df.columns.intersection(working_df.columns), errors='ignore')
working_df = working_df.join(tracking_df)
working_df = working_df.copy()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# The noise determination, if everything goes well ----------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

definitive_tt_values = [234, 123, 34, 1234, 23, 12, 124, 134, 24, 13, 14]
# Pre-seed metadata keys so CSVs always include all fit outputs, even if a
# specific combination has no data in this run.
for _tt in definitive_tt_values:
    global_variables.setdefault(f"sigmoid_width_{_tt}", np.nan)
    global_variables.setdefault(f"background_slope_{_tt}", np.nan)
    global_variables.setdefault(f"sigmoid_amplitude_{_tt}", np.nan)
    global_variables.setdefault(f"sigmoid_center_{_tt}", np.nan)
    global_variables.setdefault(f"fit_normalization_{_tt}", np.nan)

_prof["s_trigger_type_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
if time_window_fitting:

    print("---------------------------- Fitting loop ----------------------------")
    time_window_fitting_start = time.perf_counter()
    widths = np.linspace(
        0.0,
        2.0 * coincidence_window_cal_ns,
        coincidence_window_cal_number_of_points,
        dtype=float,
    )
    half_widths = 0.5 * widths
    width_chunk_size = 32
    definitive_tt_arr = pd.to_numeric(
        working_df["definitive_tt"],
        errors="coerce",
    ).fillna(0).to_numpy(dtype=np.int32)
    t_sum_cols = [col for col in working_df.columns if "_T_sum_" in col]
    t_sum_all = (
        working_df[t_sum_cols].to_numpy(dtype=float, copy=False)
        if t_sum_cols
        else np.empty((len(working_df), 0), dtype=float)
    )

    for definitive_tt in definitive_tt_values:
        # Create a mask for the current definitive_tt
        mask = definitive_tt_arr == definitive_tt
        n_selected = int(np.count_nonzero(mask))

        # Check if there are any rows in the filtered DataFrame
        if n_selected > 0:
            print(f"\nProcessing definitive_tt: {definitive_tt} with {n_selected} events.")
        t_sum_data = t_sum_all[mask]
        if t_sum_data.size == 0:
            print(f"\n[Warning] Skipping definitive_tt {definitive_tt}: no _T_sum_ columns.")
            continue
        
        nonzero_rows = np.any(t_sum_data != 0, axis=1)
        if not np.any(nonzero_rows):
            print(f"\n[Warning] Skipping definitive_tt {definitive_tt}: no non-zero T_sum data.")
            continue

        nonzero_mask = t_sum_data != 0
        nonzero_counts = nonzero_mask.sum(axis=1, keepdims=True)
        nonzero_sums = (t_sum_data * nonzero_mask).sum(axis=1, keepdims=True)
        row_stat = np.divide(
            nonzero_sums,
            nonzero_counts,
            out=np.zeros_like(nonzero_sums, dtype=float),
            where=nonzero_counts > 0,
        )
        abs_dev = np.abs(t_sum_data - row_stat)
        counts_per_width = np.empty(half_widths.shape[0], dtype=float)

        for start in range(0, half_widths.shape[0], width_chunk_size):
            stop = min(start + width_chunk_size, half_widths.shape[0])
            half_chunk = half_widths[start:stop]
            in_window = nonzero_mask[:, :, None] & (abs_dev[:, :, None] <= half_chunk[None, None, :])
            count_in_window = in_window.sum(axis=1)
            counts_per_width[start:stop] = count_in_window.mean(axis=0)
        
        # Safely compute a scalar denominator: use the maximum of counts_per_width if positive,
        # otherwise fall back to 1. This avoids passing a list mixing arrays and scalars to np.max,
        # which raises an error due to inhomogeneous shapes.
        
        if counts_per_width.size == 0:
            denom = 1.0
        else:
            denom = float(np.max(counts_per_width))
            if not np.isfinite(denom) or denom <= 0:
                denom = 1.0
        global_variables[f'fit_normalization_{definitive_tt}'] = denom
        counts_per_width_norm = counts_per_width / denom

        # # Define model function: signal (logistic) + linear background
        # def signal_plus_background(w, S, w0, tau, B):
        #     return S / (1 + np.exp(-(w - w0) / tau)) + B * w
        
        def signal_plus_background(w, S, w0, sigma, B):
            return 0.5 * S * (1 + erf((w - w0) / (np.sqrt(2) * sigma))) + B * w

        p0 = [1.0, 1.0, 0.5, 0.02]
        
        # Convert to NumPy arrays (if not already)
        widths = np.asarray(widths)
        counts_per_width_norm = np.asarray(counts_per_width_norm)

        # Create a mask for valid (finite) values
        valid_mask = np.isfinite(widths) & np.isfinite(counts_per_width_norm)

        # Apply mask to both x and y data
        widths_clean = widths[valid_mask]
        counts_clean = counts_per_width_norm[valid_mask]
        
        if len(counts_clean) == 0 or len(widths_clean) == 0:
            print(f"[Warning] Skipping definitive_tt {definitive_tt}: no valid data.")
            continue
        
        # Then fit
        try:
            popt, pcov = curve_fit(
                signal_plus_background,
                widths_clean,
                counts_clean,
                p0=p0,
                maxfev=10000,
            )
        except RuntimeError as exc:
            print(f"[Warning] Fit failed for definitive_tt {definitive_tt}: {exc}")
            global_variables[f'sigmoid_width_{definitive_tt}'] = np.nan
            global_variables[f'background_slope_{definitive_tt}'] = np.nan
            global_variables[f'sigmoid_amplitude_{definitive_tt}'] = np.nan
            global_variables[f'sigmoid_center_{definitive_tt}'] = np.nan
            continue
                
        S_fit, w0_fit, tau_fit, B_fit = popt
        print(f"definitive_tt {definitive_tt} - Fit parameters:\n  Signal amplitude S = {S_fit:.4f}\n  Transition center w0 = {w0_fit:.4f} ns\n  Transition width τ = {tau_fit:.4f} ns\n  Background slope B = {B_fit:.6f} per ns")

        global_variables[f'sigmoid_width_{definitive_tt}'] = tau_fit
        global_variables[f'background_slope_{definitive_tt}'] = B_fit
        global_variables[f'sigmoid_amplitude_{definitive_tt}'] = S_fit
        global_variables[f'sigmoid_center_{definitive_tt}'] = w0_fit

       
        if create_plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(widths, counts_per_width_norm, label='Normalized average count in window')
            ax.axvline(x=coincidence_window_cal_ns, color='red', linestyle='--', label='Time coincidence window')
            ax.set_xlabel("Window width (ns)")
            ax.set_ylabel("Normalized average # of T_sum values in window")
            ax.set_title("Fraction of hits within stat-centered window vs width")
            ax.grid(True)
            w_fit = np.linspace(min(widths), max(widths), 300)
            f_fit = signal_plus_background(w_fit, *popt)
            ax.plot(w_fit, f_fit, 'k--', label='Signal + background fit')
            ax.axhline(S_fit, color='green', linestyle=':', alpha=0.6, label=f'Signal plateau ≈ {S_fit:.2f}')
            s_vals = S_fit / (1 + np.exp(-(w_fit - w0_fit) / tau_fit))
            b_vals = B_fit * w_fit
            f_vals = s_vals + b_vals
            P_signal = s_vals / f_vals
            P_background = b_vals / f_vals
            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)
            ax_fill = fig.add_subplot(gs[0])  # Top: signal vs. background fill
            ax_main = fig.add_subplot(gs[1], sharex=ax_fill)  # Bottom: your original plot
            ax_fill.fill_between(w_fit, 0, P_signal, color='green', alpha=0.4, label='Signal')
            ax_fill.fill_between(w_fit, P_signal, 1, color='red', alpha=0.4, label='Background')
            ax_fill.set_ylabel("Fraction")
            y_min = float(np.nanmin(P_signal))
            if not np.isfinite(y_min) or y_min >= 1.0 - np.finfo(float).eps:
                y_min = 0.0
            elif y_min < 0.0:
                y_min = 0.0
            ax_fill.set_ylim(y_min, 1)
            # ax_fill.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_fill.legend(loc="upper right")
            ax_fill.set_title(f"Estimated Signal and Background Fractions per Window Width, definitive_tt = {definitive_tt}")
            plt.setp(ax_fill.get_xticklabels(), visible=False)
            ax_main.scatter(widths, counts_per_width_norm, label='Normalized average count in window')
            ax_main.plot(w_fit, f_fit, 'k--', label='Signal + background fit')
            ax_main.axhline(S_fit, color='green', linestyle=':', alpha=0.6, label=f'Signal plateau ≈ {S_fit:.2f}')
            ax_main.set_xlabel("Window width (ns)")
            ax_main.set_ylabel("Normalized average # of T_sum values in window")
            ax_main.grid(True)
            fit_summary = (f"Fit: S = {S_fit:.3f}, w₀ = {w0_fit:.3f} ns, " f"τ = {tau_fit:.3f} ns, B = {B_fit:.4f}/ns")
            ax_main.plot([], [], ' ', label=fit_summary)  # invisible handle to add text
            ax_main.legend()
            
            if save_plots:
                name_of_file = f'stat_window_accumulation_{definitive_tt}'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()

    print(
        f"[PROFILE][TASK_4] time-window fitting section: {time.perf_counter() - time_window_fitting_start:.2f}s",
        force=True,
    )

# -----------------------------------------------------------------------------
# Last filterings -------------------------------------------------------------
# -----------------------------------------------------------------------------

# Put to zero the rows with tracking in only one plane, that is, put 0 if tracking_tt < 10
low_tt_zeroed_df = pd.DataFrame()
low_tt_mask = np.zeros(len(working_df), dtype=bool)
low_tt_zeroed_count = 0
low_tt_min = int(config.get("low_tt_min", 10))
low_tt_zero_cols = config.get("low_tt_zero_cols", ["x", "xp", "y", "yp", "t0", "s"])
if not isinstance(low_tt_zero_cols, (list, tuple)):
    low_tt_zero_cols = [c for c in re.split(r"[\\s,;]+", str(low_tt_zero_cols).strip()) if c]
raw_tt_series = working_df["raw_tt"] if "raw_tt" in working_df.columns else pd.Series(0, index=working_df.index)
if create_debug_plots:
    debug_cols = [c for c in ("tracking_tt", "list_tt", "raw_tt", "definitive_tt") if c in working_df.columns]
    if debug_cols:
        debug_thresholds = {col: [low_tt_min] for col in debug_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            debug_cols,
            debug_thresholds,
            title=(
                f"Task 4 pre-filter: low_tt_min={low_tt_min} "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )
low_tt_mask = (
    (working_df["tracking_tt"] < low_tt_min)
    | (working_df["list_tt"] < low_tt_min)
    | (raw_tt_series < low_tt_min)
    | (working_df["definitive_tt"] < low_tt_min)
)
low_tt_zeroed_count = int(low_tt_mask.sum())
if low_tt_zeroed_count > 0:
    cols_to_zero = [c for c in low_tt_zero_cols if c in working_df.columns]
    if save_rejected_rows or create_reject_plots:
        low_tt_zeroed_df = working_df.loc[low_tt_mask].copy()
        low_tt_zeroed_df["reject_stage"] = "low_tt_zeroed"
        low_tt_zeroed_df["reject_reason"] = f"tracking/list/raw/definitive tt < {low_tt_min}"
        low_tt_zeroed_df["zeroed_cols"] = ",".join(cols_to_zero)
    if cols_to_zero:
        working_df.loc[low_tt_mask, cols_to_zero] = 0

# -----------------------------------------------------------------------------
# -------------- Correlate trigger types in different stages ------------------
# -----------------------------------------------------------------------------

def plot_tt_correlation(df, row_label, col_label, title, filename_suffix, fig_idx, base_dir, show_plots=False, save_plots=False, plot_list=None):

    analysis_data = df[[row_label, col_label]]
    counts = analysis_data.groupby([row_label, col_label]).size().unstack(fill_value=0)

    row_order = sorted(analysis_data[row_label].unique(), reverse=True)
    col_unique = analysis_data[col_label].unique()
    col_order = list(row_order) + [x for x in col_unique if x not in row_order]
    counts = counts.reindex(index=row_order, columns=col_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xticks(np.arange(len(counts.columns)))
    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_xticklabels(counts.columns)
    ax.set_yticklabels(counts.index)

    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)

    im = ax.imshow(counts, cmap='plasma', origin='lower')
    total = counts.values.sum()

    for i in range(len(counts.index)):
        for j in range(len(counts.columns)):
            value = counts.iloc[i, j]
            if value > 0:
                pct = 100 * value / total
                if pct > 1:
                    ax.text(j, i, f"{pct:.1f}%",
                            ha="center", va="center",
                            color="black" if value > counts.values.max() * 0.5 else "white")

    plt.tight_layout()
    if save_plots:
        final_filename = f'{fig_idx}_{filename_suffix}.png'
        save_fig_path = os.path.join(base_dir, final_filename)
        if plot_list is not None:
            plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

    return fig_idx + 1

if create_plots:
    fig_idx = plot_tt_correlation(
        df=working_df,
        row_label='raw_tt',
        col_label='list_tt',
        title='Event counts per (raw_tt, list_tt) combination',
        filename_suffix='trigger_types_raw_and_list',
        fig_idx=fig_idx,
        base_dir=base_directories["figure_directory"],
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list
    )

    fig_idx = plot_tt_correlation(
        df=working_df,
        row_label='tracking_tt',
        col_label='list_tt',
        title='Event counts per (tracking_tt, list_tt) combination',
        filename_suffix='trigger_types_tracking_and_list',
        fig_idx=fig_idx,
        base_dir=base_directories["figure_directory"],
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list
    )

    fig_idx = plot_tt_correlation(
        df=working_df,
        row_label='tracking_tt',
        col_label='raw_tt',
        title='Event counts per (tracking_tt, raw_tt) combination',
        filename_suffix='trigger_types_tracking_and_raw',
        fig_idx=fig_idx,
        base_dir=base_directories["figure_directory"],
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list
    )

if create_plots or create_essential_plots:
    fig_idx = plot_tt_correlation(
        df=working_df,
        row_label='raw_tt',
        col_label='definitive_tt',
        title='Event counts per (raw_tt, definitive_tt) combination',
        filename_suffix='trigger_types_definitive_tt_and_raw',
        fig_idx=fig_idx,
        base_dir=base_directories["figure_directory"],
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list
    )

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Define the last dataframe, the definitive one -------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

working_df = working_df.copy()

# Remove rows with zeros in key places ----------------------------------------
definitive_nonzero_cols = config.get("definitive_nonzero_cols", ["x", "y", "s", "t0", "theta", "phi"])
if not isinstance(definitive_nonzero_cols, (list, tuple)):
    definitive_nonzero_cols = [
        c for c in re.split(r"[\\s,;]+", str(definitive_nonzero_cols).strip()) if c
    ]
cols_to_check = [c for c in definitive_nonzero_cols if c in working_df.columns]

if remove_small:
    # Remove small, non-zero values -----------------------------------------------
    if create_debug_plots and cols_to_check:
        debug_thresholds = {col: [-eps, eps] for col in cols_to_check}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            cols_to_check,
            debug_thresholds,
            title=(
                f"Task 4 pre-filter (definitive): remove_small_eps={eps:g} "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )
    mask = working_df.map(is_small_nonzero)
    nonzero_numeric_mask = working_df.map(lambda x: isinstance(x, (int, float)) and x != 0)
    n_events = len(working_df)
    rows_with_small = int(mask.any(axis=1).sum())
    n_total = nonzero_numeric_mask.sum().sum()
    n_small = mask.sum().sum()
    working_df = working_df.mask(mask, 0)
    pct = 100 * n_small / n_total if n_total > 0 else 0
    print(f"\nIn working_df {n_small} out of {n_total} non-zero numeric values are below {eps} ({pct:.4f}%)")
    record_filter_metric(
        "definitive_small_values_zeroed_event_pct",
        rows_with_small,
        n_events if n_events else 0,
    )
    record_filter_metric(
        "definitive_small_values_zeroed_value_pct",
        n_small,
        n_total if n_total else 0,
    )

baseline_events = original_number_of_events if original_number_of_events else len(working_df)
if create_debug_plots and cols_to_check:
    debug_thresholds = {col: [0] for col in cols_to_check}
    debug_fig_idx = plot_debug_histograms(
        working_df,
        cols_to_check,
        debug_thresholds,
        title=(
            f"Task 4 pre-filter: definitive_nonzero_cols "
            f"[tunable] (station {station})"
        ),
        out_dir=debug_plot_directory,
        fig_idx=debug_fig_idx,
    )
record_filter_metric(
    "low_tt_zeroed_event_pct",
    int(low_tt_zeroed_count),
    baseline_events if baseline_events else 0,
)
for col in cols_to_check:
    zero_rows = int((working_df[col] == 0).sum())
    record_filter_metric(
        f"definitive_{col}_zero_rows_pct",
        zero_rows,
        baseline_events if baseline_events else 0,
    )

if cols_to_check:
    cond = (working_df[cols_to_check[0]] != 0)
    for col in cols_to_check[1:]:
        cond &= (working_df[col] != 0)
else:
    print("Warning: definitive_nonzero_cols resolved to empty; skipping zero-row removal.")
    cond = np.ones(len(working_df), dtype=bool)

removed_mask = ~cond
removed_total = int(removed_mask.sum())
zero_counts = np.zeros(len(working_df), dtype=int)
for col in cols_to_check:
    if col in working_df.columns:
        zero_counts += (working_df[col].to_numpy(copy=False) == 0)

zero_cols_present = [col for col in cols_to_check if col in working_df.columns]
primary_zero_col = np.full(len(working_df), "", dtype=object)
for col in cols_to_check:
    if col in working_df.columns:
        mask = (primary_zero_col == "") & (working_df[col] == 0)
        primary_zero_col[mask] = col

rejected_definitive_df = pd.DataFrame()
if (save_rejected_rows or create_reject_plots) and removed_total > 0:
    rejected_definitive_df = working_df.loc[removed_mask].copy()
    rejected_definitive_df["reject_stage"] = "definitive_zero_rows"
    rejected_definitive_df["reject_reason"] = "zero in x/y/s/t0/theta/phi"
    rejected_definitive_df["zero_count"] = zero_counts[removed_mask]
    rejected_definitive_df["primary_zero_col"] = primary_zero_col[removed_mask]
    if zero_cols_present:
        zero_mat_removed = np.column_stack(
            [(working_df[col].to_numpy(copy=False) == 0)[removed_mask] for col in zero_cols_present]
        )
        zero_cols_str = [
            ",".join([col for col, is_zero in zip(zero_cols_present, row) if is_zero])
            for row in zero_mat_removed
        ]
        rejected_definitive_df["zero_cols"] = zero_cols_str
    if len(low_tt_mask) == len(working_df):
        rejected_definitive_df["low_tt_zeroed"] = low_tt_mask[removed_mask]

single_zero = removed_mask & (zero_counts == 1)
multi_zero = removed_mask & (zero_counts >= 2)
record_filter_metric(
    "definitive_removed_single_zero_rows_pct",
    int(single_zero.sum()),
    baseline_events if baseline_events else 0,
)
record_filter_metric(
    "definitive_removed_multi_zero_rows_pct",
    int(multi_zero.sum()),
    baseline_events if baseline_events else 0,
)

remaining = removed_mask.copy()
for col in cols_to_check:
    if col not in working_df.columns:
        continue
    primary_mask = remaining & (working_df[col] == 0)
    record_filter_metric(
        f"definitive_removed_primary_{col}_zero_rows_pct",
        int(primary_mask.sum()),
        baseline_events if baseline_events else 0,
    )
    remaining &= ~primary_mask

n_before = len(working_df)
working_df = working_df.loc[cond].copy()
n_after = len(working_df)

# Calculate and print percentage ----------------------------------------------
percentage_retained = 100 * n_after / n_before if n_before > 0 else 0
print(f"Rows before: {n_before}")
print(f"Rows after: {n_after}")
print(f"Retained: {percentage_retained:.2f}%")
record_filter_metric(
    "definitive_rows_removed_pct",
    removed_total,
    baseline_events if baseline_events else 0,
)

if save_rejected_rows:
    os.makedirs(rejected_files_directory, exist_ok=True)
    if not low_tt_zeroed_df.empty:
        low_tt_path = os.path.join(rejected_files_directory, f"zeroed_low_tt_{basename_no_ext}.parquet")
        low_tt_zeroed_df.to_parquet(low_tt_path, engine="pyarrow", compression="zstd", index=False)
        print(f"Rejected rows (low_tt zeroed) saved to: {low_tt_path}")
    if not rejected_definitive_df.empty:
        rejected_path = os.path.join(rejected_files_directory, f"rejected_definitive_{basename_no_ext}.parquet")
        rejected_definitive_df.to_parquet(rejected_path, engine="pyarrow", compression="zstd", index=False)
        print(f"Rejected rows (definitive zero) saved to: {rejected_path}")
    if not low_tt_zeroed_df.empty or not rejected_definitive_df.empty:
        combined = pd.concat([df for df in (low_tt_zeroed_df, rejected_definitive_df) if not df.empty], ignore_index=True)
        combined_path = os.path.join(rejected_files_directory, f"rejected_combined_{basename_no_ext}.parquet")
        combined.to_parquet(combined_path, engine="pyarrow", compression="zstd", index=False)
        print(f"Rejected rows (combined) saved to: {combined_path}")

if create_reject_plots:
    # Ensure saving is enabled even if global plot flags are off.
    save_plots = True
    reject_plot_directory = os.path.join(
        base_directories["ancillary_directory"],
        "REJECTED_FILES",
        "REJECTED_PLOTS",
        f"FIGURES_EXEC_ON_{date_execution}",
    )
    os.makedirs(reject_plot_directory, exist_ok=True)

    # Save reject-only plots outside PLOTS so they won't be deleted by cleanup.
    original_figure_dir = base_directories["figure_directory"]
    original_plot_list = plot_list if "plot_list" in globals() else []
    base_directories["figure_directory"] = reject_plot_directory
    plot_list = []
    fig_idx_reject = 1
    fig_idx_backup = globals().get("fig_idx")
    fig_idx = fig_idx_reject

    print(
        f"Warning: reject-plots saving to {reject_plot_directory} | "
        f"low_tt_rows={len(low_tt_zeroed_df)} | definitive_rows={len(rejected_definitive_df)}"
    )

    if not low_tt_zeroed_df.empty and "datetime" in low_tt_zeroed_df.columns:
        cols = [c for c in ("x", "y", "theta", "phi", "s", "t0") if c in low_tt_zeroed_df.columns]
        if cols:
            plot_ts_with_side_hist(
                low_tt_zeroed_df,
                cols,
                "datetime",
                f"rejected_low_tt_{basename_no_ext}",
            )
    if not rejected_definitive_df.empty and "datetime" in rejected_definitive_df.columns:
        cols = [c for c in ("x", "y", "theta", "phi", "s", "t0") if c in rejected_definitive_df.columns]
        if cols:
            plot_ts_with_side_hist(
                rejected_definitive_df,
                cols,
                "datetime",
                f"rejected_definitive_{basename_no_ext}",
            )

    combined_rejected_df = None
    if not low_tt_zeroed_df.empty or not rejected_definitive_df.empty:
        combined_rejected_df = pd.concat(
            [df for df in (low_tt_zeroed_df, rejected_definitive_df) if not df.empty],
            ignore_index=True,
        )

    if not low_tt_zeroed_df.empty:
        plot_reject_diagnostics(
            low_tt_zeroed_df,
            "rejected_low_tt",
            basename_no_ext,
            hist_bins=reject_plot_hist_bins,
            cols_per_fig=reject_plot_hist_cols_per_fig,
            scatter_max=reject_plot_scatter_max_points,
        )
    if not rejected_definitive_df.empty:
        plot_reject_diagnostics(
            rejected_definitive_df,
            "rejected_definitive",
            basename_no_ext,
            hist_bins=reject_plot_hist_bins,
            cols_per_fig=reject_plot_hist_cols_per_fig,
            scatter_max=reject_plot_scatter_max_points,
        )
    if combined_rejected_df is not None and not combined_rejected_df.empty:
        plot_reject_diagnostics(
            combined_rejected_df,
            "rejected_all",
            basename_no_ext,
            hist_bins=reject_plot_hist_bins,
            cols_per_fig=reject_plot_hist_cols_per_fig,
            scatter_max=reject_plot_scatter_max_points,
        )

    # Restore plot destinations for the rest of the pipeline.
    base_directories["figure_directory"] = original_figure_dir
    plot_list = original_plot_list
    if fig_idx_backup is not None:
        fig_idx = fig_idx_backup

print("----------------------------------------------------------------------")
for tt_col in ("raw_tt", "clean_tt", "cal_tt", "list_tt", "tracking_tt", "definitive_tt"):
    if tt_col in working_df.columns:
        try:
            print(f"Unique {tt_col} values:", sorted(working_df[tt_col].unique()))
        except Exception:
            print(f"Could not list unique values for {tt_col}")
print("----------------------------------------------------------------------")

print("----------------------------------------------------------------------")
print("----------------------- Calculating some stuff -----------------------")
print("----------------------------------------------------------------------")

df_plot_ancillary = working_df.copy()
ancillary_before = len(df_plot_ancillary)

cond = ( df_plot_ancillary['charge_1'] < charge_plot_limit_right ) &\
    ( df_plot_ancillary['charge_2'] < charge_plot_limit_right ) &\
    ( df_plot_ancillary['charge_3'] < charge_plot_limit_right ) &\
    ( df_plot_ancillary['charge_4'] < charge_plot_limit_right ) &\
    ( df_plot_ancillary['charge_event'] > charge_plot_limit_left )

df_plot_ancillary = df_plot_ancillary.loc[cond].copy()
record_filter_metric(
    "ancillary_charge_filtered_rows_pct",
    ancillary_before - len(df_plot_ancillary),
    ancillary_before if ancillary_before else 0,
)

# -----------------------------------------------------------------------------------------------------------------------------

if create_plots:
    
    # Combined methods --------------------------------------------------------------------------------------------
    residual_columns = [
        'res_ystr_1', 'res_ystr_2', 'res_ystr_3', 'res_ystr_4',
        'res_tsum_1', 'res_tsum_2', 'res_tsum_3', 'res_tsum_4',
        'res_tdif_1', 'res_tdif_2', 'res_tdif_3', 'res_tdif_4'
    ]
    
    unique_types = df_plot_ancillary['definitive_tt'].unique()
    for t in unique_types:
        if t < 1000:
            continue
        subset_data = df_plot_ancillary[df_plot_ancillary['definitive_tt'] == t]
        plot_histograms_and_gaussian(subset_data, residual_columns, f"TimTrack Residuals with Gaussian for Processed Type {t}", figure_number=2, fit_gaussian=True, quantile=0.99)
    
    
    # Combined methods - External residues -------------------------------------------------------------------------
    residual_columns = [
        'ext_res_ystr_1', 'ext_res_ystr_2', 'ext_res_ystr_3', 'ext_res_ystr_4',
        'ext_res_tsum_1', 'ext_res_tsum_2', 'ext_res_tsum_3', 'ext_res_tsum_4',
        'ext_res_tdif_1', 'ext_res_tdif_2', 'ext_res_tdif_3', 'ext_res_tdif_4'
    ]

    unique_types = df_plot_ancillary['definitive_tt'].unique()
    for t in unique_types:
        if t < 1000:
            continue
        subset_data = df_plot_ancillary[df_plot_ancillary['definitive_tt'] == t]
        plot_histograms_and_gaussian(subset_data, residual_columns, f"External Residuals with Gaussian for Processed Type {t}", figure_number=2, fit_gaussian=True, quantile=0.99)

# -----------------------------------------------------------------------------------------------------------------------------

if create_plots or create_essential_plots:
    df_filtered = df_plot_ancillary
    # tt_values = sorted(df_filtered['definitive_tt'].dropna().unique(), key=lambda x: int(x))

    tt_values = [13, 12, 23, 34, 123, 124, 134, 234, 1234]
    
    n_tt = len(tt_values)
    ncols = 3
    nrows = (n_tt + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    phi_nbins = 28
    # theta_nbins = int(round(phi_nbins / 2) + 1)
    theta_nbins = 40
    theta_bins = np.linspace(theta_left_filter, theta_right_filter, theta_nbins )
    phi_bins = np.linspace(phi_left_filter, phi_right_filter, phi_nbins)
    colors = plt.cm.turbo

    # Select theta/phi range (optional filtering)
    theta_min, theta_max = theta_left_filter, theta_right_filter    # adjust as needed
    phi_min, phi_max     = phi_left_filter, phi_right_filter        # adjust as needed
    
    vmax_global = (
        df_filtered.groupby('definitive_tt')[['theta', 'phi']]
        .apply(
            lambda df: np.histogram2d(
                df['theta'],
                df['phi'],
                bins=[theta_bins, phi_bins],
            )[0].max()
        )
        .max()
    )
    
    for idx, tt_val in enumerate(tt_values):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]

        df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val]
        theta_vals = df_tt['theta'].dropna()
        phi_vals = df_tt['phi'].dropna()

        # Apply range filtering
        mask = (theta_vals >= theta_min) & (theta_vals <= theta_max) & \
               (phi_vals >= phi_min) & (phi_vals <= phi_max)
        theta_vals = theta_vals[mask]
        phi_vals   = phi_vals[mask]

        if len(theta_vals) < 10 or len(phi_vals) < 10:
            ax.set_visible(False)
            continue

        # Polar plot settings
        fig.delaxes(axes[row_idx][col_idx])  # remove the original non-polar Axes
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)  # add a polar Axes
        axes[row_idx][col_idx] = ax  # update reference for consistency

        ax.set_facecolor(colors(0.0))  # darkest background in colormap

        # 2D histogram: use phi as angle, theta as radius
        h, r_edges, phi_edges = np.histogram2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins])
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        R, PHI = np.meshgrid(r_centers, phi_centers, indexing='ij')
        c = ax.pcolormesh(PHI, R, h, cmap='viridis', vmin=0, vmax=vmax_global)
        local_max = h.max()
        cb = fig.colorbar(c, ax=ax, pad=0.1)
        cb.ax.hlines(local_max, *cb.ax.get_xlim(), colors='white', linewidth=2, linestyles='dashed')
        # Put as title of the subplot the definitive_tt value
        ax.set_title(f'Plane combination (definitive) {tt_val}', fontsize=10)

    plt.suptitle(r'2D Histogram of $\theta$ vs. $\phi$ for each definitive_tt Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_polar_theta_phi_definitive_tt_2D.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# Track-consistency: sample 4-fold event displays — XZ and YZ projections
# Z on vertical axis; equal mm scale so real detector proportions are visible
# ---------------------------------------------------------------------------
if (create_essential_plots or create_plots) and task4_plot_enabled("event_display_sample"):
    _evd_y_cols = [f"P{p}_Y_final" for p in range(1, 5)]
    _evd_td_cols = [f"P{p}_T_dif_final" for p in range(1, 5)]
    _evd_q_cols = [f"P{p}_Q_sum_final" for p in range(1, 5)]
    _evd_have = all(c in df_plot_ancillary.columns for c in _evd_y_cols + _evd_td_cols + _evd_q_cols)
    _evd_have_track = all(c in df_plot_ancillary.columns for c in ("x", "y", "xp", "yp"))
    if _evd_have and _evd_have_track:
        _evd_pool = df_plot_ancillary[df_plot_ancillary["definitive_tt"] == 1234]
        n_evd = min(16, len(_evd_pool))
        if n_evd > 0:
            rng_evd = np.random.default_rng(42)
            idx_evd = rng_evd.choice(len(_evd_pool), size=n_evd, replace=False)
            _evd_sample = _evd_pool.iloc[sorted(idx_evd)]

            z_arr = np.asarray(z_positions, dtype=float)
            z_margin = 25.0
            z_lo = z_arr.min() - z_margin
            z_hi = z_arr.max() + z_margin
            z_span = z_hi - z_lo                       # mm
            x_span = 2.0 * strip_half                  # mm
            y_span = 2.0 * width_half                  # mm

            # Figure sized so that 1 mm ≈ px_per_mm inches on screen
            # → GridSpec columns proportional to physical width → set_aspect('equal')
            #   will fill each subplot box exactly, showing true detector proportions.
            px_per_mm = 0.008
            col_h_in = z_span * px_per_mm
            col_xz_w_in = x_span * px_per_mm
            col_yz_w_in = y_span * px_per_mm
            events_per_row = 4
            n_rows_evd = 4
            fig_w = (col_xz_w_in + col_yz_w_in) * events_per_row + 1.5
            fig_h = col_h_in * n_rows_evd + 1.5

            fig = plt.figure(figsize=(fig_w, fig_h))
            # Columns alternate [XZ, YZ] for each of the 4 events per row
            gs = GridSpec(
                n_rows_evd, events_per_row * 2,
                figure=fig,
                width_ratios=[x_span, y_span] * events_per_row,
                hspace=0.55,
                wspace=0.20,
            )

            z_line = np.linspace(z_lo, z_hi, 80)

            for ev_i, (_, row) in enumerate(_evd_sample.iterrows()):
                r_idx = ev_i // events_per_row
                c_ev = ev_i % events_per_row
                ax_xz = fig.add_subplot(gs[r_idx, c_ev * 2])
                ax_yz = fig.add_subplot(gs[r_idx, c_ev * 2 + 1])

                y_meas = np.array([row.get(c, np.nan) for c in _evd_y_cols], dtype=float)
                td_meas = np.array([row.get(c, np.nan) for c in _evd_td_cols], dtype=float)
                x_meas = td_meas * tdiff_to_x
                q_meas = np.array([row.get(c, 0.0) for c in _evd_q_cols], dtype=float)
                active = np.isfinite(y_meas) & (y_meas != 0)

                x0_ev = float(row.get("x", 0.0))
                y0_ev = float(row.get("y", 0.0))
                xp_ev = float(row.get("xp", 0.0))
                yp_ev = float(row.get("yp", 0.0))
                x_line = x0_ev + xp_ev * z_line
                y_line = y0_ev + yp_ev * z_line

                # Plane-position guidelines (horizontal lines at each plane's Z)
                for zp in z_arr:
                    ax_xz.axhline(zp, color="lightgray", lw=0.4, zorder=0)
                    ax_yz.axhline(zp, color="lightgray", lw=0.4, zorder=0)
                # Strip Y-centre guidelines on YZ panel
                for sy in y_pos_P1_and_P3:
                    ax_yz.axvline(sy, color="lightgray", lw=0.4, ls="--", zorder=0)

                # Measured hits — errorbar (behind) + marker (in front)
                # X (T_dif): the "better" direction  → thin, subtle bar
                # Y (strip): the "worse"  direction  → slightly thicker/opaque bar
                for pp in range(4):
                    if active[pp]:
                        sz_ev = float(np.clip(q_meas[pp] * 4, 20, 200)) if q_meas[pp] > 0 else 20
                        ax_xz.errorbar(
                            x_meas[pp], z_arr[pp], xerr=anc_sx,
                            fmt="none", ecolor=f"C{pp}",
                            elinewidth=0.7, capsize=1.5, capthick=0.6,
                            alpha=0.30, zorder=2,
                        )
                        ax_yz.errorbar(
                            y_meas[pp], z_arr[pp], xerr=anc_sy,
                            fmt="none", ecolor=f"C{pp}",
                            elinewidth=1.1, capsize=2.5, capthick=0.9,
                            alpha=0.45, zorder=2,
                        )
                        ax_xz.scatter(x_meas[pp], z_arr[pp], s=sz_ev,
                                      c=f"C{pp}", zorder=3, alpha=0.9)
                        ax_yz.scatter(y_meas[pp], z_arr[pp], s=sz_ev,
                                      c=f"C{pp}", zorder=3, alpha=0.9)

                # Fitted track line
                ax_xz.plot(x_line, z_line, "r-", lw=0.9, alpha=0.8, zorder=2)
                ax_yz.plot(y_line, z_line, "r-", lw=0.9, alpha=0.8, zorder=2)

                # Axis limits — real mm detector dimensions
                ax_xz.set_xlim(-strip_half, strip_half)
                ax_xz.set_ylim(z_lo, z_hi)
                ax_xz.set_aspect("equal")
                ax_xz.set_xlabel("X (mm)", fontsize=5)
                ax_xz.set_ylabel("Z (mm)", fontsize=5)
                ax_xz.tick_params(labelsize=4)
                ax_xz.set_title(f"ev{ev_i + 1} XZ", fontsize=6)

                ax_yz.set_xlim(-width_half, width_half)
                ax_yz.set_ylim(z_lo, z_hi)
                ax_yz.set_aspect("equal")
                ax_yz.set_xlabel("Y (mm)", fontsize=5)
                ax_yz.set_ylabel("Z (mm)", fontsize=5)
                ax_yz.tick_params(labelsize=4)
                ax_yz.set_title(f"ev{ev_i + 1} YZ", fontsize=6)

            plt.suptitle(
                "Sample 4-fold event displays — equal mm scale (Z vertical)\n"
                "C0–C3 = planes 1–4  |  marker size ~ charge  |  red line = fitted track  |  "
                f"detector: X={x_span:.0f} mm  Y={y_span:.0f} mm  Z={z_arr[-1]:.0f} mm\n"
                f"errorbars: XZ = ±σ_X (T_dif, {anc_sx:.0f} mm, thin)  ·  "
                f"YZ = ±σ_Y (strip, {anc_sy:.0f} mm, thick)",
                fontsize=8,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            if save_plots:
                final_filename = f"{fig_idx}_event_display_sample.png"
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                save_plot_figure(save_fig_path, format="png", alias="event_display_sample")
            if show_plots:
                plt.show()
            plt.close()

# ---------------------------------------------------------------------------
# Track-consistency: LOO (leave-one-out) residuals per plane (2D histograms)
# ---------------------------------------------------------------------------
if (create_essential_plots or create_plots) and task4_plot_enabled("track_consistency_loo_residuals"):
    _loo_y_cols = [f"ext_res_ystr_{p}" for p in range(1, 5)]
    _loo_td_cols = [f"ext_res_tdif_{p}" for p in range(1, 5)]
    _loo_have = all(c in df_plot_ancillary.columns for c in _loo_y_cols + _loo_td_cols + ["definitive_tt"])
    if _loo_have:
        # 3-fold combos containing each plane — removing that plane leaves a 2-plane telescope,
        # which is nearly degenerate (5 params, 6 equations → residuals ≈ 0).
        # 4-fold events (1234) — removing any plane leaves a proper 3-plane telescope.
        _loo_3fold_by_plane = {
            1: [123, 124, 134],
            2: [123, 124, 234],
            3: [123, 134, 234],
            4: [124, 134, 234],
        }
        _loo_row_defs = [
            (lambda p: [1234],                  "4-fold  (3-plane telescope — non-degenerate)"),
            (lambda p: _loo_3fold_by_plane[p],  "3-fold  (2-plane telescope — degenerate, residuals → 0)"),
        ]

        fig_loo, axes_loo = plt.subplots(
            2, 4, figsize=(18, 9), squeeze=False,
            gridspec_kw={"hspace": 0.50, "wspace": 0.35},
        )
        _loo_df_all = df_plot_ancillary.copy()

        for _ri, (tt_fn, row_label) in enumerate(_loo_row_defs):
            for _pi in range(4):
                p = _pi + 1
                ax = axes_loo[_ri][_pi]
                _tt_filter = tt_fn(p)
                _mask_tt = _loo_df_all["definitive_tt"].isin(_tt_filter)
                _sub = _loo_df_all[_mask_tt]

                ry = pd.to_numeric(_sub[f"ext_res_ystr_{p}"], errors="coerce")
                rx = pd.to_numeric(_sub[f"ext_res_tdif_{p}"], errors="coerce") * tdiff_to_x

                # For the non-degenerate row: exclude events where LOO wasn't run (default 0)
                # For the degenerate row: keep all (zeros ARE the point)
                if _ri == 0:
                    _mask_loo = ry.notna() & rx.notna() & (ry != 0) & (rx != 0)
                else:
                    _mask_loo = ry.notna() & rx.notna()

                ry = ry[_mask_loo]
                rx = rx[_mask_loo]

                if len(rx) < 5:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9)
                    ax.set_title(f"Plane {p}", fontsize=9)
                    ax.set_visible(False)
                    continue

                xlim = max(float(rx.abs().quantile(0.99)), 1.0)
                ylim = max(float(ry.abs().quantile(0.99)), 1.0)
                hb = ax.hexbin(rx, ry, gridsize=40, cmap="viridis",
                               extent=(-xlim, xlim, -ylim, ylim), mincnt=1)
                plt.colorbar(hb, ax=ax, label="counts", pad=0.02)
                ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.6)
                ax.axvline(0, color="red", lw=0.8, ls="--", alpha=0.6)
                ax.set_xlabel("LOO X residual (mm)", fontsize=8)
                if _pi == 0:
                    ax.set_ylabel(f"LOO Y residual (mm)\n[{row_label}]", fontsize=7)
                else:
                    ax.set_ylabel("LOO Y residual (mm)", fontsize=8)
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(-ylim, ylim)
                ax.set_title(
                    f"Plane {p}  |  tt∈{_tt_filter}\nn={len(rx):,}",
                    fontsize=8,
                )
                ax.grid(True, alpha=0.3)

        fig_loo.suptitle(
            "Track-consistency: leave-one-out (LOO) residuals per plane\n"
            "Top: 4-fold events — 3-plane telescope (proper, non-degenerate)  |  "
            "Bottom: 3-fold events — 2-plane telescope (degenerate → residuals collapse to 0)\n"
            "X residual = (predicted − measured) × strip_speed  |  Y residual = predicted − measured strip Y",
            fontsize=9,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        if save_plots:
            final_filename = f"{fig_idx}_track_consistency_loo_residuals.png"
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format="png", alias="track_consistency_loo_residuals")
        if show_plots:
            plt.show()
        plt.close()

# ---------------------------------------------------------------------------
# Strip hit occupancy: 2D hit map per plane (X from T_dif, Y from Y_final)
# ---------------------------------------------------------------------------
if (create_essential_plots or create_plots) and task4_plot_enabled("strip_hit_occupancy"):
    _occ_y_cols = [f"P{p}_Y_final" for p in range(1, 5)]
    _occ_td_cols = [f"P{p}_T_dif_final" for p in range(1, 5)]
    _occ_have = all(c in df_plot_ancillary.columns for c in _occ_y_cols + _occ_td_cols)
    if _occ_have:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
        for pp in range(4):
            ax = axes[0][pp]
            y_hits = pd.to_numeric(df_plot_ancillary[f"P{pp + 1}_Y_final"], errors="coerce")
            x_hits = pd.to_numeric(df_plot_ancillary[f"P{pp + 1}_T_dif_final"], errors="coerce") * tdiff_to_x
            mask_occ = y_hits.notna() & x_hits.notna() & (y_hits != 0) & (x_hits != 0)
            x_hits = x_hits[mask_occ]
            y_hits = y_hits[mask_occ]
            if len(x_hits) < 5:
                ax.set_title(f"Plane {pp + 1}: no data", fontsize=9)
                ax.set_visible(False)
                continue
            hb = ax.hexbin(x_hits, y_hits, gridsize=30, cmap="hot_r", mincnt=1,
                           extent=(-strip_half, strip_half, -width_half, width_half))
            plt.colorbar(hb, ax=ax, label="hits")
            y_centers = y_pos_P1_and_P3 if pp % 2 == 0 else y_pos_P2_and_P4
            for sy in y_centers:
                ax.axhline(sy, color="cyan", lw=0.7, ls="--", alpha=0.8)
            ax.set_xlim(-strip_half, strip_half)
            ax.set_ylim(-width_half, width_half)
            ax.set_xlabel("X from T_dif (mm)", fontsize=9)
            ax.set_ylabel("Y_final (mm)", fontsize=9)
            ax.set_title(f"Plane {pp + 1} hit map  (n={len(x_hits)})", fontsize=9)
            ax.grid(True, alpha=0.2)
        plt.suptitle(
            "Strip hit occupancy: 2D hit density per plane\n"
            "Cyan dashed = strip Y centres",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        if save_plots:
            final_filename = f"{fig_idx}_strip_hit_occupancy.png"
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format="png", alias="strip_hit_occupancy")
        if show_plots:
            plt.show()
        plt.close()

# ---------------------------------------------------------------------------
# Track-based single-plane efficiency (telescope method)
# Use 3 planes as a telescope, project to the 4th, ask: did the 4th fire?
# ---------------------------------------------------------------------------
if (create_essential_plots or create_plots) and task4_plot_enabled("track_based_efficiency"):
    _eff_cols_need = ("x", "y", "xp", "yp", "definitive_tt")
    _eff_have = all(c in df_plot_ancillary.columns for c in _eff_cols_need)
    if _eff_have:
        z_arr_eff = np.asarray(z_positions, dtype=float)

        # For each plane under test, which trigger-type values form the telescope pool?
        # Pool = events where the OTHER 3 planes all fired (3-plane + 4-plane events).
        # "fired" = definitively_tt == 1234 (the test plane also fired).
        plane_pool_tt = {
            1: [234, 1234],
            2: [134, 1234],
            3: [124, 1234],
            4: [123, 1234],
        }

        fig, axes = plt.subplots(3, 4, figsize=(18, 13), squeeze=False)

        x_all = pd.to_numeric(df_plot_ancillary["x"], errors="coerce").to_numpy(dtype=float)
        y_all = pd.to_numeric(df_plot_ancillary["y"], errors="coerce").to_numpy(dtype=float)
        xp_all = pd.to_numeric(df_plot_ancillary["xp"], errors="coerce").to_numpy(dtype=float)
        yp_all = pd.to_numeric(df_plot_ancillary["yp"], errors="coerce").to_numpy(dtype=float)
        dtt_all = pd.to_numeric(df_plot_ancillary["definitive_tt"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)

        n_x_bins = 15
        n_y_bins = 20
        x_bins = np.linspace(-strip_half, strip_half, n_x_bins + 1)
        y_bins = np.linspace(-width_half, width_half, n_y_bins + 1)

        for plane_idx, p in enumerate(range(1, 5)):
            ax_2d  = axes[0][plane_idx]
            ax_1dy = axes[1][plane_idx]
            ax_1dx = axes[2][plane_idx]

            pool_tt_vals = plane_pool_tt[p]
            pool_mask = np.isin(dtt_all, pool_tt_vals)

            if pool_mask.sum() < 20:
                ax_2d.set_visible(False)
                ax_1dy.set_visible(False)
                ax_1dx.set_visible(False)
                continue

            z_test = float(z_arr_eff[p - 1])
            x_pred = x_all[pool_mask] + xp_all[pool_mask] * z_test
            y_pred = y_all[pool_mask] + yp_all[pool_mask] * z_test
            fired = (dtt_all[pool_mask] == 1234).astype(float)

            # Restrict to detector acceptance
            in_acc = (
                np.isfinite(x_pred) & np.isfinite(y_pred)
                & (np.abs(x_pred) <= strip_half)
                & (np.abs(y_pred) <= width_half)
            )
            x_pred = x_pred[in_acc]
            y_pred = y_pred[in_acc]
            fired = fired[in_acc]

            if len(x_pred) < 10:
                ax_2d.set_visible(False)
                ax_1dy.set_visible(False)
                ax_1dx.set_visible(False)
                continue

            overall_eff = float(fired.mean()) * 100
            n_denom = len(fired)
            y_ctrs = y_pos_P1_and_P3 if (p - 1) % 2 == 0 else y_pos_P2_and_P4

            # --- 2D efficiency map ---
            num_2d, _, _ = np.histogram2d(
                x_pred[fired > 0.5], y_pred[fired > 0.5], bins=[x_bins, y_bins]
            )
            den_2d, _, _ = np.histogram2d(x_pred, y_pred, bins=[x_bins, y_bins])
            with np.errstate(invalid="ignore", divide="ignore"):
                eff_2d = np.where(den_2d > 0, num_2d / den_2d, np.nan)

            im = ax_2d.imshow(
                eff_2d.T,
                origin="lower",
                aspect="auto",
                extent=[-strip_half, strip_half, -width_half, width_half],
                vmin=0,
                vmax=1,
                cmap="RdYlGn",
            )
            plt.colorbar(im, ax=ax_2d, label="efficiency")
            for sy in y_ctrs:
                ax_2d.axhline(sy, color="cyan", lw=0.7, ls="--", alpha=0.7)
            ax_2d.set_xlabel("Projected X (mm)", fontsize=8)
            ax_2d.set_ylabel("Projected Y (mm)", fontsize=8)
            ax_2d.set_title(
                f"Plane {p}  eff={overall_eff:.1f}%  (n={n_denom})", fontsize=9
            )

            # --- 1D efficiency vs projected Y ---
            num_1dy, _ = np.histogram(y_pred[fired > 0.5], bins=y_bins)
            den_1dy, _ = np.histogram(y_pred, bins=y_bins)
            y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
            with np.errstate(invalid="ignore", divide="ignore"):
                eff_1dy = np.where(den_1dy > 0, num_1dy / den_1dy, np.nan)
                err_1dy = np.where(
                    den_1dy > 0,
                    np.sqrt(np.maximum(eff_1dy * (1.0 - eff_1dy) / np.maximum(den_1dy, 1), 0)),
                    np.nan,
                )
            valid_y = np.isfinite(eff_1dy) & (den_1dy > 0)
            median_eff_y = float(np.nanmedian(eff_1dy[valid_y]))
            ax_1dy.errorbar(
                y_centers[valid_y], eff_1dy[valid_y], yerr=err_1dy[valid_y],
                fmt="o-", ms=4, color=f"C{p - 1}", alpha=0.85,
            )
            for sy in y_ctrs:
                ax_1dy.axvline(sy, color="lightgray", lw=0.9, ls="--", alpha=0.8)
            ax_1dy.axhline(
                median_eff_y, color="red", lw=0.8, ls="--", alpha=0.6,
                label=f"median {median_eff_y * 100:.1f}%",
            )
            ax_1dy.set_ylim(0, 1.08)
            ax_1dy.set_xlim(-width_half, width_half)
            ax_1dy.set_xlabel("Projected Y (mm)", fontsize=8)
            ax_1dy.set_ylabel("Efficiency", fontsize=8)
            ax_1dy.legend(fontsize=7)
            ax_1dy.grid(True, alpha=0.3)

            # --- 1D efficiency vs projected X ---
            num_1dx, _ = np.histogram(x_pred[fired > 0.5], bins=x_bins)
            den_1dx, _ = np.histogram(x_pred, bins=x_bins)
            x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
            with np.errstate(invalid="ignore", divide="ignore"):
                eff_1dx = np.where(den_1dx > 0, num_1dx / den_1dx, np.nan)
                err_1dx = np.where(
                    den_1dx > 0,
                    np.sqrt(np.maximum(eff_1dx * (1.0 - eff_1dx) / np.maximum(den_1dx, 1), 0)),
                    np.nan,
                )
            valid_x = np.isfinite(eff_1dx) & (den_1dx > 0)
            median_eff_x = float(np.nanmedian(eff_1dx[valid_x]))
            ax_1dx.errorbar(
                x_centers[valid_x], eff_1dx[valid_x], yerr=err_1dx[valid_x],
                fmt="o-", ms=4, color=f"C{p - 1}", alpha=0.85,
            )
            ax_1dx.axhline(
                median_eff_x, color="red", lw=0.8, ls="--", alpha=0.6,
                label=f"median {median_eff_x * 100:.1f}%",
            )
            ax_1dx.set_ylim(0, 1.08)
            ax_1dx.set_xlim(-strip_half, strip_half)
            ax_1dx.set_xlabel("Projected X (mm)", fontsize=8)
            ax_1dx.set_ylabel("Efficiency", fontsize=8)
            ax_1dx.legend(fontsize=7)
            ax_1dx.grid(True, alpha=0.3)

        plt.suptitle(
            "Track-based single-plane efficiency (telescope method)\n"
            "3 planes build a track → project to test plane → did the test plane fire?",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        if save_plots:
            final_filename = f"{fig_idx}_track_based_efficiency.png"
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format="png", alias="track_based_efficiency")
        if show_plots:
            plt.show()
        plt.close()

# ---------------------------------------------------------------------------
# Track-based efficiency vs theta — 4 planes on one figure
# Same telescope method; reveals angular efficiency dependence for
# comparison between real data and simulation.
# ---------------------------------------------------------------------------
if (create_essential_plots or create_plots) and task4_plot_enabled("track_based_efficiency_vs_theta"):
    _teff_need = ("x", "y", "xp", "yp", "theta", "definitive_tt")
    _teff_have = all(c in df_plot_ancillary.columns for c in _teff_need)
    if _teff_have:
        z_arr_teff = np.asarray(z_positions, dtype=float)
        plane_pool_tt_t = {1: [234, 1234], 2: [134, 1234], 3: [124, 1234], 4: [123, 1234]}

        x_all_t  = pd.to_numeric(df_plot_ancillary["x"],   errors="coerce").to_numpy(dtype=float)
        y_all_t  = pd.to_numeric(df_plot_ancillary["y"],   errors="coerce").to_numpy(dtype=float)
        xp_all_t = pd.to_numeric(df_plot_ancillary["xp"],  errors="coerce").to_numpy(dtype=float)
        yp_all_t = pd.to_numeric(df_plot_ancillary["yp"],  errors="coerce").to_numpy(dtype=float)
        th_all_t = pd.to_numeric(df_plot_ancillary["theta"], errors="coerce").to_numpy(dtype=float)
        dtt_all_t = pd.to_numeric(df_plot_ancillary["definitive_tt"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)

        n_theta_bins = 20
        theta_lo = float(np.nanpercentile(th_all_t[np.isfinite(th_all_t)], 0.5))
        theta_hi = float(np.nanpercentile(th_all_t[np.isfinite(th_all_t)], 99.5))
        theta_bins_t = np.linspace(theta_lo, theta_hi, n_theta_bins + 1)
        theta_centers = 0.5 * (theta_bins_t[:-1] + theta_bins_t[1:])

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)

        for plane_idx, p in enumerate(range(1, 5)):
            ax = axes[0][plane_idx]
            pool_mask_t = np.isin(dtt_all_t, plane_pool_tt_t[p])
            if pool_mask_t.sum() < 20:
                ax.set_visible(False)
                continue

            z_test_t = float(z_arr_teff[p - 1])
            x_pred_t = x_all_t[pool_mask_t] + xp_all_t[pool_mask_t] * z_test_t
            y_pred_t = y_all_t[pool_mask_t] + yp_all_t[pool_mask_t] * z_test_t
            theta_t  = th_all_t[pool_mask_t]
            fired_t  = (dtt_all_t[pool_mask_t] == 1234).astype(float)

            in_acc_t = (
                np.isfinite(x_pred_t) & np.isfinite(y_pred_t) & np.isfinite(theta_t)
                & (np.abs(x_pred_t) <= strip_half)
                & (np.abs(y_pred_t) <= width_half)
            )
            theta_t = theta_t[in_acc_t]
            fired_t = fired_t[in_acc_t]

            if len(theta_t) < 10:
                ax.set_visible(False)
                continue

            num_th, _ = np.histogram(theta_t[fired_t > 0.5], bins=theta_bins_t)
            den_th, _ = np.histogram(theta_t, bins=theta_bins_t)
            with np.errstate(invalid="ignore", divide="ignore"):
                eff_th = np.where(den_th > 0, num_th / den_th, np.nan)
                err_th = np.where(
                    den_th > 0,
                    np.sqrt(np.maximum(eff_th * (1.0 - eff_th) / np.maximum(den_th, 1), 0)),
                    np.nan,
                )
            valid_th = np.isfinite(eff_th) & (den_th > 0)
            median_eff_th = float(np.nanmedian(eff_th[valid_th]))

            ax.errorbar(
                np.degrees(theta_centers[valid_th]),
                eff_th[valid_th],
                yerr=err_th[valid_th],
                fmt="o-", ms=4, color=f"C{p - 1}", alpha=0.85,
            )
            ax.axhline(
                median_eff_th, color="red", lw=0.8, ls="--", alpha=0.6,
                label=f"median {median_eff_th * 100:.1f}%",
            )
            ax.set_ylim(0, 1.08)
            ax.set_xlabel("θ (deg)", fontsize=9)
            ax.set_ylabel("Efficiency", fontsize=9)
            ax.set_title(f"Plane {p}  (n={int(den_th.sum())})", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Track-based efficiency vs polar angle θ (telescope method)\n"
            "Useful for sim/data comparison of angular efficiency dependence",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.91])
        if save_plots:
            final_filename = f"{fig_idx}_track_based_efficiency_vs_theta.png"
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format="png", alias="track_based_efficiency_vs_theta")
        if show_plots:
            plt.show()
        plt.close()

# ── chi2 vs charge population diagnostic ─────────────────────────────────────
# 3 figures (one per charge metric): rows = missed / hit (separate panels),
# cols = 4 test planes.  Right marginal: chi2 histogram (horizontal, log count,
# shared Y).  Bottom marginal: charge histogram (log count, shared X per col).
# Hexbin (log-norm) per panel so density is visible even at large N.
# -----------------------------------------------------------------------------
if (create_essential_plots or create_plots) and task4_plot_enabled("chi2_charge_populations"):
    _chi2pop_need = ("tim_th_chi", "charge_event",
                     "charge_1", "charge_2", "charge_3", "charge_4",
                     "definitive_tt")
    _chi2pop_have = all(c in df_plot_ancillary.columns for c in _chi2pop_need)
    if not _chi2pop_have:
        print("[chi2_charge_populations] required columns not found, skipping.")
    else:
        _plane_pool_tt = {1: [234, 1234], 2: [134, 1234], 3: [124, 1234], 4: [123, 1234]}
        _tel_planes    = {1: [2, 3, 4],   2: [1, 3, 4],   3: [1, 2, 4],   4: [1, 2, 3]}
        _3pl_tt        = {1: 234, 2: 134, 3: 124, 4: 123}
        _pop_cmaps     = ["Blues", "Reds"]
        _pop_colors    = ["steelblue", "tomato"]
        _pop_labels    = ["Missed  (3-fold: test plane not fired)", "Hit  (4-fold: all planes fired)"]

        # Pre-compute numpy arrays once for the whole ancillary frame
        _chi2_log_all = np.log1p(np.clip(
            df_plot_ancillary["tim_th_chi"].values.astype(float), 0.0, None
        ))
        _qev_all    = df_plot_ancillary["charge_event"].values.astype(float)
        _def_tt_arr = df_plot_ancillary["definitive_tt"].values.astype(float)

        _q_min_by_tp  = {}
        _q_asym_by_tp = {}
        for _p in [1, 2, 3, 4]:
            _tq  = np.column_stack([df_plot_ancillary[f"charge_{q}"].values.astype(float)
                                    for q in _tel_planes[_p]])
            _qmn = np.nanmin(_tq, axis=1)
            _qmx = np.nanmax(_tq, axis=1)
            _den = _qmx + _qmn
            _q_min_by_tp[_p]  = _qmn
            _q_asym_by_tp[_p] = np.where(_den > 0, (_qmx - _qmn) / _den, np.nan)

        # Global chi2 Y limit — same across all scatter panels for easy comparison
        _chi2_finite = _chi2_log_all[np.isfinite(_chi2_log_all) & (_chi2_log_all > 0)]
        _global_chi2_ylim = float(np.nanpercentile(_chi2_finite, 99.5)) if _chi2_finite.size > 0 else 5.0

        _metrics = [
            ("charge_event", "Total charge (a.u.)"),
            ("min_q_tel",    "Min telescope-plane charge (a.u.)"),
            ("q_asym",       r"Charge asymmetry  $(q_\mathrm{max}-q_\mathrm{min})/(q_\mathrm{max}+q_\mathrm{min})$"),
        ]

        for _mi, (_metric_key, _metric_label) in enumerate(_metrics):
            # GridSpec: 3 rows × 5 cols
            #   rows 0-1 = missed/hit scatter;  row 2 = charge marginal histograms
            #   cols 0-3 = test planes;  col 4 = chi2 marginal histogram (spans rows 0-1)
            fig_cp = plt.figure(figsize=(20, 12))
            gs_cp  = GridSpec(
                3, 5, figure=fig_cp,
                width_ratios=[3, 3, 3, 3, 1.1],
                height_ratios=[3, 3, 1.1],
                hspace=0.07, wspace=0.10,
            )
            fig_cp.suptitle(
                rf"$\chi^2$ vs {_metric_label} — population diagnostics" + "\n"
                "Top row: missed (3-fold)  |  Bottom row: hit (4-fold)  |  "
                "Right: χ² marginal (all planes)  |  Bottom: charge marginal",
                fontsize=9,
            )

            _ref_y_ax  = None     # shared chi2 Y axis across all 8 scatter panels
            _col_x_ref = {}       # per-column shared X axis (charge metric)
            _chi2_miss_pool = []  # for right marginal histogram
            _chi2_hit_pool  = []

            for _pi, _p in enumerate([1, 2, 3, 4]):
                _pool_idx = np.where(np.isin(_def_tt_arr, _plane_pool_tt[_p]))[0]
                if _pool_idx.size == 0:
                    continue

                _chi2_p = _chi2_log_all[_pool_idx]
                _xd_p   = (
                    _qev_all[_pool_idx]          if _metric_key == "charge_event"
                    else _q_min_by_tp[_p][_pool_idx]  if _metric_key == "min_q_tel"
                    else _q_asym_by_tp[_p][_pool_idx]
                )
                _tt_p      = _def_tt_arr[_pool_idx]
                _miss_sel  = _tt_p == _3pl_tt[_p]
                _hit_sel   = _tt_p == 1234

                # X limit: 99.5th percentile of the full pool (both populations)
                _valid_pool = np.isfinite(_xd_p) & np.isfinite(_chi2_p)
                _xlim_p = (float(np.nanpercentile(_xd_p[_valid_pool], 99.5))
                           if _valid_pool.sum() > 0 else 1.0)

                for _ri, (_sel, _lbl, _cmap) in enumerate(zip(
                    [_miss_sel, _hit_sel], _pop_labels, _pop_cmaps
                )):
                    _sharey = _ref_y_ax
                    _sharex = _col_x_ref.get(_pi)
                    ax_sc = fig_cp.add_subplot(gs_cp[_ri, _pi], sharey=_sharey, sharex=_sharex)
                    if _ref_y_ax is None:
                        _ref_y_ax = ax_sc
                    if _pi not in _col_x_ref:
                        _col_x_ref[_pi] = ax_sc

                    _valid = _sel & np.isfinite(_xd_p) & np.isfinite(_chi2_p)
                    if _valid.sum() >= 5:
                        ax_sc.hexbin(
                            _xd_p[_valid], _chi2_p[_valid],
                            gridsize=30, cmap=_cmap, bins="log", mincnt=1,
                            extent=(0, _xlim_p, 0, _global_chi2_ylim),
                        )
                        (_chi2_miss_pool if _ri == 0 else _chi2_hit_pool).append(_chi2_p[_valid])

                    # Title: only top row (missed) carries the plane + count info
                    if _ri == 0:
                        ax_sc.set_title(
                            f"Test P{_p}  (pool tt∈{_plane_pool_tt[_p]})\n"
                            f"missed={int(_miss_sel.sum()):,}   hit={int(_hit_sel.sum()):,}",
                            fontsize=7,
                        )
                    if _pi == 0:
                        ax_sc.set_ylabel(
                            r"$\log(1+\chi^2)$" + f"\n[{_lbl.split('  ')[0]}]", fontsize=7
                        )
                    plt.setp(ax_sc.get_xticklabels(), visible=False)
                    ax_sc.tick_params(labelsize=6)

                # Lock X limit for this column (affects all shared-X axes)
                _col_x_ref[_pi].set_xlim(0, _xlim_p)

                # Bottom charge histogram — shared X with scatter column
                ax_qh = fig_cp.add_subplot(gs_cp[2, _pi], sharex=_col_x_ref[_pi])
                _bins_q = np.linspace(0, _xlim_p, 60)
                for _sel, _col in [(_miss_sel, _pop_colors[0]), (_hit_sel, _pop_colors[1])]:
                    _valid_q = _sel & np.isfinite(_xd_p)
                    if _valid_q.sum() > 0:
                        ax_qh.hist(_xd_p[_valid_q], bins=_bins_q,
                                   color=_col, alpha=0.55, histtype="stepfilled")
                ax_qh.set_yscale("log")
                ax_qh.set_xlabel(_metric_label, fontsize=7)
                if _pi == 0:
                    ax_qh.set_ylabel("count", fontsize=7)
                ax_qh.tick_params(labelsize=6, axis="x", rotation=25)
                ax_qh.tick_params(labelsize=6, axis="y")

            # Lock shared chi2 Y axis
            if _ref_y_ax is not None:
                _ref_y_ax.set_ylim(0, _global_chi2_ylim)

            # Right chi2 marginal histograms — one per population row, shared Y with scatter
            _bins_chi2 = np.linspace(0, _global_chi2_ylim, 60)
            _chi2h_ref_ax = None
            for _ri, (_pool_list, _col, _lbl) in enumerate(zip(
                [_chi2_miss_pool, _chi2_hit_pool],
                _pop_colors,
                ["missed", "hit"],
            )):
                ax_chi2h = fig_cp.add_subplot(
                    gs_cp[_ri, 4],
                    sharey=_ref_y_ax,
                    sharex=_chi2h_ref_ax,
                )
                if _chi2h_ref_ax is None:
                    _chi2h_ref_ax = ax_chi2h
                _arr = np.concatenate(_pool_list) if _pool_list else np.array([])
                _arr = _arr[np.isfinite(_arr)]
                if _arr.size > 0:
                    ax_chi2h.hist(_arr, bins=_bins_chi2, orientation="horizontal",
                                  color=_col, alpha=0.65, histtype="stepfilled")
                ax_chi2h.set_xscale("log")
                ax_chi2h.set_title(rf"$\chi^2$ marginal [{_lbl}]", fontsize=7)
                if _ri == 1:
                    ax_chi2h.set_xlabel("count (log)", fontsize=7)
                ax_chi2h.tick_params(labelsize=6)
                plt.setp(ax_chi2h.get_yticklabels(), visible=False)
                plt.setp(ax_chi2h.get_xticklabels(), visible=(_ri == 1))

            if save_plots:
                _chi2pop_fn = f"{fig_idx}_chi2_charge_populations_m{_mi + 1}.png"
                fig_idx += 1
                _chi2pop_path = os.path.join(base_directories["figure_directory"], _chi2pop_fn)
                plot_list.append(_chi2pop_path)
                save_plot_figure(_chi2pop_path, format="png", alias="chi2_charge_populations")
            if show_plots:
                plt.show()
            plt.close()

if create_plots:
    df_filtered = df_plot_ancillary
    # tt_values = sorted(df_filtered['definitive_tt'].dropna().unique(), key=lambda x: int(x))

    tt_values = [12, 23, 34, 123, 234, 1234]
    
    n_tt = len(tt_values)
    ncols = 3
    nrows = (n_tt + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    phi_nbins = 40
    # theta_nbins = int(round(phi_nbins / 2) + 1)
    theta_nbins = 40
    theta_bins = np.linspace(theta_left_filter, theta_right_filter, theta_nbins )
    phi_bins = np.linspace(phi_left_filter, phi_right_filter, phi_nbins)
    colors = plt.cm.turbo

    # Select theta/phi range (optional filtering)
    theta_min, theta_max = theta_left_filter, theta_right_filter    # adjust as needed
    phi_min, phi_max     = phi_left_filter, phi_right_filter        # adjust as needed
    
    vmax_global = (
        df_filtered.groupby('definitive_tt')[['theta', 'phi']]
        .apply(
            lambda df: np.histogram2d(
                df['theta'],
                df['phi'],
                bins=[theta_bins, phi_bins],
            )[0].max()
        )
        .max()
    )
    
    for idx, tt_val in enumerate(tt_values):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]

        df_tt = df_filtered[df_filtered['tracking_tt'] == tt_val]
        theta_vals = df_tt['theta'].dropna()
        phi_vals = df_tt['phi'].dropna()

        # Apply range filtering
        mask = (theta_vals >= theta_min) & (theta_vals <= theta_max) & \
               (phi_vals >= phi_min) & (phi_vals <= phi_max)
        theta_vals = theta_vals[mask]
        phi_vals   = phi_vals[mask]

        if len(theta_vals) < 10 or len(phi_vals) < 10:
            ax.set_visible(False)
            continue

        # Polar plot settings
        fig.delaxes(axes[row_idx][col_idx])  # remove the original non-polar Axes
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)  # add a polar Axes
        axes[row_idx][col_idx] = ax  # update reference for consistency

        ax.set_facecolor(colors(0.0))  # darkest background in colormap

        # 2D histogram: use phi as angle, theta as radius
        h, r_edges, phi_edges = np.histogram2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins])
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        R, PHI = np.meshgrid(r_centers, phi_centers, indexing='ij')
        c = ax.pcolormesh(PHI, R, h, cmap='viridis', vmin=0, vmax=vmax_global)
        local_max = h.max()
        cb = fig.colorbar(c, ax=ax, pad=0.1)
        cb.ax.hlines(local_max, *cb.ax.get_xlim(), colors='white', linewidth=2, linestyles='dashed')

    plt.suptitle(r'2D Histogram of $\theta$ vs. $\phi$ for each tracking_tt Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_polar_theta_phi_tracking_tt_2D.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

# -----------------------------------------------------------------------------------------------------------------------------

if create_plots:

    def plot_hexbin_matrix(df, columns_of_interest, filter_conditions, title, save_plots, show_plots, base_directories, fig_idx, plot_list, num_bins=40):
        
        axis_limits = {
            # Static
            'x': [-pos_filter, pos_filter],
            'y': [-pos_filter, pos_filter],
            'det_x': [-pos_filter, pos_filter],
            'det_y': [-pos_filter, pos_filter],
            'theta': [theta_left_filter, theta_right_filter],
            'phi': [phi_left_filter, phi_right_filter],
            'det_theta': [det_theta_left_filter, det_theta_right_filter],
            'det_phi': [det_phi_left_filter, det_phi_right_filter],
            'xp': [-1 * proj_filter, proj_filter],
            'yp': [-1 * proj_filter, proj_filter],
            's': [slowness_filter_left, slowness_filter_right],
            'det_s': [det_slowness_filter_left, det_slowness_filter_right],
            'delta_s': [delta_s_left, delta_s_right],
            # 'th_chi': [0, 0.03],
            # 'det_th_chi': [0, 12],
            
            # Dinamic
            'charge_event': [charge_plot_limit_left, charge_plot_event_limit_right],
            'charge_1': [charge_plot_limit_left, charge_plot_limit_right],
            'charge_2': [charge_plot_limit_left, charge_plot_limit_right],
            'charge_3': [charge_plot_limit_left, charge_plot_limit_right],
            'charge_4': [charge_plot_limit_left, charge_plot_limit_right],
            'res_ystr_1': [-res_ystr_filter, res_ystr_filter], 'res_ystr_2': [-res_ystr_filter, res_ystr_filter], 'res_ystr_3': [-res_ystr_filter, res_ystr_filter], 'res_ystr_4': [-res_ystr_filter, res_ystr_filter],
            'res_tsum_1': [-res_tsum_filter, res_tsum_filter], 'res_tsum_2': [-res_tsum_filter, res_tsum_filter], 'res_tsum_3': [-res_tsum_filter, res_tsum_filter], 'res_tsum_4': [-res_tsum_filter, res_tsum_filter],
            'res_tdif_1': [-res_tdif_filter, res_tdif_filter], 'res_tdif_2': [-res_tdif_filter, res_tdif_filter], 'res_tdif_3': [-res_tdif_filter, res_tdif_filter], 'res_tdif_4': [-res_tdif_filter, res_tdif_filter],
            'det_res_ystr_1': [-det_res_ystr_filter, det_res_ystr_filter], 'det_res_ystr_2': [-det_res_ystr_filter, det_res_ystr_filter], 'det_res_ystr_3': [-det_res_ystr_filter, det_res_ystr_filter], 'det_res_ystr_4': [-det_res_ystr_filter, det_res_ystr_filter],
            'det_res_tsum_1': [-det_res_tsum_filter, det_res_tsum_filter], 'det_res_tsum_2': [-det_res_tsum_filter, det_res_tsum_filter], 'det_res_tsum_3': [-det_res_tsum_filter, det_res_tsum_filter], 'det_res_tsum_4': [-det_res_tsum_filter, det_res_tsum_filter],
            'det_res_tdif_1': [-det_res_tdif_filter, det_res_tdif_filter], 'det_res_tdif_2': [-det_res_tdif_filter, det_res_tdif_filter], 'det_res_tdif_3': [-det_res_tdif_filter, det_res_tdif_filter], 'det_res_tdif_4': [-det_res_tdif_filter, det_res_tdif_filter],
            'ext_res_ystr_1': [-ext_res_ystr_filter, ext_res_ystr_filter], 'ext_res_ystr_2': [-ext_res_ystr_filter, ext_res_ystr_filter], 'ext_res_ystr_3': [-ext_res_ystr_filter, ext_res_ystr_filter], 'ext_res_ystr_4': [-ext_res_ystr_filter, ext_res_ystr_filter],
            'ext_res_tsum_1': [-ext_res_tsum_filter, ext_res_tsum_filter], 'ext_res_tsum_2': [-ext_res_tsum_filter, ext_res_tsum_filter], 'ext_res_tsum_3': [-ext_res_tsum_filter, ext_res_tsum_filter], 'ext_res_tsum_4': [-ext_res_tsum_filter, ext_res_tsum_filter],
            'ext_res_tdif_1': [-ext_res_tdif_filter, ext_res_tdif_filter], 'ext_res_tdif_2': [-ext_res_tdif_filter, ext_res_tdif_filter], 'ext_res_tdif_3': [-ext_res_tdif_filter, ext_res_tdif_filter], 'ext_res_tdif_4': [-ext_res_tdif_filter, ext_res_tdif_filter],
            'det_ext_res_ystr_1': [-det_ext_res_ystr_filter, det_ext_res_ystr_filter], 'det_ext_res_ystr_2': [-det_ext_res_ystr_filter, det_ext_res_ystr_filter], 'det_ext_res_ystr_3': [-det_ext_res_ystr_filter, det_ext_res_ystr_filter], 'det_ext_res_ystr_4': [-det_ext_res_ystr_filter, det_ext_res_ystr_filter],
            'det_ext_res_tsum_1': [-det_ext_res_tsum_filter, det_ext_res_tsum_filter], 'det_ext_res_tsum_2': [-det_ext_res_tsum_filter, det_ext_res_tsum_filter], 'det_ext_res_tsum_3': [-det_ext_res_tsum_filter, det_ext_res_tsum_filter], 'det_ext_res_tsum_4': [-det_ext_res_tsum_filter, det_ext_res_tsum_filter],
            'det_ext_res_tdif_1': [-det_ext_res_tdif_filter, det_ext_res_tdif_filter], 'det_ext_res_tdif_2': [-det_ext_res_tdif_filter, det_ext_res_tdif_filter], 'det_ext_res_tdif_3': [-det_ext_res_tdif_filter, det_ext_res_tdif_filter], 'det_ext_res_tdif_4': [-det_ext_res_tdif_filter, det_ext_res_tdif_filter],
        }
        
        # Apply filters
        for col, min_val, max_val in filter_conditions:
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]
        
        num_var = len(columns_of_interest)
        fig, axes = plt.subplots(num_var, num_var, figsize=(15, 15))
        
        auto_limits = {}
        for col in columns_of_interest:
            if col in axis_limits:
                auto_limits[col] = axis_limits[col]
            else:
                auto_limits[col] = [df[col].min(), df[col].max()]
        
        for i in range(num_var):
            for j in range(num_var):
                ax = axes[i, j]
                x_col = columns_of_interest[j]
                y_col = columns_of_interest[i]
                
                if i < j:
                    ax.axis('off')  # Leave the lower triangle blank
                elif i == j:
                    # Diagonal: 1D histogram
                    hist_data = df[x_col]
                    # Remove nans
                    hist_data = hist_data[~np.isnan(hist_data)]
                    # Remove zeroes
                    hist_data = hist_data[hist_data != 0]
                    hist, bins = np.histogram(hist_data, bins=num_bins)
                    bin_centers = 0.5 * (bins[1:] + bins[:-1])
                    norm = plt.Normalize(hist.min(), hist.max())
                    cmap = plt.get_cmap('turbo')
                    
                    for k in range(len(hist)):
                        ax.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(norm(hist[k])))
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(auto_limits[x_col])
                    
                    # If the column is 'charge_1, 2, 3 or 4', set logscale in Y
                    if x_col.startswith('charge'):
                        ax.set_yscale('log')
                    
                else:
                    # Upper triangle: hexbin
                    x_data = df[x_col]
                    y_data = df[y_col]
                    # Remove zeroes and nans
                    cond = (x_data != 0) & (y_data != 0) & (~np.isnan(x_data)) & (~np.isnan(y_data))
                    x_data = x_data[cond]
                    y_data = y_data[cond]
                    ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')
                    ax.set_facecolor(plt.cm.turbo(0))
                    
                    if "det_s" in x_col and "s" in x_col or "s" in y_col and "det_s" in y_col:
                        # Draw a line in the diagonal y = x
                        line_x = np.linspace(-0.01, 0.015, 100)
                        line_y = line_x
                        ax.plot(line_x, line_y, color='white', linewidth=1)  # Thin white line
                    
                    square_x = [-150, 150, 150, -150, -150]  # Closing the loop
                    square_y = [-150, -150, 150, 150, -150]
                    ax.plot(square_x, square_y, color='white', linewidth=1)  # Thin white line
                    
                    # Apply determined limits
                    ax.set_xlim(auto_limits[x_col])
                    ax.set_ylim(auto_limits[y_col])
                
                if i != num_var - 1:
                    ax.set_xticklabels([])
                if j != 0:
                    ax.set_yticklabels([])
                if i == num_var - 1:
                    ax.set_xlabel(x_col)
                if j == 0:
                    ax.set_ylabel(y_col)
        
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.suptitle(title)
        if save_plots:
            name_of_file = 'timtrack_results_hexbin_combination_projections'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        # Show plot if enabled
        if show_plots:
            plt.show()
        plt.close()
        return fig_idx

    # df_cases_2 = [
    #     ([("processed_tt", 12, 12)], "1-2 cases"),
    #     ([("processed_tt", 23, 23)], "2-3 cases"),
    #     ([("processed_tt", 34, 34)], "3-4 cases"),
    #     ([("processed_tt", 13, 13)], "1-3 cases"),
    #     ([("processed_tt", 14, 14)], "1-4 cases"),
    #     ([("processed_tt", 123, 123)], "1-2-3 cases"),
    #     ([("processed_tt", 234, 234)], "2-3-4 cases"),
    #     ([("processed_tt", 124, 124)], "1-2-4 cases"),
    #     ([("processed_tt", 134, 134)], "1-3-4 cases"),
    #     ([("processed_tt", 1234, 1234)], "1-2-3-4 cases"),
    # ]
    
    
    # df_cases_2 = [
    #     ([("tracking_tt", 12, 12)], "1-2 cases"),
    #     ([("tracking_tt", 23, 23)], "2-3 cases"),
    #     ([("tracking_tt", 34, 34)], "3-4 cases"),
    #     ([("tracking_tt", 123, 123)], "1-2-3 cases"),
    #     ([("tracking_tt", 234, 234)], "2-3-4 cases"),
    #     ([("tracking_tt", 1234, 1234)], "1-2-3-4 cases"),
    # ]
    
    df_cases_1 = [
        ([("definitive_tt", 12, 12)], "1-2 cases"),
        ([("definitive_tt", 23, 23)], "2-3 cases"),
        ([("definitive_tt", 34, 34)], "3-4 cases"),
        ([("definitive_tt", 123, 123)], "1-2-3 cases"),
        ([("definitive_tt", 234, 234)], "2-3-4 cases"),
        ([("definitive_tt", 1234, 1234)], "1-2-3-4 cases"),
        ([("definitive_tt", 13, 13)], "1-3 cases"),
        ([("definitive_tt", 14, 14)], "1-4 cases"),
        ([("definitive_tt", 124, 124)], "1-2-4 cases"),
        ([("definitive_tt", 134, 134)], "1-3-4 cases"),
    ]
    
    df_cases_2 = [
        ([("definitive_tt", 1234, 1234)], "1-2-3-4 cases"),
        ([("definitive_tt", 123, 123)], "1-2-3 cases"),
        ([("definitive_tt", 234, 234)], "2-3-4 cases"),
        ([("definitive_tt", 124, 124)], "1-2-4 cases"),
        ([("definitive_tt", 134, 134)], "1-3-4 cases"),
    ]
    
    df_cases_3 = [
        ([("definitive_tt", 12, 12), ("iterations", 2, 2)], "1-2 cases, iterations = 2"),
        ([("definitive_tt", 12, 12), ("iterations", 3, 3)], "1-2 cases, iterations = 3"),
        ([("definitive_tt", 12, 12), ("iterations", 4, 4)], "1-2 cases, iterations = 4"),
        ([("definitive_tt", 12, 12), ("iterations", 5, 5)], "1-2 cases, iterations = 5"),
        ([("definitive_tt", 12, 12), ("iterations", 6, iter_max)], f"1-2 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 23, 23), ("iterations", 2, 2)], "2-3 cases, iterations = 2"),
        ([("definitive_tt", 23, 23), ("iterations", 3, 3)], "2-3 cases, iterations = 3"),
        ([("definitive_tt", 23, 23), ("iterations", 4, 4)], "2-3 cases, iterations = 4"),
        ([("definitive_tt", 23, 23), ("iterations", 5, 5)], "2-3 cases, iterations = 5"),
        ([("definitive_tt", 23, 23), ("iterations", 6, iter_max)], f"2-3 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 34, 34), ("iterations", 2, 2)], "3-4 cases, iterations = 2"),
        ([("definitive_tt", 34, 34), ("iterations", 3, 3)], "3-4 cases, iterations = 3"),
        ([("definitive_tt", 34, 34), ("iterations", 4, 4)], "3-4 cases, iterations = 4"),
        ([("definitive_tt", 34, 34), ("iterations", 5, 5)], "3-4 cases, iterations = 5"),
        ([("definitive_tt", 34, 34), ("iterations", 6, iter_max)], f"3-4 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 123, 123), ("iterations", 2, 2)], "1-2-3 cases, iterations = 2"),
        ([("definitive_tt", 123, 123), ("iterations", 3, 3)], "1-2-3 cases, iterations = 3"),
        ([("definitive_tt", 123, 123), ("iterations", 4, 4)], "1-2-3 cases, iterations = 4"),
        ([("definitive_tt", 123, 123), ("iterations", 5, 5)], "1-2-3 cases, iterations = 5"),
        ([("definitive_tt", 123, 123), ("iterations", 6, iter_max)], f"1-2-3 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 234, 234), ("iterations", 2, 2)], "2-3-4 cases, iterations = 2"),
        ([("definitive_tt", 234, 234), ("iterations", 3, 3)], "2-3-4 cases, iterations = 3"),
        ([("definitive_tt", 234, 234), ("iterations", 4, 4)], "2-3-4 cases, iterations = 4"),
        ([("definitive_tt", 234, 234), ("iterations", 5, 5)], "2-3-4 cases, iterations = 5"),
        ([("definitive_tt", 234, 234), ("iterations", 6, iter_max)], f"2-3-4 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 1234, 1234), ("iterations", 2, 2)], "1-2-3-4 cases, iterations = 2"),
        ([("definitive_tt", 1234, 1234), ("iterations", 3, 3)], "1-2-3-4 cases, iterations = 3"),
        ([("definitive_tt", 1234, 1234), ("iterations", 4, 4)], "1-2-3-4 cases, iterations = 4"),
        ([("definitive_tt", 1234, 1234), ("iterations", 5, 5)], "1-2-3-4 cases, iterations = 5"),
        ([("definitive_tt", 1234, 1234), ("iterations", 6, iter_max)], f"1-2-3-4 cases, iterations = 6 to {iter_max}"),
    ]
    
    # df_cases_1 = [
    #     # From original_tt = 1234
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 123, 123)], "original=1234, processed=123"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 124, 124)], "original=1234, processed=124"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 134, 134)], "original=1234, processed=134"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 234, 234)], "original=1234, processed=234"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 12, 12)],   "original=1234, processed=12"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 13, 13)],   "original=1234, processed=13"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 14, 14)],   "original=1234, processed=14"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 23, 23)],   "original=1234, processed=23"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 24, 24)],   "original=1234, processed=24"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 34, 34)],   "original=1234, processed=34"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 1234, 1234)], "original=1234, processed=1234"),

    #     # From original_tt = 124
    #     ([("original_tt", 124, 124), ("definitive_tt", 12, 12)], "original=124, processed=12"),
    #     ([("original_tt", 124, 124), ("definitive_tt", 14, 14)], "original=124, processed=14"),
    #     ([("original_tt", 124, 124), ("definitive_tt", 24, 24)], "original=124, processed=24"),
    #     ([("original_tt", 124, 124), ("definitive_tt", 124, 124)], "original=124, processed=124"),

    #     # From original_tt = 134
    #     ([("original_tt", 134, 134), ("definitive_tt", 13, 13)], "original=134, processed=13"),
    #     ([("original_tt", 134, 134), ("definitive_tt", 14, 14)], "original=134, processed=14"),
    #     ([("original_tt", 134, 134), ("definitive_tt", 34, 34)], "original=134, processed=34"),
    #     ([("original_tt", 134, 134), ("definitive_tt", 134, 134)], "original=134, processed=134"),

    #     # From original_tt = 123
    #     ([("original_tt", 123, 123), ("definitive_tt", 12, 12)], "original=123, processed=12"),
    #     ([("original_tt", 123, 123), ("definitive_tt", 13, 13)], "original=123, processed=13"),
    #     ([("original_tt", 123, 123), ("definitive_tt", 23, 23)], "original=123, processed=23"),
    #     ([("original_tt", 123, 123), ("definitive_tt", 123, 123)], "original=123, processed=123"),

    #     # From original_tt = 234
    #     ([("original_tt", 234, 234), ("definitive_tt", 23, 23)], "original=234, processed=23"),
    #     ([("original_tt", 234, 234), ("definitive_tt", 24, 24)], "original=234, processed=24"),
    #     ([("original_tt", 234, 234), ("definitive_tt", 34, 34)], "original=234, processed=34"),
    #     ([("original_tt", 234, 234), ("definitive_tt", 234, 234)], "original=234, processed=234"),

    #     # From original_tt = 12
    #     ([("original_tt", 12, 12), ("definitive_tt", 12, 12)], "original=12, processed=12"),

    #     # From original_tt = 23
    #     ([("original_tt", 23, 23), ("definitive_tt", 23, 23)], "original=23, processed=23"),

    #     # From original_tt = 34
    #     ([("original_tt", 34, 34), ("definitive_tt", 34, 34)], "original=34, processed=34"),

    #     # From original_tt = 13
    #     ([("original_tt", 13, 13), ("definitive_tt", 13, 13)], "original=13, processed=13"),
    # ]

    # # Charge of each plane -------------------------------------------------------------------
    # for filters, title in df_cases_2:
    #     # Extract the relevant charge numbers from the title (e.g., "1-2 cases" -> [1, 2])
    #     relevant_charges = [f"charge_{n}" for n in map(int, title.split()[0].split('-'))]

    #     # Define the columns - interest dynamically
    #     # columns_of_interest = ['x', 'y', 'theta', 'phi', 'xp', 'yp'] + relevant_charges
    #     columns_of_interest = relevant_charges

    #     # Keep the original filters (if needed) and apply them
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         columns_of_interest,  # Dynamically set the columns to include relevant charges
    #         filters,  # Keep original filters
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )

    # # Residues --------------------------------------------------------------------------------------
    # for filters, title in df_cases_2:
    #     relevant_residues_tsum = [f"res_tsum_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_tdif = [f"res_tdif_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_ystr = [f"res_ystr_{n}" for n in map(int, title.split()[0].split('-'))]
        
    #     columns_of_interest = ['x', 'y', 'theta', 'phi', 'xp', 'yp', 's'] + relevant_residues_tsum + relevant_residues_tdif + relevant_residues_ystr
        
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         columns_of_interest,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    
    # for filters, title in df_cases_2:
    #     relevant_residues_tsum = [f"res_tsum_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_det_tsum = [f"det_res_tsum_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_ext_tsum = [f"ext_res_tsum_{n}" for n in map(int, title.split()[0].split('-'))]
        
    #     columns_of_interest = relevant_residues_tsum + relevant_residues_det_tsum + relevant_residues_ext_tsum
        
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         columns_of_interest,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    # for filters, title in df_cases_2:
    #     relevant_residues_tdif = [f"res_tdif_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_det_tdif = [f"det_res_tdif_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_ext_tdif = [f"ext_res_tdif_{n}" for n in map(int, title.split()[0].split('-'))]
        
    #     columns_of_interest = relevant_residues_tdif + relevant_residues_det_tdif + relevant_residues_ext_tdif
        
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         columns_of_interest,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    # for filters, title in df_cases_2:
    #     relevant_residues_ystr = [f"res_ystr_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_det_ystr = [f"det_res_ystr_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_ext_ystr = [f"ext_res_ystr_{n}" for n in map(int, title.split()[0].split('-'))]
        
    #     columns_of_interest = relevant_residues_ystr + relevant_residues_det_ystr + relevant_residues_ext_ystr
        
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         columns_of_interest,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    
    # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['x', 'y', 'theta', 'phi', 's', 'delta_s', 'det_s', 'det_phi', 'det_theta', 'det_y', 'det_x']
    # for filters, title in df_cases_1:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    # plot_col = ['x', 'y', 'theta', 'phi', 's']
    # plot_col = ['x', 'xp', 'delta_s', 'yp', 'y']
    # for filters, title in df_cases_1:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['x', 'y', 'theta', 'phi', 's', 'delta_s', 'det_s', 'det_phi', 'det_theta', 'det_y', 'det_x']
    # for filters, title in df_cases_3:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    
    # # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['t0', 's', 'delta_s', 'det_s', 'det_s_ordinate']
    # for filters, title in df_cases_3:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    
    # # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['theta', 'det_theta']
    # for filters, title in df_cases_2:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    # A pure theta vs phi map
    plot_col = ['theta', 'phi']
    for filters, title in df_cases_2:
        fig_idx = plot_hexbin_matrix(
            df_plot_ancillary,
            plot_col,
            filters,
            title,
            save_plots,
            show_plots,
            base_directories,
            fig_idx,
            plot_list
        )
    
    
    # df_plot_ancillary_conv = df_plot_ancillary[df_plot_ancillary['converged'] == 1].copy()
    # # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['x', 'y', 'theta', 'phi', 'delta_s']
    # for filters, title in df_cases_1:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    # df_plot_ancillary_conv = df_plot_ancillary[df_plot_ancillary['converged'] == 0].copy()
    # # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['x', 'y', 'theta', 'phi', 'delta_s']
    # for filters, title in df_cases_1:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )

# ------------------------------------------------------------------------------------------------------

# if create_plots:
#     df_filtered = df_plot_ancillary.copy()
#     fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
#     colors = plt.cm.tab10.colors
#     bins = np.linspace(theta_left_filter, theta_right_filter, 150)
#     tt_values = sorted(df_filtered['definitive_tt'].dropna().unique(), key=lambda x: int(x))

#     for row_idx, (theta_col, row_label) in enumerate([('tim_theta', r'$\theta$'), ('det_theta', r'$\theta_{\mathrm{alt}}$')]):
#         ax = axes[row_idx]
#         for i, tt_val in enumerate(tt_values):
#             df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val]
#             theta_vals = df_tt[theta_col].dropna()
#             if len(theta_vals) < 10:
#                 continue
#             label = f'{tt_val}'
#             ax.hist(theta_vals, bins=bins, histtype='step', linewidth=1,
#                     color=colors[i % len(colors)], label=label)
#         ax.set_xlim(theta_left_filter, theta_right_filter)
#         ax.set_xlabel(row_label + r' [rad]')
#         ax.set_ylabel('Counts')
#         ax.set_title(f'{row_label} — Zoom-in')
#         ax.grid(True)
#         if row_idx == 0:
#             ax.legend(title='definitive_tt', fontsize='small')

#     plt.suptitle(r'$\theta$ and $\theta_{\mathrm{alt}}$ (Zoom-in) by Definitive TT Type', fontsize=15)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     if save_plots:
#         final_filename = f'{fig_idx}_theta_det_theta_zoom_definitive_tt.png'
#         fig_idx += 1
#         save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#         plot_list.append(save_fig_path)
#         save_plot_figure(save_fig_path, format='png')
#     if show_plots:
#         plt.show()
#     plt.close()

if create_plots:
    df_filtered = df_plot_ancillary
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    colors = plt.cm.tab10.colors
    bins = np.linspace(theta_left_filter, theta_right_filter, 150)
    tt_values = sorted(df_filtered['tracking_tt'].dropna().unique(), key=lambda x: int(x))

    for row_idx, (theta_col, row_label) in enumerate([('tim_theta', r'$\theta$'), ('det_theta', r'$\theta_{\mathrm{alt}}$')]):
        ax = axes[row_idx]
        for i, tt_val in enumerate(tt_values):
            df_tt = df_filtered[df_filtered['tracking_tt'] == tt_val]
            theta_vals = df_tt[theta_col].dropna()
            if len(theta_vals) < 10:
                continue
            label = f'{tt_val}'
            ax.hist(theta_vals, bins=bins, histtype='step', linewidth=1,
                    color=colors[i % len(colors)], label=label)
        ax.set_xlim(theta_left_filter, theta_right_filter)
        ax.set_xlabel(row_label + r' [rad]')
        ax.set_ylabel('Counts')
        ax.set_title(f'{row_label} — Zoom-in')
        ax.grid(True)
        if row_idx == 0:
            ax.legend(title='tracking_tt', fontsize='small')

    plt.suptitle(r'$\theta$ and $\theta_{\mathrm{alt}}$ (Zoom-in) by Tracking TT Type', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_theta_det_theta_zoom_tracking_tt.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

del df_plot_ancillary
gc.collect()

# if create_plots:

#     column_chosen = "definitive_tt"
#     plot_ancillary_df = working_df.copy()
    
#     # Ensure datetime is proper and indexed
#     plot_ancillary_df['datetime'] = pd.to_datetime(plot_ancillary_df['datetime'], errors='coerce')
#     plot_ancillary_df = plot_ancillary_df.set_index('datetime')

#     # Prepare a container for each group: 2-plane, 3-plane, 4-plane cases
#     grouped_data = {
#         "Two planes": defaultdict(list),
#         "Three planes": defaultdict(list),
#         "Four planes": defaultdict(list)
#     }

#     # Classify events by number of planes in original_tt
#     for tt_code in plot_ancillary_df[column_chosen].unique():
#         planes = str(tt_code)
#         count = len(planes)
#         label = f'Case {tt_code}'
#         if count == 1:
#             grouped_data["One plane"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
#         if count == 2:
#             grouped_data["Two planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
#         elif count == 3:
#             grouped_data["Three planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
#         elif count == 4:
#             grouped_data["Four planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]

#     # Plotting
#     fig, axes = plt.subplots(1, 3, figsize=(24, 6))
#     colors = plt.colormaps['tab10']

#     for ax, (title, group_dict) in zip(axes, grouped_data.items()):
#         for i, (label, df) in enumerate(group_dict.items()):
#             df.index = pd.to_datetime(df.index, errors='coerce')
#             event_times = df.index.floor('s')
#             full_range = pd.date_range(start=event_times.min(), end=event_times.max(), freq='s')
#             events_per_second = event_times.value_counts().reindex(full_range, fill_value=0).sort_index()
            
#             hist_data = events_per_second.value_counts().sort_index()
#             lambda_estimate = events_per_second.mean()
#             x_values = np.arange(0, hist_data.index.max() + 1)
#             poisson_pmf = poisson.pmf(x_values, lambda_estimate)
#             poisson_pmf_scaled = poisson_pmf * len(events_per_second)

#             ax.plot(hist_data.index, hist_data.values, label=label, alpha=0.9, color=colors(i % 10), linewidth = 3)
#             ax.plot(x_values, poisson_pmf_scaled, '--', lw=1.5, color=colors(i % 10), alpha=0.6)
#             ax.set_xlim(0, 8)

#         ax.set_title(f'{title}')
#         ax.set_xlabel('Number of Events per Second')
#         ax.set_ylabel('Frequency')
#         ax.legend(fontsize='small', loc='upper right')
#         ax.grid(True)

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.88)
#     plt.suptitle('Event Rate Histograms by Original_tt Cardinality with Poisson Fits', fontsize=16)

#     if save_plots:
#         final_filename = f'{fig_idx}_events_per_second_by_plane_cardinality_definitive_tt.png'
#         fig_idx += 1
#         save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#         plot_list.append(save_fig_path)
#         save_plot_figure(save_fig_path, format='png')
#     if show_plots:
#         plt.show()
#     plt.close()

if create_plots:

    fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharey=True)
    colors = plt.colormaps['tab10']
    tt_types = ['raw_tt', 'definitive_tt']
    row_titles = ['Raw TT', 'Processed TT']

    for row_idx, column_chosen in enumerate(tt_types):
        plot_ancillary_df = working_df.copy()

        # Ensure datetime is proper and indexed
        plot_ancillary_df['datetime'] = pd.to_datetime(plot_ancillary_df['datetime'], errors='coerce')
        plot_ancillary_df = plot_ancillary_df.set_index('datetime')

        grouped_data = {
            "Two planes": defaultdict(list),
            "Three planes": defaultdict(list),
            "Four planes": defaultdict(list)
        }

        lambda_store = {}

        for tt_code in plot_ancillary_df[column_chosen].dropna().unique():
            planes = str(tt_code)
            count = len(planes)
            label = f'Case {tt_code}'
            if count == 2:
                grouped_data["Two planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
            elif count == 3:
                if tt_code in (124, 134):
                    grouped_data["Four planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
                else:
                    grouped_data["Three planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
            elif count == 4:
                grouped_data["Four planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]

        for col_idx, (title, group_dict) in enumerate(grouped_data.items()):
            ax = axes[row_idx, col_idx]
            for i, (label, df) in enumerate(group_dict.items()):
                df.index = pd.to_datetime(df.index, errors='coerce')
                event_times = df.index.floor('s')
                full_range = pd.date_range(start=event_times.min(), end=event_times.max(), freq='s')
                events_per_second = event_times.value_counts().reindex(full_range, fill_value=0).sort_index()

                hist_data = events_per_second.value_counts().sort_index()
                lambda_estimate = events_per_second.mean()
                lambda_store[label] = lambda_estimate
                x_values = np.arange(0, hist_data.index.max() + 1)
                poisson_pmf = poisson.pmf(x_values, lambda_estimate)
                poisson_pmf_scaled = poisson_pmf * len(events_per_second)

                ax.plot(hist_data.index, hist_data.values, label=label, alpha=0.9, color=colors(i % 10), linewidth=3)
                ax.plot(x_values, poisson_pmf_scaled, '--', lw=1.5, color=colors(i % 10), alpha=0.6)
                ax.set_xlim(0, 8)

            ax.set_title(f'{title} ({row_titles[row_idx]})')
            ax.set_xlabel('Number of Events per Second')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize='small', loc='upper right')
            ax.grid(True)

        # Annotate efficiency estimates on the rightmost plot
        ax_right = axes[row_idx, 2]
        lam_1234 = lambda_store.get('Case 1234', np.nan)
        lam_124 = lambda_store.get('Case 124', np.nan)
        lam_134 = lambda_store.get('Case 134', np.nan)
        def safe_ratio(num, den):
            return np.nan if (den is None or den == 0 or np.isnan(den)) else 1 - (num / den) if num is not None else np.nan
        eff_plane3 = safe_ratio(lam_124, lam_1234)
        eff_plane2 = safe_ratio(lam_134, lam_1234)
        text_lines = [
            f"λ1234 = {lam_1234:.3g}" if not np.isnan(lam_1234) else "λ1234 = n/a",
            f"λ124  = {lam_124:.3g}" if not np.isnan(lam_124) else "λ124  = n/a",
            f"λ134  = {lam_134:.3g}" if not np.isnan(lam_134) else "λ134  = n/a",
            f"1 - λ134/λ1234 (eff P2) = {eff_plane2:.3g}" if not np.isnan(eff_plane2) else "1 - λ134/λ1234 (eff P2) = n/a",
            f"1 - λ124/λ1234 (eff P3) = {eff_plane3:.3g}" if not np.isnan(eff_plane3) else "1 - λ124/λ1234 (eff P3) = n/a",
        ]
        ax_right.text(0.02, 0.98, "\n".join(text_lines), transform=ax_right.transAxes,
                      va='top', ha='left', fontsize='small',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Event Rate Histograms by TT Type and Plane Cardinality with Poisson Fits', fontsize=18)

    # Save and show
    if save_plots:
        final_filename = f'{fig_idx}_events_per_second_by_plane_cardinality_double_row.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        save_plot_figure(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()

_prof["s_fitting_loop_s"] = round(time.perf_counter() - _t_sec, 2)
_t_sec = time.perf_counter()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("-------------------------- Save and finish ---------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Round to 4 significant digits -----------------------------------------------
def round_array_to_significant_digits(values: np.ndarray, significant_digits: int = 4) -> np.ndarray:
    rounded = values.astype(float, copy=True)
    finite_nonzero = np.isfinite(rounded) & (rounded != 0)
    if np.any(finite_nonzero):
        magnitudes = np.floor(np.log10(np.abs(rounded[finite_nonzero]))).astype(int)
        scales = np.power(10.0, significant_digits - 1 - magnitudes)
        rounded[finite_nonzero] = np.round(rounded[finite_nonzero] * scales) / scales
    return rounded

print("Rounding the dataframe values.") 
for col in working_df.select_dtypes(include=[np.floating]).columns:
    original_dtype = working_df[col].dtype
    column_values = working_df[col].to_numpy(dtype=float, copy=False)
    rounded_values = round_array_to_significant_digits(column_values, significant_digits=4)
    working_df.loc[:, col] = rounded_values.astype(original_dtype, copy=False)

# Save the data ---------------------------------------------------------------
# if save_full_data: # Save a full version of the data, for different studies and debugging
#     working_df.to_csv(save_full_path, index=False, sep=',', float_format='%.5g')
#     print(f"Datafile saved in {save_full_filename}.")

# Save the main columns, relevant for the posterior analysis ------------------
missing_charge_cols = []
for module in ['1', '2', '3', '4']:
    for strip in range(1, 5):
        no_crstlk = f'Q{module}_Q_sum_{strip}_no_crstlk'
        with_crstlk = f'Q{module}_Q_sum_{strip}_with_crstlk'
        if no_crstlk not in working_df.columns:
            working_df[no_crstlk] = np.nan
            missing_charge_cols.append(no_crstlk)
        if with_crstlk not in working_df.columns:
            working_df[with_crstlk] = np.nan
            missing_charge_cols.append(with_crstlk)

if missing_charge_cols:
    unique_missing = sorted(set(missing_charge_cols))
    print(
        "Warning: missing charge columns; created with NaN defaults: "
        + ", ".join(unique_missing)
    )

for i, module in enumerate(['1', '2', '3', '4']):
    for j in range(4):
        strip = j + 1
        working_df[f'Q_P{module}s{strip}'] = working_df[f'Q{module}_Q_sum_{strip}_no_crstlk']
        working_df[f'Q_P{module}s{strip}_with_crstlk'] = working_df[f'Q{module}_Q_sum_{strip}_with_crstlk']

if self_trigger:
    for i, module in enumerate(['1', '2', '3', '4']):
        for j in range(4):
            strip = j + 1
            source_col = f'Q{module}_Q_sum_{strip}'
            if source_col not in working_st_df.columns:
                working_st_df[source_col] = np.nan
                print(f"Warning: missing self-trigger charge column; created NaN: {source_col}")
            working_st_df[f'Q_P{module}s{strip}'] = working_st_df[source_col]

# Charge checking --------------------------------------------------------------------------------------------------------
if self_trigger:
    if create_plots:
   
        fig, axs = plt.subplots(4, 4, figsize=(18, 12))
        for i in range(1, 5):
            for j in range(1, 5):
                # Get the column name
                col_name = f"Q_P{i}s{j}"
                col_name_2 = f"Q_P{i}s{j}_with_crstlk"
                
                # Plot the histogram
                v = working_df[col_name]
                v = v[v != 0]
                w = working_df[col_name_2]
                w = w[w != 0]
                
                # For 'no crosstalk' histogram
                counts_v, bins_v = np.histogram(v, bins=80, range=(0, 40))
                normalized_v = counts_v / max(counts_v)
                axs[i-1, j-1].stairs(normalized_v, bins_v, alpha=0.5, label='no crosstalk', color='blue', fill=True)

                # For 'with crosstalk' histogram (if uncommented)
                # counts_w, bins_w = np.histogram(w, bins=80, range=(0, 40))
                # normalized_w = counts_w / max(counts_w)
                # axs[i-1, j-1].stairs(normalized_w, bins_w, alpha=0.5, label='with crosstalk', color='orange', fill=True)

                if self_trigger:
                    x = working_st_df[col_name]
                    x = x[x != 0]
                    counts_x, bins_x = np.histogram(x, bins=40, range=(0, 40))
                    normalized_x = counts_x / max(counts_x)
                    axs[i-1, j-1].stairs(normalized_x, bins_x, alpha=0.5, label='self-trigger', color='orange', fill=True)
                
                axs[i-1, j-1].set_title(col_name)
                axs[i-1, j-1].set_xlabel("Charge / ns")
                axs[i-1, j-1].set_ylabel("Frequency")
                axs[i-1, j-1].grid(True)
                    
                if i == j == 4:
                    axs[i-1, j-1].legend(loc='upper right')
        
        plt.suptitle("Event and self trigger charge spectra comparison")
        plt.tight_layout()
        figure_name = f"all_channels_charge_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()

if self_trigger:
    if create_plots:
   
        fig, axs = plt.subplots(4, 4, figsize=(18, 12))
        # Filter once outside the loop; read-only inside so no copy needed.
        plot_def_df = working_df.loc[working_df["definitive_tt"] == "1234"]
        for i in range(1, 5):
            for j in range(1, 5):
                # Get the column name
                col_name = f"Q_P{i}s{j}"
                col_name_2 = f"Q_P{i}s{j}_with_crstlk"

                # Plot the histogram
                v = plot_def_df[col_name]
                v = v[v != 0]
                w = plot_def_df[col_name_2]
                w = w[w != 0]
                
                # For 'no crosstalk' histogram
                counts_v, bins_v = np.histogram(v, bins=80, range=(0, 40))
                normalized_v = counts_v / max(counts_v)
                axs[i-1, j-1].stairs(normalized_v, bins_v, alpha=0.5, label='no crosstalk', color='blue', fill=True)

                # For 'with crosstalk' histogram (if uncommented)
                # counts_w, bins_w = np.histogram(w, bins=80, range=(0, 40))
                # normalized_w = counts_w / max(counts_w)
                # axs[i-1, j-1].stairs(normalized_w, bins_w, alpha=0.5, label='with crosstalk', color='orange', fill=True)

                if self_trigger:
                    x = working_st_df[col_name]
                    x = x[x != 0]
                    counts_x, bins_x = np.histogram(x, bins=40, range=(0, 40))
                    normalized_x = counts_x / max(counts_x)
                    axs[i-1, j-1].stairs(normalized_x, bins_x, alpha=0.5, label='self-trigger', color='orange', fill=True)
                
                axs[i-1, j-1].set_title(col_name)
                axs[i-1, j-1].set_xlabel("Charge / ns")
                axs[i-1, j-1].set_ylabel("Frequency")
                axs[i-1, j-1].grid(True)
                    
                if i == j == 4:
                    axs[i-1, j-1].legend(loc='upper right')
        
        plt.suptitle("Event (4-fold) and self trigger charge spectra comparison")
        plt.tight_layout()
        figure_name = f"all_channels_charge_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            save_plot_figure(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()
# ------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Update pipeline status CSV with list events metadata
# -----------------------------------------------------------------------------
def _pipeline_strip_suffix(name: str) -> str:
    for suffix in ('.txt', '.csv', '.dat', '.hld.tar.gz', '.hld-tar-gz', '.hld'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name

def _pipeline_compute_start_timestamp(base: str) -> str:
    digits = base[-11:]
    if len(digits) == 11 and digits.isdigit():
        yy = int(digits[:2])
        doy = int(digits[2:5])
        hh = int(digits[5:7])
        mm = int(digits[7:9])
        ss = int(digits[9:11])
        year = 2000 + yy
        try:
            dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
            return dt.strftime('%Y-%m-%d_%H.%M.%S')
        except ValueError:
            return ''
    return ''

# def _update_pipeline_csv_for_list_event() -> None:
#     csv_headers = [
#         'basename',
#         'start_date',
#         'hld_remote_add_date',
#         'hld_local_add_date',
#         'dat_add_date',
#         'list_ev_name',
#         'list_ev_add_date',
#         'acc_name',
#         'acc_add_date',
#         'merge_add_date',
#     ]

#     station_dir = Path(home_path) / 'DATAFLOW_v3' / 'STATIONS' / f'MINGO0{station}'
#     csv_path = station_dir / f'database_status_{station}.csv'
#     csv_path.parent.mkdir(parents=True, exist_ok=True)
#     if not csv_path.exists():
#         with csv_path.open('w', newline='') as handle:
#             writer = csv.writer(handle)
#             writer.writerow(csv_headers)

#     base_name = _pipeline_strip_suffix(os.path.basename(the_filename))
#     list_event_name = save_filename
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     start_value = _pipeline_compute_start_timestamp(base_name)

#     rows: List[dict[str, str]] = []
#     with csv_path.open('r', newline='') as handle:
#         reader = csv.DictReader(handle)
#         rows.extend(reader)

#     found = False
#     for row in rows:
#         if row.get('basename', '') == base_name:
#             found = True
#             if not row.get('start_date') and start_value:
#                 row['start_date'] = start_value
#             row['list_ev_name'] = list_event_name
#             row['list_ev_add_date'] = timestamp
#             break

#     if not found:
#         new_row = {header: '' for header in csv_headers}
#         new_row['basename'] = base_name
#         if start_value:
#             new_row['start_date'] = start_value
#         new_row['list_ev_name'] = list_event_name
#         new_row['list_ev_add_date'] = timestamp
#         rows.append(new_row)

#     # Ensure existing list events on disk are reflected in the CSV
#     list_dir = Path(home_path) / 'DATAFLOW_v3' / 'STATIONS' / f'MINGO0{station}' / 'STAGE_1' / 'EVENT_DATA' / 'LIST_EVENTS_DIRECTORY'
#     existing_names = {row.get('list_ev_name', '') for row in rows}

#     if list_dir.exists():
#         for list_path in sorted(list_dir.glob('list_events_*.txt')):
#             list_name = list_path.name
#             if list_name in existing_names:
#                 continue

#             derived_base = _pipeline_strip_suffix(list_name)
#             derived_start = ''
#             stem = Path(list_name).stem
#             if stem.startswith('list_events_'):
#                 stamp = stem[len('list_events_'):]
#                 try:
#                     dt = datetime.strptime(stamp, '%Y.%m.%d_%H.%M.%S')
#                     derived_start = dt.strftime('%Y-%m-%d_%H.%M.%S')
#                 except ValueError:
#                     derived_start = ''

#             add_timestamp = datetime.fromtimestamp(list_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
#             filler = {header: '' for header in csv_headers}
#             filler['basename'] = derived_base
#             if derived_start:
#                 filler['start_date'] = derived_start
#             filler['list_ev_name'] = list_name
#             filler['list_ev_add_date'] = add_timestamp
#             rows.append(filler)
#             existing_names.add(list_name)

#     with csv_path.open('w', newline='') as handle:
#         writer = csv.DictWriter(handle, fieldnames=csv_headers)
#         writer.writeheader()
#         writer.writerows(rows)

# _update_pipeline_csv_for_list_event()

# -----------------------------------------------------------------------------
# Create and save the PDF -----------------------------------------------------
# -----------------------------------------------------------------------------

# Force PDF creation when Task 4 plotting is enabled.
if create_plots:
    create_pdf = True

if create_pdf:
    print(f"Creating PDF with all plots in {save_pdf_path}")
    existing_pngs = [png for png in plot_list if os.path.exists(png)]

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

        # Remove PNG files after creating the PDF
        for png in existing_pngs:
            try:
                os.remove(png)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")

# Erase all files in the figure_directory -------------------------------------------------
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory) if os.path.exists(figure_directory) else []

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))

# Erase the figure_directory
if os.path.exists(figure_directory):
    print("Removing figure directory...")
    os.rmdir(figure_directory)

# Move the original datafile to COMPLETED -------------------------------------
print("Moving file to COMPLETED directory...")

if user_file_selection == False:
    if os.path.exists(file_path):
        safe_move(file_path, completed_file_path)
        now = time.time()
        os.utime(completed_file_path, (now, now))
        print("************************************************************")
        print(f"File moved from\n{file_path}\nto:\n{completed_file_path}")
        print("************************************************************")
    else:
        print(f"Input file already absent (maybe previously processed): {file_path}")
        print("Skipping move; fitted output stays in OUTPUT_FILES.")

# Store the current time at the end
end_execution_time_counting = datetime.now()
time_taken = (end_execution_time_counting - start_execution_time_counting).total_seconds() / 60
print(f"Time taken for the whole execution: {time_taken:.2f} minutes")

# mark_status_complete(status_csv_path, status_timestamp)

print("----------------------------------------------------------------------")
print("------------------- Finished list_events creation --------------------")
print("----------------------------------------------------------------------\n\n\n")

record_residual_sigmas(working_df)

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

# Compute fit_tt before pruning columns, since it depends on P*_..._final fields.
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
working_df = compute_tt(working_df, "fit_tt", fit_tt_columns)

# Store TimTrack convergence controls in specific metadata for later studies.
global_variables["timtrack_d0"] = float(d0)
global_variables["timtrack_cocut"] = float(cocut)
global_variables["timtrack_iter_max"] = int(iter_max)

# Store TimTrack convergence outcomes for runout-vs-cut studies.
timtrack_outcome_keys = (
    "timtrack_attempted_fit_n",
    "timtrack_itermax_reached_n",
    "timtrack_itermax_reached_ratio",
    "timtrack_itermax_runout_n",
    "timtrack_itermax_runout_ratio",
    "timtrack_converged_on_cocut_n",
    "timtrack_converged_on_cocut_ratio",
)

if run_timtrack_fit and {"tim_iterations", "tim_conv_distance"}.issubset(working_df.columns):
    tim_iterations_arr = pd.to_numeric(
        working_df["tim_iterations"],
        errors="coerce",
    ).to_numpy(dtype=float)
    tim_conv_distance_arr = pd.to_numeric(
        working_df["tim_conv_distance"],
        errors="coerce",
    ).to_numpy(dtype=float)

    attempted_mask = np.isfinite(tim_iterations_arr) & (tim_iterations_arr > 0)
    attempted_count = int(np.count_nonzero(attempted_mask))
    global_variables["timtrack_attempted_fit_n"] = attempted_count

    if attempted_count > 0:
        itermax_reached_mask = attempted_mask & (tim_iterations_arr >= float(iter_max))
        converged_on_cocut_mask = (
            attempted_mask
            & np.isfinite(tim_conv_distance_arr)
            & (tim_conv_distance_arr <= float(cocut))
        )
        itermax_runout_mask = itermax_reached_mask & (~converged_on_cocut_mask)

        itermax_reached_count = int(np.count_nonzero(itermax_reached_mask))
        itermax_runout_count = int(np.count_nonzero(itermax_runout_mask))
        converged_on_cocut_count = int(np.count_nonzero(converged_on_cocut_mask))

        global_variables["timtrack_itermax_reached_n"] = itermax_reached_count
        global_variables["timtrack_itermax_reached_ratio"] = (
            itermax_reached_count / attempted_count
        )
        global_variables["timtrack_itermax_runout_n"] = itermax_runout_count
        global_variables["timtrack_itermax_runout_ratio"] = (
            itermax_runout_count / attempted_count
        )
        global_variables["timtrack_converged_on_cocut_n"] = converged_on_cocut_count
        global_variables["timtrack_converged_on_cocut_ratio"] = (
            converged_on_cocut_count / attempted_count
        )
    else:
        global_variables["timtrack_itermax_reached_n"] = 0
        global_variables["timtrack_itermax_reached_ratio"] = np.nan
        global_variables["timtrack_itermax_runout_n"] = 0
        global_variables["timtrack_itermax_runout_ratio"] = np.nan
        global_variables["timtrack_converged_on_cocut_n"] = 0
        global_variables["timtrack_converged_on_cocut_ratio"] = np.nan
else:
    for metadata_key in timtrack_outcome_keys:
        global_variables[metadata_key] = np.nan

# Detached vs TimTrack comparison metadata:
# median( 2 * abs(detached - timtrack) / (detached + timtrack) )
comparison_pairs = {
    "x": ("det_x", "tim_x"),
    "y": ("det_y", "tim_y"),
    "theta": ("det_theta", "tim_theta"),
    "phi": ("det_phi", "tim_phi"),
    "s": ("det_s", "tim_s"),
}

median_relerr_values = []
for metric_name, (det_col, tim_col) in comparison_pairs.items():
    metadata_key = f"fit_compare_median_relerr_{metric_name}"
    if run_detached_fit and run_timtrack_fit and det_col in working_df.columns and tim_col in working_df.columns:
        det_arr = pd.to_numeric(working_df[det_col], errors="coerce").to_numpy(dtype=float)
        tim_arr = pd.to_numeric(working_df[tim_col], errors="coerce").to_numpy(dtype=float)
        denom = det_arr + tim_arr
        relerr = np.divide(
            2.0 * np.abs(det_arr - tim_arr),
            denom,
            out=np.full(det_arr.shape, np.nan, dtype=float),
            where=denom != 0,
        )
        relerr = relerr[np.isfinite(relerr)]
        if relerr.size > 0:
            med_val = float(np.median(relerr))
            global_variables[metadata_key] = med_val
            median_relerr_values.append(med_val)
        else:
            global_variables[metadata_key] = np.nan
    else:
        global_variables[metadata_key] = np.nan

global_variables["fit_compare_mean_error"] = (
    float(np.mean(median_relerr_values)) if median_relerr_values else np.nan
)

# Simulated-data truth comparison in slowness basis.
# Classic relative error: (s_cal - s_c) / s_c, with s_c = 1/c.
# Median excludes zero slowness entries (placeholders/non-fits).
def _store_slowness_relerr_to_sc(metadata_key: str, slowness_col: str) -> None:
    if not is_simulated_file or slowness_col not in working_df.columns:
        global_variables[metadata_key] = np.nan
        return

    slowness_arr = pd.to_numeric(working_df[slowness_col], errors="coerce").to_numpy(dtype=float)
    valid_slowness = slowness_arr[np.isfinite(slowness_arr) & (slowness_arr != 0.0)]
    if valid_slowness.size == 0:
        global_variables[metadata_key] = np.nan
        return

    relerr_arr = (valid_slowness - sc) / sc
    relerr_arr = relerr_arr[np.isfinite(relerr_arr)]
    global_variables[metadata_key] = float(np.median(relerr_arr)) if relerr_arr.size > 0 else np.nan

if run_detached_fit:
    _store_slowness_relerr_to_sc("fit_compare_median_relerr_detached_s_to_1_over_c", "det_s")
else:
    global_variables["fit_compare_median_relerr_detached_s_to_1_over_c"] = np.nan

if run_timtrack_fit:
    _store_slowness_relerr_to_sc("fit_compare_median_relerr_timtrack_s_to_1_over_c", "tim_s")
else:
    global_variables["fit_compare_median_relerr_timtrack_s_to_1_over_c"] = np.nan

tt_columns_desired = ['datetime', 'raw_tt', 'clean_tt', 'cal_tt', 'list_tt', 'tracking_tt', 'definitive_tt', 'fit_tt']
tt_columns_present = [col for col in tt_columns_desired if col in working_df.columns]
param_hash_cols = ["param_hash"] if "param_hash" in working_df.columns else []

columns_to_keep = (
    tt_columns_present
    + param_hash_cols
    + [
        # New definitions
        'x', 'x_err', 'y', 'y_err', 'theta', 'theta_err', 'phi', 'phi_err', 's', 's_err',

        # Charge

        # # Chisqs
        # 'chi_timtrack', 'chi_alternative',

        # Strip-level time and charge info (ordered by plane and strip)
        *[f'Q_P{p}s{s}' for p in range(1, 5) for s in range(1, 5)],
        
        # Strip-level time and charge info with crosstalk
        # *[f'Q_P{p}s{s}_with_crstlk' for p in range(1, 5) for s in range(1, 5)]
    ]
)

working_df = working_df[columns_to_keep]

# Path to save the cleaned dataframe
# Create output directory if it does not exist.
os.makedirs(f"{output_directory}", exist_ok=True)
OUT_PATH = f"{output_directory}/fitted_{basename_no_ext}.parquet"
KEY = "df"  # HDF5 key name

# --- Example: your cleaned DataFrame is called working_df ---
# (Here, you would have your data cleaning code before saving)
# working_df = ...

if VERBOSE:
    print("Columns in the final dataframe:")
    for col in working_df.columns:
        print(f" - {col}")
    

# def collect_columns(columns: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
#     """Return all column names that match *pattern*."""
#     return [name for name in columns if pattern.match(name)]

# # Pattern for P1_Q_sum_*, P2_Q_sum_*, P3_Q_sum_*, P4_Q_sum_*
# Q_SUM_PATTERN = re.compile(r'^P[1-4]_Q_sum_.*$')

# # If Q*_F_* and Q*_B_* are zero for all cases, remove the row
# Q_cols = collect_columns(working_df.columns, Q_SUM_PATTERN)
# working_df = working_df[(working_df[Q_cols] != 0).any(axis=1)]

print(f"Original number of events in the dataframe: {original_number_of_events}")
fit_tt_total = len(working_df)
if create_debug_plots and "fit_tt" in working_df.columns:
    debug_fig_idx = plot_debug_histograms(
        working_df,
        ["fit_tt"],
        {"fit_tt": [10]},
        title=f"Task 4 pre-filter: fit_tt >= 10 [NON-TUNABLE] (station {station})",
        out_dir=debug_plot_directory,
        fig_idx=debug_fig_idx,
    )
fit_tt_mask = working_df["fit_tt"].notna() & (working_df["fit_tt"] >= 10)
working_df = working_df.loc[fit_tt_mask].copy()
record_filter_metric(
    "fit_tt_lt_10_rows_removed_pct",
    fit_tt_total - int(fit_tt_mask.sum()),
    fit_tt_total if fit_tt_total else 0,
)
list_tt_int = pd.to_numeric(working_df["list_tt"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
fit_tt_int = pd.to_numeric(working_df["fit_tt"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)

fit_vals, fit_counts = np.unique(fit_tt_int, return_counts=True)
for tt_value, count in zip(fit_vals, fit_counts):
    tt_label = normalize_tt_label(tt_value)
    global_variables[f"fit_tt_{tt_label}_count"] = int(count)

combo_pairs = np.column_stack((list_tt_int, fit_tt_int))
combo_vals, combo_counts = np.unique(combo_pairs, axis=0, return_counts=True)
for (list_tt_value, fit_tt_value), count in zip(combo_vals, combo_counts):
    combo_label = normalize_tt_label(f"{int(list_tt_value)}_{int(fit_tt_value)}")
    global_variables[f"list_to_fit_tt_{combo_label}_count"] = int(count)

# Final number of events
final_number_of_events = len(working_df)
print(f"Final number of events in the dataframe: {final_number_of_events}")

print(
    f"Writing fit parquet: rows={len(working_df)} cols={len(working_df.columns)} -> {OUT_PATH}"
)
if VERBOSE:
    print("Columns before saving list->fit parquet:")
    for col in working_df.columns:
        print(f" - {col}")

# Data purity
data_purity = final_number_of_events / original_number_of_events * 100

# End of the execution time
_prof["s_save_finish_s"] = round(time.perf_counter() - _t_sec, 2)
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

# This line is a placeholder

# -------------------------------------------------------------------------------
# Filter metadata (ancillary) ----------------------------------------------------
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

# -------------------------------------------------------------------------------
# Specific metadata ------------------------------------------------------------
# -------------------------------------------------------------------------------

global_variables.update(build_events_per_second_metadata(working_df))
ensure_global_count_keys(("list_tt", "fit_tt", "list_to_fit_tt"))
add_normalized_count_metadata(
    global_variables,
    global_variables.get("events_per_second_total_seconds", 0),
)
set_global_rate_from_tt_rates(
    global_variables,
    preferred_prefixes=("fit_tt", "list_tt"),
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

# Keep denominator available for both trigger_type and specific metadata outputs.
global_variables["count_rate_denominator_seconds"] = rate_histogram_variables.get(
    "count_rate_denominator_seconds",
    0,
)

prune_redundant_count_metadata(global_variables, log_fn=print)
trigger_type_prefixes = ("list_tt", "fit_tt", "list_to_fit_tt")
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
        (
            is_specific_metadata_excluded_column(column_name)
            and column_name != "count_rate_denominator_seconds"
        )
        or is_trigger_type_metadata_column(column_name, trigger_type_prefixes)
        or column_name.startswith("definitive_tt_")
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
    if os.path.exists(file_path):
        safe_move(file_path, completed_file_path)
        now = time.time()
        os.utime(completed_file_path, (now, now))
        print("************************************************************")
        print(f"File moved from\n{file_path}\nto:\n{completed_file_path}")
        print("************************************************************")
    else:
        print(f"Input file already absent (maybe previously processed): {file_path}")
        print("Skipping move; fitted output stays in OUTPUT_FILES.")

if status_execution_date is not None:
    update_status_progress(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        completion_fraction=1.0,
        param_hash=str(global_variables.get("param_hash", "")),
    )

# %%
