#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from MASTER.common.file_selection import select_latest_candidate
from MASTER.common.plot_utils import pdf_save_rasterized_page
from MASTER.common.status_csv import initialize_status_row, update_status_progress
from MASTER.common.reprocessing_utils import get_reprocessing_value
from MASTER.common.simulated_data_utils import resolve_simulated_z_positions

task_number = 3

# I want to chrono the execution time of the script
start_execution_time_counting = datetime.now()

VERBOSE = bool(os.environ.get("DATAFLOW_VERBOSE"))
_PRINT_ALWAYS_KEYWORDS = (
    "error",
    "warning",
    "failed",
    "exception",
    "traceback",
    "usage",
)
_print = builtins.print


def _debug_logging_enabled() -> bool:
    return bool(globals().get("debug_mode", False)) or VERBOSE


def _is_important_message(message: str) -> bool:
    lowered = message.lower()
    if any(keyword in lowered for keyword in _PRINT_ALWAYS_KEYWORDS):
        return True
    return "total execution time" in lowered or "data purity" in lowered


def print(*args, **kwargs):
    force = kwargs.pop("force", False)
    if force or _debug_logging_enabled():
        _print(*args, **kwargs)
        return
    message = " ".join(str(arg) for arg in args)
    if _is_important_message(message):
        _print(*args, **kwargs)

# Warning Filters
warnings.filterwarnings("ignore", message=".*Data has no positive values, and therefore cannot be log-scaled.*")

start_timer(__file__)
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
parameter_config_file_path = os.path.join(
    user_home,
    "DATAFLOW_v3/MASTER/CONFIG_FILES/config_parameters_task_3.csv",
)
fallback_parameter_config_file_path = os.path.join(
    user_home,
    "DATAFLOW_v3/MASTER/CONFIG_FILES/config_parameters.csv",
)
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
debug_mode = bool(config.get("debug_mode", False))
home_path = config["home_path"]


def _apply_parameter_overrides(config_obj, station_id):
    task_path = Path(parameter_config_file_path)
    if task_path.exists():
        config_obj = update_config_with_parameters(config_obj, task_path, station_id)
        print(f"Warning: Loaded task parameters from {task_path}")
        return config_obj
    fallback_path = Path(fallback_parameter_config_file_path)
    if fallback_path.exists():
        print(f"Warning: Task parameters file not found; falling back to {fallback_path}")
        config_obj = update_config_with_parameters(config_obj, fallback_path, station_id)
    else:
        print(f"Warning: No parameters file found for task 3")
    return config_obj

run_jupyter_notebook = bool(config.get("run_jupyter_notebook", False))
if run_jupyter_notebook:
    station = str(config.get("jupyter_station_default_task_3", "2"))
else:
    if len(sys.argv) < 2:
        print("Error: No station provided.")
        print("Usage: python3 script.py <station>")
        sys.exit(1)
    station = sys.argv[1]

if station not in ["0", "1", "2", "3", "4"]:
    print("Error: Invalid station. Please provide a valid station (0, 1, 2, 3 or 4).")
    sys.exit(1)

set_station(station)
config = _apply_parameter_overrides(config, station)
home_path = config["home_path"]
REFERENCE_TABLES_DIR = Path(home_path) / "DATAFLOW_v3" / "MASTER" / "CONFIG_FILES" / "METADATA_REPRISE" / "REFERENCE_TABLES"


# Round execution time to seconds and format it in YYYY-MM-DD_HH.MM.SS
execution_time = str(start_execution_time_counting).split('.')[0]  # Remove microseconds
print("Execution time is:", execution_time)

ITINERARY_FILE_PATH = Path(
    f"{home_path}/DATAFLOW_v3/MASTER/ANCILLARY/INPUT_FILES/TIME_CALIBRATION_ITINERARIES/itineraries.csv"
)


def load_itineraries_from_file(file_path: Path, required: bool = True) -> list[list[str]]:
    """Return itineraries stored as comma-separated lines in *file_path*."""
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"Cannot find itineraries file: {file_path}")
        return []

    itineraries: list[list[str]] = []
    with file_path.open("r", encoding="utf-8") as itinerary_file:
        print(f"Loading itineraries from {file_path}:")
        for line_number, raw_line in enumerate(itinerary_file, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            segments = [segment.strip() for segment in stripped_line.split(",") if segment.strip()]
            if segments:
                itineraries.append(segments)
                print(segments)

    if not itineraries and required:
        raise ValueError(f"Itineraries file {file_path} is empty.")

    return itineraries


def write_itineraries_to_file(
    file_path: Path,
    itineraries: Iterable[Iterable[str]],
) -> None:
    """Persist unique itineraries to *file_path* as comma-separated lines."""
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



not_use_q_semisum = False

stratos_save = config["stratos_save"]
fast_mode = config["fast_mode"]
debug_mode = config["debug_mode"]
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
save_plots = config["save_plots"]
show_plots = config["show_plots"]
create_pdf = config["create_pdf"]
create_debug_plots = bool(config.get("create_debug_plots", False))
limit = config["limit"]
limit_number = config["limit_number"]

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


residual_plots_fast = config["residual_plots_fast"]

timtrack_iteration_fast = config["timtrack_iteration_fast"]
timtrack_iteration_debug = config["timtrack_iteration_debug"]

time_calibration_fast = config["time_calibration_fast"]
time_calibration_debug = config["time_calibration_debug"]

charge_front_back_fast = config["charge_front_back_fast"]
charge_front_back_debug = config["charge_front_back_debug"]

create_plots = config["create_plots"]


complete_reanalysis = config["complete_reanalysis"]
complete_reanalysis = True

limit = config["limit"]
limit_fast = config["limit_fast"]
limit_debug = config["limit_debug"]

limit_number = config["limit_number"]
limit_number_fast = config["limit_number_fast"]
limit_number_debug = config["limit_number_debug"]

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
T_dif_RPC_left = config["T_dif_RPC_left"]
T_dif_RPC_right = config["T_dif_RPC_right"]
Q_RPC_left = config["Q_RPC_left"]
Q_RPC_right = config["Q_RPC_right"]
Q_dif_RPC_left = config["Q_dif_RPC_left"]
Q_dif_RPC_right = config["Q_dif_RPC_right"]
Y_RPC_left = config["Y_RPC_left"]
Y_RPC_right = config["Y_RPC_right"]

# Alternative fitter filter
det_pos_filter = config.get("det_pos_filter", 800)
det_theta_left_filter = config.get("det_theta_left_filter", 0)
det_theta_right_filter = config.get("det_theta_right_filter", 1.5708)
det_phi_left_filter = config.get("det_phi_left_filter", -3.141592)
det_phi_right_filter = config.get("det_phi_right_filter", 3.141592)
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

create_super_essential_plots = False

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

if fast_mode:
    print('Working in fast mode.')
    residual_plots = residual_plots_fast
    timtrack_iteration = timtrack_iteration_fast
    time_calibration = time_calibration_fast
    charge_front_back = charge_front_back_fast
    create_plots = create_plots_fast
    limit = limit_fast
    limit_number = limit_number_fast

if debug_mode:
    print('Working in debug mode.')
    residual_plots = True
    timtrack_iteration = timtrack_iteration_debug
    time_calibration = time_calibration_debug
    charge_front_back = charge_front_back_debug
    create_plots = create_plots_debug
    limit = limit_debug
    limit_number = limit_number_debug

if debug_mode:
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

if debug_mode:
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

FILTER_METRIC_NAMES: tuple[str, ...] = (
    "filter6_new_zero_rows_pct",
    "q_sum_all_zero_rows_removed_pct",
    "data_purity_percentage",
    "all_components_zero_rows_removed_pct",
    "list_tt_lt_10_rows_removed_pct",
)

filter_metrics: dict[str, float] = {}


def record_filter_metric(name: str, removed: float, total: float) -> None:
    """Record percentage removed for a filter."""
    pct = 0.0 if total == 0 else 100.0 * float(removed) / float(total)
    filter_metrics[name] = round(pct, 4)
    print(f"[filter-metrics] {name}: removed {removed} of {total} ({pct:.2f}%)")

def compute_tt(df: pd.DataFrame, column_name: str, columns_map: dict[int, list[str]] | None = None) -> pd.DataFrame:
    """Compute trigger type based on planes with non-zero charge."""
    def _derive_tt(row: pd.Series) -> str:
        planes_with_charge = []
        for plane in range(1, 5):
            if columns_map:
                charge_columns = [col for col in columns_map.get(plane, []) if col in row.index]
            else:
                charge_columns = [
                    f"Q{plane}_F_1",
                    f"Q{plane}_F_2",
                    f"Q{plane}_F_3",
                    f"Q{plane}_F_4",
                    f"Q{plane}_B_1",
                    f"Q{plane}_B_2",
                    f"Q{plane}_B_3",
                    f"Q{plane}_B_4",
                ]
            if any(row.get(col, 0) != 0 for col in charge_columns):
                planes_with_charge.append(str(plane))
        return "".join(planes_with_charge) if planes_with_charge else "0"

    df[column_name] = df.apply(_derive_tt, axis=1)
    df[column_name] = df[column_name].apply(builtins.int)
    return df

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

if len(sys.argv) == 3:
    user_file_path = sys.argv[2]
    user_file_selection = True
    print("User provided file path:", user_file_path)
else:
    user_file_selection = False


station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
config_file_directory = os.path.expanduser(f"~/DATAFLOW_v3/MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY/STATION_{station}")

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
home_directory = os.path.expanduser(f"~")
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
base_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/STAGE_1/EVENT_DATA")
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
    # If save_plots is False, skip creating the figure_directory
    if directory == base_directories["figure_directory"] and not save_plots:
        continue
    os.makedirs(directory, exist_ok=True)

debug_plot_directory = os.path.join(
    base_directories["base_plots_directory"],
    "DEBUG_PLOTS",
    f"FIGURES_EXEC_ON_{date_execution}",
)
debug_fig_idx = 1
if create_debug_plots:
    os.makedirs(debug_plot_directory, exist_ok=True)

csv_path = os.path.join(metadata_directory, f"task_{task_number}_metadata_execution.csv")
csv_path_specific = os.path.join(metadata_directory, f"task_{task_number}_metadata_specific.csv")
csv_path_filter = os.path.join(metadata_directory, f"task_{task_number}_metadata_filter.csv")
csv_path_status = os.path.join(metadata_directory, f"task_{task_number}_metadata_status.csv")
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









def _normalize_analysis_mode_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "0", "1"}:
        return text
    if text in {"0.0", "1.0"}:
        return text[0]
    if "1" in text:
        return "1"
    if "0" in text:
        return "0"
    return ""


def _sanitize_analysis_mode_rows(rows: List[Dict[str, object]]) -> int:
    fixed = 0
    for row in rows:
        if "analysis_mode" not in row:
            continue
        clean_value = _normalize_analysis_mode_value(row.get("analysis_mode"))
        if row.get("analysis_mode") != clean_value:
            row["analysis_mode"] = clean_value
            fixed += 1
    return fixed


def _repair_metadata_file(metadata_path: Path) -> int:
    original_limit = csv.field_size_limit()
    csv.field_size_limit(sys.maxsize)
    try:
        with metadata_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(existing) for existing in reader]
    finally:
        csv.field_size_limit(original_limit)

    if not fieldnames:
        return 0

    fixed = _sanitize_analysis_mode_rows(rows)
    if fixed:
        with metadata_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return fixed


def save_metadata(
    metadata_path: str,
    row: Dict[str, object],
    preferred_fieldnames: Iterable[str] | None = None,
) -> Path:
    """Append *row* to *metadata_path*, preserving all existing columns."""
    metadata_path = Path(metadata_path)
    rows: List[Dict[str, object]] = []
    fieldnames: List[str] = []

    def _normalise(raw: Dict[str, object]) -> Dict[str, object]:
        return {key: value for key, value in raw.items() if key is not None}

    def _load_existing_rows() -> Tuple[List[str], List[Dict[str, object]]]:
        with metadata_path.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = [_normalise(existing) for existing in reader]
            return existing_fields, existing_rows

    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        try:
            fieldnames, existing_rows = _load_existing_rows()
        except csv.Error as exc:
            if "field larger than field limit" in str(exc).lower():
                fixed = _repair_metadata_file(metadata_path)
                print(
                    f"Detected oversized analysis_mode entries in {metadata_path}; normalized {fixed} row(s)."
                )
                try:
                    fieldnames, existing_rows = _load_existing_rows()
                except csv.Error as err:
                    raise RuntimeError(
                        f"Failed to repair metadata file {metadata_path} after detecting oversized fields."
                    ) from err
            else:
                raise
        rows.extend(existing_rows)

    rows.append(_normalise(dict(row)))

    fixed_during_append = _sanitize_analysis_mode_rows(rows)
    if fixed_during_append:
        print(f"Clamped analysis_mode to 0/1 in {fixed_during_append} metadata row(s).")

    seen = set(fieldnames)
    for item in rows:
        for key in item.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    if preferred_fieldnames:
        preferred = [name for name in preferred_fieldnames if name in seen]
        remainder = [name for name in fieldnames if name not in preferred]
        fieldnames = preferred + remainder
        # Keep purity last for readability in filter metadata CSVs.
        if "data_purity_percentage" in fieldnames:
            fieldnames = [name for name in fieldnames if name != "data_purity_percentage"] + [
                "data_purity_percentage"
            ]

    with metadata_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            formatted = {}
            for key in fieldnames:
                if key in EVENTS_PER_SECOND_COLUMNS:
                    value = item.get(key, 0)
                    if value in ("", None) or (isinstance(value, float) and math.isnan(value)):
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
            writer.writerow(formatted)

    return metadata_path


# -----------------------------------------------------------------------------
# Events per second metadata helpers ------------------------------------------
# -----------------------------------------------------------------------------

EVENTS_PER_SECOND_MAX = 100
EVENTS_PER_SECOND_COLUMNS = [
    *(f"events_per_second_{idx}_count" for idx in range(EVENTS_PER_SECOND_MAX + 1)),
    "events_per_second_total_seconds",
    "events_per_second_global_rate",
]


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
    events_per_second = (
        times.value_counts().reindex(full_range, fill_value=0).sort_index()
    )

    total_seconds = int(events_per_second.size)
    total_events = int(events_per_second.sum())
    metadata["events_per_second_total_seconds"] = total_seconds
    metadata["events_per_second_global_rate"] = (
        round(total_events / total_seconds, 6) if total_seconds > 0 else 0
    )

    hist_counts = events_per_second.value_counts()
    for events_count, seconds_count in hist_counts.items():
        events_count_int = int(events_count)
        if 0 <= events_count_int <= EVENTS_PER_SECOND_MAX:
            metadata[f"events_per_second_{events_count_int}_count"] = int(seconds_count)

    return metadata


def add_normalized_count_metadata(
    metadata: Dict[str, object],
    denominator_seconds: object,
) -> None:
    """
    Add per-second normalized versions of *_count columns to *metadata*.

    - For typical event-count columns: *_count -> *_rate_hz (count / seconds).
    - For events_per_second_{k}_count (seconds with k events): -> *_fraction (seconds / seconds).
    """
    try:
        denom = float(denominator_seconds) if denominator_seconds is not None else 0.0
    except (TypeError, ValueError):
        denom = 0.0

    metadata["count_rate_denominator_seconds"] = int(denom) if denom > 0 else 0

    if denom <= 0:
        print("[count-rates] Denominator seconds is 0; skipping normalized count columns.")
        return

    for key, value in list(metadata.items()):
        if not isinstance(key, str):
            continue

        is_count = key.endswith("_count")
        is_entries = key.endswith(("_entries", "_entries_final", "_entries_initial"))
        if not (is_count or is_entries):
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(num):
            continue

        if is_count:
            if key.startswith("events_per_second_"):
                out_key = key[: -len("_count")] + "_fraction"
            else:
                out_key = key[: -len("_count")] + "_rate_hz"
        else:
            out_key = key + "_rate_hz"

        metadata[out_key] = round(num / denom, 6)


not_use_q_semisum = False

stratos_save = config["stratos_save"]
fast_mode = config["fast_mode"]
debug_mode = config["debug_mode"]
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
save_plots = config["save_plots"]
show_plots = config["show_plots"]
create_pdf = config["create_pdf"]
limit = config["limit"]
limit_number = config["limit_number"]

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


residual_plots_fast = config["residual_plots_fast"]

timtrack_iteration_fast = config["timtrack_iteration_fast"]
timtrack_iteration_debug = config["timtrack_iteration_debug"]

time_calibration_fast = config["time_calibration_fast"]
time_calibration_debug = config["time_calibration_debug"]

charge_front_back_fast = config["charge_front_back_fast"]
charge_front_back_debug = config["charge_front_back_debug"]

create_plots = config["create_plots"]



limit = config["limit"]
limit_fast = config["limit_fast"]
limit_debug = config["limit_debug"]

limit_number = config["limit_number"]
limit_number_fast = config["limit_number_fast"]
limit_number_debug = config["limit_number_debug"]

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
T_dif_RPC_left = config["T_dif_RPC_left"]
T_dif_RPC_right = config["T_dif_RPC_right"]
Q_RPC_left = config["Q_RPC_left"]
Q_RPC_right = config["Q_RPC_right"]
Q_dif_RPC_left = config["Q_dif_RPC_left"]
Q_dif_RPC_right = config["Q_dif_RPC_right"]
Y_RPC_left = config["Y_RPC_left"]
Y_RPC_right = config["Y_RPC_right"]

# Alternative fitter filter
det_pos_filter = config.get("det_pos_filter", 800)
det_theta_left_filter = config.get("det_theta_left_filter", 0)
det_theta_right_filter = config.get("det_theta_right_filter", 1.5708)
det_phi_left_filter = config.get("det_phi_left_filter", -3.141592)
det_phi_right_filter = config.get("det_phi_right_filter", 3.141592)
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

if fast_mode:
    print('Working in fast mode.')
    residual_plots = residual_plots_fast
    timtrack_iteration = timtrack_iteration_fast
    time_calibration = time_calibration_fast
    charge_front_back = charge_front_back_fast
    create_plots = create_plots_fast
    limit = limit_fast
    limit_number = limit_number_fast

if debug_mode:
    print('Working in debug mode.')
    residual_plots = True
    timtrack_iteration = timtrack_iteration_debug
    time_calibration = time_calibration_debug
    charge_front_back = charge_front_back_debug
    create_plots = create_plots_debug
    limit = limit_debug
    limit_number = limit_number_debug

if debug_mode:
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

if debug_mode:
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
fast_mode = config["fast_mode"]
debug_mode = config["debug_mode"]
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
save_plots = config["save_plots"]
show_plots = config["show_plots"]
create_pdf = config["create_pdf"]
limit = config["limit"]
limit_number = config["limit_number"]

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


residual_plots_fast = config["residual_plots_fast"]

timtrack_iteration_fast = config["timtrack_iteration_fast"]
timtrack_iteration_debug = config["timtrack_iteration_debug"]

time_calibration_fast = config["time_calibration_fast"]
time_calibration_debug = config["time_calibration_debug"]

charge_front_back_fast = config["charge_front_back_fast"]
charge_front_back_debug = config["charge_front_back_debug"]

create_plots = config["create_plots"]

limit = config["limit"]
limit_fast = config["limit_fast"]
limit_debug = config["limit_debug"]

limit_number = config["limit_number"]
limit_number_fast = config["limit_number_fast"]
limit_number_debug = config["limit_number_debug"]

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
T_dif_RPC_left = config["T_dif_RPC_left"]
T_dif_RPC_right = config["T_dif_RPC_right"]
Q_RPC_left = config["Q_RPC_left"]
Q_RPC_right = config["Q_RPC_right"]
Q_dif_RPC_left = config["Q_dif_RPC_left"]
Q_dif_RPC_right = config["Q_dif_RPC_right"]
Y_RPC_left = config["Y_RPC_left"]
Y_RPC_right = config["Y_RPC_right"]

# Alternative fitter filter
det_pos_filter = config.get("det_pos_filter", 800)
det_theta_left_filter = config.get("det_theta_left_filter", 0)
det_theta_right_filter = config.get("det_theta_right_filter", 1.5708)
det_phi_left_filter = config.get("det_phi_left_filter", -3.141592)
det_phi_right_filter = config.get("det_phi_right_filter", 3.141592)
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

if fast_mode:
    print('Working in fast mode.')
    residual_plots = residual_plots_fast
    timtrack_iteration = timtrack_iteration_fast
    time_calibration = time_calibration_fast
    charge_front_back = charge_front_back_fast
    create_plots = create_plots_fast
    limit = limit_fast
    limit_number = limit_number_fast

if debug_mode:
    print('Working in debug mode.')
    residual_plots = True
    timtrack_iteration = timtrack_iteration_debug
    time_calibration = time_calibration_debug
    charge_front_back = charge_front_back_debug
    create_plots = create_plots_debug
    limit = limit_debug
    limit_number = limit_number_debug







if debug_mode:
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

if debug_mode:
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
    
    shutil.move(source_path, dest_path)
    now = time.time()
    os.utime(dest_path, (now, now))
    return True

def get_file_path(directory, file_name):
    return os.path.join(directory, file_name)

# Create ALL directories if they don't already exist
# Create ALL directories if they don't already exist
for directory in base_directories.values():
    # If save_plots is False, skip creating the figure_directory
    if directory == base_directories["figure_directory"] and not save_plots:
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
            shutil.move(file_empty, empty_destination_path)
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
        shutil.move(src_path, dest_path)
        now = time.time()
        os.utime(dest_path, (now, now))
        print(f"Move {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to move {file_name}: {e}")


# Erase all files in the figure_directory -------------------------------------------------
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory) if os.path.exists(figure_directory) else []

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))

unprocessed_files = os.listdir(base_directories["unprocessed_directory"])
processing_files = os.listdir(base_directories["processing_directory"])
completed_files = os.listdir(base_directories["completed_directory"])

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
            shutil.move(unprocessed_file_path, processing_file_path)
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
                shutil.move(processing_file_path, error_file_path)
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
                    shutil.move(completed_file_path, processing_file_path)
                    print(f"File moved to PROCESSING: {processing_file_path}")
                else:
                    print("No files to process in COMPLETED after normalization.")
                    sys.exit(0)
            else:
                print("No files to process in UNPROCESSED, PROCESSING and decided to not reanalyze COMPLETED.")
                sys.exit(0)

    else:
        if unprocessed_files:
            print("Shuffling the files in UNPROCESSED...")
            random.shuffle(unprocessed_files)
            for file_name in unprocessed_files:
                unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

                print(f"Moving '{file_name}' to PROCESSING...")
                shutil.move(unprocessed_file_path, processing_file_path)
                print(f"File moved to PROCESSING: {processing_file_path}")
                break

        elif processing_files:
            print("Shuffling the files in PROCESSING...")
            random.shuffle(processing_files)
            for file_name in processing_files:
                # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

                print(f"Processing the last file in PROCESSING: {processing_file_path}")
                error_file_path = os.path.join(base_directories["error_directory"], file_name)
                print(f"File '{processing_file_path}' is already in PROCESSING. Moving it temporarily to ERROR for analysis...")
                shutil.move(processing_file_path, error_file_path)
                processing_file_path = error_file_path
                print(f"File moved to ERROR: {processing_file_path}")
                break

        elif completed_files:
            if complete_reanalysis:
                print("Shuffling the files in COMPLETED...")
                random.shuffle(completed_files)
                for file_name in completed_files:
                    # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
                    completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
                    processing_file_path = os.path.join(base_directories["processing_directory"], file_name)

                    print(f"Moving '{file_name}' to PROCESSING...")
                    shutil.move(completed_file_path, processing_file_path)
                    print(f"File moved to PROCESSING: {processing_file_path}")
                    break
            else:
                print("No files to process in UNPROCESSED, PROCESSING and decided to not reanalyze COMPLETED.")
                sys.exit(0)

        else:
            print("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")
            sys.exit(0)

# This is for all cases
file_path = processing_file_path

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
os.utime(processing_file_path, (now, now))

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
# Ensure param_hash is persisted for downstream tasks.
if "param_hash" not in working_df.columns:
    working_df["param_hash"] = str(simulated_param_hash) if simulated_param_hash else ""
elif simulated_param_hash:
    _ph_series = working_df["param_hash"]
    _ph_missing = _ph_series.isna()
    try:
        _ph_missing |= _ph_series.astype(str).str.strip().eq("")
    except Exception:
        pass
    if _ph_missing.any():
        working_df.loc[_ph_missing, "param_hash"] = str(simulated_param_hash)
print(f"Cleaned dataframe reloaded from: {file_path}")
print("Columns loaded from parquet:")
for col in working_df.columns:
    print(f" - {col}")

if create_debug_plots:
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

working_df = compute_tt(working_df, "cal_tt", cal_tt_columns)
cal_tt_counts_initial = working_df["cal_tt"].value_counts()
for tt_value, count in cal_tt_counts_initial.items():
    global_variables[f"cal_tt_{tt_value}_count"] = int(count)

original_number_of_events = len(working_df)
print(f"Original number of events in the dataframe: {original_number_of_events}")
if status_execution_date is not None:
    update_status_progress(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        completion_fraction=0.5,
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
save_pdf_filename = f"pdf_{save_filename_suffix}.pdf"

if create_plots == False:
    if create_essential_plots == True:
        save_pdf_filename = "essential_" + save_pdf_filename

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

if create_plots:

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
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("----------------- Some more tests (multi-strip data) -----------------")
print("----------------------------------------------------------------------")


if create_plots:

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
                    plt.savefig(save_fig_path, format='png')
                if show_plots:
                    plt.show()
                plt.close()





if create_super_essential_plots:

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
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()




if create_super_essential_plots:

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
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()


if create_plots:

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
            plt.savefig(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()







if create_plots:

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
        plt.savefig(save_fig_path, format='png')
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
        plt.savefig(save_fig_path, format='png')
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
    
    
    if create_plots:
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
                plt.savefig(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()


    

    if create_super_essential_plots:
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
            plt.savefig(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()
    

    loop_adjacent_strip_fit = False

    if loop_adjacent_strip_fit:
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

        # Sigma 1
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


                # Build unique sorted axes for charge_1 and charge_2
                charge1_values = charge_limits_1
                charge2_values = charge_limits_2

                unique_c1 = np.array(sorted(set(charge1_values)))
                unique_c2 = np.array(sorted(set(charge2_values)))

                # Meshgrid of (charge_1, charge_2)
                C1, C2 = np.meshgrid(unique_c1, unique_c2, indexing='xy')

                # Matrices for sigma1 and sigma2
                sigma1_grid = np.full(C1.shape, np.nan, dtype=float)
                sigma2_grid = np.full(C2.shape, np.nan, dtype=float)

                # Precompute index lookup for speed
                idx_c1 = {c: j for j, c in enumerate(unique_c1)}
                idx_c2 = {c: i for i, c in enumerate(unique_c2)}

                # Fill matrices
                for c1, c2, s1, s2 in zip(charge1_values, charge2_values,
                                        sigma1_values, sigma2_values):
                    i = idx_c2[c2]
                    j = idx_c1[c1]
                    sigma1_grid[i, j] = s1
                    sigma2_grid[i, j] = s2

                # Choose which sigma to plot in this figure:
                #   for sigma1 contours:
                Z = sigma1_grid
                #   for sigma2 contours instead, comment line above and use:
                # Z = sigma2_grid

                # Remove rows/cols that are all NaN (optional but avoids warnings)
                if np.all(np.isnan(Z)):
                    ax.set_title(f'Plane {i_plane}, Pattern {pattern}\n(no valid fits)')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Contour plot
                cs = ax.contourf(C1, C2, Z, levels=20)
                # Add a colorbar per row, per column, or one for the whole figure
                # Example: one colorbar per-axis:
                fig.colorbar(cs, ax=ax)

                ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
                ax.set_xlabel('Charge 1')
                ax.set_ylabel('Charge 2')
                ax.grid(True)
                
        fig.suptitle("Fitted Sigmas vs Charge Limit for Different Patterns", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_plots:
            name_of_file = f'tdiff_fitted_sigma_1_vs_charge_limit.png'
            final_filename = f'{fig_idx}_{name_of_file}'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()


        # Sigma 2
        # Plot especifically the sigmas vs charge limit for each plane and pattern, one row per plane, one column per pattern
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


                # Build unique sorted axes for charge_1 and charge_2
                charge1_values = charge_limits_1
                charge2_values = charge_limits_2

                unique_c1 = np.array(sorted(set(charge1_values)))
                unique_c2 = np.array(sorted(set(charge2_values)))

                # Meshgrid of (charge_1, charge_2)
                C1, C2 = np.meshgrid(unique_c1, unique_c2, indexing='xy')

                # Matrices for sigma1 and sigma2
                sigma1_grid = np.full(C1.shape, np.nan, dtype=float)
                sigma2_grid = np.full(C2.shape, np.nan, dtype=float)

                # Precompute index lookup for speed
                idx_c1 = {c: j for j, c in enumerate(unique_c1)}
                idx_c2 = {c: i for i, c in enumerate(unique_c2)}

                # Fill matrices
                for c1, c2, s1, s2 in zip(charge1_values, charge2_values,
                                        sigma1_values, sigma2_values):
                    i = idx_c2[c2]
                    j = idx_c1[c1]
                    sigma1_grid[i, j] = s1
                    sigma2_grid[i, j] = s2

                # Choose which sigma to plot in this figure:
                #   for sigma1 contours:
                # Z = sigma1_grid
                #   for sigma2 contours instead, comment line above and use:
                Z = sigma2_grid

                # Remove rows/cols that are all NaN (optional but avoids warnings)
                if np.all(np.isnan(Z)):
                    ax.set_title(f'Plane {i_plane}, Pattern {pattern}\n(no valid fits)')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Contour plot
                cs = ax.contourf(C1, C2, Z, levels=20)
                # Add a colorbar per row, per column, or one for the whole figure
                # Example: one colorbar per-axis:
                fig.colorbar(cs, ax=ax)

                ax.set_title(f'Plane {i_plane}, Pattern {pattern}')
                ax.set_xlabel('Charge 1')
                ax.set_ylabel('Charge 2')
                ax.grid(True)
                
        fig.suptitle("Fitted Sigmas vs Charge Limit for Different Patterns", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_plots:
            name_of_file = f'tdiff_fitted_sigma_2_vs_charge_limit.png'
            final_filename = f'{fig_idx}_{name_of_file}'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()


print("----------------------------------------------------------------------")
print("----------------------- Y position calculation -----------------------")
print("----------------------------------------------------------------------")

# Y ---------------------------------------------------------------------------
y_widths = [np.array([narrow_strip, narrow_strip, narrow_strip, wide_strip]), 
            np.array([wide_strip, narrow_strip, narrow_strip, narrow_strip])]

def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

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


if create_plots:

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
            plt.savefig(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()

print("Y position calculated.")


print("----------------------------------------------------------------------")
print("------------ Last comprobation to the per-strip variables ------------")
print("----------------------------------------------------------------------")

if create_plots or create_essential_plots:

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

            combinations = [
                (t_sum,  t_diff, f'{t_sum_col} vs {t_dif_col}'),
                (t_sum,  q_sum,  f'{t_sum_col} vs {q_sum_col}'),
                (t_diff, q_sum,  f'{t_dif_col} vs {q_sum_col}'),
                (t_sum,  q_diff, f'{t_sum_col} vs {q_dif_col}'),
                (t_diff, q_diff, f'{t_dif_col} vs {q_dif_col}'),
                (q_sum,  q_diff, f'{q_sum_col} vs {q_dif_col}')
            ]

            for offset, (x, yv, title) in enumerate(combinations):
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
            plt.savefig(save_fig_path, format='png')

        if show_plots: plt.show()
        plt.close()


if self_trigger:
    if create_plots:
        
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

                combinations = [
                    (t_sum,  t_diff, f'{t_sum_col} vs {t_dif_col}'),
                    (t_sum,  q_sum,  f'{t_sum_col} vs {q_sum_col}'),
                    (t_diff, q_sum,  f'{t_dif_col} vs {q_sum_col}'),
                    (t_sum,  q_diff, f'{t_sum_col} vs {q_dif_col}'),
                    (t_diff, q_diff, f'{t_dif_col} vs {q_dif_col}'),
                    (q_sum,  q_diff, f'{q_sum_col} vs {q_dif_col}')
                ]

                for offset, (x, yv, title) in enumerate(combinations):
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
                plt.savefig(save_fig_path, format='png')

            if show_plots: plt.show()
            plt.close()


print("----------------------------------------------------------------------")
print("----------------- Setting the variables of each RPC ------------------")
print("----------------------------------------------------------------------")

# Prepare containers for final results
final_columns = {}

for i_plane in range(1, 5):
    # Column names
    T_sum_cols = [f'T{i_plane}_T_sum_{i+1}' for i in range(4)]
    T_dif_cols = [f'T{i_plane}_T_dif_{i+1}' for i in range(4)]
    Q_sum_cols = [f'Q{i_plane}_Q_sum_{i+1}' for i in range(4)]
    Q_dif_cols = [f'Q{i_plane}_Q_dif_{i+1}' for i in range(4)]

    # Extract data
    T_sums = working_df[T_sum_cols].astype(float).fillna(0).values
    T_difs = working_df[T_dif_cols].astype(float).fillna(0).values
    Q_sums = working_df[Q_sum_cols].astype(float).fillna(0).values
    Q_difs = working_df[Q_dif_cols].astype(float).fillna(0).values

    # Decode binary topology
    active_mask = np.array([
        list(map(int, s)) for s in working_df[f'active_strips_P{i_plane}']
    ])  # shape (N, 4)

    # Compute strip activation count
    n_active = active_mask.sum(axis=1)
    n_active_safe = np.where(n_active == 0, 1, n_active)

    # Apply mask and compute means
    T_sum_masked = T_sums * active_mask
    T_dif_masked = T_difs * active_mask
    Q_dif_masked = Q_difs * active_mask

    T_sum_final = T_sum_masked.sum(axis=1) / n_active_safe
    T_dif_final = T_dif_masked.sum(axis=1) / n_active_safe

    # Enforce zero where no active strips
    T_sum_final[n_active == 0] = 0
    T_dif_final[n_active == 0] = 0

    # Store final values in dictionary
    final_columns[f'P{i_plane}_T_sum_final'] = T_sum_final
    final_columns[f'P{i_plane}_T_dif_final'] = T_dif_final
    final_columns[f'P{i_plane}_Q_sum_final'] = (Q_sums * active_mask).sum(axis=1)
    final_columns[f'P{i_plane}_Q_dif_final'] = Q_dif_masked.sum(axis=1)

# Concatenate all new final columns at once
working_df = pd.concat([working_df, pd.DataFrame(final_columns, index=working_df.index)], axis=1)



if create_plots:
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

        combinations = [
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

        for offset, (x, yv, title) in enumerate(combinations):
            ax = axes[base_idx + offset]
            ax.hexbin(x, yv, gridsize=50, cmap='turbo')
            ax.set_title(title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Hexbin Plots for All Variable Combinations by Plane', fontsize=18)

    if save_plots:
        name_of_file = 'rpc_variables_hexbin_combinations'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close()



if create_plots:
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

        combinations = [
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

        for offset, (x, yv, title) in enumerate(combinations):
            ax = axes[base_idx + offset]
            ax.hexbin(x, yv, gridsize=50, cmap='turbo')
            ax.set_title(title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Hexbin Plots for All Variable Combinations by Plane', fontsize=18)

    if save_plots:
        name_of_file = 'rpc_variables_hexbin_combinations'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close()




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


# Helper to track how many events remain non-zero for key variables around Filter 6
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

if create_debug_plots and filter6_cols:
    t_sum_cols = [col for col in filter6_cols if "T_sum_final" in col]
    if t_sum_cols:
        debug_thresholds = {col: [T_sum_RPC_left, T_sum_RPC_right] for col in t_sum_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            t_sum_cols,
            debug_thresholds,
            title=(
                f"Task 3 pre-filter6: T_sum_RPC_left/right "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

    t_dif_cols = [col for col in filter6_cols if "T_dif_final" in col]
    if t_dif_cols:
        debug_thresholds = {col: [T_dif_RPC_left, T_dif_RPC_right] for col in t_dif_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            t_dif_cols,
            debug_thresholds,
            title=(
                f"Task 3 pre-filter6: T_dif_RPC_left/right "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

    q_sum_cols = [col for col in filter6_cols if "Q_sum_final" in col]
    if q_sum_cols:
        debug_thresholds = {col: [Q_RPC_left, Q_RPC_right] for col in q_sum_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            q_sum_cols,
            debug_thresholds,
            title=(
                f"Task 3 pre-filter6: Q_RPC_left/right "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

    q_dif_cols = [col for col in filter6_cols if "Q_dif_final" in col]
    if q_dif_cols:
        debug_thresholds = {col: [Q_dif_RPC_left, Q_dif_RPC_right] for col in q_dif_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            q_dif_cols,
            debug_thresholds,
            title=(
                f"Task 3 pre-filter6: Q_dif_RPC_left/right "
                f"[tunable] (station {station})"
            ),
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

    y_cols = [col for col in filter6_cols if "Y_final" in col]
    if y_cols:
        debug_thresholds = {col: [Y_RPC_left, Y_RPC_right] for col in y_cols}
        debug_fig_idx = plot_debug_histograms(
            working_df,
            y_cols,
            debug_thresholds,
            title=f"Task 3 pre-filter6: Y_RPC_left/right [tunable] (station {station})",
            out_dir=debug_plot_directory,
            fig_idx=debug_fig_idx,
        )

record_filter6_counts(working_df, "before_filter6")

print("--------------------- Filter 6: calibrated data ----------------------")
for col in working_df.columns:
    if 'T_sum_final' in col:
        working_df[col] = np.where((working_df[col] < T_sum_RPC_left) | (working_df[col] > T_sum_RPC_right), 0, working_df[col])
    if 'T_dif_final' in col:
        working_df[col] = np.where((working_df[col] < T_dif_RPC_left) | (working_df[col] > T_dif_RPC_right), 0, working_df[col])
    if 'Q_sum_final' in col:
        working_df[col] = np.where((working_df[col] < Q_RPC_left) | (working_df[col] > Q_RPC_right), 0, working_df[col])
    if 'Q_dif_final' in col:
        working_df[col] = np.where((working_df[col] < Q_dif_RPC_left) | (working_df[col] > Q_dif_RPC_right), 0, working_df[col])
    if 'Y_' in col:
        working_df[col] = np.where((working_df[col] < Y_RPC_left) | (working_df[col] > Y_RPC_right), 0, working_df[col])

total_events = len(working_df)

for i_plane in range(1, 5):
    y_col      = f'P{i_plane}_Y_final'
    t_sum_col  = f'P{i_plane}_T_sum_final'
    t_dif_col = f'P{i_plane}_T_dif_final'
    q_sum_col  = f'P{i_plane}_Q_sum_final'
    q_dif_col = f'P{i_plane}_Q_dif_final'

    cols = [y_col, t_sum_col, t_dif_col, q_sum_col, q_dif_col]

    # Identify affected rows
    mask = (working_df[cols] == 0).any(axis=1)
    num_affected = mask.sum()

    print(f"Plane {i_plane}: {num_affected} out of {total_events} events affected ({(num_affected / total_events) * 100:.2f}%)")

    # Apply zeroing
    working_df.loc[mask, cols] = 0

if filter6_cols and filter6_before_zero_mask is not None:
    filter6_after_zero_mask = (working_df[filter6_cols] == 0).any(axis=1)
    newly_zeroed = int((filter6_after_zero_mask & ~filter6_before_zero_mask).sum())
    record_filter_metric(
        "filter6_new_zero_rows_pct",
        newly_zeroed,
        len(working_df) if len(working_df) else 0,
    )

record_filter6_counts(working_df, "after_filter6")


# ----------------------------------------------------------------------------------------------------------------
# if stratos_save and station == 2:
if stratos_save:
    print("Saving X and Y for stratos.")
    
    stratos_df = working_df.copy()
    
    # Select columns that start with "Y_" or match "T<number>_T_dif_final"
    filtered_columns = [col for col in stratos_df.columns if col.startswith("Y_") or "_T_dif_final" in col or 'datetime' in col]

    # Create a new DataFrame with the selected columns
    filtered_stratos_df = stratos_df[filtered_columns].copy()

    # Rename "T<number>_T_dif_final" to "X_<number>" and multiply by 200
    filtered_stratos_df.rename(columns=lambda col: f'X_{col.split("_")[0][1:]}' if "_T_dif_final" in col else col, inplace=True)
    filtered_stratos_df.loc[:, filtered_stratos_df.columns.str.startswith("X_")] *= 200

    # Define the save path
    save_stratos = os.path.join(stratos_list_events_directory, f'stratos_data_{save_filename_suffix}.csv')

    # Save DataFrame to CSV (correcting the method name)
    filtered_stratos_df.to_csv(save_stratos, index=False, float_format="%.1f")
# ----------------------------------------------------------------------------------------------------------------


# Same for hexbin
if create_plots or create_essential_plots:

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

        combinations = [
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

        for offset, (x, yv, title) in enumerate(combinations):
            ax = axes[base_idx + offset]
            ax.hexbin(x, yv, gridsize=50, cmap='turbo')
            ax.set_title(title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Hexbin Plots for All Variable Combinations by Plane, filtered', fontsize=18)
    if save_plots:
        name_of_file = 'rpc_variables_hexbin_combinations_filtered'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots: plt.show()
    plt.close()




# # Hexbin + histogram "pairgrid" per plane
# if create_plots or create_essential_plots:

#     for i_plane in range(1, 5):
#         # Column names
#         t_sum_col  = f'P{i_plane}_T_sum_final'
#         t_dif_col = f'P{i_plane}_T_dif_final'
#         q_sum_col  = f'P{i_plane}_Q_sum_final'
#         q_dif_col = f'P{i_plane}_Q_dif_final'
#         y_col      = f'P{i_plane}_Y_final'

#         cols = [t_sum_col, t_dif_col, q_sum_col, q_dif_col, y_col]

#         # Keep only valid rows (non-zero) and enforce q_sum < 150
#         valid_rows = (working_df[cols]
#                       .replace(0, np.nan)
#                       .dropna())
#         cond = valid_rows[q_sum_col] < 150
#         fr = valid_rows.loc[cond, cols]  # filtered rows, ordered as in cols

#         # Prepare data as a dict: {col_name: Series}
#         data = {c: fr[c] for c in cols}
#         n = len(cols)

#         # One figure per plane
#         fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n), squeeze=False)

#         for i in range(n):
#             for j in range(n):
#                 ax = axes[i, j]

#                 if i == j:
#                     # Diagonal: histogram of the variable
#                     ax.hist(data[cols[i]].dropna(), bins='auto')
#                     ax.set_ylabel('Count' if j == 0 else '')
#                 elif i > j:
#                     # Lower triangle: hexbin of (x=cols[j], y=cols[i])
#                     x = data[cols[j]]
#                     yv = data[cols[i]]
#                     hb = ax.hexbin(x, yv, gridsize=50, cmap='turbo')
#                 else:
#                     # Upper triangle: blank
#                     ax.axis('off')

#                 # Axis labels only on outer edges to reduce clutter
#                 if i == n - 1:  # bottom row
#                     ax.set_xlabel(cols[j])
#                 else:
#                     ax.set_xlabel('')
#                 if j == 0:      # left column
#                     ax.set_ylabel(cols[i] if i != j else ax.get_ylabel())
#                 else:
#                     if i != j:  # avoid clearing "Count" set above on diagonal
#                         ax.set_ylabel('')

#         # Put column headers on the (blank) top row to identify columns
#         for j in range(n):
#             axes[0, j].set_title(cols[j], fontsize=12)

#         plt.tight_layout()
#         plt.subplots_adjust(top=0.93)
#         plt.suptitle(f'P{i_plane}: Pairwise Hexbins (lower) + Histograms (diagonal), filtered (Q_sum < 150)', fontsize=16)

#         if save_plots:
#             name_of_file = f'P{i_plane}_pairgrid_hexbin_hist_filtered'
#             final_filename = f'{fig_idx}_{name_of_file}.png'
#             fig_idx += 1
#             save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#             plot_list.append(save_fig_path)
#             plt.savefig(save_fig_path, format='png', dpi=150)

#         if show_plots:
#             plt.show()

#         plt.close(fig)




if create_pdf:
    print(f"Creating PDF with all plots in {save_pdf_path}")
    if len(plot_list) > 0:
        with PdfPages(save_pdf_path) as pdf:
            if plot_list:
                for png in plot_list:
                    if os.path.exists(png) == False:
                        print(f"Error: {png} does not exist.")
                        continue
                    
                    # Open the PNG file directly using PIL to get its dimensions
                    img = Image.open(png)
                    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)  # Set figsize and dpi
                    ax.imshow(img)
                    ax.axis('off')  # Hide the axes
                    pdf_save_rasterized_page(pdf, fig, bbox_inches='tight')  # Save figure tightly fitting the image
                    plt.close(fig)  # Close the figure after adding it to the PDF

        # Remove PNG files after creating the PDF
        for png in plot_list:
            try:
                os.remove(png)
                # print(f"Deleted {png}")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")





# Path to save the cleaned dataframe
# Create output directory if it does not exist /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/DONE/
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
    
    







def _collect_columns(columns: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
    """Return all column names that match *pattern*."""
    return [name for name in columns if pattern.match(name)]

# Pattern for P1_Q_sum_*, P2_Q_sum_*, P3_Q_sum_*, P4_Q_sum_*
Q_SUM_PATTERN = re.compile(r'^P[1-4]_Q_sum_.*$')

# If Q*_F_* and Q*_B_* are zero for all cases, remove the row
Q_cols = _collect_columns(working_df.columns, Q_SUM_PATTERN)
if create_debug_plots and Q_cols:
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
list_tt_total = len(working_df)
if create_debug_plots and "list_tt" in working_df.columns:
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
    working_df["cal_tt"].astype(str) + "_" + working_df["list_tt"].astype(str)
)

list_tt_counts = working_df["list_tt"].value_counts()
for tt_value, count in list_tt_counts.items():
    global_variables[f"list_tt_{tt_value}_count"] = int(count)

cal_to_list_counts = working_df["cal_to_list_tt"].value_counts()
for combo_value, count in cal_to_list_counts.items():
    global_variables[f"cal_to_list_tt_{combo_value}_count"] = int(count)

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
end_time_execution = datetime.now()
execution_time = end_time_execution - start_execution_time_counting
# In minutes
execution_time_minutes = execution_time.total_seconds() / 60

# To save as metadata
filename_base = basename_no_ext
execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
data_purity_percentage = data_purity
total_execution_time_minutes = execution_time_minutes


# -------------------------------------------------------------------------------
# Filter metadata (ancillary) ---------------------------------------------------
# -------------------------------------------------------------------------------
if status_execution_date is not None:
    update_status_progress(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        completion_fraction=0.75,
    )

filter_metrics["data_purity_percentage"] = round(float(data_purity_percentage), 4)
filter_row = {
    "filename_base": filename_base,
    "execution_timestamp": execution_timestamp,
}
for name in FILTER_METRIC_NAMES:
    filter_row[name] = filter_metrics.get(name, "")

metadata_filter_csv_path = save_metadata(
    csv_path_filter,
    filter_row,
    preferred_fieldnames=("filename_base", "execution_timestamp", *FILTER_METRIC_NAMES),
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
        "data_purity_percentage": round(float(data_purity_percentage), 4),
        "total_execution_time_minutes": round(float(total_execution_time_minutes), 4),
    },
)
print(f"Metadata (execution) CSV updated at: {metadata_execution_csv_path}")


# -------------------------------------------------------------------------------
# Specific metadata ------------------------------------------------------------
# -------------------------------------------------------------------------------

global_variables.update(build_events_per_second_metadata(working_df))
add_normalized_count_metadata(
    global_variables,
    global_variables.get("events_per_second_total_seconds", 0),
)

global_variables["filename_base"] = filename_base
global_variables["execution_timestamp"] = execution_timestamp

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
)
print(f"Metadata (specific) CSV updated at: {metadata_specific_csv_path}")


# Save to HDF5 file
working_df.to_parquet(OUT_PATH, engine="pyarrow", compression="zstd", index=False)
print(f"Listed dataframe saved to: {OUT_PATH}")


# Move the original datafile to COMPLETED -------------------------------------
print("Moving file to COMPLETED directory...")

if user_file_selection == False:
    shutil.move(file_path, completed_file_path)
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
    )
