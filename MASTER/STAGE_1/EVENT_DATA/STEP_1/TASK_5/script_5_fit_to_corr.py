#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from MASTER.common.file_selection import select_latest_candidate
from MASTER.common.plot_utils import pdf_save_rasterized_page
from MASTER.common.status_csv import initialize_status_row, update_status_progress
from MASTER.common.reprocessing_utils import get_reprocessing_value
from MASTER.common.simulated_data_utils import resolve_simulated_z_positions

task_number = 5

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
    "DATAFLOW_v3/MASTER/CONFIG_FILES/config_parameters_task_5.csv",
)
fallback_parameter_config_file_path = os.path.join(
    user_home,
    "DATAFLOW_v3/MASTER/CONFIG_FILES/config_parameters.csv",
)
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
debug_mode = bool(config.get("debug_mode", False))


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
        print(f"Warning: No parameters file found for task 5")
    return config_obj


try:
    config = _apply_parameter_overrides(config, station)
except NameError:
    pass
home_path = config["home_path"]
REFERENCE_TABLES_DIR = Path(home_path) / "DATAFLOW_v3" / "MASTER" / "CONFIG_FILES" / "METADATA_REPRISE" / "REFERENCE_TABLES"



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


# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------


run_jupyter_notebook = bool(config.get("run_jupyter_notebook", False))
if run_jupyter_notebook:
    station = str(config.get("jupyter_station_default_task_5", "2"))
else:
    # Check if the script has an argument
    if len(sys.argv) < 2:
        print("Error: No station provided.")
        print("Usage: python3 script.py <station>")
        sys.exit(1)

    # Get the station argument
    station = sys.argv[1]

if station not in ["0", "1", "2", "3", "4"]:
    print("Error: Invalid station. Please provide a valid station (0, 1, 2, 3 or 4).")
    sys.exit(1)
# print(f"Station: {station}")

set_station(station)
config = _apply_parameter_overrides(config, station)

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


if len(sys.argv) == 3:
    user_file_path = sys.argv[2]
    user_file_selection = True
    print("User provided file path:", user_file_path)
else:
    user_file_selection = False


create_debug_plots = bool(config.get("create_debug_plots", False))

print("Creating the necessary directories...")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

# Define base working directory
home_directory = os.path.expanduser(f"~")
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
config_file_directory = os.path.expanduser(f"~/DATAFLOW_v3/MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY/STATION_{station}")
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
save_plots = config["save_plots"]

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
save_plots = config["save_plots"]

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
    default_z_positions = config.get("default_z_positions", [0, 150, 300, 450])
    if isinstance(default_z_positions, (list, tuple)) and len(default_z_positions) >= 4:
        z_1, z_2, z_3, z_4 = default_z_positions[:4]
    else:
        z_1, z_2, z_3, z_4 = 0, 150, 300, 450



unprocessed_files = os.listdir(base_directories["unprocessed_directory"])
processing_files = os.listdir(base_directories["processing_directory"])
completed_files = os.listdir(base_directories["completed_directory"])
last_file_test = bool(config.get("last_file_test", False))

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
                    sys.exit("No files to process in COMPLETED after normalization.")
            else:
                sys.exit("No files to process in UNPROCESSED, PROCESSING and decided to not reanalyze COMPLETED.")

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
                sys.exit("No files to process in UNPROCESSED, PROCESSING and decided to not reanalyze COMPLETED.")

        else:
            sys.exit("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")

# This is for all cases
file_path = processing_file_path

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

# if limit:
#     print(f'Taking the first {limit_number} rows.')


# Read the data file into a DataFrame
KEY = "df"

# Load dataframe
working_df = pd.read_parquet(file_path, engine="pyarrow")
working_df = working_df.rename(columns=lambda col: col.replace("_diff_", "_dif_"))
print(f"Listed dataframe reloaded from: {file_path}")
# Ensure param_hash is persisted for downstream tasks.
if "param_hash" not in working_df.columns:
    working_df["param_hash"] = str(simulated_param_hash) if simulated_param_hash else ""
elif simulated_param_hash:
    param_series = working_df["param_hash"]
    missing_hash = param_series.isna()
    try:
        missing_hash |= param_series.astype(str).str.strip().eq("")
    except Exception:
        pass
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
global_variables = {
    'analysis_mode': 0,
}

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
    global_variables[f"fit_tt_{tt_value}_count"] = int(count)

corr_tt_counts_initial = working_df[CORR_TT_COLUMN].value_counts()
for tt_value, count in corr_tt_counts_initial.items():
    global_variables[f"{CORR_TT_COLUMN}_{tt_value}_count"] = int(count)



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
    )



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




fast_mode = config["fast_mode"]
debug_mode = config["debug_mode"]
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
save_plots = config["save_plots"]
show_plots = config["show_plots"]
create_pdf = config["create_pdf"]

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


residual_plots_fast = config["residual_plots_fast"]

timtrack_iteration_fast = config["timtrack_iteration_fast"]
timtrack_iteration_debug = config["timtrack_iteration_debug"]

time_calibration_fast = config["time_calibration_fast"]
time_calibration_debug = config["time_calibration_debug"]

charge_front_back_fast = config["charge_front_back_fast"]
charge_front_back_debug = config["charge_front_back_debug"]

create_plots = config["create_plots"]


complete_reanalysis = config["complete_reanalysis"]

limit_fast = config["limit_fast"]
limit_debug = config["limit_debug"]

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

# Alternative fitter filter
det_pos_filter = config["det_pos_filter"]
det_theta_left_filter = config["det_theta_left_filter"]
det_theta_right_filter = config["det_theta_right_filter"]
det_phi_left_filter = config["det_phi_left_filter"]
det_phi_right_filter = config["det_phi_right_filter"]
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
    def _emit_param_debug(param_label, columns, thresholds):
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

fig_idx = 1
plot_list = []

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
y_widths = [np.array([wide_strip, wide_strip, wide_strip, narrow_strip]), 
            np.array([narrow_strip, wide_strip, wide_strip, wide_strip])]

def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

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


self_trigger = bool(config.get("self_trigger", False))








fast_mode = config["fast_mode"]
debug_mode = config["debug_mode"]
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
save_plots = config["save_plots"]
show_plots = config["show_plots"]
create_pdf = config["create_pdf"]

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


residual_plots_fast = config["residual_plots_fast"]

timtrack_iteration_fast = config["timtrack_iteration_fast"]
timtrack_iteration_debug = config["timtrack_iteration_debug"]

time_calibration_fast = config["time_calibration_fast"]
time_calibration_debug = config["time_calibration_debug"]

charge_front_back_fast = config["charge_front_back_fast"]
charge_front_back_debug = config["charge_front_back_debug"]

create_plots = config["create_plots"]



limit_fast = config["limit_fast"]
limit_debug = config["limit_debug"]

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

# Alternative fitter filter
det_pos_filter = config["det_pos_filter"]
det_theta_left_filter = config["det_theta_left_filter"]
det_theta_right_filter = config["det_theta_right_filter"]
det_phi_left_filter = config["det_phi_left_filter"]
det_phi_right_filter = config["det_phi_right_filter"]
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

fig_idx = 1
plot_list = []

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
y_widths = [np.array([wide_strip, wide_strip, wide_strip, narrow_strip]), 
            np.array([narrow_strip, wide_strip, wide_strip, wide_strip])]

def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

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



fast_mode = config["fast_mode"]
debug_mode = config["debug_mode"]
last_file_test = config["last_file_test"]

# Accessing all the variables from the configuration
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
save_plots = config["save_plots"]
show_plots = config["show_plots"]
create_pdf = config["create_pdf"]

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


residual_plots_fast = config["residual_plots_fast"]

timtrack_iteration_fast = config["timtrack_iteration_fast"]
timtrack_iteration_debug = config["timtrack_iteration_debug"]

time_calibration_fast = config["time_calibration_fast"]
time_calibration_debug = config["time_calibration_debug"]

charge_front_back_fast = config["charge_front_back_fast"]
charge_front_back_debug = config["charge_front_back_debug"]

create_plots = config["create_plots"]



limit_fast = config["limit_fast"]
limit_debug = config["limit_debug"]

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

# Alternative fitter filter
det_pos_filter = config["det_pos_filter"]
det_theta_left_filter = config["det_theta_left_filter"]
det_theta_right_filter = config["det_theta_right_filter"]
det_phi_left_filter = config["det_phi_left_filter"]
det_phi_right_filter = config["det_phi_right_filter"]
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














config_files_directory = config["config_files_directory"]

angular_corr_directory = config_files_directory + "/ANGULAR_CORRECTION"


# -----------------------------------------------------------------------------
# Some variables that define the analysis, define a dictionary with the variables:
# 'purity_of_data', etc.
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
y_widths = [np.array([wide_strip, wide_strip, wide_strip, narrow_strip]), 
            np.array([narrow_strip, wide_strip, wide_strip, wide_strip])]

def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

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


theta_boundaries_raw = config.get("theta_boundaries", [])
region_layout_raw = config.get("region_layout", [])

theta_boundaries = _coerce_numeric_sequence(theta_boundaries_raw, float)
theta_values = []
for b in theta_boundaries:
    if isinstance(b, (int, float)) and np.isfinite(b):
        b_float = float(b)
        if 0 <= b_float <= 90 and b_float not in theta_values:
            theta_values.append(b_float)
theta_boundaries = theta_values

region_layout = _coerce_numeric_sequence(region_layout_raw, int)
region_layout = [max(1, int(abs(n))) for n in region_layout if isinstance(n, (int, float))]

expected_regions = len(theta_boundaries) + 1
if not region_layout:
    region_layout = [1] * expected_regions
elif len(region_layout) < expected_regions:
    region_layout = region_layout + [region_layout[-1]] * (expected_regions - len(region_layout))
elif len(region_layout) > expected_regions:
    region_layout = region_layout[:expected_regions]

if not theta_boundaries:
    theta_boundaries = []


print(f"Theta boundaries (degrees): {theta_boundaries}")


correct_angle = bool(config.get("correct_angle", False))
global_variables['correct_angle'] = correct_angle

df = working_df.copy()
main_df = working_df.copy()
main_df['Theta_fit'] = main_df['theta']
main_df['Phi_fit'] = main_df['phi']



def plot_polar_region_grid_flexible(ax, theta_boundaries, region_layout, theta_right_limit=np.pi / 2.5):

    # Only use boundaries below or equal to theta_right_limit
    max_deg = np.degrees(theta_right_filter)
    valid_boundaries = [b for b in theta_boundaries if b <= max_deg]
    all_bounds = [0] + valid_boundaries + [max_deg]
    radii = [np.radians(b) for b in all_bounds]

    # Draw concentric circles (excluding outermost edge)
    for r in radii[1:-1]:
        ax.plot(np.linspace(0, 2 * np.pi, 1000), [r] * 1000, color='white', linestyle='--', linewidth=3)

    # Draw radial lines within each ring
    for i, (r0, r1, n_phi) in enumerate(zip(radii[:-1], radii[1:], region_layout[:len(radii)-1])):
        if n_phi <= 1:
            continue
        delta_phi = 2 * np.pi / n_phi
        for j in range(n_phi):
            phi = j * delta_phi
            ax.plot([phi, phi], [r0, r1], color='white', linestyle='--', linewidth=3)



def classify_region_flexible(row, theta_boundaries, region_layout):
    theta = row['theta'] * 180 / np.pi
    phi = (row['phi'] * 180 / np.pi + row.get('phi_north', 0)) % 360
    phi = ((phi + 180) % 360) - 180  # map to [-180, 180)

    # Build region bins: [0, t1), [t1, t2), ..., [tn, 90]
    all_bounds = [0] + theta_boundaries + [90]
    for i, (tmin, tmax) in enumerate(zip(all_bounds[:-1], all_bounds[1:])):
        if tmin <= theta < tmax or (i == len(region_layout) - 1 and theta == 90):
            n_phi = region_layout[i]
            if n_phi == 1:
                return f'R{i}.0'
            else:
                bin_width = 360 / n_phi
                idx = int((phi + 180) // bin_width) % n_phi
                return f'R{i}.{idx}'
        
    return 'None'


# Input parameters
theta_right_limit = np.pi / 2.5

# Compute angular boundaries
max_deg = np.degrees(theta_right_limit)
valid_boundaries = [b for b in theta_boundaries if b <= max_deg]
all_bounds_deg = [0] + valid_boundaries + [max_deg]
radii = np.radians(all_bounds_deg)


print(f"Plots are: {create_plots}")



if create_plots:
    
    print("----------------------- Drawing angular regions ----------------------")
    
    # Initialize plot
    fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(8, 8))
    ax.set_facecolor(plt.cm.viridis(0.0))
    ax.set_title("Region Labels for Specified Angular Segmentation", color='white')
    ax.set_theta_zero_location('N')

    # Draw concentric θ boundaries (including outermost)
    for r in radii[1:]:
        ax.plot(np.linspace(0, 2 * np.pi, 1000), [r] * 1000,
                color='white', linestyle='--', linewidth=3)

    # Draw radial (φ) separators for each region layout
    for i, (r0, r1, n_phi) in enumerate(zip(radii[:-1], radii[1:], region_layout[:len(radii) - 1])):
        if n_phi > 1:
            delta_phi = 2 * np.pi / n_phi
            for j in range(n_phi):
                phi = j * delta_phi
                ax.plot([phi, phi], [r0, r1], color='white', linestyle='--', linewidth=1.5)

    # Annotate region labels
    for i, (r0, r1, n_phi) in enumerate(zip(radii[:-1], radii[1:], region_layout[:len(radii) - 1])):
        r_label = (r0 + r1) / 2
        if n_phi == 1:
            ax.text(0, r_label, f'R{i}.0', ha='center', va='center',
                    color='white', fontsize=10, weight='bold')
        else:
            dphi = 2 * np.pi / n_phi
            for j in range(n_phi):
                phi_label = (j + 0.5) * dphi
                ax.text(phi_label, r_label, f'R{i}.{j}', ha='center', va='center',
                        rotation=0, rotation_mode='anchor',
                        color='white', fontsize=10, weight='bold')

    # Add radius labels slightly *outside* the outermost circle for clarity
    for r_deg in all_bounds_deg[1:]:
        r_rad = np.radians(r_deg)
        ax.text(np.pi + 0.09, r_rad - 0.05, f'{int(round(r_deg))}°', ha='center', va='bottom',
                color='white', fontsize=10, alpha=0.9)

    ax.grid(color='white', linestyle=':', linewidth=0.5, alpha=0.1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_yticklabels([])

    # Final layout
    title = "Region Labels for Specified Angular Segmentation"
    ax.set_ylim(0, theta_right_limit)
    plt.suptitle(title, fontsize=16, color='white')
    plt.tight_layout()
    if save_plots:
        final_filename = f'{fig_idx}_{title.replace(" ", "_")}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()




#%%



if create_plots:
    
    print("-------------------------- Angular plots -----------------------------")
        
    df_filtered = df.copy()
    
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
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()
    
    

if create_plots or create_essential_plots:
    
    theta_left_filter = 0
    theta_right_filter = np.pi / 2.5
        
    phi_left_filter = -np.pi
    phi_right_filter = np.pi
        
    df_filtered = df.copy()
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
            
        plot_polar_region_grid_flexible(ax, theta_boundaries, region_layout)
            
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
        plt.savefig(save_fig_path, format='png')
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

    print(main_df.columns.to_list())

    #%%

    df_input = main_df
    df_pred = sample_true_angles_nearest(
                df_fit=df_input,
                matrices=matrices,
                n_bins=n_bins,
                rng=np.random.default_rng(),
                show_progress=True )

    df = df_pred.copy()
    
    df['theta'] = df['Theta_pred']
    df['phi'] = df['Phi_pred']
    
    # Plotting corrected vs measured angles
    if create_plots:
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
                plt.savefig(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()

else:
    print("Angle correction is disabled.")
    df['Theta_pred'] = main_df['Theta_fit']
    df['Phi_pred'] = main_df['Phi_fit']
    
    df['theta'] = df['Theta_pred']
    df['phi'] = df['Phi_pred']




if create_plots or create_essential_plots:
    
    theta_left_filter = 0
    theta_right_filter = np.pi / 2.5
        
    phi_left_filter = -np.pi
    phi_right_filter = np.pi
        
    df_filtered = df.copy()
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
            
        plot_polar_region_grid_flexible(ax, theta_boundaries, region_layout)
            
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
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()






df['region'] = df.apply(lambda row: classify_region_flexible(row, theta_boundaries, region_layout), axis=1)
print(df['region'].value_counts())








working_df = df.copy()






# -----------------------------------------------------------------------------
# Create and save the PDF -----------------------------------------------------
# -----------------------------------------------------------------------------

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
OUT_PATH = f"{output_directory}/corrected_{basename_no_ext}.parquet"
KEY = "df"  # HDF5 key name

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# --- Example: your cleaned DataFrame is called working_df ---
# (Here, you would have your data cleaning code before saving)
# working_df = ...




# Print all column names in the dataframe
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
working_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')




# Print all column names in the dataframe
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
if create_debug_plots and CORR_TT_COLUMN in working_df.columns:
    debug_fig_idx = plot_debug_histograms(
        working_df,
        [CORR_TT_COLUMN],
        {CORR_TT_COLUMN: [10]},
        title=f"Task 5 pre-filter: {CORR_TT_COLUMN} >= 10 [NON-TUNABLE] (station {station})",
        out_dir=debug_plot_directory,
        fig_idx=debug_fig_idx,
    )
corr_tt_total = len(working_df)
corr_tt_mask = working_df[CORR_TT_COLUMN].notna() & (working_df[CORR_TT_COLUMN] >= 10)
working_df = working_df.loc[corr_tt_mask].copy()
record_filter_metric(
    "corr_tt_lt_10_rows_removed_pct",
    corr_tt_total - int(corr_tt_mask.sum()),
    corr_tt_total if corr_tt_total else 0,
)

working_df.loc[:, "task5_to_corr_tt"] = working_df[CORR_TT_COLUMN].astype(str)
# Backward-compatible alias used by downstream metadata consumers.
working_df.loc[:, "fit_to_corr_tt"] = working_df["task5_to_corr_tt"]

corr_tt_counts = working_df[CORR_TT_COLUMN].value_counts()
for tt_value, count in corr_tt_counts.items():
    global_variables[f"corr_tt_{tt_value}_count"] = int(count)

fit_to_corr_counts = working_df["task5_to_corr_tt"].value_counts()
for combo_value, count in fit_to_corr_counts.items():
    global_variables[f"task5_to_corr_tt_{combo_value}_count"] = int(count)
    global_variables[f"fit_to_corr_tt_{combo_value}_count"] = int(count)

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
# working_df.to_csv(OUT_PATH.replace('.h5', '.csv'), index=False)
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
