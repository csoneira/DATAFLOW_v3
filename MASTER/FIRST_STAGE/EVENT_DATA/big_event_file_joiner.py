from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

# Standard Library
import builtins
import os
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
import csv

# Third-party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import poisson
from tqdm import tqdm




import yaml

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

from MASTER.common.status_csv import append_status_row, mark_status_complete

user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/config.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]



load_big_event_file = config["load_big_event_file"]



# Load the config once at the top of your script
with open(f"{home_path}/DATAFLOW_v3/MASTER/config.yaml") as f:
    config = yaml.safe_load(f)

SIG_DIGITS = config["significant_digits"]


suffixes = config["suffixes"]

print(suffixes)


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------- Running big_event_file_joiner.py -------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

# Check if the script has an argument
if len(sys.argv) < 2:
    print("Error: No station provided.")
    print("Usage: python3 script.py <station>")
    sys.exit(1)

# Get the station argument
station = sys.argv[1]
print(f"Station: {station}")

PIPELINE_CSV_HEADERS = [
    'basename',
    'start_date',
    'hld_remote_add_date',
    'hld_local_add_date',
    'dat_add_date',
    'list_ev_name',
    'list_ev_add_date',
    'acc_name',
    'acc_add_date',
    'merge_add_date',
]

station_dir = Path.home() / 'DATAFLOW_v3' / 'STATIONS' / f'MINGO0{station}'
pipeline_csv_path = station_dir / f'database_status_{station}.csv'
pipeline_csv_path.parent.mkdir(parents=True, exist_ok=True)
if not pipeline_csv_path.exists():
    with pipeline_csv_path.open('w', newline='') as handle:
        csv.writer(handle).writerow(PIPELINE_CSV_HEADERS)


def ensure_start_value(row):
    base_name = row.get('basename', '')
    digits = base_name[-11:]
    if len(digits) == 11 and digits.isdigit() and not row.get('start_date'):
        yy = int(digits[:2])
        doy = int(digits[2:5])
        hh = int(digits[5:7])
        mm = int(digits[7:9])
        ss = int(digits[9:11])
        year = 2000 + yy
        try:
            dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
            row['start_date'] = dt.strftime('%Y-%m-%d_%H.%M.%S')
        except ValueError:
            pass


def load_pipeline_rows():
    rows = []
    with pipeline_csv_path.open('r', newline='') as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    return rows


def store_pipeline_rows(rows):
    with pipeline_csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=PIPELINE_CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def update_merge_timestamp(acc_filename: str, timestamp: str) -> None:
    rows = load_pipeline_rows()
    updated = False
    for row in rows:
        if row.get('acc_name') == acc_filename:
            ensure_start_value(row)
            row['merge_add_date'] = timestamp
            updated = True
    if updated:
        store_pipeline_rows(rows)


merge_excluded_acc = {row.get('acc_name') for row in load_pipeline_rows() if row.get('acc_name') and row.get('merge_add_date')}

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

fig_idx = 0
plot_list = []

station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
working_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/FIRST_STAGE/EVENT_DATA")
acc_working_directory = os.path.join(working_directory, "LIST_TO_ACC")

status_csv_path = os.path.join(working_directory, "big_event_file_joiner_status.csv")
status_timestamp = append_status_row(status_csv_path)

# Define subdirectories relative to the working directory
base_directories = {
    "list_events_directory": os.path.join(working_directory, "LIST_EVENTS_DIRECTORY"),
    
    "base_plots_directory": os.path.join(acc_working_directory, "PLOTS"),
    
    "pdf_directory": os.path.join(acc_working_directory, "PLOTS/PDF_DIRECTORY"),
    "base_figure_directory": os.path.join(acc_working_directory, "PLOTS/FIGURE_DIRECTORY"),
    "figure_directory": os.path.join(acc_working_directory, f"PLOTS/FIGURE_DIRECTORY/FIGURES_EXEC_ON_{date_execution}"),
    
    "unprocessed_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_UNPROCESSED"),
    "processing_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_PROCESSING"),
    "error_directory": os.path.join(acc_working_directory, "ACC_FILES/ERROR_DIRECTORY"),
    "completed_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_COMPLETED"),
    
    "acc_events_directory": os.path.join(working_directory, "ACC_EVENTS_DIRECTORY"),
    "acc_rejected_directory": os.path.join(working_directory, "ACC_REJECTED"),
}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

# Path to big_event_data.csv
big_event_file = os.path.join(working_directory, "big_event_data.csv")

# Erase all files in the figure_directory
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory)

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))


# --------------------------------------------------------------------------------------------
# Move small or too big files in the destination folder to a directory of rejected -----------
# --------------------------------------------------------------------------------------------

list_events_directory = base_directories["list_events_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
processing_directory = base_directories["processing_directory"]
error_directory = base_directories["error_directory"]
completed_directory = base_directories["completed_directory"]

list_event_files = set(os.listdir(list_events_directory))
unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))

# Files to copy: in LIST but not in UNPROCESSED, PROCESSING, or COMPLETED
files_to_copy = list_event_files - unprocessed_files - processing_files - completed_files

# Copy files to UNPROCESSED
for file_name in files_to_copy:
    src_path = os.path.join(list_events_directory, file_name)
    dest_path = os.path.join(unprocessed_directory, file_name)
    try:
        # Copy instead of move
        shutil.copy(src_path, dest_path)
        print(f"Copied {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to copy {file_name}: {e}")


def round_to_significant_digits(x):
    if isinstance(x, float):
        return float(f"{x:.{SIG_DIGITS}g}")
    return x

def combine_duplicates(group):
    # print("-----------------------------------------------------------------")
    
    # Drop fully NaN rows (except 'Time')
    group = group.dropna(subset=[col for col in group.columns if col not in ["Time", "execution_date"]], how='all')

    # If only one row remains, return it
    if len(group) == 1:
        return group.iloc[[0]]

    # Sort by execution_date (latest first)
    group = group.sort_values("execution_date", ascending=False)

    # Identify indices that are consecutive (i.e., abs difference of 1)
    indices = group.index.to_numpy()
    consecutive_indices = {i for i in indices for j in indices if abs(i - j) == 1}

    # Keep only the two most recent rows among the consecutive ones
    consecutive_group = group.loc[group.index.intersection(consecutive_indices)].head(2)

    if len(consecutive_group) == 2:
        print(f"Aggregating two consecutive rows: {consecutive_group.index.tolist()}")
        
        time_value = consecutive_group.iloc[0]["Time"]
        
        # Fill NaNs in each row using values from the other row (fixed)
        consecutive_group.iloc[0] = consecutive_group.iloc[0].fillna(consecutive_group.iloc[1])
        consecutive_group.iloc[1] = consecutive_group.iloc[1].fillna(consecutive_group.iloc[0])

        # Extract weights (events column)
        weights = consecutive_group["events"].fillna(0)  # Ensure no NaNs in weights

        # Columns to average (weighted)
        # suffixes = [
        #     # Summary metrics and quality flags
        #     'CRT_avg', 'sigmoid_width', 'background_slope',
        #     'one_side_events', 'purity_of_data_percentage',
        #     'unc_y', 'unc_tsum', 'unc_tdif',

        #     # Reconstruction outputs
        #     'x', 'y', 'theta', 'phi', 's', 'th_chi',
        #     'x_std', 'y_std', 'theta_std', 'phi_std', 's_std', 'th_chi_std',

        #     # Streamer fractions
        #     'streamer_percent_1', 'streamer_percent_2', 'streamer_percent_3', 'streamer_percent_4',

        #     # Config
        #     'over_P1', 'P1-P2', 'P2-P3', 'P3-P4', 'phi_north',
        # ]

        avg_columns = [col for col in group.columns if any(s in col for s in suffixes)]
    
        # All other columns (except "Time") are summed
        sum_columns = [col for col in group.columns if col not in avg_columns and col not in ["Time"]]

        # Aggregation logic: Weighted mean for avg_columns, sum for sum_columns
        aggregated_result = {}

        for col in group.columns:
            if col in ["execution_date"]:
                continue
            if col in ["Time"]:
                aggregated_result[col] = time_value
                continue  # Skip these columns
            elif col in avg_columns:
                # Compute weighted mean
                aggregated_result[col] = (consecutive_group[col] * weights).sum() / weights.sum()
            elif col == "events":
                # Sum events
                aggregated_result[col] = consecutive_group[col].sum()
            else:
                # Sum other columns
                aggregated_result[col] = consecutive_group[col].sum(min_count=1)

        return pd.DataFrame([pd.Series(aggregated_result)])

    # Otherwise, return the most recent row
    return group.iloc[[0]]


processed_files_path = Path(working_directory) / "big_event_data_processed_files.txt"
processed_files: set[str] = set()
metadata_needs_write = False

if processed_files_path.exists():
    processed_files = {
        line.strip()
        for line in processed_files_path.read_text().splitlines()
        if line.strip()
    }
else:
    metadata_needs_write = True

big_event_df = pd.DataFrame()
loaded_existing_file = False

if load_big_event_file and os.path.exists(big_event_file):
    big_event_df = pd.read_csv(big_event_file, sep=',', parse_dates=['Time'])
    loaded_existing_file = True
    print(f"Loaded existing big_event_data.csv with {len(big_event_df)} rows.")
elif load_big_event_file:
    print("Existing big_event_data.csv not found. Creating a new one from available ACC files.")
else:
    print("Configuration requests fresh big_event_data.csv creation.")

acc_directory = base_directories["acc_events_directory"]
csv_files = sorted([f for f in os.listdir(acc_directory) if f.endswith('.csv')])

if metadata_needs_write and load_big_event_file and loaded_existing_file:
    big_event_mtime = os.path.getmtime(big_event_file)
    for csv_file in csv_files:
        csv_path = os.path.join(acc_directory, csv_file)
        if os.path.getmtime(csv_path) <= big_event_mtime:
            processed_files.add(csv_file)
    if processed_files:
        print("Reconstructed processed ACC file list from timestamps (metadata was missing).")
    metadata_needs_write = True

files_to_process = [f for f in csv_files if f not in processed_files and f not in merge_excluded_acc]
processed_any_new_file = False

if files_to_process:
    iterator = tqdm(files_to_process, total=len(files_to_process), desc="Joining CSVs")
else:
    iterator = []

for csv_file in iterator:
    csv_path = os.path.join(acc_directory, csv_file)
    new_data = pd.read_csv(csv_path, sep=',', parse_dates=['Time'])
    new_data = new_data.replace(0, np.nan).copy()
    new_data['Time'] = new_data['Time'].dt.floor('1min')
    new_data = new_data.copy()
    new_data["execution_date"] = os.path.getmtime(csv_path)
    big_event_df = pd.concat([big_event_df, new_data], ignore_index=True)
    processed_files.add(csv_file)
    processed_any_new_file = True
    metadata_needs_write = True
    update_merge_timestamp(csv_file, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if not files_to_process:
    if loaded_existing_file and csv_files:
        print("No new ACC CSV files found to append.")
    elif not csv_files:
        print("No ACC CSV files found in ACC_EVENTS_DIRECTORY. Nothing to process.")

needs_aggregation = processed_any_new_file or not loaded_existing_file

if big_event_df.empty:
    print("No data available to aggregate or save.")
elif needs_aggregation:
    if "Time" not in big_event_df.columns:
        print("Column 'Time' missing; cannot aggregate big_event_df.")
    else:
        print("Grouping the CSVs by 'Time' and combining duplicates...")
        print("Columns in big_event_df:", big_event_df.columns.to_list())
        tqdm.pandas()
        big_event_df = big_event_df.groupby('Time', as_index=False).progress_apply(combine_duplicates).reset_index(drop=True)

        if not isinstance(big_event_df, pd.DataFrame):
            print("Warning: big_event_df is not a DataFrame. Converting it...")
            big_event_df = big_event_df.to_frame()

        big_event_df = big_event_df.sort_values(by="Time")
        print("Replacing 0s with NaNs...")
        big_event_df = big_event_df.replace(0, np.nan)

        big_event_df.to_csv(big_event_file, sep=',', index=False)
        print(big_event_df.columns.to_list())
        print(f"Saved big_event_data.csv with {len(big_event_df)} rows.")
else:
    print("No new ACC files to append. Existing big_event_data.csv left unchanged.")

if metadata_needs_write:
    processed_files_path.parent.mkdir(parents=True, exist_ok=True)
    processed_files_path.write_text("\n".join(sorted(processed_files)))
    print(f"Recorded {len(processed_files)} processed ACC files.")

mark_status_complete(status_csv_path, status_timestamp)
