#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

from __future__ import annotations

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------

# CONFIG 

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

from MASTER.common.execution_logger import set_station, start_timer
from MASTER.common.status_csv import append_status_row, mark_status_complete

start_timer(__file__)
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/config.yaml")

print(f"Using config file: {config_file_path}")

with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)

home_path = config["home_path"]
print(f"Home path: {home_path}")



outlier_limits = config["outlier_limits"]
print(outlier_limits)

create_new_csv = config["create_new_csv"]
print(f"Create new CSV: {create_new_csv}")


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
set_station(station)

# -----------------------------------------------------------------------------

log_base_directory = os.path.expanduser(f"{home_path}/DATAFLOW_v3/STATIONS/MINGO0{station}/STAGE_1/LAB_LOGS/")

status_csv_path = os.path.join(log_base_directory, "log_aggregate_and_join_status.csv")
status_timestamp = append_status_row(status_csv_path)

# Define directory paths relative to base_directory
base_directories = {
    "clean_logs_directory": os.path.join(log_base_directory, "CLEAN_LOGS"),
    
    "unprocessed_logs_directory": os.path.join(log_base_directory, "LOG_UNPROCESSED_DIRECTORY"),
    # "processing_logs_directory": os.path.join(log_base_directory, "LOG_PROCESSING_DIRECTORY"),
    # "completed_logs_directory": os.path.join(log_base_directory, "LOG_COMPLETED_DIRECTORY"),
    "accumulated_directory": os.path.join(log_base_directory, "LOG_ACC_DIRECTORY")
}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)


clean_logs_directory = base_directories["clean_logs_directory"]

unprocessed_logs_directory = base_directories["unprocessed_logs_directory"]
# processing_logs_directory = base_directories["processing_logs_directory"]
# completed_logs_directory = base_directories["completed_logs_directory"]

accumulated_directory = base_directories["accumulated_directory"]

final_output_path = os.path.join(log_base_directory, "big_log_lab_data.csv")


# Ensure directories exist
os.makedirs(clean_logs_directory, exist_ok=True)

os.makedirs(unprocessed_logs_directory, exist_ok=True)
# os.makedirs(processing_logs_directory, exist_ok=True)
# os.makedirs(completed_logs_directory, exist_ok=True)

os.makedirs(accumulated_directory, exist_ok=True)

clean_files = set(os.listdir(clean_logs_directory))

unprocessed_files = set(os.listdir(unprocessed_logs_directory))
# processing_files = set(os.listdir(processing_logs_directory))
# completed_files = set(os.listdir(completed_logs_directory))

# Files to move: in RAW but not in UNPROCESSED, PROCESSING, or COMPLETED
# files_to_move = clean_files - (unprocessed_files | processing_files | completed_files)
# files_to_move = clean_files - unprocessed_files
files_to_move = clean_files

# Copy files to UNPROCESSED

print('Files to move:', len(files_to_move))

for file_name in files_to_move:
    src_path = os.path.join(clean_logs_directory, file_name)
    dest_path = os.path.join(unprocessed_logs_directory, file_name)
    try:
        shutil.move(src_path, dest_path)
        #print(f"Move {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to copy {file_name}: {e}")


print('--------------------------- python script starts ---------------------------')

# Function to process each file type
def process_files(file_type_prefix, expected_columns, output_filename):

    all_files = [os.path.join(unprocessed_logs_directory, f) for f in os.listdir(unprocessed_logs_directory) if f.startswith(file_type_prefix)]
    dataframes = []

    for file in all_files:
        try:
            # Attempt to load file
            df = pd.read_csv(file, sep=r'\s+', header=None, on_bad_lines='skip')
            
            # Check column count
            if len(df.columns) > len(expected_columns):
                #print(f"Trimming extra columns in {file}")
                df = df.iloc[:, :len(expected_columns)]  # Truncate to expected column count
            elif len(df.columns) < len(expected_columns):
                #print(f"Padding missing columns in {file}")
                for _ in range(len(expected_columns) - len(df.columns)):
                    df[len(df.columns)] = None  # Add missing columns as NaN

            # Assign column names
            df.columns = expected_columns
            
            # Drop unused columns
            df = df.loc[:, ~df.columns.str.contains("Unused")]
            
            # Format datetime column if applicable
            if 'Date' in expected_columns and 'Hour' in expected_columns:
                df['Time'] = pd.to_datetime(df['Date'] + 'T' + df['Hour'], errors='coerce')
                df.drop(columns=['Hour', 'Date'], inplace=True)
                df = df.dropna(subset=['Time'])

            # Collect dataframe
            dataframes.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Aggregate and save to CSV
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        output_path = os.path.join(accumulated_directory, output_filename)
        combined_df.to_csv(output_path, index=False)
        print(f"Aggregated CSV saved: {output_path}")


print('Processing files...')

# First part --------------------------------------------------------------------------------------
# Process hv0 files
process_files('hv0_', ["Date", "Hour", "Unused1", "Unused2", "Unused3", "Unused4", "Unused5", "Unused6",
                       "CurrentNeg", "CurrentPos", "HVneg", "HVpos", "Unused7", "Unused8", "Unused9",
                       "Unused10", "Unused11", "Unused12", "Unused13", "Unused14", "Unused15"],
              "hv_aggregated.csv")

# Process rates files
process_files('rates_', ["Date", "Hour", "Asserted", "Edge", "Accepted", "Multiplexer1", "M2", "M3", "M4", "CM1", "CM2", "CM3", "CM4"],
              "rates_aggregated.csv")

# Process sensors_bus0 files
process_files('sensors_bus0_', ["Date", "Hour", "Unused1", "Unused2", "Unused3", "Unused4", "Temperature_ext", "RH_ext", "Pressure_ext"],
              "sensors_ext_aggregated.csv")

# Process sensors_bus1 files
process_files('sensors_bus1_', ["Date", "Hour", "Unused1", "Unused2", "Unused3", "Unused4", "Temperature_int", "RH_int", "Pressure_int"],
              "sensors_int_aggregated.csv")

# Process odroid files
process_files('Odroid_', ["Date", "Hour", "DiskFill1", "DiskFill2", "DiskFillX"],
              "odroid_aggregated.csv")

# Process flow files
process_files('Flow', ["Date", "Hour", "FlowRate1", "FlowRate2", "FlowRate3", "FlowRate4"],
              "flow_aggregated.csv")

print('All files processed...')

# Second part -------------------------------------------------------------------------------------
file_mappings = {
    "hv": os.path.join(accumulated_directory, "hv_aggregated.csv"),
    "rates": os.path.join(accumulated_directory, "rates_aggregated.csv"),
    "sensors_ext": os.path.join(accumulated_directory, "sensors_ext_aggregated.csv"),
    "sensors_int": os.path.join(accumulated_directory, "sensors_int_aggregated.csv"),
    "odroid": os.path.join(accumulated_directory, "odroid_aggregated.csv"),
    "flow": os.path.join(accumulated_directory, "flow_aggregated.csv"),
}


def process_csv(file_path):
    """Load CSV, calculate per-minute averages, and reindex by minute."""
    # Load CSV
    df = pd.read_csv(file_path)
    
    # df["Time"] = pd.to_datetime(df["Date"] + " " + df["Hour"])
    # df.drop(columns=["Date", "Hour"], inplace=True)
    
    # Ensure the Time column is datetime
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    
    # Drop rows with invalid Time values
    df = df.dropna(subset=['Time'])

    # Set Time as the index
    df.set_index('Time', inplace=True)

    # Ensure numeric columns only
    numeric_columns = df.select_dtypes(include=['number']).columns
    df = df[numeric_columns]
    
    # Sort by datetime index
    df = df.sort_index()
    
    # Resample to 1-minute intervals and calculate the mean
    df_resampled = df.resample('1min').mean()

    return df_resampled


def merge_dataframes(file_mappings, start_time=None):
    """Process and merge all dataframes from a given start time."""
    dataframes = []

    for name, path in file_mappings.items():
        df_resampled = process_csv(path)

        # Filter data based on start_time
        if start_time:
            df_resampled = df_resampled[df_resampled.index > start_time]

        # Rename columns to include source name
        df_resampled.columns = [f"{name}_{col}" for col in df_resampled.columns]
        
        # Filter outliers before merging
        for column, (lower, upper) in outlier_limits.items():
            if column in df_resampled.columns:
                df_resampled[column] = df_resampled[column].where((df_resampled[column] >= lower) & (df_resampled[column] <= upper), np.nan)
        
        dataframes.append(df_resampled)

    # Merge all dataframes on the Time index
    merged_df = pd.concat(dataframes, axis=1)
    
    return merged_df

# -----------------------------------------------------------------------------
# Remove this if we actually want to update the data, because this line always creates a new one
if os.path.exists(final_output_path):
    os.remove(final_output_path)
    print(f"Removed existing {final_output_path}")

updated_df = merge_dataframes(file_mappings)
# -----------------------------------------------------------------------------

# Resample to ensure no gaps and fill missing timestamps
updated_df = updated_df.resample('1min').mean() # Will achieve the same result: skipna=True is default setting for mean(), but just to make it vey clear...

# print('Interpolating missing points...')
# updated_df = updated_df.interpolate(method='linear', axis=0, limit_direction='both')

print('Saving the updated CSV...')
updated_df.reset_index(inplace=True)
updated_df.to_csv(final_output_path, index=False, float_format="%.5g")

print(f"Updated merged data saved to {final_output_path}")

mark_status_complete(status_csv_path, status_timestamp)
