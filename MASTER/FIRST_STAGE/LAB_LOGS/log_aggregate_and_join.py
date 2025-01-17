#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import sys
import shutil

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

# -----------------------------------------------------------------------------

log_base_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/FIRST_STAGE/LAB_LOGS/")


# Define directory paths relative to base_directory
base_directories = {
    
    "raw_logs_directory": os.path.join(log_base_directory, "LOG_PROCESSED_DIRECTORY"),
    "completed_logs_directory": os.path.join(log_base_directory, "LOG_COMPLETED_DIRECTORY"),
    "accumulated_directory": os.path.join(log_base_directory, "LOG_ACC_DIRECTORY")
}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

raw_logs_directory = base_directories["raw_logs_directory"]
completed_logs_directory = base_directories["completed_logs_directory"]
accumulated_directory = base_directories["accumulated_directory"]

final_output_path = os.path.join(log_base_directory, "big_log_lab_data.csv")



# Ensure directories exist
os.makedirs(raw_logs_directory, exist_ok=True)
os.makedirs(completed_logs_directory, exist_ok=True)
os.makedirs(accumulated_directory, exist_ok=True)

raw_files = set(os.listdir(raw_logs_directory))
completed_files = set(os.listdir(completed_logs_directory))

# Files to move: in RAW but not in COMPLETED
files_to_move = raw_files - completed_files

# Copy files to UNPROCESSED
for file_name in files_to_move:
    src_path = os.path.join(raw_logs_directory, file_name)
    dest_path = os.path.join(completed_logs_directory, file_name)
    try:
        shutil.move(src_path, dest_path)
        print(f"Move {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to copy {file_name}: {e}")



# def usage():
#     """Display the usage message and exit."""
#     print("""
#     This script processes and merges log files from a local directory, aggregates data, 
#     and stores the merged output as a CSV file. 

#     Description:
#       - Processes files with predefined prefixes and expected column counts.
#       - Aggregates data into resampled CSV files.
#       - Filters outliers and interpolates missing data.
#       - Produces a final merged log file.

#     Output:
#       - Processed files are saved in ~/DATAFLOW_v3/LAB_LOGS/aggregated_csvs.
#       - Merged output is saved in ~/DATAFLOW_v3/LAB_LOGS/big_log_lab_data.csv.

#     No arguments are required to run this script.

#     Example:
#       python3 <script_name>.py
#     """)
#     sys.exit(1)


# # Check for arguments
# if len(sys.argv) > 1:
#     print("Error: This script does not accept arguments.")
#     usage()


print('--------------------------- python script starts ---------------------------')

# Function to process each file type
def process_files(file_type_prefix, expected_columns, output_filename):

    all_files = [os.path.join(completed_logs_directory, f) for f in os.listdir(completed_logs_directory) if f.startswith(file_type_prefix)]
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
            if 'Date' in expected_columns and 'Time' in expected_columns:
                df['Time'] = pd.to_datetime(df['Date'] + 'T' + df['Time'], errors='coerce')
                df.drop(columns=['Date'], inplace=True)
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
process_files('hv0_', ["Date", "Time", "Unused1", "Unused2", "Unused3", "Unused4", "Unused5", "Unused6",
                       "CurrentNeg", "CurrentPos", "HVneg", "HVpos", "Unused7", "Unused8", "Unused9",
                       "Unused10", "Unused11", "Unused12", "Unused13", "Unused14", "Unused15"],
              "hv_aggregated.csv")

# Process rates files
process_files('rates_', ["Time", "Asserted", "Edge", "Accepted", "Multiplexer1", "M2", "M3", "M4", "CM1", "CM2", "CM3", "CM4"],
              "rates_aggregated.csv")

# Process sensors_bus0 files
process_files('sensors_bus0_', ["Date", "Time", "Unused1", "Unused2", "Unused3", "Unused4", "Temperature_ext", "RH_ext", "Pressure_ext"],
              "sensors_ext_aggregated.csv")

# Process sensors_bus1 files
process_files('sensors_bus1_', ["Date", "Time", "Unused1", "Unused2", "Unused3", "Unused4", "Temperature_int", "RH_int", "Pressure_int"],
              "sensors_int_aggregated.csv")

# Process odroid files
process_files('Odroid_', ["Date", "Time", "DiskFill1", "DiskFill2", "DiskFillX"],
              "odroid_aggregated.csv")

# Process flow files
process_files('Flow0_', ["Date", "Time", "FlowRate1", "FlowRate2", "Pressure1", "Pressure2"],
              "flow_aggregated.csv")


print('All files processed...')

# Second part -------------------------------------------------------------------------------------
file_mappings = {
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

    # Ensure the Time column is datetime
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

    # Drop rows with invalid Time values
    df = df.dropna(subset=['Time'])

    # Set Time as the index
    df.set_index('Time', inplace=True)

    # Ensure numeric columns only
    numeric_columns = df.select_dtypes(include=['number']).columns
    df = df[numeric_columns]

    # Resample to 1-minute intervals and calculate the mean
    df_resampled = df.resample('min').mean()

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

        dataframes.append(df_resampled)

    # Merge all dataframes on the Time index
    merged_df = pd.concat(dataframes, axis=1)

    return merged_df

# Check if merged csv exists
if os.path.exists(final_output_path):
    print(f"Existing {final_output_path} found. Checking for new data...")

    # Load existing merged data
    existing_df = pd.read_csv(final_output_path, parse_dates=['Time'])
    existing_df.set_index('Time', inplace=True)

    # Determine the last timestamp in the existing data
    last_timestamp = existing_df.index.max()
    print(f"Last timestamp in existing data: {last_timestamp}")
    
    start_timestamp = last_timestamp - pd.Timedelta(hours=3)
    
    print(f'Starting from {start_timestamp}, 3h before.')
    
    # Merge only new data
    new_data = merge_dataframes(file_mappings, start_time=start_timestamp)

    # Append new data to the existing data
    updated_df = pd.concat([existing_df, new_data]).drop_duplicates().sort_index()
else:
    print(f"No existing {final_output_path} found. Processing all data...")

    # Merge all data from the beginning
    updated_df = merge_dataframes(file_mappings)

# Resample to ensure no gaps and fill missing timestamps
updated_df = updated_df.resample('1min').mean()

# Define the limits for outliers as a dictionary
outlier_limits = {
    "rates_Edge": (10, 25),
    "rates_Accepted": (10, 20),
    "rates_Multiplexer1": (0, 200),
    "rates_M2": (0, 200),
    "rates_M3": (0, 200),
    "rates_M4": (0, 200),
    "rates_CM1": (1, 20),
    "rates_CM2": (5, 20),
    "rates_CM3": (5, 20),
    "rates_CM4": (5, 20),
    "sensors_ext_Temperature_ext": (0, 50),
    "sensors_ext_RH_ext": (0, 100),
    "sensors_ext_Pressure_ext": (900, 1100),
    "sensors_int_Temperature_int": (0, 70),
    "sensors_int_RH_int": (0, 100),
    "sensors_int_Pressure_int": (900, 1100),
    "odroid_DiskFill1": (0, 100),
    "odroid_DiskFill2": (0, 100),
    "odroid_DiskFillX": (0, 100000),
    "flow_FlowRate1": (0, 1500),
    "flow_FlowRate2": (0, 1500),
    "flow_Pressure1": (0, 1500),
    "flow_Pressure2": (0, 1500),
}

# Filter the data, replacing outliers with NaN
for column, (lower, upper) in outlier_limits.items():
    if column in updated_df.columns:
        updated_df[column] = updated_df[column].where((updated_df[column] >= lower) & (updated_df[column] <= upper), np.nan)

print('Interpolating missing points...')
updated_df = updated_df.interpolate(method='linear', axis=0, limit_direction='both')

print('Saving the updated CSV...')
updated_df.reset_index(inplace=True)
updated_df.to_csv(final_output_path, index=False, float_format="%.5g")

print(f"Updated merged data saved to {final_output_path}")