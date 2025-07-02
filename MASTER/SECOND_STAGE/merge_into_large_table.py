from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

"""
Created on Wed Dec 18 2024

@author: csoneira@ucm.es
"""

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

# Standard Library
import os
import sys
from glob import glob
from datetime import datetime

# Third-party Libraries
import pandas as pd

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

# Define station number

# Check if the script has an argument
if len(sys.argv) < 2:
    print("Error: No station provided.")
    print("Usage: python3 script.py <station>")
    sys.exit(1)

# Get the station argument
station = sys.argv[1]
print(f"Station: {station}")

# -----------------------------------------------------------------------------

base_folder = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")

# Define the directories you are looking into
directories = {
    "event_data": os.path.join(base_folder, "FIRST_STAGE/EVENT_DATA"),
    "lab_logs": os.path.join(base_folder, "FIRST_STAGE/LAB_LOGS"),
    "reanalysis": os.path.join(base_folder, "FIRST_STAGE/COPERNICUS"),
}

# Define the output directory
output_directory = os.path.join(base_folder, "SECOND_STAGE")

# Define folder paths as a list of absolute paths
folder_paths = [
    directories["event_data"],
    directories["lab_logs"],
    directories["reanalysis"],
]

# Define the output file path
output_file = os.path.join(output_directory, "total_data_table.csv")

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Collect all matching CSV files from the specified folders
file_paths = []
for folder_path in folder_paths:
    file_paths.extend(glob(os.path.join(folder_path, "big*.csv")))

# Check if any files were found
if not file_paths:
    raise FileNotFoundError("No CSV files found in the specified directories.")

print("Bringin' the data together: ", file_paths)

# Load all CSV files into dataframes and store them in a list
dataframes = [pd.read_csv(file_path, parse_dates=['Time']) for file_path in file_paths]

for df in dataframes:
    df['Time'] = df['Time'].dt.floor('1min')  # Round to minute precision

for df in dataframes:
    df.sort_values('Time', inplace=True)

for df in dataframes:
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

print(f"Loaded {len(dataframes)} CSV files.")

merged_df = dataframes[0]
for df in dataframes[1:]:
    merged_df = pd.merge(merged_df, df, on='Time', how='outer')

# Sort by Time after merging
merged_df.sort_values('Time', inplace=True)

print("Merged DataFrame Info:")
print(merged_df.info())

print("First few rows of merged data:")
print(merged_df.head())

print("Last few rows of merged data:")
print(merged_df.tail())

# Save the merged dataframe
# if os.path.exists(output_file):
#     # If the output file already exists, load it and append new rows
#     existing_df = pd.read_csv(output_file, parse_dates=['Time'])
#     combined_df = pd.concat([existing_df, merged_df]).drop_duplicates(subset=['Time'], keep='last')
# else:
    # combined_df = merged_df

if os.path.exists(output_file):
    # If the output file already exists, load it and append new rows
    print("Removing existing file: ", output_file)
    os.remove(output_file)
combined_df = merged_df

# Replace all 0s with NaNs
print("Replacing 0s with NaNs...")
combined_df.replace(0, pd.NA, inplace=True)

print("Saving the data...")
# Save the final dataframe to a CSV file
combined_df.to_csv(output_file, index=False)

print(f"Data has been merged and saved to {output_file}")

print('------------------------------------------------------')
print(f"merge_into_large_table.py completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('------------------------------------------------------')