#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 2024

@author: csoneira@ucm.es
"""

import os
import pandas as pd
from glob import glob
import sys
from datetime import datetime

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
    "reanalysis": os.path.join(base_folder, "FIRST_STAGE/REANALYSIS"),
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

# Load all CSV files into dataframes and store them in a list
dataframes = [pd.read_csv(file_path, parse_dates=['Time']) for file_path in file_paths]

print(f"Loaded {len(dataframes)} CSV files.")

# Merge the dataframes on the 'Time' column, allowing for missing data
merged_df = dataframes[0]
for df in dataframes[1:]:
    merged_df = pd.merge(merged_df, df, on='Time', how='outer')

# Save the merged dataframe
if os.path.exists(output_file):
    # If the output file already exists, load it and append new rows
    existing_df = pd.read_csv(output_file, parse_dates=['Time'])
    combined_df = pd.concat([existing_df, merged_df]).drop_duplicates(subset=['Time'], keep='first')
else:
    combined_df = merged_df

# Save the final dataframe to a CSV file
combined_df.to_csv(output_file, index=False)

print(f"Data has been merged and saved to {output_file}")

print('------------------------------------------------------')
print(f"merge_into_large_table.py completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('------------------------------------------------------')