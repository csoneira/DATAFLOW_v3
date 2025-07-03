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
import psutil

# -----------------------------------------------------------------------------

def print_memory_usage(tag=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # in MB
    print(f"[{tag}] Memory usage: {mem:.2f} MB")

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
    "copernicus": os.path.join(base_folder, "FIRST_STAGE/COPERNICUS"),
}

# Define the output directory
output_directory = os.path.join(base_folder, "SECOND_STAGE")

# Define folder paths as a list of absolute paths
folder_paths = [
    directories["event_data"],
    directories["lab_logs"],
    directories["copernicus"],
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

print("Bringin' the data together:")
for path in file_paths:
    print(f"    {path}")
    print(f"     └──> {os.path.getsize(path)/1_048_576:.2f} MB")

print("\nDuplicate-count per file:")
for p in file_paths:
    # load only the Time column
    time_col = pd.read_csv(p, usecols=['Time'], parse_dates=['Time'])
    dup_total = time_col.duplicated().sum()
    max_per_ts = time_col.value_counts().iloc[0]
    print(f"  {os.path.basename(p):30}  total duplicates = {dup_total:,}   "
          f"max rows sharing one timestamp = {max_per_ts:,}")

print("\nFile diagnostics:")
for file_path in file_paths:
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        n_cols = len(header)
    file_size = os.path.getsize(file_path) / (1024**2)
    print(f"  {os.path.basename(file_path):<30}  →  columns = {n_cols:<4}  size = {file_size:6.2f} MB")


# Load all CSV files into dataframes and store them in a list


def aggregate_csv(path, chunksize=1_000_000):
    tmp = []
    for chunk in pd.read_csv(path, parse_dates=['Time'], chunksize=chunksize):
        chunk['Time'] = chunk['Time'].dt.floor('1min')
        grp = chunk.groupby('Time', sort=False).mean()
        tmp.append(grp)
    out = pd.concat(tmp).groupby('Time').mean()    # one row per minute
    return out


import gc
from pathlib import Path

tmp_parquets = []

for p in file_paths:
    df = aggregate_csv(p).astype("float32")
    pq_path = Path(p).with_suffix(".parquet")
    df.to_parquet(pq_path, compression="zstd")
    tmp_parquets.append(pq_path)
    del df
    gc.collect()
    print_memory_usage(f"after {pq_path.name}")

# Single pass, memory-light merge
merged = None
for pq in tmp_parquets:
    df = pd.read_parquet(pq)
    df.replace(0, pd.NA, inplace=True)
    if merged is None:
        merged = df
    else:
        # merged = pd.merge_asof(merged, df, on="Time")
        merged = merged.join(df, how="outer", sort=False)
    del df
    gc.collect()
    print_memory_usage(f"after merge {pq.name}")

merged_df = merged.reset_index()

# Sort by Time after merging
merged_df = merged_df.sort_values('Time')

# print("Merged DataFrame Info:")
# print(merged_df.info())

# print("First few rows of merged data:")
# print(merged_df.head())

# print("Last few rows of merged data:")
# print(merged_df.tail())

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

# Round the values to 2 decimal places for numeric columns
print("Rounding values to 2 decimal places for numeric columns...")
for col in combined_df.select_dtypes(include=['float32', 'float64']).columns:
    combined_df[col] = combined_df[col].round(2)
    
print("Saving the data...")
# Save the final dataframe to a CSV file
combined_df.to_csv(output_file, index=False)

print(f"Data has been merged and saved to {output_file}")

print('------------------------------------------------------')
print(f"merge_into_large_table.py completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('------------------------------------------------------')