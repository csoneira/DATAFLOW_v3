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
import numpy as np

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

print("\nAggregating data from CSV files...")
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

if os.path.exists(output_file):
    # If the output file already exists, load it and append new rows
    print("\nRemoving existing file: ", output_file)
    os.remove(output_file)
combined_df = merged_df


# Replace all 0s with NaNs
print("\nReplacing 0s with NaNs...")
combined_df.replace(0, pd.NA, inplace=True)


# Round the values to 2 decimal places for numeric columns
print("\nRounding values to 2 decimal places for columns (but Time)...")
num_cols = combined_df.columns.difference(['Time'])   # 306 columns here

# --- 1. Vectorised numeric coercion ---------------------------------
# stack → to_numeric → unstack is the fastest way to coerce many columns
tmp = (combined_df[num_cols]
         .stack(dropna=False))                # 1-D view
tmp = pd.to_numeric(tmp, errors='coerce')     # one pass
combined_df[num_cols] = tmp.unstack(level=1)  # back to 2-D

# --- 2. Single cast to float32 (memory halves) ----------------------
combined_df[num_cols] = combined_df[num_cols].astype('float32')

# --- 3. In-place NumPy rounding (no extra copy) ---------------------
vals = combined_df[num_cols].to_numpy()       # float32 view
np.round(vals, 2, out=vals)                   # modifies in place


# If there are rows where all non-Time columns are NaN, drop them and count how many there are
print("\nCounting rows with all NaN values (excluding 'Time')...")
non_time_cols = combined_df.columns.difference(['Time'])
nan_rows_mask = combined_df[non_time_cols].isna().all(axis=1)
nan_rows_count = nan_rows_mask.sum()
if nan_rows_count > 0:
    print(f"Dropping {nan_rows_count} rows with all NaN values (excluding 'Time')...")
    combined_df = combined_df[~nan_rows_mask]
else:
    print("No rows with all non-'Time' NaN values found.")


# Save the final dataframe to a CSV file
print("\nSaving the data...")
combined_df.to_csv(output_file, index=False)


print(f"Data has been merged and saved to {output_file}")

print('------------------------------------------------------')
print(f"merge_into_large_table.py completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('------------------------------------------------------')