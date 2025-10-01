from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Dec 18 2024

@author: csoneira@ucm.es
"""

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

import os
import sys
from glob import glob
from datetime import datetime
import numpy as np
import pandas as pd
import psutil
import gc
from pathlib import Path

use_reference = "--reference-event" in sys.argv or "-r" in sys.argv


# -----------------------------------------------------------------------------
# ----------------------------- Config file -----------------------------------
# -----------------------------------------------------------------------------

import os
import yaml
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/config.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]

DECIMAL_PLACES = config["DECIMAL_PLACES"]


# -----------------------------------------------------------------------------
def print_memory_usage(tag=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # in MB
    print(f"[{tag}] Memory usage: {mem:.2f} MB")
# -----------------------------------------------------------------------------

# Check input argument
if len(sys.argv) < 2:
    print("Error: No station provided.")
    print("Usage: python3 script.py <station>")
    sys.exit(1)

station = sys.argv[1]
print(f"Station: {station}")

base_folder = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")

directories = {
    "event_data": os.path.join(base_folder, "FIRST_STAGE/EVENT_DATA"),
    "lab_logs": os.path.join(base_folder, "FIRST_STAGE/LAB_LOGS"),
    "copernicus": os.path.join(base_folder, "FIRST_STAGE/COPERNICUS"),
}

output_directory = os.path.join(base_folder, "SECOND_STAGE")
os.makedirs(output_directory, exist_ok=True)
output_file = os.path.join(output_directory, "total_data_table.csv")

# Collect CSVs
file_paths = []
for path in directories.values():
    file_paths.extend(glob(os.path.join(path, "big*.csv")))

if not file_paths:
    raise FileNotFoundError("No CSV files found in the specified directories.")

print("Bringin' the data together:")
for path in file_paths:
    print(f"    {path}")
    print(f"     └──> {os.path.getsize(path)/1_048_576:.2f} MB")

print("\nDuplicate-count per file:")
for p in file_paths:
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

# -----------------------------------------------------------------------------
# CSV aggregation
# -----------------------------------------------------------------------------
def aggregate_csv(path, chunksize=1_000_000):
    tmp = []
    for chunk in pd.read_csv(path, parse_dates=['Time'], chunksize=chunksize):
        chunk['Time'] = chunk['Time'].dt.floor('1min')
        grp = chunk.groupby('Time', sort=False).mean()
        tmp.append(grp)
    return pd.concat(tmp).groupby('Time').mean()

tmp_parquets = []
for p in file_paths:
    df = aggregate_csv(p).astype("float32")
    pq_path = Path(p).with_suffix(".parquet")
    df.to_parquet(pq_path, compression="zstd")
    tmp_parquets.append(pq_path)
    del df
    gc.collect()
    print_memory_usage(f"after {pq_path.name}")

# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------



for pq in tmp_parquets:
    df = pd.read_parquet(pq)
    df.replace(0, np.nan, inplace=True)
    
    if 'event_data' in pq.as_posix():
        reference_index = df.index if use_reference else None

    if merged is None:
        merged = df
    else:
        if use_reference and reference_index is not None:
            df = df[df.index.isin(reference_index)]
        merged = merged.join(df, how="outer" if not use_reference else "left", sort=False)
    
    del df
    gc.collect()
    print_memory_usage(f"after merge {pq.name}")


merged_df = merged.reset_index()
merged_df = merged_df.sort_values('Time')

if os.path.exists(output_file):
    print("\nRemoving existing file: ", output_file)
    os.remove(output_file)

combined_df = merged_df

# -----------------------------------------------------------------------------
# Cleaning and Rounding
# -----------------------------------------------------------------------------
print("\nReplacing 0s and pd.NA with np.nan...")
combined_df = combined_df.replace([0, pd.NA], np.nan).infer_objects(copy=False)

print("\nRounding values to X decimal places for columns (but Time)...")
num_cols = combined_df.columns.difference(['Time'])


# Convert to float32 safely after coercion
combined_df[num_cols] = combined_df[num_cols].apply(pd.to_numeric, errors='coerce')
combined_df[num_cols] = combined_df[num_cols].astype('float32')

vals = combined_df[num_cols].to_numpy()
np.round(vals, DECIMAL_PLACES, out=vals)

# -----------------------------------------------------------------------------
# Drop rows where all non-Time columns are NaN
# -----------------------------------------------------------------------------
print("\nCounting rows with all NaN values (excluding 'Time')...")
non_time_cols = combined_df.columns.difference(['Time'])
nan_rows_mask = combined_df[non_time_cols].isna().all(axis=1)
nan_rows_count = nan_rows_mask.sum()
if nan_rows_count > 0:
    print(f"Dropping {nan_rows_count} rows with all NaN values (excluding 'Time')...")
    combined_df = combined_df[~nan_rows_mask]
else:
    print("No rows with all non-'Time' NaN values found.")

# -----------------------------------------------------------------------------
# Save output
# -----------------------------------------------------------------------------
print("\nSaving the data...")
combined_df.to_csv(output_file, index=False)

print(f"Data has been merged and saved to {output_file}")
print('------------------------------------------------------')
print(f"merge_into_large_table.py completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('------------------------------------------------------')
