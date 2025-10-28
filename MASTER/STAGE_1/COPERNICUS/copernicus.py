#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
from __future__ import annotations

print("__| |____________________________________________________________| |__")
print("__   ____________________________________________________________   __")
print("  | |                                                            | |  ")
print("  | |                            _           _                   | |  ")
print("  | | _ __ ___  __ _ _ __   __ _| |_   _ ___(_)___   _ __  _   _ | |  ")
print("  | || '__/ _ \\/ _` | '_ \\ / _` | | | | / __| / __| | '_ \\| | | || |  ")
print("  | || | |  __/ (_| | | | | (_| | | |_| \\__ \\ \\__ \\_| |_) | |_| || |  ")
print("  | ||_|  \\___|\\__,_|_| |_|\\__,_|_|\\__, |___/_|___(_) .__/ \\__, || |  ")
print("  | |                              |___/            |_|    |___/ | |  ")
print("__| |____________________________________________________________| |__")
print("__   ____________________________________________________________   __")
print("  | |                                                            | |  ")


# The first data is available five days ago from today
# The maximum number of weeks to retrieve is 21 weeks, else the API will return an error,
# so the data range must be split in sets of 21 seeks, trying the ranges to have less than 21
# weeks but the same size to each other.


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TO PUT INTO CONGIF FILES LATER ----------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# test = False
# weeks_behind_requested = 1 # 21 maxixum
# degree_apotema = 0.25  # 0.25 degrees apotema for the area around the station

import os
import sys
from pathlib import Path

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

test = config["test_mode"]
weeks_behind_requested = config["weeks_behind_requested"]
max_weeks_allowed = config["max_weeks_allowed"]
degree_apotema = config["degree_apotema"]




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

# Standard Library
import math
import os
import sys
from datetime import datetime, timedelta

# Third-party Libraries
import numpy as np
import pandas as pd
import cdsapi
import xarray as xr

# -----------------------------------------------------------------------------

print('---------------- python copernicus retrieval starts ------------------')

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


# Location definition ------------------------------------------- good solution

station = int(station)
set_station(station)

# Define a dictionary to store location data
locations = {
    1: {"name": "Madrid", "latitude": 40.4168, "longitude": -3.7038},
    2: {"name": "Warsaw", "latitude": 52.2297, "longitude": 21.0122},
    3: {"name": "Puebla", "latitude": 19.0413, "longitude": -98.2062},
    4: {"name": "Monterrey", "latitude": 25.6866, "longitude": -100.3161},
}

# Get the location details for the specified station
if station in locations:
    location = locations[station]["name"]
    latitude = locations[station]["latitude"]
    longitude = locations[station]["longitude"]
else:
    raise ValueError(f"Invalid station number: {station}")

# ------------------------------------------------------------------


working_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/STAGE_1/COPERNICUS")

status_csv_path = os.path.join(working_directory, "copernicus_status.csv")
status_timestamp = append_status_row(status_csv_path)

# Define subdirectories relative to the working directory
base_directories = { "copernicus_directory": os.path.join(working_directory, "COPERNICUS_DATA"), }

# Access the Copernicus directory
copernicus_directory = base_directories["copernicus_directory"]

os.makedirs(working_directory, exist_ok=True)
os.makedirs(copernicus_directory, exist_ok=True)

# Construct file paths
csv_file = os.path.join(working_directory, "big_copernicus_data.csv")
nc_2m_temp_file = os.path.join(copernicus_directory, f"{location}_2m_temperature.nc")
nc_100mbar_file = os.path.join(copernicus_directory, f"{location}_100mbar_temperature_geopotential.nc")

# Define start date and file path
if test:
    start_date = datetime.now() - timedelta(weeks=weeks_behind_requested)
else:
    start_date = datetime(2023, 7, 1)

end_date = datetime.now() - timedelta(days=5)

# Round start_date and end_date to day
start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

start_date_og = start_date
end_date_og = end_date

exit_code = False

# Determine the data retrieval range
csv_exists = os.path.exists(csv_file)

if csv_exists:
    # Load existing data and find the first and last dates
    print('File exists and is being loaded. Checking the date range.')
    existing_df = pd.read_csv(csv_file, parse_dates=['Time'])
    first_date = existing_df['Time'].min()  # First date in the file
    last_date = existing_df['Time'].max()   # Last date in the file
    
    # Round to Day
    first_date = first_date.replace(hour=0, minute=0, second=0, microsecond=0)
    last_date = last_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    # Check if the file already covers the requested date range
    if first_date <= start_date and last_date >= end_date:
        # If the file already covers the requested range, no need to reload
        print(f"The file already contains the data for the range selected. Skipping data retrieval.")
        exit_code = True
    else:
        # If the range is not covered, extend the range and reload
        print(f"The file doesn't contain the full range selected. Loading missing data.")
        
        start_date = first_date if first_date < start_date else start_date
        end_date = last_date if last_date > end_date else end_date
    
    # Printing the table
    print("\n")
    date_ranges = pd.DataFrame({
        'Start Date': [start_date_og, first_date, start_date],
        'End Date': [end_date_og, last_date, end_date]
    }, index=['Requested Dates', 'File Dates', 'Final Dates'])
    date_ranges = date_ranges.map(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else x)
    print(date_ranges)
    
else:
    # If the file doesn't exist, create an empty DataFrame
    print('File does not exist, creating an empty DataFrame.')
    existing_df = pd.DataFrame()


if exit_code:
    sys.exit(0)

print(f'\nRetrieving data from\n    {start_date} to\n    {end_date}\n')

# If no new data is needed, skip the retrieval
if start_date > end_date:
    print("No new data to retrieve.")
    sys.exit(0)


# ------------------------------------------------------------------
# Helper: split the overall period in chunks of ≤ 20 weeks
# ------------------------------------------------------------------

MAX_WEEKS = max_weeks_allowed  # CDS allows < 21 weeks, but this is faster
DAY      = timedelta(days=1)
WEEK     = timedelta(weeks=1)

def block_boundaries(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Return a list of (block_start, block_end) tuples."""
    tot_weeks = math.ceil((end - start + DAY) / WEEK)          # inclusive
    n_blocks  = math.ceil(tot_weeks / MAX_WEEKS)               # minimum blocks needed
    blk_weeks = math.ceil(tot_weeks / n_blocks)                # weeks per block (≤ MAX_WEEKS)
    blk_weeks = min(blk_weeks, MAX_WEEKS)
    blocks = []
    s = start
    while s <= end:
        e = min(s + blk_weeks*WEEK - DAY, end)
        blocks.append((s, e))
        s = e + DAY
    return blocks

# ------------------------------------------------------------------
# Retrieval loop
# ------------------------------------------------------------------

print('-------------------- Copernicus retrieval --------------------')
blocks = block_boundaries(start_date, end_date)
print(f'Request split into {len(blocks)} block(s) of ≤ {MAX_WEEKS} weeks each.\n')

# Print the duration of the blocks
for b_start, b_end in blocks:
    duration = (b_end - b_start).days + 1  # inclusive
    print(f'Block {b_start:%Y-%m-%d} → {b_end:%Y-%m-%d} ({duration} days)')


frames_ground      : list[pd.DataFrame] = []
frames_temp100     : list[pd.DataFrame] = []
frames_geopot100   : list[pd.DataFrame] = []

c = cdsapi.Client()                       # initialise once

for b_start, b_end in blocks:
    print('\n--------------------------------------------------------------')
    print(f'\nBlock {b_start:%Y-%m-%d} → {b_end:%Y-%m-%d}')

    date_range = pd.date_range(b_start, b_end)
    years  = list(date_range.year.astype(str).unique())
    months = list(date_range.month.map(lambda x: f'{x:02d}').unique())
    days   = list(date_range.day.map(lambda x: f'{x:02d}').unique())
    times  = [f'{h:02d}:00' for h in range(24)]
    
    # ------- 2 m air temperature ----------------------------------
    print("\nGround level temperature")
    tmp_file = nc_2m_temp_file.replace('.nc',
               f'_{b_start:%Y%m%d}_{b_end:%Y%m%d}.nc')
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable'    : ['2m_temperature'],
            'year'        : years,
            'month'       : months,
            'day'         : days,
            'time'        : times,
            'area'        : [ latitude + degree_apotema,
                              longitude - degree_apotema,
                              latitude - degree_apotema,
                              longitude + degree_apotema ],
            'format'      : 'netcdf',                                  # Consider changing to 'grib' for larger datasets, which is faster, apparently
        },
        tmp_file
    )
    
    # ------- 100 mbar temperature & height-----------------
    print("\n100 mbar temperature & geopotential")
    prs_file = nc_100mbar_file.replace('.nc',
               f'_{b_start:%Y%m%d}_{b_end:%Y%m%d}.nc')
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type' : 'reanalysis',
            'variable'     : ['temperature', 'geopotential'],
            'pressure_level': ['100'],
            'year'         : years,
            'month'        : months,
            'day'          : days,
            'time'         : times,
            'area'         : [ latitude + degree_apotema,
                               longitude - degree_apotema,
                               latitude - degree_apotema,
                               longitude + degree_apotema ],
            'format'       : 'netcdf',
        },
        prs_file
    )

    # ------- Load and process ------------------------------------
    ds_2m   = xr.open_dataset(tmp_file).rename({'valid_time': 'Time'})
    ds_100  = xr.open_dataset(prs_file).rename({'valid_time': 'Time'})

    df_ground   = (ds_2m['t2m'] - 273.15).to_dataframe().reset_index()
    df_temp100  = (ds_100['t']  - 273.15).to_dataframe().reset_index()
    df_geop100  = (ds_100['z'] / 9.80665).to_dataframe().reset_index()

    frames_ground    .append(df_ground .groupby('Time').mean(numeric_only=True).reset_index())
    frames_temp100   .append(df_temp100.groupby('Time').mean(numeric_only=True).reset_index())
    frames_geopot100 .append(df_geop100.groupby('Time').mean(numeric_only=True).reset_index())


# ------------------------------------------------------------------
# Concatenate all chunks and finish as before
# ------------------------------------------------------------------
df_ground_all  = pd.concat(frames_ground   , ignore_index=True)
df_temp100_all = pd.concat(frames_temp100  , ignore_index=True)
df_geop100_all = pd.concat(frames_geopot100, ignore_index=True)

df_new = (df_ground_all
          .merge(df_temp100_all,  on='Time')
          .merge(df_geop100_all, on='Time')
          .loc[:, ['Time', 't2m', 't', 'z']]
          .rename(columns={'t2m':'temp_ground',
                           't'  :'temp_100mbar',
                           'z'  :'height_100mbar'}))

df_new = df_new[['Time', 'temp_ground', 'temp_100mbar', 'height_100mbar']]  # Keep only relevant columns

print("Columns in df_new:", df_new.columns)

# Rename columns for clarity
# df_new.columns = ['Time', 'temp_ground', 'temp_100mbar', 'height_100mbar']

# Debug: Check the final merged DataFrame
print("Final merged DataFrame:\n", df_new.head())

# Merge with existing data
if not existing_df.empty:
    df_updated = pd.concat([existing_df, df_new]).drop_duplicates(subset=['Time']).sort_values(by='Time')
else:
    df_updated = df_new

# Save the updated DataFrame
df_updated.to_csv(csv_file, index=False)
print(f"Data saved to {csv_file}.")

print('--------------------------- python copernicus ends ---------------------------')

print('------------------------------------------------------')
print(f"copernicus.py (Copernicus) completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('------------------------------------------------------')

mark_status_complete(status_csv_path, status_timestamp)
