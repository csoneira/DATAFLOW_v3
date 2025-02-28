#!/usr/bin/env python3
# -*- coding: utf-8 -*-

test = False

import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import cdsapi
import xarray as xr
import os


print('--------------------------- python reanalysis starts ---------------------------')


# def usage():
#     """Display the usage message and exit."""
#     print("""
#     This script retrieves ERA5 reanalysis data for a specific location and saves 
#     the processed results into a CSV file.

#     Features:
#       - Downloads 2m temperature data and 100 mbar geopotential/temperature data.
#       - Merges newly retrieved data with existing data in the CSV file.
#       - Processes data into clear and organized DataFrames.

#     Output:
#       - A CSV file named 'accumulated_reanalysis_data.csv'.

#     Note:
#       - Ensure you have access to the CDS API with valid credentials.
#       - The script does not accept any command-line arguments.

#     Example:
#       python3 <script_name>.py
#     """)
#     sys.exit(1)


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


# Location definition --------------------------------------------- good solution

station = int(station)

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


working_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/FIRST_STAGE/REANALYSIS")

# Define subdirectories relative to the working directory
base_directories = {
    "copernicus_directory": os.path.join(working_directory, "COPERNICUS_DATA"),
}

# Access the Copernicus directory
copernicus_directory = base_directories["copernicus_directory"]

os.makedirs(working_directory, exist_ok=True)
os.makedirs(copernicus_directory, exist_ok=True)

# Construct file paths
csv_file = os.path.join(working_directory, "big_reanalysis_data.csv")
nc_2m_temp_file = os.path.join(copernicus_directory, f"{location}_2m_temperature.nc")
nc_100mbar_file = os.path.join(copernicus_directory, f"{location}_100mbar_temperature_geopotential.nc")

# Define start date and file path
if test:
    start_date = datetime.now() - timedelta(weeks=2)
else:
    start_date = datetime(2023, 7, 1)
    
    
# Determine the data retrieval range
csv_exists = os.path.exists(csv_file)

if csv_exists:
    # Load existing data and find the last date
    print('File exists and is being loaded. Last date will be checked.')
    existing_df = pd.read_csv(csv_file, parse_dates=['Time'])
    last_date = existing_df['Time'].max()
    start_date = last_date - timedelta(days=1)  # Start from the previous day
    print(f'Last date is {last_date}, so loading data from {start_date}')
else:
    # If the file doesn't exist, create an empty DataFrame
    print('File does not exist so an empty dataframe is created.')
    existing_df = pd.DataFrame()

end_date = datetime.now()
print(f'Retrieving data from {start_date} to {end_date}')

# If no new data is needed, skip the retrieval
if start_date > end_date:
    print("No new data to retrieve.")
    sys.exit(0)

# Check if .nc files already exist and are loaded
temp_nc_file_exists = os.path.exists(nc_2m_temp_file)
mbar_nc_file_exists = os.path.exists(nc_100mbar_file)

if not temp_nc_file_exists or csv_exists:
    print('-------------------- Temperature files retrieving --------------------')
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date)
    years = list(date_range.year.astype(str).unique())
    months = list(date_range.month.map(lambda x: f'{x:02d}').unique())
    days = list(date_range.day.map(lambda x: f'{x:02d}').unique())
    times = [f'{hour:02d}:00' for hour in range(24)]  # Hourly data

    # Initialize the CDS API client
    c = cdsapi.Client()

    # Retrieve 2m temperature data (ground level)
    print('Retrieve 2m temperature data (ground level)')
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['2m_temperature'],
            'year': years,
            'month': months,
            'day': days,
            'time': times,
            'area': [
                latitude + 0.25, longitude - 0.25,
                latitude - 0.25, longitude + 0.25
            ],
            'format': 'netcdf'
        },
        nc_2m_temp_file
    )


if not mbar_nc_file_exists or csv_exists:
    print('-------------------- Pressure files retrieving --------------------')
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date)
    years = list(date_range.year.astype(str).unique())
    months = list(date_range.month.map(lambda x: f'{x:02d}').unique())
    days = list(date_range.day.map(lambda x: f'{x:02d}').unique())
    times = [f'{hour:02d}:00' for hour in range(24)]  # Hourly data

    # Initialize the CDS API client
    c = cdsapi.Client()

    # Retrieve temperature and geopotential height data at 100 mbar
    print('Retrieve temperature and geopotential height data at 100 mbar')
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['temperature', 'geopotential'],
            'pressure_level': ['100'],
            'year': years,
            'month': months,
            'day': days,
            'time': times,
            'area': [
                latitude + 0.25, longitude - 0.25,
                latitude - 0.25, longitude + 0.25
            ],
            'format': 'netcdf'
        },
        nc_100mbar_file
    )

print('DATA RETRIEVED or ALREADY LOADED!')
print('Processing and saving...')

# ------------------------------------------------------------------
# Load datasets and rename valid_time to time for ds_2m_temp
ds_2m_temp = xr.open_dataset(nc_2m_temp_file).rename({'valid_time': 'Time'})
ds_100mbar = xr.open_dataset(nc_100mbar_file).rename({'valid_time': 'Time'})

# Convert Kelvin to Celsius for variables
ground_temp = ds_2m_temp['t2m'] - 273.15  # Celsius
temp_100mbar = ds_100mbar['t'] - 273.15  # Celsius
geopotential_height_100mbar = ds_100mbar['z'] / 9.80665  # Convert geopotential to meters. This is not clear, since the acceleration changes with altitude...

# Convert datasets to DataFrames and reset index
df_ground_temp = ground_temp.to_dataframe().reset_index()
df_temp_100mbar = temp_100mbar.to_dataframe().reset_index()
df_geopotential_height_100mbar = geopotential_height_100mbar.to_dataframe().reset_index()

# Drop unnecessary columns
df_ground_temp = df_ground_temp.drop(columns=['expver', 'number'], errors='ignore')
df_temp_100mbar = df_temp_100mbar.drop(columns=['pressure_level', 'number', 'expver'], errors='ignore')

# Debug: Check the DataFrame structure
print("Columns in df_ground_temp:", df_ground_temp.columns)
print("Sample data in df_ground_temp:\n", df_ground_temp.head())

print("Columns in df_temp_100mbar:", df_temp_100mbar.columns)
print("Sample data in df_temp_100mbar:\n", df_temp_100mbar.head())

# Group by 'time' and calculate the mean for numeric columns
df_ground_temp = df_ground_temp.groupby('Time').mean(numeric_only=True).reset_index()
df_temp_100mbar = df_temp_100mbar.groupby('Time').mean(numeric_only=True).reset_index()
df_geopotential_height_100mbar = df_geopotential_height_100mbar.groupby('Time').mean(numeric_only=True).reset_index()

# Merge DataFrames on time
df_new = (
    df_ground_temp
    .merge(df_temp_100mbar, on='Time')
    .merge(df_geopotential_height_100mbar, on='Time')
)

df_new = df_new[['Time', 't2m', 't', 'z']]  # Keep only relevant columns

print("Columns in df_new:", df_new.columns)

# Rename columns for clarity
df_new.columns = ['Time', 'temp_ground', 'temp_100mbar', 'height_100mbar']

# Debug: Check the final merged DataFrame
print("Final merged DataFrame:\n", df_new.head())

# Merge with existing data
if not existing_df.empty:
    df_updated = pd.concat([existing_df, df_new]).drop_duplicates(subset=['Time']).sort_values(by='Time')
else:
    df_updated = df_new

# Resample the DataFrame to hourly data
# df_updated = df_updated.set_index('Time').resample('1H').mean().reset_index()

# Save the updated DataFrame
df_updated.to_csv(csv_file, index=False)
print(f"Data saved to {csv_file}.")

print('--------------------------- python reanalysis ends ---------------------------')

print('------------------------------------------------------')
print(f"reanalysis.py (Copernicus) completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('------------------------------------------------------')