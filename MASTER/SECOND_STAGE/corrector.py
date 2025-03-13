#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jun 24 19:02:22 2024

@author: gfn
"""

# globals().clear()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import sys


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
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# This works fine for total counts:
# resampling_window = '10T'  # '10T' # '5T' stands for 5 minutes. Adjust based on your needs.
# HMF_ker = 5 # It must be odd.
# MAF_ker = 1
# skip_in_limits = 1

remove_outliers = True

# Plotting configuration
show_plots = False
save_plots = True
create_plots = True
show_errorbar = False

recalculate_pressure_coeff = True

res_win_min = 30 # 180 Resampling window minutes
# HMF_ker = 1 # It must be odd. Horizontal Median Filter
# MAF_ker = 1 # Moving Average Filter

outlier_filter = 3 #3

high_order_correction = True
date_selection = True  # Set to True if you want to filter by date

skip_in_limits = 1

# This should come from an input file
eta_P = -0.162 # pressure_coeff_input
unc_eta_P = 0.013 # unc_pressure_coeff_input
set_a = -0.11357 # pressure_intercept_input
mean_pressure_used_for_the_fit = 940

systematic_unc = [0, 0, 0, 0] # From simulation
# acceptance_factor = [0.7, 1, 1, 0.8] # From simulation

systematic_unc_corr_to_real_rate = 0
z_score_th_pres_corr = 1

# -----------------------------------------------------------------------------

# Define the base folder and file paths
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")

base_folder = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/SECOND_STAGE")
filepath = f"{base_folder}/total_data_table.csv"
save_filename = f"{base_folder}/large_corrected_table.csv"
grafana_save_filename = f"{base_folder}/data_for_grafana_{station}.csv"

figure_path = f"{base_folder}/correction.png"



# -----------------------------------------------------------------------------
# To not touch unless necesary ------------------------------------------------
# -----------------------------------------------------------------------------

resampling_window = f'{res_win_min}min'  # '10min' # '5min' stands for 5 minutes.

# Columns to sum
angular_regions = ['High', 'N', 'S', 'E', 'W']
detection_types = ['type_1234', 'type_123', 'type_234', 'type_124', 'type_134', 'type_12', 'type_23', 'type_34', 'type_13', 'type_14', 'type_24']
# charge_types = ['count_in_1_sum', 'avalanche_1_sum', 'streamer_1_sum',
#                 'count_in_2_sum', 'avalanche_2_sum', 'streamer_2_sum',
#                 'count_in_3_sum', 'avalanche_3_sum', 'streamer_3_sum',
#                 'count_in_4_sum', 'avalanche_4_sum', 'streamer_4_sum']

charge_types = ['count_in_1', 'avalanche_1', 'streamer_1',
                'count_in_2', 'avalanche_2', 'streamer_2',
                'count_in_3', 'avalanche_3', 'streamer_3',
                'count_in_4', 'avalanche_4', 'streamer_4']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Function definition ---------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Introduction ----------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Add a part that sets limits on the date to complete only the lacking part in 
# the large corrected table.

# -----------------------------------------------------------------------------
# Reading ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Load the data
print('Reading the big CSV datafile...')
data_df = pd.read_csv(filepath)

print("Putting zeroes to NaNs...")
data_df = data_df.replace(0, np.nan)

print(filepath)
print('File loaded successfully.')

# Define input file path -----------------------------------------------------
input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    input_file = pd.read_csv(input_file_config_path, skiprows=1, decimal = ",")
    
    print("Input configuration file found.")
    exists_input_file = True
    
else:
    exists_input_file = False
    print("Input configuration file does not exist.")


# Preprocess the data to remove rows with invalid datetime format
print('Validating datetime format in "Time" column...')
try:
    # Try parsing 'Time' column with the specified format
    data_df['Time'] = pd.to_datetime(data_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
except Exception as e:
    print(f"Error while parsing datetime: {e}")
    exit(1)

# Drop rows where 'Time' could not be parsed
invalid_rows = data_df['Time'].isna().sum()
if invalid_rows > 0:
    print(f"Removing {invalid_rows} rows with invalid datetime format.")
    data_df = data_df.dropna(subset=['Time'])
else:
    print("No rows with invalid datetime format removed.")

print('Datetime validation completed successfully.')

min_time_original = data_df['Time'].min()
max_time_original = data_df['Time'].max()

# Check if the results file exists
if os.path.exists(save_filename):
    results_df = pd.read_csv(save_filename)
    
    # Validate and clean datetime format in results_df as well
    results_df['Time'] = pd.to_datetime(results_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    results_df = results_df.dropna(subset=['Time'])
    
    # Define start and end dates based on the last date in results_df
    last_date = results_df['Time'].max()  # Convert to datetime
    start_date = last_date - timedelta(weeks=5)
else:
    # If results_df does not exist, do not set date limits
    start_date = None

# Define the end date as today
end_date = datetime.now()

if date_selection:
# if date_selection and start_date is not None:
    start_date = pd.to_datetime("2024-03-23 00:00:00")  # Use a string in 'YYYY-MM-DD' format
    end_date = pd.to_datetime("2024-04-01 00:00:00")
    
    # start_date = pd.to_datetime("2024-05-08 00:00:00")  # Use a string in 'YYYY-MM-DD' format
    # end_date = pd.to_datetime("2024-06-30 00:00:00")
    
    # start_date = pd.to_datetime("2025-01-02")  # Use a string in 'YYYY-MM-DD' format
    # start_date = pd.to_datetime("2025-02-17 12:00:00")  # Use a string in 'YYYY-MM-DD' format
    # end_date = pd.to_datetime("2025-03-12 11:00")
    
    print("------- SELECTION BY DATE IS BEING PERFORMED -------")
    data_df = data_df[(data_df['Time'] >= start_date) & (data_df['Time'] <= end_date)]

print(f"Filtered data contains {len(data_df)} rows.")


# -----------------------------------------------------------------------------
# Outlier removal -------------------------------------------------------------
# -----------------------------------------------------------------------------

if remove_outliers:
    print('Removing outliers and zero values...')
    def remove_outliers_and_zeroes(series, z_thresh=outlier_filter):
        """
        Create a mask of rows that are outliers or have zero values.
        """
        # median = series.mean()
        median = series.median()
        std = series.std()
        z_scores = abs((series - median) / std)
        
        plt.hist(z_scores, bins=300)
        plt.title('Z-Scores Distribution')
        plt.xlabel('Z-Score')
        plt.ylabel('Frequency')
        if show_plots: 
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + "_original_z"
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format = 'png', dpi = 300)
        plt.close()
        
        # print(z_scores)
        # Create a mask for rows where z_scores > z_thresh or values are zero
        mask = (z_scores > z_thresh) | (series == 0)
        return mask

    # Initialize a mask of all False, meaning no rows are removed initially
    rows_to_remove = pd.Series(False, index=data_df.index)
    rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df['events'])
    # for region in angular_regions:
    #     rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df[region])

    data_df_cleaned = data_df[~rows_to_remove].copy()

    # Now, `data_df_cleaned` contains only rows that pass the conditions.
    print(f"Original DataFrame shape: {data_df.shape}")
    print(f"Cleaned DataFrame shape: {data_df_cleaned.shape}")

    data_df = data_df_cleaned.copy()


# -----------------------------------------------------------------------------
# Resampling the data in a larger time window: some averaged, some summed -----
# -----------------------------------------------------------------------------

data_df.set_index('Time', inplace=True)
data_df["number_of_rows"] = 1
columns_to_sum = angular_regions + detection_types + charge_types + ["number_of_rows"] + ["events"]
columns_to_mean = [col for col in data_df.columns if col not in columns_to_sum]

# Custom aggregation function
data_df = data_df.resample(resampling_window).agg({
    **{col: 'sum' for col in columns_to_sum},   # Sum the count and region columns
    **{col: 'mean' for col in columns_to_mean}  # Mean for the other columns
})

data_df.reset_index(inplace=True)

# --------------------------------------------------------------------------------
# Calculating a 'counts' column that is the sum of all regions and its uncertainty
# --------------------------------------------------------------------------------

data_df['count_regions'] = data_df[angular_regions].sum(axis=1)
data_df['count_regions_uncertainty'] = np.sqrt(data_df['count_regions'])

data_df['count_types'] = data_df[detection_types].sum(axis=1)
data_df['count_types_uncertainty'] = np.sqrt(data_df['count_types'])

# ------------------------------------------------------------------------------------------
data_df['count'] = data_df['events'] # Until I have a better definition of the count
data_df['count_uncertainty'] = np.sqrt(data_df['count'])
# ------------------------------------------------------------------------------------------

data_df['rate'] = data_df['count'] / ( data_df["number_of_rows"] * 60 )  # Counts per second (Hz)
data_df['unc_rate'] = np.sqrt(data_df['count']) / ( data_df["number_of_rows"] * 60 )

# Other calculations
data_df['hv_mean'] = ( data_df['hv_HVneg'] + data_df['hv_HVpos'] ) / 2


# Input file reading --------------------------------------------------------------------------------------------------------

if exists_input_file:
    start_time = min_time_original
    end_time = max_time_original

    input_file["start"] = pd.to_datetime(input_file["start"])
    input_file["end"] = pd.to_datetime(input_file["end"])

    # Ensure no NaN in 'end' column
    input_file["end"].fillna(pd.to_datetime('now'), inplace=True)

    # Create an empty dictionary to hold new column values
    new_columns = {
        "acc_1": [],
        "acc_2": [],
        "acc_3": [],
        "acc_4": []
    }

    # Assign values based on corresponding time range
    for timestamp in data_df["Time"]:
        # Find matching configuration
        match = input_file[ (input_file["start"] <= timestamp) & (input_file["end"] >= timestamp) ]
        
        if not match.empty:
            # Take the first matching row
            selected_conf = match.iloc[0]
            # print("Configuration number is: ", selected_conf["conf"])
            new_columns["acc_1"].append(selected_conf.get("acc_1", 1))
            new_columns["acc_2"].append(selected_conf.get("acc_2", 1))
            new_columns["acc_3"].append(selected_conf.get("acc_3", 1))
            new_columns["acc_4"].append(selected_conf.get("acc_4", 1))
        else:
            # print("No matching configuration, fill with 1...")
            new_columns["acc_1"].append(1)
            new_columns["acc_2"].append(1)
            new_columns["acc_3"].append(1)
            new_columns["acc_4"].append(1)

    # Convert dictionary to DataFrame
    df_new_cols = pd.DataFrame(new_columns)

    # Merge with the original dataframe
    df_extended = pd.concat([data_df, df_new_cols], axis=1)

    # Fill missing values with the original values where applicable
    df_extended.fillna(method='ffill', inplace=True)

    # Print the new columns
    data_df = df_extended
    
else:
    print("No input file found. Default values set.")
    data_df["acc_1"] = 1
    data_df["acc_2"] = 1
    data_df["acc_3"] = 1
    data_df["acc_4"] = 1
    
# ---------------------------------------------------------------------------------------------------------------------

# Turn all the acc_* into numeric columns
data_df["acc_1"] = pd.to_numeric(data_df["acc_1"])
data_df["acc_2"] = pd.to_numeric(data_df["acc_2"])
data_df["acc_3"] = pd.to_numeric(data_df["acc_3"])
data_df["acc_4"] = pd.to_numeric(data_df["acc_4"])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Efficiency calculation and corrections --------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Calculate efficiency and uncertainty of the efficiency
# -----------------------------------------------------------------------------

print('Calculating efficiency...')

# Define the function to calculate efficiency uncertainty
def calculate_efficiency_uncertainty(N_measured, N_passed):
    if N_passed > 0:
        return np.sqrt((N_measured / N_passed**2) + (N_measured**2 / N_passed**3))
    else:
        return np.nan

# Create explicit columns for detected and passed
data_df['detected_1'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134']].sum(axis=1, skipna=True)
data_df['detected_2'] = data_df[['type_1234', 'type_123', 'type_234', 'type_124']].sum(axis=1, skipna=True)
data_df['detected_3'] = data_df[['type_1234', 'type_123', 'type_234', 'type_134']].sum(axis=1, skipna=True)
data_df['detected_4'] = data_df[['type_1234', 'type_234', 'type_124', 'type_134']].sum(axis=1, skipna=True)

data_df['passed_1'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134', 'type_234']].sum(axis=1, skipna=True)
data_df['passed_2'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134', 'type_234']].sum(axis=1, skipna=True)
data_df['passed_3'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134', 'type_234']].sum(axis=1, skipna=True)
data_df['passed_4'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134', 'type_234']].sum(axis=1, skipna=True)

# Interpolate all the NaNs
# data_df = data_df.interpolate(method='linear', limit_direction='both')

print('Detected and passed calculated.')

# Set columns to 0 if detected or passed values are NaN or 0
def handle_zero_or_nan(row, detected_col, passed_col):
    if row[detected_col] == 0 or row[passed_col] == 0:
        return np.nan
    return row[detected_col] / row[passed_col]

def handle_uncertainty(row, detected_col, passed_col):
    if row[detected_col] == 0 or row[passed_col] == 0:
        return np.nan
    return calculate_efficiency_uncertainty(row[detected_col], row[passed_col])

print("Calculating efficiencies...")

# Calculate efficiencies and uncertainties explicitly
from scipy.ndimage import gaussian_filter1d

# Replace all zeroes by NaNs and do not interpolate
data_df = data_df.replace(0, np.nan)

rolling_effs = False
if rolling_effs:
    cols_to_interpolate = ['detected_1', 'detected_2', 'detected_3', 'detected_4', 'passed_1', 'passed_2', 'passed_3', 'passed_4']
    data_df[cols_to_interpolate] = data_df[cols_to_interpolate].replace(0, np.nan).interpolate(method='linear')
    
    sum_window, med_window = 5, 1
    rolling_sum, rolling_median = True, True
    
    if rolling_median:
        for col in cols_to_interpolate:
            data_df[col] = medfilt(data_df[col], kernel_size=med_window)
    
    if rolling_sum:
        for col in cols_to_interpolate:
            data_df[col] = data_df[col].rolling(window=sum_window, center=True, min_periods=1).sum()
    

data_df['eff_1'] = data_df.apply(lambda row: handle_zero_or_nan(row, 'detected_1', 'passed_1'), axis=1)
data_df['eff_2'] = data_df.apply(lambda row: handle_zero_or_nan(row, 'detected_2', 'passed_2'), axis=1)
data_df['eff_3'] = data_df.apply(lambda row: handle_zero_or_nan(row, 'detected_3', 'passed_3'), axis=1)
data_df['eff_4'] = data_df.apply(lambda row: handle_zero_or_nan(row, 'detected_4', 'passed_4'), axis=1)

data_df['anc_unc_eff_1'] = data_df.apply(lambda row: handle_uncertainty(row, 'detected_1', 'passed_1'), axis=1)
data_df['anc_unc_eff_2'] = data_df.apply(lambda row: handle_uncertainty(row, 'detected_2', 'passed_2'), axis=1)
data_df['anc_unc_eff_3'] = data_df.apply(lambda row: handle_uncertainty(row, 'detected_3', 'passed_3'), axis=1)
data_df['anc_unc_eff_4'] = data_df.apply(lambda row: handle_uncertainty(row, 'detected_4', 'passed_4'), axis=1)

# Add the systematic uncertainties to the efficiency calculation
data_df['unc_eff_1'] = np.sqrt( data_df['anc_unc_eff_1']**2 + systematic_unc[0]**2 )
data_df['unc_eff_2'] = np.sqrt( data_df['anc_unc_eff_2']**2 + systematic_unc[1]**2 )
data_df['unc_eff_3'] = np.sqrt( data_df['anc_unc_eff_3']**2 + systematic_unc[2]**2 )
data_df['unc_eff_4'] = np.sqrt( data_df['anc_unc_eff_4']**2 + systematic_unc[3]**2 )

data_df['final_eff_1'] = data_df['eff_1'] / data_df['acc_1']
data_df['final_eff_2'] = data_df['eff_2'] / data_df['acc_2']
data_df['final_eff_3'] = data_df['eff_3'] / data_df['acc_3']
data_df['final_eff_4'] = data_df['eff_4'] / data_df['acc_4']

data_df['unc_final_eff_1'] = data_df['unc_eff_1'] / data_df['acc_1']
data_df['unc_final_eff_2'] = data_df['unc_eff_2'] / data_df['acc_2']
data_df['unc_final_eff_3'] = data_df['unc_eff_3'] / data_df['acc_3']
data_df['unc_final_eff_4'] = data_df['unc_eff_4'] / data_df['acc_4']

# Calculate the average efficiency
# data_df['eff_global'] = data_df[['eff_1', 'eff_2', 'eff_3', 'eff_4']].mean(axis=1)
data_df['eff_global'] = data_df[['final_eff_1', 'final_eff_2', 'final_eff_3']].mean(axis=1)
# data_df['eff_global'] = data_df[['final_eff_1', 'final_eff_2', 'final_eff_3']].median(axis=1)

# Calculate the uncertainty for the average efficiency
data_df['unc_eff_global'] = np.sqrt(
    (data_df['unc_eff_1'] ** 2 +
     data_df['unc_eff_2'] ** 2 +
     data_df['unc_eff_3'] ** 2 +
     data_df['unc_eff_4'] ** 2) / 4
)


# -------------------------------------------------------------------------------------------------------------------------
# Calculate the systematic for the efficiencies ---------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

# 1
if create_plots:
    filtered_df = data_df.dropna(subset=['final_eff_1', 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    x = filtered_df['sensors_ext_Pressure_ext'].values.reshape(-1, 1)
    y = filtered_df['sensors_ext_Temperature_ext'].values.reshape(-1, 1)
    z = filtered_df['final_eff_1'].values
    X = np.hstack((x, y))
    model = LinearRegression()
    model.fit(X, z)
    a, b = model.coef_
    c = model.intercept_
    formula = f"Eff = {a:.4g} * P + {b:.4g} * T + {c:.4g}"
    data_df['eff_fit_1'] = a * data_df['sensors_ext_Pressure_ext'] + b * data_df['sensors_ext_Temperature_ext'] + c
    data_df['unc_eff_fit_1'] = 1

# 2
if create_plots:
    filtered_df = data_df.dropna(subset=['final_eff_2', 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    x = filtered_df['sensors_ext_Pressure_ext'].values.reshape(-1, 1)
    y = filtered_df['sensors_ext_Temperature_ext'].values.reshape(-1, 1)
    z = filtered_df['final_eff_2'].values
    X = np.hstack((x, y))
    model = LinearRegression()
    model.fit(X, z)
    a, b = model.coef_
    c = model.intercept_
    formula = f"Eff = {a:.4g} * P + {b:.4g} * T + {c:.4g}"
    data_df['eff_fit_2'] = a * data_df['sensors_ext_Pressure_ext'] + b * data_df['sensors_ext_Temperature_ext'] + c
    data_df['unc_eff_fit_2'] = 1

# 3
if create_plots:
    filtered_df = data_df.dropna(subset=['final_eff_3', 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    x = filtered_df['sensors_ext_Pressure_ext'].values.reshape(-1, 1)
    y = filtered_df['sensors_ext_Temperature_ext'].values.reshape(-1, 1)
    z = filtered_df['final_eff_3'].values
    X = np.hstack((x, y))
    model = LinearRegression()
    model.fit(X, z)
    a, b = model.coef_
    c = model.intercept_
    formula = f"Eff = {a:.4g} * P + {b:.4g} * T + {c:.4g}"
    data_df['eff_fit_3'] = a * data_df['sensors_ext_Pressure_ext'] + b * data_df['sensors_ext_Temperature_ext'] + c
    data_df['unc_eff_fit_3'] = 1

# 4
if create_plots:
    filtered_df = data_df.dropna(subset=['final_eff_4', 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    x = filtered_df['sensors_ext_Pressure_ext'].values.reshape(-1, 1)
    y = filtered_df['sensors_ext_Temperature_ext'].values.reshape(-1, 1)
    z = filtered_df['final_eff_4'].values
    X = np.hstack((x, y))
    model = LinearRegression()
    model.fit(X, z)
    a, b = model.coef_
    c = model.intercept_
    formula = f"Eff = {a:.4g} * P + {b:.4g} * T + {c:.4g}"
    data_df['eff_fit_4'] = a * data_df['sensors_ext_Pressure_ext'] + b * data_df['sensors_ext_Temperature_ext'] + c
    data_df['unc_eff_fit_4'] = 1

# -----------------------------------------------------------------------------
# Correct by eff using the equation system ------------------------------------
# -----------------------------------------------------------------------------

if create_plots:
    print("Creating efficiency and rate plots...")
    
    # Create four subplots in a single column, sharing the same x-axis
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(17, 14), sharex=True)
    
    for i in range(1, 5):  # Loop from 1 to 4
        ax = axes[i-1]  # pick the appropriate subplot
        
        # Plot final_eff_i
        ax.plot(data_df['Time'], data_df[f'final_eff_{i}'], 
                label=f'Eff. {i}', color=f'C{i}', alpha=0.5)
        
        # Plot eff_fit_i
        ax.plot(data_df['Time'], data_df[f'eff_fit_{i}'], 
                label=f'Eff. {i} Fit', color=f'C{i}', alpha=1)
        
        # Fill between uncertainty bands
        ax.fill_between(data_df['Time'],
                        data_df[f'final_eff_{i}'] - data_df[f'unc_final_eff_{i}'],
                        data_df[f'final_eff_{i}'] + data_df[f'unc_final_eff_{i}'],
                        color=f'C{i}', alpha=0.2)
        
        # Labeling and titles
        ax.set_ylabel('Efficiency')
        ax.set_ylim(0.9, 1.0)
        ax.set_title(f'Efficiency {i} over Time')
        ax.legend(loc='upper left')
    
    # Label the common x-axis at the bottom
    axes[-1].set_xlabel('Time')

    plt.tight_layout()

    # Save or show the plot
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + "_start.png"
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)

    plt.close()


from scipy.optimize import least_squares

##############################
# 1) Detection Probability
##############################

def detection_prob(pattern, e1, e2, e3, e4, subset):
    """Compute the probability of detecting a muon in 'pattern' given 'subset'."""
    if pattern == '1234':
        return e1*e2*e3*e4 if subset == '1234' else 0.0
    if pattern == '123':
        return e1*e2*e3 if subset == '123' else (e1*e2*e3*(1-e4) if subset == '1234' else 0.0)
    if pattern == '234':
        return e2*e3*e4 if subset == '234' else ((1-e1)*e2*e3*e4 if subset == '1234' else 0.0)
    if pattern == '124':
        return e1*e2*e4*(1-e3) if subset == '1234' else 0.0
    if pattern == '134':
        return e1*e3*e4*(1-e2) if subset == '1234' else 0.0
    if pattern == '12':
        if subset == '12':
            return e1*e2
        elif subset == '123':
            return e1*e2*(1-e3)
        elif subset == '1234':
            return e1*e2*(1-e3)*(1-e4)
        return 0.0
    if pattern == '23':
        if subset == '23':
            return e2*e3
        elif subset == '123':
            return (1-e1)*e2*e3
        elif subset == '234':
            return e2*e3*(1-e4)
        elif subset == '1234':
            return (1-e1)*e2*e3*(1-e4)
        return 0.0
    if pattern == '34':
        if subset == '34':
            return e3*e4
        elif subset == '234':
            return (1-e2)*e3*e4
        elif subset == '1234':
            return (1-e1)*(1-e2)*e3*e4
        return 0.0
    if pattern == '13':
        if subset == '123':
            return e1*(1-e2)*e3
        elif subset == '1234':
            return e1*(1-e2)*e3*(1-e4)
        return 0.0
    return 0.0


##############################
# 2) Solve for subset rates
##############################

def solve_for_subsets(row):
    """
    Solve for x_12, x_23, x_34, x_123, x_234, x_1234
    while taking e1, e2, e3, e4 from 'eff_global', 'final_eff_2', 'final_eff_3', and 'eff_global'.
    """
    # Gather measured patterns (fill NaNs with 0)
    measured_patterns = [
        ('12',   row.get('type_12', 0)),
        ('13',   row.get('type_13', 0)),
        ('23',   row.get('type_23', 0)),
        ('34',   row.get('type_34', 0)),
        ('123',  row.get('type_123', 0)),
        ('124',  row.get('type_124', 0)),
        ('134',  row.get('type_134', 0)),
        ('234',  row.get('type_234', 0)),
        ('1234', row.get('type_1234', 0)),
    ]
    # Convert any NaNs to 0
    measured_patterns = [(pat, 0 if pd.isna(v) else v) for (pat, v) in measured_patterns]

    # Fix plane efficiencies from columns (not fitting them!)
    e1_val = row.get('eff_fit_1', 0.8)
    e2_val = row.get('eff_fit_2', 0.8)
    e3_val = row.get('eff_fit_3', 0.8)
    # e4_val = row.get('eff_fit_4', 0.8)
    e4_val = (e1_val + e2_val + e3_val) / 3 # ev_val is the mean of the other three
    
    # Define the 6 unknown subsets
    subset_names = ['12', '23', '34', '123', '234', '1234']

    # Initial guess for x_... (counts). We'll assume ~100 or any positive.
    p0 = [100]*len(subset_names)

    # Bounds for [0, 1e10]
    lb = [0]*len(subset_names)
    ub = [1e10]*len(subset_names)

    # Residual function
    def residuals(p):
        x_vals = dict(zip(subset_names, p))
        res_arr = []
        for (pattern, R_meas) in measured_patterns:
            # Sum up x_subset * detection_prob( ... ) for each possible subset
            model_val = sum(x_vals[sub]*detection_prob(pattern, e1_val, e2_val, e3_val, e4_val, sub)
                            for sub in subset_names)
            res_arr.append(model_val - R_meas)
        return np.array(res_arr, dtype=float)

    # Solve with least_squares
    res = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=50000)

    if not res.success:
        return pd.Series({'status': 'failed', 'residual_sum_sq': np.nan})

    best = res.x
    out = {'status': 'success', 'residual_sum_sq': np.sum(res.fun**2)}
    for i, label in enumerate(subset_names):
        out[f'x_{label}'] = best[i]
    return pd.Series(out)


##############################
# 3) Run row-by-row & store
##############################

# Drop all the NaN in type_ and in  eff_ columns
df_unfolded = data_df.dropna(subset=['type_12', 'type_23', 'type_34', 'type_123', 'type_234', 'type_1234',
                                     'eff_fit_1', 'eff_fit_2', 'eff_fit_3', 'eff_fit_4']).apply(solve_for_subsets, axis=1)
# df_unfolded = data_df.apply(solve_for_subsets, axis=1)

# Sum of all x_... for each row
df_unfolded["total_count"] = (
    df_unfolded["x_12"] + df_unfolded["x_23"] + df_unfolded["x_34"]
    + df_unfolded["x_123"] + df_unfolded["x_234"] + df_unfolded["x_1234"]
)

# Convert from counts to rates (assuming 'data_df["number_of_rows"] * 60' is your measuring time in seconds)
for subset in ["12", "23", "34", "123", "234", "1234"]:
    data_df[f'new_sys_{subset}'] = df_unfolded[f'x_{subset}'] / (data_df["number_of_rows"] * 60)

# Overall corrected rate
data_df["eff_corr"] = df_unfolded["total_count"] / (data_df["number_of_rows"] * 60)
data_df["unc_eff_corr"] = 1

# Example of how you might derive a new efficiency measure
# (assuming data_df['rate'] is some measured rate)
data_df["eff_new"] = data_df['rate'] / data_df["eff_corr"]

# Copy plane efficiencies to new columns if desired
data_df["new_sys_e1"] = data_df["final_eff_1"]
data_df["new_sys_e2"] = data_df["final_eff_2"]
data_df["new_sys_e3"] = data_df["final_eff_3"]
data_df["new_sys_e4"] = data_df["final_eff_4"]

print("Done! 'df_unfolded' has the fitted x_... columns, and 'data_df' now has new_sys_* rates.")


# ------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Correct by the efficiency, calculate uncertainty of the corrected rate
# -----------------------------------------------------------------------------

print('Efficiency correction performed.')


# Assuming data_df is already loaded and contains the necessary columns

if create_plots:
    print("Creating efficiency and rate plots...")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(17, 10), sharex=True)

    # Plot efficiencies
    # Plot new_system_e1, new_system_e2, new_system_e3, new_system_e4
    ax1.plot(data_df['Time'], data_df['eff_fit_1'], label='Eff. fitted 1', color='C4')
    ax1.plot(data_df['Time'], data_df['eff_fit_2'], label='Eff. fitted 2', color='C5')
    ax1.plot(data_df['Time'], data_df['eff_fit_3'], label='Eff. fitted 3', color='C6')
    ax1.plot(data_df['Time'], data_df['eff_fit_4'], label='Eff. fitted 4', color='C7')
    ax1.set_ylabel('Efficiency')
    ax1.set_ylim(0.9, 1.0)
    ax1.set_title('Efficiencies over Time')
    ax1.legend(loc='upper left')
    
    
    # Plot the measured: 12, 23, 34, 123, 234, 1234, 13, 24, 14
    ax2.plot(data_df['Time'], data_df['type_12'] / (data_df["number_of_rows"] * 60), label='Measured 12', color='C1')
    ax2.plot(data_df['Time'], data_df['type_23'] / (data_df["number_of_rows"] * 60), label='Measured 23', color='C2')
    ax2.plot(data_df['Time'], data_df['type_34'] / (data_df["number_of_rows"] * 60), label='Measured 34', color='C3')
    ax2.plot(data_df['Time'], data_df['type_123'] / (data_df["number_of_rows"] * 60), label='Measured 123', color='C4')
    ax2.plot(data_df['Time'], data_df['type_234'] / (data_df["number_of_rows"] * 60), label='Measured 234', color='C5')
    ax2.plot(data_df['Time'], data_df['type_1234'] / (data_df["number_of_rows"] * 60), label='Measured 1234', color='C6')
    
    ax2.set_ylim(1, 5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Rate')
    ax2.set_title('Original Rates over Time')
    ax2.legend(loc='upper left')
    
    ax3.plot(data_df['Time'], data_df['new_sys_12'], label='Crossing 12 (fitted)', color='C1')
    ax3.plot(data_df['Time'], data_df['new_sys_23'], label='Crossing 23', color='C2')
    ax3.plot(data_df['Time'], data_df['new_sys_34'], label='Crossing 34', color='C3')
    ax3.plot(data_df['Time'], data_df['new_sys_123'], label='Crossing 123', color='C4')
    ax3.plot(data_df['Time'], data_df['new_sys_234'], label='Crossing 234', color='C5')
    ax3.plot(data_df['Time'], data_df['new_sys_1234'], label='Crossing 1234', color='C6')

    ax3.set_ylim(1, 5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Rate')
    ax3.set_title('Corrected by efficiency Rates over Time')
    ax3.legend(loc='upper left')
    
    plt.tight_layout()

    # Save or show the plot
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + "_effs_rates_new_sys.png"
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)

    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


if create_plots:
    filtered_df = data_df.dropna(subset=['eff_new', 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    
    # Extract relevant columns
    x = filtered_df['sensors_ext_Pressure_ext'].values.reshape(-1, 1)
    y = filtered_df['sensors_ext_Temperature_ext'].values.reshape(-1, 1)
    z = filtered_df['eff_new'].values
    
    # Fit a plane using linear regression
    X = np.hstack((x, y))
    model = LinearRegression()
    model.fit(X, z)
    
    # Parameters of the fitted plane
    a, b = model.coef_
    c = model.intercept_
    formula = f"Eff = {a:.4g} * P + {b:.4g} * T + {c:.4g}"
    print(f"Fitted plane: {formula}")
    
    data_df['eff_fit'] = a * data_df['sensors_ext_Pressure_ext'] + b * data_df['sensors_ext_Temperature_ext'] + c
    data_df['unc_eff_fit'] = 1
    
    # Predicted plane
    z_pred = model.predict(X)
    residuals = z - z_pred
    
    # Create meshgrid for visualization
    x_range = np.linspace(x.min(), x.max(), 30)
    y_range = np.linspace(y.min(), y.max(), 30)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    Z_mesh = model.predict(np.c_[X_mesh.ravel(), Y_mesh.ravel()]).reshape(X_mesh.shape)
    
    # Create a figure with four subplots (2x2 layout)
    fig = plt.figure(figsize=(14, 12))
    
    # 3D scatter plot with fitted plane (Isometric view)
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(x, y, z, c=z, cmap='turbo', marker='o', s=10, alpha=0.7)
    ax1.plot_surface(X_mesh, Y_mesh, Z_mesh, color='cyan', alpha=0.5)
    ax1.set_xlabel('Pressure', labelpad=10)
    ax1.set_ylabel('Temperature', labelpad=10)
    ax1.set_zlabel('Efficiency', labelpad=10)
    ax1.view_init(elev=30, azim=-60)

    # 3D scatter plot (View along Pressure axis)
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(x, y, z, c=z, cmap='turbo', marker='o', s=10, alpha=0.7)
    ax2.plot_surface(X_mesh, Y_mesh, Z_mesh, color='cyan', alpha=0.5)
    ax2.set_ylabel('Temperature', labelpad=10)
    ax2.set_zlabel('Efficiency', labelpad=10)
    ax2.set_xticks([])
    ax2.view_init(elev=0, azim=0)

    # 3D scatter plot (View along Temperature axis)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(x, y, z, c=z, cmap='turbo', marker='o', s=10, alpha=0.7)
    ax3.plot_surface(X_mesh, Y_mesh, Z_mesh, color='cyan', alpha=0.5)
    ax3.set_xlabel('Pressure', labelpad=10)
    ax3.set_zlabel('Efficiency', labelpad=10)
    ax3.set_yticks([])
    ax3.view_init(elev=0, azim=-90)

    # 3D scatter plot (Top-down view)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(x, y, z, c=z, cmap='turbo', marker='o', s=10, alpha=0.7)
    ax4.plot_surface(X_mesh, Y_mesh, Z_mesh, color='cyan', alpha=0.5)
    ax4.set_xlabel('Pressure', labelpad=10)
    ax4.set_ylabel('Temperature', labelpad=10)
    ax4.set_zticks([])
    ax4.view_init(elev=90, azim=-90)
    
    plt.suptitle(f'"Eff" New. Efficiency vs Pressure and Temperature with Fitted Plane\n{formula}')
    
    plt.tight_layout()
    if show_plots:
        plt.show()
    elif save_plots:
        plt.savefig(figure_path + "_3d_fit.png", format='png', dpi=300)
    plt.close()
    
    # Create another figure to show residuals
    fig_res = plt.figure(figsize=(14, 12))
    
    # Residuals plots
    ax1_res = fig_res.add_subplot(221, projection='3d')
    sc1_res = ax1_res.scatter(x, y, residuals, c=residuals, cmap='coolwarm', marker='o', s=10, alpha=0.7)
    ax1_res.set_xlabel('Pressure', labelpad=10)
    ax1_res.set_ylabel('Temperature', labelpad=10)
    ax1_res.set_zlabel('Residuals', labelpad=10)
    ax1_res.view_init(elev=30, azim=-60)
    # fig_res.colorbar(sc1_res, ax=ax1_res, label='Residuals')
    
    ax2_res = fig_res.add_subplot(222, projection='3d')
    ax2_res.scatter(x, y, residuals, c=residuals, cmap='coolwarm', marker='o', s=10, alpha=0.7)
    ax2_res.set_ylabel('Temperature', labelpad=10)
    ax2_res.set_zlabel('Residuals', labelpad=10)
    ax2_res.set_xticks([])
    ax2_res.view_init(elev=0, azim=0)
    
    ax3_res = fig_res.add_subplot(223, projection='3d')
    ax3_res.scatter(x, y, residuals, c=residuals, cmap='coolwarm', marker='o', s=10, alpha=0.7)
    ax3_res.set_xlabel('Pressure', labelpad=10)
    ax3_res.set_zlabel('Residuals', labelpad=10)
    ax3_res.set_yticks([])
    ax3_res.view_init(elev=0, azim=-90)
    
    ax4_res = fig_res.add_subplot(224, projection='3d')
    ax4_res.scatter(x, y, residuals, c=residuals, cmap='coolwarm', marker='o', s=10, alpha=0.7)
    ax4_res.set_xlabel('Pressure', labelpad=10)
    ax4_res.set_ylabel('Temperature', labelpad=10)
    ax4_res.set_zticks([])
    ax4_res.view_init(elev=90, azim=-90)
    
    plt.tight_layout()
    if show_plots:
        plt.show()
    elif save_plots:
        plt.savefig(figure_path + "_residuals.png", format='png', dpi=300)
    plt.close()



#%%

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Atmospheric corrections -----------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Pressure correction ---------------------------------------------------------
# -----------------------------------------------------------------------------

print('Pressure correction started...')

from scipy.optimize import curve_fit

# Define the exponential model
def fit_model(x, beta, a):
    # [beta] = %/mbar
    return beta / 100 * x + a

def calculate_eta_P(I_over_I0, unc_I_over_I0, delta_P, unc_delta_P):
    
    log_I_over_I0 = np.log(I_over_I0)
    unc_log_I_over_I0 = unc_I_over_I0 / I_over_I0  # Propagate relative errors
    
    # Prepare the data for fitting
    df = pd.DataFrame({
        'log_I_over_I0': log_I_over_I0,
        'unc_log_I_over_I0': unc_log_I_over_I0,
        'delta_P': delta_P,
        'unc_delta_P': unc_delta_P
    }).dropna()
    
    print(len(df))
    
    if not df.empty:
        # Fit the exponential model using uncertainties in Y as weights
        print("Fitting exponential model...")
        
        # Filter outliers before fitting
        z_scores = np.abs((df['log_I_over_I0'] - df['log_I_over_I0'].median()) / df['log_I_over_I0'].std())
        
        # Make a small histogram of the z_scores to see the distribution
        plt.hist(z_scores, bins=400)
        plt.title('Z-Scores Distribution')
        plt.xlabel('Z-Score')
        plt.ylabel('Frequency')
        if show_plots: 
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + "_z"
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format = 'png', dpi = 300)
        plt.close()
        
        df = df[z_scores < z_score_th_pres_corr]
        
        # WIP TO USE UNCERTAINTY OF PRESSURE ----------------------------------------------
        popt, pcov = curve_fit(
            fit_model,
            df['delta_P'],
            df['log_I_over_I0'],
            sigma=df['unc_log_I_over_I0'],  # Only the Y uncertainty
            absolute_sigma=True,
            p0=(1,0)
        )
        b, a = popt  # Extract parameters
        
        # Define eta_P as the parameter b (rate of change in the exponent)
        eta_P = b
        eta_P_uncertainty = np.sqrt(np.diag(pcov))[0]
        
        # Plot the fitting
        if create_plots:
            plt.figure()
            if show_errorbar:
                plt.errorbar(
                    df['delta_P'],
                    df['log_I_over_I0'],
                    xerr=abs(df['unc_delta_P']),
                    yerr=abs(df['unc_log_I_over_I0']),
                    fmt='o',
                    label='Data with Uncertainty'
                )
            else:
                plt.scatter(df['delta_P'], df['log_I_over_I0'], label='Data', s=1, alpha=0.5, marker='.')
            
            plt.plot(df['delta_P'], fit_model(df['delta_P'], *popt), color='red', label='Fit')

            # Extract b (beta) and its uncertainty
            b = popt[0]  # Parameter b from the fit
            unc_b = np.sqrt(np.diag(pcov))[0]  # Uncertainty of parameter b
            
            print("a of the pressure fit:", popt[1])
            
            # Add labels and title
            plt.xlabel('Delta P')
            plt.ylabel('log (I / I0)')
            plt.title(f'Exponential Fit with Uncertainty\nBeta (b) = {b:.3f} ± {unc_b:.3f} %/mbar')
            plt.legend()
            if show_plots: 
                plt.show()
            elif save_plots:
                print(f"Saving figure to {figure_path}")
                plt.savefig(figure_path, format = 'png', dpi = 300)
            plt.close()
    else:
        print("Fit not done, data empty. Returning NaN.")
        eta_P = np.nan
        eta_P_uncertainty = np.nan  # Handle case where there are no valid data points
    return eta_P, eta_P_uncertainty

region = 'eff_corr'

data_df['pressure_lab'] = data_df['sensors_ext_Pressure_ext']
# Calculate pressure differences and their uncertainties
P = data_df['pressure_lab']
unc_P = 1  # Assume a small uncertainty for P if not recalculating

if recalculate_pressure_coeff:
    P0 = data_df['pressure_lab'].mean()
    unc_P0 = unc_P / np.sqrt( len(P) )  # Uncertainty of the mean
else:
    P0 = mean_pressure_used_for_the_fit
    # unc_P0 = np.full_like(P, 1)  # Assume an arbitrary uncertainty if not recalculating
    unc_P0 = 1

delta_P = P - P0
unc_delta_P = np.sqrt(unc_P**2 + unc_P0**2)  # Combined uncertainty (propagation of errors)

def quantile_mean(data_df, region, lower_quantile=0.1, upper_quantile=0.9):
    """
    Compute the mean of the data within the specified quantile range.
    
    :param data_df: DataFrame containing the data.
    :param region: Column name to compute the small quantile mean for.
    :param lower_quantile: Lower quantile threshold (default: 10%).
    :param upper_quantile: Upper quantile threshold (default: 90%).
    :return: Mean of the values within the quantile range.
    """
    values = data_df[region].dropna()  # Remove NaNs
    q_low, q_high = np.quantile(values, [lower_quantile, upper_quantile])
    filtered_values = values[(values >= q_low) & (values <= q_high)]
    
    return filtered_values.mean()

I = data_df[region]
unc_I = data_df[f'unc_{region}']
# I0 = data_df[region].mean()
I0 = quantile_mean(data_df, region)
unc_I0 = unc_I / np.sqrt( len(I) )  # Uncertainty of the mean
I_over_I0 = I / I0
unc_I_over_I0 = I_over_I0 * np.sqrt( (unc_I / I)**2 + (unc_I0 / I0)**2 )
pressure_results = pd.DataFrame(columns=['Region', 'Eta_P'])

# Filter the negative or 0 I_over_I0 values
valid_mask = I_over_I0 > 0
I_over_I0 = I_over_I0[valid_mask]
unc_I_over_I0 = unc_I_over_I0[valid_mask]
delta_P = delta_P[valid_mask]

if recalculate_pressure_coeff:
    eta_P, unc_eta_P = calculate_eta_P(I_over_I0, unc_I_over_I0, delta_P, unc_delta_P)
    print(eta_P)
    pressure_results = pd.concat([pressure_results, pd.DataFrame({'Region': [region], 'Eta_P': [eta_P]})], ignore_index=True)
    
if (recalculate_pressure_coeff == False) or (eta_P == np.nan):
    if recalculate_pressure_coeff == False:
        print("Not recalculating because of the options.")
    
    if eta_P == np.nan:
        print("Not recalculating because the fit failed.")
    
    log_I_over_I0 = np.log(I_over_I0)
    unc_log_I_over_I0 = unc_I_over_I0 / I_over_I0
    
    if create_plots:
        log_I_over_I0 = np.log(I_over_I0)
        
        df = pd.DataFrame({
            'delta_P': delta_P,
            'log_I_over_I0': log_I_over_I0,
            'unc_delta_P': unc_delta_P,
            'unc_log_I_over_I0': unc_I_over_I0 / I_over_I0
        })
        
        plt.figure()
        if show_errorbar:
            plt.errorbar(
                df['delta_P'],
                df['log_I_over_I0'],
                xerr=abs(df['unc_delta_P']),
                yerr=abs(df['unc_log_I_over_I0']),
                fmt='o',
                label='Data with Uncertainty'
            )
        else:
            plt.scatter(df['delta_P'], df['log_I_over_I0'], label='Data', s=1, alpha=0.5, marker='.')
        
        # Plot the line using provided eta_P instead of fitted values
        plt.plot(df['delta_P'], fit_model(df['delta_P'], eta_P, set_a), color='blue', label=f'Set Eta: {eta_P:.3f} ± {unc_eta_P:.3f} %/mbar')
        
        # Add labels and title
        plt.xlabel('Delta P')
        plt.ylabel('log (I / I0)')
        plt.title(f'Plot using Set Eta_P\nEta_P = {eta_P:.3f} ± {unc_eta_P:.3f} %/mbar')
        plt.legend()
        
        if show_plots: 
            plt.show()
        elif save_plots:
            print(f"Saving figure to {figure_path}")
            plt.savefig(figure_path, format='png', dpi=300)
        
        plt.close()


# Create corrected rate column for the region
data_df[f'pres_{region}'] = I * np.exp(-1 * eta_P / 100 * delta_P)

# Final uncertainty calculation in the corrected rate
unc_rate = 1
unc_beta = unc_eta_P
unc_DP = unc_delta_P
term_1_rate = np.exp(-1 * eta_P / 100 * delta_P) * unc_rate
term_2_beta = I * delta_P / 100 * np.exp(-1 * eta_P / 100 * delta_P) * unc_beta
term_3_DP = I * eta_P / 100 * np.exp(-1 * eta_P / 100 * delta_P) * unc_DP
final_unc_combined = np.sqrt(term_1_rate**2 + term_2_beta**2 + term_3_DP**2)
data_df[f'unc_pres_{region}'] = final_unc_combined



# Add a new outlier filter to the pressure correction

if remove_outliers:
    print('Removing outliers and zero values...')
    def remove_outliers_and_zeroes(series, z_thresh=outlier_filter):
        """
        Create a mask of rows that are outliers or have zero values.
        """
        # median = series.mean()
        median = series.median()
        std = series.std()
        z_scores = abs((series - median) / std)
        
        plt.hist(z_scores, bins=300)
        plt.title('Z-Scores Distribution')
        plt.xlabel('Z-Score')
        plt.ylabel('Frequency')
        if show_plots: 
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + "_pressure_corr_z"
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format = 'png', dpi = 300)
        plt.close()
        
        # print(z_scores)
        # Create a mask for rows where z_scores > z_thresh or values are zero
        mask = (z_scores > z_thresh) | (series == 0)
        return mask

    # Initialize a mask of all False, meaning no rows are removed initially
    rows_to_remove = pd.Series(False, index=data_df.index)
    rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df[f'pres_{region}'], z_thresh = 0.5)
    # for region in angular_regions:
    #     rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df[region])

    data_df_cleaned = data_df[~rows_to_remove].copy()
    data_df = data_df_cleaned.copy()



#%%

# -----------------------------------------------------------------------------
# High order correction -------------------------------------------------------
# -----------------------------------------------------------------------------

# WIP!!!!!!

if high_order_correction:
    
    data_df[f'{region}_pressure_corrected'] = data_df[f'pres_{region}']
    
    # Use the pressure-corrected values directly
    # Calculate means for pressure and counts
    I0_count_corrected = data_df[f'{region}_pressure_corrected'].mean()

    # Calculate delta values using pressure-corrected values
    data_df['delta_I_count_corrected'] = data_df[f'{region}_pressure_corrected'] - I0_count_corrected

    # Calculate means for the required columns
    Tg0 = data_df['temp_ground'].mean()
    Th0 = data_df['temp_100mbar'].mean()
    H0 = data_df['height_100mbar'].mean()

    # Calculate delta values
    data_df['delta_Tg'] = data_df['temp_ground'] - Tg0
    data_df['delta_Th'] = data_df['temp_100mbar'] - Th0
    data_df['delta_H'] = data_df['height_100mbar'] - H0

    # Normalize delta values
    data_df['delta_Tg_over_Tg0'] = data_df['delta_Tg'] / Tg0
    data_df['delta_Th_over_Th0'] = data_df['delta_Th'] / Th0
    data_df['delta_H_over_H0'] = data_df['delta_H'] / H0

    # Initialize a DataFrame to store the results
    high_order_results = pd.DataFrame(columns=['Region', 'A', 'B', 'C'])

    # Function to fit the data and calculate coefficients A, B, C
    def calculate_coefficients(region, I0, delta_I):
        delta_I_over_I0 = delta_I / I0

        # Fit linear regression model without intercept
        model = LinearRegression(fit_intercept=False)
        df = pd.DataFrame({
            'delta_I_over_I0': delta_I_over_I0,
            'delta_Tg_over_Tg0': data_df['delta_Tg_over_Tg0'],
            'delta_Th_over_Th0': data_df['delta_Th_over_Th0'],
            'delta_H_over_H0': data_df['delta_H_over_H0']
        }).dropna()

        if not df.empty:
            X = df[['delta_Tg_over_Tg0', 'delta_Th_over_Th0', 'delta_H_over_H0']]
            y = df['delta_I_over_I0']
            model.fit(X, y)
            A, B, C = model.coef_

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

            # 1) Plot Delta Tg
            ax = axes[0]
            x = df['delta_Tg_over_Tg0']
            y_data = df['delta_I_over_I0']  # your actual y-values

            ax.scatter(x, y_data, alpha=0.7, label='Data', s=1)  # smaller points

            # Build a line from min(x) to max(x), ignoring the other variables
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = A * x_line  # slope = A, intercept = 0 (since fit_intercept=False)
            ax.plot(x_line, y_line, color='red', label=f'Partial Fit, {A*100:.1f}% = {A*100*Tg0:.1f}% / K')

            ax.set_xlabel('$\Delta T^{\mathrm{ground}} / T^{\mathrm{ground}}_{0}$')
            ax.set_ylabel('Delta I / I0')
            ax.set_title('Effect of Temp. ground')
            ax.legend()

            # 2) Plot Delta Th
            ax = axes[1]
            x = df['delta_Th_over_Th0']
            ax.scatter(x, y_data, alpha=0.7, label='Data', s=1, color = "green")

            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = B * x_line  # slope = B
            ax.plot(x_line, y_line, color='red', label=f'Partial Fit, {B*100:.1f}% = {B*100*Th0:.1f}% / K')

            ax.set_xlabel('$\Delta T^{100\ \mathrm{mbar}} / T^{100\ \mathrm{mbar}}_{0}$')
            ax.set_title('Effect of Temp. 100 mbar layer')
            ax.legend()

            # 3) Plot Delta H
            ax = axes[2]
            x = df['delta_H_over_H0']
            ax.scatter(x, y_data, alpha=0.7, label='Data', s=1, color = "purple")

            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = C * x_line  # slope = C
            ax.plot(x_line, y_line, color='red', label=f'Partial Fit, {C*100:.1f}% = {C*100*H0:.1f}% / m')

            ax.set_xlabel('$\Delta h^{100\ \mathrm{mbar}} / h^{100\ \mathrm{mbar}}_{0}$')
            ax.set_title('Effect of Height of 100 mbar layer')
            ax.legend()

            plt.tight_layout()

            plt.legend()
            if show_plots:
                plt.show()
            elif save_plots:
                new_figure_path = figure_path + "_high_order.png"
                print(f"Saving figure to {new_figure_path}")
                plt.savefig(new_figure_path, format='png', dpi=300)

            plt.close()
        else:
            A, B, C = np.nan, np.nan, np.nan  # Handle case where there are no valid data points
        return A, B, C

    # Calculate coefficients and create corrected rate columns for each region
    regions = [region]
    # regions = ['count'] + angular_regions
    for region in regions:
        I0_region_corrected = data_df[f'{region}_pressure_corrected'].mean()
        data_df[f'delta_I_{region}_corrected'] = data_df[f'{region}_pressure_corrected'] - I0_region_corrected
        A, B, C = calculate_coefficients(region, I0_region_corrected, data_df[f'delta_I_{region}_corrected'])
        high_order_results = pd.concat([high_order_results, pd.DataFrame({'Region': [region], 'A': [A], 'B': [B], 'C': [C]})], ignore_index=True)
        
        # Create corrected rate column for the region
        data_df[f'{region}_high_order_corrected'] = data_df[f'{region}_pressure_corrected'] * (1 - (A * data_df['delta_Tg'] / Tg0 + B * data_df['delta_Th'] / Th0 + C * data_df['delta_H'] / H0))

else:
    print("High order correction not applied.")
    data_df[f'{region}_high_order_corrected'] = data_df[f'pres_{region}']




if create_plots:
    print("Creating efficiency and rate plots...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 10), sharex=True)

    # Plot efficiencies
    ax1.plot(data_df['Time'], data_df['eff_fit'], label='Efficiency from the new fit', color='C5')
    ax1.plot(data_df['Time'], data_df['eff_new'], label='Original / Corrected with new system', color='C4')
    ax1.set_ylabel('Efficiency')
    ax1.set_ylim(0.9, 1)
    ax1.set_title('Efficiencies over Time')
    ax1.legend(loc='upper left')

    ax2.plot(data_df['Time'], data_df['rate'], label='OG rate', color='C3')
    ax2.plot(data_df['Time'], data_df['eff_corr'], label='Eff. (from new system) corr. rate', color='C8')
    ax2.plot(data_df['Time'], data_df[f'pres_{region}'], label='Pressure corrected rate', color='C9')
    ax2.plot(data_df['Time'], data_df[f'{region}_high_order_corrected'], label='High order corrected rate', color='C10')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Rate')
    # ax2.set_ylim(12, 16)
    # ax2.set_ylim(16, 18)
    ax2.set_ylim(10, 25)
    ax2.set_title('Rates over Time')
    ax2.legend(loc='upper left')

    plt.tight_layout()

    # Save or show the plot
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + "_effs_rates_naasd_pressure.png"
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)

    plt.close()


# -----------------------------------------------------------------------------
# Smoothing filters -----------------------------------------------------------
# -----------------------------------------------------------------------------

# # Horizontal Median Filter ----------------------------------------------------
# ker = HMF_ker # 61

# # Apply median filter to columns of interest
# if ker > 0:
#     data_df[f'pres_{region}'] = medfilt(data_df[f'pres_{region}'], kernel_size=ker)
# else:
#     print('Horizontal Median Filter not applied.')


# # Moving Average Filter -------------------------------------------------------
# window_size = MAF_ker # 5   # This includes the current point, so it averages n before and n after

# # Apply moving average filter to columns of interest
# if window_size > 0:
#     for region in angular_regions:
#         data_df[region] = data_df[region].rolling(window=window_size, center=True).mean()
    
# # Remove the points in the time limit
# skip = skip_in_limits # 15 was ok
# data_df = data_df.iloc[skip:-skip]


# -----------------------------------------------------------------------------
# One more systematic error should be added to the corrected rate:
# from simulation, the uncertainty due to the corrected rate to real rate
# value, which accounts for the size of the detector, mostly.
# -----------------------------------------------------------------------------

data_df[f'unc_sys_pres_{region}'] = np.sqrt( data_df[f'unc_pres_{region}']**2 + systematic_unc_corr_to_real_rate**2 )

# -----------------------------------------------------------------------------
# The end. Defining the total finally corrected rate
# -----------------------------------------------------------------------------
data_df[f'totally_corrected_rate'] = data_df[f'pres_{region}']
data_df[f'unc_totally_corrected_rate'] = data_df[f'unc_sys_pres_{region}']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Saving ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# If ANY value is 0, put it to NaN
data_df = data_df.replace(0, np.nan)

# Put to NaN the values in the totally_corrected_rate that are outside of [16, 20]
print("------------------------------------------------------------------------------------------")
print("WATCH OUT THIS LAST FILTER BECAUSE IT MAY GIVE PROBLEMS IN THE FUTURE!!!!!!!!!!!!!!!!!!!!!")
print("------------------------------------------------------------------------------------------")
data_df.loc[(data_df['totally_corrected_rate'] < 5) | (data_df['totally_corrected_rate'] > 30), 'totally_corrected_rate'] = np.nan
# data_df.loc[(data_df['totally_corrected_rate'] < 4.85) | (data_df['totally_corrected_rate'] > 5.3), 'totally_corrected_rate'] = np.nan

data_df.to_csv(save_filename, index=False)
print('Efficiency and atmospheric corrections completed and saved to corrected_table.csv.')


# -----------------------------------------------------------------------------
# Saving short table ----------------------------------------------------------
# -----------------------------------------------------------------------------

# Create a new DataFrame for Grafana
grafana_df = data_df[['Time', 'pressure_lab', 'totally_corrected_rate', 'unc_totally_corrected_rate', 'eff_global', 'unc_eff_global']].copy()

# Rename the columns
grafana_df.columns = ['Time', 'P', 'rate', 'u_rate', 'eff', 'u_eff']

grafana_df["norm_rate"] = grafana_df["rate"] / grafana_df["rate"].mean() - 1
grafana_df["u_norm_rate"] = grafana_df["u_rate"] / grafana_df["rate"].mean()

# Drop amy row that has Nans
grafana_df = grafana_df.dropna()

# Save the DataFrame to a CSV file
grafana_df.to_csv(grafana_save_filename, index=False)
print(f'Data for Grafana saved to {grafana_save_filename}.')

print('------------------------------------------------------')
print(f"corrector.py completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print('------------------------------------------------------')
