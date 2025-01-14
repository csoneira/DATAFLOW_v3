#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jun 24 19:02:22 2024

@author: gfn
"""

# globals().clear()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.signal import medfilt
import os
# from sklearn.linear_model import LinearRegression
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


# ALSO THE SIMULATED VALUES FOR ACCEPTANCE, EFFICIENCY FACTORS AND UNCERTAINTIES

# This should come from an input file
pressure_coeff_input = 0
unc_pressure_coeff_input = 0
mean_pressure_used_for_the_fit = 940

systematic_unc = [0, 0, 0, 0] # From simulation
acceptance_factor = [0.7, 1, 1, 0.8] # From simulation

systematic_unc_corr_to_real_rate = 0

# -----------------------------------------------------------------------------


# Define the base folder and file paths
base_folder = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/SECOND_STAGE")
filepath = f"{base_folder}/total_data_table.csv"
save_filename = f"{base_folder}/large_corrected_table.csv"
grafana_save_filename = f"{base_folder}/data_for_grafana_{station}.csv"

figure_path = f"{base_folder}/pressure_correction_fit.png"

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

# Plotting configuration
show_plots = False
save_plots = True
create_plots = True

recalculate_pressure_coeff = True

res_win_min = 3 # 180 Resampling window minutes
HMF_ker = 1 # It must be odd. Horizontal Median Filter
MAF_ker = 1 # Moving Average Filter

outlier_filter = 0.1

high_order_correction = False
date_selection = False  # Set to True if you want to filter by date

skip_in_limits = 1

# -----------------------------------------------------------------------------
# To not touch unless necesary ------------------------------------------------
# -----------------------------------------------------------------------------

resampling_window = f'{res_win_min}min'  # '10T' # '5T' stands for 5 minutes.

# Columns to sum
angular_regions = ['High', 'N', 'S', 'E', 'W']
detection_types = ['type_1234', 'type_123', 'type_234', 'type_124', 'type_134', 'type_12', 'type_23', 'type_34', 'type_13', 'type_14', 'type_24']
charge_types = ['count_in_1_sum', 'avalanche_1_sum', 'streamer_1_sum',
                'count_in_2_sum', 'avalanche_2_sum', 'streamer_2_sum',
                'count_in_3_sum', 'avalanche_3_sum', 'streamer_3_sum',
                'count_in_4_sum', 'avalanche_4_sum', 'streamer_4_sum']


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Function definition ---------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# ...

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
print('Reading the CSV file...')
data_df = pd.read_csv(filepath)
print('File loaded successfully.')


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

# Convert 'Time' column to datetime
# data_df['Time'] = pd.to_datetime(data_df['Time'].str.strip('"'), format='%Y-%m-%d %H:%M:%S')

# Filter data based on dates if start_date is set
if date_selection and start_date is not None:
    data_df = data_df[(data_df['Time'] >= start_date) & (data_df['Time'] <= end_date)]

print(f"Filtered data contains {len(data_df)} rows.")


# -----------------------------------------------------------------------------
# Outlier removal ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

# def remove_outliers_and_zeroes(series, z_thresh=outlier_filter):
#     """
#     Create a mask of rows that are outliers or have zero values.
#     """
#     # median = series.median()
#     median = series.mean()
#     z_scores = abs((series - median) / median)
#     # print(z_scores)
#     # Create a mask for rows where z_scores > z_thresh or values are zero
#     mask = (z_scores > z_thresh) | (series == 0)
#     return mask

# # Initialize a mask of all False, meaning no rows are removed initially
# rows_to_remove = pd.Series(False, index=data_df.index)
# rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df['x_custom_mean'])
# for region in angular_regions:
#     rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df[region])

# data_df_cleaned = data_df[~rows_to_remove].copy()

# # Now, `data_df_cleaned` contains only rows that pass the conditions.
# print(f"Original DataFrame shape: {data_df.shape}")
# print(f"Cleaned DataFrame shape: {data_df_cleaned.shape}")

# data_df = data_df_cleaned.copy()


# -----------------------------------------------------------------------------
# Resampling the data in a larger time window: some averaged, some summed -----
# -----------------------------------------------------------------------------

data_df.set_index('Time', inplace=True)

columns_to_sum = angular_regions + detection_types + charge_types
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
data_df['count'] = data_df['count_regions'] # Until I have a better definition of the count
data_df['count_uncertainty'] = np.sqrt(data_df['count'])
# ------------------------------------------------------------------------------------------

data_df['rate'] = data_df['count'] / ( res_win_min * 60 )  # Counts per second (Hz)
data_df['rate_uncertainty'] = np.sqrt(data_df['count']) / ( res_win_min * 60 )


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
        return 0

# Create explicit columns for detected and passed
data_df['detected_1'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134']].sum(axis=1, skipna=True)
data_df['detected_2'] = data_df[['type_1234', 'type_123', 'type_234', 'type_124']].sum(axis=1, skipna=True)
data_df['detected_3'] = data_df[['type_1234', 'type_123', 'type_234', 'type_134']].sum(axis=1, skipna=True)
data_df['detected_4'] = data_df[['type_1234', 'type_234', 'type_124', 'type_134']].sum(axis=1, skipna=True)

data_df['passed_1'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134', 'type_234']].sum(axis=1, skipna=True)
data_df['passed_2'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134', 'type_234']].sum(axis=1, skipna=True)
data_df['passed_3'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134', 'type_234']].sum(axis=1, skipna=True)
data_df['passed_4'] = data_df[['type_1234', 'type_123', 'type_124', 'type_134', 'type_234']].sum(axis=1, skipna=True)

print('Detected and passed calculated.')

# Set columns to 0 if detected or passed values are NaN or 0
def handle_zero_or_nan(row, detected_col, passed_col):
    if row[detected_col] == 0 or row[passed_col] == 0:
        return 0
    return row[detected_col] / row[passed_col]

def handle_uncertainty(row, detected_col, passed_col):
    if row[detected_col] == 0 or row[passed_col] == 0:
        return 0
    return calculate_efficiency_uncertainty(row[detected_col], row[passed_col])


# Calculate efficiencies and uncertainties explicitly
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

data_df['final_eff_1'] = data_df['eff_1'] / acceptance_factor[0]
data_df['final_eff_2'] = data_df['eff_2'] / acceptance_factor[1]
data_df['final_eff_3'] = data_df['eff_3'] / acceptance_factor[2]
data_df['final_eff_4'] = data_df['eff_4'] / acceptance_factor[3]

# Calculate the average efficiency
data_df['eff_global'] = data_df[['eff_1', 'eff_2', 'eff_3', 'eff_4']].mean(axis=1)

# Calculate the uncertainty for the average efficiency
data_df['unc_eff_global'] = np.sqrt(
    (data_df['unc_eff_1'] ** 2 +
     data_df['unc_eff_2'] ** 2 +
     data_df['unc_eff_3'] ** 2 +
     data_df['unc_eff_4'] ** 2) / 4
)

# -----------------------------------------------------------------------------
# Correct by the efficiency, calculate uncertainty of the corrected rate
# -----------------------------------------------------------------------------

# Correct the rate
data_df['eff_corr_rate'] = data_df['rate'] * (1 / data_df['eff_global'])

# Calculate the uncertainty in the corrected rate
data_df['unc_eff_corr_rate'] = data_df['eff_corr_rate'] * np.sqrt(
    (data_df['rate_uncertainty'] / data_df['rate'])**2 +
    (data_df['unc_eff_global'] / data_df['eff_global'])**2
)

print('Efficiency correction performed.')

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

show_plots = True

# Define the exponential model
def fit_model(x, a, b, c):
    return b / 100 * x

def calculate_eta_P(I_over_I0, unc_I_over_I0, delta_P, unc_delta_P, show_plots=True):
    
    log_I_over_I0 = np.log(I_over_I0)
    unc_log_I_over_I0 = unc_I_over_I0 / I_over_I0  # Propagate relative errors
    
    # Prepare the data for fitting
    df = pd.DataFrame({
        'log_I_over_I0': log_I_over_I0,
        'unc_log_I_over_I0': unc_log_I_over_I0,
        'delta_P': delta_P,
        'unc_delta_P': unc_delta_P
    }).dropna()
    
    if not df.empty:
        # Fit the exponential model using uncertainties in Y as weights
        # WIP TO USE UNCERTAINTY OF PRESSURE ----------------------------------------------
        popt, pcov = curve_fit(
            fit_model,
            df['delta_P'],
            df['log_I_over_I0'],
            sigma=df['unc_log_I_over_I0'],  # Only the Y uncertainty
            absolute_sigma=True,
            p0=(1, 0.1, 0)
        )
        a, b, c = popt  # Extract parameters
        
        # Define eta_P as the parameter b (rate of change in the exponent)
        eta_P = b
        eta_P_uncertainty = np.sqrt(np.diag(pcov))[1]
        
        # Plot the fitting
        if create_plots:
            plt.figure()
            plt.errorbar(
                df['delta_P'],
                df['log_I_over_I0'],
                xerr=abs(df['unc_delta_P']),
                yerr=abs(df['unc_log_I_over_I0']),
                fmt='o',
                label='Data with Uncertainty'
            )
            plt.plot(df['delta_P'], fit_model(df['delta_P'], *popt), color='red', label='Fit')

            # Extract b (beta) and its uncertainty
            b = popt[1]  # Parameter b from the fit
            unc_b = np.sqrt(np.diag(pcov))[1]  # Uncertainty of parameter b

            # Add labels and title
            plt.xlabel('Delta P')
            plt.ylabel('log (I / I0)')
            plt.title(f'Exponential Fit with Uncertainty\nBeta (b) = {b:.3f} ± {unc_b:.3f} %/mbar')
            plt.legend()
            if show_plots: 
                plt.show()
            if save_plots:
                plt.savefig(figure_path)
            plt.close()
    else:
        eta_P = np.nan
        eta_P_uncertainty = np.nan  # Handle case where there are no valid data points
    return eta_P, eta_P_uncertainty

region = 'eff_corr_rate'
data_df['pressure_lab'] = data_df['sensors_ext_Pressure_ext']

# Calculate pressure differences and their uncertainties
P = data_df['pressure_lab']
unc_P = 0.1  # Assume a small uncertainty for P if not recalculating

if recalculate_pressure_coeff:
    P0 = data_df['pressure_lab'].mean()
    unc_P0 = unc_P / np.sqrt( len(P) )  # Uncertainty of the mean
else:
    P0 = mean_pressure_used_for_the_fit
    unc_P0 = 1  # Assume an arbitrary uncertainty if not recalculating

delta_P = P - P0
unc_delta_P = np.sqrt(unc_P**2 + unc_P0**2)  # Combined uncertainty (propagation of errors)

if recalculate_pressure_coeff:
    I = data_df[region]
    unc_I = data_df[f'unc_{region}']
    
    I0 = data_df[region].mean()
    unc_I0 = unc_I / np.sqrt( len(I) )  # Uncertainty of the mean
    
    I_over_I0 = I / I0
    unc_I_over_I0 = I_over_I0 * np.sqrt( (unc_I / I)**2 + (unc_I0 / I0)**2 )
    
    pressure_results = pd.DataFrame(columns=['Region', 'Eta_P'])
    eta_P, unc_eta_P = calculate_eta_P(I_over_I0, unc_I_over_I0, delta_P, unc_delta_P)
    pressure_results = pd.concat([pressure_results, pd.DataFrame({'Region': [region], 'Eta_P': [eta_P]})], ignore_index=True)
else:
    eta_P = pressure_coeff_input
    unc_eta_P = unc_pressure_coeff_input

# Create corrected rate column for the region
data_df[f'pres_{region}'] = I * np.exp(-1 * eta_P / 100 * delta_P)

# Calculate uncertainty in the corrected rate
data_df[f'unc_pres_{region}'] = 1 # Placeholder


#%%

# -----------------------------------------------------------------------------
# High order correction -------------------------------------------------------
# -----------------------------------------------------------------------------

# WIP!!!!!!

# if high_order_correction:
#     # Use the pressure-corrected values directly
#     # Calculate means for pressure and counts
#     P0 = data_df['pressure_lab'].mean()
#     I0_count_corrected = data_df['count_pressure_corrected'].mean()

#     # Calculate delta values using pressure-corrected values
#     data_df['delta_I_count_corrected'] = data_df['count_pressure_corrected'] - I0_count_corrected

#     # Calculate means for the required columns
#     Tg0 = data_df['temp_ground'].mean()
#     Th0 = data_df['temp_100mbar'].mean()
#     H0 = data_df['height_100mbar'].mean()

#     # Calculate delta values
#     data_df['delta_Tg'] = data_df['temp_ground'] - Tg0
#     data_df['delta_Th'] = data_df['temp_100mbar'] - Th0
#     data_df['delta_H'] = data_df['height_100mbar'] - H0

#     # Normalize delta values
#     data_df['delta_Tg_over_Tg0'] = data_df['delta_Tg'] / Tg0
#     data_df['delta_Th_over_Th0'] = data_df['delta_Th'] / Th0
#     data_df['delta_H_over_H0'] = data_df['delta_H'] / H0

#     # Initialize a DataFrame to store the results
#     high_order_results = pd.DataFrame(columns=['Region', 'A', 'B', 'C'])

#     # Function to fit the data and calculate coefficients A, B, C
#     def calculate_coefficients(region, I0, delta_I):
#         delta_I_over_I0 = delta_I / I0

#         # Fit linear regression model without intercept
#         model = LinearRegression(fit_intercept=False)
#         df = pd.DataFrame({
#             'delta_I_over_I0': delta_I_over_I0,
#             'delta_Tg_over_Tg0': data_df['delta_Tg_over_Tg0'],
#             'delta_Th_over_Th0': data_df['delta_Th_over_Th0'],
#             'delta_H_over_H0': data_df['delta_H_over_H0']
#         }).dropna()

#         if not df.empty:
#             X = df[['delta_Tg_over_Tg0', 'delta_Th_over_Th0', 'delta_H_over_H0']]
#             y = df['delta_I_over_I0']
#             model.fit(X, y)
#             A, B, C = model.coef_

#             # Plot the fitting in subplots
#             fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#             axes[0].scatter(df['delta_Tg_over_Tg0'], y, label='Data')
#             axes[0].plot(df['delta_Tg_over_Tg0'], model.predict(X), color='red', label='Fit')
#             axes[0].set_xlabel('Delta Tg / Tg0')
#             axes[0].set_ylabel('Delta I / I0')
#             axes[0].set_title(f'Fitting for {region} - Delta Tg')

#             axes[1].scatter(df['delta_Th_over_Th0'], y, label='Data')
#             axes[1].plot(df['delta_Th_over_Th0'], model.predict(X), color='red', label='Fit')
#             axes[1].set_xlabel('Delta Th / Th0')
#             axes[1].set_title(f'Fitting for {region} - Delta Th')

#             axes[2].scatter(df['delta_H_over_H0'], y, label='Data')
#             axes[2].plot(df['delta_H_over_H0'], model.predict(X), color='red', label='Fit')
#             axes[2].set_xlabel('Delta H / H0')
#             axes[2].set_title(f'Fitting for {region} - Delta H')

#             plt.legend()
#             if show_plots: plt.show();
#             plt.close()
#         else:
#             A, B, C = np.nan, np.nan, np.nan  # Handle case where there are no valid data points
#         return A, B, C

#     # Calculate coefficients and create corrected rate columns for each region
#     regions = ['count'] + angular_regions
#     for region in regions:
#         I0_region_corrected = data_df[f'{region}_pressure_corrected'].mean()
#         data_df[f'delta_I_{region}_corrected'] = data_df[f'{region}_pressure_corrected'] - I0_region_corrected
#         A, B, C = calculate_coefficients(region, I0_region_corrected, data_df[f'delta_I_{region}_corrected'])
#         high_order_results = pd.concat([high_order_results, pd.DataFrame({'Region': [region], 'A': [A], 'B': [B], 'C': [C]})], ignore_index=True)
        
#         # Create corrected rate column for the region
#         data_df[f'{region}_corrected'] = data_df[f'{region}_pressure_corrected'] * (1 - (A * data_df['delta_Tg'] / Tg0 + B * data_df['delta_Th'] / Th0 + C * data_df['delta_H'] / H0))

# else:
#     data_df[f'{region}_corrected'] = data_df[f'{region}_pressure_corrected']


# -----------------------------------------------------------------------------
# Smoothing filters -----------------------------------------------------------
# -----------------------------------------------------------------------------

# # Horizontal Median Filter ----------------------------------------------------
# ker = HMF_ker # 61

# # Apply median filter to columns of interest
# if ker > 0:
#     data_df['count'] = medfilt(data_df['count'], kernel_size=ker)
#     data_df['x_mean'] = medfilt(data_df['x_mean'], kernel_size=ker)
#     data_df['y_mean'] = medfilt(data_df['y_mean'], kernel_size=ker)
#     data_df['t0_mean'] = medfilt(data_df['t0_mean'], kernel_size=ker)
#     data_df['s_mean'] = medfilt(data_df['s_mean'], kernel_size=ker)
#     data_df['theta_mean'] = medfilt(data_df['theta_mean'], kernel_size=ker)
#     data_df['phi_mean'] = medfilt(data_df['phi_mean'], kernel_size=ker)

#     # Apply median filter to each region
#     for region in angular_regions:
#         data_df[region] = medfilt(data_df[region], kernel_size=ker)



# # Moving Average Filter -------------------------------------------------------
# window_size = MAF_ker # 5   # This includes the current point, so it averages n before and n after

# # Apply moving average filter to columns of interest
# if window_size > 0:
#     data_df['count'] = data_df['count'].rolling(window=window_size, center=True).mean()
#     data_df['x_mean'] = data_df['x_mean'].rolling(window=window_size, center=True).mean()
#     data_df['y_mean'] = data_df['y_mean'].rolling(window=window_size, center=True).mean()
#     data_df['t0_mean'] = data_df['t0_mean'].rolling(window=window_size, center=True).mean()
#     data_df['s_mean'] = data_df['s_mean'].rolling(window=window_size, center=True).mean()
#     data_df['theta_mean'] = data_df['theta_mean'].rolling(window=window_size, center=True).mean()
#     data_df['phi_mean'] = data_df['phi_mean'].rolling(window=window_size, center=True).mean()

#     # Apply moving average filter to each region
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

data_df[f'unc_sys_pres_{region}'] = np.sqrt( data_df[f'unc_pres_{region}']**2 +
                                                 systematic_unc_corr_to_real_rate**2 ) 


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

data_df.to_csv(save_filename, index=False)
print('Efficiency and atmospheric corrections completed and saved to corrected_table.csv.')


# -----------------------------------------------------------------------------
# Saving short table ----------------------------------------------------------
# -----------------------------------------------------------------------------

# Create a new DataFrame for Grafana
grafana_df = data_df[['Time', 'pressure_lab', 'totally_corrected_rate', 'unc_totally_corrected_rate', 'eff_global', 'unc_eff_global']].copy()

# Rename the columns
grafana_df.columns = ['Time', 'P', 'rate', 'u_rate', 'eff', 'u_eff']

# Save the DataFrame to a CSV file
grafana_df.to_csv(grafana_save_filename, index=False)
print(f'Data for Grafana saved to {grafana_save_filename}.')


# %%
