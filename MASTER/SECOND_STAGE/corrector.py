#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jun 24 19:02:22 2024

@author: csoneira@ucm.es
"""

print("\n\n")
print("                                          _.oo.")
print("                 _.u[[/;:,.         .odMMMMMM'")
print("              .o888UU[[[/;:-.  .o@P^    MMM^")
print("             oN88888UU[[[/;::-.        dP^")
print("            dNMMNN888UU[[[/;:--.   .o@P^")
print("           ,MMMMMMN888UU[[/;::-. o@^")
print("           NNMMMNN888UU[[[/~.o@P^")
print("           888888888UU[[[/o@^-..")
print("          oI8888UU[[[/o@P^:--..")
print("       .@^  YUU[[[/o@^;::---..")
print("     oMP     ^/o@P^;:::---..")
print("  .dMMM    .o@^ ^;::---...")
print(" dMMMMMMM@^`       `^^^^")
print("YMMMUP^")
print(" ^^")
print("\n\n")


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
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import halfnorm
from scipy.stats import norm


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
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# This works fine for total counts:
# resampling_window = '10T'  # '10T' # '5T' stands for 5 minutes. Adjust based on your needs.
# HMF_ker = 5 # It must be odd.
# MAF_ker = 1

remove_outliers = True

# Plotting configuration
show_plots = False
save_plots = True
create_plots = True
show_errorbar = False

recalculate_pressure_coeff = True

res_win_min = 15 # 180 Resampling window minutes
# HMF_ker = 1 # It must be odd. Horizontal Median Filter
# MAF_ker = 1 # Moving Average Filter

outlier_filter = 4 #3

high_order_correction = True
date_selection = False  # Set to True if you want to filter by date

# This should come from an input file
eta_P = -0.162 # pressure_coeff_input
unc_eta_P = 0.013 # unc_pressure_coeff_input
set_a = -0.11357 # pressure_intercept_input
mean_pressure_used_for_the_fit = 940

systematic_unc = [0, 0, 0, 0] # From simulation
# acceptance_factor = [0.7, 1, 1, 0.8] # From simulation

systematic_unc_corr_to_real_rate = 0
z_score_th_pres_corr = 5

# -----------------------------------------------------------------------------

# Define the base folder and file paths
grafana_directory = os.path.expanduser(f"~/DATAFLOW_v3/GRAFANA_DATA")
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
base_folder = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/SECOND_STAGE")
filepath = f"{base_folder}/total_data_table.csv"
save_filename = f"{base_folder}/large_corrected_table.csv"
grafana_save_filename = f"{grafana_directory}/data_for_grafana_{station}.csv"
figure_path = f"{base_folder}/"
fig_idx = 0
os.makedirs(base_folder, exist_ok=True)
os.makedirs(grafana_directory, exist_ok=True)


# -----------------------------------------------------------------------------
# To not touch unless necesary ------------------------------------------------
# -----------------------------------------------------------------------------

resampling_window = f'{res_win_min}min'  # '10min' # '5min' stands for 5 minutes.

# Columns to sum
angular_regions = ['High', 'N', 'S', 'E', 'W']

original_types = [ 'original_tt_123', 'original_tt_1234', 'original_tt_23', 'original_tt_12', 'original_tt_234', 'original_tt_34', 'original_tt_124', 'original_tt_134', 'original_tt_13' ]
detection_types = [ 'processed_tt_123', 'processed_tt_34', 'processed_tt_23', 'processed_tt_12', 'processed_tt_234', 'processed_tt_124', 'processed_tt_1234', 'processed_tt_14', 'processed_tt_134', 'processed_tt_24', 'processed_tt_13' ]
tracking_types = [ 'tracking_tt_123', 'tracking_tt_234', 'tracking_tt_23', 'tracking_tt_12', 'tracking_tt_1234', 'tracking_tt_34' ]

charge_types = ['count_in_1',
                'count_in_2',
                'count_in_3',
                'count_in_4']


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Introduction ----------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Reading ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Load the data
print('Reading the big CSV datafile...')
data_df = pd.read_csv(filepath)

print("\n\n")
print(data_df.columns.to_list())
print("\n\n")

print("Putting zeroes to NaNs...")
data_df = data_df.replace(0, np.nan)

print(filepath)
print('File loaded successfully.')




# Preprocess the data to remove rows with invalid datetime format -------------------------------------------------------------------------------
print('Validating datetime format in "Time" column...')
try:
    # Try parsing 'Time' column with the specified format
    data_df['Time'] = pd.to_datetime(data_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
except Exception as e:
    print(f"Error while parsing datetime: {e}")
    exit(1)

# Drop rows where 'Time' could not be parsed -------------------------------------------------------------------------------
invalid_rows = data_df['Time'].isna().sum()
if invalid_rows > 0:
    print(f"Removing {invalid_rows} rows with invalid datetime format.")
    data_df = data_df.dropna(subset=['Time'])
else:
    print("No rows with invalid datetime format removed.")
print('Datetime validation completed successfully.')




# Check if the results file exists -------------------------------------------------------------------------------
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




# Date filtering -------------------------------------------------------------------------------
if date_selection:
    start_date = pd.to_datetime("2024-05-10 00:00:00")  # Use a string in 'YYYY-MM-DD' format
    end_date = pd.to_datetime("2024-05-11 12:00:00")
    
    # start_date = pd.to_datetime("2025-01-02")  # Use a string in 'YYYY-MM-DD' format
    start_date = pd.to_datetime("2025-04-18 12:36:00")  # Use a string in 'YYYY-MM-DD' format
    end_date = pd.to_datetime("2025-04-19 10:15")
    
    print("------- SELECTION BY DATE IS BEING PERFORMED -------")
    data_df = data_df[(data_df['Time'] >= start_date) & (data_df['Time'] <= end_date)]

print(f"Filtered data contains {len(data_df)} rows.")




remove_non_data_points = True
if remove_non_data_points:
    # Remove rows where 'events' is NaN or zero
    print(f"Original data contains {len(data_df)} rows before removing non-data points.")
    data_df = data_df.dropna(subset=['events'])
    data_df = data_df[data_df['events'] != 0]
    print(f"Filtered data contains {len(data_df)} rows after removing non-data points.")




# Define input file path -------------------------------------------------------------------------------
input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")
if os.path.exists(input_file_config_path):
    input_file = pd.read_csv(input_file_config_path, skiprows=1, decimal = ",")
    print("Input configuration file found.")
    exists_input_file = True
else:
    exists_input_file = False
    print("Input configuration file does not exist.")

if exists_input_file:
    # Parse start/end timestamps in the configuration file
    input_file["start"] = pd.to_datetime(input_file["start"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = pd.to_datetime(input_file["end"], format="%Y-%m-%d", errors="coerce")
    input_file["end"].fillna(pd.to_datetime('now'), inplace=True)

    # Prepare empty Series aligned with data_df index
    acc_cols = {
        "acc_1": pd.Series(index=data_df.index, dtype='float'),
        "acc_2": pd.Series(index=data_df.index, dtype='float'),
        "acc_3": pd.Series(index=data_df.index, dtype='float'),
        "acc_4": pd.Series(index=data_df.index, dtype='float')
    }

    # Assign values to each row in data_df based on timestamp match
    for idx, timestamp in data_df["Time"].items():
        match = input_file[(input_file["start"] <= timestamp) & (input_file["end"] >= timestamp)]
        if not match.empty:
            selected = match.iloc[0]
            acc_cols["acc_1"].at[idx] = selected.get("acc_1", 1)
            acc_cols["acc_2"].at[idx] = selected.get("acc_2", 1)
            acc_cols["acc_3"].at[idx] = selected.get("acc_3", 1)
            acc_cols["acc_4"].at[idx] = selected.get("acc_4", 1)
        else:
            # Default values if no match
            acc_cols["acc_1"].at[idx] = 1
            acc_cols["acc_2"].at[idx] = 1
            acc_cols["acc_3"].at[idx] = 1
            acc_cols["acc_4"].at[idx] = 1

    # Assign the new acc_* columns to data_df
    for col in ["acc_1", "acc_2", "acc_3", "acc_4"]:
        data_df[col] = pd.to_numeric(acc_cols[col], errors='coerce').fillna(1)

else:
    print("No input file found. Default values set.")
    for col in ["acc_1", "acc_2", "acc_3", "acc_4"]:
        data_df[col] = 1


# -----------------------------------------------------------------------------
# Outlier removal -------------------------------------------------------------
# -----------------------------------------------------------------------------


if remove_outliers:
    
    if create_plots:
        # Step 1: Clean the data
        series = data_df['events'].copy()
        series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()

        # Step 2: Z-score filter (±3σ)
        mean = series_clean.mean()
        std = series_clean.std()
        z_scores = (series_clean - mean) / std
        mask = np.abs(z_scores) <= 3
        filtered = series_clean[mask]

        # Remove outliers from original dataframe
        data_df = data_df.loc[filtered.index].copy()

        # Step 3: Fit normal distribution to filtered data
        mu, sigma = norm.fit(filtered)

        # Step 4: Histogram with Poisson uncertainties (raw data before filtering)
        bins = 200
        counts, bin_edges = np.histogram(series_clean, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = bin_edges[1] - bin_edges[0]

        total_count = np.sum(counts)
        density = counts / (total_count * bin_width)
        density_err = np.sqrt(counts) / (total_count * bin_width)

        # Step 5: PDF
        x = np.linspace(series_clean.min(), series_clean.max(), 1000)
        pdf = norm.pdf(x, loc=mu, scale=sigma)

        # Step 6: Plot
        plt.figure()
        plt.errorbar(bin_centers, density, yerr=density_err, fmt='o', alpha=0.6, label='Data with $\sqrt{N}$ error')
        plt.plot(x, pdf, 'r--', label=f'Normal Fit\n$\mu$={mu:.2f}, $\sigma$={sigma:.2f}')
        plt.axvline(norm.ppf(0.001, mu, sigma), color='k', linestyle='--', label='0.1% cutoff')
        plt.axvline(norm.ppf(0.999, mu, sigma), color='k', linestyle='--', label='99.9% cutoff')

        plt.title('Normal Fit with Z-score Filtering')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        if show_plots:
            plt.show()
        elif save_plots:
            if 'fig_idx' not in locals():
                fig_idx = 0
            new_figure_path = f"{figure_path}{fig_idx}_histo.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)
        plt.close()



if remove_outliers:
    print('Removing outliers and zero values...')
    def remove_outliers_and_zeroes(series_og):
        global fig_idx
        
        series = series_og.copy()
        median = series.mean()
        # median = series.median()
        std = series.std()
        z_scores = abs((series - median) / std)
        # z_scores = (series - median) / std
        
        # Remove zeroes for fitting
        filtered = z_scores[z_scores > 0]
        filtered = filtered[filtered < np.quantile(filtered, 0.99)]

        loc = 0
        scale = halfnorm.fit(filtered, floc=loc)[1]  # only fit the scale
        cutoff = halfnorm.ppf(0.999, loc=loc, scale=scale)

        # Plot
        x = np.linspace(filtered.min(), filtered.max(), 1000)
        pdf = halfnorm.pdf(x, loc=loc, scale=scale)

        plt.hist(filtered, bins=100, density=True, alpha=0.6, label='Data')
        plt.plot(x, pdf, 'r--', label=f'Half-Normal Fit\nσ={scale:.2f}')
        plt.axvline(cutoff, color='k', linestyle='--', label=f'99.9% cutoff = {cutoff:.2f}')
        plt.title('Half-Normal Fit and 99.9% Cutoff')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + "_half_normal_fit.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)
        plt.close()

        # Mask: outliers above cutoff or zero entries
        mask = (z_scores > cutoff) | (z_scores == 0)
        return mask

    # Initialize a mask of all False, meaning no rows are removed initially
    rows_to_remove = pd.Series(False, index=data_df.index)
    columns = ['events']
    for col in columns:
        rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df[col])
    
    # rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df['events'])
    data_df_cleaned = data_df[~rows_to_remove].copy()
    # Now, `data_df_cleaned` contains only rows that pass the conditions.
    print(f"Original DataFrame shape: {data_df.shape}")
    print(f"Cleaned DataFrame shape: {data_df_cleaned.shape}")
    data_df = data_df_cleaned.copy()


# -----------------------------------------------------------------------------
# Resampling the data in a larger time window: some averaged, some summed -----
# -----------------------------------------------------------------------------

data_df.set_index('Time', inplace=True)
data_df["number_of_mins"] = 1
columns_to_sum = angular_regions + detection_types + charge_types + ["number_of_mins"] + ["events"]
columns_to_mean = [col for col in data_df.columns if col not in columns_to_sum]

# Custom aggregation function
data_df = data_df.resample(resampling_window).agg({
    **{col: 'sum' for col in columns_to_sum},   # Sum the count and region columns
    **{col: 'mean' for col in columns_to_mean}  # Mean for the other columns
})

data_df.reset_index(inplace=True)


# -----------------------------------------------------------------------------
# ------------------------------ Plotting stuff -------------------------------
# -----------------------------------------------------------------------------

def plot_pressure_and_group(df, x_column, x_label, group_cols, time_col='Time', title=None, figsize=(14, 6), save_path=None):
    """
    Plot sensors_ext_Pressure_ext and a group of columns vs. time on two subplots.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        group_cols (list of str): List of column names to plot in lower panel.
        time_col (str): Column name representing time (default: 'Time').
        title (str): Optional title for the figure.
        figsize (tuple): Size of the figure.
        save_path (str): If provided, saves the figure to this path.
    """
    global fig_idx
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [1, 1]})

    # Plot pressure
    ax1.plot(df[time_col], df[x_column], label=x_label, color='tab:blue')
    ax1.set_ylabel(x_label)
    ax1.set_title(title if title else 'Group Signals')
    ax1.grid(True)
    ax1.legend()

    # Plot group of columns
    for col in group_cols:
        if col in df.columns:
            ax2.plot(df[time_col], df[col], label=col)
        else:
            print(f"Warning: column '{col}' not found in DataFrame")

    ax2.set_ylabel('Group Signals')
    ax2.set_xlabel('Time')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + "_multiple.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)
    plt.close()


def plot_grouped_series(df, group_cols, time_col='Time', title=None, figsize=(14, 4), save_path=None):
    """
    Plot time series for multiple groups of columns. Each sublist in `group_cols` is plotted in a separate subplot.
    
    Parameters:
        df (pd.DataFrame): DataFrame with time series data.
        group_cols (list of list of str): Each sublist contains column names to overlay in one subplot.
        time_col (str): Name of the time column.
        title (str): Title for the entire figure.
        figsize (tuple): Size of each subplot.
        save_path (str): If provided, save the figure to this path.
    """
    global fig_idx
    
    n_plots = len(group_cols)
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(figsize[0], figsize[1] * n_plots))
    
    if n_plots == 1:
        axes = [axes]  # Make iterable
    
    for idx, cols in enumerate(group_cols):
        ax = axes[idx]
        for col in cols:
            if col in df.columns:
                ax.plot(df[time_col], df[col], label=col)
            else:
                print(f"Warning: column '{col}' not found in DataFrame")
        ax.set_ylabel(' / '.join(cols))
        ax.grid(True)
        ax.legend(loc='best')

    axes[-1].set_xlabel('Time')
    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else None)

    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + "_multiple.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)
    plt.close()


# group_cols = [ 'original_tt_123', 'original_tt_12', 'original_tt_234', 'original_tt_34', 'original_tt_23', 'original_tt_1234', 'original_tt_134', 'original_tt_124', 'original_tt_13']
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'pressure', group_cols, title='Pressure and Selected Columns')

# group_cols = [ 'processed_tt_123', 'processed_tt_12', 'processed_tt_234', 'processed_tt_34', 'processed_tt_23', 'processed_tt_1234', 'processed_tt_24', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14', 'processed_tt_124' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'pressure', group_cols, title='Pressure and Selected Columns')

# group_cols = [ 'x', 'y']
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'theta' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = ['phi']
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'th_chi' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = ['s']
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'purity_of_data_percentage' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'streamer_percent_1', 'streamer_percent_2', 'streamer_percent_3', 'streamer_percent_4' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'rates_Multiplexer1', 'rates_M2', 'rates_M3', 'rates_M4' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'rates_CM1', 'rates_CM2', 'rates_CM3', 'rates_CM4' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'sigmoid_width' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'background_slope' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------- Calculating some columns ---------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

data_df['rate'] = data_df['events'] / ( data_df["number_of_mins"] * 60 )  # Counts per second (Hz)
data_df['unc_rate'] = np.sqrt(data_df['events']) / ( data_df["number_of_mins"] * 60 )

data_df['hv_mean'] = ( data_df['hv_HVneg'] + data_df['hv_HVpos'] ) / 2
data_df['current_mean'] = ( data_df['hv_CurrentNeg'] + data_df['hv_CurrentPos'] ) / 2


print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('-------------------- Calculating efficiencies ------------------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

print('----------------------------------------------------------------------')
print('--------------- 1. Standard method (includes system) -----------------')
print('----------------------------------------------------------------------')

def calculate_efficiency_uncertainty(N_measured, N_passed):
    # Ensure that the inputs are Series, handle element-wise computation
    uncertainty = np.where(N_passed > 0,
                           np.sqrt((N_measured / N_passed**2) + (N_measured**2 / N_passed**3)),
                           np.nan)  # If N_passed is 0, return NaN
    
    return uncertainty

# Print all the columns starting with processed_tt_
print(data_df.columns[data_df.columns.str.startswith('_tt')])

# Create explicit columns for detected and passed
# data_df['passed_1'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134', 'processed_tt_13',                    'processed_tt_14', 'processed_tt_12',                                                          ]].sum(axis=1, skipna=True)
# data_df['passed_2'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134', 'processed_tt_13', 'processed_tt_24', 'processed_tt_14', 'processed_tt_12', 'processed_tt_23',                                     ]].sum(axis=1, skipna=True)
# data_df['passed_3'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134', 'processed_tt_13', 'processed_tt_24', 'processed_tt_14',                    'processed_tt_23', 'processed_tt_34',                                    ]].sum(axis=1, skipna=True)
# data_df['passed_4'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',                    'processed_tt_24', 'processed_tt_14',                                       'processed_tt_34',               ]].sum(axis=1, skipna=True)

data_df['passed_1'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',   ]].sum(axis=1, skipna=True)
data_df['passed_2'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',   ]].sum(axis=1, skipna=True)
data_df['passed_3'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',   ]].sum(axis=1, skipna=True)
data_df['passed_4'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',   ]].sum(axis=1, skipna=True)

data_df['detected_1'] = data_df[['processed_tt_1234', 'processed_tt_123',                     'processed_tt_124', 'processed_tt_134' ]].sum(axis=1, skipna=True)
data_df['detected_2'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124',                    ]].sum(axis=1, skipna=True)
data_df['detected_3'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234',                     'processed_tt_134' ]].sum(axis=1, skipna=True)
data_df['detected_4'] = data_df[['processed_tt_1234',                     'processed_tt_234', 'processed_tt_124', 'processed_tt_134' ]].sum(axis=1, skipna=True)

# data_df['passed_1'] = data_df[['tracking_tt_1234', 'tracking_tt_123', 'tracking_tt_234', 'tracking_tt_12',                                    ]].sum(axis=1, skipna=True)
# data_df['passed_2'] = data_df[['tracking_tt_1234', 'tracking_tt_123', 'tracking_tt_234', 'tracking_tt_12', 'tracking_tt_23',                  ]].sum(axis=1, skipna=True)
# data_df['passed_3'] = data_df[['tracking_tt_1234', 'tracking_tt_123', 'tracking_tt_234',                   'tracking_tt_23', 'tracking_tt_34' ]].sum(axis=1, skipna=True)
# data_df['passed_4'] = data_df[['tracking_tt_1234', 'tracking_tt_123', 'tracking_tt_234',                                     'tracking_tt_34' ]].sum(axis=1, skipna=True)

# data_df['detected_1'] = data_df[['processed_tt_1234', 'processed_tt_123',                     'processed_tt_124', 'processed_tt_134', 'processed_tt_13',                    'processed_tt_14', 'processed_tt_12',                                                        ]].sum(axis=1, skipna=True)
# data_df['detected_2'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124',                                        'processed_tt_24',                    'processed_tt_12', 'processed_tt_23'                                    ]].sum(axis=1, skipna=True)
# data_df['detected_3'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234',                     'processed_tt_134', 'processed_tt_13',                                                          'processed_tt_23', 'processed_tt_34'                                    ]].sum(axis=1, skipna=True)
# data_df['detected_4'] = data_df[['processed_tt_1234',                     'processed_tt_234', 'processed_tt_124', 'processed_tt_134',                    'processed_tt_24', 'processed_tt_14',                                       'processed_tt_34'               ]].sum(axis=1, skipna=True)

data_df['non_detected_1'] = data_df[['processed_tt_234']].sum(axis=1, skipna=True)
data_df['non_detected_2'] = data_df[['processed_tt_134']].sum(axis=1, skipna=True)
data_df['non_detected_3'] = data_df[['processed_tt_124']].sum(axis=1, skipna=True)
data_df['non_detected_4'] = data_df[['processed_tt_123']].sum(axis=1, skipna=True)


print('Detected and passed calculated.')
print("Calculating efficiencies...")

# Calculate efficiencies and uncertainties explicitly

# Replace all zeroes by NaNs and do not interpolate
data_df = data_df.replace(0, np.nan)

rolling_effs = False
if rolling_effs:
    cols_to_interpolate = ['non_detected_1', 'non_detected_2', 'non_detected_3', 'non_detected_4',
                           'detected_1', 'detected_2', 'detected_3', 'detected_4',
                           'passed_1', 'passed_2', 'passed_3', 'passed_4']
    
    data_df[cols_to_interpolate] = data_df[cols_to_interpolate].replace(0, np.nan).interpolate(method='linear')
    
    sum_window, med_window = 5, 1
    rolling_sum, rolling_median = True, True
    
    if rolling_median:
        for col in cols_to_interpolate:
            data_df[col] = medfilt(data_df[col], kernel_size=med_window)
    
    if rolling_sum:
        for col in cols_to_interpolate:
            data_df[col] = data_df[col].rolling(window=sum_window, center=True, min_periods=1).sum()


def handle_efficiency_and_uncertainty_all(data_df, detected_cols, passed_cols, undetected_method):
    # Create empty columns for efficiencies and uncertainties
    efficiency_cols = [f'eff_{i}' for i in range(1, 5)]
    uncertainty_cols = [f'anc_unc_eff_{i}' for i in range(1, 5)]

    # Vectorized approach to calculate efficiency and uncertainty
    for i in range(4):  # For eff_1, eff_2, ..., eff_4
        detected_col = detected_cols[i]
        passed_col = passed_cols[i]
        
        if undetected_method == False:
        
            # Vectorized efficiency calculation (row-wise operation)
            data_df[efficiency_cols[i]] = np.where(
                (data_df[detected_col] != 0) & (data_df[passed_col] != 0),
                data_df[detected_col] / data_df[passed_col],
                np.nan
            )

            # Vectorized uncertainty calculation (row-wise operation)
            data_df[uncertainty_cols[i]] = np.where(
                (data_df[detected_col] != 0) & (data_df[passed_col] != 0),
                calculate_efficiency_uncertainty(data_df[detected_col], data_df[passed_col]),
                np.nan
            )
        
        else:
            
            # Vectorized efficiency calculation (row-wise operation)
            data_df[efficiency_cols[i]] = np.where(
                (data_df[detected_col] != 0) & (data_df[passed_col] != 0),
                1 - data_df[detected_col] / data_df[passed_col],
                np.nan
            )

            # Vectorized uncertainty calculation (row-wise operation)
            data_df[uncertainty_cols[i]] = np.where(
                (data_df[detected_col] != 0) & (data_df[passed_col] != 0),
                calculate_efficiency_uncertainty(data_df[detected_col], data_df[passed_col]),
                np.nan
            )

    return data_df


# Define columns for detected and passed values
non_detected_columns = ['non_detected_1', 'non_detected_2', 'non_detected_3', 'non_detected_4']
detected_columns = ['detected_1', 'detected_2', 'detected_3', 'detected_4']
passed_columns = ['passed_1', 'passed_2', 'passed_3', 'passed_4']

# Apply the function for all detected and passed columns
undetected_method = True
if undetected_method:
    data_df = handle_efficiency_and_uncertainty_all(data_df, non_detected_columns, passed_columns, undetected_method=True)
else:
    data_df = handle_efficiency_and_uncertainty_all(data_df, detected_columns, passed_columns, undetected_method=False)


# Add the systematic uncertainties to the efficiency calculation
data_df['unc_eff_1'] = np.sqrt( data_df['anc_unc_eff_1']**2 + systematic_unc[0]**2 )
data_df['unc_eff_2'] = np.sqrt( data_df['anc_unc_eff_2']**2 + systematic_unc[1]**2 )
data_df['unc_eff_3'] = np.sqrt( data_df['anc_unc_eff_3']**2 + systematic_unc[2]**2 )
data_df['unc_eff_4'] = np.sqrt( data_df['anc_unc_eff_4']**2 + systematic_unc[3]**2 )


acceptance_corr = False
if acceptance_corr:
    data_df['basic_eff_1'] = data_df['eff_1'] / data_df['acc_1']
    data_df['basic_eff_2'] = data_df['eff_2'] / data_df['acc_2']
    data_df['basic_eff_3'] = data_df['eff_3'] / data_df['acc_3']
    data_df['basic_eff_4'] = data_df['eff_4'] / data_df['acc_4']
else:
    data_df['basic_eff_1'] = data_df['eff_1']
    data_df['basic_eff_2'] = data_df['eff_2']
    data_df['basic_eff_3'] = data_df['eff_3']
    data_df['basic_eff_4'] = data_df['eff_4']


data_df['unc_basic_eff_1'] = data_df['unc_eff_1'] / data_df['acc_1']
data_df['unc_basic_eff_2'] = data_df['unc_eff_2'] / data_df['acc_2']
data_df['unc_basic_eff_3'] = data_df['unc_eff_3'] / data_df['acc_3']
data_df['unc_basic_eff_4'] = data_df['unc_eff_4'] / data_df['acc_4']



# Calculate the average efficiency
data_df['eff_global'] = data_df[['basic_eff_2', 'basic_eff_3']].mean(axis=1)

# Calculate the uncertainty for the average efficiency
data_df['unc_eff_global'] = np.sqrt(
    (data_df['unc_eff_2'] ** 2 +
     data_df['unc_eff_3'] ** 2) / 2
)






if undetected_method:
    data_df['basic_eff_1'] = data_df['basic_eff_2']
    data_df['basic_eff_4'] = data_df['basic_eff_3']


# # -----------------------------------------------------------------------------------------------------------------------------------
# # First equality case: corrected different trigger types separately -----------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------------------------

# data_df['eff_1234'] = data_df['basic_eff_1'] * data_df['basic_eff_2'] * data_df['basic_eff_3'] *  data_df['basic_eff_4']
# data_df['eff_134'] = data_df['basic_eff_1'] * ( 1 - data_df['basic_eff_2']) * data_df['basic_eff_3'] *  data_df['basic_eff_4']
# data_df['eff_124'] = data_df['basic_eff_1'] * data_df['basic_eff_2'] * ( 1 - data_df['basic_eff_3'] ) *  data_df['basic_eff_4']

# data_df['corrected_tt_1234'] = data_df['processed_tt_1234'] / data_df['eff_1234']
# data_df['corrected_tt_134'] = data_df['processed_tt_134'] / data_df['eff_134']
# data_df['corrected_tt_124'] = data_df['processed_tt_124'] / data_df['eff_124']

# # These should be the same, if the efficiencies are alright --------------------------------------------------------------
# group_cols = [ 'corrected_tt_1234' , 'corrected_tt_124', 'corrected_tt_134']
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')


# # -----------------------------------------------------------------------------------------------------------------------------------
# # Second equality case: the sum of cases should be the same as the corrected 1234 ---------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------------------------

# data_df['summed_tt_1234'] = data_df['processed_tt_1234'] + data_df['processed_tt_134'] + data_df['processed_tt_124']
# data_df['comp_eff'] = data_df['basic_eff_1'] * data_df['basic_eff_2'] * data_df['basic_eff_3'] *  data_df['basic_eff_4'] + \
#                         data_df['basic_eff_1'] * ( 1 - data_df['basic_eff_2']) * data_df['basic_eff_3'] *  data_df['basic_eff_4'] + \
#                         data_df['basic_eff_1'] * data_df['basic_eff_2'] * ( 1 - data_df['basic_eff_3'] ) *  data_df['basic_eff_4']
# data_df['corrected_tt_three_four'] = data_df['summed_tt_1234'] / data_df['comp_eff']

# # These should be the same, if the efficiencies are alright --------------------------------------------------------------
# group_cols = [ 'corrected_tt_three_four' , 'corrected_tt_1234']
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')




import numpy as np
import pandas as pd
from scipy.optimize import root


def solve_efficiencies(row):
    A = row['processed_tt_1234']
    B = row['processed_tt_134']
    C = row['processed_tt_124']

    # System of equations to solve
    def equations(vars):
        e1, e2, e3 = vars  # Let e4 = e1
        e4 = e1

        eq1 = B * e2 - A * (1 - e2)  # B*e2 = A*(1 - e2)
        eq2 = C * e3 - A * (1 - e3)  # C*e3 = A*(1 - e3)

        eff_combined = (
            e1 * e2 * e3 * e4 +
            e1 * (1 - e2) * e3 * e4 +
            e1 * e2 * (1 - e3) * e4
        )
        eq3 = (A + B + C) / eff_combined - A / (e1 * e2 * e3 * e4)

        return [eq1, eq2, eq3]

    # Initial guess
    initial_guess = [0.9, 0.9, 0.9]
    result = root(equations, initial_guess, method='hybr')

    if result.success and np.all((0 < result.x) & (result.x < 1)):
        e1, e2, e3 = result.x
        e4 = e1
        return pd.Series([e1, e2, e3, e4])
    else:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])

# Solve row by row
# data_df[['eff_sys_1', 'eff_sys_2', 'eff_sys_3', 'eff_sys_4']] = data_df.apply(solve_efficiencies, axis=1)




def solve_efficiencies_four_planes(row):
    A = row['processed_tt_1234']
    B = row['processed_tt_134']
    C = row['processed_tt_124']

    # System of equations to solve
    def equations_1(vars):
        e2, e3 = vars

        eq1 = B * e2 - A * (1 - e2)  # B*e2 = A*(1 - e2)
        eq2 = C * e3 - A * (1 - e3)  # C*e3 = A*(1 - e3)

        return [eq1, eq2]
    
    def equations_2(vars):
        e2, e3 = vars

        eq2 = C * e3 - A * (1 - e3)  # C*e3 = A*(1 - e3)

        eff_combined = (
            e2 * e3 +
            (1 - e2) * e3 +
            e2 * (1 - e3)
        )
        eq3 = (A + B + C) / eff_combined - A / ( e2 * e3 )
        
        return [eq2, eq3]
    
    def equations_3(vars):
        e2, e3 = vars

        eq1 = B * e2 - A * (1 - e2)  # B*e2 = A*(1 - e2)

        eff_combined = (
            e2 * e3 +
            (1 - e2) * e3 +
            e2 * (1 - e3)
        )
        eq3 = (A + B + C) / eff_combined - A / ( e2 * e3 )
        
        return [eq1, eq3]
    
    # Initial guess
    initial_guess = [0.9, 0.9]
    result_1 = root(equations_1, initial_guess, method='hybr')
    result_2 = root(equations_2, initial_guess, method='hybr')
    result_3 = root(equations_3, initial_guess, method='hybr')

    if result_1.success and np.all((0 < result_1.x) & (result_1.x < 1)) and\
        result_2.success and np.all((0 < result_2.x) & (result_2.x < 1)) and\
        result_3.success and np.all((0 < result_3.x) & (result_3.x < 1)):

        e2_1, e3_1 = result_1.x
        e2_2, e3_2 = result_2.x
        e2_3, e3_3 = result_3.x
        
        e2 = ( e2_1 + e2_2 + e2_3 ) / 3 
        e3 = ( e3_1 + e3_2 + e3_3 ) / 3 
        
        e4 = e1 = 0.9
        return pd.Series([e1, e2, e3, e4])
    else:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])


data_df[['eff_sys_1', 'eff_sys_2', 'eff_sys_3', 'eff_sys_4']] = data_df.apply(solve_efficiencies_four_planes, axis=1)


# -----------------------------------------------------------------------------------------------------------------------------------
# First equality case: corrected different trigger types separately -----------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

data_df['eff_1234'] = data_df['eff_sys_1'] * data_df['eff_sys_2'] * data_df['eff_sys_3'] *  data_df['eff_sys_4']
data_df['eff_134'] = data_df['eff_sys_1'] * ( 1 - data_df['eff_sys_2']) * data_df['eff_sys_3'] *  data_df['eff_sys_4']
data_df['eff_124'] = data_df['eff_sys_1'] * data_df['eff_sys_2'] * ( 1 - data_df['eff_sys_3'] ) *  data_df['eff_sys_4']

data_df['corrected_tt_1234'] = data_df['processed_tt_1234'] / data_df['eff_1234']
data_df['corrected_tt_134'] = data_df['processed_tt_134'] / data_df['eff_134']
data_df['corrected_tt_124'] = data_df['processed_tt_124'] / data_df['eff_124']

# -----------------------------------------------------------------------------------------------------------------------------------
# Second equality case: the sum of cases should be the same as the corrected 1234 ---------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

data_df['summed_tt_1234'] = data_df['processed_tt_1234'] + data_df['processed_tt_134'] + data_df['processed_tt_124']
data_df['comp_eff'] = data_df['eff_sys_1'] * data_df['eff_sys_2'] * data_df['eff_sys_3'] *  data_df['eff_sys_4'] + \
                        data_df['eff_sys_1'] * ( 1 - data_df['eff_sys_2']) * data_df['eff_sys_3'] *  data_df['eff_sys_4'] + \
                        data_df['eff_sys_1'] * data_df['eff_sys_2'] * ( 1 - data_df['eff_sys_3'] ) *  data_df['eff_sys_4']
data_df['corrected_tt_three_four'] = data_df['summed_tt_1234'] / data_df['comp_eff']

# These should be the same, if the efficiencies are alright --------------------------------------------------------------
group_cols = [ 'corrected_tt_three_four', 'corrected_tt_1234', 'corrected_tt_134', 'corrected_tt_124']
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')



# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# Three plane cases, strictly
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

# 'processed_tt_234', 'processed_tt_12', 'processed_tt_123', 'processed_tt_1234', 'processed_tt_23', 
# 'processed_tt_124', 'processed_tt_34', 'processed_tt_24', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'

# -------------------------------------------------------------------------------------------------------
# Subdetector 123 ---------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

data_df['subdetector_123_123'] = data_df['processed_tt_1234'] + data_df['processed_tt_123']
data_df['subdetector_123_12'] = data_df['processed_tt_12'] + data_df['processed_tt_124']
data_df['subdetector_123_23'] = data_df['processed_tt_23'] + data_df['processed_tt_234']
data_df['subdetector_123_13'] = data_df['processed_tt_13'] + data_df['processed_tt_134']

A = data_df['subdetector_123_123']
B = data_df['subdetector_123_13']
data_df['eff_sys_123_2'] = A / ( A + B )

# Newly calculated eff
data_df['subdetector_123_eff_123'] = data_df['eff_sys_1'] * data_df['eff_sys_123_2'] * data_df['eff_sys_3']
data_df['subdetector_123_123_corr'] = data_df['subdetector_123_123'] / data_df['subdetector_123_eff_123']

data_df['subdetector_123_eff_13'] = data_df['eff_sys_1'] * ( 1 - data_df['eff_sys_123_2'] ) * data_df['eff_sys_3']
data_df['subdetector_123_13_corr'] = data_df['subdetector_123_13'] / data_df['subdetector_123_eff_13']

data_df['subdetector_123_eff_summed'] = data_df['subdetector_123_eff_123'] + data_df['subdetector_123_eff_13']
data_df['subdetector_123_summed_corr'] = ( data_df['subdetector_123_123'] + data_df['subdetector_123_13'] ) / data_df['subdetector_123_eff_summed']

# group_cols = [ 'subdetector_123_summed_corr', 'subdetector_123_123_corr' , 'subdetector_123_13_corr']
# # plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'eff_sys_123_2', 'eff_sys_2' ]
# # plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')


# -------------------------------------------------------------------------------------------------------
# Subdetector 234 ---------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

data_df['subdetector_234_234'] = data_df['processed_tt_1234'] + data_df['processed_tt_234']
data_df['subdetector_234_23'] = data_df['processed_tt_23']  + data_df['processed_tt_123']
data_df['subdetector_234_34'] = data_df['processed_tt_34'] + data_df['processed_tt_134']
data_df['subdetector_234_24'] = data_df['processed_tt_24'] + data_df['processed_tt_124']

A = data_df['subdetector_234_234']
B = data_df['subdetector_234_24']
data_df['eff_sys_234_3'] = A / ( A + B )

# Newly calculated eff
data_df['subdetector_234_eff_234'] = data_df['eff_sys_2'] * data_df['eff_sys_234_3'] * data_df['eff_sys_4']
data_df['subdetector_234_234_corr'] = data_df['subdetector_234_234'] / data_df['subdetector_234_eff_234']

data_df['subdetector_234_eff_24'] = data_df['eff_sys_2'] * ( 1 - data_df['eff_sys_234_3'] ) * data_df['eff_sys_4']
data_df['subdetector_234_24_corr'] = data_df['subdetector_234_24'] / data_df['subdetector_234_eff_24']

data_df['subdetector_234_eff_summed'] = data_df['subdetector_234_eff_234'] + data_df['subdetector_234_eff_24']
data_df['subdetector_234_summed_corr'] = ( data_df['subdetector_234_234'] + data_df['subdetector_234_24'] ) / data_df['subdetector_234_eff_summed']

# group_cols = [ 'subdetector_234_summed_corr', 'subdetector_234_234_corr' , 'subdetector_234_24_corr']
# # plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'eff_sys_234_3', 'eff_sys_3' ]
# # plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')


# -------------------------------------------------------------------------------------------------------
# Stimated differences in efficiency --------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

data_df['eff_2_diff'] = ( data_df['eff_sys_123_2'] - data_df['eff_sys_2'] ) / data_df['eff_sys_2'] * 100
data_df['eff_3_diff'] = ( data_df['eff_sys_234_3'] - data_df['eff_sys_3'] ) / data_df['eff_sys_3'] * 100

# group_cols = [ 'eff_2_diff', 'eff_3_diff', 'streamer_percent_1', 'streamer_percent_2', 'streamer_percent_3', 'streamer_percent_4', ]
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')



# -------------------------------------------------------------------------------------------------------
# Noise derivations -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

data_df['eff_delta_2'] = data_df['eff_sys_123_2'] - data_df['eff_sys_2']
data_df['noise_frac_2'] = data_df['eff_delta_2'] / (data_df['eff_sys_2'] - data_df['eff_delta_2'])
data_df['noise_rate_2'] = data_df['noise_frac_2'] * data_df['subdetector_123_123'] / ( data_df["number_of_mins"] * 60 )  # or + 13 if total 2-plane

# Likewise for detector 3
data_df['eff_delta_3'] = data_df['eff_sys_234_3'] - data_df['eff_sys_3']
data_df['noise_frac_3'] = data_df['eff_delta_3'] / (data_df['eff_sys_3'] - data_df['eff_delta_3'])
data_df['noise_rate_3'] = data_df['noise_frac_3'] * data_df['subdetector_234_234'] / ( data_df["number_of_mins"] * 60 )


# group_cols = [ 'noise_rate_2', 'streamer_percent_2', ]
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'noise_rate_3', 'streamer_percent_3', ]
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')


group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['streamer_percent_2', 'streamer_percent_3'],
    ['rates_M2', 'rates_M3'],
    ['th_chi'],
    ['sigmoid_width'],
    ['background_slope'],
    ['noise_rate_2', 'noise_rate_3'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')




# -------------------------------------------------------------------------------------------------------
# Noise study -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

# 'processed_tt_234', 'processed_tt_12', 'processed_tt_123', 'processed_tt_1234', 'processed_tt_23', 
# 'processed_tt_124', 'processed_tt_34', 'processed_tt_24', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'

time_window = 200e-9  # 200 ns
data_df['noise_23'] = 2 * time_window * data_df['rates_M2'] * data_df['rates_M3']

group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['sigmoid_width'],
    ['background_slope'],
    ['noise_23'],
    ['noise_rate_2', 'noise_rate_3']
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')





group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['noise_23'],
    ['processed_tt_12'],
    ['processed_tt_23'],
    ['processed_tt_34'],
    ['processed_tt_24'],
    ['processed_tt_13'],
    ['processed_tt_14'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')

group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['noise_23'],
    ['processed_tt_123'],
    ['processed_tt_234'],
    ['processed_tt_134'],
    ['processed_tt_124'],
    ['processed_tt_1234'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')



group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['noise_23'],
    ['original_tt_12'],
    ['original_tt_23'],
    ['original_tt_34'],
    ['original_tt_13'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['noise_23'],
    ['original_tt_123'],
    ['original_tt_234'],
    ['original_tt_134'],
    ['original_tt_124'],
    ['original_tt_1234'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['tracking_tt_12'],
    ['tracking_tt_23'],
    ['tracking_tt_34'],
    ['tracking_tt_123'],
    ['tracking_tt_234'],
    ['tracking_tt_1234'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')



print('----------------------------------------------------------------------')
print('-------------- Following the subdetectors good idea ------------------')
print('----------------------------------------------------------------------')

# 'processed_tt_234', 'processed_tt_12', 'processed_tt_123', 'processed_tt_1234', 'processed_tt_23', 
# 'processed_tt_124', 'processed_tt_34', 'processed_tt_24', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'

data_df['final_eff_1'] = data_df['eff_sys_2']
data_df['final_eff_2'] = data_df['eff_sys_123_2']
data_df['final_eff_3'] = data_df['eff_sys_234_3']
data_df['final_eff_4'] = data_df['eff_sys_3']

e1 = data_df['final_eff_1']
e2 = data_df['final_eff_2']
e3 = data_df['final_eff_3']
e4 = data_df['final_eff_4']

# Detector 1234
# 'processed_tt_1234', 'processed_tt_124', 'processed_tt_134', 'processed_tt_14'
data_df['detector_1234'] = data_df['processed_tt_1234'] + data_df['processed_tt_124'] + data_df['processed_tt_134'] + data_df['processed_tt_14'] 

data_df['detector_1234'] = data_df['detector_1234']  / ( data_df["number_of_mins"] * 60 )

data_df['detector_1234_eff'] = e1 * e2 * e3 * e4 + \
    e1 * (1 - e2) * e3 * e4 + \
    e1 * e2 * (1 - e3) * e4 + \
    e1 * (1 - e2) * (1 - e3) * e4

# Detector 123
# 'processed_tt_123', 'processed_tt_1234',  
# 'processed_tt_124', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'
data_df['detector_123'] = data_df['processed_tt_1234'] + data_df['processed_tt_123'] + \
    data_df['processed_tt_13'] + data_df['processed_tt_134'] + data_df['processed_tt_124'] + data_df['processed_tt_14'] 

data_df['detector_123'] = data_df['detector_123'] / ( data_df["number_of_mins"] * 60 )

data_df['detector_123_eff'] = e1 * e2 * e3 + \
    e1 * (1 - e2) * e3

# Detector 234
# 'processed_tt_234', 'processed_tt_1234', 'processed_tt_14'
# 'processed_tt_124', 'processed_tt_24', 'processed_tt_134',
data_df['detector_234'] = data_df['processed_tt_234'] + data_df['processed_tt_1234'] + \
    data_df['processed_tt_14'] + data_df['processed_tt_124'] + data_df['processed_tt_24'] + data_df['processed_tt_134']

data_df['detector_234'] = data_df['detector_234'] / ( data_df["number_of_mins"] * 60 )

data_df['detector_234_eff'] = e2 * e3 * e4 + \
    e2 * (1 - e3) * e4

# Detector 12
# 'processed_tt_12', 'processed_tt_123', 'processed_tt_1234', 
# 'processed_tt_124', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'
data_df['detector_12'] = data_df['processed_tt_12'] + data_df['processed_tt_123'] + \
    data_df['processed_tt_1234'] + data_df['processed_tt_124'] + data_df['processed_tt_13'] + data_df['processed_tt_134'] + data_df['processed_tt_14'] 

data_df['detector_12'] = data_df['detector_12'] / ( data_df["number_of_mins"] * 60 )

data_df['detector_12_eff'] = e1 * e2

# Detector 23
# 'processed_tt_234', 'processed_tt_123', 'processed_tt_1234', 'processed_tt_23', 
# 'processed_tt_124', 'processed_tt_24', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'
data_df['detector_23'] = data_df['processed_tt_234'] + data_df['processed_tt_123'] + \
    data_df['processed_tt_1234'] + data_df['processed_tt_23'] + data_df['processed_tt_124'] + \
        data_df['processed_tt_24'] + data_df['processed_tt_13'] + data_df['processed_tt_134'] + data_df['processed_tt_14']

data_df['detector_23'] = data_df['detector_23'] / ( data_df["number_of_mins"] * 60 )

data_df['detector_23_eff'] = e2 * e3

# Detector 34
# 'processed_tt_234', 'processed_tt_1234',
# 'processed_tt_124', 'processed_tt_34', 'processed_tt_24', 'processed_tt_134', 'processed_tt_14'
data_df['detector_34'] = data_df['processed_tt_234'] + data_df['processed_tt_1234'] + \
    data_df['processed_tt_124'] + data_df['processed_tt_34'] + data_df['processed_tt_24'] + \
        data_df['processed_tt_134'] + data_df['processed_tt_14']

data_df['detector_34'] = data_df['detector_34'] / ( data_df["number_of_mins"] * 60 )

data_df['detector_34_eff'] = e3 * e4

group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['detector_1234'],
    ['detector_123'],
    ['detector_234'],
    ['detector_12'],
    ['detector_23'],
    ['detector_34'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')



group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['detector_1234_eff'],
    ['detector_123_eff'],
    ['detector_234_eff'],
    ['detector_12_eff'],
    ['detector_23_eff'],
    ['detector_34_eff'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


a = 1/0



print('----------------------------------------------------------------------')
print('------------------ Final efficiencies definition ---------------------')
print('----------------------------------------------------------------------')


data_df['final_eff_1'] = data_df['eff_sys_2']
data_df['final_eff_2'] = data_df['eff_sys_2']
data_df['final_eff_3'] = data_df['eff_sys_3']
data_df['final_eff_4'] = data_df['eff_sys_3']



# -------------------------------------------------------------------------------------------------------------------------
# Calculate the fit for the efficiencies ----------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

eff_fitting = True
if eff_fitting:
    filtered_df = data_df.dropna(subset=['basic_eff_1', 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    x = filtered_df['sensors_ext_Pressure_ext'].values.reshape(-1, 1)
    y = filtered_df['sensors_ext_Temperature_ext'].values.reshape(-1, 1)
    z = filtered_df['basic_eff_1'].values
    X = np.hstack((x, y))
    model = LinearRegression()
    model.fit(X, z)
    a, b = model.coef_
    c = model.intercept_
    formula = f"Eff = {a:.4g} * P + {b:.4g} * T + {c:.4g}"
    data_df['eff_fit_1'] = a * data_df['sensors_ext_Pressure_ext'] + b * data_df['sensors_ext_Temperature_ext'] + c
    data_df['unc_eff_fit_1'] = 1

    filtered_df = data_df.dropna(subset=['basic_eff_2', 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    x = filtered_df['sensors_ext_Pressure_ext'].values.reshape(-1, 1)
    y = filtered_df['sensors_ext_Temperature_ext'].values.reshape(-1, 1)
    z = filtered_df['basic_eff_2'].values
    X = np.hstack((x, y))
    model = LinearRegression()
    model.fit(X, z)
    a, b = model.coef_
    c = model.intercept_
    formula = f"Eff = {a:.4g} * P + {b:.4g} * T + {c:.4g}"
    data_df['eff_fit_2'] = a * data_df['sensors_ext_Pressure_ext'] + b * data_df['sensors_ext_Temperature_ext'] + c
    data_df['unc_eff_fit_2'] = 1

    filtered_df = data_df.dropna(subset=['basic_eff_3', 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    x = filtered_df['sensors_ext_Pressure_ext'].values.reshape(-1, 1)
    y = filtered_df['sensors_ext_Temperature_ext'].values.reshape(-1, 1)
    z = filtered_df['basic_eff_3'].values
    X = np.hstack((x, y))
    model = LinearRegression()
    model.fit(X, z)
    a, b = model.coef_
    c = model.intercept_
    formula = f"Eff = {a:.4g} * P + {b:.4g} * T + {c:.4g}"
    data_df['eff_fit_3'] = a * data_df['sensors_ext_Pressure_ext'] + b * data_df['sensors_ext_Temperature_ext'] + c
    data_df['unc_eff_fit_3'] = 1

    filtered_df = data_df.dropna(subset=['basic_eff_4', 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    x = filtered_df['sensors_ext_Pressure_ext'].values.reshape(-1, 1)
    y = filtered_df['sensors_ext_Temperature_ext'].values.reshape(-1, 1)
    z = filtered_df['basic_eff_4'].values
    X = np.hstack((x, y))
    model = LinearRegression()
    model.fit(X, z)
    a, b = model.coef_
    c = model.intercept_
    formula = f"Eff = {a:.4g} * P + {b:.4g} * T + {c:.4g}"
    data_df['eff_fit_4'] = a * data_df['sensors_ext_Pressure_ext'] + b * data_df['sensors_ext_Temperature_ext'] + c
    data_df['unc_eff_fit_4'] = 1



if create_plots:

    print("Creating efficiency comparison scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(1, 5):  # Modules 1 to 4
        ax.scatter(
            data_df[f'eff_fit_{i}'],
            data_df[f'basic_eff_{i}'],
            alpha=0.5,
            s=1,
            label=f'Module {i}',
            color=f'C{i}'
        )
    
    low_lim = 0.8
    
    # Plot y = x reference line
    ax.plot([low_lim, 1.0], [low_lim, 1.0], 'k--', linewidth=1, label='Ideal (y = x)')

    ax.set_xlabel('Fitted Efficiency')
    ax.set_ylabel('Measured Efficiency')
    ax.set_title('Measured vs Fitted Efficiency for All Modules')
    ax.set_xlim(low_lim, 1.0)
    ax.set_ylim(low_lim, 1.0)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + "_eff_vs_eff_fit_scatter.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)
    plt.close()






if create_plots:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(17, 14), sharex=True)
    for i in range(1, 5):  # Loop from 1 to 4
        ax = axes[i-1]  # pick the appropriate subplot
        # Plot basic_eff_i
        
        ax.plot(data_df['Time'], 1 - data_df[f'non_detected_{i}'] * data_df[f"acc_{i}"] / data_df[f'passed_{i}'], 
                label=f'Eff. {i}, undetected', color=f'C{i}', alpha=1)
        
        ax.plot(data_df['Time'], data_df[f'detected_{i}'] * data_df[f"acc_{i}"] / data_df[f'passed_{i}'], 
                label=f'Eff. {i}, detected', color=f'C{i+4}', alpha=1)
        
        
        ax.plot(data_df['Time'], data_df[f'basic_eff_{i}'], 
                label=f'Final Eff. {i}', color=f'C{i + 8}', alpha=1)
        ax.fill_between(data_df['Time'],
                        data_df[f'basic_eff_{i}'] - data_df[f'unc_basic_eff_{i}'],
                        data_df[f'basic_eff_{i}'] + data_df[f'unc_basic_eff_{i}'],
                        alpha=0.2, color=f'C{i}')
        
        ax.plot(data_df['Time'], data_df[f'eff_fit_{i}'], 
                label=f'Eff. {i} Fit', color=f'C{i + 12}', alpha=1)
        
        ax.plot(data_df['Time'], data_df[f'eff_sys_{i}'], 
                label=f'Eff. {i} from System solving', color=f'C{i + 16}', alpha=1)
        
        # Labeling and titles
        ax.set_ylabel('Efficiency')
        ax.set_ylim(0.8, 1.0)
        ax.set_title(f'Detected and passed')
        ax.legend(loc='upper left')
        
    # Label the common x-axis at the bottom
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + "_eff_all.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)
    plt.close()



# --------------------------------------------------------------------------------------------------------




a = 1/0

# --------------------------------------------------------------------------------------------------------

e1 = data_df['eff_sys_1']
e2 = data_df['eff_sys_2']
e3 = data_df['eff_sys_3']
e4 = data_df['eff_sys_4']

# Define all trigger patterns and their efficiencies
pattern_defs = {
    '1234': e1 * e2 * e3 * e4,
    '124':  e1 * e2 * (1 - e3) * e4,
    '134':  e1 * (1 - e2) * e3 * e4,
    # Add more patterns here if needed
}

# Initialize totals
total_rate = 0
total_eff = 0

# Loop over all patterns
for patt, eff in pattern_defs.items():
    colname = f'processed_tt_{patt}'
    if colname in data_df.columns:
        total_rate += data_df[colname]
        total_eff  += eff

# Compute final efficiency-corrected rate
data_df['rate_eff_corr'] = total_rate / total_eff / ( data_df["number_of_mins"] * 60 )

data_df['uncorrected'] = total_rate / ( data_df["number_of_mins"] * 60 )


group_cols = [ 'rate_eff_corr', 'uncorrected']
plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

group_cols = [ 'uncorrected',]
plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')





if create_plots:
    print("Creating efficiency and rate plots...")
    
    # Create four subplots in a single column, sharing the same x-axis
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharex=True)
    
    ax = axes  # pick the appropriate subplot
    ax.plot(data_df['Time'], data_df[f'rate_eff_corr'], 
            label=f'ad', color=f'b', alpha=0.5)
    
    ax.set_title(f'Corrected rate over Time')
    ax.legend(loc='upper left')
    
    # Label the common x-axis at the bottom
    axes.set_xlabel('Time')

    plt.tight_layout()

    # Save or show the plot
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + "_eff_new_method.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)

    plt.close()


a = 1/0


# -----------------------------------------------------------------------------

print('----------------------------------------------------------------------')
print('------ 2. Geometric method (requires complete angular spectra) -------')
print('----------------------------------------------------------------------')

print("WIP...")

# -----------------------------------------------------------------------------


print('----------------------------------------------------------------------')
print('----- 3. Inefficiency method (requires complete charge spectra) ------')
print('----------------------------------------------------------------------')

print("WIP...")

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Correct by the efficiency, calculate uncertainty of the corrected rate
# -----------------------------------------------------------------------------

print('Efficiency calculations performed.')

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
    ax1.set_ylim(0.5, 1.0)
    ax1.set_title('Efficiencies over Time')
    ax1.legend(loc='upper left')
    
    
    # Plot the measured: 12, 23, 34, 123, 234, 1234, 13, 24, 14
    ax2.plot(data_df['Time'], data_df['processed_tt_12'] / (data_df["number_of_mins"] * 60), label='Measured 12', color='C1')
    ax2.plot(data_df['Time'], data_df['processed_tt_23'] / (data_df["number_of_mins"] * 60), label='Measured 23', color='C2')
    ax2.plot(data_df['Time'], data_df['processed_tt_34'] / (data_df["number_of_mins"] * 60), label='Measured 34', color='C3')
    ax2.plot(data_df['Time'], data_df['processed_tt_123'] / (data_df["number_of_mins"] * 60), label='Measured 123', color='C4')
    ax2.plot(data_df['Time'], data_df['processed_tt_234'] / (data_df["number_of_mins"] * 60), label='Measured 234', color='C5')
    ax2.plot(data_df['Time'], data_df['processed_tt_1234'] / (data_df["number_of_mins"] * 60), label='Measured 1234', color='C6')
    
    ax2.set_ylim(0.1, 5)
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

    ax3.set_ylim(0.1, 5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Rate')
    ax3.set_title('Corrected by efficiency Rates over Time')
    ax3.legend(loc='upper left')
    
    plt.tight_layout()

    # Save or show the plot
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + "_effs_rates_new_sys.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)

    plt.close()



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
        new_figure_path = figure_path + f"{fig_idx}" + "_3D_fit.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format = 'png', dpi = 300)
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
        new_figure_path = figure_path + f"{fig_idx}" + "_residuals.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format = 'png', dpi = 300)
    plt.close()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Atmospheric corrections -----------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('-------------------- Atmospheric corrections started -----------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

# -----------------------------------------------------------------------------
# Pressure correction ---------------------------------------------------------
# -----------------------------------------------------------------------------

print('----------------------------------------------------------------------')
print('---------------------- Pressure correction started -------------------')
print('----------------------------------------------------------------------')

# Define the exponential model
def fit_model(x, beta, a):
    # [beta] = %/mbar
    return beta / 100 * x + a

def calculate_eta_P(I_over_I0, unc_I_over_I0, delta_P, unc_delta_P, region = None):
    global fig_idx
    
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
        
        # ------------------------------------------------------------------------------------------------
        # Filter outliers before fitting -----------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        
        z_scores = np.abs((df['log_I_over_I0'] - df['log_I_over_I0'].median()) / df['log_I_over_I0'].std())
        # z_scores = (df['log_I_over_I0'] - df['log_I_over_I0'].median()) / df['log_I_over_I0'].std()
        
        # Make a small histogram of the z_scores to see the distribution
        plt.hist(z_scores, bins=400)
        plt.axvline(x=z_score_th_pres_corr, color='r', linestyle='--', label='Threshold')
        plt.title(f'{region}\nZ-Scores Distribution')
        plt.xlabel('Z-Score')
        plt.ylabel('Frequency')
        if show_plots: 
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + "_pre_pressure_z" + f"{region}" + ".png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format = 'png', dpi = 300)
        plt.close()
        
        df = df[z_scores < z_score_th_pres_corr]
        
        
        # ------------------------------------------------------------------------------------------------
        
        # WIP TO USE UNCERTAINTY OF PRESSURE ----------------------------------------------
        popt, pcov = curve_fit(fit_model, df['delta_P'], df['log_I_over_I0'], sigma=df['unc_log_I_over_I0'], absolute_sigma=True, p0=(1,0))
        b, a = popt  # Extract parameters
        
        # Define eta_P as the parameter b (rate of change in the exponent)
        eta_P = b
        eta_P_ordinate = a
        eta_P_uncertainty = np.sqrt(np.diag(pcov))[0]
        
        # Plot the fitting
        if create_plots:
            plt.figure()
            if show_errorbar:
                plt.errorbar(df['delta_P'], df['log_I_over_I0'], xerr=abs(df['unc_delta_P']), yerr=abs(df['unc_log_I_over_I0']), fmt='o', label='Data with Uncertainty')
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
            plt.title(f'{region} - Exponential Fit with Uncertainty\nBeta (b) = {b:.3f} ± {unc_b:.3f} %/mbar')
            plt.legend()
            if show_plots: 
                plt.show()
            elif save_plots:
                new_figure_path = figure_path + f"{fig_idx}" + "_press_corr" + f"{region}" + ".png"
                fig_idx += 1
                print(f"Saving figure to {new_figure_path}")
                plt.savefig(new_figure_path, format = 'png', dpi = 300)
            plt.close()
    else:
        print("Fit not done, data empty. Returning NaN.")
        eta_P = np.nan
        eta_P_uncertainty = np.nan  # Handle case where there are no valid data points
    return eta_P, eta_P_uncertainty, eta_P_ordinate

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


# -------------------------------------------------------------------------------
# -------------------------- LOOPING THE DATA -----------------------------------
# -------------------------------------------------------------------------------

# region = 'eff_corr'
angular_regions = ['High', 'N', 'S', 'E', 'W']
trigger_types_corrected = ['new_sys_12', 'new_sys_23', 'new_sys_34', 'new_sys_123', 'new_sys_234', 'new_sys_1234']
regions_to_correct = ['eff_corr'] + angular_regions + trigger_types_corrected

log_delta_I_df = pd.DataFrame(columns=['Region', 'Log_I_over_I0', 'Delta_P', 'Unc_Log_I_over_I0', 'Unc_Delta_P', 'Eta_P', 'Unc_Eta_P'])

# List to store results
results = []

for region in regions_to_correct:

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

    I = data_df[region]
    try:
        unc_I = data_df[f'unc_{region}']
    except KeyError:
        unc_I = 1
    
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
        eta_P, unc_eta_P, eta_P_ordinate = calculate_eta_P(I_over_I0, unc_I_over_I0, delta_P, unc_delta_P, region)
        
        # Store entire vectors without flattening
        results.append({
            'Region': region,
            'Log_I_over_I0': np.log(I_over_I0),  # Entire vector
            'Delta_P': delta_P,  # Entire vector
            'Unc_Log_I_over_I0': unc_I_over_I0,  # Entire vector
            'Unc_Delta_P': unc_delta_P,  # Entire vector
            'Eta_P': eta_P,  # Scalar value for eta_P
            'Unc_Eta_P': unc_eta_P,  # Scalar uncertainty for eta_P
            'Eta_P_ordinate': eta_P_ordinate  # Scalar uncertainty for eta_P
        })

    # Convert the list of dictionaries into a DataFrame after the loop
    log_delta_I_df = pd.DataFrame(results)

    # if recalculate_pressure_coeff:
    #     eta_P, unc_eta_P = calculate_eta_P(I_over_I0, unc_I_over_I0, delta_P, unc_delta_P, region)
    #     print(eta_P)
    #     pressure_results = pd.concat([pressure_results, pd.DataFrame({'Region': [region], 'Eta_P': [eta_P]})], ignore_index=True)
        
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
                plt.errorbar(df['delta_P'], df['log_I_over_I0'], xerr=abs(df['unc_delta_P']), yerr=abs(df['unc_log_I_over_I0']), fmt='o', label='Data with Uncertainty')
            else:
                plt.scatter(df['delta_P'], df['log_I_over_I0'], label='Data', s=1, alpha=0.5, marker='.')
            
            # Plot the line using provided eta_P instead of fitted values
            plt.plot(df['delta_P'], fit_model(df['delta_P'], eta_P, set_a), color='blue', label=f'Set Eta: {eta_P:.3f} ± {unc_eta_P:.3f} %/mbar')
            
            # Add labels and title
            plt.xlabel('Delta P')
            plt.ylabel('log (I / I0)')
            plt.title(f'Plot of {region} using Set Eta_P\nEta_P = {eta_P:.3f} ± {unc_eta_P:.3f} %/mbar')
            plt.legend()
            
            if show_plots: 
                plt.show()
            elif save_plots:
                new_figure_path = figure_path + f"{fig_idx}" + "_press_fit" + f"{region}" + ".png"
                fig_idx += 1
                print(f"Saving figure to {new_figure_path}")
                plt.savefig(new_figure_path, format = 'png', dpi = 300)
            plt.close()


    # Create corrected rate column for the region
    data_df[f'pres_{region}'] = I * np.exp(-1 * eta_P / 100 * delta_P)

    # ------------------- Final uncertainty calculation in the corrected rate --------------------------

    unc_rate = 1
    unc_beta = unc_eta_P
    unc_DP = unc_delta_P
    term_1_rate = np.exp(-1 * eta_P / 100 * delta_P) * unc_rate
    term_2_beta = I * delta_P / 100 * np.exp(-1 * eta_P / 100 * delta_P) * unc_beta
    term_3_DP = I * eta_P / 100 * np.exp(-1 * eta_P / 100 * delta_P) * unc_DP
    final_unc_combined = np.sqrt(term_1_rate**2 + term_2_beta**2 + term_3_DP**2)
    data_df[f'unc_pres_{region}'] = final_unc_combined



# Convert the list of dictionaries into a DataFrame after the loop
log_delta_I_df = pd.DataFrame(results)

# --- Plotting the vectors ---
plt.figure(figsize=(12, 8))

# Loop through all regions
for region in log_delta_I_df['Region']:
    
    # If region does not contain new_, skip it
    if 'new_' not in region:
        continue
    
    # Extract data for the current region
    region_data = log_delta_I_df[log_delta_I_df['Region'] == region]

    # Access the full vectors (they are stored as columns, so we directly use them)
    delta_P = region_data['Delta_P'].values[0]  # Access the vector (1D)
    log_I_over_I0 = region_data['Log_I_over_I0'].values[0]  # Access the vector (1D)
    unc_delta_P = region_data['Unc_Delta_P'].values[0]  # Access the vector (1D)
    unc_log_I_over_I0 = region_data['Unc_Log_I_over_I0'].values[0]  # Access the vector (1D)
    
    eta_P = region_data['Eta_P'].values[0]  # Scalar value for eta_P
    eta_P_ordinate = region_data['Eta_P_ordinate'].values[0]  # Scalar value for eta_P_ordinate
    
    # Plot scatter for the current region
    plt.scatter(delta_P, log_I_over_I0, label=f'{region} Fit', s=2, alpha=0.8, marker='.')

    # Calculate the fitted values using the fit model
    fitted_values = fit_model(delta_P, eta_P, eta_P_ordinate)

    # Plot the line using eta_P (beta) and eta_P_ordinate (a)
    plt.plot(delta_P, fitted_values, label=f"{region} Fit Line", color=f'C{list(log_delta_I_df["Region"]).index(region)}', alpha=0.7)
    
    # Optional: plot with error bars if needed
    if show_errorbar:
        plt.errorbar(delta_P, log_I_over_I0, xerr=abs(unc_delta_P), yerr=abs(unc_log_I_over_I0), fmt='o', label=f'{region} Fit with Errors')

plt.xlabel('Delta P')
plt.ylabel('Log (I / I0)')
plt.ylim(-0.6, 0.5)
plt.title('Efficiency Fits for Different Regions')
plt.legend()
plt.grid(True)
if show_plots:
    plt.show()
elif save_plots:
    new_figure_path = figure_path + f"{fig_idx}" + "_GIANT_PRESSURE_PLOT.png"
    fig_idx += 1
    print(f"Saving figure to {new_figure_path}")
    plt.savefig(new_figure_path, format = 'png', dpi = 300)
plt.close()

# ---------------------------------------------------------------------------------------------------

# Filter regions that contain 'new_' to plot
regions_to_plot = [region for region in log_delta_I_df['Region'] if 'new_' in region]

# Create subplots dynamically based on the number of regions to plot
num_regions = len(regions_to_plot)
fig, axes = plt.subplots(nrows=num_regions, figsize=(12, 20), sharex=True, sharey=True)

# Loop through all regions and plot them in separate subplots
for idx, region in enumerate(regions_to_plot):
    
    # Extract data for the current region
    region_data = log_delta_I_df[log_delta_I_df['Region'] == region]

    # Access the full vectors (they are stored as columns, so we directly use them)
    delta_P = region_data['Delta_P'].values[0]  # Access the vector (1D)
    log_I_over_I0 = region_data['Log_I_over_I0'].values[0]  # Access the vector (1D)
    unc_delta_P = region_data['Unc_Delta_P'].values[0]  # Access the vector (1D)
    unc_log_I_over_I0 = region_data['Unc_Log_I_over_I0'].values[0]  # Access the vector (1D)
    
    eta_P = region_data['Eta_P'].values[0]  # Scalar value for eta_P
    eta_P_ordinate = region_data['Eta_P_ordinate'].values[0]  # Scalar value for eta_P_ordinate
    
    # Plot scatter for the current region on the appropriate subplot
    ax = axes[idx]  # Get the correct subplot based on idx
    ax.scatter(delta_P, log_I_over_I0, label=f'{region} Fit', s=1, alpha=0.8, marker='.')

    # Calculate the fitted values using the fit model
    fitted_values = fit_model(delta_P, eta_P, eta_P_ordinate)

    # Plot the line using eta_P (beta) and eta_P_ordinate (a)
    ax.plot(delta_P, fitted_values, label=f"{region} Fit Line", color=f'C{idx}', alpha=0.7)
    
    # Optional: plot with error bars if needed
    if show_errorbar:
        ax.errorbar(delta_P, log_I_over_I0, xerr=abs(unc_delta_P), yerr=abs(unc_log_I_over_I0), fmt='o', label=f'{region} Fit with Errors')

    # Add labels and title to the subplots
    ax.set_xlabel('Delta P')
    ax.set_ylabel('Log (I / I0)')
    ax.set_ylim(-0.6, 0.5)
    ax.set_title(f'Efficiency Fit for {region}')
    ax.legend()
    ax.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show or save the plot
if show_plots: 
    plt.show()
elif save_plots:
    new_figure_path = figure_path + f"{fig_idx}" + "_GIANT_PRESSURE_PLOT.png"
    fig_idx += 1
    print(f"Saving figure to {new_figure_path}")
    plt.savefig(new_figure_path, format='png', dpi=300)

plt.close()




# ---------------------------------------------------------------------------------------------------

# Add a new outlier filter to the pressure correction
region = "eff_corr"
after_press_z_score_th = 0.5

if remove_outliers:
    print('Removing outliers and zero values...')
    def remove_outliers_and_zeroes(series, z_thresh=outlier_filter):
        global fig_idx
        
        """
        Create a mask of rows that are outliers or have zero values.
        """
        # median = series.mean()
        median = series.median()
        std = series.std()
        # z_scores = abs((series - median) / std)
        z_scores = (series - median) / std
        
        plt.hist(z_scores, bins=300)
        plt.axvline(x=z_thresh, color='r', linestyle='--', label='Threshold')
        plt.axvline(x=-1*z_thresh, color='r', linestyle='--', label='Threshold')
        plt.title('Z-Scores Distribution')
        plt.xlabel('Z-Score')
        plt.ylabel('Frequency')
        if show_plots: 
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + "_after_pressure_corr_z.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format = 'png', dpi = 300)
        plt.close()
        
        # print(z_scores)
        # Create a mask for rows where z_scores > z_thresh or values are zero
        mask = (abs(z_scores) > z_thresh) | (series == 0)
        return mask

    # Initialize a mask of all False, meaning no rows are removed initially
    rows_to_remove = pd.Series(False, index=data_df.index)
    rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df[f'pres_{region}'], z_thresh = after_press_z_score_th)

    data_df_cleaned = data_df[~rows_to_remove].copy()
    data_df = data_df_cleaned.copy()


#%%

# -----------------------------------------------------------------------------
# High order correction -------------------------------------------------------
# -----------------------------------------------------------------------------

# WIP!!!!!!

print('----------------------------------------------------------------------')
print('--------------------- High order correction started ------------------')
print('----------------------------------------------------------------------')

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
        global fig_idx
        
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
                new_figure_path = figure_path + f"{fig_idx}" + "_high_order.png"
                fig_idx += 1
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
    ax1.set_ylim(0.5, 1)
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
    ax2.set_ylim(5, 15)
    ax2.set_title('Rates over Time')
    ax2.legend(loc='upper left')

    plt.tight_layout()

    # Save or show the plot
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + "_effs_rates_naasd_pressure.png"
        fig_idx += 1
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

sys.exit(0)

# -------------------------------------------------------------------------------------
# ----------------JUNK ----------------------------------------------------------------
# -------------------------------------------------------------------------------------

# high_regions_hans = ['V']
# mid_regions_hans = ['N.M', 'NE.M', 'E.M', 'SE.M', 'S.M', 'SW.M', 'W.M', 'NW.M']
# low_regions_hans = ['N.H', 'E.H', 'S.H', 'W.H']
# angular_regions = high_regions_hans + mid_regions_hans + low_regions_hans

# for reg in angular_regions:
#     data_df[f'pres_{reg}'] = data_df[reg] * np.exp(-1 * eta_P / 100 * delta_P)

# # Plot all the time series in angular_regions
# if create_plots:
#     print("Creating multi-panel count plots for all angular regions...")

#     # Create figures with 4 subplots each, sharing x-axis
#     fig, axes_original = plt.subplots(4, 1, figsize=(17, 12), sharex=True)
#     fig_corr, axes_corrected = plt.subplots(4, 1, figsize=(17, 12), sharex=True)

#     # Define angular region groups
#     regions_v = ['V']
#     regions_main = ['N.M', 'E.M', 'W.M', 'S.M']
#     regions_diagonal = ['NE.M', 'SE.M', 'SW.M', 'NW.M']
#     regions_h = ['N.H', 'E.H', 'S.H', 'W.H']

#     region_groups = [regions_v, regions_main, regions_diagonal, regions_h]

#     # ---- ORIGINAL COUNTS ----
#     for ax, regions in zip(axes_original, region_groups):
#         for region in regions:
#             ax.plot(data_df['Time'], data_df[region] / (60 * res_win_min), label=f'{region} (Hz)')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)

#     axes_original[0].set_title('Original Counts - V')
#     axes_original[1].set_title('Original Counts - N.M, E.M, W.M, S.M')
#     axes_original[2].set_title('Original Counts - NE.M, SE.M, SW.M, NW.M')
#     axes_original[3].set_title('Original Counts - H Regions (N.H, E.H, S.H, W.H)')
#     axes_original[-1].set_xlabel('Time')

#     # ---- PRESSURE-CORRECTED COUNTS ----
#     for ax, regions in zip(axes_corrected, region_groups):
#         for region in regions:
#             ax.plot(data_df['Time'], data_df[f'pres_{region}'], label=f'Corrected {region}')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)

#     axes_corrected[0].set_title('Pressure-Corrected Counts - V')
#     axes_corrected[1].set_title('Pressure-Corrected Counts - N.M, E.M, W.M, S.M')
#     axes_corrected[2].set_title('Pressure-Corrected Counts - NE.M, SE.M, SW.M, NW.M')
#     axes_corrected[3].set_title('Pressure-Corrected Counts - H Regions (N.H, E.H, S.H, W.H)')
#     fig.suptitle("Rates for All Angular Regions", fontsize=14, fontweight='bold')
#     axes_corrected[-1].set_xlabel('Time')

#     plt.tight_layout()

#     # Save or show the plots
#     if show_plots:
#         plt.show()
#     elif save_plots:
#         fig.savefig(figure_path + f"{fig_idx}" + "_counts_original.png", format='png', dpi=300)
#         fig_idx += 1
#         fig_corr.savefig(figure_path + f"{fig_idx}" + "_counts_corrected.png", format='png', dpi=300)
#         fig_idx += 1
#         print("Saved multi-panel count plots.")

#     plt.close(fig)
#     plt.close(fig_corr)

# if create_plots:
#     print("Creating multi-panel count plots for all angular regions...")

#     # Create figures with 4 subplots each, sharing x-axis
#     fig, axes_original = plt.subplots(4, 1, figsize=(17, 12), sharex=True)
#     fig_corr, axes_corrected = plt.subplots(4, 1, figsize=(17, 12), sharex=True)

#     # Define angular region groups
#     regions_v = ['V']
#     regions_main = ['N.M', 'E.M', 'W.M', 'S.M']
#     regions_diagonal = ['NE.M', 'SE.M', 'SW.M', 'NW.M']
#     regions_h = ['N.H', 'E.H', 'S.H', 'W.H']

#     region_groups = [regions_v, regions_main, regions_diagonal, regions_h]

#     # ---- ORIGINAL COUNTS ----
#     for ax, regions in zip(axes_original, region_groups):
#         norm_offset = 0
#         for region in regions:
#             y = data_df[region]
#             y_norm = (y - y.mean()) / y.mean() + norm_offset
#             norm_offset += 0.1
#             ax.plot(data_df['Time'], y_norm, label=f'{region}')
#             # ax.plot(data_df['Time'], data_df[region], label=f'{region}')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)

#     axes_original[0].set_title('Original Counts - V')
#     axes_original[1].set_title('Original Counts - N.M, E.M, W.M, S.M')
#     axes_original[2].set_title('Original Counts - NE.M, SE.M, SW.M, NW.M')
#     axes_original[3].set_title('Original Counts - H Regions (N.H, E.H, S.H, W.H)')
#     fig.suptitle("Normalized Counts for All Angular Regions", fontsize=14, fontweight='bold')
#     axes_original[-1].set_xlabel('Time')

#     # ---- PRESSURE-CORRECTED COUNTS ----
#     for ax, regions in zip(axes_corrected, region_groups):
#         for region in regions:
#             ax.plot(data_df['Time'], data_df[f'pres_{region}'], label=f'Corrected {region}')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)

#     axes_corrected[0].set_title('Pressure-Corrected Counts - V')
#     axes_corrected[1].set_title('Pressure-Corrected Counts - N.M, E.M, W.M, S.M')
#     axes_corrected[2].set_title('Pressure-Corrected Counts - NE.M, SE.M, SW.M, NW.M')
#     axes_corrected[3].set_title('Pressure-Corrected Counts - H Regions (N.H, E.H, S.H, W.H)')
#     axes_corrected[-1].set_xlabel('Time')

#     plt.tight_layout()

#     # Save or show the plots
#     if show_plots:
#         plt.show()
#     elif save_plots:
#         fig.savefig(figure_path + f"{fig_idx}" + "_counts_original_norm.png", format='png', dpi=300)
#         fig_idx += 1
#         fig_corr.savefig(figure_path + f"{fig_idx}" + "_counts_corrected_norm.png", format='png', dpi=300)
#         fig_idx += 1
#         print("Saved multi-panel count plots.")

#     plt.close(fig)
#     plt.close(fig_corr)



# # Define angular regions
# angular_regions = ['High', 'N', 'S', 'E', 'W']

# # Apply pressure correction
# for reg in angular_regions:
#     data_df[f'pres_{reg}'] = data_df[reg] * np.exp(-1 * eta_P / 100 * delta_P)

# # Plot all the time series in angular_regions
# if create_plots:
#     print("Creating multi-panel count plots for all angular regions...")

#     # Create figures with subplots, sharing x-axis
#     fig, axes_original = plt.subplots(len(angular_regions), 1, figsize=(17, 12), sharex=True)
#     fig_corr, axes_corrected = plt.subplots(len(angular_regions), 1, figsize=(17, 12), sharex=True)

#     # ---- ORIGINAL COUNTS ----
#     for ax, region in zip(axes_original, angular_regions):
#         ax.plot(data_df['Time'], data_df[region] / (60 * res_win_min), label=f'{region} (Hz)')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)
#         ax.set_title(f'Original Counts - {region}')
    
#     axes_original[-1].set_xlabel('Time')

#     # ---- PRESSURE-CORRECTED COUNTS ----
#     for ax, region in zip(axes_corrected, angular_regions):
#         ax.plot(data_df['Time'], data_df[f'pres_{region}'], label=f'Corrected {region}')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)
#         ax.set_title(f'Pressure-Corrected Counts - {region}')
    
#     axes_corrected[-1].set_xlabel('Time')

#     fig.suptitle("Rates for All Angular Regions", fontsize=14, fontweight='bold')
#     plt.tight_layout()

#     # Save or show the plots
#     if show_plots:
#         plt.show()
#     elif save_plots:
#         fig.savefig(figure_path + f"{fig_idx}" + "_angular_caye_OG.png", format='png', dpi=300)
#         fig_idx += 1
#         fig_corr.savefig(figure_path + f"{fig_idx}" + "_counts_caye_corrected.png", format='png', dpi=300)
#         fig_idx += 1
#         print("Saved multi-panel count plots.")

#     plt.close(fig)
#     plt.close(fig_corr)