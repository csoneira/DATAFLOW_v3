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
from scipy.stats import pearsonr


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

remove_outliers = True

# Plotting configuration
show_plots = False
save_plots = True
create_plots = True
show_errorbar = False

recalculate_pressure_coeff = True

res_win_min = 30 # 180 Resampling window minutes

if int(station) == 4:
    res_win_min = 30

print(f"Resampling window set to {res_win_min} minutes.")

HMF_ker = 3 # It must be odd. Horizontal Median Filter
MAF_ker = 0 # Moving Average Filter

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
    
    if create_plots:
        # Step 6: Plot
        plt.figure()
        plt.errorbar(bin_centers, density, yerr=density_err, fmt='o', alpha=0.6, label='Data with $\sqrt{N}$ error')
        plt.plot(x, pdf, 'r--', label=f'Normal Fit\n$\mu$={mu:.2f}, $\sigma$={sigma:.2f},\nRelative error={sigma/mu*100:.2f}%')
        plt.axvline(norm.ppf(0.001, mu, sigma), color='k', linestyle='--', label='0.1% cutoff')
        plt.axvline(norm.ppf(0.999, mu, sigma), color='k', linestyle='--', label='99.9% cutoff')

        plt.title('Normal Fit with Z-score Filtering')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = f"{figure_path}{fig_idx}_histo.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)
        plt.close()
    else:
        print("Plotting is disabled. Set `create_plots = True` to enable plotting.")


if remove_outliers:
    print('Removing outliers and zero values...')
    def remove_outliers_and_zeroes(series_og):
        global create_plots, fig_idx
        
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
        
        if create_plots:
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
    global create_plots, fig_idx
    
    if create_plots:
        
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
    else:
        print("Plotting is disabled. Set `create_plots = True` to enable plotting.")


import matplotlib.ticker as mtick

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
    global create_plots, fig_idx
    
    if create_plots:
    
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
            
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

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
    else:
        print("Plotting is disabled. Set `create_plots = True` to enable plotting.")


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

# print('----------------------------------------------------------------------')
# print('--------------- 1. Standard method (includes system) -----------------')
# print('----------------------------------------------------------------------')

# def calculate_efficiency_uncertainty(N_measured, N_passed):
#     # Ensure that the inputs are Series, handle element-wise computation
#     uncertainty = np.where(N_passed > 0,
#                            np.sqrt((N_measured / N_passed**2) + (N_measured**2 / N_passed**3)),
#                            np.nan)  # If N_passed is 0, return NaN
    
#     return uncertainty

# # Print all the columns starting with processed_tt_
# print(data_df.columns[data_df.columns.str.startswith('_tt')])

# # Create explicit columns for detected and passed
# # data_df['passed_1'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134', 'processed_tt_13',                    'processed_tt_14', 'processed_tt_12',                                                          ]].sum(axis=1, skipna=True)
# # data_df['passed_2'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134', 'processed_tt_13', 'processed_tt_24', 'processed_tt_14', 'processed_tt_12', 'processed_tt_23',                                     ]].sum(axis=1, skipna=True)
# # data_df['passed_3'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134', 'processed_tt_13', 'processed_tt_24', 'processed_tt_14',                    'processed_tt_23', 'processed_tt_34',                                    ]].sum(axis=1, skipna=True)
# # data_df['passed_4'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',                    'processed_tt_24', 'processed_tt_14',                                       'processed_tt_34',               ]].sum(axis=1, skipna=True)

# data_df['passed_1'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',   ]].sum(axis=1, skipna=True)
# data_df['passed_2'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',   ]].sum(axis=1, skipna=True)
# data_df['passed_3'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',   ]].sum(axis=1, skipna=True)
# data_df['passed_4'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134',   ]].sum(axis=1, skipna=True)

# data_df['detected_1'] = data_df[['processed_tt_1234', 'processed_tt_123',                     'processed_tt_124', 'processed_tt_134' ]].sum(axis=1, skipna=True)
# data_df['detected_2'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124',                    ]].sum(axis=1, skipna=True)
# data_df['detected_3'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234',                     'processed_tt_134' ]].sum(axis=1, skipna=True)
# data_df['detected_4'] = data_df[['processed_tt_1234',                     'processed_tt_234', 'processed_tt_124', 'processed_tt_134' ]].sum(axis=1, skipna=True)

# # data_df['passed_1'] = data_df[['tracking_tt_1234', 'tracking_tt_123', 'tracking_tt_234', 'tracking_tt_12',                                    ]].sum(axis=1, skipna=True)
# # data_df['passed_2'] = data_df[['tracking_tt_1234', 'tracking_tt_123', 'tracking_tt_234', 'tracking_tt_12', 'tracking_tt_23',                  ]].sum(axis=1, skipna=True)
# # data_df['passed_3'] = data_df[['tracking_tt_1234', 'tracking_tt_123', 'tracking_tt_234',                   'tracking_tt_23', 'tracking_tt_34' ]].sum(axis=1, skipna=True)
# # data_df['passed_4'] = data_df[['tracking_tt_1234', 'tracking_tt_123', 'tracking_tt_234',                                     'tracking_tt_34' ]].sum(axis=1, skipna=True)

# # data_df['detected_1'] = data_df[['processed_tt_1234', 'processed_tt_123',                     'processed_tt_124', 'processed_tt_134', 'processed_tt_13',                    'processed_tt_14', 'processed_tt_12',                                                        ]].sum(axis=1, skipna=True)
# # data_df['detected_2'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234', 'processed_tt_124',                                        'processed_tt_24',                    'processed_tt_12', 'processed_tt_23'                                    ]].sum(axis=1, skipna=True)
# # data_df['detected_3'] = data_df[['processed_tt_1234', 'processed_tt_123', 'processed_tt_234',                     'processed_tt_134', 'processed_tt_13',                                                          'processed_tt_23', 'processed_tt_34'                                    ]].sum(axis=1, skipna=True)
# # data_df['detected_4'] = data_df[['processed_tt_1234',                     'processed_tt_234', 'processed_tt_124', 'processed_tt_134',                    'processed_tt_24', 'processed_tt_14',                                       'processed_tt_34'               ]].sum(axis=1, skipna=True)

# data_df['non_detected_1'] = data_df[['processed_tt_234']].sum(axis=1, skipna=True)
# data_df['non_detected_2'] = data_df[['processed_tt_134']].sum(axis=1, skipna=True)
# data_df['non_detected_3'] = data_df[['processed_tt_124']].sum(axis=1, skipna=True)
# data_df['non_detected_4'] = data_df[['processed_tt_123']].sum(axis=1, skipna=True)


# print('Detected and passed calculated.')
# print("Calculating efficiencies...")

# # Calculate efficiencies and uncertainties explicitly

# # Replace all zeroes by NaNs and do not interpolate
# data_df = data_df.replace(0, np.nan)

# rolling_effs = False
# if rolling_effs:
#     cols_to_interpolate = ['non_detected_1', 'non_detected_2', 'non_detected_3', 'non_detected_4',
#                            'detected_1', 'detected_2', 'detected_3', 'detected_4',
#                            'passed_1', 'passed_2', 'passed_3', 'passed_4']
    
#     data_df[cols_to_interpolate] = data_df[cols_to_interpolate].replace(0, np.nan).interpolate(method='linear')
    
#     sum_window, med_window = 5, 1
#     rolling_sum, rolling_median = True, True
    
#     if rolling_median:
#         for col in cols_to_interpolate:
#             data_df[col] = medfilt(data_df[col], kernel_size=med_window)
    
#     if rolling_sum:
#         for col in cols_to_interpolate:
#             data_df[col] = data_df[col].rolling(window=sum_window, center=True, min_periods=1).sum()


# def handle_efficiency_and_uncertainty_all(data_df, detected_cols, passed_cols, undetected_method):
#     # Create empty columns for efficiencies and uncertainties
#     efficiency_cols = [f'eff_{i}' for i in range(1, 5)]
#     uncertainty_cols = [f'anc_unc_eff_{i}' for i in range(1, 5)]

#     # Vectorized approach to calculate efficiency and uncertainty
#     for i in range(4):  # For eff_1, eff_2, ..., eff_4
#         detected_col = detected_cols[i]
#         passed_col = passed_cols[i]
        
#         if undetected_method == False:
        
#             # Vectorized efficiency calculation (row-wise operation)
#             data_df[efficiency_cols[i]] = np.where(
#                 (data_df[detected_col] != 0) & (data_df[passed_col] != 0),
#                 data_df[detected_col] / data_df[passed_col],
#                 np.nan
#             )

#             # Vectorized uncertainty calculation (row-wise operation)
#             data_df[uncertainty_cols[i]] = np.where(
#                 (data_df[detected_col] != 0) & (data_df[passed_col] != 0),
#                 calculate_efficiency_uncertainty(data_df[detected_col], data_df[passed_col]),
#                 np.nan
#             )
        
#         else:
            
#             # Vectorized efficiency calculation (row-wise operation)
#             data_df[efficiency_cols[i]] = np.where(
#                 (data_df[detected_col] != 0) & (data_df[passed_col] != 0),
#                 1 - data_df[detected_col] / data_df[passed_col],
#                 np.nan
#             )

#             # Vectorized uncertainty calculation (row-wise operation)
#             data_df[uncertainty_cols[i]] = np.where(
#                 (data_df[detected_col] != 0) & (data_df[passed_col] != 0),
#                 calculate_efficiency_uncertainty(data_df[detected_col], data_df[passed_col]),
#                 np.nan
#             )

#     return data_df


# # Define columns for detected and passed values
# non_detected_columns = ['non_detected_1', 'non_detected_2', 'non_detected_3', 'non_detected_4']
# detected_columns = ['detected_1', 'detected_2', 'detected_3', 'detected_4']
# passed_columns = ['passed_1', 'passed_2', 'passed_3', 'passed_4']

# # Apply the function for all detected and passed columns
# undetected_method = True
# if undetected_method:
#     data_df = handle_efficiency_and_uncertainty_all(data_df, non_detected_columns, passed_columns, undetected_method=True)
# else:
#     data_df = handle_efficiency_and_uncertainty_all(data_df, detected_columns, passed_columns, undetected_method=False)



# # Add the systematic uncertainties to the efficiency calculation
# data_df['unc_eff_1'] = np.sqrt( data_df['anc_unc_eff_1']**2 + systematic_unc[0]**2 )
# data_df['unc_eff_2'] = np.sqrt( data_df['anc_unc_eff_2']**2 + systematic_unc[1]**2 )
# data_df['unc_eff_3'] = np.sqrt( data_df['anc_unc_eff_3']**2 + systematic_unc[2]**2 )
# data_df['unc_eff_4'] = np.sqrt( data_df['anc_unc_eff_4']**2 + systematic_unc[3]**2 )


# data_df['unc_basic_eff_1'] = data_df['unc_eff_1'] / data_df['acc_1']
# data_df['unc_basic_eff_2'] = data_df['unc_eff_2'] / data_df['acc_2']
# data_df['unc_basic_eff_3'] = data_df['unc_eff_3'] / data_df['acc_3']
# data_df['unc_basic_eff_4'] = data_df['unc_eff_4'] / data_df['acc_4']


# # Calculate the average efficiency
# data_df['eff_global'] = data_df[['basic_eff_2', 'basic_eff_3']].mean(axis=1)

# # Calculate the uncertainty for the average efficiency
# data_df['unc_eff_global'] = np.sqrt(
#     (data_df['unc_eff_2'] ** 2 +
#      data_df['unc_eff_3'] ** 2) / 2
# )








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


data_df[['ancillary_1', 'eff_sys_2', 'eff_sys_3', 'ancillary_4']] = data_df.apply(solve_efficiencies_four_planes, axis=1)


def solve_efficiencies_four_planes(row):
    A = row['processed_tt_1234']
    B = row['processed_tt_234']
    C = row['processed_tt_123']

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
    initial_guess = [0.6, 0.6]
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
        return pd.Series([e2, e1, e4, e3])
    else:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])


data_df[['eff_sys_1', 'ancillary_2', 'ancillary_3', 'eff_sys_4']] = data_df.apply(solve_efficiencies_four_planes, axis=1)


group_cols = [
    ['eff_sys_1'],
    ['eff_sys_2'],
    ['eff_sys_3'],
    ['eff_sys_4'] ]

# plot_grouped_series(data_df, group_cols, title='Four plane efficiencies')


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
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')



# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# Three plane cases, strictly -------------------------------------------------------------------------------------------------------
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

# Plane 2
A = data_df['subdetector_123_123']
B = data_df['subdetector_123_13']
data_df['eff_sys_123_2'] = A / ( A + B )

# Plane 1
A = data_df['subdetector_123_123']
B = data_df['subdetector_123_23']
data_df['eff_sys_123_1'] = A / ( A + B )

# Plane 3
A = data_df['subdetector_123_123']
B = data_df['subdetector_123_12']
data_df['eff_sys_123_3'] = A / ( A + B )


# Newly calculated eff --------------------------------------------------------------------------------------
data_df['subdetector_123_eff_123'] = data_df['eff_sys_1'] * data_df['eff_sys_123_2'] * data_df['eff_sys_3']
data_df['subdetector_123_123_corr'] = data_df['subdetector_123_123'] / data_df['subdetector_123_eff_123']

data_df['subdetector_123_eff_13'] = data_df['eff_sys_1'] * ( 1 - data_df['eff_sys_123_2'] ) * data_df['eff_sys_3']
data_df['subdetector_123_13_corr'] = data_df['subdetector_123_13'] / data_df['subdetector_123_eff_13']

data_df['subdetector_123_eff_summed'] = data_df['subdetector_123_eff_123'] + data_df['subdetector_123_eff_13']
data_df['subdetector_123_summed_corr'] = ( data_df['subdetector_123_123'] + data_df['subdetector_123_13'] ) / data_df['subdetector_123_eff_summed']

# group_cols = [ 'subdetector_123_summed_corr', 'subdetector_123_123_corr' , 'subdetector_123_13_corr']
# # plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

group_cols = [ 'eff_sys_123_2', 'eff_sys_2' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')


# -------------------------------------------------------------------------------------------------------
# Subdetector 234 ---------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

data_df['subdetector_234_234'] = data_df['processed_tt_1234'] + data_df['processed_tt_234']
data_df['subdetector_234_23'] = data_df['processed_tt_23']  + data_df['processed_tt_123']
data_df['subdetector_234_34'] = data_df['processed_tt_34'] + data_df['processed_tt_134']
data_df['subdetector_234_24'] = data_df['processed_tt_24'] + data_df['processed_tt_124']

# Plane 3
A = data_df['subdetector_234_234']
B = data_df['subdetector_234_24']
data_df['eff_sys_234_3'] = A / ( A + B )

# Plane 2
A = data_df['subdetector_234_234']
B = data_df['subdetector_234_34']
data_df['eff_sys_234_2'] = A / ( A + B )

# Plane 4
A = data_df['subdetector_234_234']
B = data_df['subdetector_234_23']
data_df['eff_sys_234_4'] = A / ( A + B )


# Newly calculated eff --------------------------------------------------------------------------------------
data_df['subdetector_234_eff_234'] = data_df['eff_sys_2'] * data_df['eff_sys_234_3'] * data_df['eff_sys_4']
data_df['subdetector_234_234_corr'] = data_df['subdetector_234_234'] / data_df['subdetector_234_eff_234']

data_df['subdetector_234_eff_24'] = data_df['eff_sys_2'] * ( 1 - data_df['eff_sys_234_3'] ) * data_df['eff_sys_4']
data_df['subdetector_234_24_corr'] = data_df['subdetector_234_24'] / data_df['subdetector_234_eff_24']

data_df['subdetector_234_eff_summed'] = data_df['subdetector_234_eff_234'] + data_df['subdetector_234_eff_24']
data_df['subdetector_234_summed_corr'] = ( data_df['subdetector_234_234'] + data_df['subdetector_234_24'] ) / data_df['subdetector_234_eff_summed']

# group_cols = [ 'subdetector_234_summed_corr', 'subdetector_234_234_corr' , 'subdetector_234_24_corr']
# # plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

group_cols = [ 'eff_sys_234_3', 'eff_sys_3' ]
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')






def plot_eff_vs_rate(data_df, eff_col, rate_col, label_suffix):
    """
    Plot detector efficiency vs original and corrected rate, show correlation coefficients and fit lines.

    Parameters:
    - data_df: DataFrame containing the data
    - eff_col: column name with efficiency values
    - rate_col: original raw rate column
    - corrected_col: corrected rate column
    - label_suffix: string to append to legend labels
    """
    
    global create_plots, fig_idx, show_plots, save_plots, figure_path
    
    if create_plots:

        # Drop NaNs
        valid = data_df[[eff_col, rate_col]].dropna()
        x = valid[eff_col].values
        y_orig = valid[rate_col].values

        # Compute Pearson correlations
        corr_orig, _ = pearsonr(x, y_orig)

        # Linear fits
        p_orig = np.polyfit(x, y_orig, 1)
        x_fit = np.linspace(x.min(), x.max(), 500)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(x, y_orig, alpha=0.7, label=f'{label_suffix}', s=2)
        ax.plot(x_fit, np.polyval(p_orig, x_fit), linestyle='--', linewidth=1.0, label='Fit')
        
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        
        # Plot a x = y line
        plt.plot(x_fit, x_fit, color='gray', linestyle=':', linewidth=1.0, label='y = x')
        
        plt.xlabel('Efficiency')
        plt.ylabel('Rate')
        plt.title(f'Efficiency calculated with three and two planes {label_suffix}')
        plt.grid(True)
        
        # Axes equal
        plt.axis('equal')

        textstr = f'Corr (original): {corr_orig:.3f}'
        plt.gcf().text(0.15, 0.80, textstr, fontsize=10, bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))

        plt.legend()
        plt.tight_layout()

        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + f"_eff_scatter_new_{label_suffix}.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)
        plt.close()
    else:
        print("Plotting is disabled. Set `create_plots = True` to enable plotting.")


# group_cols = [ 'eff_sys_123_2', 'eff_sys_2' ]
# plot_eff_vs_rate(data_df, 'eff_sys_123_1', 'eff_sys_1', label_suffix='1_123_1234')

# plot_eff_vs_rate(data_df, 'eff_sys_123_2', 'eff_sys_2', label_suffix='2_123_1234')
# plot_eff_vs_rate(data_df, 'eff_sys_123_2', 'eff_sys_234_2', label_suffix='2_123_234')
# plot_eff_vs_rate(data_df, 'eff_sys_234_2', 'eff_sys_2', label_suffix='2_234_1234')

# plot_eff_vs_rate(data_df, 'eff_sys_123_3', 'eff_sys_3', label_suffix='3_123_1234')
# plot_eff_vs_rate(data_df, 'eff_sys_123_3', 'eff_sys_234_3', label_suffix='3_123_234')
# plot_eff_vs_rate(data_df, 'eff_sys_234_3', 'eff_sys_3', label_suffix='3_234_1234')

# plot_eff_vs_rate(data_df, 'eff_sys_234_4', 'eff_sys_4', label_suffix='4_234_1234')


# -----------------------------------------------------------------------------------------
# Applying acceptance factors -------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# import numpy as np
# import pandas as pd
# from itertools import combinations
# from scipy.optimize import minimize

# # Ensure column names exist
# columns_required = [
#     'eff_sys_123_1', 'eff_sys_1',
#     'eff_sys_123_2', 'eff_sys_2', 'eff_sys_234_2',
#     'eff_sys_123_3', 'eff_sys_3', 'eff_sys_234_3',
#     'eff_sys_234_4', 'eff_sys_4'
# ]

# missing_columns = [col for col in columns_required if col not in data_df.columns]
# if missing_columns:
#     raise ValueError(f"Missing columns in data_df: {missing_columns}")

# # Define the system of column pairs to match via linear transform
# eff_pairs = [
#     ('eff_sys_123_1', 'eff_sys_1'),
#     ('eff_sys_123_2', 'eff_sys_2'),
#     ('eff_sys_123_2', 'eff_sys_234_2'),
#     ('eff_sys_234_2', 'eff_sys_2'),
#     ('eff_sys_123_3', 'eff_sys_3'),
#     ('eff_sys_123_3', 'eff_sys_234_3'),
#     ('eff_sys_234_3', 'eff_sys_3'),
#     ('eff_sys_234_4', 'eff_sys_4'),
# ]

# # Unique columns to transform
# unique_eff_cols = sorted(set([col for pair in eff_pairs for col in pair]))

# # Objective: sum of squared errors between transformed pairs
# def objective(params):
#     transformed = {}
#     for i, col in enumerate(unique_eff_cols):
#         a, b = params[2*i], params[2*i+1]
#         transformed[col] = a + b * data_df[col].values
#         # transformed[col] = b * data_df[col].values

#     total_error = 0.0
#     for col1, col2 in eff_pairs:
#         y1 = transformed[col1]
#         y2 = transformed[col2]
#         valid = ~np.isnan(y1) & ~np.isnan(y2)
#         delta = y1[valid] - y2[valid]
#         error = np.where(np.abs(delta) < 0.02, 0.5 * delta**2, 0.02 * (np.abs(delta) - 0.01))  # Huber loss
#         total_error += np.sum(error)
#     return total_error


# # Constraints to enforce 0.6 <= a + b * x <= 1.0
# constraints = []
# for i, col in enumerate(unique_eff_cols):
#     x = data_df[col].dropna()
#     if len(x) == 0:
#         continue
#     xmin, xmax = x.min(), x.max()
#     a_idx, b_idx = 2*i, 2*i+1
#     # Lower bound: a + b * xmin >= 0.6
#     constraints.append({'type': 'ineq', 'fun': lambda p, ai=a_idx, bi=b_idx, xval=xmin: (p[ai] + p[bi] * xval) - 0.80})
#     # Upper bound: a + b * xmax <= 1.0
#     constraints.append({'type': 'ineq', 'fun': lambda p, ai=a_idx, bi=b_idx, xval=xmax: 1.0 - (p[ai] + p[bi] * xval)})

# # Initial guess
# initial_params = np.array([0.0, 1.0] * len(unique_eff_cols))


# from scipy.optimize import Bounds

# bounds = []
# for col in unique_eff_cols:
#     bounds.append((-10, 10))  # a
#     bounds.append((0.001, 20.0))   # b

# bounds = Bounds(*zip(*bounds))  # Convert to scipy Bounds format


# # Run optimizer
# result = minimize(
#     objective,
#     initial_params,
#     constraints=constraints,
#     bounds=bounds,
#     method='trust-constr',
#     options={'verbose': 1, 'gtol': 1e-8, 'xtol': 1e-8, 'maxiter': 3000}
# )

# # Apply transformation if successful
# if result.success:
#     for i, col in enumerate(unique_eff_cols):
#         a, b = result.x[2*i], result.x[2*i+1]
#         data_df[f'{col}_corr'] = a + b * data_df[col]
#         # data_df[f'{col}_corr'] = b * data_df[col]


# print("\n=== Correction Parameters Matrix ===")
# print(f"{'Column':<25} {'a (offset)':>12} {'b (scale)':>12}")
# print("-" * 50)

# for i, col in enumerate(unique_eff_cols):
#     a, b = result.x[2*i], result.x[2*i + 1]
#     print(f"{col:<25} {a:12.6f} {b:12.6f}")


# plot_eff_vs_rate(data_df, 'eff_sys_123_1_corr', 'eff_sys_1_corr', label_suffix='new_1_123_1234')

# plot_eff_vs_rate(data_df, 'eff_sys_123_2_corr', 'eff_sys_2_corr', label_suffix='new_2_123_1234')
# plot_eff_vs_rate(data_df, 'eff_sys_123_2_corr', 'eff_sys_234_2_corr', label_suffix='new_2_123_234')
# plot_eff_vs_rate(data_df, 'eff_sys_234_2_corr', 'eff_sys_2_corr', label_suffix='new_2_234_1234')

# plot_eff_vs_rate(data_df, 'eff_sys_123_3_corr', 'eff_sys_3_corr', label_suffix='new_3_123_1234')
# plot_eff_vs_rate(data_df, 'eff_sys_123_3_corr', 'eff_sys_234_3_corr', label_suffix='new_3_123_234')
# plot_eff_vs_rate(data_df, 'eff_sys_234_3_corr', 'eff_sys_3_corr', label_suffix='new_3_234_1234')

# plot_eff_vs_rate(data_df, 'eff_sys_234_4_corr', 'eff_sys_4_corr', label_suffix='new_4_234_1234')


# group_cols = [
#     ['eff_sys_123_1_corr', 'eff_sys_1_corr'],
#     ['eff_sys_123_2_corr', 'eff_sys_2_corr', 'eff_sys_234_2_corr', 'eff_sys_2', 'eff_sys_123_2'],
#     ['eff_sys_123_3_corr', 'eff_sys_3_corr', 'eff_sys_234_3_corr', 'eff_sys_3', 'eff_sys_234_3'],
#     ['eff_sys_234_4_corr', 'eff_sys_4_corr']
# ]

# # plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_grouped_series(data_df, group_cols, title='Corrected efficiencies')


# a = 1/0


# group_cols = [
#     ['eff_sys_1'],
#     ['eff_sys_2', 'eff_sys_123_2'],
#     ['eff_sys_3', 'eff_sys_234_3'],
#     ['eff_sys_4'],
# ]


group_cols = [
    ['eff_sys_123_1', 'eff_sys_1'],
    ['eff_sys_123_2', 'eff_sys_2', 'eff_sys_234_2', 'eff_sys_2', 'eff_sys_123_2'],
    ['eff_sys_123_3', 'eff_sys_3', 'eff_sys_234_3', 'eff_sys_3', 'eff_sys_234_3'],
    ['eff_sys_234_4', 'eff_sys_4']
]

# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_grouped_series(data_df, group_cols, title='Corrected efficiencies')



data_df['eff_1'] = ( data_df['eff_sys_1'] + data_df['eff_sys_123_1'] ) / 2
data_df['eff_2'] = ( data_df['eff_sys_2'] + data_df['eff_sys_123_2'] ) / 2
data_df['eff_3'] = ( data_df['eff_sys_3'] + data_df['eff_sys_234_3'] ) / 2
data_df['eff_4'] = ( data_df['eff_sys_4'] + data_df['eff_sys_234_4'] ) / 2

acceptance_corr = True
if acceptance_corr:
    data_df['final_eff_1'] = data_df['eff_1'] / data_df['acc_1']
    data_df['final_eff_2'] = data_df['eff_2'] / data_df['acc_2']
    data_df['final_eff_3'] = data_df['eff_3'] / data_df['acc_3']
    data_df['final_eff_4'] = data_df['eff_4'] / data_df['acc_4']
else:
    data_df['final_eff_1'] = data_df['eff_1']
    data_df['final_eff_2'] = data_df['eff_2']
    data_df['final_eff_3'] = data_df['eff_3']
    data_df['final_eff_4'] = data_df['eff_4']



data_df['final_eff_pre_roll_1'] = data_df['final_eff_1']
data_df['final_eff_pre_roll_2'] = data_df['final_eff_2']
data_df['final_eff_pre_roll_3'] = data_df['final_eff_3']
data_df['final_eff_pre_roll_4'] = data_df['final_eff_4']

# a = 1/0

print("Rolling efficiencies...")

rolling_effs = True
if rolling_effs:
    cols_to_interpolate = ['final_eff_1', 'final_eff_2', 'final_eff_3', 'final_eff_4']

    # Step 1: Identify rows where interpolation will be applied
    interpolated_mask = data_df[cols_to_interpolate].replace(0, np.nan).isna()

    # Step 2: Perform interpolation
    data_df[cols_to_interpolate] = data_df[cols_to_interpolate].replace(0, np.nan).interpolate(method='linear')

    # Step 3: Apply rolling filters
    mean_window, med_window = 5, 1
    rolling_mean, rolling_median = True, True

    if rolling_median:
        for col in cols_to_interpolate:
            data_df[col] = medfilt(data_df[col], kernel_size=med_window)

    if rolling_mean:
        for col in cols_to_interpolate:
            data_df[col] = data_df[col].rolling(window=mean_window, center=True, min_periods=1).mean()

    # Step 4: Set previously interpolated positions back to NaN
    for col in cols_to_interpolate:
        data_df.loc[interpolated_mask[col], col] = np.nan

    group_cols = [ ['final_eff_pre_roll_1', 'final_eff_pre_roll_2', 'final_eff_pre_roll_3', 'final_eff_pre_roll_4'],
                ['final_eff_1', 'final_eff_2', 'final_eff_3', 'final_eff_4'] ]
    plot_grouped_series(data_df, group_cols, title='Final calculated efficiencies, rolling')

else:
    group_cols = [ ['final_eff_1', 'final_eff_2', 'final_eff_3', 'final_eff_4'] ]
    plot_grouped_series(data_df, group_cols, title='Final calculated efficiencies')


def plot_eff_vs_rate(data_df, eff_col, rate_col, corrected_col, label_suffix=''):
    global create_plots, fig_idx, show_plots, save_plots, figure_path
    
    if create_plots:
        valid = data_df[[eff_col, rate_col, corrected_col]].dropna()
        x = valid[eff_col].values
        y_orig = valid[rate_col].values
        y_corr = valid[corrected_col].values
        # Compute Pearson correlations
        corr_orig, _ = pearsonr(x, y_orig)
        corr_corr, _ = pearsonr(x, y_corr)
        # Linear fits
        p_orig = np.polyfit(x, y_orig, 1)
        p_corr = np.polyfit(x, y_corr, 1)
        x_fit = np.linspace(x.min(), x.max(), 500)
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y_orig, alpha=0.7, label=f'Original rate {label_suffix}', s=2)
        plt.scatter(x, y_corr, alpha=0.7, label=f'Corrected rate {label_suffix}', s=2)
        plt.plot(x_fit, np.polyval(p_orig, x_fit), linestyle='--', linewidth=1.0, label='Fit: Original rate')
        plt.plot(x_fit, np.polyval(p_corr, x_fit), linestyle='--', linewidth=1.0, label='Fit: Corrected rate')
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        plt.xlabel('Efficiency')
        plt.ylabel('Rate')
        plt.title(f'Efficiency vs. Rate {label_suffix}')
        plt.grid(True)
        textstr = f'Corr (original): {corr_orig:.3f}\nCorr (corrected): {corr_corr:.3f}'
        plt.gcf().text(0.15, 0.80, textstr, fontsize=10, bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
        plt.legend()
        plt.tight_layout()
        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + "_scatter.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)
        plt.close()
    else:
        print("Plotting is disabled. Set `create_plots = True` to enable plotting.")


print('----------------------------------------------------------------------')
print('----------------- Following the subdetectors idea --------------------')
print('----------------------------------------------------------------------')

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

# Now correcting by efficiency
data_df['detector_1234_eff_corr'] = data_df['detector_1234'] / data_df['detector_1234_eff']
data_df['detector_123_eff_corr'] = data_df['detector_123'] / data_df['detector_123_eff']
data_df['detector_234_eff_corr'] = data_df['detector_234'] / data_df['detector_234_eff']
data_df['detector_12_eff_corr'] = data_df['detector_12'] / data_df['detector_12_eff']
data_df['detector_23_eff_corr'] = data_df['detector_23'] / data_df['detector_23_eff']
data_df['detector_34_eff_corr'] = data_df['detector_34'] / data_df['detector_34_eff']


group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['detector_1234', 'detector_1234_eff_corr'],
    ['detector_123', 'detector_123_eff_corr'],
    ['detector_234', 'detector_234_eff_corr'],
    ['detector_12', 'detector_12_eff_corr'],
    ['detector_23', 'detector_23_eff_corr'],
    ['detector_34', 'detector_34_eff_corr'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


# group_cols = [
#     ['sensors_ext_Pressure_ext'],
#     ['sensors_ext_Temperature_ext'],
#     ['detector_1234_eff'],
#     ['detector_123_eff'],
#     ['detector_234_eff'],
#     ['detector_12_eff'],
#     ['detector_23_eff'],
#     ['detector_34_eff'],
# ]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


detector_labels = ['1234', '123', '234', '12', '23', '34']
for label in detector_labels:
    eff_col = f'detector_{label}_eff'
    rate_col = f'detector_{label}'
    corrected_col = f'detector_{label}_eff_corr'

    # plot_eff_vs_rate(
    #     data_df,
    #     eff_col=eff_col,
    #     rate_col=rate_col,
    #     corrected_col=corrected_col,
    #     label_suffix=label,
    # )


# ------------------------------------------------------------------------------------------------------------------------


# CODE TO RECALCULATE EFFICIENCIES TO MINIMIZE THE CORRELATION OF THE CORR RATES WITH THE COMBINED EFFS

# My goal is to eliminate the correlation between rate_caseX and eff_caseX. This will help us define conditions for e1, e2, e3, and e4 that are used to calculate eff_caseX.

# First, I need to determine an affine transformation for rate_caseX based on eff_caseX that removes this correlation while preserving the mean of rate_caseX. This transformation should also allow for a slightly different correction to be applied for each eff_caseX value, accounting for detector_X_eff_corr through a linear function as well.

# Once this transformation is defined, I want to incorporate all its operations into the calculation of eff_caseX. This should lead to a system of equations that defines the necessary function of e1, e2, e3, and e4 required to minimize the correlation between the transformed rate_caseX and the new eff_caseX.


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def fit_and_plot_eff_vs_rate(df, eff_col, rate_col, label_suffix):
    global create_plots, fig_idx, show_plots, save_plots, figure_path
    
    if create_plots:
        # Drop rows with NaNs in either column
        valid = df[[eff_col, rate_col]].dropna()
        x = valid[[eff_col]].values  # 2D
        y = valid[rate_col].values   # 1D

        # Fit linear model
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        a, b = model.coef_[0], model.intercept_
        r2 = model.score(x, y)
        
        y_flat = y - y_pred + y.mean()

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 6), gridspec_kw={'hspace': 0.15})

        # Top: original
        ax1.scatter(x, y, s=1, label='Data', alpha=0.7)
        ax1.plot(x, y_pred, color='red', linewidth=0.5, label=f'Fit: y = {a:.3f}x + {b:.3f}\n$R^2$ = {r2:.3f}')
        ax1.axhline(y.mean(), color='gray', linestyle='--', linewidth=0.5, label='Mean rate')
        ax1.set_ylabel(rate_col)
        ax1.set_title(f'Detector {label_suffix} — Original')
        ax1.legend(fontsize=8)
        ax1.grid(True)

        # Bottom: decorrelated
        ax2.scatter(x, y_flat, s=1, color='tab:orange', alpha=0.7, label='Decorrelated')
        ax2.axhline(y.mean(), color='gray', linestyle='--', linewidth=0.5, label='Mean rate')
        ax2.set_xlabel(eff_col)
        ax2.set_ylabel(rate_col)
        ax2.set_title(f'Detector {label_suffix} — Slope Subtracted')
        ax2.legend(fontsize=8)
        ax2.grid(True)

        plt.tight_layout()
        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + "_eff_linear.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)
        plt.close()

        # Print fitted parameters
        print(f"Detector {label_suffix}:")
        print(f"  a (slope)     = {a:.6f}")
        print(f"  b (intercept) = {b:.6f}")
        print(f"  R²            = {r2:.6f}")
        print("-" * 50)
    else:
        print("Plotting is disabled. Set `create_plots = True` to enable plotting.")

# Apply to all detectors
for label in detector_labels:
    eff_col = f'detector_{label}_eff'
    rate_col = f'detector_{label}_eff_corr'
    # fit_and_plot_eff_vs_rate(data_df, eff_col, rate_col, label)


# ---------------------------------------------------------------------------------------------

from scipy.optimize import minimize
import numpy as np

def decorrelate_efficiency_least_change(eff, rate_corr, bounds=(0.01, 1.5)):
    eff = np.asarray(eff)
    rate_corr = np.asarray(rate_corr)
    counts = eff * rate_corr  # fixed

    def objective(eff_prime):
        return np.sum((eff_prime - eff)**2)

    def constraint(eff_prime):
        rate_prime = counts / eff_prime
        r_mean = np.mean(rate_prime)
        e_mean = np.mean(eff_prime)
        return np.sum((rate_prime - r_mean) * (eff_prime - e_mean))  # covariance

    cons = {'type': 'eq', 'fun': constraint}
    bounds_list = [bounds] * len(eff)
    res = minimize(objective, eff, method='SLSQP', bounds=bounds_list, constraints=cons)

    return res.x, res



detector_labels = ['1234', '123', '234', '12', '23', '34']

for label in detector_labels:
    eff_col = f'detector_{label}_eff'
    rate_col = f'detector_{label}_eff_corr'

    df_valid = data_df[[eff_col, rate_col]].dropna()
    eff = df_valid[eff_col].values
    rate_corr = df_valid[rate_col].values

    eff_new, res = decorrelate_efficiency_least_change(eff, rate_corr)
    eff_prime_col = f'{eff_col}_decorrelated'

    # Store result in the full dataframe
    data_df.loc[df_valid.index, eff_prime_col] = eff_new

    # Diagnostics
    r_new = (rate_corr * eff) / eff_new  # back-compute N_i / eff'_i
    cov_post = np.cov(r_new, eff_new)[0, 1]
    print(f"[{label}] Final covariance after correction: {cov_post:.6e}")


for label in detector_labels:
    eff_prime_col = f'detector_{label}_eff_decorrelated'
    rate_col = f'detector_{label}_eff_corr'

    valid = data_df[[eff_prime_col, rate_col]].dropna()
    
    if create_plots:
        plt.figure(figsize=(5, 4))
        plt.scatter(valid[eff_prime_col], valid[rate_col], s=2, alpha=0.7)
        plt.xlabel(f"{eff_prime_col}")
        plt.ylabel(f"{rate_col}")
        plt.title(f"Decorrelated: {label}")
        plt.axhline(valid[rate_col].mean(), linestyle='--', color='gray', linewidth=0.5)
        plt.grid(True)
        plt.tight_layout()
        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + "_decorrelated.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)
        plt.close()
    else:
        print(f"Plotting is disabled for {label}. Set `create_plots = True` to enable plotting.")


group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['detector_1234_eff_decorrelated', 'detector_1234_eff'],
    ['detector_123_eff_decorrelated', 'detector_123_eff'],
    ['detector_234_eff_decorrelated', 'detector_234_eff'],
    ['detector_12_eff_decorrelated', 'detector_12_eff'],
    ['detector_23_eff_decorrelated', 'detector_23_eff'],
    ['detector_34_eff_decorrelated', 'detector_34_eff'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


# --------------------------------------------------------------------------
# Calculation of new efficiencies ------------------------------------------
# --------------------------------------------------------------------------

def eff_model(e, label):
    e1, e2, e3, e4 = e
    if label == '1234':
        return (
            e1 * e2 * e3 * e4 +
            e1 * (1 - e2) * e3 * e4 +
            e1 * e2 * (1 - e3) * e4 +
            e1 * (1 - e2) * (1 - e3) * e4
        )
    elif label == '123':
        return e1 * e2 * e3 + e1 * (1 - e2) * e3
    elif label == '234':
        return e2 * e3 * e4 + e2 * (1 - e3) * e4
    elif label == '12':
        return e1 * e2
    elif label == '23':
        return e2 * e3
    elif label == '34':
        return e3 * e4
    else:
        raise ValueError(f"Unknown label: {label}")

def residuals(e, eff_targets):
    return [eff_model(e, label) - eff_targets[label] for label in eff_targets]


from scipy.optimize import least_squares

def solve_eff_components_per_row(df):
    labels = ['1234', '123', '234', '12', '23', '34']

    e1_list, e2_list, e3_list, e4_list = [], [], [], []

    for i, row in df.iterrows():
        try:
            eff_targets = {label: row[f'detector_{label}_eff_decorrelated'] for label in labels}
            if any(np.isnan(list(eff_targets.values()))):
                e1_list.append(np.nan)
                e2_list.append(np.nan)
                e3_list.append(np.nan)
                e4_list.append(np.nan)
                continue

            x0 = [0.8, 0.8, 0.8, 0.8]
            res = least_squares(residuals, x0, bounds=(0, 1), args=(eff_targets,))
            e1, e2, e3, e4 = res.x
        except Exception as e:
            print(f"Row {i}: failed with error {e}")
            e1, e2, e3, e4 = [np.nan] * 4

        e1_list.append(e1)
        e2_list.append(e2)
        e3_list.append(e3)
        e4_list.append(e4)

    df['final_eff_1_decorrelated'] = e1_list
    df['final_eff_2_decorrelated'] = e2_list
    df['final_eff_3_decorrelated'] = e3_list
    df['final_eff_4_decorrelated'] = e4_list


solve_eff_components_per_row(data_df)

e1 = data_df['final_eff_1_decorrelated']
e2 = data_df['final_eff_2_decorrelated']
e3 = data_df['final_eff_3_decorrelated']
e4 = data_df['final_eff_4_decorrelated']

group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['final_eff_1', 'final_eff_1_decorrelated'],
    ['final_eff_2', 'final_eff_2_decorrelated'],
    ['final_eff_3', 'final_eff_3_decorrelated'],
    ['final_eff_4', 'final_eff_4_decorrelated'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


# ------------------------------------------------------------------------------------------------------------------------

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

# Now correcting by efficiency
data_df['detector_1234_eff_corr'] = data_df['detector_1234'] / data_df['detector_1234_eff']
data_df['detector_123_eff_corr'] = data_df['detector_123'] / data_df['detector_123_eff']
data_df['detector_234_eff_corr'] = data_df['detector_234'] / data_df['detector_234_eff']
data_df['detector_12_eff_corr'] = data_df['detector_12'] / data_df['detector_12_eff']
data_df['detector_23_eff_corr'] = data_df['detector_23'] / data_df['detector_23_eff']
data_df['detector_34_eff_corr'] = data_df['detector_34'] / data_df['detector_34_eff']


group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['detector_1234', 'detector_1234_eff_corr'],
    ['detector_123', 'detector_123_eff_corr'],
    ['detector_234', 'detector_234_eff_corr'],
    ['detector_12', 'detector_12_eff_corr'],
    ['detector_23', 'detector_23_eff_corr'],
    ['detector_34', 'detector_34_eff_corr'],
]
plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


# group_cols = [
#     ['sensors_ext_Pressure_ext'],
#     ['sensors_ext_Temperature_ext'],
#     ['detector_1234_eff'],
#     ['detector_123_eff'],
#     ['detector_234_eff'],
#     ['detector_12_eff'],
#     ['detector_23_eff'],
#     ['detector_34_eff'],
# ]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


detector_labels = ['1234', '123', '234', '12', '23', '34']
for label in detector_labels:
    eff_col = f'detector_{label}_eff'
    rate_col = f'detector_{label}'
    corrected_col = f'detector_{label}_eff_corr'

    plot_eff_vs_rate(
        data_df,
        eff_col=eff_col,
        rate_col=rate_col,
        corrected_col=corrected_col,
        label_suffix=label,
    )


data_df['definitive_eff_1'] = data_df['final_eff_1_decorrelated']
data_df['definitive_eff_2'] = data_df['final_eff_2_decorrelated']
data_df['definitive_eff_3'] = data_df['final_eff_3_decorrelated']
data_df['definitive_eff_4'] = data_df['final_eff_4_decorrelated']


# ------------------------------------------------------------------------------------------------------------------------

data_df['definitive_eff'] = ( data_df['definitive_eff_1'] + data_df['definitive_eff_2'] + data_df['definitive_eff_3'] + data_df['definitive_eff_4'] ) / 4
data_df['unc_definitive_eff'] = 1

data_df['summed'] = data_df['detector_1234'] + data_df['detector_123'] + \
    data_df['detector_234'] + data_df['detector_12'] + \
    data_df['detector_23'] + data_df['detector_34']

data_df['summed_eff_corr'] = data_df['detector_1234_eff_corr'] + data_df['detector_123_eff_corr'] + \
    data_df['detector_234_eff_corr'] + data_df['detector_12_eff_corr'] + \
    data_df['detector_23_eff_corr'] + data_df['detector_34_eff_corr']

group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    [ 'summed', 'summed_eff_corr' ]
]
plot_grouped_series(data_df, group_cols, title='Summed Detector Signals and Environment')


# -------------------------------------------------------------------------------------------------------------------------
# Calculate the fit for the efficiencies ----------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import numpy as np

def fit_efficiency_model(x, y, z, model_type='linear'):
    X = np.column_stack((x, y))
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, z)
        coeffs = model.coef_, model.intercept_
        return lambda P, T: coeffs[0][0] * P + coeffs[0][1] * T + coeffs[1], coeffs
    elif model_type == 'sigmoid':
        def sigmoid(xy, a, b, c, d):
            P, T = xy
            return d / (1 + np.exp(-a * (P - b) - c * (T - b)))
        popt, _ = curve_fit(sigmoid, (x, y), z, maxfev=10000)
        return lambda P, T: sigmoid((P, T), *popt), popt
    else:
        raise NotImplementedError(f"Model {model_type} not implemented")

def assign_efficiency_fit(df, eff_col, fit_col, model_type='linear'):
    filtered = df.dropna(subset=[eff_col, 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    x = filtered['sensors_ext_Pressure_ext'].values
    y = filtered['sensors_ext_Temperature_ext'].values
    z = filtered[eff_col].values
    fit_func, coeffs = fit_efficiency_model(x, y, z, model_type=model_type)
    df[fit_col] = fit_func(df['sensors_ext_Pressure_ext'], df['sensors_ext_Temperature_ext'])
    df[f'unc_{fit_col}'] = 1
    return filtered, fit_func, coeffs


eff_fitting = True

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

def plot_combined_efficiency_views(filtered_df, final_eff_col, fit_func, plane_number):
    global create_plots, fig_idx, show_plots, save_plots, figure_path
    
    if create_plots:
        x = filtered_df['sensors_ext_Pressure_ext'].values
        y = filtered_df['sensors_ext_Temperature_ext'].values
        z = filtered_df[final_eff_col].values

        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = np.linspace(y.min(), y.max(), 200)

        fig = plt.figure(figsize=(16, 12))

        # --- 3D Surface Plot ---
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.scatter(x, y, z, color='blue', alpha=0.6, s=8, label='Measured')

        x_surf, y_surf = np.meshgrid(
            np.linspace(x.min(), x.max(), 50),
            np.linspace(y.min(), y.max(), 50)
        )
        z_surf = fit_func(x_surf, y_surf)
        ax1.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.2, edgecolor='k', linewidth=0.1)

        ax1.set_xlabel('Pressure [P]')
        ax1.set_ylabel('Temperature [T]')
        ax1.set_zlabel('Efficiency')
        ax1.set_zlim(0.8, 1)
        ax1.set_title(f'3D Fit: Plane {plane_number}')
        ax1.legend(handles=[
            Line2D([0], [0], marker='o', color='w', label='Measured', markerfacecolor='blue', markersize=6),
            Patch(facecolor='red', edgecolor='k', label='Fitted Surface', alpha=0.3)
        ])

        # --- Eff vs Pressure ---
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(x, z, alpha=0.4, label='Measured')
        ax2.plot(x_fit, fit_func(x_fit, np.mean(y)), 'r-', label='Fit at avg T')
        ax2.set_xlabel('Pressure')
        ax2.set_ylabel('Efficiency')
        ax2.set_ylim(0.8, 1)
        ax2.set_title('Projection: Efficiency vs Pressure')
        ax2.legend()

        # --- Eff vs Temperature ---
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(y, z, alpha=0.4, label='Measured')
        ax3.plot(y_fit, fit_func(np.mean(x), y_fit), 'r-', label='Fit at avg P')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Efficiency')
        ax3.set_ylim(0.8, 1)
        ax3.set_title('Projection: Efficiency vs Temperature')
        ax3.legend()

        # --- Efficiency heatmap slice (optional projection plane) ---
        ax4 = fig.add_subplot(2, 2, 4)
        sc = ax4.scatter(x, y, c=z, cmap='viridis', s=10)
        ax4.set_xlabel('Pressure')
        ax4.set_ylabel('Temperature')
        ax4.set_title('Efficiency Color Map')
        plt.colorbar(sc, ax=ax4, label='Efficiency')

        plt.suptitle(f'Efficiency Fitting Overview – Plane {plane_number}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = f"{figure_path}{fig_idx}_overview.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)
        plt.close()
    else:
        print("Plotting is disabled. Set `create_plots = True` to enable plotting.")

if eff_fitting:
    for i in range(1, 5):
        eff_col = f'final_eff_{i}'
        fit_col = f'eff_fit_{i}'
        filtered_df, fit_func, _ = assign_efficiency_fit(
            data_df, eff_col, fit_col, model_type='linear'  # Or 'sigmoid'
        )
        plot_combined_efficiency_views(filtered_df, eff_col, fit_func, i)


if create_plots:
    print("Creating efficiency comparison scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(1, 5):  # Modules 1 to 4
        ax.scatter(
            data_df[f'eff_fit_{i}'],
            data_df[f'final_eff_{i}'],
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
    # Set equal axes
    ax.set_aspect('equal', adjustable='box')
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


data_df[f'unc_definitive_eff_1'] = 1
data_df[f'unc_definitive_eff_2'] = 1
data_df[f'unc_definitive_eff_3'] = 1
data_df[f'unc_definitive_eff_4'] = 1

if create_plots:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(17, 14), sharex=True)
    for i in range(1, 5):  # Loop from 1 to 4
        ax = axes[i-1]  # pick the appropriate subplot
        
        ax.plot(data_df['Time'], data_df[f'definitive_eff_{i}'], 
                label=f'Final Eff. {i}', color=f'C{i + 8}', alpha=1)
        ax.fill_between(data_df['Time'],
                        data_df[f'definitive_eff_{i}'] - data_df[f'unc_definitive_eff_{i}'],
                        data_df[f'definitive_eff_{i}'] + data_df[f'unc_definitive_eff_{i}'],
                        alpha=0.2, color=f'C{i}')
        
        ax.plot(data_df['Time'], data_df[f'eff_fit_{i}'], 
                label=f'Eff. {i} Fit', color=f'C{i + 12}', alpha=1)
        
        # Labeling and titles
        ax.set_ylabel('Efficiency')
        ax.set_ylim(0.8, 1.0)
        ax.grid(True)
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
else:
    print("Plotting is disabled. Set `create_plots = True` to enable plotting.")

print('Efficiency calculations performed.')


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
    global create_plots, fig_idx
    
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
        if create_plots:
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
        else:
            print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
        
        df = df[z_scores < z_score_th_pres_corr]
        
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

main_region = 'summed_eff_corr'
angular_regions = ['High', 'N', 'S', 'E', 'W']

trigger_types = ['detector_1234',
                 'detector_123',
                 'detector_234',
                 'detector_12',
                 'detector_23',
                 'detector_34']

main_region = ['summed']

eff_corr_regions = []
og_regions = trigger_types + main_region
for region in og_regions:
    region += '_eff_corr'
    eff_corr_regions.append(region)
regions_to_correct = eff_corr_regions


print(og_regions)

print(regions_to_correct)

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
        
        df = pd.DataFrame({
            'delta_P': delta_P,
            'log_I_over_I0': log_I_over_I0,
            'unc_delta_P': unc_delta_P,
            'unc_log_I_over_I0': unc_I_over_I0 / I_over_I0
        })
        
        if create_plots:
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
        else:
            print("Plotting is disabled. Set `create_plots = True` to enable plotting.")


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

if create_plots:
    # --- Plotting the vectors ---
    plt.figure(figsize=(12, 8))

    # Loop through all regions
    for region in log_delta_I_df['Region']:
        
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
else:
    print("Plotting is disabled. Set `create_plots = True` to enable plotting.")


# ---------------------------------------------------------------------------------------------------

# Filter regions that contain 'new_' to plot
# regions_to_plot = [region for region in log_delta_I_df['Region'] if 'new_' in region]

if create_plots:
    regions_to_plot = regions_to_correct
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
        new_figure_path = figure_path + f"{fig_idx}" + "_GIANT_PRESSURE_PLOT_TTs.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)
    plt.close()
else:
    print("Plotting is disabled. Set `create_plots = True` to enable plotting.")

# ---------------------------------------------------------------------------------------------------

# Add a new outlier filter to the pressure correction
after_press_z_score_th = 10

remove_outliers = True
if remove_outliers:
    print('Removing outliers and zero values...')
    def remove_outliers_and_zeroes(series, z_thresh=outlier_filter):
        global create_plots, fig_idx
        
        """
        Create a mask of rows that are outliers or have zero values.
        """
        # median = series.mean()
        median = series.median()
        std = series.std()
        # z_scores = abs((series - median) / std)
        z_scores = (series - median) / std
        
        if create_plots:
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
        else:
            print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
        
        # print(z_scores)
        # Create a mask for rows where z_scores > z_thresh or values are zero
        mask = (abs(z_scores) > z_thresh) | (series == 0)
        return mask

    # Initialize a mask of all False, meaning no rows are removed initially
    rows_to_remove = pd.Series(False, index=data_df.index)
    rows_to_remove = rows_to_remove | remove_outliers_and_zeroes(data_df[f'pres_{region}'], z_thresh = after_press_z_score_th)

    data_df_cleaned = data_df[~rows_to_remove].copy()
    data_df = data_df_cleaned.copy()


print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('--------------------- High order correction started ------------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

if high_order_correction:
    
    def calculate_coefficients(region, I0, delta_I):
        global create_plots, fig_idx
        
        delta_I_over_I0 = delta_I / I0

        # Fit linear regression model without intercept
        model = LinearRegression(fit_intercept=True)
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
            D = model.intercept_
            
            if create_plots:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

                scatter_kwargs = {'alpha': 0.5, 's': 10}
                line_kwargs = {'color': 'red', 'linewidth': 1.5}
                fontsize = 12

                # 1) ΔT_ground / T_ground_0
                ax = axes[0]
                x = df['delta_Tg_over_Tg0']
                y_data = df['delta_I_over_I0']
                x_line = np.linspace(x.min(), x.max(), 200)
                y_line = A * x_line + D

                ax.scatter(x, y_data, color='blue', label='Data', **scatter_kwargs)
                ax.plot(x_line, y_line, label=fr'Fit: $A$ = {A:.3f}', **line_kwargs)

                ax.set_xlabel(r'$\Delta T_{\mathrm{ground}} / T^{0}_{\mathrm{ground}}$', fontsize=fontsize)
                ax.set_ylabel(r'$\Delta I / I_0$', fontsize=fontsize)
                ax.set_title(f'Effect of Ground Temperature – {region}', fontsize=fontsize)
                ax.grid(True)
                ax.legend(fontsize=10)

                # 2) ΔT_100mbar / T_100mbar_0
                ax = axes[1]
                x = df['delta_Th_over_Th0']
                x_line = np.linspace(x.min(), x.max(), 200)
                y_line = B * x_line + D

                ax.scatter(x, y_data, color='green', label='Data', **scatter_kwargs)
                ax.plot(x_line, y_line, label=fr'Fit: $B$ = {B:.3f}', **line_kwargs)

                ax.set_xlabel(r'$\Delta T_{100\ \mathrm{mbar}} / T^{0}_{100\ \mathrm{mbar}}$', fontsize=fontsize)
                ax.set_title(f'Effect of 100 mbar Temp. – {region}', fontsize=fontsize)
                ax.grid(True)
                ax.legend(fontsize=10)

                # 3) Δh_100mbar / h_100mbar_0
                ax = axes[2]
                x = df['delta_H_over_H0']
                x_line = np.linspace(x.min(), x.max(), 200)
                y_line = C * x_line + D

                ax.scatter(x, y_data, color='purple', label='Data', **scatter_kwargs)
                ax.plot(x_line, y_line, label=fr'Fit: $C$ = {C:.3f}', **line_kwargs)

                ax.set_xlabel(r'$\Delta h_{100\ \mathrm{mbar}} / h^{0}_{100\ \mathrm{mbar}}$', fontsize=fontsize)
                ax.set_title(f'Effect of 100 mbar Height – {region}', fontsize=fontsize)
                ax.grid(True)
                ax.legend(fontsize=10)

                # Supertitle
                plt.suptitle(
                    fr'Normalized Correction Coefficients for {region}: '
                    fr'$A$ = {A:.5f}, $B$ = {B:.3f}, $C$ = {C:.3f}, $D$ = {D:.3f}',
                    fontsize=15,
                    y=1.05
                )

                plt.tight_layout()
                if show_plots:
                    plt.show()
                elif save_plots:
                    new_figure_path = figure_path + f"{fig_idx}" + f"_{region}_high_order.png"
                    fig_idx += 1
                    print(f"Saving figure to {new_figure_path}")
                    plt.savefig(new_figure_path, format='png', dpi=300)
                plt.close()
            else:
                print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
                
        else:
            A, B, C, D = np.nan, np.nan, np.nan  # Handle case where there are no valid data points
        return A, B, C, D
    
    
    for region in regions_to_correct:
    
        data_df[f'{region}_pressure_corrected'] = data_df[f'pres_{region}']
        
        # Use the pressure-corrected values directly
        # Calculate means for pressure and counts
        I0_count_corrected = data_df[f'{region}_pressure_corrected'].mean()
        Tg0 = data_df['temp_ground'].mean()
        Th0 = data_df['temp_100mbar'].mean()
        H0 = data_df['height_100mbar'].mean()
        
        # Calculate delta values using pressure-corrected values
        data_df['delta_I_count_corrected'] = data_df[f'{region}_pressure_corrected'] - I0_count_corrected

        # Calculate delta values
        data_df['delta_Tg'] = data_df['temp_ground'] - Tg0
        data_df['delta_Th'] = data_df['temp_100mbar'] - Th0
        data_df['delta_H'] = data_df['height_100mbar'] - H0

        # Normalize delta values
        data_df['delta_Tg_over_Tg0'] = data_df['delta_Tg'] / Tg0
        data_df['delta_Th_over_Th0'] = data_df['delta_Th'] / Th0
        data_df['delta_H_over_H0'] = data_df['delta_H'] / H0

        # Initialize a DataFrame to store the results
        high_order_results = pd.DataFrame(columns=['Region', 'A', 'B', 'C', 'D'])

        I0_region_corrected = data_df[f'{region}_pressure_corrected'].mean()
        data_df[f'delta_I_{region}_corrected'] = data_df[f'{region}_pressure_corrected'] - I0_region_corrected
        A, B, C, D = calculate_coefficients(region, I0_region_corrected, data_df[f'delta_I_{region}_corrected'])
        high_order_results = pd.concat([high_order_results, pd.DataFrame({'Region': [region], 'A': [A], 'B': [B], 'C': [C], 'D': [D]})], ignore_index=True)
        
        # Create corrected rate column for the region
        data_df[f'{region}_high_order_corrected'] = data_df[f'{region}_pressure_corrected'] * (1 - (A * data_df['delta_Tg'] / Tg0 + B * data_df['delta_Th'] / Th0 + C * data_df['delta_H'] / H0 + D))
    
    print("High order correction applied.")
else:
    print("High order correction not applied.")
    for region in regions_to_correct:
        data_df[f'{region}_high_order_corrected'] = data_df[f'pres_{region}']



print("Creating rate final plots...")
    
for region in og_regions:
    if create_plots:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 10), sharex=True)

        # Plot efficiencies
        ax1.plot(data_df['Time'], data_df['definitive_eff'], label='Final efficiency', color='C5')
        # ax1.plot(data_df['Time'], data_df['eff_new'], label='Original / Corrected with new system', color='C4')
        ax1.set_ylabel('Efficiency')
        ax1.set_ylim(0.8, 1)
        ax1.set_title('Efficiencies over Time')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2.plot(data_df['Time'], data_df[region], label='Uncorrected', color='C3')
        ax2.plot(data_df['Time'], data_df[f'{region}_eff_corr'], label='Eff. corr. rate', color='C8')
        ax2.plot(data_df['Time'], data_df[f'{region}_eff_corr_pressure_corrected'], label=f'Pressure corrected rate, {region}', color='C9')
        ax2.plot(data_df['Time'], data_df[f'{region}_eff_corr_high_order_corrected'], label=f'High order corrected rate, {region}', color='C10')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Rate')
        ax2.grid(True)
        # ax2.set_ylim(16, 18)
        ax2.set_title('Rates over Time')
        ax2.legend(loc='upper left')

        plt.tight_layout()

        # Save or show the plot
        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + f"_{region}_effs_rates_pressure.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)

        plt.close()



for region in og_regions:
    if create_plots:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 10), sharex=True)

        # Plot efficiencies
        ax1.plot(data_df['Time'], data_df['definitive_eff'], label='Final efficiency', color='C5')
        # ax1.plot(data_df['Time'], data_df['eff_new'], label='Original / Corrected with new system', color='C4')
        ax1.set_ylabel('Efficiency')
        ax1.set_ylim(0.8, 1)
        ax1.set_title('Efficiencies over Time')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        # Define the first half index
        half_idx = len(data_df) // 2

        # Compute normalization constants from the first half
        norm_uncorrected = data_df[region].iloc[:half_idx].mean()
        norm_eff_corr = data_df[f'{region}_eff_corr'].iloc[:half_idx].mean()
        norm_pressure = data_df[f'{region}_eff_corr_pressure_corrected'].iloc[:half_idx].mean()
        norm_high_order = data_df[f'{region}_eff_corr_high_order_corrected'].iloc[:half_idx].mean()

        # Plot normalized vectors
        ax2.plot(data_df['Time'], data_df[region] / norm_uncorrected - 1,
                label='Uncorrected', color='C3')

        ax2.plot(data_df['Time'], data_df[f'{region}_eff_corr'] / norm_eff_corr - 1,
                label='Eff. corr. norm. rate', color='C8')

        ax2.plot(data_df['Time'], data_df[f'{region}_eff_corr_pressure_corrected'] / norm_pressure - 1,
                label=f'Pressure corrected norm. rate, {region}', color='C9')

        ax2.plot(data_df['Time'], data_df[f'{region}_eff_corr_high_order_corrected'] / norm_high_order - 1,
         label=f'High order corrected norm. rate, {region}', color='C10')
        
        ax2.axhline(y=0, color='blue', linestyle='--', label='Baseline (0%)', alpha = 0.7)
        ax2.axhline(y=-0.05, color='green', linestyle='--', label='5% decrease', alpha = 0.7)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Normalized rate')
        ax2.grid(True)
        # ax2.set_ylim(16, 18)
        ax2.set_title('Normalized rate over Time')
        ax2.legend(loc='upper left')

        plt.tight_layout()

        # Save or show the plot
        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + f"_{region}_effs_rates_pressure_normalized.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)

        plt.close()


# trigger_types_corrected = ['detector_1234_eff_corr_high_order_corrected', 
#                            'detector_123_eff_corr_high_order_corrected', 
#                            'detector_234_eff_corr_high_order_corrected', 
#                            'detector_12_eff_corr_high_order_corrected',
#                            'detector_23_eff_corr_high_order_corrected',
#                            'detector_34_eff_corr_high_order_corrected']

trigger_types_corrected = ['detector_1234_eff_corr_pressure_corrected', 
                           'detector_123_eff_corr_pressure_corrected', 
                           'detector_234_eff_corr_pressure_corrected', 
                           'detector_12_eff_corr_pressure_corrected',
                           'detector_23_eff_corr_pressure_corrected',
                           'detector_34_eff_corr_pressure_corrected']

# Create list of column references
columns = [f'{region}' for region in trigger_types_corrected]

# Compute sum, mean, and median across selected columns
data_df['total_best_sum'] = data_df[columns].fillna(0).sum(axis=1).fillna(0)
data_df['total_best_mean'] = data_df[columns].fillna(0).mean(axis=1).fillna(0)
data_df['total_best_median'] = data_df[columns].fillna(0).median(axis=1).fillna(0)
data_df['unc_total_best_sum'] = 1

if create_plots:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 10), sharex=True)

    # Plot efficiencies
    ax1.plot(data_df['Time'], data_df['definitive_eff'], label='Final efficiency', color='C5')
    # ax1.plot(data_df['Time'], data_df['eff_new'], label='Original / Corrected with new system', color='C4')
    ax1.set_ylabel('Efficiency')
    ax1.set_ylim(0.8, 1)
    ax1.set_title('Efficiencies over Time')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    y = data_df[f'total_best_sum']
    cond = y > 0.1
    x = data_df['Time'][cond]
    y_plot = y[cond]
    ax2.plot(x, y_plot, label='Sum of the high order corrected rate', color='C1')
    # ax2.plot(data_df['Time'], data_df[f'total_best_mean'], label=f'Mean of the high order corrected rate', color='C10')
    # ax2.plot(data_df['Time'], data_df[f'total_best_median'], label=f'Median of the high order corrected rate', color='C11')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Rate')
    ax2.grid(True)
    # ax2.set_ylim(16, 18)
    ax2.set_title('Rates over Time')
    ax2.legend(loc='upper left')

    plt.tight_layout()

    # Save or show the plot
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + f"_{region}_effs_combined.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)

    plt.close()




if create_plots:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 10), sharex=True)

    # Plot efficiencies
    ax1.plot(data_df['Time'], data_df['definitive_eff'], label='Final efficiency', color='C5')
    # ax1.plot(data_df['Time'], data_df['eff_new'], label='Original / Corrected with new system', color='C4')
    ax1.set_ylabel('Efficiency')
    ax1.set_ylim(0.8, 1)
    ax1.set_title('Efficiencies over Time')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Define the first half index
    half_idx = len(data_df) // 2
    
    y_1 = data_df[f'total_best_sum']
    y_2 = data_df[f'total_best_mean']
    y_3 = data_df[f'total_best_median']
    cond = ( y_1 > 0.1 ) & ( y_2 > 0.1 ) & ( y_3 > 0.1 )
    x = data_df['Time'][cond]
    y1_plot = y_1[cond]
    y2_plot = y_2[cond]
    y3_plot = y_3[cond]
    
    # Compute normalization constants from the first half
    norm_total_best_sum = y1_plot.iloc[:half_idx].mean()
    ax2.plot(x, y1_plot / norm_total_best_sum - 1, label='total_best_sum', color='C3')
    
    norm_total_best_mean = y2_plot.iloc[:half_idx].mean()
    ax2.plot(x, y2_plot / norm_total_best_mean - 1, label='total_best_mean', color='C4')
    
    norm_total_best_median = y3_plot.iloc[:half_idx].mean()
    ax2.plot(x, y3_plot / norm_total_best_median - 1, label='total_best_median', color='C5')
    
    ax2.axhline(y=0, color='blue', linestyle='--', label='Baseline (0%)', alpha = 0.7)
    ax2.axhline(y=-0.05, color='green', linestyle='--', label='5% decrease', alpha = 0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Rate')
    ax2.grid(True)
    # ax2.set_ylim(16, 18)
    ax2.set_title('Rates over Time')
    ax2.legend(loc='upper left')

    plt.tight_layout()

    # Save or show the plot
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + f"_{region}_last_mean_median_sum.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)

    plt.close()



# -----------------------------------------------------------------------------
# Smoothing filters -----------------------------------------------------------
# -----------------------------------------------------------------------------

# Horizontal Median Filter ----------------------------------------------------

# Apply median filter to columns of interest
if HMF_ker > 0:
    print(f"Median filter applied with kernel size: {HMF_ker}, which are {HMF_ker * res_win_min} min")
    for region in regions_to_correct:
        data_df[f'pres_{region}'] = medfilt(data_df[f'pres_{region}'], kernel_size=HMF_ker)
else:
    print('Horizontal Median Filter not applied.')


# Moving Average Filter -------------------------------------------------------
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

# data_df[f'unc_sys_pres_{region}'] = np.sqrt( data_df[f'unc_pres_{region}']**2 + systematic_unc_corr_to_real_rate**2 )

# -----------------------------------------------------------------------------
# The end. Defining the total finally corrected rate
# -----------------------------------------------------------------------------
# data_df[f'totally_corrected_rate'] = data_df[f'{region}_high_order_corrected']
# data_df[f'unc_totally_corrected_rate'] = data_df[f'unc_sys_pres_{region}']


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Saving ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# If ANY value is 0, put it to NaN
data_df = data_df.replace(0, np.nan)

data_df.to_csv(save_filename, index=False)
print('Efficiency and atmospheric corrections completed and saved to corrected_table.csv.')


# -----------------------------------------------------------------------------
# Saving short table ----------------------------------------------------------
# -----------------------------------------------------------------------------

data_df['totally_corrected_rate'] = data_df[f'total_best_sum']
data_df['unc_totally_corrected_rate'] = data_df['unc_total_best_sum']
data_df['global_eff'] = data_df['definitive_eff']
data_df['unc_global_eff'] = data_df['unc_definitive_eff']

# Create a new DataFrame for Grafana
grafana_df = data_df[['Time', 'pressure_lab', 'totally_corrected_rate', 'unc_totally_corrected_rate', 'global_eff', 'unc_global_eff']].copy()

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