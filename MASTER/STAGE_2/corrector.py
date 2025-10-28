#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Created on Mon Jun 24 19:02:22 2024

@author: csoneira@ucm.es
"""

print("\n\n")
print("__| |________________________________________________________| |__")
print("__   ________________________________________________________   __")
print("  | |                                                        | |  ")
print("  | |                              _                         | |  ")
print("  | |  ___ ___  _ __ _ __ ___  ___| |_ ___  _ __ _ __  _   _ | |  ")
print("  | | / __/ _ \\| '__| '__/ _ \\/ __| __/ _ \\| '__| '_ \\| | | || |  ")
print("  | || (_| (_) | |  | | |  __/ (__| || (_) | |_ | |_) | |_| || |  ")
print("  | | \\___\\___/|_|  |_|  \\___|\\___|\\__\\___/|_(_)| .__/ \\__, || |  ")
print("  | |                                           |_|    |___/ | |  ")
print("__| |________________________________________________________| |__")
print("__   ________________________________________________________   __")
print("  | |                                                        | |  ")
print("\n\n")

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

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

# Standard Library
import os
import re
import sys
import math
import warnings
from datetime import datetime, timedelta

# Scientific Computing
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import (
    minimize,
    root,
    curve_fit,
    least_squares
)
from scipy.signal import medfilt
from scipy.stats import (
    norm,
    poisson,
    halfnorm,
    pearsonr
)

# Machine Learning
from sklearn.linear_model import LinearRegression

# Plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import warnings

from scipy.optimize import curve_fit

from scipy.stats import norm   # required for z-score threshold
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator     # SciPy ≥ 1.10


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Check if the script has an argument
run_jupyter_notebook = False
if run_jupyter_notebook:
    station = "2"
else:
    # Check if the script has an argument
    if len(sys.argv) < 2:
        print("Error: No station provided.")
        print("Usage: python3 script.py <station>")
        sys.exit(1)

    # Get the station argument
    station = sys.argv[1]

print(f"Station: {station}")
set_station(station)


print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('-------------------------------- Header ------------------------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

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
# Touch -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

use_two_planes_too = config["use_two_planes_too"]
date_selection = config["date_selection"]  # Set to True if you want to filter by date
start_date_filter = config["start_date_filter"]
end_date_filter = config["end_date_filter"]

# Aesthetic -------------------------------------
show_plots = config["show_plots"]
save_plots = config["save_plots"]
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
create_very_essential_plots = config["create_very_essential_plots"]
show_errorbar = config["show_errorbar"]

# Execution -------------------------------------
only_all = config["only_all"]
recalculate_pressure_coeff = config["recalculate_pressure_coeff"]
remove_outliers = config["remove_outliers"]

rolling_effs = config["rolling_effs"]
mean_window = config["mean_window"]
med_window = config["med_window"]
rolling_mean = config["rolling_mean"]
rolling_median = config["rolling_median"]

outlier_gaussian_quantile = config["outlier_gaussian_quantile"]

res_win_min = config["res_win_min"]
HMF_ker = config["HMF_ker"]
MAF_ker = config["MAF_ker"]
acceptance_corr = config["acceptance_corr"]
high_order_correction = config["high_order_correction"]


# -----------------------------------------------------------------------------
# To not touch unless necesary ------------------------------------------------
# -----------------------------------------------------------------------------

# This should come from an input file
eta_P = config["eta_P"]
unc_eta_P = config["unc_eta_P"]
set_a = config["set_a"]
mean_pressure_used_for_the_fit = config["mean_pressure_used_for_the_fit"]

low_lim_eff_plot = config["low_lim_eff_plot"]
systematic_unc = config["systematic_unc"]

systematic_unc_corr_to_real_rate = config["systematic_unc_corr_to_real_rate"]

remove_non_data_points = config["remove_non_data_points"]
repeat_efficiency_calculation = config["repeat_efficiency_calculation"]
decorrelate = config["decorrelate"]
fit_efficiencies = config["fit_efficiencies"]
significant_digits = config["significant_digits"]


resampling_window = f'{res_win_min}min'  # '10min' # '5min' stands for 5 minutes.
print(f"Resampling window set to {resampling_window}.")

global_variables = {}
global_variables['res_win_min'] = res_win_min
global_variables['recalculate_pressure_coeff'] = recalculate_pressure_coeff
# global_variables['outlier_filter'] = outlier_filter*remove_outliers

# Define the base folder and file paths
grafana_directory = os.path.expanduser(f"~/DATAFLOW_v3/GRAFANA_DATA")
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
base_folder = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/STAGE_2")
filepath = f"{base_folder}/total_data_table.csv"
save_filename = f"{base_folder}/large_corrected_table.csv"
grafana_save_filename = f"{grafana_directory}/data_for_grafana_{station}.csv"
figure_path = f"{base_folder}/FIGURES/"
fig_idx = 0
os.makedirs(base_folder, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)
os.makedirs(grafana_directory, exist_ok=True)

status_csv_path = os.path.join(base_folder, "corrector_status.csv")
status_timestamp = append_status_row(status_csv_path)

csv_path = os.path.join(base_folder, "corrector_metadata.csv")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('-------------------------- Function definition -----------------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')


def fit_efficiency_model(x, y, z, model_type='linear'):
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        warnings.warn("Empty arrays for fitting. Returning dummy fit.")
        dummy_fit = lambda P, T: np.full_like(P, fill_value=np.nan, dtype=np.float64)
        return dummy_fit, (np.nan, np.nan, np.nan)

    X = np.column_stack((x, y))

    if model_type == 'linear':
        try:
            model = LinearRegression()
            model.fit(X, z)
            coeffs = model.coef_, model.intercept_
            return lambda P, T: coeffs[0][0] * P + coeffs[0][1] * T + coeffs[1], coeffs
        except Exception as e:
            warnings.warn(f"Linear regression failed: {e}")
            dummy_fit = lambda P, T: np.full_like(P, fill_value=np.nan, dtype=np.float64)
            return dummy_fit, (np.nan, np.nan, np.nan)

    elif model_type == 'sigmoid':
        def sigmoid(xy, a, b, c, d):
            P, T = xy
            return d / (1 + np.exp(-a * (P - b) - c * (T - b)))
        try:
            popt, _ = curve_fit(sigmoid, (x, y), z, maxfev=10000)
            return lambda P, T: sigmoid((P, T), *popt), popt
        except Exception as e:
            warnings.warn(f"Sigmoid fit failed: {e}")
            dummy_fit = lambda P, T: np.full_like(P, fill_value=np.nan, dtype=np.float64)
            return dummy_fit, (np.nan,) * 4

    else:
        raise NotImplementedError(f"Model {model_type} not implemented")



def assign_efficiency_fit(df, eff_col, fit_col, case, plane_number, model_type='linear'):
    filtered = df.dropna(subset=[eff_col, 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
    
    if filtered.empty:
        warnings.warn(f"No valid data for {eff_col}. Skipping fit.")
        df[fit_col] = np.nan
        df[f'unc_{fit_col}'] = np.nan
        return filtered, lambda P, T: np.full_like(P, np.nan), (np.nan, np.nan, np.nan)

    x = filtered['sensors_ext_Pressure_ext'].values
    y = filtered['sensors_ext_Temperature_ext'].values
    z = filtered[eff_col].values

    fit_func, coeffs = fit_efficiency_model(x, y, z, model_type=model_type)

    # Store coefficients (even if NaN, to avoid KeyErrors later)
    if isinstance(coeffs, tuple) and len(coeffs) == 2:
        coef_array, intercept = coeffs
        if isinstance(coef_array, (list, np.ndarray)) and len(coef_array) == 2:
            global_variables[f"eff_fit_P_{case}"] = coef_array[0] if np.isfinite(coef_array[0]) else None
            global_variables[f"eff_fit_T_{case}"] = coef_array[1] if np.isfinite(coef_array[1]) else None
            global_variables[f"eff_fit_intercept_{case}"] = intercept if np.isfinite(intercept) else None
        else:
            warnings.warn(f"Invalid coefficient structure for {case}.")
    else:
        warnings.warn(f"Unexpected fit result structure for {case}.")


    df[fit_col] = fit_func(df['sensors_ext_Pressure_ext'], df['sensors_ext_Temperature_ext'])
    df[f'unc_{fit_col}'] = 1.0

    return filtered, fit_func, coeffs


def plot_combined_efficiency_views(filtered_df, final_eff_col, fit_func, plane_number):
    global create_plots, fig_idx, show_plots, save_plots, figure_path
            
    if create_plots or create_essential_plots:
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
    # else:
    #     # print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
    #     print("\n")
            
        
def plot_side_views_all_planes(data_df, planes, model_type='linear'):
    global create_plots, create_essential_plots, fig_idx, show_plots, save_plots, figure_path, low_lim_eff_plot
            
    if not (create_plots or create_essential_plots):
        # print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
        # print("\n")
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, plane in enumerate(planes):
        eff_col = f'definitive_eff_{plane}'
        fit_col = f'eff_fit_{plane}'

        # Fit model
        filtered = data_df.dropna(subset=[eff_col, 'sensors_ext_Pressure_ext', 'sensors_ext_Temperature_ext']).copy()
        x = filtered['sensors_ext_Pressure_ext'].values
        y = filtered['sensors_ext_Temperature_ext'].values
        z = filtered[eff_col].values

        # Fit linear model
        X = np.column_stack((x, y))
        model = LinearRegression().fit(X, z)
        coeffs = model.coef_, model.intercept_
        fit_func = lambda P, T: coeffs[0][0] * P + coeffs[0][1] * T + coeffs[1]

        # Create fit lines
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = np.linspace(y.min(), y.max(), 200)
                
        # --- Efficiency vs Pressure ---
        ax1 = axes[i]
        z_fit_x = fit_func(x_fit, np.full_like(x_fit, np.mean(y)))
        ax1.scatter(x, z, alpha=0.4, label='Measured')
        ax1.plot(x_fit, z_fit_x, 'r-', label=(
            f'Fit: η(P,⟨T⟩)\n'
            f'a={coeffs[0][0]:.2g}, b={coeffs[0][1]:.2g}, c={coeffs[1]:.2g}'
        ))
        ax1.set_xlabel('Pressure')
        ax1.set_ylabel('Efficiency')
        ax1.set_ylim(low_lim_eff_plot, 1)
        ax1.set_title(f'Plane {plane} – η vs P')
        ax1.legend(fontsize=8)

        # --- Efficiency vs Temperature ---
        ax2 = axes[i + 4]
        z_fit_y = fit_func(np.full_like(y_fit, np.mean(x)), y_fit)
        ax2.scatter(y, z, alpha=0.4, label='Measured')
        ax2.plot(y_fit, z_fit_y, 'r-', label=(
            f'Fit: η(⟨P⟩,T)\n'
            f'a={coeffs[0][0]:.2g}, b={coeffs[0][1]:.2g}, c={coeffs[1]:.2g}'
        ))
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Efficiency')
        ax2.set_ylim(low_lim_eff_plot, 1)
        ax2.set_title(f'Plane {plane} – η vs T')
        ax2.legend(fontsize=8)

    plt.suptitle('Side Projections of Efficiency Fits per Plane (Linear Model)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = f"{figure_path}{fig_idx}_side_views_planes_{case}.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        fig.savefig(new_figure_path, format='png', dpi=300)

    plt.close(fig)
        
        
def plot_eff_vs_rate_grid(data_df, detector_labels):
    global create_plots, fig_idx, show_plots, save_plots, figure_path, case, create_essential_plots

    if not (create_plots or create_essential_plots or create_very_essential_plots):
        # print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
        # print("\n")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, label in enumerate(detector_labels):
        ax = axes[i]
        eff_col = f'detector_{label}_eff'
        rate_col = f'detector_{label}'
        corrected_col = f'detector_{label}_eff_corr'

        valid = (
            data_df[[eff_col, rate_col, corrected_col]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
            
        x = valid[eff_col].values
        y_orig = valid[rate_col].values
        y_corr = valid[corrected_col].values

        corr_orig, _ = pearsonr(x, y_orig)
        corr_corr, _ = pearsonr(x, y_corr)

        p_orig = np.polyfit(x, y_orig, 1)
        p_corr = np.polyfit(x, y_corr, 1)
        x_fit = np.linspace(x.min(), x.max(), 500)

        ax.scatter(x, y_orig, alpha=0.7, label='Original', s=2)
        ax.scatter(x, y_corr, alpha=0.7, label='Corrected', s=2)
        ax.plot(x_fit, np.polyval(p_orig, x_fit), linestyle='--', linewidth=1.0, label='Fit: Original')
        ax.plot(x_fit, np.polyval(p_corr, x_fit), linestyle='--', linewidth=1.0, label='Fit: Corrected')

        ax.set_xlabel('Efficiency')
        ax.set_ylabel('Rate')
        ax.set_title(f'{label}', fontsize=10)
        ax.grid(True)

        textstr = f'Corr (orig): {corr_orig:.2f}\nCorr (corr): {corr_corr:.2f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.6))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9)
    fig.suptitle(f'Efficiency vs. Rate (Original and Corrected) — {case}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + f"_scatter_grid_{case}.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        fig.savefig(new_figure_path, format='png', dpi=300)
    plt.close(fig)


def fit_pressure_model(x, beta, a):
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
    
    if not df.empty:
        # Fit the exponential model using uncertainties in Y as weights
        print("Fitting pressure exponential model...")
        
        # WIP TO USE UNCERTAINTY OF PRESSURE ----------------------------------------------
        popt, pcov = curve_fit(fit_pressure_model, df['delta_P'], df['log_I_over_I0'], sigma=df['unc_log_I_over_I0'], absolute_sigma=True, p0=(1,0))
        b, a = popt  # Extract parameters
        
        # Define eta_P as the parameter b (rate of change in the exponent)
        eta_P = b
        eta_P_ordinate = a
        eta_P_uncertainty = np.sqrt(np.diag(pcov))[0]
        
        global_variables[f'eta_P_{region}'] = eta_P
        global_variables[f'eta_P_ordinate_{region}'] = eta_P_ordinate
        
        # Plot the fitting
        if create_plots:
            plt.figure()
            if show_errorbar:
                plt.errorbar(df['delta_P'], df['log_I_over_I0'], xerr=abs(df['unc_delta_P']), yerr=abs(df['unc_log_I_over_I0']), fmt='o', label='Data with Uncertainty')
            else:
                plt.scatter(df['delta_P'], df['log_I_over_I0'], label='Data', s=1, alpha=0.5, marker='.')
            
            plt.plot(df['delta_P'], fit_pressure_model(df['delta_P'], *popt), color='red', label='Fit')

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
        eta_P_ordinate = np.nan
    return eta_P, eta_P_uncertainty, eta_P_ordinate


def quantile_mean(values, lower_quantile=0.01, upper_quantile=0.99):
    q_low, q_high = np.quantile(values, [lower_quantile, upper_quantile])
    filtered_values = values[(values >= q_low) & (values <= q_high)]
    return filtered_values.mean()


def eff_model(e, label):
    e1, e2, e3, e4 = e
    if label == '1234':
        return (
            e1 * e2 * e3 * e4 +
            e1 * (1 - e2) * e3 * e4 +
            e1 * e2 * (1 - e3) * e4 +
            e1 * (1 - e2) * (1 - e3) * e4 )
            
    elif label == '123':
        return (
            e1 * e2 * e3 +
            e1 * (1 - e2) * e3 )
                
    elif label == '234':
        return (
            e2 * e3 * e4 +
            e2 * (1 - e3) * e4 )
                
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
    
    
def decorrelate_efficiency_least_change(eff, rate_corr, bounds=(0.001, 0.999), penalty=1e5, verbose=True):
    eff = np.asarray(eff, dtype=np.float64)
    rate_corr = np.asarray(rate_corr, dtype=np.float64)

    mask = np.isfinite(eff) & np.isfinite(rate_corr) & (eff > 1e-8)
    if not np.any(mask):
        warnings.warn("No valid data after masking.")
        return eff, None

    eff = eff[mask]
    rate_corr = rate_corr[mask]
    counts = eff * rate_corr
    n = len(eff)

    if verbose:
        print(f"[decorrelate_efficiency_penalty] Valid entries: {n}")

    def loss(eff_prime):
        rate_prime = counts / eff_prime
        # Mean-centered covariance
        cov = np.dot(rate_prime - rate_prime.mean(), eff_prime - eff_prime.mean()) / n
        return np.sum((eff_prime - eff) ** 2) + penalty * cov**2

    bounds_list = [bounds] * n

    try:
        res = minimize(loss, eff, method='L-BFGS-B', bounds=bounds_list, options={'maxiter': 300, 'ftol': 1e-6})
        if not res.success:
            warnings.warn(f"Optimization failed: {res.message}")
            return eff, None
        return res.x, res
    except Exception as e:
        warnings.warn(f"Optimization error: {e}")
        return eff, None


def compute_noise_percentages(est, measured):
    with np.errstate(divide='ignore', invalid='ignore'):
        explained = 100 * est / measured
        noise = 100 - explained
    for arr in (explained, noise):
        arr[np.isnan(arr) | np.isinf(arr)] = 0
    explained = np.clip(explained, 0, 100)
    noise = np.clip(noise, 0, 100)
    return noise


def akima_fill(series: pd.Series) -> pd.Series:
    """Return *series* with NaNs replaced by Akima-interpolated values."""
    y = series.to_numpy(float)
    mask = np.isnan(y)
    if not mask.any():
        return series                                 # nothing to do
    x = np.arange(len(y))
    xi = x[~mask]                                     # coordinates of valid samples
    yi = y[~mask]

    # If fewer than two valid points the interpolation is undefined
    if yi.size < 2:
        return series

    ak = Akima1DInterpolator(xi, yi)
    y[mask] = ak(x[mask])
    return pd.Series(y, index=series.index, name=series.name)


def plot_grouped_series(df, group_cols, time_col='Time', title=None, figsize=(14, 4), save_path=None, plot_after_all = False, sharey_axes =False):

    global create_plots, fig_idx
    if create_plots or create_essential_plots or plot_after_all:
        n_plots = len(group_cols)
        fig, axes = plt.subplots(n_plots, 1, sharex=True, sharey=sharey_axes, figsize=(figsize[0], figsize[1] * n_plots))
        
        if n_plots == 1:
            axes = [axes]  # Make iterable
        
        for idx, cols in enumerate(group_cols):
            ax = axes[idx]
            for col in cols:
                if col in df.columns:
                    ax.plot(df[time_col], df[col], label=col, alpha = 0.5)
                    ax.scatter(df[time_col], df[col], alpha = 0.5, s = 1)
                else:
                    print(f"Warning: column '{col}' not found in DataFrame")
            ax.set_ylabel(' / '.join(cols))
            ax.grid(True)
            ax.legend(loc='best')
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

        axes[-1].set_xlabel('Time')
        if title:
            if 'case' in locals() or 'case' in globals():
                title = title + f', {case}'
            fig.suptitle(title, fontsize=14)
            fig.subplots_adjust(top=0.95)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96] if title else None)

        if show_plots:
            plt.show()
        elif save_plots:
            new_figure_path = figure_path + f"{fig_idx}" + "_series.png"
            fig_idx += 1
            print(f"Saving figure to {new_figure_path}")
            plt.savefig(new_figure_path, format='png', dpi=300)
        plt.close()
    # else:
        # print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
    #     print("\n")


def plot_pressure_and_group(df, x_column, x_label, group_cols, time_col='Time', title=None, figsize=(14, 6), save_path=None):

    global create_plots, fig_idx
    if create_essential_plots:
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [1, 1]})

        ax1.plot(df[time_col], df[x_column], label=x_label, color='tab:blue')
        ax1.set_ylabel(x_label)
        ax1.set_title(title if title else 'Group Signals')
        ax1.grid(True)
        ax1.legend()

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
    # else:
        # print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
        # print("\n")


def solve_efficiencies(row):
    A = row['processed_tt_1234']
    B = row['processed_tt_134']
    C = row['processed_tt_124']
    def equations(vars):
        e1, e2, e3 = vars  # Let e4 = e1
        e4 = e1
        eq1 = B * e2 - A * (1 - e2)  # B*e2 = A*(1 - e2)
        eq2 = C * e3 - A * (1 - e3)  # C*e3 = A*(1 - e3)
        eff_combined = (
            e1 * e2 * e3 * e4 +
            e1 * (1 - e2) * e3 * e4 +
            e1 * e2 * (1 - e3) * e4 )
        eq3 = (A + B + C) / eff_combined - A / (e1 * e2 * e3 * e4)
        return [eq1, eq2, eq3]
    initial_guess = [0.9, 0.9, 0.9]
    result = root(equations, initial_guess, method='hybr')
    if result.success and np.all((0 < result.x) & (result.x < 1)):
        e1, e2, e3 = result.x
        e4 = e1
        return pd.Series([e1, e2, e3, e4])
    else:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])


def solve_efficiencies_four_planes_inner(row):
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
            e2 * (1 - e3) )
        eq3 = (A + B + C) / eff_combined - A / ( e2 * e3 )
        return [eq2, eq3]
            
    def equations_3(vars):
        e2, e3 = vars
        eq1 = B * e2 - A * (1 - e2)  # B*e2 = A*(1 - e2)
        eff_combined = (
            e2 * e3 +
            (1 - e2) * e3 +
            e2 * (1 - e3) )
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
        
        
def solve_efficiencies_four_planes_outer(row):
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
            e2 * (1 - e3) )
        eq3 = (A + B + C) / eff_combined - A / ( e2 * e3 )
        return [eq2, eq3]
            
    def equations_3(vars):
        e2, e3 = vars
        eq1 = B * e2 - A * (1 - e2)  # B*e2 = A*(1 - e2)
        eff_combined = (
            e2 * e3 +
            (1 - e2) * e3 +
            e2 * (1 - e3) )
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


print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('----------------------------- Introduction ---------------------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

# Load the data
print('Reading the big CSV datafile...')
data_df = pd.read_csv(filepath)

print("Putting zeroes to NaNs...")
data_df = data_df.replace(0, np.nan)

print(filepath)
print('File loaded successfully.')

# Get TT regions
detection_types = ['1234', '123', '234', '124', '134', '12', '23', '34', '13', '24', '14']
detector_labels = ['1234', '123', '234', '12', '23', '34']


# Get angular regions
rx_columns = [col for col in data_df.columns if '_R' in col]
rx_names = set()
for col in rx_columns:
    parts = col.split('_')
    if len(parts) > 1 and parts[1].startswith('R'):
        rx_names.add(parts[1])
angular_regions = sorted(rx_names)
# print(f"\nFound RX columns: {angular_regions}")

rx_columns = [c for c in data_df.columns if "_R" in c]
tt_prefixes = sorted(
    {c.split('_')[0] for c in rx_columns if c.split('_')[0].isdigit()}
)
# print("Numeric tt prefixes:", tt_prefixes)


# Define how to combine R-regions
combination_dict = {
    # new label   -> R-region suffixes to add (without leading '_')
    "Vert": ['R0.0', 'R1.0', 'R1.1', 'R1.2', 'R1.3', 'R1.4', 'R1.5', 'R1.6', 'R1.7'],
    "South": ['R2.0', 'R2.7', 'R3.0', 'R3.7'],
    "East": ['R2.2', 'R2.1', 'R3.2', 'R3.1'],
    "North": ['R2.4', 'R2.3', 'R3.4', 'R3.3'], 
    "West": ['R2.6', 'R2.5', 'R3.6', 'R3.5'],
    "all": ['R0.0', 'R1.0', 'R1.1', 'R1.2', 'R1.3', 'R1.4', 'R1.5', 'R1.6', 'R1.7',
            'R2.0', 'R2.7', 'R3.0', 'R3.7', 'R2.2', 'R2.1', 'R3.2', 'R3.1',
            'R2.4', 'R2.3', 'R3.4', 'R3.3', 'R2.6', 'R2.5', 'R3.6', 'R3.5'],
}


# Create combined columns for every tt prefix
for tt in tt_prefixes:
    for new_label, r_list in combination_dict.items():
        # build full column names to be summed
        cols_to_sum = [f"{tt}_{r}" for r in r_list]
        missing = [c for c in cols_to_sum if c not in data_df.columns]
        if missing:
            raise KeyError(f"{tt}: missing source columns {missing}")

        data_df[f"processed_tt_{tt}_{new_label}"] = data_df[cols_to_sum].sum(axis=1)

# print("New combined columns added:")
combinations = []
for new_label in combination_dict:
    combinations.append(new_label)


# Get angular regions
rx_columns = [col for col in data_df.columns if '_R' in col]
rx_names = set()
for col in rx_columns:
    parts = col.split('_')
    if len(parts) > 1 and parts[1].startswith('R'):
        rx_names.add(parts[1])
angular_regions = sorted(rx_names)
# print(f"\nFound RX columns: {angular_regions}")


# angular_regions = angular_regions + ['all']  # Add 'all' to the angular regions
processing_regions = angular_regions + combinations

if only_all:
    print("CHOSEN TO USE ONLY THE ALL COLUMN")
    # combinations = []
    # angular_regions = ['all']
    processing_regions = ['all']

print(f"\nProcessing regions: {processing_regions}")

# Get TT and angular combinations
ang_tt_cols = []
for tt in detection_types:
    for rx in angular_regions:
        col_name = f'{tt}_{rx}'
        if col_name in data_df.columns:
            col_name = f'processed_tt_{tt}_{rx}'
            ang_tt_cols.append(col_name)


# Print the angular TT columns found
# print(f"\nFound angular-TT columns: {ang_tt_cols}")

summing_columns = ang_tt_cols

# Rename columns TT_RX.Y to processed_tt_TT_RX.Y
rename_dict = {}
for col in data_df.columns:
    parts = col.split('_')
    if len(parts) == 2:
        tt, rx = parts
        if tt in detection_types and rx in angular_regions:
            new_name = f'processed_tt_{tt}_{rx}'
            rename_dict[col] = new_name

# Apply the renaming
data_df.rename(columns=rename_dict, inplace=True)


# Preprocess the data to remove rows with invalid datetime format ------------------------------------------
print('\nValidating datetime format in "Time" column...')
try:
    # Try parsing 'Time' column with the specified format
    data_df['Time'] = pd.to_datetime(data_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
except Exception as e:
    print(f"Error while parsing datetime: {e}")
    exit(1)


# Drop rows where 'Time' could not be parsed ----------------------------------------------------------------
invalid_rows = data_df['Time'].isna().sum()
if invalid_rows > 0:
    print(f"Removing {invalid_rows} rows with invalid datetime format.")
    data_df = data_df.dropna(subset=['Time'])
else:
    print("No rows with invalid datetime format removed.")
print('Datetime validation completed successfully.')


# Check if the results file exists --------------------------------------------------------------------------
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
    
    start_date = pd.to_datetime(start_date_filter)  # Use a string in 'YYYY-MM-DD' format
    end_date = pd.to_datetime(end_date_filter)  # Use a string in 'YYYY-MM-DD' format
    
    print(f"Filtering data from {start_date} to {end_date}...")
    
    print("------- SELECTION BY DATE IS BEING PERFORMED -------")
    data_df = data_df[(data_df['Time'] >= start_date) & (data_df['Time'] <= end_date)]

print(f"Filtered data contains {len(data_df)} rows.")


# Define start_time and end_time with the minimum and maximum timestamps in the data_df
start_time = data_df['Time'].min()
end_time = data_df['Time'].max()


if remove_non_data_points:
    # Remove rows where 'events' is NaN or zero
    print(f"Original data contains {len(data_df)} rows before removing non-data points.")
    data_df = data_df.dropna(subset=['events'])
    data_df = data_df[data_df['events'] != 0]
    print(f"Filtered data contains {len(data_df)} rows after removing non-data points.")


# If len == 0, then exit
if len(data_df) == 0:
    print("No data points left after filtering. Exiting.")
    sys.exit(0)


# Define input file path -------------------------------------------------------------------------------
input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")
if os.path.exists(input_file_config_path):
    input_file = pd.read_csv(input_file_config_path, skiprows=1, decimal = ",")
    print("Input configuration file found.")
    exists_input_file = True
else:
    exists_input_file = False
    print("Input configuration file does not exist.")

exists_input_file = False
if exists_input_file:
    # Parse start/end timestamps in the configuration file
    input_file["start"] = pd.to_datetime(input_file["start"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = pd.to_datetime(input_file["end"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = input_file["end"].fillna(pd.to_datetime('now'))

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



print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('-------------------------- Outlier removal ---------------------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')


# Example configuration
# filter_widths = [5, 11, 21, 31, 41, 51, 101, 151, 201, 251, 301, 451, 501, 601]
# filter_widths = [31, 101, 151, 501]
filter_widths = [41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291, 301, \
    311, 321, 331, 341, 351, 361]
quantile_range = (0.001, 0.999)

processed_labels = [f"processed_tt_{tt}_all" for tt in detection_types]
cases = ['events'] + processed_labels


# Dictionary to store results per kernel
gauss_fit_results = {}


def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)


def piecewise_linear(x, x0, k1, k2, b1):
    """
    x0: breakpoint
    k1: slope before x0
    k2: slope after x0
    b1: intercept before x0
    b2: computed as continuity condition
    """
    b2 = (k1 - k2) * x0 + b1
    return np.where(x < x0, k1 * x + b1, k2 * x + b2)


for case in cases:
    print(case)
    
    time_series = data_df['Time'].copy()
    total_series = data_df[case].copy()

    filtered_dict = {}
    residual_dict = {}

    # Apply filters
    for k in filter_widths:
        k = k if k % 2 == 1 else k + 1  # Ensure odd
        filtered = medfilt(total_series, kernel_size=k)
        residual = total_series - filtered
        filtered_dict[k] = filtered
        residual_dict[k] = residual
    
    # === Multi-panel time series plot ===
    if create_essential_plots:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        # Original
        axes[0].scatter(time_series, total_series, s=0.1, alpha=0.5, color='black', label="Original")
        axes[0].set_title("Original Time Series")
        axes[0].set_ylabel("Counts")
        axes[0].grid(True, alpha=0.3)

        # All filtered overlays
        for k, f in filtered_dict.items():
            axes[1].scatter(time_series, f, label=f"Median k={k}", s=0.1, alpha=0.5)
        axes[1].set_title("Median-Filtered Series")
        axes[1].set_ylabel("Filtered")
        axes[1].grid(True, alpha=0.3)
        # axes[1].legend(fontsize=8)

        # Residuals overlay
        for k, r in residual_dict.items():
            axes[2].scatter(time_series, r, label=f"k={k}", s=0.1, alpha=0.5)
        axes[2].set_title(f"Residuals (Original - Filtered)")
        axes[2].set_ylabel("Residuals")
        axes[2].set_xlabel("Time")
        axes[2].grid(True, alpha=0.3)
        # axes[2].legend(fontsize=8)
        
        plt.suptitle(f"Time Series Analysis: {case}", fontsize=14)
        plt.tight_layout()

        if show_plots:
            plt.show()
        elif save_plots:
            path = f"{figure_path}{fig_idx}_combined_series_allk.png"
            print(f"Saving multi-panel plot to {path}")
            plt.savefig(path, format='png', dpi=300)
        plt.close()
        fig_idx += 1

    # === Combined histogram ===
    fig, ax = plt.subplots(figsize=(8, 5))

    # Bin range: symmetric and wide enough
    all_res = np.concatenate(list(residual_dict.values()))
    bins = np.linspace(-150, 150, 100)

    for k, r in residual_dict.items():
        counts, bin_edges = np.histogram(r, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Exclude zero-count bins for fitting
        mask = counts > 0
        x_fit = bin_centers[mask]
        y_fit = counts[mask]

        # Initial guess: A, mu, sigma
        A0 = np.max(y_fit)
        mu0 = np.mean(r[np.isfinite(r)])
        sigma0 = np.std(r[np.isfinite(r)])

        try:
            popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=[A0, mu0, sigma0])
            A_fit, mu_fit, sigma_fit = popt

            # Compute fitted values and chi-squared
            y_model = gaussian(x_fit, *popt)
            chisq = np.sum((y_fit - y_model) ** 2 / y_model)
            dof = len(x_fit) - len(popt)
            chisq_ndf = chisq / dof if dof > 0 else np.nan

            # Store in dictionary
            gauss_fit_results[k] = {
                'mu': mu_fit,
                'sigma': sigma_fit,
                'chisq_ndf': chisq_ndf
            }

            # Plot histogram and fit
            ax.hist(r, bins=bins, histtype='step', label=f"k={k}", linewidth=1.0)
            ax.plot(x_fit, y_model, linestyle='--', linewidth=1.0)

        except Exception as e:
            print(f"Gaussian fit failed for k={k}: {e}")
            gauss_fit_results[k] = {
                'mu': np.nan,
                'sigma': np.nan,
                'chisq_ndf': np.nan
            }

    ax.set_title(f"Residual Histograms (log scale), {case}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    # ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    # ax.legend(title="Kernel Width")

    plt.tight_layout()

    if show_plots:
        plt.show()
    elif save_plots:
        path = f"{figure_path}{fig_idx}_residual_histogram_allk.png"
        print(f"Saving histogram to {path}")
        plt.savefig(path, format='png', dpi=300)
    plt.close()
    fig_idx += 1
    
    
    # Fit result plot
    
    # Prepare sorted kernel values
    k_values = sorted(gauss_fit_results.keys())
    mu_values = [gauss_fit_results[k]['mu'] for k in k_values]
    sigma_values = [gauss_fit_results[k]['sigma'] for k in k_values]
    chisq_values = [gauss_fit_results[k]['chisq_ndf'] for k in k_values]

    # === Multi-panel summary plot of Gaussian fit parameters ===
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # μ (mean)
    axes[0].plot(k_values, mu_values, marker='o', linestyle='-', color='blue')
    axes[0].set_ylabel(r'$\mu$ (Mean)')
    axes[0].set_title("Gaussian Fit Mean vs. Median Filter Width")
    axes[0].grid(True, alpha=0.3)

    # σ (std dev)
    x_data = np.array(k_values)
    y_data = np.array(sigma_values)

    # Initial guess
    x0_init = np.median(x_data)
    k1_init = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]) * 0.5
    k2_init = k1_init * 1.5
    b1_init = y_data[0]

    p0 = [x0_init, k1_init, k2_init, b1_init]

    # Bounds: enforce both slopes to be positive
    bounds = (
        [min(x_data), 0, 0, -np.inf],  # lower bounds: x0, k1, k2, b1
        [max(x_data), np.inf, np.inf, np.inf]  # upper bounds
    )

    # Fit
    try:
        popt, pcov = curve_fit(piecewise_linear, x_data, y_data, p0=p0, bounds=bounds)
        x0_fit, k1_fit, k2_fit, b1_fit = popt

        # Compute fitted curve
        x_fit = np.linspace(min(x_data), max(x_data), 500)
        y_fit = piecewise_linear(x_fit, *popt)

        # Plot on existing axes
        axes[1].plot(x_fit, y_fit, 'k--', linewidth=1.5, label="Piecewise linear fit")
        axes[1].axvline(x0_fit, color='gray', linestyle=':', label=f"Breakpoint at k={x0_fit:.1f}")
        
        axes[1].plot(k_values, sigma_values, marker='o', linestyle='-', color='green')
        axes[1].set_ylabel(r'$\sigma$ (Std. Dev.)')
        axes[1].set_title("Gaussian Fit Std. Dev. vs. Median Filter Width")
        axes[1].grid(True, alpha=0.3)
        
        axes[1].legend(fontsize=8)

        print(f"Piecewise fit completed: breakpoint at k = {x0_fit:.2f}, slopes = ({k1_fit:.4f}, {k2_fit:.4f})")

    except Exception as e:
        print(f"Piecewise linear fit failed: {e}")
        x0_fit = np.nan

    # χ²/NDF
    axes[2].plot(k_values, chisq_values, marker='o', linestyle='-', color='red')
    axes[2].set_ylabel(r'$\chi^2/\mathrm{NDF}$')
    axes[2].set_title("Reduced Chi-squared vs. Median Filter Width")
    axes[2].set_xlabel("Median Filter Width (k)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, max(chisq_values))  # Set y-limits to avoid clutter

    plt.suptitle(f"Gaussian Fit Summary: {case}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show_plots:
        plt.show()
    elif save_plots:
        path = f"{figure_path}{fig_idx}_gaussian_fit_summary_{case}.png"
        print(f"Saving Gaussian fit summary plot to {path}")
        plt.savefig(path, format='png', dpi=300)
    plt.close()
    fig_idx += 1
    
    
    # Actual filter of data_df: choose the closest k in filtered_dict to the piecewise fit breakpoint
    # then retrieve the residuals for that value, the sigma of the fit, and the mean. Then remove all the
    # values that are out of the 0.1% of that gaussian, that is, do ( residuals - mean ) / sigma and remove
    # and consider outliers the values that are out of the 0.1% on both sides. Plot the histogram of the residuals
    # for this specific case and print in screen the number of outliers removed, and the percentage of outliers removed.
    # put nan for the rows and columns that we are working with
    
    # === Outlier rejection using the breakpoint-optimised kernel ===
    
    try:
        # 1. Select kernel width closest to the piece-wise breakpoint
        if not np.isnan(x0_fit):
            k_sel = min(k_values, key=lambda kv: abs(kv - x0_fit))
        else:
            k_sel = k_values[len(k_values) // 2]     # fallback: median kernel
            warnings.warn("Piece-wise fit failed; using median kernel width for outlier rejection.")

        res_sel = residual_dict[k_sel]
        mu_sel = gauss_fit_results[k_sel]['mu']
        sigma_sel = gauss_fit_results[k_sel]['sigma']

        if np.isnan(mu_sel) or np.isnan(sigma_sel) or sigma_sel == 0:
            raise ValueError("Invalid Gaussian parameters for the selected kernel.")

        # 2. Identify points outside the central 99.9 % of the fitted Gaussian
        z_thr = norm.ppf(1.0 - outlier_gaussian_quantile / 2.0)          # ≈ 3.2905
        z_scores = (res_sel - mu_sel) / sigma_sel
        outlier_mask = np.abs(z_scores) > z_thr

        n_out = int(outlier_mask.sum())
        pct_out = 100.0 * n_out / len(outlier_mask)

        print(f"[{case}] k = {k_sel}: removed {n_out} samples "
              f"({pct_out:.4f} %) with |z| > {z_thr:.3f}")

        # 3. Replace outliers with NaN in the working column
        data_df.loc[outlier_mask, case] = np.nan
        
        if create_essential_plots:
            # 4. Diagnostic histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            bin_width = 0.25 * sigma_sel
            bins_diag = np.arange(res_sel.min(), res_sel.max() + bin_width, bin_width)

            ax.hist(res_sel[~outlier_mask], bins=bins_diag,
                    histtype='stepfilled', alpha=0.7, label='Kept')
            if n_out > 0:
                ax.hist(res_sel[outlier_mask], bins=bins_diag,
                        histtype='stepfilled', alpha=0.7, label='Rejected')

            # Overlay fitted Gaussian
            x_gauss = np.linspace(bins_diag[0], bins_diag[-1], 400)
            A_gauss = len(res_sel) * bin_width / (np.sqrt(2 * np.pi) * sigma_sel)
            ax.plot(x_gauss, gaussian(x_gauss, A_gauss, mu_sel, sigma_sel),
                    'k--', linewidth=1.2, label='Gaussian fit')

            ax.set_title(f"Residuals after Outlier Rejection ({case}, k={k_sel})")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Counts")
            ax.grid(True, alpha=0.3)
            # Log scale
            # ax.set_yscale("log")
            ax.legend(fontsize=8)
            plt.tight_layout()

            if show_plots:
                plt.show()
            elif save_plots:
                path = f"{figure_path}{fig_idx}_residual_histogram_k{k_sel}_{case}.png"
                print(f"Saving residual histogram to {path}")
                plt.savefig(path, format='png', dpi=300)
            plt.close()
            fig_idx += 1

    except Exception as e:
        warnings.warn(f"[{case}] Outlier rejection step skipped: {e}")
    
    
    
    # === Multi-panel time series plot ===
    if create_essential_plots:
        fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, sharey=True)

        # Original
        axes[0].scatter(time_series, total_series, s=0.1, alpha=0.5, color='black', label="Original")
        axes[0].set_title("Original Time Series")
        axes[0].set_ylabel("Counts")
        axes[0].grid(True, alpha=0.3)
        
        # All filtered overlays
        new_time_series = data_df['Time'].copy()
        new_total_series = data_df[case].copy()
        axes[1].scatter(new_time_series, new_total_series, label=f"Filtered", s=0.1, alpha=0.5)
        axes[1].set_title("Median-Filtered Series")
        axes[1].set_ylabel("Filtered")
        axes[1].grid(True, alpha=0.3)
        # axes[1].legend(fontsize=8)
        
        plt.suptitle(f"Time Series Analysis: {case}", fontsize=14)
        plt.tight_layout()

        if show_plots:
            plt.show()
        elif save_plots:
            path = f"{figure_path}{fig_idx}_combined_series_OG_and_filtered.png"
            print(f"Saving multi-panel plot to {path}")
            plt.savefig(path, format='png', dpi=300)
        plt.close()
        fig_idx += 1


# ---------------------------------------------------------------
# Fill NaN values by Akima interpolation instead of dropping rows
# ---------------------------------------------------------------

# 'cases' is the list of columns in which NaNs are not tolerated.
cols_with_nans = data_df[cases].columns[data_df[cases].isna().any()]

if len(cols_with_nans):
    print(f"Detected {len(cols_with_nans)} columns with NaNs out of {len(cases)} cases, which is a {100 * len(cols_with_nans) / len(cases):.2f}%")
    for col in cols_with_nans:
        data_df[col] = akima_fill(data_df[col])

    still_nans = data_df[cases].isna().any(axis=1).sum()
    if still_nans:
        print(f"Warning: {still_nans} rows still contain NaNs after Akima interpolation.")
    else:
        print("All missing values filled by Akima interpolation.")
else:
    print("No NaNs detected; nothing interpolated.")

del case


print('----------------------------------------------------------------------')
print('-------------- Interpolating pressure and temperature ----------------')
print('----------------------------------------------------------------------')

group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext']
]

plot_grouped_series(data_df, group_cols, title='Pre-interpolation', plot_after_all=False, sharey_axes=False)

# Ensure 'Time' is the index for resampling
data_df = data_df.set_index('Time')
# Interpolate missing values in 'sensors_ext_Pressure_ext' and 'sensors_ext_Temperature_ext'
data_df['sensors_ext_Pressure_ext'] = data_df['sensors_ext_Pressure_ext'].interpolate(method='akima')
data_df['sensors_ext_Temperature_ext'] = data_df['sensors_ext_Temperature_ext'].interpolate(method='akima')

# Reset index to keep 'Time' as a column
data_df.reset_index(inplace=True)

group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
]
plot_grouped_series(data_df, group_cols, title=f'Post-interpolation', plot_after_all=False, sharey_axes = False)

# With the events now
group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['events']
]
plot_grouped_series(data_df, group_cols, title=f'Pre-resampling', plot_after_all=True, sharey_axes = False)



print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('----------------------------- Resampling -----------------------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

data_df = data_df.copy()

data_df.set_index('Time', inplace=True)
data_df["number_of_mins"] = 1
columns_to_sum = summing_columns + ["number_of_mins"] + ["events"]
columns_to_mean = [col for col in data_df.columns if col not in columns_to_sum]

# Custom aggregation function
data_df = data_df.resample(resampling_window).agg({
    **{col: 'sum' for col in columns_to_sum},   # Sum the count and region columns
    **{col: 'mean' for col in columns_to_mean}  # Mean for the other columns
})

data_df.reset_index(inplace=True)

# Avoid division by zero or invalid mins
denominator = data_df["number_of_mins"] * 60

# Replace non-positive or invalid denominator with NaN to avoid division errors
safe_denominator = np.where((denominator > 0) & np.isfinite(denominator), denominator, np.nan)

# Compute rate and uncertainty safely
data_df['rate'] = data_df['events'] / safe_denominator
data_df['unc_rate'] = np.sqrt(data_df['events']) / safe_denominator

data_df['hv_mean'] = ( data_df['hv_HVneg'] + data_df['hv_HVpos'] ) / 2
data_df['current_mean'] = ( data_df['hv_CurrentNeg'] + data_df['hv_CurrentPos'] ) / 2


group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rate']
]
plot_grouped_series(data_df, group_cols, title=f'Resampled', plot_after_all=True, sharey_axes = False)



print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('------------------- Calculating detector rates -----------------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

# The efficiencies work should be in a loop that at least should work the same for the
# total count case. This means that I should loop on it for each region case, including the total

for case in processing_regions:
    print("\n")
    print("-" * 10)
    print(f'Processing case: {case}')
    
    for tt in detection_types:
        original_col = f'processed_tt_{tt}_{case}'
        new_col = f'processed_tt_{tt}'
        if original_col in data_df.columns:
            data_df[new_col] = data_df[original_col]
    
    
    group_cols = [
        ['processed_tt_1234'],
        ['processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134'],
        ['processed_tt_12', 'processed_tt_23', 'processed_tt_34'],
        ['processed_tt_13', 'processed_tt_24', 'processed_tt_14'],
    ]

    plot_grouped_series(data_df, group_cols, title=f'Counts pre-filtering per TT', plot_after_all=True, sharey_axes = False)
    
    
    print('----------------------------------------------------------------------')
    print('-------------------- Calculating efficiencies ------------------------')
    print('----------------------------------------------------------------------')
    
    eff_system = False
    if eff_system:
        data_df[['ancillary_1', 'eff_sys_2', 'eff_sys_3', 'ancillary_4']] = data_df.apply(solve_efficiencies_four_planes_inner, axis=1)
        data_df[[f'eff_sys_1', f'ancillary_2', f'ancillary_3', f'eff_sys_4']] = data_df.apply(solve_efficiencies_four_planes_outer, axis=1)
    else:
        print("Calculating efficiency without the system")
        data_df['eff_sys_1']  = data_df['processed_tt_1234'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_234'] )
        data_df['eff_sys_3']  = data_df['processed_tt_1234'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_134'] )
        data_df['eff_sys_2']  = data_df['processed_tt_1234'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_124'] )
        data_df['eff_sys_4']  = data_df['processed_tt_1234'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_123'] )
        
    
    if rolling_effs:
        print("Rolling efficiencies...")
        
        data_df['eff_pre_roll_1'] = data_df['eff_sys_1']
        data_df['eff_pre_roll_2'] = data_df['eff_sys_2']
        data_df['eff_pre_roll_3'] = data_df['eff_sys_3']
        data_df['eff_pre_roll_4'] = data_df['eff_sys_4']
        
        cols_to_interpolate = ['eff_sys_1', 'eff_sys_2', 'eff_sys_3', 'eff_sys_4']

        # Step 1: Identify rows where interpolation will be applied
        interpolated_mask = data_df[cols_to_interpolate].replace(0, np.nan).isna()

        # Step 2: Perform interpolation
        data_df[cols_to_interpolate] = data_df[cols_to_interpolate].replace(0, np.nan).interpolate(method='akima')

        if rolling_median:
            for col in cols_to_interpolate:
                data_df[col] = medfilt(data_df[col], kernel_size=med_window)

        if rolling_mean:
            for col in cols_to_interpolate:
                data_df[col] = data_df[col].rolling(window=mean_window, center=True, min_periods=1).mean()

        # Step 4: Set previously interpolated positions back to NaN
        for col in cols_to_interpolate:
            data_df.loc[interpolated_mask[col], col] = np.nan

        group_cols = [ ['eff_pre_roll_1', 'eff_pre_roll_2', 'eff_pre_roll_3', 'eff_pre_roll_4'],
                    ['eff_sys_1', 'eff_sys_2', 'eff_sys_3', 'eff_sys_4'] ]
        plot_grouped_series(data_df, group_cols, title='Final calculated efficiencies, rolling', plot_after_all=True)
    
    
    group_cols = [
        [f'eff_sys_1'],
        [f'eff_sys_2'],
        [f'eff_sys_3'],
        [f'eff_sys_4'] ]
    plot_grouped_series(data_df, group_cols, title=f'Four plane efficiencies', plot_after_all=True, sharey_axes = True)
    
    
    check_efficiency_calculations = False
    if check_efficiency_calculations:
        # -----------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------------------
        # Four plane cases ------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------------------

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
        # plot_grouped_series(data_df, group_cols, title='Corrected rates comparison, 4-fold')


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
        # plot_grouped_series(data_df, group_cols, title='Corrected rates comparison, 3-fold, 123')
        
        group_cols = [ 'eff_sys_123_2', 'eff_sys_2' ]
        # plot_grouped_series(data_df, group_cols, title='Corrected effs. comparison, 3-fold, 123')

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
        # plot_grouped_series(data_df, group_cols, title='Corrected rates comparison, 3-fold, 234')
        
        group_cols = [ 'eff_sys_234_3', 'eff_sys_3' ]
        # plot_grouped_series(data_df, group_cols, title='Corrected effs. comparison, 3-fold, 234')
        
        
        # Checking calculated efficiencies

        # group_cols = [
        #     ['eff_sys_123_1', 'eff_sys_1'],
        #     ['eff_sys_123_2', 'eff_sys_2', 'eff_sys_234_2', 'eff_sys_123_2'],
        #     ['eff_sys_123_3', 'eff_sys_3', 'eff_sys_234_3', 'eff_sys_234_3'],
        #     ['eff_sys_234_4', 'eff_sys_4']
        # ]
        
        group_cols = [
            ['eff_sys_123_1', 'eff_sys_1'],
            ['eff_sys_123_2', 'eff_sys_2'],
            ['eff_sys_3', 'eff_sys_234_3'],
            ['eff_sys_234_4', 'eff_sys_4'] ]
        # plot_grouped_series(data_df, group_cols, title='Corrected efficiencies')
    
    noise_removal = True
    if noise_removal:
        print('----------------------------------------------------------------------')
        print('------------------------ Calculating noise ---------------------------')
        print('----------------------------------------------------------------------')
        
        data_df['subdetector_123_123'] = data_df['processed_tt_1234'] + data_df['processed_tt_123']
        data_df['subdetector_123_13'] = data_df['processed_tt_13'] + data_df['processed_tt_134']
        
        data_df['subdetector_234_234'] = data_df['processed_tt_1234'] + data_df['processed_tt_234']
        data_df['subdetector_234_24'] = data_df['processed_tt_24'] + data_df['processed_tt_124']
        
        comp_eff_2  = data_df['processed_tt_134'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_134'] )
        comp_eff_3  = data_df['processed_tt_124'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_124'] )
        comp_eff_23 = comp_eff_2 * comp_eff_3
        comp_eff_23_true = data_df['processed_tt_14'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_14'] )
        
        eff_df = pd.DataFrame({
            'Time': data_df['Time'],
            'comp_eff_2': comp_eff_2,
            'comp_eff_3': comp_eff_3,
            'comp_eff_23': comp_eff_23,
            'comp_eff_23_true': comp_eff_23_true
            })
        
        # Plot the efficiencies
        group_cols = [
            ['comp_eff_2', 'comp_eff_3', 'comp_eff_23', 'comp_eff_23_true']
        ]
        # plot_grouped_series(eff_df, group_cols, title='Complementary Efficiencies', figsize=(14, 4))
        
        # Assign the counts
        counts_1234 = data_df['processed_tt_1234']
        counts_124 = data_df['processed_tt_124']
        counts_134 = data_df['processed_tt_134']
        
        counts_123 = data_df['processed_tt_123']
        counts_234 = data_df['processed_tt_234']
        counts_12 = data_df['processed_tt_12']
        counts_23 = data_df['processed_tt_23']
        counts_13 = data_df['processed_tt_13']
        counts_24 = data_df['processed_tt_24']
        counts_34 = data_df['processed_tt_34']
        
        counts_14 = data_df['processed_tt_14']
        
        counts_123_sd_123 = data_df['subdetector_123_123']
        counts_13_sd_123 = data_df['subdetector_123_13']
        counts_234_sd_234 = data_df['subdetector_234_234']
        counts_24_sd_234 = data_df['subdetector_234_24']
        
        # --- CASE: 124 (miss plane 3) ------------------------------------------------
        est_124    = counts_1234 * comp_eff_3
        noise_124 = compute_noise_percentages(est_124, counts_124)
            
        # --- CASE: 134 (miss plane 2) ------------------------------------------------
        est_134    = counts_1234 * comp_eff_2
        noise_134 = compute_noise_percentages(est_134, counts_134)

        # --- CASE: 14 (miss both planes 2 & 3) ---------------------------------------
        est_14    = counts_1234 * comp_eff_23
        noise_14 = compute_noise_percentages(est_14, counts_14)

        # subdetector_123_tt ----------------------------------------------------------------
        est_13_sd_123 = counts_123_sd_123 * comp_eff_2
        noise_13_sd_123 = compute_noise_percentages(est_13_sd_123, counts_13_sd_123)
            
        # subdetector_234_tt ----------------------------------------------------------------
        est_24_sd_234 = counts_234_sd_234 * comp_eff_3
        noise_24_sd_234 = compute_noise_percentages(est_24_sd_234, counts_24_sd_234)
        
        # Calculate noise counts based on the percentages
        noise_counts_124 = noise_124 / 100 * counts_124
        noise_counts_134 = noise_134 / 100 * counts_134
        noise_counts_14 = noise_14 / 100 * counts_14
        noise_counts_13_sd_123 = noise_13_sd_123 / 100 * counts_13_sd_123
        noise_counts_24_sd_234 = noise_24_sd_234 / 100 * counts_24_sd_234
        
        # Calculate the rest of noise counts using the system etc. different methods
        
        # Initialize as zeros as long as the oters
        noise_counts_1234 = np.zeros_like(counts_1234)
        noise_counts_123 = np.zeros_like(counts_123)
        noise_counts_234 = np.zeros_like(counts_234)
        noise_counts_12 = np.zeros_like(counts_12)
        noise_counts_23 = np.zeros_like(counts_23)
        noise_counts_13 = np.zeros_like(counts_13)
        noise_counts_24 = np.zeros_like(counts_24)
        noise_counts_34 = np.zeros_like(counts_34)
        
        # ------------------------------------------------------------------------------
        
        comp_eff_1 = counts_234 / (counts_1234 + counts_234)     # P(plane-1 misfire)
        comp_eff_4 = counts_123 / (counts_1234 + counts_123)     # P(plane-4 misfire)

        # 2.  Noise in the remaining three-fold coincidences (123 and 234)
        est_123   = counts_1234 * comp_eff_4           # only plane-4 missing
        noise_123 = compute_noise_percentages(est_123, counts_123)
        noise_counts_123 = noise_123 / 100 * counts_123

        est_234   = counts_1234 * comp_eff_1           # only plane-1 missing
        noise_234 = compute_noise_percentages(est_234, counts_234)
        noise_counts_234 = noise_234 / 100 * counts_234

        # 3.  Two-fold coincidences: build the required double-inefficiency terms
        comp_eff_12 = comp_eff_1 * comp_eff_2
        comp_eff_14 = comp_eff_1 * comp_eff_4
        comp_eff_34 = comp_eff_3 * comp_eff_4
        comp_eff_13 = comp_eff_1 * comp_eff_3
        comp_eff_24 = comp_eff_2 * comp_eff_4

        est_34   = counts_1234 * comp_eff_12
        est_23   = counts_1234 * comp_eff_14
        est_12   = counts_1234 * comp_eff_34
        est_13   = counts_1234 * comp_eff_24
        est_24   = counts_1234 * comp_eff_13

        noise_12 = compute_noise_percentages(est_12, counts_12)
        noise_23 = compute_noise_percentages(est_23, counts_23)
        noise_34 = compute_noise_percentages(est_34, counts_34)
        noise_13 = compute_noise_percentages(est_13, counts_13)
        noise_24 = compute_noise_percentages(est_24, counts_24)

        noise_counts_12 = noise_12 / 100 * counts_12
        noise_counts_23 = noise_23 / 100 * counts_23
        noise_counts_34 = noise_34 / 100 * counts_34
        noise_counts_13 = noise_13 / 100 * counts_13
        noise_counts_24 = noise_24 / 100 * counts_24
        
        # ------------------------------------------------------------------------------
        
        # Denoise the counts by subtracting the noise counts
        denoised_counts_124 = counts_124 - noise_counts_124
        denoised_counts_134 = counts_134 - noise_counts_134
        denoised_counts_14 = counts_14 - noise_counts_14
        denoised_counts_1234 = counts_1234 - noise_counts_1234
        denoised_counts_123 = counts_123 - noise_counts_123
        denoised_counts_234 = counts_234 - noise_counts_234
        denoised_counts_12 = counts_12 - noise_counts_12
        denoised_counts_23 = counts_23 - noise_counts_23
        denoised_counts_13 = counts_13 - noise_counts_13
        denoised_counts_24 = counts_24 - noise_counts_24
        denoised_counts_34 = counts_34 - noise_counts_34
        
        denoised_counts_13_sd_123 = counts_13_sd_123 - noise_counts_13_sd_123
        denoised_counts_24_sd_234 = counts_24_sd_234 - noise_counts_24_sd_234
        
        # Create a new dataframe with the noise vectors as columns so you can apply the plot_groupes_series
        noise_df = pd.DataFrame({
            'Time': data_df['Time'],

            # ───────────────────────────── raw counts ─────────────────────────────
            'counts_1234': counts_1234,
            'counts_124':  counts_124,
            'counts_134':  counts_134,
            'counts_14':   counts_14,
            'counts_123':  counts_123,
            'counts_234':  counts_234,
            'counts_12':   counts_12,
            'counts_23':   counts_23,
            'counts_34':   counts_34,
            'counts_13':   counts_13,
            'counts_24':   counts_24,
            'counts_13_sd_123': counts_13_sd_123,
            'counts_24_sd_234': counts_24_sd_234,

            # ────────────────────────── model estimates ───────────────────────────
            'est_124':  est_124,
            'est_134':  est_134,
            'est_14':   est_14,
            'est_123':  est_123,
            'est_234':  est_234,
            'est_12':   est_12,
            'est_23':   est_23,
            'est_34':   est_34,
            'est_13':   est_13,
            'est_24':   est_24,
            'est_13_sd_123': est_13_sd_123,
            'est_24_sd_234': est_24_sd_234,

            # ───────────────────────── noise percentages ──────────────────────────
            'noise_124':  noise_124,
            'noise_134':  noise_134,
            'noise_14':   noise_14,
            'noise_123':  noise_123,
            'noise_234':  noise_234,
            'noise_12':   noise_12,
            'noise_23':   noise_23,
            'noise_34':   noise_34,
            'noise_13':   noise_13,
            'noise_24':   noise_24,
            'noise_13_sd_123': noise_13_sd_123,
            'noise_24_sd_234': noise_24_sd_234,

            # ─────────────────────── absolute noise counts ────────────────────────
            'noise_counts_124':  noise_counts_124,
            'noise_counts_134':  noise_counts_134,
            'noise_counts_14':   noise_counts_14,
            'noise_counts_123':  noise_counts_123,
            'noise_counts_234':  noise_counts_234,
            'noise_counts_12':   noise_counts_12,
            'noise_counts_23':   noise_counts_23,
            'noise_counts_34':   noise_counts_34,
            'noise_counts_13':   noise_counts_13,
            'noise_counts_24':   noise_counts_24,
            'noise_counts_13_sd_123': noise_counts_13_sd_123,
            'noise_counts_24_sd_234': noise_counts_24_sd_234,

            # ─────────────────────────── denoised data ────────────────────────────
            'denoised_counts_124':  denoised_counts_124,
            'denoised_counts_134':  denoised_counts_134,
            'denoised_counts_14':   denoised_counts_14,
            'denoised_counts_123':  denoised_counts_123,
            'denoised_counts_234':  denoised_counts_234,
            'denoised_counts_12':   denoised_counts_12,
            'denoised_counts_23':   denoised_counts_23,
            'denoised_counts_34':   denoised_counts_34,
            'denoised_counts_13':   denoised_counts_13,
            'denoised_counts_24':   denoised_counts_24,
            'denoised_counts_13_sd_123': denoised_counts_13_sd_123,
            'denoised_counts_24_sd_234': denoised_counts_24_sd_234,
        })

        
        # Clip negative values to 0, but not the time column, which is datetime
        noise_df.loc[:, noise_df.columns != 'Time'] = noise_df.loc[:, noise_df.columns != 'Time'].clip(lower=0)
        noise_df = noise_df.replace(0, np.nan)  # Replace zeros with NaN for better plotting
        
        # Create group cols but now pair the counts_ and the denoised_counts_
        group_cols = [
            # single–plane-missing
            ['counts_124', 'denoised_counts_124', 'est_124'],
            ['counts_134', 'denoised_counts_134', 'est_134'],
            ['counts_14',  'denoised_counts_14',  'est_14'],

            # mixed sub-detector channels (keep if you still need them)
            ['counts_13_sd_123', 'denoised_counts_13_sd_123', 'est_13_sd_123'],
            ['counts_24_sd_234', 'denoised_counts_24_sd_234', 'est_24_sd_234'],

            # two-fold and three-fold coincidences just added
            ['counts_13', 'denoised_counts_13', 'est_13'],
            ['counts_24', 'denoised_counts_24', 'est_24'],
            ['counts_12', 'denoised_counts_12', 'est_12'],
            ['counts_23', 'denoised_counts_23', 'est_23'],
            ['counts_34', 'denoised_counts_34', 'est_34'],
            ['counts_123', 'denoised_counts_123', 'est_123'],
            ['counts_234', 'denoised_counts_234', 'est_234'],
        ]

        plot_grouped_series(
            noise_df,
            group_cols,
            title='Noise study: raw, denoised and estimated counts',
            plot_after_all=True,
        )
        
        data_df['processed_tt_1234'] = denoised_counts_1234
        data_df['processed_tt_124'] = denoised_counts_124
        data_df['processed_tt_134'] = denoised_counts_134
        data_df['processed_tt_123'] = denoised_counts_123
        data_df['processed_tt_234'] = denoised_counts_234
        data_df['processed_tt_14'] = denoised_counts_14
        data_df['processed_tt_12'] = denoised_counts_12
        data_df['processed_tt_13'] = denoised_counts_13
        data_df['processed_tt_23'] = denoised_counts_23
        data_df['processed_tt_24'] = denoised_counts_24
        data_df['processed_tt_34'] = denoised_counts_34
        
        
        group_cols = [
            ['processed_tt_1234'],
            ['processed_tt_123', 'processed_tt_234', 'processed_tt_124', 'processed_tt_134'],
            ['processed_tt_12', 'processed_tt_23', 'processed_tt_34'],
            ['processed_tt_13', 'processed_tt_24', 'processed_tt_14'],
        ]

        plot_grouped_series(data_df, group_cols, title=f'Denoised. Counts per TT', plot_after_all=True, sharey_axes = False)
        
        
        
        if repeat_efficiency_calculation:
            if eff_system:
                data_df[['ancillary_1', 'eff_sys_2_denoised', 'eff_sys_3_denoised', 'ancillary_4']] = data_df.apply(solve_efficiencies_four_planes_inner, axis=1)
                data_df[[f'eff_sys_1_denoised', f'ancillary_2', f'ancillary_3', f'eff_sys_4_denoised']] = data_df.apply(solve_efficiencies_four_planes_outer, axis=1)
            else:
                print("Calculating denoised efficiency value with no system.")
                data_df['eff_sys_1_denoised']  = data_df['processed_tt_1234'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_234'] )
                data_df['eff_sys_3_denoised']  = data_df['processed_tt_1234'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_134'] )
                data_df['eff_sys_2_denoised']  = data_df['processed_tt_1234'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_124'] )
                data_df['eff_sys_4_denoised']  = data_df['processed_tt_1234'] / ( data_df['processed_tt_1234'] + data_df['processed_tt_123'] )
                
            group_cols = [
                ['eff_sys_1', 'eff_sys_1_denoised'],
                ['eff_sys_2', 'eff_sys_2_denoised'],
                ['eff_sys_3' , 'eff_sys_3_denoised'],
                ['eff_sys_4', 'eff_sys_4_denoised'] ]

            plot_grouped_series(data_df, group_cols, title=f'Four plane efficiencies, denoised version', plot_after_all=True)
    
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    
    data_df['eff_1'] = ( data_df['eff_sys_1'] )
    data_df['eff_2'] = ( data_df['eff_sys_2'] )
    data_df['eff_3'] = ( data_df['eff_sys_3'] )
    data_df['eff_4'] = ( data_df['eff_sys_4'] )
    
    # data_df['eff_1'] = ( data_df['eff_sys_1'] + data_df['eff_sys_123_1'] ) / 2
    # data_df['eff_2'] = ( data_df['eff_sys_2'] + data_df['eff_sys_123_2'] ) / 2
    # data_df['eff_3'] = ( data_df['eff_sys_3'] + data_df['eff_sys_234_3'] ) / 2
    # data_df['eff_4'] = ( data_df['eff_sys_4'] + data_df['eff_sys_234_4'] ) / 2

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
    
    # --------------------------------------------------------------------------
    
    if rolling_effs:
        print("Rolling efficiencies...")
        
        data_df['eff_pre_roll_1'] = data_df['final_eff_1']
        data_df['eff_pre_roll_2'] = data_df['final_eff_2']
        data_df['eff_pre_roll_3'] = data_df['final_eff_3']
        data_df['eff_pre_roll_4'] = data_df['final_eff_4']
        
        cols_to_interpolate = ['final_eff_1', 'final_eff_2', 'final_eff_3', 'final_eff_4']

        # Step 1: Identify rows where interpolation will be applied
        interpolated_mask = data_df[cols_to_interpolate].replace(0, np.nan).isna()

        # Step 2: Perform interpolation
        data_df[cols_to_interpolate] = data_df[cols_to_interpolate].replace(0, np.nan).interpolate(method='akima')

        if rolling_median:
            for col in cols_to_interpolate:
                data_df[col] = medfilt(data_df[col], kernel_size=med_window)

        if rolling_mean:
            for col in cols_to_interpolate:
                data_df[col] = data_df[col].rolling(window=mean_window, center=True, min_periods=1).mean()

        # Step 4: Set previously interpolated positions back to NaN
        for col in cols_to_interpolate:
            data_df.loc[interpolated_mask[col], col] = np.nan

        group_cols = [ ['eff_pre_roll_1', 'eff_pre_roll_2', 'eff_pre_roll_3', 'eff_pre_roll_4'],
                    ['final_eff_1', 'final_eff_2', 'final_eff_3', 'final_eff_4'] ]
        plot_grouped_series(data_df, group_cols, title='Denoised. Final calculated efficiencies, rolling', plot_after_all=True)

    else:
        group_cols = [ ['final_eff_1', 'final_eff_2', 'final_eff_3', 'final_eff_4'] ]
        plot_grouped_series(data_df, group_cols, title='Denoised. Final calculated efficiencies, not rolling', plot_after_all=True)
    
    
    group_cols = [
        [f'final_eff_1'],
        [f'final_eff_2'],
        [f'final_eff_3'],
        [f'final_eff_4'] ]
    plot_grouped_series(data_df, group_cols, title=f'Denoised. Four plane efficiencies', plot_after_all=True, sharey_axes = True)



    print('----------------------------------------------------------------------')
    print('----------------- Following the subdetectors idea --------------------')
    print('----------------------------------------------------------------------')

    e1 = data_df['final_eff_1']
    e2 = data_df['final_eff_2']
    e3 = data_df['final_eff_3']
    e4 = data_df['final_eff_4']
    
    if use_two_planes_too:
        # Detector 1234
        # 'processed_tt_1234', 'processed_tt_124', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_1234'] = data_df['processed_tt_1234'] + data_df['processed_tt_124'] + data_df['processed_tt_134'] + data_df['processed_tt_14'] 
        data_df['detector_1234'] = data_df['detector_1234']  / ( data_df["number_of_mins"] * 60 )
        
        # Detector 123
        # 'processed_tt_123', 'processed_tt_1234',  
        # 'processed_tt_124', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_123'] = data_df['processed_tt_1234'] + data_df['processed_tt_123'] + \
            data_df['processed_tt_13'] + data_df['processed_tt_134'] + data_df['processed_tt_124'] + data_df['processed_tt_14'] 
        data_df['detector_123'] = data_df['detector_123'] / ( data_df["number_of_mins"] * 60 )

        # Detector 234
        # 'processed_tt_234', 'processed_tt_1234', 'processed_tt_14'
        # 'processed_tt_124', 'processed_tt_24', 'processed_tt_134',
        data_df['detector_234'] = data_df['processed_tt_234'] + data_df['processed_tt_1234'] + \
            data_df['processed_tt_14'] + data_df['processed_tt_124'] + data_df['processed_tt_24'] + data_df['processed_tt_134']
        data_df['detector_234'] = data_df['detector_234'] / ( data_df["number_of_mins"] * 60 )

        # Detector 12
        # 'processed_tt_12', 'processed_tt_123', 'processed_tt_1234', 
        # 'processed_tt_124', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_12'] = data_df['processed_tt_12'] + data_df['processed_tt_123'] + \
            data_df['processed_tt_1234'] + data_df['processed_tt_124'] + data_df['processed_tt_13'] + data_df['processed_tt_134'] + data_df['processed_tt_14'] 
        data_df['detector_12'] = data_df['detector_12'] / ( data_df["number_of_mins"] * 60 )

        # Detector 23
        # 'processed_tt_234', 'processed_tt_123', 'processed_tt_1234', 'processed_tt_23', 
        # 'processed_tt_124', 'processed_tt_24', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_23'] = data_df['processed_tt_234'] + data_df['processed_tt_123'] + \
            data_df['processed_tt_1234'] + data_df['processed_tt_23'] + data_df['processed_tt_124'] + \
                data_df['processed_tt_24'] + data_df['processed_tt_13'] + data_df['processed_tt_134'] + data_df['processed_tt_14']
        data_df['detector_23'] = data_df['detector_23'] / ( data_df["number_of_mins"] * 60 )

        # Detector 34
        # 'processed_tt_234', 'processed_tt_1234',
        # 'processed_tt_124', 'processed_tt_34', 'processed_tt_24', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_34'] = data_df['processed_tt_234'] + data_df['processed_tt_1234'] + \
            data_df['processed_tt_124'] + data_df['processed_tt_34'] + data_df['processed_tt_24'] + \
                data_df['processed_tt_134'] + data_df['processed_tt_14']
        data_df['detector_34'] = data_df['detector_34'] / ( data_df["number_of_mins"] * 60 )
        
        
        # ─────────────────────────────────────────────────────────────────────────────
        # Efficiencies – every term corresponds to one counted coincidence pattern
        # ─────────────────────────────────────────────────────────────────────────────

        # 1234 detector  →  patterns 1234, 124, 134, 14
        data_df['detector_1234_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )

        # 123 detector   →  patterns 1234, 123, 124, 134, 13, 14
        data_df['detector_123_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + e1 *  e2 *  e3 * (1 - e4)                  # 123
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + e1 * (1 - e2) *  e3 * (1 - e4)             # 13
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )

        # 234 detector   →  patterns 1234, 234, 124, 134, 24, 14
        data_df['detector_234_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + (1 - e1) * e2 *  e3 *  e4                  # 234
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + (1 - e1) * e2 * (1 - e3) * e4              # 24
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )

        # 12 detector    →  patterns 1234, 123, 124, 134, 13, 14, 12
        data_df['detector_12_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + e1 *  e2 *  e3 * (1 - e4)                  # 123
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + e1 * (1 - e2) *  e3 * (1 - e4)             # 13
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
            + e1 *  e2 * (1 - e3) * (1 - e4)             # 12
        )

        # 23 detector    →  patterns 1234, 234, 23, 124, 24, 13, 134, 14
        data_df['detector_23_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + (1 - e1) * e2 *  e3 *  e4                  # 234
            + (1 - e1) * e2 *  e3 * (1 - e4)             # 23
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + (1 - e1) * e2 * (1 - e3) * e4              # 24
            + e1 * (1 - e2) *  e3 * (1 - e4)             # 13
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )

        # 34 detector    →  patterns 1234, 234, 124, 134, 34, 24, 14
        data_df['detector_34_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + (1 - e1) * e2 *  e3 *  e4                  # 234
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + (1 - e1) * (1 - e2) * e3 *  e4             # 34
            + (1 - e1) * e2 * (1 - e3) * e4              # 24
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )
        
        # Now correcting by efficiency
        data_df['detector_1234_eff_corr'] = data_df['detector_1234'] / data_df['detector_1234_eff']
        data_df['detector_123_eff_corr'] = data_df['detector_123'] / data_df['detector_123_eff']
        data_df['detector_234_eff_corr'] = data_df['detector_234'] / data_df['detector_234_eff']
        data_df['detector_12_eff_corr'] = data_df['detector_12'] / data_df['detector_12_eff']
        data_df['detector_23_eff_corr'] = data_df['detector_23'] / data_df['detector_23_eff']
        data_df['detector_34_eff_corr'] = data_df['detector_34'] / data_df['detector_34_eff']
    
    else:
        
        # ────────────────────────────────────────────────────────────────────────────────
        #  SINGLE-PLANE AND THREE-PLANE TOPOLOGIES ONLY ( == False)
        # ────────────────────────────────────────────────────────────────────────────────

        # ── Detector 1234 ───────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 124, 134
        data_df['detector_1234'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_1234_eff'] = (
              e1 * e2 *  e3 *  e4          # 1234
            + e1 * e2 * (1-e3) * e4        # 124  (plane-3 missing)
            + e1 * (1-e2) * e3 *  e4       # 134  (plane-2 missing)
        )

        # ── Detector 123 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 123, 124, 134
        data_df['detector_123'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_123']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_123_eff'] = (
              e1 * e2 *  e3 *  e4          # 1234
            + e1 * e2 *  e3 * (1-e4)       # 123  (plane-4 missing)
            + e1 * e2 * (1-e3) * e4        # 124  (plane-3 missing)
            + e1 * (1-e2) * e3 *  e4       # 134  (plane-2 missing)
        )

        # ── Detector 234 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 234, 124, 134
        data_df['detector_234'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_234']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_234_eff'] = (
              e1 *  e2 *  e3 *  e4         # 1234
            + (1-e1)*e2 *  e3 *  e4        # 234  (plane-1 missing)
            + e1 *  e2 * (1-e3) * e4       # 124  (plane-3 missing)
            + e1 * (1-e2)* e3 *  e4        # 134  (plane-2 missing)
        )

        # ── Detector 12 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 123, 124, 134
        data_df['detector_12'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_123']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_12_eff'] = (
              e1 * e2 *  e3 *  e4          # 1234
            + e1 * e2 *  e3 * (1-e4)       # 123
            + e1 * e2 * (1-e3) * e4        # 124
            + e1 * (1-e2)* e3 *  e4        # 134
        )

        # ── Detector 23 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 234, 124, 134
        data_df['detector_23'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_234']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_23_eff'] = (
              e1 *  e2 *  e3 *  e4         # 1234
            + (1-e1)*e2 *  e3 *  e4        # 234
            + e1 *  e2 * (1-e3) * e4       # 124
            + e1 * (1-e2)* e3 *  e4        # 134
        )

        # ── Detector 34 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 234, 124, 134
        data_df['detector_34'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_234']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_34_eff'] = (
              e1 *  e2 *  e3 *  e4         # 1234
            + (1-e1)*e2 *  e3 *  e4        # 234
            + e1 *  e2 * (1-e3) * e4       # 124
            + e1 * (1-e2)* e3 *  e4        # 134
        )

        # ── Efficiency-corrected rates ────────────────────────────────────────────────
        data_df['detector_1234_eff_corr'] = data_df['detector_1234'] / data_df['detector_1234_eff']
        data_df['detector_123_eff_corr']  = data_df['detector_123']  / data_df['detector_123_eff']
        data_df['detector_234_eff_corr']  = data_df['detector_234']  / data_df['detector_234_eff']
        data_df['detector_12_eff_corr']   = data_df['detector_12']   / data_df['detector_12_eff']
        data_df['detector_23_eff_corr']   = data_df['detector_23']   / data_df['detector_23_eff']
        data_df['detector_34_eff_corr']   = data_df['detector_34']   / data_df['detector_34_eff']

    
    print("Efficiency corrected rates calculated in first stage.")
    
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
    plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment', plot_after_all=True)
    
    
    
    # ------------------------------------------------------------------------------------------------------------------------
    
    if decorrelate:
        
        print('----------------------------------------------------------------------')
        print('------------------ Decorrelating rate and efficiency -----------------')
        print('----------------------------------------------------------------------')
        
        # CODE TO RECALCULATE EFFICIENCIES TO MINIMIZE THE CORRELATION OF THE CORR RATES WITH THE COMBINED EFFS

        # My goal is to eliminate the correlation between rate_caseX and eff_caseX. This will help us define conditions for 
        # e1, e2, e3, and e4 that are used to calculate eff_caseX.

        # First, I need to determine an affine transformation for rate_caseX based on eff_caseX that removes this 
        # correlation while preserving the mean of rate_caseX. This transformation should also allow for a slightly 
        # different correction to be applied for each eff_caseX value, accounting for detector_X_eff_corr through 
        # a linear function as well.

        # Once this transformation is defined, I want to incorporate all its operations into the calculation of 
        # eff_caseX. This should lead to a system of equations that defines the necessary function of e1, e2, e3, 
        # and e4 required to minimize the correlation between the transformed rate_caseX and the new eff_caseX.
        
        data_df = data_df.copy()
        print(case)
        
        for label in detector_labels:
            
            print("Decorrelating for case:", label)
            
            eff_col = f'detector_{label}_eff'
            rate_col = f'detector_{label}_eff_corr'

            if eff_col not in data_df.columns or rate_col not in data_df.columns:
                print(f"[SKIP] Missing columns for {label}")
                continue

            df_valid = data_df[[eff_col, rate_col]].copy()
            mask_valid = np.isfinite(df_valid[eff_col]) & np.isfinite(df_valid[rate_col]) & (df_valid[eff_col] > 1e-8)
            df_valid = df_valid[mask_valid]

            if df_valid.empty:
                print(f"[WARN] No valid data for {label}")
                continue

            eff = df_valid[eff_col].values
            rate_corr = df_valid[rate_col].values
            
            eff_new, res = decorrelate_efficiency_least_change(eff, rate_corr)
            eff_prime_col = f'{eff_col}_decorrelated'

            if eff_new is not None:
                data_df.loc[df_valid.index, eff_prime_col] = eff_new
                r_new = (rate_corr * eff) / eff_new
                cov_post = np.cov(r_new, eff_new)[0, 1]
                print(f"[{label}] Final covariance after correction: {cov_post:.6e}")
            else:
                print(f"[{label}] Optimization failed.")

        data_df = data_df.copy()
        
        for label in detector_labels:
            eff_prime_col = f'detector_{label}_eff_decorrelated'
            rate_col = f'detector_{label}_eff_corr'

            valid = data_df[[eff_prime_col, rate_col]].dropna()
            
            if create_plots or create_essential_plots:
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

        # group_cols = [
        #     ['sensors_ext_Pressure_ext'],
        #     ['sensors_ext_Temperature_ext'],
        #     ['detector_1234_eff_decorrelated', 'detector_1234_eff'],
        #     ['detector_123_eff_decorrelated', 'detector_123_eff'],
        #     ['detector_234_eff_decorrelated', 'detector_234_eff'],
        #     ['detector_12_eff_decorrelated', 'detector_12_eff'],
        #     ['detector_23_eff_decorrelated', 'detector_23_eff'],
        #     ['detector_34_eff_decorrelated', 'detector_34_eff'],
        # ]
        # plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')
        
        
        # --------------------------------------------------------------------------
        # Calculation of new efficiencies ------------------------------------------
        # --------------------------------------------------------------------------

        print("Solving efficiency components per row...")
        solve_eff_components_per_row(data_df)

        group_cols = [
            ['sensors_ext_Pressure_ext'],
            ['sensors_ext_Temperature_ext'],
            ['final_eff_1', 'final_eff_1_decorrelated'],
            ['final_eff_2', 'final_eff_2_decorrelated'],
            ['final_eff_3', 'final_eff_3_decorrelated'],
            ['final_eff_4', 'final_eff_4_decorrelated'],
        ]
        plot_grouped_series(data_df, group_cols, title='OG eff. vs DECORRELATED eff.', plot_after_all=True)
    
    # ------------------------------------------------------------------------------------------------------------------------
    
    data_df[f'definitive_eff_1'] = data_df['final_eff_1_decorrelated']
    data_df[f'definitive_eff_2'] = data_df['final_eff_2_decorrelated']
    data_df[f'definitive_eff_3'] = data_df['final_eff_3_decorrelated']
    data_df[f'definitive_eff_4'] = data_df['final_eff_4_decorrelated']
    
    # -------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------
    # Calculate the fit for the efficiencies ----------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------
    
    if fit_efficiencies:
        
        print("Calculating the fit for the efficiencies.")

        eff_fitting = True
        if eff_fitting:
            for i in range(1, 5):
                eff_col = f'definitive_eff_{i}'
                fit_col = f'eff_fit_{i}'
                filtered_df, fit_func, _ = assign_efficiency_fit( data_df, eff_col, fit_col, case, i, model_type='linear' )
                # plot_combined_efficiency_views(filtered_df, eff_col, fit_func, i)
        
        plot_side_views_all_planes(data_df, planes=[1, 2, 3, 4], model_type='linear')
        
        
        if create_plots:
            print("Creating efficiency comparison scatter plot...")
            fig, ax = plt.subplots(figsize=(10, 7))
            for i in range(1, 5):  # Modules 1 to 4
                ax.scatter(
                    data_df[f'eff_fit_{i}'],
                    data_df[f'definitive_eff_{i}'],
                    alpha=0.5,
                    s=1,
                    label=f'Module {i}',
                    color=f'C{i}'
                )
            # Plot y = x reference line
            ax.plot([low_lim_eff_plot, 1.0], [low_lim_eff_plot, 1.0], 'k--', linewidth=1, label='Ideal (y = x)')
            ax.set_xlabel('Fitted Efficiency')
            ax.set_ylabel('Measured Efficiency')
            ax.set_title('Measured vs Fitted Efficiency for All Modules')
            ax.set_xlim(low_lim_eff_plot, 1.0)
            ax.set_ylim(low_lim_eff_plot, 1.0)
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
        
        data_df = data_df.copy()
        
        data_df[f'unc_definitive_eff_1'] = 1
        data_df[f'unc_definitive_eff_2'] = 1
        data_df[f'unc_definitive_eff_3'] = 1
        data_df[f'unc_definitive_eff_4'] = 1

        if create_plots or create_essential_plots:
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
                ax.set_ylim(low_lim_eff_plot, 1.0)
                ax.grid(True)
                ax.set_title(f'Plane {i}')
                ax.legend(loc='upper left')
                
            # Label the common x-axis at the bottom
            axes[-1].set_xlabel('Time')
            plt.tight_layout()
            if show_plots:
                plt.show()
            elif save_plots:
                new_figure_path = figure_path + f"{fig_idx}" + f"_eff_{case}.png"
                fig_idx += 1
                print(f"Saving figure to {new_figure_path}")
                plt.savefig(new_figure_path, format='png', dpi=300)
            plt.close()
        # else:
            # print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
            # print("\n")
        
        
        # --- Relative residuals and conditional interpolation ---------------------
        max_deviation = 20        # 20 % cutoff

        for i in range(1, 5):
            # ------------------------------------------------------------------ names
            eff_col   = f'definitive_eff_{i}'
            fit_col   = f'eff_fit_{i}'
            rel_u_col = f'relative_residual_unfiltered{i}'
            rel_col   = f'relative_residual_{i}'

            # ------------------------------------------------------------------ vectors
            eff_fit = data_df[fit_col].astype(float)
            eff_ref = data_df[eff_col].astype(float)

            valid_mask = np.isfinite(eff_fit) & np.isfinite(eff_ref) & (eff_ref != 0.0)

            # ------------------------------------------------------------------ residuals
            rel_unf = np.full_like(eff_ref, np.nan, dtype=float)
            rel_unf[valid_mask] = (eff_fit[valid_mask] - eff_ref[valid_mask]) / eff_ref[valid_mask]
            data_df[rel_u_col] = rel_unf

            # ------------------------------------------------------------------ outlier / inlier masks
            outlier_mask = (np.abs(rel_unf) >= max_deviation) & valid_mask
            inlier_mask  = valid_mask & ~outlier_mask

            # Store final residuals: NaN for outliers
            data_df[rel_col] = rel_unf
            data_df.loc[outlier_mask, rel_col] = np.nan

            # ------------------------------------------------------------------ replace efficiency by model prediction
            data_df.loc[outlier_mask, eff_col] = eff_fit[outlier_mask]

            # ------------------------------------------------------------------ (optional) interpolate non-outlier gaps
            interp_mask = data_df[rel_col].isna() & ~outlier_mask
            if interp_mask.any():
                data_df.loc[interp_mask, rel_col] = (
                    data_df[rel_col]
                    .interpolate(method='linear', limit_direction='both', limit_area='inside')
                    .loc[interp_mask]
                )
        # --------------------------------------------------------------------------
        
        
        # if create_plots or create_essential_plots or create_very_essential_plots:
        if create_plots or create_essential_plots or create_very_essential_plots:
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(17, 14), sharex=True)
            for i in range(1, 5):  # Loop from 1 to 4
                
                ax = axes[i-1]  # pick the appropriate subplot
                
                ax.plot(data_df['Time'], data_df[f'relative_residual_unfiltered{i}'], 
                        label=f'Fit relative error, unfiltered - P{i}', color=f'C{i + 8}', alpha=1)
                
                
                ax.plot(data_df['Time'], data_df[f'relative_residual_{i}'], 
                        label=f'Fit relative error, filtered - P{i}', color=f'C{i}', alpha=1)
                
                # Labeling and titles
                ax.set_ylabel('Efficiency residual')
                
                lim = 0.3
                
                ax.set_ylim(-lim, lim)
                ax.grid(True)
                ax.set_title(f'Plane {i}')
                ax.legend(loc='upper left')
                
            # Label the common x-axis at the bottom
            axes[-1].set_xlabel('Time')
            plt.tight_layout()
            if show_plots:
                plt.show()
            elif save_plots:
                new_figure_path = figure_path + f"{fig_idx}" + f"_eff_fit_residual_{case}.png"
                fig_idx += 1
                print(f"Saving figure to {new_figure_path}")
                plt.savefig(new_figure_path, format='png', dpi=300)
            plt.close()
        # else:
            # print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
            # print("\n")
    
    
    # ------------------------------------------------------------------------------------------------------------------------
    
    e1 = data_df['definitive_eff_1']
    e2 = data_df['definitive_eff_2']
    e3 = data_df['definitive_eff_3']
    e4 = data_df['definitive_eff_4']
    
    if use_two_planes_too:
        # Detector 1234
        # 'processed_tt_1234', 'processed_tt_124', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_1234'] = data_df['processed_tt_1234'] + data_df['processed_tt_124'] + data_df['processed_tt_134'] + data_df['processed_tt_14'] 
        data_df['detector_1234'] = data_df['detector_1234']  / ( data_df["number_of_mins"] * 60 )
        
        # Detector 123
        # 'processed_tt_123', 'processed_tt_1234',  
        # 'processed_tt_124', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_123'] = data_df['processed_tt_1234'] + data_df['processed_tt_123'] + \
            data_df['processed_tt_13'] + data_df['processed_tt_134'] + data_df['processed_tt_124'] + data_df['processed_tt_14'] 
        data_df['detector_123'] = data_df['detector_123'] / ( data_df["number_of_mins"] * 60 )

        # Detector 234
        # 'processed_tt_234', 'processed_tt_1234', 'processed_tt_14'
        # 'processed_tt_124', 'processed_tt_24', 'processed_tt_134',
        data_df['detector_234'] = data_df['processed_tt_234'] + data_df['processed_tt_1234'] + \
            data_df['processed_tt_14'] + data_df['processed_tt_124'] + data_df['processed_tt_24'] + data_df['processed_tt_134']
        data_df['detector_234'] = data_df['detector_234'] / ( data_df["number_of_mins"] * 60 )

        # Detector 12
        # 'processed_tt_12', 'processed_tt_123', 'processed_tt_1234', 
        # 'processed_tt_124', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_12'] = data_df['processed_tt_12'] + data_df['processed_tt_123'] + \
            data_df['processed_tt_1234'] + data_df['processed_tt_124'] + data_df['processed_tt_13'] + data_df['processed_tt_134'] + data_df['processed_tt_14'] 
        data_df['detector_12'] = data_df['detector_12'] / ( data_df["number_of_mins"] * 60 )

        # Detector 23
        # 'processed_tt_234', 'processed_tt_123', 'processed_tt_1234', 'processed_tt_23', 
        # 'processed_tt_124', 'processed_tt_24', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_23'] = data_df['processed_tt_234'] + data_df['processed_tt_123'] + \
            data_df['processed_tt_1234'] + data_df['processed_tt_23'] + data_df['processed_tt_124'] + \
                data_df['processed_tt_24'] + data_df['processed_tt_13'] + data_df['processed_tt_134'] + data_df['processed_tt_14']
        data_df['detector_23'] = data_df['detector_23'] / ( data_df["number_of_mins"] * 60 )

        # Detector 34
        # 'processed_tt_234', 'processed_tt_1234',
        # 'processed_tt_124', 'processed_tt_34', 'processed_tt_24', 'processed_tt_134', 'processed_tt_14'
        data_df['detector_34'] = data_df['processed_tt_234'] + data_df['processed_tt_1234'] + \
            data_df['processed_tt_124'] + data_df['processed_tt_34'] + data_df['processed_tt_24'] + \
                data_df['processed_tt_134'] + data_df['processed_tt_14']
        data_df['detector_34'] = data_df['detector_34'] / ( data_df["number_of_mins"] * 60 )
        
        
        # ─────────────────────────────────────────────────────────────────────────────
        # Efficiencies – every term corresponds to one counted coincidence pattern
        # ─────────────────────────────────────────────────────────────────────────────

        # 1234 detector  →  patterns 1234, 124, 134, 14
        data_df['detector_1234_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )

        # 123 detector   →  patterns 1234, 123, 124, 134, 13, 14
        data_df['detector_123_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + e1 *  e2 *  e3 * (1 - e4)                  # 123
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + e1 * (1 - e2) *  e3 * (1 - e4)             # 13
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )

        # 234 detector   →  patterns 1234, 234, 124, 134, 24, 14
        data_df['detector_234_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + (1 - e1) * e2 *  e3 *  e4                  # 234
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + (1 - e1) * e2 * (1 - e3) * e4              # 24
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )

        # 12 detector    →  patterns 1234, 123, 124, 134, 13, 14, 12
        data_df['detector_12_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + e1 *  e2 *  e3 * (1 - e4)                  # 123
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + e1 * (1 - e2) *  e3 * (1 - e4)             # 13
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
            + e1 *  e2 * (1 - e3) * (1 - e4)             # 12
        )

        # 23 detector    →  patterns 1234, 234, 23, 124, 24, 13, 134, 14
        data_df['detector_23_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + (1 - e1) * e2 *  e3 *  e4                  # 234
            + (1 - e1) * e2 *  e3 * (1 - e4)             # 23
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + (1 - e1) * e2 * (1 - e3) * e4              # 24
            + e1 * (1 - e2) *  e3 * (1 - e4)             # 13
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )

        # 34 detector    →  patterns 1234, 234, 124, 134, 34, 24, 14
        data_df['detector_34_eff'] = (
              e1 *  e2 *  e3 *  e4                       # 1234
            + (1 - e1) * e2 *  e3 *  e4                  # 234
            + e1 *  e2 * (1 - e3) * e4                   # 124
            + e1 * (1 - e2) *  e3 *  e4                  # 134
            + (1 - e1) * (1 - e2) * e3 *  e4             # 34
            + (1 - e1) * e2 * (1 - e3) * e4              # 24
            + e1 * (1 - e2) * (1 - e3) * e4              # 14
        )
        
        # Now correcting by efficiency
        data_df['detector_1234_eff_corr'] = data_df['detector_1234'] / data_df['detector_1234_eff']
        data_df['detector_123_eff_corr'] = data_df['detector_123'] / data_df['detector_123_eff']
        data_df['detector_234_eff_corr'] = data_df['detector_234'] / data_df['detector_234_eff']
        data_df['detector_12_eff_corr'] = data_df['detector_12'] / data_df['detector_12_eff']
        data_df['detector_23_eff_corr'] = data_df['detector_23'] / data_df['detector_23_eff']
        data_df['detector_34_eff_corr'] = data_df['detector_34'] / data_df['detector_34_eff']
    
    else:
        
        # ────────────────────────────────────────────────────────────────────────────────
        #  SINGLE-PLANE AND THREE-PLANE TOPOLOGIES ONLY (use_two_planes_too == False)
        # ────────────────────────────────────────────────────────────────────────────────

        # ── Detector 1234 ───────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 124, 134
        data_df['detector_1234'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_1234_eff'] = (
              e1 * e2 *  e3 *  e4          # 1234
            + e1 * e2 * (1-e3) * e4        # 124  (plane-3 missing)
            + e1 * (1-e2) * e3 *  e4       # 134  (plane-2 missing)
        )

        # ── Detector 123 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 123, 124, 134
        data_df['detector_123'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_123']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_123_eff'] = (
              e1 * e2 *  e3 *  e4          # 1234
            + e1 * e2 *  e3 * (1-e4)       # 123  (plane-4 missing)
            + e1 * e2 * (1-e3) * e4        # 124  (plane-3 missing)
            + e1 * (1-e2) * e3 *  e4       # 134  (plane-2 missing)
        )

        # ── Detector 234 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 234, 124, 134
        data_df['detector_234'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_234']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_234_eff'] = (
              e1 *  e2 *  e3 *  e4         # 1234
            + (1-e1)*e2 *  e3 *  e4        # 234  (plane-1 missing)
            + e1 *  e2 * (1-e3) * e4       # 124  (plane-3 missing)
            + e1 * (1-e2)* e3 *  e4        # 134  (plane-2 missing)
        )

        # ── Detector 12 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 123, 124, 134
        data_df['detector_12'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_123']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_12_eff'] = (
              e1 * e2 *  e3 *  e4          # 1234
            + e1 * e2 *  e3 * (1-e4)       # 123
            + e1 * e2 * (1-e3) * e4        # 124
            + e1 * (1-e2)* e3 *  e4        # 134
        )

        # ── Detector 23 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 234, 124, 134
        data_df['detector_23'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_234']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_23_eff'] = (
              e1 *  e2 *  e3 *  e4         # 1234
            + (1-e1)*e2 *  e3 *  e4        # 234
            + e1 *  e2 * (1-e3) * e4       # 124
            + e1 * (1-e2)* e3 *  e4        # 134
        )

        # ── Detector 34 ────────────────────────────────────────────────────────────────
        #  counted patterns: 1234, 234, 124, 134
        data_df['detector_34'] = (
              data_df['processed_tt_1234']
            + data_df['processed_tt_234']
            + data_df['processed_tt_124']
            + data_df['processed_tt_134']
        ) / (data_df["number_of_mins"] * 60)

        data_df['detector_34_eff'] = (
              e1 *  e2 *  e3 *  e4         # 1234
            + (1-e1)*e2 *  e3 *  e4        # 234
            + e1 *  e2 * (1-e3) * e4       # 124
            + e1 * (1-e2)* e3 *  e4        # 134
        )

        # ── Efficiency-corrected rates ────────────────────────────────────────────────
        data_df['detector_1234_eff_corr'] = data_df['detector_1234'] / data_df['detector_1234_eff']
        data_df['detector_123_eff_corr']  = data_df['detector_123']  / data_df['detector_123_eff']
        data_df['detector_234_eff_corr']  = data_df['detector_234']  / data_df['detector_234_eff']
        data_df['detector_12_eff_corr']   = data_df['detector_12']   / data_df['detector_12_eff']
        data_df['detector_23_eff_corr']   = data_df['detector_23']   / data_df['detector_23_eff']
        data_df['detector_34_eff_corr']   = data_df['detector_34']   / data_df['detector_34_eff']
    
    # -------------------------------------------------------------------------------------
    
    
    group_cols = [
        ['detector_1234_eff_corr'],
        ['detector_123_eff_corr', 'detector_234_eff_corr'],
        ['detector_12_eff_corr', 'detector_23_eff_corr', 'detector_34_eff_corr']
    ]

    plot_grouped_series(data_df, group_cols, title=f'Counts per detector, efficiency corrected', plot_after_all=False, sharey_axes = True)
    
    
    # Result after inerpolation
    group_cols = [
        ['sensors_ext_Pressure_ext'],
        ['sensors_ext_Temperature_ext'],
        ['detector_1234_eff_corr'],
        ['detector_123_eff_corr', 'detector_234_eff_corr'],
        ['detector_12_eff_corr', 'detector_23_eff_corr', 'detector_34_eff_corr']
    ]

    plot_grouped_series(data_df, group_cols, title=f'Counts per detector, efficiency corrected, spline filtered', plot_after_all=True, sharey_axes = False)
    
    
    # -------------------------------------------------------------------------------------
    
    
    data_df = data_df.copy()
    
    # Assign to the original dataframe
    data_df[f'detector_1234_eff_corr_{case}'] = data_df['detector_1234_eff_corr']
    data_df[f'detector_123_eff_corr_{case}'] = data_df['detector_123_eff_corr'] 
    data_df[f'detector_234_eff_corr_{case}'] = data_df['detector_234_eff_corr'] 
    data_df[f'detector_12_eff_corr_{case}'] = data_df['detector_12_eff_corr'] 
    data_df[f'detector_23_eff_corr_{case}'] = data_df['detector_23_eff_corr'] 
    data_df[f'detector_34_eff_corr_{case}'] = data_df['detector_34_eff_corr']
    
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
    plot_grouped_series(data_df, group_cols, title='Efficiencies per detector, DECORRELATED', plot_after_all=False)
    
    

    plot_eff_vs_rate_grid(data_df, detector_labels)
    
    data_df = data_df.copy()

    data_df[f'definitive_eff_1_{case}'] = data_df[f'definitive_eff_1']
    data_df[f'definitive_eff_2_{case}'] = data_df[f'definitive_eff_2']
    data_df[f'definitive_eff_3_{case}'] = data_df[f'definitive_eff_3']
    data_df[f'definitive_eff_4_{case}'] = data_df[f'definitive_eff_4']
    
    if case == 'all':
        # Average the four to get a definitive value
        data_df['global_eff'] = (data_df['definitive_eff_1'] + data_df['definitive_eff_2'] +
                                    data_df['definitive_eff_3'] + data_df['definitive_eff_4']) / 4.0
    
    
    print('Efficiency calculations performed.')


print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('-------------------- Atmospheric corrections started -----------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

print('----------------------------------------------------------------------')
print('---------------------- Pressure correction started -------------------')
print('----------------------------------------------------------------------')



# -------------------------------------------------------------------------------
# -------------------------- LOOPING THE DATA -----------------------------------
# -------------------------------------------------------------------------------

regions_to_correct = []
for col in data_df.columns:
    # If the name of the column contains '_eff_corr_', add it to the list
    if '_eff_corr_' in col:
        regions_to_correct.append(col)

# print(regions_to_correct)
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
    I0 = quantile_mean(I)
    unc_I0 = unc_I / np.sqrt( len(I) )  # Uncertainty of the mean
    I_over_I0 = I / I0
    unc_I_over_I0 = I_over_I0 * np.sqrt( (unc_I / I)**2 + (unc_I0 / I0)**2 )

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
        
        create_very_essential_plots = True
        if create_plots or create_very_essential_plots:
            plt.figure()
            if show_errorbar:
                plt.errorbar(df['delta_P'], df['log_I_over_I0'], xerr=abs(df['unc_delta_P']), yerr=abs(df['unc_log_I_over_I0']), fmt='o', label='Data with Uncertainty')
            else:
                plt.scatter(df['delta_P'], df['log_I_over_I0'], label='Data', s=1, alpha=0.5, marker='.')
            
            # Plot the line using provided eta_P instead of fitted values
            plt.plot(df['delta_P'], fit_pressure_model(df['delta_P'], eta_P, set_a), color='blue', label=f'Set Eta: {eta_P:.3f} ± {unc_eta_P:.3f} %/mbar')
            
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
    
    data_df = data_df.copy()


# Convert the list of dictionaries into a DataFrame after the loop
log_delta_I_df = pd.DataFrame(results)

create_very_essential_plots = True
if create_plots or create_essential_plots or create_very_essential_plots:
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

        # Plot the line using eta_P (beta) and eta_P_ordinate (a)
        plt.plot(delta_P, fit_pressure_model(delta_P, eta_P, eta_P_ordinate), label=f"{region} Fit Line", color=f'C{list(log_delta_I_df["Region"]).index(region)}', alpha=0.7)
        
        # Optional: plot with error bars if needed
        if show_errorbar:
            plt.errorbar(delta_P, log_I_over_I0, xerr=abs(unc_delta_P), yerr=abs(unc_log_I_over_I0), fmt='o', label=f'{region} Fit with Errors')

    plt.xlabel('Delta P')
    plt.ylabel('Log (I / I0)')
    plt.ylim(-0.5, 0.5)
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


# Initialize the matrix as a nested dictionary
eta_matrix = {}

for detector_label in detector_labels:
    detector_key = f'detector_{detector_label}'
    eta_matrix[detector_key] = {}
    
    for processing_region in processing_regions:
        col_name = f'{detector_key}_eff_corr_{processing_region}'
        key = f'eta_P_{col_name}'
        
        eta = global_variables.get(key, None)
        if eta is not None:
            eta_matrix[detector_key][processing_region] = eta

# Convert to DataFrame
df_eta = pd.DataFrame.from_dict(eta_matrix, orient='index')

# Optional: sort the columns (regions)
df_eta = df_eta.reindex(sorted(df_eta.columns), axis=1)

print("\nEta Matrix:")
print(df_eta)
        

# ---------------------------------------------------------------------------------------------------

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

        # Plot the line using eta_P (beta) and eta_P_ordinate (a)
        ax.plot(delta_P, fit_pressure_model(delta_P, eta_P, eta_P_ordinate), label=f"{region} Fit Line", color=f'C{idx}', alpha=0.7)
        
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
# else:
#     print("Plotting is disabled. Set `create_plots = True` to enable plotting.")

# ---------------------------------------------------------------------------------------------------


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
        
        # Print the length of the non-NaN DataFrame
        print(f"Length of DataFrame for {region}: {len(df)}")

        if not df.empty:
            X = df[['delta_Tg_over_Tg0', 'delta_Th_over_Th0', 'delta_H_over_H0']]
            y = df['delta_I_over_I0']
            model.fit(X, y)
            A, B, C = model.coef_
            D = model.intercept_
            
            if create_plots or create_essential_plots or create_very_essential_plots:
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

                plt.suptitle(
                    fr'Normalized Correction Coefficients for {region}: '
                    fr'$A$ = {A:.5f}, $B$ = {B:.3f}, $C$ = {C:.3f}, $D$ = {D:.3f}',
                    fontsize=15,
                    y=1.05 )

                plt.tight_layout()
                if show_plots:
                    plt.show()
                elif save_plots:
                    new_figure_path = figure_path + f"{fig_idx}" + f"_{region}_high_order.png"
                    fig_idx += 1
                    print(f"Saving figure to {new_figure_path}")
                    plt.savefig(new_figure_path, format='png', dpi=300)
                plt.close()
            # else:
            #     print("Plotting is disabled. Set `create_plots = True` to enable plotting.")
                
        else:
            print("Fit not done, data empty. Returning NaN.")
            A, B, C, D = np.nan, np.nan, np.nan, np.nan  # Handle case where there are no valid data points
            print("----------------------------------------------------------------------")
        return A, B, C, D
    
    
    print(data_df['pressure_lab'].unique())
    for region in regions_to_correct:
        
        print(region)
        # print(data_df[f'pres_{region}'])
        
        data_df[f'{region}_pressure_corrected'] = data_df[f'pres_{region}']
        
        data_df = data_df.copy()
        
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
        
        global_variables[f'high_order_coeff_Tground_{region}'] = A
        global_variables[f'high_order_coeff_T100mbar_{region}'] = B
        global_variables[f'high_order_coeff_H100mbar_{region}'] = C
        global_variables[f'high_order_coeff_intercept_{region}'] = D
        
        high_order_results = pd.concat([high_order_results, pd.DataFrame({'Region': [region], 'A': [A], 'B': [B], 'C': [C], 'D': [D]})], ignore_index=True)
        
        # Create corrected rate column for the region
        data_df[f'{region}_final_corrected'] = data_df[f'{region}_pressure_corrected'] * (1 - (A * data_df['delta_Tg'] / Tg0 + B * data_df['delta_Th'] / Th0 + C * data_df['delta_H'] / H0 + D))
    
    print("High order correction applied.")
else:
    print("High order correction not applied.")
    for region in regions_to_correct:
        data_df = data_df.copy()
        data_df[f'{region}_final_corrected'] = data_df[f'pres_{region}']



print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')
print('-------------------- Creating rate final plots -----------------------')
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')


group_cols = [
    [  # all directions
        'detector_1234_eff_corr_all_final_corrected',
        'detector_123_eff_corr_all_final_corrected',
        'detector_234_eff_corr_all_final_corrected',
        'detector_12_eff_corr_all_final_corrected',
        'detector_23_eff_corr_all_final_corrected',
        'detector_34_eff_corr_all_final_corrected',
    ],
    [  # Vert direction
        'detector_1234_eff_corr_Vert_final_corrected',
        'detector_123_eff_corr_Vert_final_corrected',
        'detector_234_eff_corr_Vert_final_corrected',
        'detector_12_eff_corr_Vert_final_corrected',
        'detector_23_eff_corr_Vert_final_corrected',
        'detector_34_eff_corr_Vert_final_corrected',
    ],
    [  # North direction
        'detector_1234_eff_corr_North_final_corrected',
        'detector_123_eff_corr_North_final_corrected',
        'detector_234_eff_corr_North_final_corrected',
        'detector_12_eff_corr_North_final_corrected',
        'detector_23_eff_corr_North_final_corrected',
        'detector_34_eff_corr_North_final_corrected',
    ],
    [  # West direction
        'detector_1234_eff_corr_West_final_corrected',
        'detector_123_eff_corr_West_final_corrected',
        'detector_234_eff_corr_West_final_corrected',
        'detector_12_eff_corr_West_final_corrected',
        'detector_23_eff_corr_West_final_corrected',
        'detector_34_eff_corr_West_final_corrected',
    ],
    [  # South direction
        'detector_1234_eff_corr_South_final_corrected',
        'detector_123_eff_corr_South_final_corrected',
        'detector_234_eff_corr_South_final_corrected',
        'detector_12_eff_corr_South_final_corrected',
        'detector_23_eff_corr_South_final_corrected',
        'detector_34_eff_corr_South_final_corrected',
    ],
    [  # East direction
        'detector_1234_eff_corr_East_final_corrected',
        'detector_123_eff_corr_East_final_corrected',
        'detector_234_eff_corr_East_final_corrected',
        'detector_12_eff_corr_East_final_corrected',
        'detector_23_eff_corr_East_final_corrected',
        'detector_34_eff_corr_East_final_corrected',
    ],
]

plot_grouped_series(
    data_df,
    group_cols,
    time_col='Time',
    title='Efficiency-corrected rates by direction and detector',
    plot_after_all=True
)




# Define all relevant column names in the dataset
all_columns = data_df.columns.tolist()

if only_all == False:
    # Extract unique RX regions from the provided list of column names
    rx_region_pattern = re.compile(r"detector_.*?_eff_corr_(R\d\.\d)_final_corrected")
    unique_rx_regions = sorted(set(rx_region_pattern.findall(" ".join(all_columns))))

    # Define all detector groups to consider
    detector_groups = ['1234', '123', '234', '12', '23', '34']

    # Group columns by RX region
    group_cols_rx = []
    for rx in unique_rx_regions:
        group = [f'detector_{dg}_eff_corr_{rx}_final_corrected' for dg in detector_groups]
        group_cols_rx.append(group)

    # Plot using the existing plot_grouped_series function
    plot_grouped_series(
        data_df,
        group_cols_rx,
        time_col='Time',
        title='Efficiency-corrected RX.X region rates by detector group',
        plot_after_all=True
    )

original_df = data_df.copy()

#%%

data_df = original_df.copy()

print("\nBefore dropping:")
print(data_df.columns.to_list())

# drop_prefixes = ("pres_detector_", "processed_tt_", "definitive_", "unc_pres_detector_")
drop_prefixes = ("subdetector_", "pres_detector_", "processed_tt_", "unc_pres_detector_") # Now it keeps efficiencies
drop_suffixes = ("_final_corrected")
cols_to_drop = [c for c in data_df.columns if c.startswith(drop_prefixes) and not c.endswith(drop_suffixes)]
data_df.drop(columns=cols_to_drop, inplace=True)

print("\nMid dropping:")
print(data_df.columns.to_list())

drop_prefixes = ("detector_")
drop_suffixes = ("_final_corrected")
cols_to_drop = [c for c in data_df.columns if c.startswith(drop_prefixes) and not c.endswith(drop_suffixes)]
data_df.drop(columns=cols_to_drop, inplace=True)

print("\nAfter dropping:")
print(data_df.columns.to_list())

# Loop over regions and compute average efficiency across all detectors for that region
for region in processing_regions:
    # Find all columns corresponding to this region and detector types
    region_cols = [
        col for col in data_df.columns
        if col.startswith("detector_") and col.endswith(f"_{region}_final_corrected")
    ]

    if not region_cols:
        print(f"Warning: No columns found for region {region}")
        continue

    # Compute row-wise average ignoring NaNs
    data_df[f"final_{region}"] = data_df[region_cols].mean(axis=1, skipna=True)
    data_df.drop(columns=region_cols, inplace=True)
    print(f"Computed final_{region} from {len(region_cols)} columns.")


print("Final DataFrame columns:")
print(data_df.columns.to_list())

#%%

group_cols = [
['final_R0.0'],
['final_R1.0', 'final_R1.1', 'final_R1.2', 'final_R1.3', 'final_R1.4', 'final_R1.5', 'final_R1.6', 'final_R1.7'], 
['final_R2.0', 'final_R2.1', 'final_R2.2', 'final_R2.3', 'final_R2.4', 'final_R2.5', 'final_R2.6', 'final_R2.7'],
['final_R3.0', 'final_R3.1', 'final_R3.2', 'final_R3.3', 'final_R3.4', 'final_R3.5', 'final_R3.6', 'final_R3.7'], 
]

plot_grouped_series(data_df,
                    group_cols,
                    time_col='Time',
                    title='Final corrected rates',
                    plot_after_all = True)


group_cols = [
    ['final_all'],
    ['final_Vert', 'final_North', 'final_West', 'final_South', 'final_East'], 
    ]

plot_grouped_series(data_df,
                        group_cols,
                        time_col='Time',
                        title='Final corrected rates',
                        plot_after_all = True)


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
# Saving metadata -------------------------------------------------------------
# -----------------------------------------------------------------------------

# Round every value in global_variables that is not a date to 4 significant digits
def round_sig(x, sig=4):
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

for key, value in global_variables.items():
    if not isinstance(value, (datetime, pd.Timestamp)):
        try:
            global_variables[key] = round_sig(value, significant_digits)
        except Exception:
            pass

print(global_variables)

# Construct the new calibration row
new_row = {'Start_Time': start_time, 'End_Time': end_time}

# Add global variables (e.g., counts, sigmoid widths, slopes)
for key, value in global_variables.items():
    new_row[key] = value

# Load or initialize metadata DataFrame
if os.path.exists(csv_path):
    metadata_df = pd.read_csv(csv_path, parse_dates=['Start_Time', 'End_Time'])
else:
    metadata_df = pd.DataFrame(columns=new_row.keys())

# Find full match in both Start_Time and End_Time
match = (
    (metadata_df['Start_Time'] == start_time) &
    (metadata_df['End_Time'] == end_time)
)
existing_row_index = metadata_df[match].index

if not existing_row_index.empty:
    metadata_df.loc[existing_row_index[0]] = new_row
    print(f"Updated existing calibration for time range: {start_time} to {end_time}")
else:
    metadata_df = pd.concat([metadata_df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Added new calibration for time range: {start_time} to {end_time}")

# Sort and save
metadata_df.sort_values(by='Start_Time', inplace=True)

# Put Start_Time and End_Time as first columns
metadata_df = metadata_df[['Start_Time', 'End_Time'] + [col for col in metadata_df.columns if col not in ['Start_Time', 'End_Time']]]

metadata_df.to_csv(csv_path, index=False, float_format='%.5g')
print(f'{csv_path} updated with the calibration summary.')


# -----------------------------------------------------------------------------
# Saving short table ----------------------------------------------------------
# -----------------------------------------------------------------------------

data_df['totally_corrected_rate'] = data_df[f'final_all']
data_df['unc_totally_corrected_rate'] = data_df['final_all'] * 0 +1
# data_df['global_eff'] = data_df['global_eff']
data_df['unc_global_eff'] = data_df['global_eff'] * 0 + 1

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

mark_status_complete(status_csv_path, status_timestamp)

sys.exit(0)
