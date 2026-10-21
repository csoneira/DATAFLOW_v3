#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:47:10 2024

@author: cayesoneira
"""

globals().clear()

import pickle
import numpy as np
import os
import shutil
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
from PyPDF2 import PdfMerger
from scipy.stats import norm
import glob

pdf_files = glob.glob(os.path.join(".", '*.pdf'))
for pdf_file in pdf_files:
    try:
        os.remove(pdf_file)
        print(f'Removed: {pdf_file}')
    except Exception as e:
        print(f'Error removing {pdf_file}: {e}')

png_files = glob.glob(os.path.join(".", '*.png'))
for png_file in png_files:
    try:
        os.remove(png_file)
        print(f'Removed: {png_file}')
    except Exception as e:
        print(f'Error removing {png_file}: {e}')

output_order = 0

def res_summary(data):
    data = data[pd.notna(data)]
    
    summary = {
        'Minimum': np.min(data),
        '0.05 quantile': np.percentile(data, 5),
        'Median': np.median(data),
        'Mean': np.mean(data),
        '0.95 quantile': np.percentile(data, 95),
        'Maximum': np.max(data),
        'Standard Deviation': np.std(data)
    }
    return summary

def hist_1d_summary(vdat, bin_number, title, axis_label, name_of_file):
    global output_order
    v = (8, 5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)
    
    stats = res_summary(vdat)
    
    the_label = f"All hits, {len(vdat)} events\n\
Mean = {stats['Mean']:.2g}\n\
Median = {stats['Median']:.2g}\n\
Standard Deviation = {stats['Standard Deviation']:.2g}"
    
    n, bins, patches = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
                               label=the_label, density=False)
    ax.legend()
    
    ax.set_title(f"Fig. {output_order}, {title}")
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order += 1
    plt.show()
    plt.close()
    
    
def scatter_2d(xdat, ydat, title, x_label, y_label, name_of_file):
    global output_order
    
    plt.close()
    
    # fig = plt.figure(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    plt.figure(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    
    plt.scatter(xdat, ydat, s=1)
    plt.title(f"Fig. {output_order}, {title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.axis("equal")
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    plt.show()
    plt.close()
    output_order = output_order + 1
    return


def hist_2d_hex(xdat, ydat, x_bins, y_bins, title, x_label, y_label, name_of_file):
    global output_order
    
    # print("xdat dimension:", xdat.shape)
    # print("ydat dimension:", ydat.shape)
    
    # Filtering ydat based on xdat
    ydat_filt = ydat[(xdat > x_bins[0]) & (xdat < x_bins[-1])]
    xdat_filt = xdat[(xdat > x_bins[0]) & (xdat < x_bins[-1])]
    ydat_filt_filt = ydat_filt[(ydat_filt > y_bins[0]) & (ydat_filt < y_bins[-1])]
    xdat_filt_filt = xdat_filt[(ydat_filt > y_bins[0]) & (ydat_filt < y_bins[-1])]
    
    xdat = xdat_filt_filt
    ydat = ydat_filt_filt
    
    hex_grid = int( 2/3 * np.sqrt( len(x_bins) * len(y_bins) ) )
    
    fig, ax = plt.subplots(figsize=(8, 5))
    # plt.hexbin(xdat, ydat, gridsize=hex_grid, cmap='viridis', marginals = True)
    plt.hexbin(xdat, ydat, gridsize=hex_grid, cmap='viridis')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_aspect("equal")
    ax.set_facecolor('#440454')
    ax.set_title(f"Fig. {output_order}, {title}, {len(xdat)} counts")
    plt.colorbar(label='Counts')
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return


from scipy.optimize import curve_fit

def exp_decay_poly(x, a, b, c, d, k):
    return (a * x**3 + b * x**2 + c * x + d) * np.exp(-k * x)

def scatter_2d_and_fit(xdat, ydat, title, x_label, y_label, name_of_file):
    global output_order
    save_charge_strip_calibration_figures = True

    residue_limit = 100

    # Filter data for plot and pre-fit
    mask = (np.abs(xdat) < residue_limit) & (np.abs(ydat) < residue_limit)
    xdat_plot = xdat[mask]
    ydat_plot = ydat[mask]

    # Fitting the data to the exponential decay polynomial model
    initial_guess = [1e-3, 1e-3, 1e-3, 1e-3, 0.01]  # Initial guess for the parameters
    params, cov = curve_fit(exp_decay_poly, xdat_plot, ydat_plot, p0=initial_guess)
    a, b, c, d, k = params

    # Compute fitted values
    x_fit = np.linspace(min(xdat_plot), max(xdat_plot), 100)
    y_fit = exp_decay_poly(x_fit, *params)

    # Calculate corrected Y values
    y_final = ydat_plot - exp_decay_poly(xdat_plot, *params)

    if save_charge_strip_calibration_figures:
        plt.close()
        plt.figure(figsize=(16,6))
        plt.scatter(xdat_plot, ydat_plot, s=1, label=f"{len(xdat_plot)} events")
        plt.scatter(xdat_plot, y_final, s=1, color="green", label="Calibrated points")
        plt.plot(x_fit, y_fit, 'r-', label=f'Exp Decay Poly Fit: a={a:.2g}, b={b:.2g}, c={c:.2g}, d={d:.2g}, k={k:.2g}')
        plt.title(f"Fig. {output_order}, {title}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([-2, 100])
        plt.ylim([-5, 5])
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_order}_{name_of_file}.png", format="png")
        if show_plots: plt.show()
        plt.close()
        output_order += 1

    return params


def load_all_arrays(filename):
    # Load the entire file into a large array
    large_array = np.loadtxt(filename, delimiter=',')
    # Number of 4x4 arrays to extract
    num_arrays = large_array.shape[0] // 4
    
    # Extract each 4x4 array
    arrays = [large_array[i*4:(i+1)*4] for i in range(num_arrays)]
    return arrays

# Load the arrays back into a list of numpy arrays
mcharge = load_all_arrays('mcharge.txt')
mcharge_slew = load_all_arrays('mcharge_slew.txt')
mcharge_slew = np.array(mcharge_slew)


mfit = np.loadtxt('mfit.txt', delimiter=',')
vchi = np.loadtxt('vchi.txt', delimiter=',')
mch2 = np.loadtxt('mch2.txt', delimiter=',')
vcsf = np.loadtxt('vcsf.txt', delimiter=',')
mres = np.loadtxt('mres.txt', delimiter=',')
    
show_plots = True

bins_y = np.linspace(-150, 150, 200)
bins_sum = np.linspace(-4, 4, 200)
bins_dif = np.linspace(-0.5, 0.5, 200)
bins_charge_slew = np.linspace(0, 75, 100)

if show_plots:
    for T in range(1,5):
        for s in range(1,5):
            charge_slewing = mcharge_slew[ :, T-1, s-1 ][(mres[:, 1] == T) & (mres[:, 2] == s)]
            mres_Ts = mres[(mres[:, 1] == T) & (mres[:, 2] == s)]
            
            res_y0 = mres_Ts[:,3]
            res_tsum = mres_Ts[:,4]
            res_tdif = mres_Ts[:,5]
            
            # scatter_slew_2d(res_tsum, charge_slewing, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            # scatter_2d_and_fit(charge_slewing, res_tsum, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            hist_1d_summary(res_tsum, len(bins_sum), "OG Tsum", "Charge", "name_of_file")
            hist_2d_hex(res_tsum, charge_slewing, bins_sum, bins_charge_slew, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            # hist_2d_hex(res_tdif, charge_slewing, bins_dif, bins_charge_slew, f"Charge vs. Residue T diff, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            # hist_2d_hex(res_y0, charge_slewing, bins_y, bins_charge_slew, f"Charge vs. Residue T diff, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")

slew_corr = [[None for _ in range(4)] for _ in range(4)]
if show_plots:
    for T in range(1,5):
        for s in range(1,5):
            charge_slewing = mcharge_slew[ :, T-1, s-1 ][(mres[:, 1] == T) & (mres[:, 2] == s)]
            mres_Ts = mres[(mres[:, 1] == T) & (mres[:, 2] == s)]
            
            res_y0 = mres_Ts[:,3]
            res_tsum = mres_Ts[:,4]
            res_tdif = mres_Ts[:,5]
            
            # scatter_slew_2d(res_tsum, charge_slewing, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            slew_corr[T-1][s-1] = scatter_2d_and_fit(charge_slewing, res_tsum, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            # hist_2d_hex(res_tsum, charge_slewing, bins_sum, bins_charge_slew, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            # hist_2d_hex(res_tdif, charge_slewing, bins_dif, bins_charge_slew, f"Charge vs. Residue T diff, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")



def save_slew_corr_to_txt(slew_corr, filename):
    with open(filename, 'w') as f:
        for row in slew_corr:
            for params in row:
                if params is not None:
                    # Join the parameters with a comma and write to file
                    param_line = ','.join(map(str, params)) + '\n'
                    f.write(param_line)
                else:
                    # Write a placeholder for None or missing data
                    f.write('None\n')

# Example usage:
save_slew_corr_to_txt(slew_corr, 'slew_corr.txt')

def load_slew_corr_from_txt(filename):
    slew_corr = [[None for _ in range(4)] for _ in range(4)]
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Index for placing parameters into the matrix
    i = 0
    j = 0
    for line in lines:
        line = line.strip()
        if line != 'None':
            params = list(map(float, line.split(',')))
            slew_corr[i][j] = params
        else:
            slew_corr[i][j] = None
        
        # Update indices
        j += 1
        if j == 4:
            j = 0
            i += 1

    return slew_corr


def scatter_2d_double(xdat1, ydat1, xdat2, ydat2, title, x_label, y_label, name_of_file):
    global output_order
    
    plt.close()
    
    plt.figure(figsize=(8, 5))
    
    # Plot the first set of data
    plt.scatter(xdat1, ydat1, s=1, label='Data Set 1')
    
    # Plot the second set of data
    plt.scatter(xdat2, ydat2, s=1, label='Data Set 2', color='r')
    
    plt.title(f"Fig. {output_order}, {title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Add a legend
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    plt.show()
    plt.close()
    
    output_order += 1
    return


# Example usage:
loaded_slew_corr = load_slew_corr_from_txt('slew_corr.txt')

if show_plots:
    for T in range(1,5):
        for s in range(1,5):
            charge_slewing = mcharge_slew[ :, T-1, s-1 ][(mres[:, 1] == T) & (mres[:, 2] == s)]
            mres_Ts = mres[(mres[:, 1] == T) & (mres[:, 2] == s)]
            
            res_y0 = mres_Ts[:,3]
            res_tsum = mres_Ts[:,4]
            res_tdif = mres_Ts[:,5]
            
            params = loaded_slew_corr[T-1][s-1]
            a, b, c, d, k = params
            res_tsum_cal = res_tsum - exp_decay_poly(charge_slewing, a, b, c, d, k)
            
            # scatter_slew_2d(res_tsum, charge_slewing, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            hist_1d_summary(res_tsum, len(bins_sum), "Slew corrected Tsum", "Charge", "name_of_file")
            hist_1d_summary(res_tsum_cal, len(bins_sum), "Slew corrected Tsum", "Charge", "name_of_file")
            hist_2d_hex(res_tsum, charge_slewing, bins_sum, bins_charge_slew, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            hist_2d_hex(res_tsum_cal, charge_slewing, bins_sum, bins_charge_slew, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            scatter_2d_double(res_tsum, charge_slewing, res_tsum_cal, charge_slewing, "title", "T sum residue", "Charge", "name_of_file")

