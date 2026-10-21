#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:22:14 2024

@author: cayesoneira
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PyPDF2 import PdfMerger

# Parameters ------------------------------------------------------------------
# Energy filtering
low_lim_ch1 = 21; up_lim_ch1 = 140
low_lim_ch2 = 60; up_lim_ch2 = 140
low_lim_ch3 = 20; up_lim_ch3 = 140
# Histograms
bins_ch1 = up_lim_ch1 - low_lim_ch1 - 2
bins_ch2 = up_lim_ch2 - low_lim_ch2 - 2
bins_ch3 = up_lim_ch3 - low_lim_ch3 - 2
# Neutron rates
interval_min = 60
# Plots
show_plots = False

# Setting up ------------------------------------------------------------------
global output_order
output_order = 4
filename = "neutron_time_energy.dat"

# Delete pdf files
current_directory = os.getcwd()
files_in_current_directory = os.listdir(current_directory)
pdf_files = [file for file in files_in_current_directory if file.lower().endswith('.pdf')]

if pdf_files:
    # Delete each PDF file
    for pdf_file in pdf_files:
        file_path = os.path.join(current_directory, pdf_file)
        os.remove(file_path)
        print(f"Deleted: {pdf_file}")
else:
    print("No PDF files found in the current directory.")

# Define some functions -------------------------------------------------------
def hist_1d(vdat, bin_number, channel, filtered):
    global output_order
    # plt.close()
    v=(8,5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)
    
    axis_label = "Energy / AU"
    
    if filtered:
        title = f"Filtered energy channel {channel}"
        name_of_file = f"ch{channel}_filt_energy_histogram"
    else:
        title = f"Energy for channel {channel}"
        name_of_file = f"ch{channel}_energy_histogram"
    
    # Plot histograms on the single axis
    ax.hist(vdat, bins=bin_number, alpha=0.5, color="red", \
            label=f"All hits, {len(vdat)} events", density = False)
    ax.legend()
    # ax.set_xlim(-5, x_axes_limit_plots)
    ax.set_title(f"Fig. {output_order}, {title}")
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    # plt.xlim([log_low_lim_fig, None])
    # plt.xscale("log"); plt.yscale("log");
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    if show_plots: plt.show(); plt.close()
    return

def time_plot(data, interval, channel):
    # Some options ------------------------------------------------------------
    size = 5
    v=(14,4)
    rel_err = 0.5
    remove_outliers = False
    x_min = None
    x_max = None
    location = "Madrid"
    # -------------------------------------------------------------------------
    
    rates = []
    dates = []

    n = 1

    while (n-1) * interval < data[:,0][-1]:
        # Find data points within the current interval
        interval_data = data[(data[:, 0] >= data[0, 0] + (n-1)*interval) & (data[:, 0] <= data[0, 0] + n*interval)]
        # print(interval_data)
        
        if len(interval_data) > 1:
            # Calculate time interval
            time_diff = interval_data[-1, 0] - interval_data[0, 0]
            # print(time_diff)

            # Calculate number of counts within the interval
            counts_within_interval = len(interval_data)
            # print(counts_within_interval)

            # Calculate rate (number of counts per unit time)
            rate = counts_within_interval / time_diff
            rates.append(rate)
            dates.append(interval_data[0, 0])
            
            # print(n)
            # print(rate)    
        n += 1
    
    plotted_rates = rates
    plotted_dates = dates
    
    big_size = 10 * size
    plt.figure(figsize=v)
    color='black'
    plt.grid()
    plt.xlabel('Timestamp / h')
    plt.ylabel('Neutron rate / (1/hr)')
    # Calculate the average and tolerances
    average_value = np.nanmean(plotted_rates)
    
    # Removing outliers
    if remove_outliers:
        criteria = abs(plotted_rates - average_value) / average_value < rel_err
        plotted_rates = plotted_rates[criteria]
        plotted_dates = plotted_dates[criteria]
        average_value = np.mean(plotted_rates)
    
    shaded_region_1pc = 0.05 * average_value
    shaded_region_5pc = 0.2 * average_value
    
    # Plot the raw data
    plt.scatter(plotted_dates, plotted_rates, marker='o', s=big_size, color="blue", alpha=0.5,\
                label = f'Neutron rate (mean is {average_value:4g} / hr)')
    
    # Fill the regions around the average
    plotted_dates_sorted = sorted(plotted_dates)
    plt.fill_between(plotted_dates_sorted, average_value - shaded_region_1pc, average_value + shaded_region_1pc, color='green', alpha=0.3, label='5% tolerance around mean')
    plt.fill_between(plotted_dates_sorted, average_value - shaded_region_5pc, average_value + shaded_region_5pc, color='red', alpha=0.08, label='20% tolerance around mean')
    
    plt.tick_params(axis='y', labelcolor=color)
    if x_min != x_max: plt.xlim(x_min, x_max)
    plt.ylim(0.6 * average_value, 1.4 * average_value)
    plt.xticks(rotation=45)
    
    if interval >= 1:
        parenthesis = f"{interval:.2f} h average"
    else:
        parenthesis = f"{interval*60:.0f} min average"
    
    plt.title(f'{location} - Neutron Ch{channel} rate ({parenthesis}) evolution from\n{x_min} to {x_max}')
    plt.legend()
    
    # Save and optionally show the plot
    name = f'tolerances_neutron_rate_{interval:.2f}'
    
    plt.tight_layout()
    plt.savefig(f'{channel}_{location}_{name}_from_{x_min}_to_{x_max}.pdf', format="pdf")
    if show_plots:
        plt.show()
    plt.close()

# Reading the file ------------------------------------------------------------
dat = []
with open(filename, 'r') as file:
    for line in file:
        # Split the line into individual values and convert them to floats
        values = [float(x) for x in line.split()]
        # Convert the list of values into a NumPy array
        matrix_data = np.array(values)
        dat.append(matrix_data)

# Creating the data arrays per channel ----------------------------------------
ch1 = []
ch2 = []
ch3 = []
for row in dat:
    row[1] = row[1] / (10 ** 12) / 3600 # In seconds
    if row[0] == 0: ch1.append(row[1:3])
    if row[0] == 1: ch2.append(row[1:3])
    if row[0] == 2: ch3.append(row[1:3])
ch1 = np.array(ch1)
ch2 = np.array(ch2)
ch3 = np.array(ch3)

# Energy histograms -----------------------------------------------------------
hist_1d(ch1[:,1], 512, "1", False)
hist_1d(ch2[:,1], 512, "2", False)
hist_1d(ch3[:,1], 512, "3", False)

# Energy filtering ------------------------------------------------------------
condition = (ch1[:,1] > low_lim_ch1) & (ch1[:,1] < up_lim_ch1)
ch1 = ch1[condition]
condition = (ch2[:,1] > low_lim_ch2) & (ch2[:,1] < up_lim_ch2)
ch2 = ch2[condition]
condition = (ch3[:,1] > low_lim_ch3) & (ch3[:,1] < up_lim_ch3)
ch3 = ch3[condition]

# Energy histograms after filtering -------------------------------------------
hist_1d(ch1[:,1], bins_ch1, "1", True)
hist_1d(ch2[:,1], bins_ch2, "2", True)
hist_1d(ch3[:,1], bins_ch3, "3", True)

# Neutron rates ---------------------------------------------------------------
time_plot(ch1, interval_min/60, "1")
time_plot(ch2, interval_min/60, "2")
time_plot(ch3, interval_min/60, "3")

# -----------------------------------------------------------------------------
# PDF report creation ---------------------------------------------------------
# -----------------------------------------------------------------------------

filename = "0_neutron_report.pdf"

if os.path.exists(filename):
    os.remove(filename)
    print(f"{filename} has been deleted.")
else:
    print(f"{filename} does not exist, so it is not replaced, but created.")

x = [a for a in os.listdir() if a.endswith(".pdf")]

merger = PdfMerger()

y = sorted(x, key=lambda s: int(s.split("_")[0]))
# y = x

for pdf in y:
    merger.append(open(pdf, "rb"))

with open(filename, "wb") as fout:
    merger.write(fout)

print("----------------------------------------------------------------------------")
print(f"Report stored as {filename}")