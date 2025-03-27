#%%

remove_crosstalk = False
crosstalk_limit = 3 #2.5

remove_streamer = False
streamer_limit = 90

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

station = 4

print("Station ", station)

if station == 1:
      # MINGO01
      file_paths = [
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_11.56.46.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_10.54.34.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_09.53.04.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_08.51.18.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_07.49.29.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_06.47.40.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_05.46.19.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_04.45.39.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_03.44.47.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_02.44.23.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_01.43.48.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_00.42.46.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_23.41.09.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_22.39.00.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_21.37.02.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_20.34.57.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_19.33.11.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_18.31.42.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_17.29.42.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_16.28.36.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_15.27.20.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_14.24.38.txt",
      "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_13.22.23.txt"
      ]
elif station == 2:
      # MINGO02
      file_paths = [
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_13.25.45.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_12.36.34.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_11.46.54.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_10.56.21.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_10.05.21.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_09.14.35.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_08.22.58.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_07.31.35.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_06.40.10.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_05.48.39.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_04.57.07.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_04.05.00.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_03.13.36.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_02.22.02.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_01.30.47.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_00.39.27.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_23.47.00.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_22.55.28.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_22.04.01.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_21.12.37.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_20.21.16.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_19.30.11.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_18.39.37.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_17.49.33.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_16.58.48.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_16.08.09.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_15.17.08.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_14.25.49.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_13.34.47.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_12.43.59.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_11.52.37.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_11.01.07.txt"
      ]
elif station == 4:
      # MINGO04
      file_paths = [
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_11.47.20.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_10.13.02.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_08.40.44.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_05.40.55.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_04.12.57.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_02.46.01.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_01.19.30.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_23.53.43.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_22.29.49.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_21.05.48.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_19.41.45.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_18.16.12.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_16.50.21.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_15.24.09.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_13.58.26.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_12.34.43.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_11.13.34.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_09.55.03.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_08.37.03.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_07.19.18.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_06.02.12.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_04.45.19.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_03.28.00.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_02.10.17.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_00.52.21.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.18_22.18.57.txt",
          "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO04/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.18_21.05.01.txt"
      ]
else:
      print("Station not defined")

# Read and concatenate all files
df_list = [pd.read_csv(file, delimiter=",") for file in file_paths]  # Adjust delimiter if needed
merged_df = pd.concat(df_list, ignore_index=True)

# Drop duplicates if necessary
merged_df.drop_duplicates(inplace=True)

# Print the column names
print(merged_df.columns.to_list())

# If any value in any column that has Q* in it is smaller than 2.5, put it to 0
figure_save_path = "/home/cayetano/DATAFLOW_v3/NEW_CHARGE_ANALYSIS"
if remove_crosstalk or remove_streamer:
    if remove_crosstalk:
        figure_save_path += "_no_crosstalk"
        for col in merged_df.columns:
            if "Q_" in col and "s" in col:
                merged_df[col] = merged_df[col].apply(lambda x: 0 if x < crosstalk_limit else x)
                      
    if remove_streamer:
        figure_save_path += "_no_streamer"
        for col in merged_df.columns:
            if "Q_" in col and "s" in col:
                merged_df[col] = merged_df[col].apply(lambda x: 0 if x > streamer_limit else x)
figure_save_path += "/"        

# Check if figures_save_path exists, create one in other case
import os
print("Storing in ", figure_save_path)
if not os.path.exists(figure_save_path):
      os.makedirs(figure_save_path)

# Create a 4x4 subfigure
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i in range(1, 5):
      for j in range(1, 5):
            # Get the column name
            col_name = f"Q_M{i}s{j}"
            
            # Plot the histogram
            v = merged_df[col_name]
            v = v[v != 0]
            axs[i-1, j-1].hist(v, bins=200, range=(0, 100))
            axs[i-1, j-1].set_title(col_name)
            axs[i-1, j-1].set_xlabel("Charge")
            axs[i-1, j-1].set_ylabel("Frequency")
            axs[i-1, j-1].grid(True)

plt.tight_layout()
figure_name = "pre_cal_all_channels.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
# plt.show()
plt.close()

# Create a 4x4 subfigure
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i in range(1, 5):
      for j in range(1, 5):
            # Get the column name
            col_name = f"Q_M{i}s{j}"
            
            # Plot the histogram
            v = merged_df[col_name]
            v = v[v != 0]
            axs[i-1, j-1].hist(v, bins=200, range=(0, 5))
            axs[i-1, j-1].set_title(col_name)
            axs[i-1, j-1].set_xlabel("Charge")
            axs[i-1, j-1].set_ylabel("Frequency")
            axs[i-1, j-1].grid(True)

plt.tight_layout()
figure_name = "zoom_pre_cal_all_channels.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
# plt.show()
plt.close()

FEE_calibration = {
    "Width": [
        0.0000001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
        160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
        300, 310, 320, 330, 340, 350, 360, 370, 380, 390
    ],
    "Fast Charge": [
        4.0530E+01, 2.6457E+02, 4.5081E+02, 6.0573E+02, 7.3499E+02, 8.4353E+02,
        9.3562E+02, 1.0149E+03, 1.0845E+03, 1.1471E+03, 1.2047E+03, 1.2592E+03,
        1.3118E+03, 1.3638E+03, 1.4159E+03, 1.4688E+03, 1.5227E+03, 1.5779E+03,
        1.6345E+03, 1.6926E+03, 1.7519E+03, 1.8125E+03, 1.8742E+03, 1.9368E+03,
        2.0001E+03, 2.0642E+03, 2.1288E+03, 2.1940E+03, 2.2599E+03, 2.3264E+03,
        2.3939E+03, 2.4625E+03, 2.5325E+03, 2.6044E+03, 2.6786E+03, 2.7555E+03,
        2.8356E+03, 2.9196E+03, 3.0079E+03, 3.1012E+03
    ]
}

# Create the DataFrame
FEE_calibration = pd.DataFrame(FEE_calibration)

from scipy.interpolate import CubicSpline

# Convert to NumPy arrays for interpolation
width_table = FEE_calibration['Width'].to_numpy()
fast_charge_table = FEE_calibration['Fast Charge'].to_numpy()

# Create a cubic spline interpolator
cs = CubicSpline(width_table, fast_charge_table, bc_type='natural')

def interpolate_fast_charge(width):
    """
    Interpolates the Fast Charge for given Width values using cubic spline interpolation.
    Parameters:
    - width (float or np.ndarray): The Width value(s) to interpolate in ns.
    Returns:
    - float or np.ndarray: The interpolated Fast Charge value(s) in fC.
    """
    width = np.asarray(width)  # Ensure input is a NumPy array

    # Keep zero values unchanged
    result = np.where(width == 0, 0, cs(width))

    return result


columns_to_drop = ['Time', 'CRT_avg', 'x', 'y', 'theta', 'phi', 't0', 's', 'type', 'charge_event']
merged_df = merged_df.drop(columns=columns_to_drop)

print(merged_df.columns.to_list())

# For all the columns apply the calibration and not change the name of the columns
for col in merged_df.columns:
    merged_df[col] = interpolate_fast_charge(merged_df[col])


# ---------------------------------------------------------------------------------------------------------------------------------
# Part 2. Charge ------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# The columns of charge are called Q_MXsY, where X and Y go from 1 to 4. X is the module number, Y is the strip number
# First, plot in a 4x4 subfigure histograms between 0 and 80 for each of the 16 strips

# Create a 4x4 subfigure
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i in range(1, 5):
      for j in range(1, 5):
            # Get the column name
            col_name = f"Q_M{i}s{j}"
            
            # Plot the histogram
            v = merged_df[col_name]
            v = v[v != 0]
            axs[i-1, j-1].hist(v, bins=200, range=(0, 1500))
            axs[i-1, j-1].set_title(col_name)
            axs[i-1, j-1].set_xlabel("Charge")
            axs[i-1, j-1].set_ylabel("Frequency")
            axs[i-1, j-1].grid(True)

plt.tight_layout()
figure_name = "all_channels.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
# plt.show()
plt.close()

# %%

# For each module, calculate the total charge per event, then store them in a dataframe
total_charge = pd.DataFrame()
for i in range(1, 5):
    total_charge[f"Q_M{i}"] = merged_df[[f"Q_M{i}s{j}" for j in range(1, 5)]].sum(axis=1)

# %%

# POLYA FIT TEST

module = 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit

# Constants
q_e = 1.602e-4  # fC (charge of 1 electron)

# Polya model adapted to charge
def polya_induced_charge(Q, theta, nbar, alpha, A, offset):
    n = Q * alpha + offset
    norm = ((theta + 1) ** (theta + 1)) / gamma(theta + 1)
    return A * norm * (n / nbar)**theta * np.exp(-(theta + 1) * n / nbar)

# Get the data (flatten just in case)
data = total_charge[f"Q_M{module}"].dropna().to_numpy().flatten()
data = data[data != 0]

data = data / q_e

# Plot histogram
# counts, bin_edges = np.histogram(data, bins=200, range=(0, 1500))
counts, bin_edges = np.histogram(data, bins=100, range=(0, 1.1e7))
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.figure(figsize=(6, 5))
plt.plot(bin_centers, counts, label="Original data")

# Filter valid bins
counts = np.asarray(counts).flatten()
bin_centers = np.asarray(bin_centers).flatten()

bin_center = bin_centers[counts >= 0.08 * max(counts)][0]

print(bin_center)

# mask = (bin_centers > bin_center) & (counts >= 0.1 * max(counts))
mask = (bin_centers > bin_center) & (counts > 0)
x_fit = bin_centers[mask]
y_fit = counts[mask]

# Initial guesses: theta, nbar, alpha, A, offset
p0 = [1, 1e6, 0.6, max(counts), 1e6]
bounds = ([0, 0, 0, 0, -1e16], [100, 1e16, 1, 1e16, 1e16])  # theta ≥ 0.01, nbar ≥ 1, alpha in [0.01, 1]

# Fit
popt, _ = curve_fit(polya_induced_charge, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=100000)
theta_fit, nbar_fit, alpha_fit, A_fit, offset_fit = popt

plt.plot(x_fit, y_fit, "o", label="Data given to the fit")

# Plot results
# plt.hist(data, bins=70, range=(0, 1500), histtype="step", label="Data", linewidth=1.5, density = False)

plt.close()

x_fine = np.linspace(0, 1.1e7, 300)
# plt.plot(
#       x_fine,
#       polya_induced_charge(x_fine, *popt),
#       "r--",
#       # theta_fit, nbar_fit, alpha_fit, A_fit, offset_fit = popt
#       # theta_fit, nbar_fit, alpha_fit, A_fit, offset_fit = popt
#       label = (rf"Fit: $\theta = {theta_fit:.2f},\ \bar{{n}} = {nbar_fit:.0f},\ "
#          rf"\alpha = {alpha_fit:.2f},\ A = {A_fit:.2f},\ \mathrm{{offset}} = {offset_fit:.2f},\ "
#          rf"\bar{{n}} / \alpha = {nbar_fit / alpha_fit:.3g}$")
# )

# plt.xlabel("Induced equivalent electrons")
# plt.ylabel("Entries")
# plt.title(f"Polya Fit: total charge in plane {module}")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Compute residuals on original data points (not x_fine)
residuals = y_fit - polya_induced_charge(x_fit, *popt)
residuals_norm = residuals / y_fit * 100

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [4, 1, 1]})

# --- Top plot: data and fit ---
ax1.plot(
    x_fit,
    y_fit,
    "r--",
    label = (rf"Fit: $\theta = {theta_fit:.2f},\ \bar{{n}} = {nbar_fit:.0f},\ "
             rf"\alpha = {alpha_fit:.2f},\ A = {A_fit:.2f},\ \mathrm{{offset}} = {offset_fit:.2f},\ "
             rf"\bar{{n}} / \alpha = {nbar_fit / alpha_fit:.3g}$")
)
ax1.plot(x_fit, y_fit, 'bo', label="Data")
ax1.set_ylabel("Entries")
ax1.set_title(f"Polya Fit: total charge in plane {module}")
ax1.legend()
ax1.grid(True)

ax2.axhline(0, color='gray', linestyle='--')
ax2.plot(x_fit, residuals)
ax2.set_xlabel("Induced equivalent electrons")
ax2.set_ylabel("Residuals (%)")
ax2.grid(True)

# --- Bottom plot: residuals ---
ax3.axhline(0, color='gray', linestyle='--')
ax3.plot(x_fit, residuals_norm)
ax3.set_xlabel("Induced equivalent electrons")
ax3.set_ylabel("Residuals (%)")
ax3.grid(True)

plt.tight_layout()
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit

# Constants
q_e = 1.602e-4  # fC

# Polya model
def polya_induced_charge(Q, theta, nbar, alpha, A, offset):
    n = Q * alpha + offset
    norm = ((theta + 1) ** (theta + 1)) / gamma(theta + 1)
    return A * norm * (n / nbar)**theta * np.exp(-(theta + 1) * n / nbar)

# Prepare figure
fig, axs = plt.subplots(
    3, 4, figsize=(17, 5), sharex='col', 
    gridspec_kw={'height_ratios': [4, 1, 1]}
)

for idx, module in enumerate(range(1, 5)):

    # Load and preprocess data
    data = total_charge[f"Q_M{module}"].dropna().to_numpy().flatten()
    data = data[data != 0] / q_e  # convert to e–

    # Histogram
    counts, bin_edges = np.histogram(data, bins=100, range=(0, 1.1e7))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    bin_center = bin_centers[counts >= 0.05 * max(counts)][0]
    mask = (bin_centers > bin_center) & (counts > 0)
    x_fit = bin_centers[mask]
    y_fit = counts[mask]

    # Fit
    p0 = [1, 1e6, 0.6, max(counts), 1e6]
    bounds = ([0, 0, 0, 0, -1e16], [100, 1e16, 1, 1e16, 1e16])
    popt, _ = curve_fit(polya_induced_charge, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=100000)
    theta_fit, nbar_fit, alpha_fit, A_fit, offset_fit = popt

    # Fine x for fit curve
    x_fine = np.linspace(0, 1.1e7, 300)
    y_model = polya_induced_charge(x_fine, *popt)

    # Residuals
    residuals = y_fit - polya_induced_charge(x_fit, *popt)
    residuals_norm = residuals / y_fit * 100

    # Plot index
    ax1 = axs[0, idx]
    ax2 = axs[1, idx]
    ax3 = axs[2, idx]

    # --- Fit plot ---
#     plot_label = (
#         rf"$\theta={theta_fit:.2f},\ \bar{{n}}={nbar_fit:.0f},\ "
#         rf"\alpha={alpha_fit:.2f},\ A={A_fit:.2f},\ \mathrm{{off}}={offset_fit:.2f},\ "
#         rf"\bar{{n}}/\alpha={nbar_fit / alpha_fit:.3g}$"
#     )
    plot_label = (
        rf"$\theta={theta_fit:.2f},\ \mathrm{{off}}={offset_fit:.2f},\ "
        rf"\bar{{n}}/\alpha={nbar_fit / alpha_fit:.3g}$"
    )
    ax1.plot(x_fine, y_model, "r--", label = plot_label)
    ax1.plot(x_fit, y_fit, 'bo', markersize = 2)
    ax1.set_title(f"Module {module}")
    ax1.legend(fontsize=8)
    ax1.grid(True)
    if idx == 0:
        ax1.set_ylabel("Entries")

    # --- Residuals ---
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.plot(x_fit, residuals, 'k.')
    if idx == 0:
        ax2.set_ylabel("Res.")

    ax2.grid(True)

    # --- Normalized residuals ---
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.plot(x_fit, residuals_norm, 'k.')
    if idx == 0:
        ax3.set_ylabel("Res. (%)")
    ax3.set_xlabel("Induced equivalent electrons")
    ax3.set_ylim(-10, 100)
    ax3.grid(True)

plt.tight_layout()
plt.show()
