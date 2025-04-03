#%%

remove_crosstalk = True
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

# Initialize dictionaries to store charge distributions
singles = {f'single_M{i}_s{j}': [] for i in range(1, 5) for j in range(1, 5)}
double_adj = {f'double_M{i}_s{j}{j+1}': [] for i in range(1, 5) for j in range(1, 4)}
double_non_adj = {f'double_M{i}_s{pair[0]}{pair[1]}': [] for i in range(1, 5) for pair in [(1,3), (2,4), (1,4)]}
triple_adj = {f'triple_M{i}_s{j}{j+1}{j+2}': [] for i in range(1, 5) for j in range(1, 3)}
triple_non_adj = {f'triple_M{i}_s{triplet[0]}{triplet[1]}{triplet[2]}': [] for i in range(1, 5) for triplet in [(1,2,4), (1,3,4)]}
quadruples = {f'quadruple_M{i}_s1234': [] for i in range(1, 5)}

# Loop over modules
for i in range(1, 5):
    charge_matrix = np.zeros((len(merged_df), 4))  # Stores strip-wise charges for this module

    for j in range(1, 5):  # Loop over strips
        col_name = f"Q_M{i}s{j}"  # Column name
        v = merged_df[col_name].fillna(0).to_numpy()  # Ensure no NaNs
        charge_matrix[:, j - 1] = v  # Store strip charge

    # Classify events based on strip charge distribution
    nonzero_counts = (charge_matrix > 0).sum(axis=1)  # Count nonzero strips per event

    for event_idx, count in enumerate(nonzero_counts):
        nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0] + 1  # Get active strip indices (1-based)
        charges = charge_matrix[event_idx, nonzero_strips - 1]  # Get nonzero charges

        # Single detection
        if count == 1:
            key = f'single_M{i}_s{nonzero_strips[0]}'
            singles[key].append((charges[0],))

        # Double adjacent
        elif count == 2 and nonzero_strips[1] - nonzero_strips[0] == 1:
            key = f'double_M{i}_s{nonzero_strips[0]}{nonzero_strips[1]}'
            double_adj[key].append(tuple(charges))

        # Double non-adjacent
        elif count == 2 and nonzero_strips[1] - nonzero_strips[0] > 1:
            key = f'double_M{i}_s{nonzero_strips[0]}{nonzero_strips[1]}'
            if key in double_non_adj:
                double_non_adj[key].append(tuple(charges))

        # Triple adjacent
        elif count == 3 and (nonzero_strips[2] - nonzero_strips[0] == 2):
            key = f'triple_M{i}_s{nonzero_strips[0]}{nonzero_strips[1]}{nonzero_strips[2]}'
            triple_adj[key].append(tuple(charges))

        # Triple non-adjacent
        elif count == 3 and (nonzero_strips[2] - nonzero_strips[0] > 2):
            key = f'triple_M{i}_s{nonzero_strips[0]}{nonzero_strips[1]}{nonzero_strips[2]}'
            if key in triple_non_adj:
                triple_non_adj[key].append(tuple(charges))

        # Quadruple detection
        elif count == 4:
            key = f'quadruple_M{i}_s1234'
            quadruples[key].append(tuple(charges))

# Convert results to DataFrames
df_singles = {k: pd.DataFrame(v, columns=["Charge1"]) for k, v in singles.items()}
df_double_adj = {k: pd.DataFrame(v, columns=["Charge1", "Charge2"]) for k, v in double_adj.items()}
df_double_non_adj = {k: pd.DataFrame(v, columns=["Charge1", "Charge2"]) for k, v in double_non_adj.items()}
df_triple_adj = {k: pd.DataFrame(v, columns=["Charge1", "Charge2", "Charge3"]) for k, v in triple_adj.items()}
df_triple_non_adj = {k: pd.DataFrame(v, columns=["Charge1", "Charge2", "Charge3"]) for k, v in triple_non_adj.items()}
df_quadruples = {k: pd.DataFrame(v, columns=["Charge1", "Charge2", "Charge3", "Charge4"]) for k, v in quadruples.items()}

#%%

# Singles
single_M1_s1 = df_singles['single_M1_s1']
single_M1_s2 = df_singles['single_M1_s2']
single_M1_s3 = df_singles['single_M1_s3']
single_M1_s4 = df_singles['single_M1_s4']

single_M2_s1 = df_singles['single_M2_s1']
single_M2_s2 = df_singles['single_M2_s2']
single_M2_s3 = df_singles['single_M2_s3']
single_M2_s4 = df_singles['single_M2_s4']

single_M3_s1 = df_singles['single_M3_s1']
single_M3_s2 = df_singles['single_M3_s2']
single_M3_s3 = df_singles['single_M3_s3']
single_M3_s4 = df_singles['single_M3_s4']

single_M4_s1 = df_singles['single_M4_s1']
single_M4_s2 = df_singles['single_M4_s2']
single_M4_s3 = df_singles['single_M4_s3']
single_M4_s4 = df_singles['single_M4_s4']

# Double adjacent
double_M1_s12 = df_double_adj['double_M1_s12']
double_M1_s23 = df_double_adj['double_M1_s23']
double_M1_s34 = df_double_adj['double_M1_s34']

double_M2_s12 = df_double_adj['double_M2_s12']
double_M2_s23 = df_double_adj['double_M2_s23']
double_M2_s34 = df_double_adj['double_M2_s34']

double_M3_s12 = df_double_adj['double_M3_s12']
double_M3_s23 = df_double_adj['double_M3_s23']
double_M3_s34 = df_double_adj['double_M3_s34']

double_M4_s12 = df_double_adj['double_M4_s12']
double_M4_s23 = df_double_adj['double_M4_s23']
double_M4_s34 = df_double_adj['double_M4_s34']

# Doubles non adjacent
double_M1_s13 = df_double_non_adj['double_M1_s13']
double_M1_s24 = df_double_non_adj['double_M1_s24']
double_M1_s14 = df_double_non_adj['double_M1_s14']

double_M2_s13 = df_double_non_adj['double_M2_s13']
double_M2_s24 = df_double_non_adj['double_M2_s24']
double_M2_s14 = df_double_non_adj['double_M2_s14']

double_M3_s13 = df_double_non_adj['double_M3_s13']
double_M3_s24 = df_double_non_adj['double_M3_s24']
double_M3_s14 = df_double_non_adj['double_M3_s14']

double_M4_s13 = df_double_non_adj['double_M4_s13']
double_M4_s24 = df_double_non_adj['double_M4_s24']
double_M4_s14 = df_double_non_adj['double_M4_s14']

# Triple adjacent
triple_M1_s123 = df_triple_adj['triple_M1_s123']
triple_M1_s234 = df_triple_adj['triple_M1_s234']

triple_M2_s123 = df_triple_adj['triple_M2_s123']
triple_M2_s234 = df_triple_adj['triple_M2_s234']

triple_M3_s123 = df_triple_adj['triple_M3_s123']
triple_M3_s234 = df_triple_adj['triple_M3_s234']

triple_M4_s123 = df_triple_adj['triple_M4_s123']
triple_M4_s234 = df_triple_adj['triple_M4_s234']

# Triple non adjacent
triple_M1_s124 = df_triple_non_adj['triple_M1_s124']
triple_M1_s134 = df_triple_non_adj['triple_M1_s134']

triple_M2_s124 = df_triple_non_adj['triple_M2_s124']
triple_M2_s134 = df_triple_non_adj['triple_M2_s134']

triple_M3_s124 = df_triple_non_adj['triple_M3_s124']
triple_M3_s134 = df_triple_non_adj['triple_M3_s134']

triple_M4_s124 = df_triple_non_adj['triple_M4_s124']
triple_M4_s134 = df_triple_non_adj['triple_M4_s134']

# Quadruple
quadruple_M1_s1234 = df_quadruples['quadruple_M1_s1234']
quadruple_M2_s1234 = df_quadruples['quadruple_M2_s1234']
quadruple_M3_s1234 = df_quadruples['quadruple_M3_s1234']
quadruple_M4_s1234 = df_quadruples['quadruple_M4_s1234']
#%%

# Helper function to rename columns based on their source dataset
def rename_columns(df, source_name):
    return df.rename(columns={col: f"{source_name}_{col}" for col in df.columns})

# Initialize dictionary
real_multiplicities = {}

# Define modules
modules = ["M1", "M2", "M3", "M4"]

# Loop over modules
for module in modules:
    real_multiplicities[f"real_single_{module}_s1"] = pd.concat([
        rename_columns(globals()[f"single_{module}_s1"], f"single_{module}_s1"),
        rename_columns(globals()[f"double_{module}_s13"][['Charge1', 'Charge2']], f"double_{module}_s13"),
        rename_columns(globals()[f"double_{module}_s14"][['Charge1', 'Charge2']], f"double_{module}_s14"),
        rename_columns(globals()[f"triple_{module}_s134"][['Charge1']], f"triple_{module}_s134")
    ], axis=1)

    real_multiplicities[f"real_single_{module}_s2"] = pd.concat([
        rename_columns(globals()[f"single_{module}_s2"], f"single_{module}_s2"),
        rename_columns(globals()[f"double_{module}_s24"][['Charge1', 'Charge2']], f"double_{module}_s24")
    ], axis=1)

    real_multiplicities[f"real_single_{module}_s3"] = pd.concat([
        rename_columns(globals()[f"single_{module}_s3"], f"single_{module}_s3"),
        rename_columns(globals()[f"double_{module}_s13"][['Charge1', 'Charge2']], f"double_{module}_s13")
    ], axis=1)

    real_multiplicities[f"real_single_{module}_s4"] = pd.concat([
        rename_columns(globals()[f"single_{module}_s4"], f"single_{module}_s4"),
        rename_columns(globals()[f"double_{module}_s24"][['Charge1', 'Charge2']], f"double_{module}_s24"),
        rename_columns(globals()[f"double_{module}_s14"][['Charge1', 'Charge2']], f"double_{module}_s14"),
        rename_columns(globals()[f"triple_{module}_s124"][['Charge3']], f"triple_{module}_s124")
    ], axis=1)

    # Doubles adjacent
    real_multiplicities[f"real_double_{module}_s12"] = pd.concat([
        rename_columns(globals()[f"double_{module}_s12"], f"double_{module}_s12"),
        rename_columns(globals()[f"triple_{module}_s124"][['Charge1', 'Charge2']], f"triple_{module}_s124")
    ], axis=1)

    real_multiplicities[f"real_double_{module}_s23"] = rename_columns(globals()[f"double_{module}_s23"], f"double_{module}_s23")

    real_multiplicities[f"real_double_{module}_s34"] = pd.concat([
        rename_columns(globals()[f"double_{module}_s34"], f"double_{module}_s34"),
        rename_columns(globals()[f"triple_{module}_s134"][['Charge2', 'Charge3']], f"triple_{module}_s134")
    ], axis=1)

    # Triples adjacent
    real_multiplicities[f"real_triple_{module}_s123"] = rename_columns(globals()[f"triple_{module}_s123"], f"triple_{module}_s123")
    real_multiplicities[f"real_triple_{module}_s234"] = rename_columns(globals()[f"triple_{module}_s234"], f"triple_{module}_s234")

    # Quadruples
    real_multiplicities[f"real_quadruple_{module}_s1234"] = rename_columns(globals()[f"quadruple_{module}_s1234"], f"quadruple_{module}_s1234")


#%%
# List the keys
print(real_multiplicities.keys())

#%%

cases = ["real_single", "real_double", "real_triple", "real_quadruple"]

for case in cases:
    fig_rows = len(modules)
    fig_cols = 0

    # First, compute the max number of columns across all modules (for consistent layout)
    max_columns = 0
    all_combined_dfs = []  # Store the per-module DataFrames

    for module in modules:
        other_key = f"{case}_{module}"
        matching_keys = sorted([key for key in real_multiplicities if key.startswith(f"{other_key}_")])

        if not matching_keys:
            print(f"No data for {other_key}")
            all_combined_dfs.append(None)
            continue

        combined_df = pd.concat([real_multiplicities[key] for key in matching_keys], axis=1)
        all_combined_dfs.append(combined_df)

        if combined_df.shape[1] > max_columns:
            max_columns = combined_df.shape[1]

    # Now that we know max_columns, build the subplot grid
    fig, axs = plt.subplots(fig_rows, max_columns, figsize=(4 * max_columns, 4 * fig_rows))

    # Make axs 2D no matter what
    if fig_rows == 1:
        axs = [axs]
    if max_columns == 1:
        axs = [[ax] for ax in axs]

    for a, (module, combined_df) in enumerate(zip(modules, all_combined_dfs)):
        if combined_df is None:
            continue  # Skip missing data

        for i, column in enumerate(combined_df.columns):
            axs[a][i].hist(combined_df[column], bins=70, range=(0, 1500), histtype="step", linewidth=1.5, density=False)
            axs[a][i].set_title(f"{module} - {column}")
            axs[a][i].set_xlabel("Charge")
            axs[a][i].set_ylabel("Frequency")
            axs[a][i].grid(True)

        # Hide unused subplots (if any)
        for j in range(i + 1, max_columns):
            axs[a][j].axis("off")

    plt.tight_layout()
    figure_name = f"{case}.png"
    plt.savefig(figure_save_path + figure_name, dpi=150)
    # plt.show()
    plt.close()

# %%

# cases = ["real_double"]

# for case in cases:
#     fig_rows = len(modules)
#     all_combined_dfs = []
#     max_pairs = 0  # to define number of scatter subplots per module

#     for module in modules:
#         other_key = f"{case}_{module}"
#         matching_keys = sorted([key for key in real_multiplicities if key.startswith(f"{other_key}_")])

#         if not matching_keys:
#             print(f"No data for {other_key}")
#             all_combined_dfs.append(None)
#             continue

#         combined_df = pd.concat([real_multiplicities[key] for key in matching_keys], axis=1)
#         all_combined_dfs.append(combined_df)

#         n_cols = combined_df.shape[1]
#         n_pairs = n_cols // 2
#         if n_pairs > max_pairs:
#             max_pairs = n_pairs

#     fig, axs = plt.subplots(fig_rows, max_pairs, figsize=(4 * max_pairs, 4 * fig_rows))

#     # Normalize axs shape
#     if fig_rows == 1:
#         axs = [axs]
#     if max_pairs == 1:
#         axs = [[ax] for ax in axs]

#     for a, (module, combined_df) in enumerate(zip(modules, all_combined_dfs)):
#         if combined_df is None:
#             continue

#         columns = list(combined_df.columns)
#         n_pairs = len(columns) // 2

#         for i in range(n_pairs):
#             col1 = columns[2*i]
#             col2 = columns[2*i + 1]
            
#             # Filter col1 and col2 between 0 and 1250
#             combined_df = combined_df[(combined_df[col1] > 0) & (combined_df[col1] < 1250)]
#             combined_df = combined_df[(combined_df[col2] > 0) & (combined_df[col2] < 1250)]
            
#             axs[a][i].hexbin(combined_df[col1], combined_df[col2], gridsize=50, cmap='viridis', mincnt=1)
            
#             # Put the background in the first viridis color as 0
#             axs[a][i].set_facecolor('#440154')
            
#             axs[a][i].set_title(f"{module}: {col1} vs {col2}")
#             axs[a][i].set_xlabel(col1)
#             axs[a][i].set_ylabel(col2)
#             axs[a][i].grid(True)

#         # Hide any unused axes in this row
#         for j in range(n_pairs, max_pairs):
#             axs[a][j].axis("off")

#     plt.tight_layout()
#     figure_name = f"{case}_scatter_pairs.png"
#     plt.savefig(figure_save_path + figure_name, dpi=150)
#     # plt.show()
#     plt.close()


# #%%

# cases = ["real_double"]
# bin_width = 10
# max_charge = 1250
# bins = np.arange(crosstalk_limit, max_charge + bin_width, bin_width)

# for case in cases:
#     for module in modules:
#         other_key = f"{case}_{module}"
#         matching_keys = sorted([key for key in real_multiplicities if key.startswith(f"{other_key}_")])

#         if not matching_keys:
#             print(f"No data for {other_key}")
#             continue

#         combined_df = pd.concat([real_multiplicities[key] for key in matching_keys], axis=1)
#         columns = list(combined_df.columns)
#         n_pairs = len(columns) // 2

#         plt.figure(figsize=(10, 6))

#         for i in range(n_pairs):
#             col1 = columns[2 * i]
#             col2 = columns[2 * i + 1]

#             x1 = combined_df[col1].values
#             y1 = (combined_df[col2] < crosstalk_limit).astype(int).values

#             x2 = combined_df[col2].values
#             y2 = (combined_df[col1] < crosstalk_limit).astype(int).values

#             # Bin col1 and compute crosstalk prob using col2
#             bin_indices1 = np.digitize(x1, bins)
#             prob1 = []
#             bin_centers = []

#             for b in range(1, len(bins)):
#                 in_bin = bin_indices1 == b
#                 total = np.sum(in_bin)
#                 if total > 0:
#                     crosstalks = np.sum(y1[in_bin])
#                     prob1.append(crosstalks / total)
#                     bin_centers.append((bins[b - 1] + bins[b]) / 2)

#             # Bin col2 and compute crosstalk prob using col1
#             bin_indices2 = np.digitize(x2, bins)
#             prob2 = []
#             bin_centers2 = []

#             for b in range(1, len(bins)):
#                 in_bin = bin_indices2 == b
#                 total = np.sum(in_bin)
#                 if total > 0:
#                     crosstalks = np.sum(y2[in_bin])
#                     prob2.append(crosstalks / total)
#                     bin_centers2.append((bins[b - 1] + bins[b]) / 2)

#             # Plot both directions
#             plt.plot(bin_centers, prob1, label=f"{col1} ➝ {col2}")
#             plt.plot(bin_centers2, prob2, label=f"{col2} ➝ {col1}", linestyle='--')

#         plt.title(f"Crosstalk probability - {case} - {module}")
#         plt.xlabel("Charge")
#         plt.ylabel("Crosstalk Probability")
#         # plt.ylim(0, 1.05)
#         plt.grid(True)
        
#         # Change the yticks to be in percentage
#         from matplotlib.ticker import FuncFormatter
#         plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
        
#         plt.legend()
#         plt.tight_layout()

#         figure_name = f"{case}_{module}_crosstalk_prob_OG.png"
#         plt.savefig(figure_save_path + figure_name, dpi=150)
#         plt.close()

# #%%


# cases = ["real_double"]
# bin_width = 100
# max_charge = 1250
# bins = np.arange(crosstalk_limit, max_charge + bin_width, bin_width)

# for case in cases:
#     for module in modules:
#         other_key = f"{case}_{module}"
#         matching_keys = sorted([key for key in real_multiplicities if key.startswith(f"{other_key}_")])

#         if not matching_keys:
#             print(f"No data for {other_key}")
#             continue

#         combined_df = pd.concat([real_multiplicities[key] for key in matching_keys], axis=1)
#         columns = list(combined_df.columns)
#         n_pairs = len(columns) // 2

#         plt.figure(figsize=(10, 6))

#         for i in range(n_pairs):
#             col1 = columns[2 * i]
#             col2 = columns[2 * i + 1]

#             # Single-strip data
#             s1_key = f"real_single_{module}_s{col1.split('_')[2][1]}"
#             s2_key = f"real_single_{module}_s{col2.split('_')[2][1]}"

#             if s1_key not in real_multiplicities or s2_key not in real_multiplicities:
#                 print(f"Missing single-strip data for {col1} or {col2}")
#                 continue

#             # Reconstruct correct column name for single-strip version
#             single_col1 = f"single_{module}_s{col1.split('_')[2][1]}_Charge1"
#             single1 = real_multiplicities[s1_key][single_col1].values

#             single_col2 = f"single_{module}_s{col2.split('_')[2][1]}_Charge1"
#             single2 = real_multiplicities[s2_key][single_col2].values


#             x1_double = combined_df[col1].values
#             x2_double = combined_df[col2].values

#             # Crosstalk condition vectors
#             y1 = (x2_double < crosstalk_limit).astype(int)
#             y2 = (x1_double < crosstalk_limit).astype(int)

#             # First direction: col1 (reference), crosstalk in col2
#             total1 = np.concatenate([single1, x1_double])
#             bin_indices_total1 = np.digitize(total1, bins)
#             bin_indices_double1 = np.digitize(x1_double, bins)

#             prob1 = []
#             bin_centers1 = []

#             for b in range(1, len(bins)):
#                 total_in_bin = np.sum(bin_indices_total1 == b)
#                 double_in_bin = bin_indices_double1 == b
#                 if total_in_bin > 0:
#                     crosstalks = np.sum(y1[double_in_bin])
#                     prob1.append(crosstalks / total_in_bin)
#                     bin_centers1.append((bins[b - 1] + bins[b]) / 2)

#             # Second direction: col2 (reference), crosstalk in col1
#             total2 = np.concatenate([single2, x2_double])
#             bin_indices_total2 = np.digitize(total2, bins)
#             bin_indices_double2 = np.digitize(x2_double, bins)

#             prob2 = []
#             bin_centers2 = []

#             for b in range(1, len(bins)):
#                 total_in_bin = np.sum(bin_indices_total2 == b)
#                 double_in_bin = bin_indices_double2 == b
#                 if total_in_bin > 0:
#                     crosstalks = np.sum(y2[double_in_bin])
#                     prob2.append(crosstalks / total_in_bin)
#                     bin_centers2.append((bins[b - 1] + bins[b]) / 2)

#             # Plot both directions
#             plt.plot(bin_centers1, prob1, label=f"{col1} ➔ {col2}")
#             plt.plot(bin_centers2, prob2, label=f"{col2} ➔ {col1}", linestyle="--")

#         plt.title(f"Crosstalk probability - {case} - {module}")
#         plt.xlabel("Charge")
#         plt.ylabel("Crosstalk Probability")
#         plt.grid(True)

#         from matplotlib.ticker import FuncFormatter
#         plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

#         plt.legend()
#         plt.tight_layout()

#         figure_name = f"{case}_{module}_crosstalk_prob.png"
#         plt.savefig(figure_save_path + figure_name, dpi=150)
#         plt.close()

# %%

modules = ["M1", "M2", "M3", "M4"]
sum_real_multiplicities = {}

for module in modules:
    # -- DOUBLES -------------------------------------------------
    # s12
    df_12 = real_multiplicities[f"real_double_{module}_s12"]
    sum_12 = df_12.sum(axis=1, numeric_only=True)  # Sum of all columns in that DataFrame
    sum_real_multiplicities[f"sum_real_double_{module}_s12"] = pd.DataFrame({"Charge12": sum_12})

    # s23
    df_23 = real_multiplicities[f"real_double_{module}_s23"]
    sum_23 = df_23.sum(axis=1, numeric_only=True)
    sum_real_multiplicities[f"sum_real_double_{module}_s23"] = pd.DataFrame({"Charge23": sum_23})

    # s34
    df_34 = real_multiplicities[f"real_double_{module}_s34"]
    sum_34 = df_34.sum(axis=1, numeric_only=True)
    sum_real_multiplicities[f"sum_real_double_{module}_s34"] = pd.DataFrame({"Charge34": sum_34})

    # -- TRIPLES -------------------------------------------------
    # s123
    df_123 = real_multiplicities[f"real_triple_{module}_s123"]
    sum_123 = df_123.sum(axis=1, numeric_only=True)
    sum_real_multiplicities[f"sum_real_triple_{module}_s123"] = pd.DataFrame({"Charge123": sum_123})

    # s234
    df_234 = real_multiplicities[f"real_triple_{module}_s234"]
    sum_234 = df_234.sum(axis=1, numeric_only=True)
    sum_real_multiplicities[f"sum_real_triple_{module}_s234"] = pd.DataFrame({"Charge234": sum_234})

    # -- QUADRUPLES ----------------------------------------------
    # s1234
    df_1234 = real_multiplicities[f"real_quadruple_{module}_s1234"]
    sum_1234 = df_1234.sum(axis=1, numeric_only=True)
    sum_real_multiplicities[f"sum_real_quadruple_{module}_s1234"] = pd.DataFrame({"Charge1234": sum_1234})

sum_real_multiplicities.update({
    k: df for k, df in real_multiplicities.items() if "real_single" in k
})

#%%

# 1) Merge single from real_multiplicities + double/triple/quad from sum_real_multiplicities
combined_multiplicities = {}

# Copy the single-case DataFrames as-is from real_multiplicities
for key, df in real_multiplicities.items():
    if key.startswith("real_single"):
        combined_multiplicities[key] = df

# Copy (and rename) the double/triple/quad entries from sum_real_multiplicities
for key, df in sum_real_multiplicities.items():
    # They have keys like "sum_real_double_M1_s12"
    # We rename them to match "real_double_M1_s12"
    new_key = key.replace("sum_", "")  # e.g. "sum_real_double_M1_s12" -> "real_double_M1_s12"
    combined_multiplicities[new_key] = df

cases = ["real_single", "real_double", "real_triple", "real_quadruple"]
modules = ["M1", "M2", "M3", "M4"]

#%%

for case in cases:
    fig_rows = len(modules)
    max_columns = 0
    all_combined_dfs = []

    # 1) Identify & combine all DataFrames for each module
    for module in modules:
        # We'll look for dictionary keys that start like "real_single_M1_...", etc.
        # Example: "real_double_M1_s12"
        prefix = f"{case}_{module}"
        matching_keys = sorted(k for k in combined_multiplicities if k.startswith(prefix))

        if not matching_keys:
            print(f"No data for {prefix}")
            all_combined_dfs.append(None)
            continue

        # Concatenate all DataFrames for this module into one big DF (columns side by side)
        combined_df = pd.concat([combined_multiplicities[k] for k in matching_keys], axis=1)
        all_combined_dfs.append(combined_df)

        # Track largest number of columns (for consistent subplot layout)
        if combined_df.shape[1] > max_columns:
            max_columns = combined_df.shape[1]

    # 2) Build the subplot grid for this case
    fig, axs = plt.subplots(fig_rows, max_columns, figsize=(4 * max_columns, 4 * fig_rows))

    # Make sure axs is 2D no matter what
    if fig_rows == 1:
        axs = [axs]  # wrap in a list so axs[a][i] won't error
    if max_columns == 1:
        axs = [[ax] for ax in axs]  # similarly wrap columns

    # 3) Plot each row (module) and column (strips or partial sums)
    for row_idx, (module, combined_df) in enumerate(zip(modules, all_combined_dfs)):
        if combined_df is None:
            # No data for this module
            continue

        for col_idx, column_name in enumerate(combined_df.columns):
            axs[row_idx][col_idx].hist(
                combined_df[column_name],
                bins=70,
                range=(0, 2000),
                histtype="step",
                linewidth=1.5,
                density=False
            )
            axs[row_idx][col_idx].set_title(f"{module} - {column_name}")
            axs[row_idx][col_idx].set_xlabel("Charge")
            axs[row_idx][col_idx].set_ylabel("Frequency")
            axs[row_idx][col_idx].grid(True)

        # Hide any unused subplots in this row
        for hidden_col_idx in range(col_idx + 1, max_columns):
            axs[row_idx][hidden_col_idx].axis("off")

    plt.tight_layout()
    figure_name = f"/{case}_sum.png"
    plt.savefig(figure_save_path + figure_name, dpi=150)
    plt.show()  # If you want interactive displays instead
    plt.close()

# %%

# POLYA FIT TEST

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit

# Constants
q_e = 1.602e-4  # fC (charge of 1 electron)

# Polya model adapted to charge
def polya_induced_charge(Q, theta, nbar, alpha, A, offset):
    n = Q * alpha
    norm = ((theta + 1) ** (theta + 1)) / gamma(theta + 1)
    return A * norm * (n / nbar)**theta * np.exp(-(theta + 1) * n / nbar) + offset

# Get the data (flatten just in case)
data = single_M1_s2["Charge1"].dropna().to_numpy().flatten()

data = data / q_e

# Plot histogram
# counts, bin_edges = np.histogram(data, bins=200, range=(0, 1500))
counts, bin_edges = np.histogram(data, bins=400, range=(0, 1.1e7))
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.figure(figsize=(6, 5))
plt.plot(bin_centers, counts, label="Original data")

# Filter valid bins
counts = np.asarray(counts).flatten()
bin_centers = np.asarray(bin_centers).flatten()
mask = (bin_centers > 0) & (counts > 0)
x_fit = bin_centers[mask]
y_fit = counts[mask]

print(data[data != 0].mean())

# Initial guesses: theta, nbar, alpha, A, offset
p0 = [1, 1e6, 0.7, max(counts), 0]
bounds = ([0, 0, 0, 0, -1e16], [10, 1e16, 1, 1e16, 1e16])  # theta ≥ 0.01, nbar ≥ 1, alpha in [0.01, 1]

# Fit
popt, _ = curve_fit(polya_induced_charge, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=100000)
theta_fit, nbar_fit, alpha_fit, A_fit, offset_fit = popt

plt.plot(x_fit, y_fit, "o", label="Data given to the fit")

# Plot results
# plt.hist(data, bins=70, range=(0, 1500), histtype="step", label="Data", linewidth=1.5, density = False)

x_fine = np.linspace(0, 1.1e7, 300)
plt.plot(
    x_fine,
    polya_induced_charge(x_fine, *popt),
    "r--",
    label=rf"Fit: $\theta={theta_fit:.2f},\ \bar{{n}}={nbar_fit:.0f},\ \alpha={alpha_fit:.2f}$"
)

plt.xlabel("Induced equivalent electrons")
plt.ylabel("Entries")
plt.title("Polya Fit: single_M1_s2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
