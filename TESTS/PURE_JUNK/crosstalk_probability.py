#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

create_plots = False

n_bins = 100
right_lim = 1400 # 1250
crosstalk_limit = 2.5 #2.6
charge_vector = np.linspace(crosstalk_limit, right_lim, n_bins)

remove_streamer = False
streamer_limit = 90

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


figure_save_path = "/home/cayetano/DATAFLOW_v3/CROSSTALK_CHARGE_ANALYSIS/"
import os
print("Storing in ", figure_save_path)
if not os.path.exists(figure_save_path):
      os.makedirs(figure_save_path)

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


df_list_OG = [pd.read_csv(file, delimiter=",") for file in file_paths]  # Adjust delimiter if needed

# %%

# -----------------------------------------------------------------------------------------------
# NO CROSSTALK SECTION --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Read and concatenate all files
df_list = df_list_OG.copy()
merged_df = pd.concat(df_list, ignore_index=True)
merged_df.drop_duplicates(inplace=True)

remove_crosstalk = True
if remove_crosstalk or remove_streamer:
    if remove_crosstalk:
        for col in merged_df.columns:
            if "Q_" in col and "s" in col:
                merged_df[col] = merged_df[col].apply(lambda x: 0 if x < crosstalk_limit else x)
                      
    if remove_streamer:
        for col in merged_df.columns:
            if "Q_" in col and "s" in col:
                merged_df[col] = merged_df[col].apply(lambda x: 0 if x > streamer_limit else x)     

if create_plots:
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
      plt.close()

columns_to_drop = ['Time', 'CRT_avg', 'x', 'y', 'theta', 'phi', 't0', 's', 'type', 'charge_event']
merged_df = merged_df.drop(columns=columns_to_drop)

# For all the columns apply the calibration and not change the name of the columns
for col in merged_df.columns:
    merged_df[col] = interpolate_fast_charge(merged_df[col])

# Initialize dictionaries to store charge distributions
singles = {f'single_M{i}_s{j}': [] for i in range(1, 5) for j in range(1, 5)}

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

# Convert results to DataFrames
df_singles = {k: pd.DataFrame(v, columns=["Charge1"]) for k, v in singles.items()}

# Assuming df_singles and crosstalk_limit are already defined
bin_edges = charge_vector
histograms_no_crosstalk = {}

print("Histograms for no crosstalk")
for m in range(1, 5):
    for s in range(1, 5):
        key = f"M{m}_s{s}"
        data = df_singles[f"single_M{m}_s{s}"]['Charge1'].values
        hist, _ = np.histogram(data, bins=bin_edges)
        histograms_no_crosstalk[key] = hist

# %%

# -----------------------------------------------------------------------------------------------
# YES CROSSTALK SECTION -------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Read and concatenate all files
df_list = df_list_OG.copy()
merged_df = pd.concat(df_list, ignore_index=True)
merged_df.drop_duplicates(inplace=True)

remove_crosstalk = False
if remove_crosstalk or remove_streamer:
    if remove_crosstalk:
        for col in merged_df.columns:
            if "Q_" in col and "s" in col:
                merged_df[col] = merged_df[col].apply(lambda x: 0 if x < crosstalk_limit else x)
                      
    if remove_streamer:
        for col in merged_df.columns:
            if "Q_" in col and "s" in col:
                merged_df[col] = merged_df[col].apply(lambda x: 0 if x > streamer_limit else x)     

if create_plots:
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
      plt.close()

columns_to_drop = ['Time', 'CRT_avg', 'x', 'y', 'theta', 'phi', 't0', 's', 'type', 'charge_event']
merged_df = merged_df.drop(columns=columns_to_drop)

# For all the columns apply the calibration and not change the name of the columns
for col in merged_df.columns:
    merged_df[col] = interpolate_fast_charge(merged_df[col])

# Initialize dictionaries to store charge distributions
singles = {f'single_M{i}_s{j}': [] for i in range(1, 5) for j in range(1, 5)}

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

# Convert results to DataFrames
df_singles = {k: pd.DataFrame(v, columns=["Charge1"]) for k, v in singles.items()}

# print(df_singles["single_M1_s1"]['Charge1'].values)

# Assuming df_singles and crosstalk_limit are already defined
bin_edges = charge_vector
histograms_yes_crosstalk = {}

print("Histograms for yes crosstalk")
for m in range(1, 5):
    for s in range(1, 5):
        key = f"M{m}_s{s}"
        data = df_singles[f"single_M{m}_s{s}"]['Charge1'].values
        hist, _ = np.histogram(data, bins=bin_edges)
        histograms_yes_crosstalk[key] = hist

# %%

def compute_fraction_and_uncertainty(
    charge_edges, hist_no, hist_yes
):
    """
    Compute fraction f = (N_no - N_yes)/(N_no + N_yes),
    plus its propagated Poisson uncertainty.
    
    Parameters
    ----------
    charge_edges : 1D array
        Bin edges used for the histograms (length = number_of_bins + 1).
    hist_no : dict of 1D arrays
        Dictionary keyed by 'M#_s#' with histogram counts for 'no crosstalk'.
    hist_yes : dict of 1D arrays
        Dictionary keyed by 'M#_s#' with histogram counts for 'yes crosstalk'.

    Returns
    -------
    x_vals : 1D array
        The x-coordinates for plotting, i.e. `charge_edges[:-1]`.
    fraction_dict : dict of 1D arrays
        fraction_dict[key] = (N_no - N_yes)/(N_no + N_yes) per bin.
    uncertainty_dict : dict of 1D arrays
        Uncertainty from Poisson statistics, same shape as fraction arrays.
    """
    fraction_dict = {}
    uncertainty_dict = {}

    # We remove the last edge to match the histogram "counts" length:
    x_vals = charge_edges[:-1]

    for key in hist_no:
        # Just for clarity
        Nn = hist_no[key]   # 'no crosstalk' counts
        Ny = hist_yes[key]  # 'yes crosstalk' counts
        D = Nn + Ny         # denominator

        # Fraction
        with np.errstate(divide='ignore', invalid='ignore'):
            f = (Nn - Ny) / D
            # If D=0 => f -> undefined => set to 0
            f[np.isnan(f)] = 0  

        # Poisson errors for counts
        sigma_Nn = np.sqrt(Nn)
        sigma_Ny = np.sqrt(Ny)

        # Partial derivatives
        # df/dNn =  2*Ny / D^2
        # df/dNy = -2*Nn / D^2
        with np.errstate(divide='ignore', invalid='ignore'):
            df_dNn = 2 * Ny / (D**2)
            df_dNy = -2 * Nn / (D**2)

            # Total variance
            sigma_f_sq = (df_dNn**2) * (sigma_Nn**2) + (df_dNy**2) * (sigma_Ny**2)
            # If D=0, that might lead to NaN
            sigma_f = np.sqrt(sigma_f_sq)
            sigma_f[np.isnan(sigma_f)] = 0

        fraction_dict[key] = f
        uncertainty_dict[key] = sigma_f

    return x_vals, fraction_dict, uncertainty_dict

# ---------------------------------------------------------------------
# Example usage to produce the 4x4 grid of plots:
# ---------------------------------------------------------------------

# 1. Compute fraction + uncertainty
x_vals, fraction_hist, frac_err = compute_fraction_and_uncertainty(
    charge_vector,  # bin edges from your snippet
    histograms_no_crosstalk,
    histograms_yes_crosstalk
)

#%%

# 2. Plot in 4x4
fig, axs = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
fig.suptitle(f"Crosstalk probability with Poisson Error Bands, mingo0{station}", fontsize=14)

for m in range(1, 5):
    for s in range(1, 5):
        ax = axs[m-1, s-1]
        key = f"M{m}_s{s}"

        y_vals = fraction_hist[key]
        y_err  = frac_err[key]

        ax.plot(x_vals, y_vals, label=key)
        ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, alpha=0.3)
        ax.set_ylim(0, 1)         # Because fraction can be negative if N_yes > N_no
        ax.set_title(key)
        ax.grid(True)

# Better spacing
for ax in axs[-1, :]:
    ax.set_xlabel("Charge")
for ax in axs[:, 0]:
    ax.set_ylabel("Probability")

# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout()
figure_name = f"crosstalk_probability_mingo0{station}.png"
plt.savefig(figure_save_path + figure_name, dpi=300)
plt.show()


# %%
