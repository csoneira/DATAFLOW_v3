#%%

# Quick plotter for the article

remove_crosstalk = True
crosstalk_limit = 3.5 #2.6

remove_streamer = True
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
figure_save_path = "/home/cayetano/DATAFLOW_v3/SIMPLE_N_TIMES_CLUSTER_SIZE_CHARGE_ANALYSIS"
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
plt.show()
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

# # Create the DataFrame
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

# Create a vector of minimum and other of maximum charge for double adjacent detections, for each module
# Dictionaries to store min and max charge values for double-adjacent detections
double_adjacent_M1_min, double_adjacent_M1_max = [], []
double_adjacent_M2_min, double_adjacent_M2_max = [], []
double_adjacent_M3_min, double_adjacent_M3_max = [], []
double_adjacent_M4_min, double_adjacent_M4_max = [], []

double_non_adjacent_M1_min, double_non_adjacent_M1_max = [], []
double_non_adjacent_M2_min, double_non_adjacent_M2_max = [], []
double_non_adjacent_M3_min, double_non_adjacent_M3_max = [], []
double_non_adjacent_M4_min, double_non_adjacent_M4_max = [], []

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
            nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0]  # Get active strip indices
            charges = charge_matrix[event_idx, nonzero_strips]  # Get nonzero charges

            if count == 2 and np.all(np.diff(nonzero_strips) == 1):  # Double adjacent
                  min_charge = np.min(charges)
                  max_charge = np.max(charges)

                  if i == 1:
                        double_adjacent_M1_min.append(min_charge)
                        double_adjacent_M1_max.append(max_charge)
                  elif i == 2:
                        double_adjacent_M2_min.append(min_charge)
                        double_adjacent_M2_max.append(max_charge)
                  elif i == 3:
                        double_adjacent_M3_min.append(min_charge)
                        double_adjacent_M3_max.append(max_charge)
                  elif i == 4:
                        double_adjacent_M4_min.append(min_charge)
                        double_adjacent_M4_max.append(max_charge)
                        
            if count == 2 and np.all(np.diff(nonzero_strips) != 1):
                  min_charge = np.min(charges)
                  max_charge = np.max(charges)

                  if i == 1:
                        double_non_adjacent_M1_min.append(min_charge)
                        double_non_adjacent_M1_max.append(max_charge)
                  elif i == 2:
                        double_non_adjacent_M2_min.append(min_charge)
                        double_non_adjacent_M2_max.append(max_charge)
                  elif i == 3:
                        double_non_adjacent_M3_min.append(min_charge)
                        double_non_adjacent_M3_max.append(max_charge)
                  elif i == 4:
                        double_non_adjacent_M4_min.append(min_charge)
                        double_non_adjacent_M4_max.append(max_charge)
                  
            

# Convert lists to DataFrames for better visualization
df_double_adj_M1 = pd.DataFrame({"Min": double_adjacent_M1_min, "Max": double_adjacent_M1_max, "Sum": np.array(double_adjacent_M1_min) + np.array(double_adjacent_M1_max)})
df_double_adj_M2 = pd.DataFrame({"Min": double_adjacent_M2_min, "Max": double_adjacent_M2_max, "Sum": np.array(double_adjacent_M2_min) + np.array(double_adjacent_M2_max)})
df_double_adj_M3 = pd.DataFrame({"Min": double_adjacent_M3_min, "Max": double_adjacent_M3_max, "Sum": np.array(double_adjacent_M3_min) + np.array(double_adjacent_M3_max)})
df_double_adj_M4 = pd.DataFrame({"Min": double_adjacent_M4_min, "Max": double_adjacent_M4_max, "Sum": np.array(double_adjacent_M4_min) + np.array(double_adjacent_M4_max)})

df_double_non_adj_M1 = pd.DataFrame({"Min": double_non_adjacent_M1_min, "Max": double_non_adjacent_M1_max, "Sum": np.array(double_non_adjacent_M1_min) + np.array(double_non_adjacent_M1_max)})
df_double_non_adj_M2 = pd.DataFrame({"Min": double_non_adjacent_M2_min, "Max": double_non_adjacent_M2_max, "Sum": np.array(double_non_adjacent_M2_min) + np.array(double_non_adjacent_M2_max)})
df_double_non_adj_M3 = pd.DataFrame({"Min": double_non_adjacent_M3_min, "Max": double_non_adjacent_M3_max, "Sum": np.array(double_non_adjacent_M3_min) + np.array(double_non_adjacent_M3_max)})
df_double_non_adj_M4 = pd.DataFrame({"Min": double_non_adjacent_M4_min, "Max": double_non_adjacent_M4_max, "Sum": np.array(double_non_adjacent_M4_min) + np.array(double_non_adjacent_M4_max)})


# Same, but for three strip cases -----------------------------------------------------------------------------------------------
# Dictionaries to store min, mid, and max charge values for triple adjacent detections
triple_adjacent_M1_min, triple_adjacent_M1_mid, triple_adjacent_M1_max = [], [], []
triple_adjacent_M2_min, triple_adjacent_M2_mid, triple_adjacent_M2_max = [], [], []
triple_adjacent_M3_min, triple_adjacent_M3_mid, triple_adjacent_M3_max = [], [], []
triple_adjacent_M4_min, triple_adjacent_M4_mid, triple_adjacent_M4_max = [], [], []

triple_non_adjacent_M1_min, triple_non_adjacent_M1_mid, triple_non_adjacent_M1_max = [], [], []
triple_non_adjacent_M2_min, triple_non_adjacent_M2_mid, triple_non_adjacent_M2_max = [], [], []
triple_non_adjacent_M3_min, triple_non_adjacent_M3_mid, triple_non_adjacent_M3_max = [], [], []
triple_non_adjacent_M4_min, triple_non_adjacent_M4_mid, triple_non_adjacent_M4_max = [], [], []

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
        nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0]  # Get active strip indices
        charges = charge_matrix[event_idx, nonzero_strips]  # Get nonzero charges

        # Triple adjacent: 3 consecutive strips
        if count == 3 and np.all(np.diff(nonzero_strips) == 1):
            min_charge, mid_charge, max_charge = np.sort(charges)

            if i == 1:
                triple_adjacent_M1_min.append(min_charge)
                triple_adjacent_M1_mid.append(mid_charge)
                triple_adjacent_M1_max.append(max_charge)
            elif i == 2:
                triple_adjacent_M2_min.append(min_charge)
                triple_adjacent_M2_mid.append(mid_charge)
                triple_adjacent_M2_max.append(max_charge)
            elif i == 3:
                triple_adjacent_M3_min.append(min_charge)
                triple_adjacent_M3_mid.append(mid_charge)
                triple_adjacent_M3_max.append(max_charge)
            elif i == 4:
                triple_adjacent_M4_min.append(min_charge)
                triple_adjacent_M4_mid.append(mid_charge)
                triple_adjacent_M4_max.append(max_charge)

        # Triple non-adjacent: 3 non-consecutive strips
        if count == 3 and not np.all(np.diff(nonzero_strips) == 1):
            min_charge, mid_charge, max_charge = np.sort(charges)

            if i == 1:
                triple_non_adjacent_M1_min.append(min_charge)
                triple_non_adjacent_M1_mid.append(mid_charge)
                triple_non_adjacent_M1_max.append(max_charge)
            elif i == 2:
                triple_non_adjacent_M2_min.append(min_charge)
                triple_non_adjacent_M2_mid.append(mid_charge)
                triple_non_adjacent_M2_max.append(max_charge)
            elif i == 3:
                triple_non_adjacent_M3_min.append(min_charge)
                triple_non_adjacent_M3_mid.append(mid_charge)
                triple_non_adjacent_M3_max.append(max_charge)
            elif i == 4:
                triple_non_adjacent_M4_min.append(min_charge)
                triple_non_adjacent_M4_mid.append(mid_charge)
                triple_non_adjacent_M4_max.append(max_charge)

# Convert lists to DataFrames for better visualization
df_triple_adj_M1 = pd.DataFrame({"Min": triple_adjacent_M1_min, "Mid": triple_adjacent_M1_mid, "Max": triple_adjacent_M1_max, "Sum": np.array(triple_adjacent_M1_min) + np.array(triple_adjacent_M1_mid) + np.array(triple_adjacent_M1_max)})
df_triple_adj_M2 = pd.DataFrame({"Min": triple_adjacent_M2_min, "Mid": triple_adjacent_M2_mid, "Max": triple_adjacent_M2_max, "Sum": np.array(triple_adjacent_M2_min) + np.array(triple_adjacent_M2_mid) + np.array(triple_adjacent_M2_max)})
df_triple_adj_M3 = pd.DataFrame({"Min": triple_adjacent_M3_min, "Mid": triple_adjacent_M3_mid, "Max": triple_adjacent_M3_max, "Sum": np.array(triple_adjacent_M3_min) + np.array(triple_adjacent_M3_mid) + np.array(triple_adjacent_M3_max)})
df_triple_adj_M4 = pd.DataFrame({"Min": triple_adjacent_M4_min, "Mid": triple_adjacent_M4_mid, "Max": triple_adjacent_M4_max, "Sum": np.array(triple_adjacent_M4_min) + np.array(triple_adjacent_M4_mid) + np.array(triple_adjacent_M4_max)})

df_triple_non_adj_M1 = pd.DataFrame({"Min": triple_non_adjacent_M1_min, "Mid": triple_non_adjacent_M1_mid, "Max": triple_non_adjacent_M1_max, "Sum": np.array(triple_non_adjacent_M1_min) + np.array(triple_non_adjacent_M1_mid) + np.array(triple_non_adjacent_M1_max)})
df_triple_non_adj_M2 = pd.DataFrame({"Min": triple_non_adjacent_M2_min, "Mid": triple_non_adjacent_M2_mid, "Max": triple_non_adjacent_M2_max, "Sum": np.array(triple_non_adjacent_M2_min) + np.array(triple_non_adjacent_M2_mid) + np.array(triple_non_adjacent_M2_max)})
df_triple_non_adj_M3 = pd.DataFrame({"Min": triple_non_adjacent_M3_min, "Mid": triple_non_adjacent_M3_mid, "Max": triple_non_adjacent_M3_max, "Sum": np.array(triple_non_adjacent_M3_min) + np.array(triple_non_adjacent_M3_mid) + np.array(triple_non_adjacent_M3_max)})
df_triple_non_adj_M4 = pd.DataFrame({"Min": triple_non_adjacent_M4_min, "Mid": triple_non_adjacent_M4_mid, "Max": triple_non_adjacent_M4_max, "Sum": np.array(triple_non_adjacent_M4_min) + np.array(triple_non_adjacent_M4_mid) + np.array(triple_non_adjacent_M4_max)})

# ---------------------------------------------------------------------------------------------------------------------------------

# Create vectors of charge for single detection, double adjacent detections, triple adjacent detections and quadruple detections for each module

# Dictionaries to store charge values for single and quadruple detections
single_M1, single_M2, single_M3, single_M4 = [], [], [], []
quadruple_M1, quadruple_M2, quadruple_M3, quadruple_M4 = [], [], [], []

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
        nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0]  # Get active strip indices
        charges = charge_matrix[event_idx, nonzero_strips]  # Get nonzero charges

        # Single detection: exactly 1 strip has charge
        if count == 1:
            if i == 1:
                single_M1.append(charges[0])
            elif i == 2:
                single_M2.append(charges[0])
            elif i == 3:
                single_M3.append(charges[0])
            elif i == 4:
                single_M4.append(charges[0])

        # Quadruple detection: all 4 strips have charge
        if count == 4:
            total_charge = np.sum(charges)
            if i == 1:
                quadruple_M1.append(total_charge)
            elif i == 2:
                quadruple_M2.append(total_charge)
            elif i == 3:
                quadruple_M3.append(total_charge)
            elif i == 4:
                quadruple_M4.append(total_charge)

# Convert lists to DataFrames for better visualization
df_single_M1 = pd.DataFrame({"Charge": single_M1})
df_single_M2 = pd.DataFrame({"Charge": single_M2})
df_single_M3 = pd.DataFrame({"Charge": single_M3})
df_single_M4 = pd.DataFrame({"Charge": single_M4})

df_quadruple_M1 = pd.DataFrame({"Total Charge": quadruple_M1})
df_quadruple_M2 = pd.DataFrame({"Total Charge": quadruple_M2})
df_quadruple_M3 = pd.DataFrame({"Total Charge": quadruple_M3})
df_quadruple_M4 = pd.DataFrame({"Total Charge": quadruple_M4})

# Now create a dataframe of double and triple adjacent detections with the sums of the charges
df_single_M1_sum = df_single_M1["Charge"]
df_single_M2_sum = df_single_M2["Charge"]
df_single_M3_sum = df_single_M3["Charge"]
df_single_M4_sum = df_single_M4["Charge"]

df_double_adj_M1_sum = df_double_adj_M1["Sum"]
df_double_adj_M2_sum = df_double_adj_M2["Sum"]
df_double_adj_M3_sum = df_double_adj_M3["Sum"]
df_double_adj_M4_sum = df_double_adj_M4["Sum"]

df_triple_adj_M1_sum = df_triple_adj_M1["Sum"]
df_triple_adj_M2_sum = df_triple_adj_M2["Sum"]
df_triple_adj_M3_sum = df_triple_adj_M3["Sum"]
df_triple_adj_M4_sum = df_triple_adj_M4["Sum"]

df_quadruple_M1_sum = df_quadruple_M1["Total Charge"]
df_quadruple_M2_sum = df_quadruple_M2["Total Charge"]
df_quadruple_M3_sum = df_quadruple_M3["Total Charge"]
df_quadruple_M4_sum = df_quadruple_M4["Total Charge"]

df_total_M1 = pd.concat([df_single_M1_sum, df_double_adj_M1_sum, df_triple_adj_M1_sum, df_quadruple_M1_sum], axis=0)
df_total_M2 = pd.concat([df_single_M2_sum, df_double_adj_M2_sum, df_triple_adj_M2_sum, df_quadruple_M2_sum], axis=0)
df_total_M3 = pd.concat([df_single_M3_sum, df_double_adj_M3_sum, df_triple_adj_M3_sum, df_quadruple_M3_sum], axis=0)
df_total_M4 = pd.concat([df_single_M4_sum, df_double_adj_M4_sum, df_triple_adj_M4_sum, df_quadruple_M4_sum], axis=0)

df_single = pd.concat([df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum], axis=0)
df_double_adj = pd.concat([df_double_adj_M1_sum, df_double_adj_M2_sum, df_double_adj_M3_sum, df_double_adj_M4_sum], axis=0)
df_triple_adj = pd.concat([df_triple_adj_M1_sum, df_triple_adj_M2_sum, df_triple_adj_M3_sum, df_triple_adj_M4_sum], axis=0)
df_quadruple = pd.concat([df_quadruple_M1_sum, df_quadruple_M2_sum, df_quadruple_M3_sum, df_quadruple_M4_sum], axis=0)
df_total = pd.concat([df_single, df_double_adj, df_triple_adj, df_quadruple], axis=0)


#%%


# PLOT 4. AMOUNT OF STRIPS TRIGGERED --------------------------------------------------------------------------------------------

# Now count the number of single, double, triple and quadruple detections for each module and histogram it
# Create vectors of counts for single, double adjacent, triple adjacent, and quadruple detections for each module

df_single_M1_sum = df_single_M1_sum[ df_single_M1_sum > 0 ]
df_single_M2_sum = df_single_M2_sum[ df_single_M2_sum > 0 ]
df_single_M3_sum = df_single_M3_sum[ df_single_M3_sum > 0 ]
df_single_M4_sum = df_single_M4_sum[ df_single_M4_sum > 0 ]

df_double_adj_M1_sum = df_double_adj_M1_sum[ df_double_adj_M1_sum > 0 ]
df_double_adj_M2_sum = df_double_adj_M2_sum[ df_double_adj_M2_sum > 0 ]
df_double_adj_M3_sum = df_double_adj_M3_sum[ df_double_adj_M3_sum > 0 ]
df_double_adj_M4_sum = df_double_adj_M4_sum[ df_double_adj_M4_sum > 0 ]

df_triple_adj_M1_sum = df_triple_adj_M1_sum[ df_triple_adj_M1_sum > 0 ]
df_triple_adj_M2_sum = df_triple_adj_M2_sum[ df_triple_adj_M2_sum > 0 ]
df_triple_adj_M3_sum = df_triple_adj_M3_sum[ df_triple_adj_M3_sum > 0 ]
df_triple_adj_M4_sum = df_triple_adj_M4_sum[ df_triple_adj_M4_sum > 0 ]

df_quadruple_M1_sum = df_quadruple_M1_sum[ df_quadruple_M1_sum > 0 ]
df_quadruple_M2_sum = df_quadruple_M2_sum[ df_quadruple_M2_sum > 0 ]
df_quadruple_M3_sum = df_quadruple_M3_sum[ df_quadruple_M3_sum > 0 ]
df_quadruple_M4_sum = df_quadruple_M4_sum[ df_quadruple_M4_sum > 0 ]

# Compute total counts for normalization per module
total_counts = [
    len(df_single_M1_sum) + len(df_double_adj_M1_sum) + len(df_triple_adj_M1_sum) + len(df_quadruple_M1_sum),
    len(df_single_M2_sum) + len(df_double_adj_M2_sum) + len(df_triple_adj_M2_sum) + len(df_quadruple_M2_sum),
    len(df_single_M3_sum) + len(df_double_adj_M3_sum) + len(df_triple_adj_M3_sum) + len(df_quadruple_M3_sum),
    len(df_single_M4_sum) + len(df_double_adj_M4_sum) + len(df_triple_adj_M4_sum) + len(df_quadruple_M4_sum)
]

# Normalize counts relative to the total counts in each module
single_counts = [
    len(df_single_M1_sum) / total_counts[0],
    len(df_single_M2_sum) / total_counts[1],
    len(df_single_M3_sum) / total_counts[2],
    len(df_single_M4_sum) / total_counts[3]
]
double_adjacent_counts = [
    len(df_double_adj_M1_sum) / total_counts[0],
    len(df_double_adj_M2_sum) / total_counts[1],
    len(df_double_adj_M3_sum) / total_counts[2],
    len(df_double_adj_M4_sum) / total_counts[3]
]
triple_adjacent_counts = [
    len(df_triple_adj_M1_sum) / total_counts[0],
    len(df_triple_adj_M2_sum) / total_counts[1],
    len(df_triple_adj_M3_sum) / total_counts[2],
    len(df_triple_adj_M4_sum) / total_counts[3]
]
quadruple_counts = [
    len(df_quadruple_M1_sum) / total_counts[0],
    len(df_quadruple_M2_sum) / total_counts[1],
    len(df_quadruple_M3_sum) / total_counts[2],
    len(df_quadruple_M4_sum) / total_counts[3]
]

M1 = [single_counts[0], double_adjacent_counts[0], triple_adjacent_counts[0], quadruple_counts[0]]
M2 = [single_counts[1], double_adjacent_counts[1], triple_adjacent_counts[1], quadruple_counts[1]]
M3 = [single_counts[2], double_adjacent_counts[2], triple_adjacent_counts[2], quadruple_counts[2]]
M4 = [single_counts[3], double_adjacent_counts[3], triple_adjacent_counts[3], quadruple_counts[3]]

# Define the labels for the detection types
detection_types = ["Single", "Double\nAdjacent", "Triple\nAdjacent", "Quadruple"]

# Define colors for each module
module_colors = ["r", "orange", "g", "b"]  # Module 1: Red, Module 2: Green, Module 3: Blue, Module 4: Magenta

# Create a single plot for all modules
fig, ax = plt.subplots(figsize=(5, 4))

# Width for each bar in the grouped bar plot
bar_width = 0.2
x = np.arange(len(detection_types))  # X-axis positions

# Plot each module's normalized counts
selected_alpha = 0.6
ax.bar(x - 1.5 * bar_width, M1, width=bar_width, color=module_colors[0], alpha=selected_alpha, label="Plane 1")
ax.bar(x - 0.5 * bar_width, M2, width=bar_width, color=module_colors[1], alpha=selected_alpha, label="Plane 2")
ax.bar(x + 0.5 * bar_width, M3, width=bar_width, color=module_colors[2], alpha=selected_alpha, label="Plane 3")
ax.bar(x + 1.5 * bar_width, M4, width=bar_width, color=module_colors[3], alpha=selected_alpha, label="Plane 4")

# Formatting the plot
ax.set_xticks(x)
ax.set_xticklabels(detection_types)
ax.set_yscale("log")
ax.set_ylabel("Frequency")
# ax.set_title("Detection Type Distribution per Module (Normalized)")
ax.legend()
ax.grid(True, alpha=0.5, zorder=0, axis = "y")

def custom_formatter(x, _):
    if x >= 0.01:  # 1% or higher
        return f'{x:.0%}'
    else:  # Less than 1%
        return f'{x:.1%}'

# Apply the custom formatter
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

plt.tight_layout()
figure_name = "barplot_detection_type_distribution_per_module.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
plt.show()
plt.close()

#%%

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

# Parameters
selected_alpha = 0.7
bin_number = 250 # 250
right_lim = 4500
module_colors = ["r", "orange", "g", "b"]
n_events = 20000
bin_edges = np.linspace(0, right_lim, bin_number + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Define detection types and data
detection_types = ['Total', 'Single', 'Double Adjacent', 'Triple Adjacent', 'Quadruple']
df_data = [
    [df_total_M1, df_total_M2, df_total_M3, df_total_M4],
    [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum],
    [df_double_adj_M1_sum, df_double_adj_M2_sum, df_double_adj_M3_sum, df_double_adj_M4_sum],
    [df_triple_adj_M1_sum, df_triple_adj_M2_sum, df_triple_adj_M3_sum, df_triple_adj_M4_sum],
    [df_quadruple_M1_sum, df_quadruple_M2_sum, df_quadruple_M3_sum, df_quadruple_M4_sum],
]
singles = [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum]

# Step 1: Precompute 1–number_of_particles_bound_up single sums for each module
hist_basis_all_modules = []  # [ [H1, H2, ..., H6] for each module ]

number_of_particles_bound_up = 20

for single_data in singles:
    single_data = np.array(single_data)
    module_hists = []
    for n in range(1, number_of_particles_bound_up + 1):
        samples = np.random.choice(single_data, size=(n_events, n), replace=True).sum(axis=1)
        hist, _ = np.histogram(samples, bins=bin_edges, density=True)
        module_hists.append(hist)
    hist_basis_all_modules.append(np.stack(module_hists, axis=1))  # shape: (bins, 6)

# Step 2: Plot 5×2 grid
fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex='col')

import pandas as pd

coeff_tables = {dt: pd.DataFrame(index=[f"S{n}" for n in range(1, number_of_particles_bound_up + 1)],
                                 columns=["M1", "M2", "M3", "M4"])
                for dt in detection_types}

# Accumulate event-weighted contributions per module
component_counts = {
    "M1": np.zeros(number_of_particles_bound_up),
    "M2": np.zeros(number_of_particles_bound_up),
    "M3": np.zeros(number_of_particles_bound_up),
    "M4": np.zeros(number_of_particles_bound_up)
}

for i, (detection_type, df_group) in enumerate(zip(detection_types, df_data)):
    ax_hist = axes[i, 0]   # Left column: histograms and fit
    ax_scatter = axes[i, 1]  # Right column: scatter plot

    for j, (df, color, module) in enumerate(zip(df_group, module_colors, ['M1', 'M2', 'M3', 'M4'])):
        # Real data histogram
        counts_df, _ = np.histogram(df, bins=bin_edges, density=False)

        # Basis matrix A for this module (bins × 6)
        A = hist_basis_all_modules[j]

        # Fit: non-negative least squares
        coeffs, _ = nnls(A, counts_df)
        coeff_tables[detection_type].loc[:, module] = coeffs
        model = A @ coeffs  # predicted density
        
        # Get total number of events for that module and detection type
        n_events = len(df)
        # Weighted contribution = coeff * n_events
        component_counts[module] += coeffs * n_events
        
        # Plot histogram and model
        ax_hist.plot(bin_centers, counts_df, color=color, linestyle='-', label=f'{module} data')
        ax_hist.plot(bin_centers, model, color=color, linestyle='--', label=f'{module} fit')

        # Coefficients text
        coeff_text = " + ".join([f"{a:.3f}×S{idx+1}" for idx, a in enumerate(coeffs) if a > 0.001])
        ax_hist.text(0.02, 0.95 - j * 0.08, f"{module}: {coeff_text}", transform=ax_hist.transAxes,
                     fontsize=8, color=color, verticalalignment='top')

        # Scatter: frequency of singles (model) vs multiple (data)
        ax_scatter.scatter(model, counts_df, label=module, color=color, s=1)
        min_val = max(np.min(model[model > 0]), np.min(counts_df[counts_df > 0]))
        max_val = max(np.max(model), np.max(counts_df))
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='y = x' if j == 0 else None)

    # Format histogram panel
    ax_hist.set_title(f"{detection_type}")
    ax_hist.set_ylabel("Density")
    ax_hist.grid(True, alpha=0.5)
    ax_hist.legend(fontsize=8)

    # Format scatter panel
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.grid(True, alpha=0.5)
    ax_scatter.set_aspect('equal', 'box')
    ax_scatter.set_title("Model vs Data")
    ax_scatter.set_ylabel("Freq. (measured)")

# Final X labels
axes[-1, 0].set_xlabel("Charge (fC)")
axes[-1, 1].set_xlabel("Freq. (fitted model)")

# Layout & save
plt.suptitle(f"Charge Distributions and Scatter Model Fit (1–{number_of_particles_bound_up} Singles)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
figure_name = f"fit_and_scatter_sum_of_1_to_{number_of_particles_bound_up}_singles.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
plt.show()
plt.close()

#%%

# Normalize the columns to the sum of each column
coeff_tables_normalized = coeff_tables.copy()
for detection_type, df_coeffs in coeff_tables.items():
    coeff_tables_normalized[detection_type] = df_coeffs.div(df_coeffs.sum(axis=0), axis=1)

for detection_type, df_coeffs in coeff_tables_normalized.items():
    df_coeffs = df_coeffs.astype(float)
    df_percent = (df_coeffs * 100).round(1)
    print(f"\n===== Coefficients for {detection_type} (in %) =====")
    print(df_percent.to_string())  # Forces output in all environments
    
# %%

import matplotlib.pyplot as plt
import numpy as np

# Module colors
module_colors = ["r", "orange", "g", "b"]

# Create a vertical stack of plots: one per detection type
fig, axes = plt.subplots(len(coeff_tables_normalized), 1, figsize=(8, 14), sharex=True)

# Loop through each detection type and its coefficients
for i, (detection_type, df_coeffs) in enumerate(coeff_tables_normalized.items()):
    ax = axes[i]
    df_coeffs = df_coeffs.astype(float)
    df_percent = (df_coeffs * 100).round(1)

    x = np.arange(len(df_percent.index))  # S1 to S6 = positions on x-axis
    width = 0.1

    for j, module in enumerate(df_percent.columns):
      #   ax.bar(x + j * width, df_percent[module], alpha = 0.7, width=width, label=module, color=module_colors[j])
        ax.plot(x + j * width, df_percent[module], alpha = 0.7, label=module, color=module_colors[j])

    ax.set_title(f"{detection_type} - Coefficient Breakdown")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df_percent.index)
    ax.legend(title="Module", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.4)

# Final formatting
axes[-1].set_xlabel("Summed singles components (S1 to S6)")
plt.tight_layout()
plt.savefig(f"{figure_save_path}coefficients_barplots_per_type.png", dpi=600)
plt.show()
plt.close()

# %%

# Now, multiply each coefficient by the total number of events in the module and the type
# then sum them all up asnd obtain for each module a coeff. vs total number plot

# Final plot: total number of events per component per module
import matplotlib.pyplot as plt

components = [f"S{i}" for i in range(1, number_of_particles_bound_up + 1)]
x = np.arange(len(components))
width = 0.1
module_colors = ["r", "orange", "g", "b"]

fig, ax = plt.subplots(figsize=(8, 5))

for j, module in enumerate(component_counts.keys()):
#     ax.bar(x + j * width, component_counts[module] / np.sum( component_counts[module] ), width=width,
#            label=module, color=module_colors[j], alpha = 0.7)
    ax.plot(x + j * width, component_counts[module] / np.sum( component_counts[module] ),
           label=module, color=module_colors[j])

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(components)
ax.set_ylabel("Total Events (Weighted by Coefficients)")
ax.set_title("Total Event Contributions from Sums of 1–6 Singles per Module")
ax.legend(title="Module")
ax.grid(True, alpha=0.4)
ax.set_yscale("log")
ax.set_ylim(1e-5, 1.5)
ax.set_xlim(-0.1, 5)

plt.tight_layout()
plt.savefig(figure_save_path + "total_event_contributions_per_component.png", dpi=600)
plt.show()


# %%
