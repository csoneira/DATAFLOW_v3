#%%

# Quick plotter for the article

remove_crosstalk = False
crosstalk_limit = 3.5 #2.6

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of file paths
# file_paths = [
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_07.08.27.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_06.03.09.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_04.57.44.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_03.52.13.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_02.47.20.txt"
# ]

file_paths = [
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_07.08.27.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_06.03.09.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_04.57.44.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_03.52.13.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_02.47.20.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_01.42.50.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_00.38.19.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_23.34.25.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_22.30.49.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_21.27.04.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_20.23.02.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_19.19.17.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_18.16.40.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_17.13.42.txt"
]


# file_paths = ["/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_00.38.19.txt"]

# file_paths = [
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_06.03.24.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_04.59.11.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_03.55.18.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_02.51.25.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_01.48.30.txt"
# ]


# Read and concatenate all files
df_list = [pd.read_csv(file, delimiter=",") for file in file_paths]  # Adjust delimiter if needed
merged_df = pd.concat(df_list, ignore_index=True)

# Drop duplicates if necessary
merged_df.drop_duplicates(inplace=True)

# Print the column names
print(merged_df.columns.to_list())

# If any value in any column that has Q* in it is smaller than 2.5, put it to 0
if remove_crosstalk:
      figure_save_path = "/home/cayetano/DATAFLOW_v3/CAL_FIGURES_ARTICLE_NO_CROSSTALK/"
      for col in merged_df.columns:
            if "Q_" in col and "s" in col:
                  merged_df[col] = merged_df[col].apply(lambda x: 0 if x < crosstalk_limit else x)
else:
      figure_save_path = "/home/cayetano/DATAFLOW_v3/CAL_FIGURES_ARTICLE/"

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
# plt.show()
plt.close()


# Take the columns that have all these characters: T, diff and cal
# Columns to consider
# columns = [col for col in merged_df.columns if "T" in col and "diff" in col and "cal" in col]


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

def interpolate_fast_charge(width):
      """
      Interpolates the Fast Charge for given Width values using the data table.

      Parameters:
      - width (float or np.ndarray): The Width value(s) to interpolate in ns.

      Returns:
      - float or np.ndarray: The interpolated Fast Charge value(s) in fC.
      """
      
      # Ensure calibration data is sorted
      width_table = FEE_calibration['Width'].to_numpy()
      fast_charge_table = FEE_calibration['Fast Charge'].to_numpy()
      
      if np.isscalar(width):  # If input is a single value
            return 0 if width == 0 else np.interp(width, width_table, fast_charge_table)
      
      width = np.asarray(width)  # Ensure input is a NumPy array
      result = np.interp(width, width_table, fast_charge_table)
      result[width == 0] = 0  # Keep zeros unchanged
      
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


# PLOT 1. HISTOGRAM OF MIN AND MAX CHARGE IN ADJACENT DOUBLE DETECTIONS --------------------------------------------------------

# Combine data from all four modules
df_double_adj_all = pd.concat([df_double_adj_M1, df_double_adj_M2, df_double_adj_M3, df_double_adj_M4], axis=0)

# Create a single plot
fig, ax = plt.subplots(figsize=(6, 4))

right_lim = 1200

# Plot histograms for Min, Max, and Sum
ax.hist(df_double_adj_all["Min"], bins=250, range=(0, right_lim), color="r", alpha=0.5, label="Minimum charge", density=True)
ax.hist(df_double_adj_all["Max"], bins=250, range=(0, right_lim), color="b", alpha=0.5, label="Maximum charge", density=True)
ax.hist(df_double_adj_all["Sum"], bins=250, range=(0, right_lim), color="g", alpha=0.5, label="Sum of charges", density=True)

# Set plot labels and formatting
# ax.set_title("Histogram of Min, Max, and Sum for Combined Modules")
ax.set_xlabel("Charge (fC)")
ax.set_ylabel("Frequency")
ax.set_xlim(-2, right_lim)
ax.grid(True, alpha=0.5, zorder=0)
ax.legend()

# Show the plot
plt.tight_layout()

figure_name = "histogram_min_max_sum_adjacent_double_detections.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
# plt.show()
plt.close()



# PLOT 2. THE SMILE FOR ALL THE DETECTOR -----------------------------------------------------------------------------------------

# Combine data from all four modules
df_double_adj_all = pd.concat([df_double_adj_M1, df_double_adj_M2, df_double_adj_M3, df_double_adj_M4], axis=0)

# Extract the necessary values
x = df_double_adj_all["Sum"]
y = (df_double_adj_all["Max"] - df_double_adj_all["Min"]) / df_double_adj_all["Sum"]

# Create a single 2D histogram plot
fig, ax = plt.subplots(figsize=(5, 4))
hist = ax.hist2d(x, y, bins=(150, 150), range=[[0, 2000], [0, 1]], cmap="turbo", cmin=1)

# Set labels and title
# ax.set_title("2D Histogram of Combined Modules")
ax.set_xlabel("Sum of charges (fC)")
ax.set_ylabel("Difference / Sum of charges")
ax.set_facecolor(hist[3].get_cmap()(0))
ax.grid(True, alpha=0.5, zorder=0)

# Add colorbar
cbar = plt.colorbar(hist[3], ax=ax)
cbar.set_label("Counts")

# Show the plot
plt.tight_layout()
figure_name = "2D_histogram_sum_diff_sum_adjacent_double_detections.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
# plt.show()
plt.close()


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

# PLOT 3. CHARGE DISTRIBUTIONS FOR DETECTION TYPES --------------------------------------------------------------------------------

# Create a single figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot histograms for each detection type
selected_alpha = 0.7
bin_number = 250
right_lim = 4500 # 150
ax.hist(df_total, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Total", histtype="step", linewidth=1.5)
ax.hist(df_single, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Single", histtype="step", linewidth=1.5)
ax.hist(df_double_adj, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Double Adjacent", histtype="step", linewidth=1.5,)
ax.hist(df_triple_adj, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Triple Adjacent", histtype="step", linewidth=1.5)
ax.hist(df_quadruple, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Quadruple", histtype="step", linewidth=1.5)

# Set plot labels and scaling
# ax.set_title("Charge Distributions for Detection Types")
ax.set_xlabel("Charge (fC)")
ax.set_ylabel("Frequency")
ax.set_xlim(-2, right_lim)
ax.set_yscale("log")  # Log scale for better visualization of frequency range
ax.grid(True, alpha=0.5, zorder=0)
ax.legend()

# Show the plot
plt.tight_layout()
figure_name = "histogram_charge_distributions_detection_types.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
# plt.show()
plt.close()


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
# plt.show()
plt.close()

