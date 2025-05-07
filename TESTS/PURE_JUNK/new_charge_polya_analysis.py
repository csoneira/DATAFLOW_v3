#%%

# Quick plotter for the article

remove_crosstalk = True
crosstalk_limit = 3.5 #2.6

calibrate_charge = True

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

# file_paths = [
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_07.08.27.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_06.03.09.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_04.57.44.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_03.52.13.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_02.47.20.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_01.42.50.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_00.38.19.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_23.34.25.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_22.30.49.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_21.27.04.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_20.23.02.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_19.19.17.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_18.16.40.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.03_17.13.42.txt"
# ]


# file_paths = ["/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.04_00.38.19.txt"]

# file_paths = [
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_06.03.24.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_04.59.11.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_03.55.18.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_02.51.25.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.02.19_01.48.30.txt"
# ]

# MINGO01
# file_paths = [
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_11.56.46.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_10.54.34.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_09.53.04.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_08.51.18.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_07.49.29.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_06.47.40.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_05.46.19.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_04.45.39.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_03.44.47.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_02.44.23.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_01.43.48.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_00.42.46.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_23.41.09.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_22.39.00.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_21.37.02.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_20.34.57.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_19.33.11.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_18.31.42.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_17.29.42.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_16.28.36.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_15.27.20.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_14.24.38.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_13.22.23.txt"
# ]


# MINGO02
# file_paths = [
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_13.25.45.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_12.36.34.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_11.46.54.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_10.56.21.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_10.05.21.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_09.14.35.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_08.22.58.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_07.31.35.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_06.40.10.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_05.48.39.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_04.57.07.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_04.05.00.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_03.13.36.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_02.22.02.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_01.30.47.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.20_00.39.27.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_23.47.00.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_22.55.28.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_22.04.01.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_21.12.37.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_20.21.16.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_19.30.11.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_18.39.37.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_17.49.33.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_16.58.48.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_16.08.09.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_15.17.08.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_14.25.49.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_13.34.47.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_12.43.59.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_11.52.37.txt",
#     "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO02/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY/list_events_2025.03.19_11.01.07.txt"
# ]



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


# Read and concatenate all files
df_list = [pd.read_csv(file, delimiter=",") for file in file_paths]  # Adjust delimiter if needed
merged_df = pd.concat(df_list, ignore_index=True)

# Drop duplicates if necessary
merged_df.drop_duplicates(inplace=True)

# Print the column names
print(merged_df.columns.to_list())

# If any value in any column that has Q* in it is smaller than 2.5, put it to 0
if remove_crosstalk:
      figure_save_path = "/home/cayetano/DATAFLOW_v3/POLYA_CAL_FIGURES_ARTICLE_NO_CROSSTALK/"
      for col in merged_df.columns:
            if "Q_" in col and "s" in col:
                  merged_df[col] = merged_df[col].apply(lambda x: 0 if x < crosstalk_limit else x)
else:
      figure_save_path = "/home/cayetano/DATAFLOW_v3/POLYA_CAL_FIGURES_ARTICLE/"

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

# # Create the DataFrame
FEE_calibration = pd.DataFrame(FEE_calibration)

# def interpolate_fast_charge(width):
#     """
#     Interpolates the Fast Charge for given Width values using the data table.

#     Parameters:
#     - width (float or np.ndarray): The Width value(s) to interpolate in ns.

#     Returns:
#     - float or np.ndarray: The interpolated Fast Charge value(s) in fC.
#     """

#     # Ensure calibration data is sorted
#     width_table = FEE_calibration['Width'].to_numpy()
#     fast_charge_table = FEE_calibration['Fast Charge'].to_numpy()

#     if np.isscalar(width):  # If input is a single value
#             return 0 if width == 0 else np.interp(width, width_table, fast_charge_table)

#     width = np.asarray(width)  # Ensure input is a NumPy array
#     result = np.interp(width, width_table, fast_charge_table)
#     result[width == 0] = 0  # Keep zeros unchanged

#     return result


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


#%%

# Histogram the theta column
fig, ax = plt.subplots(figsize=(8, 6))
plt.hist(merged_df["theta"], bins=200)
plt.title("Theta")
plt.xlabel("Theta")
plt.ylabel("Frequency")
plt.grid(True)
figure_name = "theta.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
plt.show()
plt.close()

#%%

merged_df = merged_df[ merged_df["type"] == 1234]
# merged_df = merged_df[ (merged_df["s"] > -0.05) & (merged_df["s"] < 0.05) ]
merged_df = merged_df[ (merged_df["s"] > 0) & (merged_df["s"] < 0.01) ]

# Histogram the theta column
fig, ax = plt.subplots(figsize=(8, 6))
plt.hist(merged_df["s"], bins=200)
plt.title("s")
plt.xlabel("s")
plt.ylabel("Frequency")
plt.grid(True)
figure_name = "s.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
plt.show()
plt.close()

#%%

# merged_df = merged_df[ merged_df["theta"] < 0.2]
# merged_df = merged_df[ merged_df["type"] == 1234]

columns_to_drop = ['Time', 'CRT_avg', 'x', 'y', 'theta', 'phi', 't0', 's', 'type', 'charge_event']
merged_df = merged_df.drop(columns=columns_to_drop)

print(merged_df.columns.to_list())

# For all the columns apply the calibration and not change the name of the columns
if calibrate_charge:
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

# strips_selected = 1 # Select strip from 1 to 4

# Initialize lists to store charges for single-strip events in all four planes
single_tracks_M1, single_tracks_M2, single_tracks_M3, single_tracks_M4 = [], [], [], []

# Matrix with 4 columns and as many rows as events, each column is a module
single_strip = [[], [], [], []]
single_four_plane_strip = [[], [], [], []]

# Dictionary to hold charge matrices for each module
charge_matrices = {}

# Loop over modules (planes)
for i in range(1, 5):
      charge_matrix = np.zeros((len(merged_df), 4))  # Stores strip-wise charges for this module

      for j in range(1, 5):  # Loop over strips
            col_name = f"Q_M{i}s{j}"  # Column name
            v = merged_df[col_name].fillna(0).to_numpy()  # Ensure no NaNs
            charge_matrix[:, j - 1] = v  # Store strip charge

      charge_matrices[i] = charge_matrix  # Store the charge matrix for each module

# Loop through events and check for single-strip activity
for event_idx in range(len(merged_df)):
      single_strip_indices = []  # To track the single active strip in each module
      single_strip_charges = []  # To track the charge of the single strip
      valid_single_event = True  # Assume event is valid for single-strip case
      valid_four_plane_event = True  # Assume event is valid for four-plane case

      for i in range(1, 5):  # Loop over the four planes
            charge_vector = charge_matrices[i][event_idx, :]  # Get strip charge for this event and module
            nonzero_strips = np.where(charge_vector > 0)[0]  # Find active strips

            # if len(nonzero_strips) == 1 and nonzero_strips == strips_selected - 1:  # Only one strip must be active
            if len(nonzero_strips) == 1:  # Only one strip must be active
                  single_strip_indices.append(nonzero_strips[0])
                  single_strip_charges.append(charge_vector[nonzero_strips[0]])
                  single_strip[i - 1].append(charge_vector[nonzero_strips[0]])  # Store for general single-strip case
            else:
                  valid_single_event = False  # Not a single-strip event
            
            if len(nonzero_strips) == 0:  # If any module has no active strips, it's not a four-plane event
                  valid_four_plane_event = False

      if valid_single_event:
            single_tracks_M1.append(single_strip_charges[0])
            single_tracks_M2.append(single_strip_charges[1])
            single_tracks_M3.append(single_strip_charges[2])
            single_tracks_M4.append(single_strip_charges[3])

      if valid_four_plane_event:  # Ensure we are storing only valid four-plane single-strips
            for i in range(1, 5):  # Loop over modules (now correctly from 1 to 4)
                  charge_vector = charge_matrices[i][event_idx, :]  # Get strip charge for this event and module
                  nonzero_strips = np.where(charge_vector > 0)[0]  # Find active strips
                  # if len(nonzero_strips) == 1 and nonzero_strips == strips_selected - 1:  # Only one strip must be active
                  if len(nonzero_strips) == 1:  # Only one strip must be active
                        single_four_plane_strip[i - 1].append(charge_vector[nonzero_strips[0]])  # Store charge
                  else:
                        single_four_plane_strip[i - 1].append(0)  # Append 0 when multiple strips fire


# Convert lists to arrays for easier handling
single_strips_M1 = np.array(single_strip[0])
single_strips_M2 = np.array(single_strip[1])
single_strips_M3 = np.array(single_strip[2])
single_strips_M4 = np.array(single_strip[3])

single_four_plane_strips_M1 = np.array(single_four_plane_strip[0])
single_four_plane_strips_M2 = np.array(single_four_plane_strip[1])
single_four_plane_strips_M3 = np.array(single_four_plane_strip[2])
single_four_plane_strips_M4 = np.array(single_four_plane_strip[3])

# %%

show_plots = True
bin_number = 70

left_lim = 100
right_lim = 1000

if calibrate_charge == False:
      left_lim = 4
      right_lim = 40

# Histogram of single-strip charges for each module in the same plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(single_tracks_M1, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 1', histtype='step')
ax.hist(single_tracks_M2, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 2', histtype='step')
ax.hist(single_tracks_M3, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 3', histtype='step')
ax.hist(single_tracks_M4, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 4', histtype='step')
ax.set_title('Single-strip charges for each module when all planes have n=1 strips')
ax.set_xlabel('Charge')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True)
figure_name = "single_strip_charges_1.png"
plt.tight_layout()
plt.savefig(figure_save_path + figure_name, dpi=600)
if show_plots:
      plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(single_strips_M1, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 1', histtype='step')
ax.hist(single_strips_M2, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 2', histtype='step')
ax.hist(single_strips_M3, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 3', histtype='step')
ax.hist(single_strips_M4, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 4', histtype='step')
ax.set_title('Single-strip charges for each module no matter the number of planes crossed')
ax.set_xlabel('Charge')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True)
figure_name = "single_strip_charges_2.png"
plt.tight_layout()
plt.savefig(figure_save_path + figure_name, dpi=600)
if show_plots:
      plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(single_four_plane_strips_M1, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 1', histtype='step')
ax.hist(single_four_plane_strips_M2, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 2', histtype='step')
ax.hist(single_four_plane_strips_M3, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 3', histtype='step')
ax.hist(single_four_plane_strips_M4, bins=bin_number, range=(left_lim, right_lim), density = True, alpha=0.5, label='Module 4', histtype='step')
ax.set_title('Single-strip charges for each module when four planes triggered\neven though not all with single strip')
ax.set_xlabel('Charge')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True)
figure_name = "single_strip_charges_3.png"
plt.tight_layout()
plt.savefig(figure_save_path + figure_name, dpi=600)
if show_plots:
      plt.show()
plt.close()
# %%

import matplotlib.pyplot as plt

# Define plot parameters
show_plots = True
bin_number = 70
left_lim = 100
right_lim = 1000

if calibrate_charge == False:
      left_lim = 4
      right_lim = 40

# Data sets: one for each module
data_sets = [
    (single_tracks_M1, single_strips_M1, single_four_plane_strips_M1, "Module 1"),
    (single_tracks_M2, single_strips_M2, single_four_plane_strips_M2, "Module 2"),
    (single_tracks_M3, single_strips_M3, single_four_plane_strips_M3, "Module 3"),
    (single_tracks_M4, single_strips_M4, single_four_plane_strips_M4, "Module 4"),
]

# Create subplots (4 rows, 1 column)
fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

# Titles for the three cases
case_titles = [
    "Single-strip charges when all planes have n=1 strips",
    "Single-strip charges no matter the number of planes crossed",
    "Single-strip charges when four planes triggered\neven though not all with single strip"
]

# Loop over modules
for i, (single_tracks, single_strips, single_four_plane, module_title) in enumerate(data_sets):
    ax = axes[i]

    # Plot histograms for the three cases
    ax.hist(single_tracks, bins=bin_number, range=(left_lim, right_lim), density=True, 
            alpha=0.5, label=case_titles[0], histtype='step')
    
    ax.hist(single_strips, bins=bin_number, range=(left_lim, right_lim), density=True, 
            alpha=0.5, label=case_titles[1], histtype='step')
    
    ax.hist(single_four_plane, bins=bin_number, range=(left_lim, right_lim), density=True, 
            alpha=0.5, label=case_titles[2], histtype='step')

    ax.set_title(module_title)
    ax.set_ylabel("Frequency")
    ax.grid(True)
    ax.legend()

# Set x-label for the last subplot
axes[-1].set_xlabel("Charge")

plt.tight_layout()
figure_name = "comparison_single_strip_charges.png"
plt.savefig(figure_name, dpi=600)

if show_plots:
    plt.show()

plt.close()


# %%
