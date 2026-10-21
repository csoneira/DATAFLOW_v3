import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def process_csv(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    # Read the CSV file
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Process T*_T_diff_final and Y_* columns for M = 1 to 4
    processed_columns = {}
    for i in range(1, 5):  # Loop for M = 1 to 4
        t_col = f"T{i}_T_diff_final"
        y_col = f"Y_{i}"

        if t_col in data.columns and y_col in data.columns:
            x_col = f"X_{i}"
            data[x_col] = data[t_col] * 200  # Multiply T*_T_diff_final by 200
            processed_columns[x_col] = y_col
        else:
            print(f"Warning: Missing required columns {t_col} or {y_col} in the data.")
            continue

    # Ensure at least one valid pair exists
    if not processed_columns:
        print("Error: No valid column pairs found.")
        return

    # Filter out rows where abs(value) > 200 for any selected column
    columns_to_check = list(processed_columns.keys()) + list(processed_columns.values())
    data = data[np.all(data[columns_to_check].abs() <= 200, axis=1)]

    # Reorganize columns: X_1, Y_1, X_2, Y_2, ...
    reordered_columns = []
    for x_col, y_col in processed_columns.items():
        reordered_columns.extend([x_col, y_col])
    processed_data = data[['Time'] + reordered_columns]

    # Save the processed columns to a new CSV file
    output_file = "processed_data.csv"
    processed_data.to_csv(output_file, index=False, float_format='%.1f')
    print(f"Processed data saved to {output_file}")

    # Create histograms
    for i in range(1, 5):  # Loop for M = 1 to 4
        x_col = f"X_{i}"
        y_col = f"Y_{i}"
        if x_col in processed_data.columns and y_col in processed_data.columns:
            # Filter out rows where either X_* or Y_* is zero
            filtered_data = processed_data[(processed_data[x_col] != 0) & (processed_data[y_col] != 0)]
            
            if not filtered_data.empty:
                plt.hist2d(filtered_data[x_col], filtered_data[y_col], bins=50, cmap='viridis')
                plt.colorbar(label="Counts")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"2D Histogram: {x_col} vs {y_col}")
                histogram_file = f"histogram_{x_col}_vs_{y_col}.png"
                plt.savefig(histogram_file)
                plt.close()
                print(f"Histogram saved as {histogram_file}")
            else:
                print(f"No non-zero data for {x_col} vs {y_col}, skipping histogram.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 XY_extract.py <filename_path>")
    else:
        process_csv(sys.argv[1])
