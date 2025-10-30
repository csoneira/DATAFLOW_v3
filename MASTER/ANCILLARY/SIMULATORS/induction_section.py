#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

from __future__ import annotations

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

import os
import sys
from pathlib import Path

import yaml
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]

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

from MASTER.common.execution_logger import start_timer

start_timer(__file__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

# Define the strip widths and their positions
strip_widths = [63, 63, 63, 98] # mm
strip_positions = np.cumsum([0] + strip_widths)  # Cumulative positions of strip edges
total_width = strip_positions[-1]

# Function to calculate cluster size for a given particle position and avalanche width
def calculate_cluster_size(position, avalanche_width):
      start = position - avalanche_width / 2
      end = position + avalanche_width / 2
      cluster_size = sum((start < strip_positions[i + 1]) and (end > strip_positions[i]) for i in range(len(strip_widths)))
      return cluster_size

# Simulation parameters
num_events = 100000  # Number of particles to simulate
avalanche_widths = np.linspace(1, 30, 100)  # mm. Range of avalanche widths to test
lut = []  # Lookup table to store results

# Perform the simulation
for avalanche_width in avalanche_widths:
      cluster_counts = [0, 0, 0, 0]  # To count occurrences of cluster sizes 1, 2, 3, 4
      for _ in range(num_events):
            position = np.random.uniform(0, total_width)  # Random particle position
            cluster_size = calculate_cluster_size(position, avalanche_width)
            if 1 <= cluster_size <= 4:
                  cluster_counts[cluster_size - 1] += 1

      # Calculate percentages for each cluster size
      cluster_percentages = [count / num_events for count in cluster_counts]
      lut.append([avalanche_width] + cluster_percentages)


#%%

# Round the LUT values to 3 decimal places
lut = [[round(value, 3) for value in row] for row in lut]

# Transform the LUT into a dataframe
lut_df = pd.DataFrame(lut, columns=["avalanche_width", "cluster_size_1", "cluster_size_2", "cluster_size_3", "cluster_size_4"])

print(lut_df)

# Save the LUT to a file
lut_df.to_csv(f"{home_path}/DATAFLOW_v3/MASTER/ANCILLARY/lut.csv", index=False)

print("LUT generated and saved to lut.csv")
# %%

# Extract data for plotting
avalanche_widths = [row[0] for row in lut]
cluster_size_percentages = list(zip(*[row[1:] for row in lut]))

# Plot the cluster size percentages as a function of avalanche width
plt.figure(figsize=(10, 6))
for i, percentages in enumerate(cluster_size_percentages):
      plt.plot(avalanche_widths, percentages, label=f"Cluster Size = {i + 1}")

plt.title("Cluster Size Percentages vs Avalanche Width")
plt.xlabel("Avalanche Width")
plt.ylabel("Percentage (%)")
plt.legend()
plt.grid(True)
plt.show()
# %%

# Bar plot for cluster size percentages for each avalanche width
cluster_sizes = [1, 2, 3, 4]

plt.figure(figsize=(12, 8))

# Create a bar plot for each avalanche width
bar_width = 0.8 / len(avalanche_widths)  # Adjust bar width to fit all avalanche widths
x_positions = np.arange(len(cluster_sizes))  # Base x positions for cluster sizes

for i, avalanche_width in enumerate(avalanche_widths):
      percentages = [row[1 + cluster_sizes.index(size)] for row in lut if row[0] == avalanche_width for size in cluster_sizes]
      plt.bar(
            x_positions + i * bar_width,
            percentages,
            bar_width,
            label=f"Avalanche Width = {avalanche_width:.1f}"
      )

plt.title("Cluster Size Percentages for Each Avalanche Width")
plt.xlabel("Cluster Size")
plt.ylabel("Percentage (%)")
plt.xticks(x_positions + bar_width * (len(avalanche_widths) - 1) / 2, cluster_sizes)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# %%
