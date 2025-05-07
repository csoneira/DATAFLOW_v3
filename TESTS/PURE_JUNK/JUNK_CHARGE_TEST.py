#%%

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

# PLOT 1. HISTOGRAM OF MIN AND MAX CHARGE IN ADJACENT DOUBLE DETECTIONS --------------------------------------------------------

# # Combine data from all four modules
# df_double_adj_all = pd.concat([df_double_adj_M1, df_double_adj_M2, df_double_adj_M3, df_double_adj_M4], axis=0)

# # Create a single plot
# fig, ax = plt.subplots(figsize=(6, 4))

# right_lim = 1200

# # Plot histograms for Min, Max, and Sum
# ax.hist(df_double_adj_all["Min"], bins=250, range=(0, right_lim), color="r", alpha=0.5, label="Minimum charge", density=True)
# ax.hist(df_double_adj_all["Max"], bins=250, range=(0, right_lim), color="b", alpha=0.5, label="Maximum charge", density=True)
# ax.hist(df_double_adj_all["Sum"], bins=250, range=(0, right_lim), color="g", alpha=0.5, label="Sum of charges", density=True)

# # Set plot labels and formatting
# # ax.set_title("Histogram of Min, Max, and Sum for Combined Modules")
# ax.set_xlabel("Charge (fC)")
# ax.set_ylabel("Frequency")
# ax.set_xlim(-2, right_lim)
# ax.grid(True, alpha=0.5, zorder=0)
# ax.legend()

# # Show the plot
# plt.tight_layout()

# figure_name = "histogram_min_max_sum_adjacent_double_detections.png"
# plt.savefig(figure_save_path + figure_name, dpi=600)
# # plt.show()
# plt.close()



# PLOT 2. THE SMILE FOR ALL THE DETECTOR -----------------------------------------------------------------------------------------

# # Combine data from all four modules
# df_double_adj_all = pd.concat([df_double_adj_M1, df_double_adj_M2, df_double_adj_M3, df_double_adj_M4], axis=0)

# # Extract the necessary values
# x = df_double_adj_all["Sum"]
# y = (df_double_adj_all["Max"] - df_double_adj_all["Min"]) / df_double_adj_all["Sum"]

# # Create a single 2D histogram plot
# fig, ax = plt.subplots(figsize=(5, 4))
# hist = ax.hist2d(x, y, bins=(150, 150), range=[[0, 2000], [0, 1]], cmap="turbo", cmin=1)

# # Set labels and title
# # ax.set_title("2D Histogram of Combined Modules")
# ax.set_xlabel("Sum of charges (fC)")
# ax.set_ylabel("Difference / Sum of charges")
# ax.set_facecolor(hist[3].get_cmap()(0))
# ax.grid(True, alpha=0.5, zorder=0)

# # Add colorbar
# cbar = plt.colorbar(hist[3], ax=ax)
# cbar.set_label("Counts")

# # Show the plot
# plt.tight_layout()
# figure_name = "2D_histogram_sum_diff_sum_adjacent_double_detections.png"
# plt.savefig(figure_save_path + figure_name, dpi=600)
# # plt.show()
# plt.close()


# # PLOT 3. CHARGE DISTRIBUTIONS FOR DETECTION TYPES --------------------------------------------------------------------------------

# # Create a single figure
# fig, ax = plt.subplots(figsize=(6, 4))

# # Plot histograms for each detection type
# selected_alpha = 0.7
# bin_number = 250
# right_lim = 4500 # 150
# ax.hist(df_total, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Total", histtype="step", linewidth=1.5)
# ax.hist(df_single, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Single", histtype="step", linewidth=1.5)
# ax.hist(df_double_adj, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Double Adjacent", histtype="step", linewidth=1.5,)
# ax.hist(df_triple_adj, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Triple Adjacent", histtype="step", linewidth=1.5)
# ax.hist(df_quadruple, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label="Quadruple", histtype="step", linewidth=1.5)

# # Set plot labels and scaling
# # ax.set_title("Charge Distributions for Detection Types")
# ax.set_xlabel("Charge (fC)")
# ax.set_ylabel("Frequency")
# ax.set_xlim(-2, right_lim)
# ax.set_yscale("log")  # Log scale for better visualization of frequency range
# ax.grid(True, alpha=0.5, zorder=0)
# ax.legend()

# # Show the plot
# plt.tight_layout()
# figure_name = "histogram_charge_distributions_detection_types.png"
# plt.savefig(figure_save_path + figure_name, dpi=600)
# # plt.show()
# plt.close()

# PLOT 7: Summing

import matplotlib.pyplot as plt

# Define parameters
selected_alpha = 0.7
bin_number = 250
right_lim = 4500  # 4500
module_colors = ["r", "orange", "g", "b"]  # Module 1: Red, Module 2: Orange, Module 3: Green, Module 4: Blue

# Create a figure with 5 subplots (one per detection type), sharing the x-axis
fig, axes = plt.subplots(5, 1, figsize=(7, 15), sharex=True)

# Define detection types and their corresponding data
detection_types = ['Total', 'Single', 'Double Adjacent', 'Triple Adjacent', 'Quadruple']
df_data = [
    [df_total_M1, df_total_M2, df_total_M3, df_total_M4],
    [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum],
    [df_double_adj_M1_sum, df_double_adj_M2_sum, df_double_adj_M3_sum, df_double_adj_M4_sum],
    [df_triple_adj_M1_sum, df_triple_adj_M2_sum, df_triple_adj_M3_sum, df_triple_adj_M4_sum],
    [df_quadruple_M1_sum, df_quadruple_M2_sum, df_quadruple_M3_sum, df_quadruple_M4_sum],
]

singles = [
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
]

# Iterate over detection types and plot in the corresponding subplot
i = 0
for ax, detection_type, df_group, df_singles in zip(axes, detection_types, df_data, singles):
    j = 0
    for df, colors, module, single in zip(df_group, module_colors, ['M1', 'M2', 'M3', 'M4'], singles):
      #   print(np.array(single[0]))
        histogram_charge = i * np.array(single[j])
        j += 1
        ax.hist(histogram_charge, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label=f"{module} single", histtype="step", linewidth=1.5, density = True)
        ax.hist(df, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label=f"{module}", color=colors, histtype="step", linewidth=1.5, density = True)
        
    i += 1
#     ax.set_yscale("log")  # Log scale for better visualization
    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_title(f"Charge Distributions - {detection_type}")
    
    
# Set common labels
axes[-1].set_xlabel("Charge (fC)")
for ax in axes:
    ax.set_ylabel("Frequency")
    ax.legend(title="Module", loc='upper right', fontsize=8)

# Set layout and save figure
plt.tight_layout()
figure_name = "histogram_charge_distributions_per_detection_type.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
plt.show()
plt.close()

# %%

import matplotlib.pyplot as plt

# Define parameters
selected_alpha = 0.7
bin_number = 250
right_lim = 4500  # 4500
module_colors = ["r", "orange", "g", "b"]  # Module 1: Red, Module 2: Orange, Module 3: Green, Module 4: Blue

# Create a figure with 5 subplots (one per detection type), sharing the x-axis
fig, axes = plt.subplots(5, 1, figsize=(7, 15), sharex=True)
# fig, axes = plt.subplots(5, 1, figsize=(7, 15))

# Define detection types and their corresponding data
detection_types = ['Total', 'Single', 'Double Adjacent', 'Triple Adjacent', 'Quadruple']
df_data = [
    [df_total_M1, df_total_M2, df_total_M3, df_total_M4],
    [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum],
    [df_double_adj_M1_sum, df_double_adj_M2_sum, df_double_adj_M3_sum, df_double_adj_M4_sum],
    [df_triple_adj_M1_sum, df_triple_adj_M2_sum, df_triple_adj_M3_sum, df_triple_adj_M4_sum],
    [df_quadruple_M1_sum, df_quadruple_M2_sum, df_quadruple_M3_sum, df_quadruple_M4_sum],
]

singles = [
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
      [ df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum ],
]

# Iterate over detection types and plot in the corresponding subplot
i = 0
for ax, detection_type, df_group, df_singles in zip(axes, detection_types, df_data, singles):
    j = 0
    for df, colors, module, single in zip(df_group, module_colors, ['M1', 'M2', 'M3', 'M4'], singles):
      #   print(np.array(single[j]))
        histogram_charge = i * np.array(single[j])
        j += 1
        
      #   histogram_charge = np.array(single[0])
      #   ax.hist(histogram_charge, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label=f"{module} single", histtype="step", linewidth=1.5, density = True)
      #   ax.hist(df, bins=bin_number, range=(0, right_lim), alpha=selected_alpha, label=f"{module}", color=colors, histtype="step", linewidth=1.5, density = True)
        
        counts_histogram_charge, bins_charge = np.histogram(histogram_charge, bins=bin_number, range=(0, right_lim), density=False)
        counts_df, bins_df = np.histogram(df, bins=bin_number, range=(0, right_lim), density=False)

        # The bins edges are the same for both histograms, so we can use the bin edges for the x-axis
        bin_centers = (bins_charge[:-1] + bins_charge[1:]) / 2  # Midpoint of the bins
        
        # Scatter plot of counts_histogram_charge and counts_df
        ax.scatter(counts_histogram_charge, counts_df, label=f"{module}", color=colors, s=1)
      #   ax.set_xlim(0, 0.005)
      #   # Compute the difference in counts between the two histograms
      #   difference = abs( counts_histogram_charge - counts_df )
      #   ax.plot(bin_centers, difference / np.sum(difference), label="Difference", color='k', linewidth=1)
    
    i += 1
    ax.set_xscale("log")  # Log scale for better visualization
    ax.set_yscale("log")  # Log scale for better visualization
    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Charge Distributions - {detection_type}")
    
# Set common labels
axes[-1].set_xlabel("Frequency singles")
for ax in axes:
    ax.set_ylabel("Frequency multiple")
    ax.legend(title="Module", loc='upper right', fontsize=8)

# Set layout and save figure
plt.suptitle("Scatter plot of the frequency of singles vs the frequency of multiples")
plt.tight_layout()
figure_name = "histogram_charge_distributions_per_detection_type.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
plt.show()
plt.close()

# %%


# Joined plots

import matplotlib.pyplot as plt
import numpy as np

# Define parameters
selected_alpha = 0.7
bin_number = 250
right_lim = 4500
module_colors = ["r", "orange", "g", "b"]

# Create a figure with 5 rows and 2 columns (left: histogram, right: scatter)
fig, axes = plt.subplots(5, 2, figsize=(14, 15), sharex='col')

# Define detection types and corresponding data
detection_types = ['Total', 'Single', 'Double Adjacent', 'Triple Adjacent', 'Quadruple']
df_data = [
    [df_total_M1, df_total_M2, df_total_M3, df_total_M4],
    [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum],
    [df_double_adj_M1_sum, df_double_adj_M2_sum, df_double_adj_M3_sum, df_double_adj_M4_sum],
    [df_triple_adj_M1_sum, df_triple_adj_M2_sum, df_triple_adj_M3_sum, df_triple_adj_M4_sum],
    [df_quadruple_M1_sum, df_quadruple_M2_sum, df_quadruple_M3_sum, df_quadruple_M4_sum],
]

singles = [[df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum]] * 5

def sample_from_distribution(data, bins, n_samples):
    counts, bin_edges = np.histogram(data, bins=bins, range=(0, right_lim), density=True)
    pdf = counts / np.sum(counts)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0.0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Inverse transform sampling
    random_values = np.random.rand(n_samples)
    indices = np.searchsorted(cdf, random_values, side='right') - 1
    indices = np.clip(indices, 0, len(bin_centers) - 1)
    return bin_centers[indices]

# Plot each detection type
for i, (detection_type, df_group, df_singles) in enumerate(zip(detection_types, df_data, singles)):
    ax_hist = axes[i, 0]  # Left column: histogram
    ax_scatter = axes[i, 1]  # Right column: scatter

    for j, (df, color, module, single) in enumerate(zip(df_group, module_colors, ['M1', 'M2', 'M3', 'M4'], df_singles)):
        # Histogram plot
        
        single_data = np.array(single[j])

        # NEW: draw `i` samples from the distribution defined by `single_data`
        if i > 0:
            histogram_charge = sample_from_distribution(single_data, bin_number, i)
        else:
            histogram_charge = []  # For i=0, leave it empty or zero
        
        ax_hist.hist(histogram_charge, bins=bin_number, range=(0, right_lim), alpha=selected_alpha,
                     label=f"{module} single", histtype="step", linewidth=1.5, density=True)
        ax_hist.hist(df, bins=bin_number, range=(0, right_lim), alpha=selected_alpha,
                     label=f"{module}", color=color, histtype="step", linewidth=1.5, density=True)

        # Scatter plot
        counts_histogram_charge, _ = np.histogram(histogram_charge, bins=bin_number, range=(0, right_lim), density=False)
        counts_df, _ = np.histogram(df, bins=bin_number, range=(0, right_lim), density=False)
        ax_scatter.scatter(counts_histogram_charge, counts_df, label=f"{module}", color=color, s=1)

    # Formatting
    ax_hist.grid(True, alpha=0.5, zorder=0)
    ax_hist.set_title(f"{detection_type} - Histogram")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend(loc='upper right', fontsize=7, title="Module")

    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.grid(True, alpha=0.5, zorder=0)
    ax_scatter.set_aspect('equal', 'box')
    ax_scatter.set_title(f"{detection_type} - Scatter")
    ax_scatter.legend(loc='upper right', fontsize=7, title="Module")

# Common x-labels
axes[-1, 0].set_xlabel("Charge (fC)")
axes[-1, 1].set_xlabel("Frequency singles")
axes[-1, 1].set_ylabel("Frequency multiple")

# General figure formatting
plt.suptitle("Charge Distributions per Detection Type", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
figure_name = "combined_charge_distributions.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
plt.show()
plt.close()



#%%



import matplotlib.pyplot as plt
import numpy as np

# Define parameters
selected_alpha = 0.7
bin_number = 250
right_lim = 4500
module_colors = ["r", "orange", "g", "b"]

# Create a figure with 5 subplots (one per detection type), sharing the x-axis
fig, axes = plt.subplots(5, 1, figsize=(7, 15), sharex=True)

# Define detection types and corresponding data
detection_types = ['Total', 'Single', 'Double Adjacent', 'Triple Adjacent', 'Quadruple']
df_data = [
    [df_total_M1, df_total_M2, df_total_M3, df_total_M4],
    [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum],
    [df_double_adj_M1_sum, df_double_adj_M2_sum, df_double_adj_M3_sum, df_double_adj_M4_sum],
    [df_triple_adj_M1_sum, df_triple_adj_M2_sum, df_triple_adj_M3_sum, df_triple_adj_M4_sum],
    [df_quadruple_M1_sum, df_quadruple_M2_sum, df_quadruple_M3_sum, df_quadruple_M4_sum],
]

# Use the same singles for all detection types
singles = [
    [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum]
] * 5

# Function to sample i values from the distribution of a given single
def sample_from_single(single_data, i, bins=250, range_max=4500):
    if i == 0:
        return np.array([])  # Avoid empty histogram
    counts, bin_edges = np.histogram(single_data, bins=bins, range=(0, range_max), density=True)
    pdf = counts / counts.sum()
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0.0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    random_values = np.random.rand(i)
    indices = np.searchsorted(cdf, random_values, side='right') - 1
    indices = np.clip(indices, 0, len(bin_centers) - 1)
    return bin_centers[indices]

# Iterate and plot
n_events = 10000  # number of simulated pileup events

for i, (ax, detection_type, df_group, df_singles) in enumerate(zip(axes, detection_types, df_data, singles)):
    for j, (df, color, module, single_data) in enumerate(zip(df_group, module_colors, ['M1', 'M2', 'M3', 'M4'], df_singles)):

        single_data = np.array(single_data)

        # Simulate histogram_charge as sum of i singles, repeated 10000 times
        if i > 0:
            samples = np.random.choice(single_data, size=(n_events, i), replace=True)
            histogram_charge = samples.sum(axis=1)
        else:
            histogram_charge = []

        ax.hist(histogram_charge, bins=bin_number, range=(0, right_lim), alpha=selected_alpha,
                label=f"{module} sampled sum", histtype="step", linewidth=1.5, density=True)
        ax.hist(df, bins=bin_number, range=(0, right_lim), alpha=selected_alpha,
                label=f"{module}", color=color, histtype="step", linewidth=1.5, density=True)

    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_title(f"Charge Distributions - {detection_type}")

# Set common labels
axes[-1].set_xlabel("Charge (fC)")
for ax in axes:
    ax.set_ylabel("Frequency")
    ax.legend(title="Module", loc='upper right', fontsize=8)

# Final layout
plt.tight_layout()
figure_name = "histogram_charge_distributions_per_detection_type.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
plt.show()
plt.close()

#%%


import matplotlib.pyplot as plt
import numpy as np

# Parameters
selected_alpha = 0.7
bin_number = 250
right_lim = 4500
module_colors = ["r", "orange", "g", "b"]
n_events = 10000  # number of simulated pileup events

# Create a figure with 5 subplots (one per detection type), sharing the x-axis
fig, axes = plt.subplots(5, 1, figsize=(7, 15), sharex=True)

# Define detection types and their corresponding data
detection_types = ['Total', 'Single', 'Double Adjacent', 'Triple Adjacent', 'Quadruple']
df_data = [
    [df_total_M1, df_total_M2, df_total_M3, df_total_M4],
    [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum],
    [df_double_adj_M1_sum, df_double_adj_M2_sum, df_double_adj_M3_sum, df_double_adj_M4_sum],
    [df_triple_adj_M1_sum, df_triple_adj_M2_sum, df_triple_adj_M3_sum, df_triple_adj_M4_sum],
    [df_quadruple_M1_sum, df_quadruple_M2_sum, df_quadruple_M3_sum, df_quadruple_M4_sum],
]

# Use the same singles for all detection types
singles = [
    [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum]
] * 5

# Iterate over detection types and plot in the corresponding subplot
for i, (ax, detection_type, df_group, df_singles) in enumerate(zip(axes, detection_types, df_data, singles)):
    for j, (df, color, module, single_data) in enumerate(zip(df_group, module_colors, ['M1', 'M2', 'M3', 'M4'], df_singles)):

        single_data = np.array(single_data)

        if i > 0:
            # Sample 10,000 sums of i values from single_data
            samples = np.random.choice(single_data, size=(n_events, i), replace=True)
            histogram_charge = samples.sum(axis=1)
        else:
            histogram_charge = []

        # Plot the simulated and measured histograms
        ax.hist(histogram_charge, bins=bin_number, range=(0, right_lim),
                alpha=selected_alpha, label=f"{module} sampled sum", histtype="step", linewidth=1.5, density=True)
        ax.hist(df, bins=bin_number, range=(0, right_lim),
                alpha=selected_alpha, label=f"{module}", color=color, histtype="step", linewidth=1.5, density=True)

    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_title(f"Charge Distributions - {detection_type}")

# Labels
axes[-1].set_xlabel("Charge (fC)")
for ax in axes:
    ax.set_ylabel("Frequency")
    ax.legend(title="Module", loc='upper right', fontsize=8)

# Layout and save
plt.tight_layout()
figure_name = "histogram_charge_distributions_sampled_from_singles.png"
plt.savefig(figure_save_path + figure_name, dpi=600)
plt.show()
plt.close()
