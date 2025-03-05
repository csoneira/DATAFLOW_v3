#%%

# Quick plotter for the article

remove_crosstalk = True


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# List of file paths
file_paths = [
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.13_05.27.05.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.13_04.26.05.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.13_03.25.34.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.13_02.24.51.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.13_01.23.35.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.13_00.22.35.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.12_23.21.57.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.12_22.20.43.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.12_21.18.59.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.12_20.17.47.txt",
    "/home/cayetano/DATAFLOW_v3/STATIONS/MINGO01/FIRST_STAGE/EVENT_DATA/FULL_LIST_EVENTS_DIRECTORY/full_list_events_2025.02.12_19.16.38.txt"
]


# Read and concatenate all files
df_list = [pd.read_csv(file, delimiter=",") for file in file_paths]  # Adjust delimiter if needed
merged_df = pd.concat(df_list, ignore_index=True)

# Drop duplicates if necessary
merged_df.drop_duplicates(inplace=True)

# Print the column names
# print(merged_df.columns.to_list())
# print("-----------------------------------------------------------------------")

# If any value in any column that has Q* in it is smaller than 2.5, put it to 0
if remove_crosstalk:
      figure_save_path = "/home/cayetano/DATAFLOW_v3/UNC_X_FIGURES_ARTICLE_NO_CROSSTALK/"
      for col in merged_df.columns:
            if "Q_" in col and "s" in col:
                  merged_df[col] = merged_df[col].apply(lambda x: 0 if x < 2.6 else x)
else:
      figure_save_path = "/home/cayetano/DATAFLOW_v3/UNC_X_FIGURES_ARTICLE/"

# Check if figures_save_path exists, create one in other case
import os
if not os.path.exists(figure_save_path):
      os.makedirs(figure_save_path)

# ---------------------------------------------------------------------------------------------------------------------------------
# Part 1. Uncertainty in X --------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# Take the columns that have all these characters: T, diff and not final
# Columns to consider
columns = [col for col in merged_df.columns if "T_diff" in col and not "final" in col]

# print(len(columns))
# print(columns)
# print("-----------------------------------------------------------------------")

# Create a dataframe with the columns that have T_diff in them
df = merged_df[columns]

# the columns are 'TX_T_diff_Y' being X the module and Y the strip. Rename the columns so they are PXsY
df.columns = [f"P{col.split('_')[0][1]}s{col.split('_')[-1]}" for col in df.columns]
# print(df.columns)

#%%

# Create a 4x4 subfigure
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i in range(1, 5):
      for j in range(1, 5):
            # Get the column name
            col_name = f"P{i}s{j}"
            
            # Plot the histogram
            v = df[col_name]
            v = v[v != 0]
            
            # v = v[(v > -1) & (v < 1)]
            # v = v[ ((v < -0.2) | (v > 0.2)) ]
            
            axs[i-1, j-1].hist(v, bins=200, range=(-1, 1))
            axs[i-1, j-1].set_title(col_name)
            axs[i-1, j-1].set_xlabel("T diff [ns]")
            axs[i-1, j-1].set_ylabel("Frequency")
            axs[i-1, j-1].grid(True)

plt.tight_layout()
plt.show()


# %%

# Create a new dataframe
df_new = pd.DataFrame()

# Iterate over the columns
for col in df.columns:
    # Get the values of the column
    v = df[col]
    
    # Create a new column with the negative values (reset index to align)
    df_new[col + "_neg"] = v[(v < -0.25) & (v > -1)].reset_index(drop=True)
    # df_new[col + "_neg"] = v[(v < 0)].reset_index(drop=True)
    
    # Create a new column with the positive values (reset index to align)
    df_new[col + "_pos"] = v[(v > 0.25) & (v < 1)].reset_index(drop=True)
    # df_new[col + "_pos"] = v[(v > 0)].reset_index(drop=True)


# %%

bin_number = 50

# Create a 4x4 subfigure
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i in range(1, 5):
      for j in range(1, 5):
            for sign in ["neg", "pos"]:
                  # Get the column name
                  col_name = f"P{i}s{j}_{sign}"
                  
                  # Plot the histogram
                  v = df_new[col_name]
                  v = v[v != 0]
                  axs[i-1, j-1].hist(v, bins=bin_number, range=(-1, 1))
                  axs[i-1, j-1].set_title(col_name)
                  axs[i-1, j-1].set_xlabel("T diff [ns]")
                  axs[i-1, j-1].set_ylabel("Frequency")
                  axs[i-1, j-1].grid(True)
plt.tight_layout()
plt.show()


diff_df = pd.DataFrame()

# Create a 4x4 subplot figure
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for i in range(1, 5):  # Loop over modules
    for j in range(1, 5):  # Loop over strips
        for sign in ["neg", "pos"]:
            # Get the column name
            col_name = f"P{i}s{j}_{sign}"
            
            # Extract non-zero values
            v = df_new[col_name].dropna()
            v = v[v != 0]
            
            if len(v) < 10:
                continue  # Skip if not enough data
            
            # Compute histogram (counts and bin edges)
            counts, bin_edges = np.histogram(v, bins=bin_number, range=(-1, 1), density=True)
            
            # Compute bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Compute the differential of the distribution (approximate derivative)
            differential = np.gradient(counts, bin_centers)

            # Plot histogram
            axs[i-1, j-1].hist(v, bins=100, range=(-1, 1), density=True, alpha=0.6, label="Histogram")

            # Plot differential
            axs[i-1, j-1].plot(bin_centers, differential, 'r-', label="Differential")
            
            xaxes = col_name + "_time"
            yaxes = col_name + "_diff"
            diff_df[xaxes] = bin_centers
            diff_df[yaxes] = differential
            
            # Formatting
            axs[i-1, j-1].set_title(col_name)
            axs[i-1, j-1].set_xlabel("T diff [ns]")
            axs[i-1, j-1].set_ylabel("Density & Differential")
            axs[i-1, j-1].grid(True, alpha=0.5)
            axs[i-1, j-1].legend()

plt.tight_layout()
plt.show()

# %%

new_bin_number = 40
new_diff_df = pd.DataFrame()

# Create a 4x4 subplot figure
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for i in range(1, 5):  # Loop over modules
    for j in range(1, 5):  # Loop over strips
    
        # Get the column name
        col_name = f"P{i}s{j}"
        
        # Extract non-zero values
        v = df[col_name].dropna()
        v = v[v != 0]
        
        if len(v) < 10:
            continue  # Skip if not enough data
        
        # Compute histogram (counts and bin edges)
        counts, bin_edges = np.histogram(v, bins=new_bin_number, range=(-1, 1), density=True)
        
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute the differential of the distribution (approximate derivative)
        differential = np.gradient(counts, bin_centers)

        # Plot histogram
        axs[i-1, j-1].hist(v, bins=100, range=(-1, 1), density=True, alpha=0.6, label="Histogram")

        # Plot differential
        axs[i-1, j-1].plot(bin_centers, differential, 'r-', label="Differential")
        
        xaxes = col_name + "_time"
        yaxes = col_name + "_diff"
        new_diff_df[xaxes] = bin_centers
        new_diff_df[yaxes] = differential
        
        # Formatting
        axs[i-1, j-1].set_title(col_name)
        axs[i-1, j-1].set_xlabel("T diff [ns]")
        axs[i-1, j-1].set_ylabel("Density & Differential")
        axs[i-1, j-1].grid(True, alpha=0.5)
        axs[i-1, j-1].legend()

plt.tight_layout()
plt.show()


# %%

fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for i in range(1, 5):  # Loop over modules
    for j in range(1, 5):  # Loop over strips
        for sign in ["neg", "pos"]:
            # Get the column name
            col_name = f"P{i}s{j}_{sign}"
            
            xaxes = col_name + "_time"
            yaxes = col_name + "_diff"
            
            # Plot differential
            x = diff_df[xaxes]
            y = diff_df[yaxes]
            
            if sign == "neg":
                  cond = (x < -0.35) & (x > -1)
            else:
                  cond = (x > 0.35) & (x < 1)
                  
            x = x[cond]
            y = y[cond]
            
            axs[i-1, j-1].plot(x, y, label="Differential")

            # Formatting
            axs[i-1, j-1].set_title(col_name)
            axs[i-1, j-1].set_xlabel("T diff [ns]")
            axs[i-1, j-1].set_ylabel("Density & Differential")
            axs[i-1, j-1].grid(True, alpha=0.5)
            axs[i-1, j-1].legend()

plt.tight_layout()
plt.show()

# %%


A_df = pd.DataFrame()
mu_df = pd.DataFrame()
sigma_df = pd.DataFrame()

# Define the Gaussian function
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Create the figure and subplots
fig, axs = plt.subplots(4, 4, figsize=(15, 15))

# Loop over modules and strips
for i in range(1, 5):  # Loop over modules
    for j in range(1, 5):  # Loop over strips
        for sign in ["neg", "pos"]:
            # Get the column name
            col_name = f"P{i}s{j}_{sign}"
            
            xaxes = col_name + "_time"
            yaxes = col_name + "_diff"
            
            # Plot differential
            x = diff_df[xaxes]
            y = diff_df[yaxes]
            
            if sign == "neg":
                cond = (x < -0.35) & (x > -1)
            else:
                cond = (x > 0.35) & (x < 1)
                
            x = x[cond]
            y = y[cond]
            
            axs[i-1, j-1].plot(x, y)

            # Fit a Gaussian to the data
            try:
                  # Initial guess for the parameters [a, mu, sigma]
                  initial_guess = [max(y), np.mean(x), np.std(x)]
                  params, _ = curve_fit(gaussian, x, y, p0=initial_guess)

                  # Extract the sigma value
                  A = params[0]
                  mu = params[1]
                  sigma = params[2]
                  
                  A_df[col_name] = [A]
                  mu_df[col_name] = [mu]
                  sigma_df[col_name] = [sigma]

                  # Plot the fitted Gaussian
                  x_fit = np.linspace(min(x), max(x), 500)
                  y_fit = gaussian(x_fit, *params)
                  axs[i-1, j-1].plot(x_fit, y_fit, label=f"Gaussian fit (σ={sigma:.2f} ns)")

            except RuntimeError:
                  print(f"Gaussian fit failed for {col_name}")

            # Formatting
            axs[i-1, j-1].set_title(col_name)
            axs[i-1, j-1].set_xlabel("T diff [ns]")
            axs[i-1, j-1].set_ylabel("Density & Differential")
            axs[i-1, j-1].grid(True, alpha=0.5)
            axs[i-1, j-1].legend()

plt.tight_layout()
plt.show()


og_sigma_df = sigma_df.copy()

#%%
sigma_df = abs(sigma_df) * 200 # In mm

# %%

# Create a vector of all the sigmas
sigma_values = sigma_df.values.flatten()  # Flatten in case it's a 2D array

plt.hist(sigma_values, bins=20, range=(10, 28), alpha=0.75)
plt.xlabel("σ [mm]")
plt.ylabel("Frequency")
plt.title("Histogram of σ")
plt.grid(True)
plt.show()



# %%

col_name = "P4s1"
bin_number = 150
diff_bin_number = 50 # 50

new_diff_df = pd.DataFrame()
factor = 1
factor_gaussian = 1

# Create a 4x4 subplot figure
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for i in range(1, 5):  # Loop over modules
    for j in range(1, 5):  # Loop over strips
    
        # Get the column name
        col_name = f"P{i}s{j}"
        
        # Extract non-zero values
        v = df[col_name].dropna()
        v = v[v != 0]
        
        if len(v) < 10:
            continue  # Skip if not enough data
        
        # Compute histogram (counts and bin edges)
        counts, bin_edges = np.histogram(v, bins=diff_bin_number, range=(-1, 1), density=True)
        
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute the differential of the distribution (approximate derivative)
        differential = np.gradient(counts, bin_centers)

        # Plot histogram
        axs[i-1, j-1].hist(v, bins=bin_number, range=(-1, 1), density=True, alpha=0.6, label="Histogram")

        # Plot differential
        differential[ bin_centers > 0 ] = -differential[ bin_centers > 0 ]
        differential = differential / max(differential)
        axs[i-1, j-1].plot(bin_centers, differential, 'r-', label="Differential")
        
        xaxes = col_name + "_time"
        yaxes = col_name + "_diff"
        new_diff_df[xaxes] = bin_centers
        new_diff_df[yaxes] = differential
        
        # Formatting
        axs[i-1, j-1].set_title(col_name)
        axs[i-1, j-1].set_xlabel("T diff [ns]")
        axs[i-1, j-1].set_ylabel("Density & Differential")
        axs[i-1, j-1].grid(True, alpha=0.5)
        axs[i-1, j-1].legend()

plt.tight_layout()
plt.show()

#%%

# Define the Gaussian function
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Create a single plot
plt.figure(figsize=(8, 5))

v = df[col_name]
v = v[v != 0]
plt.hist(v, bins=bin_number, range=(-1, 1), alpha=0.8, density = True, histtype = 'step', color='red')

plt.hist(v, bins=bin_number, range=(-1, 1), alpha=0.2, density = True, label="$T_{\mathrm{diff}}$", color='red')
plt.fill_between([-1,1], 0, -1, color='red', alpha=0.2)

# plt.title(col_name)
plt.grid(True, alpha=0.5, zorder=0)
plt.xlabel("Time (ns)")
plt.ylabel("Frequency")

xaxes = col_name + "_time"
yaxes = col_name + "_diff"

x = new_diff_df[xaxes].copy()
y = new_diff_df[yaxes].copy()

x_og = x.copy()
y_og = y.copy()

# Smooth the y values in an x radius of 0.5 around 0 applying a rolling mean window
# but keep the y values out of that radius the same
window_size = 1
for i in range(len(x)):
    if x[i] > 0.4:
        y[i] = y[i]
    elif x[i] < -0.4:
        y[i] = y[i]
    else:
        y[i] = np.mean(y[i - window_size:i + window_size])


# Plot the differential data
plt.plot(x, factor * y, label="$T_{\mathrm{diff}}$ derivative", color = "green", alpha = 0.35)
plt.scatter(x, factor * y, color = "green", alpha = 0.75, s = 13)

# Overlay the differential and Gaussian fit
for sign in ["neg", "pos"]:
    
    x = x_og.copy()
    y = y_og.copy()
    
    if sign == "neg":
        cond = (x < -0.25) & (x > -1)
    else:
        cond = (x > 0.25) & (x < 1)
    
    x = x[cond]
    y = y[cond]
    
    # Fit a Gaussian to the data
    try:
        # Initial guess for the parameters [a, mu, sigma]
        initial_guess = [max(y), np.mean(x), np.std(x)]
        params, _ = curve_fit(gaussian, x, y, p0=initial_guess)

        # Extract the sigma value
        A = params[0]
        mu = params[1]
        sigma = params[2]
        
    except RuntimeError:
        print(f"Gaussian fit failed for {col_name}")
    
    in_col_name = f"{col_name}_{sign}"
    
    # Plot the Gaussian fit
    # print(mu, sigma)
    x_fit = np.linspace(min(x), max(x), 500)
    plt.plot(x_fit, factor_gaussian * gaussian(x_fit, A, mu, sigma), linewidth = 2.5, label=f"Gaussian Fit, $\\sigma$ = {sigma:.3f} ns, $\Delta X$ = {sigma*20:.2g} cm")

# Add legend and finalize the plot
plt.xlim(-1, 1)
plt.ylim(-0.07, 1.3)
# plt.legend()

plt.legend(ncol=2)  # Adjust the number of columns

# Create a second x axes for values that are x*200 in mm, but only modify the xticks in that
ax2 = plt.gca().twiny()
ax2.set_xlim(-20, 20)
ax2.set_xlabel("Distance (cm)")
ax2.set_xticks(np.arange(-20, 20, 5))

plt.tight_layout()
plt.savefig(figure_save_path + f"{col_name}_gaussian_fit.png", dpi=300)
plt.show()

# %%

