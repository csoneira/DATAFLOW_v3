#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate Gaussian distributed vectors
xp_dev = 0.5
yp_dev = 0.55
xp = np.random.normal(loc=0, scale=xp_dev, size=100000)
yp = np.random.normal(loc=0, scale=yp_dev, size=100000)

# Define function to calculate angles
def calculate_angles(xproj, yproj):
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    return theta, phi

# Compute theta and phi
theta, phi = calculate_angles(xp, yp)

# Create a DataFrame for plotting
df_plot = pd.DataFrame({'xp': xp, 'yp': yp, 'theta': theta, 'phi': phi})
df_plot['charge_event'] = np.random.uniform(0, 100, len(df_plot))  # Dummy charge event data

#%%

# Define plotting parameters
columns_of_interest = ['phi', 'xp', 'yp', 'theta', 'charge_event']
num_bins = 100
fig, axes = plt.subplots(len(columns_of_interest), len(columns_of_interest), figsize=(15, 15))

for i in range(len(columns_of_interest)):
    for j in range(len(columns_of_interest)):
        ax = axes[i, j]
        if i < j:
            ax.axis('off')  # Leave the lower triangle blank
        elif i == j:
            # Diagonal: 1D histogram with independent axes
            hist_data = df_plot[columns_of_interest[i]]
            hist, bins = np.histogram(hist_data, bins=num_bins)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            norm = plt.Normalize(hist.min(), hist.max())
            cmap = plt.get_cmap('turbo')
            for k in range(len(hist)):
                ax.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(norm(hist[k])))
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Upper triangle: hexbin plots
            x_data = df_plot[columns_of_interest[j]]
            y_data = df_plot[columns_of_interest[i]]
            hb = ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')

        if i != len(columns_of_interest) - 1:
            ax.set_xticklabels([])  # Remove x-axis labels except for the last row
        if j != 0:
            ax.set_yticklabels([])  # Remove y-axis labels except for the first column
        if i == len(columns_of_interest) - 1:  # Last row, set x-labels
            ax.set_xlabel(columns_of_interest[j])
        if j == 0:  # First column, set y-labels
            ax.set_ylabel(columns_of_interest[i])

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.suptitle("Simulated data distribution", fontsize=16)
filename = f"xp{xp_dev}_yp{yp_dev}.png"
plt.savefig(filename)
plt.show()

# %%


# Correcting the error by explicitly using scipy.stats.norm
from scipy.stats import norm

# Fit Gaussians to xp and yp
xp_mean, xp_std = norm.fit(df_plot['xp'])
yp_mean, yp_std = norm.fit(df_plot['yp'])

# Define plotting parameters
columns_of_interest = ['phi', 'xp', 'yp', 'theta', 'charge_event']
num_bins = 100
fig, axes = plt.subplots(len(columns_of_interest), len(columns_of_interest), figsize=(15, 15))

for i in range(len(columns_of_interest)):
    for j in range(len(columns_of_interest)):
        ax = axes[i, j]
        if i < j:
            ax.axis('off')  # Leave the lower triangle blank
        elif i == j:
            # Diagonal: 1D histogram with independent axes
            hist_data = df_plot[columns_of_interest[i]]
            hist, bins = np.histogram(hist_data, bins=num_bins, density=True)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            cmap = plt.get_cmap('turbo')

            for k in range(len(hist)):
                ax.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(hist[k] / max(hist)))

            # If xp or yp, overlay Gaussian fit
            if columns_of_interest[i] == 'xp':
                x_vals = np.linspace(bin_centers.min(), bin_centers.max(), 100)
                ax.plot(x_vals, norm.pdf(x_vals, xp_mean, xp_std), 'r-', label=f"Mean={xp_mean:.2f}, Std={xp_std:.2f}")
                ax.legend(fontsize=8)

            elif columns_of_interest[i] == 'yp':
                y_vals = np.linspace(bin_centers.min(), bin_centers.max(), 100)
                ax.plot(y_vals, norm.pdf(y_vals, yp_mean, yp_std), 'r-', label=f"Mean={yp_mean:.2f}, Std={yp_std:.2f}")
                ax.legend(fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Upper triangle: hexbin plots
            x_data = df_plot[columns_of_interest[j]]
            y_data = df_plot[columns_of_interest[i]]
            hb = ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')

        if i != len(columns_of_interest) - 1:
            ax.set_xticklabels([])  # Remove x-axis labels except for the last row
        if j != 0:
            ax.set_yticklabels([])  # Remove y-axis labels except for the first column
        if i == len(columns_of_interest) - 1:  # Last row, set x-labels
            ax.set_xlabel(columns_of_interest[j])
        if j == 0:  # First column, set y-labels
            ax.set_ylabel(columns_of_interest[i])

print("-------------------")
print("XP std:", xp_std)
print("YP std:", yp_std)
print("-------------------")

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.suptitle("Simulated data distribution", fontsize=16)
plt.show()


# %%


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# Define number of Gaussian components
num_components = 1  # Change this to modify the number of Gaussian components

# Fit Gaussian Mixture Model (GMM) with specified components for xp and yp
gmm_xp = GaussianMixture(n_components=num_components, random_state=42)
gmm_xp.fit(df_plot[['xp']])

gmm_yp = GaussianMixture(n_components=num_components, random_state=42)
gmm_yp.fit(df_plot[['yp']])

# Extract standard deviations of the fitted Gaussians from the GMM
xp_std_devs = np.sqrt(gmm_xp.covariances_).flatten()
yp_std_devs = np.sqrt(gmm_yp.covariances_).flatten()

# Generate data for visualization
x_vals = np.linspace(df_plot['xp'].min(), df_plot['xp'].max(), 1000)
y_vals = np.linspace(df_plot['yp'].min(), df_plot['yp'].max(), 1000)

xp_gmm_pdf = np.exp(gmm_xp.score_samples(x_vals.reshape(-1, 1)))
yp_gmm_pdf = np.exp(gmm_yp.score_samples(y_vals.reshape(-1, 1)))

# Define plotting parameters
columns_of_interest = ['phi', 'xp', 'yp', 'theta', 'charge_event']
num_bins = 100
fig, axes = plt.subplots(len(columns_of_interest), len(columns_of_interest), figsize=(15, 15))

for i in range(len(columns_of_interest)):
    for j in range(len(columns_of_interest)):
        ax = axes[i, j]
        if i < j:
            ax.axis('off')  # Leave the lower triangle blank
        elif i == j:
            # Diagonal: 1D histogram with independent axes
            hist_data = df_plot[columns_of_interest[i]]
            hist, bins = np.histogram(hist_data, bins=num_bins, density=True)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            cmap = plt.get_cmap('turbo')

            for k in range(len(hist)):
                ax.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(hist[k] / max(hist)))

            # Overlay GMM fit with std devs in the legend
            if columns_of_interest[i] == 'xp':
                ax.plot(x_vals, xp_gmm_pdf, 'r-', label=f"Std devs: {', '.join([f'{std:.2f}' for std in xp_std_devs])}")
                ax.legend(fontsize=8)

            elif columns_of_interest[i] == 'yp':
                ax.plot(y_vals, yp_gmm_pdf, 'r-', label=f"Std devs: {', '.join([f'{std:.2f}' for std in yp_std_devs])}")
                ax.legend(fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Upper triangle: hexbin plots
            x_data = df_plot[columns_of_interest[j]]
            y_data = df_plot[columns_of_interest[i]]
            hb = ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')
            ax.set_facecolor(plt.cm.turbo(0))

        if i != len(columns_of_interest) - 1:
            ax.set_xticklabels([])  # Remove x-axis labels except for the last row
        if j != 0:
            ax.set_yticklabels([])  # Remove y-axis labels except for the first column
        if i == len(columns_of_interest) - 1:  # Last row, set x-labels
            ax.set_xlabel(columns_of_interest[j])
        if j == 0:  # First column, set y-labels
            ax.set_ylabel(columns_of_interest[i])

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.suptitle(f"Simulated data distribution with {num_components}-Component GMM Fit (Std Devs)", fontsize=16)
plt.show()

