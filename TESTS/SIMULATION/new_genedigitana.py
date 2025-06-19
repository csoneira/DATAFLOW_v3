#%%

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# HEADER -----------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Clear all variables
globals().clear()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import poisson
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import builtins

# ------------------------------------------------------------------------------
# Parameter definitions --------------------------------------------------------
# ------------------------------------------------------------------------------

plot_time_windows = False
show_plots = True

# Parameters and Constants
EFFS = [0.92, 0.95, 0.94, 0.93]
# EFFS = [0.2, 0.1, 0.4, 0.3]

# Iterants
n = 1  # Change this value to select 1 out of every n values
# minutes_list = np.arange(1, 181, n)
minutes_list = np.arange(1, 2, n)
TIME_WINDOWS = sorted(set(f'{num}min' for num in minutes_list if num > 0), key=lambda x: int(x[:-3]))

CROSS_EVS_LOW = 5 # 7 and 5 show a 33% of difference, which is more than the CRs will suffer
CROSS_EVS_UPP = 7
number_of_rates = 1

# Flux, area and counts calculations
xlim = 1000 # mm
ylim = 1000 # mm
z_plane = 100 # mm
FLUX = 1/12/60 # cts/s/cm^2/sr
area = 2 * xlim * 2 * ylim / 100  # cm^2
cts_sr = FLUX * area
cts = cts_sr * 2 * np.pi
# print("Counts per second:", cts_sr)

if number_of_rates == 1:
     AVG_CROSSING_EVS_PER_SEC_ARRAY = [ (CROSS_EVS_LOW + CROSS_EVS_UPP) / 2 ]
else:
     AVG_CROSSING_EVS_PER_SEC_ARRAY = np.linspace(CROSS_EVS_LOW, CROSS_EVS_UPP, number_of_rates)

# AVG_CROSSING_EVS_PER_SEC = 5.8

Z_POSITIONS = [0, 150, 310, 345.5]
Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]
N_TRACKS = 20_000_000
BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)
VALID_CROSSING_TYPES = ['1234', '123', '234', '12',  '23', '34']
VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']

read_file = True
use_binary = False  # If True, will use a binary file instead of CSV
bin_filename = f"/home/cayetano/DATAFLOW_v3/TESTS/SIMULATION/simulated_tracks_{N_TRACKS}.pkl"
csv_filename = f"/home/cayetano/DATAFLOW_v3/TESTS/SIMULATION/simulated_tracks_{N_TRACKS}.csv"

if read_file == False:

      # ------------------------------------------------------------------------------
      # Function definitions ---------------------------------------------------------
      # ------------------------------------------------------------------------------

      def calculate_efficiency_uncertainty(N_measured, N_passed):
          with np.errstate(divide='ignore', invalid='ignore'):
              delta_eff = np.where(
                  N_passed > 0,
                  np.sqrt((N_measured / N_passed**2) + (N_measured**2 / N_passed**3)),
                  0
              )
          return delta_eff

      def generate_tracks_with_timestamps(df, n_tracks, xlim, ylim, z_plane, base_time, cts, cos_n=2):
          import numpy as np
          from scipy.stats import poisson
          from datetime import timedelta
          import builtins

          rng = np.random.default_rng()
          exponent = 1 / (cos_n + 1)
          random_numbers = rng.random((n_tracks, 5))

          random_numbers[:, 0] = (random_numbers[:, 0] * 2 - 1) * xlim
          random_numbers[:, 1] = (random_numbers[:, 1] * 2 - 1) * ylim
          random_numbers[:, 2] = random_numbers[:, 2] * 0 + z_plane
          random_numbers[:, 3] = random_numbers[:, 3] * (2 * np.pi) - np.pi
          random_numbers[:, 4] = np.arccos(random_numbers[:, 4] ** exponent)

          df[['X_gen', 'Y_gen', 'Z_gen', 'Phi_gen', 'Theta_gen']] = random_numbers

      #     time_column = []
      #     current_time = base_time
      #     while len(time_column) < len(df):
      #         n_events = poisson.rvs(cts)
      #         for _ in range(n_events):
      #             time_column.append(current_time)
      #             if len(time_column) >= len(df):
      #                 break
      #         current_time += timedelta(seconds=1)

      #     df['time'] = time_column[:len(df)]

      def calculate_intersections(df, z_positions):
          df['crossing_type'] = [''] * builtins.len(df)
          for i, z in enumerate(z_positions, start=1):
              adjusted_z = df['Z_gen']
              df[f'X_gen_{i}'] = df['X_gen'] + (z + adjusted_z) * np.tan(df['Theta_gen']) * np.cos(df['Phi_gen'])
              df[f'Y_gen_{i}'] = df['Y_gen'] + (z + adjusted_z) * np.tan(df['Theta_gen']) * np.sin(df['Phi_gen'])

              out_of_bounds = (df[f'X_gen_{i}'] < -150) | (df[f'X_gen_{i}'] > 150) | \
                              (df[f'Y_gen_{i}'] < -143.5) | (df[f'Y_gen_{i}'] > 143.5)
              df.loc[out_of_bounds, [f'X_gen_{i}', f'Y_gen_{i}']] = np.nan

              in_bounds_indices = ~out_of_bounds
              df.loc[in_bounds_indices, 'crossing_type'] += builtins.str(i)

      def generate_time_dependent_efficiencies(df):
      #     df['time_seconds'] = (df['time'] - BASE_TIME).dt.total_seconds()
          df['eff_theoretical_1'] = EFFS[0]
          df['eff_theoretical_2'] = EFFS[1]
          df['eff_theoretical_3'] = EFFS[2]
          df['eff_theoretical_4'] = EFFS[3]

      # def simulate_measured_points(df, y_widths, x_noise=5, uniform_choice=True):
      #     df['measured_type'] = [''] * builtins.len(df)
      #     for i in range(1, 5):
      #         for idx in df.index:
      #             eff = df.loc[idx, f'eff_theoretical_{i}']
      #             if np.random.rand() > eff:
      #                 df.at[idx, f'X_mea_{i}'] = np.nan
      #                 df.at[idx, f'Y_mea_{i}'] = np.nan
      #             else:
      #                 df.at[idx, f'X_mea_{i}'] = df.at[idx, f'X_gen_{i}'] + np.random.normal(0, x_noise)
      #                 y_gen = df.at[idx, f'Y_gen_{i}']
      #                 if not np.isnan(y_gen):
      #                     y_width = y_widths[0] if i in [1, 3] else y_widths[1]
      #                     y_positions = np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2
      #                     strip_index = np.argmin(np.abs(y_positions - y_gen))
      #                     strip_center = y_positions[strip_index]
      #                     if uniform_choice:
      #                         df.at[idx, f'Y_mea_{i}'] = np.random.uniform(
      #                             strip_center - y_width[strip_index] / 2,
      #                             strip_center + y_width[strip_index] / 2
      #                         )
      #                     else:
      #                         df.at[idx, f'Y_mea_{i}'] = strip_center
      #                     df.at[idx, 'measured_type'] += builtins.str(i)
      
      

      def fill_measured_type(df):
          df['filled_type'] = df['measured_type']
          df['filled_type'] = df['filled_type'].replace({'124': '1234', '134': '1234'})

      def linear_fit(z, a, b):
          return a * z + b

      def fit_tracks(df, z_positions):
          z_positions = np.array(z_positions)
          x_measured_cols = [f'X_mea_{i}' for i in range(1, 5)]
          y_measured_cols = [f'Y_mea_{i}' for i in range(1, 5)]

          num_rows = builtins.len(df)
          x_fit_results = np.full((num_rows, 4), np.nan)
          y_fit_results = np.full((num_rows, 4), np.nan)
          theta_fit_results = np.full(num_rows, np.nan)
          phi_fit_results = np.full(num_rows, np.nan)
          fitted_type_results = [''] * num_rows

          for sequential_idx, idx in enumerate(tqdm(df.index, desc="Fitting tracks")):
              x_measured = pd.to_numeric(df.loc[idx, x_measured_cols], errors='coerce').values
              y_measured = pd.to_numeric(df.loc[idx, y_measured_cols], errors='coerce').values
              valid_indices = ~np.isnan(x_measured) & ~np.isnan(y_measured)

              if np.sum(valid_indices) < 2:
                  continue

              x_valid, y_valid, z_valid = x_measured[valid_indices], y_measured[valid_indices], z_positions[valid_indices]

              try:
                  if len(z_valid) == 2:
                      dz = z_valid[1] - z_valid[0]
                      if dz == 0:
                          continue
                      dx = x_valid[1] - x_valid[0]
                      dy = y_valid[1] - y_valid[0]
                      slope_x = dx / dz
                      slope_y = dy / dz
                      intercept_x = x_valid[0] - slope_x * z_valid[0]
                      intercept_y = y_valid[0] - slope_y * z_valid[0]
                  else:
                      popt_x, _ = curve_fit(linear_fit, z_valid, x_valid)
                      popt_y, _ = curve_fit(linear_fit, z_valid, y_valid)
                      slope_x, intercept_x = popt_x
                      slope_y, intercept_y = popt_y

                  theta_fit = np.arctan(np.sqrt(slope_x**2 + slope_y**2))
                  phi_fit = np.arctan2(slope_y, slope_x)
                  theta_fit_results[sequential_idx] = theta_fit
                  phi_fit_results[sequential_idx] = phi_fit

                  fitted_type = ''
                  for i, z in enumerate(z_positions):
                      x_fit = slope_x * z + intercept_x
                      y_fit = slope_y * z + intercept_y
                      x_fit_results[sequential_idx, i] = x_fit
                      y_fit_results[sequential_idx, i] = y_fit
                      if -150 <= x_fit <= 150 and -143.5 <= y_fit <= 143.5:
                          fitted_type += builtins.str(i + 1)
                  fitted_type_results[sequential_idx] = fitted_type

              except (RuntimeError, TypeError):
                  continue

          df['Theta_fit'] = theta_fit_results
          df['Phi_fit'] = phi_fit_results
          df['fitted_type'] = fitted_type_results
          for i in range(1, 5):
              df[f'X_fit_{i}'] = x_fit_results[:, i - 1]
              df[f'Y_fit_{i}'] = y_fit_results[:, i - 1]
      
      # ----------------------------------------------------------------------------
      # Remaining part of the code (simulation loop and CSV saving) ----------------
      # ----------------------------------------------------------------------------
      
      # Create a dictionary to store DataFrames with two indices: AVG_CROSSING_EVS_PER_SEC and TIME_WINDOW
      results = {}

      for AVG_CROSSING_EVS_PER_SEC in AVG_CROSSING_EVS_PER_SEC_ARRAY:
          results[AVG_CROSSING_EVS_PER_SEC] = {}

          for time_window in TIME_WINDOWS:
              results[AVG_CROSSING_EVS_PER_SEC][time_window] = pd.DataFrame()
              
              print("Calculating tracks...")
              
              columns = ['X_gen', 'Y_gen', 'Theta_gen', 'Phi_gen', 'Z_gen'] + \
                        [f'X_gen_{i}' for i in range(1, 5)] + [f'Y_gen_{i}' for i in range(1, 5)] + \
                        ['crossing_type', 'measured_type', 'fitted_type', 'time']
              df_generated = pd.DataFrame(np.nan, index=np.arange(N_TRACKS), columns=columns)

              rng = np.random.default_rng()
              generate_tracks_with_timestamps(df_generated, N_TRACKS, xlim, ylim, z_plane, BASE_TIME, cts, cos_n=2)
              real_df = df_generated.copy()
              
              print("Tracks generated. Calculating intersections...")
              
              calculate_intersections(df_generated, Z_POSITIONS)
              df = df_generated[df_generated['crossing_type'].isin(VALID_CROSSING_TYPES)].copy()
              crossing_df = df.copy()
              
              print("Intersections calculated. Generating measured points...")

              generate_time_dependent_efficiencies(df)
              simulate_measured_points(df, Y_WIDTHS)
              
              print("Measured points generated. Filling measured type...")
              
              fill_measured_type(df)
              
              print("Measured type filled. Fitting tracks...")
              
              fit_tracks(df, Z_POSITIONS)

              columns_to_keep = ['time'] + [col for col in df.columns if 'eff_' in col] + \
                                [col for col in df.columns if '_type' in col or 'Theta_' in col or 'Phi_' in col]
              df = df[columns_to_keep]

              for col in df.columns:
                  if '_type' in col:
                      df[col] = df[col].replace('', np.nan)

              theta_phi_columns = [col for col in df.columns if 'Theta_' in col or 'Phi_' in col]
              df[theta_phi_columns] = df[theta_phi_columns].replace('', np.nan)

              df['Theta_cros'] = crossing_df['Theta_gen']
              df['Phi_cros'] = crossing_df['Phi_gen']

              if use_binary:
                  df.to_pickle(bin_filename)
                  print(f"DataFrame saved to {bin_filename}")
              else:
                  df.to_csv(csv_filename, index=False)
                  print(f"DataFrame saved to {csv_filename}")

else:      
      if use_binary:
            df = pd.read_pickle(bin_filename)
            print(f"DataFrame read from {bin_filename}")
      else:
            df = pd.read_csv(csv_filename)
            print(f"DataFrame read from {csv_filename}")
      
      for col in df.columns:
            if '_type' in col:
                  # To integer before going to string. TO INTEGER
                  df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                  df[col] = df[col].astype(str).str.strip()
                  df[col] = df[col].replace('nan', np.nan)



#%%


print("Unique crossing_type values:", df['crossing_type'].dropna().unique())
print("Unique measured_type values:", df['measured_type'].dropna().unique())
print("Unique fitted_type values:", df['fitted_type'].dropna().unique())

import matplotlib.pyplot as plt

# Define binning
theta_bins = np.linspace(0, np.pi / 2, 200)
phi_bins = np.linspace(-np.pi, np.pi, 200)
tt_lists = [ VALID_MEASURED_TYPES ]

for tt_list in tt_lists:
      
      # Create figure with 2 rows and 4 columns
      fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex='row')
      
      # First column: Generated angles
      axes[0, 0].hist(df['Theta_gen'], bins=theta_bins, histtype='step', label='All', color='black')
      axes[1, 0].hist(df['Phi_gen'], bins=phi_bins, histtype='step', label='All', color='black')
      axes[0, 0].set_title("Generated θ")
      axes[1, 0].set_title("Generated ϕ")

      # Second column: Crossing detector (θ_gen, ϕ_gen)
      # axes[0, 1].hist(crossing_df['Theta_gen'], bins=theta_bins, histtype='step', color='black', label='All')
      # axes[1, 1].hist(crossing_df['Phi_gen'], bins=phi_bins, histtype='step', color='black', label='All')
      # for tt in tt_list:
      #       sel = (crossing_df['crossing_type'] == tt)
      #       axes[0, 1].hist(crossing_df.loc[sel, 'Theta_gen'], bins=theta_bins, histtype='step', label=tt)
      #       axes[1, 1].hist(crossing_df.loc[sel, 'Phi_gen'], bins=phi_bins, histtype='step', label=tt)
      #       axes[0, 1].set_title("Crossing detector θ_gen")
      #       axes[1, 1].set_title("Crossing detector ϕ_gen")
      
      # Crossing detector (θ_gen, ϕ_gen) – now using df['Theta_cros'], df['Phi_cros']
      axes[0, 1].hist(df['Theta_cros'], bins=theta_bins, histtype='step', color='black', label='All')
      axes[1, 1].hist(df['Phi_cros'], bins=phi_bins, histtype='step', color='black', label='All')

      for tt in tt_list:
          sel = (df['crossing_type'] == tt)
          axes[0, 1].hist(df.loc[sel, 'Theta_cros'], bins=theta_bins, histtype='step', label=tt)
          axes[1, 1].hist(df.loc[sel, 'Phi_cros'], bins=phi_bins, histtype='step', label=tt)

      axes[0, 1].set_title("Crossing detector θ_gen")
      axes[1, 1].set_title("Crossing detector ϕ_gen")

      
      # Third column: Measured (θ_gen, ϕ_gen)
      axes[0, 2].hist(df['Theta_gen'], bins=theta_bins, histtype='step', color='black', label='All')
      axes[1, 2].hist(df['Phi_gen'], bins=phi_bins, histtype='step', color='black', label='All')
      for tt in tt_list:
            sel = (df['measured_type'] == tt)
            axes[0, 2].hist(df.loc[sel, 'Theta_gen'], bins=theta_bins, histtype='step', label=tt)
            axes[1, 2].hist(df.loc[sel, 'Phi_gen'], bins=phi_bins, histtype='step', label=tt)
            axes[0, 2].set_title("Measured tracks θ_gen")
            axes[1, 2].set_title("Measured tracks ϕ_gen")

      # Fourth column: Measured (θ_fit, ϕ_fit)
      
      axes[0, 3].hist(df['Theta_fit'], bins=theta_bins, histtype='step', color='black', label='All')
      axes[1, 3].hist(df['Phi_fit'], bins=phi_bins, histtype='step', color='black', label='All')
      for tt in tt_list:
            sel = (df['measured_type'] == tt)
            axes[0, 3].hist(df.loc[sel, 'Theta_fit'], bins=theta_bins, histtype='step', label=tt)
            axes[1, 3].hist(df.loc[sel, 'Phi_fit'], bins=phi_bins, histtype='step', label=tt)
            axes[0, 3].set_title("Measured tracks θ_fit")
            axes[1, 3].set_title("Measured tracks ϕ_fit")

      # Common settings
      for ax in axes.flat:
            ax.legend(fontsize='x-small')
            ax.grid(True)

      axes[1, 0].set_xlabel(r'$\phi$ [rad]')
      axes[0, 0].set_ylabel('Counts')
      axes[1, 0].set_ylabel('Counts')
      axes[0, 2].set_xlim(0, np.pi / 2)
      axes[1, 2].set_xlim(-np.pi, np.pi)

      fig.tight_layout()
      plt.show()

# %%


# Define binning
theta_bins = np.linspace(0, np.pi / 2, 300)
phi_bins = np.linspace(-np.pi, np.pi, 300)

tt_lists = [ VALID_MEASURED_TYPES]

for tt_list in tt_lists:
      
      n_tt = len(tt_list)
      fig, axes = plt.subplots(n_tt, 4, figsize=(20, 4 * n_tt), sharex=True, sharey=True)

      for i, tt in enumerate(tt_list):
            # First column: Generated
            h = axes[i, 0].hist2d(
                  df['Theta_gen'], df['Phi_gen'], bins=[theta_bins, phi_bins], cmap='viridis'
            )
            axes[i, 0].set_title(f"Generated θ-ϕ (all), TT={tt}")

            # Second column: Crossing
            # crossing_sel = crossing_df['crossing_type'] == tt
            # h = axes[i, 1].hist2d(
            #       crossing_df.loc[crossing_sel, 'Theta_gen'],
            #       crossing_df.loc[crossing_sel, 'Phi_gen'],
            #       bins=[theta_bins, phi_bins], cmap='viridis'
            # )
            # axes[i, 1].set_title("Crossing θ-ϕ")
            
            crossing_sel = df['crossing_type'] == tt
            h = axes[i, 1].hist2d(
                df.loc[crossing_sel, 'Theta_cros'],
                df.loc[crossing_sel, 'Phi_cros'],
                bins=[theta_bins, phi_bins],
                cmap='viridis'
            )
            axes[i, 1].set_title("Crossing θ–ϕ")

            # Third column: Measured (generated angles)
            meas_sel = df['measured_type'] == tt
            h = axes[i, 2].hist2d(
                  df.loc[meas_sel, 'Theta_gen'],
                  df.loc[meas_sel, 'Phi_gen'],
                  bins=[theta_bins, phi_bins], cmap='viridis'
            )
            axes[i, 2].set_title("Measured (gen) θ-ϕ")

            # Fourth column: Measured (fitted angles)
            h = axes[i, 3].hist2d(
                  df.loc[meas_sel, 'Theta_fit'],
                  df.loc[meas_sel, 'Phi_fit'],
                  bins=[theta_bins, phi_bins], cmap='viridis'
            )
            axes[i, 3].set_title("Measured (fit) θ-ϕ")

      # Common labels and formatting
      for ax in axes[:, 0]:
            ax.set_ylabel(r'$\phi$ [rad]')
      for ax in axes[-1, :]:
            ax.set_xlabel(r'$\theta$ [rad]')
      for ax in axes.flat:
            ax.grid(False)

      fig.tight_layout()
      plt.show()

#%%



# Define topologies to evaluate
tt_list = ['1234', '123', '234', '12', '23', '34']  # or VALID_MEASURED_TYPES

# Create figure
n_tt = len(tt_list)
fig, axes = plt.subplots(n_tt, 2, figsize=(7, 3 * n_tt), sharex=False, sharey=False)

size_of_point = 0.1
alpha_of_point = 0.015

for i, tt in enumerate(tt_list):
      sel = df['measured_type'] == tt

      # Theta: Generated vs Fitted
      theta_gen = df.loc[sel, 'Theta_gen']
      theta_fit = df.loc[sel, 'Theta_fit']
      ax_theta = axes[i, 0]
      ax_theta.scatter(theta_gen, theta_fit, s=size_of_point, alpha=alpha_of_point)
      ax_theta.plot([0, np.pi / 2], [0, np.pi / 2], 'k--', lw=1)
      ax_theta.set_xlabel(r'$\theta_{\mathrm{gen}}$ [rad]')
      ax_theta.set_ylabel(r'$\theta_{\mathrm{fit}}$ [rad]')
      ax_theta.set_title(f'TT={tt}: $\theta$ gen vs fit')
      ax_theta.set_xlim(0, np.pi / 2)
      ax_theta.set_ylim(0, np.pi / 2)
      ax_theta.grid(True)
      ax_theta.set_aspect('equal', adjustable='box')

      # Phi: Generated vs Fitted
      phi_gen = df.loc[sel, 'Phi_gen']
      phi_fit = df.loc[sel, 'Phi_fit']
      ax_phi = axes[i, 1]
      ax_phi.scatter(phi_gen, phi_fit, s=size_of_point, alpha=alpha_of_point)
      ax_phi.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', lw=1)
      ax_phi.set_xlabel(r'$\phi_{\mathrm{gen}}$ [rad]')
      ax_phi.set_ylabel(r'$\phi_{\mathrm{fit}}$ [rad]')
      ax_phi.set_title(f'TT={tt}: $\phi$ gen vs fit')
      ax_phi.set_xlim(-np.pi, np.pi)
      ax_phi.set_ylim(-np.pi, np.pi)
      ax_phi.grid(True)
      ax_phi.set_aspect('equal', adjustable='box')

      # Set axes equal for both theta and phi plots
      # for ax in axes.flat:
      #       ax.set_aspect('equal', adjustable='box')

plt.suptitle('Theta and Phi: Generated vs Fitted', fontsize=16, y=1.002)
plt.tight_layout()
plt.show()


# %%


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa

# --- helper --------------------------------------------------------------
def to_xyz(theta, phi):
    """Map (theta, phi) → (x = sinθ·sinφ, y = sinθ·cosφ, z = cosθ)."""
    return np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta)

# --- coarse binning ------------------------------------------------------
nbins = 20
x_bins = np.linspace(-1.0, 1.0, nbins + 1)
y_bins = np.linspace(-1.0, 1.0, nbins + 1)
z_bins = np.linspace(0, 1.0, nbins + 1)

x_cent = 0.5 * (x_bins[:-1] + x_bins[1:])
y_cent = 0.5 * (y_bins[:-1] + y_bins[1:])
z_cent = 0.5 * (z_bins[:-1] + z_bins[1:])

Xc, Yc, Zc = np.meshgrid(x_cent, y_cent, z_cent, indexing='ij')

# --- figures with subplots ------------------------------------------------
tt_list = ['1234', '123', '234', '12', '23', '34']
ncols = 3
nrows = 2

fig3d, axs3d = plt.subplots(nrows, ncols, figsize=(16, 10), subplot_kw={'projection': '3d'})
fig2d, axs2d = plt.subplots(nrows, ncols, figsize=(12, 7))

min_counts = 1
clip_quantile = 0.90

for idx, tt in enumerate(tt_list):
    sub = df[df['measured_type'] == tt]
    x_fit, y_fit, z_fit = to_xyz(sub['Theta_fit'].to_numpy(), sub['Phi_fit'].to_numpy())
    x_gen, y_gen, z_gen = to_xyz(sub['Theta_gen'].to_numpy(), sub['Phi_gen'].to_numpy())

    ix = np.digitize(x_fit, x_bins) - 1
    iy = np.digitize(y_fit, y_bins) - 1
    iz = np.digitize(z_fit, z_bins) - 1

    shape = (nbins, nbins, nbins)
    dx_sum = np.zeros(shape); dy_sum = np.zeros(shape); dz_sum = np.zeros(shape)
    counts = np.zeros(shape, dtype=int)

    np.add.at(dx_sum, (ix, iy, iz), x_gen - x_fit)
    np.add.at(dy_sum, (ix, iy, iz), y_gen - y_fit)
    np.add.at(dz_sum, (ix, iy, iz), z_gen - z_fit)
    np.add.at(counts, (ix, iy, iz), 1)

    valid = counts >= min_counts
    with np.errstate(invalid='ignore', divide='ignore'):
        dx_avg = np.where(valid, dx_sum / counts, 0.0)
        dy_avg = np.where(valid, dy_sum / counts, 0.0)
        dz_avg = np.where(valid, dz_sum / counts, 0.0)

    mag = np.sqrt(dx_avg**2 + dy_avg**2 + dz_avg**2)
    clip = np.quantile(mag[valid], clip_quantile) if np.any(valid) else 0.1
    dx_avg = np.clip(dx_avg, -clip, clip)
    dy_avg = np.clip(dy_avg, -clip, clip)
    dz_avg = np.clip(dz_avg, -clip, clip)
    mag = np.sqrt(dx_avg**2 + dy_avg**2 + dz_avg**2)

    xs = Xc[valid]; ys = Yc[valid]; zs = Zc[valid]
    us = dx_avg[valid]; vs = dy_avg[valid]; ws = dz_avg[valid]
    colours = mag[valid]

    norm = np.linalg.norm(np.vstack([us, vs, ws]), axis=0)
    norm[norm == 0] = 1

    ax3d = axs3d[idx // ncols, idx % ncols]
    ax3d.quiver(xs, ys, zs, us / norm, vs / norm, ws / norm,
                length=0.05, linewidth=0.5, cmap='viridis',
                normalize=True, array=colours)

    ax3d.set_title(f'Type: {tt}')
    ax3d.set_xlabel(r'$x = \sin\theta\sin\varphi$')
    ax3d.set_ylabel(r'$y = \sin\theta\cos\varphi$')
    ax3d.set_zlabel(r'$z = \cos\theta$')
    ax3d.set_box_aspect([1, 1, 0.6])

    # --- 2D plot ------------------------------------------------------
    mag2d = np.hypot(dx_avg, dy_avg)
    clip2d = np.quantile(mag2d[valid], clip_quantile) if np.any(valid) else 0.1
    dx_avg = np.clip(dx_avg, -clip2d, clip2d)
    dy_avg = np.clip(dy_avg, -clip2d, clip2d)
    mag2d = np.hypot(dx_avg, dy_avg)

    norm2d = np.where(mag2d == 0, 1.0, mag2d)
    U = dx_avg / norm2d
    V = dy_avg / norm2d

    ax2d = axs2d[idx // ncols, idx % ncols]
    q = ax2d.quiver(Xc[valid], Yc[valid], U[valid], V[valid], mag2d[valid],
                    cmap='viridis', scale=30, width=0.004,
                    headwidth=3, headlength=3)
    ax2d.set_title(f'Type: {tt}')
    ax2d.set_xlabel(r'$x = \sin\theta\sin\varphi$')
    ax2d.set_ylabel(r'$y = \sin\theta$')
    ax2d.set_xlim(-1.1, 1.1)
    ax2d.set_ylim(-1.1, 1.1)
    ax2d.set_aspect('equal')
    ax2d.grid(True, lw=0.3)
    ax2d.set_facecolor(cm.viridis(0))

# Adjust layout and colorbar
fig3d.tight_layout()
fig2d.tight_layout()
fig2d.subplots_adjust(right=0.9)
cbar_ax = fig2d.add_axes([0.92, 0.15, 0.015, 0.7])
fig2d.colorbar(q, cax=cbar_ax, label=r'$|\Delta \vec{r}|$')

plt.show()

#%%



# ----------------------------------------------------------------------
# 2-D histograms (θ, φ) comparison
# ----------------------------------------------------------------------

import os

x_bins = np.linspace(-1, 1, 50)
y_bins   = np.linspace(-1, 1, 50)

tt_list = ['1234', '123', '234', '12', '23', '34']  # or VALID_MEASURED_TYPES

PLOT_DIR = "/home/cayetano/DATAFLOW_v3/TESTS/SIMULATION"

groups = [tt_list]                             # list of topology lists
for tt_group in groups:
      n_tt = len(tt_group)
      fig, ax = plt.subplots(n_tt, 2, figsize=(9, 4*n_tt), sharex=True, sharey=True)

      for i, tt in enumerate(tt_group):
            sel = df["measured_type"] == tt
            
            # Take only the columns where Theta_fit, Phi_fit are in an interval
            # around theta_center, phi_center, which i can select, as well as the
            # radius of the interval r0
            df_plot = df.loc[sel].copy()
            
            theta_center = np.pi / 5
            phi_center = 1
            theta_radius = 0.05
            phi_radius = 0.1
            case = 'gen'  # or 'fit', 'gen'
            
            theta_mask = (df_plot[f"Theta_{case}"] > theta_center - theta_radius) & \
                         (df_plot[f"Theta_{case}"] < theta_center + theta_radius)
            phi_mask = (df_plot[f"Phi_{case}"] > phi_center - phi_radius) & \
                       (df_plot[f"Phi_{case}"] < phi_center + phi_radius)
            df_plot = df_plot[theta_mask & phi_mask].copy()

            # Measured (gen)
            t = df_plot["Theta_gen"]
            p = df_plot["Phi_gen"]
            ax[i,0].hist2d(np.sin(t) * np.sin(p), np.sin(t) * np.cos(p),
                           bins=[x_bins, y_bins], cmap="viridis")
            # ax[i,0].hist2d(df_plot["Theta_gen"], df_plot["Phi_gen"],
            #             bins=[theta_bins, phi_bins], cmap="viridis")
            ax[i,0].set_title("meas (gen)")

            # Measured (fit)
            t = df_plot["Theta_fit"]
            p = df_plot["Phi_fit"]
            ax[i,1].hist2d(np.sin(t) * np.sin(p), np.sin(t) * np.cos(p),
                           bins=[x_bins, y_bins], cmap="viridis")
            # ax[i,1].hist2d(df_plot["Theta_fit"], df_plot["Phi_fit"],
            #             bins=[theta_bins, phi_bins], cmap="viridis")
            ax[i,1].set_title("meas (fit)")
            
            # Put the tt for that case as a title
            ax[i,0].set_title(f"{tt} – gen")
            ax[i,1].set_title(f"{tt} – fit")
            
            # For both cases, axes equal
            ax[i,0].set_aspect('equal', adjustable='box')
            ax[i,1].set_aspect('equal', adjustable='box')

      for a in ax[:,0]: a.set_ylabel(r"$\phi$ [rad]")
      for a in ax[-1,:]: a.set_xlabel(r"$\theta$ [rad]")
      fig.tight_layout()
      plt.savefig(f"{PLOT_DIR}/hist2d_{'_'.join(tt_group)}.png", dpi=150)
      plt.show()
      plt.close()

#%%


# METHOD OF LIKELYHOOD

# ---------------------------------------------------------------------
# Coordinate transform: (u, v) = (sin θ sin φ, sin θ cos φ)
# ---------------------------------------------------------------------
df["u_fit"] = np.sin(df["Theta_fit"]) * np.sin(df["Phi_fit"])
df["v_fit"] = np.sin(df["Theta_fit"]) * np.cos(df["Phi_fit"])

df["u_gen"] = np.sin(df["Theta_gen"]) * np.sin(df["Phi_gen"])
df["v_gen"] = np.sin(df["Theta_gen"]) * np.cos(df["Phi_gen"])

# ---------------------------------------------------------------------
# 1. Build empirical conditional distributions P(θ_gen, φ_gen | bin(u_fit, v_fit))
# ---------------------------------------------------------------------
n_bins = 500
u_edges = np.linspace(-1.0, 1.0, n_bins + 1)
v_edges = np.linspace(-1.0, 1.0, n_bins + 1)

u_idx = np.digitize(df["u_fit"], u_edges) - 1
v_idx = np.digitize(df["v_fit"], v_edges) - 1
df["bin"] = list(zip(u_idx, v_idx))

# Map each bin to the list of (θ_gen, φ_gen) pairs observed there
mapping = {}
for idx, grp in df.groupby("bin"):
    mapping[idx] = grp[["Theta_gen", "Phi_gen"]].values

# ---------------------------------------------------------------------
# 2. For each event, draw a (θ_pred, φ_pred) from the empirical distribution of its bin
# ---------------------------------------------------------------------
def draw_pred(row):
    idx = row["bin"]
    candidates = mapping.get(idx, None)
    if candidates is None or len(candidates) == 0:
        # Fallback: keep the measured angles
        return pd.Series({"Theta_pred": row["Theta_fit"], "Phi_pred": row["Phi_fit"]})
    th, ph = candidates[np.random.randint(len(candidates))]
    return pd.Series({"Theta_pred": th, "Phi_pred": ph})

tqdm.pandas()

# Apply the function with progress tracking
df_pred = df.join(df.progress_apply(draw_pred, axis=1))


#%%



# ----------------------------------------------------------------------
# 2-D histograms (θ, φ) comparison
# ----------------------------------------------------------------------
theta_bins = np.linspace(0, np.pi/2, 150)
phi_bins   = np.linspace(-np.pi, np.pi, 150)

groups = [tt_list]
for tt_group in groups:
    n_tt = len(tt_group)
    fig, ax = plt.subplots(n_tt, 3, figsize=(12, 4*n_tt), sharex=True, sharey=True)

    for i, tt in enumerate(tt_group):
        sel = df_pred["measured_type"] == tt

        # Measured (gen)
        ax[i,0].hist2d(df_pred.loc[sel,"Theta_gen"], df_pred.loc[sel,"Phi_gen"],
                       bins=[theta_bins, phi_bins], cmap="viridis")
        ax[i,0].set_title("meas (gen)")

        # Measured (fit)
        ax[i,1].hist2d(df_pred.loc[sel,"Theta_fit"], df_pred.loc[sel,"Phi_fit"],
                       bins=[theta_bins, phi_bins], cmap="viridis")
        ax[i,1].set_title("meas (fit)")

        # Predicted
        ax[i,2].hist2d(df_pred.loc[sel,"Theta_pred"], df_pred.loc[sel,"Phi_pred"],
                       bins=[theta_bins, phi_bins], cmap="viridis")
        ax[i,2].set_title("pred")
        
        # Put the tt for that case as a title
        ax[i,0].set_title(f"{tt} – gen")
        ax[i,1].set_title(f"{tt} – fit")

    for a in ax[:,0]: a.set_ylabel(r"$\phi$ [rad]")
    for a in ax[-1,:]: a.set_xlabel(r"$\theta$ [rad]")
    fig.tight_layout()
    plt.savefig(f"{PLOT_DIR}/hist2d_{'_'.join(tt_group)}.png", dpi=150)
    plt.show()
    plt.close()


#%%


# ----------------------------------------------------------------------
# 5 · Scatter-matrix 6 columnas  (gen, fit, map)
# ----------------------------------------------------------------------

n_tt = len(tt_list)
fig, axes = plt.subplots(n_tt, 6, figsize=(20, 3.5*n_tt), sharex=False, sharey=False)

def diag(ax, xlim, ylim):
    ax.plot(xlim, ylim, "k--", lw=1)
    ax.set_aspect("equal")
#     ax.grid(True)

for i, tt in enumerate(tt_list):
      mask = df_pred["measured_type"] == tt
      th_g, ph_g = df_pred.loc[mask,"Theta_gen"], df_pred.loc[mask,"Phi_gen"]
      th_f, ph_f = df_pred.loc[mask,"Theta_fit"], df_pred.loc[mask,"Phi_fit"]
      th_m, ph_m = df_pred.loc[mask,"Theta_pred"], df_pred.loc[mask,"Phi_pred"]
    
      # Parameters for scatter plot
      scatter_size = 0.1
      scatter_alpha = 0.015

      # θ_gen vs θ_fit
      a=axes[i,0]; a.scatter(th_g,th_f,s=scatter_size,alpha=scatter_alpha); diag(a,[0,np.pi/2],[0,np.pi/2])
      a.set_xlabel(r"$\theta_{\rm gen}$"); a.set_ylabel(r"$\theta_{\rm fit}$")

      # φ_gen vs φ_fit
      a=axes[i,1]; a.scatter(ph_g,ph_f,s=scatter_size,alpha=scatter_alpha); diag(a,[-np.pi,np.pi],[-np.pi,np.pi])
      a.set_xlabel(r"$\phi_{\rm gen}$");  a.set_ylabel(r"$\phi_{\rm fit}$")

      # θ_gen vs θ_map
      a=axes[i,2]; a.scatter(th_g,th_m,s=scatter_size,alpha=scatter_alpha); diag(a,[0,np.pi/2],[0,np.pi/2])
      a.set_xlabel(r"$\theta_{\rm gen}$"); a.set_ylabel(r"$\theta_{\rm map}$")

      # φ_gen vs φ_map
      a=axes[i,3]; a.scatter(ph_g,ph_m,s=scatter_size,alpha=scatter_alpha); diag(a,[-np.pi,np.pi],[-np.pi,np.pi])
      a.set_xlabel(r"$\phi_{\rm gen}$");  a.set_ylabel(r"$\phi_{\rm map}$")

      # θ_fit vs θ_map
      a=axes[i,4]; a.scatter(th_f,th_m,s=scatter_size,alpha=scatter_alpha); diag(a,[0,np.pi/2],[0,np.pi/2])
      a.set_xlabel(r"$\theta_{\rm fit}$"); a.set_ylabel(r"$\theta_{\rm map}$")

      # φ_fit vs φ_map
      a=axes[i,5]; a.scatter(ph_f,ph_m,s=scatter_size,alpha=scatter_alpha); diag(a,[-np.pi,np.pi],[-np.pi,np.pi])
      a.set_xlabel(r"$\phi_{\rm fit}$");  a.set_ylabel(r"$\phi_{\rm map}$")

plt.suptitle("Angular reconstruction – likelihood map", y=1.02, fontsize=15)
plt.tight_layout(); plt.show()


# %%

