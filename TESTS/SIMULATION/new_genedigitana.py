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
n = 5  # Change this value to select 1 out of every n values
minutes_list = np.arange(1, 181, n)
TIME_WINDOWS = sorted(set(f'{num}min' for num in minutes_list if num > 0), key=lambda x: int(x[:-3]))
print(TIME_WINDOWS)

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
print("Counts per second:", cts_sr)

AVG_CROSSING_EVS_PER_SEC_ARRAY = np.linspace(CROSS_EVS_LOW, CROSS_EVS_UPP, number_of_rates)

# AVG_CROSSING_EVS_PER_SEC = 5.8

Z_POSITIONS = [0, 150, 310, 345.5]
Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]
N_TRACKS = 20_000_000
BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)
VALID_CROSSING_TYPES = ['1234', '123', '234', '12',  '23', '34']
VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']

read_file = False
csv_filename = f"/home/cayetano/DATAFLOW_v3/TESTS/SIMULATION/simulated_tracks_{N_TRACKS}.csv"

if read_file == True:

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

          time_column = []
          current_time = base_time
          while len(time_column) < len(df):
              n_events = poisson.rvs(cts)
              for _ in range(n_events):
                  time_column.append(current_time)
                  if len(time_column) >= len(df):
                      break
              current_time += timedelta(seconds=1)

          df['time'] = time_column[:len(df)]

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
          df['time_seconds'] = (df['time'] - BASE_TIME).dt.total_seconds()
          df['eff_theoretical_1'] = EFFS[0]
          df['eff_theoretical_2'] = EFFS[1]
          df['eff_theoretical_3'] = EFFS[2]
          df['eff_theoretical_4'] = EFFS[3]

      def simulate_measured_points(df, y_widths, x_noise=5, uniform_choice=True):
          df['measured_type'] = [''] * builtins.len(df)
          for i in range(1, 5):
              for idx in df.index:
                  eff = df.loc[idx, f'eff_theoretical_{i}']
                  if np.random.rand() > eff:
                      df.at[idx, f'X_mea_{i}'] = np.nan
                      df.at[idx, f'Y_mea_{i}'] = np.nan
                  else:
                      df.at[idx, f'X_mea_{i}'] = df.at[idx, f'X_gen_{i}'] + np.random.normal(0, x_noise)
                      y_gen = df.at[idx, f'Y_gen_{i}']
                      if not np.isnan(y_gen):
                          y_width = y_widths[0] if i in [1, 3] else y_widths[1]
                          y_positions = np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2
                          strip_index = np.argmin(np.abs(y_positions - y_gen))
                          strip_center = y_positions[strip_index]
                          if uniform_choice:
                              df.at[idx, f'Y_mea_{i}'] = np.random.uniform(
                                  strip_center - y_width[strip_index] / 2,
                                  strip_center + y_width[strip_index] / 2
                              )
                          else:
                              df.at[idx, f'Y_mea_{i}'] = strip_center
                          df.at[idx, 'measured_type'] += builtins.str(i)

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

      for avg_crossing in AVG_CROSSING_EVS_PER_SEC_ARRAY:
          results[avg_crossing] = {}
          for time_window in TIME_WINDOWS:
              results[avg_crossing][time_window] = pd.DataFrame()

          for AVG_CROSSING_EVS_PER_SEC in AVG_CROSSING_EVS_PER_SEC_ARRAY:
              print(AVG_CROSSING_EVS_PER_SEC)

              columns = ['X_gen', 'Y_gen', 'Theta_gen', 'Phi_gen', 'Z_gen'] + \
                        [f'X_gen_{i}' for i in range(1, 5)] + [f'Y_gen_{i}' for i in range(1, 5)] + \
                        ['crossing_type', 'measured_type', 'fitted_type', 'time']
              df_generated = pd.DataFrame(np.nan, index=np.arange(N_TRACKS), columns=columns)

              rng = np.random.default_rng()
              generate_tracks_with_timestamps(df_generated, N_TRACKS, xlim, ylim, z_plane, BASE_TIME, cts, cos_n=2)
              real_df = df_generated.copy()

              calculate_intersections(df_generated, Z_POSITIONS)
              df = df_generated[df_generated['crossing_type'].isin(VALID_CROSSING_TYPES)].copy()
              crossing_df = df.copy()

              generate_time_dependent_efficiencies(df)
              simulate_measured_points(df, Y_WIDTHS)
              fill_measured_type(df)
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

              csv_filename = f"/home/cayetano/DATAFLOW_v3/TESTS/SIMULATION/simulated_tracks_{N_TRACKS}.csv"
              df.to_csv(csv_filename, index=False)
              print(f"DataFrame saved to {csv_filename}")


else:
      df = pd.read_csv(csv_filename)
      
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
tt_lists = [ VALID_MEASURED_TYPES]

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


import matplotlib.pyplot as plt

# Define binning
theta_bins = np.linspace(0, np.pi / 2, 150)
phi_bins = np.linspace(-np.pi, np.pi, 150)


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

import matplotlib.pyplot as plt
import numpy as np

# Define topologies to evaluate
tt_list = ['1234', '123', '234', '12', '23', '34']  # or VALID_MEASURED_TYPES

# Create figure
n_tt = len(tt_list)
fig, axes = plt.subplots(n_tt, 2, figsize=(7, 3 * n_tt), sharex=False, sharey=False)

size_of_point = 0.1
alpha_of_point = 0.1

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
import pandas as pd

# Coarse binning to reduce visual clutter
theta_bins = np.linspace(0, np.pi / 2, 20)
phi_bins = np.linspace(-np.pi, np.pi, 20)
theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
Phi, Theta = np.meshgrid(phi_centers, theta_centers)

# Plot for each measured_type
tt_list = ['1234', '123', '234', '12', '23', '34']

for tt in tt_list:
      subdf = df[df['measured_type'] == tt]
      
      dtheta_avg = np.zeros_like(Phi)
      dphi_avg = np.zeros_like(Phi)
      counts = np.zeros_like(Phi)

      for i in range(len(theta_bins) - 1):
            for j in range(len(phi_bins) - 1):
                  mask = (
                  (subdf['Theta_fit'] >= theta_bins[i]) & (subdf['Theta_fit'] < theta_bins[i+1]) &
                  (subdf['Phi_fit'] >= phi_bins[j]) & (subdf['Phi_fit'] < phi_bins[j+1])
                  )
                  if np.any(mask):
                        delta_theta = subdf.loc[mask, 'Theta_gen'] - subdf.loc[mask, 'Theta_fit']
                        delta_phi = subdf.loc[mask, 'Phi_gen'] - subdf.loc[mask, 'Phi_fit']
                        dtheta_avg[i, j] = np.nanmean(delta_theta)
                        dphi_avg[i, j] = np.nanmean(delta_phi)
                        counts[i, j] = np.sum(mask)

      min_counts = 1
      valid = counts >= min_counts
      magnitude = np.sqrt(dtheta_avg**2 + dphi_avg**2)
      # Do the max deformation clipping with a quantile
      try:
            max_deformation = np.quantile(magnitude[valid], 0.9)  # Use 95th percentile for clipping
      except IndexError:
            max_deformation = 0.1
      dtheta_avg = np.clip(dtheta_avg, -max_deformation, max_deformation)
      dphi_avg = np.clip(dphi_avg, -max_deformation, max_deformation)

      U = dphi_avg[valid]
      V = dtheta_avg[valid]
      
      norm = np.sqrt(U**2 + V**2)
      long_arrows = False
      if long_arrows:
            norm = np.sqrt(U**2 + V**2) * 0 + 1
      U_plot = U / (norm + 1e-6)
      V_plot = V / (norm + 1e-6)
      norm = np.sqrt(U**2 + V**2)
      
      fig, ax = plt.subplots(figsize=(9, 5))
      q = ax.quiver(
            Phi[valid], Theta[valid], U_plot, V_plot, norm,
            cmap='viridis', scale=20, width=0.004, headwidth=3, headlength=3
      )
      cb = fig.colorbar(q, ax=ax, label='|Δ angle| [rad]')
      ax.set_title(rf'$\vec{{\Delta}}$ for {tt}: $(\phi_\mathrm{{fit}} - \phi_\mathrm{{gen}}, \theta_\mathrm{{fit}} - \theta_\mathrm{{gen}})$')
      ax.set_xlabel(r'$\phi_{\mathrm{fit}}$ [rad]')
      ax.set_ylabel(r'$\theta_{\mathrm{fit}}$ [rad]')
      ax.set_xlim(-np.pi, np.pi)
      ax.set_ylim(0, np.pi / 2)
      ax.grid(True)
      
      # Facecolor: the first value of viridis
      ax.set_facecolor(cm.viridis(0))
      
      plt.tight_layout()
      plt.show()


# %%


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ============================================
# Data filtering
# ============================================

df_ml = df.dropna(subset=["Theta_fit", "Phi_fit", "Theta_gen", "Phi_gen", "measured_type"]).copy()
measured_types = ['1234', '123', '234', '12', '23', '34']

# ============================================
# Preprocessing functions
# ============================================

def transform_input(df):
    theta = df["Theta_fit"].values.astype(np.float32)
    phi = df["Phi_fit"].values.astype(np.float32)
    return np.stack([theta, np.sin(phi), np.cos(phi)], axis=1)

def transform_output(df):
    theta = df["Theta_gen"].values.astype(np.float32)
    phi = df["Phi_gen"].values.astype(np.float32)
    return np.stack([theta, np.sin(phi), np.cos(phi)], axis=1)

def inverse_transform_output(y_pred):
    theta = y_pred[:, 0]
    phi = np.arctan2(y_pred[:, 1], y_pred[:, 2])
    return np.stack([theta, phi], axis=1)

# ============================================
# Improved Neural Network Model
# ============================================

class AngleCorrectionModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(3, 512),
              nn.ReLU(),
              nn.Dropout(0.2),

              nn.Linear(512, 512),
              nn.ReLU(),
              nn.Dropout(0.2),

              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Dropout(0.2),

              nn.Linear(256, 128),
              nn.ReLU(),

              nn.Linear(128, 64),
              nn.ReLU(),

              nn.Linear(64, 3)  # Salida: theta, sin(phi), cos(phi)
          )

      def forward(self, x):
          return self.net(x)


# ============================================
# Training per measured_type
# ============================================

models = {}

for tt in measured_types:
    subdf = df_ml[df_ml["measured_type"] == tt]

    X = transform_input(subdf)
    y = transform_output(subdf)

    if len(X) < 100:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    model = AngleCorrectionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # Training loop with validation
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_tensor).numpy()
        pred = scaler_y.inverse_transform(pred_scaled)
        pred_angles = inverse_transform_output(pred)
        true_angles = inverse_transform_output(scaler_y.inverse_transform(y_test_scaled))

    # Save everything
    models[tt] = {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "X_test": scaler_X.inverse_transform(X_test_scaled),
        "predictions": pred_angles,
        "true_y": true_angles,
    }

    # Optional evaluation
    mse_theta = mean_squared_error(true_angles[:, 0], pred_angles[:, 0])
    mse_phi = mean_squared_error(true_angles[:, 1], pred_angles[:, 1])
    print(f"{tt}: MSE_theta = {mse_theta:.4e}, MSE_phi = {mse_phi:.4e}")



#%%

# Apply trained models to all data and write predictions to df
df['Theta_pred'] = np.nan
df['Phi_pred'] = np.nan

for tt in models:
    sel = df['measured_type'] == tt
    if not np.any(sel):
        continue

    # Correct: transform to (theta, sin(phi), cos(phi))
    theta = df.loc[sel, "Theta_fit"].values.astype(np.float32)
    phi = df.loc[sel, "Phi_fit"].values.astype(np.float32)
    X_all = np.stack([theta, np.sin(phi), np.cos(phi)], axis=1)

    # Transform and predict
    X_scaled = models[tt]["scaler_X"].transform(X_all)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = models[tt]["model"]
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()
        pred_cartesian = models[tt]["scaler_y"].inverse_transform(pred_scaled)
        pred = np.stack([
            pred_cartesian[:, 0],
            np.arctan2(pred_cartesian[:, 1], pred_cartesian[:, 2])
        ], axis=1)

    # Store predictions
    df.loc[sel, 'Theta_pred'] = pred[:, 0]
    df.loc[sel, 'Phi_pred'] = pred[:, 1]


#%%

# Define topologies to evaluate
tt_list = ['1234', '123', '234', '12', '23', '34']

# Create figure
n_tt = len(tt_list)

# Create a figure with 4 columns: θ_gen vs θ_fit, φ_gen vs φ_fit, θ_gen vs θ_pred, φ_gen vs φ_pred
fig, axes = plt.subplots(n_tt, 6, figsize=(20, 3.5 * n_tt), sharex=False, sharey=False)

for i, tt in enumerate(tt_list):
    sel = df['measured_type'] == tt

    theta_gen = df.loc[sel, 'Theta_gen']
    phi_gen = df.loc[sel, 'Phi_gen']
    theta_fit = df.loc[sel, 'Theta_fit']
    phi_fit = df.loc[sel, 'Phi_fit']
    theta_pred = df.loc[sel, 'Theta_pred']
    phi_pred = df.loc[sel, 'Phi_pred']

    # θ_gen vs θ_fit
    ax = axes[i, 0]
    ax.scatter(theta_gen, theta_fit, s=0.5, alpha=0.3)
    ax.plot([0, np.pi/2], [0, np.pi/2], 'k--', lw=1)
    ax.set_xlabel(r'$\theta_{\mathrm{gen}}$ [rad]')
    ax.set_ylabel(r'$\theta_{\mathrm{fit}}$ [rad]')
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(0, np.pi/2)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # φ_gen vs φ_fit
    ax = axes[i, 1]
    ax.scatter(phi_gen, phi_fit, s=0.5, alpha=0.3)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', lw=1)
    ax.set_xlabel(r'$\phi_{\mathrm{gen}}$ [rad]')
    ax.set_ylabel(r'$\phi_{\mathrm{fit}}$ [rad]')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # θ_gen vs θ_pred
    ax = axes[i, 2]
    ax.scatter(theta_gen, theta_pred, s=0.5, alpha=0.3)
    ax.plot([0, np.pi/2], [0, np.pi/2], 'k--', lw=1)
    ax.set_xlabel(r'$\theta_{\mathrm{gen}}$ [rad]')
    ax.set_ylabel(r'$\theta_{\mathrm{pred}}$ [rad]')
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(0, np.pi/2)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # φ_gen vs φ_pred
    ax = axes[i, 3]
    ax.scatter(phi_gen, phi_pred, s=0.5, alpha=0.3)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', lw=1)
    ax.set_xlabel(r'$\phi_{\mathrm{gen}}$ [rad]')
    ax.set_ylabel(r'$\phi_{\mathrm{pred}}$ [rad]')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    # θ_fit vs θ_pred
    ax = axes[i, 4]
    ax.scatter(theta_fit, theta_pred, s=0.5, alpha=0.3)
    ax.plot([0, np.pi/2], [0, np.pi/2], 'k--', lw=1)
    ax.set_xlabel(r'$\theta_{\mathrm{fit}}$ [rad]')
    ax.set_ylabel(r'$\theta_{\mathrm{pred}}$ [rad]')
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(0, np.pi/2)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # φ_fit vs φ_pred
    ax = axes[i, 5]
    ax.scatter(phi_fit, phi_pred, s=0.5, alpha=0.3)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', lw=1)
    ax.set_xlabel(r'$\phi_{\mathrm{fit}}$ [rad]')
    ax.set_ylabel(r'$\phi_{\mathrm{pred}}$ [rad]')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
plt.suptitle('Angular Reconstruction: Fitted vs Predicted (per Topology)', fontsize=16, y=1.005)
plt.tight_layout()
plt.show()

# %%


