#%%
from __future__ import annotations

#%%

import os
import yaml
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/config.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# HEADER -----------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Clear all variables
# globals().clear()

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import math
from scipy.sparse import load_npz, csc_matrix
from pathlib import Path
from typing  import Union
import numpy as np
import pandas as pd
import math
from scipy.sparse import load_npz, csc_matrix

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

from typing import Dict, Optional
import numpy as np
import pandas as pd
import math
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import math
from typing import Dict, Optional
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ------------------------------------------------------------------------------
# Parameter definitions --------------------------------------------------------
# ------------------------------------------------------------------------------

PLOT_DIR = f"{home_path}/DATAFLOW_v3/TESTS/SIMULATION"

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

# Flux, area and counts calculations, with z_plane = 6000 the x, y limits were on +-3000
z_plane = 6000 # mm
ylim = 6 * z_plane # mm
xlim = ylim # mm

cut_soon = False

# Take the first terminal argument, if there is one, and assign it to FLUX, esle put the 1/12/60 and print
import sys
import os

if len(sys.argv) > 1:
    print(f"Command line argument detected: {sys.argv[1]}")
    # Change the , by a .
    sys.argv[1] = sys.argv[1].replace(',', '.')
    
    try:
        FLUX = float(sys.argv[1])  # Read from command line argument
        print(f"Using provided FLUX value: {FLUX} cts/s/cm^2/sr")
        cut_soon = True
    except ValueError:
        print("Invalid FLUX value provided. Using default value of 0.009 cts/s/cm^2/sr.")
        FLUX =  1/12/60 # cts/s/cm^2/sr

area = 2 * xlim * 2 * ylim / 100  # cm^2
cts_sr = FLUX * area
cts = cts_sr * 2 * np.pi
# print("Counts per second:", cts_sr)

if number_of_rates == 1:
     AVG_CROSSING_EVS_PER_SEC_ARRAY = [ (CROSS_EVS_LOW + CROSS_EVS_UPP) / 2 ]
else:
     AVG_CROSSING_EVS_PER_SEC_ARRAY = np.linspace(CROSS_EVS_LOW, CROSS_EVS_UPP, number_of_rates)

# AVG_CROSSING_EVS_PER_SEC = 5.8
Z_POSITIONS = [0, 145, 290, 435]
Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]

# ----------------------------------------------
N_TRACKS = 1_000_000
# ----------------------------------------------

BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)
VALID_CROSSING_TYPES = ['1234', '123', '234', '12',  '23', '34']
VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']

read_file = True
use_binary = True  # If True, will use a binary file instead of CSV
bin_filename = f"{home_path}/DATAFLOW_v3/TESTS/SIMULATION/simulated_tracks_{N_TRACKS}.pkl"
csv_filename = f"{home_path}/DATAFLOW_v3/TESTS/SIMULATION/simulated_tracks_{N_TRACKS}.csv"

fistensor2 = False
if fistensor2:
    print("Running with configuration for fistensor2")
    bin_filename = f"/home/petgfn/ALL_MINITRASGO/MINGO_LUT/simulated_tracks_{N_TRACKS}.pkl"
    csv_filename = f"/home/petgfn/ALL_MINITRASGO/MINGO_LUT/simulated_tracks_{N_TRACKS}.csv"

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
    
    # def generate_tracks_with_timestamps(df, n_tracks, xlim, ylim, z_plane, base_time, cts, cos_n=2):
    #     import numpy as np
    #     from scipy.stats import poisson
    #     import pandas as pd

    #     rng = np.random.default_rng()
    #     exponent = 1 / (cos_n + 1)

    #     # Generate random variables for track positions and angles
    #     random_numbers = rng.random((n_tracks, 5))
    #     random_numbers[:, 0] = (random_numbers[:, 0] * 2 - 1) * xlim          # X
    #     random_numbers[:, 1] = (random_numbers[:, 1] * 2 - 1) * ylim          # Y
    #     random_numbers[:, 2] = z_plane                                        # Z (fixed)
    #     random_numbers[:, 3] = random_numbers[:, 3] * (2 * np.pi) - np.pi     # Phi
    #     random_numbers[:, 4] = np.arccos(random_numbers[:, 4] ** exponent)    # Theta

    #     df[['X_gen', 'Y_gen', 'Z_gen', 'Phi_gen', 'Theta_gen']] = random_numbers

    #     # Vectorized timestamp generation
    #     from datetime import timedelta
    #     time_stamps = []
    #     current_time = base_time
    #     while len(time_stamps) < n_tracks:
    #         n_events = poisson.rvs(cts)
    #         n_add = min(n_events, n_tracks - len(time_stamps))
    #         time_stamps.extend([current_time] * n_add)
    #         current_time += timedelta(seconds=1)

    #     df['time'] = pd.Series(time_stamps, index=df.index)
    
    
    def generate_valid_tracks(n_valid_needed, xlim, ylim, z_plane, base_time, cts, cos_n=2, batch_size=100_000, max_simulated=None):
        import numpy as np
        from scipy.stats import poisson
        import pandas as pd
        from datetime import timedelta

        rng = np.random.default_rng()
        exponent = 1 / (cos_n + 1)

        # Preallocate arrays
        X_gen_all = np.empty(n_valid_needed)
        Y_gen_all = np.empty(n_valid_needed)
        Z_gen_all = np.full(n_valid_needed, z_plane)
        Phi_gen_all = np.empty(n_valid_needed)
        Theta_gen_all = np.empty(n_valid_needed)
        crossing_type_all = np.empty(n_valid_needed, dtype=object)
        time_all = np.empty(n_valid_needed, dtype='O')  # object dtype for datetime

        total_simulated = 0
        total_retained = 0
        current_time = base_time

        print("Target:", n_valid_needed, "valid tracks")
        print("Starting generation...")

        while total_retained < n_valid_needed:
            print(
                f"Simulated: {total_simulated:>10,} | Retained: {total_retained:>10,} "
                f"| Efficiency: {total_retained / total_simulated:.2%}" if total_simulated else "Simulating first batch..."
            )

            # Generate batch
            rand = rng.random((batch_size, 5))
            X = (rand[:, 0] * 2 - 1) * xlim
            Y = (rand[:, 1] * 2 - 1) * ylim
            Phi = rand[:, 3] * (2 * np.pi) - np.pi
            Theta = np.arccos(rand[:, 4] ** exponent)
            tan_theta = np.tan(Theta)
            crossing_type = np.full(batch_size, '', dtype=object)

            for i, z in enumerate(Z_POSITIONS, start=1):
                dz = z + z_plane
                X_proj = X + dz * tan_theta * np.cos(Phi)
                Y_proj = Y + dz * tan_theta * np.sin(Phi)
                in_plane = (X_proj >= -150) & (X_proj <= 150) & (Y_proj >= -143.5) & (Y_proj <= 143.5)
                crossing_type[in_plane] += str(i)

            mask_valid = np.isin(crossing_type, VALID_CROSSING_TYPES)
            n_valid = np.count_nonzero(mask_valid)

            if n_valid == 0:
                print(f"Warning: no valid tracks in batch of {batch_size}.")
                total_simulated += batch_size
                continue

            n_store = min(n_valid, n_valid_needed - total_retained)
            indices = np.flatnonzero(mask_valid)[:n_store]

            # Assign timestamps based on Poisson counts per second
            n_assigned = 0
            while n_assigned < n_store:
                n_events = poisson.rvs(cts)
                n_add = min(n_events, n_store - n_assigned)
                time_all[total_retained + n_assigned : total_retained + n_assigned + n_add] = [current_time] * n_add
                current_time += timedelta(seconds=1)
                n_assigned += n_add

            # Store into preallocated arrays
            X_gen_all[total_retained : total_retained + n_store] = X[indices]
            Y_gen_all[total_retained : total_retained + n_store] = Y[indices]
            Phi_gen_all[total_retained : total_retained + n_store] = Phi[indices]
            Theta_gen_all[total_retained : total_retained + n_store] = Theta[indices]
            crossing_type_all[total_retained : total_retained + n_store] = crossing_type[indices]

            total_retained += n_store
            total_simulated += batch_size

            if max_simulated and total_simulated >= max_simulated:
                print(f"Reached max_simulated = {max_simulated} without collecting {n_valid_needed} valid tracks.")
                break

        df = pd.DataFrame({
            'X_gen': X_gen_all,
            'Y_gen': Y_gen_all,
            'Z_gen': Z_gen_all,
            'Phi_gen': Phi_gen_all,
            'Theta_gen': Theta_gen_all,
            'crossing_type': crossing_type_all,
            'time': time_all
        })

        print(f"Done. Total simulated: {total_simulated:,}, retained: {len(df):,} → efficiency: {len(df) / total_simulated:.3%}")
        return df



    def calculate_intersections(df, z_positions):
        import numpy as np

        n_tracks = len(df)
        crossing_array = np.full((n_tracks,), '', dtype=object)

        for i, z in enumerate(z_positions, start=1):
            dz = z + df['Z_gen']
            tan_theta = np.tan(df['Theta_gen'])

            X_proj = df['X_gen'] + dz * tan_theta * np.cos(df['Phi_gen'])
            Y_proj = df['Y_gen'] + dz * tan_theta * np.sin(df['Phi_gen'])

            in_bounds = (X_proj.between(-150, 150)) & (Y_proj.between(-143.5, 143.5))

            df[f'X_gen_{i}'] = X_proj.where(in_bounds, np.nan)
            df[f'Y_gen_{i}'] = Y_proj.where(in_bounds, np.nan)

            crossing_array[in_bounds] = crossing_array[in_bounds] + str(i)

        df['crossing_type'] = crossing_array


    def generate_time_dependent_efficiencies(df):
      #     df['time_seconds'] = (df['time'] - BASE_TIME).dt.total_seconds()
          df['eff_theoretical_1'] = EFFS[0]
          df['eff_theoretical_2'] = EFFS[1]
          df['eff_theoretical_3'] = EFFS[2]
          df['eff_theoretical_4'] = EFFS[3]


    def simulate_measured_points(df: pd.DataFrame,
                                   y_widths,
                                   x_noise: float = 5.0,
                                   uniform_choice: bool = True) -> None:
          n = len(df)
          rng = np.random.default_rng()
          measured_type = np.full(n, '', dtype=object)

          for i in range(1, 5):
              # Plane‑specific geometry
              y_width = y_widths[0] if i in (1, 3) else y_widths[1]
              y_positions = np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2
              # NumPy views (no copy)
              eff   = df[f'eff_theoretical_{i}'].to_numpy(float, copy=False)
              x_gen = df[f'X_gen_{i}'].to_numpy(float, copy=False)
              y_gen = df[f'Y_gen_{i}'].to_numpy(float, copy=False)
              # Decide which tracks are detected
              pass_mask = rng.random(n) <= eff
              # ----- X coordinate -----
              x_mea = np.full(n, np.nan)
              x_mea[pass_mask] = x_gen[pass_mask] + rng.normal(0.0, x_noise, pass_mask.sum())
              df[f'X_mea_{i}'] = x_mea
              # ----- Y coordinate -----
              y_mea = np.full(n, np.nan)
              valid_y = pass_mask & ~np.isnan(y_gen)

              if valid_y.any():
                  idx_strip = np.argmin(np.abs(y_positions[:, None] - y_gen[valid_y]), axis=0)
                  strip_centres = y_positions[idx_strip]

                  if uniform_choice:
                      widths = y_width[idx_strip]
                      y_mea[valid_y] = rng.uniform(strip_centres - widths/2,
                                                   strip_centres + widths/2)
                  else:
                      y_mea[valid_y] = strip_centres

                  measured_type[valid_y] = measured_type[valid_y] + str(i)
              df[f'Y_mea_{i}'] = y_mea
          df['measured_type'] = measured_type

    def fill_measured_type(df):
          df['filled_type'] = df['measured_type']
          df['filled_type'] = df['filled_type'].replace({'124': '1234', '134': '1234'})

    def linear_fit(z, a, b):
          return a * z + b

    def fit_tracks(df, z_positions, z_plane):
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
          
          # Extrapolate x and y to z_plane
          df['X_fit'] = df['X_fit_1'] - (z_plane - z_positions[0]) * np.tan(df['Theta_fit']) * np.cos(df['Phi_fit'])
          df['Y_fit'] = df['Y_fit_1'] - (z_plane - z_positions[0]) * np.tan(df['Theta_fit']) * np.sin(df['Phi_fit'])
      
      
    # ----------------------------------------------------------------------------
    # Remaining part of the code (simulation loop and CSV saving) ----------------
    # ----------------------------------------------------------------------------
      
    # Create a dictionary to store DataFrames with two indices: AVG_CROSSING_EVS_PER_SEC and TIME_WINDOW
    results = {}

    for AVG_CROSSING_EVS_PER_SEC in AVG_CROSSING_EVS_PER_SEC_ARRAY:
        results[AVG_CROSSING_EVS_PER_SEC] = {}
              
        print("Calculating tracks...")

        columns = ['X_gen', 'Y_gen', 'Theta_gen', 'Phi_gen', 'Z_gen'] + \
                [f'X_gen_{i}' for i in range(1, 5)] + [f'Y_gen_{i}' for i in range(1, 5)] + \
                ['crossing_type', 'measured_type', 'fitted_type', 'time']
        df_generated = pd.DataFrame(np.nan, index=np.arange(N_TRACKS), columns=columns)

        rng = np.random.default_rng()
        
        df_generated = generate_valid_tracks(N_TRACKS, xlim, ylim, z_plane, BASE_TIME, cts, cos_n=2, batch_size=10_000)

        # generate_tracks_with_timestamps(df_generated, len(df_generated), xlim, ylim, z_plane, BASE_TIME, cts, cos_n=2)
        
        # generate_tracks_with_timestamps(df_generated, N_TRACKS, xlim, ylim, z_plane, BASE_TIME, cts, cos_n=2)
        # real_df = df_generated.copy()

        print("Tracks generated. Calculating intersections...")

        calculate_intersections(df_generated, Z_POSITIONS)
        crossing_df = df_generated.copy()  # keep the full set before filtering
        df = df_generated[df_generated['crossing_type'].isin(VALID_CROSSING_TYPES)].copy()
        
        print("Intersections calculated. Generating measured points...")

        generate_time_dependent_efficiencies(df)
        simulate_measured_points(df, Y_WIDTHS)

        print("Measured points generated. Filling measured type...")

        fill_measured_type(df)

        print("Measured type filled. Fitting tracks...")

        fit_tracks(df, Z_POSITIONS, z_plane)

        columns_to_keep = ['time'] + [col for col in df.columns if 'eff_' in col] + \
                        [col for col in df.columns if '_type' in col or 'Theta_' in col or 'Phi_' in col\
                            or 'X_gen' == col or 'Y_gen' == col or 'X_fit' == col or 'Y_fit' == col]
        
        # columns_to_keep = [col for col in df.columns if 'eff_' in col] + \
        # [col for col in df.columns if '_type' in col or 'Theta_' in col or 'Phi_' in col\
        #     or 'X_gen' == col or 'Y_gen' == col or 'X_fit' == col or 'Y_fit' == col]
        
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

# Print all column names
print("Columns in DataFrame:", df.columns.tolist())

from pathlib import Path

# Remove the rows in which 'crossing_type' is NaN, 1, 2, 3, or 4
df_test = df[df['crossing_type'].notna() & ~df['crossing_type'].isin(['1', '2', '3', '4', ''])]
df_test = df_test[df_test['measured_type'].notna() & ~df_test['measured_type'].isin(['1', '2', '3', '4', ''])]

total_time_seconds = (df_test['time'].max() - df_test['time'].min()).total_seconds()

# Calculate the total rate of events
total_events = len(df_test)
total_generated_rate = total_events / total_time_seconds if total_time_seconds > 0 else 0

# Calculate the rate of events that cross the detector
crossing_events = df_test['crossing_type'].notna().sum()
total_crossing_rate = crossing_events / total_time_seconds / total_generated_rate if total_time_seconds > 0 else 0

# Calculate the rate for each measured type
measured_types = df_test['measured_type'].dropna().unique()
measured_rates = {}
for measured_type in measured_types:
    measured_events = df[df['measured_type'] == measured_type].shape[0]
    measured_rates[measured_type] = measured_events / total_time_seconds / total_generated_rate if total_time_seconds > 0 else 0

# Fixed path to output file
rates_LUT_filename = Path(f"{home_path}/DATAFLOW_v3/TESTS/SIMULATION/rates_LUT.csv")
write_header = not rates_LUT_filename.exists()

with open(rates_LUT_filename, 'a') as f:
    if write_header:
        header = ["FLUX", "total_generated_rate", "total_crossing_rate"] + list(measured_types)
        f.write(",".join(header) + "\n")

    # Replace `flux_value` with the desired definition
    flux_value = FLUX  # or total_crossing_rate or another variable
    values = [flux_value, total_generated_rate, total_crossing_rate]
    values += [measured_rates.get(mt, 0) for mt in measured_types]
    f.write(",".join(f"{v:.3f}" for v in values) + "\n")


if cut_soon:
    # Exit the script if cut_soon is True
    print("cut_soon is True, exiting the script.")
    sys.exit(0)


#%%

print("Unique crossing_type values:", df['crossing_type'].dropna().unique())
print("Unique measured_type values:", df['measured_type'].dropna().unique())
print("Unique fitted_type values:", df['fitted_type'].dropna().unique())

# Define binning
theta_bins = np.linspace(0, np.pi / 2, 200)
phi_bins = np.linspace(-np.pi, np.pi, 200)
tt_lists = [ VALID_MEASURED_TYPES ]

for tt_list in tt_lists:
      
      # Create figure with 2 rows and 4 columns
      fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex='row')
      
      # First column: Generated angles
      # if real_df exists:
      if 'real_df' in globals():
                # axes[0, 0].hist(real_df['Theta_gen'], bins=theta_bins, histtype='step', label='All', color='red', density = True)
                # axes[1, 0].hist(real_df['Phi_gen'], bins=phi_bins, histtype='step', label='All', color='red', density = True)
                axes[0, 0].hist(df['Theta_gen'], bins=theta_bins, histtype='step', label='All', color='black', density = True)
                axes[1, 0].hist(df['Phi_gen'], bins=phi_bins, histtype='step', label='All', color='black', density = True)
      else:
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
theta_bins = np.linspace(0, np.pi / 2, 100)
phi_bins = np.linspace(-np.pi, np.pi, 100)

tt_lists = [ VALID_MEASURED_TYPES ]

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

size_of_point = 0.01
alpha_of_point = 0.02

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

# Define topologies to evaluate
tt_list = ['1234', '123', '234', '12', '23', '34']  # or VALID_MEASURED_TYPES

# Create figure
n_tt = len(tt_list)
fig, axes = plt.subplots(n_tt, 2, figsize=(7, 3 * n_tt), sharex=False, sharey=False)

size_of_point = 0.05
alpha_of_point = 0.01

distance_limit = 6000

for i, tt in enumerate(tt_list):
      sel = df['measured_type'] == tt

      # Theta: Generated vs Fitted
      theta_gen = df.loc[sel, 'X_gen']
      theta_fit = df.loc[sel, 'X_fit']
      ax_theta = axes[i, 0]
      ax_theta.scatter(theta_gen, theta_fit, s=size_of_point, alpha=alpha_of_point)
      ax_theta.plot([-distance_limit, distance_limit], [-distance_limit, distance_limit], 'k--', lw=1)
      ax_theta.set_xlabel(r'$X$ [mm]')
      ax_theta.set_ylabel(r'$\theta_{\mathrm{fit}}$ [rad]')
      ax_theta.set_title(f'TT={tt}: $\theta$ gen vs fit')
      ax_theta.set_xlim(-distance_limit, distance_limit)
      ax_theta.set_ylim(-distance_limit, distance_limit)
      ax_theta.grid(True)
      ax_theta.set_aspect('equal', adjustable='box')

      # Phi: Generated vs Fitted
      phi_gen = df.loc[sel, 'Y_gen']
      phi_fit = df.loc[sel, 'Y_fit']
      ax_phi = axes[i, 1]
      ax_phi.scatter(phi_gen, phi_fit, s=size_of_point, alpha=alpha_of_point)
      ax_phi.plot([-distance_limit, distance_limit], [-distance_limit, distance_limit], 'k--', lw=1)
      ax_phi.set_xlabel(r'$\phi_{\mathrm{gen}}$ [rad]')
      ax_phi.set_ylabel(r'$\phi_{\mathrm{fit}}$ [rad]')
      ax_phi.set_title(f'TT={tt}: $\phi$ gen vs fit')
      ax_phi.set_xlim(-distance_limit, distance_limit)
      ax_phi.set_ylim(-distance_limit, distance_limit)
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

x_bins = np.linspace(-1, 1, 100)
y_bins   = np.linspace(-1, 1, 100)

tt_list = ['1234', '123', '234', '12', '23', '34']  # or VALID_MEASURED_TYPES

PLOT_DIR = f"{home_path}/DATAFLOW_v3/TESTS/SIMULATION"

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
            case = 'fit'  # or 'fit', 'gen'
            
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


# ----------------------------------------------------------------------
# 2-D histograms (θ, φ) comparison
# ----------------------------------------------------------------------

import os

x_bins = np.linspace(-distance_limit, distance_limit, 200)
y_bins   = np.linspace(-distance_limit, distance_limit, 200)

tt_list = ['1234', '123', '234', '12', '23', '34']  # or VALID_MEASURED_TYPES

PLOT_DIR = f"{home_path}/DATAFLOW_v3/TESTS/SIMULATION"

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
            
            x_center = 100
            y_center = 80
            radius = 30
            case = 'fit'  # or 'fit', 'gen'
            
            theta_mask = (df_plot[f"X_{case}"] > x_center - radius) & \
                         (df_plot[f"X_{case}"] < x_center + radius)
            phi_mask = (df_plot[f"Y_{case}"] > y_center - radius) & \
                       (df_plot[f"Y_{case}"] < y_center + radius)
            df_plot = df_plot[theta_mask & phi_mask].copy()

            # Measured (gen)
            x = df_plot["X_gen"]
            y = df_plot["Y_gen"]
            ax[i,0].hist2d(x, y,
                           bins=[x_bins, y_bins], cmap="viridis")
            # ax[i,0].hist2d(df_plot["Theta_gen"], df_plot["Phi_gen"],
            #             bins=[theta_bins, phi_bins], cmap="viridis")
            ax[i,0].set_title("meas (gen)")

            # Measured (fit)
            x = df_plot["X_fit"]
            y = df_plot["Y_fit"]
            ax[i,1].hist2d(x, y,
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

      for a in ax[:,0]: a.set_ylabel(r"$Y$ [mm]")
      for a in ax[-1,:]: a.set_xlabel(r"$X$ [mm]")
      fig.tight_layout()
      plt.savefig(f"{PLOT_DIR}/hist2d_XY_{'_'.join(tt_group)}.png", dpi=150)
      plt.show()
      plt.close()

#%%


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# METHOD OF LIKELYHOOD ------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


# # ---------------------------------------------------------------------
# # Coordinate transform: (u, v) = (sin θ sin φ, sin θ cos φ)
# # ---------------------------------------------------------------------
# df["u_fit"] = np.sin(df["Theta_fit"]) * np.sin(df["Phi_fit"])
# df["v_fit"] = np.sin(df["Theta_fit"]) * np.cos(df["Phi_fit"])

# df["u_gen"] = np.sin(df["Theta_gen"]) * np.sin(df["Phi_gen"])
# df["v_gen"] = np.sin(df["Theta_gen"]) * np.cos(df["Phi_gen"])

# # ---------------------------------------------------------------------
# # Regular (u, v) binning
# # ---------------------------------------------------------------------
# n_bins  = 3
# u_edges = np.linspace(-1.0, 1.0, n_bins + 1)
# v_edges = np.linspace(-1.0, 1.0, n_bins + 1)

# u_idx = np.digitize(df["u_fit"], u_edges) - 1
# v_idx = np.digitize(df["v_fit"], v_edges) - 1
# df["bin"] = list(zip(u_idx, v_idx))               # tuple → hashable

# # ---------------------------------------------------------------------
# # Empirical conditional distributions
# #   key   : (measured_type, bin)
# #   value : ndarray (N×2) with (θ_gen, φ_gen)
# # ---------------------------------------------------------------------
# mapping = {
#     key: g[["Theta_gen", "Phi_gen"]].to_numpy()
#     for key, g in df.groupby(["measured_type", "bin"])
# }

# # ---------------------------------------------------------------------
# # Draw a prediction for each event
# # ---------------------------------------------------------------------
# def draw_pred(row):
#     candidates = mapping.get((row["measured_type"], row["bin"]))
#     if candidates is None or len(candidates) == 0:
#         # empty cell → keep the fitted angles
#         return pd.Series(
#             {"Theta_pred": row["Theta_fit"], "Phi_pred": row["Phi_fit"]}
#         )

#     th, ph = candidates[np.random.randint(len(candidates))]
#     return pd.Series({"Theta_pred": th, "Phi_pred": ph})

# tqdm.pandas()
# df_pred = df.join(df.progress_apply(draw_pred, axis=1))


#%%


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# METHOD OF LIKELYHOOD ------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

only_angles_input = True
if only_angles_input:
    # ---------------------------------------------------------------------
    # Coordinate transform: (u, v) = (sin θ sin φ, sin θ cos φ)
    # ---------------------------------------------------------------------
    df["u_fit"] = np.sin(df["Theta_fit"]) * np.sin(df["Phi_fit"])
    df["v_fit"] = np.sin(df["Theta_fit"]) * np.cos(df["Phi_fit"])

    df["u_gen"] = np.sin(df["Theta_gen"]) * np.sin(df["Phi_gen"])
    df["v_gen"] = np.sin(df["Theta_gen"]) * np.cos(df["Phi_gen"])

    # ---------------------------------------------------------------
    # (u, v) binning – keep n_bins as defined earlier
    # ---------------------------------------------------------------
    n_bins  = 20
    u_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    v_edges = np.linspace(-1.0, 1.0, n_bins + 1)

    df["fit_u_idx"] = np.digitize(df["u_fit"], u_edges) - 1
    df["fit_v_idx"] = np.digitize(df["v_fit"], v_edges) - 1
    df["gen_u_idx"] = np.digitize(df["u_gen"], u_edges) - 1
    df["gen_v_idx"] = np.digitize(df["v_gen"], v_edges) - 1

    # restrict to valid bins
    inside = (
        df["fit_u_idx"].between(0, n_bins-1) &
        df["fit_v_idx"].between(0, n_bins-1) &
        df["gen_u_idx"].between(0, n_bins-1) &
        df["gen_v_idx"].between(0, n_bins-1)
    )
    df = df[inside]
    
    print("Filtered df")
    
    # ---------------------------------------------------------------
    # Build matrices per measured_type
    # ---------------------------------------------------------------
    # helper: flatten (u_idx, v_idx) to a single ordinal 0 … n_bins²-1
    flat = lambda u, v: u * n_bins + v

    # matrices = {}  # {type: DataFrame}

    # for ttype, g in df.groupby("measured_type"):
    #     size = n_bins ** 2
    #     M = np.zeros((size, size), dtype=float)  # rows = GEN, cols = FIT

    #     # count events
    #     for _, row in g.iterrows():
    #         col = flat(row["fit_u_idx"], row["fit_v_idx"])
    #         row_ = flat(row["gen_u_idx"], row["gen_v_idx"])
    #         M[row_, col] += 1.0

    #     # normalise each column (FIT bin) to unity if non-empty
    #     col_sums = M.sum(axis=0)
    #     non_zero = col_sums > 0
    #     M[:, non_zero] /= col_sums[non_zero]

    #     # build labelled DataFrame for clarity and CSV output
    #     labels = [f"({u},{v})" for u in range(n_bins) for v in range(n_bins)]
    #     df_M = pd.DataFrame(M, index=labels, columns=labels)
    #     matrices[ttype] = df_M

    #     # write to CSV: likelihood_matrix_<type>.csv
    #     file_name = f"likelihood_matrix_{ttype}.csv"
    #     df_M.to_csv(file_name)
    #     print(f"{file_name} written ({size}×{size})")
    
    labels = [f"({u},{v})" for u in range(n_bins) for v in range(n_bins)]

    # open a single HDF5 file
    hdf_path = "likelihood_matrices.h5"
    with pd.HDFStore(hdf_path, mode='w') as store:
        for ttype, g in df.groupby("measured_type"):
            size = n_bins ** 2
            M = np.zeros((size, size), dtype=float)

            for _, row in g.iterrows():
                col = flat(row["fit_u_idx"], row["fit_v_idx"])
                row_ = flat(row["gen_u_idx"], row["gen_v_idx"])
                M[row_, col] += 1.0

            # normalize columns
            col_sums = M.sum(axis=0)
            non_zero = col_sums > 0
            M[:, non_zero] /= col_sums[non_zero]

            df_M = pd.DataFrame(M, index=labels, columns=labels)

            # store under key for that measured type
            store.put(f"{ttype}", df_M, format="fixed")
            print(f"{ttype} written to {hdf_path} ({size}×{size})")
    
    matrices = {}
    hdf_path = "likelihood_matrices.h5"

    with pd.HDFStore(hdf_path, mode='r') as store:
        for key in store.keys():  # keys are like '/P1', '/P2', etc.
            ttype = key.strip("/")  # remove leading slash
            df_M = store.get(key)
            matrices[ttype] = df_M
            print(f"Loaded matrix for: {ttype}, shape = {df_M.shape}")
    
    # matrices["1234"]  # access the DataFrame in memory if needed


    # ------------------------------------------------------------
    # Helper to flatten / un-flatten bin indices
    # ------------------------------------------------------------
    def flat(u_idx, v_idx, n_bins):
        return u_idx * n_bins + v_idx

    


    def load_lut(path_prefix: Path, t_type: str,
                n_bins: int, prefer_csv: bool = True) -> np.ndarray:
        """
        Carga la LUT de disco (CSV o binario) solo una vez por tipo.
        """
        if prefer_csv:
            file = path_prefix / f"likelihood_matrix_{t_type}.csv"
            return pd.read_csv(file, index_col=0).to_numpy()
        else:
            file = path_prefix / f"likelihood_matrix_{t_type}.bin"
            return np.fromfile(file, dtype=np.float64).reshape(n_bins ** 2, n_bins ** 2)


    def wrap_to_pi(angle: float) -> float:
        """Devuelve el ángulo en (-π, π]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


    def sample_true_angles_nearest(
        df_fit: pd.DataFrame,
        matrices: Optional[Dict[str, pd.DataFrame]],
        u_edges: np.ndarray,
        v_edges: np.ndarray,
        n_bins: int,
        rng: Optional[np.random.Generator] = None,
        lut_dir: Optional[str] = None,
        prefer_csv: bool = True,
        show_progress: bool = True,
        print_every: int = 50_000
    ) -> pd.DataFrame:
        """
        Same interface as `sample_true_angles`, but the LUT is queried in nearest
        neighbour mode (no bilinear weights).  The LUT is assumed to be dense
        enough that interpolation is not worth the extra CPU cost.
        """
        if rng is None:
            rng = np.random.default_rng()

        # --------------------------------------------------------- LUT cache
        matrix_cache: Dict[str, np.ndarray] = {}
        if matrices is not None:
            for t, df_m in matrices.items():
                matrix_cache[t] = df_m.to_numpy()
        else:
            if lut_dir is None:
                raise ValueError("Provide either `matrices` or `lut_dir`.")
            lut_path = Path(lut_dir).expanduser()

        # --------------------------------------------------------- indices in the 2-D grid
        u_fit = np.sin(df_fit["Theta_fit"].values) * np.sin(df_fit["Phi_fit"].values)
        v_fit = np.sin(df_fit["Theta_fit"].values) * np.cos(df_fit["Phi_fit"].values)

        iu = np.clip(np.digitize(u_fit, u_edges) - 1, 0, n_bins - 2)  # lower edge
        iv = np.clip(np.digitize(v_fit, v_edges) - 1, 0, n_bins - 2)

        # decide whether the upper or lower neighbour is closer (nearest-neighbour)
        iu += (u_fit - u_edges[iu]) > (u_edges[iu + 1] - u_fit)
        iv += (v_fit - v_edges[iv]) > (v_edges[iv + 1] - v_fit)

        # flatten / unflatten helpers
        flat   = lambda u, v: u * n_bins + v
        unflat = lambda k: divmod(k, n_bins)

        # output buffers
        N = len(df_fit)
        theta_pred = np.empty(N, dtype=np.float32)
        phi_pred   = np.empty(N, dtype=np.float32)

        iterator = range(N)
        if show_progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Sampling true angles (nearest-bin)", unit="evt")

        # --------------------------------------------------------- main loop
        for n in iterator:
            if tqdm is None and show_progress and (n % print_every == 0):
                print(f"[{n:>10}/{N}]")

            t_type = df_fit["measured_type"].iat[n]

            # fetch LUT (load once per type)
            if t_type not in matrix_cache:
                matrix_cache[t_type] = load_lut(lut_path, t_type, n_bins, prefer_csv)
            M = matrix_cache[t_type]

            # nearest bin in the 2-D grid
            col_idx = flat(iu[n], iv[n])
            p = M[:, col_idx]

            s = p.sum()
            if s == 0:                                     # empty column → uniform
                p = np.full_like(p, 1.0 / len(p))
            else:
                p = p / s

            gen_idx = rng.choice(len(p), p=p)
            g_u_idx, g_v_idx = unflat(gen_idx)

            # sample uniformly inside that generated-angle cell
            u_pred = rng.uniform(u_edges[g_u_idx], u_edges[g_u_idx + 1])
            v_pred = rng.uniform(v_edges[g_v_idx], v_edges[g_v_idx + 1])

            sin_theta = min(np.hypot(u_pred, v_pred), 1.0)
            theta_pred[n] = math.asin(sin_theta)
            phi_pred[n]   = wrap_to_pi(math.atan2(u_pred, v_pred))

        # --------------------------------------------------------- return
        df_out = df_fit.copy()
        df_out["Theta_pred"] = theta_pred
        df_out["Phi_pred"]   = phi_pred
        return df_out
    
    #%%
    
    # Take only 100000 events
    # df_input = df.sample(100_000, random_state=2024).reset_index(drop=True)
    df_input = df

    df_pred = sample_true_angles_nearest(
        df_fit=df_input,
        matrices=None,                        # o `matrices` si ya las tienes en RAM
        lut_dir=f"{home_path}/DATAFLOW_v3/TESTS/SIMULATION",
        n_bins=n_bins,
        u_edges=u_edges,
        v_edges=v_edges,
        rng=np.random.default_rng(2025),
        show_progress=True
    )
    
    distance_limit_x   = 3000.0  # 500                  # e.g. mm – adjust to your system
    distance_limit_y   = 3000.0
    
    df_pred["X_pred"] = df['X_fit']
    df_pred["Y_pred"] = df['Y_fit']
    
    #%%
    
    
    
    def plot_likelihood_matrix(matrices, measured_type, bin_uv,
                            u_edges, v_edges, z_log=False):
        """
        Visualise P(u_gen, v_gen | given FIT bin) stored in the likelihood matrix.
        -------------------------------------------------------------------------
        matrices       : dict[type] -> DataFrame (rows = GEN bins, cols = FIT bins)
        measured_type  : e.g. "1234"
        bin_uv         : tuple  (u_idx_fit, v_idx_fit)
        u_edges, v_edges : 1-D bin edges used to build the matrix
        z_log          : if True, plot log10(prob)  (useful for wide dynamic range)
        """

        df_M = matrices[measured_type]          # DataFrame  (N² × N²)
        n_bins = len(u_edges) - 1
        col_idx = flat(*bin_uv, n_bins)         # column corresponding to the FIT bin

        if col_idx >= df_M.shape[1]:
            raise ValueError("bin_uv out of range")

        # Probability vector → 2-D probability grid
        P_vec = df_M.iloc[:, col_idx].to_numpy()       # length N²
        P_grid = P_vec.reshape(n_bins, n_bins)         # shape (u_gen_idx , v_gen_idx)

        # Bin centres for u_gen, v_gen
        u_centres = 0.5 * (u_edges[:-1] + u_edges[1:])
        v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])
        Uc, Vc    = np.meshgrid(u_centres, v_centres, indexing="ij")

        # Optional log-10 transform for Z-axis
        Z = np.log10(P_grid, where=P_grid > 0) if z_log else P_grid

        # --------------------------------------------------------
        # 3-D surface plot
        # --------------------------------------------------------
        fig = plt.figure(figsize=(8, 5.5))
        ax  = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(Uc, Vc, Z,
                            cmap=cm.viridis,
                            edgecolor='none',
                            rstride=1, cstride=1,
                            antialiased=True)

        ax.set_xlabel(r'$u_{\rm gen}$')
        ax.set_ylabel(r'$v_{\rm gen}$')
        ax.set_zlabel(r'$\log_{10}P$' if z_log else r'$P$')
        ax.set_title(f'P(u_gen, v_gen | FIT bin {bin_uv}, type={measured_type})')
        fig.colorbar(surf, ax=ax, shrink=0.65,
                    label=r'$\log_{10}P$' if z_log else r'$P$')
        
        # Remove grid lines for clarity
        ax.xaxis._axinfo['grid'].update(color='none')
        ax.yaxis._axinfo['grid'].update(color='none')
        ax.zaxis._axinfo['grid'].update(color='none')
        ax.set_box_aspect([1, 1, 0.3])  # aspect ratio for better visualisation
        
        plt.tight_layout()
        plt.show()
    
    
    print("Plotting...")
    
    # ------------------------------------------------------------------
    # Example usage  (assuming matrices, u_edges, v_edges already exist)
    # ------------------------------------------------------------------
    plot_likelihood_matrix(
        matrices,
        measured_type="1234",
        bin_uv=(20, 12),           # FIT-bin of interest
        u_edges=u_edges,
        v_edges=v_edges,
        z_log=False              # or True for log-scale Z
    )
    
    
else:

    og_df = df.copy()  # Original DataFrame for reference
    # Add _pred columns full of zeroes to og_df

    og_df["Theta_pred"] = 0.0
    og_df["Phi_pred"] = 0.0
    og_df["X_pred"] = 0.0
    og_df["y_pred"] = 0.0

    VALID_MEASURED_TYPES = ['1234',
    '123', '234', '134', '124',
    '12', '23', '34', '13', '24', '14']
    og_df = og_df[og_df["measured_type"].isin(VALID_MEASURED_TYPES)].reset_index(drop=True)

    df = og_df.copy()

    import math
    from typing import Dict, Optional
    from pathlib import Path

    # Drop the columns of df in which measured_type is not in VALID_MEASURED_TYPE
    df = df[df["measured_type"].isin(VALID_MEASURED_TYPES)].reset_index(drop=True)


    # Histogram in a 2x2 the Theta_gen, Phi_gen, X_gen, Y_gen

    # Create the 2x2 histogram figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot each histogram
    axs[0, 0].hist(df["Theta_gen"], bins=50)
    axs[0, 0].set_title("Theta_gen")
    axs[0, 0].set_xlabel("Theta (rad)")
    axs[0, 0].set_ylabel("Counts")

    axs[0, 1].hist(df["Phi_gen"], bins=50)
    axs[0, 1].set_title("Phi_gen")
    axs[0, 1].set_xlabel("Phi (rad)")
    axs[0, 1].set_ylabel("Counts")

    axs[1, 0].hist(df["X_gen"], bins=50)
    axs[1, 0].set_title("X_gen")
    axs[1, 0].set_xlabel("X (mm)")
    axs[1, 0].set_ylabel("Counts")

    axs[1, 1].hist(df["Y_gen"], bins=50)
    axs[1, 1].set_title("Y_gen")
    axs[1, 1].set_xlabel("Y (mm)")
    axs[1, 1].set_ylabel("Counts")

    # Layout adjustment
    plt.tight_layout()
    plt.show()

    print("-----------------------------------------------------")
    print("Creating LUTs...")
    print("-----------------------------------------------------")

    # ---------------------------------------------------------------------
    # Bin edges
    # ---------------------------------------------------------------------
    distance_limit_x   = 3000.0  # 500                  # e.g. mm – adjust to your system
    distance_limit_y   = 3000.0  # 1000                 # e.g. mm – adjust to your system

    # choose independently
    n_uv   = 200                 # bins in u  and v   (–1 … 1)
    n_xy   = 200                 # bins in X  and Y   (–d … +d)

    edges_uv = np.linspace(-0.9, 0.9, n_uv + 1)
    edges_x = np.linspace(-distance_limit_x, distance_limit_x, n_xy + 1)
    edges_y = np.linspace(-distance_limit_y, distance_limit_y, n_xy + 1)

    # def equal_count_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    #     """
    #     Returns edges so that each of the n_bins has the same number of events.
    #     The outermost edges are ± max(|x|) to keep full range.
    #     """
    #     q      = np.linspace(0, 1, n_bins + 1)
    #     edges  = np.quantile(x, q)
    #     edges[0]  = -np.abs(x).max()
    #     edges[-1] =  np.abs(x).max()
    #     return edges


    # x_data     = df["X_fit"].to_numpy()
    # edges_x    = equal_count_edges(x_data, n_bins= n_xy + 1 )   # ← choose any bin count

    # y_data     = df["Y_fit"].to_numpy()
    # edges_y    = equal_count_edges(y_data, n_bins= n_xy + 1 )   # ← choose any bin count

    # centres_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    # widths_x  = np.diff(edges_x)

    # centres_y = 0.5 * (edges_y[:-1] + edges_y[1:])
    # widths_y  = np.diff(edges_y)

    # # ------------------------------------------------------------------
    # # Line plot – bin width vs. position (X axis)
    # # ------------------------------------------------------------------
    # plt.figure(figsize=(6, 3))
    # plt.plot(centres_x, widths_x)
    # plt.plot(centres_y, widths_y)
    # plt.xlabel('x, y position (bin centre)')
    # plt.ylabel('bin width')
    # plt.title('Bin width along position')
    # plt.tight_layout()
    # plt.show()
    

    # ------------------------------------------------------------------
    # 4-D <--> 1-D ordinal  (lexicographic)
    # ------------------------------------------------------------------
    # ─────────────────────────── helpers ────────────────────────────
    def flat4(iu, iv, ix, iy, n_uv, n_xy):
        return ((iu * n_uv + iv) * n_xy + ix) * n_xy + iy

    def unflat4(k, n_uv, n_xy):
        iu, rem = divmod(k, n_uv * n_xy * n_xy)
        iv, rem = divmod(rem,       n_xy * n_xy)
        ix, iy  = divmod(rem,       n_xy)
        return iu, iv, ix, iy

    def wrap_to_pi(angle: float) -> float:
            """Devuelve el ángulo en (-π, π]."""
            return (angle + math.pi) % (2.0 * math.pi) - math.pi


    # ---------------------------------------------------------------------
    # Coordinate definitions  (unchanged)
    # ---------------------------------------------------------------------
    df["u_fit"] = np.sin(df["Theta_fit"]) * np.sin(df["Phi_fit"])
    df["v_fit"] = np.sin(df["Theta_fit"]) * np.cos(df["Phi_fit"])
    df["u_gen"] = np.sin(df["Theta_gen"]) * np.sin(df["Phi_gen"])
    df["v_gen"] = np.sin(df["Theta_gen"]) * np.cos(df["Phi_gen"])

    # X_fit, Y_fit, X_gen, Y_gen are already present in the DataFrame

    # ------------------------------------------------------------
    # Generate the *_idx columns
    # ------------------------------------------------------------
    coord_names = ["u_fit", "v_fit", "X_fit", "Y_fit",
                "u_gen", "v_gen", "X_gen", "Y_gen"]

    # ------------------------------------------------------------------
    # Create *_idx columns  (u,v  use n_uv bins;   X,Y  use n_xy bins)
    # ------------------------------------------------------------------
    for c in ("u_fit", "v_fit", "u_gen", "v_gen"):
        df[f"{c}_idx"] = np.digitize(df[c], edges_uv) - 1
        print(f"{c:8s}:  min={df[c].min():8.3f}  max={df[c].max():8.3f}")

    for c in ("X_fit", "X_gen"):
        df[f"{c}_idx"] = np.digitize(df[c], edges_x) - 1
        print(f"{c:8s}:  min={df[c].min():8.3f}  max={df[c].max():8.3f}")

    for c in ("Y_fit", "Y_gen"):
        df[f"{c}_idx"] = np.digitize(df[c], edges_y) - 1
        print(f"{c:8s}:  min={df[c].min():8.3f}  max={df[c].max():8.3f}")

    # ------------------------------------------------------------------
    # Keep only events fully inside the 4-D grid
    # ------------------------------------------------------------------
    inside_uv = np.logical_and.reduce([
        df[f"{c}_idx"].between(0, n_uv - 1)          # u, v limits
        for c in ("u_fit", "v_fit", "u_gen", "v_gen")
    ])

    inside_xy = np.logical_and.reduce([
        df[f"{c}_idx"].between(0, n_xy - 1)          # X, Y limits
        for c in ("X_fit", "Y_fit", "X_gen", "Y_gen")
    ])

    og_og_df = df.copy()  # Original DataFrame for reference
    df = df[inside_uv & inside_xy].reset_index(drop=True)


    from scipy.sparse import coo_matrix, csr_matrix, save_npz
    import numpy as np

    from pathlib import Path
    from typing import Union           # ← import Union
    from scipy.sparse import coo_matrix, csr_matrix, save_npz
    import numpy as np

    from pathlib import Path
    from typing  import Union
    from scipy.sparse import coo_matrix, csc_matrix, save_npz
    import numpy as np
    

    import numpy as np
    import pandas as pd
    from pathlib import Path
    from typing import Union
    from scipy.sparse import coo_matrix, save_npz
    import inspect

    # ────────────────────────────────────────────────────────────────
    def build_sparse_lut_prob(
        df_type:  pd.DataFrame,
        n_uv:     int,
        n_xy:     int,
        t_type:   str,
        out_dir:  Union[str, Path],
        ndigits:  int = 3,
    ) -> None:
        """
        Construct a GEN×FIT LUT whose columns are probability vectors,
        rounded to *ndigits* decimals; entries that become exactly zero are dropped.
        Compatible with SciPy versions both before and after index_dtype support.
        """
        size = n_uv * n_uv * n_xy * n_xy

        # ---- vectorised flat indices ---------------------------------
        ui_f = df_type["u_fit_idx"].to_numpy(np.int32)
        vi_f = df_type["v_fit_idx"].to_numpy(np.int32)
        xi_f = df_type["X_fit_idx"].to_numpy(np.int32)
        yi_f = df_type["Y_fit_idx"].to_numpy(np.int32)
        col  = ((ui_f * n_uv + vi_f) * n_xy + xi_f) * n_xy + yi_f

        ui_g = df_type["u_gen_idx"].to_numpy(np.int32)
        vi_g = df_type["v_gen_idx"].to_numpy(np.int32)
        xi_g = df_type["X_gen_idx"].to_numpy(np.int32)
        yi_g = df_type["Y_gen_idx"].to_numpy(np.int32)
        row  = ((ui_g * n_uv + vi_g) * n_xy + xi_g) * n_xy + yi_g

        data = np.ones_like(row, dtype=np.int32)

        # ---- create COO  →  CSC  -------------------------------------
        kwargs = dict(dtype=np.float32)
        if "index_dtype" in inspect.signature(coo_matrix).parameters:
            kwargs["index_dtype"] = np.int32

        M = coo_matrix((data, (row, col)), shape=(size, size), **kwargs).tocsc(copy=False)

        # ---- column normalisation ------------------------------------
        col_sums = np.asarray(M.sum(axis=0)).ravel()
        nz_cols  = np.flatnonzero(col_sums)
        for j in nz_cols:
            start, end = M.indptr[j], M.indptr[j + 1]
            M.data[start:end] /= col_sums[j]

        # ---- round, threshold, zero-prune -------------------------------
        M.data = np.round(M.data, ndigits).astype(np.float32)

        M.data[M.data < 1 / 10 ** ndigits] = 0.0          # ← discard tiny probabilities
        M.eliminate_zeros()                   # remove the entries that became 0.0
        M.sort_indices()

        # ---- persist -------------------------------------------------
        out_path = Path(out_dir).expanduser() / f"likelihood_matrix_{t_type}.npz"
        save_npz(out_path, M, compressed=True)
        print(f"{out_path} written  |  nnz = {M.nnz:,}")


    # ────────────────────────────────────────────────────────────────
    # Example driver
    # ────────────────────────────────────────────────────────────────
    def build_all_luts(df: pd.DataFrame,
                    valid_types: list[str],
                    n_uv: int, n_xy: int,
                    out_dir: Union[str, Path],
                    ndigits: int = 3) -> None:
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        df = df[df["measured_type"].isin(valid_types)]

        for t_type, grp in df.groupby("measured_type", sort=True):
            build_sparse_lut_prob(
                df_type = grp,
                n_uv    = n_uv,
                n_xy    = n_xy,
                t_type  = str(t_type),
                out_dir = out_dir,
                ndigits = ndigits,
            )


    build_all_luts(
        df,                       # the DataFrame that already holds your events
        VALID_MEASURED_TYPES,     # list defined earlier
        n_uv,                     # number of u / v bins
        n_xy,                     # number of X / Y bins
        out_dir="~/DATAFLOW_v3/TESTS/SIMULATION",  # target directory
        ndigits=6                 # keep three decimal places (adjust as needed)
    )


    print("-----------------------------------------------------")
    print("LUTs created and saved.")
    print("-----------------------------------------------------")

    # ─────────────────────────────────────────────────────────────────────
    # Fast nearest-bin sampler  –  compatible with rounded sparse LUTs
    # ─────────────────────────────────────────────────────────────────────

    from pathlib import Path
    from functools import lru_cache
    from typing   import Union
    import math, numpy as np, pandas as pd
    from scipy.sparse import csc_matrix, load_npz

    try:
        from tqdm.auto import tqdm               # nice progress bar if available
    except ModuleNotFoundError:
        tqdm = None

    # ---------- helpers -------------------------------------------------
    def flat4(iu: int, iv: int, ix: int, iy: int,
            n_uv: int, n_xy: int) -> int:
        """(iu,iv,ix,iy) → ordinal in 0 … n_uv²·n_xy²-1"""
        return ((iu * n_uv + iv) * n_xy + ix) * n_xy + iy

    def unflat4(k: int, n_uv: int, n_xy: int) -> tuple[int, int, int, int]:
        iu, rem = divmod(k, n_uv * n_xy * n_xy)
        iv, rem = divmod(rem,       n_xy * n_xy)
        ix, iy  = divmod(rem,       n_xy)
        return iu, iv, ix, iy


    from pathlib import Path
    from functools import lru_cache
    from typing   import Dict, Tuple, Union
    import numpy as np
    from scipy.sparse import csc_matrix, load_npz

    # ---------- matrix cache keyed by measured_type --------------------
    _matrix_cache: Dict[str, csc_matrix] = {}   # Python ≤3.8 compatible

    def load_sparse_lut(path_prefix: Path, t_type: str) -> csc_matrix:
        """Return detector LUT (CSC) for one measured_type."""
        return load_npz(path_prefix / f"likelihood_matrix_{t_type}.npz")

    # ---------- column extractor with bounded (type,col) cache ----------
    # put near the top once
    # ─── constants ──────────────────────────────────────────────────────
    N_CELLS = n_uv * n_uv * n_xy * n_xy          # 2_560_000 for 40×40 bins

    # ─── cached column lookup (unchanged, with sentinel) ───────────────
    @lru_cache(maxsize=8_192)
    def _rows_cdf_for(
        t_type:   str,
        col:      int,
        lut_base: Path,
        renorm:   bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        if t_type not in _matrix_cache:
            _matrix_cache[t_type] = load_npz(lut_base / f"likelihood_matrix_{t_type}.npz")
        M = _matrix_cache[t_type]

        start, end  = M.indptr[col], M.indptr[col + 1]
        rows, probs = M.indices[start:end], M.data[start:end]

        # sentinel for empty column  → (None, None)
        if rows.size == 0:
            return None, None

        if renorm:
            probs = probs / probs.sum(dtype=np.float32)

        cdf = np.cumsum(probs, dtype=np.float32)
        
        # last = cdf[-1]
        # if last < 1.0:                           # happens after rounding
        #     cdf /= last                          # renormalise in-place

        return rows.astype(np.int32, copy=False), cdf
    
    
    # ---------- main sampler --------------------------------------------
    def sample_true_state_nearest(
        df_fit:       pd.DataFrame,
        n_uv:         int,
        n_xy:         int,
        edges_uv:     np.ndarray,
        edges_x:      np.ndarray,
        edges_y:      np.ndarray,
        rng:          np.random.Generator,
        lut_dir:      Union[str, Path],
        show_progress: bool = True,
        print_every:   int  = 50_000,
        renorm:        bool = True,                # re-normalise column if rounding ≠1
    ) -> pd.DataFrame:
        """
        Draw (u,v,X,Y) true state from nearest-bin 4-D LUTs.
        LUTs must be those created by `build_sparse_lut_prob`.
        """

        lut_base = Path(lut_dir).expanduser()
        uniform_setting = True

        # --- nearest FIT indices ---------------------------------------
        u_fit, v_fit = df_fit["u_fit"].to_numpy(np.float32), df_fit["v_fit"].to_numpy(np.float32)
        X_fit, Y_fit = df_fit["X_fit"].to_numpy(np.float32), df_fit["Y_fit"].to_numpy(np.float32)

        iu = np.clip(np.digitize(u_fit, edges_uv) - 1, 0, n_uv - 2)
        iv = np.clip(np.digitize(v_fit, edges_uv) - 1, 0, n_uv - 2)
        ix = np.clip(np.digitize(X_fit, edges_x)  - 1, 0, n_xy - 2)
        iy = np.clip(np.digitize(Y_fit, edges_y)  - 1, 0, n_xy - 2)

        iu += (u_fit - edges_uv[iu]) > (edges_uv[iu + 1] - u_fit)
        iv += (v_fit - edges_uv[iv]) > (edges_uv[iv + 1] - v_fit)
        ix += (X_fit - edges_x[ix])  > (edges_x[ix + 1] - X_fit)
        iy += (Y_fit - edges_y[iy])  > (edges_y[iy + 1] - Y_fit)

        # --- output buffers --------------------------------------------
        N       = len(df_fit)
        u_pred  = np.empty(N, np.float32); v_pred  = np.empty_like(u_pred)
        X_pred  = np.empty_like(u_pred);   Y_pred  = np.empty_like(u_pred)
        th_pred = np.empty_like(u_pred);   ph_pred = np.empty_like(u_pred)

        iterator = tqdm(range(N), desc="Sampling 4-D state", unit="evt") \
                if show_progress and tqdm else range(N)

        # --- event loop -------------------------------------------------
        for n in iterator:
            if tqdm is None and show_progress and n % print_every == 0:
                print(f"[{n:>10}/{N}]")

            t   = str(df_fit["measured_type"].iat[n])
            col = flat4(iu[n], iv[n], ix[n], iy[n], n_uv, n_xy)

            rows, cdf = _rows_cdf_for(t, col, lut_base, renorm=False)  # ← renorm off

            # -------- draw GEN cell ----------------------------------------
            if rows is None:                         # uniform prior
                g = rng.integers(N_CELLS, dtype=np.int32)
            else:
                r   = rng.random()
                idx = np.searchsorted(cdf, r, side="right")
                if idx == rows.size:                 # r > cdf[-1]  →  clamp
                    idx = rows.size - 1
                g = rows[idx]


            iu_g, iv_g, ix_g, iy_g = unflat4(g, n_uv, n_xy)

            # --- X,Y ----------------------------------------------------
            if (ix_g == ix[n]) and (iy_g == iy[n]):
                X_pred[n], Y_pred[n] = X_fit[n], Y_fit[n]
            else:
                if uniform_setting:
                    X_pred[n] = rng.uniform(edges_x[ix_g], edges_x[ix_g + 1])
                    Y_pred[n] = rng.uniform(edges_y[iy_g], edges_y[iy_g + 1])
                else:
                    X_pred[n] = (edges_x[ix_g] + edges_x[ix_g + 1]) / 2
                    Y_pred[n] = (edges_y[iy_g] + edges_y[iy_g + 1]) / 2

            # --- u,v ----------------------------------------------------
            if (iu_g == iu[n]) and (iv_g == iv[n]):
                u_pred[n], v_pred[n] = u_fit[n], v_fit[n]
            else:
                if uniform_setting:
                    u_pred[n] = rng.uniform(edges_uv[iu_g], edges_uv[iu_g + 1])
                    v_pred[n] = rng.uniform(edges_uv[iv_g], edges_uv[iv_g + 1])
                else:
                    u_pred[n] = (edges_uv[iu_g] + edges_uv[iu_g + 1]) / 2
                    v_pred[n] = (edges_uv[iv_g] + edges_uv[iv_g + 1]) / 2

            sin_th     = min(math.hypot(u_pred[n], v_pred[n]), 1.0)
            th_pred[n] = math.asin(sin_th)
            ph_pred[n] = math.atan2(u_pred[n], v_pred[n])

        # --- assemble DataFrame ----------------------------------------
        return df_fit.assign(
            X_pred=X_pred, Y_pred=Y_pred,
            Theta_pred=th_pred, Phi_pred=ph_pred
        )


    df_input = og_og_df.sample(300_000).reset_index(drop=True)

    df_pred = sample_true_state_nearest(
        df_fit   = df_input,
        n_uv     = n_uv,
        n_xy     = n_xy,
        edges_uv = edges_uv,
        edges_x  = edges_x,
        edges_y  = edges_y,
        rng      = np.random.default_rng(),
        lut_dir  = "~/DATAFLOW_v3/TESTS/SIMULATION",
        show_progress = True,
        renorm   = False          # ← no second normalisation
    )



#%%


# Define binning
theta_bins = np.linspace(0, np.pi / 2, 200)
phi_bins = np.linspace(-np.pi, np.pi, 200)
tt_lists = [ VALID_MEASURED_TYPES ]

# df = df[ df["Theta_pred"] < 1.2 ]

for tt_list in tt_lists:
      
      # Create figure with 2 rows and 4 columns
      fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex='row')
      
      # Third column: Measured (θ_gen, ϕ_gen)
      axes[0, 0].hist(df_pred['Theta_gen'], bins=theta_bins, histtype='step', color='black', label='All')
      axes[1, 0].hist(df_pred['Phi_gen'], bins=phi_bins, histtype='step', color='black', label='All')
      for tt in tt_list:
            sel = (df_pred['measured_type'] == tt)
            axes[0, 0].hist(df_pred.loc[sel, 'Theta_gen'], bins=theta_bins, histtype='step', label=tt)
            axes[1, 0].hist(df_pred.loc[sel, 'Phi_gen'], bins=phi_bins, histtype='step', label=tt)
            axes[0, 0].set_title("Measured tracks θ_gen")
            axes[1, 0].set_title("Measured tracks ϕ_gen")

      # Fourth column: Measured (θ_fit, ϕ_fit)
      axes[0, 1].hist(df_pred['Theta_fit'], bins=theta_bins, histtype='step', color='black', label='All')
      axes[1, 1].hist(df_pred['Phi_fit'], bins=phi_bins, histtype='step', color='black', label='All')
      for tt in tt_list:
            sel = (df_pred['measured_type'] == tt)
            axes[0, 1].hist(df_pred.loc[sel, 'Theta_fit'], bins=theta_bins, histtype='step', label=tt)
            axes[1, 1].hist(df_pred.loc[sel, 'Phi_fit'], bins=phi_bins, histtype='step', label=tt)
            axes[0, 1].set_title("Measured tracks θ_fit")
            axes[1, 1].set_title("Measured tracks ϕ_fit")
      
      # Fourth column: Measured (θ_fit, ϕ_fit)
      axes[0, 2].hist(df_pred['Theta_pred'], bins=theta_bins, histtype='step', color='black', label='All')
      axes[1, 2].hist(df_pred['Phi_pred'], bins=phi_bins, histtype='step', color='black', label='All')
      for tt in tt_list:
            sel = (df_pred['measured_type'] == tt)
            axes[0, 2].hist(df_pred.loc[sel, 'Theta_pred'], bins=theta_bins, histtype='step', label=tt)
            axes[1, 2].hist(df_pred.loc[sel, 'Phi_pred'], bins=phi_bins, histtype='step', label=tt)
            axes[0, 2].set_title("Corrected tracks θ_fit")
            axes[1, 2].set_title("Corrected tracks ϕ_fit")

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


#%%

# df_pred = og_df

PLOT_DIR = f"{home_path}/DATAFLOW_v3/TESTS/SIMULATION"
tt_list = ['1234', '123', '234', '12', '23', '34']  # or VALID_MEASURED_TYPES

# ----------------------------------------------------------------------
# 2-D histograms (θ, φ) comparison
# ----------------------------------------------------------------------
theta_bins = np.linspace(0, np.pi/2, 50)
phi_bins   = np.linspace(-np.pi, np.pi, 50)

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
# 2-D histograms (X, Y) comparison
# ----------------------------------------------------------------------

distance_limit = max(distance_limit_x, distance_limit_y)

x_bins = np.linspace(-distance_limit, distance_limit, 100)
y_bins   = np.linspace(-distance_limit, distance_limit, 100)

groups = [tt_list]
for tt_group in groups:
    n_tt = len(tt_group)
    fig, ax = plt.subplots(n_tt, 3, figsize=(12, 4*n_tt), sharex=True, sharey=True)

    for i, tt in enumerate(tt_group):
        sel = df_pred["measured_type"] == tt

        # Measured (gen)
        ax[i,0].hist2d(df_pred.loc[sel,"X_gen"], df_pred.loc[sel,"Y_gen"],
                       bins=[x_bins, y_bins], cmap="viridis")
        ax[i,0].set_title("meas (gen)")

        # Measured (fit)
        ax[i,1].hist2d(df_pred.loc[sel,"X_fit"], df_pred.loc[sel,"Y_fit"],
                       bins=[x_bins, y_bins], cmap="viridis")
        ax[i,1].set_title("meas (fit)")

        # Predicted
        ax[i,2].hist2d(df_pred.loc[sel,"X_pred"], df_pred.loc[sel,"Y_pred"],
                       bins=[x_bins, y_bins], cmap="viridis")
        ax[i,2].set_title("pred")
        
        # Put the tt for that case as a title
        ax[i,0].set_title(f"{tt} – gen")
        ax[i,1].set_title(f"{tt} – fit")

    for a in ax[:,0]: a.set_ylabel(r"$Y$ [mm]")
    for a in ax[-1,:]: a.set_xlabel(r"$X$ [mm]")
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
      th_g, ph_g = df_pred.loc[mask,"Theta_gen"],  (df_pred.loc[mask,"Phi_gen"])
      th_f, ph_f = df_pred.loc[mask,"Theta_fit"],  (df_pred.loc[mask,"Phi_fit"])
      th_m, ph_m = df_pred.loc[mask,"Theta_pred"], (df_pred.loc[mask,"Phi_pred"])
    
      # Parameters for scatter plot
      scatter_size = 0.1
      scatter_alpha = 0.15

      # θ_gen vs θ_fit
      a=axes[i,0]; a.scatter(th_g,th_f,s=scatter_size,alpha=scatter_alpha); diag(a,[0,np.pi/2],[0,np.pi/2])
      a.set_xlabel(r"$\theta_{\rm gen}$"); a.set_ylabel(r"$\theta_{\rm fit}$")

      # φ_gen vs φ_fit
      a=axes[i,1]; a.scatter(ph_g,ph_f,s=scatter_size,alpha=scatter_alpha);
    #   diag(a,[-np.pi,np.pi],[-np.pi,np.pi])
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

# --------------------------------------------------------------------
# 5 · Scatter-matrix 6 columnas  (gen, fit, map)  —  X / Y version
# --------------------------------------------------------------------
xlim = ylim = (-distance_limit, distance_limit)   # same limit for X & Y

def diag(ax):
    ax.plot(xlim, ylim, "k--", lw=1)
    ax.set_aspect("equal")

n_tt = len(tt_list)
fig, axes = plt.subplots(n_tt, 6,
                         figsize=(20, 3.5 * n_tt),
                         sharex=False, sharey=False)

scatter_kw = dict(s=0.3, alpha=0.5)

for i, tt in enumerate(tt_list):
    mask = df_pred["measured_type"] == tt

    Xg, Yg = df_pred.loc[mask, "X_gen"], df_pred.loc[mask, "Y_gen"]
    Xf, Yf = df_pred.loc[mask, "X_fit"], df_pred.loc[mask, "Y_fit"]
    Xm, Ym = df_pred.loc[mask, "X_pred"], df_pred.loc[mask, "Y_pred"]

    # X_gen vs X_fit
    a = axes[i, 0]; a.scatter(Xg, Xf, **scatter_kw); diag(a)
    a.set_xlabel(r"$X_{\rm gen}$");  a.set_ylabel(r"$X_{\rm fit}$")

    # Y_gen vs Y_fit
    a = axes[i, 1]; a.scatter(Yg, Yf, **scatter_kw); diag(a)
    a.set_xlabel(r"$Y_{\rm gen}$");  a.set_ylabel(r"$Y_{\rm fit}$")

    # X_gen vs X_map
    a = axes[i, 2]; a.scatter(Xg, Xm, **scatter_kw); diag(a)
    a.set_xlabel(r"$X_{\rm gen}$");  a.set_ylabel(r"$X_{\rm map}$")

    # Y_gen vs Y_map
    a = axes[i, 3]; a.scatter(Yg, Ym, **scatter_kw); diag(a)
    a.set_xlabel(r"$Y_{\rm gen}$");  a.set_ylabel(r"$Y_{\rm map}$")

    # X_fit vs X_map
    a = axes[i, 4]; a.scatter(Xf, Xm, **scatter_kw); diag(a)
    a.set_xlabel(r"$X_{\rm fit}$");  a.set_ylabel(r"$X_{\rm map}$")

    # Y_fit vs Y_map
    a = axes[i, 5]; a.scatter(Yf, Ym, **scatter_kw); diag(a)
    a.set_xlabel(r"$Y_{\rm fit}$");  a.set_ylabel(r"$Y_{\rm map}$")

    # common limits
    for col in range(6):
        axes[i, col].set_xlim(xlim); axes[i, col].set_ylim(ylim)

plt.suptitle("Interaction-point reconstruction – likelihood map", y=1.02, fontsize=15)
plt.tight_layout()
plt.show()


#%%

n_tt = len(tt_list)
fig, axes = plt.subplots(n_tt, 4, figsize=(20, 3.5*n_tt), sharex=False, sharey=False)

for i, tt in enumerate(tt_list):
      mask = df_pred["measured_type"] == tt
      th_g, ph_g = df_pred.loc[mask,"Theta_gen"],  df_pred.loc[mask,"Phi_gen"]
      th_f, ph_f = df_pred.loc[mask,"Theta_fit"],  df_pred.loc[mask,"Phi_fit"]
      th_m, ph_m = df_pred.loc[mask,"Theta_pred"], df_pred.loc[mask,"Phi_pred"]
    
      # Parameters for scatter plot
      scatter_size = 0.5
      scatter_alpha = 0.025

      a=axes[i,0]; a.scatter(th_g,th_f - th_g,s=scatter_size,alpha=scatter_alpha);
      a.set_xlabel(r"$\theta_{\rm gen}$"); a.set_ylabel(r"$\theta_{\rm fit}$")
      a=axes[i,1]; a.scatter(ph_g,ph_f - ph_g,s=scatter_size,alpha=scatter_alpha);
      a.set_xlabel(r"$\phi_{\rm gen}$");  a.set_ylabel(r"$\phi_{\rm fit}$")
      a=axes[i,2]; a.scatter(th_g,th_m - th_g,s=scatter_size,alpha=scatter_alpha);
      a.set_xlabel(r"$\theta_{\rm gen}$"); a.set_ylabel(r"$\theta_{\rm map}$")
      a=axes[i,3]; a.scatter(ph_g,ph_m - ph_g,s=scatter_size,alpha=scatter_alpha);
      a.set_xlabel(r"$\phi_{\rm gen}$");  a.set_ylabel(r"$\phi_{\rm map}$")
      
      # Put the tt for that case as a title
      axes[i,0].set_title(f"{tt} – gen")
      axes[i,1].set_title(f"{tt} – fit")
      
      # Set y limits
      ylim = 1.5
      axes[i,0].set_ylim(-ylim, ylim)
      axes[i,1].set_ylim(-ylim, ylim)
      axes[i,2].set_ylim(-ylim, ylim)
      axes[i,3].set_ylim(-ylim, ylim)
      
plt.suptitle("Angular reconstruction – likelihood map", y=1.02, fontsize=15)
plt.tight_layout(); plt.show()

#%%

from scipy.stats import norm
from scipy.optimize import curve_fit

# Gaussian sum: 2-component model
def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return (a1 * norm.pdf(x, mu1, sigma1) +
            a2 * norm.pdf(x, mu2, sigma2))

# Residual histograms per topology with 2-Gaussian fits
n_tt = len(tt_list)
fig, axes = plt.subplots(n_tt, 4, figsize=(20, 3.0 * n_tt), sharex=False, sharey=False)

eps = 1e-6  # residuals close to 0 are discarded

# Binning and plot limits
bins_th = np.linspace(-0.4, 0.4, 150)
bins_ph = np.linspace(-0.8, 0.8, 150)
xlim_th = (-0.4, 0.4)
xlim_ph = (-0.8, 0.8)
ylim_all = [0, None]  # Automatic, can be fixed if needed

for i, tt in enumerate(tt_list):
    mask = df_pred["measured_type"] == tt
    th_g, ph_g = df_pred.loc[mask, "Theta_gen"], df_pred.loc[mask, "Phi_gen"]
    th_f, ph_f = df_pred.loc[mask, "Theta_fit"], df_pred.loc[mask, "Phi_fit"]
    th_m, ph_m = df_pred.loc[mask, "Theta_pred"], df_pred.loc[mask, "Phi_pred"]

    # Filter residuals
    res = {
        "th_fit": (th_f - th_g).abs().gt(eps),
        "th_map": (th_m - th_g).abs().gt(eps),
        "ph_fit": (ph_f - ph_g).abs().gt(eps),
        "ph_map": (ph_m - ph_g).abs().gt(eps)
    }

    residuals = {
        "th_fit": (th_f - th_g)[res["th_fit"]],
        "th_map": (th_m - th_g)[res["th_map"]],
        "ph_fit": (ph_f - ph_g)[res["ph_fit"]],
        "ph_map": (ph_m - ph_g)[res["ph_map"]]
    }

    keys = ["th_fit", "th_map", "ph_fit", "ph_map"]
    xlims = [xlim_th, xlim_th, xlim_ph, xlim_ph]
    bins_list = [bins_th, bins_th, bins_ph, bins_ph]
    labels = [r"$\theta_{\rm fit} - \theta_{\rm gen}$",
              r"$\theta_{\rm map} - \theta_{\rm gen}$",
              r"$\phi_{\rm fit} - \phi_{\rm gen}$",
              r"$\phi_{\rm map} - \phi_{\rm gen}$"]

    for j, key in enumerate(keys):
        ax = axes[i, j]
        data = residuals[key].dropna()
        if len(data) == 0:
            ax.set_title(f"{tt} – no data")
            continue

        # Histogram
        counts, bins_, _ = ax.hist(data, bins=bins_list[j], density=True,
                                   histtype="stepfilled", alpha=0.6, label="Data")

        x_mid = 0.5 * (bins_[:-1] + bins_[1:])

        # Initial guess: μ1, μ2 ~ 0, σ ~ std/2, amplitudes = ~half of max
        mu_guess = data.mean()
        sigma_guess = data.std()
        p0 = [0.5, mu_guess - 0.02, sigma_guess / 2,
              0.5, mu_guess + 0.02, sigma_guess / 2]

        try:
            popt, _ = curve_fit(double_gaussian, x_mid, counts, p0=p0)
            x_fit = np.linspace(bins_[0], bins_[-1], 500)
            y_fit = double_gaussian(x_fit, *popt)
            ax.plot(x_fit, y_fit, "r--", lw=1.5, label="2-Gauss fit")

            # Extract and convert sigmas to degrees
            σ1_deg = np.degrees(popt[2])
            σ2_deg = np.degrees(popt[5])

            # Add as text on the plot
            ax.text(0.02, 0.95,
                    fr"$\sigma_1$ = {σ1_deg:.2f}°" + "\n" +
                    fr"$\sigma_2$ = {σ2_deg:.2f}°",
                    transform=ax.transAxes,
                    fontsize=8, verticalalignment="top")
            
            ax.legend(fontsize=8)

            ax.legend(fontsize=8)
        except RuntimeError:
            ax.text(0.05, 0.9, "Fit failed", transform=ax.transAxes, color="red")

        ax.set_title(f"{tt} – {labels[j]}")
        ax.set_xlabel(labels[j])
        ax.set_ylabel("Density")
        ax.set_xlim(xlims[j])
        ax.set_ylim(ylim_all)

plt.suptitle("Residual Distributions – Sum of Gaussians Fit", y=1.005, fontsize=16)
plt.tight_layout()
plt.show()

# %%


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------- configuration -----------------------------------
nbins  = 60                       # bin granularity (x, y)
c_lim  = {"theta": 1.5,           # scale for Δθ  [rad]
          "phi"  : np.pi/3}       # scale for Δφ  [rad]  (±60°)

levels = {q: np.linspace(-c, c, 13) for q, c in c_lim.items()}
# cmap   = mpl.cm.get_cmap('RdBu_r')
cmap   = mpl.cm.get_cmap('viridis')

edges   = np.linspace(-1.0, 1.0, nbins + 1, dtype=np.float32)
centres = 0.5 * (edges[:-1] + edges[1:])
Xc, Yc  = np.meshgrid(centres, centres, indexing='ij')

# ------------------- helper ------------------------------------------
def wrap_phi(dphi):
    """Return angle in (-π, π]."""
    return (dphi + np.pi) % (2*np.pi) - np.pi

# ------------------- figure layout -----------------------------------
n_tt = len(tt_list)
fig, axes = plt.subplots(n_tt, 4, figsize=(14, 3.5*n_tt),
                         sharex=True, sharey=True, constrained_layout=True)

for i, tt in enumerate(tt_list):
    mask  = df_pred["measured_type"].values == tt

    th_g  = df_pred.loc[mask, "Theta_gen"].to_numpy(np.float32)
    ph_g  = df_pred.loc[mask, "Phi_gen"  ].to_numpy(np.float32)

    # generated direction in transverse Cartesian plane
    x_g = np.sin(th_g) * np.sin(ph_g)
    y_g = np.sin(th_g) * np.cos(ph_g)

    # reconstructed angles ------------------------------------------------
    th_f = df_pred.loc[mask, "Theta_fit" ].to_numpy(np.float32)
    ph_f = df_pred.loc[mask, "Phi_fit"   ].to_numpy(np.float32)
    th_m = df_pred.loc[mask, "Theta_pred"].to_numpy(np.float32)
    ph_m = df_pred.loc[mask, "Phi_pred"  ].to_numpy(np.float32)

    # residual arrays  (shape: 4 × N) ------------------------------------
    res = np.stack([
        th_f - th_g,                    # Δθ_fit
        wrap_phi(ph_f - ph_g),          # Δφ_fit
        th_m - th_g,                    # Δθ_map
        wrap_phi(ph_m - ph_g)           # Δφ_map
    ], axis=0)
    
    res = abs(res)
    
    # bin and plot -------------------------------------------------------
    for j in range(4):
        sum_r, _, _ = np.histogram2d(x_g, y_g,
                                     bins=(edges, edges),
                                     weights=res[j])
        count, _, _ = np.histogram2d(x_g, y_g, bins=(edges, edges))
        mean_r = np.divide(sum_r, count,
                           out=np.full_like(sum_r, np.nan, dtype=np.float32),
                           where=count > 0)

        qtype  = "theta" if j % 2 == 0 else "phi"
        ax     = axes[i, j]
        cf     = ax.contourf(Xc, Yc, mean_r.T,
                             levels=levels[qtype], cmap=cmap, extend='both')
        ax.contour (Xc, Yc, mean_r.T,
                    levels=levels[qtype], colors='k',
                    linewidths=0.25, alpha=0.3)

        ax.set_aspect('equal')
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

    # titles -------------------------------------------------------------
    labels = (r"$\langle\Delta\theta_{\rm fit}\rangle$",
              r"$\langle\Delta\phi_{\rm fit}\rangle$",
              r"$\langle\Delta\theta_{\rm map}\rangle$",
              r"$\langle\Delta\phi_{\rm map}\rangle$")
    for j in range(4):
        axes[i, j].set_title(f"{tt}: {labels[j]}")

    axes[i, 0].set_ylabel(r"$y_{\rm gen}=\sin\theta_{\rm gen}\cos\phi_{\rm gen}$")

for ax in axes[-1, :]:
    ax.set_xlabel(r"$x_{\rm gen}=\sin\theta_{\rm gen}\sin\phi_{\rm gen}$")

# Add one big colourbar on the right side
cax = fig.add_axes([1.02, 0.15, 0.02, 0.7])  # Adjust position and size
norm = mpl.colors.Normalize(vmin=-max(c_lim.values()), vmax=max(c_lim.values()))
mpl.colorbar.ColorbarBase(
    cax, cmap=cmap,
    norm=norm,
    orientation='vertical',
    ticks=np.linspace(-max(c_lim.values()), max(c_lim.values()), 7),
    label=r"Mean residual [rad]"
)

fig.suptitle(
    "Mean angular residuals in $(x_{\mathrm{gen}},\,y_{\mathrm{gen}})$ "
    "space with wrapped $\Delta\\phi$",
    y=1.04, fontsize=16)
plt.show()

#%%


import matplotlib as mpl

# ---------------- configuration ------------------------------------------
nbins   = 50
edges   = np.linspace(-1.0, 1.0, nbins + 1, dtype=np.float32)

cmap    = mpl.cm.get_cmap('RdBu_r')

def cartesian(th, ph):
    """Return (x,y) transverse components."""
    return np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph)

# ---------------- precompute relative differences ------------------------
rel_diffs = []          # will collect (rel_fit, rel_map) per tt
vmax      = 0.0         # global colour-scale half-range

for tt in tt_list:
    sel = df_pred["measured_type"].values == tt

    # generated
    th_g = df_pred.loc[sel, "Theta_gen"].to_numpy(np.float32)
    ph_g = df_pred.loc[sel, "Phi_gen"  ].to_numpy(np.float32)
    x_g, y_g = cartesian(th_g, ph_g)
    H_gen, _, _ = np.histogram2d(x_g, y_g, bins=(edges, edges))

    # likelihood-fit
    th_f = df_pred.loc[sel, "Theta_fit"].to_numpy(np.float32)
    ph_f = df_pred.loc[sel, "Phi_fit"  ].to_numpy(np.float32)
    x_f, y_f = cartesian(th_f, ph_f)
    H_fit, _, _ = np.histogram2d(x_f, y_f, bins=(edges, edges))

    # map
    th_m = df_pred.loc[sel, "Theta_pred"].to_numpy(np.float32)
    ph_m = df_pred.loc[sel, "Phi_pred" ].to_numpy(np.float32)
    x_m, y_m = cartesian(th_m, ph_m)
    H_map, _, _ = np.histogram2d(x_m, y_m, bins=(edges, edges))

    # relative differences, protecting empty bins
    rel_fit = np.divide(H_fit - H_gen, H_gen,
                        out=np.zeros_like(H_gen, dtype=np.float32),
                        where=H_gen > 0)
    rel_map = np.divide(H_map - H_gen, H_gen,
                        out=np.zeros_like(H_gen, dtype=np.float32),
                        where=H_gen > 0)

    # vmax = max(vmax, np.max(np.abs(rel_fit)), np.max(np.abs(rel_map)))
    vmax = 1.1
    rel_diffs.append((rel_fit, rel_map))

# ---------------- figure --------------------------------------------------
n_tt = len(tt_list)
fig, axes = plt.subplots(n_tt, 2,
                         figsize=(7, 3.5 * n_tt),
                         sharex=True, sharey=True,
                         constrained_layout=True)

for i, (tt, (rel_fit, rel_map)) in enumerate(zip(tt_list, rel_diffs)):

    for j, (data, label) in enumerate([(rel_fit, r"$\Delta_{\rm rel}^{\;{\it fit}}$"),
                                       (rel_map, r"$\Delta_{\rm rel}^{\;{\it map}}$")]):

        ax = axes[i, j]
        ax.pcolormesh(edges, edges, data.T,
                      cmap=cmap, vmin=-vmax, vmax=+vmax, shading='auto')
        ax.set_aspect('equal')
        ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"{tt}: {label}")

        if i == n_tt - 1:
            ax.set_xlabel(r"$x_{\rm gen}=\sin\theta_{\rm gen}\sin\phi_{\rm gen}$")
    axes[i, 0].set_ylabel(r"$y_{\rm gen}=\sin\theta_{\rm gen}\cos\phi_{\rm gen}$")

# ---------------- global colourbar ---------------------------------------
cax = fig.add_axes([1.02, 0.12, 0.02, 0.76])   # [left, bottom, width, height]
mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                          norm=mpl.colors.Normalize(vmin=-vmax, vmax=+vmax),
                          orientation='vertical',
                          ticks=np.linspace(-vmax, vmax, 7),
                          label=r"Relative difference $(N_{\rm rec}-N_{\rm gen})/N_{\rm gen}$")

fig.suptitle("Relative occupancy migration per bin",
             y=1.02, fontsize=16)
plt.show()

# %%
