#!/bin/env python3

#%%
from __future__ import annotations

#%%


# Execution variables
read_file = False


#%%

import os
import yaml
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
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

import numpy as np
import pandas as pd

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

show_plots = True

# Parameters and Constants
EFFS = [0.90, 0.91, 0.87, 0.90]
# EFFS = [0.2, 0.1, 0.4, 0.3]

# Iterants
n = 1  # Change this value to select 1 out of every n values
CROSS_EVS_LOW = 5 # 7 and 5 show a 33% of difference, which is more than the CRs will suffer
CROSS_EVS_UPP = 7
number_of_rates = 1

# Flux, area and counts calculations, keep generation area comparable to detector acceptance
z_plane = 100 # mm
ylim = 50 * z_plane # mm
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
else:
    FLUX = 1/12/60  # Default flux when no CLI override is provided

if number_of_rates == 1:
     AVG_CROSSING_EVS_PER_SEC_ARRAY = [ (CROSS_EVS_LOW + CROSS_EVS_UPP) / 2 ]
else:
     AVG_CROSSING_EVS_PER_SEC_ARRAY = np.linspace(CROSS_EVS_LOW, CROSS_EVS_UPP, number_of_rates)

# AVG_CROSSING_EVS_PER_SEC = 5.8
# Z_POSITIONS = [0, 145, 290, 435]
Z_POSITIONS = [30, 145, 290, 435]
Z_POSITIONS = np.array(Z_POSITIONS)
Z_POSITIONS = Z_POSITIONS - Z_POSITIONS[0]  # Normalize to first plane at z=0

print("Z_POSITIONS:", Z_POSITIONS)

Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]
AVALANCHE_WIDTH = 40.0  # mm of charge spread around the projected hit

def get_strip_geometry(plane_idx: int):
    y_width = Y_WIDTHS[0] if plane_idx in (1, 3) else Y_WIDTHS[1]
    total_width = np.sum(y_width)
    offsets = np.cumsum(np.concatenate(([0], y_width[:-1])))
    lower_edges = -total_width / 2 + offsets
    upper_edges = lower_edges + y_width
    centres = (lower_edges + upper_edges) / 2
    return y_width, centres, lower_edges, upper_edges


def num_strips_for_plane(plane_idx: int) -> int:
    y_width, _, _, _ = get_strip_geometry(plane_idx)
    return len(y_width)

# ----------------------------------------------
N_TRACKS = 10_000_000
# ----------------------------------------------

VALID_CROSSING_TYPES = ['1234', '123', '234', '12',  '23', '34']
VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']
TRIGGER_SELECTED = ['12', '23', '34', '13']

use_binary = True  # If True, will use a binary file instead of CSV
bin_filename = f"{home_path}/DATAFLOW_v3/MASTER/ANCILLARY/SIMULATORS/PLANE_COMBINATION_MONTECARLO/simulated_tracks_{N_TRACKS}.pkl"
csv_filename = f"{home_path}/DATAFLOW_v3/MASTER/ANCILLARY/SIMULATORS/PLANE_COMBINATION_MONTECARLO/simulated_tracks_{N_TRACKS}.csv"
Path(bin_filename).parent.mkdir(parents=True, exist_ok=True)



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
    
    
    def generate_valid_tracks(n_valid_needed, xlim, ylim, z_plane, cos_n=2, batch_size=100_000, max_simulated=None):
        

        rng = np.random.default_rng()
        exponent = 1 / (cos_n + 1)

        # Preallocate arrays
        X_gen_all = np.empty(n_valid_needed)
        Y_gen_all = np.empty(n_valid_needed)
        Z_gen_all = np.full(n_valid_needed, z_plane)
        Phi_gen_all = np.empty(n_valid_needed)
        Theta_gen_all = np.empty(n_valid_needed)
        crossing_type_all = np.empty(n_valid_needed, dtype=object)

        total_simulated = 0
        total_retained = 0

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
            'crossing_type': crossing_type_all
        })

        print(f"Done. Total simulated: {total_simulated:,}, retained: {len(df):,} --> efficiency: {len(df) / total_simulated:.3%}")
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


    def set_plane_efficiencies(df):
          df['eff_theoretical_1'] = EFFS[0]
          df['eff_theoretical_2'] = EFFS[1]
          df['eff_theoretical_3'] = EFFS[2]
          df['eff_theoretical_4'] = EFFS[3]


    def simulate_measured_points(df: pd.DataFrame,
                                   x_noise: float = 5.0,
                                   avalanche_width: float = AVALANCHE_WIDTH) -> None:
          n = len(df)
          rng = np.random.default_rng()
          measured_type = np.full(n, '', dtype=object)

          for i in range(1, 5):
              y_width, strip_centres, lower_edges, upper_edges = get_strip_geometry(i)
              n_strips = len(y_width)

              # NumPy views (no copy)
              eff   = df[f'eff_theoretical_{i}'].to_numpy(float, copy=False)
              x_gen = df[f'X_gen_{i}'].to_numpy(float, copy=False)
              y_gen = df[f'Y_gen_{i}'].to_numpy(float, copy=False)

              # Decide which tracks are detected (apply efficiency before geometry work)
              pass_mask = rng.random(n) <= eff

              # ----- X coordinate -----
              x_mea = np.full(n, np.nan)
              x_mea[pass_mask] = x_gen[pass_mask] + rng.normal(0.0, x_noise, pass_mask.sum())
              df[f'X_mea_{i}'] = x_mea

              # ----- Strip triggers -----
              strip_hits = np.zeros((n, n_strips), dtype=np.int8)
              valid_idx = np.flatnonzero(pass_mask & ~np.isnan(y_gen))
              if valid_idx.size:
                  y_valid = y_gen[valid_idx]
                  y_low = y_valid - avalanche_width / 2
                  y_high = y_valid + avalanche_width / 2

                  hits_valid = np.zeros((valid_idx.size, n_strips), dtype=np.int8)
                  for j in range(n_strips):
                      overlap = (y_low < upper_edges[j]) & (y_high > lower_edges[j])
                      hits_valid[overlap, j] = 1

                  strip_hits[valid_idx] = hits_valid

              plane_detected = strip_hits.any(axis=1)
              measured_type[plane_detected] = measured_type[plane_detected] + str(i)

              for j in range(n_strips):
                  df[f'strip_hit_{i}_{j+1}'] = strip_hits[:, j]

          df['measured_type'] = measured_type

    def fill_measured_type(df):
          df['filled_type'] = df['measured_type']
          df['filled_type'] = df['filled_type'].replace({'124': '1234', '134': '1234'})


    # ----------------------------------------------------------------------------
    # Remaining part of the code (simulation loop and CSV saving) ----------------
    # ----------------------------------------------------------------------------
    
    columns = ['X_gen', 'Y_gen', 'Theta_gen', 'Phi_gen', 'Z_gen'] + \
            [f'X_gen_{i}' for i in range(1, 5)] + [f'Y_gen_{i}' for i in range(1, 5)] + \
            ['crossing_type', 'measured_type', 'fitted_type']
    df_generated = pd.DataFrame(np.nan, index=np.arange(N_TRACKS), columns=columns)

    rng = np.random.default_rng()
    

    # 1. Generate valid tracks
    df_generated = generate_valid_tracks(N_TRACKS, xlim, ylim, z_plane, cos_n=2, batch_size=10_000)

    print("Tracks generated. Calculating intersections...")


    # 2. Calculate intersections
    calculate_intersections(df_generated, Z_POSITIONS)
    crossing_df = df_generated.copy()  # keep the full set before filtering
    df = df_generated[df_generated['crossing_type'].isin(VALID_CROSSING_TYPES)].copy()
    
    print("Intersections calculated. Generating measured points...")
    set_plane_efficiencies(df)


    # 3. Simulate measured points
    simulate_measured_points(df, avalanche_width=AVALANCHE_WIDTH)

    print("Measured points generated.")

    
    # 4. Filter based on trigger type
    # Take only the events in which any TRIGGER_SELECTED is contained in the measured_type
    df_triggered = df[df['measured_type'].apply(lambda mt: any(ts in mt for ts in TRIGGER_SELECTED))].copy()
    print(f"After trigger selection, {len(df_triggered)} tracks remain, which is a {len(df_triggered)/len(df):.2%} of the crossing tracks.")


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
    
    # 4. Filter based on trigger type
    # Take only the events in which any TRIGGER_SELECTED is contained in the measured_type
    df_triggered = df[df['measured_type'].apply(lambda mt: any(ts in mt for ts in TRIGGER_SELECTED))].copy()
    print(f"After trigger selection, {len(df_triggered)} tracks remain, which is a {len(df_triggered)/len(df):.2%} of the crossing tracks.")

#%%


# Print all column names
print("Columns in DataFrame:", df.columns.tolist())


# %%

df_test = df_triggered.copy()

# plot scatter plots and histogram in the diagonals of the main variables both generated and measured
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

plot_dir = Path(PLOT_DIR)
plot_dir.mkdir(parents=True, exist_ok=True)


def plot_scatter_grid(data: pd.DataFrame, columns: list[str], title: str, filename: str) -> None:
    """Render a scatter-matrix with histograms in the diagonal."""
    valid_cols = [col for col in columns if col in data.columns]
    if len(valid_cols) < 2:
        print(f"Not enough columns to plot {title}. Skipping...")
        return

    plot_data = data[valid_cols].dropna(how='all')
    if plot_data.empty:
        print(f"No data available for {title}. Skipping...")
        return

    axes = scatter_matrix(plot_data, alpha=0.3, figsize=(10, 10), diagonal='hist', hist_kwds={'bins': 100})
    for ax in axes.flatten():
        ax.grid(False)
    plt.suptitle(title)
    plt.tight_layout()

    out_path = plot_dir / filename
    plt.savefig(out_path, dpi=200)
    print(f"Saved {title} to {out_path}")
    if show_plots:
        plt.show()
    else:
        plt.close('all')

#%%


# Represent theta and phi generated in df, no matter the measured type
plot_scatter_grid(
    df,
    columns=['Theta_gen', 'Phi_gen'],
    title='Generated Angles (Theta vs Phi) for Triggered Tracks',
    filename='scatter_generated_angles_triggered.png'
)



# %%


# Plot a pretty complete report of the generated and measured data, per measured_type, include the angles and so on.

for measured_type in VALID_MEASURED_TYPES:
    mt_df = df_triggered[df_triggered['measured_type'] == measured_type].copy()
    if mt_df.empty:
        continue

    print(f"\n{'='*80}")
    print(f"Report for measured_type: {measured_type}")
    print(f"  Rows: {len(mt_df)}")
    print(f"  Theta range: [{mt_df['Theta_gen'].min():.4f}, {mt_df['Theta_gen'].max():.4f}] rad")
    print(f"  Phi range:   [{mt_df['Phi_gen'].min():.4f}, {mt_df['Phi_gen'].max():.4f}] rad")
    print(f"{'='*80}\n")

    # Create a multi-panel figure: 3 rows x 4 columns
    # Row 1: Theta, Phi, X_gen, Y_gen (generated angles & positions)
    # Row 2: X_mea_1-4, Y_gen_1-4 per plane (measured X, generated Y per plane)
    # Row 3: Strip hit counts and summary stats per plane
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    # --- Row 1: Generated angles and bulk positions ---
    # Theta distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(mt_df['Theta_gen'].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Theta (rad)')
    ax1.set_ylabel('Counts')
    ax1.set_title(f'{measured_type}: Theta distribution')
    ax1.grid(True, alpha=0.3)

    # Phi distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(mt_df['Phi_gen'].dropna(), bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Phi (rad)')
    ax2.set_ylabel('Counts')
    ax2.set_title(f'{measured_type}: Phi distribution')
    ax2.grid(True, alpha=0.3)

    # X_gen distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(mt_df['X_gen'].dropna(), bins=50, alpha=0.7, color='red', edgecolor='black')
    ax3.set_xlabel('X_gen (mm)')
    ax3.set_ylabel('Counts')
    ax3.set_title(f'{measured_type}: X_gen distribution')
    ax3.grid(True, alpha=0.3)

    # Y_gen distribution
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(mt_df['Y_gen'].dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Y_gen (mm)')
    ax4.set_ylabel('Counts')
    ax4.set_title(f'{measured_type}: Y_gen distribution')
    ax4.grid(True, alpha=0.3)

    # --- Row 2: Measured X and generated Y per plane ---
    for plane in range(1, 5):
        # X_mea distribution
        ax_xmea = fig.add_subplot(gs[1, plane - 1])
        col_xmea = f'X_mea_{plane}'
        if col_xmea in mt_df.columns:
            data_xmea = mt_df[col_xmea].dropna()
            if not data_xmea.empty:
                ax_xmea.hist(data_xmea, bins=40, alpha=0.7, color='purple', edgecolor='black')
        ax_xmea.set_xlabel(f'X_mea_{plane} (mm)')
        ax_xmea.set_ylabel('Counts')
        ax_xmea.set_title(f'{measured_type}: X_mea plane {plane}')
        ax_xmea.grid(True, alpha=0.3)

    # --- Row 3: Generated Y intersections per plane ---
    for plane in range(1, 5):
        ax_ygen = fig.add_subplot(gs[2, plane - 1])
        col_ygen = f'Y_gen_{plane}'
        if col_ygen in mt_df.columns:
            data_ygen = mt_df[col_ygen].dropna()
            if not data_ygen.empty:
                ax_ygen.hist(data_ygen, bins=40, alpha=0.7, color='cyan', edgecolor='black')
        ax_ygen.set_xlabel(f'Y_gen_{plane} (mm)')
        ax_ygen.set_ylabel('Counts')
        ax_ygen.set_title(f'{measured_type}: Y_gen plane {plane}')
        ax_ygen.grid(True, alpha=0.3)

    # --- Row 4: Strip hit count summary per plane ---
    for plane in range(1, 5):
        ax_strip = fig.add_subplot(gs[3, plane - 1])
        n_strips = num_strips_for_plane(plane)
        strip_hit_counts = []
        for strip in range(1, n_strips + 1):
            col_strip = f'strip_hit_{plane}_{strip}'
            if col_strip in mt_df.columns:
                count = int(mt_df[col_strip].sum())
                strip_hit_counts.append(count)
        if strip_hit_counts:
            ax_strip.bar(range(1, len(strip_hit_counts) + 1), strip_hit_counts, alpha=0.7, color='brown', edgecolor='black')
            ax_strip.set_xlabel(f'Strip')
            ax_strip.set_ylabel('Hit count')
            ax_strip.set_title(f'{measured_type}: Strip hits plane {plane}')
            ax_strip.set_xticks(range(1, len(strip_hit_counts) + 1))
            ax_strip.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Report: measured_type = {measured_type} (N={len(mt_df)})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    report_path = plot_dir / f'report_measured_type_{measured_type}.png'
    plt.savefig(report_path, dpi=180)
    print(f"Saved report to {report_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

#%%


# Save in a table the counts per measured_type and per crossing_type
measured_types = sorted(df_triggered['measured_type'].dropna().unique())
count_records: list[dict[str, object]] = []
for mt in measured_types:
    mt_mask = (df_triggered['measured_type'] == mt).to_numpy()
    for ct in VALID_CROSSING_TYPES:
        ct_mask = (df_triggered['crossing_type'] == ct).to_numpy()
        count = int(np.sum(mt_mask & ct_mask))
        # If its zero, skip
        if count == 0:
            continue
        count_records.append({
            'measured_type': mt,
            'crossing_type': ct,
            'count': count,
            'combination': f"{mt}_{ct}"
        })
counts_df = pd.DataFrame(count_records).sort_values(['measured_type', 'crossing_type'])
flat_path = plot_dir / 'measured_crossing_type_counts_flat.csv'
counts_df.to_csv(flat_path, index=False)
print(f"Saved detailed measured-crossing counts to {flat_path}")

print(counts_df)


#%%


# Save in a table the counts per measured_type only
measured_types = sorted(df_triggered['measured_type'].dropna().unique())
count_records: list[dict[str, object]] = []

for mt in measured_types:
    mt_mask = (df_triggered['measured_type'] == mt).to_numpy()
    count = int(np.sum(mt_mask))
    count_records.append({
        'measured_type': mt,
        'count': count,
    })

counts_df = pd.DataFrame(count_records).sort_values('measured_type')
counts_df['percentage'] = (counts_df['count'] / counts_df['count'].sum()) * 100
flat_path = plot_dir / 'measured_type_counts.csv'
counts_df.to_csv(flat_path, index=False)
print(f"Saved measured_type counts to {flat_path}")

print(counts_df)


#%%

# Count plane-strip combinations per measured type
measured_series = (
    df_triggered['measured_type']
    .fillna('')
    .astype(str)
    .str.strip()
)
measured_types = sorted([mt for mt in measured_series.unique() if mt])

if not measured_types:
    print("No measured_type labels available for plane-strip counting.")
else:
    measured_masks = {mt: (measured_series == mt).to_numpy() for mt in measured_types}
    pair_count_records: list[dict[str, object]] = []

    for plane_1 in range(1, 5):
        strips_p1 = num_strips_for_plane(plane_1)
        for strip_1 in range(1, strips_p1 + 1):
            col_1 = f'strip_hit_{plane_1}_{strip_1}'
            if col_1 not in df_triggered.columns:
                continue
            hits_1 = df_triggered[col_1].to_numpy(dtype=bool)

            for plane_2 in range(plane_1, 5):
                strips_p2 = num_strips_for_plane(plane_2)
                for strip_2 in range(1, strips_p2 + 1):
                    if plane_1 == plane_2 and strip_1 >= strip_2:
                        continue
                    col_2 = f'strip_hit_{plane_2}_{strip_2}'
                    if col_2 not in df_triggered.columns:
                        continue
                    hits_2 = df_triggered[col_2].to_numpy(dtype=bool)
                    pair_mask = hits_1 & hits_2

                    pair_label = f"P{plane_1}s{strip_1}_P{plane_2}s{strip_2}"
                    for mt, mt_mask in measured_masks.items():
                        count = int(np.sum(pair_mask & mt_mask))
                        pair_count_records.append({
                            'pair': pair_label,
                            'measured_type': mt,
                            'count': count,
                            'combination': f"{pair_label}_{mt}"
                        })

    if not pair_count_records:
        print("No plane-strip pair coincidences found for measured_type counting.")
    else:
        pair_counts_df = pd.DataFrame(pair_count_records).sort_values(['pair', 'measured_type'])
        flat_path = plot_dir / 'plane_strip_pair_counts_flat.csv'
        pair_counts_df.to_csv(flat_path, index=False)
        print(f"Saved detailed plane-strip counts to {flat_path}")

        counts_pivot = (
            pair_counts_df.pivot_table(
                index='pair',
                columns='measured_type',
                values='count',
                aggfunc='sum',
                fill_value=0,
            )
            .sort_index()
        )
        pivot_path = plot_dir / 'plane_strip_pair_counts.csv'
        counts_pivot.to_csv(pivot_path)
        print(f"Saved pivoted plane-strip counts to {pivot_path}")

        fig_height = max(6, 0.25 * len(counts_pivot))
        fig_width = max(6, 0.4 * len(counts_pivot.columns))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(counts_pivot.values, aspect='auto', cmap='viridis')
        ax.set_xticks(np.arange(len(counts_pivot.columns)), labels=counts_pivot.columns, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(counts_pivot.index)), labels=counts_pivot.index)
        ax.set_xlabel('Measured type')
        ax.set_ylabel('Plane-strip pair')
        ax.set_title('Plane-strip pair counts per measured type')
        fig.colorbar(im, ax=ax, label='Counts')

        counts_plot = plot_dir / 'plane_strip_pair_counts.png'
        plt.tight_layout()
        plt.savefig(counts_plot, dpi=200)
        print(f"Saved plane-strip count heatmap to {counts_plot}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        # Line-style comparison plots per plane pair
        measured_plot_types = sorted(measured_types)
        tt_colors = {tt: plt.get_cmap('tab20', len(measured_plot_types))(idx) for idx, tt in enumerate(measured_plot_types)}
        counts_by_pair = {
            pair: grp.set_index('measured_type')['count']
            for pair, grp in pair_counts_df.groupby('pair')
        }

        for plane_1 in range(1, 5):
            for plane_2 in range(plane_1, 5):
                fig, axs = plt.subplots(4, 4, figsize=(14, 10), sharex=True, sharey=True)
                legend_handles: dict[str, plt.Line2D] = {}

                for strip_1 in range(1, num_strips_for_plane(plane_1) + 1):
                    for strip_2 in range(1, num_strips_for_plane(plane_2) + 1):
                        ax = axs[strip_1 - 1, strip_2 - 1]

                        if plane_1 == plane_2 and strip_1 >= strip_2:
                            ax.axis('off')
                            continue

                        pair_label = f"P{plane_1}s{strip_1}_P{plane_2}s{strip_2}"
                        counts_series = counts_by_pair.get(pair_label)
                        if counts_series is None:
                            ax.axis('off')
                            continue

                        for tt in measured_plot_types:
                            count_val = counts_series.get(tt, 0)
                            if count_val == 0:
                                continue
                            (line,) = ax.plot(
                                [0, 1],
                                [count_val, count_val],
                                marker='.',
                                linestyle='-',
                                markersize=3,
                                color=tt_colors[tt],
                                label=tt,
                            )
                            legend_handles.setdefault(tt, line)

                        ax.set_title(f"P{plane_1}s{strip_1}-P{plane_2}s{strip_2}", fontsize=8)
                        ax.set_xticks([])
                        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

                fig.suptitle(
                    f"Plane-strip pair counts • planes P{plane_1}–P{plane_2}",
                    fontsize=14
                )

                if legend_handles:
                    fig.legend(
                        legend_handles.values(),
                        legend_handles.keys(),
                        loc='center right',
                        bbox_to_anchor=(1.15, 0.5),
                        fontsize='small'
                    )

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                counts_pair_plot = plot_dir / f"plane_pair_counts_P{plane_1}_P{plane_2}.png"
                plt.savefig(counts_pair_plot, dpi=180)
                print(f"Saved plane-pair counts comparison to {counts_pair_plot}")
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)

# %%

