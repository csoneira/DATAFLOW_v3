#%%
from __future__ import annotations

#%%

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
import math
from typing import Optional, Dict
from pathlib import Path
from tqdm import tqdm
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

import matplotlib as mpl

# ------------------------------------------------------------------------------
# Parameter definitions --------------------------------------------------------
# ------------------------------------------------------------------------------

PLOT_DIR = "/home/mingo/DATAFLOW_v3/TESTS/SIMULATION"
tt_list = ['1234', '123', '234', '12', '23', '34']  # or VALID_MEASURED_TYPES

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
else:
    FLUX =  1/12/60
    
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
N_TRACKS = 100000000
# ----------------------------------------------

BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)
VALID_CROSSING_TYPES = ['1234', '123', '234', '12',  '23', '34']
VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']

use_binary = True  # If True, will use a binary file instead of CSV
bin_filename = f"/home/mingo/DATAFLOW_v3/TESTS/SIMULATION/simulated_tracks_{N_TRACKS}.pkl"
csv_filename = f"/home/mingo/DATAFLOW_v3/TESTS/SIMULATION/simulated_tracks_{N_TRACKS}.csv"

fistensor2 = False
if fistensor2:
    print("Running with configuration for fistensor2")
    bin_filename = f"/home/petgfn/ALL_MINITRASGO/MINGO_LUT/simulated_tracks_{N_TRACKS}.pkl"
    csv_filename = f"/home/petgfn/ALL_MINITRASGO/MINGO_LUT/simulated_tracks_{N_TRACKS}.csv"
  
if use_binary:
      df = pd.read_pickle(bin_filename)
      print(f"DataFrame read from {bin_filename}")
else:
      df = pd.read_csv(csv_filename)
      print(f"DataFrame read from {csv_filename}")

# Keep only the n first rows, being n = 100000 of df
df = df.head(100000)

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

try:
    total_time_seconds = (df_test['time'].max() - df_test['time'].min()).total_seconds()
except AttributeError:
    total_time_seconds = 1

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
rates_LUT_filename = Path("/home/mingo/DATAFLOW_v3/TESTS/SIMULATION/rates_INDUCTION_SECTION/induction_section_lut.csv")
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
n_bins  = 100
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



#%%

# Load matrices from HDF5
matrices = {}
hdf_path = "likelihood_matrices.h5"

with pd.HDFStore(hdf_path, mode='r') as store:
    for key in store.keys():  # keys are like '/P1', '/P2', etc.
        ttype = key.strip("/")
        df_M = store.get(key)
        matrices[ttype] = df_M
        print(f"Loaded matrix for: {ttype}, shape = {df_M.shape}")
        n_bins = np.sqrt(df_M.shape[0]).astype(int)  # assuming square matrices

# Helpers
def flat(u_idx, v_idx, n_bins):
    return u_idx * n_bins + v_idx

def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def sample_true_angles_nearest(
      df_fit: pd.DataFrame,
      matrices: Optional[Dict[str, pd.DataFrame]],
      n_bins: int,
      rng: Optional[np.random.Generator] = None,
      show_progress: bool = True,
      print_every: int = 50_000
      ) -> pd.DataFrame:
      
    if rng is None:
        rng = np.random.default_rng()

    matrix_cache = {t: df_m.to_numpy() for t, df_m in matrices.items()}
    
    u_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    v_edges = np.linspace(-1.0, 1.0, n_bins + 1)

    u_fit = np.sin(df_fit["Theta_fit"].values) * np.sin(df_fit["Phi_fit"].values)
    v_fit = np.sin(df_fit["Theta_fit"].values) * np.cos(df_fit["Phi_fit"].values)

    iu = np.clip(np.digitize(u_fit, u_edges) - 1, 0, n_bins - 2)
    iv = np.clip(np.digitize(v_fit, v_edges) - 1, 0, n_bins - 2)

    iu += (u_fit - u_edges[iu]) > (u_edges[iu + 1] - u_fit)
    iv += (v_fit - v_edges[iv]) > (v_edges[iv + 1] - v_fit)

    flat_idx = lambda u, v: u * n_bins + v
    unflat = lambda k: divmod(k, n_bins)

    N = len(df_fit)
    theta_pred = np.empty(N, dtype=np.float32)
    phi_pred = np.empty(N, dtype=np.float32)

    iterator = tqdm(range(N), desc="Sampling true angles (nearest-bin)", unit="evt") if show_progress else range(N)

    for n in iterator:
        t_type = df_fit["measured_type"].iat[n]

        if t_type not in matrix_cache:
            raise ValueError(f"LUT not found for type: {t_type}")
        M = matrix_cache[t_type]

        col_idx = flat_idx(iu[n], iv[n])
        p = M[:, col_idx]
        s = p.sum()

        if s == 0:
            p = np.full_like(p, 1.0 / len(p))
        else:
            p /= s

        gen_idx = rng.choice(len(p), p=p)
        g_u_idx, g_v_idx = unflat(gen_idx)

        u_pred = rng.uniform(u_edges[g_u_idx], u_edges[g_u_idx + 1])
        v_pred = rng.uniform(v_edges[g_v_idx], v_edges[g_v_idx + 1])

        sin_theta = min(np.hypot(u_pred, v_pred), 1.0)
        theta_pred[n] = math.asin(sin_theta)
        phi_pred[n] = wrap_to_pi(math.atan2(u_pred, v_pred))

    df_out = df_fit.copy()
    df_out["Theta_pred"] = theta_pred
    df_out["Phi_pred"] = phi_pred
    return df_out


df_input = df.sample(900_000, random_state=2024).reset_index(drop=True)

df_input = df_input["Theta_fit", "Phi_fit"]
df_pred = sample_true_angles_nearest(
      df_fit=df_input,
      matrices=matrices,
      n_bins=n_bins,
      rng=np.random.default_rng(2025),
      show_progress=True
)

#%%

print(df_pred.columns.to_list())

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

