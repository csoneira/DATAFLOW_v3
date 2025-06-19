

# #!/usr/bin/env python3
# """
# Lookup-map  (θ_fit , φ_fit) → (θ_map , φ_map)
# ---------------------------------------------
# * Para cada topología se construye un mapa  (Nθ × Nφ × 3)  con la media
#   vectorial de los ángulos verdaderos que caen en cada celda de la malla.
# * El mapa se interpola bilinealmente (RegularGridInterpolator).
# * Se añaden columnas Theta_map / Phi_map y se dibujan los mismos paneles
#   de dispersión que para la red neuronal.
# """

# # ----------------------------------------------------------------------
# # Imports
# # ----------------------------------------------------------------------
# import numpy as np
# import pandas as pd
# from scipy.interpolate import RegularGridInterpolator
# import matplotlib.pyplot as plt

# # ----------------------------------------------------------------------
# # 0 · INPUT  ------------------------------------------------------------
# # El DataFrame df_ml debe contener, al menos:
# #   Theta_fit, Phi_fit, Theta_gen, Phi_gen, measured_type
# # ----------------------------------------------------------------------
# df_ml = df_ml.copy()                      # si ya lo tienes cargado
# TOPOLOGIES = sorted(df_ml["measured_type"].unique())  # o lista manual

# # ----------------------------------------------------------------------
# # 1 · Parámetros de la malla
# # ----------------------------------------------------------------------
# TH_EDGES = np.linspace(0, np.pi/2, 50)          # 0–90°   → 0.5° pasos
# PH_EDGES = np.linspace(-np.pi, np.pi, 50)       # −180–180° → 1° pasos
# TH_CENT  = 0.5 * (TH_EDGES[:-1] + TH_EDGES[1:])
# PH_CENT  = 0.5 * (PH_EDGES[:-1] + PH_EDGES[1:])

# # ----------------------------------------------------------------------
# # 2 · Funciones helper
# # ----------------------------------------------------------------------
# def angles_to_unit(theta, phi):
#     """(θ, φ) → vector unitario (x,y,z)"""
#     return np.column_stack([np.sin(theta)*np.cos(phi),
#                             np.sin(theta)*np.sin(phi),
#                             np.cos(theta)]).astype(np.float32)

# def unit_to_angles(vec):
#     """vector unitario → (θ, φ) con φ ∈ (−π, π]"""
#     vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8, None)
#     x, y, z = vec[:,0], vec[:,1], vec[:,2]
#     theta = np.arccos(np.clip(z, -1, 1))
#     phi   = np.arctan2(y, x)
#     return theta, phi

# # ----------------------------------------------------------------------
# # 3 · Construir mapas por topología
# # ----------------------------------------------------------------------
# maps   = {}      # {tt: unit_map}
# interp = {}      # {tt: (ix,iy,iz)}

# for tt in TOPOLOGIES:
#     sub = df_ml[df_ml["measured_type"] == tt]
#     if len(sub) < 100:
#         print(f"[{tt}] omitido por escasa estadística")
#         continue

#     # Acumulador vectorial
#     sum_vec = np.zeros((len(TH_CENT), len(PH_CENT), 3), dtype=np.float64)

#     # Índices de celda para todo el subconjunto
#     i_idx = np.searchsorted(TH_EDGES, sub["Theta_fit"].values, side="right") - 1
#     j_idx = np.searchsorted(PH_EDGES, sub["Phi_fit"].values,   side="right") - 1

#     # Suma de vectores verdaderos (vectorización con add.at)
#     true_vec = angles_to_unit(sub["Theta_gen"].values, sub["Phi_gen"].values)
#     np.add.at(sum_vec, (i_idx, j_idx, slice(None)), true_vec)

#     # Media vectorial sobre la esfera -----------------------------
#     norm = np.linalg.norm(sum_vec, axis=2, keepdims=True)
#     unit_map = np.divide(sum_vec, norm, where=norm > 0)   # evita shape-mismatch
#     unit_map[~np.isfinite(unit_map)] = 0.0                # rellena NaN

#     maps[tt] = unit_map

#     # Interpoladores bilineales ----------------------------------
#     interp_x = RegularGridInterpolator((TH_CENT, PH_CENT), unit_map[:,:,0],
#                                        bounds_error=False, fill_value=np.nan)
#     interp_y = RegularGridInterpolator((TH_CENT, PH_CENT), unit_map[:,:,1],
#                                        bounds_error=False, fill_value=np.nan)
#     interp_z = RegularGridInterpolator((TH_CENT, PH_CENT), unit_map[:,:,2],
#                                        bounds_error=False, fill_value=np.nan)
#     interp[tt] = (interp_x, interp_y, interp_z)

#     filled = np.count_nonzero(norm[...,0] > 0)
#     print(f"[{tt}]  mapa listo  (celdas con datos: {filled}/{unit_map.size//3})")

# # ----------------------------------------------------------------------
# # 4 · Predicción para todo el DataFrame
# # ----------------------------------------------------------------------
# df_pred = df_ml.copy()
# df_pred["Theta_map"] = np.nan
# df_pred["Phi_map"]   = np.nan

# for tt, (ix, iy, iz) in interp.items():
#     sel = df_pred["measured_type"] == tt
#     if not np.any(sel):
#         continue

#     # Sub-arrays
#     theta_f = df_pred.loc[sel, "Theta_fit"].values
#     phi_f   = df_pred.loc[sel, "Phi_fit"].values
#     pts     = np.column_stack([theta_f, phi_f])

#     vx = ix(pts); vy = iy(pts); vz = iz(pts)
#     vec = np.column_stack([vx, vy, vz])

#     # fallback para celdas vacías: usar ángulo ajustado
#     nan_mask = np.isnan(vec).any(axis=1)
#     if nan_mask.any():
#         vec[nan_mask] = angles_to_unit(theta_f[nan_mask], phi_f[nan_mask])

#     # normalizar (por seguridad)
#     vec /= np.linalg.norm(vec, axis=1, keepdims=True)
#     th_pred, ph_pred = unit_to_angles(vec)

#     df_pred.loc[sel, "Theta_map"] = th_pred
#     df_pred.loc[sel, "Phi_map"]   = ph_pred

# print("✓ Predicciones completadas")

# # ----------------------------------------------------------------------
# # 5 · Scatter-matrix 6 columnas  (gen, fit, map)
# # ----------------------------------------------------------------------
# tt_list = TOPOLOGIES
# n_tt = len(tt_list)
# fig, axes = plt.subplots(n_tt, 6, figsize=(20, 3.5*n_tt), sharex=False, sharey=False)

# def diag(ax, xlim, ylim):
#     ax.plot(xlim, ylim, "k--", lw=1)
#     ax.set_aspect("equal")
#     ax.grid(True)

# for i, tt in enumerate(tt_list):
#     mask = df_pred["measured_type"] == tt
#     th_g, ph_g = df_pred.loc[mask,"Theta_gen"], df_pred.loc[mask,"Phi_gen"]
#     th_f, ph_f = df_pred.loc[mask,"Theta_fit"], df_pred.loc[mask,"Phi_fit"]
#     th_m, ph_m = df_pred.loc[mask,"Theta_map"], df_pred.loc[mask,"Phi_map"]

#     # θ_gen vs θ_fit
#     a=axes[i,0]; a.scatter(th_g,th_f,s=.5,alpha=.3); diag(a,[0,np.pi/2],[0,np.pi/2])
#     a.set_xlabel(r"$\theta_{\rm gen}$"); a.set_ylabel(r"$\theta_{\rm fit}$")

#     # φ_gen vs φ_fit
#     a=axes[i,1]; a.scatter(ph_g,ph_f,s=.5,alpha=.3); diag(a,[-np.pi,np.pi],[-np.pi,np.pi])
#     a.set_xlabel(r"$\phi_{\rm gen}$");  a.set_ylabel(r"$\phi_{\rm fit}$")

#     # θ_gen vs θ_map
#     a=axes[i,2]; a.scatter(th_g,th_m,s=.5,alpha=.3); diag(a,[0,np.pi/2],[0,np.pi/2])
#     a.set_xlabel(r"$\theta_{\rm gen}$"); a.set_ylabel(r"$\theta_{\rm map}$")

#     # φ_gen vs φ_map
#     a=axes[i,3]; a.scatter(ph_g,ph_m,s=.5,alpha=.3); diag(a,[-np.pi,np.pi],[-np.pi,np.pi])
#     a.set_xlabel(r"$\phi_{\rm gen}$");  a.set_ylabel(r"$\phi_{\rm map}$")

#     # θ_fit vs θ_map
#     a=axes[i,4]; a.scatter(th_f,th_m,s=.5,alpha=.3); diag(a,[0,np.pi/2],[0,np.pi/2])
#     a.set_xlabel(r"$\theta_{\rm fit}$"); a.set_ylabel(r"$\theta_{\rm map}$")

#     # φ_fit vs φ_map
#     a=axes[i,5]; a.scatter(ph_f,ph_m,s=.5,alpha=.3); diag(a,[-np.pi,np.pi],[-np.pi,np.pi])
#     a.set_xlabel(r"$\phi_{\rm fit}$");  a.set_ylabel(r"$\phi_{\rm map}$")

# plt.suptitle("Angular reconstruction – likelihood map", y=1.02, fontsize=15)
# plt.tight_layout(); plt.show()

# #%%

# # ----------------------------------------------------------------------
# # 2-D histograms (θ, φ) comparison
# # ----------------------------------------------------------------------
# theta_bins = np.linspace(0, np.pi/2, 150)
# phi_bins   = np.linspace(-np.pi, np.pi, 150)

# groups = [tt_list]                             # list of topology lists
# for tt_group in groups:
#     n_tt = len(tt_group)
#     fig, ax = plt.subplots(n_tt, 5, figsize=(20, 4*n_tt), sharex=True, sharey=True)

#     for i, tt in enumerate(tt_group):
#         sel = df_pred["measured_type"] == tt

#         # Generated
#         ax[i,0].hist2d(df_pred.loc[sel,"Theta_gen"], df_pred.loc[sel,"Phi_gen"],
#                        bins=[theta_bins, phi_bins], cmap="viridis")
#         ax[i,0].set_title(f"{tt} – gen")

#         # Crossing (example placeholder)
#         cross = df_pred["crossing_type"] == tt
#         ax[i,1].hist2d(df_pred.loc[cross,"Theta_cros"], df_pred.loc[cross,"Phi_cros"],
#                        bins=[theta_bins, phi_bins], cmap="viridis")
#         ax[i,1].set_title("crossing")

#         # Measured (gen)
#         ax[i,2].hist2d(df_pred.loc[sel,"Theta_gen"], df_pred.loc[sel,"Phi_gen"],
#                        bins=[theta_bins, phi_bins], cmap="viridis")
#         ax[i,2].set_title("meas (gen)")

#         # Measured (fit)
#         ax[i,3].hist2d(df_pred.loc[sel,"Theta_fit"], df_pred.loc[sel,"Phi_fit"],
#                        bins=[theta_bins, phi_bins], cmap="viridis")
#         ax[i,3].set_title("meas (fit)")

#         # Predicted
#         ax[i,4].hist2d(df_pred.loc[sel,"Theta_map"], df_pred.loc[sel,"Phi_map"],
#                        bins=[theta_bins, phi_bins], cmap="viridis")
#         ax[i,4].set_title("map")

#     for a in ax[:,0]: a.set_ylabel(r"$\phi$ [rad]")
#     for a in ax[-1,:]: a.set_xlabel(r"$\theta$ [rad]")
#     fig.tight_layout()
#     plt.savefig(PLOT_DIR / f"hist2d_{'_'.join(tt_group)}.png", dpi=150)
#     plt.show()
#     plt.close()

# # %%

# df = df_pred

# # Define binning
# theta_bins = np.linspace(0, np.pi / 2, 200)
# phi_bins = np.linspace(-np.pi, np.pi, 200)
# tt_lists = [ VALID_MEASURED_TYPES]

# for tt_list in tt_lists:
      
#       # Create figure with 2 rows and 4 columns
#       fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharex='row', sharey='row')
      
#       # First column: Generated angles
#       axes[0, 0].hist(df['Theta_gen'], bins=theta_bins, histtype='step', label='All', color='black')
#       axes[1, 0].hist(df['Phi_gen'], bins=phi_bins, histtype='step', label='All', color='black')
#       axes[0, 0].set_title("Generated θ")
#       axes[1, 0].set_title("Generated ϕ")

#       # Second column: Crossing detector (θ_gen, ϕ_gen)
#       # axes[0, 1].hist(crossing_df['Theta_gen'], bins=theta_bins, histtype='step', color='black', label='All')
#       # axes[1, 1].hist(crossing_df['Phi_gen'], bins=phi_bins, histtype='step', color='black', label='All')
#       # for tt in tt_list:
#       #       sel = (crossing_df['crossing_type'] == tt)
#       #       axes[0, 1].hist(crossing_df.loc[sel, 'Theta_gen'], bins=theta_bins, histtype='step', label=tt)
#       #       axes[1, 1].hist(crossing_df.loc[sel, 'Phi_gen'], bins=phi_bins, histtype='step', label=tt)
#       #       axes[0, 1].set_title("Crossing detector θ_gen")
#       #       axes[1, 1].set_title("Crossing detector ϕ_gen")
      
#       # Crossing detector (θ_gen, ϕ_gen) – now using df['Theta_cros'], df['Phi_cros']
#       axes[0, 1].hist(df['Theta_cros'], bins=theta_bins, histtype='step', color='black', label='All')
#       axes[1, 1].hist(df['Phi_cros'], bins=phi_bins, histtype='step', color='black', label='All')

#       for tt in tt_list:
#           sel = (df['crossing_type'] == tt)
#           axes[0, 1].hist(df.loc[sel, 'Theta_cros'], bins=theta_bins, histtype='step', label=tt)
#           axes[1, 1].hist(df.loc[sel, 'Phi_cros'], bins=phi_bins, histtype='step', label=tt)

#       axes[0, 1].set_title("Crossing detector θ_gen")
#       axes[1, 1].set_title("Crossing detector ϕ_gen")

      
#       # Third column: Measured (θ_gen, ϕ_gen)
#       axes[0, 2].hist(df['Theta_gen'], bins=theta_bins, histtype='step', color='black', label='All')
#       axes[1, 2].hist(df['Phi_gen'], bins=phi_bins, histtype='step', color='black', label='All')
#       for tt in tt_list:
#             sel = (df['measured_type'] == tt)
#             axes[0, 2].hist(df.loc[sel, 'Theta_gen'], bins=theta_bins, histtype='step', label=tt)
#             axes[1, 2].hist(df.loc[sel, 'Phi_gen'], bins=phi_bins, histtype='step', label=tt)
#             axes[0, 2].set_title("Measured tracks θ_gen")
#             axes[1, 2].set_title("Measured tracks ϕ_gen")

#       # Fourth column: Measured (θ_fit, ϕ_fit)
#       axes[0, 3].hist(df['Theta_fit'], bins=theta_bins, histtype='step', color='black', label='All')
#       axes[1, 3].hist(df['Phi_fit'], bins=phi_bins, histtype='step', color='black', label='All')
#       for tt in tt_list:
#             sel = (df['measured_type'] == tt)
#             axes[0, 3].hist(df.loc[sel, 'Theta_fit'], bins=theta_bins, histtype='step', label=tt)
#             axes[1, 3].hist(df.loc[sel, 'Phi_fit'], bins=phi_bins, histtype='step', label=tt)
#             axes[0, 3].set_title("Measured tracks θ_fit")
#             axes[1, 3].set_title("Measured tracks ϕ_fit")
      
      
#       # Fifth column: Measured (θ_fit, ϕ_fit)
#       axes[0, 4].hist(df['Theta_map'], bins=theta_bins, histtype='step', color='black', label='All')
#       axes[1, 4].hist(df['Phi_map'], bins=phi_bins, histtype='step', color='black', label='All')
#       for tt in tt_list:
#             sel = (df['measured_type'] == tt)
#             axes[0, 4].hist(df.loc[sel, 'Theta_map'], bins=theta_bins, histtype='step', label=tt)
#             axes[1, 4].hist(df.loc[sel, 'Phi_map'], bins=phi_bins, histtype='step', label=tt)
#             axes[0, 4].set_title("Measured tracks θ_map")
#             axes[1, 4].set_title("Measured tracks ϕ_map")
      
#       # Common settings
#       for ax in axes.flat:
#             ax.legend(fontsize='x-small')
#             ax.grid(True)

#       axes[1, 0].set_xlabel(r'$\phi$ [rad]')
#       axes[0, 0].set_ylabel('Counts')
#       axes[1, 0].set_ylabel('Counts')
#       axes[0, 2].set_xlim(0, np.pi / 2)
#       axes[1, 2].set_xlim(-np.pi, np.pi)

#       fig.tight_layout()
#       plt.show()





from tqdm import tqdm


import numpy as np
from collections import defaultdict

class BilinearLUT:
    """
    Bilinear stochastic mapper  (θ_fit, φ_fit) → (θ_pred, φ_pred)

    Parameters
    ----------
    theta_fit, phi_fit : 1-d arrays  [rad]
    theta_gen, phi_gen : 1-d arrays  [rad]      (same length)
    n_theta            : int          # grid points in θ (default 150)
    n_phi              : int          # grid points in φ (default 300)
    min_count          : int          # discard cells with < min_count hits
    """

    def __init__(self, theta_fit, phi_fit,
                       theta_gen,  phi_gen,
                       n_theta=50, n_phi=200, min_count=1):

        # --- 1. define regular grid (edges, not centres) ----------------
        self.th_edges = np.linspace(0.0, np.pi/2.0, n_theta + 1)
        self.ph_edges = np.linspace(-np.pi, np.pi, n_phi + 1)
        self.n_theta  = n_theta
        self.n_phi    = n_phi

        # --- 2. fill cells with GEN samples ----------------------------
        cell_samples = defaultdict(list)          # (i,j) → list[[θ_gen, φ_gen]]
        i = np.digitize(theta_fit, self.th_edges) - 1   # 0 … n_theta-1
        j = np.digitize(phi_fit,   self.ph_edges) - 1   # 0 … n_phi-1

        valid = (i >= 0) & (i < n_theta) & (j >= 0) & (j < n_phi)
        for ii, jj, tg, pg in tqdm(zip(i[valid], j[valid],
                        theta_gen[valid], phi_gen[valid]), desc="Filling cells"):
            cell_samples[(ii, jj)].append((tg, pg))

        # prune empty / low-stat cells
        self.samples = {key: np.asarray(lst, dtype=np.float32)
                        for key, lst in cell_samples.items()
                        if len(lst) >= min_count}

        # pre-compute cell sizes for fast interpolation
        self.dth = np.diff(self.th_edges)             # shape (n_theta,)
        self.dph = np.diff(self.ph_edges)             # shape (n_phi,)

    # ------------------------------------------------------------------
    def _draw_from_cell(self, key, rng):
        """Return one (θ,φ) sample from cell `key`; falls back to neighbour."""
        if key in self.samples:
            arr = self.samples[key]
            return arr[rng.integers(len(arr))]
        # nearest-neighbour fallback: search outward in Chebyshev shells
        i0, j0 = key
        for radius in range(1, max(self.n_theta, self.n_phi)):
            for ii in range(i0 - radius, i0 + radius + 1):
                for jj in range(j0 - radius, j0 + radius + 1):
                    if (ii, jj) in self.samples:
                        arr = self.samples[(ii, jj)]
                        return arr[rng.integers(len(arr))]
        # absolute fallback (should be unreachable)
        return np.nan, np.nan

    # ------------------------------------------------------------------
    def predict(self, theta_fit, phi_fit, seed=None):
        """
        Vectorised prediction for arrays of FIT angles.

        Returns
        -------
        theta_pred, phi_pred : 1-d arrays (same shape as input)
        """
        rng = np.random.default_rng(seed)
        theta_fit = np.asarray(theta_fit, dtype=np.float64)
        phi_fit   = np.asarray(phi_fit,   dtype=np.float64)

        # ensure φ in (−π,π]
        phi_fit = (phi_fit + np.pi) % (2*np.pi) - np.pi

        # locate lower-left cell indices i, j
        i = np.clip(np.digitize(theta_fit, self.th_edges) - 1, 0, self.n_theta - 2)
        j = np.clip(np.digitize(phi_fit,   self.ph_edges) - 1, 0, self.n_phi  - 2)

        # relative positions within cell  (0 ≤ f < 1)
        f_th = (theta_fit - self.th_edges[i]) / self.dth[i]
        f_ph = (phi_fit   - self.ph_edges[j]) / self.dph[j]

        # bilinear weights for four neighbours
        w00 = (1 - f_th) * (1 - f_ph)      # (i,   j)
        w10 = f_th       * (1 - f_ph)      # (i+1, j)
        w01 = (1 - f_th) * f_ph            # (i,   j+1)
        w11 = f_th       * f_ph            # (i+1, j+1)

        # draw one candidate from each neighbour
        cand00 = np.vstack([self._draw_from_cell((ii,  jj),   rng) for ii, jj in zip(i,   j  )])
        cand10 = np.vstack([self._draw_from_cell((ii+1,jj),   rng) for ii, jj in zip(i,   j  )])
        cand01 = np.vstack([self._draw_from_cell((ii,  jj+1), rng) for ii, jj in zip(i,   j  )])
        cand11 = np.vstack([self._draw_from_cell((ii+1,jj+1), rng) for ii, jj in zip(i,   j  )])

        # stack weights and candidates
        W = np.stack([w00, w10, w01, w11], axis=1)
        Cθ = np.stack([cand00[:,0], cand10[:,0], cand01[:,0], cand11[:,0]], axis=1)
        Cφ = np.stack([cand00[:,1], cand10[:,1], cand01[:,1], cand11[:,1]], axis=1)

        # choose one candidate per event according to weights
        # normalise (rows may contain zeros)
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum[W_sum == 0] = 1.0
        probs = W / W_sum

        # cumulative for inverse transform sampling
        cum = np.cumsum(probs, axis=1)
        r   = rng.random(size=len(theta_fit))
        idx = (r[:,None] < cum).argmax(axis=1)

        theta_pred = Cθ[np.arange(len(theta_fit)), idx]
        phi_pred   = Cφ[np.arange(len(theta_fit)), idx]

        return theta_pred, phi_pred



lut = BilinearLUT(theta_fit=df["Theta_fit"].values,
                  phi_fit  =df["Phi_fit"  ].values,
                  theta_gen=df["Theta_gen"].values,
                  phi_gen  =df["Phi_gen"  ].values,
                  n_theta  =50,
                  n_phi    =100,
                  min_count=3)


θp, φp = lut.predict(df["Theta_fit"].values, df["Phi_fit"].values, seed=12345)

df["Theta_pred"] = θp
df["Phi_pred"]   = φp


df_pred = df.copy()

#%%

# ----------------------------------------------------------------------
# 2-D histograms (θ, φ) comparison
# ----------------------------------------------------------------------
theta_bins = np.linspace(0, np.pi/2, 50)
phi_bins   = np.linspace(-np.pi, np.pi, 100)

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


import numpy as np
import pandas as pd
from collections import defaultdict

def build_LUT(df, n_theta=150, n_phi=300, min_count=3,
              csv_file="angular_LUT_stats.csv",
              npz_file="angular_LUT_edges.npz"):
    """
    Parameters
    ----------
    df : DataFrame with columns  Theta_fit, Phi_fit, Theta_gen, Phi_gen  [rad]
    n_theta, n_phi : grid granularity
    min_count      : discard cells with < min_count events
    """

    # --- grid edges ---------------------------------------------------
    th_edges = np.linspace(0.0, np.pi/2.0, n_theta + 1)
    ph_edges = np.linspace(-np.pi, np.pi,  n_phi   + 1)

    # --- assign each event to a cell ---------------------------------
    i = np.digitize(df["Theta_fit"], th_edges) - 1   # 0 … n_theta-1
    j = np.digitize(df["Phi_fit"],   ph_edges) - 1   # 0 … n_phi-1
    valid = (i >= 0) & (i < n_theta) & (j >= 0) & (j < n_phi)

    cell = list(zip(i[valid], j[valid]))
    tg   = df.loc[valid, "Theta_gen"].to_numpy()
    pg   = df.loc[valid, "Phi_gen"  ].to_numpy()

    # --- accumulate GEN samples per cell -----------------------------
    acc = defaultdict(list)
    for key, thg, phg in zip(cell, tg, pg):
        acc[key].append((thg, phg))

    # --- compute mean & covariance per cell --------------------------
    rows = []
    for (ii, jj), lst in acc.items():
        if len(lst) < min_count:
            continue
        arr = np.asarray(lst, dtype=np.float64)
        mu  = arr.mean(axis=0)                      # μθ, μφ
        cov = np.cov(arr.T)                         # 2×2
        rows.append((ii, jj, mu[0], mu[1],
                     cov[0, 0], cov[1, 1], cov[0, 1],
                     len(lst)))

    lut_df = pd.DataFrame(
        rows,
        columns=["i", "j",
                 "mu_theta", "mu_phi",
                 "var_theta", "var_phi", "cov_tphi",
                 "count"]
    )
    lut_df.to_csv(csv_file, index=False)
    np.savez_compressed(npz_file, theta_edges=th_edges, phi_edges=ph_edges)

    print(f"LUT exported: {csv_file} ({len(lut_df)} populated cells), {npz_file}")

# --------------------------------------------------------------------
# Example call
build_LUT(df, n_theta=150, n_phi=300)



#%%



import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class AngularLUTInterpolator:
    """
    Bilinear Gaussian sampler using the pre-computed CSV / NPZ files.
    """

    def __init__(self, csv_file, npz_file):
        # read stats
        df = pd.read_csv(csv_file)
        stats = {}
        for _, row in df.iterrows():
            key = (int(row.i), int(row.j))
            μ   = np.array([row.mu_theta, row.mu_phi])
            Σ   = np.array([[row.var_theta, row.cov_tphi],
                            [row.cov_tphi, row.var_phi]])
            stats[key] = (μ, Σ)

        self.stats = stats
        # grid edges
        ed = np.load(npz_file)
        self.th_edges = ed["theta_edges"]
        self.ph_edges = ed["phi_edges"]
        self.nθ = len(self.th_edges) - 1
        self.nφ = len(self.ph_edges) - 1
        self.dθ = np.diff(self.th_edges)
        self.dφ = np.diff(self.ph_edges)

    # --------------------------------------------------------------
    def _cell(self, θ, φ):
        i = np.clip(np.digitize(θ, self.th_edges) - 1, 0, self.nθ - 2)
        j = np.clip(np.digitize(φ, self.ph_edges) - 1, 0, self.nφ - 2)
        fθ = (θ - self.th_edges[i]) / self.dθ[i]     # 0–1
        fφ = (φ - self.ph_edges[j]) / self.dφ[j]
        return i, j, fθ, fφ

    def sample(self, θ_fit, φ_fit, rng=None):
        """
        Sample one (θ_pred, φ_pred) for each event (vectorised).
        """
        if rng is None:
            rng = np.random.default_rng()

        θ_fit = np.asarray(θ_fit, dtype=np.float64)
        φ_fit = np.asarray(φ_fit, dtype=np.float64)
        φ_fit = (φ_fit + np.pi) % (2*np.pi) - np.pi

        i, j, fθ, fφ = self._cell(θ_fit, φ_fit)

        # Bilinear weights
        w = np.stack([(1-fθ)*(1-fφ),  # (i,j)
                      fθ*(1-fφ),      # (i+1,j)
                      (1-fθ)*fφ,      # (i,j+1)
                      fθ*fφ], axis=1) # (i+1,j+1)

        # Resolve neighbour keys
        keys = np.stack([np.stack([i,   j  ], axis=1),
                         np.stack([i+1, j  ], axis=1),
                         np.stack([i,   j+1], axis=1),
                         np.stack([i+1, j+1], axis=1)], axis=1)

        out = np.empty((len(θ_fit), 2))
        for n in range(len(θ_fit)):
            probs = w[n]
            # normalise (cells with no stats get weight 0)
            for k in range(4):
                if tuple(keys[n,k]) not in self.stats:
                    probs[k] = 0.0
            s = probs.sum()
            if s == 0.0:              # global fallback
                out[n] = np.array([θ_fit[n], φ_fit[n]])
                continue
            probs = probs / s
            k = rng.choice(4, p=probs)

            μ, Σ = self.stats[tuple(keys[n,k])]
            out[n] = multivariate_normal.rvs(mean=μ, cov=Σ, random_state=rng)

        θ_pred, φ_pred = out[:,0], out[:,1]
        φ_pred = (φ_pred + np.pi) % (2*np.pi) - np.pi  # wrap
        θ_pred = np.clip(θ_pred, 0.0, np.pi/2.0)
        return θ_pred, φ_pred


# ------------------------------------------------------------------
# Usage with a real-data DataFrame  (only FIT angles known)
# ------------------------------------------------------------------
lut = AngularLUTInterpolator("angular_LUT_stats.csv",
                             "angular_LUT_edges.npz")

θ_pred, φ_pred = lut.sample(df["Theta_fit"], df["Phi_fit"])

df["Theta_pred"] = θ_pred
df["Phi_pred"]   = φ_pred

#%%



