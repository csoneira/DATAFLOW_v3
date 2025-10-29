
# TO DO
# This script should be used to correct the angles according to the likelihood matrices with my method.




import glob
import pandas as pd
import random
import os
import sys

# Pick a random file in "/home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/DONE/cleaned_<file>.h5"
IN_PATH = glob.glob("/home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_4/DONE/fitted_*.h5")[random.randint(0, len(glob.glob("/home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_4/DONE/fitted_*.h5")) - 1)]
KEY = "df"

# Load dataframe
df = pd.read_hdf(IN_PATH, key=KEY)
print(f"✅ Fitted dataframe reloaded from: {IN_PATH}")

# --- Continue your calibration or analysis code here ---
# e.g.:
# run_calibration(working_df)


# Take basename of IN_PATH without extension and witouth the 'fitted_' prefix
basename_no_ext = os.path.splitext(os.path.basename(IN_PATH))[0].replace("fitted_", "")
print(f"File basename (no extension): {basename_no_ext}")



correct_angle = True




main_df = df.copy()
main_df['Theta_fit'] = main_df['theta']
main_df['Phi_fit'] = main_df['phi']

if correct_angle:
    print("----------------------------------------------------------------------")
    print("-------- 1. Correction of the fitted angle --> predicted angle -------")
    print("----------------------------------------------------------------------")

    # ---------------------------------------------------------------
    # 1. Build absolute path and sanity-check
    # ---------------------------------------------------------------
    hdf_path = os.path.join(config_files_directory, "likelihood_matrices.h5")
    if not os.path.isfile(hdf_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf_path}")

    #%%

    # ---------------------------------------------------------------
    # 2. Load all matrices into memory
    # ---------------------------------------------------------------
    matrices = {}
    n_bins = None

    with pd.HDFStore(hdf_path, mode='r') as store:
        keys = store.keys()
        if not keys:
            raise ValueError(f"{hdf_path} contains no datasets.")

        for key in keys:                     # keys like '/P1', '/P2', …
            ttype = key.strip('/')           # remove leading slash
            # df_M = store.get(key)
            
            # Reduce the precision to float32 to not kill RAM
            df_M = store.get(key).astype(np.float16)
            
            matrices[ttype] = df_M

            # set n_bins once, based on the first matrix's shape
            if n_bins is None:
                size = df_M.shape[0]
                n_bins = int(np.sqrt(size))
                if n_bins * n_bins != size:
                    raise ValueError(f"Matrix size {size} is not a perfect square.")

            print(f"Loaded matrix for {ttype}: shape {df_M.shape}")

    print(f"n_bins detected: {n_bins}")

    # Helpers
    def flat(u_idx, v_idx, n_bins):
        return u_idx * n_bins + v_idx

    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    #%%

    with pd.HDFStore(hdf_path, 'r') as store:
        print("HDF5 keys:", store.keys())

    def sample_true_angles_nearest(
        df_fit: pd.DataFrame,
        matrices: Optional[Dict[str, pd.DataFrame]],
        n_bins: int,
        rng: Optional[np.random.Generator] = None,
        show_progress: bool = True,
        print_every: int = 10_000
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
            t_type = str(df_fit["definitive_tt"].iat[n])   # ensure string

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

    #%%

    print(main_df.columns.to_list())

    #%%

    df_input = main_df
    df_pred = sample_true_angles_nearest(
                df_fit=df_input,
                matrices=matrices,
                n_bins=n_bins,
                rng=np.random.default_rng(),
                show_progress=True )

    df = df_pred.copy()
else:
    print("Angle correction is disabled.")
    df['Theta_pred'] = main_df['Theta_fit']
    df['Phi_pred'] = main_df['Phi_fit']


if create_very_essential_plots:    
    VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']
    tt_lists = [ VALID_MEASURED_TYPES ]
    
    for tt_list in tt_lists:
          fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex='row')

          # Fourth column: Measured (θ_fit, ϕ_fit)
          axes[0, 0].hist(df['Theta_fit'], bins=theta_bins, histtype='step', color='black', label='All')
          axes[1, 0].hist(df['Phi_fit'], bins=phi_bins, histtype='step', color='black', label='All')
          for tt in tt_list:
                sel = (df['definitive_tt'] == int(tt))
                axes[0, 0].hist(df.loc[sel, 'Theta_fit'], bins=theta_bins, histtype='step', label=tt)
                axes[1, 0].hist(df.loc[sel, 'Phi_fit'], bins=phi_bins, histtype='step', label=tt)
                axes[0, 0].set_title("Measured tracks θ_fit")
                axes[1, 0].set_title("Measured tracks ϕ_fit")
      
          # Fourth column: Measured (θ_fit, ϕ_fit)
          axes[0, 1].hist(df['Theta_pred'], bins=theta_bins, histtype='step', color='black', label='All')
          axes[1, 1].hist(df['Phi_pred'], bins=phi_bins, histtype='step', color='black', label='All')
          for tt in tt_list:
                sel = (df['definitive_tt'] == int(tt))
                axes[0, 1].hist(df.loc[sel, 'Theta_pred'], bins=theta_bins, histtype='step', label=tt)
                axes[1, 1].hist(df.loc[sel, 'Phi_pred'], bins=phi_bins, histtype='step', label=tt)
                axes[0, 1].set_title("Corrected tracks θ_fit")
                axes[1, 1].set_title("Corrected tracks ϕ_fit")

          # Common settings
          for ax in axes.flat:
                ax.legend(fontsize='x-small')
                ax.grid(True)

          axes[1, 0].set_xlabel(r'$\phi$ [rad]')
          axes[0, 0].set_ylabel('Counts')
          axes[1, 0].set_ylabel('Counts')
          axes[0, 1].set_xlim(0, np.pi / 2)
          axes[1, 1].set_xlim(-np.pi, np.pi)

          fig.tight_layout()
          plt.show()