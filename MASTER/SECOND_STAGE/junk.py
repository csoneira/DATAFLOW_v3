
# -------------------------------------------------------------------------------------------------------
# Stimated differences in efficiency --------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

data_df['eff_2_diff'] = ( data_df['eff_sys_123_2'] - data_df['eff_sys_2'] ) / data_df['eff_sys_2'] * 100
data_df['eff_3_diff'] = ( data_df['eff_sys_234_3'] - data_df['eff_sys_3'] ) / data_df['eff_sys_3'] * 100

# group_cols = [ 'eff_2_diff', 'eff_3_diff', 'streamer_percent_1', 'streamer_percent_2', 'streamer_percent_3', 'streamer_percent_4', ]
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')



# -------------------------------------------------------------------------------------------------------
# Noise derivations -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

data_df['eff_delta_2'] = data_df['eff_sys_123_2'] - data_df['eff_sys_2']
data_df['noise_frac_2'] = data_df['eff_delta_2'] / (data_df['eff_sys_2'] - data_df['eff_delta_2'])
data_df['noise_rate_2'] = data_df['noise_frac_2'] * data_df['subdetector_123_123'] / ( data_df["number_of_mins"] * 60 )  # or + 13 if total 2-plane

# Likewise for detector 3
data_df['eff_delta_3'] = data_df['eff_sys_234_3'] - data_df['eff_sys_3']
data_df['noise_frac_3'] = data_df['eff_delta_3'] / (data_df['eff_sys_3'] - data_df['eff_delta_3'])
data_df['noise_rate_3'] = data_df['noise_frac_3'] * data_df['subdetector_234_234'] / ( data_df["number_of_mins"] * 60 )


# group_cols = [ 'noise_rate_2', 'streamer_percent_2', ]
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')

# group_cols = [ 'noise_rate_3', 'streamer_percent_3', ]
# plot_pressure_and_group(data_df, 'sensors_ext_Pressure_ext', 'Pressure', group_cols, title='Pressure and Selected Columns')
# plot_pressure_and_group(data_df, 'sensors_ext_Temperature_ext', 'temperature', group_cols, title='Temperature and Selected Columns')


group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['streamer_percent_2', 'streamer_percent_3'],
    ['rates_M2', 'rates_M3'],
    ['th_chi'],
    ['sigmoid_width'],
    ['background_slope'],
    ['noise_rate_2', 'noise_rate_3'],
]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


# -------------------------------------------------------------------------------------------------------
# Noise study -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

# 'processed_tt_234', 'processed_tt_12', 'processed_tt_123', 'processed_tt_1234', 'processed_tt_23', 
# 'processed_tt_124', 'processed_tt_34', 'processed_tt_24', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14'

time_window = 200e-9  # 200 ns
data_df['noise_23'] = 2 * time_window * data_df['rates_M2'] * data_df['rates_M3']

group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['sigmoid_width'],
    ['background_slope'],
    ['noise_23'],
    ['noise_rate_2', 'noise_rate_3']
]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')



group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['noise_23'],
    ['processed_tt_12'],
    ['processed_tt_23'],
    ['processed_tt_34'],
    ['processed_tt_24'],
    ['processed_tt_13'],
    ['processed_tt_14'],
]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')

group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['noise_23'],
    ['processed_tt_123'],
    ['processed_tt_234'],
    ['processed_tt_134'],
    ['processed_tt_124'],
    ['processed_tt_1234'],
]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')



group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['noise_23'],
    ['original_tt_12'],
    ['original_tt_23'],
    ['original_tt_34'],
    ['original_tt_13'],
]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')


group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['noise_23'],
    ['original_tt_123'],
    ['original_tt_234'],
    ['original_tt_134'],
    ['original_tt_124'],
    ['original_tt_1234'],
]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')



group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['original_tt_134'],
    ['original_tt_124'],
    ['original_tt_1234'],
]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')



group_cols = [
    ['sensors_ext_Pressure_ext'],
    ['sensors_ext_Temperature_ext'],
    ['rates_M2', 'rates_M3'],
    ['tracking_tt_12'],
    ['tracking_tt_23'],
    ['tracking_tt_34'],
    ['tracking_tt_123'],
    ['tracking_tt_234'],
    ['tracking_tt_1234'],
]
# plot_grouped_series(data_df, group_cols, title='Detector Signals and Environment')











# -------------------------------------------------------------------------------------
# ----------------JUNK ----------------------------------------------------------------
# -------------------------------------------------------------------------------------

# high_regions_hans = ['V']
# mid_regions_hans = ['N.M', 'NE.M', 'E.M', 'SE.M', 'S.M', 'SW.M', 'W.M', 'NW.M']
# low_regions_hans = ['N.H', 'E.H', 'S.H', 'W.H']
# angular_regions = high_regions_hans + mid_regions_hans + low_regions_hans

# for reg in angular_regions:
#     data_df[f'pres_{reg}'] = data_df[reg] * np.exp(-1 * eta_P / 100 * delta_P)

# # Plot all the time series in angular_regions
# if create_plots:
#     print("Creating multi-panel count plots for all angular regions...")

#     # Create figures with 4 subplots each, sharing x-axis
#     fig, axes_original = plt.subplots(4, 1, figsize=(17, 12), sharex=True)
#     fig_corr, axes_corrected = plt.subplots(4, 1, figsize=(17, 12), sharex=True)

#     # Define angular region groups
#     regions_v = ['V']
#     regions_main = ['N.M', 'E.M', 'W.M', 'S.M']
#     regions_diagonal = ['NE.M', 'SE.M', 'SW.M', 'NW.M']
#     regions_h = ['N.H', 'E.H', 'S.H', 'W.H']

#     region_groups = [regions_v, regions_main, regions_diagonal, regions_h]

#     # ---- ORIGINAL COUNTS ----
#     for ax, regions in zip(axes_original, region_groups):
#         for region in regions:
#             ax.plot(data_df['Time'], data_df[region] / (60 * res_win_min), label=f'{region} (Hz)')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)

#     axes_original[0].set_title('Original Counts - V')
#     axes_original[1].set_title('Original Counts - N.M, E.M, W.M, S.M')
#     axes_original[2].set_title('Original Counts - NE.M, SE.M, SW.M, NW.M')
#     axes_original[3].set_title('Original Counts - H Regions (N.H, E.H, S.H, W.H)')
#     axes_original[-1].set_xlabel('Time')

#     # ---- PRESSURE-CORRECTED COUNTS ----
#     for ax, regions in zip(axes_corrected, region_groups):
#         for region in regions:
#             ax.plot(data_df['Time'], data_df[f'pres_{region}'], label=f'Corrected {region}')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)

#     axes_corrected[0].set_title('Pressure-Corrected Counts - V')
#     axes_corrected[1].set_title('Pressure-Corrected Counts - N.M, E.M, W.M, S.M')
#     axes_corrected[2].set_title('Pressure-Corrected Counts - NE.M, SE.M, SW.M, NW.M')
#     axes_corrected[3].set_title('Pressure-Corrected Counts - H Regions (N.H, E.H, S.H, W.H)')
#     fig.suptitle("Rates for All Angular Regions", fontsize=14, fontweight='bold')
#     axes_corrected[-1].set_xlabel('Time')

#     plt.tight_layout()

#     # Save or show the plots
#     if show_plots:
#         plt.show()
#     elif save_plots:
#         fig.savefig(figure_path + f"{fig_idx}" + "_counts_original.png", format='png', dpi=300)
#         fig_idx += 1
#         fig_corr.savefig(figure_path + f"{fig_idx}" + "_counts_corrected.png", format='png', dpi=300)
#         fig_idx += 1
#         print("Saved multi-panel count plots.")

#     plt.close(fig)
#     plt.close(fig_corr)

# if create_plots:
#     print("Creating multi-panel count plots for all angular regions...")

#     # Create figures with 4 subplots each, sharing x-axis
#     fig, axes_original = plt.subplots(4, 1, figsize=(17, 12), sharex=True)
#     fig_corr, axes_corrected = plt.subplots(4, 1, figsize=(17, 12), sharex=True)

#     # Define angular region groups
#     regions_v = ['V']
#     regions_main = ['N.M', 'E.M', 'W.M', 'S.M']
#     regions_diagonal = ['NE.M', 'SE.M', 'SW.M', 'NW.M']
#     regions_h = ['N.H', 'E.H', 'S.H', 'W.H']

#     region_groups = [regions_v, regions_main, regions_diagonal, regions_h]

#     # ---- ORIGINAL COUNTS ----
#     for ax, regions in zip(axes_original, region_groups):
#         norm_offset = 0
#         for region in regions:
#             y = data_df[region]
#             y_norm = (y - y.mean()) / y.mean() + norm_offset
#             norm_offset += 0.1
#             ax.plot(data_df['Time'], y_norm, label=f'{region}')
#             # ax.plot(data_df['Time'], data_df[region], label=f'{region}')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)

#     axes_original[0].set_title('Original Counts - V')
#     axes_original[1].set_title('Original Counts - N.M, E.M, W.M, S.M')
#     axes_original[2].set_title('Original Counts - NE.M, SE.M, SW.M, NW.M')
#     axes_original[3].set_title('Original Counts - H Regions (N.H, E.H, S.H, W.H)')
#     fig.suptitle("Normalized Counts for All Angular Regions", fontsize=14, fontweight='bold')
#     axes_original[-1].set_xlabel('Time')

#     # ---- PRESSURE-CORRECTED COUNTS ----
#     for ax, regions in zip(axes_corrected, region_groups):
#         for region in regions:
#             ax.plot(data_df['Time'], data_df[f'pres_{region}'], label=f'Corrected {region}')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)

#     axes_corrected[0].set_title('Pressure-Corrected Counts - V')
#     axes_corrected[1].set_title('Pressure-Corrected Counts - N.M, E.M, W.M, S.M')
#     axes_corrected[2].set_title('Pressure-Corrected Counts - NE.M, SE.M, SW.M, NW.M')
#     axes_corrected[3].set_title('Pressure-Corrected Counts - H Regions (N.H, E.H, S.H, W.H)')
#     axes_corrected[-1].set_xlabel('Time')

#     plt.tight_layout()

#     # Save or show the plots
#     if show_plots:
#         plt.show()
#     elif save_plots:
#         fig.savefig(figure_path + f"{fig_idx}" + "_counts_original_norm.png", format='png', dpi=300)
#         fig_idx += 1
#         fig_corr.savefig(figure_path + f"{fig_idx}" + "_counts_corrected_norm.png", format='png', dpi=300)
#         fig_idx += 1
#         print("Saved multi-panel count plots.")

#     plt.close(fig)
#     plt.close(fig_corr)



# # Define angular regions
# angular_regions = ['High', 'N', 'S', 'E', 'W']

# # Apply pressure correction
# for reg in angular_regions:
#     data_df[f'pres_{reg}'] = data_df[reg] * np.exp(-1 * eta_P / 100 * delta_P)

# # Plot all the time series in angular_regions
# if create_plots:
#     print("Creating multi-panel count plots for all angular regions...")

#     # Create figures with subplots, sharing x-axis
#     fig, axes_original = plt.subplots(len(angular_regions), 1, figsize=(17, 12), sharex=True)
#     fig_corr, axes_corrected = plt.subplots(len(angular_regions), 1, figsize=(17, 12), sharex=True)

#     # ---- ORIGINAL COUNTS ----
#     for ax, region in zip(axes_original, angular_regions):
#         ax.plot(data_df['Time'], data_df[region] / (60 * res_win_min), label=f'{region} (Hz)')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)
#         ax.set_title(f'Original Counts - {region}')
    
#     axes_original[-1].set_xlabel('Time')

#     # ---- PRESSURE-CORRECTED COUNTS ----
#     for ax, region in zip(axes_corrected, angular_regions):
#         ax.plot(data_df['Time'], data_df[f'pres_{region}'], label=f'Corrected {region}')
#         ax.set_ylabel('Counts')
#         ax.legend(loc='upper left', ncol=2, fontsize=8)
#         ax.set_title(f'Pressure-Corrected Counts - {region}')
    
#     axes_corrected[-1].set_xlabel('Time')

#     fig.suptitle("Rates for All Angular Regions", fontsize=14, fontweight='bold')
#     plt.tight_layout()

#     # Save or show the plots
#     if show_plots:
#         plt.show()
#     elif save_plots:
#         fig.savefig(figure_path + f"{fig_idx}" + "_angular_caye_OG.png", format='png', dpi=300)
#         fig_idx += 1
#         fig_corr.savefig(figure_path + f"{fig_idx}" + "_counts_caye_corrected.png", format='png', dpi=300)
#         fig_idx += 1
#         print("Saved multi-panel count plots.")

#     plt.close(fig)
#     plt.close(fig_corr)







# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import pearsonr
# # Efficiency model
# def compute_efficiencies_scaled(e1, e4, e2, e3):
#     return np.vstack([
#         e1 * e2 * e3 * e4 + e1 * (1 - e2) * e3 * e4 + e1 * e2 * (1 - e3) * e4 + e1 * (1 - e2) * (1 - e3) * e4,
#         e1 * e2 * e3 + e1 * (1 - e2) * e3,
#         e2 * e3 * e4 + e2 * (1 - e3) * e4,
#         e1 * e2,
#         e2 * e3,
#         e3 * e4
#     ]).T

# # Objective function: minimize variance of corrected rates
# def block_objective(params, block_df):
#     alpha, beta = params
#     e1 = alpha * block_df['final_eff_1'].values
#     e2 = block_df['final_eff_2'].values
#     e3 = block_df['final_eff_3'].values
#     e4 = beta * block_df['final_eff_4'].values

#     eff_model = compute_efficiencies_scaled(e1, e4, e2, e3)
#     measured = block_df[['detector_1234', 'detector_123', 'detector_234',
#                          'detector_12', 'detector_23', 'detector_34']].values
#     corrected = measured / eff_model
#     return np.sum(np.var(corrected, axis=0))

# # Run block-wise optimization
# block_size = 40
# n_blocks = (len(data_df) + block_size - 1) // block_size
# alpha_vals = np.zeros(len(data_df))
# beta_vals = np.zeros(len(data_df))

# for block_idx in range(n_blocks):
#     start = block_idx * block_size
#     end = min((block_idx + 1) * block_size, len(data_df))
#     block_df = data_df.iloc[start:end]

#     res = minimize(block_objective,
#                    x0=[1.0, 1.0],
#                    args=(block_df,),
#                    bounds=[(0.5, 2.0), (0.5, 2.0)],
#                    method='SLSQP')

#     alpha_opt, beta_opt = res.x if res.success else (1.0, 1.0)
#     alpha_vals[start:end] = alpha_opt
#     beta_vals[start:end] = beta_opt

# # Apply scaled efficiencies to the dataframe
# data_df['e1_scaled'] = alpha_vals * data_df['final_eff_1']
# data_df['e4_scaled'] = beta_vals * data_df['final_eff_4']

# # Apply optimal scaling
# data_df['e1_opt_scaled'] = alpha_opt * data_df['final_eff_1']
# data_df['e4_opt_scaled'] = beta_opt * data_df['final_eff_4']

# # Assign for reuse
# e1 = data_df['e1_opt_scaled']
# e2 = data_df['final_eff_2']
# e3 = data_df['final_eff_3']
# e4 = data_df['e4_opt_scaled']