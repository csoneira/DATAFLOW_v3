# LEGACY CODE ARCHIVE FOR TASK 2
#
# This file stores code deliberately removed from the active task script.
# It is not imported by the pipeline and is not expected to execute.
# Dependencies and surrounding runtime context may be missing.
# Preserve archived blocks substantially verbatim for historical reference.

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: literal `if False:` block; unconditionally unreachable
# ============================================================================
# if False:
#     print('Working in fast mode.')

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: literal `if False:` block; unconditionally unreachable
# ============================================================================
# if False:
#     print('Working in debug mode.')

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: literal `if False:` block; unconditionally unreachable
# ============================================================================
# if False:
#     T_F_left_pre_cal = T_side_left_pre_cal_debug
#     T_F_right_pre_cal = T_side_right_pre_cal_debug
#
#     T_B_left_pre_cal = T_side_left_pre_cal_debug
#     T_B_right_pre_cal = T_side_right_pre_cal_debug
#
# else:
#     T_F_left_pre_cal = T_side_left_pre_cal_default  #-130
#     T_F_right_pre_cal = T_side_right_pre_cal_default
#
#     T_B_left_pre_cal = T_side_left_pre_cal_default
#     T_B_right_pre_cal = T_side_right_pre_cal_default

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: literal `if False:` block; unconditionally unreachable
# ============================================================================
# if False:
#     T_clip_min = T_clip_min_debug
#     T_clip_max = T_clip_max_debug
#     Q_clip_min = Q_clip_min_debug
#     Q_clip_max = Q_clip_max_debug
#     num_bins = num_bins_debug
# else:
#     T_clip_min = T_clip_min_default
#     T_clip_max = T_clip_max_default
#     Q_clip_min = Q_clip_min_default
#     Q_clip_max = Q_clip_max_default
#     num_bins = num_bins_default

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: hard-setting for removed unreachable slewing-fit block
# ============================================================================
# _need_slewing_fit = False

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: hard-setting for removed superseded early T_sum branch
# ============================================================================
# _defer_tsum_calibration_until_late_slot = True

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: `_need_slewing_fit` is hard-set to False; legacy regression/fit block is unreachable
# ============================================================================
#     if _need_slewing_fit:
#         pair_labels = [
#                 (p1, s1, p2, s2)
#                 for p1 in range(1, 5)
#                 for s1 in range(1, 5)
#                 for p2 in range(p1 + 1, 5)
#                 for s2 in range(1, 5)
#             ]
#         
#         # Store fitted model parameters
#         fit_results = []
#
#         def robust_z_filter(df, cols, threshold=3.5):
#
#             mask = np.ones(len(df), dtype=bool)
#             for col in cols:
#                 median = np.median(df[col])
#                 mad = median_abs_deviation(df[col], scale='normal')  # consistent with std if normal
#                 if mad == 0:
#                     continue  # skip flat distributions
#                 z_mod = 0.6745 * (df[col] - median) / mad
#                 mask &= np.abs(z_mod) < threshold
#             return df[mask]
#         
#         for (p1, s1, p2, s2) in pair_labels:
#             mask = (
#                 (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
#                 (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
#             )
#             data = slew_df[mask]
#
#             valid_mask = (
#                 (data['Q_sum_2'] != 0) &
#                 (data['Q_sum_1'] != 0) &
#                 (data['delta_t_corrected'] != 0)
#             )
#             data = data[valid_mask]
#
#             if len(data) < strip_pair_min_events:
#                 continue  # not enough data to fit
#
#             # The T_sum calibration sample is defined upstream. Do not trim it
#             # again here with semisum/semidiff/timing windows.
#             # data = robust_z_filter(data, ['Q_sum_2', 'Q_sum_1', 'delta_t_corrected'])
#             
#             if not_use_q_semisum:
#                 X = data[['Q_sum_2']].values
#                 y = data['delta_t_corrected'].values
#
#                 model = LinearRegression()
#                 model.fit(X, y)
#                 
#                 b_semidiff = model.coef_[0]
#                 a_semisum = 0
#                 r2_score_val = model.score(X, y)
#             else:
#                 X = data[['Q_sum_1', 'Q_sum_2']].values
#                 y = data['delta_t_corrected'].values
#
#                 model = LinearRegression()
#                 model.fit(X, y)
#                 
#                 a_semisum = model.coef_[0]
#                 b_semidiff = model.coef_[1]
#                 r2_score_val = model.score(X, y)
#
#             # Store results
#             fit_results.append({
#                 'plane1': p1, 'strip1': s1,
#                 'plane2': p2, 'strip2': s2,
#                 'a_semisum': a_semisum,
#                 'b_semidiff': b_semidiff,
#                 'c_offset': model.intercept_,
#                 'n_points': len(data),
#                 'r2_score': r2_score_val
#             })
#
#         # Create dataframe with all model parameters
#         disabled_fit_df = pd.DataFrame(fit_results)
#         
#         print("Fitting results:")
#         print(disabled_fit_df)
#         
#         
#         # -------------------------------------------------------------------
#         # 3D Slewing with fit projections
#         # -------------------------------------------------------------------
#         
#         
#         if _plot_slewing_3d_fitproj:
#             pair_labels = [
#                 (p1, s1, p2, s2)
#                 for p1 in range(1, 5)
#                 for s1 in range(1, 5)
#                 for p2 in range(p1 + 1, 5)
#                 for s2 in range(1, 5)
#             ]
#
#             batch_size = 8
#             num_batches = int(np.ceil(len(pair_labels) / batch_size))
#
#             for batch in range(num_batches):
#                 start_idx = batch * batch_size
#                 end_idx = min((batch + 1) * batch_size, len(pair_labels))
#                 current_pairs = pair_labels[start_idx:end_idx]
#
#                 fig = plt.figure(figsize=(6 * batch_size, 16), constrained_layout=True)
#                 spec = gridspec.GridSpec(nrows=4, ncols=batch_size, figure=fig)
#
#                 for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
#                     mask = (
#                         (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
#                         (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
#                     )
#                     data = slew_df[mask]
#
#                     valid_mask = (
#                         (data['Q_sum_2'] != 0) &
#                         (data['Q_sum_1'] != 0) &
#                         (data['delta_t_corrected'] != 0)
#                     )
#                     data = data[valid_mask]
#
#                     if len(data) < 10:
#                         continue
#
#                     # Retrieve fitted model coefficients from disabled_fit_df
#                     fit_row = disabled_fit_df[
#                         (disabled_fit_df['plane1'] == p1) &
#                         (disabled_fit_df['strip1'] == s1) &
#                         (disabled_fit_df['plane2'] == p2) &
#                         (disabled_fit_df['strip2'] == s2)
#                     ]
#                     if fit_row.empty:
#                         continue
#
#                     a = fit_row['a_semisum'].values[0]
#                     b = fit_row['b_semidiff'].values[0]
#                     c = fit_row['c_offset'].values[0]
#
#                     x = data['delta_t_corrected'].values
#                     y = data['Q_sum_1'].values
#                     z = data['Q_sum_2'].values
#
#                     # 3D plot
#                     ax3d = fig.add_subplot(spec[0:2, col_idx], projection='3d')
#                     ax3d.scatter(x, y, z, s=5, alpha=0.6)
#                     ax3d.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
#                     ax3d.set_xlabel('ΔT corrected (ns)')
#                     ax3d.set_ylabel('Q_sum_1')
#                     ax3d.set_zlabel('Q_sum_2')
#
#                     ax3d.set_xlim([delta_t_left, delta_t_right])
#                     ax3d.set_ylim([q_sum_left, q_sum_right])
#                     ax3d.set_zlim([q_dif_left, q_dif_right])
#
#                     # XY projection
#                     ax_xy = fig.add_subplot(spec[2, col_idx])
#                     ax_xy.scatter(x, y, s=5, alpha=0.5)
#                     z_fixed = np.mean(z)
#                     y_line = np.linspace(np.min(y), np.max(y), 100)
#                     x_line = a * y_line + b * z_fixed + c
#                     ax_xy.plot(x_line, y_line, color='black', lw=1, label='Fitted projection')
#                     ax_xy.set_xlabel('ΔT corrected')
#                     ax_xy.set_ylabel('Q_sum_1')
#                     ax_xy.set_title('XY projection')
#                     ax_xy.legend(fontsize='x-small')
#                     ax_xy.set_xlim([delta_t_left, delta_t_right])
#                     ax_xy.set_ylim([q_sum_left, q_sum_right])
#
#                     # XZ projection
#                     ax_xz = fig.add_subplot(spec[3, col_idx])
#                     ax_xz.scatter(x, z, s=5, alpha=0.5, c='tab:red')
#                     y_fixed = np.mean(y)
#                     z_line = np.linspace(np.min(z), np.max(z), 100)
#                     x_line2 = a * y_fixed + b * z_line + c
#                     ax_xz.plot(x_line2, z_line, color='black', lw=1, label='Fitted projection')
#                     ax_xz.set_xlabel('ΔT corrected')
#                     ax_xz.set_ylabel('Q_sum_2')
#                     ax_xz.set_title('XZ projection')
#                     ax_xz.legend(fontsize='x-small')
#                     ax_xz.set_xlim([T_sum_corrected_dif_left, T_sum_corrected_dif_right])
#                     ax_xz.set_ylim([Q_sum_semidiff_left, Q_sum_semidiff_right])
#
#                 fig.tight_layout(rect=[0, 0, 1, 0.95])
#                 fig.suptitle(f"Batch {batch + 1}/{num_batches} — 3D Slewing + Fitted Projections", fontsize=16)
#                 
#                 if save_plots:
#                     name_of_file = 'slewing_3d_fitproj'
#                     final_filename = f'{fig_idx}_{name_of_file}.png'
#                     fig_idx += 1
#
#                     save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#                     plot_list.append(save_fig_path)
#                     save_plot_figure(save_fig_path, format='png')
#
#                 if show_plots:
#                     plt.show()
#                 plt.close()
#
#     
#     # -------------------------------------------------------------------
#     # FIT VALIDATION with y = x
#     # -------------------------------------------------------------------
#     
#     
#         if _plot_slewing_fit_validation:
#             pair_labels = [
#                 (p1, s1, p2, s2)
#                 for p1 in range(1, 5)
#                 for s1 in range(1, 5)
#                 for p2 in range(p1 + 1, 5)
#                 for s2 in range(1, 5)
#             ]
#
#             batch_size = 8
#             num_batches = int(np.ceil(len(pair_labels) / batch_size))
#             
#             for batch in range(num_batches):
#                 start_idx = batch * batch_size
#                 end_idx = min((batch + 1) * batch_size, len(pair_labels))
#                 current_pairs = pair_labels[start_idx:end_idx]
#
#                 fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8), constrained_layout=True)
#                 axes = np.atleast_2d(axes)
#
#                 for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
#                     mask = (
#                         (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
#                         (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
#                     )
#                     data = slew_df[mask]
#
#                     valid_mask = (
#                         (data['Q_sum_2'] != 0) &
#                         (data['Q_sum_1'] != 0) &
#                         (data['delta_t_corrected'] != 0)
#                     )
#                     data = data[valid_mask]
#
#                     if len(data) < 10:
#                         for row in range(2):
#                             axes[row, col_idx].set_visible(False)
#                         continue
#
#                     fit_row = disabled_fit_df[
#                         (disabled_fit_df['plane1'] == p1) &
#                         (disabled_fit_df['strip1'] == s1) &
#                         (disabled_fit_df['plane2'] == p2) &
#                         (disabled_fit_df['strip2'] == s2)
#                     ]
#                     if fit_row.empty:
#                         for row in range(2):
#                             axes[row, col_idx].set_visible(False)
#                         continue
#
#                     a = fit_row['a_semisum'].values[0]
#                     b = fit_row['b_semidiff'].values[0]
#                     c = fit_row['c_offset'].values[0]
#
#                     qsum = data['Q_sum_1'].values
#                     qdiff = data['Q_sum_2'].values
#                     t_true = data['delta_t_corrected'].values
#                     t_pred = a * qsum + b * qdiff + c
#                     residual = t_true - t_pred
#                     
#                     residual_range = slewing_residual_range
#                     cond = (residual > -1 * residual_range) & (residual < residual_range)
#                     residual = residual[cond]
#                     t_true = t_true[cond]
#                     t_pred = t_pred[cond]
#                     
#                     ax0 = axes[0, col_idx]
#                     ax0.scatter(t_true, t_pred, s=5, alpha=0.6)
#                     min_val = min(t_true.min(), t_pred.min())
#                     max_val = max(t_true.max(), t_pred.max())
#                     ax0.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
#                     ax0.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
#                     ax0.set_xlabel("T_true")
#                     ax0.set_ylabel("T_predicted")
#                     ax0.set_xlim([-t_comparison_lim, t_comparison_lim])
#                     ax0.set_ylim([-t_comparison_lim, t_comparison_lim])
#
#                     ax1 = axes[1, col_idx]
#                     ax1.hist(t_true, bins=100, alpha=0.7, color='tab:gray', label="Uncorrected")
#                     ax1.hist(residual, bins=100, alpha=0.7, color='green', label="Residuals, same as corrected")
#                     ax1.set_xlabel("Residuals (ns)")
#                     ax1.set_ylabel("Counts")
#                     ax1.set_title("Residual Distribution")
#                     ax1.set_xlim([-4, 4])
#
#                 for row in range(2):
#                     for col in range(len(current_pairs), batch_size):
#                         axes[row, col].set_visible(False)
#
#                 fig.tight_layout(rect=[0, 0, 1, 0.95])
#                 fig.suptitle(f"Batch {batch + 1}/{num_batches} — Fit Check (Predicted vs Real, Residuals)", fontsize=16)
#                 if save_plots:
#                     name_of_file = 'model_validation_simple'
#                     final_filename = f'{fig_idx}_{name_of_file}.png'
#                     fig_idx += 1
#
#                     save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#                     plot_list.append(save_fig_path)
#                     save_plot_figure(save_fig_path, format='png')
#
#                 if show_plots:
#                     plt.show()
#                 plt.close()

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: `_defer_tsum_calibration_until_late_slot` is hard-set to True; superseded early T_sum branch is unreachable
# ============================================================================
# if (not _defer_tsum_calibration_until_late_slot) and time_calibration_mode is not None:
#     if calculate_T_sum_calibration:
#         print("Using direct T_sum calibration from geometry-corrected pair residuals.")
#         strip_pair_patterns = [
#             (1, 1), (2, 2), (3, 3), (4, 4),
#             (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3),
#             (1, 3), (3, 1), (2, 4), (4, 2), (1, 4),
#         ]
#         plane_pair_order = [(1, 3), (1, 4), (2, 4), (3, 4), (1, 2), (2, 3)]
#         vector_names = [
#             f"P{plane_a}s{strip_a}_P{plane_b}s{strip_b}"
#             for plane_a, plane_b in plane_pair_order
#             for strip_a, strip_b in strip_pair_patterns
#         ]
#         vectors = []
#         rows = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
#         columns = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
#         df = pd.DataFrame(index=rows, columns=columns)
#         pair_offset_lookup.clear()
#         for plane_a, plane_b in plane_pair_order:
#             for strip_a, strip_b in strip_pair_patterns:
#                 label1 = f"P{plane_a}s{strip_a}"
#                 label2 = f"P{plane_b}s{strip_b}"
#                 pair_mask = (
#                     (calibration_time_pair_residuals_df["label1"] == label1)
#                     & (calibration_time_pair_residuals_df["label2"] == label2)
#                 )
#                 values = pd.to_numeric(
#                     calibration_time_pair_residuals_df.loc[pair_mask, "delta_t_corrected"],
#                     errors="coerce",
#                 ).dropna().to_numpy(dtype=float)
#                 vectors.append(values)
#                 if values.size == 0:
#                     continue
#                 pair_offset = summary(values)
#                 df.loc[label1, label2] = pair_offset
#                 df.loc[label2, label1] = -pair_offset
#                 pair_offset_lookup[(label1, label2)] = float(pair_offset)
#                 pair_offset_lookup[(label2, label1)] = float(-pair_offset)
#
#         print("----------------------------------------------------------------------")
#         print("--------------------- Time resolution calculation --------------------")
#         print("----------------------------------------------------------------------")
#         crt_values = {}
#         for var_name, vector in zip(vector_names, vectors):
#             vdat = np.asarray(vector, dtype=float)
#             if len(vdat) > 1:
#                 try:
#                     vdat = vdat[(vdat > np.quantile(vdat, CRT_gaussian_fit_quantile)) & (vdat < np.quantile(vdat, 1 - CRT_gaussian_fit_quantile))]
#                 except IndexError:
#                     print(f"IndexError encountered for {var_name}, setting CRT to 0")
#                     vdat = np.array([0])
#             CRT = norm.fit(vdat)[1] / np.sqrt(2) if len(vdat) > 0 else 0
#             crt_values[f'CRT_{var_name}'] = CRT
#         print("CRT values:", crt_values)
#         crt_array = np.array(list(crt_values.values()), dtype=float)
#         if crt_array.size > 0:
#             Q1, Q3 = np.percentile(crt_array, [25, 75])
#             crt_array = crt_array[crt_array <= 1]
#             filtered_crt_values = crt_array[(crt_array >= Q1 - 1.5 * (Q3 - Q1)) & (crt_array <= Q3 + 1.5 * (Q3 - Q1))]
#             global_variables['CRT_avg'] = np.mean(filtered_crt_values) * 1000 if filtered_crt_values.size else 0
#             global_variables['CRT_std'] = np.std(filtered_crt_values) * 1000 if filtered_crt_values.size else 0
#         else:
#             global_variables['CRT_avg'] = 0
#             global_variables['CRT_std'] = 0
#         print("---------------------------")
#         print(f"CRT Avg: {global_variables['CRT_avg']:.4g} ps")
#         print(f"CRT Std: {global_variables['CRT_std']:.4g} ps")
#         print("---------------------------")
#             
#         print("Antisymmetric matrix:")
#         print(df.map(lambda x: f"{x:.2f}" if pd.notnull(x) else ""))
#
#         
#         
#         
#         
#         # -----------------------------------------------------------------------------
#         # Selected paths method
#         # -----------------------------------------------------------------------------
#         try:
#             itineraries = load_itineraries_from_file(ITINERARY_FILE_PATH, echo_entries=False, header_suffix="")
#         except (FileNotFoundError, ValueError) as itinerary_error:
#             print(f"Task 2 itinerary loading failed: {itinerary_error}", force=True)
#             sys.exit(1)
#
#         def has_duplicate_sublists(lst):
#             seen = set()
#             for sub_list in lst:
#                 sub_list_tuple = tuple(sub_list)
#                 if sub_list_tuple in seen:
#                     return True
#                 seen.add(sub_list_tuple)
#             return False
#         
#         if has_duplicate_sublists(itineraries):
#             print("Duplicated itineraries.")
#         
#         selected_path_list = []
#         rows = ['P{}'.format(i) for i in range(1, 5)]
#         columns = ['s{}'.format(i) for i in range(1,5)]
#         selected_path_df = pd.DataFrame(0, index=rows, columns=columns)
#         
#         for itinerary in itineraries:
#             selected_path_df[selected_path_df.columns] = 0
#             step = itinerary
#             a = []
#             for i in range(len(step)):
#                 if i > 0:
#                     a.append( df[step[i - 1]][step[i]] )
#                 
#                 relative_time = sum(a)
#                 ind1 = str(step[i][0:2])
#                 ind2 = str(step[i][2:4])
#                 
#                 selected_path_df[ind2] = selected_path_df[ind2].astype(float)
#                 # selected_path_df.loc[ind1,ind2] = selected_path_df.loc[ind1,ind2] - relative_time
#                 selected_path_df.loc[ind1,ind2] = selected_path_df.loc[ind1,ind2] + relative_time # ORIGINALLY THERE WAS A MINUS BUT STOPPED WORKING SO I PUT THE + TO TRY
#             
#             selected_path_df = selected_path_df.sub(selected_path_df.iloc[0, 0])
#             selected_path_list.append(selected_path_df.values)
#             
#         # Calculate the mean of all the paths
#         # calibration_times = np.nanmean(selected_path_list, axis=0)
#         
#         # selected_path_list: shape (N_paths, N_points)
#         selected_path_array = np.array(selected_path_list, dtype=float)
#         if selected_path_array.size == 0:
#             calibration_times = np.zeros((4, 4), dtype=float)
#             print("Warning: No itinerary paths available; using zero T_sum calibration.")
#         else:
#             valid_counts = np.sum(np.isfinite(selected_path_array), axis=0)
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore", RuntimeWarning)
#                 median = np.nanmedian(selected_path_array, axis=0)
#             median = np.where(valid_counts > 0, median, 0.0)
#
#             abs_dev = np.abs(selected_path_array - median)
#             epsilon = 1e-6
#             weights = 1.0 / (abs_dev + epsilon)
#             weighted_sum = np.nansum(selected_path_array * weights, axis=0)
#             sum_of_weights = np.nansum(weights, axis=0)
#             calibration_times = np.divide(
#                 weighted_sum,
#                 sum_of_weights,
#                 out=np.full_like(weighted_sum, np.nan, dtype=float),
#                 where=sum_of_weights > 0,
#             )
#         
#         # Time calibration matrix calculated --------------------------------------
#         
#         print("------------------------")
#         print("Calculated calibration in times is:\n", calibration_times)
#         
#         nan_mask = np.isnan(calibration_times)
#         if np.any(nan_mask):
#             calibration_times[nan_mask] = 0.0
#             print("Some NaN values in calibration times were set to 0 (no correction applied for those strips).")
#
#         # Build Tsum_cal from freshly computed calibration_times
#         Tsum_cal = [list(calibration_times[i]) for i in range(len(calibration_times))]
#     # else (file mode): Tsum_cal is pre-populated from calibration CSV, nothing to do here
#
#     if apply_T_sum_calibration:
#         print(f"Final calibration in times used (mode={time_calibration_mode}):", _format_value_for_print(Tsum_cal))
#         print(
#             "Deferring calibration-subsample T_sum offset application until "
#             "after the Q-cut delta-T centroid diagnostic."
#         )
#
# else:
#     print(
#         "Deferring T_sum pair residual table construction and T_sum "
#         "calibration until the late charge-calibrated section."
#     )

