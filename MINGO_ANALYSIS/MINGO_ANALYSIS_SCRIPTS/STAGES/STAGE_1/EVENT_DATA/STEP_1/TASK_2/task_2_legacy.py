# LEGACY CODE ARCHIVE FOR TASK 2
#
# This file stores code deliberately removed from the active task script.
# It is not imported by the pipeline and is not expected to execute.
# Dependencies and surrounding runtime context may be missing.
# Removed blocks are preserved substantially verbatim for historical reference.

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: literal `if False:` branch; condition is unconditionally false
# ============================================================================
# if False:
#     print('Working in fast mode.')

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: literal `if False:` branch; condition is unconditionally false
# ============================================================================
# if False:
#     print('Working in debug mode.')

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: literal `if False:` branch; condition is unconditionally false
# ============================================================================
# if False:
#     T_F_left_pre_cal = T_side_left_pre_cal_debug
#     T_F_right_pre_cal = T_side_right_pre_cal_debug
#
#     T_B_left_pre_cal = T_side_left_pre_cal_debug
#     T_B_right_pre_cal = T_side_right_pre_cal_debug

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: literal `if False:` branch; condition is unconditionally false
# ============================================================================
# if False:
#     T_clip_min = T_clip_min_debug
#     T_clip_max = T_clip_max_debug
#     Q_clip_min = Q_clip_min_debug
#     Q_clip_max = Q_clip_max_debug
#     num_bins = num_bins_debug

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

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: `_need_slewing_fit` is hard-set to False; old regression/fit branch is unreachable
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
# REASON: hard-setting for removed unreachable slewing-fit branch
# ============================================================================
# _need_slewing_fit = False

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: hard-setting for removed superseded early T_sum branch
# ============================================================================
# _defer_tsum_calibration_until_late_slot = True

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: FINAL/report plot block moved to standalone final plotting (strip_combination_filter_by_tt)
# ============================================================================
#         if save_plots:
#             final_filename = f"{fig_idx_value}_strip_combination_filter_by_tt_{relation_label}_TT_{tt_label}.png"
#             fig_idx_value += 1
#             save_fig_path = os.path.join(base_dir, final_filename)
#             if plot_list is not None:
#                 plot_list.append(save_fig_path)
#             save_plot_figure(save_fig_path, fig=fig, alias="strip_combination_filter_by_tt", format="png", dpi=150)

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: FINAL/report plot block moved to standalone final plotting (strip_pair_matrix_by_planepair)
# ============================================================================
#                 if save_plots:
#                     file_name = (
#                         f"{figure_index:03d}_strip_pair_matrix_P{plane_i}P{plane_j}_"
#                         f"s{strip_i}_s{strip_j}.png"
#                     )
#                     figure_index += 1
#                     save_path = os.path.join(base_directories["figure_directory"], file_name)
#                     plot_list.append(save_path)
#                     save_plot_figure(
#                         save_path,
#                         fig=fig,
#                         alias="strip_pair_matrix_by_planepair",
#                         format="png",
#                         dpi=strip_pair_fig_dpi,
#                         bbox_inches="tight",
#                     )

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: FINAL/report plot block moved to standalone final plotting (strip_combination_filter_by_tt)
# ============================================================================
# if _plot_strip_combination_filter_by_tt:
#     strip_combination_self_hist_after = collect_task2_strip_combination_histogram_payload(
#         working_df,
#         strip_combination_tt_original,
#         relation_type="strip",
#     )
#     strip_combination_same_plane_hist_after = collect_task2_strip_combination_histogram_payload(
#         working_df,
#         strip_combination_tt_original,
#         relation_type="plane",
#     )
#     strip_combination_any_hist_after = collect_task2_strip_combination_histogram_payload(
#         working_df,
#         strip_combination_tt_original,
#         relation_type="detector",
#     )
#     strip_combination_self_limits = {
#         "q_sum_sum": (
#             task2_strip_combination_limits_by_relation["same_strip"]["q_sum_sum_left"],
#             task2_strip_combination_limits_by_relation["same_strip"]["q_sum_sum_right"],
#         ),
#         "q_dif_sum": (
#             -abs(task2_strip_combination_limits_by_relation["same_strip"]["q_dif_sum_threshold"]),
#             abs(task2_strip_combination_limits_by_relation["same_strip"]["q_dif_sum_threshold"]),
#         ),
#         "t_sum_sum": (
#             task2_strip_combination_limits_by_relation["same_strip"]["t_sum_sum_left"],
#             task2_strip_combination_limits_by_relation["same_strip"]["t_sum_sum_right"],
#         ),
#         "t_dif_sum": (
#             -abs(task2_strip_combination_limits_by_relation["same_strip"]["t_dif_sum_threshold"]),
#             abs(task2_strip_combination_limits_by_relation["same_strip"]["t_dif_sum_threshold"]),
#         ),
#     }
#     strip_combination_same_plane_limits = {
#         "q_sum_sum": (
#             task2_strip_combination_limits_by_relation["same_plane"]["q_sum_sum_left"],
#             task2_strip_combination_limits_by_relation["same_plane"]["q_sum_sum_right"],
#         ),
#         "q_sum_dif": (
#             -abs(task2_strip_combination_limits_by_relation["same_plane"]["q_sum_dif_threshold"]),
#             abs(task2_strip_combination_limits_by_relation["same_plane"]["q_sum_dif_threshold"]),
#         ),
#         "q_dif_sum": (
#             -abs(task2_strip_combination_limits_by_relation["same_plane"]["q_dif_sum_threshold"]),
#             abs(task2_strip_combination_limits_by_relation["same_plane"]["q_dif_sum_threshold"]),
#         ),
#         "q_dif_dif": (
#             -abs(task2_strip_combination_limits_by_relation["same_plane"]["q_dif_dif_threshold"]),
#             abs(task2_strip_combination_limits_by_relation["same_plane"]["q_dif_dif_threshold"]),
#         ),
#         "t_sum_sum": (
#             task2_strip_combination_limits_by_relation["same_plane"]["t_sum_sum_left"],
#             task2_strip_combination_limits_by_relation["same_plane"]["t_sum_sum_right"],
#         ),
#         "t_sum_dif": (
#             -abs(task2_strip_combination_limits_by_relation["same_plane"]["t_sum_dif_threshold"]),
#             abs(task2_strip_combination_limits_by_relation["same_plane"]["t_sum_dif_threshold"]),
#         ),
#         "t_dif_sum": (
#             -abs(task2_strip_combination_limits_by_relation["same_plane"]["t_dif_sum_threshold"]),
#             abs(task2_strip_combination_limits_by_relation["same_plane"]["t_dif_sum_threshold"]),
#         ),
#         "t_dif_dif": (
#             -abs(task2_strip_combination_limits_by_relation["same_plane"]["t_dif_dif_threshold"]),
#             abs(task2_strip_combination_limits_by_relation["same_plane"]["t_dif_dif_threshold"]),
#         ),
#     }
#     strip_combination_any_limits = {
#         "q_sum_sum": (strip_combination_q_sum_sum_left, strip_combination_q_sum_sum_right),
#         "q_sum_dif": (-abs(strip_combination_q_sum_dif_threshold), abs(strip_combination_q_sum_dif_threshold)),
#         "q_dif_sum": (-abs(strip_combination_q_dif_sum_threshold), abs(strip_combination_q_dif_sum_threshold)),
#         "q_dif_dif": (-abs(strip_combination_q_dif_dif_threshold), abs(strip_combination_q_dif_dif_threshold)),
#         "t_sum_sum": (strip_combination_t_sum_sum_left, strip_combination_t_sum_sum_right),
#         "t_sum_dif": (-abs(strip_combination_t_sum_dif_threshold), abs(strip_combination_t_sum_dif_threshold)),
#         "t_dif_sum": (-abs(strip_combination_t_dif_sum_threshold), abs(strip_combination_t_dif_sum_threshold)),
#         "t_dif_dif": (-abs(strip_combination_t_dif_dif_threshold), abs(strip_combination_t_dif_dif_threshold)),
#     }
#     fig_idx = plot_task2_strip_combination_filter_by_tt(
#         strip_combination_self_hist_original,
#         strip_combination_self_hist_original,
#         strip_combination_self_hist_after,
#         basename_no_ext,
#         fig_idx,
#         base_directories["figure_directory"],
#         show_plots=show_plots,
#         save_plots=save_plots,
#         plot_list=plot_list,
#         limits_by_observable=strip_combination_self_limits,
#         relation_label="strip",
#         observable_definitions=TASK2_STRIP_COMBINATION_SELF_OBSERVABLES,
#     )
#     fig_idx = plot_task2_strip_combination_filter_by_tt(
#         strip_combination_same_plane_hist_original,
#         strip_combination_same_plane_hist_original,
#         strip_combination_same_plane_hist_after,
#         basename_no_ext,
#         fig_idx,
#         base_directories["figure_directory"],
#         show_plots=show_plots,
#         save_plots=save_plots,
#         plot_list=plot_list,
#         limits_by_observable=strip_combination_same_plane_limits,
#         relation_label="plane",
#     )
#     fig_idx = plot_task2_strip_combination_filter_by_tt(
#         strip_combination_any_hist_original,
#         strip_combination_any_hist_original,
#         strip_combination_any_hist_after,
#         basename_no_ext,
#         fig_idx,
#         base_directories["figure_directory"],
#         show_plots=show_plots,
#         save_plots=save_plots,
#         plot_list=plot_list,
#         limits_by_observable=strip_combination_any_limits,
#         relation_label="detector",
#     )

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: FINAL/report plot block moved to standalone final plotting (channel_matrix_tq, charge_front_vs_back, rejected_histograms_q_sum_zero, rejected_histograms_tt_task1_clean, rejected_histograms_tt_task2_cal, rejected_scatter_q_sum_zero, rejected_scatter_tt_task1_clean, rejected_scatter_tt_task2_cal, strip_pair_matrix_by_planepair, time_front_vs_back)
# ============================================================================
# if any(_diag_requested[a] for a in _DIAG_ALIASES):
#     _DIAG_MAX_PTS = 5000
#     _diag_q_sum_cols = sorted(
#         c for c in unzeroed_plot_snapshot_df.columns if re.match(r"^Q\d+_Q_sum_\d+$", c)
#     )
#     _diag_t_sum_cols = sorted(
#         c for c in unzeroed_plot_snapshot_df.columns if re.match(r"^T\d+_T_sum_\d+$", c)
#     )
#
#     def _diag_total_charge(df: pd.DataFrame) -> pd.Series:
#         cols = [c for c in _diag_q_sum_cols if c in df.columns]
#         return df[cols].fillna(0).sum(axis=1) if cols else pd.Series(0.0, index=df.index)
#
#     def _diag_mean_tsum(df: pd.DataFrame) -> pd.Series:
#         cols = [c for c in _diag_t_sum_cols if c in df.columns]
#         if not cols:
#             return pd.Series(np.nan, index=df.index)
#         return df[cols].replace(0, np.nan).mean(axis=1)
#
#     def _diag_sample(df: pd.DataFrame) -> pd.DataFrame:
#         return df.sample(n=_DIAG_MAX_PTS, random_state=42) if len(df) > _DIAG_MAX_PTS else df
#
#     # ── Plot A: rejected-event scatter (mean T_sum vs total Q_sum) ────────────
#     _rej_groups = [
#         ("rejected_scatter_tt_task1_clean",  _diag_rejected_tt_task1_clean_idx, "tt_task1_clean NaN"),
#         ("rejected_scatter_q_sum_zero", _diag_rejected_q_sum_idx,   "Q_sum all-zero"),
#         ("rejected_scatter_tt_task2_cal",    _diag_rejected_tt_task2_cal_idx,   "tt_task2_cal < 10"),
#     ]
#     for _alias, _rej_idx, _reason in _rej_groups:
#         if not _diag_requested.get(_alias, False):
#             continue
#         _rej_sub = _diag_sample(unzeroed_plot_snapshot_df.loc[_rej_idx]) if len(_rej_idx) else pd.DataFrame(columns=unzeroed_plot_snapshot_df.columns)
#         _acc_sub = _diag_sample(unzeroed_plot_snapshot_df.drop(index=_rej_idx, errors="ignore"))
#         _fig, _ax = plt.subplots(figsize=(8, 6))
#         _ax.scatter(
#             _diag_total_charge(_acc_sub), _diag_mean_tsum(_acc_sub),
#             s=3, alpha=0.15, color="steelblue", rasterized=True,
#             label=f"Accepted ({len(unzeroed_plot_snapshot_df) - len(_rej_idx)})",
#         )
#         if not _rej_sub.empty:
#             _ax.scatter(
#                 _diag_total_charge(_rej_sub), _diag_mean_tsum(_rej_sub),
#                 s=4, alpha=0.4, color="tomato", rasterized=True,
#                 label=f"Rejected – {_reason} ({len(_rej_idx)})",
#             )
#         _ax.set_xlabel("Total Q_sum [ADC]")
#         _ax.set_ylabel("Mean T_sum [ADC]")
#         _ax.set_title(
#             f"Rejection diagnostic: {_reason}\n"
#             f"Station {station} · {basename_no_ext}"
#         )
#         _ax.legend(markerscale=3, fontsize=8)
#         plt.tight_layout()
#         _sp = os.path.join(base_directories["figure_directory"], f"{fig_idx:03d}_{_alias}.png")
#         plot_list.append(_sp)
#         save_plot_figure(_sp, fig=_fig, alias=_alias, format="png", dpi=150)
#         plt.close(_fig)
#         fig_idx += 1
#
#     # ── Plot B: front vs back (Q and T) ──────────────────────────────────────
#     _fb_pattern_q = re.compile(r"^Q(\d+)_F_(\d+)$")
#     _fb_raw_q_f = {
#         (int(m.group(1)), int(m.group(2))): c
#         for c in unzeroed_plot_snapshot_df.columns
#         if (m := _fb_pattern_q.match(c))
#     }
#     _fb_raw_q_b = {
#         (int(m.group(1)), int(m.group(2))): c
#         for c in unzeroed_plot_snapshot_df.columns
#         if (m := re.match(r"^Q(\d+)_B_(\d+)$", c))
#     }
#     _fb_pattern_t = re.compile(r"^T(\d+)_F_(\d+)$")
#     _fb_raw_t_f = {
#         (int(m.group(1)), int(m.group(2))): c
#         for c in unzeroed_plot_snapshot_df.columns
#         if (m := _fb_pattern_t.match(c))
#     }
#     _fb_raw_t_b = {
#         (int(m.group(1)), int(m.group(2))): c
#         for c in unzeroed_plot_snapshot_df.columns
#         if (m := re.match(r"^T(\d+)_B_(\d+)$", c))
#     }
#
#     for _fb_alias, _f_map, _b_map, _sum_cols, _dif_cols, _label, _unit in [
#         (
#             "charge_front_vs_back", _fb_raw_q_f, _fb_raw_q_b,
#             _diag_q_sum_cols,
#             sorted(c for c in unzeroed_plot_snapshot_df.columns if re.match(r"^Q\d+_Q_dif_\d+$", c)),
#             "Q", "ADC",
#         ),
#         (
#             "time_front_vs_back", _fb_raw_t_f, _fb_raw_t_b,
#             _diag_t_sum_cols,
#             sorted(c for c in unzeroed_plot_snapshot_df.columns if re.match(r"^T\d+_T_dif_\d+$", c)),
#             "T", "ADC",
#         ),
#     ]:
#         if not _diag_requested.get(_fb_alias, False):
#             continue
#         _matched_keys = sorted(set(_f_map) & set(_b_map))
#         _use_raw = bool(_matched_keys)
#         if not _use_raw:
#             # Reconstruct F = (sum+dif)/2, B = (sum-dif)/2 from sum/dif pairs
#             _dif_set = set(_dif_cols)
#             _matched_keys = []
#             _sum_to_dif: dict[str, str] = {}
#             for _sc in _sum_cols:
#                 _dc = _sc.replace("_Q_sum_", "_Q_dif_").replace("_T_sum_", "_T_dif_")
#                 if _dc in _dif_set:
#                     _matched_keys.append(_sc)
#                     _sum_to_dif[_sc] = _dc
#         if not _matched_keys:
#             continue
#         _df_fb = _diag_sample(unzeroed_plot_snapshot_df)
#         n_pairs = len(_matched_keys)
#         n_cols_fb = min(4, n_pairs)
#         n_rows_fb = (n_pairs + n_cols_fb - 1) // n_cols_fb
#         _fig, _axes = plt.subplots(
#             n_rows_fb, n_cols_fb,
#             figsize=(4 * n_cols_fb, 3.5 * n_rows_fb),
#             squeeze=False,
#         )
#         for _pi, _key in enumerate(_matched_keys):
#             _r, _c = divmod(_pi, n_cols_fb)
#             _ax = _axes[_r][_c]
#             if _use_raw:
#                 _xa = _df_fb[_f_map[_key]].fillna(0)
#                 _xb = _df_fb[_b_map[_key]].fillna(0)
#                 _xl = _f_map[_key].replace("_", " ")
#                 _yl = _b_map[_key].replace("_", " ")
#             else:
#                 _s = _df_fb[_key].fillna(0)
#                 _d = _df_fb[_sum_to_dif[_key]].fillna(0)
#                 _xa = (_s + _d) / 2.0
#                 _xb = (_s - _d) / 2.0
#                 _base = _key.replace("_Q_sum_", " ").replace("_T_sum_", " ").replace("_", " ")
#                 _xl = f"{_base} F"
#                 _yl = f"{_base} B"
#             _ax.scatter(_xa, _xb, s=2, alpha=0.15, rasterized=True)
#             _ax.set_xlabel(_xl, fontsize=7)
#             _ax.set_ylabel(_yl, fontsize=7)
#             _ax.tick_params(labelsize=6)
#         for _pi in range(n_pairs, n_rows_fb * n_cols_fb):
#             _r, _c = divmod(_pi, n_cols_fb)
#             _axes[_r][_c].set_visible(False)
#         _fig.suptitle(
#             f"{_label}: Front vs Back (unzeroed, N≤{_DIAG_MAX_PTS})\n"
#             f"Station {station} · {basename_no_ext}",
#             fontsize=9,
#         )
#         plt.tight_layout()
#         _sp = os.path.join(base_directories["figure_directory"], f"{fig_idx:03d}_{_fb_alias}.png")
#         plot_list.append(_sp)
#         save_plot_figure(_sp, fig=_fig, alias=_fb_alias, format="png", dpi=150)
#         plt.close(_fig)
#         fig_idx += 1
#
#     # ── Plot C: channel matrix (Q_sum per-strip, per plane pair) ─────────────
#     if _diag_requested.get("channel_matrix_tq", False):
#         _df_cm = _diag_sample(unzeroed_plot_snapshot_df)
#         for _pi_idx in range(1, 5):
#             for _pj_idx in range(_pi_idx + 1, 5):
#                 _ci_cols = [
#                     f"p{_pi_idx}_s{s}_qsum" for s in range(1, 5)
#                     if f"p{_pi_idx}_s{s}_qsum" in _df_cm.columns
#                 ]
#                 _cj_cols = [
#                     f"p{_pj_idx}_s{s}_qsum" for s in range(1, 5)
#                     if f"p{_pj_idx}_s{s}_qsum" in _df_cm.columns
#                 ]
#                 if not _ci_cols or not _cj_cols:
#                     continue
#                 _ni, _nj = len(_ci_cols), len(_cj_cols)
#                 _fig, _axes = plt.subplots(
#                     _nj, _ni,
#                     figsize=(3 * _ni, 3 * _nj),
#                     squeeze=False,
#                 )
#                 for _si, _ci in enumerate(_ci_cols):
#                     for _sj, _cj in enumerate(_cj_cols):
#                         _ax = _axes[_sj][_si]
#                         _both = (_df_cm[_ci] != 0) & (_df_cm[_cj] != 0)
#                         _ax.scatter(
#                             _df_cm.loc[_both, _ci], _df_cm.loc[_both, _cj],
#                             s=2, alpha=0.2, rasterized=True,
#                         )
#                         _ax.set_xlabel(_ci.replace("_", " "), fontsize=6)
#                         _ax.set_ylabel(_cj.replace("_", " "), fontsize=6)
#                         _ax.tick_params(labelsize=5)
#                 _combo = f"P{_pi_idx}_P{_pj_idx}"
#                 _fig.suptitle(
#                     f"Channel matrix Q_sum: Plane {_pi_idx} vs Plane {_pj_idx}\n"
#                     f"(unzeroed, co-fired strips, N≤{_DIAG_MAX_PTS})\n"
#                     f"Station {station} · {basename_no_ext}",
#                     fontsize=8,
#                 )
#                 plt.tight_layout()
#                 _sp = os.path.join(
#                     base_directories["figure_directory"],
#                     f"{fig_idx:03d}_channel_matrix_tq_{_combo}.png",
#                 )
#                 plot_list.append(_sp)
#                 save_plot_figure(_sp, fig=_fig, alias="channel_matrix_tq", format="png", dpi=120)
#                 plt.close(_fig)
#                 fig_idx += 1
#
#     if _diag_requested.get("strip_pair_matrix_by_planepair", False):
#         if not track_removed_rows_task2:
#             print("Task 2 strip-pair removed-row overlay disabled (track_removed_rows_task2=False).")
#         removed_plot_df = diagnostic_removed_rows_df if track_removed_rows_task2 else diagnostic_removed_rows_df.iloc[0:0].copy()
#         fig_idx = plot_task2_strip_pair_matrix_by_planepair(
#             working_df,
#             removed_plot_df,
#             basename_no_ext,
#             fig_idx,
#         )
#
#     # ── Plot D: rejected-event per-plane histograms (Q_sum and T_sum) ────────
#     _hist_rej_groups = [
#         ("rejected_histograms_tt_task1_clean",  _diag_rejected_tt_task1_clean_idx,  "tt_task1_clean NaN"),
#         ("rejected_histograms_q_sum_zero", _diag_rejected_q_sum_idx,    "Q_sum all-zero"),
#         ("rejected_histograms_tt_task2_cal",    _diag_rejected_tt_task2_cal_idx,    "tt_task2_cal < 10"),
#     ]
#     for _halias, _rej_idx, _reason in _hist_rej_groups:
#         if not _diag_requested.get(_halias, False):
#             continue
#         _rej_rows = unzeroed_plot_snapshot_df.loc[_rej_idx] if len(_rej_idx) else pd.DataFrame(columns=unzeroed_plot_snapshot_df.columns)
#         _acc_rows = unzeroed_plot_snapshot_df.drop(index=_rej_idx, errors="ignore")
#         _fig, _axes = plt.subplots(4, 2, figsize=(10, 12))
#         for _p in range(1, 5):
#             _q_cols = [f"p{_p}_s{s}_qsum" for s in range(1, 5) if f"p{_p}_s{s}_qsum" in unzeroed_plot_snapshot_df.columns]
#             _t_cols = [f"p{_p}_s{s}_tsum" for s in range(1, 5) if f"p{_p}_s{s}_tsum" in unzeroed_plot_snapshot_df.columns]
#             _ax_q = _axes[_p - 1][0]
#             _ax_t = _axes[_p - 1][1]
#             if _q_cols:
#                 _acc_q = _acc_rows[_q_cols].fillna(0).sum(axis=1)
#                 _rej_q = _rej_rows[_q_cols].fillna(0).sum(axis=1) if not _rej_rows.empty else pd.Series(dtype=float)
#                 _q_all = pd.concat([_acc_q, _rej_q])
#                 _qmin, _qmax = float(_q_all.min()), float(_q_all.max())
#                 _bins_q = np.linspace(_qmin, _qmax, 50) if _qmin < _qmax else 10
#                 _ax_q.hist(_acc_q, bins=_bins_q, alpha=0.6, color="steelblue",
#                            label=f"Accepted (N={len(_acc_q)})", histtype="stepfilled")
#                 if len(_rej_q) > 0:
#                     _ax_q.hist(_rej_q, bins=_bins_q, alpha=0.85, color="tomato",
#                                label=f"Rejected (N={len(_rej_q)})", histtype="step", linewidth=1.5)
#             _ax_q.set_xlabel(f"P{_p} Q_sum total [ADC]", fontsize=8)
#             _ax_q.set_ylabel("Counts", fontsize=8)
#             _ax_q.set_yscale("log", nonpositive="clip")
#             _ax_q.legend(fontsize=6)
#             _ax_q.tick_params(labelsize=7)
#             if _t_cols:
#                 _acc_t = _acc_rows[_t_cols].replace(0, np.nan).mean(axis=1).dropna()
#                 _rej_t = _rej_rows[_t_cols].replace(0, np.nan).mean(axis=1).dropna() if not _rej_rows.empty else pd.Series(dtype=float)
#                 _t_all = pd.concat([_acc_t, _rej_t])
#                 _tmin, _tmax = float(_t_all.min()), float(_t_all.max())
#                 _bins_t = np.linspace(_tmin, _tmax, 50) if _tmin < _tmax else 10
#                 _ax_t.hist(_acc_t, bins=_bins_t, alpha=0.6, color="steelblue",
#                            label=f"Accepted (N={len(_acc_t)})", histtype="stepfilled")
#                 if len(_rej_t) > 0:
#                     _ax_t.hist(_rej_t, bins=_bins_t, alpha=0.85, color="tomato",
#                                label=f"Rejected (N={len(_rej_t)})", histtype="step", linewidth=1.5)
#             _ax_t.set_xlabel(f"P{_p} T_sum mean [ADC]", fontsize=8)
#             _ax_t.set_ylabel("Counts", fontsize=8)
#             _ax_t.set_yscale("log", nonpositive="clip")
#             _ax_t.legend(fontsize=6)
#             _ax_t.tick_params(labelsize=7)
#         _fig.suptitle(
#             f"Rejected event shape – {_reason}\nStation {station} · {basename_no_ext}",
#             fontsize=10,
#         )
#         plt.tight_layout()
#         _sp = os.path.join(base_directories["figure_directory"], f"{fig_idx:03d}_{_halias}.png")
#         plot_list.append(_sp)
#         save_plot_figure(_sp, fig=_fig, alias=_halias, format="png", dpi=150)
#         plt.close(_fig)
#         fig_idx += 1

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: FINAL/report plot block moved to standalone final plotting (calibrated_filtered_removed_zeroes)
# ============================================================================
# if task2_plot_requested("calibrated_filtered_removed_zeroes", essential=True):
#     _plot_task2_calibrated_filtered_removed_zeroes(
#         working_df,
#         title="Calibrated filtered data (removed zeroes)",
#         filename_stem="calibrated_filtered_removed_zeroes",
#         alias="calibrated_filtered_removed_zeroes",
#     )

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: FINAL/report plot block moved to standalone final plotting (calibrated_filtered_removed_zeroes_per_plane_combination)
# ============================================================================
# if task2_plot_requested("calibrated_filtered_removed_zeroes_per_plane_combination", essential=True):
#     if "tt_task2_cal" not in working_df.columns:
#         print("Skipping calibrated_filtered_removed_zeroes_per_plane_combination: tt_task2_cal column is missing.")
#     else:
#         tt_values = pd.to_numeric(working_df["tt_task2_cal"], errors="coerce")
#         generated_per_tt_figures = 0
#         for tt_value in sorted(tt_values[np.isfinite(tt_values)].unique()):
#             tt_mask = tt_values.eq(tt_value)
#             if not bool(tt_mask.any()):
#                 continue
#             tt_label = normalize_tt_label(tt_value)
#             _plot_task2_calibrated_filtered_removed_zeroes(
#                 working_df.loc[tt_mask],
#                 title=f"Calibrated filtered data (removed zeroes), tt_task2_cal={tt_label}",
#                 filename_stem=f"calibrated_filtered_removed_zeroes_per_plane_combination_tt_{tt_label}",
#                 alias="calibrated_filtered_removed_zeroes_per_plane_combination",
#             )
#             generated_per_tt_figures += 1
#         print(
#             "calibrated_filtered_removed_zeroes_per_plane_combination: "
#             f"generated {generated_per_tt_figures} figures."
#         )

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: FINAL/report plot block moved to standalone final plotting (time_calibrated_filtered_removed_zeroes)
# ============================================================================
# if task2_plot_requested("time_calibrated_filtered_removed_zeroes", essential=True):
#     time_plot_columns = [
#         col
#         for col in working_df.columns
#         if "T_sum" in col or "T_dif" in col
#     ]
#     if time_plot_columns:
#         plot_df = working_df.loc[:, time_plot_columns]
#         cols_per_row = 8
#         num_columns = len(time_plot_columns)
#         num_rows = max(1, (num_columns + cols_per_row - 1) // cols_per_row)
#         fig, axes = plt.subplots(
#             num_rows,
#             cols_per_row,
#             figsize=(20, num_rows * 2),
#             sharex="col",
#         )
#         axes = np.atleast_1d(axes).flatten()
#
#         for i, col in enumerate(time_plot_columns):
#             y = pd.to_numeric(plot_df[col], errors="coerce")
#             color = T_sum_color if "T_sum" in col else T_dif_color
#             y_nonzero = y[(y != 0) & np.isfinite(y)]
#             axes[i].hist(y_nonzero, bins=100, alpha=0.6, color=color)
#             axes[i].set_title(col)
#
#         for j in range(i + 1, len(axes)):
#             fig.delaxes(axes[j])
#
#         fig.tight_layout(rect=[0, 0, 1, 0.95])
#         fig.suptitle("Time-calibrated filtered data (removed zeroes)", fontsize=20)
#
#         if save_plots:
#             final_filename = f"{fig_idx}_time_calibrated_filtered_removed_zeroes.png"
#             fig_idx += 1
#             save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#             plot_list.append(save_fig_path)
#             save_plot_figure(
#                 save_fig_path,
#                 fig=fig,
#                 alias="time_calibrated_filtered_removed_zeroes",
#                 format="png",
#             )
#         if show_plots:
#             plt.show()
#         plt.close(fig)
#         del plot_df

# ============================================================================
# ARCHIVED FROM: script_2_clean_to_cal.py
# REASON: FINAL/report plot block moved to standalone final plotting (acquisition_rate_vs_time_by_task_tt_with_histograms)
# ============================================================================
# if save_plots and task2_plot_enabled("acquisition_rate_vs_time_by_task_tt_with_histograms"):
#     rate_fig = create_rate_vs_time_by_task_tt_with_histograms(
#         working_df,
#         tt_column="tt_task2_cal",
#         title=(
#             f"Task 2 acquisition rate by tt_task2_cal, {basename_no_ext} "
#             f"(files processed={len(joined_input_records)})"
#         ),
#         accumulation_window_seconds=config.get("acquisition_rate_accumulation_window_seconds", 60),
#         rate_histogram_bins=config.get("acquisition_rate_task_tt_histogram_bins", 80),
#         y_limit_left=config.get("acquisition_rate_task_tt_ylim_left", 0),
#         y_limit_right=config.get("acquisition_rate_task_tt_ylim_right", 4),
#     )
#     if rate_fig is not None:
#         final_filename = f"{fig_idx}_acquisition_rate_vs_time_by_task_tt_with_histograms.png"
#         fig_idx += 1
#         save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#         plot_list.append(save_fig_path)
#         save_plot_figure(
#             save_fig_path,
#             fig=rate_fig,
#             alias="acquisition_rate_vs_time_by_task_tt_with_histograms",
#             dpi=140,
#         )
#         plt.close(rate_fig)
#     else:
#         print("Task 2 acquisition-rate-by-task-tt plot skipped: no valid tt_task2_cal/datetime rows.")

