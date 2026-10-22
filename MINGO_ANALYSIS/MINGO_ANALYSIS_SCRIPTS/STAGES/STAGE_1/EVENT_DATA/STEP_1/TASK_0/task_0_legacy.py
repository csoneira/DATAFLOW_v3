# LEGACY CODE ARCHIVE FOR TASK 0
# This file is not imported and is not expected to execute.

# ============================================================================
# ARCHIVED FROM: script_0_acq_to_raw.py
# REASON: FINAL/report plot block moved to standalone final plotting (acquisition_rate_vs_time_by_trigger_type)
# ============================================================================
# if save_plots and task0_plot_enabled("acquisition_rate_vs_time_by_trigger_type"):
#     figure_directory = (
#         task_directory
#         / "PLOTS"
#         / "FIGURE_DIRECTORY"
#         / f"FIGURES_EXEC_ON_{date_execution}"
#     )
#     plot_path = figure_directory / f"acquisition_rate_vs_time_by_trigger_type_{basename_no_ext}.png"
#     plotted = plot_acquisition_rate_vs_time_by_trigger_type(
#         read_df,
#         plot_path,
#         title=f"Task 0 acquisition rate by trigger type, {basename_no_ext}",
#         accumulation_window_seconds=acquisition_rate_accumulation_window_seconds,
#     )
#     if plotted:
#         print(f"Task 0 acquisition-rate plot saved: {plot_path}", force=True)
#         saved_plot_paths.append(plot_path)
#     else:
#         print("Task 0 acquisition-rate plot skipped: no valid datetime rows.", force=True)

# ============================================================================
# ARCHIVED FROM: script_0_acq_to_raw.py
# REASON: FINAL/report plot block moved to standalone final plotting (acquisition_rate_vs_time_by_task_tt_with_histograms)
# ============================================================================
# if save_plots and task0_plot_enabled("acquisition_rate_vs_time_by_task_tt_with_histograms"):
#     if figure_directory is None:
#         figure_directory = (
#             task_directory
#             / "PLOTS"
#             / "FIGURE_DIRECTORY"
#             / f"FIGURES_EXEC_ON_{date_execution}"
#         )
#     plot_path = figure_directory / f"acquisition_rate_vs_time_by_task_tt_with_histograms_{basename_no_ext}.png"
#     plotted = plot_acquisition_rate_vs_time_by_task_tt_with_histograms(
#         read_df,
#         plot_path,
#         title=f"Task 0 acquisition rate by tt_task0_acq, {basename_no_ext} (files processed=1)",
#         tt_column="tt_task0_acq",
#         accumulation_window_seconds=acquisition_rate_accumulation_window_seconds,
#         rate_histogram_bins=acquisition_rate_task_tt_histogram_bins,
#         y_limit_left=config.get("acquisition_rate_task_tt_ylim_left", 0),
#         y_limit_right=config.get("acquisition_rate_task_tt_ylim_right", 4),
#     )
#     if plotted:
#         print(f"Task 0 acquisition-rate-by-task-tt plot saved: {plot_path}", force=True)
#         saved_plot_paths.append(plot_path)
#     else:
#         print("Task 0 acquisition-rate-by-task-tt plot skipped: no valid tt_task0_acq/datetime rows.", force=True)

