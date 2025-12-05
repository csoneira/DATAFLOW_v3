
#%%


# TASK 1 --> channel counts
# TASK 2 --> 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from qa_shared import load_metadata, print_columns, plot_tt_pairs, plot_tt_matrix

# --- knobs to tweak ---
STATION = "MINGO04"  # e.g. MINGO01, MINGO02, ...
STEP = 1             # numeric step (1, 2, ...)
TASK = 3          # for STEP_1 use an int (1-5); keep None for steps without tasks
START_DATE = "2024-03-01 00:00:00"    # e.g. "2025-11-06 18:00:00" or leave None
END_DATE = "2025-11-20 00:00:00"      # e.g. "2025-11-06 19:00:00" or leave None
# Window used when counting events (ns) and per-combination measured counts.
# Set WINDOW_NS to the calibration window you used (e.g., coincidence_window_cal_ns),
# and fill MEASURED_COUNTS with {combo: observed_counts}.
WINDOW_NS = None
MEASURED_COUNTS = {
    # 12: 0,
    # 123: 0,
}


ctx = load_metadata(STATION, STEP, TASK, START_DATE, END_DATE)
df = ctx.df
plotted_cols: set[str] = set()

print(f"Loaded: {ctx.metadata_path}")
print(f"Rows: {len(df)}")


# Print all column names
print("Columns:")
for col in df.columns:
    print(f" - {col}")



#%%



'''
RPC variable count before and after:
- P1_T_sum_nonzero_before_filter6
- P1_T_diff_nonzero_before_filter6
- P1_Q_sum_nonzero_before_filter6
- P1_Q_diff_nonzero_before_filter6
- P1_Y_nonzero_before_filter6
- P2_T_sum_nonzero_before_filter6
- P2_T_diff_nonzero_before_filter6
- P2_Q_sum_nonzero_before_filter6
- P2_Q_diff_nonzero_before_filter6
- P2_Y_nonzero_before_filter6
- P3_T_sum_nonzero_before_filter6
- P3_T_diff_nonzero_before_filter6
- P3_Q_sum_nonzero_before_filter6
- P3_Q_diff_nonzero_before_filter6
- P3_Y_nonzero_before_filter6
- P4_T_sum_nonzero_before_filter6
- P4_T_diff_nonzero_before_filter6
- P4_Q_sum_nonzero_before_filter6
- P4_Q_diff_nonzero_before_filter6
- P4_Y_nonzero_before_filter6
- P1_T_sum_nonzero_after_filter6
- P1_T_diff_nonzero_after_filter6
- P1_Q_sum_nonzero_after_filter6
- P1_Q_diff_nonzero_after_filter6
- P1_Y_nonzero_after_filter6
- P2_T_sum_nonzero_after_filter6
- P2_T_diff_nonzero_after_filter6
- P2_Q_sum_nonzero_after_filter6
- P2_Q_diff_nonzero_after_filter6
- P2_Y_nonzero_after_filter6
- P3_T_sum_nonzero_after_filter6
- P3_T_diff_nonzero_after_filter6
- P3_Q_sum_nonzero_after_filter6
- P3_Q_diff_nonzero_after_filter6
- P3_Y_nonzero_after_filter6
- P4_T_sum_nonzero_after_filter6
- P4_T_diff_nonzero_after_filter6
- P4_Q_sum_nonzero_after_filter6
- P4_Q_diff_nonzero_after_filter6
- P4_Y_nonzero_after_filter6


Trigger type before and after:
- cal_tt_0_count
- list_tt_0_count
- cal_to_list_tt_0_0_count
- cal_tt_123_count
- cal_tt_12_count
- cal_tt_34_count
- cal_tt_1234_count
- cal_tt_234_count
- cal_tt_13_count
- cal_tt_134_count
- cal_tt_1_count
- cal_tt_23_count
- cal_tt_3_count
- cal_tt_124_count
- cal_tt_4_count
- cal_tt_2_count
- cal_tt_14_count
- cal_tt_24_count
- list_tt_123_count
- list_tt_12_count
- list_tt_34_count
- list_tt_1234_count
- list_tt_234_count
- list_tt_13_count
- list_tt_1_count
- list_tt_23_count
- list_tt_134_count
- list_tt_3_count
- list_tt_124_count
- list_tt_4_count
- list_tt_2_count
- list_tt_14_count
- list_tt_24_count
- cal_to_list_tt_123_123_count
- cal_to_list_tt_12_12_count
- cal_to_list_tt_34_34_count
- cal_to_list_tt_1234_1234_count
- cal_to_list_tt_234_234_count
- cal_to_list_tt_13_13_count
- cal_to_list_tt_134_134_count
- cal_to_list_tt_1_1_count
- cal_to_list_tt_23_23_count
- cal_to_list_tt_3_3_count
- cal_to_list_tt_124_124_count
- cal_to_list_tt_4_4_count
- cal_to_list_tt_2_2_count
- cal_to_list_tt_12_1_count
- cal_to_list_tt_14_14_count
- cal_to_list_tt_123_12_count
- cal_to_list_tt_34_4_count
- cal_to_list_tt_24_24_count
- cal_to_list_tt_123_23_count
- cal_to_list_tt_1234_123_count
- cal_to_list_tt_34_3_count
- cal_to_list_tt_12_2_count
- cal_to_list_tt_123_13_count
- cal_to_list_tt_234_34_count
- cal_to_list_tt_1234_234_count
- cal_to_list_tt_1234_134_count
- cal_to_list_tt_1234_124_count
- cal_to_list_tt_234_23_count
- cal_to_list_tt_234_24_count
- cal_to_list_tt_13_1_count
- cal_to_list_tt_13_3_count
- cal_to_list_tt_23_2_count
- cal_to_list_tt_134_13_count
- cal_to_list_tt_134_34_count
- cal_to_list_tt_23_3_count
- cal_to_list_tt_134_14_count
- cal_to_list_tt_123_1_count
- cal_to_list_tt_124_24_count
- cal_to_list_tt_124_12_count
- cal_to_list_tt_234_2_count
- cal_to_list_tt_14_1_count
- cal_to_list_tt_124_14_count
- cal_to_list_tt_14_4_count
- cal_to_list_tt_24_4_count
- cal_to_list_tt_1234_12_count
- cal_to_list_tt_24_2_count
- cal_to_list_tt_1234_1_count
- cal_to_list_tt_123_2_count
- cal_to_list_tt_234_3_count
- cal_to_list_tt_1234_23_count
- cal_to_list_tt_1234_14_count
- cal_to_list_tt_134_1_count
- cal_to_list_tt_123_3_count
- cal_to_list_tt_234_4_count
- cal_to_list_tt_1234_24_count
- cal_to_list_tt_1234_13_count
- cal_to_list_tt_124_4_count
- cal_to_list_tt_1234_34_count
- cal_to_list_tt_134_3_count
- cal_to_list_tt_1234_4_count
- cal_to_list_tt_134_4_count
- cal_to_list_tt_1234_2_count
- cal_to_list_tt_124_1_count
- cal_to_list_tt_124_2_count
- cal_to_list_tt_1234_3_count


# Topology counts:
- active_strips_P1_1000_count
- active_strips_P1_0100_count
- active_strips_P1_0010_count
- active_strips_P1_0001_count
- active_strips_P1_1100_count
- active_strips_P1_0110_count
- active_strips_P1_0011_count
- active_strips_P1_1010_count
- active_strips_P1_1001_count
- active_strips_P1_0101_count
- active_strips_P1_1110_count
- active_strips_P1_1011_count
- active_strips_P1_0111_count
- active_strips_P1_1101_count
- active_strips_P1_1111_count
- active_strips_P2_1000_count
- active_strips_P2_0100_count
- active_strips_P2_0010_count
- active_strips_P2_0001_count
- active_strips_P2_1100_count
- active_strips_P2_0110_count
- active_strips_P2_0011_count
- active_strips_P2_1010_count
- active_strips_P2_1001_count
- active_strips_P2_0101_count
- active_strips_P2_1110_count
- active_strips_P2_1011_count
- active_strips_P2_0111_count
- active_strips_P2_1101_count
- active_strips_P2_1111_count
- active_strips_P3_1000_count
- active_strips_P3_0100_count
- active_strips_P3_0010_count
- active_strips_P3_0001_count
- active_strips_P3_1100_count
- active_strips_P3_0110_count
- active_strips_P3_0011_count
- active_strips_P3_1010_count
- active_strips_P3_1001_count
- active_strips_P3_0101_count
- active_strips_P3_1110_count
- active_strips_P3_1011_count
- active_strips_P3_0111_count
- active_strips_P3_1101_count
- active_strips_P3_1111_count
- active_strips_P4_1000_count
- active_strips_P4_0100_count
- active_strips_P4_0010_count
- active_strips_P4_0001_count
- active_strips_P4_1100_count
- active_strips_P4_0110_count
- active_strips_P4_0011_count
- active_strips_P4_1010_count
- active_strips_P4_1001_count
- active_strips_P4_0101_count
- active_strips_P4_1110_count
- active_strips_P4_1011_count
- active_strips_P4_0111_count
- active_strips_P4_1101_count
- active_strips_P4_1111_count
'''


# --- Additional intelligent plotting for STEP 1 TASK 1 ---

# determine time column to use for plotting
tcol = ctx.time_col

# Example reuse: plot clean -> cal pairs. Uncomment if wanted.
try:
    plot_tt_pairs(ctx, 'cal_tt_', 'list_tt_', f"cal_tt → list_tt • {STATION} STEP {STEP} TASK {TASK}", ncols=4)
except Exception as exc:
    print(f"Could not plot cal_tt_ -> list_tt_ pairs: {exc}")

# Plot raw->clean matrix (re-usable: change prefixes to plot other matrices)
try:
    plot_tt_matrix(ctx, 'cal', 'list', f"cal_to_list matrix • {STATION} STEP {STEP} TASK {TASK}")
except Exception as exc:
    print(f"Could not plot cal -> list matrix: {exc}")

#%%




planes = range(1, 5)
types = ["T_sum", "T_diff", "Q_sum", "Q_diff", "Y"]

# Adjustable marker size
marker_size = 2
sigma_limit = 4  # y-limits for normalized plots

# ----------------------------------------------------------
# First pass: determine global y-limits (before/after)
# ----------------------------------------------------------

y_min_global = None
y_max_global = None

for plane in planes:
    for typ in types:

        before = f"P{plane}_{typ}_nonzero_before_filter6"
        after  = f"P{plane}_{typ}_nonzero_after_filter6"

        if before in df.columns and after in df.columns:
            local_min = min(df[before].min(), df[after].min())
            local_max = max(df[before].max(), df[after].max())

            if y_min_global is None or local_min < y_min_global:
                y_min_global = local_min
            if y_max_global is None or local_max > y_max_global:
                y_max_global = local_max

# ----------------------------------------------------------
# Second pass: build figure
# ----------------------------------------------------------

fig = plt.figure(figsize=(28, 14))
outer_gs = gridspec.GridSpec(
    nrows=len(planes),
    ncols=len(types),
    hspace=0.35,
    wspace=0.25,
)

for i_plane, plane in enumerate(planes):
    for j_type, typ in enumerate(types):

        before = f"P{plane}_{typ}_nonzero_before_filter6"
        after  = f"P{plane}_{typ}_nonzero_after_filter6"

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer_gs[i_plane, j_type],
            height_ratios=[3, 1],
            hspace=0.05
        )

        ax_main = fig.add_subplot(inner_gs[0, 0])
        ax_norm = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)

        if before in df.columns and after in df.columns:

            t = df["datetime"]
            y_before = df[before]
            y_after  = df[after]

            # --- Upper raw plot ---
            ax_main.plot(t, y_before, marker="o", markersize=marker_size,
                        linestyle="", label="Before")
            ax_main.plot(t, y_after,  marker="x", markersize=marker_size,
                        linestyle="", label="After")

            ax_main.set_title(f"P{plane} – {typ}", fontsize=9)
            ax_main.set_ylim(y_min_global, y_max_global)

            if j_type == 0:
                ax_main.set_ylabel("Value")

            ax_main.grid(True)

            # Only one legend in top-left panel
            if i_plane == 0 and j_type == 0:
                ax_main.legend(loc="upper right", fontsize=7)

            # Hide x tick labels on upper plot
            plt.setp(ax_main.get_xticklabels(), visible=False)

            # Normalization function
            def normalize(v):
                m = np.nanmean(v)
                s = np.nanstd(v)
                return (v - m) / s if s > 0 else np.zeros_like(v)

            y_before_norm = normalize(y_before)
            y_after_norm  = normalize(y_after)

            # --- Lower normalized plot ---
            ax_norm.plot(t, y_before_norm, marker="o", markersize=marker_size,
                        linestyle="", label="Before")
            ax_norm.plot(t, y_after_norm,  marker="x", markersize=marker_size,
                        linestyle="", label="After")

            ax_norm.set_ylim(-sigma_limit, sigma_limit)
            ax_norm.axhline(0, linestyle="--", linewidth=0.8)

            if j_type == 0:
                ax_norm.set_ylabel("Z")

            if i_plane == len(planes) - 1:
                ax_norm.set_xlabel("Datetime")

            ax_norm.grid(True)

        else:
            ax_main.set_visible(False)
            ax_norm.set_visible(False)

fig.suptitle(
    "Before/After Filter6 Comparison Per Plane/Type (Global Raw Limits + Normalized ±4σ)",
    fontsize=14
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%

# I want four figures. Each figure has 4 rows, one per plane (P1 to P4),
# and 4, 3, 3 and 5 columns respectively. Each subplot is a time series
# scatter plot of the corresponding active_strips_*_count column.

# The sets are:
# 1) 1000, 0100, 0010, 0001
# 2) 1100, 0110, 0011
# 3) 1010, 1001, 0101
# 4) 1110, 1011, 0111, 1101, 1111

active_strips_sets = [
    ["1000", "0100", "0010", "0001"],
    ["1100", "0110", "0011"],
    ["1010", "1001", "0101"],
    ["1110", "1011", "0111", "1101", "1111"],
]

marker_size = 1

for set_idx, patterns in enumerate(active_strips_sets):
    ncols = len(patterns)
    nrows = 4  # P1 to P4
    fig_w = max(6, ncols * 3)
    fig_h = max(3, nrows * 2.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)

    # normalize axes shape
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[a] for a in axes])

    for i_r, plane in enumerate(range(1, 5)):
        for j_c, pattern in enumerate(patterns):
            ax = axes[i_r, j_c]
            colname = f"active_strips_P{plane}_{pattern}_count"
            if colname in df.columns:
                series = df[["datetime", colname]].dropna()
                if not series.empty:
                    series = series.sort_values(by="datetime")
                    ax.scatter(series["datetime"], series[colname], s=marker_size, color='C0', marker='o', alpha=0.9)
                    ax.set_title(f"P{plane} – {pattern}\n{colname}", fontsize=8)
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)
            
            # I want the same y limits for all subplots in this figure
            all_values = []
            for plane_check in range(1, 5):
                for pattern_check in patterns:
                    col_check = f"active_strips_P{plane_check}_{pattern_check}_count"
                    if col_check in df.columns:
                        all_values.extend(df[col_check].dropna().values)
            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                ax.set_ylim(y_min, y_max)
            ax.grid(True, linestyle='--', alpha=0.3)

            # only label y on first column and x on bottom row
            if j_c == 0:
                ax.set_ylabel('Count')
            if i_r == nrows - 1:
                ax.set_xlabel('Datetime')

    fig.suptitle(f"Active Strips Pattern Set {set_idx + 1} • {STATION} STEP {STEP} TASK {TASK}", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    plotted_cols.update([f"active_strips_P{plane}_{pattern}_count" for plane in range(1,5) for pattern in patterns if f"active_strips_P{plane}_{pattern}_count" in df.columns])





# %%
