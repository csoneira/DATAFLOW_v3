#%%


# TASK 1 --> channel counts
# TASK 2 --> 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_shared import load_metadata, print_columns, plot_tt_pairs, plot_tt_matrix

# --- knobs to tweak ---
station = 1

STATION = f"MINGO0{station}"  # e.g. "MINGO01", "MINGO02", ...
STEP = 1             # numeric step (1, 2, ...)
TASK = 2          # for STEP_1 use an int (1-5); keep None for steps without tasks
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

print("Columns:")
print_columns(df)


# Read the /home/mingo/DATAFLOW_v3/MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY/STATION_{station}/input_file_mingo0{station}.csv
runs_df = ...



# --- Additional intelligent plotting for STEP 1 TASK 1 ---

# determine time column to use for plotting
tcol = ctx.time_col

# Example reuse: plot clean -> cal pairs. Uncomment if wanted.
try:
    plot_tt_pairs(ctx, 'clean_tt_', 'cal_tt_', f"clean_tt → cal_tt • {STATION} STEP {STEP} TASK {TASK}", ncols=4)
except Exception:
    print("Could not plot clean_tt_ -> cal_tt_ pairs.")
    pass

# Plot raw->clean matrix (re-usable: change prefixes to plot other matrices)
try:
    plot_tt_matrix(ctx, 'clean', 'cal', f"clean_to_cal matrix • {STATION} STEP {STEP} TASK {TASK}")
except Exception:
    print("Could not plot clean -> cal matrix.")
    pass


    #%%

# - CRT_avg
# Plot CRT_avg time series with error bar = CRT_std

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))
plt.plot(df["datetime"], df["CRT_avg"], marker='.', linestyle='--', markersize=0.1)
plt.fill_between(df["datetime"], df["CRT_avg"] + df["CRT_std"]/2, df["CRT_avg"] - df["CRT_std"]/2, alpha=0.2, label='CRT Std Dev')
plt.title(f"CRT for {STATION} STEP {STEP} TASK {TASK}")
plt.xlabel("Datetime")
plt.ylabel("CRT (ns)")
# plt.ylim(480, 520)
plt.grid(True)
plt.show()
plotted_cols.update({"CRT_avg", "CRT_std"})



#%%



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

planes = range(1, 5)
strips = range(1, 5)

# Adjustable marker size
marker_size = 1
sigma_limit = 4  # y-limits for normalized plots

# ----------------------------------------------------------
# First pass: determine global y-limits (original & final)
# ----------------------------------------------------------

y_min_global = None
y_max_global = None

for plane in planes:
    for strip in strips:
        orig = f"P{plane}_s{strip}_entries_original"
        fin  = f"P{plane}_s{strip}_entries_final"

        if orig in df.columns and fin in df.columns:
            local_min = min(df[orig].min(), df[fin].min())
            local_max = max(df[orig].max(), df[fin].max())

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
    ncols=len(strips),
    hspace=0.35,
    wspace=0.25,
)

for i_plane, plane in enumerate(planes):
    for j_strip, strip in enumerate(strips):

        orig = f"P{plane}_s{strip}_entries_original"
        fin  = f"P{plane}_s{strip}_entries_final"

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer_gs[i_plane, j_strip],
            height_ratios=[3, 1],
            hspace=0.05
        )

        ax_main = fig.add_subplot(inner_gs[0, 0])
        ax_norm = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)

        if orig in df.columns and fin in df.columns:

            t = df["datetime"]
            y_orig = df[orig]
            y_fin  = df[fin]

            # Upper raw plot
            ax_main.plot(t, y_orig, marker="o", markersize=marker_size,
                        linestyle="", label="Original")
            ax_main.plot(t, y_fin,  marker="x", markersize=marker_size,
                        linestyle="", label="Final")

            ax_main.set_title(f"P{plane} S{strip}", fontsize=9)
            ax_main.set_ylim(y_min_global, y_max_global)

            if j_strip == 0:
                ax_main.set_ylabel("Hits")

            ax_main.grid(True)

            if i_plane == 0 and j_strip == 0:
                ax_main.legend(loc="upper right", fontsize=7)

            # Hide upper x tick labels
            plt.setp(ax_main.get_xticklabels(), visible=False)

            # Normalized values
            def normalize(v):
                m = np.nanmean(v)
                s = np.nanstd(v)
                return (v - m) / s if s > 0 else np.zeros_like(v)

            y_orig_norm = normalize(y_orig)
            y_fin_norm  = normalize(y_fin)

            # Lower normalized plot
            ax_norm.plot(t, y_orig_norm, marker="o", markersize=marker_size,
                        linestyle="", label="Original")
            ax_norm.plot(t, y_fin_norm,  marker="x", markersize=marker_size,
                        linestyle="", label="Final")

            ax_norm.set_ylim(-sigma_limit, sigma_limit)
            ax_norm.axhline(0, linestyle="--", linewidth=0.8)

            if j_strip == 0:
                ax_norm.set_ylabel("Z")

            if i_plane == len(planes) - 1:
                ax_norm.set_xlabel("Datetime")

            ax_norm.grid(True)

        else:
            ax_main.set_visible(False)
            ax_norm.set_visible(False)

fig.suptitle("Hits per plane/strip for T_sum (global raw limits and ±3 normalized)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Loop on plane, strip, original/final
#     - T1_T_dif_1_entries_original
# It's the same


# Loop on plane, strip, original/final
#     - Q1_Q_sum_1_entries_original
# It's the same


# Loop on plane, strip, original/final
#     - Q1_Q_dif_1_entries_original
# It's the same




#%%





# Loop on plane, strip, with/without crstlk
#     - Q1_Q_sum_1_with_crstlk_entries_final
#     - Q1_Q_sum_1_no_crstlk_entries_final

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

planes = range(1, 5)
strips = range(1, 5)

fig = plt.figure(figsize=(28, 14))
outer_gs = gridspec.GridSpec(
    nrows=len(planes),
    ncols=len(strips),
    hspace=0.35,
    wspace=0.25,
)

for i_plane, plane in enumerate(planes):
    for j_strip, strip in enumerate(strips):

        ax = fig.add_subplot(outer_gs[i_plane, j_strip])

        original_col = f"Q{plane}_Q_sum_{strip}_with_crstlk_entries_final"
        final_col    = f"Q{plane}_Q_sum_{strip}_no_crstlk_entries_final"

        if original_col in df.columns and final_col in df.columns:
            ax.plot(df["datetime"], df[original_col],
                    marker=".", linestyle="", label="with_crstlk", markersize=1)
            ax.plot(df["datetime"], df[final_col],
                    marker="x", linestyle="", label="no_crstlk", markersize=1)

            ax.set_title(f"P{plane} S{strip}", fontsize=9)

            if j_strip == 0:
                ax.set_ylabel("Hits")

            if i_plane == len(planes) - 1:
                ax.set_xlabel("Datetime")

            ax.grid(True)

            if i_plane == 0 and j_strip == 0:
                ax.legend(loc="upper right", fontsize=7)

        else:
            ax.set_visible(False)

fig.suptitle("Hits per plane/strip for T_sum (with and without crosstalk)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()





# Loop on plane, strip, coeffs
#     - P1_s1_Q_FB_coeffs





#%%

# Loop on planes and strips, fill between
#     - P1_s1_crstlk_pedestal
#     - P1_s1_crstlk_limit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

planes = range(1, 5)
strips = range(1, 5)

fig = plt.figure(figsize=(28, 14))
outer_gs = gridspec.GridSpec(
    nrows=len(planes),
    ncols=len(strips),
    hspace=0.35,
    wspace=0.25,
)

for i_plane, plane in enumerate(planes):
    for j_strip, strip in enumerate(strips):

        ax = fig.add_subplot(outer_gs[i_plane, j_strip])

        low_col = f"P{plane}_s{strip}_crstlk_pedestal"
        up_col    = f"P{plane}_s{strip}_crstlk_limit"

        if low_col in df.columns and up_col in df.columns:
            ax.plot(df["datetime"], df[low_col],
                    marker=".", linestyle="", label="Original", markersize=1)
            ax.plot(df["datetime"], df[up_col],
                    marker="x", linestyle="", label="Final", markersize=1)
            
            ax.fill_between(df["datetime"], df[low_col], df[up_col], color='gray', alpha=0.3)

            ax.set_title(f"P{plane} S{strip}", fontsize=9)

            if j_strip == 0:
                ax.set_ylabel("Hits")

            if i_plane == len(planes) - 1:
                ax.set_xlabel("Datetime")

            ax.grid(True)

            if i_plane == 0 and j_strip == 0:
                ax.legend(loc="upper right", fontsize=7)

        else:
            ax.set_visible(False)

fig.suptitle("Pedestal", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()






#%%


# Loop on planes and strips
#     - P1_s1_Q_F
#     - P1_s1_Q_B

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

planes = range(1, 5)
strips = range(1, 5)

fig = plt.figure(figsize=(28, 14))
outer_gs = gridspec.GridSpec(
    nrows=len(planes),
    ncols=len(strips),
    hspace=0.35,
    wspace=0.25,
)

for i_plane, plane in enumerate(planes):
    for j_strip, strip in enumerate(strips):

        ax = fig.add_subplot(outer_gs[i_plane, j_strip])

        original_col = f"P{plane}_s{strip}_Q_F"
        final_col = f"P{plane}_s{strip}_Q_B"

        if original_col in df.columns and final_col in df.columns:
            ax.plot(df["datetime"], df[original_col],
                    marker=".", linestyle="", label="Front", markersize=1)
            ax.plot(df["datetime"], df[final_col],
                    marker="x", linestyle="", label="Back", markersize=1)
            
            ax.set_title(f"P{plane} S{strip}", fontsize=9)

            if j_strip == 0:
                ax.set_ylabel("Hits")

            if i_plane == len(planes) - 1:
                ax.set_xlabel("Datetime")

            ax.grid(True)

            if i_plane == 0 and j_strip == 0:
                ax.legend(loc="upper right", fontsize=7)

        else:
            ax.set_visible(False)

fig.suptitle("Q_F and Q_B calibration parameters", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



#%%


# Loop on planes and strips
#     - P1_s1_Q_sum

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

planes = range(1, 5)
strips = range(1, 5)

fig = plt.figure(figsize=(28, 14))
outer_gs = gridspec.GridSpec(
    nrows=len(planes),
    ncols=len(strips),
    hspace=0.35,
    wspace=0.25,
)

for i_plane, plane in enumerate(planes):
    for j_strip, strip in enumerate(strips):

        ax = fig.add_subplot(outer_gs[i_plane, j_strip])

        original_col = f"P{plane}_s{strip}_Q_sum"

        if original_col in df.columns:
            ax.plot(df["datetime"], df[original_col],
                    marker=".", linestyle="", label="Parameter", markersize=1)
            
            ax.set_title(f"P{plane} S{strip}", fontsize=9)

            if j_strip == 0:
                ax.set_ylabel("Hits")

            if i_plane == len(planes) - 1:
                ax.set_xlabel("Datetime")

            ax.grid(True)

            if i_plane == 0 and j_strip == 0:
                ax.legend(loc="upper right", fontsize=7)

        else:
            ax.set_visible(False)

fig.suptitle("Q_sum calibration parameter", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%


# Loop on planes and strips
#     - P1_s1_T_sum



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Adjustable marker size
marker_size = 1   # change this value to control point size
sigma_limit = 3  # y-limits for normalized plots

planes = range(1, 5)
strips = range(1, 5)

for variable in ["T_sum", "T_dif"]:

    fig = plt.figure(figsize=(28, 14))
    outer_gs = gridspec.GridSpec(
        nrows=len(planes),
        ncols=len(strips),
        hspace=0.35,
        wspace=0.25,
    )

    # Global y-limits for upper plots
    y_lim_upper = None
    y_lim_lower = None

    for plane in planes:
        for strip in strips:
            col = f"P{plane}_s{strip}_{variable}"
            if col in df.columns:
                current_max = df[col].max()
                current_min = df[col].min()
                if y_lim_upper is None or current_max > y_lim_upper:
                    y_lim_upper = current_max
                if y_lim_lower is None or current_min < y_lim_lower:
                    y_lim_lower = current_min

    for i_plane, plane in enumerate(planes):
        for j_strip, strip in enumerate(strips):

            col = f"P{plane}_s{strip}_{variable}"

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=outer_gs[i_plane, j_strip],
                height_ratios=[3, 1],
                hspace=0.05,
            )

            ax_main = fig.add_subplot(inner_gs[0, 0])
            ax_norm = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)

            if col in df.columns:

                y = df[col]
                t = df["datetime"]

                # Raw
                ax_main.plot(t, y, marker=".", markersize=marker_size,
                            linestyle="", label="Parameter")
                ax_main.set_title(f"P{plane} S{strip}", fontsize=9)
                ax_main.set_ylim(y_lim_lower, y_lim_upper)

                if j_strip == 0:
                    ax_main.set_ylabel("Value")

                if i_plane == len(planes) - 1:
                    ax_norm.set_xlabel("Datetime")

                ax_main.grid(True)

                # Normalized
                y_mean = np.nanmean(y)
                y_std = np.nanstd(y)
                if y_std > 0:
                    y_norm = (y - y_mean) / y_std
                else:
                    y_norm = np.zeros_like(y)

                ax_norm.plot(t, y_norm, marker="o", markersize=marker_size,
                            linestyle="", label="(x - μ)/σ")

                ax_norm.set_ylim(-sigma_limit, sigma_limit)
                ax_norm.axhline(0, linestyle="--", linewidth=0.8)
                if j_strip == 0:
                    ax_norm.set_ylabel("Z")

                ax_norm.grid(True)

                if i_plane == 0 and j_strip == 0:
                    ax_main.legend(loc="upper right", fontsize=7)

                plt.setp(ax_main.get_xticklabels(), visible=False)

            else:
                ax_main.set_visible(False)
                ax_norm.set_visible(False)

    fig.suptitle(f"{variable} calibration parameter (raw and normalized)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


#%%
