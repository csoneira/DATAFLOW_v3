#%%


# TASK 1 --> channel counts
# TASK 2 --> 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from qa_shared import load_metadata, print_columns, plot_tt_pairs, plot_tt_matrix

# --- knobs to tweak ---
station = 1

STATION = f"MINGO0{station}"  # e.g. "MINGO01", "MINGO02", ...
STEP = 1             # numeric step (1, 2, ...)
TASK = 2          # for STEP_1 use an int (1-5); keep None for steps without tasks
# START_DATE = "2024-03-01 00:00:00"    # e.g. "2025-11-06 18:00:00" or leave None
START_DATE = "2025-11-01 00:00:00"    # e.g. "2025-11-06 18:00:00" or leave None
END_DATE = "2025-12-11 00:00:00"      # e.g. "2025-11-06 19:00:00" or leave None
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


#%%

# --- Additional intelligent plotting for STEP 1 TASK 1 ---

# determine time column to use for plotting
tcol = ctx.time_col

# # Example reuse: plot clean -> cal pairs. Uncomment if wanted.
# try:
#     plot_tt_pairs(ctx, 'clean_tt_', 'cal_tt_', f"clean_tt → cal_tt • {STATION} STEP {STEP} TASK {TASK}", ncols=4)
# except Exception:
#     print("Could not plot clean_tt_ -> cal_tt_ pairs.")
#     pass

# # Plot raw->clean matrix (re-usable: change prefixes to plot other matrices)
# try:
#     plot_tt_matrix(ctx, 'clean', 'cal', f"clean_to_cal matrix • {STATION} STEP {STEP} TASK {TASK}")
# except Exception:
#     print("Could not plot clean -> cal matrix.")
#     pass


#%%

# - CRT_avg
# Plot CRT_avg time series with error bar = CRT_std

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))
plt.plot(df["datetime"], df["CRT_avg"], marker='.', linestyle='', markersize=1)
plt.fill_between(df["datetime"], df["CRT_avg"] + df["CRT_std"]/2, df["CRT_avg"] - df["CRT_std"]/2, alpha=0.2, label='CRT Std Dev')
plt.title(f"CRT for {STATION} STEP {STEP} TASK {TASK}")
plt.xlabel("Datetime")
plt.ylabel("CRT (ns)")
# plt.ylim(480, 520)
plt.grid(True)
plt.show()
plotted_cols.update({"CRT_avg", "CRT_std"})

#%%

tt_values = ["12", "13", "14", "23", "24", "34", "123", "124", "134", "234", "1234"]

# Pre-create all required difference columns to avoid dataframe fragmentation
tsum_diff_columns = []
for plane_1 in range(1, 5):
    for plane_2 in range(1, 5):
        for strip_1 in range(1, 5):
            for strip_2 in range(1, 5):
                if plane_1 > plane_2 and strip_1 > strip_2:
                    continue
                if plane_1 == plane_2 and strip_1 == strip_2:
                    continue
                for tt in tt_values:
                    tsum_diff_columns.append(f"P{plane_1}s{strip_1}_P{plane_2}s{strip_2}_{tt}")
tsum_diff_columns = sorted(set(tsum_diff_columns))

#%%



# # I want the following plot: the same rows and columns. From P1s1 to P4s4, for all tt combinations
# # and in each combination, you plot the time series of the P{plane}s{strip}_P{plane}s{strip}_{tt} column
# # for all tt combinations.

# tt_values = ["12", "13", "14", "23", "24", "34", "123", "124", "134", "234", "1234"]

# # Add a .0 to all tt values to match column names
# tt_values = [tt + ".0" for tt in tt_values]

# fig, axs = plt.subplots(16, 16, figsize=(40, 30), sharex=True, sharey=True)
# for i, plane_1 in enumerate(range(1, 5)):
#     for j, strip_1 in enumerate(range(1, 5)):
#         for k, plane_2 in enumerate(range(1, 5)):
#             for l, strip_2 in enumerate(range(1, 5)):
#                 ax = axs[i*4 + j, k*4 + l]
                
#                 if plane_1 * 4 + strip_1 >= plane_2 * 4 + strip_2:
#                     ax.axis('off')
#                     continue
                
#                 for tt in tt_values:
#                     col_name = f"P{plane_1}s{strip_1}_P{plane_2}s{strip_2}_{tt}"
#                     if col_name in df.columns:
#                         ax.plot(df["datetime"], df[col_name], marker='.', linestyle='--', markersize=3, label=tt)
                
#                 ax.set_title(f"P{plane_1}s{strip_1}-P{plane_2}s{strip_2}")
#                 ax.grid(True)
                
#                 # if i == 3 and j == 3:
#                 #     ax.legend(fontsize='small', loc='upper right')

# plt.suptitle(f"Tsum Differences for {STATION} STEP {STEP} TASK {TASK}", fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


#%%

tt_values = ["12", "13", "14", "23", "24", "34", "123", "124", "134", "234", "1234"]
tt_values = [tt + ".0" for tt in tt_values]
tt_colors = {tt: plt.get_cmap('tab20', len(tt_values))(idx) for idx, tt in enumerate(tt_values)}

for plane_1 in range(1, 5):
    for plane_2 in range(plane_1, 5):  # plane_1 <= plane_2 to avoid duplicating plane pairs
        fig, axs = plt.subplots(4, 4, figsize=(14, 10), sharex=True, sharey=True)
        legend_handles: dict[str, plt.Line2D] = {}

        for strip_1 in range(1, 5):
            for strip_2 in range(1, 5):
                ax = axs[strip_1 - 1, strip_2 - 1]

                if plane_1 == plane_2 and strip_1 <= strip_2:
                    ax.axis('off')
                    continue

                # Build column name for this plane/strip pair
                for tt in tt_values:
                    col_name = f"P{plane_1}s{strip_1}_P{plane_2}s{strip_2}_{tt}"
                    if col_name not in df.columns:
                        continue

                    y = df[col_name]
                    cond = ~y.isna()
                    if not cond.any():
                        continue

                    x = df["datetime"][cond]
                    y = y[cond]
                    (line,) = ax.plot(
                        x,
                        y,
                        marker='.',
                        linestyle='-',
                        markersize=3,
                        color=tt_colors[tt],
                        label=tt,
                    )
                    legend_handles.setdefault(tt, line)

                ax.set_title(f"P{plane_1}s{strip_1}-P{plane_2}s{strip_2}", fontsize=8)
                ax.grid(True)

        fig.suptitle(
            f"Tsum Differences for {STATION} STEP {STEP} TASK {TASK} | "
            f"Planes P{plane_1}–P{plane_2}",
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
        plt.show()

#%%



"""
Modify this using the loaded data from the csv

This section loads the plane/strip pair counts per measured_type generated by
the simulator and:
- Overlays those counts as horizontal reference lines on top of the existing
    time-series plot style for each plane-strip pair and measured_type.
- Adds a comparison scatter plot where X is the CSV count (repeated for each
    time-row with data) and Y is the data time-series value, with a y = x line.
Rules:
- Same-plane: show only upper triangle (strip_1 < strip_2); hide diagonal.
- Inter-plane: show all combinations.
"""


factor_tt = 2.0


import os as _os
import yaml as _yaml

# Resolve home_path from the same global config used by the simulator
_user_home = _os.path.expanduser("~")
_config_file_path = _os.path.join(_user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
with open(_config_file_path, "r") as _cfg:
    _cfg_data = _yaml.safe_load(_cfg)
_home_path = _cfg_data["home_path"]

_counts_csv_path = _os.path.join(
    _home_path,
    "DATAFLOW_v3/TESTS/SIMULATION/plane_strip_pair_counts.csv",
)

if not _os.path.exists(_counts_csv_path):
    print(f"Counts CSV not found: {_counts_csv_path}. Run the simulator to generate it.")
else:
    counts_pivot = pd.read_csv(_counts_csv_path, index_col=0)

    measured_plot_types = list(counts_pivot.columns)
    cmap = plt.get_cmap('tab20', max(1, len(measured_plot_types)))
    tt_colors = {tt: cmap(idx) for idx, tt in enumerate(measured_plot_types)}

    # Build quick lookup: pair -> Series(measured_type -> count)
    counts_by_pair = {pair: counts_pivot.loc[pair] for pair in counts_pivot.index}

    # Helper to locate data column matching CSV measured_type (try with and without .0)
    def _data_col_name(p1, s1, p2, s2, tt):
        base = f"P{p1}s{s1}_P{p2}s{s2}_"
        c1 = base + tt
        c2 = base + tt + ".0"
        if c1 in df.columns:
            return c1
        if c2 in df.columns:
            return c2
        return None

    # 1) Overlay counts (horizontal lines) on top of time-series data
    if not df.empty and "datetime" in df.columns:
        x_min = df["datetime"].min()
        x_max = df["datetime"].max()

        for plane_1 in range(1, 5):
            for plane_2 in range(plane_1, 5):
                fig, axs = plt.subplots(4, 4, figsize=(14, 10), sharex=True, sharey=True)
                legend_handles: dict[str, plt.Line2D] = {}

                for strip_1 in range(1, 5):
                    for strip_2 in range(1, 5):
                        ax = axs[strip_1 - 1, strip_2 - 1]

                        if plane_1 == plane_2 and strip_1 >= strip_2:
                            ax.axis('off')
                            continue

                        pair_label = f"P{plane_1}s{strip_1}_P{plane_2}s{strip_2}"
                        counts_series = counts_by_pair.get(pair_label)
                        if counts_series is None:
                            ax.axis('off')
                            continue

                        # Plot data time-series (if available)
                        for tt in measured_plot_types:
                            col_name = _data_col_name(plane_1, strip_1, plane_2, strip_2, tt)
                            if col_name is None:
                                continue
                            y = df[col_name]
                            cond = ~y.isna()
                            if cond.any():
                                (line_data,) = ax.plot(
                                    df["datetime"][cond],
                                    y[cond],
                                    marker='.',
                                    linestyle='-',
                                    markersize=2,
                                    color=tt_colors[tt],
                                    alpha=0.6,
                                    label=tt,
                                )
                                legend_handles.setdefault(tt, line_data)

                        # Overlay CSV count as horizontal reference line
                        for tt in measured_plot_types:
                            count_val = counts_series.get(tt, 0) * factor_tt
                            ax.hlines(
                                y=count_val,
                                xmin=x_min,
                                xmax=x_max,
                                colors=[tt_colors[tt]],
                                linestyles='--',
                                linewidth=1,
                            )

                        ax.set_title(f"P{plane_1}s{strip_1}-P{plane_2}s{strip_2}", fontsize=8)
                        ax.grid(True, axis='both', linestyle='--', alpha=0.3)

                fig.suptitle(
                    f"Data vs CSV counts • {STATION} STEP {STEP} TASK {TASK} | Planes P{plane_1}–P{plane_2}",
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
                plt.show()

    # 2) Comparison scatter: X = CSV count (repeated), Y = data values; add y=x line
    for plane_1 in range(1, 5):
        for plane_2 in range(plane_1, 5):
            fig, axs = plt.subplots(4, 4, figsize=(14, 10), sharex=True, sharey=True)

            for strip_1 in range(1, 5):
                for strip_2 in range(1, 5):
                    ax = axs[strip_1 - 1, strip_2 - 1]
                    if plane_1 == plane_2 and strip_1 >= strip_2:
                        ax.axis('off')
                        continue

                    pair_label = f"P{plane_1}s{strip_1}_P{plane_2}s{strip_2}"
                    counts_series = counts_by_pair.get(pair_label)
                    if counts_series is None:
                        ax.axis('off')
                        continue

                    all_x = []
                    all_y = []
                    for tt in measured_plot_types:
                        col_name = _data_col_name(plane_1, strip_1, plane_2, strip_2, tt)
                        if col_name is None:
                            continue
                        y = df[col_name]
                        y = y[~y.isna()]
                        if y.empty:
                            continue
                        count_val = counts_series.get(tt, 0) * factor_tt
                        x = np.full_like(y.values, fill_value=count_val, dtype=float)
                        all_x.append(x)
                        all_y.append(y.values)

                        ax.scatter(x, y.values, s=4, alpha=0.5, color=tt_colors[tt])

                    # y = x reference
                    if all_x and all_y:
                        xy = np.concatenate(all_x + all_y)
                        lo = np.nanmin(xy)
                        hi = np.nanmax(xy)
                        if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                            ax.plot([lo, hi], [lo, hi], color='k', linestyle=':', linewidth=1)

                    ax.set_title(f"P{plane_1}s{strip_1}-P{plane_2}s{strip_2}", fontsize=8)
                    ax.set_xlabel("CSV count")
                    ax.set_ylabel("Data value")
                    ax.grid(True, linestyle='--', alpha=0.3)

            fig.suptitle(
                f"CSV vs Data scatter • {STATION} STEP {STEP} TASK {TASK} | Planes P{plane_1}–P{plane_2}",
                fontsize=14
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()





#%%


#  - P1s1_P1s2_123.0
#  - P1s1_P1s2_12.0
#  - P1s1_P1s2_34.0
#  - P1s1_P1s2_1234.0
#  - P1s1_P1s2_234.0
#  - P1s1_P1s2_23.0
#  - P1s1_P1s2_13.0
#  - P1s1_P1s2_134.0
#  - P1s1_P1s2_124.0
#  - P1s1_P1s3_123.0
#  - P1s1_P1s3_12.0
#  - P1s1_P1s3_34.0
#  - P1s1_P1s3_1234.0
#  - P1s1_P1s3_234.0
#  - P1s1_P1s3_23.0
#  - P1s1_P1s3_13.0
#  - P1s1_P1s3_134.0
#  - P1s1_P1s3_124.0
#  - P1s1_P1s4_123.0
#  - P1s1_P1s4_12.0
#  - P1s1_P1s4_34.0
#  - P1s1_P1s4_1234.0
#  - P1s1_P1s4_234.0
#  - P1s1_P1s4_23.0
#  - P1s1_P1s4_13.0
#  - P1s1_P1s4_134.0
#  - P1s1_P1s4_124.0
#  - P1s2_P1s1_123.0
#  - P1s2_P1s1_12.0
#  - P1s2_P1s1_34.0
#  - P1s2_P1s1_1234.0
#  - P1s2_P1s1_234.0
#  - P1s2_P1s1_23.0
#  - P1s2_P1s1_13.0
#  - P1s2_P1s1_134.0
#  - P1s2_P1s1_124.0
#  - P1s2_P1s3_123.0
#  - P1s2_P1s3_12.0
#  - P1s2_P1s3_34.0
#  - P1s2_P1s3_1234.0
#  - P1s2_P1s3_234.0
#  - P1s2_P1s3_23.0
#  - P1s2_P1s3_13.0
#  - P1s2_P1s3_134.0
#  - P1s2_P1s3_124.0
#  - P1s2_P1s4_123.0
#  - P1s2_P1s4_12.0
#  - P1s2_P1s4_34.0
#  - P1s2_P1s4_1234.0
#  - P1s2_P1s4_234.0
#  - P1s2_P1s4_23.0
#  - P1s2_P1s4_13.0
#  - P1s2_P1s4_134.0
#  - P1s2_P1s4_124.0
#  - P1s3_P1s1_123.0
#  - P1s3_P1s1_12.0
#  - P1s3_P1s1_34.0
#  - P1s3_P1s1_1234.0
#  - P1s3_P1s1_234.0
#  - P1s3_P1s1_23.0
#  - P1s3_P1s1_13.0
#  - P1s3_P1s1_134.0
#  - P1s3_P1s1_124.0
#  - P1s3_P1s2_123.0
#  - P1s3_P1s2_12.0
#  - P1s3_P1s2_34.0
#  - P1s3_P1s2_1234.0
#  - P1s3_P1s2_234.0
#  - P1s3_P1s2_23.0
#  - P1s3_P1s2_13.0
#  - P1s3_P1s2_134.0
#  - P1s3_P1s2_124.0
#  - P1s3_P1s4_123.0
#  - P1s3_P1s4_12.0
#  - P1s3_P1s4_34.0
#  - P1s3_P1s4_1234.0
#  - P1s3_P1s4_234.0
#  - P1s3_P1s4_23.0
#  - P1s3_P1s4_13.0
#  - P1s3_P1s4_134.0
#  - P1s3_P1s4_124.0
#  - P1s4_P1s1_123.0
#  - P1s4_P1s1_12.0
#  - P1s4_P1s1_34.0
#  - P1s4_P1s1_1234.0
#  - P1s4_P1s1_234.0
#  - P1s4_P1s1_23.0
#  - P1s4_P1s1_13.0
#  - P1s4_P1s1_134.0
#  - P1s4_P1s1_124.0
#  - P1s4_P1s2_123.0
#  - P1s4_P1s2_12.0
#  - P1s4_P1s2_34.0
#  - P1s4_P1s2_1234.0
#  - P1s4_P1s2_234.0
#  - P1s4_P1s2_23.0
#  - P1s4_P1s2_13.0
#  - P1s4_P1s2_134.0
#  - P1s4_P1s2_124.0
#  - P1s4_P1s3_123.0
#  - P1s4_P1s3_12.0
#  - P1s4_P1s3_34.0
#  - P1s4_P1s3_1234.0
#  - P1s4_P1s3_234.0
#  - P1s4_P1s3_23.0
#  - P1s4_P1s3_13.0
#  - P1s4_P1s3_134.0
#  - P1s4_P1s3_124.0
#  - P1s1_P2s1_123.0
#  - P1s1_P2s1_12.0
#  - P1s1_P2s1_34.0
#  - P1s1_P2s1_1234.0
#  - P1s1_P2s1_234.0
#  - P1s1_P2s1_23.0
#  - P1s1_P2s1_13.0
#  - P1s1_P2s1_134.0
#  - P1s1_P2s1_124.0
#  - P1s1_P2s2_123.0
#  - P1s1_P2s2_12.0
#  - P1s1_P2s2_34.0
#  - P1s1_P2s2_1234.0
#  - P1s1_P2s2_234.0
#  - P1s1_P2s2_23.0
#  - P1s1_P2s2_13.0
#  - P1s1_P2s2_134.0
#  - P1s1_P2s2_124.0
#  - P1s1_P2s3_123.0
#  - P1s1_P2s3_12.0
#  - P1s1_P2s3_34.0
#  - P1s1_P2s3_1234.0
#  - P1s1_P2s3_234.0
#  - P1s1_P2s3_23.0
#  - P1s1_P2s3_13.0
#  - P1s1_P2s3_134.0
#  - P1s1_P2s3_124.0
#  - P1s1_P2s4_123.0
#  - P1s1_P2s4_12.0
#  - P1s1_P2s4_34.0
#  - P1s1_P2s4_1234.0
#  - P1s1_P2s4_234.0
#  - P1s1_P2s4_23.0
#  - P1s1_P2s4_13.0
#  - P1s1_P2s4_134.0
#  - P1s1_P2s4_124.0
#  - P1s2_P2s1_123.0
#  - P1s2_P2s1_12.0
#  - P1s2_P2s1_34.0
#  - P1s2_P2s1_1234.0
#  - P1s2_P2s1_234.0
#  - P1s2_P2s1_23.0
#  - P1s2_P2s1_13.0
#  - P1s2_P2s1_134.0
#  - P1s2_P2s1_124.0
#  - P1s2_P2s2_123.0
#  - P1s2_P2s2_12.0
#  - P1s2_P2s2_34.0
#  - P1s2_P2s2_1234.0
#  - P1s2_P2s2_234.0
#  - P1s2_P2s2_23.0
#  - P1s2_P2s2_13.0
#  - P1s2_P2s2_134.0
#  - P1s2_P2s2_124.0
#  - P1s2_P2s3_123.0
#  - P1s2_P2s3_12.0
#  - P1s2_P2s3_34.0
#  - P1s2_P2s3_1234.0
#  - P1s2_P2s3_234.0
#  - P1s2_P2s3_23.0
#  - P1s2_P2s3_13.0
#  - P1s2_P2s3_134.0
#  - P1s2_P2s3_124.0
#  - P1s2_P2s4_123.0
#  - P1s2_P2s4_12.0
#  - P1s2_P2s4_34.0
#  - P1s2_P2s4_1234.0
#  - P1s2_P2s4_234.0
#  - P1s2_P2s4_23.0
#  - P1s2_P2s4_13.0
#  - P1s2_P2s4_134.0
#  - P1s2_P2s4_124.0
#  - P1s3_P2s1_123.0
#  - P1s3_P2s1_12.0
#  - P1s3_P2s1_34.0
#  - P1s3_P2s1_1234.0
#  - P1s3_P2s1_234.0
#  - P1s3_P2s1_23.0
#  - P1s3_P2s1_13.0
#  - P1s3_P2s1_134.0
#  - P1s3_P2s1_124.0
#  - P1s3_P2s2_123.0
#  - P1s3_P2s2_12.0
#  - P1s3_P2s2_34.0
#  - P1s3_P2s2_1234.0
#  - P1s3_P2s2_234.0
#  - P1s3_P2s2_23.0
#  - P1s3_P2s2_13.0
#  - P1s3_P2s2_134.0
#  - P1s3_P2s2_124.0
#  - P1s3_P2s3_123.0
#  - P1s3_P2s3_12.0
#  - P1s3_P2s3_34.0
#  - P1s3_P2s3_1234.0
#  - P1s3_P2s3_234.0
#  - P1s3_P2s3_23.0
#  - P1s3_P2s3_13.0
#  - P1s3_P2s3_134.0
#  - P1s3_P2s3_124.0
#  - P1s3_P2s4_123.0
#  - P1s3_P2s4_12.0
#  - P1s3_P2s4_34.0
#  - P1s3_P2s4_1234.0
#  - P1s3_P2s4_234.0
#  - P1s3_P2s4_23.0
#  - P1s3_P2s4_13.0
#  - P1s3_P2s4_134.0
#  - P1s3_P2s4_124.0
#  - P1s4_P2s1_123.0
#  - P1s4_P2s1_12.0
#  - P1s4_P2s1_34.0
#  - P1s4_P2s1_1234.0
#  - P1s4_P2s1_234.0
#  - P1s4_P2s1_23.0
#  - P1s4_P2s1_13.0
#  - P1s4_P2s1_134.0
#  - P1s4_P2s1_124.0
#  - P1s4_P2s2_123.0
#  - P1s4_P2s2_12.0
#  - P1s4_P2s2_34.0
#  - P1s4_P2s2_1234.0
#  - P1s4_P2s2_234.0
#  - P1s4_P2s2_23.0
#  - P1s4_P2s2_13.0
#  - P1s4_P2s2_134.0
#  - P1s4_P2s2_124.0
#  - P1s4_P2s3_123.0
#  - P1s4_P2s3_12.0
#  - P1s4_P2s3_34.0
#  - P1s4_P2s3_1234.0
#  - P1s4_P2s3_234.0
#  - P1s4_P2s3_23.0
#  - P1s4_P2s3_13.0
#  - P1s4_P2s3_134.0
#  - P1s4_P2s3_124.0
#  - P1s4_P2s4_123.0
#  - P1s4_P2s4_12.0
#  - P1s4_P2s4_34.0
#  - P1s4_P2s4_1234.0
#  - P1s4_P2s4_234.0
#  - P1s4_P2s4_23.0
#  - P1s4_P2s4_13.0
#  - P1s4_P2s4_134.0
#  - P1s4_P2s4_124.0
#  - P1s1_P3s1_123.0
#  - P1s1_P3s1_12.0
#  - P1s1_P3s1_34.0
#  - P1s1_P3s1_1234.0
#  - P1s1_P3s1_234.0
#  - P1s1_P3s1_23.0
#  - P1s1_P3s1_13.0
#  - P1s1_P3s1_134.0
#  - P1s1_P3s1_124.0
#  - P1s1_P3s2_123.0
#  - P1s1_P3s2_12.0
#  - P1s1_P3s2_34.0
#  - P1s1_P3s2_1234.0
#  - P1s1_P3s2_234.0
#  - P1s1_P3s2_23.0
#  - P1s1_P3s2_13.0
#  - P1s1_P3s2_134.0
#  - P1s1_P3s2_124.0
#  - P1s1_P3s3_123.0
#  - P1s1_P3s3_12.0
#  - P1s1_P3s3_34.0
#  - P1s1_P3s3_1234.0
#  - P1s1_P3s3_234.0
#  - P1s1_P3s3_23.0
#  - P1s1_P3s3_13.0
#  - P1s1_P3s3_134.0
#  - P1s1_P3s3_124.0
#  - P1s1_P3s4_123.0
#  - P1s1_P3s4_12.0
#  - P1s1_P3s4_34.0
#  - P1s1_P3s4_1234.0
#  - P1s1_P3s4_234.0
#  - P1s1_P3s4_23.0
#  - P1s1_P3s4_13.0
#  - P1s1_P3s4_134.0
#  - P1s1_P3s4_124.0
#  - P1s2_P3s1_123.0
#  - P1s2_P3s1_12.0
#  - P1s2_P3s1_34.0
#  - P1s2_P3s1_1234.0
#  - P1s2_P3s1_234.0
#  - P1s2_P3s1_23.0
#  - P1s2_P3s1_13.0
#  - P1s2_P3s1_134.0
#  - P1s2_P3s1_124.0
#  - P1s2_P3s2_123.0
#  - P1s2_P3s2_12.0
#  - P1s2_P3s2_34.0
#  - P1s2_P3s2_1234.0
#  - P1s2_P3s2_234.0
#  - P1s2_P3s2_23.0
#  - P1s2_P3s2_13.0
#  - P1s2_P3s2_134.0
#  - P1s2_P3s2_124.0
#  - P1s2_P3s3_123.0
#  - P1s2_P3s3_12.0
#  - P1s2_P3s3_34.0
#  - P1s2_P3s3_1234.0
#  - P1s2_P3s3_234.0
#  - P1s2_P3s3_23.0
#  - P1s2_P3s3_13.0
#  - P1s2_P3s3_134.0
#  - P1s2_P3s3_124.0
#  - P1s2_P3s4_123.0
#  - P1s2_P3s4_12.0
#  - P1s2_P3s4_34.0
#  - P1s2_P3s4_1234.0
#  - P1s2_P3s4_234.0
#  - P1s2_P3s4_23.0
#  - P1s2_P3s4_13.0
#  - P1s2_P3s4_134.0
#  - P1s2_P3s4_124.0
#  - P1s3_P3s1_123.0
#  - P1s3_P3s1_12.0
#  - P1s3_P3s1_34.0
#  - P1s3_P3s1_1234.0
#  - P1s3_P3s1_234.0
#  - P1s3_P3s1_23.0
#  - P1s3_P3s1_13.0
#  - P1s3_P3s1_134.0
#  - P1s3_P3s1_124.0
#  - P1s3_P3s2_123.0
#  - P1s3_P3s2_12.0
#  - P1s3_P3s2_34.0
#  - P1s3_P3s2_1234.0
#  - P1s3_P3s2_234.0
#  - P1s3_P3s2_23.0
#  - P1s3_P3s2_13.0
#  - P1s3_P3s2_134.0
#  - P1s3_P3s2_124.0
#  - P1s3_P3s3_123.0
#  - P1s3_P3s3_12.0
#  - P1s3_P3s3_34.0
#  - P1s3_P3s3_1234.0
#  - P1s3_P3s3_234.0
#  - P1s3_P3s3_23.0
#  - P1s3_P3s3_13.0
#  - P1s3_P3s3_134.0
#  - P1s3_P3s3_124.0
#  - P1s3_P3s4_123.0
#  - P1s3_P3s4_12.0
#  - P1s3_P3s4_34.0
#  - P1s3_P3s4_1234.0
#  - P1s3_P3s4_234.0
#  - P1s3_P3s4_23.0
#  - P1s3_P3s4_13.0
#  - P1s3_P3s4_134.0
#  - P1s3_P3s4_124.0
#  - P1s4_P3s1_123.0
#  - P1s4_P3s1_12.0
#  - P1s4_P3s1_34.0
#  - P1s4_P3s1_1234.0
#  - P1s4_P3s1_234.0
#  - P1s4_P3s1_23.0
#  - P1s4_P3s1_13.0
#  - P1s4_P3s1_134.0
#  - P1s4_P3s1_124.0
#  - P1s4_P3s2_123.0
#  - P1s4_P3s2_12.0
#  - P1s4_P3s2_34.0
#  - P1s4_P3s2_1234.0
#  - P1s4_P3s2_234.0
#  - P1s4_P3s2_23.0
#  - P1s4_P3s2_13.0
#  - P1s4_P3s2_134.0
#  - P1s4_P3s2_124.0
#  - P1s4_P3s3_123.0
#  - P1s4_P3s3_12.0
#  - P1s4_P3s3_34.0
#  - P1s4_P3s3_1234.0
#  - P1s4_P3s3_234.0
#  - P1s4_P3s3_23.0
#  - P1s4_P3s3_13.0
#  - P1s4_P3s3_134.0
#  - P1s4_P3s3_124.0
#  - P1s4_P3s4_123.0
#  - P1s4_P3s4_12.0
#  - P1s4_P3s4_34.0
#  - P1s4_P3s4_1234.0
#  - P1s4_P3s4_234.0
#  - P1s4_P3s4_23.0
#  - P1s4_P3s4_13.0
#  - P1s4_P3s4_134.0
#  - P1s4_P3s4_124.0
#  - P1s1_P4s1_123.0
#  - P1s1_P4s1_12.0
#  - P1s1_P4s1_34.0
#  - P1s1_P4s1_1234.0
#  - P1s1_P4s1_234.0
#  - P1s1_P4s1_23.0
#  - P1s1_P4s1_13.0
#  - P1s1_P4s1_134.0
#  - P1s1_P4s1_124.0
#  - P1s1_P4s2_123.0
#  - P1s1_P4s2_12.0
#  - P1s1_P4s2_34.0
#  - P1s1_P4s2_1234.0
#  - P1s1_P4s2_234.0
#  - P1s1_P4s2_23.0
#  - P1s1_P4s2_13.0
#  - P1s1_P4s2_134.0
#  - P1s1_P4s2_124.0
#  - P1s1_P4s3_123.0
#  - P1s1_P4s3_12.0
#  - P1s1_P4s3_34.0
#  - P1s1_P4s3_1234.0
#  - P1s1_P4s3_234.0
#  - P1s1_P4s3_23.0
#  - P1s1_P4s3_13.0
#  - P1s1_P4s3_134.0
#  - P1s1_P4s3_124.0
#  - P1s1_P4s4_123.0
#  - P1s1_P4s4_12.0
#  - P1s1_P4s4_34.0
#  - P1s1_P4s4_1234.0
#  - P1s1_P4s4_234.0
#  - P1s1_P4s4_23.0
#  - P1s1_P4s4_13.0
#  - P1s1_P4s4_134.0
#  - P1s1_P4s4_124.0
#  - P1s2_P4s1_123.0
#  - P1s2_P4s1_12.0
#  - P1s2_P4s1_34.0
#  - P1s2_P4s1_1234.0
#  - P1s2_P4s1_234.0
#  - P1s2_P4s1_23.0
#  - P1s2_P4s1_13.0
#  - P1s2_P4s1_134.0
#  - P1s2_P4s1_124.0
#  - P1s2_P4s2_123.0
#  - P1s2_P4s2_12.0
#  - P1s2_P4s2_34.0
#  - P1s2_P4s2_1234.0
#  - P1s2_P4s2_234.0
#  - P1s2_P4s2_23.0
#  - P1s2_P4s2_13.0
#  - P1s2_P4s2_134.0
#  - P1s2_P4s2_124.0
#  - P1s2_P4s3_123.0
#  - P1s2_P4s3_12.0
#  - P1s2_P4s3_34.0
#  - P1s2_P4s3_1234.0
#  - P1s2_P4s3_234.0
#  - P1s2_P4s3_23.0
#  - P1s2_P4s3_13.0
#  - P1s2_P4s3_134.0
#  - P1s2_P4s3_124.0
#  - P1s2_P4s4_123.0
#  - P1s2_P4s4_12.0
#  - P1s2_P4s4_34.0
#  - P1s2_P4s4_1234.0
#  - P1s2_P4s4_234.0
#  - P1s2_P4s4_23.0
#  - P1s2_P4s4_13.0
#  - P1s2_P4s4_134.0
#  - P1s2_P4s4_124.0
#  - P1s3_P4s1_123.0
#  - P1s3_P4s1_12.0
#  - P1s3_P4s1_34.0
#  - P1s3_P4s1_1234.0
#  - P1s3_P4s1_234.0
#  - P1s3_P4s1_23.0
#  - P1s3_P4s1_13.0
#  - P1s3_P4s1_134.0
#  - P1s3_P4s1_124.0
#  - P1s3_P4s2_123.0
#  - P1s3_P4s2_12.0
#  - P1s3_P4s2_34.0
#  - P1s3_P4s2_1234.0
#  - P1s3_P4s2_234.0
#  - P1s3_P4s2_23.0
#  - P1s3_P4s2_13.0
#  - P1s3_P4s2_134.0
#  - P1s3_P4s2_124.0
#  - P1s3_P4s3_123.0
#  - P1s3_P4s3_12.0
#  - P1s3_P4s3_34.0
#  - P1s3_P4s3_1234.0
#  - P1s3_P4s3_234.0
#  - P1s3_P4s3_23.0
#  - P1s3_P4s3_13.0
#  - P1s3_P4s3_134.0
#  - P1s3_P4s3_124.0
#  - P1s3_P4s4_123.0
#  - P1s3_P4s4_12.0
#  - P1s3_P4s4_34.0
#  - P1s3_P4s4_1234.0
#  - P1s3_P4s4_234.0
#  - P1s3_P4s4_23.0
#  - P1s3_P4s4_13.0
#  - P1s3_P4s4_134.0
#  - P1s3_P4s4_124.0
#  - P1s4_P4s1_123.0
#  - P1s4_P4s1_12.0
#  - P1s4_P4s1_34.0
#  - P1s4_P4s1_1234.0
#  - P1s4_P4s1_234.0
#  - P1s4_P4s1_23.0
#  - P1s4_P4s1_13.0
#  - P1s4_P4s1_134.0
#  - P1s4_P4s1_124.0
#  - P1s4_P4s2_123.0
#  - P1s4_P4s2_12.0
#  - P1s4_P4s2_34.0
#  - P1s4_P4s2_1234.0
#  - P1s4_P4s2_234.0
#  - P1s4_P4s2_23.0
#  - P1s4_P4s2_13.0
#  - P1s4_P4s2_134.0
#  - P1s4_P4s2_124.0
#  - P1s4_P4s3_123.0
#  - P1s4_P4s3_12.0
#  - P1s4_P4s3_34.0
#  - P1s4_P4s3_1234.0
#  - P1s4_P4s3_234.0
#  - P1s4_P4s3_23.0
#  - P1s4_P4s3_13.0
#  - P1s4_P4s3_134.0
#  - P1s4_P4s3_124.0
#  - P1s4_P4s4_123.0
#  - P1s4_P4s4_12.0
#  - P1s4_P4s4_34.0
#  - P1s4_P4s4_1234.0
#  - P1s4_P4s4_234.0
#  - P1s4_P4s4_23.0
#  - P1s4_P4s4_13.0
#  - P1s4_P4s4_134.0
#  - P1s4_P4s4_124.0
#  - P2s1_P1s1_123.0
#  - P2s1_P1s1_12.0
#  - P2s1_P1s1_34.0
#  - P2s1_P1s1_1234.0
#  - P2s1_P1s1_234.0
#  - P2s1_P1s1_23.0
#  - P2s1_P1s1_13.0
#  - P2s1_P1s1_134.0
#  - P2s1_P1s1_124.0
#  - P2s1_P1s2_123.0
#  - P2s1_P1s2_12.0
#  - P2s1_P1s2_34.0
#  - P2s1_P1s2_1234.0
#  - P2s1_P1s2_234.0
#  - P2s1_P1s2_23.0
#  - P2s1_P1s2_13.0
#  - P2s1_P1s2_134.0
#  - P2s1_P1s2_124.0
#  - P2s1_P1s3_123.0
#  - P2s1_P1s3_12.0
#  - P2s1_P1s3_34.0
#  - P2s1_P1s3_1234.0
#  - P2s1_P1s3_234.0
#  - P2s1_P1s3_23.0
#  - P2s1_P1s3_13.0
#  - P2s1_P1s3_134.0
#  - P2s1_P1s3_124.0
#  - P2s1_P1s4_123.0
#  - P2s1_P1s4_12.0
#  - P2s1_P1s4_34.0
#  - P2s1_P1s4_1234.0
#  - P2s1_P1s4_234.0
#  - P2s1_P1s4_23.0
#  - P2s1_P1s4_13.0
#  - P2s1_P1s4_134.0
#  - P2s1_P1s4_124.0
#  - P2s2_P1s2_123.0
#  - P2s2_P1s2_12.0
#  - P2s2_P1s2_34.0
#  - P2s2_P1s2_1234.0
#  - P2s2_P1s2_234.0
#  - P2s2_P1s2_23.0
#  - P2s2_P1s2_13.0
#  - P2s2_P1s2_134.0
#  - P2s2_P1s2_124.0
#  - P2s2_P1s3_123.0
#  - P2s2_P1s3_12.0
#  - P2s2_P1s3_34.0
#  - P2s2_P1s3_1234.0
#  - P2s2_P1s3_234.0
#  - P2s2_P1s3_23.0
#  - P2s2_P1s3_13.0
#  - P2s2_P1s3_134.0
#  - P2s2_P1s3_124.0
#  - P2s2_P1s4_123.0
#  - P2s2_P1s4_12.0
#  - P2s2_P1s4_34.0
#  - P2s2_P1s4_1234.0
#  - P2s2_P1s4_234.0
#  - P2s2_P1s4_23.0
#  - P2s2_P1s4_13.0
#  - P2s2_P1s4_134.0
#  - P2s2_P1s4_124.0
#  - P2s3_P1s3_123.0
#  - P2s3_P1s3_12.0
#  - P2s3_P1s3_34.0
#  - P2s3_P1s3_1234.0
#  - P2s3_P1s3_234.0
#  - P2s3_P1s3_23.0
#  - P2s3_P1s3_13.0
#  - P2s3_P1s3_134.0
#  - P2s3_P1s3_124.0
#  - P2s3_P1s4_123.0
#  - P2s3_P1s4_12.0
#  - P2s3_P1s4_34.0
#  - P2s3_P1s4_1234.0
#  - P2s3_P1s4_234.0
#  - P2s3_P1s4_23.0
#  - P2s3_P1s4_13.0
#  - P2s3_P1s4_134.0
#  - P2s3_P1s4_124.0
#  - P2s4_P1s4_123.0
#  - P2s4_P1s4_12.0
#  - P2s4_P1s4_34.0
#  - P2s4_P1s4_1234.0
#  - P2s4_P1s4_234.0
#  - P2s4_P1s4_23.0
#  - P2s4_P1s4_13.0
#  - P2s4_P1s4_134.0
#  - P2s4_P1s4_124.0
#  - P2s1_P2s2_123.0
#  - P2s1_P2s2_12.0
#  - P2s1_P2s2_34.0
#  - P2s1_P2s2_1234.0
#  - P2s1_P2s2_234.0
#  - P2s1_P2s2_23.0
#  - P2s1_P2s2_13.0
#  - P2s1_P2s2_134.0
#  - P2s1_P2s2_124.0
#  - P2s1_P2s3_123.0
#  - P2s1_P2s3_12.0
#  - P2s1_P2s3_34.0
#  - P2s1_P2s3_1234.0
#  - P2s1_P2s3_234.0
#  - P2s1_P2s3_23.0
#  - P2s1_P2s3_13.0
#  - P2s1_P2s3_134.0
#  - P2s1_P2s3_124.0
#  - P2s1_P2s4_123.0
#  - P2s1_P2s4_12.0
#  - P2s1_P2s4_34.0
#  - P2s1_P2s4_1234.0
#  - P2s1_P2s4_234.0
#  - P2s1_P2s4_23.0
#  - P2s1_P2s4_13.0
#  - P2s1_P2s4_134.0
#  - P2s1_P2s4_124.0
#  - P2s2_P2s1_123.0
#  - P2s2_P2s1_12.0
#  - P2s2_P2s1_34.0
#  - P2s2_P2s1_1234.0
#  - P2s2_P2s1_234.0
#  - P2s2_P2s1_23.0
#  - P2s2_P2s1_13.0
#  - P2s2_P2s1_134.0
#  - P2s2_P2s1_124.0
#  - P2s2_P2s3_123.0
#  - P2s2_P2s3_12.0
#  - P2s2_P2s3_34.0
#  - P2s2_P2s3_1234.0
#  - P2s2_P2s3_234.0
#  - P2s2_P2s3_23.0
#  - P2s2_P2s3_13.0
#  - P2s2_P2s3_134.0
#  - P2s2_P2s3_124.0
#  - P2s2_P2s4_123.0
#  - P2s2_P2s4_12.0
#  - P2s2_P2s4_34.0
#  - P2s2_P2s4_1234.0
#  - P2s2_P2s4_234.0
#  - P2s2_P2s4_23.0
#  - P2s2_P2s4_13.0
#  - P2s2_P2s4_134.0
#  - P2s2_P2s4_124.0
#  - P2s3_P2s1_123.0
#  - P2s3_P2s1_12.0
#  - P2s3_P2s1_34.0
#  - P2s3_P2s1_1234.0
#  - P2s3_P2s1_234.0
#  - P2s3_P2s1_23.0
#  - P2s3_P2s1_13.0
#  - P2s3_P2s1_134.0
#  - P2s3_P2s1_124.0
#  - P2s3_P2s2_123.0
#  - P2s3_P2s2_12.0
#  - P2s3_P2s2_34.0
#  - P2s3_P2s2_1234.0
#  - P2s3_P2s2_234.0
#  - P2s3_P2s2_23.0
#  - P2s3_P2s2_13.0
#  - P2s3_P2s2_134.0
#  - P2s3_P2s2_124.0
#  - P2s3_P2s4_123.0
#  - P2s3_P2s4_12.0
#  - P2s3_P2s4_34.0
#  - P2s3_P2s4_1234.0
#  - P2s3_P2s4_234.0
#  - P2s3_P2s4_23.0
#  - P2s3_P2s4_13.0
#  - P2s3_P2s4_134.0
#  - P2s3_P2s4_124.0
#  - P2s4_P2s1_123.0
#  - P2s4_P2s1_12.0
#  - P2s4_P2s1_34.0
#  - P2s4_P2s1_1234.0
#  - P2s4_P2s1_234.0
#  - P2s4_P2s1_23.0
#  - P2s4_P2s1_13.0
#  - P2s4_P2s1_134.0
#  - P2s4_P2s1_124.0
#  - P2s4_P2s2_123.0
#  - P2s4_P2s2_12.0
#  - P2s4_P2s2_34.0
#  - P2s4_P2s2_1234.0
#  - P2s4_P2s2_234.0
#  - P2s4_P2s2_23.0
#  - P2s4_P2s2_13.0
#  - P2s4_P2s2_134.0
#  - P2s4_P2s2_124.0
#  - P2s4_P2s3_123.0
#  - P2s4_P2s3_12.0
#  - P2s4_P2s3_34.0
#  - P2s4_P2s3_1234.0
#  - P2s4_P2s3_234.0
#  - P2s4_P2s3_23.0
#  - P2s4_P2s3_13.0
#  - P2s4_P2s3_134.0
#  - P2s4_P2s3_124.0
#  - P2s1_P3s1_123.0
#  - P2s1_P3s1_12.0
#  - P2s1_P3s1_34.0
#  - P2s1_P3s1_1234.0
#  - P2s1_P3s1_234.0
#  - P2s1_P3s1_23.0
#  - P2s1_P3s1_13.0
#  - P2s1_P3s1_134.0
#  - P2s1_P3s1_124.0
#  - P2s1_P3s2_123.0
#  - P2s1_P3s2_12.0
#  - P2s1_P3s2_34.0
#  - P2s1_P3s2_1234.0
#  - P2s1_P3s2_234.0
#  - P2s1_P3s2_23.0
#  - P2s1_P3s2_13.0
#  - P2s1_P3s2_134.0
#  - P2s1_P3s2_124.0
#  - P2s1_P3s3_123.0
#  - P2s1_P3s3_12.0
#  - P2s1_P3s3_34.0
#  - P2s1_P3s3_1234.0
#  - P2s1_P3s3_234.0
#  - P2s1_P3s3_23.0
#  - P2s1_P3s3_13.0
#  - P2s1_P3s3_134.0
#  - P2s1_P3s3_124.0
#  - P2s1_P3s4_123.0
#  - P2s1_P3s4_12.0
#  - P2s1_P3s4_34.0
#  - P2s1_P3s4_1234.0
#  - P2s1_P3s4_234.0
#  - P2s1_P3s4_23.0
#  - P2s1_P3s4_13.0
#  - P2s1_P3s4_134.0
#  - P2s1_P3s4_124.0
#  - P2s2_P3s1_123.0
#  - P2s2_P3s1_12.0
#  - P2s2_P3s1_34.0
#  - P2s2_P3s1_1234.0
#  - P2s2_P3s1_234.0
#  - P2s2_P3s1_23.0
#  - P2s2_P3s1_13.0
#  - P2s2_P3s1_134.0
#  - P2s2_P3s1_124.0
#  - P2s2_P3s2_123.0
#  - P2s2_P3s2_12.0
#  - P2s2_P3s2_34.0
#  - P2s2_P3s2_1234.0
#  - P2s2_P3s2_234.0
#  - P2s2_P3s2_23.0
#  - P2s2_P3s2_13.0
#  - P2s2_P3s2_134.0
#  - P2s2_P3s2_124.0
#  - P2s2_P3s3_123.0
#  - P2s2_P3s3_12.0
#  - P2s2_P3s3_34.0
#  - P2s2_P3s3_1234.0
#  - P2s2_P3s3_234.0
#  - P2s2_P3s3_23.0
#  - P2s2_P3s3_13.0
#  - P2s2_P3s3_134.0
#  - P2s2_P3s3_124.0
#  - P2s2_P3s4_123.0
#  - P2s2_P3s4_12.0
#  - P2s2_P3s4_34.0
#  - P2s2_P3s4_1234.0
#  - P2s2_P3s4_234.0
#  - P2s2_P3s4_23.0
#  - P2s2_P3s4_13.0
#  - P2s2_P3s4_134.0
#  - P2s2_P3s4_124.0
#  - P2s3_P3s1_123.0
#  - P2s3_P3s1_12.0
#  - P2s3_P3s1_34.0
#  - P2s3_P3s1_1234.0
#  - P2s3_P3s1_234.0
#  - P2s3_P3s1_23.0
#  - P2s3_P3s1_13.0
#  - P2s3_P3s1_134.0
#  - P2s3_P3s1_124.0
#  - P2s3_P3s2_123.0
#  - P2s3_P3s2_12.0
#  - P2s3_P3s2_34.0
#  - P2s3_P3s2_1234.0
#  - P2s3_P3s2_234.0
#  - P2s3_P3s2_23.0
#  - P2s3_P3s2_13.0
#  - P2s3_P3s2_134.0
#  - P2s3_P3s2_124.0
#  - P2s3_P3s3_123.0
#  - P2s3_P3s3_12.0
#  - P2s3_P3s3_34.0
#  - P2s3_P3s3_1234.0
#  - P2s3_P3s3_234.0
#  - P2s3_P3s3_23.0
#  - P2s3_P3s3_13.0
#  - P2s3_P3s3_134.0
#  - P2s3_P3s3_124.0
#  - P2s3_P3s4_123.0
#  - P2s3_P3s4_12.0
#  - P2s3_P3s4_34.0
#  - P2s3_P3s4_1234.0
#  - P2s3_P3s4_234.0
#  - P2s3_P3s4_23.0
#  - P2s3_P3s4_13.0
#  - P2s3_P3s4_134.0
#  - P2s3_P3s4_124.0
#  - P2s4_P3s1_123.0
#  - P2s4_P3s1_12.0
#  - P2s4_P3s1_34.0
#  - P2s4_P3s1_1234.0
#  - P2s4_P3s1_234.0
#  - P2s4_P3s1_23.0
#  - P2s4_P3s1_13.0
#  - P2s4_P3s1_134.0
#  - P2s4_P3s1_124.0
#  - P2s4_P3s2_123.0
#  - P2s4_P3s2_12.0
#  - P2s4_P3s2_34.0
#  - P2s4_P3s2_1234.0
#  - P2s4_P3s2_234.0
#  - P2s4_P3s2_23.0
#  - P2s4_P3s2_13.0
#  - P2s4_P3s2_134.0
#  - P2s4_P3s2_124.0
#  - P2s4_P3s3_123.0
#  - P2s4_P3s3_12.0
#  - P2s4_P3s3_34.0
#  - P2s4_P3s3_1234.0
#  - P2s4_P3s3_234.0
#  - P2s4_P3s3_23.0
#  - P2s4_P3s3_13.0
#  - P2s4_P3s3_134.0
#  - P2s4_P3s3_124.0
#  - P2s4_P3s4_123.0
#  - P2s4_P3s4_12.0
#  - P2s4_P3s4_34.0
#  - P2s4_P3s4_1234.0
#  - P2s4_P3s4_234.0
#  - P2s4_P3s4_23.0
#  - P2s4_P3s4_13.0
#  - P2s4_P3s4_134.0
#  - P2s4_P3s4_124.0
#  - P2s1_P4s1_123.0
#  - P2s1_P4s1_12.0
#  - P2s1_P4s1_34.0
#  - P2s1_P4s1_1234.0
#  - P2s1_P4s1_234.0
#  - P2s1_P4s1_23.0
#  - P2s1_P4s1_13.0
#  - P2s1_P4s1_134.0
#  - P2s1_P4s1_124.0
#  - P2s1_P4s2_123.0
#  - P2s1_P4s2_12.0
#  - P2s1_P4s2_34.0
#  - P2s1_P4s2_1234.0
#  - P2s1_P4s2_234.0
#  - P2s1_P4s2_23.0
#  - P2s1_P4s2_13.0
#  - P2s1_P4s2_134.0
#  - P2s1_P4s2_124.0
#  - P2s1_P4s3_123.0
#  - P2s1_P4s3_12.0
#  - P2s1_P4s3_34.0
#  - P2s1_P4s3_1234.0
#  - P2s1_P4s3_234.0
#  - P2s1_P4s3_23.0
#  - P2s1_P4s3_13.0
#  - P2s1_P4s3_134.0
#  - P2s1_P4s3_124.0
#  - P2s1_P4s4_123.0
#  - P2s1_P4s4_12.0
#  - P2s1_P4s4_34.0
#  - P2s1_P4s4_1234.0
#  - P2s1_P4s4_234.0
#  - P2s1_P4s4_23.0
#  - P2s1_P4s4_13.0
#  - P2s1_P4s4_134.0
#  - P2s1_P4s4_124.0
#  - P2s2_P4s1_123.0
#  - P2s2_P4s1_12.0
#  - P2s2_P4s1_34.0
#  - P2s2_P4s1_1234.0
#  - P2s2_P4s1_234.0
#  - P2s2_P4s1_23.0
#  - P2s2_P4s1_13.0
#  - P2s2_P4s1_134.0
#  - P2s2_P4s1_124.0
#  - P2s2_P4s2_123.0
#  - P2s2_P4s2_12.0
#  - P2s2_P4s2_34.0
#  - P2s2_P4s2_1234.0
#  - P2s2_P4s2_234.0
#  - P2s2_P4s2_23.0
#  - P2s2_P4s2_13.0
#  - P2s2_P4s2_134.0
#  - P2s2_P4s2_124.0
#  - P2s2_P4s3_123.0
#  - P2s2_P4s3_12.0
#  - P2s2_P4s3_34.0
#  - P2s2_P4s3_1234.0
#  - P2s2_P4s3_234.0
#  - P2s2_P4s3_23.0
#  - P2s2_P4s3_13.0
#  - P2s2_P4s3_134.0
#  - P2s2_P4s3_124.0
#  - P2s2_P4s4_123.0
#  - P2s2_P4s4_12.0
#  - P2s2_P4s4_34.0
#  - P2s2_P4s4_1234.0
#  - P2s2_P4s4_234.0
#  - P2s2_P4s4_23.0
#  - P2s2_P4s4_13.0
#  - P2s2_P4s4_134.0
#  - P2s2_P4s4_124.0
#  - P2s3_P4s1_123.0
#  - P2s3_P4s1_12.0
#  - P2s3_P4s1_34.0
#  - P2s3_P4s1_1234.0
#  - P2s3_P4s1_234.0
#  - P2s3_P4s1_23.0
#  - P2s3_P4s1_13.0
#  - P2s3_P4s1_134.0
#  - P2s3_P4s1_124.0
#  - P2s3_P4s2_123.0
#  - P2s3_P4s2_12.0
#  - P2s3_P4s2_34.0
#  - P2s3_P4s2_1234.0
#  - P2s3_P4s2_234.0
#  - P2s3_P4s2_23.0
#  - P2s3_P4s2_13.0
#  - P2s3_P4s2_134.0
#  - P2s3_P4s2_124.0
#  - P2s3_P4s3_123.0
#  - P2s3_P4s3_12.0
#  - P2s3_P4s3_34.0
#  - P2s3_P4s3_1234.0
#  - P2s3_P4s3_234.0
#  - P2s3_P4s3_23.0
#  - P2s3_P4s3_13.0
#  - P2s3_P4s3_134.0
#  - P2s3_P4s3_124.0
#  - P2s3_P4s4_123.0
#  - P2s3_P4s4_12.0
#  - P2s3_P4s4_34.0
#  - P2s3_P4s4_1234.0
#  - P2s3_P4s4_234.0
#  - P2s3_P4s4_23.0
#  - P2s3_P4s4_13.0
#  - P2s3_P4s4_134.0
#  - P2s3_P4s4_124.0
#  - P2s4_P4s1_123.0
#  - P2s4_P4s1_12.0
#  - P2s4_P4s1_34.0
#  - P2s4_P4s1_1234.0
#  - P2s4_P4s1_234.0
#  - P2s4_P4s1_23.0
#  - P2s4_P4s1_13.0
#  - P2s4_P4s1_134.0
#  - P2s4_P4s1_124.0
#  - P2s4_P4s2_123.0
#  - P2s4_P4s2_12.0
#  - P2s4_P4s2_34.0
#  - P2s4_P4s2_1234.0
#  - P2s4_P4s2_234.0
#  - P2s4_P4s2_23.0
#  - P2s4_P4s2_13.0
#  - P2s4_P4s2_134.0
#  - P2s4_P4s2_124.0
#  - P2s4_P4s3_123.0
#  - P2s4_P4s3_12.0
#  - P2s4_P4s3_34.0
#  - P2s4_P4s3_1234.0
#  - P2s4_P4s3_234.0
#  - P2s4_P4s3_23.0
#  - P2s4_P4s3_13.0
#  - P2s4_P4s3_134.0
#  - P2s4_P4s3_124.0
#  - P2s4_P4s4_123.0
#  - P2s4_P4s4_12.0
#  - P2s4_P4s4_34.0
#  - P2s4_P4s4_1234.0
#  - P2s4_P4s4_234.0
#  - P2s4_P4s4_23.0
#  - P2s4_P4s4_13.0
#  - P2s4_P4s4_134.0
#  - P2s4_P4s4_124.0
#  - P3s1_P1s1_123.0
#  - P3s1_P1s1_12.0
#  - P3s1_P1s1_34.0
#  - P3s1_P1s1_1234.0
#  - P3s1_P1s1_234.0
#  - P3s1_P1s1_23.0
#  - P3s1_P1s1_13.0
#  - P3s1_P1s1_134.0
#  - P3s1_P1s1_124.0
#  - P3s1_P1s2_123.0
#  - P3s1_P1s2_12.0
#  - P3s1_P1s2_34.0
#  - P3s1_P1s2_1234.0
#  - P3s1_P1s2_234.0
#  - P3s1_P1s2_23.0
#  - P3s1_P1s2_13.0
#  - P3s1_P1s2_134.0
#  - P3s1_P1s2_124.0
#  - P3s1_P1s3_123.0
#  - P3s1_P1s3_12.0
#  - P3s1_P1s3_34.0
#  - P3s1_P1s3_1234.0
#  - P3s1_P1s3_234.0
#  - P3s1_P1s3_23.0
#  - P3s1_P1s3_13.0
#  - P3s1_P1s3_134.0
#  - P3s1_P1s3_124.0
#  - P3s1_P1s4_123.0
#  - P3s1_P1s4_12.0
#  - P3s1_P1s4_34.0
#  - P3s1_P1s4_1234.0
#  - P3s1_P1s4_234.0
#  - P3s1_P1s4_23.0
#  - P3s1_P1s4_13.0
#  - P3s1_P1s4_134.0
#  - P3s1_P1s4_124.0
#  - P3s2_P1s2_123.0
#  - P3s2_P1s2_12.0
#  - P3s2_P1s2_34.0
#  - P3s2_P1s2_1234.0
#  - P3s2_P1s2_234.0
#  - P3s2_P1s2_23.0
#  - P3s2_P1s2_13.0
#  - P3s2_P1s2_134.0
#  - P3s2_P1s2_124.0
#  - P3s2_P1s3_123.0
#  - P3s2_P1s3_12.0
#  - P3s2_P1s3_34.0
#  - P3s2_P1s3_1234.0
#  - P3s2_P1s3_234.0
#  - P3s2_P1s3_23.0
#  - P3s2_P1s3_13.0
#  - P3s2_P1s3_134.0
#  - P3s2_P1s3_124.0
#  - P3s2_P1s4_123.0
#  - P3s2_P1s4_12.0
#  - P3s2_P1s4_34.0
#  - P3s2_P1s4_1234.0
#  - P3s2_P1s4_234.0
#  - P3s2_P1s4_23.0
#  - P3s2_P1s4_13.0
#  - P3s2_P1s4_134.0
#  - P3s2_P1s4_124.0
#  - P3s3_P1s3_123.0
#  - P3s3_P1s3_12.0
#  - P3s3_P1s3_34.0
#  - P3s3_P1s3_1234.0
#  - P3s3_P1s3_234.0
#  - P3s3_P1s3_23.0
#  - P3s3_P1s3_13.0
#  - P3s3_P1s3_134.0
#  - P3s3_P1s3_124.0
#  - P3s3_P1s4_123.0
#  - P3s3_P1s4_12.0
#  - P3s3_P1s4_34.0
#  - P3s3_P1s4_1234.0
#  - P3s3_P1s4_234.0
#  - P3s3_P1s4_23.0
#  - P3s3_P1s4_13.0
#  - P3s3_P1s4_134.0
#  - P3s3_P1s4_124.0
#  - P3s4_P1s4_123.0
#  - P3s4_P1s4_12.0
#  - P3s4_P1s4_34.0
#  - P3s4_P1s4_1234.0
#  - P3s4_P1s4_234.0
#  - P3s4_P1s4_23.0
#  - P3s4_P1s4_13.0
#  - P3s4_P1s4_134.0
#  - P3s4_P1s4_124.0
#  - P3s1_P2s1_123.0
#  - P3s1_P2s1_12.0
#  - P3s1_P2s1_34.0
#  - P3s1_P2s1_1234.0
#  - P3s1_P2s1_234.0
#  - P3s1_P2s1_23.0
#  - P3s1_P2s1_13.0
#  - P3s1_P2s1_134.0
#  - P3s1_P2s1_124.0
#  - P3s1_P2s2_123.0
#  - P3s1_P2s2_12.0
#  - P3s1_P2s2_34.0
#  - P3s1_P2s2_1234.0
#  - P3s1_P2s2_234.0
#  - P3s1_P2s2_23.0
#  - P3s1_P2s2_13.0
#  - P3s1_P2s2_134.0
#  - P3s1_P2s2_124.0
#  - P3s1_P2s3_123.0
#  - P3s1_P2s3_12.0
#  - P3s1_P2s3_34.0
#  - P3s1_P2s3_1234.0
#  - P3s1_P2s3_234.0
#  - P3s1_P2s3_23.0
#  - P3s1_P2s3_13.0
#  - P3s1_P2s3_134.0
#  - P3s1_P2s3_124.0
#  - P3s1_P2s4_123.0
#  - P3s1_P2s4_12.0
#  - P3s1_P2s4_34.0
#  - P3s1_P2s4_1234.0
#  - P3s1_P2s4_234.0
#  - P3s1_P2s4_23.0
#  - P3s1_P2s4_13.0
#  - P3s1_P2s4_134.0
#  - P3s1_P2s4_124.0
#  - P3s2_P2s2_123.0
#  - P3s2_P2s2_12.0
#  - P3s2_P2s2_34.0
#  - P3s2_P2s2_1234.0
#  - P3s2_P2s2_234.0
#  - P3s2_P2s2_23.0
#  - P3s2_P2s2_13.0
#  - P3s2_P2s2_134.0
#  - P3s2_P2s2_124.0
#  - P3s2_P2s3_123.0
#  - P3s2_P2s3_12.0
#  - P3s2_P2s3_34.0
#  - P3s2_P2s3_1234.0
#  - P3s2_P2s3_234.0
#  - P3s2_P2s3_23.0
#  - P3s2_P2s3_13.0
#  - P3s2_P2s3_134.0
#  - P3s2_P2s3_124.0
#  - P3s2_P2s4_123.0
#  - P3s2_P2s4_12.0
#  - P3s2_P2s4_34.0
#  - P3s2_P2s4_1234.0
#  - P3s2_P2s4_234.0
#  - P3s2_P2s4_23.0
#  - P3s2_P2s4_13.0
#  - P3s2_P2s4_134.0
#  - P3s2_P2s4_124.0
#  - P3s3_P2s3_123.0
#  - P3s3_P2s3_12.0
#  - P3s3_P2s3_34.0
#  - P3s3_P2s3_1234.0
#  - P3s3_P2s3_234.0
#  - P3s3_P2s3_23.0
#  - P3s3_P2s3_13.0
#  - P3s3_P2s3_134.0
#  - P3s3_P2s3_124.0
#  - P3s3_P2s4_123.0
#  - P3s3_P2s4_12.0
#  - P3s3_P2s4_34.0
#  - P3s3_P2s4_1234.0
#  - P3s3_P2s4_234.0
#  - P3s3_P2s4_23.0
#  - P3s3_P2s4_13.0
#  - P3s3_P2s4_134.0
#  - P3s3_P2s4_124.0
#  - P3s4_P2s4_123.0
#  - P3s4_P2s4_12.0
#  - P3s4_P2s4_34.0
#  - P3s4_P2s4_1234.0
#  - P3s4_P2s4_234.0
#  - P3s4_P2s4_23.0
#  - P3s4_P2s4_13.0
#  - P3s4_P2s4_134.0
#  - P3s4_P2s4_124.0
#  - P3s1_P3s2_123.0
#  - P3s1_P3s2_12.0
#  - P3s1_P3s2_34.0
#  - P3s1_P3s2_1234.0
#  - P3s1_P3s2_234.0
#  - P3s1_P3s2_23.0
#  - P3s1_P3s2_13.0
#  - P3s1_P3s2_134.0
#  - P3s1_P3s2_124.0
#  - P3s1_P3s3_123.0
#  - P3s1_P3s3_12.0
#  - P3s1_P3s3_34.0
#  - P3s1_P3s3_1234.0
#  - P3s1_P3s3_234.0
#  - P3s1_P3s3_23.0
#  - P3s1_P3s3_13.0
#  - P3s1_P3s3_134.0
#  - P3s1_P3s3_124.0
#  - P3s1_P3s4_123.0
#  - P3s1_P3s4_12.0
#  - P3s1_P3s4_34.0
#  - P3s1_P3s4_1234.0
#  - P3s1_P3s4_234.0
#  - P3s1_P3s4_23.0
#  - P3s1_P3s4_13.0
#  - P3s1_P3s4_134.0
#  - P3s1_P3s4_124.0
#  - P3s2_P3s1_123.0
#  - P3s2_P3s1_12.0
#  - P3s2_P3s1_34.0
#  - P3s2_P3s1_1234.0
#  - P3s2_P3s1_234.0
#  - P3s2_P3s1_23.0
#  - P3s2_P3s1_13.0
#  - P3s2_P3s1_134.0
#  - P3s2_P3s1_124.0
#  - P3s2_P3s3_123.0
#  - P3s2_P3s3_12.0
#  - P3s2_P3s3_34.0
#  - P3s2_P3s3_1234.0
#  - P3s2_P3s3_234.0
#  - P3s2_P3s3_23.0
#  - P3s2_P3s3_13.0
#  - P3s2_P3s3_134.0
#  - P3s2_P3s3_124.0
#  - P3s2_P3s4_123.0
#  - P3s2_P3s4_12.0
#  - P3s2_P3s4_34.0
#  - P3s2_P3s4_1234.0
#  - P3s2_P3s4_234.0
#  - P3s2_P3s4_23.0
#  - P3s2_P3s4_13.0
#  - P3s2_P3s4_134.0
#  - P3s2_P3s4_124.0
#  - P3s3_P3s1_123.0
#  - P3s3_P3s1_12.0
#  - P3s3_P3s1_34.0
#  - P3s3_P3s1_1234.0
#  - P3s3_P3s1_234.0
#  - P3s3_P3s1_23.0
#  - P3s3_P3s1_13.0
#  - P3s3_P3s1_134.0
#  - P3s3_P3s1_124.0
#  - P3s3_P3s2_123.0
#  - P3s3_P3s2_12.0
#  - P3s3_P3s2_34.0
#  - P3s3_P3s2_1234.0
#  - P3s3_P3s2_234.0
#  - P3s3_P3s2_23.0
#  - P3s3_P3s2_13.0
#  - P3s3_P3s2_134.0
#  - P3s3_P3s2_124.0
#  - P3s3_P3s4_123.0
#  - P3s3_P3s4_12.0
#  - P3s3_P3s4_34.0
#  - P3s3_P3s4_1234.0
#  - P3s3_P3s4_234.0
#  - P3s3_P3s4_23.0
#  - P3s3_P3s4_13.0
#  - P3s3_P3s4_134.0
#  - P3s3_P3s4_124.0
#  - P3s4_P3s1_123.0
#  - P3s4_P3s1_12.0
#  - P3s4_P3s1_34.0
#  - P3s4_P3s1_1234.0
#  - P3s4_P3s1_234.0
#  - P3s4_P3s1_23.0
#  - P3s4_P3s1_13.0
#  - P3s4_P3s1_134.0
#  - P3s4_P3s1_124.0
#  - P3s4_P3s2_123.0
#  - P3s4_P3s2_12.0
#  - P3s4_P3s2_34.0
#  - P3s4_P3s2_1234.0
#  - P3s4_P3s2_234.0
#  - P3s4_P3s2_23.0
#  - P3s4_P3s2_13.0
#  - P3s4_P3s2_134.0
#  - P3s4_P3s2_124.0
#  - P3s4_P3s3_123.0
#  - P3s4_P3s3_12.0
#  - P3s4_P3s3_34.0
#  - P3s4_P3s3_1234.0
#  - P3s4_P3s3_234.0
#  - P3s4_P3s3_23.0
#  - P3s4_P3s3_13.0
#  - P3s4_P3s3_134.0
#  - P3s4_P3s3_124.0
#  - P3s1_P4s1_123.0
#  - P3s1_P4s1_12.0
#  - P3s1_P4s1_34.0
#  - P3s1_P4s1_1234.0
#  - P3s1_P4s1_234.0
#  - P3s1_P4s1_23.0
#  - P3s1_P4s1_13.0
#  - P3s1_P4s1_134.0
#  - P3s1_P4s1_124.0
#  - P3s1_P4s2_123.0
#  - P3s1_P4s2_12.0
#  - P3s1_P4s2_34.0
#  - P3s1_P4s2_1234.0
#  - P3s1_P4s2_234.0
#  - P3s1_P4s2_23.0
#  - P3s1_P4s2_13.0
#  - P3s1_P4s2_134.0
#  - P3s1_P4s2_124.0
#  - P3s1_P4s3_123.0
#  - P3s1_P4s3_12.0
#  - P3s1_P4s3_34.0
#  - P3s1_P4s3_1234.0
#  - P3s1_P4s3_234.0
#  - P3s1_P4s3_23.0
#  - P3s1_P4s3_13.0
#  - P3s1_P4s3_134.0
#  - P3s1_P4s3_124.0
#  - P3s1_P4s4_123.0
#  - P3s1_P4s4_12.0
#  - P3s1_P4s4_34.0
#  - P3s1_P4s4_1234.0
#  - P3s1_P4s4_234.0
#  - P3s1_P4s4_23.0
#  - P3s1_P4s4_13.0
#  - P3s1_P4s4_134.0
#  - P3s1_P4s4_124.0
#  - P3s2_P4s1_123.0
#  - P3s2_P4s1_12.0
#  - P3s2_P4s1_34.0
#  - P3s2_P4s1_1234.0
#  - P3s2_P4s1_234.0
#  - P3s2_P4s1_23.0
#  - P3s2_P4s1_13.0
#  - P3s2_P4s1_134.0
#  - P3s2_P4s1_124.0
#  - P3s2_P4s2_123.0
#  - P3s2_P4s2_12.0
#  - P3s2_P4s2_34.0
#  - P3s2_P4s2_1234.0
#  - P3s2_P4s2_234.0
#  - P3s2_P4s2_23.0
#  - P3s2_P4s2_13.0
#  - P3s2_P4s2_134.0
#  - P3s2_P4s2_124.0
#  - P3s2_P4s3_123.0
#  - P3s2_P4s3_12.0
#  - P3s2_P4s3_34.0
#  - P3s2_P4s3_1234.0
#  - P3s2_P4s3_234.0
#  - P3s2_P4s3_23.0
#  - P3s2_P4s3_13.0
#  - P3s2_P4s3_134.0
#  - P3s2_P4s3_124.0
#  - P3s2_P4s4_123.0
#  - P3s2_P4s4_12.0
#  - P3s2_P4s4_34.0
#  - P3s2_P4s4_1234.0
#  - P3s2_P4s4_234.0
#  - P3s2_P4s4_23.0
#  - P3s2_P4s4_13.0
#  - P3s2_P4s4_134.0
#  - P3s2_P4s4_124.0
#  - P3s3_P4s1_123.0
#  - P3s3_P4s1_12.0
#  - P3s3_P4s1_34.0
#  - P3s3_P4s1_1234.0
#  - P3s3_P4s1_234.0
#  - P3s3_P4s1_23.0
#  - P3s3_P4s1_13.0
#  - P3s3_P4s1_134.0
#  - P3s3_P4s1_124.0
#  - P3s3_P4s2_123.0
#  - P3s3_P4s2_12.0
#  - P3s3_P4s2_34.0
#  - P3s3_P4s2_1234.0
#  - P3s3_P4s2_234.0
#  - P3s3_P4s2_23.0
#  - P3s3_P4s2_13.0
#  - P3s3_P4s2_134.0
#  - P3s3_P4s2_124.0
#  - P3s3_P4s3_123.0
#  - P3s3_P4s3_12.0
#  - P3s3_P4s3_34.0
#  - P3s3_P4s3_1234.0
#  - P3s3_P4s3_234.0
#  - P3s3_P4s3_23.0
#  - P3s3_P4s3_13.0
#  - P3s3_P4s3_134.0
#  - P3s3_P4s3_124.0
#  - P3s3_P4s4_123.0
#  - P3s3_P4s4_12.0
#  - P3s3_P4s4_34.0
#  - P3s3_P4s4_1234.0
#  - P3s3_P4s4_234.0
#  - P3s3_P4s4_23.0
#  - P3s3_P4s4_13.0
#  - P3s3_P4s4_134.0
#  - P3s3_P4s4_124.0
#  - P3s4_P4s1_123.0
#  - P3s4_P4s1_12.0
#  - P3s4_P4s1_34.0
#  - P3s4_P4s1_1234.0
#  - P3s4_P4s1_234.0
#  - P3s4_P4s1_23.0
#  - P3s4_P4s1_13.0
#  - P3s4_P4s1_134.0
#  - P3s4_P4s1_124.0
#  - P3s4_P4s2_123.0
#  - P3s4_P4s2_12.0
#  - P3s4_P4s2_34.0
#  - P3s4_P4s2_1234.0
#  - P3s4_P4s2_234.0
#  - P3s4_P4s2_23.0
#  - P3s4_P4s2_13.0
#  - P3s4_P4s2_134.0
#  - P3s4_P4s2_124.0
#  - P3s4_P4s3_123.0
#  - P3s4_P4s3_12.0
#  - P3s4_P4s3_34.0
#  - P3s4_P4s3_1234.0
#  - P3s4_P4s3_234.0
#  - P3s4_P4s3_23.0
#  - P3s4_P4s3_13.0
#  - P3s4_P4s3_134.0
#  - P3s4_P4s3_124.0
#  - P3s4_P4s4_123.0
#  - P3s4_P4s4_12.0
#  - P3s4_P4s4_34.0
#  - P3s4_P4s4_1234.0
#  - P3s4_P4s4_234.0
#  - P3s4_P4s4_23.0
#  - P3s4_P4s4_13.0
#  - P3s4_P4s4_134.0
#  - P3s4_P4s4_124.0
#  - P4s1_P1s1_123.0
#  - P4s1_P1s1_12.0
#  - P4s1_P1s1_34.0
#  - P4s1_P1s1_1234.0
#  - P4s1_P1s1_234.0
#  - P4s1_P1s1_23.0
#  - P4s1_P1s1_13.0
#  - P4s1_P1s1_134.0
#  - P4s1_P1s1_124.0
#  - P4s1_P1s2_123.0
#  - P4s1_P1s2_12.0
#  - P4s1_P1s2_34.0
#  - P4s1_P1s2_1234.0
#  - P4s1_P1s2_234.0
#  - P4s1_P1s2_23.0
#  - P4s1_P1s2_13.0
#  - P4s1_P1s2_134.0
#  - P4s1_P1s2_124.0
#  - P4s1_P1s3_123.0
#  - P4s1_P1s3_12.0
#  - P4s1_P1s3_34.0
#  - P4s1_P1s3_1234.0
#  - P4s1_P1s3_234.0
#  - P4s1_P1s3_23.0
#  - P4s1_P1s3_13.0
#  - P4s1_P1s3_134.0
#  - P4s1_P1s3_124.0
#  - P4s1_P1s4_123.0
#  - P4s1_P1s4_12.0
#  - P4s1_P1s4_34.0
#  - P4s1_P1s4_1234.0
#  - P4s1_P1s4_234.0
#  - P4s1_P1s4_23.0
#  - P4s1_P1s4_13.0
#  - P4s1_P1s4_134.0
#  - P4s1_P1s4_124.0
#  - P4s2_P1s2_123.0
#  - P4s2_P1s2_12.0
#  - P4s2_P1s2_34.0
#  - P4s2_P1s2_1234.0
#  - P4s2_P1s2_234.0
#  - P4s2_P1s2_23.0
#  - P4s2_P1s2_13.0
#  - P4s2_P1s2_134.0
#  - P4s2_P1s2_124.0
#  - P4s2_P1s3_123.0
#  - P4s2_P1s3_12.0
#  - P4s2_P1s3_34.0
#  - P4s2_P1s3_1234.0
#  - P4s2_P1s3_234.0
#  - P4s2_P1s3_23.0
#  - P4s2_P1s3_13.0
#  - P4s2_P1s3_134.0
#  - P4s2_P1s3_124.0
#  - P4s2_P1s4_123.0
#  - P4s2_P1s4_12.0
#  - P4s2_P1s4_34.0
#  - P4s2_P1s4_1234.0
#  - P4s2_P1s4_234.0
#  - P4s2_P1s4_23.0
#  - P4s2_P1s4_13.0
#  - P4s2_P1s4_134.0
#  - P4s2_P1s4_124.0
#  - P4s3_P1s3_123.0
#  - P4s3_P1s3_12.0
#  - P4s3_P1s3_34.0
#  - P4s3_P1s3_1234.0
#  - P4s3_P1s3_234.0
#  - P4s3_P1s3_23.0
#  - P4s3_P1s3_13.0
#  - P4s3_P1s3_134.0
#  - P4s3_P1s3_124.0
#  - P4s3_P1s4_123.0
#  - P4s3_P1s4_12.0
#  - P4s3_P1s4_34.0
#  - P4s3_P1s4_1234.0
#  - P4s3_P1s4_234.0
#  - P4s3_P1s4_23.0
#  - P4s3_P1s4_13.0
#  - P4s3_P1s4_134.0
#  - P4s3_P1s4_124.0
#  - P4s4_P1s4_123.0
#  - P4s4_P1s4_12.0
#  - P4s4_P1s4_34.0
#  - P4s4_P1s4_1234.0
#  - P4s4_P1s4_234.0
#  - P4s4_P1s4_23.0
#  - P4s4_P1s4_13.0
#  - P4s4_P1s4_134.0
#  - P4s4_P1s4_124.0
#  - P4s1_P2s1_123.0
#  - P4s1_P2s1_12.0
#  - P4s1_P2s1_34.0
#  - P4s1_P2s1_1234.0
#  - P4s1_P2s1_234.0
#  - P4s1_P2s1_23.0
#  - P4s1_P2s1_13.0
#  - P4s1_P2s1_134.0
#  - P4s1_P2s1_124.0
#  - P4s1_P2s2_123.0
#  - P4s1_P2s2_12.0
#  - P4s1_P2s2_34.0
#  - P4s1_P2s2_1234.0
#  - P4s1_P2s2_234.0
#  - P4s1_P2s2_23.0
#  - P4s1_P2s2_13.0
#  - P4s1_P2s2_134.0
#  - P4s1_P2s2_124.0
#  - P4s1_P2s3_123.0
#  - P4s1_P2s3_12.0
#  - P4s1_P2s3_34.0
#  - P4s1_P2s3_1234.0
#  - P4s1_P2s3_234.0
#  - P4s1_P2s3_23.0
#  - P4s1_P2s3_13.0
#  - P4s1_P2s3_134.0
#  - P4s1_P2s3_124.0
#  - P4s1_P2s4_123.0
#  - P4s1_P2s4_12.0
#  - P4s1_P2s4_34.0
#  - P4s1_P2s4_1234.0
#  - P4s1_P2s4_234.0
#  - P4s1_P2s4_23.0
#  - P4s1_P2s4_13.0
#  - P4s1_P2s4_134.0
#  - P4s1_P2s4_124.0
#  - P4s2_P2s2_123.0
#  - P4s2_P2s2_12.0
#  - P4s2_P2s2_34.0
#  - P4s2_P2s2_1234.0
#  - P4s2_P2s2_234.0
#  - P4s2_P2s2_23.0
#  - P4s2_P2s2_13.0
#  - P4s2_P2s2_134.0
#  - P4s2_P2s2_124.0
#  - P4s2_P2s3_123.0
#  - P4s2_P2s3_12.0
#  - P4s2_P2s3_34.0
#  - P4s2_P2s3_1234.0
#  - P4s2_P2s3_234.0
#  - P4s2_P2s3_23.0
#  - P4s2_P2s3_13.0
#  - P4s2_P2s3_134.0
#  - P4s2_P2s3_124.0
#  - P4s2_P2s4_123.0
#  - P4s2_P2s4_12.0
#  - P4s2_P2s4_34.0
#  - P4s2_P2s4_1234.0
#  - P4s2_P2s4_234.0
#  - P4s2_P2s4_23.0
#  - P4s2_P2s4_13.0
#  - P4s2_P2s4_134.0
#  - P4s2_P2s4_124.0
#  - P4s3_P2s3_123.0
#  - P4s3_P2s3_12.0
#  - P4s3_P2s3_34.0
#  - P4s3_P2s3_1234.0
#  - P4s3_P2s3_234.0
#  - P4s3_P2s3_23.0
#  - P4s3_P2s3_13.0
#  - P4s3_P2s3_134.0
#  - P4s3_P2s3_124.0
#  - P4s3_P2s4_123.0
#  - P4s3_P2s4_12.0
#  - P4s3_P2s4_34.0
#  - P4s3_P2s4_1234.0
#  - P4s3_P2s4_234.0
#  - P4s3_P2s4_23.0
#  - P4s3_P2s4_13.0
#  - P4s3_P2s4_134.0
#  - P4s3_P2s4_124.0
#  - P4s4_P2s4_123.0
#  - P4s4_P2s4_12.0
#  - P4s4_P2s4_34.0
#  - P4s4_P2s4_1234.0
#  - P4s4_P2s4_234.0
#  - P4s4_P2s4_23.0
#  - P4s4_P2s4_13.0
#  - P4s4_P2s4_134.0
#  - P4s4_P2s4_124.0
#  - P4s1_P3s1_123.0
#  - P4s1_P3s1_12.0
#  - P4s1_P3s1_34.0
#  - P4s1_P3s1_1234.0
#  - P4s1_P3s1_234.0
#  - P4s1_P3s1_23.0
#  - P4s1_P3s1_13.0
#  - P4s1_P3s1_134.0
#  - P4s1_P3s1_124.0
#  - P4s1_P3s2_123.0
#  - P4s1_P3s2_12.0
#  - P4s1_P3s2_34.0
#  - P4s1_P3s2_1234.0
#  - P4s1_P3s2_234.0
#  - P4s1_P3s2_23.0
#  - P4s1_P3s2_13.0
#  - P4s1_P3s2_134.0
#  - P4s1_P3s2_124.0
#  - P4s1_P3s3_123.0
#  - P4s1_P3s3_12.0
#  - P4s1_P3s3_34.0
#  - P4s1_P3s3_1234.0
#  - P4s1_P3s3_234.0
#  - P4s1_P3s3_23.0
#  - P4s1_P3s3_13.0
#  - P4s1_P3s3_134.0
#  - P4s1_P3s3_124.0
#  - P4s1_P3s4_123.0
#  - P4s1_P3s4_12.0
#  - P4s1_P3s4_34.0
#  - P4s1_P3s4_1234.0
#  - P4s1_P3s4_234.0
#  - P4s1_P3s4_23.0
#  - P4s1_P3s4_13.0
#  - P4s1_P3s4_134.0
#  - P4s1_P3s4_124.0
#  - P4s2_P3s2_123.0
#  - P4s2_P3s2_12.0
#  - P4s2_P3s2_34.0
#  - P4s2_P3s2_1234.0
#  - P4s2_P3s2_234.0
#  - P4s2_P3s2_23.0
#  - P4s2_P3s2_13.0
#  - P4s2_P3s2_134.0
#  - P4s2_P3s2_124.0
#  - P4s2_P3s3_123.0
#  - P4s2_P3s3_12.0
#  - P4s2_P3s3_34.0
#  - P4s2_P3s3_1234.0
#  - P4s2_P3s3_234.0
#  - P4s2_P3s3_23.0
#  - P4s2_P3s3_13.0
#  - P4s2_P3s3_134.0
#  - P4s2_P3s3_124.0
#  - P4s2_P3s4_123.0
#  - P4s2_P3s4_12.0
#  - P4s2_P3s4_34.0
#  - P4s2_P3s4_1234.0
#  - P4s2_P3s4_234.0
#  - P4s2_P3s4_23.0
#  - P4s2_P3s4_13.0
#  - P4s2_P3s4_134.0
#  - P4s2_P3s4_124.0
#  - P4s3_P3s3_123.0
#  - P4s3_P3s3_12.0
#  - P4s3_P3s3_34.0
#  - P4s3_P3s3_1234.0
#  - P4s3_P3s3_234.0
#  - P4s3_P3s3_23.0
#  - P4s3_P3s3_13.0
#  - P4s3_P3s3_134.0
#  - P4s3_P3s3_124.0
#  - P4s3_P3s4_123.0
#  - P4s3_P3s4_12.0
#  - P4s3_P3s4_34.0
#  - P4s3_P3s4_1234.0
#  - P4s3_P3s4_234.0
#  - P4s3_P3s4_23.0
#  - P4s3_P3s4_13.0
#  - P4s3_P3s4_134.0
#  - P4s3_P3s4_124.0
#  - P4s4_P3s4_123.0
#  - P4s4_P3s4_12.0
#  - P4s4_P3s4_34.0
#  - P4s4_P3s4_1234.0
#  - P4s4_P3s4_234.0
#  - P4s4_P3s4_23.0
#  - P4s4_P3s4_13.0
#  - P4s4_P3s4_134.0
#  - P4s4_P3s4_124.0
#  - P4s1_P4s2_123.0
#  - P4s1_P4s2_12.0
#  - P4s1_P4s2_34.0
#  - P4s1_P4s2_1234.0
#  - P4s1_P4s2_234.0
#  - P4s1_P4s2_23.0
#  - P4s1_P4s2_13.0
#  - P4s1_P4s2_134.0
#  - P4s1_P4s2_124.0
#  - P4s1_P4s3_123.0
#  - P4s1_P4s3_12.0
#  - P4s1_P4s3_34.0
#  - P4s1_P4s3_1234.0
#  - P4s1_P4s3_234.0
#  - P4s1_P4s3_23.0
#  - P4s1_P4s3_13.0
#  - P4s1_P4s3_134.0
#  - P4s1_P4s3_124.0
#  - P4s1_P4s4_123.0
#  - P4s1_P4s4_12.0
#  - P4s1_P4s4_34.0
#  - P4s1_P4s4_1234.0
#  - P4s1_P4s4_234.0
#  - P4s1_P4s4_23.0
#  - P4s1_P4s4_13.0
#  - P4s1_P4s4_134.0
#  - P4s1_P4s4_124.0
#  - P4s2_P4s1_123.0
#  - P4s2_P4s1_12.0
#  - P4s2_P4s1_34.0
#  - P4s2_P4s1_1234.0
#  - P4s2_P4s1_234.0
#  - P4s2_P4s1_23.0
#  - P4s2_P4s1_13.0
#  - P4s2_P4s1_134.0
#  - P4s2_P4s1_124.0
#  - P4s2_P4s3_123.0
#  - P4s2_P4s3_12.0
#  - P4s2_P4s3_34.0
#  - P4s2_P4s3_1234.0
#  - P4s2_P4s3_234.0
#  - P4s2_P4s3_23.0
#  - P4s2_P4s3_13.0
#  - P4s2_P4s3_134.0
#  - P4s2_P4s3_124.0
#  - P4s2_P4s4_123.0
#  - P4s2_P4s4_12.0
#  - P4s2_P4s4_34.0
#  - P4s2_P4s4_1234.0
#  - P4s2_P4s4_234.0
#  - P4s2_P4s4_23.0
#  - P4s2_P4s4_13.0
#  - P4s2_P4s4_134.0
#  - P4s2_P4s4_124.0
#  - P4s3_P4s1_123.0
#  - P4s3_P4s1_12.0
#  - P4s3_P4s1_34.0
#  - P4s3_P4s1_1234.0
#  - P4s3_P4s1_234.0
#  - P4s3_P4s1_23.0
#  - P4s3_P4s1_13.0
#  - P4s3_P4s1_134.0
#  - P4s3_P4s1_124.0
#  - P4s3_P4s2_123.0
#  - P4s3_P4s2_12.0
#  - P4s3_P4s2_34.0
#  - P4s3_P4s2_1234.0
#  - P4s3_P4s2_234.0
#  - P4s3_P4s2_23.0
#  - P4s3_P4s2_13.0
#  - P4s3_P4s2_134.0
#  - P4s3_P4s2_124.0
#  - P4s3_P4s4_123.0
#  - P4s3_P4s4_12.0
#  - P4s3_P4s4_34.0
#  - P4s3_P4s4_1234.0
#  - P4s3_P4s4_234.0
#  - P4s3_P4s4_23.0
#  - P4s3_P4s4_13.0
#  - P4s3_P4s4_134.0
#  - P4s3_P4s4_124.0
#  - P4s4_P4s1_123.0
#  - P4s4_P4s1_12.0
#  - P4s4_P4s1_34.0
#  - P4s4_P4s1_1234.0
#  - P4s4_P4s1_234.0
#  - P4s4_P4s1_23.0
#  - P4s4_P4s1_13.0
#  - P4s4_P4s1_134.0
#  - P4s4_P4s1_124.0
#  - P4s4_P4s2_123.0
#  - P4s4_P4s2_12.0
#  - P4s4_P4s2_34.0
#  - P4s4_P4s2_1234.0
#  - P4s4_P4s2_234.0
#  - P4s4_P4s2_23.0
#  - P4s4_P4s2_13.0
#  - P4s4_P4s2_134.0
#  - P4s4_P4s2_124.0
#  - P4s4_P4s3_123.0
#  - P4s4_P4s3_12.0
#  - P4s4_P4s3_34.0
#  - P4s4_P4s3_1234.0
#  - P4s4_P4s3_234.0
#  - P4s4_P4s3_23.0
#  - P4s4_P4s3_13.0
#  - P4s4_P4s3_134.0
#  - P4s4_P4s3_124.0











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
