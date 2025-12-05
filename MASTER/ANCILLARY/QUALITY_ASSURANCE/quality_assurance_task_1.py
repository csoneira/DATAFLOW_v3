#%%


# TASK 1 --> channel counts
# TASK 2 --> 



from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_shared import load_metadata, print_columns, plot_tt_pairs, plot_tt_matrix

# --- knobs to tweak ---
STATION = "MINGO04"  # e.g. "MINGO01", "MINGO02", ...
STEP = 1             # numeric step (1, 2, ...)
TASK = 1          # for STEP_1 use an int (1-5); keep None for steps without tasks
START_DATE = "2025-02-15 00:00:00"    # e.g. "2025-11-06 18:00:00" or leave None
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









def rpc_variable_counts_sets() -> tuple[list[str], list[str]]:
    planes = range(1, 5)
    types = ["T_sum", "T_diff", "Q_sum", "Q_diff", "Y"]
    before = [f"P{plane}_{typ}_nonzero_before_filter6" for plane in planes for typ in types]
    after = [f"P{plane}_{typ}_nonzero_after_filter6" for plane in planes for typ in types]
    return before, after


def plot_rpc_counts(df: pd.DataFrame, tcol: Optional[str]) -> None:
    if not tcol:
        return
    before_cols, after_cols = rpc_variable_counts_sets()
    y_min = y_max = None
    for col in before_cols + after_cols:
        if col not in df.columns:
            continue
        ser = df[col].dropna()
        if ser.empty:
            continue
        lo, hi = float(ser.min()), float(ser.max())
        y_min = lo if y_min is None else min(y_min, lo)
        y_max = hi if y_max is None else max(y_max, hi)
    if y_min is None or y_max is None:
        return

    planes = range(1, 5)
    types = ["T_sum", "T_diff", "Q_sum", "Q_diff", "Y"]
    fig, axes = plt.subplots(len(planes), len(types), figsize=(len(types) * 3, len(planes) * 2.5), sharex=True)
    if axes.ndim == 1:
        axes = axes.reshape(len(planes), -1)

    for i, plane in enumerate(planes):
        for j, typ in enumerate(types):
            ax = axes[i, j]
            before = f"P{plane}_{typ}_nonzero_before_filter6"
            after = f"P{plane}_{typ}_nonzero_after_filter6"
            plotted_any = False
            for label, col in (("Before", before), ("After", after)):
                if col not in df.columns:
                    continue
                ser = df[[tcol, col]].dropna()
                if ser.empty:
                    continue
                ax.plot(ser[tcol], ser[col], marker="o" if label == "Before" else "x", markersize=2, linestyle="", label=label)
                plotted_any = True
                plotted_cols.add(col)
            if plotted_any:
                ax.set_title(f"P{plane} • {typ}", fontsize=8)
                ax.set_ylim(y_min, y_max)
                if j == 0:
                    ax.set_ylabel("Count")
                if i == 3:
                    ax.set_xlabel("Datetime")
                ax.grid(True, linestyle="--", alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
            else:
                ax.set_visible(False)

    fig.suptitle(f"RPC counts before/after filter6 per plane/type • {STATION} STEP {STEP} TASK {TASK}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def active_strips_figures(df: pd.DataFrame) -> None:
    pattern_sets = [
        ["1000", "0100", "0010", "0001"],
        ["1100", "0110", "0011"],
        ["1010", "1001", "0101"],
        ["1110", "1011", "0111", "1101", "1111"],
    ]
    for idx, patterns in enumerate(pattern_sets, start=1):
        available = [f"active_strips_P{plane}_{pattern}_count" for plane in range(1, 5) for pattern in patterns if f"active_strips_P{plane}_{pattern}_count" in df.columns]
        if not available:
            continue
        values = df[available].values.flatten()
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        vmin, vmax = float(values.min()), float(values.max())
        ncols = len(patterns)
        fig_w = max(6, ncols * 3)
        fig_h = max(3, 4 * 2.5)
        fig, axes = plt.subplots(4, ncols, figsize=(fig_w, fig_h), sharex=True)
        if axes.ndim == 1:
            axes = axes.reshape(4, -1)
        for i, plane in enumerate(range(1, 5)):
            for j, pattern in enumerate(patterns):
                ax = axes[i, j]
                colname = f"active_strips_P{plane}_{pattern}_count"
                if colname not in df.columns:
                    ax.set_visible(False)
                    continue
                ser = df[["datetime", colname]].dropna()
                if ser.empty:
                    ax.set_visible(False)
                    continue
                ax.scatter(ser["datetime"], ser[colname], s=1, color="C0", alpha=0.8)
                ax.set_title(f"P{plane} • {pattern}", fontsize=8)
                ax.set_ylim(vmin, vmax)
                if j == 0:
                    ax.set_ylabel("Count")
                if i == 3:
                    ax.set_xlabel("Datetime")
                ax.grid(True, linestyle="--", alpha=0.3)
        fig.suptitle(f"Active strips pattern set {idx} • {STATION} STEP {STEP} TASK {TASK}", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        plotted_cols.update(available)

print(f"Loaded: {ctx.metadata_path}")
print(f"Rows: {len(df)}")
print("Columns:")
print_columns(df)

#%%

COLUMN_FAMILY_DOC = '''
Channel count families for STEP 1 TASK 1
---------------------------------------
1. Raw and clean channel hits contributing to trigger selection (raw_tt_*/clean_tt_*).
2. Lists-to-fit conversions (list_to_fit_tt_*).
3. Various cal_tt and list_tt event totals tracking triggers before/after filtering.
4. Topology patterns (active_strips_P*_????_count) describing strip multiplicities per plane.
5. RPC nonzero counts before/after filter6 for timing and charge sums/differences.
'''
print(COLUMN_FAMILY_DOC)


# STEP 1 - TASK 1




# %%



# --- Additional intelligent plotting for STEP 1 TASK 1 ---

# determine time column to use for plotting
tcol = ctx.time_col

# Example reuse: plot clean -> cal pairs. Uncomment if wanted.
try:
    plot_tt_pairs(ctx, 'raw_tt_', 'clean_tt_', f"raw_tt → clean_tt • {STATION} STEP {STEP} TASK {TASK}", ncols=5)
except Exception:
    print("Could not plot raw_tt_ -> clean_tt_ pairs.")
    pass


#%%



# Plot raw->clean matrix (re-usable: change prefixes to plot other matrices)
try:
    plot_tt_matrix(ctx, 'raw', 'clean', f"raw_to_clean matrix • {STATION} STEP {STEP} TASK {TASK}")
except Exception:
    print("Could not plot raw -> clean matrix.")
    pass
# %%


# Plot the time series of valid_lines_in_binary_file_percentage percentage over datetime
if "valid_lines_in_binary_file_percentage" in df.columns:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(df["datetime"],df["valid_lines_in_binary_file_percentage"], marker='o', linestyle='-')
    plt.title(f"Valid Lines Percentage Over Time for {STATION} STEP {STEP} TASK {TASK}")
    plt.xlabel("Datetime")
    plt.ylabel("Valid Lines Percentage (%)")
    plt.grid(True)
    plt.show()
    plotted_cols.add("valid_lines_in_binary_file_percentage")
else:
    print("Required columns for plotting valid lines percentage are missing.")


#%%


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

planes = range(1, 5)   # 1..4
strips = range(1, 5)   # 1..4
sides  = ["F", "B"]    # top/bottom

# Adjustable marker size
marker_size = 1
sigma_limit = 3

# ----------------------------------------------------------
# First pass: global y-limits over all planes/strips/sides
# (for the raw plots: Original & Final)
# ----------------------------------------------------------
y_min_global = None
y_max_global = None

for plane in planes:
    for strip in strips:
        for side in sides:
            original_col = f"T{plane}_{side}_{strip}_entries_original"
            final_col    = f"T{plane}_{side}_{strip}_entries_final"

            if original_col in df.columns and final_col in df.columns:
                y_orig = df[original_col]
                y_fin  = df[final_col]

                local_min = min(np.nanmin(y_orig), np.nanmin(y_fin))
                local_max = max(np.nanmax(y_orig), np.nanmax(y_fin))

                if y_min_global is None or local_min < y_min_global:
                    y_min_global = local_min
                if y_max_global is None or local_max > y_max_global:
                    y_max_global = local_max

# ----------------------------------------------------------
# Second pass: build figure with raw + normalized/3σ
# ----------------------------------------------------------
fig = plt.figure(figsize=(32, 22))
outer_gs = gridspec.GridSpec(
    nrows=len(planes),
    ncols=len(strips),
    hspace=0.35,
    wspace=0.25,
)

for i_plane, plane in enumerate(planes):
    for j_strip, strip in enumerate(strips):
        # 4×1 grid per (plane, strip):
        # 0: F raw, 1: F norm, 2: B raw, 3: B norm
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            4,
            1,
            subplot_spec=outer_gs[i_plane, j_strip],
            height_ratios=[3, 1, 3, 1],
            hspace=0.05,
        )

        for k_side, side in enumerate(sides):  # 0 -> F, 1 -> B
            row_base = 2 * k_side
            ax_main = fig.add_subplot(inner_gs[row_base, 0])
            ax_norm = fig.add_subplot(inner_gs[row_base + 1, 0],
                                        sharex=ax_main)

            original_col = f"T{plane}_{side}_{strip}_entries_original"
            final_col    = f"T{plane}_{side}_{strip}_entries_final"

            if original_col in df.columns and final_col in df.columns:
                plotted_cols.update([original_col, final_col])
                t = df["datetime"]
                y_orig = df[original_col]
                y_fin  = df[final_col]

                # ---------------------------
                # Raw plot (upper in pair)
                # ---------------------------
                ax_main.plot(
                    t, y_orig,
                    marker=".", markersize=marker_size,
                    linestyle="", label="Original",
                )
                ax_main.plot(
                    t, y_fin,
                    marker="x", markersize=marker_size,
                    linestyle="", label="Final",
                )

                if y_min_global is not None and y_max_global is not None:
                    ax_main.set_ylim(y_min_global, y_max_global)

                # Title once per (plane, strip) on the first side (F)
                if k_side == 0:
                    ax_main.set_title(f"P{plane} S{strip}", fontsize=9)

                # Side label on the left (F/B)
                ax_main.text(
                    0.01, 0.9, side,
                    transform=ax_main.transAxes,
                    fontsize=8,
                    va="top", ha="left",
                )

                # Y label only on the first column
                if j_strip == 0:
                    ax_main.set_ylabel("Hits")

                ax_main.grid(True)

                # Legend only once, first raw subplot (P1, S1, F)
                if i_plane == 0 and j_strip == 0 and k_side == 0:
                    ax_main.legend(loc="upper right", fontsize=7)

                # Hide x tick labels on upper raw plots
                plt.setp(ax_main.get_xticklabels(), visible=False)

                # ---------------------------
                # Normalized plot (lower in pair)
                #   (x - μ) / (3σ)
                # ---------------------------
                def normalize_3sigma(v):
                    m = np.nanmean(v)
                    s = np.nanstd(v)
                    if s > 0:
                        return (v - m) / s
                    else:
                        return np.zeros_like(v)

                y_orig_n = normalize_3sigma(y_orig)
                y_fin_n  = normalize_3sigma(y_fin)

                ax_norm.plot(
                    t, y_orig_n,
                    marker="o", markersize=marker_size,
                    linestyle="", label="Original",
                )
                ax_norm.plot(
                    t, y_fin_n,
                    marker="x", markersize=marker_size,
                    linestyle="", label="Final",
                )

                ax_norm.set_ylim(-sigma_limit, sigma_limit)
                ax_norm.axhline(0.0, linestyle="--", linewidth=0.8)

                if j_strip == 0:
                    ax_norm.set_ylabel("Z/σ")

                # Only bottom-most panel (B normalized, last plane row)
                # gets the x-axis label
                if i_plane == len(planes) - 1 and k_side == 1:
                    ax_norm.set_xlabel("Datetime")
                else:
                    plt.setp(ax_norm.get_xticklabels(), visible=False)

                ax_norm.grid(True)

            else:
                ax_main.set_visible(False)
                ax_norm.set_visible(False)

fig.suptitle(
    "Hits per plane / strip / side (Original vs Final, raw and normalized / 3σ)",
    fontsize=14,
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



