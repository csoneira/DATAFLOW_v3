#%%


# TASK 1 --> channel counts
# TASK 2 --> 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_shared import load_metadata, print_columns, plot_tt_pairs, plot_tt_matrix

# --- knobs to tweak ---
STATION = "MINGO04"  # e.g. "MINGO01", "MINGO02", ...
STEP = 1             # numeric step (1, 2, ...)
TASK = 4          # for STEP_1 use an int (1-5); keep None for steps without tasks
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


#%%


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


ctx_2 = load_metadata(STATION, STEP, TASK, START_DATE, END_DATE)
df_2 = ctx_2.df
plotted_cols: set[str] = set()

print(f"Loaded: {ctx_2.metadata_path}")
print(f"Rows: {len(df_2)}")


#


# --- Additional intelligent plotting for STEP 1 TASK 1 ---

# determine time column to use for plotting
tcol = ctx.time_col

# %%


# Create a time series for clean_tt_1234_count, cal_tt_1234_count, fit_tt_1234_count

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df_2[tcol], df_2["clean_tt_1234_count"], label="clean_tt_1234_count", alpha=0.7)
ax.scatter(df_2[tcol], df_2["cal_tt_1234_count"], label="cal_tt_1234_count", alpha=0.7)
ax.scatter(df[tcol], df["list_tt_1234_count"], label="list_tt_1234_count", alpha=0.7)
ax.set_xlabel("Time")
ax.set_ylabel("Counts")
ax.set_title(f"{STATION} STEP {STEP} TASK {TASK} - TT 1234 Counts Over Time")
ax.legend()
plt.show()


# %%
