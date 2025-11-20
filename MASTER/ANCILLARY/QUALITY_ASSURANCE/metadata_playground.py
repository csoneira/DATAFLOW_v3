
#%%

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# --- knobs to tweak ---
STATION = "MINGO04"  # e.g. MINGO01, MINGO02, ...
STEP = 1             # numeric step (1, 2, ...)
TASK = 2          # for STEP_1 use an int (1-5); keep None for steps without tasks
START_DATE = "2025-11-01 00:00:00"    # e.g. "2025-11-06 18:00:00" or leave None
END_DATE = "2025-11-20 00:00:00"      # e.g. "2025-11-06 19:00:00" or leave None


repo_root = Path(__file__).resolve().parents[3]
event_dir = repo_root / "STATIONS" / STATION / "STAGE_1" / "EVENT_DATA" / f"STEP_{STEP}"

if TASK is None:
    metadata_path = event_dir / "METADATA" / f"step_{STEP}_metadata_specific.csv"
else:
    metadata_path = (
        event_dir / f"TASK_{TASK}" / "METADATA" / f"task_{TASK}_metadata_specific.csv"
    )

if not metadata_path.exists():
    raise FileNotFoundError(f"Cannot find metadata at {metadata_path}")

df = pd.read_csv(metadata_path)

filename_col = "filename_base" if "filename_base" in df.columns else None
timestamp_col = None
for candidate in ("execution_time", "execution_timestamp"):
    if candidate in df.columns:
        timestamp_col = candidate
        break

def filename_to_datetime(value: str):
    """Parse strings like mi0XYYDDDHHMMSS into real datetimes."""
    if not isinstance(value, str):
        return pd.NaT
    core = value[3:] if value.startswith("mi0") else value
    if len(core) < 12:
        return pd.NaT
    try:
        year = 2000 + int(core[1:3])
        day_of_year = int(core[3:6])
        hour = int(core[6:8])
        minute = int(core[8:10])
        second = int(core[10:12])
        return datetime(year, 1, 1) + timedelta(
            days=day_of_year - 1, hours=hour, minutes=minute, seconds=second
        )
    except Exception:
        return pd.NaT


if filename_col:
    df["datetime"] = df[filename_col].apply(filename_to_datetime)
    # Sort the dataframe by datetime
    df = df.sort_values(by="datetime")
    

filter_col = "datetime" if "datetime" in df.columns else timestamp_col

if filter_col and (START_DATE or END_DATE):
    if filter_col == "execution_timestamp":
        fmt = "%Y-%m-%d_%H.%M.%S"
        df[filter_col] = pd.to_datetime(df[filter_col], format=fmt, errors="coerce")
    else:
        df[filter_col] = pd.to_datetime(df[filter_col], errors="coerce")
    start = pd.to_datetime(START_DATE) if START_DATE else df[filter_col].min()
    end = pd.to_datetime(END_DATE) if END_DATE else df[filter_col].max()
    df = df.loc[df[filter_col].between(start, end)]


print(f"Loaded: {metadata_path}")
print(f"Rows: {len(df)}")

#%%

# Print all column names
print("Columns:")
for col in df.columns:
    print(f" - {col}")


#%%


if STEP == 1 and TASK == 1:
    # STEP 1 - TASK 1

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
    else:
        print("Required columns for plotting valid lines percentage are missing.")



    #%%


    # I have four planes, each plane has four strips, each strip two sides: F and B
    # I want to plot the amount of hits T<plane>_<F/B>_<strip>_<original/final>. I want
    # as many plots as necesary to have all in different plots except
    # original and final in the same plot.

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    planes = range(1, 5)   # 1..4
    strips = range(1, 5)   # 1..4
    sides = ["F", "B"]     # top/bottom

    fig = plt.figure(figsize=(32, 14))
    outer_gs = gridspec.GridSpec(
        nrows=len(planes),
        ncols=len(strips),
        hspace=0.35,
        wspace=0.25,
    )

    for i_plane, plane in enumerate(planes):
        for j_strip, strip in enumerate(strips):
            # Inner 2×1 grid for this (plane, strip): F on top, B on bottom
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=outer_gs[i_plane, j_strip],
                hspace=0.05,  # small vertical gap between F/B
            )

            for k_side, side in enumerate(sides):  # k_side = 0 → F (top), 1 → B (bottom)
                ax = fig.add_subplot(inner_gs[k_side, 0], sharex=None)

                original_col = f"T{plane}_{side}_{strip}_entries_original"
                final_col    = f"T{plane}_{side}_{strip}_entries_final"

                if original_col in df.columns and final_col in df.columns:
                    ax.plot(df["datetime"], df[original_col],
                            marker="o", linestyle="", label="Original")
                    ax.plot(df["datetime"], df[final_col],
                            marker="x", linestyle="", label="Final")

                    # Label only once per inner pair and avoid clutter
                    if k_side == 0:
                        ax.set_title(f"P{plane} S{strip}", fontsize=9)

                    # Side label at left
                    ax.text(0.01, 0.9, side,
                            transform=ax.transAxes,
                            fontsize=8,
                            va="top", ha="left")

                    # Y label only on the first column
                    if j_strip == 0:
                        ax.set_ylabel("Hits")

                    # X label only on the last plane row and bottom (B) subplot
                    if i_plane == len(planes) - 1 and k_side == 1:
                        ax.set_xlabel("Datetime")

                    ax.grid(True)

                    # Only show legend in one subplot to avoid repetition
                    if i_plane == 0 and j_strip == 0 and k_side == 0:
                        ax.legend(loc="upper right", fontsize=7)
                else:
                    # If there is no data for this side, hide axis
                    ax.set_visible(False)

    fig.suptitle("Hits per plane / strip / side (Original vs Final)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#%%








#     - CRT_avg
# Plot CRT_avg time series

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df["datetime"],df["CRT_avg"], marker='o', linestyle='')
plt.title(f"CRT for {STATION} STEP {STEP} TASK {TASK}")
plt.xlabel("Datetime")
plt.ylabel("CRT (ns)")
plt.grid(True)
plt.show()




#%%


# Loop on plane, strip, original/final
#     - T1_T_sum_1_entries_original


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

        original_col = f"T{plane}_T_sum_{strip}_entries_original"
        final_col    = f"T{plane}_T_sum_{strip}_entries_final"

        if original_col in df.columns and final_col in df.columns:
            ax.plot(df["datetime"], df[original_col],
                    marker="o", linestyle="", label="Original")
            ax.plot(df["datetime"], df[final_col],
                    marker="x", linestyle="", label="Final")

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

fig.suptitle("Hits per plane/strip for T_sum (Original vs Final)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




# Loop on plane, strip, original/final
#     - T1_T_diff_1_entries_original
# It's the same




# Loop on plane, strip, original/final
#     - Q1_Q_sum_1_entries_original
# It's the same


# Loop on plane, strip, original/final
#     - Q1_Q_diff_1_entries_original
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
                    marker="o", linestyle="", label="Original")
            ax.plot(df["datetime"], df[final_col],
                    marker="x", linestyle="", label="Final")

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

fig.suptitle("Hits per plane/strip for T_sum (Original vs Final)", fontsize=14)
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
                    marker="o", linestyle="", label="Original")
            ax.plot(df["datetime"], df[up_col],
                    marker="x", linestyle="", label="Final")
            
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
                    marker="o", linestyle="", label="Parameter")
            
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
                    marker="o", linestyle="", label="Front")
            ax.plot(df["datetime"], df[final_col],
                    marker="x", linestyle="", label="Back")
            
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
#     - P1_s1_T_sum

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

        original_col = f"P{plane}_s{strip}_T_sum"

        if original_col in df.columns:
            ax.plot(df["datetime"], df[original_col],
                    marker="o", linestyle="", label="Parameter")
            
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

fig.suptitle("T_sum calibration parameter", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




#%%

# Loop on planes and strips
#     - P1_s1_T_dif

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

        original_col = f"P{plane}_s{strip}_T_dif"

        if original_col in df.columns:
            ax.plot(df["datetime"], df[original_col],
                    marker="o", linestyle="", label="Parameter")
            
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

fig.suptitle("T_dif calibration parameter", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


