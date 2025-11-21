
#%%

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# --- knobs to tweak ---
STATION = "MINGO01"  # e.g. MINGO01, MINGO02, ...
STEP = 1             # numeric step (1, 2, ...)
TASK = 4          # for STEP_1 use an int (1-5); keep None for steps without tasks
START_DATE = "2025-11-01 00:00:00"    # e.g. "2025-11-06 18:00:00" or leave None
END_DATE = "2025-11-20 00:00:00"      # e.g. "2025-11-06 19:00:00" or leave None
# Window used when counting events (ns) and per-combination measured counts.
# Set WINDOW_NS to the calibration window you used (e.g., coincidence_window_cal_ns),
# and fill MEASURED_COUNTS with {combo: observed_counts}.
WINDOW_NS = None
MEASURED_COUNTS = {
    # 12: 0,
    # 123: 0,
}


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
    
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    planes = range(1, 5)   # 1..4
    strips = range(1, 5)   # 1..4
    sides  = ["F", "B"]    # top/bottom

    # Adjustable marker size
    marker_size = 2
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
                    t = df["datetime"]
                    y_orig = df[original_col]
                    y_fin  = df[final_col]

                    # ---------------------------
                    # Raw plot (upper in pair)
                    # ---------------------------
                    ax_main.plot(
                        t, y_orig,
                        marker="o", markersize=marker_size,
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

    
    
    

#%%



# STEP 1 - TASK 2
if STEP == 1 and TASK == 2:

    #     - CRT_avg
    # Plot CRT_avg time series with error bar = CRT_std

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(df["datetime"], df["CRT_avg"], marker='o', linestyle='-')
    plt.fill_between(df["datetime"], df["CRT_avg"] + df["CRT_std"]/2, df["CRT_avg"] - df["CRT_std"]/2, alpha=0.2, label='CRT Std Dev')
    plt.title(f"CRT for {STATION} STEP {STEP} TASK {TASK}")
    plt.xlabel("Datetime")
    plt.ylabel("CRT (ns)")
    plt.grid(True)
    plt.show()



    #%%



    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    planes = range(1, 5)
    strips = range(1, 5)

    # Adjustable marker size
    marker_size = 2
    sigma_limit = 4  # y-limits for normalized plots

    # ----------------------------------------------------------
    # First pass: determine global y-limits (original & final)
    # ----------------------------------------------------------

    y_min_global = None
    y_max_global = None

    for plane in planes:
        for strip in strips:
            orig = f"T{plane}_T_sum_{strip}_entries_original"
            fin  = f"T{plane}_T_sum_{strip}_entries_final"

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

            orig = f"T{plane}_T_sum_{strip}_entries_original"
            fin  = f"T{plane}_T_sum_{strip}_entries_final"

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



    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # Adjustable marker size
    marker_size = 2   # change this value to control point size
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
                    ax_main.plot(t, y, marker="o", markersize=marker_size,
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


if STEP == 1 and TASK == 4:

    # I want to plot 

    # - z_P1
    # - z_P2
    # - z_P3
    # - z_P4

    fig = plt.figure(figsize=(10, 6))
    for plane in range(1, 5):
        col = f"z_P{plane}"
        if col in df.columns:
            plt.plot(df["datetime"], df[col], marker='.', linestyle='', label=f'Plane {plane}')
    plt.title(f"Z Position per Plane for {STATION} STEP {STEP} TASK {TASK}")
    plt.xlabel("Datetime")
    plt.ylabel("Z Position (units)")
    plt.legend()
    plt.grid(True)
    plt.show()


    #%%

        
    # - sigmoid_width_234
    # - background_slope_234
    # - sigmoid_width_123
    # - background_slope_123
    # - sigmoid_width_34
    # - background_slope_34
    # - sigmoid_width_1234
    # - background_slope_1234
    # - sigmoid_width_23
    # - background_slope_23
    # - sigmoid_width_12
    # - background_slope_12
    # - sigmoid_width_124
    # - background_slope_124
    # - sigmoid_width_134
    # - background_slope_134
    # - sigmoid_width_24
    # - background_slope_24
    # - sigmoid_width_13
    # - background_slope_13
    # - sigmoid_width_14
    # - background_slope_14


    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # Adjustable poit size
    marker_size = 1
    sigma_limit = 4  # y-limits for normalized plots

    combinations = [
        "234", "123", "34", "1234", "23", "12", "124", "134", "24", "13", "14"
    ]

    # ----------------------------------------------------------
    # First pass: get global y-limits for widths and slopes separately
    # ----------------------------------------------------------
    width_min = width_max = None
    slope_min = slope_max = None

    for combo in combinations:
        wcol = f"sigmoid_width_{combo}"
        scol = f"background_slope_{combo}"

        if wcol in df.columns:
            local_min = np.nanmin(df[wcol])
            local_max = np.nanmax(df[wcol])
            width_min = local_min if width_min is None else min(width_min, local_min)
            width_max = local_max if width_max is None else max(width_max, local_max)

        if scol in df.columns:
            local_min = np.nanmin(df[scol])
            local_max = np.nanmax(df[scol])
            slope_min = local_min if slope_min is None else min(slope_min, local_min)
            slope_max = local_max if slope_max is None else max(slope_max, local_max)

    # ----------------------------------------------------------
    # Second pass: build figure with raw + normalized
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(14, 4 * len(combinations)))
    outer_gs = gridspec.GridSpec(
        nrows=len(combinations),
        ncols=2,
        hspace=0.5,
        wspace=0.3,
    )

    def zscore(v):
        m = np.nanmean(v)
        s = np.nanstd(v)
        return (v - m) / s if s > 0 else np.zeros_like(v)

    for i, combo in enumerate(combinations):

        # ======================================================
        # Column 1: Sigmoid width (raw + normalized)
        # ======================================================
        inner_gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_gs[i, 0],
            height_ratios=[3, 1],
            hspace=0.05
        )

        wcol = f"sigmoid_width_{combo}"
        ax_w_raw  = fig.add_subplot(inner_gs_left[0, 0])
        ax_w_norm = fig.add_subplot(inner_gs_left[1, 0], sharex=ax_w_raw)

        if wcol in df.columns:
            t = df["datetime"]
            y = df[wcol]

            # Raw
            ax_w_raw.plot(t, y, marker="o", markersize=marker_size, linestyle="-")
            ax_w_raw.set_title(f"Sigmoid Width {combo}", fontsize=10)
            ax_w_raw.set_ylim(width_min, width_max)
            ax_w_raw.set_ylabel("Width")
            ax_w_raw.grid(True)
            plt.setp(ax_w_raw.get_xticklabels(), visible=False)

            # Normalized
            y_norm = zscore(y)
            ax_w_norm.plot(t, y_norm, marker="o", markersize=marker_size, linestyle="-")
            ax_w_norm.set_ylim(-sigma_limit, sigma_limit)
            ax_w_norm.axhline(0, linestyle="--", linewidth=0.8)
            ax_w_norm.set_ylabel("Z")
            ax_w_norm.set_xlabel("Datetime" if i == len(combinations) - 1 else "")
            if i < len(combinations) - 1:
                plt.setp(ax_w_norm.get_xticklabels(), visible=False)
            ax_w_norm.grid(True)
        else:
            ax_w_raw.set_visible(False)
            ax_w_norm.set_visible(False)

        # ======================================================
        # Column 2: Background slope (raw + normalized)
        # ======================================================
        inner_gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_gs[i, 1],
            height_ratios=[3, 2],
            hspace=0.05
        )

        scol = f"background_slope_{combo}"
        ax_s_raw  = fig.add_subplot(inner_gs_right[0, 0])
        ax_s_norm = fig.add_subplot(inner_gs_right[1, 0], sharex=ax_s_raw)

        if scol in df.columns:
            y = df[scol]
            t = df["datetime"]

            # Raw
            ax_s_raw.plot(t, y, marker="o", markersize=marker_size, linestyle="-")
            ax_s_raw.set_title(f"Background Slope {combo}", fontsize=10)
            ax_s_raw.set_ylim(slope_min, slope_max)
            ax_s_raw.set_ylabel("Slope")
            ax_s_raw.grid(True)
            plt.setp(ax_s_raw.get_xticklabels(), visible=False)

            # Normalized
            y_norm = zscore(y)
            ax_s_norm.plot(t, y_norm, marker="o", markersize=marker_size, linestyle="-")
            ax_s_norm.set_ylim(-sigma_limit, sigma_limit)
            ax_s_norm.axhline(0, linestyle="--", linewidth=0.8)
            ax_s_norm.set_ylabel("Z")
            ax_s_norm.set_xlabel("Datetime" if i == len(combinations) - 1 else "")
            if i < len(combinations) - 1:
                plt.setp(ax_s_norm.get_xticklabels(), visible=False)
            ax_s_norm.grid(True)
        else:
            ax_s_raw.set_visible(False)
            ax_s_norm.set_visible(False)

    plt.suptitle(f"Sigmoid Widths and Background Slopes for {STATION} STEP {STEP} TASK {TASK}",
                fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
    #%%
    
    WINDOW_NS = 3  # Set this to the coincidence window (ns) used when counting
    
    #-----------------------------------------------------------------------
    # Compute noise/signal counts from fit parameters + measured counts
    #-----------------------------------------------------------------------
    import numpy as np
    from math import erf, sqrt

    combinations_int = [234, 123, 34, 1234, 23, 12, 124, 134, 24, 13, 14]

    if filter_col:
        df_sorted = df.sort_values(filter_col)
    else:
        df_sorted = df

    if len(df_sorted) == 0:
        raise SystemExit("No metadata rows available.")

    latest_row = df_sorted.iloc[-1]

    if WINDOW_NS is None:
        print("[Info] WINDOW_NS is not set. Fill it with the coincidence window (ns) used when counting.")

    results = []
    for combo in combinations_int:
        width = latest_row.get(f"sigmoid_width_{combo}", np.nan)
        slope = latest_row.get(f"background_slope_{combo}", np.nan)
        amp = latest_row.get(f"sigmoid_amplitude_{combo}", np.nan)
        center = latest_row.get(f"sigmoid_center_{combo}", np.nan)
        measured_counts = MEASURED_COUNTS.get(combo)

        if not np.all(np.isfinite([width, slope, amp, center])):
            results.append({
                "combo": combo,
                "noise_counts": np.nan,
                "signal_counts": np.nan,
                "background_fraction": np.nan,
                "note": "Missing fit parameters",
            })
            continue

        if WINDOW_NS is None or not np.isfinite(WINDOW_NS):
            results.append({
                "combo": combo,
                "noise_counts": np.nan,
                "signal_counts": np.nan,
                "background_fraction": np.nan,
                "note": "Set WINDOW_NS first",
            })
            continue

        # Signal/background split at the chosen window width (normalized units)
        s = 0.5 * amp * (1 + erf((WINDOW_NS - center) / (sqrt(2) * width)))
        b = slope * WINDOW_NS
        total = s + b

        if total <= 0 or not np.isfinite(total):
            results.append({
                "combo": combo,
                "noise_counts": np.nan,
                "signal_counts": np.nan,
                "background_fraction": np.nan,
                "note": "Non-positive total from fit",
            })
            continue

        p_bkg = b / total
        p_sig = 1 - p_bkg

        if measured_counts is None:
            results.append({
                "combo": combo,
                "noise_counts": np.nan,
                "signal_counts": np.nan,
                "background_fraction": p_bkg,
                "note": "Add MEASURED_COUNTS[combo]",
            })
            continue

        noise_counts = measured_counts * p_bkg
        signal_counts = measured_counts * p_sig
        results.append({
            "combo": combo,
            "noise_counts": noise_counts,
            "signal_counts": signal_counts,
            "background_fraction": p_bkg,
            "note": "",
        })

    results_df = pd.DataFrame(results)
    print("\nEstimated noise/signal counts per combination (latest metadata row):")
    print(results_df.to_string(index=False))
