
#%%


# TASK 1 --> channel counts
# TASK 2 --> 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_shared import (
    COMBINATIONS,
    EVENT_COMBOS,
    GAUSS_VARS,
    RES_VARS,
    EXPECTED_COLUMNS,
    expected_columns_by_family,
    load_metadata,
    print_columns,
)

# --- knobs to tweak ---
STATION = "MINGO01"  # e.g. MINGO01, MINGO02, ...
STEP = 1             # numeric step (1, 2, ...)
TASK = 4          # for STEP_1 use an int (1-5); keep None for steps without tasks
START_DATE = "2024-03-01 00:00:00"    # e.g. "2025-11-06 18:00:00" or leave None
END_DATE = "2025-11-20 00:00:00"      # e.g. "2025-11-06 19:00:00" or leave None
WINDOW_NS = None
MEASURED_COUNTS = {
    # 12: 0,
    # 123: 0,
}


ctx = load_metadata(STATION, STEP, TASK, START_DATE, END_DATE)
df = ctx.df

print(f"Loaded: {ctx.metadata_path}")
print(f"Rows: {len(df)}")
print_columns(df)


def plot_metadata_columns(context):
    df = context.df
    tcol = context.time_col
    if not tcol:
        return

    time_series = pd.to_datetime(df[tcol], errors="coerce")

    if "analysis_mode" in df.columns:
        cat = pd.Categorical(df["analysis_mode"])
        mask = time_series.notna() & (cat.codes >= 0)
        fig, ax = plt.subplots(figsize=(12, 2.5))
        ax.scatter(time_series[mask], cat.codes[mask], s=16, alpha=0.6)
        ax.set_yticks(range(len(cat.categories)))
        ax.set_yticklabels(cat.categories)
        ax.set_ylabel("Mode")
        ax.set_title("Analysis mode timeline")
        ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.show()
        context.record("analysis_mode")

    unc_columns = ["unc_y", "unc_tsum", "unc_tdif"]
    recorded = []
    fig, ax = plt.subplots(figsize=(12, 3))
    for col in unc_columns:
        if col not in df.columns:
            continue
        series = df[[tcol, col]].dropna()
        if series.empty:
            continue
        ax.plot(series[tcol], series[col], marker="o", markersize=3, label=col, linestyle="-")
        recorded.append(col)
    if recorded:
        ax.set_title("Fit uncertainties")
        ax.set_ylabel("Value")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.show()
        context.record(recorded)
    else:
        plt.close(fig)

    z_columns = []
    fig, ax = plt.subplots(figsize=(12, 4))
    for plane in range(1, 5):
        col = f"z_P{plane}"
        if col not in df.columns:
            continue
        series = df[[tcol, col]].dropna()
        if series.empty:
            continue
        ax.plot(series[tcol], series[col], marker=".", linestyle="-", label=f"Plane {plane}")
        z_columns.append(col)
    if z_columns:
        ax.set_title("Z position per plane")
        ax.set_ylabel("Z position")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.show()
        context.record(z_columns)
    else:
        plt.close(fig)

    if "filename_base" in df.columns:
        codes, uniques = pd.factorize(df["filename_base"])
        mask = time_series.notna()
        fig, ax = plt.subplots(figsize=(12, 2.5))
        ax.scatter(time_series[mask], codes[mask], s=8, alpha=0.6)
        ax.set_ylabel("Unique file index")
        ax.set_title("Filename base progression")
        ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.show()
        context.record("filename_base")

    if "execution_timestamp" in df.columns:
        exec_ts = pd.to_datetime(df["execution_timestamp"], format="%Y-%m-%d_%H.%M.%S", errors="coerce")
        mask = time_series.notna() & exec_ts.notna()
        if mask.any():
            fig, ax = plt.subplots(figsize=(12, 3))
            if tcol == "execution_timestamp":
                y = exec_ts[mask].astype("int64") // 1_000_000_000
                ax.scatter(df.index[mask], y, s=8)
                ax.set_ylabel("Timestamp (s)")
                ax.set_title("Execution timestamp sequence")
            else:
                delta = (exec_ts - time_series).dt.total_seconds()
                ax.scatter(time_series[mask], delta[mask], s=8, alpha=0.6)
                ax.set_ylabel("Δ seconds")
                ax.set_title("Execution timestamp offset from primary time column")
            ax.grid(True, linestyle="--", alpha=0.35)
            plt.tight_layout()
            plt.show()
            context.record("execution_timestamp")


def plot_sigmoid_parameters(context):
    df = context.df
    tcol = context.time_col
    if not tcol:
        return

    param_groups = [
        ("sigmoid_width", "Sigmoid widths"),
        ("sigmoid_amplitude", "Sigmoid amplitudes"),
        ("sigmoid_center", "Sigmoid centers"),
    ]

    for prefix, heading in param_groups:
        recorded = []
        fig, ax = plt.subplots(figsize=(12, 4))
        for combo in COMBINATIONS:
            col = f"{prefix}_{combo}"
            if col not in df.columns:
                continue
            series = df[[tcol, col]].dropna()
            if series.empty:
                continue
            ax.plot(series[tcol], series[col], marker="o", markersize=3, linestyle="-", label=combo)
            recorded.append(col)
        if not recorded:
            plt.close(fig)
            continue
        ax.set_title(heading)
        ax.set_ylabel(prefix.replace("sigmoid_", "").capitalize())
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.show()
        context.record(recorded)

    for suffix, title, ylabel in (
        ("background_slope", "Background slopes across combos", "Slope"),
        ("fit_normalization", "Fit normalizations across combos", "Normalization"),
    ):
        recorded = []
        fig, ax = plt.subplots(figsize=(12, 4))
        for combo in COMBINATIONS:
            col = f"{suffix}_{combo}"
            if col not in df.columns:
                continue
            series = df[[tcol, col]].dropna()
            if series.empty:
                continue
            ax.plot(series[tcol], series[col], marker="o", markersize=3, linestyle="-", label=combo)
            recorded.append(col)
        if not recorded:
            plt.close(fig)
            continue
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.show()
        context.record(recorded)


def plot_resolution_sigmas(context):
    df = context.df
    tcol = context.time_col
    if not tcol:
        return

    for prefix, desc in (("res", "Intrinsic"), ("ext_res", "External")):
        for var in RES_VARS:
            recorded = []
            fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
            for plane, ax in zip(range(1, 5), axes):
                plane_recorded = []
                for combo in COMBINATIONS:
                    col = f"{prefix}_{var}_{plane}_{combo}_sigma"
                    if col not in df.columns:
                        continue
                    series = df[[tcol, col]].dropna()
                    if series.empty:
                        continue
                    ax.plot(series[tcol], series[col], marker="o", markersize=3, linestyle="-", label=combo)
                    plane_recorded.append(col)
                if plane_recorded:
                    ax.set_ylabel(f"Plane {plane}")
                    ax.grid(True, linestyle="--", alpha=0.35)
                    recorded.extend(plane_recorded)
                else:
                    ax.set_visible(False)
            if not recorded:
                plt.close(fig)
                continue
            axes[-1].set_xlabel("Datetime")
            axes[0].set_title(f"{desc} {var} sigmas")
            plt.tight_layout()
            plt.show()
            context.record(recorded)


def plot_event_statistics(context):
    df = context.df
    tcol = context.time_col
    if not tcol:
        return

    list_cols = []
    fig, ax = plt.subplots(figsize=(12, 4))
    for combo in EVENT_COMBOS:
        col = f"list_tt_{combo}_count"
        if col not in df.columns:
            continue
        series = df[[tcol, col]].dropna()
        if series.empty:
            continue
        ax.plot(series[tcol], series[col], marker="o", markersize=3, linestyle="-", label=combo)
        list_cols.append(col)
    if list_cols:
        ax.set_title("List TT counts by combination")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=6, ncol=3)
        plt.tight_layout()
        plt.show()
        context.record(list_cols)
    else:
        plt.close(fig)

    to_fit_cols = []
    fig, ax = plt.subplots(figsize=(12, 4))
    for combo in COMBINATIONS:
        col = f"list_to_fit_tt_{combo}_0_count"
        if col not in df.columns:
            continue
        series = df[[tcol, col]].dropna()
        if series.empty:
            continue
        ax.plot(series[tcol], series[col], marker="x", markersize=3, linestyle="-", label=combo)
        to_fit_cols.append(col)
    if to_fit_cols:
        ax.set_title("List-to-fit TT counts")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=6, ncol=3)
        plt.tight_layout()
        plt.show()
        context.record(to_fit_cols)
    else:
        plt.close(fig)

    fit_col = "fit_tt_0_count"
    if fit_col in df.columns:
        series = df[[tcol, fit_col]].dropna()
        if not series.empty:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(series[tcol], series[fit_col], marker="o", markersize=3, linestyle="-")
            ax.set_title("Fit TT zero count")
            ax.set_ylabel("Count")
            ax.grid(True, linestyle="--", alpha=0.35)
            plt.tight_layout()
            plt.show()
            context.record(fit_col)


def plot_gaussian_mixtures(context):
    df = context.df
    tcol = context.time_col
    if not tcol:
        return

    colors = plt.cm.get_cmap("tab10")
    for var in GAUSS_VARS:
        recorded = []
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
        axes[0].set_title(f"{var} Gaussian mixture mus")
        axes[1].set_title(f"{var} Gaussian mixture sigmas")
        axes[2].set_title(f"{var} Gaussian mixture amplitudes")
        for idx, combo in enumerate(COMBINATIONS):
            color = colors(idx % 10)
            for comp, axis in (("gauss1", axes[0]), ("gauss2", axes[0])):
                col_mu = f"{var}_err_{combo}_{comp}_mu"
                if col_mu in df.columns:
                    series = df[[tcol, col_mu]].dropna()
                    if not series.empty:
                        axis.plot(
                            series[tcol],
                            series[col_mu],
                            linestyle="-" if comp == "gauss1" else "--",
                            color=color,
                            label=f"{combo} {comp}",
                            markersize=3,
                        )
                        recorded.append(col_mu)
            for comp in ("gauss1", "gauss2"):
                col_sigma = f"{var}_err_{combo}_{comp}_sigma"
                if col_sigma in df.columns:
                    series = df[[tcol, col_sigma]].dropna()
                    if not series.empty:
                        axes[1].plot(
                            series[tcol],
                            series[col_sigma],
                            linestyle="-" if comp == "gauss1" else "--",
                            color=color,
                            label=f"{combo} {comp}",
                            markersize=3,
                        )
                        recorded.append(col_sigma)
            col_amp = f"{var}_err_{combo}_gauss1_amp"
            if col_amp in df.columns:
                series = df[[tcol, col_amp]].dropna()
                if not series.empty:
                    axes[2].plot(series[tcol], series[col_amp], linestyle="-", color=color, label=combo, markersize=3)
                    recorded.append(col_amp)
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.35)
        axes[-1].set_xlabel("Datetime")
        if recorded:
            plt.tight_layout()
            plt.show()
            context.record(recorded)
        else:
            plt.close(fig)


def plot_error_quantiles(context):
    df = context.df
    tcol = context.time_col
    if not tcol:
        return

    comb_list = sorted(COMBINATIONS, key=lambda s: (-len(s), s))
    vars_order = GAUSS_VARS
    nrows = len(comb_list)
    ncols = len(vars_order)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 1.6), sharex=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[a] for a in axes])

    recorded = []
    for i_r, comb in enumerate(comb_list):
        for j_c, var in enumerate(vars_order):
            ax = axes[i_r, j_c]
            low_col = f"{var}_err_{comb}_q25"
            high_col = f"{var}_err_{comb}_q75"
            if low_col in df.columns and high_col in df.columns:
                series = df[[tcol, low_col, high_col]].dropna()
                if not series.empty:
                    x = series[tcol]
                    ylow = series[low_col]
                    yhigh = series[high_col]
                    ax.fill_between(x, ylow, yhigh, color="C2", alpha=0.25)
                    ax.scatter(x, ylow, s=10, color="C2", marker="o", alpha=0.7)
                    ax.scatter(x, yhigh, s=10, color="C3", marker="x", alpha=0.7)
                    ax.set_title(f"{comb} • {var}", fontsize=8)
                    recorded.extend([low_col, high_col])
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)
            ax.grid(True, linestyle="--", alpha=0.25)
            if j_c == 0:
                ax.set_ylabel(comb)
            if i_r == nrows - 1:
                ax.set_xlabel("Datetime")

    fig.suptitle(f"Error quantiles (q25-q75) • {STATION} STEP {STEP} TASK {TASK}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    if recorded:
        context.record(recorded)


def verify_plotted_columns(context):
    families = expected_columns_by_family()
    expected = EXPECTED_COLUMNS
    missing = sorted(expected - context.plotted_cols)
    extra = sorted(context.plotted_cols - expected)
    print("Documented column families:")
    for k, v in families.items():
        print(f" - {k}: {len(v)} cols")
    if missing:
        print(f"Missing plots for {len(missing)} columns described in the schema.")
        for col in missing:
            print(f" - {col}")
    else:
        print("All described columns have plotting coverage.")
    if extra:
        print("Additional plotted columns outside the documented schema:")
        for col in extra:
            print(f" - {col}")




COLUMN_FAMILY_DOC = f"""
Column families documented for STEP {STEP} TASK {TASK}
-----------------------------------------------
1. Run metadata & geometry
    - analysis_mode, unc_y, unc_tsum, unc_tdif
    - z_P1-z_P4, filename_base, execution_timestamp
2. Sigmoid fits (efficiency curves)
    - Patterns: sigmoid_(width|amplitude|center)_{{COMB}}, background_slope_{{COMB}}, fit_normalization_{{COMB}}
    - Combinations: {', '.join(COMBINATIONS)}
3. Intrinsic resolution sigmas
    - Patterns: res_(ystr|tsum|tdif)_{{PLANE}}_{{COMB}}_sigma with planes 1-4
    - Combinations: {', '.join(COMBINATIONS)}
4. External resolution sigmas
    - Same pattern as (3) prefixed with ext_res_
5. Event statistics
    - list_tt_{{combo}}_count for combos {', '.join(EVENT_COMBOS)}
    - fit_tt_0_count
    - list_to_fit_tt_{{combo}}_0_count for {', '.join(COMBINATIONS)}
6. Gaussian mixtures
    - Variables: {', '.join(GAUSS_VARS)}
    - Patterns: {{var}}_err_{{combo}}_gauss1_(mu|sigma|amp) + ...gauss2_(mu|sigma)
    - Combinations: {', '.join(COMBINATIONS)}
7. Error quantiles (q25/q75) for {', '.join(GAUSS_VARS)} across {', '.join(COMBINATIONS)}
"""
print(COLUMN_FAMILY_DOC)

plot_metadata_columns(ctx)
plot_sigmoid_parameters(ctx)
plot_resolution_sigmas(ctx)
plot_event_statistics(ctx)
plot_gaussian_mixtures(ctx)
plot_error_quantiles(ctx)
verify_plotted_columns(ctx)

# %%
