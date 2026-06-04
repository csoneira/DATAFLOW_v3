cd /home/mingo/DATAFLOW_v3

python3 - <<'PY'
from pathlib import Path
import json
import textwrap

OUTPUT_PATH = Path("/home/mingo/DATAFLOW_v3/MASTER/JUPYTER_NOTEBOOKS/task5_full_column_test.ipynb")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def md(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip() + "\n",
    }

def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip() + "\n",
    }

cells = [
    md("""
    # Task 5 random-file explorer

    This notebook loads one random Task 5 input parquet file from MINGO04.

    Search order:

    1. `UNPROCESSED_DIRECTORY`
    2. `COMPLETED_DIRECTORY`

    It provides quick schema inspection, histograms, scatter plots, hexbin plots,
    configurable pairwise comparisons, `tt_task5_post` time-binned comparisons,
    correlation matrices, and plane/strip quick-look plots.
    """),

    code(r"""
    from pathlib import Path
    from itertools import combinations
    import random
    import math

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    pd.set_option("display.max_columns", 250)
    pd.set_option("display.max_rows", 250)

    BASE = Path("/home/mingo/DATAFLOW_v3/STATIONS/MINGO04/STAGE_1/EVENT_DATA/STEP_1/TASK_5/INPUT_FILES")
    UNPROCESSED_DIR = BASE / "UNPROCESSED_DIRECTORY"
    COMPLETED_DIR = BASE / "COMPLETED_DIRECTORY"

    RANDOM_SEED = None
    MAX_ROWS_FOR_PLOTS = 100_000
    MAX_ROWS_FOR_SCATTER = 50_000
    HEXBIN_GRIDSIZE = 80

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    """),

    md("## 1. Select and load one random parquet file"),

    code(r"""
    def find_parquet_files(directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(directory.glob("*.parquet"))

    unprocessed_files = find_parquet_files(UNPROCESSED_DIR)
    completed_files = find_parquet_files(COMPLETED_DIR)

    if unprocessed_files:
        selected_pool = "UNPROCESSED_DIRECTORY"
        selected_file = random.choice(unprocessed_files)
    elif completed_files:
        selected_pool = "COMPLETED_DIRECTORY"
        selected_file = random.choice(completed_files)
    else:
        raise FileNotFoundError(
            "No parquet files found in either:\\n"
            f"  {UNPROCESSED_DIR}\\n"
            f"  {COMPLETED_DIR}"
        )

    print(f"Selected pool: {selected_pool}")
    print(f"Selected file: {selected_file}")

    df = pd.read_parquet(selected_file)

    print(f"Loaded shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    display(df.head())
    """),

    md("## 2. Schema and missing-value overview"),

    code(r"""
    print("Columns:")
    for col in df.columns:
        print(col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    print(f"\\nNumeric columns: {len(numeric_cols)}")
    print(f"Non-numeric columns: {len(non_numeric_cols)}")

    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_fraction")
        .to_frame()
    )
    display(missing.head(80))

    if numeric_cols:
        display(df[numeric_cols].describe().T)
    """),

    md("## 3. Useful column groups"),

    code(r"""
    def cols_matching(prefix=None, suffix=None, contains=None):
        cols = list(df.columns)
        if prefix is not None:
            cols = [c for c in cols if c.startswith(prefix)]
        if suffix is not None:
            cols = [c for c in cols if c.endswith(suffix)]
        if contains is not None:
            cols = [c for c in cols if contains in c]
        return cols

    groups = {
        "event": cols_matching(prefix="event_"),
        "event_det": cols_matching(prefix="event_det_"),
        "event_tim": cols_matching(prefix="event_tim_"),
        "plane_qsum": [f"p{i}_qsum" for i in range(1, 5) if f"p{i}_qsum" in df.columns],
        "plane_tsum": [f"p{i}_tsum" for i in range(1, 5) if f"p{i}_tsum" in df.columns],
        "plane_ypos": [f"p{i}_ypos" for i in range(1, 5) if f"p{i}_ypos" in df.columns],
        "strip_qsum": [c for c in df.columns if c.endswith("_qsum") and "_s" in c],
        "strip_tsum": [c for c in df.columns if c.endswith("_tsum") and "_s" in c],
        "residuals": [c for c in df.columns if "_res" in c],
        "task_times": [c for c in df.columns if c.startswith("tt_task") or c.startswith("transferred_")],
        "topology": [c for c in df.columns if c.startswith("topology_")],
        "filters": [c for c in df.columns if c.startswith("filter_")],
    }

    for name, cols in groups.items():
        print(f"\\n{name} ({len(cols)}):")
        print(cols)
    """),

    md("## 4. Histogram helper"),

    code(r"""
    def plot_hist(columns, bins=100, log_y=False, max_rows=MAX_ROWS_FOR_PLOTS):
        columns = [c for c in columns if c in df.columns]
        if not columns:
            print("No requested columns found.")
            return

        for col in columns:
            values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                print(f"{col}: no finite values")
                continue
            if len(values) > max_rows:
                values = values.sample(max_rows, random_state=RANDOM_SEED)

            plt.figure(figsize=(8, 5))
            plt.hist(values, bins=bins)
            if log_y:
                plt.yscale("log")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.title(f"Distribution of {col}")
            plt.grid(True, alpha=0.3)
            plt.show()

    plot_hist(
        ["event_charge", "event_x", "event_y", "event_theta", "event_phi", "tt_task5_post"],
        bins=100,
    )
    """),

    md("## 5. Scatter and hexbin helpers"),

    code(r"""
    def clean_xy(x_col: str, y_col: str, max_rows: int | None = None) -> pd.DataFrame:
        if x_col not in df.columns or y_col not in df.columns:
            missing = [c for c in [x_col, y_col] if c not in df.columns]
            print(f"Missing column(s): {missing}")
            return pd.DataFrame()

        local = df[[x_col, y_col]].copy()
        local[x_col] = pd.to_numeric(local[x_col], errors="coerce")
        local[y_col] = pd.to_numeric(local[y_col], errors="coerce")
        local = local.replace([np.inf, -np.inf], np.nan).dropna()

        if max_rows is not None and len(local) > max_rows:
            local = local.sample(max_rows, random_state=RANDOM_SEED)

        return local

    def scatter_xy(x_col: str, y_col: str, max_rows: int = MAX_ROWS_FOR_SCATTER, alpha: float = 0.25):
        local = clean_xy(x_col, y_col, max_rows=max_rows)
        if local.empty:
            return

        plt.figure(figsize=(7, 6))
        plt.scatter(local[x_col], local[y_col], s=4, alpha=alpha)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Scatter: {y_col} vs {x_col}  (N={len(local):,})")
        plt.grid(True, alpha=0.3)
        plt.show()

    def hexbin_xy(x_col: str, y_col: str, gridsize: int = HEXBIN_GRIDSIZE, mincnt: int = 1):
        local = clean_xy(x_col, y_col, max_rows=MAX_ROWS_FOR_PLOTS)
        if local.empty:
            return

        plt.figure(figsize=(7, 6))
        hb = plt.hexbin(local[x_col], local[y_col], gridsize=gridsize, mincnt=mincnt)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Hexbin: {y_col} vs {x_col}  (N={len(local):,})")
        plt.colorbar(hb, label="count")
        plt.grid(True, alpha=0.3)
        plt.show()

    scatter_xy("event_x", "event_y")
    hexbin_xy("event_x", "event_y")
    """),

    md("## 6. Configurable comparison groups"),

    code(r"""
    # Edit this list freely.
    comparison_columns = [
        ["event_x", "event_y", "event_charge"],
        ["event_theta", "event_phi", "p1_s1_qsum", "p1_s2_qsum", "p1_s3_qsum", "p1_s4_qsum"],
        ["p1_qsum", "p2_qsum", "p3_qsum", "p4_qsum"],
        ["p1_tsum", "p2_tsum", "p3_tsum", "p4_tsum"],
        ["p1_ypos", "p2_ypos", "p3_ypos", "p4_ypos"],
        ["tt_task0_raw", "tt_task1_clean", "tt_task2_cal", "tt_task3_list", "tt_task4_fit", "tt_task5_post"],
    ]

    def available_pairs_from_groups(groups):
        pairs = []
        for group_idx, group in enumerate(groups, start=1):
            available = [c for c in group if c in df.columns]
            missing = [c for c in group if c not in df.columns]
            if missing:
                print(f"Group {group_idx}: missing columns skipped: {missing}")

            for x_col, y_col in combinations(available, 2):
                pairs.append((group_idx, x_col, y_col))

        return pairs

    pairs = available_pairs_from_groups(comparison_columns)
    print(f"Available pairs: {len(pairs)}")
    display(pd.DataFrame(pairs, columns=["group", "x", "y"]).head(50))
    """),

    md("## 7. Plot all pairs in a selected group"),

    code(r"""
    def plot_group(group_index: int, plot_type: str = "hexbin"):
        selected = [(x, y) for g, x, y in pairs if g == group_index]
        if not selected:
            print(f"No pairs available for group {group_index}")
            return

        for x_col, y_col in selected:
            if plot_type == "scatter":
                scatter_xy(x_col, y_col)
            elif plot_type == "hexbin":
                hexbin_xy(x_col, y_col)
            else:
                raise ValueError("plot_type must be 'scatter' or 'hexbin'")

    plot_group(1, plot_type="hexbin")
    # plot_group(2, plot_type="scatter")
    """),

    md("## 8. Time-binned comparison using `tt_task5_post`"),

    code(r"""
    def make_time_bins(time_col: str = "tt_task5_post", n_bins: int = 4) -> pd.Series | None:
        if time_col not in df.columns:
            print(f"Time column not found: {time_col}")
            return None

        t = pd.to_numeric(df[time_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = t.dropna()

        if valid.empty:
            print(f"Time column has no finite values: {time_col}")
            return None

        bins = pd.qcut(t, q=n_bins, duplicates="drop")
        print(f"Created {bins.dropna().nunique()} time bins from {time_col}")
        return bins

    def time_binned_hexbin(
        x_col: str,
        y_col: str,
        time_col: str = "tt_task5_post",
        n_bins: int = 4,
        gridsize: int = HEXBIN_GRIDSIZE,
    ):
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Missing x/y column: {x_col}, {y_col}")
            return

        bins = make_time_bins(time_col, n_bins)
        if bins is None:
            hexbin_xy(x_col, y_col, gridsize=gridsize)
            return

        local = df[[x_col, y_col]].copy()
        local[x_col] = pd.to_numeric(local[x_col], errors="coerce")
        local[y_col] = pd.to_numeric(local[y_col], errors="coerce")
        local["_time_bin"] = bins
        local = local.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col, "_time_bin"])

        if local.empty:
            print("No finite rows for time-binned plot.")
            return

        if len(local) > MAX_ROWS_FOR_PLOTS:
            local = local.sample(MAX_ROWS_FOR_PLOTS, random_state=RANDOM_SEED)

        xlim = (local[x_col].min(), local[x_col].max())
        ylim = (local[y_col].min(), local[y_col].max())

        unique_bins = list(local["_time_bin"].cat.categories)
        n = len(unique_bins)
        ncols = min(2, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)

        for ax, bin_value in zip(axes.ravel(), unique_bins):
            sub = local[local["_time_bin"] == bin_value]
            if sub.empty:
                ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
            else:
                hb = ax.hexbin(sub[x_col], sub[y_col], gridsize=gridsize, mincnt=1)
                fig.colorbar(hb, ax=ax, label="count")

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{bin_value}\\nN={len(sub):,}")
            ax.grid(True, alpha=0.3)

        for ax in axes.ravel()[n:]:
            ax.axis("off")

        fig.suptitle(f"Time-binned hexbin: {y_col} vs {x_col}, binned by {time_col}", y=1.02)
        fig.tight_layout()
        plt.show()

    def time_binned_scatter(
        x_col: str,
        y_col: str,
        time_col: str = "tt_task5_post",
        n_bins: int = 4,
        max_rows: int = MAX_ROWS_FOR_SCATTER,
    ):
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Missing x/y column: {x_col}, {y_col}")
            return

        bins = make_time_bins(time_col, n_bins)
        if bins is None:
            scatter_xy(x_col, y_col, max_rows=max_rows)
            return

        local = df[[x_col, y_col]].copy()
        local[x_col] = pd.to_numeric(local[x_col], errors="coerce")
        local[y_col] = pd.to_numeric(local[y_col], errors="coerce")
        local["_time_bin"] = bins
        local = local.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col, "_time_bin"])

        if local.empty:
            print("No finite rows for time-binned plot.")
            return

        if len(local) > max_rows:
            local = local.sample(max_rows, random_state=RANDOM_SEED)

        xlim = (local[x_col].min(), local[x_col].max())
        ylim = (local[y_col].min(), local[y_col].max())

        unique_bins = list(local["_time_bin"].cat.categories)
        n = len(unique_bins)
        ncols = min(2, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)

        for ax, bin_value in zip(axes.ravel(), unique_bins):
            sub = local[local["_time_bin"] == bin_value]
            if sub.empty:
                ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.scatter(sub[x_col], sub[y_col], s=4, alpha=0.25)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{bin_value}\\nN={len(sub):,}")
            ax.grid(True, alpha=0.3)

        for ax in axes.ravel()[n:]:
            ax.axis("off")

        fig.suptitle(f"Time-binned scatter: {y_col} vs {x_col}, binned by {time_col}", y=1.02)
        fig.tight_layout()
        plt.show()

    time_binned_hexbin("event_x", "event_y", time_col="tt_task5_post", n_bins=4)
    # time_binned_scatter("event_theta", "event_phi", time_col="tt_task5_post", n_bins=4)
    """),

    md("## 9. Correlation matrix"),

    code(r"""
    corr_columns = [
        "event_x", "event_y", "event_xp", "event_yp",
        "event_theta", "event_phi", "event_charge",
        "p1_qsum", "p2_qsum", "p3_qsum", "p4_qsum",
        "p1_tsum", "p2_tsum", "p3_tsum", "p4_tsum",
        "tt_task5_post",
    ]

    def plot_corr_matrix(columns):
        columns = [c for c in columns if c in df.columns]
        if len(columns) < 2:
            print("Need at least two available columns.")
            return

        local = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

        if len(local) > MAX_ROWS_FOR_PLOTS:
            local = local.sample(MAX_ROWS_FOR_PLOTS, random_state=RANDOM_SEED)

        corr = local.corr(method="spearman", min_periods=100)

        plt.figure(figsize=(0.6 * len(columns) + 4, 0.6 * len(columns) + 4))
        im = plt.imshow(corr, vmin=-1, vmax=1)
        plt.xticks(range(len(columns)), columns, rotation=90)
        plt.yticks(range(len(columns)), columns)
        plt.colorbar(im, label="Spearman correlation")
        plt.title("Spearman correlation matrix")
        plt.tight_layout()
        plt.show()

        display(corr)

    plot_corr_matrix(corr_columns)
    """),

    md("## 10. Plane and strip quick-look plots"),

    code(r"""
    def plot_plane_quantity(quantity: str):
        cols = [f"p{i}_{quantity}" for i in range(1, 5) if f"p{i}_{quantity}" in df.columns]
        if not cols:
            print(f"No plane columns found for quantity: {quantity}")
            return
        plot_hist(cols, bins=100, log_y=False)

    def plot_strip_quantity_grid(quantity: str, bins=80):
        cols = []
        for plane in range(1, 5):
            for strip in range(1, 5):
                col = f"p{plane}_s{strip}_{quantity}"
                if col in df.columns:
                    cols.append((plane, strip, col))

        if not cols:
            print(f"No strip columns found for quantity: {quantity}")
            return

        fig, axes = plt.subplots(4, 4, figsize=(18, 14), squeeze=False)

        for plane, strip, col in cols:
            ax = axes[plane - 1][strip - 1]
            values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > MAX_ROWS_FOR_PLOTS:
                values = values.sample(MAX_ROWS_FOR_PLOTS, random_state=RANDOM_SEED)
            ax.hist(values, bins=bins)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)

        for ax in axes.ravel():
            if not ax.get_title():
                ax.axis("off")

        fig.tight_layout()
        plt.show()

    plot_plane_quantity("qsum")
    plot_strip_quantity_grid("qsum")
    """),

    md("## 11. Manual save example"),

    code(r"""
    # Example:
    #
    # local = clean_xy("event_x", "event_y", max_rows=MAX_ROWS_FOR_SCATTER)
    # plt.figure(figsize=(7, 6))
    # plt.scatter(local["event_x"], local["event_y"], s=4, alpha=0.25)
    # plt.xlabel("event_x")
    # plt.ylabel("event_y")
    # plt.title("event_y vs event_x")
    # plt.grid(True, alpha=0.3)
    # plt.savefig("event_y_vs_event_x.png", dpi=150, bbox_inches="tight")
    # plt.show()
    """),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Created notebook: {OUTPUT_PATH}")
PY
