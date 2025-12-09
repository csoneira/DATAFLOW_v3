from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMBINATIONS = ["12", "13", "14", "23", "24", "34", "123", "124", "134", "234", "1234"]
EVENT_COMBOS = COMBINATIONS + ["1", "2", "3", "4"]
GAUSS_VARS = ["x", "y", "theta", "phi", "s", "t0"]
RES_VARS = ["ystr", "tsum", "tdif"]
SIGMOID_PARAMS = ["width", "amplitude", "center"]
META_COLS = {
    "analysis_mode",
    "unc_y",
    "unc_tsum",
    "unc_tdif",
    "z_P1",
    "z_P2",
    "z_P3",
    "z_P4",
    "filename_base",
    "execution_timestamp",
}


def _filename_to_datetime(value: str) -> pd.Timestamp:
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
    except Exception:  # pragma: no cover
        return pd.NaT


class MetadataContext:
    def __init__(
        self,
        df: pd.DataFrame,
        metadata_path: Path,
        filename_col: Optional[str],
        timestamp_col: Optional[str],
        filter_col: Optional[str],
    ) -> None:
        self.df = df
        self.metadata_path = metadata_path
        self.filename_col = filename_col
        self.timestamp_col = timestamp_col
        self.filter_col = filter_col
        self.plotted_cols: set[str] = set()

    @property
    def time_col(self) -> Optional[str]:
        return "datetime" if "datetime" in self.df.columns else self.filter_col

    def record(self, *columns: Iterable[str]) -> None:
        for column in columns:
            if isinstance(column, str):
                self._add_column(column)
            else:
                for sub in column:
                    self._add_column(sub)

    def _add_column(self, column: Optional[str]) -> None:
        if column and column in self.df.columns:
            self.plotted_cols.add(column)


def load_metadata(
    station: str,
    step: int,
    task: Optional[int],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> MetadataContext:
    repo_root = Path(__file__).resolve().parents[3]
    event_dir = repo_root / "STATIONS" / station / "STAGE_1" / "EVENT_DATA" / f"STEP_{step}"
    if task is None:
        metadata_path = event_dir / "METADATA" / f"step_{step}_metadata_specific.csv"
    else:
        metadata_path = event_dir / f"TASK_{task}" / "METADATA" / f"task_{task}_metadata_specific.csv"

    if not metadata_path.exists():  # pragma: no cover
        raise FileNotFoundError(f"Cannot find metadata at {metadata_path}")

    df = pd.read_csv(metadata_path)
    filename_col = "filename_base" if "filename_base" in df.columns else None
    timestamp_col = next(
        (candidate for candidate in ("execution_time", "execution_timestamp") if candidate in df.columns),
        None,
    )

    if filename_col:
        df["datetime"] = df[filename_col].apply(_filename_to_datetime)
        df = df.sort_values(by="datetime")

    filter_col = "datetime" if "datetime" in df.columns else timestamp_col

    # Always parse the chosen time column to datetime so plotting works even without date filtering.
    if filter_col:
        if filter_col == "execution_timestamp":
            fmt = "%Y-%m-%d_%H.%M.%S"
            df[filter_col] = pd.to_datetime(df[filter_col], format=fmt, errors="coerce")
        else:
            df[filter_col] = pd.to_datetime(df[filter_col], errors="coerce")

    if filter_col and (start_date or end_date):
        start = pd.to_datetime(start_date) if start_date else df[filter_col].min()
        end = pd.to_datetime(end_date) if end_date else df[filter_col].max()
        df = df.loc[df[filter_col].between(start, end)]
        df = df.sort_values(by=filter_col)

    context = MetadataContext(
        df=df,
        metadata_path=metadata_path,
        filename_col=filename_col,
        timestamp_col=timestamp_col,
        filter_col=filter_col,
    )
    return context


def print_columns(df: pd.DataFrame) -> None:
    print("Columns:")
    for column in df.columns:
        print(f" - {column}")
    print(f"Total columns: {len(df.columns)}")


def _common_ylim(df: pd.DataFrame, columns: Iterable[str], margin: float = 0.05) -> tuple[float, float] | None:
    min_val: float | None = None
    max_val: float | None = None
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        col_min = float(series.min())
        col_max = float(series.max())
        if min_val is None or col_min < min_val:
            min_val = col_min
        if max_val is None or col_max > max_val:
            max_val = col_max
    if min_val is None or max_val is None:
        return None
    if min_val == max_val:
        padding = abs(min_val) * margin if min_val != 0 else 1.0
        return min_val - padding, max_val + padding
    padding = (max_val - min_val) * margin
    return min_val - padding, max_val + padding


def plot_tt_pairs(
    ctx: MetadataContext,
    prefix_left: str,
    prefix_right: str,
    title_prefix: str,
    ncols: int = 5,
    figsize_per_cell: tuple[int, int] = (4, 2),
    marker_size: int = 1,
) -> None:
    df = ctx.df
    tcol = ctx.time_col
    if not tcol:
        print("No time column available for plot_tt_pairs; skipping.")
        return

    left_cols = [c for c in df.columns if c.startswith(prefix_left) and c.endswith("_count")]
    right_cols = [c for c in df.columns if c.startswith(prefix_right) and c.endswith("_count")]

    def tt_id(col: str, prefix: str) -> str:
        return col[len(prefix) : -len("_count")]

    ids = sorted(
        {tt_id(c, prefix_left) for c in left_cols} | {tt_id(c, prefix_right) for c in right_cols},
        key=lambda s: (len(s), s),
    )
    if not ids:
        return

    all_columns = [f"{prefix_left}{tt}_count" for tt in ids] + [f"{prefix_right}{tt}_count" for tt in ids]
    common_ylim = _common_ylim(df, all_columns)

    n = len(ids)
    nrows = math.ceil(n / ncols)
    fig_w = ncols * figsize_per_cell[0]
    fig_h = nrows * figsize_per_cell[1] * 2.0  # extra space for z-score subplot

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    # height ratios enforce 3:1 main-to-z subplot ratio
    gs = gridspec.GridSpec(
        nrows * 2,
        ncols,
        figure=fig,
        hspace=0.15,
        height_ratios=[3, 1] * nrows,
    )

    def get_series(colname: str) -> Optional[pd.DataFrame]:
        if colname not in df.columns:
            return None
        series = df[[tcol, colname]].dropna()
        if series.empty:
            return None
        return series.sort_values(by=tcol)

    def z_scores(values: pd.Series) -> Optional[pd.Series]:
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        # Consistent estimator for normal data
        scale = mad * 1.4826
        if scale is None or scale == 0 or np.isnan(scale):
            return None
        return (values - med) / scale

    axes_pairs = []
    for idx, tt in enumerate(ids):
        grid_row = (idx // ncols) * 2
        grid_col = idx % ncols
        ax = fig.add_subplot(gs[grid_row, grid_col])
        z_ax = fig.add_subplot(gs[grid_row + 1, grid_col], sharex=ax)
        axes_pairs.append((ax, z_ax))

    for idx, ((ax, z_ax), tt) in enumerate(zip(axes_pairs, ids)):
        left_name = f"{prefix_left}{tt}_count"
        right_name = f"{prefix_right}{tt}_count"

        plotted = False

        left_series = get_series(left_name)
        right_series = get_series(right_name)

        if left_series is not None:
            ax.scatter(
                left_series[tcol],
                left_series[left_name],
                s=marker_size,
                label=f"left:{left_name}",
                color="C0",
                marker="o",
                alpha=0.85,
            )
            plotted = True
        if right_series is not None:
            ax.scatter(
                right_series[tcol],
                right_series[right_name],
                s=marker_size,
                label=f"right:{right_name}",
                color="C1",
                marker="x",
                alpha=0.85,
            )
            plotted = True

        if plotted:
            present = []
            if left_series is not None:
                present.append(prefix_left.rstrip("_"))
            if right_series is not None:
                present.append(prefix_right.rstrip("_"))
            ax.set_title(f"{tt} • {', '.join(present)}", fontsize=8)
            if common_ylim:
                ax.set_ylim(*common_ylim)
            ax.grid(True, linestyle="--", alpha=0.35)
            if left_series is not None and right_series is not None:
                ax.legend(fontsize=6)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(axis="x", labelrotation=20, labelsize=7)

            z_vals: list[float] = []
            if left_series is not None:
                z_left = z_scores(left_series[left_name])
                if z_left is not None:
                    z_ax.scatter(
                        left_series[tcol],
                        z_left,
                        s=marker_size,
                        label=f"z:{left_name}",
                        color="C0",
                        marker="o",
                        alpha=0.75,
                    )
                    z_vals.extend(z_left.tolist())
            if right_series is not None:
                z_right = z_scores(right_series[right_name])
                if z_right is not None:
                    z_ax.scatter(
                        right_series[tcol],
                        z_right,
                        s=marker_size,
                        label=f"z:{right_name}",
                        color="C1",
                        marker="x",
                        alpha=0.75,
                    )
                    z_vals.extend(z_right.tolist())
            if z_vals:
                max_abs = max(abs(v) for v in z_vals)
                z_ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
            z_ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            z_ax.grid(True, linestyle="--", alpha=0.25)
            z_ax.set_ylabel("z", fontsize=8)
            z_ax.tick_params(axis="x", labelrotation=20, labelsize=7)
            z_ax.tick_params(axis="y", labelsize=7)
            if (idx // ncols) == nrows - 1:
                z_ax.set_xlabel("Datetime", fontsize=8)
            if left_series is not None and right_series is not None:
                z_ax.legend(fontsize=6)
        else:
            ax.set_visible(False)
            z_ax.set_visible(False)

    total_slots = nrows * ncols
    for idx in range(n, total_slots):
        grid_row = (idx // ncols) * 2
        grid_col = idx % ncols
        fig.add_subplot(gs[grid_row, grid_col]).set_visible(False)
        fig.add_subplot(gs[grid_row + 1, grid_col]).set_visible(False)

    fig.suptitle(title_prefix, y=0.995)
    plt.show()

    ctx.record([c for c in [f"{prefix_left}{tt}_count" for tt in ids] if c in df.columns])
    ctx.record([c for c in [f"{prefix_right}{tt}_count" for tt in ids] if c in df.columns])


def plot_tt_matrix(
    ctx: MetadataContext,
    from_prefix: str,
    to_prefix: str,
    title_prefix: str,
) -> None:
    df = ctx.df
    tcol = ctx.time_col
    if not tcol:
        print("No time column available for plot_tt_matrix; skipping.")
        return

    pat = re.compile(rf"{from_prefix}_to_{to_prefix}_tt_(\d+)_(\d+)_count")
    mapping: dict[tuple[str, str], str] = {}
    from_set = set()
    to_set = set()

    for col in df.columns:
        m = pat.match(col)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        from_set.add(a)
        to_set.add(b)
        mapping.setdefault((a, b), col)

    if not mapping:
        return

    common_ylim = _common_ylim(df, mapping.values())

    from_list = sorted(from_set, key=lambda s: (len(s), s))
    to_list = sorted(to_set, key=lambda s: (len(s), s))
    nrows = len(to_list)
    ncols = len(from_list)
    fig_w = max(6, ncols * 3)
    fig_h = max(3, nrows * 2.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[a] for a in axes])

    for i_r, to in enumerate(to_list):
        for j_c, frm in enumerate(from_list):
            ax = axes[i_r, j_c]
            colname = mapping.get((frm, to))
            if colname and colname in df.columns:
                series = df[[tcol, colname]].dropna()
                if not series.empty:
                    series = series.sort_values(by=tcol)
                    ax.scatter(series[tcol], series[colname], s=1, color="C0", marker="o", alpha=0.9)
                    ax.set_title(f"{frm}→{to}\n{colname}", fontsize=8)
                    if common_ylim:
                        ax.set_ylim(*common_ylim)
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)

            ax.grid(True, linestyle="--", alpha=0.3)
            if j_c == 0:
                ax.set_ylabel("Count")
            if i_r == nrows - 1:
                ax.set_xlabel("Datetime")

    fig.suptitle(title_prefix)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    ctx.record(v for v in mapping.values() if v in df.columns)


def expected_columns_by_family() -> dict[str, set[str]]:
    sigmoid_cols = {
        f"sigmoid_{param}_{combo}"
        for param in SIGMOID_PARAMS
        for combo in COMBINATIONS
    }
    sigmoid_cols |= {f"background_slope_{combo}" for combo in COMBINATIONS}
    sigmoid_cols |= {f"fit_normalization_{combo}" for combo in COMBINATIONS}

    res_cols = {
        f"res_{var}_{plane}_{combo}_sigma"
        for var in RES_VARS
        for plane in range(1, 5)
        for combo in COMBINATIONS
    }
    ext_res_cols = {
        f"ext_res_{var}_{plane}_{combo}_sigma"
        for var in RES_VARS
        for plane in range(1, 5)
        for combo in COMBINATIONS
    }
    event_cols = {
        f"list_tt_{combo}_count" for combo in EVENT_COMBOS
    } | {f"list_to_fit_tt_{combo}_0_count" for combo in COMBINATIONS} | {"fit_tt_0_count"}

    gauss_cols = set()
    for var in GAUSS_VARS:
        for combo in COMBINATIONS:
            gauss_cols.add(f"{var}_err_{combo}_gauss1_mu")
            gauss_cols.add(f"{var}_err_{combo}_gauss1_sigma")
            gauss_cols.add(f"{var}_err_{combo}_gauss1_amp")
            gauss_cols.add(f"{var}_err_{combo}_gauss2_mu")
            gauss_cols.add(f"{var}_err_{combo}_gauss2_sigma")

    quantile_cols = {
        f"{var}_err_{combo}_{suffix}"
        for var in GAUSS_VARS
        for combo in COMBINATIONS
        for suffix in ("q25", "q75")
    }

    families = {
        "metadata": set(META_COLS),
        "sigmoid_fits": sigmoid_cols,
        "intrinsic_resolutions": res_cols,
        "external_resolutions": ext_res_cols,
        "event_counts": event_cols,
        "gaussian_errors": gauss_cols,
        "quantiles": quantile_cols,
    }
    return families


EXPECTED_COLUMNS = set().union(*(family for family in expected_columns_by_family().values()))
