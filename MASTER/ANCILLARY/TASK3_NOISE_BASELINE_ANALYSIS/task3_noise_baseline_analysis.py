#!/usr/bin/env python3
"""Compare TASK_3 activation matrices using MINGO00 as noiseless baseline.

Main goal:
- use `MINGO00` as baseline and `MINGO01` as real detector,
- compute per-column and per-matrix differences,
- identify columns with large deviations that can be interpreted as noise.

Outputs include:
- matrix heatmaps (baseline, real, real-baseline),
- the corresponding matrix-cell columns as time/row traces,
- CSV ranking of columns with largest differences.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

# Ensure the repository root is on sys.path so we can import shared DATAFLOW helpers.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from MASTER.common.file_selection import extract_run_datetime_from_name
except ImportError:
    extract_run_datetime_from_name = None

DEFAULT_PATTERN = (
    str(
        REPO_ROOT
        / "STATIONS"
        / "MINGO0*"
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_3"
        / "METADATA"
        / "task_3_metadata_activation.csv"
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate MINGO01 noise by comparing TASK_3 activation matrices against MINGO00 baseline."
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern to find task_3_metadata_activation.csv files (default: %(default)s)",
    )
    parser.add_argument(
        "--outdir",
        default=str(SCRIPT_DIR / "OUTPUTS"),
        help="Directory where plots and summary CSV will be written.",
    )
    parser.add_argument(
        "--baseline",
        default="MINGO00",
        help="Baseline station treated as noiseless reference.",
    )
    parser.add_argument(
        "--target",
        default="MINGO01",
        help="Target station to compare against baseline.",
    )
    parser.add_argument(
        "--family",
        default="all",
        help=(
            "Matrix family to plot (e.g. activation_plane_signal_to_signal_initial) "
            "or 'all' (default)."
        ),
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Threshold on |delta|/baseline_std used to flag large differences.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.03,
        help="Absolute delta threshold used to flag large differences.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="If set, limit the number of rows read from each CSV (for speed).",
    )
    return parser.parse_args()


def _guess_station(path: str) -> str:
    """Get station ID (e.g. MINGO00) from the path."""
    # We'll look for the first dirname that matches a MINGO station tag (e.g. MINGO00..MINGO99)
    parts = path.split(os.sep)
    for p in parts:
        p_up = p.upper()
        if p_up.startswith("MINGO") and p_up[5:].isdigit() and 6 <= len(p_up) <= 8:
            return p
    return "UNKNOWN"


def _fallback_extract_run_datetime_from_basename(basename: str) -> pd.Timestamp | None:
    """Fallback parser for run time from a basename like mi0YYDDDHHMMSS.

    When available, we prefer the shared implementation in
    `MASTER.common.file_selection.extract_run_datetime_from_name`.
    """
    if extract_run_datetime_from_name is not None:
        return extract_run_datetime_from_name(basename)

    if not isinstance(basename, str):
        return None
    # Accept either an exact 11-digit YYDDDHHMMSS suffix or a longer trailing
    # digit group and use the last 11 digits. This mirrors the more robust
    # behavior in MASTER.common.file_selection.extract_run_datetime_from_name.
    match = re.search(r"(\d{11,})$", basename)
    if not match:
        return None

    stamp_full = match.group(1)
    # If there are more than 11 trailing digits (some names include extra
    # sequencing digits), use the final 11 digits which encode YYDDDHHMMSS.
    stamp = stamp_full[-11:]
    try:
        yy = int(stamp[0:2])
        day_of_year = int(stamp[2:5])
        hour = int(stamp[5:7])
        minute = int(stamp[7:9])
        second = int(stamp[9:11])
    except ValueError:
        return None

    if not (1 <= day_of_year <= 366):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None

    year = 2000 + yy
    try:
        return pd.to_datetime(
            f"{year}-01-01 {hour:02d}:{minute:02d}:{second:02d}"
        ) + pd.Timedelta(days=day_of_year - 1)
    except Exception:
        return None


def _load_metadata(path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, nrows=max_rows)
    df["__station"] = _guess_station(path)
    df["__row_id"] = np.arange(len(df), dtype=int)
    if "execution_timestamp" in df.columns:
        df["__execution_ts"] = pd.to_datetime(
            df["execution_timestamp"],
            format="%Y-%m-%d_%H.%M.%S",
            errors="coerce",
        )
    else:
        df["__execution_ts"] = pd.NaT

    # Prefer the runtime embedded in the raw file basename (e.g. mi0YYDDDHHMMSS)
    # over the execution-time timestamp (analysis time), which is not the actual
    # acquisition time.
    if "filename_base" in df.columns:
        df["__run_ts"] = (
            df["filename_base"].astype(str).apply(_fallback_extract_run_datetime_from_basename)
        )
        df["__execution_ts"] = df["__run_ts"].combine_first(df["__execution_ts"])

    return df


def _extract_matrix_families(columns: list[str]) -> dict[str, list[str]]:
    """Return matrix families with full P1..PN to P1..PN cells (square NxN matrices).

    This is a more flexible extractor than the original 4x4-only implementation.
    It detects families named like ``<prefix>_P<i>_to_P<j>`` and returns only
    complete square families (i.e. index sets {1..N} x {1..N}). This allows
    TASK_2 outputs (larger activation matrices) to be discovered automatically.
    """
    # Two supported naming schemes observed in metadata:
    # 1) activation_plane_..._P<i>_to_P<j>  (legacy TASK_3 style)
    # 2) activation_strip_..._P{plane}S{strip}_to_P{plane}S{strip} (TASK_2 style)
    pat_plane = re.compile(r"^(activation_plane_[a-z_]+_(?:initial|filtered))_P(\d+)_to_P(\d+)$")
    pat_plane_alt = re.compile(
        r"^(activation_[a-z0-9_]+(?:_[a-z0-9_]+)*)_P(\d+)S(\d+)_to_P(\d+)S(\d+)$"
    )

    # temporary collectors
    families_plane: dict[str, dict[tuple[int, int], str]] = {}
    families_ps: dict[str, dict[tuple[tuple[int, int], tuple[int, int]], str]] = {}

    for col in columns:
        m = pat_plane.match(col)
        if m:
            family = m.group(1)
            i = int(m.group(2))
            j = int(m.group(3))
            families_plane.setdefault(family, {})[(i, j)] = col
            continue

        m2 = pat_plane_alt.match(col)
        if m2:
            family = m2.group(1)
            pi = int(m2.group(2))
            si = int(m2.group(3))
            pj = int(m2.group(4))
            sj = int(m2.group(5))
            families_ps.setdefault(family, {})[((pi, si), (pj, sj))] = col

    out: dict[str, list[str]] = {}

    # process simple plane families (P<i>_to_P<j>)
    for family, cell_map in families_plane.items():
        ij_keys = set(cell_map.keys())
        if not ij_keys:
            continue
        rows = sorted({i for (i, _) in ij_keys})
        cols_ = sorted({j for (_, j) in ij_keys})
        if not rows or not cols_:
            continue
        if rows[0] != 1 or cols_[0] != 1:
            continue
        if rows[-1] != cols_[-1]:
            continue
        N = rows[-1]
        needed = {(i, j) for i in range(1, N + 1) for j in range(1, N + 1)}
        if set(cell_map.keys()) == needed:
            out[family] = [cell_map[(i, j)] for i in range(1, N + 1) for j in range(1, N + 1)]

    # process plane-strip families (P{plane}S{strip}_to_P{plane}S{strip})
    for family, cell_map in families_ps.items():
        # collect unique from/to positions
        from_positions = sorted({k[0] for k in cell_map.keys()})
        to_positions = sorted({k[1] for k in cell_map.keys()})
        if not from_positions or not to_positions:
            continue
        # must be the same set (square matrix over the same position index set)
        if set(from_positions) != set(to_positions):
            continue

        # infer grid dims: unique planes and unique strips
        planes = sorted({p for (p, s) in from_positions})
        strips = sorted({s for (p, s) in from_positions})
        if planes[0] != 1 or strips[0] != 1:
            continue
        # contiguous check
        if planes[-1] != planes[0] + len(planes) - 1 or strips[-1] != strips[0] + len(strips) - 1:
            continue

        S = strips[-1]
        P = planes[-1]
        M = len(from_positions)
        # flatten (plane,strip) -> single index 1..M in row-major (plane-major) order
        def flat_idx(pos: tuple[int, int]) -> int:
            p, s = pos
            return (p - 1) * S + s

        # build flattened cell map
        flat_map: dict[tuple[int, int], str] = {}
        for (from_pos, to_pos), colname in cell_map.items():
            fi = flat_idx(from_pos)
            tj = flat_idx(to_pos)
            flat_map[(fi, tj)] = colname

        # expected full square
        N = M
        needed = {(i, j) for i in range(1, N + 1) for j in range(1, N + 1)}
        if set(flat_map.keys()) == needed:
            out[family] = [flat_map[(i, j)] for i in range(1, N + 1) for j in range(1, N + 1)]

    return out


def _matrix_from_means(series: pd.Series, columns: list[str]) -> np.ndarray:
    vals = [pd.to_numeric(series.get(c, np.nan), errors="coerce") for c in columns]
    arr = np.array(vals, dtype=float)
    if arr.size == 0:
        return np.empty((0, 0))
    N = int(np.sqrt(arr.size))
    if N * N != arr.size:
        raise ValueError(f"Column list length {arr.size} is not a perfect square")
    return arr.reshape(N, N)


def _plot_matrix_triptych(
    out_path: Path,
    family: str,
    baseline_name: str,
    target_name: str,
    baseline_matrix: np.ndarray,
    target_matrix: np.ndarray,
) -> None:
    delta = target_matrix - baseline_matrix
    # support NxN matrices
    N = int(baseline_matrix.shape[0]) if baseline_matrix.size else 0
    fig, axes = plt.subplots(1, 3, figsize=(max(8, 2 * N), max(4, 1.6 * N)), constrained_layout=True)

    mats = [baseline_matrix, target_matrix, delta]
    titles = [f"{baseline_name} baseline", f"{target_name} real", f"Delta ({target_name} - {baseline_name})"]
    cmaps = ["viridis", "viridis", "coolwarm"]
    vmn = [0.0, 0.0, -np.nanmax(np.abs(delta))]
    vmx = [1.0, 1.0, np.nanmax(np.abs(delta))]

    for ax, matrix, title, cmap, vmin, vmax in zip(axes, mats, titles, cmaps, vmn, vmx):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels([str(i) for i in range(1, N + 1)])
        ax.set_yticklabels([str(i) for i in range(1, N + 1)])
        ax.set_xlabel("Target index")
        ax.set_ylabel("Given index")
        ax.set_title(title, fontsize=10)
        for i in range(N):
            for j in range(N):
                v = matrix[i, j]
                if np.isfinite(v):
                    txt = f"{v:+.3f}" if "Delta" in title else f"{v:.3f}"
                    color = "white" if (abs(v) > 0.6 and "Delta" not in title) else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
        # Create a tight, fixed-percentage colorbar so it never overwhelms the
        # small matrix axes. Use make_axes_locatable to append a narrow cax.
        try:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.03)
            fig.colorbar(im, cax=cax)
        except Exception:
            # Conservative fallback
            try:
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            except Exception:
                fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle(f"{family} matrix comparison", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_family_columns(
    out_path: Path,
    family: str,
    baseline_name: str,
    target_name: str,
    baseline_df: pd.DataFrame,
    target_df: pd.DataFrame,
    columns: list[str],
) -> None:
    # Columns are provided in row-major order for an NxN matrix
    if not columns:
        return
    N = int(np.sqrt(len(columns)))
    if N * N != len(columns):
        raise ValueError(f"Expected square number of columns for family '{family}', got {len(columns)}")
    figsize = (min(32, 3 * N), min(32, 2.5 * N))
    fig, axes = plt.subplots(N, N, figsize=figsize, sharex=False, sharey=True)
    axes_flat = np.ravel(axes)

    baseline_band_label_used_q25_q75 = False
    baseline_band_label_used_q75_q95 = False
    baseline_band_label_used_q05_q25 = False

    for idx, col in enumerate(columns):
        ax = axes_flat[idx]

        b = baseline_df[["__execution_ts", "__row_id", col]].copy()
        t = target_df[["__execution_ts", "__row_id", col]].copy()

        b[col] = pd.to_numeric(b[col], errors="coerce")
        t[col] = pd.to_numeric(t[col], errors="coerce")
        b = b.dropna(subset=[col])
        t = t.dropna(subset=[col])

        b_q05 = float(b[col].quantile(0.05)) if not b.empty else np.nan
        b_q25 = float(b[col].quantile(0.25)) if not b.empty else np.nan
        b_q75 = float(b[col].quantile(0.75)) if not b.empty else np.nan
        b_q95 = float(b[col].quantile(0.95)) if not b.empty else np.nan

        if b["__execution_ts"].notna().any() and t["__execution_ts"].notna().any():
            b = b.sort_values("__execution_ts")
            t = t.sort_values("__execution_ts")

            t_x = t["__execution_ts"]
            # q05 -> q25 (orange)
            if np.isfinite(b_q05) and np.isfinite(b_q25) and not t_x.empty:
                band_label = None
                if not baseline_band_label_used_q05_q25:
                    band_label = f"{baseline_name} q05-q25"
                    baseline_band_label_used_q05_q25 = True
                ax.fill_between(
                    t_x,
                    np.full(len(t_x), b_q05),
                    np.full(len(t_x), b_q25),
                    alpha=0.18,
                    color="tab:orange",
                    linewidth=0.0,
                    label=band_label,
                )

            # q25 -> q75 (blue)
            if np.isfinite(b_q25) and np.isfinite(b_q75) and not t_x.empty:
                band_label = None
                if not baseline_band_label_used_q25_q75:
                    band_label = f"{baseline_name} q25-q75"
                    baseline_band_label_used_q25_q75 = True
                ax.fill_between(
                    t_x,
                    np.full(len(t_x), b_q25),
                    np.full(len(t_x), b_q75),
                    alpha=0.22,
                    color="tab:blue",
                    linewidth=0.0,
                    label=band_label,
                )
            # q75 -> q95 (orange)
            if np.isfinite(b_q75) and np.isfinite(b_q95) and not t_x.empty:
                band_label = None
                if not baseline_band_label_used_q75_q95:
                    band_label = f"{baseline_name} q75-q95"
                    baseline_band_label_used_q75_q95 = True
                ax.fill_between(
                    t_x,
                    np.full(len(t_x), b_q75),
                    np.full(len(t_x), b_q95),
                    alpha=0.18,
                    color="tab:orange",
                    linewidth=0.0,
                    label=band_label,
                )
            ax.plot(t["__execution_ts"], t[col], linewidth=1.0, alpha=0.85, label=target_name)
        else:
            t_x = t["__row_id"]
            # q05 -> q25 (orange)
            if np.isfinite(b_q05) and np.isfinite(b_q25) and not t_x.empty:
                band_label = None
                if not baseline_band_label_used_q05_q25:
                    band_label = f"{baseline_name} q05-q25"
                    baseline_band_label_used_q05_q25 = True
                ax.fill_between(
                    t_x,
                    np.full(len(t_x), b_q05),
                    np.full(len(t_x), b_q25),
                    alpha=0.18,
                    color="tab:orange",
                    linewidth=0.0,
                    label=band_label,
                )

            # q25 -> q75 (blue)
            if np.isfinite(b_q25) and np.isfinite(b_q75) and not t_x.empty:
                band_label = None
                if not baseline_band_label_used_q25_q75:
                    band_label = f"{baseline_name} q25-q75"
                    baseline_band_label_used_q25_q75 = True
                ax.fill_between(
                    t_x,
                    np.full(len(t_x), b_q25),
                    np.full(len(t_x), b_q75),
                    alpha=0.22,
                    color="tab:blue",
                    linewidth=0.0,
                    label=band_label,
                )

            # q75 -> q95 (orange)
            if np.isfinite(b_q75) and np.isfinite(b_q95) and not t_x.empty:
                band_label = None
                if not baseline_band_label_used_q75_q95:
                    band_label = f"{baseline_name} q75-q95"
                    baseline_band_label_used_q75_q95 = True
                ax.fill_between(
                    t_x,
                    np.full(len(t_x), b_q75),
                    np.full(len(t_x), b_q95),
                    alpha=0.18,
                    color="tab:orange",
                    linewidth=0.0,
                    label=band_label,
                )
            ax.plot(t["__row_id"], t[col], linewidth=1.0, alpha=0.85, label=target_name)

        short = col.split(family + "_", 1)[-1]
        ax.set_title(short, fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        # Show x-axis labels only on the bottom row to avoid clutter; rotate those labels
        row = idx // N
        is_bottom = (row == N - 1)
        # hide tick labels and tick marks for non-bottom rows so the grid stays readable
        ax.tick_params(axis="x", labelrotation=45 if is_bottom else 0, labelbottom=is_bottom, bottom=is_bottom)
        if not is_bottom:
            # ensure no stray tick labels remain
            try:
                ax.set_xticklabels([])
            except Exception:
                pass
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"{family} matrix-cell columns", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_name = args.baseline.upper()
    target_name = args.target.upper()

    files = sorted(glob.glob(args.pattern, recursive=True))
    if not files:
        raise SystemExit(f"No files found matching: {args.pattern}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = _load_metadata(f, max_rows=args.max_rows)
        df["__source_file"] = os.path.basename(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    station_set = set(df_all["__station"].unique())
    if baseline_name not in station_set or target_name not in station_set:
        raise SystemExit(
            f"Stations present: {sorted(station_set)}. Need baseline={baseline_name} and target={target_name}."
        )

    df_baseline = df_all[df_all["__station"] == baseline_name].copy()
    df_target = df_all[df_all["__station"] == target_name].copy()

    families = _extract_matrix_families(list(df_all.columns))
    if not families:
        raise SystemExit("No complete activation matrix families found.")

    if args.family.lower() != "all":
        if args.family not in families:
            raise SystemExit(
                f"Family '{args.family}' not available. Choices: {sorted(families.keys())}"
            )
        families = {args.family: families[args.family]}

    baseline_means = df_baseline.mean(numeric_only=True)
    target_means = df_target.mean(numeric_only=True)
    baseline_stds = df_baseline.std(numeric_only=True)

    rows: list[dict[str, object]] = []
    for family, cols in families.items():
        baseline_matrix = _matrix_from_means(baseline_means, cols)
        target_matrix = _matrix_from_means(target_means, cols)
        delta_matrix = target_matrix - baseline_matrix

        matrix_png = outdir / f"matrix_{family}_{target_name}_minus_{baseline_name}.png"
        _plot_matrix_triptych(
            matrix_png,
            family,
            baseline_name,
            target_name,
            baseline_matrix,
            target_matrix,
        )

        columns_png = outdir / f"matrix_columns_{family}_{baseline_name}_vs_{target_name}.png"
        _plot_family_columns(
            columns_png,
            family,
            baseline_name,
            target_name,
            df_baseline,
            df_target,
            cols,
        )

        # columns are returned in row-major order; infer N and iterate
        if not cols:
            continue
        N = int(np.sqrt(len(cols)))
        if N * N != len(cols):
            raise ValueError(f"Family '{family}' column count {len(cols)} not a perfect square")
        for idx, col in enumerate(cols):
            i = idx // N + 1
            j = idx % N + 1
            b_mean = float(pd.to_numeric(baseline_means.get(col, np.nan), errors="coerce"))
            t_mean = float(pd.to_numeric(target_means.get(col, np.nan), errors="coerce"))
            b_std = float(pd.to_numeric(baseline_stds.get(col, np.nan), errors="coerce"))
            delta = t_mean - b_mean
            denom = b_std if np.isfinite(b_std) and b_std > 1e-12 else np.nan
            z = delta / denom if np.isfinite(denom) else np.nan
            rows.append(
                {
                    "family": family,
                    "cell": f"P{i}_to_P{j}",
                    "column": col,
                    f"mean_{baseline_name}": b_mean,
                    f"mean_{target_name}": t_mean,
                    f"std_{baseline_name}": b_std,
                    "delta_target_minus_baseline": delta,
                    "zscore_vs_baseline_std": z,
                    "abs_delta": abs(delta),
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["abs_delta", "zscore_vs_baseline_std"], ascending=False)
    summary_path = outdir / "matrix_cell_difference_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    sig_mask = (
        summary_df["abs_delta"].ge(float(args.delta_threshold))
        | summary_df["zscore_vs_baseline_std"].abs().ge(float(args.z_threshold))
    )
    sig_df = summary_df[sig_mask].copy()
    sig_path = outdir / "matrix_cell_significant_noise_candidates.csv"
    sig_df.to_csv(sig_path, index=False)

    # Keep a compact 1D noise metric for quick sanity checks.
    rate_candidates = [
        "streamer_rate_plane_initial_1",
        "streamer_rate_plane_initial_2",
        "streamer_rate_plane_initial_3",
        "streamer_rate_plane_initial_4",
        "streamer_rate_plane_filtered_1",
        "streamer_rate_plane_filtered_2",
        "streamer_rate_plane_filtered_3",
        "streamer_rate_plane_filtered_4",
    ]
    rate_cols = [c for c in rate_candidates if c in df_all.columns]
    if rate_cols:
        df_all = df_all.copy()
        df_all["noise_rate_mean"] = df_all[rate_cols].mean(axis=1)
        fig, ax = plt.subplots(figsize=(9, 5))
        for st, grp in df_all.groupby("__station"):
            if st not in {baseline_name, target_name}:
                continue
            ax.hist(grp["noise_rate_mean"].dropna(), bins=50, alpha=0.55, density=True, label=st)
        ax.set_xlabel("Mean streamer-rate noise proxy")
        ax.set_ylabel("Density")
        ax.set_title(f"Noise proxy comparison: {target_name} vs {baseline_name}")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(outdir / "noise_rate_histogram_baseline_vs_target.png", dpi=150)
        plt.close(fig)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {sig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
