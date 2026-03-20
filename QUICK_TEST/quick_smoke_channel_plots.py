#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "FIGURES"
OUT_DIR.mkdir(exist_ok=True)


def normalize_tt_label(value: object) -> str:
    if pd.isna(value):
        return "0"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "0"
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits or "0"


def compute_tt(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    tt_str = pd.Series("", index=df.index, dtype="object")
    for plane in range(1, 5):
        charge_columns = [
            col
            for col in [f"Q{plane}_{side}_{strip}" for side in ("F", "B") for strip in range(1, 5)]
            if col in df.columns
        ]
        if not charge_columns:
            continue
        has_charge = df.loc[:, charge_columns].ne(0).any(axis=1)
        tt_str = tt_str.where(~has_charge, tt_str + str(plane))
    df.loc[:, column_name] = tt_str.replace("", "0").astype(int)
    return df


TT_COLOR_LABELS = ("0", "1", "2", "3", "4", "12", "13", "14", "23", "24", "34", "123", "124", "134", "234", "1234")
TT_COLOR_CMAP = plt.get_cmap("tab20")
TT_COLOR_MAP = {
    tt_label: TT_COLOR_CMAP(idx % TT_COLOR_CMAP.N)
    for idx, tt_label in enumerate(TT_COLOR_LABELS)
}
TT_COLOR_DEFAULT = (0.45, 0.45, 0.45, 1.0)


def get_tt_color(tt_value: object):
    return TT_COLOR_MAP.get(normalize_tt_label(tt_value), TT_COLOR_DEFAULT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test channel-pair overlays for removed Task 1 rows.")
    parser.add_argument("--track-removed-rows", action="store_true", help="Save removed/original parquet outputs and overlay removed rows.")
    parser.add_argument("--basename", default="smoke_channel_pairs", help="Base name for saved artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max-figures", type=int, default=18, help="Maximum number of PNGs to save.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_rows = 600
    pair_list = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    removed_marker = "x"
    removed_marker_size = 30
    removed_marker_alpha = 0.9
    channel_pair_min_events = 10
    sample_size = 240

    cols: dict[str, np.ndarray] = {}
    for plane in range(1, 5):
        for strip in range(1, 5):
            for side in ("F", "B"):
                q_col = f"Q{plane}_{side}_{strip}"
                t_col = f"T{plane}_{side}_{strip}"
                q_vals = rng.exponential(scale=55.0, size=n_rows)
                t_vals = rng.normal(loc=0.0, scale=10.0, size=n_rows)
                q_mask = rng.random(n_rows) < 0.3
                t_mask = rng.random(n_rows) < 0.4
                q_vals[~q_mask] = 0.0
                t_vals[~t_mask] = 0.0
                cols[q_col] = q_vals
                cols[t_col] = t_vals

    df = pd.DataFrame(cols)
    df["raw_tt"] = rng.choice([12, 13, 14, 23, 24, 34], size=n_rows, p=[0.15, 0.15, 0.15, 0.2, 0.2, 0.15])

    working_df = df.copy()
    removed_rows_df = working_df.iloc[0:0].copy()
    tracking_base_index = working_df.index.copy()
    original_columns_store: dict[str, pd.Series] = {}

    def snapshot_original_columns_once(column_names: list[str]) -> None:
        if not args.track_removed_rows:
            return
        for col in column_names:
            if col in working_df.columns and col not in original_columns_store:
                original_columns_store[col] = working_df[col].copy()

    def restore_original_values(rows: pd.DataFrame) -> pd.DataFrame:
        if not args.track_removed_rows or rows.empty:
            return rows
        restored = rows.copy()
        for col, original_series in original_columns_store.items():
            if col in restored.columns:
                restored.loc[:, col] = original_series.reindex(restored.index)
        return restored

    def append_removed_rows(rows: pd.DataFrame) -> None:
        nonlocal removed_rows_df
        nonlocal_removed_rows = restore_original_values(rows)
        if not args.track_removed_rows or nonlocal_removed_rows.empty:
            return
        removed_rows_df = pd.concat([removed_rows_df, nonlocal_removed_rows], ignore_index=False, sort=False)

    component_cols = [col for col in working_df.columns if col.startswith(("Q", "T"))]
    q_front_cols = [col for col in working_df.columns if col.startswith("Q") and "_F_" in col]
    q_back_cols = [col for col in working_df.columns if col.startswith("Q") and "_B_" in col]

    # Force deterministic removal groups so the smoke plots always show overlays.
    all_zero_idx = working_df.index[:20]
    qfb_idx = working_df.index[20:50]
    clean_low_idx = working_df.index[50:90]
    bounds_idx = working_df.index[90:130]

    snapshot_original_columns_once(component_cols)
    working_df.loc[all_zero_idx, component_cols] = 0.0
    working_df.loc[qfb_idx, q_front_cols] = 0.0
    for col in [name for name in component_cols if name.startswith(("Q2_", "Q3_", "Q4_"))]:
        working_df.loc[clean_low_idx, col] = 0.0

    bounded_q_cols = [f"Q1_F_{strip}" for strip in range(1, 5)] + [f"Q2_B_{strip}" for strip in range(1, 5)]
    bounded_t_cols = [f"T3_F_{strip}" for strip in range(1, 5)] + [f"T4_B_{strip}" for strip in range(1, 5)]
    snapshot_original_columns_once(bounded_q_cols + bounded_t_cols)
    working_df.loc[bounds_idx, bounded_q_cols] = 350.0
    working_df.loc[bounds_idx, bounded_t_cols] = 55.0

    for col in [name for name in bounded_q_cols if name in working_df.columns]:
        out_of_bounds = (working_df[col] < 5.0) | (working_df[col] > 180.0)
        working_df.loc[out_of_bounds, col] = 0.0
    for col in [name for name in bounded_t_cols if name in working_df.columns]:
        out_of_bounds = (working_df[col] < -25.0) | (working_df[col] > 25.0)
        working_df.loc[out_of_bounds, col] = 0.0

    for plane in range(1, 5):
        for strip in range(1, 5):
            col_f = f"T{plane}_F_{strip}"
            col_b = f"T{plane}_B_{strip}"
            if col_f not in working_df.columns or col_b not in working_df.columns:
                continue
            zero_or_mask = (working_df[col_f] == 0) | (working_df[col_b] == 0)
            changed_mask = zero_or_mask & ~((working_df[col_f] == 0) & (working_df[col_b] == 0))
            if changed_mask.any():
                snapshot_original_columns_once([col_f, col_b])
                working_df.loc[zero_or_mask, col_f] = 0.0
                working_df.loc[zero_or_mask, col_b] = 0.0

    working_df = compute_tt(working_df, "clean_tt")

    qfb_mask = (working_df[q_front_cols] != 0).any(axis=1) & (working_df[q_back_cols] != 0).any(axis=1)
    append_removed_rows(working_df.loc[~qfb_mask].copy())
    working_df = working_df.loc[qfb_mask].copy()

    component_cols_filtered = [col for col in working_df.columns if col.startswith(("Q", "T"))]
    all_zero_mask = (working_df[component_cols_filtered].fillna(0) == 0).all(axis=1)
    append_removed_rows(working_df.loc[all_zero_mask].copy())
    working_df = working_df.loc[~all_zero_mask].copy()

    clean_tt_mask = working_df["clean_tt"].notna() & (working_df["clean_tt"] >= 10)
    append_removed_rows(working_df.loc[~clean_tt_mask].copy())
    working_df = working_df.loc[clean_tt_mask].copy()

    if args.track_removed_rows:
        removed_rows_path = OUT_DIR / f"removed_rows_{args.basename}.parquet"
        removed_rows_csv_path = OUT_DIR / f"removed_rows_{args.basename}.csv"
        original_cols_path = OUT_DIR / f"original_cols_{args.basename}.parquet"
        original_columns_df = pd.DataFrame(
            {col: series.reindex(tracking_base_index) for col, series in original_columns_store.items()},
            index=tracking_base_index,
        )
        removed_rows_df.to_parquet(removed_rows_path, engine="pyarrow", compression="zstd", index=True)
        removed_rows_df.to_csv(removed_rows_csv_path, index=True)
        original_columns_df.to_parquet(original_cols_path, engine="pyarrow", compression="zstd", index=True)
    else:
        removed_rows_path = None
        removed_rows_csv_path = None
        original_cols_path = None

    fig_count = 0
    for pair in pair_list:
        pi, pj = pair
        retained_tt = working_df["raw_tt"].apply(normalize_tt_label).astype(str)
        retained_mask = retained_tt.str.contains(str(pi)) & retained_tt.str.contains(str(pj))
        retained_pair_df = working_df.loc[retained_mask]

        removed_pair_df = removed_rows_df.iloc[0:0].copy()
        if args.track_removed_rows and "raw_tt" in removed_rows_df.columns:
            removed_tt = removed_rows_df["raw_tt"].apply(normalize_tt_label).astype(str)
            removed_mask = removed_tt.str.contains(str(pi)) & removed_tt.str.contains(str(pj))
            removed_pair_df = removed_rows_df.loc[removed_mask]

        total_pair_events = len(retained_pair_df) + len(removed_pair_df)
        if total_pair_events < channel_pair_min_events:
            continue

        sampled_retained = retained_pair_df.sample(
            n=min(len(retained_pair_df), sample_size),
            random_state=args.seed,
        ) if not retained_pair_df.empty else retained_pair_df.copy()

        retained_row_tt = sampled_retained["raw_tt"].apply(normalize_tt_label).astype(str) if "raw_tt" in sampled_retained.columns else pd.Series(dtype=str)
        removed_row_tt = removed_pair_df["raw_tt"].apply(normalize_tt_label).astype(str) if "raw_tt" in removed_pair_df.columns else pd.Series(dtype=str)
        unique_tts = sorted(set(retained_row_tt.unique()).union(set(removed_row_tt.unique())))
        if not unique_tts:
            unique_tts = [f"{pi}{pj}"]
        tt_color_map = {tt_label: get_tt_color(tt_label) for tt_label in unique_tts}
        retained_colors = np.array([tt_color_map[label] for label in retained_row_tt], dtype=object)
        removed_colors = np.array([tt_color_map[label] for label in removed_row_tt], dtype=object)

        channels = [(plane, strip, side) for plane in (pi, pj) for strip in range(1, 5) for side in ("F", "B")]
        retained_chan: dict[tuple[int, int], np.ndarray] = {}
        removed_chan: dict[tuple[int, int], np.ndarray] = {}
        for idx, (plane, strip, side) in enumerate(channels):
            for var_idx, col in enumerate((f"Q{plane}_{side}_{strip}", f"T{plane}_{side}_{strip}")):
                retained_vals = sampled_retained[col].fillna(0).to_numpy(dtype=float) if col in sampled_retained.columns else np.zeros(len(sampled_retained), dtype=float)
                removed_vals = removed_pair_df[col].fillna(0).to_numpy(dtype=float) if col in removed_pair_df.columns else np.zeros(len(removed_pair_df), dtype=float)
                if retained_vals.size > 0:
                    lo, hi = np.nanpercentile(retained_vals, [1, 99])
                    retained_vals = np.clip(retained_vals, lo, hi)
                retained_chan[(idx, var_idx)] = retained_vals
                removed_chan[(idx, var_idx)] = removed_vals

        col_ranges: dict[int, tuple[float, float]] = {}
        for var_idx in (0, 1):
            all_vals = np.concatenate(
                [retained_chan.get((idx, var_idx), np.zeros(0, dtype=float)) for idx in range(len(channels))]
                + [removed_chan.get((idx, var_idx), np.zeros(0, dtype=float)) for idx in range(len(channels))]
            )
            nonzero = all_vals[all_vals != 0]
            if nonzero.size > 1:
                lo, hi = np.nanpercentile(nonzero, [1.0, 99.0])
            elif nonzero.size == 1:
                lo = nonzero[0] - 1.0
                hi = nonzero[0] + 1.0
            else:
                lo, hi = (0.0, 1.0)
            pad = max(1e-3, 0.03 * (hi - lo))
            col_ranges[var_idx] = (lo - pad, hi + pad)

        for ai in range(len(channels)):
            for bj in range(ai, len(channels)):
                ai_q = retained_chan.get((ai, 0), np.zeros(0, dtype=float))
                ai_t = retained_chan.get((ai, 1), np.zeros(0, dtype=float))
                bj_q = retained_chan.get((bj, 0), np.zeros(0, dtype=float))
                bj_t = retained_chan.get((bj, 1), np.zeros(0, dtype=float))
                figure_effective_events = int((((ai_q != 0) & (ai_t != 0)) | ((bj_q != 0) & (bj_t != 0))).sum())
                if figure_effective_events < channel_pair_min_events:
                    continue
                fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex="col", sharey="row")
                same_channel = ai == bj
                for r in range(2):
                    for c in range(2):
                        ax = axes[r][c]
                        if same_channel and c > r:
                            ax.set_visible(False)
                            continue
                        ax.set_xticks([])
                        ax.set_yticks([])

                        col_x = retained_chan.get((ai, c), np.zeros(0, dtype=float))
                        col_y = retained_chan.get((bj, r), np.zeros(0, dtype=float))
                        removed_x = np.clip(removed_chan.get((ai, c), np.zeros(0, dtype=float)), *col_ranges[c])
                        removed_y = np.clip(removed_chan.get((bj, r), np.zeros(0, dtype=float)), *col_ranges[r])

                        if same_channel and c == r:
                            var_name = "Q" if c == 0 else "T"
                            for tt_label in unique_tts:
                                tt_mask = retained_row_tt.values == tt_label
                                vals_tt = col_x[tt_mask]
                                vals_tt = vals_tt[vals_tt != 0]
                                if vals_tt.size > 1:
                                    hist_kwargs = dict(
                                        bins=30,
                                        histtype="step",
                                        color=tt_color_map[tt_label],
                                        linewidth=1.2,
                                        log=True,
                                        label=f"TT={tt_label}" if len(unique_tts) > 1 else None,
                                    )
                                    if var_name == "T":
                                        hist_kwargs["orientation"] = "horizontal"
                                    ax.hist(vals_tt, **hist_kwargs)
                            removed_vals = removed_x[removed_x != 0]
                            if args.track_removed_rows and removed_vals.size > 0:
                                hist_kwargs = dict(
                                    bins=30,
                                    histtype="step",
                                    color="lightgrey",
                                    linewidth=1.6,
                                    linestyle="--",
                                    log=True,
                                    label="Removed",
                                )
                                if var_name == "T":
                                    hist_kwargs["orientation"] = "horizontal"
                                ax.hist(removed_vals, **hist_kwargs)
                            if len(unique_tts) > 1 or (args.track_removed_rows and removed_vals.size > 0):
                                ax.legend(fontsize=6)
                            if var_name == "T":
                                ax.set_xlabel("Counts (log)")
                                ax.set_ylabel("T (ns)")
                                ax.set_ylim(col_ranges[c])
                            else:
                                ax.set_xlabel("Q (fC)")
                                ax.set_ylabel("Counts (log)")
                                ax.set_xlim(col_ranges[c])
                        else:
                            retained_mask = (col_x != 0) & (col_y != 0)
                            if np.any(retained_mask):
                                ax.scatter(
                                    col_x[retained_mask],
                                    col_y[retained_mask],
                                    s=12,
                                    alpha=0.75,
                                    linewidths=0,
                                    c=retained_colors[retained_mask].tolist(),
                                    edgecolors="none",
                                )
                            if args.track_removed_rows:
                                removed_mask = (removed_x != 0) & (removed_y != 0)
                                if np.any(removed_mask):
                                    ax.scatter(
                                        removed_x[removed_mask],
                                        removed_y[removed_mask],
                                        s=removed_marker_size,
                                        marker=removed_marker,
                                        alpha=removed_marker_alpha,
                                        linewidths=1.0,
                                        c=removed_colors[removed_mask].tolist(),
                                        zorder=3,
                                    )
                            ax.set_xlim(col_ranges[c])
                            ax.set_ylim(col_ranges[r])
                            ax.set_xlabel("Q" if c == 0 else "T")
                            ax.set_ylabel("Q" if r == 0 else "T")

                a_plane, a_strip, a_side = channels[ai]
                b_plane, b_strip, b_side = channels[bj]
                fig.suptitle(
                    f"P{a_plane}S{a_strip}{a_side} vs P{b_plane}S{b_strip}{b_side} · P{pi}xP{pj} [{total_pair_events} events]",
                    fontsize=9,
                )
                fig.tight_layout()
                fig_path = OUT_DIR / f"{args.basename}_P{pi}P{pj}_ch{ai+1}_ch{bj+1}.png"
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                fig_count += 1
                if fig_count >= args.max_figures:
                    break
            if fig_count >= args.max_figures:
                break
        if fig_count >= args.max_figures:
            break

    print(f"Smoke test PNGs written to: {OUT_DIR}")
    print(f"Retained rows after filters: {len(working_df)}")
    if args.track_removed_rows:
        print(f"Removed rows saved to: {removed_rows_path}")
        print(f"Removed rows CSV saved to: {removed_rows_csv_path}")
        print(f"Original columns saved to: {original_cols_path}")
        print(f"Tracked removed rows: {len(removed_rows_df)}")
    else:
        print("Removed-row tracking disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
