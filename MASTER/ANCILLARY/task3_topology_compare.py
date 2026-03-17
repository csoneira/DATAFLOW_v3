#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/ANCILLARY/task3_topology_compare.py
Purpose: Compare TASK_3 exact strip topologies between simulated and real data.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-16
Runtime: python3
Usage: python3 MASTER/ANCILLARY/task3_topology_compare.py [--stage auto] [--min-jump 2] [--top-n 25]
Inputs: task_3_metadata_specific.csv from MINGO00 and MINGO01 STAGE_1/STEP_1/TASK_3
Outputs: CSV tables and a markdown report in MASTER/ANCILLARY/OUTPUTS/noise_study/FILES
Notes: Deduplicates by filename_base and keeps the latest execution_timestamp per file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from task3_two_plane_single_strip_scan import OUT_DIR, df_to_markdown, load_task3_metadata

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SINGLE_PATTERNS = {"1000", "0100", "0010", "0001"}
ADJ_DOUBLE_PATTERNS = {"1100", "0110", "0011"}
NONADJ_DOUBLE_PATTERNS = {"1010", "1001", "0101"}
PLOT_DIR = OUT_DIR.parent / "PLOTS"
MATCH_COLORS = {
    "matched": "#1f77b4",
    "real_only": "#d62728",
    "sim_only": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TASK_3 exact topology rates between simulated and real data."
    )
    parser.add_argument("--real-station", default="MINGO01")
    parser.add_argument("--sim-station", default="MINGO00")
    parser.add_argument(
        "--stage",
        choices=["auto", "cal", "list"],
        default="auto",
        help="Pattern stage to analyze. 'auto' prefers list, then cal.",
    )
    parser.add_argument(
        "--min-jump",
        type=int,
        default=2,
        help="Minimum strip jump used for rough single-strip-path subsets.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top exact offenders to include in the markdown report.",
    )
    return parser.parse_args()


def pattern_columns(df: pd.DataFrame, stage: str) -> list[str]:
    prefix = f"{stage}_strip_pattern_"
    suffix = "_rate_hz"
    return sorted(
        [col for col in df.columns if col.startswith(prefix) and col.endswith(suffix)]
    )


def full16_from_column(col: str, stage: str) -> str:
    prefix = f"{stage}_strip_pattern_"
    suffix = "_rate_hz"
    return col[len(prefix) : -len(suffix)]


def plane_type(pattern: str) -> str:
    ones = pattern.count("1")
    if ones == 0:
        return "empty"
    if pattern in SINGLE_PATTERNS:
        return "single"
    if pattern in ADJ_DOUBLE_PATTERNS:
        return "adj_double"
    if pattern in NONADJ_DOUBLE_PATTERNS:
        return "nonadj_double"
    if ones == 3:
        return "triple"
    if ones == 4:
        return "quad"
    return "other"


def strip_index(pattern: str) -> int | None:
    if pattern.count("1") != 1:
        return None
    return pattern.index("1") + 1


def compute_turn_count(strips: list[int]) -> int:
    if len(strips) < 3:
        return 0
    deltas = [strips[idx + 1] - strips[idx] for idx in range(len(strips) - 1)]
    signs = [int(np.sign(delta)) for delta in deltas if delta != 0]
    return sum(1 for idx in range(len(signs) - 1) if signs[idx] != signs[idx + 1])


def ratio_or_inf(real_mean: float, sim_mean: float) -> float:
    if sim_mean > 0:
        return real_mean / sim_mean
    if real_mean > 0:
        return np.inf
    return np.nan


def tt_label(active_mask: str) -> str:
    label = "".join(str(idx + 1) for idx, bit in enumerate(active_mask) if bit == "1")
    return label if label else "0"


def classify_pattern(full16: str) -> dict[str, object]:
    planes = [full16[idx : idx + 4] for idx in range(0, 16, 4)]
    plane_types = [plane_type(pattern) for pattern in planes]
    multiplicities = [pattern.count("1") for pattern in planes]
    active_plane_ids = [idx + 1 for idx, mult in enumerate(multiplicities) if mult > 0]
    active_mask = "".join("1" if mult > 0 else "0" for mult in multiplicities)
    active_planes = len(active_plane_ids)
    occupied_mults = [mult for mult in multiplicities if mult > 0]
    occupied_mult_vector = "-".join(str(mult) for mult in occupied_mults) if occupied_mults else "none"
    multiplicity_vector = "-".join(str(mult) for mult in multiplicities)

    n_single = sum(kind == "single" for kind in plane_types)
    n_adj_double = sum(kind == "adj_double" for kind in plane_types)
    n_nonadj_double = sum(kind == "nonadj_double" for kind in plane_types)
    n_triple = sum(kind == "triple" for kind in plane_types)
    n_quad = sum(kind == "quad" for kind in plane_types)

    single_strip_only = active_planes > 0 and all(mult == 1 for mult in occupied_mults)
    active_strips: list[int] = []
    jumps: list[int] = []
    max_jump = 0
    total_jump = 0
    turn_count = 0
    if single_strip_only:
        active_strips = [strip_index(planes[idx - 1]) for idx in active_plane_ids]  # type: ignore[list-item]
        jumps = [abs(active_strips[idx + 1] - active_strips[idx]) for idx in range(len(active_strips) - 1)]
        max_jump = max(jumps) if jumps else 0
        total_jump = sum(jumps)
        turn_count = compute_turn_count(active_strips)

    return {
        "full16": full16,
        "active_mask": active_mask,
        "active_planes": active_planes,
        "plane_type_signature": "|".join(plane_types),
        "multiplicity_vector": multiplicity_vector,
        "occupied_mult_vector": occupied_mult_vector,
        "total_hits": sum(multiplicities),
        "n_single_planes": n_single,
        "n_adj_double_planes": n_adj_double,
        "n_nonadj_double_planes": n_nonadj_double,
        "n_triple_planes": n_triple,
        "n_quad_planes": n_quad,
        "any_irregular_plane": n_nonadj_double + n_triple + n_quad > 0,
        "single_strip_only": single_strip_only,
        "single_strip_path": "-".join(str(strip) for strip in active_strips) if active_strips else "",
        "max_jump": max_jump,
        "total_jump": total_jump,
        "turn_count": turn_count,
    }


def compute_exact_patterns(real_df: pd.DataFrame, sim_df: pd.DataFrame, stage: str) -> pd.DataFrame:
    real_cols = set(pattern_columns(real_df, stage))
    sim_cols = set(pattern_columns(sim_df, stage))
    all_cols = sorted(real_cols | sim_cols)

    rows: list[dict[str, object]] = []
    for col in all_cols:
        full16 = full16_from_column(col, stage)
        meta = classify_pattern(full16)
        real_series = real_df[col] if col in real_df.columns else pd.Series(0.0, index=real_df.index)
        sim_series = sim_df[col] if col in sim_df.columns else pd.Series(0.0, index=sim_df.index)
        real_mean = float(real_series.mean())
        sim_mean = float(sim_series.mean())
        rows.append(
            {
                **meta,
                "real_mean_hz": real_mean,
                "sim_mean_hz": sim_mean,
                "delta_hz": real_mean - sim_mean,
                "ratio_real_over_sim": ratio_or_inf(real_mean, sim_mean),
                "real_support": int((real_series > 0).sum()),
                "sim_support": int((sim_series > 0).sum()),
                "pattern_in_real": col in real_cols,
                "pattern_in_sim": col in sim_cols,
            }
        )

    patterns = pd.DataFrame(rows)
    real_total = float(patterns["real_mean_hz"].sum())
    sim_total = float(patterns["sim_mean_hz"].sum())
    patterns["real_share"] = patterns["real_mean_hz"] / real_total if real_total > 0 else np.nan
    patterns["sim_share"] = patterns["sim_mean_hz"] / sim_total if sim_total > 0 else np.nan
    patterns["share_delta"] = patterns["real_share"] - patterns["sim_share"]
    return patterns


def build_subset_table(patterns: pd.DataFrame, min_jump: int) -> pd.DataFrame:
    subsets: list[tuple[str, pd.DataFrame]] = [
        ("all_observed", patterns),
        ("active_planes_1", patterns[patterns["active_planes"] == 1]),
        ("active_planes_2", patterns[patterns["active_planes"] == 2]),
        ("active_planes_3", patterns[patterns["active_planes"] == 3]),
        ("active_planes_4", patterns[patterns["active_planes"] == 4]),
        ("single_strip_only_2planes", patterns[(patterns["single_strip_only"]) & (patterns["active_planes"] == 2)]),
        ("single_strip_only_3planes", patterns[(patterns["single_strip_only"]) & (patterns["active_planes"] == 3)]),
        ("single_strip_only_4planes", patterns[(patterns["single_strip_only"]) & (patterns["active_planes"] == 4)]),
        (
            f"single_strip_only_2planes_jump_ge_{min_jump}",
            patterns[
                (patterns["single_strip_only"])
                & (patterns["active_planes"] == 2)
                & (patterns["max_jump"] >= min_jump)
            ],
        ),
        (
            f"single_strip_only_3planes_jump_ge_{min_jump}",
            patterns[
                (patterns["single_strip_only"])
                & (patterns["active_planes"] == 3)
                & (patterns["max_jump"] >= min_jump)
            ],
        ),
        (
            f"single_strip_only_4planes_jump_ge_{min_jump}",
            patterns[
                (patterns["single_strip_only"])
                & (patterns["active_planes"] == 4)
                & (patterns["max_jump"] >= min_jump)
            ],
        ),
        (
            "single_strip_only_3planes_zigzag",
            patterns[
                (patterns["single_strip_only"])
                & (patterns["active_planes"] == 3)
                & (patterns["turn_count"] >= 1)
            ],
        ),
        (
            "single_strip_only_4planes_zigzag",
            patterns[
                (patterns["single_strip_only"])
                & (patterns["active_planes"] == 4)
                & (patterns["turn_count"] >= 1)
            ],
        ),
        ("any_irregular_plane", patterns[patterns["any_irregular_plane"]]),
        ("any_nonadj_double_plane", patterns[patterns["n_nonadj_double_planes"] > 0]),
        ("any_trip_or_quad_plane", patterns[(patterns["n_triple_planes"] > 0) | (patterns["n_quad_planes"] > 0)]),
    ]

    real_total = float(patterns["real_mean_hz"].sum())
    sim_total = float(patterns["sim_mean_hz"].sum())
    rows: list[dict[str, object]] = []
    for subset_name, subset_df in subsets:
        real_sum = float(subset_df["real_mean_hz"].sum())
        sim_sum = float(subset_df["sim_mean_hz"].sum())
        rows.append(
            {
                "subset": subset_name,
                "n_patterns": int(len(subset_df)),
                "real_mean_hz": real_sum,
                "sim_mean_hz": sim_sum,
                "delta_hz": real_sum - sim_sum,
                "ratio_real_over_sim": ratio_or_inf(real_sum, sim_sum),
                "real_share": (real_sum / real_total) if real_total > 0 else np.nan,
                "sim_share": (sim_sum / sim_total) if sim_total > 0 else np.nan,
                "share_delta": (
                    (real_sum / real_total) - (sim_sum / sim_total)
                    if real_total > 0 and sim_total > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_group(patterns: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        patterns.groupby(group_col, dropna=False)
        .agg(
            n_patterns=("full16", "size"),
            real_mean_hz=("real_mean_hz", "sum"),
            sim_mean_hz=("sim_mean_hz", "sum"),
            real_support=("real_support", "sum"),
            sim_support=("sim_support", "sum"),
        )
        .reset_index()
    )
    grouped["delta_hz"] = grouped["real_mean_hz"] - grouped["sim_mean_hz"]
    grouped["ratio_real_over_sim"] = grouped.apply(
        lambda row: ratio_or_inf(float(row["real_mean_hz"]), float(row["sim_mean_hz"])), axis=1
    )
    real_total = float(patterns["real_mean_hz"].sum())
    sim_total = float(patterns["sim_mean_hz"].sum())
    grouped["real_share"] = grouped["real_mean_hz"] / real_total if real_total > 0 else np.nan
    grouped["sim_share"] = grouped["sim_mean_hz"] / sim_total if sim_total > 0 else np.nan
    grouped["share_delta"] = grouped["real_share"] - grouped["sim_share"]
    return grouped.sort_values(["real_mean_hz", "delta_hz"], ascending=False).reset_index(drop=True)


def compute_overlap_metrics(patterns: pd.DataFrame) -> pd.DataFrame:
    real_total = float(patterns["real_mean_hz"].sum())
    sim_total = float(patterns["sim_mean_hz"].sum())
    subsets = [
        ("common_support", patterns[(patterns["pattern_in_real"]) & (patterns["pattern_in_sim"])]),
        ("real_only_support", patterns[(patterns["pattern_in_real"]) & (~patterns["pattern_in_sim"])]),
        ("sim_only_support", patterns[(~patterns["pattern_in_real"]) & (patterns["pattern_in_sim"])]),
        ("positive_in_both", patterns[(patterns["real_mean_hz"] > 0) & (patterns["sim_mean_hz"] > 0)]),
        ("positive_real_only", patterns[(patterns["real_mean_hz"] > 0) & (patterns["sim_mean_hz"] == 0)]),
        ("positive_sim_only", patterns[(patterns["real_mean_hz"] == 0) & (patterns["sim_mean_hz"] > 0)]),
    ]
    rows = []
    for name, df in subsets:
        real_sum = float(df["real_mean_hz"].sum())
        sim_sum = float(df["sim_mean_hz"].sum())
        rows.append(
            {
                "subset": name,
                "n_patterns": int(len(df)),
                "real_mean_hz": real_sum,
                "sim_mean_hz": sim_sum,
                "real_share": (real_sum / real_total) if real_total > 0 else np.nan,
                "sim_share": (sim_sum / sim_total) if sim_total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_tt_closure_points(patterns: pd.DataFrame) -> pd.DataFrame:
    points = patterns[(patterns["real_mean_hz"] > 0) | (patterns["sim_mean_hz"] > 0)].copy()
    points["tt_label"] = points["active_mask"].map(tt_label)
    points["combined_hz"] = points["real_mean_hz"] + points["sim_mean_hz"]
    points["pct_delta_within_tt"] = np.nan
    points["match_status"] = "matched"
    points.loc[(points["real_mean_hz"] > 0) & (points["sim_mean_hz"] == 0), "match_status"] = "real_only"
    points.loc[(points["real_mean_hz"] == 0) & (points["sim_mean_hz"] > 0), "match_status"] = "sim_only"

    tt_real = points.groupby("active_mask")["real_mean_hz"].transform("sum")
    tt_sim = points.groupby("active_mask")["sim_mean_hz"].transform("sum")
    points["real_pct_within_tt"] = np.where(tt_real > 0, 100.0 * points["real_mean_hz"] / tt_real, 0.0)
    points["sim_pct_within_tt"] = np.where(tt_sim > 0, 100.0 * points["sim_mean_hz"] / tt_sim, 0.0)
    points["pct_delta_within_tt"] = points["real_pct_within_tt"] - points["sim_pct_within_tt"]
    return points


def compute_tt_closure_summary(patterns: pd.DataFrame, tt_points: pd.DataFrame) -> pd.DataFrame:
    real_total = float(patterns["real_mean_hz"].sum())
    sim_total = float(patterns["sim_mean_hz"].sum())
    rows: list[dict[str, object]] = []

    for active_mask, tt_df in tt_points.groupby("active_mask", sort=True):
        tt_name = tt_label(active_mask)
        real_tt_total = float(tt_df["real_mean_hz"].sum())
        sim_tt_total = float(tt_df["sim_mean_hz"].sum())

        matched = tt_df[tt_df["match_status"] == "matched"]
        real_only = tt_df[tt_df["match_status"] == "real_only"]
        sim_only = tt_df[tt_df["match_status"] == "sim_only"]

        matched_real_hz = float(matched["real_mean_hz"].sum())
        matched_sim_hz = float(matched["sim_mean_hz"].sum())
        real_only_hz = float(real_only["real_mean_hz"].sum())
        sim_only_hz = float(sim_only["sim_mean_hz"].sum())

        rows.append(
            {
                "active_mask": active_mask,
                "tt_label": tt_name,
                "n_patterns": int(len(tt_df)),
                "matched_n_patterns": int(len(matched)),
                "real_only_n_patterns": int(len(real_only)),
                "sim_only_n_patterns": int(len(sim_only)),
                "real_tt_total_hz": real_tt_total,
                "sim_tt_total_hz": sim_tt_total,
                "matched_real_hz": matched_real_hz,
                "matched_sim_hz": matched_sim_hz,
                "real_only_hz": real_only_hz,
                "sim_only_hz": sim_only_hz,
                "matched_real_pct_of_tt": (100.0 * matched_real_hz / real_tt_total) if real_tt_total > 0 else 0.0,
                "matched_sim_pct_of_tt": (100.0 * matched_sim_hz / sim_tt_total) if sim_tt_total > 0 else 0.0,
                "real_only_pct_of_tt": (100.0 * real_only_hz / real_tt_total) if real_tt_total > 0 else 0.0,
                "sim_only_pct_of_tt": (100.0 * sim_only_hz / sim_tt_total) if sim_tt_total > 0 else 0.0,
                "matched_real_pct_of_total": (100.0 * matched_real_hz / real_total) if real_total > 0 else 0.0,
                "matched_sim_pct_of_total": (100.0 * matched_sim_hz / sim_total) if sim_total > 0 else 0.0,
                "real_only_pct_of_total": (100.0 * real_only_hz / real_total) if real_total > 0 else 0.0,
                "sim_only_pct_of_total": (100.0 * sim_only_hz / sim_total) if sim_total > 0 else 0.0,
                "plot_file": f"task3_tt_closure_{tt_name}.png",
            }
        )

    return pd.DataFrame(rows).sort_values(["real_tt_total_hz", "tt_label"], ascending=[False, True]).reset_index(
        drop=True
    )


def point_sizes(values: pd.Series) -> np.ndarray:
    vmax = float(values.max()) if len(values) else 0.0
    if vmax <= 0:
        return np.full(len(values), 36.0)
    return 30.0 + 260.0 * np.sqrt(values.to_numpy() / vmax)


def select_labels(tt_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if tt_df.empty or top_n <= 0:
        return tt_df.iloc[0:0]

    prioritized = tt_df.copy()
    prioritized["is_unmatched"] = prioritized["match_status"] != "matched"
    prioritized["label_score"] = prioritized["combined_hz"] * (
        1.0 + np.abs(prioritized["pct_delta_within_tt"]) / 100.0
    )
    return prioritized.sort_values(
        ["is_unmatched", "label_score", "combined_hz"], ascending=[False, False, False]
    ).head(top_n)


def plot_tt_closure(tt_points: pd.DataFrame, tt_summary: pd.DataFrame, top_n_labels: int) -> list[Path]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for row in tt_summary.itertuples(index=False):
        tt_df = tt_points[tt_points["active_mask"] == row.active_mask].copy()
        if tt_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        axis_max = max(
            float(tt_df["real_pct_within_tt"].max()),
            float(tt_df["sim_pct_within_tt"].max()),
            1.0,
        )
        axis_lim = axis_max * 1.08

        for status in ["matched", "real_only", "sim_only"]:
            subset = tt_df[tt_df["match_status"] == status]
            if subset.empty:
                continue
            ax.scatter(
                subset["sim_pct_within_tt"],
                subset["real_pct_within_tt"],
                s=point_sizes(subset["combined_hz"]),
                c=MATCH_COLORS[status],
                alpha=0.72,
                linewidths=0.4,
                edgecolors="black",
                label=status.replace("_", " "),
            )

        for label_row in select_labels(tt_df, top_n_labels).itertuples(index=False):
            ax.annotate(
                label_row.full16,
                (label_row.sim_pct_within_tt, label_row.real_pct_within_tt),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.9,
            )

        ax.plot([0.0, axis_lim], [0.0, axis_lim], linestyle="--", linewidth=1.0, color="#666666")
        ax.set_xlim(-0.5, axis_lim)
        ax.set_ylim(-0.5, axis_lim)
        ax.set_xlabel("Simulation share within TT (%)")
        ax.set_ylabel("Real-data share within TT (%)")
        ax.set_title(f"TASK_3 topology closure for TT {row.tt_label}")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(loc="upper right", fontsize=8, frameon=True)

        text = (
            f"Matched: {row.matched_real_pct_of_tt:.1f}% real TT, {row.matched_sim_pct_of_tt:.1f}% sim TT\n"
            f"Real-only: {row.real_only_pct_of_tt:.1f}% real TT\n"
            f"Sim-only: {row.sim_only_pct_of_tt:.1f}% sim TT\n"
            f"TT totals: real {row.real_tt_total_hz:.3f} Hz, sim {row.sim_tt_total_hz:.3f} Hz"
        )
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
        )

        out_path = PLOT_DIR / row.plot_file
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        created.append(out_path)

    return created


def build_report(
    real_station: str,
    sim_station: str,
    stage: str,
    real_rows: int,
    sim_rows: int,
    min_jump: int,
    top_n: int,
    patterns: pd.DataFrame,
    subsets: pd.DataFrame,
    overlap: pd.DataFrame,
    by_active_planes: pd.DataFrame,
    by_active_mask: pd.DataFrame,
    by_occ_mult: pd.DataFrame,
    by_plane_signature: pd.DataFrame,
    tt_summary: pd.DataFrame,
) -> str:
    top_exact_abs = patterns.sort_values(["real_mean_hz", "delta_hz"], ascending=False).head(top_n)
    top_exact_delta = patterns.sort_values(["delta_hz", "real_mean_hz"], ascending=False).head(top_n)
    top_exact_ratio = patterns[patterns["sim_mean_hz"] > 0].sort_values(
        ["ratio_real_over_sim", "real_mean_hz"], ascending=[False, False]
    ).head(top_n)
    top_3plane_rough = patterns[
        (patterns["single_strip_only"]) & (patterns["active_planes"] == 3) & (patterns["max_jump"] >= min_jump)
    ].sort_values(["real_mean_hz", "delta_hz"], ascending=False).head(top_n)
    top_4plane_rough = patterns[
        (patterns["single_strip_only"]) & (patterns["active_planes"] == 4) & (patterns["max_jump"] >= min_jump)
    ].sort_values(["real_mean_hz", "delta_hz"], ascending=False).head(top_n)
    top_3plane_zigzag = patterns[
        (patterns["single_strip_only"]) & (patterns["active_planes"] == 3) & (patterns["turn_count"] >= 1)
    ].sort_values(["real_mean_hz", "delta_hz"], ascending=False).head(top_n)
    top_4plane_zigzag = patterns[
        (patterns["single_strip_only"]) & (patterns["active_planes"] == 4) & (patterns["turn_count"] >= 1)
    ].sort_values(["real_mean_hz", "delta_hz"], ascending=False).head(top_n)
    top_irregular = patterns[patterns["any_irregular_plane"]].sort_values(
        ["real_mean_hz", "delta_hz"], ascending=False
    ).head(top_n)

    float_cols = [
        "real_mean_hz",
        "sim_mean_hz",
        "delta_hz",
        "ratio_real_over_sim",
        "real_share",
        "sim_share",
        "share_delta",
    ]

    lines = [
        "# TASK_3 Topology Comparison",
        "",
        f"- real station: `{real_station}`",
        f"- sim station: `{sim_station}`",
        f"- stage: `{stage}`",
        f"- real latest rows: `{real_rows}`",
        f"- sim latest rows: `{sim_rows}`",
        f"- rough single-strip threshold: `max_jump >= {min_jump}`",
        "",
        "## Exact Pattern Support Overlap",
        "",
        df_to_markdown(overlap, ["real_mean_hz", "sim_mean_hz", "real_share", "sim_share"]),
        "",
        "## Topology Subsets",
        "",
        df_to_markdown(subsets.sort_values("real_mean_hz", ascending=False), float_cols),
        "",
        "## By Number of Active Planes",
        "",
        df_to_markdown(by_active_planes, float_cols),
        "",
        "## By Active Plane Mask",
        "",
        df_to_markdown(by_active_mask.head(top_n), float_cols),
        "",
        "## TT Closure Summary",
        "",
        df_to_markdown(
            tt_summary[
                [
                    "tt_label",
                    "active_mask",
                    "n_patterns",
                    "matched_n_patterns",
                    "real_only_n_patterns",
                    "sim_only_n_patterns",
                    "real_tt_total_hz",
                    "sim_tt_total_hz",
                    "matched_real_pct_of_tt",
                    "matched_sim_pct_of_tt",
                    "real_only_pct_of_tt",
                    "sim_only_pct_of_tt",
                    "matched_real_pct_of_total",
                    "matched_sim_pct_of_total",
                    "plot_file",
                ]
            ],
            [
                "real_tt_total_hz",
                "sim_tt_total_hz",
                "matched_real_pct_of_tt",
                "matched_sim_pct_of_tt",
                "real_only_pct_of_tt",
                "sim_only_pct_of_tt",
                "matched_real_pct_of_total",
                "matched_sim_pct_of_total",
            ],
        ),
        "",
        "## By Occupied Multiplicity Vector",
        "",
        df_to_markdown(by_occ_mult.head(top_n), float_cols),
        "",
        "## By Plane-Type Signature",
        "",
        df_to_markdown(by_plane_signature.head(top_n), float_cols),
        "",
        f"## Top {top_n} Exact Patterns by Absolute Real Rate",
        "",
        df_to_markdown(
            top_exact_abs[
                [
                    "full16",
                    "active_mask",
                    "occupied_mult_vector",
                    "plane_type_signature",
                    "single_strip_only",
                    "single_strip_path",
                    "max_jump",
                    "turn_count",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "share_delta",
                ]
            ],
            ["real_mean_hz", "sim_mean_hz", "delta_hz", "ratio_real_over_sim", "share_delta"],
        ),
        "",
        f"## Top {top_n} Exact Patterns by Real Minus Sim Delta",
        "",
        df_to_markdown(
            top_exact_delta[
                [
                    "full16",
                    "active_mask",
                    "occupied_mult_vector",
                    "plane_type_signature",
                    "single_strip_only",
                    "single_strip_path",
                    "max_jump",
                    "turn_count",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "share_delta",
                ]
            ],
            ["real_mean_hz", "sim_mean_hz", "delta_hz", "ratio_real_over_sim", "share_delta"],
        ),
        "",
        f"## Top {top_n} Exact Patterns by Real/Sim Ratio With Positive Sim Support",
        "",
        df_to_markdown(
            top_exact_ratio[
                [
                    "full16",
                    "active_mask",
                    "occupied_mult_vector",
                    "plane_type_signature",
                    "single_strip_only",
                    "single_strip_path",
                    "max_jump",
                    "turn_count",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "share_delta",
                ]
            ],
            ["real_mean_hz", "sim_mean_hz", "delta_hz", "ratio_real_over_sim", "share_delta"],
        ),
        "",
        f"## Top {top_n} Rough Three-Plane Single-Strip Patterns",
        "",
        df_to_markdown(
            top_3plane_rough[
                [
                    "full16",
                    "active_mask",
                    "single_strip_path",
                    "max_jump",
                    "total_jump",
                    "turn_count",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "share_delta",
                ]
            ],
            ["real_mean_hz", "sim_mean_hz", "delta_hz", "ratio_real_over_sim", "share_delta"],
        ),
        "",
        f"## Top {top_n} Rough Four-Plane Single-Strip Patterns",
        "",
        df_to_markdown(
            top_4plane_rough[
                [
                    "full16",
                    "active_mask",
                    "single_strip_path",
                    "max_jump",
                    "total_jump",
                    "turn_count",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "share_delta",
                ]
            ],
            ["real_mean_hz", "sim_mean_hz", "delta_hz", "ratio_real_over_sim", "share_delta"],
        ),
        "",
        f"## Top {top_n} Zigzag Three-Plane Single-Strip Patterns",
        "",
        df_to_markdown(
            top_3plane_zigzag[
                [
                    "full16",
                    "active_mask",
                    "single_strip_path",
                    "max_jump",
                    "total_jump",
                    "turn_count",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "share_delta",
                ]
            ],
            ["real_mean_hz", "sim_mean_hz", "delta_hz", "ratio_real_over_sim", "share_delta"],
        ),
        "",
        f"## Top {top_n} Zigzag Four-Plane Single-Strip Patterns",
        "",
        df_to_markdown(
            top_4plane_zigzag[
                [
                    "full16",
                    "active_mask",
                    "single_strip_path",
                    "max_jump",
                    "total_jump",
                    "turn_count",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "share_delta",
                ]
            ],
            ["real_mean_hz", "sim_mean_hz", "delta_hz", "ratio_real_over_sim", "share_delta"],
        ),
        "",
        f"## Top {top_n} Patterns With Irregular Plane Content",
        "",
        df_to_markdown(
            top_irregular[
                [
                    "full16",
                    "active_mask",
                    "occupied_mult_vector",
                    "plane_type_signature",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "share_delta",
                ]
            ],
            ["real_mean_hz", "sim_mean_hz", "delta_hz", "ratio_real_over_sim", "share_delta"],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    real_df, stage = load_task3_metadata(args.real_station, args.stage)
    sim_df, _ = load_task3_metadata(args.sim_station, args.stage)

    patterns = compute_exact_patterns(real_df, sim_df, stage)
    subsets = build_subset_table(patterns, args.min_jump)
    overlap = compute_overlap_metrics(patterns)
    by_active_planes = summarize_by_group(patterns, "active_planes")
    by_active_mask = summarize_by_group(patterns, "active_mask")
    by_occ_mult = summarize_by_group(patterns, "occupied_mult_vector")
    by_plane_signature = summarize_by_group(patterns, "plane_type_signature")
    tt_points = compute_tt_closure_points(patterns)
    tt_summary = compute_tt_closure_summary(patterns, tt_points)
    plot_paths = plot_tt_closure(tt_points, tt_summary, top_n_labels=min(args.top_n, 10))

    exact_path = OUT_DIR / "task3_topology_exact_patterns.csv"
    subsets_path = OUT_DIR / "task3_topology_subsets.csv"
    overlap_path = OUT_DIR / "task3_topology_overlap.csv"
    active_planes_path = OUT_DIR / "task3_topology_by_active_planes.csv"
    active_mask_path = OUT_DIR / "task3_topology_by_active_mask.csv"
    occ_mult_path = OUT_DIR / "task3_topology_by_occupied_multiplicity.csv"
    signature_path = OUT_DIR / "task3_topology_by_plane_signature.csv"
    tt_points_path = OUT_DIR / "task3_topology_tt_points.csv"
    tt_summary_path = OUT_DIR / "task3_topology_tt_closure.csv"
    report_path = OUT_DIR / "task3_topology_report.md"

    patterns.to_csv(exact_path, index=False)
    subsets.to_csv(subsets_path, index=False)
    overlap.to_csv(overlap_path, index=False)
    by_active_planes.to_csv(active_planes_path, index=False)
    by_active_mask.to_csv(active_mask_path, index=False)
    by_occ_mult.to_csv(occ_mult_path, index=False)
    by_plane_signature.to_csv(signature_path, index=False)
    tt_points.to_csv(tt_points_path, index=False)
    tt_summary.to_csv(tt_summary_path, index=False)
    report_path.write_text(
        build_report(
            args.real_station,
            args.sim_station,
            stage,
            len(real_df),
            len(sim_df),
            args.min_jump,
            args.top_n,
            patterns,
            subsets,
            overlap,
            by_active_planes,
            by_active_mask,
            by_occ_mult,
            by_plane_signature,
            tt_summary,
        ),
        encoding="utf-8",
    )

    print(f"stage={stage}")
    print(f"real_rows={len(real_df)} sim_rows={len(sim_df)}")
    print(f"exact_csv={exact_path}")
    print(f"subsets_csv={subsets_path}")
    print(f"overlap_csv={overlap_path}")
    print(f"tt_points_csv={tt_points_path}")
    print(f"tt_summary_csv={tt_summary_path}")
    print(f"plots={len(plot_paths)} in {PLOT_DIR}")
    print(f"report_md={report_path}")


if __name__ == "__main__":
    main()
