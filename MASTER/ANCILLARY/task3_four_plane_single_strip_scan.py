#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/ANCILLARY/task3_four_plane_single_strip_scan.py
Purpose: Compare exact four-plane single-strip TASK_3 patterns between simulated and real data.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-16
Runtime: python3
Usage: python3 MASTER/ANCILLARY/task3_four_plane_single_strip_scan.py [--stage auto] [--min-jump 2] [--top-n 20]
Inputs: task_3_metadata_specific.csv from MINGO00 and MINGO01 STAGE_1/STEP_1/TASK_3
Outputs: CSV tables and a markdown report in MASTER/ANCILLARY/OUTPUTS/noise_study/FILES
Notes: Deduplicates by filename_base and keeps the latest execution_timestamp per file.
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from task3_two_plane_single_strip_scan import OUT_DIR, df_to_markdown, load_task3_metadata


ONEHOTS = ["1000", "0100", "0010", "0001"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan TASK_3 four-plane single-strip patterns and rank real-vs-sim offenders."
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
        help="Minimum adjacent-plane strip jump used for the rough-pattern ranking.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top exact offenders to include in the markdown report.",
    )
    return parser.parse_args()


def build_full_pattern(strips: tuple[int, int, int, int]) -> str:
    return "".join(ONEHOTS[strip - 1] for strip in strips)


def compute_turn_count(strips: tuple[int, int, int, int]) -> int:
    deltas = [strips[1] - strips[0], strips[2] - strips[1], strips[3] - strips[2]]
    signs = [int(np.sign(delta)) for delta in deltas if delta != 0]
    return sum(1 for idx in range(len(signs) - 1) if signs[idx] != signs[idx + 1])


def ratio_or_inf(real_mean: float, sim_mean: float) -> float:
    if sim_mean > 0:
        return real_mean / sim_mean
    if real_mean > 0:
        return np.inf
    return np.nan


def compute_exact_patterns(
    real_df: pd.DataFrame, sim_df: pd.DataFrame, stage: str
) -> pd.DataFrame:
    prefix = f"{stage}_strip_pattern_"
    rows: list[dict[str, object]] = []
    for strips in product(range(1, 5), repeat=4):
        full16 = build_full_pattern(strips)
        col = f"{prefix}{full16}_rate_hz"
        real_series = real_df[col] if col in real_df.columns else pd.Series(0.0, index=real_df.index)
        sim_series = sim_df[col] if col in sim_df.columns else pd.Series(0.0, index=sim_df.index)
        real_mean = float(real_series.mean())
        sim_mean = float(sim_series.mean())
        jumps = [abs(strips[1] - strips[0]), abs(strips[2] - strips[1]), abs(strips[3] - strips[2])]
        rows.append(
            {
                "s1": strips[0],
                "s2": strips[1],
                "s3": strips[2],
                "s4": strips[3],
                "full16": full16,
                "d12": jumps[0],
                "d23": jumps[1],
                "d34": jumps[2],
                "max_jump": max(jumps),
                "total_jump": sum(jumps),
                "turn_count": compute_turn_count(strips),
                "real_mean_hz": real_mean,
                "sim_mean_hz": sim_mean,
                "delta_hz": real_mean - sim_mean,
                "ratio_real_over_sim": ratio_or_inf(real_mean, sim_mean),
                "real_support": int((real_series > 0).sum()),
                "sim_support": int((sim_series > 0).sum()),
            }
        )

    patterns = pd.DataFrame(rows)
    real_total = float(patterns["real_mean_hz"].sum())
    sim_total = float(patterns["sim_mean_hz"].sum())
    patterns["real_share"] = patterns["real_mean_hz"] / real_total if real_total > 0 else np.nan
    patterns["sim_share"] = patterns["sim_mean_hz"] / sim_total if sim_total > 0 else np.nan
    patterns["share_delta"] = patterns["real_share"] - patterns["sim_share"]
    return patterns


def compute_subset_summaries(patterns: pd.DataFrame, min_jump: int) -> pd.DataFrame:
    subsets = [
        ("all", patterns),
        ("straight_all_same_strip", patterns[patterns["total_jump"] == 0]),
        ("smooth_max_jump_le_1", patterns[patterns["max_jump"] <= 1]),
        (f"rough_any_jump_ge_{min_jump}", patterns[patterns["max_jump"] >= min_jump]),
        ("extreme_any_jump_eq_3", patterns[patterns["max_jump"] == 3]),
        ("zigzag_turn_ge_1", patterns[patterns["turn_count"] >= 1]),
        ("zigzag_turn_ge_2", patterns[patterns["turn_count"] >= 2]),
        ("high_total_jump_ge_4", patterns[patterns["total_jump"] >= 4]),
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
                "n_patterns": int(len(subset_df)),
            }
        )
    return pd.DataFrame(rows)


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
) -> str:
    top_abs = patterns.sort_values(["real_mean_hz", "delta_hz"], ascending=False).head(top_n)
    rough_patterns = patterns[patterns["max_jump"] >= min_jump]
    top_rough_abs = rough_patterns.sort_values(["real_mean_hz", "delta_hz"], ascending=False).head(top_n)
    top_rough_ratio = rough_patterns.sort_values(
        ["ratio_real_over_sim", "real_mean_hz"], ascending=[False, False]
    ).head(top_n)
    top_rough_share = rough_patterns.sort_values(
        ["share_delta", "real_mean_hz"], ascending=[False, False]
    ).head(top_n)
    top_zigzag = patterns[patterns["turn_count"] >= 1].sort_values(
        ["real_mean_hz", "delta_hz"], ascending=False
    ).head(top_n)

    lines = [
        "# TASK_3 Four-Plane Single-Strip Scan",
        "",
        f"- real station: `{real_station}`",
        f"- sim station: `{sim_station}`",
        f"- stage: `{stage}`",
        f"- real latest rows: `{real_rows}`",
        f"- sim latest rows: `{sim_rows}`",
        f"- rough-pattern threshold: `max_jump >= {min_jump}`",
        "",
        "## Family Summary",
        "",
        df_to_markdown(
            subsets.sort_values("real_mean_hz", ascending=False),
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_share",
                "sim_share",
                "share_delta",
            ],
        ),
        "",
        f"## Top {top_n} Exact Four-Plane Patterns by Absolute Real Rate",
        "",
        df_to_markdown(
            top_abs[
                [
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "d12",
                    "d23",
                    "d34",
                    "max_jump",
                    "total_jump",
                    "turn_count",
                    "full16",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "real_share",
                    "sim_share",
                    "share_delta",
                ]
            ],
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_share",
                "sim_share",
                "share_delta",
            ],
        ),
        "",
        f"## Top {top_n} Rough Exact Patterns by Absolute Real Rate",
        "",
        df_to_markdown(
            top_rough_abs[
                [
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "d12",
                    "d23",
                    "d34",
                    "max_jump",
                    "total_jump",
                    "turn_count",
                    "full16",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "real_share",
                    "sim_share",
                    "share_delta",
                ]
            ],
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_share",
                "sim_share",
                "share_delta",
            ],
        ),
        "",
        f"## Top {top_n} Rough Exact Patterns by Real/Sim Ratio",
        "",
        df_to_markdown(
            top_rough_ratio[
                [
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "d12",
                    "d23",
                    "d34",
                    "max_jump",
                    "total_jump",
                    "turn_count",
                    "full16",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "real_share",
                    "sim_share",
                    "share_delta",
                ]
            ],
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_share",
                "sim_share",
                "share_delta",
            ],
        ),
        "",
        f"## Top {top_n} Rough Exact Patterns by Share Excess",
        "",
        df_to_markdown(
            top_rough_share[
                [
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "d12",
                    "d23",
                    "d34",
                    "max_jump",
                    "total_jump",
                    "turn_count",
                    "full16",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "real_share",
                    "sim_share",
                    "share_delta",
                ]
            ],
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_share",
                "sim_share",
                "share_delta",
            ],
        ),
        "",
        f"## Top {top_n} Zigzag Patterns by Absolute Real Rate",
        "",
        df_to_markdown(
            top_zigzag[
                [
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "d12",
                    "d23",
                    "d34",
                    "max_jump",
                    "total_jump",
                    "turn_count",
                    "full16",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "real_share",
                    "sim_share",
                    "share_delta",
                ]
            ],
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_share",
                "sim_share",
                "share_delta",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    real_df, stage = load_task3_metadata(args.real_station, args.stage)
    sim_df, _ = load_task3_metadata(args.sim_station, args.stage)

    patterns = compute_exact_patterns(real_df, sim_df, stage)
    subsets = compute_subset_summaries(patterns, args.min_jump)

    patterns_path = OUT_DIR / "task3_four_plane_single_strip_patterns.csv"
    subsets_path = OUT_DIR / "task3_four_plane_single_strip_subsets.csv"
    report_path = OUT_DIR / "task3_four_plane_single_strip_report.md"

    patterns.to_csv(patterns_path, index=False)
    subsets.to_csv(subsets_path, index=False)
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
        ),
        encoding="utf-8",
    )

    print(f"stage={stage}")
    print(f"real_rows={len(real_df)} sim_rows={len(sim_df)}")
    print(f"patterns_csv={patterns_path}")
    print(f"subsets_csv={subsets_path}")
    print(f"report_md={report_path}")


if __name__ == "__main__":
    main()
