#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/ANCILLARY/task3_two_plane_single_strip_scan.py
Purpose: Compare exact two-plane single-strip TASK_3 patterns between simulated and real data.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-16
Runtime: python3
Usage: python3 MASTER/ANCILLARY/task3_two_plane_single_strip_scan.py [--stage auto] [--min-separation 2] [--top-n 20]
Inputs: task_3_metadata_pattern.csv (preferred) or legacy task_3_metadata_specific.csv
from MINGO00 and MINGO01 STAGE_1/STEP_1/TASK_3
Outputs: CSV tables and a markdown report in MASTER/ANCILLARY/OUTPUTS/noise_study/FILES
Notes: Deduplicates by filename_base and keeps the latest execution_timestamp per file.
"""

from __future__ import annotations

import argparse
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = SCRIPT_DIR / "OUTPUTS" / "noise_study" / "FILES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan TASK_3 two-plane single-strip patterns and rank real-vs-sim offenders."
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
        "--min-separation",
        type=int,
        default=2,
        help="Minimum strip separation used for the 'wide-separation' offender ranking.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top exact offenders to include in the markdown report.",
    )
    return parser.parse_args()


def task3_pattern_path(station: str) -> Path:
    return (
        REPO_ROOT
        / "STATIONS"
        / station
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_3"
        / "METADATA"
        / "task_3_metadata_pattern.csv"
    )


def task3_metadata_path(station: str) -> Path:
    pattern_path = task3_pattern_path(station)
    if pattern_path.exists():
        return pattern_path
    return (
        REPO_ROOT
        / "STATIONS"
        / station
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_3"
        / "METADATA"
        / "task_3_metadata_specific.csv"
    )


def parse_exec_ts(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    parsed = pd.to_datetime(text, format="%Y-%m-%d_%H.%M.%S", errors="coerce", utc=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce", utc=True)
    return parsed


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "execution_timestamp" in df.columns:
        df = df.copy()
        df["_exec_ts"] = parse_exec_ts(df["execution_timestamp"])
        df = df.sort_values(["filename_base", "_exec_ts"]).drop_duplicates(
            subset="filename_base", keep="last"
        )
        df = df.drop(columns="_exec_ts")
    return df.reset_index(drop=True)


def load_task3_metadata(station: str, stage_choice: str) -> tuple[pd.DataFrame, str]:
    path = task3_metadata_path(station)
    if not path.exists():
        raise FileNotFoundError(f"Missing TASK_3 metadata: {path}")

    df = latest_rows(pd.read_csv(path, low_memory=False))
    list_cols = [c for c in df.columns if c.startswith("list_strip_pattern_") and c.endswith("_rate_hz")]
    cal_cols = [c for c in df.columns if c.startswith("cal_strip_pattern_") and c.endswith("_rate_hz")]

    if stage_choice == "list":
        if not list_cols:
            raise ValueError(f"No list_strip_pattern columns found in {path}")
        cols = list_cols
        stage = "list"
    elif stage_choice == "cal":
        if not cal_cols:
            raise ValueError(f"No cal_strip_pattern columns found in {path}")
        cols = cal_cols
        stage = "cal"
    else:
        if list_cols:
            cols = list_cols
            stage = "list"
        elif cal_cols:
            cols = cal_cols
            stage = "cal"
        else:
            raise ValueError(f"No strip-pattern columns found in {path}")

    use = ["filename_base"] + cols
    df = df[use].copy()
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df, stage


def build_full_pattern(plane_a: int, strip_a: int, plane_b: int, strip_b: int) -> tuple[str, str]:
    onehots = ["1000", "0100", "0010", "0001"]
    planes = ["0000", "0000", "0000", "0000"]
    planes[plane_a - 1] = onehots[strip_a - 1]
    planes[plane_b - 1] = onehots[strip_b - 1]
    return onehots[strip_a - 1] + onehots[strip_b - 1], "".join(planes)


def compute_exact_patterns(
    real_df: pd.DataFrame, sim_df: pd.DataFrame, stage: str
) -> pd.DataFrame:
    prefix = f"{stage}_strip_pattern_"
    rows: list[dict[str, object]] = []
    for plane_a, plane_b in combinations([1, 2, 3, 4], 2):
        for strip_a, strip_b in product(range(1, 5), range(1, 5)):
            inner8, full16 = build_full_pattern(plane_a, strip_a, plane_b, strip_b)
            col = f"{prefix}{full16}_rate_hz"
            real_series = real_df[col] if col in real_df.columns else pd.Series(0.0, index=real_df.index)
            sim_series = sim_df[col] if col in sim_df.columns else pd.Series(0.0, index=sim_df.index)
            real_mean = float(real_series.mean())
            sim_mean = float(sim_series.mean())
            rows.append(
                {
                    "pair": f"P{plane_a}P{plane_b}",
                    "plane_a": plane_a,
                    "plane_b": plane_b,
                    "strip_a": strip_a,
                    "strip_b": strip_b,
                    "sep": abs(strip_a - strip_b),
                    "inner8": inner8,
                    "full16": full16,
                    "real_mean_hz": real_mean,
                    "sim_mean_hz": sim_mean,
                    "delta_hz": real_mean - sim_mean,
                    "ratio_real_over_sim": (real_mean / sim_mean) if sim_mean > 0 else np.nan,
                    "real_support": int((real_series > 0).sum()),
                    "sim_support": int((sim_series > 0).sum()),
                }
            )

    patterns = pd.DataFrame(rows)
    patterns["real_family_total"] = patterns.groupby("pair")["real_mean_hz"].transform("sum")
    patterns["sim_family_total"] = patterns.groupby("pair")["sim_mean_hz"].transform("sum")
    patterns["real_family_share"] = patterns["real_mean_hz"] / patterns["real_family_total"]
    patterns["sim_family_share"] = patterns["sim_mean_hz"] / patterns["sim_family_total"]
    patterns["share_delta"] = patterns["real_family_share"] - patterns["sim_family_share"]
    return patterns


def compute_family_summaries(patterns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair, pair_df in patterns.groupby("pair", sort=True):
        real_total = float(pair_df["real_mean_hz"].sum())
        sim_total = float(pair_df["sim_mean_hz"].sum())
        for subset_name, subset_df in [
            ("all", pair_df),
            ("sep_ge_1", pair_df[pair_df["sep"] >= 1]),
            ("sep_ge_2", pair_df[pair_df["sep"] >= 2]),
            ("sep_eq_3", pair_df[pair_df["sep"] == 3]),
        ]:
            real_sum = float(subset_df["real_mean_hz"].sum())
            sim_sum = float(subset_df["sim_mean_hz"].sum())
            rows.append(
                {
                    "pair": pair,
                    "subset": subset_name,
                    "real_mean_hz": real_sum,
                    "sim_mean_hz": sim_sum,
                    "delta_hz": real_sum - sim_sum,
                    "ratio_real_over_sim": (real_sum / sim_sum) if sim_sum > 0 else np.nan,
                    "real_share_within_pair": (real_sum / real_total) if real_total > 0 else np.nan,
                    "sim_share_within_pair": (sim_sum / sim_total) if sim_total > 0 else np.nan,
                    "share_delta_within_pair": (
                        (real_sum / real_total) - (sim_sum / sim_total)
                        if real_total > 0 and sim_total > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    float_cols = float_cols or []
    df = df.copy()
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")

    headers = [str(col) for col in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(values: list[str]) -> str:
        padded = [values[i].ljust(widths[i]) for i in range(len(values))]
        return "| " + " | ".join(padded) + " |"

    header_line = fmt_row(headers)
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body_lines = [fmt_row(row) for row in rows]
    return "\n".join([header_line, sep_line] + body_lines)


def build_report(
    real_station: str,
    sim_station: str,
    stage: str,
    real_rows: int,
    sim_rows: int,
    min_separation: int,
    top_n: int,
    patterns: pd.DataFrame,
    families: pd.DataFrame,
) -> str:
    all_pairs = families[families["subset"] == "all"].sort_values(
        "ratio_real_over_sim", ascending=False
    )
    wide_pairs = families[families["subset"] == "sep_ge_2"].sort_values(
        ["real_mean_hz", "ratio_real_over_sim"], ascending=False
    )
    extreme_pairs = families[families["subset"] == "sep_eq_3"].sort_values(
        ["real_mean_hz", "ratio_real_over_sim"], ascending=False
    )

    wide_patterns = patterns[patterns["sep"] >= min_separation]
    top_abs = wide_patterns.sort_values(["real_mean_hz", "ratio_real_over_sim"], ascending=False).head(top_n)
    top_ratio = wide_patterns.sort_values("ratio_real_over_sim", ascending=False).head(top_n)
    top_share = wide_patterns.sort_values("share_delta", ascending=False).head(top_n)

    lines = [
        "# TASK_3 Two-Plane Single-Strip Scan",
        "",
        f"- real station: `{real_station}`",
        f"- sim station: `{sim_station}`",
        f"- stage: `{stage}`",
        f"- real latest rows: `{real_rows}`",
        f"- sim latest rows: `{sim_rows}`",
        f"- wide-separation threshold: `sep >= {min_separation}`",
        "",
        "## Pair Ranking: All Two-Plane Single-Strip Families",
        "",
        df_to_markdown(
            all_pairs,
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_share_within_pair",
                "sim_share_within_pair",
                "share_delta_within_pair",
            ],
        ),
        "",
        "## Pair Ranking: Wide-Separation Subset",
        "",
        df_to_markdown(
            wide_pairs,
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_share_within_pair",
                "sim_share_within_pair",
                "share_delta_within_pair",
            ],
        ),
        "",
        "## Pair Ranking: Extreme Separation Only",
        "",
        df_to_markdown(
            extreme_pairs,
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_share_within_pair",
                "sim_share_within_pair",
                "share_delta_within_pair",
            ],
        ),
        "",
        f"## Top {top_n} Wide-Separation Exact Offenders by Absolute Real Rate",
        "",
        df_to_markdown(
            top_abs[
                [
                    "pair",
                    "strip_a",
                    "strip_b",
                    "sep",
                    "inner8",
                    "full16",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "real_family_share",
                    "sim_family_share",
                    "share_delta",
                ]
            ],
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_family_share",
                "sim_family_share",
                "share_delta",
            ],
        ),
        "",
        f"## Top {top_n} Wide-Separation Exact Offenders by Real/Sim Ratio",
        "",
        df_to_markdown(
            top_ratio[
                [
                    "pair",
                    "strip_a",
                    "strip_b",
                    "sep",
                    "inner8",
                    "full16",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "real_family_share",
                    "sim_family_share",
                    "share_delta",
                ]
            ],
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_family_share",
                "sim_family_share",
                "share_delta",
            ],
        ),
        "",
        f"## Top {top_n} Wide-Separation Exact Offenders by Share Excess Within Pair",
        "",
        df_to_markdown(
            top_share[
                [
                    "pair",
                    "strip_a",
                    "strip_b",
                    "sep",
                    "inner8",
                    "full16",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "ratio_real_over_sim",
                    "real_family_share",
                    "sim_family_share",
                    "share_delta",
                ]
            ],
            [
                "real_mean_hz",
                "sim_mean_hz",
                "delta_hz",
                "ratio_real_over_sim",
                "real_family_share",
                "sim_family_share",
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
    families = compute_family_summaries(patterns)

    patterns_path = OUT_DIR / "task3_two_plane_single_strip_patterns.csv"
    families_path = OUT_DIR / "task3_two_plane_single_strip_families.csv"
    report_path = OUT_DIR / "task3_two_plane_single_strip_report.md"

    patterns.to_csv(patterns_path, index=False)
    families.to_csv(families_path, index=False)
    report_path.write_text(
        build_report(
            args.real_station,
            args.sim_station,
            stage,
            len(real_df),
            len(sim_df),
            args.min_separation,
            args.top_n,
            patterns,
            families,
        ),
        encoding="utf-8",
    )

    print(f"stage={stage}")
    print(f"real_rows={len(real_df)} sim_rows={len(sim_df)}")
    print(f"patterns_csv={patterns_path}")
    print(f"families_csv={families_path}")
    print(f"report_md={report_path}")


if __name__ == "__main__":
    main()
