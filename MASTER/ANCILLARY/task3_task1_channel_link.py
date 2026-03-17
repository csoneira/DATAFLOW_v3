#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/ANCILLARY/task3_task1_channel_link.py
Purpose: Cross-check suspicious TASK_3 single-strip topologies against compatible TASK_1 channel-pattern families.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-16
Runtime: python3
Usage: python3 MASTER/ANCILLARY/task3_task1_channel_link.py [--stage list] [--min-jump 2] [--top-n 20]
Inputs: TASK_1/TASK_3 metadata CSVs plus task3_topology_exact_patterns.csv
Outputs: CSV tables and a markdown report in MASTER/ANCILLARY/OUTPUTS/noise_study/FILES
Notes: Uses latest-per-filename_base rows and compares both Hz and percentage-of-total metrics.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from task3_two_plane_single_strip_scan import OUT_DIR, latest_rows


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_STATES = ("11", "10", "01")
ZERO_32 = "00" * 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link suspicious TASK_3 single-strip patterns to compatible TASK_1 channel-pattern families."
    )
    parser.add_argument("--real-station", default="MINGO01")
    parser.add_argument("--sim-station", default="MINGO00")
    parser.add_argument("--stage", choices=["cal", "list"], default="list")
    parser.add_argument("--min-jump", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def task1_specific_path(station: str) -> Path:
    return (
        REPO_ROOT
        / "STATIONS"
        / station
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_1"
        / "METADATA"
        / "task_1_metadata_specific.csv"
    )


def task1_trigger_type_path(station: str) -> Path:
    return (
        REPO_ROOT
        / "STATIONS"
        / station
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_1"
        / "METADATA"
        / "task_1_metadata_trigger_type.csv"
    )


def task3_specific_path(station: str) -> Path:
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


def read_latest_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    return latest_rows(df)


def read_latest_csv_fast(path: Path, wanted_cols: list[str]) -> pd.DataFrame:
    wanted = list(dict.fromkeys(wanted_cols))
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        idx_map = {col: idx for idx, col in enumerate(header)}
        present = [col for col in wanted if col in idx_map]
        for raw_row in reader:
            rows.append(
                {
                    col: raw_row[idx_map[col]] if idx_map[col] < len(raw_row) else ""
                    for col in present
                }
            )
    return latest_rows(pd.DataFrame(rows, columns=[col for col in wanted if col in idx_map]))


def csv_columns(path: Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns.tolist())


def task3_exact_path() -> Path:
    return OUT_DIR / "task3_topology_exact_patterns.csv"


def path_category(row: pd.Series, min_jump: int) -> str:
    if int(row["turn_count"]) >= 1:
        return "zigzag"
    if int(row["max_jump"]) >= min_jump:
        return "rough"
    return "smooth"


def select_suspicious_patterns(exact_df: pd.DataFrame, min_jump: int, top_n: int) -> pd.DataFrame:
    mask = exact_df["single_strip_only"] & (
        (exact_df["turn_count"] >= 1) | (exact_df["max_jump"] >= min_jump)
    )
    selected = exact_df.loc[mask].copy()
    selected["pattern_class"] = selected.apply(path_category, axis=1, min_jump=min_jump)
    selected = selected.sort_values(["delta_hz", "real_mean_hz"], ascending=False).head(top_n)
    return selected.reset_index(drop=True)


def active_slots_from_full16(full16: str) -> list[tuple[int, int]]:
    slots: list[tuple[int, int]] = []
    for plane_idx in range(4):
        pattern = full16[plane_idx * 4 : (plane_idx + 1) * 4]
        if pattern == "0000":
            continue
        if pattern.count("1") != 1:
            raise ValueError(f"Not a single-strip-only pattern: {full16}")
        strip_idx = pattern.index("1") + 1
        slots.append((plane_idx + 1, strip_idx))
    return slots


def pair_offset(plane: int, strip: int) -> int:
    return ((plane - 1) * 4 + (strip - 1)) * 2


def channel_family_members(full16: str) -> list[dict[str, object]]:
    slots = active_slots_from_full16(full16)
    members: list[dict[str, object]] = []
    for states in product(ACTIVE_STATES, repeat=len(slots)):
        pairs = ["00"] * 16
        asym_motifs: list[str] = []
        asym_count = 0
        for (plane, strip), state in zip(slots, states):
            idx = (plane - 1) * 4 + (strip - 1)
            pairs[idx] = state
            if state == "10":
                asym_motifs.append(f"P{plane}S{strip}_F_only")
                asym_count += 1
            elif state == "01":
                asym_motifs.append(f"P{plane}S{strip}_B_only")
                asym_count += 1
        code32 = "".join(pairs)
        members.append(
            {
                "code32": code32,
                "is_paired": asym_count == 0,
                "any_asym": asym_count > 0,
                "asym_count": asym_count,
                "asym_motifs": tuple(asym_motifs),
            }
        )
    return members


def selected_family_definitions(selected: pd.DataFrame) -> dict[str, dict[str, object]]:
    defs: dict[str, dict[str, object]] = {}
    for row in selected.itertuples(index=False):
        full16 = str(row.full16)
        defs[full16] = {
            "slots": active_slots_from_full16(full16),
            "members": channel_family_members(full16),
            "active_mask": row.active_mask,
            "single_strip_path": row.single_strip_path,
            "active_planes": int(row.active_planes),
            "max_jump": int(row.max_jump),
            "turn_count": int(row.turn_count),
            "pattern_class": row.pattern_class,
            "task3_real_hz": float(row.real_mean_hz),
            "task3_sim_hz": float(row.sim_mean_hz),
            "task3_real_pct": 100.0 * float(row.real_share),
            "task3_sim_pct": 100.0 * float(row.sim_share),
        }
    return defs


def task1_total_rate_df(station: str) -> pd.DataFrame:
    path = task1_trigger_type_path(station)
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    raw_cols = [c for c in cols if c.startswith("raw_tt_") and c.endswith("_rate_hz")]
    clean_cols = [c for c in cols if c.startswith("clean_tt_") and c.endswith("_rate_hz")]
    df = read_latest_csv(path, usecols=["filename_base", "execution_timestamp"] + raw_cols + clean_cols)
    df["raw_total_rate_hz"] = df[raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    df["clean_total_rate_hz"] = df[clean_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    return df[["filename_base", "raw_total_rate_hz", "clean_total_rate_hz"]]


def load_task1_specific(station: str, family_defs: dict[str, dict[str, object]]) -> pd.DataFrame:
    path = task1_specific_path(station)
    header = csv_columns(path)
    raw_cols: set[str] = set()
    clean_cols: set[str] = set()
    for info in family_defs.values():
        for member in info["members"]:
            code32 = str(member["code32"])
            raw_col = f"raw_channel_pattern_{code32}_rate_hz"
            clean_col = f"clean_channel_pattern_{code32}_rate_hz"
            if raw_col in header:
                raw_cols.add(raw_col)
            if clean_col in header:
                clean_cols.add(clean_col)
    usecols = ["filename_base", "execution_timestamp"] + sorted(raw_cols) + sorted(clean_cols)
    df = read_latest_csv_fast(path, usecols)
    value_cols = [c for c in df.columns if c.endswith("_rate_hz")]
    if value_cols:
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def load_task3_selected(station: str, selected_full16: list[str], stage: str) -> pd.DataFrame:
    path = task3_specific_path(station)
    header = csv_columns(path)
    wanted = [f"{stage}_strip_pattern_{full16}_rate_hz" for full16 in selected_full16]
    cols = [col for col in wanted if col in header]
    df = read_latest_csv(path, usecols=["filename_base", "execution_timestamp"] + cols)
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def series_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1)


def safe_mean_share(numer: pd.Series, denom: pd.Series) -> float:
    denom_safe = denom.replace(0, np.nan)
    return float((numer / denom_safe).mean())


def safe_within_family_share(numer: pd.Series, denom: pd.Series) -> float:
    denom_safe = denom.replace(0, np.nan)
    return float((numer / denom_safe).mean())


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return np.nan
    if frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(frame["x"].corr(frame["y"]))


def station_tag(station: str, real_station: str, sim_station: str) -> str:
    if station == real_station:
        return "real"
    if station == sim_station:
        return "sim"
    return station.lower()


def analyze_station(
    station: str,
    station_kind: str,
    task1_df: pd.DataFrame,
    task1_totals: pd.DataFrame,
    task3_df: pd.DataFrame,
    family_defs: dict[str, dict[str, object]],
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = task1_df.merge(task1_totals, on="filename_base", how="inner").merge(
        task3_df, on="filename_base", how="inner"
    )

    summary_rows: list[dict[str, object]] = []
    motif_rows: list[dict[str, object]] = []

    aggregate_raw_family = pd.Series(0.0, index=merged.index)
    aggregate_raw_asym = pd.Series(0.0, index=merged.index)
    aggregate_clean_family = pd.Series(0.0, index=merged.index)
    aggregate_clean_asym = pd.Series(0.0, index=merged.index)
    aggregate_t3 = pd.Series(0.0, index=merged.index)
    motif_accum: dict[tuple[str, str], pd.Series] = defaultdict(lambda: pd.Series(0.0, index=merged.index))

    for full16, info in family_defs.items():
        t3_col = f"{stage}_strip_pattern_{full16}_rate_hz"
        t3_series = merged[t3_col] if t3_col in merged.columns else pd.Series(0.0, index=merged.index)

        raw_all = [f"raw_channel_pattern_{member['code32']}_rate_hz" for member in info["members"]]
        raw_all = [col for col in raw_all if col in merged.columns]
        raw_paired = [
            f"raw_channel_pattern_{member['code32']}_rate_hz"
            for member in info["members"]
            if bool(member["is_paired"]) and f"raw_channel_pattern_{member['code32']}_rate_hz" in merged.columns
        ]
        raw_asym = [
            f"raw_channel_pattern_{member['code32']}_rate_hz"
            for member in info["members"]
            if bool(member["any_asym"]) and f"raw_channel_pattern_{member['code32']}_rate_hz" in merged.columns
        ]

        clean_all = [f"clean_channel_pattern_{member['code32']}_rate_hz" for member in info["members"]]
        clean_all = [col for col in clean_all if col in merged.columns]
        clean_paired = [
            f"clean_channel_pattern_{member['code32']}_rate_hz"
            for member in info["members"]
            if bool(member["is_paired"]) and f"clean_channel_pattern_{member['code32']}_rate_hz" in merged.columns
        ]
        clean_asym = [
            f"clean_channel_pattern_{member['code32']}_rate_hz"
            for member in info["members"]
            if bool(member["any_asym"]) and f"clean_channel_pattern_{member['code32']}_rate_hz" in merged.columns
        ]

        raw_family_series = series_sum(merged, raw_all)
        raw_paired_series = series_sum(merged, raw_paired)
        raw_asym_series = series_sum(merged, raw_asym)

        clean_family_series = series_sum(merged, clean_all)
        clean_paired_series = series_sum(merged, clean_paired)
        clean_asym_series = series_sum(merged, clean_asym)

        aggregate_raw_family = aggregate_raw_family.add(raw_family_series, fill_value=0.0)
        aggregate_raw_asym = aggregate_raw_asym.add(raw_asym_series, fill_value=0.0)
        aggregate_clean_family = aggregate_clean_family.add(clean_family_series, fill_value=0.0)
        aggregate_clean_asym = aggregate_clean_asym.add(clean_asym_series, fill_value=0.0)
        aggregate_t3 = aggregate_t3.add(t3_series, fill_value=0.0)

        for member in info["members"]:
            for stage_prefix in ("raw", "clean"):
                col = f"{stage_prefix}_channel_pattern_{member['code32']}_rate_hz"
                if col not in merged.columns:
                    continue
                for motif in member["asym_motifs"]:
                    motif_accum[(stage_prefix, motif)] = motif_accum[(stage_prefix, motif)].add(
                        merged[col], fill_value=0.0
                    )

        raw_total = merged["raw_total_rate_hz"]
        clean_total = merged["clean_total_rate_hz"]

        summary_rows.append(
            {
                "station_kind": station_kind,
                "station": station,
                "n_files": int(len(merged)),
                "full16": full16,
                "active_mask": info["active_mask"],
                "single_strip_path": info["single_strip_path"],
                "active_planes": info["active_planes"],
                "pattern_class": info["pattern_class"],
                "max_jump": info["max_jump"],
                "turn_count": info["turn_count"],
                "task3_mean_hz": float(t3_series.mean()),
                "task1_raw_family_mean_hz": float(raw_family_series.mean()),
                "task1_raw_family_mean_pct_total": 100.0 * safe_mean_share(raw_family_series, raw_total),
                "task1_raw_paired_mean_hz": float(raw_paired_series.mean()),
                "task1_raw_asym_mean_hz": float(raw_asym_series.mean()),
                "task1_raw_asym_mean_pct_total": 100.0 * safe_mean_share(raw_asym_series, raw_total),
                "task1_raw_asym_pct_within_family": 100.0
                * safe_within_family_share(raw_asym_series, raw_family_series),
                "task1_clean_family_mean_hz": float(clean_family_series.mean()),
                "task1_clean_family_mean_pct_total": 100.0 * safe_mean_share(clean_family_series, clean_total),
                "task1_clean_paired_mean_hz": float(clean_paired_series.mean()),
                "task1_clean_asym_mean_hz": float(clean_asym_series.mean()),
                "task1_clean_asym_mean_pct_total": 100.0 * safe_mean_share(clean_asym_series, clean_total),
                "task1_clean_asym_pct_within_family": 100.0
                * safe_within_family_share(clean_asym_series, clean_family_series),
                "corr_task3_vs_raw_asym_hz": safe_corr(t3_series, raw_asym_series),
                "corr_task3_vs_clean_asym_hz": safe_corr(t3_series, clean_asym_series),
                "corr_task3_vs_raw_asym_pct": safe_corr(
                    t3_series, raw_asym_series / raw_total.replace(0, np.nan)
                ),
                "corr_task3_vs_clean_asym_pct": safe_corr(
                    t3_series, clean_asym_series / clean_total.replace(0, np.nan)
                ),
            }
        )

    for (stage_prefix, motif), numer in motif_accum.items():
        denom = merged["raw_total_rate_hz"] if stage_prefix == "raw" else merged["clean_total_rate_hz"]
        motif_rows.append(
            {
                "station_kind": station_kind,
                "station": station,
                "stage": stage_prefix,
                "motif": motif,
                "mean_hz": float(numer.mean()),
                "mean_pct_total": 100.0 * safe_mean_share(numer, denom),
            }
        )

    aggregate_row = {
        "station_kind": station_kind,
        "station": station,
        "n_files": int(len(merged)),
        "full16": "__ALL_SELECTED__",
        "active_mask": "mixed",
        "single_strip_path": "mixed",
        "active_planes": np.nan,
        "pattern_class": "mixed",
        "max_jump": np.nan,
        "turn_count": np.nan,
        "task3_mean_hz": float(aggregate_t3.mean()),
        "task1_raw_family_mean_hz": float(aggregate_raw_family.mean()),
        "task1_raw_family_mean_pct_total": 100.0
        * safe_mean_share(aggregate_raw_family, merged["raw_total_rate_hz"]),
        "task1_raw_paired_mean_hz": float((aggregate_raw_family - aggregate_raw_asym).mean()),
        "task1_raw_asym_mean_hz": float(aggregate_raw_asym.mean()),
        "task1_raw_asym_mean_pct_total": 100.0
        * safe_mean_share(aggregate_raw_asym, merged["raw_total_rate_hz"]),
        "task1_raw_asym_pct_within_family": 100.0
        * safe_within_family_share(aggregate_raw_asym, aggregate_raw_family),
        "task1_clean_family_mean_hz": float(aggregate_clean_family.mean()),
        "task1_clean_family_mean_pct_total": 100.0
        * safe_mean_share(aggregate_clean_family, merged["clean_total_rate_hz"]),
        "task1_clean_paired_mean_hz": float((aggregate_clean_family - aggregate_clean_asym).mean()),
        "task1_clean_asym_mean_hz": float(aggregate_clean_asym.mean()),
        "task1_clean_asym_mean_pct_total": 100.0
        * safe_mean_share(aggregate_clean_asym, merged["clean_total_rate_hz"]),
        "task1_clean_asym_pct_within_family": 100.0
        * safe_within_family_share(aggregate_clean_asym, aggregate_clean_family),
        "corr_task3_vs_raw_asym_hz": safe_corr(aggregate_t3, aggregate_raw_asym),
        "corr_task3_vs_clean_asym_hz": safe_corr(aggregate_t3, aggregate_clean_asym),
        "corr_task3_vs_raw_asym_pct": safe_corr(
            aggregate_t3, aggregate_raw_asym / merged["raw_total_rate_hz"].replace(0, np.nan)
        ),
        "corr_task3_vs_clean_asym_pct": safe_corr(
            aggregate_t3, aggregate_clean_asym / merged["clean_total_rate_hz"].replace(0, np.nan)
        ),
    }

    summary_rows.append(aggregate_row)

    return pd.DataFrame(summary_rows), pd.DataFrame(motif_rows)


def pivot_summary(summary_df: pd.DataFrame, family_defs: dict[str, dict[str, object]]) -> pd.DataFrame:
    wide_rows: list[dict[str, object]] = []
    station_parts = {
        station_kind: df.set_index("full16").to_dict(orient="index")
        for station_kind, df in summary_df.groupby("station_kind")
    }
    for full16, info in family_defs.items():
        row = {
            "full16": full16,
            "active_mask": info["active_mask"],
            "single_strip_path": info["single_strip_path"],
            "active_planes": info["active_planes"],
            "pattern_class": info["pattern_class"],
            "max_jump": info["max_jump"],
            "turn_count": info["turn_count"],
            "task3_real_hz": info["task3_real_hz"],
            "task3_sim_hz": info["task3_sim_hz"],
            "task3_real_pct_total": info["task3_real_pct"],
            "task3_sim_pct_total": info["task3_sim_pct"],
        }
        for station_kind in ("real", "sim"):
            data = station_parts.get(station_kind, {}).get(full16, {})
            for key in [
                "task1_raw_family_mean_hz",
                "task1_raw_family_mean_pct_total",
                "task1_raw_asym_mean_hz",
                "task1_raw_asym_mean_pct_total",
                "task1_raw_asym_pct_within_family",
                "task1_clean_family_mean_hz",
                "task1_clean_family_mean_pct_total",
                "task1_clean_asym_mean_hz",
                "task1_clean_asym_mean_pct_total",
                "task1_clean_asym_pct_within_family",
                "corr_task3_vs_raw_asym_hz",
                "corr_task3_vs_clean_asym_hz",
                "corr_task3_vs_raw_asym_pct",
                "corr_task3_vs_clean_asym_pct",
            ]:
                row[f"{key}_{station_kind}"] = data.get(key, np.nan)
        wide_rows.append(row)
    return pd.DataFrame(wide_rows)


def pivot_aggregate(summary_df: pd.DataFrame) -> pd.DataFrame:
    agg = summary_df[summary_df["full16"] == "__ALL_SELECTED__"].copy()
    return agg


def pivot_motifs(motif_df: pd.DataFrame) -> pd.DataFrame:
    wide = motif_df.pivot_table(
        index=["stage", "motif"],
        columns="station_kind",
        values=["mean_hz", "mean_pct_total"],
        aggfunc="first",
    )
    wide.columns = ["_".join(col).strip() for col in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    for col in ("mean_hz_real", "mean_hz_sim", "mean_pct_total_real", "mean_pct_total_sim"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide["delta_hz_real_minus_sim"] = wide["mean_hz_real"] - wide["mean_hz_sim"]
    wide["delta_pct_real_minus_sim"] = wide["mean_pct_total_real"] - wide["mean_pct_total_sim"]
    return wide.sort_values(
        ["stage", "delta_hz_real_minus_sim", "delta_pct_real_minus_sim"], ascending=[True, False, False]
    ).reset_index(drop=True)


def df_float_markdown(df: pd.DataFrame) -> str:
    float_cols = [col for col in df.columns if df[col].dtype.kind in {"f", "i"}]
    return df_to_markdown(df, float_cols)


def build_report(
    selected_df: pd.DataFrame,
    linked_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    motif_df: pd.DataFrame,
) -> str:
    linked_sorted = linked_df.sort_values(["task3_real_hz", "task3_real_pct_total"], ascending=False)
    top_clean_gap = linked_df.sort_values(
        ["task1_clean_asym_mean_pct_total_real", "task1_clean_asym_mean_hz_real"], ascending=False
    ).head(15)
    top_motifs_clean = motif_df[motif_df["stage"] == "clean"].head(20)
    top_motifs_raw = motif_df[motif_df["stage"] == "raw"].head(20)

    lines = [
        "# TASK_3 vs TASK_1 Channel-Link Study",
        "",
        f"- suspicious TASK_3 patterns studied: `{len(selected_df)}`",
        f"- selection rule: `single_strip_only` and (`turn_count >= 1` or `max_jump >= 2`)",
        "",
        "## Selected TASK_3 Patterns",
        "",
        df_to_markdown(
            selected_df[
                [
                    "full16",
                    "active_mask",
                    "single_strip_path",
                    "active_planes",
                    "pattern_class",
                    "max_jump",
                    "turn_count",
                    "real_mean_hz",
                    "sim_mean_hz",
                    "delta_hz",
                    "real_share",
                    "sim_share",
                ]
            ].rename(
                columns={
                    "real_mean_hz": "task3_real_hz",
                    "sim_mean_hz": "task3_sim_hz",
                    "real_share": "task3_real_share",
                    "sim_share": "task3_sim_share",
                }
            ),
            ["task3_real_hz", "task3_sim_hz", "delta_hz", "task3_real_share", "task3_sim_share"],
        ),
        "",
        "## Aggregate Compatible TASK_1 Family Metrics",
        "",
        df_float_markdown(
            aggregate_df[
                [
                    "station_kind",
                    "n_files",
                    "task3_mean_hz",
                    "task1_raw_family_mean_hz",
                    "task1_raw_family_mean_pct_total",
                    "task1_raw_asym_mean_hz",
                    "task1_raw_asym_mean_pct_total",
                    "task1_raw_asym_pct_within_family",
                    "task1_clean_family_mean_hz",
                    "task1_clean_family_mean_pct_total",
                    "task1_clean_asym_mean_hz",
                    "task1_clean_asym_mean_pct_total",
                    "task1_clean_asym_pct_within_family",
                    "corr_task3_vs_raw_asym_hz",
                    "corr_task3_vs_clean_asym_hz",
                    "corr_task3_vs_raw_asym_pct",
                    "corr_task3_vs_clean_asym_pct",
                ]
            ]
        ),
        "",
        "## Per-Pattern TASK_1 Family Comparison",
        "",
        df_float_markdown(
            linked_sorted[
                [
                    "full16",
                    "single_strip_path",
                    "pattern_class",
                    "task3_real_hz",
                    "task3_sim_hz",
                    "task3_real_pct_total",
                    "task3_sim_pct_total",
                    "task1_raw_family_mean_hz_real",
                    "task1_raw_family_mean_hz_sim",
                    "task1_raw_family_mean_pct_total_real",
                    "task1_raw_family_mean_pct_total_sim",
                    "task1_raw_asym_pct_within_family_real",
                    "task1_raw_asym_pct_within_family_sim",
                    "task1_clean_family_mean_hz_real",
                    "task1_clean_family_mean_hz_sim",
                    "task1_clean_family_mean_pct_total_real",
                    "task1_clean_family_mean_pct_total_sim",
                    "task1_clean_asym_pct_within_family_real",
                    "task1_clean_asym_pct_within_family_sim",
                    "corr_task3_vs_clean_asym_hz_real",
                    "corr_task3_vs_clean_asym_hz_sim",
                    "corr_task3_vs_clean_asym_pct_real",
                    "corr_task3_vs_clean_asym_pct_sim",
                ]
            ]
        ),
        "",
        "## Patterns With The Largest Clean Asymmetry Presence In TASK_1",
        "",
        df_float_markdown(
            top_clean_gap[
                [
                    "full16",
                    "single_strip_path",
                    "pattern_class",
                    "task3_real_hz",
                    "task3_real_pct_total",
                    "task1_clean_family_mean_hz_real",
                    "task1_clean_family_mean_pct_total_real",
                    "task1_clean_asym_mean_hz_real",
                    "task1_clean_asym_mean_pct_total_real",
                    "task1_clean_asym_pct_within_family_real",
                    "task1_clean_asym_pct_within_family_sim",
                ]
            ]
        ),
        "",
        "## Dominant RAW TASK_1 Asymmetry Motifs Across Selected TASK_3 Families",
        "",
        df_float_markdown(
            top_motifs_raw[
                [
                    "motif",
                    "mean_hz_real",
                    "mean_hz_sim",
                    "delta_hz_real_minus_sim",
                    "mean_pct_total_real",
                    "mean_pct_total_sim",
                    "delta_pct_real_minus_sim",
                ]
            ]
        ),
        "",
        "## Dominant CLEAN TASK_1 Asymmetry Motifs Across Selected TASK_3 Families",
        "",
        df_float_markdown(
            top_motifs_clean[
                [
                    "motif",
                    "mean_hz_real",
                    "mean_hz_sim",
                    "delta_hz_real_minus_sim",
                    "mean_pct_total_real",
                    "mean_pct_total_sim",
                    "delta_pct_real_minus_sim",
                ]
            ]
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exact = pd.read_csv(
        task3_exact_path(),
        dtype={"full16": "string", "active_mask": "string", "single_strip_path": "string"},
    )
    exact["full16"] = exact["full16"].fillna("").str.zfill(16)
    exact["active_mask"] = exact["active_mask"].fillna("").str.zfill(4)
    exact["single_strip_path"] = exact["single_strip_path"].fillna("")
    selected = select_suspicious_patterns(exact, args.min_jump, args.top_n)
    family_defs = selected_family_definitions(selected)
    selected_full16 = list(family_defs.keys())

    real_t1 = load_task1_specific(args.real_station, family_defs)
    sim_t1 = load_task1_specific(args.sim_station, family_defs)
    real_t1_totals = task1_total_rate_df(args.real_station)
    sim_t1_totals = task1_total_rate_df(args.sim_station)
    real_t3 = load_task3_selected(args.real_station, selected_full16, args.stage)
    sim_t3 = load_task3_selected(args.sim_station, selected_full16, args.stage)

    real_summary, real_motifs = analyze_station(
        args.real_station,
        "real",
        real_t1,
        real_t1_totals,
        real_t3,
        family_defs,
        args.stage,
    )
    sim_summary, sim_motifs = analyze_station(
        args.sim_station,
        "sim",
        sim_t1,
        sim_t1_totals,
        sim_t3,
        family_defs,
        args.stage,
    )

    summary_df = pd.concat([real_summary, sim_summary], ignore_index=True)
    motif_df = pd.concat([real_motifs, sim_motifs], ignore_index=True)
    linked_df = pivot_summary(summary_df, family_defs)
    aggregate_df = pivot_aggregate(summary_df)
    motif_wide_df = pivot_motifs(motif_df)

    selected_path = OUT_DIR / "task3_task1_link_selected_patterns.csv"
    summary_path = OUT_DIR / "task3_task1_link_summary.csv"
    aggregate_path = OUT_DIR / "task3_task1_link_aggregate.csv"
    motifs_path = OUT_DIR / "task3_task1_link_motifs.csv"
    report_path = OUT_DIR / "task3_task1_link_report.md"

    selected.to_csv(selected_path, index=False)
    linked_df.to_csv(summary_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    motif_wide_df.to_csv(motifs_path, index=False)
    report_path.write_text(
        build_report(selected, linked_df, aggregate_df, motif_wide_df),
        encoding="utf-8",
    )

    print(f"selected_csv={selected_path}")
    print(f"summary_csv={summary_path}")
    print(f"aggregate_csv={aggregate_path}")
    print(f"motifs_csv={motifs_path}")
    print(f"report_md={report_path}")


if __name__ == "__main__":
    main()
