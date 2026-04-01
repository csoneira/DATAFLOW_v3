#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from MASTER.ANCILLARY.SIMULATION_TUNING.common import (
    collect_calibrated_file_entries,
    date_range_mask,
    filter_frame_by_datetime,
    load_tuning_config,
    resolve_selection,
)
from MASTER.common.file_selection import extract_run_datetime_from_name


PLANE_STRIPS = {
    plane: [f"Q{plane}_Q_sum_{strip}" for strip in range(1, 5)]
    for plane in range(1, 5)
}
GROUP_COLORS = {
    "SIM": "#d95f02",
    "REAL": "#1b9e77",
}
MULTIPLICITY_LABELS = ("single", "double_adj", "triple_adj", "quad")
MULTIPLICITY_WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)


@dataclass(frozen=True)
class GroupPayload:
    files: list[tuple[str, Path]]
    topology_counts_by_plane: dict[int, dict[str, int]]


@dataclass(frozen=True)
class InductionMatchConfig:
    efficiency_match_max_abs_difference: float
    max_pairs_per_plane: int
    min_clusterlike_entries_per_plane: int


SINGLE_PATTERNS = {"1000", "0100", "0010", "0001"}
DOUBLE_ADJ_PATTERNS = {"1100", "0110", "0011"}
TRIPLE_ADJ_PATTERNS = {"1110", "0111"}
QUAD_PATTERNS = {"1111"}
CLUSTERLIKE_PATTERNS = (
    SINGLE_PATTERNS | DOUBLE_ADJ_PATTERNS | TRIPLE_ADJ_PATTERNS | QUAD_PATTERNS
)
ZERO_PATTERN = "0000"
PLANE_NO_TRIGGER = {
    1: "234",
    2: "134",
    3: "124",
    4: "123",
}


def output_dir() -> Path:
    return Path(__file__).resolve().parent / "OUTPUTS"


def load_simulation_induction_width_mm() -> float:
    cfg_path = (
        ROOT_DIR
        / "MINGO_DIGITAL_TWIN"
        / "MASTER_STEPS"
        / "STEP_4"
        / "config_step_4_physics.yaml"
    )
    import yaml

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return float(cfg.get("lorentzian_gamma_mm", 0.5 * float(cfg.get("avalanche_width_mm", np.nan))))


def load_match_config(config: dict) -> InductionMatchConfig:
    study_cfg = config.get("induction_section", {})
    return InductionMatchConfig(
        efficiency_match_max_abs_difference=float(
            study_cfg.get("efficiency_match_max_abs_difference", 0.05)
        ),
        max_pairs_per_plane=int(study_cfg.get("max_pairs_per_plane", 80)),
        min_clusterlike_entries_per_plane=int(
            study_cfg.get("min_clusterlike_entries_per_plane", 200)
        ),
    )


def _basename_from_calibrated_path(path: Path) -> str:
    name = path.name
    if name.startswith("calibrated_") and name.endswith(".parquet"):
        return name[len("calibrated_") : -len(".parquet")]
    return path.stem


def station_trigger_metadata_path(station_label: str) -> Path:
    return (
        ROOT_DIR
        / "STATIONS"
        / station_label
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_3"
        / "METADATA"
        / "task_3_metadata_trigger_type.csv"
    )


def _preferred_rate(frame: pd.DataFrame, tt_label: str) -> pd.Series:
    list_col = f"list_tt_{tt_label}_rate_hz"
    cal_col = f"cal_tt_{tt_label}_rate_hz"
    if list_col in frame.columns:
        list_values = pd.to_numeric(frame[list_col], errors="coerce")
    else:
        list_values = pd.Series(np.nan, index=frame.index, dtype=float)
    if cal_col in frame.columns:
        cal_values = pd.to_numeric(frame[cal_col], errors="coerce")
    else:
        cal_values = pd.Series(np.nan, index=frame.index, dtype=float)
    return list_values.where(list_values > 0, cal_values)


def load_latest_trigger_metadata(
    station_labels: list[str],
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> pd.DataFrame:
    path_lookup = {
        (station_label, _basename_from_calibrated_path(path)): path
        for station_label, path in collect_calibrated_file_entries(station_labels)
    }

    frames: list[pd.DataFrame] = []
    for station_label in station_labels:
        metadata_path = station_trigger_metadata_path(station_label)
        if not metadata_path.exists():
            continue
        frame = pd.read_csv(metadata_path)
        if frame.empty or "filename_base" not in frame.columns:
            continue
        frame = frame.copy()
        frame["station_label"] = station_label
        frame["execution_ts"] = pd.to_datetime(
            frame.get("execution_timestamp"),
            format="%Y-%m-%d_%H.%M.%S",
            errors="coerce",
        )
        frame.sort_values(["filename_base", "execution_ts"], inplace=True)
        frame = frame.drop_duplicates(subset=["filename_base"], keep="last")
        frame["datetime"] = frame["filename_base"].map(extract_run_datetime_from_name)
        if date_ranges:
            mask = date_range_mask(frame["datetime"], date_ranges)
            frame = frame.loc[mask].copy()
        frame["parquet_path"] = [
            path_lookup.get((station_label, basename))
            for basename in frame["filename_base"].astype(str)
        ]
        frame = frame[frame["parquet_path"].notna()].copy()
        if frame.empty:
            continue
        base_rate = _preferred_rate(frame, "1234")
        for plane, tt_label in PLANE_NO_TRIGGER.items():
            missing_rate = _preferred_rate(frame, tt_label)
            efficiency = 1.0 - (missing_rate / base_rate)
            frame[f"empirical_eff_plane_{plane}"] = efficiency.where(
                np.isfinite(efficiency) & np.isfinite(base_rate) & (base_rate > 0)
            )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_group_payload(
    station_labels: list[str],
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> GroupPayload:
    file_entries = collect_calibrated_file_entries(station_labels)
    topology_counts_by_plane: dict[int, dict[str, int]] = {
        plane: {
            "single": 0,
            "double_adj": 0,
            "triple_adj": 0,
            "quad": 0,
            "clusterlike_total": 0,
            "noncluster_total": 0,
            "nonzero_total": 0,
        }
        for plane in range(1, 5)
    }
    used_entries: list[tuple[str, Path]] = []

    parquet_columns = ["datetime"]
    for plane_cols in PLANE_STRIPS.values():
        parquet_columns.extend(plane_cols)

    for station_label, parquet_path in file_entries:
        frame = pd.read_parquet(parquet_path, columns=parquet_columns)
        frame = filter_frame_by_datetime(frame, date_ranges)
        if frame.empty:
            continue
        used_entries.append((station_label, parquet_path))

        for plane, columns in PLANE_STRIPS.items():
            values = frame[columns].to_numpy(dtype=float)
            hit_mask = np.isfinite(values) & (values > 0)
            patterns = np.array(
                ["".join("1" if bit else "0" for bit in row) for row in hit_mask],
                dtype=object,
            )
            nonzero_mask = patterns != ZERO_PATTERN
            if not np.any(nonzero_mask):
                continue
            counts = topology_counts_by_plane[plane]
            nonzero_patterns = patterns[nonzero_mask]
            counts["nonzero_total"] += int(nonzero_patterns.size)
            counts["single"] += int(np.isin(nonzero_patterns, list(SINGLE_PATTERNS)).sum())
            counts["double_adj"] += int(np.isin(nonzero_patterns, list(DOUBLE_ADJ_PATTERNS)).sum())
            counts["triple_adj"] += int(np.isin(nonzero_patterns, list(TRIPLE_ADJ_PATTERNS)).sum())
            counts["quad"] += int(np.isin(nonzero_patterns, list(QUAD_PATTERNS)).sum())
            clusterlike = np.isin(nonzero_patterns, list(CLUSTERLIKE_PATTERNS))
            counts["clusterlike_total"] += int(clusterlike.sum())
            counts["noncluster_total"] += int((~clusterlike).sum())

    return GroupPayload(
        files=used_entries,
        topology_counts_by_plane=topology_counts_by_plane,
    )


def collect_file_topology_rows(
    station_labels: list[str],
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    file_entries = collect_calibrated_file_entries(station_labels)
    parquet_columns = ["datetime"]
    for plane_cols in PLANE_STRIPS.values():
        parquet_columns.extend(plane_cols)

    for station_label, parquet_path in file_entries:
        frame = pd.read_parquet(parquet_path, columns=parquet_columns)
        frame = filter_frame_by_datetime(frame, date_ranges)
        if frame.empty:
            continue
        basename = _basename_from_calibrated_path(parquet_path)
        for plane, columns in PLANE_STRIPS.items():
            values = frame[columns].to_numpy(dtype=float)
            hit_mask = np.isfinite(values) & (values > 0)
            patterns = np.array(
                ["".join("1" if bit else "0" for bit in row) for row in hit_mask],
                dtype=object,
            )
            nonzero_mask = patterns != ZERO_PATTERN
            if not np.any(nonzero_mask):
                continue
            nonzero_patterns = patterns[nonzero_mask]
            single = int(np.isin(nonzero_patterns, list(SINGLE_PATTERNS)).sum())
            double_adj = int(np.isin(nonzero_patterns, list(DOUBLE_ADJ_PATTERNS)).sum())
            triple_adj = int(np.isin(nonzero_patterns, list(TRIPLE_ADJ_PATTERNS)).sum())
            quad = int(np.isin(nonzero_patterns, list(QUAD_PATTERNS)).sum())
            clusterlike = int(
                np.isin(nonzero_patterns, list(CLUSTERLIKE_PATTERNS)).sum()
            )
            if clusterlike <= 0:
                continue
            fractions = np.array(
                [
                    single / clusterlike,
                    double_adj / clusterlike,
                    triple_adj / clusterlike,
                    quad / clusterlike,
                ],
                dtype=float,
            )
            rows.append(
                {
                    "station_label": station_label,
                    "basename": basename,
                    "parquet_path": str(parquet_path),
                    "plane": plane,
                    "active_plane_entries": int(nonzero_patterns.size),
                    "clusterlike_entries": clusterlike,
                    "cluster_size_1": float(fractions[0]),
                    "cluster_size_2": float(fractions[1]),
                    "cluster_size_3": float(fractions[2]),
                    "cluster_size_4": float(fractions[3]),
                    "clusterlike_mean_multiplicity": clusterlike_mean_multiplicity(fractions),
                }
            )
    return pd.DataFrame(rows)


def topology_fractions(counts: dict[str, int]) -> np.ndarray:
    cluster_total = int(counts.get("clusterlike_total", 0))
    if cluster_total <= 0:
        return np.zeros(4, dtype=float)
    return np.array(
        [
            counts["single"] / cluster_total,
            counts["double_adj"] / cluster_total,
            counts["triple_adj"] / cluster_total,
            counts["quad"] / cluster_total,
        ],
        dtype=float,
    )


def clusterlike_mean_multiplicity(fractions: np.ndarray) -> float:
    return float(np.dot(fractions, MULTIPLICITY_WEIGHTS))


def direct_direction(sim_mean: float, real_mean: float, tolerance: float = 0.03) -> str:
    if not np.isfinite(sim_mean) or not np.isfinite(real_mean):
        return "insufficient_data"
    delta = real_mean - sim_mean
    if abs(delta) <= tolerance:
        return "keep_gamma"
    return "increase_gamma" if delta > 0 else "decrease_gamma"


def select_plane_matches(
    sim_metadata: pd.DataFrame,
    real_metadata: pd.DataFrame,
    match_cfg: InductionMatchConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for plane in range(1, 5):
        eff_col = f"empirical_eff_plane_{plane}"
        sim_plane = sim_metadata[
            sim_metadata[eff_col].notna()
            & (sim_metadata[eff_col] > 0)
            & (sim_metadata[eff_col] < 1)
        ].copy()
        real_plane = real_metadata[
            real_metadata[eff_col].notna()
            & (real_metadata[eff_col] > 0)
            & (real_metadata[eff_col] < 1)
        ].copy()
        if sim_plane.empty or real_plane.empty:
            continue

        sim_values = sim_plane[eff_col].to_numpy(dtype=float)
        real_values = real_plane[eff_col].to_numpy(dtype=float)
        deltas = np.abs(sim_values[:, None] - real_values[None, :])
        sim_indices, real_indices = np.where(
            deltas <= match_cfg.efficiency_match_max_abs_difference
        )
        if sim_indices.size == 0:
            continue

        candidate_rows: list[dict[str, object]] = []
        for sim_idx, real_idx in zip(sim_indices.tolist(), real_indices.tolist()):
            sim_row = sim_plane.iloc[sim_idx]
            real_row = real_plane.iloc[real_idx]
            candidate_rows.append(
                {
                    "plane": plane,
                    "sim_station": sim_row["station_label"],
                    "sim_basename": sim_row["filename_base"],
                    "sim_empirical_efficiency": float(sim_row[eff_col]),
                    "sim_path": str(sim_row["parquet_path"]),
                    "real_station": real_row["station_label"],
                    "real_basename": real_row["filename_base"],
                    "real_empirical_efficiency": float(real_row[eff_col]),
                    "real_path": str(real_row["parquet_path"]),
                    "abs_empirical_efficiency_delta": float(deltas[sim_idx, real_idx]),
                }
            )

        candidate_df = pd.DataFrame(candidate_rows).sort_values(
            [
                "abs_empirical_efficiency_delta",
                "sim_empirical_efficiency",
                "real_empirical_efficiency",
                "sim_basename",
                "real_basename",
            ]
        )
        used_sim: set[tuple[str, str]] = set()
        used_real: set[tuple[str, str]] = set()
        taken = 0
        for row in candidate_df.itertuples(index=False):
            sim_key = (row.sim_station, row.sim_basename)
            real_key = (row.real_station, row.real_basename)
            if sim_key in used_sim or real_key in used_real:
                continue
            used_sim.add(sim_key)
            used_real.add(real_key)
            rows.append(row._asdict())
            taken += 1
            if match_cfg.max_pairs_per_plane > 0 and taken >= match_cfg.max_pairs_per_plane:
                break
    return pd.DataFrame(rows)


def matched_topology_rows(
    matches_df: pd.DataFrame,
    sim_file_topology: pd.DataFrame,
    real_file_topology: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if matches_df.empty:
        return pd.DataFrame()
    sim_lookup = sim_file_topology.set_index(["station_label", "basename", "plane"])
    real_lookup = real_file_topology.set_index(["station_label", "basename", "plane"])
    for row in matches_df.itertuples(index=False):
        sim_key = (row.sim_station, row.sim_basename, row.plane)
        real_key = (row.real_station, row.real_basename, row.plane)
        if sim_key not in sim_lookup.index or real_key not in real_lookup.index:
            continue
        sim_row = sim_lookup.loc[sim_key]
        real_row = real_lookup.loc[real_key]
        sim_vec = sim_row[[f"cluster_size_{i}" for i in range(1, 5)]].to_numpy(dtype=float)
        real_vec = real_row[[f"cluster_size_{i}" for i in range(1, 5)]].to_numpy(dtype=float)
        sim_mean = float(sim_row["clusterlike_mean_multiplicity"])
        real_mean = float(real_row["clusterlike_mean_multiplicity"])
        rows.append(
            {
                "plane": int(row.plane),
                "sim_station": row.sim_station,
                "sim_basename": row.sim_basename,
                "real_station": row.real_station,
                "real_basename": row.real_basename,
                "sim_empirical_efficiency": float(row.sim_empirical_efficiency),
                "real_empirical_efficiency": float(row.real_empirical_efficiency),
                "abs_empirical_efficiency_delta": float(row.abs_empirical_efficiency_delta),
                "sim_clusterlike_entries": int(sim_row["clusterlike_entries"]),
                "real_clusterlike_entries": int(real_row["clusterlike_entries"]),
                "sim_cluster_size_1": float(sim_vec[0]),
                "sim_cluster_size_2": float(sim_vec[1]),
                "sim_cluster_size_3": float(sim_vec[2]),
                "sim_cluster_size_4": float(sim_vec[3]),
                "real_cluster_size_1": float(real_vec[0]),
                "real_cluster_size_2": float(real_vec[1]),
                "real_cluster_size_3": float(real_vec[2]),
                "real_cluster_size_4": float(real_vec[3]),
                "sim_clusterlike_mean_multiplicity": sim_mean,
                "real_clusterlike_mean_multiplicity": real_mean,
                "mean_multiplicity_scale_real_over_sim": (
                    float(real_mean / sim_mean) if np.isfinite(sim_mean) and sim_mean > 0 else np.nan
                ),
                "topology_l1_distance": float(np.sum(np.abs(real_vec - sim_vec))),
                "topology_l2_distance": float(np.linalg.norm(real_vec - sim_vec)),
            }
        )
    return pd.DataFrame(rows)


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    denom = float(np.dot(x, x))
    if x.size == 0 or denom <= 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def per_group_rows(group_name: str, payload: GroupPayload) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for plane in range(1, 5):
        counts = payload.topology_counts_by_plane[plane]
        fractions = topology_fractions(counts)
        rows.append(
            {
                "group": group_name,
                "plane": plane,
                "files": len(payload.files),
                "active_plane_entries": int(counts["nonzero_total"]),
                "clusterlike_entries": int(counts["clusterlike_total"]),
                "noncluster_entries": int(counts["noncluster_total"]),
                "clusterlike_fraction_of_nonzero": (
                    float(counts["clusterlike_total"] / counts["nonzero_total"])
                    if counts["nonzero_total"] > 0
                    else np.nan
                ),
                "cluster_size_1": float(fractions[0]),
                "cluster_size_2": float(fractions[1]),
                "cluster_size_3": float(fractions[2]),
                "cluster_size_4": float(fractions[3]),
                "clusterlike_mean_multiplicity": clusterlike_mean_multiplicity(fractions),
            }
        )
    return rows


def comparison_rows(
    overall_df: pd.DataFrame,
    *,
    min_clusterlike_entries_per_plane: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for plane in range(1, 5):
        plane_df = overall_df[overall_df["plane"] == plane].copy()
        sim_row = plane_df[plane_df["group"] == "SIM"].iloc[0]
        real_row = plane_df[plane_df["group"] == "REAL"].iloc[0]
        sim_vec = sim_row[[f"cluster_size_{i}" for i in range(1, 5)]].to_numpy(dtype=float)
        real_vec = real_row[[f"cluster_size_{i}" for i in range(1, 5)]].to_numpy(dtype=float)
        delta_vec = real_vec - sim_vec
        sim_entries = int(sim_row["clusterlike_entries"])
        real_entries = int(real_row["clusterlike_entries"])
        enough_stats = bool(
            (sim_entries >= min_clusterlike_entries_per_plane)
            and (real_entries >= min_clusterlike_entries_per_plane)
        )
        sim_mean = float(sim_row["clusterlike_mean_multiplicity"])
        real_mean = float(real_row["clusterlike_mean_multiplicity"])
        rows.append(
            {
                "plane": plane,
                "sim_clusterlike_entries": sim_entries,
                "real_clusterlike_entries": real_entries,
                "enough_stats": float(enough_stats),
                "sim_cluster_size_1": float(sim_vec[0]),
                "sim_cluster_size_2": float(sim_vec[1]),
                "sim_cluster_size_3": float(sim_vec[2]),
                "sim_cluster_size_4": float(sim_vec[3]),
                "real_cluster_size_1": float(real_vec[0]),
                "real_cluster_size_2": float(real_vec[1]),
                "real_cluster_size_3": float(real_vec[2]),
                "real_cluster_size_4": float(real_vec[3]),
                "delta_cluster_size_1_real_minus_sim": float(delta_vec[0]),
                "delta_cluster_size_2_real_minus_sim": float(delta_vec[1]),
                "delta_cluster_size_3_real_minus_sim": float(delta_vec[2]),
                "delta_cluster_size_4_real_minus_sim": float(delta_vec[3]),
                "sim_clusterlike_mean_multiplicity": sim_mean,
                "real_clusterlike_mean_multiplicity": real_mean,
                "delta_mean_multiplicity_real_minus_sim": float(real_mean - sim_mean),
                "topology_l1_distance": float(np.sum(np.abs(delta_vec))),
                "topology_l2_distance": float(np.linalg.norm(delta_vec)),
                "suggested_gamma_direction": direct_direction(sim_mean, real_mean),
            }
        )
    return pd.DataFrame(rows)


def plot_overall_multiplicity(overall_df: pd.DataFrame, current_gamma_mm: float) -> None:
    out = output_dir()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    multiplicities = [1, 2, 3, 4]
    width = 0.35

    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        plane_df = overall_df[overall_df["plane"] == plane].copy()
        sim_row = plane_df[plane_df["group"] == "SIM"].iloc[0]
        real_row = plane_df[plane_df["group"] == "REAL"].iloc[0]
        x = np.arange(len(multiplicities))
        sim_vals = [sim_row[f"cluster_size_{m}"] for m in multiplicities]
        real_vals = [real_row[f"cluster_size_{m}"] for m in multiplicities]
        ax.bar(x - width / 2, sim_vals, width=width, color=GROUP_COLORS["SIM"], label="SIM")
        ax.bar(x + width / 2, real_vals, width=width, color=GROUP_COLORS["REAL"], label="REAL")
        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in multiplicities])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"Plane {plane}\n"
            f"SIM mean={sim_row['clusterlike_mean_multiplicity']:.3f} | "
            f"REAL mean={real_row['clusterlike_mean_multiplicity']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Strip multiplicity")
        ax.grid(axis="y", alpha=0.2)
        if plane in (1, 3):
            ax.set_ylabel("Fraction")
        if plane == 1:
            ax.legend()
        ax.text(
            0.02,
            0.98,
            f"gamma={current_gamma_mm:.2f} mm\n"
            f"clusterlike SIM={int(sim_row['clusterlike_entries'])}\n"
            f"clusterlike REAL={int(real_row['clusterlike_entries'])}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )

    fig.suptitle("Overall same-plane strip multiplicity from calibrated Task 2 charges", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "induction_overall_multiplicity.png", dpi=180)
    fig.savefig(out / "induction_overall_multiplicity.pdf")
    plt.close(fig)


def plot_direct_comparison(comparison_df: pd.DataFrame, current_gamma_mm: float) -> None:
    out = output_dir()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)

    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        row = comparison_df[comparison_df["plane"] == plane].iloc[0]
        ax.bar(
            ["SIM", "REAL"],
            [
                row["sim_clusterlike_mean_multiplicity"],
                row["real_clusterlike_mean_multiplicity"],
            ],
            color=[GROUP_COLORS["SIM"], GROUP_COLORS["REAL"]],
        )
        ax.set_title(f"Plane {plane}", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("Group")
        if plane in (1, 3):
            ax.set_ylabel("Mean multiplicity among clusterlike hits")
        ax.text(
            0.02,
            0.98,
            f"gamma={current_gamma_mm:.2f} mm\n"
            f"real-sim={row['delta_mean_multiplicity_real_minus_sim']:+.3f}\n"
            f"L1={row['topology_l1_distance']:.3f}\n"
            f"L2={row['topology_l2_distance']:.3f}\n"
            f"{row['suggested_gamma_direction']}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )

    fig.suptitle(
        "Direct induction comparison from adjacent multiplicity topology only",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "induction_direct_comparison.png", dpi=180)
    fig.savefig(out / "induction_direct_comparison.pdf")
    plt.close(fig)


def matched_summary_rows(
    matched_df: pd.DataFrame,
    current_gamma_mm: float,
    match_cfg: InductionMatchConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for plane in range(1, 5):
        plane_df = matched_df[matched_df["plane"] == plane].copy()
        if plane_df.empty:
            rows.append(
                {
                    "plane": plane,
                    "matched_pairs": 0,
                    "median_abs_empirical_efficiency_delta": np.nan,
                    "median_mean_multiplicity_scale_real_over_sim": np.nan,
                    "fit_mean_multiplicity_scale_real_vs_sim": np.nan,
                    "recommended_gamma_scale": np.nan,
                    "recommended_gamma_mm": np.nan,
                    "median_topology_l1_distance": np.nan,
                    "median_topology_l2_distance": np.nan,
                    "suggested_gamma_direction": "insufficient_data",
                }
            )
            continue
        sim_mean = plane_df["sim_clusterlike_mean_multiplicity"].to_numpy(dtype=float)
        real_mean = plane_df["real_clusterlike_mean_multiplicity"].to_numpy(dtype=float)
        scale_fit = slope_through_origin(sim_mean, real_mean)
        median_scale = float(np.nanmedian(plane_df["mean_multiplicity_scale_real_over_sim"]))
        recommended_scale = (
            scale_fit if np.isfinite(scale_fit) else median_scale
        )
        rows.append(
            {
                "plane": plane,
                "matched_pairs": int(len(plane_df)),
                "median_abs_empirical_efficiency_delta": float(
                    plane_df["abs_empirical_efficiency_delta"].median()
                ),
                "median_mean_multiplicity_scale_real_over_sim": median_scale,
                "fit_mean_multiplicity_scale_real_vs_sim": scale_fit,
                "recommended_gamma_scale": recommended_scale,
                "recommended_gamma_mm": (
                    float(current_gamma_mm * recommended_scale)
                    if np.isfinite(recommended_scale)
                    else np.nan
                ),
                "median_topology_l1_distance": float(plane_df["topology_l1_distance"].median()),
                "median_topology_l2_distance": float(plane_df["topology_l2_distance"].median()),
                "suggested_gamma_direction": direct_direction(
                    float(np.nanmedian(sim_mean)),
                    float(np.nanmedian(real_mean)),
                ),
            }
        )
    summary_df = pd.DataFrame(rows)
    enough = summary_df["matched_pairs"].to_numpy(dtype=float) >= max(3, min(match_cfg.max_pairs_per_plane, 3))
    valid_scales = summary_df.loc[enough, "recommended_gamma_scale"].to_numpy(dtype=float)
    overall_scale = float(np.nanmedian(valid_scales)) if valid_scales.size else np.nan
    summary_df["overall_recommended_gamma_scale"] = overall_scale
    summary_df["overall_recommended_gamma_mm"] = (
        float(current_gamma_mm * overall_scale) if np.isfinite(overall_scale) else np.nan
    )
    return summary_df


def plot_matched_mean_vs_efficiency(matched_df: pd.DataFrame, current_gamma_mm: float) -> None:
    out = output_dir()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        plane_df = matched_df[matched_df["plane"] == plane].copy()
        if plane_df.empty:
            ax.axis("off")
            continue
        ax.scatter(
            plane_df["sim_empirical_efficiency"],
            plane_df["sim_clusterlike_mean_multiplicity"],
            s=16,
            alpha=0.7,
            color=GROUP_COLORS["SIM"],
            label="SIM",
        )
        ax.scatter(
            plane_df["real_empirical_efficiency"],
            plane_df["real_clusterlike_mean_multiplicity"],
            s=16,
            alpha=0.7,
            color=GROUP_COLORS["REAL"],
            label="REAL",
        )
        ax.set_title(f"Plane {plane}")
        ax.set_xlabel("Empirical efficiency")
        if plane in (1, 3):
            ax.set_ylabel("Mean multiplicity among clusterlike hits")
        ax.grid(alpha=0.2)
        ax.text(
            0.02,
            0.98,
            f"gamma={current_gamma_mm:.2f} mm\npairs={len(plane_df)}\n"
            f"median |Δeff|={plane_df['abs_empirical_efficiency_delta'].median():.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
        if plane == 1:
            ax.legend()
    fig.suptitle("Matched-file topology mean vs metadata empirical efficiency", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "induction_matched_mean_vs_efficiency.png", dpi=180)
    fig.savefig(out / "induction_matched_mean_vs_efficiency.pdf")
    plt.close(fig)


def plot_matched_real_vs_sim_mean(matched_df: pd.DataFrame, matched_summary_df: pd.DataFrame) -> None:
    out = output_dir()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        plane_df = matched_df[matched_df["plane"] == plane].copy()
        if plane_df.empty:
            ax.axis("off")
            continue
        x = plane_df["sim_clusterlike_mean_multiplicity"].to_numpy(dtype=float)
        y = plane_df["real_clusterlike_mean_multiplicity"].to_numpy(dtype=float)
        scale = float(
            matched_summary_df.loc[
                matched_summary_df["plane"] == plane,
                "fit_mean_multiplicity_scale_real_vs_sim",
            ].iloc[0]
        )
        ax.scatter(x, y, s=18, alpha=0.75, color="#4c78a8")
        line_min = float(min(np.nanmin(x), np.nanmin(y)))
        line_max = float(max(np.nanmax(x), np.nanmax(y)))
        diag = np.linspace(line_min, line_max, 100)
        ax.plot(diag, diag, color="gray", linestyle="--", linewidth=1.0, label="y=x")
        if np.isfinite(scale):
            ax.plot(diag, scale * diag, color="#e15759", linewidth=1.2, label=f"fit y={scale:.3f}x")
        ax.set_title(f"Plane {plane}")
        ax.set_xlabel("SIM mean multiplicity")
        if plane in (1, 3):
            ax.set_ylabel("REAL mean multiplicity")
        ax.grid(alpha=0.2)
        if plane == 1:
            ax.legend()
    fig.suptitle("Matched-file REAL vs SIM topology mean", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "induction_matched_real_vs_sim_mean.png", dpi=180)
    fig.savefig(out / "induction_matched_real_vs_sim_mean.pdf")
    plt.close(fig)


def write_report(
    selection,
    overall_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    matched_summary_df: pd.DataFrame,
    current_gamma_mm: float,
) -> pd.DataFrame:
    out = output_dir()
    sim_means = comparison_df["sim_clusterlike_mean_multiplicity"].to_numpy(dtype=float)
    real_means = comparison_df["real_clusterlike_mean_multiplicity"].to_numpy(dtype=float)
    deltas = comparison_df["delta_mean_multiplicity_real_minus_sim"].to_numpy(dtype=float)
    l1_vals = comparison_df["topology_l1_distance"].to_numpy(dtype=float)
    l2_vals = comparison_df["topology_l2_distance"].to_numpy(dtype=float)
    enough_stats = comparison_df["enough_stats"].to_numpy(dtype=float) > 0.5

    valid_delta = deltas[enough_stats] if np.any(enough_stats) else deltas
    median_delta = float(np.nanmedian(valid_delta)) if valid_delta.size else np.nan
    overall_direction = direct_direction(
        float(np.nanmedian(sim_means)),
        float(np.nanmedian(real_means)),
    )

    overall_matched_scale = float(
        matched_summary_df["overall_recommended_gamma_scale"].dropna().iloc[0]
    ) if not matched_summary_df["overall_recommended_gamma_scale"].dropna().empty else np.nan
    overall_matched_gamma = float(
        matched_summary_df["overall_recommended_gamma_mm"].dropna().iloc[0]
    ) if not matched_summary_df["overall_recommended_gamma_mm"].dropna().empty else np.nan
    summary_df = pd.DataFrame(
        [
            {"metric": "current_step4_lorentzian_gamma_mm", "value": current_gamma_mm},
            {"metric": "median_sim_clusterlike_mean_multiplicity", "value": float(np.nanmedian(sim_means))},
            {"metric": "median_real_clusterlike_mean_multiplicity", "value": float(np.nanmedian(real_means))},
            {"metric": "median_real_minus_sim_mean_multiplicity", "value": median_delta},
            {"metric": "median_topology_l1_distance", "value": float(np.nanmedian(l1_vals))},
            {"metric": "median_topology_l2_distance", "value": float(np.nanmedian(l2_vals))},
            {"metric": "planes_with_enough_stats", "value": float(np.sum(enough_stats))},
            {"metric": "matched_overall_recommended_gamma_scale", "value": overall_matched_scale},
            {"metric": "matched_overall_recommended_gamma_mm", "value": overall_matched_gamma},
            {
                "metric": "overall_direction_code",
                "value": {
                    "decrease_gamma": -1.0,
                    "keep_gamma": 0.0,
                    "increase_gamma": 1.0,
                }.get(overall_direction, np.nan),
            },
        ]
    )
    summary_df.to_csv(out / "recommended_induction_section_summary.csv", index=False)

    per_plane_lines = []
    for plane in range(1, 5):
        row = comparison_df[comparison_df["plane"] == plane].iloc[0]
        matched_row = matched_summary_df[matched_summary_df["plane"] == plane].iloc[0]
        per_plane_lines.append(
            f"- Plane {plane}: "
            f"SIM mean={row['sim_clusterlike_mean_multiplicity']:.3f}, "
            f"REAL mean={row['real_clusterlike_mean_multiplicity']:.3f}, "
            f"delta(real-sim)={row['delta_mean_multiplicity_real_minus_sim']:+.3f}, "
            f"L1={row['topology_l1_distance']:.3f}, "
            f"direction={row['suggested_gamma_direction']}, "
            f"matched scale={matched_row['recommended_gamma_scale']:.3f}, "
            f"matched gamma={matched_row['recommended_gamma_mm']:.2f} mm"
        )

    report = f"""Induction-section tuning from direct simulated-vs-real topology comparison
==================================================

Simulation stations:
- {", ".join(selection.simulation_stations)}

Real-data stations:
- {", ".join(selection.real_stations)}

Current STEP 4 Lorentzian gamma in the digital twin:
- {current_gamma_mm:.2f} mm

Method:
- No LUT inversion is used.
- Only the adjacent cluster topology fractions `(single, double-adjacent, triple-adjacent, quad)` are compared.
- Non-adjacent patterns such as `1010`, `0101`, `1001`, `1101`, `1011` are excluded from the comparison metric.
- The main direct observable is the mean multiplicity among clusterlike hits:
  `1*f_single + 2*f_double_adj + 3*f_triple_adj + 4*f_quad`.

Overall comparison:
- SIM median clusterlike mean multiplicity: {float(np.nanmedian(sim_means)):.3f}
- REAL median clusterlike mean multiplicity: {float(np.nanmedian(real_means)):.3f}
- Median REAL minus SIM mean multiplicity: {median_delta:+.3f}
- Median topology L1 distance: {float(np.nanmedian(l1_vals)):.3f}
- Median topology L2 distance: {float(np.nanmedian(l2_vals)):.3f}
- Overall suggested direction: {overall_direction}

Matched-file comparison at similar empirical efficiency:
- Matching uses Task 3 metadata empirical efficiencies for both MINGO00 and real data.
- Overall matched gamma scale: {overall_matched_scale:.3f}
- Overall matched gamma recommendation: {overall_matched_gamma:.2f} mm
- This is a first-order scale estimate from the matched-file topology mean relation, not a fully calibrated gamma fit.

Practical interpretation:
- If REAL mean multiplicity is above SIM, the current `lorentzian_gamma_mm` is likely too small and should be increased in the next simulation test.
- If REAL mean multiplicity is below SIM, the current `lorentzian_gamma_mm` is likely too large and should be decreased in the next simulation test.
- The matched-file scale is the more useful recommendation when the empirical-efficiency matching is tight enough.
- This study now reports direction and mismatch strength directly from the simulation outputs themselves, without relying on the old induction LUT.

Plane-by-plane summary:
{chr(10).join(per_plane_lines)}
"""
    (out / "induction_section_report.txt").write_text(report, encoding="utf-8")
    return summary_df


def main() -> None:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    config = load_tuning_config()
    selection = resolve_selection(config)
    match_cfg = load_match_config(config)

    sim_payload = collect_group_payload(selection.simulation_stations, selection.simulation_date_ranges)
    real_payload = collect_group_payload(selection.real_stations, selection.real_date_ranges)
    sim_file_topology = collect_file_topology_rows(
        selection.simulation_stations,
        selection.simulation_date_ranges,
    )
    real_file_topology = collect_file_topology_rows(
        selection.real_stations,
        selection.real_date_ranges,
    )
    sim_metadata = load_latest_trigger_metadata(
        selection.simulation_stations,
        selection.simulation_date_ranges,
    )
    real_metadata = load_latest_trigger_metadata(
        selection.real_stations,
        selection.real_date_ranges,
    )

    overall_rows = []
    overall_rows.extend(per_group_rows("SIM", sim_payload))
    overall_rows.extend(per_group_rows("REAL", real_payload))
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out / "overall_induction_summary.csv", index=False)

    current_gamma_mm = load_simulation_induction_width_mm()
    if not overall_df.empty:
        plot_overall_multiplicity(overall_df, current_gamma_mm)
        comparison_df = comparison_rows(
            overall_df,
            min_clusterlike_entries_per_plane=match_cfg.min_clusterlike_entries_per_plane,
        )
        comparison_df.to_csv(out / "induction_direct_comparison_summary.csv", index=False)
        plot_direct_comparison(comparison_df, current_gamma_mm)
        matches_df = select_plane_matches(sim_metadata, real_metadata, match_cfg)
        matches_df.to_csv(out / "induction_efficiency_matches.csv", index=False)
        matched_df = matched_topology_rows(matches_df, sim_file_topology, real_file_topology)
        matched_df.to_csv(out / "induction_matched_topology_summary.csv", index=False)
        matched_summary_df = matched_summary_rows(matched_df, current_gamma_mm, match_cfg)
        matched_summary_df.to_csv(out / "induction_matched_gamma_recommendation.csv", index=False)
        if not matched_df.empty:
            plot_matched_mean_vs_efficiency(matched_df, current_gamma_mm)
            plot_matched_real_vs_sim_mean(matched_df, matched_summary_df)
        write_report(selection, overall_df, comparison_df, matched_summary_df, current_gamma_mm)


if __name__ == "__main__":
    main()
