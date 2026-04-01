#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from MASTER.ANCILLARY.SIMULATION_TUNING.common import (
    date_range_mask,
    load_tuning_config,
    resolve_selection,
)
from MASTER.common.file_selection import extract_run_datetime_from_name
from MASTER.common.tot_charge_calibration import (
    TotChargeCalibration,
    default_tot_charge_calibration_path,
)


PLANE_QSUM_FINAL = {
    plane: f"P{plane}_Q_sum_final"
    for plane in range(1, 5)
}
PLANE_ACTIVE_STRIPS = {
    plane: f"active_strips_P{plane}"
    for plane in range(1, 5)
}
PLANE_NO_TRIGGER = {
    1: "234",
    2: "134",
    3: "124",
    4: "123",
}
GROUP_COLORS = {
    "SIM": "#d95f02",
    "REAL": "#1b9e77",
    "SCALED": "#7570b3",
}
DEFAULT_SEED = 20260331
SAMPLE_CACHE: dict[tuple[str, int, str, int, float], np.ndarray] = {}
UNIT_LABELS = {
    "ns": "ns",
    "fc": "fC",
}


@dataclass(frozen=True)
class ChargeStudyConfig:
    multiplicity_mode: str
    efficiency_match_max_abs_difference: float
    efficiency_match_fallback_max_abs_difference: float
    max_pairs_per_plane: int
    min_pairs_per_plane: int
    sample_entries_per_file: int
    min_positive_charge_ns: float
    hist_bin_count: int
    alpha_min: float
    alpha_max: float
    alpha_grid_points: int


@dataclass(frozen=True)
class CurrentChargeModel:
    townsend_alpha_per_mm: float
    avalanche_gap_mm: float
    avalanche_electron_sigma: float
    lorentzian_gamma_mm: float
    induced_charge_fraction: float
    qdiff_width: float
    charge_conversion_model: str
    q_to_time_factor: float
    charge_threshold: float


def output_dir() -> Path:
    return Path(__file__).resolve().parent / "OUTPUTS"


def load_charge_study_config(config: dict) -> ChargeStudyConfig:
    study_cfg = config.get("charge_spectrum", {})
    multiplicity_mode = str(study_cfg.get("multiplicity_mode", "single_strip_only")).strip().lower()
    if multiplicity_mode not in {"single_strip_only", "all_active"}:
        raise ValueError(
            "charge_spectrum.multiplicity_mode must be 'single_strip_only' or 'all_active'."
        )
    return ChargeStudyConfig(
        multiplicity_mode=multiplicity_mode,
        efficiency_match_max_abs_difference=float(
            study_cfg.get("efficiency_match_max_abs_difference", 0.05)
        ),
        efficiency_match_fallback_max_abs_difference=float(
            study_cfg.get("efficiency_match_fallback_max_abs_difference", 0.15)
        ),
        max_pairs_per_plane=int(study_cfg.get("max_pairs_per_plane", 80)),
        min_pairs_per_plane=int(study_cfg.get("min_pairs_per_plane", 6)),
        sample_entries_per_file=int(study_cfg.get("sample_entries_per_file", 1500)),
        min_positive_charge_ns=float(study_cfg.get("min_positive_charge_ns", 0.05)),
        hist_bin_count=int(study_cfg.get("hist_bin_count", 90)),
        alpha_min=float(study_cfg.get("alpha_min", 0.40)),
        alpha_max=float(study_cfg.get("alpha_max", 2.00)),
        alpha_grid_points=int(study_cfg.get("alpha_grid_points", 161)),
    )


def load_current_charge_model() -> CurrentChargeModel:
    step3_path = ROOT_DIR / "MINGO_DIGITAL_TWIN" / "MASTER_STEPS" / "STEP_3" / "config_step_3_physics.yaml"
    step4_path = ROOT_DIR / "MINGO_DIGITAL_TWIN" / "MASTER_STEPS" / "STEP_4" / "config_step_4_physics.yaml"
    step5_path = ROOT_DIR / "MINGO_DIGITAL_TWIN" / "MASTER_STEPS" / "STEP_5" / "config_step_5_physics.yaml"
    step8_path = ROOT_DIR / "MINGO_DIGITAL_TWIN" / "MASTER_STEPS" / "STEP_8" / "config_step_8_physics.yaml"
    with step3_path.open("r", encoding="utf-8") as handle:
        step3_cfg = yaml.safe_load(handle) or {}
    with step4_path.open("r", encoding="utf-8") as handle:
        step4_cfg = yaml.safe_load(handle) or {}
    with step5_path.open("r", encoding="utf-8") as handle:
        step5_cfg = yaml.safe_load(handle) or {}
    with step8_path.open("r", encoding="utf-8") as handle:
        step8_cfg = yaml.safe_load(handle) or {}
    return CurrentChargeModel(
        townsend_alpha_per_mm=float(step3_cfg.get("townsend_alpha_per_mm", np.nan)),
        avalanche_gap_mm=float(step3_cfg.get("avalanche_gap_mm", np.nan)),
        avalanche_electron_sigma=float(step3_cfg.get("avalanche_electron_sigma", np.nan)),
        lorentzian_gamma_mm=float(
            step4_cfg.get(
                "lorentzian_gamma_mm",
                0.5 * float(step4_cfg.get("avalanche_width_mm", np.nan)),
            )
        ),
        induced_charge_fraction=float(step4_cfg.get("induced_charge_fraction", 1.0)),
        qdiff_width=float(step5_cfg.get("qdiff_width", np.nan)),
        charge_conversion_model=str(
            step8_cfg.get("charge_conversion_model", "linear_q_to_time_factor")
        ).strip(),
        q_to_time_factor=float(step8_cfg.get("q_to_time_factor", np.nan)),
        charge_threshold=float(step8_cfg.get("charge_threshold", np.nan)),
    )


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


def station_listed_input_dirs(station_label: str) -> list[Path]:
    base = (
        ROOT_DIR
        / "STATIONS"
        / station_label
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_4"
        / "INPUT_FILES"
    )
    return [
        base / "COMPLETED_DIRECTORY",
        base / "UNPROCESSED_DIRECTORY",
        base / "PROCESSING_DIRECTORY",
    ]


def collect_listed_file_entries(station_labels: list[str]) -> list[tuple[str, Path]]:
    files_by_key: dict[tuple[str, str], Path] = {}
    for station_label in station_labels:
        for input_dir in station_listed_input_dirs(station_label):
            if not input_dir.exists():
                continue
            for parquet_path in sorted(input_dir.glob("listed_*.parquet")):
                files_by_key[(station_label, parquet_path.name)] = parquet_path
    return [
        (station_label, files_by_key[(station_label, name)])
        for station_label, name in sorted(files_by_key)
    ]


def resolve_listed_file_path(
    station_label: str,
    basename: str,
    preferred_path: str | Path,
) -> Path:
    preferred = Path(preferred_path)
    if preferred.exists():
        return preferred
    target_name = f"listed_{basename}.parquet"
    for directory in station_listed_input_dirs(station_label):
        candidate = directory / target_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing listed parquet for {station_label} / {basename}")


def _basename_from_listed_path(path: Path) -> str:
    name = path.name
    if name.startswith("listed_") and name.endswith(".parquet"):
        return name[len("listed_") : -len(".parquet")]
    return path.stem


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


def _global_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    handles: list[object] = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), fontsize=9)


def load_latest_trigger_metadata(
    station_labels: list[str],
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> pd.DataFrame:
    path_lookup = {
        (station_label, _basename_from_listed_path(path)): path
        for station_label, path in collect_listed_file_entries(station_labels)
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
        frame["empirical_rate_1234"] = base_rate
        for plane, tt_label in PLANE_NO_TRIGGER.items():
            missing_rate = _preferred_rate(frame, tt_label)
            efficiency = 1.0 - (missing_rate / base_rate)
            frame[f"empirical_eff_plane_{plane}"] = efficiency.where(
                np.isfinite(efficiency) & np.isfinite(base_rate) & (base_rate > 0)
            )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined


def select_plane_matches(
    sim_metadata: pd.DataFrame,
    real_metadata: pd.DataFrame,
    study_cfg: ChargeStudyConfig,
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
        max_unique_pairs = min(len(sim_plane), len(real_plane))
        if study_cfg.max_pairs_per_plane > 0:
            max_unique_pairs = min(max_unique_pairs, study_cfg.max_pairs_per_plane)
        target_pairs = max_unique_pairs
        if study_cfg.min_pairs_per_plane > 0:
            target_pairs = min(target_pairs, study_cfg.min_pairs_per_plane)
        if target_pairs <= 0:
            continue

        sim_values = sim_plane[eff_col].to_numpy(dtype=float)
        real_values = real_plane[eff_col].to_numpy(dtype=float)
        deltas = np.abs(sim_values[:, None] - real_values[None, :])
        sim_indices, real_indices = np.where(
            deltas <= study_cfg.efficiency_match_max_abs_difference
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
                    "sim_pool_size": int(len(sim_plane)),
                    "real_pool_size": int(len(real_plane)),
                    "target_pairs": int(target_pairs),
                    "sim_station": sim_row["station_label"],
                    "sim_basename": sim_row["filename_base"],
                    "sim_empirical_efficiency": float(sim_row[eff_col]),
                    "sim_path": str(sim_row["parquet_path"]),
                    "real_station": real_row["station_label"],
                    "real_basename": real_row["filename_base"],
                    "real_empirical_efficiency": float(real_row[eff_col]),
                    "real_path": str(real_row["parquet_path"]),
                    "abs_empirical_efficiency_delta": float(deltas[sim_idx, real_idx]),
                    "match_mode": "within_threshold",
                }
            )

        candidate_df = pd.DataFrame(candidate_rows)
        fallback_rows: list[dict[str, object]] = []
        if candidate_df.empty or len(candidate_df) < target_pairs:
            fallback_limit = max(
                study_cfg.efficiency_match_max_abs_difference,
                study_cfg.efficiency_match_fallback_max_abs_difference,
            )
            fallback_sim_indices, fallback_real_indices = np.where(deltas <= fallback_limit)
            for sim_idx, real_idx in zip(fallback_sim_indices.tolist(), fallback_real_indices.tolist()):
                sim_row = sim_plane.iloc[sim_idx]
                real_row = real_plane.iloc[real_idx]
                delta_value = float(deltas[sim_idx, real_idx])
                if delta_value <= study_cfg.efficiency_match_max_abs_difference:
                    continue
                fallback_rows.append(
                    {
                        "plane": plane,
                        "sim_pool_size": int(len(sim_plane)),
                        "real_pool_size": int(len(real_plane)),
                        "target_pairs": int(target_pairs),
                        "sim_station": sim_row["station_label"],
                        "sim_basename": sim_row["filename_base"],
                        "sim_empirical_efficiency": float(sim_row[eff_col]),
                        "sim_path": str(sim_row["parquet_path"]),
                        "real_station": real_row["station_label"],
                        "real_basename": real_row["filename_base"],
                        "real_empirical_efficiency": float(real_row[eff_col]),
                        "real_path": str(real_row["parquet_path"]),
                        "abs_empirical_efficiency_delta": delta_value,
                        "match_mode": "fallback_nearest",
                    }
                )
        if fallback_rows:
            candidate_df = pd.concat([candidate_df, pd.DataFrame(fallback_rows)], ignore_index=True)
        if candidate_df.empty:
            continue
        candidate_df = candidate_df.sort_values(
            [
                "abs_empirical_efficiency_delta",
                "match_mode",
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
            if study_cfg.max_pairs_per_plane > 0 and taken >= study_cfg.max_pairs_per_plane:
                break
        if taken < target_pairs:
            remaining_rows: list[dict[str, object]] = []
            all_sim_indices, all_real_indices = np.where(np.isfinite(deltas))
            for sim_idx, real_idx in zip(all_sim_indices.tolist(), all_real_indices.tolist()):
                sim_row = sim_plane.iloc[sim_idx]
                real_row = real_plane.iloc[real_idx]
                sim_key = (sim_row["station_label"], sim_row["filename_base"])
                real_key = (real_row["station_label"], real_row["filename_base"])
                if sim_key in used_sim or real_key in used_real:
                    continue
                remaining_rows.append(
                    {
                        "plane": plane,
                        "sim_pool_size": int(len(sim_plane)),
                        "real_pool_size": int(len(real_plane)),
                        "target_pairs": int(target_pairs),
                        "sim_station": sim_row["station_label"],
                        "sim_basename": sim_row["filename_base"],
                        "sim_empirical_efficiency": float(sim_row[eff_col]),
                        "sim_path": str(sim_row["parquet_path"]),
                        "real_station": real_row["station_label"],
                        "real_basename": real_row["filename_base"],
                        "real_empirical_efficiency": float(real_row[eff_col]),
                        "real_path": str(real_row["parquet_path"]),
                        "abs_empirical_efficiency_delta": float(deltas[sim_idx, real_idx]),
                        "match_mode": "forced_nearest",
                    }
                )
            if remaining_rows:
                remaining_df = pd.DataFrame(remaining_rows).sort_values(
                    [
                        "abs_empirical_efficiency_delta",
                        "sim_empirical_efficiency",
                        "real_empirical_efficiency",
                        "sim_basename",
                        "real_basename",
                    ]
                )
                for row in remaining_df.itertuples(index=False):
                    sim_key = (row.sim_station, row.sim_basename)
                    real_key = (row.real_station, row.real_basename)
                    if sim_key in used_sim or real_key in used_real:
                        continue
                    used_sim.add(sim_key)
                    used_real.add(real_key)
                    rows.append(row._asdict())
                    taken += 1
                    if taken >= target_pairs:
                        break
                    if study_cfg.max_pairs_per_plane > 0 and taken >= study_cfg.max_pairs_per_plane:
                        break

    return pd.DataFrame(rows)


def _deterministic_sample_indices(
    total: int,
    wanted: int,
    token: str,
) -> np.ndarray:
    if total <= wanted:
        return np.arange(total)
    digest = hashlib.sha256(f"{DEFAULT_SEED}|{token}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=wanted, replace=False))


def _strip_multiplicity(series: pd.Series) -> np.ndarray:
    text = series.astype("string").fillna("0000")
    return np.array([str(value).count("1") for value in text], dtype=int)


def sample_plane_charge_ns(
    parquet_path: str | Path,
    plane: int,
    study_cfg: ChargeStudyConfig,
) -> np.ndarray:
    cache_key = (
        str(parquet_path),
        plane,
        study_cfg.multiplicity_mode,
        study_cfg.sample_entries_per_file,
        study_cfg.min_positive_charge_ns,
    )
    cached = SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    path = Path(parquet_path)
    charge_col = PLANE_QSUM_FINAL[plane]
    active_col = PLANE_ACTIVE_STRIPS[plane]
    frame = pd.read_parquet(path, columns=[charge_col, active_col])
    multiplicity = _strip_multiplicity(frame[active_col])
    plane_charge_ns = pd.to_numeric(frame[charge_col], errors="coerce").to_numpy(dtype=float)

    active = multiplicity > 0
    if study_cfg.multiplicity_mode == "single_strip_only":
        active &= multiplicity == 1

    selected = plane_charge_ns[active]
    selected = selected[np.isfinite(selected) & (selected > study_cfg.min_positive_charge_ns)]
    if selected.size == 0:
        SAMPLE_CACHE[cache_key] = selected
        return selected

    indices = _deterministic_sample_indices(
        selected.size,
        study_cfg.sample_entries_per_file,
        f"{path.name}|P{plane}|{study_cfg.multiplicity_mode}",
    )
    sampled = selected[indices]
    SAMPLE_CACHE[cache_key] = sampled
    return sampled


def pooled_plane_samples(
    matches_df: pd.DataFrame,
    plane: int,
    study_cfg: ChargeStudyConfig,
    unit_key: str,
    calibration: TotChargeCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    plane_df = matches_df[matches_df["plane"] == plane].copy()
    sim_chunks: list[np.ndarray] = []
    real_chunks: list[np.ndarray] = []
    for row in plane_df.itertuples(index=False):
        sim_path = resolve_listed_file_path(row.sim_station, row.sim_basename, row.sim_path)
        real_path = resolve_listed_file_path(row.real_station, row.real_basename, row.real_path)
        sim_values = sample_plane_charge_ns(sim_path, plane, study_cfg)
        real_values = sample_plane_charge_ns(real_path, plane, study_cfg)
        if unit_key == "fc":
            sim_values = calibration.width_ns_to_charge_fc(sim_values)
            real_values = calibration.width_ns_to_charge_fc(real_values)
        if sim_values.size:
            sim_chunks.append(sim_values)
        if real_values.size:
            real_chunks.append(real_values)

    sim_pooled = np.concatenate(sim_chunks) if sim_chunks else np.array([], dtype=float)
    real_pooled = np.concatenate(real_chunks) if real_chunks else np.array([], dtype=float)
    return sim_pooled, real_pooled


def _positive_log10(values: np.ndarray) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0)]
    return np.log10(positive)


def robust_log_sigma(values: np.ndarray) -> float:
    log_values = _positive_log10(values)
    if log_values.size == 0:
        return np.nan
    q16, q84 = np.quantile(log_values, [0.16, 0.84])
    return float(0.5 * (q84 - q16))


def ks_distance(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.sort(values_a[np.isfinite(values_a) & (values_a > 0)])
    b = np.sort(values_b[np.isfinite(values_b) & (values_b > 0)])
    if a.size == 0 or b.size == 0:
        return np.nan
    support = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, support, side="right") / a.size
    cdf_b = np.searchsorted(b, support, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def alpha_scan(
    sim_values: np.ndarray,
    real_values: np.ndarray,
    study_cfg: ChargeStudyConfig,
) -> tuple[float, float, float, np.ndarray]:
    log_sim = _positive_log10(sim_values)
    log_real = _positive_log10(real_values)
    if log_sim.size == 0 or log_real.size == 0:
        return np.nan, np.nan, np.nan, np.array([], dtype=float)

    alphas = np.linspace(
        study_cfg.alpha_min,
        study_cfg.alpha_max,
        study_cfg.alpha_grid_points,
    )
    log_shift_min = np.log10(study_cfg.alpha_min)
    log_shift_max = np.log10(study_cfg.alpha_max)
    left = float(min(log_real.min(), log_sim.min() + log_shift_min))
    right = float(max(log_real.max(), log_sim.max() + log_shift_max))
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        right = left + 1.0
    edges = np.linspace(left, right, study_cfg.hist_bin_count + 1)
    hist_real, _ = np.histogram(log_real, bins=edges, density=True)

    best_alpha = np.nan
    best_mse = np.inf
    mse_original = np.nan
    for alpha in alphas:
        hist_sim, _ = np.histogram(log_sim + np.log10(alpha), bins=edges, density=True)
        mse = float(np.mean((hist_sim - hist_real) ** 2))
        if np.isclose(alpha, 1.0):
            mse_original = mse
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)

    if np.isnan(mse_original):
        hist_sim, _ = np.histogram(log_sim, bins=edges, density=True)
        mse_original = float(np.mean((hist_sim - hist_real) ** 2))

    return best_alpha, mse_original, best_mse, edges


def summarize_charge_spectrum(
    matches_df: pd.DataFrame,
    study_cfg: ChargeStudyConfig,
    unit_key: str,
    calibration: TotChargeCalibration,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray | float]]]:
    rows: list[dict[str, float | int]] = []
    plot_payload: dict[int, dict[str, np.ndarray | float]] = {}

    for plane in range(1, 5):
        plane_matches = matches_df[matches_df["plane"] == plane].copy()
        sim_values, real_values = pooled_plane_samples(
            matches_df,
            plane,
            study_cfg,
            unit_key,
            calibration,
        )
        best_alpha, mse_original, mse_scaled, hist_edges = alpha_scan(
            sim_values,
            real_values,
            study_cfg,
        )
        sim_sigma = robust_log_sigma(sim_values)
        real_sigma = robust_log_sigma(real_values)
        ks_original = ks_distance(sim_values, real_values)
        ks_scaled = ks_distance(sim_values * best_alpha, real_values) if np.isfinite(best_alpha) else np.nan

        rows.append(
            {
                "plane": plane,
                "available_sim_files": int(plane_matches["sim_pool_size"].iloc[0]) if not plane_matches.empty and "sim_pool_size" in plane_matches.columns else 0,
                "available_real_files": int(plane_matches["real_pool_size"].iloc[0]) if not plane_matches.empty and "real_pool_size" in plane_matches.columns else 0,
                "target_pairs": int(plane_matches["target_pairs"].iloc[0]) if not plane_matches.empty and "target_pairs" in plane_matches.columns else 0,
                "matched_pairs": int(len(plane_matches)),
                "sim_samples": int(sim_values.size),
                "real_samples": int(real_values.size),
                "median_sim_empirical_efficiency": float(plane_matches["sim_empirical_efficiency"].median()) if not plane_matches.empty else np.nan,
                "median_real_empirical_efficiency": float(plane_matches["real_empirical_efficiency"].median()) if not plane_matches.empty else np.nan,
                "median_abs_empirical_efficiency_delta": float(plane_matches["abs_empirical_efficiency_delta"].median()) if not plane_matches.empty else np.nan,
                "strict_pairs": int((plane_matches["match_mode"] == "within_threshold").sum()) if "match_mode" in plane_matches.columns else 0,
                "fallback_pairs": int((plane_matches["match_mode"] == "fallback_nearest").sum()) if "match_mode" in plane_matches.columns else 0,
                "forced_pairs": int((plane_matches["match_mode"] == "forced_nearest").sum()) if "match_mode" in plane_matches.columns else 0,
                "recommended_charge_scale_alpha": best_alpha,
                "hist_mse_original": mse_original,
                "hist_mse_scaled": mse_scaled,
                "ks_original": ks_original,
                "ks_scaled": ks_scaled,
                "sim_log10_charge_sigma": sim_sigma,
                "real_log10_charge_sigma": real_sigma,
                "real_over_sim_log10_charge_sigma_ratio": (
                    float(real_sigma / sim_sigma)
                    if np.isfinite(real_sigma) and np.isfinite(sim_sigma) and sim_sigma > 0
                    else np.nan
                ),
                "match_coverage_fraction": (
                    float(len(plane_matches) / max(1, int(plane_matches["target_pairs"].iloc[0])))
                    if not plane_matches.empty
                    else np.nan
                ),
                "charge_unit": unit_key,
            }
        )

        plot_payload[plane] = {
            "sim_values": sim_values,
            "real_values": real_values,
            "alpha": best_alpha,
            "hist_edges": hist_edges,
        }

    return pd.DataFrame(rows), plot_payload


def plot_linear_charge_overlay(
    summary_df: pd.DataFrame,
    plot_payload: dict[int, dict[str, np.ndarray | float]],
    unit_key: str,
) -> None:
    out = output_dir()
    unit_label = UNIT_LABELS[unit_key]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        payload = plot_payload.get(plane, {})
        sim_values = np.asarray(payload.get("sim_values", np.array([], dtype=float)))
        real_values = np.asarray(payload.get("real_values", np.array([], dtype=float)))
        alpha = float(payload.get("alpha", np.nan))
        scaled_values = sim_values * alpha if np.isfinite(alpha) else np.array([], dtype=float)
        pooled = np.concatenate(
            [
                arr
                for arr in (sim_values, real_values, scaled_values)
                if np.asarray(arr).size
            ]
        ) if (sim_values.size or real_values.size or scaled_values.size) else np.array([], dtype=float)
        if pooled.size:
            upper = float(np.quantile(pooled[np.isfinite(pooled) & (pooled > 0)], 0.995))
            if not np.isfinite(upper) or upper <= 0:
                upper = float(np.nanmax(pooled))
            if np.isfinite(upper) and upper > 0:
                edges = np.linspace(0.0, upper, 80)
                if sim_values.size:
                    ax.hist(sim_values, bins=edges, density=True, histtype="step", linewidth=1.6, color=GROUP_COLORS["SIM"], label="MINGO00")
                if real_values.size:
                    ax.hist(real_values, bins=edges, density=True, histtype="step", linewidth=1.6, color=GROUP_COLORS["REAL"], label="MINGO01")
                if scaled_values.size:
                    ax.hist(scaled_values, bins=edges, density=True, histtype="step", linewidth=1.4, linestyle="--", color=GROUP_COLORS["SCALED"], label=f"scaled sim x{alpha:.3f}")
        row = summary_df[summary_df["plane"] == plane].iloc[0]
        ax.set_title(
            f"Plane {plane} | alpha={row['recommended_charge_scale_alpha']:.3f}\n"
            f"emp. eff delta={row['median_abs_empirical_efficiency_delta']:.3f}",
            fontsize=10,
        )
        ax.grid(alpha=0.2)
        ax.set_xlabel(f"Plane charge [{unit_label}]")
        if plane in (1, 3):
            ax.set_ylabel("Density")
    _global_legend(fig, axes)
    fig.suptitle(
        f"Linear Task 3 plane-charge overlay in {unit_label} matched by metadata empirical efficiency",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"charge_spectrum_linear_overlay_{unit_key}.png", dpi=180)
    fig.savefig(out / f"charge_spectrum_linear_overlay_{unit_key}.pdf")
    plt.close(fig)


def plot_log_charge_overlay(
    summary_df: pd.DataFrame,
    plot_payload: dict[int, dict[str, np.ndarray | float]],
    unit_key: str,
) -> None:
    out = output_dir()
    unit_label = UNIT_LABELS[unit_key]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        payload = plot_payload.get(plane, {})
        sim_values = np.asarray(payload.get("sim_values", np.array([], dtype=float)))
        real_values = np.asarray(payload.get("real_values", np.array([], dtype=float)))
        alpha = float(payload.get("alpha", np.nan))
        edges = np.asarray(payload.get("hist_edges", np.array([], dtype=float)))
        if sim_values.size and edges.size:
            ax.hist(
                np.log10(sim_values),
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=GROUP_COLORS["SIM"],
                label="MINGO00",
            )
        if real_values.size and edges.size:
            ax.hist(
                np.log10(real_values),
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=GROUP_COLORS["REAL"],
                label="MINGO01",
            )
        if sim_values.size and edges.size and np.isfinite(alpha):
            ax.hist(
                np.log10(sim_values * alpha),
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.4,
                linestyle="--",
                color=GROUP_COLORS["SCALED"],
                label=f"scaled sim x{alpha:.3f}",
            )
        row = summary_df[summary_df["plane"] == plane].iloc[0]
        ax.set_title(
            f"Plane {plane} | alpha={row['recommended_charge_scale_alpha']:.3f}\n"
            f"shape ratio={row['real_over_sim_log10_charge_sigma_ratio']:.3f}",
            fontsize=10,
        )
        ax.grid(alpha=0.2)
        ax.set_xlabel(f"log10(plane charge [{unit_label}])")
        if plane in (1, 3):
            ax.set_ylabel("Density")
    _global_legend(fig, axes)
    fig.suptitle(
        f"Log-charge overlay in {unit_label} matched by metadata empirical efficiency",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"charge_spectrum_log_overlay_{unit_key}.png", dpi=180)
    fig.savefig(out / f"charge_spectrum_log_overlay_{unit_key}.pdf")
    plt.close(fig)


def _cdf_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean = np.sort(values[np.isfinite(values) & (values > 0)])
    if clean.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    y = np.arange(1, clean.size + 1, dtype=float) / clean.size
    return clean, y


def plot_charge_cdf_overlay(
    summary_df: pd.DataFrame,
    plot_payload: dict[int, dict[str, np.ndarray | float]],
    unit_key: str,
) -> None:
    out = output_dir()
    unit_label = UNIT_LABELS[unit_key]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        payload = plot_payload.get(plane, {})
        sim_values = np.asarray(payload.get("sim_values", np.array([], dtype=float)))
        real_values = np.asarray(payload.get("real_values", np.array([], dtype=float)))
        alpha = float(payload.get("alpha", np.nan))
        for label, values, color, linestyle in (
            ("MINGO00", sim_values, GROUP_COLORS["SIM"], "-"),
            ("MINGO01", real_values, GROUP_COLORS["REAL"], "-"),
            ("scaled sim", sim_values * alpha if np.isfinite(alpha) else np.array([], dtype=float), GROUP_COLORS["SCALED"], "--"),
        ):
            x, y = _cdf_xy(values)
            if x.size:
                ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.4, label=label)
        row = summary_df[summary_df["plane"] == plane].iloc[0]
        ax.set_title(
            f"Plane {plane} | KS {row['ks_original']:.3f} -> {row['ks_scaled']:.3f}",
            fontsize=10,
        )
        ax.grid(alpha=0.2)
        ax.set_xlabel(f"Plane charge [{unit_label}]")
        if plane in (1, 3):
            ax.set_ylabel("CDF")
    _global_legend(fig, axes)
    fig.suptitle(f"Charge-spectrum CDF overlay in {unit_label}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"charge_spectrum_cdf_overlay_{unit_key}.png", dpi=180)
    fig.savefig(out / f"charge_spectrum_cdf_overlay_{unit_key}.pdf")
    plt.close(fig)


def plot_centered_shape_overlay(
    summary_df: pd.DataFrame,
    plot_payload: dict[int, dict[str, np.ndarray | float]],
    unit_key: str,
) -> None:
    out = output_dir()
    unit_label = UNIT_LABELS[unit_key]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        payload = plot_payload.get(plane, {})
        sim_values = _positive_log10(np.asarray(payload.get("sim_values", np.array([], dtype=float))))
        real_values = _positive_log10(np.asarray(payload.get("real_values", np.array([], dtype=float))))
        if sim_values.size:
            sim_centered = sim_values - np.median(sim_values)
            ax.hist(
                sim_centered,
                bins=70,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=GROUP_COLORS["SIM"],
                label="MINGO00 centered",
            )
        if real_values.size:
            real_centered = real_values - np.median(real_values)
            ax.hist(
                real_centered,
                bins=70,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=GROUP_COLORS["REAL"],
                label="MINGO01 centered",
            )
        row = summary_df[summary_df["plane"] == plane].iloc[0]
        ax.set_title(
            f"Plane {plane} | width ratio={row['real_over_sim_log10_charge_sigma_ratio']:.3f}",
            fontsize=10,
        )
        ax.grid(alpha=0.2)
        ax.set_xlabel(f"log10(charge [{unit_label}]) - median")
        if plane in (1, 3):
            ax.set_ylabel("Density")
    _global_legend(fig, axes)
    fig.suptitle(
        f"Centered log-charge shape comparison in {unit_label} after removing the scale shift",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"charge_spectrum_centered_shape_overlay_{unit_key}.png", dpi=180)
    fig.savefig(out / f"charge_spectrum_centered_shape_overlay_{unit_key}.pdf")
    plt.close(fig)


def plot_efficiency_match_quality(matches_df: pd.DataFrame) -> None:
    out = output_dir()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        plane_df = matches_df[matches_df["plane"] == plane].copy()
        if not plane_df.empty:
            color_map = {
                "within_threshold": "#4c72b0",
                "fallback_nearest": "#dd8452",
                "forced_nearest": "#c44e52",
            }
            point_colors = [color_map.get(mode, "#4c72b0") for mode in plane_df.get("match_mode", pd.Series([], dtype=str))]
            ax.scatter(
                plane_df["sim_empirical_efficiency"],
                plane_df["real_empirical_efficiency"],
                s=18,
                alpha=0.7,
                color=point_colors,
            )
            low = float(
                min(
                    plane_df["sim_empirical_efficiency"].min(),
                    plane_df["real_empirical_efficiency"].min(),
                )
            )
            high = float(
                max(
                    plane_df["sim_empirical_efficiency"].max(),
                    plane_df["real_empirical_efficiency"].max(),
                )
            )
            ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.0)
        ax.set_title(f"Plane {plane}", fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xlabel("MINGO00 empirical efficiency")
        if plane in (1, 3):
            ax.set_ylabel("MINGO01 empirical efficiency")
    fig.suptitle("Matched-file quality by metadata empirical efficiency", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "charge_spectrum_efficiency_match_quality.png", dpi=180)
    fig.savefig(out / "charge_spectrum_efficiency_match_quality.pdf")
    plt.close(fig)


def plot_summary_bars(summary_df: pd.DataFrame, unit_key: str) -> None:
    out = output_dir()
    unit_label = UNIT_LABELS[unit_key]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    planes = summary_df["plane"].to_numpy(dtype=int)

    axes[0].bar(planes, summary_df["recommended_charge_scale_alpha"], color="#4c72b0")
    axes[0].axhline(1.0, linestyle="--", color="red", linewidth=1.0)
    axes[0].set_title("Recommended charge-scale alpha")
    axes[0].set_xlabel("Plane")
    axes[0].set_ylabel("alpha")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(planes, summary_df["real_over_sim_log10_charge_sigma_ratio"], color="#55a868")
    axes[1].axhline(1.0, linestyle="--", color="red", linewidth=1.0)
    axes[1].set_title("Real / sim shape-width ratio")
    axes[1].set_xlabel("Plane")
    axes[1].set_ylabel("ratio")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(f"Charge-spectrum tuning summary in {unit_label}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"charge_spectrum_summary_bars_{unit_key}.png", dpi=180)
    fig.savefig(out / f"charge_spectrum_summary_bars_{unit_key}.pdf")
    plt.close(fig)


def write_report(
    selection,
    study_cfg: ChargeStudyConfig,
    current_model: CurrentChargeModel,
    matches_df: pd.DataFrame,
    summary_by_unit: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = output_dir()
    summary_rows = [
        {"metric": "current_step3_townsend_alpha_per_mm", "value": current_model.townsend_alpha_per_mm},
        {"metric": "current_step3_avalanche_gap_mm_fixed", "value": current_model.avalanche_gap_mm},
        {"metric": "current_step3_avalanche_electron_sigma", "value": current_model.avalanche_electron_sigma},
        {"metric": "current_step4_lorentzian_gamma_mm", "value": current_model.lorentzian_gamma_mm},
        {"metric": "current_step4_induced_charge_fraction", "value": current_model.induced_charge_fraction},
        {"metric": "current_step5_qdiff_width", "value": current_model.qdiff_width},
        {"metric": "current_step8_q_to_time_factor", "value": current_model.q_to_time_factor},
        {"metric": "current_step8_charge_threshold", "value": current_model.charge_threshold},
        {"metric": "total_matched_pairs", "value": float(len(matches_df))},
    ]
    report_sections: list[str] = []
    for unit_key, summary_df in summary_by_unit.items():
        recommended_alpha = float(
            np.nanmedian(summary_df["recommended_charge_scale_alpha"].to_numpy(dtype=float))
        )
        recommended_shape_ratio = float(
            np.nanmedian(summary_df["real_over_sim_log10_charge_sigma_ratio"].to_numpy(dtype=float))
        )
        fallback_pairs = int(summary_df["fallback_pairs"].sum()) if "fallback_pairs" in summary_df.columns else 0
        forced_pairs = int(summary_df["forced_pairs"].sum()) if "forced_pairs" in summary_df.columns else 0
        summary_rows.extend(
            [
                {"metric": f"median_recommended_charge_scale_alpha_{unit_key}", "value": recommended_alpha},
                {"metric": f"median_real_over_sim_log10_charge_sigma_ratio_{unit_key}", "value": recommended_shape_ratio},
                {"metric": f"fallback_pairs_{unit_key}", "value": float(fallback_pairs)},
                {"metric": f"forced_pairs_{unit_key}", "value": float(forced_pairs)},
            ]
        )

        per_plane_lines = []
        for row in summary_df.itertuples(index=False):
            if int(row.matched_pairs) == 0:
                status = "no matches in the selected data"
            elif int(row.matched_pairs) < 3:
                status = "weak statistics"
            elif int(getattr(row, "forced_pairs", 0)) > 0:
                status = "ok (includes forced nearest matches)"
            elif int(getattr(row, "fallback_pairs", 0)) > 0:
                status = "ok (includes nearest-neighbour fallback)"
            else:
                status = "ok"
            per_plane_lines.append(
                f"- Plane {row.plane}: alpha={row.recommended_charge_scale_alpha:.3f}, "
                f"shape ratio={row.real_over_sim_log10_charge_sigma_ratio:.3f}, "
                f"pairs={int(row.matched_pairs)}, "
                f"pool={int(getattr(row, 'available_sim_files', 0))} sim / {int(getattr(row, 'available_real_files', 0))} real, "
                f"target={int(getattr(row, 'target_pairs', 0))}, "
                f"strict={int(getattr(row, 'strict_pairs', 0))}, "
                f"median |emp. eff delta|={row.median_abs_empirical_efficiency_delta:.4f}, "
                f"fallback={int(getattr(row, 'fallback_pairs', 0))}, "
                f"forced={int(getattr(row, 'forced_pairs', 0))}, "
                f"coverage={float(getattr(row, 'match_coverage_fraction', np.nan)):.3f}, "
                f"status={status}"
            )

        report_sections.append(
            f"""{UNIT_LABELS[unit_key]} view:
- Median charge-scale alpha across planes: {recommended_alpha:.3f}
- Median real/sim log-charge width ratio across planes: {recommended_shape_ratio:.3f}
- Fallback nearest-neighbour pairs used: {fallback_pairs}
- Forced nearest pairs used to reach minimum statistics: {forced_pairs}

Plane-by-plane summary:
{chr(10).join(per_plane_lines)}
"""
        )
    summary_export = pd.DataFrame(summary_rows)
    summary_export.to_csv(out / "recommended_charge_spectrum_summary.csv", index=False)

    report = f"""Charge-spectrum tuning from Task 3 plane charge
==================================================

Simulation stations:
- {", ".join(selection.simulation_stations)}

Real-data stations:
- {", ".join(selection.real_stations)}

Selection settings:
- multiplicity_mode={study_cfg.multiplicity_mode}
- efficiency_match_max_abs_difference={study_cfg.efficiency_match_max_abs_difference:.3f}
- efficiency_match_fallback_max_abs_difference={study_cfg.efficiency_match_fallback_max_abs_difference:.3f}
- max_pairs_per_plane={study_cfg.max_pairs_per_plane}
- min_pairs_per_plane={study_cfg.min_pairs_per_plane}
- sample_entries_per_file={study_cfg.sample_entries_per_file}
- min_positive_charge_ns={study_cfg.min_positive_charge_ns:.3f}

Current digital-twin charge-model knobs:
- STEP 3 townsend_alpha_per_mm={current_model.townsend_alpha_per_mm:.6g}
- STEP 3 avalanche_gap_mm={current_model.avalanche_gap_mm:.6g} (fixed, not a tuning knob)
- STEP 3 avalanche_electron_sigma={current_model.avalanche_electron_sigma:.6g}
- STEP 4 lorentzian_gamma_mm={current_model.lorentzian_gamma_mm:.6g}
- STEP 4 induced_charge_fraction={current_model.induced_charge_fraction:.6g}
- STEP 5 qdiff_width={current_model.qdiff_width:.6g}
- STEP 8 charge_conversion_model={current_model.charge_conversion_model}
- STEP 8 q_to_time_factor={current_model.q_to_time_factor:.6g}
- STEP 8 charge_threshold={current_model.charge_threshold:.6g}

Recommendations from matched Task 3 plane-charge spectra:
{chr(10).join(report_sections)}

Interpretation:
- File matching uses only metadata-derived empirical efficiencies from Task 3 trigger-rate metadata for both MINGO00 and MINGO01.
- The matcher first takes unique pairs within the strict efficiency window, then fills with nearest-neighbour fallback pairs when needed, and only forces nearest pairs if that is required to avoid empty or near-empty planes.
- The effective minimum number of pairs is capped by the number of unique sim/real files actually available for that plane, so the report reflects the real evidence size instead of an impossible target.
- The primary quantity read from the listed Task 3 output is `P*_Q_sum_final`, which is treated here as the detector-side charge width in ns.
- The ancillary `fC` view is derived afterward by applying the same forward TOT calibration used in the real pipeline.
- With `multiplicity_mode=single_strip_only`, only Task 3 plane entries with exactly one active strip are used, based on `active_strips_P*`.
- The `log10(charge) - median` plot is made by taking those positive plane charges, applying `log10`, then centering each group by its own median. Any left bump there is therefore a real low-charge subpopulation in that matched sample, not a fit artifact.
- The scale alpha multiplies the simulated charge to best match real-data shape in log-charge space.
- Alpha below 1 means the simulation charge scale is too high; above 1 means it is too low.
- The shape ratio compares robust log-charge widths after removing the scale shift.
- A shape ratio above 1 suggests the real avalanche-charge spectrum is broader than the simulated one,
  which points first to STEP 3 fluctuation controls such as `avalanche_electron_sigma`.
- The simulated total avalanche charge is currently generated as:
  `avalanche_size_electrons = ions * exp(townsend_alpha_per_mm * avalanche_gap_mm) * lognormal(sigma=avalanche_electron_sigma)`.
- STEP 4 now converts that avalanche size into gap charge in `fC`, then applies
  `induced_charge_fraction` to get the total induced readout charge before strip sharing.
- After that, STEP 4 redistributes the induced charge with an isotropic 2D Lorentzian whose width
  parameter is `lorentzian_gamma_mm`.
- STEP 8 then converts the induced `fC` to detector-side width using the configured
  `charge_conversion_model`, which is now the physically relevant FEE-side charge knob.
- The final detector-side width scale seen here is therefore mainly controlled by
  `townsend_alpha_per_mm`, `avalanche_electron_sigma`, `induced_charge_fraction`,
  `lorentzian_gamma_mm`, and the downstream STEP 8 conversion / threshold.

This study is advisory-only: it does not modify the simulation config files.
"""
    (out / "charge_spectrum_report.txt").write_text(report, encoding="utf-8")
    return summary_export


def main() -> None:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    config = load_tuning_config()
    selection = resolve_selection(config)
    study_cfg = load_charge_study_config(config)
    current_model = load_current_charge_model()
    calibration = TotChargeCalibration.from_csv(default_tot_charge_calibration_path(ROOT_DIR))

    sim_metadata = load_latest_trigger_metadata(
        selection.simulation_stations,
        selection.simulation_date_ranges,
    )
    real_metadata = load_latest_trigger_metadata(
        selection.real_stations,
        selection.real_date_ranges,
    )
    if sim_metadata.empty or real_metadata.empty:
        raise SystemExit("No metadata with matching listed parquets found for the selected stations.")

    matches_df = select_plane_matches(sim_metadata, real_metadata, study_cfg)
    matches_df.to_csv(out / "matched_charge_spectrum_pairs.csv", index=False)
    if matches_df.empty:
        raise SystemExit("No matched files found within the configured efficiency tolerance.")

    plot_efficiency_match_quality(matches_df)
    summary_by_unit: dict[str, pd.DataFrame] = {}
    for unit_key in ("ns", "fc"):
        summary_df, plot_payload = summarize_charge_spectrum(
            matches_df,
            study_cfg,
            unit_key,
            calibration,
        )
        summary_df.to_csv(out / f"per_plane_charge_spectrum_summary_{unit_key}.csv", index=False)
        plot_linear_charge_overlay(summary_df, plot_payload, unit_key)
        plot_log_charge_overlay(summary_df, plot_payload, unit_key)
        plot_charge_cdf_overlay(summary_df, plot_payload, unit_key)
        plot_centered_shape_overlay(summary_df, plot_payload, unit_key)
        plot_summary_bars(summary_df, unit_key)
        summary_by_unit[unit_key] = summary_df

    write_report(selection, study_cfg, current_model, matches_df, summary_by_unit)


if __name__ == "__main__":
    main()
