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

from MASTER.ANCILLARY.SIMULATION_TUNING.common import date_range_mask, load_tuning_config, resolve_selection
from MASTER.common.file_selection import extract_run_datetime_from_name
from MASTER.common.tot_charge_calibration import TotChargeCalibration, default_tot_charge_calibration_path


PLANE_QSUM_FINAL = {plane: f"P{plane}_Q_sum_final" for plane in range(1, 5)}
PLANE_ACTIVE_STRIPS = {plane: f"active_strips_P{plane}" for plane in range(1, 5)}
PLANE_NO_TRIGGER = {1: "234", 2: "134", 3: "124", 4: "123"}
STREAMER_PERCENT_COLUMNS = {plane: f"streamer_P{plane}" for plane in range(1, 5)}
Z_POSITION_COLUMNS = {plane: f"z_P{plane}" for plane in range(1, 5)}
GROUP_COLORS = {
    "SIM": "#d95f02",
    "REAL": "#1b9e77",
    "SCALED": "#7570b3",
}
UNIT_LABELS = {"ns": "ns", "fc": "fC"}
DEFAULT_SEED = 20260331
SAMPLE_CACHE: dict[tuple[str, int, str, int, float], np.ndarray] = {}
STREAMER_CACHE: dict[tuple[str, float], dict[int, float]] = {}
Z_CONFIG_CACHE: dict[str, dict[int, float]] = {}


@dataclass(frozen=True)
class ChargeStudyConfig:
    simulation_station: str
    real_station: str
    simulation_basename: str | None
    real_basename: str | None
    multiplicity_mode: str
    streamer_threshold_qsum: float
    streamer_threshold_source: str
    real_streamer_min_percent: float
    real_streamer_selector: str
    max_per_plane_efficiency_abs_difference: float
    max_mean_efficiency_abs_difference: float
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


def clear_previous_outputs(out: Path) -> None:
    filenames = [
        "charge_spectrum_real_file_screening.csv",
        "charge_spectrum_pair_candidates.csv",
        "charge_spectrum_nearest_pair_candidates.csv",
        "charge_spectrum_selected_pair.csv",
        "selected_pair_charge_spectrum_summary_ns.csv",
        "selected_pair_charge_spectrum_summary_fc.csv",
        "selected_pair_charge_spectrum_report.txt",
        "selected_pair_plane_by_plane_overlay.png",
        "selected_pair_plane_by_plane_overlay_ns.png",
        "selected_pair_plane_by_plane_overlay_fc.png",
        "selected_pair_charge_overlay_linear_ns.png",
        "selected_pair_charge_overlay_linear_ns.pdf",
        "selected_pair_charge_overlay_linear_fc.png",
        "selected_pair_charge_overlay_linear_fc.pdf",
        "selected_pair_charge_overlay_log_ns.png",
        "selected_pair_charge_overlay_log_ns.pdf",
        "selected_pair_charge_overlay_log_fc.png",
        "selected_pair_charge_overlay_log_fc.pdf",
    ]
    for filename in filenames:
        path = out / filename
        if path.exists():
            path.unlink()


def _normalize_station_label(raw: object) -> str:
    if isinstance(raw, int):
        return f"MINGO{int(raw):02d}"
    text = str(raw).strip().upper()
    if not text:
        raise ValueError("Empty station token in charge_spectrum config.")
    if text.isdigit():
        return f"MINGO{int(text):02d}"
    if text.startswith("MINGO") and text[5:].isdigit():
        return f"MINGO{int(text[5:]):02d}"
    raise ValueError(f"Unsupported station token in charge_spectrum config: {raw!r}")


def _normalize_basename(raw: object | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("listed_"):
        text = text[len("listed_") :]
    if text.endswith(".parquet"):
        text = text[: -len(".parquet")]
    return text or None


def _normalize_streamer_selector(raw: object) -> str:
    text = str(raw).strip().lower()
    aliases = {
        "max": "max_plane",
        "any_plane": "max_plane",
        "mean": "mean_plane",
        "avg": "mean_plane",
        "all_planes": "min_plane",
        "min": "min_plane",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"max_plane", "mean_plane", "min_plane"}:
        raise ValueError(
            "charge_spectrum.real_streamer_selector must be one of "
            "'max_plane', 'mean_plane', or 'min_plane'."
        )
    return normalized


def _scalar_to_float(value: object) -> float:
    try:
        return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])
    except Exception:
        return float("nan")


def _safe_timestamp(value: object) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    try:
        return pd.Timestamp(value)
    except Exception:
        return pd.NaT


def task3_config_path() -> Path:
    return (
        ROOT_DIR
        / "MASTER"
        / "CONFIG_FILES"
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_3"
        / "config_task_3.yaml"
    )


def load_default_streamer_threshold_qsum() -> tuple[float, str]:
    path = task3_config_path()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    threshold = _scalar_to_float(config.get("streamer_charge_sum_threshold"))
    if not np.isfinite(threshold):
        raise ValueError(
            "Task 3 streamer_charge_sum_threshold is not set, and "
            "charge_spectrum.streamer_threshold_qsum was not provided."
        )
    return threshold, f"{path}:streamer_charge_sum_threshold"


def load_charge_study_config(config: dict, selection) -> ChargeStudyConfig:
    study_cfg = config.get("charge_spectrum", {})
    multiplicity_mode = str(study_cfg.get("multiplicity_mode", "single_strip_only")).strip().lower()
    if multiplicity_mode not in {"single_strip_only", "all_active"}:
        raise ValueError(
            "charge_spectrum.multiplicity_mode must be 'single_strip_only' or 'all_active'."
        )

    threshold_raw = study_cfg.get("streamer_threshold_qsum")
    if threshold_raw is None:
        streamer_threshold_qsum, streamer_threshold_source = load_default_streamer_threshold_qsum()
    else:
        streamer_threshold_qsum = float(threshold_raw)
        streamer_threshold_source = "charge_spectrum.streamer_threshold_qsum"
    if not np.isfinite(streamer_threshold_qsum) or streamer_threshold_qsum <= 0:
        raise ValueError("charge_spectrum.streamer_threshold_qsum must be a positive number.")

    simulation_station = _normalize_station_label(
        study_cfg.get("simulation_station", selection.simulation_stations[0])
    )
    real_station = _normalize_station_label(study_cfg.get("real_station", selection.real_stations[0]))

    alpha_min = float(study_cfg.get("alpha_min", 0.20))
    alpha_max = float(study_cfg.get("alpha_max", 2.00))
    alpha_grid_points = int(study_cfg.get("alpha_grid_points", 161))
    if alpha_min <= 0 or alpha_max <= alpha_min:
        raise ValueError("charge_spectrum alpha range must satisfy 0 < alpha_min < alpha_max.")
    if alpha_grid_points < 3:
        raise ValueError("charge_spectrum.alpha_grid_points must be at least 3.")

    return ChargeStudyConfig(
        simulation_station=simulation_station,
        real_station=real_station,
        simulation_basename=_normalize_basename(study_cfg.get("simulation_basename")),
        real_basename=_normalize_basename(study_cfg.get("real_basename")),
        multiplicity_mode=multiplicity_mode,
        streamer_threshold_qsum=streamer_threshold_qsum,
        streamer_threshold_source=streamer_threshold_source,
        real_streamer_min_percent=float(study_cfg.get("real_streamer_min_percent", 5.0)),
        real_streamer_selector=_normalize_streamer_selector(
            study_cfg.get("real_streamer_selector", "max_plane")
        ),
        max_per_plane_efficiency_abs_difference=float(
            study_cfg.get("max_per_plane_efficiency_abs_difference", 0.05)
        ),
        max_mean_efficiency_abs_difference=float(
            study_cfg.get("max_mean_efficiency_abs_difference", 0.03)
        ),
        sample_entries_per_file=int(study_cfg.get("sample_entries_per_file", 1500)),
        min_positive_charge_ns=float(study_cfg.get("min_positive_charge_ns", 0.05)),
        hist_bin_count=int(study_cfg.get("hist_bin_count", 90)),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_grid_points=alpha_grid_points,
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
            step4_cfg.get("lorentzian_gamma_mm", 0.5 * float(step4_cfg.get("avalanche_width_mm", np.nan)))
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


def station_specific_metadata_path(station_label: str) -> Path:
    return (
        ROOT_DIR
        / "STATIONS"
        / station_label
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_3"
        / "METADATA"
        / "task_3_metadata_specific.csv"
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


def collect_listed_file_entries(station_label: str) -> dict[str, Path]:
    files_by_key: dict[str, Path] = {}
    for input_dir in station_listed_input_dirs(station_label):
        if not input_dir.exists():
            continue
        for parquet_path in sorted(input_dir.glob("listed_*.parquet")):
            files_by_key[_basename_from_listed_path(parquet_path)] = parquet_path
    return files_by_key


def resolve_listed_file_path(station_label: str, basename: str, preferred_path: str | Path) -> Path:
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


def _dedupe_latest_by_filename(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "filename_base" not in frame.columns:
        return frame.iloc[0:0].copy()
    copy = frame.copy()
    copy["execution_ts"] = pd.to_datetime(
        copy.get("execution_timestamp"),
        format="%Y-%m-%d_%H.%M.%S",
        errors="coerce",
    )
    copy.sort_values(["filename_base", "execution_ts"], inplace=True)
    return copy.drop_duplicates(subset=["filename_base"], keep="last")


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


def load_station_metadata(
    station_label: str,
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> pd.DataFrame:
    path_lookup = collect_listed_file_entries(station_label)
    trigger_path = station_trigger_metadata_path(station_label)
    if not trigger_path.exists():
        return pd.DataFrame()

    trigger = pd.read_csv(trigger_path)
    trigger = _dedupe_latest_by_filename(trigger)
    if trigger.empty:
        return trigger

    trigger = trigger.copy()
    trigger["station_label"] = station_label
    trigger["datetime"] = trigger["filename_base"].map(extract_run_datetime_from_name)
    if date_ranges:
        trigger = trigger.loc[date_range_mask(trigger["datetime"], date_ranges)].copy()
    if trigger.empty:
        return trigger
    trigger["parquet_path"] = trigger["filename_base"].map(path_lookup.get)
    trigger = trigger[trigger["parquet_path"].notna()].copy()
    if trigger.empty:
        return trigger

    base_rate = _preferred_rate(trigger, "1234")
    trigger["empirical_rate_1234"] = base_rate
    for plane, tt_label in PLANE_NO_TRIGGER.items():
        missing_rate = _preferred_rate(trigger, tt_label)
        efficiency = 1.0 - (missing_rate / base_rate)
        trigger[f"empirical_eff_plane_{plane}"] = efficiency.where(
            np.isfinite(efficiency) & np.isfinite(base_rate) & (base_rate > 0)
        )

    specific_path = station_specific_metadata_path(station_label)
    if not specific_path.exists():
        return trigger

    specific = pd.read_csv(specific_path)
    specific = _dedupe_latest_by_filename(specific)
    if specific.empty:
        return trigger

    keep_columns = ["filename_base", "charge_topology_streamer_threshold_qsum"]
    keep_columns.extend(column for column in STREAMER_PERCENT_COLUMNS.values() if column in specific.columns)
    keep_columns.extend(column for column in Z_POSITION_COLUMNS.values() if column in specific.columns)
    specific = specific[[column for column in keep_columns if column in specific.columns]].copy()
    return trigger.merge(specific, on="filename_base", how="left")


def filter_metadata_by_basename(frame: pd.DataFrame, basename: str | None, *, label: str) -> pd.DataFrame:
    if basename is None:
        return frame.copy()
    filtered = frame[frame["filename_base"].astype(str) == basename].copy()
    if filtered.empty:
        raise SystemExit(f"Configured {label}_basename={basename!r} was not found in the selected metadata.")
    return filtered


def _streamer_selector_value(percentages: dict[int, float], selector: str) -> float:
    values = np.array([percentages[plane] for plane in range(1, 5)], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    if selector == "max_plane":
        return float(np.max(finite))
    if selector == "mean_plane":
        return float(np.mean(finite))
    if selector == "min_plane":
        return float(np.min(finite))
    raise ValueError(f"Unsupported streamer selector: {selector}")


def compute_streamer_percentages_from_parquet(parquet_path: str | Path, threshold_qsum: float) -> dict[int, float]:
    cache_key = (str(parquet_path), float(threshold_qsum))
    cached = STREAMER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    frame = pd.read_parquet(Path(parquet_path), columns=list(PLANE_QSUM_FINAL.values()))
    total_events = len(frame.index)
    result: dict[int, float] = {}
    for plane, column in PLANE_QSUM_FINAL.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if total_events <= 0:
            result[plane] = np.nan
        else:
            streamer_count = int(np.count_nonzero(np.isfinite(values) & (values > threshold_qsum)))
            result[plane] = float(100.0 * streamer_count / total_events)
    STREAMER_CACHE[cache_key] = result
    return result


def resolve_streamer_percentages(row: pd.Series, study_cfg: ChargeStudyConfig) -> tuple[dict[int, float], str]:
    metadata_threshold = _scalar_to_float(row.get("charge_topology_streamer_threshold_qsum"))
    metadata_values = {
        plane: _scalar_to_float(row.get(STREAMER_PERCENT_COLUMNS[plane]))
        for plane in range(1, 5)
    }
    if (
        np.isfinite(metadata_threshold)
        and np.isclose(metadata_threshold, study_cfg.streamer_threshold_qsum, rtol=0.0, atol=1e-9)
        and all(np.isfinite(metadata_values[plane]) for plane in range(1, 5))
    ):
        return metadata_values, "metadata_specific"

    parquet_path = row.get("parquet_path")
    if parquet_path is None or (isinstance(parquet_path, float) and np.isnan(parquet_path)):
        raise SystemExit(f"Missing listed parquet path for {row.get('filename_base', '<unknown>')}.")
    return compute_streamer_percentages_from_parquet(parquet_path, study_cfg.streamer_threshold_qsum), "parquet"


def _format_z_config(z_positions: dict[int, float]) -> str:
    return "/".join(f"{z_positions[plane]:g}" for plane in range(1, 5))


def resolve_z_positions(row: pd.Series) -> dict[int, float]:
    metadata_values = {
        plane: _scalar_to_float(row.get(Z_POSITION_COLUMNS[plane]))
        for plane in range(1, 5)
    }
    if all(np.isfinite(metadata_values[plane]) for plane in range(1, 5)):
        return metadata_values

    parquet_path = row.get("parquet_path")
    if parquet_path is None or (isinstance(parquet_path, float) and np.isnan(parquet_path)):
        return metadata_values
    cache_key = str(parquet_path)
    cached = Z_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    frame = pd.read_parquet(Path(parquet_path), columns=list(Z_POSITION_COLUMNS.values()))
    if frame.empty:
        return metadata_values
    result = {
        plane: _scalar_to_float(frame.iloc[0][Z_POSITION_COLUMNS[plane]])
        for plane in range(1, 5)
    }
    Z_CONFIG_CACHE[cache_key] = result
    return result


def build_real_file_screening(real_metadata: pd.DataFrame, study_cfg: ChargeStudyConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in real_metadata.iterrows():
        streamer_percentages, streamer_source = resolve_streamer_percentages(row, study_cfg)
        z_positions = resolve_z_positions(row)
        selector_value = _streamer_selector_value(streamer_percentages, study_cfg.real_streamer_selector)
        streamer_values = np.array(list(streamer_percentages.values()), dtype=float)
        finite_streamers = streamer_values[np.isfinite(streamer_values)]
        record: dict[str, object] = {
            "station_label": row["station_label"],
            "filename_base": row["filename_base"],
            "datetime": row.get("datetime"),
            "parquet_path": row.get("parquet_path"),
            "streamer_percent_source": streamer_source,
            "streamer_selector_mode": study_cfg.real_streamer_selector,
            "streamer_selector_percent": selector_value,
            "streamer_mean_percent": float(np.mean(finite_streamers)) if finite_streamers.size else np.nan,
            "z_config_label": _format_z_config(z_positions)
            if all(np.isfinite(z_positions[plane]) for plane in range(1, 5))
            else "",
            "streamer_target_shortfall": float(
                max(0.0, study_cfg.real_streamer_min_percent - selector_value)
            )
            if np.isfinite(selector_value)
            else np.nan,
            "meets_streamer_target": bool(
                np.isfinite(selector_value) and selector_value >= study_cfg.real_streamer_min_percent
            ),
        }
        for plane in range(1, 5):
            record[f"streamer_P{plane}"] = streamer_percentages[plane]
            record[f"empirical_eff_plane_{plane}"] = _scalar_to_float(row.get(f"empirical_eff_plane_{plane}"))
            record[Z_POSITION_COLUMNS[plane]] = z_positions[plane]
        rows.append(record)
    screening = pd.DataFrame(rows)
    if screening.empty:
        return pd.DataFrame(
            columns=[
                "station_label",
                "filename_base",
                "datetime",
                "parquet_path",
                "streamer_percent_source",
                "streamer_selector_mode",
                "streamer_selector_percent",
                "streamer_mean_percent",
                "z_config_label",
                "streamer_target_shortfall",
                "meets_streamer_target",
            ]
            + [f"streamer_P{plane}" for plane in range(1, 5)]
            + [f"empirical_eff_plane_{plane}" for plane in range(1, 5)]
            + [Z_POSITION_COLUMNS[plane] for plane in range(1, 5)]
        )
    screening.sort_values(
        ["streamer_target_shortfall", "streamer_selector_percent", "streamer_mean_percent", "datetime"],
        ascending=[True, False, False, False],
        inplace=True,
    )
    screening.reset_index(drop=True, inplace=True)
    screening.insert(0, "screening_rank", np.arange(1, len(screening) + 1))
    return screening


def build_pair_candidates(
    sim_metadata: pd.DataFrame,
    real_screening: pd.DataFrame,
    study_cfg: ChargeStudyConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sim_pool = sim_metadata.copy()
    for _, real_row in real_screening.iterrows():
        real_efficiencies = {
            plane: _scalar_to_float(real_row.get(f"empirical_eff_plane_{plane}"))
            for plane in range(1, 5)
        }
        real_z_positions = {
            plane: _scalar_to_float(real_row.get(Z_POSITION_COLUMNS[plane]))
            for plane in range(1, 5)
        }
        if not all(np.isfinite(real_efficiencies[plane]) for plane in range(1, 5)):
            continue
        if not all(np.isfinite(real_z_positions[plane]) for plane in range(1, 5)):
            continue

        for _, sim_row in sim_pool.iterrows():
            sim_efficiencies = {
                plane: _scalar_to_float(sim_row.get(f"empirical_eff_plane_{plane}"))
                for plane in range(1, 5)
            }
            sim_z_positions = resolve_z_positions(sim_row)
            if not all(np.isfinite(sim_efficiencies[plane]) for plane in range(1, 5)):
                continue
            if not all(np.isfinite(sim_z_positions[plane]) for plane in range(1, 5)):
                continue
            if any(
                not np.isclose(sim_z_positions[plane], real_z_positions[plane], rtol=0.0, atol=1e-9)
                for plane in range(1, 5)
            ):
                continue

            deltas = {
                plane: abs(sim_efficiencies[plane] - real_efficiencies[plane])
                for plane in range(1, 5)
            }
            max_delta = max(deltas.values())
            mean_delta = float(np.mean(list(deltas.values())))

            record: dict[str, object] = {
                "sim_station": sim_row["station_label"],
                "sim_basename": sim_row["filename_base"],
                "sim_datetime": sim_row.get("datetime"),
                "sim_path": sim_row.get("parquet_path"),
                "real_station": real_row["station_label"],
                "real_basename": real_row["filename_base"],
                "real_datetime": real_row.get("datetime"),
                "real_path": real_row.get("parquet_path"),
                "real_streamer_selector_mode": study_cfg.real_streamer_selector,
                "real_streamer_selector_percent": _scalar_to_float(real_row.get("streamer_selector_percent")),
                "real_streamer_mean_percent": _scalar_to_float(real_row.get("streamer_mean_percent")),
                "z_config_label": _format_z_config(real_z_positions),
                "real_streamer_target_shortfall": _scalar_to_float(real_row.get("streamer_target_shortfall")),
                "real_meets_streamer_target": bool(real_row.get("meets_streamer_target", False)),
                "within_efficiency_advisory_limits": bool(
                    max_delta <= study_cfg.max_per_plane_efficiency_abs_difference
                    and mean_delta <= study_cfg.max_mean_efficiency_abs_difference
                ),
                "max_efficiency_abs_delta": max_delta,
                "mean_efficiency_abs_delta": mean_delta,
            }
            for plane in range(1, 5):
                record[f"z_P{plane}"] = real_z_positions[plane]
                record[f"sim_eff_plane_{plane}"] = sim_efficiencies[plane]
                record[f"real_eff_plane_{plane}"] = real_efficiencies[plane]
                record[f"eff_delta_plane_{plane}"] = deltas[plane]
                record[f"real_streamer_P{plane}"] = _scalar_to_float(real_row.get(f"streamer_P{plane}"))
            rows.append(record)

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        return pd.DataFrame(
            columns=[
                "sim_station",
                "sim_basename",
                "sim_datetime",
                "sim_path",
                "real_station",
                "real_basename",
                "real_datetime",
                "real_path",
                "real_streamer_selector_mode",
                "real_streamer_selector_percent",
                "real_streamer_mean_percent",
                "z_config_label",
                "real_streamer_target_shortfall",
                "real_meets_streamer_target",
                "within_efficiency_advisory_limits",
                "max_efficiency_abs_delta",
                "mean_efficiency_abs_delta",
            ]
            + [f"z_P{plane}" for plane in range(1, 5)]
            + [f"sim_eff_plane_{plane}" for plane in range(1, 5)]
            + [f"real_eff_plane_{plane}" for plane in range(1, 5)]
            + [f"eff_delta_plane_{plane}" for plane in range(1, 5)]
            + [f"real_streamer_P{plane}" for plane in range(1, 5)]
        )
    candidate_df.sort_values(
        [
            "real_streamer_target_shortfall",
            "within_efficiency_advisory_limits",
            "mean_efficiency_abs_delta",
            "max_efficiency_abs_delta",
            "real_streamer_selector_percent",
            "real_streamer_mean_percent",
            "real_datetime",
            "sim_datetime",
            "real_basename",
            "sim_basename",
        ],
        ascending=[True, False, True, True, False, False, False, False, True, True],
        inplace=True,
    )
    candidate_df.reset_index(drop=True, inplace=True)
    candidate_df.insert(0, "pair_rank", np.arange(1, len(candidate_df) + 1))
    return candidate_df


def _deterministic_sample_indices(total: int, wanted: int, token: str) -> np.ndarray:
    if wanted <= 0 or total <= wanted:
        return np.arange(total, dtype=int)
    digest = hashlib.sha256(f"{DEFAULT_SEED}|{token}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=wanted, replace=False))


def _strip_multiplicity(series: pd.Series) -> np.ndarray:
    text = series.astype("string").fillna("0000")
    return np.array([str(value).count("1") for value in text], dtype=int)


def sample_plane_charge_ns(parquet_path: str | Path, plane: int, study_cfg: ChargeStudyConfig) -> np.ndarray:
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

    alphas = np.linspace(study_cfg.alpha_min, study_cfg.alpha_max, study_cfg.alpha_grid_points)
    left = float(
        min(
            log_real.min(),
            log_sim.min() + np.log10(study_cfg.alpha_min),
        )
    )
    right = float(
        max(
            log_real.max(),
            log_sim.max() + np.log10(study_cfg.alpha_max),
        )
    )
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


def _unit_values(
    parquet_path: Path,
    plane: int,
    study_cfg: ChargeStudyConfig,
    unit_key: str,
    calibration: TotChargeCalibration,
) -> np.ndarray:
    values_ns = sample_plane_charge_ns(parquet_path, plane, study_cfg)
    if unit_key == "fc":
        return np.asarray(calibration.width_ns_to_charge_fc(values_ns), dtype=float)
    return values_ns


def summarize_selected_pair(
    selected_pair: pd.Series,
    study_cfg: ChargeStudyConfig,
    calibration: TotChargeCalibration,
    unit_key: str,
    sim_streamers: dict[int, float],
    real_streamers: dict[int, float],
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray | float]]]:
    sim_path = resolve_listed_file_path(
        str(selected_pair["sim_station"]),
        str(selected_pair["sim_basename"]),
        str(selected_pair["sim_path"]),
    )
    real_path = resolve_listed_file_path(
        str(selected_pair["real_station"]),
        str(selected_pair["real_basename"]),
        str(selected_pair["real_path"]),
    )

    rows: list[dict[str, object]] = []
    plot_payload: dict[int, dict[str, np.ndarray | float]] = {}
    for plane in range(1, 5):
        sim_values = _unit_values(sim_path, plane, study_cfg, unit_key, calibration)
        real_values = _unit_values(real_path, plane, study_cfg, unit_key, calibration)
        alpha, mse_original, mse_scaled, hist_edges = alpha_scan(sim_values, real_values, study_cfg)
        rows.append(
            {
                "plane": plane,
                "charge_unit": unit_key,
                "sim_samples": int(sim_values.size),
                "real_samples": int(real_values.size),
                "sim_efficiency": _scalar_to_float(selected_pair.get(f"sim_eff_plane_{plane}")),
                "real_efficiency": _scalar_to_float(selected_pair.get(f"real_eff_plane_{plane}")),
                "efficiency_abs_delta": _scalar_to_float(selected_pair.get(f"eff_delta_plane_{plane}")),
                "sim_streamer_percent": float(sim_streamers[plane]),
                "real_streamer_percent": float(real_streamers[plane]),
                "recommended_charge_scale_alpha": alpha,
                "hist_mse_original": mse_original,
                "hist_mse_scaled": mse_scaled,
                "ks_original": ks_distance(sim_values, real_values),
                "ks_scaled": ks_distance(sim_values * alpha, real_values) if np.isfinite(alpha) else np.nan,
                "sim_log10_charge_sigma": robust_log_sigma(sim_values),
                "real_log10_charge_sigma": robust_log_sigma(real_values),
            }
        )
        plot_payload[plane] = {
            "sim_values": sim_values,
            "real_values": real_values,
            "alpha": alpha,
            "hist_edges": hist_edges,
        }

    summary_df = pd.DataFrame(rows)
    summary_df["real_over_sim_log10_charge_sigma_ratio"] = summary_df.apply(
        lambda row: (
            float(row["real_log10_charge_sigma"] / row["sim_log10_charge_sigma"])
            if np.isfinite(row["real_log10_charge_sigma"])
            and np.isfinite(row["sim_log10_charge_sigma"])
            and row["sim_log10_charge_sigma"] > 0
            else np.nan
        ),
        axis=1,
    )
    return summary_df, plot_payload


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


def plot_linear_overlay(
    selected_pair: pd.Series,
    summary_df: pd.DataFrame,
    plot_payload: dict[int, dict[str, np.ndarray | float]],
    unit_key: str,
) -> None:
    out = output_dir()
    unit_label = UNIT_LABELS[unit_key]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        row = summary_df[summary_df["plane"] == plane].iloc[0]
        payload = plot_payload.get(plane, {})
        sim_values = np.asarray(payload.get("sim_values", np.array([], dtype=float)))
        real_values = np.asarray(payload.get("real_values", np.array([], dtype=float)))
        alpha = float(payload.get("alpha", np.nan))
        scaled_values = sim_values * alpha if np.isfinite(alpha) else np.array([], dtype=float)
        pooled = np.concatenate(
            [arr for arr in (sim_values, real_values, scaled_values) if np.asarray(arr).size]
        ) if (sim_values.size or real_values.size or scaled_values.size) else np.array([], dtype=float)
        positive = pooled[np.isfinite(pooled) & (pooled > 0)]
        if positive.size:
            upper = float(np.quantile(positive, 0.995))
            if not np.isfinite(upper) or upper <= 0:
                upper = float(np.nanmax(positive))
            edges = np.linspace(0.0, upper, 80) if upper > 0 else np.array([])
            if edges.size:
                if sim_values.size:
                    ax.hist(
                        sim_values,
                        bins=edges,
                        density=True,
                        histtype="step",
                        linewidth=1.6,
                        color=GROUP_COLORS["SIM"],
                        label=f"{selected_pair['sim_station']}",
                    )
                if real_values.size:
                    ax.hist(
                        real_values,
                        bins=edges,
                        density=True,
                        histtype="step",
                        linewidth=1.6,
                        color=GROUP_COLORS["REAL"],
                        label=f"{selected_pair['real_station']}",
                    )
                if scaled_values.size:
                    ax.hist(
                        scaled_values,
                        bins=edges,
                        density=True,
                        histtype="step",
                        linewidth=1.4,
                        linestyle="--",
                        color=GROUP_COLORS["SCALED"],
                        label=f"scaled sim x{alpha:.3f}",
                    )
        ax.set_title(
            f"Plane {plane} | alpha={row['recommended_charge_scale_alpha']:.3f}\n"
            f"eff real/sim={row['real_efficiency']:.3f}/{row['sim_efficiency']:.3f} | streamer real/sim="
            f"{row['real_streamer_percent']:.2f}%/{row['sim_streamer_percent']:.2f}%",
            fontsize=10,
        )
        ax.set_xlabel(f"Plane charge [{unit_label}]")
        if plane in (1, 3):
            ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
    _global_legend(fig, axes)
    fig.suptitle(
        f"Selected charge-spectrum pair: {selected_pair['sim_basename']} vs {selected_pair['real_basename']} "
        f"| z={selected_pair['z_config_label']} ({unit_label})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path = out / f"selected_pair_plane_by_plane_overlay_{unit_key}.png"
    fig.savefig(output_path, dpi=180)
    if unit_key == "ns":
        fig.savefig(out / "selected_pair_plane_by_plane_overlay.png", dpi=180)
    plt.close(fig)


def plot_log_overlay(
    selected_pair: pd.Series,
    summary_df: pd.DataFrame,
    plot_payload: dict[int, dict[str, np.ndarray | float]],
    unit_key: str,
) -> None:
    out = output_dir()
    unit_label = UNIT_LABELS[unit_key]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for plane in range(1, 5):
        ax = axes.flat[plane - 1]
        row = summary_df[summary_df["plane"] == plane].iloc[0]
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
                label=f"{selected_pair['sim_station']}",
            )
        if real_values.size and edges.size:
            ax.hist(
                np.log10(real_values),
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=GROUP_COLORS["REAL"],
                label=f"{selected_pair['real_station']}",
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
        ax.set_title(
            f"Plane {plane} | KS {row['ks_original']:.3f} -> {row['ks_scaled']:.3f}\n"
            f"width ratio={row['real_over_sim_log10_charge_sigma_ratio']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel(f"log10(plane charge [{unit_label}])")
        if plane in (1, 3):
            ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
    _global_legend(fig, axes)
    fig.suptitle(
        f"Selected charge-spectrum pair: log overlay for {selected_pair['sim_basename']} vs {selected_pair['real_basename']} ({unit_label})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"selected_pair_charge_overlay_log_{unit_key}.png", dpi=180)
    plt.close(fig)


def build_selected_pair_export(
    selected_pair: pd.Series,
    sim_streamers: dict[int, float],
    sim_streamer_source: str,
    real_streamers: dict[int, float],
    real_streamer_source: str,
) -> pd.DataFrame:
    record = selected_pair.to_dict()
    record["sim_streamer_percent_source"] = sim_streamer_source
    record["real_streamer_percent_source"] = real_streamer_source
    for plane in range(1, 5):
        record[f"sim_streamer_P{plane}"] = sim_streamers[plane]
        record[f"real_streamer_P{plane}"] = real_streamers[plane]
    return pd.DataFrame([record])


def write_report(
    config_path: str,
    selection,
    study_cfg: ChargeStudyConfig,
    current_model: CurrentChargeModel,
    real_screening: pd.DataFrame,
    pair_candidates: pd.DataFrame,
    selected_pair: pd.Series,
    summary_by_unit: dict[str, pd.DataFrame],
    sim_streamers: dict[int, float],
    sim_streamer_source: str,
    real_streamers: dict[int, float],
    real_streamer_source: str,
) -> None:
    out = output_dir()
    best_real_count = int(real_screening["meets_streamer_target"].sum()) if not real_screening.empty else 0
    report_lines = [
        "Selected charge-spectrum tuning pair",
        "===================================",
        "",
        f"Config path: {config_path}",
        f"Simulation station: {study_cfg.simulation_station}",
        f"Real-data station: {study_cfg.real_station}",
        f"Simulation selection basename: {study_cfg.simulation_basename or '<best efficiency match>'}",
        f"Real-data selection basename: {study_cfg.real_basename or '<highest-streamer feasible match>'}",
        "",
        "Selection rules:",
        f"- streamer_threshold_qsum={study_cfg.streamer_threshold_qsum:.6g} ({study_cfg.streamer_threshold_source})",
        f"- real_streamer_selector={study_cfg.real_streamer_selector}",
        f"- preferred real streamer target percent={study_cfg.real_streamer_min_percent:.3f} (soft target)",
        f"- advisory max_per_plane_efficiency_abs_difference={study_cfg.max_per_plane_efficiency_abs_difference:.4f}",
        f"- advisory max_mean_efficiency_abs_difference={study_cfg.max_mean_efficiency_abs_difference:.4f}",
        f"- multiplicity_mode={study_cfg.multiplicity_mode}",
        f"- selection.simulation_date_ranges={selection.simulation_date_ranges}",
        f"- selection.real_date_ranges={selection.real_date_ranges}",
        "",
        "Screening summary:",
        f"- real files screened={len(real_screening)}",
        f"- real files meeting the streamer target={best_real_count}",
        f"- sim/real candidate pairs ranked={len(pair_candidates)}",
        "",
        "Selected files:",
        f"- simulation: {selected_pair['sim_station']} / {selected_pair['sim_basename']} / {selected_pair['sim_datetime']}",
        f"- real-data: {selected_pair['real_station']} / {selected_pair['real_basename']} / {selected_pair['real_datetime']}",
        f"- selected pair rank={int(selected_pair['pair_rank'])}",
        f"- shared z configuration={selected_pair['z_config_label']}",
        f"- real streamer target shortfall={float(selected_pair['real_streamer_target_shortfall']):.4f}",
        f"- within advisory efficiency limits={bool(selected_pair['within_efficiency_advisory_limits'])}",
        f"- pair mean efficiency abs delta={float(selected_pair['mean_efficiency_abs_delta']):.4f}",
        f"- pair max efficiency abs delta={float(selected_pair['max_efficiency_abs_delta']):.4f}",
        "",
        "Streamer percentages with the active threshold:",
        f"- simulation source={sim_streamer_source}",
        f"- real-data source={real_streamer_source}",
        f"- simulation P1/P2/P3/P4={sim_streamers[1]:.3f}% / {sim_streamers[2]:.3f}% / {sim_streamers[3]:.3f}% / {sim_streamers[4]:.3f}%",
        f"- real-data P1/P2/P3/P4={real_streamers[1]:.3f}% / {real_streamers[2]:.3f}% / {real_streamers[3]:.3f}% / {real_streamers[4]:.3f}%",
        "",
        "Current digital-twin charge model:",
        f"- STEP 3 townsend_alpha_per_mm={current_model.townsend_alpha_per_mm:.6g}",
        f"- STEP 3 avalanche_gap_mm={current_model.avalanche_gap_mm:.6g}",
        f"- STEP 3 avalanche_electron_sigma={current_model.avalanche_electron_sigma:.6g}",
        f"- STEP 4 lorentzian_gamma_mm={current_model.lorentzian_gamma_mm:.6g}",
        f"- STEP 4 induced_charge_fraction={current_model.induced_charge_fraction:.6g}",
        f"- STEP 5 qdiff_width={current_model.qdiff_width:.6g}",
        f"- STEP 8 charge_conversion_model={current_model.charge_conversion_model}",
        f"- STEP 8 q_to_time_factor={current_model.q_to_time_factor:.6g}",
        f"- STEP 8 charge_threshold={current_model.charge_threshold:.6g}",
        "",
    ]

    for unit_key, summary_df in summary_by_unit.items():
        report_lines.append(f"Per-plane summary in {UNIT_LABELS[unit_key]}:")
        for row in summary_df.itertuples(index=False):
            report_lines.append(
                f"- Plane {row.plane}: alpha={row.recommended_charge_scale_alpha:.4f}, "
                f"eff real/sim={row.real_efficiency:.4f}/{row.sim_efficiency:.4f}, "
                f"streamer real/sim={row.real_streamer_percent:.3f}%/{row.sim_streamer_percent:.3f}%, "
                f"KS={row.ks_original:.4f}->{row.ks_scaled:.4f}, "
                f"width ratio={row.real_over_sim_log10_charge_sigma_ratio:.4f}, "
                f"samples sim/real={int(row.sim_samples)}/{int(row.real_samples)}"
            )
        report_lines.append("")

    report_lines.extend(
        [
            "Notes:",
            "- The real-data file is ranked by streamer richness, but the streamer target is soft: files below it remain eligible and simply rank lower.",
            "- Only sim/real files with the same z-position configuration are eligible for comparison.",
            "- Within the same z configuration, sim and real-data files are ranked by closest per-plane empirical efficiencies; the configured efficiency limits are advisory only and do not reject the pair.",
            "- The streamer percentage uses the configured Task 3 Q_sum threshold; metadata_specific is used when it matches, otherwise the listed parquet is recomputed directly.",
            "- The main plot is the plane-by-plane overlaid spectrum PNG for the selected pair.",
            "- Plots are advisory only and do not modify simulation settings.",
            "",
        ]
    )
    (out / "selected_pair_charge_spectrum_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(out)

    config_arg = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_tuning_config(config_arg)
    selection = resolve_selection(config)
    study_cfg = load_charge_study_config(config, selection)
    current_model = load_current_charge_model()
    calibration = TotChargeCalibration.from_csv(default_tot_charge_calibration_path(ROOT_DIR))

    sim_metadata = load_station_metadata(study_cfg.simulation_station, selection.simulation_date_ranges)
    real_metadata = load_station_metadata(study_cfg.real_station, selection.real_date_ranges)
    if sim_metadata.empty:
        raise SystemExit(f"No simulation metadata with listed parquets found for {study_cfg.simulation_station}.")
    if real_metadata.empty:
        raise SystemExit(f"No real-data metadata with listed parquets found for {study_cfg.real_station}.")

    sim_metadata = filter_metadata_by_basename(
        sim_metadata,
        study_cfg.simulation_basename,
        label="simulation",
    )
    real_metadata = filter_metadata_by_basename(
        real_metadata,
        study_cfg.real_basename,
        label="real",
    )

    real_screening = build_real_file_screening(real_metadata, study_cfg)
    real_screening.to_csv(out / "charge_spectrum_real_file_screening.csv", index=False)
    if real_screening.empty:
        raise SystemExit("No real-data files remained after metadata filtering.")

    pair_candidates = build_pair_candidates(sim_metadata, real_screening, study_cfg)
    pair_candidates.to_csv(out / "charge_spectrum_pair_candidates.csv", index=False)
    if pair_candidates.empty:
        raise SystemExit(
            "No sim/real pair with the same z-position configuration and finite per-plane efficiencies was found. "
            "See OUTPUTS/charge_spectrum_pair_candidates.csv and "
            "OUTPUTS/charge_spectrum_real_file_screening.csv."
        )

    selected_pair = pair_candidates.iloc[0].copy()
    sim_metadata_row = sim_metadata[
        sim_metadata["filename_base"].astype(str) == str(selected_pair["sim_basename"])
    ].iloc[0]
    real_metadata_row = real_metadata[
        real_metadata["filename_base"].astype(str) == str(selected_pair["real_basename"])
    ].iloc[0]
    sim_streamers, sim_streamer_source = resolve_streamer_percentages(sim_metadata_row, study_cfg)
    real_streamers, real_streamer_source = resolve_streamer_percentages(real_metadata_row, study_cfg)

    selected_pair_export = build_selected_pair_export(
        selected_pair,
        sim_streamers,
        sim_streamer_source,
        real_streamers,
        real_streamer_source,
    )
    selected_pair_export.to_csv(out / "charge_spectrum_selected_pair.csv", index=False)

    summary_by_unit: dict[str, pd.DataFrame] = {}
    for unit_key in ("ns", "fc"):
        summary_df, plot_payload = summarize_selected_pair(
            selected_pair,
            study_cfg,
            calibration,
            unit_key,
            sim_streamers,
            real_streamers,
        )
        summary_df.to_csv(out / f"selected_pair_charge_spectrum_summary_{unit_key}.csv", index=False)
        plot_linear_overlay(selected_pair, summary_df, plot_payload, unit_key)
        summary_by_unit[unit_key] = summary_df

    write_report(
        config_path=str(config.get("_config_path", config_arg or "<default>")),
        selection=selection,
        study_cfg=study_cfg,
        current_model=current_model,
        real_screening=real_screening,
        pair_candidates=pair_candidates,
        selected_pair=selected_pair,
        summary_by_unit=summary_by_unit,
        sim_streamers=sim_streamers,
        sim_streamer_source=sim_streamer_source,
        real_streamers=real_streamers,
        real_streamer_source=real_streamer_source,
    )


if __name__ == "__main__":
    main()
