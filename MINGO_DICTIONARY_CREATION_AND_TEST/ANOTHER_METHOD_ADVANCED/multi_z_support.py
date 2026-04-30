#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import CANONICAL_Z_COLUMNS

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parents[1]
ONLINE_RUN_DICTIONARY_ROOT = (
    REPO_ROOT
    / "MASTER"
    / "CONFIG_FILES"
    / "STAGE_0"
    / "ONLINE_RUN_DICTIONARY"
)

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_z_vector(values: Any) -> tuple[float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"Expected four z positions, got: {values!r}")
    return tuple(float(value) for value in values)


def format_z_vector(z_vector: tuple[float, float, float, float]) -> str:
    return "[" + ", ".join(f"{float(value):g}" for value in z_vector) + "]"


def z_vector_to_id(z_vector: tuple[float, float, float, float]) -> str:
    pieces: list[str] = []
    for value in z_vector:
        text = f"{float(value):.3f}".rstrip("0").rstrip(".")
        text = text.replace("-", "m").replace(".", "p")
        pieces.append(text or "0")
    return "z_" + "__".join(pieces)


def z_mask_for_vector(
    dataframe: pd.DataFrame,
    z_vector: tuple[float, float, float, float],
    *,
    z_columns: list[str] | None = None,
) -> pd.Series:
    columns = list(z_columns or CANONICAL_Z_COLUMNS)
    mask = np.ones(len(dataframe), dtype=bool)
    for column, value in zip(columns, z_vector):
        mask &= np.isclose(pd.to_numeric(dataframe[column], errors="coerce"), float(value), equal_nan=False)
    return pd.Series(mask, index=dataframe.index)


def filter_by_z_vectors(
    dataframe: pd.DataFrame,
    z_vectors: list[tuple[float, float, float, float]],
    *,
    z_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = list(z_columns or CANONICAL_Z_COLUMNS)
    if not z_vectors:
        return dataframe.iloc[0:0].copy()
    mask = np.zeros(len(dataframe), dtype=bool)
    for z_vector in z_vectors:
        mask |= z_mask_for_vector(dataframe, z_vector, z_columns=columns).to_numpy(dtype=bool)
    return dataframe.loc[mask].copy()


def add_z_config_columns(
    dataframe: pd.DataFrame,
    *,
    z_columns: list[str] | None = None,
    id_column: str = "z_config_id",
    label_column: str = "z_config_label",
) -> pd.DataFrame:
    columns = list(z_columns or CANONICAL_Z_COLUMNS)
    work = dataframe.copy()
    z_numeric = work[columns].apply(pd.to_numeric, errors="coerce")

    config_ids: list[str | None] = []
    labels: list[str | None] = []
    for row in z_numeric.itertuples(index=False, name=None):
        if any(pd.isna(value) for value in row):
            config_ids.append(None)
            labels.append(None)
            continue
        z_vector = normalize_z_vector(list(row))
        config_ids.append(z_vector_to_id(z_vector))
        labels.append(format_z_vector(z_vector))

    work[id_column] = pd.Series(config_ids, index=work.index, dtype="object")
    work[label_column] = pd.Series(labels, index=work.index, dtype="object")
    return work


def unique_z_vectors(
    dataframe: pd.DataFrame,
    *,
    z_columns: list[str] | None = None,
) -> list[tuple[float, float, float, float]]:
    columns = list(z_columns or CANONICAL_Z_COLUMNS)
    z_numeric = dataframe[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if z_numeric.empty:
        return []
    return [
        normalize_z_vector(list(row))
        for row in z_numeric.drop_duplicates().itertuples(index=False, name=None)
    ]


def parse_station_id(raw: object) -> int:
    if raw in (None, "", "null", "None"):
        raise ValueError("step5.station must not be empty.")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)
    text = str(raw).strip()
    match = re.fullmatch(r"(?i)MINGO(\d{1,2})", text)
    if match is not None:
        return int(match.group(1))
    return int(text)


def parse_time_bound(value: object, *, end_of_day: bool) -> pd.Timestamp | None:
    if value in (None, "", "null", "None"):
        return None
    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Could not parse datetime bound: {value!r}")
    if end_of_day and _DATE_ONLY_RE.fullmatch(text):
        parsed = parsed + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return pd.Timestamp(parsed)


def online_run_dictionary_path(station_id: int) -> Path:
    suffix = f"{int(station_id):02d}"
    candidates = [
        ONLINE_RUN_DICTIONARY_ROOT / f"STATION_{int(station_id)}" / f"input_file_mingo{suffix}.csv",
        ONLINE_RUN_DICTIONARY_ROOT / f"STATION_{suffix}" / f"input_file_mingo{suffix}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = sorted(ONLINE_RUN_DICTIONARY_ROOT.glob(f"STATION_*/input_file_mingo{suffix}.csv"))
    if found:
        return found[0]
    raise FileNotFoundError(f"ONLINE_RUN_DICTIONARY CSV not found for station {station_id}")


def load_online_schedule(station_id: int) -> tuple[pd.DataFrame, Path]:
    path = online_run_dictionary_path(station_id)
    raw = pd.read_csv(path, header=[0, 1], low_memory=False)
    if isinstance(raw.columns, pd.MultiIndex):
        columns = []
        for col in raw.columns:
            top = str(col[0]).strip()
            sub = str(col[1]).strip()
            columns.append(sub if sub and not sub.lower().startswith("unnamed") else top)
        dataframe = raw.copy()
        dataframe.columns = columns
    else:
        dataframe = raw.copy()

    col_by_lower = {str(column).strip().lower(): column for column in dataframe.columns}

    def pick(*names: str) -> str | None:
        for name in names:
            column = col_by_lower.get(name.lower())
            if column is not None:
                return str(column)
        return None

    station_col = pick("station", "detector")
    start_col = pick("start", "date_start")
    end_col = pick("end", "date_end")
    p1_col = pick("p1")
    p2_col = pick("p2")
    p3_col = pick("p3")
    p4_col = pick("p4")

    required = [start_col, p1_col, p2_col, p3_col, p4_col]
    if any(column is None for column in required):
        raise ValueError(f"Could not parse ONLINE_RUN_DICTIONARY schema in {path}")

    work = pd.DataFrame(index=dataframe.index)
    if station_col is not None:
        station_series = pd.to_numeric(dataframe[station_col], errors="coerce")
        work = work.loc[station_series == int(station_id)].copy()

    work["start_utc"] = pd.to_datetime(dataframe[start_col], errors="coerce", utc=True)
    work["end_utc"] = pd.to_datetime(dataframe[end_col], errors="coerce", utc=True) if end_col is not None else pd.NaT
    work["z_tuple"] = dataframe[[p1_col, p2_col, p3_col, p4_col]].apply(
        lambda row: normalize_z_vector(list(row.values)),
        axis=1,
    )
    work = work.dropna(subset=["start_utc"]).sort_values(["start_utc", "end_utc"], kind="mergesort")
    return work.reset_index(drop=True), path


def select_schedule_rows_for_window(
    schedule_df: pd.DataFrame,
    *,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
) -> pd.DataFrame:
    if schedule_df.empty:
        return schedule_df.copy()
    keep = pd.Series(True, index=schedule_df.index)
    far_future = pd.Timestamp("2100-01-01", tz="UTC")
    if date_from is not None:
        keep &= schedule_df["end_utc"].fillna(far_future) >= date_from
    if date_to is not None:
        keep &= schedule_df["start_utc"] <= date_to
    return schedule_df.loc[keep].copy()


def resolve_active_z_vectors_from_config(config: dict[str, Any]) -> tuple[list[tuple[float, float, float, float]], dict[str, Any]]:
    step1_config = config.get("step1", {})
    if not isinstance(step1_config, dict):
        step1_config = {}
    step5_config = config.get("step5", {})
    if not isinstance(step5_config, dict):
        step5_config = {}

    configured_vector = step1_config.get("z_position_vector")
    raw_mode = step1_config.get("z_selection_mode")
    if raw_mode in (None, "", "null", "None"):
        selection_mode = "configured_vector" if configured_vector not in (None, "", "null", "None") else "step5_window"
    else:
        selection_mode = str(raw_mode).strip().lower()
    selection_mode = {
        "configured": "configured_vector",
        "fixed": "configured_vector",
        "window": "step5_window",
        "step5": "step5_window",
        "date_window": "step5_window",
    }.get(selection_mode, selection_mode)

    if selection_mode == "configured_vector":
        if configured_vector in (None, "", "null", "None"):
            raise ValueError("step1.z_selection_mode='configured_vector' requires step1.z_position_vector.")
        z_vector = normalize_z_vector(configured_vector)
        return [z_vector], {
            "selection_mode": selection_mode,
            "configured_z_position_vector": list(z_vector),
            "selected_z_positions": [list(z_vector)],
            "station_id": None,
            "online_run_dictionary_csv": None,
            "date_from": None,
            "date_to": None,
            "online_schedule_rows_total": None,
            "online_schedule_rows_in_requested_window": None,
        }

    if selection_mode != "step5_window":
        raise ValueError(
            "Unsupported step1.z_selection_mode: "
            f"{step1_config.get('z_selection_mode')!r}. Supported values are 'step5_window' and 'configured_vector'."
        )

    station_id = parse_station_id(step5_config.get("station", "MINGO01"))
    date_from = parse_time_bound(step5_config.get("date_from"), end_of_day=False)
    date_to = parse_time_bound(step5_config.get("date_to"), end_of_day=True)
    schedule_all, schedule_path = load_online_schedule(station_id)
    schedule_window = select_schedule_rows_for_window(
        schedule_all,
        date_from=date_from,
        date_to=date_to,
    )
    z_vectors = [normalize_z_vector(value) for value in sorted(set(schedule_window["z_tuple"].tolist()))]
    if not z_vectors and configured_vector not in (None, "", "null", "None"):
        z_vector = normalize_z_vector(configured_vector)
        z_vectors = [z_vector]

    return z_vectors, {
        "selection_mode": selection_mode,
        "configured_z_position_vector": configured_vector,
        "selected_z_positions": [list(z_vector) for z_vector in z_vectors],
        "station_id": int(station_id),
        "online_run_dictionary_csv": str(schedule_path),
        "date_from": str(date_from) if date_from is not None else None,
        "date_to": str(date_to) if date_to is not None else None,
        "online_schedule_rows_total": int(len(schedule_all)),
        "online_schedule_rows_in_requested_window": int(len(schedule_window)),
    }


def load_reference_curve_table(flux_cells_path: Path) -> pd.DataFrame:
    if not flux_cells_path.exists():
        raise FileNotFoundError(
            f"Step 2 flux-cell diagnostics not found: {flux_cells_path}. "
            "Run Step 2 before building rate-to-flux mappings."
        )

    flux_cells = pd.read_csv(flux_cells_path)
    required_columns = [
        "flux_bin_index",
        "flux_bin_lo",
        "flux_bin_hi",
        "flux_bin_center",
        "reference_rate_median",
    ]
    missing = [column for column in required_columns if column not in flux_cells.columns]
    if missing:
        raise ValueError(
            "Step 2 flux-cell diagnostics are missing required columns: " + ", ".join(missing)
        )
    return flux_cells.copy()


def build_rate_to_flux_lines(reference_table: pd.DataFrame) -> pd.DataFrame:
    if reference_table.empty:
        return pd.DataFrame()

    group_columns = [column for column in [*CANONICAL_Z_COLUMNS, "z_config_id"] if column in reference_table.columns]
    required_curve_columns = ["reference_rate_median", "flux_bin_center"]
    missing = [column for column in required_curve_columns if column not in reference_table.columns]
    if missing:
        raise ValueError(
            "Reference table is missing columns required to fit rate-to-flux lines: " + ", ".join(missing)
        )

    if not group_columns:
        grouped_items = [((), reference_table.copy())]
    else:
        grouped_items = list(reference_table.groupby(group_columns, dropna=False, sort=True))

    rows: list[dict[str, Any]] = []
    for group_key, subset in grouped_items:
        curve = (
            subset.groupby("flux_bin_index", dropna=False)
            .agg(
                reference_rate_median=("reference_rate_median", "median"),
                flux_bin_center=("flux_bin_center", "median"),
            )
            .dropna(subset=["reference_rate_median", "flux_bin_center"])
            .reset_index(drop=True)
            .sort_values("reference_rate_median")
            .reset_index(drop=True)
        )
        if curve.empty:
            continue

        x_values = curve["reference_rate_median"].to_numpy(dtype=float)
        y_values = curve["flux_bin_center"].to_numpy(dtype=float)

        if len(curve) == 1:
            slope = 0.0
            intercept = float(y_values[0])
            r_squared = np.nan
            fit_method = "single_point_constant"
        else:
            slope, intercept = np.polyfit(x_values, y_values, deg=1)
            y_fit = slope * x_values + intercept
            ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
            ss_res = float(np.sum((y_values - y_fit) ** 2))
            r_squared = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
            fit_method = "linear_fit"

        row: dict[str, Any] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "fit_method": fit_method,
            "n_reference_points": int(len(curve)),
            "reference_rate_min": float(np.min(x_values)),
            "reference_rate_max": float(np.max(x_values)),
            "reference_flux_min": float(np.min(y_values)),
            "reference_flux_max": float(np.max(y_values)),
            "reference_rate_mean": float(np.mean(x_values)),
            "reference_flux_mean": float(np.mean(y_values)),
            "r_squared": None if not np.isfinite(r_squared) else float(r_squared),
        }
        if group_columns:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            for column, value in zip(group_columns, group_key):
                row[column] = value
        rows.append(row)

    return pd.DataFrame(rows)


def load_rate_to_flux_lines(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Step 2 rate-to-flux lines file not found: {path}")
    dataframe = pd.read_csv(path, low_memory=False)
    required = ["slope", "intercept"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Rate-to-flux lines file is missing required columns: " + ", ".join(missing)
        )
    return dataframe


def apply_rate_to_flux_lines(
    rate_values: pd.Series,
    *,
    row_z_frame: pd.DataFrame | None,
    line_table: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    rate_numeric = pd.to_numeric(rate_values, errors="coerce")
    mapped = pd.Series(np.nan, index=rate_numeric.index, dtype=float)
    method = pd.Series("missing_rate", index=rate_numeric.index, dtype="object")

    if line_table.empty:
        method.loc[rate_numeric.notna()] = "missing_rate_to_flux_line"
        return mapped, method

    line_has_z = all(column in line_table.columns for column in CANONICAL_Z_COLUMNS)
    if not line_has_z:
        row = line_table.iloc[0]
        valid_mask = rate_numeric.notna()
        mapped.loc[valid_mask] = float(row["slope"]) * rate_numeric.loc[valid_mask] + float(row["intercept"])
        method.loc[valid_mask] = str(row.get("fit_method", "linear_fit"))
        return mapped, method

    if row_z_frame is None:
        method.loc[rate_numeric.notna()] = "missing_row_z"
        return mapped, method

    row_z_numeric = row_z_frame[CANONICAL_Z_COLUMNS].apply(pd.to_numeric, errors="coerce")
    for z_vector in unique_z_vectors(row_z_numeric):
        row_mask = z_mask_for_vector(row_z_numeric, z_vector)
        line_mask = z_mask_for_vector(line_table, z_vector)
        line_subset = line_table.loc[line_mask].reset_index(drop=True)
        if line_subset.empty:
            continue
        line = line_subset.iloc[0]
        valid_mask = row_mask & rate_numeric.notna()
        mapped.loc[valid_mask] = float(line["slope"]) * rate_numeric.loc[valid_mask] + float(line["intercept"])
        method.loc[valid_mask] = str(line.get("fit_method", "linear_fit"))

    unmatched = rate_numeric.notna() & mapped.isna()
    method.loc[unmatched] = "missing_z_line"
    return mapped, method


def reference_table_has_z(reference_table: pd.DataFrame) -> bool:
    return all(column in reference_table.columns for column in CANONICAL_Z_COLUMNS)


def _curve_subset_for_z(
    reference_table: pd.DataFrame,
    z_vector: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    if not reference_table_has_z(reference_table):
        return reference_table.copy()
    if z_vector is None:
        return reference_table.iloc[0:0].copy()
    mask = np.ones(len(reference_table), dtype=bool)
    for column, value in zip(CANONICAL_Z_COLUMNS, z_vector):
        mask &= np.isclose(pd.to_numeric(reference_table[column], errors="coerce"), float(value), equal_nan=False)
    return reference_table.loc[mask].copy()


def _reference_curve_from_subset(subset: pd.DataFrame) -> pd.DataFrame:
    curve = (
        subset.groupby("flux_bin_index", dropna=False)
        .agg(
            flux_bin_lo=("flux_bin_lo", "min"),
            flux_bin_hi=("flux_bin_hi", "max"),
            flux_bin_center=("flux_bin_center", "median"),
            reference_rate_median=("reference_rate_median", "median"),
        )
        .reset_index()
        .sort_values("flux_bin_index")
        .reset_index(drop=True)
    )
    return curve.dropna(subset=["flux_bin_lo", "flux_bin_hi", "reference_rate_median"]).reset_index(drop=True)


def _map_reference_rate_by_flux_single_curve(
    flux_values: pd.Series,
    curve: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    flux_numeric = pd.to_numeric(flux_values, errors="coerce")
    if curve.empty:
        return (
            pd.Series(np.nan, index=flux_numeric.index, dtype=float),
            pd.Series("missing_reference_curve", index=flux_numeric.index, dtype="object"),
        )

    bin_edges = [float(curve["flux_bin_lo"].iloc[0])] + [float(value) for value in curve["flux_bin_hi"].tolist()]
    labels = [int(value) for value in curve["flux_bin_index"].tolist()]
    binned = pd.cut(
        flux_numeric,
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    reference_rate_by_bin = curve.set_index("flux_bin_index")["reference_rate_median"]
    mapped = pd.to_numeric(binned.map(reference_rate_by_bin), errors="coerce")

    assignment_method = pd.Series("flux_bin_edges", index=flux_numeric.index, dtype="object")
    missing_mask = flux_numeric.notna() & mapped.isna()
    if missing_mask.any():
        centers = curve["flux_bin_center"].to_numpy(dtype=float)
        reference_values = curve["reference_rate_median"].to_numpy(dtype=float)
        flux_missing = flux_numeric.loc[missing_mask].to_numpy(dtype=float)
        nearest_indices = np.abs(flux_missing[:, None] - centers[None, :]).argmin(axis=1)
        mapped.loc[missing_mask] = reference_values[nearest_indices]
        assignment_method.loc[missing_mask] = "nearest_flux_bin_center"

    assignment_method.loc[flux_numeric.isna()] = "missing_flux"
    return mapped, assignment_method


def map_reference_rate_by_flux(
    flux_values: pd.Series,
    *,
    row_z_frame: pd.DataFrame | None,
    reference_table: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    flux_numeric = pd.to_numeric(flux_values, errors="coerce")
    mapped = pd.Series(np.nan, index=flux_numeric.index, dtype=float)
    method = pd.Series("missing_flux", index=flux_numeric.index, dtype="object")

    if not reference_table_has_z(reference_table):
        return _map_reference_rate_by_flux_single_curve(flux_numeric, _reference_curve_from_subset(reference_table))

    if row_z_frame is None:
        method.loc[flux_numeric.notna()] = "missing_row_z"
        return mapped, method

    row_z_numeric = row_z_frame[CANONICAL_Z_COLUMNS].apply(pd.to_numeric, errors="coerce")
    for z_vector in unique_z_vectors(row_z_numeric):
        row_mask = z_mask_for_vector(row_z_numeric, z_vector)
        curve = _reference_curve_from_subset(_curve_subset_for_z(reference_table, z_vector))
        sub_mapped, sub_method = _map_reference_rate_by_flux_single_curve(flux_numeric.loc[row_mask], curve)
        mapped.loc[row_mask] = sub_mapped
        method.loc[row_mask] = sub_method

    unmatched = flux_numeric.notna() & mapped.isna()
    method.loc[unmatched] = "missing_z_curve"
    return mapped, method


def _map_flux_by_rate_single_curve(
    rate_values: pd.Series,
    curve: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    rate_numeric = pd.to_numeric(rate_values, errors="coerce")
    if curve.empty:
        return (
            pd.Series(np.nan, index=rate_numeric.index, dtype=float),
            pd.Series("missing_reference_curve", index=rate_numeric.index, dtype="object"),
        )

    rate_curve = (
        curve.groupby("reference_rate_median", dropna=False)
        .agg(flux_bin_center=("flux_bin_center", "median"))
        .reset_index()
        .sort_values("reference_rate_median")
        .dropna(subset=["reference_rate_median", "flux_bin_center"])
        .reset_index(drop=True)
    )
    if rate_curve.empty:
        return (
            pd.Series(np.nan, index=rate_numeric.index, dtype=float),
            pd.Series("missing_reference_curve", index=rate_numeric.index, dtype="object"),
        )

    if len(rate_curve) == 1:
        mapped = pd.Series(float(rate_curve["flux_bin_center"].iloc[0]), index=rate_numeric.index, dtype=float)
        method = pd.Series("single_reference_point", index=rate_numeric.index, dtype="object")
        method.loc[rate_numeric.isna()] = "missing_rate"
        mapped.loc[rate_numeric.isna()] = np.nan
        return mapped, method

    x_values = rate_curve["reference_rate_median"].to_numpy(dtype=float)
    y_values = rate_curve["flux_bin_center"].to_numpy(dtype=float)
    valid_mask = rate_numeric.notna()
    mapped = pd.Series(np.nan, index=rate_numeric.index, dtype=float)
    method = pd.Series("missing_rate", index=rate_numeric.index, dtype="object")
    if valid_mask.any():
        rate_valid = rate_numeric.loc[valid_mask].to_numpy(dtype=float)
        mapped_valid = np.interp(rate_valid, x_values, y_values, left=y_values[0], right=y_values[-1])
        mapped.loc[valid_mask] = mapped_valid
        method.loc[valid_mask] = "interpolated_rate_curve"
        below = rate_valid < x_values[0]
        above = rate_valid > x_values[-1]
        valid_index = rate_numeric.loc[valid_mask].index
        method.loc[valid_index[below]] = "rate_curve_clamped_low"
        method.loc[valid_index[above]] = "rate_curve_clamped_high"
    return mapped, method


def map_flux_by_reference_rate(
    rate_values: pd.Series,
    *,
    row_z_frame: pd.DataFrame | None,
    reference_table: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    rate_numeric = pd.to_numeric(rate_values, errors="coerce")
    mapped = pd.Series(np.nan, index=rate_numeric.index, dtype=float)
    method = pd.Series("missing_rate", index=rate_numeric.index, dtype="object")

    if not reference_table_has_z(reference_table):
        return _map_flux_by_rate_single_curve(rate_numeric, _reference_curve_from_subset(reference_table))

    if row_z_frame is None:
        method.loc[rate_numeric.notna()] = "missing_row_z"
        return mapped, method

    row_z_numeric = row_z_frame[CANONICAL_Z_COLUMNS].apply(pd.to_numeric, errors="coerce")
    for z_vector in unique_z_vectors(row_z_numeric):
        row_mask = z_mask_for_vector(row_z_numeric, z_vector)
        curve = _reference_curve_from_subset(_curve_subset_for_z(reference_table, z_vector))
        sub_mapped, sub_method = _map_flux_by_rate_single_curve(rate_numeric.loc[row_mask], curve)
        mapped.loc[row_mask] = sub_mapped
        method.loc[row_mask] = sub_method

    unmatched = rate_numeric.notna() & mapped.isna()
    method.loc[unmatched] = "missing_z_curve"
    return mapped, method


def write_ascii_table_with_comments(
    path: Path,
    dataframe: pd.DataFrame,
    *,
    comments: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for comment in comments:
            handle.write(comment.rstrip("\n") + "\n")
        dataframe.to_csv(handle, sep=" ", index=False, float_format="%.6f")


def json_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))
