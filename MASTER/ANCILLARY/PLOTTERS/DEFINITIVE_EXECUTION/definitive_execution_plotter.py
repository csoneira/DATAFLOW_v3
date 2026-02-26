#!/usr/bin/env python3
"""Generate definitive execution timeline maps for each station."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex


SCRIPT_PATH = Path(__file__).resolve()


def detect_repo_root() -> Path:
    for parent in SCRIPT_PATH.parents:
        if (parent / "MASTER").is_dir() and (parent / "STATIONS").is_dir():
            return parent
    return Path.home() / "DATAFLOW_v3"


REPO_ROOT = detect_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.common.plot_utils import pdf_save_rasterized_page  # noqa: E402


STATIONS_ROOT = REPO_ROOT / "STATIONS"
PLOTTER_DIR = SCRIPT_PATH.parent
PLOTS_DIR = PLOTTER_DIR / "PLOTS"
DEFAULT_OUTPUT_FILENAME = "definitive_execution_map_report.pdf"
DEFAULT_CONFIG_PATH = PLOTTER_DIR / "definitive_execution_config.json"
DEFAULT_LAST_HOURS = 2.0
DEFAULT_POINT_SIZE = 16.0
DEFAULT_SHARED_X_STATIONS: Tuple[str, ...] = ("MINGO01", "MINGO02", "MINGO03", "MINGO04")
DEFAULT_FREE_X_STATIONS: Tuple[str, ...] = ("MINGO01",)
DEFAULT_PANEL_HEIGHT_RATIOS: Tuple[float, float, float] = (1.0, 4.0, 1.0)
DEFAULT_MIDDLE_LOG_SCALE_SECONDS = 600.0
NOW_Y_MARGIN_MINUTES = 10.0

TASK_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASENAME_TIMESTAMP_DIGITS = 11
FILENAME_TIMESTAMP_PATTERN = re.compile(r"mi0\d(\d{11})$", re.IGNORECASE)


@dataclass(frozen=True)
class StageSpec:
    index: int
    label: str
    color: str
    csv_path: Path
    basename_columns: Tuple[str, ...]
    execution_columns: Tuple[str, ...]
    execution_format: Optional[str]
    use_basename_as_execution: bool = False
    use_csv_timestamp_as_execution: bool = False


def build_discrete_turbo_stage_colors() -> Tuple[str, ...]:
    # Stage 0 is rendered as a hollow marker; all other stages use a discretized Turbo palette.
    turbo_steps = 7  # STEP 1, STEP 2, TASK 1..5
    turbo_cmap = colormaps["turbo"].resampled(turbo_steps)
    colors: List[str] = ["#ffffff"]
    colors.extend(to_hex(turbo_cmap(step)) for step in range(turbo_steps))
    return tuple(colors)


STAGE_COLORS = build_discrete_turbo_stage_colors()


def configure_matplotlib_style() -> None:
    plt.style.use("default")


def normalize_station_token(token: str) -> Optional[str]:
    cleaned = token.strip().upper()
    if not cleaned:
        return None
    if cleaned.startswith("MINGO"):
        cleaned = cleaned[5:]
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if not digits:
        return None
    return f"MINGO{int(digits):02d}"


def list_available_stations() -> List[str]:
    if not STATIONS_ROOT.exists():
        return []
    stations: List[str] = []
    for entry in STATIONS_ROOT.iterdir():
        if entry.is_dir() and re.fullmatch(r"MINGO\d{2}", entry.name.upper()):
            stations.append(entry.name.upper())
    stations.sort()
    return stations


def resolve_station_selection(tokens: Sequence[str]) -> List[str]:
    available = list_available_stations()
    if not tokens:
        return available

    selected: List[str] = []
    invalid: List[str] = []
    for token in tokens:
        station = normalize_station_token(token)
        if station is None or station not in available:
            invalid.append(token)
            continue
        selected.append(station)

    if invalid:
        print(
            "[definitive_execution_plotter] Ignoring unknown station(s): "
            + ", ".join(invalid),
            file=sys.stderr,
        )

    return sorted(dict.fromkeys(selected))


def normalize_existing_station_tokens(tokens: Sequence[str]) -> List[str]:
    available = set(list_available_stations())
    selected: List[str] = []
    for token in tokens:
        station = normalize_station_token(str(token))
        if station is None or station not in available:
            continue
        selected.append(station)
    return sorted(dict.fromkeys(selected))


def extract_timestamp_from_basename(value: str) -> Optional[datetime]:
    if not value:
        return None

    stem = Path(value).stem.strip()
    if not stem:
        return None

    try:
        return datetime.strptime(stem, "%Y-%m-%d_%H.%M.%S")
    except ValueError:
        pass

    match = FILENAME_TIMESTAMP_PATTERN.search(stem)
    if match:
        digits = match.group(1)
    else:
        digits = "".join(ch for ch in stem if ch.isdigit())
        if len(digits) < BASENAME_TIMESTAMP_DIGITS:
            return None
        digits = digits[-BASENAME_TIMESTAMP_DIGITS:]

    try:
        year = 2000 + int(digits[0:2])
        day_of_year = int(digits[2:5])
        hour = int(digits[5:7])
        minute = int(digits[7:9])
        second = int(digits[9:11])
    except ValueError:
        return None

    if not (1 <= day_of_year <= 366):
        return None
    if not (0 <= hour <= 23):
        return None
    if not (0 <= minute <= 59):
        return None
    if not (0 <= second <= 59):
        return None

    base = datetime(year, 1, 1)
    return base + timedelta(
        days=day_of_year - 1,
        hours=hour,
        minutes=minute,
        seconds=second,
    )


def first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def parse_execution_series(series: pd.Series, fmt: Optional[str]) -> pd.Series:
    text = series.astype(str).str.strip()
    if fmt:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        missing = parsed.isna()
        if missing.any():
            parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce")
        return parsed

    parsed = pd.to_datetime(text, format="%Y-%m-%d_%H.%M.%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce")
    return parsed


def csv_creation_like_timestamp(path: Path) -> Optional[pd.Timestamp]:
    try:
        stat = path.stat()
    except OSError:
        return None

    created = getattr(stat, "st_birthtime", None)
    if created is not None:
        return pd.Timestamp(datetime.fromtimestamp(created))
    return pd.Timestamp(datetime.fromtimestamp(stat.st_mtime))


def load_stage_dataframe(stage: StageSpec) -> pd.DataFrame:
    if not stage.csv_path.exists():
        return pd.DataFrame(columns=["basename", "file_timestamp", "execution_timestamp"])

    try:
        raw_df = pd.read_csv(stage.csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[definitive_execution_plotter] Failed to read {stage.csv_path}: {exc}",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=["basename", "file_timestamp", "execution_timestamp"])

    if raw_df.empty:
        return pd.DataFrame(columns=["basename", "file_timestamp", "execution_timestamp"])

    basename_col = first_existing_column(raw_df.columns, stage.basename_columns)
    if basename_col is None:
        return pd.DataFrame(columns=["basename", "file_timestamp", "execution_timestamp"])

    data = pd.DataFrame()
    data["basename"] = raw_df[basename_col].astype(str).str.strip()
    data["file_timestamp"] = pd.to_datetime(
        data["basename"].map(extract_timestamp_from_basename),
        errors="coerce",
    )

    execution_col = first_existing_column(raw_df.columns, stage.execution_columns)
    if execution_col is None:
        execution_ts = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    else:
        execution_ts = parse_execution_series(raw_df[execution_col], stage.execution_format)

    if stage.use_csv_timestamp_as_execution:
        csv_ts = csv_creation_like_timestamp(stage.csv_path)
        if csv_ts is not None:
            execution_ts = execution_ts.fillna(
                pd.Series(csv_ts, index=data.index, dtype="datetime64[ns]")
            )

    if stage.use_basename_as_execution:
        execution_ts = execution_ts.fillna(data["file_timestamp"])

    data["execution_timestamp"] = execution_ts
    data = data[
        data["basename"].ne("")
        & data["file_timestamp"].notna()
        & data["execution_timestamp"].notna()
    ]

    if data.empty:
        return pd.DataFrame(columns=["basename", "file_timestamp", "execution_timestamp"])

    grouped = (
        data.sort_values("execution_timestamp")
        .groupby("basename", as_index=False)
        .agg(
            file_timestamp=("file_timestamp", "first"),
            execution_timestamp=("execution_timestamp", "max"),
        )
    )
    return grouped


def stage_specs_for_station(station: str) -> List[StageSpec]:
    station_num = int(station[-2:])
    root = STATIONS_ROOT / station
    if station == "MINGO00":
        stage0_history_csv = root / "STAGE_0" / "SIMULATION" / "imported_basenames_history.csv"
        stage0_live_csv = root / "STAGE_0" / "SIMULATION" / "imported_basenames.csv"
        if stage0_history_csv.exists():
            stage0_csv = stage0_history_csv
            stage0_label = "STEP 0 - imported_basenames_history"
        else:
            stage0_csv = stage0_live_csv
            stage0_label = "STEP 0 - imported_basenames"
        stage0_execution_columns: Tuple[str, ...] = ("execution_time", "execution_timestamp")
    else:
        stage0_csv = (
            root
            / "STAGE_0"
            / "REPROCESSING"
            / "STEP_0"
            / "OUTPUT_FILES"
            / f"clean_remote_database_{station_num}.csv"
        )
        stage0_label = "STEP 0 - clean_remote_database"
        stage0_execution_columns = ()

    specs: List[StageSpec] = [
        StageSpec(
            index=0,
            label=stage0_label,
            color=STAGE_COLORS[0],
            csv_path=stage0_csv,
            basename_columns=("basename", "filename_base", "hld_name", "dat_name"),
            execution_columns=stage0_execution_columns,
            execution_format=None,
            use_csv_timestamp_as_execution=True,
        ),
    ]

    if station != "MINGO00":
        specs.extend(
            [
                StageSpec(
                    index=1,
                    label="STEP 1 - hld_files_brought",
                    color=STAGE_COLORS[1],
                    csv_path=root
                    / "STAGE_0"
                    / "REPROCESSING"
                    / "STEP_1"
                    / "METADATA"
                    / "hld_files_brought.csv",
                    basename_columns=("basename", "hld_name", "filename_base", "dat_name"),
                    execution_columns=("bring_timestamp", "bring_timesamp", "bring_time"),
                    execution_format=None,
                ),
                StageSpec(
                    index=2,
                    label="STEP 2 - dat_files_unpacked",
                    color=STAGE_COLORS[2],
                    csv_path=root
                    / "STAGE_0"
                    / "REPROCESSING"
                    / "STEP_2"
                    / "METADATA"
                    / "dat_files_unpacked.csv",
                    basename_columns=("basename", "dat_name", "filename_base", "hld_name"),
                    execution_columns=("execution_timestamp", "execution_time"),
                    execution_format=None,
                ),
            ]
        )

    task_stage_offset = 1 if station == "MINGO00" else 3
    for task_id in TASK_IDS:
        stage_index = task_stage_offset + (task_id - 1)
        specs.append(
            StageSpec(
                index=stage_index,
                label=f"TASK {task_id} - metadata_execution",
                color=STAGE_COLORS[stage_index],
                csv_path=root
                / "STAGE_1"
                / "EVENT_DATA"
                / "STEP_1"
                / f"TASK_{task_id}"
                / "METADATA"
                / f"task_{task_id}_metadata_execution.csv",
                basename_columns=("filename_base", "basename", "dat_name", "hld_name"),
                execution_columns=("execution_timestamp",),
                execution_format="%Y-%m-%d_%H.%M.%S",
            )
        )

    return specs


def _empty_stage_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["basename", "file_timestamp", "execution_timestamp"])


def build_station_stage_tables(stages: Sequence[StageSpec]) -> Dict[int, pd.DataFrame]:
    tables: Dict[int, pd.DataFrame] = {}
    ordered_stages = sorted(stages, key=lambda stage: stage.index)
    eligible_basenames: Optional[set[str]] = None

    for stage in ordered_stages:
        if eligible_basenames is not None and not eligible_basenames:
            tables[stage.index] = _empty_stage_dataframe()
            continue

        stage_df = load_stage_dataframe(stage)
        if eligible_basenames is not None and not stage_df.empty:
            stage_df = stage_df[stage_df["basename"].isin(eligible_basenames)]

        if stage_df.empty:
            eligible_basenames = set()
            tables[stage.index] = _empty_stage_dataframe()
            continue

        stage_df = stage_df.sort_values("execution_timestamp").reset_index(drop=True)
        current_stage_basenames = set(stage_df["basename"])
        if eligible_basenames is None:
            # Stage 0 defines the baseline universe of files that are allowed to appear.
            eligible_basenames = current_stage_basenames
        else:
            # A file can only advance if it is present in every previous stage.
            eligible_basenames = current_stage_basenames

        tables[stage.index] = stage_df

    for stage in ordered_stages:
        tables.setdefault(stage.index, _empty_stage_dataframe())
    return tables


def build_station_dataframe(
    stages: Sequence[StageSpec],
    stage_tables: Optional[Dict[int, pd.DataFrame]] = None,
) -> pd.DataFrame:
    empty_columns = [
        "basename",
        "file_timestamp",
        "execution_timestamp",
        "stage_index",
        "stage_label",
        "stage_color",
    ]

    if not stages:
        return pd.DataFrame(columns=empty_columns)

    if stage_tables is None:
        stage_tables = build_station_stage_tables(stages)

    ordered_stages = sorted(stages, key=lambda stage: stage.index)
    rows_by_basename: Dict[str, Dict[str, object]] = {}

    for stage in ordered_stages:
        stage_df = stage_tables.get(stage.index, _empty_stage_dataframe())
        if stage_df is None or stage_df.empty:
            continue

        for row in stage_df.itertuples(index=False):
            existing = rows_by_basename.get(row.basename)
            should_replace = (
                existing is None
                or stage.index > int(existing["stage_index"])
                or (
                    stage.index == int(existing["stage_index"])
                    and row.execution_timestamp > existing["execution_timestamp"]
                )
            )
            if not should_replace:
                continue

            rows_by_basename[row.basename] = {
                "basename": row.basename,
                "file_timestamp": row.file_timestamp,
                "execution_timestamp": row.execution_timestamp,
                "stage_index": stage.index,
                "stage_label": stage.label,
                "stage_color": stage.color,
            }

    if not rows_by_basename:
        return pd.DataFrame(columns=empty_columns)

    result = pd.DataFrame(rows_by_basename.values())
    return result.sort_values(["execution_timestamp", "file_timestamp"]).reset_index(drop=True)


def build_completeness_dataframe(
    stages: Sequence[StageSpec],
    stage_tables: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    ordered_stages = sorted(stages, key=lambda stage: stage.index)
    if not ordered_stages:
        return pd.DataFrame(columns=["execution_timestamp"])

    stage0_df = stage_tables.get(ordered_stages[0].index, _empty_stage_dataframe())
    denominator = int(stage0_df["basename"].nunique()) if not stage0_df.empty else 0
    if denominator <= 0:
        return pd.DataFrame(columns=["execution_timestamp"])

    time_chunks: List[pd.Series] = []
    for stage in ordered_stages:
        stage_df = stage_tables.get(stage.index, _empty_stage_dataframe())
        if stage_df.empty:
            continue
        ts = pd.to_datetime(stage_df["execution_timestamp"], errors="coerce").dropna()
        if not ts.empty:
            time_chunks.append(ts)

    if not time_chunks:
        return pd.DataFrame(columns=["execution_timestamp"])

    timeline = (
        pd.concat(time_chunks, ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if timeline.empty:
        return pd.DataFrame(columns=["execution_timestamp"])

    result = pd.DataFrame({"execution_timestamp": timeline})
    timeline_np = timeline.to_numpy(dtype="datetime64[ns]")
    stage_columns: List[str] = []
    for stage in ordered_stages:
        stage_col = f"stage_{stage.index}"
        stage_columns.append(stage_col)
        stage_df = stage_tables.get(stage.index, _empty_stage_dataframe())
        if stage_df.empty:
            result[stage_col] = 0.0
            continue

        stage_times = (
            pd.to_datetime(stage_df["execution_timestamp"], errors="coerce")
            .dropna()
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
        )
        if len(stage_times) == 0:
            result[stage_col] = 0.0
            continue

        counts = np.searchsorted(stage_times, timeline_np, side="right")
        result[stage_col] = (counts / float(denominator)) * 100.0

    if stage_columns:
        values = result[stage_columns].to_numpy(dtype=float)
        values = np.minimum.accumulate(values, axis=1)
        values = np.clip(values, 0.0, 100.0)
        result[stage_columns] = values

    return result


def _scatter_stage_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    stages: Sequence[StageSpec],
    point_size: float,
) -> None:
    for stage in stages:
        mask = data["stage_index"] == stage.index
        if not mask.any():
            continue
        stage_points = data.loc[mask]
        if stage.index == 0:
            ax.scatter(
                stage_points["file_timestamp"],
                stage_points["execution_timestamp"],
                s=point_size,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax.scatter(
                stage_points["file_timestamp"],
                stage_points["execution_timestamp"],
                s=point_size,
                facecolors=stage.color,
                edgecolors="black",
                linewidths=0.25,
                alpha=0.9,
                zorder=3,
            )


def _scatter_stage_presence_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    stages: Sequence[StageSpec],
    point_size: float,
) -> None:
    for stage in stages:
        mask = data["stage_index"] == stage.index
        if not mask.any():
            continue
        stage_points = data.loc[mask]
        y_values = np.zeros(len(stage_points), dtype=float)
        if stage.index == 0:
            ax.scatter(
                stage_points["file_timestamp"],
                y_values,
                s=point_size,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax.scatter(
                stage_points["file_timestamp"],
                y_values,
                s=point_size,
                facecolors=stage.color,
                edgecolors="black",
                linewidths=0.25,
                alpha=0.9,
                zorder=3,
            )


def add_stage_colorbar(fig: plt.Figure, axes: Sequence[plt.Axes], stages: Sequence[StageSpec]) -> None:
    colors = [stage.color for stage in stages]
    ticks = [stage.index for stage in stages]
    labels = [stage.label for stage in stages]
    bounds = [ticks[0] - 0.5] + [tick + 0.5 for tick in ticks]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cbar = fig.colorbar(
        mappable,
        ax=list(axes),
        boundaries=bounds,
        spacing="uniform",
        drawedges=True,
        ticks=ticks,
        fraction=0.025,
        pad=0.02,
    )
    cbar.ax.invert_yaxis()
    cbar.set_label("Pipeline stage (top earliest -> bottom latest)")
    cbar.ax.set_yticklabels(labels, fontsize=8)
    cbar.ax.text(
        1.45,
        1.01,
        "Earliest stage",
        transform=cbar.ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    cbar.ax.text(
        1.45,
        -0.03,
        "Latest stage",
        transform=cbar.ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    if cbar.solids is not None:
        cbar.solids.set_edgecolor("black")
        cbar.solids.set_linewidth(0.4)


def plot_completeness_fill(
    ax: plt.Axes,
    completeness_df: pd.DataFrame,
    stages: Sequence[StageSpec],
) -> None:
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.65)

    if completeness_df.empty:
        ax.text(
            0.5,
            0.5,
            "No completeness data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        return

    ordered_stages = sorted(stages, key=lambda stage: stage.index)
    y_values = pd.to_datetime(completeness_df["execution_timestamp"], errors="coerce")
    if y_values.dropna().empty:
        return

    for idx in range(len(ordered_stages) - 1, -1, -1):
        stage = ordered_stages[idx]
        upper_col = f"stage_{stage.index}"
        if upper_col not in completeness_df.columns:
            continue

        upper = completeness_df[upper_col].to_numpy(dtype=float)
        if idx == len(ordered_stages) - 1:
            lower = np.zeros_like(upper)
        else:
            lower_col = f"stage_{ordered_stages[idx + 1].index}"
            if lower_col not in completeness_df.columns:
                lower = np.zeros_like(upper)
            else:
                lower = completeness_df[lower_col].to_numpy(dtype=float)

        ax.fill_betweenx(
            y_values,
            lower,
            upper,
            facecolor=stage.color,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.95,
            step="post",
            zorder=2,
        )


def completeness_pie_payload(
    completeness_df: pd.DataFrame,
    stages: Sequence[StageSpec],
) -> Tuple[List[str], List[str], List[float]]:
    ordered_stages = sorted(stages, key=lambda stage: stage.index)
    if not ordered_stages:
        return [], [], []
    if completeness_df.empty:
        return [stage.label for stage in ordered_stages], [stage.color for stage in ordered_stages], [0.0] * len(ordered_stages)

    last_row = completeness_df.iloc[-1]
    cumulative: List[float] = []
    for stage in ordered_stages:
        cumulative.append(float(last_row.get(f"stage_{stage.index}", 0.0)))

    if cumulative:
        values = np.asarray(cumulative, dtype=float)
        values = np.clip(values, 0.0, 100.0)
        values = np.minimum.accumulate(values)
        cumulative = values.tolist()

    slices: List[float] = []
    for idx in range(len(ordered_stages)):
        if idx == len(ordered_stages) - 1:
            value = cumulative[idx]
        else:
            value = cumulative[idx] - cumulative[idx + 1]
        slices.append(max(0.0, float(value)))

    total = float(sum(slices))
    if total <= 0:
        return [stage.label for stage in ordered_stages], [stage.color for stage in ordered_stages], [0.0] * len(ordered_stages)

    normalized = [(value / total) * 100.0 for value in slices]
    labels = [stage.label for stage in ordered_stages]
    colors = [stage.color for stage in ordered_stages]
    return labels, colors, normalized


def plot_completeness_pie(
    ax: plt.Axes,
    completeness_df: pd.DataFrame,
    stages: Sequence[StageSpec],
) -> None:
    labels, colors, values = completeness_pie_payload(completeness_df, stages)
    ax.set_title("Completeness split (%)\n(STEP 0 sample only)")
    ax.set_aspect("equal")

    if not values or max(values) <= 0:
        ax.text(
            0.5,
            0.5,
            "No completeness data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        return

    ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="black", linewidth=0.5),
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.0 else "",
        pctdistance=0.75,
        textprops=dict(fontsize=7),
    )


def format_datetime_axes(top_ax: plt.Axes, middle_ax: plt.Axes, bottom_ax: plt.Axes) -> None:
    top_y_formatter = mdates.DateFormatter("%H:%M")
    middle_y_formatter = mdates.DateFormatter("%Y-%m-%d")
    top_ax.yaxis.set_major_formatter(top_y_formatter)
    middle_ax.yaxis.set_major_formatter(middle_y_formatter)

    x_formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M")
    bottom_ax.xaxis.set_major_formatter(x_formatter)
    bottom_ax.tick_params(axis="x", rotation=0)


def default_output_path() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR / DEFAULT_OUTPUT_FILENAME


def load_config(config_path: Path) -> Dict[str, object]:
    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse config JSON {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return loaded


def resolve_output_path(raw_output: Optional[str], config_path: Path) -> Path:
    if not raw_output:
        return default_output_path()
    candidate = Path(raw_output).expanduser()
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def resolve_runtime_options(
    args: argparse.Namespace,
) -> Tuple[
    List[str],
    float,
    float,
    Path,
    List[str],
    Tuple[float, float, float],
    float,
]:
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)

    if args.last_hours is not None:
        last_hours = float(args.last_hours)
    else:
        try:
            last_hours = float(config.get("last_hours", DEFAULT_LAST_HOURS))
        except (TypeError, ValueError):
            raise ValueError("'last_hours' in config must be numeric")

    if args.point_size is not None:
        point_size = float(args.point_size)
    else:
        try:
            point_size = float(config.get("point_size", DEFAULT_POINT_SIZE))
        except (TypeError, ValueError):
            raise ValueError("'point_size' in config must be numeric")

    output_raw = args.output
    if output_raw is None:
        config_output = config.get("output")
        output_raw = str(config_output) if config_output is not None else None
    output_path = resolve_output_path(output_raw, config_path)

    shared_x_raw = config.get("shared_x_stations", list(DEFAULT_SHARED_X_STATIONS))
    free_x_raw = config.get("free_x_stations", list(DEFAULT_FREE_X_STATIONS))

    if not isinstance(shared_x_raw, list):
        raise ValueError("'shared_x_stations' in config must be a list")
    if not isinstance(free_x_raw, list):
        raise ValueError("'free_x_stations' in config must be a list")

    shared_x_stations = normalize_existing_station_tokens([str(item) for item in shared_x_raw])
    free_x_stations = normalize_existing_station_tokens([str(item) for item in free_x_raw])

    config_stations_raw = config.get("stations")
    if args.stations is not None:
        stations = resolve_station_selection(args.stations)
    elif isinstance(config_stations_raw, list) and len(config_stations_raw) > 0:
        stations = resolve_station_selection([str(item) for item in config_stations_raw])
    else:
        grouped_scope = sorted(dict.fromkeys(shared_x_stations + free_x_stations))
        stations = grouped_scope if grouped_scope else list_available_stations()

    selected_set = set(stations)
    shared_effective = sorted(
        set(station for station in shared_x_stations if station in selected_set)
        - set(station for station in free_x_stations if station in selected_set)
    )

    panel_ratios_raw = config.get("panel_height_ratios", list(DEFAULT_PANEL_HEIGHT_RATIOS))
    if not isinstance(panel_ratios_raw, list):
        raise ValueError("'panel_height_ratios' in config must be a list")
    if len(panel_ratios_raw) != 3:
        raise ValueError("'panel_height_ratios' in config must have exactly 3 numeric values")
    try:
        panel_height_ratios = (
            float(panel_ratios_raw[0]),
            float(panel_ratios_raw[1]),
            float(panel_ratios_raw[2]),
        )
    except (TypeError, ValueError):
        raise ValueError("'panel_height_ratios' in config must contain numeric values")
    if any(value <= 0 for value in panel_height_ratios):
        raise ValueError("'panel_height_ratios' values must be > 0")

    try:
        middle_log_scale_seconds = float(
            config.get("middle_log_scale_seconds", DEFAULT_MIDDLE_LOG_SCALE_SECONDS)
        )
    except (TypeError, ValueError):
        raise ValueError("'middle_log_scale_seconds' in config must be numeric")
    if middle_log_scale_seconds <= 0:
        raise ValueError("'middle_log_scale_seconds' in config must be > 0")

    return (
        stations,
        last_hours,
        point_size,
        output_path,
        shared_effective,
        panel_height_ratios,
        middle_log_scale_seconds,
    )


def compute_shared_x_limits(
    station_data: Dict[str, pd.DataFrame],
    stations_to_share: Sequence[str],
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    min_values: List[pd.Timestamp] = []
    max_values: List[pd.Timestamp] = []

    for station in stations_to_share:
        data = station_data.get(station)
        if data is None or data.empty:
            continue
        min_values.append(pd.Timestamp(data["file_timestamp"].min()))
        max_values.append(pd.Timestamp(data["file_timestamp"].max()))

    if not min_values:
        return None

    shared_min = min(min_values)
    shared_max = max(max_values)
    if shared_min == shared_max:
        margin = timedelta(minutes=30)
    else:
        margin = (shared_max - shared_min) / 30
    return shared_min - margin, shared_max + margin


def plot_station_page(
    station: str,
    data: pd.DataFrame,
    completeness_df: pd.DataFrame,
    stages: Sequence[StageSpec],
    pdf: PdfPages,
    last_hours: float,
    point_size: float,
    panel_height_ratios: Tuple[float, float, float],
    middle_log_scale_seconds: float,
    x_limits_override: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> None:
    title = f"{station} - Definitive execution map"

    if data.empty:
        fig, ax = plt.subplots(figsize=(13, 5))
        fig.suptitle(title, fontsize=14)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No valid rows found in stage metadata files for this station.",
            ha="center",
            va="center",
            fontsize=11,
        )
        pdf_save_rasterized_page(pdf, fig, dpi=150)
        plt.close(fig)
        return

    fig_width = 16.6
    fig_height = 9.7
    top_ratio = max(float(panel_height_ratios[0]), 1e-6)
    middle_ratio = max(float(panel_height_ratios[1]), 1e-6)
    sum_height_ratios = max(float(sum(panel_height_ratios)), 1e-6)

    # Size the right column so top-right can be square without leaving unused horizontal gap.
    left_col_units = 4.0
    target_right_fraction = (fig_height / fig_width) * (top_ratio / sum_height_ratios)
    target_right_fraction = min(max(target_right_fraction, 0.05), 0.45)
    right_col_units = (target_right_fraction * left_col_units) / max(
        1e-6, (1.0 - target_right_fraction)
    )

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=list(panel_height_ratios),
        width_ratios=[left_col_units, right_col_units],
    )
    ax_zoom = fig.add_subplot(grid[0, 0])
    ax_zoom_comp = fig.add_subplot(grid[0, 1], sharey=ax_zoom)
    ax_full = fig.add_subplot(grid[1, 0], sharex=ax_zoom)
    ax_full_comp = fig.add_subplot(grid[1, 1], sharey=ax_full, sharex=ax_zoom_comp)
    ax_presence = fig.add_subplot(grid[2, 0], sharex=ax_zoom)
    ax_pie = fig.add_subplot(grid[2, 1])
    fig.suptitle(title, fontsize=14)

    _scatter_stage_points(ax_zoom, data, stages, point_size=point_size)
    _scatter_stage_points(ax_full, data, stages, point_size=point_size)
    _scatter_stage_presence_points(ax_presence, data, stages, point_size=point_size)
    plot_completeness_fill(ax_zoom_comp, completeness_df, stages)
    plot_completeness_fill(ax_full_comp, completeness_df, stages)
    plot_completeness_pie(ax_pie, completeness_df, stages)

    # Keep top-right square; make middle-right have the same X size as top-right.
    middle_comp_box_aspect = middle_ratio / top_ratio

    ax_zoom_comp.set_box_aspect(1)
    ax_full_comp.set_box_aspect(middle_comp_box_aspect)
    ax_pie.set_box_aspect(1)
    ax_zoom_comp.set_anchor("W")
    ax_full_comp.set_anchor("W")
    ax_pie.set_anchor("W")

    for ax in (ax_zoom, ax_full):
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.65)
        ax.set_ylabel("Execution time")
    ax_presence.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.65)
    ax_presence.set_ylabel("Presence")
    ax_zoom_comp.set_title("Completeness (%) - zoom")
    ax_full_comp.set_title("Completeness (%) - full")
    scope_note = "Scope: STEP 0 chain only (residual files excluded)"
    for ax in (ax_zoom_comp, ax_full_comp):
        ax.text(
            0.01,
            0.99,
            scope_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65),
        )
    ax_zoom_comp.tick_params(axis="x", labelbottom=False)
    ax_zoom_comp.tick_params(axis="y", labelleft=False)
    ax_full_comp.tick_params(axis="y", labelleft=False)
    ax_full_comp.set_xlabel("Completeness (%)")

    ax_zoom.set_title(f"Last {last_hours:g} hours from now (UTC)")
    ax_full.set_title("Full execution history (soft log-scaled by recency)")
    ax_presence.set_title("Presence vs data time (execution-time independent)")
    ax_presence.set_xlabel("Data time (from basename)")
    ax_presence.set_yticks([0])
    ax_presence.set_yticklabels(["present"])
    ax_presence.set_ylim(-0.8, 0.8)

    x_min = data["file_timestamp"].min()
    x_max = data["file_timestamp"].max()
    y_min = data["execution_timestamp"].min()
    y_max = data["execution_timestamp"].max()
    if y_min.tzinfo is None:
        now = pd.Timestamp.utcnow().tz_localize(None)
    else:
        now = pd.Timestamp.utcnow().tz_convert(y_min.tzinfo)

    if x_limits_override is None:
        if x_min == x_max:
            x_margin = timedelta(minutes=30)
        else:
            x_margin = (x_max - x_min) / 30
        x_limits = (x_min - x_margin, x_max + x_margin)
    else:
        x_limits = x_limits_override

    full_y_margin = timedelta(minutes=3)
    zoom_y_margin = timedelta(minutes=1)
    zoom_lower = now - timedelta(hours=last_hours)

    for ax in (ax_zoom, ax_full, ax_presence):
        ax.set_xlim(*x_limits)

    full_y_lower = min(y_min, now) - full_y_margin
    now_visible_upper = now + timedelta(minutes=NOW_Y_MARGIN_MINUTES)
    full_y_upper = max(y_max + timedelta(seconds=1), now_visible_upper)
    ax_full.set_ylim(full_y_lower, full_y_upper)
    zoom_upper = max(y_max + zoom_y_margin, now_visible_upper)
    if zoom_lower >= zoom_upper:
        zoom_lower = zoom_upper - timedelta(minutes=5)
    ax_zoom.set_ylim(zoom_lower, zoom_upper)

    # Middle panel: log-scale by recency (age) so recent executions get more visual space.
    middle_ref_num = mdates.date2num(full_y_upper)
    scale_seconds = float(middle_log_scale_seconds)

    def _middle_y_forward(values):
        arr = np.asarray(values, dtype=float)
        age_seconds = np.maximum(0.0, (middle_ref_num - arr) * 86400.0)
        return -np.log10((age_seconds / scale_seconds) + 1.0)

    def _middle_y_inverse(values):
        arr = np.asarray(values, dtype=float)
        age_seconds = (np.power(10.0, -arr) - 1.0) * scale_seconds
        return middle_ref_num - (age_seconds / 86400.0)

    ax_full.set_yscale("function", functions=(_middle_y_forward, _middle_y_inverse))

    format_datetime_axes(ax_zoom, ax_full, ax_presence)
    for ax in (ax_zoom, ax_full, ax_zoom_comp, ax_full_comp):
        ax.axhline(now, color="red", linestyle="--", alpha=0.3, zorder=10)

    # Re-apply after axis transforms so layout is preserved.
    ax_zoom_comp.set_box_aspect(1)
    ax_full_comp.set_box_aspect(middle_comp_box_aspect)
    ax_pie.set_box_aspect(1)
    ax_zoom_comp.set_anchor("W")
    ax_full_comp.set_anchor("W")
    ax_pie.set_anchor("W")

    stage_counts = data["stage_index"].value_counts().to_dict()
    count_lines = [f"Total files: {len(data)}"]
    for stage in stages:
        count_lines.append(f"{stage.label}: {int(stage_counts.get(stage.index, 0))}")

    ax_full.text(
        0.01,
        0.99,
        "\n".join(count_lines),
        transform=ax_full.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
    )

    add_stage_colorbar(
        fig,
        (ax_zoom, ax_zoom_comp, ax_full, ax_full_comp, ax_presence),
        stages,
    )
    pdf_save_rasterized_page(pdf, fig, dpi=160)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-station definitive execution maps "
            "(file-time on X, execution-time on Y)."
        )
    )
    parser.add_argument(
        "-s",
        "--stations",
        nargs="+",
        metavar="STATION",
        help="Stations to include (e.g. 1, 01, MINGO01). Defaults to all detected stations.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"JSON config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--last-hours",
        type=float,
        default=None,
        help="Height of the top zoom panel in hours (overrides config).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=None,
        help="Scatter marker size (overrides config).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional output PDF path (overrides config).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_matplotlib_style()

    try:
        (
            stations,
            last_hours,
            point_size,
            output_path,
            shared_x_stations,
            panel_height_ratios,
            middle_log_scale_seconds,
        ) = resolve_runtime_options(args)
    except ValueError as exc:
        print(f"[definitive_execution_plotter] {exc}", file=sys.stderr)
        return 1

    if last_hours <= 0:
        print("[definitive_execution_plotter] --last-hours must be > 0", file=sys.stderr)
        return 1
    if point_size <= 0:
        print("[definitive_execution_plotter] --point-size must be > 0", file=sys.stderr)
        return 1

    if not stations:
        print("[definitive_execution_plotter] No stations found to process.", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    station_stages: Dict[str, List[StageSpec]] = {}
    station_data: Dict[str, pd.DataFrame] = {}
    station_completeness: Dict[str, pd.DataFrame] = {}
    for station in stations:
        stages = stage_specs_for_station(station)
        station_stages[station] = stages
        stage_tables = build_station_stage_tables(stages)
        station_data[station] = build_station_dataframe(stages, stage_tables=stage_tables)
        station_completeness[station] = build_completeness_dataframe(stages, stage_tables)

    shared_x_limits = compute_shared_x_limits(station_data, shared_x_stations)
    shared_x_station_set = set(shared_x_stations)

    with PdfPages(output_path) as pdf:
        for station in stations:
            stages = station_stages[station]
            station_df = station_data[station]
            completeness_df = station_completeness[station]
            x_limits_override = shared_x_limits if station in shared_x_station_set else None
            plot_station_page(
                station=station,
                data=station_df,
                completeness_df=completeness_df,
                stages=stages,
                pdf=pdf,
                last_hours=last_hours,
                point_size=point_size,
                panel_height_ratios=panel_height_ratios,
                middle_log_scale_seconds=middle_log_scale_seconds,
                x_limits_override=x_limits_override,
            )

    print(f"[definitive_execution_plotter] Saved PDF to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
