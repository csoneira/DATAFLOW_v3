#!/usr/bin/env python3
"""Plot HLD files brought timelines for each station."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


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
CSV_RELATIVE_PATH = Path("STAGE_0/REPROCESSING/STEP_1/METADATA/hld_files_brought.csv")
PLOTTER_DIR = SCRIPT_PATH.parent
PLOTS_DIR = PLOTTER_DIR / "PLOTS"
DEFAULT_FILENAME_TEMPLATE = "files_brought_report_{mode}.pdf"

CANONICAL_COLUMNS = {
    "basename": ("basename", "hld_name", "hld_basename", "file_name", "filename", "name"),
    "bring_timestamp": (
        "bring_timestamp",
        "bring_timesamp",
        "bring_time",
        "bring_datetime",
        "bring_date",
        "bring_ts",
    ),
}

BASENAME_TIMESTAMP_DIGITS = 11


@dataclass
class StationPayload:
    station: str
    dataframe: pd.DataFrame
    csv_path: Path
    total_rows: int
    parsed_rows: int
    error: Optional[str] = None


def configure_matplotlib_style() -> None:
    plt.style.use("default")


def normalize_station_token(token: str) -> Optional[str]:
    cleaned = token.strip().upper()
    if not cleaned:
        return None
    if cleaned.startswith("MINGO"):
        suffix = cleaned[5:]
    else:
        suffix = cleaned
    digits = "".join(ch for ch in suffix if ch.isdigit())
    if not digits:
        return None
    number = int(digits)
    return f"MINGO{number:02d}"


def list_available_stations() -> List[str]:
    if not STATIONS_ROOT.exists():
        return []
    stations: List[str] = []
    for entry in STATIONS_ROOT.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name.upper()
        if re.fullmatch(r"MINGO\d{2}", name):
            stations.append(name)
    stations.sort()
    return stations


def resolve_station_selection(tokens: Sequence[str]) -> List[str]:
    available = list_available_stations()
    if not tokens:
        return available

    normalized: List[str] = []
    invalid: List[str] = []
    for token in tokens:
        station = normalize_station_token(token)
        if not station:
            invalid.append(token)
            continue
        if station not in available:
            invalid.append(token)
            continue
        normalized.append(station)

    if invalid:
        joined = ", ".join(invalid)
        print(f"[files_brought_plot] Ignoring unknown station(s): {joined}", file=sys.stderr)

    deduped = sorted(dict.fromkeys(normalized))
    return deduped


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for canonical, candidates in CANONICAL_COLUMNS.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = canonical
                break
    return df.rename(columns=rename_map)


def extract_timestamp_from_basename(basename: str) -> Optional[datetime]:
    if not basename:
        return None
    digits = "".join(ch for ch in basename if ch.isdigit())
    if len(digits) < BASENAME_TIMESTAMP_DIGITS:
        return None
    stamp = digits[-BASENAME_TIMESTAMP_DIGITS :]
    try:
        year = 2000 + int(stamp[0:2])
        day_of_year = int(stamp[2:5])
        hour = int(stamp[5:7])
        minute = int(stamp[7:9])
        second = int(stamp[9:11])
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
    delta = timedelta(days=day_of_year - 1, hours=hour, minutes=minute, seconds=second)
    return base + delta


def load_station_payload(station: str) -> StationPayload:
    csv_path = STATIONS_ROOT / station / CSV_RELATIVE_PATH
    if not csv_path.exists():
        return StationPayload(
            station=station,
            dataframe=pd.DataFrame(),
            csv_path=csv_path,
            total_rows=0,
            parsed_rows=0,
            error="Metadata CSV not found",
        )
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        return StationPayload(
            station=station,
            dataframe=pd.DataFrame(),
            csv_path=csv_path,
            total_rows=0,
            parsed_rows=0,
            error=f"Failed to read CSV: {exc}",
        )

    total_rows = len(df)
    df = rename_columns(df)
    if "basename" not in df.columns or "bring_timestamp" not in df.columns:
        missing = []
        if "basename" not in df.columns:
            missing.append("basename")
        if "bring_timestamp" not in df.columns:
            missing.append("bring_timestamp")
        reason = f"Missing column(s): {', '.join(missing)}"
        return StationPayload(
            station=station,
            dataframe=pd.DataFrame(),
            csv_path=csv_path,
            total_rows=total_rows,
            parsed_rows=0,
            error=reason,
        )

    df["basename"] = df["basename"].astype(str).str.strip()
    df["bring_timestamp"] = pd.to_datetime(df["bring_timestamp"], errors="coerce")
    df["file_timestamp"] = pd.to_datetime(
        df["basename"].apply(extract_timestamp_from_basename), errors="coerce"
    )
    df = df.dropna(subset=["bring_timestamp", "file_timestamp"])

    if df.empty:
        return StationPayload(
            station=station,
            dataframe=df,
            csv_path=csv_path,
            total_rows=total_rows,
            parsed_rows=0,
            error="No rows with valid timestamps",
        )

    df = df.sort_values("bring_timestamp").reset_index(drop=True)
    return StationPayload(
        station=station,
        dataframe=df,
        csv_path=csv_path,
        total_rows=total_rows,
        parsed_rows=len(df),
        error=None,
    )


def axis_configuration(use_real_time: bool) -> Tuple[str, str, str, str]:
    if use_real_time:
        return (
            "file_timestamp",
            "bring_timestamp",
            "File timestamp (real time)",
            "Bring timestamp (execution time)",
        )
    return (
        "bring_timestamp",
        "file_timestamp",
        "Bring timestamp (execution time)",
        "File timestamp (real time)",
    )


def default_output_path(use_real_time: bool) -> Path:
    mode = "real_time" if use_real_time else "execution_time"
    PLOTS_DIR.mkdir -p(parents=True, exist_ok=True)
    return PLOTS_DIR / DEFAULT_FILENAME_TEMPLATE.format(mode=mode)


def plot_station_payload(
    payload: StationPayload,
    pdf: PdfPages,
    use_real_time: bool,
) -> None:
    x_column, y_column, x_label, y_label = axis_configuration(use_real_time)

    fig_title = f"{payload.station} – Files Brought"
    if payload.error and payload.dataframe.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        fig.suptitle(fig_title, fontsize=14)
        ax.axis("off")
        msg = f"{payload.error}\nSource: {payload.csv_path}"
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, wrap=True)
        pdf_save_rasterized_page(pdf, fig, dpi=150)
        plt.close(fig)
        return

    df = payload.dataframe
    if df.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        fig.suptitle(fig_title, fontsize=14)
        ax.axis("off")
        msg = (
            "No rows with both bring_timestamp and basename timestamps.\n"
            f"Source: {payload.csv_path}"
        )
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, wrap=True)
        pdf_save_rasterized_page(pdf, fig, dpi=150)
        plt.close(fig)
        return

    x_series = df[x_column]
    y_series = df[y_column]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig.suptitle(fig_title, fontsize=14)

    ax.scatter(
        x_series,
        y_series,
        color="tab:blue",
        s=18,
        alpha=0.85,
        edgecolors="none",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    info_lines = [
        f"CSV rows: {payload.total_rows}",
        f"Plotted rows: {payload.parsed_rows}",
        f"{x_label}: {x_series.min():%Y-%m-%d %H:%M} → {x_series.max():%Y-%m-%d %H:%M}",
        f"{y_label}: {y_series.min():%Y-%m-%d %H:%M} → {y_series.max():%Y-%m-%d %H:%M}",
    ]
    ax.text(
        0.01,
        0.99,
        "\n".join(info_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7),
    )

    pdf_save_rasterized_page(pdf, fig, dpi=150)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PDF plots comparing bring timestamps vs file timestamps per station.",
    )
    parser.add_argument(
        "-s",
        "--stations",
        nargs="+",
        metavar="STATION",
        help="Stations to include (e.g. 1, 01, MINGO01). Defaults to every detected station.",
    )
    axes_group = parser.add_mutually_exclusive_group()
    axes_group.add_argument(
        "-r",
        "--real-time",
        action="store_true",
        help="Use file timestamps (real time) on the X axis.",
    )
    axes_group.add_argument(
        "-e",
        "--execution-time",
        action="store_true",
        help="Use bring timestamps (execution time) on the X axis (default).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional output PDF path. Defaults to the plotter PLOTS directory.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_matplotlib_style()

    use_real_time = bool(args.real_time)
    stations = resolve_station_selection(args.stations or [])
    if not stations:
        print("[files_brought_plot] No stations found to process.", file=sys.stderr)
        return 1

    output_path = Path(args.output).expanduser() if args.output else default_output_path(use_real_time)
    output_path.parent.mkdir -p(parents=True, exist_ok=True)

    payloads = [load_station_payload(station) for station in stations]

    with PdfPages(output_path) as pdf:
        for payload in payloads:
            plot_station_payload(payload, pdf, use_real_time=use_real_time)

    print(f"[files_brought_plot] Saved PDF to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
