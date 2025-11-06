#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import cdsapi
import pandas as pd
import xarray as xr
import yaml

CURRENT_PATH = Path(__file__).resolve()
REPO_ROOT = None
for parent in CURRENT_PATH.parents:
    if parent.name == "MASTER":
        REPO_ROOT = parent.parent
        break
if REPO_ROOT is None:
    REPO_ROOT = CURRENT_PATH.parents[-1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.execution_logger import set_station, start_timer
from MASTER.common.status_csv import append_status_row, mark_status_complete


def print_banner() -> None:
    print("__| |____________________________________________________________| |__")
    print("__   ____________________________________________________________   __")
    print("  | |                                                            | |  ")
    print("  | |                            _           _                   | |  ")
    print(
        "  | | _ __ ___  __ _ _ __   __ _| |_   _ ___(_)___   _ __  _   _ | |  "
    )
    print(
        "  | || '__/ _ \\/ _` | '_ \\ / _` | | | | / __| / __| | '_ \\| | | || |  "
    )
    print(
        "  | || | |  __/ (_| | | | | (_| | | |_| \\__ \\ \\__ \\_| |_) | |_| || |  "
    )
    print(
        "  | ||_|  \\___|\\__,_|_| |_|\\__,_|_|\\__, |___/_|___(_) .__/ \\__, || |  "
    )
    print(
        "  | |                              |___/            |_|    |___/ | |  "
    )
    print("__| |____________________________________________________________| |__")
    print("__   ____________________________________________________________   __")
    print("  | |                                                            | |  ")


def parse_existing_days(output_dir: Path) -> Set[date]:
    existing_days: Set[date] = set()
    if not output_dir.exists():
        return existing_days
    for file_path in output_dir.rglob("copernicus_*.csv"):
        if not file_path.is_file():
            continue
        parts = file_path.stem.split("_")
        if len(parts) != 4:
            continue
        try:
            existing_days.add(date(int(parts[1]), int(parts[2]), int(parts[3])))
        except ValueError:
            continue
    return existing_days


def rebuild_big_csv(output_dir: Path, destination: Path) -> None:
    if not output_dir.exists():
        if destination.exists():
            destination.unlink()
        return

    frames: List[pd.DataFrame] = []
    for csv_file in sorted(output_dir.rglob("copernicus_*.csv")):
        try:
            df = pd.read_csv(csv_file, parse_dates=["Time"])
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"Warning: unable to load {csv_file}: {exc}")
            continue
        frames.append(df)

    if not frames:
        if destination.exists():
            destination.unlink()
        return

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="Time", keep="last")
        .sort_values("Time")
    )
    combined.to_csv(destination, index=False, float_format="%.5g")


def write_daily_outputs(df: pd.DataFrame, output_dir: Path) -> List[date]:
    written_days: List[date] = []
    if df.empty:
        return written_days

    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])

    for day_key, day_frame in df.groupby(df["Time"].dt.date):
        if day_frame.empty:
            continue

        month_dir = output_dir / f"{day_key:%Y}" / f"{day_key:%m}"
        month_dir.mkdir(parents=True, exist_ok=True)
        output_path = month_dir / f"copernicus_{day_key:%Y_%m_%d}.csv"
        day_frame = (
            day_frame.sort_values("Time")
            .drop_duplicates(subset="Time", keep="last")
        )

        if output_path.exists():
            try:
                existing = pd.read_csv(output_path, parse_dates=["Time"])
            except Exception as exc:  # pragma: no cover - best effort logging
                print(f"Warning: unable to load existing {output_path}: {exc}")
                existing = pd.DataFrame()
            day_frame = (
                pd.concat([existing, day_frame], ignore_index=True)
                .drop_duplicates(subset="Time", keep="last")
                .sort_values("Time")
            )

        day_frame.to_csv(output_path, index=False, float_format="%.5g")
        written_days.append(day_key)

    return written_days


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def daterange(start_day: date, end_day: date) -> List[date]:
    span = (end_day - start_day).days
    return [start_day + timedelta(days=offset) for offset in range(span + 1)]


def main() -> int:
    print_banner()

    start_timer(__file__)

    user_home = os.path.expanduser("~")
    config_file_path = Path(user_home) / "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml"
    print(f"Using config file: {config_file_path}")

    with config_file_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    test_mode = config["test_mode"]
    weeks_behind_requested = config["weeks_behind_requested"]
    max_weeks_allowed = config["max_weeks_allowed"]
    degree_apotema = config["degree_apotema"]

    if len(sys.argv) < 2:
        print("Error: No station provided.")
        print("Usage: python3 copernicus.py <station>")
        return 1

    station = int(sys.argv[1])
    print(f"Station: {station}")
    set_station(station)

    locations = {
        1: {"name": "Madrid", "latitude": 40.4168, "longitude": -3.7038},
        2: {"name": "Warsaw", "latitude": 52.2297, "longitude": 21.0122},
        3: {"name": "Puebla", "latitude": 19.0413, "longitude": -98.2062},
        4: {"name": "Monterrey", "latitude": 25.6866, "longitude": -100.3161},
    }

    if station not in locations:
        print(f"Invalid station number: {station}")
        return 1

    location = locations[station]["name"]
    latitude = locations[station]["latitude"]
    longitude = locations[station]["longitude"]

    home_path = Path(config["home_path"]).expanduser()
    station_dir = home_path / "DATAFLOW_v3" / "STATIONS" / f"MINGO0{station}"
    copernicus_root = station_dir / "STAGE_1" / "COPERNICUS"
    step1_root = copernicus_root / "STEP_1"
    input_root = step1_root / "INPUT_FILES"
    netcdf_root = input_root / "NETCDF"
    output_root = step1_root / "OUTPUT_FILES"

    ensure_directories([input_root, netcdf_root, output_root])

    # status_csv_path = copernicus_root / "copernicus_status.csv"
    # status_timestamp = append_status_row(status_csv_path)

    existing_days = parse_existing_days(output_root)

    if test_mode:
        start_day = (datetime.now() - timedelta(weeks=weeks_behind_requested)).date()
    else:
        start_day = date(2023, 7, 1)

    end_day = (datetime.now() - timedelta(days=5)).date()
    if start_day > end_day:
        start_day = end_day

    requested_days = daterange(start_day, end_day)
    missing_days = [day for day in requested_days if day not in existing_days]

    if not missing_days:
        print("Copernicus data already up to date for the requested range. Nothing to do.")
        rebuild_big_csv(output_root, copernicus_root / "big_copernicus_data.csv")
        # mark_status_complete(status_csv_path, status_timestamp)
        return 0

    fetch_start = missing_days[0]
    fetch_end = missing_days[-1]

    start_datetime = datetime.combine(fetch_start, datetime.min.time())
    end_datetime = datetime.combine(fetch_end, datetime.min.time())

    print(
        f"\nRetrieving data from {start_datetime:%Y-%m-%d} "
        f"to {end_datetime:%Y-%m-%d}\n"
    )

    client = cdsapi.Client()

    frames_ground: List[pd.DataFrame] = []
    frames_temp100: List[pd.DataFrame] = []
    frames_geopot100: List[pd.DataFrame] = []

    times = [f"{hour:02d}:00" for hour in range(24)]

    for day_value in missing_days:
        print("\n--------------------------------------------------------------")
        print(f"Processing {day_value:%Y-%m-%d}")

        ground_file = (
            netcdf_root / f"{location}_2m_temperature_{day_value:%Y%m%d}.nc"
        )
        pressure_file = (
            netcdf_root / f"{location}_100mbar_temperature_geopotential_{day_value:%Y%m%d}.nc"
        )

        if ground_file.exists() and pressure_file.exists():
            print("  NetCDF files already present; reusing cached downloads.")
        else:
            if not ground_file.exists():
                print("  Downloading ground level temperature")
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": ["2m_temperature"],
                        "year": f"{day_value:%Y}",
                        "month": f"{day_value:%m}",
                        "day": f"{day_value:%d}",
                        "time": times,
                        "area": [
                            latitude + degree_apotema,
                            longitude - degree_apotema,
                            latitude - degree_apotema,
                            longitude + degree_apotema,
                        ],
                        "format": "netcdf",
                    },
                    str(ground_file),
                )
            else:
                print("  Ground level NetCDF already present.")

            if not pressure_file.exists():
                print("  Downloading 100 mbar temperature & geopotential")
                client.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": ["temperature", "geopotential"],
                        "pressure_level": ["100"],
                        "year": f"{day_value:%Y}",
                        "month": f"{day_value:%m}",
                        "day": f"{day_value:%d}",
                        "time": times,
                        "area": [
                            latitude + degree_apotema,
                            longitude - degree_apotema,
                            latitude - degree_apotema,
                            longitude + degree_apotema,
                        ],
                        "format": "netcdf",
                    },
                    str(pressure_file),
                )
            else:
                print("  Pressure level NetCDF already present.")

        ds_2m = xr.open_dataset(ground_file).rename({"valid_time": "Time"})
        ds_100 = xr.open_dataset(pressure_file).rename({"valid_time": "Time"})

        df_ground = (ds_2m["t2m"] - 273.15).to_dataframe().reset_index()
        df_temp100 = (ds_100["t"] - 273.15).to_dataframe().reset_index()
        df_geopot100 = (ds_100["z"] / 9.80665).to_dataframe().reset_index()

        frames_ground.append(
            df_ground.groupby("Time", as_index=False).mean(numeric_only=True)
        )
        frames_temp100.append(
            df_temp100.groupby("Time", as_index=False).mean(numeric_only=True)
        )
        frames_geopot100.append(
            df_geopot100.groupby("Time", as_index=False).mean(numeric_only=True)
        )

        ds_2m.close()
        ds_100.close()

    if not frames_ground:
        print("No data retrieved from Copernicus.")
        # mark_status_complete(status_csv_path, status_timestamp)
        return 0

    df_ground_all = pd.concat(frames_ground, ignore_index=True)
    df_temp100_all = pd.concat(frames_temp100, ignore_index=True)
    df_geopot100_all = pd.concat(frames_geopot100, ignore_index=True)

    df_new = (
        df_ground_all.merge(df_temp100_all, on="Time")
        .merge(df_geopot100_all, on="Time")
        .loc[:, ["Time", "t2m", "t", "z"]]
        .rename(
            columns={
                "t2m": "temp_ground",
                "t": "temp_100mbar",
                "z": "height_100mbar",
            }
        )
    )

    days_written = write_daily_outputs(df_new, output_root)
    if days_written:
        print(
            "Daily files written: "
            + ", ".join(sorted(day.strftime("%Y-%m-%d") for day in days_written))
        )
    else:
        print("No new daily files were produced.")

    # rebuild_big_csv(output_root, copernicus_root / "big_copernicus_data.csv")

    # mark_status_complete(status_csv_path, status_timestamp)

    print("\n--------------------------- python copernicus ends ---------------------------")
    print(
        "------------------------------------------------------\n"
        f"copernicus.py (Copernicus) completed on: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        "------------------------------------------------------"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
