#!/usr/bin/env python3
"""Utilities to generate and persist time-calibration itineraries."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(os.getenv("DATAFLOW_PROJECT_ROOT", Path.home() / "DATAFLOW_v3")).resolve()
CONFIG_DIR = PROJECT_ROOT / "MASTER" / "CONFIG_FILES"
GLOBAL_CONFIG_PATH = CONFIG_DIR / "config_global.yaml"


def _load_home_path(config_path: Path) -> Path:
    """Return the `home_path` declared in *config_path* or fall back to `~`."""
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as config_file:
            try:
                config_data = yaml.safe_load(config_file) or {}
            except yaml.YAMLError as config_error:
                raise RuntimeError(
                    f"Unable to parse configuration file {config_path}"
                ) from config_error
            home_path_value = config_data.get("home_path")
            if home_path_value:
                return Path(home_path_value).expanduser()
    return Path.home()


HOME_PATH = _load_home_path(GLOBAL_CONFIG_PATH)

ITINERARY_FILE_PATH = (
    HOME_PATH
    / "DATAFLOW_v3"
    / "MASTER"
    / "ANCILLARY"
    / "INPUT_FILES"
    / "TIME_CALIBRATION_ITINERARIES"
    / "itineraries.csv"
)




def load_itineraries_from_file(file_path: Path, required: bool = True) -> list[list[str]]:
    """Return itineraries stored as comma-separated lines in *file_path*."""
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"Cannot find itineraries file: {file_path}")
        return []

    itineraries: list[list[str]] = []
    with file_path.open("r", encoding="utf-8") as itinerary_file:
        print(f"Loading itineraries from {file_path}:")
        for line_number, raw_line in enumerate(itinerary_file, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            segments = [segment.strip() for segment in stripped_line.split(",") if segment.strip()]
            if segments:
                itineraries.append(segments)
                print(segments)

    if not itineraries and required:
        raise ValueError(f"Itineraries file {file_path} is empty.")

    return itineraries



def write_itineraries_to_file(
    file_path: Path,
    itineraries: Iterable[Iterable[str]],
) -> None:
    """Persist unique itineraries to *file_path* as comma-separated lines."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    unique_itineraries: dict[tuple[str, ...], None] = {}

    for itinerary in itineraries:
        itinerary_tuple = tuple(itinerary)
        if not itinerary_tuple:
            continue
        unique_itineraries.setdefault(itinerary_tuple, None)

    with file_path.open("w", encoding="utf-8") as itinerary_file:
        for itinerary_tuple in unique_itineraries:
            itinerary_file.write(",".join(itinerary_tuple) + "\n")


# Main itinerary
itinerary = ["P1s1", "P3s1", "P1s2", "P3s2", "P1s3", "P3s3", "P1s4", "P3s4", "P4s4", "P2s4", "P4s3", "P2s3", "P4s2", "P2s2", "P4s1", "P2s1"]
k = 0
max_iter = 2000000
brute_force_list = []
# Create row and column indices
rows = ['P{}'.format(i) for i in range(1, 5)]
columns = ['s{}'.format(i) for i in range(1,5)]
brute_force_df = pd.DataFrame(0, index=rows, columns=columns)
jump = False

existing_itineraries = load_itineraries_from_file(ITINERARY_FILE_PATH, required=False)
successful_itineraries: dict[tuple[str, ...], None] = {
    tuple(existing_itinerary): None
    for existing_itinerary in existing_itineraries
}
found_new_itinerary = False

while k < max_iter:
    if k % 50000 == 0: print(f"Itinerary {k}")
    brute_force_df[brute_force_df.columns] = 0
    step = itinerary
    a = []
    for i in range(len(itinerary)):
        if i > 0:
            # Storing new values
            a.append( df[step[i - 1]][step[i]] )
        relative_time = sum(a)
        if np.isnan(relative_time):
            jump = True
            break
        ind1 = str(step[i][0:2])
        ind2 = str(step[i][2:4])
        brute_force_df.loc[ind1,ind2] = brute_force_df.loc[ind1,ind2] + relative_time
    # If the path is succesful, print it, then we can copy it from terminal
    # and save it for the next step.
    if jump == False:
        print(itinerary)
        itinerary_tuple = tuple(step)
        if itinerary_tuple not in successful_itineraries:
            successful_itineraries[itinerary_tuple] = None
            found_new_itinerary = True
    # Shuffle the path
    random.shuffle(itinerary)
    # Iterate
    k += 1
    if jump:
        jump = False
        continue
    # Substract a value from the entire DataFrame
    brute_force_df = brute_force_df.sub(brute_force_df.iloc[0, 0])
    # Append the matrix to the big list
    brute_force_list.append(brute_force_df.values)
# Calculate the mean of all the paths
calibrated_times_bf = np.nanmean(brute_force_list, axis=0)
calibration_times = calibrated_times_bf

if successful_itineraries and (found_new_itinerary or not ITINERARY_FILE_PATH.exists()):
    write_itineraries_to_file(ITINERARY_FILE_PATH, successful_itineraries.keys())
