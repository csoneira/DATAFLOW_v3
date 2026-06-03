from __future__ import annotations

import re

import pandas as pd


RAW_CHANNEL_COLUMN_INDICES = {
    "T1_F": range(55, 59),
    "T1_B": range(59, 63),
    "Q1_F": range(63, 67),
    "Q1_B": range(67, 71),
    "T2_F": range(39, 43),
    "T2_B": range(43, 47),
    "Q2_F": range(47, 51),
    "Q2_B": range(51, 55),
    "T3_F": range(23, 27),
    "T3_B": range(27, 31),
    "Q3_F": range(31, 35),
    "Q3_B": range(35, 39),
    "T4_F": range(7, 11),
    "T4_B": range(11, 15),
    "Q4_F": range(15, 19),
    "Q4_B": range(19, 23),
}


def station_matches_file(file_name: str, station: str) -> bool:
    match = re.match(r"^mi0(?P<station>[0-9I]).*\.dat$", file_name, re.IGNORECASE)
    if match is None:
        return False
    label = match.group("station")
    file_station = "1" if label.upper() == "I" else label
    return int(file_station) == int(station)


def raw_channel_rename_map() -> dict[str, str]:
    rename_map: dict[str, str] = {}
    for key, idx_range in RAW_CHANNEL_COLUMN_INDICES.items():
        for strip, col_idx in enumerate(idx_range, start=1):
            rename_map[f"column_{col_idx}"] = f"{key}_{strip}"
    return rename_map


def datetime_bounds(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty or "datetime" not in frame.columns:
        return "", ""
    values = pd.to_datetime(frame["datetime"], errors="coerce").dropna()
    if values.empty:
        return "", ""
    return str(values.iloc[0]), str(values.iloc[-1])


def duration_seconds(frame: pd.DataFrame) -> int:
    if frame.empty or "datetime" not in frame.columns:
        return 0
    values = pd.to_datetime(frame["datetime"], errors="coerce").dropna()
    if values.empty:
        return 0
    return max(0, int((values.iloc[-1] - values.iloc[0]).total_seconds()))


def rate_hz(count: int, duration_seconds_value: int) -> float:
    if duration_seconds_value <= 0:
        return 0.0
    return round(float(count) / float(duration_seconds_value), 6)


def compute_acq_tt(df: pd.DataFrame, column_name: str = "acq_tt") -> pd.DataFrame:
    tt_str = pd.Series("", index=df.index, dtype="object")
    for plane in range(1, 5):
        charge_columns = [
            col
            for col in [
                f"Q{plane}_F_1",
                f"Q{plane}_F_2",
                f"Q{plane}_F_3",
                f"Q{plane}_F_4",
                f"Q{plane}_B_1",
                f"Q{plane}_B_2",
                f"Q{plane}_B_3",
                f"Q{plane}_B_4",
            ]
            if col in df.columns
        ]
        if charge_columns:
            has_charge = df.loc[:, charge_columns].ne(0).any(axis=1)
            tt_str = tt_str.where(~has_charge, tt_str + str(plane))
    df.loc[:, column_name] = tt_str.replace("", "0").astype(int)
    return df
