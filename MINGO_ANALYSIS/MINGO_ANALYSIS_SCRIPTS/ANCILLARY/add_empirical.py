#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


EFF_COLUMNS = [
    "eff_empirical_1",
    "eff_empirical_2",
    "eff_empirical_3",
    "eff_empirical_4",
]

# eff_i = R_1234 / (R_1234 + R_missing_i)
RATE_COLUMNS = {
    "eff_empirical_1": "tt_task5_post_234_rate_hz",
    "eff_empirical_2": "tt_task5_post_134_rate_hz",
    "eff_empirical_3": "tt_task5_post_124_rate_hz",
    "eff_empirical_4": "tt_task5_post_123_rate_hz",
}

REFERENCE_RATE_COLUMN = "tt_task5_post_1234_rate_hz"


def resolve_column(dataframe: pd.DataFrame, column: str) -> str:
    """
    Return the actual column name in the dataframe.

    Supports both:
        tt_task5_post_1234_rate_hz
    and:
        rate_hz__tt_task5_post_1234_rate_hz
    """
    if column in dataframe.columns:
        return column

    prefixed = f"rate_hz__{column}"
    if prefixed in dataframe.columns:
        return prefixed

    raise KeyError(
        f"Required column not found: {column!r} "
        f"or {prefixed!r}"
    )


def calculate_empirical_efficiencies(
    dataframe: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Add or update eff_empirical_1 ... eff_empirical_4.

    By default, only missing/NaN values are filled.
    If overwrite=True, existing values are recalculated.
    """
    dataframe = dataframe.copy()

    r_1234_col = resolve_column(dataframe, REFERENCE_RATE_COLUMN)
    r_1234 = pd.to_numeric(dataframe[r_1234_col], errors="coerce")

    for eff_col in EFF_COLUMNS:
        if eff_col not in dataframe.columns:
            dataframe[eff_col] = np.nan

    for eff_col, missing_rate_col in RATE_COLUMNS.items():
        missing_col = resolve_column(dataframe, missing_rate_col)
        r_missing = pd.to_numeric(dataframe[missing_col], errors="coerce")

        denominator = r_1234 + r_missing

        recalculated = pd.Series(
            np.where(
                denominator > 0,
                r_1234 / denominator,
                np.nan,
            ),
            index=dataframe.index,
            dtype="float64",
        )

        if overwrite:
            dataframe[eff_col] = recalculated
        else:
            dataframe[eff_col] = dataframe[eff_col].fillna(recalculated)

    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate eff_empirical_1 ... eff_empirical_4 from "
            "tt_task5_post_*_rate_hz columns and modify the CSV file in place."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file to modify in place.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate efficiencies even if eff_empirical_* columns already contain values.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak backup before modifying the file.",
    )

    args = parser.parse_args()
    csv_path = args.csv_path.expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"File does not exist: {csv_path}")

    if not csv_path.is_file():
        raise ValueError(f"Path is not a file: {csv_path}")

    if not args.no_backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        shutil.copy2(csv_path, backup_path)
        print(f"Backup written to: {backup_path}")

    dataframe = pd.read_csv(csv_path)

    print("Before:")
    print(dataframe.reindex(columns=EFF_COLUMNS).head())

    dataframe = calculate_empirical_efficiencies(
        dataframe,
        overwrite=args.overwrite,
    )

    print("\nAfter:")
    print(dataframe[EFF_COLUMNS].head())

    dataframe.to_csv(csv_path, index=False)
    print(f"\nUpdated file in place: {csv_path}")


if __name__ == "__main__":
    main()