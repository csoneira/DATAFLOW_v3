# MINGO_PRESSURE legacy archive

> **LEGACY / HISTORICAL ONLY.** This directory preserves the conceptual source code of the former `MINGO_PRESSURE` repository for later review and selective recovery. It is not part of the active `mingo-computing` analysis or operations pipeline.

## Provenance

- Original repository: `https://github.com/csoneira/MINGO_PRESSURE`
- Original branch: `master`
- Archived source commit: `d71adace37abab7ecdf0cddd0817e84358ebb5d2`
- Archived into `mingo-computing`: 2026-10-21
- Original `pressure_CALMA.py` blob: `33ff74f3b327a7c43ffa11c3e0585c65682a49fa`
- Original `README.md` blob: `2fb3e70f9627a585118a3e0396cc405067efad69`

## Preserved source

- `pressure_CALMA.py`: historical pressure/cosmic-ray-rate analysis script, preserved byte-for-byte.
- `ORIGINAL_README.md`: original repository documentation, preserved byte-for-byte.

The script studies the barometric coefficient by combining pressure and cosmic-ray count-rate time series, performing sliding-window regressions, estimating coefficient uncertainties and R² values, and visualizing their evolution with window length and date.

## Intentionally excluded data

The following tracked CSV files from the original repository are **not** copied into this archive:

- `filtered_data.csv`
- `filtered_data_1_52_1.csv`
- `pressure_OULU.csv`
- `rate_OULU.csv`

They are data/results payloads rather than the conceptual implementation requested for preservation. The historical script still refers to these filenames; that is intentional because the source is archived rather than repaired or made runnable in place.

If this analysis is revived, recover the useful concepts into the current analysis architecture instead of treating this directory as production code.
