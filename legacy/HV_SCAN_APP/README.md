# HV_SCAN_APP legacy archive

> **LEGACY / HISTORICAL ONLY.** This directory is preserved for reference and information recovery. It is not part of the active `mingo-computing` runtime, analysis pipeline, or operational tooling, and should not be invoked by production scripts.

## Provenance

- Original repository: `https://github.com/csoneira/HV_SCAN_APP`
- Archived source branch: `main`
- Archived source commit: `e6533e7a6cd7febb124546c07538953724560afc`
- Archived into `mingo-computing`: 2026-10-21
- Original license: GNU General Public License v3.0
- Original license blob SHA: `e72bfddabc15be5718a7cc061ac10e47741d8219`

## What is preserved

This directory contains the complete project-specific tracked information needed to understand or reconstruct the source repository at the commit above:

- `hv_scan_analysis.py`: the historical interactive HV-scan analysis program, preserved byte-for-byte from the source repository.
- `CONFIG/drive_links.txt`: Google Drive locations used by the program for station CSV data, preserved byte-for-byte.
- `requirements.list`: Python dependencies, preserved byte-for-byte.
- `ORIGINAL_README.md`: the original repository README, preserved byte-for-byte.
- `LICENSE`: archival license notice recording the original GPLv3 licensing and exact upstream license blob SHA.
- `.gitignore`: the original CSV-ignore rule, preserved byte-for-byte.

The source repository did not track the station CSV datasets themselves; its `.gitignore` excludes `*.csv`. The original README describes the expected `DATA/MINGO0X/data_X.csv` layout.

## Important historical behavior

The script analyzes miniTRASGO HV-scan CSV data for stations 1 through 4. It can download station data through `gdown`, select a time interval, optionally use the historical `HV * T / P` reduced-HV quantity, and plot plane efficiencies, streamer fractions, pressure, temperature, CRT/timing information, and HV versus time.

The code is preserved as historical source, not repaired or modernized. In particular, it contains an absolute configuration lookup at `~/HV_SCAN_APP/CONFIG/drive_links.txt`. That path is intentionally left unchanged because this snapshot is archival. If this code is ever revived, it should be reviewed and integrated deliberately rather than run in place from this directory.

For the original project documentation, see [`ORIGINAL_README.md`](ORIGINAL_README.md).
