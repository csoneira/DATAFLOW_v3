# Dataflow overview

*Last updated: March 2026*

This page describes the end‑to‑end data path inside a miniTRASGO station,
from raw Ethernet packets produced by the TRB through to the analysed
MAT files used by the monitoring scripts.  The flow is divided into three
logical stages: acquisition, unpacking, and ancillary analysis.

## A. Acquisition (TRB → `.hld`)

1. The TRB3sc digitiser receives LVDS pulses from the front‑end electronics
   and streams them over Ethernet to the on‑board Odroid computer.
2. The DABC program running on the Odroid collects these packets and writes
them to binary **`.hld`** files under
`/media/externalDisk/hlds/`.  Each file contains several thousand events and
is rotated periodically (typically every hour).
3. A helper script (`CopyFiles`) watches the `.hld` directory and, once two or
   more `.hld` files exist, copies them to
   `~/gate/system/devices/TRB3/data/daqData/rawData/dat/` for processing.

## B. Unpacking (`.hld` → `.mat` / `.dat`)

The **unpacker** program is the core component of this stage; it runs
continuously and supports parallel executions.

1. It ingests the next `.hld` file(s) from the `dat` directory and creates a
   temporary working folder.
2. The file is parsed into an intermediate MATLAB‑compatible structure
   containing raw epoch/coarse/fine time values.  This intermediate file is
   stored in `~/gate/system/devices/TRB3/data/daqData/rawData/mat/` and is
   primarily useful for low‑level debugging.
3. In a second pass the unpacker applies Look‑Up Tables (LUTs) and converts
the raw values into calibrated **time** and **charge** tuples.  The result is
saved in `~/gate/system/devices/TRB3/data/daqData/varData/`; an ASCII copy is
also written to `.../asci/` for easy inspection.
4. During this process `daq_anal` is called to perform the binary‑to‑hex
   conversion.

Unpacked output files are named with a timestamp encoding year, day‑of‑year
and time (e.g. `minI23202060708.dat` corresponds to 2023 day 206 at 07:08).

## C. Automatic analysis and distribution

Once at least two unpacked files are present, the cron‑driven `ana.sh` script
launches the `ana.m` MATLAB routine every 5 s to compute monitoring variables.
The analysis has two branches:

- **TT1 (coincidence trigger)** and **TT2 (self‑trigger)** results are stored
  in separate subfolders under `~/gate/system/devices/mingo01/data/ana/`.
- Within each trigger type the `Vars` directory contains the most useful data
  for physics and QA (see table below).

After processing, the results are copied to the station’s report area for
PDF generation (see [Monitoring](../operation/monitoring.md)).

### Common variables produced by `ana.m` (Vars/TT1 example)

| Index | Name         | Type | Description                                |
|-------|--------------|------|--------------------------------------------|
| 01    | EBTime       |      | Event builder time                         |
| 02    | rawEvents    | num  | Initial number of events                   |
| 03    | Events       | num  | Events after pedestal and cut filters      |
| 04    | runTime      | num  | Duration of input `.hld` file (s)          |
| 05    | Xraw         | 1D   | Raw X strip index                          |
| 06    | Yraw         | 1D   | Raw Y strip index                          |
| 07    | Q            | 1D   | Final charge (AU)                          |
| 08    | Xmm          | 1D   | Calibrated X position (mm)                 |
| 09    | Ymm          | 1D   | Calibrated Y position (mm)                 |
| 10    | T            | 1D   | Calibrated event time                      |
| 11    | Qmean        | num  | Mean charge per event                      |
| 12    | QmeanNoST    | num  | Mean charge excluding streamers            |
| 13    | Qmedian      | num  | Median charge                              |
| 14    | QmedianNoST  | num  | Median charge excluding streamers          |
| 15    | ST           | num  | Streamer fraction (%)                      |
| 16    | XY           | 2D   | Hit map                                   |
| 17    | XY_Qmean     | 2D   | Mean-charge map                            |
| 18    | XY_Qmedian   | 2D   | Median-charge map                          |
| 19    | XY_ST        | 2D   | Streamer-hit map                           |
| 20    | Qhist        | 2D   | Charge histogram                           |

This table is not exhaustive; additional diagnostic arrays may be produced
by later versions of `ana.m`.

## Software repository connection

The unpacker and analysis scripts are part of the station software tree
(`~/gate/`).  They are separate from the central analysis pipeline in this
repository but follow the same data conventions, which facilitates
interoperability with simulated data from the digital twin.

---

_Local preview: run `mkdocs serve` at the root of the `DOCUMENTATION/` folder._

