# Data: general considerations

*Last updated: March 2026*

This section describes the raw and intermediate data formats produced by a
miniTRASGO station together with guidance on retrieving and interpreting
them.  Familiarity with these files is useful when debugging hardware issues
or when ingesting data into the analysis pipeline.

## Retrieving data from a station

Files can be copied from the station via SSH/SCP.  Example:

```bash
scp rpcuser@minitrasgo.fis.ucm.es:/home/rpcuser/gate/system/devices/RPC01/data/dcData/data/2023-07-13-EffMap.mat ~/
```

You will need the station credentials; see the [Operation](../operation/index.md)
section for connection details.

## Time conventions

The on‑board PC clock is set to UTC.  MATLAB serial date numbers (e.g.
`739077.6528125`) count days since 0000‑01‑00; convert with `datestr()` or
`datetime(...,'ConvertFrom','datenum')`.  The native timestamp resolution is
nanoseconds, but most log formats store only milliseconds.

## Detector Control System (DCS) logs

The slow‑control system communicates with the HV supply, gas flow meters and
environmental sensors over an I2C hub.  Data collection scripts run from
`crontab` and deposit `.log` files under `~/logs/` on the station computer;
each file is rotated daily into `~/logs/done/`.

Log file formats are fixed-width ASCII; key types are listed below.  All
timestamps use the ISO 8601 format with `T` separator.

| Filename pattern | Description | Example row |
|------------------|-------------|-------------|
| `sensors_bus0_YYYY-MM-DD.log` | External environment (T, humidity, pressure) | `2023-07-21T23:45:03; nan nan nan nan 24.7 54.5 1007.8` |
| `sensors_bus1_YYYY-MM-DD.log` | Internal box environment | same format as bus0 |
| `Flow0_YYYY-MM-DD.log` | Output gas flow of four RPCs (AU units) | `2023-08-01 13:35:04 805 802 860 667` |
| `hv0_YYYY-MM-DD.log` | HV/current readings; voltages in kV, currents µA; MAC address prefix included | see sample in text |
| `rates_YYYY-MM-DD.log` | Trigger statistics from CTS web interface | `2023-08-01T13:45:51; 9.0 7.9 7.9 56.5 83.9 91.0 97.0 5.0 2.6 2.8 4.7` |

Additional rate files are generated via a Python script
`~/gate/python/log_CTSrates_multiProcessing.py` and are not produced by the
I2C hub; they rely on the `trbnetd` daemon for communication with the TRB.

### Look‑Up Tables

The DCS stores calibration and configuration tables under
`~/gate/system/lookUpTables/`.  These are currently plain text but may be
migrated to Excel or JSON in the future; they encode device-specific LUTs for
HV, temperature conversion, etc.

## MATLAB-specific files

The analysis software used on the station writes `.mat` files (e.g.
`EffMap.mat`) for quick plotting.  Consult the MATLAB code under
`~/gate` for the schema; most fields are named identically to Python
column names used in the offline pipeline.

## Raw DAQ output

The TRB3sc data acquisition system produces binary `.hld` event files which
are normally archived under `/media/externalDisk/hlds/`.  These contain the
full time‑stamp and width information for each hit and are read using the
`daq_anal` utility on the station.  For pipeline ingestion, `.hld` files are
converted to `.dat` format by the digital twin (simulated data) or by
`MASTER/STAGES/STAGE_0/SIMULATION/ingest_simulated_station_data.py` when
processing real data.

## Recommended workflows

1. **Troubleshooting hardware:** examine the most recent DCS logs to check
   HV stability, gas flow and environmental conditions before analysing
   `.hld` files.
2. **Data quality checks:** use `daq_anal` to verify that `.hld` files contain
   events before executing the analysis pipeline.  Empty files are skipped by
   the automated scheduler.
3. **Archiving:** compress old `.log` and `.mat` files with `gzip` and record
   their SHA256 checksum to ensure data provenance.

For additional details on the meaning of each field and examples of analysis
scripts, see the [Operation](../operation/) and [Tasks](../tasks/) sections of
this documentation, as well as the MATLAB code in the station repository.

