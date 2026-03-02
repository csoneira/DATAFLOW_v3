# Software environment and station setup

*Last updated: March 2026*

This page collects notes on the UNIX environment, folder layout and key
software tools installed on the miniTRASGO control computer.  Operators will
find it useful when debugging, updating scripts, or inspecting logs.

## Terminal sessions

We use `tmux` to multiplex several shell windows on the station PC.  Three
named sessions are typically available:

```bash
tmux ls
# 0: 7 windows (attached)      # measurement operations
# 1: 13 windows (attached)     # data treatment
# webserver: 1 window          # trigger monitoring (non-interactive)
```

Attach to a session with:

```bash
tmux attach -t <session-name>   # e.g. 0, 1 or webserver
```

Common `tmux` shortcuts:

- `CTRL+B C` – create a new window.
- `CTRL+B D` – detach from the session.
- `tmux rename-window <name>` – set a descriptive name.
- `exit` – close the current window.

Using `tmux` allows multiple users to log in simultaneously and observe the
same operations in real time.

## Folder structure

Key directories on the station host:

```
/home/rpcuser/
├─ bin/                     # custom utilities (powercycle, HV control, etc.)
├─ logs/                    # daily DCS/environment logs
├─ gate/                    # main software tree
│   ├─ bin/                 # helper scripts for file copying, alarms
│   ├─ python/              # Python logging scripts (flow, etc.)
│   ├─ software/            # MATLAB/Octave analysis code and alarms
│   └─ system/
│       ├─ lookUpTables/    # calibration LUTs for devices
│       └─ devices/         # per-device data (see Dataflow page)
└─ …
```

`/media/externalDisk/gate` mirrors much of the `~/gate` tree and is used for
salvaging data when the root filesystem is nearly full.

## Relevant tools

### crontab

The station crontab (`crontab -e` as `rpcuser`) schedules DAQ startup, DCS
polling, report generation, and various housekeeping tasks.  Use
[crontab.guru](https://crontab.guru/) to compose expressions.

### Look‑Up Tables (LUTs)

LUTs live in `/media/externalDisk/gate/system/lookUpTables/`.  Each device
(e.g. TRB, HV, flowmeter) has its own table describing how values are
interpreted and under what conditions alarms fire.  Entries use the format

```
{<enabled?>, <min threshold>, <count>}, {<enabled?>, <max threshold>, <count>}
```

Migrating these tables to a more portable format (e.g. Excel or JSON) is a
ongoing objective.

### Data Acquisition Backbone Core (DABC)

DABC is the framework used to receive TRB3 data packets over UDP and write
`.hld` files.  It is extensible via plug‑ins; the miniTRASGO installation uses
a custom HADES/TRB3 plugin.  Familiarity with DABC is helpful when diagnosing
network or buffering issues.

---

_Local preview: run `mkdocs serve` in the documentation directory._

