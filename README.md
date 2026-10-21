# DATAFLOW_v3

For a new analysis PC, clone this repository to `~/DATAFLOW_v3` and run:

```bash
bash CONFIG/SETUP/setup.sh --apply
```

The setup installs a fixed analysis-PC cron schedule and prints what is ready,
missing, or failed. Run `bash CONFIG/SETUP/setup.sh --check` for a later
point-in-time configuration check. Processing health, cron-log errors, and
status CSVs belong to the existing `OPERATIONS` scripts. See the
[setup guide](CONFIG/SETUP/README.md) and [recovery audit](CONFIG/SETUP/AUDIT.md).
