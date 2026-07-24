# Remote Log Backup

The scripts in this directory maintain an append-safe backup for the remote log
trees on `mingo01`, `mingo02`, `mingo03`, and `mingo04`.

Mutable backup data is stored separately under:

`OPERATIONS/OPERATIONS_RUNTIME/REMOTE_LOG_BACKUP/`

## Layout

- `hosts/<host>/current/`: exact copy of the latest successful backup of `/home/rpcuser/logs`.
- `hosts/<host>/history/<timestamp>/`: old versions and deleted files moved out of `current/` during that run.
- `hosts/<host>/latest`: symlink to `current/`.
- `hosts/<host>/current_manifest.tsv`: inventory of the latest backed-up files.
- `hosts/<host>/run_logs/`: per-run rsync logs.
- `runtime/run_summaries/`: one TSV summary per full backup run.

## Backup and retention behavior

- The runner uses `rsync --backup --backup-dir=... --delete`.
- `current/` stays faithful to the remote tree, including moves into `done/`.
- Anything that would be overwritten or deleted from `current/` is moved into `history/<timestamp>/`.
- If a remote machine becomes empty, its previous logs are moved into `history/` instead of disappearing from the backup.
- After each successful host backup, history is pruned against
  `MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/config_selection.yaml`: a file is retained
  when its dated filename (or mtime for undated files) falls inside a selected
  station date range, with one calendar month of context before and after.
- Retention fails closed: an unreadable or invalid selection config leaves
  history untouched.

## Usage

Run the backup manually:

```bash
bash /home/mingo/DATAFLOW_v3/OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/REMOTE_LOG_BACKUP/run_log_backup.sh
```

Install the 12-hour cron entry:

```bash
bash /home/mingo/DATAFLOW_v3/OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/REMOTE_LOG_BACKUP/install_log_backup_cron.sh
```

Print the cron line without installing it:

```bash
bash /home/mingo/DATAFLOW_v3/OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/REMOTE_LOG_BACKUP/install_log_backup_cron.sh --print-only
```

## Notes

- The default remote source is `rpcuser@mingo0X:/home/rpcuser/logs/`.
- The runner is locked with `flock`, so overlapping cron executions are skipped.
- For local testing, set `LOG_BACKUP_SOURCE_BASE=/path/to/mock_sources` and create one folder per host under that path.
