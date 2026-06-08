# LOG_BACKUP

This directory contains an append-safe backup for the remote log trees on `mingo01`, `mingo02`, `mingo03`, and `mingo04`.

## Layout

- `hosts/<host>/current/`: exact copy of the latest successful backup of `/home/rpcuser/logs`.
- `hosts/<host>/history/<timestamp>/`: old versions and deleted files moved out of `current/` during that run.
- `hosts/<host>/latest`: symlink to `current/`.
- `hosts/<host>/current_manifest.tsv`: inventory of the latest backed-up files.
- `hosts/<host>/run_logs/`: per-run rsync logs.
- `runtime/run_summaries/`: one TSV summary per full backup run.

## Why this does not lose logs

- The runner uses `rsync --backup --backup-dir=... --delete`.
- `current/` stays faithful to the remote tree, including moves into `done/`.
- Anything that would be overwritten or deleted from `current/` is moved into `history/<timestamp>/`.
- If a remote machine becomes empty, its previous logs are moved into `history/` instead of disappearing from the backup.

## Usage

Run the backup manually:

```bash
bash /home/mingo/DATAFLOW_v3/LOG_BACKUP/run_log_backup.sh
```

Install the 12-hour cron entry:

```bash
bash /home/mingo/DATAFLOW_v3/LOG_BACKUP/install_log_backup_cron.sh
```

Print the cron line without installing it:

```bash
bash /home/mingo/DATAFLOW_v3/LOG_BACKUP/install_log_backup_cron.sh --print-only
```

## Notes

- The default remote source is `rpcuser@mingo0X:/home/rpcuser/logs/`.
- The runner is locked with `flock`, so overlapping cron executions are skipped.
- For local testing, set `LOG_BACKUP_SOURCE_BASE=/path/to/mock_sources` and create one folder per host under that path.
