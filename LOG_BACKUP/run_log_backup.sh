#!/usr/bin/env bash
# =============================================================================
# Script: LOG_BACKUP/run_log_backup.sh
# Purpose: Maintain an append-safe backup of remote MINGO log trees.
# =============================================================================

set -euo pipefail

show_help() {
  cat <<'EOF'
run_log_backup.sh

Pulls /home/rpcuser/logs from mingo01..mingo04 into:
  LOG_BACKUP/hosts/<host>/current/

Rules:
  - current/ is an exact mirror of the latest successful remote state.
  - history/<timestamp>/ stores files that were replaced or deleted from
    current/ during a run, so old data is never lost.
  - If a remote log tree becomes empty, the old logs are moved into history/
    instead of being erased from the backup.

Usage:
  bash run_log_backup.sh [options] [host ...]

Options:
  --host NAME           Backup only one host. Can be repeated.
  --root PATH           Override backup root. Default: ~/DATAFLOW_v3/LOG_BACKUP
  --source-base PATH    Local test source base; expects PATH/<host>/...
  --remote-user USER    Override remote SSH user. Default: rpcuser
  --remote-root PATH    Override remote log root. Default: /home/rpcuser/logs
  -h, --help            Show this help text.

Environment overrides:
  LOG_BACKUP_ROOT
  LOG_BACKUP_HOSTS
  LOG_BACKUP_SOURCE_BASE
  LOG_BACKUP_REMOTE_USER
  LOG_BACKUP_REMOTE_ROOT
  LOG_BACKUP_RUN_TIMESTAMP
  LOG_BACKUP_SSH_OPTIONS
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
}

require_option_value() {
  local option_name="$1"
  local option_value="${2:-}"
  if [[ -z "$option_value" ]]; then
    echo "Missing value for $option_name" >&2
    exit 1
  fi
}

remove_empty_history_dir_tree() {
  local start_dir="$1"
  local stop_dir="$2"
  local current_dir="$start_dir"

  while [[ -n "$current_dir" && "$current_dir" != "/" ]]; do
    rmdir "$current_dir" 2>/dev/null || break
    if [[ "$current_dir" == "$stop_dir" ]]; then
      break
    fi
    current_dir="$(dirname "$current_dir")"
  done
}

write_current_manifest() {
  local current_dir="$1"
  local manifest_path="$2"
  local tmp_manifest

  tmp_manifest="$(mktemp)"
  {
    printf 'mtime_utc\tsize_bytes\trelative_path\n'
    if [[ -d "$current_dir" ]]; then
      LC_ALL=C find "$current_dir" -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS\t%s\t%P\n' | sort
    fi
  } > "$tmp_manifest"
  mv "$tmp_manifest" "$manifest_path"
}

backup_one_host() {
  local host="$1"
  local host_dir="$BACKUP_ROOT/hosts/$host"
  local current_dir="$host_dir/current"
  local history_root="$host_dir/history"
  local history_run_dir="$history_root/$RUN_TIMESTAMP"
  local run_log_dir="$host_dir/run_logs"
  local state_dir="$host_dir/state"
  local latest_link="$host_dir/latest"
  local manifest_path="$host_dir/current_manifest.tsv"
  local partial_dir="$host_dir/.rsync-partial"
  local run_log_path="$run_log_dir/$RUN_TIMESTAMP.log"
  local source_spec=""
  local status=0
  local current_file_count=0
  local history_file_count=0
  local history_bytes=0

  mkdir -p "$current_dir" "$history_root" "$run_log_dir" "$state_dir" "$partial_dir"

  if [[ -n "$SOURCE_BASE" ]]; then
    source_spec="${SOURCE_BASE%/}/$host/"
    if [[ ! -d "${SOURCE_BASE%/}/$host" ]]; then
      printf '%s\t%s\tfailed\t0\t0\t0\tmissing local source\n' \
        "$RUN_TIMESTAMP" "$host" >> "$SUMMARY_FILE"
      log "Skipping $host: local test source not found at ${SOURCE_BASE%/}/$host"
      return 1
    fi
  else
    source_spec="${REMOTE_USER}@${host}:${REMOTE_ROOT%/}/"
  fi

  {
    printf 'run_timestamp=%s\n' "$RUN_TIMESTAMP"
    printf 'host=%s\n' "$host"
    printf 'source=%s\n' "$source_spec"
    printf 'destination=%s\n' "$current_dir"
    printf 'history_run_dir=%s\n' "$history_run_dir"
  } > "$run_log_path"

  log "Backing up $host"

  local -a rsync_cmd=(
    rsync
    -rlptz
    --delete
    --delete-delay
    --backup
    "--backup-dir=$history_run_dir"
    --partial
    "--partial-dir=$partial_dir"
    --itemize-changes
  )

  if [[ -z "$SOURCE_BASE" ]]; then
    rsync_cmd+=(-e "ssh $SSH_OPTIONS")
  fi

  rsync_cmd+=("$source_spec" "$current_dir/")

  if "${rsync_cmd[@]}" >> "$run_log_path" 2>&1; then
    :
  else
    status=$?
    printf '%s\n' "$RUN_TIMESTAMP" > "$state_dir/last_failed_run.txt"
    printf '%s\t%s\tfailed\t0\t0\t0\trsync exit %s\n' \
      "$RUN_TIMESTAMP" "$host" "$status" >> "$SUMMARY_FILE"
    log "Backup failed for $host. See $run_log_path"
    return "$status"
  fi

  write_current_manifest "$current_dir" "$manifest_path"
  ln -sfn "$current_dir" "$latest_link"
  printf '%s\n' "$RUN_TIMESTAMP" > "$state_dir/last_successful_run.txt"

  current_file_count=$(find "$current_dir" -type f | wc -l | tr -d ' ')
  if [[ -d "$history_run_dir" ]]; then
    history_file_count=$(find "$history_run_dir" -type f | wc -l | tr -d ' ')
    if [[ "$history_file_count" -gt 0 ]]; then
      history_bytes=$(du -sb "$history_run_dir" | awk '{print $1}')
    else
      remove_empty_history_dir_tree "$history_run_dir" "$history_root"
    fi
  fi

  printf '%s\t%s\tsuccess\t%s\t%s\t%s\tok\n' \
    "$RUN_TIMESTAMP" "$host" "$current_file_count" "$history_file_count" "$history_bytes" \
    >> "$SUMMARY_FILE"
  log "Backup completed for $host"
}

DEFAULT_BACKUP_ROOT="$HOME/DATAFLOW_v3/LOG_BACKUP"
DEFAULT_REMOTE_USER="rpcuser"
DEFAULT_REMOTE_ROOT="/home/rpcuser/logs"
DEFAULT_HOSTS=(mingo01 mingo02 mingo03 mingo04)

BACKUP_ROOT="${LOG_BACKUP_ROOT:-$DEFAULT_BACKUP_ROOT}"
REMOTE_USER="${LOG_BACKUP_REMOTE_USER:-$DEFAULT_REMOTE_USER}"
REMOTE_ROOT="${LOG_BACKUP_REMOTE_ROOT:-$DEFAULT_REMOTE_ROOT}"
SOURCE_BASE="${LOG_BACKUP_SOURCE_BASE:-}"
RUN_TIMESTAMP="${LOG_BACKUP_RUN_TIMESTAMP:-$(date -u '+%Y%m%dT%H%M%SZ')}"
SSH_OPTIONS="${LOG_BACKUP_SSH_OPTIONS:--o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3}"

declare -a requested_hosts=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      require_option_value "$1" "${2:-}"
      requested_hosts+=("$2")
      shift 2
      ;;
    --root)
      require_option_value "$1" "${2:-}"
      BACKUP_ROOT="$2"
      shift 2
      ;;
    --source-base)
      require_option_value "$1" "${2:-}"
      SOURCE_BASE="$2"
      shift 2
      ;;
    --remote-user)
      require_option_value "$1" "${2:-}"
      REMOTE_USER="$2"
      shift 2
      ;;
    --remote-root)
      require_option_value "$1" "${2:-}"
      REMOTE_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        requested_hosts+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      requested_hosts+=("$1")
      shift
      ;;
  esac
done

if [[ ${#requested_hosts[@]} -eq 0 ]]; then
  if [[ -n "${LOG_BACKUP_HOSTS:-}" ]]; then
    read -r -a requested_hosts <<< "${LOG_BACKUP_HOSTS}"
  else
    requested_hosts=("${DEFAULT_HOSTS[@]}")
  fi
fi

require_command rsync
require_command flock
if [[ -z "$SOURCE_BASE" ]]; then
  require_command ssh
fi

RUNTIME_DIR="$BACKUP_ROOT/runtime"
SUMMARY_DIR="$RUNTIME_DIR/run_summaries"
LOCK_DIR="$RUNTIME_DIR/locks"
LOCK_FILE="$LOCK_DIR/run_log_backup.lock"
SUMMARY_FILE="$SUMMARY_DIR/$RUN_TIMESTAMP.tsv"

mkdir -p "$BACKUP_ROOT/hosts" "$SUMMARY_DIR" "$LOCK_DIR"

exec {lock_fd}> "$LOCK_FILE"
if ! flock -n "$lock_fd"; then
  log "Another log-backup run is already in progress."
  exit 0
fi

{
  printf 'run_timestamp\thost\tstatus\tcurrent_files\thistory_files\thistory_bytes\tmessage\n'
} > "$SUMMARY_FILE"

failure_count=0
for host in "${requested_hosts[@]}"; do
  if [[ -z "$host" ]]; then
    continue
  fi
  if ! backup_one_host "$host"; then
    failure_count=$((failure_count + 1))
  fi
done

if [[ "$failure_count" -gt 0 ]]; then
  log "Finished with $failure_count host failure(s). Summary: $SUMMARY_FILE"
  exit 1
fi

log "Finished successfully. Summary: $SUMMARY_FILE"
