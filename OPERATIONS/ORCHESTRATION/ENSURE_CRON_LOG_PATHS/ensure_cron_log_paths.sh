#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/DATAFLOW_v3}"
CRONTAB_FILE="${CRONTAB_FILE:-$REPO_ROOT/CONFIG/add_to_crontab.info}"
OPERATIONS_RUNTIME_DIR="${REPO_ROOT}/OPERATIONS_RUNTIME"
LEGACY_EXECUTION_LOGS_DIR="${REPO_ROOT}/EXECUTION_LOGS"
CREATE_FILES=false
QUIET=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --create-files)
      CREATE_FILES=true
      shift
      ;;
    --quiet)
      QUIET=true
      shift
      ;;
    --crontab-file)
      CRONTAB_FILE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$CRONTAB_FILE" ]]; then
  echo "Crontab file not found: $CRONTAB_FILE" >&2
  exit 1
fi

created_dirs=0
created_files=0
legacy_redirect_action="none"
legacy_backup_path=""
legacy_conflict_path=""

ensure_legacy_execution_logs_redirect() {
  mkdir -p "$OPERATIONS_RUNTIME_DIR"

  if [[ -L "$LEGACY_EXECUTION_LOGS_DIR" ]]; then
    local resolved_target
    resolved_target="$(readlink -f "$LEGACY_EXECUTION_LOGS_DIR" 2>/dev/null || true)"
    if [[ "$resolved_target" != "$OPERATIONS_RUNTIME_DIR" ]]; then
      rm -f "$LEGACY_EXECUTION_LOGS_DIR"
      ln -s "$OPERATIONS_RUNTIME_DIR" "$LEGACY_EXECUTION_LOGS_DIR"
      legacy_redirect_action="relinked"
    fi
    return
  fi

  if [[ -d "$LEGACY_EXECUTION_LOGS_DIR" ]]; then
    local legacy_items
    local item
    local item_name
    local target_path
    local moved_any=false
    shopt -s dotglob nullglob
    legacy_items=("$LEGACY_EXECUTION_LOGS_DIR"/*)
    shopt -u dotglob nullglob

    for item in "${legacy_items[@]}"; do
      item_name="$(basename "$item")"
      target_path="${OPERATIONS_RUNTIME_DIR}/${item_name}"
      if [[ ! -e "$target_path" ]]; then
        mv "$item" "$target_path"
        moved_any=true
        continue
      fi

      if [[ -z "$legacy_conflict_path" ]]; then
        legacy_conflict_path="${OPERATIONS_RUNTIME_DIR}/LEGACY_EXECUTION_LOGS_CONFLICTS_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$legacy_conflict_path"
      fi
      mv "$item" "${legacy_conflict_path}/${item_name}"
      moved_any=true
    done

    if ! rmdir "$LEGACY_EXECUTION_LOGS_DIR" 2>/dev/null; then
      legacy_backup_path="${LEGACY_EXECUTION_LOGS_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
      mv "$LEGACY_EXECUTION_LOGS_DIR" "$legacy_backup_path"
    fi

    ln -s "$OPERATIONS_RUNTIME_DIR" "$LEGACY_EXECUTION_LOGS_DIR"
    if $moved_any; then
      legacy_redirect_action="migrated_and_linked"
    else
      legacy_redirect_action="linked"
    fi
    return
  fi

  if [[ -e "$LEGACY_EXECUTION_LOGS_DIR" ]]; then
    legacy_backup_path="${LEGACY_EXECUTION_LOGS_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
    mv "$LEGACY_EXECUTION_LOGS_DIR" "$legacy_backup_path"
    ln -s "$OPERATIONS_RUNTIME_DIR" "$LEGACY_EXECUTION_LOGS_DIR"
    legacy_redirect_action="replaced_non_directory"
    return
  fi

  ln -s "$OPERATIONS_RUNTIME_DIR" "$LEGACY_EXECUTION_LOGS_DIR"
  legacy_redirect_action="linked"
}

ensure_legacy_execution_logs_redirect

while IFS= read -r target; do
  [[ -z "$target" ]] && continue
  [[ "$target" != /* ]] && continue

  log_dir="$(dirname "$target")"
  if [[ ! -d "$log_dir" ]]; then
    mkdir -p "$log_dir"
    created_dirs=$((created_dirs + 1))
  fi

  if $CREATE_FILES && [[ ! -e "$target" ]]; then
    : >"$target"
    created_files=$((created_files + 1))
  fi
done < <(
  awk '
    /^[[:space:]]*#/ {next}
    NF == 0 {next}
    {
      for (i = 1; i <= NF; i++) {
        tok = $i
        if (tok ~ /^>>/) {
          p = substr(tok, 3)
          if (p == "") {
            if (i + 1 <= NF) print $(i + 1)
          } else {
            print p
          }
          continue
        }
        if (tok ~ /^>/ && tok !~ /^[0-9]+>/ && tok !~ /^>&/) {
          p = substr(tok, 2)
          if (p == "") {
            if (i + 1 <= NF) print $(i + 1)
          } else {
            print p
          }
        }
      }
    }
  ' "$CRONTAB_FILE" | sort -u
)

if ! $QUIET; then
  echo "ensure_cron_log_paths: created_dirs=$created_dirs created_files=$created_files legacy_redirect_action=$legacy_redirect_action"
  if [[ -n "$legacy_backup_path" ]]; then
    echo "ensure_cron_log_paths: legacy_backup_path=$legacy_backup_path"
  fi
  if [[ -n "$legacy_conflict_path" ]]; then
    echo "ensure_cron_log_paths: legacy_conflict_path=$legacy_conflict_path"
  fi
fi
