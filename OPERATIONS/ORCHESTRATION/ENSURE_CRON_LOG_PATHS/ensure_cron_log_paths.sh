#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/DATAFLOW_v3}"
CRONTAB_FILE="${CRONTAB_FILE:-$REPO_ROOT/add_to_crontab.info}"
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
  echo "ensure_cron_log_paths: created_dirs=$created_dirs created_files=$created_files"
fi

