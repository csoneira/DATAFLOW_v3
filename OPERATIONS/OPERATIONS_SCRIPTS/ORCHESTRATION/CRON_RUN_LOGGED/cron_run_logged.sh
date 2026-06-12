#!/usr/bin/env bash
# Run a cron command while avoiding empty destination logs.

set -uo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "Usage: cron_run_logged.sh LOG_PATH -- COMMAND [ARG ...]" >&2
  exit 2
fi

log_path="$1"
shift 2

started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
command_text="$(printf '%q ' "$@")"
marker_path="$(mktemp "${TMPDIR:-/tmp}/dataflow_cron_run_output.XXXXXX")"
trap 'rm -f "$marker_path"' EXIT

/usr/bin/env CRON_LOG_PATH="$log_path" "$@" 2>&1 |
  {
    first_line=true
    while IFS= read -r line || [[ -n "$line" ]]; do
      if $first_line; then
        printf '1\n' >"$marker_path"
        mkdir -p "$(dirname "$log_path")"
        printf '%s [CRON_RUN] command=%s\n' "$started_at" "$command_text" >>"$log_path"
        first_line=false
      fi
      printf '%s\n' "$line" >>"$log_path"
    done
  }
rc="${PIPESTATUS[0]}"

silent_expected_lock_contention=false
if [[ $rc -eq 1 && ! -s "$marker_path" && "${1:-}" == */flock ]]; then
  silent_expected_lock_contention=true
fi

if [[ $rc -ne 0 && "$silent_expected_lock_contention" != true ]]; then
  mkdir -p "$(dirname "$log_path")"
  printf '%s [CRON_RUN] status=failed exit_code=%d command=%s\n' \
    "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$rc" "$command_text" >>"$log_path"
elif [[ ! -s "$marker_path" && -f "$log_path" && ! -s "$log_path" ]]; then
  rm -f "$log_path"
fi

exit "$rc"
