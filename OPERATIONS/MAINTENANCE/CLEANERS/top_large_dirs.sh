#!/usr/bin/env bash
# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: OPERATIONS/MAINTENANCE/CLEANERS/top_large_dirs.sh
# Purpose: Top large dirs.
# Owner: DATAFLOW_v3 contributors
# Sign-off: csoneira <csoneira@ucm.es>
# Last Updated: 2026-03-02
# Runtime: bash
# Usage: bash OPERATIONS/MAINTENANCE/CLEANERS/top_large_dirs.sh [options]
# Inputs: CLI args, config files, environment variables, and/or upstream files.
# Outputs: Files, logs, or process-level side effects.
# Notes: Keep behavior configuration-driven and reproducible.
# =============================================================================

set -euo pipefail

show_help() {
  cat <<'EOF'
top_large_dirs.sh
Lists the largest top-level directories (links=2) under /home/mingo.

Usage:
  top_large_dirs.sh [count]

Options:
  -h, --help    Show this help message and exit.

Arguments:
  count         Number of entries to show (default: 30).

By default the script runs without sudo and only examines the current user's home.
EOF
}

LIMIT=30

if (( $# )); then
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      LIMIT="$1"
      ;;
  esac
fi

find /home/mingo -xdev -type d -links 2 -print0 \
  | xargs -0 du -sk 2>/dev/null \
  | sort -nrk1 | head -"${LIMIT}" \
  | awk '{ printf "%8.2f GiB\t%s\n", $1/1024/1024, substr($0, index($0,$2)) }'
