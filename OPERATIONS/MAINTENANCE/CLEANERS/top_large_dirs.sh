#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
top_large_dirs.sh
Lists the largest top-level directories (links=2) under /home.

Usage:
  top_large_dirs.sh [count]

Options:
  -h, --help    Show this help message and exit.

Arguments:
  count         Number of entries to show (default: 30).

The script relies on sudo to traverse directory ownership boundaries.
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

sudo find /home -xdev -type d -links 2 -print0 \
  | sudo xargs -0 du -sk 2>/dev/null \
  | sort -nrk1 | head -"${LIMIT}" \
  | awk '{ printf "%8.2f GiB\t%s\n", $1/1024/1024, substr($0, index($0,$2)) }'
