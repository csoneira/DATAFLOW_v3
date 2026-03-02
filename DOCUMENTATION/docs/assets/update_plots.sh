#!/usr/bin/env bash
# update_plots.sh - synchronize selected plot images into the documentation
# assets folder.
#
# The list of files to refresh is kept in "plot_list.txt" (one entry per
# line, relative to the workspace root).  Lines beginning with "#" or blank
# are ignored.  Each entry may contain shell globs; matching files will be
# copied to the assets directory if they are newer or missing.

set -euo pipefail

# Determine directories relative to this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"    # workspace root
ASSETS_DIR="$SCRIPT_DIR"
CONFIG_FILE="$ASSETS_DIR/plot_list.txt"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

while IFS= read -r entry; do
  # strip whitespace
  entry="${entry%%#*}"      # remove comments after #
  entry="$(echo "$entry" | xargs)"  # trim spaces
  [[ -z "$entry" ]] && continue

  # expand glob(s)
  shopt -s nullglob
  files=("$ROOT_DIR"/$entry)
  shopt -u nullglob

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "[update_plots] no matches for '$entry'" >&2
    continue
  fi

  for file in "${files[@]}"; do
    cp -u "$file" "$ASSETS_DIR/"
  done

done < "$CONFIG_FILE"

exit 0
