#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: move_completed_to_unprocessed.sh [OPTIONS] [BASE_DIR ...]

Search each BASE_DIR (default: repository root) for directories named COMPLETED
and move every file they contain into the sibling UNPROCESSED directory.
Files are skipped when the UNPROCESSED directory is missing or a naming
collision would overwrite an existing file. Multiple base directories can be
provided and will be processed sequentially.

Options:
  -p, --preview   Print a formatted list of planned moves per directory.
  -n, --dry-run   Preview moves without moving any files (implies --preview).
  -h, --help      Show this help message and exit.
EOF
}

PREVIEW=0
DRY_RUN=0
BASE_DIRS=()
SKIP_PATTERNS=(
  "/home/mingo/DATAFLOW_v3/STATIONS/MINGO0*/STAGE_1/LAB_LOGS"
  "/home/mingo/DATAFLOW_v3/STATIONS/MINGO0*/STAGE_1/LAB_LOGS/*"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--preview)
      PREVIEW=1
      shift
      ;;
    -n|--dry-run)
      PREVIEW=1
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      BASE_DIRS+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      BASE_DIRS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BASE="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if ((${#BASE_DIRS[@]} == 0)); then
  BASE_DIRS=("$DEFAULT_BASE")
fi

shopt -s nullglob dotglob

files_moved=0
files_skipped=0
missing_targets=0
found_completed=0

for BASE_DIR in "${BASE_DIRS[@]}"; do
  if [[ ! -d "$BASE_DIR" ]]; then
    echo "Base directory not found: $BASE_DIR" >&2
    exit 1
  fi

  mapfile -t COMPLETED_DIRS < <(find "$BASE_DIR" -type d \( -name COMPLETED -o -name COMPLETED_DIRECTORY \) | sort)

  if ((${#COMPLETED_DIRS[@]} == 0)); then
    echo "No COMPLETED directories found under $BASE_DIR"
    continue
  fi

  found_completed=1

  for completed_dir in "${COMPLETED_DIRS[@]}"; do
    completed_name="$(basename "$completed_dir")"
    case "$completed_name" in
      COMPLETED) target_name="UNPROCESSED" ;;
      COMPLETED_DIRECTORY) target_name="UNPROCESSED_DIRECTORY" ;;
      *)
        target_name="${completed_name/COMPLETED/UNPROCESSED}"
        ;;
    esac
    skip_match=""
    for skip_pattern in "${SKIP_PATTERNS[@]}"; do
      [[ -z "$skip_pattern" ]] && continue
      if [[ "$completed_dir" == $skip_pattern ]]; then
        skip_match="$skip_pattern"
        break
      fi
    done
    if [[ -n "$skip_match" ]]; then
      echo "Skipping $completed_dir (inside excluded path: $skip_match)"
      continue
    fi

    target_dir="$(dirname "$completed_dir")/$target_name"

    if ((PREVIEW)); then
      echo
      echo "=== COMPLETED directory ==="
      echo "  $completed_dir"
      echo "  -> $target_dir"
    fi

    if [[ ! -d "$target_dir" ]]; then
      ((missing_targets+=1))
      echo "Skipping $completed_dir (missing UNPROCESSED directory)" >&2
      continue
    fi

    entries=("$completed_dir"/*)
    if ((${#entries[@]} == 0)); then
      echo "No files to move in $completed_dir"
      continue
    fi

    for entry in "${entries[@]}"; do
      if [[ ! -f "$entry" ]]; then
        continue
      fi

      filename="$(basename "$entry")"
      destination="$target_dir/$filename"

      if [[ -e "$destination" ]]; then
        ((files_skipped+=1))
        echo "Skipping $entry (destination exists)" >&2
        continue
      fi

      if ((PREVIEW)); then
        printf '    - %s -> %s\n' "$filename" "$destination"
      fi

      if ((DRY_RUN)); then
        echo "DRY RUN: would move $entry -> $destination"
        continue
      fi

      mv "$entry" "$destination"
      ((files_moved+=1))
      echo "Moved $entry -> $destination"
    done
  done
done

if (( ! found_completed )); then
  echo "No COMPLETED directories found in the provided base directories."
  exit 0
fi

if ((DRY_RUN)); then
  echo "Dry run complete. No files were moved."
fi

echo "Finished. Files moved: $files_moved, skipped: $files_skipped, missing UNPROCESSED dirs: $missing_targets."
