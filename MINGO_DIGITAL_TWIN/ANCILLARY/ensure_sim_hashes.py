#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

def parse_first_line_hash(line: str) -> Optional[str]:
    if not line:
        return None
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    if stripped.lower().startswith("# param_hash="):
        value = stripped.split("=", 1)[1].strip()
        return value or None
    return None


def load_param_hash_lookup(params_path: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not params_path.exists():
        return lookup
    with params_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            param_hash = (row.get("param_hash") or "").strip()
            if not file_name or not param_hash:
                continue
            lookup[file_name.lower()] = param_hash
    return lookup


def iter_mi00_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("mi00*.dat"):
            if path.is_file():
                yield path


def iter_mi00_parquets(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.parquet"):
            if path.is_file() and "mi00" in path.name.lower():
                yield path


def normalize_hash(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def extract_mi00_basename(path: Path) -> Optional[str]:
    match = re.search(r"(mi00\d+)", path.name.lower())
    if not match:
        return None
    return match.group(1)


def lookup_param_hash_for_parquet(path: Path, lookup: Dict[str, str]) -> Optional[str]:
    basename = extract_mi00_basename(path)
    if not basename:
        return None
    return lookup.get(f"{basename}.dat")


def parquet_hash_status(path: Path) -> Tuple[str, Optional[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return "error_read", None
    try:
        parquet = pq.ParquetFile(path)
    except Exception:
        return "error_read", None
    if "param_hash" not in parquet.schema.names:
        return "missing_column", None
    try:
        found_value = None
        found_missing = False
        for i in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(i, columns=["param_hash"])
            values = table.column(0).to_pylist()
            for value in values:
                normalized = normalize_hash(value)
                if normalized:
                    if found_value is None:
                        found_value = normalized
                else:
                    found_missing = True
    except Exception:
        return "error_read", None
    if found_value and found_missing:
        return "needs_fill", found_value
    if found_value:
        return "has_hash", found_value
    return "empty_column", None


def ensure_param_hash_column(path: Path, param_hash: str, apply_changes: bool) -> bool:
    if not apply_changes:
        return True
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df = pd.read_parquet(path, engine="pyarrow")
    if "param_hash" not in df.columns:
        df["param_hash"] = param_hash
    else:
        series = df["param_hash"]
        missing = series.isna()
        try:
            missing |= series.astype(str).str.strip().eq("")
        except Exception:
            pass
        if missing.any():
            df.loc[missing, "param_hash"] = param_hash
        else:
            return True
    df.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
    tmp_path.replace(path)
    return True


def ensure_hash_header(path: Path, param_hash: str, apply_changes: bool) -> bool:
    header = f"# param_hash={param_hash}\n".encode("ascii")
    if not apply_changes:
        return True
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as out_handle:
        out_handle.write(header)
        with path.open("rb") as in_handle:
            shutil.copyfileobj(in_handle, out_handle)
    tmp_path.replace(path)
    return True


def remove_file(path: Path, apply_changes: bool) -> bool:
    if not apply_changes:
        return True
    path.unlink()
    return True


def process_file(
    path: Path,
    lookup: Dict[str, str],
    apply_changes: bool,
) -> Tuple[str, Optional[str]]:
    try:
        with path.open("r", encoding="ascii", errors="replace") as handle:
            first_line = handle.readline()
    except OSError:
        return "error_read", None

    existing_hash = parse_first_line_hash(first_line)
    if existing_hash:
        return "has_hash", existing_hash

    lookup_hash = lookup.get(path.name.lower())
    if lookup_hash:
        try:
            ensure_hash_header(path, lookup_hash, apply_changes)
            return "added_hash", lookup_hash
        except OSError:
            return "error_write", lookup_hash

    try:
        remove_file(path, apply_changes)
        return "deleted", None
    except OSError:
        return "error_delete", None


def process_parquet(
    path: Path,
    lookup: Dict[str, str],
    apply_changes: bool,
) -> Tuple[str, Optional[str]]:
    status, detail = parquet_hash_status(path)
    if status == "has_hash":
        return "parquet_has_hash", detail
    if status == "needs_fill" and detail:
        try:
            ensure_param_hash_column(path, detail, apply_changes)
            return "parquet_filled_hash", detail
        except OSError:
            return "parquet_error_write", detail
    if status == "error_read":
        return "parquet_error_read", None

    lookup_hash = lookup_param_hash_for_parquet(path, lookup)
    if not lookup_hash:
        try:
            remove_file(path, apply_changes)
            return "parquet_deleted", None
        except OSError:
            return "parquet_error_delete", None

    try:
        ensure_param_hash_column(path, lookup_hash, apply_changes)
        return "parquet_added_hash", lookup_hash
    except OSError:
        return "parquet_error_write", lookup_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure mi00*.dat files include a param_hash header line. "
            "Ensure mi00*.parquet files include a param_hash column. "
            "If missing, attempt to insert from step_final_simulation_params.csv; "
            "otherwise delete the .dat file."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (write headers and delete files). Default is dry-run.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Path to step_final_simulation_params.csv (optional).",
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        help="Additional root directories to scan (optional).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    default_params = (
        repo_root
        / "MINGO_DIGITAL_TWIN"
        / "SIMULATED_DATA"
        / "step_final_simulation_params.csv"
    )
    params_path = args.params or default_params
    lookup = load_param_hash_lookup(params_path)

    roots = [
        repo_root / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA",
        repo_root / "STATIONS" / "MINGO00",
    ]
    roots.extend(args.root)

    stats = {
        "checked": 0,
        "has_hash": 0,
        "added_hash": 0,
        "deleted": 0,
        "error_read": 0,
        "error_write": 0,
        "error_delete": 0,
        "parquet_checked": 0,
        "parquet_has_hash": 0,
        "parquet_filled_hash": 0,
        "parquet_added_hash": 0,
        "parquet_deleted": 0,
        "parquet_error_read": 0,
        "parquet_error_write": 0,
        "parquet_error_delete": 0,
    }

    for path in iter_mi00_files(roots):
        stats["checked"] += 1
        status, detail = process_file(path, lookup, args.apply)
        if status in stats:
            stats[status] += 1

        if status == "has_hash":
            continue
        if status == "added_hash":
            action = "ADD" if args.apply else "WOULD_ADD"
            print(f"{action} {path} param_hash={detail}")
        elif status == "deleted":
            action = "DELETE" if args.apply else "WOULD_DELETE"
            print(f"{action} {path}")
        elif status.startswith("error_"):
            print(f"ERROR {status} {path}")

    for path in iter_mi00_parquets(roots):
        stats["parquet_checked"] += 1
        status, detail = process_parquet(path, lookup, args.apply)
        if status in stats:
            stats[status] += 1
        if status == "parquet_has_hash":
            continue
        if status == "parquet_filled_hash":
            action = "FILL" if args.apply else "WOULD_FILL"
            print(f"{action} {path} param_hash={detail}")
        elif status == "parquet_added_hash":
            action = "ADD" if args.apply else "WOULD_ADD"
            print(f"{action} {path} param_hash={detail}")
        elif status == "parquet_deleted":
            action = "DELETE" if args.apply else "WOULD_DELETE"
            print(f"{action} {path}")
        elif status.startswith("parquet_error_"):
            print(f"ERROR {status} {path}")

    mode = "APPLY" if args.apply else "DRY_RUN"
    print(f"Mode: {mode}")
    print(
        "Checked: {checked} | Has hash: {has_hash} | Added: {added_hash} | Deleted: {deleted} | "
        "Read errors: {error_read} | Write errors: {error_write} | Delete errors: {error_delete}".format(
            **stats
        )
    )
    print(
        "Parquet checked: {parquet_checked} | Has hash: {parquet_has_hash} | Added: {parquet_added_hash} | "
        "Filled: {parquet_filled_hash} | Deleted: {parquet_deleted} | Read errors: {parquet_error_read} | "
        "Write errors: {parquet_error_write} | Delete errors: {parquet_error_delete}".format(**stats)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
