#!/usr/bin/env python3
"""
Repair orphan simulated .dat / .parquet files whose param_hash is missing
from step_final_simulation_params.csv.

Two categories of orphan are handled:

1. **Hash-not-in-CSV** – the .dat header contains ``# param_hash=XXXX`` but
   no CSV row has that hash.  The script matches the file to an existing
   parameter set (by decoding the date from the filename and finding
   neighbouring CSV entries) and inserts a new row with the file's actual
   hash and the matched physics parameters.

2. **Hash-mismatch** – the CSV has a row for the file_name but with a
   *different* hash (caused by the float/int normalisation bug in
   ``_normalize_hash_value``).  The script rewrites the CSV row's
   ``param_hash`` to the canonical value computed with the fixed normaliser.

Usage
-----
Dry-run (default – no files changed)::

    python repair_orphan_hashes.py

Apply changes::

    python repair_orphan_hashes.py --apply
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Hash computation (mirrors step_final but with the int-normalisation fix)
# ---------------------------------------------------------------------------

PARAM_HASH_FIELDS = (
    "cos_n",
    "flux_cm2_min",
    "z_plane_1",
    "z_plane_2",
    "z_plane_3",
    "z_plane_4",
    "efficiencies",
    "trigger_combinations",
    "requested_rows",
    "sample_start_index",
)


def _normalize_hash_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_normalize_hash_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_hash_value(val) for key, val in value.items()}
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if float(value) == int(value):
            return int(value)
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (
            (stripped.startswith("[") and stripped.endswith("]"))
            or (stripped.startswith("{") and stripped.endswith("}"))
        ):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            return _normalize_hash_value(parsed)
        return stripped
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return str(value)


def compute_param_hash(values: dict) -> str:
    payload = {key: _normalize_hash_value(values.get(key)) for key in PARAM_HASH_FIELDS}
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def extract_hash_from_dat(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="ascii", errors="replace") as fh:
            first_line = fh.readline().strip()
    except OSError:
        return None
    if first_line.startswith("# param_hash="):
        return first_line.split("=", 1)[1].strip() or None
    return None


def decode_filename_date(name: str) -> Optional[datetime]:
    """mi00YYDDDHHMMSS.dat → datetime"""
    m = re.match(r"mi00(\d{2})(\d{3})(\d{2})(\d{2})(\d{2})\.dat", name)
    if not m:
        return None
    yy, ddd = int(m.group(1)), int(m.group(2))
    hh, mm, ss = int(m.group(3)), int(m.group(4)), int(m.group(5))
    year = 2000 + yy
    try:
        dt = datetime(year, 1, 1) + timedelta(days=ddd - 1)
        dt = dt.replace(hour=hh, minute=mm, second=ss)
    except ValueError:
        return None
    return dt


def find_nearest_param_set(
    target_date: datetime,
    df: pd.DataFrame,
) -> Optional[pd.Series]:
    """Return the CSV row whose param_date is closest to *target_date*."""
    if df.empty or "param_date" not in df.columns:
        return None
    dates = pd.to_datetime(df["param_date"], errors="coerce")
    valid = dates.notna()
    if not valid.any():
        return None
    diffs = (dates[valid] - pd.Timestamp(target_date)).abs()
    closest_idx = diffs.idxmin()
    closest_diff = diffs.loc[closest_idx]
    # Only accept if within 30 days — anything further is suspect
    if closest_diff > pd.Timedelta(days=30):
        return None
    return df.loc[closest_idx]


def iter_mi00_dats(roots: List[Path]):
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("mi00*.dat"):
            if path.is_file():
                yield path


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    parser.add_argument("--params", type=Path, default=None, help="Path to step_final_simulation_params.csv")
    parser.add_argument("--root", type=Path, action="append", default=[], help="Additional root dirs to scan")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    default_params = repo_root / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA" / "step_final_simulation_params.csv"
    params_path = args.params or default_params

    if not params_path.exists():
        print(f"ERROR: CSV not found: {params_path}")
        return 1

    df = pd.read_csv(params_path)

    # Build lookups  ─────────────────────────────────────────────────────────
    valid_hashes: set[str] = set()
    filename_to_idx: Dict[str, int] = {}
    for idx, row in df.iterrows():
        h = str(row.get("param_hash") or "").strip()
        fn = str(row.get("file_name") or "").strip()
        if h:
            valid_hashes.add(h)
        if fn:
            filename_to_idx[fn.lower()] = idx

    roots = [
        repo_root / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA",
        repo_root / "STATIONS" / "MINGO00",
    ]
    roots.extend(args.root)

    # Counters  ──────────────────────────────────────────────────────────────
    stats = {
        "scanned": 0,
        "ok": 0,
        "repaired_mismatch": 0,
        "repaired_missing": 0,
        "unrecoverable": 0,
        "no_hash": 0,
    }

    new_rows: list[dict] = []
    updated_indices: Dict[int, str] = {}  # idx → corrected hash

    for dat_path in iter_mi00_dats(roots):
        stats["scanned"] += 1
        file_hash = extract_hash_from_dat(dat_path)
        if not file_hash:
            stats["no_hash"] += 1
            continue
        if file_hash in valid_hashes:
            stats["ok"] += 1
            continue

        fn_lower = dat_path.name.lower()

        # ── Category 1: file_name IS in CSV but hash differs ──────────────
        csv_idx = filename_to_idx.get(fn_lower)
        if csv_idx is not None:
            csv_row = df.loc[csv_idx]
            # Recompute the canonical hash with the fixed normaliser
            canonical = compute_param_hash(csv_row.to_dict())
            if canonical == file_hash:
                # The file's hash matches what the fixed normaliser produces;
                # the CSV stored a stale hash from the buggy normaliser.
                action = "FIX_HASH" if args.apply else "WOULD_FIX_HASH"
                print(f"{action} {dat_path.name}  csv_hash→{file_hash[:24]}…")
                updated_indices[csv_idx] = file_hash
                valid_hashes.add(file_hash)
                stats["repaired_mismatch"] += 1
            else:
                # Hash mismatch that can't be explained by the float/int bug.
                # Still add the file's hash alongside the existing row so the
                # hash lookup works downstream.
                action = "ADD_ALIAS" if args.apply else "WOULD_ADD_ALIAS"
                print(f"{action} {dat_path.name}  file_hash={file_hash[:24]}…  csv_hash={str(csv_row.get('param_hash',''))[:24]}…")
                alias_row = csv_row.to_dict()
                alias_row["param_hash"] = file_hash
                alias_row["file_name"] = dat_path.name
                new_rows.append(alias_row)
                valid_hashes.add(file_hash)
                stats["repaired_mismatch"] += 1
            continue

        # ── Category 2: file_name NOT in CSV — reconstruct entry ──────────
        file_date = decode_filename_date(dat_path.name)
        if file_date is None:
            print(f"SKIP {dat_path} (cannot decode date from filename)")
            stats["unrecoverable"] += 1
            continue

        nearest = find_nearest_param_set(file_date, df)
        if nearest is None:
            print(f"SKIP {dat_path} (no nearby param_date in CSV)")
            stats["unrecoverable"] += 1
            continue

        # Build a new CSV row with the orphan's actual hash and the
        # physics parameters from the nearest matching entry.
        new_row = {
            "file_name": dat_path.name,
            "param_hash": file_hash,
            "param_set_id": nearest.get("param_set_id"),
            "param_date": file_date.strftime("%Y-%m-%d"),
            "cos_n": nearest.get("cos_n"),
            "flux_cm2_min": nearest.get("flux_cm2_min"),
            "z_plane_1": nearest.get("z_plane_1"),
            "z_plane_2": nearest.get("z_plane_2"),
            "z_plane_3": nearest.get("z_plane_3"),
            "z_plane_4": nearest.get("z_plane_4"),
            "efficiencies": nearest.get("efficiencies"),
            "trigger_combinations": nearest.get("trigger_combinations"),
        }
        action = "ADD_ROW" if args.apply else "WOULD_ADD_ROW"
        z = f"[{new_row['z_plane_1']},{new_row['z_plane_2']},{new_row['z_plane_3']},{new_row['z_plane_4']}]"
        print(
            f"{action} {dat_path.name}  "
            f"param_set_id={new_row['param_set_id']}  z={z}  "
            f"hash={file_hash[:24]}…"
        )
        new_rows.append(new_row)
        valid_hashes.add(file_hash)
        stats["repaired_missing"] += 1

    # ── Apply changes ─────────────────────────────────────────────────────
    if args.apply and (updated_indices or new_rows):
        for idx, corrected_hash in updated_indices.items():
            df.at[idx, "param_hash"] = corrected_hash
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        # Atomic write
        import tempfile, os
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{params_path.name}.",
            suffix=".tmp",
            dir=str(params_path.parent),
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, params_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        print(f"\nCSV updated: {params_path}")

    mode = "APPLY" if args.apply else "DRY_RUN"
    print(f"\nMode: {mode}")
    print(
        f"Scanned: {stats['scanned']} | OK: {stats['ok']} | "
        f"Fixed hash: {stats['repaired_mismatch']} | "
        f"Added missing: {stats['repaired_missing']} | "
        f"Unrecoverable: {stats['unrecoverable']} | "
        f"No hash header: {stats['no_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
