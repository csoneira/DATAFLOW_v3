#!/usr/bin/env python3
"""Cross-step lineage and integrity validator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common_io import (
    STEP_TO_INTERSTEP,
    StepArtifact,
    compute_exact_manifest_count,
    find_registry_entry,
    get_metadata_for_data_path,
    load_frame,
    resolve_source_dataset_path,
)
from .common_report import RESULT_COLUMNS, ResultBuilder


def _upstream_fingerprint(upstream_meta: dict | None) -> str | None:
    if not isinstance(upstream_meta, dict):
        return None
    payload = {
        "step": upstream_meta.get("step"),
        "config_hash": upstream_meta.get("config_hash"),
        "upstream_hash": upstream_meta.get("upstream_hash"),
    }
    payload_json = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _plot_row_counts(row_counts: dict[str, int | None], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    keys = [k for k in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] if k in row_counts]
    vals = [row_counts[k] if row_counts[k] is not None else 0 for k in keys]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(keys, vals, color="steelblue", alpha=0.8)
    ax.set_title("Row counts by step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(plot_dir / "cross_step_row_counts.png", dpi=140)
    plt.close(fig)


def _load_event_id_set(art: StepArtifact | None) -> set[int] | None:
    if art is None or art.data_path is None or not art.data_path.exists():
        return None
    try:
        df = load_frame(art.data_path, columns=["event_id"])
    except Exception:
        return None
    if "event_id" not in df.columns:
        return None
    return set(df["event_id"].astype(int).tolist())


def run(
    artifacts: dict[str, StepArtifact],
    run_timestamp: str,
    output_dir: Path,
    make_plots: bool = False,
) -> pd.DataFrame:
    art10 = artifacts.get("10")
    rb = ResultBuilder(
        run_timestamp=run_timestamp,
        validator="validate_cross_step_lineage",
        step="cross",
        sim_run=art10.sim_run if art10 else None,
        config_hash=art10.config_hash if art10 else None,
        upstream_hash=art10.upstream_hash if art10 else None,
        n_rows_in=None,
        n_rows_out=None,
    )

    # Per-step registry and metadata integrity.
    row_counts: dict[str, int | None] = {}
    for step_key in [str(i) for i in range(1, 11)]:
        art = artifacts.get(step_key)
        if art is None:
            rb.add(
                test_id=f"cross_step{step_key}_artifact",
                test_name=f"STEP {step_key} artifact discovered",
                metric_name="exists",
                metric_value=0,
                status="SKIP",
            )
            continue

        exists = int(art.exists)
        rb.add(
            test_id=f"cross_step{step_key}_exists",
            test_name=f"STEP {step_key} data exists",
            metric_name="exists",
            metric_value=exists,
            expected_value=1,
            threshold_low=1,
            threshold_high=1,
            status="PASS" if exists else "SKIP",
        )

        if not art.exists:
            row_counts[step_key] = None
            continue

        row_counts[step_key] = art.row_count

        cfg_hash_ok = int(bool(art.config_hash))
        up_hash_ok = int(bool(art.upstream_hash)) if step_key != "1" else 1
        rb.add(
            test_id=f"cross_step{step_key}_hashes",
            test_name=f"STEP {step_key} config/upstream hashes available",
            metric_name="hash_fields_present",
            metric_value=cfg_hash_ok + up_hash_ok,
            expected_value=2,
            threshold_low=2,
            threshold_high=2,
            status="PASS" if cfg_hash_ok + up_hash_ok == 2 else "WARN",
        )

        entry = find_registry_entry(art.interstep_dir, art.sim_run)
        registry_ok = 1 if entry is not None else 0
        rb.add(
            test_id=f"cross_step{step_key}_registry_entry",
            test_name=f"STEP {step_key} SIM_RUN in registry",
            metric_name="entry_exists",
            metric_value=registry_ok,
            expected_value=1,
            threshold_low=1,
            threshold_high=1,
            status="PASS" if registry_ok else "WARN",
            notes=f"sim_run={art.sim_run}",
        )

        # Check manifest row_count by exact concatenation when feasible.
        exact_count, note = compute_exact_manifest_count(art.manifest_path, max_rows_to_scan=2_000_000)
        if note == "exact_count" and exact_count is not None and art.row_count is not None:
            status = "PASS" if int(exact_count) == int(art.row_count) else "FAIL"
            rb.add(
                test_id=f"cross_step{step_key}_manifest_rowcount",
                test_name=f"STEP {step_key} manifest row_count exact check",
                metric_name="difference",
                metric_value=int(exact_count) - int(art.row_count),
                expected_value=0,
                threshold_low=0,
                threshold_high=0,
                status=status,
            )
        elif note == "skipped_large_manifest":
            rb.add(
                test_id=f"cross_step{step_key}_manifest_rowcount",
                test_name=f"STEP {step_key} manifest row_count exact check",
                metric_name="status",
                metric_value=np.nan,
                status="WARN",
                notes="Skipped exact count for large manifest",
            )
        elif note.startswith("missing_chunk"):
            rb.add(
                test_id=f"cross_step{step_key}_manifest_rowcount",
                test_name=f"STEP {step_key} manifest row_count exact check",
                metric_name="status",
                metric_value=np.nan,
                status="FAIL",
                notes=note,
            )

        # Upstream consistency check via metadata hash recomputation.
        meta = art.metadata
        upstream_meta = meta.get("upstream") if isinstance(meta, dict) else None
        recomputed = _upstream_fingerprint(upstream_meta)
        if step_key != "1":
            if recomputed is None or not art.upstream_hash:
                rb.add(
                    test_id=f"cross_step{step_key}_upstream_hash",
                    test_name=f"STEP {step_key} upstream hash recomputation",
                    metric_name="status",
                    metric_value=np.nan,
                    status="WARN",
                    notes="Missing nested upstream metadata",
                )
            else:
                match = int(recomputed == art.upstream_hash)
                rb.add(
                    test_id=f"cross_step{step_key}_upstream_hash",
                    test_name=f"STEP {step_key} upstream hash recomputation",
                    metric_name="match",
                    metric_value=match,
                    expected_value=1,
                    threshold_low=1,
                    threshold_high=1,
                    status="PASS" if match == 1 else "FAIL",
                )

        # source_dataset path is expected from STEP 3 onward.
        # STEP 2 may legitimately omit it depending on generation mode.
        if int(step_key) >= 2:
            src_path = resolve_source_dataset_path(meta, art.run_dir)
            src_exists = int(src_path is not None and src_path.exists())
            source_required = int(step_key) >= 3
            source_status = "PASS" if src_exists else ("WARN" if not source_required else "FAIL")
            rb.add(
                test_id=f"cross_step{step_key}_source_dataset",
                test_name=f"STEP {step_key} source_dataset exists",
                metric_name="path_exists",
                metric_value=src_exists,
                expected_value=1 if source_required else None,
                threshold_low=1 if source_required else None,
                threshold_high=1 if source_required else None,
                status=source_status,
                notes=str(src_path) if src_path is not None else "source_dataset missing",
            )

            # If source metadata is readable, compare hash hints.
            if src_path is not None and src_path.exists() and isinstance(upstream_meta, dict):
                src_meta = get_metadata_for_data_path(src_path)
                src_cfg_hash = src_meta.get("config_hash")
                up_cfg_hash = upstream_meta.get("config_hash")
                if src_cfg_hash is not None and up_cfg_hash is not None:
                    match = int(str(src_cfg_hash) == str(up_cfg_hash))
                    rb.add(
                        test_id=f"cross_step{step_key}_source_hash_match",
                        test_name=f"STEP {step_key} source metadata hash matches nested upstream",
                        metric_name="match",
                        metric_value=match,
                        expected_value=1,
                        threshold_low=1,
                        threshold_high=1,
                        status="PASS" if match == 1 else "WARN",
                    )

    # Row delta expectations across steps.
    pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6"), ("6", "7"), ("7", "8"), ("8", "9"), ("9", "10")]
    same_expected = {("2", "3"), ("3", "4"), ("4", "5"), ("5", "6"), ("6", "7"), ("7", "8"), ("9", "10")}
    selection_expected = {("8", "9")}

    for a, b in pairs:
        na = row_counts.get(a)
        nb = row_counts.get(b)
        if na is None or nb is None:
            rb.add(
                test_id=f"cross_rows_{a}_{b}",
                test_name=f"Row delta STEP {a}->{b}",
                metric_name="delta",
                metric_value=np.nan,
                status="SKIP",
                notes="Missing row_count",
            )
            continue

        delta = int(nb) - int(na)
        ratio = (float(nb) / float(na)) if na else np.nan
        if (a, b) in same_expected:
            status = "PASS" if delta == 0 else "FAIL"
        elif (a, b) in selection_expected:
            status = "PASS" if nb <= na else "FAIL"
        else:
            # STEP 1->2 can naturally reduce strongly due acceptance.
            status = "PASS" if (0 < nb <= na) else "WARN"

        rb.add(
            test_id=f"cross_rows_{a}_{b}",
            test_name=f"Row delta STEP {a}->{b}",
            metric_name="delta",
            metric_value=delta,
            expected_value=0 if (a, b) in same_expected else None,
            status=status,
            notes=f"ratio={ratio:.6f}" if np.isfinite(ratio) else "",
        )

    # Event-id conservation checks.
    id_sets: dict[str, set[int] | None] = {}
    for step_key in [str(i) for i in range(2, 11)]:
        id_sets[step_key] = _load_event_id_set(artifacts.get(step_key))

    # STEP 1 is huge; range check against row_count.
    ids2 = id_sets.get("2")
    n1 = row_counts.get("1")
    if ids2 is not None and n1 is not None:
        out_of_range = int(sum((eid < 0 or eid >= n1) for eid in ids2))
        rb.add(
            test_id="cross_ids_1_2_range",
            test_name="STEP 2 event_id values are within STEP 1 row range",
            metric_name="out_of_range_ids",
            metric_value=out_of_range,
            expected_value=0,
            threshold_low=0,
            threshold_high=0,
            status="PASS" if out_of_range == 0 else "FAIL",
        )

    # Exact set equality where expected.
    exact_pairs = [("2", "3"), ("3", "4"), ("4", "5"), ("5", "6"), ("6", "7"), ("7", "8"), ("9", "10")]
    for a, b in exact_pairs:
        sa = id_sets.get(a)
        sb = id_sets.get(b)
        if sa is None or sb is None:
            rb.add(
                test_id=f"cross_ids_{a}_{b}",
                test_name=f"event_id conservation STEP {a}->{b}",
                metric_name="set_diff",
                metric_value=np.nan,
                status="SKIP",
                notes="Missing event_id set",
            )
            continue
        diff = len(sa.symmetric_difference(sb))
        rb.add(
            test_id=f"cross_ids_{a}_{b}",
            test_name=f"event_id conservation STEP {a}->{b}",
            metric_name="set_diff",
            metric_value=diff,
            expected_value=0,
            threshold_low=0,
            threshold_high=0,
            status="PASS" if diff == 0 else "FAIL",
        )

    # STEP 8->9 should be subset after trigger.
    s8 = id_sets.get("8")
    s9 = id_sets.get("9")
    if s8 is not None and s9 is not None:
        missing = len(s9 - s8)
        rb.add(
            test_id="cross_ids_8_9_subset",
            test_name="STEP 9 event_id subset of STEP 8",
            metric_name="unexpected_ids",
            metric_value=missing,
            expected_value=0,
            threshold_low=0,
            threshold_high=0,
            status="PASS" if missing == 0 else "FAIL",
        )

    if make_plots:
        _plot_row_counts(row_counts, output_dir / "plots" / "validate_cross_step_lineage")

    return rb.to_frame().reindex(columns=RESULT_COLUMNS)
