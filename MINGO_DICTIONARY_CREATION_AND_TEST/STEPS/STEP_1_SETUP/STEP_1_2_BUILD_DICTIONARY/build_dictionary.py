#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY/build_dictionary.py
Purpose: STEP 1.2 — Dictionary and dataset creation.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY/build_dictionary.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import ast
from itertools import combinations
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
# Support both layouts:
#   - <pipeline>/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY
#   - <pipeline>/STEPS/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY
if STEP_DIR.parents[1].name == "STEPS":
    PIPELINE_DIR = STEP_DIR.parents[2]
else:
    PIPELINE_DIR = STEP_DIR.parents[1]
DEFAULT_CONFIG = PIPELINE_DIR / "config_method.json"
DEFAULT_INPUT = (
    STEP_DIR.parent / "STEP_1_1_COLLECT_DATA" / "OUTPUTS" / "FILES" / "collected_data.csv"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_FEATURE_COLUMNS_PATH = FILES_DIR / "selected_feature_columns.json"

STEP2_INFERENCE_DIR = (
    PIPELINE_DIR / "STEPS" / "STEP_2_INFERENCE"
    if (PIPELINE_DIR / "STEPS" / "STEP_2_INFERENCE").exists()
    else PIPELINE_DIR / "STEP_2_INFERENCE"
)
CONFIG_COLUMNS_PATH = PIPELINE_DIR / "config_columns.json"
if STEP2_INFERENCE_DIR.exists():
    sys.path.insert(0, str(STEP2_INFERENCE_DIR))

try:
    from estimate_parameters import (  # noqa: E402
        _append_derived_physics_feature_columns,
        _append_derived_tt_global_rate_column,
        _auto_feature_columns as _shared_auto_feature_columns,
        _derived_feature_columns as _shared_derived_feature_columns,
        _normalize_derived_physics_features,
    )
    from feature_columns_config import (  # noqa: E402
        parse_explicit_feature_columns,
        resolve_feature_columns_from_catalog,
    )
    _STEP21_FEATURE_HELPERS_AVAILABLE = True
    _STEP21_FEATURE_HELPERS_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - defensive fallback
    _STEP21_FEATURE_HELPERS_AVAILABLE = False
    _STEP21_FEATURE_HELPERS_IMPORT_ERROR = str(exc)

_FIGURE_COUNTER = 0
FIGURE_STEP_PREFIX = "1_2"


def _save_figure(fig: plt.Figure, path: Path, **kwargs) -> None:
    """Save figure with a per-script sequential numeric prefix."""
    global _FIGURE_COUNTER
    _FIGURE_COUNTER += 1
    out_path = Path(path)
    out_path = out_path.with_name(f"{FIGURE_STEP_PREFIX}_{_FIGURE_COUNTER}_{out_path.name}")
    fig.savefig(out_path, **kwargs)


_PLOT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".eps",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}

CANONICAL_TT_LABELS = frozenset(
    {
        "0",
        "1",
        "2",
        "3",
        "4",
        "12",
        "13",
        "14",
        "23",
        "24",
        "34",
        "123",
        "124",
        "134",
        "234",
        "1234",
    }
)
TT_RATE_COLUMN_RE = re.compile(r"^(?P<prefix>.+_tt)_(?P<label>[^_]+)_rate_hz$")


def _clear_plots_dir() -> None:
    """Remove previously generated plot files from the plots directory."""
    removed = 0
    for candidate in PLOTS_DIR.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in _PLOT_EXTENSIONS:
            try:
                candidate.unlink()
                removed += 1
            except OSError as exc:
                log.warning("Could not remove old plot file %s: %s", candidate, exc)
    log.info("Cleared %d plot file(s) from %s", removed, PLOTS_DIR)

logging.basicConfig(
    format="[%(levelname)s] STEP_1.2 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_1.2")


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    def _merge_dicts(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    cfg: dict = {}
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        log.warning("Config file not found: %s", path)

    plots_path = path.with_name("config_plots.json")
    if plots_path != path and plots_path.exists():
        plots_cfg = json.loads(plots_path.read_text(encoding="utf-8"))
        cfg = _merge_dicts(cfg, plots_cfg)
        log.info("Loaded plot config: %s", plots_path)

    runtime_path = path.with_name("config_runtime.json")
    if runtime_path.exists():
        runtime_cfg = json.loads(runtime_path.read_text(encoding="utf-8"))
        cfg = _merge_dicts(cfg, runtime_cfg)
        log.info("Loaded runtime overrides: %s", runtime_path)
    return cfg


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


COMMON_FEATURE_SPACE_KEYS = (
    "feature_columns",
    "derived_features",
    "include_global_rate",
    "global_rate_col",
)


def _merge_common_feature_space_cfg(
    config: dict,
    cfg_21: dict | None = None,
) -> tuple[dict, dict]:
    """Merge top-level common feature-space config with STEP 2.1 overrides."""
    merged: dict = {}
    source_info: dict = {
        "common_keys": [],
        "step_2_1_override_keys": [],
        "has_common_feature_space": False,
    }
    common_cfg = config.get("common_feature_space", {}) if isinstance(config, dict) else {}
    if isinstance(common_cfg, dict):
        source_info["has_common_feature_space"] = True
        for key in COMMON_FEATURE_SPACE_KEYS:
            if key in common_cfg and common_cfg.get(key) is not None:
                merged[key] = common_cfg.get(key)
                source_info["common_keys"].append(key)
    cfg_21_local = cfg_21 if isinstance(cfg_21, dict) else {}
    for key in COMMON_FEATURE_SPACE_KEYS:
        if key in cfg_21_local and cfg_21_local.get(key) is not None:
            merged[key] = cfg_21_local.get(key)
            source_info["step_2_1_override_keys"].append(key)
    return merged, source_info


def _parse_explicit_feature_columns_local(value: object) -> list[str]:
    if _STEP21_FEATURE_HELPERS_AVAILABLE:
        return parse_explicit_feature_columns(value)
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [t.strip() for t in text.split(",") if t.strip()]
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _load_feature_catalog(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_step12_selected_feature_columns(path: Path) -> tuple[list[str], dict]:
    """Load selected feature columns artifact written by STEP 1.2."""
    info = {"path": str(path), "exists": bool(path.exists())}
    if not path.exists():
        return [], info
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        info["error"] = str(exc)
        return [], info

    selected = payload.get("selected_feature_columns", [])
    if not isinstance(selected, list):
        info["error"] = "selected_feature_columns_not_list"
        return [], info
    cols = [str(c).strip() for c in selected if str(c).strip()]
    info["selected_count"] = int(len(cols))
    info["selection_strategy"] = payload.get("selection_strategy", None)
    return cols, info


def _write_step12_selected_feature_columns(
    path: Path,
    *,
    selected_feature_columns: list[str],
    selection_report: dict,
) -> None:
    payload = {
        "selected_feature_columns": [str(c) for c in selected_feature_columns],
        "selection_strategy": "step_1_2_injectivity_selection_v1",
        "selection_report": selection_report if isinstance(selection_report, dict) else {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_feature_pool_for_selection(
    dictionary: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    param_cols: list[str],
) -> list[str]:
    shared = sorted(set(dictionary.columns) & set(dataset.columns))
    blocked = {
        *param_cols,
        "filename_base",
        "param_hash_x",
        "param_hash_y",
        "param_set_id",
        "task_id",
        "is_dictionary_entry",
    }
    blocked_prefixes = ("z_plane_", "eff_sim_", "flux_", "cos_")
    out: list[str] = []
    for col in shared:
        if col in blocked:
            continue
        if col.startswith(blocked_prefixes):
            continue
        if not (
            col.startswith("eff_empirical_")
            or col.startswith("__derived_")
            or col.endswith("_rate_hz")
            or col.startswith("events_per_second_")
        ):
            continue
        d_vals = pd.to_numeric(dictionary[col], errors="coerce").dropna()
        s_vals = pd.to_numeric(dataset[col], errors="coerce").dropna()
        if len(d_vals) < 3 or len(s_vals) < 3:
            continue
        d_span = float(d_vals.max() - d_vals.min())
        if not np.isfinite(d_span) or d_span <= 1e-12:
            continue
        out.append(col)
    return out


def _feature_relevance_score(
    dictionary: pd.DataFrame,
    *,
    feature_col: str,
    param_cols: list[str],
) -> float:
    feat = pd.to_numeric(dictionary[feature_col], errors="coerce")
    score = 0.0
    for pcol in param_cols:
        if pcol not in dictionary.columns:
            continue
        p = pd.to_numeric(dictionary[pcol], errors="coerce")
        mask = feat.notna() & p.notna()
        if int(mask.sum()) < 5:
            continue
        fvals = feat[mask].to_numpy(dtype=float)
        pvals = p[mask].to_numpy(dtype=float)
        if np.nanstd(fvals) <= 1e-12 or np.nanstd(pvals) <= 1e-12:
            continue
        corr = np.corrcoef(fvals, pvals)[0, 1]
        if np.isfinite(corr):
            score = max(score, float(abs(corr)))
    return float(score)


def _score_feature_subset(
    *,
    dictionary: pd.DataFrame,
    param_cols: list[str],
    feature_cols: list[str],
    cv_cfg: dict,
) -> dict:
    local_chk = _cv_check_local_continuity(
        dictionary,
        param_cols,
        feature_cols=feature_cols,
        k=int(cv_cfg.get("local_continuity_k", 10)),
        cv_threshold=float(cv_cfg.get("local_continuity_cv_p95_max", 0.50)),
    )
    topo_chk = _cv_check_bidirectional_topology_continuity(
        dictionary,
        param_cols,
        feature_cols=feature_cols,
        k=int(cv_cfg.get("topology_k", int(cv_cfg.get("local_continuity_k", 10)))),
        overlap_p10_min=float(cv_cfg.get("topology_overlap_p10_min", 0.20)),
        overlap_median_min=float(cv_cfg.get("topology_overlap_median_min", 0.30)),
        forward_expansion_p95_max=float(cv_cfg.get("topology_forward_expansion_p95_max", 8.0)),
        backward_expansion_p95_max=float(cv_cfg.get("topology_backward_expansion_p95_max", 8.0)),
        bad_fraction_max=float(cv_cfg.get("topology_bad_fraction_max", 0.20)),
    )
    inj_chk = _cv_check_local_injectivity(
        dictionary,
        param_cols,
        feature_cols=feature_cols,
        k=int(cv_cfg.get("injectivity_k", int(cv_cfg.get("local_continuity_k", 10)))),
        span_fraction_p95_max=float(cv_cfg.get("injectivity_span_fraction_p95_max", 0.35)),
        flux_span_fraction_p95_max=float(cv_cfg.get("injectivity_flux_span_fraction_p95_max", 0.30)),
        point_span_fraction_max=float(cv_cfg.get("injectivity_point_span_fraction_max", 0.45)),
        bad_fraction_max=float(cv_cfg.get("injectivity_bad_fraction_max", 0.10)),
        span_fraction_p95_max_by_param=(
            cv_cfg.get("injectivity_span_fraction_p95_max_by_param", {})
            if isinstance(cv_cfg.get("injectivity_span_fraction_p95_max_by_param", {}), dict)
            else {}
        ),
        point_span_fraction_max_by_param=(
            cv_cfg.get("injectivity_point_span_fraction_max_by_param", {})
            if isinstance(cv_cfg.get("injectivity_point_span_fraction_max_by_param", {}), dict)
            else {}
        ),
    )

    def _status_penalty(status: str) -> float:
        st = str(status).upper()
        if st == "FAIL":
            return 6.0
        if st == "WARN":
            return 2.0
        if st == "PASS":
            return 0.0
        return 1.0

    disc_frac = float(local_chk.get("discontinuous_fraction", 1.0))
    overlap_med = float(topo_chk.get("overlap_median", 0.0))
    topo_bad_frac = float(topo_chk.get("bad_fraction", 1.0))
    inj_flux_p95 = float(inj_chk.get("flux_span_fraction_p95", 1.0))
    inj_bad_frac = float(inj_chk.get("bad_fraction", 1.0))
    score = (
        4.0 * max(inj_flux_p95, 0.0)
        + 3.0 * max(inj_bad_frac, 0.0)
        + 1.5 * max(disc_frac, 0.0)
        + 1.2 * max(1.0 - overlap_med, 0.0)
        + 1.5 * max(topo_bad_frac, 0.0)
        + _status_penalty(local_chk.get("status", "SKIPPED"))
        + _status_penalty(topo_chk.get("status", "SKIPPED"))
        + _status_penalty(inj_chk.get("status", "SKIPPED"))
    )
    size_penalty = max(float(cv_cfg.get("feature_selection_size_penalty", 0.0)), 0.0)
    score += size_penalty * max(float(len(feature_cols)), 0.0)
    return {
        "score": float(score),
        "size_penalty_per_feature": float(size_penalty),
        "local_continuity": local_chk,
        "topology": topo_chk,
        "injectivity": inj_chk,
    }


def _select_features_for_continuity_and_inference(
    *,
    dictionary: pd.DataFrame,
    dataset: pd.DataFrame,
    param_cols: list[str],
    base_feature_cols: list[str],
    cv_cfg: dict,
) -> tuple[list[str], dict]:
    pool = _candidate_feature_pool_for_selection(
        dictionary,
        dataset,
        param_cols=param_cols,
    )
    if not pool:
        return base_feature_cols, {
            "enabled": True,
            "status": "SKIPPED",
            "reason": "empty_candidate_pool",
            "selected_feature_columns": list(base_feature_cols),
            "candidate_count": 0,
        }

    mandatory = [c for c in base_feature_cols if c in pool]
    for c in ("eff_empirical_1", "eff_empirical_2", "eff_empirical_3", "eff_empirical_4"):
        if c in pool and c not in mandatory:
            mandatory.append(c)
    if "__derived_tt_global_rate_hz" in pool and "__derived_tt_global_rate_hz" not in mandatory:
        mandatory.insert(0, "__derived_tt_global_rate_hz")
    mandatory = list(dict.fromkeys(mandatory))

    default_max_selected = int(
        np.clip(max(len(mandatory), min(len(pool), 48)), 2, 48)
    )
    max_selected = int(
        np.clip(
            int(cv_cfg.get("feature_selection_max_selected_features", default_max_selected)),
            2,
            96,
        )
    )
    relevance_by_col = {
        c: _feature_relevance_score(dictionary, feature_col=c, param_cols=param_cols)
        for c in pool
    }
    ranked_pool = sorted(
        pool,
        key=lambda c: (relevance_by_col.get(c, 0.0), c),
        reverse=True,
    )

    extras = [c for c in pool if c not in mandatory]
    ranked_extras = sorted(
        extras,
        key=lambda c: (relevance_by_col.get(c, 0.0), c),
        reverse=True,
    )
    default_max_extra = int(
        np.clip(
            min(len(extras), max(max_selected - len(mandatory), 0)),
            0,
            96,
        )
    )
    max_extra = int(
        np.clip(
            int(cv_cfg.get("feature_selection_max_extra_features", default_max_extra)),
            0,
            96,
        )
    )
    ranked_extras = ranked_extras[:max_extra]

    candidate_sets: list[list[str]] = []

    def _cap_candidate_size(cols: list[str]) -> list[str]:
        unique_cols = [c for c in dict.fromkeys(cols) if c in pool]
        if len(unique_cols) <= max_selected:
            return unique_cols
        kept: list[str] = []
        for c in mandatory:
            if c in unique_cols and c not in kept:
                kept.append(c)
            if len(kept) >= max_selected:
                return kept[:max_selected]
        for c in ranked_pool:
            if c in unique_cols and c not in kept:
                kept.append(c)
            if len(kept) >= max_selected:
                return kept[:max_selected]
        return kept[:max_selected]

    def _push(cols: list[str]) -> None:
        unique_cols = _cap_candidate_size(cols)
        if not unique_cols:
            return
        if unique_cols not in candidate_sets:
            candidate_sets.append(unique_cols)

    _push(mandatory)
    extra_size_schedule_cfg = cv_cfg.get("feature_selection_extra_size_schedule", None)
    if isinstance(extra_size_schedule_cfg, (list, tuple)):
        extra_size_schedule: list[int] = []
        for val in extra_size_schedule_cfg:
            try:
                extra_size_schedule.append(int(val))
            except (TypeError, ValueError):
                continue
    else:
        extra_size_schedule = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48]
    extra_size_schedule.extend([len(ranked_extras), max_extra])
    for k in sorted(set(int(v) for v in extra_size_schedule if int(v) > 0)):
        if k <= len(ranked_extras):
            _push(mandatory + ranked_extras[:k])

    rate_cols = [c for c in pool if c.endswith("_rate_hz")]
    _push(mandatory + rate_cols)
    _push(pool)

    evals: list[dict] = []
    best_cols = list(base_feature_cols) if base_feature_cols else list(candidate_sets[0])
    best_score = np.inf
    for idx, cols in enumerate(candidate_sets, start=1):
        res = _score_feature_subset(
            dictionary=dictionary,
            param_cols=param_cols,
            feature_cols=cols,
            cv_cfg=cv_cfg,
        )
        rel_vals = [float(relevance_by_col.get(c, 0.0)) for c in cols]
        rel_sum = float(np.sum(rel_vals)) if rel_vals else 0.0
        rel_mean = float(np.mean(rel_vals)) if rel_vals else 0.0
        inj_flux = float(res["injectivity"].get("flux_span_fraction_p95", np.inf))
        inj_bad = float(res["injectivity"].get("bad_fraction", np.inf))
        local_disc = float(res["local_continuity"].get("discontinuous_fraction", np.inf))
        score = float(res.get("score", np.inf))
        eval_entry = {
            "candidate_index": int(idx),
            "n_features": int(len(cols)),
            "feature_columns": cols,
            "score": score,
            "local_status": res["local_continuity"].get("status"),
            "local_discontinuous_fraction": local_disc,
            "topology_status": res["topology"].get("status"),
            "topology_bad_fraction": res["topology"].get("bad_fraction"),
            "injectivity_status": res["injectivity"].get("status"),
            "injectivity_flux_span_p95": inj_flux,
            "injectivity_bad_fraction": inj_bad,
            "topology_overlap_median": res["topology"].get("overlap_median"),
            "feature_relevance_sum": rel_sum,
            "feature_relevance_mean": rel_mean,
            "injectivity_severity": float(inj_flux + 0.2 * inj_bad),
            "size_penalty_per_feature": float(res.get("size_penalty_per_feature", 0.0)),
        }
        evals.append(eval_entry)
        if score < best_score:
            best_score = score
            best_cols = cols

    all_candidates_injectivity_fail = bool(
        evals
        and all(str(e.get("injectivity_status", "")).upper() == "FAIL" for e in evals)
    )
    selection_override_reason = None
    if all_candidates_injectivity_fail:
        # If all candidate sets fail injectivity, prefer the least severe
        # injectivity failure and then higher feature relevance.
        best_eval = min(
            evals,
            key=lambda e: (
                float(e.get("injectivity_bad_fraction", np.inf)),
                float(e.get("injectivity_flux_span_p95", np.inf)),
                float(e.get("local_discontinuous_fraction", np.inf)),
                -float(e.get("feature_relevance_mean", 0.0)),
                -float(e.get("n_features", 0)),
            ),
        )
        best_cols = list(best_eval.get("feature_columns", best_cols))
        best_score = float(best_eval.get("score", best_score))
        selection_override_reason = (
            "all_candidates_injectivity_fail_select_least_bad_injectivity_then_relevance"
        )

    best_eval = None
    for e in evals:
        if e["feature_columns"] == best_cols:
            best_eval = e
            break
    report = {
        "enabled": True,
        "status": "selected",
        "candidate_count": int(len(candidate_sets)),
        "candidate_pool_count": int(len(pool)),
        "max_selected_feature_count": int(max_selected),
        "mandatory_feature_count": int(len(mandatory)),
        "ranked_extra_feature_count": int(len(ranked_extras)),
        "selected_feature_columns": list(best_cols),
        "selected_feature_count": int(len(best_cols)),
        "selected_score": float(best_score),
        "selected_metrics": best_eval,
        "evaluated_candidates": evals,
    }
    if all_candidates_injectivity_fail:
        report["all_candidates_injectivity_fail"] = True
        report["selection_override_reason"] = selection_override_reason
    else:
        report["all_candidates_injectivity_fail"] = False
    return list(best_cols), report


def _resolve_step21_feature_space_for_continuity(
    dictionary: pd.DataFrame,
    dataset: pd.DataFrame,
    config: dict,
    *,
    prefer_step12_selected_artifact: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Resolve STEP 1.2 continuity feature-space from STEP 2.1 config."""
    dict_out = dictionary.copy()
    data_out = dataset.copy()
    fallback_features = [
        c for c in (f"eff_empirical_{p}" for p in (1, 2, 3, 4))
        if c in dict_out.columns and c in data_out.columns
    ]
    info: dict = {
        "feature_source": "step_2_1",
        "feature_columns_config": None,
        "strategy": "fallback_eff_empirical",
        "resolved_feature_columns": fallback_features,
        "resolved_feature_count": int(len(fallback_features)),
    }
    if not _STEP21_FEATURE_HELPERS_AVAILABLE:
        info["warning"] = (
            "STEP_2_INFERENCE feature helpers unavailable; "
            f"fallback to empirical efficiencies ({_STEP21_FEATURE_HELPERS_IMPORT_ERROR})."
        )
        return dict_out, data_out, fallback_features, info

    cfg_21_raw = config.get("step_2_1", {})
    if not isinstance(cfg_21_raw, dict):
        cfg_21_raw = {}
    cfg_21, cfg_sources = _merge_common_feature_space_cfg(config, cfg_21_raw)
    feature_cfg = cfg_21.get("feature_columns", "auto")
    info["feature_columns_config"] = feature_cfg

    include_global_rate = _as_bool(cfg_21.get("include_global_rate", True), True)
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate")).strip()
    derived_cfg = cfg_21.get("derived_features", {})
    if not isinstance(derived_cfg, dict):
        derived_cfg = {}

    selected: list[str] = []
    strategy = "explicit"
    selection_info: dict = {}
    feature_mode = feature_cfg.strip().lower() if isinstance(feature_cfg, str) else ""
    selected_feature_modes = {
        "step12_selected",
        "step_1_2_selected",
        "selected_from_step12",
        "selected_from_step_1_2",
    }
    if feature_mode in selected_feature_modes:
        if not bool(prefer_step12_selected_artifact):
            # When STEP 1.2 is actively selecting features, avoid bootstrapping
            # from a previous run's selected-feature artifact.
            selection_info["fallback_reason"] = "step12_selected_bypassed_for_reselection"
            selection_info["prefer_step12_selected_artifact"] = False
            feature_mode = "derived"
        else:
            selected_path_cfg = cfg_21.get("selected_feature_columns_path", None)
            selected_path = (
                Path(str(selected_path_cfg)).expanduser()
                if selected_path_cfg
                else SELECTED_FEATURE_COLUMNS_PATH
            )
            selected_from_step12_raw, selected_info = _load_step12_selected_feature_columns(selected_path)
            tt_prefix = str(derived_cfg.get("prefix", "last")).strip() or "last"
            trigger_types = _parse_explicit_feature_columns_local(derived_cfg.get("trigger_types", []))
            include_to_tt = _as_bool(derived_cfg.get("include_to_tt_rate_hz", False), False)
            physics_features = _normalize_derived_physics_features(
                derived_cfg.get("physics_features", [])
            )
            dict_out, data_out, derived_rate_col, _ = _append_derived_tt_global_rate_column(
                dict_df=dict_out,
                data_df=data_out,
                prefix_selector=tt_prefix,
                trigger_type_allowlist=trigger_types,
                include_to_tt_rate_hz=include_to_tt,
            )
            if (
                derived_rate_col is None
                and global_rate_col in dict_out.columns
                and global_rate_col in data_out.columns
            ):
                derived_rate_col = global_rate_col
            if derived_rate_col is not None and physics_features:
                dict_out, data_out, _ = _append_derived_physics_feature_columns(
                    dict_df=dict_out,
                    data_df=data_out,
                    rate_column=derived_rate_col,
                    physics_features=physics_features,
                )
            selected_from_step12 = [
                str(c) for c in selected_from_step12_raw
                if str(c) in dict_out.columns and str(c) in data_out.columns
            ]
            if selected_from_step12:
                strategy = "step12_selected"
                selected = selected_from_step12
                selection_info["source"] = "step_1_2.selected_feature_columns"
                selection_info["selected_feature_columns_path"] = str(selected_path)
                selection_info["selected_feature_columns_info"] = selected_info
                info.update(
                    {
                        "strategy": strategy,
                        "resolved_feature_columns": selected,
                        "resolved_feature_count": int(len(selected)),
                        "selection_info": selection_info,
                        "config_sources": cfg_sources,
                    }
                )
                return dict_out, data_out, selected, info
            selection_info["selected_feature_columns_path"] = str(selected_path)
            selection_info["selected_feature_columns_info"] = selected_info
            selection_info["fallback_reason"] = "step12_selected_not_available"
            feature_mode = "derived"

    if feature_mode == "auto":
        strategy = "auto"
        feat_from_dict = _shared_auto_feature_columns(dict_out, include_global_rate, global_rate_col)
        feat_from_data = _shared_auto_feature_columns(data_out, include_global_rate, global_rate_col)
        selected = sorted(set(feat_from_dict) & set(feat_from_data))
        selection_info["source"] = "step_2_1.feature_columns=auto"
    elif feature_mode == "derived":
        strategy = "derived"
        tt_prefix = str(derived_cfg.get("prefix", "last")).strip() or "last"
        trigger_types = _parse_explicit_feature_columns_local(derived_cfg.get("trigger_types", []))
        include_to_tt = _as_bool(derived_cfg.get("include_to_tt_rate_hz", False), False)
        include_trigger_type_rates = _as_bool(
            derived_cfg.get("include_trigger_type_rates", False), False
        )
        include_rate_histogram = _as_bool(derived_cfg.get("include_rate_histogram", False), False)
        physics_features = _normalize_derived_physics_features(
            derived_cfg.get("physics_features", [])
        )

        dict_out, data_out, derived_rate_col, derived_rate_sources = _append_derived_tt_global_rate_column(
            dict_df=dict_out,
            data_df=data_out,
            prefix_selector=tt_prefix,
            trigger_type_allowlist=trigger_types,
            include_to_tt_rate_hz=include_to_tt,
        )
        if (
            derived_rate_col is None
            and global_rate_col in dict_out.columns
            and global_rate_col in data_out.columns
        ):
            derived_rate_col = global_rate_col
        derived_physics_cols: list[str] = []
        if derived_rate_col is not None:
            dict_out, data_out, derived_physics_cols = _append_derived_physics_feature_columns(
                dict_df=dict_out,
                data_df=data_out,
                rate_column=derived_rate_col,
                physics_features=physics_features,
            )
            feat_dict = _shared_derived_feature_columns(
                dict_out,
                rate_column=derived_rate_col,
                trigger_type_rate_columns=(
                    derived_rate_sources if include_trigger_type_rates else None
                ),
                include_rate_histogram=include_rate_histogram,
                physics_feature_columns=derived_physics_cols,
            )
            feat_data = _shared_derived_feature_columns(
                data_out,
                rate_column=derived_rate_col,
                trigger_type_rate_columns=(
                    derived_rate_sources if include_trigger_type_rates else None
                ),
                include_rate_histogram=include_rate_histogram,
                physics_feature_columns=derived_physics_cols,
            )
            selected = sorted(set(feat_dict) & set(feat_data))
        selection_info["source"] = "step_2_1.feature_columns=derived"
        selection_info["derived_options"] = {
            "tt_prefix": tt_prefix,
            "trigger_types": trigger_types,
            "include_to_tt_rate_hz": bool(include_to_tt),
            "include_trigger_type_rates": bool(include_trigger_type_rates),
            "include_rate_histogram": bool(include_rate_histogram),
            "physics_features": physics_features,
            "resolved_global_rate_feature": derived_rate_col,
            "resolved_global_rate_sources": derived_rate_sources,
        }
    elif feature_mode in {"config_columns", "catalog", "config_columns_json"}:
        strategy = "config_columns"
        catalog = _load_feature_catalog(CONFIG_COLUMNS_PATH)
        available = sorted(set(dict_out.columns) & set(data_out.columns))
        if catalog:
            selected, catalog_info = resolve_feature_columns_from_catalog(
                catalog=catalog,
                available_columns=available,
            )
            selection_info.update(catalog_info)
            selection_info["catalog_path"] = str(CONFIG_COLUMNS_PATH)
            selection_info["source"] = "config_columns.json"
        else:
            selected = []
            selection_info["source"] = "config_columns.json_missing"
            selection_info["catalog_path"] = str(CONFIG_COLUMNS_PATH)
        if not selected:
            feat_from_dict = _shared_auto_feature_columns(dict_out, include_global_rate, global_rate_col)
            feat_from_data = _shared_auto_feature_columns(data_out, include_global_rate, global_rate_col)
            selected = sorted(set(feat_from_dict) & set(feat_from_data))
            selection_info["auto_fallback_used"] = True
    else:
        requested = _parse_explicit_feature_columns_local(feature_cfg)
        selected = [c for c in requested if c in dict_out.columns and c in data_out.columns]
        missing = [c for c in requested if c not in selected]
        selection_info["source"] = "step_2_1.feature_columns explicit list"
        selection_info["explicit_requested_count"] = int(len(requested))
        selection_info["explicit_missing_count"] = int(len(missing))
        if missing:
            selection_info["explicit_missing"] = missing

    selected = [str(c).strip() for c in selected if str(c).strip()]
    if not selected:
        strategy = "fallback_eff_empirical"
        selected = fallback_features
        selection_info["fallback_reason"] = "empty_intersection_after_resolution"

    info.update(
        {
            "strategy": strategy,
            "resolved_feature_columns": selected,
            "resolved_feature_count": int(len(selected)),
            "selection_info": selection_info,
            "config_sources": cfg_sources,
        }
    )
    return dict_out, data_out, selected, info

def _parse_efficiencies(value: object) -> list[float]:
    """Parse stringified [e1, e2, e3, e4] into four floats."""
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(value[i]) for i in range(4)]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [np.nan] * 4
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
            return [float(parsed[i]) for i in range(4)]
    return [np.nan] * 4


def _safe_task_ids(raw: object) -> list[int]:
    if raw is None:
        return [1]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return [1]
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = [x.strip() for x in stripped.split(",") if x.strip()]
        raw = parsed
    out: list[int] = []
    if isinstance(raw, (list, tuple)):
        for value in raw:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(set(out)) or [1]


def _resolve_relerr_eff_fit_threshold(cfg_12: dict, plane: int, default: float) -> float:
    """Resolve per-plane fit relative-error threshold, with legacy key fallback."""
    new_key = f"dictionary_relerr_eff_{plane}_fit_max_pct"
    old_key = f"dictionary_relerr_eff_{plane}_max_pct"
    if new_key in cfg_12:
        return cfg_12.get(new_key, default)
    if old_key in cfg_12:
        log.warning(
            "Deprecated config key step_1_2.%s detected; use step_1_2.%s.",
            old_key,
            new_key,
        )
        return cfg_12.get(old_key, default)
    return default


def _preferred_count_prefixes_for_task_ids(task_ids: list[int]) -> list[str]:
    max_task_id = max(task_ids) if task_ids else 1
    if max_task_id <= 1:
        return ["raw_tt_"]
    if max_task_id == 2:
        return ["clean_tt_", "raw_to_clean_tt_", "raw_tt_"]
    if max_task_id == 3:
        return ["cal_tt_", "clean_tt_", "raw_to_clean_tt_", "raw_tt_"]
    if max_task_id == 4:
        return ["list_tt_", "list_to_fit_tt_", "cal_tt_", "clean_tt_", "raw_to_clean_tt_", "raw_tt_"]
    return [
        "post_tt_",
        "fit_to_post_tt_",
        "fit_tt_",
        "list_to_fit_tt_",
        "list_tt_",
        "cal_tt_",
        "clean_tt_",
        "raw_to_clean_tt_",
        "raw_tt_",
        "corr_tt_",
        "task5_to_corr_tt_",
        "fit_to_corr_tt_",
        "definitive_tt_",
    ]


def _compute_eff(n_four: pd.Series, n_three_missing: pd.Series) -> pd.Series:
    """Efficiency = N_four / (N_four + N_three_missing)."""
    denom = n_four + n_three_missing
    return n_four / denom.replace({0: np.nan})


def _find_topology_prefix(
    df: pd.DataFrame,
    preferred_prefixes: list[str] | None = None,
) -> tuple[str, str]:
    """
    Detect trigger-topology prefix and value kind.

    Returns:
      (prefix, value_suffix) where value_suffix is "count" or "rate_hz".
    """
    default_prefixes = [
        "raw_tt_",
        "clean_tt_",
        "cal_tt_",
        "list_tt_",
        "fit_tt_",
        "post_tt_",
        "fit_to_post_tt_",
        "corr_tt_",
        "definitive_tt_",
        "task5_to_corr_tt_",
        "fit_to_corr_tt_",
        "list_to_fit_tt_",
        "raw_to_clean_tt_",
    ]
    ordered_prefixes = preferred_prefixes if preferred_prefixes else default_prefixes
    for prefix in ordered_prefixes:
        if f"{prefix}1234_count" in df.columns:
            return prefix, "count"
    for prefix in ordered_prefixes:
        if f"{prefix}1234_rate_hz" in df.columns:
            return prefix, "rate_hz"
    raise KeyError(
        "No trigger-topology columns found (expected e.g. raw_tt_1234_count or raw_tt_1234_rate_hz)."
    )


def _normalize_tt_label(label: object) -> str:
    text = str(label).strip()
    if not text:
        return ""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(value):
        return ""
    if float(value).is_integer():
        return str(int(value))
    return text


def _derive_global_rate_from_tt_sum(
    df: pd.DataFrame,
    *,
    target_col: str = "events_per_second_global_rate",
) -> str | None:
    by_prefix: dict[str, list[str]] = {}
    for col in df.columns:
        match = TT_RATE_COLUMN_RE.match(str(col))
        if match is None:
            continue
        prefix = str(match.group("prefix")).strip()
        label = _normalize_tt_label(match.group("label"))
        if label not in CANONICAL_TT_LABELS:
            continue
        by_prefix.setdefault(prefix, []).append(col)

    if not by_prefix:
        return None

    prefix_priority = [
        "post_tt",
        "fit_tt",
        "list_tt",
        "cal_tt",
        "clean_tt",
        "raw_tt",
    ]
    selected_prefix: str | None = None
    for prefix in prefix_priority:
        if prefix in by_prefix and by_prefix[prefix]:
            selected_prefix = prefix
            break
    if selected_prefix is None:
        selected_prefix = min(by_prefix.keys(), key=lambda p: (-len(by_prefix[p]), p))

    cols = sorted(set(by_prefix[selected_prefix]))
    if not cols:
        return None

    summed = pd.Series(0.0, index=df.index, dtype=float)
    valid_any = pd.Series(False, index=df.index)
    for col in cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        summed = summed + numeric.fillna(0.0)
        valid_any = valid_any | numeric.notna()

    df[target_col] = summed.where(valid_any, np.nan)
    return target_col


def _pick_global_rate_column(df: pd.DataFrame, preferred: str = "events_per_second_global_rate") -> str | None:
    """Pick/derive global-rate column without 1234-only shortcuts."""
    ordered = [
        preferred,
        "events_per_second_global_rate",
        "global_rate_hz",
        "global_rate_hz_mean",
    ]

    seen: set[str] = set()
    candidates: list[str] = []
    for col in ordered:
        if col and col not in seen:
            candidates.append(col)
            seen.add(col)

    for col in df.columns:
        cl = str(col).strip().lower()
        if not cl or col in seen:
            continue
        if "global_rate" in cl and ("hz" in cl or cl.endswith("_rate")):
            candidates.append(col)
            seen.add(col)

    for col in candidates:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            return col
    for col in candidates:
        if col in df.columns:
            return col

    return _derive_global_rate_from_tt_sum(df, target_col="events_per_second_global_rate")


def _fit_simulated_vs_empirical(
    df: pd.DataFrame,
    plane: int,
    poly_order: int = 3,
) -> np.ndarray | None:
    """Fit simulated = P(empirical) for one plane using polynomial order *poly_order*."""
    sim_col = f"eff_sim_{plane}"
    emp_col = f"eff_empirical_{plane}"
    sim = pd.to_numeric(df.get(sim_col), errors="coerce")
    emp = pd.to_numeric(df.get(emp_col), errors="coerce")
    m = sim.notna() & emp.notna()
    if m.sum() < 2:
        return None
    x = emp[m].to_numpy(dtype=float)
    y = sim[m].to_numpy(dtype=float)
    requested_order = max(1, int(poly_order))
    used_order = min(requested_order, int(len(x) - 1))
    if used_order < 1:
        return None
    coeffs = np.polyfit(x, y, used_order)
    return np.asarray(coeffs, dtype=float)


def _fit_isotonic_calibration(
    df: pd.DataFrame,
    plane: int,
    n_grid: int = 500,
) -> dict | None:
    """Fit monotonic calibration mapping empirical → simulated using isotonic regression.

    Returns a dict with 'x_knots' and 'y_knots' arrays defining a piecewise-linear
    monotonic mapping, plus 'slope_lo' and 'slope_hi' for principled linear
    extrapolation beyond the calibration support.
    """
    sim_col = f"eff_sim_{plane}"
    emp_col = f"eff_empirical_{plane}"
    sim = pd.to_numeric(df.get(sim_col), errors="coerce")
    emp = pd.to_numeric(df.get(emp_col), errors="coerce")
    m = sim.notna() & emp.notna()
    if m.sum() < 2:
        return None
    x = emp[m].to_numpy(dtype=float)
    y = sim[m].to_numpy(dtype=float)

    # Sort by empirical efficiency
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Pool Adjacent Violators Algorithm (PAVA) for isotonic regression
    n = len(y_sorted)
    blocks: list[list[int]] = [[i] for i in range(n)]
    values: list[float] = [float(y_sorted[i]) for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        if values[i] > values[i + 1]:
            # Merge blocks and average
            merged_indices = blocks[i] + blocks[i + 1]
            merged_value = float(np.mean(y_sorted[merged_indices]))
            blocks[i] = merged_indices
            values[i] = merged_value
            del blocks[i + 1]
            del values[i + 1]
            # Step back to check previous block
            if i > 0:
                i -= 1
        else:
            i += 1

    # Build the isotonic fit at every data point
    y_iso = np.empty(n, dtype=float)
    for block, val in zip(blocks, values):
        for idx in block:
            y_iso[idx] = val

    # Create a compact piecewise-linear representation on a regular grid
    x_min = float(x_sorted[0])
    x_max = float(x_sorted[-1])
    if x_max <= x_min:
        return None
    grid_n = max(10, min(int(n_grid), n))
    x_grid = np.linspace(x_min, x_max, grid_n)
    y_grid = np.interp(x_grid, x_sorted, y_iso)

    # Simplify: remove interior points where the slope doesn't change (collinear)
    keep = [0]
    for i in range(1, len(x_grid) - 1):
        dx1 = x_grid[i] - x_grid[keep[-1]]
        dy1 = y_grid[i] - y_grid[keep[-1]]
        dx2 = x_grid[i + 1] - x_grid[keep[-1]]
        dy2 = y_grid[i + 1] - y_grid[keep[-1]]
        # Check if point is collinear with previous kept point and next point
        cross = abs(dx1 * dy2 - dx2 * dy1)
        if cross > 1e-10:
            keep.append(i)
    keep.append(len(x_grid) - 1)
    x_knots = x_grid[keep]
    y_knots = y_grid[keep]

    # Estimate boundary derivatives for extrapolation.
    # Using only isotonic knot secants can collapse to zero at boundaries when
    # the fit has a flat plateau, which creates artificial saturation outside
    # support. Blend secants with robust tail slopes from raw data.
    def _tail_slope(
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        *,
        side: str,
        frac: float = 0.12,
        min_points: int = 30,
    ) -> float:
        n_vals = len(x_vals)
        if n_vals < 3:
            return float("nan")
        n_tail = max(int(np.ceil(frac * n_vals)), min_points, 3)
        n_tail = min(n_tail, n_vals)
        if side == "low":
            x_tail = x_vals[:n_tail]
            y_tail = y_vals[:n_tail]
        else:
            x_tail = x_vals[-n_tail:]
            y_tail = y_vals[-n_tail:]
        m_tail = np.isfinite(x_tail) & np.isfinite(y_tail)
        if int(np.sum(m_tail)) < 3:
            return float("nan")
        x_tail = x_tail[m_tail]
        y_tail = y_tail[m_tail]
        x_span = float(np.nanmax(x_tail) - np.nanmin(x_tail))
        if not np.isfinite(x_span) or x_span <= 1e-10:
            return float("nan")
        try:
            slope = float(np.polyfit(x_tail, y_tail, 1)[0])
        except Exception:
            return float("nan")
        return slope if np.isfinite(slope) else float("nan")

    slope_lo_secant = np.nan
    slope_hi_secant = np.nan
    if len(x_knots) >= 2:
        dx_lo = float(x_knots[1] - x_knots[0])
        if dx_lo > 1e-12:
            slope_lo_secant = float((y_knots[1] - y_knots[0]) / dx_lo)
        dx_hi = float(x_knots[-1] - x_knots[-2])
        if dx_hi > 1e-12:
            slope_hi_secant = float((y_knots[-1] - y_knots[-2]) / dx_hi)

    slope_lo_tail = _tail_slope(x_sorted, y_sorted, side="low")
    slope_hi_tail = _tail_slope(x_sorted, y_sorted, side="high")

    slope_lo_candidates = [v for v in (slope_lo_secant, slope_lo_tail) if np.isfinite(v)]
    slope_hi_candidates = [v for v in (slope_hi_secant, slope_hi_tail) if np.isfinite(v)]
    slope_lo = max(slope_lo_candidates) if slope_lo_candidates else 0.0
    slope_hi = max(slope_hi_candidates) if slope_hi_candidates else 0.0
    slope_lo = float(np.clip(slope_lo, 0.0, 3.0))
    slope_hi = float(np.clip(slope_hi, 0.0, 3.0))

    return {
        "x_knots": [float(v) for v in x_knots],
        "y_knots": [float(v) for v in y_knots],
        "slope_lo": slope_lo,
        "slope_hi": slope_hi,
        "x_min": x_min,
        "x_max": x_max,
        "n_data_points": n,
        "slope_lo_secant": float(slope_lo_secant) if np.isfinite(slope_lo_secant) else None,
        "slope_hi_secant": float(slope_hi_secant) if np.isfinite(slope_hi_secant) else None,
        "slope_lo_tail": float(slope_lo_tail) if np.isfinite(slope_lo_tail) else None,
        "slope_hi_tail": float(slope_hi_tail) if np.isfinite(slope_hi_tail) else None,
        "extrapolation_model": "asymptotic_monotonic",
    }


def _format_polynomial(
    coeffs: np.ndarray | list[float] | tuple[float, ...],
    *,
    variable: str = "x",
    precision: int = 4,
) -> str:
    """Return compact human-readable polynomial expression."""
    arr = np.asarray(coeffs, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return "invalid"
    degree = arr.size - 1
    parts: list[tuple[str, str]] = []
    for idx, coef in enumerate(arr):
        if not np.isfinite(coef) or abs(float(coef)) < 1e-14:
            continue
        power = degree - idx
        mag = f"{abs(float(coef)):.{precision}g}"
        if power == 0:
            term = mag
        elif power == 1:
            term = variable if np.isclose(abs(float(coef)), 1.0) else f"{mag}{variable}"
        else:
            term = f"{variable}^{power}" if np.isclose(abs(float(coef)), 1.0) else f"{mag}{variable}^{power}"
        sign = "-" if coef < 0 else "+"
        parts.append((sign, term))
    if not parts:
        return "0"
    first_sign, first_term = parts[0]
    expr = f"-{first_term}" if first_sign == "-" else first_term
    for sign, term in parts[1:]:
        expr += f" {sign} {term}"
    return expr


def _compute_fit_relerr_columns(
    df: pd.DataFrame,
    plane: int,
    coeffs: np.ndarray | list[float] | tuple[float, ...],
) -> None:
    """Add fit-prediction and fit-relative-error columns for one plane."""
    sim_col = f"eff_sim_{plane}"
    emp_col = f"eff_empirical_{plane}"
    fit_col = f"eff_fitline_{plane}"
    re_col = f"relerr_eff_{plane}_fit_pct"
    abs_re_col = f"abs_relerr_eff_{plane}_fit_pct"

    sim = pd.to_numeric(df.get(sim_col), errors="coerce")
    emp = pd.to_numeric(df.get(emp_col), errors="coerce")
    fit = pd.Series(np.polyval(np.asarray(coeffs, dtype=float), emp.to_numpy(dtype=float)), index=sim.index)
    df[fit_col] = fit
    df[re_col] = (sim - fit) / fit.replace({0: np.nan}) * 100.0
    df[abs_re_col] = df[re_col].abs()


def _mask_all_sim_eff_equal(df: pd.DataFrame) -> np.ndarray:
    """Rows where eff_sim_1..4 are all finite and equal (within numerical tolerance)."""
    eff_cols = [f"eff_sim_{i}" for i in range(1, 5)]
    if not all(col in df.columns for col in eff_cols):
        return np.zeros(len(df), dtype=bool)
    eff_arrays = [
        pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        for col in eff_cols
    ]
    finite = np.isfinite(eff_arrays[0])
    for arr in eff_arrays[1:]:
        finite &= np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(len(df), dtype=bool)
    equal_mask = finite.copy()
    ref = eff_arrays[0]
    atol = 1e-12
    for arr in eff_arrays[1:]:
        equal_mask &= np.isclose(arr, ref, rtol=0.0, atol=atol)
    return equal_mask


def _mask_sim_eff_within_tolerance_band(
    df: pd.DataFrame,
    tolerance_pct: float,
) -> np.ndarray:
    """Rows where all 4 simulated efficiencies are finite and within one band."""
    n_rows = len(df)
    if n_rows == 0:
        return np.zeros(0, dtype=bool)

    tol_pct = float(tolerance_pct)
    if not np.isfinite(tol_pct):
        tol_pct = 10.0
    tol_pct = max(0.0, tol_pct)
    tol_abs = tol_pct / 100.0

    eff_cols = [f"eff_sim_{i}" for i in range(1, 5)]
    if all(col in df.columns for col in eff_cols):
        eff_mat = np.column_stack(
            [pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in eff_cols]
        )
    elif "efficiencies" in df.columns:
        parsed = df["efficiencies"].apply(_parse_efficiencies)
        eff_mat = np.asarray(parsed.tolist(), dtype=float)
        if eff_mat.ndim != 2 or eff_mat.shape[1] < 4:
            return np.zeros(n_rows, dtype=bool)
        eff_mat = eff_mat[:, :4]
    else:
        return np.zeros(n_rows, dtype=bool)

    finite = np.isfinite(eff_mat).all(axis=1)
    if not np.any(finite):
        return np.zeros(n_rows, dtype=bool)
    span = np.full(n_rows, np.nan, dtype=float)
    eff_finite = eff_mat[finite]
    span[finite] = np.max(eff_finite, axis=1) - np.min(eff_finite, axis=1)
    return finite & (span <= (tol_abs + 1e-12))


# ── Dictionary coverage helpers ──────────────────────────────────────────

def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone-chain convex hull (2-D)."""
    if len(points) <= 1:
        return points.copy()
    pts = np.unique(points, axis=0)
    if len(pts) <= 1:
        return pts
    pts_list = sorted((float(x), float(y)) for x, y in pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts_list:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts_list):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula for simple polygon area."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _nearest_neighbor_distances(points: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Per-point nearest-neighbour distances (brute force, chunked)."""
    n = len(points)
    if n < 2:
        return np.full(n, np.nan)
    out = np.full(n, np.nan, dtype=float)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        diff = points[s:e, None, :] - points[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        ri = np.arange(e - s)
        ci = np.arange(s, e)
        d2[ri, ci] = np.inf
        out[s:e] = np.sqrt(np.min(d2, axis=1))
    return out


def _min_distance_to_points(queries: np.ndarray, points: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """For each query, distance to nearest point in *points*."""
    if len(points) == 0:
        return np.full(len(queries), np.nan)
    out = np.full(len(queries), np.nan, dtype=float)
    for s in range(0, len(queries), chunk):
        e = min(len(queries), s + chunk)
        diff = queries[s:e, None, :] - points[None, :, :]
        out[s:e] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return out


def _sanitize_plot_token(token: str) -> str:
    out = []
    for char in str(token):
        if char.isalnum() or char in {"_", "-"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "param"


def _plot_dictionary_coverage(
    dictionary: pd.DataFrame,
    path,
    x_col: str,
    y_col: str,
) -> None:
    """Coverage diagnostics in normalized 2D parameter space."""
    if x_col not in dictionary.columns or y_col not in dictionary.columns:
        log.warning("Cannot plot dictionary coverage: missing %s or %s", x_col, y_col)
        return
    x_vals = pd.to_numeric(dictionary[x_col], errors="coerce")
    y_vals = pd.to_numeric(dictionary[y_col], errors="coerce")
    mask = x_vals.notna() & y_vals.notna()
    if int(mask.sum()) < 3:
        log.warning("Too few dictionary points for coverage plot in (%s, %s).", x_col, y_col)
        return
    x = x_vals[mask].to_numpy(dtype=float)
    y = y_vals[mask].to_numpy(dtype=float)

    # Normalize to [0, 1] bounding box
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0
    xy = np.column_stack([(x - x_min) / x_span, (y - y_min) / y_span])
    unique_xy = np.unique(xy, axis=0)
    if len(unique_xy) < 3:
        log.warning(
            "Too few unique dictionary points for coverage plot in (%s, %s).",
            x_col,
            y_col,
        )
        return

    # NN distances
    nn_d = _nearest_neighbor_distances(unique_xy)
    nn_valid = nn_d[np.isfinite(nn_d)]

    # MC coverage by radius
    rng = np.random.default_rng(42)
    q = rng.random((5000, 2))
    q_min_d = _min_distance_to_points(q, unique_xy)
    # Convert to "percent of bbox side" units
    nn_pct = nn_valid * 100.0
    q_min_pct = q_min_d[np.isfinite(q_min_d)] * 100.0

    # Coverage curve — extend radii to reach full coverage
    radii_extended = np.linspace(0.005, 0.60, 60)
    coverage_pct = np.array(
        [float(100.0 * np.mean(q_min_d <= r)) for r in radii_extended],
        dtype=float,
    )
    radii_pct = radii_extended * 100.0

    # Percentiles / thresholds
    nn_p50, nn_p90, nn_p95 = np.nanpercentile(nn_pct, [50, 90, 95])
    q_p50, q_p90, q_p95 = np.nanpercentile(q_min_pct, [50, 90, 95])
    cov_targets = [80, 90, 95]
    cov_radius_at = {}
    for t in cov_targets:
        idx = np.where(coverage_pct >= t)[0]
        cov_radius_at[t] = float(radii_pct[idx[0]]) if len(idx) else np.nan

    # Plot (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Top-left: normalized dictionary map colored by local spacing
    ax = axes[0, 0]
    sc = ax.scatter(
        unique_xy[:, 0],
        unique_xy[:, 1],
        c=nn_d * 100.0,
        cmap="viridis",
        s=35,
        alpha=0.9,
        edgecolors="0.25",
        linewidths=0.3,
    )
    hull = _convex_hull(unique_xy)
    if len(hull) >= 3:
        closed = np.vstack([hull, hull[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#E45756", linewidth=1.5,
                linestyle="--", label="Convex hull")
        ax.fill(closed[:, 0], closed[:, 1], color="#E45756", alpha=0.08, zorder=0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("NN distance [% bbox side]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Normalized {x_col}")
    ax.set_ylabel(f"Normalized {y_col}")
    ax.set_title("Dictionary map (color = local spacing)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Top-right: NN spacing histogram with percentile guides
    ax = axes[0, 1]
    bins = max(20, min(60, int(np.sqrt(len(nn_pct)) * 3)))
    ax.hist(nn_pct, bins=bins, color="#F58518", alpha=0.85, edgecolor="white")
    ax.axvline(nn_p50, color="#1D3557", linestyle="--", linewidth=1.2, label=f"P50={nn_p50:.2f}%")
    ax.axvline(nn_p90, color="#2A9D8F", linestyle="-.", linewidth=1.2, label=f"P90={nn_p90:.2f}%")
    ax.axvline(nn_p95, color="#E63946", linestyle="-.", linewidth=1.2, label=f"P95={nn_p95:.2f}%")
    ax.set_xlabel("NN distance [% bbox side]")
    ax.set_ylabel("Count")
    ax.set_title("Nearest-neighbor spacing distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Bottom-left: CDF of random-point nearest dictionary distance
    ax = axes[1, 0]
    sorted_q = np.sort(q_min_pct)
    cdf = np.linspace(0.0, 100.0, len(sorted_q))
    ax.plot(sorted_q, cdf, color="#4C78A8", linewidth=1.8)
    ax.axvline(q_p50, color="#1D3557", linestyle="--", linewidth=1.2, label=f"P50={q_p50:.2f}%")
    ax.axvline(q_p90, color="#2A9D8F", linestyle="-.", linewidth=1.2, label=f"P90={q_p90:.2f}%")
    ax.axvline(q_p95, color="#E63946", linestyle="-.", linewidth=1.2, label=f"P95={q_p95:.2f}%")
    ax.set_xlim(0.0, max(1.0, float(np.nanmax(sorted_q)) * 1.05))
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Distance to nearest dictionary point [% bbox side]")
    ax.set_ylabel("Random points with distance <= x [%]")
    ax.set_title("Coverage gap CDF (Monte Carlo)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Bottom-right: coverage by radius with operating-point markers
    ax = axes[1, 1]
    ax.plot(radii_pct, coverage_pct, "o-", color="#4C78A8", markersize=3, linewidth=1.5)
    for t in cov_targets:
        ax.axhline(t, color="0.65", linestyle=":", linewidth=1.0)
        rr = cov_radius_at[t]
        if np.isfinite(rr):
            ax.axvline(rr, color="#2A9D8F", linestyle="-.", linewidth=1.2)
            ax.scatter([rr], [t], color="#2A9D8F", s=25, zorder=3)
            ax.text(rr + 0.8, t + 1.0, f"{t}% @ {rr:.2f}%", fontsize=8, color="#2A9D8F")
    ax.set_xlim(0.0, radii_pct[-1])
    ax.set_ylim(0.0, 105.0)
    ax.set_xlabel("Coverage radius [% bbox side]")
    ax.set_ylabel("Covered random points [%]")
    ax.set_title("Dictionary filling vs coverage radius")
    ax.grid(True, alpha=0.2)

    # Annotate key metrics
    hull = _convex_hull(unique_xy)
    hull_pct = _polygon_area(hull) * 100.0
    r90 = cov_radius_at.get(90, np.nan)
    info = (f"Unique pts: {len(unique_xy)} | "
            f"Hull area: {hull_pct:.1f}% of bbox | "
            f"NN median: {np.median(nn_pct):.2f}% | "
            f"Radius@90% cov: {r90:.2f}%")
    fig.suptitle(f"{x_col} vs {y_col} | {info}", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Dictionary continuity validation
# ═══════════════════════════════════════════════════════════════════

def _validate_dictionary_continuity(
    dictionary: pd.DataFrame,
    dataset: pd.DataFrame,
    param_cols: list[str],
    cfg: dict,
    isotonic_by_plane: dict[int, dict],
    feature_cols: list[str] | None = None,
    feature_resolution: dict | None = None,
) -> tuple[str, dict, list[str]]:
    """Validate dictionary continuity and coverage.

    Runs continuity checks on the dictionary after selection:
      1. Parameter-space coverage (NN distances)
      2. Local continuity (feature → parameter CV)
      3. Bidirectional neighborhood continuity (parameter ↔ feature topology)
      4. Local injectivity in feature -> parameter mapping
      5. Isotonic calibration bounds
      6. Support adequacy per efficiency plane
      7. Density uniformity in flux × eff space

    Returns (overall_status, metrics_dict, messages_list).
    overall_status is "PASS", "WARN", or "FAIL".
    """
    checks: dict[str, dict] = {}
    messages: list[str] = []
    if feature_cols:
        messages.append(
            f"feature_space={len(feature_cols)} cols (source={((feature_resolution or {}).get('strategy', 'unknown'))})"
        )

    # ── Check 1: Parameter-Space Coverage ─────────────────────────
    check1 = _cv_check_param_coverage(dictionary, dataset, param_cols)
    checks["param_space_coverage"] = check1
    if check1["status"] == "SKIPPED":
        messages.append(
            f"param_space_coverage=SKIPPED ({check1.get('reason', 'no_reason')})"
        )
    else:
        messages.append(
            f"param_space_coverage={check1['status']} "
            f"(nn_p95_pct={check1.get('nn_p95_pct', np.nan):.2f}%, threshold=25.00%)"
        )

    # ── Check 2: Local Continuity ─────────────────────────────────
    k = cfg.get("local_continuity_k", 10)
    cv_max = cfg.get("local_continuity_cv_p95_max", 0.50)
    check2 = _cv_check_local_continuity(
        dictionary,
        param_cols,
        feature_cols=feature_cols,
        k=k,
        cv_threshold=cv_max,
    )
    checks["local_continuity"] = check2
    if check2["status"] == "SKIPPED":
        messages.append(
            f"local_continuity=SKIPPED ({check2.get('reason', 'no_reason')})"
        )
    else:
        messages.append(
            f"local_continuity={check2['status']} "
            f"(n_discontinuous={check2.get('n_discontinuous', 'na')}, "
            f"fraction={check2.get('discontinuous_fraction', np.nan):.4f}, "
            f"worst_param={check2.get('worst_param', 'na')})"
        )

    # ── Check 3: Bidirectional neighborhood continuity ────────────
    topo_k = cfg.get("topology_k", k)
    topo_overlap_p10_min = cfg.get("topology_overlap_p10_min", 0.20)
    topo_overlap_median_min = cfg.get("topology_overlap_median_min", 0.30)
    topo_fwd_p95_max = cfg.get("topology_forward_expansion_p95_max", 8.0)
    topo_bwd_p95_max = cfg.get("topology_backward_expansion_p95_max", 8.0)
    topo_bad_fraction_max = cfg.get("topology_bad_fraction_max", 0.20)
    check3 = _cv_check_bidirectional_topology_continuity(
        dictionary,
        param_cols,
        feature_cols=feature_cols,
        k=topo_k,
        overlap_p10_min=topo_overlap_p10_min,
        overlap_median_min=topo_overlap_median_min,
        forward_expansion_p95_max=topo_fwd_p95_max,
        backward_expansion_p95_max=topo_bwd_p95_max,
        bad_fraction_max=topo_bad_fraction_max,
    )
    checks["topology_bidirectional_continuity"] = check3
    if check3["status"] == "SKIPPED":
        messages.append(
            f"topology_bidirectional_continuity=SKIPPED ({check3.get('reason', 'no_reason')})"
        )
    else:
        messages.append(
            f"topology_bidirectional_continuity={check3['status']} "
            f"(overlap_p10={check3.get('overlap_p10', np.nan):.3f}, "
            f"overlap_median={check3.get('overlap_median', np.nan):.3f}, "
            f"forward_expansion_p95={check3.get('forward_expansion_p95', np.nan):.3f}, "
            f"backward_expansion_p95={check3.get('backward_expansion_p95', np.nan):.3f}, "
            f"bad_fraction={check3.get('bad_fraction', np.nan):.4f})"
        )

    # ── Check 4: Local Injectivity ────────────────────────────────
    injectivity_enabled = bool(cfg.get("injectivity_enabled", True))
    if injectivity_enabled:
        injectivity_k = cfg.get("injectivity_k", k)
        injectivity_span_p95_max = cfg.get("injectivity_span_fraction_p95_max", 0.35)
        injectivity_flux_span_p95_max = cfg.get(
            "injectivity_flux_span_fraction_p95_max",
            injectivity_span_p95_max,
        )
        injectivity_point_span_max = cfg.get(
            "injectivity_point_span_fraction_max",
            max(float(injectivity_span_p95_max), float(injectivity_flux_span_p95_max)),
        )
        injectivity_bad_fraction_max = cfg.get("injectivity_bad_fraction_max", 0.10)
        injectivity_span_p95_max_by_param = cfg.get("injectivity_span_fraction_p95_max_by_param", {})
        injectivity_point_span_max_by_param = cfg.get("injectivity_point_span_fraction_max_by_param", {})
        check4 = _cv_check_local_injectivity(
            dictionary,
            param_cols,
            feature_cols=feature_cols,
            k=injectivity_k,
            span_fraction_p95_max=injectivity_span_p95_max,
            flux_span_fraction_p95_max=injectivity_flux_span_p95_max,
            point_span_fraction_max=injectivity_point_span_max,
            bad_fraction_max=injectivity_bad_fraction_max,
            span_fraction_p95_max_by_param=injectivity_span_p95_max_by_param,
            point_span_fraction_max_by_param=injectivity_point_span_max_by_param,
        )
    else:
        check4 = {"status": "SKIPPED", "reason": "injectivity_disabled"}
    checks["local_injectivity"] = check4
    if check4["status"] == "SKIPPED":
        messages.append(
            f"local_injectivity=SKIPPED ({check4.get('reason', 'no_reason')})"
        )
    else:
        messages.append(
            f"local_injectivity={check4['status']} "
            f"(flux_span_p95={check4.get('flux_span_fraction_p95', np.nan):.3f}, "
            f"bad_fraction={check4.get('bad_fraction', np.nan):.4f})"
        )

    # ── Check 5: Isotonic Calibration Bounds ──────────────────────
    check5 = _cv_check_isotonic_bounds(isotonic_by_plane)
    checks["isotonic_bounds"] = check5
    if check5["status"] == "SKIPPED":
        messages.append(
            f"isotonic_bounds=SKIPPED ({check5.get('reason', 'no_reason')})"
        )
    elif check5["status"] == "FAIL":
        messages.append(
            f"isotonic_bounds=FAIL (violations={check5.get('violations', [])})"
        )
    else:
        messages.append(
            f"isotonic_bounds=PASS (planes_checked={check5.get('planes_checked', [])})"
        )

    # ── Check 6: Support Adequacy ─────────────────────────────────
    oos_max = cfg.get("out_of_support_max_fraction", 0.30)
    check6 = _cv_check_support_adequacy(dictionary, dataset, oos_max=oos_max)
    checks["support_adequacy"] = check6
    oos_by_plane = check6.get("out_of_support_fraction_by_plane", {})
    if oos_by_plane:
        worst_plane = max(oos_by_plane, key=oos_by_plane.get)
        worst_frac = float(oos_by_plane[worst_plane])
        messages.append(
            f"support_adequacy={check6['status']} "
            f"(worst_plane={worst_plane}, out_of_support_fraction={worst_frac:.4f}, "
            f"threshold={float(oos_max):.4f})"
        )
    else:
        messages.append("support_adequacy=SKIPPED (no_plane_overlap)")

    # ── Check 7: Density Uniformity ───────────────────────────────
    n_bins = cfg.get("density_grid_bins", 5)
    ratio_max = cfg.get("density_ratio_max", 20.0)
    check7 = _cv_check_density_uniformity(dictionary, n_bins=n_bins, ratio_max=ratio_max)
    checks["density_uniformity"] = check7
    if check7["status"] == "SKIPPED":
        messages.append(
            f"density_uniformity=SKIPPED ({check7.get('reason', 'no_reason')})"
        )
    else:
        messages.append(
            f"density_uniformity={check7['status']} "
            f"(density_ratio={check7.get('density_ratio', np.nan):.2f}, "
            f"threshold={float(ratio_max):.2f}, "
            f"empty_bin_fraction={check7.get('empty_bin_fraction', np.nan):.4f})"
        )

    # ── Overall status ────────────────────────────────────────────
    statuses = [c["status"] for c in checks.values() if c["status"] != "SKIPPED"]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    metrics = {
        "enabled": True,
        "status": overall,
        "feature_columns_used": (
            list(check2.get("feature_columns", []))
            if check2.get("status") != "SKIPPED"
            else (list(feature_cols) if feature_cols else [])
        ),
        "feature_column_count": int(
            len(
                list(check2.get("feature_columns", []))
                if check2.get("status") != "SKIPPED"
                else (list(feature_cols) if feature_cols else [])
            )
        ),
        "feature_resolution": feature_resolution if isinstance(feature_resolution, dict) else None,
        "checks": checks,
        "messages": messages,
    }
    return overall, metrics, messages


def _cv_metrics_for_json(cv_metrics: dict) -> dict:
    """Return a copy of cv_metrics with large per-entry arrays stripped for JSON."""
    import copy
    out = copy.deepcopy(cv_metrics)
    checks = out.get("checks", {})
    for key in ("param_space_coverage", "local_continuity"):
        chk = checks.get(key, {})
        chk.pop("nn_distances_normalized", None)
        chk.pop("nn_distances_pct", None)
        chk.pop("worst_cv_per_entry", None)
        chk.pop("row_indices", None)
    topo = checks.get("topology_bidirectional_continuity", {})
    topo.pop("overlap_per_entry", None)
    topo.pop("forward_expansion_per_entry", None)
    topo.pop("backward_expansion_per_entry", None)
    topo.pop("bad_point_mask", None)
    topo.pop("row_indices", None)
    inj = checks.get("local_injectivity", {})
    inj.pop("span_fraction_per_entry_by_param", None)
    inj.pop("max_span_fraction_per_entry", None)
    inj.pop("bad_point_mask", None)
    inj.pop("row_indices", None)
    return out


def _apply_local_continuity_filter(
    dictionary: pd.DataFrame,
    cv_metrics: dict,
    cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """Optionally remove discontinuous dictionary entries from continuity checks."""
    eps = 1e-12
    report = {
        "enabled": bool(cfg.get("filter_enabled", False)),
        "applied": False,
        "reason": "disabled",
        "rows_before": int(len(dictionary)),
        "rows_after": int(len(dictionary)),
        "rows_flagged": 0,
        "rows_flagged_local": 0,
        "rows_flagged_topology": 0,
        "rows_flagged_injectivity": 0,
        "rows_flagged_before_cap": 0,
        "rows_removed": 0,
        "removed_fraction": 0.0,
        "capped_by_max_drop_fraction": False,
        "cv_threshold": float(cfg.get("local_continuity_cv_p95_max", 0.50)),
        "max_drop_fraction": float(cfg.get("filter_max_drop_fraction", 0.25)),
        "allow_large_drop": bool(cfg.get("filter_allow_large_drop", False)),
        "filter_include_topology": bool(cfg.get("filter_include_topology", True)),
        "filter_include_injectivity": bool(cfg.get("filter_include_injectivity", True)),
    }
    if dictionary.empty:
        report["reason"] = "empty_dictionary"
        return dictionary, report
    if not report["enabled"]:
        return dictionary, report

    checks = cv_metrics.get("checks", {})
    local_chk = checks.get("local_continuity", {})
    if local_chk.get("status") == "SKIPPED":
        report["reason"] = f"local_continuity_skipped:{local_chk.get('reason', 'unknown')}"
        return dictionary, report

    row_indices = np.asarray(local_chk.get("row_indices", []), dtype=int)
    worst_cv = np.asarray(local_chk.get("worst_cv_per_entry", []), dtype=float)
    if len(row_indices) == 0 or len(worst_cv) == 0 or len(row_indices) != len(worst_cv):
        report["reason"] = "missing_or_misaligned_local_continuity_arrays"
        return dictionary, report

    threshold = float(local_chk.get("cv_threshold", report["cv_threshold"]))
    report["cv_threshold"] = threshold
    flagged_mask = np.isfinite(worst_cv) & (worst_cv > threshold)
    flagged_idx = row_indices[flagged_mask]
    flagged_idx = flagged_idx[(flagged_idx >= 0) & (flagged_idx < len(dictionary))]
    flagged_local = np.unique(flagged_idx)
    report["rows_flagged_local"] = int(len(flagged_local))
    local_severity_by_row = {}
    if threshold > 0:
        local_severity_vals = np.zeros_like(worst_cv, dtype=float)
        finite_cv = np.isfinite(worst_cv)
        local_severity_vals[finite_cv] = np.maximum(
            0.0, (worst_cv[finite_cv] - threshold) / max(threshold, eps)
        )
    else:
        local_severity_vals = np.where(np.isfinite(worst_cv), worst_cv, 0.0)
    for ridx, sev in zip(row_indices, local_severity_vals):
        i = int(ridx)
        if 0 <= i < len(dictionary):
            local_severity_by_row[i] = float(max(sev, local_severity_by_row.get(i, 0.0)))

    flagged_topology = np.asarray([], dtype=int)
    topology_severity_by_row: dict[int, float] = {}
    if report["filter_include_topology"]:
        topo_chk = checks.get("topology_bidirectional_continuity", {})
        topo_rows = np.asarray(topo_chk.get("row_indices", []), dtype=int)
        topo_bad = np.asarray(topo_chk.get("bad_point_mask", []), dtype=bool)
        topo_overlap = np.asarray(topo_chk.get("overlap_per_entry", []), dtype=float)
        topo_forward = np.asarray(topo_chk.get("forward_expansion_per_entry", []), dtype=float)
        topo_backward = np.asarray(topo_chk.get("backward_expansion_per_entry", []), dtype=float)
        topo_thr = topo_chk.get("thresholds", {}) if isinstance(topo_chk.get("thresholds", {}), dict) else {}
        overlap_min = float(topo_thr.get("overlap_p10_min", cfg.get("topology_overlap_p10_min", 0.20)))
        forward_max = float(
            topo_thr.get("forward_expansion_p95_max", cfg.get("topology_forward_expansion_p95_max", 8.0))
        )
        backward_max = float(
            topo_thr.get("backward_expansion_p95_max", cfg.get("topology_backward_expansion_p95_max", 8.0))
        )
        if (
            topo_chk.get("status") != "SKIPPED"
            and len(topo_rows) > 0
            and len(topo_rows) == len(topo_bad)
        ):
            topo_idx = topo_rows[topo_bad]
            topo_idx = topo_idx[(topo_idx >= 0) & (topo_idx < len(dictionary))]
            flagged_topology = np.unique(topo_idx)
            # Build a per-row topology severity score for capped filtering.
            if (
                len(topo_overlap) == len(topo_rows)
                and len(topo_forward) == len(topo_rows)
                and len(topo_backward) == len(topo_rows)
            ):
                for j, ridx in enumerate(topo_rows):
                    i = int(ridx)
                    if i < 0 or i >= len(dictionary):
                        continue
                    sev = 0.0
                    ov = float(topo_overlap[j]) if np.isfinite(topo_overlap[j]) else np.nan
                    fw = float(topo_forward[j]) if np.isfinite(topo_forward[j]) else np.nan
                    bw = float(topo_backward[j]) if np.isfinite(topo_backward[j]) else np.nan
                    if np.isfinite(ov):
                        sev += max(0.0, (overlap_min - ov) / max(overlap_min, eps))
                    if np.isfinite(fw):
                        sev += max(0.0, (fw - forward_max) / max(forward_max, eps))
                    if np.isfinite(bw):
                        sev += max(0.0, (bw - backward_max) / max(backward_max, eps))
                    if bool(topo_bad[j]):
                        sev += 1.0
                    topology_severity_by_row[i] = float(max(sev, topology_severity_by_row.get(i, 0.0)))
        report["rows_flagged_topology"] = int(len(flagged_topology))

    flagged_injectivity = np.asarray([], dtype=int)
    injectivity_severity_by_row: dict[int, float] = {}
    if report["filter_include_injectivity"]:
        inj_chk = checks.get("local_injectivity", {})
        inj_rows = np.asarray(inj_chk.get("row_indices", []), dtype=int)
        inj_bad = np.asarray(inj_chk.get("bad_point_mask", []), dtype=bool)
        inj_max_span = np.asarray(inj_chk.get("max_span_fraction_per_entry", []), dtype=float)
        thr = inj_chk.get("thresholds", {}) if isinstance(inj_chk.get("thresholds", {}), dict) else {}
        point_thr_map = (
            thr.get("point_span_fraction_max_by_param", {})
            if isinstance(thr.get("point_span_fraction_max_by_param", {}), dict)
            else {}
        )
        point_thr_ref = float(max(point_thr_map.values())) if point_thr_map else float(
            cfg.get("injectivity_point_span_fraction_max", 0.45)
        )
        if (
            inj_chk.get("status") != "SKIPPED"
            and len(inj_rows) > 0
            and len(inj_rows) == len(inj_bad)
        ):
            inj_idx = inj_rows[inj_bad]
            inj_idx = inj_idx[(inj_idx >= 0) & (inj_idx < len(dictionary))]
            flagged_injectivity = np.unique(inj_idx)
            if len(inj_max_span) == len(inj_rows):
                for j, ridx in enumerate(inj_rows):
                    i = int(ridx)
                    if i < 0 or i >= len(dictionary):
                        continue
                    sev = 0.0
                    ms = float(inj_max_span[j]) if np.isfinite(inj_max_span[j]) else np.nan
                    if np.isfinite(ms):
                        sev += max(0.0, (ms - point_thr_ref) / max(point_thr_ref, eps))
                    if bool(inj_bad[j]):
                        sev += 1.0
                    injectivity_severity_by_row[i] = float(max(sev, injectivity_severity_by_row.get(i, 0.0)))
        report["rows_flagged_injectivity"] = int(len(flagged_injectivity))

    flagged_unique = np.unique(np.concatenate([flagged_local, flagged_topology, flagged_injectivity]))
    n_flagged_initial = int(len(flagged_unique))
    report["rows_flagged_before_cap"] = n_flagged_initial
    report["rows_flagged"] = n_flagged_initial
    if n_flagged_initial == 0:
        report["reason"] = "no_rows_exceed_filter_conditions"
        return dictionary, report

    max_drop_fraction = float(report["max_drop_fraction"])
    allow_large_drop = bool(report["allow_large_drop"])
    drop_fraction_initial = n_flagged_initial / max(len(dictionary), 1)
    selected_to_remove = flagged_unique
    n_flagged = n_flagged_initial
    drop_fraction = drop_fraction_initial
    if (drop_fraction > max_drop_fraction) and (not allow_large_drop):
        max_drop_rows = int(np.floor(max_drop_fraction * len(dictionary)))
        if max_drop_rows < 1:
            report["reason"] = (
                f"rows_flagged_fraction={drop_fraction:.4f} exceeds max_drop_fraction={max_drop_fraction:.4f}"
            )
            report["removed_fraction"] = float(drop_fraction)
            return dictionary, report

        # Rank flagged rows by combined continuity severity and cap removals to budget.
        flagged_local_set = set(int(i) for i in flagged_local.tolist())
        flagged_topo_set = set(int(i) for i in flagged_topology.tolist())
        flagged_inj_set = set(int(i) for i in flagged_injectivity.tolist())
        ranking: list[tuple[float, int]] = []
        for idx in flagged_unique:
            i = int(idx)
            sev_local = float(local_severity_by_row.get(i, 0.0))
            sev_topo = float(topology_severity_by_row.get(i, 0.0))
            sev_inj = float(injectivity_severity_by_row.get(i, 0.0))
            sev = sev_local + sev_topo + sev_inj
            if (i in flagged_local_set) and (i in flagged_topo_set):
                sev += 0.25
            if (i in flagged_inj_set) and ((i in flagged_local_set) or (i in flagged_topo_set)):
                sev += 0.25
            ranking.append((sev, i))
        ranking.sort(key=lambda t: (t[0], t[1]), reverse=True)

        selected_to_remove = np.asarray(
            [idx for _, idx in ranking[:max_drop_rows]],
            dtype=int,
        )
        n_flagged = int(len(selected_to_remove))
        drop_fraction = n_flagged / max(len(dictionary), 1)
        report["rows_flagged"] = n_flagged
        report["capped_by_max_drop_fraction"] = True

    filtered = dictionary.drop(index=selected_to_remove).reset_index(drop=True)
    report["applied"] = True
    if report["capped_by_max_drop_fraction"]:
        report["reason"] = (
            "removed_rows_flagged_by_continuity_filters_capped_to_max_drop_fraction"
        )
    else:
        report["reason"] = "removed_rows_flagged_by_continuity_filters"
    report["rows_after"] = int(len(filtered))
    report["rows_removed"] = int(len(dictionary) - len(filtered))
    report["removed_fraction"] = float(report["rows_removed"] / max(len(dictionary), 1))
    return filtered, report


def _cv_check_param_coverage(
    dictionary: pd.DataFrame,
    dataset: pd.DataFrame,
    param_cols: list[str],
) -> dict:
    """Check 1: Parameter-space coverage via nearest-neighbour distances."""
    usable = [c for c in param_cols if c in dictionary.columns and c in dataset.columns]
    if len(usable) < 2:
        return {"status": "SKIPPED", "reason": "insufficient_param_cols"}

    dict_vals = dictionary[usable].apply(pd.to_numeric, errors="coerce").dropna()
    ds_vals = dataset[usable].apply(pd.to_numeric, errors="coerce").dropna()
    if len(dict_vals) < 3:
        return {"status": "SKIPPED", "reason": "too_few_dictionary_points"}
    if len(ds_vals) < 1:
        return {"status": "SKIPPED", "reason": "empty_dataset_parameter_space"}

    # Normalize to [0, 1] using dataset range
    ds_min = ds_vals.min()
    ds_max = ds_vals.max()
    spread = ds_max - ds_min
    spread = spread.replace(0.0, 1.0)  # avoid division by zero
    norm_pts = ((dict_vals - ds_min) / spread).to_numpy(dtype=float)

    nn_d = _nearest_neighbor_distances(norm_pts)
    diag = np.sqrt(len(usable))  # bbox diagonal in normalized space
    nn_pct = nn_d / diag * 100.0

    p50 = float(np.nanpercentile(nn_pct, 50))
    p95 = float(np.nanpercentile(nn_pct, 95))
    status = "WARN" if p95 > 25.0 else "PASS"

    return {
        "status": status,
        "nn_p50_pct": round(p50, 4),
        "nn_p95_pct": round(p95, 4),
        "nn_p95_threshold_pct": 25.0,
        "n_points": int(len(dictionary)),
        "n_points_valid": int(len(dict_vals)),
        "n_dimensions": int(len(usable)),
        "nn_distances_pct": nn_pct.tolist(),
        "nn_distances_normalized": nn_d.tolist(),
    }


def _cv_check_local_continuity(
    dictionary: pd.DataFrame,
    param_cols: list[str],
    feature_cols: list[str] | None = None,
    k: int = 10,
    cv_threshold: float = 0.50,
) -> dict:
    """Check 2: Local continuity — nearby features should map to nearby parameters."""
    requested_feature_cols = (
        [str(c).strip() for c in feature_cols if str(c).strip()]
        if feature_cols
        else [f"eff_empirical_{p}" for p in (1, 2, 3, 4)]
    )
    requested_feature_cols = list(dict.fromkeys(requested_feature_cols))
    feat_cols = [c for c in requested_feature_cols if c in dictionary.columns]
    missing_feats = [c for c in requested_feature_cols if c not in dictionary.columns]
    if len(feat_cols) < 1:
        return {
            "status": "SKIPPED",
            "reason": "missing_feature_columns",
            "requested_feature_columns": requested_feature_cols,
            "missing_columns": missing_feats or requested_feature_cols,
        }
    usable_params = [c for c in param_cols if c in dictionary.columns]

    if len(usable_params) < 1:
        return {"status": "SKIPPED", "reason": "insufficient_columns"}

    used_cols = feat_cols + usable_params
    work = dictionary[used_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3:
        return {"status": "SKIPPED", "reason": "too_few_valid_rows"}
    feat = work[feat_cols].to_numpy(dtype=float)
    params = work[usable_params].to_numpy(dtype=float)
    row_indices = [int(i) for i in work.index.to_list()]
    n = len(feat)
    if n < 3:
        return {"status": "SKIPPED", "reason": "too_few_points"}

    try:
        k_int = int(k)
    except (TypeError, ValueError):
        k_int = 10
    effective_k = max(1, min(k_int, n - 1))

    # For each point, find k nearest neighbors in feature space
    worst_cv = np.zeros(n, dtype=float)
    cv_by_param: dict[str, list[float]] = {c: [] for c in usable_params}

    for i in range(n):
        diffs = feat - feat[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, effective_k)[:effective_k]

        entry_worst = 0.0
        for j, pcol in enumerate(usable_params):
            nn_vals = params[nn_idx, j]
            mean_val = np.mean(nn_vals)
            std_val = np.std(nn_vals)
            cv = std_val / max(abs(mean_val), 1e-12)
            cv_by_param[pcol].append(float(cv))
            entry_worst = max(entry_worst, cv)
        worst_cv[i] = entry_worst

    # Per-parameter P95 CV
    cv_p95_by_param = {}
    for pcol in usable_params:
        cv_p95_by_param[pcol] = round(float(np.percentile(cv_by_param[pcol], 95)), 4)

    n_discontinuous = int(np.sum(worst_cv > cv_threshold))
    disc_frac = n_discontinuous / max(n, 1)

    # WARN if discontinuous fraction > 10% for any parameter
    param_disc_fracs = {}
    for pcol in usable_params:
        arr = np.array(cv_by_param[pcol])
        param_disc_fracs[pcol] = float(np.mean(arr > cv_threshold))

    worst_param = max(param_disc_fracs, key=param_disc_fracs.get) if param_disc_fracs else ""
    status = "WARN" if any(f > 0.10 for f in param_disc_fracs.values()) else "PASS"

    return {
        "status": status,
        "feature_columns": feat_cols,
        "n_feature_dimensions": int(len(feat_cols)),
        "missing_feature_columns": missing_feats,
        "cv_p95_by_param": cv_p95_by_param,
        "n_discontinuous": n_discontinuous,
        "discontinuous_fraction": round(disc_frac, 4),
        "worst_param": worst_param,
        "n_points": n,
        "k_used": int(effective_k),
        "cv_threshold": float(cv_threshold),
        "param_discontinuous_fraction_by_param": {
            k: round(float(v), 4) for k, v in param_disc_fracs.items()
        },
        "worst_cv_per_entry": worst_cv.tolist(),
        "row_indices": row_indices,
    }


def _cv_check_bidirectional_topology_continuity(
    dictionary: pd.DataFrame,
    param_cols: list[str],
    *,
    feature_cols: list[str] | None = None,
    k: int = 12,
    overlap_p10_min: float = 0.20,
    overlap_median_min: float = 0.30,
    forward_expansion_p95_max: float = 8.0,
    backward_expansion_p95_max: float = 8.0,
    bad_fraction_max: float = 0.20,
) -> dict:
    """Check 3: Bidirectional neighborhood continuity in parameter and feature spaces.

    For each dictionary row i:
      - Np(i): k nearest neighbors in parameter space
      - Nf(i): k nearest neighbors in feature space

    Metrics:
      - overlap_i = |Np(i) ∩ Nf(i)| / k
      - forward expansion_i = radius_f(Np(i)) / local_feature_radius_i
      - backward expansion_i = radius_p(Nf(i)) / local_param_radius_i

    A point is flagged if overlap is too low or either expansion is too high.
    """
    requested_feature_cols = (
        [str(c).strip() for c in feature_cols if str(c).strip()]
        if feature_cols
        else [f"eff_empirical_{p}" for p in (1, 2, 3, 4)]
    )
    requested_feature_cols = list(dict.fromkeys(requested_feature_cols))
    feat_cols = [c for c in requested_feature_cols if c in dictionary.columns]
    missing_feats = [c for c in requested_feature_cols if c not in dictionary.columns]
    if len(feat_cols) < 1:
        return {
            "status": "SKIPPED",
            "reason": "missing_feature_columns",
            "requested_feature_columns": requested_feature_cols,
            "missing_columns": missing_feats or requested_feature_cols,
        }

    usable_params = [c for c in param_cols if c in dictionary.columns]
    if len(usable_params) < 2:
        return {"status": "SKIPPED", "reason": "insufficient_param_columns"}

    used_cols = feat_cols + usable_params
    work = dictionary[used_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 6:
        return {"status": "SKIPPED", "reason": "too_few_valid_rows"}

    row_indices = np.asarray(work.index.to_list(), dtype=int)
    p = work[usable_params].to_numpy(dtype=float)
    f = work[feat_cols].to_numpy(dtype=float)
    n = len(work)

    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        lo = np.nanmin(arr, axis=0)
        hi = np.nanmax(arr, axis=0)
        span = hi - lo
        span = np.where(np.isfinite(span) & (span > 0.0), span, 1.0)
        return (arr - lo) / span

    p_norm = _minmax_norm(p)
    f_norm = _minmax_norm(f)

    try:
        k_int = int(k)
    except (TypeError, ValueError):
        k_int = 12
    k_use = max(2, min(k_int, n - 1))

    # Pairwise distances in normalized spaces.
    p_diff = p_norm[:, None, :] - p_norm[None, :, :]
    f_diff = f_norm[:, None, :] - f_norm[None, :, :]
    p_dist = np.sqrt(np.sum(p_diff * p_diff, axis=2))
    f_dist = np.sqrt(np.sum(f_diff * f_diff, axis=2))
    np.fill_diagonal(p_dist, np.inf)
    np.fill_diagonal(f_dist, np.inf)

    # k-NN indices in each space.
    idx_p = np.argpartition(p_dist, kth=k_use - 1, axis=1)[:, :k_use]
    idx_f = np.argpartition(f_dist, kth=k_use - 1, axis=1)[:, :k_use]

    local_param_radius = np.max(np.take_along_axis(p_dist, idx_p, axis=1), axis=1)
    local_feat_radius = np.max(np.take_along_axis(f_dist, idx_f, axis=1), axis=1)

    overlap = np.zeros(n, dtype=float)
    forward_exp = np.zeros(n, dtype=float)
    backward_exp = np.zeros(n, dtype=float)
    eps = 1e-12
    for i in range(n):
        np_set = set(idx_p[i].tolist())
        nf_set = set(idx_f[i].tolist())
        overlap[i] = float(len(np_set.intersection(nf_set)) / k_use)

        d_feat_from_param_neighbors = f_dist[i, idx_p[i]]
        d_param_from_feat_neighbors = p_dist[i, idx_f[i]]
        forward_exp[i] = float(np.max(d_feat_from_param_neighbors) / max(local_feat_radius[i], eps))
        backward_exp[i] = float(np.max(d_param_from_feat_neighbors) / max(local_param_radius[i], eps))

    overlap_p10 = float(np.percentile(overlap, 10))
    overlap_median = float(np.percentile(overlap, 50))
    forward_exp_p95 = float(np.percentile(forward_exp, 95))
    backward_exp_p95 = float(np.percentile(backward_exp, 95))

    severe_forward_thr = 1.5 * float(forward_expansion_p95_max)
    severe_backward_thr = 1.5 * float(backward_expansion_p95_max)
    bad_mask = (
        (overlap < float(overlap_p10_min))
        | (forward_exp > severe_forward_thr)
        | (backward_exp > severe_backward_thr)
    )
    bad_fraction = float(np.mean(bad_mask))

    warn = (
        (overlap_p10 < float(overlap_p10_min))
        or (overlap_median < float(overlap_median_min))
        or (forward_exp_p95 > float(forward_expansion_p95_max))
        or (backward_exp_p95 > float(backward_expansion_p95_max))
        or (bad_fraction > float(bad_fraction_max))
    )
    status = "WARN" if warn else "PASS"

    return {
        "status": status,
        "feature_columns": feat_cols,
        "n_feature_dimensions": int(len(feat_cols)),
        "missing_feature_columns": missing_feats,
        "n_points": int(n),
        "k_used": int(k_use),
        "overlap_p10": round(overlap_p10, 4),
        "overlap_median": round(overlap_median, 4),
        "forward_expansion_p95": round(forward_exp_p95, 4),
        "backward_expansion_p95": round(backward_exp_p95, 4),
        "bad_fraction": round(bad_fraction, 4),
        "thresholds": {
            "overlap_p10_min": float(overlap_p10_min),
            "overlap_median_min": float(overlap_median_min),
            "forward_expansion_p95_max": float(forward_expansion_p95_max),
            "backward_expansion_p95_max": float(backward_expansion_p95_max),
            "bad_fraction_max": float(bad_fraction_max),
        },
        "overlap_per_entry": overlap.tolist(),
        "forward_expansion_per_entry": forward_exp.tolist(),
        "backward_expansion_per_entry": backward_exp.tolist(),
        "bad_point_mask": bad_mask.astype(bool).tolist(),
        "row_indices": row_indices.tolist(),
    }


def _cv_check_local_injectivity(
    dictionary: pd.DataFrame,
    param_cols: list[str],
    *,
    feature_cols: list[str] | None = None,
    k: int = 10,
    span_fraction_p95_max: float = 0.35,
    flux_span_fraction_p95_max: float | None = None,
    point_span_fraction_max: float = 0.45,
    bad_fraction_max: float = 0.10,
    span_fraction_p95_max_by_param: dict | None = None,
    point_span_fraction_max_by_param: dict | None = None,
) -> dict:
    """Check local injectivity: close features must map to tight parameter neighborhoods."""
    requested_feature_cols = (
        [str(c).strip() for c in feature_cols if str(c).strip()]
        if feature_cols
        else [f"eff_empirical_{p}" for p in (1, 2, 3, 4)]
    )
    requested_feature_cols = list(dict.fromkeys(requested_feature_cols))
    feat_cols = [c for c in requested_feature_cols if c in dictionary.columns]
    missing_feats = [c for c in requested_feature_cols if c not in dictionary.columns]
    if len(feat_cols) < 1:
        return {
            "status": "SKIPPED",
            "reason": "missing_feature_columns",
            "requested_feature_columns": requested_feature_cols,
            "missing_columns": missing_feats or requested_feature_cols,
        }

    usable_params = [c for c in param_cols if c in dictionary.columns]
    if len(usable_params) < 1:
        return {"status": "SKIPPED", "reason": "insufficient_param_columns"}

    used_cols = feat_cols + usable_params
    work = dictionary[used_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3:
        return {"status": "SKIPPED", "reason": "too_few_valid_rows"}

    row_indices = np.asarray(work.index.to_list(), dtype=int)
    feat = work[feat_cols].to_numpy(dtype=float)
    params = work[usable_params].to_numpy(dtype=float)
    n = len(work)
    try:
        k_int = int(k)
    except (TypeError, ValueError):
        k_int = 10
    effective_k = max(1, min(k_int, n - 1))

    p_min = np.nanmin(params, axis=0)
    p_max = np.nanmax(params, axis=0)
    p_span = p_max - p_min
    p_span = np.where(np.isfinite(p_span) & (p_span > 0.0), p_span, 1.0)

    local_span_frac = np.zeros((n, len(usable_params)), dtype=float)
    for i in range(n):
        diffs = feat - feat[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, effective_k)[:effective_k]
        nn_vals = params[nn_idx]
        nn_span = np.nanmax(nn_vals, axis=0) - np.nanmin(nn_vals, axis=0)
        local_span_frac[i, :] = nn_span / p_span

    span_p95_by_param = {
        pcol: round(float(np.percentile(local_span_frac[:, j], 95)), 4)
        for j, pcol in enumerate(usable_params)
    }
    span_p50_by_param = {
        pcol: round(float(np.percentile(local_span_frac[:, j], 50)), 4)
        for j, pcol in enumerate(usable_params)
    }

    p95_default = float(span_fraction_p95_max)
    point_default = float(point_span_fraction_max)
    p95_by_param_cfg = (
        span_fraction_p95_max_by_param
        if isinstance(span_fraction_p95_max_by_param, dict)
        else {}
    )
    point_by_param_cfg = (
        point_span_fraction_max_by_param
        if isinstance(point_span_fraction_max_by_param, dict)
        else {}
    )

    p95_threshold_by_param: dict[str, float] = {}
    point_threshold_by_param: dict[str, float] = {}
    for pcol in usable_params:
        p95_thr = float(p95_by_param_cfg.get(pcol, p95_default))
        point_thr = float(point_by_param_cfg.get(pcol, point_default))
        if (pcol == "flux_cm2_min") and (flux_span_fraction_p95_max is not None):
            p95_thr = float(flux_span_fraction_p95_max)
        p95_threshold_by_param[pcol] = p95_thr
        point_threshold_by_param[pcol] = point_thr

    bad_mask = np.zeros(n, dtype=bool)
    for j, pcol in enumerate(usable_params):
        bad_mask |= local_span_frac[:, j] > point_threshold_by_param[pcol]
    bad_fraction = float(np.mean(bad_mask))

    failed_params = [
        pcol
        for pcol in usable_params
        if float(span_p95_by_param.get(pcol, np.nan)) > float(p95_threshold_by_param.get(pcol, p95_default))
    ]
    status = "FAIL" if (len(failed_params) > 0 or bad_fraction > float(bad_fraction_max)) else "PASS"

    flux_span_p95 = np.nan
    flux_idx = None
    if "flux_cm2_min" in usable_params:
        flux_idx = usable_params.index("flux_cm2_min")
        flux_span_p95 = float(np.percentile(local_span_frac[:, flux_idx], 95))

    max_span_per_entry = np.max(local_span_frac, axis=1)
    return {
        "status": status,
        "feature_columns": feat_cols,
        "n_feature_dimensions": int(len(feat_cols)),
        "missing_feature_columns": missing_feats,
        "n_points": int(n),
        "k_used": int(effective_k),
        "span_fraction_p50_by_param": span_p50_by_param,
        "span_fraction_p95_by_param": span_p95_by_param,
        "flux_span_fraction_p95": (
            round(float(flux_span_p95), 4) if np.isfinite(flux_span_p95) else np.nan
        ),
        "bad_fraction": round(float(bad_fraction), 4),
        "n_bad_points": int(np.sum(bad_mask)),
        "failed_params": failed_params,
        "thresholds": {
            "span_fraction_p95_max_by_param": p95_threshold_by_param,
            "point_span_fraction_max_by_param": point_threshold_by_param,
            "bad_fraction_max": float(bad_fraction_max),
        },
        "span_fraction_per_entry_by_param": {
            pcol: local_span_frac[:, j].tolist()
            for j, pcol in enumerate(usable_params)
        },
        "max_span_fraction_per_entry": max_span_per_entry.tolist(),
        "bad_point_mask": bad_mask.astype(bool).tolist(),
        "row_indices": row_indices.tolist(),
    }


def _cv_check_isotonic_bounds(
    isotonic_by_plane: dict[int, dict],
) -> dict:
    """Check 3: Isotonic calibration slopes and knot values within bounds."""
    if not isotonic_by_plane:
        return {"status": "SKIPPED", "reason": "no_isotonic_data"}

    planes_checked = []
    slope_ranges = {}
    y_knot_ranges = {}
    violations: list[str] = []

    for plane in (1, 2, 3, 4):
        iso = isotonic_by_plane.get(plane)
        if iso is None:
            continue
        planes_checked.append(plane)

        sl = iso.get("slope_lo", 0.0)
        sh = iso.get("slope_hi", 0.0)
        slope_ranges[f"plane_{plane}"] = {"slope_lo": round(float(sl), 4), "slope_hi": round(float(sh), 4)}

        y_knots = np.asarray(iso.get("y_knots", []), dtype=float)
        if len(y_knots) > 0:
            y_knot_ranges[f"plane_{plane}"] = {
                "min": round(float(np.min(y_knots)), 4),
                "max": round(float(np.max(y_knots)), 4),
            }

        if sl < 0:
            violations.append(f"plane {plane}: slope_lo={sl:.4f} < 0")
        if sh < 0:
            violations.append(f"plane {plane}: slope_hi={sh:.4f} < 0")
        if len(y_knots) > 0 and (np.min(y_knots) < 0.0 or np.max(y_knots) > 1.05):
            violations.append(
                f"plane {plane}: y_knots range [{np.min(y_knots):.4f}, {np.max(y_knots):.4f}] outside [0, 1.05]"
            )

    if not planes_checked:
        return {"status": "SKIPPED", "reason": "no_planes_checked"}

    status = "FAIL" if violations else "PASS"
    return {
        "status": status,
        "planes_checked": planes_checked,
        "slope_ranges": slope_ranges,
        "y_knot_ranges": y_knot_ranges,
        "violations": violations,
    }


def _cv_check_support_adequacy(
    dictionary: pd.DataFrame,
    dataset: pd.DataFrame,
    oos_max: float = 0.30,
) -> dict:
    """Check 4: Per-plane efficiency support coverage."""
    oos_by_plane: dict[str, float] = {}
    dict_range: dict[str, dict] = {}
    ds_range: dict[str, dict] = {}

    for p in (1, 2, 3, 4):
        col = f"eff_sim_{p}"
        if col not in dictionary.columns or col not in dataset.columns:
            continue
        d_vals = pd.to_numeric(dictionary[col], errors="coerce").dropna()
        s_vals = pd.to_numeric(dataset[col], errors="coerce").dropna()
        if len(d_vals) == 0 or len(s_vals) == 0:
            continue

        d_min, d_max = float(d_vals.min()), float(d_vals.max())
        s_min, s_max = float(s_vals.min()), float(s_vals.max())
        dict_range[col] = {"min": round(d_min, 4), "max": round(d_max, 4)}
        ds_range[col] = {"min": round(s_min, 4), "max": round(s_max, 4)}

        oos = float(((s_vals < d_min) | (s_vals > d_max)).mean())
        oos_by_plane[col] = round(oos, 4)

    status = "WARN" if any(f > oos_max for f in oos_by_plane.values()) else "PASS"
    return {
        "status": status,
        "out_of_support_max_fraction_threshold": float(oos_max),
        "out_of_support_fraction_by_plane": oos_by_plane,
        "dict_range_by_plane": dict_range,
        "dataset_range_by_plane": ds_range,
    }


def _cv_check_density_uniformity(
    dictionary: pd.DataFrame,
    n_bins: int = 5,
    ratio_max: float = 20.0,
) -> dict:
    """Check 5: Density uniformity in flux × eff_sim_2 space."""
    x_col, y_col = "flux_cm2_min", "eff_sim_2"
    if x_col not in dictionary.columns or y_col not in dictionary.columns:
        return {"status": "SKIPPED", "reason": f"missing {x_col} or {y_col}"}

    xy = dictionary[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    x = xy[x_col].to_numpy(dtype=float)
    y = xy[y_col].to_numpy(dtype=float)
    if len(x) < 4:
        return {"status": "SKIPPED", "reason": "too_few_points"}

    try:
        n_bins_int = max(1, int(n_bins))
    except (TypeError, ValueError):
        n_bins_int = 5

    counts, _, _ = np.histogram2d(x, y, bins=n_bins_int)
    flat = counts.ravel()
    total_bins = len(flat)
    nonzero = flat[flat > 0]
    empty_frac = float((flat == 0).sum()) / total_bins

    if len(nonzero) == 0:
        return {"status": "SKIPPED", "reason": "all_bins_empty"}

    ratio = float(np.max(nonzero) / np.median(nonzero))
    status = "WARN" if ratio > ratio_max else "PASS"

    return {
        "status": status,
        "density_ratio": round(ratio, 2),
        "density_ratio_max_threshold": float(ratio_max),
        "empty_bin_fraction": round(empty_frac, 4),
        "grid_shape": [n_bins_int, n_bins_int],
    }


def _plot_continuity_validation(
    dictionary: pd.DataFrame,
    dataset: pd.DataFrame,
    param_cols: list[str],
    cv_metrics: dict,
    feature_cols: list[str] | None,
    path,
) -> None:
    """Generate actionable 2×3 continuity validation diagnostic plot."""
    output_path = Path(path)
    checks = cv_metrics.get("checks", {})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    status_colors = {
        "PASS": "#2A9D8F",
        "WARN": "#E9C46A",
        "FAIL": "#E45756",
        "SKIPPED": "#9AA0A6",
    }
    status_faces = {
        "PASS": "#E8F6F2",
        "WARN": "#FFF4DC",
        "FAIL": "#FDE8E8",
        "SKIPPED": "#ECEFF1",
    }

    def _status_badge(ax_obj, status: str, extra: str = "") -> None:
        st = str(status).upper()
        color = status_colors.get(st, "#4A4A4A")
        face = status_faces.get(st, "#F1F3F4")
        text = st if not extra else f"{st} | {extra}"
        ax_obj.text(
            0.99,
            0.98,
            text,
            transform=ax_obj.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=color,
            bbox={"boxstyle": "round,pad=0.24", "facecolor": face, "edgecolor": color, "linewidth": 0.8},
        )

    # ── (0,0) Parameter-Space Coverage ────────────────────────────
    ax = axes[0, 0]
    c1 = checks.get("param_space_coverage", {})
    c1_status = str(c1.get("status", "SKIPPED")).upper()
    nn_thr = float(c1.get("nn_p95_threshold_pct", 25.0))
    nn_dists = np.asarray(c1.get("nn_distances_pct", c1.get("nn_distances_normalized", [])), dtype=float)
    nn_dists = nn_dists[np.isfinite(nn_dists)]
    if nn_dists.size > 0:
        n_bins = int(np.clip(np.sqrt(nn_dists.size) * 1.6, 18, 70))
        ax.hist(nn_dists, bins=n_bins, color="#4C78A8", alpha=0.82, edgecolor="white", linewidth=0.35)
        p95 = float(c1.get("nn_p95_pct", np.nan))
        if np.isfinite(p95):
            ax.axvline(p95, color="#1F3A5F", linewidth=1.4, linestyle="--", label=f"P95={p95:.2f}%")
        ax.axvline(nn_thr, color="#E45756", linewidth=1.4, linestyle="-.", label=f"Threshold={nn_thr:.2f}%")
        x_max = float(max(nn_thr * 1.15, np.max(nn_dists) * 1.1, 1e-6))
        ax.axvspan(nn_thr, x_max, color="#FDE2E1", alpha=0.35)
        ax.set_xlim(0.0, x_max)
        ax.set_xlabel("Nearest-neighbor distance [% bbox diagonal]")
        ax.set_ylabel("Dictionary entries")
        ax.legend(fontsize=7, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No coverage distances available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(
        "Coverage: NN Distance Distribution\n"
        f"P95={float(c1.get('nn_p95_pct', np.nan)):.2f}% (threshold {nn_thr:.2f}%)"
    )
    _status_badge(ax, c1_status, "lower is better")

    # ── (0,1) Local Continuity ────────────────────────────────────
    ax = axes[0, 1]
    c2 = checks.get("local_continuity", {})
    c2_status = str(c2.get("status", "SKIPPED")).upper()
    cv_thr = float(c2.get("cv_threshold", 0.50))
    worst_cv = np.asarray(c2.get("worst_cv_per_entry", []), dtype=float)
    worst_cv = worst_cv[np.isfinite(worst_cv)]
    if worst_cv.size > 0:
        n_bins = int(np.clip(np.sqrt(worst_cv.size) * 1.5, 15, 60))
        ax.hist(worst_cv, bins=n_bins, color="#2A9D8F", alpha=0.82, edgecolor="white", linewidth=0.35)
        ax.axvline(cv_thr, color="#E45756", linewidth=1.4, linestyle="-.", label=f"Threshold={cv_thr:.3f}")
        x_hi = float(max(cv_thr * 1.35, np.percentile(worst_cv, 99) * 1.15, 1e-4))
        ax.axvspan(cv_thr, x_hi, color="#FDE2E1", alpha=0.35)
        ax.set_xlim(left=0.0, right=x_hi)
        ax.set_xlabel("Worst-parameter local CV per entry")
        ax.set_ylabel("Dictionary entries")
        disc_frac = float(c2.get("discontinuous_fraction", np.nan))
        ax2 = ax.twinx()
        cv_sorted = np.sort(worst_cv)
        cdf_pct = np.linspace(100.0 / len(cv_sorted), 100.0, len(cv_sorted))
        ax2.plot(cv_sorted, cdf_pct, color="#1F3A5F", linewidth=1.0, alpha=0.9)
        ax2.set_ylabel("CDF [%]", color="#1F3A5F")
        ax2.tick_params(axis="y", labelsize=7, colors="#1F3A5F")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title(
            "Local Continuity (Feature -> Parameter)\n"
            f"discontinuous_fraction={disc_frac:.3f} (target <= 0.10)"
        )
    else:
        ax.text(0.5, 0.5, "No local-CV vectors available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Local Continuity (Feature -> Parameter)")
    _status_badge(ax, c2_status, f"CV threshold={cv_thr:.3f}")

    # ── (0,2) Topology Continuity ─────────────────────────────────
    ax = axes[0, 2]
    c3 = checks.get("topology_bidirectional_continuity", {})
    c3_status = str(c3.get("status", "SKIPPED")).upper()
    c_inj = checks.get("local_injectivity", {})
    c_inj_status = str(c_inj.get("status", "SKIPPED")).upper()
    topo_panel_status = (
        "FAIL"
        if "FAIL" in {c3_status, c_inj_status}
        else ("WARN" if "WARN" in {c3_status, c_inj_status} else ("PASS" if "PASS" in {c3_status, c_inj_status} else "SKIPPED"))
    )
    overlap = np.asarray(c3.get("overlap_per_entry", []), dtype=float)
    fwd_exp = np.asarray(c3.get("forward_expansion_per_entry", []), dtype=float)
    bwd_exp = np.asarray(c3.get("backward_expansion_per_entry", []), dtype=float)
    bad_mask = np.asarray(c3.get("bad_point_mask", []), dtype=bool)
    topo_thr = c3.get("thresholds", {}) if isinstance(c3.get("thresholds", {}), dict) else {}
    overlap_min = float(topo_thr.get("overlap_p10_min", np.nan))
    fwd_max = float(topo_thr.get("forward_expansion_p95_max", np.nan))
    bwd_max = float(topo_thr.get("backward_expansion_p95_max", np.nan))
    expansion_thr = float(max(fwd_max, bwd_max)) if np.isfinite(max(fwd_max, bwd_max)) else np.nan
    if (
        overlap.size > 0
        and overlap.size == fwd_exp.size
        and overlap.size == bwd_exp.size
    ):
        expansion = np.maximum(fwd_exp, bwd_exp)
        finite = np.isfinite(overlap) & np.isfinite(expansion)
        if bad_mask.size == overlap.size:
            bad = bad_mask[finite]
        else:
            bad = np.zeros(np.count_nonzero(finite), dtype=bool)
        ov = overlap[finite]
        ex = expansion[finite]
        if ov.size > 2500:
            hb = ax.hexbin(ov, ex, gridsize=42, mincnt=1, cmap="Blues")
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label="Point count")
            if np.any(bad):
                idx_bad = np.where(bad)[0]
                if len(idx_bad) > 350:
                    idx_bad = idx_bad[:350]
                ax.scatter(ov[idx_bad], ex[idx_bad], s=10, facecolors="none", edgecolors="#E45756", linewidths=0.6)
        else:
            ax.scatter(ov[~bad], ex[~bad], s=12, color="#4C78A8", alpha=0.45, edgecolors="none", label="OK")
            if np.any(bad):
                ax.scatter(
                    ov[bad],
                    ex[bad],
                    s=16,
                    facecolors="none",
                    edgecolors="#E45756",
                    linewidths=0.8,
                    label="Flagged severe",
                )
            ax.legend(fontsize=7, loc="upper right")
        if np.isfinite(overlap_min):
            ax.axvline(overlap_min, color="#E45756", linestyle="-.", linewidth=1.3, label="Overlap threshold")
            ax.axvspan(0.0, overlap_min, color="#FDE2E1", alpha=0.28)
        if np.isfinite(expansion_thr):
            ax.axhline(expansion_thr, color="#E45756", linestyle="--", linewidth=1.3, label="Expansion threshold")
        ax.set_xlim(-0.02, 1.02)
        y_hi = float(max(np.nanpercentile(ex, 99), expansion_thr if np.isfinite(expansion_thr) else 0.0))
        ax.set_ylim(0.0, max(1.05, y_hi * 1.15))
        ax.set_xlabel("k-NN overlap ratio")
        ax.set_ylabel("max(forward, backward) expansion")
        ax.set_title(
            "Topology Continuity\n"
            f"overlap_p10={float(c3.get('overlap_p10', np.nan)):.3f}, "
            f"median={float(c3.get('overlap_median', np.nan)):.3f}, "
            f"bad_fraction={float(c3.get('bad_fraction', np.nan)):.3f}"
        )
    else:
        ax.text(0.5, 0.5, "No topology vectors available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Topology Continuity")
    inj_thr = (
        (
            c_inj.get("thresholds", {}).get("span_fraction_p95_max_by_param", {}).get("flux_cm2_min", np.nan)
            if isinstance(c_inj.get("thresholds", {}), dict)
            else np.nan
        )
    )
    inj_bad_thr = (
        (
            c_inj.get("thresholds", {}).get("bad_fraction_max", np.nan)
            if isinstance(c_inj.get("thresholds", {}), dict)
            else np.nan
        )
    )
    inj_text = (
        f"injectivity={c_inj_status}\n"
        f"flux_span_p95={float(c_inj.get('flux_span_fraction_p95', np.nan)):.3f} (thr={float(inj_thr):.3f})\n"
        f"bad_fraction={float(c_inj.get('bad_fraction', np.nan)):.3f} (thr={float(inj_bad_thr):.3f})"
    )
    ax.text(
        0.015,
        0.985,
        inj_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B0B0B0", "alpha": 0.88},
    )
    _status_badge(
        ax,
        topo_panel_status,
        (
            f"overlap>={overlap_min:.2f}, exp<={expansion_thr:.2f}"
            if np.isfinite(overlap_min) and np.isfinite(expansion_thr)
            else ""
        ),
    )

    # ── (1,0) Support Adequacy ────────────────────────────────────
    ax = axes[1, 0]
    c4 = checks.get("support_adequacy", {})
    c4_status = str(c4.get("status", "SKIPPED")).upper()
    oos_by_plane = c4.get("out_of_support_fraction_by_plane", {})
    oos_thr = float(c4.get("out_of_support_max_fraction_threshold", 0.30))
    if isinstance(oos_by_plane, dict) and oos_by_plane:
        def _plane_key(name: str) -> tuple[int, str]:
            token = str(name).split("_")[-1]
            return (int(token), str(name)) if token.isdigit() else (99, str(name))

        plane_keys = sorted(oos_by_plane.keys(), key=_plane_key)
        labels = [f"P{str(k).split('_')[-1]}" for k in plane_keys]
        vals = np.asarray([float(oos_by_plane[k]) for k in plane_keys], dtype=float)
        colors = np.where(vals > oos_thr, "#E45756", "#2A9D8F")
        bars = ax.bar(labels, vals, color=colors, alpha=0.88, edgecolor="black", linewidth=0.35)
        ax.axhline(oos_thr, color="#1F3A5F", linestyle="-.", linewidth=1.3, label=f"Threshold={oos_thr:.2f}")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(0.01, 0.02 * max(np.max(vals), oos_thr)),
                f"{100.0 * val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0.0, max(0.05, np.max(vals) * 1.25, oos_thr * 1.35))
        ax.set_ylabel("Out-of-support fraction")
        ax.set_xlabel("Efficiency plane")
        ax.legend(fontsize=7, loc="upper right")
        worst_key = max(plane_keys, key=lambda key: float(oos_by_plane[key]))
        worst_val = float(oos_by_plane[worst_key])
        ax.set_title(
            "Support Adequacy\n"
            f"worst={worst_key} ({100.0 * worst_val:.1f}%), threshold={100.0 * oos_thr:.1f}%"
        )
    else:
        ax.text(0.5, 0.5, "No support-adequacy metrics available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Support Adequacy")
    _status_badge(ax, c4_status, f"max OOS={oos_thr:.2f}")

    # ── (1,1) Density + Isotonic ──────────────────────────────────
    ax = axes[1, 1]
    c5 = checks.get("density_uniformity", {})
    c5_status = str(c5.get("status", "SKIPPED")).upper()
    c_iso = checks.get("isotonic_bounds", {})
    iso_status = str(c_iso.get("status", "SKIPPED")).upper()
    panel_status = "FAIL" if ("FAIL" in {c5_status, iso_status}) else ("WARN" if "WARN" in {c5_status, iso_status} else ("PASS" if "PASS" in {c5_status, iso_status} else "SKIPPED"))
    x_col, y_col = "flux_cm2_min", "eff_sim_2"
    if x_col in dictionary.columns and y_col in dictionary.columns:
        x_vals = pd.to_numeric(dictionary[x_col], errors="coerce").to_numpy(dtype=float)
        y_vals = pd.to_numeric(dictionary[y_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[finite]
        y_vals = y_vals[finite]
        if x_vals.size >= 4:
            grid_shape = c5.get("grid_shape", [5, 5])
            try:
                n_bins = int(grid_shape[0]) if isinstance(grid_shape, (list, tuple)) else int(grid_shape)
            except (TypeError, ValueError, IndexError):
                n_bins = 5
            n_bins = int(np.clip(n_bins, 3, 40))
            counts, x_edges, y_edges = np.histogram2d(x_vals, y_vals, bins=n_bins)
            mesh = ax.pcolormesh(x_edges, y_edges, counts.T, cmap="YlGnBu", shading="auto")
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02, label="Bin count")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        else:
            ax.text(0.5, 0.5, "Too few finite points for density map", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Missing flux/eff columns for density map", ha="center", va="center", transform=ax.transAxes)
    dens_ratio = float(c5.get("density_ratio", np.nan))
    dens_thr = float(c5.get("density_ratio_max_threshold", np.nan))
    empty_frac = float(c5.get("empty_bin_fraction", np.nan))
    iso_violations = len(c_iso.get("violations", [])) if isinstance(c_iso.get("violations", []), list) else 0
    detail = (
        f"density_ratio={dens_ratio:.2f} (thr={dens_thr:.2f})\n"
        f"empty_bins={empty_frac:.3f}\n"
        f"isotonic={iso_status} (violations={iso_violations})"
    )
    ax.text(
        0.01,
        0.99,
        detail,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B0B0B0", "alpha": 0.88},
    )
    ax.set_title("Density + Isotonic Diagnostics")
    _status_badge(ax, panel_status, "density/isotonic")

    # ── (1,2) Summary ─────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    ordered_checks = [
        ("param_space_coverage", "Coverage"),
        ("local_continuity", "Local continuity"),
        ("topology_bidirectional_continuity", "Topology continuity"),
        ("local_injectivity", "Local injectivity"),
        ("isotonic_bounds", "Isotonic bounds"),
        ("support_adequacy", "Support adequacy"),
        ("density_uniformity", "Density uniformity"),
    ]
    features = cv_metrics.get("feature_columns_used", [])
    if not features:
        features = [c for c in (feature_cols or []) if c in dictionary.columns]
    feature_preview = ", ".join(features[:6]) if features else "none"
    if len(features) > 6:
        feature_preview += ", ..."

    summary_lines: list[tuple[str, str]] = []
    summary_lines.append((f"overall: {cv_metrics.get('status', 'N/A')}", str(cv_metrics.get("status", "N/A")).upper()))
    summary_lines.append((f"dictionary_rows: {len(dictionary)}", ""))
    summary_lines.append((f"dataset_rows: {len(dataset)}", ""))
    summary_lines.append((f"feature_cols({len(features)}): {feature_preview}", ""))
    summary_lines.append(("", ""))
    for key, label in ordered_checks:
        st = str((checks.get(key, {}) or {}).get("status", "SKIPPED")).upper()
        summary_lines.append((f"{label}: {st}", st))

    msgs = cv_metrics.get("messages", [])
    if isinstance(msgs, list) and msgs:
        summary_lines.append(("", ""))
        summary_lines.append(("messages:", ""))
        for msg in msgs[:5]:
            summary_lines.append((f"- {msg}", ""))

    y0 = 0.97
    step = 0.067
    for i, (line, tag) in enumerate(summary_lines):
        color = status_colors.get(tag, "#2E2E2E")
        ax.text(
            0.03,
            y0 - i * step,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            family="monospace",
            color=color,
        )
    ax.set_title("Validation Summary")

    fig.suptitle(
        "Dictionary Continuity Validation (STEP 1.2) "
        f"| overall={cv_metrics.get('status', 'N/A')}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.01, 1.0, 0.96))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_continuity_neighborhood_example(
    dictionary: pd.DataFrame,
    path,
    *,
    param_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    cv_metrics: dict | None = None,
    n_points: int = 12,
) -> None:
    """Plot topology-aware neighborhood continuity using deterministic local selections.

    Compatibility outputs:
      - ..._param_to_feature_matrix.png
      - ..._feature_to_param_matrix.png
    New combined output:
      - ..._bidirectional.png
    """
    out_base = Path(path)
    param_cols = [
        c for c in (param_cols or ("flux_cm2_min", "cos_n", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"))
        if c in dictionary.columns
    ]
    preferred_feature_cols: list[str] = []
    if isinstance(cv_metrics, dict):
        checks = cv_metrics.get("checks", {}) if isinstance(cv_metrics.get("checks", {}), dict) else {}
        for key in ("topology_bidirectional_continuity", "local_continuity"):
            chk = checks.get(key, {}) if isinstance(checks.get(key, {}), dict) else {}
            cols = chk.get("feature_columns", [])
            if isinstance(cols, list) and cols:
                preferred_feature_cols = [str(c).strip() for c in cols if str(c).strip()]
                break
    requested_feature_cols = (
        preferred_feature_cols
        or [str(c).strip() for c in (feature_cols or []) if str(c).strip()]
        or [f"eff_empirical_{i}" for i in (1, 2, 3, 4)]
    )
    feat_cols = [
        c for c in (
            requested_feature_cols
        )
        if c in dictionary.columns
    ]
    max_plot_feature_dims = 8
    if len(feat_cols) > max_plot_feature_dims:
        preferred = [
            "__derived_tt_global_rate_hz",
            "events_per_second_global_rate",
            "eff_empirical_1",
            "eff_empirical_2",
            "eff_empirical_3",
            "eff_empirical_4",
        ]
        kept: list[str] = []
        for c in preferred:
            if c in feat_cols and c not in kept:
                kept.append(c)
            if len(kept) >= max_plot_feature_dims:
                break
        remaining = [c for c in feat_cols if c not in kept]
        ranked_remaining: list[tuple[float, str]] = []
        for c in remaining:
            vals = pd.to_numeric(dictionary[c], errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            var = float(np.nanvar(finite)) if finite.size >= 3 else -1.0
            ranked_remaining.append((var, str(c)))
        ranked_remaining.sort(key=lambda t: (-t[0], t[1]))
        for _, c in ranked_remaining:
            if len(kept) >= max_plot_feature_dims:
                break
            kept.append(c)
        log.info(
            "Continuity neighborhood plot: limiting feature dimensions from %d to %d for readability.",
            len(feat_cols),
            len(kept),
        )
        feat_cols = kept
    if len(param_cols) < 2 or len(feat_cols) < 2:
        log.warning(
            "Skipping continuity neighborhood matrices: need >=2 parameter cols and >=2 feature cols."
        )
        return

    used_cols = sorted(set(param_cols + feat_cols))
    work = dictionary[used_cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if len(work) < 8:
        log.warning("Skipping continuity neighborhood matrices: too few valid dictionary rows.")
        return

    param_arr = work[param_cols].to_numpy(dtype=float)
    feat_arr = work[feat_cols].to_numpy(dtype=float)

    def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
        lo = np.nanmin(arr, axis=0)
        hi = np.nanmax(arr, axis=0)
        span = hi - lo
        span = np.where(np.isfinite(span) & (span > 0.0), span, 1.0)
        return (arr - lo) / span

    def _pick_local_knn_neighborhood(arr_norm: np.ndarray, n_pick: int) -> np.ndarray:
        n_rows = int(arr_norm.shape[0])
        if n_rows <= n_pick:
            return np.arange(n_rows, dtype=int)
        diffs = arr_norm[:, None, :] - arr_norm[None, :, :]
        dmat = np.sqrt(np.sum(diffs * diffs, axis=2))
        np.fill_diagonal(dmat, np.inf)
        k_anchor = max(2, min(n_pick - 1, n_rows - 1))
        near_idx = np.argpartition(dmat, kth=k_anchor - 1, axis=1)[:, :k_anchor]
        local_compactness = np.mean(np.take_along_axis(dmat, near_idx, axis=1), axis=1)
        anchor = int(np.argmin(local_compactness))
        order = np.argsort(dmat[anchor], kind="mergesort")
        neighbors = [anchor]
        for idx in order:
            j = int(idx)
            if j == anchor:
                continue
            neighbors.append(j)
            if len(neighbors) >= n_pick:
                break
        return np.asarray(neighbors, dtype=int)

    def _build_subset_knn_edges(
        source_norm: np.ndarray,
        idx_subset: np.ndarray,
        k_graph: int = 2,
    ) -> list[tuple[int, int]]:
        n_subset = int(len(idx_subset))
        if n_subset < 2:
            return []
        local = source_norm[idx_subset]
        diffs = local[:, None, :] - local[None, :, :]
        dmat = np.sqrt(np.sum(diffs * diffs, axis=2))
        np.fill_diagonal(dmat, np.inf)
        k_use = max(1, min(int(k_graph), n_subset - 1))
        edges: set[tuple[int, int]] = set()
        for i in range(n_subset):
            nn = np.argsort(dmat[i], kind="mergesort")[:k_use]
            for j in nn:
                a, b = sorted((int(i), int(j)))
                edges.add((a, b))
        return sorted(edges)

    def _selected_topology_overlap(
        source_norm: np.ndarray,
        target_norm: np.ndarray,
        idx_subset: np.ndarray,
        k_eval: int,
    ) -> tuple[float, float]:
        n_rows = int(source_norm.shape[0])
        if n_rows < 3 or len(idx_subset) == 0:
            return np.nan, np.nan
        k_use = max(2, min(int(k_eval), n_rows - 1))
        overlaps: list[float] = []
        for ridx in idx_subset:
            i = int(ridx)
            src_d = np.sqrt(np.sum((source_norm - source_norm[i]) ** 2, axis=1))
            tgt_d = np.sqrt(np.sum((target_norm - target_norm[i]) ** 2, axis=1))
            src_d[i] = np.inf
            tgt_d[i] = np.inf
            src_nn = np.argpartition(src_d, kth=k_use - 1)[:k_use]
            tgt_nn = np.argpartition(tgt_d, kth=k_use - 1)[:k_use]
            overlaps.append(float(len(set(src_nn.tolist()).intersection(tgt_nn.tolist())) / k_use))
        arr = np.asarray(overlaps, dtype=float)
        if arr.size == 0:
            return np.nan, np.nan
        return float(np.percentile(arr, 10)), float(np.median(arr))

    def _selected_distance_correlation(
        source_norm: np.ndarray,
        target_norm: np.ndarray,
        idx_subset: np.ndarray,
    ) -> float:
        n_subset = int(len(idx_subset))
        if n_subset < 3:
            return np.nan
        src_local = source_norm[idx_subset]
        tgt_local = target_norm[idx_subset]
        src_d = np.sqrt(np.sum((src_local[:, None, :] - src_local[None, :, :]) ** 2, axis=2))
        tgt_d = np.sqrt(np.sum((tgt_local[:, None, :] - tgt_local[None, :, :]) ** 2, axis=2))
        tri = np.triu_indices(n_subset, k=1)
        src_v = src_d[tri]
        tgt_v = tgt_d[tri]
        if src_v.size < 2:
            return np.nan
        if np.nanstd(src_v) < 1e-12 or np.nanstd(tgt_v) < 1e-12:
            return np.nan
        corr = np.corrcoef(src_v, tgt_v)[0, 1]
        return float(corr) if np.isfinite(corr) else np.nan

    p_norm = _minmax_normalize(param_arr)
    f_norm = _minmax_normalize(feat_arr)
    n_pick = int(np.clip(int(n_points), 6, min(18, len(work))))
    idx_param_near = _pick_local_knn_neighborhood(p_norm, n_pick)
    idx_feat_near = _pick_local_knn_neighborhood(f_norm, n_pick)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, n_pick))

    def _draw_lower_triangle(
        subfig,
        cols: list[str],
        idx_subset: np.ndarray,
        edges: list[tuple[int, int]],
        marker_colors: np.ndarray,
        title: str,
    ) -> None:
        n_dim = len(cols)
        if n_dim < 2:
            ax = subfig.subplots(1, 1)
            ax.axis("off")
            ax.text(0.5, 0.5, "Insufficient dimensions", ha="center", va="center")
            return
        axes = subfig.subplots(n_dim - 1, n_dim - 1, squeeze=False)
        subfig.suptitle(title, fontsize=10, y=0.995)
        for row in range(n_dim - 1):
            y_idx = row + 1
            y_col = cols[y_idx]
            for col in range(n_dim - 1):
                ax = axes[row, col]
                if col >= y_idx:
                    ax.axis("off")
                    continue
                x_col = cols[col]
                x_all = work[x_col].to_numpy(dtype=float)
                y_all = work[y_col].to_numpy(dtype=float)
                ax.scatter(x_all, y_all, s=5, color="#C7C7C7", alpha=0.22, edgecolors="none", zorder=1)
                for edge_a, edge_b in edges:
                    idx_a = int(idx_subset[edge_a])
                    idx_b = int(idx_subset[edge_b])
                    ax.plot(
                        [x_all[idx_a], x_all[idx_b]],
                        [y_all[idx_a], y_all[idx_b]],
                        color="#7F7F7F",
                        alpha=0.32,
                        linewidth=0.6,
                        zorder=2,
                    )
                for rank, row_idx in enumerate(idx_subset):
                    ax.scatter(
                        [x_all[row_idx]],
                        [y_all[row_idx]],
                        s=26,
                        color=marker_colors[rank],
                        edgecolors="black",
                        linewidths=0.35,
                        zorder=3,
                    )
                if row == n_dim - 2:
                    ax.set_xlabel(x_col, fontsize=7)
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(y_col, fontsize=7)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=6, length=2)
                ax.grid(True, alpha=0.18, linewidth=0.4, zorder=0)

    def _direction_summary(
        source_norm: np.ndarray,
        target_norm: np.ndarray,
        idx_subset: np.ndarray,
    ) -> str:
        k_eval = max(2, min(8, len(work) - 1))
        overlap_p10, overlap_med = _selected_topology_overlap(source_norm, target_norm, idx_subset, k_eval)
        dist_corr = _selected_distance_correlation(source_norm, target_norm, idx_subset)
        return (
            f"selected_k={len(idx_subset)}, "
            f"overlap_p10={overlap_p10:.3f}, "
            f"overlap_median={overlap_med:.3f}, "
            f"distance_corr={dist_corr:.3f}"
        )

    def _plot_direction(
        idx_subset: np.ndarray,
        source_cols: list[str],
        target_cols: list[str],
        source_name: str,
        target_name: str,
        source_norm: np.ndarray,
        target_norm: np.ndarray,
        out_name_suffix: str,
    ) -> None:
        summary = _direction_summary(source_norm, target_norm, idx_subset)
        edges = _build_subset_knn_edges(source_norm, idx_subset, k_graph=2)
        fig = plt.figure(figsize=(18, 9))
        subfigs = fig.subfigures(1, 2, wspace=0.03)
        _draw_lower_triangle(
            subfigs[0],
            source_cols,
            idx_subset,
            edges,
            colors,
            f"Selected Neighborhood in {source_name} Space",
        )
        _draw_lower_triangle(
            subfigs[1],
            target_cols,
            idx_subset,
            edges,
            colors,
            f"Mapped Neighborhood in {target_name} Space",
        )
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=colors[i],
                markeredgecolor="black",
                markeredgewidth=0.35,
                markersize=5.5,
                label=str(i + 1),
            )
            for i in range(n_pick)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(12, n_pick),
            fontsize=7,
            frameon=False,
            title="Selected points (same colors in source and mapped matrices)",
            title_fontsize=8,
            bbox_to_anchor=(0.5, -0.005),
        )
        fig.suptitle(
            f"Neighborhood Continuity Matrix: {source_name} -> {target_name} | {summary}",
            fontsize=12,
            y=0.998,
        )
        out_path = out_base.with_name(f"{out_base.stem}_{out_name_suffix}{out_base.suffix}")
        # Stable unnumbered artifact for downstream compatibility.
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        # Keep sequentially numbered artifact family.
        _save_figure(fig, out_path, dpi=150, bbox_inches="tight")
        legacy_name = None
        if out_name_suffix == "param_to_feature_matrix":
            legacy_name = "1_2_18_continuity_neighborhood_example_param_to_feature_matrix.png"
        elif out_name_suffix == "feature_to_param_matrix":
            legacy_name = "1_2_19_continuity_neighborhood_example_feature_to_param_matrix.png"
        if legacy_name:
            fig.savefig(out_base.parent / legacy_name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _draw_direction_half(
        half_subfig,
        idx_subset: np.ndarray,
        source_cols: list[str],
        target_cols: list[str],
        source_name: str,
        target_name: str,
        source_norm: np.ndarray,
        target_norm: np.ndarray,
    ) -> None:
        edges = _build_subset_knn_edges(source_norm, idx_subset, k_graph=2)
        summary = _direction_summary(source_norm, target_norm, idx_subset)
        inner = half_subfig.subfigures(1, 2, wspace=0.03)
        _draw_lower_triangle(
            inner[0],
            source_cols,
            idx_subset,
            edges,
            colors,
            f"Selected Neighborhood in {source_name} Space",
        )
        _draw_lower_triangle(
            inner[1],
            target_cols,
            idx_subset,
            edges,
            colors,
            f"Mapped Neighborhood in {target_name} Space",
        )
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=colors[i],
                markeredgecolor="black",
                markeredgewidth=0.35,
                markersize=5.2,
                label=str(i + 1),
            )
            for i in range(n_pick)
        ]
        half_subfig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(12, n_pick),
            fontsize=7,
            frameon=False,
            title="Selected points",
            title_fontsize=8,
            bbox_to_anchor=(0.5, -0.005),
        )
        half_subfig.suptitle(
            f"{source_name} -> {target_name} | {summary}",
            fontsize=11,
            y=0.998,
        )

    _plot_direction(
        idx_param_near,
        param_cols,
        feat_cols,
        source_name="Parameter",
        target_name="Feature",
        source_norm=p_norm,
        target_norm=f_norm,
        out_name_suffix="param_to_feature_matrix",
    )
    _plot_direction(
        idx_feat_near,
        feat_cols,
        param_cols,
        source_name="Feature",
        target_name="Parameter",
        source_norm=f_norm,
        target_norm=p_norm,
        out_name_suffix="feature_to_param_matrix",
    )

    fig = plt.figure(figsize=(32, 11))
    halves = fig.subfigures(1, 2, wspace=0.02)
    _draw_direction_half(
        halves[0],
        idx_param_near,
        param_cols,
        feat_cols,
        source_name="Parameter",
        target_name="Feature",
        source_norm=p_norm,
        target_norm=f_norm,
    )
    _draw_direction_half(
        halves[1],
        idx_feat_near,
        feat_cols,
        param_cols,
        source_name="Feature",
        target_name="Parameter",
        source_norm=f_norm,
        target_norm=p_norm,
    )
    fig.suptitle(
        "Bidirectional Neighborhood Continuity (Topology-Aware, Deterministic Source Neighborhoods)",
        fontsize=13,
        fontweight="bold",
        y=0.997,
    )
    combined_path = out_base.with_name(f"{out_base.stem}_bidirectional{out_base.suffix}")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_eff_sim_vs_empirical(
    dataset: pd.DataFrame,
    dictionary: pd.DataFrame,
    path,
    fit_poly_by_plane: dict[int, np.ndarray | list[float] | tuple[float, ...]] | None = None,
) -> None:
    """2×2 scatter of empirical (x) vs simulated (y) efficiency for all 4 planes."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    x_min_vals, x_max_vals = [], []
    y_min_vals, y_max_vals = [], []
    for plane in range(1, 5):
        sim_col = f"eff_sim_{plane}"
        emp_col = f"eff_empirical_{plane}"
        for frame in (dataset, dictionary):
            if frame is None or frame.empty:
                continue
            if emp_col in frame.columns:
                x_vals = pd.to_numeric(frame[emp_col], errors="coerce").dropna()
                if not x_vals.empty:
                    x_min_vals.append(float(x_vals.min()))
                    x_max_vals.append(float(x_vals.max()))
            if sim_col in frame.columns:
                y_vals = pd.to_numeric(frame[sim_col], errors="coerce").dropna()
                if not y_vals.empty:
                    y_min_vals.append(float(y_vals.min()))
                    y_max_vals.append(float(y_vals.max()))

    x_min = min(x_min_vals) if x_min_vals else 0.0
    x_max = max(x_max_vals) if x_max_vals else 1.0
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        x_min, x_max = 0.0, 1.0
    if x_max <= x_min:
        x_min -= 0.01
        x_max += 0.01
    x_pad = 0.02 * (x_max - x_min)
    x_lo = x_min - x_pad
    x_hi = x_max + x_pad

    y_min_data = min(y_min_vals) if y_min_vals else 0.0
    y_max_data = max(y_max_vals) if y_max_vals else 1.0
    if not np.isfinite(y_min_data) or not np.isfinite(y_max_data):
        y_min_data, y_max_data = 0.0, 1.0
    # Simulated efficiencies are physical in [0, 1], but keep room for tiny numerical spillover.
    y_min = min(0.0, y_min_data)
    y_max = max(1.0, y_max_data)
    if y_max <= y_min:
        y_min -= 0.01
        y_max += 0.01
    y_pad = 0.02 * (y_max - y_min)
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    for plane in range(1, 5):
        ax = axes[(plane - 1) // 2, (plane - 1) % 2]
        sim_col = f"eff_sim_{plane}"
        emp_col = f"eff_empirical_{plane}"
        sim = pd.to_numeric(dataset.get(sim_col), errors="coerce")
        emp = pd.to_numeric(dataset.get(emp_col), errors="coerce")
        m = sim.notna() & emp.notna()
        if m.any():
            ax.scatter(emp[m], sim[m], s=12, alpha=0.4, color="#AAAAAA", zorder=2)
        # Dictionary points
        if not dictionary.empty:
            ds = pd.to_numeric(dictionary.get(sim_col), errors="coerce")
            de = pd.to_numeric(dictionary.get(emp_col), errors="coerce")
            dm = ds.notna() & de.notna()
            if dm.any():
                ax.scatter(de[dm], ds[dm], s=25, alpha=0.8, marker="x",
                           color="#E45756", linewidths=1.0, zorder=3, label="Dictionary")
        diag_lo = max(x_lo, y_lo)
        diag_hi = min(x_hi, y_hi)
        if diag_hi > diag_lo:
            ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", linewidth=1)
        if fit_poly_by_plane and plane in fit_poly_by_plane:
            coeffs = np.asarray(fit_poly_by_plane[plane], dtype=float)
            degree = max(int(coeffs.size) - 1, 0)
            xline = np.linspace(x_lo, x_hi, 200, dtype=float)
            yline = np.polyval(coeffs, xline)
            ax.plot(
                xline,
                yline,
                linestyle="--",
                linewidth=1.5,
                color="#2A9D8F",
                zorder=4,
                label=f"fit deg {degree}",
            )
            # Build multi-line fit-parameter annotation
            fit_lines = [f"fit deg {degree}:"]
            for idx, coef in enumerate(coeffs):
                power = degree - idx
                if power == 0:
                    fit_lines.append(f"  c0 = {coef:+.4g}")
                else:
                    fit_lines.append(f"  c{power} = {coef:+.4g}")
            ax.text(
                0.03, 0.97, "\n".join(fit_lines),
                transform=ax.transAxes, fontsize=6,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#2A9D8F", alpha=0.8),
                color="#2A9D8F", family="monospace",
            )
            ax.set_title(f"Plane {plane} (poly-fit dict cut)")
            ax.legend(fontsize=7, loc="lower right")
        else:
            ax.set_title(f"Plane {plane} (direct dict cut)")
        ax.set_xlabel("Empirical efficiency")
        ax.set_ylabel("Simulated efficiency")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Empirical vs Simulated Efficiency per Plane (grey=data, red×=dict)\n"
        "All planes use polynomial-fit cuts (simulated = P(empirical)).",
        fontsize=11,
    )
    fig.tight_layout()
    _save_figure(fig, path, dpi=150)
    plt.close(fig)


def _plot_empirical_eff2_vs_eff3(dataset: pd.DataFrame, dictionary: pd.DataFrame, path) -> None:
    """Single-panel scatter of empirical efficiency plane 2 vs plane 3."""
    x_col = "eff_empirical_2"
    y_col = "eff_empirical_3"
    if x_col not in dataset.columns or y_col not in dataset.columns:
        log.warning("Cannot plot empirical eff2 vs eff3: missing %s or %s", x_col, y_col)
        return

    x = pd.to_numeric(dataset[x_col], errors="coerce")
    y = pd.to_numeric(dataset[y_col], errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() == 0:
        log.warning("No valid points for empirical eff2 vs eff3 plot.")
        return

    xmin = float(min(x[m].min(), y[m].min()))
    xmax = float(max(x[m].max(), y[m].max()))
    pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.01

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x[m], y[m], s=12, alpha=0.4, color="#AAAAAA", zorder=2, label="Dataset")

    if not dictionary.empty:
        dx = pd.to_numeric(dictionary.get(x_col), errors="coerce")
        dy = pd.to_numeric(dictionary.get(y_col), errors="coerce")
        dm = dx.notna() & dy.notna()
        if dm.any():
            ax.scatter(dx[dm], dy[dm], s=28, alpha=0.85, marker="x",
                       color="#E45756", linewidths=1.0, zorder=3, label="Dictionary")

    # Reference line where empirical efficiencies in planes 2 and 3 are equal.
    ax.plot([xmin - pad, xmax + pad], [xmin - pad, xmax + pad], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Empirical efficiency (plane 2)")
    ax.set_ylabel("Empirical efficiency (plane 3)")
    ax.set_title("Empirical Efficiency: Plane 2 vs Plane 3")
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(xmin - pad, xmax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, path, dpi=150)
    plt.close(fig)


def _plot_iso_rate(
    dictionary: pd.DataFrame,
    path,
    *,
    eff_band_tolerance_pct: float = 10.0,
) -> None:
    """Scatter + iso-contours of global rate in the flux–eff plane (dictionary only)."""
    import ast as _ast
    from matplotlib.tri import Triangulation, LinearTriInterpolator

    gr_col = _pick_global_rate_column(dictionary, preferred="events_per_second_global_rate")
    if gr_col is None:
        log.warning("No global rate column — skipping iso-rate plot.")
        return
    if gr_col != "events_per_second_global_rate":
        log.info("Iso-rate plot using global-rate column: %s", gr_col)

    eff_band_tol_pct = float(eff_band_tolerance_pct)
    if not np.isfinite(eff_band_tol_pct):
        eff_band_tol_pct = 10.0
    eff_band_tol_pct = max(0.0, eff_band_tol_pct)

    n_total = int(len(dictionary))
    band_mask = _mask_sim_eff_within_tolerance_band(dictionary, eff_band_tol_pct)
    n_band = int(np.count_nonzero(band_mask))
    if n_band == 0:
        log.warning(
            "No dictionary rows satisfy iso-rate efficiency-band tolerance (<= %.3f%%).",
            eff_band_tol_pct,
        )
        return
    dict_use = dictionary.loc[band_mask].copy()

    try:
        if "eff_sim_1" in dict_use.columns:
            eff = pd.to_numeric(dict_use["eff_sim_1"], errors="coerce")
        else:
            effs = dict_use["efficiencies"].apply(_ast.literal_eval)
            eff = effs.apply(lambda x: float(x[0]))
    except Exception:
        log.warning("Could not parse efficiencies — skipping iso-rate plot.")
        return

    flux = pd.to_numeric(dict_use["flux_cm2_min"], errors="coerce")
    rate = pd.to_numeric(dict_use[gr_col], errors="coerce")
    xm = flux.to_numpy(dtype=float)
    ym = eff.to_numpy(dtype=float)
    zm = rate.to_numpy(dtype=float)
    mask = np.isfinite(xm) & np.isfinite(ym) & np.isfinite(zm)
    n_points = int(mask.sum())
    if n_points < 1:
        log.warning("Too few valid points for iso-rate plot.")
        return
    xm, ym, zm = xm[mask], ym[mask], zm[mask]

    flux_lo = float(np.min(xm))
    flux_hi = float(np.max(xm))
    eff_lo = float(np.min(ym))
    eff_hi = float(np.max(ym))
    x_span = max(0.0, flux_hi - flux_lo)
    y_span = max(0.0, eff_hi - eff_lo)
    x_pad = max(0.03 * x_span, 0.02)
    y_pad = max(0.03 * y_span, 0.02)
    flux_lo -= x_pad
    flux_hi += x_pad
    eff_lo -= y_pad
    eff_hi += y_pad

    g = 220
    xi = np.linspace(flux_lo, flux_hi, g)
    yi = np.linspace(eff_lo, eff_hi, g)
    Xi, Yi = np.meshgrid(xi, yi)

    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    rendered_contours = False
    try:
        if len(xm) >= 3:
            tri = Triangulation(xm, ym)
            interp = LinearTriInterpolator(tri, zm)
            Zi = interp(Xi, Yi)
            Zi = np.asarray(np.ma.filled(Zi, np.nan), dtype=float)
            finite_z = Zi[np.isfinite(Zi)]
            if finite_z.size >= 2:
                z_min = float(np.min(finite_z))
                z_max = float(np.max(finite_z))
                if z_max > z_min:
                    levels = np.linspace(z_min, z_max, 16)
                    cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, alpha=0.35, zorder=0)
                    cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.048)
                    cbar.set_label("Global rate [Hz]")
                    ax.contour(
                        Xi,
                        Yi,
                        Zi,
                        levels=levels[::2],
                        colors="k",
                        linewidths=0.35,
                        alpha=0.25,
                        zorder=1,
                    )
                    rendered_contours = True
    except Exception as exc:
        log.warning("Contour interpolation failed: %s", exc)

    if not rendered_contours:
        z_min = float(np.min(zm))
        z_max = float(np.max(zm))
        scatter_kwargs = {}
        if z_max > z_min:
            scatter_kwargs["vmin"] = z_min
            scatter_kwargs["vmax"] = z_max
        sc = ax.scatter(
            xm,
            ym,
            c=zm,
            cmap=cmap,
            s=72,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.6,
            zorder=2,
            **scatter_kwargs,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.048)
        cbar.set_label("Global rate [Hz]")
        ax.text(
            0.02,
            0.02,
            f"Contours unavailable (n={n_points} filtered point{'s' if n_points != 1 else ''})",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5", alpha=0.9),
            zorder=6,
        )

    ax.scatter(
        xm,
        ym,
        s=26,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label="Dictionary points",
    )
    ax.set_xlim(flux_lo, flux_hi)
    ax.set_ylim(eff_lo, eff_hi)
    ax.set_xlabel("Flux [cm^-2 min^-1]", fontsize=11)
    ax.set_ylabel("Efficiency (plane 1)", fontsize=11)
    ax.set_title(
        f"Iso-global-rate map ({n_band}/{n_total} dict entries, "
        f"4-eff band <= {eff_band_tol_pct:.2f}%)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, path, dpi=150)
    plt.close(fig)


def _plot_relerr_report(
    dataset: pd.DataFrame,
    dictionary: pd.DataFrame,
    path,
    relerr_cut_by_plane: dict[int, float] | None = None,
    min_events_cut: float | None = None,
    hist_y_scale: str = "log",
    fit_order_by_plane: dict[int, int] | None = None,
) -> None:
    """Comprehensive 4×3 report of relative errors for eff planes 1..4.

    Layout (rows = plane 1, plane 2, plane 3, plane 4):
      col 0: relerr histogram (signed, filtered to ±3 %)
      col 1: relerr vs n_events scatter (signed)
      col 2: relerr vs empirical efficiency scatter (signed)
    """
    BASE_HIST_CLIP_PCT = 3.0  # minimum histogram clip
    BASE_SCATTER_Y_CLIP_PCT = 5.0  # minimum scatter y-range clip
    CUT_COLOR = "#2A9D8F"
    hist_y_scale = str(hist_y_scale).strip().lower()
    if hist_y_scale not in {"linear", "log"}:
        log.warning("Invalid hist_y_scale=%r; falling back to 'log'.", hist_y_scale)
        hist_y_scale = "log"
    planes = [1, 2, 3, 4]
    fig, axes = plt.subplots(len(planes), 3, figsize=(16, 5 * len(planes)))

    for row, plane in enumerate(planes):
        re_col = f"relerr_eff_{plane}_fit_pct"
        if fit_order_by_plane and plane in fit_order_by_plane:
            plane_mode = f"poly-fit (deg {fit_order_by_plane[plane]})"
        else:
            plane_mode = "fit-based"
        emp_col = f"eff_empirical_{plane}"
        ev_col = "n_events"
        relerr_cut = None
        if relerr_cut_by_plane and plane in relerr_cut_by_plane:
            try:
                relerr_cut = abs(float(relerr_cut_by_plane[plane]))
            except (TypeError, ValueError):
                relerr_cut = None

        hist_clip_pct = BASE_HIST_CLIP_PCT
        scatter_y_clip_pct = BASE_SCATTER_Y_CLIP_PCT
        if relerr_cut is not None and np.isfinite(relerr_cut):
            hist_clip_pct = max(BASE_HIST_CLIP_PCT, relerr_cut)
            scatter_y_clip_pct = max(BASE_SCATTER_Y_CLIP_PCT, relerr_cut)

        hist_x_plot_lim = hist_clip_pct * 1.05
        scatter_y_plot_lim = scatter_y_clip_pct * 1.05

        re = pd.to_numeric(dataset.get(re_col), errors="coerce")
        emp_e = pd.to_numeric(dataset.get(emp_col), errors="coerce")
        n_ev = pd.to_numeric(dataset.get(ev_col), errors="coerce")
        is_dict = dataset.get("is_dictionary_entry", pd.Series(dtype=bool)).astype(bool)

        # ── Col 0: histogram (signed, filtered to ±hist_clip_pct) ──
        ax = axes[row, 0]
        filt = re.notna() & (re.abs() <= hist_clip_pct)
        data_vals = re[filt & (~is_dict if is_dict.any() else True)].dropna()
        dict_vals = re[filt & is_dict].dropna() if is_dict.any() else pd.Series(dtype=float)
        if not data_vals.empty:
            ax.hist(data_vals, bins=50, alpha=0.5, color="#4C78A8",
                    edgecolor="white", label="Dataset")
        if not dict_vals.empty:
            ax.hist(dict_vals, bins=50, alpha=0.6, color="#E45756",
                    edgecolor="white", label="Dictionary")
        all_filt = re[filt].dropna()
        med = all_filt.median() if not all_filt.empty else 0
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(med, color="#E45756", linestyle="--", linewidth=1,
                   label=f"median = {med:.2f}%")
        if relerr_cut is not None and np.isfinite(relerr_cut):
            ax.axvspan(-relerr_cut, relerr_cut, color=CUT_COLOR, alpha=0.07, zorder=0)
            ax.axvline(relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5,
                       label=f"dict cut ±{relerr_cut:.2f}%")
            ax.axvline(-relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5)
        ax.set_xlabel(f"Rel. error eff {plane} [%]")
        ax.set_ylabel("Count")
        ax.set_yscale(hist_y_scale)
        ax.set_title(
            f"Plane {plane} ({plane_mode}) — Rel. error dist. (|re| ≤ {hist_clip_pct:.1f}%)"
        )
        ax.set_xlim(-hist_x_plot_lim, hist_x_plot_lim)
        ax.legend(fontsize=7)

        # ── Col 1: relerr vs n_events ──
        ax = axes[row, 1]
        m = re.notna() & n_ev.notna() & (re.abs() <= scatter_y_clip_pct)
        if m.sum() > 0:
            off = m & ~is_dict if is_dict.any() else m
            on = m & is_dict if is_dict.any() else pd.Series(False, index=m.index)
            if off.sum() > 0:
                ax.scatter(n_ev[off], re[off], s=10, alpha=0.4,
                           color="#AAAAAA", zorder=2, label="Dataset")
            if on.sum() > 0:
                ax.scatter(n_ev[on], re[on], s=25, alpha=0.8, marker="x",
                           color="#E45756", linewidths=1.0, zorder=3, label="Dict")
        ax.axhline(0, color="black", linewidth=0.8)
        if relerr_cut is not None and np.isfinite(relerr_cut):
            ax.axhline(relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5,
                       label=f"relerr cut ±{relerr_cut:.2f}%")
            ax.axhline(-relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5)
        if min_events_cut is not None:
            try:
                min_ev = float(min_events_cut)
                if np.isfinite(min_ev):
                    ax.axvline(min_ev, color="#F4A261", linestyle="-.", linewidth=1.5,
                               label=f"events cut {min_ev:,.0f}")
            except (TypeError, ValueError):
                pass
        ax.set_ylim(-scatter_y_plot_lim, scatter_y_plot_lim)
        ax.set_xlabel("Number of events")
        ax.set_ylabel(f"Rel. error eff {plane} [%]")
        ax.set_title(
            f"Plane {plane} ({plane_mode}) — Rel. error vs Events (|re| ≤ {scatter_y_clip_pct:.1f}%)"
        )
        ax.legend(fontsize=7)

        # ── Col 2: relerr vs empirical efficiency ──
        ax = axes[row, 2]
        m = re.notna() & emp_e.notna() & (re.abs() <= scatter_y_clip_pct)
        if m.sum() > 0:
            off = m & ~is_dict if is_dict.any() else m
            on = m & is_dict if is_dict.any() else pd.Series(False, index=m.index)
            if off.sum() > 0:
                ax.scatter(emp_e[off], re[off], s=10, alpha=0.4,
                           color="#AAAAAA", zorder=2, label="Dataset")
            if on.sum() > 0:
                ax.scatter(emp_e[on], re[on], s=25, alpha=0.8, marker="x",
                           color="#E45756", linewidths=1.0, zorder=3, label="Dict")
        ax.axhline(0, color="black", linewidth=0.8)
        if relerr_cut is not None and np.isfinite(relerr_cut):
            ax.axhline(relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5,
                       label=f"relerr cut ±{relerr_cut:.2f}%")
            ax.axhline(-relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5)
        ax.set_ylim(-scatter_y_plot_lim, scatter_y_plot_lim)
        ax.set_xlabel(f"Empirical eff {plane}")
        ax.set_ylabel(f"Rel. error eff {plane} [%]")
        ax.set_title(
            f"Plane {plane} ({plane_mode}) — Rel. error vs Emp. eff (|re| ≤ {scatter_y_clip_pct:.1f}%)"
        )
        ax.legend(fontsize=7)

    fig.suptitle(
        "Relative Error Report — Efficiency Planes 1, 2, 3 & 4\n"
        "(all planes computed versus fitted polynomial simulated = P(empirical))",
        fontsize=13,
    )
    fig.tight_layout()
    _save_figure(fig, path, dpi=150)
    plt.close(fig)


def _normalize_dictionary_paramset_prototype_mode(value: object) -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"all", "all_candidates", "all_rows"}:
        return "all_candidates"
    if text in {"mean", "mean_numeric", "avg", "average"}:
        return "mean_numeric"
    if text in {"median", "median_numeric"}:
        return "median_numeric"
    return "max_events"


def _build_dictionary_from_candidates(
    dict_candidates: pd.DataFrame,
    *,
    param_cols: list[str],
    prototype_mode: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Build one dictionary prototype per unique physical parameter set.

    - `all_candidates`: keep every candidate row (no param-set deduplication).
    - `max_events`: legacy behavior (keep row with highest n_events).
    - `mean_numeric` / `median_numeric`: aggregate numeric columns per param set,
      while preserving representative non-numeric metadata from the max-events row.
    """
    if dict_candidates.empty:
        return (
            pd.DataFrame(columns=dict_candidates.columns),
            {
                "prototype_mode": str(prototype_mode),
                "rows_candidates": 0,
                "rows_unique_param_sets": 0,
                "numeric_columns_aggregated": 0,
                "aggregation_applied": False,
            },
        )

    mode = _normalize_dictionary_paramset_prototype_mode(prototype_mode)
    sorted_candidates = dict_candidates.sort_values("n_events", ascending=False).copy()
    if mode == "all_candidates":
        unique_param_sets = int(len(sorted_candidates.drop_duplicates(subset=param_cols, keep="first")))
        return sorted_candidates.reset_index(drop=True), {
            "prototype_mode": mode,
            "rows_candidates": int(len(dict_candidates)),
            "rows_unique_param_sets": unique_param_sets,
            "numeric_columns_aggregated": 0,
            "aggregation_applied": False,
        }

    representative = (
        dict_candidates.sort_values("n_events", ascending=False)
        .drop_duplicates(subset=param_cols, keep="first")
        .copy()
        .reset_index(drop=True)
    )
    info = {
        "prototype_mode": mode,
        "rows_candidates": int(len(dict_candidates)),
        "rows_unique_param_sets": int(len(representative)),
        "numeric_columns_aggregated": 0,
        "aggregation_applied": False,
    }
    if mode == "max_events":
        return representative, info

    numeric_cols: list[str] = []
    for col in dict_candidates.columns:
        if col in param_cols:
            continue
        series = dict_candidates[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
    if not numeric_cols:
        return representative, info

    gb = dict_candidates.groupby(param_cols, dropna=False, sort=False)[numeric_cols]
    if mode == "median_numeric":
        agg_num = gb.median().reset_index()
    else:
        agg_num = gb.mean().reset_index()

    rep_non_num_cols = [
        c for c in representative.columns
        if (c in param_cols) or (c not in numeric_cols)
    ]
    out = representative[rep_non_num_cols].merge(
        agg_num,
        on=param_cols,
        how="left",
        sort=False,
    )
    ordered_cols = [c for c in dict_candidates.columns if c in out.columns]
    out = out[ordered_cols].copy().reset_index(drop=True)

    info["numeric_columns_aggregated"] = int(len(numeric_cols))
    info["aggregation_applied"] = True
    return out, info


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 1.2: Build dictionary and dataset from collected data."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-csv", default=None,
                        help="Override input collected_data.csv path.")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    _clear_plots_dir()
    cfg_12 = config.get("step_1_2", {})

    input_path = Path(args.input_csv) if args.input_csv else DEFAULT_INPUT

    eff2_range = cfg_12.get("outlier_eff_2_range", [0.5, 1.0])
    eff3_range = cfg_12.get("outlier_eff_3_range", [0.5, 1.0])
    dict_relerr_eff1_fit_max = _resolve_relerr_eff_fit_threshold(cfg_12, plane=1, default=3.0)
    dict_relerr_eff2_fit_max = _resolve_relerr_eff_fit_threshold(cfg_12, plane=2, default=5.0)
    dict_relerr_eff3_fit_max = _resolve_relerr_eff_fit_threshold(cfg_12, plane=3, default=5.0)
    dict_relerr_eff4_fit_max = _resolve_relerr_eff_fit_threshold(cfg_12, plane=4, default=3.0)
    dict_min_events = cfg_12.get("dictionary_min_events", 20000)
    dict_exclude_all_equal_eff = bool(
        cfg_12.get("dictionary_exclude_rows_with_all_sim_eff_equal", False)
    )
    dict_paramset_prototype_mode = _normalize_dictionary_paramset_prototype_mode(
        cfg_12.get("dictionary_paramset_prototype", "max_events")
    )
    relerr_hist_y_scale = cfg_12.get("relerr_hist_y_scale", "log")
    plot_params = config.get("plot_parameters", None)  # None = use all param_cols
    if plot_params is None:
        legacy_plot_params = cfg_12.get("plot_parameters", None)
        if legacy_plot_params is not None:
            log.warning(
                "Deprecated config key step_1_2.plot_parameters detected; use top-level plot_parameters."
            )
            plot_params = legacy_plot_params
    fit_polynomial_order_cfg = cfg_12.get(
        "eff_fit_polynomial_order",
        cfg_12.get("fit_polynomial_order", 3),
    )
    iso_rate_eff_band_tol_cfg = cfg_12.get(
        "iso_rate_efficiency_band_tolerance_pct",
        config.get("iso_rate_efficiency_band_tolerance_pct", 10.0),
    )
    task_ids_used = _safe_task_ids(config.get("task_ids", [1]))
    preferred_count_prefixes = _preferred_count_prefixes_for_task_ids(task_ids_used)
    try:
        fit_polynomial_order = max(1, int(fit_polynomial_order_cfg))
    except (TypeError, ValueError):
        log.warning(
            "Invalid eff_fit_polynomial_order=%r; falling back to 3.",
            fit_polynomial_order_cfg,
        )
        fit_polynomial_order = 3
    try:
        iso_rate_eff_band_tolerance_pct = max(0.0, float(iso_rate_eff_band_tol_cfg))
    except (TypeError, ValueError):
        log.warning(
            "Invalid iso_rate_efficiency_band_tolerance_pct=%r; falling back to 10.",
            iso_rate_eff_band_tol_cfg,
        )
        iso_rate_eff_band_tolerance_pct = 10.0

    # ── Load ─────────────────────────────────────────────────────────
    if not input_path.exists():
        log.error("Input CSV not found: %s", input_path)
        return 1

    log.info("Loading collected data: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    log.info("  Rows loaded: %d", len(df))

    # ── Compute empirical efficiencies ───────────────────────────────
    prefix, topology_value_suffix = _find_topology_prefix(
        df,
        preferred_prefixes=preferred_count_prefixes,
    )
    log.info("  Using topology prefix: %s (source=%s)", prefix, topology_value_suffix)

    four_col = f"{prefix}1234_{topology_value_suffix}"
    miss_cols = {
        1: f"{prefix}234_{topology_value_suffix}",
        2: f"{prefix}134_{topology_value_suffix}",
        3: f"{prefix}124_{topology_value_suffix}",
        4: f"{prefix}123_{topology_value_suffix}",
    }

    # Coerce to numeric
    for col in [four_col, *miss_cols.values()]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    empirical_cols = {
        f"eff_empirical_{plane}": _compute_eff(df[four_col], df[miss_col])
        for plane, miss_col in miss_cols.items()
    }
    df = pd.concat([df, pd.DataFrame(empirical_cols, index=df.index)], axis=1)

    # Parse simulated efficiencies from the 'efficiencies' column
    if "efficiencies" in df.columns:
        effs = df["efficiencies"].apply(_parse_efficiencies)
        effs_arr = np.asarray(effs.to_list(), dtype=float)
        if effs_arr.ndim == 2 and effs_arr.shape[1] >= 4:
            eff_sim_df = pd.DataFrame(
                effs_arr[:, :4],
                index=df.index,
                columns=[f"eff_sim_{i}" for i in range(1, 5)],
            )
            df = pd.concat([df, eff_sim_df], axis=1)

    # Determine event count column
    event_col = None
    for candidate in ("selected_rows", "requested_rows", "generated_events_count",
                       "num_events", "event_count"):
        if candidate in df.columns:
            event_col = candidate
            break
    if event_col:
        if event_col != "selected_rows":
            log.info("Event count column resolved via fallback: '%s'.", event_col)
        n_events_series = pd.to_numeric(df[event_col], errors="coerce")
    else:
        log.warning("No event count column found; setting n_events = NaN.")
        n_events_series = pd.Series(np.nan, index=df.index, dtype=float)
    df = pd.concat([df, n_events_series.rename("n_events")], axis=1)

    # ── Outlier removal ──────────────────────────────────────────────
    n_before = len(df)
    mask = np.ones(len(df), dtype=bool)
    mask &= df["eff_empirical_2"].between(eff2_range[0], eff2_range[1])
    mask &= df["eff_empirical_3"].between(eff3_range[0], eff3_range[1])
    # Also remove rows where eff_empirical values are NaN
    mask &= df["eff_empirical_2"].notna()
    mask &= df["eff_empirical_3"].notna()

    df_clean = df.loc[mask].copy().reset_index(drop=True)
    n_outliers = n_before - len(df_clean)
    log.info("  Outlier removal: %d outliers dropped, %d rows remain.", n_outliers, len(df_clean))

    # Fit acceptance-adjusted polynomial relations for all 4 planes:
    # simulated ≈ P(empirical)
    fit_poly_by_plane: dict[int, list[float]] = {}
    fit_order_by_plane: dict[int, int] = {}
    for plane in (1, 2, 3, 4):
        fit_coeffs = _fit_simulated_vs_empirical(
            df_clean,
            plane,
            poly_order=fit_polynomial_order,
        )
        if fit_coeffs is None:
            log.warning("  Plane %d fit unavailable; fit-based cut disabled for this plane.", plane)
            df_clean[f"eff_fitline_{plane}"] = np.nan
            df_clean[f"relerr_eff_{plane}_fit_pct"] = np.nan
            df_clean[f"abs_relerr_eff_{plane}_fit_pct"] = np.nan
            continue
        coeff_list = [float(c) for c in np.asarray(fit_coeffs, dtype=float)]
        fit_poly_by_plane[plane] = coeff_list
        fit_order_by_plane[plane] = len(coeff_list) - 1
        _compute_fit_relerr_columns(df_clean, plane, coeff_list)
        if fit_order_by_plane[plane] < fit_polynomial_order:
            log.warning(
                "  Plane %d requested poly order %d, using %d (insufficient points).",
                plane,
                fit_polynomial_order,
                fit_order_by_plane[plane],
            )
        log.info(
            "  Plane %d polynomial fit (deg %d): simulated = %s",
            plane,
            fit_order_by_plane[plane],
            _format_polynomial(coeff_list, variable="empirical", precision=6),
        )

    # Fit isotonic (monotonic) calibration for all 4 planes:
    isotonic_by_plane: dict[int, dict] = {}
    for plane in (1, 2, 3, 4):
        iso = _fit_isotonic_calibration(df_clean, plane)
        if iso is not None:
            isotonic_by_plane[plane] = iso
            log.info(
                "  Plane %d isotonic calibration: %d knots, domain [%.4f, %.4f] -> [%.4f, %.4f], "
                "slope_lo=%.3f slope_hi=%.3f",
                plane,
                len(iso["x_knots"]),
                iso["x_min"],
                iso["x_max"],
                iso["y_knots"][0],
                iso["y_knots"][-1],
                iso["slope_lo"],
                iso["slope_hi"],
            )
        else:
            log.warning("  Plane %d isotonic calibration unavailable.", plane)

    # ── Dataset = the full clean table ───────────────────────────────
    dataset = df_clean.copy()

    # ── Dictionary selection ─────────────────────────────────────────
    # Criteria:
    # 1. abs relative error of eff_2 and eff_3 < threshold
    # 2. abs relative error to fit line for eff_1 and eff_4 < threshold
    # 3. n_events >= minimum
    # 4. One prototype per unique parameter set (max-events row or numeric aggregation).
    log.info("  Dictionary param-set prototype mode: %s", dict_paramset_prototype_mode)
    dict_mask = np.ones(len(df_clean), dtype=bool)
    if df_clean["abs_relerr_eff_1_fit_pct"].notna().any():
        dict_mask &= df_clean["abs_relerr_eff_1_fit_pct"] < dict_relerr_eff1_fit_max
    if df_clean["abs_relerr_eff_2_fit_pct"].notna().any():
        dict_mask &= df_clean["abs_relerr_eff_2_fit_pct"] < dict_relerr_eff2_fit_max
    if df_clean["abs_relerr_eff_3_fit_pct"].notna().any():
        dict_mask &= df_clean["abs_relerr_eff_3_fit_pct"] < dict_relerr_eff3_fit_max
    if df_clean["abs_relerr_eff_4_fit_pct"].notna().any():
        dict_mask &= df_clean["abs_relerr_eff_4_fit_pct"] < dict_relerr_eff4_fit_max
    dict_mask &= df_clean["n_events"] >= dict_min_events

    n_equal_eff_rows_excluded = 0
    if dict_exclude_all_equal_eff:
        equal_eff_mask = _mask_all_sim_eff_equal(df_clean)
        n_equal_eff_rows_excluded = int(np.count_nonzero(dict_mask & equal_eff_mask))
        dict_mask &= ~equal_eff_mask
        log.info(
            "  Dictionary candidates: excluded %d row(s) with eff_sim_1==eff_sim_2==eff_sim_3==eff_sim_4.",
            n_equal_eff_rows_excluded,
        )

    dict_candidates = df_clean.loc[dict_mask].copy()
    log.info("  Dictionary candidates (pass quality): %d / %d",
             len(dict_candidates), len(df_clean))

    # Unique parameter set = (flux_cm2_min, cos_n, eff_sim_1, eff_sim_2, eff_sim_3, eff_sim_4)
    # plus z-planes (already filtered to one config)
    param_cols = ["flux_cm2_min", "cos_n"]
    for i in range(1, 5):
        if f"eff_sim_{i}" in dict_candidates.columns:
            param_cols.append(f"eff_sim_{i}")

    if not dict_candidates.empty:
        dictionary, dict_prototype_info = _build_dictionary_from_candidates(
            dict_candidates,
            param_cols=param_cols,
            prototype_mode=dict_paramset_prototype_mode,
        )
        if dict_prototype_info.get("aggregation_applied", False):
            log.info(
                "  Dictionary prototypes: mode=%s aggregated %d numeric column(s) per param set.",
                dict_prototype_info.get("prototype_mode"),
                int(dict_prototype_info.get("numeric_columns_aggregated", 0)),
            )
    else:
        dictionary = pd.DataFrame(columns=df_clean.columns)
        dict_prototype_info = {
            "prototype_mode": str(dict_paramset_prototype_mode),
            "rows_candidates": 0,
            "rows_unique_param_sets": 0,
            "numeric_columns_aggregated": 0,
            "aggregation_applied": False,
        }

    log.info("  Dictionary entries (unique param sets): %d", len(dictionary))

    # ── Continuity validation ─────────────────────────────────────
    cv_cfg = cfg_12.get("continuity_validation", {})
    cv_metrics_pre_filter: dict | None = None
    continuity_feature_cols: list[str] = []
    continuity_feature_resolution: dict | None = None
    feature_selection_report: dict = {
        "enabled": False,
        "status": "disabled",
        "selected_feature_columns": [],
    }
    cv_filtering = {
        "enabled": False,
        "applied": False,
        "reason": "continuity_validation_disabled",
        "rows_before": int(len(dictionary)),
        "rows_after": int(len(dictionary)),
        "rows_flagged": 0,
        "rows_removed": 0,
        "removed_fraction": 0.0,
    }
    if cv_cfg.get("enabled", True):
        feature_selection_enabled = bool(cv_cfg.get("feature_selection_enabled", True))
        dict_cv_pre, data_cv_pre, continuity_feature_cols, continuity_feature_resolution = (
            _resolve_step21_feature_space_for_continuity(
                dictionary,
                dataset,
                config,
                prefer_step12_selected_artifact=not feature_selection_enabled,
            )
        )
        if continuity_feature_resolution and continuity_feature_resolution.get("warning"):
            log.warning("  CONTINUITY feature-space: %s", continuity_feature_resolution["warning"])
        log.info(
            "  CONTINUITY feature-space: strategy=%s, features=%d",
            (continuity_feature_resolution or {}).get("strategy", "unknown"),
            len(continuity_feature_cols),
        )
        if feature_selection_enabled:
            selected_cols, selection_report = _select_features_for_continuity_and_inference(
                dictionary=dict_cv_pre,
                dataset=data_cv_pre,
                param_cols=param_cols,
                base_feature_cols=continuity_feature_cols,
                cv_cfg=cv_cfg,
            )
            feature_selection_report = selection_report if isinstance(selection_report, dict) else {}
            feature_selection_report["enabled"] = True
            continuity_feature_cols = [
                c for c in selected_cols
                if c in dict_cv_pre.columns and c in data_cv_pre.columns
            ]
            feature_selection_report["selected_feature_columns"] = list(continuity_feature_cols)
            feature_selection_report["selected_feature_count"] = int(len(continuity_feature_cols))
            if continuity_feature_cols:
                try:
                    _write_step12_selected_feature_columns(
                        SELECTED_FEATURE_COLUMNS_PATH,
                        selected_feature_columns=continuity_feature_cols,
                        selection_report=feature_selection_report,
                    )
                    log.info(
                        "  CONTINUITY feature selection: wrote %d selected feature(s) to %s",
                        len(continuity_feature_cols),
                        SELECTED_FEATURE_COLUMNS_PATH,
                    )
                except OSError as exc:
                    log.warning(
                        "  CONTINUITY feature selection: could not write %s (%s).",
                        SELECTED_FEATURE_COLUMNS_PATH,
                        exc,
                    )
                continuity_feature_resolution = (
                    dict(continuity_feature_resolution) if isinstance(continuity_feature_resolution, dict) else {}
                )
                continuity_feature_resolution["strategy"] = "step_1_2_selected_auto"
                continuity_feature_resolution["selected_feature_columns_path"] = str(SELECTED_FEATURE_COLUMNS_PATH)
                continuity_feature_resolution["feature_selection"] = feature_selection_report
                log.info(
                    "  CONTINUITY selected features: %d (best_score=%.4f)",
                    len(continuity_feature_cols),
                    float(feature_selection_report.get("selected_score", np.nan)),
                )
            else:
                log.warning(
                    "  CONTINUITY feature selection returned no usable columns; continuing with resolved base feature-space."
                )

        cv_status_pre, cv_metrics_pre_filter, cv_messages_pre = _validate_dictionary_continuity(
            dict_cv_pre,
            data_cv_pre,
            param_cols,
            cv_cfg,
            isotonic_by_plane,
            feature_cols=continuity_feature_cols,
            feature_resolution=continuity_feature_resolution,
        )

        for msg in cv_messages_pre:
            if cv_status_pre == "FAIL":
                log.error("  CONTINUITY(pre): %s", msg)
            elif cv_status_pre == "WARN":
                log.warning("  CONTINUITY(pre): %s", msg)
            else:
                log.info("  CONTINUITY(pre): %s", msg)

        dictionary, cv_filtering = _apply_local_continuity_filter(dictionary, cv_metrics_pre_filter, cv_cfg)
        if cv_filtering.get("enabled", False):
            if cv_filtering.get("applied", False):
                log.warning(
                    "  CONTINUITY FILTER: removed %d/%d dictionary row(s) (%.2f%%); "
                    "flagged_local=%d flagged_topology=%d flagged_injectivity=%d (local_cv_threshold=%.3f).",
                    cv_filtering.get("rows_removed", 0),
                    cv_filtering.get("rows_before", 0),
                    100.0 * float(cv_filtering.get("removed_fraction", 0.0)),
                    cv_filtering.get("rows_flagged_local", 0),
                    cv_filtering.get("rows_flagged_topology", 0),
                    cv_filtering.get("rows_flagged_injectivity", 0),
                    float(cv_filtering.get("cv_threshold", cv_cfg.get("local_continuity_cv_p95_max", 0.50))),
                )
                dict_cv_post, data_cv_post, continuity_feature_cols, continuity_feature_resolution = (
                    _resolve_step21_feature_space_for_continuity(
                        dictionary,
                        dataset,
                        config,
                        prefer_step12_selected_artifact=not feature_selection_enabled,
                    )
                )
                selected_after_filter = [
                    c for c in (feature_selection_report.get("selected_feature_columns", []) or [])
                    if c in dict_cv_post.columns and c in data_cv_post.columns
                ]
                if selected_after_filter:
                    continuity_feature_cols = selected_after_filter
                    continuity_feature_resolution = (
                        dict(continuity_feature_resolution) if isinstance(continuity_feature_resolution, dict) else {}
                    )
                    continuity_feature_resolution["strategy"] = "step_1_2_selected_auto"
                    continuity_feature_resolution["selected_feature_columns_path"] = str(SELECTED_FEATURE_COLUMNS_PATH)
                    continuity_feature_resolution["feature_selection"] = feature_selection_report
                cv_status, cv_metrics, cv_messages = _validate_dictionary_continuity(
                    dict_cv_post,
                    data_cv_post,
                    param_cols,
                    cv_cfg,
                    isotonic_by_plane,
                    feature_cols=continuity_feature_cols,
                    feature_resolution=continuity_feature_resolution,
                )
                for msg in cv_messages:
                    if cv_status == "FAIL":
                        log.error("  CONTINUITY(post): %s", msg)
                    elif cv_status == "WARN":
                        log.warning("  CONTINUITY(post): %s", msg)
                    else:
                        log.info("  CONTINUITY(post): %s", msg)
            else:
                cv_status = cv_status_pre
                cv_metrics = cv_metrics_pre_filter
                log.info(
                    "  CONTINUITY FILTER: not applied (%s).",
                    cv_filtering.get("reason", "unknown"),
                )
        else:
            cv_status = cv_status_pre
            cv_metrics = cv_metrics_pre_filter

        if cv_cfg.get("fail_on_error", False) and cv_status == "FAIL":
            log.error("Dictionary continuity validation FAILED. Aborting.")
            return 1
    else:
        cv_metrics = {"enabled": False, "status": "SKIPPED", "feature_columns_used": []}

    # Mark dictionary membership in dataset for downstream awareness
    if "filename_base" in dataset.columns and "filename_base" in dictionary.columns:
        dict_ids = set(dictionary["filename_base"].dropna().astype(str))
        dataset["is_dictionary_entry"] = (
            dataset["filename_base"].astype(str).isin(dict_ids)
        )
    else:
        dataset["is_dictionary_entry"] = False

    # ── Save ─────────────────────────────────────────────────────────
    dataset_path = FILES_DIR / "dataset.csv"
    dictionary_path = FILES_DIR / "dictionary.csv"
    dataset.to_csv(dataset_path, index=False)
    dictionary.to_csv(dictionary_path, index=False)
    log.info("Wrote dataset:    %s (%d rows)", dataset_path, len(dataset))
    log.info("Wrote dictionary: %s (%d rows)", dictionary_path, len(dictionary))

    # Record z-plane positions used in this analysis (plane order: 1..4, in mm).
    z_positions_used = None
    z_plane_map = None
    z_cols = [f"z_plane_{i}" for i in range(1, 5)]
    if all(col in dataset.columns for col in z_cols):
        z_vals = []
        for i, col in enumerate(z_cols, start=1):
            vals = pd.to_numeric(dataset[col], errors="coerce").dropna().unique()
            if len(vals) == 0:
                z_vals = []
                break
            if len(vals) > 1:
                log.warning("Multiple %s values found; using first value in summary.", col)
            z_vals.append(float(vals[0]))
        if len(z_vals) == 4:
            z_positions_used = z_vals
            z_plane_map = {str(i): z for i, z in enumerate(z_vals, start=1)}
    if z_positions_used is None:
        cfg_z = config.get("z_position_config")
        if isinstance(cfg_z, (list, tuple)) and len(cfg_z) >= 4:
            z_positions_used = [float(cfg_z[i]) for i in range(4)]
            z_plane_map = {str(i): z for i, z in enumerate(z_positions_used, start=1)}

    summary = {
        "input_rows": n_before,
        "outliers_removed": n_outliers,
        "dataset_rows": len(dataset),
        "dictionary_candidates": int(dict_mask.sum()),
        "dictionary_rows": len(dictionary),
        "z_positions_used_mm": z_positions_used,
        "z_plane_map_mm": z_plane_map,
        "task_ids_used": task_ids_used,
        "count_prefix_preferred_order": preferred_count_prefixes,
        "count_prefix_used": prefix,
        "topology_value_source": topology_value_suffix,
        "eff2_range": eff2_range,
        "eff3_range": eff3_range,
        "fit_polynomial_order_requested": int(fit_polynomial_order),
        "fit_polynomial_order_by_plane": {
            str(k): int(v) for k, v in fit_order_by_plane.items()
        },
        "fit_polynomial_relation": "simulated = P(empirical)",
        "fit_polynomial_x_variable": "empirical_efficiency",
        "fit_polynomial_y_variable": "simulated_efficiency",
        "fit_polynomial_eff_1": list(fit_poly_by_plane[1]) if 1 in fit_poly_by_plane else None,
        "fit_polynomial_eff_2": list(fit_poly_by_plane[2]) if 2 in fit_poly_by_plane else None,
        "fit_polynomial_eff_3": list(fit_poly_by_plane[3]) if 3 in fit_poly_by_plane else None,
        "fit_polynomial_eff_4": list(fit_poly_by_plane[4]) if 4 in fit_poly_by_plane else None,
        "fit_line_eff_1": list(fit_poly_by_plane[1]) if 1 in fit_poly_by_plane else None,
        "fit_line_eff_2": list(fit_poly_by_plane[2]) if 2 in fit_poly_by_plane else None,
        "fit_line_eff_3": list(fit_poly_by_plane[3]) if 3 in fit_poly_by_plane else None,
        "fit_line_eff_4": list(fit_poly_by_plane[4]) if 4 in fit_poly_by_plane else None,
        "dict_relerr_eff1_fit_max_pct": dict_relerr_eff1_fit_max,
        "dict_relerr_eff2_fit_max_pct": dict_relerr_eff2_fit_max,
        "dict_relerr_eff3_fit_max_pct": dict_relerr_eff3_fit_max,
        "dict_relerr_eff4_fit_max_pct": dict_relerr_eff4_fit_max,
        "dict_min_events": dict_min_events,
        "dictionary_paramset_prototype_mode": dict_paramset_prototype_mode,
        "dictionary_paramset_prototype_info": dict_prototype_info,
        "dictionary_exclude_rows_with_all_sim_eff_equal": dict_exclude_all_equal_eff,
        "dictionary_rows_excluded_all_sim_eff_equal": n_equal_eff_rows_excluded,
        "relerr_hist_y_scale": relerr_hist_y_scale,
        "iso_rate_efficiency_band_tolerance_pct": float(iso_rate_eff_band_tolerance_pct),
        "isotonic_calibration_eff_1": isotonic_by_plane.get(1),
        "isotonic_calibration_eff_2": isotonic_by_plane.get(2),
        "isotonic_calibration_eff_3": isotonic_by_plane.get(3),
        "isotonic_calibration_eff_4": isotonic_by_plane.get(4),
        "isotonic_calibration_relation": "simulated = f_isotonic(empirical)",
        "continuity_validation": _cv_metrics_for_json(cv_metrics),
        "continuity_validation_pre_filter": (
            _cv_metrics_for_json(cv_metrics_pre_filter) if cv_metrics_pre_filter else None
        ),
        "continuity_feature_resolution": continuity_feature_resolution,
        "continuity_feature_selection": feature_selection_report,
        "continuity_selected_feature_columns_path": (
            str(SELECTED_FEATURE_COLUMNS_PATH) if SELECTED_FEATURE_COLUMNS_PATH.exists() else None
        ),
        "continuity_filtering": cv_filtering,
    }
    with open(FILES_DIR / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    dataset_for_plots = dataset
    dictionary_for_plots = dictionary
    continuity_feature_cols_for_plots = continuity_feature_cols
    if not dictionary.empty and not dataset.empty:
        (
            dictionary_for_plots,
            dataset_for_plots,
            continuity_feature_cols_for_plots,
            _,
        ) = _resolve_step21_feature_space_for_continuity(dictionary, dataset, config)
        selected_for_plots = [
            c for c in (feature_selection_report.get("selected_feature_columns", []) or [])
            if c in dictionary_for_plots.columns and c in dataset_for_plots.columns
        ]
        if selected_for_plots:
            continuity_feature_cols_for_plots = selected_for_plots

    # ── Diagnostic plots ─────────────────────────────────────────────
    _make_plots(
        dataset_for_plots,
        dictionary_for_plots,
        param_cols,
        fit_poly_by_plane=fit_poly_by_plane,
        fit_order_by_plane=fit_order_by_plane,
        dict_relerr_eff1_fit_max=dict_relerr_eff1_fit_max,
        dict_relerr_eff2_fit_max=dict_relerr_eff2_fit_max,
        dict_relerr_eff3_fit_max=dict_relerr_eff3_fit_max,
        dict_relerr_eff4_fit_max=dict_relerr_eff4_fit_max,
        dict_min_events=dict_min_events,
        relerr_hist_y_scale=relerr_hist_y_scale,
        iso_rate_eff_band_tolerance_pct=iso_rate_eff_band_tolerance_pct,
        plot_params=plot_params,
        cv_metrics=cv_metrics,
        continuity_feature_cols=continuity_feature_cols_for_plots,
    )

    log.info("Done.")
    return 0


def _make_plots(
    dataset: pd.DataFrame,
    dictionary: pd.DataFrame,
    param_cols: list[str],
    fit_poly_by_plane: dict[int, list[float]] | None,
    fit_order_by_plane: dict[int, int] | None,
    dict_relerr_eff1_fit_max: float,
    dict_relerr_eff2_fit_max: float,
    dict_relerr_eff3_fit_max: float,
    dict_relerr_eff4_fit_max: float,
    dict_min_events: float,
    relerr_hist_y_scale: str,
    iso_rate_eff_band_tolerance_pct: float,
    plot_params: list[str] | None = None,
    cv_metrics: dict | None = None,
    continuity_feature_cols: list[str] | None = None,
) -> None:
    """Generate concise diagnostic plots.

    *plot_params* selects which parameters appear in histograms, scatter
    matrix, and coverage plots.  If None, all *param_cols* are used.
    """
    # Resolve which params to plot
    if plot_params:
        plot_cols = [c for c in plot_params if c in dataset.columns]
    else:
        plot_cols = [c for c in param_cols if c in dataset.columns]
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    if dataset.empty:
        log.warning("Dataset is empty — skipping plots.")
        return

    # ── 1. Parameter histograms: data vs dictionary ──────────────────
    hist_cols = [c for c in plot_cols if c in dataset.columns]
    if hist_cols:
        n_cols = len(hist_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        for ax, col in zip(axes, hist_cols):
            d_vals = pd.to_numeric(dataset[col], errors="coerce").dropna()
            if not d_vals.empty:
                ax.hist(d_vals, bins=30, alpha=0.5, color="#4C78A8",
                        label="Dataset", density=True)
            if not dictionary.empty:
                dict_vals = pd.to_numeric(dictionary[col], errors="coerce").dropna()
                if not dict_vals.empty:
                    ax.hist(dict_vals, bins=30, alpha=0.6, color="#E45756",
                            label="Dictionary", density=True)
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.set_title(col)
            ax.legend(fontsize=7)
        fig.suptitle("Parameter distributions: dataset vs dictionary", fontsize=11)
        fig.tight_layout()
        _save_figure(fig, PLOTS_DIR / "parameter_histograms.png")
        plt.close(fig)

    # ── 2. Scatter matrix of key params (data vs dictionary) ─────────
    scatter_cols = [c for c in plot_cols if c in dataset.columns]
    n_sc = len(scatter_cols)
    if n_sc >= 2:
        fig, axes = plt.subplots(n_sc, n_sc, figsize=(3 * n_sc, 3 * n_sc))
        for i, cy in enumerate(scatter_cols):
            for j, cx in enumerate(scatter_cols):
                ax = axes[i][j] if n_sc > 1 else axes
                if j < i:
                    # Below diagonal: hide
                    ax.set_visible(False)
                elif i == j:
                    # Diagonal: histogram with log scale
                    d_vals = pd.to_numeric(dataset[cx], errors="coerce").dropna()
                    if not d_vals.empty:
                        ax.hist(d_vals, bins=25, alpha=0.5, color="#4C78A8")
                    if not dictionary.empty:
                        dict_vals = pd.to_numeric(dictionary[cx], errors="coerce").dropna()
                        if not dict_vals.empty:
                            ax.hist(dict_vals, bins=25, alpha=0.6, color="#E45756")
                    ax.set_yscale("log")
                    ax.set_xlabel(cx, fontsize=7)
                else:
                    # Above diagonal: scatter (keeps first plot parameter on y-axis)
                    dx = pd.to_numeric(dataset[cx], errors="coerce")
                    dy = pd.to_numeric(dataset[cy], errors="coerce")
                    m = dx.notna() & dy.notna()
                    if m.sum() > 0:
                        ax.scatter(dx[m], dy[m], s=5, alpha=0.3, color="#AAAAAA")
                    if not dictionary.empty:
                        ddx = pd.to_numeric(dictionary[cx], errors="coerce")
                        ddy = pd.to_numeric(dictionary[cy], errors="coerce")
                        dm = ddx.notna() & ddy.notna()
                        if dm.sum() > 0:
                            ax.scatter(ddx[dm], ddy[dm], s=15, alpha=0.7,
                                       marker="x", color="#E45756", linewidths=0.8)
                    ax.set_xlabel(cx, fontsize=7)
                    ax.set_ylabel(cy, fontsize=7)
                ax.tick_params(labelsize=6)
        fig.suptitle("Parameter scatter matrix (grey=data, red×=dictionary)", fontsize=10)
        fig.tight_layout()
        _save_figure(fig, PLOTS_DIR / "parameter_scatter_matrix.png")
        plt.close(fig)

    # ── 3. Comprehensive relative error report for eff 1..4 ────
    _plot_relerr_report(
        dataset,
        dictionary,
        PLOTS_DIR / "relerr_eff_report.png",
        relerr_cut_by_plane={
            1: dict_relerr_eff1_fit_max,
            2: dict_relerr_eff2_fit_max,
            3: dict_relerr_eff3_fit_max,
            4: dict_relerr_eff4_fit_max,
        },
        min_events_cut=dict_min_events,
        hist_y_scale=relerr_hist_y_scale,
        fit_order_by_plane=fit_order_by_plane,
    )

    # ── 4. Events distribution ───────────────────────────────────────
    if "n_events" in dataset.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        d_ev = dataset["n_events"].dropna()
        if not d_ev.empty:
            ax.hist(d_ev, bins=40, alpha=0.5, color="#4C78A8", label="Dataset")
        if not dictionary.empty and "n_events" in dictionary.columns:
            dd_ev = dictionary["n_events"].dropna()
            if not dd_ev.empty:
                ax.hist(dd_ev, bins=40, alpha=0.6, color="#E45756", label="Dictionary")
        ax.set_xlabel("Number of events")
        ax.set_ylabel("Count")
        ax.set_title("Event count distribution: dataset vs dictionary")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_figure(fig, PLOTS_DIR / "event_count_histogram.png")
        plt.close(fig)

    # ── 5. Dictionary coverage (NN spacing + radius-based filling) ──
    coverage_cols = [c for c in plot_cols if c in dictionary.columns]
    coverage_pairs = list(combinations(coverage_cols, 2))
    if not coverage_pairs and "flux_cm2_min" in dictionary.columns and "eff_sim_1" in dictionary.columns:
        coverage_pairs = [("flux_cm2_min", "eff_sim_1")]
    if not coverage_pairs:
        log.warning("No valid parameter pairs for dictionary coverage diagnostics.")
    for x_col, y_col in coverage_pairs:
        sx = _sanitize_plot_token(x_col)
        sy = _sanitize_plot_token(y_col)
        _plot_dictionary_coverage(
            dictionary,
            PLOTS_DIR / f"dictionary_coverage_{sx}__{sy}.png",
            x_col=x_col,
            y_col=y_col,
        )

    # ── 6. Efficiency sim vs empirical (2×2 scatter) ─────────────────
    _plot_eff_sim_vs_empirical(
        dataset,
        dictionary,
        PLOTS_DIR / "scatter_eff_sim_vs_estimated.png",
        fit_poly_by_plane=fit_poly_by_plane,
    )

    # ── 7. Empirical efficiency 2 vs 3 (single-panel scatter) ──────
    _plot_empirical_eff2_vs_eff3(
        dataset,
        dictionary,
        PLOTS_DIR / "scatter_empirical_eff2_vs_eff3.png",
    )

    # ── 8. Iso-rate contour in flux–eff space ────────────────────────
    _plot_iso_rate(
        dictionary,
        PLOTS_DIR / "iso_rate_global_rate.png",
        eff_band_tolerance_pct=iso_rate_eff_band_tolerance_pct,
    )

    # ── 9. Continuity validation ──────────────────────────────────
    if cv_metrics and cv_metrics.get("enabled", False):
        _plot_continuity_validation(
            dictionary,
            dataset,
            param_cols,
            cv_metrics,
            continuity_feature_cols,
            PLOTS_DIR / "continuity_validation.png",
        )

    # ── 10. Neighborhood mapping example (parameter ↔ feature) ───────
    if not dictionary.empty:
        _plot_continuity_neighborhood_example(
            dictionary,
            PLOTS_DIR / "continuity_neighborhood_example.png",
            param_cols=param_cols,
            feature_cols=continuity_feature_cols,
            cv_metrics=cv_metrics,
        )


if __name__ == "__main__":
    raise SystemExit(main())
