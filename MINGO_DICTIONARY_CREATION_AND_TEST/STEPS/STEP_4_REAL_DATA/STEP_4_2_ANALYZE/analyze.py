#!/usr/bin/env python3
"""STEP 4.2 - Run inference on real data and attach LUT uncertainties."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
import numpy as np
import pandas as pd

# -- Paths --------------------------------------------------------------------
STEP_DIR = Path(__file__).resolve().parent
# Support both layouts:
#   - <pipeline>/STEP_4_REAL_DATA/STEP_4_2_ANALYZE
#   - <pipeline>/STEPS/STEP_4_REAL_DATA/STEP_4_2_ANALYZE
if STEP_DIR.parents[2].name == "STEPS":
    PIPELINE_DIR = STEP_DIR.parents[3]
else:
    PIPELINE_DIR = STEP_DIR.parents[2]
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"

if (PIPELINE_DIR / "STEP_1_SETUP").exists() and (PIPELINE_DIR / "STEP_2_INFERENCE").exists():
    STEP_ROOT = PIPELINE_DIR
else:
    STEP_ROOT = PIPELINE_DIR / "STEPS"

INFERENCE_DIR = STEP_ROOT / "STEP_2_INFERENCE"

DEFAULT_REAL_COLLECTED = (
    STEP_DIR.parent
    / "STEP_4_1_COLLECT_REAL_DATA"
    / "OUTPUTS"
    / "FILES"
    / "real_collected_data.csv"
)
DEFAULT_DICTIONARY = (
    STEP_ROOT
    / "STEP_1_SETUP"
    / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS"
    / "FILES"
    / "dictionary.csv"
)
DEFAULT_LUT = (
    STEP_ROOT
    / "STEP_2_INFERENCE"
    / "STEP_2_3_UNCERTAINTY"
    / "OUTPUTS"
    / "FILES"
    / "uncertainty_lut.csv"
)
DEFAULT_LUT_META = (
    STEP_ROOT
    / "STEP_2_INFERENCE"
    / "STEP_2_3_UNCERTAINTY"
    / "OUTPUTS"
    / "FILES"
    / "uncertainty_lut_meta.json"
)
DEFAULT_BUILD_SUMMARY = (
    STEP_ROOT
    / "STEP_1_SETUP"
    / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS"
    / "FILES"
    / "build_summary.json"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_EFF = PLOTS_DIR / "STEP_4_2_2_efficiency_vs_time.png"
PLOT_EFF2_RATE = PLOTS_DIR / "STEP_4_2_5_eff2_vs_global_rate.png"
PLOT_EST_CURVE = PLOTS_DIR / "STEP_4_2_6_estimated_curve_flux_vs_eff.png"
PLOT_RECOVERY_STORY = PLOTS_DIR / "STEP_4_2_7_flux_recovery_vs_global_rate.png"

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

logging.basicConfig(format="[%(levelname)s] STEP_4.2 - %(message)s", level=logging.INFO)
log = logging.getLogger("STEP_4.2")

# Import estimator directly from STEP_2_INFERENCE.
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
try:
    from estimate_parameters import estimate_from_dataframes  # noqa: E402
except Exception as exc:
    log.error("Could not import estimate_from_dataframes from %s: %s", INFERENCE_DIR, exc)
    raise


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
    runtime_path = path.with_name("config_runtime.json")
    if runtime_path.exists():
        runtime_cfg = json.loads(runtime_path.read_text(encoding="utf-8"))
        cfg = _merge_dicts(cfg, runtime_cfg)
        log.info("Loaded runtime overrides: %s", runtime_path)
    return cfg


def _safe_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isfinite(out):
        return out
    return float(default)


def _safe_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


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


def _preferred_tt_prefixes_for_task_ids(task_ids: list[int]) -> list[str]:
    """Preferred TT-rate prefixes for efficiency extraction by most advanced task."""
    max_task_id = max(task_ids) if task_ids else 1
    if max_task_id <= 1:
        return ["raw"]
    if max_task_id == 2:
        return ["clean", "raw_to_clean", "raw"]
    if max_task_id == 3:
        return ["cal", "clean", "raw_to_clean", "raw"]
    if max_task_id == 4:
        return ["list", "list_to_fit", "cal", "clean", "raw_to_clean", "raw"]
    return [
        "corr",
        "task5_to_corr",
        "fit_to_corr",
        "definitive",
        "fit",
        "list_to_fit",
        "list",
        "cal",
        "clean",
        "raw_to_clean",
        "raw",
    ]


def _coalesce(primary: object, fallback: object) -> object:
    if primary in (None, "", "null", "None"):
        return fallback
    return primary


def _resolve_input_path(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    candidate_pipeline = PIPELINE_DIR / p
    if candidate_pipeline.exists():
        return candidate_pipeline
    candidate_step = STEP_DIR / p
    if candidate_step.exists():
        return candidate_step
    return candidate_pipeline


def _parse_ts(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    parsed = pd.to_datetime(s, format="%Y-%m-%d_%H.%M.%S", errors="coerce", utc=True)
    missing = parsed.isna()
    if missing.any():
        parsed_fallback = pd.to_datetime(s[missing], errors="coerce", utc=True)
        parsed.loc[missing] = parsed_fallback
    return parsed


def _extract_tt_parts(col: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<prefix>.+?)_tt_(?P<rest>.+)_rate_hz$", col)
    if match is None:
        return None
    rest = match.group("rest")
    # Some task outputs use names like tt_1234.0_rate_hz; normalize to tt_1234_rate_hz.
    rest = re.sub(r"\.0$", "", rest)
    return (match.group("prefix"), f"tt_{rest}_rate_hz")


def _tt_rate_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if re.search(r"_tt_.+_rate_hz$", c)])


def _prefix_rank(prefix: str) -> int:
    order = [
        "raw",
        "clean",
        "cal",
        "list",
        "fit",
        "corr",
        "definitive",
        "raw_to_clean",
        "list_to_fit",
        "fit_to_corr",
        "task5_to_corr",
    ]
    try:
        return order.index(prefix)
    except ValueError:
        return len(order)


def _choose_best_col(columns: list[str]) -> str:
    scored: list[tuple[int, str]] = []
    for col in columns:
        parts = _extract_tt_parts(col)
        rank = _prefix_rank(parts[0]) if parts is not None else 999
        scored.append((rank, col))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][1]


def _resolve_feature_columns_auto(
    dict_df: pd.DataFrame,
    real_df: pd.DataFrame,
    include_global_rate: bool,
    global_rate_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, list[dict[str, str]]]:
    """Build a robust feature set even when task prefixes differ."""
    dict_tt = _tt_rate_columns(dict_df)
    real_tt = _tt_rate_columns(real_df)

    feature_mapping: list[dict[str, str]] = []

    # 1) Prefer same-prefix direct intersections (mirrors STEP 2.1 behavior).
    prefixes = ["raw", "clean", "cal", "list", "fit", "corr", "definitive"]
    for prefix in prefixes:
        common = sorted(
            [c for c in dict_tt if c.startswith(f"{prefix}_tt_") and c in set(real_tt)]
        )
        if common:
            features = common.copy()
            if (
                include_global_rate
                and global_rate_col in dict_df.columns
                and global_rate_col in real_df.columns
                and global_rate_col not in features
            ):
                features.append(global_rate_col)
            return (
                dict_df,
                real_df,
                features,
                f"direct_prefix:{prefix}",
                feature_mapping,
            )

    # 2) Any exact common tt-rate columns.
    exact_common = sorted(set(dict_tt) & set(real_tt))
    if exact_common:
        features = exact_common.copy()
        if (
            include_global_rate
            and global_rate_col in dict_df.columns
            and global_rate_col in real_df.columns
            and global_rate_col not in features
        ):
            features.append(global_rate_col)
        return (dict_df, real_df, features, "direct_exact", feature_mapping)

    # 3) Align by tt topology key and create temporary aliases.
    dict_key_to_cols: dict[str, list[str]] = {}
    real_key_to_cols: dict[str, list[str]] = {}

    for col in dict_tt:
        parts = _extract_tt_parts(col)
        if parts is None:
            continue
        key = parts[1]
        dict_key_to_cols.setdefault(key, []).append(col)

    for col in real_tt:
        parts = _extract_tt_parts(col)
        if parts is None:
            continue
        key = parts[1]
        real_key_to_cols.setdefault(key, []).append(col)

    common_keys = sorted(set(dict_key_to_cols) & set(real_key_to_cols))
    if not common_keys:
        raise ValueError(
            "No compatible *_tt_*_rate_hz features found between dictionary and real data."
        )

    dict_work = dict_df.copy()
    real_work = real_df.copy()
    features: list[str] = []
    for idx, key in enumerate(common_keys):
        dcol = _choose_best_col(dict_key_to_cols[key])
        rcol = _choose_best_col(real_key_to_cols[key])
        alias = f"tt_feature_{idx:03d}_rate_hz"
        dict_work[alias] = pd.to_numeric(dict_work[dcol], errors="coerce")
        real_work[alias] = pd.to_numeric(real_work[rcol], errors="coerce")
        features.append(alias)
        feature_mapping.append(
            {
                "feature_alias": alias,
                "dictionary_column": dcol,
                "real_column": rcol,
                "tt_key": key,
            }
        )

    if include_global_rate and global_rate_col in dict_df.columns and global_rate_col in real_df.columns:
        alias = "global_rate_feature_hz"
        dict_work[alias] = pd.to_numeric(dict_work[global_rate_col], errors="coerce")
        real_work[alias] = pd.to_numeric(real_work[global_rate_col], errors="coerce")
        features.append(alias)
        feature_mapping.append(
            {
                "feature_alias": alias,
                "dictionary_column": global_rate_col,
                "real_column": global_rate_col,
                "tt_key": "global_rate",
            }
        )

    return (dict_work, real_work, features, "aligned_by_tt_key", feature_mapping)


def _pick_n_events_column(df: pd.DataFrame) -> str | None:
    priority = [
        "n_events",
        "selected_rows",
        "requested_rows",
        "raw_tt_1234_count",
        "clean_tt_1234_count",
        "list_tt_1234_count",
        "fit_tt_1234_count",
        "corr_tt_1234_count",
        "definitive_tt_1234_count",
        "events_per_second_total_seconds",
    ]
    for col in priority:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return col

    patt = re.compile(r"_tt_1234(?:\.0)?_count$")
    for col in sorted([c for c in df.columns if patt.search(c)]):
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            return col
    return None


def _pick_global_rate_column(df: pd.DataFrame, preferred: str = "events_per_second_global_rate") -> str | None:
    if preferred in df.columns:
        vals = pd.to_numeric(df[preferred], errors="coerce")
        if vals.notna().any():
            return preferred

    candidates = []
    for c in df.columns:
        cl = c.lower()
        if "global_rate" in cl and ("hz" in cl or cl.endswith("_rate")):
            candidates.append(c)
    for c in sorted(candidates):
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            return c
    return None


def _find_tt_rate_column(
    df: pd.DataFrame,
    tt_code: str,
    preferred_prefixes: list[str] | None = None,
) -> str | None:
    pattern = re.compile(rf"_tt_{re.escape(tt_code)}(?:\.0)?_rate_hz$")
    candidates = [c for c in df.columns if pattern.search(c)]
    if not candidates:
        return None
    preferred_order: dict[str, int] = {}
    if preferred_prefixes:
        preferred_order = {str(p): i for i, p in enumerate(preferred_prefixes)}

    def _sort_key(col: str) -> tuple[int, int, str]:
        parts = _extract_tt_parts(col)
        prefix = parts[0] if parts is not None else ""
        pref_rank = preferred_order.get(prefix, len(preferred_order) + 100)
        base_rank = _prefix_rank(prefix) if parts is not None else 999
        return (pref_rank, base_rank, col)

    candidates = sorted(candidates, key=_sort_key)
    for c in candidates:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            return c
    return candidates[0]


def _compute_eff_from_rates(
    df: pd.DataFrame,
    *,
    col_missing_rate: str | None,
    col_1234_rate: str | None,
) -> tuple[pd.Series, str]:
    if col_missing_rate is None or col_1234_rate is None:
        return (pd.Series(np.nan, index=df.index), "missing_rate_columns")

    r_miss = pd.to_numeric(df[col_missing_rate], errors="coerce")
    r_1234 = pd.to_numeric(df[col_1234_rate], errors="coerce")
    denom = r_1234.replace({0.0: np.nan})
    eff = 1.0 - (r_miss / denom)
    eff = eff.where(np.isfinite(eff), np.nan)
    return (eff, f"1 - {col_missing_rate}/{col_1234_rate}")


def _compute_empirical_efficiencies_from_rates(
    df: pd.DataFrame,
) -> tuple[dict[int, pd.Series], dict[int, str], dict[int, dict[str, str | None]], str | None]:
    """Compute plane efficiencies from 1 - (three-plane / four-plane) using TT rates."""
    preferred_prefixes: list[str] | None = None
    if isinstance(df.attrs.get("preferred_tt_prefixes"), list):
        preferred_prefixes = [str(v) for v in df.attrs.get("preferred_tt_prefixes", [])]

    four_col = _find_tt_rate_column(df, "1234", preferred_prefixes=preferred_prefixes)
    miss_by_plane = {1: "234", 2: "134", 3: "124", 4: "123"}

    selected_prefix: str | None = None
    four_parts = _extract_tt_parts(four_col) if four_col is not None else None
    if four_parts is not None:
        selected_prefix = four_parts[0]

    eff_by_plane: dict[int, pd.Series] = {}
    formula_by_plane: dict[int, str] = {}
    cols_by_plane: dict[int, dict[str, str | None]] = {}
    for plane, miss_code in miss_by_plane.items():
        miss_col = _find_tt_rate_column(df, miss_code, preferred_prefixes=preferred_prefixes)
        eff, formula = _compute_eff_from_rates(
            df,
            col_missing_rate=miss_col,
            col_1234_rate=four_col,
        )
        eff_by_plane[plane] = eff
        formula_by_plane[plane] = formula
        cols_by_plane[plane] = {
            "three_plane_col": miss_col,
            "four_plane_col": four_col,
        }
    return (eff_by_plane, formula_by_plane, cols_by_plane, selected_prefix)


def _format_polynomial_expr(
    coeffs: np.ndarray | list[float] | tuple[float, ...],
    *,
    variable: str = "x",
    precision: int = 6,
) -> str:
    """Return compact polynomial expression from highest to lowest degree."""
    arr = np.asarray(coeffs, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return "invalid"
    degree = arr.size - 1
    parts: list[tuple[str, str]] = []
    for idx, coef in enumerate(arr):
        if abs(float(coef)) < 1e-14:
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


def _invert_polynomial_values(
    y_values: pd.Series,
    coeffs: np.ndarray,
) -> pd.Series:
    """Solve P(x)=y row-wise and select a physically meaningful real root.

    Primary selection uses real roots within [0, 1] (efficiency domain), choosing
    the one closest to y as a stable tie-break. If no in-range root exists,
    fallback to all real roots and pick the closest-to-y root.
    """
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    out = np.full(y.shape, np.nan, dtype=float)
    degree = int(len(coeffs) - 1)

    x_grid = np.linspace(0.0, 1.0, 4001, dtype=float)
    y_grid = np.polyval(coeffs, x_grid)

    if degree == 1:
        a, b = float(coeffs[0]), float(coeffs[1])
        if np.isfinite(a) and abs(a) >= 1e-12:
            out = (y - b) / a
            out[~np.isfinite(out)] = np.nan
            out = np.clip(out, 0.0, 1.0)
        return pd.Series(out, index=y_values.index)

    for idx, target in enumerate(y):
        if not np.isfinite(target):
            continue
        coeff_eq = coeffs.copy()
        coeff_eq[-1] -= float(target)
        try:
            roots = np.roots(coeff_eq)
        except Exception:
            continue
        if roots.size == 0:
            continue
        real_mask = np.isfinite(roots.real) & np.isfinite(roots.imag) & (np.abs(roots.imag) < 1e-8)
        real_roots = roots.real[real_mask]
        if real_roots.size == 0:
            idx_best = int(np.argmin(np.abs(y_grid - target)))
            out[idx] = float(x_grid[idx_best])
            continue
        physical_roots = real_roots[(real_roots >= 0.0) & (real_roots <= 1.0)]
        if physical_roots.size > 0:
            out[idx] = float(physical_roots[np.argmin(np.abs(physical_roots - target))])
            continue
        idx_best = int(np.argmin(np.abs(y_grid - target)))
        out[idx] = float(x_grid[idx_best])
    return pd.Series(out, index=y_values.index)


def _load_eff_fit_lines(summary_path: Path) -> tuple[dict[int, list[float]], str]:
    """Load fit coefficients from STEP 1.2 build_summary.json (fit_line_eff_i = coeff list)."""
    if not summary_path.exists():
        return ({}, f"missing:{summary_path}")
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ({}, f"invalid_json:{exc}")

    out: dict[int, list[float]] = {}
    for plane in (1, 2, 3, 4):
        raw = payload.get(f"fit_line_eff_{plane}")
        if raw is None:
            raw = payload.get(f"fit_poly_eff_{plane}")
        if isinstance(raw, dict):
            if "coefficients" in raw:
                raw = raw.get("coefficients")
            elif "a" in raw and "b" in raw:
                raw = [raw.get("a"), raw.get("b")]
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        coeffs: list[float] = []
        valid = True
        for value in raw:
            c = _safe_float(value, np.nan)
            if not np.isfinite(c):
                valid = False
                break
            coeffs.append(float(c))
        if valid and len(coeffs) >= 2:
            out[plane] = coeffs
    return (out, "ok")


def _read_fit_order_info(summary_path: Path) -> tuple[int | None, dict[str, int]]:
    if not summary_path.exists():
        return (None, {})
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return (None, {})
    requested_raw = payload.get("fit_polynomial_order_requested")
    requested: int | None = None
    try:
        if requested_raw is not None:
            requested = int(requested_raw)
    except (TypeError, ValueError):
        requested = None

    used_by_plane: dict[str, int] = {}
    raw_used = payload.get("fit_polynomial_order_by_plane", {})
    if isinstance(raw_used, dict):
        for k, v in raw_used.items():
            try:
                used_by_plane[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
    return (requested, used_by_plane)


def _transform_efficiencies_with_fits(
    eff_by_plane: dict[int, pd.Series],
    fit_by_plane: dict[int, list[float]],
    *,
    mode: str = "inverse",
) -> tuple[dict[int, pd.Series], dict[int, str]]:
    """Apply polynomial fit transform per plane.

    Fits are from STEP 1.2: empirical = P(simulated).
    mode='inverse' solves P(simulated_equivalent)=empirical.
    mode='forward' applies corrected = P(empirical).
    """
    transformed: dict[int, pd.Series] = {}
    formula: dict[int, str] = {}
    use_inverse = str(mode).strip().lower() != "forward"
    for plane in (1, 2, 3, 4):
        raw = pd.to_numeric(eff_by_plane.get(plane), errors="coerce")
        if plane not in fit_by_plane:
            transformed[plane] = pd.Series(np.nan, index=raw.index)
            formula[plane] = "missing_fit_polynomial"
            continue
        coeffs = np.asarray(fit_by_plane[plane], dtype=float)
        if coeffs.ndim != 1 or coeffs.size < 2 or not np.isfinite(coeffs).all():
            transformed[plane] = pd.Series(np.nan, index=raw.index)
            formula[plane] = "invalid_fit_polynomial"
            continue
        degree = int(coeffs.size - 1)
        poly_expr = _format_polynomial_expr(coeffs, variable="x", precision=8)
        if use_inverse:
            corr = _invert_polynomial_values(raw, coeffs)
            formula[plane] = f"inverse_root(P(x)=eff_raw), deg={degree}, P(x)={poly_expr}"
        else:
            corr = pd.Series(np.polyval(coeffs, raw.to_numpy(dtype=float)), index=raw.index)
            formula[plane] = f"P(eff_raw), deg={degree}, P(x)={poly_expr}"
        # Keep transformed efficiencies fully unconstrained (no clipping).
        corr = corr.where(np.isfinite(corr), np.nan)
        transformed[plane] = corr
    return (transformed, formula)


def _fraction_in_closed_interval(series: pd.Series, lo: float, hi: float) -> float:
    """Fraction of finite values inside [lo, hi]. Returns 0 when no finite values."""
    s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(s)
    if not np.any(m):
        return 0.0
    vals = s[m]
    return float(np.mean((vals >= float(lo)) & (vals <= float(hi))))


def _build_rate_model(
    *,
    flux: pd.Series,
    eff: pd.Series,
    rate: pd.Series,
) -> dict | None:
    x = pd.to_numeric(flux, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(eff, errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(rate, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(mask.sum()) < 10:
        return None
    x = x[mask]
    y = y[mask]
    z = z[mask]
    tri = None
    interp = None
    try:
        tri = Triangulation(x, y)
        interp = LinearTriInterpolator(tri, z)
    except Exception:
        tri = None
        interp = None
    return {
        "x": x,
        "y": y,
        "z": z,
        "tri": tri,
        "interp": interp,
        "flux_min": float(np.nanmin(x)),
        "flux_max": float(np.nanmax(x)),
        "eff_min": float(np.nanmin(y)),
        "eff_max": float(np.nanmax(y)),
    }


def _predict_rate(
    model: dict,
    flux_values: np.ndarray,
    eff_values: np.ndarray,
) -> np.ndarray:
    xq = np.asarray(flux_values, dtype=float)
    yq = np.asarray(eff_values, dtype=float)
    qx = xq.ravel()
    qy = yq.ravel()
    zq = np.full_like(qx, np.nan, dtype=float)

    interp = model.get("interp")
    if interp is not None:
        zi = interp(qx, qy)
        zq = np.asarray(np.ma.filled(zi, np.nan), dtype=float)

    missing = ~np.isfinite(zq)
    if missing.any():
        x = np.asarray(model["x"], dtype=float)
        y = np.asarray(model["y"], dtype=float)
        z = np.asarray(model["z"], dtype=float)
        qx_m = qx[missing]
        qy_m = qy[missing]
        out = np.empty(len(qx_m), dtype=float)
        chunk = 4096
        for s in range(0, len(qx_m), chunk):
            e = min(len(qx_m), s + chunk)
            dx = qx_m[s:e, None] - x[None, :]
            dy = qy_m[s:e, None] - y[None, :]
            idx = np.argmin(dx * dx + dy * dy, axis=1)
            out[s:e] = z[idx]
        zq[missing] = out
    return zq.reshape(xq.shape)


def _estimate_flux_from_eff_and_rate(
    model: dict,
    *,
    eff_values: pd.Series,
    rate_values: pd.Series,
    n_flux_grid: int = 480,
) -> pd.Series:
    eff = pd.to_numeric(eff_values, errors="coerce")
    rate = pd.to_numeric(rate_values, errors="coerce")
    out = pd.Series(np.nan, index=eff.index, dtype=float)
    flux_grid = np.linspace(model["flux_min"], model["flux_max"], max(80, int(n_flux_grid)), dtype=float)
    for i in eff.index:
        e_val = float(eff.loc[i]) if np.isfinite(eff.loc[i]) else np.nan
        r_val = float(rate.loc[i]) if np.isfinite(rate.loc[i]) else np.nan
        if not (np.isfinite(e_val) and np.isfinite(r_val)):
            continue
        pred = _predict_rate(
            model,
            flux_values=flux_grid,
            eff_values=np.full_like(flux_grid, e_val, dtype=float),
        )
        valid = np.isfinite(pred)
        if not np.any(valid):
            continue
        pred_v = np.asarray(pred[valid], dtype=float)
        flux_v = np.asarray(flux_grid[valid], dtype=float)
        diff = pred_v - r_val

        # Prefer local linear interpolation between neighboring flux samples
        # that bracket the target rate. Fall back to nearest sample.
        nn_idx = int(np.argmin(np.abs(diff)))
        nn_flux = float(flux_v[nn_idx])
        best_flux = np.nan
        best_score = (np.inf, np.inf)

        cross_idx = np.where((diff[:-1] <= 0.0) & (diff[1:] >= 0.0) | ((diff[:-1] >= 0.0) & (diff[1:] <= 0.0)))[0]
        for j in cross_idx:
            p0 = float(pred_v[j])
            p1 = float(pred_v[j + 1])
            f0 = float(flux_v[j])
            f1 = float(flux_v[j + 1])
            if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(f0) and np.isfinite(f1)):
                continue
            den = p1 - p0
            if abs(den) <= 1e-12:
                f_interp = 0.5 * (f0 + f1)
                interp_err = min(abs(p0 - r_val), abs(p1 - r_val))
            else:
                t = (r_val - p0) / den
                t = float(np.clip(t, 0.0, 1.0))
                f_interp = f0 + t * (f1 - f0)
                p_interp = p0 + t * den
                interp_err = abs(p_interp - r_val)
            score = (float(interp_err), abs(f_interp - nn_flux))
            if score < best_score:
                best_score = score
                best_flux = f_interp

        if np.isfinite(best_flux):
            out.loc[i] = float(best_flux)
        else:
            out.loc[i] = nn_flux
    return out


def _ordered_row_indices(df: pd.DataFrame, valid_mask: pd.Series) -> np.ndarray:
    valid_idx = np.where(valid_mask.to_numpy(dtype=bool))[0]
    if len(valid_idx) == 0:
        return valid_idx
    if "file_timestamp_utc" in df.columns:
        ts = pd.to_datetime(df.loc[valid_mask, "file_timestamp_utc"], errors="coerce", utc=True)
        if ts.notna().any():
            ts_ns = ts.astype("int64", copy=False).to_numpy(dtype=np.int64, copy=False)
            ts_ns = np.where(ts.notna().to_numpy(), ts_ns, np.iinfo(np.int64).max)
            return valid_idx[np.argsort(ts_ns)]
    if "execution_timestamp_utc" in df.columns:
        ts = pd.to_datetime(df.loc[valid_mask, "execution_timestamp_utc"], errors="coerce", utc=True)
        if ts.notna().any():
            ts_ns = ts.astype("int64", copy=False).to_numpy(dtype=np.int64, copy=False)
            ts_ns = np.where(ts.notna().to_numpy(), ts_ns, np.iinfo(np.int64).max)
            return valid_idx[np.argsort(ts_ns)]
    return valid_idx


def _pick_estimated_eff_col_for_plane(df: pd.DataFrame, plane: int = 2) -> str | None:
    preferred = f"est_eff_sim_{int(plane)}"
    if preferred in df.columns and pd.to_numeric(df[preferred], errors="coerce").notna().any():
        return preferred
    return _choose_primary_eff_est_col(df)


def _pick_dictionary_eff_col_for_plane(df: pd.DataFrame, plane: int = 2) -> str | None:
    preferred = f"eff_sim_{int(plane)}"
    if preferred in df.columns and pd.to_numeric(df[preferred], errors="coerce").notna().any():
        return preferred
    for c in ("eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"):
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def _load_lut(lut_path: Path) -> pd.DataFrame:
    return pd.read_csv(lut_path, comment="#", low_memory=False)


def _lut_param_names(lut_df: pd.DataFrame, lut_meta_path: Path | None = None) -> list[str]:
    if lut_meta_path is not None and lut_meta_path.exists():
        try:
            meta = json.loads(lut_meta_path.read_text(encoding="utf-8"))
            params = meta.get("param_names", [])
            if isinstance(params, list):
                cleaned = [str(p) for p in params if str(p)]
                if cleaned:
                    return cleaned
        except Exception as exc:
            log.warning(
                "Could not parse LUT metadata at %s (%s). Falling back to column scan.",
                lut_meta_path,
                exc,
            )

    params: list[str] = []
    for c in lut_df.columns:
        if not c.startswith("sigma_"):
            continue
        if "_p" in c:
            pname = c[len("sigma_") :].split("_p", 1)[0]
        elif c.endswith("_std"):
            pname = c[len("sigma_") : -len("_std")]
        else:
            continue
        if pname and pname not in params:
            params.append(pname)
    return params


def _interpolate_uncertainties(
    query_df: pd.DataFrame,
    lut_df: pd.DataFrame,
    param_names: list[str],
    quantile: float,
) -> pd.DataFrame:
    if lut_df.empty or query_df.empty:
        return pd.DataFrame(index=query_df.index)

    q_label = str(int(round(float(quantile) * 100.0)))
    centre_cols = [c for c in lut_df.columns if c.endswith("_centre")]
    if not centre_cols:
        return pd.DataFrame(index=query_df.index)

    lut_centres_df = lut_df[centre_cols].apply(pd.to_numeric, errors="coerce")
    lut_centres = lut_centres_df.to_numpy(dtype=float)
    valid_centres = np.all(np.isfinite(lut_centres), axis=1)
    if not np.any(valid_centres):
        return pd.DataFrame(index=query_df.index)

    mins = np.nanmin(lut_centres[valid_centres], axis=0)
    maxs = np.nanmax(lut_centres[valid_centres], axis=0)
    ranges = maxs - mins
    ranges[~np.isfinite(ranges) | (ranges <= 0.0)] = 1.0
    dim_fallbacks = np.nanmedian(lut_centres[valid_centres], axis=0)

    n_rows = len(query_df)
    n_dims = len(centre_cols)
    query_vals = np.zeros((n_rows, n_dims), dtype=float)
    for j, cc in enumerate(centre_cols):
        dim = cc.replace("_centre", "")
        if dim in query_df.columns:
            qv = pd.to_numeric(query_df[dim], errors="coerce").to_numpy(dtype=float)
        elif dim == "n_events":
            qv = pd.to_numeric(query_df.get("n_events"), errors="coerce").to_numpy(dtype=float)
        else:
            qv = np.full(n_rows, np.nan, dtype=float)
        qv = np.where(np.isfinite(qv), qv, dim_fallbacks[j])
        query_vals[:, j] = qv

    d = (lut_centres[np.newaxis, :, :] - query_vals[:, np.newaxis, :]) / ranges[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    dist = np.where(valid_centres[np.newaxis, :], dist, np.inf)

    out = pd.DataFrame(index=query_df.index)
    for pname in param_names:
        pref_col = f"sigma_{pname}_p{q_label}"
        sigma_col = pref_col if pref_col in lut_df.columns else None
        if sigma_col is None:
            alt = f"sigma_{pname}_std"
            sigma_col = alt if alt in lut_df.columns else None
        if sigma_col is None:
            out[f"unc_{pname}_pct_raw"] = np.nan
            out[f"unc_{pname}_pct"] = np.nan
            continue

        sigma_vals = pd.to_numeric(lut_df[sigma_col], errors="coerce").to_numpy(dtype=float)
        valid_sigma = valid_centres & np.isfinite(sigma_vals)
        if not np.any(valid_sigma):
            out[f"unc_{pname}_pct_raw"] = np.nan
            out[f"unc_{pname}_pct"] = np.nan
            continue

        masked_dist = np.where(valid_sigma[np.newaxis, :], dist, np.inf)
        nearest_idx = np.argmin(masked_dist, axis=1)
        nearest_dist = masked_dist[np.arange(n_rows), nearest_idx]
        raw = sigma_vals[nearest_idx]
        raw = np.where(np.isfinite(nearest_dist), raw, np.nan)

        sigma_median = float(np.nanmedian(sigma_vals[valid_sigma]))
        raw = np.where(np.isfinite(raw), raw, sigma_median)
        out[f"unc_{pname}_pct_raw"] = raw
        out[f"unc_{pname}_pct"] = np.abs(raw)
    return out


def _choose_primary_eff_est_col(df: pd.DataFrame) -> str | None:
    for candidate in ("est_eff_sim_1", "est_eff_sim_2", "est_eff_sim_3", "est_eff_sim_4"):
        if candidate in df.columns:
            return candidate
    generic = [c for c in df.columns if c.startswith("est_eff_")]
    return sorted(generic)[0] if generic else None


def _time_axis(df: pd.DataFrame) -> tuple[pd.Series, str, bool]:
    if "file_timestamp_utc" in df.columns:
        ts = _parse_ts(df["file_timestamp_utc"])
        if ts.notna().any():
            return (ts.dt.tz_convert(None), "Data time from filename_base [UTC]", True)
    if "execution_timestamp_utc" in df.columns:
        ts = _parse_ts(df["execution_timestamp_utc"])
        if ts.notna().any():
            return (ts.dt.tz_convert(None), "Execution time [UTC]", True)
    if "execution_timestamp" in df.columns:
        ts = _parse_ts(df["execution_timestamp"])
        if ts.notna().any():
            return (ts.dt.tz_convert(None), "Execution time [UTC]", True)
    return (pd.Series(np.arange(len(df), dtype=float), index=df.index), "Row index", False)


def _plot_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.7))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_series_with_uncertainty(
    *,
    x: pd.Series,
    has_time_axis: bool,
    y: pd.Series,
    y_unc: pd.Series | None,
    title: str,
    ylabel: str,
    xlabel: str,
    out_path: Path,
) -> None:
    yv = pd.to_numeric(y, errors="coerce")
    if yv.notna().sum() == 0:
        _plot_placeholder(out_path, title, f"No finite values found for '{ylabel}'.")
        return

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    x_values = x.to_numpy()
    y_values = yv.to_numpy(dtype=float)

    ax.plot(x_values, y_values, color="#1F77B4", linewidth=1.2, alpha=0.9, marker="o", markersize=2.4)

    if y_unc is not None:
        uv = pd.to_numeric(y_unc, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y_values) & np.isfinite(uv)
        if finite.any():
            lower = y_values[finite] - np.abs(uv[finite])
            upper = y_values[finite] + np.abs(uv[finite])
            ax.fill_between(
                x_values[finite],
                lower,
                upper,
                color="#1F77B4",
                alpha=0.16,
                linewidth=0.0,
                label="Estimate +/- uncertainty",
            )
            ax.legend(loc="best", frameon=True, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)
    if not has_time_axis:
        ax.set_xlim(float(np.nanmin(x_values)), float(np.nanmax(x_values)) if len(x_values) > 1 else 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pre_estimation_efficiencies(
    *,
    x: pd.Series,
    has_time_axis: bool,
    xlabel: str,
    global_rate: pd.Series | None,
    global_rate_label: str,
    raw_eff_by_plane: dict[int, pd.Series],
    raw_eff_source_prefix: str | None,
    transformed_eff_by_plane: dict[int, pd.Series],
    transform_mode: str,
    out_path: Path,
) -> None:
    """Three-panel diagnostic:
    top: global-rate only,
    middle: raw eff(1..4) from 1 - three/four,
    bottom: transformed efficiencies using STEP 1.2 fit polynomials.
    """
    raw_valid = any(pd.to_numeric(raw_eff_by_plane.get(p), errors="coerce").notna().any() for p in (1, 2, 3, 4))
    tr_valid = any(
        pd.to_numeric(transformed_eff_by_plane.get(p), errors="coerce").notna().any() for p in (1, 2, 3, 4)
    )
    gr_valid = False
    gr = pd.Series(np.nan, index=x.index, dtype=float)
    if global_rate is not None:
        gr = pd.to_numeric(global_rate, errors="coerce")
        gr_valid = gr.notna().any()

    if not (raw_valid or tr_valid or gr_valid):
        _plot_placeholder(
            out_path,
            "Pre-estimation efficiency diagnostics",
            "No finite global-rate or source/transformed efficiency values available.",
        )
        return

    source_prefix_label = (
        f"{str(raw_eff_source_prefix)}_tt"
        if raw_eff_source_prefix not in (None, "", "None")
        else "selected_tt"
    )

    def _eff_ylim(vmin: float, vmax: float) -> tuple[float, float]:
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            return (-0.02, 1.02)
        lo = min(-0.02, float(vmin))
        hi = max(1.02, float(vmax))
        if hi <= lo:
            hi = lo + 0.05
        pad = 0.03 * (hi - lo)
        return (lo - pad, hi + pad)

    xv = x.to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 11.2), sharex=True)

    # Top panel: global rate only
    ax_rate = axes[0]
    color_by_plane = {1: "#1F77B4", 2: "#FF7F0E", 3: "#2CA02C", 4: "#9467BD"}
    if gr_valid:
        ax_rate.scatter(
            xv,
            gr.to_numpy(dtype=float),
            color="#111111",
            s=9,
            alpha=0.85,
            label=f"{global_rate_label} [Hz]",
        )
        ax_rate.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax_rate.text(0.5, 0.5, "No finite global-rate values", ha="center", va="center")
    ax_rate.set_ylabel("Global rate [Hz]")
    ax_rate.set_title("Before dictionary estimation: global_rate_hz")
    ax_rate.grid(True, alpha=0.22)

    # Middle panel: efficiencies from selected TT prefix
    ax_raw = axes[1]
    n_raw = 0
    raw_y_min = np.inf
    raw_y_max = -np.inf
    for plane in (1, 2, 3, 4):
        eff = pd.to_numeric(raw_eff_by_plane.get(plane), errors="coerce")
        valid = eff.notna()
        if valid.any():
            yvals = eff[valid].to_numpy(dtype=float)
            if yvals.size:
                raw_y_min = min(raw_y_min, float(np.nanmin(yvals)))
                raw_y_max = max(raw_y_max, float(np.nanmax(yvals)))
            ax_raw.scatter(
                xv[valid.to_numpy()],
                yvals,
                s=10,
                alpha=0.88,
                color=color_by_plane[plane],
                label=f"eff_{plane} ({source_prefix_label}) = 1 - three/four",
            )
            n_raw += 1
    if n_raw == 0:
        ax_raw.text(0.5, 0.5, "No raw efficiency values available", ha="center", va="center")
    else:
        ax_raw.legend(loc="best", fontsize=8, frameon=True, ncol=2)
    ax_raw.set_ylim(*_eff_ylim(raw_y_min, raw_y_max))
    ax_raw.set_ylabel("Source efficiencies")
    ax_raw.set_title(f"Efficiencies from {source_prefix_label} rates (1 - threeplane/fourplane)")
    ax_raw.grid(True, alpha=0.22)

    # Bottom panel: transformed efficiencies
    ax_bot = axes[2]
    n_drawn = 0
    tr_y_min = np.inf
    tr_y_max = -np.inf
    for plane in (1, 2, 3, 4):
        eff_t = pd.to_numeric(transformed_eff_by_plane.get(plane), errors="coerce")
        valid = eff_t.notna()
        if valid.any():
            yvals_t = eff_t[valid].to_numpy(dtype=float)
            if yvals_t.size:
                tr_y_min = min(tr_y_min, float(np.nanmin(yvals_t)))
                tr_y_max = max(tr_y_max, float(np.nanmax(yvals_t)))
            ax_bot.scatter(
                xv[valid.to_numpy()],
                yvals_t,
                s=10,
                alpha=0.9,
                color=color_by_plane[plane],
                label=f"transformed_eff_{plane}",
            )
            n_drawn += 1
    if n_drawn == 0:
        ax_bot.text(0.5, 0.5, "No transformed efficiency values available", ha="center", va="center")

    ax_bot.set_ylim(*_eff_ylim(tr_y_min, tr_y_max))
    ax_bot.set_ylabel("Transformed efficiencies")
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_title(
        "Transformed efficiency (using STEP 1.2 fit polynomials; "
        f"mode={transform_mode})"
    )
    ax_bot.grid(True, alpha=0.22)
    if n_drawn > 0:
        ax_bot.legend(loc="best", fontsize=8, frameon=True, ncol=2)

    if not has_time_axis and len(xv) > 0:
        xmin = float(np.nanmin(xv))
        xmax = float(np.nanmax(xv)) if len(xv) > 1 else xmin + 1.0
        ax_rate.set_xlim(xmin, xmax)
        ax_raw.set_xlim(xmin, xmax)
        ax_bot.set_xlim(xmin, xmax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_flux_recovery_story_real(
    *,
    x: pd.Series,
    has_time_axis: bool,
    xlabel: str,
    distance_series: pd.Series | None,
    distance_label: str,
    eff_series: pd.Series,
    eff_label: str,
    global_rate_series: pd.Series,
    global_rate_label: str,
    flux_est_series: pd.Series,
    flux_unc_series: pd.Series | None,
    flux_reference_series: pd.Series | None,
    flux_reference_label: str,
    out_path: Path,
) -> None:
    """STEP 3.3-like story using real-data inferred quantities, prefixed by best distance."""

    def _apply_striped_background(ax: plt.Axes, y_vals: np.ndarray) -> None:
        y_min, y_max = ax.get_ylim()
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            return
        span = y_max - y_min
        if span <= 0.0:
            return

        valid = np.isfinite(y_vals)
        if not np.any(valid):
            return
        mean_val = float(np.mean(y_vals[valid]))
        band = abs(mean_val) * 0.01
        if not np.isfinite(band) or band <= 0.0:
            band = span * 0.01
        if band <= 0.0:
            return

        ax.set_facecolor("#FFFFFF")
        idx = int(np.floor((y_min - mean_val) / band))
        y0 = mean_val + idx * band
        while y0 < y_max:
            y1 = y0 + band
            lo = max(y0, y_min)
            hi = min(y1, y_max)
            color = "#FFFFFF" if (idx % 2 == 0) else "#D8DDE4"
            if hi > lo:
                ax.axhspan(lo, hi, facecolor=color, alpha=1.0, linewidth=0.0, zorder=0)
            y0 = y1
            idx += 1
        ax.set_ylim(y_min, y_max)

    xv = x.to_numpy()
    distance = (
        pd.to_numeric(distance_series, errors="coerce").to_numpy(dtype=float)
        if distance_series is not None
        else np.full(len(xv), np.nan, dtype=float)
    )
    eff = pd.to_numeric(eff_series, errors="coerce").to_numpy(dtype=float)
    rate = pd.to_numeric(global_rate_series, errors="coerce").to_numpy(dtype=float)
    flux_est = pd.to_numeric(flux_est_series, errors="coerce").to_numpy(dtype=float)
    flux_unc = (
        pd.to_numeric(flux_unc_series, errors="coerce").to_numpy(dtype=float)
        if flux_unc_series is not None
        else None
    )
    flux_ref = (
        pd.to_numeric(flux_reference_series, errors="coerce").to_numpy(dtype=float)
        if flux_reference_series is not None
        else None
    )

    valid_any = (
        np.isfinite(distance).any()
        or np.isfinite(eff).any()
        or np.isfinite(rate).any()
        or np.isfinite(flux_est).any()
        or (flux_ref is not None and np.isfinite(flux_ref).any())
    )
    if not valid_any:
        _plot_placeholder(
            out_path,
            "Flux-recovery style real-data story",
            "No finite series available for best_distance / estimated efficiency / global rate / estimated flux.",
        )
        return

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.6, 9.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0, 1.15]},
    )
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, alpha=0.24)

    # 1) Best dictionary distance.
    m_dist = np.isfinite(xv) & np.isfinite(distance)
    if np.any(m_dist):
        order = np.argsort(xv[m_dist])
        xd = xv[m_dist][order]
        yd = distance[m_dist][order]
        axes[0].plot(
            xd,
            yd,
            color="#9467BD",
            linewidth=1.15,
            marker="o",
            markersize=2.8,
            markerfacecolor="#9467BD",
            markeredgewidth=0.0,
            alpha=0.88,
            label=f"Best dictionary distance ({distance_label})",
        )
    else:
        axes[0].text(0.5, 0.5, f"No finite values for {distance_label}", ha="center", va="center")
    axes[0].set_ylabel("Best distance")
    _apply_striped_background(axes[0], distance)
    axes[0].legend(loc="best", fontsize=8)

    # 2) Estimated efficiency.
    m_eff = np.isfinite(xv) & np.isfinite(eff)
    if np.any(m_eff):
        order = np.argsort(xv[m_eff])
        xe = xv[m_eff][order]
        ye = eff[m_eff][order]
        axes[1].plot(
            xe,
            ye,
            color="#FF7F0E",
            linewidth=1.15,
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.65,
            alpha=0.90,
            label=f"Estimated efficiency ({eff_label})",
        )
    else:
        axes[1].text(0.5, 0.5, f"No finite values for {eff_label}", ha="center", va="center")
    axes[1].set_ylabel("Estimated eff")
    _apply_striped_background(axes[1], eff)
    axes[1].legend(loc="best", fontsize=8)

    # 3) Global rate.
    m_rate = np.isfinite(xv) & np.isfinite(rate)
    if np.any(m_rate):
        order = np.argsort(xv[m_rate])
        xr = xv[m_rate][order]
        yr = rate[m_rate][order]
        axes[2].plot(
            xr,
            yr,
            color="#2E8B57",
            linewidth=2.4,
            alpha=0.46,
            solid_capstyle="round",
            label=f"Global rate ({global_rate_label})",
        )
    else:
        axes[2].text(0.5, 0.5, f"No finite values for {global_rate_label}", ha="center", va="center")
    axes[2].set_ylabel("Global rate")
    _apply_striped_background(axes[2], rate)
    axes[2].legend(loc="best", fontsize=8)

    # 4) Estimated flux (+ uncertainty), with optional real-data-derived reference.
    m_flux = np.isfinite(xv) & np.isfinite(flux_est)
    if np.any(m_flux):
        order = np.argsort(xv[m_flux])
        xf = xv[m_flux][order]
        yf = flux_est[m_flux][order]
        axes[3].plot(
            xf,
            yf,
            color="#D62728",
            linewidth=1.3,
            marker="o",
            markersize=3.0,
            markerfacecolor="#D62728",
            markeredgewidth=0.0,
            alpha=0.88,
            label="Estimated reconstructed flux",
            zorder=3,
        )
        if flux_unc is not None and len(flux_unc) == len(xv):
            uf = np.abs(np.asarray(flux_unc, dtype=float)[m_flux][order])
            valid_uf = np.isfinite(uf)
            if np.any(valid_uf):
                axes[3].fill_between(
                    xf[valid_uf],
                    yf[valid_uf] - uf[valid_uf],
                    yf[valid_uf] + uf[valid_uf],
                    color="#D62728",
                    alpha=0.16,
                    linewidth=0.0,
                    label="Estimated ± uncertainty",
                    zorder=2,
                )
    else:
        axes[3].text(0.5, 0.5, "No finite estimated flux values", ha="center", va="center")

    if flux_ref is not None and len(flux_ref) == len(xv):
        m_ref = np.isfinite(xv) & np.isfinite(flux_ref)
        if np.any(m_ref):
            order = np.argsort(xv[m_ref])
            xr = xv[m_ref][order]
            yr = flux_ref[m_ref][order]
            axes[3].plot(
                xr,
                yr,
                color="#1F77B4",
                linewidth=1.0,
                linestyle="--",
                alpha=0.62,
                label=flux_reference_label,
                zorder=1,
            )

    axes[3].set_ylabel("Estimated flux")
    axes[3].set_xlabel(xlabel)
    _apply_striped_background(
        axes[3],
        flux_est if np.isfinite(flux_est).any() else (flux_ref if flux_ref is not None else np.array([])),
    )
    axes[3].legend(loc="best", fontsize=8)

    if not has_time_axis and len(xv) > 0:
        xmin = float(np.nanmin(xv))
        xmax = float(np.nanmax(xv)) if len(xv) > 1 else xmin + 1.0
        for ax in axes:
            ax.set_xlim(xmin, xmax)

    fig.suptitle(
        "Real-data story: best distance -> estimated efficiency -> global-rate response -> reconstructed flux",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_eff2_vs_global_rate(
    *,
    real_df: pd.DataFrame,
    dict_df: pd.DataFrame,
    real_eff2_col: str,
    real_rate_col: str,
    dict_eff2_col: str,
    dict_rate_col: str,
    out_path: Path,
) -> tuple[int, int, str]:
    """Contour-derived (flux, eff2) trajectory from (eff2, global_rate), styled like STEP 4.2.6."""
    flux_col = "flux_cm2_min"
    if flux_col not in dict_df.columns:
        _plot_placeholder(
            out_path,
            "Eff2 contour placement in flux-eff plane",
            f"Dictionary missing required column '{flux_col}'.",
        )
        return (0, 0, "missing_flux_column")

    model = _build_rate_model(
        flux=dict_df[flux_col],
        eff=dict_df[dict_eff2_col],
        rate=dict_df[dict_rate_col],
    )
    if model is None:
        _plot_placeholder(
            out_path,
            "Eff2 contour placement in flux-eff plane",
            "Not enough finite dictionary points to build iso-rate contours.",
        )
        return (0, 0, "model_unavailable")

    real_eff = pd.to_numeric(real_df[real_eff2_col], errors="coerce")
    real_rate = pd.to_numeric(real_df[real_rate_col], errors="coerce")
    flux_from_contour_col = "flux_from_eff2_global_rate_hz"
    real_df[flux_from_contour_col] = _estimate_flux_from_eff_and_rate(
        model,
        eff_values=real_eff,
        rate_values=real_rate,
        n_flux_grid=520,
    )

    dict_flux = pd.to_numeric(dict_df[flux_col], errors="coerce")
    dict_eff = pd.to_numeric(dict_df[dict_eff2_col], errors="coerce")
    dict_rate = pd.to_numeric(dict_df[dict_rate_col], errors="coerce")
    dict_valid = dict_flux.notna() & dict_eff.notna() & dict_rate.notna()
    n_dict = int(dict_valid.sum())

    real_flux = pd.to_numeric(real_df[flux_from_contour_col], errors="coerce")
    real_valid = real_flux.notna() & real_eff.notna() & real_rate.notna()
    n_real = int(real_valid.sum())
    if n_real == 0:
        _plot_placeholder(
            out_path,
            "Eff2 contour placement in flux-eff plane",
            "No finite real points available after contour-based flux inversion.",
        )
        return (n_real, n_dict, flux_from_contour_col)

    # Contour grid spans dictionary and real flux/eff locations.
    x_ref = np.asarray(model["x"], dtype=float)
    y_ref = np.asarray(model["y"], dtype=float)
    x_real = real_flux[real_valid].to_numpy(dtype=float)
    y_real = real_eff[real_valid].to_numpy(dtype=float)
    x_all = np.concatenate([x_ref[np.isfinite(x_ref)], x_real[np.isfinite(x_real)]])
    y_all = np.concatenate([y_ref[np.isfinite(y_ref)], y_real[np.isfinite(y_real)]])
    x_lo = float(np.nanmin(x_all))
    x_hi = float(np.nanmax(x_all))
    y_lo = float(np.nanmin(y_all))
    y_hi = float(np.nanmax(y_all))
    x_span = max(x_hi - x_lo, 1e-9)
    y_span = max(y_hi - y_lo, 1e-9)
    x_lo -= 0.03 * x_span
    x_hi += 0.03 * x_span
    y_lo -= 0.03 * y_span
    y_hi += 0.03 * y_span

    xi = np.linspace(x_lo, x_hi, 230, dtype=float)
    yi = np.linspace(y_lo, y_hi, 230, dtype=float)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = _predict_rate(model, Xi, Yi)
    finite_z = Zi[np.isfinite(Zi)]

    fig, ax = plt.subplots(figsize=(9.2, 7.1))
    if finite_z.size >= 10:
        levels = np.linspace(float(np.nanmin(finite_z)), float(np.nanmax(finite_z)), 16)
        cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap="viridis", alpha=0.35, zorder=0)
        cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.048)
        cbar.set_label("Global rate [Hz]")
        ax.contour(Xi, Yi, Zi, levels=levels[::2], colors="k", linewidths=0.35, alpha=0.28, zorder=1)
    else:
        sc_fallback = ax.scatter(
            dict_flux[dict_valid],
            dict_eff[dict_valid],
            c=dict_rate[dict_valid],
            cmap="viridis",
            s=12,
            alpha=0.5,
            edgecolors="none",
            zorder=0,
        )
        cbar = fig.colorbar(sc_fallback, ax=ax, pad=0.02, fraction=0.048)
        cbar.set_label("Global rate [Hz]")

    # Reference dictionary points.
    ax.scatter(
        dict_flux[dict_valid],
        dict_eff[dict_valid],
        s=10,
        alpha=0.18,
        color="#606060",
        zorder=1,
        label="Dictionary points",
    )

    # Time-ordered contour-derived real trajectory.
    ordered_idx = _ordered_row_indices(real_df, real_valid)
    x_ord = real_df.iloc[ordered_idx][flux_from_contour_col].to_numpy(dtype=float)
    y_ord = real_df.iloc[ordered_idx][real_eff2_col].to_numpy(dtype=float)

    ax.plot(
        x_ord,
        y_ord,
        linewidth=1.8,
        color="#1F77B4",
        alpha=0.92,
        zorder=3,
        label="Contour-derived trajectory",
    )
    ax.scatter(
        x_ord,
        y_ord,
        s=24,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label="Contour-derived points",
    )

    ax.scatter(
        [x_ord[0]],
        [y_ord[0]],
        color="#2CA02C",
        marker="o",
        s=82,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="Start",
    )
    ax.scatter(
        [x_ord[-1]],
        [y_ord[-1]],
        color="#D62728",
        marker="X",
        s=95,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="End",
    )

    if len(x_ord) >= 3:
        i = min(len(x_ord) - 2, max(0, int(0.85 * len(x_ord))))
        ax.annotate(
            "",
            xy=(x_ord[i + 1], y_ord[i + 1]),
            xytext=(x_ord[i], y_ord[i]),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            zorder=5,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Flux [cm^-2 min^-1]")
    ax.set_ylabel("Efficiency (eff2)")
    ax.set_title(
        "Contour-derived real-data curve in flux-eff plane with global-rate contours\n"
        f"real_y={real_eff2_col} | dict_y={dict_eff2_col}"
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return (n_real, n_dict, flux_from_contour_col)


def _plot_estimated_curve_flux_vs_eff(
    *,
    real_df: pd.DataFrame,
    dict_df: pd.DataFrame,
    est_flux_col: str,
    est_eff_col: str,
    dict_eff_col: str,
    dict_rate_col: str,
    out_path: Path,
) -> tuple[int, int]:
    """Plot estimated (flux, eff) trajectory over dictionary global-rate contours."""
    flux_col = "flux_cm2_min"
    model = _build_rate_model(
        flux=dict_df[flux_col],
        eff=dict_df[dict_eff_col],
        rate=dict_df[dict_rate_col],
    )
    if model is None:
        _plot_placeholder(
            out_path,
            "Estimated curve in flux-eff plane",
            "Not enough finite dictionary points to build global-rate contours.",
        )
        return (0, 0)

    dict_flux = pd.to_numeric(dict_df[flux_col], errors="coerce")
    dict_eff = pd.to_numeric(dict_df[dict_eff_col], errors="coerce")
    dict_rate = pd.to_numeric(dict_df[dict_rate_col], errors="coerce")
    dict_valid = dict_flux.notna() & dict_eff.notna() & dict_rate.notna()
    n_dict = int(dict_valid.sum())

    real_flux = pd.to_numeric(real_df[est_flux_col], errors="coerce")
    real_eff = pd.to_numeric(real_df[est_eff_col], errors="coerce")
    real_valid = real_flux.notna() & real_eff.notna()
    n_real = int(real_valid.sum())
    if n_real == 0:
        _plot_placeholder(
            out_path,
            "Estimated curve in flux-eff plane",
            "No finite estimated (flux, eff) points available.",
        )
        return (n_real, n_dict)

    x_ref = np.asarray(model["x"], dtype=float)
    y_ref = np.asarray(model["y"], dtype=float)
    x_real = real_flux[real_valid].to_numpy(dtype=float)
    y_real = real_eff[real_valid].to_numpy(dtype=float)
    x_all = np.concatenate([x_ref[np.isfinite(x_ref)], x_real[np.isfinite(x_real)]])
    y_all = np.concatenate([y_ref[np.isfinite(y_ref)], y_real[np.isfinite(y_real)]])
    x_lo = float(np.nanmin(x_all))
    x_hi = float(np.nanmax(x_all))
    y_lo = float(np.nanmin(y_all))
    y_hi = float(np.nanmax(y_all))
    x_span = max(x_hi - x_lo, 1e-9)
    y_span = max(y_hi - y_lo, 1e-9)
    x_lo -= 0.03 * x_span
    x_hi += 0.03 * x_span
    y_lo -= 0.03 * y_span
    y_hi += 0.03 * y_span

    xi = np.linspace(x_lo, x_hi, 230, dtype=float)
    yi = np.linspace(y_lo, y_hi, 230, dtype=float)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = _predict_rate(model, Xi, Yi)
    finite_z = Zi[np.isfinite(Zi)]

    fig, ax = plt.subplots(figsize=(9.2, 7.1))
    if finite_z.size >= 10:
        levels = np.linspace(float(np.nanmin(finite_z)), float(np.nanmax(finite_z)), 16)
        cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap="viridis", alpha=0.35, zorder=0)
        cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.048)
        cbar.set_label("Global rate [Hz]")
        ax.contour(Xi, Yi, Zi, levels=levels[::2], colors="k", linewidths=0.35, alpha=0.28, zorder=1)
    else:
        sc_fallback = ax.scatter(
            dict_flux[dict_valid],
            dict_eff[dict_valid],
            c=dict_rate[dict_valid],
            cmap="viridis",
            s=12,
            alpha=0.5,
            edgecolors="none",
            zorder=0,
        )
        cbar = fig.colorbar(sc_fallback, ax=ax, pad=0.02, fraction=0.048)
        cbar.set_label("Global rate [Hz]")

    # Reference dictionary points.
    ax.scatter(
        dict_flux[dict_valid],
        dict_eff[dict_valid],
        s=10,
        alpha=0.18,
        color="#606060",
        zorder=1,
        label="Dictionary points",
    )

    order_idx = _ordered_row_indices(real_df, real_valid)
    x_ord = real_df.iloc[order_idx][est_flux_col].to_numpy(dtype=float)
    y_ord = real_df.iloc[order_idx][est_eff_col].to_numpy(dtype=float)

    ax.plot(
        x_ord,
        y_ord,
        linewidth=1.8,
        color="#1F77B4",
        alpha=0.92,
        zorder=3,
        label="Estimated trajectory",
    )
    ax.scatter(
        x_ord,
        y_ord,
        s=24,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label="Estimated points",
    )

    ax.scatter([x_ord[0]], [y_ord[0]], color="#2CA02C", marker="o", s=82, edgecolor="black", linewidth=0.8, zorder=5)
    ax.scatter([x_ord[-1]], [y_ord[-1]], color="#D62728", marker="X", s=95, edgecolor="black", linewidth=0.8, zorder=5)

    if len(x_ord) >= 3:
        i = min(len(x_ord) - 2, max(0, int(0.85 * len(x_ord))))
        ax.annotate(
            "",
            xy=(x_ord[i + 1], y_ord[i + 1]),
            xytext=(x_ord[i], y_ord[i]),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            zorder=5,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Flux [cm^-2 min^-1]")
    ax.set_ylabel(f"Estimated efficiency ({est_eff_col})")
    ax.set_title("Estimated real-data curve in flux-eff plane with global-rate contours")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return (n_real, n_dict)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 4.2: Infer real-data parameters and attach uncertainty LUT."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--real-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--lut-csv", default=None)
    parser.add_argument("--lut-meta-json", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_21 = config.get("step_2_1", {})
    cfg_41 = config.get("step_4_1", {})
    cfg_42 = config.get("step_4_2", {})
    _clear_plots_dir()

    real_csv_cfg = cfg_42.get("real_collected_csv", None)
    dictionary_csv_cfg = cfg_42.get("dictionary_csv", None)
    lut_csv_cfg = cfg_42.get("uncertainty_lut_csv", None)
    lut_meta_cfg = cfg_42.get("uncertainty_lut_meta_json", None)
    build_summary_cfg = cfg_42.get("build_summary_json", None)

    real_path = _resolve_input_path(args.real_csv or real_csv_cfg or DEFAULT_REAL_COLLECTED)
    dict_path = _resolve_input_path(args.dictionary_csv or dictionary_csv_cfg or DEFAULT_DICTIONARY)
    lut_path = _resolve_input_path(args.lut_csv or lut_csv_cfg or DEFAULT_LUT)
    lut_meta_path = _resolve_input_path(args.lut_meta_json or lut_meta_cfg or DEFAULT_LUT_META)
    build_summary_path = _resolve_input_path(build_summary_cfg or DEFAULT_BUILD_SUMMARY)

    for label, path in (
        ("Real collected CSV", real_path),
        ("Dictionary CSV", dict_path),
        ("Uncertainty LUT CSV", lut_path),
    ):
        if not path.exists():
            log.error("%s not found: %s", label, path)
            return 1

    # STEP 4.2 always inherits matching criteria from STEP 2.1 to avoid duplicate/overriding knobs.
    ignored_step42_criteria_keys = [
        "feature_columns",
        "distance_metric",
        "interpolation_k",
        "include_global_rate",
        "global_rate_col",
    ]
    for key in ignored_step42_criteria_keys:
        if cfg_42.get(key, None) not in (None, "", "null", "None"):
            log.info("Ignoring step_4_2.%s; using step_2_1.%s instead.", key, key)

    feature_columns_cfg = cfg_21.get("feature_columns", "auto")
    distance_metric = str(cfg_21.get("distance_metric", "l2_zscore"))
    interpolation_k_raw = cfg_21.get("interpolation_k", None)
    if interpolation_k_raw in (None, "", "null", "None"):
        interpolation_k: int | None = None
    else:
        interpolation_k = int(interpolation_k_raw)
    include_global_rate = _safe_bool(cfg_21.get("include_global_rate", True), True)
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate"))
    exclude_same_file = _safe_bool(cfg_42.get("exclude_same_file", False), False)
    uncertainty_quantile = _safe_float(cfg_42.get("uncertainty_quantile", 0.68), 0.68)
    uncertainty_quantile = float(np.clip(uncertainty_quantile, 0.0, 1.0))
    n_events_column_cfg = cfg_42.get("n_events_column", "auto")
    task_ids_cfg = cfg_41.get("task_ids", config.get("task_ids", [1]))
    selected_task_ids = _safe_task_ids(task_ids_cfg)
    preferred_tt_prefixes = _preferred_tt_prefixes_for_task_ids(selected_task_ids)
    eff_transform_mode = str(cfg_42.get("eff_transform_mode", "inverse")).strip().lower()
    if eff_transform_mode not in {"inverse", "forward"}:
        eff_transform_mode = "inverse"

    log.info("Real collected: %s", real_path)
    log.info("Dictionary:     %s", dict_path)
    log.info("LUT:            %s", lut_path)
    log.info("Fit summary:    %s", build_summary_path)
    log.info("Task IDs used for efficiency source: %s", selected_task_ids)
    log.info("Preferred TT prefix order for efficiencies: %s", preferred_tt_prefixes)
    log.info("Metric=%s, k=%s, uncertainty_quantile=%.3f", distance_metric, interpolation_k, uncertainty_quantile)

    real_df = pd.read_csv(real_path, low_memory=False)
    dict_df = pd.read_csv(dict_path, low_memory=False)
    if real_df.empty:
        log.error("Real collected table is empty: %s", real_path)
        return 1
    if dict_df.empty:
        log.error("Dictionary table is empty: %s", dict_path)
        return 1

    if isinstance(feature_columns_cfg, str) and feature_columns_cfg == "auto":
        dict_work, real_work, feature_columns, feature_strategy, feature_mapping = _resolve_feature_columns_auto(
            dict_df=dict_df,
            real_df=real_df,
            include_global_rate=include_global_rate,
            global_rate_col=global_rate_col,
        )
    else:
        if isinstance(feature_columns_cfg, str):
            explicit_features = [c.strip() for c in feature_columns_cfg.split(",") if c.strip()]
        elif isinstance(feature_columns_cfg, (list, tuple)):
            explicit_features = [str(c) for c in feature_columns_cfg]
        else:
            explicit_features = []
        feature_columns = [c for c in explicit_features if c in dict_df.columns and c in real_df.columns]
        if not feature_columns:
            log.error("No explicit feature columns found in both dictionary and real data.")
            return 1
        if include_global_rate and global_rate_col in dict_df.columns and global_rate_col in real_df.columns:
            if global_rate_col not in feature_columns:
                feature_columns.append(global_rate_col)
        dict_work, real_work = dict_df, real_df
        feature_strategy = "explicit"
        feature_mapping = []

    if not feature_columns:
        log.error("Feature column set is empty after resolution.")
        return 1
    log.info("Using %d features (%s).", len(feature_columns), feature_strategy)

    est_df = estimate_from_dataframes(
        dict_df=dict_work,
        data_df=real_work,
        feature_columns=feature_columns,
        distance_metric=distance_metric,
        interpolation_k=interpolation_k,
        include_global_rate=False,
        global_rate_col=global_rate_col,
        exclude_same_file=exclude_same_file,
    )

    real_with_idx = real_df.copy()
    real_with_idx["dataset_index"] = np.arange(len(real_with_idx), dtype=int)
    merged = pd.merge(est_df, real_with_idx, on="dataset_index", how="left", suffixes=("", "_real"))

    if n_events_column_cfg == "auto":
        n_events_col_used = _pick_n_events_column(merged)
    else:
        n_events_col_used = str(n_events_column_cfg) if str(n_events_column_cfg) in merged.columns else None
    if n_events_col_used is not None:
        merged["n_events"] = pd.to_numeric(merged[n_events_col_used], errors="coerce")
    elif "n_events" not in merged.columns:
        merged["n_events"] = np.nan

    # Real-data efficiencies from rates: eff_i = 1 - threeplane_i / fourplane.
    merged.attrs["preferred_tt_prefixes"] = preferred_tt_prefixes
    raw_eff_by_plane, raw_eff_formula_by_plane, raw_eff_cols_by_plane, raw_eff_selected_prefix = _compute_empirical_efficiencies_from_rates(merged)
    for plane in (1, 2, 3, 4):
        merged[f"eff{plane}_raw_from_data"] = raw_eff_by_plane[plane]
    merged["eff2_from_data"] = merged["eff2_raw_from_data"]
    eff2_formula = raw_eff_formula_by_plane.get(2, "missing_rate_columns")

    # Dictionary-side efficiencies from rates with the same definition.
    dict_df_plot = dict_df.copy()
    dict_df_plot.attrs["preferred_tt_prefixes"] = preferred_tt_prefixes
    dict_eff_by_plane, _, dict_eff_cols_by_plane, dict_eff_selected_prefix = _compute_empirical_efficiencies_from_rates(dict_df_plot)
    for plane in (1, 2, 3, 4):
        dict_df_plot[f"dict_eff{plane}_raw_from_rates"] = dict_eff_by_plane[plane]
    dict_eff2_col = "dict_eff2_raw_from_rates"

    # Preferred global-rate columns for real and dictionary data.
    real_global_rate_col = _pick_global_rate_column(merged, preferred=global_rate_col)
    dict_global_rate_col = _pick_global_rate_column(dict_df_plot, preferred=global_rate_col)

    # Fit-polynomial transform of raw efficiencies using STEP 1.2 summary.
    fit_models_by_plane, fit_status = _load_eff_fit_lines(build_summary_path)
    fit_order_requested, fit_order_by_plane_from_summary = _read_fit_order_info(build_summary_path)
    transformed_eff_by_plane, transformed_eff_formula_by_plane = _transform_efficiencies_with_fits(
        raw_eff_by_plane,
        fit_models_by_plane,
        mode=eff_transform_mode,
    )
    for plane in (1, 2, 3, 4):
        merged[f"eff{plane}_transformed"] = transformed_eff_by_plane[plane]

    dict_transformed_eff_by_plane, dict_transformed_eff_formula_by_plane = _transform_efficiencies_with_fits(
        dict_eff_by_plane,
        fit_models_by_plane,
        mode=eff_transform_mode,
    )
    for plane in (1, 2, 3, 4):
        dict_df_plot[f"dict_eff{plane}_transformed_from_rates"] = dict_transformed_eff_by_plane[plane]

    lut_df = _load_lut(lut_path)
    lut_params = _lut_param_names(lut_df, lut_meta_path if lut_meta_path.exists() else None)
    lut_params = [p for p in lut_params if f"est_{p}" in merged.columns]

    if not lut_params:
        log.warning("No matching LUT parameters found in inference output. Uncertainty columns will be NaN.")

    unc_df = _interpolate_uncertainties(
        query_df=merged,
        lut_df=lut_df,
        param_names=lut_params,
        quantile=uncertainty_quantile,
    )
    merged = pd.concat([merged, unc_df], axis=1)

    for pname in lut_params:
        est_col = f"est_{pname}"
        unc_pct_col = f"unc_{pname}_pct"
        unc_abs_col = f"unc_{pname}_abs"
        if est_col in merged.columns and unc_pct_col in merged.columns:
            est_v = pd.to_numeric(merged[est_col], errors="coerce").to_numpy(dtype=float)
            up = pd.to_numeric(merged[unc_pct_col], errors="coerce").to_numpy(dtype=float)
            abs_unc = np.abs(est_v) * np.abs(up) / 100.0
            merged[unc_abs_col] = np.where(np.isfinite(abs_unc), abs_unc, np.nan)
        else:
            merged[unc_abs_col] = np.nan

    flux_est_col = "est_flux_cm2_min" if "est_flux_cm2_min" in merged.columns else None
    eff_est_col = _choose_primary_eff_est_col(merged)
    distance_col = "best_distance" if "best_distance" in merged.columns else None

    if flux_est_col is None:
        fallback_flux = [c for c in merged.columns if c.startswith("est_") and "flux" in c]
        if fallback_flux:
            flux_est_col = sorted(fallback_flux)[0]
    if distance_col is None:
        log.error("Inference output has no 'best_distance' column.")
        return 1

    success = pd.to_numeric(merged[distance_col], errors="coerce").notna()
    if flux_est_col is not None:
        success &= pd.to_numeric(merged[flux_est_col], errors="coerce").notna()
    if eff_est_col is not None:
        success &= pd.to_numeric(merged[eff_est_col], errors="coerce").notna()
    merged["inference_success"] = success.astype(int)

    x, x_label, has_time_axis = _time_axis(merged)
    if has_time_axis:
        merged = merged.assign(execution_time_for_plot=x)

    out_csv = FILES_DIR / "real_results.csv"
    merged.to_csv(out_csv, index=False)
    log.info("Wrote real results: %s (%d rows)", out_csv, len(merged))

    flux_unc_abs_col = None
    eff_unc_abs_col = None
    if flux_est_col is not None:
        flux_param = flux_est_col.replace("est_", "", 1)
        candidate = f"unc_{flux_param}_abs"
        if candidate in merged.columns:
            flux_unc_abs_col = candidate
    if eff_est_col is not None:
        eff_param = eff_est_col.replace("est_", "", 1)
        candidate = f"unc_{eff_param}_abs"
        if candidate in merged.columns:
            eff_unc_abs_col = candidate

    _plot_pre_estimation_efficiencies(
        x=x,
        has_time_axis=has_time_axis,
        xlabel=x_label,
        global_rate=merged[real_global_rate_col] if real_global_rate_col is not None else None,
        global_rate_label=real_global_rate_col if real_global_rate_col is not None else "global_rate_hz",
        raw_eff_by_plane=raw_eff_by_plane,
        raw_eff_source_prefix=raw_eff_selected_prefix,
        transformed_eff_by_plane=transformed_eff_by_plane,
        transform_mode=eff_transform_mode,
        out_path=PLOT_EFF,
    )

    n_eff2_real = 0
    n_eff2_dict = 0
    flux_from_contour_col = "flux_from_eff2_global_rate_hz"
    eff2_plot_source_cfg = str(cfg_42.get("eff2_global_rate_eff_source", "auto")).strip().lower()
    if eff2_plot_source_cfg not in {"auto", "transformed", "raw"}:
        log.warning(
            "Invalid step_4_2.eff2_global_rate_eff_source=%r; using 'auto'.",
            eff2_plot_source_cfg,
        )
        eff2_plot_source_cfg = "auto"

    has_real_eff2_trans = (
        "eff2_transformed" in merged.columns
        and pd.to_numeric(merged["eff2_transformed"], errors="coerce").notna().any()
    )
    has_dict_eff2_trans = (
        "dict_eff2_transformed_from_rates" in dict_df_plot.columns
        and pd.to_numeric(dict_df_plot["dict_eff2_transformed_from_rates"], errors="coerce").notna().any()
    )
    transformed_available = has_real_eff2_trans and has_dict_eff2_trans
    real_eff2_trans_frac_physical = (
        _fraction_in_closed_interval(merged["eff2_transformed"], 0.0, 1.0)
        if has_real_eff2_trans
        else 0.0
    )
    dict_eff2_trans_frac_physical = (
        _fraction_in_closed_interval(dict_df_plot["dict_eff2_transformed_from_rates"], 0.0, 1.0)
        if has_dict_eff2_trans
        else 0.0
    )

    if eff2_plot_source_cfg == "raw":
        use_transformed_eff2_for_plane_plot = False
    elif eff2_plot_source_cfg == "transformed":
        use_transformed_eff2_for_plane_plot = transformed_available
    else:
        # Auto: only trust transformed eff2 when both real and dictionary are
        # predominantly within physical bounds.
        use_transformed_eff2_for_plane_plot = (
            transformed_available
            and real_eff2_trans_frac_physical >= 0.95
            and dict_eff2_trans_frac_physical >= 0.95
        )
        if transformed_available and not use_transformed_eff2_for_plane_plot:
            log.warning(
                "STEP_4.2.5 fallback to raw eff2: transformed physical fractions "
                "(real=%.3f, dict=%.3f) below threshold 0.95.",
                real_eff2_trans_frac_physical,
                dict_eff2_trans_frac_physical,
            )

    eff2_real_col_for_plane_plot = (
        "eff2_transformed" if use_transformed_eff2_for_plane_plot else "eff2_from_data"
    )
    dict_eff2_col_for_plane_plot = (
        "dict_eff2_transformed_from_rates" if use_transformed_eff2_for_plane_plot else dict_eff2_col
    )
    if real_global_rate_col is not None and dict_global_rate_col is not None:
        n_eff2_real, n_eff2_dict, flux_from_contour_col = _plot_eff2_vs_global_rate(
            real_df=merged,
            dict_df=dict_df_plot,
            real_eff2_col=eff2_real_col_for_plane_plot,
            real_rate_col=real_global_rate_col,
            dict_eff2_col=dict_eff2_col_for_plane_plot,
            dict_rate_col=dict_global_rate_col,
            out_path=PLOT_EFF2_RATE,
        )
    else:
        missing = []
        if real_global_rate_col is None:
            missing.append("real global_rate column")
        if dict_global_rate_col is None:
            missing.append("dictionary global_rate column")
        _plot_placeholder(
            PLOT_EFF2_RATE,
            "Eff2 contour placement in flux-eff plane",
            "Cannot build plot: missing " + ", ".join(missing) + ".",
        )

    n_est_curve_real = 0
    n_est_curve_dict = 0
    est_curve_eff_col = _pick_estimated_eff_col_for_plane(merged, plane=2)
    dict_curve_eff_col = _pick_dictionary_eff_col_for_plane(dict_df_plot, plane=2)
    if (
        flux_est_col is not None
        and est_curve_eff_col is not None
        and dict_curve_eff_col is not None
        and dict_global_rate_col is not None
    ):
        n_est_curve_real, n_est_curve_dict = _plot_estimated_curve_flux_vs_eff(
            real_df=merged,
            dict_df=dict_df_plot,
            est_flux_col=flux_est_col,
            est_eff_col=est_curve_eff_col,
            dict_eff_col=dict_curve_eff_col,
            dict_rate_col=dict_global_rate_col,
            out_path=PLOT_EST_CURVE,
        )
    else:
        missing = []
        if flux_est_col is None:
            missing.append("estimated flux column")
        if est_curve_eff_col is None:
            missing.append("estimated efficiency column")
        if dict_curve_eff_col is None:
            missing.append("dictionary efficiency column")
        if dict_global_rate_col is None:
            missing.append("dictionary global_rate column")
        _plot_placeholder(
            PLOT_EST_CURVE,
            "Estimated curve in flux-eff plane",
            "Cannot build plot: missing " + ", ".join(missing) + ".",
        )

    story_eff_col = eff_est_col if eff_est_col is not None else est_curve_eff_col
    flux_reference_series = None
    flux_reference_label = "Contour-derived flux reference"
    if flux_from_contour_col in merged.columns:
        cand_ref = pd.to_numeric(merged[flux_from_contour_col], errors="coerce")
        if cand_ref.notna().any():
            flux_reference_series = merged[flux_from_contour_col]

    if flux_est_col is not None and story_eff_col is not None and real_global_rate_col is not None:
        _plot_flux_recovery_story_real(
            x=x,
            has_time_axis=has_time_axis,
            xlabel=x_label,
            distance_series=merged[distance_col],
            distance_label=distance_col,
            eff_series=merged[story_eff_col],
            eff_label=story_eff_col,
            global_rate_series=merged[real_global_rate_col],
            global_rate_label=real_global_rate_col,
            flux_est_series=merged[flux_est_col],
            flux_unc_series=merged[flux_unc_abs_col] if flux_unc_abs_col is not None else None,
            flux_reference_series=flux_reference_series,
            flux_reference_label=flux_reference_label,
            out_path=PLOT_RECOVERY_STORY,
        )
    else:
        missing = []
        if flux_est_col is None:
            missing.append("estimated flux column")
        if story_eff_col is None:
            missing.append("estimated efficiency column")
        if real_global_rate_col is None:
            missing.append("real global_rate column")
        _plot_placeholder(
            PLOT_RECOVERY_STORY,
            "Real-data recovery story",
            "Cannot build plot: missing " + ", ".join(missing) + ".",
        )

    ts_valid = pd.Series([], dtype="datetime64[ns]")
    if has_time_axis:
        ts_valid = pd.to_datetime(x, errors="coerce").dropna()

    summary = {
        "real_collected_csv": str(real_path),
        "dictionary_csv": str(dict_path),
        "uncertainty_lut_csv": str(lut_path),
        "matching_criteria_source": "step_2_1",
        "distance_metric": distance_metric,
        "interpolation_k": interpolation_k,
        "feature_strategy": feature_strategy,
        "n_features_used": int(len(feature_columns)),
        "feature_columns_used": feature_columns,
        "n_rows": int(len(merged)),
        "n_successful_rows": int(merged["inference_success"].sum()),
        "coverage_fraction": float(merged["inference_success"].mean()),
        "flux_estimate_column": flux_est_col,
        "eff_estimate_column": eff_est_col,
        "distance_column": distance_col,
        "n_events_column_used": n_events_col_used,
        "real_global_rate_column_used": real_global_rate_col,
        "dictionary_global_rate_column_used": dict_global_rate_col,
        "build_summary_json_used": str(build_summary_path),
        "fit_lines_load_status": fit_status,
        "fit_polynomial_order_requested": fit_order_requested,
        "fit_polynomial_order_by_plane": fit_order_by_plane_from_summary,
        "fit_lines_by_plane": {
            str(k): {"a": float(v[0]), "b": float(v[1])}
            for k, v in fit_models_by_plane.items()
            if len(v) == 2
        },
        "fit_polynomials_by_plane": {
            str(k): {
                "order": int(len(v) - 1),
                "coefficients": [float(c) for c in v],
            }
            for k, v in fit_models_by_plane.items()
        },
        "eff_transform_mode": eff_transform_mode,
        "raw_efficiency_columns": {
            "eff1": "eff1_raw_from_data",
            "eff2": "eff2_raw_from_data",
            "eff3": "eff3_raw_from_data",
            "eff4": "eff4_raw_from_data",
        },
        "efficiency_source_task_ids": selected_task_ids,
        "efficiency_source_most_advanced_task_id": int(max(selected_task_ids)) if selected_task_ids else 1,
        "efficiency_source_preferred_prefix_order": preferred_tt_prefixes,
        "efficiency_source_prefix_used_real": raw_eff_selected_prefix,
        "efficiency_source_prefix_used_dictionary": dict_eff_selected_prefix,
        "raw_efficiency_formulas": {f"eff{p}": raw_eff_formula_by_plane.get(p) for p in (1, 2, 3, 4)},
        "raw_efficiency_rate_columns": {
            f"eff{p}": raw_eff_cols_by_plane.get(p, {})
            for p in (1, 2, 3, 4)
        },
        "transformed_efficiency_columns": {
            "eff1": "eff1_transformed",
            "eff2": "eff2_transformed",
            "eff3": "eff3_transformed",
            "eff4": "eff4_transformed",
        },
        "transformed_efficiency_formulas": {
            f"eff{p}": transformed_eff_formula_by_plane.get(p)
            for p in (1, 2, 3, 4)
        },
        "eff2_real_column": eff2_real_col_for_plane_plot,
        "eff2_global_rate_eff_source_requested": eff2_plot_source_cfg,
        "eff2_global_rate_eff_source_used": (
            "transformed" if use_transformed_eff2_for_plane_plot else "raw"
        ),
        "eff2_global_rate_transformed_fraction_in_0_1": {
            "real": float(real_eff2_trans_frac_physical),
            "dictionary": float(dict_eff2_trans_frac_physical),
        },
        "eff2_formula": (
            transformed_eff_formula_by_plane.get(2)
            if eff2_real_col_for_plane_plot == "eff2_transformed"
            else eff2_formula
        ),
        "eff2_raw_formula": eff2_formula,
        "eff2_transformed_formula": transformed_eff_formula_by_plane.get(2),
        "eff2_real_rate_columns": raw_eff_cols_by_plane.get(2, {}),
        "eff2_dictionary_eff_column": dict_eff2_col_for_plane_plot,
        "eff2_dictionary_eff_formula": (
            dict_transformed_eff_formula_by_plane.get(2)
            if dict_eff2_col_for_plane_plot == "dict_eff2_transformed_from_rates"
            else "raw_from_rates"
        ),
        "eff2_dictionary_rate_columns": {
            "three_plane_col": dict_eff_cols_by_plane.get(2, {}).get("three_plane_col"),
            "four_plane_col": dict_eff_cols_by_plane.get(2, {}).get("four_plane_col"),
        },
        "n_eff2_real_points_for_plane_plot": int(n_eff2_real),
        "n_dictionary_points_for_plane_plot": int(n_eff2_dict),
        "eff2_global_rate_plot": str(PLOT_EFF2_RATE),
        "eff2_contour_flux_column": flux_from_contour_col,
        "pre_estimation_efficiency_plot": str(PLOT_EFF),
        "estimated_curve_eff_column": est_curve_eff_col,
        "dictionary_curve_eff_column": dict_curve_eff_col,
        "n_estimated_curve_points": int(n_est_curve_real),
        "n_dictionary_curve_background_points": int(n_est_curve_dict),
        "estimated_curve_plot": str(PLOT_EST_CURVE),
        "recovery_story_plot": str(PLOT_RECOVERY_STORY),
        "recovery_story_eff_column": story_eff_col,
        "recovery_story_global_rate_column": real_global_rate_col,
        "recovery_story_flux_column": flux_est_col,
        "recovery_story_flux_reference_column": (
            flux_from_contour_col if flux_reference_series is not None else None
        ),
        "lut_param_names_used": lut_params,
        "uncertainty_quantile": uncertainty_quantile,
        "has_time_axis": bool(has_time_axis),
        "time_min_utc": str(ts_valid.min()) if len(ts_valid) else None,
        "time_max_utc": str(ts_valid.max()) if len(ts_valid) else None,
        "feature_mapping_preview": feature_mapping[:25],
    }
    out_summary = FILES_DIR / "real_analysis_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote summary: %s", out_summary)
    log.info("Wrote plots in: %s", PLOTS_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
