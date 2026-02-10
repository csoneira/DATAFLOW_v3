#!/usr/bin/env python3
"""Build a 3-D empirical uncertainty Look-Up Table (LUT).

Given Step-3 validation results (simulated DATA matched against the
DICTIONARY), this script produces a LUT that maps

    (estimated_flux, estimated_eff, n_events) ──▸ (σ_flux_pct, σ_eff_pct)

Terminology
-----------
- **Dictionary**: the reference lookup table of (flux, eff) → observables,
  produced by Step 1 from simulations.  This is fixed.
- **Data**: the samples being analysed.  During validation these are also
  simulated, but in production they will be *real detector measurements*.
  The LUT maps data-side quantities to uncertainties.

The approach is purely empirical:

1. **Filter** — exclude exact self-matches where a simulated data sample
   matched its own dictionary entry (error ≡ 0), which would bias the
   uncertainty downward.  Controlled by ``--exclude-exact-matches``.
2. **Bin** the remaining validation data into a regular 3-D grid
   (est_flux × est_eff × n_events).
3. **Remove outliers** per cell using inter-quartile range (IQR) filtering.
4. **Store** the chosen quantile (default p68) of the filtered absolute
   relative errors as the representative σ for each cell.

Empty cells (too few samples) are filled by nearest-neighbour so that the
resulting 3-D grid is dense and suitable for trilinear interpolation at
query time.

A global fall-back — the same quantile computed over the entire dataset —
covers edge cases that slip past the grid boundaries.

Outputs (written to ``out_dir``):
    ● ``uncertainty_lut.csv``               – dense 3-D empirical LUT
    ● ``uncertainty_lut_meta.json``          – metadata (ranges, edges, …)
    ● ``lut_diagnostic_overview.png``        – 2×2 diagnostic overview
    ● ``lut_2d_sigma_maps.png``              – 2-D maps of σ at representative N
    ● ``lut_ellipse_validation.png``         – 5-point ellipse sanity check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from msv_utils import (  # noqa: E402
    load_config,
    maybe_log_x,
    parse_list,
    resolve_param,
    setup_logger,
)

log = setup_logger("LUT_BUILD")

DEFAULT_ALL_RESULTS = REPO_ROOT / "STEP_4_SELF_CONSISTENCY" / "output" / "all_samples_results.csv"
DEFAULT_OUT = STEP_DIR / "output" / "lut"
DEFAULT_CONFIG = STEP_DIR / "config.json"


# ── helpers ─────────────────────────────────────────────────────────────────

def _prepare_validation_data(
    df: pd.DataFrame,
    *,
    exclude_exact_matches: bool = True,
) -> pd.DataFrame:
    """Prepare validation data for LUT construction.

    Steps
    -----
    1. Keep only rows with ``status == 'ok'``.
    2. Coerce numeric columns and compute absolute errors if missing.
    3. Drop rows with missing required columns.
    4. (Optional, default on) **Exclude exact self-matches**: when a
       simulated data sample is also present in the dictionary
       (``sample_in_dictionary == True``) and matched with zero error
       on both flux and efficiency, the match found the sample's own
       dictionary entry (or an identical twin).  Including these would
       artificially push σ toward zero.  When real data is used,
       ``sample_in_dictionary`` will be ``False`` and this filter has
       no effect.

    Parameters
    ----------
    df : DataFrame
        Raw Step-3 all-mode results.
    exclude_exact_matches : bool
        If True, remove rows where the data sample matched itself in
        the dictionary (both flux and eff errors are exactly 0).

    Returns
    -------
    Cleaned DataFrame ready for LUT construction.
    """
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"] == "ok"].copy()
    num = [
        "sample_events_count",
        "estimated_flux_cm2_min", "estimated_eff_1",
        "true_flux_cm2_min", "true_eff_1",
        "abs_flux_rel_error_pct", "abs_eff_rel_error_pct",
        "flux_rel_error_pct", "eff_rel_error_pct",
    ]
    for c in num:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "abs_flux_rel_error_pct" not in out.columns:
        out["abs_flux_rel_error_pct"] = out["flux_rel_error_pct"].abs()
    if "abs_eff_rel_error_pct" not in out.columns:
        out["abs_eff_rel_error_pct"] = out["eff_rel_error_pct"].abs()

    required = [
        "sample_events_count",
        "estimated_flux_cm2_min", "estimated_eff_1",
        "abs_flux_rel_error_pct", "abs_eff_rel_error_pct",
    ]
    out = out.dropna(subset=required)
    out = out[out["sample_events_count"] > 0]

    # ── exclude exact self-matches (dictionary leakage) ────────────
    if exclude_exact_matches and "sample_in_dictionary" in out.columns:
        is_in_dict = out["sample_in_dictionary"].astype(str).str.lower().isin(
            ["true", "1", "yes"])
        exact_flux = out["abs_flux_rel_error_pct"] == 0.0
        exact_eff  = out["abs_eff_rel_error_pct"] == 0.0
        self_match = is_in_dict & exact_flux & exact_eff
        n_removed = int(self_match.sum())
        if n_removed > 0:
            log.info("Excluding %d exact self-matches "
                     "(data sample = dictionary entry, error ≡ 0)",
                     n_removed)
            out = out[~self_match]

    return out.reset_index(drop=True)


def _make_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based edges that produce roughly equal-count bins."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.nanquantile(values, qs))
    if len(edges) < 3:
        edges = np.linspace(float(np.nanmin(values)),
                            float(np.nanmax(values)),
                            max(3, n_bins + 1))
    return edges


def _iqr_filter(arr: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Return array with IQR-based outliers removed (NaN-safe)."""
    valid = arr[np.isfinite(arr)]
    if len(valid) < 4:
        return valid
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    return valid[(valid >= lo) & (valid <= hi)]


# ── 3-D empirical LUT ──────────────────────────────────────────────────────

def build_empirical_lut(
    df: pd.DataFrame,
    *,
    flux_bins: int = 5,
    eff_bins: int = 5,
    events_bins: int = 4,
    min_cell_count: int = 5,
    quantile: float = 0.68,
    iqr_factor: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """Build a dense 3-D empirical uncertainty LUT.

    The LUT is built from validation data (data samples matched against
    the dictionary).  At query time the LUT returns the expected
    uncertainty for a new data measurement — whether that measurement
    comes from simulation or from the real detector.

    Parameters
    ----------
    df : DataFrame
        Prepared validation data (output of ``_prepare_validation_data``)
        with columns ``estimated_flux_cm2_min``, ``estimated_eff_1``,
        ``sample_events_count``, ``abs_flux_rel_error_pct``,
        ``abs_eff_rel_error_pct``.
    flux_bins, eff_bins, events_bins : int
        Number of bins along each axis.
    min_cell_count : int
        Cells with fewer samples are left empty (filled later by NN).
    quantile : float
        Quantile of the filtered error distribution to keep (e.g. 0.68).
    iqr_factor : float
        IQR multiplier for outlier removal (1.5 = standard Tukey fence).

    Returns
    -------
    (lut_df, meta_dict)
    """
    flux = df["estimated_flux_cm2_min"].to_numpy(dtype=float)
    eff  = df["estimated_eff_1"].to_numpy(dtype=float)
    evts = df["sample_events_count"].to_numpy(dtype=float)

    flux_edges = _make_edges(flux, flux_bins)
    eff_edges  = _make_edges(eff,  eff_bins)
    evts_edges = _make_edges(evts, events_bins)

    # Digitize
    fi = np.clip(np.digitize(flux, flux_edges) - 1, 0, len(flux_edges) - 2)
    ei = np.clip(np.digitize(eff,  eff_edges)  - 1, 0, len(eff_edges)  - 2)
    ni = np.clip(np.digitize(evts, evts_edges) - 1, 0, len(evts_edges) - 2)

    n_outliers_removed = 0
    rows: list[dict] = []
    for f_i in range(len(flux_edges) - 1):
        for e_i in range(len(eff_edges) - 1):
            for n_i in range(len(evts_edges) - 1):
                mask = (fi == f_i) & (ei == e_i) & (ni == n_i)
                n_in_raw = int(mask.sum())

                row: dict = {
                    "flux_lo": float(flux_edges[f_i]),
                    "flux_hi": float(flux_edges[f_i + 1]),
                    "flux_mid": float((flux_edges[f_i] + flux_edges[f_i + 1]) / 2),
                    "eff_lo":  float(eff_edges[e_i]),
                    "eff_hi":  float(eff_edges[e_i + 1]),
                    "eff_mid": float((eff_edges[e_i] + eff_edges[e_i + 1]) / 2),
                    "events_lo":  float(evts_edges[n_i]),
                    "events_hi":  float(evts_edges[n_i + 1]),
                    "events_mid": float((evts_edges[n_i] + evts_edges[n_i + 1]) / 2),
                    "n_samples_raw": n_in_raw,
                }

                if n_in_raw < min_cell_count:
                    row["n_samples"] = 0
                    row["n_outliers"] = 0
                    row["sigma_flux_pct"] = np.nan
                    row["sigma_eff_pct"]  = np.nan
                    rows.append(row)
                    continue

                sub = df.loc[mask]
                f_err_raw = sub["abs_flux_rel_error_pct"].to_numpy(dtype=float)
                e_err_raw = sub["abs_eff_rel_error_pct"].to_numpy(dtype=float)

                # IQR outlier filtering
                f_err = _iqr_filter(f_err_raw, factor=iqr_factor)
                e_err = _iqr_filter(e_err_raw, factor=iqr_factor)
                n_out = n_in_raw - min(len(f_err), len(e_err))
                n_outliers_removed += max(n_out, 0)

                n_after = min(len(f_err), len(e_err))
                if n_after < 2:
                    row["n_samples"] = 0
                    row["n_outliers"] = max(n_out, 0)
                    row["sigma_flux_pct"] = np.nan
                    row["sigma_eff_pct"]  = np.nan
                    rows.append(row)
                    continue

                row["n_samples"]      = n_after
                row["n_outliers"]     = max(n_out, 0)
                row["sigma_flux_pct"] = float(np.nanpercentile(f_err, quantile * 100))
                row["sigma_eff_pct"]  = float(np.nanpercentile(e_err, quantile * 100))
                rows.append(row)

    lut_df = pd.DataFrame(rows)

    # ── nearest-neighbour infilling of NaN cells ───────────────────────
    nan_mask = lut_df["sigma_flux_pct"].isna()
    n_empty = int(nan_mask.sum())
    if n_empty > 0 and not nan_mask.all():
        valid = lut_df[~nan_mask]
        for idx in lut_df.index[nan_mask]:
            r = lut_df.loc[idx]
            d = (
                ((valid["flux_mid"] - r["flux_mid"])
                 / max(valid["flux_mid"].std(), 1e-12)) ** 2
                + ((valid["eff_mid"] - r["eff_mid"])
                   / max(valid["eff_mid"].std(), 1e-12)) ** 2
                + ((valid["events_mid"] - r["events_mid"])
                   / max(valid["events_mid"].std(), 1e-12)) ** 2
            )
            nearest = valid.loc[d.idxmin()]
            lut_df.at[idx, "sigma_flux_pct"] = nearest["sigma_flux_pct"]
            lut_df.at[idx, "sigma_eff_pct"]  = nearest["sigma_eff_pct"]

    # ── compute global fallback quantile ───────────────────────────────
    f_all = _iqr_filter(df["abs_flux_rel_error_pct"].to_numpy(dtype=float),
                        iqr_factor)
    e_all = _iqr_filter(df["abs_eff_rel_error_pct"].to_numpy(dtype=float),
                        iqr_factor)
    global_sigma_flux = float(np.nanpercentile(f_all, quantile * 100))
    global_sigma_eff  = float(np.nanpercentile(e_all, quantile * 100))

    meta = {
        "flux_edges": flux_edges.tolist(),
        "eff_edges":  eff_edges.tolist(),
        "events_edges": evts_edges.tolist(),
        "min_cell_count": min_cell_count,
        "quantile": quantile,
        "iqr_factor": iqr_factor,
        "n_total_samples": int(len(df)),
        "n_cells_total":   len(rows),
        "n_cells_filled":  int((~nan_mask).sum()) if n_empty > 0 else len(rows),
        "n_cells_infilled": n_empty,
        "n_outliers_removed": n_outliers_removed,
        "global_sigma_flux_pct": global_sigma_flux,
        "global_sigma_eff_pct":  global_sigma_eff,
    }
    return lut_df, meta


# ── plotting ────────────────────────────────────────────────────────────────

def _plot_overview(
    lut_df: pd.DataFrame,
    meta: dict,
    path: Path,
) -> None:
    """2×2 diagnostic overview of the empirical LUT."""
    filled = lut_df[lut_df["n_samples"] > 0]
    if filled.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0, 0) — cell population histogram
    axes[0, 0].hist(filled["n_samples"], bins=30, color="#4C78A8",
                    alpha=0.85, edgecolor="white")
    axes[0, 0].set_xlabel("Samples per cell (after IQR filter)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(
        f"Cell population — {meta.get('n_outliers_removed', 0)} "
        f"outliers removed total")
    axes[0, 0].grid(True, alpha=0.2)

    # (0, 1) — σ_flux vs events
    axes[0, 1].scatter(filled["events_mid"], filled["sigma_flux_pct"],
                       c=filled["n_samples"], s=30, cmap="viridis", alpha=0.7)
    axes[0, 1].set_xlabel("Events (bin mid)")
    axes[0, 1].set_ylabel(f"σ_flux p{int(meta['quantile'] * 100)} [%]")
    axes[0, 1].set_title("σ_flux vs events — colour = cell count")
    axes[0, 1].grid(True, alpha=0.2)
    maybe_log_x(axes[0, 1], filled["events_mid"])

    # (1, 0) — σ_eff vs events
    axes[1, 0].scatter(filled["events_mid"], filled["sigma_eff_pct"],
                       c=filled["n_samples"], s=30, cmap="viridis", alpha=0.7)
    axes[1, 0].set_xlabel("Events (bin mid)")
    axes[1, 0].set_ylabel(f"σ_eff p{int(meta['quantile'] * 100)} [%]")
    axes[1, 0].set_title("σ_eff vs events — colour = cell count")
    axes[1, 0].grid(True, alpha=0.2)
    maybe_log_x(axes[1, 0], filled["events_mid"])

    # (1, 1) — σ_flux vs σ_eff
    axes[1, 1].scatter(filled["sigma_flux_pct"], filled["sigma_eff_pct"],
                       c=filled["events_mid"], s=30, cmap="coolwarm", alpha=0.7)
    axes[1, 1].set_xlabel("σ_flux [%]")
    axes[1, 1].set_ylabel("σ_eff [%]")
    axes[1, 1].set_title("σ_flux vs σ_eff — colour = events")
    axes[1, 1].grid(True, alpha=0.2)

    fig.suptitle("Empirical LUT diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_error_scatter_by_events(
    df: pd.DataFrame,
    path: Path,
    events_edges: list[float] | None = None,
    n_ranges: int = 4,
) -> None:
    """Scatter plots of actual data points coloured by relative error.

    Instead of a discretised LUT grid, show the raw (est_flux, est_eff)
    points coloured by their |relative error|, split into event-count
    ranges.  This reveals continuous tendencies that the binned LUT
    cannot show.

    Layout: 2 rows (flux error, eff error) × N columns (event ranges).
    """
    needed = [
        "estimated_flux_cm2_min", "estimated_eff_1",
        "sample_events_count",
        "abs_flux_rel_error_pct", "abs_eff_rel_error_pct",
    ]
    work = df.dropna(subset=needed).copy()
    if work.empty:
        return

    evts = work["sample_events_count"].to_numpy(dtype=float)

    # Build event-range edges
    if events_edges is not None and len(events_edges) >= 2:
        edges = sorted(set(events_edges))
    else:
        qs = np.linspace(0.0, 1.0, n_ranges + 1)
        edges = list(np.unique(np.nanquantile(evts, qs)))
        if len(edges) < 2:
            edges = [float(np.nanmin(evts)), float(np.nanmax(evts))]

    n_cols = len(edges) - 1
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 9),
                             squeeze=False)

    # Common colour limits (clipped at p95 so outliers don't wash out)
    vmax_f = float(np.nanpercentile(
        work["abs_flux_rel_error_pct"].to_numpy(dtype=float), 95))
    vmax_e = float(np.nanpercentile(
        work["abs_eff_rel_error_pct"].to_numpy(dtype=float), 95))
    vmax_f = max(vmax_f, 0.1)
    vmax_e = max(vmax_e, 0.1)

    for col_i in range(n_cols):
        lo, hi = edges[col_i], edges[col_i + 1]
        if col_i < n_cols - 1:
            mask = (evts >= lo) & (evts < hi)
        else:
            mask = (evts >= lo) & (evts <= hi)  # include upper edge
        sub = work.loc[mask]
        n_pts = len(sub)

        for row_i, (tag, vmax) in enumerate([
            ("flux", vmax_f), ("eff", vmax_e),
        ]):
            ax = axes[row_i, col_i]
            col = f"abs_{tag}_rel_error_pct"
            if sub.empty:
                ax.set_title(f"N ∈ [{int(lo)}, {int(hi)})  (0 pts)")
                ax.set_xlabel("Estimated flux")
                ax.set_ylabel("Estimated eff_1")
                continue

            sc = ax.scatter(
                sub["estimated_flux_cm2_min"],
                sub["estimated_eff_1"],
                c=sub[col],
                s=40,
                cmap="YlOrRd",
                vmin=0,
                vmax=vmax,
                alpha=0.8,
                edgecolors="grey",
                linewidths=0.3,
            )
            cb = fig.colorbar(sc, ax=ax, shrink=0.8)
            cb.set_label(f"|Δ{tag}| [%]", fontsize=9)

            ax.set_xlabel("Estimated flux")
            ax.set_ylabel("Estimated eff_1")
            title_tag = "flux" if tag == "flux" else "efficiency"
            ax.set_title(
                f"|Δ{title_tag}| — N ∈ [{int(lo)}, {int(hi)}]  "
                f"({n_pts} pts)",
                fontsize=10,
            )
            ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Relative error in (flux, eff) space — split by event count",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_ellipse_validation(
    df: pd.DataFrame,
    lut_dir: Path,
    path: Path,
    n_points: int = 5,
) -> None:
    """Show representative data samples with uncertainty ellipses.

    For each sample the plot shows:
      - the dictionary-matched estimate  (est_flux, est_eff) – filled marker
      - the true value (only available for simulated data) – open marker
      - a 1-σ ellipse centred on the estimate whose semi-axes equal the
        LUT-predicted uncertainty

    When real detector data is used, ``true_flux`` / ``true_eff`` will be
    absent — the plot still works but true points are omitted.
    """
    from matplotlib.patches import Ellipse as _Ellipse
    from uncertainty_lut import UncertaintyLUT

    lut = UncertaintyLUT.load(lut_dir)

    # true_flux / true_eff are only available for simulated data
    has_truth = ({"true_flux_cm2_min", "true_eff_1"}
                 .issubset(df.columns)
                 and df["true_flux_cm2_min"].notna().any())

    needed = [
        "estimated_flux_cm2_min", "estimated_eff_1",
        "sample_events_count",
    ]
    sub = df.dropna(subset=needed).copy()
    if sub.empty:
        log.warning("No valid rows for ellipse plot.")
        return

    # Furthest-point sampling on normalised (flux, eff) plane
    flux = sub["estimated_flux_cm2_min"].values
    eff  = sub["estimated_eff_1"].values
    f_rng = float(flux.max() - flux.min()) or 1e-12
    e_rng = float(eff.max() - eff.min()) or 1e-12
    x_n = (flux - flux.min()) / f_rng
    y_n = (eff  - eff.min())  / e_rng
    coords = np.column_stack([x_n, y_n])

    chosen: list[int] = []
    centre = np.array([0.5, 0.5])
    d_centre = np.linalg.norm(coords - centre, axis=1)
    chosen.append(int(d_centre.argmin()))
    for _ in range(n_points - 1):
        d_min = np.full(len(coords), np.inf)
        for ci in chosen:
            d_min = np.minimum(
                d_min, np.linalg.norm(coords - coords[ci], axis=1))
        d_min[chosen] = -1
        chosen.append(int(d_min.argmax()))

    sel = sub.iloc[chosen].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (_, row) in enumerate(sel.iterrows()):
        ef  = row["estimated_flux_cm2_min"]
        ee  = row["estimated_eff_1"]
        nev = row["sample_events_count"]

        sig_f, sig_e = lut.query(ef, ee, nev)
        w = ef * sig_f / 100.0
        h = ee * sig_e / 100.0
        c = colours[i % len(colours)]

        # 1-σ uncertainty ellipse around the dictionary-matched estimate
        ell = _Ellipse(
            (ef, ee), width=2 * w, height=2 * h,
            facecolor=c, alpha=0.15, edgecolor=c, linewidth=1.5,
            label="1σ ellipse" if i == 0 else None,
        )
        ax.add_patch(ell)

        # Data estimate (dictionary match result)
        ax.plot(ef, ee, "o", color=c, markersize=8, markeredgecolor="k",
                markeredgewidth=0.6,
                label="Dict. estimate" if i == 0 else None)

        # True value (only for simulated data; absent for real data)
        if has_truth:
            tf = row.get("true_flux_cm2_min")
            te = row.get("true_eff_1")
            if pd.notna(tf) and pd.notna(te):
                ax.plot(tf, te, "x", color=c, markersize=10,
                        markeredgewidth=2.0,
                        label="True (sim.)" if i == 0 else None)
                ax.plot([ef, tf], [ee, te], "-", color=c,
                        linewidth=0.8, alpha=0.5)

        ax.annotate(
            f"σ_f={sig_f:.2f}%  σ_e={sig_e:.2f}%\nN={int(nev):,}",
            xy=(ef, ee), xytext=(8, 8),
            textcoords="offset points", fontsize=7,
            color=c, weight="bold",
        )

    ax.set_xlabel("Flux [cm⁻² min⁻¹]")
    ax.set_ylabel("Efficiency")
    title_suffix = "(simulated data)" if has_truth else "(real data)"
    ax.set_title(f"Uncertainty ellipse validation  {title_suffix}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.autoscale_view()
    ax.margins(0.1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log.info("Ellipse validation plot → %s", path)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build 3-D empirical uncertainty LUT from STEP_3 "
                    "all-mode results."
    )
    parser.add_argument("--config",           default=str(DEFAULT_CONFIG))
    parser.add_argument("--all-results-csv",  default=None)
    parser.add_argument("--out-dir",          default=None)
    parser.add_argument("--flux-bins",        type=int,   default=None)
    parser.add_argument("--eff-bins",         type=int,   default=None)
    parser.add_argument("--events-bins",      type=int,   default=None)
    parser.add_argument("--min-cell-count",   type=int,   default=None)
    parser.add_argument("--quantile",         type=float, default=None,
                        help="Target quantile for cell σ (default 0.68).")
    parser.add_argument("--iqr-factor",       type=float, default=None,
                        help="IQR multiplier for outlier removal "
                             "(default 1.5).")
    parser.add_argument("--exclude-exact-matches", type=str, default=None,
                        help="Exclude data samples that matched their own "
                             "dictionary entry exactly (default true).")
    parser.add_argument("--sigma-events",     default=None,
                        help="Comma-separated N values for 2-D sigma maps.")
    parser.add_argument("--no-plots",         action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    def _rp(cli, key, default):
        return resolve_param(cli, config, key, default)

    all_csv    = Path(_rp(args.all_results_csv,
                          "all_results_csv", str(DEFAULT_ALL_RESULTS)))
    out_dir    = Path(_rp(args.out_dir,
                          "lut_out_dir", str(DEFAULT_OUT)))
    flux_bins  = int(_rp(args.flux_bins,      "lut_flux_bins",      5))
    eff_bins   = int(_rp(args.eff_bins,       "lut_eff_bins",       5))
    evts_bins  = int(_rp(args.events_bins,    "lut_events_bins",    4))
    min_cell   = int(_rp(args.min_cell_count, "lut_min_cell_count", 5))
    quantile   = float(_rp(args.quantile,     "lut_quantile",       0.68))
    iqr_factor = float(_rp(args.iqr_factor,   "lut_iqr_factor",     1.5))
    exclude_exact = str(
        _rp(getattr(args, 'exclude_exact_matches', None),
            "lut_exclude_exact_matches", "true")
    ).lower() in ("true", "1", "yes")
    sigma_events = parse_list(
        _rp(args.sigma_events, "lut_sigma_events",
            [10000, 30000, 50000, 80000]),
        cast=float,
    )

    if not all_csv.exists():
        raise FileNotFoundError(f"All-results CSV not found: {all_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading validation data (data matched against dictionary): %s",
             all_csv)
    raw = pd.read_csv(all_csv, low_memory=False)
    df = _prepare_validation_data(raw, exclude_exact_matches=exclude_exact)
    log.info("Valid data rows for LUT: %d / %d", len(df), len(raw))

    if df.empty:
        log.error("No valid data rows — cannot build LUT.")
        return 1

    # ── Build empirical LUT from validation data ────────────────────────
    log.info("Building 3-D empirical LUT (%d×%d×%d bins, min_cell=%d, "
             "quantile=%.2f, IQR factor=%.1f, exclude_exact=%s)",
             flux_bins, eff_bins, evts_bins, min_cell, quantile,
             iqr_factor, exclude_exact)

    lut_df, lut_meta = build_empirical_lut(
        df,
        flux_bins=flux_bins,
        eff_bins=eff_bins,
        events_bins=evts_bins,
        min_cell_count=min_cell,
        quantile=quantile,
        iqr_factor=iqr_factor,
    )
    lut_csv = out_dir / "uncertainty_lut.csv"
    lut_df.to_csv(lut_csv, index=False)
    log.info("LUT: %d total cells (%d direct, %d infilled) → %s",
             lut_meta["n_cells_total"],
             lut_meta["n_cells_filled"],
             lut_meta["n_cells_infilled"],
             lut_csv)
    log.info("Outliers removed: %d", lut_meta["n_outliers_removed"])
    log.info("Global fallback: σ_flux=%.3f%%  σ_eff=%.3f%%",
             lut_meta["global_sigma_flux_pct"],
             lut_meta["global_sigma_eff_pct"])

    # ── Metadata ────────────────────────────────────────────────────────
    full_meta = {
        "source_csv": str(all_csv),
        "description": "Empirical uncertainty LUT built from validation "
                       "data (data samples matched against dictionary).  "
                       "Query with real or simulated data-side quantities.",
        "exclude_exact_matches": exclude_exact,
        "n_input_rows": int(len(raw)),
        "n_valid_rows": int(len(df)),
        "estimated_flux_range": [
            float(df["estimated_flux_cm2_min"].min()),
            float(df["estimated_flux_cm2_min"].max())],
        "estimated_eff_range": [
            float(df["estimated_eff_1"].min()),
            float(df["estimated_eff_1"].max())],
        "events_range": [
            float(df["sample_events_count"].min()),
            float(df["sample_events_count"].max())],
        "lut": lut_meta,
    }
    meta_path = out_dir / "uncertainty_lut_meta.json"
    meta_path.write_text(json.dumps(full_meta, indent=2), encoding="utf-8")
    log.info("Wrote metadata: %s", meta_path)

    # ── Plots ───────────────────────────────────────────────────────────
    if not args.no_plots:
        _plot_overview(lut_df, lut_meta,
                       out_dir / "lut_diagnostic_overview.png")
        _plot_error_scatter_by_events(
            df, out_dir / "lut_error_scatter_by_events.png",
            events_edges=sigma_events,
        )
        _plot_ellipse_validation(df, out_dir,
                                 out_dir / "lut_ellipse_validation.png")
        log.info("Diagnostic plots saved to %s", out_dir)

    log.info("Done. LUT outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
