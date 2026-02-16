#!/usr/bin/env python3
"""Plots for STEP 10 — adapted from MASTER_STEPS/STEP_10/step_10_triggered_to_jitter.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_rate_summary(df: pd.DataFrame, pdf: PdfPages) -> bool:
    if "T_thick_s" not in df.columns:
        return False
    t0_s = df["T_thick_s"].to_numpy(dtype=float)
    t0_s = t0_s[np.isfinite(t0_s)]
    if t0_s.size == 0:
        return False
    t0_s = np.sort(t0_s.astype(int))
    sec_min = int(t0_s[0])
    counts = np.bincount(t0_s - sec_min)
    seconds = np.arange(sec_min, sec_min + len(counts))
    if len(counts) > 1:
        counts = counts[1:]
        seconds = seconds[1:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(seconds, counts, linewidth=1.0, color="slateblue")
    axes[0].set_title("Counts per second (detector)")
    axes[0].set_xlabel("Second")
    axes[0].set_ylabel("Events")

    axes[1].hist(counts, bins=60, color="teal", alpha=0.8)
    axes[1].set_title("Histogram of counts per second")
    axes[1].set_xlabel("Events per second")
    axes[1].set_ylabel("Counts")

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return True


def pick_tt_column(df: pd.DataFrame) -> str | None:
    for col in ("tt_trigger", "tt_hit", "tt_avalanche", "tt_crossing"):
        if col in df.columns:
            return col
    return None


def plot_rate_by_tt(df: pd.DataFrame, pdf: PdfPages) -> bool:
    if "T_thick_s" not in df.columns:
        return False
    tt_col = pick_tt_column(df)
    if not tt_col:
        return False
    t0_s = df["T_thick_s"].to_numpy(dtype=float)
    tt_vals = df[tt_col].astype("string").fillna("")
    valid = np.isfinite(t0_s) & (tt_vals != "")
    if not valid.any():
        return False
    t0_s = t0_s[valid].astype(int)
    tt_vals = tt_vals[valid].to_numpy()
    minute_idx = (t0_s // 60).astype(int)
    minute_min = int(minute_idx.min())
    times = minute_idx - minute_min
    unique_tt = sorted(set(tt_vals))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for tt in unique_tt:
        mask = tt_vals == tt
        if not mask.any():
            continue
        counts = np.bincount(times[mask])
        minutes = np.arange(minute_min, minute_min + len(counts))
        if len(counts) > 1:
            counts = counts[1:]
            minutes = minutes[1:]
        ax.plot(minutes, counts, linewidth=1.0, label=str(tt))
    ax.set_title(f"Counts per minute by {tt_col}")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Events")
    ax.legend(title=tt_col, ncol=3, fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return True


def plot_timing_closure_summary(
    df6: pd.DataFrame | None,
    df7: pd.DataFrame | None,
    df8: pd.DataFrame | None,
    df9: pd.DataFrame | None,
    df10: pd.DataFrame | None,
    cfg7: dict,
    cfg10: dict,
    pdf: PdfPages,
) -> None:
    if df10 is None:
        return
    tfront_offsets = cfg7.get("tfront_offsets", [[0, 0, 0, 0]] * 4)
    tback_offsets = cfg7.get("tback_offsets", [[0, 0, 0, 0]] * 4)
    tdc_sigma = float(cfg10.get("tdc_sigma_ns", 0.0))
    jitter_width = float(cfg10.get("jitter_width_ns", 0.0))
    expected_tdc_rms = np.sqrt(tdc_sigma ** 2 + (jitter_width ** 2) / 12.0)

    def collect_residual(after: pd.DataFrame, before: pd.DataFrame, offsets: list[list[float]] | None) -> np.ndarray:
        residuals = []
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                for side, offset_map in (("front", offsets), ("back", offsets)):
                    col = f"T_{side}_{plane_idx}_s{strip_idx}"
                    if col not in after.columns or col not in before.columns:
                        continue
                    a = after[col].to_numpy(dtype=float)
                    b = before[col].to_numpy(dtype=float)
                    n = min(len(a), len(b))
                    if n == 0:
                        continue
                    a = a[:n]
                    b = b[:n]
                    mask = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
                    if not mask.any():
                        continue
                    res = a[mask] - b[mask]
                    if offset_map is not None:
                        expected = float(offset_map[plane_idx - 1][strip_idx - 1])
                        res = res - expected
                    residuals.append(res)
        if not residuals:
            return np.array([])
        return np.concatenate(residuals)

    def plot_hist(ax: plt.Axes, data: np.ndarray, title: str, expected_rms: float | None = None) -> None:
        if data.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        ax.hist(data, bins=80, color="slateblue", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Residual (ns)")
        ax.set_ylabel("Counts")
        if expected_rms is not None and np.isfinite(expected_rms):
            ax.axvline(expected_rms, color="black", linestyle="--", linewidth=1.0)
            ax.axvline(-expected_rms, color="black", linestyle="--", linewidth=1.0)

    if df6 is not None:
        residuals = []
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                tf = f"T_front_{plane_idx}_s{strip_idx}"
                tb = f"T_back_{plane_idx}_s{strip_idx}"
                td = f"T_diff_{plane_idx}_s{strip_idx}"
                if tf not in df6.columns or tb not in df6.columns or td not in df6.columns:
                    continue
                front = df6[tf].to_numpy(dtype=float)
                back = df6[tb].to_numpy(dtype=float)
                tdiff = df6[td].to_numpy(dtype=float)
                mask = np.isfinite(front) & np.isfinite(back) & np.isfinite(tdiff)
                if not mask.any():
                    continue
                residuals.append((front[mask] - back[mask]) + 2.0 * tdiff[mask])
        fig, ax = plt.subplots(figsize=(8, 5))
        data = np.concatenate(residuals) if residuals else np.array([])
        plot_hist(ax, data, "STEP 6: (T_front - T_back) + 2*T_diff")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    if df7 is not None and df6 is not None:
        res_front = collect_residual(df7, df6, tfront_offsets)
        res_back = collect_residual(df7, df6, tback_offsets)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_hist(axes[0], res_front, "STEP 7: T_front delta - offset")
        plot_hist(axes[1], res_back, "STEP 7: T_back delta - offset")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    if df8 is not None and df7 is not None:
        res_front = collect_residual(df8, df7, None)
        res_back = collect_residual(df8, df7, None)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_hist(axes[0], res_front, "STEP 8: T_front delta (FEE noise)")
        plot_hist(axes[1], res_back, "STEP 8: T_back delta (FEE noise)")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    if df9 is not None and df10 is not None:
        res_front = collect_residual(df10, df9, None)
        res_back = collect_residual(df10, df9, None)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_hist(axes[0], res_front, "STEP 10: T_front delta (TDC + jitter)", expected_rms=expected_tdc_rms)
        plot_hist(axes[1], res_back, "STEP 10: T_back delta (TDC + jitter)", expected_rms=expected_tdc_rms)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def plot_jitter_summary(
    df: pd.DataFrame,
    output_path: Path,
    rate_df: pd.DataFrame | None = None,
    closure_dfs: dict | None = None,
    cfg7: dict | None = None,
    cfg10: dict | None = None,
) -> None:
    with PdfPages(output_path) as pdf:
        if rate_df is not None:
            added = plot_rate_summary(rate_df, pdf)
            if not added:
                print("Step 10 plots: skipped rate summary (T_thick_s missing).")
            added_tt = plot_rate_by_tt(rate_df, pdf)
            if not added_tt:
                print("Step 10 plots: skipped rate-by-TT summary (T_thick_s or tt column missing).")
        if closure_dfs and cfg7 is not None and cfg10 is not None:
            plot_timing_closure_summary(
                closure_dfs.get("step6"),
                closure_dfs.get("step7"),
                closure_dfs.get("step8"),
                closure_dfs.get("step9"),
                closure_dfs.get("step10"),
                cfg7,
                cfg10,
                pdf,
            )
        tt_col = pick_tt_column(df)
        if tt_col:
            fig, ax = plt.subplots(figsize=(8, 6))
            counts = df[tt_col].astype("string").fillna("").value_counts().sort_index()
            bars = ax.bar(counts.index.astype(str), counts.values, color="slateblue", alpha=0.8)
            for patch in bars:
                patch.set_rasterized(True)
            ax.set_title(f"{tt_col} counts")
            ax.set_xlabel(tt_col)
            ax.set_ylabel("Counts")
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                tf_col = f"T_front_{plane_idx}_s{strip_idx}"
                tb_col = f"T_back_{plane_idx}_s{strip_idx}"
                if tf_col not in df.columns and tb_col not in df.columns:
                    ax.axis("off")
                    continue
                if tf_col in df.columns:
                    vals = df[tf_col].to_numpy(dtype=float)
                    vals = vals[~np.isnan(vals)]
                    ax.hist(vals, bins=80, color="steelblue", alpha=0.6, label="front")
                if tb_col in df.columns:
                    vals = df[tb_col].to_numpy(dtype=float)
                    vals = vals[~np.isnan(vals)]
                    ax.hist(vals, bins=80, color="darkorange", alpha=0.6, label="back")
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("time (ns)")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                qf_col = f"Q_front_{plane_idx}_s{strip_idx}"
                qb_col = f"Q_back_{plane_idx}_s{strip_idx}"
                if qf_col not in df.columns and qb_col not in df.columns:
                    ax.axis("off")
                    continue
                if qf_col in df.columns:
                    vals = df[qf_col].to_numpy(dtype=float)
                    vals = vals[vals != 0]
                    ax.hist(vals, bins=80, color="steelblue", alpha=0.6, label="front")
                if qb_col in df.columns:
                    vals = df[qb_col].to_numpy(dtype=float)
                    vals = vals[vals != 0]
                    ax.hist(vals, bins=80, color="darkorange", alpha=0.6, label="back")
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("charge")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

def find_any_chunk_for_step(step: int) -> Path | None:
    root = Path(__file__).resolve().parents[3] / "INTERSTEPS"
    parts = sorted(root.glob(f"**/step_{step}_chunks/part_*.pkl"))
    if parts:
        return parts[0]
    manifests = sorted(root.rglob(f"*step_{step}_chunks.chunks.json"))
    if manifests:
        try:
            j = json.loads(manifests[0].read_text())
            parts_list = [p for p in j.get("parts", []) if p.endswith(".pkl")]
            if parts_list:
                return Path(parts_list[0])
        except Exception:
            return manifests[0]
    return None


def load_df(path: Path) -> pd.DataFrame:
    if path.name.endswith(".chunks.json"):
        j = json.loads(path.read_text())
        parts = [p for p in j.get("parts", []) if p.endswith(".pkl")]
        if parts:
            path = Path(parts[0])
        else:
            raise FileNotFoundError("Manifest contains no .pkl parts")
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError("Unsupported chunk type")


# --- helpers to inspect upstream/metadata so we can render closure/jitter pages ---
def load_metadata(path: Path) -> dict:
    if path.name.endswith(".chunks.json"):
        return json.loads(path.read_text()).get("metadata", {})
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return {}
    return {}


def resolve_upstream_chain(start_path: Path) -> dict:
    chain = {}
    # start with the step10 path passed in
    step10_path = start_path
    meta10 = load_metadata(step10_path)
    if meta10.get("step") == "STEP_10":
        src = meta10.get("source_dataset")
        if src:
            step10_path = Path(src)
    chain["step10"] = step10_path

    # walk upstream via metadata.source_dataset
    def _src(path: Path) -> Path | None:
        m = load_metadata(path)
        src = m.get("source_dataset")
        return Path(src) if src else None

    step9 = _src(step10_path)
    if step9:
        chain["step9"] = step9
        step8 = _src(step9)
        if step8:
            chain["step8"] = step8
            step7 = _src(step8)
            if step7:
                chain["step7"] = step7
                step6 = _src(step7)
                if step6:
                    chain["step6"] = step6
    return chain


def main() -> None:
    step = 10
    sample = find_any_chunk_for_step(step)
    if sample is None:
        print("No sample chunk found for STEP 10; exiting.")
        return
    print(f"Using sample: {sample}")

    # load the primary dataframe for STEP 10
    df = load_df(sample)

    # attempt to resolve upstream sample files and configs for closure plots
    chain = resolve_upstream_chain(sample)
    closure_dfs: dict = {}
    cfg7 = {}
    cfg10 = {}
    for sname in ("step6", "step7", "step8", "step9", "step10"):
        p = chain.get(sname)
        if p and p.exists():
            try:
                closure_dfs[sname] = load_df(p)
            except Exception:
                closure_dfs[sname] = None
    # try to load cfg7/cfg10 from metadata if available
    if "step7" in chain:
        cfg7 = load_metadata(chain["step7"]).get("config", {}) or {}
    if "step10" in chain:
        cfg10 = load_metadata(chain["step10"]).get("config", {}) or {}

    out_dir = Path(__file__).resolve().parent / "PLOTS"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"step_{step}_sample_plots.pdf"

    # produce the full STEP_10 PDF (rate + rate_by_tt + closure/jitter pages)
    plot_jitter_summary(df, out_path, rate_df=df, closure_dfs=closure_dfs, cfg7=cfg7, cfg10=cfg10)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
