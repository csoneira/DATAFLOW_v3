from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _rate_series_by_window(frame: pd.DataFrame, mask: pd.Series, window_seconds: int) -> pd.Series:
    selected = frame.loc[mask, "elapsed_s"]
    if selected.empty:
        return pd.Series(dtype=float)
    window_index = (selected // window_seconds).astype(int)
    counts = window_index.groupby(window_index).size().sort_index()
    return counts.astype(float) / float(window_seconds)


def plot_acquisition_rate_vs_time_by_trigger_type(
    read_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    accumulation_window_seconds: int = 60,
) -> bool:
    if read_df.empty or "datetime" not in read_df.columns or "column_6" not in read_df.columns:
        return False

    datetimes = pd.to_datetime(read_df["datetime"], errors="coerce")
    valid_mask = datetimes.notna()
    if not valid_mask.any():
        return False

    accumulation_window_seconds = max(1, int(accumulation_window_seconds))
    frame = read_df.loc[valid_mask, ["column_6"]].copy()
    frame.loc[:, "elapsed_s"] = (
        datetimes.loc[valid_mask] - datetimes.loc[valid_mask].min()
    ).dt.total_seconds().astype(int)
    trigger_values = pd.to_numeric(frame["column_6"], errors="coerce")

    series_by_label: list[tuple[str, pd.Series]] = [
        ("all valid", _rate_series_by_window(frame, pd.Series(True, index=frame.index), accumulation_window_seconds)),
        ("coincidence", _rate_series_by_window(frame, trigger_values.eq(1), accumulation_window_seconds)),
        ("self-trigger", _rate_series_by_window(frame, trigger_values.eq(2), accumulation_window_seconds)),
    ]
    other_mask = ~(trigger_values.eq(1) | trigger_values.eq(2))
    if other_mask.any():
        series_by_label.append(("other/unknown", _rate_series_by_window(frame, other_mask, accumulation_window_seconds)))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, counts in series_by_label:
        if counts.empty:
            continue
        ax.plot(counts.index.to_numpy() * accumulation_window_seconds, counts.to_numpy(), linewidth=1.2, label=label)

    ax.set_title(title)
    ax.set_xlabel("Seconds from first valid acquisition timestamp")
    ax.set_ylabel(f"Rate [Hz], {accumulation_window_seconds}s accumulation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def plot_acquisition_rate_vs_time_by_acq_tt_with_histograms(
    read_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    accumulation_window_seconds: int = 60,
) -> bool:
    required_columns = {"datetime", "acq_tt"}
    if read_df.empty or not required_columns.issubset(read_df.columns):
        return False

    datetimes = pd.to_datetime(read_df["datetime"], errors="coerce")
    acq_tt_values = pd.to_numeric(read_df["acq_tt"], errors="coerce")
    valid_mask = datetimes.notna() & acq_tt_values.notna()
    if not valid_mask.any():
        return False

    accumulation_window_seconds = max(1, int(accumulation_window_seconds))
    frame = pd.DataFrame(
        {
            "elapsed_s": (
                datetimes.loc[valid_mask] - datetimes.loc[valid_mask].min()
            ).dt.total_seconds().astype(int),
            "acq_tt": acq_tt_values.loc[valid_mask].astype(int),
        }
    )
    if frame.empty:
        return False

    frame.loc[:, "window_index"] = (frame["elapsed_s"] // accumulation_window_seconds).astype(int)
    full_windows = np.arange(int(frame["window_index"].min()), int(frame["window_index"].max()) + 1)
    rates_by_tt = (
        frame.groupby(["window_index", "acq_tt"])
        .size()
        .unstack(fill_value=0)
        .reindex(full_windows, fill_value=0)
        .sort_index(axis=1)
        .astype(float)
        / float(accumulation_window_seconds)
    )
    if rates_by_tt.empty:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_time, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [4.5, 1.3], "wspace": 0.04},
    )
    colors = plt.get_cmap("tab10").colors
    all_rates = rates_by_tt.to_numpy(dtype=float).ravel()
    finite_rates = all_rates[np.isfinite(all_rates)]
    max_rate = float(finite_rates.max()) if finite_rates.size else 1.0
    bin_count = max(8, min(40, int(np.ceil(max_rate)) + 1))
    bins = np.linspace(0.0, max(max_rate, 1.0), bin_count)
    if np.unique(bins).size < 2:
        bins = np.array([0.0, 1.0])

    for idx, acq_tt in enumerate(rates_by_tt.columns):
        rates = rates_by_tt[acq_tt].to_numpy(dtype=float)
        color = colors[idx % len(colors)]
        label = f"acq_tt={int(acq_tt)}"
        ax_time.plot(rates_by_tt.index.to_numpy() * accumulation_window_seconds, rates, linewidth=1.1, label=label, color=color)
        ax_hist.hist(
            rates[np.isfinite(rates)],
            bins=bins,
            orientation="horizontal",
            histtype="step",
            linewidth=1.2,
            color=color,
            label=label,
        )

    ax_time.set_title("Rate versus time")
    ax_time.set_xlabel("Seconds from first valid acquisition timestamp")
    ax_time.set_ylabel(f"Rate [Hz], {accumulation_window_seconds}s accumulation")
    ax_time.grid(True, alpha=0.25)
    ax_time.legend(loc="best", fontsize=8)

    ax_hist.set_title("Rate histogram")
    ax_hist.set_xlabel("Windows")
    ax_hist.grid(True, alpha=0.25, axis="y")
    ax_hist.tick_params(axis="y", labelleft=False)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True
