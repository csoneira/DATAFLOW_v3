from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_rate_vs_time_by_task_tt_with_histograms(
    dataframe: pd.DataFrame,
    *,
    tt_column: str,
    title: str,
    accumulation_window_seconds: int = 60,
    rate_histogram_bins: int = 80,
) -> plt.Figure | None:
    required_columns = {"datetime", tt_column}
    if dataframe.empty or not required_columns.issubset(dataframe.columns):
        return None

    datetimes = pd.to_datetime(dataframe["datetime"], errors="coerce")
    tt_values = pd.to_numeric(dataframe[tt_column], errors="coerce")
    valid_mask = datetimes.notna() & tt_values.notna()
    if not valid_mask.any():
        return None

    accumulation_window_seconds = max(1, int(accumulation_window_seconds))
    frame = pd.DataFrame(
        {
            "elapsed_s": (
                datetimes.loc[valid_mask] - datetimes.loc[valid_mask].min()
            ).dt.total_seconds().astype(int),
            "task_tt": tt_values.loc[valid_mask].astype(int),
        }
    )
    if frame.empty:
        return None

    frame.loc[:, "window_index"] = (
        frame["elapsed_s"] // accumulation_window_seconds
    ).astype(int)
    full_windows = np.arange(
        int(frame["window_index"].min()),
        int(frame["window_index"].max()) + 1,
    )
    rates_by_tt = (
        frame.groupby(["window_index", "task_tt"])
        .size()
        .unstack(fill_value=0)
        .reindex(full_windows, fill_value=0)
        .sort_index(axis=1)
        .astype(float)
        / float(accumulation_window_seconds)
    )
    if rates_by_tt.empty:
        return None

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
    try:
        bin_count = int(rate_histogram_bins)
    except (TypeError, ValueError):
        bin_count = 80
    bin_count = max(8, min(300, bin_count))
    bins = np.linspace(0.0, max(max_rate, 1.0), bin_count)
    if np.unique(bins).size < 2:
        bins = np.array([0.0, 1.0])

    for idx, task_tt in enumerate(rates_by_tt.columns):
        rates = rates_by_tt[task_tt].to_numpy(dtype=float)
        color = colors[idx % len(colors)]
        label = f"{tt_column}={int(task_tt)}"
        ax_time.plot(
            rates_by_tt.index.to_numpy() * accumulation_window_seconds,
            rates,
            linewidth=1.1,
            label=label,
            color=color,
        )
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
    ax_time.set_xlabel("Seconds from first valid task timestamp")
    ax_time.set_ylabel(f"Rate [Hz], {accumulation_window_seconds}s accumulation")
    ax_time.grid(True, alpha=0.25)
    ax_time.legend(loc="best", fontsize=8)

    ax_hist.set_title("Rate histogram")
    ax_hist.set_xlabel("Windows")
    ax_hist.grid(True, alpha=0.25, axis="y")
    ax_hist.tick_params(axis="y", labelleft=False)

    fig.suptitle(title)
    return fig
