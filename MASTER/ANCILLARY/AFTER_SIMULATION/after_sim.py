#%%

# TASK 1 --> channel counts
# TASK 2 --> 


from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- knobs to tweak ---
STATION_A = "MINGO02"  # e.g. MINGO01, MINGO02, ...
STATION_B = "MINGO06"
STEP = 1             # numeric step (1, 2, ...)
TASK = 1          # for STEP_1 use an int (1-5); keep None for steps without tasks
START_DATE_A = "2025-07-01 00:00:00"    # e.g. "2025-11-06 18:00:00" or leave None
END_DATE_A = "2026-01-20 00:00:00"      # e.g. "2025-11-06 19:00:00" or leave None
START_DATE_B = "2025-07-01 00:00:00"
END_DATE_B = "2026-01-20 00:00:00"
# Resample window for station A mean points (e.g. "1D", "7D", "30D").
A_MEAN_RESAMPLE = "7D"
# Window used when counting events (ns) and per-combination measured counts.
# Set WINDOW_NS to the calibration window you used (e.g., coincidence_window_cal_ns),
# and fill MEASURED_COUNTS with {combo: observed_counts}.
WINDOW_NS = None
MEASURED_COUNTS = {
    # 12: 0,
    # 123: 0,
}


repo_root = Path(__file__).resolve().parents[3]


def resolve_metadata_path(station: str) -> Path:
    event_dir = (
        repo_root / "STATIONS" / station / "STAGE_1" / "EVENT_DATA" / f"STEP_{STEP}"
    )
    if TASK is None:
        return event_dir / "METADATA" / f"step_{STEP}_metadata_specific.csv"
    return event_dir / f"TASK_{TASK}" / "METADATA" / f"task_{TASK}_metadata_specific.csv"


def load_station_metadata(
    station: str, start_date: str | None, end_date: str | None
) -> tuple[pd.DataFrame, Path, str]:
    metadata_path = resolve_metadata_path(station)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot find metadata at {metadata_path}")

    df = pd.read_csv(metadata_path)
    filename_col = "filename_base" if "filename_base" in df.columns else None
    timestamp_col = None
    for candidate in ("execution_time", "execution_timestamp"):
        if candidate in df.columns:
            timestamp_col = candidate
            break

    if filename_col:
        df["datetime"] = df[filename_col].apply(filename_to_datetime)
        df = df.sort_values(by="datetime")

    filter_col = "datetime" if "datetime" in df.columns else timestamp_col
    if not filter_col:
        raise ValueError(f"No timestamp column found for {station}")

    if filter_col and (start_date or end_date):
        if filter_col == "execution_timestamp":
            fmt = "%Y-%m-%d_%H.%M.%S"
            df[filter_col] = pd.to_datetime(df[filter_col], format=fmt, errors="coerce")
        else:
            df[filter_col] = pd.to_datetime(df[filter_col], errors="coerce")
        start = pd.to_datetime(start_date) if start_date else df[filter_col].min()
        end = pd.to_datetime(end_date) if end_date else df[filter_col].max()
        df = df.loc[df[filter_col].between(start, end)]
    else:
        df[filter_col] = pd.to_datetime(df[filter_col], errors="coerce")

    df = df.rename(columns={filter_col: "timestamp"})
    return df, metadata_path, station


def filename_to_datetime(value: str):
    """Parse strings like mi0XYYDDDHHMMSS into real datetimes."""
    if not isinstance(value, str):
        return pd.NaT
    core = value[3:] if value.startswith("mi0") else value
    if len(core) < 12:
        return pd.NaT
    try:
        year = 2000 + int(core[1:3])
        day_of_year = int(core[3:6])
        hour = int(core[6:8])
        minute = int(core[8:10])
        second = int(core[10:12])
        return datetime(year, 1, 1) + timedelta(
            days=day_of_year - 1, hours=hour, minutes=minute, seconds=second
        )
    except Exception:
        return pd.NaT


df_a, metadata_a, station_a = load_station_metadata(
    STATION_A, START_DATE_A, END_DATE_A
)
df_b, metadata_b, station_b = load_station_metadata(
    STATION_B, START_DATE_B, END_DATE_B
)

print(f"Loaded: {metadata_a}")
print(f"Rows ({station_a}): {len(df_a)}")
print(f"Loaded: {metadata_b}")
print(f"Rows ({station_b}): {len(df_b)}")

tt_count_cols_a = [c for c in df_a.columns if "raw_tt_" in c and c.endswith("_count")]
tt_count_cols_b = [c for c in df_b.columns if "raw_tt_" in c and c.endswith("_count")]
common_tt_cols = [c for c in tt_count_cols_a if c in tt_count_cols_b]

if not tt_count_cols_a:
    raise ValueError(f"No TT count columns found for {station_a}.")
if not tt_count_cols_b:
    raise ValueError(f"No TT count columns found for {station_b}.")

def to_percentage_df(df: pd.DataFrame, tt_cols: list[str]) -> pd.DataFrame:
    total_hits = df[tt_cols].sum(axis=1)
    total_hits = total_hits.where(total_hits != 0)
    pct_df = df[tt_cols].div(total_hits, axis=0)
    pct_df.insert(0, "timestamp", df["timestamp"])
    return pct_df


pct_a = to_percentage_df(df_a, tt_count_cols_a)
pct_b = to_percentage_df(df_b, tt_count_cols_b)

plot_a = pct_a[["timestamp"] + tt_count_cols_a].dropna(subset=["timestamp"]).copy()
plot_a = plot_a.set_index("timestamp")
ax = plot_a.plot(figsize=(12, 6), marker="o", linewidth=1)
ax.set_title(f"TT % over time • {station_a} STEP {STEP} TASK {TASK}")
ax.set_xlabel("timestamp")
ax.set_ylabel("Fraction of total hits (0-1)")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

plot_b = pct_b[["timestamp"] + tt_count_cols_b].dropna(subset=["timestamp"]).copy()
plot_b = plot_b.set_index("timestamp")
ax = plot_b.plot(figsize=(12, 6), marker="o", linewidth=1)
ax.set_title(f"TT % over time • {station_b} STEP {STEP} TASK {TASK}")
ax.set_xlabel("timestamp")
ax.set_ylabel("Fraction of total hits (0-1)")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

if not common_tt_cols:
    raise ValueError("No common TT count columns found between stations.")

mean_a_segments = (
    pct_a[["timestamp"] + common_tt_cols]
    .dropna(subset=["timestamp"])
    .set_index("timestamp")
    .resample(A_MEAN_RESAMPLE)
    .mean()
)
mean_b = pct_b[common_tt_cols].mean(numeric_only=True)

mean_a_long = mean_a_segments.stack().reset_index()
mean_a_long.columns = ["window_start", "tt_col", "mean_a"]
mean_a_long["mean_b"] = mean_a_long["tt_col"].map(mean_b)
mean_merged = mean_a_long.dropna(subset=["mean_a", "mean_b"])

fig, ax = plt.subplots(figsize=(7, 6))
for tt_col in mean_merged["tt_col"].unique():
    subset = mean_merged[mean_merged["tt_col"] == tt_col]
    ax.scatter(subset["mean_a"], subset["mean_b"], alpha=0.75, label=tt_col)
min_val = min(mean_merged["mean_a"].min(), mean_merged["mean_b"].min())
max_val = max(mean_merged["mean_a"].max(), mean_merged["mean_b"].max())
if pd.notna(min_val) and pd.notna(max_val):
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
ax.set_title(
    f"Station A mean windows vs Station B total mean • {station_a} vs {station_b}"
)
ax.set_xlabel(f"{station_a} mean fraction per {A_MEAN_RESAMPLE}")
ax.set_ylabel(f"{station_b} total mean fraction")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()

print("Columns (station A):")
for col in df_a.columns:
    print(f" - {col}")
print("Columns (station B):")
for col in df_b.columns:
    print(f" - {col}")

# %%
