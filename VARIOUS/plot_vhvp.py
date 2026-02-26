#!/usr/bin/env python3
"""Simple script to plot VHVp vs time from scan_hv_oct_24.csv.

Usage:
    python plot_vhvp.py

Generates: /home/mingo/DATAFLOW_v3/VARIOUS/scan_hv_oct_24_20240916-30_4-6.png (filtered)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# paths
BASE_DIR = os.path.dirname(__file__)
CSV_FILE = os.path.join(BASE_DIR, "scan_hv_oct_24.csv")
# output includes date and vhvp range
OUTPUT_PNG = os.path.join(BASE_DIR, "scan_hv_oct_24.png")

# read data
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

# parse dates
df = pd.read_csv(CSV_FILE, parse_dates=["time"])
df["VHVp"] = pd.to_numeric(df["VHVp"], errors="coerce")

# apply date range filter
start_date = pd.to_datetime("2024-09-27")
end_date = pd.to_datetime("2024-10-01")
mask_date = (df["time"] >= start_date) & (df["time"] <= end_date)

# filter VHVp between 4 and 6
mask_vhvp = (df["VHVp"] >= 4.0) & (df["VHVp"] <= 7.0)

filtered = df[mask_date & mask_vhvp]

# create plot
plt.figure(figsize=(10, 4))
plt.plot(filtered["time"], filtered["VHVp"], marker=".", linestyle="-", color="C0")
plt.xlabel("time")
plt.ylabel("VHVp")
plt.title("VHVp vs time (4 ≤ VHVp ≤ 6)")
plt.grid(True)
plt.tight_layout()

# save
plt.savefig(OUTPUT_PNG)
print(f"Saved plot to {OUTPUT_PNG}")
