import os
import pandas as pd

# Base directory
grafana_directory = os.path.expanduser("~/DATAFLOW_v3/GRAFANA_DATA")

for x in range(1, 5):
    print("--------------------------------------------------------------------")
    # Input CSV path
    csv_path = os.path.join(grafana_directory, f"data_for_grafana_{x}.csv")
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        continue

    # Output directory
    output_dir = os.path.join(grafana_directory, f"MINGO0{x}")
    existing_dates = set()

    # If the directory exists, collect all existing dates
    if os.path.isdir(output_dir):
        existing_dates = {
            fname.replace(".dat", "")
            for fname in os.listdir(output_dir)
            if fname.endswith(".dat")
        }

    # Read and parse CSV
    df = pd.read_csv(csv_path)
    if 'Time' not in df.columns:
        raise ValueError(f"No 'Time' column in {csv_path}")
    df['Time'] = pd.to_datetime(df['Time'])
    df['date_str'] = df['Time'].dt.date.astype(str)  # Format: 'YYYY-MM-DD'

    # Filter rows whose date does NOT already exist as a .dat file
    df_new = df[~df['date_str'].isin(existing_dates)]

    if df_new.empty:
        print(f"All .dat files already exist for data_for_grafana_{x}.csv")
        continue

    # Create output directory only if needed
    os.makedirs(output_dir, exist_ok=True)

    # Group and save only new dates
    for date_str, group in df_new.groupby('date_str'):
        output_file = os.path.join(output_dir, f"{date_str}.dat")
        group.drop(columns='date_str').to_csv(output_file, index=False)
        print(f"Created {output_file}")
print("--------------------------------------------------------------------")