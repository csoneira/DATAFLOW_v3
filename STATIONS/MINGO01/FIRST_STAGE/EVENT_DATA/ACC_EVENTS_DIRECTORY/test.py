import pandas as pd

resampled_df = pd.read_csv("accumulated_events_24-03-26_23.43.46.csv", sep=',', parse_dates=['Time'])
import pandas as pd
for i in range(1, 5):
    col_name = f"streamer_percent_{i}"
    resampled_df[col_name] = pd.to_numeric(resampled_df[col_name], errors='coerce')

import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(10, 5))
plt.plot(resampled_df["Time"], resampled_df["streamer_percent_1"], marker='o', linestyle='-', label="Streamer Percent 1")
plt.xlabel("Time")
plt.ylabel("Streamer Percentage")
plt.title("Streamer Percentage Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
