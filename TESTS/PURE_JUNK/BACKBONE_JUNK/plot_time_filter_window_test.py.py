#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulated data (replace with calibrated_data and T_sum_columns in your real case)
np.random.seed(0)
n_events = 10
n_detectors = 5
time_coincidence_window = 5  # example value
data = np.random.normal(100, 10, size=(n_events, n_detectors))
mask = np.random.rand(n_events, n_detectors) > 0.2
data[~mask] = 0  # Introduce some zeros

T_sum_columns = pd.DataFrame(data, columns=[f'det_T_sum_{i}' for i in range(n_detectors)])
mean_T_sum = T_sum_columns.apply(lambda row: row[row != 0].median() if row[row != 0].size > 0 else 0, axis=1)
diff_T_sum = T_sum_columns.sub(mean_T_sum, axis=0)
time_window_mask = np.abs(diff_T_sum) <= time_coincidence_window
time_window_mask[T_sum_columns == 0] = True

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, (row, mask_row, mean_val) in enumerate(zip(T_sum_columns.values, time_window_mask.values, mean_T_sum)):
    for j, (val, keep) in enumerate(zip(row, mask_row)):
        if val == 0:
            continue
        color = 'green' if keep else 'red'
        ax.plot(val, i, 'o', color=color)
    ax.axvline(mean_val, ymin=(i - 0.4)/n_events, ymax=(i + 0.4)/n_events, color='black', linestyle='--')
    ax.fill_betweenx([i - 0.4, i + 0.4], mean_val - time_coincidence_window, mean_val + time_coincidence_window,
                     color='gray', alpha=0.2)

ax.set_xlabel('T_sum')
ax.set_ylabel('Event Index')
ax.set_title('T_sum Filtering by Coincidence Window')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %%
