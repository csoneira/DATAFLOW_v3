#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulation parameters
duration = 60 * 15  # total time in seconds
coincidence_window = 200e-9  # coincidence time window in seconds (20 ns)
noise_rates = np.arange(50, 501, 25)  # noise rates from 50 to 300 Hz
n_detectors = 4

def generate_noise_hits(rate, duration):
    """Generate timestamps for a Poisson process with given rate."""
    n_hits = np.random.poisson(rate * duration)
    return np.sort(np.random.uniform(0, duration, n_hits))

def count_coincidences(timestamps_list, coincidence_window, n_planes):
    """Count n-plane random coincidences within the time window."""
    from itertools import combinations

    total_coincidences = 0
    for combo in combinations(range(len(timestamps_list)), n_planes):
        hits_combo = [timestamps_list[i] for i in combo]
        # Use a multi-way coincidence search
        indices = [0] * n_planes
        while True:
            current_times = [hits_combo[j][indices[j]] for j in range(n_planes)]
            t_min, t_max = min(current_times), max(current_times)
            if t_max - t_min <= coincidence_window:
                total_coincidences += 1
                for j in range(n_planes):
                    indices[j] += 1
                    if indices[j] >= len(hits_combo[j]):
                        return total_coincidences
            else:
                j_min = current_times.index(t_min)
                indices[j_min] += 1
                if indices[j_min] >= len(hits_combo[j_min]):
                    return total_coincidences

results = {
    "noise_rate": [],
    "2-plane": [],
    "3-plane": [],
    "4-plane": []
}

for rate in noise_rates:
    # Generate independent noise timestamps for each detector
    timestamps_list = [generate_noise_hits(rate, duration) for _ in range(n_detectors)]

    # Count coincidences
    c2 = count_coincidences(timestamps_list, coincidence_window, 2)
    c3 = count_coincidences(timestamps_list, coincidence_window, 3)
    c4 = count_coincidences(timestamps_list, coincidence_window, 4)

    results["noise_rate"].append(rate)
    results["2-plane"].append(c2 / duration)
    results["3-plane"].append(c3 / duration)
    results["4-plane"].append(c4 / duration)

# Display results
df = pd.DataFrame(results)
print(df)

#%%

# Optional: plot
plt.figure()
plt.plot(df["noise_rate"], df["2-plane"], label="2-plane")
plt.plot(df["noise_rate"], df["3-plane"], label="3-plane")
plt.plot(df["noise_rate"], df["4-plane"], label="4-plane")
plt.xlabel("Noise Rate per Detector (Hz)")
plt.ylabel("Random Coincidence Rate (Hz)")
plt.legend()
plt.grid(True)
plt.title("Random Coincidence Rates vs Noise Rate")
# plt.yscale("log")
plt.tight_layout()
plt.savefig("random_coincidence_rates.png")
# plt.show()

# %%