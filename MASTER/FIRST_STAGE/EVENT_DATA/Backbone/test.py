#%%

import numpy as np
import matplotlib.pyplot as plt

n_events = 1000000
sigma = 0.4  # ns, time resolution per detector

# Simulate 2-plane ΔT

d = 1.4 / 4

t2 = np.random.normal([0, 3*d], sigma, size=(n_events, 2))
delta_t2 = np.ptp(t2, axis=1)

# Simulate 4-plane ΔT
t4 = np.random.normal([0, d, 2*d, 3*d], sigma, size=(n_events, 4))
delta_t4 = np.ptp(t4, axis=1)

# Plot comparison
plt.hist(delta_t2, bins=200, density=True, alpha=0.6, label='ΔT (2 planes)')
plt.hist(delta_t4, bins=200, density=True, alpha=0.6, label='ΔT (4 planes)')
plt.xlabel("ΔT (ns)")
plt.ylabel("Density")
plt.title("Peak-to-peak ΔT distribution from Gaussian timing")
plt.legend()
# Logscale
plt.yscale('log')
plt.tight_layout()

plt.show()

# %%
