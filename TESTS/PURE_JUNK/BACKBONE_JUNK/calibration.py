# %%
from scipy.interpolate import CubicSpline
import numpy as np

# Extract width and fast charge data
width_table = np.array([
    0.0000001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
    160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
    300, 310, 320, 330, 340, 350, 360, 370, 380, 390
])

fast_charge_table = np.array([
    4.0530E+01, 2.6457E+02, 4.5081E+02, 6.0573E+02, 7.3499E+02, 8.4353E+02,
    9.3562E+02, 1.0149E+03, 1.0845E+03, 1.1471E+03, 1.2047E+03, 1.2592E+03,
    1.3118E+03, 1.3638E+03, 1.4159E+03, 1.4688E+03, 1.5227E+03, 1.5779E+03,
    1.6345E+03, 1.6926E+03, 1.7519E+03, 1.8125E+03, 1.8742E+03, 1.9368E+03,
    2.0001E+03, 2.0642E+03, 2.1288E+03, 2.1940E+03, 2.2599E+03, 2.3264E+03,
    2.3939E+03, 2.4625E+03, 2.5325E+03, 2.6044E+03, 2.6786E+03, 2.7555E+03,
    2.8356E+03, 2.9196E+03, 3.0079E+03, 3.1012E+03
])

# Create a cubic spline interpolator
cs = CubicSpline(width_table, fast_charge_table, bc_type='natural')

# Generate smooth interpolation points
width_smooth = np.linspace(min(width_table), max(width_table), 500)
fast_charge_smooth = cs(width_smooth)

# Plot the smooth interpolation
plt.figure(figsize=(8, 5))
plt.plot(width_table, fast_charge_table, 'o', label="Original Data", color='red')
plt.plot(width_smooth, fast_charge_smooth, '-', label="Cubic Spline Interpolation", color='blue')

# Labels and title
plt.xlabel("Width")
plt.ylabel("Fast Charge")
plt.title("Smooth Interpolation of Fast Charge vs Width")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Show the plot
plt.show()

# %%
