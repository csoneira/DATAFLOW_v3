# %%

import numpy as np

# Y ----------------------------
def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

y_widths = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]  # T1-T3 and T2-T4 widths
y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]

# ---------------------------------------------------------------------------------------------
# Plotting the centers and edges of the widths
fig, ax = plt.subplots()

# Plot centers (blue)
for i, pos in enumerate(y_pos_T):
    ax.plot(pos, np.zeros_like(pos) + i, 'o', label=f'T{i+1}-T{i+3} Centers', color='blue')

# Plot edges (red)
for i, (widths, pos) in enumerate(zip(y_widths, y_pos_T)):
    edges = np.hstack(([pos[0] - widths[0] / 2], pos + widths / 2))
    ax.plot(edges, np.zeros_like(edges) + i, 'x', label=f'T{i+1}-T{i+3} Edges', color='red')

# Add labels and legend
ax.set_title('Centers and Edges of Widths')
ax.set_xlabel('Position')
ax.set_yticks(range(len(y_pos_T)))
ax.set_yticklabels([f'T{i+1}-T{i+3}' for i in range(len(y_pos_T))])
ax.legend()

plt.show()
# ---------------------------------------------------------------------------------------------