import numpy as np
import matplotlib.pyplot as plt

# Define the normalized magnitude scale
s01 = np.linspace(0, 1, 256)

# Compute RGB components
r = s01
g = 4 * s01 * (1 - s01)  # Bell-shaped
b = 1 - s01
colors = np.stack([r, g, b], axis=1)

# Plot the color bar
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = plt.cm.colors.ListedColormap(colors)
norm = plt.Normalize(vmin=0, vmax=1)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=ax, orientation='horizontal')
cb.set_label('Feature Magnitude (Normalized)')

plt.title("Magnitude-to-Color Mapping: Blue → Green → Red")
plt.show()
