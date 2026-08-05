import matplotlib.pyplot as plt

# Create
fig = plt.figure(figsize=(1, 10))

# Normalize
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])

# Add color bars
cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=18)

# Preserve the color bar
cbar_save_name = 'color_bar.png'
plt.savefig(cbar_save_name, format='png', bbox_inches='tight', pad_inches=0.1)
plt.close()
