import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
import seaborn as sns
from scipy import stats


# Load hyperspectral image data
dataset_name = 'cri'
data = scio.loadmat(f'../data/{dataset_name}.mat')
hsi = data['data']
anomaly_map = data['map']

# Obtain shape information
nrow, ncol, nbands = hsi.shape
manual_colors = ['#cc6b5f', '#c03b4c', '#2e80b5',
                 '#46668a', '#456f97', '#d3b6c8',
                 '#8d7e95', '#545166', '#988196', '#6280a5']

# Extract background pixel points
background_mask = anomaly_map == 0
background_data = hsi[background_mask]

# Normalize
normalized_pixels = (background_data - np.min(background_data, axis=0)) / (
    np.max(background_data, axis=0) - np.min(background_data, axis=0))

background_data_reshaped = normalized_pixels.reshape(-1, nbands)

# Convert the NumPy array to a Pandas DataFrame
background_df = pd.DataFrame(background_data_reshaped, columns=[f'Spectral Band {i+1}' for i in range(nbands)])

# Select the spectral dimensions (X1, X2) that need to be drawn
x_index = 17
y_index = 18

g = sns.JointGrid(data=background_df, x=background_df.columns[x_index], y=background_df.columns[y_index])
g = g.plot_joint(sns.scatterplot)
g.ax_joint.collections[0].set_facecolor(manual_colors[9])
g.ax_joint.collections[0].set_alpha(0.7)
g.ax_joint.collections[0].set_sizes([25] * len(background_df))

# Draw contour lines
kde = stats.gaussian_kde(background_data_reshaped[:, [x_index, y_index]].T)
x_vals = np.linspace(np.min(background_data_reshaped[:, x_index]), np.max(background_data_reshaped[:, x_index]), 100)
y_vals = np.linspace(np.min(background_data_reshaped[:, y_index]), np.max(background_data_reshaped[:, y_index]), 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

g.ax_joint.contour(X, Y, Z, levels=10, cmap='Reds', alpha=0.5)
g.ax_marg_x.hist(background_df[background_df.columns[x_index]], bins=np.arange(0, 1, 0.0224), color=manual_colors[9], alpha=0.7)
g.ax_marg_y.hist(background_df[background_df.columns[y_index]], orientation='horizontal', bins=np.arange(0, 1, 0.0224), color=manual_colors[9], alpha=0.7)

g.ax_joint.set_xlabel(f'Spectral Band {x_index + 1} of Cri', fontsize=15)
g.ax_joint.set_ylabel(f'Spectral Band {y_index + 1} of Cri', fontsize=15)
g.ax_joint.tick_params(axis='both', which='major', labelsize=13)  # 设置主刻度标签的字体大小

plt.savefig(f'cri1819_joint.png')
plt.show()
