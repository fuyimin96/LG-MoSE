import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as scio
import scipy.stats as stats

# Load hyperspectral image data
data = scio.loadmat('../data/cri.mat')
hsi = data['data']  # 提取 'hsi' 数据
anomaly_map = data['map']
nrow, ncol, nbands = hsi.shape

# Extract background pixel points
background_mask = anomaly_map == 0
background_data = hsi[background_mask]

# Normalize
normalized_pixels = (background_data - np.min(background_data, axis=0)) / (
    np.max(background_data, axis=0) - np.min(background_data, axis=0))

band_index = 18
band_data = normalized_pixels[:, band_index].flatten()
manual_colors = ['#cc6b5f', '#c03b4c', '#2e80b5',
                 '#46668a', '#456f97', '#d3b6c8',
                 '#8d7e95', '#545166', '#988196']
# Fit the normal distribution
mu, std = stats.norm.fit(band_data)

# Draw the fitted normal distribution curve
plt.figure(figsize=(13.5, 6))
binwidth = 0.01
sns.histplot(band_data, binwidth=binwidth, kde=False, stat='density', color=manual_colors[5], label='Cri Dataset: Band 19', alpha=.5)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, manual_colors[7], linewidth=4, alpha=.8, label='Fitted Normal Distribution')

plt.title(f'Normal Distribution Fit: μ = {mu:.2f}, σ = {std:.2f}', fontsize=22)
plt.xlabel('Normalized Pixel Intensity', fontsize=22)
plt.ylabel('Density', fontsize=22)

plt.tick_params(axis='both', which='major', labelsize=18)
plt.legend(fontsize=18)
plt.savefig(f'cri_band19_normal_distribution1.png')
plt.show()
