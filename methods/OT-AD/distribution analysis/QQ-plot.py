import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.stats as stats

# Load hyperspectral image data
data = scio.loadmat('../data/cri.mat')
hsi = data['data']
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

# Create a Q-Q graph
plt.figure(figsize=(4, 4))
res = stats.probplot(band_data, dist="norm", plot=plt)

# Customize the color, thickness and transparency of the actual quantile line
actual_line = plt.gca().get_lines()[0]
actual_line.set_color(manual_colors[5])
actual_line.set_linewidth(10)
actual_line.set_alpha(1)

# Customize the attributes of the theoretical quantile line
theoretical_line = plt.gca().get_lines()[1]
theoretical_line.set_color(manual_colors[7])
theoretical_line.set_linewidth(2)
theoretical_line.set_alpha(0.8)

# Add legend
plt.legend(['Sample Quantiles', 'Theoretical Quantiles'], fontsize=10)
plt.title('Q-Q Plot of Background in Cri: Band 19', fontsize=12)
plt.xlabel('Theoretical Normal Quantiles', fontsize=10)
plt.ylabel('Sample Quantiles', fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.grid()
# plt.savefig(f'cri_band19_normal_QQ.png')
plt.show()
