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

band_index = 2
band_data = normalized_pixels[:, band_index].flatten()

# Filter out positive values
band_data = band_data[band_data > 0]

# Fit the lognormal distribution
shape, loc, scale = stats.lognorm.fit(band_data, floc=0)  # 固定位置参数为0

# Calculate the mean and standard deviation of the fitting parameters
mu = np.log(scale)
std = shape

# Draw the fitted lognormal distribution curve
plt.figure(figsize=(13.5, 6))
binwidth = 0.01
manual_colors = ['#cc6b5f', '#c03b4c', '#2e80b5',
                 '#46668a', '#456f97', '#d3b6c8',
                 '#8d7e95', '#545166', '#988196']
sns.histplot(band_data, binwidth=binwidth, kde=False, stat='density', color=manual_colors[5], label='Cri Dataset: Band 3', alpha=.5)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.lognorm.pdf(x, shape, loc, scale)
plt.plot(x, p, manual_colors[7], linewidth=4, alpha=.8, label='Fitted Log-Normal Distribution')

plt.title(f'Log-Normal Distribution Fit: Shape = {shape:.2f}, Scale = {scale:.2f}', fontsize=22)
plt.xlabel('Normalized Pixel Intensity', fontsize=22)
plt.ylabel('Density', fontsize=22)

plt.tick_params(axis='both', which='major', labelsize=18)  # 设置主刻度标签的字体大小
plt.legend(fontsize=18)
plt.savefig(f'cri_band3_log_normal1.png')
plt.show()
