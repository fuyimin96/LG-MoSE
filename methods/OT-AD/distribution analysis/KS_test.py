import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.stats import ks_2samp, probplot


# Load hyperspectral image data
dataset_name = 'hydice'
data = scio.loadmat(f'../data/{dataset_name}.mat')
hsi = data['data']
anomaly_map = data['map']
nrow, ncol, nbands = hsi.shape
print(f"The shape of HSI data: row={nrow}, column={ncol}, bands={nbands}")

# Obtain normal pixels
anomaly_mask = anomaly_map == 1
normal_mask = ~anomaly_mask
normal_pixels = hsi[normal_mask]

# Normalize
normalized_pixels = (normal_pixels - np.min(normal_pixels, axis=0)) / (
        np.max(normal_pixels, axis=0) - np.min(normal_pixels, axis=0))


# Visualize the background distribution: Draw the histogram of each band
def plot_background_distribution(normalized_pixels, nbands, dataset_name):
    cols = 20
    rows = (nbands + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 2))
    for i in range(nbands):
        plt.subplot(rows, cols, i + 1)
        plt.hist(normalized_pixels[:, i], bins=20, color='gray', alpha=0.7)
        plt.title(f'Band {i + 1}')
        plt.xlabel('Normalized Intensity')
        plt.ylabel('Frequency')

    plt.tight_layout()

    save_path = f'./bgdistri/normalized histogram/{dataset_name}_normalized_background_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'图像已保存至: {save_path}')
    plt.show()


# Kolmogorov-Smirnov test
def ks_assumption_test(normalized_pixels, nbands):
    results = []

    # Generate samples with normal distribution
    for i in range(nbands):
        mean = np.mean(normalized_pixels[:, i])
        std = np.std(normalized_pixels[:, i])
        normal_sample = np.random.normal(loc=mean, scale=std, size=normalized_pixels.shape[0])
        # KS test
        statistic, p_value = ks_2samp(normalized_pixels[:, i], normal_sample)

        # Judged based on the p value
        if p_value > 0.05:
            hypothesis_result = "   Do not reject Gaussian assumption"
        else:
            hypothesis_result = "   Reject Gaussian assumption"

        results.append((f'Band {i + 1}', statistic, p_value, hypothesis_result))

    return results


# Save the KS test results
def save_results_to_file(results, dataset_name):
    save_path = f'./bgdistri/gaussian_assumption_results/{dataset_name}_ks_assumption_results.txt'
    with open(save_path, 'w') as f:
        f.write(f'Kolmogorov-Smirnov Test Results for {dataset_name}\n')
        f.write(f'{"Band":<10}{"Statistic":<15}{"P-Value":<15}{"Result"}\n')
        f.write("Hypothesis Test: KS Test for normality\n")
        f.write("Hypothesis: The data follows a Gaussian distribution.\n")
        for band, stat, p_val, result in results:
            f.write(f'{band:<10}{stat:<20} {p_val:<20} {result}\n')
    print(f'The test results have been saved to: {save_path}')


# Visualize the Q-Q graph
def plot_qq_distribution(normalized_pixels, nbands, dataset_name):
    cols = 20
    rows = (nbands + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(nbands):
        plt.subplot(rows, cols, i + 1)
        probplot(normalized_pixels[:, i], dist="norm", plot=plt)
        plt.title(f'Band {i + 1} Q-Q Plot')

    plt.tight_layout()

    save_path = f'./bgdistri/Q-Q/{dataset_name}_QQ_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'The Q-Q image has been saved to: {save_path}')

    plt.show()


# Call function (Multiple options)
# plot_background_distribution(normalized_pixels, nbands, dataset_name)
test_results = ks_assumption_test(normalized_pixels, nbands)
save_results_to_file(test_results, dataset_name)
# plot_qq_distribution(normalized_pixels, nbands, dataset_name)
