import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio


# Load hyperspectral image data
dataset_name = 'sandiego'
data = scio.loadmat(f'../data/{dataset_name}.mat')
hsi = data['data']
anomaly_map = data['map']
nrow, ncol, nbands = hsi.shape
one_coords = np.argwhere(anomaly_map == 1)

plt.figure(figsize=(14, 6))
plt.title(f'Spectral Curve of Anomalous Pixels in San Diego', fontsize=24)
plt.xlabel('Wavelength Band', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)

# Extract the spectra for each abnormal coordinate and plot them
for coord in one_coords:
    spectrum = hsi[coord[0], coord[1], :]
    plt.plot(spectrum, alpha=0.18)

# Define the pixel coordinates
# manual_pixel_coords = [(79, 27), (81, 35), (84, 27), (85, 51), (85, 60)]  # abu-airport-4
manual_pixel_coords = [(8, 85), (10, 89), (20, 67), (21, 71), (33, 52)]  # sandiego
manual_colors = ['#cc6b5f', '#c03b4c', '#2e80b5',
                 '#46668a', '#456f97']

# Draw the spectral curve
for (i, (pixel_row, pixel_col)) in enumerate(manual_pixel_coords):
    if 0 <= pixel_row < nrow and 0 <= pixel_col < ncol:
        manual_spectrum = hsi[pixel_row, pixel_col, :]
        plt.plot(manual_spectrum, color=manual_colors[i], linestyle='-', linewidth=2, alpha=1,
                 label=f'Surrounding Pixel ({pixel_row}, {pixel_col})')
    else:
        print(f"The manually set pixel coordinates ({pixel_row}, {pixel_col}) are out of range!")

plt.legend(fontsize=13.2)
plt.savefig(f'{dataset_name}_surrounding.png')
plt.show()
