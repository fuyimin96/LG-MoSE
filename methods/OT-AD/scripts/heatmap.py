import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio

save_dir = '../heat_map/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def heatmap(img, save_name):
    plt.figure(figsize=(8, 8))
    img_real = img.real
    _plt = sns.heatmap(img_real, cmap='RdBu_r', vmax=1.0, annot=False, xticklabels=False,
                       yticklabels=False, cbar=False, linewidths=0.0, rasterized=True)
    _plt.figure.savefig(save_name, format='png', bbox_inches='tight', pad_inches=0.0)
    plt.close()


method_list = ['RX', 'CRD', 'GTVLRR', 'PTA', 'IEEPST', 'Auto-AD', 'DirectNet', 'GT-HAD', 'DFAN', 'OT-AD']
file_list = ['abu-airport-4', 'abu-beach-3', 'abu-urban-2', 'abu-urban-3', 'abu-urban-4']
# file_list = ['hydice']

for file in file_list:
    mat_dir = os.path.join('../results/', file)
    for method in method_list:
        mat_name = os.path.join(mat_dir, 'Detection_map_' + method + '.mat')
        mat = sio.loadmat(mat_name)
        img = mat['map']
        # Normalize
        img = img - img.min()
        img = img / (img.max() - img.min())
        # save fig
        save_subdir = os.path.join(save_dir, file)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)
        save_name = os.path.join(save_subdir, method + '.png')
        heatmap(img, save_name)
