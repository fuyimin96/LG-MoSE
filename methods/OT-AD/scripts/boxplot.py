import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np

gt_dir = '../data/'
save_dir = '../box_plot/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def del_zero(x):
    tmp = []
    for i in range(x.shape[0]):
        if x[i] != 0:
            tmp.append(x[i])
    y = np.array(tmp)

    return y


method_list = ['RX', 'CRD', 'GTVLRR', 'PTA', 'IEEPST', 'Auto-AD', 'DirectNet', 'GT-HAD', 'DFAN', 'OT-AD']
# method_list = ['OT-AD-nowd', 'OT-AD-wd1', 'OT-AD-wd2', 'OT-AD-wd12']
file_list = ['abu-airport-4', 'abu-beach-3', 'abu-urban-2', 'abu-urban-3', 'abu-urban-4', 'hydice']

for file in file_list:
    gt_path = os.path.join(gt_dir, file + '.mat')
    mat = sio.loadmat(gt_path)
    gt = mat['map']

    mat_dir = os.path.join('../results/', file)
    data = []

    for method in method_list:
        mat_name = os.path.join(mat_dir, 'Detection_map_' + method + '.mat')
        mat = sio.loadmat(mat_name)
        img = mat['map']
        # Normalize
        img = img.real
        img = img - img.min()
        img = img / (img.max() - img.min())
        bk = img.copy()
        ab = img.copy()

        bk[gt != 0] = 0
        bk = bk.flatten()
        bk = del_zero(bk)
        ab[gt == 0] = 0
        ab = ab.flatten()
        ab = del_zero(ab)
        data.append(ab)
        data.append(bk)

    # Draw boxplot fig
    ax = plt.subplot()
    ax.grid(False)
    ax.set_ylim(0.0, 1.19)
    plt.ylabel('Normalized Detection Statistic Range', fontsize=13)
    plt.yticks(size=12)

    num = 4
    manual_colors = ['#cc6b5f', '#c03b4c', '#2e80b5', '#46668a',
                     '#456f97', '#d3b6c8', '#8d7e95', '#545166',
                     '#988196', '#456F97', '#C03B4C']

    color_list = [manual_colors[4], manual_colors[5]] * num
    color_list2 = [manual_colors[5], manual_colors[4]] * num
    box_position = []
    method_position = []
    pos = 0.0

    for i in range(num * 2):
        if i % 2 == 0:
            pos += 1.0
            method_position.append(pos + 0.2)
        else:
            pos += 0.4
        box_position.append(pos)

    bp = ax.boxplot(data, widths=0.35, patch_artist=True, showfliers=False,
                    positions=box_position, medianprops={'color': 'black'},
                    whiskerprops={'linestyle': '--'})

    # Set boxes
    for patch, color in zip(bp['boxes'], color_list):
        patch.set_facecolor(color)
        patch.set(linewidth=0.75)

    # Set whiskers
    color_list_double = [manual_colors[4], manual_colors[4], manual_colors[5], manual_colors[5]] * num
    for patch, color in zip(bp['whiskers'], color_list_double):
        patch.set(color=color, linewidth=2.2)

    # Set caps
    for patch, color in zip(bp['caps'], color_list_double):
        patch.set(color=color, linewidth=2.5)

    # Set medians
    for patch, color in zip(bp['medians'], color_list2):
        patch.set(color='white', linewidth=1.0)

    plt.xticks(method_position, method_list)
    plt.xticks(rotation=0, size=11)
    labels = ["Anomaly", "Background"]
    plt.legend(bp['boxes'], labels, loc='upper right',
               prop={'size': 12})

    # Save fig
    save_path = os.path.join(save_dir, file + '.png')
    plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close()
