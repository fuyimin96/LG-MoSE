import os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np

manual_colors = ['#cc6b5f', '#c03b4c', '#2e80b5', '#46668a',
                 '#456f97', '#d3b6c8', '#8d7e95', '#545166',
                 '#988196', '#456F97', '#C03B4C']
save_dir = '../roc/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

color_list = [manual_colors[0], 'slategray', manual_colors[2], manual_colors[3], manual_colors[4],
              manual_colors[5], manual_colors[6], manual_colors[7], manual_colors[8], manual_colors[10]]
method_list = ['RX', 'CRD', 'GTVLRR', 'PTA', 'IEEPST', 'Auto-AD', 'DirectNet', 'GT-HAD', 'DFAN', 'OT-AD']
file_list = ['abu-airport-4', 'abu-beach-3', 'abu-urban-2', 'abu-urban-3', 'abu-urban-4', 'hydice']

for file in file_list:
    method_dict = {}
    mat_dir = os.path.join('../results/', file)
    data_path = os.path.join('../data/', f'{file}.mat')
    mat = sio.loadmat(data_path)
    gt = mat['map'].flatten()
    for method in method_list:
        map_name = os.path.join(mat_dir, 'Detection_map_' + method + '.mat')
        if not os.path.exists(map_name):
            continue
        result = sio.loadmat(map_name)
        det_map = result['map']
        det_map = det_map - det_map.min()
        det_map = det_map / (det_map.max() - det_map.min())
        det_map = det_map.flatten()
        threshold = np.unique(det_map)
        PD_list = []
        FAR_list = []

        # Calculate TPR and FPR one threshold at a time
        for t in threshold:
            pred = (det_map >= t).astype(int)
            TP = np.sum((pred == 1) & (gt == 1))
            FP = np.sum((pred == 1) & (gt == 0))
            FN = np.sum((pred == 0) & (gt == 1))
            TN = np.sum((pred == 0) & (gt == 0))

            PD = TP / (TP + FN) if (TP + FN) > 0 else 0
            FAR = FP / (FP + TN) if (FP + TN) > 0 else 0

            PD_list.append(PD)
            FAR_list.append(FAR)

        PD = np.array(PD_list)
        FAR = np.array(FAR_list)

        sort_idx = np.argsort(threshold)[::-1]
        PD = PD[sort_idx]
        FAR = FAR[sort_idx]
        threshold = threshold[sort_idx]

        method_dict[method] = [FAR, PD, threshold]

    # Draw 3D roc fig
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.set_zlim(0.0, 1.0)

    ax.tick_params(axis='x', pad=-3)
    ax.tick_params(axis='y', pad=-3)
    ax.tick_params(axis='z', pad=0)

    # Obtain the boundary range
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Change the elevation Angle and azimuth Angle
    ax.view_init(elev=15, azim=140)

    # The eight connected corner points
    corners = np.array([[xlim[0], ylim[0], zlim[0]],
                        [xlim[1], ylim[0], zlim[0]],
                        [xlim[1], ylim[1], zlim[0]],
                        [xlim[0], ylim[1], zlim[0]],
                        [xlim[0], ylim[0], zlim[1]],
                        [xlim[1], ylim[0], zlim[1]],
                        [xlim[1], ylim[1], zlim[1]],
                        [xlim[0], ylim[1], zlim[1]]])

    # Define the connecting line
    edges1 = [
        [0, 1], [1, 2], [1, 5]
    ]
    edges2 = [
        [3, 0], [2, 3], [4, 5],
        [5, 6], [6, 7], [7, 4],
        [0, 4], [3, 7], [2, 6]
    ]

    for start, end in edges1:
        ax.plot([corners[start][0], corners[end][0]],
                [corners[start][1], corners[end][1]],
                [corners[start][2], corners[end][2]],
                color='black', linewidth=0.75)

    ax.set_ylabel('False Alarm Rate', fontsize=11, labelpad=-2)
    ax.set_xlabel('Threshold', fontsize=11, labelpad=-2)
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    ax.text2D(-0.04, 0.53, 'Probability of Detection', transform=ax.transAxes, rotation=90, fontsize=11, va='center')

    idx = 0
    for key in method_dict.keys():
        FAR = method_dict[key][0]
        PD = method_dict[key][1]
        threshold = method_dict[key][2]
        color = color_list[idx]
        idx += 1

        if key == 'OT-AD':
            ax.plot(threshold, FAR, PD, color=color, lw=3, label=key)
        else:
            ax.plot(threshold, FAR, PD, color=color, lw=2.2, label=key)
        plt.legend(loc="right", prop={'size': 10})

        # Calculate AUC
        roc_auc = auc(FAR, PD)
        print(key + ':%.4f' % roc_auc)

    for start, end in edges2:
        ax.plot([corners[start][0], corners[end][0]],
                [corners[start][1], corners[end][1]],
                [corners[start][2], corners[end][2]],
                color='black', linewidth=0.75)

    save_name = os.path.join(save_dir, file + '.png')
    plt.savefig(save_name, format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close()


