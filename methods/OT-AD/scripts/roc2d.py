import os 
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import auc


manual_colors = ['#cc6b5f', '#c03b4c', '#2e80b5',
                 '#46668a', '#456f97', '#d3b6c8',
                 '#8d7e95', '#545166', '#988196', '#456F97', '#C03B4C']
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
    for method in method_list:
        mat_name = os.path.join(mat_dir, 'ROC_curve_' + method + '.mat')
        if not os.path.exists(mat_name):
            continue
        mat = sio.loadmat(mat_name)
        tpr = mat['PD'] 
        fpr = mat['PF']
        method_dict[method] = [tpr, fpr]

    # Draw roc fig
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_xscale("log", base=10)
    ax.grid(False)
    ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(0.0, 1.05)
    plt.xlabel('False Alarm Rate', fontsize=15)
    plt.ylabel('Probability of Detection', fontsize=15)
    plt.xticks(size=12)
    plt.yticks(size=12)

    idx = 0
    for key in method_dict.keys():
        fpr = method_dict[key][1]
        tpr = method_dict[key][0]
        color = color_list[idx]
        idx += 1
        if len(tpr.shape) == 2 and tpr.shape[0] == 1:
            fpr = fpr[0]
            tpr = tpr[0]
        if key == 'OT-AD':
            ax.semilogx(fpr, tpr, color=color, lw=3.5, label=key)
        else:
            ax.semilogx(fpr, tpr, color=color, lw=2.2, label=key)
        plt.legend(loc="lower right", prop={'size': 12})

        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        print(key + ':%.4f' % roc_auc)

    save_name = os.path.join(save_dir, file + '.png')
    plt.savefig(save_name, format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close() 


