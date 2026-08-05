# https://github.com/xdjiangkai/LREN.
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'

import tensorflow as tf
import numpy as np
from sklearn.datasets import make_blobs
from datetime import datetime
import scipy.io as sio
from sklearn.decomposition import PCA, KernelPCA
from lren import LREN
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, roc_curve
from lrr.lrr import lrr
import time
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
data_dir = '../../data/'
save_dir = '../../results/'
    
def parameter_setting(file):
    Lam = 1.0
    epoch_size = 200
    clusters_num = 7
    hidden_nodes = 9

    return clusters_num, hidden_nodes, Lam, epoch_size

def main(file):
    # load data
    print(file)
    data_path = data_dir + file + '.mat'
    save_subdir = os.path.join(save_dir, file)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)

    clusters_num, hidden_nodes, Lam, epoch_size = parameter_setting(file)
    
    load_data = sio.loadmat(data_path)
    load_matrix = load_data['data']
    
    load_matrix = np.array(load_matrix)
    [r, c, x_dim]=load_matrix.shape
    load_matrix = load_matrix.reshape([load_matrix.shape[0]*load_matrix.shape[1], x_dim])
    load_matrix = ((load_matrix-load_matrix.min()) /
                        (load_matrix.max()-load_matrix.min()))
    data = load_matrix
    
    anomal_target_map = load_data['map']
    anomal_target_map = np.array(anomal_target_map)

    normal_data=data
    tf.compat.v1.reset_default_graph()
        
    start = time.time()
    model_lren = LREN([400,hidden_nodes], tf.nn.tanh, est_hiddens=[60,clusters_num], 
        est_activation=tf.nn.tanh, est_dropout_ratio=0.5, epoch_size=epoch_size, minibatch_size=int(4096)
    )

    model_lren.Perform_Density_Estimation(normal_data)
    
    print('Training on %s...' % file)
    if 'airport' in file:
        file_name = ['airport-1', 'airport-2', 'airport-3', 'airport-4']

    for file_test in file_name:
        data_path = data_dir + file_test + '.mat'
        save_subdir = os.path.join(save_dir, "LREN", file)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)
        print('Detecting on %s...' % file_test)
        mat = sio.loadmat(data_path)
        load_matrix = mat['data']
        load_matrix = np.array(load_matrix)
        [r, c, x_dim]=load_matrix.shape
        load_matrix = load_matrix.reshape([load_matrix.shape[0]*load_matrix.shape[1], x_dim])
        load_matrix = ((load_matrix-load_matrix.min()) /
                            (load_matrix.max()-load_matrix.min()))
        data = load_matrix
        
        anomal_target_map = mat['map']
        anomal_target_map = np.array(anomal_target_map)

        Dict, S = model_lren.construct_Dict(data)

        X,E,obj,err,Iter = lrr(S.T, Dict.T, Lam)

        energy = np.linalg.norm(E.T,axis=1,ord=2)
        energy = (energy-energy.min())/(energy.max()-energy.min())

        auc_score = roc_auc_score(anomal_target_map.flatten(), energy.flatten())
        print('Auc:%.4f' % auc_score)
        # running time
        end = time.time()
        print("runtime：%.2f" % (end - start))
        # save results
        fpr, tpr, thre = roc_curve(anomal_target_map.flatten(), energy.flatten())
        map_path = os.path.join(save_subdir, "LREN_map_"+file_test+".mat")
        sio.savemat(map_path, {'show': energy.reshape(anomal_target_map.shape)})
        roc_path = os.path.join(save_subdir, "LREN_roc_"+file_test+".mat")
        sio.savemat(roc_path, {'PD': tpr, 'PF': fpr, 'thre':thre})  

        
if __name__ == '__main__':
    for file in ['airport-1', 'airport-2', 'airport-3', 'airport-4']:
        main(file)