import torch
from .detect import detect
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import scipy.io as sio
import os
import utils
import time

def roc_auc(dm: np.ndarray,
            gt: np.ndarray):
    rows, cols = gt.shape

    gt = gt.reshape(rows * cols)
    dm = dm.reshape(rows * cols)

    fpr, tpr, _ = metrics.roc_curve(gt, dm)
    auc = metrics.auc(fpr, tpr)

    return fpr, tpr, auc

def train_model(
        x,
        text_feature,
        model,
        criterion,
        cri_kwargs,
        epochs,
        optimizer,
        verbose):

    epoch_iter = iter(_ for _ in range(epochs))
    if verbose:
        epoch_iter = tqdm(list(epoch_iter))
    for _ in epoch_iter:

        # Clear gradient information
        optimizer.zero_grad()

        # Forward propagation
        y = model(x,text_feature)

        # Calculate loss
        loss = criterion(x=x, y=y, **cri_kwargs)

        # Backward propagation
        loss.backward()

        # Update network parameters
        optimizer.step()

        if verbose:
            epoch_iter.set_postfix({'loss': '{0:.4f}'.format(loss)})


def separation_training(
        x: torch.Tensor,
        gt: np.ndarray,
        text_feature: torch.Tensor,
        model,
        loss,
        mask,
        optimizer,
        epochs,
        output_iter,
        max_iter,
        verbose) -> (np.ndarray, list):
    """
    The main process of the separation training algorithm.

    """

    history = []
    output_dm = np.zeros_like(gt)

    for i in range(1, max_iter + 1):
        if verbose:
            print('Iter {0}'.format(i))

        # Feed the model with x
        model_input = x
        # Train the model for some epochs
        train_model(
            model_input,
            text_feature,
            model,
            loss,
            {'mask': mask},
            epochs,
            optimizer,
            verbose
        )

        # Update the mask using detection map obtained in this iteration
        model_output = model(model_input, text_feature)
        dm = detect(x, model_output)
        mask.update(dm.detach())

        # Evaluation
        np_dm = dm.cpu().detach().numpy()
        fpr, tpr, auc = roc_auc(np_dm, gt)
        if verbose:
            print('Current AUC score: {0:.4f}'.format(auc))

        # Record history
        history.append(auc)
        
    output_dm = np_dm

    return output_dm, history

def test(
    device,
    data_norm,
    text_feature,
    model,
    data_train,
    data_dir,
    save_dir
):
    model.eval()

    if 'airport' in data_train:
        data_name = ['airport-1', 'airport-2', 'airport-3', 'airport-4']

    c = [x for x in data_name if x not in data_train]
    z = 0.0
    for data_test in c:

        data_path = data_dir + data_test + '.mat'
        save_subdir = os.path.join(save_dir, "LG-MoSE", data_train)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)
        mat = sio.loadmat(data_path)
        data = mat['data'].astype(float)
        gt = mat['map'].astype(bool)
        print('The shape of original HSI is : ' ,data.shape)
        rows, cols, bands = data.shape
        print('Detecting on %s...' % data_test)
        
        if data_norm:
            data_bs = utils.ZScoreNorm().fit(data).transform(data)
        
        x = torch.from_numpy(data_bs).to(device).float()
        output_dm = np.zeros_like(gt)
        model_input = x
        model_output = model(model_input, text_feature)
        dm = detect(x, model_output)

        # Evaluation
        np_dm = dm.cpu().detach().numpy()
        
        # Cal auc
        auc = roc_auc_score(gt.flatten(), np_dm.flatten())
        print('Auc: %.4f' % auc)
        
        # Normalization
        np_dm = (np_dm - np_dm.min()) / (np_dm.max() - np_dm.min())
        norm_auc = roc_auc_score(gt.flatten(), np_dm.flatten())
        print('Norm_Auc: %.4f' % norm_auc)
        z += norm_auc
    
        # save results
        fpr, tpr, thre = roc_curve(gt.flatten(), np_dm.flatten())
        map_path = os.path.join(save_subdir, "LG-MoSE_map_"+data_test+".mat")
        sio.savemat(map_path, {'show': np_dm})
        roc_path = os.path.join(save_subdir, "LG-MoSE_roc_"+data_test+".mat")
        sio.savemat(roc_path, {'PD': tpr, 'PF': fpr, 'thre':thre})
    print('Average Auc: %.4f' % (z / 3))