from model import LG_MoSE
from torch.optim import Adam
from torch.optim import lr_scheduler
import argparse
import torch
import utils
import clip
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.io as sio
import os
import random
from SeT import (
    TotalLoss,
    Mask,
    separation_training
)
from SeT.train import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.cuda.FloatTensor
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(data_train, opt):

    num_bs = opt.bands
    scene_type = opt.scene_type
    data_dir = opt.data_dir
    save_dir = opt.save_dir
    num_layers = opt.num_layers
    lr = opt.lr
    epochs = opt.epochs
    output_iter = opt.output_iter
    max_iter = opt.max_iter
    data_norm = opt.data_norm
    Net = LG_MoSE
    net_kwargs = dict()
    net_kwargs['num_layers'] = num_layers

    # Load data
    dataset = data_train
    data_path = data_dir + data_train + '.mat'
    mat = sio.loadmat(data_path)
    data = mat['data'].astype(float)
    gt = mat['map'].astype(bool)
    print('The shape of original HSI is : ' ,data.shape)
    rows, cols, bands = data.shape
    net_kwargs['shape'] = (rows, cols, num_bs)
    textual_description = {'Airport' : 'The background components of an airport are typically composed of runways, taxiways, and terminals.',
                           'Beach' : 'The background components of a beach are typically composed of sand, sea, and shoreline.'}
    text_prompt = textual_description[scene_type]
    print('Detecting on %s...' % data_train)

    if data_norm:
        data_bs = utils.ZScoreNorm().fit(data).transform(data)

    # Load model
    model = Net(**net_kwargs).to(device).float()
    
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    clip_model.eval()
    text_encoder = clip_model.encode_text
    text_token = clip.tokenize(text_prompt).to(device)
    text_feature = text_encoder(text_token).float().detach()

    # Loss
    loss = TotalLoss(device)

    # Mask
    mask = Mask((rows, cols), device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # Separation Training
    x_bs = torch.from_numpy(data_bs).to(device).float()

    pr_dm, history = separation_training(
        x=x_bs,
        gt=gt,
        text_feature = text_feature,
        model=model,
        loss=loss,
        mask=mask,
        optimizer=optimizer,
        epochs=epochs,
        output_iter=output_iter,
        max_iter=max_iter,
        verbose=True
    )

    # Cal auc
    auc = roc_auc_score(gt.flatten(), pr_dm.flatten())
    print('Auc: %.4f' % auc)
    
    # Normalization
    pr_dm = (pr_dm - pr_dm.min()) / (pr_dm.max() - pr_dm.min())
    norm_auc = roc_auc_score(gt.flatten(), pr_dm.flatten())
    print('Norm_Auc: %.4f' % norm_auc)
    test(device, data_norm, text_feature, model, data_train, data_dir, save_dir)

    return 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Hyperspectral anomaly detection with multiple files')
    parser.add_argument('--data_dir', type=str, default='../../data/', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='../../results/', help='Results save directory')
    parser.add_argument('--bands', type=int, default=180, help='Number of bands')
    parser.add_argument('--scene_type', type=str, default='Airport', help='Scene type')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--output_iter', type=int, default=5, help='output iter')   
    parser.add_argument('--max_iter', type=int, default=5, help='Training iterations')
    parser.add_argument('--data_norm', type=bool, default=True, help='Data normalization')

    args = parser.parse_args()

    for data_train in ['airport-1', 'airport-2', 'airport-3', 'airport-4']:   # train dataset
        main(data_train, args)

