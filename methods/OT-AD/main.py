import os
import time
import pywt
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import scipy.io as sio
from thop import profile
from typing import Optional
from progress.bar import Bar
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from dataset import Dataset, BlockRestore
from utils import seed_dic, hyper_norm
from net import Net
dtype = torch.cuda.FloatTensor
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0")


def wavelet_denoise_2d_band_level(band_img: np.ndarray, wavelet: str = 'db1', level: int = 2,
                            thresh_type: str = 'soft', thresh: Optional[float] = None) -> np.ndarray:
    """
    band_img: 2D numpy array (H, W) - Single band image
    Return the denoised image of the same size.
    thresh_type: 'soft' or 'hard'
    thresh: Custom threshold. If None, a threshold based on noise estimation will be used.
    """
    coeffs = pywt.wavedec2(band_img, wavelet=wavelet, mode='symmetric', level=level)
    # coeffs[0] represents the approximate coefficient, followed by the three detailed coefficients (cH, cV, cD) for each scale.
    approx = coeffs[0]

    # Calculate the threshold for each layer (if adaptive is required).
    if thresh is None:
        thresh_layers = []
        for i in range(1, len(coeffs)):
            details = coeffs[i]  
            t_h = t_v = t_d = 0.0
            # Calculate the MAD-based threshold for each subband.
            dH = details[0]
            if dH.size != 0:
                flatH = dH.ravel()
                madH = np.median(np.abs(flatH - np.median(flatH)))
                t_h = madH / 0.6745

            dV = details[1]
            if dV.size != 0:
                flatV = dV.ravel()
                madV = np.median(np.abs(flatV - np.median(flatV)))
                t_v = madV / 0.6745

            dD = details[2]
            if dD.size != 0:
                flatD = dD.ravel()
                madD = np.median(np.abs(flatD - np.median(flatD)))
                t_d = madD / 0.6745

            thresh_layers.append((t_h, t_v, t_d))

    else:
        thresh_layers = [(thresh, thresh, thresh) for _ in range(len(coeffs)-1)]

    # Thresholding detail coefficients.
    new_coeffs = [approx]
    for i in range(1, len(coeffs)):
        detail = coeffs[i]  # (cH, cV, cD)
        t_h, t_v, t_d = thresh_layers[i - 1]
        if thresh_type == 'soft':
            detail_thresh = (
                pywt.threshold(detail[0], t_h, mode='soft'),
                pywt.threshold(detail[1], t_v, mode='soft'),
                pywt.threshold(detail[2], t_d, mode='soft')
            )

        elif thresh_type == 'hard':
            detail_thresh = (
                pywt.threshold(detail[0], t_h, mode='hard'),
                pywt.threshold(detail[1], t_v, mode='hard'),
                pywt.threshold(detail[2], t_d, mode='hard')
            )
        else:
            raise ValueError("thresh_type must be 'soft' or 'hard'")
            
        new_coeffs.append(detail_thresh)

    # Reconstruction
    band_denoised = pywt.waverec2(new_coeffs, wavelet=wavelet, mode='symmetric')
    # The function waverec2 may return an array that is slightly larger than the original image (due to border padding, etc.), so crop it to the original size.
    H, W = band_img.shape
    if band_denoised.shape[0] != H or band_denoised.shape[1] != W:
        band_denoised = band_denoised[:H, :W]

    try:
        if np.issubdtype(band_img.dtype, np.integer):
            return band_denoised.astype(np.float32)
        else:
            return band_denoised.astype(band_img.dtype)
    except Exception:
        return band_denoised


def wavelet_denoise_2d_band_total(band_img: np.ndarray, wavelet: str = 'db1', level: int = 2,
                            thresh_type: str = 'soft', thresh: Optional[float] = None) -> np.ndarray:
    """
    band_img: 2D numpy array (H, W) - Single band image
    Return the denoised image of the same size.
    thresh_type: 'soft' or 'hard'
    thresh: Custom threshold. If None, a threshold based on noise estimation will be used.
    """
    coeffs = pywt.wavedec2(band_img, wavelet=wavelet, mode='symmetric', level=level)
    # coeffs[0] represents the approximate coefficient, followed by the three detailed coefficients (cH, cV, cD) for each scale.
    approx = coeffs[0]

    # Calculate the threshold for each layer (if adaptive is required).
    if thresh is None:
        thresh_layers = []
        for i in range(1, len(coeffs)):
            details = coeffs[i]  
            # Flatten and concatenate the three sub-bands and then take the overall statistical quantity.
            detail_vals = np.concatenate([d.ravel() for d in details])
            if detail_vals.size == 0:
                t = 0.0
            else:
                # Roughly robust estimation of threshold.
                t = np.median(np.abs(detail_vals)) / 0.6745
            thresh_layers.append(t)
    else:
        thresh_layers = [thresh] * (len(coeffs) - 1)

    # Thresholding detail coefficients.
    new_coeffs = [approx]
    for i in range(1, len(coeffs)):
        detail = coeffs[i]  # (cH, cV, cD)
        t = thresh_layers[i - 1]
        if thresh_type == 'soft':
            detail_thresh = tuple(pywt.threshold(d, t, mode='soft') for d in detail)
        elif thresh_type == 'hard':
            detail_thresh = tuple(pywt.threshold(d, t, mode='hard') for d in detail)
        else:
            raise ValueError("thresh_type must be 'soft' or 'hard'")
        new_coeffs.append(detail_thresh)

    # Reconstruction
    band_denoised = pywt.waverec2(new_coeffs, wavelet=wavelet, mode='symmetric')
    # The function waverec2 may return an array that is slightly larger than the original image (due to border padding, etc.), so crop it to the original size.
    if band_denoised.shape != band_img.shape:
        band_denoised = band_denoised[:band_img.shape[0], :band_img.shape[1]]
    return band_denoised


def wavelet_denoise_3d_batch(X: torch.Tensor, device: torch.device,
                             wavelet: str = 'db1', level: int = 2,
                             thresh_type: str = 'soft', thresh: Optional[float] = None) -> torch.Tensor:
    N, C, H, W = X.shape
    X_cpu = X.detach().cpu().numpy()
    denoised_cpu = np.empty((N, C, H, W), dtype=np.float32)

    def _process_one(n: int, c: int):
        band_img = X_cpu[n, c, :, :].astype(np.float32)
        # Optional: wavelet_denoise_2d_band_level or wavelet_denoise_2d_band_total
        band_denoised = wavelet_denoise_2d_band_level(
            band_img,
            wavelet=wavelet,
            level=level,
            thresh_type=thresh_type,
            thresh=thresh
        )
        if band_denoised.shape != (H, W):
            band_denoised = band_denoised[:H, :W]
        return n, c, band_denoised.astype(np.float32)

    for n in range(N):
        for c in range(C):
            n_idx, c_idx, band_out = _process_one(n, c)
            denoised_cpu[n_idx, c_idx, :, :] = band_out

    denoised_tensor = torch.from_numpy(denoised_cpu).to(device)
    return denoised_tensor


def run_single_file(args):
    # Set random seeds
    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f'Processing: {args.file}')

    data_path = os.path.join(args.data_dir, f'{args.file}.mat')
    save_dir = os.path.join(args.save_total_dir, args.file)
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    mat = sio.loadmat(data_path)
    hsi = mat['data']
    gt = mat['map']

    # Normalization and wavelet decomposition
    img = hsi.transpose(2, 0, 1)  # (bands, height, width)
    img = hyper_norm(img)
    img = torch.from_numpy(img).type(dtype).unsqueeze(0)
    img = wavelet_denoise_3d_batch(img, device=device, level=1)
    bands, height, width = img.size()[1:]

    # DataLoader setup
    block_size = args.patch_size * args.patch_grid
    data_set = Dataset(img, block_size=block_size, stride=args.stride)
    block_restore = BlockRestore(block_size=block_size, stride=args.stride)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

    # Model
    net = Net(
        input_channels=bands,
        embedding_dim=64,
        patch_size=args.patch_size,
        patch_grid=args.patch_grid,
        num_heads=args.num_heads,
        mlp_ratio=2.0,
        attn_drop_ratio=0.0,
        drop_ratio=0.0
    ).cuda()


    l1_loss = nn.L1Loss().type(dtype)
    optimizer = torch.optim.Adam(list(net.parameters()), lr=args.lr)

    # Train
    end_iter = args.num_iterations
    bar = Bar('Training', max=end_iter)
    start_time = time.time()
    for i in range(1, end_iter + 1):
        for data in train_loader:
            optimizer.zero_grad()
            net_in = data['block_input'].cuda()
            net_gt = data['block_gt'].cuda()
            net_out, w_loss = net(net_in)
            loss = l1_loss(net_out, net_gt) + w_loss
            loss.backward()
            optimizer.step()
        bar.next()

        # Test
        if i == end_iter:
            bar.finish()
            print(f"Finished training iteration: {i}")
            net.eval()
            if 'airport' in args.file:
                file_name = ['airport-1', 'airport-2', 'airport-3', 'airport-4']

            c = [z for z in file_name if z not in args.file]
            for file_test in c:
                data_path = args.data_dir + file_test + '.mat'
                
                save_dir = os.path.join(args.save_total_dir, "OT_AD", args.file)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            
                # Load data
                mat = sio.loadmat(data_path)
                hsi = mat['data']
                gt = mat['map']
            
                # Normalization and wavelet decomposition
                img = hsi.transpose(2, 0, 1)  # (bands, height, width)
                img = hyper_norm(img)
                img = torch.from_numpy(img).type(dtype).unsqueeze(0)
                img = wavelet_denoise_3d_batch(img, device=device, level=1)
                bands, height, width = img.size()[1:]
            
                # DataLoader setup
                block_size = args.patch_size * args.patch_grid
                data_set = Dataset(img, block_size=block_size, stride=args.stride)
                test_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
                res_map = []

                with torch.no_grad():
                    for data in test_loader:
                        test_in = data['block_input']
                        test_out = net(test_in)
                        res = torch.abs(test_in - test_out[0])
                        res_map.append(res)

                res_map = torch.cat(res_map, dim=0)
                res_map = block_restore(res_map, data_set.padding, height, width)
                res_map = res_map[0].sum(0)
                res_map = res_map.detach().cpu().numpy()
                res_map = hyper_norm(res_map)

                # Calculate AUC
                print(gt.shape)
                print(res_map.shape)
                auc_score = roc_auc_score(gt.flatten(), res_map.flatten())
                print(f'AUC: {auc_score:.4f}')

                # Calculation time consumption
                total_time = time.time() - start_time
                print(f"Runtime: {total_time:.2f} seconds")
                print(f"-------------------------------------------------------------------------------------------")

                # Save the anomaly detection map and the ROC curve
                fpr, tpr, thre = roc_curve(gt.flatten(), res_map.flatten())
                sio.savemat(os.path.join(save_dir, "OT_AD_map_"+file_test+".mat"), {'map': res_map})
                sio.savemat(os.path.join(save_dir, "OT_AD_roc_"+file_test+".mat"), {'PD': tpr, 'PF': fpr, 'thre':thre})

            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperspectral anomaly detection with multiple files')
    parser.add_argument('--data_dir', type=str, default='../../data/', help='Data directory')
    parser.add_argument('--save_total_dir', type=str, default='../../results/', help='Results save directory')
    parser.add_argument('--patch_size', type=int, default=3, help='Patch size')
    parser.add_argument('--patch_grid', type=int, default=5, help='Patch grid size')
    parser.add_argument('--stride', type=int, default=5, help='Sliding window stride')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_iterations', type=int, default=150, help='Training iterations')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of heads in transformer')
    parser.add_argument('--files', type=str, default='abu-airport-4, abu-beach-3, abu-urban-2, abu-urban-3, abu-urban-4, hydice',
                        help='Comma-separated list of files to process')

    args = parser.parse_args()

   
    for file in ['airport-1', 'airport-2', 'airport-3', 'airport-4']:
        args.file = file
        run_single_file(args)





