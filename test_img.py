import os
import sys

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from pytorch_msssim import ms_ssim
from tqdm import tqdm
import torch.nn.functional as F
import lpips
from cleanfid import fid
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
from DISTS_pytorch import DISTS
D = DISTS().to('cuda')
def load_images_from_folder(folder):
    jpg_file = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".png"):
                jpg_file.append(os.path.join(root, file))
    return jpg_file


def calculate_psnr(img1, img2):
    # img1 = img1 / 255.0
    # img2 = img2 / 255.0
    # img1 = torch.from_numpy(img1)
    # img2 = torch.from_numpy(img2)
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


def calculate_msssim(img1, img2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0).float()
    img2 = img2.permute(2, 0, 1).unsqueeze(0).float()
    msssim_value = ms_ssim(img1, img2, data_range=1.0).item()
    return msssim_value

def calculate_lpips(img1, img2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0).float()
    img2 = img2.permute(2, 0, 1).unsqueeze(0).float()
    lpips_value = lpips_fn(img1, img2, normalize=True).item()
    return lpips_value

def calculate_dists(img1, img2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0).float()
    img2 = img2.permute(2, 0, 1).unsqueeze(0).float()
    dists_value = D(img1, img2, batch_average=True).item()
    return dists_value

def fid_kid(folder1, folder2):
    """compute FID, KID based on previously extracted patches"""


    # compute FID, KID
    print('\ncomputing FID, KID...')
    fid_score = fid.compute_fid(folder1, folder2)
    kid_score = fid.compute_kid(folder1, folder2)
    print(f'\nFID-score: {fid_score}\nKID-score: {kid_score}')
    print('Done!')

def main(folder1, folder2, modify_path_func):
    image_paths1 = load_images_from_folder(folder1)

    psnr_values = []
    msssim_values = []
    lpips_values = []
    dists_values = []
    # fid_kid(folder1, folder2)
    # sys.exit(0)
    for img_path1 in tqdm(image_paths1, desc="Calculating Metrics"):
        img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
        img_path2 = modify_path_func(img_path1, folder1, folder2)
        if not os.path.isfile(img_path2):
            print(f"Image not found: {img_path2}")
            psnr_values.append(0.)
            msssim_values.append(0.)
            lpips_values.append(1.)
            dists_values.append(0.8)
            continue

        img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
        img1 = torch.from_numpy(img1/255.0).to('cuda')
        img2 = torch.from_numpy(img2/255.0).to('cuda')
        psnr = calculate_psnr(img1, img2)
        msssim = calculate_msssim(img1, img2)
        lpips_ = calculate_lpips(img1, img2)
        # dists_ = calculate_dists(img1, img2)
        # print(psnr, msssim)
        psnr_values.append(psnr)
        msssim_values.append(msssim)
        lpips_values.append(lpips_)
        # dists_values.append(dists_)
    avg_psnr = np.mean(psnr_values)
    avg_msssim = np.mean(msssim_values)
    avg_lpips = np.mean(lpips_values)
    avg_dist = np.mean(dists_values)
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average MSSSIM: {avg_msssim}")
    print(f"Average LPIPS: {avg_lpips}")
    # print(f"Average DISTS: {avg_dist}")
# Example path modification function
def modify_path(img_path, folder1, folder2):
    return img_path.replace(folder1, folder2)


# Example usage:
folder1 = '/home/xuyang/taming-transformers-master/data/UVG/UVG_h265_1920x1080/original_frames/'
folder2 = '/home/xuyang/taming-transformers-master/data/UVG/UVG_h265_1920x1080/1692kb_frames/'
main(folder1, folder2, modify_path)
