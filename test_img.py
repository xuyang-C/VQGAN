import os
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from pytorch_msssim import ms_ssim


def load_images_from_folder(folder):
    image_paths = []
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path):
                image_paths.append(img_path)
    return image_paths


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    psnr = peak_signal_noise_ratio(img1, img2, data_range=1.0)
    return psnr


def calculate_msssim(img1, img2):
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    msssim_value = ms_ssim(img1, img2, data_range=1.0).item()
    return msssim_value


def main(folder1, folder2, modify_path_func):
    image_paths1 = load_images_from_folder(folder1)

    psnr_values = []
    msssim_values = []

    for img_path1 in image_paths1:
        img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)

        img_path2 = modify_path_func(img_path1, folder1, folder2)
        if not os.path.isfile(img_path2):
            print(f"Image not found: {img_path2}")
            continue

        img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

        psnr = calculate_psnr(img1, img2)
        msssim = calculate_msssim(img1, img2)

        psnr_values.append(psnr)
        msssim_values.append(msssim)

    avg_psnr = np.mean(psnr_values)
    avg_msssim = np.mean(msssim_values)

    print(f"Average PSNR: {avg_psnr}")
    print(f"Average MSSSIM: {avg_msssim}")


# Example path modification function
def modify_path(img_path, folder1, folder2):
    return img_path.replace(folder1, folder2)


# Example usage:
folder1 = '/home/xuyang/VQGAN-pytorch/data/1ki58847CNs/frames/test/after5s/'
folder2 = '/home/xuyang/VQGAN-pytorch/data/1ki58847CNs/frames/reconstruct/after5s/'
main(folder1, folder2, modify_path)
