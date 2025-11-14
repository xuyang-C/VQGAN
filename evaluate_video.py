import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips
import glob
import os
import torch
from tqdm import tqdm  # 导入 tqdm
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
# 初始化设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='vgg').to(device)
# 初始化 LPIPS 模型，并将其移到 GPU（如果有的话）


def to_rgb01_torch(frame_bgr_np):
    # OpenCV 读出是 BGR uint8，先转 RGB，再归一化到 [0,1]，再转 torch
    rgb = cv2.cvtColor(frame_bgr_np, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float() / 255.0  # HWC, [0,1]
    return t.to(device)

def calculate_psnr(original_frame, output_frame):
    """
    计算 PSNR
    :param original_frame: 原始帧（numpy 数组）
    :param output_frame: 输出帧（numpy 数组）
    :return: PSNR 值
    """
    mse = F.mse_loss(original_frame, output_frame)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


def calculate_ssim(original_frame, output_frame):
    """
    计算 SSIM
    :param original_frame: 原始帧（numpy 数组）
    :param output_frame: 输出帧（numpy 数组）
    :return: SSIM 值
    """
    return ssim(original_frame, output_frame, multichannel=True)

def calculate_msssim(img1, img2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0).float()
    img2 = img2.permute(2, 0, 1).unsqueeze(0).float()
    msssim_value = ms_ssim(img1, img2, data_range=1.0).item()
    return msssim_value


def calculate_lpips(original_frame, output_frame):
    """
    计算 LPIPS
    :param original_frame: 原始帧（numpy 数组）
    :param output_frame: 输出帧（numpy 数组）
    :return: LPIPS 值
    """
    img1 = original_frame.permute(2, 0, 1).unsqueeze(0).float()
    img2 = output_frame.permute(2, 0, 1).unsqueeze(0).float()
    lpips_value = lpips_fn(img1, img2, normalize=True).item()
    return lpips_value

def evaluate_videos(original_video_path, output_video_path):
    # 读取视频
    original_video = cv2.VideoCapture(original_video_path)
    output_video = cv2.VideoCapture(output_video_path)
    # 获取视频帧数
    total_frames = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    psnr_values = []
    msssim_values = []
    lpips_values = []
    skipped_frames = 0

    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(output_video_path)}") as pbar:
        while True:
            ret1, original_frame = original_video.read()
            ret2, output_frame = output_video.read()

            # 如果有一个视频结束，退出
            if not ret1 or not ret2:
                break

            # 检查是否损坏的帧（例如，输出帧为空或解码失败）
            if output_frame is None:
                skipped_frames += 1
                psnr_values.append(0)  # 将损坏帧的 PSNR 设置为 0
                msssim_values.append(0)  # 将损坏帧的 SSIM 设置为 0
                lpips_values.append(1)  # 将损坏帧的 LPIPS 设置为 1
                continue  # 跳过这帧
            original_frame = to_rgb01_torch(original_frame)
            output_frame = to_rgb01_torch(output_frame)
            # 计算 PSNR
            psnr_value = calculate_psnr(original_frame, output_frame)
            psnr_values.append(psnr_value)

            # 计算 SSIM
            msssim_value = calculate_msssim(original_frame, output_frame)
            msssim_values.append(msssim_value)

            # 计算 LPIPS
            lpips_value = calculate_lpips(original_frame, output_frame)
            lpips_values.append(lpips_value)  # .item() 提取数值

            frame_count += 1
            pbar.update(1)  # 更新进度条
        original_video.release()
        output_video.release()
    # 计算平均 PSNR, SSIM 和 LPIPS
    avg_psnr = np.mean(psnr_values) if psnr_values else float('nan')
    avg_msssim = np.mean(msssim_values) if msssim_values else float('nan')
    avg_lpips = np.mean(lpips_values) if lpips_values else float('nan')

    print(f"PSNR: {avg_psnr}")
    print(f"MS-SSIM: {avg_msssim}")
    print(f"LPIPS: {avg_lpips}")
    print(f"跳过了 {skipped_frames} 帧")

    # 释放视频对象
    original_video.release()
    output_video.release()
    return avg_psnr, avg_msssim, avg_lpips
def batch_evaluate(original_video_path, folder_path):
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    all_psnr, all_msssim, all_lpips = [], [], []

    for video_path in video_files:
        if os.path.abspath(video_path) == os.path.abspath(original_video_path):
            continue  # 跳过原始视频
        psnr_val, msssim_val, lpips_val = evaluate_videos(original_video_path, video_path)
        all_psnr.append(psnr_val)
        all_msssim.append(msssim_val)
        all_lpips.append(lpips_val)

    print("\n====== 最终平均结果 ======")
    print(f"平均 PSNR: {np.nanmean(all_psnr):.4f}")
    print(f"平均 MS-SSIM: {np.nanmean(all_msssim):.4f}")
    print(f"平均 LPIPS: {np.nanmean(all_lpips):.4f}")
# 用法示例
original_video_path = "/home/xuyang/taming-transformers-master/data/UVG/UVG_h265_1920x1080/original/Beauty_265.mp4"
output_video_folder = "/home/xuyang/taming-transformers-master/data/UVG/UVG_h265_1920x1080/FEC01_GE_outputs"

batch_evaluate(original_video_path, output_video_folder)
