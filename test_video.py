import subprocess as sp
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
import lpips
import glob, os
from DISTS_pytorch import DISTS
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_fn = lpips.LPIPS(net='vgg').to(device)
D = DISTS().to(device)


def ffmpeg_video_reader(video_path, width, height):
    """
    用 ffmpeg 流式读取视频帧 (RGB, HWC, np.uint8)
    """
    command = [
        "ffmpeg",
        "-i", video_path,
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-"
    ]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    frame_size = width * height * 3
    while True:
        raw_frame = pipe.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            break
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        yield frame
    pipe.stdout.close()
    pipe.wait()


def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


def calculate_msssim(img1, img2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0).float()
    img2 = img2.permute(2, 0, 1).unsqueeze(0).float()
    return ms_ssim(img1, img2, data_range=1.0).item()


def calculate_lpips(img1, img2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0).float()
    img2 = img2.permute(2, 0, 1).unsqueeze(0).float()
    return lpips_fn(img1, img2, normalize=True).item()


def calculate_dists(img1, img2):
    img1 = img1.permute(2, 0, 1).unsqueeze(0).float()
    img2 = img2.permute(2, 0, 1).unsqueeze(0).float()
    return D(img1, img2, batch_average=True).item()


def evaluate_videos(original_path, output_path, width, height):
    psnr_values, msssim_values, lpips_values, dists_values = [], [], [], []

    gen1 = ffmpeg_video_reader(original_path, width, height)
    gen2 = ffmpeg_video_reader(output_path, width, height)

    for f1, f2 in tqdm(zip(gen1, gen2), desc="Calculating Metrics"):
        f1 = torch.from_numpy(f1 / 255.0).to(device)
        f2 = torch.from_numpy(f2 / 255.0).to(device)

        psnr_values.append(calculate_psnr(f1, f2))
        msssim_values.append(calculate_msssim(f1, f2))
        lpips_values.append(calculate_lpips(f1, f2))
        # dists_values.append(calculate_dists(f1, f2))  # 可选
    avg_psnr = np.mean(psnr_values)
    avg_msssim = np.mean(msssim_values)
    avg_lpips = np.mean(lpips_values)
    print("平均结果:")
    print(f"PSNR: {np.mean(psnr_values):.4f}")
    print(f"MS-SSIM: {np.mean(msssim_values):.4f}")
    print(f"LPIPS: {np.mean(lpips_values):.4f}")
    if dists_values:
        print(f"DISTS: {np.mean(dists_values):.4f}")
    return avg_psnr, avg_msssim, avg_lpips

def batch_evaluate_videos(original_video_path, output_folder, width, height):
    video_files = glob.glob(os.path.join(output_folder, "*.mp4"))
    all_psnr, all_msssim, all_lpips = [], [], []
    for video_path in video_files:
        if os.path.abspath(video_path) == os.path.abspath(original_video_path):
            continue  # 跳过原始视频
        psnr_val, msssim_val, lpips_val = evaluate_videos(original_video_path, video_path, width, height)
        all_psnr.append(psnr_val)
        all_msssim.append(msssim_val)
        all_lpips.append(lpips_val)

    print("\n====== 最终平均结果 ======")
    print(f"平均 PSNR: {np.mean(all_psnr):.4f}")
    print(f"平均 MS-SSIM: {np.mean(all_msssim):.4f}")
    print(f"平均 LPIPS: {np.mean(all_lpips):.4f}")
# 示例调用
original_video = "/home/xuyang/taming-transformers-master/data/UVG/UVG_h265_1920x1080/original/Beauty_265.mp4"
output_video = "/home/xuyang/taming-transformers-master/data/UVG/UVG_h265_1920x1080/FEC100%_GE75%_outputs/"
width, height = 1920, 1080  # 注意要写对分辨率！

batch_evaluate_videos(original_video, output_video, width, height)
