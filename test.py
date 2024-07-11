from taming.models.vqgan import VQModel as VQGAN
import torch, os
from tqdm import tqdm
import argparse
from utils import load_test_data
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torchvision.utils import save_image
def test(args):
    model = VQGAN(args).to(device=args.device)
    model.load_checkpoint(args.reload)
    print(f"Reload model from {args.reload}")
    model = model.eval()
    test_dataset = load_test_data(args)
    psnr_list = []
    msssim_list = []
    os.makedirs("data/1ki58847CNs/frames/reconstruct/after5s", exist_ok=True)
    with tqdm(range(len(test_dataset))) as pbar: # 由于需要保存图片，batchsize=1
        for i, imgs_and_paths in zip(pbar, test_dataset):
        # for i in range(len(test_dataset)):
            imgs = imgs_and_paths[0]
            paths = imgs_and_paths[1][0] # paths为tuple，取第一个元素
            modified_path = paths.replace('test', 'reconstruct')
            imgs = imgs.to(device=args.device)
            decoded_images, q_loss = model(imgs)
            save_image(decoded_images, modified_path ,nrow=args.batch_size)
            '''calculate PSNR and MS-SSIM'''
            # imgs = transforms.CenterCrop((224, 224))(imgs)
            # decoded_images = transforms.CenterCrop((224, 224))(decoded_images)
            mse = torch.mean((imgs - decoded_images) ** 2)
            psnr = 10 * torch.log10(1 / mse).item()
            # ssim = 1 - torch.mean(torch.abs(imgs - decoded_images) / torch.abs(imgs))
            msssim = ms_ssim(imgs, decoded_images, data_range=1, size_average=True).item()
            psnr_list.append(psnr)
            msssim_list.append(msssim)
    print(f'PSNR: {sum(psnr_list)/len(psnr_list)}, MS-SSIM: {sum(msssim_list)/len(msssim_list)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=224, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/home/xuyang/VQGAN-pytorch/data/1ki58847CNs/frames/test/', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=30000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--resume', type=str, default=None, help='Restore model address.')
    parser.add_argument('--reload', type=str, default='/home/xuyang/VQGAN-pytorch/checkpoints/vqgan_epoch_99.pt', help='Reload model from checkpoint')

    args = parser.parse_args()
    test(args)