from taming.models.vqgan import VQModel as VQGAN
import torch, os
from tqdm import tqdm
import argparse
import time
from utils import load_test_data
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torchvision.utils import save_image
from transformer import VQGANTransformer
from utils import load_video_data
import lpips
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
from DISTS_pytorch import DISTS
D = DISTS().to('cuda')
def test(args):
    model = VQGANTransformer(args).to(device=args.device)
    # model.load_state_dict(torch.load(args.reload))
    print(f"Reload model from {args.reload}")
    model = model.eval()
    test_dataset = load_video_data(args, shuffle=False)
    psnr_list = []
    msssim_list = []
    lpips_list = []
    masked_accuracy_list = []
    dists_list = []
    byte_list = []
    # save_path = '/home/xuyang/VQGAN-pytorch/transformer_recon_imgs_vivit_02_loss/'
    # os.makedirs(save_path, exist_ok=True)
    start_time = time.time()
    with tqdm(range(len(test_dataset))) as pbar: # 由于需要保存图片，batchsize=1
        for i, imgs in zip(pbar, test_dataset):
        # for i in range(len(test_dataset)):
        #     imgs = imgs_and_paths[0]
        #     paths = imgs_and_paths[1][0] # paths为tuple，取第一个元素
        #     modified_path = paths.replace('test', 'reconstruct')
            imgs = imgs.to(device=args.device)
            logits, targets, reconstruct, mask, masked_accuracy = model(imgs)

            # reconstruct_ = (reconstruct+1.)/2.
            # for j, img in enumerate(reconstruct_):
            #     save_image(img, os.path.join(save_path, f"{i}_{j}.png"))
            '''calculate PSNR and MS-SSIM'''
            # mse = torch.mean((imgs[:,-1] - reconstruct) ** 2,dim=[1,2,3])
            # psnr = 10 * torch.log10(2**2 / mse)
            # psnr_avg = torch.mean(psnr).item()
            # msssim = ms_ssim(imgs[:,-1], reconstruct, data_range=2, size_average=True).item()
            # lpips_ = lpips_fn(imgs[:,-1], reconstruct, normalize=False).squeeze().mean().item()
            # dists_value = D((imgs[:,-1]+1)/2, (reconstruct+1)/2, batch_average=True).item()
            # psnr_list.append(psnr_avg)
            # msssim_list.append(msssim)
            # lpips_list.append(lpips_)
            # dists_list.append(dists_value)
            # masked_accuracy_list.append(masked_accuracy.item())
    print(f"Decoding Time: {model.decode_time}")
    end_time = time.time()
    print(f"Test time: {end_time-start_time}")
    # print(f'PSNR: {sum(psnr_list)/len(psnr_list)}, MS-SSIM: {sum(msssim_list)/len(msssim_list)}, LPIPS: {sum(lpips_list)/len(lpips_list)}, DISTS: {sum(dists_list)/len(dists_list)}, Masked Accuracy: {sum(masked_accuracy_list)/len(masked_accuracy_list)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=240, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=128, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='/home/xuyang/taming-transformers-master/data/vimeo/',
                        help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str,
                        default='/home/xuyang/taming-transformers-master/logs/HorseRiding_EMA_128_nembed/epoch=220-step=482129.ckpt',
                        help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=3, help='Input batch size for training.')
    parser.add_argument('--num-frames', type=int, default=5, help='Number of frames in the video.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                        help='Weighting factor for perceptual loss.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
    parser.add_argument('--reload', type=str,
                        default=None, #'/home/xuyang/VQGAN-pytorch/checkpoints_transformer_04_loss/transformer_248.pt',
                        help='Reload model from checkpoint')
    args = parser.parse_args()
    test(args)