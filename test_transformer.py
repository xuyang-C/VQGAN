from taming.models.vqgan import VQModel as VQGAN
import torch, os
from tqdm import tqdm
import argparse
from utils import load_test_data
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torchvision.utils import save_image
from transformer import VQGANTransformer
from utils import load_video_data
def test(args):
    model = VQGANTransformer(args).to(device=args.device)
    model.load_state_dict(torch.load(args.reload))
    print(f"Reload model from {args.reload}")
    model = model.eval()
    test_dataset = load_video_data(args)
    psnr_list = []
    msssim_list = []
    save_path = 'data/1ki58847CNs/frames/reconstruct_transformer/after5s/'
    os.makedirs(save_path, exist_ok=True)
    with tqdm(range(len(test_dataset))) as pbar: # 由于需要保存图片，batchsize=1
        for i, imgs in zip(pbar, test_dataset):
        # for i in range(len(test_dataset)):
        #     imgs = imgs_and_paths[0]
        #     paths = imgs_and_paths[1][0] # paths为tuple，取第一个元素
        #     modified_path = paths.replace('test', 'reconstruct')
            imgs = imgs.to(device=args.device)
            logits, targets, reconstruct = model(imgs)
            save_image(reconstruct, os.path.join(save_path, f"{i}.jpg"), nrow=args.batch_size)
            '''calculate PSNR and MS-SSIM'''
            # imgs = transforms.CenterCrop((224, 224))(imgs)
            # decoded_images = transforms.CenterCrop((224, 224))(decoded_images)
            mse = torch.mean((imgs[:,-1] - reconstruct) ** 2)
            psnr = 10 * torch.log10(1 / mse).item()
            # ssim = 1 - torch.mean(torch.abs(imgs - decoded_images) / torch.abs(imgs))
            msssim = ms_ssim(imgs[:,-1], reconstruct, data_range=1, size_average=True).item()
            psnr_list.append(psnr)
            msssim_list.append(msssim)
    print(f'PSNR: {sum(psnr_list)/len(psnr_list)}, MS-SSIM: {sum(msssim_list)/len(msssim_list)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=224, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='/home/xuyang/VQGAN-pytorch/data/1ki58847CNs/frames/test-after5s/',
                        help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str,
                        default='/home/xuyang/VQGAN-pytorch/checkpoints-vox2-224/vqgan_last_58.pt',
                        help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training.')
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
                        default='/home/xuyang/VQGAN-pytorch/checkpoints_transformer/transformer_0.pt',
                        help='Reload model from checkpoint')
    args = parser.parse_args()
    test(args)