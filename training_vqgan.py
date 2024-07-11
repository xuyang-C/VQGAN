import os, datetime, sys
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
# from discriminator import Discriminator
from discriminator import NLayerDiscriminator as Discriminator
from taming.modules.losses.lpips import LPIPS
# from vqgan import VQGAN
from taming.models.vqgan import VQModel
from utils import load_data, weights_init
from torch.utils.tensorboard import SummaryWriter
print(os.getcwd())
class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQModel(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.quantize.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        if args.resume:
            self.vqgan.init_from_ckpt(args.resume)
            print(f"Resuming training from epoch {args.resume}")
        if args.reload:
            self.vqgan.load_checkpoint(args.reload)
            print(f"Reload model from {args.reload}")
        # torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_0.pt"))
        # self.vqgan.load_checkpoint(os.path.join("checkpoints", f"vqgan_epoch_25.pt"))
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(f'tensorboard_logs/ffhq_vqgan/{current_time}')

        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    try:
                        imgs, _ = imgs # [batch_size, channels, height, width]
                        imgs = imgs.to(device=args.device)
                        decoded_images, q_loss = self.vqgan(imgs)

                        disc_real = self.discriminator(imgs)
                        disc_fake = self.discriminator(decoded_images)

                        disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start) # 达到一定步数后，开始训练判别器

                        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                        rec_loss = torch.abs(imgs - decoded_images)
                        perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                        perceptual_rec_loss = perceptual_rec_loss.mean()
                        g_loss = -torch.mean(disc_fake)

                        disc_weight = self.vqgan.calculate_adaptive_weight(perceptual_rec_loss, g_loss, last_layer=self.vqgan.get_last_layer())
                        # disc_weight = 0.1
                        vq_loss = perceptual_rec_loss + q_loss + disc_factor * disc_weight * g_loss # 感知重建损失+ 量化损失 + 生成器损失

                        d_loss_real = torch.mean(F.relu(1. - disc_real)) # 真实图片的损失 使真实样本的判别器输出接近1
                        d_loss_fake = torch.mean(F.relu(1. + disc_fake)) # 生成图片的损失 使虚假样本的判别器输出接近-1
                        gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake) # 判别器损失

                        writer.add_scalar('VQ_Loss', vq_loss, epoch * steps_per_epoch + i)
                        writer.add_scalar('GAN_Loss', gan_loss, epoch * steps_per_epoch + i)
                        writer.add_scalar('Perceptual_rec_loss', perceptual_rec_loss, epoch * steps_per_epoch + i)
                        writer.add_scalar('rec_loss', rec_loss.mean(), epoch * steps_per_epoch + i)
                        writer.add_scalar('perceptual_loss', perceptual_loss.mean(), epoch * steps_per_epoch + i)
                        writer.add_scalar('q_loss', q_loss, epoch * steps_per_epoch + i)
                        writer.add_scalar('g_loss', g_loss, epoch * steps_per_epoch + i)
                        writer.add_scalar('d_loss_real', d_loss_real, epoch * steps_per_epoch + i)
                        writer.add_scalar('d_loss_fake', d_loss_fake, epoch * steps_per_epoch + i)
                        writer.add_scalar('disc_factor', disc_factor, epoch * steps_per_epoch + i)
                        writer.add_scalar('disc_weight', disc_weight, epoch * steps_per_epoch + i)

                        self.opt_vq.zero_grad()
                        vq_loss.backward(retain_graph=True)

                        self.opt_disc.zero_grad()
                        gan_loss.backward()

                        self.opt_vq.step()
                        self.opt_disc.step()

                        if i % 500 == 0:
                            with torch.no_grad():
                                real_fake_images = torch.cat((imgs[:4], decoded_images[:4]))
                                vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)

                        pbar.set_postfix(
                            epoch=epoch,
                            VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                            GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                        )
                        pbar.update(0)
                    except KeyboardInterrupt:
                        torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_last_{epoch}.pt"))
                        print("Interrupted, model saved.")
                        sys.exit(0)
            if epoch % 5 == 0 or epoch== args.epochs - 1:
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=224, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/home/xuyang/VQGAN-pytorch/data/face224/', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=30000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--resume', type=str, default= 'coco_oi_epoch12.ckpt', help='Restore model address.') # 'coco_oi_epoch12.ckpt' 'coco_epoch117.ckpt'
    parser.add_argument('--reload', type=str, default= None, help='Reload model from checkpoint')

    args = parser.parse_args()

    train_vqgan = TrainVQGAN(args)
    # train_vqgan.train(args)


