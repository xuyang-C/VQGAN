import os,datetime
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_video_data, plot_images
from torch.utils.tensorboard import SummaryWriter
import sys
class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers(args)
        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_embedding_space")
        no_decay.add("pos_embedding_time")
        # no_decay.add("pos_embedding")
        # no_decay.add("pos_encoder.time_pe")
        # no_decay.add("pos_encoder.space_pe")
        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.05)
        return optimizer

    @staticmethod
    def prepare_training():
        os.makedirs("results_transformer", exist_ok=True)
        os.makedirs("checkpoints_transformer", exist_ok=True)

    # 添加标签平滑的修改方案
    def smooth_cross_entropy(self, logits, targets, smoothing=0.1):
        """
        logits: [batch*seq_len, num_classes]
        targets: [batch*seq_len]
        smoothing: 标签平滑系数 (0.0~1.0)
        """
        num_classes = logits.size(-1)

        # 创建平滑后的标签分布
        with torch.no_grad():
            smooth_targets = torch.empty_like(logits).fill_(smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

        # 计算交叉熵
        log_probs = F.log_softmax(logits, dim=-1)
        return (-smooth_targets * log_probs).sum(dim=-1).mean()
    def train(self, args):
        train_dataset = load_video_data(args, shuffle=True)
        steps_per_epoch = len(train_dataset)
        if args.reload:
            self.model.load_state_dict(torch.load(args.reload))
            print(f"Reload model from {args.reload}")
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(f'tensorboard_logs/transformer/{current_time}')
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    try:
                        self.optim.zero_grad()
                        imgs = imgs.to(device=args.device) # b,t,c,h,w
                        logits, targets, reconstruct, mask, masked_accuracy = self.model(imgs) # reconstruct: b,c,h,w logits:b,n,d targets:b,n
                        # loss = F.cross_entropy(logits.view(-1,logits.shape[-1]), targets.view(-1))
                        loss = self.smooth_cross_entropy(logits.view(-1,logits.shape[-1]), targets.view(-1), smoothing=0.1)
                        # rec_loss = torch.abs(imgs[:,-1] - reconstruct).mean() # 没什么帮助
                        # loss = 0.1*loss+ rec_loss
                        # # 生成有效位置掩码
                        # valid_mask = (mask == 0).float().view(-1)  # (B, N)
                        #
                        # # 计算有效损失
                        # valid_loss = (loss * valid_mask).sum()
                        #
                        # # 计算有效元素数量（防止除零）
                        # num_valid = valid_mask.sum() + 1e-8
                        #
                        # # 归一化损失
                        # loss = valid_loss / num_valid
                        writer.add_scalar("Loss", loss, epoch*steps_per_epoch+i)
                        writer.add_scalar("masked_accuracy", masked_accuracy, epoch*steps_per_epoch+i)
                        loss.backward()
                        self.optim.step()

                        if i % 200 ==0:
                            with torch.no_grad():
                                real_fake_images = torch.cat((imgs[:args.batch_size,-1,:], reconstruct[:args.batch_size])) # 取imgs最后一帧和重建图像 [-1,1]
                                vutils.save_image((real_fake_images+1)/2, os.path.join("results_transformer", f"{epoch}_{i}.jpg"), nrow=args.batch_size) # save_image需输入[0,1]
                        pbar.set_postfix(epoch=epoch,Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4), masked_accuracy=np.round(masked_accuracy.cpu().detach().numpy().item(), 4))
                        pbar.update(0)
                    except KeyboardInterrupt:
                        torch.save(self.model.state_dict(), os.path.join("checkpoints_transformer", f"transformer_last_{epoch}.pt"))
                        print("Interrupted, model saved.")
                        sys.exit(0)
            # log, sampled_imgs = self.model.log_images(imgs[0][None])
            # vutils.save_image(sampled_imgs, os.path.join("results", f"transformer_{epoch}.jpg"), nrow=4)
            # plot_images(log)
            if epoch % 2 == 0 or epoch== args.epochs - 1:
                torch.save(self.model.state_dict(), os.path.join("checkpoints_transformer", f"transformer_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=240, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=16384, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='/home/xuyang/taming-transformers-master/data/HorseRiding/train_random/', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='/home/xuyang/taming-transformers-master/logs/HorseRiding_32dim_16384_EMA/checkpoints/last.ckpt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=3, help='Input batch size for training.')
    parser.add_argument('--num-frames', type=int, default=5, help='Number of frames in the video.') # 取num_frames+1帧，最后一帧是待预测帧
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1.5e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=100000000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=0., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
    parser.add_argument('--reload', type=str, default=None,
                        help='Reload model from checkpoint')
    args = parser.parse_args()

    train_transformer = TrainTransformer(args)
    # train_transformer.train(args)

