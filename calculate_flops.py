from taming.models.vqgan import VQModel as VQGAN
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
import torch
from thop import profile, clever_format
import argparse
import torchprofile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=224, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=8192, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/home/xuyang/VQGAN-pytorch/data/vox2_train/', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=5000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--resume', type=str, default= 'coco_oi_epoch12.ckpt', help='Restore model address.') # 'coco_oi_epoch12.ckpt' 'coco_epoch117.ckpt'
    parser.add_argument('--reload', type=str, default= None, help='Reload model from checkpoint')

    args = parser.parse_args()


    def calculate_module_flops(module, input,custom_ops=None):
        flops, params = profile(module, inputs=(input,),custom_ops=custom_ops)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    def count_vector_quantizer(m, x, y):
        x = x[0]
        flops = x.numel() * m.n_e
        m.total_ops += torch.DoubleTensor([flops])

    model = VQGAN(args).to(device=args.device)
    vqmodel = VectorQuantizer(args.num_codebook_vectors, args.latent_dim, beta=0.25).to(device=args.device)
    imgs = torch.randn(1, 3, 224, 224).to(device=args.device)
    latent = torch.randn(1, 256, 14, 14).to(device=args.device)
    output_encoder = model.encoder(imgs)
    output_quant_conv = model.quant_conv(output_encoder)
    quant, diff, _ = model.encode(imgs)

    custom_ops = {VectorQuantizer: count_vector_quantizer}
    print(model.encoder)
    flops, params = calculate_module_flops(model.encoder, imgs, custom_ops=None)
    print(f"FLOPs: {flops}, Params: {params}")
    flops = torchprofile.profile_macs(model.encoder,imgs)
    print(f"FLOPs: {flops}")
