import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors # 1024
        self.latent_dim = args.latent_dim # 256
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim) # quantized latent vectors
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous() # (B, C, H, W) -> (B, H, W, C)
        z_flattened = z.view(-1, self.latent_dim) # (B*H*W, C)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t())) # (B*H*W, 1024) 矩阵欧氏距离平方

        min_encoding_indices = torch.argmin(d, dim=1) # 1024
        z_q = self.embedding(min_encoding_indices).view(z.shape) # (B, H, W, C)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2) # 使特征和codebook的距离尽可能小，前半部分z_q.detach()，因此需要将z的梯度传递到z_q

        z_q = z + (z_q - z).detach() # 将z的梯度传递到z_q

        z_q = z_q.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

        return z_q, min_encoding_indices, loss