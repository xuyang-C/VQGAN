import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(self, num_frames=8, dim=256, depth=4, heads=3, pool='cls', in_channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4, token_size=16, codebook_size=1024):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.tok_emb = nn.Embedding(codebook_size, dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, token_size ** 2, num_frames + 2, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, codebook_size)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.tok_emb(x)
        b, t, n, _ = x.shape
        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b n t d', b=b, n=n)
        x = rearrange(x, 'b t n d -> b n t d')
        x = torch.cat((cls_temporal_tokens, x), dim=2)
        x += self.pos_embedding
        x = self.dropout(x)

        x = rearrange(x, 'b n t d -> (b n) t d')
        x = self.temporal_transformer(x)

        x = rearrange(x[:, 0], '(b n) ... -> b n ...', b=b)  # 取出cls token

        # cls_space_tokens = repeat(self.space_token, '() t d -> b t d', b=b)
        # x = torch.cat((cls_space_tokens, x), dim=1)

        x = self.space_transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


if __name__ == "__main__":
    img = torch.ones([1, 16, 3, 224, 224]).to("cuda:1")

    model = ViViT(224, 16, 100, 16).to("cuda:1")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model(img)

    print("Shape of out :", out.shape)  # [B, num_classes]


