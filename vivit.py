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


# 替换原有Transformer为交替结构
class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, ff_dim, dropout):
        super().__init__()

        self.time_transformer = Transformer(dim, depth, heads, dim_head, ff_dim, dropout)
        self.space_transformer = Transformer(dim, depth, heads, dim_head, ff_dim, dropout)

    def forward(self, x):
        b, t, n, d = x.shape
        x = rearrange(x, 'b t n d -> (b n) t d')
        x = self.time_transformer(x)  # 时间建模
        x = rearrange(x, '(b n) t d -> b t n d', b=b)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)  # 空间建模
        return rearrange(x, '(b t) n d -> b t n d', b=b)

class ViViT(nn.Module):
    def __init__(self, num_frames=8, dim=256, depth=4, heads=3, pool='cls', in_channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4, token_size=16, codebook_size=1024):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.tok_emb = nn.Embedding(codebook_size+1, dim) # 预留mask token
        # self.pos_embedding_time = nn.Parameter(torch.randn(1, 1, num_frames + 1, dim)) # residual quantization token_size需要再×n_codebook b,n,t,d
        # self.pos_embedding_space = nn.Parameter(torch.randn(1, token_size ** 2 *2, dim)) # b,n,d
        # self.pos_embedding = nn.Parameter(torch.randn(1, token_size ** 2 *2, num_frames + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.mask_token = nn.Parameter(torch.randn(1,dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.pos_embedding_time = nn.Parameter(
            torch.randn(1, num_frames + 1, 1, dim))  # 1,t,1,d
        self.pos_embedding_space = nn.Parameter(torch.randn(1, 1, token_size ** 2 * 2, dim)) # 1,1,n,d
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        # 在ViViT中使用多个交替块
        self.st_blocks = nn.ModuleList([
            SpatioTemporalBlock(dim, 1, heads, dim_head, dim * scale_dim, emb_dropout)
            for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, codebook_size)  # 假设n_codebook=2
        )
        self.initialize_weights()
        self.apply(self._init_weights)

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embedding_time, std=.02)
        torch.nn.init.normal_(self.pos_embedding_space, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.space_token, std=.02)
        torch.nn.init.normal_(self.temporal_token, std=.02)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask):
        x = self.tok_emb(x)
        b, t, n, d = x.shape
        # 扩展mask维度用于广播
        # mask = mask.unsqueeze(-1)  # (B,T,N,1)
        #
        # # 替换mask位置的嵌入
        # x = torch.where(
        #     mask == 0,
        #     self.mask_token.expand_as(x),  # 广播到(B,T,N,D)
        #     x
        # )
        # 仅对最后一帧应用mask
        # last_frame_mask = mask[:, -1:, :]  # (B, 1, N)
        # last_frame_mask = last_frame_mask.unsqueeze(-1)  # (B, 1, N, 1)
        #
        # # 替换最后一帧的mask位置
        # x_last = x[:, -1:]  # (B, 1, N, D)
        # x_last = torch.where(
        #     last_frame_mask == 0,
        #     self.mask_token.expand_as(x_last),
        #     x_last
        # )
        # x = torch.cat([x[:, :-1], x_last], dim=1)  # (B, T, N, D)

        # cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b n t d', b=b, n=n)
        # x = rearrange(x, 'b t n d -> b n t d')
        # x = torch.cat((cls_temporal_tokens, x), dim=2)
        # x += self.pos_embedding_time
        #
        # x = rearrange(x, 'b n t d -> (b n) t d')
        # x = self.temporal_transformer(x)
        #
        # x = rearrange(x[:, 0], '(b n) ... -> b n ...', b=b)  # 取出cls token
        #
        # x += self.pos_embedding_space
        #
        # x = self.space_transformer(x)

        x += self.pos_embedding_time[:, :x.size(1)] + self.pos_embedding_space[:, :, :x.size(2)]
        # 时空交替处理
        for block in self.st_blocks:
            x = block(x)

        # x=x[:, -1]  # 取最后一帧
        return self.mlp_head(x)


if __name__ == "__main__":
    img = torch.ones([1, 16, 3, 224, 224]).to("cuda:1")

    model = ViViT(224, 16, 100, 16).to("cuda:1")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model(img)

    print("Shape of out :", out.shape)  # [B, num_classes]


