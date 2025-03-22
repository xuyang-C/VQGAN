import torch
import torch.nn as nn


class SpatioTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_frames, num_nodes):
        super().__init__()
        self.time_pe = nn.Parameter(torch.randn(1, num_frames, 1, d_model))  # (1, T, 1, D)
        self.space_pe = nn.Parameter(torch.randn(1, 1, num_nodes, d_model))  # (1, 1, N, D)

    def forward(self, x):
        # x: (B, T, N, D)
        return x + self.time_pe[:, :x.size(1)] + self.space_pe[:, :, :x.size(2)]


class SpatialTransformer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B*T, N, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B*N, T, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.spatial_transformer = SpatialTransformer(d_model, num_heads, dropout)
        self.temporal_transformer = TemporalTransformer(d_model, num_heads, dropout)

    def forward(self, x):
        # 空间处理 (B, T, N, D) -> (B*T, N, D)
        B, T, N, D = x.shape
        spatial_out = self.spatial_transformer(x.reshape(B * T, N, D))
        spatial_out = spatial_out.view(B, T, N, D)

        # 时间处理 (B, T, N, D) -> (B*N, T, D)
        temporal_in = spatial_out.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        temporal_out = self.temporal_transformer(temporal_in)
        temporal_out = temporal_out.view(B, N, T, D).permute(0, 2, 1, 3)

        return temporal_out


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers,
                 num_frames, num_nodes, vocab_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SpatioTemporalPositionalEncoding(d_model, num_frames, num_nodes)
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, 1, d_model)  # (1,1,1,D) 支持四维广播
        )
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        # 最后一帧预测头
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def forward(self, x, mask):
        # x: (B, T, N) 输入为token indices
        x = self.token_embed(x)  # (B, T, N, D)
        # 扩展mask维度用于广播
        mask = mask.unsqueeze(-1)  # (B,T,N,1)

        # 替换mask位置的嵌入
        x = torch.where(
            mask == 0,
            self.mask_token.expand_as(x),  # 广播到(B,T,N,D)
            x
        )
        x = self.pos_encoder(x)

        for block in self.blocks:
            x = block(x)  # (B, T, N, D)

        # 提取最后一帧特征
        last_frame = x[:, -1, :, :]  # (B, N, D)

        # 预测每个位置的token分布
        logits = self.head(last_frame)  # (B, N, vocab_size)
        return logits
