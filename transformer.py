import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from taming.models.vqgan import VQModel as VQGAN
from vivit import ViViT
from st_transformer import SpatioTemporalTransformer
from scipy.stats import truncnorm
class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()
        self.codebook_size = args.num_codebook_vectors
        self.sos_token = args.sos_token
        self.latent_dim = args.latent_dim
        self.vqgan = self.load_vqgan(args)

        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": 256,
            "n_layer": 24,
            "n_head": 8,
            "n_embd": 1024
        }

        # 定义截断高斯分布的参数
        a, b = 0, 0.6
        self.mu, self.sigma = 0.3, 0.3 # 均值，标准差
        # 将截断区间转换为标准正态分布下的对应区间
        self.a_std, self.b_std = (a - self.mu) / self.sigma, (b - self.mu) / self.sigma  # 丢弃概率0到0.6的截断高斯分布
        np.random.seed(3407)
        torch.manual_seed(3407)
        # 定义均匀分布的区间
        # low, high = 0, 0.8

        # 从均匀分布中采样一个点
        # sample = np.random.uniform(low, high)
        # 采样截断高斯一个值


        # self.transformer = GPT(**transformer_config)
        self.transformer = ViViT(num_frames=args.num_frames+1, dim=768, token_size=15, heads=12, depth=10, scale_dim=4,
                                 codebook_size=args.num_codebook_vectors, emb_dropout=0.1)
        # self.transformer = SpatioTemporalTransformer(d_model=768, num_heads=12, num_layers=16,num_frames=args.num_frames+1, num_nodes=int(args.image_size/16)**2*2, vocab_size=args.num_codebook_vectors)

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.init_from_ckpt(args.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, indices = self.vqgan.encode(x)
        # indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=15, p2=15,quant_emb=256):
        # xx=self.vqgan.quantize.embedding(indices)
        idx_to_vector = self.vqgan.quantize.embedding(indices).view(indices.shape[0], p1, p2, quant_emb)
        idx_to_vector = idx_to_vector.permute(0, 3, 1, 2)
        # quant = self.vqgan.post_quant_conv(idx_to_vector)
        dec = self.vqgan.decode(idx_to_vector)
        return dec

    def z_to_image_res(self, indices, p1=15, p2=15,quant_emb=256): # indices: b, h*w*2
        # idx_to_vector = torch.tensor([]).to(indices.device)
        # for i in range(indices.shape[0]):
        # idx_to_vector = self.vqgan.quantize.embed_code(indices.view(indices.shape[0],p1, p2, -1))
        idx_to_vector = self.vqgan.quantize.embedding(indices).view(indices.shape[0], p1, p2, -1)
        # idx_to_vector = torch.concat()
        dec = self.vqgan.decode(idx_to_vector)
        return dec
    # def z_to_image(self, indices, p1=8, p2=8):
    #     ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
    #     ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
    #     image = self.vqgan.decode(ix_to_vectors)
    #     return image

    def forward(self, x): # x: b,t,c,h,w
        all_indices = None
        for i in range(x.shape[1]):
            quan_z, indices = self.encode_to_z(x[:, i])
            indices = indices.unsqueeze(1)
            if i == x.shape[1] - 1:  # 取最后一帧的indices作为target
                target = indices
            if all_indices is None:
                all_indices = indices
            else:
                all_indices = torch.cat((all_indices, indices), dim=1)
        indices = all_indices # batch, frames, h, w, n_codebook h*w=225 2个codebook
        if indices.dim() < 5:
            indices = indices.unsqueeze(-1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loss_rate = truncnorm.rvs(self.a_std, self.b_std, loc=self.mu, scale=self.sigma)
        # loss_rate = torch.rand(1)*0.6
        loss_rate = 0.0143
        # 为每个样本生成独立掩码
        b,t,h,w,n = indices.shape
        # final_mask = torch.stack([
        #     self.generate_spatial_mask(h, w, loss_rate, device)
        #     for _ in range(b)
        # ]).squeeze()  # [batch_size, 1, h, w] final_mask==1为丢弃部分

        # 扩展维度适配原始张量形状
        # final_mask = final_mask.unsqueeze(0)  # batch size=1时需要添加这个
        # final_mask = final_mask.unsqueeze(1)  # 添加时间维度
        #
        # final_mask = final_mask.unsqueeze(-1).expand(  # 扩展codebook维度
        #     -1, t, -1, -1, n
        # )  # 最终形状：[batch_size, seq_len, h, w, n_codebooks]
        #
        # mask = 1-(final_mask).to(torch.int64) # 令mask==0的部分为丢弃部分
        # mask_gen = DynamicBurstMaskGenerator(mu=0.3,sigma=0.3)
        # results = []
        # for _ in range(3000):
        #     mask = mask_gen.generate_mask((b, t, h, w, n), "cpu")
        #     actual_loss = 1 - mask.mean().item()
        #     results.append(actual_loss)
        #
        # # 计算均值和标准差
        # mean_loss = np.mean(results)
        # std_loss = np.std(results)
        # print(f"Mean Actual Loss: {mean_loss:.2%}, Std: {std_loss:.2%}")
        # mask = mask_gen.generate_mask(shape=indices.shape, device=indices.device)
        # pkeep = 1 - loss_rate
        # mask = torch.bernoulli(pkeep * torch.ones(indices.shape, device=indices.device))
        # mask = mask.round().to(dtype=torch.int64)
        # mask = self.create_drop_mask(shape=indices.shape, drop_ratio=0.4)
        # random_indices = torch.randint_like(indices, self.codebook_size)
        mask = self.create_drop_mask(indices.shape, loss_rate)
        new_indices = torch.where(mask == 1, indices, self.codebook_size)


        # new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        # target = indices
        # 将输入丢失的token用可学习的mask表示？
        new_indices = new_indices.view(b,t, h*w*n)
        indices_reshape = indices.view(b,t,h*w*n)
        target = target.squeeze().view(b,h*w*n)
        mask = mask.view(b,t,h*w*n)
        logits = self.transformer(new_indices, mask)
        # mask_last = mask[:,-1].unsqueeze(-1).expand(-1,-1,logits.size(2)) # 取最后一帧的mask
        mask_last = mask[:, -1]
        # mask_last_reverse = 1 - mask_last
        lost_logits = logits[:,-1][mask_last==0] # 只保留丢失的最后一帧labels用于更新
        # lost_logits = lost_logits.view(-1, logits.size(2))
        lost_target = target[mask_last==0] # 只保留丢失的target indices
        # lost_target = lost_target.view(-1,1)
        pre_labels = torch.argmax(logits,dim=-1)[:,-1]
        # mask_labels = mask[:,-1] * target.squeeze() + (1 - mask)[:,-1] * pre_labels # 预测的labels替换丢失的labels

        mask_labels = torch.where(mask_last==0,pre_labels, target) # 对于 mask_last 中的每个元素： 如果该元素等于 0，则从 pre_labels 中提取对应位置的元素，放入 mask_labels。 如果该元素不等于 0，则从 target 中提取对应位置的元素，放入 mask_labels
        correct = (pre_labels == target).float()  # (B, seq_len)
        masked_accuracy = (correct * (mask_last == 0)).sum() / (mask_last == 0).sum()
        reconstruct = self.z_to_image_res(mask_labels,p1=h,p2=w,quant_emb=self.latent_dim)
        return lost_logits, lost_target, reconstruct, mask_last, masked_accuracy

    def create_drop_mask(self, shape, drop_rate):
        """
        Args:
            shape: 输入张量形状 (b, t, h, w, n)
            drop_rate: 目标丢包率
        Returns:
            mask: 形状与输入相同的布尔张量，True表示对应位置被丢弃
        """
        b, t, h, w, n = shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = torch.zeros(shape, dtype=torch.bool, device=device)

        # 预处理四个包的空间位置掩码（动态适应任意h,w）
        packet_masks = []
        for packet_id in range(4):
            p_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
            for i in range(h):
                for j in range(w):
                    if 2 * (i % 2) + (j % 2) == packet_id:
                        p_mask[i, j] = True
            packet_masks.append(p_mask)

        for batch in range(b):
            for frame in range(t):
                total = h * w * n
                target = int(round(drop_rate * total))
                remaining = target

                # 阶段1: 完整包丢弃
                packet_order = torch.randperm(4, device=device)
                full_dropped = torch.zeros(4, dtype=torch.bool, device=device)

                for p in packet_order:
                    pid = p.item()
                    p_ij = packet_masks[pid]
                    p_size = p_ij.sum().item() * n

                    if remaining >= p_size and not full_dropped[pid]:
                        # 三维掩码扩展
                        p_3d = p_ij.unsqueeze(-1).expand(-1, -1, n)
                        mask[batch, frame] |= p_3d
                        remaining -= p_size
                        full_dropped[pid] = True

                # 阶段2: 部分包丢弃
                while remaining > 0:
                    candidates = [pid for pid in range(4) if not full_dropped[pid]]
                    if not candidates: break

                    # 随机选择候选包
                    selected_pid = torch.tensor(candidates, device=device)[
                        torch.randperm(len(candidates), device=device)[0]].item()
                    p_ij = packet_masks[selected_pid]

                    # 获取可用位置的三维索引
                    p_3d = p_ij.unsqueeze(-1).expand(-1, -1, n)
                    available = p_3d & (~mask[batch, frame])
                    indices = torch.where(available)

                    if len(indices[0]) == 0: continue

                    # 随机选择要丢弃的token
                    select_num = min(remaining, len(indices[0]))
                    chosen = torch.randperm(len(indices[0]), device=device)[:select_num]

                    # 更新掩码
                    mask[batch, frame,
                    indices[0][chosen],
                    indices[1][chosen],
                    indices[2][chosen]] = True
                    remaining -= select_num

        return 1-mask.float()
    # def create_drop_mask(self, shape, drop_ratio):
    #     """
    #     Args:
    #         shape: 输入张量的形状 (b, t, h, w, n)
    #         drop_ratio: 每帧丢弃的比例
    #     Returns:
    #         drop_mask: 形状与输入相同，True表示该位置被丢弃
    #     帧间丢弃不同部分
    #     """
    #     b, t, h, w, n = shape
    #     total_positions = h * w * n
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    #     # 全局记录哪些位置已被丢弃过（按样本和位置）
    #     global_mask = torch.zeros((b, total_positions), dtype=torch.bool, device=device)
    #
    #     # 初始化丢弃掩码
    #     drop_mask = torch.zeros((b, t, total_positions), dtype=torch.bool, device=device)
    #
    #     num_drop_per_frame = int(drop_ratio * total_positions)
    #
    #     for curr_t in range(t):
    #         for batch in range(b):
    #             # 获取当前可用的未丢弃位置
    #             available = torch.where(~global_mask[batch])[0]
    #             num_available = len(available)
    #
    #             # 计算可以从可用位置中丢弃的数量
    #             num_available_drop = min(num_drop_per_frame, num_available)
    #             num_used_drop = num_drop_per_frame - num_available_drop
    #
    #             # 优先丢弃未使用过的位置
    #             if num_available_drop > 0:
    #                 selected_available = available[torch.randperm(num_available, device=device)[:num_available_drop]]
    #                 drop_mask[batch, curr_t, selected_available] = True
    #                 global_mask[batch, selected_available] = True  # 标记为已使用
    #
    #             # 剩余需要从已丢弃位置中选择
    #             if num_used_drop > 0:
    #                 used = torch.where(global_mask[batch])[0]
    #                 if len(used) > 0:
    #                     selected_used = used[torch.randperm(len(used), device=device)[:num_used_drop]]
    #                     drop_mask[batch, curr_t, selected_used] = True
    #
    #     # 将掩码恢复为原始形状
    #     return drop_mask.view(b, t, h, w, n)
    def generate_spatial_mask(self, h, w, ratio, device='cpu', max_iters=10):
        """
        生成空间掩码，优先保证相邻位置不掩码，高比例时允许相邻
        返回掩码形状：(1, 1, h, w)
        """
        total = int(h * w * ratio)
        mask = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)

        # 四邻域卷积核（上下左右）
        kernel = torch.tensor([[[[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]]]], dtype=torch.float32, device=device)
        pad = 1

        remaining = total
        iter = 0
        while remaining > 0 and iter < max_iters:
            # 计算周围邻域是否有掩码
            neighbor_mask = F.conv2d(mask, kernel, padding=pad)
            available = (neighbor_mask == 0) & (mask == 0)

            # 获取可用位置索引
            available_indices = available.squeeze().nonzero(as_tuple=False)
            if len(available_indices) == 0:
                break

            # 随机选择
            num_select = min(remaining, len(available_indices))
            selected = available_indices[torch.randperm(len(available_indices))[:num_select]]

            # 更新掩码
            mask[0, 0, selected[:, 0], selected[:, 1]] = 1
            remaining -= num_select
            iter += 1

        # 处理剩余需要掩码的位置（允许相邻）
        if remaining > 0:
            flat_mask = mask.view(-1)
            zero_indices = (flat_mask == 0).nonzero().squeeze()
            num_select = min(remaining, len(zero_indices))
            if num_select > 0:
                selected = zero_indices[torch.randperm(len(zero_indices))[:num_select]]
                flat_mask[selected] = 1
            mask = flat_mask.view_as(mask)

        return mask.bool()

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x.device)

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))


class AdaptiveBurstMaskGenerator:
    def __init__(self, mu=0.3, sigma=0.15):
        self.mu = mu
        self.sigma = sigma
        self.a = (0 - mu) / sigma  # 截断下限标准化
        self.b = (0.6 - mu) / sigma  # 截断上限标准化

    def map_params(self, target_avg_loss):
        if target_avg_loss < 0.2:
            L_bad = 0.3
            P_BG = 0.9
        elif target_avg_loss < 0.4:
            L_bad = 0.6
            P_BG = 0.7
        else:
            L_bad = 0.9
            P_BG = 0.5
        return L_bad, P_BG
    def generate_mask(self, shape, device="cpu"):
        # 1. 采样目标丢包率
        target_avg_loss = truncnorm.rvs(self.a, self.b, loc=self.mu, scale=self.sigma)
        target_avg_loss = np.clip(target_avg_loss, 0, 0.6)
        # target_avg_loss = 0.8
        # 2. 动态映射参数
        L_bad, P_BG = self.map_params(target_avg_loss)

        # 3. 生成马尔可夫丢包掩码
        b, t, h, w, n = shape
        mask = torch.ones((b, t), device=device)
        for batch in range(b):
            state = 0  # 初始为Good
            for i in range(t):
                if state == 0:
                    # 计算P_GB
                    pi_bad = target_avg_loss / L_bad
                    P_GB = (pi_bad * P_BG) / (1 - pi_bad + 1e-6)
                    if torch.rand(1) < P_GB:
                        state = 1
                else:
                    if torch.rand(1) < P_BG:
                        state = 0
                    if state == 1 and torch.rand(1) < L_bad:
                        mask[batch, i] = 0
        return mask.view(b, t, 1, 1, 1).expand(shape)


class BurstMaskGenerator:
    def __init__(self, target_avg_loss=0.1, L_bad=0.5, P_BG=0.8):
        # 参数校验
        if target_avg_loss > L_bad:
            raise ValueError(f"target_avg_loss({target_avg_loss}) 必须 ≤ L_bad({L_bad})")
        if not (0 <= P_BG <= 1):
            raise ValueError("P_BG 必须在 [0,1] 范围内")

        self.pi_bad = target_avg_loss / L_bad
        self.P_GB = (self.pi_bad * P_BG) / (1 - self.pi_bad + 1e-6)  # 避免除零
        self.P_BG = P_BG
        self.L_bad = L_bad

    def generate_mask(self, shape, device="cpu"):
        b, t, h, w, n = shape
        mask = torch.ones((b, t), device=device)
        for batch in range(b):
            state = 0  # 0=Good, 1=Bad
            for i in range(t):
                if state == 0:
                    # Good → Bad 转移
                    if torch.rand(1) < self.P_GB:
                        state = 1
                else:
                    # Bad → Good 转移
                    if torch.rand(1) < self.P_BG:
                        state = 0
                    # 在Bad状态下丢包
                    if state == 1 and torch.rand(1) < self.L_bad:
                        mask[batch, i] = 0
        return mask.view(b, t, 1, 1, 1).expand(shape)
class DynamicBurstMaskGenerator:
    def __init__(self, mu=0.3, sigma=0.15, trunc_range=(0, 0.6)):
        # 截断高斯分布参数
        self.mu = mu
        self.sigma = sigma
        self.trunc_range = trunc_range
        self.a, self.b = (trunc_range[0] - mu) / sigma, (trunc_range[1] - mu) / sigma

    def sample_target(self):
        """采样目标丢包率"""
        return truncnorm.rvs(self.a, self.b, loc=self.mu, scale=self.sigma)

    def get_dynamic_params(self, target):
        """根据目标丢包率动态调整L_bad和P_BG"""
        # 归一化到[0,1]
        t = (target - self.trunc_range[0]) / (self.trunc_range[1] - self.trunc_range[0])

        # L_bad从0.3到0.9线性增加
        L_bad = 0.3 + t * 0.6

        # P_BG从0.9到0.3线性减少
        P_BG = 0.9 - t * 0.6

        return L_bad, P_BG

    def generate_mask(self, shape, device="cpu"):
        # 采样目标丢包率
        target = self.sample_target()
        target = 0.6
        # 动态获取参数
        L_bad, P_BG = self.get_dynamic_params(target)

        # 生成Mask（复用之前的BurstMaskGenerator）
        mask_gen = BurstMaskGenerator(target_avg_loss=target, L_bad=L_bad, P_BG=P_BG)
        return mask_gen.generate_mask(shape, device)













