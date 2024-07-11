import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from taming.models.vqgan import VQModel as VQGAN
from vivit import ViViT

class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()
        self.codebook_size = args.num_codebook_vectors
        self.sos_token = args.sos_token

        self.vqgan = self.load_vqgan(args)

        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": 256,
            "n_layer": 24,
            "n_head": 8,
            "n_embd": 1024
        }
        # self.transformer = GPT(**transformer_config)
        self.transformer = ViViT(num_frames=args.num_frames, dim=args.num_codebook_vectors,token_size=14, depth=4,scale_dim=4,
                                 codebook_size=args.num_codebook_vectors)
        self.pkeep = args.pkeep

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.vqgan.encode(x)
        indices = info[-1]
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=14, p2=14):
        # xx=self.vqgan.quantize.embedding(indices)
        idx_to_vector = self.vqgan.quantize.embedding(indices).view(indices.shape[0], p1, p2, 256)
        idx_to_vector = idx_to_vector.permute(0, 3, 1, 2)
        quant = self.vqgan.post_quant_conv(idx_to_vector)
        dec = self.vqgan.decoder(quant)
        return dec
    # def z_to_image(self, indices, p1=8, p2=8):
    #     ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
    #     ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
    #     image = self.vqgan.decode(ix_to_vectors)
    #     return image

    def forward(self, x):
        all_indices = None
        for i in range(x.shape[1]):
            _, indices = self.encode_to_z(x[:, i])
            indices = indices.unsqueeze(1)
            if i == x.shape[1] - 1:  # 取最后一帧的indices作为target
                target = indices
            if all_indices is None:
                all_indices = indices
            else:
                all_indices = torch.cat((all_indices, indices), dim=1)
        indices = all_indices # batch, frames, h*w h*w=196

        # sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        # sos_tokens = sos_tokens.long().to("cuda:1")
        # xx = indices[:,0].squeeze()
        # not_eequal = torch.sum(torch.ne(target.squeeze(),xx)).item()
        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.codebook_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        # new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        # target = indices

        logits = self.transformer(new_indices)
        reconstruct = self.z_to_image(new_indices[:,-1])
        return logits, target, reconstruct

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
















