import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import time
import math

from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


class Args:
    image_size = 32
    num_channels = 128
    num_res_blocks = 3
    num_heads = 4
    num_heads_upsample = -1
    attention_resolutions = "16,8"
    dropout = 0.3
    learn_sigma = True
    sigma_small = False
    class_cond = False
    diffusion_steps = 4000
    noise_schedule = "cosine"
    timestep_respacing = ""
    use_kl = False
    predict_xstart = False
    rescale_timesteps = True
    rescale_learned_sigmas = True
    use_checkpoint = False
    use_scale_shift_norm = True


class TemporaryGrad:
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)


class DiffusionPurificationModel(nn.Module):
    def __init__(
            self,
            device,
            lowrank_weight=0.05,
            lowrank_patch=4,
            lowrank_rank=8,
            contrast_weight=0.0,
            contrast_temperature=0.2,
            guidance_normalize=False,
    ):
        super().__init__()
        self.device = device
        self.lowrank_weight = lowrank_weight
        self.lowrank_patch = lowrank_patch
        self.lowrank_rank = lowrank_rank
        self.contrast_weight = contrast_weight
        self.contrast_temperature = contrast_temperature
        self.guidance_normalize = guidance_normalize
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            # torch.load("cifar10_uncond_50M_500K.pt", map_location=torch.device(self.device))
            torch.load("cifar10_uncond_50M_500K.pt")
        )

        self.model = model
        self.diffusion = diffusion
        self.mse_loss = nn.MSELoss(reduction='none')

    def patch_features(self, x):
        """
        Patch特征(Patch Features): 将图像切成小块，形成结构特征矩阵。
        x: [B, C, H, W] -> [B, C * patch * patch, num_patches]
        """
        patch = self.lowrank_patch
        return F.unfold(x, kernel_size=patch, stride=patch)

    def low_rank_approx(self, features):
        """
        低秩近似(Low-Rank Approximation): 只保留主要结构成分，压制细碎扰动。
        """
        min_dim = min(features.shape[-2], features.shape[-1])
        rank = max(1, min(self.lowrank_rank, min_dim))

        with torch.no_grad():
            features_f = features.float()
            u, s, vh = torch.linalg.svd(features_f, full_matrices=False)
            approx = (u[:, :, :rank] * s[:, None, :rank]) @ vh[:, :rank, :]

        return approx.to(dtype=features.dtype)

    def lowrank_feature_loss(self, x_t, x_0_t):
        """
        低秩特征约束(Low-Rank Feature Constraint)。
        让当前扩散状态的patch结构靠近preliminary image的低秩主结构。
        """
        if self.lowrank_weight <= 0:
            return torch.zeros(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)

        features_t = self.patch_features(x_t)
        features_ref = self.patch_features(x_0_t.detach())
        lowrank_ref = self.low_rank_approx(features_ref)
        loss = self.mse_loss(features_t, lowrank_ref).flatten(1).sum(dim=1)
        return loss

    def contrastive_embedding(self, x):
        """
        对比嵌入(Contrastive Embedding)。
        用patch结构的平均表示作为轻量语义向量，避免额外引入分类器特征。
        """
        features = self.patch_features(x)
        emb = features.mean(dim=-1)
        return F.normalize(emb, dim=1)

    def contrastive_guidance_loss(self, x_t, x_0_t):
        """
        对比引导损失(Contrastive Guidance Loss)。
        拉近当前样本与自己的preliminary image，推远batch内其他样本。
        """
        if self.contrast_weight <= 0 or x_t.shape[0] <= 1:
            return torch.zeros((), device=x_t.device, dtype=x_t.dtype)

        anchor = self.contrastive_embedding(x_t)
        positive_bank = self.contrastive_embedding(x_0_t.detach())
        logits = anchor @ positive_bank.t()
        logits = logits / max(self.contrast_temperature, 1e-6)
        targets = torch.arange(x_t.shape[0], device=x_t.device)
        return F.cross_entropy(logits, targets)

    def guide(self, x_t, x_0_t):
        _x_t = x_t.detach().clone()
        _x_t.requires_grad = True
        _x_0_t = x_0_t.detach().clone()
        _x_0_t.requires_grad = True

        pixel_loss = self.mse_loss(_x_t, _x_0_t).flatten(1).sum(dim=1)
        lowrank_loss = self.lowrank_feature_loss(_x_t, _x_0_t)
        contrast_loss = self.contrastive_guidance_loss(_x_t, _x_0_t)

        if self.guidance_normalize:
            pixel_grad = torch.autograd.grad(
                pixel_loss.sum(), _x_t, retain_graph=True, create_graph=False
            )[0]
            lowrank_grad = torch.autograd.grad(
                lowrank_loss.sum(), _x_t, retain_graph=True, create_graph=False
            )[0]
            if self.contrast_weight > 0:
                contrast_grad = torch.autograd.grad(
                    contrast_loss, _x_t, retain_graph=False, create_graph=False
                )[0]
            else:
                contrast_grad = torch.zeros_like(pixel_grad)

            def match_norm(grad, ref_grad):
                grad_norm = grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                ref_norm = ref_grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                return grad * (ref_norm / (grad_norm + 1e-12))

            grad = pixel_grad
            if self.lowrank_weight > 0:
                grad = grad + self.lowrank_weight * match_norm(lowrank_grad, pixel_grad)
            if self.contrast_weight > 0:
                grad = grad + self.contrast_weight * match_norm(contrast_grad, pixel_grad)
            return grad

        loss = pixel_loss + self.lowrank_weight * lowrank_loss
        if self.contrast_weight > 0:
            loss = loss + (self.contrast_weight * contrast_loss / max(_x_t.shape[0], 1))
        loss.requires_grad_(True)

        loss.backward(torch.ones_like(loss))
        grad = _x_t.grad
        assert grad is not None
        return grad

    def denoise(self, x, t, s=None, **kwgs):
        start_time = time.time()
        t_batch = torch.tensor([t] * len(x)).to(x.device)
        x_t_ = self.diffusion.q_sample(x_start=x, t=t_batch)
        x_pre = self.diffusion.p_sample(
            self.model,
            x_t_,
            t_batch,
            clip_denoised=True
        )['pred_xstart']

        noise = torch.randn_like(x)
        x_t = self.diffusion.q_sample(x_start=x, t=t_batch, noise=noise)
        x_0_t = self.diffusion.q_sample(x_start=x_pre, t=t_batch, noise=noise)

        with TemporaryGrad():
            grad = self.guide(x_t, x_0_t)
            # print(grad.shape)

        s = s or 0
        S = s * self.diffusion.get_sqrt_one_minus_alphas_cumprod(x, t_batch) / self.diffusion.get_sqrt_alphas_cumprod(x,
                                                                                                                      t_batch)
        # print("s", s.max(), s.min())
        out = self.diffusion.p_mean_variance(
            self.model,
            x_t,
            t_batch,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None
        )
        var = torch.exp(out["log_variance"])
        sqrt_var = torch.exp(0.5 * out["log_variance"])
        noise = torch.randn_like(x)

        sample = (out["mean"] - S * var * grad) + sqrt_var * noise
        # print((s*var*grad).mean(), out["mean"].mean())
        # sample = out["mean"] + sqrt_var * noise

        sample = self.diffusion.p_sample(
            self.model,
            sample,
            t_batch,
            clip_denoised=True
        )['pred_xstart']
        end_time = time.time()
        # print(f"time: {end_time - start_time}")
        save_image((sample + 1) / 2, os.path.join('./', 'guide_sample.png'))
        return sample

    # multi-step
    def denoiseN(self, x, t, s=None, steps=1, p=0.8, **kwgs):
        t_batch = torch.tensor([t] * len(x)).to(x.device)
        x_t_ = self.diffusion.q_sample(x_start=x, t=t_batch)
        x_pre = self.diffusion.p_sample(
            self.model,
            x_t_,
            t_batch,
            clip_denoised=True
        )['pred_xstart']

        noise = torch.randn_like(x)
        x_t = self.diffusion.q_sample(x_start=x, t=t_batch, noise=noise)
        x_0_t = self.diffusion.q_sample(x_start=x_pre, t=t_batch, noise=noise)

        def exponential_decay(start_value, decay_rate, num_steps):
            current_value = start_value
            for step in range(num_steps):
                yield current_value
                current_value *= math.exp(-decay_rate)

        for s_ in exponential_decay(s, p, steps):
            with TemporaryGrad():
                grad = self.guide(x_t, x_0_t)
                # print(grad.shape)

            S = s_ * self.diffusion.get_sqrt_one_minus_alphas_cumprod(x,
                                                                      t_batch) / self.diffusion.get_sqrt_alphas_cumprod(
                x, t_batch)
            # print("s", s.max(), s.min())
            out = self.diffusion.p_mean_variance(
                self.model,
                x_t,
                t_batch,
                clip_denoised=True,
                denoised_fn=None,
                model_kwargs=None
            )
            var = torch.exp(out["log_variance"])
            sqrt_var = torch.exp(0.5 * out["log_variance"])
            noise = torch.randn_like(x)

            # x_{t-1}
            x_t = (out["mean"] - S * var * grad) + sqrt_var * noise
            t_batch = t_batch - 1

        t_batch = t_batch + 1
        sample = self.diffusion.p_sample(
            self.model,
            x_t,
            t_batch,
            clip_denoised=True
        )['pred_xstart']
        save_image((sample + 1) / 2, os.path.join('./', 'guide_sample.png'))
        return sample
