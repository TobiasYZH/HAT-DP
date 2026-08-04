import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from robustbench.utils import load_model
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from diffusion import Args
from improved_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


class LoRAConv2d(nn.Module):
    def __init__(self, base, rank=4, alpha=4.0):
        super().__init__()
        if base.groups != 1:
            raise ValueError("LoRAConv2d only supports groups=1.")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.lora_down = nn.Conv2d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_up(self.lora_down(x))

    def merged_weight(self):
        down = self.lora_down.weight
        up = self.lora_up.weight[:, :, 0, 0]
        delta = torch.einsum("or,rihw->oihw", up, down) * self.scale
        return self.base.weight + delta

    def merged_bias(self):
        return self.base.bias


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=4.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_up(self.lora_down(x))

    def merged_weight(self):
        return self.base.weight + self.scale * (self.lora_up.weight @ self.lora_down.weight)

    def merged_bias(self):
        return self.base.bias


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_scale(diffusion, x, t_batch, lambda_unit, lambda_min, lambda_max):
    sqrt_alpha = diffusion.get_sqrt_alphas_cumprod(x, t_batch)
    sqrt_one_minus = diffusion.get_sqrt_one_minus_alphas_cumprod(x, t_batch)
    scale = lambda_unit * sqrt_alpha / (sqrt_one_minus + 1e-12)
    return torch.clamp(scale, min=lambda_min, max=lambda_max)


def rank_based_gaussian_mapping(delta, noise):
    flat_delta = delta.flatten(1)
    flat_noise = noise.flatten(1)
    sorted_noise, _ = torch.sort(flat_noise, dim=1)
    delta_rank = torch.argsort(torch.argsort(flat_delta, dim=1), dim=1)
    mapped = torch.gather(sorted_noise, dim=1, index=delta_rank)
    mapped = mapped.view_as(delta)
    return mapped.detach() + delta - delta.detach()


def predict_xstart(diffusion, model, x_t, t_batch):
    return diffusion.p_mean_variance(
        model,
        x_t,
        t_batch,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    )["pred_xstart"]


def build_addt_xt(diffusion, x0, t_batch, delta, lambda_unit, lambda_min, lambda_max):
    noise = torch.randn_like(x0)
    mapped = rank_based_gaussian_mapping(delta, noise)
    lam = get_scale(diffusion, x0, t_batch, lambda_unit, lambda_min, lambda_max)
    sqrt_alpha = diffusion.get_sqrt_alphas_cumprod(x0, t_batch)
    sqrt_one_minus = diffusion.get_sqrt_one_minus_alphas_cumprod(x0, t_batch)
    mixed_noise = torch.sqrt(torch.clamp(1.0 - lam ** 2, min=0.0)) * noise + lam * mapped
    return sqrt_alpha * x0 + sqrt_one_minus * mixed_noise


def classifier_loss_from_xt(diffusion, model, classifier, x_t, t_batch, labels):
    pred_xstart = predict_xstart(diffusion, model, x_t, t_batch)
    classifier_input = torch.clamp((pred_xstart + 1) / 2, 0, 1)
    logits = classifier(classifier_input)
    return F.cross_entropy(logits, labels)


def run_cgpo(args, diffusion, model, classifier, x0, labels, t_batch):
    delta = torch.empty_like(x0).uniform_(-args.delta_init, args.delta_init)
    delta.requires_grad = True

    was_training = model.training
    previous_requires_grad = [param.requires_grad for param in model.parameters()]
    model.eval()
    model.requires_grad_(False)
    for _ in range(args.cgpo_steps):
        x_t = build_addt_xt(
            diffusion,
            x0,
            t_batch,
            delta,
            args.lambda_unit,
            args.lambda_min,
            args.lambda_max,
        )
        loss = classifier_loss_from_xt(diffusion, model, classifier, x_t, t_batch, labels)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta.add_(args.cgpo_lr * grad.sign())
            delta.clamp_(-args.delta_eps, args.delta_eps)
    for param, requires_grad in zip(model.parameters(), previous_requires_grad):
        param.requires_grad_(requires_grad)
    if was_training:
        model.train()
    return delta.detach()


def weighted_reconstruction_loss(diffusion, x0, pred_xstart, t_batch):
    weight = diffusion.get_sqrt_alphas_cumprod(x0, t_batch) / (
        diffusion.get_sqrt_one_minus_alphas_cumprod(x0, t_batch) + 1e-12
    )
    return (weight * (x0 - pred_xstart)).flatten(1).pow(2).mean(dim=1).mean()


def addt_loss(args, diffusion, model, x0, t_batch, delta):
    x_t = build_addt_xt(
        diffusion,
        x0,
        t_batch,
        delta,
        args.lambda_unit,
        args.lambda_min,
        args.lambda_max,
    )
    pred_xstart = predict_xstart(diffusion, model, x_t, t_batch)
    return weighted_reconstruction_loss(diffusion, x0, pred_xstart, t_batch), pred_xstart


def pgd_attack(classifier, imgs, labels, eps, step_size, steps):
    was_training = classifier.training
    classifier.eval()

    adv = imgs.detach() + torch.empty_like(imgs).uniform_(-eps, eps)
    adv = torch.clamp(adv, 0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        logits = classifier(adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        with torch.no_grad():
            adv = adv + step_size * grad.sign()
            adv = torch.max(torch.min(adv, imgs + eps), imgs - eps)
            adv = torch.clamp(adv, 0, 1)

    if was_training:
        classifier.train()
    return adv.detach()


def eot_purifier_attack(args, diffusion, model, classifier, imgs, labels, device):
    adv = imgs.detach() + torch.empty_like(imgs).uniform_(-args.eot_eps, args.eot_eps)
    adv = torch.clamp(adv, 0, 1)

    was_training = model.training
    previous_requires_grad = [param.requires_grad for param in model.parameters()]
    model.eval()
    model.requires_grad_(False)
    classifier.eval()

    for _ in range(args.eot_steps):
        adv.requires_grad_(True)
        loss = torch.zeros((), device=device)
        for _ in range(args.eot_iters):
            adv_norm = Normalize(0.5, 0.5)(adv)
            t_batch = torch.randint(args.t_min, args.t_max + 1, (adv.shape[0],), device=device)
            x_t = diffusion.q_sample(x_start=adv_norm, t=t_batch)
            pred_xstart = predict_xstart(diffusion, model, x_t, t_batch)
            classifier_input = torch.clamp((pred_xstart + 1) / 2, 0, 1)
            logits = classifier(classifier_input)
            loss = loss + F.cross_entropy(logits, labels) / max(args.eot_iters, 1)

        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        with torch.no_grad():
            adv = adv + args.eot_step_size * grad.sign()
            adv = torch.max(torch.min(adv, imgs + args.eot_eps), imgs - args.eot_eps)
            adv = torch.clamp(adv, 0, 1)

    for param, requires_grad in zip(model.parameters(), previous_requires_grad):
        param.requires_grad_(requires_grad)
    if was_training:
        model.train()
    return adv.detach()


def supervised_adv_loss(diffusion, model, x_clean, x_adv, t_batch):
    x_t = diffusion.q_sample(x_start=x_adv, t=t_batch)
    pred_xstart = predict_xstart(diffusion, model, x_t, t_batch)
    return weighted_reconstruction_loss(diffusion, x_clean, pred_xstart, t_batch), pred_xstart


def classification_loss(classifier, pred_xstart, labels):
    classifier_input = torch.clamp((pred_xstart + 1) / 2, 0, 1)
    logits = classifier(classifier_input)
    return F.cross_entropy(logits, labels)


def should_train_last_layer(name, train_output_blocks, num_output_blocks):
    if name.startswith("out."):
        return True
    if name.startswith("middle_block."):
        return True
    if name.startswith("output_blocks."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            block_idx = int(parts[1])
            return block_idx >= max(0, num_output_blocks - train_output_blocks)
    return False


def apply_last_layer_finetune(model, train_output_blocks):
    model.requires_grad_(False)
    num_output_blocks = len(model.output_blocks)
    for name, param in model.named_parameters():
        if should_train_last_layer(name, train_output_blocks, num_output_blocks):
            param.requires_grad_(True)


def inject_lora(module, rank, alpha):
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, child_name, LoRAConv2d(child, rank=rank, alpha=alpha))
        elif isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            inject_lora(child, rank, alpha)


def apply_lora_finetune(model, train_output_blocks, rank, alpha):
    model.requires_grad_(False)
    inject_lora(model.middle_block, rank, alpha)
    inject_lora(model.out, rank, alpha)
    start_idx = max(0, len(model.output_blocks) - train_output_blocks)
    for block in model.output_blocks[start_idx:]:
        inject_lora(block, rank, alpha)


def count_trainable_parameters(model):
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def merged_state_dict_for_lora(model):
    state = model.state_dict()
    merged = {}
    skip_prefixes = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRAConv2d, LoRALinear)):
            merged[f"{name}.weight"] = module.merged_weight().detach().cpu()
            if module.merged_bias() is not None:
                merged[f"{name}.bias"] = module.merged_bias().detach().cpu()
            skip_prefixes.append(f"{name}.")

    for key, value in state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        merged[key] = value.detach().cpu()
    return merged


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state_dict_for_lora(model), path)


def main():
    parser = argparse.ArgumentParser(description="Supervised ADDT training for CIFAR-10 diffusion purification.")
    parser.add_argument("--base-checkpoint", type=str, default="cifar10_uncond_50M_500K.pt")
    parser.add_argument("--out-checkpoint", type=str, default="checkpoints/strong_supervised_addt_latest2.pt")
    parser.add_argument("--finetune-mode", type=str, default="last", choices=["lora", "last", "full"])
    parser.add_argument("--train-output-blocks", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=6)
    parser.add_argument("--lora-alpha", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--t-min", type=int, default=150)
    parser.add_argument("--t-max", type=int, default=600)
    parser.add_argument("--lambda-unit", type=float, default=0.3)
    parser.add_argument("--lambda-min", type=float, default=0.05)
    parser.add_argument("--lambda-max", type=float, default=0.4)
    parser.add_argument("--addt-weight", type=float, default=1.0)
    parser.add_argument("--adv-weight", type=float, default=0.3)
    parser.add_argument("--eot-weight", type=float, default=0.8)
    parser.add_argument("--cls-weight", type=float, default=0.05)
    parser.add_argument("--cgpo-steps", type=int, default=2)
    parser.add_argument("--cgpo-lr", type=float, default=1 / 255)
    parser.add_argument("--delta-eps", type=float, default=8 / 255)
    parser.add_argument("--delta-init", type=float, default=1 / 255)
    parser.add_argument("--adv-steps", type=int, default=10)
    parser.add_argument("--adv-step-size", type=float, default=2 / 255)
    parser.add_argument("--adv-eps", type=float, default=8 / 255)
    parser.add_argument("--eot-steps", type=int, default=7)
    parser.add_argument("--eot-iters", type=int, default=3)
    parser.add_argument("--eot-step-size", type=float, default=2 / 255)
    parser.add_argument("--eot-eps", type=float, default=8 / 255)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        "Strong supervised ADDT config: "
        f"steps={args.steps}, bs={args.bs}, lr={args.lr}, t=[{args.t_min},{args.t_max}], "
        f"finetune_mode={args.finetune_mode}, train_output_blocks={args.train_output_blocks}, "
        f"lora_rank={args.lora_rank}, lora_alpha={args.lora_alpha}, "
        f"lambda_unit={args.lambda_unit}, lambda=[{args.lambda_min},{args.lambda_max}], "
        f"weights(addt/adv/eot/cls)=({args.addt_weight}/{args.adv_weight}/{args.eot_weight}/{args.cls_weight}), "
        f"cgpo_steps={args.cgpo_steps}, adv_steps={args.adv_steps}, "
        f"eot_steps={args.eot_steps}, eot_iters={args.eot_iters}, out={args.out_checkpoint}"
    )

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(Args(), model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.base_checkpoint, map_location=device))
    model.to(device)
    model.train()

    if args.finetune_mode == "lora":
        apply_lora_finetune(model, args.train_output_blocks, args.lora_rank, args.lora_alpha)
    elif args.finetune_mode == "last":
        apply_last_layer_finetune(model, args.train_output_blocks)
    elif args.finetune_mode == "full":
        model.requires_grad_(True)
    else:
        raise ValueError(f"Unknown finetune_mode: {args.finetune_mode}")
    model.to(device)

    trainable, total = count_trainable_parameters(model)
    print(f"Trainable diffusion parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    classifier = load_model(model_name="Standard", dataset="cifar10", threat_model="Linf")
    classifier.to(device).eval()
    classifier.requires_grad_(False)

    dataset = torchvision.datasets.CIFAR10(
        "CIFAR10",
        train=True,
        download=True,
        transform=Compose([ToTensor()]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr)

    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True)
    for step in progress:
        try:
            imgs, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            imgs, labels = next(iterator)

        imgs = imgs.to(device)
        labels = labels.to(device)
        x0 = Normalize(0.5, 0.5)(imgs)
        t_batch = torch.randint(args.t_min, args.t_max + 1, (x0.shape[0],), device=device)

        x_adv = pgd_attack(
            classifier,
            imgs,
            labels,
            eps=args.adv_eps,
            step_size=args.adv_step_size,
            steps=args.adv_steps,
        )
        x_adv_norm = Normalize(0.5, 0.5)(x_adv)

        x_eot_adv = eot_purifier_attack(
            args,
            diffusion,
            model,
            classifier,
            imgs,
            labels,
            device,
        )
        x_eot_adv_norm = Normalize(0.5, 0.5)(x_eot_adv)

        delta = run_cgpo(args, diffusion, model, classifier, x0, labels, t_batch)
        addt_recon, pred_addt = addt_loss(args, diffusion, model, x0, t_batch, delta)
        adv_recon, pred_adv = supervised_adv_loss(diffusion, model, x0, x_adv_norm, t_batch)
        eot_recon, pred_eot = supervised_adv_loss(diffusion, model, x0, x_eot_adv_norm, t_batch)

        cls_loss = torch.zeros((), device=device)
        if args.cls_weight > 0:
            cls_loss = (
                classification_loss(classifier, pred_addt, labels)
                + classification_loss(classifier, pred_adv, labels)
                + classification_loss(classifier, pred_eot, labels)
            ) / 3.0

        loss = (
            args.addt_weight * addt_recon
            + args.adv_weight * adv_recon
            + args.eot_weight * eot_recon
            + args.cls_weight * cls_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        progress.set_postfix(
            loss=f"{loss.item():.5f}",
            addt=f"{addt_recon.item():.5f}",
            adv=f"{adv_recon.item():.5f}",
            eot=f"{eot_recon.item():.5f}",
            cls=f"{cls_loss.item():.5f}",
            t_min=int(t_batch.min()),
            t_max=int(t_batch.max()),
        )

    save_checkpoint(model, args.out_checkpoint)
    print(f"Saved supervised ADDT checkpoint to: {args.out_checkpoint}")


if __name__ == "__main__":
    main()
