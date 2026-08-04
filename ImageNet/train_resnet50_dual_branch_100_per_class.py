import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from tqdm import tqdm

from diffusion import Args
from guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def center_crop_224(x):
    _, _, h, w = x.shape
    if h < 224 or w < 224:
        return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    top = (h - 224) // 2
    left = (w - 224) // 2
    return x[:, :, top:top + 224, left:left + 224]


class ImageNetClassifier(nn.Module):
    def __init__(self, weights_name="v2"):
        super().__init__()
        if weights_name == "v1":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        elif weights_name in {"v2", "default"}:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        else:
            raise ValueError(f"Unknown ResNet50 weights: {weights_name}")
        self.model = torchvision.models.resnet50(weights=weights)
        self.model.eval()

    def forward(self, x):
        mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
        std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
        x = center_crop_224(x)
        x = (x - mean) / std
        return self.model(x)


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
    with torch.no_grad():
        pred_xstart = predict_xstart(diffusion, model, x_t, t_batch)

    proxy_input = torch.clamp((x_t + 1) / 2, 0, 1)
    purified_input = torch.clamp((pred_xstart + 1) / 2, 0, 1)
    classifier_input = proxy_input + (purified_input - proxy_input).detach()
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


def project_linf(adv, clean, eps):
    adv = torch.max(torch.min(adv, clean + eps), clean - eps)
    return torch.clamp(adv, 0, 1)


def purify_once(args, diffusion, model, imgs, device):
    imgs_norm = imgs * 2 - 1
    t_batch = torch.randint(args.t_min, args.t_max + 1, (imgs.shape[0],), device=device)
    x_t = diffusion.q_sample(x_start=imgs_norm, t=t_batch)
    pred_xstart = predict_xstart(diffusion, model, x_t, t_batch)
    return torch.clamp((pred_xstart + 1) / 2, 0, 1)


def bpda_purifier_classifier_grad(args, diffusion, model, classifier, adv, labels, device):
    adv_bpda = adv.detach().requires_grad_(True)
    with torch.no_grad():
        purified = purify_once(args, diffusion, model, adv_bpda, device)

    classifier_input = adv_bpda + (purified - adv_bpda).detach()
    logits = classifier(classifier_input)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, adv_bpda, only_inputs=True)[0]
    return grad.detach()


def pgd_eot_purifier_attack(args, diffusion, model, classifier, imgs, labels, device):
    adv = imgs.detach() + torch.empty_like(imgs).uniform_(-args.eot_eps, args.eot_eps)
    adv = project_linf(adv, imgs, args.eot_eps)

    was_training = model.training
    previous_requires_grad = [param.requires_grad for param in model.parameters()]
    model.eval()
    model.requires_grad_(False)
    classifier.eval()

    for _ in range(args.train_pgd_eot_steps):
        grad = torch.zeros_like(adv)
        for _ in range(args.train_pgd_eot_iters):
            grad = grad + bpda_purifier_classifier_grad(
                args,
                diffusion,
                model,
                classifier,
                adv,
                labels,
                device,
            )
        grad = grad / max(args.train_pgd_eot_iters, 1)

        with torch.no_grad():
            adv = adv + args.train_pgd_eot_step_size * grad.sign()
            adv = project_linf(adv, imgs, args.eot_eps)

    for param, requires_grad in zip(model.parameters(), previous_requires_grad):
        param.requires_grad_(requires_grad)
    if was_training:
        model.train()
    return adv.detach()


def supervised_loss(diffusion, model, x_clean, x_input, t_batch):
    x_t = diffusion.q_sample(x_start=x_input, t=t_batch)
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


def count_trainable_parameters(model):
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(description="ImageNet ResNet50-V2 dual-branch diffusion purifier training.")
    parser.add_argument("--train-dir", type=str, default="imagenet_train_100_per_class/train")
    parser.add_argument("--base-checkpoint", type=str, default="256x256_diffusion_uncond.pt")
    parser.add_argument("--out-checkpoint", type=str, default="checkpoints/resnet50_v2_pgd_eot_4_255_100_bs2_100style_v1.pt")
    parser.add_argument("--resnet-weights", type=str, default="v2", choices=["v1", "v2", "default"])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--train-output-blocks", type=int, default=5)
    parser.add_argument("--t-min", type=int, default=150)
    parser.add_argument("--t-max", type=int, default=600)
    parser.add_argument("--lambda-unit", type=float, default=0.3)
    parser.add_argument("--lambda-min", type=float, default=0.05)
    parser.add_argument("--lambda-max", type=float, default=0.4)
    parser.add_argument("--addt-weight", type=float, default=1.0)
    parser.add_argument("--adv-weight", type=float, default=0.3)
    parser.add_argument("--eot-weight", type=float, default=0.2)
    parser.add_argument("--clean-weight", type=float, default=0.7)
    parser.add_argument("--cls-weight", type=float, default=0.005)
    parser.add_argument("--cgpo-steps", type=int, default=2)
    parser.add_argument("--cgpo-lr", type=float, default=1 / 255)
    parser.add_argument("--delta-eps", type=float, default=4 / 255)
    parser.add_argument("--delta-init", type=float, default=1 / 255)
    parser.add_argument("--adv-steps", type=int, default=10)
    parser.add_argument("--adv-step-size", type=float, default=1 / 255)
    parser.add_argument("--adv-eps", type=float, default=4 / 255)
    parser.add_argument("--eot-eps", type=float, default=4 / 255)
    parser.add_argument("--train-pgd-eot-steps", type=int, default=20)
    parser.add_argument("--train-pgd-eot-iters", type=int, default=2)
    parser.add_argument("--train-pgd-eot-step-size", type=float, default=1 / 255)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_pgd_eot_step_size is None:
        args.train_pgd_eot_step_size = args.eot_eps / 4

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        "ImageNet dual-branch config: "
        f"weights={args.resnet_weights}, steps={args.steps}, bs={args.bs}, lr={args.lr}, "
        f"t=[{args.t_min},{args.t_max}], train_output_blocks={args.train_output_blocks}, "
        f"loss(addt/adv/eot/clean/cls)=({args.addt_weight}/{args.adv_weight}/"
        f"{args.eot_weight}/{args.clean_weight}/{args.cls_weight}), "
        f"adv_steps={args.adv_steps}, train_pgd_eot_steps={args.train_pgd_eot_steps}, "
        f"train_pgd_eot_iters={args.train_pgd_eot_iters}, "
        f"train_pgd_eot_step_size={args.train_pgd_eot_step_size}, eot_eps={args.eot_eps}, "
        f"out={args.out_checkpoint}"
    )

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(Args(), model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.base_checkpoint, map_location=device))
    model.to(device)
    model.train()
    apply_last_layer_finetune(model, args.train_output_blocks)
    model.to(device)

    trainable, total = count_trainable_parameters(model)
    print(f"Trainable diffusion parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    classifier = ImageNetClassifier(args.resnet_weights).to(device).eval()
    classifier.requires_grad_(False)

    dataset = torchvision.datasets.ImageFolder(
        args.train_dir,
        transform=Compose([
            RandomResizedCrop(256, antialias=True),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr)

    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True)
    for _ in progress:
        try:
            imgs, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            imgs, labels = next(iterator)

        imgs = torch.clamp(imgs.to(device), 0, 1)
        labels = labels.to(device)
        x0 = imgs * 2 - 1
        t_batch = torch.randint(args.t_min, args.t_max + 1, (x0.shape[0],), device=device)

        clean_recon, _ = supervised_loss(diffusion, model, x0, x0, t_batch)

        x_adv = pgd_attack(
            classifier,
            imgs,
            labels,
            eps=args.adv_eps,
            step_size=args.adv_step_size,
            steps=args.adv_steps,
        )
        x_adv_norm = x_adv * 2 - 1

        x_eot_adv = pgd_eot_purifier_attack(args, diffusion, model, classifier, imgs, labels, device)
        x_eot_adv_norm = x_eot_adv * 2 - 1

        delta = run_cgpo(args, diffusion, model, classifier, x0, labels, t_batch)
        addt_recon, pred_addt = addt_loss(args, diffusion, model, x0, t_batch, delta)
        adv_recon, pred_adv = supervised_loss(diffusion, model, x0, x_adv_norm, t_batch)
        eot_recon, pred_eot = supervised_loss(diffusion, model, x0, x_eot_adv_norm, t_batch)

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
            + args.clean_weight * clean_recon
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
            clean=f"{clean_recon.item():.5f}",
            cls=f"{cls_loss.item():.5f}",
            t_min=int(t_batch.min()),
            t_max=int(t_batch.max()),
        )

    save_checkpoint(model, args.out_checkpoint)
    print(f"Saved ImageNet ResNet50-V2 PGD+EOT diffusion checkpoint to: {args.out_checkpoint}")


if __name__ == "__main__":
    main()
