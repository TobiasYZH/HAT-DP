from enum import Enum
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from autoattack import AutoAttack
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm
import foolbox as fb

from bpda_eot_attack import BPDA_EOT_Attack
from diffusion import DiffusionPurificationModel


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NROW = 16
DEFAULT_DATA_DIR_CANDIDATES = [
    "data/imagenetval/imagenetval",
    "/home/data/imagenetval/imagenetval",
    "/home/data/imagenet/val",
    "/home/data/imagenet/val_imagefolder",
    "/data/imagenetval/imagenetval",
    "/data/imagenet/val",
    "/data/imagenet/val_imagefolder",
    "imagenet_val",
    "val",
    "val_imagefolder",
    "imagenet_train_100_per_class/train",
    "imagenet_subset/train",
    "data/imagenet/train",
    "/home/data/imagenet/train",
    "/data/imagenet/train",
]


class AttackMode(Enum):
    Plain = "plain"
    AutoAttack = "auto"
    PGDAttack = "pgd"
    PGD_EOT = "pgd_eot"
    BPDA_EOT = "bpda"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ImagenetDataset(Dataset):
    def __init__(self, data_dir, max_samples=None, seed=42, random_subset=True):
        self.transform = Compose([
            Resize(256, antialias=True),
            CenterCrop(256),
            ToTensor(),
        ])
        self.dataset = torchvision.datasets.ImageFolder(data_dir, transform=self.transform)
        self.indices = None
        if max_samples is not None:
            sample_count = min(int(max_samples), len(self.dataset))
            if sample_count < 0:
                raise ValueError("--max-samples must be non-negative.")
            if random_subset:
                rng = np.random.default_rng(seed)
                self.indices = rng.choice(len(self.dataset), size=sample_count, replace=False).tolist()
            else:
                self.indices = list(range(sample_count))

    def __len__(self):
        if self.indices is None:
            return len(self.dataset)
        return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        return self.dataset[idx]


def center_crop_224(x):
    _, _, h, w = x.shape
    if h < 224 or w < 224:
        return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    top = (h - 224) // 2
    left = (w - 224) // 2
    return x[:, :, top:top + 224, left:left + 224]


class ImageNetClassifier(nn.Module):
    def __init__(self, model_name="resnet50", weights_name="v2"):
        super().__init__()
        if model_name == "resnet50":
            if weights_name == "v1":
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            elif weights_name in {"v2", "default"}:
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            else:
                raise ValueError(f"Unknown ResNet50 weights: {weights_name}")
            self.model = torchvision.models.resnet50(weights=weights)
        elif model_name == "resnet152":
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.model = torchvision.models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported ImageNet classifier: {model_name}")
        self.model.eval()

    def forward(self, x):
        mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
        std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
        x = center_crop_224(x)
        x = (x - mean) / std
        return self.model(x)


class DModel(nn.Module):
    def __init__(self, base_model, pure_model, T, scale):
        super().__init__()
        self.T = T
        self.scale = scale
        self.base_model = base_model
        self.pure_model = pure_model

    def forward(self, imgs, mode="purify_and_classify"):
        try:
            imgs = imgs.raw
        except AttributeError:
            pass

        if mode == "purify":
            imgs = imgs * 2 - 1
            p_imgs = self.pure_model.denoise(imgs, self.T, self.scale)
            return torch.clamp((p_imgs + 1) / 2, 0, 1)

        if mode == "classify":
            return self.base_model(imgs)

        if mode == "purify_and_classify":
            imgs = imgs * 2 - 1
            p_imgs = self.pure_model.denoise(imgs, self.T, self.scale)
            p_imgs = torch.clamp((p_imgs + 1) / 2, 0, 1)
            return self.base_model(p_imgs)

        raise ValueError(f"Unknown DModel mode: {mode}")

    def surrogate_purify(self, imgs, t=None):
        """Differentiable one-step diffusion surrogate for adaptive attacks."""
        t = self.T if t is None else int(t)
        imgs = imgs * 2 - 1
        t_batch = torch.full((imgs.shape[0],), t, device=imgs.device, dtype=torch.long)
        x_t = self.pure_model.diffusion.q_sample(x_start=imgs, t=t_batch)
        pred_xstart = self.pure_model.diffusion.p_mean_variance(
            self.pure_model.model,
            x_t,
            t_batch,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
        )["pred_xstart"]
        pred_xstart = straight_through_clamp(pred_xstart, -1, 1)
        return straight_through_clamp((pred_xstart + 1) / 2, 0, 1)


def pgd_attack_model(fmodel, imgs, labels, eps, steps):
    attack = fb.attacks.LinfPGD(rel_stepsize=0.25, steps=steps)
    _, clipped, _ = attack(fmodel, imgs, labels, epsilons=eps)
    return clipped


def linf_project(adv, clean, eps):
    adv = torch.max(torch.min(adv, clean + eps), clean - eps)
    return torch.clamp(adv, 0, 1)


def straight_through_clamp(x, min_value, max_value):
    clipped = torch.clamp(x, min_value, max_value)
    return x + (clipped - x).detach()


def defense_input_grad(model, adv, labels, surrogate_t):
    adv_surrogate = adv.detach().requires_grad_(True)
    purified = model.surrogate_purify(adv_surrogate, t=surrogate_t)
    logits = model(purified, mode="classify")
    losses = F.cross_entropy(logits, labels, reduction="none")
    loss = losses.sum()
    grad = torch.autograd.grad(loss, adv_surrogate, only_inputs=True)[0]
    return grad.detach(), losses.detach()


def pgd_eot_attack_model(
    model,
    imgs,
    labels,
    eps,
    steps,
    step_size,
    eot_iters,
    random_start,
    surrogate_t,
    keep_best,
    restarts,
):
    was_training = model.training
    model.eval()

    eot_iters = max(int(eot_iters), 1)
    restarts = max(int(restarts), 1)
    best_adv = imgs.detach().clone()
    best_loss = torch.full((imgs.shape[0],), -float("inf"), device=imgs.device)

    for restart_idx in range(restarts):
        adv = imgs.detach()
        if random_start:
            adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
            adv = linf_project(adv, imgs, eps)

        for _ in range(steps):
            grad_sum = torch.zeros_like(adv)
            loss_sum = torch.zeros((adv.shape[0],), device=adv.device)
            for _ in range(eot_iters):
                grad, losses = defense_input_grad(
                    model,
                    adv,
                    labels,
                    surrogate_t=surrogate_t,
                )
                grad_sum += grad
                loss_sum += losses

            avg_losses = loss_sum / eot_iters
            if keep_best:
                improve = avg_losses > best_loss
                best_loss[improve] = avg_losses[improve]
                best_adv[improve] = adv.detach()[improve]

            grad = grad_sum / eot_iters
            with torch.no_grad():
                adv = adv + step_size * grad.sign()
                adv = linf_project(adv, imgs, eps)

        if not keep_best and restart_idx == restarts - 1:
            best_adv = adv.detach()

    if was_training:
        model.train()
    return best_adv.detach()


def auto_attack_model(adversary, imgs, labels, bs):
    return adversary.run_standard_evaluation(imgs, labels, bs=bs)


def bpda_eot_model(adversary, imgs, labels, bs):
    class_batch, ims_adv_batch = adversary.attack_all(imgs, labels, batch_size=bs)
    init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
    robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]
    print("BPDA+EOT batch init acc: {:.2%}, robust acc: {:.2%}".format(init_acc, robust_acc))
    return ims_adv_batch.to(imgs.device)


def build_attack(attack_mode, classifier_fmodel, classifier_model, dmodel, args, device):
    if attack_mode == AttackMode.BPDA_EOT:
        adversary = BPDA_EOT_Attack(
            dmodel,
            adv_eps=args.eps,
            adv_steps=args.bpda_steps,
            eot_defense_reps=args.eot_defense_reps,
            eot_attack_reps=args.eot_attack_reps,
        )
        return adversary, bpda_eot_model

    if attack_mode == AttackMode.AutoAttack:
        adversary = AutoAttack(
            classifier_model,
            norm="Linf",
            eps=args.eps,
            version="standard",
            device=device,
        )
        return adversary, auto_attack_model

    if attack_mode == AttackMode.PGDAttack:
        return classifier_fmodel, pgd_attack_model

    if attack_mode == AttackMode.PGD_EOT:
        return dmodel, pgd_eot_attack_model

    return None, None


def maybe_save_debug_images(args, imgs, adv_imgs, idx):
    if not args.save_images or idx > 0:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    save_image(imgs, os.path.join(args.output_dir, "clean_imgs.png"), nrow=NROW)
    if adv_imgs is not None:
        save_image(adv_imgs, os.path.join(args.output_dir, "adv_imgs.png"), nrow=NROW)


def logits_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def purify_ensemble_logits(dmodel, imgs, reps):
    reps = max(int(reps), 1)
    logits_sum = None
    for _ in range(reps):
        logits = dmodel(imgs, mode="purify_and_classify")
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / reps


@torch.no_grad()
def defense_accuracy(dmodel, imgs, labels, purify_ensemble):
    logits = purify_ensemble_logits(dmodel, imgs, purify_ensemble)
    return logits_accuracy(logits, labels)


@torch.no_grad()
def print_attack_diagnostics(dmodel, base_model, imgs, adv_imgs, labels, idx, purify_ensemble):
    delta = (adv_imgs - imgs).detach()
    linf = delta.flatten(1).abs().max(dim=1).values
    l2 = delta.flatten(1).pow(2).sum(dim=1).sqrt()

    base_clean_logits = base_model(imgs)
    base_adv_logits = base_model(adv_imgs)
    defense_clean_logits = purify_ensemble_logits(dmodel, imgs, purify_ensemble)
    defense_adv_logits = purify_ensemble_logits(dmodel, adv_imgs, purify_ensemble)

    base_clean_loss = F.cross_entropy(base_clean_logits, labels).item()
    base_adv_loss = F.cross_entropy(base_adv_logits, labels).item()
    defense_clean_loss = F.cross_entropy(defense_clean_logits, labels).item()
    defense_adv_loss = F.cross_entropy(defense_adv_logits, labels).item()

    print(
        "Attack diagnostic "
        f"batch={idx}: "
        f"purify_ensemble={purify_ensemble}, "
        f"linf(mean/max)={linf.mean().item():.6f}/{linf.max().item():.6f}, "
        f"l2(mean)={l2.mean().item():.4f}, "
        f"base_acc(clean/adv)={logits_accuracy(base_clean_logits, labels):.4f}/"
        f"{logits_accuracy(base_adv_logits, labels):.4f}, "
        f"def_acc(clean/adv)={logits_accuracy(defense_clean_logits, labels):.4f}/"
        f"{logits_accuracy(defense_adv_logits, labels):.4f}, "
        f"base_ce(clean/adv)={base_clean_loss:.4f}/{base_adv_loss:.4f}, "
        f"def_ce(clean/adv)={defense_clean_loss:.4f}/{defense_adv_loss:.4f}"
    )


def evaluate(dataloader, attack_mode, args, device):
    print(
        f"{attack_mode.value} attack, T={args.T}, scale={args.scale}, "
        f"model={args.model}, weights={args.resnet_weights}, eps={args.eps}, "
        f"purify_ensemble={args.purify_ensemble}"
    )
    if attack_mode == AttackMode.PGD_EOT:
        print(
            f"PGD+EOT config: steps={args.pgd_steps}, step_size={args.pgd_step_size}, "
            f"eot_iters={args.pgd_eot_iters}, random_start={args.pgd_random_start}, "
            f"keep_best={args.pgd_keep_best}, restarts={args.pgd_restarts}, "
            f"target=diffusion+classifier, gradient={args.pgd_gradient}, "
            f"surrogate_t={args.pgd_surrogate_t}"
        )
    base_model = ImageNetClassifier(args.model, args.resnet_weights).to(device).eval()
    base_model.requires_grad_(False)
    classifier_fmodel = fb.PyTorchModel(base_model, bounds=(0, 1), device=device)
    diffusion_model = DiffusionPurificationModel(
        device=device,
        checkpoint_path=args.diffusion_checkpoint,
    )
    diffusion_model.requires_grad_(False)
    dmodel = DModel(base_model, diffusion_model, args.T, args.scale).to(device).eval()
    dmodel.requires_grad_(False)

    adversary, attack_fn = build_attack(attack_mode, classifier_fmodel, base_model, dmodel, args, device)

    clean_sum = 0.0
    robust_sum = 0.0
    sample_count = 0

    with tqdm(dataloader, dynamic_ncols=True) as progress:
        for idx, (imgs, labels) in enumerate(progress):
            imgs = torch.clamp(imgs.to(device), 0, 1)
            labels = labels.to(device)

            clean_acc = defense_accuracy(dmodel, imgs, labels, args.purify_ensemble)

            if attack_mode == AttackMode.Plain:
                adv_imgs = None
                robust_acc = clean_acc
            elif attack_mode == AttackMode.PGDAttack:
                adv_imgs = attack_fn(adversary, imgs, labels, args.eps, args.pgd_steps)
                robust_acc = defense_accuracy(dmodel, torch.clamp(adv_imgs, 0, 1), labels, args.purify_ensemble)
            elif attack_mode == AttackMode.PGD_EOT:
                adv_imgs = attack_fn(
                    adversary,
                    imgs,
                    labels,
                    args.eps,
                    args.pgd_steps,
                    args.pgd_step_size,
                    args.pgd_eot_iters,
                    args.pgd_random_start,
                    args.pgd_surrogate_t,
                    args.pgd_keep_best,
                    args.pgd_restarts,
                )
                robust_acc = defense_accuracy(dmodel, torch.clamp(adv_imgs, 0, 1), labels, args.purify_ensemble)
                if args.attack_diagnostics and idx < args.diagnostic_batches:
                    print_attack_diagnostics(
                        dmodel,
                        base_model,
                        imgs,
                        torch.clamp(adv_imgs, 0, 1),
                        labels,
                        idx,
                        args.purify_ensemble,
                    )
            else:
                adv_imgs = attack_fn(adversary, imgs, labels, args.bs)
                robust_acc = defense_accuracy(dmodel, torch.clamp(adv_imgs, 0, 1), labels, args.purify_ensemble)

            maybe_save_debug_images(args, imgs, adv_imgs, idx)

            batch_size = imgs.shape[0]
            clean_sum += clean_acc * batch_size
            robust_sum += robust_acc * batch_size
            sample_count += batch_size

            progress.set_postfix({
                "clean": clean_sum / sample_count,
                "robust": robust_sum / sample_count,
                "n": sample_count,
            })

    print(f"Clean ACC: {clean_sum / sample_count:.5%}")
    print(f"Robust ACC: {robust_sum / sample_count:.5%}")


def parse_mode(mode):
    if mode == "plain":
        return AttackMode.Plain
    if mode == "pgd":
        return AttackMode.PGDAttack
    if mode in {"pgd_eot", "pgd-eot", "pgd+eot"}:
        return AttackMode.PGD_EOT
    if mode == "bpda":
        return AttackMode.BPDA_EOT
    if mode == "auto":
        return AttackMode.AutoAttack
    raise ValueError(f"Unknown mode: {mode}")


def looks_like_imagefolder(path):
    path = Path(path).expanduser()
    if not path.is_dir():
        return False
    try:
        return any(child.is_dir() for child in path.iterdir())
    except OSError:
        return False


def resolve_data_dir(data_dir):
    if data_dir:
        path = Path(data_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"--data-dir does not exist or is not a directory: {path}")
        if not looks_like_imagefolder(path):
            raise RuntimeError(
                f"--data-dir is not a valid ImageFolder directory: {path}. "
                "Expected class subfolders like n01440764/."
            )
        return str(path.resolve())

    for candidate in DEFAULT_DATA_DIR_CANDIDATES:
        path = Path(candidate).expanduser()
        if looks_like_imagefolder(path):
            return str(path.resolve())

    candidates = "\n  - ".join(DEFAULT_DATA_DIR_CANDIDATES)
    raise RuntimeError(
        "No ImageNet val directory was found. Pass --data-dir explicitly.\n"
        "Expected ImageFolder layout: <data-dir>/<class_id>/*.JPEG\n"
        "Common candidates checked:\n"
        f"  - {candidates}\n"
        "For run.py evaluation, use ImageNet val, for example:\n"
        "  python run.py --data-dir /home/data/imagenetval/imagenetval\n"
        "Only use imagenet_train_100_per_class/train if you intentionally want a train-subset sanity check."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="pgd+eot",
        choices=["plain", "pgd", "pgd_eot", "pgd-eot", "pgd+eot", "bpda", "auto"],
    )
    parser.add_argument("--T", type=int, default=110)
    parser.add_argument("--scale", type=float, default=2000)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet152"])
    parser.add_argument("--resnet-weights", type=str, default="v2", choices=["v1", "v2", "default"])
    parser.add_argument(
        "--data-dir",
        type=str,
        default="imagenet_val",
        help=(
            "ImageFolder directory containing class subfolders. "
            "If omitted, common local/server ImageNet paths are checked automatically."
        ),
    )
    parser.add_argument("--diffusion-checkpoint", type=str, default="checkpoints/resnet50_v2_pgd_eot_4_255.pt")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=512,
        help="Number of evaluation images. Default 512 uses a fixed random subset; set <=0 for full validation.",
    )
    parser.add_argument(
        "--sequential-subset",
        action="store_true",
        help="Use the first --max-samples images instead of a seed-fixed random subset.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument(
        "--pgd-step-size",
        type=float,
        default=0.007,
        help="PGD step size. Defaults to eps/4. Pei 2025 ImageNet uses 0.007.",
    )
    parser.add_argument(
        "--pgd-eot-iters",
        type=int,
        default=20,
        help="Number of EOT gradient samples for --mode pgd_eot.",
    )
    parser.add_argument(
        "--purify-ensemble",
        type=int,
        default=5,
        help="Average logits over this many stochastic purification forwards only for final accuracy evaluation.",
    )
    parser.add_argument(
        "--no-pgd-random-start",
        dest="pgd_random_start",
        action="store_false",
        help="Disable random initialization inside the Linf epsilon ball for PGD+EOT.",
    )
    parser.add_argument(
        "--pgd-gradient",
        type=str,
        default="surrogate",
        choices=["surrogate"],
        help="Gradient estimator for PGD+EOT. Only surrogate diffusion-process gradients are supported.",
    )
    parser.add_argument("--pgd-surrogate-t", type=int, default=None)
    parser.add_argument(
        "--no-pgd-keep-best",
        dest="pgd_keep_best",
        action="store_false",
        help="Return the final PGD iterate instead of the highest-loss iterate seen during the attack.",
    )
    parser.add_argument("--pgd-restarts", type=int, default=1)
    parser.set_defaults(pgd_random_start=True)
    parser.set_defaults(pgd_keep_best=True)
    parser.add_argument("--bpda-steps", type=int, default=40)
    parser.add_argument("--eot-defense-reps", type=int, default=2)
    parser.add_argument("--eot-attack-reps", type=int, default=1)
    parser.add_argument(
        "--attack-diagnostics",
        action="store_true",
        help="Print attack sanity checks for the first diagnostic batches.",
    )
    parser.add_argument("--diagnostic-batches", type=int, default=1)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-dir", type=str, default="res")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None
    if args.pgd_step_size is None:
        args.pgd_step_size = args.eps / 4
    if args.pgd_surrogate_t is None:
        args.pgd_surrogate_t = args.T

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data dir: {data_dir}")
    dataset = ImagenetDataset(
        data_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        random_subset=not args.sequential_subset,
    )
    if args.max_samples is None:
        print(f"Evaluation subset: full validation set ({len(dataset)} images)")
    else:
        subset_type = "fixed random" if not args.sequential_subset else "sequential"
        print(
            f"Evaluation subset: {subset_type} {len(dataset)} / {len(dataset.dataset)} images "
            f"(seed={args.seed})"
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    evaluate(dataloader, parse_mode(args.mode), args, device)


if __name__ == "__main__":
    main()
