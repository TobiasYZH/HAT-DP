from enum import Enum
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm
import foolbox as fb
from autoattack import AutoAttack
from robustbench.utils import load_model
from torchattacks import EOTPGD

from bpda_eot_attack import BPDA_EOT_Attack
import diffusion as diffusion_module
from diffusion import DiffusionPurificationModel


os.environ["OMP_NUM_THREADS"] = "8"

args = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NROW = 16


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AttackMode(Enum):
    Plain = "plain"
    AutoAttackLinf = "AutoAttack-Linf"
    AutoAttackL2 = "AutoAttack-L2"
    PGDAttack = "PGDAttack"
    BPDA_EOT = "BPDA_EOT"
    PGD_EOT = "PGD_EOT"
    PGD_EOT_L2 = "PGD_EOT_L2"
    AutoAttack_EOT = "AutoAttack_EOT"


class BModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = load_model(model_name="Standard", dataset="cifar10", threat_model="Linf")

    def forward(self, imgs, mode="purify_and_classify"):
        if mode == "purify":
            return imgs
        if mode in ("classify", "purify_and_classify"):
            return self.base_model(imgs)
        raise ValueError(f"Unknown mode: {mode}")


class DModel(torch.nn.Module):
    def __init__(
            self,
            T,
            scale,
            purify_ensemble=1,
            diffusion_checkpoint="cifar10_uncond_50M_500K.pt",
            save_intermediates=True,
    ):
        super().__init__()
        self.T = T
        self.scale = scale
        self.purify_ensemble = max(1, purify_ensemble)
        self.save_intermediates = save_intermediates
        self.base_model = BModel()
        self.pure_model = DiffusionPurificationModel(
            device=device,
            guide_type="osgd",
            checkpoint_path=diffusion_checkpoint,
        )

    def diffusion_step(self, imgs):
        diffusion_imgs = self.pure_model.denoise(imgs, self.T, self.scale)
        if self.save_intermediates:
            save_image((diffusion_imgs + 1) / 2, os.path.join("./", "diff_cifar10_guided.png"), nrow=NROW)
        return diffusion_imgs

    def purify(self, imgs):
        imgs_ = Normalize(0.5, 0.5)(imgs)
        p_imgs = self.diffusion_step(imgs_)
        return torch.clamp((p_imgs + 1) / 2, 0, 1)

    def forward(self, imgs, mode="purify_and_classify"):
        if mode == "purify":
            return self.purify(imgs)
        if mode == "classify":
            return self.base_model(imgs)
        if mode == "purify_and_classify":
            logits = None
            for _ in range(self.purify_ensemble):
                p_imgs = self.purify(imgs)
                current_logits = self.base_model(p_imgs)
                logits = current_logits if logits is None else logits + current_logits
            return logits / self.purify_ensemble
        raise ValueError(f"Unknown mode: {mode}")


def PGDAttack_model(fmodel, imgs, labels, bs=128):
    attack = fb.attacks.LinfPGD(rel_stepsize=0.25, steps=10)
    _, clipped, _ = attack(fmodel, imgs, labels, epsilons=8 / 255)
    save_image(clipped, os.path.join("./", "adv_cifar10.png"), nrow=NROW)
    return clipped


def autoAttack_model(adversary, imgs, labels, bs=128):
    x_adv = adversary.run_standard_evaluation(imgs, labels, bs=bs)
    save_image(x_adv, os.path.join("./", "auto_adv_cifar10.png"), nrow=NROW)
    return x_adv


def BPDAEOT_model(adversary, imgs, labels, bs=128):
    _, ims_adv_batch = adversary.attack_all(imgs, labels, batch_size=1)
    ims_adv_batch = ims_adv_batch.to(device)
    save_image(ims_adv_batch, os.path.join("./", "bpda_adv_cifar10.png"), nrow=NROW)
    return ims_adv_batch


def PGDEOT_model(model, imgs, labels, bs=128):
    attack = EOTPGD(model, eps=8 / 255, alpha=args.alpha, steps=20, eot_iter=20, random_start=True)
    adv_images = attack(imgs, labels)
    save_image(adv_images, os.path.join("./", "pgdeot_adv_cifar10.png"), nrow=NROW)
    return adv_images


def normalize_l2(tensor):
    norm = tensor.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
    return tensor / norm.view(-1, 1, 1, 1)


def project_l2(adv, clean, eps):
    delta = adv - clean
    norm = delta.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
    factor = torch.clamp(eps / norm, max=1.0)
    return torch.clamp(clean + delta * factor.view(-1, 1, 1, 1), 0, 1)


class L2PGDEOT:
    def __init__(self, model, eps, alpha, steps, eot_iter, random_start=True):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.eot_iter = eot_iter
        self.random_start = random_start

    def __call__(self, imgs, labels):
        clean = imgs.detach()
        if self.random_start:
            direction = normalize_l2(torch.randn_like(clean))
            dimensions = clean[0].numel()
            radius = torch.rand(clean.shape[0], 1, 1, 1, device=clean.device)
            radius = radius.pow(1.0 / dimensions) * self.eps
            adv = project_l2(clean + radius * direction, clean, self.eps)
        else:
            adv = clean.clone()

        for _ in range(self.steps):
            gradient = torch.zeros_like(adv)
            for _ in range(self.eot_iter):
                current_adv = adv.detach().requires_grad_(True)
                loss = F.cross_entropy(self.model(current_adv), labels)
                current_gradient = torch.autograd.grad(
                    loss,
                    current_adv,
                    only_inputs=True,
                )[0]
                gradient.add_(current_gradient.detach())

            gradient.div_(self.eot_iter)
            with torch.no_grad():
                adv = adv + self.alpha * normalize_l2(gradient)
                adv = project_l2(adv, clean, self.eps)

        return adv.detach()


def PGDEOTL2_model(model, imgs, labels, bs=128):
    attack = L2PGDEOT(
        model,
        eps=args.pgd_eot_l2_eps,
        alpha=args.pgd_eot_l2_alpha,
        steps=args.pgd_eot_l2_steps,
        eot_iter=args.pgd_eot_l2_iters,
        random_start=args.pgd_eot_l2_random_start,
    )
    adv_images = attack(imgs, labels)
    max_l2 = (adv_images - imgs).flatten(1).norm(p=2, dim=1).max().item()
    if max_l2 > args.pgd_eot_l2_eps + 1e-6:
        raise RuntimeError(
            f"L2 PGD projection failed: observed {max_l2}, "
            f"budget is {args.pgd_eot_l2_eps}."
        )
    save_image(adv_images, os.path.join("./", "pgdeot_l2_adv_cifar10.png"), nrow=NROW)
    return adv_images


def AutoAttackEOT_model(adversary, imgs, labels, bs=128):
    x_adv = adversary.run_standard_evaluation(imgs, labels, bs=bs)
    save_image(x_adv, os.path.join("./", "auto_eot_adv_cifar10.png"), nrow=NROW)
    return x_adv


def attack(
        dataloader,
        batch_size,
        attack_mode,
        T,
        scale,
        purify_ensemble,
        diffusion_checkpoint,
        aa_linf_eps,
        aa_l2_eps,
):
    print(
        f"{attack_mode.value} attack, T = {T}, scale = {scale}, "
        f"purify_ensemble = {purify_ensemble}, diffusion_checkpoint = {diffusion_checkpoint}"
    )
    model = BModel().to(device=device).eval()
    is_l2_eot = attack_mode == AttackMode.PGD_EOT_L2
    if is_l2_eot:
        diffusion_module.save_image = lambda *unused_args, **unused_kwargs: None

    dmodel = DModel(
        T,
        scale,
        purify_ensemble=purify_ensemble,
        diffusion_checkpoint=diffusion_checkpoint,
        save_intermediates=not is_l2_eot,
    ).to(device=device).eval()
    print("model done!")

    if attack_mode == AttackMode.BPDA_EOT:
        adversary = BPDA_EOT_Attack(
            dmodel,
            adv_eps=args.bpda_eps,
            adv_eta=args.bpda_adv_eta,
            adv_steps=args.bpda_adv_steps,
            eot_attack_reps=args.bpda_eot_attack_reps,
            eot_defense_reps=args.bpda_eot_defense_reps,
        )
        attack_model = BPDAEOT_model
    elif attack_mode == AttackMode.AutoAttackLinf:
        adversary = AutoAttack(
            model,
            norm="Linf",
            eps=aa_linf_eps,
            version="standard",
            verbose=False,
            device=device,
        )
        attack_model = autoAttack_model
    elif attack_mode == AttackMode.AutoAttackL2:
        adversary = AutoAttack(
            model,
            norm="L2",
            eps=aa_l2_eps,
            version="standard",
            verbose=False,
            device=device,
        )
        attack_model = autoAttack_model
    elif attack_mode == AttackMode.PGDAttack:
        adversary = fb.PyTorchModel(model, bounds=(0, 1), device=device)
        attack_model = PGDAttack_model
    elif attack_mode == AttackMode.PGD_EOT:
        adversary = dmodel
        attack_model = PGDEOT_model
    elif attack_mode == AttackMode.PGD_EOT_L2:
        adversary = dmodel
        attack_model = PGDEOTL2_model
    elif attack_mode == AttackMode.AutoAttack_EOT:
        adversary = AutoAttack(
            dmodel,
            norm="Linf",
            eps=8 / 255,
            verbose=False,
            version="rand",
            device=device,
        )
        attack_model = AutoAttackEOT_model
    else:
        adversary = None
        attack_model = None

    clean_correct = 0
    robust_correct = 0
    total = 0

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for idx, (imgs, labels) in enumerate(tqdmDataLoader):
            imgs = torch.clamp(imgs, 0, 1).to(device)
            labels = labels.to(device)
            total += labels.size(0)
            save_image(imgs, os.path.join("./", "clean_imgs.png"), nrow=NROW)

            with torch.no_grad():
                clean_outputs = dmodel(imgs)
                clean_correct += (clean_outputs.argmax(1) == labels).sum().item()

            if attack_mode == AttackMode.Plain:
                adv_imgs = imgs
            else:
                adv_imgs = attack_model(adversary, imgs, labels, batch_size)

            torch.cuda.empty_cache()
            adv_imgs = torch.clamp(adv_imgs, 0, 1)
            with torch.no_grad():
                robust_outputs = dmodel(adv_imgs)
                robust_correct += (robust_outputs.argmax(1) == labels).sum().item()

            clean_acc = clean_correct / total
            robust_acc = robust_correct / total
            tqdmDataLoader.set_postfix({
                "batch": idx,
                "clean_acc": f"{clean_acc:.4f}",
                "robust_acc": f"{robust_acc:.4f}",
            })

    clean_acc = clean_correct / total if total > 0 else 0
    robust_acc = robust_correct / total if total > 0 else 0

    print("\n" + "=" * 50)
    print(f"Attack Mode : {attack_mode.value}")
    print(f"T / scale   : {T} / {scale}")
    print(f"AA eps      : Linf={aa_linf_eps}, L2={aa_l2_eps}")
    if attack_mode == AttackMode.PGD_EOT_L2:
        print(
            f"PGD+EOT L2  : eps={args.pgd_eot_l2_eps}, alpha={args.pgd_eot_l2_alpha}, "
            f"steps={args.pgd_eot_l2_steps}, eot={args.pgd_eot_l2_iters}"
        )
    print(f"Ensemble    : {purify_ensemble}")
    print(f"Checkpoint  : {diffusion_checkpoint}")
    print(f"Clean ACC   : {clean_acc:.4%}")
    print(f"Robust ACC  : {robust_acc:.4%}")
    print("=" * 50 + "\n")


def main(
        load_batch_size,
        attack_mode,
        T,
        scale,
        purify_ensemble,
        diffusion_checkpoint,
        aa_linf_eps,
        aa_l2_eps,
        seed,
):
    set_seed(seed)
    val_dataset = torchvision.datasets.CIFAR10(
        "CIFAR10",
        train=False,
        download=True,
        transform=Compose([ToTensor()]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        load_batch_size,
        shuffle=False,
        num_workers=4,
        generator=generator,
        worker_init_fn=lambda worker_id: set_seed(seed + worker_id),
    )

    attack(
        dataloader,
        load_batch_size,
        attack_mode,
        T,
        scale,
        purify_ensemble,
        diffusion_checkpoint,
        aa_linf_eps,
        aa_l2_eps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="bpda+eot")
    parser.add_argument("--T", type=int, default=240)
    parser.add_argument("--scale", type=float, default=92000)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--purify-ensemble", type=int, default=4)
    parser.add_argument("--diffusion-checkpoint", type=str, default="checkpoints/strong_supervised_addt_latest2_last.pt")
    parser.add_argument("--aa-linf-eps", type=float, default=8 / 255)
    parser.add_argument("--aa-l2-eps", type=float, default=0.5)
    parser.add_argument("--bpda-eps", type=float, default=8 / 255)
    parser.add_argument("--bpda-adv-eta", type=float, default=2 / 255)
    parser.add_argument("--bpda-adv-steps", type=int, default=40)
    parser.add_argument("--bpda-eot-attack-reps", type=int, default=15)
    parser.add_argument("--bpda-eot-defense-reps", type=int, default=20)
    parser.add_argument("--pgd-eot-l2-eps", type=float, default=0.5)
    parser.add_argument("--pgd-eot-l2-alpha", type=float, default=0.007)
    parser.add_argument("--pgd-eot-l2-steps", type=int, default=200)
    parser.add_argument("--pgd-eot-l2-iters", type=int, default=20)
    parser.add_argument(
        "--pgd-eot-l2-random-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.pgd_eot_l2_eps <= 0 or args.pgd_eot_l2_alpha <= 0:
        raise ValueError("L2 PGD+EOT epsilon and alpha must be positive.")
    if args.pgd_eot_l2_steps < 1 or args.pgd_eot_l2_iters < 1:
        raise ValueError("L2 PGD+EOT steps and EOT iterations must be positive.")
    if args.bpda_eps <= 0 or args.bpda_adv_eta <= 0:
        raise ValueError("BPDA+EOT epsilon and adv_eta must be positive.")
    if args.bpda_adv_steps < 1 or args.bpda_eot_attack_reps < 1 or args.bpda_eot_defense_reps < 1:
        raise ValueError("BPDA+EOT adv_steps, eot_attack_reps, and eot_defense_reps must be positive.")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    set_seed(args.seed)
    print(f"Running on single device: {device}")
    print(f"Random seed: {args.seed}")

    attackMode = AttackMode.Plain
    if args.mode == "pgd":
        attackMode = AttackMode.PGDAttack
    if args.mode == "bpda+eot":
        attackMode = AttackMode.BPDA_EOT
    if args.mode in ("auto", "auto_linf"):
        attackMode = AttackMode.AutoAttackLinf
    if args.mode == "auto_l2":
        attackMode = AttackMode.AutoAttackL2
    if args.mode == "pe_linf":
        attackMode = AttackMode.PGD_EOT
    if args.mode in ("pe_l2", "pgd_eot_l2"):
        attackMode = AttackMode.PGD_EOT_L2
    if args.mode == "ae":
        attackMode = AttackMode.AutoAttack_EOT

    main(
        args.bs,
        attackMode,
        args.T,
        args.scale,
        args.purify_ensemble,
        args.diffusion_checkpoint,
        args.aa_linf_eps,
        args.aa_l2_eps,
        args.seed,
    )
