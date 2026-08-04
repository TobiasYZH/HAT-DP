# HAT-DP

This repository is the implementation accompanying the paper **“A Hierarchical Adversarial Training Framework for Diffusion Purifier Adaptation.”**

HAT-DP adapts a pretrained diffusion purifier with hierarchical adversarial threats while keeping the target classifier frozen.

## Repository structure

- `CIFAR10/`: CIFAR-10 experiments and diffusion-purifier implementation.
- `ImageNet/`: ImageNet experiments and guided-diffusion implementation.
- `requirements.txt`: core Python dependencies.

## Installation

```bash
pip install -r requirements.txt
```
## Pre-trained Models
You can download pretrained models here:

- DDPM on ImageNet https://github.com/openai/guided-diffusion
  -  checkpoint 256x256_diffusion_uncond.pt
- DDPM on Cifar10 https://github.com/openai/improved-diffusion
  - checkpoint cifar10_uncond_50M_500K.pt

## Citation

Citation information will be added after publication.
