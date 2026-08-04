# HAT-DP

This repository contains the implementation accompanying the paper **“A Hierarchical Adversarial Training Framework for Diffusion Purifier Adaptation.”**

HAT-DP adapts a pretrained diffusion purifier with hierarchical adversarial threats while keeping the target classifier frozen.

## Repository structure

- `CIFAR10/`: CIFAR-10 experiments and diffusion-purifier implementation.
- `ImageNet/`: ImageNet experiments and guided-diffusion implementation.
- `requirements.txt`: core Python dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Checkpoints and data

Model checkpoints, pretrained weights, datasets, generated samples, and experiment outputs are not included in this repository. Place the required checkpoints and datasets at the paths expected by the corresponding scripts before running training or evaluation.

## Citation

Citation information will be added after publication.
