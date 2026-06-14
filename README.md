# project01

Common diffusion-model utilities for image robustness research.

This public snapshot intentionally contains only reusable infrastructure code.
Project-specific training, evaluation, purification, checkpoints, datasets, and
experiment result files are excluded.

## Included

- `improved_diffusion/`: reusable diffusion model components, schedules,
  UNet definitions, samplers, logging, and training utilities.

## Excluded

- project-specific entry points such as `run.py`, `diffusion.py`, and
  `train_addt.py`
- model checkpoints and weights
- datasets and downloaded archives
- experiment spreadsheets and local caches

## Install

```bash
pip install -r requirements.txt
```

