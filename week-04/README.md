# Week 4 — MLOps experiment tracking and live metric monitoring

> Hyperparameter sweep analysis, real-time training/validation metric tracking, and
> multi-dimensional model artefact versioning.

## Learning objectives

By the end of this week you should be able to:

1. Instrument a PyTorch training loop with **MLflow** and / or **Weights & Biases**
   so that every run's metrics, hyperparameters, and artefacts are logged automatically.
2. Compare runs in the tracker UI using **parallel-coordinates** and
   **parameter-importance** views, and reproduce the same plots offline in Matplotlib /
   Plotly when you need them in a paper.
3. Run a small **hyperparameter sweep** (grid, random, or Bayesian via Optuna) and
   visualize its results to pick the next experiment.
4. Version model artefacts and tie them back to the runs that produced them, so a chart
   shown in a paper can be re-rendered from a checkpoint.

## Contents

| File | Purpose |
|------|---------|
| `notebooks/01-theory.ipynb` | Tracking concepts, sweep design, what to log |
| `notebooks/02-lab.ipynb` | Train and sweep a small CNN on FashionMNIST under MLflow + W&B |
| `exercises/` | Build a parallel-coordinates plot from a sweep dataframe |
| `solutions/` | Reference solution |

## Dataset

FashionMNIST, downloaded automatically by torchvision on first run (~30 MB).

## Setup notes

The lab works fully offline: MLflow logs to `./mlruns` and W&B is run with
`WANDB_MODE=offline`, producing local artefacts that you can sync to the W&B cloud
later or just inspect locally with `wandb sync`. **No account or API key needed to
follow the lab.**
