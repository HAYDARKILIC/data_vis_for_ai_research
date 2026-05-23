"""Generate a synthetic but realistic sweep results CSV for the week-4 exercise.

The shape: 60 trials of a hyperparameter search, with hyperparams
(lr, weight_decay, batch_size, dropout, optimizer) and an outcome val_loss.
There is a real signal: low loss requires lr in [1e-4, 3e-3], optimizer == 'adamw',
and modest dropout. weight_decay and batch_size have small effects with noise.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260201)

N = 60
lr   = 10 ** RNG.uniform(np.log10(1e-5), np.log10(1e-1), N)
wd   = 10 ** RNG.uniform(np.log10(1e-7), np.log10(1e-1), N)
bs   = RNG.choice([32, 64, 128, 256, 512], N)
do   = RNG.uniform(0.0, 0.6, N)
opt  = RNG.choice(["sgd", "adam", "adamw"], p=[0.3, 0.3, 0.4], size=N)

# A simulated val_loss with realistic structure
log_lr = np.log10(lr)
lr_pen   = ((log_lr - np.log10(7e-4)) / 0.6) ** 2          # U around 7e-4
do_pen   = ((do  - 0.25) / 0.20) ** 2                       # U around 0.25
opt_pen  = np.where(opt == "adamw", 0.0,
                   np.where(opt == "adam", 0.08, 0.25))     # adamw best, sgd worst
wd_pen   = 0.03 * ((np.log10(wd) - np.log10(1e-4)) / 1.5) ** 2
bs_pen   = 0.02 * np.where(bs >= 256, (bs - 128) / 384, 0.0)

val_loss = 0.30 + 0.20 * lr_pen + 0.18 * do_pen + opt_pen + wd_pen + bs_pen
val_loss = val_loss + RNG.normal(0, 0.04, N)
val_loss = np.clip(val_loss, 0.25, 2.0)

val_acc  = 1.0 - 0.55 * np.tanh(val_loss - 0.3) - RNG.normal(0, 0.01, N)
val_acc  = np.clip(val_acc, 0.5, 0.98)

df = pd.DataFrame({
    "trial":         np.arange(N),
    "lr":            lr,
    "weight_decay":  wd,
    "batch_size":    bs,
    "dropout":       do,
    "optimizer":     opt,
    "val_loss":      val_loss.round(4),
    "val_acc":       val_acc.round(4),
})

out = Path(__file__).parent / "sweep_results.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} trials to {out}")
print(df.head())
