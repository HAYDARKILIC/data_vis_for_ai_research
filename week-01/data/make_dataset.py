"""Generate a synthetic but realistic LLM-evaluation results table.

Schema (one row per (model, benchmark, seed)):
    model              : str   — name of the model
    family             : str   — model family (e.g. 'gpt', 'llama', 'mistral')
    params_b           : float — parameter count in billions (spans 4 orders of magnitude)
    benchmark          : str   — benchmark name
    benchmark_type     : str   — 'reasoning', 'knowledge', 'coding', 'safety'
    seed               : int   — evaluation seed
    accuracy           : float — primary metric, in [0, 1]
    latency_ms         : float — median per-query latency, log-normal
    cost_per_1k_tokens : float — USD, log-normal
    release_year       : int   — calendar year
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(20260101)

MODELS = [
    # (name, family, params_b, release_year, base_skill)
    ("tiny-lm-125m",     "research", 0.125, 2021, 0.18),
    ("small-lm-350m",    "research", 0.35,  2022, 0.24),
    ("base-lm-1b",       "research", 1.0,   2022, 0.31),
    ("mid-lm-3b",        "research", 3.0,   2023, 0.39),
    ("llama-7b",         "llama",    7.0,   2023, 0.46),
    ("mistral-7b",       "mistral",  7.3,   2023, 0.49),
    ("llama-13b",        "llama",    13.0,  2023, 0.52),
    ("mixtral-8x7b",     "mistral",  46.7,  2024, 0.61),
    ("llama-70b",        "llama",    70.0,  2024, 0.64),
    ("gpt-3.5",          "gpt",      175.0, 2022, 0.58),
    ("gpt-4-class",      "gpt",      900.0, 2024, 0.74),
    ("frontier-1.5t",    "gpt",      1500.0, 2025, 0.78),
]

BENCHMARKS = [
    # (name, type, difficulty_offset)
    ("mmlu",          "knowledge",  0.00),
    ("hellaswag",     "knowledge", -0.08),  # easier
    ("arc-challenge", "reasoning",  0.04),
    ("gsm8k",         "reasoning",  0.10),  # hard
    ("math",          "reasoning",  0.22),  # very hard
    ("humaneval",     "coding",     0.06),
    ("mbpp",          "coding",     0.02),
    ("toxigen",       "safety",    -0.04),
]


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_dataset(n_seeds: int = 5) -> pd.DataFrame:
    rows = []
    for name, family, params, year, skill in MODELS:
        # log-scale latency: bigger models slower, with noise
        base_latency = 40 * (params ** 0.45)
        # cost roughly tracks params but tier-discounted for some families
        family_cost_mult = {"gpt": 1.6, "llama": 0.8, "mistral": 0.9, "research": 0.6}[family]
        base_cost = 0.0008 * (params ** 0.7) * family_cost_mult

        for bench, btype, offset in BENCHMARKS:
            # safety is scored as "non-toxic rate", which scales weakly with size
            if btype == "safety":
                bench_skill = 0.55 + 0.25 * skill + offset
            else:
                bench_skill = skill - offset
            # logit-space mean accuracy, with realistic ceiling
            bench_skill = float(np.clip(bench_skill, 0.02, 0.95))
            mu_logit = np.log(bench_skill / (1 - bench_skill))

            for seed in range(n_seeds):
                acc = _logistic(mu_logit + RNG.normal(0, 0.18))
                acc = float(np.clip(acc, 0.0, 0.995))
                latency = float(np.exp(np.log(base_latency) + RNG.normal(0, 0.25)))
                cost = float(np.exp(np.log(base_cost + 1e-6) + RNG.normal(0, 0.20)))
                rows.append(
                    dict(
                        model=name,
                        family=family,
                        params_b=params,
                        benchmark=bench,
                        benchmark_type=btype,
                        seed=seed,
                        accuracy=acc,
                        latency_ms=latency,
                        cost_per_1k_tokens=cost,
                        release_year=year,
                    )
                )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("llm_eval_results.csv"))
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    df = make_dataset(n_seeds=args.seeds)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
