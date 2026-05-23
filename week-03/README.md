# Week 3 — Model diagnostics and evaluation visualization

> Visual diagnosis of classification and regression models — decision boundaries,
> residual analysis, and performance curves.

## Learning objectives

By the end of this week you should be able to:

1. Choose between **ROC**, **PR**, and **calibration** curves and explain when each is
   the right summary of a classifier.
2. Produce **decision-boundary plots** for low-dimensional toy problems and explain why
   the same trick is useful on a 2D projection of a real classifier.
3. Read **residual** and **prediction-error** plots for regression models and identify
   heteroskedasticity, non-linearity, and outliers from them.
4. Use **Yellowbrick** and **scikit-plot** as productivity tools, and fall back to
   custom Matplotlib when they don't cover the case you need.
5. Build a **confusion matrix with row-normalization** that reports per-class recall
   honestly even under heavy class imbalance.

## Contents

| File | Purpose |
|------|---------|
| `notebooks/01-theory.ipynb` | Diagnostic curve theory and what each one tells you |
| `notebooks/02-lab.ipynb` | Full diagnostic suite for a classifier and a regressor |
| `exercises/` | Two exercises with starter code |
| `solutions/` | Reference solutions |

## Datasets

We use scikit-learn built-ins (no download required) — the breast-cancer Wisconsin
dataset for classification and the California housing dataset for regression.

## Suggested reading

- Kuhn & Johnson (2019), *Feature Engineering and Selection*, ch. 11 (model performance).
- Niculescu-Mizil & Caruana (2005), "Predicting good probabilities with supervised
  learning", *ICML*. The reference for why calibration deserves its own chart.
- Yellowbrick documentation, scikit-yb.org.
