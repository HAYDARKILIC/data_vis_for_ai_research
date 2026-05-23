# Week 5 — Local interpretability and feature attribution

> Shapley values and local perturbation theory; attribution analysis on tabular data,
> text tokens, and multimodal inputs.

## Learning objectives

By the end of this week you should be able to:

1. State the **Shapley axioms** (efficiency, symmetry, dummy, additivity) and explain
   why they uniquely determine the Shapley value as an attribution.
2. Apply **SHAP** to a tree model with `TreeExplainer` and a deep model with
   `DeepExplainer` or `KernelExplainer`, and read both **local** (per-instance) and
   **global** (summary) views.
3. Apply **LIME** to a text classifier and explain its outputs at the token level.
4. Apply **Captum** attribution methods (Integrated Gradients, DeepLIFT, Saliency,
   Layer Attribution) to a PyTorch model.
5. Recognise the **failure modes** of attribution methods — saturation, baseline
   sensitivity, sanity check failures — and the diagnostics for each.

## Contents

| File | Purpose |
|------|---------|
| `notebooks/01-theory.ipynb` | Shapley math, IG axioms, when each method applies |
| `notebooks/02-lab.ipynb` | SHAP on tabular + LIME on text + Captum IG on a small PyTorch net |
| `exercises/` | Two attribution exercises |
| `solutions/` | Reference solutions |

## Datasets

- **Tabular:** scikit-learn's California housing (regression) — for SHAP TreeExplainer.
- **Text:** a 6-class subset of 20-newsgroups — for LIME on a TF-IDF + LogReg classifier.
- **Image:** a tiny PyTorch CNN trained briefly on FashionMNIST in the lab — for Captum
  Integrated Gradients.

All datasets are downloaded automatically by their respective notebooks.

## Suggested reading

- Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions" (SHAP).
- Ribeiro, Singh, Guestrin (2016), "Why Should I Trust You?" (LIME).
- Sundararajan, Taly, Yan (2017), "Axiomatic Attribution for Deep Networks"
  (Integrated Gradients).
- Adebayo et al. (2018), "Sanity Checks for Saliency Maps" — the paper everyone should
  read before publishing an attribution figure.
