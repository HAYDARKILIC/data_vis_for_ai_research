# Week 6 — Computer vision explainability and layer-wise visualization

> Layer-wise visualization in CNNs and Vision Transformers; concept extraction and
> faithfulness metrics.

## Learning objectives

By the end of this week you should be able to:

1. Apply the **Grad-CAM family** (Grad-CAM, Grad-CAM++, Score-CAM, EigenCAM) to a
   pretrained CNN and read the output as a coarse spatial attribution.
2. Apply **attention rollout** to a Vision Transformer (ViT) and explain its differences
   from Grad-CAM.
3. Evaluate an explanation **quantitatively** with a faithfulness metric — typically a
   deletion or insertion AUC — rather than only by visual inspection.
4. Distinguish "the explanation **looks plausible**" from "the explanation **reflects
   the decision**", and design experiments that test the latter.
5. Choose between input-space attribution (week 5) and layer-wise attribution (this week)
   for a given research question.

## Contents

| File | Purpose |
|------|---------|
| `notebooks/01-theory.ipynb` | Grad-CAM derivation, ViT attention rollout, faithfulness metrics |
| `notebooks/02-lab.ipynb` | Grad-CAM and attention rollout on real images, with quantitative checks |
| `exercises/` | Two exercises on layer choice and faithfulness |
| `solutions/` | Reference solutions |

## Datasets

A handful of ImageNet-class images is bundled (`data/`) so the lab works without a
download. The CNN backbone is a pretrained `resnet50` from torchvision; the ViT is a
pretrained `vit_base_patch16_224` from `timm`.

## Suggested reading

- Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization", *ICCV*.
- Chattopadhay et al. (2018), "Grad-CAM++: Improved Visual Explanations for Deep
  Convolutional Networks", *WACV*.
- Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers", *ACL* — the
  "attention rollout" paper.
- Petsiuk et al. (2018), "RISE: Randomized Input Sampling for Explanation" — the source
  of the deletion / insertion faithfulness metrics.
- Hooker et al. (2019), "A Benchmark for Interpretability Methods in Deep Neural
  Networks" — sceptical of pixel-attribution evaluations done badly.
