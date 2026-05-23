# Week 2 — Manifold learning and high-dimensional embeddings

> Nonlinear dimensionality reduction theory; 2D / 3D projection of text and image
> embedding spaces, and hyperparameter optimization.

## Learning objectives

By the end of this week you should be able to:

1. Explain the **manifold hypothesis** and why linear methods (PCA) fall short for the
   kinds of representations modern neural networks produce.
2. Apply **t-SNE** and **UMAP** correctly, including a defensible choice of the most
   important hyperparameters (`perplexity`, `n_neighbors`, `min_dist`, distance metric).
3. Read a 2D embedding plot critically — distinguishing structure that is real from
   structure that is an artefact of the projection.
4. Build **interactive 3D scatter plots** with Plotly for embeddings that genuinely
   benefit from the extra dimension.
5. Run a small **hyperparameter sweep** for a projection method and produce a
   comparison figure that lets a reader pick the right setting for their needs.

## Contents

| File | Purpose |
|------|---------|
| `notebooks/01-theory.ipynb` | Manifold hypothesis, PCA → t-SNE → UMAP, what projections do not preserve |
| `notebooks/02-lab.ipynb` | Project a sentence-embedding space and an image-embedding space |
| `exercises/` | Three exercises on hyperparameter sensitivity |
| `solutions/` | Reference solutions |

## Dataset

The lab uses pre-computed embeddings of the 20-newsgroups dataset (text) and a subset
of CIFAR-10 (images). Both are downloaded by `data/download.py` and cached locally;
total size is roughly 30 MB.

## Suggested reading

- Wattenberg, Viégas & Johnson (2016), "How to Use t-SNE Effectively", *Distill*. The
  canonical reference on t-SNE failure modes.
- McInnes, Healy & Melville (2018), "UMAP: Uniform Manifold Approximation and Projection",
  arXiv:1802.03426.
- Coenen & Pearce, "Understanding UMAP", *Google PAIR*. The interactive companion to the
  paper above.
