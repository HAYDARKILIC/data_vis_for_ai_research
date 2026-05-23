# Data Visualization for AI Research

A 6-week, hands-on course on visualization techniques for modern AI research workflows.
Each week is built as a pair of Jupyter notebooks — a **theory** notebook that builds the
conceptual foundation, and a **lab** notebook that puts the ideas into practice on real
datasets and models.

---

## Course at a glance

| Week | Topic | Core libraries |
|------|-------|----------------|
| [01](week-01/) | Scientific data types, magnitudes, and distributions | Matplotlib, Seaborn, Pandas |
| [02](week-02/) | Manifold learning and high-dimensional embeddings | UMAP, scikit-learn, Plotly |
| [03](week-03/) | Model diagnostics and evaluation visualization | Yellowbrick, scikit-plot, scikit-learn |
| [04](week-04/) | MLOps experiment tracking and live metric monitoring | MLflow, Weights & Biases, PyTorch |
| [05](week-05/) | Local interpretability and feature attribution | SHAP, LIME, Captum |
| [06](week-06/) | Explainability in computer vision and layer-wise visualization | pytorch-grad-cam, Xplique |

Every week ships with:

- `notebooks/01-theory.ipynb` — concepts, math, design principles, worked figures
- `notebooks/02-lab.ipynb` — end-to-end practical walkthrough on a real dataset
- `exercises/` — open-ended problems with starter code
- `solutions/` — reference solutions (try the exercises first!)
- `data/` — small datasets used by the week (large ones are downloaded by a script)
- `figures/` — exported figures used in the README and slides

---

## Weekly breakdown

### Week 1 — Scientific data types, magnitudes, and distributions
Exploratory analysis of distributions, proportions, and variable relationships. Covers
how data type (nominal, ordinal, interval, ratio) constrains visual encoding choices,
log/symlog scaling for skewed magnitudes, faceting for conditional distributions, and the
ethics of chart design — including a critical look at common failure modes in AI model
output charts.

### Week 2 — Manifold learning and high-dimensional embeddings
Theory of nonlinear dimensionality reduction (PCA → t-SNE → UMAP), and how to project
text and image embedding spaces into 2D / 3D for inspection. Strong focus on
hyperparameter sensitivity (`n_neighbors`, `min_dist`, perplexity), distance metrics, and
the things these projections do **not** preserve.

### Week 3 — Model diagnostics and evaluation visualization
Visual diagnosis of classification and regression models: decision boundaries, residual
plots, learning curves, calibration curves, ROC / PR curves, confusion matrices, and
class-prediction-error plots. Built on Yellowbrick and scikit-plot, with custom
Matplotlib variants for the cases the libraries don't cover.

### Week 4 — MLOps experiment tracking and live metric monitoring
Hyperparameter sweep analysis, real-time tracking of training and validation metrics, and
versioning of multi-dimensional model artefacts. A small PyTorch model is trained under
both MLflow and Weights & Biases, with parallel-coordinates plots, parameter-importance
plots, and run comparisons.

### Week 5 — Local interpretability and feature attribution
Shapley values and local perturbation theory; feature attribution analysis on tabular,
text-token, and multimodal data. Covers SHAP (TreeExplainer, DeepExplainer, KernelExplainer),
LIME for text and tabular, and Captum (Integrated Gradients, DeepLIFT, Layer Attribution)
for PyTorch models.

### Week 6 — Computer vision explainability and layer-wise visualization
Layer-wise visualization in CNNs and Vision Transformers: Grad-CAM, Grad-CAM++, Score-CAM,
EigenCAM, attention rollout, and concept-based methods. We also discuss faithfulness
metrics and the difference between an explanation that *looks* plausible and one that
actually reflects the model's decision.

---

## Setup

### Option A — local install (recommended)

```bash
git clone https://github.com/HAYDARKILIC/data_vis_for_ai_research.git
cd data_vis_for_ai_research
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Python 3.10 or newer is required. A CUDA-capable GPU is recommended for weeks 4 and 6 but
not required — the notebooks fall back to CPU and use small enough models that everything
still runs in a few minutes.

### Option B — conda

```bash
conda env create -f environment.yml
conda activate data_vis_for_ai_research
jupyter lab
```

### Option C — Google Colab

Every notebook starts with a Colab badge. Click it to open the notebook in Colab; the
first cell installs any missing dependencies.

---

## How to use this course

There are three reasonable ways to work through the material:

1. **Linear** — one week per week, theory first, then lab, then exercises. This is the
   intended path and the one the difficulty curve is tuned for.
2. **Topic-driven** — jump to the week you need. Weeks are mostly independent, with the
   exception that week 6 assumes you've seen the attribution methods in week 5.
3. **Reference** — keep the repo open while you work on your own project and crib the
   plotting recipes from the labs.

Each lab notebook ends with a short **"what to do differently in your own research"**
section that translates the lab's choices into guidance for your own work.

---
## License

The course materials are released under the MIT License (see [`LICENSE`](LICENSE)).
Third-party datasets retain their original licenses; each week's `data/README.md`
documents the provenance and terms of the data it uses.
