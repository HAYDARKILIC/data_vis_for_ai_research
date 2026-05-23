"""Download / cache the small datasets used in week 2.

We use:
  - A 1500-document subset of 20-newsgroups (text classification, 6 categories).
  - 2000 images from CIFAR-10 (vision, 10 categories).

Embeddings are pre-computed and cached as .npz to avoid forcing a GPU on every learner.
Run once:
    python data/download.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent


def make_text_embeddings():
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer

    cats = ["sci.space", "sci.med", "rec.sport.hockey",
            "rec.autos", "comp.graphics", "talk.politics.guns"]
    data = fetch_20newsgroups(subset="train", categories=cats,
                              remove=("headers", "footers", "quotes"))
    # Truncate to keep things light
    docs = [d[:1500] for d in data.data[:1500]]
    labels = np.asarray(data.target[:1500])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeds = model.encode(docs, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)

    out = HERE / "text_embeddings.npz"
    np.savez_compressed(out, embeddings=embeds, labels=labels,
                        label_names=np.asarray(cats))
    print(f"Wrote {out}  shape={embeds.shape}")


def make_image_embeddings():
    import torch
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights).to(device)
    # Strip the classification head -> use the 512-d pooled feature
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.Resize(224), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = torchvision.datasets.CIFAR10(root=str(HERE / "cifar10"),
                                      train=True, download=True, transform=transform)
    idx = np.random.default_rng(0).choice(len(ds), size=2000, replace=False)
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=64, num_workers=0)

    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(model(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    out = HERE / "image_embeddings.npz"
    np.savez_compressed(out, embeddings=feats, labels=labels,
                        label_names=np.asarray(ds.classes))
    print(f"Wrote {out}  shape={feats.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", action="store_true")
    p.add_argument("--images", action="store_true")
    args = p.parse_args()
    if not (args.text or args.images):
        args.text = args.images = True
    if args.text:
        make_text_embeddings()
    if args.images:
        make_image_embeddings()


if __name__ == "__main__":
    main()
