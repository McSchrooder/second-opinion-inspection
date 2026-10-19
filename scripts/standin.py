"""Stand-ins for the primary inspection camera.

We do not have the line's camera, so we simulate it with deliberately shallow
detectors, tuned (in lane.py) to catch every defect the way real primaries are.
Whatever they flag is the simulated reject lane.

usage: python scripts/standin.py visa candle [--root DATA_DIR] [--out results]

Writes results/standin/{dataset}_{category}_{standin}.csv with image,label,score
for the test split. Two stand-ins:

  template  : absolute difference to the mean good image (grayscale, low-res),
              scored by the 99th percentile pixel. Classic golden-template check.
  tinypadim : PaDiM on resnet18 layer1+layer2 at 128px. A weak learned baseline.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.post_processing import PostProcessor

DATASETS = {"visa": Visa, "mvtec": MVTecAD}


def to_gray_small(img: torch.Tensor, size: int) -> torch.Tensor:
    """img: Bx3xHxW in [0,1] -> Bx1xsxs grayscale."""
    g = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    return F.interpolate(g[:, None], size=(size, size), mode="bilinear", align_corners=False)


def write_csv(path: Path, rows: list[tuple[str, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("image,label,score\n")
        for img, lab, sc in rows:
            f.write(f"{img},{lab},{sc:.6f}\n")


def rel(paths: list[str], root: Path) -> list[str]:
    return [str(Path(p).relative_to(root)) for p in paths]


def template_standin(dm, root: Path, size: int) -> list[tuple[str, int, float]]:
    dm.setup()
    train = DataLoader(dm.train_data, batch_size=16, num_workers=4, collate_fn=dm.train_data.collate_fn)
    acc, n = None, 0
    for b in train:
        g = to_gray_small(b.image, size)
        acc = g.sum(0) if acc is None else acc + g.sum(0)
        n += g.shape[0]
    template = acc / n
    rows = []
    test = DataLoader(dm.test_data, batch_size=16, num_workers=4, collate_fn=dm.test_data.collate_fn)
    for b in test:
        g = to_gray_small(b.image, size)
        diff = (g - template).abs().flatten(1)
        score = torch.quantile(diff, 0.99, dim=1)
        rows += list(zip(rel(b.image_path, root), b.gt_label.int().tolist(), score.tolist()))
    return rows


def pca_standin(dm, root: Path, size: int, k: int = 20) -> list[tuple[str, int, float]]:
    """PCA reconstruction error on low-res grayscale. CPU only, classic eigen-image check."""
    dm.setup()
    train = DataLoader(dm.train_data, batch_size=16, num_workers=4, collate_fn=dm.train_data.collate_fn)
    xs = torch.cat([to_gray_small(b.image, size).flatten(1) for b in train])
    mean = xs.mean(0)
    _, _, vt = torch.linalg.svd(xs - mean, full_matrices=False)
    basis = vt[:k]
    rows = []
    test = DataLoader(dm.test_data, batch_size=16, num_workers=4, collate_fn=dm.test_data.collate_fn)
    for b in test:
        x = to_gray_small(b.image, size).flatten(1) - mean
        recon = (x @ basis.T) @ basis
        err = (x - recon).abs()
        score = torch.quantile(err, 0.99, dim=1)
        rows += list(zip(rel(b.image_path, root), b.gt_label.int().tolist(), score.tolist()))
    return rows


def tinypadim_standin(dm, root: Path, size: int, out: str) -> list[tuple[str, int, float]]:
    model = Padim(
        backbone="resnet18", layers=["layer1", "layer2"], n_features=50,
        pre_processor=Padim.configure_pre_processor(image_size=(size, size)),
        post_processor=PostProcessor(enable_normalization=False, enable_thresholding=False),
        evaluator=False, visualizer=False,
    )
    engine = Engine(accelerator="auto", devices=1, logger=False, default_root_dir=out)
    engine.fit(model=model, datamodule=dm)
    preds = engine.predict(model=model, datamodule=dm)
    rows = []
    for b in preds:
        rows += list(zip(rel(b.image_path, root), b.gt_label.int().tolist(), b.pred_score.flatten().tolist()))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("category")
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--only", choices=["template", "pca", "tinypadim"], default=None,
                   help="default runs template and pca; tinypadim is opt-in (hangs on ROCm/WSL)")
    a = p.parse_args()

    root = Path(a.root) / a.dataset
    dm = DATASETS[a.dataset](root=root, category=a.category, train_batch_size=16, eval_batch_size=16, num_workers=4)
    tag = f"{a.dataset}_{a.category}"
    out = Path(a.out) / "standin"

    if a.only in (None, "template"):
        rows = template_standin(dm, root, a.size)
        write_csv(out / f"{tag}_template.csv", rows)
        print(f"template: {len(rows)} test images -> {out / (tag + '_template.csv')}")
    if a.only in (None, "pca"):
        rows = pca_standin(dm, root, a.size // 2)
        write_csv(out / f"{tag}_pca.csv", rows)
        print(f"pca: {len(rows)} test images -> {out / (tag + '_pca.csv')}")
    if a.only == "tinypadim":
        rows = tinypadim_standin(dm, root, a.size, a.out)
        write_csv(out / f"{tag}_tinypadim.csv", rows)
        print(f"tinypadim: {len(rows)} test images -> {out / (tag + '_tinypadim.csv')}")


if __name__ == "__main__":
    main()
