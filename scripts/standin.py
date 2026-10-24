"""Stand-ins for the primary line camera. Each writes one score per test image.

usage: python scripts/standin.py visa candle [--root data] [--out results] [--only aligned]

We have no line camera, so these play it. lane.py later turns their scores into the reject bin.
Four of them, from dumb to learned, so we can check that recovery does not depend on who built the bin:

  template   golden image (mean of goods) minus test image, 99th percentile pixel. Position-locked.
  aligned    same, after sliding the test image onto the golden image (phase correlation).
  pca        reconstruction error from the 20 main patterns of good images. Tolerates small shifts.
  tinypadim  PaDiM on resnet18 at 128 px, fitted on the goods, run on CPU. A learned primary
             with different features than our wide_resnet50 PatchCore, so their mistakes differ.

Writes results/standin/<dataset>_<category>_<name>.csv with image,label,score.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize

from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.post_processing import PostProcessor

DATASETS = {"visa": Visa, "mvtec": MVTecAD, "goodsad": MVTecAD}  # GoodsAD ships in MVTec AD layout
NAMES = ("template", "aligned", "pca", "tinypadim")


def to_gray_small(img: torch.Tensor, size: int) -> torch.Tensor:
    """Bx3xHxW in [0,1] -> Bx1xsxs grayscale."""
    g = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    return F.interpolate(g[:, None], size=(size, size), mode="bilinear", align_corners=False)


def align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Shift each image in x (Bx1xsxs) so it best overlaps ref (1xsxs). Phase correlation, translation only."""
    fx, fr = torch.fft.fft2(x[:, 0]), torch.fft.fft2(ref)
    cross = fx * fr.conj()
    corr = torch.fft.ifft2(cross / (cross.abs() + 1e-8)).real
    s = x.shape[-1]
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        k = int(corr[i].argmax())
        dy, dx = k // s, k % s
        dy, dx = (dy - s if dy > s // 2 else dy), (dx - s if dx > s // 2 else dx)
        out[i, 0] = torch.roll(x[i, 0], shifts=(-dy, -dx), dims=(0, 1))
    return out


def loader(ds, dm) -> DataLoader:
    return DataLoader(ds, batch_size=16, num_workers=4, collate_fn=ds.collate_fn)


def rows_of(batch, root: Path, score: torch.Tensor) -> list[tuple[str, int, float]]:
    return list(zip([str(Path(p).relative_to(root)) for p in batch.image_path],
                    batch.gt_label.int().tolist(), score.tolist()))


def template_standin(dm, root: Path, size: int, aligned: bool) -> list[tuple[str, int, float]]:
    """Golden-image difference. The classic rule camera, with or without lining the item up first."""
    acc, n = None, 0
    for b in loader(dm.train_data, dm):
        g = to_gray_small(b.image, size)
        acc = g.sum(0) if acc is None else acc + g.sum(0)
        n += g.shape[0]
    template = acc / n
    rows = []
    for b in loader(dm.test_data, dm):
        g = to_gray_small(b.image, size)
        if aligned:
            g = align(g, template)
        diff = (g - template).abs().flatten(1)
        rows += rows_of(b, root, torch.quantile(diff, 0.99, dim=1))
    return rows


def pca_standin(dm, root: Path, size: int, k: int = 20) -> list[tuple[str, int, float]]:
    """Reconstruction error after projecting onto the k main patterns of good images."""
    xs = torch.cat([to_gray_small(b.image, size).flatten(1) for b in loader(dm.train_data, dm)])
    mean = xs.mean(0)
    _, _, vt = torch.linalg.svd(xs - mean, full_matrices=False)
    basis = vt[:k]
    rows = []
    for b in loader(dm.test_data, dm):
        x = to_gray_small(b.image, size).flatten(1) - mean
        err = (x - (x @ basis.T) @ basis).abs()
        rows += rows_of(b, root, torch.quantile(err, 0.99, dim=1))
    return rows


def tinypadim_standin(dm, root: Path, size: int, out: str) -> list[tuple[str, int, float]]:
    """Small PaDiM on CPU. Raw scores, no anomalib threshold, same reasons as run.py."""
    model = Padim(
        backbone="resnet18", layers=["layer1", "layer2"], n_features=50,
        pre_processor=Padim.configure_pre_processor(image_size=(size, size)),
        post_processor=PostProcessor(enable_normalization=False, enable_thresholding=False),
        evaluator=False, visualizer=False,
    )
    engine = Engine(accelerator="cpu", devices=1, logger=False, default_root_dir=out)
    engine.fit(model=model, datamodule=dm)
    preds = engine.predict(model=model, datamodule=dm)
    return [r for b in preds for r in rows_of(b, root, b.pred_score.flatten())]


def write_csv(path: Path, rows: list[tuple[str, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("image,label,score\n")
        for img, lab, sc in rows:
            f.write(f"{img},{lab},{sc:.6f}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("category")
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--only", choices=NAMES, nargs="*", default=NAMES)
    a = p.parse_args()

    root = Path(a.root) / a.dataset
    # resize at load time, GoodsAD images are 3000 px
    dm = DATASETS[a.dataset](root=root, category=a.category, train_batch_size=16, eval_batch_size=16, num_workers=4,
                             augmentations=Resize((256, 256)))
    dm.setup()
    dm._is_setup = True  # keep this setup through the PaDiM trainer  # noqa: SLF001
    out = Path(a.out) / "standin"
    tag = f"{a.dataset}_{a.category}"
    makers = {
        "template": lambda: template_standin(dm, root, a.size, aligned=False),
        "aligned": lambda: template_standin(dm, root, a.size, aligned=True),
        "pca": lambda: pca_standin(dm, root, a.size // 2),
        "tinypadim": lambda: tinypadim_standin(dm, root, a.size, a.out),
    }
    for name in a.only:
        rows = makers[name]()
        write_csv(out / f"{tag}_{name}.csv", rows)
        print(f"{name}: {len(rows)} test images")


if __name__ == "__main__":
    main()
