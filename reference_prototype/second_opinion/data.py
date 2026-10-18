"""Load images in the MVTec AD folder layout.

Expected layout (MVTec AD and the VisA "1cls" export both follow it):

    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/<name>_mask.png
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".JPG"}


@dataclass
class Sample:
    path: Path
    is_defect: bool
    defect_type: str
    mask_path: Path | None


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix in IMG_EXT)


def load_split(root: str | Path, category: str) -> tuple[list[Sample], list[Sample]]:
    """Return (train_good, test_all) samples for one category."""
    base = Path(root) / category
    train = [Sample(p, False, "good", None) for p in _images(base / "train" / "good")]
    test: list[Sample] = []
    for sub in sorted((base / "test").iterdir()):
        if not sub.is_dir():
            continue
        for p in _images(sub):
            if sub.name == "good":
                test.append(Sample(p, False, "good", None))
            else:
                mask = base / "ground_truth" / sub.name / f"{p.stem}_mask.png"
                test.append(Sample(p, True, sub.name, mask if mask.exists() else None))
    if not train:
        raise FileNotFoundError(f"No training images in {base / 'train' / 'good'}")
    return train, test


def read_image(path: Path, size: int) -> np.ndarray:
    """RGB uint8 array of shape (size, size, 3)."""
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))


def read_mask(path: Path | None, size: int) -> np.ndarray:
    """Boolean mask of shape (size, size); all False when there is no mask."""
    if path is None:
        return np.zeros((size, size), dtype=bool)
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(m) > 127
