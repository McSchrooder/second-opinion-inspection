"""Simulate what a conservative line camera does: reject real defects AND some good items.

Public datasets contain good and defective items, but not the mistakes of a real
inspection camera. We create them:

1. Nuisance: copies of good test images get harmless changes that a camera often
   misreads (light change, water droplets, dust specks, small misalignment, blur).
2. Primary inspector: a simple rule-based check (difference to the mean good image)
   tuned so it catches every real defect in a calibration set. Being tuned to never
   miss, it also rejects many harmless items. Those are our false rejects.

The output is a "reject pile": everything the primary inspector rejected, with the
truth (real defect or false reject) kept for evaluation only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

NUISANCE_TYPES = ("light", "droplets", "dust", "shift", "blur")


@dataclass
class Item:
    image: np.ndarray      # uint8 HxWx3
    mask: np.ndarray       # bool HxW (all False for good items)
    is_defect: bool
    source: str            # original file name
    variant: str           # "original" or a nuisance type
    defect_type: str


def add_nuisance(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32)
    h, w = x.shape[:2]
    if kind == "light":
        x = x * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    elif kind == "droplets":
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(rng.integers(3, 8)):
            cy, cx = rng.integers(0, h), rng.integers(0, w)
            r = rng.uniform(0.01, 0.03) * h
            blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2)))
            x = x + 70 * blob[..., None]
    elif kind == "dust":
        n = rng.integers(10, 30)
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        x[ys, xs] = rng.uniform(0, 60)
        x = gaussian_filter(x, sigma=(0.6, 0.6, 0))
    elif kind == "shift":
        dy, dx = rng.uniform(-0.04, 0.04, 2) * h
        x = nd_shift(x, (dy, dx, 0), mode="nearest")
    elif kind == "blur":
        x = gaussian_filter(x, sigma=(rng.uniform(1.0, 2.0),) * 2 + (0,))
    return np.clip(x, 0, 255).astype(np.uint8)


class RuleInspector:
    """Mean-template difference inspector, the kind of simple rule many lines still run."""

    def __init__(self, train_images: np.ndarray):
        self.template = train_images.astype(np.float32).mean(axis=0)
        self.threshold: float | None = None

    def score(self, img: np.ndarray) -> float:
        diff = np.abs(img.astype(np.float32) - self.template).mean(axis=2)
        diff = gaussian_filter(diff, 2)
        return float(np.percentile(diff, 99.5))

    def calibrate_never_miss(self, defect_images: list[np.ndarray], margin: float = 0.98) -> float:
        """Set the threshold just below the lowest-scoring real defect, so none is missed."""
        lowest = min(self.score(i) for i in defect_images)
        self.threshold = lowest * margin
        return self.threshold

    def rejects(self, img: np.ndarray) -> bool:
        assert self.threshold is not None, "calibrate first"
        return self.score(img) >= self.threshold


def build_reject_pile(test_items: list[Item], inspector: RuleInspector,
                      nuisance_per_good: int, rng: np.random.Generator) -> tuple[list[Item], dict]:
    """Apply nuisance to good items, run the primary inspector, keep what it rejects."""
    candidates: list[Item] = []
    for it in test_items:
        candidates.append(it)
        if not it.is_defect:
            for k in rng.choice(len(NUISANCE_TYPES), size=nuisance_per_good, replace=False):
                kind = NUISANCE_TYPES[k]
                candidates.append(Item(add_nuisance(it.image, kind, rng), it.mask, False,
                                       it.source, kind, "good"))
    pile = [c for c in candidates if inspector.rejects(c.image)]
    stats = {
        "items_inspected": len(candidates),
        "defects_inspected": sum(c.is_defect for c in candidates),
        "rejected": len(pile),
        "rejected_true_defects": sum(c.is_defect for c in pile),
        "false_rejects": sum(not c.is_defect for c in pile),
    }
    stats["primary_false_reject_rate"] = stats["false_rejects"] / max(1, stats["items_inspected"] - stats["defects_inspected"])
    stats["primary_defect_recall"] = stats["rejected_true_defects"] / max(1, stats["defects_inspected"])
    return pile, stats
