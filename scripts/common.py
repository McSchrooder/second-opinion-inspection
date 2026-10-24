"""Shared helpers: datamodules, result-file naming, the reject lane, recovery at fixed escape."""
import re
from pathlib import Path

import numpy as np
import pandas as pd


def make_datamodule(dataset: str, root: Path, category: str, **kw):
    """VisA and MVTec AD through anomalib's own loaders. GoodsAD through the generic Folder loader:
    its layout is MVTec-like but the mask files do not pair up for anomalib's MVTec reader."""
    from anomalib.data import Folder, MVTecAD, Visa

    if dataset == "visa":
        return Visa(root=root, category=category, **kw)
    if dataset == "mvtec":
        return MVTecAD(root=root, category=category, **kw)
    if dataset == "goodsad":
        cat = Path(root) / category
        bad = sorted(str(d.relative_to(cat)) for d in (cat / "test").iterdir() if d.is_dir() and d.name != "good")
        return Folder(name=category, root=cat, normal_dir="train/good", normal_test_dir="test/good",
                      abnormal_dir=bad, val_split_mode="same_as_test", seed=0, **kw)
    raise ValueError(dataset)


DATASETS = ("visa", "mvtec", "goodsad")


def main_rows(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Rows of `model`, one per category. Where the full run does not exist (bank too big for VRAM) the
    -n<subset> run of the same model stands in, e.g. patchcore-n400 on GoodsAD, rad-n300 on VisA."""
    exact = df[df.model == model]
    sub = df[df.model.str.fullmatch(re.escape(model) + r"-n\d+")]
    return pd.concat([exact, sub[~sub.category.isin(exact.category)]])


def family(model: str) -> str:
    """Model name with the candidate subset tag folded in: rad-n300 -> rad. Keeps -n150 (a deliberate variant)."""
    return re.sub(r"-n(300|400|450)$", "", model)


def split_tag(tag: str) -> tuple[str, str, str]:
    """'visa_pipe_fryum_patchcore-320' -> ('visa', 'pipe_fryum', 'patchcore-320'). Model tokens never contain '_'."""
    ds, rest = tag.split("_", 1)
    cat, model = rest.rsplit("_", 1)
    return ds, cat, model


def build_lane(standin: pd.DataFrame) -> tuple[pd.Index, float]:
    """Reject lane = everything the stand-in camera scores at or above its lowest defect, so it misses nothing.

    Returns the flagged image index and the stand-in's false reject rate (share of goods flagged).
    """
    thr = standin.loc[standin.label == 1, "score"].min()
    flagged = standin[standin.score >= thr]
    frr = float((flagged.label == 0).sum() / (standin.label == 0).sum())
    return flagged.index, frr


def t_recover(defect_scores: np.ndarray, escape: float) -> float:
    """Threshold that lets through at most `escape` of the given defects (k-th lowest defect score)."""
    d = np.sort(defect_scores)
    k = int(np.floor(escape * len(d)))
    return float(d[min(k, len(d) - 1)])


def recovery_at(scores: np.ndarray, labels: np.ndarray, escape: float) -> float:
    """Share of goods below t_recover. Same quantity as anomalib's PGn, threshold set on these defects, so optimistic."""
    thr = t_recover(scores[labels == 1], escape)
    return float((scores[labels == 0] < thr).mean()) if (labels == 0).any() else float("nan")
