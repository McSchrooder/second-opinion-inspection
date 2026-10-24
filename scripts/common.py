"""Shared helpers: result-file naming, the reject lane, recovery at fixed escape."""
import numpy as np
import pandas as pd


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
