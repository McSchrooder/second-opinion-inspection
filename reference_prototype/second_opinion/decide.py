"""Three-way decision and the metrics that matter for this product.

On a reject pile, the tool answers per item:
  "recover"  - score below t_low: likely good, propose to recover
  "defect"   - score above t_high: confirm the reject
  "human"    - in between: a quality technician decides

Thresholds are chosen on a VALIDATION part of the pile and then frozen:
  t_low  = just below the lowest-scoring real defect  -> zero escapes on validation
  t_high = just above the highest-scoring good item   -> no good item auto-confirmed as defect
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def choose_thresholds(scores: np.ndarray, is_defect: np.ndarray, margin: float = 0.0) -> tuple[float, float]:
    """margin is a fraction of the score range: a bigger margin widens the human band
    (safer, fewer escapes on unseen items, but more manual review)."""
    d, g = scores[is_defect], scores[~is_defect]
    m = margin * float(scores.max() - scores.min()) if len(scores) else 0.0
    t_low = float(d.min()) - m if len(d) else float(np.inf)
    t_high = float(g.max()) + m if len(g) else float(-np.inf)
    if t_high < t_low:  # perfectly separated: one cut point, no human band
        t_high = t_low
    return t_low, t_high


def decide(scores: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    out = np.full(len(scores), "human", dtype=object)
    out[scores < t_low] = "recover"
    out[scores > t_high] = "defect"
    return out


def pile_metrics(decisions: np.ndarray, is_defect: np.ndarray) -> dict:
    fr = ~is_defect
    n_fr, n_def = int(fr.sum()), int(is_defect.sum())
    recovered = (decisions == "recover") & fr
    escapes = (decisions == "recover") & is_defect
    wrongly_confirmed = (decisions == "defect") & fr
    return {
        "items": int(len(decisions)),
        "false_rejects": n_fr,
        "true_defects": n_def,
        "false_rejects_recovered": int(recovered.sum()),
        "recovery_rate": float(recovered.sum() / n_fr) if n_fr else 0.0,
        "defect_escapes": int(escapes.sum()),
        "escape_rate": float(escapes.sum() / n_def) if n_def else 0.0,
        "sent_to_human": int((decisions == "human").sum()),
        "human_share": float((decisions == "human").mean()) if len(decisions) else 0.0,
        "good_items_auto_confirmed_as_defect": int(wrongly_confirmed.sum()),
    }


def recovery_at_zero_escape(scores: np.ndarray, is_defect: np.ndarray) -> float:
    """Best possible recovery on this set with zero escapes (an upper bound, for the curve)."""
    if not is_defect.any():
        return 1.0
    t = scores[is_defect].min()
    fr = ~is_defect
    return float((scores[fr] < t).mean()) if fr.any() else 0.0


def heatmap_quality(maps: np.ndarray, masks: np.ndarray, is_defect: np.ndarray) -> dict:
    """Pixel AUROC and best-threshold IoU of heatmaps against ground-truth masks (defects only)."""
    sel = is_defect & masks.reshape(len(masks), -1).any(axis=1)
    if not sel.any():
        return {"pixel_auroc": None, "mean_iou": None, "n_images": 0}
    m, y = maps[sel], masks[sel]
    auroc = float(roc_auc_score(y.ravel(), m.ravel()))
    ious = []
    for hm, gt in zip(m, y):
        thr = np.percentile(hm, 100 * (1 - gt.mean()))  # same area as the true defect
        pred = hm >= thr
        ious.append((pred & gt).sum() / max(1, (pred | gt).sum()))
    return {"pixel_auroc": auroc, "mean_iou": float(np.mean(ious)), "n_images": int(sel.sum())}
