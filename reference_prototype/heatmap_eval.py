"""How good are the explanations? Heatmaps against the true defect masks, per defect type.

Three measures, from strict to practical:
  pixel AUROC   - does the heatmap rank defect pixels above normal pixels (threshold-free)
  hit rate      - is the hottest point of the heatmap inside the defect (dilated by a few pixels)?
                  This is what a technician looks at first.
  IoU at the red threshold - overlap between the red area the app shows (heat above the lower
                  decision threshold) and the true defect. No oracle: the threshold is the one the
                  tool uses, not one tuned to the defect size.
Also writes a figure with examples for the report and the slides.

Example:
    python heatmap_eval.py --data-root data/mvtec --category bottle --weights w.pth --out results/heatmaps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.metrics import roc_auc_score

from run_pipeline import overlay, stratified_split
from second_opinion.data import load_split, read_image, read_mask
from second_opinion.decide import choose_thresholds
from second_opinion.model import PatchCore
from second_opinion.simulate import Item, RuleInspector, build_reject_pile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--coreset-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dilate", type=int, default=8, help="pixels of tolerance for the hit rate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, out = a.image_size, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train, test = load_split(a.data_root, a.category)
    train_imgs = np.stack([read_image(s.path, S) for s in train])
    base = [Item(read_image(s.path, S), read_mask(s.mask_path, S), s.is_defect, s.path.name, "original",
                 s.defect_type) for s in test]
    insp = RuleInspector(train_imgs)
    insp.calibrate_never_miss([i.image for i in base if i.is_defect])
    rng = np.random.default_rng(a.seed)
    pile, _ = build_reject_pile(base, insp, 2, rng)
    is_def = np.array([p.is_defect for p in pile])
    groups = np.array([f"{p.defect_type}/{p.source}" for p in pile])
    val = stratified_split(is_def, 0.5, rng, groups)

    model = PatchCore("wide_resnet50_2", a.weights, S, a.coreset_ratio, seed=a.seed)
    model.fit(train_imgs)
    scores, maps = model.predict(np.stack([p.image for p in pile]), S)
    t_low, t_high = choose_thresholds(scores[val], is_def[val], 0.0)

    rows = []
    for i, p in enumerate(pile):
        if not p.is_defect or not p.mask.any():
            continue
        hm, gt = maps[i], p.mask.astype(bool)
        peak = np.unravel_index(np.argmax(hm), hm.shape)
        hit = bool(binary_dilation(gt, iterations=a.dilate)[peak])
        red = hm >= t_low
        iou = float((red & gt).sum() / max(1, (red | gt).sum()))
        rows.append({"i": i, "type": p.defect_type, "source": p.source, "score": float(scores[i]),
                     "hit": hit, "iou_red": iou, "escaped_if_test": bool(scores[i] < t_low)})

    sel = [r["i"] for r in rows]
    res = {"category": a.category, "seed": a.seed, "t_low": t_low, "t_high": t_high,
           "n_defects": len(rows),
           "pixel_auroc": float(roc_auc_score(np.stack([pile[i].mask for i in sel]).ravel(),
                                              maps[sel].ravel())),
           "hit_rate": float(np.mean([r["hit"] for r in rows])),
           "iou_red_mean": float(np.mean([r["iou_red"] for r in rows])),
           "per_type": {}}
    for t in sorted({r["type"] for r in rows}):
        rr = [r for r in rows if r["type"] == t]
        ii = [r["i"] for r in rr]
        res["per_type"][t] = {"n": len(rr),
                              "pixel_auroc": float(roc_auc_score(np.stack([pile[i].mask for i in ii]).ravel(),
                                                                 maps[ii].ravel())),
                              "hit_rate": float(np.mean([r["hit"] for r in rr])),
                              "iou_red_mean": float(np.mean([r["iou_red"] for r in rr]))}
    low = sorted(rows, key=lambda r: r["score"])[:3]
    res["lowest_scoring_defects"] = [{k: r[k] for k in ("type", "source", "score", "hit", "iou_red")} for r in low]
    json.dump(res, open(out / "heatmap_eval.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "lowest_scoring_defects"}, indent=2))
    print("lowest-scoring defects:", res["lowest_scoring_defects"])

    # figure: one clear defect per type, the lowest-scoring defect, two false rejects
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = []
    for t in sorted({r["type"] for r in rows}):
        best = max((r for r in rows if r["type"] == t), key=lambda r: r["iou_red"])
        pick.append((best["i"], f"{t.replace('_', ' ')}\nscore {best['score']:.2f}"))
    pick.append((low[0]["i"], f"faintest defect ({low[0]['type']})\nscore {low[0]['score']:.2f}"))
    fr = [i for i, p in enumerate(pile) if not p.is_defect]
    fr = sorted(fr, key=lambda i: scores[i])
    for i in (fr[len(fr) // 2], fr[-1]):
        pick.append((i, f"false reject ({pile[i].variant})\nscore {scores[i]:.2f}"))
    fig, ax = plt.subplots(2, len(pick), figsize=(2.2 * len(pick), 4.6))
    for c, (i, title) in enumerate(pick):
        ax[0, c].imshow(pile[i].image); ax[0, c].set_title(title, fontsize=8)
        ax[1, c].imshow(overlay(pile[i].image, maps[i], t_low, t_high))
        if pile[i].mask.any():
            ax[1, c].contour(pile[i].mask, levels=[0.5], colors="cyan", linewidths=0.8)
        for r in (0, 1):
            ax[r, c].axis("off")
    ax[0, 0].text(-0.08, 0.5, "camera image", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    ax[1, 0].text(-0.08, 0.5, "tool heatmap\n(cyan = true defect)", transform=ax[1, 0].transAxes, rotation=90,
                  va="center", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "heatmap_examples.png", dpi=160)


if __name__ == "__main__":
    main()
