"""End-to-end run: simulate a camera's reject pile, score it, decide, evaluate, export for the app.

Example:
    python run_pipeline.py --data-root data/mvtec --category bottle \
        --weights weights/wide_resnet50_2.pth --out results/bottle
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from second_opinion.data import load_split, read_image, read_mask
from second_opinion.decide import (choose_thresholds, decide, heatmap_quality, pile_metrics,
                                   recovery_at_zero_escape)
from second_opinion.model import PatchCore
from second_opinion.simulate import Item, RuleInspector, build_reject_pile


def overlay(img: np.ndarray, heat: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Red heat overlay; intensity scaled between the decision thresholds so colours mean the same on every image."""
    span = max(hi - lo, 1e-6)
    a = np.clip((heat - lo) / span, 0, 1)[..., None]
    red = np.zeros_like(img, dtype=np.float32); red[..., 0] = 255
    return (img * (1 - 0.6 * a) + red * 0.6 * a).astype(np.uint8)


def stratified_split(is_defect: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean array: True = validation. Same defect share in both halves."""
    val = np.zeros(len(is_defect), dtype=bool)
    for cls in (True, False):
        idx = np.flatnonzero(is_defect == cls)
        rng.shuffle(idx)
        val[idx[: int(round(len(idx) * frac))]] = True
    return val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--weights", default=None, help="local ImageNet weights (.pth) if there is no internet")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    ap.add_argument("--nuisance-per-good", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--safety-margin", type=float, default=0.05,
                    help="widens the human band by this fraction of the score range (fewer escapes, more review)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); (out / "items").mkdir(parents=True, exist_ok=True)
    S = a.image_size

    train, test = load_split(a.data_root, a.category)
    train_imgs = np.stack([read_image(s.path, S) for s in train])
    test_items = [Item(read_image(s.path, S), read_mask(s.mask_path, S), s.is_defect, s.path.name,
                       "original", s.defect_type) for s in test]

    # 1. simulated primary camera, tuned to never miss a defect
    inspector = RuleInspector(train_imgs)
    inspector.calibrate_never_miss([i.image for i in test_items if i.is_defect])
    pile, primary = build_reject_pile(test_items, inspector, a.nuisance_per_good, rng)
    is_def = np.array([p.is_defect for p in pile])
    val = stratified_split(is_def, a.val_frac, rng)

    # 2. second opinion model, trained on good images only
    model = PatchCore(a.backbone, a.weights, S, a.coreset_ratio, seed=a.seed)
    model.fit(train_imgs)
    pile_imgs = np.stack([p.image for p in pile])
    scores, maps = model.predict(pile_imgs, S)
    rule_scores = np.array([inspector.score(p.image) for p in pile])
    masks = np.stack([p.mask for p in pile])

    # 3. thresholds on validation, frozen, applied to test
    res = {"config": vars(a), "pretrained_backbone": model.pretrained, "primary_camera": primary}
    for name, sc in (("patchcore", scores), ("baseline_rule_score", rule_scores)):
        t_low, t_high = choose_thresholds(sc[val], is_def[val], a.safety_margin)
        dec = decide(sc, t_low, t_high)
        res[name] = {
            "t_low": t_low, "t_high": t_high,
            "test": pile_metrics(dec[~val], is_def[~val]),
            "test_recovery_at_zero_escape_upper_bound": recovery_at_zero_escape(sc[~val], is_def[~val]),
            "test_image_auroc": None,
        }
        if is_def[~val].any() and (~is_def[~val]).any():
            from sklearn.metrics import roc_auc_score
            res[name]["test_image_auroc"] = float(roc_auc_score(is_def[~val], sc[~val]))
        if name == "patchcore":
            decisions, tl, th = dec, t_low, t_high
    res["baseline_keep_all_rejects"] = {"recovery_rate": 0.0, "escape_rate": 0.0}
    res["patchcore"]["test_heatmap"] = heatmap_quality(maps[~val], masks[~val], is_def[~val])
    res["runtime_seconds"] = round(time.time() - t0, 1)
    res["ms_per_image_scoring"] = None

    # 4. export items for the review app
    t1 = time.time(); model.predict(pile_imgs[:4], S); res["ms_per_image_scoring"] = round((time.time() - t1) / 4 * 1000, 1)
    rows = []
    for i, p in enumerate(pile):
        Image.fromarray(p.image).save(out / "items" / f"{i:04d}.png")
        Image.fromarray(overlay(p.image, maps[i], tl, th)).save(out / "items" / f"{i:04d}_heat.png")
        rows.append({"id": f"{i:04d}", "split": "val" if val[i] else "test", "source": p.source,
                     "variant": p.variant, "defect_type": p.defect_type, "is_defect": int(p.is_defect),
                     "score": round(float(scores[i]), 5), "decision": decisions[i],
                     "rule_score": round(float(rule_scores[i]), 5)})
    with open(out / "items.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # 5. two balanced sets of 10 test items for the user tests (A and B)
    test_ids = [r["id"] for r in rows if r["split"] == "test"]
    d_ids = [r["id"] for r in rows if r["split"] == "test" and r["is_defect"]]
    g_ids = [r["id"] for r in rows if r["split"] == "test" and not r["is_defect"]]
    rng.shuffle(d_ids); rng.shuffle(g_ids)
    k_d = min(4, len(d_ids) // 2); k_g = min(10 - k_d, len(g_ids) // 2)
    sets = {"A": d_ids[:k_d] + g_ids[:k_g], "B": d_ids[k_d:2 * k_d] + g_ids[k_g:2 * k_g]}
    for s in sets.values(): rng.shuffle(s)
    json.dump(sets, open(out / "user_test_sets.json", "w"), indent=2)
    json.dump({"t_low": tl, "t_high": th}, open(out / "thresholds.json", "w"))
    json.dump(res, open(out / "results.json", "w"), indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
