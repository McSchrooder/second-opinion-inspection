"""Robustness sweep: several seeds x several safety margins, one model fit per seed.

Answers two questions for the report:
  1. How much does recovery vary between runs?  (seeds)
  2. What does extra safety cost in manual review?  (margins)

Example:
    python sweep.py --data-root data/mvtec --category bottle --weights w.pth \
        --seeds 0 1 2 3 4 --margins 0 0.05 0.1 0.2 --out results/sweep_bottle.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from run_pipeline import stratified_split
from second_opinion.data import load_split, read_image, read_mask
from second_opinion.decide import choose_thresholds, decide, pile_metrics
from second_opinion.model import PatchCore
from second_opinion.simulate import Item, RuleInspector, build_reject_pile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--coreset-ratio", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = a.image_size

    train, test = load_split(a.data_root, a.category)
    train_imgs = np.stack([read_image(s.path, S) for s in train])
    base_items = [Item(read_image(s.path, S), read_mask(s.mask_path, S), s.is_defect, s.path.name,
                       "original", s.defect_type) for s in test]
    inspector = RuleInspector(train_imgs)
    inspector.calibrate_never_miss([i.image for i in base_items if i.is_defect])

    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        pile, _ = build_reject_pile(base_items, inspector, 2, rng)
        is_def = np.array([p.is_defect for p in pile])
        val = stratified_split(is_def, 0.5, rng)
        model = PatchCore("wide_resnet50_2", a.weights, S, a.coreset_ratio, seed=seed)
        model.fit(train_imgs)
        scores, _ = model.predict(np.stack([p.image for p in pile]), S)
        for m in a.margins:
            tl, th = choose_thresholds(scores[val], is_def[val], m)
            met = pile_metrics(decide(scores, tl, th)[~val], is_def[~val])
            rows.append({"seed": seed, "margin": m, **met})
            print(seed, m, met["recovery_rate"], met["defect_escapes"], met["human_share"], flush=True)
        with open(a.out, "w", newline="") as fh:  # rewrite after every seed so partial runs are kept
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
