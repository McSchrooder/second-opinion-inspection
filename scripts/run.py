"""Train one anomaly model on one category and report recovery at fixed escape rate.

usage: python scripts/run.py visa candle patchcore [--root DATA_DIR] [--out results]

recovery@e = share of good items scored below the threshold that lets through
             a fraction e of true defects (TNR at fixed FNR). This is the metric
             we care about; AUROC is reported only for reference.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore, ReverseDistillation
from anomalib.post_processing import PostProcessor

DATASETS = {"visa": Visa, "mvtec": MVTecAD}
MODELS = {
    "padim": Padim,
    "patchcore": Patchcore,
    "efficientad": EfficientAd,
    "rd": ReverseDistillation,
}
ESCAPE_RATES = (0.0, 0.01, 0.05)


def recovery_at_escape(scores: np.ndarray, labels: np.ndarray, escape: float) -> tuple[float, float]:
    """Return (recovery, threshold). labels: 1 = defect, 0 = good."""
    defect = np.sort(scores[labels == 1])
    # threshold = lowest defect score we are willing to miss `escape` of
    k = int(np.floor(escape * len(defect)))
    thr = defect[k] if k < len(defect) else defect[-1]
    good = scores[labels == 0]
    return float((good < thr).mean()), float(thr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("category")
    p.add_argument("model", choices=MODELS)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()

    dm = DATASETS[a.dataset](
        root=Path(a.root) / a.dataset, category=a.category,
        train_batch_size=a.batch, eval_batch_size=a.batch, num_workers=4,
    )
    # raw scores: no normalisation, no F1 threshold; we set the operating point ourselves
    model = MODELS[a.model](
        post_processor=PostProcessor(enable_normalization=False, enable_thresholding=False),
        evaluator=False, visualizer=False,
    )
    kw = {"max_epochs": a.epochs} if a.epochs else {}
    engine = Engine(accelerator="auto", devices=1, logger=False, default_root_dir=a.out, **kw)

    t = time.time()
    engine.fit(model=model, datamodule=dm)
    fit_s = time.time() - t

    t = time.time()
    preds = engine.predict(model=model, datamodule=dm)
    pred_s = time.time() - t

    scores = torch.cat([b.pred_score.flatten().cpu() for b in preds]).numpy()
    labels = torch.cat([b.gt_label.flatten().cpu() for b in preds]).numpy().astype(int)

    res = {
        "dataset": a.dataset, "category": a.category, "model": a.model,
        "n_good": int((labels == 0).sum()), "n_defect": int((labels == 1).sum()),
        "auroc": round(float(roc_auc_score(labels, scores)), 4),
        "fit_s": round(fit_s, 1), "predict_s": round(pred_s, 1),
        "ms_per_image": round(1000 * pred_s / len(scores), 1),
    }
    for e in ESCAPE_RATES:
        r, thr = recovery_at_escape(scores, labels, e)
        res[f"recovery@{e:g}"] = round(r, 4)
        res[f"thr@{e:g}"] = round(thr, 4)

    out = Path(a.out) / f"{a.dataset}_{a.category}_{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
