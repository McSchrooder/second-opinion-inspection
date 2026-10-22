"""Train one anomaly model on one category and report recovery at fixed escape rate.

usage: python scripts/run.py visa candle patchcore [--root DATA_DIR] [--out results]
       python scripts/run.py visa macaroni1 patchcore --size 384 --train-n 400
       python scripts/run.py visa candle patchcore --backbone resnet18

recovery@e = share of good items scored below the threshold that lets through
             a fraction e of true defects (TNR at fixed FNR). This is the metric
             we care about; AUROC is reported only for reference.

Variants (resolution, backbone, train subset, coreset ratio) get their own tag,
e.g. visa_macaroni1_patchcore-384-n400, so they sit next to the baseline in the
results tables. Model tokens never contain underscores (lane.py splits on them).

VRAM is capped per process (--mem-frac, default 60 % of the card) so a run that
needs more fails fast with an OOM instead of thrashing the GPU.
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
MAP_STAT_NAMES = ("map_max", "map_mean", "map_std", "map_p90", "map_p99", "map_frac_hi", "map_cx", "map_cy")
BACKBONE_SHORT = {"resnet18": "r18", "resnet50": "r50", "wide_resnet50_2": "wr50"}
DEFAULT_BACKBONE = {"padim": "resnet18", "patchcore": "wide_resnet50_2"}


def anomaly_map_stats(amap: torch.Tensor) -> np.ndarray:
    """Cheap per-image summary of the anomaly map: spread, tail, how much of the image fires, where."""
    m = amap.detach().float().cpu()
    if m.ndim == 4:
        m = m[:, 0]
    n, h, w = m.shape
    flat = m.reshape(n, -1)
    mx = flat.max(1).values
    hi = flat >= (0.5 * mx[:, None])
    ys = torch.arange(h).float()[:, None].expand(h, w).reshape(-1)
    xs = torch.arange(w).float()[None, :].expand(h, w).reshape(-1)
    wsum = hi.float().sum(1).clamp(min=1)
    cy = (hi.float() * ys).sum(1) / wsum / h
    cx = (hi.float() * xs).sum(1) / wsum / w
    return torch.stack([
        mx, flat.mean(1), flat.std(1),
        torch.quantile(flat, 0.90, dim=1), torch.quantile(flat, 0.99, dim=1),
        hi.float().mean(1), cx, cy,
    ], dim=1).numpy()


def recovery_at_escape(scores: np.ndarray, labels: np.ndarray, escape: float) -> tuple[float, float]:
    """Return (recovery, threshold). labels: 1 = defect, 0 = good."""
    defect = np.sort(scores[labels == 1])
    # threshold = lowest defect score we are willing to miss `escape` of
    k = int(np.floor(escape * len(defect)))
    thr = defect[k] if k < len(defect) else defect[-1]
    good = scores[labels == 0]
    return float((good < thr).mean()), float(thr)


def variant_tag(a) -> str:
    """Model token plus non-default settings, hyphen separated (no underscores)."""
    parts = [a.model]
    if a.size != 256:
        parts.append(str(a.size))
    if a.backbone and a.backbone != DEFAULT_BACKBONE.get(a.model):
        parts.append(BACKBONE_SHORT.get(a.backbone, a.backbone.replace("_", "")))
    if a.train_n:
        parts.append(f"n{a.train_n}")
    if a.coreset is not None and a.coreset != 0.1:
        parts.append(f"c{a.coreset:g}".replace(".", "p"))
    return "-".join(parts)


def build_model(a):
    cls = MODELS[a.model]
    kw = {}
    if a.model == "efficientad":
        kw["imagenet_dir"] = Path(a.root) / "imagenette"
    if a.backbone and a.model in ("padim", "patchcore"):
        kw["backbone"] = a.backbone
    if a.coreset is not None and a.model == "patchcore":
        kw["coreset_sampling_ratio"] = a.coreset
    # raw scores: no normalisation, no F1 threshold; we set the operating point ourselves
    return cls(
        pre_processor=cls.configure_pre_processor(image_size=(a.size, a.size)),
        post_processor=PostProcessor(enable_normalization=False, enable_thresholding=False),
        evaluator=False, visualizer=False, **kw,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("category")
    p.add_argument("model", choices=MODELS)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="max training steps (EfficientAD)")
    p.add_argument("--size", type=int, default=256, help="input resolution")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--backbone", default=None, help="feature extractor for padim/patchcore")
    p.add_argument("--coreset", type=float, default=None, help="patchcore coreset sampling ratio")
    p.add_argument("--train-n", type=int, default=None, help="random subset of training images (seeded)")
    p.add_argument("--mem-frac", type=float, default=0.6, help="cap on this process's share of VRAM")
    a = p.parse_args()

    if torch.cuda.is_available() and 0 < a.mem_frac < 1:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)

    train_batch = 1 if a.model == "efficientad" else a.batch  # anomalib requires 1 for EfficientAD
    dm = DATASETS[a.dataset](
        root=Path(a.root) / a.dataset, category=a.category,
        train_batch_size=train_batch, eval_batch_size=a.batch, num_workers=4,
    )
    dm.setup()
    if a.train_n and a.train_n < len(dm.train_data):
        rng = np.random.default_rng(0)
        idx = sorted(rng.choice(len(dm.train_data), a.train_n, replace=False).tolist())
        dm.train_data.subsample(idx, inplace=True)
    dm._is_setup = True  # keep the subset: without this the trainer's setup() rebuilds train_data from scratch
    n_train = len(dm.train_data)

    model = build_model(a)
    kw = {}
    if a.epochs:
        kw["max_epochs"] = a.epochs
    if a.steps:
        kw["max_steps"] = a.steps
    engine = Engine(accelerator="auto", devices=1, logger=False, default_root_dir=a.out, **kw)

    t = time.time()
    engine.fit(model=model, datamodule=dm)
    fit_s = time.time() - t

    t = time.time()
    preds = engine.predict(model=model, datamodule=dm)
    pred_s = time.time() - t

    scores = torch.cat([b.pred_score.flatten().cpu() for b in preds]).numpy()
    labels = torch.cat([b.gt_label.flatten().cpu() for b in preds]).numpy().astype(int)
    paths = [str(Path(pth).relative_to(Path(a.root) / a.dataset)) for b in preds for pth in b.image_path]
    map_stats = np.concatenate([anomaly_map_stats(b.anomaly_map) for b in preds])

    # per-image scores plus anomaly-map statistics, so lane evaluation,
    # calibration and a second-stage classifier can run offline
    tag = f"{a.dataset}_{a.category}_{variant_tag(a)}"
    score_file = Path(a.out) / "scores" / f"{tag}.csv"
    score_file.parent.mkdir(parents=True, exist_ok=True)
    with score_file.open("w") as f:
        f.write("image,label,score," + ",".join(MAP_STAT_NAMES) + "\n")
        for pth, lab, sc, ms in zip(paths, labels, scores, map_stats):
            f.write(f"{pth},{lab},{sc:.6f}," + ",".join(f"{v:.6f}" for v in ms) + "\n")

    res = {
        "dataset": a.dataset, "category": a.category, "model": variant_tag(a), "size": a.size,
        "backbone": a.backbone or DEFAULT_BACKBONE.get(a.model), "coreset": a.coreset,
        "epochs": a.epochs, "steps": a.steps, "n_train": n_train,
        "n_good": int((labels == 0).sum()), "n_defect": int((labels == 1).sum()),
        "auroc": round(float(roc_auc_score(labels, scores)), 4),
        "fit_s": round(fit_s, 1), "predict_s": round(pred_s, 1),
        "ms_per_image": round(1000 * pred_s / len(scores), 1),
    }
    if torch.cuda.is_available():
        res["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        res["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    for e in ESCAPE_RATES:
        r, thr = recovery_at_escape(scores, labels, e)
        res[f"recovery@{e:g}"] = round(r, 4)
        res[f"thr@{e:g}"] = round(thr, 4)

    out = Path(a.out) / f"{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
