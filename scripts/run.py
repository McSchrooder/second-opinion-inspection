"""Fit one anomaly model on one category's good images, score the test split, write scores to disk.

usage: python scripts/run.py visa candle patchcore [--root data] [--out results]
       python scripts/run.py visa macaroni1 patchcore --size 384 --train-n 150
       python scripts/run.py visa candle patchcore --backbone resnet18

Output: results/scores/<tag>.csv (one row per test image: label, raw score, heatmap stats)
and results/<tag>.json (AUROC, recovery at 0/1/5% escape, latency, peak VRAM).
No threshold is set here. lane.py does that from the CSVs. The model is not saved.
"""
import argparse
import json
import platform
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, ReverseDistillation
from anomalib.post_processing import PostProcessor
from common import recovery_at, t_recover

DATASETS = {"visa": Visa, "mvtec": MVTecAD, "goodsad": MVTecAD}  # GoodsAD ships in MVTec AD layout
MODELS = {"padim": Padim, "patchcore": Patchcore, "rd": ReverseDistillation}
DEFAULT_BACKBONE = {"padim": "resnet18", "patchcore": "wide_resnet50_2"}
BACKBONE_SHORT = {"resnet18": "r18", "resnet50": "r50", "wide_resnet50_2": "wr50"}
ESCAPES = (0.0, 0.01, 0.05)
MAP_STATS = ("map_max", "map_mean", "map_std", "map_p90", "map_p99", "map_frac_hi", "map_cx", "map_cy")


def variant_tag(a) -> str:
    """Model token plus non-default settings, e.g. patchcore-320-n300. Hyphens only: '_' splits dataset/category."""
    parts = [a.model]
    if a.size != 256:
        parts.append(str(a.size))
    if a.backbone and a.backbone != DEFAULT_BACKBONE[a.model]:
        parts.append(BACKBONE_SHORT.get(a.backbone, a.backbone.replace("_", "")))
    if a.train_n:
        parts.append(f"n{a.train_n}")
    if a.coreset is not None and a.coreset != 0.1:
        parts.append(f"c{a.coreset:g}".replace(".", "p"))
    if a.accelerator == "cpu":
        parts.append("cpu")
    return "-".join(parts)


def device_name(accelerator: str) -> str:
    if accelerator != "cpu" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    try:
        line = next(ln for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
        return line.split(":", 1)[1].strip()
    except (OSError, StopIteration):
        return platform.processor() or "cpu"


def build_model(a):
    """Stock anomalib model with its own post-processing and metrics switched off.

    Thresholding off: anomalib picks the F1-best threshold, which weighs a missed defect like a
    false reject. We set thresholds at a fixed escape rate in lane.py instead.
    Normalisation off: its 0-1 rescale clamps scores far from the threshold to exactly 0 or 1,
    which erases the tails where our operating point sits.
    Evaluator off: its AUROC sigmoids raw scores and reports 0.5. We use scikit-learn.
    Visualizer off: it only writes overlay PNGs per image.
    """
    cls = MODELS[a.model]
    kw = {}
    if a.backbone:
        kw["backbone"] = a.backbone
    if a.coreset is not None and a.model == "patchcore":
        kw["coreset_sampling_ratio"] = a.coreset
    return cls(
        pre_processor=cls.configure_pre_processor(image_size=(a.size, a.size)),
        post_processor=PostProcessor(enable_normalization=False, enable_thresholding=False),
        evaluator=False, visualizer=False, **kw,
    )


def subsample_train(dm, n: int | None) -> int:
    """Keep a seeded random subset of the training goods. Returns the training size actually used."""
    dm.setup()
    if n and n < len(dm.train_data):
        idx = sorted(np.random.default_rng(0).choice(len(dm.train_data), n, replace=False).tolist())
        dm.train_data.subsample(idx, inplace=True)
    dm._is_setup = True  # anomalib rebuilds train_data in the trainer's setup() unless this is set  # noqa: SLF001
    return len(dm.train_data)


def anomaly_map_stats(amap: torch.Tensor) -> np.ndarray:
    """Per-image heatmap summary (max, mean, std, p90, p99, hot-area share, hot-area centre) for classify.py."""
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
    return torch.stack([
        mx, flat.mean(1), flat.std(1),
        torch.quantile(flat, 0.90, dim=1), torch.quantile(flat, 0.99, dim=1),
        hi.float().mean(1), (hi.float() * xs).sum(1) / wsum / w, (hi.float() * ys).sum(1) / wsum / h,
    ], dim=1).numpy()


@contextmanager
def timed(out: dict, key: str):
    t = time.time()
    yield
    out[key] = round(time.time() - t, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("category")
    p.add_argument("model", choices=MODELS)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int, default=None, help="cap training epochs (rd)")
    p.add_argument("--size", type=int, default=256, help="input resolution")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--backbone", default=None)
    p.add_argument("--coreset", type=float, default=None, help="patchcore coreset ratio")
    p.add_argument("--train-n", type=int, default=None, help="random subset of training goods")
    p.add_argument("--mem-frac", type=float, default=0.6, help="VRAM share this process may use, fails fast past it")
    p.add_argument("--accelerator", choices=["auto", "cpu"], default="auto", help="cpu = time the model on the CPU")
    a = p.parse_args()

    use_gpu = a.accelerator != "cpu" and torch.cuda.is_available()
    if use_gpu and 0 < a.mem_frac < 1:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)

    dm = DATASETS[a.dataset](root=Path(a.root) / a.dataset, category=a.category,
                             train_batch_size=a.batch, eval_batch_size=a.batch, num_workers=4)
    n_train = subsample_train(dm, a.train_n)
    model = build_model(a)
    scratch = tempfile.mkdtemp(prefix="anomalib_")  # anomalib writes a checkpoint per run, we never load it
    engine = Engine(accelerator=a.accelerator, devices=1, logger=False, default_root_dir=scratch,
                    **({"max_epochs": a.epochs} if a.epochs else {}))

    res = {"dataset": a.dataset, "category": a.category, "model": variant_tag(a), "size": a.size,
           "backbone": a.backbone or DEFAULT_BACKBONE[a.model], "coreset": a.coreset,
           "epochs": a.epochs, "n_train": n_train, "device": device_name(a.accelerator)}
    with timed(res, "fit_s"):
        engine.fit(model=model, datamodule=dm)       # learns "normal" from goods only
    with timed(res, "predict_s"):
        preds = engine.predict(model=model, datamodule=dm)  # raw score + heatmap per test image

    scores = torch.cat([b.pred_score.flatten().cpu() for b in preds]).numpy()
    labels = torch.cat([b.gt_label.flatten().cpu() for b in preds]).numpy().astype(int)
    paths = [str(Path(pth).relative_to(Path(a.root) / a.dataset)) for b in preds for pth in b.image_path]
    stats = np.concatenate([anomaly_map_stats(b.anomaly_map) for b in preds])
    shutil.rmtree(scratch, ignore_errors=True)

    tag = f"{a.dataset}_{a.category}_{res['model']}"
    score_file = Path(a.out) / "scores" / f"{tag}.csv"
    score_file.parent.mkdir(parents=True, exist_ok=True)
    with score_file.open("w") as f:
        f.write("image,label,score," + ",".join(MAP_STATS) + "\n")
        for pth, lab, sc, st in zip(paths, labels, scores, stats):
            f.write(f"{pth},{lab},{sc:.6f}," + ",".join(f"{v:.6f}" for v in st) + "\n")

    res.update({
        "n_good": int((labels == 0).sum()), "n_defect": int((labels == 1).sum()),
        "auroc": round(float(roc_auc_score(labels, scores)), 4),
        "ms_per_image": round(1000 * res["predict_s"] / len(scores), 1),
    })
    if use_gpu:
        res["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    for e in ESCAPES:  # optimistic: threshold on these same defects; lane.py has the held-out version
        res[f"recovery@{e:g}"] = round(recovery_at(scores, labels, e), 4)
        res[f"thr@{e:g}"] = round(t_recover(scores[labels == 1], e), 4)
    (Path(a.out) / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
