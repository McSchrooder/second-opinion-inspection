"""Explain the band: for items a technician would see, show the image, the model's heatmap and the closest good.

usage: python scripts/overlays.py --results results --data data [--standins results] [--out docs/figures/overlays]

Needs the maps that run.py writes with --save-maps (results/maps/<tag>.npz). For each tag: build the aligned
lane, set t_recover (1% escape) and t_confirm (2% of lane goods above) on the lane, take items in between,
and draw up to 8 of them. Closest good = the training image whose ImageNet resnet18 embedding is nearest.
One more panel per tag: every lane item's score on a line with the two thresholds.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import build_lane, split_tag, t_recover  # noqa: E402
from figures import INK, INK2, SERIES, style  # noqa: E402

ESCAPE, CONFIRM_GOOD_FPR, MAX_ITEMS, MAX_TRAIN = 0.01, 0.02, 8, 300


def load_img(path: Path, size: int = 256) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)))


def embedder():
    """ImageNet resnet18 without its classifier: one 512-vector per image, used only to pick the closest good."""
    from torchvision.models import ResNet18_Weights, resnet18
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    net.fc = torch.nn.Identity()
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

    @torch.no_grad()
    def embed(imgs: list[np.ndarray]) -> torch.Tensor:
        x = torch.stack([torch.from_numpy(i.copy()).permute(2, 0, 1).float() / 255 for i in imgs])
        x = torch.nn.functional.interpolate(x, 224, mode="bilinear", antialias=True)
        x = (x - mean[:, None, None]) / std[:, None, None]
        return torch.nn.functional.normalize(net(x), dim=1)
    return embed


def train_goods(data: Path, ds: str, cat: str) -> list[Path]:
    for d in (data / ds / cat / "train" / "good", data / ds / "visa_pytorch" / cat / "train" / "good"):
        if d.is_dir():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            return list(np.random.default_rng(0).choice(files, min(MAX_TRAIN, len(files)), replace=False))
    return []


def draw(tag: str, items: pd.DataFrame, maps: np.ndarray, closest: list[Path], lane: pd.DataFrame,
         tr: float, tc: float, data: Path, out: Path) -> None:
    ds, cat, model = split_tag(tag)
    n = len(items)
    fig = plt.figure(figsize=(7.2, 2.3 * n + 1.6))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[1] * n + [0.55], hspace=0.35, wspace=0.08)
    for i, ((path, r), m, near) in enumerate(zip(items.iterrows(), maps, closest)):
        img = load_img(data / ds / path)
        ax = fig.add_subplot(gs[i, 0]); ax.imshow(img); ax.axis("off")
        kind = "good" if r.label == 0 else Path(path).parent.name
        ax.set_title(f"{kind}, score {r.score:.3g}", fontsize=8, color=INK, loc="left")
        ax = fig.add_subplot(gs[i, 1]); ax.imshow(img); ax.axis("off")
        hot = np.kron(m, np.ones((4, 4)))
        hot = np.ma.masked_less(hot, np.quantile(m, 0.7))  # only the hottest 30% of the map, the image stays visible
        ax.imshow(hot, cmap="inferno", alpha=0.6, vmin=float(maps.min()), vmax=float(maps.max()))
        ax.set_title("where the model looks", fontsize=8, color=INK2, loc="left")
        ax = fig.add_subplot(gs[i, 2]); ax.axis("off")
        if near is not None:
            ax.imshow(load_img(near))
            ax.set_title("closest good in training", fontsize=8, color=INK2, loc="left")
    ax = fig.add_subplot(gs[n, :])
    good, bad = lane[lane.label == 0].score, lane[lane.label == 1].score
    rng = np.random.default_rng(0)
    ax.scatter(good, rng.uniform(0.2, 0.8, len(good)), s=9, color=SERIES[0], alpha=0.6, label=f"lane goods ({len(good)})")
    ax.scatter(bad, rng.uniform(0.2, 0.8, len(bad)), s=9, color=SERIES[1], alpha=0.6, label=f"lane defects ({len(bad)})")
    ax.axvline(tr, color=INK, lw=0.8); ax.axvline(tc, color=INK, lw=0.8)
    ax.text(tr, 0.97, "pass below ", fontsize=7, color=INK2, va="top", ha="right")
    ax.text(tc, 0.03, " confirm above", fontsize=7, color=INK2, va="bottom", ha="left")
    ax.set_yticks([]); ax.set_ylim(0, 1)
    ax.set_xlabel("score. Between the lines: review", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    style(ax); ax.yaxis.grid(False)
    fig.suptitle(f"{ds} {cat}, {model}: items in the review band", fontsize=10, color=INK, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(top=0.975, bottom=0.03, left=0.04, right=0.98)
    fig.savefig(out / f"{tag}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results", help="holds scores/ and maps/")
    p.add_argument("--standins", default=None, help="holds standin/, defaults to --results")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="docs/figures/overlays")
    p.add_argument("--tags", nargs="*", default=None)
    a = p.parse_args()
    res, data, out = Path(a.results), Path(a.data), Path(a.out)
    standins_dir = Path(a.standins or a.results) / "standin"
    out.mkdir(parents=True, exist_ok=True)
    embed = embedder()
    tags = a.tags or [f.stem for f in sorted((res / "maps").glob("*.npz"))]
    for tag in tags:
        ds, cat, model = split_tag(tag)
        df = pd.read_csv(res / "scores" / f"{tag}.csv").set_index("image")
        z = np.load(res / "maps" / f"{tag}.npz")
        maps = pd.Series(list(z["maps"].astype(np.float32)), index=z["paths"])
        st = standins_dir / f"{ds}_{cat}_aligned.csv"
        lane = df.loc[df.index.intersection(build_lane(pd.read_csv(st).set_index("image"))[0])] if st.exists() else df
        bad, good = lane[lane.label == 1].score.values, lane[lane.label == 0].score.values
        tr = t_recover(bad, ESCAPE)
        tc = max(float(np.quantile(good, 1 - CONFIRM_GOOD_FPR)), tr)
        band = lane[(lane.score >= tr) & (lane.score < tc)].sort_values("score")
        if band.empty:
            print(f"{tag}: empty band"); continue
        pick = band.iloc[np.unique(np.linspace(0, len(band) - 1, min(MAX_ITEMS, len(band))).astype(int))]
        goods = train_goods(data, ds, cat)
        closest = [None] * len(pick)
        if goods:
            g_emb = embed([load_img(g) for g in goods])
            q_emb = embed([load_img(data / ds / pth) for pth in pick.index])
            closest = [goods[i] for i in (q_emb @ g_emb.T).argmax(1).tolist()]
        draw(tag, pick, np.stack([maps[pth] for pth in pick.index]), closest, lane, tr, tc, data, out)
        print(f"{tag}: band {len(band)} of {len(lane)} lane items, drew {len(pick)}")


if __name__ == "__main__":
    main()
