"""Summary figures from results/lane_summary.csv (written by lane.py).

usage: python scripts/figures.py [--results results] [--out docs/figures]

  recovery_heatmap.png   category (grouped by tier) x model, recovery at 1% escape on the lane
  resolution_effect.png  PatchCore at 256/320/384 px and 150/300/all training images, weak categories
  speed_vs_recovery.png  mean lane recovery vs inference latency per model
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# tiers: rigid single objects / textured single items / multi-instance trays
TIERS = [
    ("rigid object", ["bottle", "hazelnut"]),
    ("textured single item", ["pipe_fryum", "chewinggum", "cashew", "candle", "fryum"]),
    ("multi-instance tray", ["macaroni1", "capsules", "macaroni2"]),
]
MODELS = [("padim", "PaDiM 256"), ("patchcore", "PatchCore 256"), ("rd", "Reverse Dist. 256"),
          ("patchcore-320-n300", "PatchCore 320 (300 img)"), ("patchcore-r18", "PatchCore 256 resnet18")]
VARIANTS = [("patchcore", "256 px, all images"), ("patchcore-n150", "256 px, 150 images"),
            ("patchcore-320-n150", "320 px, 150 images"), ("patchcore-384-n150", "384 px, 150 images"),
            ("patchcore-320-n300", "320 px, 300 images")]
WEAK = ["candle", "cashew", "fryum", "macaroni1", "capsules", "macaroni2"]
COL = "aligned_rec@0.01"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def heatmap(df: pd.DataFrame, out: Path) -> None:
    cats = [c for _, cs in TIERS for c in cs]
    keys = [m for m, _ in MODELS]
    mat = np.full((len(cats), len(keys)), np.nan)
    for i, c in enumerate(cats):
        for j, m in enumerate(keys):
            r = df[(df.category == c) & (df.model == m)]
            if len(r):
                mat[i, j] = r[COL].iloc[0]
    cmap = matplotlib.colors.ListedColormap(SEQ)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.imshow(np.nan_to_num(mat, nan=0), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([n for _, n in MODELS], fontsize=8, color=INK2)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=8, color=INK2)
    for i in range(len(cats)):
        for j in range(len(keys)):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=INK2)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.55 else INK)
    # white gaps between cells and tier separators
    ax.set_xticks(np.arange(-0.5, len(keys)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats)), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=2)
    ax.tick_params(which="minor", length=0)
    row = -0.5
    for name, cs in TIERS:
        ax.text(len(keys) - 0.4, row + len(cs) / 2, name, rotation=270, va="center", ha="left", fontsize=8, color=INK2)
        row += len(cs)
        if row < len(cats) - 0.5:
            ax.axhline(row, color=INK2, linewidth=1.2)
    ax.set_title("Share of good items recovered at 1% escape, simulated reject lane", fontsize=10, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "recovery_heatmap.png", dpi=160)
    plt.close(fig)


def resolution(df: pd.DataFrame, out: Path) -> None:
    keys = [m for m, _ in VARIANTS]
    x = np.arange(len(WEAK))
    w = 0.15
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for j, (m, label) in enumerate(VARIANTS):
        vals = [df[(df.category == c) & (df.model == m)][COL].iloc[0] if len(df[(df.category == c) & (df.model == m)]) else 0 for c in WEAK]
        ax.bar(x + (j - 2) * (w + 0.01), vals, width=w, color=SERIES[j], label=label, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(WEAK, fontsize=8, color=INK2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("recovery at 1% escape (lane)", fontsize=8, color=INK2)
    ax.set_title("PatchCore recovery by input resolution and training-set size, lane, 1% escape", fontsize=10, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "resolution_effect.png", dpi=160)
    plt.close(fig)


def speed(df: pd.DataFrame, devices: dict, out: Path) -> None:
    """Mean lane recovery vs latency per model on the GPU. A hollow marker at the same height shows
    the same model's latency on the CPU where it was timed (tag suffix -cpu, bottle only)."""
    names = dict(MODELS)
    gpu = df[df.model.isin(names)].groupby("model").agg(rec=(COL, "mean"), ms=("ms_per_image", "mean"))
    cpu = df[df.model.str.endswith("-cpu")].groupby("model").agg(ms=("ms_per_image", "mean"))
    offsets = {"patchcore": (-6, 8), "patchcore-320-n300": (6, 4), "patchcore-r18": (6, -11)}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m, r in gpu.iterrows():
        ax.scatter(r.ms, r.rec, s=48, color=SERIES[0], edgecolor="#ffffff", linewidth=1.5, zorder=3)
        dx, dy = offsets.get(m, (6, 2))
        ax.annotate(names[m], (r.ms, r.rec), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK2,
                    ha="right" if dx < 0 else "left")
        if f"{m}-cpu" in cpu.index:
            cms = cpu.loc[f"{m}-cpu", "ms"]
            ax.plot([r.ms, cms], [r.rec, r.rec], color=SERIES[0], lw=0.8, ls=":", zorder=2)
            ax.scatter(cms, r.rec, s=48, facecolor="#ffffff", edgecolor=SERIES[0], linewidth=1.5, zorder=3)
            ax.annotate("cpu", (cms, r.rec), xytext=(6, 2), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("inference ms per image, log scale. Filled: GPU. Hollow: same model on CPU", fontsize=8, color=INK2)
    ax.set_ylabel("mean recovery at 1% escape, lane, 10 categories", fontsize=8, color=INK2)
    ax.set_ylim(0.3, 0.8)
    ax.set_title("Latency vs recovery per model", fontsize=10, color=INK, loc="left")
    fig.text(0.01, 0.01, ", ".join(f"{k}: {v}" for k, v in sorted(devices.items())), fontsize=6.5, color=INK2)
    style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "speed_vs_recovery.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="docs/figures")
    a = p.parse_args()
    df = pd.read_csv(Path(a.results) / "lane_summary.csv")
    devices = {}
    for f in Path(a.results).glob("*.json"):
        j = json.load(open(f))
        if "device" in j:
            devices["cpu" if j["model"].endswith("-cpu") else "gpu"] = j["device"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    heatmap(df, out)
    resolution(df, out)
    speed(df, devices, out)
    print(f"wrote 3 figures to {out}")


if __name__ == "__main__":
    main()
