"""Tiny synthetic dataset in MVTec layout, ONLY to smoke-test the code end to end.
Results on it mean nothing about real performance.

Good item: a bright disc (bottle seen from above) on a dark background with texture.
Defect: a dark scratch or a dark spot on the disc, with its mask.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def item(rng, defect=None, size=128):
    img = Image.new("RGB", (size, size), (30, 30, 35))
    d = ImageDraw.Draw(img)
    c, r = size // 2 + rng.integers(-2, 3), int(size * 0.35)
    d.ellipse([c - r, c - r, c + r, c + r], fill=(200, 200, 190))
    d.ellipse([c - r // 3, c - r // 3, c + r // 3, c + r // 3], fill=(150, 150, 140))
    mask = Image.new("L", (size, size), 0); m = ImageDraw.Draw(mask)
    if defect == "scratch":
        x0, y0 = rng.integers(c - r // 2, c), rng.integers(c - r // 2, c)
        pts = [x0, y0, x0 + rng.integers(15, 30), y0 + rng.integers(5, 25)]
        d.line(pts, fill=(60, 60, 60), width=3); m.line(pts, fill=255, width=5)
    elif defect == "spot":
        x, y = rng.integers(c - r // 2, c + r // 2, 2)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(70, 50, 40)); m.ellipse([x - 6, y - 6, x + 6, y + 6], fill=255)
    arr = np.asarray(img).astype(np.float32) + rng.normal(0, 4, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), mask


def main(root):
    rng = np.random.default_rng(0)
    base = Path(root) / "synth"
    for sub in ["train/good", "test/good", "test/scratch", "test/spot", "ground_truth/scratch", "ground_truth/spot"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    for i in range(40):
        item(rng)[0].save(base / "train/good" / f"{i:03d}.png")
    for i in range(20):
        item(rng)[0].save(base / "test/good" / f"{i:03d}.png")
    for kind in ("scratch", "spot"):
        for i in range(10):
            im, mk = item(rng, kind)
            im.save(base / "test" / kind / f"{i:03d}.png"); mk.save(base / "ground_truth" / kind / f"{i:03d}_mask.png")
    print(base)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
