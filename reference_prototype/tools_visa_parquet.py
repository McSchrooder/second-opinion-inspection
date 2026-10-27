"""Convert VisA parquet files (Hugging Face mirror BrachioLab/visa) into the MVTec folder layout
that second_opinion/data.py reads: <root>/<cat>/train/good, test/good, test/bad, ground_truth/bad.

Images are resized to --size (square) to keep the files small; masks with nearest-neighbour.

Example:
    python tools_visa_parquet.py --parquet-dir downloads --out data/visa --categories cashew chewinggum
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def to_img(cell) -> Image.Image | None:
    if cell is None:
        return None
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return None if b is None else Image.open(io.BytesIO(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    S = a.size
    for cat in a.categories:
        counts = {}
        for split in ("train", "test"):
            df = pd.read_parquet(next(Path(a.parquet_dir).glob(f"{cat}.{split}-*.parquet")))
            for i, r in df.iterrows():
                lab = int(r["label"])
                kind = "good" if lab == 0 else "bad"
                d = Path(a.out) / cat / split / kind
                d.mkdir(parents=True, exist_ok=True)
                to_img(r["image"]).convert("RGB").resize((S, S), Image.BILINEAR).save(d / f"{i:04d}.png")
                if lab == 1:
                    m = to_img(r["mask"])
                    g = Path(a.out) / cat / "ground_truth" / "bad"
                    g.mkdir(parents=True, exist_ok=True)
                    arr = np.zeros((S, S), np.uint8) if m is None else \
                        (np.array(m.convert("L").resize((S, S), Image.NEAREST)) > 0).astype(np.uint8) * 255
                    Image.fromarray(arr).save(g / f"{i:04d}_mask.png")
                counts[(split, kind)] = counts.get((split, kind), 0) + 1
        print(cat, counts)


if __name__ == "__main__":
    main()
