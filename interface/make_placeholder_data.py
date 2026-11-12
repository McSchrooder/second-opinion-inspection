"""Create PLACEHOLDER data in the exact format run_pipeline.py + make_user_test_sets.py produce.

The images are synthetic drawings, NOT real MVTec bottles and NOT real model output. They exist only so the
interface can be built and tested before the real results folder (results/bottle) is available.
When the real folder arrives, nothing in app.py changes: just point --results at it.

    python make_placeholder_data.py            # writes results/placeholder/
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

S = 224
T_LOW, T_HIGH = 0.60, 1.10          # illustrative thresholds, same role as thresholds.json
OUT = Path("results/placeholder")
rng = np.random.default_rng(7)


def bottle() -> np.ndarray:
    yy, xx = np.mgrid[:S, :S]
    r = np.hypot(xx - S / 2, yy - S / 2)
    img = np.full((S, S, 3), 236.0)
    for lo, hi, v in [(0, 34, 28), (34, 52, 62), (52, 70, 118), (70, 84, 78), (84, 100, 40), (100, 104, 150)]:
        img[(r >= lo) & (r < hi)] = v
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255)


def blob(cx, cy, sigma):
    yy, xx = np.mgrid[:S, :S]
    return np.exp(-(((xx - cx * S) ** 2 + (yy - cy * S) ** 2) / (2 * (sigma * S) ** 2)))


def paint(img, kind, cx, cy):
    m = blob(cx, cy, 0.05)[..., None]
    if kind == "broken_rim":
        img = img * (1 - m) + np.array([225, 225, 225]) * m
        img = img * (1 - 0.6 * blob(cx, cy, 0.09)[..., None]) + 0.6 * blob(cx, cy, 0.09)[..., None] * 15
    elif kind == "contamination":          # faint, low contrast
        img = img * (1 - 0.18 * m) + 0.18 * m * 90
    elif kind == "droplets":
        for dx, dy in [(0, 0), (0.05, 0.04), (-0.04, 0.06)]:
            d = blob(cx + dx, cy + dy, 0.018)[..., None]
            img = img * (1 - d) + 235 * d
    elif kind == "dust":
        for _ in range(7):
            d = blob(rng.uniform(0.04, 0.96), rng.choice([0.06, 0.94]), 0.012)[..., None]
            img = img * (1 - 0.8 * d) + 0.8 * d * 60
    elif kind == "light":
        img = img * 0.93 + 8
    return np.clip(img, 0, 255)


def save(arr, path):
    Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(path)


def overlay(img, cx, cy, score, sigma):
    heat = blob(cx, cy, sigma) * score
    a = np.clip((heat - T_LOW) / (T_HIGH - T_LOW), 0, 1)[..., None]
    red = np.zeros_like(img); red[..., 0] = 255
    return img * (1 - 0.6 * a) + red * 0.6 * a


# (defect_type, variant, is_defect, score, decision, where, paint kind)
def spec(is_def, decision, score, kind, variant, dtype, planted=False):
    return dict(is_defect=int(is_def), decision=decision, score=score, kind=kind, variant=variant,
                defect_type=dtype, planted=planted)

def one_set(tag):
    s = [spec(1, "recover", 0.46, "contamination", "original", "contamination", planted=True),   # planted error
         spec(1, "human", 0.84, "broken_rim", "original", "broken_large"),
         spec(1, "defect", 1.45, "broken_rim", "original", "broken_large"),
         spec(1, "defect", 1.62, "broken_rim", "original", "broken_large")]
    s += [spec(0, "recover", v, k, k, "good") for v, k in [(0.31, "light"), (0.38, "light"), (0.44, "light")]]
    s += [spec(0, "human", v, k, k, "good") for v, k in [(0.71, "droplets"), (0.82, "dust"), (0.93, "droplets")]]
    return s

items, sets, planted = [], {}, {}
OUT.mkdir(parents=True, exist_ok=True); (OUT / "items").mkdir(exist_ok=True)
n = 0
for tag in "AB":
    ids = []
    for sp in one_set(tag):
        sp["id"] = f"{n:04d}"; n += 1
        cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        if sp["kind"] == "broken_rim":
            ang = rng.uniform(0, 2 * np.pi); cx, cy = 0.5 + 0.30 * np.cos(ang), 0.5 + 0.30 * np.sin(ang)
        base = paint(bottle(), sp["kind"], cx, cy)
        sigma = 0.09 if sp["kind"] == "broken_rim" else 0.05
        save(base, OUT / "items" / f"{sp['id']}.png")
        save(overlay(base, cx, cy, sp["score"] + (0.2 if sp["kind"] == "broken_rim" else 0), sigma), OUT / "items" / f"{sp['id']}_heat.png")
        sp.update(cx=cx, cy=cy); items.append(sp); ids.append(sp["id"])
        if sp["planted"]: planted[sp["id"]] = "recover"
    rng.shuffle(ids); sets[tag] = ids

# heatmap task items (3): one clear defect, dust, droplets
task = []
for kind, is_def, score in [("broken_rim", 1, 1.5), ("dust", 0, 0.8), ("droplets", 0, 0.75)]:
    iid = f"{n:04d}"; n += 1
    cx, cy = (0.25, 0.5) if kind == "broken_rim" else (0.5, 0.55)
    base = paint(bottle(), kind, cx, cy)
    save(base, OUT / "items" / f"{iid}.png")
    save(overlay(base, cx, cy, score + 0.2, 0.09 if kind == "broken_rim" else 0.05), OUT / "items" / f"{iid}_heat.png")
    items.append(dict(id=iid, is_defect=is_def, decision="defect" if is_def else "human", score=score, kind=kind,
                      variant=kind, defect_type="broken_large" if is_def else "good", cx=cx, cy=cy, planted=False))
    task.append(iid)

with open(OUT / "items.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "split", "source", "variant", "defect_type", "is_defect", "score",
                                       "decision", "rule_score", "peak_x", "peak_y"])
    w.writeheader()
    for it in items:
        w.writerow(dict(id=it["id"], split="test", source=f"placeholder/{it['id']}.png", variant=it["variant"],
                        defect_type=it["defect_type"], is_defect=it["is_defect"], score=round(it["score"], 5),
                        decision=it["decision"], rule_score=0.0, peak_x=round(it["cx"], 4), peak_y=round(it["cy"], 4)))
json.dump(sets, open(OUT / "user_test_sets.json", "w"), indent=2)
json.dump({"t_low": T_LOW, "t_high": T_HIGH}, open(OUT / "thresholds.json", "w"))
json.dump(planted, open(OUT / "user_test_planted.json", "w"), indent=2)
json.dump({"items": task}, open(OUT / "heatmap_task.json", "w"), indent=2)
print("placeholder data written to", OUT, "| sets:", {k: len(v) for k, v in sets.items()}, "| planted:", list(planted))
