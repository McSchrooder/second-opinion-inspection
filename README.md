# Second Opinion for Vision Inspection

AI second opinion for camera inspection on production lines.
Inline inspection systems reject good products, which end up as waste.
This tool re-checks rejected items, shows a heatmap of where it sees a defect,
and recovers the good ones. Uncertain cases go to a human.

## Data

Datasets are not committed. `data/` is ignored by git.

## Run

```bash
pip install torch torchvision   # pick the build for your GPU
pip install -r requirements.txt
python scripts/download.py visa
python scripts/run.py visa candle patchcore
```

Prints recovery of good items at 0, 1 and 5 percent escape of true defects,
plus AUROC for reference. Results land in `results/` as JSON.
