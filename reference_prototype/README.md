# Reference prototype (Tiago, v0.2, 18 Oct 2026)

A complete, small baseline of the whole product loop: simulate a camera's reject pile, score it with a PatchCore-style model, make a three-way decision, evaluate, and review items in an app. It lives in its own folder so it does not clash with the main model code. Take whatever pieces are useful.

## What is inside

| File | What it does |
|---|---|
| `second_opinion/data.py` | Loads any dataset in the MVTec AD folder layout (MVTec AD, VisA 1cls export) |
| `second_opinion/simulate.py` | Builds the reject pile: harmless nuisance on good images (light, droplets, dust, shift, blur) and a rule-based "primary camera" tuned to never miss a defect. Its mistakes are our false rejects |
| `second_opinion/model.py` | PatchCore-style detector: memory bank of good patches (wide_resnet50_2 layer2 + layer3), greedy coreset, nearest-neighbour distance as heatmap, max as image score |
| `second_opinion/decide.py` | Three-way decision (recover / human / defect), thresholds from validation then frozen, safety margin, metrics |
| `run_pipeline.py` | One full run: results, images, heatmaps and user-test sets (A and B) |
| `sweep.py` | Robustness: several seeds x several safety margins, one model fit per seed |
| `app.py` | Streamlit review app for the user tests (modes with/without tool, logs time and decisions) |
| `tests/make_synthetic.py` | Tiny synthetic dataset, only to check the code runs |

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py --data-root data/mvtec --category bottle --coreset-ratio 0.01 --out results/bottle
python sweep.py --data-root data/mvtec --category bottle --seeds 0 1 2 3 4 --margins 0 0.05 0.1 0.2 --out results/sweep.csv
streamlit run app.py -- --results results/bottle
```

Without internet, pass ImageNet weights with `--weights wide_resnet50_2-95faca4d.pth`.

## Data note

The official MVTec per-category link returned 404 on 18 Oct 2026. The bottle category was taken from the Hugging Face mirror `foersben/mvtec-ad` and checked against Table 1 of Bergmann et al., IJCV 2021: 209 train, 20 good test, 63 defective test (3 types), 900 px. Cite the official paper: https://doi.org/10.1007/s11263-020-01400-4

## Results so far (MVTec bottle, test half: 25 to 28 false rejects + 31 defects per run)

| Safety margin | Recovery, mean (min to max) | Defects escaped (5 runs) | Runs with an escape | Sent to human |
|---|---|---|---|---|
| 0% | 80% (64 to 96%) | 15 of 155 | 5 of 5 | 21% |
| 5% | 74% (56 to 93%) | 10 of 155 | 4 of 5 | 31% |
| 10% | 69% (52 to 93%) | 9 of 155 | 3 of 5 | 38% |
| 20% | 50% (8 to 74%) | 3 of 155 | 1 of 5 | 61% |

Image AUROC 0.955 to 0.981, heatmap pixel AUROC about 0.98, about 106 ms per image on a 2-core CPU.

**Takeaway:** the model ranks well, but thresholds learned on about 30 validation defects do not hold on unseen defects. Proposal: no autonomous recovery; the tool sorts and explains, a technician approves.

## Open questions for the model work

- Which defect types escape, and is the nuisance in good images pushing thresholds down?
- Does a calibrated threshold (e.g. conformal prediction) or a 10% coreset reduce escapes?
- How does it do on VisA food categories?

## Limitations

Simulated false rejects, a harsh simulated camera (85% of good items rejected), public images only.
