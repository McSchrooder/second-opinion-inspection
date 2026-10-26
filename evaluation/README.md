# Evaluation

One evaluation for every model. It reads score files, not models, so it needs
no GPU, no images and no model code.

## Input

The `items.csv` that `reference_prototype/run_pipeline.py` already writes, one
row per item in the reject pile:

| Column | Required | Meaning |
|---|---|---|
| `id` | yes | unique per item |
| `source` | yes | the original image; variants of one image share it |
| `is_defect` | yes | 1 = real defect, 0 = false reject |
| `score` | yes | higher = more likely a defect |
| `split`, `variant`, `defect_type`, `category` | no | used when present |

A model only has to export this file to be evaluated. A file that does not
match fails with a message saying what is wrong, rather than producing numbers.

## Run

```bash
pip install -r evaluation/requirements.txt
python -m evaluation.evaluate --scores "shared/items/*.csv"
python -m evaluation.evaluate --scores prototype=results/bottle/items.csv --category bottle
python evaluation/tests/test_evaluate.py
```

A glob reads every file and takes model and category from the files' own
columns. `NAME=PATH` names a single file instead, and `--category` is needed
when the file has no category column.

Writes `results/evaluation/runs.csv` (one row per model, run and cut) and
`summary.csv`, and prints a table.

## Reading the table

| Row | Meaning |
|---|---|
| `conformal <= x%` | recovery cut chosen on validation so the expected escape rate is at most x%. **The one to quote.** `(unreachable)` means there are too few validation defects to promise x% at all, so nothing is recovered |
| `prototype (lowest val)` | the v0.2 rule, for comparison |
| `in-sample (not real)` | cut at the lowest *test* defect: zero escapes by construction, not reachable by any threshold set in advance. Shown so such a number is recognisable |

| Column | Meaning |
|---|---|
| recovery | share of false rejects recovered: mean (min to max) over runs |
| escaped | defects wrongly recovered, summed over runs |
| runs | runs with at least one escape |
| expected | escapes a random split predicts for this cut. Far more observed than expected means validation and test defects are not alike |
| 95% bound | worst run's upper bound on the true escape rate |
| to human | share of the pile left for a technician |

## Why these choices

- **Conformal cut instead of the lowest validation defect.** With n validation
  defects, cutting at the k-th lowest gives an expected escape rate of
  k/(n+1) on new defects, for any model. The lowest defect is k = 1.
- **Split by source image.** The pile holds several variants of one good
  image. Splitting by item can put twins on both sides.
- **A bound per run, not pooled over seeds.** Every seed reuses the same defect
  images, so pooling would count them several times.

## The number to keep in mind

Zero escapes out of n defects still allows a true rate of about 3/n. To show
an escape rate below a target with 95% confidence, this many defects must all
be caught:

| Target | Defects needed |
|---|---|
| 1% | 299 |
| 2% | 149 |
| 5% | 59 |

MVTec bottle has 63 defective test images in total.
