"""Shared evaluation for the second opinion, run on exported score files.

    python -m evaluation.evaluate --scores prototype=results/bottle/items.csv --category bottle
    python -m evaluation.evaluate --scores tiago=a.csv --scores andrija=b.csv --category bottle

Why it exists
-------------
Two models gave two numbers that do not agree, 2-10% of defects escaping
against zero, and there was no shared rule for how either was measured. This
is that rule. Every model exports one file with a score per item, and every
file goes through the same split, the same thresholds and the same metrics.
No GPU, no images and no model code are needed.

The question it answers
-----------------------
The tool may only recover an item that is good, and an escaped defect costs far
more than a false reject. So the question is not accuracy. It is: how many
false rejects can we recover while the escape rate stays below a target, and
how sure can we be of that escape rate? This module answers it for a range of
targets rather than at one threshold.

Three choices, each with its reason
-----------------------------------
1. The recovery cut is a conformal quantile of the validation defect scores.
   Cutting at the k-th lowest of n validation defects gives an expected escape
   rate of k/(n+1) on new defects, whatever the model, as long as validation
   and test defects are exchangeable. A target below 1/(n+1) cannot be met at
   all, and the module then recovers nothing instead of reporting a number it
   cannot back. The prototype's rule, cutting at the lowest validation defect,
   is the case k = 1.

2. The split is by source image, not by item. The simulated reject pile holds
   several variants of the same good image (light, blur, ...). If one lands in
   validation and its twin in test, the test half partly repeats validation.

3. Every observed escape rate comes with a one-sided 95% Clopper-Pearson upper
   bound. Zero escapes out of 30 defects is consistent with a true rate of up
   to about 10%. Seeds are not pooled into one bound, because every seed reuses
   the same defect images, so the bound is reported per run.

Three cuts are reported per file:
  conformal  - the cut above, one row per target. The one to quote.
  prototype  - the lowest validation defect (k = 1), for comparison with v0.2.
  in-sample  - the lowest TEST defect. Zero escapes by construction, so it is
               an upper bound on recovery that no deployed threshold reaches.
               Reported only so such a number is recognisable when it appears.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

REQUIRED_COLUMNS = ("id", "source", "is_defect", "score")
OPTIONAL_COLUMNS = ("split", "variant", "defect_type", "category")
CONFIDENCE = 0.95
DEFAULT_TARGETS = (0.01, 0.02, 0.05, 0.10)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


class ScoreFileError(ValueError):
    """A score file does not match the shared format. Raised, never patched over."""


# --------------------------------------------------------------------- reading

def load_scores(path: str | Path, model: str | None = None,
                category: str | None = None) -> pd.DataFrame:
    """Read and check one score file in the shared format.

    The format is the items.csv that reference_prototype/run_pipeline.py writes:
    one row per item in the reject pile with at least id, source, is_defect
    (0/1) and score (higher = more likely a defect). Anything that would make
    the evaluation quietly wrong raises instead.

    Without a model name, the file's own `model` column is used, as in the
    files under shared/items/, and failing that the file name.
    """
    path = Path(path)
    df = pd.read_csv(path, dtype={"id": str, "source": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ScoreFileError(
            f"{path.name}: missing column(s) {missing}; found {list(df.columns)}. "
            "Expected the items.csv format from reference_prototype/run_pipeline.py: "
            f"{', '.join(REQUIRED_COLUMNS)} (optional: {', '.join(OPTIONAL_COLUMNS)})."
        )

    if "category" in df.columns:
        if category is not None:
            df = df[df["category"] == category]
    elif category is None:
        raise ScoreFileError(f"{path.name}: no category column, so --category is required.")
    else:
        df = df.assign(category=category)
    if df.empty:
        raise ScoreFileError(f"{path.name}: no rows" + (f" for category {category!r}." if category else "."))

    dup = df.loc[df["id"].duplicated(), "id"]
    if len(dup):
        raise ScoreFileError(f"{path.name}: {len(dup)} duplicate id(s), e.g. {list(dup[:3])}.")

    raw = df["is_defect"].astype(str).str.strip().str.lower()
    if not set(raw) <= {"0", "1", "true", "false"}:
        raise ScoreFileError(f"{path.name}: is_defect must be 0/1, found {sorted(set(raw))}.")
    df = df.assign(is_defect=raw.isin({"1", "true"}),
                   score=pd.to_numeric(df["score"], errors="coerce"))
    if df["score"].isna().any():
        raise ScoreFileError(
            f"{path.name}: {int(df['score'].isna().sum())} score(s) missing or not numeric."
        )

    labels_per_source = df.groupby("source")["is_defect"].nunique()
    mixed = labels_per_source[labels_per_source > 1]
    if len(mixed):
        raise ScoreFileError(
            f"{path.name}: source image(s) labelled both good and defective, e.g. "
            f"{list(mixed.index[:3])}. A split by source image would be meaningless."
        )
    if not df["is_defect"].any() or df["is_defect"].all():
        raise ScoreFileError(f"{path.name}: the pile needs both defects and false rejects.")

    if model is None:
        named = df["model"].dropna().unique() if "model" in df.columns else []
        if len(named) > 1:
            raise ScoreFileError(f"{path.name}: several models in one file {list(named)}; "
                                 "pass NAME=PATH or split the file.")
        model = str(named[0]) if len(named) else path.stem
    return df.assign(model=model).reset_index(drop=True)


# ----------------------------------------------------------------- splitting

def source_split(df: pd.DataFrame, val_frac: float, seed: int) -> np.ndarray:
    """True = validation. Whole source images go to one side, stratified by label."""
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    rng = np.random.default_rng(seed)
    label = df.groupby("source")["is_defect"].first().astype(bool)
    val_sources: set[str] = set()
    for is_defect in (True, False):
        sources = np.array(sorted(label.index[label.to_numpy() == is_defect]))
        rng.shuffle(sources)
        val_sources.update(sources[: int(round(len(sources) * val_frac))])
    return df["source"].isin(val_sources).to_numpy()


def given_split(df: pd.DataFrame) -> np.ndarray:
    """The split column as exported. Kept for comparison; it can leak variants."""
    if "split" not in df.columns:
        raise ScoreFileError("--split given needs a 'split' column holding val/test.")
    values = set(df["split"])
    if values != {"val", "test"}:
        raise ScoreFileError(f"split column must hold 'val' and 'test', found {sorted(values)}.")
    return (df["split"] == "val").to_numpy()


# ---------------------------------------------------------- cuts and bounds

def conformal_rank(n_val_defects: int, target: float) -> int:
    """Largest k with k/(n+1) <= target. 0 means the target cannot be met."""
    return math.floor(target * (n_val_defects + 1) + 1e-9)


def cut_at_rank(defect_scores: np.ndarray, k: int) -> float:
    """The k-th lowest defect score. Items strictly below it are recovered."""
    if k < 1:
        return -math.inf
    return float(np.sort(np.asarray(defect_scores, dtype=float))[k - 1])


def escape_upper_bound(escapes: int, defects: int, confidence: float = CONFIDENCE) -> float:
    """One-sided Clopper-Pearson upper bound on the escape rate."""
    if defects <= 0:
        return math.nan
    if escapes >= defects:
        return 1.0
    return float(beta.ppf(confidence, escapes + 1, defects - escapes))


def defects_needed(target: float, confidence: float = CONFIDENCE) -> int:
    """How many defects must all be caught to bound the escape rate below target."""
    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1")
    return math.ceil(math.log(1 - confidence) / math.log(1 - target))


# ------------------------------------------------------------------ metrics

def _metrics(test_scores: np.ndarray, test_defect: np.ndarray,
             t_low: float, t_high: float) -> dict:
    t_high = max(t_high, t_low)
    recover = test_scores < t_low
    confirm = test_scores > t_high
    n_fr, n_def = int((~test_defect).sum()), int(test_defect.sum())
    recovered = int((recover & ~test_defect).sum())
    escapes = int((recover & test_defect).sum())
    # A cut below every score recovers nothing, so nothing can escape: the
    # true escape rate is exactly zero, not merely bounded.
    upper = 0.0 if t_low == -math.inf else escape_upper_bound(escapes, n_def)
    return {
        "test_items": int(len(test_scores)),
        "test_false_rejects": n_fr,
        "test_defects": n_def,
        "recovered": recovered,
        "recovery_rate": recovered / n_fr,
        "escapes": escapes,
        "escape_rate": escapes / n_def,
        "escape_upper95": upper,
        "human_share": float((~recover & ~confirm).mean()),
        "t_low": t_low,
        "t_high": t_high,
    }


def evaluate_run(df: pd.DataFrame, val: np.ndarray,
                 targets: tuple[float, ...] = DEFAULT_TARGETS) -> list[dict]:
    """All cuts for one validation/test split of one model's pile."""
    s = df["score"].to_numpy(dtype=float)
    d = df["is_defect"].to_numpy().astype(bool)
    val_def, val_good = s[val & d], s[val & ~d]
    test_scores, test_defect = s[~val], d[~val]
    if len(val_def) == 0:
        raise ScoreFileError("the validation half has no defects; use more data or another val fraction.")
    if not test_defect.any() or test_defect.all():
        raise ScoreFileError("the test half needs both defects and false rejects.")

    n = len(val_def)
    t_high = float(val_good.max()) if len(val_good) else math.inf
    n_test_def = int(test_defect.sum())
    rows = []

    for target in targets:
        k = conformal_rank(n, target)
        rows.append({"rule": "conformal", "target_escape": target, "k": k,
                     "n_val_defects": n, "guaranteed_escape": k / (n + 1),
                     "expected_escapes": n_test_def * k / (n + 1),
                     **_metrics(test_scores, test_defect, cut_at_rank(val_def, k), t_high)})

    rows.append({"rule": "prototype", "target_escape": math.nan, "k": 1,
                 "n_val_defects": n, "guaranteed_escape": 1 / (n + 1),
                 "expected_escapes": n_test_def / (n + 1),
                 **_metrics(test_scores, test_defect, cut_at_rank(val_def, 1), t_high)})

    t_insample = float(test_scores[test_defect].min())
    rows.append({"rule": "in-sample", "target_escape": math.nan, "k": math.nan,
                 "n_val_defects": n, "guaranteed_escape": math.nan,
                 "expected_escapes": math.nan,
                 **_metrics(test_scores, test_defect, t_insample, t_high)})
    return rows


def evaluate(frames: list[pd.DataFrame], split: str = "source", val_frac: float = 0.5,
             seeds: tuple[int, ...] = DEFAULT_SEEDS,
             targets: tuple[float, ...] = DEFAULT_TARGETS) -> pd.DataFrame:
    """One row per model, category, run and cut."""
    rows = []
    for df in frames:
        for (model, category), part in df.groupby(["model", "category"], sort=False):
            part = part.reset_index(drop=True)
            if split == "given":
                runs = [("given", given_split(part))]
            else:
                runs = [(seed, source_split(part, val_frac, seed)) for seed in seeds]
            for seed, val in runs:
                for r in evaluate_run(part, val, tuple(targets)):
                    rows.append({"model": model, "category": category,
                                 "split": split, "seed": seed, **r})
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Across runs. Escapes are summed for reading, never turned into one bound."""
    keys = ["model", "category", "rule", "target_escape"]
    return runs.groupby(keys, sort=False, dropna=False).agg(
        runs=("seed", "nunique"),
        n_val_defects=("n_val_defects", "max"),
        test_defects_per_run=("test_defects", "max"),
        recovery_mean=("recovery_rate", "mean"),
        recovery_min=("recovery_rate", "min"),
        recovery_max=("recovery_rate", "max"),
        escapes=("escapes", "sum"),
        test_defects=("test_defects", "sum"),
        runs_with_escape=("escapes", lambda x: int((x > 0).sum())),
        expected_escapes=("expected_escapes", "sum"),
        guaranteed_escape=("guaranteed_escape", "max"),
        escape_upper95_worst=("escape_upper95", "max"),
        human_share_mean=("human_share", "mean"),
    ).reset_index()


# -------------------------------------------------------------------- output

def _label(row: pd.Series) -> str:
    if row["rule"] == "conformal":
        tag = f"conformal <= {row['target_escape']:.0%}"
        return tag + (" (unreachable)" if row["guaranteed_escape"] == 0 else "")
    return {"prototype": "prototype (lowest val)", "in-sample": "in-sample (not real)"}[row["rule"]]


def print_summary(summary: pd.DataFrame) -> None:
    for (model, category), part in summary.groupby(["model", "category"], sort=False):
        first = part.iloc[0]
        print(f"\n=== {model} / {category} ===")
        print(f"  {int(first['runs'])} run(s), {int(first['n_val_defects'])} validation and "
              f"{int(first['test_defects_per_run'])} test defects per run\n")
        print(f"  {'cut':<30}{'recovery':>18}{'escaped':>12}{'runs':>7}"
              f"{'expected':>10}{'95% bound':>11}{'to human':>10}")
        for _, r in part.iterrows():
            rec = (f"{r['recovery_mean']:.0%} ({r['recovery_min']:.0%}-"
                   f"{r['recovery_max']:.0%})")
            # The in-sample cut is set on the test defects, so neither an
            # expectation nor a bound means anything for it.
            insample = r["rule"] == "in-sample"
            expected = "-" if insample else f"{r['expected_escapes']:.1f}"
            bound = "-" if insample else f"{r['escape_upper95_worst']:.1%}"
            print(f"  {_label(r):<30}{rec:>18}"
                  f"{int(r['escapes']):>5} of {int(r['test_defects']):<4}"
                  f"{int(r['runs_with_escape']):>3}/{int(r['runs']):<3}"
                  f"{expected:>10}{bound:>11}"
                  f"{r['human_share_mean']:>10.0%}")
    print("\n  runs      = runs with at least one escape")
    print("  expected  = escapes a random split predicts for the cut; far more observed")
    print("              means validation and test defects are not alike")
    print("  95% bound = worst run's upper bound on the true escape rate")
    print("  in-sample = cut at the lowest test defect: zero escapes by construction,")
    print("              not reachable by any threshold set in advance")
    needs = ", ".join(f"{t:.0%}: {defects_needed(t)}" for t in (0.01, 0.02, 0.05))
    print(f"\n  Defects that must all be caught to show an escape rate below a target "
          f"with 95% confidence: {needs}.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Shared evaluation of the second opinion on exported score files.")
    ap.add_argument("--scores", action="append", required=True, metavar="[NAME=]PATH",
                    help="score file(s) in the items.csv format; PATH may be a glob such as "
                         "'shared/items/*.csv'. Without NAME the file's model column is used. "
                         "Repeatable")
    ap.add_argument("--category", default=None,
                    help="category name, required if the file has no category column")
    ap.add_argument("--split", choices=("source", "given"), default="source",
                    help="'source' (default) splits by source image; 'given' uses the file's split column")
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS),
                    help="escape-rate targets for the conformal cut, as fractions")
    ap.add_argument("--out", default="results/evaluation")
    a = ap.parse_args(argv)

    frames = []
    for spec in a.scores:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = None, spec
        if not path or (sep and not name):
            ap.error(f"--scores expects PATH or NAME=PATH, got {spec!r}")
        paths = sorted(glob.glob(path)) if glob.has_magic(path) else [path]
        if not paths:
            ap.error(f"--scores {path!r} matches no file")
        frames.extend(load_scores(p, name, a.category) for p in paths)

    runs = evaluate(frames, a.split, a.val_frac, tuple(a.seeds), tuple(a.targets))
    summary = summarise(runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print_summary(summary)
    print(f"\n  wrote {out / 'runs.csv'} and {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ScoreFileError as err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(2)
