"""Checks for the shared evaluation, on scores whose answer is known.

    python evaluation/tests/test_evaluate.py      (or: pytest evaluation)

Each check is a property the evaluation relies on. If one fails, the numbers it
produces cannot be quoted.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.evaluate import (  # noqa: E402
    ScoreFileError,
    conformal_rank,
    cut_at_rank,
    defects_needed,
    escape_upper_bound,
    evaluate_run,
    load_scores,
    main,
    source_split,
)


def synthetic_pile(n_good_sources: int = 40, variants: int = 3, n_defects: int = 60,
                   gap: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """A reject pile in the prototype's items.csv format with a known score shift.

    Good items come in several variants per source image, like the simulated
    nuisance copies. The exported split alternates by row, so it puts variants
    of one source on both sides, which is the leak the source split avoids.
    """
    rng = np.random.default_rng(seed)
    rows, i = [], 0
    for s in range(n_good_sources):
        for v in range(variants):
            rows.append({"id": f"{i:04d}", "split": "val" if i % 2 else "test",
                         "source": f"good_{s:03d}.png",
                         "variant": ("original", "light", "blur")[v % 3],
                         "defect_type": "good", "is_defect": 0,
                         "score": rng.normal(0.0, 1.0)})
            i += 1
    for s in range(n_defects):
        rows.append({"id": f"{i:04d}", "split": "val" if i % 2 else "test",
                     "source": f"defect_{s:03d}.png", "variant": "original",
                     "defect_type": "crack", "is_defect": 1,
                     "score": rng.normal(gap, 1.0)})
        i += 1
    return pd.DataFrame(rows)


def test_rule_of_three():
    # Zero escapes in 30 defects still allows about 9.5%.
    assert abs(escape_upper_bound(0, 30) - 0.0950) < 0.0005
    assert defects_needed(0.01) == 299
    assert escape_upper_bound(0, 299) <= 0.01 < escape_upper_bound(0, 298)
    assert defects_needed(0.05) == 59


def test_conformal_rank():
    assert conformal_rank(30, 0.01) == 0       # 0.31: cannot be met
    assert conformal_rank(30, 0.05) == 1       # 1.55
    assert conformal_rank(30, 0.10) == 3       # 3.1
    assert conformal_rank(99, 0.03) == 3       # exactly 3.0, float error must not lose it
    assert cut_at_rank(np.array([3.0, 1.0, 2.0]), 0) == -math.inf
    assert cut_at_rank(np.array([3.0, 1.0, 2.0]), 2) == 2.0


def test_conformal_guarantee_holds():
    # Over many random draws the conformal cut escapes k/(n+1) of new defects
    # on average, and so stays at or under the target.
    rng = np.random.default_rng(1)
    n_val, n_test, target = 30, 30, 0.10
    k = conformal_rank(n_val, target)
    rates = [(rng.normal(2, 1, n_test) < cut_at_rank(rng.normal(2, 1, n_val), k)).mean()
             for _ in range(4000)]
    mean, expected = float(np.mean(rates)), k / (n_val + 1)
    assert abs(mean - expected) < 0.01, (mean, expected)
    assert mean <= target + 0.01, mean


def test_unreachable_target_recovers_nothing():
    df = synthetic_pile(n_defects=60)          # about 30 validation defects
    rows = evaluate_run(df, source_split(df, 0.5, seed=0), targets=(0.01,))
    r = next(r for r in rows if r["rule"] == "conformal")
    assert r["k"] == 0
    assert r["recovered"] == 0 and r["escapes"] == 0
    assert r["escape_upper95"] == 0.0          # nothing recovered, so exactly zero


def test_source_split_keeps_variants_together():
    df = synthetic_pile()
    for seed in range(5):
        val = source_split(df, 0.5, seed)
        assert not set(df.loc[val, "source"]) & set(df.loc[~val, "source"])
        assert df.loc[val, "is_defect"].any() and df.loc[~val, "is_defect"].any()


def test_exported_split_does_leak():
    # Documents why splitting by source is the default.
    df = synthetic_pile()
    given = (df["split"] == "val").to_numpy()
    assert set(df.loc[given, "source"]) & set(df.loc[~given, "source"])


def test_in_sample_cut_never_escapes():
    df = synthetic_pile(gap=0.5)               # heavy overlap: honest cuts do escape
    for seed in range(5):
        rows = evaluate_run(df, source_split(df, 0.5, seed))
        assert next(r for r in rows if r["rule"] == "in-sample")["escapes"] == 0


def test_schema_errors_are_loud():
    df = synthetic_pile()
    cases = {
        "missing score": df.drop(columns="score"),
        "nan score": df.assign(score=df["score"].where(df.index != 3)),
        "bad label": df.assign(is_defect=df["is_defect"].where(df.index != 3, 2)),
        "duplicate id": pd.concat([df, df.iloc[[0]]]),
        "source both good and defect": df.assign(
            source=df["source"].where(df["is_defect"] == 0, "good_000.png")),
        "no defects": df[df["is_defect"] == 0],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "items.csv"
        for name, frame in cases.items():
            frame.to_csv(path, index=False)
            try:
                load_scores(path, "m", "bottle")
            except ScoreFileError:
                continue
            raise AssertionError(f"{name}: accepted silently")
        df.to_csv(path, index=False)
        try:
            load_scores(path, "m", None)
        except ScoreFileError:
            pass
        else:
            raise AssertionError("missing category accepted silently")


def test_reads_prototype_items_csv():
    df = synthetic_pile().assign(decision="human", rule_score=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "items.csv"
        df.to_csv(path, index=False)
        got = load_scores(path, "prototype", "bottle")
    assert got["id"].iloc[0] == "0000"         # leading zeros survive
    assert got["is_defect"].dtype == bool
    assert (got["category"] == "bottle").all()


def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "items.csv"
        synthetic_pile().to_csv(path, index=False)
        out = Path(tmp) / "eval"
        assert main(["--scores", f"a={path}", "--scores", f"b={path}",
                     "--category", "bottle", "--out", str(out)]) == 0
        runs = pd.read_csv(out / "runs.csv")
        summary = pd.read_csv(out / "summary.csv")
    assert set(runs["model"]) == {"a", "b"}
    assert {"conformal", "prototype", "in-sample"} <= set(summary["rule"])
    assert runs["seed"].nunique() == 5


def test_model_from_file_and_glob():
    # The layout of shared/items/: one file per model and category, with the
    # model and category in their own columns, read in one go by a glob.
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("patchcore", "padim"):
            synthetic_pile().assign(model=name, category="cashew").to_csv(
                Path(tmp) / f"visa_cashew_{name}.csv", index=False)
        out = Path(tmp) / "eval"
        assert main(["--scores", str(Path(tmp) / "*.csv"), "--out", str(out)]) == 0
        runs = pd.read_csv(out / "runs.csv")
    assert set(runs["model"]) == {"patchcore", "padim"}
    assert set(runs["category"]) == {"cashew"}


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  [PASS] {name}")
            except Exception as err:  # noqa: BLE001
                failed += 1
                print(f"  [FAIL] {name}: {err!r}")
    print("\nAll checks passed." if not failed else f"\n{failed} check(s) failed.")
    sys.exit(1 if failed else 0)
