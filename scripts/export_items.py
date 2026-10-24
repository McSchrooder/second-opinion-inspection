"""Export per-item scores in the reference prototype's items.csv layout for the shared evaluation.

usage: python scripts/export_items.py [--results results] [--out results/items]

One file per run (dataset_category_model.csv) with the prototype's columns:
  id, split, source, variant, defect_type, is_defect, score, decision, rule_score
plus dataset, category, model appended. `split` is "test" throughout: we do not
hold out a validation subset up front, thresholds in docs/results.md are
calibrated 5-fold on the test split instead. `variant` is "original" (no nuisance
copies). `rule_score` is the golden-template stand-in score (our simulated
primary camera) when available. `decision` is left empty for the evaluator.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd


def defect_type(path: str, label: int) -> str:
    parent = Path(path).parent.name
    return "good" if label == 0 else parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "items"
    out.mkdir(parents=True, exist_ok=True)

    standins = {f.stem: pd.read_csv(f).set_index("image") for f in (res / "standin").glob("*_template.csv")}
    n = 0
    for f in sorted((res / "scores").glob("*.csv")):
        ds, rest = f.stem.split("_", 1)
        cat, model = rest.rsplit("_", 1)
        df = pd.read_csv(f)
        rule = standins.get(f"{ds}_{cat}_template")
        rows = []
        for i, r in df.iterrows():
            rs = rule.loc[r.image, "score"] if rule is not None and r.image in rule.index else ""
            rows.append({
                "id": f"{i:04d}", "split": "test", "source": r.image, "variant": "original",
                "defect_type": defect_type(r.image, int(r.label)), "is_defect": int(r.label),
                "score": round(float(r.score), 5), "decision": "",
                "rule_score": round(float(rs), 5) if rs != "" else "",
                "dataset": ds, "category": cat, "model": model,
            })
        with (out / f"{f.stem}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        n += 1
    print(f"wrote {n} item files to {out}")


if __name__ == "__main__":
    main()
