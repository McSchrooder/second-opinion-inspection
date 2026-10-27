"""Build two balanced image sets (A and B) for the user tests, with one planted tool error each.

Why balanced: with 4 to 6 participants, a set that is easier with the tool would look like a
tool effect. Both sets get the same mix of ground truth and tool answers.

Why a planted error: our real escape rate is 1 to 3%, so in 20 images the tool would almost
never be wrong, and we could not see whether technicians check the tool or follow it.
In each set, one real defect is shown with the tool answer "Likely good". We use the two
defects that actually escaped most often in our robustness sweep (low-contrast
contamination), so the error is realistic. Participants are told at the end (debrief).

Per set (10 images):
  4 real defects:   1 shown as "likely good" (planted), 1 "uncertain", 2 "defect confirmed"
  6 false rejects:  3 "likely good", 3 "uncertain"

Example:
    python make_user_test_sets.py --results results/bottle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PLANTED_SOURCES = ["contamination/003.png", "contamination/019.png"]


def pick_pairs(df: pd.DataFrame, n_pairs: int) -> tuple[list[str], list[str]]:
    """Sort by score and deal neighbouring items to A and B, so both sets get similar scores."""
    df = df.sort_values("score")
    step = max(1, len(df) // (2 * n_pairs))
    chosen = df.iloc[::step].head(2 * n_pairs)
    ids = list(chosen.id)
    return ids[0::2], ids[1::2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    R = Path(a.results)
    d = pd.read_csv(R / "items.csv", dtype={"id": str})
    d["key"] = d.defect_type + "/" + d.source

    planted = d[d.key.isin(PLANTED_SOURCES) & (d.is_defect == 1)]
    if len(planted) != 2:
        raise SystemExit(f"expected 2 planted defects, found {len(planted)}")
    rest = d[~d.id.isin(planted.id)]

    A, B = [planted.id.iloc[0]], [planted.id.iloc[1]]
    for is_def, dec, n in [(1, "human", 1), (1, "defect", 2), (0, "recover", 3), (0, "human", 3)]:
        pool = rest[(rest.is_defect == is_def) & (rest.decision == dec)]
        if len(pool) < 2 * n:
            raise SystemExit(f"not enough items for is_defect={is_def}, decision={dec}")
        a_ids, b_ids = pick_pairs(pool, n)
        A += a_ids
        B += b_ids

    # fixed shuffled order, the same for every participant
    order = lambda ids: list(pd.Series(ids).sample(frac=1, random_state=7))
    sets = {"A": order(A), "B": order(B)}
    json.dump(sets, open(R / "user_test_sets.json", "w"), indent=2)
    json.dump({i: "recover" for i in planted.id}, open(R / "user_test_planted.json", "w"), indent=2)

    for k, ids in sets.items():
        s = d.set_index("id").loc[ids]
        shown = s.decision.where(~s.index.isin(planted.id), "recover")
        print(k, "defects", int(s.is_defect.sum()), "| tool shown:", shown.value_counts().to_dict(),
              "| median score", round(float(s.score.median()), 2))


if __name__ == "__main__":
    main()
