"""Does a second-stage classifier on heatmap statistics beat the raw score on the lane?

Trees on [score + heatmap stats + stand-in score], 5-fold on the lane, same recovery-at-escape
metric as the raw score. Also a separability check (silhouette, nearest-neighbour purity).
Result so far: no gain with 40-90 lane goods per category.

usage: python scripts/classify.py [--results results] [--escape 0.01] [--standin template]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from common import build_lane, split_tag, t_recover

FOLDS = 5
SEED = 0
FEATURES = ["score", "map_max", "map_mean", "map_std", "map_p90", "map_p99", "map_frac_hi", "map_cx", "map_cy"]


def cv_recovery(scores_fn, X: np.ndarray, y: np.ndarray, escape: float) -> tuple[float, float]:
    """scores_fn(X_cal, y_cal, X_ev) -> (cal_scores, ev_scores), higher = more defect-like."""
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    rec, esc = [], []
    for cal, ev in skf.split(X, y):
        s_cal, s_ev = scores_fn(X[cal], y[cal], X[ev])
        thr = t_recover(s_cal[y[cal] == 1], escape)
        good, bad = s_ev[y[ev] == 0], s_ev[y[ev] == 1]
        rec.append((good < thr).mean() if len(good) else np.nan)
        esc.append((bad < thr).mean() if len(bad) else np.nan)
    return float(np.nanmean(rec)), float(np.nanmean(esc))


def raw_score(X_cal, y_cal, X_ev):
    return X_cal[:, 0], X_ev[:, 0]


def gbt_score(X_cal, y_cal, X_ev):
    from sklearn.model_selection import cross_val_predict

    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=SEED)
    # threshold must come from out-of-fold probabilities, otherwise training fit makes it optimistic
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    p_cal = cross_val_predict(clf, X_cal, y_cal, cv=inner, method="predict_proba")[:, 1]
    clf.fit(X_cal, y_cal)
    return p_cal, clf.predict_proba(X_ev)[:, 1]


def separability(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    Z = StandardScaler().fit_transform(X)
    sil = silhouette_score(Z, y) if len(np.unique(y)) == 2 and len(y) > 3 else np.nan
    k = min(5, len(y) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    purity = (y[idx] == y[:, None]).mean()
    return float(sil), float(purity)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--escape", type=float, default=0.01)
    p.add_argument("--standin", default="aligned")
    a = p.parse_args()
    res = Path(a.results)

    standins = {f.stem: pd.read_csv(f).set_index("image") for f in (res / "standin").glob(f"*_{a.standin}.csv")}
    rows = []
    for f in sorted((res / "scores").glob("*.csv")):
        df = pd.read_csv(f).set_index("image")
        if not set(FEATURES).issubset(df.columns):
            continue
        ds, cat, model = split_tag(f.stem)
        key = f"{ds}_{cat}_{a.standin}"
        if key in standins:
            lane = df.loc[df.index.intersection(build_lane(standins[key])[0])].copy()
            lane["standin"] = standins[key].loc[lane.index, "score"].values
            feats = FEATURES + ["standin"]
        else:
            lane, feats = df.copy(), FEATURES
        y = lane.label.values.astype(int)
        X = lane[feats].values.astype(float)
        if (y == 0).sum() < FOLDS or (y == 1).sum() < FOLDS:
            continue
        r_raw, e_raw = cv_recovery(raw_score, X, y, a.escape)
        r_gbt, e_gbt = cv_recovery(gbt_score, X, y, a.escape)
        sil, pur = separability(X, y)
        rows.append({
            "dataset": ds, "category": cat, "model": model,
            "lane_good": int((y == 0).sum()), "lane_defect": int((y == 1).sum()),
            "raw_recovery": r_raw, "raw_escape": e_raw,
            "gbt_recovery": r_gbt, "gbt_escape": e_gbt,
            "silhouette": sil, "nn_purity": pur,
        })
    out = pd.DataFrame(rows).sort_values(["model", "dataset", "category"])
    out.to_csv(res / "classify_summary.csv", index=False)
    print(f"escape target {a.escape:.0%}, lane = {a.standin} stand-in, {FOLDS}-fold held-out")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
