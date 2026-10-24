"""Evaluate scores on the simulated reject lane and write docs/results.md.

Reads results/scores/*.csv (run.py) and results/standin/*.csv (standin.py). The lane is
what the stand-in camera flags. On it we report recovery (goods auto-passed), escape
(defects auto-passed), review (sent to a human) with two thresholds calibrated on 4 folds
and measured on the 5th, plus a 95% upper bound on escape.

usage: python scripts/lane.py [--results results] [--docs docs/results.md] [--escape 0.01]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import build_lane, recovery_at, split_tag, t_recover

ESCAPES = (0.0, 0.01, 0.05)
CONFIRM_GOOD_FPR = 0.02   # t_confirm: at most 2 % of calibration goods above it
FOLDS = 5
SEED = 0


def escape_upper_bound(n_defects: int, observed: int = 0, conf: float = 0.95) -> float:
    """Clopper-Pearson upper bound on the true escape rate given n calibration defects."""
    if n_defects == 0:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(conf, observed + 1, n_defects - observed))


def cv_band(scores: np.ndarray, labels: np.ndarray, escape: float) -> dict:
    """5-fold: calibrate t_recover / t_confirm on 4 folds, evaluate on the 5th."""
    n_def = int((labels == 1).sum())
    n_good = int((labels == 0).sum())
    if n_good < FOLDS or n_def < FOLDS:
        return {"recovery": np.nan, "escape": np.nan, "review": np.nan, "confirm_prec": np.nan}
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    rec, esc, rev, prec = [], [], [], []
    for cal, ev in skf.split(scores, labels):
        tr = t_recover(scores[cal][labels[cal] == 1], escape)
        tc = float(np.quantile(scores[cal][labels[cal] == 0], 1 - CONFIRM_GOOD_FPR))
        tc = max(tc, tr)
        s, y = scores[ev], labels[ev]
        good, bad = s[y == 0], s[y == 1]
        rec.append((good < tr).mean() if len(good) else np.nan)
        esc.append((bad < tr).mean() if len(bad) else np.nan)
        rev.append(((s >= tr) & (s < tc)).mean())
        conf = s >= tc
        prec.append((y[conf] == 1).mean() if conf.any() else np.nan)
    return {
        "recovery": float(np.nanmean(rec)), "escape": float(np.nanmean(esc)),
        "review": float(np.nanmean(rev)), "confirm_prec": float(np.nanmean(prec)),
    }


STANDINS = ("aligned", "pca", "tinypadim")
FRRS = (0.05, 0.10, 0.20)   # primary reject rates to simulate besides the zero-miss cutoff


def build_lane_frr(standin: pd.DataFrame, frr: float) -> pd.Index:
    """Bin when the primary rejects the `frr` most suspicious-looking goods (defects below that cutoff never reach us)."""
    cut = float(standin.loc[standin.label == 0, "score"].quantile(1 - frr))
    return standin[standin.score >= cut].index


def frr_rows(tag: str, df: pd.DataFrame, standins: dict, escape: float) -> list[dict]:
    """Held-out recovery on bins of increasing strictness, one row per stand-in and reject rate."""
    ds, cat, model = split_tag(tag)
    rows = []
    for sname in STANDINS + ("template",):
        key = f"{ds}_{cat}_{sname}"
        if key not in standins:
            continue
        zero_idx, zero_frr = build_lane(standins[key])
        bins = [("zero-miss", zero_frr, zero_idx)] + [(f"{f:.2f}", f, build_lane_frr(standins[key], f)) for f in FRRS]
        for name, frr, idx in bins:
            lane = df.loc[df.index.intersection(idx)]
            y, s = lane.label.values.astype(int), lane.score.values
            cv = cv_band(s, y, escape)
            rows.append({"dataset": ds, "category": cat, "model": model, "standin": sname, "bin": name, "frr": frr,
                         "lane_good": int((y == 0).sum()), "lane_defect": int((y == 1).sum()),
                         "recovery": cv["recovery"], "escape": cv["escape"]})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--docs", default="docs/results.md")
    p.add_argument("--escape", type=float, default=0.01, help="escape target for the band table")
    a = p.parse_args()
    res = Path(a.results)

    scores = {f.stem: pd.read_csv(f).set_index("image") for f in sorted((res / "scores").glob("*.csv"))}
    standins = {f.stem: pd.read_csv(f).set_index("image") for f in sorted((res / "standin").glob("*.csv"))}
    runs = {f.stem: json.loads(f.read_text()) for f in res.glob("*.json")}

    rows, frr_list = [], []
    for tag, df in scores.items():
        frr_list += frr_rows(tag, df, standins, a.escape)
        ds, cat, model = split_tag(tag)
        y, s = df.label.values.astype(int), df.score.values
        base = {
            "dataset": ds, "category": cat, "model": model,
            "n_good": int((y == 0).sum()), "n_defect": int((y == 1).sum()),
            "auroc": roc_auc_score(y, s),
            "ms_per_image": runs.get(tag, {}).get("ms_per_image"),
            "fit_s": runs.get(tag, {}).get("fit_s"),
        }
        for e in ESCAPES:
            base[f"all_rec@{e:g}"] = recovery_at(s, y, e)
        cv = cv_band(s, y, a.escape)
        base.update({f"all_cv_{k}": v for k, v in cv.items()})
        base["escape_bound_95"] = escape_upper_bound(int(0.8 * (y == 1).sum()))

        for sname in STANDINS:
            key = f"{ds}_{cat}_{sname}"
            if key not in standins:
                continue
            lane_idx, frr = build_lane(standins[key])
            lane = df.loc[df.index.intersection(lane_idx)]
            ly, ls = lane.label.values.astype(int), lane.score.values
            base[f"{sname}_frr"] = frr
            base[f"{sname}_lane_good"] = int((ly == 0).sum())
            for e in ESCAPES:
                base[f"{sname}_rec@{e:g}"] = recovery_at(ls, ly, e)
            cv = cv_band(ls, ly, a.escape)
            base.update({f"{sname}_cv_{k}": v for k, v in cv.items()})
        rows.append(base)

    out = pd.DataFrame(rows).sort_values(["dataset", "category", "model"])
    out.to_csv(res / "lane_summary.csv", index=False)
    frr_df = pd.DataFrame(frr_list).sort_values(["dataset", "category", "model", "standin", "frr"])
    frr_df.to_csv(res / "lane_frr.csv", index=False)
    write_docs(out, frr_df, Path(a.docs), a.escape, res / "classify_summary.csv")
    make_figures(out, frr_df, scores, standins, res / "figures")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def fmt(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v))


def write_docs(df: pd.DataFrame, frr_df: pd.DataFrame, path: Path, escape: float, classify_csv: Path) -> None:
    lines = [
        "# Results",
        "",
        "Generated by `scripts/lane.py` from the per-image scores in `results/`.",
        "",
        "recovery@e: share of good items passed when the threshold lets through a fraction e of defects.",
        "`all`: every test good. `aligned` / `pca` / `tinypadim`: only the goods that stand-in camera",
        "rejected (the simulated reject lane). FRR: share of goods the stand-in rejected at its zero-miss cutoff.",
        "",
        f"Band: two thresholds set on 4 of 5 folds, measured on the 5th, escape target {escape:.0%}.",
        "recovery = auto-passed goods, escape = auto-passed defects, review = sent to a human.",
        "",
        "## Recovery at fixed escape, all goods vs simulated lane (zero-miss cutoff)",
        "",
        "| dataset | category | model | good/defect | AUROC | all@0 | all@1% | all@5% | "
        + " | ".join(f"{s} FRR | {s} lane@1%" for s in STANDINS) + " | ms/img |",
        "|---|---|---|---|---|---|---|---|" + "---|---|" * len(STANDINS) + "---|",
    ]
    for _, r in df.iterrows():
        cells = [r.dataset, r.category, r.model, f"{r.n_good}/{r.n_defect}", fmt(r.auroc),
                 fmt(r["all_rec@0"]), fmt(r["all_rec@0.01"]), fmt(r["all_rec@0.05"])]
        for s in STANDINS:
            cells += [fmt(r.get(f"{s}_frr")), fmt(r.get(f"{s}_rec@0.01"))]
        cells.append(fmt(r.ms_per_image))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"## Calibrated band on the lane (held-out folds, escape target {escape:.0%})",
        "",
        "| dataset | category | model | lane | lane goods | recovery | escape | review | confirm precision | 95% escape bound |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        for sname in STANDINS:
            if f"{sname}_cv_recovery" not in r or np.isnan(r[f"{sname}_cv_recovery"]):
                continue
            lines.append("| " + " | ".join([
                r.dataset, r.category, r.model, sname, str(int(r[f"{sname}_lane_good"])),
                fmt(r[f"{sname}_cv_recovery"]), fmt(r[f"{sname}_cv_escape"]), fmt(r[f"{sname}_cv_review"]),
                fmt(r[f"{sname}_cv_confirm_prec"]), fmt(r.escape_bound_95),
            ]) + " |")
    lines += [
        "",
        "95% escape bound: Clopper-Pearson upper limit with zero escapes on the calibration defects (80% of test defects).",
        "",
        "Figures: `docs/figures/auroc_vs_recovery.png`, `docs/figures/recovery_curves.png`, `docs/figures/recovery_vs_frr.png`.",
    ]
    lines += frr_section(frr_df, escape)
    lines += classifier_section(classify_csv, escape)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def frr_section(frr_df: pd.DataFrame, escape: float) -> list[str]:
    """PatchCore recovery on bins built by each stand-in at each primary reject rate. Does the bin builder matter?"""
    d = frr_df[frr_df.model == "patchcore"]
    if d.empty:
        return []
    bins = ["zero-miss"] + [f"{f:.2f}" for f in FRRS]
    lines = [
        "",
        f"## PatchCore recovery by stand-in camera and primary reject rate (held-out, {escape:.0%} escape)",
        "",
        "Columns: share of goods the stand-in rejects. `zero-miss`: cutoff at its weakest defect, resulting",
        "reject rate in brackets. `template`: the unaligned rule camera.",
        "",
        "The stand-in dilemma: a dumb primary rejects goods for reasons our model does not share, so the",
        "bin is easy and recovery looks good. A smart primary rejects the goods our model also doubts, so",
        "the bin is hard and small. Neither is the real line. We report both ends.",
        "",
        "| dataset | category | stand-in | " + " | ".join(bins) + " |",
        "|---|---|---|" + "---|" * len(bins),
    ]
    for (ds, cat), g in d.groupby(["dataset", "category"], sort=False):
        for sname in STANDINS + ("template",):
            gs = g[g.standin == sname].set_index("bin")
            if gs.empty:
                continue
            cells = []
            for b in bins:
                if b not in gs.index:
                    cells.append("")
                    continue
                r = gs.loc[b]
                v = fmt(r.recovery) + (f" ({r.frr:.2f})" if b == "zero-miss" else "")
                cells.append(v + f" n={int(r.lane_good)}")
            lines.append("| " + " | ".join([ds, cat, sname] + cells) + " |")
    lines += ["", "n: goods in the bin. Blank: fewer than 5 goods or 5 defects in the bin."]
    return lines


def classifier_section(csv: Path, escape: float) -> list[str]:
    """Append the second-stage comparison written by classify.py, if it exists."""
    if not csv.exists():
        return []
    c = pd.read_csv(csv)
    lines = [
        "",
        f"## Second stage: trees on anomaly-map features vs raw score (lane, {escape:.0%} escape, held-out folds)",
        "",
        "| category | model | lane goods | raw recovery | raw escape | trees recovery | trees escape | silhouette | nn purity |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in c.iterrows():
        lines.append("| " + " | ".join([
            r.category, r.model, str(int(r.lane_good)), fmt(r.raw_recovery), fmt(r.raw_escape),
            fmt(r.gbt_recovery), fmt(r.gbt_escape), fmt(r.silhouette), fmt(r.nn_purity),
        ]) + " |")
    lines += [
        "",
        "With 40 to 90 lane goods per category the trees do not beat the raw score.",
    ]
    return lines


def make_figures(df: pd.DataFrame, frr_df: pd.DataFrame, scores: dict, standins: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    col = "aligned_rec@0.01" if "aligned_rec@0.01" in df else "all_rec@0.01"

    # one dot per model and category, main models only, fixed colours
    main = [("padim", "#2a78d6"), ("patchcore", "#eb6834"), ("rd", "#1baf7a"),
            ("patchcore-320-n300", "#eda100"), ("patchcore-r18", "#e87ba4")]
    fig, ax = plt.subplots(figsize=(6, 5))
    for model, color in main:
        g = df[df.model == model]
        if g.empty:
            continue
        ax.scatter(g.auroc, g[col], label=model, s=36, color=color, edgecolor="#ffffff", linewidth=0.8)
    g = df[df.model == "patchcore"]
    for _, r in g.iterrows():
        ax.annotate(r.category, (r.auroc, r[col]), fontsize=6, color="#52514e", xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("image AUROC", fontsize=8)
    ax.set_ylabel("recovery at 1% escape, " + ("simulated lane" if col.startswith("aligned") else "all goods"), fontsize=8)
    ax.set_title("Image AUROC vs recovery at 1% escape, labels on PatchCore", fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "auroc_vs_recovery.png", dpi=150)
    plt.close(fig)

    cats = sorted(df[["dataset", "category"]].drop_duplicates().itertuples(index=False))
    n = len(cats)
    if n == 0:
        return
    cols = min(4, n)
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3.2 * cols, 2.8 * rws), squeeze=False)
    for ax, (ds, cat) in zip(axes.flat, cats):
        key = f"{ds}_{cat}_aligned"
        lane_idx = build_lane(standins[key])[0] if key in standins else None
        for tag, sdf in scores.items():
            tds, tcat, model = split_tag(tag)
            if (tds, tcat) != (ds, cat):
                continue
            d = sdf.loc[sdf.index.intersection(lane_idx)] if lane_idx is not None else sdf
            y, s = d.label.values.astype(int), d.score.values
            if (y == 0).sum() == 0:
                continue
            es = np.linspace(0, 0.1, 41)
            ax.plot(es * 100, [recovery_at(s, y, e) for e in es], label=model)
        ax.set_title(f"{ds} {cat}" + (" (lane)" if lane_idx is not None else ""), fontsize=9)
        ax.set_xlabel("escape %", fontsize=8)
        ax.set_ylabel("recovery", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "recovery_curves.png", dpi=150)
    plt.close(fig)

    # recovery vs primary reject rate, PatchCore, one line per stand-in
    d = frr_df[(frr_df.model == "patchcore") & (frr_df.bin != "zero-miss")]
    if d.empty:
        return
    colors = {"aligned": "#2a78d6", "pca": "#eb6834", "tinypadim": "#1baf7a", "template": "#8a8a8a"}
    fig, axes = plt.subplots(rws, cols, figsize=(3.2 * cols, 2.8 * rws), squeeze=False)
    for ax, (ds, cat) in zip(axes.flat, cats):
        g = d[(d.dataset == ds) & (d.category == cat)]
        for sname, gs in g.groupby("standin"):
            gs = gs.sort_values("frr")
            ax.plot(gs.frr * 100, gs.recovery, marker="o", ms=4, lw=1.5, color=colors.get(sname), label=sname)
        ax.set_title(f"{ds} {cat}", fontsize=9)
        ax.set_xlabel("primary reject rate %", fontsize=8)
        ax.set_ylabel("recovery at 1% escape", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([5, 10, 20])
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("PatchCore recovery by stand-in camera and primary reject rate", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "recovery_vs_frr.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
