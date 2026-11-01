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

from common import build_lane, family, main_rows, recovery_at, split_tag, t_recover

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


TYPE_MODELS = ("patchcore", "patchcore-n400", "rad", "rad-n300")   # the reference run and the best candidate


def defect_type_rows(scores: dict, standins: dict) -> list[dict]:
    """Per defect type on the aligned lane: how well it separates from goods, how many of it slip through at the
    1% and 5% thresholds (set on all lane defects, like the quick-numbers table), and the share of goods that
    score above its weakest example, i.e. what catching every one of this type at zero escape would cost."""
    rows = []
    for tag, df in scores.items():
        ds, cat, model = split_tag(tag)
        if model not in TYPE_MODELS:
            continue
        key = f"{ds}_{cat}_aligned"
        lane = df.loc[df.index.intersection(build_lane(standins[key])[0])] if key in standins else df
        y, s = lane.label.values.astype(int), lane.score.values
        good, bad = s[y == 0], s[y == 1]
        if len(good) == 0 or len(bad) == 0:
            continue
        thr = {e: t_recover(bad, e) for e in (0.01, 0.05)}
        types = pd.Series([Path(i).parent.name for i in lane.index], index=lane.index)
        for t, g in lane[lane.label == 1].groupby(types[lane.label == 1]):
            sc = g.score.values
            rows.append({"dataset": ds, "category": cat, "model": model, "defect_type": t, "n": len(sc),
                         "auroc_vs_good": roc_auc_score(np.r_[np.zeros(len(good)), np.ones(len(sc))], np.r_[good, sc]),
                         "escapes@0.01": int((sc < thr[0.01]).sum()), "escapes@0.05": int((sc < thr[0.05]).sum()),
                         "goods_above_weakest": float((good >= sc.min()).mean())})
    return rows


def defect_type_section(dt: pd.DataFrame) -> list[str]:
    if dt.empty:
        return []
    lines = ["", "## Escapes by defect type (aligned lane, PatchCore and Rad)", "",
             "n: defects of this type in the lane. AUROC: this type against the lane goods. escapes@e: how many of",
             "this type sit below the threshold that lets e of all defects through. goods above weakest: share of",
             "goods that score higher than the easiest-looking defect of this type, the review load for catching all of it.",
             "", "| dataset | category | model | defect type | n | AUROC | escapes@1% | escapes@5% | goods above weakest |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in dt.iterrows():
        lines.append(f"| {r.dataset} | {r.category} | {r.model} | {r.defect_type} | {r.n} | {fmt(r.auroc_vs_good)} | "
                     f"{r['escapes@0.01']} | {r['escapes@0.05']} | {fmt(r.goods_above_weakest)} |")
    return lines


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
    dt = pd.DataFrame(defect_type_rows(scores, standins))
    if not dt.empty:
        dt = dt.sort_values(["dataset", "category", "model", "defect_type"])
    dt.to_csv(res / "defect_types.csv", index=False)
    write_docs(out, frr_df, Path(a.docs), a.escape, res / "classify_summary.csv", dt)
    make_figures(out, frr_df, scores, standins, res / "figures")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def fmt(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v))


def write_docs(df: pd.DataFrame, frr_df: pd.DataFrame, path: Path, escape: float, classify_csv: Path,
               defect_df: pd.DataFrame) -> None:
    lines = [
        "# Results",
        "",
        "Generated by `scripts/lane.py` from the per-image scores in `results/`.",
        "",
        "recovery@e: share of good items passed when the threshold lets through a fraction e of defects. In the",
        "quick-numbers table that threshold is set on the same test defects, so use it to compare models only.",
        "`all`: every test good. `aligned` / `pca` / `tinypadim`: only the goods that stand-in camera",
        "rejected (the simulated reject lane). FRR: share of goods the stand-in rejected at its zero-miss cutoff.",
        "",
        "Band: two thresholds set on 4 of 5 folds, measured on the 5th. The lower line lets through at most",
        f"{escape:.0%} of the calibration defects. Under 100 calibration defects that is the weakest one, so a new defect",
        "slips under it about 1/(n+1) of the time: 1.2% on VisA (80 defects), 2% on bottle (50). On GoodsAD the line",
        "sits at the 2nd to 4th weakest of 117 to 340, 1.0 to 1.7% expected. The rule and its check are in evaluation/.",
        "Trained models (dinomaly, foundad, cflow, cfa, fre, fastflow, stfpm) ran with a 6-minute training cap.",
        "UniNet and SuperSimpleNet are left out: their image scores had not converged in that time.",
        "gemma4-*: Gemma 4 vision-language model (E2B, E4B), one image per call at 1120 image tokens, score = logP(defect) - logP(good)",
        "at the verdict token of a forced JSON answer. -zs: zero-shot (-ref1: one good training image in the prompt). -lora: one LoRA",
        "over all 15 categories, trained on train/good plus the test defects of the other half, 2-fold, scored out of fold.",
        "Unlike every other row the -lora models have seen real defects of the category, so compare them with that in mind.",
        "recovery = auto-passed goods, escape = auto-passed defects, review = sent to a human.",
        *glance_section(df, escape),
        "",
        "## Quick numbers: threshold set on the test defects themselves (compare models only)",
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
        "## Calibrated band on the lane (held-out folds, line at the weakest calibration defect)",
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
    lines += defect_type_section(defect_df)
    overlays = sorted((path.parent / "figures" / "overlays").glob("*.png"))
    if overlays:
        lines += ["", "## Review band overlays", "",
                  "Items between the two thresholds, the ones a technician would see: image, heatmap, closest good",
                  "training image, and every lane item's score against the thresholds. From `scripts/overlays.py`.", ""]
        lines += [f"- `docs/figures/overlays/{f.name}`" for f in overlays]
    lines += classifier_section(classify_csv, escape)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def glance_section(df: pd.DataFrame, escape: float) -> list[str]:
    """Means over categories per model. Only models that ran on every category of the group."""
    d = df.assign(fam=df.model.map(family))
    lines = ["", "## Models at a glance", "",
             "Means over categories, aligned lane. Candidate subset runs (-n300, -n400, -n450) count as their model.",
             "Only models that ran on every category of the group. In-sample: threshold on the test defects, compare only.",
             "Held-out: the calibrated band below, about 1.2% expected escape on VisA, 2% on bottle. The number to quote.", ""]
    groups = (("VisA and MVTec AD, 10 categories", d[d.dataset != "goodsad"]), ("GoodsAD, 5 categories", d[d.dataset == "goodsad"]))
    for title, part in groups:
        if part.empty:
            continue
        g = part.groupby("fam").agg(n=("category", "nunique"), auroc=("auroc", "mean"), lane=("aligned_rec@0.01", "mean"),
                                    cv_rec=("aligned_cv_recovery", "mean"), cv_esc=("aligned_cv_escape", "mean"),
                                    ms=("ms_per_image", "mean"))
        g = g[g.n == part.category.nunique()].sort_values("cv_rec", ascending=False)
        lines += [f"**{title}**", "", f"| model | AUROC | in-sample lane@{escape:.0%} | held-out recovery | held-out escape | ms/img |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {m} | {fmt(r.auroc)} | {fmt(r.lane)} | {fmt(r.cv_rec)} | {fmt(r.cv_esc)} | {fmt(r.ms)} |" for m, r in g.iterrows()]
        lines.append("")
    return lines


def frr_section(frr_df: pd.DataFrame, escape: float) -> list[str]:
    """PatchCore recovery on bins built by each stand-in at each primary reject rate. Does the bin builder matter?"""
    d = main_rows(frr_df, "patchcore")
    if d.empty:
        return []
    bins = ["zero-miss"] + [f"{f:.2f}" for f in FRRS]
    lines = [
        "",
        f"## PatchCore recovery by stand-in camera and primary reject rate (held-out, {escape:.0%} escape)",
        "",
        "Columns: share of goods the stand-in rejects. `zero-miss`: cutoff at its weakest defect, resulting",
        "reject rate in brackets. `template`: the unaligned rule camera. GoodsAD rows: PatchCore on 400 training images.",
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
    col = "aligned_cv_recovery" if "aligned_cv_recovery" in df else "all_cv_recovery"  # held-out

    # one dot per model and category, main models only, fixed colours
    main = [("padim", "#2a78d6"), ("patchcore", "#eb6834"), ("rd", "#1baf7a"),
            ("patchcore-320-n300", "#eda100"), ("patchcore-r18", "#e87ba4"),
            ("mhpatchcore", "#008300"), ("anomalydino", "#e34948"), ("rad", "#6250d6"), ("dinomaly", "#00a3a3"),
            ("gemma4-e4b-zs", "#8a8a8a"), ("gemma4-e4b-lora", "#000000")]
    fig, ax = plt.subplots(figsize=(6, 5))
    for model, color in main:
        g = main_rows(df, model)
        if g.empty:
            continue
        ax.scatter(g.auroc, g[col], label=model, s=36, color=color, edgecolor="#ffffff", linewidth=0.8)
    g = main_rows(df, "patchcore")
    for _, r in g.iterrows():
        ax.annotate(r.category, (r.auroc, r[col]), fontsize=6, color="#52514e", xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("image AUROC", fontsize=8)
    ax.set_ylabel("held-out recovery, " + ("simulated lane" if col.startswith("aligned") else "all goods"), fontsize=8)
    ax.set_title("Image AUROC vs held-out recovery, labels on PatchCore", fontsize=10, loc="left")
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
        shown = {m for m, _ in main}
        for tag, sdf in scores.items():
            tds, tcat, model = split_tag(tag)
            if (tds, tcat) != (ds, cat) or family(model) not in shown:
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
    fig.suptitle("Recovery vs escape, threshold set on the test defects (in-sample, compare models only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "recovery_curves.png", dpi=150)
    plt.close(fig)

    # recovery vs primary reject rate, PatchCore, one line per stand-in
    d = main_rows(frr_df[frr_df.bin != "zero-miss"], "patchcore")
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
        ax.set_ylabel("held-out recovery", fontsize=8)
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
