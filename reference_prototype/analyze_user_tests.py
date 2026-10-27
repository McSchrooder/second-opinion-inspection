"""Analyse the user tests: review logs from app.py plus the questionnaire sheet.

Inputs
  logs/<participant>.csv          written by app.py (one row per reviewed image)
  User test results.xlsx          sheets "Sessions" and "Questionnaire" (template in 04_User_Tests)

Output
  A markdown summary for report Section 4, printed and saved next to the workbook.

Example:
    python analyze_user_tests.py --logs logs --sheet "User test results.xlsx" --round 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sus_score(row: pd.Series) -> float:
    """Standard SUS: odd items score-1, even items 5-score, sum x 2.5 (0 to 100)."""
    total = 0
    for k in range(1, 11):
        v = float(row[f"SUS{k}"])
        total += (v - 1) if k % 2 else (5 - v)
    return total * 2.5


def wilcoxon_p(diff: np.ndarray) -> float | None:
    diff = diff[~np.isnan(diff)]
    if len(diff) < 4:
        return None  # too few pairs to say anything
    from scipy.stats import wilcoxon
    return float(wilcoxon(diff).pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--round", type=int, default=None, help="only this round (from the Sessions sheet)")
    a = ap.parse_args()

    logs = pd.concat([pd.read_csv(f) for f in sorted(Path(a.logs).glob("*.csv"))], ignore_index=True)
    sessions = pd.read_excel(a.sheet, sheet_name="Sessions")
    quest = pd.read_excel(a.sheet, sheet_name="Questionnaire")
    if a.round is not None:
        keep = set(sessions.loc[sessions["Round"] == a.round, "Participant"])
        logs, quest = logs[logs.participant.isin(keep)], quest[quest.Participant.isin(keep)]

    logs["correct"] = np.where(logs.is_defect == 1, logs.participant_decision == "defect",
                               logs.participant_decision == "recover")
    logs["defect_recovered"] = (logs.is_defect == 1) & (logs.participant_decision == "recover")
    logs["good_recovered"] = (logs.is_defect == 0) & (logs.participant_decision == "recover")

    per = logs.groupby(["participant", "mode"]).agg(
        median_s=("seconds", "median"), accuracy=("correct", "mean"),
        defects_recovered=("defect_recovered", "sum"), goods_recovered=("good_recovered", "sum"),
        n=("item", "count")).unstack("mode")

    lines = [f"# User test summary{'' if a.round is None else f' – round {a.round}'}", "",
             f"Participants: {logs.participant.nunique()}, images reviewed: {len(logs)}", ""]

    # 1. speed and accuracy, paired per participant
    t_wo, t_w = per[("median_s", "Without tool")], per[("median_s", "With tool")]
    acc_wo, acc_w = per[("accuracy", "Without tool")], per[("accuracy", "With tool")]
    p = wilcoxon_p((t_wo - t_w).to_numpy(dtype=float))
    lines += ["## Speed and accuracy", "",
              "| | Without tool | With tool |", "|---|---|---|",
              f"| Median seconds per image (median over participants) | {t_wo.median():.1f} | {t_w.median():.1f} |",
              f"| Correct decisions (mean over participants) | {acc_wo.mean():.0%} | {acc_w.mean():.0%} |",
              f"| Real defects the participant recovered (total) | {int(per[('defects_recovered', 'Without tool')].sum())} | {int(per[('defects_recovered', 'With tool')].sum())} |",
              f"| Good items the participant recovered (total) | {int(per[('goods_recovered', 'Without tool')].sum())} | {int(per[('goods_recovered', 'With tool')].sum())} |",
              "",
              f"Participants faster with the tool: {int((t_w < t_wo).sum())} of {int(t_w.notna().sum())}. "
              + ("Too few participants for a significance test." if p is None
                 else f"Wilcoxon signed-rank on per-participant medians: p = {p:.3f}."), ""]

    # 2. agreement with the tool (only where the tool gave a call)
    w = logs[(logs["mode"] == "With tool") & logs.tool_decision.isin(["recover", "defect"])].copy()
    if "planted" in w:
        w = w[w.planted != 1]  # planted errors are reported separately below
    if len(w):
        w["agreed"] = w.agreed_with_tool.astype(int)
        lines += ["## Agreement with the tool", "",
                  "| Tool call | Items | Participant agreed | Override was right |", "|---|---|---|---|"]
        for call, g in w.groupby("tool_decision"):
            over = g[g.agreed == 0]
            right = int(over.correct.sum())
            lines.append(f"| {call} | {len(g)} | {g.agreed.mean():.0%} | {right} of {len(over)} |")
        lines.append("")

    # 2b. planted errors: did the participant catch a defect the tool called "likely good"?
    if "planted" in logs:
        pl = logs[(logs.planted == 1) & (logs["mode"] == "With tool")]
        if len(pl):
            lines += ["## Planted tool errors", "",
                      f"Real defects shown as 'likely good': {len(pl)}. "
                      f"Caught by the participant (kept as defect): {int((pl.participant_decision == 'defect').sum())}. "
                      f"Followed the tool (recovered): {int((pl.participant_decision == 'recover').sum())}.", ""]

    # 3. questionnaire
    if len(quest):
        quest = quest.copy()
        quest["SUS"] = quest.apply(sus_score, axis=1)
        lines += ["## Questionnaire", "",
                  f"SUS: median {quest.SUS.median():.0f} (range {quest.SUS.min():.0f} to {quest.SUS.max():.0f}); "
                  "68 is the commonly cited average.", "",
                  "| Trust item | Median (1 to 5) |", "|---|---|"]
        for k, txt in [("T1", "I understood why the tool made each decision"),
                       ("T2", "I would trust the tool's 'good' decision on my line"),
                       ("T3", "The heatmap helped me decide faster")]:
            lines.append(f"| {txt} | {quest[k].median():.1f} |")
        if "Autonomy" in quest:
            lines += ["", "Autonomy answer for their own line: "
                      + ", ".join(f"{k}: {v}" for k, v in quest.Autonomy.value_counts().items())]
        for col, txt in [("Heatmap place correct (0-3)", "pointed at the right place"),
                         ("Heatmap call correct (0-3)", "called defect or harmless correctly")]:
            if col in quest and quest[col].notna().any():
                v = quest[col].dropna()
                lines.append(f"Heatmap task, {txt}: median {v.median():.0f} of 3; "
                             f"{int((v >= 2).sum())} of {len(v)} participants got at least 2 of 3")
        lines.append("")

    # 4. order effect check (counterbalancing)
    if "First mode" in sessions:
        order = sessions.set_index("Participant")["First mode"]
        gain = (t_wo - t_w).rename("gain").to_frame().join(order)
        lines += ["## Order check", "",
                  "Median time saved per image, by which mode came first: "
                  + ", ".join(f"{k}: {v:.1f} s" for k, v in gain.groupby("First mode").gain.median().items()), ""]

    text = "\n".join(lines)
    print(text)
    out = Path(a.sheet).with_name(f"User test summary{'' if a.round is None else f' round {a.round}'}.md")
    out.write_text(text)


if __name__ == "__main__":
    main()
