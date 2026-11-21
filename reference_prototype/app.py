"""Review app for quality technicians, also used for the user tests.

Run:  streamlit run app.py -- --results results/bottle

Two modes, matching the test plan:
  - "Without tool": the technician sees only the rejected image and decides.
  - "With tool":    the technician also sees the tool's decision, score and heatmap.
Every decision is logged with the time it took (logs/<participant>.csv).
The ground truth is never shown on screen.
If user_test_planted.json exists (make_user_test_sets.py), those items show the tool answer
"Likely good" although they are real defects: a planted error to measure over-reliance.
Participants are told about it in the debrief.

"Decide first" (round 3, off by default): with the tool on, the heatmap is shown but the tool's
answer stays hidden until the technician has decided. Then both answers are shown. If they match,
the decision stands. If they differ, or the tool says "Uncertain", the item goes to a second check
and stays rejected for now. The log keeps the technician's own first decision (first_decision) and the
effective outcome (participant_decision: "recover" only when both said recover), so
analyze_user_tests.py measures planted errors followed under the new flow without changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/synth")
args, _ = ap.parse_known_args(sys.argv[1:])
R = Path(args.results)

LABEL = {"recover": "Likely good: propose to recover",
         "defect": "Defect confirmed: keep the reject",
         "human": "Uncertain: check the marked area before recovering"}


@st.cache_data
def load(results: str):
    rows = list(csv.DictReader(open(Path(results) / "items.csv")))
    sets = json.load(open(Path(results) / "user_test_sets.json"))
    thr = json.load(open(Path(results) / "thresholds.json"))
    p = Path(results) / "user_test_planted.json"
    planted = json.load(open(p)) if p.exists() else {}
    return {r["id"]: r for r in rows}, sets, thr, planted


items, sets, thr, planted = load(str(R))

st.set_page_config(page_title="Second Opinion – reject review", layout="wide")
st.title("Second opinion on rejected items")

with st.sidebar:
    participant = st.text_input("Participant ID", "P1")
    line_set = st.selectbox("Image set", list(sets))
    mode = st.radio("Mode", ["Without tool", "With tool"])
    mark_peak = st.checkbox("Mark where to look", value=True,
                            help="Draws a ring at the hottest point of the heatmap, even when the score is low. "
                                 "Off in round 1; on from round 2, after two of three participants followed the planted error.")
    decide_first = st.checkbox("Decide first (round 3)", value=False,
                               help="Hide the tool's answer until the technician has decided; "
                                    "a difference sends the item to a second check.")
    show_score = st.checkbox("Show score and thresholds", value=False,
                             help="Off in the user tests: with a planted error the score would contradict "
                                  "the answer shown. Useful for demos and debugging.")
    if st.button("Start / restart this set"):
        st.session_state.update(i=0, t_start=time.time(), started=True, first=None)
    st.caption("Ground truth is never shown. Decisions are saved to logs/.")

if not st.session_state.get("started"):
    st.info("Enter the participant ID, choose the set and mode, then press Start.")
    st.stop()

queue = sets[line_set]
i = st.session_state.get("i", 0)
if i >= len(queue):
    st.success(f"Set {line_set} finished. Thank you!")
    st.stop()

it = items[queue[i]]
shown = planted.get(it["id"], it["decision"])  # the answer the participant sees
st.progress(i / len(queue), text=f"Item {i + 1} of {len(queue)}")

left, right = st.columns(2)
left.image(str(R / "items" / f"{it['id']}.png"), caption="Rejected by the line camera", width="stretch")
if mode == "With tool":
    heat_img = str(R / "items" / f"{it['id']}_heat.png")
    if mark_peak and it.get("peak_x"):
        from PIL import Image, ImageDraw
        im = Image.open(heat_img).convert("RGB")
        d = ImageDraw.Draw(im)
        x, y, r = float(it["peak_x"]) * im.width, float(it["peak_y"]) * im.height, im.width * 0.06
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 210, 0), width=max(2, im.width // 100))
        heat_img = im
    right.image(heat_img, caption="Where the model sees something unusual (red; yellow ring = look here)"
                if mark_peak else "Where the model sees something unusual (red)", width="stretch")
    score = float(it["score"])
    hide_answer = decide_first and st.session_state.get("first") is None
    if hide_answer:
        st.info("**Tool's answer (hidden):** shown after you decide")
    elif not decide_first:
        st.markdown(f"**Tool says:** {LABEL[shown]}")
    if show_score:
        st.caption(f"Anomaly score {score:.2f}  ·  recover below {thr['t_low']:.2f}  ·  "
                   f"confirm defect above {thr['t_high']:.2f}")

YOURS = {"recover": "Good item: recover", "defect": "Real defect: keep reject"}
choice = None
first = None
if mode == "With tool" and decide_first and st.session_state.get("first") is not None:
    first = st.session_state["first"]
    agree = shown != "human" and shown == first
    a, b = st.columns(2)
    a.markdown(f"**Your decision:** {YOURS[first]}")
    b.markdown(f"**Tool's answer:** {LABEL[shown]}")
    if agree:
        st.success("Same answer: your decision stands.")
        if st.button("Next item", width="stretch"):
            choice = first
    else:
        st.warning("The tool is uncertain: this item goes to a second check and stays rejected for now."
                   if shown == "human" else
                   "Different answers: this item goes to a second check and stays rejected for now.")
        if st.button("Send to second check", width="stretch"):
            choice = "second_check"
else:
    c1, c2 = st.columns(2)
    if c1.button("Good item: recover", width="stretch"):
        choice = "recover"
    if c2.button("Real defect: keep reject", width="stretch"):
        choice = "defect"
    if choice and mode == "With tool" and decide_first:
        st.session_state["first"] = choice
        st.rerun()

if choice:
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    f = log_dir / f"{participant}.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "participant", "set", "mode", "item", "seconds",
                        "participant_decision", "tool_decision", "agreed_with_tool", "is_defect", "planted", "mark_peak",
                        "decide_first", "first_decision"])
        tool = shown if mode == "With tool" else ""
        df = int(mode == "With tool" and decide_first)
        if df:
            first_decision = first
            choice = "defect" if choice == "second_check" else choice  # second check: stays rejected for now
            agreed = "" if tool == "human" else int(tool == first_decision)
        else:
            first_decision = ""
            agreed = "" if mode == "Without tool" or tool == "human" else int(tool == choice)
        w.writerow([datetime.now().isoformat(timespec="seconds"), participant, line_set, mode, it["id"],
                    round(time.time() - st.session_state["t_start"], 1), choice, tool, agreed, it["is_defect"],
                    int(it["id"] in planted), int(mark_peak and mode == "With tool"), df, first_decision])
    st.session_state.update(i=i + 1, t_start=time.time(), first=None)
    st.rerun()
