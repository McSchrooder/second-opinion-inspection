"""Review app for quality technicians, also used for the user tests.

Run:  streamlit run app.py -- --results results/bottle

Two modes, matching the test plan:
  - "Without tool": the technician sees only the rejected image and decides.
  - "With tool":    the technician also sees the tool's decision, score and heatmap.
Every decision is logged with the time it took (logs/<participant>.csv).
The ground truth is never shown on screen.
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
         "human": "Uncertain: your decision"}


@st.cache_data
def load(results: str):
    rows = list(csv.DictReader(open(Path(results) / "items.csv")))
    sets = json.load(open(Path(results) / "user_test_sets.json"))
    thr = json.load(open(Path(results) / "thresholds.json"))
    return {r["id"]: r for r in rows}, sets, thr


items, sets, thr = load(str(R))

st.set_page_config(page_title="Second Opinion – reject review", layout="wide")
st.title("Second opinion on rejected items")

with st.sidebar:
    participant = st.text_input("Participant ID", "P1")
    line_set = st.selectbox("Image set", list(sets))
    mode = st.radio("Mode", ["Without tool", "With tool"])
    if st.button("Start / restart this set"):
        st.session_state.update(i=0, t_start=time.time(), started=True)
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
st.progress(i / len(queue), text=f"Item {i + 1} of {len(queue)}")

left, right = st.columns(2)
left.image(str(R / "items" / f"{it['id']}.png"), caption="Rejected by the line camera", width="stretch")
if mode == "With tool":
    right.image(str(R / "items" / f"{it['id']}_heat.png"),
                caption="Where the model sees something unusual (red)", width="stretch")
    score = float(it["score"])
    st.markdown(f"**Tool says:** {LABEL[it['decision']]}")
    st.caption(f"Anomaly score {score:.2f}  ·  recover below {thr['t_low']:.2f}  ·  "
               f"confirm defect above {thr['t_high']:.2f}")

c1, c2 = st.columns(2)
choice = None
if c1.button("Good item: recover", width="stretch"):
    choice = "recover"
if c2.button("Real defect: keep reject", width="stretch"):
    choice = "defect"

if choice:
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    f = log_dir / f"{participant}.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "participant", "set", "mode", "item", "seconds",
                        "participant_decision", "tool_decision", "agreed_with_tool", "is_defect"])
        tool = it["decision"] if mode == "With tool" else ""
        agreed = "" if mode == "Without tool" or tool == "human" else int(tool == choice)
        w.writerow([datetime.now().isoformat(timespec="seconds"), participant, line_set, mode, it["id"],
                    round(time.time() - st.session_state["t_start"], 1), choice, tool, agreed, it["is_defect"]])
    st.session_state.update(i=i + 1, t_start=time.time())
    st.rerun()
