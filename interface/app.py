"""Second Opinion – reject review (interface version).

Restyled version of reference_prototype/app.py. STUDY LOGIC AND LOG FORMAT ARE UNCHANGED:
same items, same sets, same planted-error rule, same timing, same 12 log columns.

Run:   streamlit run app.py -- --results results/bottle
       (or, with placeholder data:  streamlit run app.py)

Moderator panel (sidebar, collapsed by default):
  Tool  : "Without tool" / "With tool"      -> logged as `mode`   (same strings as the original)
  Mode  : Study                             -> score hidden, no ring            (round 1)
          Study + look-here ring            -> ring on hottest point            (round 2) -> logged as mark_peak=1
          Demo                              -> only available if started with SO_DEMO=1; shows score; logs to logs/demo/
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.environ.get("SO_RESULTS", str(HERE / "results" / "placeholder")))
args, _ = ap.parse_known_args(sys.argv[1:])
R = Path(args.results)

LABEL = {"recover": "Likely good: propose to recover",
         "defect": "Defect confirmed: keep the reject",
         "human": "Uncertain: your decision"}
ICON = {"recover": "✓", "defect": "✕", "human": "?"}   # neutral glyphs, same weight for all three
# Demo mode (shows the score) exists ONLY when started with SO_DEMO=1, so a study session can never show it.
MODES = ["Study", "Study + look-here ring"] + (["Demo"] if os.environ.get("SO_DEMO") == "1" else [])
IMG_W = 400   # source images are 224 px: keep them near 2x so they stay sharp


@st.cache_data
def load(results: str):
    rows = list(csv.DictReader(open(Path(results) / "items.csv")))
    sets = json.load(open(Path(results) / "user_test_sets.json"))
    thr = json.load(open(Path(results) / "thresholds.json"))
    p = Path(results) / "user_test_planted.json"
    planted = json.load(open(p)) if p.exists() else {}
    return {r["id"]: r for r in rows}, sets, thr, planted


def logo_html() -> str:
    for name, mime in [("logo.svg", "image/svg+xml"), ("logo.png", "image/png")]:
        f = HERE / "assets" / name
        if f.exists():
            b64 = base64.b64encode(f.read_bytes()).decode()
            return f'<img class="so-logo" src="data:{mime};base64,{b64}" alt="Second Opinion">'
    return '<span class="so-mono">SO</span>'


items, sets, thr, planted = load(str(R))

st.set_page_config(page_title="Second Opinion – reject review", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --bg:#F3F2EF; --surface:#FFFFFF; --line:#D5D3CE; --ink:#1B1F23; --ink2:#5A6169; --accent:#35506B; --press:#E9E7E2; }
html, body, [class*="css"], .stApp, button, input, textarea { font-family:'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }
.stApp { background:var(--bg); color:var(--ink); }
#MainMenu, footer, .stAppDeployButton { display:none !important; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding:1rem 2.5rem 1.5rem; max-width:1280px; }

.so-head { display:flex; align-items:center; gap:24px; background:var(--surface); border:1px solid var(--line);
           border-radius:6px; padding:10px 20px; min-height:64px; margin-bottom:20px; }
.so-brand { display:flex; align-items:center; gap:12px; font-weight:600; font-size:20px; }
.so-logo { height:40px; }
.so-mono { border:1.5px solid var(--ink); border-radius:6px; padding:4px 8px; font-weight:600; font-size:16px; }
.so-prog { flex:1; display:flex; flex-direction:column; align-items:center; gap:6px; }
.so-count { font-variant-numeric:tabular-nums; font-weight:500; font-size:18px; }
.so-bar { width:min(420px,100%); height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
.so-bar > div { height:100%; background:var(--accent); }
.so-badge { background:#E4E9EE; color:var(--accent); border-radius:999px; padding:4px 14px; font-weight:600; font-size:15px; }

div[data-testid="stImage"] img { border:1px solid var(--line); border-radius:6px; background:var(--surface); padding:8px; }
.so-cap { color:var(--ink2); font-size:16px; margin-top:8px; }
.so-off { height:416px; width:400px; display:flex; align-items:center; justify-content:center;
          border:1px dashed var(--line); border-radius:6px; color:var(--ink2); font-size:16px; }

.so-tool { background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--accent);
           border-radius:6px; padding:16px 24px; display:flex; align-items:center; gap:20px; margin-top:16px; }
.so-ico { width:44px; height:44px; border-radius:50%; background:var(--press); display:flex; align-items:center;
          justify-content:center; font-size:22px; font-weight:600; flex:none; }
.so-lab { font-size:13px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink2); font-weight:600; }
.so-ans { font-size:28px; font-weight:500; line-height:1.2; }
.so-score { font-family:'IBM Plex Mono', monospace; font-size:15px; color:var(--ink2); margin-top:6px; }
.so-decide { margin-top:20px; }

.so-center { text-align:center; padding:72px 0 24px; }
.so-center h1 { font-size:44px; font-weight:600; margin:0 0 12px; }
.so-center p { font-size:20px; color:var(--ink2); }

/* decision buttons: identical in size, weight and style */
div[class*="st-key-btn_"] button, div.st-key-start_main button {
    min-height:80px; border-radius:6px; }
div[class*="st-key-btn_"] button p, div.st-key-start_main button p { font-size:20px !important; font-weight:600 !important; }
div.st-key-start_main { display:flex; justify-content:center; }
div[class*="st-key-btn_"] button { background:var(--surface); color:var(--ink); border:2px solid var(--ink); }
div[class*="st-key-btn_"] button:hover { background:var(--press); border-color:var(--ink); color:var(--ink); }
div[class*="st-key-btn_"] button:focus-visible, div.st-key-start_main button:focus-visible { outline:3px solid var(--accent); outline-offset:2px; }
div.st-key-start_main button { background:var(--accent); color:#fff; border:2px solid var(--accent); min-width:280px; }
div.st-key-start_main button:hover { background:#263B50; border-color:#263B50; color:#fff; }
section[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

# ---------------- moderator panel (hidden from the participant by default) ----------------
with st.sidebar:
    st.markdown("### Moderator panel")
    participant = st.text_input("Participant ID", "P1")
    line_set = st.radio("Image set", list(sets), horizontal=True)
    tool_mode = st.radio("Tool", ["Without tool", "With tool"], horizontal=True)
    app_mode = st.radio("Mode", MODES, help="Study = standard test. Study + look-here ring = round 2 option. "
                        "Demo (only when started with SO_DEMO=1) shows the score and logs to logs/demo/.")
    if st.button("Start / restart this set", key="start_side"):
        st.session_state.update(i=0, t_start=time.time(), started=True)
    st.caption("Ground truth is never shown. Decisions are saved to logs/ "
               "(Demo mode saves to logs/demo/ so it never mixes with study data).")

mark_peak = app_mode == "Study + look-here ring"
demo = app_mode == "Demo"
mode = tool_mode

queue = sets[line_set]
i = st.session_state.get("i", 0)


def header(progress: float | None, label: str = "") -> None:
    badge = '<span class="so-badge">Demo mode</span>' if demo else ""
    prog = (f'<div class="so-prog"><span class="so-count">{label}</span>'
            f'<div class="so-bar"><div style="width:{progress * 100:.0f}%"></div></div></div>'
            if progress is not None else '<div class="so-prog"></div>')
    st.markdown(f'<div class="so-head"><div class="so-brand">{logo_html()}'
                f'{"" if (HERE / "assets" / "logo.svg").exists() or (HERE / "assets" / "logo.png").exists() else "Second Opinion"}'
                f'</div>{prog}{badge}</div>', unsafe_allow_html=True)


# ---------------- start screen ----------------
if not st.session_state.get("started"):
    header(None)
    st.markdown('<div class="so-center"><h1>Ready when you are.</h1>'
                '<p>Review each rejected item and make the final decision.</p></div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Start", key="start_main"):
            st.session_state.update(i=0, t_start=time.time(), started=True)
            st.rerun()
    st.stop()

# ---------------- completion screen ----------------
if i >= len(queue):
    header(1.0, f"Item {len(queue)} of {len(queue)}")
    st.markdown(f'<div class="so-center"><h1>Set {line_set} finished. Thank you!</h1>'
                '<p style="font-size:16px">Moderator: switch the tool setting and start the other set.</p></div>',
                unsafe_allow_html=True)
    st.stop()

# ---------------- inspection screen ----------------
it = items[queue[i]]
shown = planted.get(it["id"], it["decision"])      # the answer the participant sees (planted errors override)
header(i / len(queue), f"Item {i + 1} of {len(queue)}")

left, right = st.columns(2, gap="large")
with left:
    st.image(str(R / "items" / f"{it['id']}.png"), width=IMG_W)
    st.markdown('<div class="so-cap">Rejected by the line camera</div>', unsafe_allow_html=True)
with right:
    if mode == "With tool":
        heat_img = str(R / "items" / f"{it['id']}_heat.png")
        if mark_peak and it.get("peak_x"):
            from PIL import Image, ImageDraw
            im = Image.open(heat_img).convert("RGB")
            d = ImageDraw.Draw(im)
            x, y, r = float(it["peak_x"]) * im.width, float(it["peak_y"]) * im.height, im.width * 0.06
            d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 210, 0), width=max(2, im.width // 100))
            heat_img = im
        st.image(heat_img, width=IMG_W)
        cap = ("Where the model sees something unusual (red; yellow ring = look here)" if mark_peak
               else "Where the model sees something unusual (red)")
        st.markdown(f'<div class="so-cap">{cap}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="so-off">Tool switched off</div>', unsafe_allow_html=True)

if mode == "With tool":
    score_line = ""
    if demo:
        score_line = (f'<div class="so-score">Anomaly score {float(it["score"]):.2f} · recover below '
                      f'{thr["t_low"]:.2f} · confirm defect above {thr["t_high"]:.2f}<br>'
                      'The score is a model measure, not a probability of a defect.</div>')
    st.markdown(f'<div class="so-tool"><div class="so-ico">{ICON[shown]}</div><div>'
                f'<div class="so-lab">Tool suggestion (advisory only)</div>'
                f'<div class="so-ans">{LABEL[shown]}</div>{score_line}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="so-decide so-lab">Your decision</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")
choice = None
if c1.button("Good item: recover", key="btn_recover", width="stretch"):
    choice = "recover"
if c2.button("Real defect: keep reject", key="btn_defect", width="stretch"):
    choice = "defect"

# ---------------- logging: identical columns and semantics to the original app ----------------
if choice:
    log_dir = Path("logs") / ("demo" if demo else "")
    log_dir.mkdir(parents=True, exist_ok=True)
    f = log_dir / f"{participant}.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "participant", "set", "mode", "item", "seconds",
                        "participant_decision", "tool_decision", "agreed_with_tool", "is_defect", "planted", "mark_peak"])
        tool = shown if mode == "With tool" else ""
        agreed = "" if mode == "Without tool" or tool == "human" else int(tool == choice)
        w.writerow([datetime.now().isoformat(timespec="seconds"), participant, line_set, mode, it["id"],
                    round(time.time() - st.session_state["t_start"], 1), choice, tool, agreed, it["is_defect"],
                    int(it["id"] in planted), int(mark_peak and mode == "With tool")])
    st.session_state.update(i=i + 1, t_start=time.time())
    st.rerun()
