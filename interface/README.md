# Second Opinion – interface

Review app for the user tests. **Study logic and log format are identical to `reference_prototype/app.py`**
(same items, sets, planted-error rule, timing and the same 12 log columns), so `analyze_user_tests.py` keeps working.

## Run
```
pip install -r requirements.txt
python make_placeholder_data.py          # only until the real results folder exists
streamlit run app.py                     # placeholder data
streamlit run app.py -- --results results/bottle   # real data (items.csv, user_test_sets.json, thresholds.json, items/ ...)
SO_DEMO=1 streamlit run app.py                     # demo version with the score (never for study sessions)
```

## Moderator panel (arrow at the top-left, collapsed by default)
| Setting | What it does |
|---|---|
| Participant ID, Image set, Tool | Same as the original sidebar. `Tool` is logged as `mode` ("Without tool" / "With tool"). |
| Mode: **Study** | Round 1. No score, no ring. |
| Mode: **Study + look-here ring** | Round 2. Yellow ring on the hottest point; logged as `mark_peak = 1`. |
| Mode: **Demo** | **Only exists if you start the app with `SO_DEMO=1`** (presentations, videos). Shows score and thresholds, "Demo mode" badge, logs go to `logs/demo/`. A normal study session can never show the score. |

## Differences from the original (all visual, except the last)
- New layout, IBM Plex Sans (needs internet on the moderator's laptop; falls back to the system font).
- "Start" button on the first screen; the sidebar "Start / restart this set" still works.
- Demo mode writes to `logs/demo/` so demos cannot pollute study data. 
## Logo
Drop `assets/logo.svg` (or `logo.png`) next to `app.py`; it replaces the "SO" placeholder.

## Placeholder data
`results/placeholder/` is synthetic (drawn rings, not bottles; invented scores and thresholds 0.60 / 1.10).

