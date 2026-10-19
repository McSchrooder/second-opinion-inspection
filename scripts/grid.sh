#!/bin/bash
# Overnight grid: stand-ins + models x categories. Skips runs whose JSON exists.
# usage: bash scripts/grid.sh [DATA_ROOT] [RESULTS_DIR]
set -u
ROOT="${1:-data}"
OUT="${2:-results}"
PY="${PYTHON:-python}"
LOG="$OUT/grid.log"
mkdir -p "$OUT"

VISA="candle cashew chewinggum fryum pipe_fryum macaroni1 macaroni2 capsules"
MVTEC="bottle hazelnut"

run() { # dataset category model [extra args]
  local tag="$1_$2_$3"
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag" | tee -a "$LOG"; return; fi
  echo "$(date +%H:%M:%S) start $tag ${*:4}" | tee -a "$LOG"
  timeout -s KILL "${RUN_TIMEOUT:-5400}" "$PY" scripts/run.py "$1" "$2" "$3" --root "$ROOT" --out "$OUT" "${@:4}" > "$OUT/logs/$tag.log" 2>&1 \
    && echo "$(date +%H:%M:%S) done  $tag $(grep -o '"recovery@0.01": [0-9.]*' "$OUT/$tag.json")" | tee -a "$LOG" \
    || echo "$(date +%H:%M:%S) FAIL  $tag (see $OUT/logs/$tag.log)" | tee -a "$LOG"
}
standin() { # dataset category
  local tag="$1_$2"
  if [ -f "$OUT/standin/${tag}_pca.csv" ]; then echo "skip standin $tag" | tee -a "$LOG"; return; fi
  echo "$(date +%H:%M:%S) standin $tag" | tee -a "$LOG"
  timeout -s KILL 1800 "$PY" scripts/standin.py "$1" "$2" --root "$ROOT" --out "$OUT" > "$OUT/logs/standin_$tag.log" 2>&1 \
    || echo "$(date +%H:%M:%S) FAIL  standin $tag" | tee -a "$LOG"
}
mkdir -p "$OUT/logs"

# pass 0: stand-ins (fast)
for c in $MVTEC; do standin mvtec "$c"; done
for c in $VISA;  do standin visa  "$c"; done

# pass 1: fast models everywhere
for c in $MVTEC; do run mvtec "$c" padim; run mvtec "$c" patchcore; done
for c in $VISA;  do run visa  "$c" padim; run visa  "$c" patchcore; done

# pass 2: reverse distillation, capped epochs (ROCm/WSL needs MIOPEN_FIND_MODE=2 for the backward conv)
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
for c in $MVTEC; do run mvtec "$c" rd --epochs 10; done
for c in $VISA;  do run visa  "$c" rd --epochs 10; done

# pass 3: efficientad, capped steps
for c in $MVTEC; do run mvtec "$c" efficientad --steps 5000; done
for c in $VISA;  do run visa  "$c" efficientad --steps 5000; done

echo "$(date +%H:%M:%S) grid finished" | tee -a "$LOG"
