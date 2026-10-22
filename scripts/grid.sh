#!/bin/bash
# Overnight grid: stand-ins + models x categories. Skips runs whose JSON exists.
# usage: bash scripts/grid.sh [DATA_ROOT] [RESULTS_DIR]
#
# Safety: before every run a 60 s GPU probe (small matmul) must pass. If it fails
# the grid STOPS instead of launching the next job against a wedged GPU; repeated
# launches after a driver reset is what took the whole machine down twice.
# Each run is also capped in VRAM (run.py --mem-frac) and wall time (RUN_TIMEOUT).
set -u
ROOT="${1:-data}"
OUT="${2:-results}"
PY="${PYTHON:-python}"
LOG="$OUT/grid.log"
mkdir -p "$OUT" "$OUT/logs"

VISA="candle cashew chewinggum fryum pipe_fryum macaroni1 macaroni2 capsules"
MVTEC="bottle hazelnut"
WEAK="candle cashew fryum macaroni1 macaroni2 capsules"   # recovery@1% < 0.7 at 256px

gpu_ok() {
  timeout -s KILL 60 "$PY" - <<'PY' > "$OUT/logs/gpu_probe.log" 2>&1
import torch, sys
assert torch.cuda.is_available(), "no gpu"
x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum().item(); torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info(); print(f"probe ok free={free/2**30:.1f}GiB total={total/2**30:.1f}GiB")
PY
}
guard() {
  if ! gpu_ok; then
    echo "$(date +%H:%M:%S) GPU PROBE FAILED - stopping grid (see $OUT/logs/gpu_probe.log)" | tee -a "$LOG"
    exit 1
  fi
}

run() { # dataset category model tagsuffix [extra args]
  local tag="$1_$2_$3$4"
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag" | tee -a "$LOG"; return; fi
  guard
  echo "$(date +%H:%M:%S) start $tag ${*:5}" | tee -a "$LOG"
  timeout -s KILL "${RUN_TIMEOUT:-5400}" "$PY" scripts/run.py "$1" "$2" "$3" --root "$ROOT" --out "$OUT" "${@:5}" > "$OUT/logs/$tag.log" 2>&1 \
    && echo "$(date +%H:%M:%S) done  $tag $(grep -o '"recovery@0.01": [0-9.]*' "$OUT/$tag.json") $(grep -o '"peak_vram_gib": [0-9.]*' "$OUT/$tag.json")" | tee -a "$LOG" \
    || { echo "$(date +%H:%M:%S) FAIL  $tag (see $OUT/logs/$tag.log)" | tee -a "$LOG"; guard; }
}
standin() { # dataset category
  local tag="$1_$2"
  if [ -f "$OUT/standin/${tag}_pca.csv" ]; then echo "skip standin $tag" | tee -a "$LOG"; return; fi
  echo "$(date +%H:%M:%S) standin $tag" | tee -a "$LOG"
  timeout -s KILL 1800 "$PY" scripts/standin.py "$1" "$2" --root "$ROOT" --out "$OUT" > "$OUT/logs/standin_$tag.log" 2>&1 \
    || echo "$(date +%H:%M:%S) FAIL  standin $tag" | tee -a "$LOG"
}

echo "$(date +%H:%M:%S) grid start" | tee -a "$LOG"
guard

# pass 0: stand-ins (CPU, fast)
for c in $MVTEC; do standin mvtec "$c"; done
for c in $VISA;  do standin visa  "$c"; done

# pass 1: fast models everywhere, 256px baseline
for c in $MVTEC; do run mvtec "$c" padim ""; run mvtec "$c" patchcore ""; done
for c in $VISA;  do run visa  "$c" padim ""; run visa  "$c" patchcore ""; done

# pass 2: reverse distillation, capped epochs (ROCm/WSL needs MIOPEN_FIND_MODE=2 for the backward conv)
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
for c in $MVTEC; do run mvtec "$c" rd "" --epochs 10; done
for c in $VISA;  do run visa  "$c" rd "" --epochs 10; done

# pass 3: does resolution fix the multi-instance categories? PatchCore memory bank grows
# with pixels x images, so higher resolution comes with a capped train subset.
for c in $WEAK;  do run visa  "$c" patchcore "-384-n300" --size 384 --train-n 300; done
for c in $WEAK;  do run visa  "$c" patchcore "-256-n300" --train-n 300; done   # control: subset alone
for c in $MVTEC; do run mvtec "$c" patchcore "-320-n300" --size 320 --train-n 300; done
for c in $VISA;  do run visa  "$c" patchcore "-320-n300" --size 320 --train-n 300; done

# pass 4: small backbone for the edge-inference story
for c in $MVTEC; do run mvtec "$c" patchcore "-r18" --backbone resnet18; done
for c in $VISA;  do run visa  "$c" patchcore "-r18" --backbone resnet18; done

# EfficientAD is not run here: it hung the GPU twice on this box (WSL/ROCm, long
# gradient training). Run it on a rented GPU if we need it.

echo "$(date +%H:%M:%S) grid finished" | tee -a "$LOG"
