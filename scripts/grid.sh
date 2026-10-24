#!/bin/bash
# Run every stand-in and every model on every category. Skips what already has a result.
# usage: PYTHON=path/to/python bash scripts/grid.sh [DATA_ROOT] [RESULTS_DIR]
set -u
ROOT="${1:-data}"
OUT="${2:-results}"
PY="${PYTHON:-python}"
LOG="$OUT/grid.log"
mkdir -p "$OUT/logs"

MVTEC="bottle hazelnut"
VISA="candle cashew chewinggum fryum pipe_fryum macaroni1 macaroni2 capsules"
WEAK="candle cashew fryum macaroni1 macaroni2 capsules"
GOODSAD="drink_can drink_bottle food_bottle food_package food_box"

log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

# 60 s matmul. If the GPU is gone the grid stops instead of hammering a dead card.
gpu_ok() {
  timeout -s KILL 60 "$PY" -c "import torch; x=torch.randn(2048,2048,device='cuda'); (x@x).sum().item()" > "$OUT/logs/gpu_probe.log" 2>&1
}
guard() { gpu_ok || { log "GPU probe failed, stopping"; exit 1; }; }

# run DATASET CATEGORY MODEL TAGSUFFIX [run.py args]
run() {
  local tag="$1_$2_$3$4"
  [ -f "$OUT/$tag.json" ] && { log "skip $tag"; return; }
  guard
  log "start $tag ${*:5}"
  if timeout -s KILL "${RUN_TIMEOUT:-3600}" "$PY" scripts/run.py "$1" "$2" "$3" --root "$ROOT" --out "$OUT" "${@:5}" > "$OUT/logs/$tag.log" 2>&1; then
    log "done  $tag $(grep -o '"recovery@0.01": [0-9.]*' "$OUT/$tag.json") $(grep -o '"peak_vram_gib": [0-9.]*' "$OUT/$tag.json")"
  else
    log "FAIL  $tag"
    guard
  fi
}

# standin DATASET CATEGORY
standin() {
  local tag="$1_$2"
  [ -f "$OUT/standin/${tag}_tinypadim.csv" ] && { log "skip standin $tag"; return; }
  log "standin $tag"
  timeout -s KILL 1800 "$PY" scripts/standin.py "$1" "$2" --root "$ROOT" --out "$OUT" > "$OUT/logs/standin_$tag.log" 2>&1 || log "FAIL  standin $tag"
}

log "grid start"
guard

for c in $MVTEC; do standin mvtec "$c"; done
for c in $VISA;  do standin visa  "$c"; done

for c in $MVTEC; do run mvtec "$c" padim ""; run mvtec "$c" patchcore ""; done
for c in $VISA;  do run visa  "$c" padim ""; run visa  "$c" patchcore ""; done

export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"   # ROCm needs this for the backward conv in rd
for c in $MVTEC; do run mvtec "$c" rd "" --epochs 10; done
for c in $VISA;  do run visa  "$c" rd "" --epochs 10; done

# resolution and training-set size, small fixed subsets so the PatchCore bank fits in VRAM
for c in $WEAK;  do run visa  "$c" patchcore "-384-n150" --size 384 --train-n 150; done
for c in $WEAK;  do run visa  "$c" patchcore "-n150" --train-n 150; done
for c in $WEAK;  do run visa  "$c" patchcore "-320-n150" --size 320 --train-n 150; done
for c in $MVTEC; do run mvtec "$c" patchcore "-320-n300" --size 320 --train-n 300; done
for c in $VISA;  do run visa  "$c" patchcore "-320-n300" --size 320 --train-n 300; done

for c in $MVTEC; do run mvtec "$c" patchcore "-r18" --backbone resnet18; done
for c in $VISA;  do run visa  "$c" patchcore "-r18" --backbone resnet18; done

# GoodsAD: 3000 px supermarket goods, up to 1000 training images, so PatchCore gets a 400-image subset
for c in $GOODSAD; do standin goodsad "$c"; done
for c in $GOODSAD; do run goodsad "$c" padim ""; run goodsad "$c" patchcore "-n400" --train-n 400; done

log "grid finished"
