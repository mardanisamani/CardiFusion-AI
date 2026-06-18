#!/usr/bin/env bash
# AI-assisted. Inference-only experiment: standard vs high-res vs SAHI tiled
# inference for stenosis detection. Does NOT retrain or add a weight file.
# Usage: bash scripts/run_sahi_experiment.sh [val|test]
set -euo pipefail
SPLIT="${1:-val}"
W="weights/yolov8_baseline.pt"        # also try weights/yolov8_improved.pt
IMG="data/arcade/stenosis/${SPLIT}/images"
GT="data/arcade/stenosis/${SPLIT}/annotations/${SPLIT}.json"
OUT="report/figures/sahi"

echo "== full-image @640 (standard baseline, re-scored in this pipeline) =="
python -m src.eval.eval_yolo_sahi --mode full --imgsz 640  --weights "$W" --images "$IMG" --coco-gt "$GT" --out "$OUT/full640"

echo "== full-image @1280 (high-res 'SAHI-lite') =="
python -m src.eval.eval_yolo_sahi --mode full --imgsz 1280 --weights "$W" --images "$IMG" --coco-gt "$GT" --out "$OUT/full1280"

echo "== SAHI tiled (slice 256, overlap 0.25) =="
python -m src.eval.eval_yolo_sahi --mode sahi --slice 256 --overlap 0.25 --weights "$W" --images "$IMG" --coco-gt "$GT" --out "$OUT/sahi256"

echo "== summary =="
python - <<'PY'
import json, glob, os
rows=[]
for d in ["full640","full1280","sahi256"]:
    p=f"report/figures/sahi/{d}/metrics.json"
    if os.path.exists(p):
        m=json.load(open(p))
        rows.append((d, m.get("precision"),m.get("recall"),m.get("f1"),m.get("map50"),m.get("map50_95")))
print(f"{'config':10} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'mAP5095':>8}")
for r in rows:
    print(f"{r[0]:10} "+" ".join(f"{x:6.3f}" if isinstance(x,(int,float)) else f"{str(x):>6}" for x in r[1:]))
PY
echo "Keep whichever config genuinely beats full640 on F1/mAP. SAHI is inference-only — safe to drop if it does not help."
