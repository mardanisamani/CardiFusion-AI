#!/usr/bin/env bash
# AI-assisted. PURE multi-seed WBF ensemble for stenosis detection.
# Step 1 trains 3 members that are identical except the random seed (different
# weight init + augmentation order). Step 2 runs each model on the same split and
# fuses overlapping detections with Weighted Box Fusion. Step 3 reports the
# members' mean±std (variance estimate) and the fused result.
# Members are experimental (weights/wbf/), NOT the 4 submission weights.
# Usage: bash scripts/run_wbf.sh [val|test]
set -euo pipefail
SPLIT="${1:-val}"
IMG="data/arcade/stenosis/${SPLIT}/images"
GT="data/arcade/stenosis/${SPLIT}/annotations/${SPLIT}.json"
DATA="data/yolo/data.yaml"

echo "== Step 1: train 3 seed-only members (skip any already trained) =="
for c in s0 s1 s2; do
  if [ ! -f "weights/wbf/yolov8_${c}.pt" ]; then
    python -m src.train.train_yolo --config "configs/yolo_wbf_${c}.yaml"
  else
    echo "  weights/wbf/yolov8_${c}.pt exists — skipping"
  fi
done

echo "== Step 2: per-member metrics via the SAME COCO-eval engine as the fusion =="
# (single-weight WBF == that model's own predictions, scored through coco_map/pr_f1)
for c in s0 s1 s2; do
  python -m src.ensemble.wbf_ensemble \
    --weights "weights/wbf/yolov8_${c}.pt" --imgsz 640 \
    --images "$IMG" --coco-gt "$GT" \
    --out "report/figures/wbf/${SPLIT}/member_${c}" \
    --iou-thr 0.55 --skip-box-thr 0.001
done

echo "== Step 3: fuse the 3 seeds with WBF (same 640 inference for all) =="
python -m src.ensemble.wbf_ensemble \
  --weights weights/wbf/yolov8_s0.pt weights/wbf/yolov8_s1.pt weights/wbf/yolov8_s2.pt \
  --imgsz 640 \
  --images "$IMG" --coco-gt "$GT" \
  --out "report/figures/wbf/${SPLIT}" \
  --iou-thr 0.55 --skip-box-thr 0.001
# (omit --eval-conf so the script sweeps and reports the F1-optimal operating point)

echo "== Step 4: summary (members mean±std vs. fused) =="
python - "$SPLIT" <<'PY'
import json, glob, os, statistics, sys
split = sys.argv[1]
base = f"report/figures/wbf/{split}"

def flatten(w):
    bp = w.get("best_operating_point", {})
    return {"map50": w.get("map50"), "map50_95": w.get("map50_95"),
            "recall": bp.get("recall"), "precision": bp.get("precision"), "f1": bp.get("f1")}

members = [flatten(json.load(open(d))) for d in sorted(glob.glob(f"{base}/member_*/wbf_metrics.json"))]
if members:
    print(f"  members (n={len(members)}), mean ± std at each member's best-F1 point:")
    for k in ["map50", "map50_95", "recall", "precision", "f1"]:
        vals = [m[k] for m in members if m.get(k) is not None]
        if vals:
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"    {k:9s}: {statistics.mean(vals):.4f} ± {sd:.4f}")
fused = f"{base}/wbf_metrics.json"
if os.path.exists(fused):
    f = flatten(json.load(open(fused)))
    print(f"  FUSED: mAP@50={f['map50']}  mAP@50:95={f['map50_95']}  "
          f"R={f['recall']} P={f['precision']} F1={f['f1']}")
print("Read: fused recall/mAP should exceed the members' mean; the ±std quantifies seed variance "
      "(your answer to the single-seed limitation).")
PY
