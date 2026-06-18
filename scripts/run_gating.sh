#!/usr/bin/env bash
# AI-assisted. Corrected ensemble: vessel-guided gating (inference-only).
# Sweeps the tolerance band on val, prints a summary, keeps the best vs ungated.
# Usage: bash scripts/run_gating.sh [val|test] [weights/yolov8_baseline.pt]
set -euo pipefail
SPLIT="${1:-val}"
YW="${2:-weights/yolov8_baseline.pt}"
UNET="weights/unet_weighted_edge.pt"
IMG="data/arcade/stenosis/${SPLIT}/images"
GT="data/arcade/stenosis/${SPLIT}/annotations/${SPLIT}.json"

for D in 15 25 40; do
  for C in 0.02 0.05; do
    OUT="report/figures/gating/d${D}_c${C}"
    echo "== dilate=${D}px  min_cov=${C} =="
    python -m src.ensemble.vessel_gating --yolo-weights "$YW" --unet "$UNET" \
      --images "$IMG" --coco-gt "$GT" --out "$OUT" --dilate "$D" --min-cov "$C"
  done
done

echo "== sweep summary (gated minus ungated) =="
python - <<'PY'
import json, glob, os
best=None
print(f"{'config':14} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'dF1':>7} {'dmAP50':>7}")
for f in sorted(glob.glob("report/figures/gating/*/gating_metrics.json")):
    m=json.load(open(f)); g=m["gated"]; d=m["delta"]; name=os.path.basename(os.path.dirname(f))
    print(f"{name:14} {g['precision']:6.3f} {g['recall']:6.3f} {g['f1']:6.3f} {g['map50']:7.3f} {d['f1']:+7.3f} {d['map50']:+7.3f}")
    if best is None or g['map50']>best[1]: best=(name,g['map50'])
if best: print(f"\nBest mAP50 config: {best[0]} ({best[1]:.3f}). Use its --dilate/--min-cov in the report.")
PY
