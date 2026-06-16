#!/usr/bin/env bash
# AI-assisted. Phase 1 — convert raw ARCADE COCO into U-Net masks + YOLO labels,
# then write visual QA overlays. Adjust RAW to your download location.
set -euo pipefail
# ARCADE lives here after download/unzip:
#   data/arcade/{syntax,stenosis}/{train,val,test}/{images, annotations/<split>.json}
RAW="${1:-data/arcade}"

echo "== syntax -> binary vessel masks (U-Net targets) =="
for split in train val test; do
  ANN=$(ls "${RAW}/syntax/${split}/annotations/"*.json | head -n1)
  python -m src.data.coco_to_mask \
    --images "${RAW}/syntax/${split}/images" \
    --ann "${ANN}" \
    --out "data/syntax/${split}"
done

echo "== stenosis -> YOLO labels =="
python -m src.data.coco_to_yolo --root "${RAW}/stenosis" --out data/yolo

echo "== visual QA overlays -> report/qa =="
python -m src.data.qa_viz --seg-root data/syntax/train --yolo-root data/yolo --split train --n 10

echo "Phase 1 done. Inspect report/qa/ before training."
