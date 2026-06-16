#!/usr/bin/env bash
# AI-assisted. End-to-end driver AFTER data is prepared (scripts/prepare_data.sh).
# Each long training step is explicit so you can run them individually on a GPU.
set -euo pipefail

echo "== Phase 2: baseline U-Net =="
python -m src.train.train_unet --config configs/unet_baseline.yaml

echo "== Phase 3: weighted-edge U-Net (sweep lambda externally; default 6) =="
python -m src.train.train_unet --config configs/unet_weighted_edge.yaml

echo "== Stage-1 comparison (table + 4-panel failure figure) =="
python -m src.eval.eval_unet --compare \
  --baseline weights/unet_baseline.pt \
  --improved weights/unet_weighted_edge.pt \
  --val-dir data/syntax/val --out report/figures

echo "== Phase 4: baseline YOLOv8 =="
python -m src.train.train_yolo --config configs/yolo_baseline.yaml

echo "== Phase 5: improved YOLOv8 =="
python -m src.train.train_yolo --config configs/yolo_improved.yaml

echo "== Stage-2 comparison (table + side-by-side failure figure) =="
python -m src.eval.eval_yolo --compare \
  --baseline weights/yolov8_baseline.pt \
  --improved weights/yolov8_improved.pt \
  --data data/yolo/data.yaml --split val --out report/figures

echo "== Phase 6: ensemble (vessel prior as YOLO input; reuses improved slot) =="
python -m src.ensemble.build_vessel_channel \
  --unet weights/unet_weighted_edge.pt --yolo-root data/yolo --out data/yolo_ensemble
python -m src.train.train_yolo --config configs/ensemble.yaml
python -m src.eval.eval_yolo --weights weights/yolov8_improved.pt \
  --data data/yolo_ensemble/data.yaml --split val --out report/figures/ensemble

echo "== Phase 7: render report + build submission zip =="
python scripts/render_report.py
python scripts/package_submission.py
echo "ALL DONE. See submission.zip"
