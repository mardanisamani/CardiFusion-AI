# ARCADE Interview — Vessel Segmentation (U-Net) + Stenosis Detection (YOLOv8)

Reproducible solution for the ML Software Developer interview project on the ARCADE coronary
X-ray angiography dataset (Popov et al., 2024). Two paradigms — a U-Net for binary vessel
segmentation and YOLOv8 for stenosis detection — each reproduced, improved with one targeted
change, compared, and finally combined.

> **AI usage:** this repository was built with **Claude (Anthropic)** as an ML pair-programmer.
> Every source file carries an `# AI-assisted` banner; see `report/report.md` §6. (Hard grading
> requirement — cited in both code and report.)

## Deliverables
- Source code (`src/`, `configs/`, `scripts/`).
- **Exactly four** trained weights in `weights/`: `unet_baseline.pt`, `unet_weighted_edge.pt`,
  `yolov8_baseline.pt`, `yolov8_improved.pt` (the ensemble reuses the improved slot — no fifth file).
- `report/report.pdf` (≤ 5 pages). Build the submission bundle with `scripts/package_submission.py`.

## Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pinned; CUDA 11.8 / Python 3.10 tested
```
A CUDA GPU is required for training (U-Net ~80–150 epochs; YOLOv8 ~100 epochs). Seeds and
library/GPU versions are logged to `run_manifest.json` at every run.

## Data
The ARCADE dataset is expected at `data/arcade/`:
```
data/arcade/{syntax,stenosis}/{train,val,test}/{images/*.png, annotations/<split>.json}
```
`syntax` = vessel segmentation (categories 1–25), `stenosis` = lesion detection (single
`stenosis` category, id 26), each split 1000/200/300. If you need to fetch it,
`bash scripts/download_data.sh` prints the exact Zenodo/GitHub commands.
```bash
bash scripts/prepare_data.sh           # defaults to data/arcade
# Phase 1: COCO -> binary masks (data/syntax/) + YOLO labels (data/yolo/) + QA overlays
```
Phase 1 writes U-Net targets to `data/syntax/{split}/{images,masks}`, YOLO data to `data/yolo/`,
and visual QA overlays to `report/qa/`. **Inspect the QA overlays before training.**

## Reproduce every result
End to end (each step also runs standalone):
```bash
bash scripts/run_all.sh
```
Step by step:
```bash
# Stage 1 — U-Net
python -m src.train.train_unet --config configs/unet_baseline.yaml
python -m src.train.train_unet --config configs/unet_weighted_edge.yaml   # sweep lambda {2,4,6,8,10}
python -m src.eval.eval_unet --compare \
  --baseline weights/unet_baseline.pt --improved weights/unet_weighted_edge.pt \
  --val-dir data/syntax/val --out report/figures

# Stage 2 — YOLOv8
python -m src.train.train_yolo --config configs/yolo_baseline.yaml
python -m src.train.train_yolo --config configs/yolo_improved.yaml
python -m src.eval.eval_yolo --compare \
  --baseline weights/yolov8_baseline.pt --improved weights/yolov8_improved.pt \
  --data data/yolo/data.yaml --split val --out report/figures

# Stage 3 — ensemble (vessel prior as YOLO input; reuses improved slot)
python -m src.ensemble.build_vessel_channel \
  --unet weights/unet_weighted_edge.pt --yolo-root data/yolo --out data/yolo_ensemble
python -m src.train.train_yolo --config configs/ensemble.yaml

# Report + package
python scripts/render_report.py        # report.md -> report.pdf (<=5 pages)
python scripts/package_submission.py   # -> submission.zip
```

## Stage 2 improvement — pick the branch that matches your baseline failure
`configs/yolo_improved.yaml` ships **branch A** (domain-specific augmentation) for *missed small
lesions*. If your baseline instead makes *catheter/artifact false positives*, switch to **branch B**
(advanced preprocessing): rebuild data with enhancement and point the config at it —
```bash
python -m src.data.coco_to_yolo --root data/raw/stenosis --out data/yolo_enhanced --enhance
# then set data_yaml: data/yolo_enhanced/data.yaml in configs/yolo_improved.yaml and revert aug knobs
```
Change **one** thing only; keep everything else identical for a clean comparison.

## Layout
```
src/data/    COCO->mask, COCO->YOLO, preprocessing (top-hat+CLAHE), dataset, QA viz
src/models/  unet, losses (Dice+CE, weighted-edge), edge-mask generation
src/train/   train_unet.py, train_yolo.py
src/eval/    seg_metrics (dice/iou/clDice/P/R/thin-recall), det_metrics, eval_unet, eval_yolo
src/ensemble/ build_vessel_channel.py (vessel prior -> 3-ch composite)
configs/     one YAML per experiment        weights/  the four .pt files
report/      report.md (+figures/, qa/)     scripts/  data + run_all + render + package
```

## Notes & assumptions
- ARCADE masks label only ~60–80% of vessels → unlabeled vessels look like background under a naive
  loss; this motivates the weighted-edge loss (stated in the report).
- "Reproduce" = faithful recipe + comparable metrics, not bit-identical paper numbers.
- The white top-hat operates on the photographic negative (XCA vessels are dark-on-bright); see
  `src/data/preprocess.py` (`negative` flag) for this explicit assumption.
