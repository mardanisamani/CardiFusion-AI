# Multi-Seed YOLOv8 on Google Colab

Three identical training notebooks that differ **only by the random seed**, plus a fusion/comparison
notebook. Purpose: measure seed variability (mean ± std) and prepare a Weighted-Box-Fusion ensemble.

## Files
- `train_seed1.ipynb`, `train_seed2.ipynb`, `train_seed3.ipynb` — one per seed (1, 2, 3). The **only**
  difference between them is the `SEED` value; all training settings are identical
  (YOLOv8m, 100 epochs, imgsz 640, batch 16, SGD, lr0 0.01, `close_mosaic=10`, `deterministic=True`).
- `fusion_compare.ipynb` — run after all three; reports per-seed mean ± std and fuses the seeds with WBF.

## One-time setup (Google Drive)
Upload to `MyDrive/arcade_multiseed/` (edit `PROJECT` in the notebooks if you use another path):
1. `data/yolo/` — the YOLO dataset built locally by `src/data/coco_to_yolo.py`
   (contains `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`).
   Tip: `zip -r yolo.zip data/yolo` locally, upload, then unzip in Drive.
2. `data/test.json` — the ARCADE stenosis **test** COCO annotations
   (`data/arcade/stenosis/test/annotations/test.json`), used by the fusion notebook for mAP.

## Run order
1. Set **Runtime → Change runtime type → GPU (T4)** in each notebook.
2. Run `train_seed1.ipynb` → produces `outputs/best_seed1.pt`, `metrics_seed1.json`, `pred_seed1.json`.
3. Repeat for `train_seed2.ipynb` and `train_seed3.ipynb` (can run in parallel Colab sessions).
4. Run `fusion_compare.ipynb` → prints seed mean ± std and the fused WBF mAP; writes `pred_fused_wbf.json`.

## Outputs (all in `MyDrive/arcade_multiseed/outputs/`)
| File | Contents |
|---|---|
| `best_seed{1,2,3}.pt` | trained weights, one per seed |
| `metrics_seed{1,2,3}.json` | test-set precision, recall, F1, mAP@50, mAP@50:95 |
| `pred_seed{1,2,3}.json` | per-image detections (COCO format, conf≥0.001) for fusion |
| `pred_fused_wbf.json` | WBF-fused detections across the 3 seeds |

## Notes
- Predictions are dumped at a very low confidence (0.001) so the full precision–recall curve is
  preserved for fair mAP and for fusion. Apply a higher threshold only when reporting an operating point.
- Member predictions use `category_id = 0` (YOLO class); the fusion notebook remaps to the ARCADE
  stenosis category id (26) for COCOeval.
- For the strongest configuration later, fuse **seeds × scales** (run each seed's prediction cell at
  640/768/1024 and pool all nine into the same WBF call).
