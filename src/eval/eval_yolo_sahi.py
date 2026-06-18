# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Inference-only experiment: does SAHI tiled inference (or higher-res full-image
inference) improve stenosis detection over standard full-image inference?

This is EVAL-ONLY — it does not train or create a fifth weight file. It runs the
*existing* YOLOv8 checkpoint two ways through the *same* COCO-eval pipeline so the
comparison is fair:

  * mode=full  : one forward pass on the whole image at --imgsz (standard, or
                 raise --imgsz to 1024/1280 for "SAHI-lite" high-res inference).
  * mode=sahi  : SAHI sliced prediction (slice --slice, overlap --overlap) with
                 NMS merge, then the same scoring.

Both modes emit COCO-format detections evaluated with pycocotools COCOeval
against the original ARCADE stenosis annotations, yielding mAP@50 and mAP@50:95
directly. Greedy IoU@0.5 matching adds precision/recall/F1 at --conf.

Why this design: ARCADE images are 512x512 and stenosis boxes are medium-sized
(median ~50px), so SAHI is NOT in its usual tiny-object sweet spot. Measure both
and keep whichever genuinely wins on val/test.

Examples
--------
    # standard full-image baseline (re-scored through this pipeline)
    python -m src.eval.eval_yolo_sahi --mode full --imgsz 640 \
        --weights weights/yolov8_baseline.pt \
        --images data/arcade/stenosis/val/images \
        --coco-gt data/arcade/stenosis/val/annotations/val.json \
        --out report/figures/sahi/full640

    # high-res full-image ("SAHI-lite")
    python -m src.eval.eval_yolo_sahi --mode full --imgsz 1280 ... --out .../full1280

    # SAHI tiled inference
    python -m src.eval.eval_yolo_sahi --mode sahi --slice 256 --overlap 0.25 \
        --weights weights/yolov8_baseline.pt \
        --images data/arcade/stenosis/val/images \
        --coco-gt data/arcade/stenosis/val/annotations/val.json \
        --out report/figures/sahi/sahi256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# PyTorch >= 2.6 defaults torch.load to weights_only=True, which breaks ultralytics
# checkpoint loading; patch before importing model libraries.
import torch
_orig_load = torch.load
def _patched_load(f, *a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(f, *a, **k)
torch.load = _patched_load

STENOSIS_CAT_ID = 26  # ARCADE COCO category id for 'stenosis'


# ----------------------------------------------------------------------------
# Prediction back-ends
# ----------------------------------------------------------------------------
def predict_full(weights, image_path, imgsz, conf, augment=False):
    """Standard whole-image YOLOv8 inference. Returns list of (xywh, score).
    Set augment=True for built-in test-time augmentation (TTA)."""
    from ultralytics import YOLO
    model = getattr(predict_full, "_m", None)
    if model is None or predict_full._w != weights:
        predict_full._m = YOLO(weights); predict_full._w = weights
        model = predict_full._m
    r = model.predict(str(image_path), imgsz=imgsz, conf=conf,
                      augment=augment, verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        out.append(([x1, y1, x2 - x1, y2 - y1], float(b.conf)))
    return out


def predict_sahi(weights, image_path, slice_size, overlap, conf):
    """SAHI sliced inference. Returns list of (xywh, score)."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    model = getattr(predict_sahi, "_m", None)
    if model is None or predict_sahi._w != (weights, conf):
        predict_sahi._m = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=weights,
            confidence_threshold=conf,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        predict_sahi._w = (weights, conf)
        model = predict_sahi._m
    result = get_sliced_prediction(
        str(image_path), model,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        verbose=0,
    )
    out = []
    for op in result.object_prediction_list:
        x, y, w, h = op.bbox.to_xywh()  # [x,y,w,h]
        out.append(([x, y, w, h], float(op.score.value)))
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def pr_f1(all_preds, all_gts, iou_thr=0.5, conf_thr=0.25):
    """Greedy IoU matching at a single conf/IoU threshold -> P/R/F1."""
    tp = fp = fn = 0
    for img_id, gts in all_gts.items():
        preds = [p for p in all_preds.get(img_id, []) if p[1] >= conf_thr]
        preds.sort(key=lambda x: -x[1])
        matched = set()
        for box, _ in preds:
            best, bj = iou_thr, -1
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                iou = _iou_xywh(box, g)
                if iou >= best:
                    best, bj = iou, j
            if bj >= 0:
                tp += 1; matched.add(bj)
            else:
                fp += 1
        fn += len(gts) - len(matched)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def coco_map(coco_gt_path, coco_dets):
    """mAP@50 and mAP@50:95 via pycocotools, stenosis category only."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(str(coco_gt_path))
    if not coco_dets:
        return {"map50": 0.0, "map50_95": 0.0}
    coco_dt = coco_gt.loadRes(coco_dets)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.catIds = [STENOSIS_CAT_ID]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"map50_95": float(ev.stats[0]), "map50": float(ev.stats[1])}


# ----------------------------------------------------------------------------
def run(args):
    from pycocotools.coco import COCO
    gt = COCO(str(args.coco_gt))
    img_ids = gt.getImgIds()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    all_preds, all_gts, coco_dets = {}, {}, []
    images_dir = Path(args.images)
    for img_id in img_ids:
        info = gt.loadImgs(img_id)[0]
        path = images_dir / info["file_name"]
        if not path.exists():
            continue
        # GT boxes (stenosis only)
        gts = [a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id))
               if a["category_id"] == STENOSIS_CAT_ID]
        all_gts[img_id] = gts
        # predictions
        if args.mode == "sahi":
            preds = predict_sahi(args.weights, path, args.slice, args.overlap, args.conf)
        else:
            preds = predict_full(args.weights, path, args.imgsz, args.conf)
        all_preds[img_id] = preds
        for box, score in preds:
            coco_dets.append({"image_id": img_id, "category_id": STENOSIS_CAT_ID,
                              "bbox": [round(c, 2) for c in box], "score": round(score, 5)})

    metrics = {"mode": args.mode,
               "imgsz": args.imgsz if args.mode == "full" else None,
               "slice": args.slice if args.mode == "sahi" else None,
               "overlap": args.overlap if args.mode == "sahi" else None,
               "conf": args.conf}
    metrics.update(pr_f1(all_preds, all_gts, conf_thr=args.conf))
    metrics.update(coco_map(args.coco_gt, coco_dets))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "detections.json").write_text(json.dumps(coco_dets))
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="SAHI / high-res YOLOv8 eval (inference-only)")
    ap.add_argument("--mode", choices=["full", "sahi"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="dir of split images")
    ap.add_argument("--coco-gt", required=True, help="ARCADE stenosis COCO json for the split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640, help="full-mode inference size")
    ap.add_argument("--slice", type=int, default=256, help="sahi slice size")
    ap.add_argument("--overlap", type=float, default=0.25, help="sahi slice overlap ratio")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
