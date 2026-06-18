# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Recall-focused, inference-only improvement for stenosis detection.

Diagnosis: the baseline is precision-decent (~0.50) but recall-starved (~0.31);
its failure mode is MISSED lesions (false negatives). Vessel gating / NMS tricks
only move the operating point and cannot help recall. This script attacks recall
two legitimate, training-free ways:

  1. Test-Time Augmentation (TTA, `augment=True`): flip/scale-averaged inference
     recovers borderline lesions.
  2. Operating-point selection: collect detections at a very low conf, then sweep
     the confidence threshold and pick the F1-optimal point on VAL (report it on
     TEST). Reporting at conf=0.25 by default leaves F1 on the table.

mAP@50 / mAP@50:95 (threshold-independent) are also reported so you can see the
true ceiling. Everything is scored through the shared COCOeval pipeline.

    # tune the operating point on val (with TTA)
    python -m src.eval.eval_yolo_oppoint --weights weights/yolov8_baseline.pt \
        --images data/arcade/stenosis/val/images \
        --coco-gt data/arcade/stenosis/val/annotations/val.json \
        --tta --out report/figures/oppoint/val

    # then evaluate the chosen threshold on test
    python -m src.eval.eval_yolo_oppoint --weights weights/yolov8_baseline.pt \
        --images data/arcade/stenosis/test/images \
        --coco-gt data/arcade/stenosis/test/annotations/test.json \
        --tta --fixed-conf <BEST_FROM_VAL> --out report/figures/oppoint/test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.eval_yolo_sahi import predict_full, pr_f1, coco_map, STENOSIS_CAT_ID


def collect(weights, images_dir, coco_gt_path, imgsz, tta, low_conf):
    """Run inference once at a low conf; return preds-by-image, gts-by-image, dets."""
    from pycocotools.coco import COCO
    gt = COCO(str(coco_gt_path))
    images_dir = Path(images_dir)
    preds, gts, dets = {}, {}, []
    for img_id in gt.getImgIds():
        info = gt.loadImgs(img_id)[0]
        p = images_dir / info["file_name"]
        if not p.exists():
            continue
        gts[img_id] = [a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id))
                       if a["category_id"] == STENOSIS_CAT_ID]
        pr = predict_full(weights, p, imgsz, low_conf, augment=tta)
        preds[img_id] = pr
        for box, score in pr:
            dets.append({"image_id": img_id, "category_id": STENOSIS_CAT_ID,
                         "bbox": [round(c, 2) for c in box], "score": round(score, 5)})
    return preds, gts, dets


def filter_preds(preds, thr):
    return {k: [(b, s) for b, s in v if s >= thr] for k, v in preds.items()}


def run(args):
    preds, gts, dets = collect(args.weights, args.images, args.coco_gt,
                               args.imgsz, args.tta, low_conf=0.001)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # mAP is threshold-independent (uses all scored dets).
    ap = coco_map(args.coco_gt, dets)

    if args.fixed_conf is not None:
        confs = [args.fixed_conf]
    else:
        confs = [round(c, 3) for c in np.arange(0.05, 0.85, 0.05)]

    curve, best = [], None
    for c in confs:
        m = pr_f1(filter_preds(preds, c), gts, conf_thr=c)
        row = {"conf": c, **{k: round(m[k], 4) for k in ["precision", "recall", "f1"]}}
        curve.append(row)
        if best is None or m["f1"] > best["f1"]:
            best = {"conf": c, **m}

    summary = {"tta": args.tta, "imgsz": args.imgsz, **ap,
               "best_operating_point": {k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in best.items()},
               "pr_curve": curve}
    (out / "oppoint_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"mAP@50={ap['map50']:.4f}  mAP@50:95={ap['map50_95']:.4f}  (ceiling, threshold-free)")
    print(f"best F1 operating point: conf={best['conf']}  "
          f"P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="TTA + operating-point tuning (inference-only)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--coco-gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tta", action="store_true", help="enable test-time augmentation")
    ap.add_argument("--fixed-conf", type=float, default=None,
                    help="evaluate only this conf (use the val-optimal value on test)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
