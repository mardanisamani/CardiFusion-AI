# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Multi-seed / multi-scale ensemble with Weighted Box Fusion (WBF).

Rationale: the detector is data-limited and recall-bound (missed lesions).
Independently-seeded / multi-scale YOLOv8 models miss DIFFERENT lesions, so
fusing their boxes with WBF recovers false negatives -> recall and mAP rise. It
also yields the multi-run variance that addresses the single-seed limitation.

Trains nothing here — it fuses the predictions of already-trained member weights
(see configs/yolo_wbf_s*.yaml). Members are experimental, NOT among the four
submission weights. Scored through the shared COCOeval pipeline for comparability.

    python -m src.ensemble.wbf_ensemble \
        --weights weights/wbf/yolov8_s0.pt weights/wbf/yolov8_s1.pt weights/wbf/yolov8_s2.pt \
        --imgsz 640 640 768 \
        --images data/arcade/stenosis/val/images \
        --coco-gt data/arcade/stenosis/val/annotations/val.json \
        --out report/figures/wbf/val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# torch.load patch (PyTorch>=2.6) before importing ultralytics.
import torch
_orig_load = torch.load
def _patched_load(f, *a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(f, *a, **k)
torch.load = _patched_load

from src.eval.eval_yolo_sahi import pr_f1, coco_map, STENOSIS_CAT_ID


def _predict_norm(model, image_path, imgsz, conf):
    """Return normalized xyxy boxes + scores for one model on one image."""
    r = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    h, w = r.orig_shape
    boxes, scores = [], []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
        scores.append(float(b.conf))
    return boxes, scores, (w, h)


def run(args):
    from ultralytics import YOLO
    from ensemble_boxes import weighted_boxes_fusion
    from pycocotools.coco import COCO

    n = len(args.weights)
    imgszs = args.imgsz if len(args.imgsz) == n else [args.imgsz[0]] * n
    weights_w = args.member_weights if args.member_weights else [1.0] * n
    models = [YOLO(w) for w in args.weights]

    gt = COCO(str(args.coco_gt))
    images_dir = Path(args.images)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    fused_preds, all_gts, fused_dets = {}, {}, []
    for img_id in gt.getImgIds():
        info = gt.loadImgs(img_id)[0]
        p = images_dir / info["file_name"]
        if not p.exists():
            continue
        all_gts[img_id] = [a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id))
                           if a["category_id"] == STENOSIS_CAT_ID]

        boxes_list, scores_list, labels_list, wh = [], [], [], None
        for m, sz in zip(models, imgszs):
            b, s, wh = _predict_norm(m, p, sz, args.conf)
            boxes_list.append(b if b else [[0, 0, 0, 0]])
            scores_list.append(s if s else [0.0])
            labels_list.append([0] * len(b) if b else [0])

        fb, fs, _ = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights_w, iou_thr=args.iou_thr, skip_box_thr=args.skip_box_thr)
        W, H = wh
        preds = []
        for (x1, y1, x2, y2), sc in zip(fb, fs):
            xywh = [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H]
            preds.append((xywh, float(sc)))
            fused_dets.append({"image_id": img_id, "category_id": STENOSIS_CAT_ID,
                               "bbox": [round(c, 2) for c in xywh], "score": round(float(sc), 5)})
        fused_preds[img_id] = preds

    # mAP is threshold-free (COCOeval sweeps all confidences) -> the trustworthy metric.
    ap = coco_map(args.coco_gt, fused_dets)

    # WBF renormalizes/compresses scores, so the F1-optimal threshold is much lower
    # than the usual 0.25. Sweep it and report F1 at the WBF-optimal point — the fair
    # comparison is "baseline at its best threshold vs. WBF at its best threshold".
    if args.eval_conf is not None:
        confs = [args.eval_conf]
    else:
        confs = [round(c, 3) for c in np.arange(0.02, 0.60, 0.02)]
    curve, best = [], None
    for c in confs:
        fp = {k: [(b, s) for b, s in v if s >= c] for k, v in fused_preds.items()}
        m = pr_f1(fp, all_gts, conf_thr=c)
        curve.append({"conf": c, **{k: round(m[k], 4) for k in ["precision", "recall", "f1"]}})
        if best is None or m["f1"] > best["f1"]:
            best = {"conf": c, **m}

    summary = {"members": args.weights, "imgsz": imgszs, "iou_thr": args.iou_thr,
               "skip_box_thr": args.skip_box_thr,
               "map50": round(ap["map50"], 4), "map50_95": round(ap["map50_95"], 4),
               "best_operating_point": {k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in best.items()},
               "pr_curve": curve}
    (out / "wbf_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"mAP@50={ap['map50']:.4f}  mAP@50:95={ap['map50_95']:.4f}  (threshold-free, trustworthy)")
    b = best
    print(f"WBF-optimal F1 point: conf={b['conf']}  P={b['precision']:.4f} "
          f"R={b['recall']:.4f} F1={b['f1']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="WBF multi-seed/scale ensemble (fusion-only)")
    ap.add_argument("--weights", nargs="+", required=True, help="member .pt files")
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640],
                    help="per-member inference size (1 value = same for all)")
    ap.add_argument("--member-weights", nargs="+", type=float, default=None)
    ap.add_argument("--images", required=True)
    ap.add_argument("--coco-gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.55, help="WBF IoU merge threshold")
    ap.add_argument("--skip-box-thr", type=float, default=0.001, help="WBF min score to keep")
    ap.add_argument("--conf", type=float, default=0.001, help="per-model predict conf (keep low)")
    ap.add_argument("--eval-conf", type=float, default=None,
                    help="fixed conf for P/R/F1; omit to SWEEP and report the F1-optimal point")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
