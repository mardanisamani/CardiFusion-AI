# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Stage 3 (corrected ensemble): VESSEL-GUIDED GATING, post-hoc.

The naive channel-fusion ensemble regressed. This is the corrected version the
report proposed: run the improved (weighted-edge) U-Net to get a vessel mask,
dilate it into an anatomical tolerance band, then SUPPRESS any YOLOv8 stenosis
detection that does not sit on the coronary tree. Stenoses lie on vessels by
definition, so true positives are kept while off-vessel false positives
(catheter/rib/background) are removed -> precision and mAP up, recall preserved.

Inference-only: no retraining, no new weight file. It post-processes the EXISTING
detector + U-Net. Baseline (ungated) and gated detections are scored through the
SAME COCOeval pipeline (reused from eval_yolo_sahi) so the comparison is fair.

    python -m src.ensemble.vessel_gating \
        --yolo-weights weights/yolov8_baseline.pt \
        --unet weights/unet_weighted_edge.pt \
        --images data/arcade/stenosis/val/images \
        --coco-gt data/arcade/stenosis/val/annotations/val.json \
        --out report/figures/gating --dilate 25 --min-cov 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# torch.load patch (PyTorch>=2.6) must run before importing model libs.
import torch
_orig_load = torch.load
def _patched_load(f, *a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(f, *a, **k)
torch.load = _patched_load

from src.data.preprocess import enhance
from src.models.unet import build_unet
from src.eval.eval_yolo_sahi import predict_full, pr_f1, coco_map, STENOSIS_CAT_ID


# ----------------------------------------------------------------------------
def load_unet(weights, device):
    ckpt = torch.load(weights, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = build_unet(cfg.get("encoder", "resnet34"), 1, 2, encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model


@torch.no_grad()
def vessel_mask(model, gray, device, size=512, thr=0.5, dilate_px=25):
    """Binary vessel mask at the image's native size, dilated into a band."""
    h, w = gray.shape
    enh = enhance(gray)
    x = cv2.resize(enh, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x.astype(np.float32) / 255.0)[None, None].to(device)
    prob = torch.softmax(model(t), 1)[0, 1].cpu().numpy()
    m = (cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) >= thr).astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    return m


def box_on_vessel(box_xywh, mask, min_cov):
    """Keep a box if >= min_cov of its area overlaps the (dilated) vessel mask."""
    x, y, w, h = [int(round(v)) for v in box_xywh]
    H, W = mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = mask[y0:y1, x0:x1]
    cov = crop.mean() if crop.size else 0.0
    return cov >= min_cov


# ----------------------------------------------------------------------------
def run(args):
    from pycocotools.coco import COCO
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    gt = COCO(str(args.coco_gt))
    unet = load_unet(args.unet, device)
    images_dir = Path(args.images)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    base_preds, gated_preds, all_gts = {}, {}, {}
    base_dets, gated_dets = [], []
    kept = removed = 0

    for img_id in gt.getImgIds():
        info = gt.loadImgs(img_id)[0]
        p = images_dir / info["file_name"]
        if not p.exists():
            continue
        all_gts[img_id] = [a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id))
                           if a["category_id"] == STENOSIS_CAT_ID]
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        vmask = vessel_mask(unet, gray, device, dilate_px=args.dilate)

        preds = predict_full(args.yolo_weights, p, args.imgsz, args.conf)
        base_preds[img_id] = preds
        gated = []
        for box, score in preds:
            if box_on_vessel(box, vmask, args.min_cov):
                gated.append((box, score)); kept += 1
            else:
                removed += 1
        gated_preds[img_id] = gated

        for box, score in preds:
            base_dets.append({"image_id": img_id, "category_id": STENOSIS_CAT_ID,
                              "bbox": [round(c, 2) for c in box], "score": round(score, 5)})
        for box, score in gated:
            gated_dets.append({"image_id": img_id, "category_id": STENOSIS_CAT_ID,
                               "bbox": [round(c, 2) for c in box], "score": round(score, 5)})

    base = {**pr_f1(base_preds, all_gts, conf_thr=args.conf), **coco_map(args.coco_gt, base_dets)}
    gated = {**pr_f1(gated_preds, all_gts, conf_thr=args.conf), **coco_map(args.coco_gt, gated_dets)}
    delta = {k: round(gated[k] - base[k], 4) for k in ["precision", "recall", "f1", "map50", "map50_95"]}
    summary = {"params": {"dilate_px": args.dilate, "min_cov": args.min_cov, "conf": args.conf,
                          "imgsz": args.imgsz, "boxes_kept": kept, "boxes_removed": removed},
               "baseline": base, "gated": gated, "delta": delta}
    (out / "gating_metrics.json").write_text(json.dumps(summary, indent=2))

    keys = ["precision", "recall", "f1", "map50", "map50_95"]
    rows = ["| Metric | YOLOv8 (ungated) | + Vessel gating | Δ |", "|---|---|---|---|"]
    for k in keys:
        rows.append(f"| {k} | {base[k]:.4f} | {gated[k]:.4f} | {gated[k]-base[k]:+.4f} |")
    (out / "gating_table.md").write_text("\n".join(rows))
    print("\n".join(rows))
    print(f"\nboxes kept={kept} removed_off_vessel={removed}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Vessel-guided gating (post-hoc ensemble, inference-only)")
    ap.add_argument("--yolo-weights", required=True)
    ap.add_argument("--unet", default="weights/unet_weighted_edge.pt")
    ap.add_argument("--images", required=True)
    ap.add_argument("--coco-gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dilate", type=int, default=25, help="vessel-mask dilation band (px)")
    ap.add_argument("--min-cov", type=float, default=0.05, help="min box-area fraction on vessel")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
