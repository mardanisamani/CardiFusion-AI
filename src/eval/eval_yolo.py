# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Evaluate YOLOv8 stenosis models: precision, recall, F1, mAP@50, mAP@50:95;
auto-select a failure case; build the Stage-2 comparison table and a side-by-side
prediction figure.

    python -m src.eval.eval_yolo --weights weights/yolov8_baseline.pt \
        --data data/yolo/data.yaml --split val --out report/figures/yolo_baseline

    python -m src.eval.eval_yolo --compare \
        --baseline weights/yolov8_baseline.pt \
        --improved weights/yolov8_improved.pt \
        --data data/yolo/data.yaml --split val --out report/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src.eval.det_metrics import f1_from_pr, summarize_val_metrics, \
    classify_failure


def val_metrics(weights, data, split):
    model = YOLO(weights)
    res = model.val(data=data, split=split, verbose=False)
    return summarize_val_metrics(res)


def _load_split_paths(data_yaml, split):
    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d["path"])
    img_dir = root / d[split]
    return sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))


def _draw(img, boxes, color, label):
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if conf is not None:
            cv2.putText(img, f"{label}{conf:.2f}", (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def select_and_draw_failure(baseline, improved, data, split, out):
    """Pick a baseline failure (missed GT or low-confidence/extra box) and draw
    baseline vs improved predictions against GT side-by-side."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = YOLO(baseline)
    imp = YOLO(improved) if improved else None
    paths = _load_split_paths(data, split)

    worst_path, worst_score, worst_type = None, 1e9, "none"
    for p in paths:
        r = base.predict(str(p), verbose=False, conf=0.25)[0]
        lbl = p.parent.parent.parent / "labels" / split / (p.stem + ".txt")
        n_gt = 0
        if lbl.exists():
            n_gt = len([l for l in lbl.read_text().splitlines() if l.strip()])
        n_pred = len(r.boxes)
        confs = r.boxes.conf.cpu().numpy() if n_pred else np.array([])
        score, ftype = classify_failure(n_gt, n_pred, confs)
        if score < worst_score:
            worst_score, worst_path, worst_type = score, p, ftype

    print(f"failure case: {worst_path.name}  type={worst_type}")
    img = cv2.imread(str(worst_path))
    panels = [("Ground truth", img.copy())]

    # GT boxes
    lbl = worst_path.parent.parent.parent / "labels" / split / (worst_path.stem + ".txt")
    h, w = img.shape[:2]
    gt_boxes = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            _, xc, yc, bw, bh = map(float, line.split())
            gt_boxes.append((( xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h, None))
    _draw(panels[0][1], gt_boxes, (0, 255, 0), "")

    for name, m in [("Baseline", base), ("Improved", imp)]:
        if m is None:
            continue
        r = m.predict(str(worst_path), verbose=False, conf=0.25)[0]
        bx = [(*b.xyxy[0].cpu().numpy(), float(b.conf)) for b in r.boxes]
        canvas = img.copy()
        _draw(canvas, bx, (0, 0, 255), "")
        panels.append((f"{name} pred", canvas))

    strip = np.hstack([
        cv2.copyMakeBorder(c, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for _, c in panels])
    x = 5
    for name, c in panels:
        cv2.putText(strip, name, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        x += c.shape[1]
    save = out / "stage2_failure_case.png"
    cv2.imwrite(str(save), strip)
    print(f"wrote {save}")
    return worst_path.name, worst_type


def compare(baseline, improved, data, split, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    bm = val_metrics(baseline, data, split)
    im = val_metrics(improved, data, split)
    keys = ["precision", "recall", "f1", "map50", "map50_95"]
    rows = ["| Metric | Baseline YOLOv8 | Improved YOLOv8 | Δ |", "|---|---|---|---|"]
    for k in keys:
        rows.append(f"| {k} | {bm[k]:.4f} | {im[k]:.4f} | {im[k]-bm[k]:+.4f} |")
    table = "\n".join(rows)
    (out / "stage2_table.md").write_text(table)
    fail, ftype = select_and_draw_failure(baseline, improved, data, split, out)
    (out / "stage2_metrics.json").write_text(json.dumps(
        {"baseline": bm, "improved": im, "failure_case": fail,
         "failure_type": ftype}, indent=2))
    print(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--baseline")
    ap.add_argument("--improved")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.compare:
        compare(args.baseline, args.improved, args.data, args.split, args.out)
    else:
        m = val_metrics(args.weights, args.data, args.split)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "metrics.json").write_text(json.dumps(m, indent=2))
        print(json.dumps(m, indent=2))
        if args.baseline is None and args.improved is None:
            # single-model failure case (improved omitted)
            select_and_draw_failure(args.weights, None, args.data, args.split, args.out)


if __name__ == "__main__":
    main()
