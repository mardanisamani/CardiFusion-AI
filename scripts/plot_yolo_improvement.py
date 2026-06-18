"""Generate a figure showing cases where YOLOv8 baseline fails but improved succeeds.

    python scripts/plot_yolo_improvement.py \
        --baseline weights/yolov8_baseline.pt \
        --improved weights/yolov8_improved.pt \
        --data     data/yolo/data.yaml \
        --out      report/figures/yolo_improvement.png \
        --n        3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

os.environ["WANDB_MODE"] = "disabled"
_orig_load = torch.load
def _patched_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(f, *args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO


CONF_THRESH = 0.25

# ── colours (BGR for cv2, RGB for matplotlib) ────────────────────────────────
GT_COLOR      = (0.18, 0.80, 0.44)   # green
BASELINE_COLOR = (0.90, 0.22, 0.21)  # red
IMPROVED_COLOR = (0.13, 0.59, 0.95)  # blue


def _load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    """Read YOLO label file -> list of (x1, y1, x2, y2) in pixels."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        _, xc, yc, bw, bh = map(float, line.split())
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def _pred_boxes(model: YOLO, img_path: str) -> list:
    """Run inference -> list of (x1, y1, x2, y2, conf)."""
    result = model.predict(img_path, verbose=False, conf=CONF_THRESH)[0]
    out = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf)
        out.append((x1, y1, x2, y2, conf))
    return out


def _draw_boxes(ax, img_rgb: np.ndarray, gt: list, preds: list,
                pred_color, title: str) -> None:
    ax.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")
    for (x1, y1, x2, y2) in gt:
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0", linewidth=1.5,
            edgecolor=GT_COLOR, facecolor="none")
        ax.add_patch(rect)
    for (x1, y1, x2, y2, conf) in preds:
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0", linewidth=1.5,
            edgecolor=pred_color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, f"{conf:.2f}", color=pred_color,
                fontsize=7, fontweight="bold")


def find_improvement_cases(baseline: YOLO, improved: YOLO,
                           img_dir: Path, lbl_dir: Path,
                           n: int = 3) -> list:
    """Find up to n images where baseline fails and improved succeeds."""
    cases = []
    img_paths = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))

    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = _load_gt_boxes(lbl_path, w, h)

        if not gt:          # skip images with no ground-truth lesions
            continue

        base_preds  = _pred_boxes(baseline, str(img_path))
        impr_preds  = _pred_boxes(improved, str(img_path))

        baseline_misses = len(base_preds) == 0          # baseline found nothing
        improved_hits   = len(impr_preds) > 0           # improved found something

        if baseline_misses and improved_hits:
            cases.append({
                "img": img,
                "img_path": img_path,
                "gt": gt,
                "baseline": base_preds,
                "improved": impr_preds,
                "n_gt": len(gt),
            })
            if len(cases) >= n:
                break

    return cases


def plot_cases(cases: list, out: Path) -> None:
    n = len(cases)
    if n == 0:
        print("No improvement cases found — try lowering CONF_THRESH or check weights.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.25})
    if n == 1:
        axes = [axes]

    col_titles = ["Ground Truth", "YOLOv8 Baseline\n(missed)", "YOLOv8 Improved\n(detected)"]

    for row, case in enumerate(cases):
        img_rgb = cv2.cvtColor(case["img"], cv2.COLOR_BGR2RGB)
        stem    = case["img_path"].stem

        _draw_boxes(axes[row][0], img_rgb, case["gt"], [],
                    GT_COLOR, f"{col_titles[0]}\n{stem}  ({case['n_gt']} lesion{'s' if case['n_gt']>1 else ''})")
        _draw_boxes(axes[row][1], img_rgb, case["gt"], case["baseline"],
                    BASELINE_COLOR, col_titles[1])
        _draw_boxes(axes[row][2], img_rgb, case["gt"], case["improved"],
                    IMPROVED_COLOR, col_titles[2])

    # Legend
    legend_elements = [
        mpatches.Patch(edgecolor=GT_COLOR,       facecolor="none", label="Ground truth"),
        mpatches.Patch(edgecolor=BASELINE_COLOR, facecolor="none", label="Baseline prediction"),
        mpatches.Patch(edgecolor=IMPROVED_COLOR, facecolor="none", label="Improved prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("YOLOv8 Baseline vs Improved: cases where baseline misses, improved detects",
                 fontsize=11, fontweight="bold", y=1.01)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {out}  ({n} cases)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--improved", required=True, type=Path)
    ap.add_argument("--data",     required=True, type=Path,
                    help="data/yolo/data.yaml")
    ap.add_argument("--split",    default="val")
    ap.add_argument("--out",      default="report/figures/yolo_improvement.png", type=Path)
    ap.add_argument("--n",        default=3, type=int,
                    help="number of improvement cases to show")
    args = ap.parse_args()

    d = yaml.safe_load(args.data.read_text())
    root    = Path(d["path"])
    img_dir = root / d[args.split]
    lbl_dir = root / d[args.split].replace("images", "labels")

    print(f"Loading baseline: {args.baseline}")
    baseline = YOLO(str(args.baseline))
    print(f"Loading improved: {args.improved}")
    improved = YOLO(str(args.improved))

    print(f"Scanning {img_dir} for improvement cases ...")
    cases = find_improvement_cases(baseline, improved, img_dir, lbl_dir, n=args.n)
    print(f"Found {len(cases)} cases where baseline fails, improved succeeds.")
    plot_cases(cases, args.out)


if __name__ == "__main__":
    main()
