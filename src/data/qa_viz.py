# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Phase 1 visual QA: overlay N random vessel masks and N YOLO boxes, save to
report/qa/ so the converters can be eyeballed before any training."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def qa_masks(seg_root: Path, out_dir: Path, n: int = 10, seed: int = 1337):
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in (seg_root / "images").glob("*.png"))
    random.Random(seed).shuffle(ids)
    for sid in ids[:n]:
        img = cv2.imread(str(seg_root / "images" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(seg_root / "masks" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        rgb[msk > 127] = (0, 0, 255)  # red vessel overlay
        cv2.imwrite(str(out_dir / f"mask_qa_{sid}.png"), rgb)
    print(f"wrote {min(n, len(ids))} mask QA overlays to {out_dir}")


def qa_boxes(yolo_root: Path, split: str, out_dir: Path, n: int = 10, seed: int = 1337):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = yolo_root / "images" / split
    lbl_dir = yolo_root / "labels" / split
    ids = sorted(p.stem for p in img_dir.glob("*.png"))
    random.Random(seed).shuffle(ids)
    for sid in ids[:n]:
        img = cv2.imread(str(img_dir / f"{sid}.png"))
        h, w = img.shape[:2]
        lbl = lbl_dir / f"{sid}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                if not line.strip():
                    continue
                _, xc, yc, bw, bh = map(float, line.split())
                x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"box_qa_{sid}.png"), img)
    print(f"wrote {min(n, len(ids))} box QA overlays to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Phase-1 visual QA overlays")
    ap.add_argument("--seg-root", type=Path, help="dir with images/ masks/")
    ap.add_argument("--yolo-root", type=Path, help="YOLO dataset root")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("report/qa"))
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    if args.seg_root:
        qa_masks(args.seg_root, args.out, args.n)
    if args.yolo_root:
        qa_boxes(args.yolo_root, args.split, args.out, args.n)


if __name__ == "__main__":
    main()
