# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Phase 1 — COCO -> YOLO label converter for the `stenosis` sub-dataset.

Each COCO bbox (x, y, w, h in pixels) becomes a normalized YOLO line
`class x_center y_center w h`. Single class `stenosis` (id 0). We also emit the
Ultralytics `data.yaml`. Images are copied (and optionally enhanced) into the
train/val/test split folders Ultralytics expects.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import yaml
from pycocotools.coco import COCO
from tqdm import tqdm

from src.data.preprocess import enhance


def convert_split(
    images_dir: Path,
    ann_file: Path,
    out_root: Path,
    split: str,
    apply_enhance: bool = False,
    class_name: str = "stenosis",
) -> int:
    coco = COCO(str(ann_file))
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Restrict to the target class. ARCADE's stenosis json shares the full 26-
    # category schema with syntax; the stenosis boxes are category 'stenosis'.
    keep_ids = set(coco.getCatIds(catNms=[class_name])) if class_name else None
    if class_name and not keep_ids:
        print(f"  WARN category '{class_name}' not found in {ann_file.name}; "
              f"keeping all annotations")
        keep_ids = None

    n = 0
    for img_id in tqdm(coco.getImgIds(), desc=f"yolo:{split}"):
        info = coco.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]
        fname = info["file_name"]
        src = images_dir / fname
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARN missing image {src}; skipping")
            continue

        stem = Path(fname).stem
        if apply_enhance:
            img = enhance(img)
        cv2.imwrite(str(img_out / f"{stem}.png"), img)

        lines = []
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=list(keep_ids)) if keep_ids \
            else coco.getAnnIds(imgIds=img_id)
        for ann in coco.loadAnns(ann_ids):
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            # clip to [0,1] to guard against off-by-one COCO boxes
            xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
            nw, nh = min(max(nw, 0), 1), min(max(nh, 0), 1)
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
        n += 1
    print(f"  {split}: wrote {n} images + labels")
    return n


def write_data_yaml(out_root: Path) -> Path:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "stenosis"},
        "nc": 1,
    }
    p = out_root / "data.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"wrote {p}")
    return p


def main():
    ap = argparse.ArgumentParser(description="COCO stenosis -> YOLO dataset")
    ap.add_argument("--root", required=True, type=Path,
                    help="ARCADE stenosis root containing {train,val,test} folders")
    ap.add_argument("--out", required=True, type=Path, help="YOLO dataset output dir")
    ap.add_argument("--enhance", action="store_true",
                    help="apply paper top-hat+CLAHE enhancement to images")
    ap.add_argument("--class-name", default="stenosis",
                    help="COCO category to keep (others dropped); '' keeps all")
    args = ap.parse_args()

    # Expected ARCADE layout: <root>/<split>/images + <root>/<split>/annotations/<split>.json
    for split in ("train", "val", "test"):
        split_dir = args.root / split
        ann = next((split_dir / "annotations").glob("*.json"), None)
        if ann is None:
            print(f"  skip {split}: no annotation json under {split_dir/'annotations'}")
            continue
        convert_split(split_dir / "images", ann, args.out, split, args.enhance,
                      class_name=args.class_name)
    write_data_yaml(args.out)


if __name__ == "__main__":
    main()
