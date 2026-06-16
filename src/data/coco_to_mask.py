# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Phase 1 — COCO -> binary vessel mask converter for the `syntax` sub-dataset.

For each image we take the UNION of every annotation polygon (all SYNTAX vessel
classes) into a single binary vessel mask, then resize to 512x512. This is the
U-Net target. Masks are saved as 0/255 PNGs alongside a copy of the (optionally
enhanced) input image so training reads from a flat, fast directory layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from tqdm import tqdm

IMG_SIZE = 512


def _ann_to_mask(coco: COCO, ann: dict, h: int, w: int) -> np.ndarray:
    """Rasterise one annotation (polygon or RLE) to a HxW uint8 {0,1} mask."""
    seg = ann.get("segmentation")
    if not seg:
        return np.zeros((h, w), np.uint8)
    if isinstance(seg, list):  # polygon(s)
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg["counts"], list):  # uncompressed RLE
        rle = mask_utils.frPyObjects(seg, h, w)
    else:  # compressed RLE
        rle = seg
    return mask_utils.decode(rle).astype(np.uint8)


def convert(
    images_dir: Path,
    ann_file: Path,
    out_dir: Path,
    img_size: int = IMG_SIZE,
) -> int:
    coco = COCO(str(ann_file))
    out_img = out_dir / "images"
    out_msk = out_dir / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_id in tqdm(coco.getImgIds(), desc=f"masks:{ann_file.parent.name}"):
        info = coco.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]
        fname = info["file_name"]
        img_path = images_dir / fname
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARN missing image {img_path}; skipping")
            continue

        mask = np.zeros((h, w), np.uint8)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            mask |= _ann_to_mask(coco, ann, h, w)

        img_r = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        msk_r = cv2.resize(
            mask * 255, (img_size, img_size), interpolation=cv2.INTER_NEAREST
        )
        stem = Path(fname).stem
        cv2.imwrite(str(out_img / f"{stem}.png"), img_r)
        cv2.imwrite(str(out_msk / f"{stem}.png"), msk_r)
        n += 1
    print(f"wrote {n} image/mask pairs to {out_dir}")
    return n


def main():
    ap = argparse.ArgumentParser(description="COCO syntax -> binary vessel masks")
    ap.add_argument("--images", required=True, type=Path, help="dir of source images")
    ap.add_argument("--ann", required=True, type=Path, help="COCO json annotation file")
    ap.add_argument("--out", required=True, type=Path, help="output dir (images/ masks/)")
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    args = ap.parse_args()
    convert(args.images, args.ann, args.out, args.size)


if __name__ == "__main__":
    main()
