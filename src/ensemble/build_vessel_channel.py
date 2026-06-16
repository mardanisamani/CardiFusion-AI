# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Phase 6 — build the ensemble input dataset.

Run the improved U-Net over every stenosis image to get a vessel mask, then form
a 3-channel composite that injects the vessel prior into the detector while
staying compatible with stock Ultralytics (no architecture surgery, no 4th
weight file):

    channel 0 = enhanced grayscale (top-hat + CLAHE)   -> what the detector sees
    channel 1 = predicted vessel probability mask      -> the anatomical prior
    channel 2 = original grayscale                     -> raw signal fallback

This matches the paper's Fig. 4 "binary mask as input" idea. Labels are copied
unchanged; only the images differ, so the improved-YOLO recipe can be fine-tuned
on this input by pointing its data.yaml here.

    python -m src.ensemble.build_vessel_channel \
        --unet weights/unet_weighted_edge.pt \
        --yolo-root data/yolo --out data/yolo_ensemble
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.data.preprocess import enhance
from src.models.unet import build_unet


def load_unet(weights: Path, device: str):
    ckpt = torch.load(weights, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = build_unet(cfg.get("encoder", "resnet34"), 1, 2, encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def vessel_prob(model, gray: np.ndarray, device: str, size: int = 512) -> np.ndarray:
    enh = enhance(gray)
    x = cv2.resize(enh, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x.astype(np.float32) / 255.0)[None, None].to(device)
    prob = torch.softmax(model(t), dim=1)[0, 1].cpu().numpy()
    return (prob * 255).astype(np.uint8)


def build(unet_weights, yolo_root, out_root, device):
    model = load_unet(Path(unet_weights), device)
    yolo_root, out_root = Path(yolo_root), Path(out_root)

    for split in ("train", "val", "test"):
        img_dir = yolo_root / "images" / split
        if not img_dir.exists():
            continue
        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for p in sorted(img_dir.glob("*.png")):
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            h, w = gray.shape
            prob = vessel_prob(model, gray, device)
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
            enh = enhance(gray)
            composite = cv2.merge([enh, prob, gray])  # BGR-ordered 3-channel
            cv2.imwrite(str(out_img / p.name), composite)
            src_lbl = yolo_root / "labels" / split / (p.stem + ".txt")
            if src_lbl.exists():
                shutil.copy(src_lbl, out_lbl / src_lbl.name)
        print(f"  {split}: composites written")

    data = {
        "path": str(out_root.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "stenosis"}, "nc": 1,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"wrote {out_root/'data.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True)
    ap.add_argument("--yolo-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    build(args.unet, args.yolo_root, args.out, device)


if __name__ == "__main__":
    main()
