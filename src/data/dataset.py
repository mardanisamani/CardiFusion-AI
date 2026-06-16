# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Torch Dataset for U-Net vessel segmentation.

Reads the (image, mask) PNG pairs produced by coco_to_mask.py, applies the
paper's top-hat+CLAHE enhancement on the fly, light geometric augmentation for
training, and returns (1xHxW float image, HxW long mask) tensors.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocess import enhance


class VesselSegDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        img_size: int = 512,
        train: bool = False,
        do_enhance: bool = True,
    ):
        root = Path(root)
        self.img_dir = root / "images"
        self.msk_dir = root / "masks"
        self.ids = sorted(p.stem for p in self.img_dir.glob("*.png"))
        if not self.ids:
            raise FileNotFoundError(f"no images under {self.img_dir}")
        self.img_size = img_size
        self.train = train
        self.do_enhance = do_enhance

    def __len__(self) -> int:
        return len(self.ids)

    def _augment(self, img: np.ndarray, msk: np.ndarray):
        # Mild, label-preserving geometric augmentation only (no intensity tricks
        # so the baseline/weighted-edge comparison stays clean).
        if np.random.rand() < 0.5:
            img, msk = np.fliplr(img).copy(), np.fliplr(msk).copy()
        if np.random.rand() < 0.5:
            img, msk = np.flipud(img).copy(), np.flipud(msk).copy()
        k = np.random.randint(0, 4)
        if k:
            img, msk = np.rot90(img, k).copy(), np.rot90(msk, k).copy()
        return img, msk

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        img = cv2.imread(str(self.img_dir / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(self.msk_dir / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        if img.shape[0] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.do_enhance:
            img = enhance(img)
        msk = (msk > 127).astype(np.uint8)

        if self.train:
            img, msk = self._augment(img, msk)

        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # 1xHxW
        msk_t = torch.from_numpy(msk.astype(np.int64))  # HxW in {0,1}
        return img_t, msk_t, sid
