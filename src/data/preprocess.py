# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Paper-faithful XCA enhancement (Popov et al., 2024, "Deep learning model
training"): white top-hat (50x50) on the image to sharpen vessel edges, rescale
to 0-255, then CLAHE (grid 8x8, clip limit 2).

We follow the paper literally. The white top-hat enhances bright-on-dark
structures; XCA vessels are dark-on-bright, so we operate on the photographic
negative (255 - img) and invert back, which is the standard reading of the
paper's intent and matches Fig. 4b's "contrast- and sharpness-enhanced" output.
The `negative` flag exposes this assumption rather than hiding it.
"""
from __future__ import annotations

import cv2
import numpy as np

TOPHAT_KERNEL = 50  # paper: 50 x 50 kernel of ones
CLAHE_CLIP = 2.0    # paper: clip limit 2
CLAHE_GRID = 8      # paper: 8 x 8 grid


def to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 image."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def enhance(
    img: np.ndarray,
    tophat_kernel: int = TOPHAT_KERNEL,
    clahe_clip: float = CLAHE_CLIP,
    clahe_grid: int = CLAHE_GRID,
    negative: bool = True,
) -> np.ndarray:
    """Apply paper preprocessing. Input may be gray/BGR; output is uint8 gray.

    Steps: (optional negative) -> white top-hat -> add back & clip -> rescale
    0-255 -> CLAHE.
    """
    gray = to_gray_uint8(img)
    work = (255 - gray) if negative else gray

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (tophat_kernel, tophat_kernel)
    )
    tophat = cv2.morphologyEx(work, cv2.MORPH_TOPHAT, kernel)

    # The top-hat isolates fine bright structures; recombine to boost edges,
    # then rescale to the full 0-255 range before histogram equalisation.
    boosted = cv2.add(work, tophat)
    boosted = cv2.normalize(boosted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    out = clahe.apply(boosted)

    if negative:
        out = 255 - out
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "enhanced.png"
    raw = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(dst, enhance(raw))
    print(f"wrote {dst}")
