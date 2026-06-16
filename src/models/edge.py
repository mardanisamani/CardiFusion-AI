# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Edge-mask generation for the weighted-edge loss (Popov et al., 2024).

Paper recipe, applied per ground-truth mask:
  1. gx, gy = gradient(mask); edge = gx**2 + gy**2   (squared & summed gradients)
  2. take the gradient again -> highlights the vessel edges
  3. normalise to [0, 1]; optionally dilate a few px so the weight band has width

The result is a per-pixel edge map in [0,1] used to build the loss weight
`w = 1 + lambda * edge_norm`.
"""
from __future__ import annotations

import cv2
import numpy as np


def edge_map_from_mask(
    mask: np.ndarray,
    dilate_px: int = 2,
) -> np.ndarray:
    """mask: HxW in {0,1} or {0,255}. Returns HxW float32 edge map in [0,1]."""
    m = (mask > 0).astype(np.float32)

    # Step 1: squared+summed first gradients.
    gx, gy = np.gradient(m)
    edge = gx ** 2 + gy ** 2

    # Step 2: gradient again to emphasise the boundary.
    g2x, g2y = np.gradient(edge)
    edge2 = np.sqrt(g2x ** 2 + g2y ** 2)

    # Combine both responses; the first-order term carries most of the signal.
    combined = edge + edge2

    # Step 3: normalise to [0,1].
    mx = combined.max()
    if mx > 0:
        combined = combined / mx

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        combined = cv2.dilate(combined, k)
        mx = combined.max()
        if mx > 0:
            combined = combined / mx
    return combined.astype(np.float32)


def weight_map_from_mask(
    mask: np.ndarray,
    lam: float = 6.0,
    dilate_px: int = 2,
) -> np.ndarray:
    """Per-pixel loss weight `w = 1 + lambda * edge_norm`, HxW float32 >= 1."""
    edge = edge_map_from_mask(mask, dilate_px=dilate_px)
    return (1.0 + lam * edge).astype(np.float32)
