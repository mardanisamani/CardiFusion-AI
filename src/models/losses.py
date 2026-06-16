# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Segmentation losses.

`DiceCELoss`        — baseline (Phase 2): mean(Dice + CrossEntropy).
`WeightedEdgeLoss`  — Phase 3: identical Dice+CE but every per-pixel term is
                      multiplied by w = 1 + lambda*edge_norm before reduction,
                      so boundary pixels (and, per the paper, missed thin
                      vessels near them) dominate the gradient.

Keeping the two losses structurally identical except for the per-pixel weight is
what makes the baseline-vs-improved comparison clean.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_dice_per_pixel(probs_fg, target_fg, eps: float = 1e-6):
    """Return (numerator_map, denom_terms) for a weighted soft Dice.

    We expose a per-pixel formulation so the same weight map used for CE can be
    applied consistently to Dice.
    """
    inter = probs_fg * target_fg
    return inter, probs_fg, target_fg


class DiceCELoss(nn.Module):
    def __init__(self, ce_weight: torch.Tensor | None = None, dice_w: float = 1.0,
                 ce_w: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_w = dice_w
        self.ce_w = ce_w

    def forward(self, logits, target, pixel_weight=None):
        # logits: B,C,H,W ; target: B,H,W (long) ; pixel_weight: B,H,W or None
        ce_map = F.cross_entropy(
            logits, target, weight=self.ce_weight, reduction="none"
        )  # B,H,W
        probs = F.softmax(logits, dim=1)
        fg = probs[:, 1]  # B,H,W foreground prob
        tgt = (target == 1).float()

        if pixel_weight is None:
            ce = ce_map.mean()
            inter = (fg * tgt).sum(dim=(1, 2))
            denom = fg.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
            dice = 1.0 - (2 * inter + 1e-6) / (denom + 1e-6)
            dice = dice.mean()
        else:
            w = pixel_weight
            ce = (ce_map * w).sum() / (w.sum() + 1e-6)
            inter = (w * fg * tgt).sum(dim=(1, 2))
            denom = (w * fg).sum(dim=(1, 2)) + (w * tgt).sum(dim=(1, 2))
            dice = 1.0 - (2 * inter + 1e-6) / (denom + 1e-6)
            dice = dice.mean()
        return self.ce_w * ce + self.dice_w * dice


class WeightedEdgeLoss(DiceCELoss):
    """Thin wrapper: the per-pixel weight map is supplied by the training loop
    (built from edge.weight_map_from_mask). Behaviour with pixel_weight=None is
    identical to the baseline, guaranteeing a controlled comparison."""

    pass
