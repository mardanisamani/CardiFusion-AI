# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""U-Net factory.

The paper uses a Residual U-Net. We use segmentation-models-pytorch's U-Net with
a ResNet-34 encoder: ResNet residual blocks in the encoder make this a faithful,
well-justified stand-in for the paper's Residual U-Net while remaining a single
pinned dependency. 1-channel input (grayscale XCA), 2-class output (bg, vessel).
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    encoder_name: str = "resnet34",
    in_channels: int = 1,
    classes: int = 2,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,  # set None for a from-scratch ablation
        in_channels=in_channels,
        classes=classes,
    )
