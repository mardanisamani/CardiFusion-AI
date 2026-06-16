# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Segmentation metrics for vessel masks.

Dice + IoU       — region overlap.
clDice           — topology/connectivity (Shit et al., 2021): vessels are thin
                   connected trees, so centerline overlap matters more than area.
precision/recall — pixel-level error balance.
thin_vessel_recall — recall restricted to thin structures (mask minus its
                   morphological opening), the exact failure weighted-edge targets.

All functions take binary numpy arrays (HxW, {0,1}).
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def precision_recall(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return float(precision), float(recall)


def cl_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """clDice = harmonic mean of topology precision (skel(pred) inside gt) and
    topology sensitivity (skel(gt) inside pred)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)
    tprec = (np.logical_and(skel_pred, gt).sum() + eps) / (skel_pred.sum() + eps)
    tsens = (np.logical_and(skel_gt, pred).sum() + eps) / (skel_gt.sum() + eps)
    return float(2 * tprec * tsens / (tprec + tsens + eps))


def thin_vessel_recall(pred: np.ndarray, gt: np.ndarray, kernel: int = 7,
                       eps: float = 1e-6) -> float:
    """Recall computed only on thin GT vessels = gt minus its morphological
    opening (which removes the thin branches)."""
    gt_u8 = (gt.astype(np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    opened = cv2.morphologyEx(gt_u8, cv2.MORPH_OPEN, k)
    thin = np.logical_and(gt.astype(bool), ~opened.astype(bool))
    if thin.sum() == 0:
        return float("nan")
    tp = np.logical_and(pred.astype(bool), thin).sum()
    return float((tp + eps) / (thin.sum() + eps))


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, r = precision_recall(pred, gt)
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "cldice": cl_dice(pred, gt),
        "precision": p,
        "recall": r,
        "thin_vessel_recall": thin_vessel_recall(pred, gt),
    }
