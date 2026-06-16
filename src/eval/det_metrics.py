# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Detection-metric helpers for the single `stenosis` class.

We read the standard Ultralytics validation results (box.mp/box.mr/box.map50/
box.map) and add F1. We emphasise recall/F1 in the report because in stenosis
screening a false negative (missed lesion) is clinically costlier than a false
alarm.
"""
from __future__ import annotations

import numpy as np


def f1_from_pr(precision: float, recall: float, eps: float = 1e-9) -> float:
    return float(2 * precision * recall / (precision + recall + eps))


def summarize_val_metrics(results) -> dict:
    """Pull P/R/mAP from an Ultralytics DetMetrics object (single class)."""
    box = results.box
    precision = float(np.atleast_1d(box.mp)[0]) if hasattr(box, "mp") else float(box.p.mean())
    recall = float(np.atleast_1d(box.mr)[0]) if hasattr(box, "mr") else float(box.r.mean())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_from_pr(precision, recall),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def classify_failure(n_gt: int, n_pred: int, confs: np.ndarray):
    """Heuristic failure scorer for auto-selecting one representative case.

    Lower score = worse / more interesting failure. Returns (score, type).
    - missed lesion: GT present, nothing predicted -> most clinically severe.
    - false positive: predictions but no GT (e.g. catheter) -> next.
    - low-confidence / count-mismatch otherwise.
    """
    if n_gt > 0 and n_pred == 0:
        return -10.0 - n_gt, "missed_lesion (false negative)"
    if n_gt == 0 and n_pred > 0:
        return -5.0 - n_pred, "false_positive (e.g. catheter/artifact)"
    if n_gt > 0 and n_pred > 0:
        mean_conf = float(confs.mean()) if len(confs) else 0.0
        mismatch = abs(n_gt - n_pred)
        # low confidence and/or count mismatch -> localisation/duplication issue
        return mean_conf - mismatch, "low_confidence / count_mismatch"
    return 100.0, "none"
