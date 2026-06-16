# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Evaluate a U-Net checkpoint on a split, write per-image + aggregate metrics,
auto-select the representative failure case, and (optionally) build the Stage-1
comparison table and the 4-panel failure figure.

    # per-model metrics + failure case
    python -m src.eval.eval_unet --weights weights/unet_baseline.pt \
        --val-dir data/syntax/val --out report/figures/unet_baseline

    # baseline vs weighted-edge comparison (table + 4-panel figure)
    python -m src.eval.eval_unet --compare \
        --baseline weights/unet_baseline.pt \
        --improved weights/unet_weighted_edge.pt \
        --val-dir data/syntax/val --out report/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import VesselSegDataset
from src.eval.seg_metrics import all_metrics
from src.models.unet import build_unet
from src.utils.seed import set_global_seed


def load_model(weights: Path, device: str):
    ckpt = torch.load(weights, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = build_unet(cfg.get("encoder", "resnet34"), 1, 2,
                       encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, ds, device):
    """Return dict sid -> (image, gt, pred, metrics)."""
    out = {}
    ld = DataLoader(ds, batch_size=4, shuffle=False)
    for img, msk, sids in ld:
        logits = model(img.to(device))
        preds = logits.argmax(1).cpu().numpy()
        for i, sid in enumerate(sids):
            g = msk[i].numpy()
            p = preds[i]
            out[sid] = (img[i, 0].numpy(), g, p, all_metrics(p, g))
    return out


def aggregate(results) -> dict:
    agg = {}
    for _, _, _, m in results.values():
        for k, v in m.items():
            agg.setdefault(k, []).append(v)
    return {k: float(np.nanmean(v)) for k, v in agg.items()}


def select_failure(results) -> str:
    """Worst thin-vessel recall (ties broken by low clDice)."""
    def key(sid):
        m = results[sid][3]
        tvr = m["thin_vessel_recall"]
        tvr = 1.0 if np.isnan(tvr) else tvr
        return (tvr, m["cldice"])
    return min(results, key=key)


def eval_single(weights, val_dir, out, device):
    set_global_seed()
    ds = VesselSegDataset(val_dir, train=False)
    model = load_model(Path(weights), device)
    results = predict_all(model, ds, device)
    agg = aggregate(results)
    fail = select_failure(results)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"aggregate": agg, "failure_case": fail,
         "per_image": {k: v[3] for k, v in results.items()}}, indent=2))
    print(f"aggregate: {json.dumps(agg, indent=2)}")
    print(f"failure case: {fail}")
    return agg, fail, results


def four_panel(image, gt, base_pred, imp_pred, title, save):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a in ax:
        a.axis("off")
    ax[0].imshow(image, cmap="gray"); ax[0].set_title("Original (enhanced)")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("Ground truth")
    ax[2].imshow(base_pred, cmap="gray"); ax[2].set_title("Baseline U-Net")
    ax[3].imshow(imp_pred, cmap="gray"); ax[3].set_title("Weighted-edge U-Net")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save}")


def compare(baseline, improved, val_dir, out, device):
    set_global_seed()
    ds = VesselSegDataset(val_dir, train=False)
    base_m = load_model(Path(baseline), device)
    imp_m = load_model(Path(improved), device)
    base_res = predict_all(base_m, ds, device)
    imp_res = predict_all(imp_m, ds, device)
    base_agg, imp_agg = aggregate(base_res), aggregate(imp_res)

    fail = select_failure(base_res)  # failure picked on the baseline
    out = Path(out); out.mkdir(parents=True, exist_ok=True)

    image, gt, base_pred, _ = base_res[fail]
    _, _, imp_pred, _ = imp_res[fail]
    four_panel(image, gt, base_pred, imp_pred,
               f"Stage-1 failure case: {fail}", out / "stage1_failure_case.png")

    # Markdown comparison table.
    keys = ["dice", "iou", "cldice", "precision", "recall", "thin_vessel_recall"]
    rows = ["| Metric | Baseline U-Net | Weighted-edge U-Net | Δ |",
            "|---|---|---|---|"]
    for k in keys:
        b, im = base_agg[k], imp_agg[k]
        rows.append(f"| {k} | {b:.4f} | {im:.4f} | {im-b:+.4f} |")
    table = "\n".join(rows)
    (out / "stage1_table.md").write_text(table)
    (out / "stage1_metrics.json").write_text(json.dumps(
        {"baseline": base_agg, "weighted_edge": imp_agg, "failure_case": fail}, indent=2))
    print(table)
    print(f"failure case: {fail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--baseline")
    ap.add_argument("--improved")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.compare:
        compare(args.baseline, args.improved, args.val_dir, args.out, device)
    else:
        eval_single(args.weights, args.val_dir, args.out, device)


if __name__ == "__main__":
    main()
