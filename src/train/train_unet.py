# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Phase 2 & 3 — train the U-Net.

One script trains both models; the config decides whether the weighted-edge loss
is active. Everything else (architecture, optimiser, schedule, augmentation) is
held identical so the baseline-vs-weighted-edge comparison is controlled.

    python -m src.train.train_unet --config configs/unet_baseline.yaml
    python -m src.train.train_unet --config configs/unet_weighted_edge.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VesselSegDataset
from src.eval.seg_metrics import all_metrics
from src.models.edge import weight_map_from_mask
from src.models.losses import DiceCELoss
from src.models.unet import build_unet
from src.utils.config import load_config
from src.utils.manifest import record_run
from src.utils.seed import seed_worker, set_global_seed


def build_weight_batch(masks: torch.Tensor, lam: float, dilate_px: int,
                       device) -> torch.Tensor:
    """masks: B,H,W long -> per-pixel weight maps B,H,W float on device."""
    w = np.stack([
        weight_map_from_mask(m.cpu().numpy(), lam=lam, dilate_px=dilate_px)
        for m in masks
    ])
    return torch.from_numpy(w).to(device)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    agg: dict[str, list] = {}
    for img, msk, _ in loader:
        img = img.to(device)
        logits = model(img)
        pred = logits.argmax(1).cpu().numpy()
        gt = msk.numpy()
        for p, g in zip(pred, gt):
            for k, v in all_metrics(p, g).items():
                agg.setdefault(k, []).append(v)
    return {k: float(np.nanmean(v)) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)

    seed = set_global_seed(cfg.get("seed", 1337))
    record_run({"experiment": cfg.get("name"), "config": str(args.config)})
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[train_unet] device={device} seed={seed} exp={cfg.get('name')}")

    train_ds = VesselSegDataset(cfg.train_dir, cfg.img_size, train=True)
    val_ds = VesselSegDataset(cfg.val_dir, cfg.img_size, train=False)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.get("num_workers", 4),
                          worker_init_fn=seed_worker, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.get("num_workers", 4))

    model = build_unet(cfg.get("encoder", "resnet34"), in_channels=1, classes=2,
                       encoder_weights=cfg.get("encoder_weights", "imagenet")).to(device)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 100))

    use_we = bool(cfg.get("weighted_edge", False))
    lam = cfg.get("lambda", 6.0)
    dilate = cfg.get("edge_dilate_px", 2)

    out_weights = Path(cfg.weights_path)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_weights.with_suffix(".metrics.json")

    best_dice, best_epoch, patience = -1.0, -1, cfg.get("patience", 20)
    history = []
    for epoch in range(cfg.get("epochs", 100)):
        model.train()
        running = 0.0
        for img, msk, _ in tqdm(train_ld, desc=f"epoch {epoch}"):
            img, msk = img.to(device), msk.to(device)
            pw = build_weight_batch(msk, lam, dilate, device) if use_we else None
            optimizer.zero_grad()
            loss = criterion(model(img), msk, pixel_weight=pw)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        val = evaluate(model, val_ld, device)
        val["train_loss"] = running / max(1, len(train_ld))
        val["epoch"] = epoch
        history.append(val)
        print(f"  epoch {epoch}: loss={val['train_loss']:.4f} "
              f"dice={val['dice']:.4f} cldice={val['cldice']:.4f} "
              f"thin_recall={val['thin_vessel_recall']:.4f}")

        if val["dice"] > best_dice:
            best_dice, best_epoch = val["dice"], epoch
            torch.save({"model": model.state_dict(), "cfg": dict(cfg),
                        "val": val, "seed": seed}, out_weights)
            print(f"  * saved best -> {out_weights} (dice={best_dice:.4f})")
        elif epoch - best_epoch >= patience:
            print(f"  early stop (no val dice gain for {patience} epochs)")
            break

    log_path.write_text(json.dumps(
        {"best_dice": best_dice, "best_epoch": best_epoch, "history": history}, indent=2))
    print(f"done. best dice={best_dice:.4f} @ epoch {best_epoch}. log -> {log_path}")


if __name__ == "__main__":
    main()
