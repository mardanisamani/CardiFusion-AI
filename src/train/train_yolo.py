# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Phase 4 & 5 — train YOLOv8 for stenosis detection via Ultralytics.

One script trains both the baseline and the improved model; the YAML config
carries every hyper-parameter so the only difference between runs is the single
justified change (augmentation knobs and/or enhanced inputs). `close_mosaic`
turns mosaic off for the last N epochs as the brief requests.

    python -m src.train.train_yolo --config configs/yolo_baseline.yaml
    python -m src.train.train_yolo --config configs/yolo_improved.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"

from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.manifest import record_run
from src.utils.seed import set_global_seed

# Keys we forward straight to Ultralytics' trainer.
_PASSTHROUGH = [
    "epochs", "imgsz", "batch", "optimizer", "lr0", "lrf", "patience",
    "close_mosaic", "mosaic", "mixup", "copy_paste", "degrees", "translate",
    "scale", "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s",
    "hsv_v", "erasing", "cos_lr", "weight_decay",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)

    seed = set_global_seed(cfg.get("seed", 1337))
    record_run({"experiment": cfg.get("name"), "config": str(args.config)})

    model = YOLO(cfg.get("model", "yolov8m.pt"))  # COCO-pretrained start
    train_kwargs = {k: cfg[k] for k in _PASSTHROUGH if k in cfg}
    project = cfg.get("project", "yolo_runs").replace("/", "_")
    results = model.train(
        data=cfg.data_yaml,
        seed=seed,
        deterministic=True,
        project=project,
        name=cfg.get("name", "yolo"),
        exist_ok=True,
        **train_kwargs,
    )

    # Copy the best checkpoint into the submission weights slot.
    best = Path(results.save_dir) / "weights" / "best.pt"
    dst = Path(cfg.weights_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy(best, dst)
        print(f"copied {best} -> {dst}")
    else:
        print(f"WARN: expected best.pt at {best} not found")


if __name__ == "__main__":
    main()
