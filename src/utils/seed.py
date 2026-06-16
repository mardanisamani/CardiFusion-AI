# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Global seeding for reproducibility across numpy, torch, and CUDA."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 1337, deterministic: bool = True) -> int:
    """Seed every RNG we rely on. Call once at the top of each entrypoint.

    Returns the seed so callers can log it into the run manifest.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Trade a little speed for run-to-run reproducibility.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Opt into deterministic algos where available (no hard error if not).
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            except Exception:
                pass
    except ImportError:
        pass
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn so augmentation RNG is reproducible."""
    worker_seed = int((np.random.get_state()[1][0] + worker_id) % (2 ** 32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)
