# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Capture runtime provenance (versions, GPU, git commit) into run_manifest.json."""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "run_manifest.json"

_TRACKED_PKGS = [
    "torch",
    "torchvision",
    "segmentation-models-pytorch",
    "ultralytics",
    "pycocotools",
    "numpy",
    "opencv-python-headless",
    "scikit-image",
]


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def record_run(extra: dict | None = None) -> dict:
    """Update run_manifest.json runtime block and return the snapshot."""
    snapshot = {
        "git_commit": _git_commit(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": _gpu_name(),
        "versions": {p: _pkg_version(p) for p in _TRACKED_PKGS},
        "started_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        snapshot.update(extra)

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["runtime"] = snapshot
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return snapshot
