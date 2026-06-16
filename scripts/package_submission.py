# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Build submission.zip: source code + the FOUR weights + report.pdf.

Verifies the four required weights exist (warns if missing so you don't ship an
incomplete bundle) and excludes data/, runs/, and caches.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_WEIGHTS = [
    "weights/unet_baseline.pt",
    "weights/unet_weighted_edge.pt",
    "weights/yolov8_baseline.pt",
    "weights/yolov8_improved.pt",
]
INCLUDE_DIRS = ["src", "configs", "scripts", "report/figures", "report/qa"]
INCLUDE_FILES = [
    "README.md", "requirements.txt", "run_manifest.json", ".gitignore",
    "report/report.md", "report/report.pdf",
]
EXCLUDE = {"__pycache__", ".ipynb_checkpoints"}


def _add(zf, path: Path):
    if any(part in EXCLUDE for part in path.parts):
        return
    zf.write(path, path.relative_to(ROOT))


def main():
    missing = [w for w in REQUIRED_WEIGHTS if not (ROOT / w).exists()]
    if missing:
        print("WARNING — missing weights (train them before final submission):")
        for m in missing:
            print(f"  - {m}")

    out = ROOT / "submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for w in REQUIRED_WEIGHTS:
            if (ROOT / w).exists():
                _add(zf, ROOT / w)
        for d in INCLUDE_DIRS:
            for p in (ROOT / d).rglob("*"):
                if p.is_file():
                    _add(zf, p)
        for f in INCLUDE_FILES:
            if (ROOT / f).exists():
                _add(zf, ROOT / f)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB)")
    if missing:
        print("NOTE: zip built without the missing weights above.")


if __name__ == "__main__":
    main()
