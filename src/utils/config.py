# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Tiny YAML config loader with attribute access."""
from __future__ import annotations

from pathlib import Path

import yaml


class Config(dict):
    """dict that also supports attribute access (cfg.lr as well as cfg['lr'])."""

    __getattr__ = dict.get

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)
