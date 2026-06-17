# AI-assisted: package init generated with Claude (Anthropic) for the ARCADE interview project.

# PyTorch >= 2.6 changed torch.load to weights_only=True by default, which breaks
# ultralytics checkpoint loading. Patch it once here so all scripts are covered.
import torch as _torch
_orig_load = _torch.load
def _patched_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(f, *args, **kwargs)
_torch.load = _patched_load
