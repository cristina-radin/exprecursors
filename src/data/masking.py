"""
Single source of truth for NS-box spatial masking.

NS box (index space): lat[100:127], lon[150:187]
  ≈ lat 50–63°N, lon -5–13°E (North Sea)

Used by:
  - scripts/train_partition.py (RemoteOnlyLightningModule, LocalOnlyLightningModule)
  - scripts/eval_onset_skill.py
"""

import torch

_NS_LAT = slice(100, 127)
_NS_LON = slice(150, 187)


def mask_remote(xs: torch.Tensor) -> torch.Tensor:
    """Zero all channels inside the NS box — model sees only remote information."""
    masked = xs.clone()
    masked[:, :, :, _NS_LAT, _NS_LON] = 0.0
    return masked


def mask_local(xs: torch.Tensor) -> torch.Tensor:
    """Zero all channels outside the NS box — model sees only local NS information."""
    masked = torch.zeros_like(xs)
    masked[:, :, :, _NS_LAT, _NS_LON] = xs[:, :, :, _NS_LAT, _NS_LON]
    return masked
