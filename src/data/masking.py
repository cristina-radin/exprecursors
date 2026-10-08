"""
Single source of truth for the North Sea (NS) box used in spatial masking.

NS box (index space): lat[100:127], lon[150:187]
  = lat 50 to 63 N, lon 5 W to 13 E, both edges included (27 x 37 points).
  Checked against merged_daily.nc (0.5 deg grid, lat 0 to 70 N, lon 80 W to 20 E).
  Same region as the target box in src/utils/hobday.py (defined there in degrees).

Masked pixels are set to 0 in normalized space, i.e. the variable's mean value.

Used by:
  - src/models/cnn_lstm.py (CNNLSTMModel._encode, modes local_only / remote_only)
"""

import torch

NS_LAT = slice(100, 127)
NS_LON = slice(150, 187)


def mask_remote(xs: torch.Tensor) -> torch.Tensor:
    """Zero all channels inside the NS box: model sees only remote information."""
    masked = xs.clone()
    masked[:, :, :, NS_LAT, NS_LON] = 0.0
    return masked


def mask_local(xs: torch.Tensor) -> torch.Tensor:
    """Zero all channels outside the NS box: model sees only local NS information."""
    masked = torch.zeros_like(xs)
    masked[:, :, :, NS_LAT, NS_LON] = xs[:, :, :, NS_LAT, NS_LON]
    return masked
