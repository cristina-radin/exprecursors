"""
Test that mask_remote and mask_local (src/data/masking.py) behave correctly.
Used by CNNLSTMModel._encode() -- see tests/test_model.py for the
mode-masking tests on the model itself.

Input tensor shape: (batch, channels, window, lat, lon) = (1, C, W, 141, 201)
NS box: lat[100:127], lon[150:187]
"""

import torch

from src.data.masking import NS_LAT, NS_LON, mask_local, mask_remote

BATCH, C, W, LAT, LON = 1, 3, 60, 141, 201


def _ones():
    return torch.ones(BATCH, C, W, LAT, LON)


# ── mask_remote ───────────────────────────────────────────────────────────────


def test_remote_zeros_ns_box():
    out = mask_remote(_ones())
    assert out[:, :, :, NS_LAT, NS_LON].abs().max().item() == 0.0


def test_remote_preserves_outside_ns():
    out = mask_remote(_ones())
    outside = out.clone()
    outside[:, :, :, NS_LAT, NS_LON] = 1.0  # ignore NS box
    assert outside.min().item() == 1.0


def test_remote_does_not_modify_input():
    xs = _ones()
    mask_remote(xs)
    assert xs.min().item() == 1.0  # original tensor unchanged (clone inside)


# ── mask_local ────────────────────────────────────────────────────────────────


def test_local_zeros_outside_ns():
    out = mask_local(_ones())
    outside = out.clone()
    outside[:, :, :, NS_LAT, NS_LON] = 0.0  # ignore NS box
    assert outside.abs().max().item() == 0.0


def test_local_preserves_ns_box():
    out = mask_local(_ones())
    assert out[:, :, :, NS_LAT, NS_LON].min().item() == 1.0


# ── consistency ───────────────────────────────────────────────────────────────


def test_remote_and_local_are_complementary():
    """remote + local masks should sum to all-ones (no pixel lost, no pixel doubled)."""
    xs = _ones()
    total = mask_remote(xs) + mask_local(xs)
    assert total.min().item() == 1.0
    assert total.max().item() == 1.0
