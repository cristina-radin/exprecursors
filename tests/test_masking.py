"""
Test that remote_only and local_only masking functions behave correctly.

Both _mask() methods use module-level constants (_NS_LAT, _NS_LON) from
train_partition.py, so no model weights are needed — instantiate with __new__.

Input tensor shape: (batch, channels, window, lat, lon) = (1, C, W, 141, 201)
NS box: lat[100:127], lon[150:187]
"""

import sys
import torch
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from partition.train_partition import (
    RemoteOnlyLightningModule,
    LocalOnlyLightningModule,
    _NS_LAT,
    _NS_LON,
)

BATCH, C, W, LAT, LON = 1, 3, 60, 141, 201


def _ones():
    return torch.ones(BATCH, C, W, LAT, LON)


def _remote():
    lm = object.__new__(RemoteOnlyLightningModule)
    return lm


def _local():
    lm = object.__new__(LocalOnlyLightningModule)
    return lm


# ── remote_only ───────────────────────────────────────────────────────────────

def test_remote_zeros_ns_box():
    out = _remote()._mask(_ones())
    assert out[:, :, :, _NS_LAT, _NS_LON].abs().max().item() == 0.0


def test_remote_preserves_outside_ns():
    out = _remote()._mask(_ones())
    outside = out.clone()
    outside[:, :, :, _NS_LAT, _NS_LON] = 1.0   # ignore NS box
    assert outside.min().item() == 1.0


def test_remote_does_not_modify_input():
    xs = _ones()
    _remote()._mask(xs)
    assert xs.min().item() == 1.0  # original tensor unchanged (clone inside _mask)


# ── local_only ────────────────────────────────────────────────────────────────

def test_local_zeros_outside_ns():
    out = _local()._mask(_ones())
    outside = out.clone()
    outside[:, :, :, _NS_LAT, _NS_LON] = 0.0   # ignore NS box
    assert outside.abs().max().item() == 0.0


def test_local_preserves_ns_box():
    out = _local()._mask(_ones())
    assert out[:, :, :, _NS_LAT, _NS_LON].min().item() == 1.0


# ── consistency ───────────────────────────────────────────────────────────────

def test_remote_and_local_are_complementary():
    """remote + local masks should sum to all-ones (no pixel lost, no pixel doubled)."""
    xs = _ones()
    r = _remote()._mask(xs)
    l = _local()._mask(xs)
    total = r + l
    assert total.min().item() == 1.0
    assert total.max().item() == 1.0
