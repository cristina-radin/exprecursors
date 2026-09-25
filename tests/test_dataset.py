"""
LazyDataset (src/data/dataset.py) on the synthetic dataset from
tests/synthetic.py:
  * per-variable land mask (known_issues.md #2)
  * land_fill_mode="nearest"
  * compute_stats uses ocean pixels only, in both land_fill modes (#55)
  * hobday_smooth_target (#40)
"""

import numpy as np
import pytest
import torch
import xarray as xr
from scipy.ndimage import uniform_filter1d

from src.data.dataset import LazyDataset
from tests import synthetic as syn

TRAIN_IDX = list(range(0, 1500))


def _dataset(tmp_path, data_file=syn.DATA_FILE, **overrides):
    cfg = syn.write_config(tmp_path, data_dir=str(data_file), **overrides)
    return LazyDataset(str(data_file), config_path=cfg)


def _bool_mask(land_pixels):
    m = np.zeros((syn.NLAT, syn.NLON), dtype=bool)
    for r, c in land_pixels:
        m[r, c] = True
    return m


# ── #2: per-variable land mask ──────────────────────────────────────────────


def test_ptho_bot_uses_tbottom_mask(tmp_path):
    ds = _dataset(tmp_path)
    expected = _bool_mask(syn.TBOTTOM_LAND_PIXELS)
    assert np.array_equal(ds.land_masks["ptho_bot"].numpy(), expected)
    # (2, 3) is land only in land_mask: must NOT be masked for ptho_bot
    assert not ds.land_masks["ptho_bot"][2, 3]
    # (4, 5) is land only in land_mask_tbottom: must be masked
    assert ds.land_masks["ptho_bot"][4, 5]


def test_other_ocean_variables_use_land_mask(tmp_path):
    ds = _dataset(tmp_path, ocean_variables=["ptho_bot", "u10"])
    expected = _bool_mask(syn.LAND_MASK_LAND_PIXELS)
    assert np.array_equal(ds.land_masks["u10"].numpy(), expected)
    assert np.array_equal(ds.is_land.numpy(), expected)


def test_missing_tbottom_mask_raises(tmp_path):
    with pytest.raises(ValueError, match="land_mask_tbottom"):
        _dataset(tmp_path, data_file=syn.DATA_FILE_NO_TBOTTOM)


# ── land_fill_mode ──────────────────────────────────────────────────────────


def test_invalid_land_fill_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="land_fill_mode"):
        _dataset(tmp_path, land_fill_mode="bogus")


def test_nearest_fill_copies_nearest_ocean_pixel(tmp_path):
    zero = _dataset(tmp_path, land_fill_mode="zero")
    near = _dataset(tmp_path, land_fill_mode="nearest")
    ocean = ~near.land_masks["ptho_bot"]

    # ocean pixels and non-ocean variables are untouched by the fill
    assert torch.equal(near.data["ptho_bot"][:, ocean], zero.data["ptho_bot"][:, ocean])
    assert torch.equal(near.data["u10"], zero.data["u10"])
    # (3, 0): neighbours (2, 0) and (4, 0) are land, so (3, 1) is the unique
    # nearest ocean pixel
    assert torch.equal(near.data["ptho_bot"][:, 3, 0], zero.data["ptho_bot"][:, 3, 1])
    # (4, 5): an isolated land pixel takes the value of one of its 4 neighbours
    filled = near.data["ptho_bot"][:, 4, 5]
    neighbours = [(3, 5), (5, 5), (4, 4), (4, 6)]
    assert any(torch.equal(filled, zero.data["ptho_bot"][:, r, c]) for r, c in neighbours)
    assert not torch.any(near.data["ptho_bot"] == syn.PTHO_LAND_VALUE)


def test_zero_mode_land_is_zero_in_input(tmp_path):
    ds = _dataset(tmp_path, land_fill_mode="zero")
    ds.compute_stats(TRAIN_IDX)
    x, _, _ = ds[10]
    i = ds.variables.index("ptho_bot")
    assert torch.all(x[:, i][:, ds.land_masks["ptho_bot"]] == 0)


# ── #55: normalisation statistics use ocean pixels only ─────────────────────


@pytest.mark.parametrize("mode", ["zero", "nearest"])
def test_compute_stats_uses_ocean_pixels_only(tmp_path, mode):
    ds = _dataset(tmp_path, land_fill_mode=mode)
    ds.compute_stats(TRAIN_IDX)

    raw = xr.open_dataset(syn.DATA_FILE)
    t_start, t_end = min(TRAIN_IDX), max(TRAIN_IDX) + syn.WINDOW
    ocean = ~_bool_mask(syn.TBOTTOM_LAND_PIXELS)
    vals = raw["ptho_bot"].values[t_start:t_end][:, ocean]
    i = ds.variables.index("ptho_bot")
    assert ds.input_means[i].item() == pytest.approx(float(vals.mean()), abs=1e-5)
    assert ds.input_stds[i].item() == pytest.approx(float(vals.std(ddof=1)), abs=1e-5)
    assert ds.input_stds[i].item() < 2.0  # land value 50.0 would blow this up


def test_compute_stats_identical_across_land_fill_modes(tmp_path):
    zero = _dataset(tmp_path, land_fill_mode="zero")
    near = _dataset(tmp_path, land_fill_mode="nearest")
    zero.compute_stats(TRAIN_IDX)
    near.compute_stats(TRAIN_IDX)
    assert torch.allclose(zero.input_means, near.input_means, atol=1e-6)
    assert torch.allclose(zero.input_stds, near.input_stds, atol=1e-6)


# ── #40: hobday_smooth_target ───────────────────────────────────────────────


def test_hobday_smooth_target_flag_off_leaves_target_unchanged(tmp_path):
    ds = _dataset(tmp_path, hobday_smooth_target=False)
    raw = xr.open_dataset(syn.DATA_FILE)["target"].values
    assert np.array_equal(ds.target.numpy(), raw)


def test_hobday_smooth_target_adds_unsmoothed_minus_smoothed_clim(tmp_path):
    ds = _dataset(tmp_path, hobday_smooth_target=True)
    raw = xr.open_dataset(syn.DATA_FILE)["target"].values

    clim = xr.open_dataset(syn.CLIM_FILE)
    ns = clim.mean_clim.sel(lat=slice(50.0, 63.0), lon=slice(-5.0, 13.0))
    mean_clim = ns.mean(dim=["lat", "lon"]).values
    delta = mean_clim - uniform_filter1d(mean_clim, size=31, mode="wrap")
    assert np.abs(delta).max() > 1e-3  # the synthetic seasonal cycle is not flat

    doys = ds.doys.copy()
    doys[doys >= 365] = 365  # leap day folded into 365
    expected = raw + delta[doys - 1]
    assert np.allclose(ds.target.numpy(), expected, atol=1e-5)
