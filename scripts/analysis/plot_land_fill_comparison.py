"""
plot_land_fill_comparison.py — visual + quantitative verification,
Aug 21 2026, of the new land_fill_mode="nearest" option in
src/data/dataset.py (user's concern: IG highlights coastlines strongly,
suspected the hard land=0 "cliff" is an artifact the CNN reacts to, not
real precursor physics -- known_issues.md's existing comment already
flagged the same suspicion before this was implemented). Compares
ptho_bot for one real day, land_fill_mode="zero" (old default) vs
"nearest" (new), to confirm the fill actually removes the flat-region
edge AND that ocean pixels are provably untouched (bit-identical), not
just visually similar.

CPU only. Not part of the permanent pipeline.

Usage:
  python scripts/analysis/plot_land_fill_comparison.py
"""

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import LazyDataset  # noqa: E402

CFG_PATH = REPO_ROOT / "configs" / "partition" / "full_gnll_quantile_v2" / "fold0.yaml"
OUT_DIR = REPO_ROOT / "experiments" / "figures" / "step_land_fill_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)
data_dir = cfg["data_dir"]

fields = {}
datasets = {}
for mode in ["zero", "nearest"]:
    cfg_mode = dict(cfg)
    cfg_mode["land_fill_mode"] = mode
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_mode, f)
        tmp_path = f.name
    print(f"\n=== land_fill_mode={mode} ===", flush=True)
    ds = LazyDataset(data_dir, config_path=tmp_path)
    day_idx = np.where((ds.years == 2014) & (ds.doys == 150))[0][0]
    fields[mode] = ds.data["ptho_bot"][day_idx].numpy().copy()
    datasets[mode] = ds
    print(
        f"  ptho_bot day field: min={np.nanmin(fields[mode]):.3f} "
        f"max={np.nanmax(fields[mode]):.3f}",
        flush=True,
    )

is_land = (
    datasets["zero"].land_masks["ptho_bot"].numpy()
)  # True=land, same geometry both modes
n_land = int(is_land.sum())
n_ocean = int((~is_land).sum())

ocean_zero = fields["zero"][~is_land]
ocean_nearest = fields["nearest"][~is_land]
land_zero = fields["zero"][is_land]
land_nearest = fields["nearest"][is_land]

print("\n=== Quantitative check ===")
print(f"n_ocean={n_ocean}  n_land={n_land}")
ocean_identical = np.array_equal(ocean_zero, ocean_nearest)
print(f"Ocean pixels bit-identical between modes: {ocean_identical}")
if not ocean_identical:
    max_diff = np.abs(ocean_zero - ocean_nearest).max()
    print(f"  MAX DIFF (should be 0!): {max_diff}")
print(
    f"land pixels, mode=zero:    min={np.nanmin(land_zero):.4f} max={np.nanmax(land_zero):.4f} (should be ~0, the pre-normalization NaN->0 fill applies at __getitem__, not here -- raw values shown)"
)
print(
    f"land pixels, mode=nearest: min={land_nearest.min():.4f} max={land_nearest.max():.4f} std={land_nearest.std():.4f} (should span a real range, not be flat)"
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmax = np.nanpercentile(np.abs(fields["nearest"]), 98)

for ax, mode in zip(axes[:2], ["zero", "nearest"]):
    field_show = np.nan_to_num(fields[mode], nan=0.0)
    im = ax.imshow(field_show, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_title(f"ptho_bot, 2014 doy150 — land_fill_mode='{mode}'")
    plt.colorbar(im, ax=ax, shrink=0.7)

diff = np.nan_to_num(fields["nearest"], nan=0.0) - np.nan_to_num(
    fields["zero"], nan=0.0
)
im2 = axes[2].imshow(diff, cmap="PuOr", origin="lower")
axes[2].set_title(
    "Difference (nearest - zero)\nshould be 0 over ocean, nonzero only over land"
)
plt.colorbar(im2, ax=axes[2], shrink=0.7)

plt.tight_layout()
out_path = OUT_DIR / "land_fill_comparison_ptho_bot_2014doy150.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}", flush=True)

diff_ocean_max = np.abs(diff[~is_land]).max()
print(
    f"\nFINAL CHECK: max |diff| over ocean pixels = {diff_ocean_max} (must be exactly 0.0)"
)
assert (
    diff_ocean_max == 0.0
), "land_fill_mode changed OCEAN values -- this must never happen!"
print("PASSED: ocean values completely untouched by land_fill_mode.")
