#!/usr/bin/env python
"""
Spatially-selective perturbation of T_bottom per temporal bin.
Masks Gulf Stream region vs North Sea region separately for each time bin,
to test whether Gulf Stream or local ocean state drives the prediction.

Gulf Stream box : 30–50°N, 80°W–40°W
North Sea box   : 51–62.5°N, 5.2°W–13.2°E  (= target region)

For each top MHW event, seed, temporal bin, and spatial region:
  pred_drop = pred_full - pred_masked_region_in_bin

Output: bar chart with 6 groups (3 bins × 2 regions) for T_bottom only.

Usage:
  python eval/poster_perturb_spatial_temporal.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz      split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output   poster_figures/fig_perturb_spatial_temporal.png \
    --n_events 50
"""

import argparse
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))
from datamodule import LazyDataModule
from model import CNNLightningModule, CNNLSTMModel
from xai.utils import load_config

FONTSIZE  = 18
TITLESIZE = 20
TICKSIZE  = 14

BINS = [
    ("Early\n(day −60–41)",  slice(0,  20)),
    ("Mid\n(day −40–21)",    slice(20, 40)),
    ("Recent\n(day −20–1)",  slice(40, 60)),
]
BIN_COLORS = ["#2166ac", "#92c5de", "#d7191c"]

# Spatial regions
REGIONS = {
    "Gulf Stream": dict(lat_min=30,   lat_max=50,   lon_min=-80, lon_max=-40),
    "North Sea":   dict(lat_min=51,   lat_max=62.5, lon_min=-5.2, lon_max=13.2),
}
REGION_HATCHES = ["", "///"]
REGION_ALPHA   = [0.9, 0.6]


def best_checkpoint(exp_dir):
    ckpts = list((exp_dir / "checkpoints").glob("*.ckpt"))
    def val_loss(p):
        try:   return float(str(p).split("val_loss=")[1].replace(".ckpt", ""))
        except: return float("inf")
    return min(ckpts, key=val_loss)


def load_model(ckpt_path, config, device):
    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden", 256),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=config.get("temporal_features", 3),
        dropout=config.get("dropout", 0.3),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path), model=cnn_lstm, map_location=device
    )
    lm.eval().to(device)
    return lm


def sep_indices(preds, n, sep=60):
    order = np.argsort(preds)[::-1]
    sel = []
    for i in order:
        if all(abs(i - j) >= sep for j in sel):
            sel.append(i)
        if len(sel) == n:
            break
    return np.array(sel)


def make_spatial_mask(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Boolean mask (n_lat, n_lon) — True where pixels are inside the box."""
    lat_ok = (lat >= lat_min) & (lat <= lat_max)
    lon_ok = (lon >= lon_min) & (lon <= lon_max)
    return lat_ok[:, None] & lon_ok[None, :]   # (n_lat, n_lon)


@torch.no_grad()
def predict(lm, xs, xt, device):
    pred, _ = lm.model.forward_with_attention(
        xs.unsqueeze(0).float().to(device),
        xt.unsqueeze(0).float().to(device),
    )
    return pred.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--npz",      required=True)
    parser.add_argument("--output",   default="poster_figures/fig_perturb_spatial_temporal.png")
    parser.add_argument("--n_events", type=int, default=50)
    parser.add_argument("--no_cuda",  action="store_true")
    args = parser.parse_args()

    device   = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exp_dirs = [Path(d) for d in args.exp_dirs]

    config0   = load_config(str(exp_dirs[0] / "config.yaml"))
    variables = config0["variables"]
    if "ptho_bot" not in variables:
        raise ValueError("ptho_bot not in variables — nothing to do")
    tbot_idx = variables.index("ptho_bot")

    ds_xr = xr.open_dataset(config0["data_dir"])
    lat   = ds_xr.lat.values
    lon   = ds_xr.lon.values
    ds_xr.close()

    # Build spatial masks
    region_masks = {}
    for name, box in REGIONS.items():
        mask = make_spatial_mask(lat, lon, **box)
        region_masks[name] = mask
        n_pix = mask.sum()
        print(f"  {name}: {n_pix} pixels within box")

    d          = np.load(args.npz, allow_pickle=True)
    ens_pred   = d["ens_full"]
    idx_events = sep_indices(ens_pred, args.n_events, sep=60)
    print(f"Selected {len(idx_events)} top events")

    # degradation[region][bin] = mean pred drop
    region_names = list(REGIONS.keys())
    n_bins   = len(BINS)
    n_regions = len(region_names)
    degradation = np.zeros((n_regions, n_bins))
    counts = 0

    for exp_dir in exp_dirs:
        config  = load_config(str(exp_dir / "config.yaml"))
        lm      = load_model(best_checkpoint(exp_dir), config, device)
        dm      = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset
        seed    = config.get("seed", "?")
        print(f"  Seed={seed}...")

        for k, idx in enumerate(idx_events):
            xs, xt, _ = full_ds[int(idx)]
            pred_full = predict(lm, xs, xt, device)

            for r, rname in enumerate(region_names):
                mask = region_masks[rname]   # (n_lat, n_lon)
                for b, (_, sl) in enumerate(BINS):
                    xs_masked = xs.clone()
                    # Zero out tbot_idx in time bin sl, only within spatial mask
                    mask_t = torch.from_numpy(mask)
                    xs_masked[sl, tbot_idx] = xs_masked[sl, tbot_idx].masked_fill(
                        mask_t.unsqueeze(0), 0.0
                    )
                    pred_m = predict(lm, xs_masked, xt, device)
                    degradation[r, b] += pred_full - pred_m

            counts += 1
            print(f"    {k+1}/{len(idx_events)}", end="\r")
        print()

    degradation /= counts

    # ------------------------------------------------------------------ #
    # Figure: grouped bars — x=bins, groups=regions, for T_bottom
    # ------------------------------------------------------------------ #
    x     = np.arange(n_bins)
    width = 0.35
    bin_labels = [name.replace("\n", " ") for name, _ in BINS]

    fig, ax = plt.subplots(figsize=(10, 6))

    for r, rname in enumerate(region_names):
        offset = (r - 0.5) * width
        bars = ax.bar(x + offset, degradation[r], width,
                      label=rname,
                      color=[BIN_COLORS[b] for b in range(n_bins)],
                      alpha=REGION_ALPHA[r],
                      hatch=REGION_HATCHES[r],
                      edgecolor="white" if r == 0 else "gray")
        # Uniform color per region — override with single color
        color = "#1a1a2e" if r == 0 else "#e94560"
        for bar in bars:
            bar.set_facecolor(color)
            bar.set_alpha(REGION_ALPHA[r])

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=TICKSIZE)
    ax.set_ylabel("Mean T$_{bottom}$ prediction drop when region masked", fontsize=FONTSIZE - 1)
    ax.tick_params(axis="y", labelsize=TICKSIZE)
    ax.legend(fontsize=FONTSIZE, title="Masked region", title_fontsize=FONTSIZE - 2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        f"T$_{{bottom}}$ spatial–temporal perturbation\n"
        f"Ensemble of {len(exp_dirs)} seeds  ·  top-{len(idx_events)} MHW events  ·  7-day lead",
        fontsize=TITLESIZE, fontweight="bold",
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print summary table
    print(f"\n{'':16}", end="")
    for name, _ in BINS:
        print(f"  {name.split(chr(10))[0]:>12}", end="")
    print()
    for r, rname in enumerate(region_names):
        print(f"  {rname:14}", end="")
        for b in range(n_bins):
            print(f"  {degradation[r, b]:+.5f}", end="")
        print()


if __name__ == "__main__":
    main()
