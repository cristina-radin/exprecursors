#!/usr/bin/env python
"""
Recompute mean IG spatial maps per period and variable, then plot a
clean poster-quality panel with consistent colormaps.

Usage:
  python eval/make_ig_spatial_poster.py --output_dir poster_figures
"""
import argparse, sys, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model import CNNLightningModule, CNNLSTMModel
from datamodule import LazyDataModule
from xai.utils import load_config
from xai.integrated_gradients import IntegratedGradients

CKPT   = "split_blockyear/TbotAtm/checkpoints/cnn-lstm-epoch=26-val_loss=0.5301.ckpt"
CONFIG = "split_blockyear/TbotAtm/config.yaml"

PERIODS = [(1985, 2004), (2005, 2014), (2015, 2024)]
PERIOD_LABELS = ["1985–2004", "2005–2014", "2015–2024"]

VARS_SHOW = ["ptho_bot", "u10", "ssr"]
VAR_LABELS = {
    "ptho_bot": "Bottom temperature  (T$_{bot}$)",
    "u10":      "Zonal wind  (u10)",
    "ssr":      "Solar radiation  (ssr)",
}
N_TOP   = 50   # top high-anomaly events per period
N_STEPS = 50   # IG steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="poster_figures")
    args = parser.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(CONFIG)
    variables = config["variables"]

    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden", 256),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=3,
        dropout=config.get("dropout", 0.3),
    )
    lm = CNNLightningModule.load_from_checkpoint(CKPT, model=cnn_lstm, map_location=device)
    lm.eval().to(device)

    dm = LazyDataModule(config_path=CONFIG)
    dm.setup()
    full_ds = dm.train_dataset.dataset

    ig_engine = IntegratedGradients(lm.model)

    # Quick pass: collect all predictions + year
    print("Collecting predictions...")
    preds, years_all = [], []
    with torch.no_grad():
        for idx in range(len(full_ds)):
            xs, xt, _ = full_ds[idx]
            p, _ = lm.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device))
            preds.append(p.item())
            t = idx + full_ds.window_size - 1 + full_ds.lead_time
            years_all.append(int(full_ds.years[t]))
    preds = np.array(preds); years_all = np.array(years_all)
    print(f"  {len(preds)} samples")

    # Compute mean |IG| per period: shape (n_vars, lat, lon)
    var_to_idx = {v: i for i, v in enumerate(variables)}
    ig_maps = {}   # (period_label, var) -> 2D array

    for (y0, y1), label in zip(PERIODS, PERIOD_LABELS):
        print(f"Period {label} ...")
        mask = (years_all >= y0) & (years_all <= y1)
        idx_period = np.where(mask)[0]
        # greedy top-N with minimum spacing to avoid duplicate events
        sorted_idx = idx_period[np.argsort(preds[idx_period])[::-1]]
        selected, last = [], -999
        for i in sorted_idx:
            if i - last > full_ds.window_size:
                selected.append(i); last = i
            if len(selected) >= N_TOP:
                break
        print(f"  {len(selected)} events")

        ig_sum = None
        for k, idx in enumerate(selected):
            xs, xt, _ = full_ds[idx]
            xs_t = xs.unsqueeze(0).float().to(device)
            xt_t = xt.unsqueeze(0).float().to(device)
            attrs = ig_engine.attribute(xs_t, xt_t, n_steps=N_STEPS)
            # attrs: (1, window, n_vars, lat, lon) → mean |attr| over window
            a = attrs.squeeze(0).abs().mean(dim=0).detach().cpu().numpy()
            ig_sum = a if ig_sum is None else ig_sum + a
            if (k+1) % 10 == 0:
                print(f"  {k+1}/{len(selected)}", end="\r")
        print()
        ig_mean = ig_sum / len(selected)   # (n_vars, lat, lon)

        for var in VARS_SHOW:
            vi = var_to_idx[var]
            ig_maps[(label, var)] = ig_mean[vi]

    # Get lat/lon from dataset
    import xarray as xr
    ds_tmp = xr.open_dataset(config["data_dir"])
    lat = ds_tmp["lat"].values
    lon = ds_tmp["lon"].values
    ds_tmp.close()
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    # --- Plot ---
    nrows, ncols = len(VARS_SHOW), len(PERIODS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.08})
    plt.rcParams.update({"font.size": 12})

    for ri, var in enumerate(VARS_SHOW):
        # Consistent vmax across periods for this variable
        vmax = max(ig_maps[(lab, var)].max() for lab in PERIOD_LABELS)

        for ci, label in enumerate(PERIOD_LABELS):
            ax = axes[ri, ci]
            data = ig_maps[(label, var)]
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap="YlOrRd", aspect="auto",
                           vmin=0, vmax=vmax)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            if ri == 0:
                ax.set_title(label, fontsize=13, fontweight="bold", pad=6)
            if ci == 0:
                ax.set_ylabel(VAR_LABELS[var], fontsize=11)
            else:
                ax.set_yticklabels([])
            if ri < nrows - 1:
                ax.set_xticklabels([])

            # Colorbar only on last column
            if ci == ncols - 1:
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("|IG attribution|", fontsize=9)

    fig.suptitle("Integrated Gradients — mean spatial attribution for MHW events\n"
                 f"Top-{N_TOP} high-anomaly events per period  |  TbotAtm  (7-day lead)",
                 fontsize=13, y=1.01)

    out = out_dir / "fig_xai_ig_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
