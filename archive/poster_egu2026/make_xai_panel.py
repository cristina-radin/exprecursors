#!/usr/bin/env python
"""
Composite XAI panel for poster:
  Row 1: IG spatial attribution — ptho_bot (3 periods)
  Row 2: IG spatial attribution — u10     (3 periods)

Usage:
  python eval/make_xai_panel.py --output_dir poster_figures
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
sys.path.append(".")
from xai.utils import load_config

BASE    = Path("split_blockyear/TbotAtm")
CKPT    = "split_blockyear/TbotAtm/checkpoints/cnn-lstm-epoch=26-val_loss=0.5301.ckpt"
CONFIG  = "split_blockyear/TbotAtm/config.yaml"
PERIODS = [("1985", "2004"), ("2005", "2014"), ("2015", "2024")]
VARS    = ["ptho_bot", "u10", "v10", "msl", "ssr"]

VAR_LABELS = {
    "ptho_bot": "Bottom temperature  (T$_{bot}$)",
    "u10":      "Zonal wind  (u10)",
    "v10":      "Meridional wind  (v10)",
    "msl":      "Sea level pressure  (msl)",
    "ssr":      "Solar radiation  (ssr)",
}

PERIOD_LABELS = ["1985–2004", "2005–2014", "2015–2024"]


def load_ig_arrays():
    """Load raw IG attributions by running inference."""
    import torch
    from model import CNNLightningModule, CNNLSTMModel
    from datamodule import LazyDataModule

    config = load_config(CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    from xai.integrated_gradients import IntegratedGradients
    ig = IntegratedGradients(lm.model)

    preds, years = [], []
    with torch.no_grad():
        for idx in range(len(full_ds)):
            xs, xt, y = full_ds[idx]
            p, _ = lm.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device))
            preds.append(p.item())
            t = idx + full_ds.window_size - 1 + full_ds.lead_time
            years.append(int(full_ds.years[t]))

    preds = np.array(preds)
    years = np.array(years)

    # For each period, pick top-N high events and average their IG maps
    results = {}  # (period_label, var) -> 2D array
    n_top = 30

    for (y0, y1), label in zip(PERIODS, PERIOD_LABELS):
        mask = (years >= int(y0)) & (years <= int(y1))
        idx_period = np.where(mask)[0]
        top_n = idx_period[np.argsort(preds[idx_period])[-n_top:]]

        ig_sum = None
        count = 0
        for idx in top_n:
            xs, xt, y = full_ds[idx]
            xs_t = xs.unsqueeze(0).float().to(device).requires_grad_(True)
            xt_t = xt.unsqueeze(0).float().to(device)

            attrs = ig.attribute(xs_t, xt_t, n_steps=30)
            # attrs: (1, window, n_vars, lat, lon) → mean over window, abs
            a = attrs.squeeze(0).abs().mean(dim=0).detach().cpu().numpy()  # (n_vars, lat, lon)
            if ig_sum is None:
                ig_sum = a
            else:
                ig_sum += a
            count += 1

        ig_mean = ig_sum / count  # (n_vars, lat, lon)
        for vi, var in enumerate(VARS):
            results[(label, var)] = ig_mean[vi]

    return results


def make_panel_from_saved(xai_dir, vars_to_show, out_dir):
    """Assemble composite figure from already-saved ig_spatial_*.npy if available,
    otherwise from saved PNG files (less ideal but fast)."""
    import matplotlib.image as mpimg

    periods_dirs = {
        "1985–2004": xai_dir / "1985-2004",
        "2005–2014": xai_dir / "2005-2014",
        "2015–2024": xai_dir / "2015-2024",
    }

    nrows = len(vars_to_show)
    ncols = len(PERIOD_LABELS)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for ri, var in enumerate(vars_to_show):
        for ci, (label, pdir) in enumerate(periods_dirs.items()):
            ax = axes[ri, ci]
            npy_path = pdir / f"ig_spatial_{var}.npy"
            png_path = pdir / f"ig_spatial_{var}.png"

            if npy_path.exists():
                data = np.load(npy_path)
                # data: (lat, lon)
                im = ax.imshow(data, origin="lower", cmap="YlOrRd", aspect="auto")
                plt.colorbar(im, ax=ax, fraction=0.03, label="|IG|")
            elif png_path.exists():
                img = mpimg.imread(png_path)
                ax.imshow(img)
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")

            if ri == 0:
                ax.set_title(label, fontsize=13, fontweight="bold", pad=8)
            if ci == 0 and npy_path.exists():
                ax.set_ylabel(VAR_LABELS.get(var, var), fontsize=12)

    fig.suptitle("Integrated Gradients — spatial attribution for MHW events\n"
                 f"Top-30 high-anomaly events per period  |  TbotAtm model (7-day lead)",
                 fontsize=13, y=1.01)

    out = out_dir / "fig_xai_ig_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default="poster_figures")
    parser.add_argument("--vars",        nargs="+",
                        default=["ptho_bot", "u10", "ssr"],
                        help="Variables to show (rows)")
    parser.add_argument("--xai_dir",     default="split_blockyear/TbotAtm/xai_results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_panel_from_saved(Path(args.xai_dir), args.vars, out_dir)


if __name__ == "__main__":
    main()
