#!/usr/bin/env python
"""
Poster-ready IG temporal profile per variable.
Uses _integrated_gradients which returns (window, n_vars, lat, lon).
Averages over (lat, lon) → (window, n_vars), then groups into 3 windows.
Shows: for each variable, WHEN in the 60-day window does it matter most?

Usage:
  python eval/poster_ig_temporal.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz      split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output   poster_figures/fig_ig_temporal.png \
    --n_ig     50
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))
from datamodule import LazyDataModule
from model import CNNLightningModule, CNNLSTMModel
from xai.utils import load_config
from xai.integrated_gradients import _integrated_gradients
from xai.run_xai import collect_predictions, top_indices_for_period

FONTSIZE  = 20
TITLESIZE = 22
TICKSIZE  = 16

VAR_LABELS = {
    "ptho_bot": "T$_{bottom}$",
    "to_anom":  "SST anom.",
    "u10":      "U-wind",
    "v10":      "V-wind",
    "msl":      "SLP",
    "ssr":      "Solar rad.",
}

WINDOWS = [
    ("Early\n(day −60–41)",  slice(0,  20)),
    ("Mid\n(day −40–21)",    slice(20, 40)),
    ("Recent\n(day −20–1)",  slice(40, 60)),
]

WIN_COLORS = ["#2166ac", "#92c5de", "#d7191c"]   # early=dark blue, mid=light blue, recent=red


def best_checkpoint(exp_dir):
    ckpts = list((exp_dir / "checkpoints").glob("*.ckpt"))
    def val_loss(p):
        try: return float(str(p).split("val_loss=")[1].replace(".ckpt",""))
        except: return float("inf")
    return min(ckpts, key=val_loss)


def load_model(ckpt_path, config, device):
    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden", 256),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=3,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--npz",      required=True)
    parser.add_argument("--output",   default="poster_figures/fig_ig_temporal.png")
    parser.add_argument("--n_ig",     type=int, default=50)
    parser.add_argument("--no_cuda",  action="store_true")
    args = parser.parse_args()

    device   = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exp_dirs = [Path(d) for d in args.exp_dirs]

    config0  = load_config(str(exp_dirs[0] / "config.yaml"))
    variables = config0["variables"]
    n_vars    = len(variables)

    d        = np.load(args.npz, allow_pickle=True)
    ens_pred = d["ens_full"]
    years    = d["years"]
    idx_events = sep_indices(ens_pred, args.n_ig, sep=60)
    print(f"Selected {len(idx_events)} top events")

    # Collect (window, n_vars) IG profiles per seed
    # temporal_stack: list of (n_events, window, n_vars) — one per seed
    temporal_stack = []

    for exp_dir in exp_dirs:
        config  = load_config(str(exp_dir / "config.yaml"))
        lm      = load_model(best_checkpoint(exp_dir), config, device)
        dm      = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset
        seed    = config.get("seed", "?")
        print(f"  IG seed={seed}...")

        event_profiles = []
        for k, idx in enumerate(idx_events):
            xs, xt, _ = full_ds[idx]
            attrs = _integrated_gradients(lm,
                                          xs.unsqueeze(0).to(device),
                                          xt.unsqueeze(0).to(device))
            # attrs: (window, n_vars, lat, lon) → mean over (lat, lon)
            profile = attrs.abs().mean(dim=(2, 3)).numpy()   # (window, n_vars)
            event_profiles.append(profile)
            print(f"    {k+1}/{len(idx_events)}", end="\r")
        print()
        temporal_stack.append(np.stack(event_profiles))     # (n_events, window, n_vars)

    # (n_seeds, n_events, window, n_vars) → (n_seeds*n_events, window, n_vars)
    all_profiles = np.concatenate(temporal_stack, axis=0)

    # Aggregate into 3 windows: (n_samples, 3, n_vars)
    win_data = np.stack(
        [all_profiles[:, s, :].mean(axis=1) for _, s in WINDOWS],
        axis=1
    )   # (n_samples, 3, n_vars)

    win_mean = win_data.mean(axis=0)   # (3, n_vars)
    win_std  = win_data.std(axis=0)

    # --- Plot: grouped bars, x=variables, groups=windows ---
    n_wins = len(WINDOWS)
    x      = np.arange(n_vars)
    width  = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for w_idx, (win_name, _) in enumerate(WINDOWS):
        offset = (w_idx - 1) * width
        ax.bar(x + offset, win_mean[w_idx], width,
               yerr=win_std[w_idx],
               label=win_name.replace("\n", " "),
               color=WIN_COLORS[w_idx],
               alpha=0.85, capsize=5,
               error_kw=dict(elinewidth=1.5, ecolor="dimgray"))

    var_labels = [VAR_LABELS.get(v, v) for v in variables]
    ax.set_xticks(x)
    ax.set_xticklabels(var_labels, fontsize=TICKSIZE)
    ax.set_ylabel("Mean |IG attribution|", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICKSIZE - 2)
    ax.legend(fontsize=FONTSIZE - 2, framealpha=0.9, title="Window", title_fontsize=FONTSIZE - 4)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    n_seeds = len(exp_dirs)
    ax.set_title(
        f"When does each variable matter? — Integrated Gradients\n"
        f"Ensemble of {n_seeds} seeds  ·  top-{len(idx_events)} MHW events  ·  7-day lead",
        fontsize=TITLESIZE, fontweight="bold", pad=10,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print summary
    print("\nSummary (mean |IG| per window per variable):")
    print(f"{'':12}", end="")
    for name, _ in WINDOWS:
        print(f"  {name.split(chr(10))[0]:>10}", end="")
    print()
    for i, var in enumerate(variables):
        print(f"  {var:10}", end="")
        for w in range(n_wins):
            print(f"  {win_mean[w, i]:.5f}", end="")
        print()


if __name__ == "__main__":
    main()
