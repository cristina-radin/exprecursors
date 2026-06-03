#!/usr/bin/env python
"""
Poster-ready 3-window attention bar chart.
Divides the 60-day attention weights into early/mid/late segments
and shows mean weight per segment for MHW vs non-MHW events.
No retraining needed — uses existing model weights.

Usage:
  python eval/poster_attention_3win.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz      split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output   poster_figures/fig_attention_3win.png
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
from xai.run_xai import collect_predictions, top_indices_for_period

FONTSIZE  = 22
TITLESIZE = 24
TICKSIZE  = 18

WINDOWS = [
    ("Early\n(day −60 to −41)", slice(0,  20)),
    ("Mid\n(day −40 to −21)",   slice(20, 40)),
    ("Recent\n(day −20 to −1)", slice(40, 60)),
]


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


def sep_indices(scores, n, sep=60):
    order = np.argsort(scores)[::-1]
    sel = []
    for i in order:
        if all(abs(i - j) >= sep for j in sel):
            sel.append(i)
        if len(sel) == n:
            break
    return np.array(sel)


@torch.no_grad()
def get_attn_windows(lm, dataset, indices, device):
    """Returns (n_events, 3) — mean attention weight per window segment."""
    result = []
    for idx in indices:
        xs, xt, _ = dataset[idx]
        _, attn = lm.model.forward_with_attention(
            xs.unsqueeze(0).to(device),
            xt.unsqueeze(0).to(device),
        )
        w = attn.squeeze(0).cpu().numpy()   # (60,)
        result.append([w[s].sum() for _, s in WINDOWS])
    return np.array(result)   # (n_events, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--npz",      required=True)
    parser.add_argument("--output",   default="poster_figures/fig_attention_3win.png")
    parser.add_argument("--n_events", type=int, default=50)
    parser.add_argument("--no_cuda",  action="store_true")
    args = parser.parse_args()

    device   = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    exp_dirs = [Path(d) for d in args.exp_dirs]

    d        = np.load(args.npz, allow_pickle=True)
    ens_pred = d["ens_full"]
    trues    = d["trues"]

    n = args.n_events
    mhw_score    = ens_pred * (trues > 0).astype(float)
    nonmhw_score = -ens_pred * (trues <= 0).astype(float)
    idx_mhw    = sep_indices(mhw_score,    n, sep=60)
    idx_nonmhw = sep_indices(nonmhw_score, n, sep=60)

    # Collect per-seed window weights
    mhw_seeds, nonmhw_seeds = [], []
    for exp_dir in exp_dirs:
        config  = load_config(str(exp_dir / "config.yaml"))
        lm      = load_model(best_checkpoint(exp_dir), config, device)
        dm      = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset
        seed    = config.get("seed", "?")
        print(f"  seed={seed}...")
        mhw_seeds.append(get_attn_windows(lm, full_ds, idx_mhw,    device))
        nonmhw_seeds.append(get_attn_windows(lm, full_ds, idx_nonmhw, device))

    # (n_seeds * n_events, 3)
    mhw_all    = np.concatenate(mhw_seeds,    axis=0)
    nonmhw_all = np.concatenate(nonmhw_seeds, axis=0)

    labels   = [name for name, _ in WINDOWS]
    x        = np.arange(len(labels))
    width    = 0.35
    uniform  = 20 / 60   # each window = 20/60 days

    mhw_mean    = mhw_all.mean(axis=0)
    mhw_std     = mhw_all.std(axis=0)
    nonmhw_mean = nonmhw_all.mean(axis=0)
    nonmhw_std  = nonmhw_all.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(x - width/2, mhw_mean,    width, yerr=mhw_std,    label="MHW events",
           color="#d7191c", alpha=0.85, capsize=6,
           error_kw=dict(elinewidth=2, ecolor="dimgray"))
    ax.bar(x + width/2, nonmhw_mean, width, yerr=nonmhw_std, label="Non-MHW events",
           color="#2c7bb6", alpha=0.85, capsize=6,
           error_kw=dict(elinewidth=2, ecolor="dimgray"))

    ax.axhline(uniform, color="gray", lw=1.5, ls="--", label="Uniform (1/3)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICKSIZE)
    ax.set_ylabel("Total attention weight", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICKSIZE - 2)
    ax.legend(fontsize=FONTSIZE - 2, framealpha=0.9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    n_seeds = len(exp_dirs)
    ax.set_title(
        f"Temporal attention — 20-day segments\n"
        f"Ensemble of {n_seeds} seeds  ·  top-{n} events each  ·  7-day lead",
        fontsize=TITLESIZE, fontweight="bold", pad=10,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
