#!/usr/bin/env python
"""
Poster-ready temporal attention figure.
Compares attention weight profiles for top-MHW events vs non-MHW events
across all 4 seeds, showing mean ± 1σ.

Run on CPU — no GPU needed (~3 min).

Usage:
  python eval/poster_attention.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output poster_figures/fig_attention_poster.png \
    --n_events 50
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

FONTSIZE  = 20
TITLESIZE = 22
TICKSIZE  = 16


def best_checkpoint(exp_dir: Path) -> Path:
    ckpts = list((exp_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {exp_dir}/checkpoints/")
    def val_loss(p):
        try:
            return float(str(p).split("val_loss=")[1].replace(".ckpt", ""))
        except Exception:
            return float("inf")
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
        str(ckpt_path), model=cnn_lstm, map_location=device,
    )
    lm.eval().to(device)
    return lm


@torch.no_grad()
def get_attention(lm, dataset, indices, device):
    """Return attention weights (n_events, window_size) for given indices."""
    attn_list = []
    for idx in indices:
        xs, xt, _ = dataset[idx]
        _, attn = lm.model.forward_with_attention(
            xs.unsqueeze(0).to(device),
            xt.unsqueeze(0).to(device),
        )
        attn_list.append(attn.squeeze(0).cpu().numpy())
    return np.stack(attn_list)   # (n_events, window_size)


def sep_indices(preds, n, sep=60):
    """Top-n indices with minimum separation sep."""
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
    parser.add_argument("--npz",      required=True,
                        help="predictions.npz from ensemble_skill.py")
    parser.add_argument("--output",   default="poster_figures/fig_attention_poster.png")
    parser.add_argument("--n_events", type=int, default=50)
    parser.add_argument("--no_cuda",  action="store_true")
    args = parser.parse_args()

    device   = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    exp_dirs = [Path(d) for d in args.exp_dirs]

    # Load ensemble predictions
    d        = np.load(args.npz, allow_pickle=True)
    ens_pred = d["ens_full"]    # (N,) ensemble mean prediction
    trues    = d["trues"]

    n = args.n_events
    sep = 60  # minimum days between selected events

    # Select top-MHW events (highest predicted AND observed MHW days)
    mhw_score  = ens_pred * (trues > 0).astype(float)   # high pred + true MHW
    non_mhw_score = -ens_pred * (trues <= 0).astype(float)  # low pred + true non-MHW

    idx_mhw    = sep_indices(mhw_score, n, sep)
    idx_nonmhw = sep_indices(non_mhw_score, n, sep)
    print(f"Selected {len(idx_mhw)} MHW events, {len(idx_nonmhw)} non-MHW events")

    # Load models and datasets, collect attention weights
    attn_mhw_all    = []
    attn_nonmhw_all = []

    for exp_dir in exp_dirs:
        config  = load_config(str(exp_dir / "config.yaml"))
        ckpt    = best_checkpoint(exp_dir)
        lm      = load_model(ckpt, config, device)
        dm      = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset
        seed    = config.get("seed", "?")
        print(f"  Seed {seed}: getting attention weights...")

        a_mhw    = get_attention(lm, full_ds, idx_mhw,    device)
        a_nonmhw = get_attention(lm, full_ds, idx_nonmhw, device)
        attn_mhw_all.append(a_mhw)
        attn_nonmhw_all.append(a_nonmhw)

    # Stack across seeds: (n_seeds, n_events, window_size) → (n_seeds*n_events, window_size)
    attn_mhw    = np.concatenate(attn_mhw_all,    axis=0)
    attn_nonmhw = np.concatenate(attn_nonmhw_all, axis=0)

    window = attn_mhw.shape[1]
    days   = np.arange(-window + 1, 1)   # -59 … 0

    mean_mhw     = attn_mhw.mean(axis=0)
    std_mhw      = attn_mhw.std(axis=0)
    mean_nonmhw  = attn_nonmhw.mean(axis=0)
    std_nonmhw   = attn_nonmhw.std(axis=0)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(days, mean_mhw - std_mhw,    mean_mhw + std_mhw,
                    alpha=0.25, color="#d7191c")
    ax.fill_between(days, mean_nonmhw - std_nonmhw, mean_nonmhw + std_nonmhw,
                    alpha=0.25, color="#2c7bb6")

    ax.plot(days, mean_mhw,    color="#d7191c", lw=2.5, label=f"MHW events (top-{len(idx_mhw)})")
    ax.plot(days, mean_nonmhw, color="#2c7bb6", lw=2.5,
            label=f"Non-MHW events (top-{len(idx_nonmhw)})")

    # Uniform reference line
    uniform = np.ones(window) / window
    ax.axhline(uniform[0], color="gray", lw=1.2, ls="--", label="Uniform (1/60)")

    ax.set_xlabel("Day relative to prediction target", fontsize=FONTSIZE)
    ax.set_ylabel("Attention weight", fontsize=FONTSIZE)
    ax.tick_params(labelsize=TICKSIZE)
    ax.set_xlim(days[0], days[-1])
    ax.legend(fontsize=FONTSIZE - 2, framealpha=0.9)
    ax.set_title(
        "Temporal attention — MHW vs non-MHW events\n"
        f"Ensemble of {len(exp_dirs)} seeds  ·  top-{n} events each",
        fontsize=TITLESIZE, fontweight="bold", pad=10,
    )
    ax.grid(alpha=0.25)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
