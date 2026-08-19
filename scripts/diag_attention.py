#!/usr/bin/env python
"""
Diagnose why temporal attention weights are flat.
Checks:
  1. LSTM hidden state variance across timesteps
  2. Raw attention scores before softmax
  3. Effect of forced attention on specific days vs learned attention

Usage:
  python eval/diag_attention.py \
    --exp_dir split_blockyear/TbotAtm_seed100 \
    --n_events 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from scripts.run_xai import collect_predictions, top_indices_for_period
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.xai.utils import load_config


def best_checkpoint(exp_dir):
    ckpts = list((exp_dir / "checkpoints").glob("*.ckpt"))

    def val_loss(p):
        try:
            return float(str(p).split("val_loss=")[1].replace(".ckpt", ""))
        except Exception:
            return float("inf")

    return min(ckpts, key=val_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--n_events", type=int, default=20)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.exp_dir)
    config = load_config(str(exp_dir / "config.yaml"))

    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden", 256),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=config.get("temporal_features", 3),
        dropout=config.get("dropout", 0.3),
        arch=config.get("arch", "lstm_only"),
        gaussian_nll=config.get("gaussian_nll", False),
        pooling=config.get("pooling", "max"),
        padding_mode=config.get("padding_mode", "zeros"),
        quantile_head=config.get("quantile_head", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(best_checkpoint(exp_dir)), model=cnn_lstm, map_location=device
    )
    lm.eval().to(device)

    dm = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
    dm.setup()
    full_ds = dm.train_dataset.dataset

    preds, _, years, _ = collect_predictions(lm, full_ds, device)
    sep = config.get("window_size", 60)
    idx_events = top_indices_for_period(
        preds, years, int(years.min()), int(years.max()), args.n_events, "high", sep
    )

    all_scores = []  # raw pre-softmax scores
    all_weights = []  # post-softmax weights
    all_lstm_stds = []  # LSTM hidden state std per timestep

    with torch.no_grad():
        for idx in idx_events:
            xs, xt, _ = full_ds[idx]
            xs = xs.unsqueeze(0).to(device)
            xt = xt.unsqueeze(0).to(device)

            batch, window, n_vars, lat, lon = xs.shape
            x_flat = xs.view(batch * window, n_vars, lat, lon)
            features = lm.model.cnn_encoder(x_flat).view(batch, window, -1)
            lstm_out, _ = lm.model.lstm(features)  # (1, window, hidden)

            scores = lm.model.attention.attn(lstm_out).squeeze()  # (window,) raw
            weights = torch.softmax(scores, dim=-1)  # (window,)

            all_scores.append(scores.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
            all_lstm_stds.append(
                lstm_out.squeeze(0).std(dim=-1).cpu().numpy()
            )  # (window,)

    all_scores = np.stack(all_scores)  # (n_events, window)
    all_weights = np.stack(all_weights)
    all_lstm_stds = np.stack(all_lstm_stds)

    window = all_scores.shape[1]
    uniform = 1.0 / window

    print("=" * 60)
    print(f"Diagnosis over {len(idx_events)} top-MHW events")
    print("=" * 60)

    print("\n── 1. LSTM hidden state variance across timesteps ──")
    mean_std_per_t = all_lstm_stds.mean(axis=0)
    print("  Mean std across hidden dim, averaged over events:")
    print(
        f"  min={mean_std_per_t.min():.4f}  max={mean_std_per_t.max():.4f}  "
        f"range={mean_std_per_t.max()-mean_std_per_t.min():.4f}"
    )
    if mean_std_per_t.max() - mean_std_per_t.min() < 0.01:
        print(
            "  ⚠ LSTM hidden states nearly identical across timesteps → collapsed LSTM"
        )
    else:
        print("  ✓ LSTM outputs vary across timesteps")

    print("\n── 2. Raw attention scores (before softmax) ──")
    score_ranges = all_scores.max(axis=1) - all_scores.min(axis=1)
    print(
        f"  Per-event score range:  mean={score_ranges.mean():.4f}  "
        f"max={score_ranges.max():.4f}  min={score_ranges.min():.4f}"
    )
    if score_ranges.mean() < 1e-3:
        print("  ⚠ Scores nearly constant → attention layer not learning")
    elif score_ranges.mean() < 0.1:
        print("  ⚠ Scores have small range → softmax collapses to near-uniform")
    else:
        print("  ✓ Scores have meaningful range")

    print("\n── 3. Post-softmax weights ──")
    weight_stds = all_weights.std(axis=1)  # std over time per event
    print(
        f"  Weight std over time:   mean={weight_stds.mean():.5f}  "
        f"(uniform would be {np.sqrt(uniform*(1-uniform)/window):.5f})"
    )
    print(
        f"  Max weight per event:   mean={all_weights.max(axis=1).mean():.4f}  "
        f"(uniform={uniform:.4f})"
    )
    if all_weights.max(axis=1).mean() < uniform * 1.5:
        print("  ⚠ Weights nearly uniform — attention is not focusing")
    else:
        print("  ✓ Attention is showing some focus")

    print("\n── 4. Forced attention test ──")
    # Compare prediction with learned weights vs forced on last 7 days vs first 7 days
    diffs_last = []
    diffs_first = []
    with torch.no_grad():
        for idx in idx_events:
            xs, xt, _ = full_ds[idx]
            xs = xs.unsqueeze(0).to(device)
            xt = xt.unsqueeze(0).to(device)

            # Normal prediction
            pred_normal = lm.model(xs, xt).item()

            # Force attention on last 7 days
            x_flat = xs.view(1 * window, n_vars, lat, lon)
            features = lm.model.cnn_encoder(x_flat).view(1, window, -1)
            lstm_out, _ = lm.model.lstm(features)

            w_last = torch.zeros(window, device=device)
            w_last[-7:] = 1 / 7
            w_first = torch.zeros(window, device=device)
            w_first[:7] = 1 / 7

            ctx_last = (lstm_out * w_last.view(1, window, 1)).sum(dim=1)
            ctx_first = (lstm_out * w_first.view(1, window, 1)).sum(dim=1)

            t_sum = xt.mean(dim=1)
            pred_last = lm.model.fc(torch.cat([ctx_last, t_sum], dim=-1)).item()
            pred_first = lm.model.fc(torch.cat([ctx_first, t_sum], dim=-1)).item()

            diffs_last.append(abs(pred_normal - pred_last))
            diffs_first.append(abs(pred_normal - pred_first))

    print(f"  |pred_normal - pred_last7days|:   mean={np.mean(diffs_last):.4f}")
    print(f"  |pred_normal - pred_first7days|:  mean={np.mean(diffs_first):.4f}")
    if np.mean(diffs_last) < 0.05 and np.mean(diffs_first) < 0.05:
        print("  → Forcing attention to ANY 7-day window barely changes prediction")
        print(
            "    Diagnosis: LSTM already integrates the full sequence; attention is decorative"
        )
    elif np.mean(diffs_last) < np.mean(diffs_first):
        print("  → Recent days matter more (last-7 closer to learned than first-7)")
    else:
        print("  → Early days matter more (first-7 closer to learned than last-7)")

    print("\n── Conclusion ──")
    print("  Score range mean:", round(score_ranges.mean(), 5))
    print(
        "  Max weight mean: ",
        round(all_weights.max(axis=1).mean(), 4),
        f"(uniform={round(uniform,4)})",
    )


if __name__ == "__main__":
    main()
