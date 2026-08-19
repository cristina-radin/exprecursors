#!/usr/bin/env python
"""
eval_partition_timeseries.py — batched test-set inference for
TbotAtm_{partition}_{model_tag}_seed42_fold{0-4}, saved per-fold plus one
comparison figure of predicted vs. true to_anom across all 5 folds' test
periods (each fold's test years are a random, non-contiguous subset — see
src/data/datamodule.py "kfold" split).

Batched (vectorised) inference — no per-sample Python loop.

Usage:
  python scripts/eval_partition_timeseries.py --partition full --model_tag mse_v2
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.checkpoints import best_ckpt, load_model_config

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "partition"


def _config_subdir(partition: str, model_tag: str) -> str:
    return f"{partition}_{model_tag}" if model_tag else partition


def _run_name(partition: str, model_tag: str, fold: int) -> str:
    tag = f"_{model_tag}" if model_tag else ""
    return f"TbotAtm_{partition}{tag}_seed42_fold{fold}"


def pearson_r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def run_fold(fold: int, partition: str, model_tag: str, device: str, batch_size: int):
    cfg_path = (
        REPO_ROOT
        / "configs"
        / "partition"
        / _config_subdir(partition, model_tag)
        / f"fold{fold}.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    run_dir = EXPERIMENTS_DIR / _run_name(partition, model_tag, fold)
    # Ground truth for the architecture: run_dir/model_config.json if the
    # run wrote one (train_partition.py does, as of Aug 19 2026), else
    # derived from cfg using the same defaults train_partition.py uses —
    # single source (src/utils/checkpoints.py) instead of re-guessing here.
    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    quantile_head = model_kwargs["quantile_head"]
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    # strict=True: the config drives both training and eval construction
    # identically, so a real architecture mismatch (e.g. quantile_head not
    # wired into `cfg` here) must fail loudly instead of silently dropping
    # checkpoint weights (strict=False swallowed this for quantile_head
    # before — the head's weights were trained but never loaded/used, with
    # no error and no warning surfaced to the user).
    try:
        lm = CNNLightningModule.load_from_checkpoint(
            str(ckpt), model=inner, strict=True, map_location=device
        )
    except RuntimeError as e:
        # Only fall back for an actual state_dict key mismatch (PyTorch's
        # load_state_dict error text) — anything else (corrupted checkpoint,
        # CUDA error, etc.) is a different failure mode that strict=False
        # would not fix, and must not be swallowed here.
        if "Missing key(s)" not in str(e) and "Unexpected key(s)" not in str(e):
            raise
        print(
            f"fold{fold}: STRICT LOAD FAILED (key mismatch) — falling back to "
            f"strict=False. Some checkpoint weights will NOT be loaded:\n{e}"
        )
        lm = CNNLightningModule.load_from_checkpoint(
            str(ckpt), model=inner, strict=False, map_location=device
        )
    lm.eval().to(device)
    print(
        f"fold{fold}: loaded {ckpt.name}  (config: {cfg_path.relative_to(REPO_ROOT)})"
    )

    dm = LazyDataModule(str(cfg_path))
    dm.setup()
    full_ds = dm.test_dataset.dataset
    test_indices = dm.test_dataset.indices
    # num_workers=0: multiprocessing workers die silently on compute nodes during inference
    test_dl = DataLoader(
        dm.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    preds, log_vars, trues, quantile_preds = [], [], [], []
    with torch.no_grad():
        for xs, xt, y in test_dl:
            xs, xt = xs.float().to(device), xt.float().to(device)
            if quantile_head:
                p, q = lm.model.forward_with_quantile(xs, xt)
                quantile_preds.append(q.squeeze(-1).cpu())
            else:
                p = lm.model(xs, xt)
            if p.ndim == 2 and p.shape[-1] == 2:
                preds.append(p[:, 0].cpu())
                log_vars.append(p[:, 1].cpu())
            else:
                preds.append(p.squeeze(-1).cpu())
            trues.append(y.squeeze(-1).cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    quantile_preds_norm = torch.cat(quantile_preds).numpy() if quantile_head else None

    # Denormalise to degC
    preds_c = preds * lm.target_std + lm.target_mean
    trues_c = trues * lm.target_std + lm.target_mean
    quantile_pred_c = (
        quantile_preds_norm * lm.target_std + lm.target_mean
        if quantile_preds_norm is not None
        else None
    )

    has_std = len(log_vars) > 0
    if has_std:
        std_norm = np.exp(np.array(np.concatenate(log_vars)) / 2.0)
        std_c = std_norm * lm.target_std
    else:
        std_c = None

    times = full_ds.ds.time.values
    dates = np.array(
        [times[i + full_ds.window_size - 1 + full_ds.lead_time] for i in test_indices]
    )

    order = np.argsort(dates)
    dates, trues_c, preds_c = dates[order], trues_c[order], preds_c[order]
    if std_c is not None:
        std_c = std_c[order]
    if quantile_pred_c is not None:
        quantile_pred_c = quantile_pred_c[order]

    r = pearson_r(preds_c, trues_c)
    mae = float(np.mean(np.abs(preds_c - trues_c)))
    print(f"fold{fold}: n={len(dates)}  r={r:.4f}  MAE={mae:.4f} degC", end="")
    if std_c is not None:
        print(f"  mean_std={np.mean(std_c):.4f} degC", end="")
    print()

    out_path = run_dir / "test_predictions.npz"
    save_kwargs = dict(
        dates=dates.astype("datetime64[D]").astype(str),
        trues_degC=trues_c,
        preds_degC=preds_c,
        r=r,
        mae_degC=mae,
        checkpoint=ckpt.name,
        config=str(cfg_path.relative_to(REPO_ROOT)),
    )
    if std_c is not None:
        save_kwargs["std_degC"] = std_c
    if quantile_pred_c is not None:
        # Model's own predictive quantile (per-timestep), NOT Hobday's
        # p90_thresh (fixed climatological threshold, src/utils/hobday.py).
        # Name carries the tau it was trained at so it's unambiguous
        # downstream regardless of what value quantile_tau takes.
        save_kwargs[f"quantile_pred_tau{cfg['quantile_tau']:.2f}_degC"] = (
            quantile_pred_c
        )
    np.savez(out_path, **save_kwargs)
    print(f"fold{fold}: saved {out_path}")

    return dates, trues_c, preds_c, r, mae, std_c


def _with_gaps(dates, values, gap_days=10):
    """Insert NaN wherever consecutive dates are more than gap_days apart,
    so plt.plot doesn't draw a straight line across a fold's missing years."""
    dates = dates.astype("datetime64[D]")
    gaps = np.diff(dates).astype("timedelta64[D]").astype(int) > gap_days
    gap_idx = np.where(gaps)[0]
    values = values.astype(float).copy()
    dates_out = dates.copy()
    if len(gap_idx) == 0:
        return dates_out, values
    dates_out = np.insert(dates_out, gap_idx + 1, dates_out[gap_idx])
    values = np.insert(values, gap_idx + 1, np.nan)
    return dates_out, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition", choices=["full", "remote", "local"], default="full"
    )
    parser.add_argument("--model_tag", default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--output", default=None, help="Output figure path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    results = [
        run_fold(fold, args.partition, args.model_tag, device, args.batch_size)
        for fold in range(5)
    ]

    tag_label = f" ({args.model_tag})" if args.model_tag else ""
    partition_label = {"full": "Full", "remote": "Remote", "local": "Local"}[
        args.partition
    ]

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"TbotAtm {partition_label}{tag_label} — 5-fold test-set predictions vs. truth",
        fontsize=13,
        y=1.005,
    )
    for fold, (ax, (dates, trues_c, preds_c, r, mae, std_c)) in enumerate(
        zip(axes, results)
    ):
        d_true, v_true = _with_gaps(dates, trues_c)
        d_pred, v_pred = _with_gaps(dates, preds_c)
        ax.plot(d_true, v_true, color="steelblue", lw=1.0, label="Truth")
        ax.plot(d_pred, v_pred, color="tomato", lw=1.0, ls="--", label="Prediction")
        if std_c is not None:
            d_s, v_s = _with_gaps(dates, std_c)
            d_m, v_m = _with_gaps(dates, preds_c)
            ax.fill_between(
                d_m, v_m - v_s, v_m + v_s, color="tomato", alpha=0.2, label="±1σ"
            )
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("to_anom (°C)", fontsize=9)
        ax.set_title(
            f"Fold {fold}   n={len(dates)}   r={r:.3f}   MAE={mae:.3f}°C", fontsize=10
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, fontsize=8)

    plt.tight_layout()
    out_path = (
        Path(args.output)
        if args.output
        else (
            REPO_ROOT
            / "figures"
            / f"TbotAtm_{args.partition}{'_' + args.model_tag if args.model_tag else ''}_fold_comparison_timeseries.png"
        )
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
