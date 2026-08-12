#!/usr/bin/env python
"""
Ensemble skill scores: average predictions across N seed models, then compute skill.

Two evaluations are reported:
  1. Full-ensemble  — all samples, predictions averaged over all seeds
  2. Out-of-sample  — each sample's prediction averaged only from seeds where it
                      was in the TEST set (proper out-of-sample evaluation)

Usage:
  python eval/ensemble_skill.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 split_blockyear/TbotAtm_seed200 \
               split_blockyear/TbotAtm_seed300 split_blockyear/TbotAtm_seed400 \
    --output_dir split_blockyear/TbotAtm_ensemble/eval_results \
    --threshold 0.0
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from glob import glob

import torch

sys.path.append(str(Path(__file__).parent.parent))
from datamodule import LazyDataModule
from model import CNNLightningModule, CNNLSTMModel
from xai.utils import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def load_model(ckpt_path: Path, config: dict, device: str):
    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden",   256),
        lstm_layers=config.get("lstm_layers",     2),
        temporal_features=config.get("temporal_features", 3),
        dropout=config.get("dropout", 0.3),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path), model=cnn_lstm, map_location=device,
    )
    lm.eval().to(device)
    return lm


def run_inference(lm, full_ds, device):
    preds, years, months = [], [], []
    with torch.no_grad():
        for idx in range(len(full_ds)):
            xs, xt, y = full_ds[idx]
            p, _ = lm.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device),
            )
            preds.append(p.item())
            t = idx + full_ds.window_size - 1 + full_ds.lead_time
            years.append(int(full_ds.years[t]))
            months.append(int(full_ds.months[t]))
            if (idx + 1) % 2000 == 0:
                print(f"    {idx+1}/{len(full_ds)}", end="\r")
    print()
    return np.array(preds), np.array(years), np.array(months)


def contingency(y_true_bin, y_pred_bin):
    H  = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    FA = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    M  = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    CN = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
    return H, FA, M, CN


def skill_scores(H, FA, M, CN):
    N   = H + FA + M + CN
    pod = H / (H + M)       if (H + M)   > 0 else np.nan
    far = FA / (H + FA)     if (H + FA)  > 0 else np.nan
    csi = H / (H + M + FA)  if (H + M + FA) > 0 else np.nan
    H_r = (H + M) * (H + FA) / N if N > 0 else 0
    ets = (H - H_r) / (H + M + FA - H_r) if (H + M + FA - H_r) > 0 else np.nan
    hss = 2 * (H * CN - FA * M) / ((H + M) * (M + CN) + (H + FA) * (FA + CN)) \
          if ((H + M) * (M + CN) + (H + FA) * (FA + CN)) > 0 else np.nan
    return pod, far, csi, ets, hss


def pearson_r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def print_and_write(fp, line):
    print(line)
    fp.write(line + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs",   nargs="+", required=True,
                        help="Paths to experiment directories (one per seed)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold",  type=float, default=0.0)
    parser.add_argument("--no_cuda",    action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    exp_dirs = [Path(d) for d in args.exp_dirs]

    # ---------------------------------------------------------------------------
    # Load all models and run inference
    # ---------------------------------------------------------------------------
    all_preds    = []   # list of (N,) arrays, one per seed
    all_splits   = []   # list of (N,) string arrays ["train"/"val"/"test"]
    trues        = None
    years        = None
    months       = None

    for exp_dir in exp_dirs:
        ckpt   = best_checkpoint(exp_dir)
        config = load_config(str(exp_dir / "config.yaml"))
        seed   = config.get("seed", "?")
        print(f"\n[seed={seed}] Loading {ckpt.name}")

        lm = load_model(ckpt, config, device)

        dm = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset

        print(f"  Running inference ({len(full_ds)} samples)...")
        preds_i, years_i, months_i = run_inference(lm, full_ds, device)
        print(f"  r={pearson_r(preds_i, np.array([full_ds[i][2].item() for i in range(len(full_ds))])):.4f}")

        # Split mask for this seed
        split_mask = np.array(["train"] * len(full_ds), dtype=object)
        for idx in dm.val_dataset.indices:
            split_mask[idx] = "val"
        for idx in dm.test_dataset.indices:
            split_mask[idx] = "test"

        all_preds.append(preds_i)
        all_splits.append(split_mask)

        if trues is None:
            trues  = np.array([full_ds[i][2].item() for i in range(len(full_ds))])
            years  = years_i
            months = months_i

    N = len(trues)
    n_seeds = len(all_preds)
    print(f"\nLoaded {n_seeds} seed models. N={N} samples.")

    # ---------------------------------------------------------------------------
    # Ensemble predictions: full (all seeds averaged)
    # ---------------------------------------------------------------------------
    ens_full = np.stack(all_preds).mean(axis=0)   # (N,)
    print(f"\nFull ensemble r = {pearson_r(ens_full, trues):.4f}")

    # ---------------------------------------------------------------------------
    # Out-of-sample ensemble: for each sample, average only from seeds where
    # that sample was in the TEST set. Fall back to VAL seeds if no TEST seeds.
    # ---------------------------------------------------------------------------
    ens_oos = np.full(N, np.nan)
    oos_seed_count = np.zeros(N, dtype=int)

    for i in range(N):
        test_preds = [all_preds[s][i] for s in range(n_seeds) if all_splits[s][i] == "test"]
        if test_preds:
            ens_oos[i] = np.mean(test_preds)
            oos_seed_count[i] = len(test_preds)
        else:
            val_preds = [all_preds[s][i] for s in range(n_seeds) if all_splits[s][i] == "val"]
            if val_preds:
                ens_oos[i] = np.mean(val_preds)
                oos_seed_count[i] = -len(val_preds)   # negative = val

    oos_mask = ~np.isnan(ens_oos)
    print(f"Out-of-sample samples: {oos_mask.sum()} ({100*oos_mask.mean():.1f}%)")
    print(f"  - test: {(oos_seed_count > 0).sum()}")
    print(f"  - val only: {(oos_seed_count < 0).sum()}")

    thr = args.threshold
    txt_path = output_dir / "skill_scores.txt"

    with open(txt_path, "w") as fp:

        def report(label, mask, pred_arr):
            if mask.sum() < 10:
                return
            yb = (pred_arr[mask] > thr).astype(int)
            tb = (trues[mask] > thr).astype(int)
            H, FA, M, CN = contingency(tb, yb)
            pod, far, csi, ets, hss = skill_scores(H, FA, M, CN)
            r = pearson_r(pred_arr[mask], trues[mask])
            yrs = sorted(set(years[mask]))
            print_and_write(fp, f"\n{'='*60}")
            print_and_write(fp, f"{label}  (N={mask.sum()}, years={yrs[0]}–{yrs[-1]})")
            print_and_write(fp, f"  POD={pod:.4f}  FAR={far:.4f}  CSI={csi:.4f}  ETS={ets:.4f}  HSS={hss:.4f}  r={r:.4f}")
            # per-year
            print_and_write(fp, f"\n  {'Year':>5}  {'N':>5}  {'POD':>6}  {'FAR':>6}  {'CSI':>6}  {'ETS':>6}  {'HSS':>6}  {'r':>6}")
            for yr in sorted(set(years[mask])):
                ymask = mask & (years == yr)
                if ymask.sum() < 5:
                    continue
                yb_ = (pred_arr[ymask] > thr).astype(int)
                tb_ = (trues[ymask] > thr).astype(int)
                H_, FA_, M_, CN_ = contingency(tb_, yb_)
                pod_, far_, csi_, ets_, hss_ = skill_scores(H_, FA_, M_, CN_)
                r_ = pearson_r(pred_arr[ymask], trues[ymask])
                print_and_write(fp, f"  {yr:>5}  {ymask.sum():>5}  {pod_:>6.3f}  {far_:>6.3f}  {csi_:>6.3f}  {ets_:>6.3f}  {hss_:>6.3f}  {r_:>6.3f}")

        print_and_write(fp, f"Ensemble skill scores — {n_seeds} seeds")
        print_and_write(fp, f"Seeds: {[str(d) for d in exp_dirs]}")
        print_and_write(fp, f"Threshold: {thr}")

        report("FULL ENSEMBLE (all samples)", np.ones(N, bool), ens_full)
        report("OUT-OF-SAMPLE ENSEMBLE (test+val)", oos_mask, ens_oos)
        report("  → TEST only", oos_seed_count > 0, ens_oos)

        # Per-seed reference
        print_and_write(fp, f"\n{'='*60}")
        print_and_write(fp, "INDIVIDUAL SEED REFERENCE (test split for each seed)")
        for s, (exp_dir, preds_s, splits_s) in enumerate(zip(exp_dirs, all_preds, all_splits)):
            test_mask = splits_s == "test"
            if test_mask.sum() < 5:
                continue
            yb = (preds_s[test_mask] > thr).astype(int)
            tb = (trues[test_mask] > thr).astype(int)
            H, FA, M, CN = contingency(tb, yb)
            pod, far, csi, ets, hss = skill_scores(H, FA, M, CN)
            r = pearson_r(preds_s[test_mask], trues[test_mask])
            cfg = load_config(str(exp_dir / "config.yaml"))
            seed = cfg.get("seed", s)
            print_and_write(fp, f"  seed={seed:>4}  N={test_mask.sum():>5}  POD={pod:.4f}  FAR={far:.4f}  CSI={csi:.4f}  ETS={ets:.4f}  HSS={hss:.4f}  r={r:.4f}")

    print(f"\nSaved: {txt_path}")

    # ---------------------------------------------------------------------------
    # Save predictions for downstream figures / XAI
    # ---------------------------------------------------------------------------
    np.savez(
        output_dir / "predictions.npz",
        ens_full=ens_full,
        ens_oos=ens_oos,
        oos_mask=oos_mask,
        trues=trues,
        years=years,
        months=months,
        all_preds=np.stack(all_preds),
        all_splits=np.array(all_splits),
    )
    print(f"Saved: {output_dir}/predictions.npz")

    # ---------------------------------------------------------------------------
    # ROC curve: full ensemble vs individual seeds
    # ---------------------------------------------------------------------------
    from sklearn.metrics import roc_curve, roc_auc_score
    y_true_bin = (trues > thr).astype(int)

    fig, ax = plt.subplots(figsize=(6, 6))
    for s, (exp_dir, preds_s) in enumerate(zip(exp_dirs, all_preds)):
        cfg  = load_config(str(exp_dir / "config.yaml"))
        seed = cfg.get("seed", s)
        fpr_, tpr_, _ = roc_curve(y_true_bin, preds_s)
        auc_ = roc_auc_score(y_true_bin, preds_s)
        ax.plot(fpr_, tpr_, lw=1, alpha=0.5, ls="--", label=f"seed={seed} (AUC={auc_:.3f})")

    fpr_e, tpr_e, _ = roc_curve(y_true_bin, ens_full)
    auc_e = roc_auc_score(y_true_bin, ens_full)
    ax.plot(fpr_e, tpr_e, lw=2.5, color="k", label=f"Ensemble (AUC={auc_e:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR (POD)")
    ax.set_title("ROC — ensemble vs individual seeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/roc_curve.png  AUC_ensemble={auc_e:.3f}")

    # ---------------------------------------------------------------------------
    # Time series: ensemble prediction with spread (std across seeds)
    # ---------------------------------------------------------------------------
    import xarray as xr
    config0 = load_config(str(exp_dirs[0] / "config.yaml"))
    ds_tmp  = xr.open_dataset(config0["data_dir"])
    times   = ds_tmp.time.values
    ds_tmp.close()

    dm0     = LazyDataModule(config_path=str(exp_dirs[0] / "config.yaml"))
    dm0.setup()
    full_ds0 = dm0.train_dataset.dataset

    date_arr = np.array([
        times[idx + full_ds0.window_size - 1 + full_ds0.lead_time]
        for idx in range(N)
    ])

    seed_std = np.stack(all_preds).std(axis=0)
    decades  = [(1985, 1994), (1995, 2004), (2005, 2014), (2015, 2024)]
    y_true_bin = (trues > thr).astype(int)

    fig, axes = plt.subplots(len(decades), 1, figsize=(18, 14))
    fig.suptitle(f"Ensemble prediction vs truth — {n_seeds} seeds  (7-day lead)", fontsize=11)

    for ax, (yr_start, yr_end) in zip(axes, decades):
        mask = (years >= yr_start) & (years <= yr_end)
        if not mask.any():
            ax.set_visible(False)
            continue
        ax.fill_between(date_arr[mask],
                        ens_full[mask] - seed_std[mask],
                        ens_full[mask] + seed_std[mask],
                        color="tomato", alpha=0.25, label="±1σ seeds")
        ax.plot(date_arr[mask], trues[mask],    color="steelblue", lw=0.9, label="Truth",    alpha=0.9)
        ax.plot(date_arr[mask], ens_full[mask], color="tomato",    lw=0.9, label="Ensemble", alpha=0.9, ls="--")
        ax.axhline(thr, color="k", lw=0.7, ls=":")
        ax.fill_between(date_arr[mask], 0, 1,
                        where=(y_true_bin[mask] == 1),
                        transform=ax.get_xaxis_transform(),
                        alpha=0.12, color="gold", label="True MHW")
        ax.set_title(f"{yr_start}–{yr_end}", fontsize=10)
        ax.set_ylabel("to_anom (norm.)", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "timeseries_ensemble.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/timeseries_ensemble.png")

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
