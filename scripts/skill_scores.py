#!/usr/bin/env python
"""
Binary MHW skill scores for CNN-LSTM regression model.

Thresholds both prediction and truth at 0 (to_anom > 0 = MHW day) and
computes POD, FAR, CSI, ETS per year and for the full dataset.
Also plots: prediction time series, ROC curve, reliability diagram.

Usage:
  python eval/skill_scores.py --checkpoint <ckpt> --config <yaml> --output_dir <dir>
"""

import argparse
import sys

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import torch

sys.path.append(str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.xai.utils import load_config

# ---------------------------------------------------------------------------
# Skill score functions
# ---------------------------------------------------------------------------


def contingency(y_true_bin, y_pred_bin):
    H = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    FA = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    M = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    CN = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
    return H, FA, M, CN


def skill_scores(H, FA, M, CN):
    N = H + FA + M + CN
    pod = H / (H + M) if (H + M) > 0 else np.nan
    far = FA / (H + FA) if (H + FA) > 0 else np.nan
    csi = H / (H + M + FA) if (H + M + FA) > 0 else np.nan
    # ETS: expected hits by random chance
    H_r = (H + M) * (H + FA) / N if N > 0 else 0
    ets = (H - H_r) / (H + M + FA - H_r) if (H + M + FA - H_r) > 0 else np.nan
    return pod, far, csi, ets


def pearson_r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="to_anom threshold for MHW (default=0)",
    )
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    config = load_config(args.config)

    # --- Load model ---
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
        quantile_head=config.get("quantile_head", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        args.checkpoint,
        model=cnn_lstm,
        map_location=device,
    )
    lm.eval().to(device)
    print(f"Loaded: {args.checkpoint}")
    print(f"  target_mean={lm.target_mean:.4f}  target_std={lm.target_std:.4f}")

    # --- Dataset ---
    dm = LazyDataModule(config_path=args.config)
    dm.setup()
    full_ds = dm.train_dataset.dataset

    print(f"\nFull dataset: {len(full_ds)} samples")

    # --- Inference ---
    preds, trues, years, months, doys = [], [], [], [], []
    with torch.no_grad():
        for idx in range(len(full_ds)):
            xs, xt, y = full_ds[idx]
            p, _ = lm.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device),
            )
            preds.append(p.item())
            trues.append(y.item())
            t = idx + full_ds.window_size - 1 + full_ds.lead_time
            years.append(int(full_ds.years[t]))
            months.append(int(full_ds.months[t]))
            doys.append(int(full_ds.doys[t]) if hasattr(full_ds, "doys") else 0)
            if (idx + 1) % 2000 == 0:
                print(f"  {idx+1}/{len(full_ds)}", end="\r")

    preds = np.array(preds)
    trues = np.array(trues)
    years = np.array(years)
    months = np.array(months)
    print(f"\nInference done. r={pearson_r(preds, trues):.3f}")

    thr = args.threshold
    y_pred_bin = (preds > thr).astype(int)
    y_true_bin = (trues > thr).astype(int)

    # Split masks
    split_mask = np.array(["train"] * len(preds), dtype=object)
    for idx in dm.val_dataset.indices:
        split_mask[idx] = "val"
    for idx in dm.test_dataset.indices:
        split_mask[idx] = "test"

    def report_metrics(label, mask):
        if mask.sum() == 0:
            return
        H_, FA_, M_, CN_ = contingency(y_true_bin[mask], y_pred_bin[mask])
        pod_, far_, csi_, ets_ = skill_scores(H_, FA_, M_, CN_)
        r_ = pearson_r(preds[mask], trues[mask])
        yrs = sorted(set(years[mask]))
        print(f"\n{'='*50}")
        print(f"{label}  (N={mask.sum()}, years={yrs})")
        print(f"  MHW days (truth): {H_+M_} ({100*(H_+M_)/mask.sum():.1f}%)")
        print(f"  H={H_}  FA={FA_}  M={M_}  CN={CN_}")
        print(
            f"  POD={pod_:.4f}  FAR={far_:.4f}  CSI={csi_:.4f}  ETS={ets_:.4f}  r={r_:.4f}"
        )

    report_metrics("TRAIN", split_mask == "train")
    report_metrics("VAL", split_mask == "val")
    report_metrics("TEST", split_mask == "test")

    # --- Overall metrics ---
    H, FA, M, CN = contingency(y_true_bin, y_pred_bin)
    pod, far, csi, ets = skill_scores(H, FA, M, CN)
    r = pearson_r(preds, trues)
    print(f"\n{'='*50}")
    print(f"OVERALL — ALL SPLITS (threshold={thr}°C)")
    print(f"  MHW days (truth):   {H+M} / {len(trues)} ({100*(H+M)/len(trues):.1f}%)")
    print(f"  H={H}  FA={FA}  M={M}  CN={CN}")
    print(f"  POD={pod:.4f}  FAR={far:.4f}  CSI={csi:.4f}  ETS={ets:.4f}  r={r:.4f}")

    # --- Per-year metrics (with split label) ---
    unique_years = sorted(set(years))
    rows = []
    for yr in unique_years:
        mask = years == yr
        if mask.sum() < 10:
            continue
        h, fa, m, cn = contingency(y_true_bin[mask], y_pred_bin[mask])
        p, f, c, e = skill_scores(h, fa, m, cn)
        rr = pearson_r(preds[mask], trues[mask])
        n_mhw = int((y_true_bin[mask]).sum())
        sp = split_mask[mask][0]
        rows.append((yr, n_mhw, p, f, c, e, rr, sp))

    print(
        f"\n{'Year':>5}  {'Split':>5}  {'MHW':>4}  {'POD':>5}  {'FAR':>5}  {'CSI':>5}  {'ETS':>5}  {'r':>5}"
    )
    for yr, n_mhw, p, f, c, e, rr, sp in rows:
        print(
            f"{yr:>5}  {sp:>5}  {n_mhw:>4}  {p:>5.3f}  {f:>5.3f}  {c:>5.3f}  {e:>5.3f}  {rr:>5.3f}"
        )

    # --- Save scores to text ---
    txt_path = output_dir / "skill_scores.txt"
    with open(txt_path, "w") as fp:
        fp.write(f"Checkpoint: {args.checkpoint}\n")
        fp.write(f"Threshold: {thr}°C\n\n")

        def write_split(label, mask):
            if mask.sum() == 0:
                return
            H_, FA_, M_, CN_ = contingency(y_true_bin[mask], y_pred_bin[mask])
            pod_, far_, csi_, ets_ = skill_scores(H_, FA_, M_, CN_)
            r_ = pearson_r(preds[mask], trues[mask])
            yrs = sorted(set(years[mask]))
            fp.write(f"{label}  years={yrs}\n")
            fp.write(
                f"  N={mask.sum()}  MHW_truth={H_+M_} ({100*(H_+M_)/mask.sum():.1f}%)\n"
            )
            fp.write(f"  H={H_}  FA={FA_}  M={M_}  CN={CN_}\n")
            fp.write(
                f"  POD={pod_:.4f}  FAR={far_:.4f}  CSI={csi_:.4f}  ETS={ets_:.4f}  r={r_:.4f}\n\n"
            )

        write_split("TRAIN", split_mask == "train")
        write_split("VAL", split_mask == "val")
        write_split("TEST", split_mask == "test")

        fp.write("OVERALL (all splits)\n")
        fp.write(
            f"  N={len(trues)}  MHW_truth={H+M} ({100*(H+M)/len(trues):.1f}%)  MHW_pred={H+FA} ({100*(H+FA)/len(preds):.1f}%)\n"
        )
        fp.write(f"  H={H}  FA={FA}  M={M}  CN={CN}\n")
        fp.write(
            f"  POD={pod:.4f}  FAR={far:.4f}  CSI={csi:.4f}  ETS={ets:.4f}  r={r:.4f}\n\n"
        )
        fp.write(
            f"{'Year':>5}  {'Split':>6}  {'MHW':>4}  {'POD':>6}  {'FAR':>6}  {'CSI':>6}  {'ETS':>6}  {'r':>6}\n"
        )
        for yr, n_mhw, p, f, c, e, rr, sp in rows:
            fp.write(
                f"{yr:>5}  {sp:>6}  {n_mhw:>4}  {p:>6.4f}  {f:>6.4f}  {c:>6.4f}  {e:>6.4f}  {rr:>6.4f}\n"
            )
    print(f"\nSaved: {txt_path}")

    # --- Save raw predictions for poster/downstream figures ---
    npz_path = output_dir / "predictions.npz"
    np.savez(
        npz_path,
        preds=preds,
        trues=trues,
        years=years,
        months=months,
        split_mask=split_mask,
        target_mean=np.float32(lm.target_mean),
        target_std=np.float32(lm.target_std),
    )
    print(f"Saved: {npz_path}")

    # --- Plot 1: Per-year ETS + POD ---
    yr_arr = np.array([r[0] for r in rows])
    pod_arr = np.array([r[2] for r in rows])
    far_arr = np.array([r[3] for r in rows])
    ets_arr = np.array([r[5] for r in rows])
    r_arr = np.array([r[6] for r in rows])
    sp_arr = np.array([r[7] for r in rows])

    split_colors = {"train": "steelblue", "val": "darkorange", "test": "crimson"}
    bar_colors = [split_colors[s] for s in sp_arr]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    from matplotlib.patches import Patch

    legend_patches = [Patch(color=c, label=s) for s, c in split_colors.items()]

    ax = axes[0]
    ax.bar(yr_arr, ets_arr, color=bar_colors, alpha=0.85, label="ETS")
    ax.bar(yr_arr, pod_arr, color=bar_colors, alpha=0.25)
    # dashed line at pod level so it's visible
    ax.step(
        yr_arr, pod_arr, color="k", lw=0.6, ls="--", where="mid", label="POD (dashed)"
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("ETS / POD")
    ax.set_title(
        "Per-year skill scores — binary MHW (to_anom > 0, 7-day lead)\n"
        "colour = split: blue=train  orange=val  red=test"
    )
    ax.legend(
        handles=legend_patches
        + [
            Patch(color="grey", alpha=0.85, label="ETS (solid)"),
            Patch(color="grey", alpha=0.25, label="POD (faded)"),
        ],
        fontsize=8,
        ncol=3,
    )
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(yr_arr, far_arr, color=bar_colors, alpha=0.85)
    ax.set_ylabel("FAR")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=legend_patches, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar(yr_arr, r_arr, color=bar_colors, alpha=0.85)
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("Year")
    ax.legend(handles=legend_patches, fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "skill_scores_peryear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Full prediction time series, split by decade ---
    ds_tmp = xr.open_dataset(config["data_dir"])
    times = ds_tmp.time.values
    ds_tmp.close()

    date_arr = np.array(
        [
            times[idx + full_ds.window_size - 1 + full_ds.lead_time]
            for idx in range(len(full_ds))
        ]
    )

    decades = [(1985, 1994), (1995, 2004), (2005, 2014), (2015, 2024)]
    # Build per-year split lookup
    yr2split = {}
    for idx in range(len(full_ds)):
        yr2split[years[idx]] = split_mask[idx]

    fig, axes = plt.subplots(len(decades), 1, figsize=(18, 14), sharex=False)
    fig.suptitle(
        "Prediction vs truth — full period (7-day lead)\n"
        "background: blue=train  orange=val  red=test",
        fontsize=11,
        y=1.01,
    )

    split_bg = {
        "train": ("steelblue", 0.07),
        "val": ("darkorange", 0.13),
        "test": ("crimson", 0.20),
    }

    for ax, (yr_start, yr_end) in zip(axes, decades):
        mask = (years >= yr_start) & (years <= yr_end)
        if not mask.any():
            ax.set_visible(False)
            continue

        # Shade background by split, one block per year
        for yr in range(yr_start, yr_end + 1):
            sp = yr2split.get(yr, "train")
            color, alpha = split_bg[sp]
            ax.axvspan(
                np.datetime64(f"{yr}-01-01"),
                np.datetime64(f"{yr+1}-01-01"),
                color=color,
                alpha=alpha,
                lw=0,
            )

        ax.plot(
            date_arr[mask],
            trues[mask],
            color="steelblue",
            lw=0.9,
            label="Truth",
            alpha=0.9,
        )
        ax.plot(
            date_arr[mask],
            preds[mask],
            color="tomato",
            lw=0.9,
            label="Prediction",
            alpha=0.85,
            ls="--",
        )
        ax.axhline(0, color="k", lw=0.7, ls="-")
        ax.fill_between(
            date_arr[mask],
            0,
            1,
            where=(y_true_bin[mask] == 1),
            transform=ax.get_xaxis_transform(),
            alpha=0.12,
            color="gold",
            label="True MHW",
        )
        ax.set_title(f"{yr_start}–{yr_end}", fontsize=10)
        ax.set_ylabel("to_anom (norm.)", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "timeseries_full.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/timeseries_full.png")

    # --- Plot 3: ROC curve ---
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr_arr, tpr_arr, _ = roc_curve(y_true_bin, preds)
    auc = roc_auc_score(y_true_bin, preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_arr, tpr_arr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate (1-specificity)")
    ax.set_ylabel("True Positive Rate (POD)")
    ax.set_title("ROC curve — MHW detection (7-day lead)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/roc_curve.png  AUC={auc:.3f}")

    # --- Plot 4: Reliability diagram ---
    n_bins = 10
    bin_edges = np.linspace(preds.min(), preds.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    obs_freq = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() > 0:
            obs_freq.append(y_true_bin[mask].mean())
        else:
            obs_freq.append(np.nan)
    obs_freq = np.array(obs_freq)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(bin_centers, obs_freq, "o-", lw=2, color="steelblue", label="Model")
    ax.plot([bin_edges[0], bin_edges[-1]], [0, 1], "k--", lw=1, label="Perfect")
    ax.axhline(
        y_true_bin.mean(),
        color="gray",
        ls=":",
        lw=1,
        label=f"Climatology ({y_true_bin.mean():.2f})",
    )
    ax.set_xlabel("Predicted to_anom (normalised)")
    ax.set_ylabel("Observed MHW frequency")
    ax.set_title("Reliability diagram — MHW detection")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/reliability_diagram.png")

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
