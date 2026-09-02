"""
plot_combined_test_timeseries.py — Truth vs. Prediction, ALL 5 folds
combined chronologically, same structure as the pre-session reference
figure (`plot_test_timeseries.py`'s output for mse_v2: top panel Truth
vs Prediction with r/MAE in the title, bottom panel residuals colored by
sign) -- requested Aug 21 2026 ("quiero la serie temporal de los 5
folds, misma estructura que la imagen que te he mandado"), for the
committed full_gnll_quantile_v2 model's MEAN head (point-forecast
comparison, matching what "Prediction" meant in the reference figure --
not q_pred, which is a different, exceedance-detection signal, see
docs/narrative.md's Aug 21 2026 DECISIVE finding entry).

Uses real calendar dates (full_ds.ds.time, not just day-of-year) so the
5 folds' non-contiguous test years plot correctly on one chronological
x-axis without overlapping.

Reuses: LazyDataModule, best_ckpt()/load_model_config()/
CNNLightningModule.load_from_checkpoint() (same pattern as
scripts/analysis/quantile_head_recall_v2_all5.py).

CPU only. Not part of the permanent pipeline.

Usage:
  python scripts/analysis/plot_combined_test_timeseries.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments" / "figures" / "step5_quantile_v2_predictions"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)


def run_fold(fold):
    cfg_path = (
        REPO_ROOT
        / "configs"
        / "partition"
        / "full_gnll_quantile_v2"
        / f"fold{fold}.yaml"
    )
    cfg = yaml.safe_load(open(cfg_path))
    dm = LazyDataModule(str(cfg_path))
    dm.setup()
    full_ds = dm.test_dataset.dataset
    test_indices = dm.test_dataset.indices

    run_dir = Path(cfg["output_dir"])
    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    print(f"  fold {fold}: loading {ckpt.name}", flush=True)
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=True, map_location=device
    )
    lm.eval().to(device)

    test_dl = DataLoader(dm.test_dataset, batch_size=32, shuffle=False, num_workers=0)
    mean_preds, log_vars, trues = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            xs, xt, y = batch[0], batch[1], batch[2]
            xs, xt = xs.float().to(device), xt.float().to(device)
            y_hat, _ = lm.model.forward_with_quantile(xs, xt)
            mean_preds.append(y_hat[:, 0].cpu())
            log_vars.append(y_hat[:, 1].cpu())  # GNLL's own uncertainty output
            trues.append(y.squeeze(-1).cpu())
    mean_preds = torch.cat(mean_preds).numpy() * lm.target_std + lm.target_mean
    # std in normalized space = exp(0.5*log_var); physical-unit std scales
    # by target_std the same way the mean does (var scales as std^2).
    std_preds = torch.exp(0.5 * torch.cat(log_vars)).numpy() * lm.target_std
    trues = torch.cat(trues).numpy() * lm.target_std + lm.target_mean

    target_idx = np.array(
        [i + full_ds.window_size - 1 + full_ds.lead_time for i in test_indices]
    )
    dates = full_ds.ds.time.values[target_idx]

    print(
        f"  fold {fold}: n_test={len(trues)}  dates {np.min(dates)} .. {np.max(dates)}",
        flush=True,
    )
    return dates, trues, mean_preds, std_preds


all_dates, all_trues, all_preds, all_stds = [], [], [], []
for fold in range(5):
    dates, trues, preds, stds = run_fold(fold)
    all_dates.append(dates)
    all_trues.append(trues)
    all_preds.append(preds)
    all_stds.append(stds)

dates = np.concatenate(all_dates)
trues = np.concatenate(all_trues)
preds = np.concatenate(all_preds)
stds = np.concatenate(all_stds)

order = np.argsort(dates)
dates, trues, preds, stds = dates[order], trues[order], preds[order], stds[order]
print(
    f"\nPredicted std (GNLL, physical units): mean={stds.mean():.4f}  min={stds.min():.4f}  max={stds.max():.4f} degC",
    flush=True,
)

r, _ = pearsonr(trues, preds)
mae = np.abs(trues - preds).mean()
print(f"\nPooled 5-fold: n={len(trues)}  r={r:.3f}  MAE={mae:.3f} degC", flush=True)

residual = preds - trues

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
)
ax1.fill_between(
    dates,
    preds - stds,
    preds + stds,
    color="#e08283",
    alpha=0.25,
    lw=0,
    label="Prediction ±1 std (GNLL)",
)
ax1.plot(dates, trues, color="#2c5f8a", lw=1.0, label="Truth")
ax1.plot(dates, preds, color="#e08283", lw=0.9, ls="--", label="Prediction (mean)")
ax1.axhline(0, color="black", lw=0.8)
ax1.set_ylabel("to_anom (°C)")
ax1.set_title(
    f"full_gnll_quantile_v2 (mean head) — combined test predictions (5-fold)   "
    f"r={r:.3f}   MAE={mae:.3f}°C"
)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3)

pos = np.where(residual >= 0, residual, 0)
neg = np.where(residual < 0, residual, 0)
ax2.fill_between(dates, pos, color="#e08283", lw=0, step=None)
ax2.fill_between(dates, neg, color="#6a9fc0", lw=0, step=None)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("Residual (°C)")
ax2.set_xlabel("Year")
ax2.grid(alpha=0.3)

plt.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "combined_test_timeseries_5fold.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}", flush=True)

np.savez(
    OUT_DIR / "combined_test_predictions_5fold.npz",
    dates=dates,
    trues=trues,
    preds=preds,
    stds=stds,
    r=r,
    mae=mae,
)
print(f"Saved {OUT_DIR / 'combined_test_predictions_5fold.npz'}", flush=True)
