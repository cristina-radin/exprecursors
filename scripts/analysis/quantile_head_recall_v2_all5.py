"""
quantile_head_recall_v2_all5.py — Paso 5 decisive analysis, Aug 21 2026:
pooled quantile-head (tau=0.9) recall/precision/FPR across all 5 folds of
full_gnll_quantile_v2, now that the array job (29426208, folds 1-4) has
completed alongside the already-existing fold0 (job 29417405).

q_pred lives in the same normalized space as `y` (pinball loss compares
q_pred directly against the same normalized target used for NLL) -- same
target_mean/target_std conversion to physical units as the mean head.

Reuses: src/utils/hobday.py, src/utils/checkpoints.py (best_ckpt,
load_model_config), configs/partition/full_gnll_quantile_v2/fold0.yaml,
checkpoint at experiments/partition/TbotAtm_full_gnll_quantile_v2_seed42_fold0/.

CPU only. Not part of the permanent pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path("/raven/u/cradin/exprecursors")
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.hobday import load_ns_p90  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "partition"  # v2, not old_v1
FIGURES_DIR = REPO_ROOT / "experiments" / "figures"
AREA_FRAC_THRESHOLD = (
    0.05  # MedECC 2023, same as mhw_definition_agreement_and_recall.py
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)

p90 = load_ns_p90()
area_frac = np.load(FIGURES_DIR / "area_frac_timeseries.npy")


def run_fold(fold):
    cfg_path = (
        REPO_ROOT
        / "configs"
        / "partition"
        / "full_gnll_quantile_v2"
        / f"fold{fold}.yaml"
    )
    cfg = yaml.safe_load(open(cfg_path))
    run_dir = EXPERIMENTS_DIR / f"TbotAtm_full_gnll_quantile_v2_seed42_fold{fold}"

    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    assert model_kwargs.get("quantile_head", False), (
        f"fold{fold}: expected quantile_head=True in saved model_config.json, "
        "got False -- wrong checkpoint or config drift, do not proceed silently."
    )
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    print(f"  fold {fold}: loading {ckpt.name}", flush=True)
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=True, map_location=device
    )
    lm.eval().to(device)

    dm = LazyDataModule(str(cfg_path))
    dm.setup()
    full_ds = dm.test_dataset.dataset
    test_indices = dm.test_dataset.indices
    test_dl = DataLoader(dm.test_dataset, batch_size=32, shuffle=False, num_workers=0)

    mean_preds, q_preds, trues = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            xs, xt, y = batch[0], batch[1], batch[2]
            xs, xt = xs.float().to(device), xt.float().to(device)
            y_hat, q_pred = lm.model.forward_with_quantile(xs, xt)
            mean_preds.append(
                (y_hat[:, 0] if y_hat.ndim == 2 else y_hat.squeeze(-1)).cpu()
            )
            q_preds.append(q_pred.squeeze(-1).cpu())
            trues.append(y.squeeze(-1).cpu())
    mean_preds = torch.cat(mean_preds).numpy()
    q_preds = torch.cat(q_preds).numpy()
    trues = torch.cat(trues).numpy()

    mean_preds_c = mean_preds * lm.target_std + lm.target_mean
    q_preds_c = q_preds * lm.target_std + lm.target_mean
    trues_c = trues * lm.target_std + lm.target_mean

    target_idx = np.array(
        [i + full_ds.window_size - 1 + full_ds.lead_time for i in test_indices]
    )
    doys = full_ds.doys[target_idx]
    doys = np.where(doys >= 365, 365, doys)
    thresh1 = p90[doys - 1]
    area_frac_test = area_frac[target_idx]

    print(
        f"  fold {fold}: n_test={len(trues_c)}  "
        f"q_pred mean={q_preds_c.mean():.3f}  mean_pred mean={mean_preds_c.mean():.3f}  "
        f"(sanity: q_pred should sit above mean_pred on average, tau=0.9)",
        flush=True,
    )
    return trues_c, mean_preds_c, q_preds_c, thresh1, area_frac_test


all_trues, all_mean, all_q, all_thresh1, all_area_frac = [], [], [], [], []
print("\n=== full_gnll_quantile_v2 ALL 5 FOLDS (Paso 5 decisive analysis) ===")
for fold in range(5):
    trues_c, mean_c, q_c, thresh1, area_frac_test = run_fold(fold)
    all_trues.append(trues_c)
    all_mean.append(mean_c)
    all_q.append(q_c)
    all_thresh1.append(thresh1)
    all_area_frac.append(area_frac_test)

trues_c = np.concatenate(all_trues)
mean_c = np.concatenate(all_mean)
q_c = np.concatenate(all_q)
thresh1 = np.concatenate(all_thresh1)
area_frac_c = np.concatenate(all_area_frac)

ext1 = trues_c > thresh1
n1 = int(ext1.sum())
recall_mean_def1 = (mean_c[ext1] > thresh1[ext1]).mean() if n1 else float("nan")
recall_q_def1 = (q_c[ext1] > thresh1[ext1]).mean() if n1 else float("nan")

ext2 = area_frac_c >= AREA_FRAC_THRESHOLD
n2 = int(ext2.sum())
recall_mean_def2 = (mean_c[ext2] > thresh1[ext2]).mean() if n2 else float("nan")
recall_q_def2 = (q_c[ext2] > thresh1[ext2]).mean() if n2 else float("nan")

print(
    "\n=== SUMMARY: full_gnll_quantile_v2 ALL 5 FOLDS, mean head vs quantile head (tau=0.9) ==="
)
print(
    f"def1 (basin-mean):        mean_head recall={recall_mean_def1*100:.1f}%  "
    f"quantile_head recall={recall_q_def1*100:.1f}%  (n={n1})"
)
print(
    f"def2 (pixel+area>=0.05):  mean_head recall={recall_mean_def2*100:.1f}%  "
    f"quantile_head recall={recall_q_def2*100:.1f}%  (n={n2})"
)
print("\nFor reference (already computed, mean head only, full_gnll_focal v1):")
print("  def1: 34.9%   def2: 12.8%")

# High recall alone is cheap to get if q_pred is just systematically
# shifted up (tau=0.9 biases it above the mean by construction) -- that
# would inflate recall without adding real discriminative skill, and is
# exactly the "spurious result" risk flagged before trusting this. Check
# the other side: on days that were NOT actually extreme, how often does
# q_pred/mean STILL cross the threshold (false positive rate), and what
# fraction of ALL flagged days were genuinely extreme (precision). A
# meaningful signal needs high recall AND a false-positive rate well
# below the flag rate you'd get from a naive "always predict high" model.
print("\n=== False-positive rate + precision (def1, basin-mean) ===")
for name, pred in [("mean_head", mean_c), ("quantile_head", q_c)]:
    flagged = pred > thresh1
    n_flagged = int(flagged.sum())
    fp = int((flagged & ~ext1).sum())
    tp = int((flagged & ext1).sum())
    fpr = fp / max(
        1, int((~ext1).sum())
    )  # of non-extreme days, how many falsely flagged
    precision = tp / max(1, n_flagged)
    print(
        f"  {name:14s}: n_flagged={n_flagged} ({n_flagged/len(pred)*100:.1f}% of all days)  "
        f"FPR={fpr*100:.1f}%  precision={precision*100:.1f}%"
    )

print("\n=== False-positive rate + precision (def2, pixel+area>=0.05) ===")
for name, pred in [("mean_head", mean_c), ("quantile_head", q_c)]:
    flagged = pred > thresh1
    n_flagged = int(flagged.sum())
    fp = int((flagged & ~ext2).sum())
    tp = int((flagged & ext2).sum())
    fpr = fp / max(1, int((~ext2).sum()))
    precision = tp / max(1, n_flagged)
    print(
        f"  {name:14s}: n_flagged={n_flagged} ({n_flagged/len(pred)*100:.1f}% of all days)  "
        f"FPR={fpr*100:.1f}%  precision={precision*100:.1f}%"
    )

np.savez(
    FIGURES_DIR / "_fold_cache" / "quantile_head_check_v2_all5.npz",
    trues_c=trues_c,
    mean_c=mean_c,
    q_c=q_c,
    thresh1=thresh1,
    area_frac_c=area_frac_c,
)
print(
    f"\nSaved raw arrays to {FIGURES_DIR / '_fold_cache' / 'quantile_head_check_v2_all5.npz'}"
)
