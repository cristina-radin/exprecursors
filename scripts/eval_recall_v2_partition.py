"""
eval_recall_v2_partition.py — Aug 22 2026: generalizes
quantile_head_recall_v2_all5.py (previously hardcoded to
full_gnll_quantile_v2) to any of the 7 v2 experiment families now that
all of item 1 (land_fill folds1-4), item 2 (local/remote), and item 3
(lead-time sweep) have finished training. Same methodology exactly (def1
basin-mean, def2 pixel+area>=5%, recall/precision/FPR, both heads,
pooled across 5 folds) -- not reimplemented, just parameterized, to
avoid this project's own documented anti-pattern of duplicated-and-
drifting eval logic (known_issues.md).

Correctly applies mask_local/mask_remote (src/data/masking.py, single
source of truth) BEFORE forward_with_quantile() for --mode local_only/
remote_only -- evaluating those checkpoints on unmasked input would be
out-of-distribution and wrong, since they were trained on masked input
via train_partition.py's RemoteOnlyLightningModule/
LocalOnlyLightningModule (which only override the LightningModule's
step methods, not model.forward_with_quantile() itself -- calling that
directly bypasses masking unless applied explicitly here).

Uses each fold's own `cfg["output_dir"]` for the checkpoint path (not a
reconstructed/hardcoded pattern) -- more robust to config drift.

CPU only.

Usage:
  python scripts/eval_recall_v2_partition.py --config_dir full_gnll_quantile_v2_landfill --mode full --label full_lead7
  python scripts/eval_recall_v2_partition.py --config_dir local --mode local_only --label local
  python scripts/eval_recall_v2_partition.py --config_dir remote --mode remote_only --label remote
  python scripts/eval_recall_v2_partition.py --config_dir lead3_landfill --mode full --label lead3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.data.masking import mask_local, mask_remote  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.hobday import load_ns_p90  # noqa: E402

FIGURES_DIR = REPO_ROOT / "experiments" / "figures"
AREA_FRAC_THRESHOLD = (
    0.05  # MedECC 2023, same as mhw_definition_agreement_and_recall.py
)
device = "cuda" if torch.cuda.is_available() else "cpu"

MASK_FNS = {
    "full": lambda xs: xs,
    "local_only": mask_local,
    "remote_only": mask_remote,
}


def run_fold(config_dir, fold, mask_fn, p90, area_frac, return_years=False):
    cfg_path = REPO_ROOT / "configs" / "partition" / config_dir / f"fold{fold}.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    run_dir = Path(cfg["output_dir"])

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
            xs = mask_fn(xs)
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

    r_mean = np.corrcoef(trues_c, mean_preds_c)[0, 1]
    r_q = np.corrcoef(trues_c, q_preds_c)[0, 1]
    print(
        f"  fold {fold}: n_test={len(trues_c)}  r_mean={r_mean:.4f}  r_quantile={r_q:.4f}  "
        f"q_pred mean={q_preds_c.mean():.3f}  mean_pred mean={mean_preds_c.mean():.3f}",
        flush=True,
    )
    if return_years:
        years_test = full_ds.years[target_idx]
        return trues_c, mean_preds_c, q_preds_c, thresh1, area_frac_test, years_test
    return trues_c, mean_preds_c, q_preds_c, thresh1, area_frac_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir", required=True, help="dir under configs/partition/"
    )
    parser.add_argument("--mode", required=True, choices=list(MASK_FNS.keys()))
    parser.add_argument("--label", required=True, help="used for output filenames")
    args = parser.parse_args()

    print(f"device={device}", flush=True)
    p90 = load_ns_p90()
    area_frac = np.load(FIGURES_DIR / "area_frac_timeseries.npy")
    mask_fn = MASK_FNS[args.mode]

    all_trues, all_mean, all_q, all_thresh1, all_area_frac = [], [], [], [], []
    print(f"\n=== {args.label} ALL 5 FOLDS ===", flush=True)
    for fold in range(5):
        trues_c, mean_c, q_c, thresh1, area_frac_test = run_fold(
            args.config_dir, fold, mask_fn, p90, area_frac
        )
        all_trues.append(trues_c)
        all_mean.append(mean_c)
        all_q.append(q_c)
        all_thresh1.append(thresh1)
        all_area_frac.append(area_frac_test)
        # incremental save per fold (CLAUDE.md rule for jobs >1h)
        np.savez(
            FIGURES_DIR / "_fold_cache" / f"eval_recall_v2_{args.label}_partial.npz",
            trues_c=np.concatenate(all_trues),
            mean_c=np.concatenate(all_mean),
            q_c=np.concatenate(all_q),
            thresh1=np.concatenate(all_thresh1),
            area_frac_c=np.concatenate(all_area_frac),
            n_folds_done=fold + 1,
        )

    trues_c = np.concatenate(all_trues)
    mean_c = np.concatenate(all_mean)
    q_c = np.concatenate(all_q)
    thresh1 = np.concatenate(all_thresh1)
    area_frac_c = np.concatenate(all_area_frac)

    r_mean_pooled = np.corrcoef(trues_c, mean_c)[0, 1]
    r_q_pooled = np.corrcoef(trues_c, q_c)[0, 1]

    ext1 = trues_c > thresh1
    n1 = int(ext1.sum())
    recall_mean_def1 = (mean_c[ext1] > thresh1[ext1]).mean() if n1 else float("nan")
    recall_q_def1 = (q_c[ext1] > thresh1[ext1]).mean() if n1 else float("nan")

    ext2 = area_frac_c >= AREA_FRAC_THRESHOLD
    n2 = int(ext2.sum())
    recall_mean_def2 = (mean_c[ext2] > thresh1[ext2]).mean() if n2 else float("nan")
    recall_q_def2 = (q_c[ext2] > thresh1[ext2]).mean() if n2 else float("nan")

    print(
        f"\n=== SUMMARY: {args.label}, pooled 5 folds, mean head vs quantile head (tau=0.9) ==="
    )
    print(
        f"pooled_r_mean_head={r_mean_pooled:.4f}  pooled_r_quantile_head={r_q_pooled:.4f}  n={len(trues_c)}"
    )
    print(
        f"def1 (basin-mean):        mean_head recall={recall_mean_def1*100:.1f}%  "
        f"quantile_head recall={recall_q_def1*100:.1f}%  (n={n1})"
    )
    print(
        f"def2 (pixel+area>=0.05):  mean_head recall={recall_mean_def2*100:.1f}%  "
        f"quantile_head recall={recall_q_def2*100:.1f}%  (n={n2})"
    )

    print(f"\n=== False-positive rate + precision (def1, basin-mean), {args.label} ===")
    for name, pred in [("mean_head", mean_c), ("quantile_head", q_c)]:
        flagged = pred > thresh1
        n_flagged = int(flagged.sum())
        fp = int((flagged & ~ext1).sum())
        tp = int((flagged & ext1).sum())
        fpr = fp / max(1, int((~ext1).sum()))
        precision = tp / max(1, n_flagged)
        print(
            f"  {name:14s}: n_flagged={n_flagged} ({n_flagged/len(pred)*100:.1f}% of all days)  "
            f"FPR={fpr*100:.1f}%  precision={precision*100:.1f}%"
        )

    print(
        f"\n=== False-positive rate + precision (def2, pixel+area>=0.05), {args.label} ==="
    )
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

    out_path = FIGURES_DIR / "_fold_cache" / f"eval_recall_v2_{args.label}.npz"
    np.savez(
        out_path,
        trues_c=trues_c,
        mean_c=mean_c,
        q_c=q_c,
        thresh1=thresh1,
        area_frac_c=area_frac_c,
        r_mean_pooled=r_mean_pooled,
        r_q_pooled=r_q_pooled,
    )
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
