"""Ad-hoc analysis: for full_gnll_quantile and full_gnll_focal (5 folds each),
compute pooled-across-folds recall on truth>p90 days (pred>p90), matching
narrative.md's existing GNLL-baseline methodology so the numbers are
directly comparable. Not a permanent script -- scratchpad only.
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

EXPERIMENTS_DIR = (
    REPO_ROOT / "experiments" / "partition" / "old_v1"
)  # v1 checkpoints only, moved Aug 20 2026 -- see known_issues.md #45
device = "cuda" if torch.cuda.is_available() else "cpu"
p90 = load_ns_p90()  # (365,) physical units


def run_fold(config_subdir, run_name, fold):
    cfg_path = REPO_ROOT / "configs" / "partition" / config_subdir / f"fold{fold}.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    run_dir = EXPERIMENTS_DIR / f"{run_name}_fold{fold}"

    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    extra = {}
    if cfg.get("focal_weight", False):
        # p90_by_doy isn't persisted in the checkpoint (known_issues.md #39)
        # -- recompute it exactly as train_partition.py did and pass it
        # explicitly so load_from_checkpoint's __init__ validation passes.
        extra["p90_by_doy"] = torch.tensor(p90, dtype=torch.float32)
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=True, map_location=device, **extra
    )
    lm.eval().to(device)

    dm = LazyDataModule(str(cfg_path))
    dm.setup()
    full_ds = dm.test_dataset.dataset
    test_indices = dm.test_dataset.indices
    test_dl = DataLoader(dm.test_dataset, batch_size=32, shuffle=False, num_workers=0)

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_dl:
            xs, xt, y = batch[0], batch[1], batch[2]
            xs, xt = xs.float().to(device), xt.float().to(device)
            p = lm.model(xs, xt)
            preds.append((p[:, 0] if p.ndim == 2 else p.squeeze(-1)).cpu())
            trues.append(y.squeeze(-1).cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    preds_c = preds * lm.target_std + lm.target_mean
    trues_c = trues * lm.target_std + lm.target_mean

    target_idx = np.array(
        [i + full_ds.window_size - 1 + full_ds.lead_time for i in test_indices]
    )
    doys = full_ds.doys[target_idx]
    doys = np.where(doys >= 365, 365, doys)
    thresh = p90[doys - 1]

    return trues_c, preds_c, thresh, ckpt.name


def analyze(config_subdir, run_name, label):
    all_trues, all_preds, all_thresh = [], [], []
    print(f"\n=== {label} ===")
    for fold in range(5):
        trues_c, preds_c, thresh, ckpt_name = run_fold(config_subdir, run_name, fold)
        r = np.corrcoef(preds_c, trues_c)[0, 1]
        mae = np.mean(np.abs(preds_c - trues_c))
        n_ext = (trues_c > thresh).sum()
        recall = (
            (preds_c[trues_c > thresh] > thresh[trues_c > thresh]).mean()
            if n_ext
            else float("nan")
        )
        print(
            f"  fold{fold}: ckpt={ckpt_name}  n={len(trues_c)}  r={r:.4f}  MAE={mae:.4f}degC  "
            f"n_extreme={n_ext} ({n_ext/len(trues_c)*100:.1f}%)  recall(pred>p90|truth>p90)={recall*100:.1f}%"
        )
        all_trues.append(trues_c)
        all_preds.append(preds_c)
        all_thresh.append(thresh)

    trues_c = np.concatenate(all_trues)
    preds_c = np.concatenate(all_preds)
    thresh = np.concatenate(all_thresh)
    r_pooled = np.corrcoef(preds_c, trues_c)[0, 1]
    mae_pooled = np.mean(np.abs(preds_c - trues_c))
    ext_mask = trues_c > thresh
    n_ext = ext_mask.sum()
    recall_pooled = (preds_c[ext_mask] > thresh[ext_mask]).mean()
    mean_pred_extreme = preds_c[ext_mask].mean()
    mean_true_extreme = trues_c[ext_mask].mean()
    std_ratio = preds_c.std() / trues_c.std()
    print(
        f"  POOLED (5 folds, n={len(trues_c)}): r={r_pooled:.4f}  MAE={mae_pooled:.4f}degC  "
        f"std(pred)/std(truth)={std_ratio:.3f}"
    )
    print(
        f"  POOLED extreme days (truth>p90, n={n_ext}, {n_ext/len(trues_c)*100:.1f}%): "
        f"recall(pred>p90)={recall_pooled*100:.1f}%  mean(pred)={mean_pred_extreme:.3f}degC  "
        f"mean(truth)={mean_true_extreme:.3f}degC"
    )
    return dict(
        r=r_pooled,
        mae=mae_pooled,
        recall=recall_pooled,
        n_extreme=n_ext,
        mean_pred_extreme=mean_pred_extreme,
        mean_true_extreme=mean_true_extreme,
        std_ratio=std_ratio,
    )


results = {}
# quantile already completed successfully in the previous run (job 29410634)
# -- rerunning only focal here, which crashed on p90_by_doy (fixed above).
results["focal"] = analyze(
    "full_gnll_focal", "TbotAtm_full_gnll_focal_seed42", "full_gnll_focal"
)

print(
    "\n=== SUMMARY vs narrative.md plain-GNLL baseline (r=0.218 restricted, 15.2% recall, n=1454) ==="
)
for k, v in results.items():
    print(
        f"{k:10s}: r={v['r']:.4f}  MAE={v['mae']:.4f}degC  recall={v['recall']*100:.1f}%  "
        f"n_extreme={v['n_extreme']}  std_ratio={v['std_ratio']:.3f}"
    )
