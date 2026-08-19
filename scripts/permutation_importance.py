"""
permutation_importance.py — Variable importance via permutation on TbotAtm models.

For each variable: shuffle its values across test samples (breaking the
variable→target relationship), re-evaluate R², and report the drop.
No retraining — just forward passes on shuffled data.

Usage:
  python scripts/permutation_importance.py --fold 0 --partition full --model_tag mse_v2
  python scripts/permutation_importance.py --fold 0 --partition full --model_tag mse_v2 --n_repeats 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.checkpoints import best_ckpt

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "partition"


def _config_subdir(partition, model_tag):
    return f"{partition}_{model_tag}" if model_tag else partition


def _run_name(partition, model_tag, fold):
    tag = f"_{model_tag}" if model_tag else ""
    return f"TbotAtm_{partition}{tag}_seed42_fold{fold}"


def load_model(fold, partition, device, model_tag=""):
    cfg_path = (
        REPO_ROOT
        / "configs"
        / "partition"
        / _config_subdir(partition, model_tag)
        / f"fold{fold}.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    inner = CNNLSTMModel(
        in_channels=cfg["in_channels"],
        cnn_features=cfg.get("cnn_features", 128),
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
        temporal_features=cfg.get("temporal_features", 0),
        dropout=cfg.get("dropout", 0.2),
        arch=cfg.get("arch", "lstm_only"),
        gaussian_nll=cfg.get("gaussian_nll", False),
        pooling=cfg.get("pooling", "max"),
        quantile_head=cfg.get("quantile_head", False),
    )
    run_dir = EXPERIMENTS_DIR / _run_name(partition, model_tag, fold)
    ckpt = best_ckpt(run_dir / "checkpoints")
    print(f"  Checkpoint: {ckpt.name}")
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=False, map_location=device
    )
    return lm.eval().to(device), cfg


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Forward pass over entire dataloader. Returns (all_preds, all_targets) as numpy."""
    preds, targets = [], []
    for xs, xt, y in dataloader:
        xs, xt = xs.to(device), xt.to(device)
        pred = model(xs.float(), xt.float())  # (B, 1)
        preds.append(pred.cpu().numpy().squeeze())
        targets.append(y.numpy().squeeze())
    return np.concatenate(preds), np.concatenate(targets)


def compute_r2(preds, targets):
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument(
        "--partition", choices=["full", "remote", "local"], default="full"
    )
    parser.add_argument("--model_tag", default="mse_v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=1,
        help="Repeat each permutation N times with different seeds (more stable)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"\n=== Permutation importance | fold={args.fold} | partition={args.partition} "
        f"| model_tag={args.model_tag} | device={device} | n_repeats={args.n_repeats} ==="
    )

    lm, cfg = load_model(args.fold, args.partition, device, args.model_tag)
    variables = cfg["variables"]
    C = len(variables)
    model = lm.model

    cfg_path = str(
        REPO_ROOT
        / "configs"
        / "partition"
        / _config_subdir(args.partition, args.model_tag)
        / f"fold{args.fold}.yaml"
    )
    dm = LazyDataModule(cfg_path)
    dm.setup()
    test_dl = DataLoader(
        dm.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    n_test = len(dm.test_dataset)
    print(f"  Test samples: {n_test}")

    # Collect all data into memory for efficient shuffling
    print("  Loading test set into memory...")
    all_xs, all_xt, all_y = [], [], []
    for xs, xt, y in test_dl:
        all_xs.append(xs)
        all_xt.append(xt)
        all_y.append(y)
    all_xs = torch.cat(all_xs)  # (N, T, C, H, W)
    all_xt = torch.cat(all_xt)  # (N, T, 3)
    all_y = torch.cat(all_y)  # (N,)
    print(f"  Loaded: xs={all_xs.shape}, xt={all_xt.shape}, y={all_y.shape}")

    # Baseline R²
    print("\n--- Baseline ---")
    t0 = time.time()
    baseline_ds = torch.utils.data.TensorDataset(all_xs, all_xt, all_y)
    baseline_dl = DataLoader(
        baseline_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    preds_base, _ = evaluate(model, baseline_dl, device)
    r2_base = compute_r2(preds_base, all_y.numpy())
    print(f"  Baseline R² = {r2_base:.6f}  ({time.time() - t0:.1f}s)")

    # Permutation importance
    results = {}
    print("\n--- Permutation importance ---")
    for c_idx, var in enumerate(variables):
        r2_perms = []
        for rep in range(args.n_repeats):
            rng = np.random.RandomState(42 + rep)
            perm_idx = rng.permutation(len(all_xs))

            # Shuffle only this variable's channel across samples
            xs_perm = all_xs.clone()
            xs_perm[:, :, c_idx, :, :] = all_xs[perm_idx, :, c_idx, :, :]

            perm_ds = torch.utils.data.TensorDataset(xs_perm, all_xt, all_y)
            perm_dl = DataLoader(
                perm_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            preds_perm, _ = evaluate(model, perm_dl, device)
            r2_perm = compute_r2(preds_perm, all_y.numpy())
            r2_perms.append(r2_perm)

        r2_mean = np.mean(r2_perms)
        r2_std = np.std(r2_perms) if len(r2_perms) > 1 else 0.0
        importance = r2_base - r2_mean
        results[var] = {"r2_perm": r2_mean, "r2_std": r2_std, "importance": importance}
        print(
            f"  {var:10s}: R²_perm={r2_mean:.6f} ± {r2_std:.6f}  "
            f"| ΔR² = {importance:+.6f} ({importance / abs(r2_base) * 100:.1f}% of baseline)"
            if r2_base != 0
            else f"  {var:10s}: R²_perm={r2_mean:.6f} ± {r2_std:.6f}  | ΔR² = {importance:+.6f}"
        )

    # Summary
    print("\n--- Summary (sorted by importance) ---")
    ranked = sorted(results.items(), key=lambda x: x[1]["importance"], reverse=True)
    for rank, (var, r) in enumerate(ranked, 1):
        print(
            f"  {rank}. {var:10s}  ΔR² = {r['importance']:+.6f}  "
            f"(R²_perm={r['r2_perm']:.6f})"
        )
    print(f"\n  Baseline R² = {r2_base:.6f}")

    # Save
    out_dir = REPO_ROOT / "experiments" / f"permutation_{args.partition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"permutation_importance_fold{args.fold}.npz",
        variables=np.array(variables),
        r2_baseline=r2_base,
        r2_permuted=np.array([results[v]["r2_perm"] for v in variables]),
        r2_permuted_std=np.array([results[v]["r2_std"] for v in variables]),
        importance=np.array([results[v]["importance"] for v in variables]),
        n_samples=n_test,
        fold=args.fold,
        partition=args.partition,
        model_tag=args.model_tag,
    )

    # Bar chart
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    imp = np.array([results[v]["importance"] for v in variables])
    order = np.argsort(imp)[::-1]
    colors = ["#d6604d" if imp[i] > 0 else "#4393c3" for i in order]
    ax.bar(range(C), imp[order], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(C))
    ax.set_xticklabels([variables[i] for i in order], fontsize=11)
    ax.set_ylabel("ΔR² (baseline − permuted)", fontsize=10)
    ax.set_title(
        f"Permutation importance — TbotAtm {args.partition.title()} fold{args.fold}",
        fontsize=11,
    )
    ax.axhline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / f"permutation_importance_fold{args.fold}.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {out_dir}/permutation_importance_fold{args.fold}.png")
    print(f"  Saved: {out_dir}/permutation_importance_fold{args.fold}.npz")


if __name__ == "__main__":
    main()
