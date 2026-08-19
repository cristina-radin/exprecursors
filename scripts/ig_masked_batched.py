"""
ig_masked_batched.py — IG on masked model with batch processing (~5-7x faster).

Same logic as ig_masked_model.py but processes BATCH_SIZE samples per forward
pass. Run one job per seed (--seed 42 / 123 / 456 / 789 / 1337) and save
partial results; then run ig_masked_merge.py to combine and plot.

Usage (SLURM array, task 0-4 → seed index):
  python eval/ig_masked_batched.py --seed 42  --output_dir experiments/figures/xai_masked
  python eval/ig_masked_batched.py --seed 123 --output_dir experiments/figures/xai_masked
  ...

Outputs per seed (saved immediately, so partial results survive):
  ig_partial_seed{SEED}.npz — spatial_sum, peryear_mat, peryear_cnt, years, variables
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.paths import (
    EXPERIMENTS_DIR,
)

MASKED_DIR = Path(str(EXPERIMENTS_DIR))
ALL_SEEDS = [42, 123, 456, 789, 1337]
N_FOLDS = 5
BATCH_SIZE = 8


def best_ckpt(run_dir: Path) -> Path:
    ckpts = list((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/checkpoints/")

    def val_loss(p):
        m = re.search(r"val_loss=(-?[0-9]+\.[0-9]+)", str(p))
        return float(m.group(1)) if m else 0.0

    return min(ckpts, key=val_loss)


def load_model(ckpt_path: Path, cfg: dict, device: str) -> CNNLightningModule:
    inner = CNNLSTMModel(
        in_channels=cfg["in_channels"],
        cnn_features=cfg.get("cnn_features", 256),
        lstm_hidden=cfg.get("lstm_hidden", 512),
        lstm_layers=cfg.get("lstm_layers", 4),
        temporal_features=cfg.get("temporal_features", 0),
        dropout=cfg.get("dropout", 0.3),
        arch=cfg.get("arch", "lstm_only"),
        gaussian_nll=cfg.get("gaussian_nll", False),
        pooling=cfg.get("pooling", "max"),
        padding_mode=cfg.get("padding_mode", "zeros"),
        quantile_head=cfg.get("quantile_head", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path),
        model=inner,
        strict=False,
        map_location=device,
    )
    return lm.eval().to(device)


def compute_ig_batch(
    lm, xs_batch: torch.Tensor, xt_batch: torch.Tensor, n_steps: int
) -> torch.Tensor:
    """
    Batched Integrated Gradients.
    xs_batch: (B, T, C, H, W) on device
    xt_batch: (B, T, temporal_features) on device
    Returns:  (B, T, C, H, W) abs attributions on CPU
    """
    baseline = torch.zeros_like(xs_batch)
    grads = []
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    for step in range(n_steps):
        alpha = step / max(n_steps - 1, 1)
        interp = (baseline + alpha * (xs_batch - baseline)).requires_grad_(True)
        pred, _ = lm.model.forward_with_attention(interp.float(), xt_batch.float())
        pred[
            :, 0
        ].sum().backward()  # grad correct per-sample: d(sum pred) / d(interp_i) = d(pred_i)/d(interp_i)
        grads.append(interp.grad.detach().clone())  # (B, T, C, H, W)
    torch.backends.cudnn.enabled = prev_cudnn
    avg_grads = torch.stack(grads).mean(0)  # (B, T, C, H, W)
    attributions = (xs_batch - baseline) * avg_grads
    return attributions.abs().cpu()  # (B, T, C, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=ALL_SEEDS,
        help="Seed to process (one job per seed)",
    )
    parser.add_argument("--output_dir", default="experiments/figures/xai_masked")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Seed={args.seed}  device={device}  batch_size={args.batch_size}  n_steps={args.n_steps}"
    )

    spatial_sum = None  # (C, H, W)
    peryear_mat = None  # (n_years, C) — sum across runs
    peryear_cnt = None  # (n_years,)
    all_years = None
    variables = None
    n_runs = 0

    for fold in range(N_FOLDS):
        run_name = f"SSTAtm_lstmonly_gnll_masked_seed{args.seed}_fold{fold}"
        run_dir = MASKED_DIR / run_name

        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            print(f"SKIP {run_name}: no config")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        try:
            ckpt = best_ckpt(run_dir)
        except FileNotFoundError:
            print(f"SKIP {run_name}: no checkpoint")
            continue

        print(f"\n[seed={args.seed} fold={fold}] {run_name}")
        lm = load_model(ckpt, cfg, device)

        dm = LazyDataModule(config_path=str(cfg_path))
        dm.setup()
        test_ds = dm.test_dataloader().dataset
        base_ds = test_ds.dataset if hasattr(test_ds, "dataset") else test_ds
        indices = list(
            test_ds.indices if hasattr(test_ds, "indices") else range(len(test_ds))
        )

        if variables is None:
            variables = cfg["variables"]
            xs0, _, _ = base_ds[indices[0]]
            _, C_, H, W = xs0.shape
            spatial_sum = np.zeros((C_, H, W), dtype=np.float64)

        # Collect per-year info for this run
        run_peryear: dict = {}  # year -> list of (C,)

        B = args.batch_size
        n_done = 0
        for batch_start in range(0, len(indices), B):
            batch_gidx = indices[batch_start : batch_start + B]

            xs_list, xt_list, yr_list = [], [], []
            for gi in batch_gidx:
                xs, xt, _ = base_ds[gi]
                target_t = gi + base_ds.window_size - 1 + base_ds.lead_time
                yr = int(base_ds.years[target_t])
                xs_list.append(xs)
                xt_list.append(xt)
                yr_list.append(yr)

            xs_batch = torch.stack(xs_list).to(device)  # (B, T, C, H, W)
            xt_batch = torch.stack(xt_list).to(device)  # (B, T, tf)

            with torch.no_grad():
                pass  # model weights frozen — grad only on interp
            attrs = compute_ig_batch(
                lm, xs_batch, xt_batch, args.n_steps
            )  # (B,T,C,H,W)

            for b, yr in enumerate(yr_list):
                a = attrs[b]  # (T, C, H, W)
                spatial_sum += a.mean(0).numpy()
                var_imp = a.mean(dim=(0, 2, 3)).numpy()  # (C,)
                run_peryear.setdefault(yr, []).append(var_imp)

            n_done += len(batch_gidx)
            if n_done % 200 < B:
                print(f"  {n_done}/{len(indices)}", end="\r", flush=True)

        print(f"  Done: {len(indices)} samples")

        # Accumulate peryear into matrix
        run_yrs = sorted(run_peryear.keys())
        if all_years is None:
            all_years = run_yrs
            peryear_mat = np.zeros((len(all_years), len(variables)), dtype=np.float64)
            peryear_cnt = np.zeros(len(all_years), dtype=np.int64)

        for yr, samples in run_peryear.items():
            if yr in all_years:
                yi = all_years.index(yr)
            else:
                # Extend arrays if needed
                all_years.append(yr)
                all_years.sort()
                yi = all_years.index(yr)
                peryear_mat = np.insert(peryear_mat, yi, 0, axis=0)
                peryear_cnt = np.insert(peryear_cnt, yi, 0)
            peryear_mat[yi] += np.mean(samples, axis=0)
            peryear_cnt[yi] += 1

        n_runs += 1

    if n_runs == 0:
        print("No runs completed.")
        return

    # Save partial results for this seed
    out_npz = out_dir / f"ig_partial_seed{args.seed}.npz"
    np.savez(
        str(out_npz),
        spatial_sum=spatial_sum,
        peryear_mat=peryear_mat,
        peryear_cnt=peryear_cnt,
        years=np.array(all_years),
        variables=np.array(variables, dtype=object),
        n_runs=np.array(n_runs),
    )
    print(f"\nSaved {out_npz}  (n_runs={n_runs})")
    print("=== Variable importance (this seed) ===")
    avg = spatial_sum / n_runs
    imp = avg.mean(axis=(1, 2))
    for v, i in zip(variables, imp / imp.sum() * 100):
        print(f"  {v:<12} {i:.1f}%")


if __name__ == "__main__":
    main()
