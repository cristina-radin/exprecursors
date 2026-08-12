"""
temporal_shuffle_eval.py — Tests whether masked model uses temporal dynamics.

Loads existing masked model checkpoints (SSTAtm_lstmonly_gnll_masked) and
evaluates on the test set with the 60-day input window TIME-SHUFFLED.

If the model only uses the average state of the 60-day window (ignoring order),
skill should be unchanged. If skill drops significantly → model uses temporal
dynamics (not just mean-state remote SST correlation).

Compares:
  r_original  — normal test set evaluation
  r_shuffled  — same samples, but T=60 window randomly shuffled per sample

Usage:
  python eval/temporal_shuffle_eval.py
  python eval/temporal_shuffle_eval.py --n_shuffle 10 --output experiments/figures/
"""

import argparse, sys, glob, numpy as np, torch, yaml
from pathlib import Path
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset    import MHWDataset
from src.data.datamodule import MHWDataModule
from model      import CNNLightningModule, build_model


MASKED_DIR = Path("/p/project1/hai_1127/radin1/exprecursors/experiments/masked")
N_FOLDS    = 5
SEEDS      = [42, 123, 456, 789, 1337]


def best_ckpt(run_dir):
    ckpts = sorted(glob.glob(str(run_dir / "checkpoints" / "*.ckpt")))
    if not ckpts:
        return None
    # pick lowest val_loss from filename
    def val_loss(p):
        import re
        m = re.search(r"val_loss=([-\d.]+)", p)
        return float(m.group(1)) if m else 0.0
    return min(ckpts, key=val_loss)


def evaluate(model, loader, shuffle_window=False, n_shuffle=1, rng=None):
    """Run inference; optionally shuffle the T dimension of x_spatial."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_spatial, x_temporal, y in loader:
            if shuffle_window:
                # Average r over n_shuffle permutations
                preds_batch = []
                for _ in range(n_shuffle):
                    perm = rng.permutation(x_spatial.shape[1])
                    xs_perm = x_spatial[:, perm, :, :, :]
                    xt_perm = x_temporal[:, perm, :]
                    out = model(xs_perm.float(), xt_perm.float())
                    mu = out[:, 0] if model.gaussian_nll else out[:, 0]
                    preds_batch.append(mu.cpu().numpy())
                preds = np.mean(preds_batch, axis=0)
            else:
                out   = model(x_spatial.float(), x_temporal.float())
                preds = (out[:, 0]).cpu().numpy()

            y_denorm = (y.squeeze().numpy() * model.target_std + model.target_mean)
            p_denorm = preds * model.target_std + model.target_mean
            all_preds.append(p_denorm)
            all_targets.append(y_denorm)

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    r, _    = pearsonr(preds, targets)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shuffle", type=int, default=20,
                        help="Number of random permutations to average per sample")
    parser.add_argument("--output", default="experiments/figures")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    r_orig_all, r_shuf_all = [], []

    for fold in range(N_FOLDS):
        for seed in SEEDS:
            run_name = f"SSTAtm_lstmonly_gnll_masked_seed{seed}_fold{fold}"
            run_dir  = MASKED_DIR / run_name
            ckpt     = best_ckpt(run_dir)
            if ckpt is None:
                print(f"  SKIP {run_name}: no checkpoint")
                continue

            # Load config
            cfg_path = run_dir / "config.yaml"
            with open(cfg_path) as f:
                config = yaml.safe_load(f)

            # Build dataset + datamodule to get test split
            dm = MHWDataModule(config)
            dm.setup()
            test_loader = dm.test_dataloader()

            # Load model
            lm = CNNLightningModule.load_from_checkpoint(
                ckpt,
                model=build_model(config, dm.dataset),
                strict=False,
            )
            lm.eval()

            r_orig = evaluate(lm, test_loader, shuffle_window=False)
            r_shuf = evaluate(lm, test_loader, shuffle_window=True,
                              n_shuffle=args.n_shuffle, rng=rng)

            r_orig_all.append(r_orig)
            r_shuf_all.append(r_shuf)
            print(f"  {run_name}: r_orig={r_orig:.4f}  r_shuffled={r_shuf:.4f}  "
                  f"drop={r_orig-r_shuf:.4f}")

    print(f"\n{'='*60}")
    print(f"Original (ordered window):    r = {np.mean(r_orig_all):.4f} ± {np.std(r_orig_all):.4f}")
    print(f"Shuffled window (no order):   r = {np.mean(r_shuf_all):.4f} ± {np.std(r_shuf_all):.4f}")
    print(f"Drop from shuffling:          Δr = {np.mean(r_orig_all)-np.mean(r_shuf_all):.4f}")
    print(f"\nIf Δr ≈ 0: model ignores temporal order → signal is correlation with mean state.")
    print(f"If Δr >> 0: model uses temporal dynamics → signal is real predictability.")

    with open(out_dir / "temporal_shuffle_eval.txt", "w") as f:
        f.write(f"n_shuffle={args.n_shuffle}  n_runs={len(r_orig_all)}\n")
        f.write(f"Original:  r = {np.mean(r_orig_all):.4f} ± {np.std(r_orig_all):.4f}\n")
        f.write(f"Shuffled:  r = {np.mean(r_shuf_all):.4f} ± {np.std(r_shuf_all):.4f}\n")
        f.write(f"Drop:     Δr = {np.mean(r_orig_all)-np.mean(r_shuf_all):.4f}\n")
    print(f"Saved to {out_dir / 'temporal_shuffle_eval.txt'}")


if __name__ == "__main__":
    main()
