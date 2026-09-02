"""
linear_ceiling_ridge.py — Aug 23 2026, user's idea 4: locate the CNN-
LSTM's r (~0.81 for full_lead7) relative to the LINEAR information
extractable from the precursor window, using a cheap ridge regression
-- not the full raw spatial input (60x5x141x201 per sample, ~8.5M
features, hopelessly overfit at n~14500 and not what "linear ceiling"
usually means in practice), but the NS-box-mean time series per
variable per day (5 vars x 60-day window = 300 features), a standard,
defensible reduction. Computed directly from the raw netCDF via
xarray (spatial mean over the NS box), NOT via LazyDataset/DataLoader
-- avoids loading any per-sample spatial tensor, orders of magnitude
cheaper.

Fits ridge per fold (5-fold stratified_kfold split, same
train/val/test years as the CNN-LSTM pipeline, reused via
LazyDataModule for split indices only, not tensor loading), pools test
predictions across folds exactly like the CNN's own pooled r.

CPU only, no GPU.

Usage:
  python scripts/analysis/linear_ceiling_ridge.py --config_dir full_gnll_quantile_v2_landfill --label full_lead7
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from sklearn.linear_model import RidgeCV

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402

_NS_LAT_SLICE = slice(50.0, 63.0)
_NS_LON_SLICE = slice(-5.0, 13.0)
VARIABLES = ["ptho_bot", "u10", "v10", "msl", "ssr"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    print(
        "Loading NS-box-mean series for all 5 variables (cheap, xarray only)...",
        flush=True,
    )
    ds = xr.open_dataset(DATA_FILE)
    ns_series = {}
    for var in VARIABLES:
        field = ds[var].sel(lat=_NS_LAT_SLICE, lon=_NS_LON_SLICE)
        ns_series[var] = field.mean(dim=["lat", "lon"], skipna=True).values.astype(
            np.float64
        )
    X_full = np.column_stack([ns_series[v] for v in VARIABLES])  # (T, 5)

    all_test_true, all_test_pred = [], []
    for fold in range(5):
        cfg_path = (
            REPO_ROOT / "configs" / "partition" / args.config_dir / f"fold{fold}.yaml"
        )
        cfg = yaml.safe_load(open(cfg_path))
        window_size, lead_time = cfg["window_size"], cfg["lead_time"]

        dm = LazyDataModule(str(cfg_path))
        dm.setup()
        train_idx = dm.train_dataset.indices
        test_idx = dm.test_dataset.indices
        full_ds = dm.test_dataset.dataset

        def build_features(indices):
            feats = np.zeros((len(indices), window_size * len(VARIABLES)))
            targets = np.zeros(len(indices))
            for k, i in enumerate(indices):
                window = X_full[i : i + window_size]  # (window, 5)
                feats[k] = window.flatten()
                targets[k] = full_ds.target[i + window_size - 1 + lead_time].item()
            return feats, targets

        X_train, y_train = build_features(train_idx)
        X_test, y_test = build_features(test_idx)

        # standardize using train stats only
        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        X_train_s = (X_train - mu) / sigma
        X_test_s = (X_test - mu) / sigma

        model = RidgeCV(alphas=np.logspace(-2, 4, 20))
        model.fit(X_train_s, y_train)
        pred_test = model.predict(X_test_s)

        r_fold = np.corrcoef(y_test, pred_test)[0, 1]
        print(
            f"  fold {fold}: n_train={len(train_idx)} n_test={len(test_idx)} best_alpha={model.alpha_:.3g}  r_test={r_fold:.4f}",
            flush=True,
        )

        all_test_true.append(y_test)
        all_test_pred.append(pred_test)

    y_true_pooled = np.concatenate(all_test_true)
    y_pred_pooled = np.concatenate(all_test_pred)
    r_pooled = np.corrcoef(y_true_pooled, y_pred_pooled)[0, 1]
    mae_pooled = np.abs(y_true_pooled - y_pred_pooled).mean()

    print(
        f"\n=== LINEAR CEILING (ridge, NS-box-mean x 60d x 5 vars = 300 features), {args.label} ===",
        flush=True,
    )
    print(f"Pooled 5-fold r = {r_pooled:.4f}, MAE = {mae_pooled:.4f}", flush=True)

    out_dir = REPO_ROOT / "experiments" / "figures" / "step7_persistence"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"linear_ceiling_ridge_{args.label}.npz",
        y_true=y_true_pooled,
        y_pred=y_pred_pooled,
        r_pooled=r_pooled,
        mae_pooled=mae_pooled,
    )
    print(f"Saved {out_dir / f'linear_ceiling_ridge_{args.label}.npz'}", flush=True)


if __name__ == "__main__":
    main()
