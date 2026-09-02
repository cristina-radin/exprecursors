"""
incremental_value_regression.py — Aug 23 2026, user's reframe: "el
modelo pierde ante persistencia" conflates two different questions
(precursor-based forecast vs. state-based forecast) since the model was
never given y(t) (the target's own recent value) as input, only
ptho_bot (a different, imperfect proxy) + atmosphere. Persistence wins
on raw r trivially because it has access to information (today's true
state) the model was deliberately never shown.

Idea 1 (do first, free, decisive): fit y_true ~ persist + q_pred (OLS),
check if q_pred's coefficient is significant beyond persist alone --
if so, the model adds real information over the state-based baseline,
and the fitted hybrid alpha*persist + beta*q_pred should beat pure
persistence on raw r (flips the headline from negative to positive:
"precursors add X% over the state baseline"). Also computes the
partial correlation of q_pred with y_true controlling for persist, a
single citable number.

Idea 3 (same aligned data, do alongside): does q_pred capture the TREND
away from persistence -- correlate (q_pred - persist) against
(y_true - persist). Directly connects to the onset story: onset is
exactly when true value moves fastest away from the recent state.

Reuses scripts/eval_recall_v2_partition.py's already-saved npz
(trues_c, mean_c, q_c) -- no model reload. Only NEW computation is
persist_pred, aligned to the exact same fold-pooling order (data-only,
no model, same logic as persistence_recall_baseline.py). Alignment is
VERIFIED, not assumed: freshly recomputed trues (from the data-only
loop) must exactly match the saved trues_c before trusting anything
downstream.

CPU only, no GPU, no model inference at all.

Usage:
  python scripts/analysis/incremental_value_regression.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402

FIGURES_DIR = REPO_ROOT / "experiments" / "figures"

FAMILY_CONFIG_DIR = {
    "full_lead7": "full_gnll_quantile_v2_landfill",
    "local": "local",
    "remote": "remote",
    "lead3": "lead3_landfill",
    "lead5": "lead5_landfill",
    "lead14": "lead14_landfill",
    "lead30": "lead30_landfill",
}


def compute_persist_pred(config_dir):
    """Data-only, no model: persistence prediction aligned to the exact
    same fold-pooling order eval_recall_v2_partition.py uses (fold 0..4,
    each fold's test_indices in their own order, concatenated)."""
    all_trues, all_persist = [], []
    for fold in range(5):
        cfg_path = REPO_ROOT / "configs" / "partition" / config_dir / f"fold{fold}.yaml"
        dm = LazyDataModule(str(cfg_path))
        dm.setup()
        full_ds = dm.test_dataset.dataset
        test_indices = dm.test_dataset.indices

        target_idx = np.array(
            [i + full_ds.window_size - 1 + full_ds.lead_time for i in test_indices]
        )
        persist_idx = target_idx - full_ds.lead_time
        all_trues.append(full_ds.target[target_idx].numpy())
        all_persist.append(full_ds.target[persist_idx].numpy())
    return np.concatenate(all_trues), np.concatenate(all_persist)


def partial_corr(x, y, z):
    """Partial correlation of x,y controlling for z: correlate the
    residuals of x~z and y~z."""
    bx = np.polyfit(z, x, 1)
    by = np.polyfit(z, y, 1)
    rx = x - np.polyval(bx, z)
    ry = y - np.polyval(by, z)
    return np.corrcoef(rx, ry)[0, 1]


def main():
    for label, config_dir in FAMILY_CONFIG_DIR.items():
        npz_path = FIGURES_DIR / "_fold_cache" / f"eval_recall_v2_{label}.npz"
        if not npz_path.exists():
            print(f"{label}: {npz_path} not found, skipping", flush=True)
            continue
        d = np.load(npz_path)
        trues_c, mean_c, q_c = d["trues_c"], d["mean_c"], d["q_c"]

        trues_fresh, persist = compute_persist_pred(config_dir)

        # Alignment check -- do not trust anything below if this fails.
        if len(trues_fresh) != len(trues_c):
            print(
                f"{label}: LENGTH MISMATCH fresh={len(trues_fresh)} saved={len(trues_c)} -- ABORTING for this family",
                flush=True,
            )
            continue
        max_diff = np.abs(trues_fresh - trues_c).max()
        if max_diff > 1e-3:
            print(
                f"{label}: ALIGNMENT CHECK FAILED, max|diff|={max_diff:.6f} -- ABORTING for this family",
                flush=True,
            )
            continue
        print(f"{label}: alignment verified, max|diff|={max_diff:.2e}", flush=True)

        r_persist = np.corrcoef(trues_c, persist)[0, 1]

        for head_name, pred in [("mean", mean_c), ("quantile", q_c)]:
            # OLS: trues_c = a0 + a1*persist + a2*pred
            X = np.column_stack([np.ones_like(persist), persist, pred])
            beta, residuals, rank, sv = np.linalg.lstsq(X, trues_c, rcond=None)
            n, k = X.shape
            y_hat = X @ beta
            resid = trues_c - y_hat
            sigma2 = (resid @ resid) / (n - k)
            XtX_inv = np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(sigma2 * XtX_inv))
            t_stats = beta / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

            r2_full = 1 - (resid @ resid) / (
                (trues_c - trues_c.mean()) @ (trues_c - trues_c.mean())
            )
            r2_persist_only = r_persist**2

            pcorr = partial_corr(pred, trues_c, persist)

            hybrid_pred = X @ beta
            r_hybrid = np.corrcoef(trues_c, hybrid_pred)[0, 1]

            # Idea 3: trend skill -- does pred capture deviation from persistence?
            trend_true = trues_c - persist
            trend_pred = pred - persist
            r_trend = np.corrcoef(trend_pred, trend_true)[0, 1]

            print(
                f"  [{label}/{head_name}] r_persist={r_persist:.4f}  r_hybrid={r_hybrid:.4f}  "
                f"beta(pred)={beta[2]:.4f} (p={p_values[2]:.2e})  "
                f"R2_persist_only={r2_persist_only:.4f}  R2_full={r2_full:.4f}  "
                f"partial_corr={pcorr:.4f}  r_trend(pred-persist vs true-persist)={r_trend:.4f}",
                flush=True,
            )

        print(flush=True)


if __name__ == "__main__":
    main()
