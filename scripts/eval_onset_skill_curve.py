"""
eval_onset_skill_curve.py — skill (Pearson r) as a function of "days
since MHW onset" (0, 1, 2, ...), Aug 21 2026, requested by the user after
seeing the binary onset/mid_event split (eval_onset_skill_quantile_v2.py,
n=56 onset days, r_mean=-0.29 significant negative): "por si es demasiado
restrictivo, quiza no solo un dia sino algunos mas".

Avoids picking an arbitrary "first N days" window by showing the full
recovery curve instead -- how quickly (if at all) skill goes from
negative/near-zero at day 0 to positive as an event matures, for BOTH
model heads and persistence, pooled across all 5 folds. Also checks
per-fold consistency at day 0 specifically, since n=56 total (~11/fold)
is small enough that a couple of unusual events could dominate the
pooled result.

Reuses: src/utils/hobday.py (apply_hobday, load_ns_p90), LazyDataModule +
best_ckpt() + load_model_config() + CNNLightningModule.load_from_checkpoint
(same pattern as eval_onset_skill_quantile_v2.py / quantile_head_recall_v2_all5.py).

CPU only. Not part of the permanent pipeline.

Usage:
  python scripts/eval_onset_skill_curve.py
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.hobday import apply_hobday, load_ns_p90  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments" / "figures" / "step7_persistence"
LEAD = 7
N_FOLDS = 5
MAX_DAYS = 14  # days since onset to show; sample size will thin out toward the tail
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)

p90_ns = load_ns_p90()


def days_since_onset(mhw):
    """For each day, how many days into its (merged, Hobday) MHW event it
    is (0 = onset day itself). -1 for non-MHW days."""
    dso = np.full(len(mhw), -1, dtype=int)
    counter = -1
    for i in range(len(mhw)):
        if mhw[i]:
            counter = 0 if (i == 0 or not mhw[i - 1]) else counter + 1
            dso[i] = counter
        else:
            counter = -1
    return dso


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
    mean_preds, q_preds, trues = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            xs, xt, y = batch[0], batch[1], batch[2]
            xs, xt = xs.float().to(device), xt.float().to(device)
            y_hat, q_pred = lm.model.forward_with_quantile(xs, xt)
            mean_preds.append(y_hat[:, 0].cpu())
            q_preds.append(q_pred.squeeze(-1).cpu())
            trues.append(y.squeeze(-1).cpu())
    mean_preds = torch.cat(mean_preds).numpy() * lm.target_std + lm.target_mean
    q_preds = torch.cat(q_preds).numpy() * lm.target_std + lm.target_mean
    trues = torch.cat(trues).numpy() * lm.target_std + lm.target_mean

    target_idx = np.array(
        [i + full_ds.window_size - 1 + full_ds.lead_time for i in test_indices]
    )
    order = np.argsort(target_idx)
    target_idx, trues, mean_preds, q_preds = (
        target_idx[order],
        trues[order],
        mean_preds[order],
        q_preds[order],
    )
    doys = full_ds.doys[target_idx]
    doys = np.where(doys >= 365, 365, doys).astype(int)
    years = full_ds.years[target_idx]

    # stratified_kfold assigns NON-CONSECUTIVE years to each fold's test
    # set (e.g. fold0: 1985, 1991, 2000, ...) -- apply_hobday's gap-closure
    # logic assumes a contiguous daily series, so running it on the whole
    # fold at once could spuriously bridge the tail of one test year into
    # the head of an unrelated, calendar-distant one. Apply per-year
    # instead (each year's own samples ARE internally contiguous), then
    # reassemble in the same sorted order.
    thr = p90_ns[doys - 1]
    exceed = trues > thr
    mhw = np.zeros(len(trues), dtype=bool)
    for yr in np.unique(years):
        ymask = years == yr
        mhw[ymask] = apply_hobday(exceed[ymask])
    dso = np.zeros(len(trues), dtype=int) - 1
    for yr in np.unique(years):
        ymask = years == yr
        dso[ymask] = days_since_onset(mhw[ymask])

    persist = np.full_like(trues, np.nan)
    persist[LEAD:] = trues[:-LEAD]
    # invalidate persistence pairs that cross a year boundary too (same
    # reasoning -- trues[i-7] must be 7 real days before trues[i], not 7
    # positions before in a year-jumping concatenated array)
    year_of_i = years
    year_of_i_minus_lead = np.concatenate([np.full(LEAD, -999), years[:-LEAD]])
    persist[year_of_i != year_of_i_minus_lead] = np.nan

    return trues, mean_preds, q_preds, persist, dso


all_true, all_mean, all_q, all_persist, all_dso, fold_ids = [], [], [], [], [], []
for fold in range(N_FOLDS):
    trues, mean_preds, q_preds, persist, dso = run_fold(fold)
    all_true.append(trues)
    all_mean.append(mean_preds)
    all_q.append(q_preds)
    all_persist.append(persist)
    all_dso.append(dso)
    fold_ids.append(np.full(len(trues), fold))
    n_events_this_fold = int((dso == 0).sum())
    print(f"  fold{fold}: n_onset_days={n_events_this_fold}", flush=True)

true_a = np.concatenate(all_true)
mean_a = np.concatenate(all_mean)
q_a = np.concatenate(all_q)
persist_a = np.concatenate(all_persist)
dso_a = np.concatenate(all_dso)
fold_a = np.concatenate(fold_ids)

# ── Per-fold robustness check at day 0 (onset) specifically ────────────────
print(f"\n{'='*70}")
print("Per-fold check at day 0 (onset) -- is the pooled result driven by 1-2 folds?")
print(f"{'='*70}")
for fold in range(N_FOLDS):
    mask = (dso_a == 0) & (fold_a == fold) & np.isfinite(persist_a)
    n = int(mask.sum())
    if n < 4:
        print(f"  fold{fold}: n={n} (too few for r)")
        continue
    r_m, _ = pearsonr(mean_a[mask], true_a[mask])
    r_q, _ = pearsonr(q_a[mask], true_a[mask])
    r_p, _ = pearsonr(persist_a[mask], true_a[mask])
    print(
        f"  fold{fold}: n={n:2d}  r_mean={r_m:+.3f}  r_quantile={r_q:+.3f}  r_persist={r_p:+.3f}"
    )

# ── Skill curve vs days since onset ─────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Skill vs. days since onset (pooled 5 folds, up to day {MAX_DAYS})")
print(f"{'='*70}")

days_range = list(range(0, MAX_DAYS + 1))
r_mean_curve, r_q_curve, r_persist_curve, n_curve = [], [], [], []
ci_mean_curve, ci_q_curve, ci_persist_curve = [], [], []


def r_ci(r, n):
    if n <= 3 or np.isnan(r):
        return (float("nan"), float("nan"))
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


for d in days_range:
    mask = (dso_a == d) & np.isfinite(persist_a)
    n = int(mask.sum())
    n_curve.append(n)
    if n < 8:
        r_mean_curve.append(np.nan)
        r_q_curve.append(np.nan)
        r_persist_curve.append(np.nan)
        ci_mean_curve.append((np.nan, np.nan))
        ci_q_curve.append((np.nan, np.nan))
        ci_persist_curve.append((np.nan, np.nan))
        print(f"  day {d:2d}: n={n:4d}  (too few, skipped)")
        continue
    r_m, _ = pearsonr(mean_a[mask], true_a[mask])
    r_q, _ = pearsonr(q_a[mask], true_a[mask])
    r_p, _ = pearsonr(persist_a[mask], true_a[mask])
    r_mean_curve.append(r_m)
    r_q_curve.append(r_q)
    r_persist_curve.append(r_p)
    ci_mean_curve.append(r_ci(r_m, n))
    ci_q_curve.append(r_ci(r_q, n))
    ci_persist_curve.append(r_ci(r_p, n))
    print(
        f"  day {d:2d}: n={n:4d}  r_mean={r_m:+.3f} [{ci_mean_curve[-1][0]:+.3f},{ci_mean_curve[-1][1]:+.3f}]  "
        f"r_quantile={r_q:+.3f}  r_persist={r_p:+.3f}"
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

days_arr = np.array(days_range)
r_mean_arr = np.array(r_mean_curve)
r_q_arr = np.array(r_q_curve)
r_persist_arr = np.array(r_persist_curve)
ci_mean_lo = np.array([c[0] for c in ci_mean_curve])
ci_mean_hi = np.array([c[1] for c in ci_mean_curve])

ax1.axhline(0, color="black", lw=0.8)
ax1.fill_between(days_arr, ci_mean_lo, ci_mean_hi, color="#2166ac", alpha=0.15)
ax1.plot(days_arr, r_mean_arr, "o-", color="#2166ac", label="Model (mean head)")
ax1.plot(days_arr, r_q_arr, "o-", color="#d6604d", label="Model (quantile head)")
ax1.plot(days_arr, r_persist_arr, "o-", color="#e08214", label="Persistence (lag-7)")
ax1.set_ylabel("Pearson r")
ax1.set_title(
    "Skill vs. days since MHW onset — full_gnll_quantile_v2 (pooled 5 folds, 95% CI shaded for mean head)"
)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.bar(days_arr, n_curve, color="gray", alpha=0.6)
ax2.set_xlabel("Days since onset")
ax2.set_ylabel("n samples")
ax2.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "onset_skill_curve.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}", flush=True)

np.savez(
    OUT_DIR / "onset_skill_curve.npz",
    days=days_arr,
    r_mean=r_mean_arr,
    r_q=r_q_arr,
    r_persist=r_persist_arr,
    n=np.array(n_curve),
)
print(f"Saved {OUT_DIR / 'onset_skill_curve.npz'}", flush=True)
