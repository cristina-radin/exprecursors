#!/usr/bin/env python
"""
mhw_ensemble_hobday.py — Monte Carlo Hobday MHW detection for GNLL predictions.

Respects the literal Hobday algorithm (threshold-exceedance + min-duration +
gap-closure) by applying it to sampled deterministic trajectories drawn from
the GNLL predictive distribution N(mean, std) at each day, then aggregating
across the ensemble — instead of thresholding the predicted mean directly.

Usage:
  python scripts/mhw_ensemble_hobday.py --n_samples 500 --vote_threshold 0.5
"""

import argparse
import datetime as _dt

import numpy as np

from src.utils.hobday import apply_hobday, load_ns_p90
from src.utils.paths import EXPERIMENTS_DIR

PARTITION = "full"
MODEL_TAG = "gnll"


def _run_name(fold: int) -> str:
    return f"TbotAtm_{PARTITION}_{MODEL_TAG}_seed42_fold{fold}"


def load_all_folds():
    all_d, all_t, all_p, all_s = [], [], [], []
    for f in range(5):
        path = EXPERIMENTS_DIR / "partition" / _run_name(f) / "test_predictions.npz"
        d = np.load(path, allow_pickle=True)
        all_d.append(d["dates"].astype("datetime64[D]"))
        all_t.append(d["trues_degC"])
        all_p.append(d["preds_degC"])
        all_s.append(d["std_degC"])
    dates = np.concatenate(all_d)
    trues = np.concatenate(all_t)
    preds = np.concatenate(all_p)
    stds = np.concatenate(all_s)
    order = np.argsort(dates)
    return dates[order], trues[order], preds[order], stds[order]


def doy_threshold(dates: np.ndarray, p90: np.ndarray) -> np.ndarray:
    doy = np.array(
        [
            _dt.date(int(str(d)[:4]), int(str(d)[5:7]), int(str(d)[8:10]))
            .timetuple()
            .tm_yday
            for d in dates.astype(str)
        ]
    )
    return p90[np.clip(doy - 1, 0, 364)]


def events_from_mask(mask: np.ndarray, dates: np.ndarray):
    """Return list of (start_idx, end_idx_exclusive, duration, year_of_start)."""
    events = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            year = dates[i].astype("datetime64[Y]").astype(int) + 1970
            events.append((i, j, j - i, int(year)))
            i = j
        else:
            i += 1
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument(
        "--vote_threshold",
        type=float,
        default=0.5,
        help="Fraction of MC samples that must flag a day as MHW "
        "for it to count in the majority-vote series.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    dates, trues, preds, stds = load_all_folds()
    p90_doy = load_ns_p90()
    thr = doy_threshold(dates, p90_doy)

    mhw_truth = apply_hobday(trues > thr)
    truth_events = events_from_mask(mhw_truth, dates)

    # OLD approach: Hobday on the point-estimate mean directly
    mhw_pred_mean = apply_hobday(preds > thr)
    old_events = events_from_mask(mhw_pred_mean, dates)

    # NEW approach: Monte Carlo — sample per-day trajectories, apply literal
    # Hobday to each sample, then aggregate the per-day flag fraction.
    n = len(dates)
    flag_count = np.zeros(n, dtype=np.int64)
    for s in range(args.n_samples):
        sample = rng.normal(loc=preds, scale=stds)
        mhw_sample = apply_hobday(sample > thr)
        flag_count += mhw_sample

    flag_frac = flag_count / args.n_samples
    mhw_pred_ensemble = flag_frac >= args.vote_threshold
    ens_events = events_from_mask(mhw_pred_ensemble, dates)

    def summarize(name, events, mask):
        total_days = sum(e[2] for e in events)
        mean_dur = total_days / len(events) if events else 0.0
        print(f"=== {name} ===")
        print(
            f"  {len(events)} events, total {total_days} days, mean duration {mean_dur:.1f} days"
        )

    summarize("truth", truth_events, mhw_truth)
    summarize("gnll OLD (mean > threshold)", old_events, mhw_pred_mean)
    summarize(
        f"gnll NEW (MC ensemble, {args.n_samples} samples, vote>={args.vote_threshold})",
        ens_events,
        mhw_pred_ensemble,
    )

    # Year-by-year table: truth vs old vs new
    years = sorted(
        set(e[3] for e in truth_events)
        | set(e[3] for e in old_events)
        | set(e[3] for e in ens_events)
    )
    truth_by_year = {}
    old_by_year = {}
    new_by_year = {}
    for e in truth_events:
        truth_by_year[e[3]] = truth_by_year.get(e[3], 0) + 1
    for e in old_events:
        old_by_year[e[3]] = old_by_year.get(e[3], 0) + 1
    for e in ens_events:
        new_by_year[e[3]] = new_by_year.get(e[3], 0) + 1

    print(f"\n  {'Year':6}{'Truth':>7}{'OLD':>7}{'NEW':>7}")
    for y in years:
        print(
            f"  {y:<6}{truth_by_year.get(y,0):>7}{old_by_year.get(y,0):>7}{new_by_year.get(y,0):>7}"
        )

    # Event-overlap recall: fraction of truth events with >=1 overlapping day in pred
    def recall(events_ref, mask_pred):
        hit = sum(1 for (i, j, *_r) in events_ref if mask_pred[i:j].any())
        return hit / len(events_ref) if events_ref else 0.0

    print("\n  Event-overlap recall (truth events with >=1 overlapping predicted day):")
    print(f"    OLD: {recall(truth_events, mhw_pred_mean):.3f}")
    print(f"    NEW: {recall(truth_events, mhw_pred_ensemble):.3f}")

    np.savez(
        EXPERIMENTS_DIR.parent / "figures" / "gnll_ensemble_hobday.npz",
        dates=dates,
        flag_frac=flag_frac,
        mhw_truth=mhw_truth,
        mhw_pred_mean=mhw_pred_mean,
        mhw_pred_ensemble=mhw_pred_ensemble,
    )
    print("\nSaved per-day arrays to figures/gnll_ensemble_hobday.npz")


if __name__ == "__main__":
    main()
