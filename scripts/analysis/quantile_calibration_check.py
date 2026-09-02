"""
quantile_calibration_check.py — Aug 22 2026, user's idea #3: coverage
check for the quantile head (tau=0.9) -- the one diagnostic persistence
cannot have by construction (it makes no probabilistic claim at all).
If q_pred is well-calibrated for tau=0.9, true values should fall BELOW
q_pred on ~90% of days (empirical coverage ~= tau).

Pure post-processing of scripts/eval_recall_v2_partition.py's already-
saved npz (trues_c, q_c) -- no model reload, no GPU, no new SLURM job.

Usage:
  python scripts/analysis/quantile_calibration_check.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "experiments" / "figures"
TAU = 0.9

FAMILIES = ["full_lead7", "local", "remote", "lead3", "lead5", "lead14", "lead30"]


def main():
    print(f"=== Quantile head (tau={TAU}) coverage check, all families ===", flush=True)
    print(
        f"Target empirical coverage: {TAU*100:.0f}% (fraction of days with true <= q_pred)\n",
        flush=True,
    )

    results = {}
    for label in FAMILIES:
        npz_path = FIGURES_DIR / "_fold_cache" / f"eval_recall_v2_{label}.npz"
        if not npz_path.exists():
            print(f"  {label}: {npz_path} not found yet, skipping", flush=True)
            continue
        d = np.load(npz_path)
        trues_c, q_c = d["trues_c"], d["q_c"]
        coverage = float((trues_c <= q_c).mean())
        # PIT-lite: also report coverage split by whether the day was itself
        # extreme (above thresh1) -- calibration commonly degrades exactly
        # on the days that matter most.
        thresh1 = d["thresh1"]
        extreme = trues_c > thresh1
        cov_extreme = (
            float((trues_c[extreme] <= q_c[extreme]).mean())
            if extreme.sum()
            else float("nan")
        )
        cov_normal = (
            float((trues_c[~extreme] <= q_c[~extreme]).mean())
            if (~extreme).sum()
            else float("nan")
        )
        results[label] = {
            "coverage": coverage,
            "cov_extreme": cov_extreme,
            "cov_normal": cov_normal,
            "n": len(trues_c),
            "n_extreme": int(extreme.sum()),
        }
        print(
            f"  {label:10s}  overall={coverage*100:5.1f}%  "
            f"on-extreme-days={cov_extreme*100:5.1f}% (n={int(extreme.sum())})  "
            f"on-normal-days={cov_normal*100:5.1f}%  "
            f"gap-vs-target={  (coverage-TAU)*100:+.1f}pp",
            flush=True,
        )

    out_path = FIGURES_DIR / "_fold_cache" / "quantile_calibration_check.npz"
    np.savez(
        out_path, **{f"{k}_{m}": v for k, d in results.items() for m, v in d.items()}
    )
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
