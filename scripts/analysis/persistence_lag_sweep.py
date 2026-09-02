"""
persistence_lag_sweep.py — item 2 from the user's Aug 21 2026 idea list:
"barrido de lead time (1-14d): la persistencia decae con el lead;
localizar el crossover donde el modelo gana". No retraining -- persistence
needs no model at all, just the target series itself.

NOTE (documented, not hidden): this only sweeps PERSISTENCE across lags.
Without models actually trained at those other lead times, there is no
"crossover where the model wins" curve to compare against -- only where
persistence ALONE decays. Shown against the current lead=7 model's own
r (already measured) as a single reference point, not a full sweep of
both. A full model-side sweep needs actual retraining (deferred by the
user, "mas adelante").

Uses `target` directly (NS-box-mean to_anom, already computed upstream
-- see docs/data.md), full 40-year series (not test-only, since this is
model-independent).

CPU only. Not part of the permanent pipeline.

Usage:
  python scripts/analysis/persistence_lag_sweep.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import DATA_FILE  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments" / "figures" / "step7_persistence"
# Extended Aug 22 2026: item 3's lead-time sweep now has actual trained
# models at lead={3,5,7,14,30}d (see docs/narrative.md, jobs 29457750/
# 53/54/55/56) -- added lag=30 to match, so persistence and model curves
# share the same x-axis range for the final combined plot
# (lead_time_sweep_model_vs_persistence.py).
LAGS = list(range(1, 15)) + [30]

# Reference points already measured this session, for context on the plot
# (not re-derived here -- see docs/narrative.md's Aug 21 2026 entries).
MODEL_LEAD7_R_MEAN = 0.850  # full_gnll_quantile_v2, mean head, pooled 5-fold
MODEL_LEAD7_R_QUANTILE = (
    0.838  # quantile head (r vs raw target, not the detection metric)
)

ds = xr.open_dataset(DATA_FILE)
target = ds["target"].values.astype(np.float64)
ds.close()
target = target[~np.isnan(target)]
print(f"Target series: n={len(target)}", flush=True)

rs = []
for lag in LAGS:
    y_true = target[lag:]
    y_pred = target[:-lag]
    r, _ = pearsonr(y_true, y_pred)
    rs.append(r)
    print(f"  lag={lag:2d}d  r_persist={r:.4f}", flush=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(LAGS, rs, "o-", color="#e08214", label="Persistence (lag-N)")
ax.axhline(
    MODEL_LEAD7_R_MEAN,
    color="#2166ac",
    ls="--",
    label=f"Model @ lead=7 (mean head, r={MODEL_LEAD7_R_MEAN})",
)
ax.axvline(7, color="gray", ls=":", lw=1)
ax.set_xlabel("Lag / lead time (days)")
ax.set_ylabel("Pearson r")
ax.set_title(
    "Persistence decay vs. lag (no retraining -- model shown as single reference point at lead=7)"
)
ax.set_xticks(LAGS)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
out_path = OUT_DIR / "persistence_lag_sweep.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}", flush=True)

np.savez(OUT_DIR / "persistence_lag_sweep.npz", lags=np.array(LAGS), r=np.array(rs))
print(f"Saved {OUT_DIR / 'persistence_lag_sweep.npz'}", flush=True)
