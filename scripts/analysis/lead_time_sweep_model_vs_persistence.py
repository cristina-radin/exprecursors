"""
lead_time_sweep_model_vs_persistence.py — item 3's actual deliverable
(Aug 22 2026): "una grafica de sweep de lead time, modelo vs
persistencia". Combines:
  - Model pooled r (mean + quantile head) at lead={3,5,7,14,30}d, from
    scripts/eval_recall_v2_partition.py's saved npz per family (job
    29463377).
  - Persistence r at lag={1..14,30}d, from
    scripts/analysis/persistence_lag_sweep.py (already extended to
    lag=30 same day).

Both curves cover the SAME 40-year record: the model's pooled r is
computed across all 5 stratified_kfold test folds, which partition all
40 years exactly once (each year evaluated by whichever fold didn't
train on it) -- so it's a fair, not-cherry-picked comparison against
persistence's own full-40-year sweep, not an apples-to-oranges mismatch.

CPU only, no GPU, reads already-saved npz files.

Usage:
  python scripts/analysis/lead_time_sweep_model_vs_persistence.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "experiments" / "figures"
PERSIST_NPZ = FIGURES_DIR / "step7_persistence" / "persistence_lag_sweep.npz"
OUT_DIR = FIGURES_DIR / "step7_persistence"

MODEL_LEADS = [3, 5, 7, 14, 30]
MODEL_LABELS = {3: "lead3", 5: "lead5", 7: "full_lead7", 14: "lead14", 30: "lead30"}


def main():
    persist = np.load(PERSIST_NPZ)
    lags = persist["lags"]
    r_persist = persist["r"]

    model_r_mean, model_r_q = [], []
    for lead in MODEL_LEADS:
        npz_path = (
            FIGURES_DIR / "_fold_cache" / f"eval_recall_v2_{MODEL_LABELS[lead]}.npz"
        )
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} not found -- run eval_recall_v2_partition.py for "
                f"{MODEL_LABELS[lead]} first (job 29463377)."
            )
        d = np.load(npz_path)
        model_r_mean.append(float(d["r_mean_pooled"]))
        model_r_q.append(float(d["r_q_pooled"]))
        print(
            f"  lead={lead:2d}d  model r_mean={model_r_mean[-1]:.4f}  r_quantile={model_r_q[-1]:.4f}",
            flush=True,
        )

    # Two panels: raw r curves, and the gap (persist_r - model_r) so the
    # absence of a crossover is visually explicit, not something the
    # reader has to infer by eyeballing two overlapping lines (user's
    # request, Aug 22 2026).
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax.plot(
        lags,
        r_persist,
        "o-",
        color="#e08214",
        label="Persistence (lag-N, no model, zero free parameters)",
    )
    ax.plot(MODEL_LEADS, model_r_mean, "s-", color="#2166ac", label="Model, mean head")
    ax.plot(MODEL_LEADS, model_r_q, "^-", color="#4dac26", label="Model, quantile head")
    ax.set_ylabel("Pearson r (pooled, full 40yr test coverage)")
    ax.set_title("Lead-time sweep: model vs. persistence (land_fill_mode=nearest)")
    ax.legend()
    ax.grid(alpha=0.3)

    persist_at_model_leads = np.array(
        [r_persist[np.where(lags == lead)[0][0]] for lead in MODEL_LEADS]
    )
    gap_mean = persist_at_model_leads - np.array(model_r_mean)
    gap_q = persist_at_model_leads - np.array(model_r_q)
    ax2.axhline(0, color="black", lw=1)
    ax2.plot(
        MODEL_LEADS,
        gap_mean,
        "s-",
        color="#2166ac",
        label="persist_r - model_r (mean head)",
    )
    ax2.plot(
        MODEL_LEADS,
        gap_q,
        "^-",
        color="#4dac26",
        label="persist_r - model_r (quantile head)",
    )
    ax2.set_xlabel("Lead / lag time (days)")
    ax2.set_ylabel("Gap (persist_r - model_r)")
    ax2.set_title("Gap: positive = persistence still ahead (no crossover)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax.set_xticks(sorted(set(list(lags) + MODEL_LEADS)))

    plt.tight_layout()
    out_path = OUT_DIR / "lead_time_sweep_model_vs_persistence.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}", flush=True)
    print("\n=== Gap (persist_r - model_r) at each model lead ===")
    for lead, gm, gq in zip(MODEL_LEADS, gap_mean, gap_q):
        print(f"  lead={lead:2d}d  gap_mean={gm:+.4f}  gap_quantile={gq:+.4f}")

    np.savez(
        OUT_DIR / "lead_time_sweep_model_vs_persistence.npz",
        model_leads=np.array(MODEL_LEADS),
        model_r_mean=np.array(model_r_mean),
        model_r_quantile=np.array(model_r_q),
        persist_lags=lags,
        persist_r=r_persist,
        gap_mean=gap_mean,
        gap_quantile=gap_q,
    )
    print(f"Saved {OUT_DIR / 'lead_time_sweep_model_vs_persistence.npz'}", flush=True)

    print("\n=== Crossover check: does persistence ever fall below the model? ===")
    for lead in MODEL_LEADS:
        idx = np.where(lags == lead)[0]
        if len(idx) == 0:
            continue
        rp = r_persist[idx[0]]
        rm = model_r_mean[MODEL_LEADS.index(lead)]
        rq = model_r_q[MODEL_LEADS.index(lead)]
        winner = (
            "persistence"
            if rp > max(rm, rq)
            else ("model_mean" if rm > rq else "model_quantile")
        )
        print(
            f"  lead={lead:2d}d  persist={rp:.4f}  model_mean={rm:.4f}  model_quantile={rq:.4f}  -> {winner}"
        )


if __name__ == "__main__":
    main()
