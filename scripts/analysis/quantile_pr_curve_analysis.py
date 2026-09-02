"""
quantile_pr_curve_analysis.py — Aug 22 2026, user's idea: the model-vs-
persistence recall/precision comparison so far uses a SINGLE operating
point (whatever threshold q_pred/mean_pred happens to cross thresh1
at) -- the "recall/precision TRADE, not a strict win" framing depends
entirely on that one point. Sweeps the decision threshold over the
model's continuous score (q_pred or mean_pred) to get the full
precision-recall curve + AUPRC per family, and overlays persistence's
existing single point for comparison. If the model's curve dominates
persistence's point at some point on the curve (higher precision at
persistence's recall, or vice versa), that changes the "trade" framing
to a real win in that operating region.

Pure post-processing of scripts/eval_recall_v2_partition.py's saved
npz (trues_c, mean_c, q_c, thresh1, area_frac_c) and
scripts/analysis/persistence_recall_baseline.py's saved npz -- no
retraining, no GPU, no new model inference at all.

Usage:
  python scripts/analysis/quantile_pr_curve_analysis.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "experiments" / "figures"
OUT_DIR = FIGURES_DIR / "step7_persistence"

FAMILY_LEAD = {
    "full_lead7": 7,
    "local": 7,
    "remote": 7,
    "lead3": 3,
    "lead5": 5,
    "lead14": 14,
    "lead30": 30,
}


def main():
    persist = np.load(
        FIGURES_DIR / "_fold_cache" / "persistence_recall_baseline.npz",
        allow_pickle=True,
    )
    persist_by_lead = {r["lead"]: r for r in persist["results"]}

    results = {}
    for label, lead in FAMILY_LEAD.items():
        npz_path = FIGURES_DIR / "_fold_cache" / f"eval_recall_v2_{label}.npz"
        if not npz_path.exists():
            print(f"  {label}: {npz_path} not found, skipping", flush=True)
            continue
        d = np.load(npz_path)
        trues_c, mean_c, q_c, thresh1, area_frac_c = (
            d["trues_c"],
            d["mean_c"],
            d["q_c"],
            d["thresh1"],
            d["area_frac_c"],
        )
        ext1 = trues_c > thresh1
        ext2 = area_frac_c >= 0.05

        fam_result = {"lead": lead}
        for head_name, score in [("mean", mean_c), ("quantile", q_c)]:
            for def_name, ext in [("def1", ext1), ("def2", ext2)]:
                # score itself (raw physical prediction) as the continuous
                # classifier score -- sklearn sweeps every unique value as
                # a threshold candidate, giving the exact PR curve.
                prec, rec, thr = precision_recall_curve(ext.astype(int), score)
                auprc = average_precision_score(ext.astype(int), score)
                fam_result[f"{head_name}_{def_name}_precision"] = prec
                fam_result[f"{head_name}_{def_name}_recall"] = rec
                fam_result[f"{head_name}_{def_name}_auprc"] = auprc
        results[label] = fam_result

        p = persist_by_lead[lead]
        print(
            f"{label} (lead={lead}d): "
            f"AUPRC def1 mean={fam_result['mean_def1_auprc']:.3f} quant={fam_result['quantile_def1_auprc']:.3f} "
            f"| AUPRC def2 mean={fam_result['mean_def2_auprc']:.3f} quant={fam_result['quantile_def2_auprc']:.3f} "
            f"| persistence single point: def1 R={p['def1_recall']:.3f}/P={p['def1_precision']:.3f}  "
            f"def2 R={p['def2_recall']:.3f}/P={p['def2_precision']:.3f}",
            flush=True,
        )

    # Plot: one panel per lead, def1 only (def2's base rate is much
    # higher so the curve shape differs -- kept in the saved npz/print
    # above for def2 too, just not plotted to keep the figure readable).
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for i, (label, lead) in enumerate(FAMILY_LEAD.items()):
        if label not in results:
            continue
        ax = axes[i]
        r = results[label]
        ax.plot(
            r["mean_def1_recall"],
            r["mean_def1_precision"],
            label=f"mean head (AUPRC={r['mean_def1_auprc']:.3f})",
            color="#2166ac",
        )
        ax.plot(
            r["quantile_def1_recall"],
            r["quantile_def1_precision"],
            label=f"quantile head (AUPRC={r['quantile_def1_auprc']:.3f})",
            color="#4dac26",
        )
        p = persist_by_lead[lead]
        ax.scatter(
            [p["def1_recall"]],
            [p["def1_precision"]],
            color="#e08214",
            s=80,
            zorder=5,
            label="persistence (single point)",
        )
        ax.set_title(f"{label} (lead={lead}d)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(FAMILY_LEAD), len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "Model PR curve (def1, basin-mean) vs. persistence's single operating point"
    )
    plt.tight_layout()
    out_path = OUT_DIR / "quantile_pr_curve_vs_persistence.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}", flush=True)

    np.savez(
        OUT_DIR / "quantile_pr_curve_vs_persistence.npz",
        **{f"{label}_{k}": v for label, r in results.items() for k, v in r.items()},
    )
    print(f"Saved {OUT_DIR / 'quantile_pr_curve_vs_persistence.npz'}", flush=True)

    print("\n=== Does the model's curve ever dominate persistence's point? ===")
    print(
        "(dominate = a point on the model's curve with recall>=persist_recall AND precision>=persist_precision)"
    )
    for label, lead in FAMILY_LEAD.items():
        if label not in results:
            continue
        r = results[label]
        p = persist_by_lead[lead]
        for head_name in ["mean", "quantile"]:
            rec = r[f"{head_name}_def1_recall"]
            prec = r[f"{head_name}_def1_precision"]
            dominates = np.any(
                (rec >= p["def1_recall"]) & (prec >= p["def1_precision"])
            )
            print(
                f"  {label} / {head_name} head (def1): dominates persistence? {dominates}"
            )


if __name__ == "__main__":
    main()
