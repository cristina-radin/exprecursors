"""
persistence_recall_baseline.py — Aug 22 2026, user's idea: the def1/def2
recall/precision/FPR table (eval_recall_v2_partition.py) is inconclusive
on its own -- "48% recall" needs a baseline to know if it's good. Adds
the persistence flag: predict day t is extreme if day (t-lead_time) was
already extreme (target(t-lead) > p90(doy(t-lead))). No model, no
checkpoint, no GPU -- pure target-series + config metadata.

One persistence baseline per DISTINCT lead_time value {3,5,7,14,30} --
local/remote/full_lead7 all share lead=7 and identical stratified_kfold
splits (masking only changes the INPUT, never the target/test-split),
so persistence is identical across those three and only needs computing
once per lead.

CPU only, no GPU, no SLURM needed (same cost class as
persistence_lag_sweep.py).

Usage:
  python scripts/analysis/persistence_recall_baseline.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.utils.hobday import load_ns_p90  # noqa: E402

FIGURES_DIR = REPO_ROOT / "experiments" / "figures"
AREA_FRAC_THRESHOLD = 0.05

# one representative config per distinct lead_time (fold0 is enough --
# persistence needs only the target series + split, identical across
# masking modes at the same lead)
LEAD_CONFIGS = {
    3: "lead3_landfill",
    5: "lead5_landfill",
    7: "full_gnll_quantile_v2_landfill",
    14: "lead14_landfill",
    30: "lead30_landfill",
}


def run_lead(lead, config_dir, p90, area_frac):
    results = {"trues": [], "persist_pred": [], "thresh1": [], "area_frac": []}
    for fold in range(5):
        cfg_path = REPO_ROOT / "configs" / "partition" / config_dir / f"fold{fold}.yaml"
        dm = LazyDataModule(str(cfg_path))
        dm.setup()
        full_ds = dm.test_dataset.dataset
        test_indices = dm.test_dataset.indices

        target_idx = np.array(
            [i + full_ds.window_size - 1 + full_ds.lead_time for i in test_indices]
        )
        persist_idx = (
            target_idx - full_ds.lead_time
        )  # value LEAD days before target day

        trues = full_ds.target[target_idx].numpy()
        persist_pred = full_ds.target[persist_idx].numpy()
        doys = full_ds.doys[target_idx]
        doys = np.where(doys >= 365, 365, doys)
        thresh1 = p90[doys - 1]
        area_frac_test = area_frac[target_idx]

        results["trues"].append(trues)
        results["persist_pred"].append(persist_pred)
        results["thresh1"].append(thresh1)
        results["area_frac"].append(area_frac_test)
        print(f"  lead={lead}d fold={fold}: n={len(trues)}", flush=True)

    trues = np.concatenate(results["trues"])
    persist_pred = np.concatenate(results["persist_pred"])
    thresh1 = np.concatenate(results["thresh1"])
    area_frac_c = np.concatenate(results["area_frac"])

    ext1 = trues > thresh1
    ext2 = area_frac_c >= AREA_FRAC_THRESHOLD
    flagged = (
        persist_pred > thresh1
    )  # persistence flags tomorrow extreme if it's extreme today

    def recall_precision_fpr(ext):
        n = int(ext.sum())
        recall = (flagged[ext]).mean() if n else float("nan")
        tp = int((flagged & ext).sum())
        fp = int((flagged & ~ext).sum())
        precision = tp / max(1, int(flagged.sum()))
        fpr = fp / max(1, int((~ext).sum()))
        return recall, precision, fpr, n

    r1 = recall_precision_fpr(ext1)
    r2 = recall_precision_fpr(ext2)
    r_persist = np.corrcoef(trues, persist_pred)[0, 1]

    print(
        f"lead={lead:2d}d  r_persist={r_persist:.4f}  "
        f"def1: recall={r1[0]*100:.1f}% precision={r1[1]*100:.1f}% FPR={r1[2]*100:.1f}% (n={r1[3]})  "
        f"def2: recall={r2[0]*100:.1f}% precision={r2[1]*100:.1f}% FPR={r2[2]*100:.1f}% (n={r2[3]})",
        flush=True,
    )
    return {
        "lead": lead,
        "r_persist": r_persist,
        "def1_recall": r1[0],
        "def1_precision": r1[1],
        "def1_fpr": r1[2],
        "def1_n": r1[3],
        "def2_recall": r2[0],
        "def2_precision": r2[1],
        "def2_fpr": r2[2],
        "def2_n": r2[3],
    }


def main():
    p90 = load_ns_p90()
    area_frac = np.load(FIGURES_DIR / "area_frac_timeseries.npy")

    all_results = []
    for lead, config_dir in LEAD_CONFIGS.items():
        res = run_lead(lead, config_dir, p90, area_frac)
        all_results.append(res)
        np.savez(
            FIGURES_DIR / "_fold_cache" / "persistence_recall_baseline_partial.npz",
            results=all_results,
        )

    print("\n=== SUMMARY: persistence recall/precision baseline, all leads ===")
    for res in all_results:
        print(
            f"  lead={res['lead']:2d}d  r={res['r_persist']:.4f}  "
            f"def1 recall={res['def1_recall']*100:.1f}%  def2 recall={res['def2_recall']*100:.1f}%"
        )

    out_path = FIGURES_DIR / "_fold_cache" / "persistence_recall_baseline.npz"
    np.savez(out_path, results=all_results)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
