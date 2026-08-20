"""
mhw_definition_agreement_and_recall.py — Step 2 (items 3-4) of the plan:
confusion matrix between the two MHW ground-truth definitions (basin-mean
Hobday vs. per-pixel+area_frac>=0.05, threshold chosen with the user), and
recall of full_gnll_quantile/full_gnll_focal recomputed under BOTH
definitions.

Threshold history: originally 0.405 (top10%/p90 of area_frac's own
distribution, chosen Aug 20 2026), replaced same day with 0.05 (MedECC
2023's default areal-extent convention) after checking what each
candidate threshold implies in days/year and finding 0.05 reproduces the
independently-sourced North Sea literature benchmark (~140 MHW days/year,
Ocean Science 2025) almost exactly (135.1/year), while 0.405 landed in
the NOAA-Blobtracker "extreme basin-wide events only" tier (36.5/year).
See docs/narrative.md for the full reasoning and docs/known_issues.md #41.

Reuses: src/utils/hobday.py (apply_hobday, load_ns_p90), the inference
logic from _adhoc_eval_extreme_recall.py, and the area_frac(t) series
already computed and saved by calibrate_mhw_area_threshold.py — does not
recompute per-pixel Hobday classification a second time.

Per-fold (trues, preds, thresh1, area_frac_test) arrays are cached to
experiments/figures/_fold_cache/ so that changing AREA_FRAC_THRESHOLD
again does not require re-running CPU inference for all 10 folds.

CPU only. Not part of the permanent pipeline. Run with `python3 -u` so
progress prints aren't held back by stdout buffering when redirected to
a log file.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path("/raven/u/cradin/exprecursors")
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.hobday import apply_hobday, load_ns_p90  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402

EXPERIMENTS_DIR = (
    REPO_ROOT / "experiments" / "partition" / "old_v1"
)  # v1 checkpoints only, moved Aug 20 2026 -- see known_issues.md #45
FIGURES_DIR = REPO_ROOT / "experiments" / "figures"
FOLD_CACHE_DIR = FIGURES_DIR / "_fold_cache"
FOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
AREA_FRAC_THRESHOLD = 0.05  # MedECC 2023 default areal-extent threshold, chosen with the user Aug 20 2026 (see docs/narrative.md)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)
p90 = load_ns_p90()  # (365,) physical units, basin-mean definition

# ---------------------------------------------------------------------------
# Part A: confusion matrix between the two ground-truth definitions
# ---------------------------------------------------------------------------

ds = xr.open_dataset(DATA_FILE)
raw_years = ds.time.dt.year.values
raw_doys = ds.time.dt.dayofyear.values
raw_doys_clamped = raw_doys.copy()
raw_doys_clamped[raw_doys_clamped >= 365] = 365
raw_target = ds["target"].values

thresh_per_day = p90[raw_doys_clamped - 1]
def1_mhw_day = apply_hobday(raw_target > thresh_per_day)  # basin-mean definition

area_frac = np.load(FIGURES_DIR / "area_frac_timeseries.npy")
assert len(area_frac) == len(
    def1_mhw_day
), "area_frac length mismatch vs raw daily series"
def2_mhw_day = area_frac >= AREA_FRAC_THRESHOLD  # per-pixel+area definition

both = int((def1_mhw_day & def2_mhw_day).sum())
only1 = int((def1_mhw_day & ~def2_mhw_day).sum())
only2 = int((~def1_mhw_day & def2_mhw_day).sum())
neither = int((~def1_mhw_day & ~def2_mhw_day).sum())
n_total = len(def1_mhw_day)

print(
    "=== Confusion matrix: def1 (basin-mean Hobday) vs def2 (per-pixel+area>=%.3f) ==="
    % AREA_FRAC_THRESHOLD
)
print(f"  both extreme:        {both:5d} ({both/n_total*100:.1f}%)")
print(f"  only def1 (basin):   {only1:5d} ({only1/n_total*100:.1f}%)")
print(f"  only def2 (pixel):   {only2:5d} ({only2/n_total*100:.1f}%)")
print(f"  neither:             {neither:5d} ({neither/n_total*100:.1f}%)")
print(f"  def1 total extreme days: {int(def1_mhw_day.sum())}")
print(f"  def2 total extreme days: {int(def2_mhw_day.sum())}")
agreement = (both + neither) / n_total
jaccard = both / (both + only1 + only2) if (both + only1 + only2) > 0 else float("nan")
print(
    f"  simple agreement: {agreement*100:.1f}%   Jaccard (both / union of extremes): {jaccard*100:.1f}%"
)

fig, ax = plt.subplots(figsize=(5, 5))
mat = np.array([[both, only1], [only2, neither]])
im = ax.imshow(mat, cmap="Blues")
for (i, j), v in np.ndenumerate(mat):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=14)
ax.set_xticks([0, 1])
ax.set_xticklabels(["def2 extreme", "def2 not extreme"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["def1 extreme", "def1 not extreme"])
ax.set_title(
    f"Agreement: def1 (basin-mean) vs def2 (pixel+area>={AREA_FRAC_THRESHOLD})\nJaccard={jaccard*100:.1f}%"
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mhw_definition_agreement.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIGURES_DIR / 'mhw_definition_agreement.png'}")

# ---------------------------------------------------------------------------
# Part B: recall recomputed under BOTH definitions for quantile + focal
# ---------------------------------------------------------------------------


def run_fold(config_subdir, run_name, fold):
    cache_path = FOLD_CACHE_DIR / f"{run_name}_fold{fold}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        print(f"  fold {fold}: loaded from cache {cache_path.name}", flush=True)
        return d["trues_c"], d["preds_c"], d["thresh1"], d["area_frac_test"]

    cfg_path = REPO_ROOT / "configs" / "partition" / config_subdir / f"fold{fold}.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    run_dir = EXPERIMENTS_DIR / f"{run_name}_fold{fold}"

    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    extra = {}
    if cfg.get("focal_weight", False):
        extra["p90_by_doy"] = torch.tensor(p90, dtype=torch.float32)
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=True, map_location=device, **extra
    )
    lm.eval().to(device)

    dm = LazyDataModule(str(cfg_path))
    dm.setup()
    full_ds = dm.test_dataset.dataset
    test_indices = dm.test_dataset.indices
    test_dl = DataLoader(dm.test_dataset, batch_size=32, shuffle=False, num_workers=0)

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_dl:
            xs, xt, y = batch[0], batch[1], batch[2]
            xs, xt = xs.float().to(device), xt.float().to(device)
            p = lm.model(xs, xt)
            preds.append((p[:, 0] if p.ndim == 2 else p.squeeze(-1)).cpu())
            trues.append(y.squeeze(-1).cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    preds_c = preds * lm.target_std + lm.target_mean
    trues_c = trues * lm.target_std + lm.target_mean

    target_idx = np.array(
        [i + full_ds.window_size - 1 + full_ds.lead_time for i in test_indices]
    )
    doys = full_ds.doys[target_idx]
    doys = np.where(doys >= 365, 365, doys)
    thresh1 = p90[doys - 1]  # def1 threshold, per test sample
    area_frac_test = area_frac[target_idx]  # def2 ground truth, per test sample

    np.savez(
        cache_path,
        trues_c=trues_c,
        preds_c=preds_c,
        thresh1=thresh1,
        area_frac_test=area_frac_test,
    )
    print(f"  fold {fold}: inference done, cached to {cache_path.name}", flush=True)
    return trues_c, preds_c, thresh1, area_frac_test


def analyze(config_subdir, run_name, label):
    all_trues, all_preds, all_thresh1, all_area_frac = [], [], [], []
    print(f"\n=== {label} ===")
    for fold in range(5):
        trues_c, preds_c, thresh1, area_frac_test = run_fold(
            config_subdir, run_name, fold
        )
        all_trues.append(trues_c)
        all_preds.append(preds_c)
        all_thresh1.append(thresh1)
        all_area_frac.append(area_frac_test)

    trues_c = np.concatenate(all_trues)
    preds_c = np.concatenate(all_preds)
    thresh1 = np.concatenate(all_thresh1)
    area_frac_c = np.concatenate(all_area_frac)

    # def1: basin-mean truth exceeds basin-mean p90
    ext1 = trues_c > thresh1
    n1 = int(ext1.sum())
    recall1 = (preds_c[ext1] > thresh1[ext1]).mean() if n1 else float("nan")

    # def2: per-pixel+area regional MHW day (ground truth independent of
    # the model's own target scale) -- recall = did the model's own
    # basin-mean-exceedance signal (pred > basin-mean p90) fire on days
    # that were REALLY regional MHW events per the rigorous definition.
    ext2 = area_frac_c >= AREA_FRAC_THRESHOLD
    n2 = int(ext2.sum())
    recall2 = (preds_c[ext2] > thresh1[ext2]).mean() if n2 else float("nan")

    print(
        f"  def1 (basin-mean):        n_extreme={n1} ({n1/len(trues_c)*100:.1f}%)  recall={recall1*100:.1f}%"
    )
    print(
        f"  def2 (pixel+area>={AREA_FRAC_THRESHOLD}): n_extreme={n2} ({n2/len(trues_c)*100:.1f}%)  recall={recall2*100:.1f}%"
    )
    return dict(recall_def1=recall1, n_def1=n1, recall_def2=recall2, n_def2=n2)


results = {}
results["quantile"] = analyze(
    "full_gnll_quantile", "TbotAtm_full_gnll_quantile_seed42", "full_gnll_quantile"
)
results["focal"] = analyze(
    "full_gnll_focal", "TbotAtm_full_gnll_focal_seed42", "full_gnll_focal"
)

print("\n=== SUMMARY: recall under both ground-truth definitions ===")
for k, v in results.items():
    print(
        f"{k:10s}: def1(basin-mean) recall={v['recall_def1']*100:.1f}% (n={v['n_def1']})  "
        f"def2(pixel+area) recall={v['recall_def2']*100:.1f}% (n={v['n_def2']})"
    )

fig, ax = plt.subplots(figsize=(7, 5))
labels = list(results.keys())
x = np.arange(len(labels))
w = 0.35
ax.bar(
    x - w / 2,
    [results[k]["recall_def1"] * 100 for k in labels],
    w,
    label="def1 (basin-mean)",
)
ax.bar(
    x + w / 2,
    [results[k]["recall_def2"] * 100 for k in labels],
    w,
    label="def2 (pixel+area)",
)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Recall (%)")
ax.set_title("Extreme-day recall under both ground-truth definitions")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "recall_both_definitions.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIGURES_DIR / 'recall_both_definitions.png'}")
