"""
eval_onset_skill_quantile_v2.py — re-verify onset skill on the committed
full_gnll_quantile_v2 model (stratified_kfold), Aug 21 2026.

The existing negative result in docs/narrative.md ("Onset skill", r=0.006
to 0.093, not significant) was computed with the buggy `kfold` split and
an older model family (full/remote_only/local_only, plain MSE/no
quantile head) -- flagged as needing re-verification under
stratified_kfold since the plan's Paso 7 item 2, not yet done. The user
pushed back on accepting the negative result at face value ("si vamos a
detectar drivers, TIENE QUE PREDECIR EL INICIO") -- this re-runs it
properly rather than assuming the old number still holds.

Computes skill (Pearson r, 95% CI via Fisher z) separately for:
  - onset days (first day of a Hobday MHW event) -- the real question
  - mid_event days
  - no_mhw days
  - all days
for BOTH the mean head and the quantile head (q_pred), against lag-7
persistence, pooled across all 5 folds (not per-fold-averaged -- the
already-documented reason: an unweighted fold average lets a low-n fold
dominate, see narrative.md's "Onset skill" methodology note).

Reuses: src/utils/hobday.py (apply_hobday, mhw_phase_labels, load_ns_p90)
-- does not reimplement this a third time. LazyDataModule + best_ckpt()
+ load_model_config() + CNNLightningModule.load_from_checkpoint() (same
pattern as scripts/analysis/quantile_head_recall_v2_all5.py).

CPU only. Not part of the permanent pipeline.

Usage:
  python scripts/eval_onset_skill_quantile_v2.py
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

import argparse

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.hobday import apply_hobday, load_ns_p90  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--config_dir",
    default="full_gnll_quantile_v2",
    help="dir under configs/partition/ -- Aug 22 2026: generalized so this "
    "can run against full_gnll_quantile_v2_landfill (the current nearest "
    "committed model) without duplicating the script.",
)
_parser.add_argument("--label", default=None, help="defaults to --config_dir")
_args = _parser.parse_args()
CONFIG_DIR = _args.config_dir
LABEL = _args.label or _args.config_dir
# LEAD used to be a hardcoded module constant (=7); reading it from the
# config itself fixes a real bug found Aug 22 2026 (NameError -- removing
# the old `LEAD = 7` line to generalize this script broke `persist[LEAD:]`
# below, missed by a too-narrow grep before this edit) AND makes the
# script correctly reusable for other lead_time families, not just
# lead=7 by coincidence. All folds in one config_dir share the same
# lead_time, so fold0's is representative.
LEAD = yaml.safe_load(
    open(REPO_ROOT / "configs" / "partition" / CONFIG_DIR / "fold0.yaml")
)["lead_time"]

OUT_DIR = REPO_ROOT / "experiments" / "figures" / "step7_persistence"
N_FOLDS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  config_dir={CONFIG_DIR}  label={LABEL}", flush=True)

p90_ns = load_ns_p90()
print(f"p90_ns: min={p90_ns.min():.3f}  max={p90_ns.max():.3f}", flush=True)


def onset_mask(trues, doys, years, p90_ns):
    """True at the first day of each Hobday MHW event (identical logic
    to mhw_phase_labels' onset detection, exposed separately here because
    we also need mid_event/no_mhw masks from the same labels).

    Bug fixed Aug 21 2026 (user's methodological review, known_issues.md
    #53's bug class -- this script never had the fix): stratified_kfold
    assigns NON-CONSECUTIVE years to each fold's test set (e.g. fold0:
    1985, 1991, 2000, ...). Running apply_hobday() on the whole fold's
    concatenated test series at once lets its gap-closure logic bridge
    the tail of one test year into the head of a calendar-distant one --
    sorting samples chronologically (order = np.argsort(target_idx) in
    run_fold) makes them ordered, not calendar-contiguous; the two are
    not the same thing. Fixed by running apply_hobday per calendar year
    (each year's own samples ARE internally contiguous), matching
    eval_onset_skill_curve.py's already-correct pattern."""
    thr = p90_ns[doys - 1]
    exceed = trues > thr
    mhw = np.zeros(len(trues), dtype=bool)
    for yr in np.unique(years):
        ymask = years == yr
        mhw[ymask] = apply_hobday(exceed[ymask])
    onset = np.zeros(len(mhw), dtype=bool)
    for yr in np.unique(years):
        ymask = np.where(years == yr)[0]
        for k, i in enumerate(ymask):
            if mhw[i] and (k == 0 or not mhw[ymask[k - 1]]):
                onset[i] = True
    return onset, mhw


def r_ci(r, n, alpha=0.05):
    if n <= 3 or np.isnan(r):
        return (float("nan"), float("nan"))
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def run_fold(fold):
    cfg_path = REPO_ROOT / "configs" / "partition" / CONFIG_DIR / f"fold{fold}.yaml"
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
    doys = full_ds.doys[target_idx]
    doys = np.where(doys >= 365, 365, doys).astype(int)
    years = full_ds.years[target_idx]

    # NOTE (corrected Aug 21 2026): sorting by target_idx makes these
    # samples chronologically ORDERED, not calendar-CONTIGUOUS --
    # stratified_kfold assigns non-consecutive years to each fold's test
    # set (e.g. fold0: 1985, 1991, 2000, ...), so consecutive array
    # positions can jump straight from Dec 31 of one test year to Jan 1
    # of a calendar-distant one. onset_mask() below now runs
    # apply_hobday() per calendar year to avoid bridging that gap -- see
    # its docstring and known_issues.md #53's bug class. This comment
    # previously (incorrectly) claimed the sort alone made the series
    # contiguous.
    order = np.argsort(target_idx)
    return (
        trues[order],
        mean_preds[order],
        q_preds[order],
        doys[order],
        years[order],
    )


pooled = {
    "onset": {"true": [], "mean": [], "q": [], "persist": []},
    "mid_event": {"true": [], "mean": [], "q": [], "persist": []},
    "no_mhw": {"true": [], "mean": [], "q": [], "persist": []},
    "all": {"true": [], "mean": [], "q": [], "persist": []},
}
n_boundary_nan = 0

for fold in range(N_FOLDS):
    trues, mean_preds, q_preds, doys, years = run_fold(fold)
    onset, mhw = onset_mask(trues, doys, years, p90_ns)
    mid_event = mhw & ~onset
    no_mhw = ~mhw

    persist = np.full_like(trues, np.nan)
    persist[LEAD:] = trues[:-LEAD]
    # invalidate persistence pairs that cross a year boundary too (same
    # non-contiguity issue as onset_mask -- trues[i-LEAD] must be LEAD
    # real days before trues[i], not LEAD positions before in a
    # year-jumping concatenated array). Fixed alongside onset_mask,
    # Aug 21 2026.
    year_of_i = years
    year_of_i_minus_lead = np.concatenate([np.full(LEAD, -999), years[:-LEAD]])
    persist[year_of_i != year_of_i_minus_lead] = np.nan

    n_onset_fold = int(onset.sum())
    boundary_nan = onset & np.isnan(persist)
    if boundary_nan.sum() > 0:
        n_boundary_nan += int(boundary_nan.sum())
        print(
            f"  WARNING fold{fold}: {int(boundary_nan.sum())} onset day(s) in first {LEAD} positions (excluded)",
            flush=True,
        )

    for name, mask in [
        ("onset", onset),
        ("mid_event", mid_event),
        ("no_mhw", no_mhw),
        ("all", np.ones(len(trues), dtype=bool)),
    ]:
        valid = mask & np.isfinite(persist)
        pooled[name]["true"].extend(trues[valid])
        pooled[name]["mean"].extend(mean_preds[valid])
        pooled[name]["q"].extend(q_preds[valid])
        pooled[name]["persist"].extend(persist[valid])

    print(
        f"  fold{fold}: n_onset={n_onset_fold}  n_mid_event={int(mid_event.sum())}  n_no_mhw={int(no_mhw.sum())}",
        flush=True,
    )

print(
    f"\nTotal onset days excluded for boundary NaN persistence: {n_boundary_nan}",
    flush=True,
)

print(f"\n{'='*80}")
print(f"ONSET / MID-EVENT / NO-MHW SKILL, {LABEL} (stratified_kfold, pooled 5 folds)")
print(f"{'='*80}")
results = {}
for phase in ["onset", "mid_event", "no_mhw", "all"]:
    true_a = np.array(pooled[phase]["true"])
    mean_a = np.array(pooled[phase]["mean"])
    q_a = np.array(pooled[phase]["q"])
    pers_a = np.array(pooled[phase]["persist"])
    n = len(true_a)
    if n < 4:
        print(f"  {phase:<10}: n={n} (too few for r)")
        results[phase] = dict(n=n)
        continue
    r_mean, _ = pearsonr(mean_a, true_a)
    r_q, _ = pearsonr(q_a, true_a)
    r_pers, _ = pearsonr(pers_a, true_a)
    ci_mean = r_ci(r_mean, n)
    ci_q = r_ci(r_q, n)
    ci_pers = r_ci(r_pers, n)
    print(
        f"  {phase:<10}: n={n:4d}  r_mean_head={r_mean:+.4f} [{ci_mean[0]:+.3f},{ci_mean[1]:+.3f}]  "
        f"r_quantile_head={r_q:+.4f} [{ci_q[0]:+.3f},{ci_q[1]:+.3f}]  "
        f"r_persist={r_pers:+.4f} [{ci_pers[0]:+.3f},{ci_pers[1]:+.3f}]"
    )
    results[phase] = dict(n=n, r_mean=r_mean, r_q=r_q, r_persist=r_pers)

OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 5))
phases = [p for p in ["onset", "mid_event", "no_mhw", "all"] if "r_mean" in results[p]]
x = np.arange(len(phases))
width = 0.25
ax.bar(
    x - width,
    [results[p]["r_persist"] for p in phases],
    width,
    label="Persistence (lag-7)",
    color="#e08214",
)
ax.bar(
    x,
    [results[p]["r_mean"] for p in phases],
    width,
    label="Model (mean head)",
    color="#2166ac",
)
ax.bar(
    x + width,
    [results[p]["r_q"] for p in phases],
    width,
    label="Model (quantile head)",
    color="#d6604d",
)
ax.axhline(0, color="k", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels([f"{p}\n(n={results[p]['n']})" for p in phases])
ax.set_ylabel("Pearson r")
ax.set_title(
    f"Onset/mid-event/no-MHW skill — {LABEL} (stratified_kfold, pooled 5 folds)"
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
out_path = OUT_DIR / f"onset_skill_{LABEL}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}", flush=True)

out_npz = OUT_DIR / f"onset_skill_{LABEL}_pooled.npz"
np.savez(
    out_npz,
    **{
        f"{phase}_{k}": np.array(v) for phase, d in pooled.items() for k, v in d.items()
    },
)
print(f"Saved {out_npz}", flush=True)
