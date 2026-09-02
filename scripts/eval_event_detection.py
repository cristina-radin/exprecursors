"""
eval_event_detection.py — event-level onset detection (POD/FAR/CSI +
lead-time distribution + uncertainty), Aug 21 2026, requested by the
user as "la pregunta real de si predice el inicio, y la version
publicable" of the day-level onset-skill analysis.

Reviewed before launch (user caught 3 real gaps in the first draft, all
fixed here, not launched until addressed):
  1. Mean head added as a third arm (was quantile-vs-persistence only).
     `forward_with_quantile()` already returns both heads for free -- the
     real story for the paper is whether the quantile head catches
     onsets the mean head misses (ties into the -0.39C onset bias
     already documented), not just model-vs-persistence.
  2. Uncertainty added: Clopper-Pearson exact binomial CI for POD/FAR/CSI
     per system (n_events~56 is small, a bare point estimate like
     "POD=0.75 vs 0.60" means nothing without it), PLUS an event-paired
     bootstrap (resample events with replacement, not systems
     independently) for the POD *difference* between each model head and
     persistence -- the paired design matters because all systems are
     scored on the same events.
  3. False-alarm convention made explicit and a sensitivity check added:
     an alarm episode counts as a false alarm whenever it does not
     overlap an onset's [onset-LEAD, onset] window -- this INCLUDES a
     re-activation partway through an already-ongoing (already-hit)
     event, since that episode doesn't overlap any onset window either.
     Defensible and applied identically to every system (symmetric), but
     documented here explicitly rather than left implicit, plus a
     stricter sensitivity variant (FAR_strict/CSI_strict) that excludes
     false alarms whose episode overlaps ANY event's full duration (not
     just the onset window) -- i.e. only counts alarms on genuinely
     quiet (non-MHW) days as false alarms.

Definitions:
  - alarm(t)   = head_pred(t) > p90_ns[doy(t)]  (raw per-day threshold
                 crossing, no Hobday duration/gap smoothing applied to
                 the alarm itself).
  - real event = Hobday MHW event on the true target.
  - HIT        = a real event with >=1 alarm day in [t_onset-7, t_onset].
  - lead_time (hits) = t_onset - (earliest alarm day in the window).

Per-year Hobday/alarm-run processing (known_issues.md #53 --
stratified_kfold's non-consecutive test years must not be concatenated
before running duration/gap logic).

Reuses: src/utils/hobday.py (apply_hobday, load_ns_p90), LazyDataModule +
best_ckpt() + load_model_config() + CNNLightningModule.load_from_checkpoint.

CPU only. Not part of the permanent pipeline.

Usage:
  python scripts/eval_event_detection.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import beta as beta_dist
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse  # noqa: E402

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.data.masking import mask_local, mask_remote  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.hobday import apply_hobday, load_ns_p90  # noqa: E402

# Aug 24 2026: same masking bug already fixed in eval_recall_v2_partition.py
# -- local_only/remote_only checkpoints were trained on masked input via
# train_partition.py's LocalOnlyLightningModule/RemoteOnlyLightningModule,
# which only override the LightningModule's step methods, not
# model.forward_with_quantile() itself. Calling that directly (as run_fold()
# below does) bypasses masking unless applied explicitly here -- evaluating
# those checkpoints on unmasked input is out-of-distribution and wrong.
MASK_FNS = {
    "full": lambda xs: xs,
    "local_only": mask_local,
    "remote_only": mask_remote,
}

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--config_dir",
    required=True,
    help="dir under configs/partition/ -- Aug 23 2026: generalized across "
    "the lead-time sweep (this script used to be hardcoded to the OLD "
    "committed full_gnll_quantile_v2 (zero-fill) config, predating both "
    "the lead-sweep and the land_fill_mode=nearest decision).",
)
_parser.add_argument("--label", required=True, help="used for output filenames")
_parser.add_argument("--mode", default="full", choices=list(MASK_FNS.keys()))
_parser.add_argument(
    "--folds",
    default=None,
    help="comma-separated fold indices to evaluate and pool (default: all "
    "N_FOLDS=5). For configs with partial fold coverage -- e.g. the "
    "state_feature hybrid, which only has folds 0,1 trained -- pass "
    "--folds 0,1 rather than let best_ckpt() fail on a missing checkpoint "
    "dir for folds 2-4.",
)
_args = _parser.parse_args()
CONFIG_DIR = _args.config_dir
LABEL = _args.label
MASK_FN = MASK_FNS[_args.mode]

OUT_DIR = REPO_ROOT / "experiments" / "figures" / "step7_persistence"
# LEAD used to be hardcoded to 7; read from the target config's own
# lead_time so this script is correctly reusable across the lead-time
# sweep (3/5/7/14/30d), not just lead=7 by coincidence. All folds in one
# config_dir share the same lead_time.
LEAD = yaml.safe_load(
    open(REPO_ROOT / "configs" / "partition" / CONFIG_DIR / "fold0.yaml")
)["lead_time"]
N_FOLDS = 5
FOLDS = (
    [int(x) for x in _args.folds.split(",")] if _args.folds else list(range(N_FOLDS))
)
N_BOOT = 2000
SYSTEMS = ["quantile_head", "mean_head", "persist"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)

p90_ns = load_ns_p90()
rng = np.random.default_rng(42)


def alarm_episodes(alarm_bool):
    episodes = []
    i, n = 0, len(alarm_bool)
    while i < n:
        if alarm_bool[i]:
            j = i
            while j < n and alarm_bool[j]:
                j += 1
            episodes.append((i, j - 1))
            i = j
        else:
            i += 1
    return episodes


def event_onsets_and_spans(mhw_bool):
    """Returns (onset_indices, (start,end) span per event) -- span used
    only for the FAR_strict sensitivity check."""
    onsets, spans = [], []
    i, n = 0, len(mhw_bool)
    while i < n:
        if mhw_bool[i]:
            j = i
            while j < n and mhw_bool[j]:
                j += 1
            onsets.append(i)
            spans.append((i, j - 1))
            i = j
        else:
            i += 1
    return onsets, spans


def score_alarms(alarm_bool, onset_idx_list, event_spans):
    """Returns per-event hit booleans (aligned to onset_idx_list),
    false_alarm_count, false_alarm_count_strict (excludes re-activation
    FAs that overlap ANY event's full span), lead_times."""
    episodes = alarm_episodes(alarm_bool)
    hit_bool = np.zeros(len(onset_idx_list), dtype=bool)
    lead_times = []
    episode_hits_onset = [False] * len(episodes)

    for oi, onset in enumerate(onset_idx_list):
        window = range(max(0, onset - LEAD), onset + 1)
        earliest = None
        for day in window:
            if alarm_bool[day] and (earliest is None or day < earliest):
                earliest = day
        if earliest is not None:
            hit_bool[oi] = True
            lead_times.append(onset - earliest)
            for ei, (s, e) in enumerate(episodes):
                if s <= onset and e >= max(0, onset - LEAD):
                    episode_hits_onset[ei] = True

    fa = sum(1 for x in episode_hits_onset if not x)

    # strict: an episode is only a "real" false alarm if it doesn't
    # overlap ANY event's full [start,end] span, not just onset windows
    fa_strict = 0
    for ei, (s, e) in enumerate(episodes):
        if episode_hits_onset[ei]:
            continue
        overlaps_any_event = any(s <= es and e >= vs for vs, es in event_spans)
        if not overlaps_any_event:
            fa_strict += 1

    return hit_bool, fa, fa_strict, lead_times


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

    # Aug 24 2026: hybrid (use_state_feature=true) configs return a 4th
    # batch element (x_state, see dataset.py __getitem__) that must be
    # threaded through to forward_with_quantile() -- the model raises
    # explicitly (no silent fallback) if state_feature=True and x_state is
    # None, so this can't be skipped for those checkpoints.
    use_state = cfg.get("use_state_feature", False)

    test_dl = DataLoader(dm.test_dataset, batch_size=32, shuffle=False, num_workers=0)
    mean_preds, q_preds, trues = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            xs, xt, y = batch[0], batch[1], batch[2]
            x_state = batch[3].float().to(device) if use_state else None
            xs, xt = xs.float().to(device), xt.float().to(device)
            xs = MASK_FN(xs)
            y_hat, q_pred = lm.model.forward_with_quantile(xs, xt, x_state)
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

    persist = np.full_like(trues, np.nan)
    persist[LEAD:] = trues[:-LEAD]
    year_of_i_minus_lead = np.concatenate([np.full(LEAD, -999), years[:-LEAD]])
    persist[years != year_of_i_minus_lead] = np.nan

    thr = p90_ns[doys - 1]

    per_system = {s: dict(hit=[], fa=0, fa_strict=0, lt=[]) for s in SYSTEMS}
    n_events_fold = 0

    for yr in np.unique(years):
        ymask = years == yr
        true_y, thr_y = trues[ymask], thr[ymask]
        q_y, mean_y, pers_y = q_preds[ymask], mean_preds[ymask], persist[ymask]
        mhw_y = apply_hobday(true_y > thr_y)
        onsets_y, spans_y = event_onsets_and_spans(mhw_y)
        n_events_fold += len(onsets_y)
        if not onsets_y:
            continue

        alarms = {
            "quantile_head": q_y > thr_y,
            "mean_head": mean_y > thr_y,
            "persist": np.where(np.isfinite(pers_y), pers_y > thr_y, False),
        }
        for sysname, alarm_y in alarms.items():
            hit_bool, fa, fa_strict, lt = score_alarms(alarm_y, onsets_y, spans_y)
            per_system[sysname]["hit"].append(hit_bool)
            per_system[sysname]["fa"] += fa
            per_system[sysname]["fa_strict"] += fa_strict
            per_system[sysname]["lt"].extend(lt)

    for s in SYSTEMS:
        per_system[s]["hit"] = (
            np.concatenate(per_system[s]["hit"])
            if per_system[s]["hit"]
            else np.array([], dtype=bool)
        )

    print(
        f"  fold {fold}: n_events={n_events_fold}  "
        + "  ".join(
            f"{s}(hit={per_system[s]['hit'].sum()},fa={per_system[s]['fa']})"
            for s in SYSTEMS
        ),
        flush=True,
    )
    return per_system, n_events_fold


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


pooled = {s: dict(hit=[], fa=0, fa_strict=0, lt=[]) for s in SYSTEMS}
n_events_total = 0
for fold in FOLDS:
    per_system, n_events_fold = run_fold(fold)
    n_events_total += n_events_fold
    for s in SYSTEMS:
        pooled[s]["hit"].append(per_system[s]["hit"])
        pooled[s]["fa"] += per_system[s]["fa"]
        pooled[s]["fa_strict"] += per_system[s]["fa_strict"]
        pooled[s]["lt"].extend(per_system[s]["lt"])

for s in SYSTEMS:
    pooled[s]["hit"] = np.concatenate(pooled[s]["hit"])

n_events = len(pooled[SYSTEMS[0]]["hit"])
assert n_events == n_events_total, (n_events, n_events_total)

print(f"\n{'='*76}")
folds_desc = f"pooled {len(FOLDS)} fold(s) {FOLDS}" + (
    "" if len(FOLDS) == N_FOLDS else f" -- PARTIAL, not all {N_FOLDS} folds"
)
print(
    f"EVENT-LEVEL DETECTION, {LABEL} (n_events={n_events}, {folds_desc}, lead={LEAD}d window)"
)
print(f"{'='*76}")

summary = {}
for s in SYSTEMS:
    hits = int(pooled[s]["hit"].sum())
    misses = n_events - hits
    fa = pooled[s]["fa"]
    fa_strict = pooled[s]["fa_strict"]
    pod = hits / max(1, n_events)
    far = fa / max(1, hits + fa)
    far_strict = fa_strict / max(1, hits + fa_strict)
    csi = hits / max(1, hits + misses + fa)
    csi_strict = hits / max(1, hits + misses + fa_strict)
    pod_ci = clopper_pearson(hits, n_events)
    far_ci = clopper_pearson(fa, hits + fa)
    csi_denom = hits + misses + fa
    print(f"\n{s}:")
    print(f"  hits={hits}  misses={misses}  false_alarms={fa} (strict={fa_strict})")
    print(f"  POD={pod:.3f} [{pod_ci[0]:.3f},{pod_ci[1]:.3f}] (95% Clopper-Pearson)")
    print(
        f"  FAR={far:.3f} [{far_ci[0]:.3f},{far_ci[1]:.3f}]   FAR_strict={far_strict:.3f}"
    )
    print(f"  CSI={csi:.3f}   CSI_strict={csi_strict:.3f}")
    if pooled[s]["lt"]:
        lt = pooled[s]["lt"]
        print(
            f"  lead_time: mean={np.mean(lt):.2f}  median={np.median(lt):.1f}  n={len(lt)}"
        )
    summary[s] = dict(
        hits=hits,
        misses=misses,
        fa=fa,
        fa_strict=fa_strict,
        pod=pod,
        pod_ci=pod_ci,
        far=far,
        far_ci=far_ci,
        far_strict=far_strict,
        csi=csi,
        csi_strict=csi_strict,
        lead_times=pooled[s]["lt"],
    )

# ── Event-paired bootstrap for POD difference vs persistence ───────────────
print(f"\n{'='*76}")
print(f"Event-paired bootstrap (B={N_BOOT}) -- POD(head) - POD(persist), 95% CI")
print(f"{'='*76}")
boot_results = {}
for s in ["quantile_head", "mean_head"]:
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n_events, size=n_events)
        pod_s = pooled[s]["hit"][idx].mean()
        pod_p = pooled["persist"]["hit"][idx].mean()
        diffs[b] = pod_s - pod_p
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    mean_diff = diffs.mean()
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
    print(
        f"  {s} vs persist: mean_diff={mean_diff:+.3f}  95% CI=[{lo:+.3f},{hi:+.3f}]  ({sig})"
    )
    boot_results[s] = dict(
        mean_diff=float(mean_diff),
        ci=(float(lo), float(hi)),
        significant=bool(lo > 0 or hi < 0),
    )

print(f"\n{'='*76}")
print(
    "Mean head vs quantile head -- does the quantile head catch onsets the mean head misses?"
)
print(f"{'='*76}")
mean_hit = pooled["mean_head"]["hit"]
q_hit = pooled["quantile_head"]["hit"]
both = int((mean_hit & q_hit).sum())
only_q = int((~mean_hit & q_hit).sum())
only_mean = int((mean_hit & ~q_hit).sum())
neither = int((~mean_hit & ~q_hit).sum())
print(
    f"  both hit: {both}   only quantile hit: {only_q}   only mean hit: {only_mean}   neither: {neither}"
)
diffs_qm = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = rng.integers(0, n_events, size=n_events)
    diffs_qm[b] = q_hit[idx].mean() - mean_hit[idx].mean()
lo, hi = np.percentile(diffs_qm, [2.5, 97.5])
print(
    f"  POD(quantile) - POD(mean): mean_diff={diffs_qm.mean():+.3f}  95% CI=[{lo:+.3f},{hi:+.3f}]"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
metrics = ["POD", "FAR", "CSI"]
x = np.arange(len(metrics))
width = 0.25
colors = {"quantile_head": "#d6604d", "mean_head": "#2166ac", "persist": "#e08214"}
for i, s in enumerate(SYSTEMS):
    vals = [summary[s]["pod"], summary[s]["far"], summary[s]["csi"]]
    errs_lo = [
        summary[s]["pod"] - summary[s]["pod_ci"][0],
        summary[s]["far"] - summary[s]["far_ci"][0],
        0,
    ]
    errs_hi = [
        summary[s]["pod_ci"][1] - summary[s]["pod"],
        summary[s]["far_ci"][1] - summary[s]["far"],
        0,
    ]
    ax.bar(
        x + (i - 1) * width,
        vals,
        width,
        label=s,
        color=colors[s],
        yerr=[errs_lo, errs_hi],
        capsize=3,
    )
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.set_title(f"Event detection, {LABEL}, lead={LEAD}d (n_events={n_events}, 95% CI)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
bins = np.arange(-0.5, LEAD + 1.5, 1)
for s in SYSTEMS:
    if summary[s]["lead_times"]:
        ax.hist(
            summary[s]["lead_times"], bins=bins, alpha=0.5, label=s, color=colors[s]
        )
ax.set_xlabel("Lead time (days before onset)")
ax.set_ylabel("n hits")
ax.set_title("Lead-time distribution (hits only)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / f"event_detection_pod_far_csi_{LABEL}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}", flush=True)

import json  # noqa: E402

with open(OUT_DIR / f"event_detection_summary_{LABEL}.json", "w") as f:
    json.dump(
        {
            "summary": summary,
            "bootstrap_vs_persist": boot_results,
            "n_events": n_events,
        },
        f,
        indent=2,
        default=str,
    )
print(f"Saved {OUT_DIR / f'event_detection_summary_{LABEL}.json'}", flush=True)
