"""
sustained_lead_time.py — Aug 24 2026, user's methodological question after
reviewing scripts/eval_event_detection.py::score_alarms(): the current
"lead time" per hit is the FARTHEST alarm day within [onset-LEAD, onset]
that is True, with NO requirement that the alarm stay on afterward -- a
single isolated day counts exactly the same as a 14-day unbroken block.
This script does NOT modify eval_event_detection.py. It reuses the exact
same run_fold() logic (same p90_ns, same apply_hobday, same LEAD-from-
config, same persistence definition) to regenerate alarm_bool per day per
system, but additionally computes two stricter/alternative lead-time
definitions per event and -- unlike eval_event_detection.py -- SAVES the
full per-event alarm_bool sequence to disk so this never has to be
re-derived by rerunning inference again.

Three lead-time definitions computed side by side, per event:
  1. original   -- eval_event_detection.py's score_alarms(): earliest
                    (farthest-from-onset) day in [onset-LEAD,onset] with
                    alarm_bool==True, ANYWHERE in the window, no
                    sustainment required. Miss if no True day in window.
  2. sustained_full -- alarm must be True on the onset day itself AND
                    stay True with NO gap all the way back to some day d;
                    lead_time = onset - d (d = start of that unbroken
                    run). Miss if alarm_bool[onset] is False (no run
                    reaches onset at all).
  3. sustained_K (K=3 by default) -- laxer than (2): does not require the
                    run to reach/include the onset day. Finds the
                    earliest (farthest-from-onset, same preference as (1))
                    start day s such that alarm_bool[s:s+K] are ALL True
                    and s+K-1 <= onset; lead_time = onset - s. Miss if no
                    K-consecutive-day block exists anywhere in the window.
                    NOTE: this K-definition is my own reading of an
                    ambiguous spec ("en al menos K de los ultimos K dias")
                    -- flagged explicitly in the printed output, confirm
                    before trusting it as final.

Scope (per user's explicit instructions, Aug 24 2026): full_lead7 family
ONLY this run (config_dir=full_gnll_quantile_v2_landfill, mode=full,
lead=7) -- NOT the other 4 lead families yet, NOT SLURM, NOT touching
eval_event_detection.py or any checkpoint/config.

CPU only.

Usage:
  python scripts/analysis/sustained_lead_time.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.data.masking import mask_local, mask_remote  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.hobday import apply_hobday, load_ns_p90  # noqa: E402

# Aug 24 2026: generalized from the hardcoded full_lead7-only version (kept
# defaults identical so the already-validated full_lead7 run/output is
# unaffected) to also cover lead14 etc., using the exact same
# config_dir/mode/label mapping already used by
# scripts/slurm/submit_event_detection_all_families.sh -- not reinventing
# the mapping.
MASK_FNS = {"full": lambda xs: xs, "local_only": mask_local, "remote_only": mask_remote}
_parser = argparse.ArgumentParser()
_parser.add_argument("--config_dir", default="full_gnll_quantile_v2_landfill")
_parser.add_argument("--mode", default="full", choices=list(MASK_FNS.keys()))
_parser.add_argument("--label", default="full_lead7")
_args = _parser.parse_args()

CONFIG_DIR = _args.config_dir
LABEL = _args.label
MASK_FN = MASK_FNS[_args.mode]
N_FOLDS = 5
K_SUSTAIN = 3
device = "cpu"
SYSTEMS = ["quantile_head", "mean_head", "persist"]

FIGURES_DIR = REPO_ROOT / "experiments" / "figures"
OUT_DIR = FIGURES_DIR / "_fold_cache"

p90_ns = load_ns_p90()
LEAD = yaml.safe_load(
    open(REPO_ROOT / "configs" / "partition" / CONFIG_DIR / "fold0.yaml")
)["lead_time"]
print(
    f"CONFIG_DIR={CONFIG_DIR}  LABEL={LABEL}  LEAD={LEAD}  K_SUSTAIN={K_SUSTAIN}",
    flush=True,
)


def event_onsets_and_spans(mhw_bool):
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


def lead_time_original(alarm_bool, onset):
    """Identical logic to eval_event_detection.py::score_alarms -- earliest
    (farthest) True day anywhere in [onset-LEAD, onset], no sustainment."""
    window = range(max(0, onset - LEAD), onset + 1)
    earliest = None
    for day in window:
        if alarm_bool[day] and (earliest is None or day < earliest):
            earliest = day
    if earliest is None:
        return None
    return onset - earliest


def lead_time_sustained_full(alarm_bool, onset):
    """Alarm must be True at onset itself and stay True with zero gaps
    back to day d; lead_time = onset - d. Miss if alarm_bool[onset] is
    False."""
    if not alarm_bool[onset]:
        return None
    lo_bound = max(0, onset - LEAD)
    d = onset
    while d - 1 >= lo_bound and alarm_bool[d - 1]:
        d -= 1
    return onset - d


def lead_time_sustained_K(alarm_bool, onset, K):
    """Earliest (farthest-from-onset) K-consecutive-day True block fully
    inside [onset-LEAD, onset], NOT required to touch onset itself."""
    lo_bound = max(0, onset - LEAD)
    for s in range(lo_bound, onset - K + 2):  # s+K-1 <= onset
        if s + K - 1 > onset:
            break
        if all(alarm_bool[s : s + K]):
            return onset - s
    return None


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
            xs = MASK_FN(xs)
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

    persist = np.full_like(trues, np.nan)
    persist[LEAD:] = trues[:-LEAD]
    year_of_i_minus_lead = np.concatenate([np.full(LEAD, -999), years[:-LEAD]])
    persist[years != year_of_i_minus_lead] = np.nan

    thr = p90_ns[doys - 1]

    fold_result = {
        "n_days": len(trues),
        "alarm_counts": {s: 0 for s in SYSTEMS},
        "persist_valid_count": 0,
        "events": [],  # list of dicts: yr, onset_local_idx, per-system alarm seq + 3 lead-time defs
        # Aug 24 2026: user's block-structure question -- needs the FULL
        # per-year alarm_bool series (not just the windowed slice around
        # each event) to characterize alarm episodes across the whole test
        # set. Kept as a SEPARATE list from "events" (different downstream
        # file, see main loop below) so the already-verified events/
        # lead-time output is untouched by this addition.
        "full_series": [],  # list of dicts: yr, per-system full alarm_bool array (+ persist validity mask)
    }

    for yr in np.unique(years):
        ymask = years == yr
        true_y, thr_y = trues[ymask], thr[ymask]
        q_y, mean_y, pers_y = q_preds[ymask], mean_preds[ymask], persist[ymask]
        mhw_y = apply_hobday(true_y > thr_y)
        onsets_y, spans_y = event_onsets_and_spans(mhw_y)

        alarms = {
            "quantile_head": q_y > thr_y,
            "mean_head": mean_y > thr_y,
            "persist": np.where(np.isfinite(pers_y), pers_y > thr_y, False),
        }
        for s in SYSTEMS:
            fold_result["alarm_counts"][s] += int(
                alarms[s].sum()
                if s != "persist"
                else alarms[s][np.isfinite(pers_y)].sum()
            )
        fold_result["persist_valid_count"] += int(np.isfinite(pers_y).sum())

        # same local-day indexing convention as onset_local_idx/window_start
        # below, so these arrays are directly indexable with those values.
        fold_result["full_series"].append(
            {
                "year": int(yr),
                "n_days_year": int(ymask.sum()),
                "quantile_head_alarm": alarms["quantile_head"].astype(bool),
                "mean_head_alarm": alarms["mean_head"].astype(bool),
                "persist_alarm": alarms["persist"].astype(bool),
                "persist_valid": np.isfinite(pers_y),
            }
        )

        for onset in onsets_y:
            lo_bound = max(0, onset - LEAD)
            ev = {
                "year": int(yr),
                "onset_local_idx": int(onset),
                "window_start": int(lo_bound),
            }
            for s in SYSTEMS:
                seq = alarms[s][lo_bound : onset + 1].astype(bool)
                ev[f"{s}_seq"] = seq
                ev[f"{s}_lead_original"] = lead_time_original(alarms[s], onset)
                ev[f"{s}_lead_sustained_full"] = lead_time_sustained_full(
                    alarms[s], onset
                )
                ev[f"{s}_lead_sustained_K{K_SUSTAIN}"] = lead_time_sustained_K(
                    alarms[s], onset, K_SUSTAIN
                )
            fold_result["events"].append(ev)

    print(
        f"  fold {fold}: n_days={fold_result['n_days']}  n_events={len(fold_result['events'])}  "
        + "  ".join(f"{s}_alarm={fold_result['alarm_counts'][s]}" for s in SYSTEMS),
        flush=True,
    )
    return fold_result


OUT_DIR.mkdir(parents=True, exist_ok=True)

all_events = []
all_full_series = []
tot_days = 0
tot_alarm = {s: 0 for s in SYSTEMS}
tot_persist_valid = 0

for fold in range(N_FOLDS):
    fr = run_fold(fold)
    tot_days += fr["n_days"]
    for s in SYSTEMS:
        tot_alarm[s] += fr["alarm_counts"][s]
    tot_persist_valid += fr["persist_valid_count"]
    for ev in fr["events"]:
        ev["fold"] = fold
        all_events.append(ev)
    for fs in fr["full_series"]:
        fs["fold"] = fold
        all_full_series.append(fs)
    # incremental save per fold (CLAUDE.md rule for jobs >1h, same pattern
    # as eval_recall_v2_partition.py) -- so a mid-run failure/timeout still
    # leaves the folds done so far on disk instead of losing everything.
    # UNCHANGED from before -- still only ever writes event_alarm_sequences_*,
    # same content/logic as the already-verified full_lead7/lead14 runs.
    np.savez(
        OUT_DIR / f"event_alarm_sequences_{LABEL}_partial.npz",
        events=np.array(all_events, dtype=object),
        n_folds_done=fold + 1,
        lead=LEAD,
        k_sustain=K_SUSTAIN,
        tot_days=tot_days,
        tot_alarm=tot_alarm,
        tot_persist_valid=tot_persist_valid,
    )
    # NEW, separate file -- full per-year alarm_bool series for the whole
    # test set (block-structure question), written in parallel, never to
    # the events file above.
    np.savez(
        OUT_DIR / f"full_alarm_series_{LABEL}_partial.npz",
        full_series=np.array(all_full_series, dtype=object),
        n_folds_done=fold + 1,
        lead=LEAD,
    )

n_events = len(all_events)
print(f"\n{'='*76}")
print(f"{LABEL}: n_events={n_events} pooled, n_days={tot_days} pooled, LEAD={LEAD}")
print(f"{'='*76}")

print("\nFraction of ALL test days with alarm_bool==True, per system:")
for s in SYSTEMS:
    denom = tot_persist_valid if s == "persist" else tot_days
    frac = tot_alarm[s] / denom
    print(
        f"  {s:14s}: {tot_alarm[s]}/{denom} = {frac:.4f}"
        + (
            f"  (of {tot_days} total incl. boundary-NaN excluded)"
            if s == "persist"
            else ""
        )
    )

print(f"\n{'='*76}")
print("POD + lead-time distribution, 3 definitions, per system")
print(f"{'='*76}")

results_summary = {}
for s in SYSTEMS:
    for defname in ["original", "sustained_full", f"sustained_K{K_SUSTAIN}"]:
        key = f"{s}_lead_{defname}"
        lts = [ev[key] for ev in all_events]
        hits = [lt for lt in lts if lt is not None]
        pod = len(hits) / n_events
        print(f"\n{s} / {defname}: hits={len(hits)}/{n_events}  POD={pod:.4f}")
        if hits:
            vals, counts = np.unique(hits, return_counts=True)
            hist = {int(v): int(c) for v, c in zip(vals, counts)}
            print(f"  lead-time histogram (day: n_hits): {hist}")
            print(f"  mean={np.mean(hits):.2f}  median={np.median(hits):.1f}")
        results_summary[key] = dict(pod=pod, n_hits=len(hits), lead_times=hits)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"event_alarm_sequences_{LABEL}.npz"
np.savez(
    out_path,
    events=np.array(all_events, dtype=object),
    n_events=n_events,
    lead=LEAD,
    k_sustain=K_SUSTAIN,
    tot_days=tot_days,
    tot_alarm=tot_alarm,
    tot_persist_valid=tot_persist_valid,
    results_summary=results_summary,
)
print(
    f"\nSaved {out_path} (per-event alarm_bool sequences + all 3 lead-time defs, reusable without rerunning inference)",
    flush=True,
)

# Separate, parallel output -- full per-year alarm_bool series for the
# whole test set, for block/episode-structure analysis. Independent file,
# never overwrites/depends on event_alarm_sequences_{LABEL}.npz above.
full_series_out_path = OUT_DIR / f"full_alarm_series_{LABEL}.npz"
np.savez(
    full_series_out_path, full_series=np.array(all_full_series, dtype=object), lead=LEAD
)
print(
    f"Saved {full_series_out_path} (full per-year alarm_bool series, all 3 systems, for block-structure analysis)",
    flush=True,
)
