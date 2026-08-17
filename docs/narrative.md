# Paper narrative — scientific decisions

_Fill this in as decisions are made. One section per topic.
Dates help track when a decision was settled._

---

## Framing and objective

<!-- What is the paper about, in one paragraph.
     What is NOT the paper about (scope limits). -->

---

## Key results

<!-- The 3-5 results that carry the paper.
     One bullet per result, with the number and its source experiment. -->

---

## Onset skill

**Onset skill (n=84, pooled across 5 folds, corrected MHW definition):**
neither the model (r=0.006 to 0.093 depending on partition mode, all CI
crossing zero) nor lag-7 persistence (r=−0.067) predict MHW onset using
information available up to 7 days before. This is a genuine negative
result, not a data artifact — power to detect |r|>0.21 at n=84 (p<0.05).
The earlier r=0.245/0.201 figures came from unweighted fold-averaging
(a low-n fold inflated the mean) and are superseded by this pooled
estimate. Physical interpretation: the atmospheric/oceanic state 7 days
before onset does not yet carry a distinguishable precursor signal for
these 84 specific transition days, even though the model has real skill
overall (r=0.86–0.91) and at mid-event (once the anomaly is already
elevated).

| Mode        |  n  | r\_model | 95% CI             | r\_persist | 95% CI             |
|-------------|----:|---------:|:------------------:|-----------:|:------------------:|
| full        |  84 |   0.0055 | [−0.209, +0.220]   |    −0.0673 | [−0.278, +0.149]   |
| remote\_only|  84 |   0.0536 | [−0.163, +0.265]   |    −0.0673 | [−0.278, +0.149]   |
| local\_only |  84 |   0.0931 | [−0.124, +0.301]   |    −0.0673 | [−0.278, +0.149]   |

_onset = first day of Hobday MHW event in test target series (trues > p90\_thresh,
≥5 consecutive days, gap ≤2d merged). persistence = to\_anom at last input day
(7 days before target, i.e. trues[i−7] within each fold's consecutive test series).
CI = 95% Fisher z-transform. r\_persist identical across modes because it depends
only on the target series, not on model predictions. Source: scripts/eval\_onset\_persistence.py,
Aug 16 2026._

---

## Partition: local vs remote drivers

<!-- Partition table (MSE, kfold, 5 folds).
     Interpretation of local_only > full.
     What remote_only onset skill implies for the narrative. -->

> **⚠ SUPERSEDED — Full=0.860±0.031 DO NOT CITE**
>
> The Full baseline cited previously (r=0.860±0.031) was sourced from
> `experiments/kfold/TbotAtm_lstmonly_fold0-4` (Jun 23 2026, PRE-FIX #21).
> Forensic check: `data_dir` hardcoded to `merged_daily_deepSST.nc` in all
> 5 configs; mtime=ctime=Jun 23 for all 15 checkpoints (no post-date
> manipulation). The fix (ptho_bot anomaly, job 14070999) ran Jun 29 2026 —
> 5 days 22 hours after these checkpoints were written.
>
> The 93% ratio (Remote=0.802/Full=0.860) is INVALID: denominator
> contaminated (pre-fix ptho_bot ≈ absolute temperature 7-8°C, not anomaly),
> numerators Remote=0.802 and Local=0.906 are clean (Aug 11 2026, post-fix).
>
> Retraining in progress: `experiments/partition/TbotAtm_full_seed42_fold*`
> (seed=42, 5 folds, `merged_daily.nc`, arch=lstm_only).
> Submitted Aug 16 2026 — job 14199765 (array 0-4, booster partition).
> Update this section when job completes.

---

## XAI methodology

<!-- Which IG flavor (signed vs unsigned, per-year vs pooled).
     What Hobday decomposition adds.
     What GradCAM adds (or why it's not in the paper). -->

---

## Granger methodology

<!-- Disclaimer on F-stat comparability across variables.
     What Granger is used for (causal direction only, not ranking).
     How it complements IG (short-lag coupling vs multi-week preconditioning).
     Whether ΔR² is included. -->

---

## Window size and lead time

<!-- Why 60-day window (link to ACF / tau_ns).
     Why lead=7d (operational relevance + skill curve).
     Lead sweep results if included. -->

---

## Architecture choices

<!-- Why LSTM-only (vs attention, TCN, ConvLSTM).
     Why MSE (vs GNLL — include GNLL as supplementary or drop).
     K-fold CV: why 5 folds, how splits were made. -->

---

## Spatial prediction (2D, encoder-ConvLSTM-decoder)

Parallel experiment in `/p/project1/hai_1127/radin1/spatial_forecast/` — not
integrated into exprecursors repo. Audited Fase 4-bis, Aug 16-17 2026.

**NF-S-2 (NaN normalization bug): root cause confirmed in current code.**
`to_anom` has 489 NaN ocean pixels (coastal boundary, incomplete ICON-COAST
coverage — confirmed by direct `data_vars` inspection). Old code: `pix_std =
target_frames.std(dim=0)` propagated those NaN → `pix_std[ocean_mask].mean()`
returned NaN (any-NaN-in-mean → NaN) → logged as "Target std (ocean mean): nan".
Current code (`dataset_spatial.py` modified Jul 7 22:31): lines 129-130 apply
`pix_std[self.land_mask] = 1.0` then `nan_to_num(pix_std, nan=1.0)` BEFORE
the print and BEFORE the division → those 489 pixels get pix_std=1.0, no NaN
propagation to targets. Timeline confirms: Jun 30 runs (slurm-14075xxx, buggy,
deleted) used old code; Jul 7+ runs (checkpoints 23:44–, valid) used current code.

**Result citable as preliminary (single-fold):**
SSTAtm_phys_fold0_l0p0 (λ_lap=0, fold 0): ocean-mean r = **0.8055** (median 0.8411).
Ran via `train_spatial_phys.py` (not train_spatial.py). The NEEDS_FIX issues in
train_spatial_phys.py do not affect this result: NF-S-3 (fold_seed=0 hardcoded)
means test years were assigned with seed=0, not config seed=42 — but all spatial
phys runs share this same fold assignment, so r=0.8055 is internally consistent.
NF-S-4 (duplicated build_splits) has no computational effect on any single run.
Caveats: single fold only; MLD-weighted MSE with λ_lap=0 reduces to plain
MLD-MSE (not a composite metric here).

**⚠️ INVALID — spatial onset skill figures (Aug 10 2026):**
`eval/figures/onset_skill_maps.png`, `eval/figures/onset_skill_ns_SSTAtm.png`,
`eval/figures/onset_skill_ns_TbotAtm.png`, and `eval/figures/onset_skill_table.txt`
were generated by `eval/mhw_onset_skill.py` using `is_mhw_tgt = tgt > 0` as the
MHW criterion (lines 117-118). This labels **~31.5% of target pixels as MHW**
(Hobday expected ~10% — no p90(DOY), no persistence filter). All onset/mid-event/
no-MHW pixel-level skill maps are physically meaningless and must NOT be cited
or shown. Fix requires a 2D per-pixel Hobday extension (NF-S-5, docs/audit_plan.md
Fase 4-bis).

---

## Open scientific questions

<!-- Things not settled yet. Remove entries when resolved. -->
