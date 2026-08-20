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

**Clean results (job 14199778, post-fix, seed=42, 5 folds, `merged_daily.nc`, arch=lstm_only):**

| Partition | r (mean±std) | ratio vs Full |
|-----------|-------------|---------------|
| Full      | 0.852±0.082 | —             |
| Remote    | 0.802±0.041 | 94.1%         |
| Local     | 0.906±0.032 | 106.3%        |

Local > Full: local SST signal alone outperforms the full input set — consistent
with the short-range thermal inertia dominating 7-day predictability in the NS box.

### Fold0 outlier — Full underperforms Remote (0.715 vs 0.812)

fold0 test years [1995, 1997, 2003, 2004, 2014, 2019] include 3 of the
8 highest-anomaly years in the 1993–2022 record (2003, 2014, 2019;
mean max_anom 4.24°C vs 3.75°C for fold2/fold4). This alone would explain
Full scoring lower here than in other folds — but does not by itself
explain why Full < Remote specifically in this fold (Full ≥ Remote in
all other 4 folds).

Convergence diagnosis (val_loss gap between best checkpoint and final epoch, fold0 only):
- Full: best_epoch=14, gap=0.057 (most severe overfitting)
- Remote: best_epoch=26, gap=0.024
- Local: best_epoch=16, gap=0.031

All three show some overfitting in this fold, but severity ranks exactly opposite to
model complexity: Full (most input variables) is most fragile, Remote (forced to rely
on the consistent remote signal) is most stable. Likely mechanism: in extreme-anomaly
years, Full's larger effective parameter space overfits more easily than the more
constrained Remote/Local models — checkpoint selection (early stopping on val_loss)
caught this before it got worse, but could not fully compensate.

Not a bug, not a reason to rerun — but the mechanism (model richness + extreme-year
fragility) is worth a sentence in the paper if the fold0 number is shown.

---

> ~~**⚠ SUPERSEDED — Full=0.860±0.031 DO NOT CITE**~~
>
> ~~The Full baseline cited previously (r=0.860±0.031) was sourced from~~
> ~~`experiments/kfold/TbotAtm_lstmonly_fold0-4` (Jun 23 2026, PRE-FIX #21).~~
> ~~Forensic check: `data_dir` hardcoded to `merged_daily_deepSST.nc` in all~~
> ~~5 configs; mtime=ctime=Jun 23 for all 15 checkpoints (no post-date~~
> ~~manipulation). The fix (ptho_bot anomaly, job 14070999) ran Jun 29 2026 —~~
> ~~5 days 22 hours after these checkpoints were written.~~
>
> ~~The 93% ratio (Remote=0.802/Full=0.860) is INVALID: denominator~~
> ~~contaminated (pre-fix ptho_bot ≈ absolute temperature 7-8°C, not anomaly),~~
> ~~numerators Remote=0.802 and Local=0.906 are clean (Aug 11 2026, post-fix).~~
>
> Superseded by job 14199778 (Aug 17 2026). Kept for audit trail.

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
     K-fold CV: why 5 folds, how splits were made. -->

_Aug 19 2026 — GNLL vs MSE: global skill hides near-zero tail skill._

Overall r≈0.87 for both gnll and mse_v2 (TbotAtm full, 5-fold test,
`test_predictions.npz`) is driven by the seasonal cycle, which dominates
total variance and is easy for any model to track. Restricted to the days
that actually matter for MHW (truth > p90, n=1454 days), skill collapses:
r=0.218 (gnll), r=0.224 (mse_v2). Amplitude is compressed in both
(std(pred)/std(truth)≈0.83), and on those extreme days mean(pred) is
0.78–0.86°C vs mean(truth)=1.25°C — classic MSE/NLL shrinkage-to-the-mean,
worse for gnll (only 15.2% of true-extreme days end up with pred>p90) than
mse_v2 (34.4%). Do not cite the global r as evidence of MHW-detection
skill — report the two numbers separately.

**MHW event detection — Hobday on the predicted mean underestimates
severely (12/52 events, gnll), and does not use GNLL's own std output.**
Fix: `scripts/mhw_ensemble_hobday.py` samples per-day trajectories from
the predicted N(mean, std), applies the literal Hobday algorithm
(`src/utils/hobday.py:apply_hobday`, unmodified) to each sampled
trajectory, then aggregates by majority vote (fixed at 0.5 — not tuned on
test, to avoid threshold-shopping). Result: 23/52 events, 282/1061 days,
event-overlap recall 19.2%→30.8% vs. the naive mean-threshold approach.
This is the legitimate way to use GNLL's predictive distribution for
event detection; thresholding `P(exceed) = 1-Φ((p90-mean)/std)` per day
directly (no Hobday duration/gap logic) was tried first and rejected —
it silently drops the Hobday definition (classifies a probability, not a
temperature series) and the cutoff (0.3) was picked by eye on test data.

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

## Hypothesis: mechanism behind local/remote redundancy

_Aug 17 2026_

**H1 (atmospheric bridge):** the redundancy between Local-only (r=0.906) and
Full (r=0.852) reflects a shared atmospheric pathway — remote circulation
state (correlated with Gulf Stream position) modulates the North Atlantic
storm track / NAO-like patterns, which reach the North Sea as local wind/
pressure forcing. Prediction: Full's spatial IG attribution in the remote
region should overlap with u10/v10/msl patterns, not with to_anom/ptho_bot
patterns, and should resemble what Remote-only produces independently in
that same region.

**H0 (no shared structure):** redundancy is a numerical coincidence with no
shared spatial pattern — local and remote channels reach similar skill via
unrelated routes.

**Falsification test:** compare spatial IG maps (Full vs. Remote-only vs.
Local-only, TbotAtm, this week's clean checkpoints) — check which variable
class overlaps between Full-remote-region and Remote-only.

**Literature grounding (Aug 17 verification):**
- Climate modes synergistically influence MHW in the North Sea (Ocean
  Science, 2026): winter southern-NS MHW variability regulated by EAP via
  two pathways — atmospheric (strengthened SW winds) and oceanic (Atlantic
  inflow/MOC); positive NAO intensifies winter MHW occurrence. Confirms
  both pathways coexist and are separable — directly supports testing H1
  vs. an oceanic alternative, not assuming one.
- Drivers of the extreme North Atlantic MHW 2023 (Nature, 2025): dominant
  driver was anomalously weak winds over an extremely shallow mixed layer,
  not anomalous ocean heat transport — supports the atmospheric-pathway
  side specifically for North Atlantic basin-scale events.
- Atmospheric patterns drive MHWs in North Atlantic/Mediterranean summer
  2023 (2025): NAO-/SCAN+ compound configurations generate persistent
  ridges, weaken the Azores High, suppress winds, alter heat flux and MLD,
  promote stratification — gives the specific circulation-index mechanism
  (NAO/SCAN/EAP) that a "remote circulation state" in H1 would concretely
  correspond to.

**Honest framing for presentation:** if H1 confirms, the contribution is
methodological validation — showing the model (trained with no explicit
NAO/EAP index) recovers a known atmospheric-bridge mechanism via XAI,
addressing the EGU reviewer critique that "IG shows association, not
causation." This is not a novel physical mechanism (NAO/EAP-driven NS MHW
is established in the cited literature) — it is evidence that the model's
learned redundancy has a real physical basis rather than being spurious.
If H0 (no overlap in wind/pressure), that would be the more novel finding,
pointing to an undocumented pathway.

_Status: pending IG maps from job 14200537 (ETA ~3h)._

---

## Open scientific questions

<!-- Things not settled yet. Remove entries when resolved. -->

**⚠️ SUPERSEDED — the entry below described a mixed-mean design that was
never actually shipped; see the corrected entry further down.**

~~Aug 19 2026 — quantile_tau combined loss: design risk, not yet run.~~
~~`src/models/cnn_lstm.py` now supports `quantile_tau`/`quantile_weight`:~~
~~`loss = quantile_weight·pinball(mean, y, tau) + (1-quantile_weight)·NLL(mean, y, var)`,~~
~~both terms applied to the *same* `mean` output~~
~~(`configs/partition/full_gnll_quantile/fold{0-4}.yaml`: tau=0.9, weight=0.7).~~
~~`train_partition.py` wiring verified (smoke test only, no real training run~~
~~yet — no `experiments/partition/*quantile*` output exists).~~

~~Risk: `mean` cannot simultaneously be E[Y|X] (what NLL assumes) and~~
~~Q_0.9(Y|X) (what pinball τ=0.9 targets) — pinball(τ=0.9) pushes the point~~
~~estimate up on *all* days, not just extreme ones, since a 0.9-quantile~~
~~regressor is biased high everywhere by construction.~~ [...] ~~A~~
~~safer alternative not yet tried: pinball as an auxiliary third output~~
~~head, not mixed into the same `mean` channel used by NLL.~~

**Corrected — Aug 19-20 2026 — the safer alternative WAS what shipped.**
The same commit that added this narrative entry (`f35cb6e`, Aug 19 15:39)
also implemented the "safer alternative" it proposed, but this doc wasn't
updated to say so until now: `CNNLSTMModel` has an independent
`quantile_head` (own `nn.Linear` params, no weight sharing with `self.fc`)
attached alongside the GNLL head, sharing only the backbone via
`_encode()`. `loss = NLL(mean, y, var) + quantile_weight · pinball(q_pred,
y, tau)`, where `q_pred` comes from the separate head — `mean`/`var` never
see a pinball gradient. Verified by direct `torch.autograd.grad` check
(disjoint-gradient smoke test, Aug 19 2026): backpropagating NLL alone
leaves `quantile_head`'s gradient `None`. `full_gnll_quantile/fold{0-4}.yaml`
uses tau=0.9, weight=0.3 (not 0.7 as in the superseded entry above).

**First real run — fold0, 3-epoch diagnostic (job 29403121, Raven, Aug 19-20
2026), not the full run**: `test_nll_loss=-0.1356`, `test_pinball_loss=0.0846`
→ combined loss ≈ `-0.1356 + 0.3×0.0846 ≈ -0.110` — pinball contributes ~19%
of |nll| even after weighting, neither term dominates. Test MAE=0.30°C,
Pearson r=0.724 already after 3 epochs (promising, not yet a converged
result). `val_loss` went 0.037 → -0.197 → -0.173 across the 3 epochs —
expected behavior, not a bug: `GaussianNLLLoss(reduction="mean",
full=False)` is `0.5·(log(var) + (y-mean)²/var)`, unbounded below 0 (unlike
MSE) whenever predicted `var` is small relative to the residual scale,
which is normal once the target is z-scored to std≈1. A negative loss is
only meaningful once checked against calibration (`mean(var_pred)` vs.
`mean((y_true-mean_pred)²)` on held-out data) — not yet done for this run
specifically; on the to-do list before citing r/MAE as final numbers.
Run: https://wandb.ai/hereon-ksn-expercursors/mhw-precursors/runs/z7jh2nqb.

**Compute cost (Raven `gpu1`, measured directly, job 29403121)**: ~5.4
min/epoch (589 batches/epoch @ ~1.83 it/s), `billing=124`/hour (`TRESBillingWeights:
CPU=1.0, GPU=108.0`, 16 cpu + 1 A100). The 3-epoch diagnostic cost ~42
billing-hours. A full 5-fold array (target for both `full_gnll_quantile`
and `full_gnll_focal`) is estimated at ~1700-2250 billing-hours each
(~30-40 epochs/fold to early-stop, unconfirmed until a real fold completes)
— no confirmed hard SLURM-side quota found for account `mmm_gpu`
(`sacctmgr show assoc` shows no `GrpTRESMins`/`MaxTRESMins`); any project-level
cap would be on the MPCDF portal, not visible from the CLI. Decision: launch
fold0 of each variant first to get a real per-fold cost before committing
all 10 folds (5×2 variants).

**Platform note**: this experiment (`full_gnll_quantile`,
`full_gnll_focal`) runs on **MPCDF Raven**, not JUWELS — ported Aug 19 2026
because this session had no JUWELS data/venv access. `configs/partition/
full_gnll_quantile/fold{0-4}.yaml` `data_dir`/`output_dir` point at Raven
paths (`/raven/u/cradin/data/merged_daily.nc`,
`/raven/u/cradin/exprecursors/experiments/...`), NOT the JUWELS paths used
by every other experiment in this doc — do not assume these two experiments'
paths are portable to JUWELS without editing back. `scripts/slurm/
submit_gnll_quantile_partition.sh`/`submit_gnll_focal_partition.sh` use
Raven's `gpu1` partition, account `mmm_gpu`, module `python-waterboa/2024.06`
— JUWELS-specific `--partition=booster`/`Stages/2025` module stack does
not apply to these two runs.

**Aug 20 2026 — focal-weighted NLL: third variant, alternative to the
quantile head.** User's proposal: instead of an auxiliary quantile-regression
head, reweight the per-sample `GaussianNLLLoss` toward exceedance days
(`truth > Hobday p90(DOY)`) — `loss_i × (1 + focal_alpha)` if extreme, else
`loss_i × 1`, weighted average. Statistical rationale (cleaner than the
quantile head for a paper): `mean` stays exactly `E[Y|X]` and `var` stays
exactly the conditional variance — nothing about what they're fit to
predict changes, only which samples get more weight. Implemented
opt-in (`focal_weight`/`focal_alpha`/`return_target_doy` in config) without
changing `LazyDataset.__getitem__`'s default 3-tuple return (would have
broken ~20 existing call sites) —
see `src/models/cnn_lstm.py::CNNLightningModule._focal_weighted_loss` and
`known_issues.md` #37-39 for gaps/gotchas found while wiring this up.
`configs/partition/full_gnll_focal/fold{0-4,0_shorttest}.yaml`,
`scripts/slurm/submit_gnll_focal_{shorttest,partition}.sh`.

**Diagnostic result (job 29403991, fold0, 3 epochs, Aug 20 2026)**:
`test_nll_loss_unweighted=-0.0519`, `test_nll_loss_weighted=-0.0549` — close,
weighting isn't distorting the loss scale. `test_frac_extreme=0.0356`
(3.56% of fold0's test-year samples flagged extreme) looked low against the
~10% textbook Hobday rate, so checked directly against the full record
before trusting it: overall 1985-2024 exceedance = 7.4%, reference period
1985-2014 = 6.2%, recent decade 2015-2024 = 11.2% (consistent with the
warming trend already documented elsewhere in this file) — mechanism
confirmed correct, fold0's 3.56% is just normal sampling variability from
an 8-year test subset, not a units/threshold bug. MAE=0.34°C, r=0.699 (vs.
quantile-head's MAE=0.30°C, r=0.724 at the same 3 epochs — not conclusive,
too early to compare variants on 3-epoch numbers).
Run: https://wandb.ai/hereon-ksn-expercursors/mhw-precursors/runs/l903ypre.

**Both 5-fold arrays launched Aug 20 2026** after their respective
diagnostics passed: `full_gnll_quantile` job 29404086, `full_gnll_focal`
job 29404257. Real per-fold cost (not yet known — see compute-cost note
above) will determine whether the ~1700-2250 billing-hour/variant estimate
holds.

---

## Ground-truth "MHW day" definition and areal-extent threshold (Aug 20 2026)

**The problem.** The model predicts a scalar (NS-box-mean `to_anom`, 7 days
ahead) — that doesn't change. But *evaluating* the model (recall on
"extreme"/MHW days) needs a ground-truth definition of "MHW day", and two
reasonable ones disagree: (1) basin-mean-first — average the NS box, then
apply Hobday's p90 exceedance to that single series (what the model's own
target is built from, self-consistent, but not the field's standard
convention); (2) per-pixel-first — apply Hobday p90 exceedance at every
grid cell independently, then require some fraction of the box's area to
be simultaneously exceeding (the field's actual convention — Hobday et al.
2016 defines MHW "at a single location"; regional/basin claims are
built by aggregating point-based classifications, not by classifying an
already-averaged series). Definition (1) is what the "15.2% recall"-style
numbers computed so far this project use. Both are computed and compared
here so downstream analysis isn't locked into a choice before checking
which one the paper should actually claim.

**Why definition (2) needs an areal-extent threshold, and why that choice
is not free.** "Some fraction of the box's area" requires picking a
percentage, and there is no universal field convention for it — different
studies calibrate against their own region's historical distribution.
Two structurally different ways to pick that percentage were considered:

- **Percentile of our own `area_frac(t)` distribution** (e.g. "the top
  10% of days by areal MHW coverage"). First choice (Aug 20 2026,
  morning): top10% = p90 of the empirical `area_frac` distribution over
  1985-2024 = area ≥ 40.5%. Justified at the time by consistency with
  Hobday's own use of the 90th-percentile-exceedance convention for the
  temperature threshold itself.
- **Absolute areal-extent threshold**, taken directly from published
  conventions rather than derived from our own data's shape. The user
  supplied a table of real literature values: MedECC (2023) uses **≥5%**
  of basin area as a default; Darmaraki et al. (2024) uses **10-20%** to
  isolate large-scale summer MHW events; NOAA's Blob-tracker convention
  uses **30-50%** to isolate basin-wide extreme events only (the "Blob"
  tier).

**Why the percentile choice was rejected.** Mapping the literature's
absolute thresholds onto our actual `area_frac(t)` series (14,600 days,
1985-2024, `experiments/figures/area_frac_timeseries.npy`) showed that
the "top10%" pick (area ≥ 40.5%) lands almost exactly in the NOAA
Blob-tracker's 30-50% "extreme events only" tier — not a moderate
default, as the p90-consistency argument made it sound. In days/year
terms: area≥5% → 135.1 days/yr; area≥10% → 107.6 days/yr; area≥20% →
74.4 days/yr; area≥30% → 51.0 days/yr; area≥50% → 24.5 days/yr; our
top10% pick (area≥40.5%) → 36.5 days/yr.

**Why 5% (MedECC) was chosen instead — external validation, not
aesthetics.** Paso 1/2's literature review (see known_issues.md #41)
already established, independently, that the North Sea averages **~140
MHW days/year** (Ocean Science 2025, per-pixel-aggregated convention).
MedECC's 5% threshold reproduces this almost exactly: **135.1 days/yr**
(3.5% off). None of the other candidates come close (107.6, 74.4, 51.0,
24.5, or our original 36.5 all undercount relative to that external
benchmark). This is a stronger justification than internal consistency
with Hobday's own percentile convention, because it's validated against
an independently-sourced number, not just self-referential.

**Decision (Aug 20 2026): `AREA_FRAC_THRESHOLD = 0.05`** (MedECC 2023
default) in `scripts/analysis/mhw_definition_agreement_and_recall.py`.
Confusion matrix between def1/def2 and recall of `full_gnll_quantile`/
`full_gnll_focal` under both definitions recomputed under this threshold
— results pending (job restarted Aug 20 2026 with per-fold `.npz` caching
so a future threshold change doesn't require re-running CPU inference for
all 10 folds again). Not treated as final: if reviewers push back on
using a Mediterranean-derived convention for the North Sea, revisit with
Darmaraki et al. (2024)'s North-Atlantic-specific 10-20% range as the
fallback — kept as the second candidate specifically because it's the
one grounded in the right ocean basin, not just any external number.

---

## Paso 4 — `full_gnll_focal_v2` fold0 GPU test launched (Aug 20 2026)

Order followed per the approved plan: pytest (16 passed/2 skipped) →
synthetic smoke test → `--fast_dev_run` against real data (job 29417018,
completed cleanly in 1:51 once run on a dedicated `small`-partition node
instead of the shared login node — the login node was badly contended,
load average 40-80+ with 178-200 concurrent users, which is why earlier
CPU-bound analysis jobs looked stalled; not a code bug) → 3 parallel
code-review agents covering (1) the `hobday_smooth_target` patch, (2) the
`stratified_kfold` split + its tests + `persistence_remote_sst.py`'s RNG
fix, (3) the cosine scheduler + `full_gnll_focal_v2` configs.

Reviews 1 and 2 found nothing. Review 3 found a real bug, fixed before
launch: the cosine scheduler's decay horizon (`T_max`) was implicitly
`max_epochs=1000` (an early-stopping ceiling, not a realistic training
length) — with `EarlyStopping(patience=30)` typically stopping training
around 150-200 epochs, LR would still be ~90% of peak at stop time,
silently defeating the point of switching away from `ReduceLROnPlateau`.
Fixed with a new `cosine_t_max_epochs` config key (`=60`, chosen as
roughly double the ~30-40 epoch early-stop range observed for the v1
`full_gnll_quantile`/`full_gnll_focal` runs, with margin for v2's
different LR/target-scale) — see `known_issues.md` #44 for the full
before/after verification (LR now reaches ~0 by epoch 60 instead of
staying near peak). `fold0_shorttest.yaml` also fixed (`warmup_epochs: 5`
→ `1`, `cosine_t_max_epochs: 3`) so its own 3-epoch smoke test actually
exercises the decay branch instead of staying in warmup throughout.

**`full_gnll_focal_v2` fold0 launched Aug 20 2026, job 29417248** (single
fold, not the full 5-fold array — Paso 5 gate: full array only after the
user confirms this fold0 run looks right). `--time=12:00:00`,
`gpu1`/`mmm_gpu`, `--mail-user=cristina.radin@uni-hamburg.de` (the SLURM
email-notification bug from `known_issues.md` #43 is now fixed, so
END/FAIL mail should actually arrive this time).

**Two more fold0-only comparison runs launched same day, same session, on
user request** (a 3-way fold0 comparison before committing any full
5-fold array — a reasonable front-loading of information ahead of the
plan's stricter Paso 5→6 sequential gate): `full_gnll_quantile_v2/fold0`
(job 29417405) and `full_mse_v3/fold0` (job 29417406, "v3" because
`full_mse_v2` already exists as an old, never-run-on-Raven JUWELS-path
config — kept untouched, not overwritten, to avoid confusion). Both get
the identical v2 treatment as focal (`stratified_kfold`,
`hobday_smooth_target`, `padding_mode: reflect`, `lr_scheduler: cosine`,
`warmup_epochs: 5`, `cosine_t_max_epochs: 60`, `learning_rate: 0.00005`)
— only the loss-specific flags differ (`quantile_head`/`quantile_tau=0.9`/
`quantile_weight=0.3` for quantile_v2; all three of `gaussian_nll`/
`quantile_head`/`focal_weight` false and `loss_fn: MSELoss` for mse_v3).
Both `--fast_dev_run`-tested clean first (job 29417321, `small` partition)
before the real GPU submission — quantile_v2's dry-run log correctly
shows `test_pinball_loss` (confirms the quantile head path is active),
mse_v3's shows neither `test_nll_loss` nor `test_pinball_loss` (confirms
the plain-MSE path, no GNLL/quantile machinery accidentally engaged).
Not re-run through the full 3-agent code review from focal_v2's launch —
these configs reuse the same scheduler/split/hobday code already cleared
by that review; only the new loss-specific flag combinations were novel,
and those were smoke-tested directly via the dry run instead.

---

## Results: Paso 2 recall (v1 checkpoints) + Paso 4 fold0 comparison (Aug 20 2026)

**Paso 2 — confusion matrix + recall under both ground-truth definitions,
job 29417019, `full_gnll_quantile`/`full_gnll_focal` v1 checkpoints
(`kfold`-mode, pre-existing 5-fold results, NOT the new v2 runs)**:

Confusion matrix (def1=basin-mean Hobday, def2=per-pixel+area≥5%, 40 yrs):
def1 total extreme days=1061, def2=5402. Every def1 day is also a def2
day (0% "only def1"). Simple agreement 70.3%, Jaccard 19.6%.

Recall (pooled across all 5 folds' test years):
| model | def1 (basin-mean) recall | def2 (pixel+area≥5%) recall |
|---|---|---|
| `full_gnll_quantile` | 19.0% (n=1084) | 5.3% (n=5402) |
| `full_gnll_focal` | 34.9% (n=1084) | 12.8% (n=5402) |

Focal beats quantile under **both** definitions (roughly 2-2.5x). Sobering
finding: recall drops sharply under def2 (the field-standard definition)
vs. def1 (the self-consistent one) for both models — the model's
basin-mean-exceedance signal fires much less reliably on the "real"
per-pixel+area MHW days than on the basin-mean-defined ones. Whichever
definition the paper commits to (see `known_issues.md` #41) materially
changes how good these numbers look. Figures: `experiments/figures/
mhw_definition_agreement.png`, `recall_both_definitions.png`.

**Paso 4 — fold0-only 3-way comparison, `full_gnll_focal_v2` (job
29417248) / `full_gnll_quantile_v2` (29417405) / `full_mse_v3`
(29417406)**, all completed clean (exit 0:0) within the same ~3-4h
window while unattended, all early-stopped well before both the 12h
SLURM limit and the `cosine_t_max_epochs=60` horizon (retrospectively
validates that choice — none needed anywhere near 60 epochs, but none
were cut off early either):

| variant | stopped at epoch | MAE (°C) | Pearson r |
|---|---|---|---|
| `full_gnll_focal_v2` | 35 | 0.303 | 0.815 |
| `full_gnll_quantile_v2` | 43 | 0.313 | 0.806 |
| `full_mse_v3` | 42 | 0.314 | 0.796 |

Focal_v2 leads on both MAE and r at fold0, consistent with the v1
focal-vs-quantile recall gap above. Not conclusive on a single fold —
no full 5-fold array launched yet for any v2 variant (Paso 5 gate, needs
explicit go-ahead). Extreme-day recall not yet computed for the v2
checkpoints under either ground-truth definition (only MAE/r come
straight out of `train_partition.py`'s test loop) — natural next step if
these fold0 results are judged good enough to justify the full arrays.

---

## Decisive check: quantile head vs mean head, and full array launch (Aug 20-21 2026)

**The comparison above was incomplete.** Every recall number computed so
far for `full_gnll_quantile` (v1 and the fold0 v2 comparison) used the
model's shared MEAN head (`model(xs, xt)`), never `model.
forward_with_quantile()`'s own `q_pred` — despite `quantile_tau=0.9`
being trained specifically to predict "will the target exceed its own
90th percentile", which is conceptually much closer to Hobday
exceedance than the mean ever is (user's observation). Checked directly
on `full_gnll_quantile` v1's 5 already-trained folds
(`scripts/analysis/quantile_head_recall.py`, job 29425720):

| | def1 (basin-mean) | def2 (pixel+area≥5%) |
|---|---|---|
| mean head recall | 19.0% | 5.3% |
| **quantile head recall** | **81.8%** | **44.7%** |
| mean head precision | 71.5% | 99.7% |
| **quantile head precision** | 33.8% | **91.8%** |
| quantile head FPR | 12.9% | 2.4% |

High recall alone would be cheap to fake by predicting systematically
high (tau=0.9 biases `q_pred` above the mean by construction) — checked
precision/FPR specifically to rule that out. Under def2, precision=91.8%
is far above the 37.2% base rate, and FPR is only 2.4% — real
discriminative signal, not spurious bias. Under def1 the trade-off is
harder (precision 33.8%, base rate 7.5% — still ~4.5x lift, but noisier)
but def2 was the pre-agreed decisive metric. **`full_gnll_quantile`'s
quantile head beats `full_gnll_focal`'s mean-head recall (12.8% def2) by
3.5x** — reverses the earlier focal-favoring read, which was comparing
focal's actual mechanism against quantile's *unused* mechanism.

**Decision (made under the user's pre-agreed autonomous-decision
framework, Aug 20 2026 evening — see project memory): commit to
`full_gnll_quantile_v2`, full 5-fold array.** fold0 already completed
clean (job 29417405, checkpoint `epoch=13, val_loss=-0.0245` confirmed
correct via `best_ckpt()` — matches the independently-computed true
per-epoch minimum, unaffected by the `trainer.test()` last-epoch bug,
which only affected that job's own printed summary metrics, not the
saved checkpoint files) — **not re-run**, would waste ~4h of GPU for an
identical deterministic-seeded config. Folds 1-4 launched as an array,
job 29426208, `gpu1`/`mmm_gpu`, Aug 20-21 2026.

**Important methodological note for all downstream analysis (XAI,
persistence, spatial map, final paper numbers)**: this model must be
evaluated via `model.forward_with_quantile()`'s `q_pred`, never the bare
mean — the mean head alone badly underperforms (see table above). Any
script/notebook touching `full_gnll_quantile_v2` checkpoints needs to
call `forward_with_quantile()`, matching `quantile_head_recall.py`'s
pattern, not `_adhoc_eval_extreme_recall.py`'s (which never does).
