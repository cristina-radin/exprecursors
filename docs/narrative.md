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


---

## Paso 5: full array completed, per-fold loss curves checked (Aug 21 2026)

`full_gnll_quantile_v2` array job 29426208 (folds 1-4) completed clean
(exit 0:0 all four, ~3.5-4h each). Checked each fold's real per-epoch
`val_loss` trajectory (not just final numbers) per the plan's explicit
requirement to look at loss curves, not just final metrics:

| fold | best epoch | best val_loss | stopped epoch | gap (patience=30) |
|---|---|---|---|---|
| 0 | 5 | -0.0245 | 35 | 30 |
| 1 | 9 | -0.119 | 39 | 30 |
| 2 | 11 | -0.254 | 41 | 30 |
| 3 | 6 | -0.231 | 36 | 30 |
| 4 | 11 | -0.203 | 41 | 30 |

Same pattern in every fold as already characterized for fold0 (see the
Aug 20 2026 training-stability entry): true best lands early (epoch
5-11), `EarlyStopping(patience=30)` triggers exactly 30 epochs later in
all 5 cases (mechanically correct, not a bug), substantial post-best
oscillation in every fold (final `val_loss` 1.2-2.1, vs. best -0.03 to
-0.25) — consistent GNLL variance-collapse behavior across the whole
array, not a fold0-specific fluke. No crashes, no NaN, no fold behaving
qualitatively differently from the others. `ckpt_path="best"` fix means
the actual checkpoint used downstream is unaffected by this late-training
noise regardless.

Decisive pooled recall/precision/FPR analysis (all 5 folds,
`forward_with_quantile()`, job 29430333) — results pending, see next
entry once it completes.

**DECISIVE result (job 29430333, all 5 folds pooled, `forward_with_quantile()`):**

| | def1 (basin-mean) | def2 (pixel+area>=5%, decisive) |
|---|---|---|
| mean head recall | 32.1% | 10.0% |
| **quantile head recall** | **80.7%** | **48.1%** |
| mean head precision | 59.8% | 95.8% |
| **quantile head precision** | 27.8% | **84.9%** |
| quantile head FPR | 16.4% | 5.0% |

**Clears the pre-agreed bar decisively**: def2 recall 48.1% vs v1
`full_gnll_focal`'s mean-head 12.8% baseline (3.75x). Precision (84.9%)
far above the 37.2% base rate confirms genuine signal, not systematic
bias -- consistent with v1's check (91.8%) and the v2-fold0-only early
read (57.5%/87.0%), so this holds up pooled across all 5 folds, not just
a lucky single fold. Sanity check (`q_pred` mean > mean_pred mean, tau=0.9)
holds in every one of the 5 folds individually.

**Paso 5 closed. `full_gnll_quantile_v2` is the committed model.**
Per the user's pre-agreed framework: proceeding to Paso 7 (persistence
baseline, then XAI adaptation if time allows) without launching the other
two experiment variants (MSE, focal) -- not worth the GPU-hours given the
time constraint, and the quantile-head result already clearly beats both.

**Update Aug 21 2026 — def2 had a reference-mismatch bug (known_issues.md
#56.1), corrected numbers below, conclusion unchanged.**
`quantile_head_recall_v2_all5.py`'s def2 used `area_frac_timeseries.npy`
computed from unsmoothed `to_anom`/`p90_thresh`, while def1 uses the
smoothed `load_ns_p90()` threshold in the same table -- fixed (both now
smoothed consistently), job 29450802 rerun with the corrected
`area_frac`:

| | def1 (basin-mean) | def2 (pixel+area>=5%, decisive) |
|---|---|---|
| mean head recall | 32.1% | 10.3% |
| **quantile head recall** | **80.7%** | **48.7%** |
| mean head precision | 59.8% | 96.5% |
| **quantile head precision** | 27.8% | **84.3%** |
| quantile head FPR | 16.4% | 5.2% |

Changes are tiny (recall 48.1%->48.7%, precision 84.9%->84.3%) despite
423/14600 days (2.9%) flipping sides of the area-fraction threshold --
**the decisive conclusion is robust to this bug**, still ~3.8x the v1
baseline (12.8%) with precision far above the base rate. def1 is exactly
unchanged (expected -- def1 never touched `area_frac`).


---

## Paso 7: persistence baseline gate (Aug 21 2026)

`scripts/persistence_remote_sst.py` repurposed for `full_gnll_quantile_v2`
(bug found and fixed first: script was missing `sys.path.insert()`,
crashed with `ModuleNotFoundError` on the very first run -- unrelated to
the Paso 3 RNG fix, a separate, simpler bug, never actually exercised
end-to-end on Raven before). Added `get_test_years_stratified()` (exact
replica of `datamodule.py`'s branch) since the old `get_test_years()`
only matched the legacy `kfold` split, not the committed model's
`stratified_kfold`. Also extended to compute exceedance-detection
recall/precision for persistence under both ground-truth definitions
(previously the script only computed Pearson r baselines), so it's a
fair apples-to-apples comparison against how the model itself is
evaluated (job 29430897).

**Point-forecast r — persistence wins, an honest caveat, not a
problem**: NS lag-7 local persistence r=0.9309±0.0166 vs. the committed
model's mean head r≈0.87. The model's raw point prediction does NOT beat
trivial "assume it stays the same for 7 days" on correlation -- expected
given this project's own earlier finding that short-lag local SST is
dominated by thermal inertia (see "Architecture choices" /"Partition"
sections above: Local > Full for the same reason). This should be
reported plainly: the paper's contribution is not "better SST point
forecasting", it's "better MHW precursor/exceedance detection" --
specifically via the quantile head, not the mean.

**Exceedance detection — the real gate, and the model passes it clearly
under the decisive metric**:

| | def1 (basin-mean) | def2 (pixel+area>=5%, decisive) |
|---|---|---|
| persistence recall | 54.6% | 16.5% |
| **model (quantile head) recall** | **80.7%** | **48.1%** |
| persistence precision | 65.2% | **98.1%** |
| model (quantile head) precision | 27.8% | 84.9% |

Under def2: model recall is ~3x persistence's (48.1% vs 16.5%) at a
still-high precision (84.9% vs persistence's near-perfect but overly
conservative 98.1%) -- persistence rarely cries wolf but misses most real
widespread events; the model catches far more of them at a modest,
acceptable precision cost. This is the intended trade-off for an
early-warning/precursor framing, where missing events is costlier than
extra false alarms. Under def1 the trade-off is less favorable (much
lower precision) -- consistent with everything else already observed
about def1 vs def2, and exactly why def2 was agreed as the decisive
metric rather than def1.

**Paso 7 persistence gate: PASSED (def2).** Model demonstrably adds
precursor-detection value beyond trivial persistence, even though it
does not beat persistence on raw point-forecast r -- these are different
claims and the paper should keep them separate, not conflate "r" with
"detects MHW precursors".


---

## Loss-curve defensibility for the paper: literature grounding (Aug 21 2026)

User's concern: the val_loss curve for `full_gnll_quantile_v2` (and every
GNLL-based variant this session) rises substantially after an early best
epoch (see Aug 20 2026 training-stability entry: best at epoch 5-13, then
75-100x swings) -- "eso es poco defendible" without a citable reason.

**This is a known, published phenomenon, not something we're guessing
at**: Seitzer, Tavakoli, Antic & Martius, ["On the Pitfalls of
Heteroscedastic Uncertainty Estimation with Probabilistic Neural
Networks"](https://arxiv.org/abs/2203.09168) (ICLR 2022). The paper
identifies exactly this mechanism: naively minimizing Gaussian NLL by
jointly learning mean and variance lets the variance-gradient dominate
the mean-gradient once predicted variance shrinks, producing
overconfident (small) variance estimates that destabilize training --
matching what we measured directly (val_loss, which is dominated by the
`-log(var)` term, swings 75-100x after the true best epoch while MSE,
with no variance term, only swings ~3x under the identical schedule/data).
Their proposed fix (β-NLL, a stop-gradient reweighting by variance^β) is
a legitimate follow-up to try later, not applied here -- we instead rely
on `ckpt_path="best"` (known_issues.md #46) to select the pre-collapse
checkpoint, which the recall/precision/persistence gates (Aug 21 2026
entries) confirm is a genuinely good, non-spurious model despite the
noisy curve after that point.

**LR diagnostic launched same day** (`full_gnll_quantile_v2_lr2e5`,
fold0, job 29433645) -- does halving peak LR (5e-5 -> 2e-5) produce a
visually cleaner curve without sacrificing the recall/precision that
already passed the gates? Result pending.

**For the paper**: cite Seitzer et al. 2022 when presenting the loss
curve; frame it as "known GNLL variance-gradient-dominance pathology,
mitigated via best-checkpoint selection, not a training failure" --
defensible with a citation, not just an empirical observation.


---

## XAI: IG on full_gnll_quantile_v2 (Aug 21 2026)

`scripts/ig_partition_quantile.py` (new, replaces `ig_simple.py` --
known_issues.md #49/#50) produced a real result: Integrated Gradients
for BOTH the mean head and the quantile head (q_pred), 5 variables each,
fold0, n=300 test samples, n_steps=50, job 29433738, 27 min wall clock
(matches the pre-launch estimate of 20-30 min once the memory/cudnn
bugs found on the first two attempts were fixed -- see known_issues.md
#50 for the full account: cuDNN LSTM backward needs
`torch.backends.cudnn.enabled=False` in eval mode, and naive interpolation
batching OOM'd a 40GB A100 because n_steps multiplies through
window_size in the CNN encoder, not just the nominal batch).

Outputs: `experiments/figures/ig_quantile_v2_fold0/ig_{mean_head,
quantile_head}_{ptho_bot,u10,v10,msl,ssr}.png` + `.npy` arrays. Also
generated `experiments/figures/prediction_vs_target_2014.png` (true
target vs. mean head vs. quantile head vs. Hobday p90 threshold, 2014 --
the most active MHW year, in fold0's test set) and saved the full
test-set predictions to `experiments/figures/
test_predictions_quantile_v2_fold0.npz` for reuse (no old-convention
`test_predictions.npz`/`figures/` repo-root pipeline exists for this
model family -- `train_partition.py` never wrote that file; the
Raven-ported pipeline never populated the legacy `figures/` dir either).

**Interpretation (Aug 21 2026, fresh session, no GPU used -- numpy
analysis of the already-saved `.npy` arrays + visual read of the PNGs):**

**Caveat, flagged by the user, not yet addressed: everything below is
fold0 only** (n=300 test samples from fold0's own test-year split). All 5
folds passed the same recall/precision gate (`known_issues.md`/
`narrative.md`'s Paso 5 entry), so there's no reason to expect a
categorically different picture in other folds, but that's an assumption,
not checked -- variable ranking, sign structure, NS-box enrichment, and
especially the coastal-artifact finding below should be treated as
"true for fold0, plausibly general" until at least 1-2 more folds are run
and compared. Deliberately not done in this pass (see the mean-vs-quantile
"near-identical maps" finding -- diagnosed as not worth GPU time on its
own; a multi-fold check is a separate, still-open question, lower priority
than the differential-IG run this session actually spent GPU time on).

**Variable importance (share of total domain-summed |IG|, identical rank
order for both heads):**

| variable   | mean_head | quantile_head |
|------------|----------:|---------------:|
| `ssr`      |     29.6% |          28.5% |
| `ptho_bot` |     22.0% |          21.5% |
| `u10`      |     17.1% |          17.7% |
| `v10`      |     16.6% |          16.8% |
| `msl`      |     14.6% |          15.4% |

`ssr` and `ptho_bot` together carry ~50% of total attribution, roughly as
much as the three wind/pressure variables combined. Inputs are all
normalized to comparable scale (`normalize: true`, per-variable train-set
mean/std, config `fold0.yaml`), so these shares are a fair
magnitude-for-magnitude comparison, not an artifact of differing input
units.

**Why is there IG signal over land? Checked directly, not assumed (Aug 21
2026) — answer differs by variable, no bug found.** `configs/partition/
full_gnll_quantile_v2/fold0.yaml` sets `ocean_variables: [ptho_bot]` only
— per `dataset.py`'s `__getitem__` (lines 362-365, 384-387), `ptho_bot` is
NaN-masked over land (via the correct, already-fixed `land_mask_tbottom`,
known_issues.md #2) then `nan_to_num(nan=0.0)`'d in *normalized* space,
i.e. land pixels get exactly the same value (0) as IG's own zero baseline
— so `diff = x - baseline = 0` there identically, and IG must be exactly
0 on land for `ptho_bot` regardless of gradient. **Verified numerically**:
`ptho_bot`'s IG map is **exactly 0.0** at all 9473 `land_mask_tbottom`
land pixels (`np.abs(land_vals).max() == 0.0`, computed directly from
`ig_mean_head.npy`) — no bug, works exactly as designed. `u10`, `v10`,
`msl`, `ssr` are **never masked at all** (not in `ocean_variables`) —
ERA5 wind/pressure/radiation fields are physically defined over land, so
real, non-artifact signal there is expected. Confirmed their land-pixel
IG magnitude is the same order as ocean-pixel IG (land/ocean mean-|IG|
ratio: `u10` 0.43, `v10` 0.50, `ssr` 0.92, `msl` **1.47 — land signal is
actually larger**, consistent with the observed positive `msl` band sitting
over France/Bay-of-Biscay, i.e. a real synoptic pressure-system footprint
extending onto land, not noise).

**User pushback on `msl` specifically (justified, checked directly, Aug 21
2026): "mean sea level pressure" over land sounds wrong at first read.**
Not a bug, but the first-pass explanation (above) only justified it from
the masking *code*, not the *data itself* — went back and checked the
raw values in `merged_daily.nc`, no metadata to lean on (`msl.attrs = {}`,
completely empty). Two checks: (1) land-pixel `msl` is **not** a fill
value or a broadcast constant — same-day spatial std across 10045 land
pixels = 702 Pa, and four spot-checked land points on one arbitrary day
show a coherent synoptic pattern, not noise (`France (47N,8E)` +436 Pa,
`Spain (40N,-4E)` −69 Pa, `Germany (52N,10E)` +714 Pa, `Poland (52N,20E)`
−118 Pa — a real high/low pressure gradient across the continent). (2)
Raw `msl` values center near 0 (mean 0.04-15.9 Pa, not ~101325 Pa) because
per `known_issues.md` #28, `merged_daily.nc` stores `msl` (and `u10`,
`v10`, `ssr`) as DOY-climatology anomalies already, computed during
preprocessing (`±5-day window, ref 1985-2014`), not absolute pressure —
expected, not an artifact. Physical justification for land coverage at
all: ERA5 "mean sea level pressure" is not literally pressure measured at
sea — it is surface pressure reduced to sea level via the barometric
formula (using local temperature/lapse-rate), computed at every grid
point including land, which is exactly why synoptic weather charts show
isobars over continents. Standard meteorological convention, not a
dataset quirk. **Conclusion stands: land-area signal is only a concern
for `ptho_bot` (confirmed absent), and legitimate physical signal,
verified against the raw values (not just the masking code), for all 4
atmospheric variables — nothing to fix here.**

**Per-variable spatial pattern (domain: lat 0-70N, lon -80..20E; NS/local
box = lat 50-63N, lon -5..13E, per `src/data/masking.py`):**
- `ssr` (dominant): strong, spatially coherent positive band tracking the
  Gulf Stream separation / Grand Banks region (lat 35-50N, lon -75..-60),
  and a broad negative band across the tropical Atlantic (lat 0-15N).
  Near-zero over the NS box itself -- this variable's signal is almost
  entirely remote.
- `ptho_bot`: sharp, high-magnitude structure concentrated in a thin rim
  hugging every coastline (Grand Banks/Nova Scotia shelf lat 42-48N lon
  -70..-55, and the whole North Sea/British-Isles/Scandinavia coastal
  margin) -- **quantitatively confirmed to be predominantly a
  coastal-masking edge artifact, not primarily a physical shelf-break
  signal, see the dedicated check below.** Quantified NS-box concentration
  (naive, all grid cells): **~29% of this variable's total |IG|** despite
  being **3.5% of the domain's grid cells** (~8.5x enrichment) -- by far
  the most locally-concentrated of the 5 variables. Some of this survives
  an open-water-only control (see below), so "locally important" is not
  entirely an artifact, but the sharp coastal *rim* features specifically
  (what the raw maps visually emphasize) mostly are.
- `u10`/`v10`: much flatter spatial distribution (NS-box enrichment only
  ~1.9-2.0x, vs. `ptho_bot`'s 8.5x) -- `v10` is broadly positive across nearly
  the whole mid-latitude Atlantic (0-60N) with a continuous band from the
  Grand Banks through to the NS box; `u10` is broadly negative over the
  subtropical gyre (20-40N) and again over the NS/British-Isles region
  specifically.
- `msl`: intermediate concentration (~3.1x NS-box enrichment) -- mostly
  negative over the northern half of the domain (lat >45N) with a distinct
  positive band just south of the NS box (~48-53N, France/Bay-of-Biscay
  latitude).

**Sign structure (fraction of total |IG| mass that is positive vs.
negative, mean_head; quantile_head nearly identical, both verified
directly from the saved `.npy`, Aug 21 2026):**

| variable   | % positive | % negative |
|------------|-----------:|-----------:|
| `v10`      |      83.7% |      16.3% |
| `ptho_bot` |      35.9% |      64.1% |
| `u10`      |      11.7% |      88.3% |
| `ssr`      |       5.2% |      94.8% |
| `msl`      |       4.5% |      95.5% |

`v10` is massively one-signed positive, `u10`/`msl`/`ssr` are massively
one-signed negative, and `ptho_bot` is the only genuinely mixed-sign
variable (consistent with its coastal sign-alternating structure --
though see below, part of that alternation is now suspected to be the
edge artifact, not a physical dipole). Reading: southerly wind (`v10`>0
in this domain's convention) is a uniformly warming/risk-raising signal,
while zonal wind, pressure, and radiation act predominantly as
cooling/risk-lowering signals in the direction IG attributes them --
`ptho_bot` alone contributes substantial signal in both directions,
locally.

**Coastal signal in `ptho_bot`: is it real or an artifact? Checked
quantitatively, Aug 21 2026 (user's explicit concern, right to be
skeptical) -- verdict: mostly artifact, with a smaller but real
non-artifact component underneath.** `ptho_bot` is the only variable of
the 5 that is land-masked (`nan_to_num(nan=0.0)` on land, in normalized
space, matching IG's own zero baseline -- see the land-signal check
above), which structurally creates an artificial ocean/zero-fill edge at
every coastline; `u10`/`v10`/`msl`/`ssr` are never masked and have no such
edge. Tested directly: binned mean |IG| by distance-to-nearest-land (px,
via `land_mask_tbottom`, `scipy.ndimage.distance_transform_edt`):

| distance to coast | `ptho_bot` mean\|IG\| | `u10` mean\|IG\| (control) | `ssr` mean\|IG\| (control) |
|---|---:|---:|---:|
| 1-2 px | 3.76e-6 | 5.69e-7 | 8.80e-7 |
| 2-3 px | 3.80e-6 | 6.05e-7 | 8.98e-7 |
| 3-5 px | 2.44e-6 | 6.34e-7 | 1.11e-6 |
| 5-9 px | 0.93e-6 | 6.64e-7 | 1.30e-6 |
| >9 px (open ocean) | 0.24e-6 | 9.27e-7 | 1.13e-6 |

`ptho_bot` decays **~15x** from the coast to open ocean, smoothly and
monotonically -- the textbook signature of a boundary/edge artifact (the
CNN's receptive field reacting to the real-ocean-value-next-to-zero-fill
discontinuity, the same underlying mechanism `known_issues.md` #2
originally described, except that bug was about a *misaligned* mask
falsely zeroing real ocean pixels, already fixed; this is a *structural*
edge effect present even with the correct mask, not previously checked).
The two unmasked control variables (`u10`, `ssr`) show **no such decay**
-- flat or even slightly *increasing* with distance from land, exactly
what's expected with no artificial edge. This is new, not previously in
`known_issues.md` -- added as #52 there with the full account and
practical guidance (don't over-read the coastal rim, e.g. the Grand
Banks dipole, as bathymetric/shelf-break physics).

**But the NS-box result is not entirely explained by this artifact**:
repeating the NS-box-enrichment calculation restricted to genuinely
open-water pixels (>5px / ~2.5° from any coast, n=177 NS-box pixels,
n=12889 domain-wide) still gives **16.8% of open-water |IG| in the NS
box vs. 1.4% of open-water pixel count -- 12.2x enrichment**, actually
*higher* than the naive 8.5x figure, not lower. So `ptho_bot`'s broad
local importance in the NS box is not simply "NS box = disproportionately
coastal" -- there is a real signal in the open-water interior too. What
should NOT be trusted without more work: the specific sharp coastal-rim
*features* the raw maps visually emphasize (the Grand Banks dipole, the
thin ring around the North Sea/British Isles) -- those are dominated by
the edge artifact per the distance-decay table above.

**Independent cross-check with occlusion (Aug 21 2026, job 29441983,
n=300, committed fold0 checkpoint) — partial corroboration, partial
contradiction.** Occlusion (zero out `ptho_bot`, measure the actual
forward-pass |output delta|, no gradient involved) confirms the coastal
decay is real and not an IG-specific gradient artifact: 1-2px vs >9px
ratio is 7.29x (mean head) / 7.73x (quantile head) — same monotonic
shape as IG, about half IG's ~15.7x magnitude. But it does **not**
reproduce the "real open-water signal survives" claim above: the same
NS-box-restricted-to-open-water calculation that gave IG's 12.2x
enrichment gives occlusion **0.71x for both heads** — i.e. no special
NS-box importance, if anything marginally under-represented relative to
its pixel share. This directly contradicts the 12.2x figure. Not
resolved — do not cite the 12.2x NS-box open-water enrichment as
confirmed by an independent method; treat `ptho_bot`'s open-water NS-box
importance as method-dependent and unsettled pending further work (e.g.
Shapley or GradCAM as a third method, or investigating why IG and
occlusion diverge specifically in open water). See `known_issues.md`
#52 and `results/all_results.csv` for the full numbers.

Plotting fix applied (Aug 21 2026): `ig_partition_quantile.py` now greys
out land (`ax.set_facecolor("lightgray")`, data set to NaN) for variables
in the config's `ocean_variables` (`ptho_bot`) only, using the correct
`land_mask_tbottom`; atmospheric variables are plotted unmasked, since
their land-area values are real. All 10 existing fold0 PNGs regenerated
from the already-saved `.npy` arrays (no GPU touched) with this fix --
the greyed-out `ptho_bot` maps make the coastal-rim-vs-open-water
distinction visually obvious in a way the un-greyed maps did not.

**Relation to the H1 atmospheric-bridge hypothesis (see "Hypothesis:
mechanism behind local/remote redundancy" above):** the wind/pressure
variables (`u10`, `v10`, `msl`) do show attribution spanning continuously
from the remote Gulf-Stream/subtropical region into the local NS box,
qualitatively consistent with H1's predicted atmospheric bridge -- but
this run only used the `full_gnll_quantile_v2` partition (no separate
Remote-only/Local-only quantile checkpoints exist), so this is a
suggestive observation, NOT the falsification test H1 actually specifies
(that requires comparing Full vs. Remote-only vs. Local-only IG maps for
the same model family, still undone for the quantile head). Do not cite
this as H1 confirmed.

**Mean head vs. quantile head: nearly indistinguishable at this level of
aggregation.** Spatial correlation between the two heads' maps is
0.998-0.999 for every variable, and visually the rendered PNGs are
indistinguishable. This is a real, architecturally-explained result, not
an analysis gap -- full quantitative account and the proposed sharper
follow-up (differential IG on `q_pred - y_hat_mean`) in `known_issues.md`
#51. Practical upshot: this round of IG answers "which inputs drive a
high predicted temperature/risk" in general, but does not yet show what
*specifically* pushes the quantile head's exceedance signal beyond the
mean forecast -- that would need the differential approach, not more
folds/samples of the same per-head maps.

**SUPERSEDED, Aug 21 2026 result (job 29435465) -- computed with the same
sampling bug that invalidated other numbers in this doc.** The original
differential-IG run used `committed` (not `nearest`) AND the pre-fix
`test_indices[:max_samples]` sampling (known_issues.md #57 -- chronological
slicing, 299/300 samples from a single year), discovered only Aug 24 2026
while preparing a proper rerun. Its specific numbers (msl/u10 "co-dominant"
at 24.3% each, "concentrated north of 45N, most intense over the North
Sea/British Isles") should NOT be cited -- kept below, struck through in
spirit, superseded by the corrected rerun immediately after.

**Corrected rerun, Aug 24 2026 (jobs 29565673/74/75, `nearest` model,
stratified sampling, folds 0-2, n=~300, n_steps=50, ~14 min/fold):**

**GPU spend was unnecessary, found immediately after by the user asking
"isn't differential just a head comparison?"** IG is linear (the
integral of a gradient of a difference is the difference of the
integrals), so `IG(q_pred - mean)` is exactly `IG(q_pred) - IG(mean)`,
sample for sample -- confirmed numerically, max abs difference between
the directly-computed diff map and `ig_quantile_head.npy -
ig_mean_head.npy` (both already on disk for `nearest` fold0/1/2, computed
Aug 22 with the correct sampling) is 9.4e-12 (float rounding only). The 3
GPU jobs above (~42 min total) reproduced something obtainable instantly
from files that already existed -- a plain `np.load(quantile) -
np.load(mean)` on the existing `ig_quantile_v2_landfill_fold0`/
`ig_quantile_v2_nearest_fold{1,2}` outputs gives the identical result.
Not wrong (the launched-run numbers below are correct), just avoidable --
next time a `--heads diff` run is wanted and same-sampling per-head maps
already exist for the target model/folds, subtract them instead of
re-running IG.

Variable importance for the differential signal (mean±std over 3 folds):

| variable | diff-head share | 3-fold std | note |
|---|---:|---:|---|
| `u10` | **27.0%** | 6.1% | top-2 in every fold, sign extremely stable |
| `ptho_bot` | 21.8% | 4.5% | consistently top-3, but sign is NOT stable (see below) |
| `msl` | 19.8% | 11.0% | unstable -- #1 in fold0 (35.3%), #4-5 in folds 1-2 (13.0%, 11.1%) |
| `ssr` | 19.5% | 7.4% | unstable -- #1 in fold1 (29.8%), #3-4 elsewhere |
| `v10` | 11.8% | 1.2% | smallest magnitude, but most stable of all 5 |

`u10` is the only variable that is both consistently high-magnitude AND
low-variance across folds -- `msl` and `ssr` swap which one looks
"important" fold to fold (each is #1 in exactly one of the 3 folds), so
neither should be cited alone as "the" secondary driver; `u10` is the one
number here solid enough to lead with.

**Sign structure (3-fold mean fraction positive)**: `u10` 3.7% ± 1.3%
positive -- i.e. **96%+ negative in every single fold**, the tightest,
most reproducible sign signal of the whole differential analysis.
`ptho_bot` 54.2% ± 11.5% (no consistent direction -- roughly a coin flip,
and it varies by fold), `ssr` 69.2% ± 28.4% and `v10` 42.1% ± 21.7% (both
too unstable to characterize by sign), `msl` 33.0% ± 24.7% (also
unstable). Only `u10`'s sign is a fixed, citable fact.

**Spatial pattern, corrected -- NOT a simple "everything north of 45N"
band (that read the fold0/committed/biased map, which does look that
simple; the corrected 3-fold maps show a sharper structure).** Visually
consistent across all 3 folds (`ig_diff_nearest_fold{0,1,2}/
ig_diff_head_u10.png`): a broad, coherent NEGATIVE band spanning almost
the entire mid-latitude North Atlantic (roughly lat 20-60N, the full
basin width), contrasted with a distinct, consistently POSITIVE patch
localized specifically over the North Sea/Denmark/Baltic corner (roughly
lat 50-65N, lon 0-20E) -- a real dipole, not a uniform band. The naive
NS-box-average metric used elsewhere in this doc actually masks this
(the box straddles the blue/red boundary and partially cancels): u10's
diff-signal "NS-box enrichment" computes to 0.66x ± 0.17x (i.e. BELOW
domain average) despite the box visibly containing the one region that's
oppositely-signed from everywhere else -- a good demonstration of why a
single box-average number can hide a real, visually-obvious pattern, and
why this section leads with the images, not just the ratio. A broader
north/south (>45N vs <45N) split is also NOT a stable descriptor (ratio
1.05-1.06x in folds 0/2, 2.82x in fold1) -- the dipole's north-positive
patch is spatially tighter than "everything above 45N".

Reading, with the sign caveat already noted (u10 is the signed zonal
component, not wind speed): across most of the basin, an anomalously
easterly/low u10 is what specifically pushes the quantile head's risk
estimate above the point forecast, beyond what it already contributes to
the mean forecast -- but for the North Sea's own immediate corner
specifically, the same variable contributes in the opposite direction.
This is a sharper, more specific candidate mechanism than the original
"north of 45N" framing, but still not physically confirmed (same caveat
as before -- would need wind-speed magnitude/divergence, not the raw
signed component, to test directly against the H1 literature's "weak
wind" mechanism).

**Lesson repeated a third time in this project**: always check which
sampling/model version produced a number before citing it -- this is the
same class of correction as the NS-box-enrichment "12.2x -> 1.39x" and the
Grand-Banks-is-artifact findings earlier in this doc, now demonstrated on
the differential-IG result too.

**MHW-day-conditioned differential IG, Aug 24 2026 (jobs 29566587/88/89,
`nearest` model, folds 0-2, `--stratify_mhw` -- new flag on
`ig_partition_quantile.py`, splits the population average by def2 ground
truth instead of pooling everything).** User's direct question: the
unconditional diff map above answers "what matters on an average day",
not "what matters specifically on MHW days" -- this run answers the
actual question. Zero extra IG cost vs. the pooled run (same samples/
n_steps, just accumulated into 2 buckets by outcome instead of 1).
Sample sizes: fold0 121 MHW/176 non-MHW, fold1 100/200, fold2 109/191
(base rate ~37-41%, consistent with the project's established ~37.2%).

**Pooled (whole-domain) picture barely changes between MHW and non-MHW
days**: `u10` still dominates and is still overwhelmingly negative in
both conditions (mhw: 26.5% share, 93.5% negative; non-mhw: 26.1% share,
97.1% negative) -- if you only look at the domain-wide average, MHW-day
and ordinary-day differential attribution look almost the same.

**But inside the North Sea box specifically, `u10`'s sign flips
completely depending on MHW status -- robust across all 3 folds, the
single cleanest MHW-specific signal in this whole investigation**
(NS-box mean SIGNED u10 diff, not absolute):

| fold | MHW days | non-MHW days |
|---|---:|---:|
| 0 | +2.9e-8 | −2.5e-8 |
| 1 | +3.0e-8 | −6.3e-8 |
| 2 | +4.6e-8 | −2.3e-8 |

Positive in all 3 folds on real MHW days, negative in all 3 folds on
ordinary days -- zero exceptions. Visually confirmed
(`ig_diff_mhw_stratified_fold0/ig_diff_head_{mhw,nonmhw}_u10.png`): the
North Sea/Baltic positive patch (the dipole's "opposite sign" component
described above) is strong and clearly visible on MHW days, and nearly
vanishes on non-MHW days -- the broad basin-wide negative band is present
in both and doesn't visibly change. So the dipole structure itself is not
uniformly "always there" -- its local (North Sea) component is
specifically an MHW-day phenomenon; its remote (basin-wide) component is
present regardless of outcome.

Reading (same signed-component caveat as always -- `u10` is zonal wind,
not speed): on days that go on to become a real MHW event, local u10 over
the North Sea itself pushes the quantile head's risk estimate above what
it already contributes to the mean forecast; on an ordinary day, the same
local wind pushes the other way. This is the most specific, most directly
MHW-relevant finding produced by the differential-IG line of
investigation -- the remote/basin-wide `u10` signal is present regardless
of outcome and should not be over-read as MHW-specific, but the local
North Sea sign-flip is.


---

## Case study: model lags at trend inflections, tracks sustained trends (Aug 21 2026)

User's observation looking at `prediction_vs_target_2014.png`: both heads
undershoot around doy~150 and (initially misidentified, then corrected
to) doy~202/220. Checked with exact numbers from
`test_predictions_quantile_v2_fold0.npz` (fold0, 2014), not by eye:

**doy~150-160 event (true MHW, target rises to 1.67°C, threshold~1.18-1.22°C)**:
input window (doy147-153, true=1.22-1.28) is nearly flat -- no rising
trend visible yet in the 7-day-lagged input. Both heads predict
flat/declining (mean 0.82->0.63, q_pred 1.15->0.98) while the true value
keeps accelerating. Not an inconsistent response given the input -- the
model had no signal yet that the event would keep intensifying.

**doy~202-210 event (true MHW, target rises to 2.08°C)**: input window
(doy185-196, true=0.81->1.21) shows a clear, sustained rising trend
already. Both heads respond appropriately, rising substantially in step
(mean 0.68->0.86, q_pred 1.05->1.22 over the same window) -- good
tracking when the precursor trend is actually present in the input.

**doy~218-232 (same event's decline)**: predictions start dropping
doy218-220 (mean 1.35->0.96, q_pred 1.77->1.40) *while true is still near
its peak* (1.99-2.02) -- anticipates the turn slightly early. But then
lags on the way down: once true crashes (1.69->0.34, doy222-232),
predictions stay elevated longer (q_pred still ~1.4-1.5 at doy229 when
true=0.76) -- consistent with lead=7 using input that was still elevated
at doy222 (true=1.69).

**Pattern, stated precisely**: the model (both heads) is fundamentally
trend-following -- tracks sustained, already-visible trends well, lags
at sharp inflections (both accelerations and decelerations) because the
7-day-lagged input window simply doesn't contain the turning-point
information yet. Directly consistent with the Paso 7 persistence finding
(lag-7 persistence alone gets r=0.93, beating the model's own mean head
r=0.87) -- the model's behavior is fundamentally persistence-like, just
smoothed. Good candidate for a concrete failure-case figure in the paper
(known limitation: detects ongoing/established events, does not
anticipate turning points) rather than something to "fix" -- a genuine
predictability limit at lead=7d, not a training bug.


---

## Onset skill re-verified on full_gnll_quantile_v2 (Aug 21 2026) — supersedes the old negative result

User pushed back on accepting the old onset-skill negative result at
face value ("si vamos a detectar drivers, TIENE QUE PREDECIR EL
INICIO") -- correctly, since that number was computed on the buggy
`kfold` split and a pre-quantile-head model family (Paso 7 item 2, never
actually re-verified until now). Re-ran with `scripts/eval_onset_skill_
quantile_v2.py` (job 29436340) on the committed `full_gnll_quantile_v2`,
`stratified_kfold`, pooled across all 5 folds, both heads, with 95%
Fisher-z CI:

| Phase | n | r (mean head) | r (quantile head) | r (persistence) |
|---|---|---|---|---|
| **Onset** | 56 | **-0.293 [-0.516,-0.032]** | -0.131 [-0.381,+0.137] | +0.265 [+0.002,+0.494] |
| Mid-event | 987 | +0.292 [+0.233,+0.348] | +0.375 [+0.320,+0.428] | +0.686 [+0.651,+0.717] |
| No-MHW | 13456 | +0.839 [+0.834,+0.844] | +0.823 [+0.818,+0.829] | +0.916 [+0.913,+0.919] |
| All | 14499 | +0.850 [+0.845,+0.854] | +0.838 [+0.833,+0.843] | +0.929 [+0.927,+0.931] |

**Update Aug 21 2026 — the Onset/Mid-event/No-MHW row split above needs
re-verification (user's methodological review, confirmed against the
code, known_issues.md #56.2).** `eval_onset_skill_quantile_v2.py` never
got the #53 per-year fix: `onset_mask()` ran `apply_hobday()` on each
fold's full concatenated test series at once, and `stratified_kfold`'s
non-consecutive test years (fold0: 1985, 1991, 2000, ...) mean that's
NOT calendar-contiguous even though it's chronologically sorted — the
script's own old comment wrongly claimed otherwise. Same issue in the
`persist` lag calculation (no year-boundary invalidation). **The "All"
row (n=14499, r=0.850/0.838/0.929) is NOT affected** — a pooled Pearson r
over every test day doesn't depend on `apply_hobday()`'s event-boundary
logic at all, only day-level values. **The Onset/Mid-event/No-MHW
row split IS affected** — those depend on the `mhw`/`onset` day labels
that `apply_hobday()` produces, which could spuriously merge/split
events across the hidden year boundaries. Fixed in the script (per-year
loop, matching `eval_onset_skill_curve.py`'s already-correct pattern);
relaunched (job 29450811, completed). **Corrected numbers, use these
instead of the table above**:

| Phase | n | r (mean head) | r (quantile head) | r (persistence) |
|---|---|---|---|---|
| **Onset** | 54 | **-0.337 [-0.555,-0.076]** | -0.166 [-0.415,+0.107] | +0.304 [+0.039,+0.528] |
| Mid-event | 978 | +0.291 [+0.233,+0.348] | +0.374 [+0.319,+0.427] | +0.695 [+0.661,+0.726] |
| No-MHW | 13222 | +0.838 [+0.833,+0.843] | +0.822 [+0.817,+0.828] | +0.925 [+0.923,+0.928] |
| All | 14254 | +0.848 [+0.844,+0.853] | +0.837 [+0.832,+0.842] | +0.937 [+0.935,+0.939] |

Direction unchanged, onset row if anything slightly MORE negative
(-0.337 vs -0.293) and still clearly significant (CI excludes zero).
n dropped slightly everywhere (boundary-crossing samples now correctly
excluded/NaN'd rather than silently merged across a fake year gap).
The core conclusion — the model's mean/quantile head do not anticipate
onsets, raw persistence does slightly better and is also weak — is
unchanged, now on a methodologically sound footing.

**This supersedes the old result and is stronger, not just "confirmed
negative"**: the old number (r=0.006-0.093, both model and persistence
indistinguishable from zero) is REPLACED by this. Under the corrected
split/model: the mean head's onset-day correlation is **negative and
statistically significant** (95% CI excludes zero) -- not merely "no
skill", the model's mean prediction is *anti-correlated* with truth
specifically at onset. Persistence retains a weak-but-significant
positive correlation at onset (CI barely excludes zero, [+0.002,+0.494]).

**Directly confirms, quantitatively across all 56 onset days (not just
the one 2014 case), the mechanism found in the case-study entry above**:
the model's predictions decline/plateau right as a real event is about
to accelerate, because the 7-day-lagged input doesn't yet contain the
turning-point signal. The earlier case study (doy~150-160, 2014) wasn't
an isolated anecdote -- this is systematic.

**For the paper**: this is a real, well-characterized, citable negative
result at the onset phase specifically (not swept under the rug) --
strengthens rather than weakens the paper's honesty. The model adds
clear value at mid-event/no-MHW/overall (matches the def2 recall/
precision result already established), the genuine, quantified
limitation is onset-day prediction at lead=7d.


---

## Onset-day-boundary bug confirmed real (small effect) + skill-recovery curve (Aug 21 2026)

Re-ran the day-0 onset check with `eval_onset_skill_curve.py`'s per-year
fix (known_issues.md #53 -- stratified_kfold's non-consecutive test
years must not be concatenated before `apply_hobday()`). n_onset drops
from 56 -> 54 (2 spuriously-labeled onset days removed). **Finding
holds, slightly strengthens**: r_mean at day0 = -0.337 [-0.555,-0.076]
(was -0.293 with the bug). Per-fold: negative in 3/5 folds (-0.45 to
-0.52), near-zero in 1, **positive in 1** (fold3: +0.32) -- majority
direction confirmed, not perfectly uniform, worth stating honestly.

**Skill-recovery curve (user's request: "es demasiado restrictivo mirar
solo un dia?")** -- r vs. days-since-onset, pooled 5 folds:

| day | r_mean | r_quantile | r_persist | n |
|---|---|---|---|---|
| 0 | -0.337 | -0.166 | +0.304 | 54 |
| 3 | -0.185 | +0.001 | +0.531 | 54 |
| 8 | -0.043 | +0.102 | +0.868 | 41 |
| 9 | +0.007 | +0.103 | +0.832 | 37 |
| 14 | +0.294 | +0.344 | +0.790 | 24 |

Quantile head crosses to positive ~day 3, mean head ~day 9, persistence
rockets to >0.8 by day 8 (once an event is a few days old, "yesterday's
value" becomes a very strong predictor of tomorrow's). **Refines the
onset finding**: not "the model never predicts MHW conditions", but
specifically "the transition INTO an event (days 0-2) is a blind spot
that resolves within about a week" -- a sharper, more defensible, more
publication-ready characterization. Full table in `experiments/figures/
step7_persistence/onset_skill_curve.png`/`.npz`.


---

## Event-level onset detection: quantile head decisively beats both mean head and persistence (Aug 21 2026) -- headline result

`scripts/eval_event_detection.py` (job 29437897), reviewed by the user
before launch (3 real gaps fixed pre-launch: added the mean-head arm,
added Clopper-Pearson + event-paired bootstrap uncertainty, made the
false-alarm convention explicit + a strict sensitivity variant --
verified with synthetic unit tests before spending compute, see the
script's own docstring for the full account).

Definitions: alarm(t) = head_pred(t) > p90_ns[doy(t)] (raw threshold
crossing); HIT = a real Hobday event with >=1 alarm in
[t_onset-LEAD, t_onset] (LEAD=7); per-year processing throughout
(known_issues.md #53).

| System | n hits/misses | POD (95% CP) | FAR | CSI | lead_time median |
|---|---|---|---|---|---|
| **Quantile head** | 39/17 | **0.696 [0.559,0.812]** | 0.777 (strict 0.748) | 0.203 | **7.0 d** |
| Mean head | 9/47 | 0.161 [0.076,0.283] | 0.878 (strict 0.757) | 0.074 | 6.0 d |
| Persistence (lag-7) | 10/46 | 0.179 [0.089,0.304] | 0.865 (strict 0.643) | 0.083 | 7.0 d |

n_events=56 total (all real Hobday onsets across 40 years, pooled 5 folds).

**Event-paired bootstrap (B=2000, resample events not systems
independently) POD differences**:
- quantile_head vs persist: **+0.517 [+0.393,+0.643], SIGNIFICANT**
- mean_head vs persist: -0.018 [-0.143,+0.107], **not significant** --
  the mean head is statistically indistinguishable from trivial
  persistence at event-level onset detection.
- quantile_head vs mean_head: +0.536 [+0.411,+0.661], significant.

**Cleanest single number**: of the 56 events, mean-head hits and
quantile-head misses = **0** ("only_mean" = 0) -- the quantile head is
essentially a strict superset detector over the mean head at event
level: every event the mean head catches (9), the quantile head also
catches, plus 30 more the mean head misses entirely.

**This is the headline result connecting the whole day's work**: the
-0.39C onset-day mean-head bias (Aug 21 bias entry), the mean-head's
statistically-negative onset-day correlation (Aug 21 onset-skill entry),
and the Seitzer et al. 2022 GNLL variance-gradient-dominance mechanism
(Aug 21 loss-curve-defensibility entry) all converge on the same
practical conclusion: **the mean head cannot be trusted for onset
detection, but the quantile head -- trained on the exact same backbone,
same data, same checkpoint -- decisively can**, catching ~70% of real
onsets with a median 7-day lead time (the full forecast horizon this
model was trained for).

**Honest caveat, report alongside POD, not instead of it**: FAR is high
for every system (78-88%) -- this is not a clean, low-false-alarm
signal. The quantile head fires much more reliably when a real event is
coming, but it also fires often when nothing happens. CSI (0.203 for the
quantile head) reflects this balance more completely than POD alone.

Figures: `experiments/figures/step7_persistence/event_detection_pod_far_csi.png`,
`event_detection_summary.json`.


---

## LR diagnostic (2e-5) result: modest improvement, not worth switching (Aug 21 2026)

`full_gnll_quantile_v2_lr2e5` fold0 (job 29433645) completed (3h43min).
Real per-epoch data: best epoch=9 (val_loss=+0.0373), stopped epoch 39
(gap=30, patience honored exactly again). Oscillation ratio
worst/best~32x (vs. ~69x for the committed 5e-5 model's fold0) -- a real
reduction, roughly half, but not a fix. The new minimum itself is
slightly worse in absolute NLL terms (+0.037 vs -0.025). Test metrics
(fold0 only): MAE=0.298C, r=0.834 -- comparable to, slightly below, the
committed model's pooled 5-fold numbers (MAE=0.279C, r=0.850).

**Decision: keep the committed LR=5e-5 model, do not switch.** The
improvement from lowering LR is real but modest, doesn't clearly
outperform the current model on the metrics that matter, and would cost
a full 5-fold retrain to adopt. The current model's loss-curve noise is
already defensible (Seitzer et al. 2022 citation, known GNLL
variance-gradient-dominance mechanism) and its downstream results
(event-level POD=0.696 significantly beating persistence, clean def2
recall/precision) are excellent regardless of the noisy training curve,
since `ckpt_path="best"` correctly selects the pre-collapse checkpoint
either way.

---

## Correction: event-detection "lead time" was target-day-gap, not issue-to-onset lead (Aug 21 2026)

User question ("cada dia que se detecte que a los 7 dias habra MHW, no?")
prompted a precise re-check of `eval_event_detection.py`'s alarm
semantics. Confirmed: `alarm` is indexed by **target day** `t`
(`q_pred(t) > p90_ns[doy(t)]`), and `q_pred(t)` is the forecast issued 7
days earlier, on day `t-7`. So yes — her reading is exactly right: an
alarm on day `t` means "7 days ago, the model already predicted MHW for
today."

The `lead_time` value previously reported ("median 7-day lead time",
above) measured `onset_day - target_day_of_earliest_alarm` — the gap
between the actual onset and the target day the winning forecast was
*about*, not the gap between when that forecast was *issued* and the
onset. Since issue day = target day − 7, the correct issue-to-onset lead
time is `target-day-gap + 7`. Recomputed from the already-saved
`event_detection_summary.json` (no rerun needed, pure arithmetic on
saved lead_time arrays):

| system | old (target-day-gap) mean/median | corrected (issue-to-onset) mean/median | range |
|---|---|---|---|
| quantile_head | 6.03 / 7.0 | **13.03 / 14.0** | 7-14d |
| mean_head | 4.67 / 6.0 | **11.67 / 13.0** | 7-14d |
| persist | 6.30 / 7.0 | **13.30 / 14.0** | 9-14d |

The corrected numbers are *more* favorable, not less: the model typically
had already issued a correct warning 13-14 days before the real onset,
not 7. POD/FAR/CSI/bootstrap results are unaffected (they don't depend
on this label, only the alarm window logic, which was correct). Only the
"lead time" statistic and its prose description need updating wherever
cited (this entry, and the figure/JSON's field name if reused in the
paper — `event_detection_summary.json`'s raw `lead_times` lists are
unchanged, still target-day-gap; add 7 when quoting as issue-to-onset
lead).

---

## Documented limitation: per-year Hobday processing can split real
## Dec31/Jan1-spanning events (Aug 21 2026, user-requested)

See `known_issues.md` #54 for full detail. Summary: the #53 fix (process
`apply_hobday()` one calendar year at a time, to stop spurious merging
across a fold's non-consecutive test years) has a real cost — a genuine
MHW event that spans Dec31→Jan1 of two *consecutive* years gets split at
the boundary regardless of foldedness, since no per-year processing ever
sees both halves as one contiguous run. Quantified against the full
40-year contiguous series (not per-fold): **3 of 39 consecutive-year
boundaries (2006→2007, 2015→2016, 2022→2023) have a real event spanning
the boundary** — 3 of 52 total Hobday events over the record (~5.8%).
This is a caveat on the onset-skill curve and event-detection headline
numbers (`n_events=56`), not a correction to them — not re-verified
per-fold, not fixed, flagged as a known limitation per the user's
explicit documentation request (not an immediate fix request).

---

## Item 2 (persistence lag sweep, no retrain): decay curve only, model
## comparison deferred (Aug 21 2026)

`persistence_lag_sweep.py` (job 29438585, CPU, 7s) swept raw
lag-N persistence Pearson r over the full 40-year `target` series,
lag=1..14 (no model involved):

| lag(d) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r | .997 | .990 | .981 | .971 | .960 | .949 | .938 | .926 | .916 | .905 | .895 | .885 | .875 | .865 |

Smooth, monotonic decay, no crossover visible against the model's own
lead=7 r (0.850 mean head / 0.838 quantile head) since there is no
model-side curve to compare against yet — only actual retrains at other
lead times would produce that crossover. Per user's explicit sequencing,
deferred until "la configuracion final del modelo" (land_fill_mode
decision) is settled, expected later today. Figure:
`experiments/figures/step7_persistence/persistence_lag_sweep.png`.

---

## Item 4 (composite precursor + Granger causality): descriptive only,
## explicitly NOT a paper-selling ML result (user's own framing, Aug 21 2026)

`composite_precursor_analysis.py` (job 29442085, CPU, 5min, no model, no
train/test split) computed two things on the full 40-year contiguous
record: (1) a spatial composite — mean(anomaly field, k days before a
real MHW onset) minus mean(anomaly field, k days before a season-matched
non-onset control day) — NS-box mean, k=1-14d, n=52 real onsets; (2)
Granger causality, NS-box-mean series per variable vs. target, lags
1-14.

**Granger causality result: METHODOLOGICALLY BROKEN, do not cite.**
p=0.0000 for all 5 variables × all 14 lags, no exceptions — the
textbook false-positive signature of testing highly autocorrelated
series without differencing/pre-whitening (`target`'s own lag-1
autocorrelation is r=0.997, decays very slowly per the persistence
sweep above; with series this "sticky," any smooth variable spuriously
"Granger-causes" any other). Not fixed — would need `Δtarget(t)` vs.
`Δpredictor(t-k)`, or an AR fit per series tested on residuals, before
repeating. Offered to the user, not yet confirmed as wanted.

**Composite NS-box curve, point estimates only (no significance test
yet at this stage)**: `ptho_bot` gave a smooth, monotonic curve from
+0.316°C (k=1) to +0.277°C (k=14) — visually the cleanest of the 5
variables. `u10`/`v10`/`msl`/`ssr` all looked noisy/non-monotonic by
eye.

**Update Aug 21 2026, bootstrap CI added (user request, after asking
what a bootstrap CI is)**: `composite_bootstrap_ci.py` (job 29443125,
CPU, 5.5min) reused the exact same onset/control definitions and
independently bootstrapped both the n=52 onset events and the
season-matched control pool (resample with replacement, B=10000,
2.5/97.5 percentile CI per variable per k). Point estimates matched
`composite_ns_box_curve.npz` exactly (consistency check, zero
mismatches) before trusting the CI.

| variable | k's with 95% CI excluding zero | read |
|---|---|---|
| `ptho_bot` | **14 / 14** (all) | clean, real, monotonic signal — CI never crosses zero, width ~±0.02-0.04°C |
| `ssr` | 10 / 14 | intermittently significant, not monotonic |
| `v10` | 7 / 14 | intermittently significant, not monotonic |
| `u10` | 4 / 14 (only k=1-4d) | significant only at short lag, noise beyond that |
| `msl` | 0 / 14 (none) | no detectable NS-box composite signal at any lag |

This confirms the by-eye impression quantitatively: `ptho_bot`'s
precursor signal is qualitatively different from the other 4 — always
significant, not just visually smoother — while `msl` in particular has
no real composite signal at all. Still descriptive/for-understanding
only, per the user's explicit framing at the start of this analysis —
not to be presented as a sellable ML result. Figures/data:
`experiments/figures/step7_persistence/composite_precursor/
composite_ns_box_curve_bootstrap.{png,npz}`.

---

## land_fill_mode=nearest fold0 retrain (job 29438575, Aug 21 2026) —
## measurably WORSE than committed on every test metric, once compared fairly

Item 1's first retrain finished: same seed=42, identical config to the
committed `full_gnll_quantile_v2` fold0 except `land_fill_mode: nearest`
(configs diffed directly — only `output_dir` and `land_fill_mode` differ,
nothing else confounds the comparison). Best checkpoint: epoch=06,
val_loss=0.1015.

**A naive first comparison looked like a huge win and was wrong** — the
committed fold0's previously-logged test metrics (job 29417405, Aug 20
15:29, r=0.8062, test_loss=3.08) predate the `ckpt_path="best"` fix
(commit `33ebb70`, Aug 21 00:34 — confirmed via `git log`, not assumed).
`known_issues.md` #46 names this exact fold0 comparison as affected:
`trainer.test()` evaluated whatever epoch `EarlyStopping` stopped at (a
GaussianNLL variance-collapse epoch), not the real best checkpoint
(epoch=13, val_loss=-0.0245, still on disk, never freshly evaluated with
the fix). User asked directly whether the loss had improved and whether
LR had changed (Aug 21 2026) — LR had NOT changed (both configs
`learning_rate: 5.0e-05`, confirmed), but the loss comparison itself
needed re-doing properly before answering.

**Fair comparison** (`scripts/eval_test_metrics_from_best_ckpt.py`, job
29444030, reloads committed fold0's actual best checkpoint epoch=13 and
evaluates it the same way land_fill's run did):

| metric | committed (epoch13, corrected) | land_fill (epoch06) | delta |
|---|---:|---:|---|
| test_corr | 0.8657 | 0.8313 | worse |
| test_mae (°C) | 0.2665 | 0.2958 | worse |
| test_loss | 0.0233 | 0.0481 | worse, ~2x |
| test_nll_loss | -0.0055 | +0.0133 | worse |
| test_pinball_loss | 0.0962 | 0.1160 | worse |

Cross-check: 0.8657 fold0-only is consistent with the headline
`pooled_r_mean_head=0.850` (5-fold average) — that number's own script
(`eval_onset_skill_quantile_v2.py`) already used `best_ckpt()` correctly,
so it needs no correction, only fold0's train_partition.py-reported
numbers were stale.

**land_fill_mode=nearest measurably hurts model performance** — worse on
every single metric, not noise (same seed, same everything else). It was
only ever intended to fix the IG coastal-edge artifact
(`known_issues.md` #52), not to change performance. This is a real
tradeoff to weigh before relaunching folds 1-4 or the local/remote
configs: even if the IG check (pending) comes out visually clean, the
underlying model got worse. Flagged to the user, not yet decided how to
proceed — options include (a) accept the performance hit if the paper
cares more about clean XAI than the last ~3-4% of r, (b) keep
`land_fill_mode=zero` for the "real" model and use `nearest` only for a
separate XAI-focused checkpoint, (c) investigate why nearest-fill hurts
performance (plausible reading, not verified: the model may have been
partly relying on the zero-fill edge itself as a — non-physical but
real — predictive signal) before deciding.

**Update Aug 21 2026 — IG on the land_fill checkpoint finished (job
29443440), directly answers "physical or artifact?" (user's explicit
framing of what would drive the decision) — verdict: mixed, both are
real, neither alone explains it.** `scripts/analysis/
ig_coastal_decay_check.py` reproduces #52's exact methodology (distance-
to-coast binning + NS-box open-water enrichment) on the new checkpoint's
saved `.npy`, directly comparable to the committed run:

| | committed (zero-fill) | land_fill (nearest) | change |
|---|---:|---:|---|
| coastal decay (1-2px/>9px), mean/quantile head | 18.76x / 18.24x | 12.99x / 13.51x | ↓~30%, NOT eliminated |
| NS-box open-water enrichment, mean/quantile head | 12.22x / 10.75x | 11.96x / 10.57x | **essentially unchanged** |

Two things follow from this, together:
1. **The NS-box open-water enrichment (>5px from any coast) does not
   move at all** when the land-fill convention changes, despite ocean
   pixel *values* being bit-identical between modes either way — strong
   evidence this specific signal is not a land-fill artifact. This is
   consistent with (not contradicting) #52's original "survives the
   artifact control" finding, now cross-checked a second way. Still
   unresolved against occlusion's contradicting 0.71x figure (see
   above) — that disagreement is about IG-vs-occlusion methodology and
   is independent of land_fill_mode.
2. **The near-coast decay is only partially explained by the zero-fill
   edge** — if it were purely the artificial cliff, smoothing it away
   should have collapsed the ratio toward ~1x; instead it dropped ~30%
   and plateaued at a still-large 13x. Something genuinely tied to
   coastal proximity survives removing the hard edge.

Read together with the performance regression above: the model lost
real skill (r, MAE, loss all worse) specifically when the near-coast
decay dropped, while the land-fill-independent open-water signal stayed
put. That pattern fits a mixed story — part of what the model exploited
near the coast really was the artificial zero/real-value discontinuity
(now partially removed, some skill genuinely lost with it), and part is
a real coastal-proximity signal that neither fill convention destroys.
**Proposed next step to fully disentangle the two** (not yet run, not
authorized): retrain with `land_fill_mode=nearest` PLUS an explicit
`land_mask_tbottom` input channel (known_issues #52's option 2) — gives
the model a clean, non-data-artifact way to know "distance to coast"
without an exploitable discontinuity. If performance recovers close to
committed, that confirms coastal-proximity information itself is
legitimately useful (whatever its physical status) and was only lost
because nearest-fill removed the cheap way to access it. If it doesn't
recover, that points more specifically at the exact shape/magnitude of
the zero-edge gradient as what mattered, not coastal proximity as a
concept.

**Update Aug 21 2026 — final check, resolves this cleanly: does the
pattern show up in the RAW data?** User pushed back hard on the mixed
IG-vs-occlusion, land_fill-partial-fix story ("no me esta aportando
nada esto") — fair, since neither method alone was settling physical-
vs-artifact. Went to the actual data instead: `scripts/analysis/
raw_ptho_bot_coastal_check.py` computes raw pointwise Pearson r between
`ptho_bot(t)` and NS-box `target(t+7)` over the full 40yr record, no
model involved at all.

**The domain-wide coastal decay is NOT in the raw data** — mean|r| is
flat/slightly reversed from coast to open ocean (0.313 -> 0.360, ratio
0.87x, vs. IG's ~18x). `ptho_bot` DOES have a real coastal variance
gradient (std 0.429 -> 0.029, ~15x, physically real — shallow coastal
water is more thermally variable) but that variance does not carry
predictive power — the simplest explanation is IG's own known
sensitivity to input variance/gradient-path magnitude, not a real
precursor effect, and not something land_fill_mode was ever going to
fix (it's about variance, not the fill discontinuity).

**Region breakdown answers the user's actual question ("why would Gulf
coast proximity matter for North Sea MHWs?") directly**:

| region | raw \|r\|, near-coast (≤2px) | vs. domain far-from-coast baseline (0.360) |
|---|---:|---:|
| **North Sea / British Isles** | **0.685** | **1.91x — real** |
| Grand Banks / Nova Scotia | 0.164 | **0.46x — below average, no support** |

Grand Banks' near-coast raw correlation with the NS target is *below*
the domain's own far-from-coast baseline — the raw data gives it zero
support, consistent with the physical implausibility already raised
(ocean advection from Grand Banks to the North Sea takes weeks-months
via the NAC, not a 7-day mechanism). The North Sea's own coastal margin
is the only regional piece that survives this check with real backing.

**Final verdict for the paper**: only the North Sea's own near-coast
`ptho_bot` signal is a genuine, raw-data-verified precursor effect — cite
it, it's real (r=0.685 vs 0.360 domain baseline, model-independent).
Everything else in the "coastal IG signal" story (the domain-wide 18x
decay, Grand Banks specifically, the general "any coast matters"
reading) is artifact — do not present as physical, do not cite Grand
Banks or any other remote coastline as a precursor location.

**Update Aug 21 2026, user correction — the artifact is TWO stacked
causes, not one, "100% the data" overstated it.** Putting all three
coastal-decay-ratio numbers together (mean head, 1-2px/>9px):

| | ratio | vs. raw truth (0.87x) |
|---|---:|---|
| raw data (no model) | 0.87x | — (ground truth) |
| land_fill=nearest (no zero-edge, mask still applied) | 12.99x | large gap — NOT explained by the zero-edge |
| land_fill=zero (committed) | 18.76x | larger gap still |

This is two separate, additive artifact sources, not one: (1) the jump
from 0.87x truth to 12.99x (`nearest`) is IG's own sensitivity to
`ptho_bot`'s real local variance (present under ANY land-fill
convention, confirmed since `nearest` has no discontinuity and still
shows it) -- the dominant source; (2) the further jump from 12.99x to
18.76x (`nearest` -> `zero`) is specifically caused by the zero-fill
edge/mask itself -- a real, separate, additive inflation on top of (1),
recovered exactly by switching fill modes. Both are non-physical (raw
correlation stays flat at 0.87x-equivalent regardless), but they have
different causes and only (2) is a preprocessing choice — (1) would
persist under any land_fill_mode, including one we haven't tried.
Precise statement for the paper: "the coastal IG inflation has two
compounding artifact sources — an IG-intrinsic sensitivity to input
variance (dominant, method-level) and an additional contribution from
the zero-fill land-masking convention (secondary, preprocessing-level)
— together explain the ~18x domain-wide coastal decay pattern; neither
one is real precursor physics, confirmed against the raw pointwise
correlation, except at the North Sea's own coastal margin."

This closed the item-1 investigation for the moment, pending one more
user-designed check (below) that changed the final call.

**Update Aug 21 2026 — weight-swap ablation (user-designed experiment),
confirms causally that land-fill CONTENT (not just training-time
adaptation) drives most of the artifact, with an asymmetry.** Jobs
29448417/29448418, `configs/partition/_adhoc_swap/*.yaml`: same frozen
weights, swap only which land_fill_mode built the INPUT at inference
(no retraining).

| | decay ratio (mean/quant) | NS-box open-water enrichment (mean/quant) |
|---|---:|---:|
| zero weights + zero input (committed) | 18.76x / 18.24x | 12.22x / 10.75x |
| **zero weights + nearest input (swap A)** | **13.19x / 12.49x** | 11.86x / 10.80x |
| nearest weights + nearest input (full retrain) | 12.99x / 13.51x | 11.96x / 10.57x |
| nearest weights + zero input (swap B) | 14.34x / 15.54x | 8.83x / 7.04x |

Swap A (zero-trained weights, frozen, just fed nearest-filled input)
lands at 13.19x/12.49x — nearly identical to the full nearest retrain
(12.99x/13.51x). This is clean causal confirmation, no retraining
needed: land-pixel CONTENT itself, not the model having learned
different weights, drives most of the difference between the two
conventions. Swap B is asymmetric and weaker (14.34x/15.54x, much
closer to nearest's own 12.99x/13.51x than to zero's 18.76x/18.24x) —
consistent with the zero-trained model having learned dedicated filters
tuned to the artificial zero-edge (removing it at inference collapses
most of the extra magnitude), while the nearest-trained model never
developed that specific sensitivity, so a novel hard edge at test time
only partially perturbs it. NS-box open-water enrichment stays flat
(~11-12x) across every configuration except swap B (8.83x/7.04x, lower)
— a mild instability specific to feeding the nearest-trained model an
input pattern it never saw, treated as a secondary/noisier effect, not
overinterpreted.

**Decision, per user's own pre-stated threshold ("si mejora un poco, lo
vamos a hacer")**: the artifact reduction is confirmed by two
independent methods (full retrain AND frozen-weight swap) landing on the
same ~13x number, not a fluke of one experiment. This meets the user's
bar — **`land_fill_mode=nearest` adopted for the main full model**,
folds 1-4 launched (fold0 already done, from the earlier diagnostic
run). Note: item 2's local/remote 5-fold arrays were already launched
with `land_fill_mode=zero` before this decision — left as-is (user's
plan listed them as a separate parallel track, not conditioned on this
choice) — flagged to the user as a possible inconsistency to revisit if
the paper wants a single land_fill_mode across all model variants.

---

## EVERYTHING ABOVE ABOUT land_fill_mode=nearest IS UNRELIABLE —
## normalization bug found Aug 21 2026, user caught it, invalidates the
## whole chain of `nearest` results/decisions above

`compute_stats()` (`src/data/dataset.py`) only excluded land pixels from
`ptho_bot`'s normalization mean/std when `land_fill_mode=="zero"` —
`"nearest"` silently computed stats over the FULL grid, including the
9473 land pixels `nearest` fills with copied ocean-neighbor values.
Confirmed against the actual saved logs: zero-mode (correct) gives
`mean=0.0229, std=0.2775`; nearest-mode (bugged) gave `mean=0.0371,
std=0.3425` — **std inflated +23.4%**. Every real ocean value fed to the
model was silently attenuated by ~19% in normalized space. See
`known_issues.md` #55 for the full account.

**This invalidates, specifically**: the fold0 land_fill retrain's test
metrics (r=0.8313 etc. — the "worse than committed" comparison), the IG
coastal-decay numbers (12.99x/13.51x and the "two stacked causes"
decomposition), and the weight-swap ablation (13.19x/12.49x,
14.34x/15.54x, and its "confirms input content dominates" conclusion) —
IG runs in normalized space with a zero baseline, so a silently
mis-scaled input directly changes IG's own gradient magnitudes. None of
this was necessarily WRONG in direction, but the exact numbers, and any
conclusion drawn from comparing precise magnitudes (e.g. "swap A nearly
matches the full retrain" or "the artifact reduces by ~30%"), cannot be
trusted until redone with the fix.

**Fixed and verified** (Aug 21 2026): land pixels now excluded from
stats for both modes; re-running `LazyDataModule.setup()` on the
land_fill config post-fix gives `ptho_bot: mean=0.0229, std=0.2775` —
bit-identical to zero-mode, as expected.

**Action taken**: all `land_fill_mode=nearest` SLURM jobs running at
discovery time were cancelled (fold0-retrain's folds 1-4, and the
already-relaunched local/remote 5-fold arrays, which had also just been
switched to `nearest` for consistency) — none of them should be allowed
to finish on the buggy stats. **Not yet done, pending user decision**:
redo fold0 training, the IG coastal-decay check, and the weight-swap
ablation with the fix before deciding zero-vs-nearest again — the
previous "nearest adopted" decision (immediately above) is ON HOLD, not
confirmed, until this rework happens.

**Update Aug 22 2026 — fold0 land_fill retrain redone with the
normalization fix (job 29449798, 3h27, completed). Performance
confirmed genuinely worse, not a normalization artifact.** Verified
`ptho_bot: mean=0.0229, std=0.2775` in this run's log — bit-identical to
committed's zero-mode, fix confirmed applied in production, not just in
the standalone check. Test metrics (correct normalization this time):
r=0.8237, MAE=0.3004°C, test_loss=0.1046, test_nll_loss=0.0688,
test_pinball_loss=0.1192 — still clearly worse than committed's real
best-checkpoint numbers (r=0.8657, MAE=0.2665°C, test_loss=0.0233,
test_nll_loss=-0.0055, test_pinball_loss=0.0962), and if anything
slightly worse than the earlier BUGGY nearest run's r=0.8313. So the
performance regression from `land_fill_mode=nearest` is real, not an
artifact of the normalization bug -- it was already correctly measuring
"nearest is worse," just via a differently-miscalibrated input.

**Checkpoint directory contamination found and fixed**: relaunching job
29449798 into the same `output_dir` as the earlier buggy run (job
29438575) left 6 `.ckpt` files in `checkpoints/` -- 3 from the old buggy
run (best: epoch=06, val_loss=0.1015) and 3 from the new correct run
(best: epoch=06, val_loss=0.1399). `best_ckpt()` scans the whole
directory and picks lowest val_loss regardless of which run produced it
-- since 0.1015 < 0.1399, every future script calling `best_ckpt()` on
this directory (IG, eval, anything) would have silently loaded the
OLD, wrong-normalization checkpoint instead of the new correct one.
Caught before any downstream use; the 3 old files (timestamps confirm
which is which) were deleted, leaving only the new run's 3 checkpoints.
**Lesson for next time**: don't relaunch training into an existing
`output_dir` without clearing/renaming its `checkpoints/` first, or
`best_ckpt()`'s "lowest val_loss wins" logic can silently cross runs.

**Status**: `land_fill_mode=nearest`'s performance cost is now
confirmed on solid ground (correct normalization, real regression, not
a bug artifact). The artifact-reduction side of the decision (IG
coastal-decay comparison) still needs redoing with BOTH fixes (correct
normalization AND the stratified year-sampling fix, known_issues.md
#57 P1) before the "adopt nearest" call can be revisited -- per the
user's own sequencing choice, deferred until folds/items 2-3 are all
done, then IG gets redone across the board at once.

**Decision Aug 22 2026 — `land_fill_mode=nearest` adopted for
EVERYTHING, final call.** User's own reframing after reviewing the
fold0 comparison in depth: the performance drop under `nearest` is
plausibly explained by the model having learned to exploit the
zero-fill edge as a non-physical but statistically real shortcut
(consistent with the coastal-decay-ratio reduction and the flat raw
correlation found earlier) -- so the drop is not evidence `nearest` is
a worse *modeling choice*, it's a measure of how much skill depended on
an artifact. Given that, `nearest` is adopted as the single
land_fill_mode across the whole project, no more mixed zero/nearest
tracks.

**Launched (Aug 22 2026), all `land_fill_mode=nearest`**:
- Item 1: `full_gnll_quantile_v2_landfill` folds 1-4 (job 29457750,
  array 1-4; fold0 already done, job 29449798, correct normalization
  confirmed).
- Item 2: `local/` (job 29457751) and `remote/` (job 29457752) 5-fold
  arrays -- configs already had `land_fill_mode: nearest` added
  earlier.
- Item 3: lead-time sweep, 5 folds each, at lead={3,5,14,30}d (jobs
  29457753/54/55/56) -- new configs generated at `configs/partition/
  lead{3,5,14,30}_landfill/fold{0-4}.yaml` from the same template
  (only `lead_time`/`output_dir` differ). lead=7d's point in the sweep
  is `full_gnll_quantile_v2_landfill` itself (item 1 above) -- no
  duplicate config needed.

34 GPU jobs total in flight. Once all land, IG/GradCAM/Shapley get
redone across the board with the stratified-sampling fix
(known_issues.md #57 P1) -- deferred per the user's explicit
sequencing choice ("IG lo haremos cuando tengamos todos los runs").

---

## Complete analysis of the 34-job batch (Aug 22 2026), before moving to
## XAI, per user's request ("haz un analisis de ellas... sin gastar
## muchisima mas GPU") -- all of the below is CPU-only, no GPU spent

All 34 GPU jobs completed cleanly (exit 0:0, no failures). Generalized
two existing evaluation scripts (`eval_recall_v2_partition.py` from
`quantile_head_recall_v2_all5.py`, and `eval_onset_skill_quantile_v2.py`
in place) to run against any of the 7 families instead of duplicating
per-family scripts (this project's own documented anti-pattern). Applied
`mask_local`/`mask_remote` (`src/data/masking.py`) before evaluating the
local/remote checkpoints -- evaluating them on unmasked input would be
out-of-distribution and wrong, since `RemoteOnlyLightningModule`/
`LocalOnlyLightningModule` only override the LightningModule's step
methods, not `model.forward_with_quantile()` itself, which the eval
script calls directly.

**Pooled r + def1/def2 recall/precision/FPR, all 7 families** (job
29463377):

| family | lead | r_mean | r_quantile | def1 recall (q) | def1 precision (q) | def2 recall (q) | def2 precision (q) |
|---|---|---:|---:|---:|---:|---:|---:|
| full | 7 | 0.8141 | 0.7981 | 73.0% | 26.6% | 41.9% | — |
| local | 7 | 0.9081 | 0.8997 | 90.2% | — | 50.1% | — |
| remote | 7 | 0.7965 | 0.7719 | 69.8% | — | 41.0% | — |
| lead3 | 3 | 0.8896 | 0.8840 | 86.8% | 36.3% | 42.7% | — |
| lead5 | 5 | 0.8548 | 0.8429 | 76.9% | 35.6% | 37.4% | — |
| lead14 | 14 | 0.7473 | 0.7331 | 57.2% | 18.7% | 43.3% | — |
| lead30 | 30 | 0.6501 | 0.6012 | 50.5% | 15.9% | 40.6% | — |

Sensible ordering throughout: local > full > remote at fixed lead=7
(thermal inertia dominance, already documented); monotonic r decay with
lead (3→30d) for both raw r and def1 recall.

**Persistence recall/precision/FPR baseline added** (user's idea #1,
job 29463887, `scripts/analysis/persistence_recall_baseline.py`) --
makes the model's def1/def2 numbers interpretable instead of a number
with no reference point:

| lead | persist r | persist def1 recall/precision/FPR | persist def2 recall/precision/FPR |
|---|---:|---|---|
| 3 | 0.9812 | 82.2% / 81.2% / 1.5% | 20.1% / 99.9% / 0.0% |
| 5 | 0.9597 | 72.2% / 71.5% / 2.2% | 20.0% / 99.4% / 0.1% |
| 7 | 0.9369 | 63.9% / 62.5% / 3.0% | 20.0% / 98.6% / 0.2% |
| 14 | 0.8647 | 45.4% / 44.1% / 4.5% | 19.1% / 93.4% / 0.8% |
| 30 | 0.7398 | 31.2% / 29.9% / 5.7% | 17.8% / 85.8% / 1.7% |

**The real "where can the model still win" answer, honestly framed —
recall/precision TRADE, not a strict win**: the model's quantile head
beats persistence's def1 recall at EVERY lead (+4.6pp at lead3, growing
to +19.3pp at lead30) and def2 recall even more (+17-24pp, roughly flat
across leads) -- and the recall gap WIDENS as lead grows, exactly where
persistence naturally weakens most. But this is not free: the model's
precision is LOWER than persistence's at every lead (def1: 36.3% vs
81.2% at lead3, down to 15.9% vs 29.9% at lead30) and FPR is higher
(11.9-20.9% vs persistence's 1.5-5.7%) -- the model flags more days
overall, which mechanically buys recall at a real false-alarm cost.
**Correct characterization for the paper**: the quantile head shifts
the operating point toward higher recall at the cost of precision,
consistently across all lead times, with the recall advantage growing
at longer leads where persistence decays fastest -- not a strict
Pareto improvement, a real trade whose value depends on the
application's cost asymmetry (missing a real MHW vs. a false alarm).

**Quantile head calibration (tau=0.9) — user's idea #3, the one
diagnostic persistence cannot have by construction**
(`scripts/analysis/quantile_calibration_check.py`, post-processes
already-saved npz, no new job): overall coverage is close-ish to the
90% target (82.5-88.9% across families) but **collapses specifically on
extreme/MHW days** (29.8-72.0%, vs 85.3-93.5% on normal days) --
i.e. `q_pred` underestimates how extreme a real event gets, far more
often than the calibration target would allow, precisely on the days
that matter most:

| family | lead | overall coverage | on-extreme-days | on-normal-days |
|---|---|---:|---:|---:|
| full | 7 | 86.4% | 42.5% | 89.9% |
| local | 7 | 87.8% | **72.0%** | 89.0% |
| remote | 7 | 84.8% | 42.2% | 88.1% |
| lead3 | 3 | 83.5% | 60.5% | 85.3% |
| lead5 | 5 | 82.5% | 46.4% | 85.3% |
| lead14 | 14 | 84.7% | 35.3% | 88.6% |
| lead30 | 30 | 88.9% | **29.8%** | 93.5% |

Extreme-day coverage decays close to monotonically with lead
(60.5%→46.4%→42.5%→35.3%→29.8% at lead 3→5→7→14→30) -- makes physical
sense: the further out the forecast, the harder to know in advance how
severe an event will become, so the quantile head increasingly
under-claims severity. `local` (72.0%) is much better calibrated on
extreme days than `remote` (42.2%) at the same lead=7 -- consistent
with local information mattering for magnitude, not just detection.
**This is a genuine, citable limitation**: don't present q_pred as a
trustworthy upper bound specifically during real events, especially at
longer leads.

**Lead-time sweep, model vs. persistence, final figure**
(`scripts/analysis/lead_time_sweep_model_vs_persistence.py`, now with a
second gap panel and "zero free parameters" annotation per the user's
request): **confirms the original Aug 21 2026 question with a clean
negative answer on raw r — no crossover at any lead time.** Persistence
wins at every point (3/5/7/14/30d), gap stable around +0.09 to +0.14
(mean head) and +0.10 to +0.14 (quantile head), if anything widening at
the extremes rather than closing. This directly answers "localizar el
crossover donde el modelo gana" from the Aug 21 idea list: **there is
no crossover in raw correlation** -- the crossover, where it exists at
all, is specifically in event-detection recall (above), not point-
forecast skill. Figure: `experiments/figures/step7_persistence/
lead_time_sweep_model_vs_persistence.png`.

**Transition-day (onset/mid-event) analysis for the main model, user's
idea #2** -- job launched (29463904), crashed on a bug introduced while
generalizing `eval_onset_skill_quantile_v2.py` (removed the old
`LEAD = 7` module constant to parametrize the script by `--config_dir`,
missed that `persist[LEAD:]` still referenced it -- a grep with too
narrow a pattern (`LEAD)`) failed to catch the colon-terminated usage).
Fixed properly: `LEAD` now read from the target config's own
`lead_time` rather than hardcoded, making the script correctly reusable
for any lead-time family, not just lead=7 by coincidence. Relaunched
(job 29464344, COMPLETED) -- results below, this note now final.

**Onset/mid-event/no-MHW skill, `full_gnll_quantile_v2_landfill`
(nearest), pooled 5 folds:**

| Phase | n | r (mean head) | r (quantile head) | r (persistence) |
|---|---|---|---|---|
| **Onset** | 54 | -0.018 [-0.284,+0.251] | +0.091 [-0.181,+0.350] | +0.304 [+0.039,+0.528] |
| Mid-event | 978 | +0.341 [+0.284,+0.395] | +0.372 [+0.317,+0.425] | +0.695 [+0.661,+0.726] |
| No-MHW | 13222 | +0.798 [+0.791,+0.804] | +0.778 [+0.771,+0.785] | +0.925 [+0.923,+0.928] |
| All | 14254 | +0.813 [+0.808,+0.819] | +0.797 [+0.791,+0.803] | +0.937 [+0.935,+0.939] |

**The negative result idea #2 was looking for, found**: comparing
against the committed (`zero`) model's onset row (r_mean=-0.337
[-0.555,-0.076], r_quantile=-0.166 [-0.415,+0.107], same persistence
r=+0.304 [+0.039,+0.528] both times, as expected -- persistence doesn't
depend on which model is being evaluated) -- **`nearest`'s onset-day
skill for BOTH heads is statistically indistinguishable from zero** (CI
includes zero both times), whereas `zero`'s mean head was significantly
NEGATIVE. Persistence, meanwhile, retains weak but statistically real
skill at onset in both cases (CI excludes zero, r~0.30) -- persistence
does not completely fail at the exact transition moment, contrary to
the naive expectation that it necessarily must; the MODEL is the one
with no detectable onset skill. **Honest characterization for the
paper**: the model (mean or quantile head, either land_fill convention)
does not reliably anticipate MHW onsets -- at best indistinguishable
from noise there, at worst (the zero-fill model) actively anti-
correlated -- while a trivial 7-day persistence flag retains weak real
skill even at the transition itself. This is a genuine limitation to
report plainly, not a result to explain away.

This closes the complete post-batch analysis (recall/precision vs
persistence baseline, quantile calibration, lead-time sweep, onset/
transition skill) -- moving to XAI (GradCAM/IG/Shapley) next, per the
user's explicit instruction to proceed autonomously from here.

---

## XAI battery, Aug 22 2026: GradCAM + IG (re-run with the fixed
## sampling) + GradientSHAP, all on committed (zero) and current
## (nearest) fold0, autonomously per the user's instruction

**Bug found and fixed before running GradCAM at all**: `src/xai/
grad_cam.py`'s `AttentionGradCAM.compute()` called `self.model(xs, xt)`
directly and `.backward()`'d the result -- for `gaussian_nll=True`
models `forward()` returns `(batch, 2)` = `[mean, log_var]`, so
`.squeeze()` does NOT give a scalar and `.backward()` either errors or
implicitly mixes mean+log_var gradients -- the exact bug class already
documented for IG (known_issues.md #49/#51: never average/conflate
heads). Added a `head` parameter (`"mean"`/`"quantile"`/legacy
`"mean_mse"`) so GradCAM explicitly picks one output column, same
discipline as IG. Verified backward-compatible with existing callers
(`scripts/gradcam_partition.py`, `scripts/run_xai.py`, the two archived
poster scripts) -- all call with no `head` arg, defaulting to `"mean"`,
which for their non-gaussian_nll MSE checkpoints falls through
unchanged to the original code path.

New generalized script `gradcam_quantile_partition.py` (matches
`ig_partition_quantile.py`'s conventions: `stratified_test_sample()`,
both heads separate, land-mask-greyout plotting). Ran on committed
(job 29464891) and nearest (job 29464892) fold0, 297 stratified samples
each, ~1.5min per run (GradCAM's own docstring claim of "~50x cheaper
than IG" holds).

**GradCAM's coastal-decay/NS-box numbers -- independent triangulation,
third method alongside IG and occlusion**:

| method | committed decay | nearest decay | committed NS-box enrichment | nearest NS-box enrichment |
|---|---:|---:|---:|---:|
| IG | 18.76x | 12.99x | 12.22x | 11.96x |
| Occlusion | 7.29x | (not run) | 0.71x | (not run) |
| **GradCAM** | **2.33x** | **1.20x** | **6.43x** | **1.64x** |

GradCAM shows a MUCH weaker coastal decay than either IG or occlusion
(2.33x committed, 1.20x -- essentially flat -- for nearest), consistent
with most of IG's sharp domain-wide decay being an IG-specific artifact
(gradient/variance sensitivity, per the raw-data check already done)
that a structurally coarser method (GradCAM's native resolution is
17x25, bilinear-upsampled -- cannot represent fine coastal structure by
construction) simply cannot reproduce. **New wrinkle GradCAM adds**:
unlike IG (whose NS-box enrichment barely moved, 12.22x->11.96x, across
land_fill modes), GradCAM's NS-box enrichment DROPS substantially
(6.43x->1.64x) when switching zero->nearest -- suggesting GradCAM's
version of the "NS box matters" signal is more entangled with the
land-fill convention than IG's version. Given GradCAM's coarse spatial
resolution, exact multiplier comparisons across methods need this
caveat; the qualitative takeaway stands: **the sharpest claims about
`ptho_bot`'s coastal importance are the most method-fragile**, while the
North Sea's own coastal margin (already verified directly in raw data,
independent of any attribution method, known_issues.md #57) remains the
one claim not resting on a single method's idiosyncrasies.

**IG re-run with the stratified-sampling fix**: jobs 29464844
(committed) / 29464845 (nearest) launched, old sampling-biased `.npy`
files backed up first to `experiments/figures/xai_integrated_gradients/
_biased_sampling_backup_2026-08-21/` for the record (so the sampling
bug's own effect size is measurable later, not just asserted) --
results pending, will supersede the corresponding numbers throughout
this doc and `known_issues.md` #52/#57 once landed.

**GradientSHAP (Expected Gradients) -- the "Shapley" leg of the
original priority list, since full KernelSHAP is intractable for this
input size**: `gradientshap_quantile_partition.py`, captum's
`GradientShap`, background = 16 real training windows (not IG's single
zero baseline) + `n_samples=10` random interpolations/noise draws per
analyzed sample -- a genuinely different baseline philosophy from IG's
single fixed zero point, not just a re-skin. Dry-run (2 samples, CPU)
verified correct shapes/no errors before spending GPU. Jobs 29464959
(committed) / 29464960 (nearest) launched.

**All 4 XAI jobs completed (IG rerun, GradientSHAP) -- full
triangulation table, plus a real correction to a previously-stated
conclusion.** Coastal decay ratio (1-2px/>9px) and NS-box open-water
enrichment, ALL numbers below on the fixed stratified sampling
(known_issues.md #57 P1), IG additionally on the fixed normalization
(known_issues.md #55):

| method | committed decay | nearest decay | committed NS-box enrichment | nearest NS-box enrichment |
|---|---:|---:|---:|---:|
| IG (mean/quant) | 15.26x / 15.04x | 8.07x / 8.74x | 16.43x / 16.82x | 4.83x / 5.31x |
| GradientSHAP (mean/quant) | 12.86x / 12.68x | 9.60x / 10.79x | 20.35x / 18.63x | 17.19x / 18.64x |
| GradCAM (mean/quant) | 2.33x / 2.27x | 1.20x / 1.21x | 6.43x / 6.28x | 1.64x / 1.64x |
| Occlusion | pending rerun (job 29465151) | pending rerun (job 29465152) | pending | pending |

**Correcting a previously-stated conclusion, found while re-running IG
with the sampling fix, not flagged by the user this time**: the earlier
"12.2x enrichment... survives the artifact control" framing
(known_issues.md #52, based on IG numbers computed WITH the sampling
bug -- 299/300 samples from 1985) is now superseded. With the fix, IG's
own number is actually higher (16.43x), not lower. **The real
correction comes from a direct raw-data check** (no model, no
attribution method, same pointwise-correlation methodology already used
for the Grand-Banks-vs-North-Sea check): NS-box open-water enrichment
in the RAW data is only **1.39x** (mean|r|=0.498 vs domain-wide 0.357)
-- an order of magnitude below what IG or GradientSHAP attribute to
this region on the committed model. **Corrected verdict**: a small,
real, raw-data-verified enrichment does exist (~1.4x, not zero), but
every gradient-based attribution method inflates it, IG and
GradientSHAP most severely (~12-15x overstatement on the committed
model). Notably, **`nearest`'s attributions land much closer to the raw
truth** than committed's (IG: 4.83x vs 1.39x truth, only ~3.5x off, vs
committed's 16.43x, ~12x off; GradCAM: 1.64x vs 1.39x truth, nearly
exact) -- a new, independent argument for `land_fill_mode=nearest`
beyond the coastal-decay-ratio reduction already documented: its
attributions are not just about a smaller artifact magnitude, they are
demonstrably closer to what the raw data actually supports.
**GradientSHAP is the outlier** -- it stays elevated (17-19x) even on
the nearest model, the least raw-data-consistent of the three methods
here; plausible reading (not verified further): GradientSHAP's
background-distribution-based baseline may carry its own bias distinct
from IG's zero-baseline sensitivity, worth a dedicated check before
citing GradientSHAP numbers on this specific question.

**Lesson restated, now demonstrated twice this session**: never cite a
bare attribution-method effect size as if it were the real physical
effect size without a raw-data cross-check -- this is exactly why item
1's original plan asked for "IG y otros XAI models... no solo IG": the
triangulation itself, not any single method, is what caught both this
correction and the earlier Grand-Banks-is-artifact finding.

**Occlusion rerun with the sampling fix lands (jobs 29465151/29465152)
-- the complete 4-method table, plus the raw-data ground truth:**

| method | committed decay | nearest decay | committed NS-box enrichment | nearest NS-box enrichment |
|---|---:|---:|---:|---:|
| IG | 15.26x / 15.04x | 8.07x / 8.74x | 16.43x / 16.82x | 4.83x / 5.31x |
| GradientSHAP | 12.86x / 12.68x | 9.60x / 10.79x | 20.35x / 18.63x | 17.19x / 18.64x |
| Occlusion | 4.25x / 4.31x | 1.65x / 1.84x | **0.50x / 0.53x** | **0.31x / 0.35x** |
| GradCAM | 2.33x / 2.27x | 1.20x / 1.21x | 6.43x / 6.28x | 1.64x / 1.64x |
| **Raw data (no model)** | **0.87x** | n/a (input-independent) | **1.39x** | n/a |

**Full picture, now genuinely complete**: all 4 methods agree on
DIRECTION for the coastal decay ratio (large, real, shrinks with
`nearest`) though disagree wildly on magnitude (1.65x-15x) -- the raw
data (0.87x, essentially flat) confirms this whole decay is
predominantly artifact regardless of which method measures it, exactly
as already concluded. **NS-box enrichment is more interesting**: IG,
GradientSHAP, and GradCAM all show real enrichment (>1x, from 1.6x to
20x) while **occlusion alone shows de-enrichment** (<1x, 0.3-0.5x) --
the only method that disagrees on the SIGN of the effect, not just the
magnitude. The raw truth (1.39x) sits between occlusion's
under-estimate and the other three's over-estimates, closest to
GradCAM's nearest-model number (1.64x). **One thing every method DOES
agree on**: switching zero->nearest reduces whatever the NS-box
enrichment metric shows, in every single method tested (occlusion
0.50->0.31, IG 16.4->4.8, GradientSHAP 20.3->17.2, GradCAM 6.4->1.6) --
a universal, convergent direction across 4 methodologically very
different attribution approaches, even though they disagree on the
absolute value. This is additional, independent support for
`land_fill_mode=nearest` producing less extreme/more consistent
attributions across the board, beyond the specific raw-data-closeness
argument already made for IG and GradCAM above.

This closes the XAI battery for fold0. Not yet done, lower priority:
extending any of this to folds 1-4 or the other 5 experiment families
(local/remote/lead-sweep) -- fold0's triangulation already answers the
scientific question (physical vs. artifact) the whole investigation was
for; further folds would mainly firm up precision on numbers already
directionally settled. **Update same day: user asked for exactly
this -- fold1/fold2 launched for both committed and nearest, all 4
methods, 16 jobs, results pending.**

---

## Tau-sweep / PR-curve analysis (Aug 22 2026) -- closes the one gap
## that conditioned the "trade, not Pareto" conclusion

User's own flag: the model-vs-persistence recall/precision comparison
used a SINGLE operating point (whatever threshold `q_pred`/`mean_pred`
crosses `thresh1` at) -- the "trade, not Pareto" framing rested entirely
on that one point, and there was a real possibility the model could
dominate persistence somewhere else on the threshold range, which would
flip the framing favorably for the paper.
`scripts/analysis/quantile_pr_curve_analysis.py` sweeps the full
precision-recall curve (sklearn `precision_recall_curve`/
`average_precision_score`) over both heads' continuous scores, purely
post-processing `eval_recall_v2_partition.py`'s already-saved npz --
zero new compute, no retraining, no GPU.

**Answer, checked exhaustively, not just at one point: the model NEVER
dominates persistence's single point, at ANY threshold, for ANY of the
7 families, either head, either ground-truth definition (def1 or
def2).** "Dominates" = a point on the model's curve with recall >=
persistence's recall AND precision >= persistence's precision
simultaneously -- checked directly, `False` in all 28 combinations (7
families x 2 heads x def1/def2). This is a STRONGER, not weaker,
version of the earlier "trade, not Pareto" conclusion: it's not just
that one arbitrary threshold trades recall for precision, the entire
achievable frontier sits below persistence's operating point for the
raw exceedance-detection metric. AUPRC per family (def1, quantile head):
full_lead7=0.321, local=0.596, remote=0.307, lead3=0.574, lead5=0.404,
lead14=0.207, lead30=0.170 -- decays with lead as expected, `local`
notably ahead of `full`/`remote` at the same lead=7 (consistent with
local information mattering, already seen elsewhere in this doc).
Figure: `experiments/figures/step7_persistence/
quantile_pr_curve_vs_persistence.png` (8-panel PR curves, def1, one per
family, persistence overlaid as a single point).

**This closes the gap cleanly**: the "recall/precision trade, not a
Pareto win" characterization of the model vs. persistence (documented
earlier this session) is now confirmed threshold-independent, not an
artifact of which single operating point was checked. No reframing to
"favorable" is supported by the data.

---

## Calibration-collapse diagnosis (Aug 22 2026) -- is it sigma or mean
## bias? User's second idea, also zero new compute

Decomposed the extreme-day coverage collapse (documented earlier this
session) entirely from already-saved `mean_c`/`q_c`/`trues_c` -- no
rerun needed, the implicit predicted spread (`q_c - mean_c`, since the
quantile head has no separate stored sigma) is directly comparable
against the actual gap needed (`trues_c - mean_c`) on extreme days:

| family | mean bias, normal days | mean bias, extreme days | implied spread | actual gap needed | spread/needed ratio |
|---|---:|---:|---:|---:|---:|
| full_lead7 | +0.089 | -0.491 | 0.434 | 0.491 | 0.883 |
| local | +0.005 | -0.294 | 0.429 | 0.294 | 1.458 |
| remote | +0.072 | -0.529 | 0.440 | 0.529 | 0.833 |
| lead3 | -0.013 | -0.331 | 0.380 | 0.331 | 1.150 |
| lead5 | -0.002 | -0.439 | 0.398 | 0.439 | 0.908 |
| lead14 | +0.022 | -0.721 | 0.519 | 0.721 | 0.721 |
| lead30 | +0.043 | -0.857 | 0.561 | 0.857 | 0.654 |

**Mean bias is near-zero on normal days but substantially negative on
extreme days everywhere** (-0.29 to -0.86°C) -- the mean head really
does underestimate severity specifically when things get extreme, as
expected (the model can't see the future intensification). But the
DIAGNOSTIC question is whether the quantile head's spread compensates
enough for this known blind spot: the spread/needed ratio answers this
directly, and it **decays with lead time** (0.883->0.654 from lead
3->30d, `local` the outlier at 1.458, already over-covering) --
confirming the coverage collapse is fundamentally a SPREAD/scale
inadequacy that WORSENS with lead, not an unfixable mean-bias problem
(the mean bias itself, while real, isn't growing disproportionately
faster than the spread already tries to compensate for -- it's that the
compensation itself falls short, increasingly so at longer leads).

**Tested the proposed cheap fix (single per-family scale factor on the
spread) directly**: sweeping `q_corrected = mean_c + c*(q_pred-mean_c)`
to find the `c` hitting 90% coverage on extreme days works cleanly in
every family (lands at 0.898-0.900 exactly) -- **confirms the collapse
is fixable in principle with pure post-hoc rescaling, no retraining**.
Required `c` ranges 1.49 (local, needs least correction) to 2.99
(lead30, needs most) -- consistent with the ratio table above (smaller
ratio -> larger correction needed). **But a single global scale
factor over-corrects normal-day coverage substantially** (rises to
96.6-100%, from an original ~85-93%) -- the same multiplier that fixes
extremes makes everyday predictions needlessly over-wide. **Honest
conclusion**: yes, it's a spread/scale problem, not mean bias, and yes
it's cheaply fixable in principle -- but a single constant scale factor
per family is not the right practical fix (it trades one miscalibration
for another). A conditional/isotonic recalibration (scale factor that
depends on how far `mean_c`/`q_c` already sit from typical, or on
`doy`/season, rather than one constant per family) would be needed to
fix extreme-day coverage without over-widening normal days -- not
implemented, this was a diagnostic pass per the user's own framing
("lo barato es diagnóstico"), not a request to ship a fix yet.

---

## Cross-fold check on the XAI triangulation (Aug 22 2026) -- was fold0
## representative? User's own instinct to check, all 16 jobs (fold1+
## fold2, both models, all 4 methods) completed cleanly

Real early signal before even running the XAI: `nearest`'s fold0
checkpoint never reached negative val_loss (best=+0.1399), while
fold1/fold2 both did (-0.0223 to -0.1557) -- fold0 was already suspected
to be a weaker-than-typical fold for `nearest` specifically. Ran the
same coastal-decay-ratio + NS-box-open-water-enrichment check
(mean head) on all 4 new fold/model combinations, 3-fold means ± std
below (fold0 included):

**Coastal decay ratio**, mean±std over 3 folds:

| method | committed | nearest | reduction |
|---|---:|---:|---:|
| IG | 14.80 ± 0.45 | 7.82 ± 2.59 | 47.2% |
| GradCAM | 1.74 ± 0.45 | 1.35 ± 0.30 | 22.5% |
| GradientSHAP | 24.54 ± 8.26 | 14.41 ± 3.43 | 41.3% |
| Occlusion | 5.33 ± 0.88 | 1.62 ± 0.17 | 69.6% |

**NS-box open-water enrichment**, mean±std over 3 folds:

| method | committed | nearest | reduction |
|---|---:|---:|---:|
| IG | 9.47 ± 5.18 | 6.39 ± 3.40 | 32.6% |
| GradCAM | 5.16 ± 1.34 | 1.91 ± 0.81 | 63.0% |
| GradientSHAP | 23.48 ± 2.40 | 15.29 ± 2.63 | 34.9% |
| Occlusion | 0.71 ± 0.16 | 0.38 ± 0.06 | 46.9% |

**Good news first**: the core comparative conclusion -- `nearest`
reduces both metrics relative to `committed` -- holds up robustly for
EVERY method on EVERY metric once averaged over 3 folds (reductions
22.5%-69.6% for decay, 32.6%-63.0% for enrichment), a stronger,
n=3-backed version of fold0's already-universal 4-method agreement on
direction. `committed`'s average enrichment (9.47x IG, 23.48x
GradientSHAP) and `nearest`'s (6.39x, 15.29x) both remain far above the
raw-data truth (1.39x) -- the "gradient-based methods overstate the
real signal, `nearest` overstates it less" conclusion is unchanged.

**Honest caveat, the reason to check more folds mattered**: fold0
alone was NOT a reliable point estimate for the exact magnitude on
either metric -- some real instability across folds:
- IG's NS-box enrichment for `committed` swung 16.43x (fold0) -> 7.97x
  (fold1) -> 4.01x (fold2) -- fold0 was actually the HIGH outlier, not
  typical (3-fold mean 9.47 ± 5.18, ~55% relative spread).
- GradientSHAP's decay ratio for `committed` swung 12.86x (fold0) ->
  30.40x (fold1) -> 30.35x (fold2) -- fold0 was the LOW outlier here,
  more than 2x below folds 1-2's consistent ~30x. GradientSHAP is the
  least fold-stable of the 4 methods on this metric.
- `nearest`'s NS-box enrichment via IG was also noisy: 4.83 -> 3.23 ->
  11.10 -- no consistent trend, fold2 alone would have suggested
  `nearest` is WORSE than committed's fold2 (4.01x) on this specific
  pairing, which the 3-fold averages correctly show is not the general
  pattern.
- Occlusion was the most fold-STABLE method throughout (smallest
  relative std on both metrics, both models) -- its qualitative
  "de-enrichment" (<1x) reading is the most trustworthy of the 4 in
  terms of fold-to-fold consistency, even though it's the outlier in
  absolute direction (below 1 vs the other three's above 1).

**Practical conclusion for future citation**: use the 3-fold means
above, not fold0 alone, if any of these numbers go in the paper --
fold0-only point estimates for the NS-box enrichment metric
specifically (the noisiest of the two) should not be treated as
precise. The qualitative/directional conclusions from the fold0-only
triangulation (land_fill=nearest reduces the artifact; the raw-data
check is the only fold-independent ground truth; GradientSHAP is the
least raw-data-consistent method) all survive this check unchanged.

**Synthesis figure (Aug 24 2026, separate session, requested after a
multi-day gap -- "analiza las tres carpetas de xai... graficas que
valgan la pena")**: no compiled visual existed for the 4-method
triangulation above, only these text tables -- built one from the
numbers in this section (not recomputed, this section's tables are
already twice-corrected and treated as settled), grouped bar chart, log
scale, both metrics, raw-data truth as a reference line on each panel.
`scripts/analysis/xai_triangulation_summary_plot.py` ->
`experiments/figures/xai_triangulation_summary/coastal_artifact_triangulation.png`.
CPU-only, seconds to run, no GPU spent.

**Structural note, not previously written down**: the 3 XAI folders are
not directly comparable at the per-variable level. IG and GradientSHAP
both save per-variable maps (`(5, 141, 201)`, one panel per input
variable); GradCAM saves a single combined `(141, 201)` map (it operates
on the last conv layer's pooled activations, upstream of the per-variable
input channels, so it structurally cannot attribute to one variable vs.
another); Occlusion saves no spatial map at all, only scalar
decay/enrichment summary numbers (`occlusion_summary.npz`). So
variable-level ranking can only be cross-checked between IG and
GradientSHAP; spatial-hotspot agreement can be checked across IG,
GradientSHAP, and GradCAM; the artifact-magnitude table (already in this
doc) is the only thing all 4 are commensurable on.

**GradientSHAP vs. IG variable-importance ranking -- disagree on #1, Aug
24 2026, computed from the already-saved `.npy` (no GPU)**: IG
(committed, fold0, mean_head): `ssr` 29.6% > `ptho_bot` 22.0% > `u10`
17.1% > `v10` 16.6% > `msl` 14.6%. GradientSHAP (same run): `ptho_bot`
**37.2%** > `ssr` 18.8% > `v10` 15.9% > `u10` 14.2% > `msl` 13.9%. Both
agree `msl` is least important and the other 3 (`u10`/`v10`/`ssr`) sit in
a similar mid-teens-to-20% band -- the disagreement is specifically about
whether `ptho_bot` or `ssr` is #1. This is not a mystery given the rest
of this doc: GradientSHAP's own coastal-artifact numbers are
systematically higher than IG's on the committed model (e.g. NS-box
enrichment 23.5x vs. 9.5x, this doc's triangulation table) -- GradientSHAP
inflates the same `ptho_bot` coastal-edge artifact more severely than IG
does, which mechanically pulls `ptho_bot` to the top of its
variable-importance ranking. Visually (`gradientshap_mean_head.png`):
`ptho_bot`'s panel is dominated by a single bright hotspot exactly on the
North Sea/British Isles coastline (same location flagged as artifact in
known_issues.md #52); `u10`/`v10`/`msl`/`ssr` are visibly noisier/more
speckled than IG's smooth signed maps (GradientSHAP uses unsigned
`|attr|` with a 16-sample noisy background baseline vs. IG's single fixed
zero baseline -- a real methodological difference in what "smooth" means
for each), but `v10` and `msl` both still show a clear, non-speckled
hotspot at the same North Sea/Scandinavia location that IG's differential
(`q_pred - mean`) map highlighted as the dominant driver of the
quantile-vs-mean divergence -- an independent method landing on the same
region for the same two variables is a real point of convergence, not
noted before this pass.

## Fixed conclusions for the `nearest` model, all 3 spatially-informative
## methods x 3 folds (Aug 24 2026, user's explicit request for "fixed"
## takeaways -- variable ranking, spatial hotspots, cross-method agreement)

**Scope**: `land_fill_mode=nearest` only (the adopted, current model) --
`committed` (`land_fill_mode=zero`) is the original/superseded checkpoint,
kept only as the artifact-comparison baseline throughout this
investigation, not a candidate for the paper's actual results. Folds 0-2
(the only folds with XAI run). Occlusion excluded here (no spatial map,
already covered in the triangulation table above).

**Variable importance (mean±std over 3 folds, % of total domain-summed
|attr|, mean_head)**:

| variable | IG | GradientSHAP |
|---|---:|---:|
| `ptho_bot` | 32.6% ± 9.2% | **47.7% ± 5.5%** |
| `ssr` | 19.6% ± 6.6% | 14.4% ± 2.0% |
| `u10` | 18.9% ± 7.0% | 13.2% ± 1.1% |
| `msl` | 15.1% ± 6.0% | 11.3% ± 1.4% |
| `v10` | 13.8% ± 1.6% | 13.3% ± 1.2% |

`ptho_bot` is #1 in both methods once averaged over folds (fold0 alone,
analyzed earlier, was misleading for IG specifically -- IG's fold0 number
had `ssr` on top, an outlier; folds 1-2 both put `ptho_bot` clearly first,
consistent with GradientSHAP all 3 folds). GradientSHAP is far more
fold-stable on this ranking (std 1.1-5.5%) than IG (std 1.6-9.2%).
**Caveat, do not drop when citing**: part of `ptho_bot`'s magnitude is
still residual coastal-edge artifact even under `nearest` (this doc's
triangulation table: `nearest`'s coastal decay ratio is still ~8x the
raw-data truth of 0.87x, down from committed's ~15-18x but not
eliminated) -- read this ranking as "what the model's attention is
dominated by," not a physical-importance ranking on its own.

**Spatial hotspots: North Sea box vs. Grand Banks/Gulf-Stream-separation
box, region mean|attr| / domain mean|attr|, 3-fold mean**:

| variable | IG: NS-box | IG: Grand Banks | GradientSHAP: NS-box | GradientSHAP: Grand Banks |
|---|---:|---:|---:|---:|
| `ptho_bot` | 2.55x | 1.48x | **5.53x** | 1.41x |
| `v10` | 2.35x | 1.18x | 2.08x | 1.30x |
| `msl` | 1.64x | 1.16x | 2.53x | 1.08x |
| `u10` | 1.57x | 1.09x | 3.80x | 1.05x |
| `ssr` | 0.92x | 1.29x | 2.49x | 1.31x |

GradCAM (combined, all-variable map, no per-variable split): NS-box
1.59x ± 0.45, Grand Banks 0.94x ± 0.04 (3-fold mean±std) -- same
direction as the other two.

**Fixed conclusions**:
1. **The North Sea box is enriched relative to the Grand Banks/Gulf-Stream
   region for every single variable, in all 3 spatially-informative
   methods (IG, GradientSHAP, GradCAM), consistently across all 3
   folds.** This is the most robust cross-method spatial finding in the
   whole XAI investigation -- three methodologically unrelated
   attribution techniques agree on direction every time, even though
   they disagree sharply on magnitude (GradientSHAP's ratios run
   1.5-3.6x higher than IG's, same pattern as the triangulation table's
   already-established "GradientSHAP amplifies more" finding).
2. **Grand Banks/Gulf-Stream separation is only mildly enriched at best
   (1.0-1.5x, i.e. barely above domain average) for every variable on
   the `nearest` model** -- confirms, independently of the coastal-decay-
   ratio analysis, that Grand Banks was never a real hotspot on the
   corrected model; its earlier prominence (fold0, `committed`, before
   the artifact was understood) was a `land_fill=zero` edge-artifact
   effect on `ptho_bot` specifically, not a genuine remote precursor
   signal, consistent with `known_issues.md`'s "do not cite Grand Banks"
   verdict.
3. **`ssr` is the one variable where the North Sea box is NOT enriched**
   (0.92x IG -- at or slightly below domain average) -- `ssr`'s
   attribution mass sits elsewhere (the Gulf-Stream/tropical bands
   described earlier in this doc), not locally. Every other variable
   (`ptho_bot`, `u10`, `v10`, `msl`) is consistently NS-box-enriched.
4. **Cross-method agreement is about direction, not magnitude** -- treat
   "NS box matters more than Grand Banks" as solid (3 methods, 3 folds,
   5 variables, zero exceptions to the direction), but do not average or
   directly compare the absolute ratios across methods for a paper
   number; report method-specific ranges or use the raw-data check
   (already the only fold-independent, method-independent ground truth
   in this whole investigation) for any number that needs to stand alone.

Script: computed directly from the already-saved `.npy` files listed
above (no GPU, no new attribution run) -- boxes are index slices
(`NS_LAT=100:127, NS_LON=150:187`; `GS_LAT=70:100, GS_LON=10:50`,
approx. lat 35-50N lon -75..-55, per `src/data/masking.py`'s NS
convention), not re-derived from scratch.

---

## Reframing "the model loses to persistence" (Aug 23 2026) -- user's
## insight: precursor-based forecast vs. state-based forecast are
## different questions, the raw comparison conflates them

The model was NEVER given `y(t)` (the target's own recent value) as
input -- only `ptho_bot` (a different, imperfect proxy) + atmosphere.
Persistence, by construction, uses `y(t)` directly. So persistence
winning on raw r isn't "ML fails a trivial baseline" -- it's a
different task entirely (forecast from external drivers alone vs.
forecast from the current state). Consistent with everything already
found: the deficit vs. persistence is UNIFORM across every phase
(no_mhw 0.80 vs 0.93 too, not concentrated at onset) -- exactly what a
missing key input predicts, not what bad training predicts.

**Idea 1 (free, decisive, done first) -- does q_pred add real
information beyond persistence?** `scripts/analysis/
incremental_value_regression.py`: OLS `y_true ~ persist + q_pred` (and
`~ persist + mean_pred`), all 7 families, pure post-processing of
`eval_recall_v2_partition.py`'s saved npz + a freshly-computed,
alignment-VERIFIED persistence series (max|diff|=2.4e-7 against the
saved `trues_c` in every family -- not assumed, checked).

| family | r_persist | r_hybrid (mean/quant) | beta(pred) p-value | partial corr (mean/quant) | r_trend (mean/quant) |
|---|---:|---:|---:|---:|---:|
| full_lead7 | 0.9369 | 0.9379 / 0.9377 | ~0 (both) | 0.124 / 0.107 | 0.204 / 0.180 |
| local | 0.9369 | 0.9395 / 0.9385 | ~0 | 0.197 / 0.156 | 0.250 / 0.198 |
| remote | 0.9369 | 0.9378 / 0.9376 | ~0 | 0.116 / 0.101 | 0.197 / 0.176 |
| lead3 | 0.9812 | 0.9814 / 0.9814 | ~0 | 0.110 / 0.098 | 0.140 / 0.127 |
| lead5 | 0.9597 | 0.9602 / 0.9601 | ~0 | 0.117 / 0.102 | 0.174 / 0.155 |
| lead14 | 0.8647 | 0.8672 / 0.8664 | ~0 | 0.131 / 0.110 | 0.260 / 0.231 |
| lead30 | 0.7398 | 0.7526 / 0.7486 | ~0 | 0.206 / 0.170 | 0.394 / 0.389 |

**Answer: yes, unambiguously.** `beta(q_pred)` is statistically
significant at essentially machine-precision p-values in EVERY family
-- the model adds real information beyond the state-based baseline, not
noise. Two clear, growing-with-lead patterns: (1) the fitted hybrid
`alpha*persist + beta*q_pred` beats pure persistence's r in every
single family, with the gap widening at longer leads (+0.001 at lead3
-> +0.013-0.017 at lead30) -- small in absolute r at short leads
(persistence is already near-ceiling there) but real and growing; (2)
partial correlation and the "trend skill" (idea 3: does `q_pred-persist`
track `y_true-persist`, i.e. does the model anticipate WHEN persistence
will be wrong) both grow with lead too (partial corr 0.10-0.12 at short
leads -> 0.17-0.21 at lead30; trend skill 0.13-0.18 -> 0.39 at lead30)
-- `local` again stands out at lead=7 (partial corr 0.197 vs
full=0.124, remote=0.116), consistent with local information mattering
throughout this whole investigation, not just for coastal IG.
`R^2_full - R^2_persist_only` is small in absolute terms at short leads
(+0.002 at lead3) but reaches +0.019 at lead30 -- modest-looking numbers
that are nonetheless highly significant given n~14500-14600.

**Idea 4 (linear ceiling) -- and a genuinely surprising, slightly
uncomfortable result**: `scripts/analysis/linear_ceiling_ridge.py`,
ridge regression on NS-box-mean time series (5 vars x 60-day window =
300 features, NOT the full spatial field -- that would be ~8.5M
features per sample, hopeless overfit at n~14500) computed directly
from the raw netCDF via xarray (cheap, no LazyDataset tensor loading),
same 5-fold stratified_kfold split, pooled test r exactly like the
CNN's own reporting convention. **Result: pooled r = 0.8797** -- HIGHER
than the CNN-LSTM's own pooled r for `full_lead7` (0.8141 mean head /
0.7981 quantile head). A simple linear ridge on spatially-averaged
precursor summaries outperforms the full nonlinear spatial CNN-LSTM on
raw point-forecast correlation. Per-fold r: 0.857/0.919/0.916/0.796/
0.853 -- reasonably consistent across folds, no single-fold fluke.

**What this means, held carefully, not oversold**: this does NOT mean
the CNN pipeline has no value -- the linear ridge has no quantile head,
no exceedance-detection capability, no spatial attribution story, and
cannot do anything for the def1/def2 recall results (the project's own
"decisive" finding). It DOES mean there is real headroom being left on
the table by the CNN's mean-forecast pathway specifically -- plausible
contributing factors: the CNN's well-documented GNLL variance-collapse
training instability (best checkpoint often reached very early, e.g.
`nearest` fold0's best at epoch 6, val_loss never went negative) vs.
ridge's clean convex optimum with no such instability; and/or the full
spatial field genuinely adding more noise than signal relative to a
NS-box-focused summary for this specific point-forecast task. **This
substantially raises the expected payoff of idea 2** (retrain a hybrid
model with `y(t)`/its recent history as an explicit input) -- if a
naive linear box-mean summary already beats the CNN by this margin,
giving the network direct access to the state variable (which the
linear ridge does NOT have either, yet the ridge still wins) suggests a
properly-built hybrid could plausibly beat persistence outright, not
just the current CNN. Not yet built -- next natural step if the user
wants to pursue this reframe into an actual architecture change.

**Reframe for the paper (idea 5, no computation, framing only)**:
precursor-only forecasting matters operationally when no reliable
surface/state observation exists (cloud cover, reanalysis gaps, forecast
initialization) or for mechanistic attribution (which precursors matter,
independent of autocorrelation) -- persistence was never a fair
competitor for that question, it was the wrong bar for what the model
was actually built to do. The uniform-across-phases deficit, the
significant incremental value of q_pred beyond persistence at every
lead, and the growing partial correlation/trend-skill with lead time
are all consistent with "the model is doing a harder, different task
reasonably well," not "the model failed to learn." The paper's headline
does not have to stay negative -- "precursor information adds
significant, growing-with-lead skill beyond persistence" is a genuine
positive claim, backed by significant beta coefficients, not a reframe
of convenience.

## Aug 24 2026 — event_detection figure extended across the full lead-time sweep

User's most recent instructions (two messages) set a definitive priority
order for tonight: (1) spatial pipeline — thorough multi-agent bug audit,
then launch 2 folds; (2) the `event_detection_pod_far_csi.png` figure,
extended across the full lead-time sweep — user called this "quiza el
principal resultado del paper" and asked for "mas lead times y mas
informacion"; (3) hybrid model — deprioritized, only if time remains.

Generalized `scripts/eval_event_detection.py` from a hardcoded
single-family (`full_gnll_quantile_v2`, zero-fill, lead=7) script to a
reusable `--config_dir`/`--label`-parametrized tool, following the exact
pattern already used this session for `eval_recall_v2_partition.py` and
`eval_onset_skill_quantile_v2.py`. `LEAD` is now read dynamically from
the target config's own `lead_time` (not hardcoded), avoiding the exact
`NameError` regression made earlier this session when generalizing
`eval_onset_skill_quantile_v2.py` the same way.

While generalizing, found and fixed a real bug before launch (known_issues
#60): the script called `forward_with_quantile()` directly, bypassing the
masking that `train_partition.py`'s `LocalOnlyLightningModule`/
`RemoteOnlyLightningModule` apply only inside their step methods — the
local/remote checkpoints would have been evaluated on unmasked,
out-of-distribution input. Added `--mode {full,local_only,remote_only}`
using the same `src/data/masking.py` functions and the same pattern
`eval_recall_v2_partition.py` already established for this exact failure
mode. Caught by tracing `train_partition.py` before submitting, not by a
failed run.

Launched `scripts/slurm/submit_event_detection_all_families.sh`
(job 29526761, array 0-6, CPU, `--mail-type=END,FAIL`) covering all 7
families: full_lead7 (nearest-fill), local, remote, lead3/5/14/30
(all nearest-fill). Verified beforehand that all 35 (7 families x 5
folds) checkpoints exist. Results pending — will compile a POD/FAR/CSI +
bootstrap-CI comparison table/figure across all leads once the array
finishes, plus the lead-time-distribution histograms per family.

Spatial-pipeline audit (priority 1) in progress in parallel via 3
background agents (data/split, model/eval, Raven-migration readiness).
Raven-migration-readiness agent finished first: draft SLURM script
written to `scripts/slurm/submit_spatial_tbotatm_folds0_1.sh` (not
submitted). Three small, purely mechanical blockers identified, none of
them logic bugs: (1) `DATA_FILE` in `src_spatial/dataset_spatial.py` (and
matching constants in 3 other files) is a hardcoded JUWELS path, not
env-var-driven like the scalar pipeline — one-line-per-file fix to read
`MHW_DATA_FILE`; (2) `configs/spatial/TbotAtm_fold0.yaml`'s `output_dir`
needs repointing to a Raven path (its `data_dir` field is dead config —
`dataset_spatial.py` never reads it, only the hardcoded constant, so
editing the YAML's `data_dir` alone would silently do nothing); (3) no
`configs/spatial/TbotAtm_fold1.yaml` exists yet — needs to be created
for the 2-fold launch. No missing pip packages (torch/lightning/xarray
all present in venv; ConvLSTM is hand-implemented in `model_spatial.py`,
no third-party dependency). MLD-weights prerequisite only blocks the phys
variant (not tonight's standard-model job) and doesn't exist on Raven
yet. Awaiting the other two agents' logic-bug findings before applying
any fixes or launching.

## Aug 24 2026 — spatial pipeline: audit complete, fixes applied, launched on Raven

All 3 background audit agents (data/split, model/eval, Raven-migration
readiness) reported back. Synthesized findings, all fixes mechanical (no
design changes needed):

- **F1 (migration-readiness + data/split agents, blocking)**:
  `src_spatial/dataset_spatial.py` hardcoded a JUWELS `DATA_FILE` constant,
  never read from config despite every `configs/spatial/*.yaml` already
  having a (dead) `data_dir` field. Fixed by wiring `data_dir` straight
  from the resolved config inside `SpatialDataset.__init__` — matches the
  scalar pipeline's actual convention (`LazyDataModule`/`LazyDataset` also
  read `data_dir` from the YAML directly, not an env var at dataset-load
  time; `MHW_DATA_FILE` is a separate convention used by `src/utils/paths.py`
  for other standalone scripts). `configs/spatial/TbotAtm_fold0.yaml`'s
  `data_dir`/`output_dir` repointed to Raven paths.
- **F2 (data/split agent)**: no `configs/spatial/TbotAtm_fold1.yaml`
  existed. Created as a copy of fold0 with `fold: 1` and a distinct
  `output_dir`.
- **F3 (data/split agent, real bug, not just a documentation gap as the
  Aug 16-17 audit's NF-S-3 had characterized it)**: `train_spatial.py`'s
  `build_splits()` val-year shuffle used `rng(seed)` alone — since
  `remaining` shares `n_folds-2` of its `n_folds-1` year-blocks between
  any two folds, reshuffling with the same seed produced near-identical
  `val_years` (measured 83-100% pairwise overlap on real data), same bug
  class as known_issues #1/#42's kfold val_years collision. Fixed:
  `rng(seed + fold)`. **Verified directly against real data post-fix**:
  fold0 val_years=[1990,1999,2004,2015,2016,2021], fold1
  val_years=[1987,1992,1997,2002,2012,2020] — zero overlap, test_years
  properly disjoint too.
- **F4 (data/split agent, new finding, real bug)**: `compute_stats()`
  masked every input variable to the SST ocean mask uniformly, but
  `__getitem__` only spatially masks `ptho_bot` (to `land_mask_tbottom`)
  and never masks the ERA5 variables at all (u10/v10/msl/ssr aren't in
  `ocean_variables` for TbotAtm) — the model sees their full domain
  (land+ocean) values every timestep, but their normalization stats were
  computed ocean-only. Measured effect on real data: land-region wind
  variance ≈50% of ocean-region variance, so wind/pressure/radiation
  channels' stds were inflated ~16-17% (u10 ocean-only std=4.469 vs
  correct full-domain 3.842, etc.) — biasing every land pixel's
  normalized input. Fixed to mirror `__getitem__`'s exact per-variable
  masking logic. **Verified directly**: post-fix stds (u10=3.855,
  v10=3.749, msl=750.9, ssr=8.586) now match the agent's independently
  measured full-domain values almost exactly.
- **F5 (data/split agent)**: `train_spatial.py` never printed train/val/
  test year lists (only counts) and never dumped a resolved config —
  violates the user's standing rule ([[feedback_output_resolved_config_and_splits]]).
  Fixed: added year-list printing to `build_splits()` and a
  `resolved_config.yaml` dump in `main()`, matching `train_partition.py`'s
  existing convention exactly.
- **NF-S-5 (model/eval agent, HIGH, still unfixed, deliberately deferred)**:
  `scripts_spatial/eval/mhw_onset_skill.py` uses `tgt>0`/`to_t>0` instead
  of proper per-pixel Hobday p90+persistence — every existing onset/
  mid-event spatial map is invalid. A full fix was designed (regrid
  `sst_climatology_doy.nc`'s `p90_thresh` field onto the spatial model's
  coarser/offset grid via the same `interp(...,fill_value="extrapolate")`
  pattern already used in `scripts/mhw_hobday_stats.py` and
  `scripts/analysis/calibrate_mhw_area_threshold.py`, then run the
  existing scalar `apply_hobday()` per-pixel — benchmarked on real data
  at ~74s one-time CPU cost for 18,296 ocean pixels, giving 12.55% mean
  MHW fraction on a sample vs. the buggy 31.5%, a strong sanity check the
  design is correct). **Not implemented tonight** — real design/testing
  effort, and the script also hardcodes `N_FOLDS=5` so it can't run
  meaningfully against only 2 trained folds regardless. Do not trust or
  regenerate any spatial onset-skill figure until this fix lands AND all
  5 folds exist.
- **model_spatial.py itself, checkpoint selection, `persistence_baseline_spatial.py`,
  `dataset_spatial_phys.py`, NF-S-1/NF-S-2/NF-S-6/NF-S-7**: all confirmed
  either clean or correctly isolated from tonight's code path (model/eval
  + data/split agents, cross-checked independently).

Smoke-tested both fixes directly against real data before spending GPU
budget (dataset load + splits + compute_stats + one `__getitem__` call
for fold0, plus a splits-only comparison for fold0 vs fold1) — all
matched expectations, no surprises deferred to the actual GPU run.

Launched `scripts/slurm/submit_spatial_tbotatm_folds0_1.sh`
(job 29527143, array 0-1, GPU, 12h budget, `--mail-type=END,FAIL`) —
first-ever Raven run of the spatial pipeline. Confirmed both tasks
progressed cleanly past dataset load/splits/resolved-config-dump/
normalization stats within the first minute (checked stdout directly,
not assumed).

Priority order for tonight (per user's explicit instructions) is now:
(1) spatial — DONE (audited, fixed, launched); (2) event_detection
lead-time sweep — DONE (launched, job 29526761, pending results);
(3) hybrid model — next, since 1 and 2 are both launched and only
awaiting results.

## Aug 24 2026 — hybrid model launched (priority 3, both higher-priority items already launched)

Created `configs/partition/full_gnll_quantile_v2_landfill_hybrid/fold{0,1}.yaml`
-- identical to `full_gnll_quantile_v2_landfill/fold{0,1}.yaml` in every
respect except `use_state_feature: true` and a `_hybrid` output_dir suffix,
for a clean A/B comparison against that already-trained pair. Justification
(already established): the zero-cost post-hoc OLS hybrid computed the
realistic r-gain ceiling (+0.001-0.013 by lead); this GPU retrain is
justified only for joint uncertainty (GNLL with y(t) as an explicit input
channel) or to check whether the post-hoc linear stack leaves nonlinear
value on the table.

Smoke-tested the full pipeline end-to-end on real data before submitting
(`--fast_dev_run 1`, fold0): dataset load, stratified_kfold split
(train=24/val=8/test=8 yrs, correctly printed with MHW-day counts per the
standing rule), normalization stats, 4.2M-param model construction,
train/val/test steps all completed without error with
`use_state_feature=true` wired through (single-batch metrics are
meaningless at fast_dev_run=1, only used to catch wiring bugs).

Launched `scripts/slurm/submit_gnll_quantile_v2_landfill_hybrid_folds0_1.sh`
(job 29527399, array 0-1, GPU, 12h budget, `--mail-type=END,FAIL`).

All 3 of tonight's priorities are now launched and running/pending in
parallel: spatial (job 29527143), event_detection lead-time sweep
(job 29526761), hybrid model (job 29527399). Next steps once each
completes: compile the event_detection POD/FAR/CSI comparison across all
7 lead/masking families; check the spatial pipeline's first-ever Raven
training curves for sanity; evaluate whether the trained hybrid beats the
linear post-hoc ceiling (r_hybrid≈0.9379 at lead7, ≈0.7526 at lead30).

## Aug 24 2026 — event_detection lead-time sweep: results compiled

Job 29526761 (all 7 families) completed cleanly (COMPLETED, exit 0,
~33-38min each). Compiled via new `scripts/analysis/
event_detection_lead_time_comparison.py` (reads-only, no recomputation)
into `experiments/figures/step7_persistence/event_detection_lead_time_comparison.png`
and `.md`.

**Headline finding — this IS the strengthened version of "quiza el
principal resultado del paper"**: quantile_head's POD significantly
exceeds persistence's POD (event-paired bootstrap, 95% CI excludes 0) at
EVERY lead time tested (3, 5, 7, 14, 30 days) and in both the local-only
and remote-only partition experiments — 7/7 families significant, not
just the original single lead=7 result.

| Family | POD(quantile) | POD(persist) | Δ (bootstrap 95% CI) | Sig? |
|---|---|---|---|---|
| lead=3d | 0.679 | 0.071 | +0.607 [+0.464,+0.732] | YES |
| lead=5d | 0.679 | 0.125 | +0.555 [+0.411,+0.696] | YES |
| lead=7d (full) | 0.679 | 0.179 | +0.501 [+0.375,+0.625] | YES |
| lead=14d | 0.554 | 0.286 | +0.268 [+0.089,+0.446] | YES |
| lead=30d | 0.589 | 0.375 | +0.217 [+0.071,+0.375] | YES |
| lead=7d, local-only | 0.768 | 0.179 | +0.587 [+0.464,+0.714] | YES |
| lead=7d, remote-only | 0.589 | 0.179 | +0.411 [+0.268,+0.571] | YES |

**Why "7 days easier than 1" resolves cleanly now**: quantile_head's own
POD is essentially flat (0.55-0.68) across all 5 leads — it does NOT get
easier for the model as lead grows. What changes is persistence's POD,
which rises monotonically with lead (0.071 -> 0.375) because a longer
`[onset-lead, onset]` scoring window makes it progressively easier for
an already-elevated pre-onset state to cross threshold at some point in
that wider window — a scoring-window artifact of the event-detection
definition, not evidence the forecasting task itself gets easier. This
is why the model-vs-persistence GAP shrinks with lead even though the
model's own absolute skill doesn't degrade.

**New finding this sweep surfaced**: local-only (NS box only, no remote
input at all) achieves the single highest POD of any condition tested
(0.768) — higher than the full model (0.679). Remote-only alone still
significantly beats persistence (0.589 vs 0.179), confirming precursor
information beyond the NS box's own local persistence adds real
event-detection skill, not just local autocorrelation. mean_head remains
far worse than quantile_head everywhere (e.g. 0.125 vs 0.679 at lead=7),
consistent with the quantile head being specifically what catches
onsets.

Added to `results/all_results.csv`
(`event_detection_lead_time_sweep`).

## Aug 24 2026 — spatial pipeline: first-ever successful Raven training run

Job 29527143 (folds 0-1) completed cleanly (COMPLETED, exit 0; fold0
3h55m, fold1 4h13m — well within the 12h budget; ~23.3GB peak RSS per
node, both nodes). This is the first time the spatial (2D ConvLSTM)
pipeline has ever trained successfully on Raven, after the Aug 24
3-agent audit and 5 fixes (known_issues.md #61-63).

**Test results** (per-fold, `logger=False` on the test Trainer means
these only appear in stdout, not `metrics.csv` — extracted directly from
the SLURM log):
- fold0: test_loss=0.4593, test_mean_r=0.6385
- fold1: test_loss=0.5262, test_mean_r=0.6550

Both folds show consistent, meaningfully positive pixelwise mean
Pearson r (~0.64-0.66) between predicted and true 2D SST anomaly field
at lead=7 days — a real, working model, not a degenerate/broken run.
Val_loss trajectories are sane: best val_loss at epoch 3 (fold0,
0.4701) / epoch 6 (fold1, 0.5153), then monotonic overfitting until
EarlyStopping(patience=30) triggered — exactly the expected shape, not
a red flag.

Re-verified (from the actual full-run logs, not just the pre-launch
smoke test) that both audit fixes hold under real training: resolved
split years printed and saved correctly (train=26/val=6/test=8 yrs each
fold, `resolved_config.yaml` written to both output dirs), and fold0/
fold1 val_years remain fully disjoint
([1990,1999,2004,2015,2016,2021] vs [1987,1992,1997,2002,2012,2020]) —
confirming the F3 seed+fold fix worked in production, not just in the
isolated smoke test.

**Not done this session** (explicitly deferred, tracked in
known_issues.md #64 / audit_plan.md): no onset-skill evaluation
(`mhw_onset_skill.py`, still has the unfixed NF-S-5 bug and requires
all 5 folds); no persistence-baseline comparison run yet (safe to run,
just not done). Both are reasonable next steps once time allows, but
were not part of tonight's "solo lanzar" scope.

## Aug 24 2026 — hybrid model (state_feature) results: retrain gain far exceeds the post-hoc ceiling estimate

Job 29527399 (folds 0-1) completed cleanly (COMPLETED, exit 0, no
errors in either fold's stderr, checkpoint directories clean/
uncontaminated). Same-fold, same-model-family comparison against the
already-trained `full_gnll_quantile_v2_landfill` baseline (test_corr,
`ckpt_path="best"` in both cases, so this is a fair like-for-like
comparison):

| Fold | Baseline test_corr | Hybrid test_corr | Δ |
|---|---|---|---|
| fold0 | 0.8313 | 0.9181 | +0.0868 |
| fold1 | 0.8838 | 0.9457 | +0.0619 |

**This is a real, substantial, and surprising result — the gain is far
larger than the pre-launch expectation.** The decision to launch this
retrain was explicitly justified by the zero-cost post-hoc OLS hybrid
(`alpha*persist + beta*q_pred`, no retraining) as a "realistic ceiling":
that ceiling estimated only +0.001 to +0.013 in r depending on lead
(`incremental_value_over_persistence` in `results/all_results.csv`,
full_lead7: 0.9369->0.9379). The actual end-to-end retrain, giving the
network explicit access to y(t) as an input feature (not just an
after-the-fact linear blend of two already-fixed predictions), gained
+0.062 to +0.087 — roughly 5-9x the "ceiling" the cheap version
predicted. This means the earlier reasoning ("if the post-hoc version
doesn't justify the claim, the retrain is unlikely to beat it by much")
was WRONG in this case: the nonlinear network extracts substantially
more value from direct access to the current state than a simple linear
post-hoc combination of two independently-trained predictions can
capture — plausibly because state_feature lets the network condition
its whole nonlinear forecast (not just a final linear blend) on y(t),
including interactions with the precursor fields themselves. Worth
flagging explicitly as a correction to the pre-launch expectation, not
quietly updating the number.

**Not yet done**: full pooled-5-fold evaluation (only 2/5 folds trained,
matching the user's explicit 2-fold launch scope), quantile-head/def1/
def2 recall comparison against the baseline's already-computed numbers,
and IG/attribution check on whether the state feature is actually what's
driving the gain (vs. some other training difference between the two
runs) — recommended as next steps, not done tonight.

## Aug 24 2026 — hybrid model: does it actually detect MORE MHW events? (launched, pending)

User's pushback on the test_corr-only result above, and correctly so:
r_persist is already ~0.94 at lead=7 (near-ceiling), so a model with
direct access to y(t) can look like a better-correlated smoothed
persistence on ordinary days while doing WORSE at the thing that
actually matters -- catching the onset jump itself (the quantile head's
job, not the mean head's; POD(quantile)=0.679 vs POD(mean)=0.125 at
lead=7 in the baseline, per the event_detection_lead_time_sweep entry
above). +0.087 test_corr does not establish this either way. User also
flagged, independently and correctly, that folding the hybrid straight
into the existing 4-method XAI triangulation next would be premature --
a near-deterministic y(t) input will likely dominate IG/gradient/
permutation attribution and swamp the precursor-field signal the XAI
work is actually about, so that's deferred until event-level skill is
checked.

Generalized `scripts/eval_event_detection.py` (not launched as-is --
audited its assumptions against the hybrid model before running):
  1. `--folds` argument (default all 5, comma-separated to override) --
     the hybrid only has folds 0,1 trained, so the existing hardcoded
     `range(N_FOLDS=5)` would have failed on `best_ckpt()` finding no
     checkpoint dir for folds 2-4.
  2. `use_state_feature` batch handling -- confirmed from
     `src/data/dataset.py` `__getitem__` (line ~502) that
     `use_state_feature=true` configs return a 4-tuple
     `(x_spatial, x_temporal, y, x_state)`, not 3. The script's loop now
     extracts `batch[3]` as `x_state` when `cfg["use_state_feature"]` is
     set and threads it into `forward_with_quantile(xs, xt, x_state)` --
     previously omitted, which would have hit the model's explicit
     `ValueError` (`state_feature=True but x_state is None`, cnn_lstm.py
     `_encode()`, "no silent fallback" convention) on the very first
     hybrid batch.

Both changes verified by reading the actual source (dataset.py
__getitem__ return order, CNNLSTMModel.forward_with_quantile signature,
load_model_config's model_config.json-first behavior) rather than
assumed -- no live smoke run first this time (an interactive
`--folds 0` dry run was started but killed after >60s with no output
yet, consistent with this pipeline's known slow CPU-only netCDF/dataset
load rather than a hang; correctness was established by code reading
instead of waiting it out, per the user's standing preference against
babysitting jobs with polling -- SLURM's `--mail-type=END,FAIL` is the
notification path).

Wrote `scripts/slurm/submit_event_detection_hybrid_vs_baseline.sh`
(array 0-1, CPU, `small` partition, 1h budget): index 0 = hybrid
(`full_gnll_quantile_v2_landfill_hybrid`, `--folds 0,1`), index 1 =
baseline (`full_gnll_quantile_v2_landfill`) evaluated on the SAME
fold subset 0,1 -- not the existing 5-fold `full_lead7` summary --
so the comparison isn't confounded by per-fold differences in event
count/difficulty. Submitted: job 29551714
(`sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de`).

**Next steps once it lands**: compare POD/FAR/CSI (quantile_head vs
mean_head vs persist) between the two `event_detection_summary_*.json`
outputs (`hybrid_folds01` vs `baseline_folds01`) -- the real answer to
"does it detect more MHW" is whether hybrid's quantile-head POD beats
baseline's quantile-head POD on the matched fold subset, not the
test_corr delta. Only then decide whether the hybrid is worth carrying
into the XAI triangulation.

## Aug 24 2026 — hybrid model: event-detection results land, +0.087 test_corr does NOT translate into a clear event-detection gain

Job 29551714 (array 0-1) completed cleanly, both tasks exit 0, ~13min
each. Same 21 test events in both runs (fold0 n_events=11, fold1
n_events=10 -- identical between hybrid and baseline, confirming the
matched-fold-subset comparison is clean: same test years, same
underlying MHW events, only the model differs). `persist`'s POD is
byte-identical between the two runs (0.143, hits=3) as a sanity check --
persistence doesn't depend on either model, so this had to match, and
it does.

| System | Baseline POD | Hybrid POD | Δ hits |
|---|---:|---:|---:|
| quantile_head | 0.429 (9/21) [0.218,0.660] | 0.524 (11/21) [0.298,0.743] | +2 |
| mean_head | 0.000 (0/21) [0.000,0.161] | 0.095 (2/21) [0.012,0.304] | +2 |
| persist | 0.143 (3/21) | 0.143 (3/21) | 0 |

**Not statistically distinguishable from noise.** Fisher's exact test on
the quantile_head 9/21 vs 11/21 table: p=0.758. The two Clopper-Pearson
95% CIs overlap almost entirely ([0.218,0.660] vs [0.298,0.743]). With
only 21 events pooled from 2 folds, a 2-hit difference is well within
sampling variation -- this result neither confirms nor rules out a real
event-detection gain, it's simply underpowered to say either way at
this fold count.

**This directly answers the user's question from earlier tonight** ("does
this detect more MHW, or is the +0.087 test_corr just from ordinary
days") -- the honest answer is: it might detect slightly more, but the
data in hand can't support that claim with any confidence, and the
correlation gain is clearly NOT accompanied by an equally dramatic
event-detection gain the way the test_corr numbers alone would suggest.
The user's skepticism about the correlation gain (smoothed-persistence
hypothesis: a model with y(t) as input can look better on ordinary days
without getting better at the onset jump) is at minimum not
contradicted by this result, and the magnitude mismatch (+0.087 r vs
+0.095 POD, not even the same kind of quantity but illustratively both
"small" relative to what a genuine step-change in skill would look
like) is consistent with it.

**FAR remains very high in both** (baseline 0.871, hybrid 0.823) -- the
hybrid does not fix the false-alarm rate, if anything it's marginally
better but still >80%. **mean_head stays essentially non-functional in
both** (0 vs 2 hits out of 21) -- all real event-detection skill in
both systems comes from quantile_head, consistent with everything
established previously about the two heads' roles.

**Decision: hold off on folding the hybrid into the 4-method XAI
triangulation.** No event-detection evidence currently justifies
spending that effort on this checkpoint over the baseline, and the
state_feature input carries the specific risk (flagged before this eval
ran) of dominating attribution and obscuring the precursor-field signal
the XAI work exists to characterize. Options going forward, not decided
yet: (a) stick with the baseline for XAI, treat the hybrid as a
correlation-only side result; (b) train the remaining 3/5 folds first
to get a properly powered (n_events~50+ pooled) event-detection
comparison before deciding either way.

Added to `results/all_results.csv`
(`hybrid_event_detection_folds01_result`).
