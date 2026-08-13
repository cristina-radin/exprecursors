# Audit plan — pre-supervision (2 weeks)

## Goal
Reconstruct a reliable narrative + results table for the supervision
meeting in 2 weeks. Chronological audit, oldest first. Phase 1 scripts
(March 2026, supervisor-reviewed) are the style/structure template.
Home memory (~/.claude/.../memory/) is resolved IN PARALLEL with each
phase, not before or after — each memory doc is tied to the code era
it documents.

## Reference documents (read first, always up to date)
- docs/known_issues.md — the 14-point bug checklist. Add new entries here,
  never delete without discussion.
- docs/data.md — verified pipeline + variable definitions
- CONTRIBUTING.md — process
- docs/narrative.md — CURRENT scientific state. Only what's confirmed
  post-audit goes here — nothing from home memory gets copied in without
  being re-verified against the current code first.

## Status labels (code)
CONFIRMED | NEEDS_RERUN | ARCHIVE | (Phase 1 only:) STYLE_TEMPLATE

## Status labels (memory docs)
KEEP | KEEP_AS_HISTORICAL | UPDATE | ARCHIVE

---

## Phase 1 — March 2026 (baseline, supervisor-reviewed)

### Code
- [x] src/data/dataset.py — **CONFIRMED + STYLE_TEMPLATE**
      Issues checked: all 18. Clean on #1 (target def), #2 (is_land = land_mask==0
      correct), #13 (no forbidden file).
      Issue #11 instance: line 174-175, if a variable listed in
      `detrend_variables` is absent from `self.data`, the code prints a
      warning and skips it instead of raising a ValueError. That is a
      misconfiguration — the variable name is wrong in the config — and
      should fail loudly. Not a correctness bug if `detrend_variables` is
      always configured correctly.
      #18: "# Vectorised OLS" comment at line 188 misleading — actual
      implementation is a pixel-wise double loop (NF-1).
- [x] src/data/datamodule.py — **CONFIRMED + STYLE_TEMPLATE**
      All 18 issues clean. Block-year split correctly uses TARGET year
      (line 63: i + window_size - 1 + lead_time) — no data leakage.
- [x] src/models/cnn_lstm.py — **CONFIRMED + STYLE_TEMPLATE**
      #11 (NF-3): line 226 `if loss_fn=="MAELoss" else nn.MSELoss()` silently
      defaults to MSE for any unrecognised string (confirmed live: "GNLLLoss"
      → MSELoss, no error). Does not affect stored results if loss_fn was
      always "MSELoss" or "MAELoss" during training.
- [x] src/xai/grad_cam.py — **CONFIRMED + STYLE_TEMPLATE**
      #8 (NF-4): `analyze_gradcam()` lines 187-195 — full test-set prediction
      scan at batch=1 before selecting top-N. GradCAM backward is inherently
      per-sample; the collection loop is the fixable part.
- [x] src/xai/utils.py — **CONFIRMED + STYLE_TEMPLATE**
      12 lines. All 18 issues N/A. Trivially clean.
- [x] scripts/train.py — **CONFIRMED + STYLE_TEMPLATE**
      #17 (NF-6): line 111 WandbLogger entity "hereon-ksn-expercursors"
      hardcoded — should come from config/env.
      NF-5 (cosmetic): LossCurvePlotCallback ylabel hardcodes "MSE loss"
      regardless of configured loss_fn.
- [x] scripts/run_xai.py — **ARCHIVE** (pending Phase 2 confirmation that
      run_xai_ensemble.py covers its functionality)
      Crash bug (issue #15 / NF-7): line 517 `full_ds.tierra_mask` →
      AttributeError; correct attr is `full_ds.is_land`. NOT fixed here —
      fix applied pre-audit to the 4 Phase 2/4 scripts that carry this
      functionality forward (see Phase 2 and Phase 4 notes below).
      #16 (NF-9): line 310 `temporal_features=3` hardcoded; with strict=False
      load, a mismatch silently leaves params randomly initialized.
      #8 (NF-8): `collect_predictions()` lines 46-63 unbatched full-dataset loop.

### Memory
(none from this era — Phase 1 predates most memory docs)

**Status: COMPLETE — 6 CONFIRMED, 1 ARCHIVE**

---

## Phase 2 — April 2026

### Pre-audit fix applied (Aug 2026)
`scripts/run_xai_ensemble.py` line 253 and `scripts/composite_ig.py` line 150:
`tierra_mask` → `is_land`. Crash bug (issue #15) inherited from run_xai.py
(Phase 1 ARCHIVE). Applied before Phase 2 audit to prevent AttributeError
from blocking further runs. Audit of the rest of these scripts not yet done.

### Data file provenance finding (Aug 13 investigation)
The April 24 ensemble job (SLURM 13759273) used `merged_daily_deepSST.nc`
(now `merged_daily_deepSST_OLD.nc`), where `ptho_bot` was raw ICON-COAST
absolute temperature — not a DOY-mean anomaly. The preprocessing fix was
applied June 29, 2026 (job 14070999), when `ptho_bot` DOY climatology
subtraction was first added to the pipeline and `merged_daily.nc` was
rebuilt as a single unified file.
Consequence: all April 2026 TbotAtm results (skill numbers, IG attribution
patterns, the "T_bottom dominance" XAI finding) were produced by a model
trained on a fundamentally different variable. Documented as known_issues.md
#21. Do not cite any April 2026 TbotAtm result without this caveat.

### Code
- [x] scripts/ensemble_skill.py — **NEEDS_RERUN**
      Issues: #9/#19 (lines 235-236: `to_anom > 0` binary label, ~51% MHW
      frequency — meaningless ETS/POD/FAR); #11 (lines 46-52: broad
      `except Exception: return float("inf")` in `val_loss()` silently swallows
      parse errors). Data file used: merged_daily_deepSST.nc (pre-fix, #21).
- [x] scripts/run_xai_ensemble.py — **NEEDS_RERUN**
      Issues: #16 (line 73: `temporal_features=3` hardcoded — silent wrong
      params with strict=False); imports from ARCHIVE'd run_xai.py at lines
      36-42 (SEASONS, collect_predictions, plot_ig_comparison,
      plot_ig_temporal, top_indices_for_period); #11 (same broad except in
      its own best_checkpoint copy). Pre-audit is_land fix applied (line 253).
- [x] scripts/composite_ig.py — **NEEDS_RERUN**
      Issues: #9/#19 (docstring line 4 explicitly states "to_anom > 0 = MHW",
      and line 118 uses `trues > thr` with default thr=0.0); #16 (line 73:
      `temporal_features=3` hardcoded). Pre-audit is_land fix applied
      (line 150).
- [x] scripts/plot_split_scatter.py — **ARCHIVE**
      Old experiment naming (noSST/SST/deepSST), hardcoded paths to March
      2026 experiment directories, stale skill values. Superseded by new
      naming convention (#12). No functionality to preserve.
- [x] scripts/plot_variable_scatter.py — **ARCHIVE**
      Same: old naming, hardcoded March 2026 paths. No active callers.
- [x] scripts/plot_variable_xai_panel.py — **ARCHIVE**
      Old naming, "TBD" placeholder code for undefined future content,
      hardcoded stale skill values. No functionality to preserve.
- [x] src/xai/integrated_gradients.py — **CONFIRMED**
      Core function `_integrated_gradients()` correct: baseline zeros,
      50-step trapezoid, cuDNN disabled per step, returns CPU tensor.
      NF-P2-A (→ issue #20): `analyze_integrated_gradients()` at line 82 is
      dead code never called from any active script; contains data==0 land
      mask bug (#15) and unbatched loop (#8). Flag for deletion in Phase 5.

### Memory
- [x] project_decisions.md (folder 1 — hai-1127) — **KEEP_AS_HISTORICAL**
      April 20 2026 snapshot: old naming (noSST/deepSST), two-file data
      structure (merged_daily.nc + merged_daily_deepSST.nc — now unified),
      block-year jobs as "running". Records why block-year was adopted.
      Not current state; do not use skill numbers from this doc.
- [x] project_decisions.md (folder 2 — exprecursors) — **KEEP**
      Current standing decision: random split is defensible for XAI purpose
      (not a pure forecasting system); temporal K-fold CV is the plan when
      the time comes. This is the current position, not contradicted by
      folder 1 — the two docs reflect different moments: folder 1 records
      concern about inflated r=0.98, folder 2 records the accepted resolution
      (block-year for reported skill, random split acknowledged).
- [x] project_ensemble_plan.md — **KEEP_AS_HISTORICAL**
      Records April 24 2026 results: r=0.784, ETS=0.483, AUC=0.955 (OOS
      test). ALL numbers contaminated by NF-P2-B (#9/#19: binary label
      to_anom>0) AND by #21 (model trained on absolute ptho_bot). Must not
      be cited as current results. Seed table and job numbers are useful
      historical record only.
- [x] project_experiment_order.md — **ARCHIVE**
      Dependency chain uses old naming throughout (deepSST_layers4,
      noSST/SST/baseline). The logical ordering logic (data→split→vars→arch→
      skill→XAI→robustness) is sound but the content refers to a superseded
      experiment set. Superseded by new naming convention (#12).
- [x] project_compute_allocation.md — **KEEP_AS_HISTORICAL**
      April 24 2026 JUWELS budget estimate. Not current state (~118 runs
      logged then vs far more now), but the scaling logic (training×20h +
      skill×2h + XAI×5h) and the text block for a JUWELS petition renewal
      remain useful reference. Keep; do not treat numbers as current totals.
- [x] project_ig_peryear_decomposition.md — **KEEP_AS_HISTORICAL**
      Documents the April 30 XAI per-year figures and the "T_bottom
      +2.62%/decade" main finding. This finding is flagged by #21: the model
      that produced it was trained on absolute ptho_bot. Script paths cited
      (`eval/poster_ig_peryear*.py`) no longer exist at those locations after
      repo restructure. Re-verification against the corrected-data model is
      required before citing this result. Do not treat the trend number as
      confirmed. Addressed when project_ig_peryear_decomposition.md comes up
      in Phase 4 memory audit.

**Status: COMPLETE — 3 NEEDS_RERUN, 3 ARCHIVE, 1 CONFIRMED (code);
6 KEEP_AS_HISTORICAL, 1 KEEP (memory)**

---

## Phase 3 — June 2026

### Code
- [x] scripts/diag_attention.py — **NEEDS_RERUN**
      Issues: #16 (line 57: `temporal_features=3` hardcoded — with strict=False,
      mismatch silently leaves params randomly initialized); #11 (lines 33-37:
      broad `except Exception: return float("inf")` in `val_loss()`).
      NF-P2-C confirmed shared: line 23 imports `collect_predictions,
      top_indices_for_period` from ARCHIVE'd run_xai.py — a subset of
      run_xai_ensemble.py's dependency on the same script. Both scripts
      must be refactored before run_xai.py can be deleted.
- [x] scripts/mhw_hobday_stats.py — **CONFIRMED**
      Hobday definition verified complete: P90 threshold per DOY (line 98),
      31-day smooth applied at runtime (line 90, not from stored file — #7
      clean), gap closure ≤2d vectorised (apply_hobday_1d lines 45-48),
      persistence ≥5d via run-length encoding (lines 51-57). No to_anom>0
      pattern anywhere (#9/#19 clean). land_mask==1 for ocean (#2 clean).
      DATA_FILE from env var via paths.py (raises EnvironmentError if absent,
      #13 clean). is_land not accessed directly (#15 N/A).
      NF-P3-A: line 74 `doys` was after `ds.close()` — FIXED directly
      (moved doys before ds.close()). Back-check across all scripts confirmed
      this was NOT a systemic pattern; no other script has genuine post-close
      data access. NF-P3-A not added to known_issues.md.

### Memory
- [x] project_egu_feedback.md — **KEEP**
      Strategic paper roadmap from EGU Jun 2026. Priorities (signed IG,
      autocorrelation baseline, Gulf Stream vs. NS discrepancy) remain
      current. Step 1 (anomaly inconsistency) is resolved; others still
      pending. No stale scientific claims — framed as open questions.
- [x] project_poster_content.md — **KEEP_AS_HISTORICAL**
      EGU poster content is frozen — cannot retroactively change.
      Three sets of flags: (1) r=0.77/ETS=0.47/POD=0.83/FAR=0.15 in
      caption contaminated by #9/#19 (to_anom>0 binary label) AND #21
      (model trained on absolute ptho_bot); (2) "attribution concentrated
      along the Gulf Stream" (#14, superseded by perturbation result: masking
      GS T_bottom does NOT degrade skill); (3) T_bottom +2.6%/decade trend
      (#21, produced by model trained on absolute ptho_bot). All three claims
      need re-verification before the paper. Do not cite from this doc.
- [x] project_spatial_forecast.md — **KEEP**
      Current scientific direction (encoder-ConvLSTM-decoder spatial
      prediction). Data path uses v2 file (corrected anomaly). Status
      checklist may be stale (48d old) but architectural decisions are valid.
- [x] project_update_jun2026.md — **KEEP_AS_HISTORICAL**
      Supervisor update Jun 2026. Contains same contaminated skill numbers
      as project_poster_content.md (#9/#19, #21). "preprocessing
      inconsistency" listed as open → now resolved (project_anomaly_
      inconsistency.md). Gulf Stream IG claim superseded by perturbation
      result (#14). Not current state.
- [x] project_anomaly_inconsistency.md — **KEEP**
      Confirmed: this is the document that records the ptho_bot fix (#21).
      "RESUELTO Jun 2026" + description of DOY climatology subtraction
      matches exactly the June 29 preprocessing log (job 14070999). Date
      consistent. Minor window difference (11d SST vs 5d ERA5/ptho_bot)
      acknowledged and still accurate.
- [x] project_kfold_pending.md — **UPDATE applied**
      Scientific need (k-fold block-year CV for robust OOS XAI) still
      valid. Implementation plan updated: run_xai.py reference marked
      "ARCHIVE'd, needs replanning"; naming updated to TbotAtm. Core
      scientific rationale and k=5 plan unchanged.
- [x] project_qnet_future.md — **KEEP**
      9 days old, active future task, no stale content.
- [x] feedback_naming_reorganization.md — **ARCHIVE**
      Documents intermediate naming (deepSST_layers4, SST_layers2, noSST)
      which was completed and then superseded by Atm/SSTAtm/TbotAtm
      convention (#12). Standing instructions obsolete; task done.

**Status: COMPLETE — 1 NEEDS_RERUN, 1 CONFIRMED (code);
3 KEEP, 3 KEEP_AS_HISTORICAL, 1 UPDATE applied, 1 ARCHIVE (memory)**

---

## Phase 4 — August 2026 (masking, partition, onset, causal, tau)

### Pre-audit fix applied (Aug 2026)
`scripts/eval_ig.py` line 221 and `scripts/composite_ig_signed.py` line 132:
`tierra_mask` → `is_land`. Same crash bug (issue #15) as Phase 2 above.
Applied before Phase 4 audit. Audit of the rest of these scripts not yet done.

### Code
- [ ] scripts/train_partition.py
- [ ] scripts/eval_onset_skill.py
- [ ] scripts/ig_masked_batched.py + ig_masked_merge.py
- [ ] scripts/persistence_baseline.py
- [ ] scripts/persistence_remote_sst.py
- [ ] scripts/analysis/causal_triangulation.py
- [ ] scripts/analysis/check_tau_methodology.py
- [ ] scripts/analysis/thermal_inertia_test.py
- [ ] scripts/eval_ig.py
- [ ] scripts/composite_ig_signed.py

### Memory
- [ ] project_paper_narrative_aug2026.md
- [ ] project_open_items_aug2026.md
- [ ] project_granger_methodology.md
- [ ] project_results_full_jul2026.md
- [ ] project_input_variables.md
- [ ] project_target_definition.md
- [ ] project_results_status.md
- [ ] project_results_summary.md
- [ ] project_overview.md
- [ ] project_reviewer_responses.md

### Known open item (carried from Aug 13 session)
Onset skill with corrected MHW definition shows r≈0.02-0.03 (n=84),
contradicting the earlier (pre-fix) result that anchored the paper
narrative. Needs: persistence r conditioned on the same 84 onset samples,
to know if the model beats/ties/loses persistence there. This must be
resolved before project_paper_narrative_aug2026.md can be marked KEEP.

**Status: NOT STARTED**

---

## Phase 5 — human-readable rewrite (pair coding, VS Code)
Only for scripts confirmed as feeding results/all_results.csv after
Phase 4 is complete. Not started. Do not begin early — polishing
readability of a script that might still get archived wastes effort.

---

## Permanent / no phase needed
- feedback_ask_before_acting.md — KEEP (standing rule)
- feedback_xai_and_loss.md — KEEP (standing rule)
- user_contact.md — KEEP (permanent reference)
- project_fdl2026_slides.md — KEEP_AS_HISTORICAL (past event record, FDL
  application, no scientific claims to verify)