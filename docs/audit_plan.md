# Audit plan — pre-supervision (2 weeks)

## Goal
Reconstruct a reliable narrative + results table for the supervision
meeting in 2 weeks. Chronological audit, oldest first. Phase 1 scripts
(March 2026, supervisor-reviewed) are the style/structure template.
Home memory (~/.claude/.../memory/) is resolved IN PARALLEL with each
phase, not before or after — each memory doc is tied to the code era
it documents.

## Reference documents (read first, always up to date)
- docs/known_issues.md — the 22-point bug checklist. Add new entries here,
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
- [x] scripts/train_partition.py — **CONFIRMED** (indirect evidence only)
      Training log for job 14193976 NOT FOUND in repo. Indirect evidence
      strongly supports post-fix data: checkpoints dated Aug 11, 2026 (6
      weeks after June 29 fix); merged_daily_deepSST.nc does not exist —
      LazyDataset would raise FileNotFoundError if configs pointed there;
      eval log (job 14194729, Aug 12) shows ptho_bot mean=0.0186,
      std=0.2757 (anomaly scale, not absolute temperature ~5-15°C). Cannot
      confirm from surviving artifacts; reasoning by exclusion.
      #13: CLEAN — data_dir:"" in current configs → MHW_DATA_FILE env var.
      #16: CLEAN — all arch params from config.get() (L138-146).
      #17: CLEAN — CSVLogger, no WandbLogger (L171). #8: N/A (Lightning).

- [x] scripts/eval_onset_skill.py — **CONFIRMED** (known limitation, NF-P4-F RESOLVED)
      #9: CLEAN — full Hobday in L64-119 (load_ns_p90 + apply_hobday +
      P90 threshold). #11: CLEAN — val_loss() L245-247 returns float("inf")
      on regex failure. #16: CLEAN — all config.get() (L130-138).
      #8: per-sample inference loop (L152-173) — flag; no IG in this script.
      Known limitation: persistence baseline in skill_by_phase() (L182-202)
      computed over ALL test samples, not conditioned on the 84 onset
      indices — open item from Aug 13, must be addressed before the onset
      skill result can be cited.
      NF-P4-F RESOLVED (Aug 16 2026): L233-235 path corrected from
      partition/configs/{folder}/fold{fold}.yaml to
      configs/partition/{folder}/fold{fold}.yaml. Dry-run confirms 10/10
      configs found (remote fold0-4, local fold0-4).

- [x] scripts/eval_onset_persistence.py — **NEW (post-audit, Aug 16 2026)**
      Written to close the onset/persistence open item (NF-P4-F blocker).
      Loads existing NPZ files from eval_onset_skill.py (no inference);
      computes lag-7 persistence (persist[i] = trues[i-7] within each fold's
      consecutive test series); pools 84 onset samples across 5 folds;
      computes Pearson r with 95% Fisher z-transform CI for model and
      persistence. No boundary-NaN issues (0 onset days in first 7 positions
      of any fold). Full results in docs/narrative.md Onset skill section.

- [x] scripts/ig_masked_batched.py — **CONFIRMED** (NF-P4-B latent,
      non-triggered)
      NF-P4-B: LOCAL best_ckpt() at L40-49, not importing canonical from
      src/utils/checkpoints.py. Empirical check of all SSTAtm_lstmonly_
      gnll_masked checkpoint names (Aug 2026 full set):
        cnn-lstm-epoch=XX-val_loss=Y[.ckpt | -v1.ckpt]
      Local regex (-?[0-9]+\.[0-9]+) correctly matches all real filenames
      including GNLL negatives (e.g. val_loss=-0.0076, -0.1087). Fallback
      to 0.0 NEVER triggered for these files. Additional nuance: local
      version does not exclude -v1 duplicates (canonical does), so min()
      is non-deterministic when two files tie on val_loss; since both
      represent the model at that loss value, result validity is unaffected.
      Existing masked-IG results are citable.
      #8: FIXED — batch loop at L166. #16: CLEAN — config.get() L54-62.
      #21: N/A (SSTAtm experiment, not TbotAtm).
      Phase 5: import canonical best_ckpt() (see Phase 5 note below).

- [x] scripts/ig_masked_merge.py — **CONFIRMED**
      No model loading. DATA_FILE from env var (L28) — CLEAN #13.
      land_mask.astype(bool) (L116) — CLEAN #2. Clean on all 22 points.

- [x] scripts/persistence_baseline.py — **CONFIRMED**
      --data required CLI arg — CLEAN #13. Clean on all 22 points.
      NF-P4-C (cosmetic): L155 label "CNN-LSTM + GNLL" hardcoded — stale
      if model is retrained; not a computation bug.

- [x] scripts/persistence_remote_sst.py — **CONFIRMED**
      DATA_FILE from env var (L28) — CLEAN #13. L59 comment confirms
      land_mask=1 means ocean — CLEAN #2.
      NF-P4-C (cosmetic): L123/L140 r = 0.807 ± 0.038 hardcoded — stale
      if model is retrained; not a computation bug.

- [x] scripts/analysis/causal_triangulation.py — **CONFIRMED**
      #13: CLEAN — DATA_FILE from env var (L18).
      #10: CONFIRMED in results — triangulation_results.csv shows rho≈1.0
      for all drivers (CCM saturation per known issue #10, as expected).
      #11: except Exception path (L95-107) present but NEVER triggered in
      current run — granger_fstats.csv shows all granger_lag > 0 (values:
      1, 1, 1, 2, 2, 3) and no granger_p=1.0 rows. Result not contaminated.
      L287: warnings.filterwarnings("ignore") suppresses all warnings
      (anti-pattern, not a bug affecting results).
      F-stats non-comparability disclaimer ABSENT from code: the agreement
      that "F-stats are not comparable across variables for ranking, only
      for direction" exists in conversation and project_granger_
      methodology.md but has zero occurrences in the script itself (grep
      confirmed: no "comparable/ranking/caveat/direction/F-stat").
      Documentation gap only; does not change the CSV numbers.

- [x] scripts/analysis/check_tau_methodology.py — **CONFIRMED** (NF-P4-E)
      #13: CLEAN — DATA_FILE from env var (L32/L41).
      #2: CLEAN — ocean = ds["land_mask"].values.astype(bool) (L61).
      #21: N/A — pure ACF analysis on to_anom, no model loading.
      L428: print("Loading NS to_anom from merged_daily.nc ...") confirms
      correct data file at runtime.
      NF-P4-E: L389-393 boxplot bug — data_box uses loop variable `late`
      (= tau_late, full NS series after loop ends) instead of tau_n_late /
      tau_s_late. North-late and South-late boxes in tau_check2_boxplot.png
      are IDENTICAL (both show full-NS tau). Printed Check 2 numbers used
      in the current narrative (North -46d, South -31d) come from text
      output (L278-296) and are CORRECT / UNAFFECTED. Spatial maps correct.
      tau_check2_boxplot.png MUST be regenerated before any paper use.

- [x] scripts/analysis/thermal_inertia_test.py — **NEEDS_RERUN**
      NF-P4-B: LOCAL best_ckpt() (L156-165), L163 returns 0.0 on failure —
      same violation as ig_masked_batched.py; same canonical fix applies.
      NF-P4-D (#22): patched_config_path() (L168-193) silently routes to
      merged_daily_deepSST_OLD.nc when multiseed configs' data_dir
      (merged_daily_deepSST.nc) is missing. CONFIRMED ACTIVE: all inspected
      multiseed configs have data_dir pointing to the non-existent
      merged_daily_deepSST.nc → fallback to _OLD (pre-fix absolute ptho_bot,
      issue #21). Step 2 (per-year IG computation) has NEVER completed —
      no ig_peryear or ig_tbot_peryear files found anywhere in experiments/
      (confirmed by find). Risk detected before contamination; no existing
      result is affected.
      #16: CLEAN — L199-207 all config.get(). #8: per-sample IG loop
      (L287-298) — flag.

- [x] scripts/eval_ig.py — **CONFIRMED**
      Pre-audit is_land fix applied (L221: full_ds.is_land.numpy()).
      #9: CLEAN — full Hobday in L45-97 (load_ns_p90 + apply_hobday +
      P90 threshold). #15: CLEAN post-fix.
      #16: CLEAN — config.get() (L121-130).
      #8: per-sample loops for both inference (L162-176) AND IG (L237-248)
      — performance flag, does not affect correctness.
      L143: xr.open_dataset(config["data_dir"]) for lat/lon — fails loudly
      if data_dir:"". Reference implementation for MHW/non-MHW signed IG
      attribution with correct Hobday labels.

- [x] scripts/composite_ig_signed.py — **NEEDS_RERUN**
      Pre-audit is_land fix applied (L132).
      #9/#19 CONFIRMED: L105 mhw_idx = np.where(trues > thr)[0] with
      default thr=0.0 — uses raw normalised target, not Hobday MHW. ~50%
      of all days labelled MHW (Hobday expected ~10%). All MHW/non-MHW
      signed attribution results from this script are invalid.
      #15: CLEAN post-fix. #16: CLEAN. #8: per-sample inference (L91-100).
      Note: any claim of MHW/non-MHW asymmetry in signed IG attributions
      in memory docs (project_ig_peryear_decomposition.md, poster content)
      must be re-verified using eval_ig.py (correct Hobday version) once
      the corrected-data model is available.

### Memory

> **CRITICAL — Gulf Stream vs NS T_bottom perturbation result (Jun 1 2026):**
> The finding that "GS T_bottom masking slightly IMPROVES skill while NS T_bottom
> masking DROPS skill" — used as the empirical basis for the IG-vs-causality
> distinction — was produced by a pre-#21 model (absolute ptho_bot, before the
> June 29 fix). The qualitative direction (GS not causally necessary, NS T_bottom
> is) may survive re-verification but has NOT been confirmed with the corrected-data
> model. Do not present this as verified fact in the supervision meeting without
> this caveat.

> **GNLL fold-identical bug (for the record):** A "Full TbotAtm GNLL" experiment
> (5 seeds × 5 folds) was discarded Aug 16 because 3 folds showed r=0.886
> identical to 3 decimal places (val_loss overfit at epoch 7). This bug occurred
> ONLY in that discarded experiment — it does NOT affect SSTAtm_lstmonly_gnll
> (n=25, all distinct, confirmed from metrics.csv Aug 16) or TbotAtm MSE partition
> (5 folds, all distinct).

- [x] project_paper_narrative_aug2026.md — **UPDATE applied**
      Numbers updated: SSTAtm GNLL pair r=0.865/r=0.807 replaced with canonical
      TbotAtm MSE partition numbers (Full=0.860±0.031, Remote=0.802±0.037,
      Local=0.906±0.029, seed42 5 folds, post-fix Aug 11 2026). The 93% ratio
      (0.802/0.860 = 93.3%) is unchanged. Four reasons for TbotAtm over SSTAtm:
      (1) best model; (2) post-fix (#21 clean); (3) directly measures partition;
      (4) TbotAtm includes ptho_bot, the variable central to the reframed narrative
      — citing SSTAtm would answer the wrong question even though ratio matches.
      Note on discarded TbotAtm GNLL added (fold-identical bug, not in audited exps).
      Remaining open: Step 3 of narrative arc (thermal_inertia IG per decade) is
      BLOCKED on NF-P4-D fix. Onset skill for partition models needs 5-fold aggregate.

- [x] project_open_items_aug2026.md — **UPDATE**
      Scientific decisions valid (Granger methodology, onset table, Granger+IG
      complementarity). "Jobs activos" section stale: partition jobs 14194108/09 are
      COMPLETE — TbotAtm_remote_seed42_fold0-4 and TbotAtm_local_seed42_fold0-4 all
      have metrics.csv (confirmed from disk Aug 16). Job 14194129 (ig_hobday signed
      IG): status not confirmed from surviving artifacts. Onset skill table (SSTAtm
      r=0.245, TbotAtm r=0.201 at onset vs persist r=0.111): from spatial_forecast/
      eval/figures/ — origin model independent of scalar kfold. TbotAtm row potential
      #21 caveat (spatial model training history not verified). Must refresh "Jobs
      activos" section and add partition fold-level results.

- [x] project_granger_methodology.md — **KEEP**
      Disclaimer text ("F-stats not directly comparable in magnitude across variables")
      correct and complete. F-stat table consistent with granger_fstats.csv verified in
      Phase 4 code audit (v10 F=2166 lag1, msl F=1054 lag2, u10 F=618 lag1, ssr F=592
      lag1, Tbot_NS F=137 lag2, Tbot_GS F=27 lag3). Tbot autocorrelation finding
      (GS lag60 ACF=0.837 > NS=0.719) consistent with check_tau_methodology.py text
      output (L278-296, correct per NF-P4-E analysis). Disclaimer absent from code
      itself (noted in Phase 4 code audit — documentation gap only). Granger+IG
      complementarity narrative valid and uncontradicted by audit.

- [x] project_results_full_jul2026.md — **KEEP_AS_HISTORICAL**
      Read from actual metrics.csv files Jul 7, 2026. TbotAtm numbers (kfold
      r=0.860, multi-seed r=0.863±0.041, GNLL r=0.871±0.036, lead sweep all leads)
      are #21-flagged: kfold TbotAtm configs have data_dir: merged_daily_deepSST.nc;
      if runs completed before June 19 2026 (rename to _OLD), ptho_bot was absolute
      temperature. Individual kfold checkpoint dates not verified. Do not cite any
      TbotAtm number from this doc without re-verification on corrected-data model.
      SSTAtm, Atm, and masked r=0.807±0.038 (93% skill retained) are CLEAN for #21.
      GNLL fold-identical bug did NOT affect SSTAtm GNLL (r=0.865 ± 0.022 in this
      doc accurate — confirmed from metrics.csv Aug 16). Spatial forecast spatial r≈0.004
      (land_mask bug) pre-dates Jul 2026 bug fix; post-fix results not in this doc.

- [x] project_input_variables.md — **UPDATE applied**
      to_anom definition corrected: was "SST − P90_climatology(DOY±5days) CDO
      ydrunpctl,90,5" — wrong (#1). Fixed to "SST − mean_clim(DOY) CDO ydaysub",
      matching project_target_definition.md (authoritative corrected doc).
      Confirmed from merged_daily.nc: to_anom mean=0.091°C (SST-mean_clim + warming
      trend); SST-P90 would give mean≈−0.7°C. All other content clean.

- [x] project_target_definition.md — **KEEP**
      Already corrected Aug 2026. to_anom = SST − mean_clim(DOY), CDO pipeline
      verified (ydrunmean,11 → ydaysub → to_anom; ydrunpctl,90,11 → p90_thresh).
      Authoritative definition, consistent with known_issues.md #1 and code audit.
      p90_thresh stored without 31-day runtime smooth — consistent with known
      issue #7 (smooth applied only in load_ns_p90() at evaluation time).

- [x] project_results_status.md — **KEEP_AS_HISTORICAL**
      April 2026 snapshot, 118 days old. Triple contamination:
      #9/#19 — to_anom > 0 threshold (MHW days 51.2%, ETS/CSI/POD/FAR meaningless);
      #21 — absolute ptho_bot;
      #12 — old naming (deepSST_layers4 vs TbotAtm).
      Do not cite any number from this doc. Historical record of April 2026 state only.

- [x] project_results_summary.md — **KEEP_AS_HISTORICAL**
      Contains two-era data. April 2026 XAI conclusions (#1-7): #21-flagged. GS
      hotspot claim (#14, superseded by perturbation) and ptho_bot +2.6%/decade
      regime shift are NOT citable. Multi-seed ensemble skill (r=0.904, ETS=0.612,
      Apr 2026): #9/#19 + #21. Jun 1 perturbation (GS vs NS T_bottom): pre-fix
      (#21) — see CRITICAL NOTE at top of Memory section. Qualitative direction may
      survive re-verification. Superseded as scientific reference by
      project_paper_narrative_aug2026.md.

- [x] project_overview.md — **UPDATE**
      April 2026 snapshot, 118 days old. Objective ("MHW predictability via XAI,
      CNN-LSTM + Attention, 60d window, lead=7, NS target") and scientific question
      ("which signals in North Atlantic precede NS thermal anomaly?") still valid.
      Stale: naming (#12: noSST/SST/deepSST → Atm/SSTAtm/TbotAtm); "Loss: MSELoss"
      (GNLL now preferred); code section references ARCHIVE'd run_xai.py (#20);
      "XAI aplicado (todos completados)" refers to pre-fix April 2026 XAI; WandbLogger
      in code description (#17); all skill numbers from pre-fix April 2026 model.
      Must update before writing methods/intro section of paper.

- [x] project_reviewer_responses.md — **KEEP_AS_HISTORICAL**
      Q1 (block-year split, 31 test years across 14 seeds): methodologically valid,
      still the correct answer for paper. Q2 (MSE loss → smooth conditional mean):
      technically valid; GNLL now preferred (adds uncertainty) but smoothness point
      stands. Q3 (Gulf Stream vs NS T_bottom perturbation, Jun 1): pre-fix (#21) —
      see CRITICAL NOTE above. Qualitative direction (NS T_bottom matters, GS doesn't)
      must not be stated as confirmed without re-running on corrected-data model.

### Known open item — CLOSED (Aug 16 2026)
Onset skill pooled across 5 folds and persistence baseline computed by
eval_onset_persistence.py. NF-P4-F (stale path in eval_onset_skill.py) RESOLVED.
Results: neither model (r=0.006–0.093) nor lag-7 persistence (r=−0.067) predict
MHW onset at n=84 with statistical power (need |r|>0.21 for p<0.05).
See docs/narrative.md Onset skill section for full table and interpretation.

**Status: COMPLETE — code: 9 CONFIRMED (incl. 1 new post-audit), 2 NEEDS_RERUN;
memory: 2 UPDATE applied, 2 UPDATE, 2 KEEP, 4 KEEP_AS_HISTORICAL**

---

## Phase 5 — human-readable rewrite (pair coding, VS Code)
Only for scripts confirmed as feeding results/all_results.csv after
Phase 4 is complete. Not started. Do not begin early — polishing
readability of a script that might still get archived wastes effort.

Consolidation note (to apply in Phase 5 rewrite): 4 scripts
(ensemble_skill.py, run_xai_ensemble.py, ig_masked_batched.py,
thermal_inertia_test.py) re-implement best_ckpt() locally instead of
importing src/utils/checkpoints.py. The canonical version uses
float("inf") on failure (correct) and excludes -v1/-v2 duplicates.
Replace all local copies with the import in the Phase 5 pass.

---

## Fase 4-bis — spatial_forecast (Jun–Aug 2026, parallel experiment)

Self-contained experiment directory at `/p/project1/hai_1127/radin1/spatial_forecast/`.
NOT integrated into exprecursors repo. Architecture: CNN encoder (per frame) →
ConvLSTM → decoder → 2D to_anom field (141×201). Audited Aug 16–17 2026.

All 9 scripts read in full. Dry-run performed (CPU login node, forward pass shape
verified [2,1,16,20], physics loss verified). All imports clean.

Data file confirmed: `merged_daily.nc` — both land masks present (verified via
`data_vars`: `land_mask` + `land_mask_tbottom`, 572 pixels differ, same 1=ocean
convention — see known_issues.md #2 note added Fase 4-bis).

### Cross-cutting: hardcoded data paths (LOW — standalone experiment)
All 9 scripts hardcode `DATA_FILE` / `DATA_NC` / `MASK_FILE` as the absolute
path `/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily.nc`.
Path is correct (post-fix file). The exprecursors `MHW_DATA_FILE` env-var
convention does not apply to this standalone experiment; the hardcoded path
is acceptable within JUWELS. Not a correctness bug.

### Cross-cutting: NaN bug resolved (NF-S-2, RESOLVED — #24)
Issue #24 absent in current code. `dataset_spatial.py` lines 128–131:
`pix_std[self.land_mask] = 1.0` → `nan_to_num(pix_std, nan=1.0)` → `clamp(min=1e-4)`.
The Jun 25–30 runs had "Target std (ocean mean): nan" from an earlier version
without these guards (deleted Aug 16 per #24). Current valid runs (Jul–Aug 2026)
print a finite std.

### New findings

**NF-S-1** (MEDIUM — instance of #11): `dataset_spatial.py` lines 55–56.
Silent fallback `self.land_mask_tbottom = self.land_mask` when `land_mask_tbottom`
not in dataset. No warning printed. In production, `merged_daily.nc` DOES contain
`land_mask_tbottom` (confirmed via `data_vars`) — fallback is NEVER triggered with
the current file. Latent bug: if the data file were replaced without this variable,
`ptho_bot` would silently receive the SST mask (wrong for 572 coastal pixels) with no
log or error. Fix: add `warnings.warn()` or `print()` in the else branch.
No existing result is affected.

**NF-S-3** (MEDIUM — instance of #23): `train_spatial.py` line 38 and
`train_spatial_phys.py` line 40. `rng_fold = np.random.default_rng(0)` — fold
assignment seed hardcoded as 0, independent of the config `seed` parameter. All
existing runs share the same fold-year assignment (consistent internally) but the
`seed` field in config controls only val/train shuffling, not test year assignment.
This is undocumented. Fix: expose as `config.get("fold_seed", 0)` and add a comment.

**NF-S-4** (LOW — DRY): `build_splits` duplicated verbatim between `train_spatial.py`
and `train_spatial_phys.py`. If one is patched and the other is not, fold assignments
diverge silently. Fix: extract to shared `utils_spatial.py` in Phase 5 pass.

**NF-S-5** (HIGH — instance of #9/#19): `eval/mhw_onset_skill.py` lines 117–118.
```python
is_mhw_tgt = tgt  > 0   # (N, H, W)
is_mhw_inp = to_t > 0   # (N, H, W)
```
`tgt` is the per-pixel normalised regression target (to_anom / pix_std). `> 0` on
the normalised scale labels ~31.5% of target pixels as MHW (confirmed empirically:
SSTAtm_seed42_fold0 test_targets.npy). Hobday expected ~10%. No p90(DOY) threshold,
no persistence ≥5d, no gap-merging. Onset / mid-event / no-MHW pixel-level skill
maps computed against this criterion are physically meaningless and cannot be cited.
Fix: replace with Hobday filter applied per pixel using `p90_thresh(DOY)` from
climatology — this requires a non-trivial 2D extension of the scalar `apply_hobday`.

**NF-S-6** (LOW — instance of #11): `model_spatial_phys.py` lines 71–73. Silent
fallback from MLD-weighted MSE to plain `_masked_mse` when `self.mld_weights is None`.
Prevented in practice by `train_spatial_phys.py` lines 93–97 (RuntimeError if
`mld_file` not provided). Latent but not triggered; no existing result affected.

**NF-S-7** (LOW): `model_spatial_phys.py` val_loss = MLD-MSE + λ_lap × Laplacian.
Composite metric: checkpoint val_loss values are not comparable across runs with
different `lambda_lap`. Not a bug; must be noted when comparing physics vs standard
checkpoints or when interpreting best_ckpt() selections across phys configs.

### Per-script verdicts

- [x] `dataset_spatial.py` — **NEEDS_FIX** (NF-S-1)
      #1: CLEAN — target = `ds["to_anom"]` line 64 (correct variable, post-fix).
      #2: CLEAN — lines 46–49: explicit `lm=1→ocean`, `ocean_mask=tensor(lm)` (True=ocean),
        `land_mask=tensor(~lm)` (True=land); `land_mask_tbottom` handled lines 52–54.
      #11: NF-S-1 — line 55–56 silent fallback (latent; not triggered in production).
      #21: CLEAN — DATA_FILE hardcoded to merged_daily.nc (post-Jun-29 fix).
      #24: NF-S-2 RESOLVED — lines 128–131 NaN guard confirmed present.
      All other issues: N/A.

- [x] `dataset_spatial_phys.py` — **CONFIRMED**
      24-line thin subclass: appends `month` to return tuple, no new logic.
      All relevant checks inherit from `dataset_spatial.py` verdict above.

- [x] `model_spatial.py` — **CONFIRMED**
      CNN encoder → ConvLSTM → decoder (225 lines). SpatialLightningModule with
      `_masked_mse` using `ocean_mask` (True=ocean, from dataset).
      #2: CLEAN — ocean_mask used correctly throughout.
      #4: CLEAN — MSELoss only, no GNLL/MSE confusion.
      #11: No silent fallbacks in main computation path.
      All other issues: N/A.

- [x] `model_spatial_phys.py` — **CONFIRMED** (NF-S-6, NF-S-7 LOW)
      PhysicsLightningModule: MLD-weighted MSE + λ_lap × Laplacian (126 lines).
      NF-S-6 (LOW, #11): silent fallback at line 71–73 blocked by train-script guard.
      NF-S-7 (LOW): composite val_loss not directly comparable across λ values.
      All 24 issues otherwise N/A or CLEAN.

- [x] `train_spatial.py` — **NEEDS_FIX** (NF-S-3)
      #21: CLEAN — DATA_FILE = merged_daily.nc.
      #23: NF-S-3 — line 38: `rng_fold = np.random.default_rng(0)` (fold seed hardcoded).
      NF-S-4 (LOW): `build_splits` duplicated from train_spatial_phys.py.
      All other issues: N/A.

- [x] `train_spatial_phys.py` — **NEEDS_FIX** (NF-S-3)
      #21: CLEAN — DATA_FILE = merged_daily.nc.
      #23: NF-S-3 — line 40: same `rng_fold = np.random.default_rng(0)`.
      NF-S-4 (LOW): `build_splits` duplicated from train_spatial.py.
      All other issues: N/A.

- [x] `preprocessing/compute_mld_weights.py` — **CONFIRMED**
      #2: CLEAN — `land_mask=1→ocean`; mask inverted before use; explicit comment.
      MASK_FILE hardcoded (merged_daily.nc, correct).
      #11: No silent fallbacks.
      All other issues: N/A.

- [x] `eval/persistence_baseline_spatial.py` — **CONFIRMED**
      #2: CLEAN — `land_mask=1 means OCEAN` at line 62; correctly inverted.
      #9/#19: CLEAN — computes lag-1 persistence (MSE-based), no MHW classification.
      DATA_FILE hardcoded (merged_daily.nc, correct).
      All other issues: N/A.

- [x] `eval/mhw_onset_skill.py` — **NEEDS_FIX** (NF-S-5)
      #9/#19: NF-S-5 (HIGH) — lines 117–118: `tgt > 0` criterion, 31.5% pixels
      flagged vs Hobday expected ~10%. No p90(DOY), no persistence filter.
      All onset/mid-event/no-MHW skill maps from this script are invalid.
      #2: CLEAN — ocean_mask loaded from dataset (True=ocean), applied correctly.
      DATA_NC hardcoded at line 29 (merged_daily.nc, correct).
      All other issues: N/A.

### Overall
5 CONFIRMED | 4 NEEDS_FIX | 0 ARCHIVE | 0 NEEDS_RERUN

Priority fix order before any spatial onset maps are cited:
1. **NF-S-5** (HIGH): `mhw_onset_skill.py` — fix MHW criterion to Hobday p90 per
   pixel + persistence. All current onset/mid-event maps are invalid.
2. **NF-S-3** (MEDIUM): Expose fold_seed in config; document what `seed` controls.
3. **NF-S-1** (MEDIUM): Add warning in `dataset_spatial.py` else branch.
4. **NF-S-4** (LOW): Extract shared `build_splits` in Phase 5 pass.

**Status: COMPLETE (confirmed Aug 17 2026) — 5 CONFIRMED | 4 NEEDS_FIX | 0 ARCHIVE | 0 NEEDS_RERUN**

---

## Fase 4-bis follow-up — Raven migration + re-audit (Aug 24 2026)

Before this session's first-ever Raven launch of the spatial pipeline,
user asked for a fresh, thorough re-audit ("coger otro agente nuevo y
volver a revisar todo") — 3 parallel background agents (data/split,
model/eval, Raven-migration readiness), independent of the Aug 16-17
pass above. Findings, resolved against the NF-S-* items above:

- **NF-S-1** (silent fallback, `dataset_spatial.py` land_mask_tbottom):
  confirmed still present, confirmed NOT triggered on real Raven data
  (the fallback branch never executes — `land_mask_tbottom` is present
  in `merged_daily.nc`). Deferred, LOW, unchanged.
- **NF-S-2** (NaN guard): re-confirmed RESOLVED, no regression.
- **NF-S-3** (fold-seed hardcode) — **upgraded from MEDIUM
  documentation-gap to a real bug, and FIXED.** The Aug 16-17 pass
  characterized this as "seed independent of config's own seed, document
  it." The Aug 24 re-audit found something worse by simulating against
  real 1985-2024 data: the *val*-year shuffle (a separate `rng(seed)`
  call, not the `rng_fold(0)` one originally flagged) produces
  83-100% pairwise val_years overlap between adjacent folds when `seed`
  is held constant — exactly what naturally happens when `fold1.yaml` is
  created by copying `fold0.yaml` (this project's own naming
  convention). Fixed: `rng(seed + fold)`. See known_issues.md #62 for
  full detail and post-fix verification.
- **NF-S-4** (`build_splits` duplication): confirmed still present in
  both `train_spatial.py`/`train_spatial_phys.py`. Not addressed this
  session (phys variant not being launched) — still recommended for a
  future Phase 5 `utils_spatial.py` extraction, now with one more
  duplicate site added (`mhw_onset_skill.py::get_test_idx` and
  `persistence_baseline_spatial.py::get_test_years` were found to
  independently reimplement the same fold-year logic — currently in
  agreement with `train_spatial.py`, verified by direct comparison of
  hardcoded constants, but a latent divergence risk).
- **NF-S-5** (HIGH, `mhw_onset_skill.py` MHW criterion): re-confirmed
  present and unchanged (line numbers shifted slightly, 129-130). A full
  fix is now designed (regrid `p90_thresh` onto the spatial grid, run
  `apply_hobday()` per-pixel on the full contiguous record) and
  benchmarked (~74s CPU, one-time) — see known_issues.md #64 — but
  deliberately not implemented this session; still **NEEDS_FIX**, do not
  cite any spatial onset map.
- **NF-S-6/NF-S-7** (`model_spatial_phys.py`): re-confirmed isolated,
  LOW, out of scope (phys variant not launched tonight).
- **Two NEW findings**, not in the Aug 16-17 pass:
  - **Raven-migration path fixes** (JUWELS-hardcoded `DATA_FILE`,
    `output_dir`, missing `fold1.yaml`) — this pass's actual trigger;
    see known_issues.md #61. All fixed and verified before launch.
  - **`compute_stats()` per-variable masking mismatch** — new bug, same
    family as #55 (a `__getitem__`/`compute_stats()` scope divergence).
    Fixed; see known_issues.md #63.

**Outcome**: 2 folds (fold0, fold1) of the plain TbotAtm model launched
on Raven for the first time (job 29527143), after applying every fix
above that was in scope for tonight's launch and smoke-testing both (F3,
F4) directly against real data pre-launch. NF-S-4 and NF-S-5 remain
open, tracked, and deliberately deferred — not silently dropped.

**Status: Raven-migration sub-pass COMPLETE (Aug 24 2026) — 2 new bugs found and fixed, 1 bug reclassified (documentation-gap → real bug) and fixed, NF-S-4/NF-S-5 still open and tracked.**

---

## Pending decision — land_mask grid-offset fix (Aug 18 2026, NOT applied yet)

Full detail in `known_issues.md` #2. One-line status for quick reference:
`land_mask_05.nc` (source of the `land_mask` variable in `merged_daily.nc`)
has a confirmed 0.25° latitude grid-offset bug vs the ERA5 target grid
(140 vs 141 points) — causes a systematic ~1px coastal misalignment,
uniform across the whole domain (572 pixels, every coastline, not
concentrated anywhere). `land_mask_tbottom_05.nc` is already correctly
aligned. Root cause traced and fixed at the source
(`/p/project1/hai_1127/inputs/daily/preprocess_data/preprocess_all.py`,
outside this repo) so future full regenerations won't reintroduce it.

**A corrected data file already exists and is verified**:
`/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily_v2.nc` —
`land_mask` replaced with `land_mask_tbottom`'s (correct) values; every
other variable (`to_anom`, `ptho_bot`, `u10`, `v10`, `msl`, `ssr`, `target`,
`land_mask_tbottom`) confirmed byte-identical to the original. Comparison
figures: `figures/sst_vs_tbot_mask_comparison{,_AFTER_FIX}.png`,
`figures/mask_mismatch_full_domain{,_AFTER_FIX}.png`.

**Explicit decision (user, Aug 18 2026): do NOT migrate or rerun anything
now.** `merged_daily.nc` (original) stays canonical; no config points to
`_v2`. Practical urgency is low for current TbotAtm work — `dataset.py`'s
earlier per-variable mask fix already routes ptho_bot through the
(already-correct) `land_mask_tbottom` directly, bypassing the buggy
`land_mask` variable for the one ocean variable current TbotAtm configs
actually mask. `merged_daily_v2.nc` mainly matters for SSTAtm/Atm-family
pipelines that mask `to_anom` (or anything else) via the generic
`land_mask` field — not yet audited which configs do that.

**If/when a rerun is ever decided**: use `merged_daily_v2.nc`, not
`merged_daily.nc` — that's the whole point of having it ready. Until then,
this is parked, not forgotten.

---

## Phase 6 — Aug 21-22 2026 (land_fill artifact investigation, full
## re-training batch, XAI triangulation) — NEW, not previously tracked here

This entire phase happened in a separate working session AFTER Phase 4's
Aug 18 cutoff and was never logged in this plan until now (Aug 22 2026).
Chronological audit discipline (oldest-first, CONFIRMED/NEEDS_RERUN/
ARCHIVE labels) applied retroactively below so the supervision-meeting
narrative has one place that covers everything, not just Phases 1-4.

### Code — bugs found and fixed (chronological)

- [x] `src/data/dataset.py` — **NEEDS_RERUN was correct, now CONFIRMED
      post-fix** (known_issues.md #55)
      `compute_stats()`'s land-pixel-exclusion branch was gated on
      `land_fill_mode == "zero"` only — `"nearest"` silently computed
      `ptho_bot` normalization mean/std over the full grid including
      9473 copied-value land pixels, inflating std +23.4% and
      attenuating every real ocean value ~19% in normalized space.
      Fixed: land now excluded for both modes. Verified: post-fix stats
      bit-identical to zero-mode (mean=0.0229, std=0.2775).

- [x] `scripts/analysis/calibrate_mhw_area_threshold.py` — **NEEDS_RERUN
      was correct, now CONFIRMED post-fix** (known_issues.md #56.1)
      Computed `area_frac_timeseries.npy` from raw/unsmoothed
      `to_anom`/`p90_thresh`, while `quantile_head_recall_v2_all5.py`'s
      def1 uses the smoothed `load_ns_p90()` threshold in the same
      table — mismatched climatology references. Fixed: both `p90_thresh`
      and `mean_clim` now smoothed per-pixel before exceedance. 423/14600
      days (2.9%) flipped sides of the 0.05 threshold on rerun.

- [x] `scripts/eval_onset_skill_quantile_v2.py` — **NEEDS_RERUN was
      correct, now CONFIRMED post-fix** (known_issues.md #56.2, #56.3
      config trap also fixed same pass)
      Never got the #53 per-year Hobday fix — ran `apply_hobday()` on
      each fold's full concatenated test series, and its own comment
      wrongly claimed chronological sort = calendar contiguity. Only
      affects the Onset/Mid-event/No-MHW row split, NOT the pooled "All"
      row. Fixed (per-year loop) + generalized with `--config_dir` so it
      isn't duplicated per model family. A regression was introduced and
      caught during this same fix (removing the old `LEAD = 7` constant
      broke `persist[LEAD:]`, missed by a too-narrow grep) — fixed
      properly by reading `LEAD` from the target config's own
      `lead_time`.

- [x] `configs/partition/local.yaml`, `remote.yaml` — **ARCHIVE**
      (known_issues.md #56.3)
      Leftover v1 configs (`split_mode: kfold`, MSELoss, no
      quantile_head) whose filenames collided with the current v2
      directories `local/`/`remote/` and whose own header invited
      running them. Moved to `configs/partition/_deprecated_v1/` with
      deprecation headers, not deleted (historical record convention,
      #45). No checkpoints depended on them.

- [x] `scripts/ig_partition_quantile.py`,
      `scripts/occlusion_ptho_bot_sanity_check.py` — **NEEDS_RERUN was
      correct, now CONFIRMED post-fix** (known_issues.md #57 P1)
      Both used `test_indices[:max_samples]` — since `stratified_kfold`
      builds `test_indices` in chronological order over non-consecutive
      test years, this concentrated 299/300 "population" IG samples in
      a single year (1985) for fold0. Every IG/occlusion map produced
      before the fix represented one early year, not the 40-year
      test-year span. Fixed via new shared `src/utils/sampling.py::
      stratified_test_sample()` (draws proportionally by target year,
      seeded) — avoids duplicating the fix per script (this project's
      own documented anti-pattern).

- [x] `src/xai/grad_cam.py` — **NEEDS_FIX was correct, now CONFIRMED
      post-fix** (known_issues.md #59 — same bug CLASS as Phase 2's
      composite_ig.py finding, #9/#19-adjacent but for head-conflation
      not MHW-labeling)
      `AttentionGradCAM.compute()` backward()'d the raw `(batch,2)`
      [mean, log_var] output for `gaussian_nll=True` models without
      selecting a column. Fixed: added a `head` parameter
      (`"mean"`/`"quantile"`/legacy `"mean_mse"`), verified
      backward-compatible with all 4 existing callers (none pass `head`,
      all predate `gaussian_nll` models so fall through unchanged).

- [x] `scripts/gradcam_quantile_partition.py`,
      `scripts/gradientshap_quantile_partition.py`,
      `scripts/eval_recall_v2_partition.py`,
      `scripts/analysis/{persistence_recall_baseline,
      quantile_calibration_check,lead_time_sweep_model_vs_persistence,
      raw_ptho_bot_coastal_check,ig_coastal_decay_check}.py`,
      `scripts/eval_test_metrics_from_best_ckpt.py` — **NEW, CONFIRMED**
      Built this session, each dry-run/smoke-tested before any GPU
      spend, all reused this project's canonical `best_ckpt()`/
      `load_model_config()`/`LazyDataModule` pattern rather than
      reimplementing (avoiding the exact Phase 5 consolidation issue
      already flagged for `ensemble_skill.py`/`run_xai_ensemble.py`/
      `ig_masked_batched.py`/`thermal_inertia_test.py`'s local
      `best_ckpt()` copies).

### Findings requiring a correction to a previously-stated conclusion

- **known_issues.md #52's "12.2x enrichment... survives the artifact
  control" claim was itself an artifact** of the sampling bug above,
  caught while re-running IG with the fix. A direct raw-data check (no
  model, no attribution method) gives the true NS-box open-water
  enrichment as **1.39x**, not 12-20x — every gradient-based attribution
  method (IG, GradientSHAP, GradCAM) overstates this, IG/GradientSHAP
  most severely. `land_fill_mode=nearest`'s attributions land much
  closer to the 1.39x truth than the committed (zero-fill) model's.
  Full 4-method + raw-truth table in `docs/narrative.md`'s Aug 22 XAI
  battery entries.

### Decision made this phase

**`land_fill_mode=nearest` adopted project-wide** (full, local, remote,
lead-time sweep) — user's call after weighing the performance cost
(confirmed real: r 0.8657->0.8237 on fold0, not a normalization
artifact) against the coastal-artifact reduction (confirmed via 4
independent XAI methods + a frozen-weight swap ablation). 34 GPU jobs
(folds1-4 of the main model, local/remote 5-fold arrays, lead-time
sweep at 3/5/14/30d x 5 folds) launched and completed cleanly. Full
post-batch analysis done: pooled r + def1/def2 recall/precision/FPR per
family, a persistence recall/precision/FPR baseline (so def1/def2
numbers are interpretable, not bare percentages), quantile-head
calibration (coverage collapses on extreme days, worsening with lead),
and the final lead-time-sweep-vs-persistence figure (NO crossover in
raw r at any lead — confirms the negative answer to Aug 21's "localizar
el crossover" question — but a real, growing recall/precision TRADE
favors the model at longer leads).

### Not yet done (deferred deliberately, not forgotten)

1. XAI battery (IG/GradCAM/GradientSHAP/occlusion) only ran on fold0 of
   the main model family (committed + nearest) — not extended to
   folds1-4, local/remote, or the lead-time-sweep families. Fold0's
   triangulation already answers the physical-vs-artifact question the
   whole investigation was for; further folds would mainly firm up
   precision on numbers already directionally settled.
2. The onset/transition-day skill fix (`eval_onset_skill_quantile_v2.py`,
   per-year Hobday) was only run for the committed and nearest full
   models — not for local/remote/lead-sweep. No request for this yet.
3. Granger causality (item 4, composite/causal analysis) remains
   methodologically broken (autocorrelation false-positive, p=0.0000
   everywhere) — offered a fix (differencing/pre-whitening), never
   confirmed as wanted. Explicitly the user's own lowest priority
   ("no es un resultado de ML vendible").
4. Three small, real P2 issues documented but not fixed (known_issues.md
   #57 P2-1/2/3): NS box defined with two different lat/lon boxes across
   modules; `compute_stats()`'s "train data only" claim has a small real
   leak (~1% std); CLIM (0.1°)/DATA (0.5°) grid resolution mismatch,
   unregridded. All confirmed <0.05°C effect, low priority.
5. The planned physical folder reorg (`experiments/partition/README.md`
   and `configs/partition/README.md`'s "Planned reorg" sections) is now
   ACTIONABLE — all 34 jobs that were blocking it have finished. Not
   done yet; still just an index/README, not a physical move.

**Status: phase complete for its own stated scope (land_fill decision +
34-job batch + XAI triangulation); items above are known, deliberate
deferrals, not gaps in this phase's own execution.**

---

## Permanent / no phase needed
- feedback_ask_before_acting.md — KEEP (standing rule)
- feedback_xai_and_loss.md — KEEP (standing rule)
- user_contact.md — KEEP (permanent reference)
- project_fdl2026_slides.md — KEEP_AS_HISTORICAL (past event record, FDL
  application, no scientific claims to verify)