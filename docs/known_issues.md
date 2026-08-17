# Known issues — audit checklist

Use this as the lupa (magnifying glass) for auditing any script or memory
document in this project. For each item found, cite the exact line as
evidence — do not assume.

1. **Target definition**: `to_anom = SST - mean_clim(DOY)`, NOT `SST - P90`.
   The P90 threshold is only used for MHW classification, never for the
   regression target.

2. **`land_mask` convention**: `1 = ocean`, `0 = land` (inverted from the
   intuitive reading). Must be inverted before use as a boolean mask.
   Note (Fase 4-bis): `merged_daily.nc` contains TWO separate land masks —
   `land_mask` (SST/atmosphere, used by the scalar model and SSTAtm/Atm
   spatial variants) and `land_mask_tbottom` (ptho_bot-specific, 572 pixels
   differ from `land_mask`, used only by TbotAtm spatial variant). Both follow
   the same 1=ocean convention. Any script working with ptho_bot at spatial
   resolution must use `land_mask_tbottom`, not `land_mask` — using the wrong
   one silently masks/unmasks the wrong 572 pixels.

3. **Checkpoint filename regex**: must handle a trailing dot in
   `val_loss=X.ckpt` filenames — a naive regex can silently fall back to a
   wrong default instead of raising an error.

4. **GNLL vs MSE**: never compare two models' skill if they were trained
   with different loss functions — the comparison is confounded by
   optimization difficulty, not just by the experimental condition.

5. **Skill aggregation method**: any reported r must state explicitly
   whether it's mean-of-folds or pooled-across-folds — the two are not
   interchangeable and can give different numbers.

6. **|IG| vs signed IG**: any variable-importance ranking must state which
   one was used. Absolute IG can inflate high-variance regions (e.g. Gulf
   Stream) without them being causally necessary.

7. **`p90_thresh` smoothing**: stored in `sst_climatology_doy.nc` WITHOUT
   the 31-day circular smooth. The smooth is applied only at evaluation
   time (`eval_onset_skill.py::load_ns_p90`). Do not assume the stored
   file already includes it.

8. **No unbatched per-sample loops** for IG/inference — this caused a real
   ~2,400 core-hour loss (Aug 11). Vectorize or batch.

9. **MHW/onset definition**: must include persistence ≥5 consecutive days
   + gap-merging ≤2 days (Hobday et al. 2016), not just a raw threshold
   crossing (`to_anom > 0` or `to_anom > p90` alone is NOT sufficient).

10. **CCM unreliable** with daily autocorrelated ocean data — saturates at
    rho=1.0, no discriminative power. Do not use CCM results without this
    caveat; Granger + masking are the working causal evidence instead.

11. **No silent fallbacks**: any `try/except` or regex-based parsing must
    raise or warn on failure — never silently default (e.g. `if m else 0.0`).

12. **Experiment naming**: old convention `noSST / SST / deepSST / baseline`
    is superseded by `Atm / SSTAtm / TbotAtm`. Any document using the old
    names needs translation, not just a number update.

13. **Forbidden data file**: `merged_daily_deepSST.nc` must never be used —
    superseded by `merged_daily.nc`. Any config or doc referencing the old
    file is stale.

14. **Gulf Stream narrative superseded**: "Gulf Stream Tbot is the dominant
    hotspot" (from IG spatial maps) is contradicted by the perturbation/
    masking experiment — masking Gulf Stream Tbot does not degrade skill,
    masking local NS Tbot does. IG reflects association (input magnitude),
    not causal necessity. Any doc still claiming GS as "dominant driver"
    needs correction or explicit historical framing.

15. **`LazyDataset` attribute name for land mask is `is_land`**: any script
    that accesses the land mask boolean from a `LazyDataset` instance must use
    `dataset.is_land`, not `tierra_mask` or any other alias. Using the wrong
    name causes an `AttributeError` crash at runtime. Found in `run_xai.py`
    line 517 (Aug 2026 audit).

16. **Hardcoded architecture hyperparameters at checkpoint load**: when
    instantiating a model for checkpoint loading (`load_from_checkpoint`), all
    architecture parameters (`in_channels`, `cnn_features`, `lstm_hidden`,
    `lstm_layers`, `temporal_features`, `dropout`) must come from
    `config.get(...)`, not hardcoded. Because this codebase loads checkpoints
    with `strict=False` (standard pattern across scripts), a mismatch does
    NOT raise a shape error — it silently skips the incompatible parameters,
    leaving them randomly initialized. The model runs without any error and
    produces plausible-looking but wrong predictions. This is more dangerous
    than a crash, not less. Found in `run_xai.py` line 310
    (`temporal_features=3` hardcoded).

17. **External service identifiers must not be hardcoded**: WandbLogger
    `entity`, `project`, and any equivalent external-service parameters must
    come from config or env var, never be literal strings in the script. A
    hardcoded entity silently logs to the wrong account after an org rename or
    credential change. Found in `train.py` line 111.

18. **"Vectorised" comments must be accurate**: if a code block is labelled
    `# Vectorised OLS` (or similar), the implementation must use NumPy/PyTorch
    batch ops, not a Python loop. A misleading comment hides a performance
    problem and teaches the wrong pattern to readers. Found in
    `dataset._compute_trend` (the "Vectorised OLS" block uses a pixel-wise
    `for i in range(lat): for j in range(lon):` loop).

19. **`to_anom > 0` binary MHW label** (specific instance of #9): using the
    raw value of the normalized regression target (to_anom = 0 on the
    normalised scale) as the MHW/non-MHW split labels roughly half of all
    days as MHW (~51%), which is physically impossible under Hobday (expected
    ~10%). Binary skill metrics (ETS, HSS, POD, FAR) computed against this
    label cannot be compared with observation-based MHW statistics and are
    meaningless as reported numbers. The correct MHW label requires p90(DOY)
    exceedance followed by the persistence + gap-merging filter (issue #9).
    Found in `ensemble_skill.py` lines 235-236, `composite_ig.py` docstring
    line 4, and any script that sets `thr=0.0` on the normalised target
    without the Hobday persistence filter.

20. **Dead-code wrappers `analyze_integrated_gradients()` /
    `analyze_gradcam()`**: these functions exist in
    `src/xai/integrated_gradients.py` (line 82) and `src/xai/grad_cam.py`
    (lines 187-195) but are never called from any active script. Both
    contain two additional bugs beyond being dead: (a) they use `data == 0`
    as a land-mask proxy, which incorrectly masks ocean pixels with near-zero
    values — the correct source is `dataset.is_land` (issue #15); (b) they
    run unbatched per-sample prediction loops (issue #8). Do not import or
    invoke either function; they are candidates for deletion once issue #15
    back-checks across Phase 3/4 scripts are complete.

21. **Pre-June-2026 TbotAtm results trained on absolute ptho_bot, not
    anomaly**: models trained before the June 29, 2026 preprocessing fix
    (see project_anomaly_inconsistency.md) used merged_daily_deepSST.nc
    (now merged_daily_deepSST_OLD.nc), where ptho_bot was raw ICON-COAST
    absolute temperature (range -1.8 to 32°C), not a DOY-climatology
    anomaly. Any TbotAtm result (skill, IG attribution, XAI findings)
    dated before ~June 29, 2026 was produced by a model that saw a
    fundamentally different variable than what "ptho_bot" means today.
    This includes the historical r=0.77/ETS=0.47/AUC=0.955 ensemble
    numbers and likely the original IG per-year "T_bottom dominance"
    finding — both need re-verification against a model trained on the
    corrected file before being cited anywhere.

22. **Silent routing to forbidden data file as fallback**: `patched_config_path()`
    in `thermal_inertia_test.py` automatically routes to
    `merged_daily_deepSST_OLD.nc` (forbidden, issue #13) when
    `merged_daily_deepSST.nc` is not found, writing a temp config instead of
    raising. This is more dangerous than issue #13 alone — it actively
    silently substitutes the wrong file rather than failing to find the
    right one. As of Aug 13 2026, the code path that would use this (Step 2:
    per-year IG(T_bottom) for the tau-vs-IG correlation) never completed a
    full run, so no existing result is contaminated by this — but it must
    be fixed before that step is ever rerun.
    
23. **NumPy RNG API must match exactly across all split-generating code**:
    `np.random.RandomState(seed)` (legacy) and `np.random.default_rng(seed)`
    (modern) produce DIFFERENT permutations from the same seed — neither is
    wrong, but reconstructing `datamodule.py` kfold logic with the wrong
    API silently assigns different years to each fold, breaking
    comparability between models meant to share the same split (e.g.
    partition full/remote/local). Always verify reconstructed split logic
    against actual test-year lists from existing checkpoints (not just
    sample counts, which can coincidentally match — n=2854 matched both
    the wrong and the right permutation here). The canonical RNG is
    `np.random.RandomState(self.seed)` — do not "modernize" it without
    verifying against existing checkpoints first.

24. **Stale buggy-normalization checkpoints in spatial_forecast — deleted Aug 16 2026**:
    `spatial_forecast/experiments/runs/*/checkpoints/` contained 60 checkpoints
    (437 MB) from early runs of the SAME `train_spatial.py` / 2D pipeline that
    produced anomalously low val_loss (< 0.10) due to a broken target normalization:
    SLURM logs show "Target std (ocean mean): nan" for all affected runs, which
    likely caused division by NaN somewhere in the 2D target normalization path
    (root cause not further investigated — confirmed irrelevant since the runs were
    deleted and never used in any result). These are NOT a separate scalar-target
    experiment; the pipeline, variables, and config were identical to the valid runs.
    Affected dates: Jun 25–26 2026 (Atm, val_loss ~0.016), Jun 27 2026 (TbotAtm,
    val_loss ~0.001 — additionally PRE-FIX, contaminated by issue #21), Jun 30 2026
    (Atm/SSTAtm/TbotAtm, val_loss ~0.016 — post-fix in date but NaN-normalized).
    The valid 2D runs (Jul–Aug 2026, val_loss 0.26–0.63) ran the same script with
    corrected normalization. Since `best_ckpt()` selects minimum val_loss, any
    external eval script would have silently picked these stale checkpoints over the
    correct ones. Grep evidence: zero references to any of these checkpoint filenames
    in spatial_forecast/ or exprecursors/ (scripts, configs, SLURM logs). Existing
    figures and test_preds.npy were generated by Aug 4–10 training sessions using
    `trainer.checkpoint_callback.best_model_path` (session-scoped), so were never
    contaminated. Decision: deleted. Valid 2D checkpoints remain untouched.

## How to use
For each script or memory doc reviewed, check against all 24 items and
report: applies / does not apply / unclear — with the specific line as
evidence for each "applies".