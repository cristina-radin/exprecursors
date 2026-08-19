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
   `land_mask` (SST/atmosphere) and `land_mask_tbottom` (ptho_bot-specific,
   572 pixels differ from `land_mask`). Both follow the same 1=ocean
   convention. Any script working with ptho_bot must use `land_mask_tbottom`,
   not `land_mask` — using the wrong one silently masks/unmasks the wrong 572
   pixels.
   **CORRECTION (Aug 17 2026)**: this note originally said the scalar
   pipeline (`src/data/dataset.py`) was fine using `land_mask` for
   everything — that was never actually verified and was wrong.
   `dataset.py` applied `land_mask` (not `land_mask_tbottom`) to `ptho_bot`
   in every scalar-pipeline TbotAtm/SSTAtm+ptho_bot experiment. Confirmed:
   all 572 mismatched pixels are ones `land_mask` marks as land but
   `land_mask_tbottom` marks as ocean — i.e. dataset.py was zeroing 572 real
   ocean pixels of ptho_bot on every sample, every experiment, since the file
   gained `land_mask_tbottom`. Consequence, found while investigating a user
   report of spurious positive/negative sign contrast right at the coastline
   in ptho_bot signed-IG maps (`experiments/xai_partition_full/
   ig_spatial_ptho_bot.png`): real ptho_bot values sitting directly next to
   572 permanently-zeroed pixels create an artificial step discontinuity —
   the CNN's gradient reacts to that edge (an artifact of the missing
   data), not to physically meaningful bottom-temperature structure. This is
   NOT a spatial-resolution attribution nuance (see #26) — it is data
   corruption at those 572 pixels in every training sample.
   Same severity class as #21 (affects every TbotAtm scalar-pipeline result
   to date, training and XAI alike — not just plotting).
   **Fixed Aug 17 2026**: `dataset.py` now builds a per-variable
   `self.land_masks` dict — `ptho_bot` uses `land_mask_tbottom` (raises
   `ValueError` if absent, no silent fallback), all other ocean variables
   keep using `land_mask` as before. Applied in both `compute_stats()` and
   `__getitem__()`. Verified: `is_land.sum()=10045` (SST mask) vs
   `land_masks["ptho_bot"].sum()=9473` — exactly the expected 572 pixel
   difference.
   **Superseded, not left half-fixed**: the original `TbotAtm_full_gnll_seed42`
   5-fold run (fold0 complete, folds 1-4 job 14202115 already `RUNNING`, all
   pre-fix/max-pool) was cancelled outright (`scancel 14202115`) rather than
   left to finish inconsistently. All 5 folds' `checkpoints/`, `wandb/`,
   `loss_curves.png` deleted and the run relaunched from scratch as job
   14202206 (`--array=0-4`, all 5 folds together), with BOTH this land-mask
   fix AND `pooling: avg` (see #27) applied uniformly — decided together
   since #27 was adopted for "next experiments" at the same time this fix
   was found, and mixing partially-fixed folds within one 5-fold set would
   repeat the exact kind of inconsistency #23/#4 warn against.
   **ROOT CAUSE FOUND Aug 18 2026 — corrects the explanation above**: the
   572-pixel mismatch is NOT a genuine physical/bathymetric difference
   between two ICON fields (that was an unverified guess). It is a **grid
   alignment bug** in one source file. Traced the full preprocessing chain
   (`/p/project1/hai_1127/inputs/daily/preprocess_data/preprocess_all.py`,
   outside this git repo): both `land_mask_05.nc` (source of `land_mask`)
   and `land_mask_tbottom_05.nc` are regridded onto `TARGET_LAT`/`TARGET_LON`
   (the ERA5 grid) via `.interp(..., method="nearest")` in `load_mask()`.
   Confirmed directly from the raw coordinate arrays:
   `TARGET_LAT[:3] = [0.0, 0.5, 1.0]` (141 points) vs
   `land_mask_05.nc` `lat[:3] = [0.25, 0.75, 1.25]` (**140** points, 0.25°
   offset) vs `land_mask_tbottom_05.nc` `lat[:3] = [0.0, 0.5, 1.0]` (141
   points, already exactly ERA5-aligned). `land_mask_05.nc`'s nearest-neighbour
   snap onto a grid it doesn't natively align with produces a systematic
   ~1-pixel coastal shift, uniformly across the whole domain — exactly the
   thin, near-continuous coastal band observed on every coastline (Labrador
   36px, Caribbean 62px, NS box 66px, Iberia 27px, US East Coast 6px, Gulf of
   Mexico 1px — spread everywhere, not concentrated). `land_mask_tbottom_05.nc`
   only avoids this because it happens to already be correctly aligned, not
   because ptho_bot has a genuinely different valid-ocean footprint.
   **Fix applied Aug 18 2026**: single unified mask, sourced from
   `land_mask_tbottom_05.nc` for both output variables.
   - `preprocess_all.py` edited (`mask_sst = mask_bot`, `mask_sst_file` no
     longer read) so any future full regeneration from raw sources doesn't
     reintroduce the bug.
   - Current `merged_daily.nc` NOT modified in place (blocked by the safety
     classifier as a high-risk edit to a shared 9.9GB production file — the
     user confirmed the safer path). Instead: `land_mask_tbottom_05.nc`'s
     values copied into a NEW file,
     `/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily_v2.nc`
     (full `cp`, then only the `land_mask` variable rewritten in place via
     `netCDF4 r+` — cheap, no need to recompute `to_anom`/`ptho_bot`/ERA5
     anomalies from the 40+ GB raw sources). Verified: `land_mask` in v2 now
     equals `land_mask_tbottom` exactly (18868 ocean / 9473 land, was 18296/
     10045); all other variables (`to_anom`, `ptho_bot`, `u10`, `v10`, `msl`,
     `ssr`, `target`, `land_mask_tbottom`) confirmed byte-identical to the
     original `merged_daily.nc` (spot-checked first 50 timesteps + full mask
     arrays). Backup of both original mask arrays saved at
     `docs/deleted_manifests/land_mask_backup_2026-08-18.npz`.
   - **`merged_daily.nc` (original) is UNCHANGED and still the file every
     config/env var points to.** `merged_daily_v2.nc` exists alongside it but
     nothing reads it yet — switching the canonical file is a project-wide
     decision (same class as #21) not made unilaterally here. Practical
     impact meanwhile: the `dataset.py` fix earlier in this issue already
     makes ptho_bot use the correct (already-aligned) `land_mask_tbottom`
     directly, bypassing the buggy `land_mask` variable for the one ocean
     variable current TbotAtm experiments actually mask — so `merged_daily_v2.nc`
     mainly matters for any SSTAtm/Atm pipeline that masks `to_anom` or other
     variables via the generic `land_mask` field. Not yet audited which
     configs do that.
   **DECISION (user, Aug 18 2026): parked, no rerun now.** `merged_daily.nc`
   stays canonical; `merged_daily_v2.nc` is ready and verified for whenever a
   rerun is next needed for any other reason — use `_v2` at that point, not
   the original. Not an open action item until then. Cross-referenced in
   `audit_plan.md` ("Pending decision" section) and `data.md`.

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

25. **`gaussian_nll` is a dead parameter — no GNLL model or loss exists anywhere in the
    codebase, "gnll"-named experiments are plain MSE mislabeled — deleted Aug 17 2026**:
    `CNNLSTMModel.__init__` ([src/models/cnn_lstm.py:119](../src/models/cnn_lstm.py#L119))
    accepts `gaussian_nll: bool = False` but never stores it (`self.gaussian_nll` does not
    exist) and never branches on it — the FC head is unconditionally
    `nn.Linear(64, 1)` (single scalar output, never mean+variance). No
    `nn.GaussianNLLLoss`, `F.gaussian_nll_loss`, `logvar`, or `out[:, 1]` occurs
    anywhere in `src/` or `scripts/` (repo-wide grep, zero hits). Loss selection
    in `CNNLightningModule` ([cnn_lstm.py:232](../src/models/cnn_lstm.py#L232)) is
    `nn.L1Loss() if loss_fn=="MAELoss" else nn.MSELoss()` — any other string,
    including "GNLLLoss", silently resolves to MSE (this half was already flagged
    as NF-3 in the Phase 1 audit, but assumed to be latent/untriggered).
    Confirmed 100% against the actual checkpoint weights, not just config/code:
    `TbotAtm_full_gnll_seed42_fold0/checkpoints/cnn-lstm-epoch=14-val_loss=0.1826.ckpt`
    has `model.fc.3.weight` shape `(1, 64)` (one output, not two) and its saved
    `hyper_parameters` dict contains `'loss_fn': 'MSELoss'` — the checkpoint itself
    records that it was trained with plain MSE, not GNLL. Every config with
    `gaussian_nll: true` checked also has `loss_fn: MSELoss` set explicitly (not
    even an attempted "GNLLLoss" string). `scripts/temporal_shuffle_eval.py:72`
    reads `model.gaussian_nll`, which would raise `AttributeError` if that code
    path (`shuffle_window=True`) were ever exercised — further evidence the
    attribute never existed and was never exercised end-to-end.
    Consequence: every "gnll"-named experiment directory (~200 found:
    `*_lstmonly_gnll*`, `*_gnll_masked*`, `*_gnll_local*`, `*/full_gnll/*`,
    `lead_sweep/*_gnll_*`) is architecturally and functionally identical to an
    MSE run — the name is the only thing that differs. Any prior claim that
    referenced a "GNLL" result (skill number, uncertainty estimate, or the
    `project_egu_feedback.md` / EGU reviewer "compare architectures incl.
    Gaussian NLL" item) was never actually testing GNLL. Decision: all 285
    directories (87 GB, `experiments/{gaussian_nll,lead_sweep,masked,local_only,
    partition}/*gnll*`) deleted Aug 17 2026 — full path manifest saved at
    `docs/deleted_manifests/gnll_dirs_deleted_2026-08-17.txt` before deletion
    (not recoverable from version control — `experiments/` is git-untracked).
    Real GNLL (2-output
    head + `nn.GaussianNLLLoss`) implemented from scratch afterward; see
    `docs/narrative.md` Architecture choices section for the first valid result.

26. **Spatial IG maps show a real ~8px periodic grid artifact from the CNN's
    MaxPool stride — confirmed via FFT, NOT fixed by SmoothGrad (Aug 17 2026)**:
    All signed IG spatial maps (`scripts/ig_signed_partition.py`, any partition,
    any variable) show a fine grid/checkerboard pattern, most visible in
    low-signal open-ocean regions (it is masked by real signal in high-magnitude
    regions like the Gulf Stream — do not judge "is it there" from a strong-signal
    region, check an open-ocean band instead). Confirmed to be a genuine periodic
    artifact, not random noise or a code bug:
    - 1D FFT along longitude of the column-mean profile in the open-ocean band
      (lon -40..-15, lat 10..60) of `ig_spatial_v10.png`'s underlying array shows
      a **dominant spectral peak at period ≈ 8.5 px** (20.6% of total spectral
      power in that single frequency bin for TbotAtm Remote fold-mean; harmonics
      at ~4.25 px and ~2.1 px also present). This is not a broadband-noise
      spectrum (which would decay smoothly across frequencies); it is a discrete
      peak.
    - Period 8.5 px ≈ 2×2×2 = 8, matching exactly the cumulative stride of the
      3 `nn.MaxPool2d(2)` layers in `CNNEncoder`
      ([src/models/cnn_lstm.py:39,43,47](../src/models/cnn_lstm.py#L39)).
      MaxPool backprops gradient only through the argmax position in each 2×2
      window; three cascaded poolings mean gradient can only vary at ~8px
      granularity, not per-pixel — a well-documented property of vanilla
      gradient/IG saliency through strided pooling (motivates SmoothGrad in the
      literature for *random* per-sample gradient noise).
    - **Ruled out as the cause**: input-regrid resolution (checked: present
      equally in native-0.5° ERA5 variables that never went through regrid, see
      conversation Aug 17 2026), IG implementation bug (completeness axiom
      Σattr ≈ f(x)−f(baseline) holds to 5-13% on 4/5 test samples — expected
      numerical error for a 50-step Riemann-sum IG — with one 30% outlier
      explained by a near-zero denominator, not a broken implementation;
      model weights/buffers all finite, no NaN/Inf), and land/coast masking
      (present regardless of `plot_mask`).
    - **SmoothGrad tested and does NOT fix it** (diagnostic run: TbotAtm Full
      fold0 checkpoint, 8 test samples, 6 noise draws/sample at σ=0.15,
      n_steps=25 — reduced from production n_steps=50 for speed, CPU, ~18 min):
      power in the 8px band went from 15.2% (vanilla) to **20.4% (SmoothGrad)**
      — slightly worse, not better. This is expected once the cause is
      understood: SmoothGrad cancels *random* per-sample gradient noise by
      averaging over noisy inputs; this artifact is *structural* (same
      argmax-routing grid regardless of input), so noise-averaging does not
      cancel it and can even reinforce the dominant routing path.
    - **Practical mitigation in place**: the display-only Gaussian smoothing
      already added to `ig_signed_partition.py::merge_and_plot`
      (`_smooth_ignore_nan`, σ=1 px) suppresses this visually; it does not
      remove the underlying bias. Any interpretation of spatial IG structure
      below ~8px should be treated as a pooling-grid artifact, not a real
      finding, regardless of smoothing.
    - Not investigated further: architectural fixes (e.g. avg-pool or
      strided-conv instead of max-pool, occlusion-based attribution instead of
      gradient-based) would remove the root cause but require retraining — out
      of scope unless requested.

27. **`pooling` option added to `CNNEncoder`/`CNNLSTMModel` — decision to use
    `avg` in future experiments (Aug 17 2026)**: direct follow-up to #26.
    `pooling: "max"|"avg"` added to `CNNEncoder.__init__` / `CNNLSTMModel.__init__`
    ([src/models/cnn_lstm.py](../src/models/cnn_lstm.py)), default `"max"` —
    existing configs/checkpoints unaffected, no config changes needed for past
    runs. `scripts/train_partition.py` reads `config.get("pooling", "max")`.
    Decision: use `pooling: avg` going forward to avoid the #26 artifact
    (avg-pool backprops gradient through the whole 2×2 window instead of a
    single argmax position, removing the routing asymmetry that produces the
    ~8px periodic grid). Not yet propagated to any eval/IG/XAI script that
    reconstructs `CNNLSTMModel` from a config for checkpoint loading (e.g.
    `ig_signed_partition.py`, `eval_ig.py`, `composite_ig_signed.py`,
    `ig_masked_batched.py`) — pooling layers have no learnable weights, so
    loading an avg-pool checkpoint into a script that still hardcodes/defaults
    to max-pool will NOT raise an error (`strict=False` pattern, same failure
    mode as #16) — it will silently evaluate with the wrong pooling type. Must
    add `pooling=cfg.get("pooling", "max")` to every such script before
    evaluating any avg-pool checkpoint.
    First experiment: `TbotAtm_full_gnll_seed42_fold{0-4}`, job 14202206,
    combines this with the #2 land-mask fix (see #2 "Superseded" note) — both
    landed together, relaunched as one clean 5-fold set with neither fix
    partially applied.

28. **Double climatology subtraction — `dataset.py` re-anomalised every
    variable except `to_anom`, on top of data that arrives already anomalised
    — fixed Aug 18 2026**: `preprocess_all.py` (outside this repo, at
    `/p/project1/hai_1127/inputs/daily/preprocess_data/`) anomalises **all**
    variables before writing `merged_daily.nc`/`merged_daily_v2.nc` — `to_anom`
    (CDO `ydrunmean,11`) and `ptho_bot`/`u10`/`v10`/`msl`/`ssr` (Python DOY
    climatology, ±5-day window, ref 1985-2014) — confirmed by reading
    `preprocess_all.py` lines 152-174 directly; its own docstring (line 14)
    states "dataset.py does NOT subtract climatology — all vars are already
    anomalies." `dataset.py`'s `_compute_clim()` ignored this and computed a
    **second** DOY climatology for every variable except `to_anom`
    (`vars_to_anom = [v for v in self.variables if v != "to_anom"]`), using a
    **different, narrower window** than preprocessing: `half_w = clim_window
    // 2 = 5 // 2 = 2` (±2 days) vs preprocessing's ±5 days — so the second
    subtraction was not an exact no-op. This logic existed unchanged in
    `dataset.py` since 13 Mar 2026 (`git log`), predating the 29 Jun 2026
    regeneration of `merged_daily.nc` with all-variables-anomalised — i.e. it
    was correct when written and went stale silently when the upstream
    contract changed, with nothing to catch the mismatch (no test in `tests/`
    covers climatology/anomalisation; `grep -l "clim\|anomal" tests/*.py` →
    zero hits).
    **Quantified** (computed directly on `merged_daily.nc`, reproducing
    `dataset.py`'s exact algorithm): RMS of the spurious second climatology
    relative to each field's own std, ocean pixels only —
    `ptho_bot` 1.78%, `u10` 6.51%, `v10` 6.74%, `msl` 6.56%, `ssr` 6.16%.
    Systematic and deterministic (fixed function of day-of-year, same every
    year), not random noise.
    **Fix**: `dataset.py` module docstring corrected; `_compute_clim()`
    hard-codes `vars_to_anom = []` (kept as a method, not deleted, in case a
    future dataset file ever ships absolute values again — see its docstring).
    **Verified two ways**: (1) direct instantiation of the real `LazyDataset`
    class against `merged_daily_v2.nc` shows `self.clim_means == {}` and
    `__getitem__`'s output matches `(raw − mean) / std` to `0.00e+00` for all
    5 variables — i.e. zero climatology subtraction happens, not inferred from
    reading the code; (2) `scripts/train_partition.py --fast_dev_run 2` runs
    clean end-to-end post-fix, no NaN/crash, and the printed
    "Computing day-of-year climatology..." block (previously present, and the
    source of the `ptho_bot: clim mean=nan, std=nan` print that triggered this
    investigation) no longer appears.
    **Scope**: affects every TbotAtm/ERA5 experiment trained on
    `merged_daily.nc` since 29 Jun 2026 (kfold, partition remote/local/full,
    the GNLL run in item 27) — not specific to any one job. Results from
    before that date used absolute (non-anomalised) ERA5/`ptho_bot` in the
    file, so `dataset.py`'s subtraction was correct at the time.

29. **`attention_only` and `lstm_attention` were the same architecture —
    every existing `attentiononly`-labeled checkpoint is mislabeled — code
    fixed Aug 18 2026, checkpoints not**: `CNNLSTMModel.forward()`
    unconditionally ran `self.lstm(features)` for every `arch` value; `arch`
    only branched what happened *after* the LSTM — last timestep for
    `"lstm_only"`, `self.attention(lstm_out)` for anything else, including
    both `"attention_only"` and `"lstm_attention"`. So a checkpoint trained
    with `arch: attention_only` was structurally identical to one trained
    with `arch: lstm_attention` (CNN→LSTM→Attention in both cases) — there
    was no true CNN+Attention-without-LSTM path anywhere in the scalar
    pipeline. Affects real trained data, not just a hypothetical: dozens of
    `*_attentiononly_*` run directories exist alongside `*_lstmattention_*`
    ones under `experiments/architecture_variables/` and
    `experiments/multiseed/` (5 folds × {Atm, SSTAtm, TbotAtm} and 5 folds ×
    5 seeds × {SSTAtm, TbotAtm} respectively) — any plot comparing the two
    labels as if they were an attention-vs-no-attention ablation is comparing
    two runs of the same architecture, not a real architectural difference.
    Found while reviewing architecture-comparison plots ahead of a meeting
    (Aug 18 2026); flagged by the user, confirmed by reading the code.
    **Fixed 2026-08-18 (code only, no retraining, no relabeling)**:
    `CNNLSTMModel.__init__` ([src/models/cnn_lstm.py:142](../src/models/cnn_lstm.py#L142))
    now sets `self.lstm = None` and `context_dim = cnn_features` when
    `arch == "attention_only"`, vs. building the LSTM and using
    `context_dim = lstm_hidden` otherwise; `self.fc`'s input width follows
    `context_dim` instead of being hardcoded to `lstm_hidden`.
    `forward()` ([cnn_lstm.py:192](../src/models/cnn_lstm.py#L192)) runs
    `self.attention(features)` directly (no LSTM call at all) when
    `arch == "attention_only"`. `forward_with_attention()`
    ([cnn_lstm.py:224](../src/models/cnn_lstm.py#L224)) mirrors the same
    branch. Verified with a smoke test (`CNNLSTMModel` on random tensors,
    all three `arch` values): `attention_only` now reports `has_lstm=False`
    and fewer total params than `lstm_attention` (395,426 vs 402,866 for a
    toy config); `lstm_only`/`lstm_attention` output shapes and param counts
    are byte-for-byte unchanged from before the fix (regression-safe).
    `pytest tests/test_splits.py tests/test_masking.py` still passes
    (`tests/test_checkpoints.py` requires `MHW_DATA_FILE`, not set in the
    shell used to verify — unrelated to this change).
    **Not done, deliberately out of scope**: no existing checkpoint was
    retrained, so every `attentiononly`-labeled directory under
    `experiments/architecture_variables/` and `experiments/multiseed/` is
    still the pre-fix, `lstm_attention`-equivalent architecture — do not
    treat them as a real ablation regardless of this fix landing. No plot or
    doc citing those runs was touched or relabeled. A genuine
    CNN+Attention-only checkpoint does not exist yet.
    **New follow-up surfaced by this fix**: `scripts/diag_attention.py`
    calls `lm.model.lstm(features)` directly at
    [lines 88 and 168](../scripts/diag_attention.py#L88) — will raise
    (`NoneType` is not callable) if ever pointed at a checkpoint genuinely
    trained post-fix with `arch: attention_only`. Not urgent since no such
    checkpoint exists yet; must be fixed before `diag_attention.py` is run
    against one.
    Separately (same investigation, not a bug): ConvLSTM exists only in the
    spatial pipeline (`src_spatial/model_spatial.py`), not in the scalar
    pipeline (`src/models/cnn_lstm.py`) — that is why it is absent from
    `experiments/architecture_variables/`, not an oversight to fix here.

## How to use
For each script or memory doc reviewed, check against all 29 items and
report: applies / does not apply / unclear — with the specific line as
evidence for each "applies".