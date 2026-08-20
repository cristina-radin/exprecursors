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

30. **`cnn_features` default drift across scripts that build `CNNLSTMModel`
    directly (not via `load_model_config`) — found during
    `full_gnll_quantile` launch review, Aug 19 2026, not fixed**: 13 scripts
    hardcode `cfg.get("cnn_features", 128)` (`permutation_importance.py`,
    `composite_ig_signed.py`, `ig_signed_partition.py`, `ensemble_skill.py`,
    `composite_ig.py`, `run_xai_ensemble.py`, `gradcam_partition.py`,
    `diag_attention.py`, `plot_variable_scatter.py`, `plot_split_scatter.py`,
    `train.py`, `skill_scores.py`, `run_xai.py`) while 10 others hardcode
    `cfg.get("cnn_features", 256)` (`eval_onset_skill.py`,
    `local_only_skill.py`, `eval_ig.py`, `train_local_only.py`,
    `ig_masked_model.py`, `ig_masked_casestudy_2023.py`, `train_partition.py`,
    `analysis/thermal_inertia_test.py`, `ig_masked_batched.py`,
    `tests/test_checkpoints.py`). 128 matches `CNNLSTMModel`'s own
    constructor default ([cnn_lstm.py](../src/models/cnn_lstm.py), same as
    `checkpoints.py::CNNLSTM_DEFAULTS`); 256 matches what several kfold/
    partition configs (including `full_gnll_quantile/fold{0-4}.yaml`)
    explicitly set. Same silent-mismatch class as issue #16: `cfg.get(key,
    default)` only falls back to the hardcoded default when `key` is absent
    from the config file being read — for any config that DOES set
    `cnn_features` explicitly (all `full_gnll_quantile` folds do, =256) this
    is a no-op and not a live bug. Risk is latent: a future config that omits
    `cnn_features` will silently get 128 or 256 depending on which of these
    24 scripts happens to load it, with `strict=False`-style checkpoint
    loads (issue #16) masking the resulting shape mismatch instead of
    raising. Not fixed here (would mean editing 13 files, out of scope for
    launching the quantile-head training run) — the real fix is migrating
    all 24 direct-`CNNLSTMModel(...)`-construction call sites to
    `load_model_config()` (already the documented single source of truth,
    `checkpoints.py` lines 33-38), not aligning the hardcoded defaults
    against each other.

31. **`loss_fn` was dead code once `gaussian_nll=True` — fixed Aug 19 2026**:
    `CNNLightningModule.__init__` ([cnn_lstm.py:381-395](../src/models/cnn_lstm.py))
    built `self.nll_loss` whenever `gaussian_nll=True` regardless of `loss_fn`'s
    value, and separately built `self.loss_fn` (MAE/MSE) from an unguarded
    `if/elif` that ran even when `gaussian_nll=True` — so a config setting
    `loss_fn: GaussianNLLLoss` (as every `full_gnll*` config does) had that
    value silently ignored; GNLL activation is driven entirely by the
    `gaussian_nll` boolean. Not the same failure mode as the now-superseded
    issue #25 (which was "no real GNLL exists at all") — GNLL was real, just
    `loss_fn` was decorative. **Fix**: `gaussian_nll=True` now requires
    `loss_fn == "GaussianNLLLoss"` exactly, raising `ValueError` otherwise —
    makes a future config typo fail loud instead of silently doing nothing.
    Found by the user during `full_gnll_quantile` Raven launch review.

32. **`test_preds`/`test_targets` not reset between `trainer.test()` calls —
    fixed Aug 19 2026**: `CNNLightningModule.__init__` initialised these as
    empty lists but nothing reset them before a test epoch
    ([cnn_lstm.py](../src/models/cnn_lstm.py)) — harmless today (each fold is
    a separate SLURM process) but would silently accumulate predictions
    across multiple `trainer.test()` calls in the same process (e.g. a
    notebook or an ensemble script). **Fix**: added `on_test_epoch_start()`
    resetting both lists. Found by the user during `full_gnll_quantile`
    Raven launch review.

33. **`num_sanity_val_steps=0` in `train_partition.py` — changed to 2, Aug 19
    2026**: disabled Lightning's pre-training validation sanity check, so a
    broken config (wrong path, missing variable, shape mismatch) wasn't
    caught until after a full first epoch — on this codebase's ~5.4 min/epoch
    (measured on Raven A100, job 29403121) that's a real but modest cost, not
    the ~2,400 core-hour class of issue #8. Changed to `2` (negligible added
    time, runs once before training starts, not every epoch). Found by the
    user during `full_gnll_quantile` Raven launch review.

34. **Reflect padding added to `CNNEncoder`'s 4 `Conv2d` layers, Aug 19-20
    2026**: previously always `padding_mode="zeros"`. Zero padding adds a
    hard discontinuity at the domain boundary on top of the land-mask NaN→0
    edge already there, and the user reported this shows up as boundary-band
    artifacts in IG spatial maps ("cuatro franjas que opacan todos los
    posibles resultados") — distinct from the #26 ~8px pooling-grid artifact.
    `padding_mode: "zeros"|"reflect"` added to `CNNEncoder`/`CNNLSTMModel`
    ([cnn_lstm.py](../src/models/cnn_lstm.py)), default `"zeros"` (existing
    checkpoints unaffected). Wired into `checkpoints.py::CNNLSTM_DEFAULTS`
    (so `model_config.json`/`load_model_config()` persist and restore it
    correctly) and into all 24 scripts that construct `CNNLSTMModel`
    directly — same treatment as `pooling` (#27) to avoid re-creating the
    #16-class silent-mismatch bug for a brand new parameter. Enabled
    (`padding_mode: reflect`) in `full_gnll_quantile` and `full_gnll_focal`
    fold configs. Verified via smoke test (real-sized synthetic tensors):
    `model.cnn_encoder.cnn[0].padding_mode == "reflect"`, full forward+backward
    finite and gradients flow.

35. **`best_ckpt()` silent `float("inf")` fallback — fixed Aug 19 2026**:
    `src/utils/checkpoints.py::best_ckpt()`'s `_val_loss()` helper returned
    `float("inf")` when the `val_loss=X.ckpt` regex failed to match a
    filename, silently ranking that checkpoint as worst instead of raising —
    a direct instance of issue #11 (no silent fallbacks), pre-existing since
    at least the #3 note about trailing dots. **Fix**: raises `ValueError`
    on a non-matching filename instead. Found by the user during
    `full_gnll_quantile` Raven launch review; not triggered by the current
    launch (only used by downstream eval/XAI scripts, not training).

36. **`cnn_features` default drift across 23 direct-`CNNLSTMModel`-construction
    scripts — found, not fixed, Aug 19 2026**: see the full writeup at the
    end of this list (kept as item #30 for continuity with the pre-existing
    numbering at the time this was found — do not renumber).

37. **`quantile_head` and `focal_weight` are alternative extreme-day
    strategies selected by config, not auto-detected — `_step()` branches on
    `self.focal_weight` only after checking it, but a config that sets
    `focal_weight: true` while leaving a stale `quantile_head: true` from a
    copy-pasted `full_gnll_quantile` config would raise loudly at
    `CNNLightningModule.__init__` (mutual-exclusion `ValueError`), not run
    silently with the wrong branch — confirmed by construction
    ([cnn_lstm.py](../src/models/cnn_lstm.py), the `if focal_weight and
    quantile_head: raise` check added alongside `focal_weight` itself).
    Flagged by the user before launch as a risk to double check when copying
    `full_gnll_quantile/fold*.yaml` as a template for `full_gnll_focal` —
    confirmed the `full_gnll_focal` configs set `quantile_head: false`,
    `focal_weight: true`, `return_target_doy: true` correctly. Separately: if
    `focal_weight: true` but `return_target_doy` is missing/false, `_step()`'s
    `x_spatial, x_temporal, y, target_doy = batch` unpack fails with a loud
    `ValueError` (not enough values to unpack) — also a fail-loud, not a
    silent-bug, path.

38. **`focal_weight`/`focal_alpha` not persisted in `model_config.json` —
    known gap, Aug 20 2026**: `save_model_config()`/`CNNLSTM_MODEL_KEYS`
    ([checkpoints.py](../src/utils/checkpoints.py)) cover only
    `CNNLSTMModel` architecture kwargs; `focal_weight`/`focal_alpha` are
    `CNNLightningModule` training-time loss parameters, not needed to
    reconstruct the model for eval/XAI, so this is not a correctness bug.
    But it means reporting these values (e.g. in a paper) requires reading
    the original fold yaml or the W&B run config, not `model_config.json`.
    Flagged by the user, not fixed (deliberately out of scope — would mean
    widening `CNNLSTM_MODEL_KEYS`'s contract to cover training-time params,
    not just architecture).

39. **`p90_by_doy` not restored on checkpoint load — known gap, Aug 20
    2026**: `CNNLightningModule.__init__` does
    `self.register_buffer("p90_by_doy", ...)` when `focal_weight=True`, but
    `save_hyperparameters(ignore=["model", "p90_by_doy"])` excludes it from
    the saved hyperparameters, and no other mechanism restores it —
    `load_from_checkpoint()` on a focal-weighted checkpoint leaves
    `p90_by_doy` as whatever (or `None`) the caller passes at reconstruction
    time. Does not affect current eval/XAI scripts (they only ever run
    inference through `forward()`/GNLL, never recompute the focal-weighted
    training loss at eval time). Would need `p90_by_doy` recomputed
    (`load_ns_p90()`) and passed explicitly by any future script that wants
    to report focal-weighted loss values on held-out data. Flagged by the
    user, not fixed.

40. **`mean_clim.nc`'s `mean_clim` never receives the 31-day smooth Hobday
    et al. 2016 prescribes for the climatological mean — verified against
    the source, fixed as opt-in Aug 20 2026**: `to_anom`/`target` (the
    model's actual regression target, `docs/data.md`) are computed as
    `SST - mean_clim`, where `mean_clim` is the raw CDO `ydrunmean,11`
    output (±5-day window climatological mean), reference period
    1985-2014. Hobday et al. 2016 ("A hierarchical approach to defining
    marine heatwaves", Progress in Oceanography 141, 227-238) prescribes
    smoothing BOTH the climatological mean and the p90 threshold curve
    with an additional 31-day moving average after the windowed
    calculation — confirmed against the paper directly, not from memory.
    `p90_thresh` already gets this smooth (at runtime, via
    `load_ns_p90()`, known_issues #7) but `mean_clim` never has, since
    this repo existed — a real, previously-undocumented deviation from
    the canonical methodology, affecting `target` project-wide (not just
    MHW classification).
    **Quantified** (NS-box mean, computed directly on
    `sst_climatology_doy.nc`, no raw-source regeneration needed): delta
    between unsmoothed and 31-day-smoothed `mean_clim` — RMS **0.046°C**
    (6.3% of target's std, 0.7347°C), max **0.131°C** (18%). Real but
    modest — smaller than current models' MAE (~0.25-0.30°C), and affects
    `full_gnll_quantile`/`full_gnll_focal`/any future variant equally (not
    a differential effect between models already compared).
    **Fix (opt-in, no raw-source regeneration required)**:
    `src/utils/hobday.py::load_ns_mean_clim_smooth_delta()` (sister
    function to `load_ns_p90()`) returns `delta[doy] = mean_clim_unsmoothed
    - mean_clim_smoothed` (365,), derived on the fly from `mean_clim`
    already present in `sst_climatology_doy.nc`. New config flag
    `hobday_smooth_target` (default `False`) in
    `LazyDataset.__init__` applies `target += delta[doy]` before
    `compute_stats()` runs, so the corrected target's normalisation stats
    are self-consistent. `src.utils.hobday` is imported lazily inside
    `__init__` only when the flag is set — importing it at module level
    would make `dataset.py` (imported by nearly every script) require
    `MHW_CLIM_FILE` even for configs that don't use this flag
    (`src.utils.paths.CLIM_FILE` raises at import time if unset).
    **Not activated in any existing config** — no experiment's target
    changes without explicitly opting in.
    **Deliberately not fixed at the source**: regenerating
    `mean_clim.nc`/`to_anom` with the proper 31-day-smoothed mean from raw
    sources (`preprocess_all.py`, outside this repo) would have a larger
    blast radius than the kfold split fix (issue #30-class) — affects the
    target of every experiment in the project, not just kfold-mode ones.
    Parked as an explicit pending decision, same class as the
    `land_mask`/`land_mask_v2` situation in `docs/data.md`.

41. **Ground-truth "MHW day" definition — basin-mean-then-threshold is not
    the field-standard definition; quantified and a corrected alternative
    prototyped, Aug 20 2026, not yet adopted anywhere**: `target` is the
    NS-box spatial mean of `to_anom`, and the project's existing MHW-day
    classification (e.g. the "15.2%"/"19.0%"/"34.9%" extreme-day recall
    numbers from Aug 19-20 2026) applies Hobday to that single averaged
    series. Confirmed against the literature that this is **not** the
    standard approach: Hobday et al. 2016 defines an MHW "at a single
    location" (point-based), and standard regional aggregation is
    "bin-averaged... regional aggregation of point-based definitions" —
    i.e. classify per-pixel first, aggregate second, not the reverse.
    Basin-mean-first systematically smooths out real extremes: measured
    directly, our 2022 count (66 days) is far below the ~140-day regional
    average reported in Ocean Science 2025 for the North Sea (range 60
    Norwegian Trench to 200+ English Channel by sub-region) — our number
    sits near that range's low end, consistent with the smoothing
    explanation, not a data bug (verified: the year-level ranking of
    which years were extreme matches literature well, 9/10 confirmed —
    see the Aug 20 2026 narrative.md entry for the full literature
    cross-check).
    **Prototyped fix, not yet the project's standard**: per-pixel Hobday
    classification (`merged_daily.nc`'s `to_anom` at 0.5°, 457 ocean
    pixels in the NS box + `sst_climatology_doy.nc`'s `p90_thresh`
    regridded to match) + an area-fraction threshold for "regional MHW
    day", using only data already present on Raven — no new transfers.
    Exploratory run at 50% area (not calibrated, no literature basis for
    that specific number — confirmed no universal convention exists; the
    field either reports "daily MHW area" as a continuous quantity or
    calibrates a cutoff against the region's own historical distribution,
    e.g. "Blob-Class" events use area >400,000 km² because that is the
    top 20% of that region's own historical record, not a fixed
    percentage): 2022→74 days, 2014→115, 2007→105 — closer to literature
    than the current 66/134/110. "Any pixel" (near-0% threshold) gives
    365/365 days — useless, confirms a real threshold is required, not
    just switching to per-pixel.
    **Not resolved — this is a paper-framing decision, not a pure
    technical one**: if the paper's claim is "we forecast a North-Sea
    basin-mean anomaly index", the current definition is self-consistent
    and nothing needs to change. If the claim is closer to "the model
    detects/predicts marine heatwaves" (which the project's own name
    suggests), "MHW" should use the field-standard per-pixel definition,
    and the model's scalar prediction becomes a proxy/early-warning signal
    for that independently-defined event — which requires reporting how
    well the proxy and the real event agree, not assuming they're
    interchangeable. See the calibration/agreement work tracked against
    this issue before either number is cited as final in the paper.
    **Threshold calibrated Aug 20 2026** (full reasoning in the Aug 20 2026
    narrative.md entry): area-fraction cutoff set to **5% (MedECC 2023
    default)**, not the earlier exploratory 50% nor the same-day 40.5%
    (top10%/p90-of-own-distribution) choice — 5% reproduces the
    independently-sourced ~140 days/yr North Sea literature benchmark
    almost exactly (135.1 days/yr), while 40.5% landed in the NOAA
    Blob-tracker "extreme events only" tier (36.5 days/yr), undercounting
    like the original basin-mean definition it was meant to correct.
    Darmaraki et al. (2024)'s North-Atlantic-specific 10-20% kept as
    fallback if reviewers object to a Mediterranean-derived convention.
    `scripts/analysis/mhw_definition_agreement_and_recall.py`'s
    `AREA_FRAC_THRESHOLD = 0.05`.

42. **`split_mode: kfold`'s `val_years` collision (see #1's own detailed
    write-up above under "Hallazgo #1" during the Aug 20 2026 review) —
    fixed via a new `stratified_kfold` mode, `kfold` left unfixed on
    purpose, Aug 20 2026**: `src/data/datamodule.py` gained
    `elif self.split_mode == "stratified_kfold":` — rotating buckets
    (`val_years = buckets[(fold+1) % n_folds]`) built by round-robin
    assignment of years ranked by Hobday MHW-day count (descending, reusing
    `load_ns_p90()`/`apply_hobday()`, not reimplemented). Verified
    empirically: test MHW-days per fold went from `[99, 110, 299, 218,
    335]` (kfold) to `[264, 226, 212, 196, 163]` (stratified_kfold) — max/min
    ratio 3.4x → 1.6x. `val_years` differs by construction across every
    fold (unlike kfold's folds 1-4 sharing one set). `kfold` itself is
    deliberately left unfixed — fixing it in place would silently change
    every existing kfold-mode experiment's actual train/val/test
    composition without anyone asking for that, invalidating
    reproducibility of everything already cited in `narrative.md`.
    **`tests/test_splits.py` rewritten**: previously replicated kfold with
    the wrong RNG (`default_rng(0)` vs. production's `RandomState(seed)`)
    and never modeled `val_years` at all — see #1's meta-hallazgo. Now: (a)
    `_kfold_test_groups`/`_kfold_val_years` use the correct
    `RandomState(seed=42)`, matching production exactly; (b)
    `test_kfold_val_years_collision_KNOWN_BUG` turns the bug itself into an
    explicit characterization-test assertion, so it can't silently regress
    or silently get "fixed" without the test (and this doc) being updated
    together; (c) new `stratified_kfold` tests (val differs per fold, no
    overlap, full coverage, determinism) use a synthetic MHW-day ranking,
    not real Hobday output — the bucket-rotation algorithm being tested
    doesn't depend on where the ranking numbers came from, and this keeps
    the test suite fast and independent of `MHW_DATA_FILE`/`MHW_CLIM_FILE`
    being set. All 10 tests pass.
    **Same RNG bug, found duplicated in 2 more places, 1 fixed here,
    1 left as a known gap**: `scripts/persistence_remote_sst.py::get_test_years`
    had the identical `default_rng(0)` bug — its own module docstring
    claims to compute "the same kfold test splits as the masked
    experiment", which was never true. Fixed alongside this issue (now
    `RandomState(seed=42)`, matching production, `seed` parameterized
    instead of hardcoded). `scripts/ig_simple.py:84` has the exact same bug
    and was **not** fixed in this pass — it only affects which years'
    samples get used for IG/XAI plots, not any training or eval-metric
    result, and is lower priority than the training-pipeline-facing fixes.
    Flagged here so it isn't forgotten before `ig_simple.py` is next used
    against a `stratified_kfold`-trained checkpoint.

43. **SLURM job-completion emails never actually sent — found Aug 20 2026,
    verified empirically**: every `scripts/slurm/*.sh` submit script either
    (a) had `#SBATCH --mail-type=...` with no `--mail-user` directive at
    all, relying on a header comment instructing `export
    SBATCH_MAIL_USER=you@example.com` before `sbatch --export=ALL ...`, or
    (b) had `#SBATCH --mail-user=${SLURM_MAIL:-}` directly. Both are
    broken: `#SBATCH` lines are parsed literally by `sbatch`, not through a
    shell, so `${VAR}` is **never expanded** there. `SBATCH_MAIL_USER` is
    also not a real sbatch-recognised env override — only `SBATCH_ACCOUNT`
    (mapping to `--account`) is documented as such (`man sbatch`, INPUT
    ENVIRONMENT VARIABLES). Verified by submitting real (held, then
    cancelled before running) test jobs and inspecting `scontrol show job`:
    case (a) produces a job with the *default* mail-user (submitting user,
    which doesn't resolve to a working address on Raven); case (b)
    produces a job with `MailUser=${SLURM_MAIL:-}` — the literal
    unexpanded string. This means **no job submitted through this repo's
    scripts has ever emailed a completion/failure notification**,
    including every `full_gnll_quantile`/`full_gnll_focal` array job to
    date. Fixed in all `scripts/slurm/*.sh`: header comments now instruct
    passing `--mail-user=you@example.com` (and `--account=...`) directly on
    the `sbatch` command line, and the inert `#SBATCH --mail-user=${...}`
    lines were removed. `CLAUDE.md`'s SLURM-email rule updated to match.
    Going forward: `sbatch --account=mmm_gpu
    --mail-user=cristina.radin@uni-hamburg.de --export=ALL ...`.

44. **Cosine LR scheduler's `T_max` used the artificial `max_epochs` ceiling
    instead of the realistic early-stop horizon — found by multi-agent
    code review Aug 20 2026, before the `full_gnll_focal_v2` fold0 GPU
    launch, fixed same day**: `configure_optimizers()`'s cosine branch
    computed decay progress against `self.trainer.max_epochs` (`1000` in
    every partition config — an early-stopping ceiling, `EarlyStopping`
    patience=30 always stops training long before that). Verified
    directly (not just reasoned about): with `T_max=1000`, LR is still
    ~90% of peak by the time a run of ~150-200 epochs would realistically
    early-stop — the cosine schedule barely does anything, silently
    defeating the reason it was added over `ReduceLROnPlateau`. Fixed by
    adding an explicit `cosine_t_max_epochs` config key (`CNNLightningModule.
    __init__`, `configure_optimizers()`, threaded through
    `train_partition.py`), set to `60` in `configs/partition/
    full_gnll_focal_v2/fold{0-4}.yaml` (`fold0_shorttest.yaml` uses `3`,
    matching its own `max_epochs: 3`, plus `warmup_epochs: 1` instead of
    `5` so the 3-epoch smoke test actually reaches the cosine-decay
    branch instead of staying in warmup the whole time — also found by
    the same review). No silent fallback if neither `cosine_t_max_epochs`
    nor an attached `Trainer` is available — raises instead of guessing a
    number. `60` is a judgment call, not a proven-optimal value — pick
    based on the ~30-40 epoch early-stop range observed for the v1
    `full_gnll_quantile`/`full_gnll_focal` runs (narrative.md, Aug 20
    2026 compute-cost note), doubled for margin since v2 changes LR
    (5e-5 vs 1e-4) and target scale (`hobday_smooth_target`). Revisit
    once fold0's real epoch count is known.

45. **`configs/partition/` naming has sprawled across a session of rapid
    iteration (Aug 20 2026) — TODO cleanup next week, mapped here so the
    cleanup has a clear starting inventory instead of having to
    reverse-engineer it from git history.** As of Aug 20 2026 (updated
    same day: `experiments/partition/`'s v1 output directories — all 12
    of `TbotAtm_full_gnll_{focal,quantile}_seed42_fold{0-4,0_shorttest}`
    — moved into `experiments/partition/old_v1/` at the user's request,
    to stop confusing them with the new v2/v3 runs sitting at the top
    level. Updated in lockstep: the 12 corresponding `configs/partition/
    full_gnll_{focal,quantile}/fold*.yaml`'s `output_dir:`, plus
    `scripts/analysis/{_adhoc_eval_extreme_recall,
    mhw_definition_agreement_and_recall}.py`'s `EXPERIMENTS_DIR` constant
    — verified checkpoints are still reachable at the new path before
    calling this done. `configs/` themselves were NOT moved/renamed, only
    their `output_dir:` values edited, so the "historical record, don't
    rename configs" guidance from before still applies to the yaml files
    — it's specifically the big `experiments/` checkpoint directories that
    moved):
    - `full/`, `full_gnll/` — original JUWELS-path configs (`data_dir`/
      `output_dir` under `/p/project1/hai_1127/...`), plain MSE and plain
      GNLL respectively. Real completed results already cited in
      `narrative.md` (the "15.2%" GNLL baseline). **Do not delete or
      rename** — historical record.
    - `full_gnll_quantile/`, `full_gnll_focal/` — Raven-ported (Aug 19-20
      2026), `split_mode: kfold` (the buggy one, #1/#2), no
      `hobday_smooth_target`, `ReduceLROnPlateau`. Real completed 5-fold
      results, jobs 29404086/29404257. **Do not delete or rename** —
      historical record, and `full_gnll_quantile/fold0.yaml` is also
      still the one used by `scripts/analysis/
      mhw_definition_agreement_and_recall.py` to reload those exact
      checkpoints (loading a checkpoint requires the config that matches
      how it was trained, not a "cleaned up" one).
    - `full_gnll_focal_v2/` — fold0 GPU-launched Aug 20 2026 (job
      29417248), `stratified_kfold` + `hobday_smooth_target` + cosine
      LR/warmup + `reflect` padding. Fold0-only so far; full 5-fold array
      not yet launched (Paso 5 gate).
    - `full_gnll_quantile_v2/` — fold0 GPU-launched Aug 20 2026 (job
      29417405), same v2 treatment as focal_v2, added same day
      specifically to compare against it at the fold0 level before
      committing to any full array.
    - `full_mse_v2/` — **pre-existing, unrelated to this session's "v2"
      meaning**: an old JUWELS-path plain-MSE config (`merged_daily_v2.nc`
      — the "v2" refers to a dataset version, not the stratified_kfold/
      cosine/reflect treatment used everywhere else in this list). Never
      run on Raven. Kept untouched to avoid destroying whatever it was
      for, but the name collision with this session's "_v2 = new
      treatment" convention is exactly the kind of thing to fix during
      cleanup (rename to something like `full_mse_juwels_legacy/` or
      delete if confirmed unused).
    - `full_mse_v3/` — fold0 GPU-launched Aug 20 2026 (job 29417406), the
      *actual* stratified_kfold/hobday_smooth_target/cosine/reflect
      treatment, named "v3" only to dodge the collision above. Rename
      candidate once `full_mse_v2` is resolved.
    **Suggested cleanup direction (not decided, discuss together next
    week)**: adopt one consistent suffix meaning "stratified_kfold +
    hobday_smooth_target + reflect padding + cosine LR" across all three
    loss variants (e.g. `full_{focal,quantile,mse}_stratified/` or similar
    self-describing name instead of `_v2`/`_v3`), retire the old
    `kfold`-mode configs' directories only after confirming nothing still
    depends on reloading their checkpoints, and decide whether `full`/
    `full_gnll` (JUWELS-path, unrunnable on Raven as-is) should be ported
    or archived.

46. **`trainer.test()` used last-epoch weights, not the best checkpoint —
    found Aug 20 2026 while investigating why wandb val_loss curves looked
    "terrible", fixed same day, not yet re-run.** `train_partition.py`
    called `trainer.test(lightning_module, datamodule=datamodule)` with
    no `ckpt_path` — confirmed against the installed `pytorch_lightning`
    2.6.0 docstring: "If ckpt_path is None and the model instance was
    passed, use the current weights" (NOT the best checkpoint, despite
    `ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3)` being
    configured and actively saving a better one). So every
    `test_mae`/`test_corr`/`test_loss`/`test_nll_loss`/`test_pinball_loss`
    number reported by any `train_partition.py` run to date — including
    the fold0 `full_gnll_focal_v2`/`full_gnll_quantile_v2`/`full_mse_v3`
    comparison and the already-cited v1 `full_gnll_quantile`/
    `full_gnll_focal` 5-fold results — reflects whatever epoch
    `EarlyStopping` happened to stop at, not the model that was actually
    selected as best. Verified this matters, not just theoretically:
    `full_gnll_focal_v2`'s real per-epoch `val_loss` trajectory (grepped
    from the SLURM log, not assumed) hits a minimum of **0.034 around
    epoch 18-19**, then oscillates up to **1.35-1.7 by epoch 32-35**
    (where it stopped) — a ~50x swing, consistent with GaussianNLLLoss's
    known variance-collapse pathology (predicted variance shrinks on some
    validation batches, `log(var)` spikes). `full_mse_v3` (no variance
    term) oscillates too but far less (~3x, 0.30-0.86) — some of the
    swing is normal small-val-set noise, but the GNLL-specific portion is
    real and much larger. **Not affected**: the extreme-day recall
    numbers in `scripts/analysis/_adhoc_eval_extreme_recall.py` and
    `mhw_definition_agreement_and_recall.py` — those separately call
    `best_ckpt()` and `load_from_checkpoint()` explicitly, bypassing
    `trainer.test()`'s default entirely. Fixed: `ckpt_path = "best" if
    not args.fast_dev_run else None` (fast_dev_run disables checkpoint
    saving, so "best" doesn't exist there). **Not yet re-run** — the user
    wants to keep analyzing before relaunching anything (Aug 20 2026:
    "OBVIAMENTE PON EL MEJOR CHECKPOINT. PERO NO LANCES AUN"). Every
    MAE/r/loss number discussed before this fix should be treated as
    unreliable until re-run with the corrected checkpoint selection.

47. **XAI (`src/xai/integrated_gradients.py`, `grad_cam.py`) only explains
    the mean head, never the quantile head — real gap, flagged before it's
    needed, not yet fixed, Aug 21 2026.** Confirmed by reading both files:
    `_integrated_gradients_forward()` calls `model(interp, x_temporal)`
    (plain `forward()`, mean/log_var only) and backprops from that;
    `grad_cam.py:67` does the same. Neither calls `forward_with_quantile()`.
    This was already a known gap noted in-code (`cnn_lstm.py`'s
    `forward_with_attention()` docstring: "quantile_head + attention + XAI
    is unimplemented"). Matters now because `full_gnll_quantile_v2` is the
    committed model (Aug 20-21 2026, see narrative.md) and its actual value
    is in `q_pred` (44.7%/91.8% recall/precision under def2), not the mean
    (5.3% recall) — running IG/GradCAM as-is would explain "why does the
    model predict this temperature", not "why does the model flag this day
    as high MHW risk", which is the more interesting question for a
    precursor-detection paper. Fix is scoped, not a rewrite: since
    `_encode()` is already shared between both heads, adapt
    `_integrated_gradients_forward()` (and GradCAM's analogous function) to
    optionally call `forward_with_quantile()` and backprop from `q_pred`
    instead of `pred`. Also unverified: whether `pred.squeeze().backward()`
    in `_integrated_gradients_forward()` even works for `gaussian_nll=True`
    models today (`pred` would be shape `(batch,2)` — `.backward()` on a
    non-scalar tensor without a `gradient` arg normally errors) — check
    this before assuming the mean-head path itself is even currently
    exercised correctly for GNLL models, separately from the quantile gap.

## How to use
For each script or memory doc reviewed, check against all 47 items and
report: applies / does not apply / unclear — with the specific line as
evidence for each "applies".