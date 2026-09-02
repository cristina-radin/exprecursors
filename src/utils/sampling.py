"""
Shared test-sample selection for XAI/occlusion "population" runs.

Bug found Aug 21 2026 (user's methodological review): every IG/occlusion
script capped its sample with `test_indices[:max_samples]` -- since
stratified_kfold's test_indices are built by iterating time in ascending
order and filtering to the fold's (non-consecutive) test years, this
takes the samples in pure chronological order, not a representative draw.
Verified directly for fold0: 299 of the first 300 samples fall in 1985
alone (1 in 1991), out of 8 test years spanning 1985-2018. Every
"population" IG/occlusion map produced before this fix represents one
early year, not the full test-year distribution.
"""

import numpy as np


def stratified_test_sample(test_indices, full_ds, max_samples, seed=42):
    """Return up to max_samples indices from test_indices, drawn
    proportionally by target year (not the first N in time order), so a
    population-level XAI/occlusion run represents the whole test-year
    span. Reproducible via `seed`.
    """
    rng = np.random.default_rng(seed)
    target_years = np.array(
        [
            full_ds.years[i + full_ds.window_size - 1 + full_ds.lead_time]
            for i in test_indices
        ]
    )
    n_total = len(test_indices)
    if max_samples >= n_total:
        return list(test_indices)

    unique_years = np.unique(target_years)
    selected_positions = []
    for yr in unique_years:
        yr_positions = np.where(target_years == yr)[0]
        n_yr = max(1, round(max_samples * len(yr_positions) / n_total))
        n_yr = min(n_yr, len(yr_positions))
        chosen = rng.choice(yr_positions, size=n_yr, replace=False)
        selected_positions.extend(chosen.tolist())

    selected_positions = sorted(selected_positions)[:max_samples]
    return [test_indices[i] for i in selected_positions]
