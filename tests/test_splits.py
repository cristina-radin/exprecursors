"""
Test that kfold and block_year splits produce disjoint year sets.

Replicates the split logic from datamodule.py directly so no data loading is
needed — the invariant is on the year assignment math, not the full dataset.
"""

import numpy as np

N_FOLDS = 5


def _kfold_fold_groups(unique_years, n_folds=N_FOLDS):
    """Replicate the fold assignment from datamodule.py (seed=0, always)."""
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(unique_years)
    return [set(g.tolist()) for g in np.array_split(shuffled, n_folds)]


def _block_year_split(unique_years, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Replicate block_year split from datamodule.py."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_years)
    n = len(unique_years)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    return (
        set(shuffled[:n_train].tolist()),
        set(shuffled[n_train : n_train + n_val].tolist()),
        set(shuffled[n_train + n_val :].tolist()),
    )


UNIQUE_YEARS = np.arange(1985, 2025)  # 40 years, matches dataset


def test_kfold_test_sets_disjoint():
    """No year should appear in more than one fold's test set."""
    groups = _kfold_fold_groups(UNIQUE_YEARS)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = groups[i] & groups[j]
            assert not overlap, f"Folds {i} and {j} share test years: {overlap}"


def test_kfold_covers_all_years():
    """Union of all fold test sets must equal the full year range."""
    groups = _kfold_fold_groups(UNIQUE_YEARS)
    covered = set().union(*groups)
    assert covered == set(UNIQUE_YEARS.tolist())


def test_kfold_deterministic():
    """Same fold assignments must be produced on every call (seed=0)."""
    g1 = _kfold_fold_groups(UNIQUE_YEARS)
    g2 = _kfold_fold_groups(UNIQUE_YEARS)
    for a, b in zip(g1, g2):
        assert a == b


def test_block_year_sets_disjoint():
    """Train, val, test year sets must be mutually disjoint."""
    train, val, test = _block_year_split(UNIQUE_YEARS)
    assert not (train & val), f"train ∩ val = {train & val}"
    assert not (train & test), f"train ∩ test = {train & test}"
    assert not (val & test), f"val ∩ test = {val & test}"


def test_block_year_covers_all_years():
    train, val, test = _block_year_split(UNIQUE_YEARS)
    assert train | val | test == set(UNIQUE_YEARS.tolist())
