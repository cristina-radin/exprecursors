"""
Splits produced by the REAL LazyDataModule (src/data/datamodule.py) on the
synthetic dataset from tests/synthetic.py -- no replicated split logic.

Invariants of split_mode="stratified_kfold" (known_issues.md #1/#42):
  * within a fold, train/val/test year sets are disjoint and cover every year;
  * test years are disjoint across folds and together cover every year;
  * val years differ from fold to fold (the bug kfold had);
  * the split is deterministic;
  * years are ranked by MHW days (ties by increasing year) and dealt in
    serpentine order;
  * the synthetic target gives years with different MHW-day counts, so the
    ranking is not just the tie-break.
"""

import pytest

from src.data.datamodule import (
    LazyDataModule,
    _mhw_days_per_year,
    _rank_years,
    _serpentine_buckets,
)
from src.data.dataset import LazyDataset
from tests import synthetic as syn

N_FOLDS = 5


def _setup(fold, **overrides):
    cfg = syn.write_config(fold=fold, **overrides)
    dm = LazyDataModule(config=cfg)
    dm.setup()
    return dm


def _years(dm, subset):
    ds = subset.dataset
    return {
        int(ds.years[i + ds.window_size - 1 + ds.lead_time]) for i in subset.indices
    }


def _fold_years(dm):
    return (
        _years(dm, dm.train_dataset),
        _years(dm, dm.val_dataset),
        _years(dm, dm.test_dataset),
    )


@pytest.fixture(scope="module")
def folds():
    return [_fold_years(_setup(f)) for f in range(N_FOLDS)]


def test_train_val_test_disjoint_within_fold(folds):
    for train, val, test in folds:
        assert not (train & val) and not (train & test) and not (val & test)


def test_every_year_used_within_fold(folds):
    all_years = set(syn.YEARS)
    for train, val, test in folds:
        assert train | val | test == all_years


def test_test_years_disjoint_across_folds_and_cover_all(folds):
    tests = [t for _, _, t in folds]
    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            assert not (tests[i] & tests[j])
    assert set().union(*tests) == set(syn.YEARS)


def test_val_years_differ_per_fold(folds):
    vals = [frozenset(v) for _, v, _ in folds]
    assert len(set(vals)) == N_FOLDS


def test_split_is_deterministic(folds):
    assert _fold_years(_setup(2)) == folds[2]


def test_data_dir_key_is_rejected():
    cfg = syn.write_config(data_dir=str(syn.DATA_FILE))
    with pytest.raises(ValueError, match="MHW_DATA_FILE"):
        LazyDataModule(config=cfg)


@pytest.mark.parametrize(
    "n_years, expected",
    [
        (10, [[0, 9], [1, 8], [2, 7], [3, 6], [4, 5]]),
        (12, [[0, 9, 10], [1, 8, 11], [2, 7], [3, 6], [4, 5]]),
        (15, [[0, 9, 10], [1, 8, 11], [2, 7, 12], [3, 6, 13], [4, 5, 14]]),
    ],
)
def test_serpentine_buckets(n_years, expected):
    assert _serpentine_buckets(list(range(n_years)), N_FOLDS) == expected


def test_rank_years_most_days_first_ties_by_increasing_year():
    days = {2003: 5, 2001: 5, 2002: 7, 2000: 0, 1999: 0}
    assert _rank_years(days) == [2002, 2001, 2003, 1999, 2000]


def test_synthetic_years_have_different_mhw_day_counts():
    ds = LazyDataset(str(syn.DATA_FILE), config=syn.write_config())
    counts = _mhw_days_per_year(ds, list(syn.YEARS))
    assert len(set(counts.values())) >= len(syn.YEARS) // 2
