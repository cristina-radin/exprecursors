"""
Splits produced by the REAL LazyDataModule (src/data/datamodule.py) on the
synthetic dataset from tests/synthetic.py -- no replicated split logic.

Invariants of split_mode="stratified_kfold" (known_issues.md #1/#42):
  * within a fold, train/val/test year sets are disjoint and cover every year;
  * test years are disjoint across folds and together cover every year;
  * val years differ from fold to fold (the bug kfold had);
  * the split is deterministic.
"""

import pytest

from src.data.datamodule import LazyDataModule
from tests import synthetic as syn

N_FOLDS = 5


def _setup(tmp_path, fold, **overrides):
    cfg = syn.write_config(tmp_path, f"fold{fold}.yaml", fold=fold, **overrides)
    dm = LazyDataModule(config_path=cfg)
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
def folds(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("splits")
    return [_fold_years(_setup(tmp, f)) for f in range(N_FOLDS)]


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


def test_split_is_deterministic(tmp_path, folds):
    assert _fold_years(_setup(tmp_path, 2)) == folds[2]
