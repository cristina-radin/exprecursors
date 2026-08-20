"""
Test that kfold, stratified_kfold, and block_year splits produce disjoint
year sets — and lock in the known kfold val_years collision bug
(known_issues.md #1) as an explicit, documented characterization test so it
can never regress silently if kfold's logic is ever touched.

block_year and kfold's core (year-shuffling) logic is replicated here with
synthetic year arrays so no data loading is needed — the invariant is on
the year-assignment math, not the full dataset. stratified_kfold's bucket
rotation is likewise tested with a synthetic (not real Hobday-derived)
MHW-day ranking, since the algorithm itself doesn't care where the ranking
came from — real-data validation of the actual MHW-day counts happens via
the mandatory per-fold year/day prints in datamodule.py itself when a real
config is run (see the plan's "criterio de fin" for this step), not here.

IMPORTANT: the RNG type must match production EXACTLY, not just be *a*
valid RNG — np.random.RandomState(seed) and np.random.default_rng(seed)
produce different permutations from the same seed (known_issues.md #23).
Prior to Aug 20 2026, _kfold_fold_groups here used default_rng(0), which
does NOT match src/data/datamodule.py's actual kfold branch
(np.random.RandomState(self.seed), seed from the config, not hardcoded 0)
-- meaning this test suite tested a fictional split, never the real one,
and could not have caught the val_years collision bug it happened to have.
"""

import numpy as np

N_FOLDS = 5
UNIQUE_YEARS = np.arange(1985, 2025)  # 40 years, matches dataset


# ---------------------------------------------------------------------------
# kfold (legacy, known-buggy val_years — see known_issues.md #1)
# ---------------------------------------------------------------------------


def _kfold_test_groups(unique_years, n_folds=N_FOLDS, seed=42):
    """Replicate ONLY test_years from datamodule.py's kfold branch, with the
    correct production RNG (RandomState, not default_rng). test_years'
    disjointness/coverage/determinism hold regardless of which RNG produced
    the permutation (any permutation split into contiguous, non-overlapping
    chunks has these properties) -- what matters here is exercising the
    REAL algorithm, not a stand-in for it.
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_years)
    n = len(unique_years)
    fold_size = n // n_folds
    return [
        set(shuffled[fold * fold_size : (fold + 1) * fold_size].tolist())
        for fold in range(n_folds)
    ]


def _kfold_val_years(unique_years, fold, n_folds=N_FOLDS, seed=42, val_ratio=0.15):
    """Replicate datamodule.py's kfold val_years EXACTLY, bug included."""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_years)
    n = len(unique_years)
    fold_size = n // n_folds
    test_years = set(shuffled[fold * fold_size : (fold + 1) * fold_size].tolist())
    remaining_years = [y for y in shuffled if y not in test_years]
    n_val = int(val_ratio * n)
    return set(remaining_years[:n_val])


def test_kfold_test_sets_disjoint():
    """No year should appear in more than one fold's test set."""
    groups = _kfold_test_groups(UNIQUE_YEARS)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = groups[i] & groups[j]
            assert not overlap, f"Folds {i} and {j} share test years: {overlap}"


def test_kfold_covers_all_years():
    """Union of all fold test sets must equal the full year range."""
    groups = _kfold_test_groups(UNIQUE_YEARS)
    covered = set().union(*groups)
    assert covered == set(UNIQUE_YEARS.tolist())


def test_kfold_deterministic():
    """Same fold assignments must be produced on every call (fixed seed)."""
    g1 = _kfold_test_groups(UNIQUE_YEARS)
    g2 = _kfold_test_groups(UNIQUE_YEARS)
    for a, b in zip(g1, g2):
        assert a == b


def test_kfold_val_years_collision_KNOWN_BUG():
    """CHARACTERIZATION TEST, not a correctness assertion: documents that
    kfold's val_years is IDENTICAL across folds 1..N_FOLDS-1 (only fold 0
    differs) -- known_issues.md #1, root cause: val_years always takes the
    first n_val years of "remaining" (years minus that fold's test block),
    and for every fold whose test block doesn't start at index 0, the
    leading years of "remaining" don't change.

    This test exists so nobody "fixes" kfold in place without noticing —
    if this assertion ever starts failing, kfold's val-year logic changed
    and this test (and known_issues.md #1) need updating together. The
    actual fix lives in stratified_kfold, not here — kfold is kept
    deliberately unfixed so existing kfold-mode experiments in
    narrative.md stay reproducible.
    """
    val_sets = [_kfold_val_years(UNIQUE_YEARS, fold) for fold in range(N_FOLDS)]
    assert val_sets[1] == val_sets[2] == val_sets[3] == val_sets[4], (
        "kfold's val_years collision bug (known_issues.md #1) no longer "
        "reproduces -- if kfold's logic changed, update this test AND "
        "known_issues.md #1 together, don't just delete this assertion."
    )
    assert val_sets[0] != val_sets[1], "fold 0 should be the one fold that differs"


# ---------------------------------------------------------------------------
# stratified_kfold (fixed k-fold, known_issues.md #1/#2 fix)
# ---------------------------------------------------------------------------


def _stratified_kfold_buckets(unique_years, ranking_key, n_folds=N_FOLDS):
    """Replicate datamodule.py's stratified_kfold bucket assignment exactly.

    ranking_key: callable, year -> sortable value (real code ranks by
    descending Hobday MHW-day count; here a synthetic key is passed in so
    this test needs no real data -- the bucket ROTATION logic being tested
    doesn't care where the ranking numbers came from).
    """
    ranked_years = sorted(unique_years, key=lambda y: -ranking_key(y))
    buckets = [[] for _ in range(n_folds)]
    for i, y in enumerate(ranked_years):
        buckets[i % n_folds].append(int(y))
    return buckets


def _synthetic_mhw_ranking(y):
    """Deterministic stand-in for MHW-day count, same role as the real
    Hobday-derived dict in datamodule.py — arbitrary but fixed per year."""
    return (int(y) * 2654435761) % 1000  # cheap hash, spreads years around


def test_stratified_kfold_val_years_differ_per_fold():
    """The whole point of the fix: unlike kfold, every fold's val_years
    must be distinct from every other fold's."""
    buckets = _stratified_kfold_buckets(UNIQUE_YEARS, _synthetic_mhw_ranking)
    val_sets = [set(buckets[(fold + 1) % N_FOLDS]) for fold in range(N_FOLDS)]
    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            assert val_sets[i] != val_sets[j], (
                f"stratified_kfold fold {i} and {j} have the same val_years "
                f"{val_sets[i]} -- this is exactly the bug it was built to fix"
            )


def test_stratified_kfold_no_overlap_within_fold():
    """train/val/test must be mutually disjoint within each fold."""
    buckets = _stratified_kfold_buckets(UNIQUE_YEARS, _synthetic_mhw_ranking)
    all_years = set(UNIQUE_YEARS.tolist())
    for fold in range(N_FOLDS):
        test_years = set(buckets[fold])
        val_years = set(buckets[(fold + 1) % N_FOLDS])
        train_years = all_years - test_years - val_years
        assert not (train_years & val_years)
        assert not (train_years & test_years)
        assert not (val_years & test_years)
        assert train_years | val_years | test_years == all_years


def test_stratified_kfold_covers_all_years():
    """Union of all fold test sets (the buckets themselves) must equal the
    full year range, and no bucket should be empty for 40 years / 5 folds."""
    buckets = _stratified_kfold_buckets(UNIQUE_YEARS, _synthetic_mhw_ranking)
    covered = set()
    for b in buckets:
        covered |= set(b)
    assert covered == set(UNIQUE_YEARS.tolist())
    assert all(len(b) == len(UNIQUE_YEARS) // N_FOLDS for b in buckets)


def test_stratified_kfold_deterministic():
    """Same bucket assignment on every call, for a fixed ranking."""
    b1 = _stratified_kfold_buckets(UNIQUE_YEARS, _synthetic_mhw_ranking)
    b2 = _stratified_kfold_buckets(UNIQUE_YEARS, _synthetic_mhw_ranking)
    assert b1 == b2


# ---------------------------------------------------------------------------
# block_year (unchanged, already correctly replicated production RNG)
# ---------------------------------------------------------------------------


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


def test_block_year_sets_disjoint():
    """Train, val, test year sets must be mutually disjoint."""
    train, val, test = _block_year_split(UNIQUE_YEARS)
    assert not (train & val), f"train ∩ val = {train & val}"
    assert not (train & test), f"train ∩ test = {train & test}"
    assert not (val & test), f"val ∩ test = {val & test}"


def test_block_year_covers_all_years():
    train, val, test = _block_year_split(UNIQUE_YEARS)
    assert train | val | test == set(UNIQUE_YEARS.tolist())
