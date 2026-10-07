"""
scripts/train_partition.py's small, directly-testable pieces:

_require_clean_output_dir: refuses to start a run into an output_dir that
still has checkpoints from an earlier run (known_issues.md #58) --
best_ckpt() would otherwise silently pick among checkpoints of more than
one run. No exemption: every run, including a --limit_batches smoke run,
saves checkpoints.

_resolve_fold: fold comes from --fold only (never the yaml), and must be a
valid fold for n_folds.

_apply_limit_batches_suffix: a --limit_batches run's run_name gets a
"_limit{N}" suffix, so it can never collide with a real run's output_dir.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train_partition import (
    _apply_limit_batches_suffix,
    _require_clean_output_dir,
    _resolve_fold,
)


def test_raises_when_checkpoints_exist(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "cnn-lstm-epoch=05-val_loss=0.5000.ckpt").touch()
    with pytest.raises(RuntimeError, match="clean output_dir"):
        _require_clean_output_dir(tmp_path)


def test_passes_when_no_checkpoints_dir(tmp_path):
    _require_clean_output_dir(tmp_path)


def test_passes_when_checkpoints_dir_empty(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    _require_clean_output_dir(tmp_path)


# ── _resolve_fold ─────────────────────────────────────────────────────────────


def _cfg(**overrides):
    cfg = dict(n_folds=5, run_name="gnll_quantile")
    cfg.update(overrides)
    return cfg


def test_resolve_fold_injects_fold_and_suffixes_run_name():
    cfg = _cfg()
    _resolve_fold(cfg, 2)
    assert cfg["fold"] == 2
    assert cfg["run_name"] == "gnll_quantile_fold2"


def test_resolve_fold_raises_if_yaml_already_has_fold():
    cfg = _cfg(fold=0)
    with pytest.raises(ValueError, match="comes from --fold"):
        _resolve_fold(cfg, 0)


@pytest.mark.parametrize("fold", [-1, 5, 100])
def test_resolve_fold_raises_if_out_of_range(fold):
    cfg = _cfg(n_folds=5)
    with pytest.raises(ValueError, match="0 <= fold < n_folds"):
        _resolve_fold(cfg, fold)


def test_resolve_fold_accepts_every_valid_fold():
    for fold in range(5):
        cfg = _cfg()
        _resolve_fold(cfg, fold)
        assert cfg["fold"] == fold


# ── _apply_limit_batches_suffix ───────────────────────────────────────────────


def test_limit_batches_suffix_applied_when_positive():
    cfg = _cfg(run_name="gnll_quantile_fold0")
    _apply_limit_batches_suffix(cfg, 5)
    assert cfg["run_name"] == "gnll_quantile_fold0_limit5"


def test_limit_batches_suffix_not_applied_when_zero():
    cfg = _cfg(run_name="gnll_quantile_fold0")
    _apply_limit_batches_suffix(cfg, 0)
    assert cfg["run_name"] == "gnll_quantile_fold0"
