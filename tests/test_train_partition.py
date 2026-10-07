"""
_require_clean_output_dir (scripts/train_partition.py): refuses to start a
run into an output_dir that still has checkpoints from an earlier run
(known_issues.md #58) -- best_ckpt() would otherwise silently pick among
checkpoints of more than one run. No exemption: every run, including a
--limit_batches smoke run, saves checkpoints.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train_partition import _require_clean_output_dir


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
