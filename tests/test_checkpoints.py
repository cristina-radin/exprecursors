"""
Test that different folds produce distinct model predictions.

Catches the bug where GNLL folds 0/1/2 returned identical r=0.886 because
checkpoint loading silently fell back to the same file.

Marked @pytest.mark.slow — requires actual kfold checkpoints on disk.
Run with: pytest tests/test_checkpoints.py -m slow
Skip in fast CI with: pytest -m "not slow"
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.paths import DATA_FILE as DATA_FILE_ENV
from src.utils.paths import EXPERIMENTS_DIR

KFOLD_BASE = EXPERIMENTS_DIR / "kfold"
EXPERIMENT = "TbotAtm_lstmonly"
DATA_FILE = str(DATA_FILE_ENV)


def _best_ckpt(ckpt_dir: Path) -> Path:
    ckpts = [
        c for c in ckpt_dir.glob("*.ckpt") if not re.search(r"-v\d+\.ckpt$", str(c))
    ]
    if not ckpts:
        return None

    def val_loss(c):
        m = re.search(r"val_loss=([-\d.]+?)\.ckpt", c.name)
        return float(m.group(1).rstrip(".")) if m else float("inf")

    return min(ckpts, key=val_loss)


def _load_fold_pred(
    fold: int, dummy_xs: torch.Tensor, dummy_xt: torch.Tensor
) -> np.ndarray:
    from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel

    fold_dir = KFOLD_BASE / f"{EXPERIMENT}_fold{fold}"
    ckpt_dir = fold_dir / "checkpoints"
    cfg_path = fold_dir / "config.yaml"

    if not (ckpt_dir.exists() and cfg_path.exists()):
        return None

    ckpt = _best_ckpt(ckpt_dir)
    if ckpt is None:
        return None

    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    config["data_dir"] = DATA_FILE  # override stale path

    model = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 256),
        lstm_hidden=config.get("lstm_hidden", 512),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=config.get("temporal_features", 0),
        dropout=0.0,
        arch=config.get("arch", "lstm_only"),
        gaussian_nll=config.get("gaussian_nll", False),
        pooling=config.get("pooling", "max"),
        padding_mode=config.get("padding_mode", "zeros"),
        quantile_head=config.get("quantile_head", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        ckpt, model=model, map_location="cpu", strict=True
    )
    lm.eval()
    with torch.no_grad():
        pred = lm.model(dummy_xs, dummy_xt)
    return pred.numpy().flatten()


@pytest.mark.slow
def test_kfold_predictions_not_identical():
    """
    Load each fold's best checkpoint, run a fixed dummy input, assert outputs differ.

    If any two folds return bit-identical predictions, the checkpoint loading
    mechanism is broken (same file loaded for multiple folds).
    """
    cfg0 = KFOLD_BASE / f"{EXPERIMENT}_fold0" / "config.yaml"
    if not cfg0.exists():
        pytest.skip(f"Checkpoint dir not found: {KFOLD_BASE / (EXPERIMENT + '_fold0')}")

    with open(cfg0) as f:
        config = yaml.safe_load(f)

    n_vars = config["in_channels"]
    window = config.get("window_size", 60)
    t_feats = config.get("temporal_features", 0)

    torch.manual_seed(0)
    # Model expects (batch, window, n_vars, lat, lon) and (batch, window, t_feats)
    dummy_xs = torch.randn(1, window, n_vars, 20, 20)
    dummy_xt = (
        torch.zeros(1, window, t_feats) if t_feats > 0 else torch.zeros(1, window, 0)
    )

    fold_preds = {}
    for fold in range(5):
        pred = _load_fold_pred(fold, dummy_xs, dummy_xt)
        if pred is not None:
            fold_preds[fold] = pred

    if len(fold_preds) < 2:
        pytest.skip("Need at least 2 fold checkpoints to compare")

    folds = sorted(fold_preds)
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            fi, fj = folds[i], folds[j]
            assert not np.array_equal(fold_preds[fi], fold_preds[fj]), (
                f"Fold {fi} and fold {fj} produce bit-identical predictions — "
                "checkpoint loading bug (same file loaded twice?)"
            )


@pytest.mark.slow
def test_kfold_val_losses_not_all_identical():
    """
    Cheap proxy: if every fold has the same val_loss in its checkpoint filename,
    the same checkpoint was likely used for all folds.
    """
    losses = {}
    for fold in range(5):
        ckpt_dir = KFOLD_BASE / f"{EXPERIMENT}_fold{fold}" / "checkpoints"
        if not ckpt_dir.exists():
            continue
        ckpt = _best_ckpt(ckpt_dir)
        if ckpt is None:
            continue
        m = re.search(r"val_loss=([-\d.]+?)\.ckpt", ckpt.name)
        if m:
            losses[fold] = float(m.group(1).rstrip("."))

    if len(losses) < 2:
        pytest.skip("Need at least 2 fold checkpoints")

    values = list(losses.values())
    assert len(set(values)) > 1, (
        f"All folds have identical val_loss={values[0]} — "
        "likely the same checkpoint loaded for every fold"
    )
