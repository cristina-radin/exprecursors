"""
save_model_config / load_model_config (src/utils/checkpoints.py): the keys
they read and write must always match CNNLSTMModel.__init__ exactly, for
every loss variant -- this is what broke when arch/temporal_features/
state_feature were removed from the constructor but not from
CNNLSTM_DEFAULTS (docs/open_issues.md).
"""

import json

import pytest

from src.models.cnn_lstm import CNNLSTMModel
from src.utils.checkpoints import best_ckpt, load_model_config, save_model_config

VARIANTS = [
    dict(gaussian_nll=True, quantile_head=True),
    dict(gaussian_nll=True, quantile_head=False),
    dict(gaussian_nll=False, quantile_head=False),
]


def _model_kwargs(gaussian_nll, quantile_head):
    return dict(
        in_channels=3,
        cnn_features=8,
        lstm_hidden=16,
        lstm_layers=1,
        dropout=0.0,
        gaussian_nll=gaussian_nll,
        pooling="avg",
        padding_mode="zeros",
        quantile_head=quantile_head,
        mode="full",
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_save_then_load_round_trips_and_builds_model(tmp_path, variant):
    kwargs = _model_kwargs(**variant)

    save_model_config(tmp_path, **kwargs)
    loaded = load_model_config(tmp_path)

    assert loaded == kwargs
    CNNLSTMModel(**loaded)  # must build without error


def test_save_model_config_missing_key_raises(tmp_path):
    kwargs = _model_kwargs(gaussian_nll=True, quantile_head=True)
    del kwargs["dropout"]
    with pytest.raises(ValueError, match="missing required keys"):
        save_model_config(tmp_path, **kwargs)


def test_save_model_config_extra_key_raises(tmp_path):
    kwargs = _model_kwargs(gaussian_nll=True, quantile_head=True)
    kwargs["arch"] = "lstm_only"
    with pytest.raises(ValueError, match="unexpected keys"):
        save_model_config(tmp_path, **kwargs)


def test_load_model_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_config(tmp_path)


def test_load_model_config_missing_key_in_json_raises(tmp_path):
    kwargs = _model_kwargs(gaussian_nll=True, quantile_head=True)
    save_model_config(tmp_path, **kwargs)
    (tmp_path / "model_config.json").write_text(
        (tmp_path / "model_config.json").read_text().replace('"dropout": 0.0,', "")
    )
    with pytest.raises(ValueError, match="missing required keys"):
        load_model_config(tmp_path)


def test_load_model_config_extra_key_in_json_raises(tmp_path):
    kwargs = _model_kwargs(gaussian_nll=True, quantile_head=True)
    save_model_config(tmp_path, **kwargs)
    path = tmp_path / "model_config.json"
    saved = json.loads(path.read_text())
    saved["arch"] = "lstm_only"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="unexpected keys"):
        load_model_config(tmp_path)


# ── best_ckpt() ───────────────────────────────────────────────────────────────


def _touch(tmp_path, name):
    (tmp_path / name).touch()


def test_best_ckpt_picks_lowest_val_loss(tmp_path):
    _touch(tmp_path, "cnn-lstm-epoch=05-val_loss=0.5000.ckpt")
    _touch(tmp_path, "cnn-lstm-epoch=10-val_loss=0.1399.ckpt")
    _touch(tmp_path, "cnn-lstm-epoch=15-val_loss=0.9000.ckpt")
    assert best_ckpt(tmp_path).name == "cnn-lstm-epoch=10-val_loss=0.1399.ckpt"


def test_best_ckpt_handles_negative_val_loss(tmp_path):
    _touch(tmp_path, "cnn-lstm-epoch=05-val_loss=-2.5000.ckpt")
    _touch(tmp_path, "cnn-lstm-epoch=10-val_loss=0.1399.ckpt")
    assert best_ckpt(tmp_path).name == "cnn-lstm-epoch=05-val_loss=-2.5000.ckpt"


def test_best_ckpt_parses_val_loss_with_decimal_point_correctly(tmp_path):
    # known_issues.md #3: the regex must extract exactly "0.1399", not stop
    # early at the decimal point or swallow it into the trailing ".ckpt".
    _touch(tmp_path, "cnn-lstm-epoch=05-val_loss=0.1399.ckpt")
    ckpt = best_ckpt(tmp_path)
    assert ckpt.name == "cnn-lstm-epoch=05-val_loss=0.1399.ckpt"


def test_best_ckpt_raises_on_non_matching_filename(tmp_path):
    # known_issues.md #35: a filename the regex can't parse must raise, not
    # silently rank as float("inf") / worst.
    _touch(tmp_path, "some-checkpoint-without-val-loss-in-the-name.ckpt")
    with pytest.raises(ValueError, match="Could not parse val_loss"):
        best_ckpt(tmp_path)


def test_best_ckpt_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No checkpoints"):
        best_ckpt(tmp_path)


def test_best_ckpt_raises_on_duplicate_suffix(tmp_path):
    # A "-vN.ckpt" file means ckpt_dir has checkpoints from more than one
    # training run (known_issues.md #58) -- best_ckpt() must refuse to pick
    # among them instead of silently skipping the duplicate.
    _touch(tmp_path, "cnn-lstm-epoch=05-val_loss=0.5000.ckpt")
    _touch(tmp_path, "cnn-lstm-epoch=05-val_loss=0.5000-v1.ckpt")
    with pytest.raises(ValueError, match="more than one training"):
        best_ckpt(tmp_path)
