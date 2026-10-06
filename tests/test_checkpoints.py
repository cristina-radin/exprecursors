"""
save_model_config / load_model_config (src/utils/checkpoints.py): the keys
they read and write must always match CNNLSTMModel.__init__ exactly, for
every loss variant -- this is what broke when arch/temporal_features/
state_feature were removed from the constructor but not from
CNNLSTM_DEFAULTS (docs/open_issues.md).
"""

import pytest

from src.models.cnn_lstm import CNNLSTMModel
from src.utils.checkpoints import load_model_config, save_model_config

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
