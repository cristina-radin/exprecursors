"""
Smoke test for the state_feature mechanism (Aug 23 2026) -- gives the
CNN-LSTM explicit access to the target's own recent value (same quantity
lag-persistence uses), concatenated directly to the LSTM context, NOT
mean-pooled into temporal_features like the calendar features are (see
src/models/cnn_lstm.py's _encode() docstring for why averaging would
destroy the "value right now" information).

Synthetic tensors only, tiny random model, CPU, no real data/checkpoint.
Checks: (1) state_feature=True requires x_state (no silent fallback);
(2) shapes correct with/without state_feature, both heads; (3) gradient
actually reaches the FC/quantile_head weights connected to x_state
(proves it is used, not silently dropped); (4) CNNLightningModule
enforces use_state_feature matches model.state_feature at construction.

Run with: pytest tests/test_state_feature.py
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel


def _tiny_model(state_feature, quantile_head=True, gaussian_nll=True):
    torch.manual_seed(0)
    return CNNLSTMModel(
        in_channels=3,
        cnn_features=8,
        lstm_hidden=8,
        lstm_layers=1,
        temporal_features=3,
        dropout=0.0,
        arch="lstm_only",
        gaussian_nll=gaussian_nll,
        pooling="avg",
        quantile_head=quantile_head,
        padding_mode="zeros",
        state_feature=state_feature,
    )


def _synthetic_batch(batch=2, window=5, n_vars=3, h=16, w=16):
    xs = torch.randn(batch, window, n_vars, h, w, requires_grad=False)
    xt = torch.randn(batch, window, 3)
    x_state = torch.randn(batch, 1)
    return xs, xt, x_state


def test_missing_x_state_raises():
    model = _tiny_model(state_feature=True)
    xs, xt, _ = _synthetic_batch()
    with pytest.raises(ValueError, match="state_feature=True but x_state is None"):
        model.forward(xs, xt)  # no x_state passed


def test_forward_shapes_with_state_feature():
    model = _tiny_model(state_feature=True)
    xs, xt, x_state = _synthetic_batch()
    y_hat = model.forward(xs, xt, x_state)
    assert y_hat.shape == (2, 2)  # gaussian_nll -> [mean, log_var]

    y_hat2, q_pred = model.forward_with_quantile(xs, xt, x_state)
    assert y_hat2.shape == (2, 2)
    assert q_pred.shape == (2, 1)


def test_forward_shapes_without_state_feature_unchanged():
    """state_feature=False must reproduce the exact pre-existing behavior
    -- no x_state needed, no shape change."""
    model = _tiny_model(state_feature=False)
    xs, xt, _ = _synthetic_batch()
    y_hat = model.forward(xs, xt)
    assert y_hat.shape == (2, 2)


def test_gradient_reaches_state_input():
    """Proves x_state is actually used by the FC head, not silently
    dropped -- the first Linear layer's weight column(s) corresponding
    to the state feature must receive a nonzero gradient."""
    model = _tiny_model(state_feature=True, quantile_head=False)
    xs, xt, x_state = _synthetic_batch()
    x_state.requires_grad_(True)
    y_hat = model.forward(xs, xt, x_state)
    y_hat.sum().backward()
    assert x_state.grad is not None
    assert x_state.grad.abs().sum().item() > 0


def test_lightning_module_enforces_flag_match():
    model_with_state = _tiny_model(state_feature=True)
    with pytest.raises(
        ValueError, match="use_state_feature=False but model.state_feature=True"
    ):
        CNNLightningModule(
            model=model_with_state,
            loss_fn="GaussianNLLLoss",
            gaussian_nll=True,
            quantile_head=True,
            quantile_tau=0.9,
            use_state_feature=False,  # mismatch on purpose
        )

    # matching flags should construct without error
    CNNLightningModule(
        model=model_with_state,
        loss_fn="GaussianNLLLoss",
        gaussian_nll=True,
        quantile_head=True,
        quantile_tau=0.9,
        use_state_feature=True,
    )


def test_lightning_step_unpacks_4tuple_with_state_feature():
    model = _tiny_model(state_feature=True)
    lm = CNNLightningModule(
        model=model,
        loss_fn="GaussianNLLLoss",
        gaussian_nll=True,
        quantile_head=True,
        quantile_tau=0.9,
        use_state_feature=True,
    )
    xs, xt, x_state = _synthetic_batch()
    y = torch.randn(2, 1)
    loss, pred, y_out = lm._step((xs, xt, y, x_state), "train")
    assert torch.isfinite(loss)
    assert pred.shape == (2, 1)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["pytest", __file__, "-v"])
