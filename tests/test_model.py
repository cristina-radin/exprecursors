"""
CNNLSTMModel / CNNLightningModule (src/models/cnn_lstm.py) on tiny synthetic
tensors -- no real data, CPU, seconds. Covers the three loss variants, the
gradient isolation between the two heads, the lr schedule, and the invalid
constructor combinations that raise.
"""

import math

import pytest
import torch

from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel

# LAT/LON must survive 3 halvings (CNNEncoder's 3 Pool2d(2) layers) without
# collapsing to 0 before the final AdaptiveAvgPool2d((1, 1)).
BATCH, WINDOW, N_VARS, LAT, LON = 2, 4, 3, 16, 16

# (gaussian_nll, quantile_head, loss_fn)
VARIANTS = [
    ("gnll_quantile", True, True, "GaussianNLLLoss"),
    ("gnll_only", True, False, "GaussianNLLLoss"),
    ("mse", False, False, "MSELoss"),
]


def _build(gaussian_nll, quantile_head, loss_fn, mode="full", **module_overrides):
    model = CNNLSTMModel(
        in_channels=N_VARS,
        cnn_features=8,
        lstm_hidden=16,
        lstm_layers=1,
        dropout=0.0,
        gaussian_nll=gaussian_nll,
        pooling="avg",
        padding_mode="zeros",
        quantile_head=quantile_head,
        mode=mode,
    )
    kwargs = dict(
        model=model,
        learning_rate=1e-3,
        weight_decay=1e-4,
        target_mean=0.0,
        target_std=1.0,
        loss_fn=loss_fn,
        gaussian_nll=gaussian_nll,
        quantile_head=quantile_head,
        quantile_tau=0.9,
        quantile_weight=0.3,
        lr_scheduler="cosine",
        warmup_epochs=2,
        cosine_t_max_epochs=10,
    )
    kwargs.update(module_overrides)
    module = CNNLightningModule(**kwargs)
    return model, module


def _batch():
    x = torch.randn(BATCH, WINDOW, N_VARS, LAT, LON)
    y = torch.randn(BATCH, 1)
    return x, y


# ── forward shapes ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,gaussian_nll,quantile_head,loss_fn", VARIANTS)
def test_forward_shape(name, gaussian_nll, quantile_head, loss_fn):
    model, _ = _build(gaussian_nll, quantile_head, loss_fn)
    x, _ = _batch()
    out = model.forward(x)
    assert out.shape == (BATCH, 2 if gaussian_nll else 1)


def test_forward_with_quantile_shapes():
    model, _ = _build(True, True, "GaussianNLLLoss")
    x, _ = _batch()
    y_hat, q_pred = model.forward_with_quantile(x)
    assert y_hat.shape == (BATCH, 2)
    assert q_pred.shape == (BATCH, 1)


def test_forward_with_quantile_requires_quantile_head():
    model, _ = _build(True, False, "GaussianNLLLoss")
    x, _ = _batch()
    with pytest.raises(RuntimeError, match="quantile_head=True"):
        model.forward_with_quantile(x)


# ── _step: finite loss + gradients reach every trainable parameter ──────────


@pytest.mark.parametrize("name,gaussian_nll,quantile_head,loss_fn", VARIANTS)
def test_step_loss_finite_and_all_params_get_gradient(
    name, gaussian_nll, quantile_head, loss_fn
):
    model, module = _build(gaussian_nll, quantile_head, loss_fn)
    batch = _batch()

    loss, pred, y = module._step(batch, "val")
    assert torch.isfinite(loss)
    assert pred.shape == (BATCH, 1)
    assert y.shape == (BATCH, 1)

    loss.backward()
    for name_, p in module.named_parameters():
        assert p.grad is not None, f"{name_} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name_} got a non-finite gradient"


# ── (a) gaussian_nll + quantile_head: gradient isolation between the heads ──


def test_pinball_gradient_does_not_reach_fc_and_nll_gradient_does_not_reach_quantile_head():
    model, module = _build(True, True, "GaussianNLLLoss")
    x, y = _batch()

    y_hat, q_pred = model.forward_with_quantile(x)

    model.zero_grad()
    pinball = module._pinball_loss(q_pred, y)
    pinball.backward(retain_graph=True)
    assert all(p.grad is None for p in model.fc.parameters())
    assert any(p.grad is not None for p in model.quantile_head.parameters())

    model.zero_grad()
    nll_loss, _ = module._loss_and_pred(y_hat, y)
    nll_loss.backward()
    assert all(p.grad is None for p in model.quantile_head.parameters())
    assert any(p.grad is not None for p in model.fc.parameters())


# ── test_step/on_test_epoch_end: accumulate mean/log_var/q_pred/y only ──────


@pytest.mark.parametrize("name,gaussian_nll,quantile_head,loss_fn", VARIANTS)
def test_on_test_epoch_end_concatenates_exactly_what_was_accumulated(
    name, gaussian_nll, quantile_head, loss_fn
):
    model, module = _build(gaussian_nll, quantile_head, loss_fn)
    module.eval()

    module.on_test_epoch_start()
    n_batches = 3
    with torch.no_grad():
        for _ in range(n_batches):
            module.test_step(_batch(), 0)
    module.on_test_epoch_end()

    assert module.test_mean.shape == (n_batches * BATCH,)
    assert module.test_y.shape == (n_batches * BATCH,)
    if gaussian_nll:
        assert module.test_log_var.shape == (n_batches * BATCH,)
    else:
        assert module.test_log_var is None
    if quantile_head:
        assert module.test_q_pred.shape == (n_batches * BATCH,)
    else:
        assert module.test_q_pred is None


# ── lr schedule: warmup then cosine decay ────────────────────────────────────


def test_lr_schedule_warmup_then_cosine_decay():
    model, module = _build(
        True, True, "GaussianNLLLoss", warmup_epochs=5, cosine_t_max_epochs=60
    )
    opt_config = module.configure_optimizers()
    scheduler = opt_config["lr_scheduler"]["scheduler"]
    lr_lambda = scheduler.lr_lambdas[0]

    expected_warmup = [0.2, 0.4, 0.6, 0.8, 1.0]
    for epoch, expected in enumerate(expected_warmup):
        assert lr_lambda(epoch) == pytest.approx(expected)

    post_warmup = [lr_lambda(e) for e in range(5, 61)]
    assert post_warmup[0] == pytest.approx(1.0)
    assert post_warmup[-1] == pytest.approx(0.0, abs=1e-9)
    assert all(
        post_warmup[i + 1] <= post_warmup[i] for i in range(len(post_warmup) - 1)
    ), "factor is not monotonically non-increasing after warmup"


# ── invalid constructor combinations raise ───────────────────────────────────


def test_quantile_head_without_gaussian_nll_raises():
    with pytest.raises(ValueError, match="quantile_head=True requires gaussian_nll"):
        _build(False, True, "MSELoss")


def test_gaussian_nll_with_wrong_loss_fn_raises():
    with pytest.raises(ValueError, match="gaussian_nll=True requires loss_fn"):
        _build(True, False, "MSELoss")


def test_mse_with_wrong_loss_fn_raises():
    with pytest.raises(ValueError, match="gaussian_nll=False requires loss_fn"):
        _build(False, False, "GaussianNLLLoss")


def test_quantile_tau_out_of_range_raises():
    with pytest.raises(ValueError, match="quantile_tau in \\(0, 1\\)"):
        _build(True, True, "GaussianNLLLoss", quantile_tau=1.5)


def test_lr_scheduler_not_cosine_raises():
    with pytest.raises(ValueError, match="lr_scheduler must be 'cosine'"):
        _build(True, True, "GaussianNLLLoss", lr_scheduler="step")


@pytest.mark.parametrize("pooling", ["bogus", ""])
def test_invalid_pooling_raises(pooling):
    with pytest.raises(ValueError, match="pooling must be"):
        CNNLSTMModel(
            in_channels=N_VARS,
            cnn_features=8,
            lstm_hidden=16,
            lstm_layers=1,
            dropout=0.0,
            gaussian_nll=True,
            pooling=pooling,
            padding_mode="zeros",
            quantile_head=False,
            mode="full",
        )


@pytest.mark.parametrize("padding_mode", ["bogus", ""])
def test_invalid_padding_mode_raises(padding_mode):
    with pytest.raises(ValueError, match="padding_mode must be"):
        CNNLSTMModel(
            in_channels=N_VARS,
            cnn_features=8,
            lstm_hidden=16,
            lstm_layers=1,
            dropout=0.0,
            gaussian_nll=True,
            pooling="avg",
            padding_mode=padding_mode,
            quantile_head=False,
            mode="full",
        )


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        _build(True, False, "GaussianNLLLoss", mode="bogus")


# ── mode masking: applied once, inside _encode() ─────────────────────────────

# The NS box (src/data/masking.py) is lat[100:127], lon[150:187] on the real
# grid -- a tiny synthetic grid would make the mask a silent no-op, so these
# two tests use the real spatial size.
REAL_LAT, REAL_LON = 141, 201
NS_LAT, NS_LON = slice(100, 127), slice(150, 187)


@pytest.mark.parametrize("mode", ["local_only", "remote_only"])
def test_mode_masks_gradient_exactly_zero_in_masked_region(mode):
    model, _ = _build(True, False, "GaussianNLLLoss", mode=mode)
    x = torch.randn(1, 2, N_VARS, REAL_LAT, REAL_LON, requires_grad=True)

    model.forward(x).sum().backward()
    grad = x.grad

    if mode == "local_only":
        # everything OUTSIDE the NS box is masked -> no gradient reaches it
        outside = grad.clone()
        outside[:, :, :, NS_LAT, NS_LON] = 0.0
        assert torch.all(outside == 0.0)
    else:  # remote_only: everything INSIDE the NS box is masked
        assert torch.all(grad[:, :, :, NS_LAT, NS_LON] == 0.0)


def test_mode_masking_is_identical_across_entry_points():
    """forward(), forward_with_quantile() and _step() all go through the
    same _encode(), so they must mask (and predict) identically regardless
    of which one is called."""
    model, module = _build(True, True, "GaussianNLLLoss", mode="local_only")
    model.eval()
    module.eval()
    x = torch.randn(1, 2, N_VARS, REAL_LAT, REAL_LON)
    y = torch.randn(1, 1)

    with torch.no_grad():
        y_hat_forward = model.forward(x)
        y_hat_fwq, _ = model.forward_with_quantile(x)
        _, pred_step, _ = module._step((x, y), "val")

    assert torch.equal(y_hat_forward, y_hat_fwq)
    assert torch.equal(pred_step, y_hat_forward[:, 0:1])
