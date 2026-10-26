"""
CNNLSTMModel / CNNLightningModule (src/models/cnn_lstm.py) on tiny synthetic
tensors -- no real data, CPU, seconds. Covers the 2x2 loss variants
(mse / gnll, with and without the quantile head), the gradient isolation
between the two heads, the logged names and the val/test metrics, the lr
schedule, and the invalid constructor combinations that raise.
"""

import math

import pytest
import torch

from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel

# LAT/LON must survive 3 halvings (CNNEncoder's 3 Pool2d(2) layers) without
# collapsing to 0 before the final AdaptiveAvgPool2d((1, 1)).
BATCH, WINDOW, N_VARS, LAT, LON = 2, 4, 3, 16, 16

# (name, loss, quantile_head)
VARIANTS = [
    ("gnll_quantile", "gnll", True),
    ("gnll_only", "gnll", False),
    ("mse_quantile", "mse", True),
    ("mse", "mse", False),
]


def _build(loss, quantile_head, mode="full", gaussian_nll=None, **module_overrides):
    """gaussian_nll (the model's architecture flag) follows loss unless set
    explicitly, to test the mismatch check."""
    if gaussian_nll is None:
        gaussian_nll = loss == "gnll"
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
        loss=loss,
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


@pytest.mark.parametrize("name,loss,quantile_head", VARIANTS)
def test_forward_shape(name, loss, quantile_head):
    model, _ = _build(loss, quantile_head)
    x, _ = _batch()
    out = model.forward(x)
    assert out.shape == (BATCH, 2 if loss == "gnll" else 1)


def test_forward_with_quantile_shapes():
    model, _ = _build("gnll", True)
    x, _ = _batch()
    y_hat, q_pred = model.forward_with_quantile(x)
    assert y_hat.shape == (BATCH, 2)
    assert q_pred.shape == (BATCH, 1)


def test_forward_with_quantile_requires_quantile_head():
    model, _ = _build("gnll", False)
    x, _ = _batch()
    with pytest.raises(RuntimeError, match="quantile_head=True"):
        model.forward_with_quantile(x)


# ── _step: finite loss + gradients reach every trainable parameter ──────────


@pytest.mark.parametrize("name,loss,quantile_head", VARIANTS)
def test_step_loss_finite_and_all_params_get_gradient(name, loss, quantile_head):
    model, module = _build(loss, quantile_head)
    batch = _batch()

    loss, pred, y = module._step(batch, "val")
    assert torch.isfinite(loss)
    assert pred.shape == (BATCH, 1)
    assert y.shape == (BATCH, 1)

    loss.backward()
    for name_, p in module.named_parameters():
        assert p.grad is not None, f"{name_} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name_} got a non-finite gradient"


# ── (a) quantile_head: gradient isolation between the heads ───────────────


@pytest.mark.parametrize("loss", ["gnll", "mse"])
def test_pinball_gradient_does_not_reach_fc_and_main_gradient_does_not_reach_quantile_head(
    loss,
):
    model, module = _build(loss, True)
    x, y = _batch()

    y_hat, q_pred = model.forward_with_quantile(x)

    model.zero_grad()
    pinball = module._pinball_loss(q_pred, y)
    pinball.backward(retain_graph=True)
    assert all(p.grad is None for p in model.fc.parameters())
    assert any(p.grad is not None for p in model.quantile_head.parameters())

    model.zero_grad()
    main_loss, _ = module._loss_and_pred(y_hat, y)
    main_loss.backward()
    assert all(p.grad is None for p in model.quantile_head.parameters())
    assert any(p.grad is not None for p in model.fc.parameters())


# ── test_step/on_test_epoch_end: accumulate mean/log_var/q_pred/y only ──────


@pytest.mark.parametrize("name,loss,quantile_head", VARIANTS)
def test_on_test_epoch_end_concatenates_exactly_what_was_accumulated(
    name, loss, quantile_head
):
    model, module = _build(loss, quantile_head)
    module.eval()

    module.on_test_epoch_start()
    n_batches = 3
    with torch.no_grad():
        for _ in range(n_batches):
            module.test_step(_batch(), 0)
    module.on_test_epoch_end()

    assert module.test_mean.shape == (n_batches * BATCH,)
    assert module.test_y.shape == (n_batches * BATCH,)
    if loss == "gnll":
        assert module.test_log_var.shape == (n_batches * BATCH,)
    else:
        assert module.test_log_var is None
    if quantile_head:
        assert module.test_q_pred.shape == (n_batches * BATCH,)
    else:
        assert module.test_q_pred is None


# ── lr schedule: warmup then cosine decay ────────────────────────────────────


def test_lr_schedule_warmup_then_cosine_decay():
    model, module = _build("gnll", True, warmup_epochs=5, cosine_t_max_epochs=60)
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


@pytest.mark.parametrize("loss", ["GaussianNLLLoss", "MSELoss", "", "nll"])
def test_invalid_loss_raises(loss):
    with pytest.raises(ValueError, match="loss must be one of"):
        _build(loss, False, gaussian_nll=False)


@pytest.mark.parametrize("loss,gaussian_nll", [("gnll", False), ("mse", True)])
def test_loss_and_model_head_mismatch_raises(loss, gaussian_nll):
    with pytest.raises(ValueError, match="needs a model built with"):
        _build(loss, False, gaussian_nll=gaussian_nll)


def test_quantile_head_flag_mismatch_raises():
    model, _ = _build("mse", False)
    with pytest.raises(ValueError, match="model was built with quantile_head"):
        CNNLightningModule(
            model=model, learning_rate=1e-3, weight_decay=1e-4, target_mean=0.0,
            target_std=1.0, loss="mse", quantile_head=True, quantile_tau=0.9,
            quantile_weight=0.3, lr_scheduler="cosine", warmup_epochs=2,
            cosine_t_max_epochs=10,
        )


def test_quantile_tau_out_of_range_raises():
    with pytest.raises(ValueError, match="quantile_tau in \\(0, 1\\)"):
        _build("gnll", True, quantile_tau=1.5)


def test_lr_scheduler_not_cosine_raises():
    with pytest.raises(ValueError, match="lr_scheduler must be 'cosine'"):
        _build("gnll", True, lr_scheduler="step")


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
        _build("gnll", False, mode="bogus")


# ── mode masking: applied once, inside _encode() ─────────────────────────────

# The NS box (src/data/masking.py) is lat[100:127], lon[150:187] on the real
# grid -- a tiny synthetic grid would make the mask a silent no-op, so these
# two tests use the real spatial size.
REAL_LAT, REAL_LON = 141, 201
NS_LAT, NS_LON = slice(100, 127), slice(150, 187)


@pytest.mark.parametrize("mode", ["local_only", "remote_only"])
def test_mode_masks_gradient_exactly_zero_in_masked_region(mode):
    model, _ = _build("gnll", False, mode=mode)
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
    model, module = _build("gnll", True, mode="local_only")
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


# ── logged names and val/test metrics ────────────────────────────────────────


def _record_logs(module):
    logged = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)
    return logged


@pytest.mark.parametrize("name,loss,quantile_head", VARIANTS)
def test_step_logs_main_and_pinball_loss_names(name, loss, quantile_head):
    _, module = _build(loss, quantile_head)
    logged = _record_logs(module)
    module._step(_batch(), "val")
    if quantile_head:
        assert set(logged) == {"val_main_loss", "val_pinball_loss"}
    else:
        assert logged == {}
    assert not any("nll" in k for k in logged)


@pytest.mark.parametrize("name,loss,quantile_head", VARIANTS)
def test_validation_metrics_match_manual_values(name, loss, quantile_head):
    model, module = _build(loss, quantile_head)
    module.eval()
    logged = _record_logs(module)
    preds, qs, ys = [], [], []
    with torch.no_grad():
        for _ in range(3):
            x, y = _batch()
            module.validation_step((x, y), 0)
            y_hat, q_pred = module._forward_dual(x)
            preds.append(y_hat[:, 0]); ys.append(y[:, 0])
            if quantile_head:
                qs.append(q_pred[:, 0])
    pred, y = torch.cat(preds), torch.cat(ys)
    assert {"val_loss", "val_mae", "val_corr"} <= set(logged)
    assert module.val_mae.compute() == pytest.approx(float((pred - y).abs().mean()), abs=1e-6)
    expected_r = float(torch.corrcoef(torch.stack([pred, y]).double())[0, 1])
    # torchmetrics accumulates running sums in float32: ~1e-4 off on 6 samples.
    assert module.val_corr.compute() == pytest.approx(expected_r, abs=1e-3)
    if quantile_head:
        expected_cov = float((y > torch.cat(qs)).float().mean())
        assert module.val_q_coverage.compute() == pytest.approx(expected_cov)
        assert "val_q_coverage" in logged
    else:
        assert not hasattr(module, "val_q_coverage")


@pytest.mark.parametrize("name,loss,quantile_head", VARIANTS)
def test_test_q_coverage_logged_only_with_quantile_head(name, loss, quantile_head):
    model, module = _build(loss, quantile_head)
    module.eval()
    logged = _record_logs(module)
    module.on_test_epoch_start()
    with torch.no_grad():
        for _ in range(3):
            module.test_step(_batch(), 0)
    module.on_test_epoch_end()
    if quantile_head:
        expected = float((module.test_y > module.test_q_pred).float().mean())
        assert float(logged["test_q_coverage"]) == pytest.approx(expected)
    else:
        assert "test_q_coverage" not in logged


@pytest.mark.parametrize("name,loss,quantile_head", VARIANTS)
def test_repeated_test_runs_do_not_accumulate_metrics(name, loss, quantile_head):
    model, module = _build(loss, quantile_head)
    module.eval()
    _record_logs(module)
    batches = [_batch() for _ in range(3)]

    def run():
        module.on_test_epoch_start()
        with torch.no_grad():
            for b in batches:
                module.test_step(b, 0)
        module.on_test_epoch_end()
        out = [float(module.test_mae.compute()), float(module.test_corr.compute())]
        if quantile_head:
            out.append(float(module.test_q_coverage.compute()))
        return out

    first = run()
    module.on_test_epoch_start()  # what a second trainer.test() does first
    assert module.test_mae.update_count == 0
    assert module.test_corr.update_count == 0
    if quantile_head:
        assert module.test_q_coverage.update_count == 0
    assert run() == pytest.approx(first)
