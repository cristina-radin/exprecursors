"""
Smoke test for the 'diff' head added to scripts/ig_partition_quantile.py
(Aug 21 2026) -- IG on q_pred - y_hat_mean directly, to isolate the
gradient direction that differentiates the quantile head from the mean
head (population-averaged per-head maps are 0.998-0.999 correlated and
don't show this, see known_issues.md #51).

Synthetic tensors only, tiny random model, CPU, no real data/checkpoint --
this checks the new code path is wired correctly (shapes, gradient flow,
completeness axiom, and that diff IS mathematically q_pred - mean and not
some trivial zero/constant), not that the trained model's real diff map
looks a particular way.

Run with: pytest tests/test_ig_diff_head.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ig_partition_quantile import integrated_gradients_single_output
from src.models.cnn_lstm import CNNLSTMModel


def _tiny_model():
    torch.manual_seed(0)
    model = CNNLSTMModel(
        in_channels=3,
        cnn_features=8,
        lstm_hidden=16,
        lstm_layers=1,
        temporal_features=0,
        dropout=0.0,
        arch="lstm_only",
        gaussian_nll=True,
        pooling="avg",
        quantile_head=True,
        padding_mode="reflect",
    )
    model.eval()
    return model


def _head_fns(model):
    def mean_head_fn(xs, xt):
        y_hat, _ = model.forward_with_quantile(xs, xt)
        return y_hat[:, 0]

    def quantile_head_fn(xs, xt):
        _, q_pred = model.forward_with_quantile(xs, xt)
        return q_pred[:, 0]

    def diff_head_fn(xs, xt):
        y_hat, q_pred = model.forward_with_quantile(xs, xt)
        return q_pred[:, 0] - y_hat[:, 0]

    return mean_head_fn, quantile_head_fn, diff_head_fn


def test_diff_head_shape_and_nonzero():
    model = _tiny_model()
    mean_fn, q_fn, diff_fn = _head_fns(model)

    torch.manual_seed(1)
    xs = torch.randn(1, 5, 3, 16, 16)  # (batch, window, n_vars, lat, lon)
    xt = torch.zeros(1, 5, 0)

    ig_diff = integrated_gradients_single_output(
        diff_fn, xs, xt, n_steps=6, chunk_size=3
    )
    assert ig_diff.shape == xs.shape
    assert torch.isfinite(ig_diff).all()
    assert ig_diff.abs().sum() > 0, "diff-head IG is all-zero -- gradient not flowing"


def test_diff_head_equals_quantile_minus_mean_not_trivial():
    """diff_head_fn's raw output must actually equal quantile - mean (not,
    say, a copy-paste bug returning just one head), and its IG map must not
    be numerically identical to either head's own IG map (that would mean
    the two heads' gradients happened to be indistinguishable even in this
    tiny untrained model, defeating the purpose of computing it separately)."""
    model = _tiny_model()
    mean_fn, q_fn, diff_fn = _head_fns(model)

    torch.manual_seed(2)
    xs = torch.randn(1, 5, 3, 16, 16)
    xt = torch.zeros(1, 5, 0)

    with torch.no_grad():
        y_hat, q_pred = model.forward_with_quantile(xs, xt)
    assert torch.allclose(diff_fn(xs, xt), q_pred[:, 0] - y_hat[:, 0])

    ig_mean = integrated_gradients_single_output(
        mean_fn, xs, xt, n_steps=6, chunk_size=3
    )
    ig_q = integrated_gradients_single_output(q_fn, xs, xt, n_steps=6, chunk_size=3)
    ig_diff = integrated_gradients_single_output(
        diff_fn, xs, xt, n_steps=6, chunk_size=3
    )

    assert not torch.allclose(ig_diff, ig_mean)
    assert not torch.allclose(ig_diff, ig_q)
    # completeness-style sanity, not an exact check (linearity of IG under
    # subtraction of two heads sharing the same baseline/interpolation path)
    assert torch.allclose(ig_diff, ig_q - ig_mean, atol=1e-5)


def test_diff_head_completeness_axiom():
    """Sum of IG attributions should approximate f(x) - f(baseline) for the
    diff function itself (baseline=0 in normalized input space, same
    convention as integrated_gradients_single_output's own zero baseline).
    Expect 50-step-Riemann-sum-level numerical error, not exact equality --
    same tolerance style as known_issues.md #26's own IG diagnostic."""
    model = _tiny_model()
    _, _, diff_fn = _head_fns(model)

    torch.manual_seed(3)
    xs = torch.randn(1, 5, 3, 16, 16)
    xt = torch.zeros(1, 5, 0)

    with torch.no_grad():
        f_x = diff_fn(xs, xt).item()
        f_baseline = diff_fn(torch.zeros_like(xs), xt).item()

    ig = integrated_gradients_single_output(diff_fn, xs, xt, n_steps=50, chunk_size=5)
    total_attr = ig.sum().item()
    expected = f_x - f_baseline

    rel_err = abs(total_attr - expected) / (abs(expected) + 1e-8)
    assert rel_err < 0.15, (
        f"completeness axiom violated beyond expected Riemann-sum error: "
        f"sum(IG)={total_attr:.6f} vs f(x)-f(baseline)={expected:.6f} "
        f"(rel_err={rel_err:.3f})"
    )
