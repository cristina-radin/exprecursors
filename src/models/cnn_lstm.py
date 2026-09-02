"""
CNN-LSTM + Temporal Attention model for MHW precursor detection.

Architecture:
  1. CNN encoder:   each spatial frame (n_vars, lat, lon) → feature vector
  2. LSTM:          sequence of feature vectors (window_size,) → hidden state
  3. Attention:     weighted sum over LSTM outputs (which timesteps matter most)
  4. FC head:       [attended_features + temporal_features] → scalar prediction
"""

import math
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef

# =============================================================================
# CNN Encoder — one spatial frame → feature vector
# =============================================================================


class CNNEncoder(nn.Module):
    """
    Encodes a single spatial frame (n_vars, lat, lon) into a feature vector.

    Args:
        in_channels: number of input variables (e.g. 5)
        out_features: size of the output feature vector
        pooling: "max" (default, original architecture) or "avg". AvgPool
            spreads the IG/gradient backward pass over the full 2×2 window
            instead of routing it through a single argmax position — avoids
            the ~8px periodic grid artifact documented in known_issues.md #26
            (vanilla-gradient saliency through 3 cascaded MaxPool2d layers).
            Changes only these 3 pooling layers, not AdaptiveAvgPool2d at the
            end (already an avg-pool, unaffected either way).
        padding_mode: "zeros" (default, original architecture) or "reflect".
            Zero padding introduces a hard discontinuity at the domain
            boundary (land/edge pixels are already NaN→0 via the land mask,
            so zero-padding adds a second, purely artificial edge on top of
            that) — the CNN's gradient reacts to this edge, producing
            boundary-band artifacts in IG/gradient saliency maps distinct
            from the #26 pooling-grid artifact. Reflect padding mirrors the
            interior signal across the border instead of introducing a new
            zero discontinuity. Applies to all 4 Conv2d layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int = 128,
        pooling: str = "max",
        padding_mode: str = "zeros",
    ):
        super().__init__()

        if pooling not in ("max", "avg"):
            raise ValueError(f"pooling must be 'max' or 'avg', got {pooling!r}")
        if padding_mode not in ("zeros", "reflect"):
            raise ValueError(
                f"padding_mode must be 'zeros' or 'reflect', got {padding_mode!r}"
            )
        Pool2d = nn.MaxPool2d if pooling == "max" else nn.AvgPool2d

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=3, padding=1, padding_mode=padding_mode
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Pool2d(2),  # 141×201 → 70×100
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Pool2d(2),  # 70×100 → 35×50
            nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Pool2d(2),  # 35×50 → 17×25
            nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (batch, 256, 1, 1)
            nn.Flatten(),  # → (batch, 256)
        )

        self.fc = nn.Linear(256, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_vars, lat, lon) → (batch, out_features)"""
        return self.fc(self.cnn(x))


# =============================================================================
# Temporal Attention — which timesteps in the window matter most
# =============================================================================


class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over LSTM output sequence.

    Args:
        hidden_size: LSTM hidden size
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (batch, window_size, hidden_size)
        Returns:
            context: (batch, hidden_size) — weighted sum over time
        """
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, window_size)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, window_size, 1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden_size)
        return context


# =============================================================================
# Full model
# =============================================================================


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM + Temporal Attention for regression.

    Args:
        in_channels:     number of spatial input variables
        cnn_features:    CNN encoder output size
        lstm_hidden:     LSTM hidden size
        lstm_layers:     number of LSTM layers
        temporal_features: number of temporal scalar features (year, sin, cos = 3)
        dropout:         dropout in LSTM
        pooling:         "max" (default) or "avg" — see CNNEncoder docstring
        padding_mode:    "zeros" (default) or "reflect" — see CNNEncoder docstring
    """

    def __init__(
        self,
        in_channels: int = 5,
        cnn_features: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        temporal_features: int = 3,
        dropout: float = 0.3,
        arch: str = "lstm_only",
        gaussian_nll: bool = False,
        pooling: str = "max",
        quantile_head: bool = False,
        padding_mode: str = "zeros",
        state_feature: bool = False,
    ):
        super().__init__()

        self.temporal_features = temporal_features
        self.arch = arch
        self.gaussian_nll = gaussian_nll
        self.pooling = pooling
        self.padding_mode = padding_mode
        self.quantile_head_enabled = quantile_head
        # Aug 23 2026: opt-in extra scalar input, the target's own value at
        # the last input-window day (same quantity lag-persistence uses) --
        # concatenated directly to the LSTM/attention context, NOT mixed
        # into `temporal_features`'s mean-pooled summary (see _encode()
        # below) since averaging over the 60-day window would destroy
        # exactly the "value right now" information this feature exists to
        # provide. See docs/narrative.md's Aug 23 2026 hybrid-model entry
        # for why (linear post-hoc hybrid ceiling was small, testing
        # whether a nonlinear/state-dependent combination beats it).
        self.state_feature = state_feature
        state_dim = 1 if state_feature else 0
        self.cnn_encoder = CNNEncoder(
            in_channels,
            out_features=cnn_features,
            pooling=pooling,
            padding_mode=padding_mode,
        )

        if arch == "attention_only":
            # True CNN + Attention, no recurrence: attention operates directly
            # on the per-timestep CNN features. Previously this branch still
            # built and ran an LSTM (arch only changed the pooling *after* the
            # LSTM), making "attention_only" structurally identical to
            # "lstm_attention" — see project_arch_naming_bug memory.
            self.lstm = None
            context_dim = cnn_features
        else:
            self.lstm = nn.LSTM(
                input_size=cnn_features,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            context_dim = lstm_hidden

        if arch != "lstm_only":
            self.attention = TemporalAttention(context_dim)

        # FC head: LSTM/attention/CNN-attention features (+ temporal features
        # if any) → [mean, log_var] if gaussian_nll else [mean] only.
        out_dim = 2 if gaussian_nll else 1
        self.fc = nn.Sequential(
            nn.Linear(context_dim + temporal_features + state_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

        # Independent auxiliary head: predicts a single conditional quantile
        # of the target (tau set by the caller's pinball loss, e.g. 0.9).
        # Own parameters, no weight sharing with self.fc — only the backbone
        # (cnn_encoder / lstm / attention) is shared, so a pinball-loss
        # gradient on this head's output never reaches self.fc's mean/log_var
        # and vice versa. NOT the same thing as Hobday's p90_thresh (a fixed
        # climatological, day-of-year threshold defined in src/utils/hobday.py)
        # — this is a per-timestep model output. Call it `quantile_pred` /
        # `pred_quantile`, never `p90`/`q90`, in any downstream eval code.
        if quantile_head:
            self.quantile_head = nn.Sequential(
                nn.Linear(context_dim + temporal_features + state_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

    def _encode(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        x_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Backbone: (x_spatial, x_temporal[, x_state]) -> combined feature
        vector, fed into self.fc and (if enabled) self.quantile_head.
        Single source for this computation — forward() and
        forward_with_quantile() both call this instead of each keeping
        their own copy, so a future backbone change (new layer, dropout,
        etc.) can't silently diverge between the two entry points.

        Args:
            x_spatial:  (batch, window_size, n_vars, lat, lon)
            x_temporal: (batch, window_size, 3)
            x_state:    (batch, 1) or None -- required iff
                self.state_feature=True. Concatenated directly, NOT
                mean-pooled like x_temporal -- it is a single "value right
                now" scalar (same quantity lag-persistence uses), and
                averaging it over the window would destroy exactly the
                information it exists to carry.
        Returns:
            combined: (batch, context_dim [+ temporal_features] [+ 1])
        """
        batch, window, n_vars, lat, lon = x_spatial.shape

        # Encode each frame with the CNN
        x_flat = x_spatial.view(batch * window, n_vars, lat, lon)
        features = self.cnn_encoder(x_flat)  # (batch*window, cnn_features)
        features = features.view(batch, window, -1)  # (batch, window, cnn_features)

        if self.arch == "attention_only":
            context = self.attention(
                features
            )  # attention directly over CNN features, no LSTM
        else:
            lstm_out, _ = self.lstm(features)  # (batch, window, lstm_hidden)
            if self.arch == "lstm_only":
                context = lstm_out[:, -1, :]  # last timestep, no attention
            else:
                context = self.attention(lstm_out)  # (batch, lstm_hidden)

        if self.temporal_features > 0:
            temporal_summary = x_temporal.mean(dim=1)
            context = torch.cat([context, temporal_summary], dim=-1)

        if self.state_feature:
            if x_state is None:
                raise ValueError(
                    "state_feature=True but x_state is None -- the dataset "
                    "must be configured with use_state_feature=True to "
                    "produce it (no silent fallback, known_issues.md "
                    "convention)."
                )
            context = torch.cat([context, x_state], dim=-1)

        return context

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        x_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x_spatial:  (batch, window_size, n_vars, lat, lon)
            x_temporal: (batch, window_size, 3)
            x_state:    (batch, 1) or None -- see _encode()
        Returns:
            (batch, 2) — [mean, log_var] if gaussian_nll else (batch, 1) — [mean]
        """
        combined = self._encode(x_spatial, x_temporal, x_state)
        return self.fc(combined)  # (batch, 1)

    def forward_with_quantile(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        x_state: torch.Tensor = None,
    ):
        """Like forward(), but also returns the auxiliary quantile head's
        output. Requires quantile_head=True at construction.

        Shares the exact backbone computation with forward() via _encode()
        — the gaussian head's (mean, log_var) output and loss are unaffected
        whether or not this method is ever called.

        Returns:
            y_hat:  (batch, 2) or (batch, 1) — identical to forward()
            q_pred: (batch, 1) — predicted conditional quantile (tau is a
                training-loss concept, not stored on the model). Distinct
                from Hobday's p90_thresh (climatological, fixed by DOY) —
                do not conflate the two downstream.
        """
        if not self.quantile_head_enabled:
            raise RuntimeError(
                "forward_with_quantile() requires quantile_head=True at construction"
            )

        combined = self._encode(x_spatial, x_temporal, x_state)
        y_hat = self.fc(combined)
        q_pred = self.quantile_head(combined)
        return y_hat, q_pred

    def forward_with_attention(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
    ):
        """Like forward() but also returns attention weights.

        NOTE: pre-existing third copy of the backbone, not routed through
        _encode() — it needs attn_weights, which _encode()/TemporalAttention
        don't expose. No quantile-head path here (quantile_head + attention
        + XAI is unimplemented — would need TemporalAttention to return
        weights so this can share _encode() instead of reimplementing).

        Returns:
            pred:         (batch, 1)
            attn_weights: (batch, window_size)  — softmax weights over time
        """
        batch, window, n_vars, lat, lon = x_spatial.shape

        x_flat = x_spatial.view(batch * window, n_vars, lat, lon)
        features = self.cnn_encoder(x_flat).view(batch, window, -1)

        attn_input = (
            features if self.arch == "attention_only" else self.lstm(features)[0]
        )

        scores = self.attention.attn(attn_input).squeeze(-1)  # (batch, window)
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, window)
        context = (attn_input * attn_weights.unsqueeze(-1)).sum(dim=1)

        if self.temporal_features > 0:
            temporal_summary = x_temporal.mean(dim=1)
            combined = torch.cat([context, temporal_summary], dim=-1)
        else:
            combined = context
        return self.fc(combined), attn_weights


# =============================================================================
# Lightning module
# =============================================================================


class CNNLightningModule(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        loss_fn: str = "MSELoss",
        gaussian_nll: bool = False,
        quantile_head: bool = False,
        quantile_tau: float = 0.0,
        quantile_weight: float = 0.7,
        focal_weight: bool = False,
        focal_alpha: float = 1.0,
        p90_by_doy: torch.Tensor = None,
        lr_scheduler: str = "reduce_on_plateau",
        warmup_epochs: int = 5,
        cosine_t_max_epochs: int = None,
        use_state_feature: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "p90_by_doy"])

        self.model = model
        self.learning_rate = learning_rate
        self.target_mean = target_mean
        self.target_std = target_std
        self.gaussian_nll = gaussian_nll
        self.quantile_head = quantile_head
        self.quantile_tau = quantile_tau
        self.quantile_weight = quantile_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        # Aug 23 2026: must match the dataset's use_state_feature AND the
        # inner model's state_feature -- mismatches would silently unpack
        # the batch tuple wrong (see _step() below), so enforced at
        # construction (checked against model.state_feature, not just
        # trusted) rather than left to fail confusingly downstream.
        if use_state_feature != getattr(model, "state_feature", False):
            raise ValueError(
                f"use_state_feature={use_state_feature} but model.state_feature="
                f"{getattr(model, 'state_feature', False)} -- these must match "
                "(the batch tuple shape and the model's forward signature both "
                "depend on it)."
            )
        self.use_state_feature = use_state_feature
        if lr_scheduler not in ("reduce_on_plateau", "cosine"):
            raise ValueError(
                f"lr_scheduler must be 'reduce_on_plateau' or 'cosine', got {lr_scheduler!r}"
            )
        self.lr_scheduler_type = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.cosine_t_max_epochs = cosine_t_max_epochs

        if quantile_head and not gaussian_nll:
            raise ValueError(
                "quantile_head=True requires gaussian_nll=True — the dual-head "
                "design attaches the auxiliary quantile head alongside the GNLL "
                "head, it does not replace it."
            )
        if quantile_head and not (0.0 < quantile_tau < 1.0):
            raise ValueError(
                f"quantile_head=True requires quantile_tau in (0, 1), got {quantile_tau}"
            )
        if focal_weight and quantile_head:
            raise ValueError(
                "focal_weight and quantile_head are alternative ways of biasing "
                "the model toward extreme days — not designed to combine. Use "
                "one or the other."
            )
        if focal_weight and not gaussian_nll:
            raise ValueError(
                "focal_weight=True requires gaussian_nll=True — it reweights the "
                "per-sample GaussianNLLLoss term, there is no MSE/MAE equivalent."
            )
        if focal_weight and use_state_feature:
            raise ValueError(
                "focal_weight and use_state_feature not designed to combine -- "
                "_step()'s focal_weight branch unpacks a fixed 4-tuple "
                "(..., target_doy) and does not thread x_state through. "
                "Extend _step() first if this combination is ever needed."
            )
        if focal_weight:
            if p90_by_doy is None or tuple(p90_by_doy.shape) != (365,):
                raise ValueError(
                    "focal_weight=True requires p90_by_doy, a (365,) tensor of "
                    "the Hobday p90 threshold (physical units, same scale as "
                    "the un-normalised target) for each day-of-year — got "
                    f"{None if p90_by_doy is None else tuple(p90_by_doy.shape)}."
                )
            self.register_buffer("p90_by_doy", p90_by_doy.float())

        if gaussian_nll:
            if loss_fn != "GaussianNLLLoss":
                raise ValueError(
                    f"gaussian_nll=True but loss_fn={loss_fn!r} — loss_fn has no "
                    "effect once gaussian_nll is set (the GNLL branch in "
                    "_loss_and_pred() is unconditional), so a mismatched value "
                    "here almost certainly means the config is wrong, not that "
                    "MSE/MAE is actually being used. Set loss_fn: GaussianNLLLoss "
                    "explicitly to make that clear at the call site."
                )
            self.nll_loss = nn.GaussianNLLLoss()
            self.nll_loss_elementwise = nn.GaussianNLLLoss(reduction="none")
        elif loss_fn == "MAELoss":
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        self.test_mae = MeanAbsoluteError()
        self.test_corr = PearsonCorrCoef()

        self.test_preds = []
        self.test_targets = []

    def forward(self, x_spatial, x_temporal, x_state=None):
        return self.model(
            x_spatial.float(),
            x_temporal.float(),
            x_state.float() if x_state is not None else None,
        )

    def _pinball_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pinball (quantile) loss at self.quantile_tau. `pred` must be the
        dedicated quantile-head output (model.forward_with_quantile's q_pred)
        — never the gaussian head's `mean`, so its gradient cannot reach
        self.model.fc (mean/log_var)."""
        tau = self.quantile_tau
        err = target - pred
        return torch.max(tau * err, (tau - 1) * err).mean()

    def _loss_and_pred(self, y_hat, y):
        """
        y_hat: (batch, 2) [mean, log_var] if gaussian_nll else (batch, 1) [mean].
        Returns (loss, pred) where pred is always (batch, 1) — the mean, for
        metrics/logging/plots (all downstream code expects a single value).
        Unaffected by quantile_head — this is exactly the GNLL/MSE loss,
        whether or not an auxiliary quantile head exists.
        """
        if self.gaussian_nll:
            mean = y_hat[:, 0:1]
            log_var = y_hat[:, 1:2].clamp(min=-10.0, max=10.0)  # numerical stability
            var = torch.exp(log_var)
            loss = self.nll_loss(mean, y, var)
            return loss, mean
        return self.loss_fn(y_hat, y), y_hat

    def _forward_dual(self, x_spatial, x_temporal, x_state=None):
        """Returns (y_hat, q_pred). q_pred is None unless quantile_head=True.
        y_hat is identical either way — forward_with_quantile() recomputes
        the same self.fc(combined) as forward(), just also returns the
        independent quantile head's output alongside it."""
        if self.quantile_head:
            return self.model.forward_with_quantile(
                x_spatial.float(),
                x_temporal.float(),
                x_state.float() if x_state is not None else None,
            )
        return self(x_spatial, x_temporal, x_state), None

    def _focal_weighted_loss(self, y_hat, y, target_doy):
        """Per-sample GaussianNLLLoss reweighted toward exceedance days
        (truth > Hobday p90(DOY)), instead of the auxiliary quantile head.
        Keeps mean = E[Y|X] and var = conditional variance both statistically
        unperturbed by any quantile objective — only the sample WEIGHTING
        changes, not what mean/var are fit to predict. Weighted average
        (sum(loss_i * w_i) / sum(w_i)), not weighted sum, so the loss stays
        on the same scale as plain GNLL regardless of how many samples in
        the batch are extreme.
        """
        mean = y_hat[:, 0:1]
        log_var = y_hat[:, 1:2].clamp(min=-10.0, max=10.0)
        var = torch.exp(log_var)
        per_sample_nll = self.nll_loss_elementwise(mean, y, var)  # (batch, 1)

        y_physical = y * self.target_std + self.target_mean
        thresh = self.p90_by_doy[target_doy - 1].unsqueeze(-1)  # (batch, 1)
        is_extreme = (y_physical > thresh).float()
        weight = 1.0 + self.focal_alpha * is_extreme

        loss = (per_sample_nll * weight).sum() / weight.sum()
        return loss, mean, per_sample_nll.mean(), is_extreme.mean()

    def _step(self, batch, split: str):
        """Shared step logic. loss = NLL(mean, log_var) [+ quantile_weight *
        pinball(q_pred, y, tau) if quantile_head] [OR focal-weighted NLL if
        focal_weight — mutually exclusive with quantile_head, see __init__].
        The quantile_head term depends on a disjoint parameter set
        (self.model.fc vs. self.model.quantile_head), so that sum does not
        blend gradients into either head — it only combines them at the
        shared backbone."""
        if self.focal_weight:
            x_spatial, x_temporal, y, target_doy = batch
            y_hat, _ = self._forward_dual(x_spatial, x_temporal)
            loss, pred, plain_nll, frac_extreme = self._focal_weighted_loss(
                y_hat, y, target_doy
            )
            self.log(
                f"{split}_nll_loss_unweighted", plain_nll, on_step=False, on_epoch=True
            )
            self.log(f"{split}_nll_loss_weighted", loss, on_step=False, on_epoch=True)
            self.log(
                f"{split}_frac_extreme", frac_extreme, on_step=False, on_epoch=True
            )
            return loss, pred, y

        if self.use_state_feature:
            x_spatial, x_temporal, y, x_state = batch
        else:
            x_spatial, x_temporal, y = batch
            x_state = None
        y_hat, q_pred = self._forward_dual(x_spatial, x_temporal, x_state)
        loss, pred = self._loss_and_pred(y_hat, y)

        if self.quantile_head:
            pinball = self._pinball_loss(q_pred, y)
            self.log(f"{split}_nll_loss", loss, on_step=False, on_epoch=True)
            self.log(f"{split}_pinball_loss", pinball, on_step=False, on_epoch=True)
            loss = loss + self.quantile_weight * pinball

        return loss, pred, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, "train")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, "val")
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        # Each fold is its own SLURM process today, so this never actually
        # accumulates across folds — but trainer.test() can be called more
        # than once in the same process (e.g. a notebook or an ensemble
        # script), and without this reset test_preds/test_targets would
        # silently grow across calls instead of reflecting just the latest
        # test pass.
        self.test_preds = []
        self.test_targets = []

    def test_step(self, batch, batch_idx):
        loss, pred, y = self._step(batch, "test")

        self.test_mae.update(pred.squeeze(), y.squeeze())
        self.test_corr.update(pred.squeeze(), y.squeeze())

        self.test_preds.append(pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        self.log("test_loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        mae = self.test_mae.compute()
        corr = self.test_corr.compute()

        self.log("test_mae", mae)
        self.log("test_corr", corr)

        mae_physical = mae * self.target_std  # back to °C
        print(
            f"\nTest results:  MAE={mae:.4f} (norm)  MAE={mae_physical:.4f} °C  Pearson r={corr:.4f}"
        )

        # Save plot only from rank 0 to avoid race condition on shared filesystem
        if not self.trainer.is_global_zero:
            return

        import os

        import matplotlib.pyplot as plt

        log_dir = "outputs"
        if self.trainer and hasattr(self.trainer, "default_root_dir"):
            log_dir = self.trainer.default_root_dir

        preds = torch.cat(self.test_preds).squeeze()
        targets = torch.cat(self.test_targets).squeeze()

        plt.figure(figsize=(12, 4))
        plt.plot(targets.numpy(), label="True", alpha=0.7)
        plt.plot(preds.numpy(), label="Predicted", alpha=0.7)
        plt.legend()
        plt.title(
            f"Test predictions vs truth  (MAE={mae_physical:.3f} °C, r={corr:.3f})"
        )
        plt.xlabel("Sample")
        plt.ylabel("SST anomaly normalised (North Sea)")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "test_predictions.png"))
        plt.close()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        if self.lr_scheduler_type == "cosine":
            # Linear warmup for warmup_epochs, then cosine decay to 0 over
            # the remaining epochs. T_max is cosine_t_max_epochs if given
            # (the REALISTIC expected training length, e.g. from prior
            # early-stopping history — not `max_epochs`, which is usually
            # an artificial early-stopping ceiling; annealing against the
            # literal `max_epochs` barely decays at all if the run stops
            # far earlier, silently defeating the point of using cosine).
            # Falls back to the attached Trainer's max_epochs only if
            # cosine_t_max_epochs isn't set. No silent numeric fallback if
            # neither is available -- config bug, must be visible, not
            # guessed at (CLAUDE.md "no silent fallbacks").
            if self.cosine_t_max_epochs is not None:
                max_epochs = self.cosine_t_max_epochs
            elif self.trainer is not None:
                max_epochs = self.trainer.max_epochs
            else:
                raise RuntimeError(
                    "lr_scheduler='cosine' needs either cosine_t_max_epochs set "
                    "explicitly or an attached Trainer with max_epochs -- got neither."
                )
            warmup = self.warmup_epochs

            def lr_lambda(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / max(1, max_epochs - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
