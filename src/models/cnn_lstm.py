"""
CNN-LSTM model for MHW precursor detection.

Architecture:
  1. CNN encoder: each spatial frame (n_vars, lat, lon) → feature vector
  2. LSTM:        sequence of feature vectors (window_size,) → hidden state
                  (last timestep is used)
  3. FC head:     hidden state → scalar prediction
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
        out_features: int,
        pooling: str,
        padding_mode: str,
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
# Full model
# =============================================================================


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM for regression.

    Args:
        in_channels:  number of spatial input variables
        cnn_features: CNN encoder output size
        lstm_hidden:  LSTM hidden size
        lstm_layers:  number of LSTM layers
        dropout:      dropout in LSTM
        pooling:      "max" (default) or "avg" — see CNNEncoder docstring
        padding_mode: "zeros" (default) or "reflect" — see CNNEncoder docstring
    """

    def __init__(
        self,
        in_channels: int,
        cnn_features: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
        gaussian_nll: bool,
        pooling: str,
        quantile_head: bool,
        padding_mode: str,
    ):
        super().__init__()

        self.gaussian_nll = gaussian_nll
        self.pooling = pooling
        self.padding_mode = padding_mode
        self.quantile_head_enabled = quantile_head
        self.cnn_encoder = CNNEncoder(
            in_channels,
            out_features=cnn_features,
            pooling=pooling,
            padding_mode=padding_mode,
        )

        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        context_dim = lstm_hidden

        # FC head: last LSTM timestep → [mean, log_var] if gaussian_nll
        # else [mean] only.
        out_dim = 2 if gaussian_nll else 1
        self.fc = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

        # Independent auxiliary head: predicts a single conditional quantile
        # of the target (tau set by the caller's pinball loss, e.g. 0.9).
        # Own parameters, no weight sharing with self.fc — only the backbone
        # (cnn_encoder / lstm) is shared, so a pinball-loss
        # gradient on this head's output never reaches self.fc's mean/log_var
        # and vice versa. NOT the same thing as Hobday's p90_thresh (a fixed
        # climatological, day-of-year threshold defined in src/utils/hobday.py)
        # — this is a per-timestep model output. Call it `quantile_pred` /
        # `pred_quantile`, never `p90`/`q90`, in any downstream eval code.
        if quantile_head:
            self.quantile_head = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

    def _encode(self, x_spatial: torch.Tensor) -> torch.Tensor:
        """Backbone: x_spatial -> feature vector, fed into self.fc and (if
        enabled) self.quantile_head. Single source for this computation —
        forward() and forward_with_quantile() both call this instead of
        each keeping their own copy, so a future backbone change (new
        layer, dropout, etc.) can't silently diverge between the two entry
        points.

        Args:
            x_spatial: (batch, window_size, n_vars, lat, lon)
        Returns:
            combined: (batch, context_dim)
        """
        batch, window, n_vars, lat, lon = x_spatial.shape

        # Encode each frame with the CNN
        x_flat = x_spatial.view(batch * window, n_vars, lat, lon)
        features = self.cnn_encoder(x_flat)  # (batch*window, cnn_features)
        features = features.view(batch, window, -1)  # (batch, window, cnn_features)

        lstm_out, _ = self.lstm(features)  # (batch, window, lstm_hidden)
        return lstm_out[:, -1, :]  # last timestep

    def forward(self, x_spatial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_spatial: (batch, window_size, n_vars, lat, lon)
        Returns:
            (batch, 2) — [mean, log_var] if gaussian_nll else (batch, 1) — [mean]
        """
        combined = self._encode(x_spatial)
        return self.fc(combined)  # (batch, 1)

    def forward_with_quantile(self, x_spatial: torch.Tensor):
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

        combined = self._encode(x_spatial)
        y_hat = self.fc(combined)
        q_pred = self.quantile_head(combined)
        return y_hat, q_pred


# =============================================================================
# Lightning module
# =============================================================================


class CNNLightningModule(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        target_mean: float,
        target_std: float,
        loss_fn: str,
        gaussian_nll: bool,
        quantile_head: bool,
        quantile_tau: float,
        quantile_weight: float,
        lr_scheduler: str,
        warmup_epochs: int,
        cosine_t_max_epochs: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate
        self.target_mean = target_mean
        self.target_std = target_std
        self.gaussian_nll = gaussian_nll
        self.quantile_head = quantile_head
        self.quantile_tau = quantile_tau
        self.quantile_weight = quantile_weight
        if lr_scheduler != "cosine":
            raise ValueError(f"lr_scheduler must be 'cosine', got {lr_scheduler!r}")
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
        if gaussian_nll:
            if loss_fn != "GaussianNLLLoss":
                raise ValueError(
                    f"gaussian_nll=True requires loss_fn='GaussianNLLLoss', got "
                    f"{loss_fn!r}."
                )
            self.nll_loss = nn.GaussianNLLLoss()
        elif loss_fn != "MSELoss":
            raise ValueError(
                f"gaussian_nll=False requires loss_fn='MSELoss', got {loss_fn!r}."
            )
        else:
            self.loss_fn = nn.MSELoss()

        self.test_mae = MeanAbsoluteError()
        self.test_corr = PearsonCorrCoef()

        self.test_preds = []
        self.test_targets = []

    def forward(self, x_spatial):
        return self.model(x_spatial.float())

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

    def _forward_dual(self, x_spatial):
        """Returns (y_hat, q_pred). q_pred is None unless quantile_head=True.
        y_hat is identical either way — forward_with_quantile() recomputes
        the same self.fc(combined) as forward(), just also returns the
        independent quantile head's output alongside it."""
        if self.quantile_head:
            return self.model.forward_with_quantile(x_spatial.float())
        return self(x_spatial), None

    def _step(self, batch, split: str):
        """Shared step logic. loss = NLL(mean, log_var) [+ quantile_weight *
        pinball(q_pred, y, tau) if quantile_head]. The quantile_head term
        depends on a disjoint parameter set (self.model.fc vs.
        self.model.quantile_head), so that sum does not blend gradients
        into either head — it only combines them at the shared backbone."""
        x_spatial, y = batch
        y_hat, q_pred = self._forward_dual(x_spatial)
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
        # Linear warmup for warmup_epochs, then cosine decay to 0 over the
        # remaining epochs. max_epochs is cosine_t_max_epochs if set (the
        # realistic expected training length), else the attached Trainer's
        # max_epochs -- which is usually an early-stopping ceiling far
        # longer than any real run, so cosine_t_max_epochs should be set
        # explicitly whenever the real length is known.
        if self.cosine_t_max_epochs is not None:
            max_epochs = self.cosine_t_max_epochs
        elif self.trainer is not None:
            max_epochs = self.trainer.max_epochs
        else:
            raise RuntimeError(
                "cosine_t_max_epochs is not set and no Trainer is attached -- "
                "need one of the two to pick the decay horizon."
            )
        warmup = self.warmup_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, max_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
