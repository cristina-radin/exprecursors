"""
CNN-LSTM + Temporal Attention model for MHW precursor detection.

Architecture:
  1. CNN encoder:   each spatial frame (n_vars, lat, lon) → feature vector
  2. LSTM:          sequence of feature vectors (window_size,) → hidden state
  3. Attention:     weighted sum over LSTM outputs (which timesteps matter most)
  4. FC head:       [attended_features + temporal_features] → scalar prediction
"""

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
    """

    def __init__(self, in_channels: int, out_features: int = 128, pooling: str = "max"):
        super().__init__()

        if pooling not in ("max", "avg"):
            raise ValueError(f"pooling must be 'max' or 'avg', got {pooling!r}")
        Pool2d = nn.MaxPool2d if pooling == "max" else nn.AvgPool2d

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Pool2d(2),  # 141×201 → 70×100
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Pool2d(2),  # 70×100 → 35×50
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Pool2d(2),  # 35×50 → 17×25
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
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
    ):
        super().__init__()

        self.temporal_features = temporal_features
        self.arch = arch
        self.gaussian_nll = gaussian_nll
        self.pooling = pooling
        self.cnn_encoder = CNNEncoder(
            in_channels, out_features=cnn_features, pooling=pooling
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
            nn.Linear(context_dim + temporal_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_spatial:  (batch, window_size, n_vars, lat, lon)
            x_temporal: (batch, window_size, 3)
        Returns:
            (batch, 2) — [mean, log_var] if gaussian_nll else (batch, 1) — [mean]
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
            combined = torch.cat([context, temporal_summary], dim=-1)
        else:
            combined = context
        return self.fc(combined)  # (batch, 1)

    def forward_with_attention(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
    ):
        """Like forward() but also returns attention weights.

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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate
        self.target_mean = target_mean
        self.target_std = target_std
        self.gaussian_nll = gaussian_nll
        if gaussian_nll:
            self.loss_fn = nn.GaussianNLLLoss()
        else:
            self.loss_fn = nn.L1Loss() if loss_fn == "MAELoss" else nn.MSELoss()

        self.test_mae = MeanAbsoluteError()
        self.test_corr = PearsonCorrCoef()

        self.test_preds = []
        self.test_targets = []

    def forward(self, x_spatial, x_temporal):
        return self.model(x_spatial.float(), x_temporal.float())

    def _loss_and_pred(self, y_hat, y):
        """
        y_hat: (batch, 2) [mean, log_var] if gaussian_nll else (batch, 1) [mean].
        Returns (loss, pred) where pred is always (batch, 1) — the mean, for
        metrics/logging/plots (all downstream code expects a single value).
        """
        if self.gaussian_nll:
            mean = y_hat[:, 0:1]
            log_var = y_hat[:, 1:2].clamp(min=-10.0, max=10.0)  # numerical stability
            var = torch.exp(log_var)
            loss = self.loss_fn(mean, y, var)
            return loss, mean
        return self.loss_fn(y_hat, y), y_hat

    def training_step(self, batch, batch_idx):
        x_spatial, x_temporal, y = batch
        y_hat = self(x_spatial, x_temporal)
        loss, _ = self._loss_and_pred(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_spatial, x_temporal, y = batch
        y_hat = self(x_spatial, x_temporal)
        loss, _ = self._loss_and_pred(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x_spatial, x_temporal, y = batch
        y_hat = self(x_spatial, x_temporal)
        loss, pred = self._loss_and_pred(y_hat, y)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
