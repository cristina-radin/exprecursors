"""
model_spatial.py — Spatial SST anomaly forecast model.

Architecture: CNN encoder (per frame) → ConvLSTM (temporal) → decoder → (1, H, W)

Input:  x_spatial  (B, T, C, H, W)   e.g. (B, 60, 4, 141, 201)  [ERA5 only]
             or                             (B, 60, 5, 141, 201)  [ERA5 + to_anom]
Output: (B, 1, 141, 201)  — predicted to_anom field at t + lead_time (normalised)

Config keys:
  in_channels       int   number of input variables
  enc_channels      int   CNN encoder output channels (default 128)
  convlstm_hidden   int   ConvLSTM hidden channels (default 64)
  dropout           float (default 0.2)
"""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new


class SpatialEncoder(nn.Module):
    """CNN per frame keeping spatial dims. 141×201 → 17×25."""

    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 70×100
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 35×50
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 17×25
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialDecoder(nn.Module):
    """Upsample (hidden_ch, 17, 25) → (1, 141, 201)."""

    def __init__(self, in_channels: int, target_size: tuple = (141, 201)):
        super().__init__()
        self.target_size = target_size
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),  # → 34×50
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),  # → 68×100
            nn.Upsample(size=target_size, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1),  # → 141×201
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────


class SpatialForecastModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        enc_channels: int = 128,
        convlstm_hidden: int = 64,
        target_size: tuple = (141, 201),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convlstm_hidden = convlstm_hidden
        self.gaussian_nll = False  # compat attribute

        self.encoder = SpatialEncoder(in_channels, enc_channels)
        self.convlstm = ConvLSTMCell(enc_channels, convlstm_hidden)
        self.drop = nn.Dropout2d(dropout)
        self.decoder = SpatialDecoder(convlstm_hidden, target_size)

    def forward(
        self, x_spatial: torch.Tensor, x_temporal: torch.Tensor
    ) -> torch.Tensor:
        B, T, C, H, W = x_spatial.shape

        enc = self.encoder(x_spatial.view(B * T, C, H, W))
        _, ec, h, w = enc.shape
        enc = enc.view(B, T, ec, h, w)

        hid = torch.zeros(B, self.convlstm_hidden, h, w, device=x_spatial.device)
        cel = torch.zeros(B, self.convlstm_hidden, h, w, device=x_spatial.device)
        for t in range(T):
            hid, cel = self.convlstm(enc[:, t], hid, cel)

        return self.decoder(self.drop(hid))  # (B, 1, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning module
# ──────────────────────────────────────────────────────────────────────────────


class SpatialLightningModule(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        ocean_mask: Optional[torch.Tensor] = None,  # (H, W) bool, True = ocean
        target_std: Optional[torch.Tensor] = None,  # (H, W) per-pixel std
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "ocean_mask", "target_std"])
        self.model = model
        self.learning_rate = learning_rate

        # Register ocean_mask and target_std as buffers so they move to GPU automatically
        if ocean_mask is not None:
            self.register_buffer("ocean_mask", ocean_mask.bool())
        else:
            self.ocean_mask = None
        if target_std is not None:
            self.register_buffer("target_std", target_std.float())
        else:
            self.target_std = None

        # Accumulate test predictions for per-pixel Pearson r
        self.test_preds = []
        self.test_targets = []

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE over ocean pixels only. pred/target: (B, 1, H, W) or (B, H, W)."""
        p = pred.squeeze(1)  # (B, H, W)
        t = target.squeeze(1)
        if self.ocean_mask is not None:
            mask = self.ocean_mask.unsqueeze(0).expand_as(p)
            return F.mse_loss(p[mask], t[mask])
        return F.mse_loss(p, t)

    def forward(self, x_spatial, x_temporal):
        return self.model(x_spatial.float(), x_temporal.float())

    def training_step(self, batch, batch_idx):
        xs, xt, y = batch
        loss = self._masked_mse(self(xs, xt), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, xt, y = batch
        loss = self._masked_mse(self(xs, xt), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        xs, xt, y = batch
        pred = self(xs, xt)
        self.test_preds.append(pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        loss = self._masked_mse(pred, y)
        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds, dim=0).squeeze(1)  # (N, H, W)
        targets = torch.cat(self.test_targets, dim=0).squeeze(1)

        # Per-pixel Pearson r
        p_mu = preds.mean(0)
        p_s = preds.std(0)
        t_mu = targets.mean(0)
        t_s = targets.std(0)
        cov = ((preds - p_mu) * (targets - t_mu)).mean(0)
        r_map = cov / (p_s * t_s + 1e-8)  # (H, W)

        # Mean r over ocean (preds/targets are CPU; bring mask to CPU too)
        import numpy as _np

        if self.ocean_mask is not None:
            mean_r = float(_np.nanmean(r_map[self.ocean_mask.cpu()].numpy()))
        else:
            mean_r = float(_np.nanmean(r_map.numpy()))

        self.log("test_mean_r", mean_r)
        print(f"\nTest mean Pearson r (ocean pixels): {mean_r:.4f}")

        # Save r_map and predictions for plotting
        import os

        import numpy as np

        out_dir = self.trainer.default_root_dir if self.trainer else "."
        np.save(os.path.join(out_dir, "r_map.npy"), r_map.numpy())
        np.save(os.path.join(out_dir, "test_preds.npy"), preds.numpy())
        np.save(os.path.join(out_dir, "test_targets.npy"), targets.numpy())
        print(f"Saved r_map.npy, test_preds.npy, test_targets.npy to {out_dir}")

        self.test_preds = []
        self.test_targets = []

    def configure_optimizers(self) -> Dict[str, Any]:
        opt = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }
