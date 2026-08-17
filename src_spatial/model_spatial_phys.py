"""
model_spatial_phys.py — Physics-informed loss for spatial SST prediction.

Subclasses SpatialLightningModule with two additional loss terms:

  1. MLD-weighted MSE
     L_mld = Σ_i w_i(month) · (pred_i - target_i)²  /  Σ_i w_i
     w_i = 1 / MLD_clim(i, month)  — thin MLD (summer) gets higher penalty.
     Physical motivation: thin mixed layer responds faster to heat flux →
     surface anomalies are more sensitive → errors matter more.

  2. Spatial Laplacian smoothness
     L_lap = mean_ocean[(∇² pred)²]
     Encourages spatially coherent predictions consistent with SST physics.

  Total:  L = L_mld + λ_lap * L_lap

λ_lap=0 reduces to pure MLD-weighted MSE (a useful ablation).
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model_spatial import SpatialLightningModule

# Discrete Laplacian kernel (reused across calls)
_LAP_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


class PhysicsLightningModule(SpatialLightningModule):

    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        ocean_mask: Optional[torch.Tensor] = None,
        target_std: Optional[torch.Tensor] = None,
        mld_weights: Optional[torch.Tensor] = None,  # (12, H, W) float
        lambda_lap: float = 0.01,
    ):
        super().__init__(model, learning_rate, ocean_mask, target_std)
        self.lambda_lap = lambda_lap

        if mld_weights is not None:
            self.register_buffer("mld_weights", mld_weights.float())  # (12, H, W)
        else:
            self.mld_weights = None

    # ── Loss helpers ──────────────────────────────────────────────────────────

    def _mld_weighted_mse(
        self,
        pred: torch.Tensor,  # (B, 1, H, W)
        target: torch.Tensor,  # (B, H, W)
        months: torch.Tensor,  # (B,) long, 0-11
    ) -> torch.Tensor:
        p = pred.squeeze(1)  # (B, H, W)
        t = target  # already (B, H, W) from DataLoader

        if self.mld_weights is None:
            # Fallback: plain MSE over ocean
            return self._masked_mse(pred, target)

        w = self.mld_weights[months]  # (B, H, W), indexed by month per sample

        if self.ocean_mask is not None:
            mask = self.ocean_mask  # (H, W) bool
            err_w = ((p - t) ** 2 * w)[:, mask].sum(dim=1)  # (B,)
            w_sum = w[:, mask].sum(dim=1)  # (B,)
            return (err_w / w_sum.clamp(min=1e-8)).mean()
        else:
            return ((p - t) ** 2 * w).mean()

    def _laplacian_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Mean squared discrete Laplacian over ocean pixels."""
        kernel = _LAP_KERNEL.to(pred.device)
        lap = F.conv2d(pred, kernel, padding=1)  # (B, 1, H, W)
        if self.ocean_mask is not None:
            mask = self.ocean_mask.unsqueeze(0).unsqueeze(0).expand_as(lap)
            return (lap[mask] ** 2).mean()
        return (lap**2).mean()

    # ── Lightning steps ───────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        xs, xt, y, months = batch
        pred = self(xs, xt)
        loss_mld = self._mld_weighted_mse(pred, y, months)
        loss_lap = self._laplacian_loss(pred)
        loss = loss_mld + self.lambda_lap * loss_lap

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_mld", loss_mld, on_epoch=True)
        self.log("train_loss_lap", loss_lap, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, xt, y, months = batch
        pred = self(xs, xt)
        loss_mld = self._mld_weighted_mse(pred, y, months)
        loss_lap = self._laplacian_loss(pred)
        loss = loss_mld + self.lambda_lap * loss_lap

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_loss_mld", loss_mld, on_epoch=True)
        self.log("val_loss_lap", loss_lap, on_epoch=True)

    def test_step(self, batch, batch_idx):
        xs, xt, y, months = batch
        pred = self(xs, xt)
        self.test_preds.append(pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        loss_mld = self._mld_weighted_mse(pred, y, months)
        self.log("test_loss", loss_mld)
