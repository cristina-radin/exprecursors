"""
train_local_only.py — Train NS-local-only model.

Identical to train.py but masks ALL variables outside the North Sea bounding box
during BOTH training and validation (applied in the Lightning module, not the dataset).
This gives a proper "local-only" counterpart to the existing masked model (mask_ns_sst).

NS box (same as dataset.py): lat[100:127], lon[150:187] (~50-63N, -5-13E)
Everything outside this box is set to 0.0 before every forward pass.

Usage:
  python local_only/train_local_only.py --config local_only/configs/fold0.yaml
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel


# ── NS-only masking Lightning wrapper ─────────────────────────────────────────

class LocalOnlyLightningModule(CNNLightningModule):
    """Zeros all spatial inputs outside the NS box before every forward pass."""

    _NS_LAT = slice(100, 127)
    _NS_LON = slice(150, 187)

    def _mask(self, xs: torch.Tensor) -> torch.Tensor:
        """xs: (B, T, C, H, W) — keep only NS box, zero the rest."""
        masked = torch.zeros_like(xs)
        masked[:, :, :, self._NS_LAT, self._NS_LON] = xs[:, :, :, self._NS_LAT, self._NS_LON]
        return masked

    def training_step(self, batch, batch_idx):
        xs, xt, y = batch
        return super().training_step((self._mask(xs), xt, y), batch_idx)

    def validation_step(self, batch, batch_idx):
        xs, xt, y = batch
        return super().validation_step((self._mask(xs), xt, y), batch_idx)

    def test_step(self, batch, batch_idx):
        xs, xt, y = batch
        return super().test_step((self._mask(xs), xt, y), batch_idx)


# ── Loss curve callback (same as train.py) ────────────────────────────────────

class LossCurvePlotCallback(Callback):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.train_losses = []
        self.val_losses   = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss_epoch")
        if loss is not None:
            self.train_losses.append(float(loss))

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(float(loss))

    def on_train_end(self, trainer, pl_module):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.train_losses, label="train_loss")
        ax.plot(self.val_losses,   label="val_loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); plt.tight_layout()
        plt.savefig(self.output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision("medium")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = LazyDataModule(config_path=args.config)
    datamodule.setup()

    model = CNNLSTMModel(
        in_channels      = config["in_channels"],
        cnn_features     = config.get("cnn_features",   256),
        lstm_hidden      = config.get("lstm_hidden",     512),
        lstm_layers      = config.get("lstm_layers",       4),
        temporal_features= config.get("temporal_features", 0),
        dropout          = config.get("dropout",          0.3),
        arch             = config.get("arch",      "lstm_only"),
        gaussian_nll     = config.get("gaussian_nll",   True),
    )

    lightning_module = LocalOnlyLightningModule(
        model        = model,
        learning_rate= config["learning_rate"],
        target_mean  = datamodule.target_mean,
        target_std   = datamodule.target_std,
        loss_fn      = config.get("loss_fn", "MSELoss"),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath  = output_dir / "checkpoints",
            filename = "cnn-lstm-{epoch:02d}-{val_loss:.4f}",
            monitor  = "val_loss", mode="min", save_top_k=3,
        ),
        EarlyStopping(monitor="val_loss", patience=30, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        LossCurvePlotCallback(output_dir),
    ]

    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger(save_dir=str(output_dir), name="logs")

    trainer = pl.Trainer(
        max_epochs        = config["max_epochs"],
        callbacks         = callbacks,
        logger            = logger,
        accelerator       = "auto",
        devices           = "auto",
        num_sanity_val_steps = 0,
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    trainer.test(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
