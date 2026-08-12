"""
Training script for MHW precursor detection (CNN-LSTM + Temporal Attention).
"""

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
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel


class LossCurvePlotCallback(Callback):
    """Saves train/val loss curves as a PNG at the end of training."""

    def __init__(self, output_dir: Path):
        self.output_dir   = Path(output_dir)
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
        epochs = range(1, len(self.train_losses) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, self.train_losses, label="train_loss")
        ax.plot(epochs, self.val_losses,   label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss (normalised target)")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        plt.tight_layout()
        path = self.output_dir / "loss_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nLoss curves saved to {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision("medium")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = LazyDataModule(config_path=args.config)
    datamodule.setup()

    model = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden",   256),
        lstm_layers=config.get("lstm_layers",     2),
        temporal_features=config.get("temporal_features", 3),
        dropout=config.get("dropout", 0.3),
    )

    lightning_module = CNNLightningModule(
        model=model,
        learning_rate=config["learning_rate"],
        target_mean=datamodule.target_mean,
        target_std=datamodule.target_std,
        loss_fn=config.get("loss_fn", "MSELoss"),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="cnn-lstm-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=30,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        LossCurvePlotCallback(output_dir),
    ]

    logger = WandbLogger(
        entity="hereon-ksn-expercursors",
        project="mhw-precursors",
        save_dir=str(output_dir),
        config=config,
    )

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
        num_sanity_val_steps=0,
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    trainer.test(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
