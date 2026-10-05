"""
train_partition.py — Train partition experiments for local vs remote predictability.

Three conditions, same TbotAtm variable set (ptho_bot + atmosphere):
  --mode full        : no masking (baseline)
  --mode remote_only : zero ALL channels inside NS box  → only remote info
  --mode local_only  : zero ALL channels outside NS box → only local NS info

NS box (same as dataset.py): lat[100:127], lon[150:187]

Usage:
  python scripts/train_partition.py --config configs/partition/remote/fold0.yaml --mode remote_only
  python scripts/train_partition.py --config configs/partition/local/fold0.yaml  --mode local_only
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.data.masking import mask_local, mask_remote
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.checkpoints import save_model_config
from src.utils.paths import EXPERIMENTS_DIR

# ── Remote-only: zero everything INSIDE NS box ───────────────────────────────


class RemoteOnlyLightningModule(CNNLightningModule):
    """Zeros all spatial channels inside the NS box in every forward pass."""

    def _mask(self, xs: torch.Tensor) -> torch.Tensor:
        return mask_remote(xs)

    def training_step(self, batch, batch_idx):
        xs, y = batch
        return super().training_step((self._mask(xs), y), batch_idx)

    def validation_step(self, batch, batch_idx):
        xs, y = batch
        return super().validation_step((self._mask(xs), y), batch_idx)

    def test_step(self, batch, batch_idx):
        xs, y = batch
        return super().test_step((self._mask(xs), y), batch_idx)


# ── Local-only: zero everything OUTSIDE NS box ───────────────────────────────


class LocalOnlyLightningModule(CNNLightningModule):
    """Zeros all spatial channels outside the NS box in every forward pass."""

    def _mask(self, xs: torch.Tensor) -> torch.Tensor:
        return mask_local(xs)

    def training_step(self, batch, batch_idx):
        xs, y = batch
        return super().training_step((self._mask(xs), y), batch_idx)

    def validation_step(self, batch, batch_idx):
        xs, y = batch
        return super().validation_step((self._mask(xs), y), batch_idx)

    def test_step(self, batch, batch_idx):
        xs, y = batch
        return super().test_step((self._mask(xs), y), batch_idx)


# ── Loss curve callback ───────────────────────────────────────────────────────


class LossCurvePlotCallback(Callback):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.train_losses, self.val_losses = [], []

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
        ax.plot(self.val_losses, label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

MODE_MAP = {
    "full": CNNLightningModule,
    "remote_only": RemoteOnlyLightningModule,
    "local_only": LocalOnlyLightningModule,
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=list(MODE_MAP.keys()))
    parser.add_argument("--fast_dev_run", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if config["in_channels"] != len(config["variables"]):
        raise ValueError(
            f"in_channels={config['in_channels']} does not match "
            f"len(variables)={len(config['variables'])}"
        )

    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision("medium")

    output_dir = EXPERIMENTS_DIR / "partition" / config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print and save the exact resolved config this run is using, both in
    # the SLURM/stdout log and as a standalone file in output_dir (survives
    # after the run and doesn't require wandb access to inspect later).
    print(f"\n=== Resolved config: {args.config} ===")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    datamodule = LazyDataModule(config_path=args.config)
    datamodule.setup()

    model_kwargs = dict(
        in_channels=config["in_channels"],
        cnn_features=config["cnn_features"],
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
        gaussian_nll=config["gaussian_nll"],
        pooling=config["pooling"],
        padding_mode=config["padding_mode"],
        quantile_head=config["quantile_head"],
    )
    model = CNNLSTMModel(**model_kwargs)
    # Ground truth for eval/XAI scripts (load_model_config) — the exact
    # resolved kwargs used to build `model`, so they can't independently
    # re-derive them. See src/utils/checkpoints.py.
    save_model_config(output_dir, **model_kwargs)

    LightningClass = MODE_MAP[args.mode]
    lightning_module = LightningClass(
        model=model,
        learning_rate=config["learning_rate"],
        target_mean=datamodule.target_mean,
        target_std=datamodule.target_std,
        loss_fn=config["loss_fn"],
        gaussian_nll=config["gaussian_nll"],
        quantile_head=config["quantile_head"],
        quantile_tau=config["quantile_tau"],
        quantile_weight=config["quantile_weight"],
        lr_scheduler=config["lr_scheduler"],
        warmup_epochs=config["warmup_epochs"],
        cosine_t_max_epochs=config["cosine_t_max_epochs"],
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="cnn-lstm-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val_loss", patience=30, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        LossCurvePlotCallback(output_dir),
    ]

    import os

    from pytorch_lightning.loggers import WandbLogger

    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")
    if not wandb_entity or not wandb_project:
        raise RuntimeError(
            "WANDB_ENTITY and WANDB_PROJECT must be set. "
            "Add them to .env or export before running."
        )
    fold = config.get("fold", 0)
    seed = config.get("seed", 42)
    run_name = f"{args.mode}_fold{fold}_seed{seed}"
    logger = WandbLogger(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        save_dir=str(output_dir),
        mode=os.environ.get("WANDB_MODE", "online"),
        config=config,
    )

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
        num_sanity_val_steps=2,
        fast_dev_run=args.fast_dev_run if args.fast_dev_run > 0 else False,
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    # ckpt_path=None tests the current in-memory weights (whatever epoch
    # EarlyStopping stopped at), not the best val_loss checkpoint
    # ModelCheckpoint actually saved -- GaussianNLLLoss's variance term can
    # spike val_loss well above its own best epoch late in training, so
    # "best" and "last" can differ a lot. fast_dev_run disables checkpoint
    # saving entirely, so "best" isn't available there.
    ckpt_path = "best" if not args.fast_dev_run else None
    trainer.test(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
