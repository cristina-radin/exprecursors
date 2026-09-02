"""
train_partition.py — Train partition experiments for local vs remote predictability.

Three conditions, same TbotAtm variable set (ptho_bot + atmosphere):
  --mode full        : no masking (baseline)
  --mode remote_only : zero ALL channels inside NS box  → only remote info
  --mode local_only  : zero ALL channels outside NS box → only local NS info

NS box (same as dataset.py): lat[100:127], lon[150:187]

Usage:
  python partition/train_partition.py --config partition/configs/remote/fold0.yaml --mode remote_only
  python partition/train_partition.py --config partition/configs/local/fold0.yaml  --mode local_only
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
from src.utils.hobday import load_ns_p90

# ── Remote-only: zero everything INSIDE NS box ───────────────────────────────


class RemoteOnlyLightningModule(CNNLightningModule):
    """Zeros all spatial channels inside the NS box in every forward pass."""

    def _mask(self, xs: torch.Tensor) -> torch.Tensor:
        return mask_remote(xs)

    def training_step(self, batch, batch_idx):
        # *rest so this works with both the standard 3-tuple batch and the
        # 4-tuple (..., target_doy) that focal_weight=True adds — the mask
        # only ever touches x_spatial (index 0).
        xs, xt, y, *rest = batch
        return super().training_step((self._mask(xs), xt, y, *rest), batch_idx)

    def validation_step(self, batch, batch_idx):
        xs, xt, y, *rest = batch
        return super().validation_step((self._mask(xs), xt, y, *rest), batch_idx)

    def test_step(self, batch, batch_idx):
        xs, xt, y, *rest = batch
        return super().test_step((self._mask(xs), xt, y, *rest), batch_idx)


# ── Local-only: zero everything OUTSIDE NS box ───────────────────────────────


class LocalOnlyLightningModule(CNNLightningModule):
    """Zeros all spatial channels outside the NS box in every forward pass."""

    def _mask(self, xs: torch.Tensor) -> torch.Tensor:
        return mask_local(xs)

    def training_step(self, batch, batch_idx):
        xs, xt, y, *rest = batch
        return super().training_step((self._mask(xs), xt, y, *rest), batch_idx)

    def validation_step(self, batch, batch_idx):
        xs, xt, y, *rest = batch
        return super().validation_step((self._mask(xs), xt, y, *rest), batch_idx)

    def test_step(self, batch, batch_idx):
        xs, xt, y, *rest = batch
        return super().test_step((self._mask(xs), xt, y, *rest), batch_idx)


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

    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision("medium")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mandatory output (plan rule, Aug 20 2026: "sacar las configs reales
    # resueltas como output"): the exact resolved config this run is using,
    # both in the SLURM/stdout log (readable without opening wandb) and as
    # a standalone file in output_dir (survives after the run, unlike the
    # log, and doesn't require network/wandb access to inspect later).
    print(f"\n=== Resolved config: {args.config} ===")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    datamodule = LazyDataModule(config_path=args.config)
    datamodule.setup()

    model_kwargs = dict(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 256),
        lstm_hidden=config.get("lstm_hidden", 512),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=config.get("temporal_features", 0),
        dropout=config.get("dropout", 0.2),
        arch=config.get("arch", "lstm_only"),
        gaussian_nll=config.get("gaussian_nll", False),
        pooling=config.get("pooling", "max"),
        padding_mode=config.get("padding_mode", "zeros"),
        quantile_head=config.get("quantile_head", False),
        state_feature=config.get("use_state_feature", False),
    )
    model = CNNLSTMModel(**model_kwargs)
    # Ground truth for eval/XAI scripts (load_model_config) — the exact
    # resolved kwargs used to build `model`, so they can't independently
    # re-derive (and drift from) these defaults. See src/utils/checkpoints.py.
    save_model_config(output_dir, **model_kwargs)

    focal_weight = config.get("focal_weight", False)
    p90_by_doy = None
    if focal_weight:
        # NS-box-mean Hobday p90(DOY), same physical units as `target`
        # (un-normalised North Sea SST anomaly) — see src/utils/hobday.py.
        # Requires the config to also set return_target_doy: true so
        # LazyDataset yields the 4-tuple _step()'s focal branch expects;
        # if it doesn't, the mismatch fails loudly on the first training
        # step (unpacking error), not silently.
        p90_by_doy = torch.tensor(load_ns_p90(), dtype=torch.float32)

    LightningClass = MODE_MAP[args.mode]
    lightning_module = LightningClass(
        model=model,
        learning_rate=config["learning_rate"],
        target_mean=datamodule.target_mean,
        target_std=datamodule.target_std,
        loss_fn=config.get("loss_fn", "MSELoss"),
        gaussian_nll=config.get("gaussian_nll", False),
        quantile_head=config.get("quantile_head", False),
        quantile_tau=config.get("quantile_tau", 0.0),
        quantile_weight=config.get("quantile_weight", 0.7),
        focal_weight=focal_weight,
        focal_alpha=config.get("focal_alpha", 1.0),
        p90_by_doy=p90_by_doy,
        lr_scheduler=config.get("lr_scheduler", "reduce_on_plateau"),
        warmup_epochs=config.get("warmup_epochs", 5),
        cosine_t_max_epochs=config.get("cosine_t_max_epochs"),
        use_state_feature=config.get("use_state_feature", False),
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
    # ckpt_path=None (the previous default) tests the CURRENT in-memory
    # weights -- i.e. whatever epoch EarlyStopping happened to stop at,
    # not the best val_loss checkpoint ModelCheckpoint actually selected
    # and saved. Found Aug 20 2026 (known_issues.md #46) after val_loss
    # curves looked "terrible" in wandb: GaussianNLLLoss's variance term
    # can spike val_loss 10-50x above its own best epoch late in training,
    # and every test_mae/test_corr/test_loss number reported so far was
    # silently computed from whichever point training stopped at, not the
    # model that was actually selected as best. fast_dev_run disables
    # checkpoint saving entirely, so "best" isn't available there --
    # falls back to current weights only in that case (dry runs don't
    # need a real checkpoint to sanity-check the code path).
    ckpt_path = "best" if not args.fast_dev_run else None
    trainer.test(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
