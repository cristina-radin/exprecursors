"""
train_partition.py — Train partition experiments for local vs remote predictability.

Three conditions, same TbotAtm variable set (ptho_bot + atmosphere), selected
by the yaml's "mode" key (CNNLSTMModel masks the input inside _encode(),
see src/models/cnn_lstm.py):
  mode: full        : no masking (baseline)
  mode: remote_only : zero ALL channels inside NS box  → only remote info
  mode: local_only  : zero ALL channels outside NS box → only local NS info

NS box (same as src/data/masking.py): lat[100:127], lon[150:187]

Usage:
  python scripts/train_partition.py --config configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.checkpoints import save_model_config
from src.utils.paths import EXPERIMENTS_DIR

# ── Main ─────────────────────────────────────────────────────────────────────


def _require_clean_output_dir(output_dir: Path) -> None:
    """Raise if output_dir/checkpoints already has a .ckpt file from an
    earlier run -- best_ckpt() picks the lowest val_loss across the whole
    directory and would silently mix runs.
    """
    existing = list((output_dir / "checkpoints").glob("*.ckpt"))
    if existing:
        raise RuntimeError(
            f"{output_dir / 'checkpoints'} already has {len(existing)} "
            f"checkpoint(s) from a previous run: "
            f"{[c.name for c in existing]}. Use a clean output_dir."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=0,
        help="limit train/val/test to this many batches per epoch (0 = no limit)",
    )
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
    _require_clean_output_dir(output_dir)

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
        mode=config["mode"],
    )
    model = CNNLSTMModel(**model_kwargs)
    # Ground truth for eval/XAI scripts (load_model_config) — the exact
    # resolved kwargs used to build `model`, so they can't independently
    # re-derive them. See src/utils/checkpoints.py.
    save_model_config(output_dir, **model_kwargs)

    lightning_module = CNNLightningModule(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
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
            save_top_k=config["save_top_k"],
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")
    if not wandb_entity or not wandb_project:
        raise RuntimeError(
            "WANDB_ENTITY and WANDB_PROJECT must be set. "
            "Add them to .env or export before running."
        )
    wandb_logger = WandbLogger(
        entity=wandb_entity,
        project=wandb_project,
        name=config["run_name"],
        save_dir=str(output_dir),
        mode=os.environ.get("WANDB_MODE", "online"),
        config=config,
    )
    # name="" version="" so metrics.csv lands directly in output_dir,
    # not output_dir/lightning_logs/version_0/metrics.csv.
    csv_logger = CSVLogger(save_dir=str(output_dir), name="", version="")

    limit = args.limit_batches if args.limit_batches > 0 else 1.0
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
        logger=[wandb_logger, csv_logger],
        accelerator="auto",
        devices="auto",
        num_sanity_val_steps=2,
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    # ckpt_path="best" (not "last"): GaussianNLLLoss's variance term can
    # spike val_loss well above its own best epoch late in training, so
    # "best" and "last" can differ a lot.
    trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
