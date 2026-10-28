"""
Train and test the CNN-LSTM model on one fold, and save its checkpoints,
metrics and test predictions.

"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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

REPO_ROOT = Path(__file__).resolve().parent.parent


def _require_clean_output_dir(output_dir: Path) -> None:
    """Raise if output_dir/checkpoints already has a .ckpt file from an
    earlier run
    """
    existing = list((output_dir / "checkpoints").glob("*.ckpt"))
    if existing:
        raise RuntimeError(
            f"{output_dir / 'checkpoints'} already has {len(existing)} "
            f"checkpoint(s) from a previous run: "
            f"{[c.name for c in existing]}. Use a clean output_dir."
        )


def _resolve_fold(config: dict, fold: int) -> None:
    """Add "_fold{N}" to run_name. Error if the yaml already has a fold
       or if fold is not in 0..n_folds-1.
    """
    if "fold" in config:
        raise ValueError(
            "fold comes from --fold, remove it from the yaml "
            f"(found fold: {config['fold']!r})."
        )
    if not (0 <= fold < config["n_folds"]):
        raise ValueError(
            f"--fold must be 0 <= fold < n_folds ({config['n_folds']}), "
            f"got {fold}"
        )
    config["fold"] = fold
    config["run_name"] = f"{config['run_name']}_fold{fold}"


def _apply_limit_batches_suffix(config: dict, limit_batches: int) -> None:
    """Short smoke/debug runs
    """
    if limit_batches > 0:
        config["run_name"] = f"{config['run_name']}_limit{limit_batches}"


def _git_commit() -> str:
    """ To track the version for each run
    """
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {proc.stderr}")
    return proc.stdout.strip()


def _write_test_outputs(
    output_dir, lightning_module, datamodule, config, trainer_test_results,
    checkpoint_callback, limit_batches,
):
    """Save the test results for each fold:
    - test_predictions.npz: for each test day, its date, the prediction
      (mean, log-variance, quantile) and the real value (y).
    - test_metrics.json: the test metrics plus the fold, the best
      checkpoint (name, val_loss, epoch) and the git commit.
    All values are SST anomalies: _norm in normalised units, _degC in °C.

    """
    #sample number = first day of its window
    all_sample_indices = np.array(datamodule.test_dataset.indices)
    n_preds = len(lightning_module.test_y)  #number of predictions
    if limit_batches:
        # to run just a few batches for debugging
        sample_indices = all_sample_indices[:n_preds]
    else:
        if n_preds != len(all_sample_indices):
            raise RuntimeError(
                f"{n_preds} test predictions but {len(all_sample_indices)} "
                "test sample indices: devices must be 1 so prediction order "
                "matches the dataset order."
            )
        sample_indices = all_sample_indices

    window_size = config["window_size"]
    lead_time = config["lead_time"]
    # day each sample predicts = first day + window_size - 1 + lead_time
    target_indices = sample_indices + window_size - 1 + lead_time
    full_ds = datamodule.test_dataset.dataset
    dates = full_ds.ds.time.values[target_indices]

    target_mean = datamodule.target_mean
    target_std = datamodule.target_std

    mean_norm = lightning_module.test_mean.numpy()
    y_norm = lightning_module.test_y.numpy()
    npz_kwargs = dict(
        sample_index=sample_indices,
        target_index=target_indices,
        date=dates,
        target_mean=target_mean,
        target_std=target_std,
        mean_norm=mean_norm,
        mean_physical=mean_norm * target_std + target_mean,
        y_norm=y_norm,
        y_physical=y_norm * target_std + target_mean,
    )
    if lightning_module.test_log_var is not None:
        log_var_norm = lightning_module.test_log_var.numpy()
        npz_kwargs["log_var_norm"] = log_var_norm
        npz_kwargs["log_var_physical"] = log_var_norm + 2 * np.log(target_std)
    if lightning_module.test_q_pred is not None:
        q_pred_norm = lightning_module.test_q_pred.numpy()
        npz_kwargs["q_pred_norm"] = q_pred_norm
        npz_kwargs["q_pred_physical"] = q_pred_norm * target_std + target_mean
    np.savez(output_dir / "test_predictions.npz", **npz_kwargs)

    best_ckpt_path = Path(checkpoint_callback.best_model_path)
    best_ckpt_data = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    test_metrics = dict(trainer_test_results[0])
    test_metrics.update(
        fold=config["fold"],
        mode=config["mode"],
        run_name=config["run_name"],
        best_checkpoint_name=best_ckpt_path.name,
        best_checkpoint_val_loss=float(checkpoint_callback.best_model_score),
        best_checkpoint_epoch=best_ckpt_data["epoch"],
        git_commit=_git_commit(),
    )
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=0,
        help="limit train/val/test to this many batches per epoch (0 = no limit)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    _resolve_fold(config, args.fold)
    _apply_limit_batches_suffix(config, args.limit_batches)

    if config["in_channels"] != len(config["variables"]):
        raise ValueError(
            f"in_channels={config['in_channels']} does not match "
            f"len(variables)={len(config['variables'])}"
        )

    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision("medium") #runs faster, negligible effect on results

    output_dir = EXPERIMENTS_DIR / "partition" / config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_clean_output_dir(output_dir)

    # Print and save the exact resolved config this run is using
    print(f"\n=== Resolved config: {args.config} (fold {args.fold}) ===")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    datamodule = LazyDataModule(config=config)
    # Called here (not only by Lightning) because the module below needs
    # target_mean/target_std; Lightning's later setup() call does nothing.
    datamodule.setup()

    model_kwargs = dict(
        in_channels=config["in_channels"],
        cnn_features=config["cnn_features"],
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
        gaussian_nll=config["loss"] == "gnll",  # architecture: [mean, log_var] head
        pooling=config["pooling"],
        padding_mode=config["padding_mode"],
        quantile_head=config["quantile_head"],
        mode=config["mode"],
    )
    model = CNNLSTMModel(**model_kwargs)

    save_model_config(output_dir, **model_kwargs)

    lightning_module = CNNLightningModule(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        target_mean=datamodule.target_mean,
        target_std=datamodule.target_std,
        loss=config["loss"],
        quantile_head=config["quantile_head"],
        quantile_tau=config["quantile_tau"],
        quantile_weight=config["quantile_weight"],
        lr_scheduler=config["lr_scheduler"],
        warmup_epochs=config["warmup_epochs"],
        cosine_t_max_epochs=config["cosine_t_max_epochs"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="cnn-lstm-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=config["save_top_k"],
    )
    callbacks = [
        checkpoint_callback,
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

    csv_logger = CSVLogger(save_dir=str(output_dir), name="", version="")

    limit = args.limit_batches if args.limit_batches > 0 else 1.0
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
        logger=[wandb_logger, csv_logger],
        accelerator="auto",
        devices=1,  # single device: test prediction order must match the dataset
        num_sanity_val_steps=2,
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    # ckpt_path="best" (not "last")
    test_results = trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")
    _write_test_outputs(
        output_dir, lightning_module, datamodule, config, test_results,
        checkpoint_callback, args.limit_batches,
    )


if __name__ == "__main__":
    main()
