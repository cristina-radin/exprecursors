"""
train_spatial.py — Training script for spatial SST anomaly prediction.

Usage:
  python scripts_spatial/train_spatial.py --config configs/spatial/SSTAtm_fold0.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src_spatial"))
from dataset_spatial import SpatialDataset
from model_spatial import SpatialForecastModel, SpatialLightningModule


def build_splits(ds: SpatialDataset, config: dict):
    """K-fold block-year split — same logic as scalar experiments for comparability."""
    n_folds = config.get("n_folds", 5)
    fold = config.get("fold", 0)
    seed = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.15)

    total = len(ds)
    target_years = np.array(
        [int(ds.years[i + ds.window_size - 1 + ds.lead_time]) for i in range(total)]
    )
    unique_years = np.sort(np.unique(target_years))
    n_years = len(unique_years)

    rng_fold = np.random.default_rng(0)
    shuffled = rng_fold.permutation(unique_years)
    fold_groups = np.array_split(shuffled, n_folds)
    test_years = set(fold_groups[fold].tolist())

    remaining = np.concatenate([fold_groups[i] for i in range(n_folds) if i != fold])
    # Aug 24 2026 (spatial data/split audit, F3): was rng(seed) alone --
    # `remaining` shares n_folds-2 of its n_folds-1 year-blocks between any
    # two folds, so reshuffling it with the SAME seed across folds produced
    # near-identical val_years sets (measured 83-100% pairwise overlap on
    # real 1985-2024 data) -- undermining independence between folds'
    # early-stopping/checkpoint-selection. Same bug class as known_issues
    # #1/#42's kfold val_years collision. seed+fold breaks the alignment.
    rng = np.random.default_rng(seed + fold)
    rng.shuffle(remaining)
    n_val = max(1, round(n_years * val_ratio))
    val_years = set(remaining[:n_val].tolist())
    train_years = set(remaining[n_val:].tolist())

    train_idx = [i for i in range(total) if target_years[i] in train_years]
    val_idx = [i for i in range(total) if target_years[i] in val_years]
    test_idx = [i for i in range(total) if target_years[i] in test_years]

    n_test = len(test_years)
    print(
        f"K-fold (fold={fold}/{n_folds}): "
        f"train={n_years - n_val - n_test} yrs  val={n_val} yrs  test={n_test} yrs"
    )
    print(
        f"  Samples — train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}"
    )
    # Mandatory output (plan rule: "sacar las configs reales resueltas como
    # output", extended here per the spatial audit's F5 finding -- this
    # script used to print only counts, never the actual years).
    print(f"  train_years ({len(train_years)}): {sorted(train_years)}")
    print(f"  val_years   ({len(val_years)}): {sorted(val_years)}")
    print(f"  test_years  ({len(test_years)}): {sorted(test_years)}")

    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config.get("seed", 42))
    torch.set_float32_matmul_precision("medium")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mandatory output (plan rule, per spatial audit's F5 finding -- this
    # script never dumped its resolved config, unlike train_partition.py).
    print(f"\n=== Resolved config: {args.config} ===")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    # Dataset and splits
    ds = SpatialDataset(args.config)
    train_idx, val_idx, test_idx = build_splits(ds, config)
    ds.compute_stats(train_idx)

    num_workers = config.get("num_workers", 4)
    batch_size = config.get("batch_size", 4)

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Model
    model = SpatialForecastModel(
        in_channels=config["in_channels"],
        enc_channels=config.get("enc_channels", 128),
        convlstm_hidden=config.get("convlstm_hidden", 64),
        target_size=(141, 201),
        dropout=config.get("dropout", 0.2),
    )

    lm = SpatialLightningModule(
        model=model,
        learning_rate=config.get("learning_rate", 1e-4),
        ocean_mask=ds.ocean_mask,
        target_std=ds.target_pix_std,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="spatial-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
        ),
        EarlyStopping(monitor="val_loss", patience=30, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 200),
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
        accelerator="auto",
        devices="auto",
        num_sanity_val_steps=0,
        default_root_dir=str(output_dir),
    )

    trainer.fit(lm, train_loader, val_loader)

    # Test on single GPU to avoid DistributedSampler repeating samples
    test_trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        default_root_dir=str(output_dir),
    )
    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_trainer.test(lm, test_loader, ckpt_path=best_ckpt)

    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
