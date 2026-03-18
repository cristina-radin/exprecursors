"""
Training script for climate model.
"""

import argparse
from pathlib import Path
import torch
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import LazyDataModule
from model import CNNLightningModule, CNNModel
from dataset import LazyDataset 


def parse_args():
    """Parse command line arguments.
    Ignore this for jupyter notebook usage
    """
    parser = argparse.ArgumentParser(description="Train climate model")
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path or pattern to data files",
    )
    
    # Model arguments
    parser.add_argument(
        "--in_channels",
        type=int,
        default=10,
        help="Number of input channels",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=1,
        help="Number of output channels",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for logs and checkpoints",
    )
    
    return parser.parse_args()


def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    args = argparse.Namespace(**config)

    use_wandb = getattr(args, "use_wandb", False)
    wandb_project = getattr(args, "wandb_project", "climate-model")
    wandb_entity = getattr(args, "wandb_entity", None)
    wandb_run_name = getattr(args, "wandb_run_name", None)
    wandb_tags = getattr(args, "wandb_tags", None)
    
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Imbalanced classes
    print("\nClasses balance...")
    temp_dataset = LazyDataset(
        file_name=args.data_dir,
        variables=args.variables,
        lazy_loading=True,
        normalize=args.normalize,
        add_temporal_features=args.add_temporal_features
    )
    pos_weight = temp_dataset.get_class_balance()
    print(f"pos_weight calculated: {pos_weight.item():.2f}\n")
    
    datamodule = LazyDataModule(config_path="config.yaml")
    
    n_physical = len([v for v in args.variables if v not in ['target', 'land_mask']])
    n_temporal = 3 if args.add_temporal_features else 0
    in_channels = n_physical + 1 + n_temporal 
    print(f"args.variables: {args.variables}")
    print(f"n_physical: {n_physical} (vars: {[v for v in args.variables if v not in ['target', 'land_mask']]})")
    print(f"n_temporal: {n_temporal}")
    print(f"in_channels: {in_channels}")

    model = CNNModel(
        in_channels=in_channels,  
        out_channels=args.out_channels,
    )

    # 6. Lightning module
    lightning_module = CNNLightningModule(
        model=model,
        learning_rate=args.learning_rate,
        pos_weight=pos_weight
    )
    
      # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="cnn-{epoch:02d}-{val_loss:.4f}",
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
    ]
    
    # Set up logger
    loggers = [
        TensorBoardLogger(
            save_dir=output_dir,
            name="logs",
        )
    ]

    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except Exception as exc:
            raise RuntimeError(
                "Wandb logging requested but wandb is not installed. Install it with `pip install wandb`."
            ) from exc

        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            tags=wandb_tags,
            save_dir=str(output_dir),
        )
        loggers.append(wandb_logger)

    for logger in loggers:
        logger.log_hyperparams(config)

    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=loggers if len(loggers) > 1 else loggers[0],
        accelerator="auto",
        devices="auto",
        num_sanity_val_steps=0, 
    )
    
    # Train
    trainer.fit(lightning_module, datamodule=datamodule)
    
    # Test
    trainer.test(lightning_module, datamodule=datamodule)

if __name__ == "__main__":
    main()
