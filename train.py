"""
Training script for climate model.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import LazyDataModule
from model import CNNLightningModule, CNNModel


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
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    datamodule = LazyDataModule(
        train_files=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    # Initialize model
    model = CNNModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    
    # Wrap in Lightning module
    lightning_module = CNNLightningModule(
        model=model,
        learning_rate=args.learning_rate,
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
            patience=10,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    ) # we could also use wandb: https://wandb.ai/home
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
    )
    
    # Train
    trainer.fit(lightning_module, datamodule=datamodule)
    
    # Test
    trainer.test(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
