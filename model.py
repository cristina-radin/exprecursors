"""
Model definitions for climate data prediction.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)

class CNNModel(nn.Module):
    """
    Base neural network model.
    
    Args:
        in_channels: Number of input channels/features
        out_channels: Number of output channels/features
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_temporal_features: bool = True,  # ¡NUEVO!

    ):
        super().__init__()
        
               
        # TODO: Define your model architecture
        # now we have (batch, 3, 140, 201)

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(16),
        
            nn.Conv2d(16, 32, kernel_size=5, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, out_channels)
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, lat, lon ...)
            
        Returns:
            Output tensor
        """
        # TODO: Adjust based on your data shape
        # Flatten if needed for MLP
        
        x = self.network(x)
        return x


class CNNLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module wrapping the NN model.
    
    Args:
        model: The neural network model
        learning_rate: Learning rate for optimizer
        loss_fn: Loss function (default: MSELoss)
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        loss_fn: Optional[nn.Module] = None,
        pos_weight: Optional[torch.Tensor] = None, 

        
        
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Metrics for a binbary classification task
        self.test_acc = BinaryAccuracy()
        self.test_auc = BinaryAUROC()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()    
        
        self.test_preds = []
        self.test_targets = []
        
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = x.float()     
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        x, y = batch
        print(f"batch dtypes - x: {x.dtype}, y: {y.dtype}")  
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """Test step."""
        x, y = batch
        y = y.float() 
        y_hat = self(x) # y_hat is the raw output of the model, which is a logit for binary classification. We will apply sigmoid to get probabilities before computing metrics.
        loss = self.loss_fn(y_hat, y)
        
        probs = torch.sigmoid(y_hat) # Convert logits to probabilities for metrics
        preds = (probs > 0.5).float() # Convert probabilities to binary predictions for metrics
        
        self.test_acc.update(preds, y)
        self.test_auc.update(probs, y)
        self.test_precision.update(probs, y)
        self.test_recall.update(probs, y)
        self.test_f1.update(probs, y)

        self.test_preds.append(probs.detach().cpu())
        self.test_targets.append(y.detach().cpu())

   
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auc', self.test_auc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)


        
        probs = torch.sigmoid(y_hat)
        print(f"\nBatch test {batch_idx}:")
        print(f"  Targets: {y.squeeze().cpu().numpy()}")
        print(f"  Probabilities: {probs.squeeze().cpu().numpy()}")
        print(f"  Predictions (th=0.5): {(probs > 0.5).float().squeeze().cpu().numpy()}")


        return loss, 
    
    def on_test_epoch_end(self):

        import matplotlib.pyplot as plt
        import torch
        import os 
        log_dir = self.logger.log_dir


        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
       # print(targets.mean())

        plt.figure()
        plt.plot(targets.numpy(), label="True")
        plt.plot(preds.numpy(), label="Predicted")
        plt.legend()
        plt.title("Test predictions vs truth")
        plt.savefig(os.path.join(log_dir, "test_predictions.png"))
        plt.close()

        self.log("test_acc", self.test_acc.compute())
        self.log("test_auc", self.test_auc.compute())


    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,     weight_decay=1e-3 ) #Regularización L2
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
           # verbose=True,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }
