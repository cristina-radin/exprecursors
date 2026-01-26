"""
Model definitions for climate data prediction.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl


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
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 64]
        
        # TODO: Define your model architecture here
        # Example: Simple MLP
        layers = []
        prev_dim = in_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, out_channels))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, ...)
            
        Returns:
            Output tensor
        """
        # TODO: Adjust based on your data shape
        # Flatten if needed for MLP
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        return self.network(x)


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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        x, y = batch
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
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('test_loss', loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }
