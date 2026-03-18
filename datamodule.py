"""
PyTorch Lightning DataModule for climate data.
"""

from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from dataset import LazyDataset
import yaml



class LazyDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for climate data.
    
    Args:
        train_files: Path or pattern to training data files.
        val_files: Path or pattern to validation data files. If none, will select randomly from train_files
        test_files: Path or pattern to test data files.
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        train_val_test_split: Tuple of (train, val, test) fractions
        seed: Random seed for splitting
    """
    
    def __init__(
        self,
                config_path: str = "/p/project1/hai_1127/radin1/exprecursors/config.yaml"  # ← ÚNICO parámetro
            ):
        super().__init__()
        self.save_hyperparameters()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.train_files = self.config['data_dir']
            self.val_files = self.config.get('val_files')  # Opcional
            self.test_files = self.config.get('test_files')  # Opcional
            self.batch_size = self.config['batch_size']
            self.num_workers = self.config['num_workers']
            self.seed = self.config['seed']
            self.variables = self.config['variables']
            self.normalize = self.config['normalize']
            self.lazy_loading = self.config.get('lazy_loading', False)
            self.add_temporal_features = self.config['add_temporal_features']
            
            self.train_ratio = self.config['train_ratio']
            self.val_ratio = self.config['val_ratio']
            self.test_ratio = self.config['test_ratio']
            
            total = self.train_ratio + self.val_ratio + self.test_ratio
            assert abs(total - 1.0) < 1e-6, f"Sum ratios {total}"

            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            self.norm_stats = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for each stage (fit, validate, test, predict).
        Currently, we load the full dataset and split it into train/val/test.
        You could also provide separate datasets for each stage, but needs to be implemented.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """

        self.stage = stage

        if self.val_files is not None: #separate files for train/val/test
            self.train_dataset = LazyDataset(
                    self.train_files, 
                    variables=self.variables,
                    lazy_loading=self.lazy_loading,
                    normalize=self.normalize,
                    add_temporal_features=self.add_temporal_features
                )
            self.val_dataset = LazyDataset(
                    self.val_files, 
                    variables=self.variables,
                    lazy_loading=self.lazy_loading,
                    normalize=self.normalize,
                    add_temporal_features=self.add_temporal_features
                )
            if self.test_files is not None:
                self.test_dataset = LazyDataset(
                    self.test_files, 
                    variables=self.variables,
                    lazy_loading=self.lazy_loading,
                    normalize=self.normalize,
                    add_temporal_features=self.add_temporal_features
                )    

            if self.normalize:
                if not hasattr(self.train_dataset, 'input_means'):
                    self.train_dataset._compute_stats()
                
                for dataset in [self.val_dataset, self.test_dataset]:
                    if dataset is not None:
                        dataset.input_means = self.train_dataset.input_means
                        dataset.input_stds = self.train_dataset.input_stds
                        dataset.normalize = True

                    
        else : #current case, just 1 file for train/val/test # SPLIT TEMPORAL
              
            full_ds = LazyDataset(
            self.train_files, 
            variables=self.variables,
            lazy_loading=self.lazy_loading,
            normalize=self.normalize,
            add_temporal_features=self.add_temporal_features
            )
        
            if self.normalize and not hasattr(full_ds, 'input_means'):
                full_ds._compute_stats()
            
            # Calculate split sizes 
            total_size = len(full_ds)
            train_size = int(self.train_ratio * total_size)
            val_size = int(self.val_ratio * total_size)
            test_size = total_size - train_size - val_size

            #temporal indexes for splitting
            train_indices = list(range(0, train_size))
            val_indices = list(range(train_size, train_size + val_size))
            test_indices = list(range(train_size + val_size, total_size))

            #testing that the splits are disjoint
            assert len(set(train_indices) & set(val_indices)) == 0
            assert len(set(train_indices) & set(test_indices)) == 0
            assert len(set(val_indices) & set(test_indices)) == 0

            #subsets with temporal splits
            self.train_dataset = Subset(full_ds, train_indices)
            self.val_dataset = Subset(full_ds, val_indices)
            self.test_dataset = Subset(full_ds, test_indices)


            print(f"\n SPLIT TEMPORAL:")
            print(f"  Total samples: {total_size}")
            print(f"  Train (past): {train_size} samples (indexes 0-{train_size-1})")
            print(f"  Val (intermediate): {val_size} samples (indexes {train_size}-{train_size+val_size-1})")
            print(f"  Test (future): {test_size} samples (indexes {train_size+val_size}-{total_size-1})")
            print(f"  Percentages: train={train_size/total_size:.1%}, val={val_size/total_size:.1%}, test={test_size/total_size:.1%}\n")
            

            
            if self.normalize and hasattr(full_ds, 'input_means'):
                for subset in [self.train_dataset, self.val_dataset, self.test_dataset]:
                    subset.dataset.input_means = full_ds.input_means
                    subset.dataset.input_stds = full_ds.input_stds

        if self.test_files is not None:
            self.test_dataset = LazyDataset(
                self.test_files, 
                variables=self.variables,
                lazy_loading=self.lazy_loading,
                normalize=self.normalize,
                add_temporal_features=self.add_temporal_features

            )
        elif self.test_dataset is None and stage == 'test':
            raise ValueError("Test files must be provided for test stage.")
        

    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )



