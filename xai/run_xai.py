#!/usr/bin/env python

import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from dataset import LazyDataset
from datamodule import LazyDataModule
from model import CNNModel, CNNLightningModule  # Importa ambos
from xai.utils import load_config  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_dir', type=str, default='xai_results')
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()
    config = load_config(args.config)

    print(f"Args: {args}")
    
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    
    print(f"\nCheckpoint: {args.checkpoint}")
    
    physical_vars = [v for v in config['variables'] if v not in ['target', 'land_mask']]
    n_physical = len(physical_vars)
    has_land = 'land_mask' in config['variables']
    n_temporal = 3 if config.get('add_temporal_features', False) else 0

    in_channels = n_physical + (1 if has_land else 0) + n_temporal


    # CNNModel 
    cnn_model = CNNModel(
        in_channels=in_channels, 
        out_channels=1
    )
    
    #  chekpoint CNNLightningModule
    model = CNNLightningModule.load_from_checkpoint(
        args.checkpoint,
        model=cnn_model, 
        map_location=device
    )
    model.eval()
    model.to(device)
    
    dm = LazyDataModule(config_path=args.config)
    dm.setup()
    
    print(f"\nVariables: {config['variables']}")
    print(f"Channels: {in_channels}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)


    print("\nSHAP...")
    try:
        from xai.shap_analysis import analyze_shap
        shap_results = analyze_shap(
            model,
            dm.train_dataloader(),
            dm.test_dataloader(),
            args.output_dir,
            config_path=args.config
        )
        print("SHAP completed ")
    except Exception as e:
        print(f"SHAP failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nGrad-CAM...")
    try:
        from xai.grad_cam import analyze_gradcam
        gradcam_results = analyze_gradcam(
            model,
            dm.test_dataset,
            args.output_dir,
            num_samples=5,
            config_path=args.config
        )
        print("Grad-CAM completed")
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nXAI analysis completed. Results in: {args.output_dir}\n")
    
        
if __name__ == '__main__':
    main()