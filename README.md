# Climate Model Deep Learning Project

A PyTorch Lightning-based framework for deep learning with climate model data, with integrated Explainable AI (XAI) capabilities.

## Project Structure

```
.
├── dataset.py          # Dataset classes for loading data
├── datamodule.py       # PyTorch Lightning DataModule
├── model.py            # Neural network model definitions
├── train.py            # Training script (CLI)
├── config.yaml         # Configuration file
├── requirements.txt    # Python dependencies
 README.md              # This file
└── xai/               
  ├── grad_cam.py       # Grad-CAM implementation
  ├── shap_analysis.py  # SHAP analysis tools    
  ├── run_xai.py        # main script
  └── utils.py          # import configurations

```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Inputs: "/p/project1/hai_1127/inputs/processed_data/training_dataset1.nc"

### Run Training and XAI

```bash
    srun -A hai_1127 \
        --nodes=1 \
        --ntasks=1 \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        python train.py
```el proyecto que estoy haciendo? 


Select a checkpoint from output folder and run: 

```bash
        python xai/run_xai.py --checkpoint  /p/project1/hai_1127/radin1/exprecursors/outputs/checkpoints/cnn-epoch=39-val_loss=0.7477.ckpt --config config.yaml
```

