# Climate Model Deep Learning Project

A PyTorch Lightning-based skeleton for deep learning with climate model data.

## Project Structure

```
.
├── dataset.py          # Dataset classes for loading climate data
├── datamodule.py       # PyTorch Lightning DataModule
├── model.py            # Neural network model definitions
├── train.py            # Training script (CLI)
├── train_example.ipynb # Jupyter notebook with training example
├── requirements.txt    # Python dependencies
└── README.md           # This file
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

### 1. Implement Data Loading

Edit `dataset.py`


### 2. Run Training

**Using the Jupyter notebook:**
check out `train_example.ipynb`

**Using the Command Line:**
```bash
python train.py --data_dir /path/to/your/data --in_channels 10 --out_channels 1
```


