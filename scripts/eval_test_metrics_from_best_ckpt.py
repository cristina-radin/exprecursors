"""
eval_test_metrics_from_best_ckpt.py — Aug 21 2026, needed to answer a
direct user question honestly: "did the land_fill fold0 retrain actually
improve the loss?"

The committed full_gnll_quantile_v2 fold0 run (job 29417405, Aug 20
15:29-19:41) predates the ckpt_path="best" fix in train_partition.py
(commit 33ebb70, Aug 21 00:34) -- confirmed via git log, not assumed.
known_issues.md #46 names this exact fold0 comparison as one of the
runs affected by the bug: trainer.test() evaluated whatever epoch
EarlyStopping happened to stop at (a GaussianNLL variance-collapse
epoch), not the actual best checkpoint (epoch=13, val_loss=-0.0245,
still on disk). So the committed fold0's previously-logged test_loss=
3.08/test_corr=0.8062 are NOT a fair baseline for the land_fill run
(which correctly used ckpt_path="best" since it ran after the fix).

This script re-evaluates any already-trained fold's checkpoint (the
actual best one, loaded explicitly, same as eval_event_detection.py's
pattern) via trainer.test(), so the two land_fill-vs-committed numbers
are computed identically and can be compared fairly. No training, no
GPU required for a single fold's ~2800 test samples.

Usage:
  python scripts/eval_test_metrics_from_best_ckpt.py --config configs/partition/full_gnll_quantile_v2/fold0.yaml
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    print(f"device={device}", flush=True)
    print(f"config={args.config}", flush=True)

    dm = LazyDataModule(args.config)
    dm.setup()

    run_dir = Path(cfg["output_dir"])
    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    print(f"Loading best checkpoint: {ckpt.name}", flush=True)
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=True, map_location=device
    )

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    trainer.test(lm, datamodule=dm)


if __name__ == "__main__":
    main()
