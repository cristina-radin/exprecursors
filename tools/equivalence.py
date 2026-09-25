"""
Bit-for-bit equivalence harness for the committed model (fold0 of
full_gnll_quantile_v2_landfill), on real data.

  python tools/equivalence.py --generate   # write tools/reference/equivalence_fold0.npz
  python tools/equivalence.py --check      # recompute and compare with np.array_equal

Fixed seed, torch.use_deterministic_algorithms(True), CPU, 1 thread. It builds
the model from fold0.yaml, runs the real LazyDataModule.setup() (real split and
normalisation statistics), takes 4 fixed test samples and stores the model
outputs (mean, log_var, q_pred) and the training-step loss, plus the inputs
that produced them (hash of the input tensor, normalisation constants, sample
indices) so that a mismatch can be attributed to data or to model code.

It only uses code paths that survive the planned simplifications: constructor
arguments are filtered by signature, and the values the removed options had in
the committed model are pinned in EXPECTED. Needs MHW_DATA_FILE and
MHW_CLIM_FILE to point at the real files.
"""

import argparse
import hashlib
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFIG = REPO / "configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml"
REFERENCE = REPO / "tools/reference/equivalence_fold0.npz"
SAMPLE_POSITIONS = (0, 731, 1462, 2193)  # positions inside the fold-0 test set
SEED = 42
# Values of the options that later simplifications remove, as used by the
# committed model. If the yaml still carries them they must match.
EXPECTED = dict(
    arch="lstm_only", gaussian_nll=True, temporal_features=0, state_feature=False
)


def _filter(callable_, kwargs):
    params = inspect.signature(callable_).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def build_model(cfg):
    from src.models.cnn_lstm import CNNLSTMModel

    for key, value in EXPECTED.items():
        if key in cfg and cfg[key] != value:
            raise ValueError(f"{key}={cfg[key]!r} in yaml, expected {value!r}")
    kwargs = dict(
        in_channels=cfg["in_channels"],
        cnn_features=cfg["cnn_features"],
        lstm_hidden=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
        pooling=cfg["pooling"],
        padding_mode=cfg["padding_mode"],
        quantile_head=cfg["quantile_head"],
        **EXPECTED,
    )
    return CNNLSTMModel(**_filter(CNNLSTMModel.__init__, kwargs))


def build_module(cfg, model, dm):
    from src.models.cnn_lstm import CNNLightningModule

    kwargs = dict(
        model=model,
        learning_rate=cfg["learning_rate"],
        target_mean=dm.target_mean,
        target_std=dm.target_std,
        loss_fn="GaussianNLLLoss",
        gaussian_nll=True,
        quantile_head=cfg["quantile_head"],
        quantile_tau=cfg["quantile_tau"],
        quantile_weight=cfg["quantile_weight"],
        lr_scheduler=cfg["lr_scheduler"],
        warmup_epochs=cfg["warmup_epochs"],
        cosine_t_max_epochs=cfg["cosine_t_max_epochs"],
    )
    return CNNLightningModule(**_filter(CNNLightningModule.__init__, kwargs))


def setup_datamodule():
    from src.data.datamodule import LazyDataModule

    dm = LazyDataModule(config_path=str(CONFIG))
    dm.setup()
    return dm


def compute(dm, cfg):
    """One deterministic pass. Returns dict of numpy arrays."""
    pl.seed_everything(SEED, workers=True)
    model = build_model(cfg)
    module = build_module(cfg, model, dm)
    model.eval()
    module.eval()

    subset = dm.test_dataset
    batch = default_collate([subset[p] for p in SAMPLE_POSITIONS])
    xs = batch[0]
    fwd_args = (xs, batch[1]) if "x_temporal" in inspect.signature(model.forward).parameters else (xs,)

    with torch.no_grad():
        y_hat, q_pred = model.forward_with_quantile(*fwd_args)
        loss, pred, y = module._step(batch, "val")

    full_ds = subset.dataset
    return {
        "mean": y_hat[:, 0].numpy().astype(np.float32),
        "log_var": y_hat[:, 1].numpy().astype(np.float32),
        "q_pred": q_pred[:, 0].numpy().astype(np.float32),
        "loss": np.array(loss.item(), dtype=np.float32),
        "y": y.numpy().astype(np.float32),
        "x_sha256": np.array(hashlib.sha256(xs.numpy().tobytes()).hexdigest()),
        "input_means": full_ds.input_means.numpy(),
        "input_stds": full_ds.input_stds.numpy(),
        "target_mean": np.array(dm.target_mean, dtype=np.float64),
        "target_std": np.array(dm.target_std, dtype=np.float64),
        "sample_indices": np.array([subset.indices[p] for p in SAMPLE_POSITIONS]),
        "split_sizes": np.array(
            [len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset)]
        ),
    }


def diff(a, b):
    lines = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            lines.append(f"  {key}: only in {'reference' if key in b else 'new'}")
        elif not np.array_equal(a[key], b[key]):
            extra = ""
            if a[key].dtype.kind == "f":
                extra = f" max|diff|={np.max(np.abs(a[key] - b[key])):.3e}"
            lines.append(f"  {key}: DIFFERENT{extra}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite the reference")
    args = ap.parse_args()

    for var in ("MHW_DATA_FILE", "MHW_CLIM_FILE"):
        if not os.environ.get(var):
            raise EnvironmentError(f"{var} must point at the real data file")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    cfg = yaml.safe_load(open(CONFIG))

    dm = setup_datamodule()
    first = compute(dm, cfg)
    second = compute(dm, cfg)
    problems = diff(first, second)
    if problems:
        print("NOT DETERMINISTIC: two passes in the same process differ:")
        print("\n".join(problems))
        sys.exit(2)

    if args.generate:
        if REFERENCE.exists() and not args.force:
            raise FileExistsError(f"{REFERENCE} exists; use --force to overwrite")
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(REFERENCE, **first)
        print(f"reference written: {REFERENCE}")
        for k in ("mean", "log_var", "q_pred", "loss"):
            print(f"  {k}: {first[k]}")
        return

    ref = dict(np.load(REFERENCE))
    problems = diff(first, ref)
    if problems:
        print("MISMATCH against reference:")
        print("\n".join(problems))
        sys.exit(1)
    print("OK: bit-for-bit identical to the reference")


if __name__ == "__main__":
    main()
