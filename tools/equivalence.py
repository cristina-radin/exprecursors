"""
Bit-for-bit equivalence harness for the committed model (fold0 of
full_gnll_quantile_v2_landfill), on real data and the real trained checkpoint.

  python tools/equivalence.py --generate   # write tools/reference/equivalence_fold0.npz
  python tools/equivalence.py --check      # recompute and compare with np.array_equal

Environment: MHW_DATA_FILE, MHW_CLIM_FILE (real files) and
MHW_REFERENCE_CKPT_DIR, a directory with exactly one *.ckpt (the best fold-0
checkpoint) plus its model_config.json and resolved_config.yaml.

Fixed seed, torch.use_deterministic_algorithms(True), CPU, 1 thread. It builds
the model from fold0.yaml, checks that this equals the checkpoint's
model_config.json key by key, loads the weights with strict=True, runs the real
LazyDataModule.setup() (real split and normalisation statistics), takes 4 fixed
test samples and stores the model outputs (mean, log_var, q_pred) and the
training-step loss, plus the inputs that produced them (hash of the input
tensor, normalisation constants, sample indices, checkpoint hash) so that a
mismatch can be attributed to data, weights or model code.

It only uses code paths that survive the planned simplifications: constructor
arguments are filtered by signature, and the values the removed options had in
the committed model are pinned in EXPECTED.
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
    gaussian_nll=True, temporal_features=0, state_feature=False
)
# model_config.json keys that older runs did not write; the value the current
# build uses must equal the value that was implicit at the time.
IMPLICIT_IN_OLD_MODEL_CONFIG = {"state_feature": False}
# yaml keys that legitimately differ between fold0.yaml and a run's
# resolved_config.yaml (locations moved to environment variables).
LOCATION_KEYS = {"data_dir", "output_dir", "run_name"}
# Options removed from fold0.yaml because they had no effect on the committed
# model (dead/no-op code paths at the time this checkpoint was trained) -- see
# docs/open_issues.md for why each one was inert. A key here may be absent
# from the current fold0.yaml only if the checkpoint's own
# resolved_config.yaml has exactly this value; any other value, or the key
# still present with a different value, is an error like any other mismatch.
REMOVED_INERT_KEYS = {
    # CNNLSTMModel no longer takes an arch argument: the attention_only and
    # lstm_attention branches were removed, lstm_only (what this checkpoint
    # was built with) is now the only behaviour, hardcoded.
    "arch": "lstm_only",
    # dataset.py's _compute_clim() returned immediately (vars_to_anom = [])
    # because every variable in merged_daily.nc already arrives anomalised --
    # clim_ref_start only bounded the reference period for that dead branch.
    "clim_ref_start": 1985,
    # see clim_ref_start: same dead branch.
    "clim_ref_end": 2014,
    # half-window size for the same dead climatology computation.
    "clim_window": 5,
    # datamodule.py's stratified_kfold branch never read train_ratio -- the
    # split is fixed by the n_folds-bucket rotation; these three values were
    # only read to satisfy an assert that they sum to 1.0 (datamodule.py
    # itself printed a NOTE saying it ignored them for this split_mode).
    "train_ratio": 0.75,
    # see train_ratio.
    "val_ratio": 0.15,
    # see train_ratio.
    "test_ratio": 0.1,
}


def _filter(callable_, kwargs):
    params = inspect.signature(callable_).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def model_kwargs(cfg):
    for key, value in EXPECTED.items():
        if key in cfg and cfg[key] != value:
            raise ValueError(f"{key}={cfg[key]!r} in yaml, expected {value!r}")
    return dict(
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


def build_model(kwargs):
    from src.models.cnn_lstm import CNNLSTMModel

    return CNNLSTMModel(**_filter(CNNLSTMModel.__init__, kwargs))


def load_reference_checkpoint(cfg):
    """Return (ckpt path, sha256, checkpoint dict) after verifying that the
    checkpoint's model_config.json and resolved_config.yaml agree with
    fold0.yaml. Raises on any difference."""
    import json

    var = "MHW_REFERENCE_CKPT_DIR"
    if not os.environ.get(var):
        raise EnvironmentError(f"{var} must point at the reference checkpoint dir")
    ckpt_dir = Path(os.environ[var])
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if len(ckpts) != 1:
        raise ValueError(f"expected exactly one .ckpt in {ckpt_dir}, found {ckpts}")

    saved = json.load(open(ckpt_dir / "model_config.json"))
    built = model_kwargs(cfg)
    problems = []
    for key in sorted(set(saved) | set(built)):
        if key not in built:
            if key in REMOVED_INERT_KEYS and saved[key] == REMOVED_INERT_KEYS[key]:
                continue
            problems.append(f"model_config.json has {key}={saved[key]!r}, not built")
        elif key not in saved:
            if key not in IMPLICIT_IN_OLD_MODEL_CONFIG:
                problems.append(f"{key}={built[key]!r} built, missing in model_config.json")
            elif built[key] != IMPLICIT_IN_OLD_MODEL_CONFIG[key]:
                problems.append(
                    f"{key}={built[key]!r} built, implicit value was "
                    f"{IMPLICIT_IN_OLD_MODEL_CONFIG[key]!r}"
                )
        elif saved[key] != built[key]:
            problems.append(f"{key}: model_config.json={saved[key]!r} built={built[key]!r}")

    resolved = yaml.safe_load(open(ckpt_dir / "resolved_config.yaml"))
    for key in sorted((set(resolved) | set(cfg)) - LOCATION_KEYS):
        if key not in cfg:
            if key in REMOVED_INERT_KEYS and resolved.get(key) == REMOVED_INERT_KEYS[key]:
                continue
            problems.append(f"resolved_config.yaml has {key}={resolved[key]!r}, fold0.yaml does not")
        elif key not in resolved:
            problems.append(f"fold0.yaml has {key}={cfg[key]!r}, resolved_config.yaml does not")
        elif resolved[key] != cfg[key]:
            problems.append(f"{key}: resolved={resolved[key]!r} fold0.yaml={cfg[key]!r}")
    if problems:
        raise ValueError("checkpoint does not match fold0.yaml:\n  " + "\n  ".join(problems))

    path = ckpts[0]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha, torch.load(path, map_location="cpu", weights_only=False)


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


def compute(dm, cfg, ckpt_sha, ckpt):
    """One deterministic pass. Returns dict of numpy arrays."""
    pl.seed_everything(SEED, workers=True)
    model = build_model(model_kwargs(cfg))
    module = build_module(cfg, model, dm)
    # strict=True: every key must match (known_issues #16, no strict=False)
    module.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    module.eval()
    hp = ckpt["hyper_parameters"]
    for name, now in (("target_mean", dm.target_mean), ("target_std", dm.target_std)):
        if not np.isclose(hp[name], now, rtol=1e-6, atol=0):
            raise ValueError(
                f"{name}: checkpoint was trained with {hp[name]!r}, the data now "
                f"gives {now!r} (different data file or split?)"
            )

    subset = dm.test_dataset
    batch = default_collate([subset[p] for p in SAMPLE_POSITIONS])
    xs = batch[0]
    fwd_args = (xs, batch[1]) if "x_temporal" in inspect.signature(model.forward).parameters else (xs,)

    with torch.no_grad():
        y_hat, q_pred = model.forward_with_quantile(*fwd_args)
        loss, pred, y = module._step(batch, "val")

    full_ds = subset.dataset
    return {
        "ckpt_sha256": np.array(ckpt_sha),
        "ckpt_epoch": np.array(ckpt["epoch"]),
        "ckpt_target_mean": np.array(ckpt["hyper_parameters"]["target_mean"]),
        "ckpt_target_std": np.array(ckpt["hyper_parameters"]["target_std"]),
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

    _, ckpt_sha, ckpt = load_reference_checkpoint(cfg)
    dm = setup_datamodule()
    first = compute(dm, cfg, ckpt_sha, ckpt)
    second = compute(dm, cfg, ckpt_sha, ckpt)
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
        print(f"checkpoint {ckpt_sha[:16]}... epoch {int(first['ckpt_epoch'])}")
        for k in ("mean", "log_var", "q_pred"):
            v = first[k].astype(np.float64)
            print(
                f"  {k}: {first[k]}  mean={v.mean():.6f} std={v.std():.6f} "
                f"min={v.min():.6f} max={v.max():.6f} range={v.max() - v.min():.6f}"
            )
        print(f"  loss: {first['loss']}")
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
