"""Checkpoint selection utilities."""

import inspect
import json
import re
from pathlib import Path

from src.models.cnn_lstm import CNNLSTMModel

# Exact CNNLSTMModel.__init__ keyword arguments, read from the constructor's
# own signature so this list can never drift from what CNNLSTMModel takes.
CNNLSTM_MODEL_KEYS = tuple(
    name
    for name in inspect.signature(CNNLSTMModel.__init__).parameters
    if name != "self"
)


def save_model_config(output_dir: Path, **kwargs) -> None:
    """Write the exact CNNLSTMModel kwargs used for this run to
    output_dir/model_config.json, so the architecture can always be
    reconstructed exactly as trained, instead of each eval/XAI script
    independently re-deriving it. Call with the same kwargs dict used to
    build the model (e.g. `CNNLSTMModel(**model_kwargs)` then
    `save_model_config(output_dir, **model_kwargs)`).
    """
    missing = [k for k in CNNLSTM_MODEL_KEYS if k not in kwargs]
    if missing:
        raise ValueError(f"save_model_config missing required keys: {missing}")
    extra = [k for k in kwargs if k not in CNNLSTM_MODEL_KEYS]
    if extra:
        raise ValueError(f"save_model_config got unexpected keys: {extra}")
    path = Path(output_dir) / "model_config.json"
    with open(path, "w") as f:
        json.dump({k: kwargs[k] for k in CNNLSTM_MODEL_KEYS}, f, indent=2)


def load_model_config(run_dir: Path) -> dict:
    """Return the exact CNNLSTMModel kwargs for the run in run_dir, read
    from run_dir/model_config.json (written by save_model_config at train
    time). Raises if the file is missing a key CNNLSTMModel needs, or has
    an extra one — no defaults are filled in.
    """
    path = Path(run_dir) / "model_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    with open(path) as f:
        saved = json.load(f)
    missing = [k for k in CNNLSTM_MODEL_KEYS if k not in saved]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    extra = [k for k in saved if k not in CNNLSTM_MODEL_KEYS]
    if extra:
        raise ValueError(f"{path} has unexpected keys: {extra}")
    return saved


def best_ckpt(ckpt_dir: Path) -> Path:
    """Return the checkpoint with the lowest val_loss in ckpt_dir. Raises if
    ckpt_dir mixes checkpoints from more than one training run (any
    "-vN.ckpt" file -- PyTorch Lightning adds that suffix instead of
    overwriting when a filename already exists) or has no checkpoints at
    all.
    """
    ckpt_dir = Path(ckpt_dir)
    duplicates = [
        c for c in ckpt_dir.glob("*.ckpt") if re.search(r"-v\d+\.ckpt$", c.name)
    ]
    if duplicates:
        raise ValueError(
            f"{ckpt_dir} contains checkpoints from more than one training "
            f"run: {[c.name for c in duplicates]}. Use a clean output_dir."
        )

    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    def _val_loss(c):
        m = re.search(r"val_loss=([-\d.]+?)\.ckpt", c.name)
        if not m:
            raise ValueError(
                f"Could not parse val_loss from checkpoint filename {c.name!r}."
            )
        return float(m.group(1).rstrip("."))

    return min(ckpts, key=_val_loss)
