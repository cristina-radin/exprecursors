"""Checkpoint selection utilities."""

import json
import re
from pathlib import Path

# Exact CNNLSTMModel.__init__ kwargs (minus in_channels, required separately).
# Defaults here MUST match CNNLSTMModel's own defaults — this is the one
# place both save_model_config's fallback and every legacy caller derive
# them from, instead of each eval/XAI script re-guessing its own.
CNNLSTM_DEFAULTS = {
    "cnn_features": 128,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "temporal_features": 3,
    "dropout": 0.3,
    "arch": "lstm_only",
    "gaussian_nll": False,
    "pooling": "max",
    "quantile_head": False,
}
CNNLSTM_MODEL_KEYS = ("in_channels",) + tuple(CNNLSTM_DEFAULTS)


def save_model_config(output_dir: Path, **kwargs) -> None:
    """Write the exact resolved CNNLSTMModel kwargs used for this run to
    output_dir/model_config.json — the single source of truth for
    reconstructing this checkpoint's architecture.

    Call with the same dict used to build the model (e.g.
    `CNNLSTMModel(**model_kwargs)` then `save_model_config(output_dir,
    **model_kwargs)`) so the recorded config can never drift from what was
    actually trained. Every eval/XAI script should read it back via
    load_model_config() instead of independently calling
    `cfg.get("quantile_head", False)` etc. — that per-script duplication is
    what caused the quantile_head silent-weight-drop bug (docs/narrative.md,
    Aug 19 2026): nine scripts each guessed their own defaults and none of
    them matched what train_partition.py actually built.
    """
    missing = [k for k in CNNLSTM_MODEL_KEYS if k not in kwargs]
    if missing:
        raise ValueError(f"save_model_config missing required keys: {missing}")
    path = Path(output_dir) / "model_config.json"
    with open(path, "w") as f:
        json.dump({k: kwargs[k] for k in CNNLSTM_MODEL_KEYS}, f, indent=2)


def load_model_config(run_dir: Path, fallback_cfg: dict = None) -> dict:
    """Return the exact CNNLSTMModel kwargs for the run in run_dir.

    Prefers run_dir/model_config.json (written by save_model_config at train
    time — the ground truth). Falls back to deriving kwargs from
    fallback_cfg (a raw fold{N}.yaml dict) using CNNLSTM_DEFAULTS, for runs
    that predate model_config.json. Pass fallback_cfg=None to require the
    file to exist (e.g. once all active runs have one) instead of silently
    re-deriving defaults.
    """
    path = Path(run_dir) / "model_config.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    if fallback_cfg is None:
        raise FileNotFoundError(
            f"{path} not found and no fallback_cfg given — this run predates "
            "model_config.json and there is no config to derive defaults from."
        )
    resolved = {"in_channels": fallback_cfg["in_channels"]}
    for key, default in CNNLSTM_DEFAULTS.items():
        resolved[key] = fallback_cfg.get(key, default)
    return resolved


def best_ckpt(ckpt_dir: Path) -> Path:
    """Return the checkpoint with lowest val_loss, skipping -v1/-v2 duplicates."""
    ckpts = [
        c
        for c in Path(ckpt_dir).glob("*.ckpt")
        if not re.search(r"-v\d+\.ckpt$", str(c))
    ]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    def _val_loss(c):
        m = re.search(r"val_loss=([-\d.]+?)\.ckpt", c.name)
        return float(m.group(1).rstrip(".")) if m else float("inf")

    return min(ckpts, key=_val_loss)
