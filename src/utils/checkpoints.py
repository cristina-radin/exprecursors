"""Checkpoint selection utilities."""

import re
from pathlib import Path


def best_ckpt(ckpt_dir: Path) -> Path:
    """Return the checkpoint with lowest val_loss, skipping -v1/-v2 duplicates."""
    ckpts = [c for c in Path(ckpt_dir).glob("*.ckpt")
             if not re.search(r"-v\d+\.ckpt$", str(c))]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    def _val_loss(c):
        m = re.search(r"val_loss=([-\d.]+?)\.ckpt", c.name)
        return float(m.group(1).rstrip(".")) if m else float("inf")

    return min(ckpts, key=_val_loss)
