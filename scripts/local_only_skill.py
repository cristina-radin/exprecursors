"""
local_only_skill.py — Partition of NS MHW predictability: local vs remote.

Three conditions evaluated on the SAME trained SSTAtm full model:
  full      : original input (all domain)         → should reproduce r≈0.865
  local_only: all vars zeroed OUTSIDE NS box       → isolates local signal
  baseline  : zero input everywhere (all zeros)   → model bias check

NS bounding box (same as mask_ns_sst in dataset.py):
  lat: slice(100, 127)  ≈ 50–63°N
  lon: slice(150, 187)  ≈ -5–13°E

Uses all 25 SSTAtm_lstmonly multiseed runs (5 seeds × 5 folds) for consistency
with the masked-model r=0.807 result. Reports mean ± std across runs.

Output:
  local_only_skill.txt  — table of r values per condition
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.paths import (
    EXPERIMENTS_DIR,
)

MULTISEED_DIR = Path(str(EXPERIMENTS_DIR))
SEEDS = [42, 123, 456, 789, 1337]
N_FOLDS = 5

# NS box — same as dataset.py
NS_LAT = slice(100, 127)
NS_LON = slice(150, 187)


def best_ckpt(run_dir: Path) -> Path:
    ckpts = list((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/checkpoints/")

    def val_loss(p):
        m = re.search(r"val_loss=(-?[0-9]+\.[0-9]+)", str(p))
        return float(m.group(1)) if m else 0.0

    return min(ckpts, key=val_loss)


def load_model(ckpt_path: Path, cfg: dict, device: str) -> CNNLightningModule:
    inner = CNNLSTMModel(
        in_channels=cfg["in_channels"],
        cnn_features=cfg.get("cnn_features", 256),
        lstm_hidden=cfg.get("lstm_hidden", 512),
        lstm_layers=cfg.get("lstm_layers", 4),
        temporal_features=cfg.get("temporal_features", 0),
        dropout=cfg.get("dropout", 0.3),
        arch=cfg.get("arch", "lstm_only"),
        gaussian_nll=cfg.get("gaussian_nll", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path),
        model=inner,
        strict=False,
        map_location=device,
    )
    return lm.eval().to(device)


def run_conditions(lm, test_ds, device, batch_size=32):
    """
    Run inference under 3 input conditions.
    Returns preds_full, preds_local, targets — all (N,) numpy arrays.
    """
    base_ds = test_ds.dataset if hasattr(test_ds, "dataset") else test_ds
    indices = list(
        test_ds.indices if hasattr(test_ds, "indices") else range(len(test_ds))
    )

    preds_full = []
    preds_local = []
    targets = []

    with torch.no_grad():
        for b_start in range(0, len(indices), batch_size):
            batch_idx = indices[b_start : b_start + batch_size]

            xs_list, xt_list, y_list = [], [], []
            for gi in batch_idx:
                xs, xt, y = base_ds[gi]
                xs_list.append(xs)
                xt_list.append(xt)
                y_list.append(y)

            xs_b = torch.stack(xs_list).to(device)  # (B, T, C, H, W)
            xt_b = torch.stack(xt_list).to(device)

            # Full input
            p_full, _ = lm.model.forward_with_attention(xs_b.float(), xt_b.float())
            preds_full.extend(p_full[:, 0].cpu().numpy().tolist())

            # Local only: zero everything outside NS box
            xs_local = torch.zeros_like(xs_b)
            xs_local[:, :, :, NS_LAT, NS_LON] = xs_b[:, :, :, NS_LAT, NS_LON]
            p_local, _ = lm.model.forward_with_attention(xs_local.float(), xt_b.float())
            preds_local.extend(p_local[:, 0].cpu().numpy().tolist())

            targets.extend([y.item() for y in y_list])

    return np.array(preds_full), np.array(preds_local), np.array(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/figures/xai_masked")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  batch_size={args.batch_size}")
    print(f"NS box: lat[{NS_LAT}] lon[{NS_LON}]")

    r_full_runs = []
    r_local_runs = []

    run_count = 0
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            run_name = f"SSTAtm_lstmonly_seed{seed}_fold{fold}"
            run_dir = MULTISEED_DIR / run_name
            cfg_path = run_dir / "config.yaml"
            if not cfg_path.exists():
                print(f"SKIP {run_name}: no config")
                continue
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            try:
                ckpt = best_ckpt(run_dir)
            except FileNotFoundError:
                print(f"SKIP {run_name}: no checkpoint")
                continue

            run_count += 1
            print(f"\n[{run_count}/25] {run_name}")
            lm = load_model(ckpt, cfg, device)

            dm = LazyDataModule(config_path=str(cfg_path))
            dm.setup()
            test_ds = dm.test_dataloader().dataset

            p_full, p_local, tgts = run_conditions(lm, test_ds, device, args.batch_size)

            r_full, _ = pearsonr(p_full, tgts)
            r_local, _ = pearsonr(p_local, tgts)

            r_full_runs.append(r_full)
            r_local_runs.append(r_local)
            print(f"  r_full={r_full:.4f}  r_local_only={r_local:.4f}")

    # Summary
    r_full_arr = np.array(r_full_runs)
    r_local_arr = np.array(r_local_runs)

    lines = [
        "=== Local vs Remote Predictability Partition ===",
        f"n_runs = {len(r_full_arr)}",
        "",
        "Full input (should ≈ 0.865):",
        f"  mean r = {r_full_arr.mean():.4f}  std = {r_full_arr.std():.4f}",
        f"  min={r_full_arr.min():.4f}  max={r_full_arr.max():.4f}",
        "",
        "Local only (NS box, all other vars zeroed):",
        f"  mean r = {r_local_arr.mean():.4f}  std = {r_local_arr.std():.4f}",
        f"  min={r_local_arr.min():.4f}  max={r_local_arr.max():.4f}",
        "",
        "Remote (NS SST zeroed, separately trained masked model): r ≈ 0.807",
        "",
        "Partition summary:",
        f"  Full         r = {r_full_arr.mean():.3f}",
        f"  Remote only  r = 0.807  ({0.807/r_full_arr.mean()*100:.0f}% of full)",
        f"  Local only   r = {r_local_arr.mean():.3f}  ({r_local_arr.mean()/r_full_arr.mean()*100:.0f}% of full)",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    out_file = out_dir / "local_only_skill.txt"
    with open(str(out_file), "w") as f:
        f.write(report + "\n")
    print(f"\nSaved {out_file}")


if __name__ == "__main__":
    main()
