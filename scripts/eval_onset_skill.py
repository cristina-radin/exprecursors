"""
eval_onset_skill.py — Onset / mid-event / no-MHW skill for partition models.

Uses the correct masking class per mode (RemoteOnlyLightningModule /
LocalOnlyLightningModule / CNNLightningModule) so inference matches training exactly.

Saves per-fold .npz files immediately after each fold finishes.

Usage (CPU, login node):
  python partition/eval_onset_skill.py --mode remote_only
  python partition/eval_onset_skill.py --mode local_only
  python partition/eval_onset_skill.py --mode full
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train_partition import LocalOnlyLightningModule, RemoteOnlyLightningModule
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.paths import (
    CLIM_FILE as CLIM_FILE_ENV,
)
from src.utils.paths import (
    DATA_FILE as DATA_FILE_ENV,
)
from src.utils.paths import (
    EXPERIMENTS_DIR,
)

# ── Constants ────────────────────────────────────────────────────────────────

_NS_LAT = slice(100, 127)
_NS_LON = slice(150, 187)

CLIM_FILE = CLIM_FILE_ENV
DATA_FILE = DATA_FILE_ENV

PARTITION_BASE = EXPERIMENTS_DIR / "partition"
FULL_BASE = EXPERIMENTS_DIR / "kfold"
OUT_DIR = EXPERIMENTS_DIR / "partition" / "onset_skill"

MODULE_MAP = {
    "remote_only": RemoteOnlyLightningModule,
    "local_only": LocalOnlyLightningModule,
    "full": CNNLightningModule,
}

N_FOLDS = 5


# ── Hobday helpers ───────────────────────────────────────────────────────────


def load_ns_p90(smooth_days=31):
    ds = xr.open_dataset(CLIM_FILE)
    p90 = ds.p90_thresh.sel(lat=slice(50.0, 63.0), lon=slice(-5.0, 13.0))
    p90 = p90.mean(dim=["lat", "lon"], skipna=True).values
    ds.close()
    return uniform_filter1d(p90, size=smooth_days, mode="wrap")  # (365,)


def apply_hobday(exceedance, min_dur=5, max_gap=2):
    mhw = exceedance.copy().astype(bool)
    n = len(mhw)
    i = 0
    while i < n:
        if mhw[i]:
            j = i
            while j < n and mhw[j]:
                j += 1
            k = j
            while k < n and not mhw[k]:
                k += 1
            if 0 < (k - j) <= max_gap and k < n:
                mhw[j:k] = True
                i = k
            else:
                i = j
        else:
            i += 1
    i = 0
    while i < n:
        if mhw[i]:
            j = i
            while j < n and mhw[j]:
                j += 1
            if j - i < min_dur:
                mhw[i:j] = False
            i = j
        else:
            i += 1
    return mhw


def mhw_phase_labels(trues, doys, p90_ns):
    """Returns array: 0=no_mhw, 1=onset, 2=mid_event"""
    thr = p90_ns[doys - 1]
    exceed = trues > thr
    mhw = apply_hobday(exceed)
    onset = np.zeros(len(mhw), dtype=bool)
    i = 0
    while i < len(mhw):
        if mhw[i] and (i == 0 or not mhw[i - 1]):
            onset[i] = True
        i += 1
    labels = np.zeros(len(mhw), dtype=int)
    labels[mhw] = 2
    labels[onset] = 1
    return labels


# ── Inference for one fold ───────────────────────────────────────────────────


def run_fold(fold, mode, lm_class, config_path, ckpt_path):
    print(f"\n  fold{fold}: loading checkpoint...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 256),
        lstm_hidden=config.get("lstm_hidden", 512),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=config.get("temporal_features", 0),
        dropout=0.0,
        arch=config.get("arch", "lstm_only"),
        gaussian_nll=config.get("gaussian_nll", False),
        pooling=config.get("pooling", "max"),
        padding_mode=config.get("padding_mode", "zeros"),
        quantile_head=config.get("quantile_head", False),
    )
    lm = lm_class.load_from_checkpoint(ckpt_path, model=cnn_lstm, map_location="cpu")
    lm.eval()

    dm = LazyDataModule(config_path=str(config_path))
    dm.setup()
    test_ds = dm.test_dataset
    full_ds = dm.train_dataset.dataset

    window_size = config.get("window_size", 60)
    lead_time = config.get("lead_time", 7)

    preds, trues, doys = [], [], []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            abs_idx = test_ds.indices[idx]
            xs, xt, y = full_ds[abs_idx]

            xs_b = xs.unsqueeze(0).float()
            if hasattr(lm, "_mask"):
                xs_b = lm._mask(xs_b)

            out = lm.model(xs_b, xt.unsqueeze(0).float())
            pred = out[0, 0].item() if out.shape[-1] == 1 else out[0, 0].item()
            preds.append(pred)
            trues.append(y.item())

            target_abs = abs_idx + window_size - 1 + lead_time
            doy = int(full_ds.doys[target_abs])
            if doy == 366:
                doy = 365
            doys.append(doy)

            if (idx + 1) % 500 == 0:
                print(f"    {idx+1}/{len(test_ds)}", end="\r")
    print(f"    {len(test_ds)}/{len(test_ds)} done")

    return np.array(preds), np.array(trues), np.array(doys, dtype=int)


# ── Skill by phase ───────────────────────────────────────────────────────────


def skill_by_phase(preds, trues, doys, p90_ns):
    labels = mhw_phase_labels(trues, doys, p90_ns)

    results = {}
    for phase_name, mask in [
        ("all", np.ones(len(labels), dtype=bool)),
        ("onset", labels == 1),
        ("mid_event", labels == 2),
        ("no_mhw", labels == 0),
    ]:
        if mask.sum() < 10:
            results[phase_name] = {"n": int(mask.sum()), "r_model": np.nan}
            continue
        r_model, _ = pearsonr(preds[mask], trues[mask])
        results[phase_name] = {"n": int(mask.sum()), "r_model": float(r_model)}

    if len(trues) > 7:
        r_p, _ = pearsonr(trues[:-7], trues[7:])
        results["persist_lag7_all"] = float(r_p)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["remote_only", "local_only", "full"]
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p90_ns = load_ns_p90()
    print(f"p90_ns: min={p90_ns.min():.3f}  max={p90_ns.max():.3f}")

    lm_class = MODULE_MAP[args.mode]
    print(f"\nMode: {args.mode}  →  {lm_class.__name__}")

    all_results = {}

    for fold in range(N_FOLDS):
        import re

        if args.mode == "full":
            cfg = FULL_BASE / f"TbotAtm_lstmonly_fold{fold}" / "config.yaml"
            ckpt_dir = FULL_BASE / f"TbotAtm_lstmonly_fold{fold}" / "checkpoints"
        else:
            folder = "remote" if args.mode == "remote_only" else "local"
            cfg = (
                Path(__file__).parents[1]
                / f"configs/partition/{folder}/fold{fold}.yaml"
            )
            name = "remote" if args.mode == "remote_only" else "local"
            ckpt_dir = (
                PARTITION_BASE / f"TbotAtm_{name}_seed42_fold{fold}" / "checkpoints"
            )

        ckpts = [
            c for c in ckpt_dir.glob("*.ckpt") if not re.search(r"-v\d+\.ckpt$", str(c))
        ]

        def val_loss(c):
            m = re.search(r"val_loss=([-\d.]+?)\.ckpt", c.name)
            return float(m.group(1).rstrip(".")) if m else float("inf")

        best_ckpt = min(ckpts, key=val_loss)
        print(f"\nfold{fold}: {best_ckpt.name}")

        preds, trues, doys = run_fold(fold, args.mode, lm_class, cfg, best_ckpt)
        results = skill_by_phase(preds, trues, doys, p90_ns)
        all_results[fold] = results

        out_path = OUT_DIR / f"{args.mode}_fold{fold}.npz"
        np.savez(out_path, preds=preds, trues=trues, doys=doys)

        print(f"  Results fold{fold}:")
        for phase, res in results.items():
            if isinstance(res, dict):
                print(
                    f"    {phase:<12}: n={res['n']:4d}  r={res.get('r_model', float('nan')):.4f}"
                )
            else:
                print(f"    {phase}: {res:.4f}")

    print(f"\n{'='*55}")
    print(f"SUMMARY — {args.mode}  (mean ± std across {N_FOLDS} folds)")
    print(f"{'='*55}")
    for phase in ["all", "onset", "mid_event", "no_mhw"]:
        rs = [
            all_results[f][phase]["r_model"]
            for f in range(N_FOLDS)
            if phase in all_results[f]
            and not np.isnan(all_results[f][phase].get("r_model", np.nan))
        ]
        ns = [
            all_results[f][phase]["n"]
            for f in range(N_FOLDS)
            if phase in all_results[f]
        ]
        print(
            f"  {phase:<12}: r = {np.mean(rs):.4f} ± {np.std(rs):.4f}  (n_total={sum(ns)})"
        )

    import json

    summary = {
        str(f): {
            k: (
                v
                if not isinstance(v, dict)
                else {
                    kk: float(vv) if not isinstance(vv, int) else vv
                    for kk, vv in v.items()
                }
            )
            for k, v in r.items()
        }
        for f, r in all_results.items()
    }
    with open(OUT_DIR / f"{args.mode}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/{args.mode}_summary.json")


if __name__ == "__main__":
    main()
