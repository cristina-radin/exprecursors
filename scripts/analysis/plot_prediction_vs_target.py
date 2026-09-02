"""
plot_prediction_vs_target.py — time series: true target vs. mean head vs.
quantile head (q_pred) vs. Hobday p90 threshold, over an illustrative
year, for the committed full_gnll_quantile_v2 model. Requested Aug 21
2026 ("quiero hacer un plot para mostrar con leyenda el prediction").

Uses fold0 (test_years include 2014, the single most active MHW year in
the 1985-2024 record -- see docs/known_issues.md #45's stratified_kfold
ranking) so the illustrative year actually has real MHW activity to show.

Reuses: LazyDataModule (split+normalization), best_ckpt()/
load_model_config()/CNNLightningModule.load_from_checkpoint() (checkpoint
loading, same pattern as scripts/analysis/quantile_head_recall*.py and
scripts/ig_partition_quantile.py), src/utils/hobday.py (p90 threshold).

CPU only (plain inference, no IG/backward needed). Not part of the
permanent pipeline.

Usage:
  python scripts/analysis/plot_prediction_vs_target.py --year 2014
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.hobday import load_ns_p90  # noqa: E402

CFG_PATH = REPO_ROOT / "configs" / "partition" / "full_gnll_quantile_v2" / "fold0.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "experiments" / "figures" / "step5_quantile_v2_predictions"
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    cfg = yaml.safe_load(open(CFG_PATH))
    dm = LazyDataModule(str(CFG_PATH))
    dm.setup()
    full_ds = dm.test_dataset.dataset
    test_indices = dm.test_dataset.indices

    run_dir = Path(cfg["output_dir"])
    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    print(f"Loading {ckpt.name}", flush=True)
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=True, map_location=device
    )
    lm.eval().to(device)

    p90 = load_ns_p90()
    test_dl = DataLoader(dm.test_dataset, batch_size=32, shuffle=False, num_workers=0)

    mean_preds, q_preds, trues = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            xs, xt, y = batch[0], batch[1], batch[2]
            xs, xt = xs.float().to(device), xt.float().to(device)
            y_hat, q_pred = lm.model.forward_with_quantile(xs, xt)
            mean_preds.append((y_hat[:, 0]).cpu())
            q_preds.append(q_pred.squeeze(-1).cpu())
            trues.append(y.squeeze(-1).cpu())
    mean_preds = torch.cat(mean_preds).numpy() * lm.target_std + lm.target_mean
    q_preds = torch.cat(q_preds).numpy() * lm.target_std + lm.target_mean
    trues = torch.cat(trues).numpy() * lm.target_std + lm.target_mean

    target_idx = np.array(
        [i + full_ds.window_size - 1 + full_ds.lead_time for i in test_indices]
    )
    years = full_ds.years[target_idx]
    doys = full_ds.doys[target_idx]
    doys_clamped = np.where(doys >= 365, 365, doys)
    thresh = p90[doys_clamped - 1]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "test_predictions_quantile_v2_fold0.npz"
    np.savez(
        npz_path,
        trues=trues,
        mean_preds=mean_preds,
        q_preds=q_preds,
        thresh=thresh,
        years=years,
        doys=doys_clamped,
    )
    print(
        f"Saved full test-set predictions to {npz_path} (reusable for other years)",
        flush=True,
    )

    mask = years == args.year
    n = int(mask.sum())
    print(f"Year {args.year}: {n} test samples found in fold0's test set", flush=True)
    if n == 0:
        raise ValueError(
            f"Year {args.year} is not in fold0's test years -- pick one of the "
            "fold0 test years (see docs/known_issues.md #45 or narrative.md)."
        )

    order = np.argsort(doys_clamped[mask])
    doy_sel = doys_clamped[mask][order]
    true_sel = trues[mask][order]
    mean_sel = mean_preds[mask][order]
    q_sel = q_preds[mask][order]
    thresh_sel = thresh[mask][order]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        doy_sel, true_sel, color="black", lw=1.6, label="True target (NS-mean to_anom)"
    )
    ax.plot(
        doy_sel, mean_sel, color="#2166ac", lw=1.4, label="Mean head (point forecast)"
    )
    ax.plot(
        doy_sel,
        q_sel,
        color="#d6604d",
        lw=1.4,
        label="Quantile head q_pred (τ=0.9, MHW risk signal)",
    )
    ax.plot(
        doy_sel,
        thresh_sel,
        color="gray",
        lw=1.2,
        ls="--",
        label="Hobday p90 threshold (basin-mean)",
    )
    ax.fill_between(
        doy_sel,
        thresh_sel,
        true_sel,
        where=(true_sel > thresh_sel),
        color="gray",
        alpha=0.15,
        label="True MHW days (basin-mean)",
    )
    ax.set_xlabel("Day of year")
    ax.set_ylabel("NS-box mean to_anom (°C)")
    ax.set_title(
        f"full_gnll_quantile_v2 (fold0) — {args.year}: prediction vs. target, "
        f"7-day lead"
    )
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prediction_vs_target_{args.year}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
