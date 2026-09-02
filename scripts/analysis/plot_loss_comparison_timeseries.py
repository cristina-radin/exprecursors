"""
plot_loss_comparison_timeseries.py -- Aug 24 2026, meeting-prep figure for
the "loss function comparison" slide (MSE -> GNLL -> GNLL+quantile).

Same visual language as experiments/figures/step5_quantile_v2_predictions/
combined_test_timeseries_5fold.png (truth vs. prediction, +-1 std shaded
band where the model has an uncertainty output), but as three stacked
panels for a direct side-by-side comparison, all on fold0 (the only fold
trained for the MSE/focal families -- so all three loss variants are
compared on the exact same architecture and the exact same test years,
verified by asserting identical dates arrays):

  panel 1 - full_mse_v3           (gaussian_nll=False, quantile_head=False, focal_weight=False): truth vs prediction, no std (model has none)
  panel 2 - full_gnll_focal_v2    (gaussian_nll=True,  quantile_head=False, focal_weight=True):  truth vs prediction +-1 std -- this project's actual first attempt to bias toward extremes (loss reweighting), superseded by the quantile head below
  panel 3 - full_gnll_quantile_v2 (gaussian_nll=True,  quantile_head=True,  focal_weight=False): truth vs mean prediction +-1 std, plus the quantile head's own q_pred

fold0's test years are non-consecutive (1985, 1991, 2000, 2002, 2005,
2014, 2017, 2018) -- plotted on one chronological x-axis with real gaps
where a year isn't in this fold's test set, same approach as
plot_combined_test_timeseries.py.

CPU only.

Usage:
  python scripts/analysis/plot_loss_comparison_timeseries.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.hobday import load_ns_p90  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "figures" / "loss"
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)

FAMILIES = {
    "mse": dict(config_dir="full_mse_v3", label="MSE"),
    "gnll": dict(config_dir="full_gnll_focal_v2", label="GNLL + focal reweighting"),
    "gnll_quantile": dict(
        config_dir="full_gnll_quantile_v2", label="GNLL + quantile head (tau=0.9)"
    ),
}


def run_fold0(config_dir):
    cfg_path = REPO_ROOT / "configs" / "partition" / config_dir / "fold0.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    run_dir = Path(cfg["output_dir"])

    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    print(f"  {config_dir}: loading {ckpt.name}", flush=True)
    extra = {}
    if cfg.get("focal_weight", False):
        # p90_by_doy isn't persisted in the checkpoint (known_issues.md #39)
        # -- recompute exactly as train_partition.py did and pass it
        # explicitly, same pattern as _adhoc_eval_extreme_recall.py.
        extra["p90_by_doy"] = torch.tensor(load_ns_p90(), dtype=torch.float32)
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=True, map_location=device, **extra
    )
    lm.eval().to(device)

    dm = LazyDataModule(str(cfg_path))
    dm.setup()
    full_ds = dm.test_dataset.dataset
    test_indices = dm.test_dataset.indices
    test_dl = DataLoader(dm.test_dataset, batch_size=32, shuffle=False, num_workers=0)

    gaussian_nll = model_kwargs.get("gaussian_nll", False)
    has_quantile = model_kwargs.get("quantile_head", False)

    mean_preds, log_vars, q_preds, trues = [], [], [], []
    with torch.no_grad():
        for batch in test_dl:
            xs, xt, y = batch[0], batch[1], batch[2]
            xs, xt = xs.float().to(device), xt.float().to(device)
            if has_quantile:
                y_hat, q_pred = lm.model.forward_with_quantile(xs, xt)
                q_preds.append(q_pred.squeeze(-1).cpu())
            else:
                y_hat = lm.model(xs, xt)
            mean_preds.append(
                (y_hat[:, 0] if y_hat.ndim == 2 else y_hat.squeeze(-1)).cpu()
            )
            if gaussian_nll:
                log_vars.append(y_hat[:, 1].cpu())
            trues.append(y.squeeze(-1).cpu())

    mean_preds = torch.cat(mean_preds).numpy() * lm.target_std + lm.target_mean
    trues = torch.cat(trues).numpy() * lm.target_std + lm.target_mean
    std_preds = None
    if gaussian_nll:
        std_preds = torch.exp(0.5 * torch.cat(log_vars)).numpy() * lm.target_std
    q_preds_c = None
    if has_quantile:
        q_preds_c = torch.cat(q_preds).numpy() * lm.target_std + lm.target_mean

    target_idx = np.array(
        [i + full_ds.window_size - 1 + full_ds.lead_time for i in test_indices]
    )
    dates = full_ds.ds.time.values[target_idx]

    r, _ = pearsonr(trues, mean_preds)
    mae = np.abs(trues - mean_preds).mean()
    print(
        f"  {config_dir}: n_test={len(dates)}  r={r:.3f}  MAE={mae:.3f}degC", flush=True
    )

    return dict(
        dates=dates,
        trues=trues,
        mean_preds=mean_preds,
        std_preds=std_preds,
        q_preds=q_preds_c,
        r=r,
        mae=mae,
    )


def main():
    results = {}
    ref_dates = None
    for key, meta in FAMILIES.items():
        res = run_fold0(meta["config_dir"])
        if ref_dates is None:
            ref_dates = res["dates"]
        else:
            assert np.array_equal(res["dates"], ref_dates), (
                f"{key}: fold0 test dates differ from the reference family -- "
                "the three loss variants are not directly comparable, investigate."
            )
        results[key] = res
    print(
        f"\nAll three families share identical fold0 test dates ({len(ref_dates)} samples).",
        flush=True,
    )

    COL_TRUTH = "#2c5f8a"
    COL_PRED = "#e08283"
    COL_Q = "#8a2c2c"

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, facecolor="#fcfcfb")

    # ---- Panel 1: MSE (no uncertainty output) ----
    ax = axes[0]
    res = results["mse"]
    ax.plot(res["dates"], res["trues"], color=COL_TRUTH, lw=1.0, label="Truth")
    ax.plot(
        res["dates"],
        res["mean_preds"],
        color=COL_PRED,
        lw=0.9,
        ls="--",
        label="Prediction",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("to_anom (°C)")
    ax.set_title(
        f"MSE loss — fold 0 test predictions   r={res['r']:.3f}   MAE={res['mae']:.3f}°C",
        fontsize=12.5,
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # ---- Panel 2: GNLL (mean +- std) ----
    ax = axes[1]
    res = results["gnll"]
    ax.fill_between(
        res["dates"],
        res["mean_preds"] - res["std_preds"],
        res["mean_preds"] + res["std_preds"],
        color=COL_PRED,
        alpha=0.25,
        lw=0,
        label="Prediction ±1 std (GNLL)",
    )
    ax.plot(res["dates"], res["trues"], color=COL_TRUTH, lw=1.0, label="Truth")
    ax.plot(
        res["dates"],
        res["mean_preds"],
        color=COL_PRED,
        lw=0.9,
        ls="--",
        label="Prediction (mean)",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("to_anom (°C)")
    ax.set_title(
        f"GNLL + focal reweighting — fold 0 test predictions   r={res['r']:.3f}   MAE={res['mae']:.3f}°C",
        fontsize=12.5,
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # ---- Panel 3: GNLL + quantile head (mean +- std, plus q_pred) ----
    ax = axes[2]
    res = results["gnll_quantile"]
    ax.fill_between(
        res["dates"],
        res["mean_preds"] - res["std_preds"],
        res["mean_preds"] + res["std_preds"],
        color=COL_PRED,
        alpha=0.25,
        lw=0,
        label="Mean prediction ±1 std (GNLL)",
    )
    ax.plot(res["dates"], res["trues"], color=COL_TRUTH, lw=1.0, label="Truth")
    ax.plot(
        res["dates"],
        res["mean_preds"],
        color=COL_PRED,
        lw=0.9,
        ls="--",
        label="Mean-head prediction",
    )
    ax.plot(
        res["dates"],
        res["q_preds"],
        color=COL_Q,
        lw=1.1,
        label="Quantile-head prediction (tau=0.9)",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("to_anom (°C)")
    ax.set_xlabel("Year")
    ax.set_title(
        f"GNLL + quantile-head loss — fold 0 test predictions   r_mean={res['r']:.3f}   MAE_mean={res['mae']:.3f}°C",
        fontsize=12.5,
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Ground truth vs. prediction by loss function (fold 0, same architecture & test years)",
        fontsize=14.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = OUT_DIR / "loss_comparison_timeseries_fold0.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
