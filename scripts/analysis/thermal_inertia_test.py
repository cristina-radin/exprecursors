"""
thermal_inertia_test.py — Test whether the temporal trend in T_bottom IG
attribution is explained by changes in North Sea SST thermal inertia.

HYPOTHESIS: If NS SST persistence (decorrelation timescale τ) increases in the
same decades as IG(ptho_bot), the T_bottom regime shift may be an artifact of
global warming increasing SST memory, not a genuine deep ocean driver signal.

METHOD:
  1. Compute ACF e-folding timescale τ of NS to_anom in rolling 10-year windows
     (step=1 yr, n_windows≈31). Done from raw data, no model.

  2. Compute mean |IG|(ptho_bot) from TbotAtm multiseed (25 runs, 5 seeds×5 folds)
     in the same windows. Test-set samples only.

  3. Pearson correlation + circular block bootstrap (block_len=10 windows, B=5000)
     because consecutive windows share 9 years of data → not independent.

OUTPUTS (in --output_dir):
  tau_ns.npz              — τ series (window_starts, tau_days)
  ig_tbot_peryear.npz     — mean |IG|(ptho_bot) per year and per window
  thermal_inertia.png     — 4-panel figure: τ trend, IG trend, scatter, ACF examples

NOTE: step 2 takes ~3-4h on GPU. Steps 1 and 3 run in seconds.
      If ig_tbot_peryear.npz already exists, step 2 is skipped.

Usage:
  python eval/thermal_inertia_test.py
  python eval/thermal_inertia_test.py --skip_ig  # if ig_tbot_peryear.npz exists
"""

import argparse
import os
import re
import sys
import tempfile

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.paths import (
    DATA_FILE as DATA_FILE_ENV,
)
from src.utils.paths import (
    EXPERIMENTS_DIR,
)

# ── constants ────────────────────────────────────────────────────────────────

DAILY_NC = str(DATA_FILE_ENV)
MULTISEED = Path(str(EXPERIMENTS_DIR))
SEEDS = [42, 123, 456, 789, 1337]
N_FOLDS = 5
N_IG_STEPS = 20  # fewer steps OK for correlation test

# NS ocean box (lat 50°N–63°N, lon -5°E–13°E in array indices)
NS_LAT = slice(100, 127)
NS_LON = slice(150, 187)

WINDOW_YR = 10  # rolling window length in years
STEP_YR = 1  # step
MAX_ACF_LAG = 180  # days; enough to capture e-folding for NS SST (~60-90d)
INV_E = 1.0 / np.e  # ≈ 0.368

BLOCK_LEN = 10  # bootstrap block length (windows)
N_BOOT = 5000


# ── step 1: decorrelation timescale ──────────────────────────────────────────


def compute_ns_daily_series():
    """Return daily NS-mean to_anom as 1D array, and the corresponding year/date arrays."""
    import xarray as xr

    ds = xr.open_dataset(DAILY_NC)
    # land_mask=1 means ocean in merged_daily.nc (inverted convention)
    ocean = ds["land_mask"].values.astype(bool)  # True = ocean
    ocean_ns = ocean[NS_LAT, NS_LON]  # (27, 37)

    to = ds["to_anom"].values[:, NS_LAT, NS_LON]  # (T, 27, 37)
    ds_time = ds["time"].values
    ds.close()

    # Spatial mean over NS ocean pixels only
    ocean_ns_flat = ocean_ns.flatten()
    to_flat = to.reshape(len(to), -1)[:, ocean_ns_flat]  # (T, n_ocean)
    ns_series = to_flat.mean(axis=1)  # (T,)

    # Parse years
    import pandas as pd

    times = pd.DatetimeIndex(ds_time)
    years = times.year.values
    return ns_series, years


def acf_efold_timescale(series, max_lag=MAX_ACF_LAG):
    """
    Compute ACF of `series` and return the e-folding timescale τ (days).
    τ = smallest lag k where ACF(k) < 1/e.
    If ACF never drops that low, returns max_lag.
    """
    n = len(series)
    s = series - series.mean()
    var = np.var(s)
    if var == 0:
        return max_lag

    r = np.array([np.mean(s[: n - k] * s[k:]) / var for k in range(max_lag + 1)])

    # First lag where ACF drops below 1/e
    crossings = np.where(r < INV_E)[0]
    if len(crossings) == 0:
        return float(max_lag)
    return float(crossings[0])


def compute_tau_rolling(ns_series, years, window_yr=WINDOW_YR, step_yr=STEP_YR):
    """
    For each rolling window of `window_yr` years, compute the e-folding timescale.
    Returns: window_start_years (array), tau_days (array)
    """
    first_yr = int(years[0])
    last_yr = int(years[-1])
    window_starts = range(first_yr, last_yr - window_yr + 2, step_yr)

    tau_list = []
    start_list = []

    for t_start in window_starts:
        t_end = t_start + window_yr  # exclusive
        mask = (years >= t_start) & (years < t_end)
        seg = ns_series[mask]
        if len(seg) < 365:
            continue
        tau_list.append(acf_efold_timescale(seg))
        start_list.append(t_start)
        print(f"  [{t_start}–{t_end-1}]  τ = {tau_list[-1]:.1f} days")

    return np.array(start_list), np.array(tau_list)


# ── step 2: IG(ptho_bot) from TbotAtm multiseed ─────────────────────────────


def best_ckpt(run_dir: Path) -> Path:
    ckpts = list((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/checkpoints/")

    def val_loss(p):
        m = re.search(r"val_loss=(-?[\d.]+)", str(p))
        return float(m.group(1)) if m else 0.0

    return min(ckpts, key=val_loss)


def patched_config_path(cfg_path: Path) -> str:
    """
    Write a temp YAML with corrected data_dir if the original file doesn't exist.
    Falls back to <stem>_OLD<suffix> automatically.
    Returns path to the (possibly patched) YAML.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_path = Path(cfg["data_dir"])
    if not data_path.exists():
        old_path = data_path.parent / (data_path.stem + "_OLD" + data_path.suffix)
        if old_path.exists():
            cfg["data_dir"] = str(old_path)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, dir="/tmp"
            )
            yaml.dump(cfg, tmp)
            tmp.close()
            return tmp.name
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Fallback also missing: {old_path}"
            )
    return str(cfg_path)


def load_model(ckpt_path, cfg, device):
    from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel

    inner = CNNLSTMModel(
        in_channels=cfg["in_channels"],
        cnn_features=cfg.get("cnn_features", 256),
        lstm_hidden=cfg.get("lstm_hidden", 512),
        lstm_layers=cfg.get("lstm_layers", 4),
        temporal_features=cfg.get("temporal_features", 0),
        dropout=cfg.get("dropout", 0.3),
        arch=cfg.get("arch", "lstm_only"),
        gaussian_nll=cfg.get("gaussian_nll", False),
        pooling=cfg.get("pooling", "max"),
        padding_mode=cfg.get("padding_mode", "zeros"),
        quantile_head=cfg.get("quantile_head", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path),
        model=inner,
        strict=False,
        map_location=device,
    )
    return lm.eval().to(device)


def compute_ig_signed(lm, xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """
    Signed IG w.r.t. x_spatial.
    xs: (1, T, C, H, W) on device
    Returns: (T, C, H, W) CPU — SIGNED, mean over T returned by caller
    """
    baseline = torch.zeros_like(xs)
    grads = []
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    for step in range(N_IG_STEPS):
        alpha = step / max(N_IG_STEPS - 1, 1)
        interp = (baseline + alpha * (xs - baseline)).requires_grad_(True)
        pred, _ = lm.model.forward_with_attention(interp.float(), xt.float())
        pred[:, 0].sum().backward()
        grads.append(interp.grad.detach().squeeze(0).clone())
    torch.backends.cudnn.enabled = prev_cudnn
    avg_grads = torch.stack(grads).mean(0)
    attributions = (xs.squeeze(0) - baseline.squeeze(0)) * avg_grads
    return attributions.cpu()


def compute_ig_peryear(device):
    """
    Run |IG| for ptho_bot (var index 0) on all 25 TbotAtm runs.
    Returns: ig_per_year dict {year: list of per-seed mean |IG|(ptho_bot) values}
    """
    from src.data.datamodule import LazyDataModule

    ig_per_year = {}  # year (int) → list of float
    n_runs = 0

    for seed in SEEDS:
        for fold in range(N_FOLDS):
            run_name = f"TbotAtm_lstmonly_seed{seed}_fold{fold}"
            run_dir = MULTISEED / run_name

            cfg_path = run_dir / "config.yaml"
            if not cfg_path.exists():
                print(f"  SKIP {run_name}: no config")
                continue
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            try:
                ckpt = best_ckpt(run_dir)
            except FileNotFoundError:
                print(f"  SKIP {run_name}: no checkpoint")
                continue

            try:
                patched = patched_config_path(cfg_path)
            except FileNotFoundError as e:
                print(f"  SKIP {run_name}: {e}")
                continue

            print(f"\n[{n_runs+1}/25] {run_name}")
            lm = load_model(ckpt, cfg, device)

            dm = LazyDataModule(config_path=patched)
            dm.setup()
            test_ds = dm.test_dataloader().dataset
            base_ds = test_ds.dataset if hasattr(test_ds, "dataset") else test_ds
            indices = (
                test_ds.indices if hasattr(test_ds, "indices") else range(len(test_ds))
            )

            run_peryear = {}  # year → list of |IG|(ptho_bot) scalar

            for local_i, global_i in enumerate(indices):
                xs, xt, _ = base_ds[global_i]
                target_t = global_i + base_ds.window_size - 1 + base_ds.lead_time
                yr = int(base_ds.years[target_t])

                xs_dev = xs.unsqueeze(0).to(device)
                xt_dev = xt.unsqueeze(0).to(device)

                attrs = compute_ig_signed(lm, xs_dev, xt_dev)  # (T, C, H, W)
                # ptho_bot is variable index 0; mean over T and all pixels
                ptho_bot_ig = attrs[:, 0, :, :].abs().mean().item()
                run_peryear.setdefault(yr, []).append(ptho_bot_ig)

                if (local_i + 1) % 100 == 0:
                    print(f"  {local_i+1}/{len(indices)}", end="\r")

            print(f"  Done: {len(indices)} samples, {len(run_peryear)} years")

            # Merge into global dict (each year gets one value per run)
            for yr, vals in run_peryear.items():
                ig_per_year.setdefault(yr, []).append(np.mean(vals))

            n_runs += 1

            # Clean up temp yaml if created
            if patched != str(cfg_path) and os.path.exists(patched):
                os.unlink(patched)

    print(f"\nFinished IG across {n_runs} runs.")
    return ig_per_year, n_runs


# ── step 3: align to windows + block bootstrap ───────────────────────────────


def window_mean_ig(ig_per_year: dict, window_starts, window_yr=WINDOW_YR):
    """Mean |IG|(ptho_bot) for each rolling window."""
    means = []
    for t_start in window_starts:
        yrs = range(t_start, t_start + window_yr)
        vals = []
        for y in yrs:
            if y in ig_per_year:
                vals.extend(ig_per_year[y])
        means.append(np.mean(vals) if vals else np.nan)
    return np.array(means)


def block_bootstrap_corr(x, y, block_len=BLOCK_LEN, n_boot=N_BOOT, rng_seed=0):
    """
    Circular block bootstrap for Pearson correlation.
    Returns (r_obs, ci_low, ci_high, p_two_tail)
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    r_obs = float(np.corrcoef(x, y)[0, 1])

    rng = np.random.default_rng(rng_seed)
    r_boots = []
    for _ in range(n_boot):
        n_blocks = int(np.ceil(n / block_len))
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]
        r_boot = float(np.corrcoef(x[idx], y[idx])[0, 1])
        r_boots.append(r_boot)

    r_boots = np.array(r_boots)
    ci_low = float(np.percentile(r_boots, 2.5))
    ci_high = float(np.percentile(r_boots, 97.5))
    # Two-tailed p-value: fraction of boots more extreme than observed
    p_val = float(np.mean(np.abs(r_boots) >= np.abs(r_obs)))
    return r_obs, ci_low, ci_high, p_val, r_boots


# ── plotting ──────────────────────────────────────────────────────────────────


def make_plots(
    window_starts,
    tau_days,
    ig_means,
    r_obs,
    ci_low,
    ci_high,
    p_val,
    r_boots,
    ns_series,
    years,
    out_dir,
):
    center_yrs = window_starts + WINDOW_YR / 2  # center of each window

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: τ trend
    ax1.plot(center_yrs, tau_days, "o-", color="steelblue", ms=5)
    ax1.set_xlabel("Window center year")
    ax1.set_ylabel("e-folding timescale τ (days)")
    ax1.set_title("NS SST decorrelation timescale\n(rolling 10-yr window)")
    ax1.grid(True, alpha=0.3)
    # Simple linear fit
    valid = ~np.isnan(tau_days)
    if valid.sum() > 2:
        p = np.polyfit(center_yrs[valid], tau_days[valid], 1)
        ax1.plot(
            center_yrs,
            np.polyval(p, center_yrs),
            "--",
            color="steelblue",
            alpha=0.6,
            label=f"trend: {p[0]:.2f} d/yr",
        )
        ax1.legend(fontsize=9)

    # Panel 2: IG(ptho_bot) trend
    valid_ig = ~np.isnan(ig_means)
    ax2.plot(center_yrs[valid_ig], ig_means[valid_ig], "o-", color="tomato", ms=5)
    ax2.set_xlabel("Window center year")
    ax2.set_ylabel("Mean |IG|(T_bottom) [a.u.]")
    ax2.set_title("T_bottom IG attribution\n(TbotAtm multiseed, rolling 10-yr window)")
    ax2.grid(True, alpha=0.3)
    if valid_ig.sum() > 2:
        p = np.polyfit(center_yrs[valid_ig], ig_means[valid_ig], 1)
        ax2.plot(
            center_yrs,
            np.polyval(p, center_yrs),
            "--",
            color="tomato",
            alpha=0.6,
            label=f"trend: {p[0]:.4f}/yr",
        )
        ax2.legend(fontsize=9)

    # Panel 3: scatter τ vs IG with bootstrap CI
    valid = ~(np.isnan(tau_days) | np.isnan(ig_means))
    ax3.scatter(tau_days[valid], ig_means[valid], color="purple", zorder=3)
    for i, (t, ig, yr) in enumerate(
        zip(tau_days[valid], ig_means[valid], center_yrs[valid])
    ):
        ax3.annotate(
            f"{int(yr)}",
            (t, ig),
            fontsize=7,
            alpha=0.6,
            textcoords="offset points",
            xytext=(3, 2),
        )
    ax3.set_xlabel("τ (days)")
    ax3.set_ylabel("|IG|(T_bottom)")
    significance = "significant" if p_val < 0.05 else "NOT significant"
    ax3.set_title(
        f"τ vs IG(T_bottom)\n"
        f"r = {r_obs:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]  "
        f"p = {p_val:.3f} ({significance})\n"
        f"block bootstrap, L={BLOCK_LEN}, B={N_BOOT}"
    )
    ax3.grid(True, alpha=0.3)
    # Draw regression line
    if valid.sum() > 2:
        p = np.polyfit(tau_days[valid], ig_means[valid], 1)
        t_range = np.linspace(tau_days[valid].min(), tau_days[valid].max(), 50)
        ax3.plot(t_range, np.polyval(p, t_range), "--", color="purple", alpha=0.7)

    # Panel 4: bootstrap distribution
    ax4.hist(r_boots, bins=50, color="gray", alpha=0.7, density=True)
    ax4.axvline(r_obs, color="red", lw=2, label=f"r_obs = {r_obs:.3f}")
    ax4.axvline(ci_low, color="blue", lw=1.5, ls="--", label="95% CI")
    ax4.axvline(ci_high, color="blue", lw=1.5, ls="--")
    ax4.axvline(0, color="black", lw=0.8, ls=":")
    ax4.set_xlabel("Bootstrap Pearson r")
    ax4.set_ylabel("Density")
    ax4.set_title(f"Bootstrap distribution (B={N_BOOT})\np = {p_val:.3f}")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "Thermal inertia test — NS SST persistence vs T_bottom IG attribution\n"
        "Null hypothesis: τ and IG(T_bottom) are uncorrelated",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = out_dir / "thermal_inertia.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Extra: ACF example curves for first, middle, last window
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    n_wins = len(window_starts)
    example_idxs = [0, n_wins // 2, n_wins - 1]
    for ax_ex, wi in zip(axes2, example_idxs):
        t_start = window_starts[wi]
        mask = (years >= t_start) & (years < t_start + WINDOW_YR)
        seg = ns_series[mask]
        s = seg - seg.mean()
        var = np.var(s)
        lags = range(MAX_ACF_LAG + 1)
        n = len(s)
        acf = np.array([np.mean(s[: n - k] * s[k:]) / var for k in lags])
        ax_ex.plot(lags, acf, color="steelblue", lw=1)
        ax_ex.axhline(INV_E, color="red", ls="--", lw=1, label=f"1/e ≈ {INV_E:.2f}")
        ax_ex.axvline(
            tau_days[wi],
            color="orange",
            ls="--",
            lw=1.5,
            label=f"τ = {tau_days[wi]:.0f} d",
        )
        ax_ex.axhline(0, color="gray", lw=0.6, ls=":")
        ax_ex.set_title(f"ACF — {t_start}–{t_start+WINDOW_YR-1}")
        ax_ex.set_xlabel("Lag (days)")
        ax_ex.legend(fontsize=8)
        ax_ex.grid(True, alpha=0.3)
    axes2[0].set_ylabel("Autocorrelation r(k)")
    plt.tight_layout()
    out_acf = out_dir / "thermal_inertia_acf_examples.png"
    plt.savefig(str(out_acf), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_acf}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/figures/thermal_inertia")
    parser.add_argument(
        "--skip_ig",
        action="store_true",
        help="Skip IG computation (load existing ig_tbot_peryear.npz)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── STEP 1: τ computation ─────────────────────────────────────────────────
    tau_cache = out_dir / "tau_ns.npz"
    if tau_cache.exists():
        print(f"\nLoading cached τ from {tau_cache}")
        d = np.load(str(tau_cache))
        window_starts = d["window_starts"]
        tau_days = d["tau_days"]
        ns_series = d["ns_series"]
        years = d["years"]
    else:
        print("\n=== STEP 1: Computing NS decorrelation timescale ===")
        ns_series, years = compute_ns_daily_series()
        print(f"NS series: {len(ns_series)} days, {years[0]}–{years[-1]}")
        window_starts, tau_days = compute_tau_rolling(ns_series, years)
        np.savez(
            str(tau_cache),
            window_starts=window_starts,
            tau_days=tau_days,
            ns_series=ns_series,
            years=years,
        )
        print(f"Saved {tau_cache}")

    # ── STEP 2: IG per year ───────────────────────────────────────────────────
    ig_cache = out_dir / "ig_tbot_peryear.npz"
    if ig_cache.exists() or args.skip_ig:
        if ig_cache.exists():
            print(f"\nLoading cached IG from {ig_cache}")
            d = np.load(str(ig_cache), allow_pickle=True)
            ig_per_year = d["ig_per_year"].item()
            n_runs = int(d["n_runs"])
        else:
            print(
                "\n--skip_ig set but no ig_tbot_peryear.npz found — cannot run step 3"
            )
            return
    else:
        print("\n=== STEP 2: Computing |IG|(ptho_bot) from TbotAtm multiseed ===")
        ig_per_year, n_runs = compute_ig_peryear(device)
        np.savez(
            str(ig_cache),
            ig_per_year=np.array(ig_per_year, dtype=object),
            n_runs=np.array(n_runs),
        )
        print(f"Saved {ig_cache}")

    # ── STEP 3: Align + correlate ─────────────────────────────────────────────
    print("\n=== STEP 3: Correlation + block bootstrap ===")

    ig_means = window_mean_ig(ig_per_year, window_starts)
    print("\nWindow summary (center yr | τ days | mean |IG|):")
    for ws, tau, ig in zip(window_starts, tau_days, ig_means):
        center = ws + WINDOW_YR / 2
        print(f"  {center:.0f}   τ={tau:.1f}   IG={ig:.5f}")

    r_obs, ci_low, ci_high, p_val, r_boots = block_bootstrap_corr(
        tau_days, ig_means, block_len=BLOCK_LEN, n_boot=N_BOOT
    )

    print("\n=== RESULT ===")
    print(f"  Pearson r    = {r_obs:.4f}")
    print(f"  95% CI       = [{ci_low:.4f}, {ci_high:.4f}]")
    print(
        f"  p (bootstrap, two-tail) = {p_val:.4f}  "
        f"({'significant' if p_val<0.05 else 'NOT significant'} at α=0.05)"
    )
    print(
        f"  n_windows = {(~np.isnan(tau_days) & ~np.isnan(ig_means)).sum()}  "
        f"block_len={BLOCK_LEN}  B={N_BOOT}"
    )
    print()
    if p_val < 0.05 and r_obs > 0:
        print("INTERPRETATION: Positive significant correlation suggests the T_bottom")
        print("  IG trend may (partly) reflect increasing NS SST thermal inertia.")
        print("  This is consistent with the 'regime shift as artifact' hypothesis.")
    elif p_val < 0.05 and r_obs < 0:
        print("INTERPRETATION: Negative correlation — longer τ is associated with LESS")
        print(
            "  T_bottom attribution. Inconsistent with thermal inertia artifact hypothesis."
        )
    else:
        print(
            "INTERPRETATION: No significant correlation. Thermal inertia does NOT explain"
        )
        print("  the T_bottom IG trend. The regime shift signal is likely genuine.")

    # ── STEP 4: Plots ─────────────────────────────────────────────────────────
    make_plots(
        window_starts,
        tau_days,
        ig_means,
        r_obs,
        ci_low,
        ci_high,
        p_val,
        r_boots,
        ns_series,
        years,
        out_dir,
    )
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
