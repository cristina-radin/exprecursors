"""
eval_onset_persistence.py — Persistence baseline on onset samples.

Loads existing NPZ files from eval_onset_skill.py (no inference needed).
Computes lag-7 persistence on the 84 onset indices identified by the
Hobday-based onset detection in eval_onset_skill.py, and compares against
model skill (full / remote_only / local_only).

Persistence definition:
  persist[i] = trues[i - 7]  (last value of the 60-day input window)
  within each fold's test array (test samples are consecutive in time).
  First 7 samples of each fold: NaN if any onset day falls there (reported).

Usage (CPU, login node):
  python scripts/eval_onset_persistence.py --output experiments/figures/onset_skill_vs_persistence.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.paths import CLIM_FILE, EXPERIMENTS_DIR

ONSET_DIR = EXPERIMENTS_DIR / "partition" / "onset_skill"
LEAD = 7
N_FOLDS = 5
MODES = ["full", "remote_only", "local_only"]


# ── Hobday helpers (identical to eval_onset_skill.py) ────────────────────────


def load_ns_p90(smooth_days=31):
    import xarray as xr

    ds = xr.open_dataset(CLIM_FILE)
    p90 = ds.p90_thresh.sel(lat=slice(50.0, 63.0), lon=slice(-5.0, 13.0))
    p90 = p90.mean(dim=["lat", "lon"], skipna=True).values
    ds.close()
    return uniform_filter1d(p90, size=smooth_days, mode="wrap")


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


def onset_mask(trues, doys, p90_ns):
    """Returns boolean array: True at first day of each Hobday MHW event."""
    thr = p90_ns[doys - 1]
    mhw = apply_hobday(trues > thr)
    onset = np.zeros(len(mhw), dtype=bool)
    for i in range(len(mhw)):
        if mhw[i] and (i == 0 or not mhw[i - 1]):
            onset[i] = True
    return onset


# ── Fisher z-transform CI ────────────────────────────────────────────────────


def r_ci(r, n, alpha=0.05):
    """95% CI for Pearson r via Fisher z-transform."""
    if n <= 3:
        return (float("nan"), float("nan"))
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    z_lo = z - 1.96 * se
    z_hi = z + 1.96 * se
    return float(np.tanh(z_lo)), float(np.tanh(z_hi))


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_onset_skill(results: dict, output: Path):
    """Grouped bar chart: r_model vs r_persist per partition mode, with 95% CI."""
    modes = [m for m in MODES if m in results and not np.isnan(results[m]["r_model"])]
    r_model = np.array([results[m]["r_model"] for m in modes])
    r_persist = np.array([results[m]["r_persist"] for m in modes])
    ci_model = np.array([results[m]["r_model_ci"] for m in modes])
    ci_persist = np.array([results[m]["r_persist_ci"] for m in modes])

    err_model = np.abs(ci_model.T - r_model)
    err_persist = np.abs(ci_persist.T - r_persist)

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        x - width / 2,
        r_persist,
        width,
        yerr=err_persist,
        capsize=5,
        color="#e08214",
        alpha=0.85,
        label="Persistence (lag-7)",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        r_model,
        width,
        yerr=err_model,
        capsize=5,
        color="#2166ac",
        alpha=0.85,
        label="Model",
        zorder=3,
    )
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title(
        f"Onset skill vs persistence (n={results[modes[0]]['n']}, pooled 5 folds)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35, axis="y")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="experiments/figures/onset_skill_vs_persistence.png",
        help="Output path for the r_model vs r_persist bar chart",
    )
    args = parser.parse_args()

    p90_ns = load_ns_p90()
    print(f"p90_ns loaded: min={p90_ns.min():.3f}  max={p90_ns.max():.3f}\n")

    results = {}

    for mode in MODES:
        pooled_preds = []
        pooled_persist = []
        pooled_trues = []
        n_boundary_nan = 0

        for fold in range(N_FOLDS):
            path = ONSET_DIR / f"{mode}_fold{fold}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing NPZ: {path}")

            npz = np.load(path)
            preds = npz["preds"]
            trues = npz["trues"]
            doys = npz["doys"].astype(int)

            # lag-7 persistence within this fold's consecutive test series
            persist = np.full_like(trues, np.nan)
            persist[LEAD:] = trues[:-LEAD]

            omask = onset_mask(trues, doys, p90_ns)
            n_onset_fold = omask.sum()

            # check for onset days in the NaN boundary region
            boundary_nan = omask & np.isnan(persist)
            n_boundary = boundary_nan.sum()
            if n_boundary > 0:
                n_boundary_nan += n_boundary
                print(
                    f"  WARNING {mode} fold{fold}: {n_boundary} onset day(s) "
                    f"in first {LEAD} positions (persistence=NaN, excluded)"
                )

            valid = omask & np.isfinite(persist)
            pooled_preds.extend(preds[valid])
            pooled_persist.extend(persist[valid])
            pooled_trues.extend(trues[valid])

            print(
                f"  {mode} fold{fold}: n_onset={n_onset_fold}, "
                f"n_valid={valid.sum()}"
            )

        pooled_preds = np.array(pooled_preds)
        pooled_persist = np.array(pooled_persist)
        pooled_trues = np.array(pooled_trues)
        n = len(pooled_trues)

        if n < 4:
            results[mode] = {"n": n, "r_model": float("nan"), "r_persist": float("nan")}
            continue

        r_model, _ = pearsonr(pooled_preds, pooled_trues)
        r_persist, _ = pearsonr(pooled_persist, pooled_trues)

        results[mode] = {
            "n": n,
            "n_boundary_nan": n_boundary_nan,
            "r_model": float(r_model),
            "r_model_ci": r_ci(r_model, n),
            "r_persist": float(r_persist),
            "r_persist_ci": r_ci(r_persist, n),
        }

    # ── Print table ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("ONSET SKILL vs. LAG-7 PERSISTENCE  (n=84 onset days, pooled 5 folds)")
    print("  onset = first day of Hobday MHW event in target series")
    print("  persistence = to_anom at last input day (7 days before target)")
    print("  CI: 95% Fisher z-transform")
    print("=" * 72)
    print(
        f"{'Mode':<14} {'n':>4}   {'r_model':>8}  {'95% CI':>18}   {'r_persist':>9}  {'95% CI':>18}"
    )
    print("-" * 72)
    for mode in MODES:
        r = results[mode]
        n = r["n"]
        rm = r["r_model"]
        rp = r["r_persist"]
        ci_m = r.get("r_model_ci", (float("nan"), float("nan")))
        ci_p = r.get("r_persist_ci", (float("nan"), float("nan")))
        print(
            f"{mode:<14} {n:>4}   "
            f"{rm:>8.4f}  [{ci_m[0]:+.3f}, {ci_m[1]:+.3f}]   "
            f"{rp:>9.4f}  [{ci_p[0]:+.3f}, {ci_p[1]:+.3f}]"
        )
    print("=" * 72)

    # also print overall all-phase r for context
    print()
    print("Context — all-phase r (from existing JSON summaries):")
    import json

    for mode in MODES:
        with open(ONSET_DIR / f"{mode}_summary.json") as f:
            s = json.load(f)
        rs_all = [s[str(f)]["all"]["r_model"] for f in range(N_FOLDS)]
        print(f"  {mode:<14}: r_all = {np.mean(rs_all):.4f} ± {np.std(rs_all):.4f}")

    plot_onset_skill(results, Path(args.output))


if __name__ == "__main__":
    main()
