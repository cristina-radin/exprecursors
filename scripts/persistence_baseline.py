"""
persistence_baseline.py — Compare model skill vs lag-τ persistence.

Persistence forecast at lead τ: predict target[t+τ] = target[t].
Computes Pearson r for each lead time from the full dataset and overlays
the model's ensemble r from the lead sweep results.

Usage:
  python eval/persistence_baseline.py \
      --data   /path/to/merged_daily.nc \
      --sweep  experiments/lead_sweep \
      --output experiments/figures/lead_sweep_vs_persistence.png
"""

import argparse
import re

import matplotlib
import numpy as np
import xarray as xr
import yaml

matplotlib.use("Agg")
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

LEAD_TIMES = [1, 3, 5, 7, 10, 14, 21, 30]


def persistence_skill(target: np.ndarray, leads: list) -> dict:
    """Pearson r of lag-τ persistence on full time series."""
    results = {}
    for tau in leads:
        y_true = target[tau:]
        y_pred = target[:-tau]
        r, _ = pearsonr(y_true, y_pred)
        results[tau] = r
    return results


def load_model_results(sweep_dir: Path):
    """Parse lead sweep run dirs → {lead: [r values]}."""
    groups = defaultdict(list)
    for rd in sorted(sweep_dir.iterdir()):
        if not rd.is_dir():
            continue
        cfg_path = rd / "config.yaml"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        lead = cfg.get("lead_time")
        if lead is None:
            continue
        logs = sorted(rd.glob("slurm-*.out"))
        if not logs:
            continue
        text = logs[-1].read_text(errors="ignore")
        m = re.search(r"Pearson r=([\d.]+)", text)
        if m:
            groups[lead].append(float(m.group(1)))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Path to merged_daily.nc (must have 'target' variable)",
    )
    parser.add_argument(
        "--sweep",
        default="experiments/lead_sweep",
        help="Lead sweep experiment directory",
    )
    parser.add_argument(
        "--output", default="experiments/figures/lead_sweep_vs_persistence.png"
    )
    args = parser.parse_args()

    # Load target series
    ds = xr.open_dataset(args.data)
    target = ds["target"].values.astype(float)
    target = target[~np.isnan(target)]
    ds.close()
    print(
        f"Target series: {len(target)} days, mean={target.mean():.4f}, std={target.std():.4f}"
    )

    # Persistence skill
    pers = persistence_skill(target, LEAD_TIMES)
    pers_leads = np.array(sorted(pers))
    pers_r = np.array([pers[lead] for lead in pers_leads])
    print("\nPersistence r:")
    for lead, r in zip(pers_leads, pers_r):
        print(f"  lead={lead:2d}d  r={r:.3f}")

    # Model results
    groups = load_model_results(Path(args.sweep))
    mod_leads, mod_r_mean, mod_r_std = [], [], []
    for lead in LEAD_TIMES:
        if lead in groups and groups[lead]:
            vals = groups[lead]
            mod_leads.append(lead)
            mod_r_mean.append(np.mean(vals))
            mod_r_std.append(np.std(vals))
    mod_leads = np.array(mod_leads)
    mod_r_mean = np.array(mod_r_mean)
    mod_r_std = np.array(mod_r_std)

    print("\nModel r:")
    for lead, m, s in zip(mod_leads, mod_r_mean, mod_r_std):
        print(f"  lead={lead:2d}d  r={m:.3f} ±{s:.3f}")

    # Skill score: SS = (r_model - r_persist) / (1 - r_persist)
    ss_leads, ss_vals = [], []
    for lead, rm in zip(mod_leads, mod_r_mean):
        if lead in pers:
            ss = (rm - pers[lead]) / (1.0 - pers[lead])
            ss_leads.append(lead)
            ss_vals.append(ss)
    ss_leads = np.array(ss_leads)
    ss_vals = np.array(ss_vals)

    print("\nSkill score vs persistence:")
    for lead, ss in zip(ss_leads, ss_vals):
        print(f"  lead={lead:2d}d  SS={ss:.3f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: r vs lead
    ax = axes[0]
    ax.plot(
        pers_leads,
        pers_r,
        "^--",
        color="#e08214",
        lw=2,
        ms=8,
        label="Persistence (lag-τ)",
        zorder=3,
    )
    ax.plot(
        mod_leads,
        mod_r_mean,
        "o-",
        color="#2166ac",
        lw=2,
        ms=8,
        label="CNN-LSTM + GNLL",
        zorder=4,
    )
    ax.fill_between(
        mod_leads,
        mod_r_mean - mod_r_std,
        mod_r_mean + mod_r_std,
        alpha=0.2,
        color="#2166ac",
    )
    ax.axhline(0.7, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("Lead time (days)", fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title("Model vs persistence skill", fontsize=13)
    ax.set_ylim(0.3, 1.02)
    ax.set_xticks(LEAD_TIMES)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35)

    # Panel 2: skill score
    ax = axes[1]
    bars = ax.bar(
        ss_leads,
        ss_vals,
        width=1.8,
        color=["#2166ac" if v > 0 else "#d6604d" for v in ss_vals],
        alpha=0.85,
        zorder=3,
    )
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xlabel("Lead time (days)", fontsize=12)
    ax.set_ylabel("Skill score", fontsize=12)
    ax.set_title(
        "Skill score vs persistence\nSS = (r_model − r_persist) / (1 − r_persist)",
        fontsize=11,
    )
    ax.set_xticks(ss_leads)
    ax.grid(alpha=0.35, axis="y")
    for bar, v in zip(bars, ss_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.005 * np.sign(v),
            f"{v:.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )

    fig.suptitle(
        "CNN-LSTM (lstm_only) + GNLL — TbotAtm — 5 seeds × 5 folds",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
