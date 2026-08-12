"""
plot_lead_sweep.py — Figure: Pearson r and MAE vs lead time.

Usage:
  python eval/plot_lead_sweep.py \
      --dir experiments/lead_sweep \
      --output figures/lead_sweep.png
"""

import argparse
import re

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_run(run_dir: Path):
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    lead = cfg.get("lead_time")
    if lead is None:
        return None

    logs = sorted(run_dir.glob("slurm-*.out"))
    if not logs:
        return None
    text = logs[-1].read_text(errors="ignore")

    m_r = re.search(r"Pearson r=([\d.]+)", text)
    m_mae = re.search(r"MAE=([\d.]+) °C", text)
    if not m_r:
        return None

    return {
        "lead": lead,
        "r": float(m_r.group(1)),
        "mae": float(m_mae.group(1)) if m_mae else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="experiments/lead_sweep")
    parser.add_argument("--output", default="figures/lead_sweep.png")
    args = parser.parse_args()

    base = Path(args.dir)
    groups = defaultdict(list)
    for rd in sorted(base.iterdir()):
        if not rd.is_dir():
            continue
        row = parse_run(rd)
        if row:
            groups[row["lead"]].append(row)

    leads, r_mean, r_std, mae_mean, mae_std = [], [], [], [], []
    for lead in sorted(groups):
        vals = groups[lead]
        rs = [v["r"] for v in vals]
        maes = [v["mae"] for v in vals if v["mae"] is not None]
        leads.append(lead)
        r_mean.append(np.mean(rs))
        r_std.append(np.std(rs))
        mae_mean.append(np.mean(maes))
        mae_std.append(np.std(maes))

    leads = np.array(leads)
    r_mean = np.array(r_mean)
    r_std = np.array(r_std)
    mae_mean = np.array(mae_mean)
    mae_std = np.array(mae_std)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Pearson r ---
    ax = axes[0]
    ax.plot(leads, r_mean, "o-", color="#2166ac", lw=2, ms=7, zorder=3)
    ax.fill_between(
        leads,
        r_mean - r_std,
        r_mean + r_std,
        alpha=0.2,
        color="#2166ac",
        label="±1 std (25 runs)",
    )
    ax.axhline(0.7, color="gray", ls="--", lw=1.2, label="r = 0.70")
    ax.set_xlabel("Lead time (days)", fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title("Forecast skill vs lead time", fontsize=13)
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(leads)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35)

    # Annotate r values
    for x, y in zip(leads, r_mean):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            color="#2166ac",
        )

    # --- MAE ---
    ax = axes[1]
    ax.plot(leads, mae_mean, "s-", color="#d6604d", lw=2, ms=7, zorder=3)
    ax.fill_between(
        leads,
        mae_mean - mae_std,
        mae_mean + mae_std,
        alpha=0.2,
        color="#d6604d",
        label="±1 std (25 runs)",
    )
    ax.set_xlabel("Lead time (days)", fontsize=12)
    ax.set_ylabel("MAE (°C)", fontsize=12)
    ax.set_title("Mean absolute error vs lead time", fontsize=13)
    ax.set_xticks(leads)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35)

    for x, y in zip(leads, mae_mean):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            color="#d6604d",
        )

    fig.suptitle(
        "CNN-LSTM (lstm_only) + GNLL — TbotAtm — 5 seeds × 5 folds",
        fontsize=11,
        color="dimgray",
    )
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
