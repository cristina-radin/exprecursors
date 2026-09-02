"""
plot_two_regimes_of_skill.py -- Aug 24 2026, meeting-prep Figure 1 for the
persistence slide ("dos regimenes de skill"): two panels sharing the same
lead-time x-axis (3/5/7/14/30 days).

  Panel A: pooled r, model (mean head) vs. persistence -- persistence
           wins/ties at every lead. From experiments/figures/step7_persistence/
           lead_time_sweep_model_vs_persistence.npz (already computed,
           not re-derived here).
  Panel B: POD (sustained_full definition -- alarm must be True on the
           onset day itself, unbroken back to day d), quantile head vs.
           persistence -- the model wins by a wide margin at every lead.
           lead=7/14 from already-run sustained_lead_time.py logs;
           lead=3/5/30 parsed from the same script's newly-run output
           (submit_sustained_lead_time_remaining.sh, job 29566556).

Both panels pooled across all 5 folds, n_events=56 (event-level, same
ground-truth MHW events for every lead-time family per known_issues.md's
event-detection methodology).

Usage:
  python scripts/analysis/plot_two_regimes_of_skill.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "results" / "figures" / "persistence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEADS = [3, 5, 7, 14, 30]

# ---- Panel A data: pooled r, already cached ----
sweep = np.load(
    REPO_ROOT
    / "experiments/figures/step7_persistence/lead_time_sweep_model_vs_persistence.npz"
)
model_leads = list(sweep["model_leads"])
model_r_mean = dict(zip(model_leads, sweep["model_r_mean"]))
persist_lags = list(sweep["persist_lags"])
persist_r_by_lag = dict(zip(persist_lags, sweep["persist_r"]))

r_model = [model_r_mean[l] for l in LEADS]
r_persist = [persist_r_by_lag[l] for l in LEADS]

# ---- Panel B data: POD sustained_full, parse from real SLURM logs ----
LOG_BY_LEAD = {
    3: "slurm-sustained_lead_remaining-29566556_0.out",
    5: "slurm-sustained_lead_remaining-29566556_1.out",
    7: "slurm-sustained_lead_time-29558204.out",
    14: "slurm-sustained_lead_time_lead14-29564625.out",
    30: "slurm-sustained_lead_remaining-29566556_2.out",
}

POD_RE = re.compile(
    r"^(quantile_head|persist) / sustained_full: hits=\d+/\d+\s+POD=([\d.]+)"
)


def parse_pod(log_path):
    pods = {}
    for line in open(log_path):
        m = POD_RE.match(line.strip())
        if m:
            pods[m.group(1)] = float(m.group(2))
    assert (
        "quantile_head" in pods and "persist" in pods
    ), f"{log_path}: missing POD lines, got {pods}"
    return pods["quantile_head"], pods["persist"]


pod_quantile, pod_persist = [], []
for lead in LEADS:
    log_path = REPO_ROOT / LOG_BY_LEAD[lead]
    q, p = parse_pod(log_path)
    pod_quantile.append(q)
    pod_persist.append(p)
    print(
        f"lead={lead}: POD quantile_head={q:.4f}  POD persist={p:.4f}  (from {log_path.name})"
    )

# ---- Plot ----
COL_MODEL = "#2a78d6"
COL_PERSIST = "#eb6834"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), facecolor=SURFACE)
for ax in (axA, axB):
    ax.set_facecolor(SURFACE)

x = np.arange(len(LEADS))

# Panel A
axA.plot(
    x, r_model, "o-", color=COL_MODEL, lw=2, markersize=7, label="Model (mean head)"
)
axA.plot(x, r_persist, "o-", color=COL_PERSIST, lw=2, markersize=7, label="Persistence")
axA.set_xticks(x)
axA.set_xticklabels([f"{l}d" for l in LEADS])
axA.set_ylabel("Pooled r (point forecast)")
axA.set_title(
    "A. Continuous tracking skill", fontsize=13, fontweight="bold", loc="left"
)
axA.legend(loc="lower left", fontsize=10, frameon=False)
axA.grid(alpha=0.3)
axA.set_ylim(0.5, 1.0)
for spine in ["top", "right"]:
    axA.spines[spine].set_visible(False)

# Panel B
axB.plot(
    x,
    pod_quantile,
    "o-",
    color=COL_MODEL,
    lw=2,
    markersize=7,
    label="Model (quantile head)",
)
axB.plot(
    x, pod_persist, "o-", color=COL_PERSIST, lw=2, markersize=7, label="Persistence"
)
axB.set_xticks(x)
axB.set_xticklabels([f"{l}d" for l in LEADS])
axB.set_ylabel("POD (sustained_full, event-level)")
axB.set_title(
    "B. Sustained event detection", fontsize=13, fontweight="bold", loc="left"
)
axB.legend(loc="upper right", fontsize=10, frameon=False)
axB.grid(alpha=0.3)
axB.set_ylim(0, 0.7)
for spine in ["top", "right"]:
    axB.spines[spine].set_visible(False)

fig.suptitle(
    "Two regimes of skill: persistence wins point-forecast tracking, the model wins sustained event detection",
    fontsize=14.5,
    fontweight="bold",
    y=1.02,
)
fig.text(
    0.5,
    -0.04,
    "Panel A: correlation between predicted and true SST anomaly, pooled across all test days (5 folds).  "
    "Panel B: fraction of the n=56 pooled MHW events where the alarm was continuously on through the onset day itself\n"
    "(sustained_full -- a stricter, more paper-defensible definition than 'flagged at least once in the lead window'). "
    "These measure two different tasks: A is continuous point-forecast tracking, B is event-level detection.",
    ha="center",
    fontsize=9.5,
    color="#52514e",
)

fig.tight_layout()
out_path = OUT_DIR / "two_regimes_of_skill.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print(f"\nSaved {out_path}")
