"""
plot_sustained_lead_histogram.py -- Aug 24 2026, meeting-prep Figure 2 for
the persistence slide: lead-time-achieved histogram, quantile_head vs
persistence, sustained_full definition, at lead=7d (the deployed model).

Numbers are the real printed output of
scripts/analysis/sustained_lead_time.py (slurm-sustained_lead_time-
29558204.out, full_lead7 family, sustained_full definition, n_events=56
pooled across 5 folds) -- not re-derived here, this script only plots
them. sustained_full = alarm must be True on the onset day itself AND
stay True with no gap back to day d; lead_time_achieved = onset - d.
Events where the alarm never reaches the onset day at all count as a
miss (0 lead time achieved), plotted as its own bar so the histogram
doesn't silently hide non-detections.

Usage:
  python scripts/analysis/plot_sustained_lead_histogram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent.parent.parent / "results" / "figures" / "persistence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_EVENTS = 56
LEAD = 7

# hits histogram: {lead_time_achieved (days): n_events}, from the real printed output
QUANTILE_HITS = {0: 1, 2: 2, 3: 2, 5: 1, 7: 21}
PERSIST_HITS = {5: 1, 7: 3}

QUANTILE_POD = sum(QUANTILE_HITS.values()) / N_EVENTS  # 0.4821
PERSIST_POD = sum(PERSIST_HITS.values()) / N_EVENTS  # 0.0714

days = list(range(0, LEAD + 1))  # 0..7
q_counts = [QUANTILE_HITS.get(d, 0) for d in days]
p_counts = [PERSIST_HITS.get(d, 0) for d in days]
q_miss = N_EVENTS - sum(QUANTILE_HITS.values())
p_miss = N_EVENTS - sum(PERSIST_HITS.values())

labels = [str(d) for d in days] + ["miss"]
q_all = q_counts + [q_miss]
p_all = p_counts + [p_miss]

COL_Q = "#2a78d6"
COL_P = "#eb6834"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"

fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

x = np.arange(len(labels))
w = 0.36
bars_q = ax.bar(
    x - w / 2,
    q_all,
    width=w,
    color=COL_Q,
    label=f"Quantile head (POD={QUANTILE_POD:.1%})",
)
bars_p = ax.bar(
    x + w / 2, p_all, width=w, color=COL_P, label=f"Persistence (POD={PERSIST_POD:.1%})"
)

for bars in (bars_q, bars_p):
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.annotate(
                str(int(h)),
                (b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
                color="#52514e",
            )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_xlabel(
    "Lead time achieved (days before onset the alarm was continuously on) -- 'miss' = alarm never reached onset day",
    fontsize=10.5,
    color="#52514e",
)
ax.set_ylabel(f"Number of events (of n={N_EVENTS} pooled, 5 folds)", fontsize=11)
ax.set_title(
    "Sustained-alarm lead time at onset -- quantile head vs. persistence (lead=7d)",
    fontsize=14,
    fontweight="bold",
    loc="left",
    pad=14,
)
ax.legend(loc="upper left", fontsize=10.5, frameon=False)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
out_path = OUT_DIR / "sustained_lead_histogram_lead7.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print(f"Saved {out_path}")
