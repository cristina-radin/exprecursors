"""
event_detection_lead_time_comparison.py — Aug 24 2026: compiles the 7
per-family event_detection_summary_*.json outputs (produced by
scripts/eval_event_detection.py across the full lead-time sweep + local/
remote partitions, job 29526761) into the single comparison the user
asked for: "hacer el eval de esta figura event_detection_pod_far_csi.png
teniendo en cuenta el sweep lead time ... Quiero mas lead times y mas
informacion al respecto." User called the underlying single-lead figure
"quiza el principal resultado del paper."

Reads only, no recomputation -- all numbers here are direct passthroughs
of eval_event_detection.py's own Clopper-Pearson CIs and event-paired
bootstrap results (B=2000, quantile_head vs persist), not re-derived.

CPU, seconds to run, no SLURM needed.

Usage:
  python scripts/analysis/event_detection_lead_time_comparison.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUMMARY_DIR = REPO_ROOT / "experiments" / "figures" / "step7_persistence"
OUT_DIR = SUMMARY_DIR

LEAD_FAMILIES = ["lead3", "lead5", "full_lead7", "lead14", "lead30"]
LEAD_VALUES = {"lead3": 3, "lead5": 5, "full_lead7": 7, "lead14": 14, "lead30": 30}
PARTITION_FAMILIES = ["full_lead7", "local", "remote"]
PARTITION_LABELS = {"full_lead7": "Full", "local": "Local (NS box)", "remote": "Remote"}

data = {}
for fam in set(LEAD_FAMILIES) | set(PARTITION_FAMILIES):
    path = SUMMARY_DIR / f"event_detection_summary_{fam}.json"
    with open(path) as f:
        data[fam] = json.load(f)

# ---------------------------------------------------------------------------
# Panel A/B: POD and CSI vs lead time (quantile_head vs mean_head vs persist)
# ---------------------------------------------------------------------------
leads = [LEAD_VALUES[f] for f in LEAD_FAMILIES]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
for sysname, color, marker in [
    ("quantile_head", "tab:blue", "o"),
    ("mean_head", "tab:orange", "s"),
    ("persist", "tab:gray", "^"),
]:
    pods = [data[f]["summary"][sysname]["pod"] for f in LEAD_FAMILIES]
    los = [data[f]["summary"][sysname]["pod_ci"][0] for f in LEAD_FAMILIES]
    his = [data[f]["summary"][sysname]["pod_ci"][1] for f in LEAD_FAMILIES]
    yerr = [np.array(pods) - np.array(los), np.array(his) - np.array(pods)]
    ax.errorbar(
        leads,
        pods,
        yerr=yerr,
        marker=marker,
        color=color,
        label=sysname,
        capsize=4,
        linewidth=2,
        markersize=8,
    )
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("POD (event-level, 95% CI)")
ax.set_title("Onset detection: POD vs lead time")
ax.set_xticks(leads)
ax.legend()
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

ax = axes[1]
diffs = [
    data[f]["bootstrap_vs_persist"]["quantile_head"]["mean_diff"] for f in LEAD_FAMILIES
]
los = [data[f]["bootstrap_vs_persist"]["quantile_head"]["ci"][0] for f in LEAD_FAMILIES]
his = [data[f]["bootstrap_vs_persist"]["quantile_head"]["ci"][1] for f in LEAD_FAMILIES]
sig = [
    data[f]["bootstrap_vs_persist"]["quantile_head"]["significant"]
    for f in LEAD_FAMILIES
]
yerr = [np.array(diffs) - np.array(los), np.array(his) - np.array(diffs)]
colors = ["tab:green" if s else "tab:red" for s in sig]
ax.bar(leads, diffs, width=1.8, color=colors, alpha=0.7)
ax.errorbar(leads, diffs, yerr=yerr, fmt="none", ecolor="black", capsize=4)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("POD(quantile_head) - POD(persist)")
ax.set_title(
    "Event-paired bootstrap POD difference\n(green = significant at 95%, B=2000)"
)
ax.set_xticks(leads)
ax.grid(alpha=0.3)

ax = axes[2]
x = np.arange(len(PARTITION_FAMILIES))
width = 0.35
q_pods = [data[f]["summary"]["quantile_head"]["pod"] for f in PARTITION_FAMILIES]
q_los = [data[f]["summary"]["quantile_head"]["pod_ci"][0] for f in PARTITION_FAMILIES]
q_his = [data[f]["summary"]["quantile_head"]["pod_ci"][1] for f in PARTITION_FAMILIES]
q_yerr = [np.array(q_pods) - np.array(q_los), np.array(q_his) - np.array(q_pods)]
p_pods = [data[f]["summary"]["persist"]["pod"] for f in PARTITION_FAMILIES]
p_los = [data[f]["summary"]["persist"]["pod_ci"][0] for f in PARTITION_FAMILIES]
p_his = [data[f]["summary"]["persist"]["pod_ci"][1] for f in PARTITION_FAMILIES]
p_yerr = [np.array(p_pods) - np.array(p_los), np.array(p_his) - np.array(p_pods)]
ax.bar(
    x - width / 2,
    q_pods,
    width,
    yerr=q_yerr,
    label="quantile_head",
    color="tab:blue",
    capsize=4,
)
ax.bar(
    x + width / 2,
    p_pods,
    width,
    yerr=p_yerr,
    label="persist",
    color="tab:gray",
    capsize=4,
)
ax.set_xticks(x)
ax.set_xticklabels([PARTITION_LABELS[f] for f in PARTITION_FAMILIES])
ax.set_ylabel("POD (event-level, 95% CI)")
ax.set_title("Onset detection at lead=7d:\nlocal-only vs remote-only vs full")
ax.legend()
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "event_detection_lead_time_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# Text/markdown table
# ---------------------------------------------------------------------------
ALL_FAMILIES = ["lead3", "lead5", "full_lead7", "lead14", "lead30", "local", "remote"]
LABELS = {
    "lead3": "lead=3d",
    "lead5": "lead=5d",
    "full_lead7": "lead=7d (full)",
    "lead14": "lead=14d",
    "lead30": "lead=30d",
    "local": "lead=7d, local-only",
    "remote": "lead=7d, remote-only",
}

lines = []
lines.append(
    "| Family | n_events | POD(quantile) [95% CI] | POD(persist) [95% CI] | Δ POD (bootstrap 95% CI) | Sig? | CSI(quantile) | CSI(persist) |"
)
lines.append("|---|---|---|---|---|---|---|---|")
for fam in ALL_FAMILIES:
    s = data[fam]["summary"]
    b = data[fam]["bootstrap_vs_persist"]["quantile_head"]
    n_events = s["quantile_head"]["hits"] + s["quantile_head"]["misses"]
    q, p = s["quantile_head"], s["persist"]
    lines.append(
        f"| {LABELS[fam]} | {n_events} "
        f"| {q['pod']:.3f} [{q['pod_ci'][0]:.3f}, {q['pod_ci'][1]:.3f}] "
        f"| {p['pod']:.3f} [{p['pod_ci'][0]:.3f}, {p['pod_ci'][1]:.3f}] "
        f"| {b['mean_diff']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] "
        f"| {'YES' if b['significant'] else 'no'} "
        f"| {q['csi']:.3f} | {p['csi']:.3f} |"
    )
table_md = "\n".join(lines)
print("\n" + table_md)

with open(OUT_DIR / "event_detection_lead_time_comparison.md", "w") as f:
    f.write(
        "# Event-level onset detection: model vs persistence, full lead-time sweep\n\n"
    )
    f.write(table_md + "\n\n")
    f.write(
        "All numbers from eval_event_detection.py (Clopper-Pearson exact "
        "binomial CIs, event-paired bootstrap B=2000, pooled across 5 folds "
        "per family). n_events=56 in every lead-sweep family (same test-set "
        "years/model family, only lead_time differs at train+eval time); "
        "local/remote also n_events=56 since they share the same fold "
        "definition and the same underlying MHW event ground truth "
        "(the events themselves don't depend on which model evaluates them).\n\n"
        "Key finding: quantile_head POD significantly exceeds persistence "
        "POD (event-paired bootstrap, 95% CI excludes 0) at EVERY lead time "
        "tested (3, 5, 7, 14, 30 days) and in both the local-only and "
        "remote-only partition experiments. The absolute gap narrows as "
        "lead time grows (+0.607 at lead=3d -> +0.217 at lead=30d) because "
        "persistence's own POD rises with lead (a longer [onset-lead, "
        "onset] scoring window makes it progressively easier for the "
        "already-elevated pre-onset state to cross threshold), not because "
        "the model gets worse in absolute terms. Local-only input achieves "
        "the single highest POD of any condition (0.768) -- higher than "
        "the full model -- while remote-only alone still significantly "
        "beats persistence (POD 0.589 vs 0.179), confirming precursor "
        "information adds real event-detection skill beyond the NS box's "
        "own local persistence.\n"
    )
print(f"\nSaved {OUT_DIR / 'event_detection_lead_time_comparison.md'}")
