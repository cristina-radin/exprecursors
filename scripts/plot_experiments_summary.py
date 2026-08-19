import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)

OUT = Path("/p/project1/hai_1127/radin1/exprecursors/figures/experiments_summary")
OUT.mkdir(exist_ok=True)

# Arch name mapping: config value → display name
ARCH_DISPLAY = {
    "attention_only": "CNN+Attention",
    "lstm_only": "CNN+LSTM",
    "lstm_attention": "CNN+LSTM+Attn",
    "convlstm": "ConvLSTM",
}


def parse_arch_from_name(name):
    """Parse architecture display name from directory name."""
    parts = name.split("_")
    for p in parts:
        if p == "lstmonly":
            return "CNN+LSTM"
        if p == "lstmattention":
            return "CNN+LSTM+Attn"
        if p == "attentiononly":
            return "CNN+Attention"
        if p == "convlstm":
            return "ConvLSTM"
    return "?"


def parse_var_from_name(name):
    if name.startswith("SSTAtm"):
        return "SST+Atm"
    if name.startswith("TbotAtm"):
        return "Tbot+Atm"
    if name.startswith("Atm"):
        return "Atm"
    return "?"


# ═══════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════

# 1) Hypersearch
hyper_dir = Path("/p/project1/hai_1127/radin1/exprecursors/experiments/hypersearch")
hyper_results = {}
for run_dir in sorted(hyper_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    mf = run_dir / "logs/version_0/metrics.csv"
    if not mf.exists():
        continue
    df = pd.read_csv(mf)
    if "val_loss" not in df.columns:
        continue
    name = run_dir.name
    lr_m = re.search(r"lr([\d.]+e-?\d+)", name)
    h_m = re.search(r"h(\d+)", name)
    l_m = re.search(r"_l(\d+)_", name)
    d_m = re.search(r"_d(\d+)$", name)
    if all([lr_m, h_m, l_m, d_m]):
        hyper_results[(lr_m.group(1), h_m.group(1), l_m.group(1), d_m.group(1))] = df[
            "val_loss"
        ].min()

# 2) Multiseed: (var, arch, seed) → list of val_loss per fold
ms_dir = Path("/p/project1/hai_1127/radin1/exprecursors/experiments/multiseed")
ms_results = defaultdict(list)
for run_dir in sorted(ms_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    mf = run_dir / "logs/version_0/metrics.csv"
    if not mf.exists():
        continue
    df = pd.read_csv(mf)
    if "val_loss" not in df.columns:
        continue
    name = run_dir.name
    var = parse_var_from_name(name)
    arch = parse_arch_from_name(name)
    seed_m = [p for p in name.split("_") if p.startswith("seed")]
    seed = seed_m[0].replace("seed", "") if seed_m else "?"
    ms_results[(var, arch, seed)].append(df["val_loss"].min())

# 3) Architecture_variables (includes Atm, seed42 only)
arch_dir = Path(
    "/p/project1/hai_1127/radin1/exprecursors/experiments/architecture_variables"
)
av_results = defaultdict(list)
for run_dir in sorted(arch_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    mf = run_dir / "logs/version_0/metrics.csv"
    if not mf.exists():
        continue
    df = pd.read_csv(mf)
    if "val_loss" not in df.columns:
        continue
    name = run_dir.name
    var = parse_var_from_name(name)
    arch = parse_arch_from_name(name)
    av_results[(var, arch)].append(df["val_loss"].min())

# 4) Merge: for TbotAtm and SSTAtm prefer multiseed (more data), for Atm use arch_variables
all_results = defaultdict(list)  # (var, arch, seed) → list of val_loss
for key, vals in ms_results.items():
    all_results[key].extend(vals)
# Add Atm from arch_variables as seed='42'
for (var, arch), vals in av_results.items():
    if var == "Atm":
        all_results[(var, arch, "42")].extend(vals)

# 5) Final model test metrics
test_folds = []
for fold in range(5):
    npz_path = f"/p/project1/hai_1127/radin1/exprecursors/experiments/partition/TbotAtm_full_mse_v2_seed42_fold{fold}/test_predictions.npz"
    if Path(npz_path).exists():
        npz = np.load(npz_path, allow_pickle=True)
        test_folds.append(
            {
                "r": float(npz["r"]),
                "mae": float(npz["mae_degC"]),
                "n": len(npz["dates"]),
            }
        )

# ═══════════════════════════════════════════════════════
# Plot A: Architecture comparison
# ═══════════════════════════════════════════════════════
fig_a, ax_a = plt.subplots(figsize=(8, 5))

categories = ["Atm", "Tbot+Atm", "SST+Atm"]
arch_types = ["CNN+LSTM", "CNN+LSTM+Attn", "CNN+Attention", "ConvLSTM"]
colors_arch = ["#45B7D1", "#4ECDC4", "#96CEB4", "#F7DC6F"]
x = np.arange(len(categories))
width = 0.18

for i, (arch, color) in enumerate(zip(arch_types, colors_arch)):
    means = []
    stds = []
    for cat in categories:
        # For Atm, use seed='42' from arch_variables; for others use multiseed
        vals = all_results.get((cat, arch, "42"), [])
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals) if vals else np.nan)
    ax_a.bar(
        x + i * width,
        means,
        width,
        yerr=stds,
        label=arch,
        color=color,
        alpha=0.85,
        capsize=3,
    )

ax_a.set_ylabel("Best Validation Loss")
ax_a.set_title("Architecture & Input Variables (seed=42)\n(5-fold mean ± std)")
ax_a.set_xticks(x + 1.5 * width)
ax_a.set_xticklabels(categories, fontsize=10)
ax_a.legend(title="Architecture", fontsize=9)
ax_a.grid(axis="y", alpha=0.3)
ax_a.set_ylim(bottom=0)
fig_a.tight_layout()
fig_a.savefig(OUT / "plot_A_architecture_comparison.png")
print("Saved: plot_A_architecture_comparison.png")

# ═══════════════════════════════════════════════════════
# Plot B: Hypersearch heatmap
# ═══════════════════════════════════════════════════════
lrs = ["1e-04", "5e-04", "1e-03"]
col_labels = []
for h in ["256", "512"]:
    for nlayers in ["2", "4"]:
        for d in ["2", "3", "4"]:
            col_labels.append(f"{nlayers}-{d}")
matrix_rows = []
for lr in lrs:
    row = []
    for h in ["256", "512"]:
        for nlayers in ["2", "4"]:
            for d in ["2", "3", "4"]:
                row.append(hyper_results.get((lr, h, nlayers, d), np.nan))
    matrix_rows.append(row)

matrix = np.array(matrix_rows)

fig_b, (ax_b, cax_b) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 0.05]}
)
im = ax_b.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0.22, vmax=0.35)
ax_b.set_xticks(range(len(col_labels)))
ax_b.set_xticklabels(col_labels, fontsize=10)
ax_b.set_yticks(range(len(lrs)))
ax_b.set_yticklabels([f"lr={lr}" for lr in lrs], fontsize=11)
ax_b.text(2.5, -1.2, "h=256", ha="center", fontsize=12, fontweight="bold")
ax_b.text(8.5, -1.2, "h=512", ha="center", fontsize=12, fontweight="bold")
ax_b.set_title("Hyperparameter Search — TbotAtm, CNN+LSTM+Attn, fold0", fontsize=13)
for i in range(len(lrs)):
    for j in range(matrix.shape[1]):
        if not np.isnan(matrix[i, j]):
            ax_b.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if matrix[i, j] > 0.30 else "black",
            )
plt.colorbar(im, cax=cax_b, label="val_loss")
ax_b.axvline(x=5.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
fig_b.tight_layout()
fig_b.savefig(OUT / "plot_B_hypersearch_heatmap.png")
print("Saved: plot_B_hypersearch_heatmap.png")

# ═══════════════════════════════════════════════════════
# Plot C: Seed stability (boxplot)
# ═══════════════════════════════════════════════════════
seeds = ["42", "123", "456", "789", "1337"]
fig_c, ax_c = plt.subplots(figsize=(10, 5))

plot_data = defaultdict(list)
for var in ["Tbot+Atm", "SST+Atm"]:
    for arch in arch_types:
        for seed in seeds:
            vals = all_results.get((var, arch, seed), [])
            if vals:
                plot_data[(var, arch)].append(np.mean(vals))

group_labels = []
group_data = []
group_colors = []
for var in ["Tbot+Atm", "SST+Atm"]:
    for i, arch in enumerate(arch_types):
        label = f"{var}\n{arch}"
        group_labels.append(label)
        group_data.append(plot_data.get((var, arch), []))
        group_colors.append(colors_arch[i])

bp = ax_c.boxplot(group_data, tick_labels=group_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], group_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_c.set_ylabel("Mean Val Loss (across seeds)")
ax_c.set_title("Seed Stability — 5 seeds × 5 folds\n(boxes show variance across seeds)")
ax_c.grid(axis="y", alpha=0.3)
ax_c.set_ylim(bottom=0)
plt.xticks(rotation=0, fontsize=8)
fig_c.tight_layout()
fig_c.savefig(OUT / "plot_C_seed_stability.png")
print("Saved: plot_C_seed_stability.png")

# ═══════════════════════════════════════════════════════
# Plot D: Summary table
# ═══════════════════════════════════════════════════════
fig_d, ax_d = plt.subplots(figsize=(12, 4.5))
ax_d.axis("off")

mae_arr = np.array([f["mae"] for f in test_folds])
r_arr = np.array([f["r"] for f in test_folds])

table_data = [
    ["Phase", "Key Decision", "Val Loss", "Test MAE (°C)", "Test r"],
    [
        "1. Hypersearch",
        "Best: lr=1e-4, h=512, l=2, d=2\n(TbotAtm, CNN+LSTM+Attn, fold0)",
        "0.2225",
        "—",
        "—",
    ],
    [
        "2. Architecture",
        "CNN+LSTM wins over CNN+LSTM+Attn,\nAttention, ConvLSTM",
        "0.178–0.217",
        "—",
        "0.734–0.860",
    ],
    [
        "3. Seed stability",
        "5 seeds (42,123,456,789,1337)\nCNN+LSTM consistent across seeds",
        "0.179–0.218",
        "—",
        "0.856–0.872",
    ],
    [
        "4. Final model",
        "CNN-LSTM, MSE, full partition\n5-fold CV, seed=42",
        "—",
        f"{mae_arr.mean():.3f}±{mae_arr.std():.3f}",
        f"{r_arr.mean():.3f}±{r_arr.std():.3f}",
    ],
]

table = ax_d.table(
    cellText=table_data,
    loc="center",
    cellLoc="center",
    colWidths=[0.15, 0.35, 0.14, 0.18, 0.13],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

for j in range(5):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

for i in range(1, 5):
    table[i, 0].set_facecolor("#ECF0F1")
    table[i, 0].set_text_props(fontweight="bold")

ax_d.set_title("Experiment Summary", fontsize=14, fontweight="bold", pad=20)
fig_d.tight_layout()
fig_d.savefig(OUT / "plot_D_summary_table.png")
print("Saved: plot_D_summary_table.png")

# ═══════════════════════════════════════════════════════
# Combined: A + B
# ═══════════════════════════════════════════════════════
fig_ab, (ax_ab1, ax_ab2) = plt.subplots(1, 2, figsize=(16, 5))

for i, (arch, color) in enumerate(zip(arch_types, colors_arch)):
    means = [
        np.mean(all_results.get((cat, arch, "42"), [np.nan])) for cat in categories
    ]
    stds = [np.std(all_results.get((cat, arch, "42"), [np.nan])) for cat in categories]
    ax_ab1.bar(
        x + i * width,
        means,
        width,
        yerr=stds,
        label=arch,
        color=color,
        alpha=0.85,
        capsize=3,
    )
ax_ab1.set_ylabel("Best Validation Loss")
ax_ab1.set_title("A) Architecture & Variables (seed=42)")
ax_ab1.set_xticks(x + 1.5 * width)
ax_ab1.set_xticklabels(categories)
ax_ab1.legend(fontsize=8)
ax_ab1.grid(axis="y", alpha=0.3)
ax_ab1.set_ylim(bottom=0)

ax_ab2.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0.22, vmax=0.35)
ax_ab2.set_xticks(range(len(col_labels)))
ax_ab2.set_xticklabels(col_labels, fontsize=8)
ax_ab2.set_yticks(range(len(lrs)))
ax_ab2.set_yticklabels([f"lr={lr}" for lr in lrs])
ax_ab2.text(2.5, -0.9, "h=256", ha="center", fontsize=9, fontweight="bold")
ax_ab2.text(8.5, -0.9, "h=512", ha="center", fontsize=9, fontweight="bold")
ax_ab2.set_xlabel("layers - depth")
ax_ab2.set_title("B) Hyperparameter Search (TbotAtm, CNN+LSTM+Attn)")
for i in range(len(lrs)):
    for j in range(matrix.shape[1]):
        if not np.isnan(matrix[i, j]):
            ax_ab2.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color="white" if matrix[i, j] > 0.30 else "black",
            )

fig_ab.tight_layout()
fig_ab.savefig(OUT / "combined_AB.png")
print("Saved: combined_AB.png")

# ═══════════════════════════════════════════════════════
# Combined: C + D
# ═══════════════════════════════════════════════════════
fig_cd, (ax_cd1, ax_cd2) = plt.subplots(
    1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.2, 1]}
)

bp2 = ax_cd1.boxplot(
    group_data, tick_labels=group_labels, patch_artist=True, widths=0.6
)
for patch, color in zip(bp2["boxes"], group_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax_cd1.set_ylabel("Mean Val Loss (across seeds)")
ax_cd1.set_title("C) Seed Stability (5 seeds × 5 folds)")
ax_cd1.grid(axis="y", alpha=0.3)
ax_cd1.set_ylim(bottom=0)
plt.sca(ax_cd1)
plt.xticks(rotation=0, fontsize=8)

ax_cd2.axis("off")
table2 = ax_cd2.table(
    cellText=table_data,
    loc="center",
    cellLoc="center",
    colWidths=[0.15, 0.35, 0.14, 0.18, 0.13],
)
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1.0, 1.8)
for j in range(5):
    table2[0, j].set_facecolor("#2C3E50")
    table2[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, 5):
    table2[i, 0].set_facecolor("#ECF0F1")
    table2[i, 0].set_text_props(fontweight="bold")
ax_cd2.set_title("D) Experiment Summary", fontsize=12, fontweight="bold", pad=20)

fig_cd.tight_layout()
fig_cd.savefig(OUT / "combined_CD.png")
print("Saved: combined_CD.png")

print("\nDone!")
