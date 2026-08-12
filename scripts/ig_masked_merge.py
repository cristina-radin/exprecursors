"""
ig_masked_merge.py — Merge partial seed results from ig_masked_batched.py
and generate final figures. CPU-only, runs in seconds.

Usage:
  python eval/ig_masked_merge.py --output_dir experiments/figures/xai_masked
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr
from pathlib import Path

ALL_SEEDS = [42, 123, 456, 789, 1337]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/figures/xai_masked")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    # Find all partial files
    partial_files = sorted(out_dir.glob("ig_partial_seed*.npz"))
    if not partial_files:
        print(f"No ig_partial_seed*.npz found in {out_dir}"); return
    print(f"Merging {len(partial_files)} seed files: {[f.name for f in partial_files]}")

    spatial_sum  = None
    peryear_sum  = None
    peryear_cnt  = None
    all_years    = None
    variables    = None
    total_runs   = 0

    for pf in partial_files:
        d = np.load(str(pf), allow_pickle=True)
        n   = int(d["n_runs"])
        yrs = d["years"].tolist()

        if spatial_sum is None:
            spatial_sum = d["spatial_sum"].copy()
            variables   = d["variables"].tolist()
            all_years   = yrs
            peryear_sum = d["peryear_mat"].copy()
            peryear_cnt = d["peryear_cnt"].copy()
        else:
            spatial_sum += d["spatial_sum"]

            # Align year axes
            for i, yr in enumerate(yrs):
                if yr in all_years:
                    yi = all_years.index(yr)
                else:
                    all_years.append(yr); all_years.sort()
                    yi = all_years.index(yr)
                    peryear_sum = np.insert(peryear_sum, yi, 0, axis=0)
                    peryear_cnt = np.insert(peryear_cnt, yi, 0)
                peryear_sum[yi] += d["peryear_mat"][i]
                peryear_cnt[yi] += d["peryear_cnt"][i]

        total_runs += n
        print(f"  {pf.name}: n_runs={n}")

    print(f"\nTotal runs merged: {total_runs}")

    # Normalise
    spatial_avg  = spatial_sum / total_runs
    peryear_avg  = np.where(peryear_cnt[:, None] > 0,
                            peryear_sum / peryear_cnt[:, None], 0.0)
    row_sums     = peryear_avg.sum(axis=1, keepdims=True)
    peryear_pct  = np.where(row_sums > 0, peryear_avg / row_sums * 100, 0.0)

    np.save(str(out_dir / "ig_peryear.npy"), peryear_pct)
    np.save(str(out_dir / "ig_spatial_avg.npy"), spatial_avg)

    # ── Per-year heatmap ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(peryear_pct.T, aspect="auto", cmap="YlOrRd",
                   extent=[all_years[0]-0.5, all_years[-1]+0.5,
                           len(variables)-0.5, -0.5])
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables)
    ax.set_xlabel("Year")
    ax.set_title(f"Variable importance (% |IG|) — MASKED model (NS zeroed)\n"
                 f"Only remote Atlantic + atmosphere  ({total_runs} runs)")
    plt.colorbar(im, ax=ax, label="% of total |IG|")
    plt.tight_layout()
    plt.savefig(str(out_dir / "ig_peryear.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ig_peryear.png")

    # ── Spatial maps ──────────────────────────────────────────────────────────
    ds_meta = xr.open_dataset(
        "/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily.nc"
    )
    lat   = ds_meta.lat.values; lon = ds_meta.lon.values
    ocean = ds_meta["land_mask"].values.astype(bool)
    ds_meta.close()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    ns_lon_start = float(lon[150]) if len(lon) > 186 else -5.0
    ns_lat_start = float(lat[100]) if len(lat) > 126 else 50.0
    ns_lon_width = float(lon[186] - lon[150]) if len(lon) > 186 else 18.0
    ns_lat_height = float(lat[126] - lat[100]) if len(lat) > 126 else 13.0

    for ci, var in enumerate(variables):
        data = spatial_avg[ci].copy()
        data[~ocean] = np.nan
        fig, ax = plt.subplots(figsize=(10, 5))
        vmax = np.nanpercentile(data, 98)
        im = ax.imshow(data, origin="lower", extent=extent,
                       cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ns_rect = mpatches.Rectangle(
            (ns_lon_start, ns_lat_start), ns_lon_width, ns_lat_height,
            fill=False, edgecolor="blue", linewidth=1.5, linestyle="--"
        )
        ax.add_patch(ns_rect)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title(f"Mean |IG| — {var}\nMASKED model (NS zeroed, {total_runs} runs)")
        plt.colorbar(im, ax=ax, label="|attribution|")
        plt.tight_layout()
        plt.savefig(str(out_dir / f"ig_spatial_{var}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved ig_spatial_{var}.png")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    global_imp     = spatial_avg.mean(axis=(1, 2))
    global_imp_pct = global_imp / global_imp.sum() * 100
    colors = ["tomato" if v == "to_anom" else "steelblue" for v in variables]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(variables, global_imp_pct, color=colors)
    ax.set_xlabel("% of total |IG|")
    ax.set_title(f"Variable importance — MASKED model ({total_runs} runs)\n"
                 f"(NS SST zeroed: only remote drivers)")
    plt.tight_layout()
    plt.savefig(str(out_dir / "ig_barplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ig_barplot.png")

    print(f"\n=== Variable importance (global, {total_runs} runs) ===")
    for var, pct in zip(variables, global_imp_pct):
        print(f"  {var:<12} {pct:.1f}%")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
