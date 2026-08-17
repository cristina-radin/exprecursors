"""
mhw_onset_skill.py — Evaluación condicional de skill en onset de MHW.

Categorías (pixel a pixel):
  all        : todos los samples
  onset      : target > 0  AND  to_anom(t) <= 0  (primer día MHW — persistencia ciega)
  mid_event  : target > 0  AND  to_anom(t) > 0   (MHW en curso — persistencia lo ve)
  no_mhw     : target <= 0

Pearson r: computado por fold, promediado entre folds (igual que r_map.npy).
Persistence: to_anom(t) como forecast de to_anom(t+7) — cargado por fold lazy.

Outputs en --output_dir:
  onset_skill_table.txt
  onset_skill_maps.png        — mapas de advantage (model - persist) por categoria
  onset_skill_ns_SSTAtm.png
  onset_skill_ns_TbotAtm.png
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import xarray as xr

DATA_NC = "/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily.nc"
RUNS_DIR = Path("/p/project1/hai_1127/radin1/spatial_forecast/experiments/runs")
VARSETS = ["Atm", "SSTAtm", "TbotAtm"]
N_FOLDS = 5
WINDOW = 60
LEAD = 7
NS_BOX = dict(lat_min=50, lat_max=63, lon_min=-5, lon_max=13)


def get_test_idx(yrs_arr, fold):
    uniq = np.sort(np.unique(yrs_arr))
    rng = np.random.default_rng(0)
    shuf = rng.permutation(uniq)
    grp = np.array_split(shuf, N_FOLDS)
    test_yrs = set(grp[fold].tolist())
    total = len(yrs_arr) - WINDOW - LEAD + 1
    tgt_yrs = np.array([int(yrs_arr[i + WINDOW - 1 + LEAD]) for i in range(total)])
    return np.where(np.isin(tgt_yrs, list(test_yrs)))[0]


def pearson_r_pixelwise(pred, tgt, mask_3d, min_samples=5):
    """
    pred, tgt : (N, H, W)
    mask_3d   : (N, H, W) bool — which samples to use
    Returns   : (H, W) Pearson r, NaN where n < min_samples
    """
    x = np.where(mask_3d, pred, np.nan)
    y = np.where(mask_3d, tgt, np.nan)
    n = mask_3d.sum(axis=0)
    xm = np.nanmean(x, axis=0)
    ym = np.nanmean(y, axis=0)
    cov = np.nanmean((x - xm) * (y - ym), axis=0)
    r = cov / (np.nanstd(x, axis=0) * np.nanstd(y, axis=0) + 1e-9)
    r[n < min_samples] = np.nan
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="eval/figures")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-open dataset — metadata only until .values is called
    ds = xr.open_dataset(DATA_NC)
    ocean = ds["land_mask"].values.astype(bool)
    lat = ds.lat.values
    lon = ds.lon.values
    yrs = ds.time.dt.year.values

    H, W = ocean.shape
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    def region_mask(lat_min, lat_max, lon_min, lon_max):
        return (
            np.outer(
                (lat >= lat_min) & (lat <= lat_max), (lon >= lon_min) & (lon <= lon_max)
            )
            & ocean
        )

    ns = region_mask(**NS_BOX)
    ns_n = region_mask(57, 63, -5, 13)
    ns_s = region_mask(50, 57, -5, 13)

    cats = ["all", "onset", "mid_event", "no_mhw"]
    results = {}

    for vs in VARSETS:
        print(f"\n=== {vs} ===")
        r_model_folds = {c: [] for c in cats}
        r_persist_folds = {c: [] for c in cats}

        for fold in range(N_FOLDS):
            run = RUNS_DIR / f"{vs}_seed42_fold{fold}"
            pred = np.load(run / "test_preds.npy")  # (N, H, W) normalised
            tgt = np.load(run / "test_targets.npy")  # (N, H, W) normalised

            # Sample indices and input-end time indices
            test_idx = get_test_idx(yrs, fold)
            assert len(test_idx) == len(pred), f"len mismatch fold{fold}"
            t_idx = test_idx + WINDOW - 1  # time index of input end (t)

            # Load to_anom at time t — lazy, only needed slices
            print(
                f"  fold {fold}: loading {len(t_idx)} time slices ...",
                end=" ",
                flush=True,
            )
            to_t = ds["to_anom"].isel(time=t_idx).values  # (N, H, W) raw °C
            print("done")

            # Persistence forecast: to_anom(t) — for Pearson r, scale doesn't matter
            persist = to_t  # (N, H, W)

            # MHW flags (pixel-wise)
            is_mhw_tgt = tgt > 0  # (N, H, W)
            is_mhw_inp = to_t > 0  # (N, H, W)

            masks = {
                "all": np.ones((len(pred), H, W), dtype=bool),
                "onset": is_mhw_tgt & ~is_mhw_inp,
                "mid_event": is_mhw_tgt & is_mhw_inp,
                "no_mhw": ~is_mhw_tgt,
            }
            for c in cats:
                masks[c] &= ocean[None]

            # Diagnostics
            ns3 = ns[None]
            for c in ["onset", "mid_event"]:
                n = (masks[c] & ns3).sum()
                print(f"    {c}: {n:,} NS pixel-samples")

            for c in cats:
                rm = pearson_r_pixelwise(pred, tgt, masks[c])
                rp = pearson_r_pixelwise(persist, tgt, masks[c])
                rm[~ocean] = np.nan
                rp[~ocean] = np.nan
                r_model_folds[c].append(rm)
                r_persist_folds[c].append(rp)

        # Average across folds
        r_model = {c: np.nanmean(r_model_folds[c], axis=0) for c in cats}
        r_persist = {c: np.nanmean(r_persist_folds[c], axis=0) for c in cats}
        results[vs] = {"rm": r_model, "rp": r_persist}

        # Print NS summary
        print(
            f"\n  {'Category':<12} {'Model_r':>9} {'Persist_r':>10} {'Advantage':>10}"
        )
        for c in cats:
            for rname, rmask in [
                ("NS total", ns),
                ("NS North", ns_n),
                ("NS South", ns_s),
            ]:
                rm_v = np.nanmean(r_model[c][rmask])
                rp_v = np.nanmean(r_persist[c][rmask])
                flag = " ← WINS" if rm_v > rp_v else ""
                print(
                    f"  {c:<12} {rname:<10} model={rm_v:.4f}  persist={rp_v:.4f}  "
                    f"adv={rm_v-rp_v:+.4f}{flag}"
                )

    ds.close()

    # ── Figures ───────────────────────────────────────────────────────────────
    def ns_rect():
        return mpatches.Rectangle(
            (-5, 50), 18, 13, fill=False, edgecolor="black", lw=1.5, ls="--"
        )

    # 6-panel: advantage maps for 3 categories × 3 varsets
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    cat_rows = ["onset", "mid_event", "all"]
    for row, c in enumerate(cat_rows):
        for col, vs in enumerate(VARSETS):
            ax = axes[row, col]
            adv = results[vs]["rm"][c] - results[vs]["rp"][c]
            im = ax.imshow(
                adv,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-0.3,
                vmax=0.3,
                aspect="auto",
            )
            ax.add_patch(ns_rect())
            ax.set_title(f"{vs} — {c}\nmodel − persistence", fontsize=9)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Model − Persistence advantage by MHW category\n"
        "Red = model beats persistence | Blue = persistence wins",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "onset_skill_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved onset_skill_maps.png")

    # NS bar charts for SSTAtm and TbotAtm
    for vs in ["SSTAtm", "TbotAtm"]:
        cat_labels = [
            "All days",
            "Onset\n(input=normal)",
            "Mid-event\n(input=MHW)",
            "Non-MHW",
        ]
        regions = [
            ("NS Total", ns, "steelblue"),
            ("NS North ≥57°N", ns_n, "tomato"),
            ("NS South <57°N", ns_s, "orange"),
        ]
        x = np.arange(len(cats))
        w = 0.22

        fig, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        for ax_i, (ylabel, fn) in enumerate(
            [
                ("Pearson r", lambda rm, rp: rm),
                ("Advantage (model−persistence)", lambda rm, rp: rm - rp),
            ]
        ):
            ax = axes2[ax_i]
            for ri, (rname, rmask, col) in enumerate(regions):
                vals = [
                    fn(
                        np.nanmean(results[vs]["rm"][c][rmask]),
                        np.nanmean(results[vs]["rp"][c][rmask]),
                    )
                    for c in cats
                ]
                ax.bar(x + ri * w, vals, w, label=rname, color=col, alpha=0.85)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x + w)
            ax.set_xticklabels(cat_labels, fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{vs} — {ylabel}")
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / f"onset_skill_ns_{vs}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved onset_skill_ns_{vs}.png")

    # Save table
    with open(out_dir / "onset_skill_table.txt", "w") as f:
        for vs in VARSETS:
            f.write(f"\n=== {vs} ===\n")
            for c in cats:
                for rname, rmask in [
                    ("ocean", ocean),
                    ("NS", ns),
                    ("NS_N", ns_n),
                    ("NS_S", ns_s),
                ]:
                    rm = np.nanmean(results[vs]["rm"][c][rmask])
                    rp = np.nanmean(results[vs]["rp"][c][rmask])
                    f.write(
                        f"  {c:<12} {rname:<8} model={rm:.4f} persist={rp:.4f} "
                        f"adv={rm-rp:+.4f}\n"
                    )
    print("Saved onset_skill_table.txt")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
