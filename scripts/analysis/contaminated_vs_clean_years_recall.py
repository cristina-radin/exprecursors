"""
contaminated_vs_clean_years_recall.py -- Aug 24 2026.

Post-hoc check (no GPU, no retraining) for the year-boundary leakage
question raised during meeting prep: `stratified_kfold` assigns whole
calendar years to train/val/test, with no buffer around Dec31->Jan1.
17/39 year-boundaries in the raw 40-yr series have a REAL MHW event
(def2, area>=5%) active on both Dec31 and Jan1 -- of those, 4-8 per fold
land on a boundary between test and a differently-labelled neighbour
(known_issues.md candidate, not yet numbered).

This script does NOT decide whether that matters -- it measures it:
recomputes def2 recall (quantile head) for `full_gnll_quantile_v2_landfill`
(the committed model, all 5 folds pooled), split into "contaminated" test
years (share a real straddling event with a differently-labelled
neighbour) vs "clean" test years (the rest). Reuses
`eval_recall_v2_partition.run_fold()` unchanged (same methodology as the
project's other recall numbers) -- does not reimplement inference.

Usage (CPU, submit via scripts/slurm/_contaminated_vs_clean_recall.sh):
  python scripts/analysis/contaminated_vs_clean_years_recall.py
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_recall_v2_partition import (  # noqa: E402
    AREA_FRAC_THRESHOLD,
    FIGURES_DIR,
    MASK_FNS,
    run_fold,
)
from src.utils.hobday import load_ns_p90  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402

CONFIG_DIR = "full_gnll_quantile_v2_landfill"

FOLD_YEARS = {
    0: dict(
        train=[
            1987,
            1988,
            1989,
            1990,
            1992,
            1994,
            1995,
            1996,
            1999,
            2001,
            2003,
            2004,
            2007,
            2008,
            2009,
            2011,
            2012,
            2013,
            2015,
            2016,
            2020,
            2021,
            2023,
            2024,
        ],
        val=[1986, 1993, 1997, 1998, 2006, 2010, 2019, 2022],
        test=[1985, 1991, 2000, 2002, 2005, 2014, 2017, 2018],
    ),
    1: dict(
        train=[
            1985,
            1988,
            1989,
            1990,
            1991,
            1995,
            1996,
            1999,
            2000,
            2001,
            2002,
            2003,
            2005,
            2008,
            2009,
            2012,
            2013,
            2014,
            2015,
            2017,
            2018,
            2020,
            2021,
            2023,
        ],
        val=[1987, 1992, 1994, 2004, 2007, 2011, 2016, 2024],
        test=[1986, 1993, 1997, 1998, 2006, 2010, 2019, 2022],
    ),
    2: dict(
        train=[
            1985,
            1986,
            1989,
            1990,
            1991,
            1993,
            1996,
            1997,
            1998,
            2000,
            2001,
            2002,
            2003,
            2005,
            2006,
            2009,
            2010,
            2013,
            2014,
            2015,
            2017,
            2018,
            2019,
            2022,
        ],
        val=[1988, 1995, 1999, 2008, 2012, 2020, 2021, 2023],
        test=[1987, 1992, 1994, 2004, 2007, 2011, 2016, 2024],
    ),
    3: dict(
        train=[
            1985,
            1986,
            1987,
            1991,
            1992,
            1993,
            1994,
            1997,
            1998,
            2000,
            2002,
            2004,
            2005,
            2006,
            2007,
            2010,
            2011,
            2014,
            2016,
            2017,
            2018,
            2019,
            2022,
            2024,
        ],
        val=[1989, 1990, 1996, 2001, 2003, 2009, 2013, 2015],
        test=[1988, 1995, 1999, 2008, 2012, 2020, 2021, 2023],
    ),
    4: dict(
        train=[
            1986,
            1987,
            1988,
            1992,
            1993,
            1994,
            1995,
            1997,
            1998,
            1999,
            2004,
            2006,
            2007,
            2008,
            2010,
            2011,
            2012,
            2016,
            2019,
            2020,
            2021,
            2022,
            2023,
            2024,
        ],
        val=[1985, 1991, 2000, 2002, 2005, 2014, 2017, 2018],
        test=[1989, 1990, 1996, 2001, 2003, 2009, 2013, 2015],
    ),
}


def real_event_boundaries(area_frac, ds):
    years_arr = ds.time.dt.year.values
    months = ds.time.dt.month.values
    days = ds.time.dt.day.values
    idx = {
        (int(y), int(m), int(d)): i
        for i, (y, m, d) in enumerate(zip(years_arr, months, days))
    }
    mhw = area_frac >= AREA_FRAC_THRESHOLD
    boundaries = set()
    for y in range(1985, 2024):
        i_dec31 = idx[(y, 12, 31)]
        i_jan1 = idx[(y + 1, 1, 1)]
        if mhw[i_dec31] and mhw[i_jan1]:
            boundaries.add((y, y + 1))
    return boundaries


def contaminated_test_years(fold_years, boundaries):
    role = {}
    for y in fold_years["train"]:
        role[y] = "train"
    for y in fold_years["val"]:
        role[y] = "val"
    for y in fold_years["test"]:
        role[y] = "test"
    contaminated = set()
    for y1, y2 in boundaries:
        r1, r2 = role.get(y1), role.get(y2)
        if r1 != r2 and "test" in (r1, r2):
            contaminated.add(y1 if r1 == "test" else y2)
    return contaminated


def def2_recall(trues_c, q_c, thresh1, area_frac_c):
    ext2 = area_frac_c >= AREA_FRAC_THRESHOLD
    n2 = int(ext2.sum())
    if n2 == 0:
        return float("nan"), 0
    recall = (q_c[ext2] > thresh1[ext2]).mean()
    return recall, n2


def main():
    print("device selection handled inside run_fold()", flush=True)
    p90 = load_ns_p90()
    area_frac = np.load(FIGURES_DIR / "area_frac_timeseries.npy")
    ds = xr.open_dataset(DATA_FILE)
    boundaries = real_event_boundaries(area_frac, ds)
    print(
        f"Real Dec31->Jan1 event-straddling boundaries in raw series: {len(boundaries)}/39",
        flush=True,
    )

    all_trues, all_q, all_thresh1, all_area_frac, all_years = [], [], [], [], []
    all_contaminated_test_years = {}

    for fold in range(5):
        trues_c, mean_c, q_c, thresh1, area_frac_test, years_test = run_fold(
            CONFIG_DIR, fold, MASK_FNS["full"], p90, area_frac, return_years=True
        )
        all_trues.append(trues_c)
        all_q.append(q_c)
        all_thresh1.append(thresh1)
        all_area_frac.append(area_frac_test)
        all_years.append(years_test)
        cty = contaminated_test_years(FOLD_YEARS[fold], boundaries)
        all_contaminated_test_years[fold] = cty
        print(f"  fold {fold}: contaminated test years = {sorted(cty)}", flush=True)
        # incremental save (CLAUDE.md rule)
        np.savez(
            FIGURES_DIR / "_fold_cache" / "contaminated_vs_clean_partial.npz",
            trues_c=np.concatenate(all_trues),
            q_c=np.concatenate(all_q),
            thresh1=np.concatenate(all_thresh1),
            area_frac_c=np.concatenate(all_area_frac),
            years=np.concatenate(all_years),
            n_folds_done=fold + 1,
        )

    trues_c = np.concatenate(all_trues)
    q_c = np.concatenate(all_q)
    thresh1 = np.concatenate(all_thresh1)
    area_frac_c = np.concatenate(all_area_frac)
    years_c = np.concatenate(all_years)

    is_contaminated = np.zeros(len(years_c), dtype=bool)
    offset = 0
    for fold in range(5):
        n = len(all_years[fold])
        cty = all_contaminated_test_years[fold]
        is_contaminated[offset : offset + n] = np.isin(all_years[fold], list(cty))
        offset += n

    recall_contam, n_contam = def2_recall(
        trues_c[is_contaminated],
        q_c[is_contaminated],
        thresh1[is_contaminated],
        area_frac_c[is_contaminated],
    )
    recall_clean, n_clean = def2_recall(
        trues_c[~is_contaminated],
        q_c[~is_contaminated],
        thresh1[~is_contaminated],
        area_frac_c[~is_contaminated],
    )
    recall_all, n_all = def2_recall(trues_c, q_c, thresh1, area_frac_c)

    print(
        "\n=== SUMMARY: def2 quantile-head recall, contaminated vs clean test years ==="
    )
    print(f"  ALL test years   : recall={recall_all*100:.1f}%  (n_extreme={n_all})")
    print(
        f"  CONTAMINATED yrs : recall={recall_contam*100:.1f}%  (n_extreme={n_contam})"
    )
    print(f"  CLEAN yrs        : recall={recall_clean*100:.1f}%  (n_extreme={n_clean})")
    print(f"  delta (contam - clean): {(recall_contam-recall_clean)*100:+.1f} pp")

    np.savez(
        FIGURES_DIR / "_fold_cache" / "contaminated_vs_clean_final.npz",
        trues_c=trues_c,
        q_c=q_c,
        thresh1=thresh1,
        area_frac_c=area_frac_c,
        years=years_c,
        is_contaminated=is_contaminated,
    )
    print(
        "\nSaved final arrays to experiments/figures/_fold_cache/contaminated_vs_clean_final.npz"
    )


if __name__ == "__main__":
    main()
