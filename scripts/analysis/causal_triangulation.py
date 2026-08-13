"""
causal_triangulation.py — Granger causality + CCM for convergent validity.

Tests whether candidate drivers (Tbot_NS, Tbot_GS, u10_NS, msl_NS, v10_NS, ssr_NS)
Granger-cause and CCM-cause NS to_anom (the model target).

Output: triangulation_results.csv + triangulation_matrix.png

Usage (CPU, login node):
  python eval/causal_triangulation.py
"""

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

from src.utils.paths import (
    DATA_FILE as DATA_FILE_ENV,
)
from src.utils.paths import (
    EXPERIMENTS_DIR,
)

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import detrend
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.stattools import grangercausalitytests

DATA_FILE = DATA_FILE_ENV
OUT_DIR = Path(str(EXPERIMENTS_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid boxes (lat/lon slices, 0.5° grid starting at 0N/80W)
NS_LAT = slice(100, 127)  # 50.0–63.0°N
NS_LON = slice(150, 187)  # -5.0–13.0°E
GS_LAT = slice(70, 95)  # 35.0–47.0°N
GS_LON = slice(10, 71)  # -75.0– -45.0°E

REF_START = "1985-01-01"
REF_END = "2024-12-31"
MAX_LAG = 30  # Granger max lag (days)
CCM_E = 4  # CCM embedding dimension
CCM_TAU = 1  # CCM time delay


# ── Data extraction ──────────────────────────────────────────────────────────


def box_mean(da, lat_sl, lon_sl):
    """Spatial mean over a box, ignoring NaN (land)."""
    sub = da.isel(lat=lat_sl, lon=lon_sl)
    return sub.mean(dim=["lat", "lon"], skipna=True).values


def load_series():
    print("Loading data...")
    ds = xr.open_dataset(DATA_FILE).sel(time=slice(REF_START, REF_END))
    target = ds["target"].values  # NS-averaged to_anom
    tbot_ns = box_mean(ds["ptho_bot"], NS_LAT, NS_LON)
    tbot_gs = box_mean(ds["ptho_bot"], GS_LAT, GS_LON)
    u10_ns = box_mean(ds["u10"], NS_LAT, NS_LON)
    v10_ns = box_mean(ds["v10"], NS_LAT, NS_LON)
    msl_ns = box_mean(ds["msl"], NS_LAT, NS_LON)
    ssr_ns = box_mean(ds["ssr"], NS_LAT, NS_LON)
    ds.close()
    print(f"  Loaded {len(target)} days")
    return {
        "target": target,
        "Tbot_NS": tbot_ns,
        "Tbot_GS": tbot_gs,
        "u10_NS": u10_ns,
        "v10_NS": v10_ns,
        "msl_NS": msl_ns,
        "ssr_NS": ssr_ns,
    }


def preprocess(series: dict) -> dict:
    """Detrend (linear) + standardise each series. Remove NaN rows."""
    out = {}
    for k, v in series.items():
        v2 = detrend(v, type="linear")
        out[k] = (v2 - v2.mean()) / (v2.std() + 1e-12)
    return out


# ── Granger causality ────────────────────────────────────────────────────────


def granger_pvalue(X: np.ndarray, Y: np.ndarray, maxlag: int) -> float:
    """
    Test whether X Granger-causes Y.
    Returns the minimum p-value across lags 1..maxlag (F-test).
    """
    data = np.column_stack([Y, X])
    try:
        res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)]
        return float(np.min(pvals)), int(np.argmin(pvals) + 1)
    except Exception as e:
        print(f"  Granger error: {e}")
        return 1.0, -1


# ── Convergent Cross Mapping ─────────────────────────────────────────────────


def time_delay_embed(x: np.ndarray, E: int, tau: int) -> np.ndarray:
    """Returns (N - (E-1)*tau, E) delay embedding matrix."""
    N = len(x)
    L = N - (E - 1) * tau
    M = np.zeros((L, E))
    for i in range(E):
        M[:, i] = x[(E - 1 - i) * tau : (E - 1 - i) * tau + L]
    return M


def ccm_rho(
    X: np.ndarray, Y: np.ndarray, E: int, tau: int, lib_sizes: np.ndarray
) -> np.ndarray:
    """
    CCM: test whether X causes Y.
    Uses the manifold of Y to predict X (Sugihara 2012 convention).
    Returns Pearson r at each library size.
    """
    My = time_delay_embed(Y, E, tau)
    Mx = time_delay_embed(X, E, tau)
    L = len(My)
    rhos = []

    nn = NearestNeighbors(n_neighbors=E + 1, algorithm="ball_tree")
    nn.fit(My)

    for lib in lib_sizes:
        lib = min(lib, L)
        idx = np.random.choice(L, lib, replace=False)
        idx_lib = np.sort(idx)
        My_lib = My[idx_lib]
        nn.fit(My_lib)

        # For all points, find E+1 nearest neighbours in library
        all_pts = np.arange(L)
        dists, inds = nn.kneighbors(My[all_pts])

        # Weights: exponential kernel
        u = dists / (dists[:, 0:1] + 1e-12)
        w = np.exp(-u)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)

        # Predicted X values from Y manifold
        Mx_pred_all = Mx[all_pts][:, 0]
        Mx_lib_vals = Mx[idx_lib][:, 0]
        x_hat = (w * Mx_lib_vals[inds]).sum(axis=1)

        r, _ = stats.pearsonr(Mx_pred_all, x_hat)
        rhos.append(r)

    return np.array(rhos)


def ccm_test(
    X: np.ndarray,
    Y: np.ndarray,
    E: int = CCM_E,
    tau: int = CCM_TAU,
    n_lib: int = 10,
    n_boot: int = 20,
    seed: int = 42,
) -> tuple:
    """
    Returns (rho_max, converges: bool).
    Convergence = rho increases significantly with library size.
    """
    L = len(time_delay_embed(Y, E, tau))
    lib_sizes = np.unique(np.linspace(E + 2, L, n_lib, dtype=int))

    rho_curves = []
    for _ in range(n_boot):
        rhos = ccm_rho(X, Y, E, tau, lib_sizes)
        rho_curves.append(rhos)

    rho_mean = np.mean(rho_curves, axis=0)
    rho_max = float(rho_mean[-1])

    # Convergence: last 25% of lib sizes has higher rho than first 25%
    q1 = rho_mean[: len(lib_sizes) // 4].mean()
    q4 = rho_mean[-len(lib_sizes) // 4 :].mean()
    converges = bool(q4 > q1 + 0.05) and rho_max > 0.1

    return rho_max, converges, rho_mean, lib_sizes


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    raw = load_series()
    series = preprocess(raw)
    Y = series["target"]

    drivers = ["Tbot_NS", "Tbot_GS", "u10_NS", "v10_NS", "msl_NS", "ssr_NS"]

    results = []
    print(
        f"\n{'Driver':<12} {'Granger p':>10} {'Granger lag':>12} {'CCM rho':>10} {'Converges':>10}"
    )
    print("-" * 60)

    for drv in drivers:
        X = series[drv]

        # Granger
        g_pval, g_lag = granger_pvalue(X, Y, MAX_LAG)

        # CCM (X causes Y → use manifold of Y to predict X)
        ccm_rho_max, ccm_conv, ccm_curve, lib_sz = ccm_test(X, Y)

        sig_g = (
            "***"
            if g_pval < 0.001
            else ("**" if g_pval < 0.01 else ("*" if g_pval < 0.05 else "ns"))
        )
        sig_ccm = "YES" if ccm_conv else "no"
        print(
            f"{drv:<12} {g_pval:>10.4f} {g_lag:>12d} {ccm_rho_max:>10.3f} {sig_ccm:>10}  {sig_g}"
        )

        results.append(
            {
                "driver": drv,
                "granger_p": g_pval,
                "granger_lag": g_lag,
                "granger_sig": sig_g,
                "ccm_rho": ccm_rho_max,
                "ccm_conv": ccm_conv,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "triangulation_results.csv", index=False)
    print("\nSaved: triangulation_results.csv")

    # ── Plot convergence matrix ───────────────────────────────────────────────
    methods = ["Granger\ncausality", "CCM\nconvergence"]
    n_drv = len(drivers)
    matrix = np.zeros((len(methods), n_drv))
    # Granger: -log10(p) clipped at 4
    for j, row in enumerate(results):
        matrix[0, j] = min(-np.log10(row["granger_p"] + 1e-10), 4)
        matrix[1, j] = row["ccm_rho"] if row["ccm_conv"] else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

    for ax, method_idx, title, fmt in [
        (axes[0], 0, "Granger: −log₁₀(p)", ".2f"),
        (axes[1], 1, "CCM: ρ (converging)", ".3f"),
    ]:
        data = matrix[method_idx : method_idx + 1]
        im = ax.imshow(
            data, cmap="YlOrRd", vmin=0, vmax=4 if method_idx == 0 else 1, aspect="auto"
        )
        ax.set_xticks(range(n_drv))
        ax.set_xticklabels(drivers, rotation=30, ha="right", fontsize=10)
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.05)
        for j in range(n_drv):
            ax.text(j, 0, f"{data[0, j]:{fmt}}", ha="center", va="center", fontsize=9)

    fig.suptitle(
        "Causal triangulation: Granger + CCM\nTarget = NS to_anom", fontsize=12
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "triangulation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: triangulation_matrix.png")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
