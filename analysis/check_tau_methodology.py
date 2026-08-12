"""
check_tau_methodology.py — Diagnóstico de dos riesgos metodológicos en el test de inercia térmica.

CHECK 1 — Fuga estacional:
  ¿El ACF de NS to_anom muestra un pico anual (lag≈365) que desplazaría el cruce de 1/e
  por artefacto y no por física?
  Método: comparar τ con el to_anom original vs con ciclo anual eliminado (armónico 365d).
  Si τ cambia poco entre ambos → la climatología es suficientemente suave.
  Si τ cambia mucho → hay fuga estacional que invalida la comparación entre décadas.

CHECK 2 — Heterogeneidad espacial:
  ¿La caída de τ está concentrada en el Norte del NS (zona estratificada, termoclina real)
  o es uniforme incluyendo el Sur (somero, bien mezclado, T_bot ≈ SST)?
  Método: mapa de τ(1985-2004) vs τ(2005-2024) pixel a pixel. Dividir norte (≥57°N)
  vs sur (<57°N) y comparar Δτ por subdominio.

Outputs:
  tau_check1_seasonal.png   — ACF con/sin deseasonalización + tabla de τ por periodo
  tau_check2_spatial.png    — mapas de τ, Δτ, y boxplot N vs S

Corre en CPU, <5 min.

Usage:
  python eval/check_tau_methodology.py --output_dir experiments/figures/thermal_inertia
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

DAILY_NC    = "/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily.nc"
NS_LAT      = slice(100, 127)
NS_LON      = slice(150, 187)
MAX_LAG     = 365
INV_E       = 1.0 / np.e
# lat≈57°N divide NS norte (estratificado) vs sur (somero)
# lat[100]=50°N, lat[127]=63°N → Δ13° en 27 pasos → 57°N ≈ índice 115
NS_LAT_N    = slice(115, 127)   # norte ≥57°N — estratificado, termoclina
NS_LAT_S    = slice(100, 115)   # sur  <57°N  — somero, bien mezclado
PERIOD_EARLY = (1985, 2004)
PERIOD_LATE  = (2005, 2024)


# ── utilidades ────────────────────────────────────────────────────────────────

def load_ns(ds):
    """Devuelve (to_anom_ns, ocean_ns, lat, lon, years, months)."""
    import pandas as pd
    ocean = ds["land_mask"].values.astype(bool)
    to    = ds["to_anom"].values[:, NS_LAT, NS_LON]   # (T, 27, 37)
    ocean_ns = ocean[NS_LAT, NS_LON]
    times    = pd.DatetimeIndex(ds["time"].values)
    years    = times.year.values
    months   = times.month.values
    doys     = times.dayofyear.values
    lat      = ds["lat"].values[NS_LAT]
    lon      = ds["lon"].values[NS_LON]
    return to, ocean_ns, lat, lon, years, months, doys


def nanmean_series(to_3d, ocean_2d):
    """Spatial nanmean over ocean pixels → 1D (T,)."""
    to_flat = to_3d.reshape(len(to_3d), -1)[:, ocean_2d.flatten()]
    return np.nanmean(to_flat, axis=1)


def remove_annual_cycle(series, doys):
    """
    Subtract smooth annual cycle estimated by Fourier regression (first 3 harmonics
    of the 365-day period). This is the same deseasonalization that an ideal
    climatology would achieve.
    """
    t = 2 * np.pi * doys / 365.0
    # Design matrix: const + sin/cos for harmonics 1, 2, 3
    X = np.column_stack([
        np.ones(len(t)),
        np.sin(t), np.cos(t),
        np.sin(2*t), np.cos(2*t),
        np.sin(3*t), np.cos(3*t),
    ])
    coef, _, _, _ = np.linalg.lstsq(X, series, rcond=None)
    cycle = X @ coef
    return series - cycle, cycle


def acf(series, max_lag=MAX_LAG):
    """Autocorrelation function, lag 0..max_lag. NaN-safe."""
    s = series - np.nanmean(series)
    var = np.nanvar(s)
    if var == 0:
        return np.full(max_lag + 1, np.nan)
    n = len(s)
    return np.array([
        np.nanmean(s[:n-k] * s[k:]) / var
        for k in range(max_lag + 1)
    ])


def efold_tau(r):
    """E-folding lag (days) from ACF array r. NaN → max_lag."""
    if np.any(np.isnan(r)):
        return np.nan
    cross = np.where(r < INV_E)[0]
    return float(cross[0]) if len(cross) > 0 else float(len(r) - 1)


def tau_pixel_period(series_3d, ocean_2d, years, t_start, t_end):
    """
    Per-pixel τ for years [t_start, t_end). Returns (H, W) array, NaN for land.
    series_3d: (T, H, W)
    """
    mask = (years >= t_start) & (years < t_end)
    seg  = series_3d[mask]          # (T_sub, H, W)
    H, W = seg.shape[1], seg.shape[2]
    tau_map = np.full((H, W), np.nan)
    for i in range(H):
        for j in range(W):
            if not ocean_2d[i, j]:
                continue
            pix = seg[:, i, j]
            pix = pix[~np.isnan(pix)]
            if len(pix) < 365:
                continue
            r = acf(pix)
            tau_map[i, j] = efold_tau(r)
    return tau_map


# ── CHECK 1: fuga estacional ──────────────────────────────────────────────────

def check1_seasonal(out_dir, to_ns, ocean_ns, years, doys):
    ns = nanmean_series(to_ns, ocean_ns)
    ns_deseas, cycle = remove_annual_cycle(ns, doys)

    # ACF for 4 sub-periods + full
    periods = [
        ("1985-1994", (1985, 1995)),
        ("1995-2004", (1995, 2005)),
        ("2005-2014", (2005, 2015)),
        ("2015-2024", (2015, 2025)),
        ("Full 1985-2024", (1985, 2025)),
    ]
    colors_orig   = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","black"]
    colors_deseas = ["#aec7e8","#ffbb78","#98df8a","#ff9896","gray"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: ACF original per period
    ax = axes[0]
    rows = []
    for (label, (y0, y1)), col in zip(periods, colors_orig):
        m   = (years >= y0) & (years < y1)
        seg = ns[m]
        r   = acf(seg)
        tau = efold_tau(r)
        ax.plot(range(MAX_LAG+1), r, color=col, lw=1.5, label=f"{label}: τ={tau:.0f}d")
        rows.append((label, "original", tau))
    ax.axhline(INV_E, color="red", ls="--", lw=1, label=f"1/e ≈ {INV_E:.2f}")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(365, color="purple", ls=":", lw=0.8, alpha=0.6, label="lag=365d")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF r(k)")
    ax.set_title("ACF de NS to_anom — ORIGINAL\n(con ciclo anual potencialmente residual)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_LAG)

    # Panel B: ACF deseasonalizado per period
    ax = axes[1]
    for (label, (y0, y1)), col in zip(periods, colors_deseas):
        m   = (years >= y0) & (years < y1)
        doy_sub = doys[m]
        seg_orig = ns[m]
        seg_des, _ = remove_annual_cycle(seg_orig, doy_sub)
        r   = acf(seg_des)
        tau = efold_tau(r)
        ax.plot(range(MAX_LAG+1), r, color=col, lw=1.5, label=f"{label}: τ={tau:.0f}d")
        rows.append((label, "deseas.", tau))
    ax.axhline(INV_E, color="red", ls="--", lw=1, label=f"1/e ≈ {INV_E:.2f}")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(365, color="purple", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Lag (days)")
    ax.set_title("ACF de NS to_anom — DESEASONALIZADO\n(armónicos 1+2+3 de 365d eliminados)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_LAG)

    # Panel C: tabla τ original vs deseas.
    ax = axes[2]
    ax.axis("off")
    labels_uniq = [r[0] for r in rows if r[1] == "original"]
    tau_orig    = [r[2] for r in rows if r[1] == "original"]
    tau_des     = [r[2] for r in rows if r[1] == "deseas."]
    delta       = [d - o for o, d in zip(tau_orig, tau_des)]
    cell_text   = [[f"{o:.0f}", f"{d:.0f}", f"{dlt:+.0f}"]
                   for o, d, dlt in zip(tau_orig, tau_des, delta)]
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=labels_uniq,
        colLabels=["τ original (d)", "τ deseas. (d)", "Δτ (d)"],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(True)
    tbl.scale(1, 1.8)
    ax.set_title(
        "Δτ pequeño → climatología suficientemente suave\n"
        "Δτ grande → fuga estacional → τ original no confiable",
        fontsize=10
    )
    # Color code
    for (row, col), cell in tbl.get_celld().items():
        if col == 2 and row > 0:
            val = delta[row - 1]
            cell.set_facecolor("#ffcccc" if abs(val) > 15 else "#ccffcc")

    plt.tight_layout()
    out = out_dir / "tau_check1_seasonal.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    print("\n=== CHECK 1: τ original vs deseasonalizado ===")
    print(f"{'Periodo':<20} {'τ_orig':>10} {'τ_deseas':>10} {'Δτ':>8}  {'OK?':>6}")
    for lbl, tau_o, tau_d in zip(labels_uniq, tau_orig, tau_des):
        dlt = tau_d - tau_o
        ok = "✓" if abs(dlt) <= 15 else "⚠ FUGA"
        print(f"  {lbl:<20} {tau_o:>10.0f} {tau_d:>10.0f} {dlt:>8.0f}  {ok}")


# ── CHECK 2: heterogeneidad espacial ─────────────────────────────────────────

def check2_spatial(out_dir, to_ns, ocean_ns, lat, lon, years):
    print("\nComputing pixel-wise τ (early period 1985-2004)...")
    tau_early = tau_pixel_period(to_ns, ocean_ns, years, *PERIOD_EARLY)
    print("Computing pixel-wise τ (late period 2005-2024)...")
    tau_late  = tau_pixel_period(to_ns, ocean_ns, years, *PERIOD_LATE)
    delta_tau = tau_late - tau_early   # negative = shorter memory

    # North/South split (lat index 115 ≈ 57°N)
    split_i = 15   # relative to NS_LAT.start=100 → abs index 115
    tau_n_early = tau_early[split_i:, :]
    tau_n_late  = tau_late [split_i:, :]
    tau_s_early = tau_early[:split_i, :]
    tau_s_late  = tau_late [:split_i, :]

    # Valid ocean values only
    def ocean_vals(arr):
        return arr[~np.isnan(arr)]

    print("\n=== CHECK 2: τ por subdominio ===")
    for name, e, l in [
        ("Norte ≥57°N (estratificado)", tau_n_early, tau_n_late),
        ("Sur  <57°N  (somero, mezclado)", tau_s_early, tau_s_late),
        ("Todo NS",                        tau_early,  tau_late),
    ]:
        ve, vl = ocean_vals(e), ocean_vals(l)
        print(f"  {name}:")
        print(f"    1985-2004: τ = {np.nanmedian(ve):.0f}d (median)  "
              f"{np.nanmean(ve):.0f}d (mean)")
        print(f"    2005-2024: τ = {np.nanmedian(vl):.0f}d (median)  "
              f"{np.nanmean(vl):.0f}d (mean)")
        print(f"    Δτ = {np.nanmedian(vl)-np.nanmedian(ve):+.0f}d "
              f"({'caída coherente con estratificación' if name.startswith('Norte') else ''})")

    # ── plots ─────────────────────────────────────────────────────────────────
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    vmax_t = np.nanpercentile(np.concatenate([tau_early[ocean_ns],
                                              tau_late[ocean_ns]]), 95)
    dmax   = np.nanpercentile(np.abs(delta_tau[ocean_ns]), 95)
    split_lat = lat[split_i]   # ~57°N

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, title, cmap, vmin, vmax in [
        (axes[0], tau_early,
         f"τ e-folding — {PERIOD_EARLY[0]}–{PERIOD_EARLY[1]}\n(days)",
         "YlOrRd", 0, vmax_t),
        (axes[1], tau_late,
         f"τ e-folding — {PERIOD_LATE[0]}–{PERIOD_LATE[1]}\n(days)",
         "YlOrRd", 0, vmax_t),
        (axes[2], delta_tau,
         f"Δτ = τ(late) − τ(early)\n(días, negativo = memoria más corta en periodo reciente)",
         "RdBu", -dmax, dmax),
    ]:
        d = data.copy().astype(float)
        d[~ocean_ns] = np.nan
        im = ax.imshow(d, origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto")
        # North/south dividing line
        ax.axhline(split_lat, color="black", lw=1.5, ls="--",
                   label=f"57°N — N/S boundary")
        ax.text(lon.min()+0.3, split_lat+0.3, "Norte (estratif.)", fontsize=8,
                color="black", va="bottom")
        ax.text(lon.min()+0.3, split_lat-1.5, "Sur (somero)", fontsize=8,
                color="black", va="top")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        "CHECK 2: Distribución espacial del cambio en τ (NS)\n"
        "Si Δτ < 0 concentrado en Norte → caída de memoria coherente con refuerzo de termoclina\n"
        "Si Δτ uniforme o mayor en Sur → posible artefacto de datos",
        fontsize=10
    )
    plt.tight_layout()
    out = out_dir / "tau_check2_spatial.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")

    # Extra: boxplot N vs S por periodo
    fig, ax = plt.subplots(figsize=(8, 5))
    data_box = [
        ocean_vals(tau_n_early), ocean_vals(tau_n_late),
        ocean_vals(tau_s_early), ocean_vals(tau_s_late),
    ]
    labels_box = [
        f"Norte\n1985-{PERIOD_EARLY[1]}", f"Norte\n2005-{PERIOD_LATE[1]}",
        f"Sur\n1985-{PERIOD_EARLY[1]}",  f"Sur\n2005-{PERIOD_LATE[1]}",
    ]
    colors_box = ["#2196F3", "#1565C0", "#FF9800", "#E65100"]
    bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True,
                    medianprops=dict(color="white", lw=2))
    for patch, col in zip(bp["boxes"], colors_box):
        patch.set_facecolor(col)
    ax.set_ylabel("τ e-folding (días)")
    ax.set_title(
        "τ por subdominio y periodo\n"
        "Si Norte muestra caída mayor que Sur → interpretación física robusta"
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = out_dir / "tau_check2_boxplot.png"
    plt.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/figures/thermal_inertia")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading NS to_anom from merged_daily.nc ...")
    import xarray as xr
    ds = xr.open_dataset(DAILY_NC)
    to_ns, ocean_ns, lat, lon, years, months, doys = load_ns(ds)
    ds.close()
    print(f"  Shape: {to_ns.shape}  Ocean pixels NS: {ocean_ns.sum()}")
    print(f"  lat range NS: {lat[[0,-1]]}  lon range NS: {lon[[0,-1]]}")

    check1_seasonal(out_dir, to_ns, ocean_ns, years, doys)
    check2_spatial(out_dir, to_ns, ocean_ns, lat, lon, years)

    print("\nDone.")


if __name__ == "__main__":
    main()
