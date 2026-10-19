"""One figure per variant with 24 global maps, one per configuration (REF x POOL x PERC), of the 1940-2022 trend of
yearly exceedance counts (days per decade). All panels of all variants share one colour scale, so any two panels
can be compared directly. Rows: percentile x REF; columns: POOL.
Usage: python plot_config_grid.py DIR OUTDIR LSM_ZARR   (DIR holds the global_minmax_1940-2022*.zarr stores;
LSM_ZARR is the ERA5 land_sea_mask, lat 90 -> -90, lon 0 -> 359.75)"""
import os, sys, numpy as np, xarray as xr, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D, OUT, LSM = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True)
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2})
TAGS = {"00": "raw", "10": "seasonal mean removed", "01": "Zhang in-base bootstrap", "11": "seasonal mean removed + bootstrap"}
rows, cols = np.arange(721), np.arange(1440)
lat, lon = -90 + 0.25 * rows, -180 + 0.25 * cols
land = xr.open_zarr(LSM, consolidated=False)["land_sea_mask"].values[720 - rows][:, (cols - 720) % 1440] > 0.5
W = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 1440))
EXT = [-180 - 0.125, 180 - 0.125, -90 - 0.125, 90 + 0.125]


def wq(a, q, m):
    v, w = a[m].ravel(), W[m].ravel(); o = np.argsort(v); c = np.cumsum(w[o]); c /= c[-1]
    return float(v[o][np.searchsorted(c, q)])


data = {}
for t in TAGS:
    p = f"{D}/global_minmax_1940-2022.zarr" if t == "00" else f"{D}/global_minmax_1940-2022_{t}.zarr"
    if os.path.exists(p):
        ds = xr.open_zarr(p)
        data[t] = ds.beta.values                                   # (721, 1440, 24) = (row, col, config)
        ref, pool, perc = ds.config_ref.values.astype(str), ds.config_pool.values, ds.config_perc.values
# one shared symmetric scale: area-weighted 99th percentile of |beta| over every panel of every variant
sub = np.abs(np.stack([B[::2, ::2] for B in data.values()], 0))   # (V, 361, 720, 24), every 2nd cell is plenty
Wb = np.broadcast_to(W[::2, ::2][None, :, :, None], sub.shape)
v, w = sub.ravel(), Wb.ravel(); o = np.argsort(v); c = np.cumsum(w[o]); c /= c[-1]
L = float(np.ceil(v[o][np.searchsorted(c, 0.99)]))
del sub, v, w, o, c
print(f"shared colour scale: -{L} .. {L} days per decade (99th area-weighted percentile of |beta|)")

REFS, POOLS, PERCS = list(dict.fromkeys(ref)), list(dict.fromkeys(pool)), list(dict.fromkeys(perc))
ROWS = [(q, r) for q in PERCS for r in REFS]
lines = ["variant,perc,ref,pool,land_median,globe_median,land_frac_negative"]
for t, B in data.items():
    fig, axs = plt.subplots(len(ROWS), len(POOLS), figsize=(16, 2.25 * len(ROWS) + 1.6))
    for i, (q, r) in enumerate(ROWS):
        for j, w_ in enumerate(POOLS):
            k = int(np.where((ref == r) & (pool == w_) & np.isclose(perc, q))[0][0])
            b = B[..., k]; ax = axs[i, j]
            im = ax.imshow(b, origin="lower", extent=EXT, cmap="RdBu_r", vmin=-L, vmax=L, interpolation="nearest")
            ax.contour(lon, lat, land, levels=[0.5], colors=INK, linewidths=0.25)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color(INK2); s.set_linewidth(0.5)
            lm, gm, neg = wq(b, 0.5, land), wq(b, 0.5, np.ones_like(land)), float((W * (b < 0))[land].sum() / W[land].sum())
            ax.text(0.01, 0.03, f"land median {lm:.1f}", transform=ax.transAxes, fontsize=7, color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
            lines.append(f"{t},{q},{r},{w_},{lm:.3f},{gm:.3f},{neg:.3f}")
            if i == 0: ax.set_title(f"POOL {w_}", fontsize=10, color=INK)
            if j == 0: ax.set_ylabel(f"P{int(round(q * 100))}\nREF {r.replace('-', '–')}", fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 0.7 / fig.get_figheight(), bottom=1.15 / fig.get_figheight(),
                        hspace=0.06, wspace=0.03)
    cax = fig.add_axes([0.3, 0.62 / fig.get_figheight(), 0.4, 0.15 / fig.get_figheight()])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("trend of yearly exceedance count, days per decade (same scale in every panel)", fontsize=9)
    fig.suptitle(f"ERA5 daily Tmax, 1940–2022, OLS trend of yearly exceedance counts per configuration · {TAGS[t]}",
                 fontsize=11, color=INK, y=1 - 0.2 / fig.get_figheight())
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_global_trend_per_config_{t}.{ext}", dpi=150 if ext == "png" else None)
    plt.close(fig); print("saved", t)
open(f"{OUT}/global_trend_per_config_summary.csv", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
