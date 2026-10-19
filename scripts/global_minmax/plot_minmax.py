"""Maps of the max, min and range (max - min) of the 1940-2022 exceedance-count trends over configurations, plus their
standard errors. Input: global_minmax_summary.npz from export_minmax.py. Usage: python plot_minmax.py SUMMARY.npz OUTDIR LSM_ZARR (ERA5 land_sea_mask, lat 90 -> -90, lon 0 -> 359.75)"""
import os, sys, numpy as np, xarray as xr, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
d = np.load(sys.argv[1]); OUT = sys.argv[2]; LSM = sys.argv[3]; os.makedirs(OUT, exist_ok=True)
rows, cols = np.arange(721), np.arange(1440)
lat, lon = -90 + 0.25 * rows, -180 + 0.25 * cols
lsm = xr.open_zarr(LSM, consolidated=False)["land_sea_mask"].values
land = lsm[720 - rows][:, (cols - 720) % 1440]
W = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 1440))
EXT = [-180 - 0.125, 180 - 0.125, -90 - 0.125, 90 + 0.125]
PL = {"all": "all 24 configurations", "p90": "P90 (8 configurations)", "p95": "P95 (8 configurations)", "p99": "P99 (8 configurations)"}


def wmean(a, m=None):
    m = np.isfinite(a) if m is None else m & np.isfinite(a)
    return float((a * W)[m].sum() / W[m].sum())


def wq(a, q, m=None):
    m = np.ones(a.shape, bool) if m is None else m
    v, w = a[m].ravel(), W[m].ravel(); o = np.argsort(v); c = np.cumsum(w[o]); c /= c[-1]
    return float(v[o][np.searchsorted(c, q)])


def lim(a, q=0.99):
    return wq(np.abs(a), q)


def panel(ax, a, cmap, vmin, vmax, title):
    im = ax.imshow(a, origin="lower", extent=EXT, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.contour(lon, lat, land, levels=[0.5], colors=INK, linewidths=0.35)
    ax.set_title(title, loc="left", fontsize=9, color=INK); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(INK2); s.set_linewidth(0.5)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, extend="both")
    cb.ax.tick_params(labelsize=7.5)
    return cb


def fig_values(groups, name, suptitle):
    n = len(groups)
    fig, axs = plt.subplots(n, 3, figsize=(16, 3.7 * n + 0.6), squeeze=False)
    for r, g in enumerate(groups):
        mx, mn, rg = d[f"{g}_max"], d[f"{g}_min"], d[f"{g}_range"]
        L = max(lim(mx), lim(mn))
        panel(axs[r, 0], mx, "RdBu_r", -L, L, f"{'abc'[0] if n == 1 else chr(97 + 3 * r)}   {PL[g]}: maximum trend β").set_label("days per decade", fontsize=8)
        panel(axs[r, 1], mn, "RdBu_r", -L, L, f"{chr(98 + 3 * r)}   {PL[g]}: minimum trend β").set_label("days per decade", fontsize=8)
        panel(axs[r, 2], rg, "magma_r", 0, wq(rg, 0.99), f"{chr(99 + 3 * r)}   {PL[g]}: range = max − min").set_label("days per decade", fontsize=8)
    fig.suptitle(suptitle, fontsize=10.5, color=INK, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.55 / fig.get_figheight(), bottom=0.02, hspace=0.42, wspace=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=170 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def fig_se(groups, name, suptitle):
    n = len(groups)
    fig, axs = plt.subplots(n, 3, figsize=(16, 3.7 * n + 0.6), squeeze=False)
    for r, g in enumerate(groups):
        se_own = 0.5 * (d[f"{g}_max_se"] + d[f"{g}_min_se"])
        se_rg = d[f"{g}_range_se"]
        tq = stats.t.ppf(0.975, np.clip(d[f"{g}_range_neff"] - 2, 1, None))
        z = d[f"{g}_range"] / (tq * se_rg)            # > 1: range significant at the 95% level (paired)
        hi = wq(se_own, 0.99)
        panel(axs[r, 0], se_own, "viridis", 0, hi, f"{chr(97 + 3 * r)}   {PL[g]}: SE of one trend (mean of max and min config)").set_label("days per decade", fontsize=8)
        panel(axs[r, 1], se_rg, "viridis", 0, hi, f"{chr(98 + 3 * r)}   {PL[g]}: paired SE of the range").set_label("days per decade", fontsize=8)
        cb = panel(axs[r, 2], z, "PuOr_r", 0, 2, f"{chr(99 + 3 * r)}   {PL[g]}: range / paired 95% half-width")
        cb.set_label("> 1: range significant (paired 95% CI excludes 0)", fontsize=8)
    fig.suptitle(suptitle, fontsize=10.5, color=INK, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.55 / fig.get_figheight(), bottom=0.02, hspace=0.42, wspace=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=170 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


BASE = "ERA5 daily Tmax, 1940–2022, no seasonal-cycle removal, no in-base bootstrap; REF 1961–1990 / 1991–2020 × POOL 1, 5, 15, 31"
fig_values(["all"], "fig_global_minmax_all", f"Trend of yearly exceedance counts over all configurations (P90, P95, P99). {BASE}")
fig_values(["p90", "p95", "p99"], "fig_global_minmax_per_percentile", f"Trend of yearly exceedance counts, per percentile. {BASE}")
fig_se(["p90", "p95", "p99"], "fig_global_minmax_se_per_percentile",
       f"Standard errors (AR(1)-adjusted OLS) of the trends and of their range, per percentile. {BASE}")
fig_se(["all"], "fig_global_minmax_se_all", f"Standard errors over all configurations. {BASE}")

# summary table: area-weighted, all cells and land only
m_land = (land > 0.5) & (np.abs(lat)[:, None] <= 90)
lines = ["group,domain,median_max,median_min,median_range,median_se_one_trend,median_se_range_paired,area_frac_sign_change,area_frac_range_significant"]
for g in ("all", "p90", "p95", "p99"):
    se_own = 0.5 * (d[f"{g}_max_se"] + d[f"{g}_min_se"])
    tq = stats.t.ppf(0.975, np.clip(d[f"{g}_range_neff"] - 2, 1, None))
    sig = d[f"{g}_range"] > tq * d[f"{g}_range_se"]
    cross = (d[f"{g}_min"] < 0) & (d[f"{g}_max"] > 0)
    for dom, m in (("globe", np.ones_like(m_land)), ("land", m_land)):
        lines.append(",".join([g, dom] + [f"{wq(a, 0.5, m):.3f}" for a in (d[f"{g}_max"], d[f"{g}_min"], d[f"{g}_range"], se_own, d[f"{g}_range_se"])]
                              + [f"{wmean(cross.astype(float), m):.3f}", f"{wmean(sig.astype(float), m):.3f}"]))
open(f"{OUT}/global_minmax_summary.csv", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
# which configuration gives max / min (share of area)
for g in ("all", "p90", "p95", "p99"):
    for k in ("imax", "imin"):
        i = d[f"{g}_{k}"].astype(int); u = np.unique(i)
        sh = sorted(((wmean((i == v).astype(float)), v) for v in u), reverse=True)[:4]
        print(g, k, ", ".join(f"REF {d['config_ref'][v]} POOL {d['config_pool'][v]} P{int(round(d['config_perc'][v] * 100))}: {s:.0%}" for s, v in sh))
