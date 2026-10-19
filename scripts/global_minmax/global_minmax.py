"""Global per-gridpoint trends of yearly exceedance counts, 1940-2022. Default: no seasonal-cycle removal, no in-base bootstrap;
--deseason removes the per-DOY REF mean (Brunner & Voigt 2024), --bootstrap applies the Zhang et al. (2005) in-base bootstrap.
Configurations: REF {1961-1990, 1991-2020} x POOL {1, 5, 15, 31} x PERC {0.90, 0.95, 0.99} (24 per point).
For every point: OLS slope (days per decade), its AR(1)-adjusted standard error (Santer et al. 2000: n_eff = n(1-r1)/(1+r1)),
and, for the groups 'all' (24 configs) and each percentile (8 configs), the max and min slope, which configs they are,
and the paired standard error of max - min (OLS of the yearly difference series N_max - N_min, AR(1)-adjusted).
Uses gpu_trends.thresholds (checked earlier against ExceedancesCalculator). Resumable by bands.
Usage: python global_minmax.py ZARR OUT_DIR [--deseason] [--bootstrap] [--rows 4] [--batch 256]
       python global_minmax.py ZARR OUT_DIR --verify WE2_REPO   (Vienna + 2 cells vs ExceedancesCalculator, then exit)"""
import argparse, os, sys, time, numpy as np, torch, xarray as xr
sys.path.insert(0, os.path.expanduser("~/claude_exp/gpu_trends"))
import gpu_trends as G
ap = argparse.ArgumentParser(); ap.add_argument("zarr"); ap.add_argument("out")
ap.add_argument("--rows", type=int, default=4); ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--verify", default=None)
ap.add_argument("--deseason", action="store_true"); ap.add_argument("--bootstrap", action="store_true")
args = ap.parse_args()
AN0, AN1 = 1940, 2022
REFS, POOLS, PERCS = [(1961, 1990), (1991, 2020)], [1, 5, 15, 31], [0.90, 0.95, 0.99]
CONFIGS = [(r, w, q) for r in range(len(REFS)) for w in range(len(POOLS)) for q in range(len(PERCS))]  # index order
GROUPS = {"all": list(range(len(CONFIGS)))}
for qi, q in enumerate(PERCS):
    GROUPS[f"p{int(q * 100)}"] = [i for i, c in enumerate(CONFIGS) if c[2] == qi]
years = np.arange(AN0, AN1 + 1); NY = len(years)
dev = torch.device("cuda")
yt = torch.tensor(years, dtype=torch.float64, device=dev)


def ols_ar1(c):
    """c (B, n) -> slope per decade, AR(1)-adjusted se per decade, n_eff."""
    t = yt - yt.mean(); stt = (t * t).sum(); n = c.shape[-1]
    b = (c * t).sum(-1) / stt
    r = c - c.mean(-1, keepdim=True) - b[:, None] * t
    ss = (r * r).sum(-1)
    r1 = torch.where(ss > 0, (r[:, 1:] * r[:, :-1]).sum(-1) / ss.clamp(min=1e-30), torch.zeros_like(ss)).clamp(min=0)
    neff = n * (1 - r1) / (1 + r1)
    dof = (neff - 2).clamp(min=1.0)
    se = torch.sqrt(ss / (n - 2) / stt) * torch.sqrt((n - 2) / dof)
    return 10 * b, 10 * se, neff


def counts_batch(x):
    """x (P, NY, 365) float32 deg C -> counts (P, 24, NY), config order CONFIGS."""
    out = torch.empty(x.shape[0], len(CONFIGS), NY, dtype=x.dtype, device=x.device)
    for ri, (r0, r1) in enumerate(REFS):
        rs = slice(r0 - AN0, r1 - AN0 + 1); ref = x[:, rs]; an = x
        if args.deseason:
            sc = G.seasonal_cycle(ref, None); ref, an = ref - sc, x - sc   # per-DOY REF mean
        for wi, w in enumerate(POOLS):
            thr = G.thresholds(ref, w, PERCS)                       # (nq, P, 365)
            rates = G.in_base_rates(ref, an[:, rs], w, PERCS) if args.bootstrap else None
            for qi in range(len(PERCS)):
                ex = (an >= thr[qi][:, None, :]).to(x.dtype)
                if args.bootstrap:
                    ex[:, rs] = rates[qi]
                out[:, CONFIGS.index((ri, wi, qi))] = ex.sum(-1)
    return out


def summarise(C):
    """C (P, 24, NY) -> dict of numpy arrays."""
    C = C.double(); P = C.shape[0]
    b, se, neff = ols_ar1(C.reshape(-1, NY)); b, se, neff = (v.reshape(P, -1) for v in (b, se, neff))
    res = {"beta": b, "se": se, "neff": neff, "mean": C.mean(-1)}
    for g, idx in GROUPS.items():
        ix = torch.tensor(idx, device=C.device)
        bg = b[:, ix]; imax = ix[bg.argmax(1)]; imin = ix[bg.argmin(1)]
        ar = torch.arange(P, device=C.device)
        D = C[ar, imax] - C[ar, imin]
        db, dse, dneff = ols_ar1(D)
        res.update({f"{g}_imax": imax, f"{g}_imin": imin, f"{g}_range": db, f"{g}_range_se": dse, f"{g}_range_neff": dneff})
    return {k: v.cpu().numpy() for k, v in res.items()}


def load_band(src, i, j):
    da = src.isel(latitude=slice(i, j)).sel(time=slice(f"{AN0}-01-01", f"{AN1}-12-31")).convert_calendar("noleap").load()
    arr = (da.values - 273.15).astype(np.float32)          # same as the per-city exports
    assert arr.shape[0] == 365 * NY, arr.shape
    assert not np.isnan(arr).any(), f"NaN in band {i}"
    return arr.reshape(NY, 365, -1).transpose(2, 0, 1).copy()      # (P, NY, 365)


src = xr.open_zarr(args.zarr)["daily_max_2m_temperature"]
nlat, nlon = src.sizes["latitude"], src.sizes["longitude"]

if args.verify:
    sys.path.insert(0, args.verify)
    from src.synthetic_vienna.exceedances import ExceedancesCalculator
    calc = ExceedancesCalculator(); row = 360 + 4 * 48 + 1          # 48.25N (Vienna row), cols: Vienna + 2 others
    x = load_band(src, row, row + 1)[[785, 100, 1000]]
    C = counts_batch(torch.tensor(x, device=dev)).cpu().numpy(); md = 0.0
    for n in range(3):
        for k, (ri, wi, qi) in enumerate(CONFIGS):
            r0, r1 = REFS[ri]; rs = slice(r0 - AN0, r1 - AN0 + 1); ref = x[n, rs]; an = x[n]
            if args.deseason:
                sc = ref.mean(0); ref, an = ref - sc, an - sc
            ex = (an >= calc.compute_thresholds(ref, POOLS[wi], PERCS[qi])).astype(float)
            if args.bootstrap:
                ex[rs] = calc.compute_in_base_exceedance_rates(ref, an[rs], np.arange(r1 - r0 + 1), POOLS[wi], PERCS[qi])
            c = ex.sum(1)
            md = max(md, np.abs(c - C[n, k]).max())
    s = summarise(torch.tensor(C, device=dev))
    print(f"verify: max |count diff| vs ExceedancesCalculator over 3 cells x 24 configs = {md}")
    print("Vienna beta (days/decade) per config:", np.round(s["beta"][0], 3).tolist())
    print("Vienna all: max", s["beta"][0].max(), "min", s["beta"][0].min(), "range se", s["all_range_se"][0])
    sys.exit(0)

os.makedirs(args.out, exist_ok=True)
TAG = f"{int(args.deseason)}{int(args.bootstrap)}"                 # 00 raw, 10 deseason, 01 bootstrap, 11 both
store = os.path.join(args.out, "global_minmax_1940-2022.zarr" if TAG == "00" else f"global_minmax_1940-2022_{TAG}.zarr"); done_file = store + ".done"
if not os.path.exists(store):
    import dask.array as dsa
    cfg = np.arange(len(CONFIGS)); v = {}
    for k in ("beta", "se", "neff", "mean"):
        v[k] = (("latitude", "longitude", "config"), dsa.full((nlat, nlon, len(CONFIGS)), np.nan, dtype=np.float32, chunks=(args.rows, nlon, len(CONFIGS))))
    for g in GROUPS:
        for k in ("imax", "imin"):
            v[f"{g}_{k}"] = (("latitude", "longitude"), dsa.full((nlat, nlon), -1, dtype=np.int8, chunks=(args.rows, nlon)))
        for k in ("range", "range_se", "range_neff"):
            v[f"{g}_{k}"] = (("latitude", "longitude"), dsa.full((nlat, nlon), np.nan, dtype=np.float32, chunks=(args.rows, nlon)))
    ds = xr.Dataset(v, coords={"latitude": np.arange(nlat), "longitude": np.arange(nlon), "config": cfg,
                               "config_ref": ("config", [f"{REFS[c[0]][0]}-{REFS[c[0]][1]}" for c in CONFIGS]),
                               "config_pool": ("config", [POOLS[c[1]] for c in CONFIGS]),
                               "config_perc": ("config", [PERCS[c[2]] for c in CONFIGS])})
    ds.attrs.update(analysis=f"{AN0}-{AN1}", seasonal_cycle_removal="per-DOY REF mean (Brunner & Voigt 2024)" if args.deseason else "none",
                    in_base_bootstrap="Zhang et al. 2005" if args.bootstrap else "none", agg=1,
                    units="beta, se, range: exceedance days per decade; mean: days per year",
                    grid="row = 360 + 4 lat, col = 720 + 4 lon", se="AR(1)-adjusted OLS standard error (Santer et al. 2000)",
                    source=args.zarr)
    ds.to_zarr(store, compute=False, mode="w")
done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
for i in range(0, nlat, args.rows):
    if str(i) in done: continue
    t0 = time.perf_counter(); j = min(i + args.rows, nlat)
    x = load_band(src, i, j); t1 = time.perf_counter()
    parts = []
    for s in range(0, x.shape[0], args.batch):
        parts.append(summarise(counts_batch(torch.tensor(x[s:s + args.batch], device=dev))))
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    band = {}
    for k, a in res.items():
        if a.ndim == 2:
            band[k] = (("latitude", "longitude", "config"), a.reshape(j - i, nlon, -1).astype(np.float32))
        elif k.endswith(("imax", "imin")):
            band[k] = (("latitude", "longitude"), a.reshape(j - i, nlon).astype(np.int8))
        else:
            band[k] = (("latitude", "longitude"), a.reshape(j - i, nlon).astype(np.float32))
    xr.Dataset(band).to_zarr(store, region={"latitude": slice(i, j), "longitude": slice(None), "config": slice(None)})
    with open(done_file, "a") as f: f.write(f"{i}\n")
    print(f"band {i}-{j}: read {t1 - t0:.1f} s, compute+write {time.perf_counter() - t1:.1f} s", flush=True)
print("finished")
