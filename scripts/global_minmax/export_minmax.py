"""Compact export of global_minmax_1940-2022.zarr for plotting: per group (all, p90, p95, p99) the max and min slope,
their own AR(1)-adjusted se and n_eff, which config they are, and the paired range, its se and n_eff.
Usage: python export_minmax.py STORE OUT.npz"""
import sys, numpy as np, xarray as xr
ds = xr.open_zarr(sys.argv[1])
assert int((ds["all_imax"] < 0).sum()) == 0, "run not finished"
beta, se, neff = (ds[k].values for k in ("beta", "se", "neff"))           # (721, 1440, 24)
out = dict(config_ref=ds.config_ref.values.astype(str), config_pool=ds.config_pool.values, config_perc=ds.config_perc.values)
for g in ("all", "p90", "p95", "p99"):
    for k in ("imax", "imin"):
        i = ds[f"{g}_{k}"].values.astype(np.int64)
        tag = k[1:]
        out[f"{g}_{tag}"] = np.take_along_axis(beta, i[..., None], -1)[..., 0].astype(np.float32)
        out[f"{g}_{tag}_se"] = np.take_along_axis(se, i[..., None], -1)[..., 0].astype(np.float32)
        out[f"{g}_{tag}_neff"] = np.take_along_axis(neff, i[..., None], -1)[..., 0].astype(np.float32)
        out[f"{g}_i{tag}"] = i.astype(np.int8)
    for k in ("range", "range_se", "range_neff"):
        out[f"{g}_{k}"] = ds[f"{g}_{k}"].values.astype(np.float32)
out["mean"] = ds["mean"].values.astype(np.float32)                            # (721, 1440, 24) days per year
np.savez_compressed(sys.argv[2], **out)
print("saved", sys.argv[2])
