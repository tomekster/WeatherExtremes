"""Max / min / range of the 1940-2022 trends over all four variants (seasonal-mean removal x in-base bootstrap: 00, 10, 01, 11)
and all 24 configurations, i.e. 96 runs per grid cell; also per percentile (32 runs each).
Usage: python combine_minmax.py DIR OUT.npz   (DIR holds the global_minmax_1940-2022*.zarr stores)"""
import sys, numpy as np, xarray as xr
D, OUT = sys.argv[1], sys.argv[2]
TAGS = ["00", "10", "01", "11"]
B, SE, NE = [], [], []
for t in TAGS:
    ds = xr.open_zarr(f"{D}/global_minmax_1940-2022.zarr" if t == "00" else f"{D}/global_minmax_1940-2022_{t}.zarr")
    assert int((ds["all_imax"] < 0).sum()) == 0, f"variant {t} not finished"
    B.append(ds.beta.values); SE.append(ds.se.values); NE.append(ds.neff.values)
ref, pool, perc = ds.config_ref.values.astype(str), ds.config_pool.values, ds.config_perc.values
B, SE, NE = (np.concatenate(a, -1) for a in (B, SE, NE))           # (721, 1440, 96); index = variant * 24 + config
assert np.isfinite(B).all()
out = dict(config_ref=np.tile(ref, 4), config_pool=np.tile(pool, 4), config_perc=np.tile(perc, 4),
           config_variant=np.repeat(TAGS, 24))
groups = {"all": np.arange(96)}
for q in (0.90, 0.95, 0.99):
    groups[f"p{int(round(q * 100))}"] = np.where(np.isclose(out["config_perc"], q))[0]
for g, idx in groups.items():
    b = B[..., idx]
    for tag, i in (("max", idx[b.argmax(-1)]), ("min", idx[b.argmin(-1)])):
        out[f"{g}_{tag}"] = np.take_along_axis(B, i[..., None], -1)[..., 0].astype(np.float32)
        out[f"{g}_{tag}_se"] = np.take_along_axis(SE, i[..., None], -1)[..., 0].astype(np.float32)
        out[f"{g}_{tag}_neff"] = np.take_along_axis(NE, i[..., None], -1)[..., 0].astype(np.float32)
        out[f"{g}_i{tag}"] = i.astype(np.int16)
    out[f"{g}_range"] = out[f"{g}_max"] - out[f"{g}_min"]
np.savez_compressed(OUT, **out)
print("saved", OUT)
