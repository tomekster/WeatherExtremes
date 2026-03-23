#!/usr/bin/env python3
"""
Rechunk a zarr store for efficient band-by-band time-series reads.

The default chunk layout of many zarr datasets is (1, nlat, nlon) — one full
spatial slice per chunk.  For the band-by-band pipeline in main.py this is
extremely inefficient: reading 7 lat rows × 22 000 days loads all 721 lat rows
per time step (~91 GB of I/O to use ~880 MB of data).

This script rewrites the zarr store with chunks shaped
    (time_chunk, 1, nlon)
so that each chunk covers many time steps for a single lat row.
After rechunking, the band-by-band pipeline reads only the chunks it needs —
roughly 50–3 000× fewer chunk reads depending on time_chunk size.

I/O strategy
------------
The source is read in time batches (default: 365 days = ~1.5 GB per batch).
Each batch covers ALL lat rows, so the source is read exactly once.
The output chunks are written complete (one per lat row per time batch),
avoiding expensive partial-chunk read-modify-write cycles.

Example
-------
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate

    python src/rechunk.py \\
        --input  data/preprocessed/t2max.zarr \\
        --var    daily_max_2m_temperature \\
        --output data/preprocessed/rechunked/t2max_rechunked.zarr

    # Tune time_chunk (default 365). Larger = fewer chunks, more RAM per batch:
    python src/rechunk.py ... --time-chunk 730   # 2-year chunks, ~3 GB/batch

Notes
-----
* Rechunking is a one-time cost.  Point --input in main.py at the rechunked store.
* Peak RAM ≈ time_chunk × nlat × nlon × 4 bytes (365 default → ~1.5 GB).
* The output store uses Blosc/LZ4 compression on the data variable.
* All coordinate arrays and attributes are copied verbatim.
* Progress is shown as a tqdm bar over time batches.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  required=True, help="Path to input zarr store")
    p.add_argument("--var",    required=True, help="Variable name to rechunk")
    p.add_argument("--output", required=True, help="Path for rechunked zarr store")
    p.add_argument(
        "--time-chunk", type=int, default=365, metavar="N",
        help="Time steps per output chunk, also used as the read batch size "
             "(default: 365). Larger values → fewer chunks and faster pipeline "
             "reads, but more RAM per batch (365 → ~1.5 GB for a global grid).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output store if it already exists.",
    )
    return p


def rechunk(
    input_path: str,
    var: str,
    output_path: str,
    time_chunk: int | None = None,
    overwrite: bool = False,
) -> None:
    if os.path.exists(output_path):
        if overwrite:
            shutil.rmtree(output_path)
        else:
            print(f"Output already exists: {output_path}")
            print("Use --overwrite to replace it.")
            return

    print(f"Opening {input_path} ...")
    ds = xr.open_zarr(input_path, consolidated=False)
    da = ds[var]

    # Normalise dim names
    if "lat" in da.dims:
        da = da.rename({"lat": "latitude"})
    if "lon" in da.dims:
        da = da.rename({"lon": "longitude"})

    ntime = da.sizes["time"]
    nlat  = da.sizes["latitude"]
    nlon  = da.sizes["longitude"]

    tc = time_chunk
    n_batches = -(-ntime // tc)
    chunk_shape = (tc, 1, nlon)
    batch_bytes = tc * nlat * nlon * 4
    print(
        f"Grid      : {ntime} time × {nlat} lat × {nlon} lon\n"
        f"Chunks    : {chunk_shape}  ({tc * nlon * 4 / 1e6:.1f} MB each)\n"
        f"Batches   : {n_batches} × {batch_bytes / 1e9:.2f} GB RAM per batch\n"
        f"Total chunks: {n_batches * nlat}\n"
        f"Output    : {output_path}"
    )

    # Pre-create output zarr group
    grp = zarr.open_group(output_path, mode="w")

    grp.create_array(
        var,
        shape=(ntime, nlat, nlon),
        chunks=chunk_shape,
        dtype=np.float32,
        compressors=[BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.shuffle)],
        dimension_names=["time", "latitude", "longitude"],
    )

    # Copy coordinate arrays
    for coord_name, coord_da in ds.coords.items():
        vals = coord_da.values
        # Encode cftime time coordinates as int32 days since 1900-01-01 if needed
        if coord_name == "time" and hasattr(vals[0], "timetuple"):
            import cftime
            epoch = cftime.DatetimeNoLeap(1900, 1, 1)
            vals = np.array([(t - epoch).days for t in vals], dtype=np.int32)
        # Normalise coordinate name to match dimension name so xarray can
        # link them (zarr v3 requires array name == dimension name for coords)
        out_name = "latitude" if coord_name == "lat" else \
                   "longitude" if coord_name == "lon" else coord_name
        cdims = [out_name] if len(coord_da.dims) == 1 else list(coord_da.dims)
        grp.create_array(out_name, data=vals, chunks=vals.shape,
                         dimension_names=cdims)

    # Copy global attributes
    grp.attrs.update(ds.attrs)

    # Read the source in time batches (aligned with output chunks) so it is
    # read exactly once.  Each batch covers ALL lat rows → writes are complete
    # output chunks, no partial-chunk read-modify-write overhead.
    data_arr = grp[var]
    print("Rechunking ...")
    for t_start in tqdm(range(0, ntime, tc), desc="time batches"):
        t_end = min(t_start + tc, ntime)
        # Load batch: (batch_size, nlat, nlon) — reads only t_end-t_start chunks
        batch = da.isel(time=slice(t_start, t_end)).values.astype(np.float32)
        data_arr[t_start:t_end, :, :] = batch

    zarr.consolidate_metadata(output_path)
    print(f"\nDone. Rechunked store written to: {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    rechunk(
        input_path=args.input,
        var=args.var,
        output_path=args.output,
        time_chunk=args.time_chunk,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
