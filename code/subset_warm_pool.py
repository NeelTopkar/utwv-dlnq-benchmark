"""Restrict the aggregated datasets to the western Pacific warm pool.

Selects grid cells whose centers lie inside 120E-180, 15S-15N.  Because every
sample is tagged by grid cell and the aggregation is purely per-cell, this is
equivalent to having run the aggregation on a warm-pool domain.
"""
import collections
import os

import numpy as np
import xarray as xr

LON_MIN, LON_MAX = 120.0, 180.0
LAT_MIN, LAT_MAX = -15.0, 15.0

# Read the six tropics aggregated datasets from here ...
DS5 = os.environ.get(
    "WARMPOOL_AGGREGATE_DIR",
    r"data/aggregates/tropical_oceans",
)
# ... and write the fresh warm-pool subsets here. By default this writes to results/warm_pool so the distributed data/aggregates/warm_pool tree is left untouched.
OUT_BASE = os.environ.get(
    "WARMPOOL_OUT_DIR",
    r"results/warm_pool",
)

JOBS = [
    # (input folder, output folder)
    (r"sst_297-307_coarse", r"sst_297-307_coarse"),
    (r"sst_295-305_0.5K", r"sst_295-305_0.5K"),
    (r"sst_299-303_0.5K", r"sst_299-303_0.5K"),
    (r"sst_287-307_variable", r"sst_287-307_variable"),
    (r"sst_303-307_0.5K", r"sst_303-307_0.5K"),
    (r"sst_290-299_1K", r"sst_290-299_1K"),
]
FILES = [("aggregated_gridcells.nc", "sample"),
         ("profiles.nc", "profile")]

def subset(ds, dim):
    latc = ds["lat_bin_center"].values
    lonc = ds["lon_bin_center"].values
    mask = (latc >= LAT_MIN) & (latc <= LAT_MAX) & (lonc >= LON_MIN) & (lonc <= LON_MAX)
    idx = np.where(mask)[0]
    return ds.isel({dim: idx}), len(idx), int(ds.sizes[dim])

def main() -> None:
    for in_fold, out_fold in JOBS:
        out_dir = os.path.join(OUT_BASE, out_fold)
        os.makedirs(out_dir, exist_ok=True)
        for fname, dim in FILES:
            ip = os.path.join(DS5, in_fold, fname)
            # This step reads the full tropical-ocean per-profile files, which are too large to distribute and are available from the author on request.
            if not os.path.exists(ip):
                print(f"[skip] {ip} not present (not distributed in the public archive; available from the author on request)")
                continue
            ds = xr.open_dataset(ip)
            sub, kept, total = subset(ds, dim)
            op = os.path.join(out_dir, fname)
            sub.to_netcdf(op)
            ds.close(); sub.close()
            # quick per-SST sample count for the aggregated file
            if dim == "sample":
                s2 = xr.open_dataset(op)
                sb = [x.decode() if isinstance(x, bytes) else str(x) for x in s2["sst_bin"].values]
                cnt = collections.Counter(sb)
                cells = set(zip(s2["lat_bin_center"].values, s2["lon_bin_center"].values))
                s2.close()
                print(f"  {out_fold}/{fname}: kept {kept:,} of {total:,} "
                      f"in {len(cells)} cells; per SST bin {dict(sorted(cnt.items()))}")
            else:
                print(f"  {out_fold}/{fname}: kept {kept:,} of {total:,}")


if __name__ == "__main__":
    main()
