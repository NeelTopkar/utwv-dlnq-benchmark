"""Build a compact per-profile file from a screened-profile dataset.

The full per-profile files are 0.16-1.1 GB each, too large to distribute in a
code repository.  Most of that is per-level diagnostics and fixed-width string
bin labels that the published analysis never reads back.

This keeps only what robustness_checks.py and time_aware_sensitivity.py use,
stores bin labels as integer codes with the mapping in a variable attribute,
and compresses: 171 MB becomes about 22 MB.  No profile is aggregated or
dropped, and every per-profile scalar keeps full float64 precision, so the
published statistics are reproduced exactly.  The per-level q array is stored
as float32, the precision MLS retrieves it at.

Usage:
    python reduce_profiles.py <input.nc> <output.nc>
    python reduce_profiles.py            # warm-pool 0.5 K default
"""
import os
import sys

import numpy as np
import xarray as xr

ROOT = os.environ.get("UTWV_DATA_ROOT", ".")
DEFAULT_IN = os.path.join(ROOT, "data/profiles/warm_pool",
                          "sst_295-305_0.5K",
                          "profiles.nc")
DEFAULT_OUT = os.path.join(ROOT, "data/profiles/warm_pool",
                           "sst_295-305_0.5K",
                           "profiles_reduced.nc")

# Per-profile scalars the analysis scripts read.
SCALARS = ["time", "sst", "dlnq", "q_uut_mean", "q_lut_mean",
           "lat_bin_center", "lon_bin_center"]
# String labels stored as integer codes.
LABELS = ["sst_bin", "omega_bin", "olr_bin"]
# Per-level array needed by the layer-boundary sensitivity test.
LEVELS = ["q"]


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def reduce_file(src, dst):
    ds = xr.open_dataset(src)
    out = xr.Dataset()

    for v in SCALARS:
        if v not in ds:
            raise KeyError(f"{src} is missing required variable {v!r}")
        # All per-profile scalars stay float64. time carries seconds since 1993 and is what the observation date is reconstructed from; sst and dlnq feed a quadratic fit near 303 K, whose Vandermonde matrix is badly conditioned at that offset -- in float32 the curvature term collapses and the vertex comes out meaningless. The extra size is a few MB.
        out[v] = ds[v].astype("float64")

    for v in LABELS:
        if v not in ds:
            continue
        vals = _dec(ds[v].values)
        cats = sorted(set(vals))
        code = {c: i for i, c in enumerate(cats)}
        arr = np.array([code[x] for x in vals], dtype="int8")
        da = xr.DataArray(arr, dims=ds[v].dims)
        da.attrs["categories"] = "|".join(cats)
        da.attrs["description"] = (
            f"integer code into the 'categories' attribute; "
            f"decode with categories.split('|')[code]")
        out[v] = da

    for v in LEVELS:
        if v in ds:
            out[v] = ds[v].astype("float32")
    if "plev" in ds.coords or "plev" in ds:
        out = out.assign_coords(plev=ds["plev"])

    out.attrs = dict(ds.attrs)
    out.attrs["reduced_from"] = os.path.basename(src)
    out.attrs["reduction_note"] = (
        "Per-profile subset of the screened-profile file. Profile count "
        "is unchanged. Dropped: lnq, precision, valid, per-level valid counts, "
        "raw lat/lon, w500, olr, day, and the bin-center variables. Bin labels "
        "are stored as integer codes; see each variable's 'categories' attribute.")

    n = ds.sizes.get("profile", 0)
    ds.close()

    enc = {v: {"zlib": True, "complevel": 5}
           for v in out.data_vars if out[v].dtype.kind in "fiu"}
    out.to_netcdf(dst, encoding=enc)
    out.close()

    a, b = os.path.getsize(src), os.path.getsize(dst)
    print(f"  {os.path.basename(src)}")
    print(f"    profiles retained : {n:,}")
    print(f"    {a / 1e6:8.1f} MB  ->  {b / 1e6:.1f} MB   ({100 * b / a:.0f}%)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    dst = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    reduce_file(src, dst)
