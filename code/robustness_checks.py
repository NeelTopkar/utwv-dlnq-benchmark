"""Cloud-subset and layer-boundary robustness checks.

Three tests behind Section 3.6: re-fitting on the high-outgoing-longwave-
radiation subset, to check the result is not produced only by cloud-affected
scenes; moving the 215.44 hPa boundary level from the upper to the lower layer;
and mapping which grid cells carry the samples.

Reuses the estimators in turning_point_statistics.py unchanged.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import turning_point_statistics as tb

ROOT = os.environ.get("UTWV_DATA_ROOT", ".")
WP_SCR = os.path.join(ROOT, "data/profiles/warm_pool/sst_295-305_0.5K/profiles.nc")
TROP_SCR = os.path.join(ROOT, "data/profiles/tropical_oceans/sst_295-305_0.5K/profiles.nc")
OUTDIR = os.path.join(ROOT, "results/robustness")


def _dec(a):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in a]


# Bin labels from either the full or the reduced per-profile file.
def _labels(ds, name):
    da = ds[name]
    cats = da.attrs.get("categories")
    if cats is not None:
        cats = cats.split("|")
        return [cats[int(i)] for i in da.values]
    return _dec(da.values)


# Prefer the full per-profile file; fall back to the reduced one.
def _resolve(path):
    if os.path.exists(path):
        return path
    alt = path.replace("profiles.nc", "profiles_reduced.nc")
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(
        f"neither {path} nor its reduced counterpart was found; see README")


def available(path):
    try:
        _resolve(path)
        return True
    except FileNotFoundError:
        return False


def load_profiles(path):
    ds = xr.open_dataset(_resolve(path))
    df = pd.DataFrame({
        "dlnq": ds["dlnq"].values,
        "sst": ds["sst"].values,
        "quut": ds["q_uut_mean"].values,
        "qlut": ds["q_lut_mean"].values,
        "lat": ds["lat_bin_center"].values,
        "lon": ds["lon_bin_center"].values,
        "sstbin": _labels(ds, "sst_bin"),
        "omega": _labels(ds, "omega_bin"),
        "olr": _labels(ds, "olr_bin") if "olr_bin" in ds else "na",
    })
    q = ds["q"].values            # (profile, plev), NaN where invalid
    plev = ds["plev"].values
    ds.close()
    return df, q, plev


# Cell-climatology aggregation: one sample per (cell, SST bin, omega).
def agg_cell(df_sub):
    g = (df_sub.groupby(["lat", "lon", "sstbin", "omega"], observed=True)
                .agg(dlnq=("dlnq", "mean"), sst=("sst", "mean")).reset_index())
    return g[["sst", "dlnq", "lat", "lon"]].reset_index(drop=True)


def fit_all(d, tag):
    sp, sl, sh, n, nb = tb.slope(d, 300.0, 303.0, "spatial", 10.0)
    a_pt, a_ci, v_pt, v_ci, _, _ = tb.quad(d, 300.0, 305.0, "spatial", 10.0)
    tau, tau_ci, cb, cb_ci, ca, ca_ci, _, _ = tb.changepoint(d, 300.0, 305.0, "spatial", 10.0)
    am, am_ci, _, _ = tb.argmin_smooth(d, 300.0, 305.0, "spatial", 10.0)
    print(f"   {tag:22s}: slope={sp:+.3f} [{sl:+.3f},{sh:+.3f}]; curv={a_pt:+.4f} [{a_ci[0]:+.4f},{a_ci[1]:+.4f}]; "
          f"vertex={v_pt:.2f}K [{v_ci[0]:.2f},{v_ci[1]:.2f}]; changept={tau:.2f}K [{tau_ci[0]:.2f},{tau_ci[1]:.2f}]; "
          f"argmin={am:.2f}K [{am_ci[0]:.2f},{am_ci[1]:.2f}]  (n_samples={len(d)}, tiles={nb})")


def clearsky_and_baseline(name, path):
    print(f"\n################ {name}: CLEAR-SKY check ################")
    df, q, plev = load_profiles(path)
    ok = np.isfinite(df.dlnq) & np.isfinite(df.sst) & (df.quut > 0) & (df.qlut > 0)
    dfv = df[ok]
    print(f"   profiles: all={len(dfv):,}; clear_sky={int((dfv.olr=='clear_sky').sum()):,} "
          f"({100*(dfv.olr=='clear_sky').mean():.0f}%); deep_conv={int((dfv.olr=='deep_convective').sum()):,}")
    fit_all(agg_cell(dfv), "ALL profiles (baseline)")
    fit_all(agg_cell(dfv[dfv.olr == "clear_sky"]), "CLEAR-SKY only")
    fit_all(agg_cell(dfv[dfv.olr != "clear_sky"]), "cloudy (non-clear) only")


def layer_boundary(name, path):
    print(f"\n################ {name}: LAYER-BOUNDARY sensitivity ################")
    df, q, plev = load_profiles(path)
    plev = np.asarray(plev, dtype=float)
    # original split: UUT = p <= 215.44 (3 levels), LUT = p > 215.44 (2 levels)
    # alt split:      UUT = p <  215     (2 levels), LUT = p >= 215 (3 levels)
    uut_alt = plev < 215.0
    lut_alt = plev >= 215.0
    with np.errstate(invalid="ignore", divide="ignore"):
        quut = np.nanmean(q[:, uut_alt], axis=1)
        qlut = np.nanmean(q[:, lut_alt], axis=1)
        dlnq_alt = np.log(quut / qlut)
    df2 = df.copy()
    df2["dlnq"] = dlnq_alt
    df2["quut"] = quut; df2["qlut"] = qlut
    ok = np.isfinite(df2.dlnq) & np.isfinite(df2.sst) & (df2.quut > 0) & (df2.qlut > 0)
    print(f"   alt metric: UUT={{{', '.join(f'{p:.0f}' for p in plev[uut_alt])}}} hPa (2 lvls), "
          f"LUT={{{', '.join(f'{p:.0f}' for p in plev[lut_alt])}}} hPa (3 lvls); n={int(ok.sum()):,}")
    fit_all(agg_cell(df2[ok]), "ALT boundary (215->LUT)")


def coverage_map(name, path, box=None, fname="fig_coverage.png"):
    df, q, plev = load_profiles(path)
    ok = np.isfinite(df.dlnq) & np.isfinite(df.sst) & (df.quut > 0) & (df.qlut > 0)
    dfv = df[ok]
    cells = dfv.groupby(["lat", "lon"], observed=True).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sc = ax.scatter(cells.lon, cells.lat, c=cells.n, s=70, cmap="viridis",
                    norm=matplotlib.colors.LogNorm(), edgecolor="0.3", linewidth=0.3)
    fig.colorbar(sc, ax=ax, label="profiles per 5° cell")
    if box:
        lo, hi, la, lb = box
        ax.add_patch(plt.Rectangle((lo, la), hi - lo, lb - la, fill=False, edgecolor="red", lw=1.5))
    for g in np.arange(-180, 181, 10):
        ax.axvline(g, color="0.85", lw=0.4, zorder=0)
    for g in np.arange(-30, 31, 10):
        ax.axhline(g, color="0.85", lw=0.4, zorder=0)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"{name}: 5° gridcell coverage (10° tile grid shown)")
    fig.tight_layout()
    p = os.path.join(OUTDIR, fname)
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"   wrote {p}  ({len(cells)} cells)")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    for name, path, fig in [("WARM POOL 0.5 K", WP_SCR, "fig_coverage_warmpool.png"),
                            ("TROPICS 0.5 K", TROP_SCR, "fig_coverage_tropics.png")]:
        if not available(path):
            print(f"\n[skip] {name}: per-profile file not present "
                  f"(not distributed in the public archive; see README)")
            continue
        clearsky_and_baseline(name, path)
        layer_boundary(name, path)
        coverage_map(name, path, box=(120, 180, -15, 15), fname=fig)
