"""Turning-point statistics for the ~303 K feature, with spatial-block intervals.

Four summaries of the dlnq-SST curve, each recomputed under a bootstrap that
resamples whole 10 degree spatial tiles:

  1. a linear slope over 300-303 K, the steepening rate;
  2. a quadratic fit over 300-305 K, giving curvature and a vertex;
  3. a continuous two-segment linear fit with an estimated breakpoint;
  4. the minimum of a Gaussian-kernel local mean, which assumes no shape.

Agreement among estimators with different assumptions is what makes the
turning point defensible.  Produces Table 3 and the tropical-ocean values in
Table 4.  1000 replicates, seed 12345, 2.5-97.5 percentile intervals.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.environ.get("UTWV_DATA_ROOT", ".")
WP_05 = os.path.join(ROOT, "data/aggregates/warm_pool/sst_295-305_0.5K/aggregated_gridcells.nc")
WP_1K = os.path.join(ROOT, "data/aggregates/warm_pool/sst_287-307_variable/aggregated_gridcells.nc")
TROP_05 = os.path.join(ROOT, "data/aggregates/tropical_oceans/sst_295-305_0.5K/aggregated_gridcells.nc")

N_REP, SEED = 1000, 12345


def load(path):
    ds = xr.open_dataset(path)
    df = pd.DataFrame({
        "dlnq": ds["dlnq_mean"].values,
        "sst": ds["mean_sst"].values,
        "lat": ds["lat_bin_center"].values,
        "lon": ds["lon_bin_center"].values,
        "quut": ds["q_uut_mean"].values,
        "qlut": ds["q_lut_mean"].values,
    })
    ds.close()
    ok = np.isfinite(df.dlnq) & np.isfinite(df.sst) & (df.quut > 0) & (df.qlut > 0)
    return df[ok].reset_index(drop=True)


def tiles(df, deg):
    return (np.floor(df.lat.values / deg) * deg).astype(int).astype(str) + "|" + \
           (np.floor(df.lon.values / deg) * deg).astype(int).astype(str)


# Yield index arrays: spatial-block resamples of whole tiles.
def _resamplers(df, deg, n_rep, seed):
    blk = tiles(df, deg)
    groups = pd.Series(np.arange(len(df))).groupby(blk).apply(lambda s: s.values).tolist()
    rng = np.random.default_rng(seed)
    nb = len(groups)
    out = []
    for _ in range(n_rep):
        chosen = rng.integers(0, nb, nb)
        out.append(np.concatenate([groups[c] for c in chosen]))
    return out, nb


def _iid_resamplers(n, n_rep, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_rep)]


def slope(df, lo, hi, scheme, deg=10.0):
    sub = df[(df.sst >= lo) & (df.sst < hi)].reset_index(drop=True)
    x, y = sub.sst.values, sub.dlnq.values
    pt = np.polyfit(x, y, 1)[0]
    if scheme == "iid":
        res = _iid_resamplers(len(sub), N_REP, SEED); nb = len(sub)
    else:
        res, nb = _resamplers(sub, deg, N_REP, SEED)
    bs = np.array([np.polyfit(x[i], y[i], 1)[0] for i in res])
    lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5])
    return pt, lo_ci, hi_ci, len(sub), nb


def quad(df, lo, hi, scheme, deg=10.0):
    sub = df[(df.sst >= lo) & (df.sst < hi)].reset_index(drop=True)
    x, y = sub.sst.values, sub.dlnq.values

    def fit(xx, yy):
        a, b, c = np.polyfit(xx, yy, 2)
        v = -b / (2 * a) if a != 0 else np.nan
        return a, v

    a_pt, v_pt = fit(x, y)
    if scheme == "iid":
        res = _iid_resamplers(len(sub), N_REP, SEED); nb = len(sub)
    else:
        res, nb = _resamplers(sub, deg, N_REP, SEED)
    aa, vv = [], []
    for i in res:
        a, v = fit(x[i], y[i]); aa.append(a); vv.append(v)
    aa, vv = np.array(aa), np.array(vv)
    n_bad = int(np.sum(~np.isfinite(vv)))
    if n_bad:
        print(f"      note: {n_bad} of {len(vv)} replicates gave no finite vertex and were dropped")
    vv = vv[np.isfinite(vv)]
    return (a_pt, np.percentile(aa, [2.5, 97.5]),
            v_pt, np.percentile(vv, [2.5, 97.5]), len(sub), nb)


# Continuous piecewise-linear (segmented) fit with one breakpoint tau: dlnq = b0 + b1*SST + b2*max(0, SST-tau) Slope below tau = b1; slope above = b1+b2; tau = the turnaround SST.
def changepoint(df, lo, hi, scheme, deg=10.0, grid=None):
    sub = df[(df.sst >= lo) & (df.sst < hi)].reset_index(drop=True)
    x, y = sub.sst.values, sub.dlnq.values
    if grid is None:
        grid = np.arange(301.0, 304.01, 0.1)

    def fit_seg(xx, yy):
        best = None
        for tau in grid:
            hinge = np.maximum(0.0, xx - tau)
            X = np.column_stack([np.ones_like(xx), xx, hinge])
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            rss = float(np.sum((yy - X @ beta) ** 2))
            if best is None or rss < best[0]:
                best = (rss, tau, beta)
        _, tau, beta = best
        return tau, beta[1], beta[1] + beta[2]   # tau, slope_below, slope_above

    tau_pt, sb_pt, sa_pt = fit_seg(x, y)
    if scheme == "iid":
        res = _iid_resamplers(len(sub), N_REP, SEED); nb = len(sub)
    else:
        res, nb = _resamplers(sub, deg, N_REP, SEED)
    taus, sbs, sas = [], [], []
    for i in res:
        t, sb, sa = fit_seg(x[i], y[i]); taus.append(t); sbs.append(sb); sas.append(sa)
    taus, sbs, sas = np.array(taus), np.array(sbs), np.array(sas)
    return (tau_pt, np.percentile(taus, [2.5, 97.5]),
            sb_pt, np.percentile(sbs, [2.5, 97.5]),
            sa_pt, np.percentile(sas, [2.5, 97.5]), len(sub), nb)


# MODEL-FREE turnaround location: arg-min of a Gaussian-kernel local mean (Nadaraya-Watson) of dlnq vs SST.
def argmin_smooth(df, lo, hi, scheme, deg=10.0, bw=0.6, grid=None):
    sub = df[(df.sst >= lo) & (df.sst < hi)].reset_index(drop=True)
    x, y = sub.sst.values, sub.dlnq.values
    if grid is None:
        grid = np.arange(301.0, 304.51, 0.1)

    def loc(xx, yy):
        sm = np.empty(len(grid))
        for k, g in enumerate(grid):
            w = np.exp(-0.5 * ((xx - g) / bw) ** 2)
            sw = w.sum()
            sm[k] = (w @ yy) / sw if sw > 0 else np.nan
        return grid[int(np.nanargmin(sm))]

    pt = loc(x, y)
    if scheme == "iid":
        res = _iid_resamplers(len(sub), N_REP, SEED); nb = len(sub)
    else:
        res, nb = _resamplers(sub, deg, N_REP, SEED)
    locs = np.array([loc(x[i], y[i]) for i in res])
    return pt, np.percentile(locs, [2.5, 97.5]), len(sub), nb


def _dec(a):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in a]


# Build PER-DAY samples (gridcell x SST bin x omega x day) from the screened profiles, so the slope/curvature/changepoint can be re-estimated under observation (per-day) weighting instead of the cell climatology.
def load_perday(screened_path):
    ds = xr.open_dataset(screened_path)
    # The profile files carry no date column, so reconstruct the observation date from the raw 'time' (seconds since 1993-01-01) before grouping by day.
    t = ds["time"].values.astype("float64")
    real_day = (np.datetime64("1993-01-01T00:00:00")
                + np.round(t).astype("timedelta64[s]")).astype("datetime64[D]")
    df = pd.DataFrame({
        "dlnq": ds["dlnq"].values,
        "sst": ds["sst"].values,
        "quut": ds["q_uut_mean"].values,
        "qlut": ds["q_lut_mean"].values,
        "lat": ds["lat_bin_center"].values,
        "lon": ds["lon_bin_center"].values,
        "sstbin": _dec(ds["sst_bin"].values),
        "omega": _dec(ds["omega_bin"].values),
        "day": real_day,
    })
    ds.close()
    ok = np.isfinite(df.dlnq) & np.isfinite(df.sst) & (df.quut > 0) & (df.qlut > 0)
    df = df[ok]
    g = (df.groupby(["lat", "lon", "sstbin", "omega", "day"], observed=True)
           .agg(dlnq=("dlnq", "mean"), sst=("sst", "mean")).reset_index())
    return g[["sst", "dlnq", "lat", "lon"]].reset_index(drop=True)


def report_perday(name, screened_path, cap=100000):
    df = load_perday(screened_path)
    n_full = len(df)
    if n_full > cap:
        df = df.sample(cap, random_state=SEED).reset_index(drop=True)
    print(f"\n################ {name}  (n={len(df):,} of {n_full:,} per-day samples) ################")
    for sch, deg in [("iid", None), ("spatial", 10.0)]:
        tag = "iid" if sch == "iid" else f"spatial-{deg:g}deg"
        sp, sl, sh, n, nb = slope(df, 300.0, 303.0, sch, deg or 10.0)
        a_pt, a_ci, v_pt, v_ci, _, _ = quad(df, 300.0, 305.0, sch, deg or 10.0)
        cp_grid = np.arange(301.0, 304.01, 0.2)
        tau_pt, tau_ci, cb, cb_ci, ca, ca_ci, _, _ = changepoint(df, 300.0, 305.0, sch, deg or 10.0, cp_grid)
        am_pt, am_ci, _, _ = argmin_smooth(df, 300.0, 305.0, sch, deg or 10.0)
        print(f"   {tag:14s}: slope(300-303)={sp:+.3f} [{sl:+.3f},{sh:+.3f}]; "
              f"curv={a_pt:+.4f} [{a_ci[0]:+.4f},{a_ci[1]:+.4f}]; vertex={v_pt:.2f}K [{v_ci[0]:.2f},{v_ci[1]:.2f}]; "
              f"changepoint={tau_pt:.2f}K [{tau_ci[0]:.2f},{tau_ci[1]:.2f}]; "
              f"model-free argmin={am_pt:.2f}K [{am_ci[0]:.2f},{am_ci[1]:.2f}]  (n={n:,}, blocks={nb})")


def report(name, path):
    df = load(path)
    print(f"\n################ {name}  (n={len(df)} samples, SST {df.sst.min():.1f}-{df.sst.max():.1f}) ################")
    print("\n-- Steepening slope d(dlnq)/d(SST) over the RISING LIMB --")
    for lo, hi, label in [(300.0, 303.0, "300-303 K"), (297.5, 303.0, "297.5-303 K")]:
        for sch, deg in [("iid", None), ("spatial", 10.0)]:
            pt, l, h, n, nb = slope(df, lo, hi, sch, deg or 10.0)
            tag = "iid" if sch == "iid" else f"spatial-{deg:g}deg"
            rob = "ROBUST" if (l < 0 and h < 0) or (l > 0 and h > 0) else "not robust"
            print(f"   [{label}] {tag:14s}: slope={pt:+.4f} dlnq/K  95% CI [{l:+.4f}, {h:+.4f}]  ({rob}; n={n}, blocks={nb})")
    print("\n-- Turnaround via quadratic fit (curvature a>0 => real minimum; vertex = turnaround SST) --")
    for lo, hi, label in [(299.0, 305.0, "299-305 K"), (300.0, 305.0, "300-305 K")]:
        for sch, deg in [("iid", None), ("spatial", 10.0)]:
            a_pt, a_ci, v_pt, v_ci, n, nb = quad(df, lo, hi, sch, deg or 10.0)
            tag = "iid" if sch == "iid" else f"spatial-{deg:g}deg"
            arob = "curv>0 ROBUST" if a_ci[0] > 0 else "curv not robust"
            print(f"   [{label}] {tag:14s}: curvature a={a_pt:+.4f} CI [{a_ci[0]:+.4f},{a_ci[1]:+.4f}] ({arob}); "
                  f"vertex={v_pt:.2f} K CI [{v_ci[0]:.2f},{v_ci[1]:.2f}]  (n={n}, blocks={nb})")
    print("\n-- Turnaround via piecewise-linear CHANGEPOINT (breakpoint tau; sign change of slope) --")
    for sch, deg in [("iid", None), ("spatial", 10.0)]:
        tau_pt, tau_ci, sb, sb_ci, sa, sa_ci, n, nb = changepoint(df, 300.0, 305.0, sch, deg or 10.0)
        tag = "iid" if sch == "iid" else f"spatial-{deg:g}deg"
        sign = "sign change ROBUST" if (sb_ci[1] < 0 and sa_ci[0] > 0) else "sign change not robust"
        print(f"   [300-305 K] {tag:14s}: breakpoint tau={tau_pt:.2f} K CI [{tau_ci[0]:.2f},{tau_ci[1]:.2f}]; "
              f"slope_below={sb:+.3f} CI [{sb_ci[0]:+.3f},{sb_ci[1]:+.3f}]; "
              f"slope_above={sa:+.3f} CI [{sa_ci[0]:+.3f},{sa_ci[1]:+.3f}]  ({sign}; n={n}, blocks={nb})")
    print("\n-- MODEL-FREE turnaround location (arg-min of a Gaussian-kernel local mean; no shape assumption) --")
    for sch, deg in [("iid", None), ("spatial", 10.0)]:
        am_pt, am_ci, n, nb = argmin_smooth(df, 300.0, 305.0, sch, deg or 10.0)
        tag = "iid" if sch == "iid" else f"spatial-{deg:g}deg"
        print(f"   [300-305 K] {tag:14s}: arg-min SST = {am_pt:.2f} K  95% CI [{am_ci[0]:.2f}, {am_ci[1]:.2f}]  (n={n}, blocks={nb})")


# Screened-profile files for the per-day (observation-weighted) re-bootstrap.
# Prefer the full file; fall back to the reduced one shipped with the archive, which
# carries every column load_perday needs.
def _profiles(domain):
    base = os.path.join(ROOT, f"data/profiles/{domain}/sst_295-305_0.5K")
    full = os.path.join(base, "profiles.nc")
    return full if os.path.exists(full) else os.path.join(base, "profiles_reduced.nc")


WP_05_SCR = _profiles("warm_pool")
TROP_05_SCR = _profiles("tropical_oceans")


# Skip a section whose input is not distributed, rather than crashing. The aggregate files are included in the public archive; the per-profile files that the per-day re-weighting needs are not (available from the author on request).
def _run_if_present(fn, name, path):
    if os.path.exists(path):
        fn(name, path)
    else:
        print(f"\n[skip] {name}: {os.path.basename(path)} not present "
              f"(not distributed in the public archive; see README)")


if __name__ == "__main__":
    _run_if_present(report, "WARM POOL 0.5 K", WP_05)
    _run_if_present(report, "WARM POOL 1 K", WP_1K)
    _run_if_present(report, "TROPICS 0.5 K", TROP_05)
    print("\n========== PER-DAY (observation) WEIGHTING re-bootstrap ==========")
    _run_if_present(report_perday, "WARM POOL 0.5 K (per-day)", WP_05_SCR)
    _run_if_present(report_perday, "TROPICS 0.5 K (per-day)", TROP_05_SCR)
