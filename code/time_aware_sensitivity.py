r"""
Time-aware sensitivity of the UTWV steepening slope and the ~303 K turning point.

The primary analysis pools 2004-2023 into a cell-regime climatology and bootstraps
10-degree spatial tiles, so its intervals describe spatial sampling only: every
replicate contains all twenty years.  This script asks the complementary question
-- which YEARS drive the result -- with four tests:

  1. Leave-one-year-out.  Rebuild the climatology 19 times, each omitting one year,
     and refit all four turning-point estimators.  Reports the range across folds.
     August-December 2004 is merged into the 2005 fold, so every fold removes a
     comparable share of the record.
  2. Season.  Refit within DJF / MAM / JJA / SON.
  3. ENSO.  Refit within El Nino / neutral / La Nina months, classified from the
     CPC Oceanic Nino Index at the conventional +/- 0.5 K thresholds.
  4. Warm-bin support.  Per-year and per-season count of 5-degree cells and profiles
     contributing to the 303-305 K range, which is the geographic-concentration half
     of the reviewer's question.

Estimator definitions are identical to turning_point_statistics.py; only the
spatial bootstrap is omitted, because the quantity of interest here is the spread
across folds rather than a per-fold interval.  The aggregation path is the one
validated in robustness_checks.py, which reproduces the published point estimates
exactly from the screened profiles.

Run with the project venv (numpy/pandas/xarray).
"""
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.environ.get("UTWV_DATA_ROOT", ".")
WP_SCR = os.path.join(ROOT, "data/profiles/warm_pool",
                      "sst_295-305_0.5K",
                      "profiles.nc")
TROP_SCR = os.path.join(ROOT, "data/profiles/tropical_oceans",
                        "sst_295-305_0.5K",
                        "profiles.nc")
ONI_PATH = os.environ.get("UTWV_ONI", os.path.join(ROOT, "data/enso", "oni_cpc.txt"))
OUTDIR = os.path.join(ROOT, "results", "time_aware")

MLS_EPOCH = np.datetime64("1993-01-01T00:00:00")

# Fit windows and estimator settings, matching turning_point_statistics.py.
SLOPE_LO, SLOPE_HI = 300.0, 303.0
CURVE_LO, CURVE_HI = 300.0, 305.0
TAU_GRID = np.arange(301.0, 304.01, 0.1)
KERN_GRID = np.arange(301.0, 304.51, 0.1)
KERN_BW = 0.6

# Minimum cell-regime aggregates required before a fold is fitted at all.
MIN_AGGREGATES = 200
ENSO_THRESHOLD = 0.5
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


# Bin labels from either the full or the reduced screened-profile file.
def _labels(ds, name):
    da = ds[name]
    cats = da.attrs.get("categories")
    if cats is not None:
        cats = cats.split("|")
        return np.array([cats[int(i)] for i in da.values])
    return _dec(da.values)


# Prefer the full screened-profile file; fall back to the reduced one.
def _resolve(path):
    if os.path.exists(path):
        return path
    alt = path.replace("profiles.nc",
                       "profiles_reduced.nc")
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(
        f"neither {path} nor its reduced counterpart was found; available from the author on request")


# ---------------------------------------------------------------- data loading

# Screened profiles with the TRUE observation date reconstructed.
def load_profiles(path):
    ds = xr.open_dataset(_resolve(path))
    t = ds["time"].values.astype("float64")
    date = (MLS_EPOCH + np.round(t).astype("timedelta64[s]")).astype("datetime64[D]")
    df = pd.DataFrame({
        "dlnq": ds["dlnq"].values,
        "sst": ds["sst"].values,
        "quut": ds["q_uut_mean"].values,
        "qlut": ds["q_lut_mean"].values,
        "lat": ds["lat_bin_center"].values,
        "lon": ds["lon_bin_center"].values,
        "sstbin": _labels(ds, "sst_bin"),
        "omega": _labels(ds, "omega_bin"),
        "date": date,
    })
    ds.close()
    ok = np.isfinite(df.dlnq) & np.isfinite(df.sst) & (df.quut > 0) & (df.qlut > 0)
    df = df[ok].reset_index(drop=True)
    idx = pd.DatetimeIndex(df["date"])
    df["year"] = idx.year
    df["month"] = idx.month
    df["season"] = df["month"].map(SEASON)
    # Aug-Dec 2004 is a five-month year; merge it into 2005 so every leave-one-out fold removes a comparable share of the record.
    df["fold_year"] = df["year"].where(df["year"] != 2004, 2005)
    return df


# CPC Oceanic Nino Index -> {(year, month): value}.
def load_oni(path):
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 13:
                continue
            if not re.fullmatch(r"(19|20)\d{2}", parts[0]):
                continue
            try:
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            year = int(parts[0])
            for m, v in enumerate(vals, start=1):
                if v > -99.0:
                    out[(year, m)] = v
    if not out:
        raise RuntimeError(f"no ONI rows parsed from {path}")
    return out


def classify_enso(df, oni):
    v = np.array([oni.get((y, m), np.nan)
                  for y, m in zip(df["year"].values, df["month"].values)])
    lab = np.full(len(df), "unclassified", dtype=object)
    lab[v >= ENSO_THRESHOLD] = "El Nino"
    lab[v <= -ENSO_THRESHOLD] = "La Nina"
    lab[(v > -ENSO_THRESHOLD) & (v < ENSO_THRESHOLD)] = "neutral"
    return pd.Series(lab, index=df.index), v


# ------------------------------------------------------------------ estimators

# One sample per (5-degree cell, SST bin, omega) -- the published unit. lat/lon are retained so the same aggregates can be fed to the 10-degree spatial-block bootstrap in turning_point_statistics.py.
def agg_cell(df):
    g = (df.groupby(["lat", "lon", "sstbin", "omega"], observed=True)
           .agg(dlnq=("dlnq", "mean"), sst=("sst", "mean")).reset_index())
    return g[["sst", "dlnq", "lat", "lon"]].reset_index(drop=True)


def _window(agg, lo, hi):
    sub = agg[(agg.sst >= lo) & (agg.sst < hi)]
    return sub.sst.values, sub.dlnq.values


# Point estimates of the four turning-point summaries.
def fit_all(agg):
    out = {}
    x, y = _window(agg, SLOPE_LO, SLOPE_HI)
    out["n_slope"] = len(x)
    out["slope"] = np.polyfit(x, y, 1)[0] if len(x) >= 3 else np.nan

    x, y = _window(agg, CURVE_LO, CURVE_HI)
    out["n_curve"] = len(x)
    if len(x) < 5:
        out.update(curvature=np.nan, vertex=np.nan, breakpoint=np.nan, kernel=np.nan)
        return out

    a, b, _c = np.polyfit(x, y, 2)
    out["curvature"] = a
    out["vertex"] = -b / (2 * a) if a != 0 else np.nan

    best = None
    for tau in TAU_GRID:
        X = np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - tau)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(np.sum((y - X @ beta) ** 2))
        if best is None or rss < best[0]:
            best = (rss, tau)
    out["breakpoint"] = best[1]

    sm = np.empty(len(KERN_GRID))
    for k, g in enumerate(KERN_GRID):
        w = np.exp(-0.5 * ((x - g) / KERN_BW) ** 2)
        sw = w.sum()
        sm[k] = (w @ y) / sw if sw > 0 else np.nan
    out["kernel"] = KERN_GRID[int(np.nanargmin(sm))] if np.any(np.isfinite(sm)) else np.nan
    return out


def fit_subset(df, label, note=""):
    agg = agg_cell(df)
    if len(agg) < MIN_AGGREGATES:
        return {"label": label, "note": note, "n_profiles": len(df),
                "n_aggregates": len(agg), "slope": np.nan, "curvature": np.nan,
                "vertex": np.nan, "breakpoint": np.nan, "kernel": np.nan,
                "n_slope": 0, "n_curve": 0}
    r = fit_all(agg)
    r.update(label=label, note=note, n_profiles=len(df), n_aggregates=len(agg))
    return r


# ----------------------------------------------------------------------- tests

def leave_one_year_out(df):
    rows = [fit_subset(df, "ALL YEARS", "full sample")]
    for y in sorted(df["fold_year"].unique()):
        sub = df[df["fold_year"] != y]
        note = "omits Aug-Dec 2004 and 2005" if y == 2005 else f"omits {y}"
        rows.append(fit_subset(sub, f"drop {y}", note))
    return pd.DataFrame(rows)


def by_season(df):
    rows = [fit_subset(df, "ALL SEASONS", "full sample")]
    for s in ["DJF", "MAM", "JJA", "SON"]:
        rows.append(fit_subset(df[df["season"] == s], s, ""))
    return pd.DataFrame(rows)


def by_enso(df):
    rows = [fit_subset(df, "ALL PHASES", "full sample")]
    for phase in ["El Nino", "neutral", "La Nina"]:
        rows.append(fit_subset(df[df["enso"] == phase], phase, ""))
    unc = int((df["enso"] == "unclassified").sum())
    if unc:
        rows.append({"label": "unclassified", "note": "no ONI value",
                     "n_profiles": unc, "n_aggregates": np.nan,
                     "slope": np.nan, "curvature": np.nan, "vertex": np.nan,
                     "breakpoint": np.nan, "kernel": np.nan,
                     "n_slope": 0, "n_curve": 0})
    return pd.DataFrame(rows)


COARSE_WP = os.path.join(ROOT, "data/profiles/warm_pool",
                         "sst_297-307_coarse",
                         "profiles.nc")
COARSE_TROP = os.path.join(ROOT, "data/profiles/tropical_oceans",
                           "sst_297-307_coarse",
                           "profiles.nc")


# Cell-weighted Delta dlnq = dlnq(303-305 K) - dlnq(300-303 K).
def coarse_contrast(df):
    g = (df.groupby(["lat", "lon", "sstbin", "omega"], observed=True)
           .agg(dlnq=("dlnq", "mean")).reset_index())
    m = g.groupby("sstbin")["dlnq"].mean()
    if "sst_300_303" not in m.index or "sst_303_305" not in m.index:
        return np.nan, len(g)
    return float(m["sst_303_305"] - m["sst_300_303"]), len(g)


# Leave-one-year-out on the coarse headline contrast.
def loyo_contrast(path, domain):
    df = load_profiles(path)
    rows = []
    val, n = coarse_contrast(df)
    rows.append({"label": "ALL YEARS", "delta_dlnq": val, "n_aggregates": n})
    for y in sorted(df["fold_year"].unique()):
        v, n = coarse_contrast(df[df["fold_year"] != y])
        rows.append({"label": f"drop {y}", "delta_dlnq": v, "n_aggregates": n})
    tbl = pd.DataFrame(rows)
    folds = tbl[tbl.label.str.startswith("drop")]["delta_dlnq"].dropna()
    print(f"\n--- {domain}: leave-one-year-out on the coarse contrast ---")
    print(f"  full sample      Delta dlnq = {val:+.4f}")
    print(f"  {len(folds)} folds        min {folds.min():+.4f}   max {folds.max():+.4f}   "
          f"range {folds.max() - folds.min():.4f}")
    print(f"  negative in every fold: {'YES' if (folds < 0).all() else 'NO'}")
    worst = tbl.loc[tbl.delta_dlnq.idxmax()]
    print(f"  least negative fold: {worst.label} ({worst.delta_dlnq:+.4f})")
    return tbl


# 10-degree spatial-block bootstrap CIs for each season / ENSO stratum. Uses the published estimators in turning_point_statistics.py unchanged, so the intervals are directly comparable with Tables 3 and 4.
def bootstrap_strata(strata, out_csv=None):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tsb", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "turning_point_statistics.py"))
    tsb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tsb)

    rows = []
    for label, sub in strata:
        agg = agg_cell(sub)
        if len(agg) < MIN_AGGREGATES:
            continue
        sp, sl, sh, n, nb = tsb.slope(agg, SLOPE_LO, SLOPE_HI, "spatial", 10.0)
        am, am_ci, _, _ = tsb.argmin_smooth(agg, CURVE_LO, CURVE_HI, "spatial", 10.0)
        robust = (sl < 0 and sh < 0)
        rows.append({"label": label, "n_aggregates": len(agg), "tiles": nb,
                     "slope": sp, "slope_lo": sl, "slope_hi": sh,
                     "slope_robust_negative": robust,
                     "kernel": am, "kernel_lo": am_ci[0], "kernel_hi": am_ci[1]})
        print(f"    {label:12s} slope={sp:+.4f} [{sl:+.4f},{sh:+.4f}] "
              f"{'ROBUST' if robust else 'not robust':10s} "
              f"kernel={am:.2f} [{am_ci[0]:.2f},{am_ci[1]:.2f}]  "
              f"(n={len(agg):,}, tiles={nb})")
    tbl = pd.DataFrame(rows)
    if out_csv:
        tbl.to_csv(out_csv, index=False)
    return tbl


# Geographic and temporal support of the 303-305 K range.
def warm_bin_support(df):
    warm = df[(df.sst >= 303.0) & (df.sst < 305.0)]
    n_cells_all = warm.groupby(["lat", "lon"]).ngroups
    by_year = (warm.groupby("year")
                   .apply(lambda s: pd.Series({
                       "profiles": len(s),
                       "cells": s.groupby(["lat", "lon"]).ngroups,
                       "share_pct": 100.0 * len(s) / len(warm)}),
                          include_groups=False)
                   .reset_index())
    by_season_ = (warm.groupby("season")
                      .apply(lambda s: pd.Series({
                          "profiles": len(s),
                          "cells": s.groupby(["lat", "lon"]).ngroups,
                          "share_pct": 100.0 * len(s) / len(warm)}),
                             include_groups=False)
                      .reset_index())
    return warm, n_cells_all, by_year, by_season_


# ---------------------------------------------------------------------- report

FIT_COLS = ["label", "n_profiles", "n_aggregates", "slope", "curvature",
            "vertex", "breakpoint", "kernel", "note"]


def _show(name, tbl):
    print(f"\n--- {name} ---")
    t = tbl.reindex(columns=FIT_COLS).copy()
    for c in ["slope", "curvature"]:
        t[c] = t[c].map(lambda v: f"{v:+.4f}" if np.isfinite(v) else "--")
    for c in ["vertex", "breakpoint", "kernel"]:
        t[c] = t[c].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "--")
    for c in ["n_profiles", "n_aggregates"]:
        t[c] = t[c].map(lambda v: f"{int(v):,}" if np.isfinite(v) else "--")
    print(t.to_string(index=False))


def _spread(name, tbl, cols=("slope", "vertex", "breakpoint", "kernel")):
    folds = tbl[tbl.label.str.startswith("drop")]
    print(f"\n  spread across {len(folds)} {name} folds:")
    for c in cols:
        v = folds[c].dropna().values
        if not len(v):
            continue
        fmt = "{:+.4f}" if c in ("slope", "curvature") else "{:.2f}"
        lo, hi = v.min(), v.max()
        print(f"    {c:11s} min {fmt.format(lo)}   max {fmt.format(hi)}   "
              f"range {hi - lo:.4f}   sign flips: "
              f"{'YES' if (c == 'slope' and (v > 0).any()) else 'no'}")


def run(domain, path, oni):
    print("\n" + "=" * 78)
    print(f"### {domain}")
    print("=" * 78)
    df = load_profiles(path)
    df["enso"], _ = classify_enso(df, oni)
    print(f"profiles: {len(df):,}   dates {df.date.min()} .. {df.date.max()}   "
          f"years {df.year.min()}-{df.year.max()}")

    loyo = leave_one_year_out(df)
    _show("Leave-one-year-out", loyo)
    _spread("leave-one-year-out", loyo)

    seas = by_season(df)
    _show("Season", seas)

    enso = by_enso(df)
    _show("ENSO phase", enso)

    warm, ncells, by_y, by_s = warm_bin_support(df)
    print(f"\n--- 303-305 K support: {len(warm):,} profiles over {ncells} cells ---")
    by_y["share_pct"] = by_y["share_pct"].map("{:.1f}".format)
    print(by_y.to_string(index=False))
    by_s["share_pct"] = by_s["share_pct"].map("{:.1f}".format)
    print(by_s.to_string(index=False))

    os.makedirs(OUTDIR, exist_ok=True)
    tag = domain.lower().replace(" ", "_")

    print("\n--- Spatial-block bootstrap CIs by season and ENSO phase ---")
    strata = ([(s, df[df["season"] == s]) for s in ["DJF", "MAM", "JJA", "SON"]]
              + [(p, df[df["enso"] == p]) for p in ["El Nino", "neutral", "La Nina"]])
    boot = bootstrap_strata(strata,
                            out_csv=os.path.join(OUTDIR, f"{tag}_strata_ci.csv"))
    for name, tbl in [("loyo", loyo), ("season", seas), ("enso", enso),
                      ("warmbin_by_year", by_y), ("warmbin_by_season", by_s)]:
        p = os.path.join(OUTDIR, f"{tag}_{name}.csv")
        tbl.to_csv(p, index=False)
    print(f"\nwrote CSVs to {OUTDIR}")
    return loyo, seas, enso


# True if the full or the reduced per-profile file is present.
def _available(path):
    try:
        _resolve(path)
        return True
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    oni = load_oni(ONI_PATH)
    yrs = sorted({y for y, _ in oni})
    print(f"ONI: {len(oni)} monthly values, {yrs[0]}-{yrs[-1]}, from {ONI_PATH}")
    for domain, path, coarse in [("WARM POOL", WP_SCR, COARSE_WP),
                                 ("TROPICS", TROP_SCR, COARSE_TROP)]:
        if not _available(path):
            print(f"\n[skip] {domain}: per-profile file not present "
                  f"(not distributed in the public archive; available from the author on request)")
            continue
        run(domain, path, oni)
        if _available(coarse):
            tag = domain.lower().replace(" ", "_")
            loyo_contrast(coarse, domain).to_csv(
                os.path.join(OUTDIR, f"{tag}_loyo_contrast.csv"), index=False)
