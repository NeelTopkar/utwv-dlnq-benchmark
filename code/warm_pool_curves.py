"""Warm-pool dlnq curves and weighting sensitivity.

Draws the fine-bin index against sea-surface temperature under cell, per-day
and profile weighting, and the layer decomposition behind Figures 1 and 3.
"""
import os, re
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve all paths against a single data root.
ROOT = os.environ.get("UTWV_DATA_ROOT", ".")

# Aggregates and per-profile files live in separate trees; both are env-overridable so a batch driver can repoint them.
DS9 = os.environ.get(
    "WARMPOOL_DATASETS_DIR",
    os.path.join(ROOT, "data/aggregates/warm_pool"),
)
PROF = os.environ.get(
    "WARMPOOL_PROFILES_DIR",
    os.path.join(ROOT, "data/profiles/warm_pool"),
)
WP3 = os.environ.get("WP_AGG3_DIR", os.path.join(DS9, "sst_297-307_coarse"))
WP05 = os.environ.get("WP_AGG05_DIR", os.path.join(DS9, "sst_295-305_0.5K"))
WP3_PROF = os.environ.get("WP_PROF3_DIR", os.path.join(PROF, "sst_297-307_coarse"))
WP05_PROF = os.environ.get("WP_PROF05_DIR", os.path.join(PROF, "sst_295-305_0.5K"))
OUT = os.environ.get("WP_PLOTS_DIR", os.path.join(ROOT, "figures/warm_pool"))
dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]
ctr = lambda lbl: (lambda m: (float(m[0]) + float(m[1])) / 2)(re.findall(r"[-+]?\d*\.?\d+", lbl))
keyf = lambda x: (lambda m: float(m[0]) if m else 1e9)(re.findall(r"[-+]?\d*\.?\d+", x))


# Prefer the full per-profile file; fall back to the reduced one shipped with the archive.
def _profile_file(folder):
    full = os.path.join(folder, "profiles.nc")
    if os.path.exists(full):
        return full
    reduced = os.path.join(folder, "profiles_reduced.nc")
    if os.path.exists(reduced):
        return reduced
    raise FileNotFoundError(f"no per-profile file in {folder}; available from the author on request")


# Bin labels: strings in the full files, integer codes in the reduced ones.
def _labels(ds, name):
    da = ds[name]
    cats = da.attrs.get("categories")
    if cats is not None:
        cats = cats.split("|")
        return [cats[int(i)] for i in da.values]
    return dec(da.values)


def load_scr(path):
    ds = xr.open_dataset(path)
    t = ds["time"].values.astype("float64")
    day = (np.datetime64("1993-01-01T00:00:00") + np.round(t).astype("timedelta64[s]")).astype("datetime64[D]")
    df = pd.DataFrame({
        "dlnq": ds["dlnq"].values, "quut": ds["q_uut_mean"].values, "qlut": ds["q_lut_mean"].values,
        "sst": _labels(ds, "sst_bin"), "omega": _labels(ds, "omega_bin"),
        "lat": ds["lat_bin_center"].values, "lon": ds["lon_bin_center"].values, "day": day,
        "olr": _labels(ds, "olr_bin") if "olr_bin" in ds else "na",
    })
    df = df[np.isfinite(df.dlnq) & np.isfinite(df.quut) & np.isfinite(df.qlut)].copy()
    df["cell"] = df.lat.astype(str) + "|" + df.lon.astype(str)
    return df

def three_way(df, val="dlnq"):
    prof = df.groupby("sst")[val].mean()
    cell = df.groupby(["cell", "sst", "omega"], observed=True)[val].mean().reset_index().groupby("sst")[val].mean()
    perday = df.groupby(["cell", "sst", "omega", "day"], observed=True)[val].mean().reset_index().groupby("sst")[val].mean()
    n = df.groupby("sst").size()
    out = pd.DataFrame({"cell": cell, "perday": perday, "profile": prof, "n": n})
    out["x"] = [ctr(i) for i in out.index]
    return out.sort_values("x")

def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    print("###################### WARM POOL (120E-180, 15S-15N) ######################")
    scr3 = load_scr(_profile_file(WP3_PROF))
    print(f"\n3K screened profiles in box: {len(scr3):,}  | gridcells: {scr3.cell.nunique()}  | days: {scr3.day.nunique()}")
    t3 = three_way(scr3)
    print("\n--- dlnq by SST (3K) : cell / per-day / profile ---")
    print(t3[["cell", "perday", "profile", "n"]].round(4).to_string())

    def diff(t, w="cell"):
        d = t.set_index(t.index)[w]
        return d.get("sst_303_305", np.nan) - d.get("sst_300_303", np.nan)
    print("\nHeadline Δdlnq (303_305 - 300_303):")
    for w in ["cell", "perday", "profile"]:
        print(f"  {w:8s}: {diff(t3, w):+.4f}")

    # decomposition q_uut / q_lut (cell-weighted)
    qu = three_way(scr3, "quut"); ql = three_way(scr3, "qlut")
    print("\n--- q_UUT / q_LUT (cell-weighted) and % change 300_303 -> 303_305 ---")
    for lbl in ["sst_300_303", "sst_303_305"]:
        print(f"  {lbl}: q_UUT={qu['cell'].get(lbl):.3e}  q_LUT={ql['cell'].get(lbl):.3e}")
    du = 100 * (qu['cell'].get('sst_303_305') - qu['cell'].get('sst_300_303')) / qu['cell'].get('sst_300_303')
    dl = 100 * (ql['cell'].get('sst_303_305') - ql['cell'].get('sst_300_303')) / ql['cell'].get('sst_300_303')
    print(f"  Δq_UUT = {du:+.1f}%   Δq_LUT = {dl:+.1f}%")

    # OLR deep-convective fraction per SST
    print("\n--- OLR deep_convective fraction by SST (per-profile) ---")
    for lbl in sorted(scr3.sst.unique(), key=keyf):
        sub = scr3[scr3.sst == lbl]
        frac = (sub.olr == "deep_convective").mean()
        print(f"  {lbl}: deep_conv {100*frac:.1f}%  (n={len(sub):,})")

    # Fine 0.5K
    scr05 = load_scr(_profile_file(WP05_PROF))
    t05 = three_way(scr05)
    print(f"\n--- fine 0.5K dlnq by SST (warm pool) : profiles in box {len(scr05):,} ---")
    print(t05[["x", "cell", "perday", "n"]].round(4).to_string(index=False))

    # ---------------- figures ---------------- weighting sensitivity (fine + coarse)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axA.plot(t05.x, t05.cell, "-o", color="tab:blue", ms=4, label="cell / time-collapsed (benchmark)")
    axA.plot(t05.x, t05.perday, "--s", color="tab:red", ms=4, label="per-day")
    axA.set_title("Warm pool — fine 0.5 K bins"); axA.set_xlabel("SST bin center (K)"); axA.set_ylabel("mean Δln(q)")
    axA.grid(True, alpha=0.3); axA.legend(fontsize=8); axA.invert_yaxis()
    ok = t3[t3.n >= 50]
    axB.plot(ok.x, ok.cell, "-o", color="tab:blue", ms=7, label="cell (benchmark)")
    axB.plot(ok.x, ok.perday, "--s", color="tab:red", ms=7, label="per-day")
    axB.set_title("Warm pool — coarse 2 K bins"); axB.set_xlabel("SST bin center (K)"); axB.set_ylabel("mean Δln(q)")
    axB.grid(True, alpha=0.3); axB.legend(fontsize=8); axB.invert_yaxis()
    fig.suptitle("Warm pool: dlnq vs SST and weighting sensitivity", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(OUT, "fig_wp_weighting_sensitivity.png"); fig.savefig(p1, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("\nwrote", p1)

    # dlnq vs SST fine curve (cell-weighted) with peak band
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t05.x, t05.cell, "-o", color="tab:blue", ms=4)
    ax.axvspan(303, 304, color="gold", alpha=0.18)
    ax.set_xlabel("SST bin center (K)"); ax.set_ylabel("mean Δln(q)  (cell-weighted)")
    ax.set_title("Warm pool — dlnq vs SST (fine 0.5 K bins)"); ax.grid(True, alpha=0.3); ax.invert_yaxis()
    p2 = os.path.join(OUT, "fig_wp_dlnq_vs_sst_fine.png"); fig.savefig(p2, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    main()
