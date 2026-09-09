"""Warm-pool benchmark profiles and heat maps.

Draws the profile figures and the SST-by-vertical-motion heat maps for the warm
pool, using the same file basenames as the tropical-ocean figures so the same
generators can be repointed at either domain.
"""
import os, re, shutil
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve all paths against a single data root.
ROOT = os.environ.get("UTWV_DATA_ROOT", ".")

# Warm-pool data now lives in the warm-pool subsets (the warm-pool restriction Stuff); env-overridable.
AGG = os.environ.get(
    "WP_AGG3_NC",
    os.path.join(ROOT, "data/aggregates/warm_pool/sst_297-307_coarse/aggregated_gridcells.nc"),
)
OUT = os.environ.get("WP_PLOTS_DIR", os.path.join(ROOT, "figures/warm_pool"))
dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]
keylo = lambda x: (lambda m: float(m[0]) if m else 1e9)(re.findall(r"[-+]?\d*\.?\d+", x))
OMO = ["strong_ascent", "weak_ascent", "neutral", "weak_subsidence", "strong_subsidence"]

COL = {"sst_297_300": "#2C7FB8", "sst_300_303": "#41AB5D", "sst_303_305": "#E8743B", "sst_305_307": "#888888"}

ds = plev = sst = omega = qmp = lqmp = dlnq = molr = sst_bins = keep_bins = None


# Read the aggregate into the module-level arrays the figure functions draw from. Called from main() so that importing this module does no I/O.
def load() -> None:
    global ds, plev, sst, omega, qmp, lqmp, dlnq, molr, sst_bins, keep_bins
    ds = xr.open_dataset(AGG)
    plev = ds["plev"].values
    sst = np.array(dec(ds["sst_bin"].values))
    omega = np.array(dec(ds["omega_bin"].values))
    qmp = ds["q_mean_profile"].values         # (sample, plev)
    lqmp = ds["lnq_mean_profile"].values
    dlnq = ds["dlnq_mean"].values
    molr = ds["mean_olr"].values
    sst_bins = [b for b in sorted(set(sst), key=keylo)]
    # keep well-sampled SST bins
    keep_bins = [b for b in sst_bins if (sst == b).sum() >= 30]

def profile_fig(arr, fname, xlabel, title):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for b in keep_bins:
        m = sst == b
        prof = np.nanmean(arr[m], axis=0)
        p25 = np.nanpercentile(arr[m], 25, axis=0)
        p75 = np.nanpercentile(arr[m], 75, axis=0)
        c = COL.get(b, None)
        ax.plot(prof, plev, "-o", color=c, ms=4, label=b.replace("sst_", "").replace("_", "–") + " K")
        ax.fill_betweenx(plev, p25, p75, color=c, alpha=0.15)
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.set_ylabel("Pressure (hPa)")
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(title="SST bin", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=200); plt.close(fig)
    print("wrote", fname)

# heatmap helper
def heatmap(values, fname, title, cbar, fmt="{:.2f}"):
    df = pd.DataFrame({"sst": sst, "omega": omega, "v": values})
    piv = df.groupby(["omega", "sst"])["v"].mean().reset_index().pivot(index="omega", columns="sst", values="v")
    piv = piv.reindex(index=[o for o in OMO if o in piv.index], columns=keep_bins)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([c.replace("sst_", "").replace("_", "–") for c in piv.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if np.isfinite(piv.values[i, j]):
                ax.text(j, i, fmt.format(piv.values[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if not np.isnan(piv.values[i, j]) else "black")
    ax.set_xlabel("SST bin"); ax.set_ylabel("ω regime"); ax.set_title(title)
    cb = fig.colorbar(im, ax=ax); cb.set_label(cbar)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=200); plt.close(fig)
    print("wrote", fname)

def main() -> None:
    load()
    os.makedirs(OUT, exist_ok=True)

    profile_fig(qmp, "fig1_main_profiles_by_sst.png", "q  (benchmark mean across samples)",
                "Warm pool — benchmark UTWV profiles by SST bin")
    profile_fig(lqmp, "fig1b_lnq_profiles_by_sst.png", "ln q", "Warm pool — ln(q) profiles by SST bin")

    heatmap(dlnq, "fig4_dlnq_heatmap_sst_omega.png", "Warm pool — mean Δln(q) by SST × ω", "mean Δln(q)")
    heatmap(molr, "fig5_olr_mean_heatmap.png", "Warm pool — mean OLR by SST × ω", "mean OLR (W/m²)", "{:.0f}")

    # copy the already-made warm-pool figures to standard basenames so generators just repoint dirs
    shutil.copyfile(os.path.join(OUT, "fig_wp_dlnq_vs_sst_fine.png"), os.path.join(OUT, "fig3_dlnq_vs_sst.png"))
    shutil.copyfile(os.path.join(OUT, "fig_wp_weighting_sensitivity.png"), os.path.join(OUT, "fig_weighting_sensitivity_dlnq_vs_sst.png"))
    print("copied fig3_dlnq_vs_sst.png and fig_weighting_sensitivity_dlnq_vs_sst.png")


if __name__ == "__main__":
    main()
