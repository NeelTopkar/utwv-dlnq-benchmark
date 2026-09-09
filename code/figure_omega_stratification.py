r"""
Figure 2: warm-pool vertical moisture contrast stratified by large-scale vertical motion.

Panel A  mean dlnq for the coarse SST bins, by omega regime.
Panel B  change in median dlnq relative to the 300.0-300.5 K bin, 0.5 K SST bins,
         by omega regime.

Replaces the previous hand-assembled figure, whose two panels came from different
scripts with inconsistent conventions (warm_pool_figures.py used a cell-weighted
mean and the axis label "omega regime" spelled with the Greek symbol, while the
heatmap in conditional_tables.py used a median and spelled it out).  Both panels
are now drawn here with one convention and one label set.

Run with the project venv (numpy/pandas/xarray/matplotlib).
"""
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("UTWV_DATA_ROOT", ".")
WP_COARSE = os.path.join(ROOT, "data/aggregates/warm_pool",
                         "sst_297-307_coarse",
                         "aggregated_gridcells.nc")
WP_FINE = os.path.join(ROOT, "data/aggregates/warm_pool",
                       "sst_295-305_0.5K",
                       "aggregated_gridcells.nc")
OUT = os.environ.get("UTWV_FIG2_OUT",
                     os.path.join(ROOT, "figures", "warm_pool", "fig2_dlnq_by_sst_and_omega.png"))

# Physical ordering, ascent at the top.
OMEGA_ORDER = ["strong_ascent", "weak_ascent", "neutral",
               "weak_subsidence", "strong_subsidence"]
OMEGA_LABEL = {"strong_ascent": "strong ascent", "weak_ascent": "weak ascent",
               "neutral": "near-neutral", "weak_subsidence": "weak subsidence",
               "strong_subsidence": "strong subsidence"}

# Panel A drops SST bins with thin support. The warm-pool 305-307 K bin holds 10 cell-regime aggregates from 31 profiles and is excluded from inference in the manuscript, so it is excluded here by the same rule.
MIN_AGGREGATES = 30
REFERENCE_BIN = "sst_300_300.5"


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _bin_sort_key(label):
    nums = re.findall(r"[-+]?\d*\.?\d+", label)
    return float(nums[0]) if nums else 1e9


def _pretty(label):
    return label.replace("sst_", "").replace("_", "-")


def load(path):
    ds = xr.open_dataset(path)
    df = pd.DataFrame({
        "dlnq": ds["dlnq_mean"].values,
        "sst_bin": _dec(ds["sst_bin"].values),
        "omega": _dec(ds["omega_bin"].values),
        "quut": ds["q_uut_mean"].values,
        "qlut": ds["q_lut_mean"].values,
    })
    ds.close()
    ok = np.isfinite(df.dlnq) & (df.quut > 0) & (df.qlut > 0)
    return df[ok].reset_index(drop=True)


def panel_a(ax):
    df = load(WP_COARSE)
    counts = df.sst_bin.value_counts()
    bins = [b for b in sorted(counts.index, key=_bin_sort_key) if counts[b] >= MIN_AGGREGATES]
    piv = (df.groupby(["omega", "sst_bin"])["dlnq"].mean()
             .reset_index().pivot(index="omega", columns="sst_bin", values="dlnq"))
    piv = piv.reindex(index=[o for o in OMEGA_ORDER if o in piv.index], columns=bins)

    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([_pretty(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([OMEGA_LABEL.get(o, o) for o in piv.index])
    ax.set_xlabel("SST bin (K)")
    ax.set_ylabel("ω$_{500}$ regime")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="white")
    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("mean dlnq")
    return piv


def panel_b(ax):
    df = load(WP_FINE)
    # Same support rule as panel A, applied per omega row: a fine SST bin is shown only where every vertical-motion category carries at least MIN_AGGREGATES cell-regime aggregates. Below about 299 K the warm pool holds only a handful of aggregates per category, and the resulting medians are not interpretable.
    support = df.groupby(["omega", "sst_bin"]).size().unstack(fill_value=0)
    well_sampled = [c for c in support.columns if support[c].min() >= MIN_AGGREGATES]
    df = df[df.sst_bin.isin(well_sampled)]

    piv = (df.groupby(["omega", "sst_bin"])["dlnq"].median()
             .reset_index().pivot(index="omega", columns="sst_bin", values="dlnq"))
    cols = sorted(piv.columns, key=_bin_sort_key)
    piv = piv.reindex(index=[o for o in OMEGA_ORDER if o in piv.index], columns=cols)
    if REFERENCE_BIN not in piv.columns:
        raise RuntimeError(f"reference bin {REFERENCE_BIN} absent; have {list(piv.columns)}")
    anom = piv.sub(piv[REFERENCE_BIN], axis=0)

    vmax = float(np.nanmax(np.abs(anom.values)))
    im = ax.imshow(anom.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(anom.columns)))
    ax.set_xticklabels([_pretty(c) for c in anom.columns], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(anom.index)))
    ax.set_yticklabels([OMEGA_LABEL.get(o, o) for o in anom.index])
    ax.set_xlabel("SST bin (K)")
    ax.set_ylabel("ω$_{500}$ regime")
    for i in range(anom.shape[0]):
        for j in range(anom.shape[1]):
            v = anom.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("median dlnq minus 300.0-300.5 K value\npositive = flatter, negative = steeper")
    return anom


def main():
    fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                             gridspec_kw={"height_ratios": [1.0, 1.25]})
    a = panel_a(axes[0])
    b = panel_b(axes[1])
    for ax, tag in zip(axes, "AB"):
        ax.text(-0.075, 1.04, tag, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT}")
    print("\nPanel A - mean dlnq")
    print(a.round(3).to_string())
    print("\nPanel B - median anomaly vs 300.0-300.5 K (strong_ascent row)")
    print(b.loc["strong_ascent"].round(2).to_string())


if __name__ == "__main__":
    main()
