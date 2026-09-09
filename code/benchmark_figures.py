"""Figures for the benchmark profiles and support diagnostics.

Draws the profile, heat-map and support figures from the benchmark outputs.
Input and output locations are set by the environment variables listed below.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# Configuration. Paths are overridable through environment variables so a batch driver like run_warm_pool_full.py can point this script at one benchmark folder and any figure/table output directory without editing this file. With the variables unset, the defaults below give the single-run behavior: the 0.5 K benchmark, with figures and tables written under the working directory.
_DEFAULT_BENCH_DIR = r"results/benchmarks/sst_295-305_0.5K"

BENCHMARK_NC = Path(os.environ.get(
    "BENCHFIG_BENCHMARK_NC",
    os.path.join(_DEFAULT_BENCH_DIR, "benchmark_profiles.nc"),
))
SUMMARY_CSV = Path(os.environ.get(
    "BENCHFIG_SUMMARY_CSV",
    os.path.join(_DEFAULT_BENCH_DIR, "benchmark_summary.csv"),
))
COUNTS_CSV = Path(os.environ.get(
    "BENCHFIG_COUNTS_CSV",
    os.path.join(_DEFAULT_BENCH_DIR, "counts_summary.csv"),
))

FIG_DIR = Path(os.environ.get("BENCHFIG_FIG_DIR", "figures"))
TABLE_DIR = Path(os.environ.get("BENCHFIG_TABLE_DIR", "tables"))



# Helper functions

# Clean labels that may come out of NetCDF as bytes, strings, or NaN.
def clean_label(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip()
    return str(x).strip()


# Make sure important group labels are normal strings.
def add_clean_labels(df):
    for col in ["group_type", "sst_bin", "omega_bin", "dominant_olr_bin_mode"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_label)

    return df


# Sort SST bins in physical order.
# Handles labels such as "299-301"and "301–303" "303_305", "<299", ">305", and "ALL" If exact labels differ, this still usually works because it extracts the first number in the label.
def sst_sort_key(label):
    s = str(label)

    if s == "ALL":
        return -9999

    if s.startswith("<"):
        return -1000

    if s.startswith(">"):
        nums = [
            float(part)
            for part in s.replace(">", " ").replace("_", " ").replace("-", " ").split()
            if part.replace(".", "", 1).isdigit()
        ]
        return nums[0] + 1000 if nums else 9999

    # Normalize common separators
    s2 = s.replace("–", "-").replace("_", "-").replace("[", "").replace("]", "")
    pieces = s2.replace(",", " ").replace("(", " ").replace(")", " ").replace("-", " ").split()

    nums = []
    for p in pieces:
        try:
            nums.append(float(p))
        except ValueError:
            pass

    if len(nums) > 0:
        return nums[0]

    return 9999

# Convert labels like 'sst_300_303' into '300–303 K'.
def pretty_sst_label(label):
    s = str(label)

    if s.startswith("sst_"):
        s = s.replace("sst_", "")

    parts = s.split("_")

    if len(parts) == 2:
        return f"{parts[0]}–{parts[1]} K"

    return s

# Sort omega regimes in: ascent -> neutral -> subsidence.
def omega_sort_key(label):
    order = {
        "strong_ascent": 0,
        "very_strong_ascent": 0,
        "weak_ascent": 1,
        "neutral": 2,
        "weak_subsidence": 3,
        "strong_subsidence": 4,
        "very_strong_subsidence": 4,
        "ALL": -1,
    }
    return order.get(str(label), 999)


# Return benchmark_group indices matching a group_type.
def get_group_indices(ds, group_type):
    labels = np.array([clean_label(x) for x in ds["group_type"].values])
    return np.where(labels == group_type)[0]


# Extract a string variable from xarray Dataset.
def get_string_var(ds, var):
    return np.array([clean_label(x) for x in ds[var].values])


# Atmospheric pressure decreases upward, so invert y-axis.
def invert_pressure_axis(ax):
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (hPa)")
    ax.grid(True, alpha=0.3)

# Find common positive q-axis limits for profile panels.
def get_common_q_xlim(ds, group_type="sst_by_omega"):
    labels = np.array([clean_label(x) for x in ds["group_type"].values])
    idx = np.where(labels == group_type)[0]

    vals = []

    for i in idx:
        arr = ds["q_profile_p75"].isel(benchmark_group=i).values
        arr = arr[np.isfinite(arr) & (arr > 0)]
        vals.extend(arr)

    vals = np.array(vals)

    if len(vals) == 0:
        return None, None

    return np.nanmin(vals) * 0.8, np.nanmax(vals) * 1.2

# Save high-resolution figure.
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def dedupe_preserve_order(cols):
    return list(dict.fromkeys(cols))


# Load data


ds = summary = counts = plev = None


# Read the benchmark into the module-level names the figure functions draw from. Called from main() so that importing this module does no I/O.
def load() -> None:
    global ds, summary, counts, plev
    ds = xr.open_dataset(BENCHMARK_NC)
    summary = add_clean_labels(pd.read_csv(SUMMARY_CSV))

    if COUNTS_CSV.exists():
        counts = add_clean_labels(pd.read_csv(COUNTS_CSV))
    else:
        counts = None

    plev = ds["plev"].values


# Figure 1: Main benchmark profiles by SST

# Median UTWV profile by SST bin, with 25-75% spread envelope.
def plot_fig1_main_profiles_by_sst():
    idx = get_group_indices(ds, "sst_only")

    if len(idx) == 0:
        return

    sst_labels = get_string_var(ds, "sst_bin")[idx]
    order = np.argsort([sst_sort_key(x) for x in sst_labels])
    idx = idx[order]

    plt.figure(figsize=(7, 6))

    for i in idx:
        label = pretty_sst_label(clean_label(ds["sst_bin"].values[i]))
        n = int(ds["n_samples"].values[i])

        q_med = ds["q_profile_median"].isel(benchmark_group=i).values
        q_p25 = ds["q_profile_p25"].isel(benchmark_group=i).values
        q_p75 = ds["q_profile_p75"].isel(benchmark_group=i).values

        q_med = np.where(q_med > 0, q_med, np.nan)
        q_p25 = np.where(q_p25 > 0, q_p25, np.nan)
        q_p75 = np.where(q_p75 > 0, q_p75, np.nan)

        plt.plot(q_med, plev, label=f"{label} (n={n})")
        plt.fill_betweenx(plev, q_p25, q_p75, alpha=0.15)

    ax = plt.gca()
    invert_pressure_axis(ax)
    ax.set_xscale("log")
    ax.set_xlabel("UTWV q, log scale")
    ax.set_title("Benchmark UTWV profiles by SST bin")
    ax.legend(fontsize=8)

    savefig(FIG_DIR / "fig1_main_profiles_by_sst.png")

# Median ln(q) profile by SST bin, with 25-75% spread envelope.
def plot_fig1b_lnq_profiles_by_sst():
    required = [
        "lnq_profile_median",
        "lnq_profile_p25",
        "lnq_profile_p75",
    ]

    missing = [v for v in required if v not in ds]

    if missing:
        return

    idx = get_group_indices(ds, "sst_only")

    if len(idx) == 0:
        return

    sst_labels = get_string_var(ds, "sst_bin")[idx]
    order = np.argsort([sst_sort_key(x) for x in sst_labels])
    idx = idx[order]

    plt.figure(figsize=(7, 6))

    for i in idx:
        label = pretty_sst_label(clean_label(ds["sst_bin"].values[i]))
        n = int(ds["n_samples"].values[i])

        lnq_med = ds["lnq_profile_median"].isel(benchmark_group=i).values
        lnq_p25 = ds["lnq_profile_p25"].isel(benchmark_group=i).values
        lnq_p75 = ds["lnq_profile_p75"].isel(benchmark_group=i).values

        plt.plot(lnq_med, plev, label=f"{label} (n={n})")
        plt.fill_betweenx(plev, lnq_p25, lnq_p75, alpha=0.15)

    ax = plt.gca()
    invert_pressure_axis(ax)
    ax.set_xlabel("ln(q)")
    ax.set_title("Benchmark ln(q) profiles by SST bin")
    ax.legend(fontsize=8)

    savefig(FIG_DIR / "fig1b_lnq_profiles_by_sst.png")


# Figure 2: SST profiles within omega regimes

# One panel per omega regime, SST-bin profiles within each panel.
def plot_fig2_profiles_by_sst_within_omega():
    df = summary[summary["group_type"] == "sst_by_omega"].copy()

    if df.empty:
        return

    omega_bins = sorted(df["omega_bin"].dropna().unique(), key=omega_sort_key)

    xmin, xmax = get_common_q_xlim(ds, "sst_by_omega")

    n_panels = len(omega_bins)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows),
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, omega in zip(axes, omega_bins):
        sub = df[df["omega_bin"] == omega].copy()
        sub = sub.sort_values("sst_bin", key=lambda s: s.map(sst_sort_key))

        for _, row in sub.iterrows():
            group_id = int(row["group_id"])
            label = pretty_sst_label(row["sst_bin"])
            n = int(row["n_samples"])

            q_med = ds["q_profile_median"].isel(benchmark_group=group_id).values
            q_p25 = ds["q_profile_p25"].isel(benchmark_group=group_id).values
            q_p75 = ds["q_profile_p75"].isel(benchmark_group=group_id).values

            # Log-scale plotting requires positive values.
            q_med = np.where(q_med > 0, q_med, np.nan)
            q_p25 = np.where(q_p25 > 0, q_p25, np.nan)
            q_p75 = np.where(q_p75 > 0, q_p75, np.nan)

            ax.plot(q_med, plev, label=f"{label} (n={n})")
            ax.fill_betweenx(plev, q_p25, q_p75, alpha=0.12)

        invert_pressure_axis(ax)
        ax.set_xscale("log")

        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)

        ax.set_xlabel("UTWV q, log scale")
        ax.set_title(str(omega))
        ax.legend(fontsize=7)


    # Hide unused panels.
    for ax in axes[len(omega_bins):]:
        ax.axis("off")

    fig.suptitle("Benchmark UTWV profiles by SST within omega regimes", y=1.02)

    savefig(FIG_DIR / "fig2_profiles_by_sst_within_omega.png")


# Figure 3: dlnq vs SST

# Main dlnq plot by SST bin. This is descriptive spread.
def plot_fig3_dlnq_vs_sst():
    df = summary[summary["group_type"] == "sst_only"].copy()

    if df.empty:
        return

    df = df.sort_values("sst_bin", key=lambda s: s.map(sst_sort_key))

    x = np.arange(len(df))
    labels = [pretty_sst_label(x) for x in df["sst_bin"].values]

    y = df["median_dlnq"].values
    y_low = df["p25_dlnq"].values
    y_high = df["p75_dlnq"].values

    yerr = np.vstack([y - y_low, y_high - y])

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4)

    plt.axhline(0, linewidth=1, alpha=0.5)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Median dlnq = ln(q_UUT / q_LUT)")
    plt.ylim(-2.25, -1.25)
    plt.xlabel("SST bin")
    plt.title("dlnq by SST bin: lower values indicate stronger vertical contrast")
    plt.grid(True, alpha=0.3)

    savefig(FIG_DIR / "fig3_dlnq_vs_sst.png")


# Figure 4: dlnq heatmap by SST and omega

# Heatmap of dlnq across SST x omega regimes.
def plot_fig4_dlnq_heatmap_sst_omega():
    df = summary[summary["group_type"] == "sst_by_omega"].copy()

    if df.empty:
        return

    sst_bins = sorted(df["sst_bin"].dropna().unique(), key=sst_sort_key)
    omega_bins = sorted(df["omega_bin"].dropna().unique(), key=omega_sort_key)

    mat = np.full((len(omega_bins), len(sst_bins)), np.nan)

    for i, omega in enumerate(omega_bins):
        for j, sst in enumerate(sst_bins):
            row = df[(df["omega_bin"] == omega) & (df["sst_bin"] == sst)]
            if not row.empty:
                mat[i, j] = row["median_dlnq"].iloc[0]

    plt.figure(figsize=(1.3 * len(sst_bins) + 4, 1.0 * len(omega_bins) + 3))
    im = plt.imshow(mat, aspect="auto")

    plt.colorbar(im, label="Median dlnq")
    plt.xticks(
        np.arange(len(sst_bins)),
        [pretty_sst_label(x) for x in sst_bins],
        rotation=30,
        ha="right"
    )
    plt.yticks(np.arange(len(omega_bins)), omega_bins)

    plt.xlabel("SST bin")
    plt.ylabel("Omega regime")
    plt.title("Median dlnq by SST and omega: lower values indicate stronger vertical contrast")

    # Write values on cells.
    for i in range(len(omega_bins)):
        for j in range(len(sst_bins)):
            if np.isfinite(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    savefig(FIG_DIR / "fig4_dlnq_heatmap_sst_omega.png")

# Heatmap of dlnq change relative to the coolest SST bin within each omega regime.
def plot_fig4b_dlnq_anomaly_heatmap_sst_omega():
    df = summary[summary["group_type"] == "sst_by_omega"].copy()

    if df.empty:
        return

    sst_bins = sorted(df["sst_bin"].dropna().unique(), key=sst_sort_key)
    omega_bins = sorted(df["omega_bin"].dropna().unique(), key=omega_sort_key)

    mat = np.full((len(omega_bins), len(sst_bins)), np.nan)

    for i, omega in enumerate(omega_bins):
        for j, sst in enumerate(sst_bins):
            row = df[(df["omega_bin"] == omega) & (df["sst_bin"] == sst)]
            if not row.empty:
                mat[i, j] = row["median_dlnq"].iloc[0]

    # Use the first SST bin as the baseline within each omega regime.
    baseline = mat[:, [0]]
    anom = mat - baseline

    vmax = np.nanmax(np.abs(anom))

    plt.figure(figsize=(1.3 * len(sst_bins) + 4, 1.0 * len(omega_bins) + 3))
    im = plt.imshow(
        anom,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    plt.colorbar(im, label="Δ median dlnq")
    plt.xticks(
        np.arange(len(sst_bins)),
        [pretty_sst_label(x) for x in sst_bins],
        rotation=30,
        ha="right"
    )
    plt.yticks(np.arange(len(omega_bins)), omega_bins)

    plt.xlabel("SST bin")
    plt.ylabel("Omega regime")
    plt.title("Change in dlnq relative to 297–300 K within each omega regime")

    for i in range(len(omega_bins)):
        for j in range(len(sst_bins)):
            if np.isfinite(anom[i, j]):
                plt.text(
                    j,
                    i,
                    f"{anom[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    savefig(FIG_DIR / "fig4b_dlnq_anomaly_heatmap_sst_omega.png")


# Figure 5: mean OLR heatmap by SST and omega

# OLR diagnostic heatmap.
def plot_fig5_olr_mean_heatmap():
    df = summary[summary["group_type"] == "sst_by_omega"].copy()

    if df.empty:
        return

    sst_bins = sorted(df["sst_bin"].dropna().unique(), key=sst_sort_key)
    omega_bins = sorted(df["omega_bin"].dropna().unique(), key=omega_sort_key)

    mat = np.full((len(omega_bins), len(sst_bins)), np.nan)

    for i, omega in enumerate(omega_bins):
        for j, sst in enumerate(sst_bins):
            row = df[(df["omega_bin"] == omega) & (df["sst_bin"] == sst)]
            if not row.empty:
                mat[i, j] = row["mean_olr"].iloc[0]

    plt.figure(figsize=(1.3 * len(sst_bins) + 4, 1.0 * len(omega_bins) + 3))
    im = plt.imshow(mat, aspect="auto", cmap="viridis_r")

    plt.colorbar(im, label="Mean OLR (W m$^{-2}$)")
    plt.xticks(
        np.arange(len(sst_bins)),
        [pretty_sst_label(x) for x in sst_bins],
        rotation=30,
        ha="right"
    )
    plt.yticks(np.arange(len(omega_bins)), omega_bins)

    plt.xlabel("SST bin")
    plt.ylabel("Omega regime")
    plt.title("Mean OLR by SST and omega: lower values indicate colder cloud tops")

    for i in range(len(omega_bins)):
        for j in range(len(sst_bins)):
            if np.isfinite(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8)

    savefig(FIG_DIR / "fig5_olr_mean_heatmap.png")


# Figure 6: OLR fraction diagnostic panels

# True OLR class composition by SST and omega.
def plot_fig6_olr_true_fraction_panels():
    df = summary[summary["group_type"] == "sst_by_omega"].copy()

    if df.empty:
        return

    frac_cols = [
        "frac_olr_deep_convective",
        "frac_olr_convective_high_cloud",
        "frac_olr_transition",
        "frac_olr_clear_sky",
    ]

    frac_cols = [c for c in frac_cols if c in df.columns]

    if len(frac_cols) == 0:
        return

    if "olr_fraction_source" in df.columns:
        sources = sorted(df["olr_fraction_source"].dropna().unique())

    sst_bins = sorted(df["sst_bin"].dropna().unique(), key=sst_sort_key)
    omega_bins = sorted(df["omega_bin"].dropna().unique(), key=omega_sort_key)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = axes.reshape(-1)
    im = None

    for ax, col in zip(axes, frac_cols):
        mat = np.full((len(omega_bins), len(sst_bins)), np.nan)

        for i, omega in enumerate(omega_bins):
            for j, sst in enumerate(sst_bins):
                row = df[(df["omega_bin"] == omega) & (df["sst_bin"] == sst)]
                if not row.empty:
                    mat[i, j] = row[col].iloc[0]

        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(sst_bins)))
        ax.set_xticklabels(
            [pretty_sst_label(x) for x in sst_bins],
            rotation=30,
            ha="right",
        )

        ax.set_yticks(np.arange(len(omega_bins)))
        ax.set_yticklabels(omega_bins)

        title = col.replace("frac_olr_", "").replace("_", " ")
        ax.set_title(title)

        for i in range(len(omega_bins)):
            for j in range(len(sst_bins)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(
        im,
        ax=axes.tolist(),
        location="right",
        shrink=0.85,
        pad=0.02,
        label="Mean within-sample OLR fraction",
    )

    fig.suptitle(
        "Mean within-sample OLR class fractions by SST and omega",
        fontsize=16,
    )

    plt.savefig(
        FIG_DIR / "fig6_olr_true_fraction_panels.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# Older dominant-class OLR diagnostic.
def plot_fig6b_dominant_olr_fraction_panels():
    df = summary[summary["group_type"] == "sst_by_omega"].copy()

    if df.empty:
        return

    frac_cols = [
        "frac_dominant_olr_deep_convective",
        "frac_dominant_olr_convective_high_cloud",
        "frac_dominant_olr_transition",
        "frac_dominant_olr_clear_sky",
    ]

    frac_cols = [c for c in frac_cols if c in df.columns]

    if len(frac_cols) == 0:
        return

    sst_bins = sorted(df["sst_bin"].dropna().unique(), key=sst_sort_key)
    omega_bins = sorted(df["omega_bin"].dropna().unique(), key=omega_sort_key)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = axes.reshape(-1)
    im = None

    for ax, col in zip(axes, frac_cols):
        mat = np.full((len(omega_bins), len(sst_bins)), np.nan)

        for i, omega in enumerate(omega_bins):
            for j, sst in enumerate(sst_bins):
                row = df[(df["omega_bin"] == omega) & (df["sst_bin"] == sst)]
                if not row.empty:
                    mat[i, j] = row[col].iloc[0]

        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(sst_bins)))
        ax.set_xticklabels(
            [pretty_sst_label(x) for x in sst_bins],
            rotation=30,
            ha="right",
        )

        ax.set_yticks(np.arange(len(omega_bins)))
        ax.set_yticklabels(omega_bins)

        title = col.replace("frac_dominant_olr_", "").replace("_", " ")
        ax.set_title(title)

        for i in range(len(omega_bins)):
            for j in range(len(sst_bins)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(
        im,
        ax=axes.tolist(),
        location="right",
        shrink=0.85,
        pad=0.02,
        label="Fraction of cell-regime aggregates",
    )

    fig.suptitle(
        "Dominant OLR class fractions by SST and omega",
        fontsize=16,
    )

    plt.savefig(
        FIG_DIR / "fig6b_olr_dominant_fraction_panels.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# Figure 7: sample support heatmap

# Sample count heatmap.
def plot_fig7_sample_support_heatmap():
    df = summary[summary["group_type"] == "sst_by_omega"].copy()

    if df.empty:
        return

    sst_bins = sorted(df["sst_bin"].dropna().unique(), key=sst_sort_key)
    omega_bins = sorted(df["omega_bin"].dropna().unique(), key=omega_sort_key)

    mat = np.full((len(omega_bins), len(sst_bins)), np.nan)

    for i, omega in enumerate(omega_bins):
        for j, sst in enumerate(sst_bins):
            row = df[(df["omega_bin"] == omega) & (df["sst_bin"] == sst)]
            if not row.empty:
                mat[i, j] = row["n_samples"].iloc[0]

    plt.figure(figsize=(1.3 * len(sst_bins) + 4, 1.0 * len(omega_bins) + 3))
    im = plt.imshow(mat, aspect="auto")

    plt.colorbar(im, label="Number of cell-regime aggregates")
    plt.xticks(
        np.arange(len(sst_bins)),
        [pretty_sst_label(x) for x in sst_bins],
        rotation=30,
        ha="right"
    )
    plt.yticks(np.arange(len(omega_bins)), omega_bins)

    plt.xlabel("SST bin")
    plt.ylabel("Omega regime")
    plt.title("Sample support by SST and omega")

    for i in range(len(omega_bins)):
        for j in range(len(sst_bins)):
            if np.isfinite(mat[i, j]):
                plt.text(j, i, f"{int(mat[i, j]):,}", ha="center", va="center", fontsize=8)

    savefig(FIG_DIR / "fig7_sample_support_heatmap.png")

# Raw contributing MLS profile count heatmap.
# The primary independent support metric is still n_samples, because the benchmark summarizes independent aggregates.
def plot_fig8_raw_profile_support_heatmap():
    df = summary[summary["group_type"] == "sst_by_omega"].copy()

    if df.empty:
        return

    if "n_raw_profiles_total" not in df.columns:
        return

    sst_bins = sorted(df["sst_bin"].dropna().unique(), key=sst_sort_key)
    omega_bins = sorted(df["omega_bin"].dropna().unique(), key=omega_sort_key)

    mat = np.full((len(omega_bins), len(sst_bins)), np.nan)

    for i, omega in enumerate(omega_bins):
        for j, sst in enumerate(sst_bins):
            row = df[(df["omega_bin"] == omega) & (df["sst_bin"] == sst)]
            if not row.empty:
                mat[i, j] = row["n_raw_profiles_total"].iloc[0]

    plt.figure(figsize=(1.3 * len(sst_bins) + 4, 1.0 * len(omega_bins) + 3))
    im = plt.imshow(mat, aspect="auto")

    plt.colorbar(im, label="Raw contributing MLS profiles")
    plt.xticks(
        np.arange(len(sst_bins)),
        [pretty_sst_label(x) for x in sst_bins],
        rotation=30,
        ha="right",
    )
    plt.yticks(np.arange(len(omega_bins)), omega_bins)

    plt.xlabel("SST bin")
    plt.ylabel("Omega regime")
    plt.title("Raw profile support by SST and omega")

    for i in range(len(omega_bins)):
        for j in range(len(sst_bins)):
            if np.isfinite(mat[i, j]):
                plt.text(j, i, f"{int(mat[i, j]):,}", ha="center", va="center", fontsize=8)

    savefig(FIG_DIR / "fig8_raw_profile_support_heatmap.png")


# Better pressure-level validity diagnostic.
def plot_fig9_valid_q_fraction_by_pressure():
    var = "n_valid_q_profile_total"

    if var not in ds:
        return

    if "n_raw_profiles_total" not in ds:
        return

    idx = get_group_indices(ds, "sst_only")

    if len(idx) == 0:
        return

    sst_labels = get_string_var(ds, "sst_bin")[idx]
    order = np.argsort([sst_sort_key(x) for x in sst_labels])
    idx = idx[order]

    plt.figure(figsize=(7, 6))

    for i in idx:
        label = pretty_sst_label(clean_label(ds["sst_bin"].values[i]))
        n = int(ds["n_samples"].values[i])

        valid_total = ds[var].isel(benchmark_group=i).values
        raw_total = float(ds["n_raw_profiles_total"].isel(benchmark_group=i).values)

        valid_frac = valid_total / raw_total

        plt.plot(valid_frac, plev, label=f"{label} (n={n})")

    ax = plt.gca()
    invert_pressure_axis(ax)
    ax.set_xlabel("Fraction of raw q values valid")
    ax.set_title("q validity fraction by pressure level")
    ax.legend(fontsize=8)

    ax.set_xlim(0.95, 1.005)

    savefig(FIG_DIR / "fig9_valid_q_fraction_by_pressure.png")


# Tables for conclusions and diagnosis

# Create compact CSV tables that are easy to inspect and use.
def make_conclusion_tables():
    key_cols = [
        "group_id",
        "group_type",
        "sst_bin",
        "omega_bin",
        "n_samples",
        "n_raw_profiles_total",
        "mean_profiles_per_sample",
        "mean_sst",
        "mean_omega",
        "mean_olr",
        "median_dlnq",
        "p25_dlnq",
        "p75_dlnq",
        "dominant_olr_bin_mode",
        "olr_fraction_source",
        "frac_olr_deep_convective",
        "frac_olr_convective_high_cloud",
        "frac_olr_transition",
        "frac_olr_clear_sky",
    ]

    key_cols = dedupe_preserve_order([c for c in key_cols if c in summary.columns])

    paper_key = summary[key_cols].copy()
    paper_key.to_csv(TABLE_DIR / "paper_key_results.csv", index=False)

    olr_cols = [
        "group_id",
        "group_type",
        "sst_bin",
        "omega_bin",
        "mean_olr",
        "median_olr",
        "dominant_olr_bin_mode",
        "olr_fraction_source",
        "n_olr_total",

        "frac_olr_deep_convective",
        "frac_olr_convective_high_cloud",
        "frac_olr_transition",
        "frac_olr_clear_sky",
        "frac_olr_other_or_unexpected",

        "frac_dominant_olr_deep_convective",
        "frac_dominant_olr_convective_high_cloud",
        "frac_dominant_olr_transition",
        "frac_dominant_olr_clear_sky",
        "frac_dominant_olr_other_or_unexpected",

        "raw_frac_olr_deep_convective",
        "raw_frac_olr_convective_high_cloud",
        "raw_frac_olr_transition",
        "raw_frac_olr_clear_sky",

        "n_olr_deep_convective",
        "n_olr_convective_high_cloud",
        "n_olr_transition",
        "n_olr_clear_sky",

        "n_samples",
        "n_raw_profiles_total",
    ]

    olr_cols = dedupe_preserve_order([c for c in olr_cols if c in summary.columns])

    olr_diag = summary[summary["group_type"] == "sst_by_omega"][olr_cols].copy()
    olr_diag.to_csv(TABLE_DIR / "olr_diagnostic_matrix.csv", index=False)

    support_cols = [
        "group_id",
        "group_type",
        "sst_bin",
        "omega_bin",
        "n_samples",
        "n_raw_profiles_total",
        "mean_profiles_per_sample",
        "median_profiles_per_sample",
        "min_profiles_per_sample",
        "max_profiles_per_sample",
        "mean_sst",
        "mean_omega",
        "mean_olr",
    ]

    support_cols = [c for c in support_cols if c in summary.columns]


    support = summary[support_cols].copy()

    support["low_support"] = (
            (support["n_samples"] < 50) |
            (support["n_raw_profiles_total"] < 500)
    )

    support.to_csv(TABLE_DIR / "support_matrix.csv", index=False)
    support.to_csv(TABLE_DIR / "sample_support_matrix.csv", index=False)


    valid_rows = []

    for i in range(ds.sizes["benchmark_group"]):
        row = {
            "group_id": int(ds["group_id"].isel(benchmark_group=i).values),
            "group_type": clean_label(ds["group_type"].isel(benchmark_group=i).values),
            "sst_bin": clean_label(ds["sst_bin"].isel(benchmark_group=i).values),
            "omega_bin": clean_label(ds["omega_bin"].isel(benchmark_group=i).values),
        }

        for var in [
            "n_valid_q_profile_total",
            "n_valid_lnq_profile_total",
            "n_valid_q_profile_mean_per_sample",
            "n_valid_lnq_profile_mean_per_sample",
        ]:
            if var in ds:
                vals = ds[var].isel(benchmark_group=i).values
                row[f"{var}_min"] = float(np.nanmin(vals))
                row[f"{var}_mean"] = float(np.nanmean(vals))
                row[f"{var}_max"] = float(np.nanmax(vals))

        valid_rows.append(row)

    valid_support = pd.DataFrame(valid_rows)
    valid_support.to_csv(TABLE_DIR / "valid_profile_support_table.csv", index=False)


# Run everything

def main() -> None:
    load()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plot_fig1_main_profiles_by_sst()
    plot_fig1b_lnq_profiles_by_sst()
    plot_fig2_profiles_by_sst_within_omega()
    plot_fig3_dlnq_vs_sst()
    plot_fig4_dlnq_heatmap_sst_omega()
    plot_fig4b_dlnq_anomaly_heatmap_sst_omega()
    plot_fig5_olr_mean_heatmap()
    plot_fig6_olr_true_fraction_panels()
    plot_fig6b_dominant_olr_fraction_panels()
    plot_fig7_sample_support_heatmap()
    plot_fig8_raw_profile_support_heatmap()
    plot_fig9_valid_q_fraction_by_pressure()
    make_conclusion_tables()


if __name__ == "__main__":
    main()
