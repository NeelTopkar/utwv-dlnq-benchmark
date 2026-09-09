"""Descriptive dlnq tables by SST and by SST and vertical motion.

Summarizes the index within each SST bin and each SST-by-vertical-motion cell,
and expresses fine-bin values as differences from a configurable reference bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import re
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

# Matplotlib is only needed for figures. The Agg backend lets the script run on servers or terminals that do not have a display window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Main aggregated input file.
_default_input_path = r"data/aggregates/warm_pool/sst_295-305_0.5K/aggregated_gridcells.nc"

# Main SST comparison. These must exactly match SST bins that already exist in the selected aggregated file.
REFERENCE_SST_BOUNDS = (300,300.5)
WARM_SST_BOUNDS = (300.5, 301)

CONDITIONAL_OUTPUT_DIR = r"results/conditional"
CONDITIONAL_FIGURES_DIR = r"results/conditional/figures"

# Optional label written into the CSV/NetCDF to identify the binning experiment.
BENCHMARK_NAME = "current_benchmark"

# but single mode keeps the old folder names and avoids extra subfolders.
RUN_MODE = "single"  # options: "single" or "batch"


# Configuration for the conditional tables.
@dataclass
class ConditionalConfig:

    # Name for this benchmark/binning experiment. This is used only for labeling outputs and combined comparisons.
    benchmark_name: str = "current_benchmark"

    # Main input file from the aggregation.
    input_path: Path = Path(
        _default_input_path
    )

    # Output directory. The script will create this folder if it does not already exist.
    output_dir: Path = Path(CONDITIONAL_OUTPUT_DIR)

    # Figure output directory.
    figures_dir: Path = Path(CONDITIONAL_FIGURES_DIR)

    # Reference and warm SST bins used for the main comparison. The code is flexible about labels, but these numerical bounds define what it searches for in labels such as "300-303", "[300, 303)", etc.
    reference_sst_bounds: tuple[float, float] = REFERENCE_SST_BOUNDS
    warm_sst_bounds: tuple[float, float] = WARM_SST_BOUNDS

    # Used only for text warnings/conclusions, not filtering.
    min_samples_warning: int = 30

    # Small descriptive threshold for labeling a dlnq change as nearly flat. This is NOT a significance threshold; bootstrap_intervals.py handles robustness.
    small_dlnq_threshold: float = 0.02

    # Preferred order for common omega-regime labels. Unknown labels still work.
    omega_order: tuple[str, ...] = (
        "strong_ascent",
        "weak_ascent",
        "neutral",
        "weak_subsidence",
        "strong_subsidence",
    )

    # Optional variable aliases, in case earlier pipeline scripts used a slightly different variable name. The first present name is used.
    aliases: dict[str, tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.aliases is None:
            self.aliases = {
                "sst_bin": ("sst_bin", "sst_label", "sst_group"),
                "omega_bin": ("omega_bin", "omega_label", "w500_bin", "w_bin"),
                "q_uut_mean": ("q_uut_mean", "uut_q_mean", "q_uut"),
                "q_lut_mean": ("q_lut_mean", "lut_q_mean", "q_lut"),
                "dlnq_mean": ("dlnq_mean", "delta_lnq_mean", "flattening_mean", "dlnq"),
                "dlnq_median": ("dlnq_median", "delta_lnq_median", "flattening_median"),
                "mean_sst": ("mean_sst", "sst_mean", "sst"),
                "mean_omega": ("mean_omega", "omega_mean", "w500_mean", "w500"),
                "mean_olr": ("mean_olr", "olr_mean", "olr"),
                "dominant_olr_bin": ("dominant_olr_bin", "olr_dominant_bin", "olr_bin"),
                "dominant_olr_fraction": (
                    "dominant_olr_fraction",
                    "olr_dominant_fraction",
                    "olr_fraction",
                ),
                "n_profiles": ("n_profiles", "n_raw_profiles", "profile_count"),
                "region": ("region", "region_name", "basin", "ocean_region"),
            }


# General helpers

# Return the first variable/coordinate name present in an xarray Dataset.
def _first_present(ds: xr.Dataset, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.variables or name in ds.coords:
            return name
    return None


# Find a required variable using aliases, or raise a clear error.
def _require(ds: xr.Dataset, logical_name: str, cfg: ConditionalConfig) -> str:
    name = _first_present(ds, cfg.aliases[logical_name])
    if name is None:
        raise KeyError(
            f"Required variable '{logical_name}' was not found. "
            f"Tried aliases: {cfg.aliases[logical_name]}\n"
            f"Available variables: {list(ds.variables)}"
        )
    return name


# Find an optional variable using aliases.
def _optional(ds: xr.Dataset, logical_name: str, cfg: ConditionalConfig) -> Optional[str]:
    return _first_present(ds, cfg.aliases[logical_name])


# Convert a DataArray to a 1D numpy array. the aggregation scalar sample variables should be 1D over sample. If the variable is not reducible to 1D, the script raises an error so the issue is caught early.
def _to_1d_numpy(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values).squeeze()
    if arr.ndim != 1:
        raise ValueError(
            f"Variable {da.name!r} is expected to be 1D after squeeze, "
            f"but has shape {arr.shape}."
        )
    return arr


# Decode byte strings and keep numeric values unchanged.
def _decode_values(values: np.ndarray) -> list:
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(v)
    return out


# Coerce a pandas Series to numeric, preserving NaN for bad values.
def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# Extract first two numbers from an SST-bin label. Works for labels such as: "300-303" "300_303" "[300, 303)" "sst_300_303" Returns None if the label does not contain at least two numbers.
def _extract_bounds(label) -> tuple[float, float] | None:
    text = str(label)
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


# Sort SST-bin labels by their lower and upper numerical bounds.
def _sst_sort_key(label) -> tuple[float, float, str]:
    bounds = _extract_bounds(label)
    if bounds is None:
        return (math.inf, math.inf, str(label))
    return (bounds[0], bounds[1], str(label))


# Sort omega labels in the expected physical order.
def _omega_sort_key(label, cfg: ConditionalConfig) -> tuple[int, str]:
    text = str(label)
    if text in cfg.omega_order:
        return (cfg.omega_order.index(text), text)
    return (len(cfg.omega_order), text)


# Find the label whose numerical bounds match target_bounds.
def _find_sst_label(labels: Iterable, target_bounds: tuple[float, float]) -> Optional[str]:
    lo_t, hi_t = target_bounds
    for label in labels:
        bounds = _extract_bounds(label)
        if bounds is None:
            continue
        lo, hi = bounds
        if abs(lo - lo_t) < 1e-6 and abs(hi - hi_t) < 1e-6:
            return str(label)
    return None


# Return the mode and its fraction among non-null values.
def _mode_and_fraction(series: pd.Series) -> tuple[object, float]:
    clean = series.dropna()
    if len(clean) == 0:
        return np.nan, np.nan
    counts = clean.value_counts(dropna=True)
    mode = counts.index[0]
    fraction = float(counts.iloc[0] / counts.sum())
    return mode, fraction


# Quantile helper that returns NaN for empty/all-NaN groups.
def _safe_quantile(series: pd.Series, q: float) -> float:
    clean = _as_numeric(series).dropna()
    if len(clean) == 0:
        return np.nan
    return float(clean.quantile(q))


# Pretty formatting for report text.
def _format_float(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "NaN"
    return f"{float(x):.{digits}f}"


# Load aggregated samples

# Load the aggregated file into a clean sample-level DataFrame.
def load_aggregate_samples(cfg: ConditionalConfig) -> pd.DataFrame:
    ds = xr.open_dataset(cfg.input_path)

    # Required variables.
    sst_bin_name = _require(ds, "sst_bin", cfg)
    q_uut_name = _require(ds, "q_uut_mean", cfg)
    q_lut_name = _require(ds, "q_lut_mean", cfg)

    # Optional variables.
    omega_bin_name = _optional(ds, "omega_bin", cfg)
    dlnq_mean_name = _optional(ds, "dlnq_mean", cfg)
    dlnq_median_name = _optional(ds, "dlnq_median", cfg)
    mean_sst_name = _optional(ds, "mean_sst", cfg)
    mean_omega_name = _optional(ds, "mean_omega", cfg)
    mean_olr_name = _optional(ds, "mean_olr", cfg)
    dominant_olr_bin_name = _optional(ds, "dominant_olr_bin", cfg)
    dominant_olr_fraction_name = _optional(ds, "dominant_olr_fraction", cfg)
    n_profiles_name = _optional(ds, "n_profiles", cfg)
    region_name = _optional(ds, "region", cfg)

    # Assemble a simple DataFrame. This keeps the tables fast because we only use scalar sample-level quantities, not full vertical arrays.
    data = {}
    data["sst_bin"] = _decode_values(_to_1d_numpy(ds[sst_bin_name]))
    data["q_uut_mean"] = _to_1d_numpy(ds[q_uut_name])
    data["q_lut_mean"] = _to_1d_numpy(ds[q_lut_name])

    if omega_bin_name is not None:
        data["omega_bin"] = _decode_values(_to_1d_numpy(ds[omega_bin_name]))
    else:
        data["omega_bin"] = "all_omega"
        warnings.warn("omega_bin not found; only SST-only summaries will be meaningful.")

    if dlnq_mean_name is not None:
        data["dlnq_mean"] = _to_1d_numpy(ds[dlnq_mean_name])
    else:
        # Recompute the metric if the aggregated file does not already contain it. This is safe only where both q values are positive and finite.
        q_uut = np.asarray(data["q_uut_mean"], dtype=float)
        q_lut = np.asarray(data["q_lut_mean"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            data["dlnq_mean"] = np.log(q_uut / q_lut)
        warnings.warn("dlnq_mean not found; recomputed from q_uut_mean and q_lut_mean.")

    if dlnq_median_name is not None:
        data["dlnq_median"] = _to_1d_numpy(ds[dlnq_median_name])
    else:
        # If absent, keep the column so downstream code is simple.
        data["dlnq_median"] = np.nan

    if mean_sst_name is not None:
        data["mean_sst"] = _to_1d_numpy(ds[mean_sst_name])
    else:
        data["mean_sst"] = np.nan

    if mean_omega_name is not None:
        data["mean_omega"] = _to_1d_numpy(ds[mean_omega_name])
    else:
        data["mean_omega"] = np.nan

    if mean_olr_name is not None:
        data["mean_olr"] = _to_1d_numpy(ds[mean_olr_name])
    else:
        data["mean_olr"] = np.nan

    if dominant_olr_bin_name is not None:
        data["dominant_olr_bin"] = _decode_values(_to_1d_numpy(ds[dominant_olr_bin_name]))
    else:
        data["dominant_olr_bin"] = np.nan

    if dominant_olr_fraction_name is not None:
        data["dominant_olr_fraction"] = _to_1d_numpy(ds[dominant_olr_fraction_name])
    else:
        data["dominant_olr_fraction"] = np.nan

    if n_profiles_name is not None:
        data["n_profiles"] = _to_1d_numpy(ds[n_profiles_name])
    else:
        data["n_profiles"] = 1
        warnings.warn("n_profiles not found; raw-profile support totals will equal n_samples.")

    if region_name is not None:
        data["region"] = _decode_values(_to_1d_numpy(ds[region_name]))
    else:
        data["region"] = np.nan

    df = pd.DataFrame(data)

    # Normalize dtypes.
    for col in [
        "q_uut_mean", "q_lut_mean", "dlnq_mean", "dlnq_median", "mean_sst",
        "mean_omega", "mean_olr", "dominant_olr_fraction", "n_profiles",
    ]:
        df[col] = _as_numeric(df[col])

    # Remove samples where the core metric cannot be interpreted.
    before = len(df)
    valid = (
        np.isfinite(df["dlnq_mean"])
        & np.isfinite(df["q_uut_mean"])
        & np.isfinite(df["q_lut_mean"])
        & (df["q_uut_mean"] > 0)
        & (df["q_lut_mean"] > 0)
        & df["sst_bin"].notna()
    )
    df = df.loc[valid].copy()
    after = len(df)
    print(f"  usable samples: {after:,} of {before:,}", flush=True)

    # Store sort keys for consistent plotting/tables.
    df["sst_bin"] = df["sst_bin"].astype(str)
    df["omega_bin"] = df["omega_bin"].astype(str)
    df["sst_sort_lo"] = df["sst_bin"].map(lambda x: _sst_sort_key(x)[0])
    df["sst_sort_hi"] = df["sst_bin"].map(lambda x: _sst_sort_key(x)[1])
    df["omega_sort"] = df["omega_bin"].map(lambda x: _omega_sort_key(x, cfg)[0])

    return df


# Summaries

# Compute one row of the conditional summary statistics for a group.
def summarize_group(group: pd.DataFrame, group_cols: list[str], group_type: str) -> dict:
    row: dict = {"group_type": group_type}

    # Copy group labels into the output row.
    for col in group_cols:
        row[col] = group[col].iloc[0]

    # Include blank columns so all summary rows have a common schema.
    for col in ["region", "sst_bin", "omega_bin"]:
        row.setdefault(col, "all" if col != "region" else np.nan)

    row["n_samples"] = int(len(group))
    row["n_raw_profiles_total"] = float(group["n_profiles"].sum(skipna=True))
    row["mean_profiles_per_sample"] = float(group["n_profiles"].mean(skipna=True))

    d = _as_numeric(group["dlnq_mean"])
    row["dlnq_mean_of_samples"] = float(d.mean(skipna=True))
    row["dlnq_median_of_samples"] = float(d.median(skipna=True))
    row["dlnq_std"] = float(d.std(skipna=True, ddof=1)) if len(d.dropna()) > 1 else np.nan
    row["dlnq_p10"] = _safe_quantile(d, 0.10)
    row["dlnq_p25"] = _safe_quantile(d, 0.25)
    row["dlnq_p75"] = _safe_quantile(d, 0.75)
    row["dlnq_p90"] = _safe_quantile(d, 0.90)
    row["dlnq_iqr"] = row["dlnq_p75"] - row["dlnq_p25"]

    q_uut = _as_numeric(group["q_uut_mean"])
    q_lut = _as_numeric(group["q_lut_mean"])
    row["q_uut_mean_group"] = float(q_uut.mean(skipna=True))
    row["q_uut_median_group"] = float(q_uut.median(skipna=True))
    row["q_uut_p25"] = _safe_quantile(q_uut, 0.25)
    row["q_uut_p75"] = _safe_quantile(q_uut, 0.75)
    row["q_lut_mean_group"] = float(q_lut.mean(skipna=True))
    row["q_lut_median_group"] = float(q_lut.median(skipna=True))
    row["q_lut_p25"] = _safe_quantile(q_lut, 0.25)
    row["q_lut_p75"] = _safe_quantile(q_lut, 0.75)

    # Environmental diagnostics.
    for var in ["mean_sst", "mean_omega", "mean_olr", "dominant_olr_fraction"]:
        vals = _as_numeric(group[var])
        row[f"{var}_group_mean"] = float(vals.mean(skipna=True))
        row[f"{var}_group_median"] = float(vals.median(skipna=True))

    mode, frac = _mode_and_fraction(group["dominant_olr_bin"])
    row["dominant_olr_bin_group"] = mode
    row["dominant_olr_bin_group_fraction"] = frac

    # Sorting helpers retained in CSV for transparency.
    row["sst_sort_lo"] = float(group["sst_sort_lo"].iloc[0])
    row["sst_sort_hi"] = float(group["sst_sort_hi"].iloc[0])
    row["omega_sort"] = int(group["omega_sort"].iloc[0]) if "omega_bin" in group_cols else -1

    return row


# Create all group summaries.
def make_summaries(df: pd.DataFrame, cfg: ConditionalConfig) -> pd.DataFrame:
    rows: list[dict] = []

    # 1. Main analysis: SST only.
    for _, g in df.groupby(["sst_bin"], dropna=False, sort=False):
        rows.append(summarize_group(g, ["sst_bin"], "sst_only"))

    # 2. Regime-separated analysis: SST x omega.
    for _, g in df.groupby(["sst_bin", "omega_bin"], dropna=False, sort=False):
        rows.append(summarize_group(g, ["sst_bin", "omega_bin"], "sst_omega"))

    # 3. Optional regional analysis, only if a real region variable exists.
    has_region = df["region"].notna().any() and not (df["region"].astype(str) == "nan").all()
    if has_region:
        for _, g in df.groupby(["region", "sst_bin"], dropna=False, sort=False):
            rows.append(summarize_group(g, ["region", "sst_bin"], "region_sst"))

        for _, g in df.groupby(["region", "sst_bin", "omega_bin"], dropna=False, sort=False):
            rows.append(summarize_group(g, ["region", "sst_bin", "omega_bin"], "region_sst_omega"))

    summary = pd.DataFrame(rows)

    # Sort rows so the CSV is readable and stable across runs.
    summary = summary.sort_values(
        by=["group_type", "region", "sst_sort_lo", "sst_sort_hi", "omega_sort", "omega_bin"],
        na_position="last",
    ).reset_index(drop=True)

    summary = add_reference_differences(summary, cfg)
    return summary


# Add differences relative to the 300-303 K reference bin. For SST-only groups, each SST bin is compared against the SST-only 300-303 K group.
def add_reference_differences(summary: pd.DataFrame, cfg: ConditionalConfig) -> pd.DataFrame:
    out = summary.copy()

    # This must be an object/string column because the reference SST bin is a label like "sst_300_303", not a number.
    out["reference_sst_bin"] = pd.Series(
        pd.NA,
        index=out.index,
        dtype="object"
    )

    # These stay numeric because they store differences.
    out["delta_dlnq_mean_vs_ref"] = np.nan
    out["delta_dlnq_median_vs_ref"] = np.nan
    out["delta_q_uut_mean_vs_ref"] = np.nan
    out["delta_q_lut_mean_vs_ref"] = np.nan
    out["delta_mean_olr_vs_ref"] = np.nan

    ref_lo, ref_hi = cfg.reference_sst_bounds

    # Determine comparison keys. These define the context in which the reference bin should be searched.
    key_map = {
        "sst_only": [],
        "sst_omega": ["omega_bin"],
        "region_sst": ["region"],
        "region_sst_omega": ["region", "omega_bin"],
    }

    for group_type, key_cols in key_map.items():
        sub_idx = out.index[out["group_type"] == group_type]
        if len(sub_idx) == 0:
            continue

        sub = out.loc[sub_idx]
        if key_cols:
            grouped = sub.groupby(key_cols, dropna=False, sort=False)
            iterable = grouped
        else:
            iterable = [((), sub)]

        for _, context in iterable:
            ref_mask = context.apply(
                lambda r: (
                    abs(float(r["sst_sort_lo"]) - ref_lo) < 1e-6
                    and abs(float(r["sst_sort_hi"]) - ref_hi) < 1e-6
                ),
                axis=1,
            )
            if not ref_mask.any():
                continue
            ref = context.loc[ref_mask].iloc[0]
            context_indices = context.index

            out.loc[context_indices, "reference_sst_bin"] = str(ref["sst_bin"])
            out.loc[context_indices, "delta_dlnq_mean_vs_ref"] = (
                out.loc[context_indices, "dlnq_mean_of_samples"] - ref["dlnq_mean_of_samples"]
            )
            out.loc[context_indices, "delta_dlnq_median_vs_ref"] = (
                out.loc[context_indices, "dlnq_median_of_samples"] - ref["dlnq_median_of_samples"]
            )
            out.loc[context_indices, "delta_q_uut_mean_vs_ref"] = (
                out.loc[context_indices, "q_uut_mean_group"] - ref["q_uut_mean_group"]
            )
            out.loc[context_indices, "delta_q_lut_mean_vs_ref"] = (
                out.loc[context_indices, "q_lut_mean_group"] - ref["q_lut_mean_group"]
            )
            out.loc[context_indices, "delta_mean_olr_vs_ref"] = (
                out.loc[context_indices, "mean_olr_group_mean"] - ref["mean_olr_group_mean"]
            )

    return out


# Figures

def _prepare_sst_only(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary.loc[summary["group_type"] == "sst_only"].copy()
    s = s.sort_values(["sst_sort_lo", "sst_sort_hi"])
    return s


def _prepare_sst_omega(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary.loc[summary["group_type"] == "sst_omega"].copy()
    s = s.sort_values(["omega_sort", "sst_sort_lo", "sst_sort_hi"])
    return s


# Plot median dlnq by SST bin with IQR bars.
def plot_dlnq_vs_sst(summary: pd.DataFrame, cfg: ConditionalConfig) -> None:
    s = _prepare_sst_only(summary)
    if s.empty:
        return

    x = np.arange(len(s))
    y = s["dlnq_median_of_samples"].to_numpy(float)
    lower = y - s["dlnq_p25"].to_numpy(float)
    upper = s["dlnq_p75"].to_numpy(float) - y

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y, yerr=[lower, upper], marker="o", capsize=4)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(s["sst_bin"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Median Δln(q) across cell-regime aggregates")
    ax.set_xlabel("SST bin")
    ax.set_title("dlnq by SST bin")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = cfg.figures_dir / "fig_dlnq_vs_sst.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Plot median dlnq by SST bin with one line per omega regime.
def plot_dlnq_by_sst_and_omega(summary: pd.DataFrame, cfg: ConditionalConfig) -> None:
    s = _prepare_sst_omega(summary)
    if s.empty:
        return

    # Consistent SST axis across all omega regimes.
    sst_labels = sorted(s["sst_bin"].unique(), key=_sst_sort_key)
    x_map = {label: i for i, label in enumerate(sst_labels)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for omega, g in s.groupby("omega_bin", sort=False):
        g = g.sort_values(["sst_sort_lo", "sst_sort_hi"])
        x = [x_map[v] for v in g["sst_bin"]]
        y = g["dlnq_median_of_samples"].to_numpy(float)
        ax.plot(x, y, marker="o", label=str(omega))

    ax.axhline(0, linewidth=1)
    ax.set_xticks(np.arange(len(sst_labels)))
    ax.set_xticklabels([str(v) for v in sst_labels], rotation=30, ha="right")
    ax.set_ylabel("Median Δln(q) across cell-regime aggregates")
    ax.set_xlabel("SST bin")
    ax.set_title("dlnq by SST and omega regime")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Omega regime", fontsize=8)
    fig.tight_layout()
    path = cfg.figures_dir / "fig_dlnq_by_sst_and_omega.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Create a simple SST x omega heatmap of median dlnq.
def plot_dlnq_heatmap(summary: pd.DataFrame, cfg: ConditionalConfig) -> None:
    s = _prepare_sst_omega(summary)
    if s.empty:
        return

    sst_labels = sorted(s["sst_bin"].unique(), key=_sst_sort_key)
    omega_labels = sorted(s["omega_bin"].unique(), key=lambda x: _omega_sort_key(x, cfg))

    pivot = s.pivot_table(
        index="omega_bin",
        columns="sst_bin",
        values="dlnq_median_of_samples",
        aggfunc="first",
    ).reindex(index=omega_labels, columns=sst_labels)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto")
    ax.set_xticks(np.arange(len(sst_labels)))
    ax.set_xticklabels([str(v) for v in sst_labels], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(omega_labels)))
    ax.set_yticklabels([str(v) for v in omega_labels])
    ax.set_xlabel("SST bin")
    ax.set_ylabel("Omega regime")
    ax.set_title("Median Δln(q) by SST and omega")

    # Put values directly on the heatmap so it is readable without guessing.
    arr = pivot.to_numpy(float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median Δln(q)")
    fig.tight_layout()
    path = cfg.figures_dir / "fig_dlnq_heatmap_sst_omega.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

# Create a paper-oriented SST x omega heatmap of delta dlnq. This figure is better for paper interpretation because it directly shows flatter-vs-steeper behavior relative to the selected reference SST bin.
def plot_delta_dlnq_heatmap(summary: pd.DataFrame, cfg: ConditionalConfig) -> None:
    s = _prepare_sst_omega(summary)

    if s.empty:
        return

    if "delta_dlnq_median_vs_ref" not in s.columns:
        return

    if s["delta_dlnq_median_vs_ref"].isna().all():
        return

    # Consistent axis order.
    sst_labels = sorted(s["sst_bin"].unique(), key=_sst_sort_key)
    omega_labels = sorted(
        s["omega_bin"].unique(),
        key=lambda x: _omega_sort_key(x, cfg)
    )

    # Pivot into omega x SST matrix.
    pivot = s.pivot_table(
        index="omega_bin",
        columns="sst_bin",
        values="delta_dlnq_median_vs_ref",
        aggfunc="first",
    ).reindex(index=omega_labels, columns=sst_labels)

    arr = pivot.to_numpy(dtype=float)

    # Choose symmetric color limits around zero. This makes positive and negative changes visually comparable.
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size == 0:
        return

    vmax = float(np.nanmax(np.abs(finite_vals)))

    # Avoid a broken color scale if all values are exactly zero.
    if vmax == 0:
        vmax = 1e-6

    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10, 5.5))

    im = ax.imshow(
        arr,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(sst_labels)))
    ax.set_xticklabels([str(v) for v in sst_labels], rotation=30, ha="right")

    ax.set_yticks(np.arange(len(omega_labels)))
    ax.set_yticklabels([str(v) for v in omega_labels])

    ax.set_xlabel("SST bin")
    ax.set_ylabel("Omega regime")

    ref_text = f"{cfg.reference_sst_bounds[0]:g}–{cfg.reference_sst_bounds[1]:g} K"
    ax.set_title(
        f"Δ median Δln(q) relative to {ref_text}"
    )

    # Put values directly on the heatmap. Positive = flatter than reference. Negative = steeper than reference.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    # Add a vertical reference note under the title by labeling colorbar clearly.
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Δ median Δln(q) vs reference\npositive = flatter, negative = steeper"
    )

    # Add a zero-contour-like visual guide through the colorbar center by using the diverging color scale centered at 0.
    ax.grid(False)

    fig.tight_layout()

    path = cfg.figures_dir / "fig_delta_dlnq_heatmap_sst_omega.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Plot q_UUT and q_LUT separately to diagnose what drives dlnq.
def plot_uut_lut_by_sst(summary: pd.DataFrame, cfg: ConditionalConfig) -> None:
    s = _prepare_sst_only(summary)
    if s.empty:
        return

    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, s["q_uut_median_group"].to_numpy(float), marker="o", label="q_UUT median")
    ax.plot(x, s["q_lut_median_group"].to_numpy(float), marker="o", label="q_LUT median")
    ax.set_xticks(x)
    ax.set_xticklabels(s["sst_bin"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("q layer mean, median across cell-regime aggregates")
    ax.set_xlabel("SST bin")
    ax.set_title("UUT and LUT layer means by SST bin")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = cfg.figures_dir / "fig_uut_lut_by_sst.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Plot mean OLR diagnostic by SST x omega, if OLR is present.
def plot_olr_by_sst_omega(summary: pd.DataFrame, cfg: ConditionalConfig) -> None:
    s = _prepare_sst_omega(summary)
    if s.empty or s["mean_olr_group_mean"].isna().all():
        return

    sst_labels = sorted(s["sst_bin"].unique(), key=_sst_sort_key)
    omega_labels = sorted(s["omega_bin"].unique(), key=lambda x: _omega_sort_key(x, cfg))

    pivot = s.pivot_table(
        index="omega_bin",
        columns="sst_bin",
        values="mean_olr_group_mean",
        aggfunc="first",
    ).reindex(index=omega_labels, columns=sst_labels)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto")
    ax.set_xticks(np.arange(len(sst_labels)))
    ax.set_xticklabels([str(v) for v in sst_labels], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(omega_labels)))
    ax.set_yticklabels([str(v) for v in omega_labels])
    ax.set_xlabel("SST bin")
    ax.set_ylabel("Omega regime")
    ax.set_title("Diagnostic: mean OLR by SST and omega")

    arr = pivot.to_numpy(float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean OLR")
    fig.tight_layout()
    path = cfg.figures_dir / "fig_olr_by_sst_omega.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Create all first-pass figures.
def make_figures(summary: pd.DataFrame, cfg: ConditionalConfig) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    # Main diagnostic figures.
    plot_dlnq_vs_sst(summary, cfg)
    plot_dlnq_by_sst_and_omega(summary, cfg)

    # Raw dlnq heatmap: useful diagnostic, shows actual dlnq values.
    plot_dlnq_heatmap(summary, cfg)

    # Delta dlnq heatmap: paper-oriented, shows flatter/steeper vs reference.
    plot_delta_dlnq_heatmap(summary, cfg)

    # Layer and OLR diagnostics.
    plot_uut_lut_by_sst(summary, cfg)
    plot_olr_by_sst_omega(summary, cfg)


# Conclusions report

# Plain-English interpretation of a dlnq difference.
def _describe_change(delta: float, cfg: ConditionalConfig) -> str:
    if pd.isna(delta):
        return "could not be computed"
    if delta > cfg.small_dlnq_threshold:
        return "less negative / flatter"
    if delta < -cfg.small_dlnq_threshold:
        return "more negative / steeper"
    return "nearly unchanged descriptively"


# Write a text report with descriptive conclusions and caveats.
def write_conclusions(summary: pd.DataFrame, cfg: ConditionalConfig) -> None:
    lines: list[str] = []
    lines.append("CONDITIONAL dlnq ANALYSIS REPORT")
    lines.append("================================")
    lines.append("")
    lines.append("Metric:")
    lines.append("  dlnq = ln(q_UUT / q_LUT)")
    lines.append("  Less negative / closer to zero means a flatter vertical q profile.")
    lines.append("  More negative means a steeper decrease from LUT to UUT.")
    lines.append("")
    lines.append("Important caveat:")
    lines.append("  These are descriptive results; bootstrap_intervals.py is still needed for confidence intervals.")
    lines.append("")

    ref_label = _find_sst_label(
        summary.loc[summary["group_type"] == "sst_only", "sst_bin"].unique(),
        cfg.reference_sst_bounds,
    )
    warm_label = _find_sst_label(
        summary.loc[summary["group_type"] == "sst_only", "sst_bin"].unique(),
        cfg.warm_sst_bounds,
    )

    sst_only = _prepare_sst_only(summary)
    lines.append("1. SST-only steepening summary")
    lines.append("------------------------------")
    if sst_only.empty:
        lines.append("  No SST-only rows were available.")
    else:
        for _, r in sst_only.iterrows():
            lines.append(
                f"  {r['sst_bin']}: n={int(r['n_samples'])}, "
                f"median dlnq={_format_float(r['dlnq_median_of_samples'])}, "
                f"IQR=[{_format_float(r['dlnq_p25'])}, {_format_float(r['dlnq_p75'])}]"
            )

        if ref_label is None:
            lines.append(
                f"  WARNING: could not find reference SST bin {cfg.reference_sst_bounds}; "
                "warm-bin comparison was not computed."
            )
        if warm_label is None:
            lines.append(
                f"  WARNING: could not find warm SST bin {cfg.warm_sst_bounds}; "
                "warm-bin comparison was not computed."
            )
        if ref_label is not None and warm_label is not None:
            warm_row = sst_only.loc[sst_only["sst_bin"].astype(str) == str(warm_label)]
            if not warm_row.empty:
                r = warm_row.iloc[0]
                delta = r["delta_dlnq_median_vs_ref"]
                lines.append("")
                lines.append(
                    f"  Main descriptive comparison: {warm_label} minus {ref_label}"
                )
                lines.append(
                    f"  Δ median dlnq = {_format_float(delta)} -> {_describe_change(delta, cfg)}"
                )
                lines.append(
                    f"  Δ q_UUT mean = {_format_float(r['delta_q_uut_mean_vs_ref'])}"
                )
                lines.append(
                    f"  Δ q_LUT mean = {_format_float(r['delta_q_lut_mean_vs_ref'])}"
                )
                lines.append(
                    f"  Δ mean OLR = {_format_float(r['delta_mean_olr_vs_ref'])}"
                )
                if r["n_samples"] < cfg.min_samples_warning:
                    lines.append(
                        f"  WARNING: warm-bin n_samples={int(r['n_samples'])}, below "
                        f"the warning threshold {cfg.min_samples_warning}."
                    )
    lines.append("")

    lines.append("2. SST x omega regime comparison")
    lines.append("---------------------------------")
    sst_omega = _prepare_sst_omega(summary)
    if sst_omega.empty:
        lines.append("  No SST x omega rows were available.")
    else:
        omega_labels = sorted(sst_omega["omega_bin"].unique(), key=lambda x: _omega_sort_key(x, cfg))
        for omega in omega_labels:
            g = sst_omega.loc[sst_omega["omega_bin"] == omega].copy()
            ref = _find_sst_label(g["sst_bin"].unique(), cfg.reference_sst_bounds)
            warm = _find_sst_label(g["sst_bin"].unique(), cfg.warm_sst_bounds)
            if ref is None or warm is None:
                lines.append(f"  {omega}: missing reference or warm bin; no comparison.")
                continue
            wr = g.loc[g["sst_bin"].astype(str) == str(warm)].iloc[0]
            delta = wr["delta_dlnq_median_vs_ref"]
            lines.append(
                f"  {omega}: {warm} minus {ref}: "
                f"Δ median dlnq={_format_float(delta)} -> {_describe_change(delta, cfg)}; "
                f"n_warm={int(wr['n_samples'])}"
            )
            if wr["n_samples"] < cfg.min_samples_warning:
                lines.append(
                    f"    WARNING: n_warm below {cfg.min_samples_warning}; treat this regime cautiously."
                )
    lines.append("")

    lines.append("3. OLR diagnostic note")
    lines.append("----------------------")
    if summary["mean_olr_group_mean"].isna().all():
        lines.append("  mean_olr was absent or all NaN, so no OLR diagnostic conclusion is available.")
    else:
        lines.append(
            "  OLR was summarized only as a diagnostic, not as a primary grouping axis. "
            "Use these values to interpret convective/radiative state, but keep the main "
            "benchmark organized by SST and omega."
        )
        # Report the most dominant OLR group in the SST x omega table.
        candidate = summary.loc[summary["group_type"] == "sst_omega"].copy()
        if not candidate.empty and candidate["dominant_olr_bin_group_fraction"].notna().any():
            idx = candidate["dominant_olr_bin_group_fraction"].idxmax()
            r = candidate.loc[idx]
            lines.append(
                f"  Highest OLR-bin dominance among SST x omega groups: "
                f"sst_bin={r['sst_bin']}, omega_bin={r['omega_bin']}, "
                f"dominant_olr_bin={r['dominant_olr_bin_group']}, "
                f"fraction={_format_float(r['dominant_olr_bin_group_fraction'], 3)}."
            )
    lines.append("")

    lines.append("4. Recommended next step")
    lines.append("------------------------")
    lines.append(
        "  Run bootstrap_intervals.py over aggregated samples, not raw profiles, "
        "to place confidence intervals on these dlnq differences."
    )

    report = "\n".join(lines)
    path = cfg.output_dir / "conclusions.txt"
    path.write_text(report, encoding="utf-8")


# Output writers

# Write the summary CSV and NetCDF files. The NetCDF writer below is intentionally careful with string columns.
def write_summary_outputs(summary: pd.DataFrame, cfg: ConditionalConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Always write CSV first.
    csv_path = cfg.output_dir / "steepening_summary.csv"
    summary.to_csv(csv_path, index=False)

    # 2. Write NetCDF in a NetCDF-safe way.
    nc_path = cfg.output_dir / "steepening_summary.nc"

    nc_df = summary.copy().reset_index(drop=True)
    n = len(nc_df)

    data_vars = {}

    for col in nc_df.columns:
        s = nc_df[col]

        # Handle string/object/categorical columns.
        if (
            s.dtype == object
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
        ):
            # Convert every value to a plain Python string.
            clean_strings = []
            for v in s.tolist():
                if pd.isna(v):
                    clean_strings.append("")
                else:
                    clean_strings.append(str(v))

            # NetCDF-safe fixed-length byte strings. The max length must be at least 1.
            max_len = max(1, max(len(x) for x in clean_strings))
            arr = np.asarray(clean_strings, dtype=f"S{max_len}")

            data_vars[col] = (("summary_group",), arr)

        # Handle boolean columns if any appear later.
        elif pd.api.types.is_bool_dtype(s):
            arr = s.fillna(False).to_numpy(dtype=np.int8)
            data_vars[col] = (("summary_group",), arr)

        # Handle numeric columns.
        else:
            arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
            data_vars[col] = (("summary_group",), arr)

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "summary_group": np.arange(n, dtype=np.int32)
        },
        attrs={
            "description": "Conditional dlnq summary from aggregated samples",
            "metric": "dlnq = ln(q_UUT / q_LUT)",
            "note": "Descriptive summaries only; bootstrap_intervals.py is needed for confidence intervals.",
            "csv_companion_file": str(csv_path),
        },
    )

    try:
        ds_out.to_netcdf(nc_path)
    except Exception as exc:
        warnings.warn(
            f"Could not write NetCDF summary: {exc}. "
            f"CSV output was still written at {csv_path}."
        )


# Main entry point

# Run the full conditional-table analysis.
def run_conditional(cfg: ConditionalConfig) -> pd.DataFrame:

    # otherwise fail later inside xarray with a less obvious error message.
    if not cfg.input_path.exists():
        raise FileNotFoundError(
            "Could not find the aggregated NetCDF file.\n"
            f"Current input_path is:\n  {cfg.input_path}\n\n"
            "Fix the input_path inside ConditionalConfig or PYCHARM_CONFIG at the bottom of this file."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_aggregate_samples(cfg)
    summary = make_summaries(df, cfg)
    write_summary_outputs(summary, cfg)
    make_figures(summary, cfg)
    write_conclusions(summary, cfg)

    return summary


# That section controls:
#   - _default_input_path/input file
#   - REFERENCE_SST_BOUNDS
#   - WARM_SST_BOUNDS
#   - output folders

# Single-run config This keeps the same folder names as before: results/conditional results/conditional/figures
SINGLE_CONFIG = ConditionalConfig(
    benchmark_name=BENCHMARK_NAME,
    input_path=Path(_default_input_path),
    output_dir=Path(CONDITIONAL_OUTPUT_DIR),
    figures_dir=Path(CONDITIONAL_FIGURES_DIR),
    reference_sst_bounds=REFERENCE_SST_BOUNDS,
    warm_sst_bounds=WARM_SST_BOUNDS,
    min_samples_warning=30,
)


# Optional batch configs
# Leave RUN_MODE = "single" unless you intentionally want to run several
# differently binned aggregated files in one execution.
#
# To use batch mode:
#   2. Add one ConditionalConfig block below for each aggregated benchmark file.
#   3. Use unique output_dir/figures_dir folders for each batch item, otherwise
#      each run will overwrite the previous run's files.
BATCH_CONFIGS = [
    # Example only. Uncomment and edit if you want batch mode later. ConditionalConfig( benchmark_name="bins_297_300_303_305_307", input_path=Path(r"data/aggregates/tropical_oceans/sst_297-307_coarse/aggregated_gridcells.nc"), output_dir=Path(r"results/conditional/bins_297_300_303_305_307"), figures_dir=Path(r"results/conditional/bins_297_300_303_305_307/figures"), reference_sst_bounds=(300.0, 303.0), warm_sst_bounds=(303.0, 305.0), min_samples_warning=30, ),
]

COMBINED_OUTPUT_DIR = Path(
    r"results/conditional/combined_sst_bin_comparisons"
)


# Command-line parser.
def parse_args() -> ConditionalConfig:
    defaults = ConditionalConfig()

    parser = argparse.ArgumentParser(
        description="Conditional dlnq analysis for the MLS + ERA5 UTWV pipeline."
    )
    parser.add_argument(
        "--input",
        default=str(defaults.input_path),
        help="Path to aggregated_gridcells.nc. Default comes from the EASY PYCHARM SETTINGS section.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(defaults.output_dir),
        help="Directory for the CSV/NetCDF/report outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(defaults.figures_dir),
        help="Directory for the figures.",
    )
    parser.add_argument(
        "--reference-low",
        type=float,
        default=defaults.reference_sst_bounds[0],
        help="Lower edge of reference SST bin.",
    )
    parser.add_argument(
        "--reference-high",
        type=float,
        default=defaults.reference_sst_bounds[1],
        help="Upper edge of reference SST bin.",
    )
    parser.add_argument(
        "--warm-low",
        type=float,
        default=defaults.warm_sst_bounds[0],
        help="Lower edge of warm SST bin.",
    )
    parser.add_argument(
        "--warm-high",
        type=float,
        default=defaults.warm_sst_bounds[1],
        help="Upper edge of warm SST bin.",
    )
    parser.add_argument(
        "--min-samples-warning",
        type=int,
        default=defaults.min_samples_warning,
        help="Warn in conclusions when a key bin has fewer than this many samples.",
    )
    args = parser.parse_args()

    return ConditionalConfig(
        benchmark_name=BENCHMARK_NAME,
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        figures_dir=Path(args.figures_dir),
        reference_sst_bounds=(args.reference_low, args.reference_high),
        warm_sst_bounds=(args.warm_low, args.warm_high),
        min_samples_warning=args.min_samples_warning,
    )


if __name__ == "__main__":
    if RUN_MODE == "single":
        run_conditional(SINGLE_CONFIG)

    elif RUN_MODE == "batch":
        if not BATCH_CONFIGS:
            raise ValueError(
                "RUN_MODE is set to 'batch', but BATCH_CONFIGS is empty. "
                "Either set RUN_MODE = 'single' near the top, or add ConditionalConfig blocks to BATCH_CONFIGS."
            )
        run_conditional_batch(BATCH_CONFIGS, COMBINED_OUTPUT_DIR)

    else:
        raise ValueError("RUN_MODE must be either 'single' or 'batch'.")
