"""Bootstrap confidence intervals for the benchmark contrasts.

Resamples either individual cell-regime aggregates or whole spatial tiles and
reports percentile intervals for mean dlnq by SST bin and for warm-minus-
reference contrasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import argparse
import math
import re
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Main aggregated file.
AGGREGATE_INPUT_PATH = r"data/aggregates/tropical_oceans/sst_297-307_coarse/aggregated_gridcells.nc"

# Main SST comparison. These must match SST bins present in the selected aggregate file. For the paper's headline contrast use (300, 303) K vs (303, 305) K.
REFERENCE_SST_BOUNDS = (300.0, 303.0)
WARM_SST_BOUNDS = (303.0, 305.0)

# Output folders.
INTERVAL_OUTPUT_DIR = r"results/bootstrap"
INTERVAL_FIGURES_DIR = r"results/bootstrap/figures"

# Bootstrap controls.
N_REPLICATES = 1000      # 500 for quick tests, 1000 for the paper, 2000 if fast
RANDOM_SEED = 12345      # fixed seed -> reproducible CIs
CI_LOW = 2.5             # lower percentile for the 95% interval
CI_HIGH = 97.5           # upper percentile for the 95% interval

# Resampling method.
#   "iid"   : resample individual cell-regime aggregates with replacement (default).
#   "block" : resample whole BLOCKS of correlated samples with replacement.
#             Use this to stop spatially/temporally correlated samples from being
#             treated as independent, which otherwise makes the CIs too narrow.

BOOTSTRAP_METHOD = "iid"   # "iid" or "block"

# Block definition, used only when BOOTSTRAP_METHOD == "block".
#   "gridcell" : all samples sharing one EXACT lat/lon cell move together.
#                If each cell appears once per group this collapses to i.i.d.
#   "spatial"  : group NEIGHBORING cells into coarse SPATIAL_BLOCK_DEG tiles and
#                resample whole tiles. This is the right choice when samples are
#                time-aggregated gridcells and the remaining dependence is the
#                spatial correlation between adjacent cells.
# There is no time-based block: the aggregate is a time-pooled cell-regime
# climatology (Section 2.4), so every sample spans the whole record.
BLOCK_BY = "spatial"           # "gridcell" or "spatial"

# Tile size (degrees) for BLOCK_BY == "spatial". Larger tiles assume correlation over a wider area and give wider, more conservative CIs.
# Applies only when RUN_COMPARISON is False; the comparison path below fixes its two block runs at 10 and 20 degrees.
SPATIAL_BLOCK_DEG = 20.0

# Set True to run THREE methods in one execution — i.i.d., spatial-block 10°, and spatial-block 20° — and produce a side-by-side comparison of confidence- interval widths. Each method's full outputs go in its own subfolder (iid/, block_spatial_10deg/, block_spatial_20deg/) and a three-way comparison table + figures are written at the top level. When True, BOOTSTRAP_METHOD and BLOCK_BY above are ignored (all three are run). Spatial blocking is used because exact 'gridcell' collapses to i.i.d. when each cell appears once per group.
RUN_COMPARISON = True

# Optional quality gate: drop cell-regime aggregates built from very few raw profiles before bootstrapping. 0 disables the filter.
MIN_PROFILES_PER_SAMPLE = 0

# Label written into outputs so you know which binning experiment produced them.
BENCHMARK_NAME = "current_benchmark"


# Configuration object
@dataclass
class IntervalConfig:
    benchmark_name: str = BENCHMARK_NAME
    input_path: Path = Path(AGGREGATE_INPUT_PATH)
    output_dir: Path = Path(INTERVAL_OUTPUT_DIR)
    figures_dir: Path = Path(INTERVAL_FIGURES_DIR)

    reference_sst_bounds: tuple[float, float] = REFERENCE_SST_BOUNDS
    warm_sst_bounds: tuple[float, float] = WARM_SST_BOUNDS

    n_reps: int = N_REPLICATES
    seed: int = RANDOM_SEED
    ci_low: float = CI_LOW
    ci_high: float = CI_HIGH

    bootstrap_method: str = BOOTSTRAP_METHOD   # "iid" or "block"
    block_by: str = BLOCK_BY                   # "gridcell" or "spatial"
    spatial_block_deg: float = SPATIAL_BLOCK_DEG

    min_profiles_per_sample: int = MIN_PROFILES_PER_SAMPLE
    min_samples_warning: int = 30

    omega_order: tuple[str, ...] = (
        "strong_ascent",
        "weak_ascent",
        "neutral",
        "weak_subsidence",
        "strong_subsidence",
    )

    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.aliases:
            self.aliases = {
                "sst_bin": ("sst_bin", "sst_label", "sst_group"),
                "omega_bin": ("omega_bin", "omega_label", "w500_bin", "w_bin"),
                "q_uut_mean": ("q_uut_mean", "uut_q_mean", "q_uut"),
                "q_lut_mean": ("q_lut_mean", "lut_q_mean", "q_lut"),
                "dlnq_mean": ("dlnq_mean", "delta_lnq_mean", "flattening_mean", "dlnq"),
                "mean_sst": ("mean_sst", "sst_mean", "sst"),
                "mean_omega": ("mean_omega", "omega_mean", "w500_mean", "w500"),
                "mean_olr": ("mean_olr", "olr_mean", "olr"),
                "n_profiles": ("n_profiles", "n_raw_profiles", "profile_count"),
                "region": ("region", "region_name", "basin", "ocean_region"),
                # block-bootstrap coordinates
                "lat_bin_center": ("lat_bin_center", "lat", "latitude", "lat_center"),
                "lon_bin_center": ("lon_bin_center", "lon", "longitude", "lon_center"),
                # profile arrays
                "q_mean_profile": ("q_mean_profile", "q_profile_mean", "q_mean"),
                "lnq_mean_profile": ("lnq_mean_profile", "lnq_profile_mean", "lnq_mean"),
                "plev": ("plev", "pressure", "lev", "level", "p"),
            }


# General helpers (shared style with conditional_tables.py)

def _first_present(ds: xr.Dataset, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.variables or name in ds.coords:
            return name
    return None


def _require(ds: xr.Dataset, logical_name: str, cfg: IntervalConfig) -> str:
    name = _first_present(ds, cfg.aliases[logical_name])
    if name is None:
        raise KeyError(
            f"Required interval variable '{logical_name}' was not found. "
            f"Tried aliases: {cfg.aliases[logical_name]}\n"
            f"Available variables: {list(ds.variables)}"
        )
    return name


def _optional(ds: xr.Dataset, logical_name: str, cfg: IntervalConfig) -> Optional[str]:
    return _first_present(ds, cfg.aliases[logical_name])


def _to_1d_numpy(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values).squeeze()
    if arr.ndim != 1:
        raise ValueError(
            f"Variable {da.name!r} expected 1D after squeeze, got shape {arr.shape}."
        )
    return arr


def _decode_values(values: np.ndarray) -> list:
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(v)
    return out


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _extract_bounds(label) -> tuple[float, float] | None:
    nums = re.findall(r"[-+]?\d*\.?\d+", str(label))
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def _sst_sort_key(label) -> tuple[float, float, str]:
    bounds = _extract_bounds(label)
    if bounds is None:
        return (math.inf, math.inf, str(label))
    return (bounds[0], bounds[1], str(label))


def _omega_sort_key(label, cfg: IntervalConfig) -> tuple[int, str]:
    text = str(label)
    if text in cfg.omega_order:
        return (cfg.omega_order.index(text), text)
    return (len(cfg.omega_order), text)


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


def _format_float(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NaN"
    return f"{float(x):.{digits}f}"


# Load aggregated samples + optional profile arrays

@dataclass
class AggregateData:
    df: pd.DataFrame                     # scalar sample-level table, reset index
    plev: Optional[np.ndarray]           # pressure levels (n_plev,)
    q_profiles: Optional[np.ndarray]     # (n_valid_samples, n_plev) or None
    lnq_profiles: Optional[np.ndarray]   # (n_valid_samples, n_plev) or None


def load_aggregates(cfg: IntervalConfig) -> AggregateData:
    ds = xr.open_dataset(cfg.input_path)

    # Required scalars.
    sst_bin_name = _require(ds, "sst_bin", cfg)
    q_uut_name = _require(ds, "q_uut_mean", cfg)
    q_lut_name = _require(ds, "q_lut_mean", cfg)

    sample_dim = ds[sst_bin_name].dims[0]
    n_sample = ds.sizes[sample_dim]

    # Optional scalars.
    omega_bin_name = _optional(ds, "omega_bin", cfg)
    dlnq_mean_name = _optional(ds, "dlnq_mean", cfg)
    mean_sst_name = _optional(ds, "mean_sst", cfg)
    mean_omega_name = _optional(ds, "mean_omega", cfg)
    mean_olr_name = _optional(ds, "mean_olr", cfg)
    n_profiles_name = _optional(ds, "n_profiles", cfg)
    region_name = _optional(ds, "region", cfg)

    data: dict[str, object] = {}
    data["sst_bin"] = _decode_values(_to_1d_numpy(ds[sst_bin_name]))
    data["q_uut_mean"] = _to_1d_numpy(ds[q_uut_name])
    data["q_lut_mean"] = _to_1d_numpy(ds[q_lut_name])

    if omega_bin_name is not None:
        data["omega_bin"] = _decode_values(_to_1d_numpy(ds[omega_bin_name]))
    else:
        data["omega_bin"] = ["all_omega"] * n_sample
        warnings.warn("omega_bin not found; only SST-only summaries will be meaningful.")

    if dlnq_mean_name is not None:
        data["dlnq_mean"] = _to_1d_numpy(ds[dlnq_mean_name])
    else:
        q_uut = np.asarray(data["q_uut_mean"], dtype=float)
        q_lut = np.asarray(data["q_lut_mean"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            data["dlnq_mean"] = np.log(q_uut / q_lut)
        warnings.warn("dlnq_mean not found; recomputed from q_uut_mean / q_lut_mean.")

    data["mean_sst"] = _to_1d_numpy(ds[mean_sst_name]) if mean_sst_name else np.full(n_sample, np.nan)
    data["mean_omega"] = _to_1d_numpy(ds[mean_omega_name]) if mean_omega_name else np.full(n_sample, np.nan)
    data["mean_olr"] = _to_1d_numpy(ds[mean_olr_name]) if mean_olr_name else np.full(n_sample, np.nan)

    if n_profiles_name is not None:
        data["n_profiles"] = _to_1d_numpy(ds[n_profiles_name])
    else:
        data["n_profiles"] = np.ones(n_sample)
        warnings.warn("n_profiles not found; raw-profile totals will equal n_samples.")

    if region_name is not None:
        data["region"] = _decode_values(_to_1d_numpy(ds[region_name]))
    else:
        data["region"] = [np.nan] * n_sample

    # Block-bootstrap coordinates (optional; only needed for method="block").
    lat_name = _optional(ds, "lat_bin_center", cfg)
    lon_name = _optional(ds, "lon_bin_center", cfg)
    data["lat_bin_center"] = _to_1d_numpy(ds[lat_name]) if lat_name else np.full(n_sample, np.nan)
    data["lon_bin_center"] = _to_1d_numpy(ds[lon_name]) if lon_name else np.full(n_sample, np.nan)

    df = pd.DataFrame(data)
    for col in ["q_uut_mean", "q_lut_mean", "dlnq_mean", "mean_sst",
                "mean_omega", "mean_olr", "n_profiles",
                "lat_bin_center", "lon_bin_center"]:
        df[col] = _as_numeric(df[col])

    # Optional profile arrays, aligned to (sample, plev).
    plev = None
    q_prof = None
    lnq_prof = None
    plev_name = _optional(ds, "plev", cfg)
    q_prof_name = _optional(ds, "q_mean_profile", cfg)
    lnq_prof_name = _optional(ds, "lnq_mean_profile", cfg)

    def _read_profile(var_name: str) -> np.ndarray:
        da = ds[var_name]
        plev_dim = [d for d in da.dims if d != sample_dim]
        if len(plev_dim) != 1:
            raise ValueError(
                f"Profile variable {var_name!r} must be 2D over (sample, plev); "
                f"got dims {da.dims}."
            )
        return da.transpose(sample_dim, plev_dim[0]).values.astype(float)

    if plev_name is not None and q_prof_name is not None:
        plev = np.asarray(ds[plev_name].values, dtype=float).squeeze()
        q_prof = _read_profile(q_prof_name)
        if lnq_prof_name is not None:
            lnq_prof = _read_profile(lnq_prof_name)
    else:
        warnings.warn(
            "q_mean_profile and/or plev not found; profile envelopes will be "
            "SKIPPED and only the dlnq bootstrap will run."
        )

    # Validity mask, applied identically to df and the profile arrays.
    valid = (
        np.isfinite(df["dlnq_mean"])
        & np.isfinite(df["q_uut_mean"])
        & np.isfinite(df["q_lut_mean"])
        & (df["q_uut_mean"] > 0)
        & (df["q_lut_mean"] > 0)
        & df["sst_bin"].notna()
    ).to_numpy()

    if cfg.min_profiles_per_sample > 0:
        valid = valid & (df["n_profiles"].to_numpy() >= cfg.min_profiles_per_sample)

    before = len(df)
    df = df.loc[valid].reset_index(drop=True)
    if q_prof is not None:
        q_prof = q_prof[valid]
    if lnq_prof is not None:
        lnq_prof = lnq_prof[valid]

    # Sort/label helpers and positional index into the profile arrays.
    df["sst_bin"] = df["sst_bin"].astype(str)
    df["omega_bin"] = df["omega_bin"].astype(str)
    df["region"] = df["region"].astype(str)
    df["sst_sort_lo"] = df["sst_bin"].map(lambda x: _sst_sort_key(x)[0])
    df["sst_sort_hi"] = df["sst_bin"].map(lambda x: _sst_sort_key(x)[1])
    df["omega_sort"] = df["omega_bin"].map(lambda x: _omega_sort_key(x, cfg)[0])
    df["_pos"] = np.arange(len(df))

    # Build the block label and verify the block method is actually feasible. If the required coordinate is missing, fall back to i.i.d. so the run never silently produces meaningless (collapsed) confidence intervals.
    df["_block"] = np.nan
    if cfg.bootstrap_method == "block":
        if cfg.block_by == "gridcell":
            feasible = df["lat_bin_center"].notna().any() and df["lon_bin_center"].notna().any()
            if feasible:
                df["_block"] = (df["lat_bin_center"].astype(str) + "|"
                                + df["lon_bin_center"].astype(str))
        elif cfg.block_by == "spatial":
            feasible = df["lat_bin_center"].notna().any() and df["lon_bin_center"].notna().any()
            if feasible:
                deg = float(cfg.spatial_block_deg)
                lat_tile = np.floor(df["lat_bin_center"].to_numpy(float) / deg) * deg
                lon_tile = np.floor(df["lon_bin_center"].to_numpy(float) / deg) * deg
                df["_block"] = (pd.Series(lat_tile, index=df.index).astype(str) + "|"
                                + pd.Series(lon_tile, index=df.index).astype(str))
        else:
            feasible = False
            warnings.warn(f"Unknown block_by={cfg.block_by!r}; falling back to i.i.d.")

        if not feasible:
            warnings.warn(
                f"block bootstrap requested with block_by={cfg.block_by!r}, but the "
                f"required coordinate was not found in the aggregated file. "
                f"Falling back to bootstrap_method='iid'."
            )
            cfg.bootstrap_method = "iid"
        else:
            n_blocks = df["_block"].nunique()
            tag = (f"spatial {cfg.spatial_block_deg:g}deg"
                   if cfg.block_by == "spatial" else cfg.block_by)
            print(f"  block bootstrap: {n_blocks} {tag} blocks over {len(df):,} aggregates",
                  flush=True)

    return AggregateData(df=df, plev=plev, q_profiles=q_prof, lnq_profiles=lnq_prof)


# Bootstrap primitives

# A "resample" is either:
#   - a 2D ndarray (n_reps, n) of local indices            -> i.i.d. bootstrap
#   - a list of n_reps 1D ndarrays of local indices        -> block bootstrap
#       (variable length per replicate, because blocks differ in size)
Resample = "np.ndarray | list[np.ndarray]"


# I.i.d.
def _boot_indices(n: int, n_reps: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_reps, n))


# Block resample: draw whole blocks with replacement.
def _block_resample_indices(
    block_labels: np.ndarray, n_reps: int, rng: np.random.Generator
) -> list[np.ndarray]:
    unique_blocks = pd.unique(block_labels)
    members = [np.where(block_labels == b)[0] for b in unique_blocks]
    n_blocks = len(members)
    out: list[np.ndarray] = []
    for _ in range(n_reps):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        out.append(np.concatenate([members[c] for c in chosen]))
    return out


# Build the per-group resample (i.i.d.
def make_resample(group: pd.DataFrame, cfg: IntervalConfig, rng: np.random.Generator):
    n = len(group)
    if cfg.bootstrap_method == "block":
        return _block_resample_indices(group["_block"].to_numpy(), cfg.n_reps, rng)
    return _boot_indices(n, cfg.n_reps, rng)


# Bootstrap a scalar group statistic.
def boot_scalar(values: np.ndarray, resample, kind: str) -> np.ndarray:
    if isinstance(resample, np.ndarray):
        samp = values[resample]  # (n_reps, n)
        return np.nanmean(samp, axis=1) if kind == "mean" else np.nanmedian(samp, axis=1)
    # block: variable-length resamples
    reducer = np.nanmean if kind == "mean" else np.nanmedian
    out = np.full(len(resample), np.nan)
    for r, idx in enumerate(resample):
        out[r] = reducer(values[idx])
    return out


# Bootstrap an across-sample profile statistic.
def boot_profile(matrix: np.ndarray, resample, kind: str) -> np.ndarray:
    reducer = np.nanmean if kind == "mean" else np.nanmedian
    n_plev = matrix.shape[1]
    n_reps = resample.shape[0] if isinstance(resample, np.ndarray) else len(resample)
    out = np.full((n_reps, n_plev), np.nan)
    for r in range(n_reps):
        idx = resample[r]
        out[r] = reducer(matrix[idx], axis=0)
    return out


# Return (ci_lo, ci_hi, boot_mean, boot_se) for a 1D bootstrap array.
def _ci(boot: np.ndarray, cfg: IntervalConfig) -> tuple[float, float, float, float]:
    finite = boot[np.isfinite(boot)]
    if finite.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    lo = float(np.percentile(finite, cfg.ci_low))
    hi = float(np.percentile(finite, cfg.ci_high))
    return (lo, hi, float(finite.mean()),
            float(finite.std(ddof=1)) if finite.size > 1 else np.nan)


# Per-group bootstrap cache

GroupKey = tuple  # (group_type, region, sst_bin, omega_bin)


# Bootstrap every statistic for one group and cache the replicate arrays.
def compute_group_cache(
    data: AggregateData,
    group: pd.DataFrame,
    group_type: str,
    cfg: IntervalConfig,
    rng: np.random.Generator,
) -> dict:
    n = len(group)
    pos = group["_pos"].to_numpy()
    # One resample shared by the metric and the profiles so dlnq and the profile envelope of a group are bootstrapped on the same draws.
    idx = make_resample(group, cfg, rng)

    dlnq = group["dlnq_mean"].to_numpy(float)
    q_uut = group["q_uut_mean"].to_numpy(float)
    q_lut = group["q_lut_mean"].to_numpy(float)

    cache: dict = {
        "group_type": group_type,
        "region": str(group["region"].iloc[0]),
        "sst_bin": str(group["sst_bin"].iloc[0]),
        "omega_bin": str(group["omega_bin"].iloc[0]),
        "n_samples": int(n),
        "n_blocks": int(group["_block"].nunique()) if cfg.bootstrap_method == "block" else int(n),
        "n_raw_profiles_total": float(group["n_profiles"].sum(skipna=True)),
        "sst_sort_lo": float(group["sst_sort_lo"].iloc[0]),
        "sst_sort_hi": float(group["sst_sort_hi"].iloc[0]),
        "omega_sort": int(group["omega_sort"].iloc[0]),
        # point estimates
        "dlnq_mean_point": float(np.nanmean(dlnq)),
        "dlnq_median_point": float(np.nanmedian(dlnq)),
        "q_uut_point": float(np.nanmean(q_uut)),
        "q_lut_point": float(np.nanmean(q_lut)),
        "mean_olr_point": float(np.nanmean(group["mean_olr"].to_numpy(float))),
        # bootstrap replicate arrays
        "dlnq_mean_boot": boot_scalar(dlnq, idx, "mean"),
        "dlnq_median_boot": boot_scalar(dlnq, idx, "median"),
    }

    if data.q_profiles is not None:
        mat = data.q_profiles[pos]
        cache["q_groupmean_point"] = np.nanmean(mat, axis=0)
        cache["q_groupmedian_point"] = np.nanmedian(mat, axis=0)
        cache["q_groupmean_boot"] = boot_profile(mat, idx, "mean")
        cache["q_groupmedian_boot"] = boot_profile(mat, idx, "median")
    if data.lnq_profiles is not None:
        matl = data.lnq_profiles[pos]
        cache["lnq_groupmean_point"] = np.nanmean(matl, axis=0)
        cache["lnq_groupmean_boot"] = boot_profile(matl, idx, "mean")

    return cache


# Bootstrap all groups (SST, SST x omega, optional regional).
def build_caches(data: AggregateData, cfg: IntervalConfig, rng: np.random.Generator) -> dict[GroupKey, dict]:
    caches: dict[GroupKey, dict] = {}

    def _store(g: pd.DataFrame, group_type: str) -> None:
        c = compute_group_cache(data, g, group_type, cfg, rng)
        key = (group_type, c["region"], c["sst_bin"], c["omega_bin"])
        caches[key] = c

    df = data.df

    for _, g in df.groupby(["sst_bin"], dropna=False, sort=False):
        _store(g, "sst_only")

    for _, g in df.groupby(["sst_bin", "omega_bin"], dropna=False, sort=False):
        _store(g, "sst_omega")

    has_region = df["region"].notna().any() and not (df["region"] == "nan").all()
    if has_region:
        for _, g in df.groupby(["region", "sst_bin"], dropna=False, sort=False):
            _store(g, "region_sst")
        for _, g in df.groupby(["region", "sst_bin", "omega_bin"], dropna=False, sort=False):
            _store(g, "region_sst_omega")

    return caches


# Assemble the steepening CSV (group rows + difference rows)

def build_group_rows(caches: dict[GroupKey, dict], cfg: IntervalConfig) -> list[dict]:
    rows = []
    for c in caches.values():
        lo_m, hi_m, bm_m, se_m = _ci(c["dlnq_mean_boot"], cfg)
        lo_md, hi_md, _, se_md = _ci(c["dlnq_median_boot"], cfg)
        rows.append({
            "record_type": "group",
            "group_type": c["group_type"],
            "region": c["region"],
            "sst_bin": c["sst_bin"],
            "omega_bin": c["omega_bin"] if c["group_type"] != "sst_only" else "all",
            "n_samples": c["n_samples"],
            "n_blocks": c["n_blocks"],
            "n_raw_profiles_total": c["n_raw_profiles_total"],
            "bootstrap_method": cfg.bootstrap_method,
            "dlnq_mean_point": c["dlnq_mean_point"],
            "dlnq_mean_boot_mean": bm_m,
            "dlnq_mean_se": se_m,
            "dlnq_mean_ci_lo": lo_m,
            "dlnq_mean_ci_hi": hi_m,
            "dlnq_median_point": c["dlnq_median_point"],
            "dlnq_median_ci_lo": lo_md,
            "dlnq_median_ci_hi": hi_md,
            "q_uut_point": c["q_uut_point"],
            "q_lut_point": c["q_lut_point"],
            "mean_olr_point": c["mean_olr_point"],
            "sst_sort_lo": c["sst_sort_lo"],
            "sst_sort_hi": c["sst_sort_hi"],
            "omega_sort": c["omega_sort"],
        })
    return rows


# Context keys: which dimensions must match between warm and reference groups.
DIFF_CONTEXTS = {
    "sst_only": ("diff_sst_only", []),
    "sst_omega": ("diff_sst_omega", ["omega_bin"]),
    "region_sst": ("diff_region_sst", ["region"]),
    "region_sst_omega": ("diff_region_sst_omega", ["region", "omega_bin"]),
}


def _match_bin(cache: dict, bounds: tuple[float, float]) -> bool:
    lo, hi = bounds
    return (abs(cache["sst_sort_lo"] - lo) < 1e-6
            and abs(cache["sst_sort_hi"] - hi) < 1e-6)


# Warm-minus-reference dlnq differences with bootstrap CIs, by context.
def build_difference_rows(caches: dict[GroupKey, dict], cfg: IntervalConfig) -> list[dict]:
    rows = []
    by_type: dict[str, list[dict]] = {}
    for c in caches.values():
        by_type.setdefault(c["group_type"], []).append(c)

    for group_type, (diff_type, key_cols) in DIFF_CONTEXTS.items():
        members = by_type.get(group_type, [])
        if not members:
            continue

        # Bucket members by their context (e.g. omega regime, region).
        contexts: dict[tuple, list[dict]] = {}
        for c in members:
            ctx = tuple(c[k] for k in key_cols)
            contexts.setdefault(ctx, []).append(c)

        for ctx, group_caches in contexts.items():
            ref = next((c for c in group_caches if _match_bin(c, cfg.reference_sst_bounds)), None)
            warm = next((c for c in group_caches if _match_bin(c, cfg.warm_sst_bounds)), None)
            if ref is None or warm is None:
                continue

            # Independent resampling already lives in each cache's replicate arrays; subtract them element-wise for the difference distribution.
            diff_mean = warm["dlnq_mean_boot"] - ref["dlnq_mean_boot"]
            diff_median = warm["dlnq_median_boot"] - ref["dlnq_median_boot"]
            lo_m, hi_m, _, se_m = _ci(diff_mean, cfg)
            lo_md, hi_md, _, se_md = _ci(diff_median, cfg)
            point_mean = warm["dlnq_mean_point"] - ref["dlnq_mean_point"]
            point_median = warm["dlnq_median_point"] - ref["dlnq_median_point"]
            robust = bool(np.isfinite(lo_m) and np.isfinite(hi_m) and (lo_m > 0 or hi_m < 0))

            rows.append({
                "record_type": "difference",
                "group_type": diff_type,
                "region": ctx[key_cols.index("region")] if "region" in key_cols else "all",
                "omega_bin": ctx[key_cols.index("omega_bin")] if "omega_bin" in key_cols else "all",
                "warm_sst_bin": warm["sst_bin"],
                "ref_sst_bin": ref["sst_bin"],
                "n_warm": warm["n_samples"],
                "n_ref": ref["n_samples"],
                "n_warm_blocks": warm["n_blocks"],
                "n_ref_blocks": ref["n_blocks"],
                "bootstrap_method": cfg.bootstrap_method,
                "ddlnq_mean_point": point_mean,
                "ddlnq_mean_ci_lo": lo_m,
                "ddlnq_mean_ci_hi": hi_m,
                "ddlnq_mean_se": se_m,
                "ddlnq_median_point": point_median,
                "ddlnq_median_ci_lo": lo_md,
                "ddlnq_median_ci_hi": hi_md,
                "robust_95_mean": robust,
                "dq_uut_point": warm["q_uut_point"] - ref["q_uut_point"],
                "dq_lut_point": warm["q_lut_point"] - ref["q_lut_point"],
                "q_uut_ref_point": ref["q_uut_point"],
                "q_uut_warm_point": warm["q_uut_point"],
                "q_lut_ref_point": ref["q_lut_point"],
                "q_lut_warm_point": warm["q_lut_point"],
                "pct_dq_uut": (100.0 * (warm["q_uut_point"] - ref["q_uut_point"]) / ref["q_uut_point"]
                               if ref["q_uut_point"] else np.nan),
                "pct_dq_lut": (100.0 * (warm["q_lut_point"] - ref["q_lut_point"]) / ref["q_lut_point"]
                               if ref["q_lut_point"] else np.nan),
                "dmean_olr_point": warm["mean_olr_point"] - ref["mean_olr_point"],
            })
    return rows


def write_steepening_csv(group_rows: list[dict], diff_rows: list[dict], cfg: IntervalConfig) -> pd.DataFrame:
    table = pd.DataFrame(group_rows + diff_rows)
    sort_cols = [c for c in ["record_type", "group_type", "region",
                             "sst_sort_lo", "sst_sort_hi", "omega_sort"]
                 if c in table.columns]
    table = table.sort_values(by=sort_cols, na_position="last").reset_index(drop=True)
    path = cfg.output_dir / "bootstrap_steepening.csv"
    table.to_csv(path, index=False)
    return table


# Profile envelope NetCDF

def write_profiles_nc(data: AggregateData, caches: dict[GroupKey, dict], cfg: IntervalConfig) -> None:
    if data.q_profiles is None or data.plev is None:
        return

    plev = data.plev
    n_plev = len(plev)
    groups = list(caches.values())
    G = len(groups)

    def _stack_point(name: str) -> np.ndarray:
        return np.vstack([g[name] for g in groups])

    def _stack_ci(boot_name: str, which: str) -> np.ndarray:
        out = np.full((G, n_plev), np.nan)
        for i, g in enumerate(groups):
            boot = g[boot_name]  # (n_reps, n_plev)
            pct = cfg.ci_low if which == "lo" else cfg.ci_high
            out[i] = np.nanpercentile(boot, pct, axis=0)
        return out

    def _bytes(values: list[str]) -> np.ndarray:
        clean = [("" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)) for v in values]
        max_len = max(1, max(len(x) for x in clean))
        return np.asarray(clean, dtype=f"S{max_len}")

    data_vars = {
        "group_type": (("benchmark_group",), _bytes([g["group_type"] for g in groups])),
        "region": (("benchmark_group",), _bytes([g["region"] for g in groups])),
        "sst_bin": (("benchmark_group",), _bytes([g["sst_bin"] for g in groups])),
        "omega_bin": (("benchmark_group",), _bytes([g["omega_bin"] for g in groups])),
        "n_samples": (("benchmark_group",), np.asarray([g["n_samples"] for g in groups], dtype=np.int32)),
        "q_groupmean_est": (("benchmark_group", "plev"), _stack_point("q_groupmean_point")),
        "q_groupmean_lo": (("benchmark_group", "plev"), _stack_ci("q_groupmean_boot", "lo")),
        "q_groupmean_hi": (("benchmark_group", "plev"), _stack_ci("q_groupmean_boot", "hi")),
        "q_groupmedian_est": (("benchmark_group", "plev"), _stack_point("q_groupmedian_point")),
        "q_groupmedian_lo": (("benchmark_group", "plev"), _stack_ci("q_groupmedian_boot", "lo")),
        "q_groupmedian_hi": (("benchmark_group", "plev"), _stack_ci("q_groupmedian_boot", "hi")),
    }
    if data.lnq_profiles is not None:
        data_vars.update({
            "lnq_groupmean_est": (("benchmark_group", "plev"), _stack_point("lnq_groupmean_point")),
            "lnq_groupmean_lo": (("benchmark_group", "plev"), _stack_ci("lnq_groupmean_boot", "lo")),
            "lnq_groupmean_hi": (("benchmark_group", "plev"), _stack_ci("lnq_groupmean_boot", "hi")),
        })

    # Warm-minus-reference profile differences, one per difference context.
    diff_records = []
    by_type: dict[str, list[dict]] = {}
    for c in groups:
        by_type.setdefault(c["group_type"], []).append(c)
    for group_type, (diff_type, key_cols) in DIFF_CONTEXTS.items():
        members = by_type.get(group_type, [])
        contexts: dict[tuple, list[dict]] = {}
        for c in members:
            contexts.setdefault(tuple(c[k] for k in key_cols), []).append(c)
        for ctx, gc in contexts.items():
            ref = next((c for c in gc if _match_bin(c, cfg.reference_sst_bounds)), None)
            warm = next((c for c in gc if _match_bin(c, cfg.warm_sst_bounds)), None)
            if ref is None or warm is None:
                continue
            diff_boot = warm["q_groupmean_boot"] - ref["q_groupmean_boot"]
            diff_records.append({
                "diff_type": diff_type,
                "region": ctx[key_cols.index("region")] if "region" in key_cols else "all",
                "omega_bin": ctx[key_cols.index("omega_bin")] if "omega_bin" in key_cols else "all",
                "warm_sst_bin": warm["sst_bin"],
                "ref_sst_bin": ref["sst_bin"],
                "est": warm["q_groupmean_point"] - ref["q_groupmean_point"],
                "lo": np.nanpercentile(diff_boot, cfg.ci_low, axis=0),
                "hi": np.nanpercentile(diff_boot, cfg.ci_high, axis=0),
            })

    if diff_records:
        data_vars.update({
            "diff_type": (("diff_context",), _bytes([d["diff_type"] for d in diff_records])),
            "diff_region": (("diff_context",), _bytes([d["region"] for d in diff_records])),
            "diff_omega_bin": (("diff_context",), _bytes([d["omega_bin"] for d in diff_records])),
            "diff_warm_sst_bin": (("diff_context",), _bytes([d["warm_sst_bin"] for d in diff_records])),
            "diff_ref_sst_bin": (("diff_context",), _bytes([d["ref_sst_bin"] for d in diff_records])),
            "dq_groupmean_est": (("diff_context", "plev"), np.vstack([d["est"] for d in diff_records])),
            "dq_groupmean_lo": (("diff_context", "plev"), np.vstack([d["lo"] for d in diff_records])),
            "dq_groupmean_hi": (("diff_context", "plev"), np.vstack([d["hi"] for d in diff_records])),
        })

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "benchmark_group": np.arange(G, dtype=np.int32),
            "plev": plev.astype(np.float64),
        },
        attrs={
            "description": "Bootstrap benchmark-profile envelopes and warm-minus-reference differences",
            "ci_low_pct": cfg.ci_low,
            "ci_high_pct": cfg.ci_high,
            "n_replicates": cfg.n_reps,
            "seed": cfg.seed,
            "bootstrap_method": cfg.bootstrap_method,
            "block_by": cfg.block_by if cfg.bootstrap_method == "block" else "n/a",
            "bootstrap_unit": ("block of samples (" + cfg.block_by + ")"
                               if cfg.bootstrap_method == "block"
                               else "aggregated sample (cell-regime)"),
            "reference_sst_bounds": str(cfg.reference_sst_bounds),
            "warm_sst_bounds": str(cfg.warm_sst_bounds),
        },
    )
    path = cfg.output_dir / "bootstrap_profiles.nc"
    try:
        ds_out.to_netcdf(path)
    except Exception as exc:
        warnings.warn(f"Could not write {path}: {exc}")


# Figures

def _sst_only_sorted(table: pd.DataFrame) -> pd.DataFrame:
    s = table[(table["record_type"] == "group") & (table["group_type"] == "sst_only")].copy()
    return s.sort_values(["sst_sort_lo", "sst_sort_hi"])


def plot_dlnq_ci_by_sst(table: pd.DataFrame, cfg: IntervalConfig) -> None:
    s = _sst_only_sorted(table)
    if s.empty:
        return
    x = np.arange(len(s))
    y = s["dlnq_mean_point"].to_numpy(float)
    lo = y - s["dlnq_mean_ci_lo"].to_numpy(float)
    hi = s["dlnq_mean_ci_hi"].to_numpy(float) - y

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=4, linestyle="-")
    ax.axhline(0, linewidth=1, color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels(s["sst_bin"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Mean Δln(q)  (95% bootstrap CI)")
    ax.set_xlabel("SST bin")
    ax.set_title("dlnq by SST bin with bootstrap CI")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = cfg.figures_dir / "fig_dlnq_ci_by_sst.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_dlnq_ci_by_sst_omega(table: pd.DataFrame, cfg: IntervalConfig) -> None:
    s = table[(table["record_type"] == "group") & (table["group_type"] == "sst_omega")].copy()
    if s.empty:
        return
    sst_labels = sorted(s["sst_bin"].unique(), key=_sst_sort_key)
    x_map = {lab: i for i, lab in enumerate(sst_labels)}
    omega_labels = sorted(s["omega_bin"].unique(), key=lambda x: _omega_sort_key(x, cfg))

    fig, ax = plt.subplots(figsize=(10, 6))
    n_omega = len(omega_labels)
    for k, omega in enumerate(omega_labels):
        g = s[s["omega_bin"] == omega].sort_values(["sst_sort_lo", "sst_sort_hi"])
        # small horizontal offset so overlapping CIs stay readable
        offset = (k - (n_omega - 1) / 2) * 0.12
        x = np.array([x_map[v] for v in g["sst_bin"]], dtype=float) + offset
        y = g["dlnq_mean_point"].to_numpy(float)
        lo = y - g["dlnq_mean_ci_lo"].to_numpy(float)
        hi = g["dlnq_mean_ci_hi"].to_numpy(float) - y
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=3, label=str(omega))

    ax.axhline(0, linewidth=1, color="0.4")
    ax.set_xticks(np.arange(len(sst_labels)))
    ax.set_xticklabels([str(v) for v in sst_labels], rotation=30, ha="right")
    ax.set_ylabel("Mean Δln(q)  (95% bootstrap CI)")
    ax.set_xlabel("SST bin")
    ax.set_title("dlnq by SST and omega regime")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Omega regime", fontsize=8)
    fig.tight_layout()
    path = cfg.figures_dir / "fig_dlnq_ci_by_sst_omega.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_dlnq_difference(table: pd.DataFrame, cfg: IntervalConfig) -> None:
    d = table[table["record_type"] == "difference"].copy()
    d = d[d["group_type"].isin(["diff_sst_only", "diff_sst_omega"])]
    if d.empty:
        return
    # Label each difference by its context.
    def _label(r):
        if r["group_type"] == "diff_sst_only":
            return "pooled (all omega)"
        return str(r["omega_bin"])
    d["ctx_label"] = d.apply(_label, axis=1)
    d = d.sort_values(["group_type", "omega_bin"])

    x = np.arange(len(d))
    y = d["ddlnq_mean_point"].to_numpy(float)
    lo = y - d["ddlnq_mean_ci_lo"].to_numpy(float)
    hi = d["ddlnq_mean_ci_hi"].to_numpy(float) - y
    robust = d["robust_95_mean"].to_numpy(bool)
    colors = ["tab:red" if r else "0.5" for r in robust]

    fig, ax = plt.subplots(figsize=(9, 5))
    for xi, yi, loi, hii, ci in zip(x, y, lo, hi, colors):
        ax.errorbar([xi], [yi], yerr=[[loi], [hii]], marker="o", capsize=4, color=ci)
    ax.axhline(0, linewidth=1, color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels(d["ctx_label"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Δln(q): warm − reference  (95% bootstrap CI)")
    ax.set_title("Warm-minus-reference dlnq difference\n"
                 "red = CI excludes 0 (robust), grey = not robust")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = cfg.figures_dir / "fig_dlnq_difference.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _open_profiles(cfg: IntervalConfig) -> Optional[xr.Dataset]:
    path = cfg.output_dir / "bootstrap_profiles.nc"
    if not path.exists():
        return None
    return xr.open_dataset(path)


def plot_profile_envelope(cfg: IntervalConfig) -> None:
    ds = _open_profiles(cfg)
    if ds is None:
        return
    plev = ds["plev"].values
    gtype = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in ds["group_type"].values])
    sst = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in ds["sst_bin"].values])

    sel = np.where(gtype == "sst_only")[0]
    ref_lab = _find_sst_label(sst[sel], cfg.reference_sst_bounds)
    warm_lab = _find_sst_label(sst[sel], cfg.warm_sst_bounds)
    targets = [(ref_lab, "reference"), (warm_lab, "warm")]

    fig, ax = plt.subplots(figsize=(6.5, 7))
    plotted = False
    for lab, role in targets:
        if lab is None:
            continue
        gi = sel[np.where(sst[sel] == lab)[0]]
        if len(gi) == 0:
            continue
        i = int(gi[0])
        est = ds["q_groupmedian_est"].values[i]
        lo = ds["q_groupmedian_lo"].values[i]
        hi = ds["q_groupmedian_hi"].values[i]
        line, = ax.plot(est, plev, marker="o", label=f"{lab} ({role})")
        ax.fill_betweenx(plev, lo, hi, alpha=0.2, color=line.get_color())
        plotted = True

    ds.close()
    if not plotted:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel("q  (benchmark median across samples)")
    ax.set_title("Benchmark median UTWV profile with 95% bootstrap envelope")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = cfg.figures_dir / "fig_profile_envelope.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_profile_difference(cfg: IntervalConfig) -> None:
    ds = _open_profiles(cfg)
    if ds is None or "dq_groupmean_est" not in ds:
        if ds is not None:
            ds.close()
        return
    plev = ds["plev"].values
    dtype = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in ds["diff_type"].values])
    sel = np.where(dtype == "diff_sst_only")[0]
    if len(sel) == 0:
        ds.close()
        return
    i = int(sel[0])
    est = ds["dq_groupmean_est"].values[i]
    lo = ds["dq_groupmean_lo"].values[i]
    hi = ds["dq_groupmean_hi"].values[i]
    ds.close()

    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.plot(est, plev, marker="o", color="tab:red", label="warm − reference")
    ax.fill_betweenx(plev, lo, hi, alpha=0.2, color="tab:red")
    ax.axvline(0, linewidth=1, color="0.4")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel("Δq  (warm − reference benchmark mean)")
    ax.set_title("Warm-minus-reference profile difference\nwith 95% bootstrap band")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = cfg.figures_dir / "fig_profile_difference.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_figures(table: pd.DataFrame, cfg: IntervalConfig) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_dlnq_ci_by_sst(table, cfg)
    plot_dlnq_ci_by_sst_omega(table, cfg)
    plot_dlnq_difference(table, cfg)
    plot_profile_envelope(cfg)
    plot_profile_difference(cfg)


# Conclusions report

def write_conclusions(table: pd.DataFrame, cfg: IntervalConfig) -> None:
    lines: list[str] = []
    lines.append("BOOTSTRAP UNCERTAINTY REPORT")
    lines.append("============================")
    lines.append("")
    lines.append(f"Replicates: {cfg.n_reps}   Seed: {cfg.seed}   "
                 f"CI: {cfg.ci_low:g}-{cfg.ci_high:g} percentile")
    if cfg.bootstrap_method == "block":
        lines.append(f"Resampling: BLOCK bootstrap by '{cfg.block_by}' "
                     f"(whole blocks of correlated samples resampled together).")
    else:
        lines.append("Resampling: i.i.d. bootstrap of cell-regime aggregates (NOT raw profiles).")
    lines.append(f"Reference SST bin bounds: {cfg.reference_sst_bounds}")
    lines.append(f"Warm SST bin bounds:      {cfg.warm_sst_bounds}")
    lines.append("")
    lines.append("Metric: dlnq = ln(q_UUT / q_LUT). Less negative = flatter profile.")
    lines.append("A warm-minus-reference difference is 'robust' when its 95% CI excludes 0.")
    lines.append("")

    diffs = table[table["record_type"] == "difference"].copy()
    if diffs.empty:
        lines.append("No warm-vs-reference differences could be formed. Check that both")
        lines.append("SST bins exist in the aggregated file and match the configured bounds.")
    else:
        # Headline: pooled SST-only difference.
        pooled = diffs[diffs["group_type"] == "diff_sst_only"]
        lines.append("1. Headline pooled comparison (all omega)")
        lines.append("-----------------------------------------")
        if pooled.empty:
            lines.append("  Not available (missing reference or warm SST-only bin).")
        else:
            r = pooled.iloc[0]
            verdict = "ROBUST (CI excludes 0)" if r["robust_95_mean"] else "not robust (CI includes 0)"
            lines.append(f"  {r['warm_sst_bin']} minus {r['ref_sst_bin']}:")
            lines.append(f"    Δ mean dlnq = {_format_float(r['ddlnq_mean_point'])}  "
                         f"95% CI [{_format_float(r['ddlnq_mean_ci_lo'])}, "
                         f"{_format_float(r['ddlnq_mean_ci_hi'])}]  -> {verdict}")
            lines.append(f"    Δ q_UUT = {_format_float(r['dq_uut_point'], 8)} "
                         f"({_format_float(r.get('pct_dq_uut'), 1)}%)   "
                         f"Δ q_LUT = {_format_float(r['dq_lut_point'], 8)} "
                         f"({_format_float(r.get('pct_dq_lut'), 1)}%)")
            lines.append(f"    Δ mean OLR = {_format_float(r['dmean_olr_point'], 2)} W/m^2")
            lines.append(f"    (q_UUT: {_format_float(r.get('q_uut_ref_point'), 8)} -> "
                         f"{_format_float(r.get('q_uut_warm_point'), 8)};  "
                         f"q_LUT: {_format_float(r.get('q_lut_ref_point'), 8)} -> "
                         f"{_format_float(r.get('q_lut_warm_point'), 8)})")
            lines.append("    Interpretation: " + (
                "both layers moisten but q_LUT rises faster -> profile STEEPENS"
                if (pd.notna(r.get('pct_dq_uut')) and pd.notna(r.get('pct_dq_lut'))
                    and r.get('pct_dq_lut') > r.get('pct_dq_uut'))
                else "q_UUT rises faster than q_LUT -> profile FLATTENS"
                if (pd.notna(r.get('pct_dq_uut')) and pd.notna(r.get('pct_dq_lut')))
                else "n/a"))
            lines.append(f"    n_warm = {int(r['n_warm'])}, n_ref = {int(r['n_ref'])}")
            for nm, val in [("warm", r["n_warm"]), ("reference", r["n_ref"])]:
                if val < cfg.min_samples_warning:
                    lines.append(f"    WARNING: {nm} n_samples={int(val)} below "
                                 f"{cfg.min_samples_warning}; treat CI cautiously.")
        lines.append("")

        # Regime-resolved differences.
        regime = diffs[diffs["group_type"] == "diff_sst_omega"]
        lines.append("2. Comparison within each omega regime")
        lines.append("--------------------------------------")
        if regime.empty:
            lines.append("  Not available.")
        else:
            for _, r in regime.sort_values("omega_bin").iterrows():
                verdict = "robust" if r["robust_95_mean"] else "not robust"
                lines.append(f"  {r['omega_bin']}: {r['warm_sst_bin']} − {r['ref_sst_bin']}  "
                             f"Δ mean dlnq = {_format_float(r['ddlnq_mean_point'])}  "
                             f"[{_format_float(r['ddlnq_mean_ci_lo'])}, "
                             f"{_format_float(r['ddlnq_mean_ci_hi'])}] -> {verdict}  "
                             f"(n_warm={int(r['n_warm'])})")
        lines.append("")

        # Regional, if present.
        reg = diffs[diffs["group_type"].isin(["diff_region_sst", "diff_region_sst_omega"])]
        if not reg.empty:
            lines.append("3. Regional comparisons")
            lines.append("-----------------------")
            for _, r in reg.sort_values(["region", "omega_bin"]).iterrows():
                verdict = "robust" if r["robust_95_mean"] else "not robust"
                lines.append(f"  region={r['region']}, omega={r['omega_bin']}: "
                             f"Δ mean dlnq = {_format_float(r['ddlnq_mean_point'])} "
                             f"[{_format_float(r['ddlnq_mean_ci_lo'])}, "
                             f"{_format_float(r['ddlnq_mean_ci_hi'])}] -> {verdict}")
            lines.append("")

    lines.append("Caveat on independence")
    lines.append("----------------------")
    if cfg.bootstrap_method == "block":
        lines.append(f"  These CIs come from a BLOCK bootstrap by '{cfg.block_by}', which keeps")
        lines.append(f"  correlated samples together and is more honest than i.i.d. when that")
        lines.append(f"  axis carries autocorrelation. Note it controls correlation along ONE")
        lines.append(f"  axis only; the other axis (and any block with few members) can still")
        lines.append(f"  leave residual dependence. Compare against an 'iid' run to see how much")
        lines.append(f"  the {cfg.block_by} structure widens the intervals.")
    else:
        lines.append("  This is an i.i.d. resample of cell-regime aggregates. Neighboring gridcells")
        lines.append("  are spatially correlated, so these CIs are likely a lower bound on the")
        lines.append("  true uncertainty. Re-run with BOOTSTRAP_METHOD='block' and")
        lines.append("  BLOCK_BY='spatial' to check robustness.")

    report = "\n".join(lines)
    path = cfg.output_dir / "conclusions.txt"
    path.write_text(report, encoding="utf-8")


# Main entry point

def run_intervals(cfg: IntervalConfig) -> pd.DataFrame:
    if not cfg.input_path.exists():
        raise FileNotFoundError(
            "Could not find the aggregated NetCDF file.\n"
            f"Current input_path is:\n  {cfg.input_path}\n\n"
            "Fix AGGREGATE_INPUT_PATH in the EASY PYCHARM SETTINGS section."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    data = load_aggregates(cfg)
    caches = build_caches(data, cfg, rng)

    group_rows = build_group_rows(caches, cfg)
    diff_rows = build_difference_rows(caches, cfg)
    table = write_steepening_csv(group_rows, diff_rows, cfg)

    write_profiles_nc(data, caches, cfg)
    make_figures(table, cfg)
    write_conclusions(table, cfg)

    return table


# i.i.d. vs block comparison

# Readable label for a difference row, used on comparison plots.
def _diff_context_label(row: pd.Series) -> str:
    gt = row["group_type"]
    if gt == "diff_sst_only":
        return "pooled (all omega)"
    if gt == "diff_sst_omega":
        return str(row["omega_bin"])
    if gt == "diff_region_sst":
        return f"{row['region']}"
    if gt == "diff_region_sst_omega":
        return f"{row['region']} / {row['omega_bin']}"
    return gt


# Display names and colors for each method, keyed by the short method label.
METHOD_DISPLAY = {
    "iid": "i.i.d.",
    "spatial10": "spatial 10°",
    "spatial20": "spatial 20°",
    "gridcell": "block (gridcell)",
}
METHOD_COLOR = {
    "iid": "0.6",
    "spatial10": "tab:blue",
    "spatial20": "tab:green",
    "gridcell": "tab:green",
}


# Merge any number of method tables and compute CI-width comparisons.
def build_method_comparison(
    method_tables: dict[str, pd.DataFrame], cfg: IntervalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    methods = list(method_tables.keys())
    baseline = methods[0]

    # --- group dlnq CIs ---
    g_keys = ["group_type", "region", "sst_bin", "omega_bin"]
    group_cmp: Optional[pd.DataFrame] = None
    for label, t in method_tables.items():
        g = t[t["record_type"] == "group"].copy()
        if g.empty:
            continue
        g[f"ci_width_{label}"] = g["dlnq_mean_ci_hi"] - g["dlnq_mean_ci_lo"]
        if label == baseline:
            take = g[g_keys + ["n_samples", "dlnq_mean_point",
                               "sst_sort_lo", "sst_sort_hi", "omega_sort",
                               f"ci_width_{label}"]].copy()
        else:
            g = g.rename(columns={"n_blocks": f"n_blocks_{label}"})
            take = g[g_keys + [f"n_blocks_{label}", f"ci_width_{label}"]].copy()
        group_cmp = take if group_cmp is None else group_cmp.merge(take, on=g_keys, how="outer")
    if group_cmp is None:
        group_cmp = pd.DataFrame()
    else:
        group_cmp["record_type"] = "group"
        for label in methods:
            wcol = f"ci_width_{label}"
            if label != baseline and wcol in group_cmp:
                group_cmp[f"width_ratio_{label}_over_{baseline}"] = (
                    group_cmp[wcol] / group_cmp[f"ci_width_{baseline}"]
                )

    # --- warm-minus-reference difference CIs ---
    d_keys = ["group_type", "region", "omega_bin", "warm_sst_bin", "ref_sst_bin"]
    diff_cmp: Optional[pd.DataFrame] = None
    robust_cols: list[str] = []
    for label, t in method_tables.items():
        d = t[t["record_type"] == "difference"].copy()
        if d.empty:
            continue
        d[f"ci_width_{label}"] = d["ddlnq_mean_ci_hi"] - d["ddlnq_mean_ci_lo"]
        d = d.rename(columns={"robust_95_mean": f"robust_{label}"})
        robust_cols.append(f"robust_{label}")
        if label == baseline:
            take = d[d_keys + ["n_warm", "n_ref", "ddlnq_mean_point",
                               f"ci_width_{label}", f"robust_{label}"]].copy()
        else:
            take = d[d_keys + [f"ci_width_{label}", f"robust_{label}"]].copy()
        diff_cmp = take if diff_cmp is None else diff_cmp.merge(take, on=d_keys, how="outer")

    if diff_cmp is None:
        diff_cmp = pd.DataFrame()
    else:
        diff_cmp["record_type"] = "difference"
        for label in methods:
            wcol = f"ci_width_{label}"
            if label != baseline and wcol in diff_cmp:
                diff_cmp[f"width_ratio_{label}_over_{baseline}"] = (
                    diff_cmp[wcol] / diff_cmp[f"ci_width_{baseline}"]
                )
        # Verdict flips if the robust/not-robust call disagrees across any method.
        avail = [c for c in robust_cols if c in diff_cmp]
        if avail:
            rb = diff_cmp[avail].astype("boolean").fillna(False)
            diff_cmp["robust_flip"] = rb.nunique(axis=1) > 1
        else:
            diff_cmp["robust_flip"] = False
        diff_cmp["ctx_label"] = diff_cmp.apply(_diff_context_label, axis=1)

    # --- write a combined CSV ---
    combined = pd.concat([group_cmp, diff_cmp], ignore_index=True, sort=False)
    combined.insert(0, "methods", "+".join(methods))
    path = cfg.output_dir / "method_comparison.csv"
    combined.to_csv(path, index=False)
    return group_cmp, diff_cmp, methods


# Draw one grouped-bar figure of CI widths, one bar cluster per method.
def _grouped_width_bars(df: pd.DataFrame, label_col: str, methods: list[str],
                        title: str, ylabel: str, xlabel: str,
                        path: Path, flip_col: Optional[str] = None) -> None:
    present = [m for m in methods if f"ci_width_{m}" in df.columns]
    if df.empty or not present:
        return
    labels = df[label_col].astype(str).tolist()
    n = len(labels)
    m = len(present)
    x = np.arange(n)
    total = 0.8
    w = total / m

    fig, ax = plt.subplots(figsize=(max(8, n * (0.5 + 0.25 * m)), 5.5))
    width_arrays = []
    for k, label in enumerate(present):
        vals = df[f"ci_width_{label}"].to_numpy(float)
        width_arrays.append(vals)
        off = (k - (m - 1) / 2) * w
        ax.bar(x + off, vals, w,
               label=METHOD_DISPLAY.get(label, label),
               color=METHOD_COLOR.get(label, None))

    if flip_col is not None and flip_col in df.columns:
        ymax = np.nanmax([np.nanmax(a) if a.size else 0 for a in width_arrays] + [0])
        for xi, fl in zip(x, df[flip_col].to_numpy(bool)):
            if fl:
                ax.text(xi, ymax * 1.02, "verdict\nflips", ha="center", va="bottom",
                        fontsize=7, color="tab:red")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_group_width_comparison(group_cmp: pd.DataFrame, cfg: IntervalConfig,
                                methods: list[str]) -> None:
    s = group_cmp[group_cmp["group_type"] == "sst_only"].copy() if not group_cmp.empty else group_cmp
    if s.empty:
        return
    s = s.sort_values(["sst_sort_lo", "sst_sort_hi"])
    _grouped_width_bars(
        s, "sst_bin", methods,
        title="Δln(q) CI width by SST bin — i.i.d. vs block bootstrap",
        ylabel="95% CI width of mean Δln(q)", xlabel="SST bin",
        path=cfg.figures_dir / "fig_compare_width_by_sst.png",
    )


def plot_diff_width_comparison(diff_cmp: pd.DataFrame, cfg: IntervalConfig,
                               methods: list[str]) -> None:
    if diff_cmp.empty:
        return
    d = diff_cmp[diff_cmp["group_type"].isin(["diff_sst_only", "diff_sst_omega"])].copy()
    if d.empty:
        return
    d = d.sort_values(["group_type", "omega_bin"])
    _grouped_width_bars(
        d, "ctx_label", methods,
        title="Warm−reference difference CI width — i.i.d. vs block",
        ylabel="95% CI width of Δln(q): warm − reference",
        xlabel="Comparison context",
        path=cfg.figures_dir / "fig_compare_width_difference.png",
        flip_col="robust_flip",
    )


# Run i.i.d. plus two SPATIAL block scales (10deg, 20deg) and compare CI widths.
def run_comparison(cfg: IntervalConfig) -> None:
    base_out = cfg.output_dir
    base_fig = cfg.figures_dir

    # (short label, method, block_by, spatial_deg, subfolder)
    runs = [
        ("iid", "iid", "spatial", cfg.spatial_block_deg, "iid"),
        ("spatial10", "block", "spatial", 10.0, "block_spatial_10deg"),
        ("spatial20", "block", "spatial", 20.0, "block_spatial_20deg"),
    ]

    method_tables: dict[str, pd.DataFrame] = {}
    effective: dict[str, str] = {}
    for label, method, block_by, deg, sub in runs:
        rcfg = replace(
            cfg, bootstrap_method=method, block_by=block_by, spatial_block_deg=deg,
            output_dir=base_out / sub,
            figures_dir=base_out / sub / "figures",
        )
        method_tables[label] = run_intervals(rcfg)
        effective[label] = rcfg.bootstrap_method  # may have fallen back to iid

    for label in ("spatial10", "spatial20"):
        if effective.get(label) != "block":
            warnings.warn(
                f"Block bootstrap '{label}' fell back to i.i.d. (lat/lon coordinates "
                f"missing); its column will equal the i.i.d. column. Check that "
                f"lat_bin_center / lon_bin_center exist in the aggregated file."
            )

    # Comparison artifacts go at the top level.
    cmp_cfg = replace(cfg, output_dir=base_out, figures_dir=base_fig)
    cmp_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cmp_cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    group_cmp, diff_cmp, methods = build_method_comparison(method_tables, cmp_cfg)
    plot_group_width_comparison(group_cmp, cmp_cfg, methods)
    plot_diff_width_comparison(diff_cmp, cmp_cfg, methods)

    # Console summary of the headline pooled contrast across all methods.
    if not diff_cmp.empty:
        pooled = diff_cmp[diff_cmp["group_type"] == "diff_sst_only"]
        if not pooled.empty:
            r = pooled.iloc[0]
            widths = "  ".join(
                f"{METHOD_DISPLAY.get(m, m)}={_format_float(r.get(f'ci_width_{m}'))}"
                for m in methods if f"ci_width_{m}" in pooled.columns
            )
            flip = bool(r.get("robust_flip", False))


SINGLE_CONFIG = IntervalConfig(
    benchmark_name=BENCHMARK_NAME,
    input_path=Path(AGGREGATE_INPUT_PATH),
    output_dir=Path(INTERVAL_OUTPUT_DIR),
    figures_dir=Path(INTERVAL_FIGURES_DIR),
    reference_sst_bounds=REFERENCE_SST_BOUNDS,
    warm_sst_bounds=WARM_SST_BOUNDS,
    n_reps=N_REPLICATES,
    seed=RANDOM_SEED,
    ci_low=CI_LOW,
    ci_high=CI_HIGH,
    bootstrap_method=BOOTSTRAP_METHOD,
    block_by=BLOCK_BY,
    spatial_block_deg=SPATIAL_BLOCK_DEG,
    min_profiles_per_sample=MIN_PROFILES_PER_SAMPLE,
)


def parse_args() -> IntervalConfig:
    d = IntervalConfig()
    p = argparse.ArgumentParser(description="Bootstrap uncertainty for the MLS + ERA5 UTWV pipeline.")
    p.add_argument("--input", default=str(d.input_path))
    p.add_argument("--output-dir", default=str(d.output_dir))
    p.add_argument("--figures-dir", default=str(d.figures_dir))
    p.add_argument("--reference-low", type=float, default=d.reference_sst_bounds[0])
    p.add_argument("--reference-high", type=float, default=d.reference_sst_bounds[1])
    p.add_argument("--warm-low", type=float, default=d.warm_sst_bounds[0])
    p.add_argument("--warm-high", type=float, default=d.warm_sst_bounds[1])
    p.add_argument("--n-reps", type=int, default=d.n_reps)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--bootstrap-method", choices=["iid", "block"], default=d.bootstrap_method)
    p.add_argument("--block-by", choices=["gridcell", "spatial"], default=d.block_by)
    p.add_argument("--spatial-block-deg", type=float, default=d.spatial_block_deg)
    p.add_argument("--min-profiles-per-sample", type=int, default=d.min_profiles_per_sample)
    a = p.parse_args()
    return IntervalConfig(
        benchmark_name=BENCHMARK_NAME,
        input_path=Path(a.input),
        output_dir=Path(a.output_dir),
        figures_dir=Path(a.figures_dir),
        reference_sst_bounds=(a.reference_low, a.reference_high),
        warm_sst_bounds=(a.warm_low, a.warm_high),
        n_reps=a.n_reps,
        seed=a.seed,
        bootstrap_method=a.bootstrap_method,
        block_by=a.block_by,
        spatial_block_deg=a.spatial_block_deg,
        min_profiles_per_sample=a.min_profiles_per_sample,
    )


if __name__ == "__main__":
    if RUN_COMPARISON:
        run_comparison(SINGLE_CONFIG)
    else:
        run_intervals(SINGLE_CONFIG)
