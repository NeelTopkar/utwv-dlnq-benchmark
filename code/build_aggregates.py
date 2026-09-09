"""Aggregate collocated profiles into cell-regime samples.

Assigns each profile to a 5 x 5 degree ocean cell, an SST bin and a 500 hPa
vertical-motion category, computes the two-layer index dlnq = ln(q_UUT/q_LUT)
per profile, and averages within each cell, bin and regime combination.

Writes both the per-profile file and the aggregated file that every downstream
analysis reads.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from contextlib import contextmanager
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr


# CONFIG

@dataclass
class AggregateConfig:
    # Input / output
    input_paths: Sequence[str] = field(default_factory=list)
    output_dir: str = r"data/aggregates/tropical_oceans/sst_295-305_0.5K"

    # NetCDF engine. If you installed netCDF4, keep this as "netcdf4". If this causes issues, set to None and xarray will choose.
    netcdf_engine: str | None = "netcdf4"

    # Debug mode. Use 50_000 or 100_000 for testing. Use None for full run.
    debug_max_profiles: int | None = None

    # Dimension names
    profile_dim: str = "profile"
    lev_dim: str = "plev"

    # Variable names
    q_var: str = "q"
    lnq_var: str = "lnq"

    # MLS QC confirmation variables
    valid_var_candidates: Sequence[str] = field(default_factory=lambda: ("valid",))
    precision_var_candidates: Sequence[str] = field(default_factory=lambda: ("precision",))

    # If True, the aggregation stops if q/lnq still contain finite values where valid=False.
    require_q_masked_by_valid: bool = True

    # Full QC check can be slow on millions of profiles. Keep True for final run. For speed testing, set False.
    run_valid_precision_qc_check: bool = True

    pressure_var_candidates: Sequence[str] = field(
        default_factory=lambda: ("pressure", "p", "plev")
    )
    time_var: str = "time"
    lat_var: str = "lat"
    lon_var: str = "lon"
    sst_var: str = "sst"
    omega_var_candidates: Sequence[str] = field(
        default_factory=lambda: ("w500", "omega500", "omega", "w")
    )

    # OLR support
    olr_var_candidates: Sequence[str] = field(
        default_factory=lambda: ("olr", "toa_olr", "rlut")
    )
    olr_bin_edges: Sequence[float] = field(
        default_factory=lambda: (0.0, 220.0, 240.0, 250.0, np.inf)
    )
    olr_bin_labels: Sequence[str] = field(
        default_factory=lambda: [
            "deep_convective",
            "convective_high_cloud",
            "transition",
            "clear_sky",
        ]
    )
    group_by_olr: bool = False

    # Science domain
    lat_min: float = -20.0
    lat_max: float = 20.0

    # UT pressure range in hPa
    p_min_hpa: float = 146.0
    p_max_hpa: float = 317.0

    # Layer bounds for the dlnq metric
    uut_min_hpa: float = 146.0
    uut_max_hpa: float = 216.0
    lut_min_hpa: float = 216.0
    lut_max_hpa: float = 317.0

    # Missing-data rules
    min_valid_levels_in_ut: int = 4
    min_valid_levels_in_uut: int = 2
    min_valid_levels_in_lut: int = 2

    # Grid design
    lat_bin_deg: float = 5.0
    lon_bin_deg: float = 5.0

    # SST bins
    sst_bin_edges: Sequence[float] = field(
        default_factory=lambda: [295, 295.5, 296, 296.5, 297, 297.5, 298, 298.5, 299, 299.5, 300, 300.5, 301, 301.5, 302, 302.5, 303]
    )

    # Omega bins in Pa/s
    omega_bin_edges: Sequence[float] = field(
        default_factory=lambda: [-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf]
    )
    omega_bin_labels: Sequence[str] = field(
        default_factory=lambda: [
            "strong_ascent",
            "weak_ascent",
            "neutral",
            "weak_subsidence",
            "strong_subsidence",
        ]
    )

    # Dask chunks
    dask_profile_chunks: int | None = 50_000

    # File-loading batch size
    file_batch_size: int = 100

    # Output names
    screened_profiles_name: str = "profiles.nc"
    aggregated_name: str = "aggregated_gridcells.nc"

    # Compression
    complevel: int = 4

    # Aggregation progress print frequency
    aggregation_progress_every: int = 500

    def __post_init__(self) -> None:
        n_omega_bins = len(self.omega_bin_edges) - 1
        if len(self.omega_bin_labels) != n_omega_bins:
            raise ValueError(
                f"omega_bin_labels has {len(self.omega_bin_labels)} entries "
                f"but omega_bin_edges defines {n_omega_bins} bins."
            )

        n_olr_bins = len(self.olr_bin_edges) - 1
        if len(self.olr_bin_labels) != n_olr_bins:
            raise ValueError(
                f"olr_bin_labels has {len(self.olr_bin_labels)} entries "
                f"but olr_bin_edges defines {n_olr_bins} bins."
            )


# TIMING

# Report the steps that actually cost time; a full run is long and otherwise silent.
TIMED_REPORT_SECONDS = 1.0


@contextmanager
def timed(label: str):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        if dt >= TIMED_REPORT_SECONDS:
            print(f"  [{dt:7.1f}s] {label}", flush=True)


# HELPERS

def _find_existing_var(ds: xr.Dataset, candidates: Sequence[str], kind: str) -> str:
    for name in candidates:
        if name in ds.variables or name in ds.coords:
            return name
    raise KeyError(f"Could not find {kind}. Tried: {list(candidates)}")


# Confirm that q/lnq are masked wherever valid=False. This can be slow because it scans q/lnq/valid/precision.
def _confirm_valid_precision_masking(
    ds: xr.Dataset,
    cfg: AggregateConfig,
    valid_var: str,
    precision_var: str,
) -> None:

    if not cfg.run_valid_precision_qc_check:
        return


    q = ds[cfg.q_var]
    lnq = ds[cfg.lnq_var]
    valid_raw = ds[valid_var]
    precision = ds[precision_var]

    if q.dims != valid_raw.dims:
        raise ValueError(
            f"QC shape mismatch: q dims are {q.dims}, "
            f"but {valid_var} dims are {valid_raw.dims}."
        )

    if q.dims != precision.dims:
        raise ValueError(
            f"QC shape mismatch: q dims are {q.dims}, "
            f"but {precision_var} dims are {precision.dims}."
        )

    if valid_raw.dtype == bool:
        valid_bool = valid_raw
    else:
        valid_bool = valid_raw.fillna(0) != 0

    finite_q = np.isfinite(q)
    finite_lnq = np.isfinite(lnq)
    finite_precision = np.isfinite(precision)

    def _count(label: str, x: xr.DataArray) -> int:
        with timed(f"QC count: {label}"):
            return int(x.sum().compute().item())

    n_total = int(np.prod([q.sizes[d] for d in q.dims]))

    n_valid_true = _count("valid=True gridpoints", valid_bool)
    n_finite_q = _count("finite q gridpoints", finite_q)
    n_finite_lnq = _count("finite lnq gridpoints", finite_lnq)
    n_finite_precision = _count("finite precision gridpoints", finite_precision)

    n_finite_q_where_invalid = _count("finite q where valid=False", finite_q & ~valid_bool)
    n_finite_lnq_where_invalid = _count("finite lnq where valid=False", finite_lnq & ~valid_bool)
    n_finite_q_nonfinite_precision = _count(
        "finite q but precision nonfinite", finite_q & ~finite_precision
    )

    print(f"  QC: {n_total:,} gridpoints; valid={n_valid_true:,}; finite q={n_finite_q:,}; "
          f"finite lnq={n_finite_lnq:,}; finite precision={n_finite_precision:,}")
    print(f"  QC: finite q where valid=False={n_finite_q_where_invalid:,}; "
          f"finite lnq where valid=False={n_finite_lnq_where_invalid:,}; "
          f"finite q with non-finite precision={n_finite_q_nonfinite_precision:,}")

    if n_finite_q_where_invalid > 0 or n_finite_lnq_where_invalid > 0:
        msg = (
            "\n[aggregate] ERROR: q/lnq contain finite values where valid=False.\n"
            "This means aggregation may accidentally use invalid MLS retrieval levels,\n"
            "because later profile screening counts finite q/lnq values.\n\n"
            "Fix this either by masking q/lnq during screening or collocation, or by explicitly applying\n"
            "the valid mask during aggregation before computing n_valid/q_uut/q_lut/dlnq.\n"
        )

        if cfg.require_q_masked_by_valid:
            raise RuntimeError(msg)


# Apply the MLS valid mask directly inside the aggregation.
def _apply_valid_mask_to_q_lnq(
    ds: xr.Dataset,
    cfg: AggregateConfig,
    valid_var: str,
) -> xr.Dataset:
    with timed("applying valid mask to q and lnq"):
        valid_raw = ds[valid_var]

        if valid_raw.dtype == bool:
            valid_bool = valid_raw
        else:
            valid_bool = valid_raw.fillna(0) != 0

        ds = ds.copy(deep=False)
        ds[cfg.q_var] = ds[cfg.q_var].where(valid_bool)
        ds[cfg.lnq_var] = ds[cfg.lnq_var].where(valid_bool)

    return ds


# Convert longitude to [-180, 180).
def _normalize_lon_to_180(lon: xr.DataArray) -> xr.DataArray:
    return ((lon + 180) % 360) - 180


def _make_interval_labels(edges: Sequence[float], prefix: str) -> list[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_str = f"{lo:g}" if np.isfinite(lo) else "-inf"
        hi_str = f"{hi:g}" if np.isfinite(hi) else "inf"
        labels.append(f"{prefix}_{lo_str}_{hi_str}")
    return labels

# Convert a label into a safe variable-name component.
def _safe_label_for_var(label: str) -> str:
    return (
        str(label)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )

def _bin_midpoints_from_edges(edges: Sequence[float]) -> np.ndarray:
    mids = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isfinite(lo) and np.isfinite(hi):
            mids.append(0.5 * (lo + hi))
        else:
            mids.append(np.nan)
    return np.array(mids, dtype=float)


# Return boolean mask for levels in [pmin, pmax].
def _select_ut_levels(pressure_hpa: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    return (pressure_hpa >= pmin) & (pressure_hpa <= pmax)


def _safe_nanmean(x: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _safe_nanmedian(x: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(x, axis=axis)


# Build a profile-level DataFrame for grouping.
def _build_profile_dataframe(
    ds: xr.Dataset,
    cfg: AggregateConfig,
    omega_var: str,
    olr_var: str,
) -> pd.DataFrame:

    with timed("building profile DataFrame and bins"):
        profile_dim = cfg.profile_dim

        lat = ds[cfg.lat_var].values
        lon = _normalize_lon_to_180(ds[cfg.lon_var]).values
        sst = ds[cfg.sst_var].values
        omega = ds[omega_var].values
        olr = ds[olr_var].values

        df = pd.DataFrame({
            "profile_index": np.arange(ds.sizes[profile_dim]),
            "lat": lat,
            "lon": lon,
            "sst": sst,
            "omega": omega,
            "olr": olr,
        })

        lat_edges = np.arange(-90, 90 + cfg.lat_bin_deg, cfg.lat_bin_deg)
        lon_edges = np.arange(-180, 180 + cfg.lon_bin_deg, cfg.lon_bin_deg)

        df["lat_bin"] = pd.cut(df["lat"], bins=lat_edges, right=False)
        df["lon_bin"] = pd.cut(df["lon"], bins=lon_edges, right=False)

        sst_labels = _make_interval_labels(cfg.sst_bin_edges, "sst")
        df["sst_bin"] = pd.cut(
            df["sst"],
            bins=np.array(cfg.sst_bin_edges, dtype=float),
            labels=sst_labels,
            right=False,
            include_lowest=True,
        )

        df["omega_bin"] = pd.cut(
            df["omega"],
            bins=np.array(cfg.omega_bin_edges, dtype=float),
            labels=list(cfg.omega_bin_labels),
            right=False,
            include_lowest=True,
        )

        df["olr_bin"] = pd.cut(
            df["olr"],
            bins=np.array(cfg.olr_bin_edges, dtype=float),
            labels=list(cfg.olr_bin_labels),
            right=False,
            include_lowest=True,
        )

        df["lat_bin_center"] = (
            df["lat_bin"]
            .apply(lambda x: float(x.left + 0.5 * (x.right - x.left)) if pd.notnull(x) else np.nan)
            .astype(float)
        )

        df["lon_bin_center"] = (
            df["lon_bin"]
            .apply(lambda x: float(x.left + 0.5 * (x.right - x.left)) if pd.notnull(x) else np.nan)
            .astype(float)
        )

        sst_mids = _bin_midpoints_from_edges(cfg.sst_bin_edges)
        sst_map = {lab: mid for lab, mid in zip(sst_labels, sst_mids)}
        df["sst_bin_center"] = df["sst_bin"].astype(str).map(sst_map)

        olr_mids = _bin_midpoints_from_edges(cfg.olr_bin_edges)
        olr_map = {lab: mid for lab, mid in zip(cfg.olr_bin_labels, olr_mids)}
        df["olr_bin_center"] = df["olr_bin"].astype(str).map(olr_map)

        df["lat_bin_center"] = pd.to_numeric(df["lat_bin_center"], errors="coerce").astype(float)
        df["lon_bin_center"] = pd.to_numeric(df["lon_bin_center"], errors="coerce").astype(float)
        df["sst_bin_center"] = pd.to_numeric(df["sst_bin_center"], errors="coerce").astype(float)
        df["olr_bin_center"] = pd.to_numeric(df["olr_bin_center"], errors="coerce").astype(float)

    return df


# Return a dict of validity DataArrays without copying the full dataset.
def _compute_profile_validity_flags(
    ds: xr.Dataset,
    cfg: AggregateConfig,
    pressure_var: str,
    omega_var: str,
    olr_var: str,
) -> dict[str, xr.DataArray]:

    with timed("constructing profile validity flags"):
        p = ds[pressure_var].values.astype(float)
        q = ds[cfg.q_var]
        lnq = ds[cfg.lnq_var]

        ut_mask = _select_ut_levels(p, cfg.p_min_hpa, cfg.p_max_hpa)
        uut_mask = _select_ut_levels(p, cfg.uut_min_hpa, cfg.uut_max_hpa)
        lut_mask = _select_ut_levels(p, cfg.lut_min_hpa, cfg.lut_max_hpa)

        def _n_valid(da, mask):
            idx = np.where(mask)[0]
            return np.isfinite(da.isel({cfg.lev_dim: idx})).sum(dim=cfg.lev_dim)

        n_valid_ut = _n_valid(q, ut_mask)
        n_valid_lnq = _n_valid(lnq, ut_mask)
        n_valid_uut = _n_valid(q, uut_mask)
        n_valid_lut = _n_valid(q, lut_mask)

        lat_ok = (ds[cfg.lat_var] >= cfg.lat_min) & (ds[cfg.lat_var] <= cfg.lat_max)
        sst_ok = np.isfinite(ds[cfg.sst_var])
        omega_ok = np.isfinite(ds[omega_var])
        olr_ok = np.isfinite(ds[olr_var])
        time_ok = xr.DataArray(
            ~pd.isnull(pd.to_datetime(ds[cfg.time_var].values)),
            dims=(cfg.profile_dim,),
        )

        profile_ok = (
            lat_ok & sst_ok & omega_ok & olr_ok & time_ok
            & (n_valid_ut >= cfg.min_valid_levels_in_ut)
            & (n_valid_lnq >= cfg.min_valid_levels_in_ut)
            & (n_valid_uut >= cfg.min_valid_levels_in_uut)
            & (n_valid_lut >= cfg.min_valid_levels_in_lut)
        )

    return {
        "n_valid_q_ut": n_valid_ut,
        "n_valid_lnq_ut": n_valid_lnq,
        "n_valid_q_uut": n_valid_uut,
        "n_valid_q_lut": n_valid_lut,
        "profile_ok": profile_ok,
    }


# Mean over selected vertical indices, ignoring NaN.
def _layer_mean_numpy(q_np: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return _safe_nanmean(q_np[:, idx], axis=1)


# Aggregate all groups using pre-extracted numpy arrays.
def _aggregate_all_groups(
    df: pd.DataFrame,
    q_ut_np: np.ndarray,
    lnq_ut_np: np.ndarray,
    dlnq_np: np.ndarray,
    q_uut_np: np.ndarray,
    q_lut_np: np.ndarray,
    cfg: AggregateConfig,
) -> list[dict]:

    # One sample is a cell-regime climatology pooled over the whole record: a grid cell,
    # an SST bin and a vertical-motion category, with no time key. See Section 2.4.
    group_cols = ["lat_bin", "lon_bin", "sst_bin", "omega_bin"]
    if cfg.group_by_olr:
        group_cols.append("olr_bin")

    with timed("creating pandas groupby object"):
        grouped = df.groupby(group_cols, observed=True, sort=True)
        n_groups = int(grouped.ngroups)


    rows = []

    with timed("aggregating all groups"):
        for group_i, (_, g) in enumerate(grouped, start=1):
            if (
                group_i == 1
                or group_i % cfg.aggregation_progress_every == 0
                or group_i == n_groups
            ):
                print(f"  aggregating group {group_i:,}/{n_groups:,}", flush=True)

            prof_idx = g["profile_index"].to_numpy(dtype=int)

            q_sub = q_ut_np[prof_idx, :]
            lnq_sub = lnq_ut_np[prof_idx, :]

            # Vertical benchmark profile for this independent sample.
            q_mean_profile = _safe_nanmean(q_sub, axis=0)
            q_median_profile = _safe_nanmedian(q_sub, axis=0)
            lnq_mean_profile = _safe_nanmean(lnq_sub, axis=0)
            lnq_median_profile = _safe_nanmedian(lnq_sub, axis=0)

            # New: valid-count diagnostics by pressure level.
            n_valid_q_profile = np.sum(np.isfinite(q_sub), axis=0).astype(np.int32)
            n_valid_lnq_profile = np.sum(np.isfinite(lnq_sub), axis=0).astype(np.int32)

            # Optional within-sample spread diagnostics.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                q_profile_std = np.nanstd(q_sub, axis=0)
                lnq_profile_std = np.nanstd(lnq_sub, axis=0)

            dlnq_g = dlnq_np[prof_idx]
            q_uut_g = q_uut_np[prof_idx]
            q_lut_g = q_lut_np[prof_idx]

            # Modal OLR class, retained for backward compatibility and quick diagnosis.
            olr_mode = g["olr_bin"].mode(dropna=True)
            dominant_olr_bin = str(olr_mode.iloc[0]) if len(olr_mode) > 0 else "nan"

            # New: true within-sample OLR composition.
            olr_labels_in_group = g["olr_bin"].dropna().astype(str)
            olr_counts = olr_labels_in_group.value_counts()
            n_olr_total = int(len(olr_labels_in_group))

            olr_count_values = {}
            olr_frac_values = {}

            for label in cfg.olr_bin_labels:
                safe_label = _safe_label_for_var(label)
                count = int(olr_counts.get(str(label), 0))

                olr_count_values[f"n_olr_{safe_label}"] = count
                olr_frac_values[f"frac_olr_{safe_label}"] = (
                    count / n_olr_total if n_olr_total > 0 else np.nan
                )

            row0 = g.iloc[0]

            rows.append({
                "lat_bin_center": float(row0["lat_bin_center"]),
                "lon_bin_center": float(row0["lon_bin_center"]),
                "sst_bin": str(row0["sst_bin"]),
                "sst_bin_center": float(row0["sst_bin_center"]) if pd.notnull(row0["sst_bin_center"]) else np.nan,
                "omega_bin": str(row0["omega_bin"]),
                "dominant_olr_bin": dominant_olr_bin,

                "n_profiles": int(len(g)),
                "mean_sst": float(g["sst"].mean()),
                "median_sst": float(g["sst"].median()),
                "mean_omega": float(g["omega"].mean()),
                "median_omega": float(g["omega"].median()),
                "mean_olr": float(g["olr"].mean()),
                "median_olr": float(g["olr"].median()),

                # New OLR composition fields.
                "n_olr_total": n_olr_total,
                **olr_count_values,
                **olr_frac_values,

                "q_uut_mean": float(_safe_nanmean(q_uut_g)),
                "q_lut_mean": float(_safe_nanmean(q_lut_g)),
                "dlnq_mean": float(_safe_nanmean(dlnq_g)),
                "dlnq_median": float(_safe_nanmedian(dlnq_g)),

                "q_mean_profile": q_mean_profile,
                "q_median_profile": q_median_profile,
                "lnq_mean_profile": lnq_mean_profile,
                "lnq_median_profile": lnq_median_profile,

                # New profile-support/spread diagnostics.
                "n_valid_q_profile": n_valid_q_profile,
                "n_valid_lnq_profile": n_valid_lnq_profile,
                "q_profile_std": q_profile_std,
                "lnq_profile_std": lnq_profile_std,
            })

    return rows


# PRESSURE UNIT AUTO-DETECTION

# Auto-detect Pa vs hPa pressure values.
def _detect_and_fix_pressure_units(
    ds: xr.Dataset,
    cfg: AggregateConfig,
    pressure_var: str,
) -> xr.Dataset:

    with timed("detecting/fixing pressure units"):
        p = ds[pressure_var].values.astype(float)

        in_range = int(_select_ut_levels(p, cfg.p_min_hpa, cfg.p_max_hpa).sum())
        if in_range > 0:
            return ds

        in_range_pa = int(_select_ut_levels(p / 100.0, cfg.p_min_hpa, cfg.p_max_hpa).sum())
        if in_range_pa > 0:

            fixed = p / 100.0

            if pressure_var in ds.coords:
                ds = ds.assign_coords({pressure_var: fixed})
            else:
                ds[pressure_var] = xr.DataArray(fixed, dims=ds[pressure_var].dims)

            return ds


    return ds


# FILE LOADING

# Open one collocated file and standardize its profile axis.
def _open_one_profile_file(path: str, cfg: AggregateConfig) -> xr.Dataset:

    chunks = (
        {cfg.profile_dim: cfg.dask_profile_chunks}
        if cfg.dask_profile_chunks is not None
        else None
    )

    open_kwargs = {}
    if cfg.netcdf_engine is not None:
        open_kwargs["engine"] = cfg.netcdf_engine

    ds = xr.open_dataset(path, chunks=chunks, **open_kwargs)

    if cfg.profile_dim not in ds.dims:
        raise ValueError(
            f"{path} does not contain expected profile dim '{cfg.profile_dim}'. "
            f"Found dims: {dict(ds.dims)}"
        )

    try:
        if cfg.profile_dim in ds.indexes:
            ds = ds.drop_indexes(cfg.profile_dim)
    except Exception:
        pass

    ds = ds.assign_coords({cfg.profile_dim: np.arange(ds.sizes[cfg.profile_dim])})

    return ds


# Load many profile files by concatenating in batches along profile.
def _load_profile_files_batched(
    input_paths: Sequence[str],
    cfg: AggregateConfig,
) -> xr.Dataset:

    if not input_paths:
        raise FileNotFoundError("No input files were provided to the aggregation.")

    batch_size = cfg.file_batch_size
    batch_datasets = []
    n_batches = (len(input_paths) + batch_size - 1) // batch_size

    with timed(f"loading {len(input_paths):,} files in {n_batches:,} batches"):
        for i in range(0, len(input_paths), batch_size):
            batch_num = i // batch_size + 1
            batch_paths = input_paths[i:i + batch_size]

            with timed(f"loading batch {batch_num:,}/{n_batches:,}"):
                parts = [_open_one_profile_file(p, cfg) for p in batch_paths]

                batch_ds = xr.concat(
                    parts,
                    dim=cfg.profile_dim,
                    data_vars="all",
                    coords="minimal",
                    compat="override",
                    combine_attrs="override",
                )

                batch_ds = batch_ds.assign_coords(
                    {cfg.profile_dim: np.arange(batch_ds.sizes[cfg.profile_dim])}
                )

                batch_datasets.append(batch_ds)


        with timed("concatenating all batches"):
            ds = xr.concat(
                batch_datasets,
                dim=cfg.profile_dim,
                data_vars="all",
                coords="minimal",
                compat="override",
                combine_attrs="override",
            )

            ds = ds.assign_coords({cfg.profile_dim: np.arange(ds.sizes[cfg.profile_dim])})

        with timed("sorting profiles by time"):
            ds = ds.sortby(cfg.time_var)

    return ds


# SAVE

# Save dataset with zlib compression on floating-point variables. Drops unsupported dtype variables instead of crashing.
def _save_compressed(ds: xr.Dataset, path: Path, cfg: AggregateConfig) -> None:

    with timed(f"preparing compressed save: {path.name}"):
        # pandas >= 3 returns Arrow-backed string arrays from astype(str), whose dtype is
        # not a numpy dtype. Convert those to object arrays so the bin-label variables are
        # written; without this they are silently dropped and no downstream script can run.
        ds = ds.copy(deep=False)
        for v in list(ds.data_vars):
            if not isinstance(ds[v].dtype, np.dtype):
                ds[v] = (ds[v].dims, np.asarray(ds[v].values, dtype=object))

        drop_vars = []
        for v in list(ds.data_vars):
            try:
                np.dtype(ds[v].dtype)
            except TypeError:
                drop_vars.append(v)

        if drop_vars:
            print(f"  WARNING: dropping variables with unsupported dtypes: {drop_vars}")
            ds = ds.drop_vars(drop_vars)

        encoding = {}
        for v in ds.data_vars:
            try:
                if np.issubdtype(ds[v].dtype, np.floating):
                    encoding[v] = {"zlib": True, "complevel": cfg.complevel}
            except TypeError:
                pass

    save_kwargs = {"encoding": encoding}
    if cfg.netcdf_engine is not None:
        save_kwargs["engine"] = cfg.netcdf_engine

    with timed(f"writing NetCDF file: {path}"):
        ds.to_netcdf(path, **save_kwargs)


# MAIN PIPELINE

def run_aggregation(cfg: AggregateConfig) -> tuple[Path, Path]:
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load files safely
    # ------------------------------------------------------------------ #
    ds = _load_profile_files_batched(
        input_paths=list(cfg.input_paths),
        cfg=cfg,
    )

    # Optional debug profile subset. This is extremely useful to verify the code before running all 4.5M profiles.
    if cfg.debug_max_profiles is not None:
        n0 = ds.sizes[cfg.profile_dim]
        n_debug = min(cfg.debug_max_profiles, n0)
        ds = ds.isel({cfg.profile_dim: slice(0, n_debug)})

    # Detect required variables
    with timed("detecting required variables"):
        pressure_var = _find_existing_var(ds, cfg.pressure_var_candidates, "pressure variable")
        omega_var = _find_existing_var(ds, cfg.omega_var_candidates, "omega variable")
        olr_var = _find_existing_var(ds, cfg.olr_var_candidates, "olr variable")
        valid_var = _find_existing_var(ds, cfg.valid_var_candidates, "valid/QC variable")
        precision_var = _find_existing_var(ds, cfg.precision_var_candidates, "precision variable")

    # Auto-fix pressure units if needed
    ds = _detect_and_fix_pressure_units(ds, cfg, pressure_var)


    # Aggregation must not count invalid MLS retrieval levels.
    ds = _apply_valid_mask_to_q_lnq(ds, cfg, valid_var)

    # Confirm the mask worked.
    _confirm_valid_precision_masking(ds, cfg, valid_var, precision_var)

    # ------------------------------------------------------------------ #
    # 2. Profile screening
    # ------------------------------------------------------------------ #
    flags = _compute_profile_validity_flags(ds, cfg, pressure_var, omega_var, olr_var)

    with timed("computing profile_ok boolean array"):
        profile_ok = flags["profile_ok"].compute()

    n_before = int(ds.sizes[cfg.profile_dim])
    ok_indices = np.where(profile_ok.values)[0]
    n_after = len(ok_indices)
    print(f"  screening: {n_after:,} of {n_before:,} profiles retained", flush=True)


    if n_after == 0:
        raise RuntimeError(
            "0 profiles survived screening.\n"
            "Most common causes:\n"
            "  1. plev is in Pa not hPa\n"
            "  2. SST or omega variables are all NaN / fill values\n"
            "  3. q / lnq are all NaN\n"
            "  4. All profiles are outside the lat window "
            f"[{cfg.lat_min}, {cfg.lat_max}]"
        )

    # ------------------------------------------------------------------ #
    # 3. Compute UT level indices BEFORE loading screened data
    # ------------------------------------------------------------------ #
    with timed("computing UT/UUT/LUT level indices"):
        p_all = ds[pressure_var].values.astype(float)

        ut_idx = np.where(_select_ut_levels(p_all, cfg.p_min_hpa, cfg.p_max_hpa))[0]

        if len(ut_idx) == 0:
            raise RuntimeError("No UT pressure levels found after pressure unit correction.")


    # ------------------------------------------------------------------ #
    # 4. Load only needed variables + screened profiles + UT levels
    # ------------------------------------------------------------------ #
    needed_vars = [
        cfg.q_var,
        cfg.lnq_var,
        cfg.time_var,
        cfg.lat_var,
        cfg.lon_var,
        cfg.sst_var,
        omega_var,
        olr_var,
        valid_var,
        precision_var,
    ]

    # Keep pressure variable if it is a data variable.
    if pressure_var not in needed_vars:
        needed_vars.append(pressure_var)

    # Keep only existing needed variables.
    needed_vars = [v for v in needed_vars if v in ds.variables or v in ds.coords]

    with timed(
        "computing screened dataset into memory "
        "(this is usually the slow point after the 77% message)"
    ):
        ds_good = (
            ds[needed_vars]
            .isel({cfg.profile_dim: ok_indices, cfg.lev_dim: ut_idx})
            .compute()
        )


    # Attach flag arrays.
    with timed("attaching screening flag arrays to screened dataset"):
        for k, v in flags.items():
            if k == "profile_ok":
                continue
            ds_good[k] = v.isel({cfg.profile_dim: ok_indices}).compute()

    # ------------------------------------------------------------------ #
    # 5. Bin assignment
    # ------------------------------------------------------------------ #
    df = _build_profile_dataframe(ds_good, cfg, omega_var, olr_var)

    required_cols = ["lat_bin", "lon_bin", "sst_bin", "omega_bin", "olr_bin"]

    with timed("dropping profiles outside bin definitions"):
        n_pre_bins = len(df)
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        n_post_bins = len(df)
        print(f"  bin assignment: {n_post_bins:,} of {n_pre_bins:,} profiles fall inside all bins",
              flush=True)


    if len(df) == 0:
        raise RuntimeError("No profiles survived bin assignment. Check SST/omega/OLR bin edges.")

    with timed("realigning ds_good after bin drop"):
        ds_good = ds_good.isel({cfg.profile_dim: df["profile_index"].to_numpy()})
        df["profile_index"] = np.arange(len(df))


    # ------------------------------------------------------------------ #
    # 6. Vertical layer masks in UT-only dataset
    # ------------------------------------------------------------------ #
    with timed("computing layer indices inside UT-only dataset"):
        p_ut = ds_good[pressure_var].values.astype(float)

        # Because ds_good only has UT levels now, these are indices in the UT array.
        uut_idx = np.where(_select_ut_levels(p_ut, cfg.uut_min_hpa, cfg.uut_max_hpa))[0]
        lut_idx = np.where(_select_ut_levels(p_ut, cfg.lut_min_hpa, cfg.lut_max_hpa))[0]


        if len(uut_idx) == 0 or len(lut_idx) == 0:
            raise RuntimeError(
                "UUT or LUT layer has zero levels. Check uut/lut pressure bounds."
            )

    # ------------------------------------------------------------------ #
    # 7. Pre-extract numpy arrays ONCE
    # ------------------------------------------------------------------ #
    with timed("extracting q/lnq numpy arrays once"):
        q_ut_np = ds_good[cfg.q_var].values
        lnq_ut_np = ds_good[cfg.lnq_var].values


    with timed("computing q_uut, q_lut, dlnq once"):
        q_uut_np = _layer_mean_numpy(q_ut_np, uut_idx)
        q_lut_np = _layer_mean_numpy(q_ut_np, lut_idx)

        with np.errstate(invalid="ignore", divide="ignore"):
            dlnq_np = np.log(q_uut_np / q_lut_np)


    # ------------------------------------------------------------------ #
    # 8. Build screened profile-level output
    # ------------------------------------------------------------------ #
    with timed("building screened profile-level xarray dataset"):
        ds_good_ut = ds_good.copy()

        ds_good_ut["lat_bin_center"] = xr.DataArray(
            df["lat_bin_center"].values,
            dims=(cfg.profile_dim,),
        )
        ds_good_ut["lon_bin_center"] = xr.DataArray(
            df["lon_bin_center"].values,
            dims=(cfg.profile_dim,),
        )
        ds_good_ut["sst_bin_center"] = xr.DataArray(
            df["sst_bin_center"].values,
            dims=(cfg.profile_dim,),
        )
        ds_good_ut["sst_bin"] = xr.DataArray(
            df["sst_bin"].astype(str).values,
            dims=(cfg.profile_dim,),
        )
        ds_good_ut["omega_bin"] = xr.DataArray(
            df["omega_bin"].astype(str).values,
            dims=(cfg.profile_dim,),
        )
        ds_good_ut["olr_bin_center"] = xr.DataArray(
            df["olr_bin_center"].values,
            dims=(cfg.profile_dim,),
        )
        ds_good_ut["olr_bin"] = xr.DataArray(
            df["olr_bin"].astype(str).values,
            dims=(cfg.profile_dim,),
        )
        ds_good_ut["q_uut_mean"] = xr.DataArray(
            q_uut_np,
            dims=(cfg.profile_dim,),
        )
        ds_good_ut["q_lut_mean"] = xr.DataArray(
            q_lut_np,
            dims=(cfg.profile_dim,),
        )
        ds_good_ut["dlnq"] = xr.DataArray(
            dlnq_np,
            dims=(cfg.profile_dim,),
        )

        ds_good_ut[cfg.lev_dim].attrs["units"] = "hPa"
        ds_good_ut.attrs["description"] = (
            "Screened profile-level dataset. "
            "Profiles are screened by lat, SST, omega, OLR, time validity, and UT valid-level count."
        )

    screened_path = outdir / cfg.screened_profiles_name


    _save_compressed(ds_good_ut, screened_path, cfg)

    with timed("verifying screened file was saved"):
        test = xr.open_dataset(screened_path)
        test.close()


    # ------------------------------------------------------------------ #
    # 9. Aggregate into independent samples
    # ------------------------------------------------------------------ #
    rows = _aggregate_all_groups(
        df, q_ut_np, lnq_ut_np, dlnq_np, q_uut_np, q_lut_np, cfg
    )

    if not rows:
        raise RuntimeError(
            "No aggregated groups were produced. "
            "Check that profiles survive screening and fall inside valid bins."
        )

    with timed("creating aggregation DataFrame"):
        agg_df = pd.DataFrame(rows)


    # ------------------------------------------------------------------ #
    # 10. Build aggregated xarray dataset
    # ------------------------------------------------------------------ #
    with timed("building aggregated xarray dataset"):
        agg_ds = xr.Dataset(coords={"sample": np.arange(len(agg_df)), cfg.lev_dim: p_ut})

        scalar_cols = [
            "lat_bin_center",
            "lon_bin_center",
            "sst_bin",
            "sst_bin_center",
            "omega_bin",
            "dominant_olr_bin",
            "n_profiles",
            "mean_sst",
            "median_sst",
            "mean_omega",
            "median_omega",
            "mean_olr",
            "median_olr",

            # New OLR support / composition variable.
            "n_olr_total",

            "q_uut_mean",
            "q_lut_mean",
            "dlnq_mean",
            "dlnq_median",
        ]

        # Add the OLR count/fraction variables dynamically so the code stays synced with cfg.olr_bin_labels.
        for label in cfg.olr_bin_labels:
            safe_label = _safe_label_for_var(label)
            scalar_cols.append(f"n_olr_{safe_label}")
            scalar_cols.append(f"frac_olr_{safe_label}")

        for col in scalar_cols:
            agg_ds[col] = xr.DataArray(agg_df[col].values, dims=("sample",))

        profile_arrays = [
            "q_mean_profile",
            "q_median_profile",
            "lnq_mean_profile",
            "lnq_median_profile",

            # New per-pressure-level support diagnostics.
            "n_valid_q_profile",
            "n_valid_lnq_profile",

            # Optional descriptive spread inside each cell-regime aggregate.
            "q_profile_std",
            "lnq_profile_std",
        ]

        for key in profile_arrays:
            arr = np.stack(agg_df[key].values)

            agg_ds[key] = xr.DataArray(
                arr,
                dims=("sample", cfg.lev_dim),
            )

        agg_ds.attrs["description"] = (
            "aggregated benchmark-ready dataset. "
            "Each sample is a unique (lat_bin, lon_bin, sst_bin, omega_bin) unit, pooled "
            "over the whole record, unless group_by_olr=True. Per-profile observation times "
            "are kept in the profile file as 'time', seconds since 1993-01-01."
        )

        agg_ds.attrs["group_by_olr"] = str(cfg.group_by_olr)
        agg_ds.attrs["olr_role"] = "diagnostic_proxy" if not cfg.group_by_olr else "grouping_key"

        # Store bin definitions for reproducibility.
        agg_ds.attrs["sst_bin_edges"] = str(tuple(cfg.sst_bin_edges))
        agg_ds.attrs["omega_bin_edges"] = str(tuple(cfg.omega_bin_edges))
        agg_ds.attrs["omega_bin_labels"] = str(tuple(cfg.omega_bin_labels))
        agg_ds.attrs["olr_bin_edges"] = str(tuple(cfg.olr_bin_edges))
        agg_ds.attrs["olr_bin_labels"] = str(tuple(cfg.olr_bin_labels))

        agg_ds.attrs["olr_fraction_note"] = (
            "frac_olr_* variables are within-sample fractions of raw contributing "
            "profiles in each OLR class. dominant_olr_bin is the modal OLR class."
        )

        agg_ds.attrs["valid_count_note"] = (
            "n_valid_q_profile and n_valid_lnq_profile give the number of finite "
            "profile values contributing to each pressure level within each sample."
        )

    aggregated_path = outdir / cfg.aggregated_name

    # Variable metadata
    if "n_olr_total" in agg_ds:
        agg_ds["n_olr_total"].attrs["description"] = (
            "Number of raw profiles in the cell-regime aggregate with a valid OLR bin."
        )

    for label in cfg.olr_bin_labels:
        safe_label = _safe_label_for_var(label)

        n_name = f"n_olr_{safe_label}"
        f_name = f"frac_olr_{safe_label}"

        if n_name in agg_ds:
            agg_ds[n_name].attrs["description"] = (
                f"Number of raw profiles in this cell-regime aggregate classified as OLR bin '{label}'."
            )

        if f_name in agg_ds:
            agg_ds[f_name].attrs["description"] = (
                f"Fraction of raw profiles in this cell-regime aggregate classified as OLR bin '{label}'."
            )

    if "n_valid_q_profile" in agg_ds:
        agg_ds["n_valid_q_profile"].attrs["description"] = (
            "Number of finite q values contributing to each pressure level "
            "within this cell-regime aggregate."
        )

    if "n_valid_lnq_profile" in agg_ds:
        agg_ds["n_valid_lnq_profile"].attrs["description"] = (
            "Number of finite lnq values contributing to each pressure level "
            "within this cell-regime aggregate."
        )

    if "q_profile_std" in agg_ds:
        agg_ds["q_profile_std"].attrs["description"] = (
            "Within-sample standard deviation of q at each pressure level."
        )

    if "lnq_profile_std" in agg_ds:
        agg_ds["lnq_profile_std"].attrs["description"] = (
            "Within-sample standard deviation of lnq at each pressure level."
        )

    _save_compressed(agg_ds, aggregated_path, cfg)


    return screened_path, aggregated_path


# RUN SCRIPT

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DATA_ROOT = Path(os.environ.get("UTWV_DATA_ROOT", "."))
    INPUT_DIR = DATA_ROOT / "data/collocated"
    OUTPUT_DIR = DATA_ROOT / "data/aggregates/tropical_oceans/sst_295-305_0.5K"

    input_files = sorted(str(p) for p in INPUT_DIR.glob("*.nc"))
    if not input_files:
        raise FileNotFoundError(f"No .nc files found in {INPUT_DIR}")


    cfg = AggregateConfig(
        input_paths=input_files,
        output_dir=str(OUTPUT_DIR),

        netcdf_engine="netcdf4",

        # FULL DATASET RUN
        debug_max_profiles=None,

        q_var="q",
        lnq_var="lnq",
        time_var="time",
        lat_var="lat",
        lon_var="lon",
        sst_var="sst",
        omega_var_candidates=("w500", "omega500", "omega", "w"),
        olr_var_candidates=("olr", "toa_olr", "rlut"),

        lat_bin_deg=5.0,
        lon_bin_deg=5.0,
        sst_bin_edges=(295, 295.5, 296, 296.5, 297, 297.5, 298, 298.5, 299, 299.5, 300, 300.5, 301, 301.5, 302, 302.5, 303, 303.5, 304, 304.5, 305),
        omega_bin_edges=(-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf),
        omega_bin_labels=(
            "strong_ascent",
            "weak_ascent",
            "neutral",
            "weak_subsidence",
            "strong_subsidence",
        ),
        olr_bin_edges=(0.0, 220.0, 240.0, 250.0, np.inf),
        olr_bin_labels=(
            "deep_convective",
            "convective_high_cloud",
            "transition",
            "clear_sky",
        ),
        group_by_olr=False,

        dask_profile_chunks=50_000,
        file_batch_size=100,

        run_valid_precision_qc_check=True,

        aggregation_progress_every=500,
        complevel=4,
    )

    screened_path, aggregated_path = run_aggregation(cfg)


    with timed("quick plot"):
        agg = xr.open_dataset(aggregated_path)
        i = 0
        plt.plot(agg["q_mean_profile"].isel(sample=i), agg[cfg.lev_dim])
        plt.gca().invert_yaxis()
        plt.xlabel("q")
        plt.ylabel("Pressure (hPa)")
        plt.title(f"Aggregated q profile — sample {i}")
        plt.tight_layout()
        plt.show()
        agg.close()
