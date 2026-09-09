"""Benchmark profiles and per-bin summaries.

Produces mean and median water-vapor profiles by SST bin and vertical-motion
category, together with sample-support tables and outgoing-longwave-radiation
composition diagnostics.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
import xarray as xr


# Configuration


# Central configuration for the benchmark build. You should only need to change paths or the minimum sample threshold for testing/full-scale runs.
@dataclass
class BenchmarkConfig:

    # Paths
    input_path: str = r"data/aggregates/tropical_oceans/sst_303-307_0.5K/aggregated_gridcells.nc"

    output_dir: str = r"results/benchmarks/sst_303-307_0.5K"
    benchmark_nc_name: str = "benchmark_profiles.nc"
    summary_csv_name: str = "benchmark_summary.csv"
    counts_csv_name: str = "counts_summary.csv"
    qc_report_name: str = "qc_report.txt"

    # Dimension candidates
    sample_dim_candidates: Tuple[str, ...] = ("sample",)
    plev_dim_candidates: Tuple[str, ...] = ("plev", "lev", "pressure", "p")

    # Profile variable candidates These candidate lists make the script more tolerant of small naming differences between the aggregation versions.
    q_mean_profile_candidates: Tuple[str, ...] = (
        "q_mean_profile",
        "q_profile_mean",
        "mean_q_profile",
    )

    q_median_profile_candidates: Tuple[str, ...] = (
        "q_median_profile",
        "q_profile_median",
        "median_q_profile",
    )

    lnq_mean_profile_candidates: Tuple[str, ...] = (
        "lnq_mean_profile",
        "lnq_profile_mean",
        "mean_lnq_profile",
    )

    lnq_median_profile_candidates: Tuple[str, ...] = (
        "lnq_median_profile",
        "lnq_profile_median",
        "median_lnq_profile",
    )

    # Scalar metadata variable candidates
    sst_bin_candidates: Tuple[str, ...] = ("sst_bin", "sst_bin_label")
    sst_bin_center_candidates: Tuple[str, ...] = ("sst_bin_center",)

    omega_bin_candidates: Tuple[str, ...] = ("omega_bin", "omega_bin_label")

    olr_bin_candidates: Tuple[str, ...] = (
        "dominant_olr_bin",
        "olr_bin",
        "olr_bin_label",
    )

    mean_sst_candidates: Tuple[str, ...] = ("mean_sst", "sst_mean")
    median_sst_candidates: Tuple[str, ...] = ("median_sst", "sst_median")

    mean_omega_candidates: Tuple[str, ...] = (
        "mean_omega",
        "omega_mean",
        "mean_w500",
        "w500_mean",
    )
    median_omega_candidates: Tuple[str, ...] = (
        "median_omega",
        "omega_median",
        "median_w500",
        "w500_median",
    )

    mean_olr_candidates: Tuple[str, ...] = ("mean_olr", "olr_mean")
    median_olr_candidates: Tuple[str, ...] = ("median_olr", "olr_median")

    q_uut_mean_candidates: Tuple[str, ...] = ("q_uut_mean", "mean_q_uut")
    q_lut_mean_candidates: Tuple[str, ...] = ("q_lut_mean", "mean_q_lut")

    dlnq_mean_candidates: Tuple[str, ...] = ("dlnq_mean", "mean_dlnq")
    dlnq_median_candidates: Tuple[str, ...] = ("dlnq_median", "median_dlnq")

    n_profiles_candidates: Tuple[str, ...] = ("n_profiles", "profile_count")

    # Aggregation diagnostic variables These are optional so the script can still read older aggregated files.
    n_olr_total_candidates: Tuple[str, ...] = ("n_olr_total",)

    n_valid_q_profile_candidates: Tuple[str, ...] = ("n_valid_q_profile",)
    n_valid_lnq_profile_candidates: Tuple[str, ...] = ("n_valid_lnq_profile",)

    q_profile_std_candidates: Tuple[str, ...] = ("q_profile_std",)
    lnq_profile_std_candidates: Tuple[str, ...] = ("lnq_profile_std",)

    day_candidates: Tuple[str, ...] = ("day", "day_utc", "date")
    lat_bin_center_candidates: Tuple[str, ...] = ("lat_bin_center",)
    lon_bin_center_candidates: Tuple[str, ...] = ("lon_bin_center",)

    # Benchmark groupings group_type is the label stored in the output. group_keys are columns in the benchmark metadata DataFrame.
    groupings: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("sst_only", ("sst_bin",)),
        ("sst_by_omega", ("sst_bin", "omega_bin")),
    )

    # Minimum sample threshold Full-scale run recommendation: 10 or higher. Small test run: set this to 1.
    min_samples_per_group: int = 10

    # Percentiles saved for profile envelopes
    profile_percentiles: Tuple[int, ...] = (10, 25, 75, 90)

    # OLR labels expected from the aggregation These are used to create stable fraction columns even if a group has zero samples in one OLR class.
    olr_labels: Tuple[str, ...] = (
        "deep_convective",
        "convective_high_cloud",
        "transition",
        "clear_sky",
    )

    # Output behavior
    overwrite_outputs: bool = True

    # Optional diagnostic profile family: If True and q_median_profile / lnq_median_profile exist, the script also stores benchmark statistics using those median-per-sample profiles.
    write_median_input_sensitivity: bool = True


# Small utility helpers


# Find the first candidate that exists as either a data variable or coordinate.
def _find_first_existing_name(
    ds: xr.Dataset,
    candidates: Sequence[str],
    *,
    required: bool,
    kind: str,
) -> Optional[str]:
    for name in candidates:
        if name in ds.variables or name in ds.coords:
            return name

    if required:
        available = sorted(list(ds.variables))
        raise ValueError(
            f"Could not find required {kind}. Tried candidates: {candidates}\n"
            f"Available variables/coords include:\n{available}"
        )

    return None


# Find the first candidate dimension that exists in the dataset.
def _find_first_existing_dim(
    ds: xr.Dataset,
    candidates: Sequence[str],
    *,
    required: bool,
    kind: str,
) -> Optional[str]:
    for name in candidates:
        if name in ds.dims:
            return name

    if required:
        available = sorted(list(ds.dims))
        raise ValueError(
            f"Could not find required {kind}. Tried candidates: {candidates}\n"
            f"Available dimensions: {available}"
        )

    return None


# Convert string/bytes/numeric label arrays into clean object strings.
def _decode_label_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)

    # Handle possible NetCDF char arrays shaped like (sample, strlen). If the array is 2-D and string-like, join across the last axis.
    if arr.ndim == 2 and arr.dtype.kind in {"S", "U"}:
        joined = []
        for row in arr:
            pieces = []
            for x in row:
                if isinstance(x, bytes):
                    pieces.append(x.decode("utf-8", errors="ignore"))
                else:
                    pieces.append(str(x))
            joined.append("".join(pieces).strip())
        arr = np.asarray(joined, dtype=object)

    # Flatten possible singleton dimensions.
    arr = np.asarray(arr).reshape(-1)

    cleaned = []
    for x in arr:
        if isinstance(x, bytes):
            s = x.decode("utf-8", errors="ignore").strip()
        elif x is None:
            s = ""
        else:
            s = str(x).strip()

        if s.lower() in {"", "nan", "none", "nat", "<na>"}:
            cleaned.append(np.nan)
        else:
            cleaned.append(s)

    return np.asarray(cleaned, dtype=object)


# Convert an array to float, replacing non-numeric values with NaN.
def _to_numeric_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)


# Extract a 1-D sample-level variable from the dataset.
def _get_1d_sample_values(
    ds: xr.Dataset,
    var_name: Optional[str],
    sample_dim: str,
    *,
    required: bool,
    numeric: bool,
    fallback: Optional[float] = None,
) -> Optional[np.ndarray]:
    n = int(ds.sizes[sample_dim])

    if var_name is None:
        if required:
            raise ValueError("Internal error: required variable name was None.")
        if fallback is not None:
            return np.full(n, fallback, dtype=float)
        return None

    da = ds[var_name]

    if sample_dim not in da.dims:
        raise ValueError(
            f"Variable {var_name!r} must contain sample dimension "
            f"{sample_dim!r}. Actual dims: {da.dims}"
        )

    # If a sample-level scalar has extra singleton dimensions, squeeze them. If it has non-singleton extra dimensions, that is probably a mistake.
    extra_dims = [d for d in da.dims if d != sample_dim]
    for d in extra_dims:
        if da.sizes[d] != 1:
            raise ValueError(
                f"Variable {var_name!r} should be sample-level, but has "
                f"non-singleton extra dimension {d!r} with size {da.sizes[d]}."
            )

    values = da.squeeze().values

    if numeric:
        return _to_numeric_array(values)

    return _decode_label_array(values)


# np.nanpercentile with warning suppression for all-NaN slices.
def _nanpercentile_safe(
    arr: np.ndarray,
    q: float,
    axis: int = 0,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(arr, q, axis=axis)


# np.nanmean with warning suppression for all-NaN slices.
def _nanmean_safe(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis)


# np.nanmedian with warning suppression for all-NaN slices.
def _nanmedian_safe(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(arr, axis=axis)


# Return the most common non-null label in a pandas Series.
def _mode_or_missing(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if len(s) == 0:
        return "missing"

    counts = s.value_counts()
    if len(counts) == 0:
        return "missing"

    return str(counts.index[0])


# Fraction of non-null samples in a group matching one label.
def _fraction_by_label(series: pd.Series, label: str) -> float:
    s = series.dropna().astype(str)
    if len(s) == 0:
        return np.nan

    return float(np.mean(s == label))


# Protect against accidental overwrite when overwrite_outputs=False.
def _ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists and overwrite_outputs=False:\n{path}"
        )

# Convert an OLR class label into a safe variable-name component.
def _safe_label_for_var(label: str) -> str:
    return (
        str(label)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


# Load an optional 2-D sample x pressure diagnostic array.
def _load_optional_profile_array(
    ds: xr.Dataset,
    var_name: Optional[str],
    names: "DetectedNames",
) -> Optional[np.ndarray]:
    if var_name is None:
        return None

    validate_profile_variable_dims(ds, names, var_name)
    return np.asarray(ds[var_name].values, dtype=float)

# Dataset detection and validation


# Stores actual variable/dimension names found in the aggregated file.
@dataclass
class DetectedNames:

    sample_dim: str
    plev_dim: str

    q_mean_profile: str
    q_median_profile: Optional[str]

    lnq_mean_profile: str
    lnq_median_profile: Optional[str]

    sst_bin: str
    sst_bin_center: Optional[str]

    omega_bin: str
    olr_bin: str

    mean_sst: str
    median_sst: Optional[str]

    mean_omega: str
    median_omega: Optional[str]

    mean_olr: str
    median_olr: Optional[str]

    q_uut_mean: Optional[str]
    q_lut_mean: Optional[str]

    dlnq_mean: str
    dlnq_median: Optional[str]

    n_profiles: str

    # Aggregation OLR-composition diagnostics.
    n_olr_total: Optional[str]
    olr_count_vars: Dict[str, str]
    olr_fraction_vars: Dict[str, str]

    # Aggregation per-level support and spread diagnostics.
    n_valid_q_profile: Optional[str]
    n_valid_lnq_profile: Optional[str]
    q_profile_std: Optional[str]
    lnq_profile_std: Optional[str]

    day: Optional[str]
    lat_bin_center: Optional[str]
    lon_bin_center: Optional[str]


# Detect dimensions and variables in the aggregated dataset.
def detect_variable_names(ds: xr.Dataset, cfg: BenchmarkConfig) -> DetectedNames:
    sample_dim = _find_first_existing_dim(
        ds,
        cfg.sample_dim_candidates,
        required=True,
        kind="sample dimension",
    )

    plev_dim = _find_first_existing_dim(
        ds,
        cfg.plev_dim_candidates,
        required=True,
        kind="pressure/vertical dimension",
    )

    q_mean_profile = _find_first_existing_name(
        ds,
        cfg.q_mean_profile_candidates,
        required=True,
        kind="mean q profile variable",
    )

    q_median_profile = _find_first_existing_name(
        ds,
        cfg.q_median_profile_candidates,
        required=False,
        kind="median q profile variable",
    )

    lnq_mean_profile = _find_first_existing_name(
        ds,
        cfg.lnq_mean_profile_candidates,
        required=True,
        kind="mean lnq profile variable",
    )

    lnq_median_profile = _find_first_existing_name(
        ds,
        cfg.lnq_median_profile_candidates,
        required=False,
        kind="median lnq profile variable",
    )

    sst_bin = _find_first_existing_name(
        ds,
        cfg.sst_bin_candidates,
        required=True,
        kind="SST bin label variable",
    )

    sst_bin_center = _find_first_existing_name(
        ds,
        cfg.sst_bin_center_candidates,
        required=False,
        kind="SST bin center variable",
    )

    omega_bin = _find_first_existing_name(
        ds,
        cfg.omega_bin_candidates,
        required=True,
        kind="omega bin label variable",
    )

    olr_bin = _find_first_existing_name(
        ds,
        cfg.olr_bin_candidates,
        required=True,
        kind="OLR diagnostic bin variable",
    )

    mean_sst = _find_first_existing_name(
        ds,
        cfg.mean_sst_candidates,
        required=True,
        kind="mean SST variable",
    )

    median_sst = _find_first_existing_name(
        ds,
        cfg.median_sst_candidates,
        required=False,
        kind="median SST variable",
    )

    mean_omega = _find_first_existing_name(
        ds,
        cfg.mean_omega_candidates,
        required=True,
        kind="mean omega variable",
    )

    median_omega = _find_first_existing_name(
        ds,
        cfg.median_omega_candidates,
        required=False,
        kind="median omega variable",
    )

    mean_olr = _find_first_existing_name(
        ds,
        cfg.mean_olr_candidates,
        required=True,
        kind="mean OLR variable",
    )

    median_olr = _find_first_existing_name(
        ds,
        cfg.median_olr_candidates,
        required=False,
        kind="median OLR variable",
    )

    q_uut_mean = _find_first_existing_name(
        ds,
        cfg.q_uut_mean_candidates,
        required=False,
        kind="UUT q mean variable",
    )

    q_lut_mean = _find_first_existing_name(
        ds,
        cfg.q_lut_mean_candidates,
        required=False,
        kind="LUT q mean variable",
    )

    dlnq_mean = _find_first_existing_name(
        ds,
        cfg.dlnq_mean_candidates,
        required=True,
        kind="mean dlnq variable",
    )

    dlnq_median = _find_first_existing_name(
        ds,
        cfg.dlnq_median_candidates,
        required=False,
        kind="median dlnq variable",
    )

    n_profiles = _find_first_existing_name(
        ds,
        cfg.n_profiles_candidates,
        required=True,
        kind="raw profile count variable",
    )

    # New optional the aggregation diagnostics.
    n_olr_total = _find_first_existing_name(
        ds,
        cfg.n_olr_total_candidates,
        required=False,
        kind="valid OLR-bin count variable",
    )

    olr_count_vars: Dict[str, str] = {}
    olr_fraction_vars: Dict[str, str] = {}

    for label in cfg.olr_labels:
        safe_label = _safe_label_for_var(label)

        n_name = _find_first_existing_name(
            ds,
            (f"n_olr_{safe_label}",),
            required=False,
            kind=f"OLR count variable for {label}",
        )

        f_name = _find_first_existing_name(
            ds,
            (f"frac_olr_{safe_label}",),
            required=False,
            kind=f"OLR fraction variable for {label}",
        )

        if n_name is not None:
            olr_count_vars[label] = n_name

        if f_name is not None:
            olr_fraction_vars[label] = f_name

    n_valid_q_profile = _find_first_existing_name(
        ds,
        cfg.n_valid_q_profile_candidates,
        required=False,
        kind="per-level valid q count variable",
    )

    n_valid_lnq_profile = _find_first_existing_name(
        ds,
        cfg.n_valid_lnq_profile_candidates,
        required=False,
        kind="per-level valid lnq count variable",
    )

    q_profile_std = _find_first_existing_name(
        ds,
        cfg.q_profile_std_candidates,
        required=False,
        kind="within-sample q profile standard deviation variable",
    )

    lnq_profile_std = _find_first_existing_name(
        ds,
        cfg.lnq_profile_std_candidates,
        required=False,
        kind="within-sample lnq profile standard deviation variable",
    )

    day = _find_first_existing_name(
        ds,
        cfg.day_candidates,
        required=False,
        kind="day variable",
    )

    lat_bin_center = _find_first_existing_name(
        ds,
        cfg.lat_bin_center_candidates,
        required=False,
        kind="latitude bin center variable",
    )

    lon_bin_center = _find_first_existing_name(
        ds,
        cfg.lon_bin_center_candidates,
        required=False,
        kind="longitude bin center variable",
    )

    return DetectedNames(
        sample_dim=sample_dim,
        plev_dim=plev_dim,
        q_mean_profile=q_mean_profile,
        q_median_profile=q_median_profile,
        lnq_mean_profile=lnq_mean_profile,
        lnq_median_profile=lnq_median_profile,
        sst_bin=sst_bin,
        sst_bin_center=sst_bin_center,
        omega_bin=omega_bin,
        olr_bin=olr_bin,
        mean_sst=mean_sst,
        median_sst=median_sst,
        mean_omega=mean_omega,
        median_omega=median_omega,
        mean_olr=mean_olr,
        median_olr=median_olr,
        q_uut_mean=q_uut_mean,
        q_lut_mean=q_lut_mean,
        dlnq_mean=dlnq_mean,
        dlnq_median=dlnq_median,
        n_profiles=n_profiles,
        n_olr_total=n_olr_total,
        olr_count_vars=olr_count_vars,
        olr_fraction_vars=olr_fraction_vars,
        n_valid_q_profile=n_valid_q_profile,
        n_valid_lnq_profile=n_valid_lnq_profile,
        q_profile_std=q_profile_std,
        lnq_profile_std=lnq_profile_std,
        day=day,
        lat_bin_center=lat_bin_center,
        lon_bin_center=lon_bin_center,
    )


# Make sure a profile variable has dimensions (sample, plev). The order must be exactly (sample, plev), because the code later treats axis 0 as sample and axis 1 as vertical pressure.
def validate_profile_variable_dims(
    ds: xr.Dataset,
    names: DetectedNames,
    var_name: str,
) -> None:
    da = ds[var_name]
    expected = (names.sample_dim, names.plev_dim)

    if da.dims != expected:
        raise ValueError(
            f"Variable {var_name!r} must have dims {expected}, "
            f"but found {da.dims}."
        )


# Validate the pieces the benchmark truly depends on.
def validate_aggregate_dataset(ds: xr.Dataset, names: DetectedNames) -> None:
    validate_profile_variable_dims(ds, names, names.q_mean_profile)
    validate_profile_variable_dims(ds, names, names.lnq_mean_profile)

    if names.q_median_profile is not None:
        validate_profile_variable_dims(ds, names, names.q_median_profile)

    if names.lnq_median_profile is not None:
        validate_profile_variable_dims(ds, names, names.lnq_median_profile)

    for optional_profile_var in [
        names.n_valid_q_profile,
        names.n_valid_lnq_profile,
        names.q_profile_std,
        names.lnq_profile_std,
    ]:
        if optional_profile_var is not None:
            validate_profile_variable_dims(ds, names, optional_profile_var)

    n_sample = ds.sizes[names.sample_dim]
    n_plev = ds.sizes[names.plev_dim]

    if n_sample <= 0:
        raise ValueError("aggregated dataset has zero samples.")

    if n_plev <= 0:
        raise ValueError("aggregated dataset has zero pressure levels.")

    plev = np.asarray(ds[names.plev_dim].values, dtype=float)

    if not np.any(np.isfinite(plev)):
        raise ValueError("Pressure coordinate contains no finite values.")

    # This is a sanity check, not a hard scientific test. For the UT window, pressure should be around 147-316 hPa, not 14700-31600 Pa.
    finite_plev = plev[np.isfinite(plev)]
    p_min = float(np.nanmin(finite_plev))
    p_max = float(np.nanmax(finite_plev))

    if p_max > 2000:
        raise ValueError(
            "Pressure coordinate looks like it may still be in Pa, not hPa. "
            f"Found range approximately {p_min:.2f} to {p_max:.2f}. "
            "the aggregation should output plev in hPa."
        )


# Build sample metadata table


# Convert cell-regime aggregate-level metadata into a pandas DataFrame.
def build_sample_dataframe(
    ds: xr.Dataset,
    names: DetectedNames,
) -> pd.DataFrame:
    sample_dim = names.sample_dim
    n = int(ds.sizes[sample_dim])

    df = pd.DataFrame(
        {
            "sample_index": np.arange(n, dtype=int),

            # Main regime labels.
            "sst_bin": _get_1d_sample_values(
                ds, names.sst_bin, sample_dim, required=True, numeric=False
            ),
            "omega_bin": _get_1d_sample_values(
                ds, names.omega_bin, sample_dim, required=True, numeric=False
            ),
            "dominant_olr_bin": _get_1d_sample_values(
                ds, names.olr_bin, sample_dim, required=True, numeric=False
            ),

            # Environmental scalars.
            "mean_sst": _get_1d_sample_values(
                ds, names.mean_sst, sample_dim, required=True, numeric=True
            ),
            "mean_omega": _get_1d_sample_values(
                ds, names.mean_omega, sample_dim, required=True, numeric=True
            ),
            "mean_olr": _get_1d_sample_values(
                ds, names.mean_olr, sample_dim, required=True, numeric=True
            ),

            # Profile-shape metric from the aggregation.
            "dlnq_mean": _get_1d_sample_values(
                ds, names.dlnq_mean, sample_dim, required=True, numeric=True
            ),

            # Number of raw profiles that contributed to each cell-regime aggregate.
            "n_profiles": _get_1d_sample_values(
                ds, names.n_profiles, sample_dim, required=True, numeric=True
            ),
        }
    )

    # New optional OLR-composition diagnostics from the aggregation. These are stronger than dominant_olr_bin because they preserve the within-sample fraction/count of raw contributing profiles in each OLR class.
    df["n_olr_total"] = _get_1d_sample_values(
        ds,
        names.n_olr_total,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    for label in names.olr_fraction_vars:
        safe_label = _safe_label_for_var(label)
        df[f"sample_frac_olr_{safe_label}"] = _get_1d_sample_values(
            ds,
            names.olr_fraction_vars[label],
            sample_dim,
            required=False,
            numeric=True,
            fallback=np.nan,
        )

    for label in names.olr_count_vars:
        safe_label = _safe_label_for_var(label)
        df[f"n_olr_{safe_label}"] = _get_1d_sample_values(
            ds,
            names.olr_count_vars[label],
            sample_dim,
            required=False,
            numeric=True,
            fallback=np.nan,
        )

    # Optional scalar variables.
    df["median_sst"] = _get_1d_sample_values(
        ds,
        names.median_sst,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    df["median_omega"] = _get_1d_sample_values(
        ds,
        names.median_omega,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    df["median_olr"] = _get_1d_sample_values(
        ds,
        names.median_olr,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    df["dlnq_median"] = _get_1d_sample_values(
        ds,
        names.dlnq_median,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    df["q_uut_mean"] = _get_1d_sample_values(
        ds,
        names.q_uut_mean,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    df["q_lut_mean"] = _get_1d_sample_values(
        ds,
        names.q_lut_mean,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    df["sst_bin_center"] = _get_1d_sample_values(
        ds,
        names.sst_bin_center,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    df["lat_bin_center"] = _get_1d_sample_values(
        ds,
        names.lat_bin_center,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    df["lon_bin_center"] = _get_1d_sample_values(
        ds,
        names.lon_bin_center,
        sample_dim,
        required=False,
        numeric=True,
        fallback=np.nan,
    )

    # Day can be datetime-like, numeric, or absent. Keep it mostly for possible QC.
    if names.day is not None:
        day_values = ds[names.day].values
        if np.asarray(day_values).reshape(-1).shape[0] == n:
            df["day"] = np.asarray(day_values).reshape(-1)
        else:
            df["day"] = pd.NaT
    else:
        df["day"] = pd.NaT

    # Drop unusable samples. These are the minimum values needed for benchmark grouping.
    before = len(df)

    df = df.dropna(
        subset=[
            "sst_bin",
            "omega_bin",
            "dominant_olr_bin",
            "mean_sst",
            "mean_omega",
            "mean_olr",
            "dlnq_mean",
            "n_profiles",
        ]
    ).copy()

    # Keep only samples with at least one raw contributing profile.
    df = df[df["n_profiles"] > 0].copy()

    after = len(df)
    print(f"  usable samples: {after:,} of {before:,}", flush=True)

    return df


# Group summary functions


# Compute vertical benchmark statistics for one profile family. Example: prefix="q" creates: q_profile_mean q_profile_median q_profile_p25 q_profile_p75 cfg: BenchmarkConfig.
def summarize_profile_family(
    profile_array: np.ndarray,
    idx: np.ndarray,
    prefix: str,
    cfg: BenchmarkConfig,
) -> Dict[str, np.ndarray]:
    sub = profile_array[idx, :]

    out = {
        f"{prefix}_profile_mean": _nanmean_safe(sub, axis=0),
        f"{prefix}_profile_median": _nanmedian_safe(sub, axis=0),
        f"{prefix}_profile_n_valid": np.sum(np.isfinite(sub), axis=0).astype(float),
    }

    for p in cfg.profile_percentiles:
        out[f"{prefix}_profile_p{p}"] = _nanpercentile_safe(sub, p, axis=0)

    return out


# Compute all scalar and vertical profile summaries for one benchmark group.
def summarize_one_group(
    *,
    group_id: int,
    group_type: str,
    group_keys: Sequence[str],
    group_values: Sequence[object],
    df_group: pd.DataFrame,
    q_mean_profiles: np.ndarray,
    lnq_mean_profiles: np.ndarray,
    q_median_profiles: Optional[np.ndarray],
    lnq_median_profiles: Optional[np.ndarray],
    n_valid_q_profiles: Optional[np.ndarray],
    n_valid_lnq_profiles: Optional[np.ndarray],
    q_profile_stds: Optional[np.ndarray],
    lnq_profile_stds: Optional[np.ndarray],
    cfg: BenchmarkConfig,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    idx = df_group["sample_index"].to_numpy(dtype=int)

    scalar: Dict[str, object] = {}

    # Basic identifiers.
    scalar["group_id"] = int(group_id)
    scalar["group_type"] = str(group_type)

    # Store all grouping labels in stable columns. For SST-only groups, omega_bin is set to "ALL".
    scalar["sst_bin"] = "ALL"
    scalar["omega_bin"] = "ALL"

    for key, value in zip(group_keys, group_values):
        scalar[key] = str(value)

    # Sample support.
    scalar["n_samples"] = int(len(df_group))
    scalar["n_raw_profiles_total"] = int(np.nansum(df_group["n_profiles"]))
    scalar["mean_profiles_per_sample"] = float(np.nanmean(df_group["n_profiles"]))
    scalar["median_profiles_per_sample"] = float(np.nanmedian(df_group["n_profiles"]))
    scalar["min_profiles_per_sample"] = float(np.nanmin(df_group["n_profiles"]))
    scalar["max_profiles_per_sample"] = float(np.nanmax(df_group["n_profiles"]))

    # Environmental summaries.
    scalar["mean_sst"] = float(np.nanmean(df_group["mean_sst"]))
    scalar["median_sst"] = float(np.nanmedian(df_group["mean_sst"]))

    if np.any(np.isfinite(df_group["median_sst"])):
        scalar["mean_of_sample_median_sst"] = float(np.nanmean(df_group["median_sst"]))
    else:
        scalar["mean_of_sample_median_sst"] = np.nan

    scalar["mean_omega"] = float(np.nanmean(df_group["mean_omega"]))
    scalar["median_omega"] = float(np.nanmedian(df_group["mean_omega"]))

    if np.any(np.isfinite(df_group["median_omega"])):
        scalar["mean_of_sample_median_omega"] = float(
            np.nanmean(df_group["median_omega"])
        )
    else:
        scalar["mean_of_sample_median_omega"] = np.nan

    scalar["mean_olr"] = float(np.nanmean(df_group["mean_olr"]))
    scalar["median_olr"] = float(np.nanmedian(df_group["mean_olr"]))

    if np.any(np.isfinite(df_group["median_olr"])):
        scalar["mean_of_sample_median_olr"] = float(np.nanmean(df_group["median_olr"]))
    else:
        scalar["mean_of_sample_median_olr"] = np.nan

    # dlnq summaries. These are descriptive; bootstrap_intervals.py bootstraps these sample-level values.
    scalar["mean_dlnq"] = float(np.nanmean(df_group["dlnq_mean"]))
    scalar["median_dlnq"] = float(np.nanmedian(df_group["dlnq_mean"]))
    scalar["p10_dlnq"] = float(np.nanpercentile(df_group["dlnq_mean"], 10))
    scalar["p25_dlnq"] = float(np.nanpercentile(df_group["dlnq_mean"], 25))
    scalar["p75_dlnq"] = float(np.nanpercentile(df_group["dlnq_mean"], 75))
    scalar["p90_dlnq"] = float(np.nanpercentile(df_group["dlnq_mean"], 90))

    if np.any(np.isfinite(df_group["dlnq_median"])):
        scalar["mean_sample_dlnq_median"] = float(np.nanmean(df_group["dlnq_median"]))
        scalar["median_sample_dlnq_median"] = float(
            np.nanmedian(df_group["dlnq_median"])
        )
    else:
        scalar["mean_sample_dlnq_median"] = np.nan
        scalar["median_sample_dlnq_median"] = np.nan

    # Layer-q summaries from the aggregation, if present.
    if np.any(np.isfinite(df_group["q_uut_mean"])):
        scalar["mean_q_uut"] = float(np.nanmean(df_group["q_uut_mean"]))
        scalar["median_q_uut"] = float(np.nanmedian(df_group["q_uut_mean"]))
    else:
        scalar["mean_q_uut"] = np.nan
        scalar["median_q_uut"] = np.nan

    if np.any(np.isfinite(df_group["q_lut_mean"])):
        scalar["mean_q_lut"] = float(np.nanmean(df_group["q_lut_mean"]))
        scalar["median_q_lut"] = float(np.nanmedian(df_group["q_lut_mean"]))
    else:
        scalar["mean_q_lut"] = np.nan
        scalar["median_q_lut"] = np.nan

    # Spatial support diagnostics. These are not the main science results, but they help identify whether one group is geographically concentrated.
    if np.any(np.isfinite(df_group["lat_bin_center"])):
        scalar["lat_min"] = float(np.nanmin(df_group["lat_bin_center"]))
        scalar["lat_max"] = float(np.nanmax(df_group["lat_bin_center"]))
    else:
        scalar["lat_min"] = np.nan
        scalar["lat_max"] = np.nan

    if np.any(np.isfinite(df_group["lon_bin_center"])):
        scalar["lon_min"] = float(np.nanmin(df_group["lon_bin_center"]))
        scalar["lon_max"] = float(np.nanmax(df_group["lon_bin_center"]))
    else:
        scalar["lon_min"] = np.nan
        scalar["lon_max"] = np.nan

    # OLR diagnostic summaries.
    scalar["dominant_olr_bin_mode"] = _mode_or_missing(df_group["dominant_olr_bin"])

    # Prefer the true within-sample OLR fractions from the aggregation. If absent, fall back to the older dominant-class fraction method.
    expected_frac_cols = [
        f"sample_frac_olr_{_safe_label_for_var(label)}"
        for label in cfg.olr_labels
    ]

    has_true_olr_fractions = all(
        col in df_group and np.any(np.isfinite(df_group[col]))
        for col in expected_frac_cols
    )

    scalar["olr_fraction_source"] = (
        "within_sample_fractions"
        if has_true_olr_fractions
        else "dominant_olr_bin_sample_fractions"
    )

    if "n_olr_total" in df_group and np.any(np.isfinite(df_group["n_olr_total"])):
        scalar["n_olr_total"] = int(np.nansum(df_group["n_olr_total"]))
    else:
        scalar["n_olr_total"] = np.nan

    true_fraction_sum = 0.0
    true_fraction_count = 0

    for label in cfg.olr_labels:
        safe_label = _safe_label_for_var(label)

        sample_frac_col = f"sample_frac_olr_{safe_label}"
        count_col = f"n_olr_{safe_label}"

        # Keep the older dominant-sample diagnostic with a clearer name.
        scalar[f"frac_dominant_olr_{label}"] = _fraction_by_label(
            df_group["dominant_olr_bin"], label
        )

        if (
            has_true_olr_fractions
            and sample_frac_col in df_group
            and np.any(np.isfinite(df_group[sample_frac_col]))
        ):
            # Main frac_olr_* output: equal-weighted mean across cell-regime aggregates.
            scalar[f"frac_olr_{label}"] = float(np.nanmean(df_group[sample_frac_col]))

            true_fraction_sum += scalar[f"frac_olr_{label}"]
            true_fraction_count += 1

            # Raw-profile-weighted fraction, diagnostic only.
            if (
                count_col in df_group
                and np.any(np.isfinite(df_group[count_col]))
                and np.isfinite(scalar["n_olr_total"])
                and scalar["n_olr_total"] > 0
            ):
                scalar[f"n_olr_{label}"] = int(np.nansum(df_group[count_col]))
                scalar[f"raw_frac_olr_{label}"] = float(
                    np.nansum(df_group[count_col]) / scalar["n_olr_total"]
                )
            else:
                scalar[f"n_olr_{label}"] = np.nan
                scalar[f"raw_frac_olr_{label}"] = np.nan

        else:
            # Backward-compatible behavior for older aggregated files.
            scalar[f"frac_olr_{label}"] = scalar[f"frac_dominant_olr_{label}"]
            scalar[f"n_olr_{label}"] = np.nan
            scalar[f"raw_frac_olr_{label}"] = np.nan

    known = set(cfg.olr_labels)
    s = df_group["dominant_olr_bin"].dropna().astype(str)

    if has_true_olr_fractions and true_fraction_count > 0:
        scalar["frac_olr_other_or_unexpected"] = float(
            max(0.0, 1.0 - true_fraction_sum)
        )
    elif len(s) > 0:
        scalar["frac_olr_other_or_unexpected"] = float(np.mean(~s.isin(known)))
    else:
        scalar["frac_olr_other_or_unexpected"] = np.nan

    if len(s) > 0:
        scalar["frac_dominant_olr_other_or_unexpected"] = float(np.mean(~s.isin(known)))
    else:
        scalar["frac_dominant_olr_other_or_unexpected"] = np.nan

    # Vertical profile summaries.
    profiles: Dict[str, np.ndarray] = {}

    profiles.update(
        summarize_profile_family(
            q_mean_profiles,
            idx,
            prefix="q",
            cfg=cfg,
        )
    )

    profiles.update(
        summarize_profile_family(
            lnq_mean_profiles,
            idx,
            prefix="lnq",
            cfg=cfg,
        )
    )

    # Optional sensitivity product: This uses the aggregation q_median_profile and lnq_median_profile as the input profile family instead of the aggregation q_mean_profile and lnq_mean_profile.
    if (
        cfg.write_median_input_sensitivity
        and q_median_profiles is not None
        and lnq_median_profiles is not None
    ):
        profiles.update(
            summarize_profile_family(
                q_median_profiles,
                idx,
                prefix="q_samplemedian",
                cfg=cfg,
            )
        )

        profiles.update(
            summarize_profile_family(
                lnq_median_profiles,
                idx,
                prefix="lnq_samplemedian",
                cfg=cfg,
            )
        )

    # Aggregation support diagnostics carried forward to the benchmark. These tell you how many raw finite values contributed to each pressure level after aggregation across cell-regime aggregates.
    if n_valid_q_profiles is not None:
        sub = n_valid_q_profiles[idx, :]

        profiles["n_valid_q_profile_total"] = np.nansum(sub, axis=0).astype(float)
        profiles["n_valid_q_profile_mean_per_sample"] = _nanmean_safe(sub, axis=0)
        profiles["n_valid_q_profile_median_per_sample"] = _nanmedian_safe(sub, axis=0)

    if n_valid_lnq_profiles is not None:
        sub = n_valid_lnq_profiles[idx, :]

        profiles["n_valid_lnq_profile_total"] = np.nansum(sub, axis=0).astype(float)
        profiles["n_valid_lnq_profile_mean_per_sample"] = _nanmean_safe(sub, axis=0)
        profiles["n_valid_lnq_profile_median_per_sample"] = _nanmedian_safe(sub, axis=0)

    # Optional within-sample spread diagnostics from the aggregation. These are QC/supplemental descriptors, not final uncertainty intervals.
    if q_profile_stds is not None:
        profiles["q_within_sample_std_mean"] = _nanmean_safe(
            q_profile_stds[idx, :],
            axis=0,
        )

    if lnq_profile_stds is not None:
        profiles["lnq_within_sample_std_mean"] = _nanmean_safe(
            lnq_profile_stds[idx, :],
            axis=0,
        )

    return scalar, profiles


# Build all benchmark groups requested in cfg.groupings. profile_lists: Dictionary mapping profile variable names to list of 1-D arrays.
def build_benchmarks(
    ds: xr.Dataset,
    names: DetectedNames,
    df: pd.DataFrame,
    cfg: BenchmarkConfig,
) -> Tuple[pd.DataFrame, Dict[str, List[np.ndarray]], np.ndarray, Dict[str, int]]:

    q_mean_profiles = np.asarray(ds[names.q_mean_profile].values, dtype=float)
    lnq_mean_profiles = np.asarray(ds[names.lnq_mean_profile].values, dtype=float)

    q_median_profiles = None
    lnq_median_profiles = None

    if (
        cfg.write_median_input_sensitivity
        and names.q_median_profile is not None
        and names.lnq_median_profile is not None
    ):
        q_median_profiles = np.asarray(ds[names.q_median_profile].values, dtype=float)
        lnq_median_profiles = np.asarray(
            ds[names.lnq_median_profile].values,
            dtype=float,
        )

    # Optional the aggregation diagnostics added after the OLR/support update.
    n_valid_q_profiles = _load_optional_profile_array(
        ds,
        names.n_valid_q_profile,
        names,
    )

    n_valid_lnq_profiles = _load_optional_profile_array(
        ds,
        names.n_valid_lnq_profile,
        names,
    )

    q_profile_stds = _load_optional_profile_array(
        ds,
        names.q_profile_std,
        names,
    )

    lnq_profile_stds = _load_optional_profile_array(
        ds,
        names.lnq_profile_std,
        names,
    )

    plev = np.asarray(ds[names.plev_dim].values, dtype=float)

    scalar_rows: List[Dict[str, object]] = []
    profile_lists: Dict[str, List[np.ndarray]] = {}

    skipped_counts: Dict[str, int] = {}
    group_id = 0

    for group_type, group_keys_tuple in cfg.groupings:
        group_keys = list(group_keys_tuple)
        skipped_counts[group_type] = 0


        grouped = df.groupby(group_keys, dropna=True, sort=True)

        for group_values, df_group in grouped:
            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            if len(df_group) < cfg.min_samples_per_group:
                skipped_counts[group_type] += 1
                continue

            scalar, profiles = summarize_one_group(
                group_id=group_id,
                group_type=group_type,
                group_keys=group_keys,
                group_values=group_values,
                df_group=df_group,
                q_mean_profiles=q_mean_profiles,
                lnq_mean_profiles=lnq_mean_profiles,
                q_median_profiles=q_median_profiles,
                lnq_median_profiles=lnq_median_profiles,
                n_valid_q_profiles=n_valid_q_profiles,
                n_valid_lnq_profiles=n_valid_lnq_profiles,
                q_profile_stds=q_profile_stds,
                lnq_profile_stds=lnq_profile_stds,
                cfg=cfg,
            )

            scalar_rows.append(scalar)

            for name, arr in profiles.items():
                profile_lists.setdefault(name, []).append(arr)

            group_id += 1


    if len(scalar_rows) == 0:
        raise ValueError(
            "No benchmark groups were created. This usually means "
            "min_samples_per_group is too high for your test dataset, or the "
            "aggregation bins are empty. For a small test run, set "
            "min_samples_per_group=1."
        )

    summary_df = pd.DataFrame(scalar_rows)

    return summary_df, profile_lists, plev, skipped_counts


# Build output Dataset


# Convert benchmark summaries into an xarray Dataset.
def build_output_dataset(
    summary_df: pd.DataFrame,
    profile_lists: Dict[str, List[np.ndarray]],
    plev: np.ndarray,
    cfg: BenchmarkConfig,
    names: DetectedNames,
    input_path: str,
) -> xr.Dataset:
    n_groups = len(summary_df)

    coords = {
        "benchmark_group": np.arange(n_groups, dtype=int),
        "plev": plev.astype(float),
    }

    data_vars = {}

    # Add vertical profile arrays.
    for var_name, list_of_arrays in profile_lists.items():
        stacked = np.vstack(list_of_arrays).astype(float)

        data_vars[var_name] = (
            ("benchmark_group", "plev"),
            stacked,
            {
                "description": (
                    f"Benchmark {var_name}, computed across "
                    "independent aggregates."
                )
            },
        )

    # Add scalar metadata columns.
    for col in summary_df.columns:
        values = summary_df[col].to_numpy()

        if values.dtype == object:
            # Store strings as fixed-width unicode. xarray/netCDF usually handles this correctly with netCDF4/h5netcdf engines.
            str_values = summary_df[col].astype(str).to_numpy(dtype=str)
            data_vars[col] = (
                ("benchmark_group",),
                str_values,
                {"description": f"Scalar benchmark metadata: {col}"},
            )
        else:
            data_vars[col] = (
                ("benchmark_group",),
                values,
                {"description": f"Scalar benchmark metadata: {col}"},
            )

    out = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "step": "6",
            "description": (
                "Benchmark UTWV profile statistics generated from the aggregation "
                "aggregated independent samples."
            ),
            "input_file": str(input_path),
            "sample_unit": (
                "cell-regime aggregate, usually day x lat/lon bin x SST bin x omega bin."
            ),
            "main_groupings": "sst_only; sst_by_omega",
            "olr_role": (
                "OLR is summarized diagnostically, not used as a default "
                "primary grouping axis."
            ),
            "olr_fraction_note": (
                "If the aggregate provides frac_olr_* variables, the benchmark frac_olr_* "
                "outputs are equal-weighted means of within-sample OLR fractions. "
                "frac_dominant_olr_* columns preserve the older modal-class "
                "sample-fraction diagnostic."
            ),
            "profile_support_note": (
                "n_valid_q_profile_* and n_valid_lnq_profile_* variables, when "
                "present, summarize how many raw finite profile values contributed "
                "at each pressure level."
            ),
            "uncertainty_note": (
                "p25-p75 and p10-p90 are descriptive sample-spread envelopes. "
                "Bootstrap uncertainty is computed by bootstrap_intervals.py."
            ),
            "source_q_profile_variable": names.q_mean_profile,
            "source_lnq_profile_variable": names.lnq_mean_profile,
            "min_samples_per_group": cfg.min_samples_per_group,
        },
    )

    # Add useful units/metadata when obvious.
    out["plev"].attrs["units"] = "hPa"
    out["plev"].attrs["description"] = "Pressure level coordinate"

    if "mean_sst" in out:
        out["mean_sst"].attrs["units"] = "K"

    if "median_sst" in out:
        out["median_sst"].attrs["units"] = "K"

    if "mean_omega" in out:
        out["mean_omega"].attrs["units"] = "Pa s-1"

    if "median_omega" in out:
        out["median_omega"].attrs["units"] = "Pa s-1"

    if "mean_olr" in out:
        out["mean_olr"].attrs["units"] = "W m-2"

    if "median_olr" in out:
        out["median_olr"].attrs["units"] = "W m-2"

    if "mean_dlnq" in out:
        out["mean_dlnq"].attrs["description"] = (
            "Mean of cell-regime aggregate-level dlnq_mean values within benchmark group"
        )

    if "olr_fraction_source" in out:
        out["olr_fraction_source"].attrs["description"] = (
            "Source used for frac_olr_* columns: true the aggregation within-sample "
            "OLR fractions when available, otherwise fallback dominant-class fractions."
        )

    if "n_olr_total" in out:
        out["n_olr_total"].attrs["description"] = (
            "Total number of raw contributing profiles with valid OLR class labels "
            "across the cell-regime aggregates in this benchmark group."
        )

    for label in cfg.olr_labels:
        if f"frac_olr_{label}" in out:
            out[f"frac_olr_{label}"].attrs["description"] = (
                "Mean within-sample OLR fraction across cell-regime aggregates if available; "
                "otherwise fallback fraction of samples whose dominant OLR class "
                f"is {label}."
            )

        if f"frac_dominant_olr_{label}" in out:
            out[f"frac_dominant_olr_{label}"].attrs["description"] = (
                f"Fraction of cell-regime aggregates whose dominant OLR class is {label}."
            )

        if f"raw_frac_olr_{label}" in out:
            out[f"raw_frac_olr_{label}"].attrs["description"] = (
                f"Raw-profile-weighted OLR fraction for class {label}, computed "
                "from the aggregation n_olr_* counts when available."
            )

    for v in [
        "n_valid_q_profile_total",
        "n_valid_lnq_profile_total",
        "n_valid_q_profile_mean_per_sample",
        "n_valid_lnq_profile_mean_per_sample",
        "n_valid_q_profile_median_per_sample",
        "n_valid_lnq_profile_median_per_sample",
        "q_within_sample_std_mean",
        "lnq_within_sample_std_mean",
    ]:
        if v in out:
            out[v].attrs["description"] = (
                f"Benchmark diagnostic variable {v}, summarized from optional "
                "Per-aggregate support/spread variables."
            )

    return out


# QC report and output writing


# Build a compact counts/support table for quick QC.
def make_counts_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
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
        "dominant_olr_bin_mode",
        "olr_fraction_source",
        "n_olr_total",
        "frac_olr_deep_convective",
        "frac_olr_convective_high_cloud",
        "frac_olr_transition",
        "frac_olr_clear_sky",
    ]

    cols = [c for c in preferred_cols if c in summary_df.columns]
    return summary_df[cols].copy()


# Create a human-readable QC report for the benchmark.
def make_qc_report_text(
    ds_in: xr.Dataset,
    ds_out: xr.Dataset,
    names: DetectedNames,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    skipped_counts: Dict[str, int],
    cfg: BenchmarkConfig,
) -> str:
    lines = []

    lines.append("BENCHMARK QC REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Input dataset")
    lines.append("-" * 80)
    lines.append(f"Input path: {cfg.input_path}")
    lines.append(f"Detected sample dimension: {names.sample_dim}")
    lines.append(f"Detected pressure dimension: {names.plev_dim}")
    lines.append(f"Input samples: {ds_in.sizes[names.sample_dim]:,}")
    lines.append(f"Input pressure levels: {ds_in.sizes[names.plev_dim]:,}")
    lines.append("")

    plev = np.asarray(ds_in[names.plev_dim].values, dtype=float)
    finite_plev = plev[np.isfinite(plev)]
    if finite_plev.size > 0:
        lines.append(
            f"Pressure range: {float(np.nanmin(finite_plev)):.3f} to "
            f"{float(np.nanmax(finite_plev)):.3f} hPa"
        )
    else:
        lines.append("Pressure range: no finite pressure values found")

    lines.append("")

    lines.append("Detected variables")
    lines.append("-" * 80)
    for key, value in names.__dict__.items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("Aggregation diagnostic support")
    lines.append("-" * 80)
    lines.append(f"n_olr_total variable: {names.n_olr_total}")
    lines.append(f"OLR count variables found: {names.olr_count_vars}")
    lines.append(f"OLR fraction variables found: {names.olr_fraction_vars}")
    lines.append(f"n_valid_q_profile variable: {names.n_valid_q_profile}")
    lines.append(f"n_valid_lnq_profile variable: {names.n_valid_lnq_profile}")
    lines.append(f"q_profile_std variable: {names.q_profile_std}")
    lines.append(f"lnq_profile_std variable: {names.lnq_profile_std}")
    lines.append("")

    lines.append("Filtered sample metadata")
    lines.append("-" * 80)
    lines.append(f"Usable cell-regime aggregates after filtering: {len(df):,}")
    lines.append(f"Minimum samples per benchmark group: {cfg.min_samples_per_group}")
    lines.append("")

    lines.append("Benchmark group creation")
    lines.append("-" * 80)
    lines.append(f"Benchmark groups created: {len(summary_df):,}")
    for group_type, skipped in skipped_counts.items():
        made = int(np.sum(summary_df["group_type"] == group_type))
        lines.append(
            f"{group_type}: created {made:,}, skipped {skipped:,} "
            f"below sample threshold"
        )
    lines.append("")

    lines.append("Group counts by type")
    lines.append("-" * 80)
    lines.append(str(summary_df["group_type"].value_counts()))
    lines.append("")

    lines.append("SST bins present")
    lines.append("-" * 80)
    lines.append(str(summary_df["sst_bin"].value_counts()))
    lines.append("")

    lines.append("Omega bins present")
    lines.append("-" * 80)
    lines.append(str(summary_df["omega_bin"].value_counts()))
    lines.append("")

    lines.append("Output dataset")
    lines.append("-" * 80)
    lines.append(f"Output benchmark groups: {ds_out.sizes['benchmark_group']:,}")
    lines.append(f"Output pressure levels: {ds_out.sizes['plev']:,}")
    lines.append("")

    lines.append("Physical range checks")
    lines.append("-" * 80)

    for col in [
        "mean_sst",
        "mean_omega",
        "mean_olr",
        "mean_dlnq",
        "n_samples",
        "n_olr_total",
        "frac_olr_deep_convective",
        "frac_olr_clear_sky",
    ]:
        if col in summary_df:
            vals = pd.to_numeric(summary_df[col], errors="coerce")
            lines.append(
                f"{col}: min={np.nanmin(vals):.6g}, "
                f"median={np.nanmedian(vals):.6g}, "
                f"max={np.nanmax(vals):.6g}"
            )

    lines.append("")
    lines.append("Notes")
    lines.append("-" * 80)
    lines.append(
        "Descriptive envelopes are not final uncertainty intervals. "
        "bootstrap_intervals.py bootstraps cell-regime aggregates."
    )
    lines.append(
        "OLR was summarized diagnostically inside each benchmark group. "
        "It was not used as a default grouping axis."
    )

    return "\n".join(lines)


# Write the benchmark NetCDF, summary CSV, counts CSV, and QC report.
def write_outputs(
    ds_in: xr.Dataset,
    ds_out: xr.Dataset,
    names: DetectedNames,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    skipped_counts: Dict[str, int],
    cfg: BenchmarkConfig,
) -> Tuple[Path, Path, Path, Path]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nc_path = out_dir / cfg.benchmark_nc_name
    summary_path = out_dir / cfg.summary_csv_name
    counts_path = out_dir / cfg.counts_csv_name
    qc_path = out_dir / cfg.qc_report_name

    for path in [nc_path, summary_path, counts_path, qc_path]:
        _ensure_output_path(path, cfg.overwrite_outputs)

    counts_df = make_counts_summary(summary_df)

    qc_text = make_qc_report_text(
        ds_in=ds_in,
        ds_out=ds_out,
        names=names,
        df=df,
        summary_df=summary_df,
        skipped_counts=skipped_counts,
        cfg=cfg,
    )

    ds_out.to_netcdf(nc_path)

    summary_df.to_csv(summary_path, index=False)

    counts_df.to_csv(counts_path, index=False)

    qc_path.write_text(qc_text, encoding="utf-8")

    return nc_path, summary_path, counts_path, qc_path


# Main runner


# Run the full benchmark-building pipeline.
def run_benchmark(cfg: BenchmarkConfig) -> Tuple[xr.Dataset, pd.DataFrame]:
    input_path = Path(cfg.input_path)


    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find aggregated file:\n{input_path}\n\n"
            "Update BenchmarkConfig.input_path to the correct location."
        )


    ds = xr.open_dataset(input_path)

    names = detect_variable_names(ds, cfg)


    validate_aggregate_dataset(ds, names)

    df = build_sample_dataframe(ds, names)

    summary_df, profile_lists, plev, skipped_counts = build_benchmarks(
        ds=ds,
        names=names,
        df=df,
        cfg=cfg,
    )

    ds_out = build_output_dataset(
        summary_df=summary_df,
        profile_lists=profile_lists,
        plev=plev,
        cfg=cfg,
        names=names,
        input_path=str(input_path),
    )

    nc_path, summary_path, counts_path, qc_path = write_outputs(
        ds_in=ds,
        ds_out=ds_out,
        names=names,
        df=df,
        summary_df=summary_df,
        skipped_counts=skipped_counts,
        cfg=cfg,
    )


    return ds_out, summary_df


# Script entry point


if __name__ == "__main__":
    cfg = BenchmarkConfig(
        # Change this path if the aggregate folder is somewhere else.
        input_path=r"data/aggregates/tropical_oceans/sst_303-307_0.5K/aggregated_gridcells.nc",

        # For a full-scale run, 10 is reasonable. For a tiny test file, set this to 1.
        min_samples_per_group=10,

        # Keep True unless you intentionally want to protect old outputs.
        overwrite_outputs=True,
    )

    run_benchmark(cfg)
