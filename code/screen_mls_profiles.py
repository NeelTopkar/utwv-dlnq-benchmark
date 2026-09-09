"""Quality-screen Aura MLS Level-2 water-vapor granules.

Applies the version-5 screening recommendations (Quality > 0.7, Convergence
< 2.0, even Status, finite mixing ratio and finite positive precision),
restricts each profile to the five upper-tropospheric retrieval levels between
146.78 and 316.23 hPa, and writes one screened file per granule.

Raw MLS granules are not distributed with this archive; adjust the input glob
at the foot of this file before re-running.
"""

import os
from glob import glob

import numpy as np
import xarray as xr
import h5py
from dataclasses import dataclass
from typing import Dict, Optional

P_UT_MIN, P_UT_MAX = 146.7799225, 316.2277527
P_UUT_MIN, P_UUT_MAX = 146.7799225, 215.4434662
P_LUT_MIN, P_LUT_MAX = 215.4434662, 316.2277527

# One-time helper: print the dataset paths in a granule to confirm variable names.
# Usage: inspect_h5("MLS-Aura_L2GP-H2O_v05-01_YYYYdDDD.he5")
def inspect_h5(filepath: str, max_keys: int = 200) -> None:
    with h5py.File(filepath, "r") as f:
        keys = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)

        f.visititems(visitor)

    for k in keys[:max_keys]:
        print(k)
    if len(keys) > max_keys:
        print(f"... {len(keys) - max_keys} more datasets")


# Field mapping. Only the strings on the right need adjusting for another product.
@dataclass
class MLSFields:
    q: str
    precision: Optional[str]
    pressure: str
    quality: Optional[str]
    status: Optional[str]
    convergence: Optional[str]
    latitude: Optional[str]
    longitude: Optional[str]
    time: Optional[str]

    # q: "HDFEOS/SWATHS/H2O/Data_Fields/L2gpValue"                 # water vapor (e.g. "H2O" or "H2O_MixingRatio")
    # precision: Optional["HDFEOS/SWATHS/H2O/Data Fields/H2OPrecision"]  # optional
    # pressure: "HDFEOS/SWATHS/H2O/Geolocation_Fields/Pressure"          # pressure levels (hPa or Pa)
    # quality: Optional["HDFEOS/SWATHS/H2O/Data_Fields/Quality"] # profile-level quality metric
    # status: Optional["HDFEOS/SWATHS/H2O/Data_Fields/Status"]  # profile-level integer flags (or per-level)
    # convergence: Optional["HDFEOS/SWATHS/H2O/Data_Fields/Convergence"]  # convergence indicator
    # latitude: Optional["HDFEOS/SWATHS/H2O/Geolocation_Fields/Latitude"]
    # longitude: Optional["HDFEOS/SWATHS/H2O/Geolocation_Fields/Longitude"]
    # time: Optional["HDFEOS/SWATHS/H2O/Geolocation_Fields/Time"]    # time variable (optional)

FIELDS = MLSFields(
    q="HDFEOS/SWATHS/H2O/Data Fields/L2gpValue",
    precision="HDFEOS/SWATHS/H2O/Data Fields/H2OPrecision",
    pressure="HDFEOS/SWATHS/H2O/Geolocation Fields/Pressure",
    quality="HDFEOS/SWATHS/H2O/Data Fields/Quality",
    status="HDFEOS/SWATHS/H2O/Data Fields/Status",
    convergence="HDFEOS/SWATHS/H2O/Data Fields/Convergence",
    latitude="HDFEOS/SWATHS/H2O/Geolocation Fields/Latitude",
    longitude="HDFEOS/SWATHS/H2O/Geolocation Fields/Longitude",
    time="HDFEOS/SWATHS/H2O/Geolocation Fields/Time"
)

# Raw array access.
def read_dataset(f: h5py.File, path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path not in f:
        raise KeyError(f"Dataset not found: {path}")
    return f[path][...]

def load_mls_l2(filepath: str, fields: MLSFields) -> Dict[str, Optional[np.ndarray]]:
    with h5py.File(filepath, "r") as f:
        out = {
            "q": read_dataset(f, fields.q),
            "precision": read_dataset(f, fields.precision),
            "pressure": read_dataset(f, fields.pressure),
            "quality": read_dataset(f, fields.quality),
            "status": read_dataset(f, fields.status),
            "convergence": read_dataset(f, fields.convergence),
            "lat": read_dataset(f, fields.latitude),
            "lon": read_dataset(f, fields.longitude),
            "time": read_dataset(f, fields.time),
        }
    return out

# Profile-level hard-fail checks: Status even, Quality > 0.7, Convergence < 2.0.
# ML2H2O v005 hard-fail screening: - Status even - Quality > 0.7 - Convergence < 2.0 - plus basic sanity: at least 1 finite q >0 somewhere Returns keep_profile (n_profile,) boolean.
def hard_fail_mask_ml2h2o_v5(q, quality, status, convergence):
    n_prof = q.shape[0]
    keep = np.ones(n_prof, dtype=bool)

    # must exist / finite
    keep &= np.isfinite(quality)
    keep &= np.isfinite(convergence)
    keep &= np.isfinite(status)

    # exact thresholds
    keep &= (quality > 0.7)
    keep &= (convergence < 2.0)
    keep &= ((status.astype(np.int64) % 2) == 0)


    # sanity: at least some finite retrieved values
    keep &= np.any(np.isfinite(q) & (q > 0), axis=1)

    return keep

# Level-by-level QC (positive precision + physical validity + fill handling) Level QC for ML2H2O: - q finite - q > 0 (required for ln(q)) - precision finite and > 0 (MLS convention: negative precision = flagged/bad) Returns valid_level (n_profile, n_level) boolean.
def level_valid_mask_ml2h2o(q, precision):
    valid = np.isfinite(q) & (q > 0)

    if precision is not None:
        valid &= np.isfinite(precision) & (precision > 0)

    return valid

# Subset the vertical coordinate to the upper troposphere (147–316 hPa) and keep UT masks pressure can be (n_level,) or (n_profile, n_level).
def subset_ut(pressure, q, precision, valid_level):
    p1 = pressure[0, :] if pressure.ndim == 2 else pressure
    ut_idx = (p1 >= P_UT_MIN) & (p1 <= P_UT_MAX)

    p_ut = p1[ut_idx]
    q_ut = q[:, ut_idx]
    prec_ut = precision[:, ut_idx] if precision is not None else None
    valid_ut = valid_level[:, ut_idx]

    return p_ut, q_ut, prec_ut, valid_ut

# Enforce UUT/LUT coverage, so dlnq is not computed from a couple of surviving levels.
# Require at least min_frac valid levels in BOTH UUT and LUT per profile.
def apply_ut_layer_coverage(p_ut, valid_ut, min_frac=0.5):
    uut_idx = (p_ut >= P_UUT_MIN) & (p_ut <= P_UUT_MAX)
    lut_idx = (p_ut >= P_LUT_MIN) & (p_ut <= P_LUT_MAX)

    uut_total = max(np.sum(uut_idx), 1)
    lut_total = max(np.sum(lut_idx), 1)

    uut_frac = np.sum(valid_ut[:, uut_idx], axis=1) / uut_total
    lut_frac = np.sum(valid_ut[:, lut_idx], axis=1) / lut_total

    keep_cov = (uut_frac >= min_frac) & (lut_frac >= min_frac)
    return keep_cov, uut_frac, lut_frac

# ln(q), only where the level passed QC.
def compute_lnq(q_ut, valid_ut):
    lnq = np.full_like(q_ut, np.nan, dtype=float)
    lnq[valid_ut] = np.log(q_ut[valid_ut])
    return lnq

# Record which pressure levels survived QC, per profile.
# Returns a Python list of lists: valid pressures per profile. (Use only for small samples; storing this for millions of profiles is heavy.).
def valid_levels_as_pressure_lists(p_ut, valid_ut, max_profiles=10):
    out = []
    for i in range(min(valid_ut.shape[0], max_profiles)):
        out.append(p_ut[valid_ut[i]].tolist())
    return out

# Compact representation: uint8 packed bits per profile.
def pack_valid_mask(valid_ut):
    return np.packbits(valid_ut, axis=1)

# Print and return the per-granule QC tally.
def qc_report_ml2h2o(
        quality: np.ndarray,
        status: np.ndarray,
        convergence: np.ndarray,
        keep_hard: np.ndarray,
        keep_cov: np.ndarray,
        sst: np.ndarray | None = None,
        sst_bins: tuple[tuple[float, float], ...] = ((300.0, 303.0), (303.0, 305.0)),
) -> dict:
    n = len(keep_hard)
    finite_q = np.isfinite(quality) & np.isfinite(status) & np.isfinite(convergence)

    fail_quality = finite_q & ~(quality > 0.7)
    fail_conv = finite_q & ~(convergence < 2.0)
    fail_status = finite_q & ~((status.astype(np.int64) % 2) == 0)
    fail_any_hard = ~keep_hard

    # coverage limiting factor among those that pass hard fail
    pass_hard = keep_hard
    fail_cov_given_hard = pass_hard & ~keep_cov

    report = {
        "N_total": int(n),
        "N_fail_quality": int(np.sum(fail_quality)),
        "N_fail_convergence": int(np.sum(fail_conv)),
        "N_fail_status_even": int(np.sum(fail_status)),
        "N_fail_any_hard": int(np.sum(fail_any_hard)),
        "N_pass_hard": int(np.sum(pass_hard)),
        "N_fail_cov_given_hard": int(np.sum(fail_cov_given_hard)),
        "frac_fail_cov_given_hard": float(np.sum(fail_cov_given_hard) / max(np.sum(pass_hard), 1)),
        "N_pass_final": int(np.sum(pass_hard & keep_cov)),
    }

    def _print_block(label: str, m: np.ndarray):
        nn = int(np.sum(m))
        if nn > 0:
            ph = int(np.sum(pass_hard & m))
            fc = int(np.sum(fail_cov_given_hard & m))
            print(f"    {label}: n={nn:,}  pass_hard={ph:,}  fail_coverage={fc:,}")

    print(f"  QC: N={report['N_total']:,}  fail_quality={report['N_fail_quality']:,}  "
          f"fail_convergence={report['N_fail_convergence']:,}  fail_status={report['N_fail_status_even']:,}  "
          f"pass_hard={report['N_pass_hard']:,}  fail_coverage={report['N_fail_cov_given_hard']:,}  "
          f"pass_final={report['N_pass_final']:,}")

    # Optional breakdown by SST bin.
    if sst is not None:
        sst = np.asarray(sst)
        for lo, hi in sst_bins:
            m = np.isfinite(sst) & (sst >= lo) & (sst < hi)
            _print_block(f"SST in [{lo},{hi}) K", m)

    return report

# Full per-granule screening pipeline.
def process_ml2h2o_file(filepath, fields=FIELDS):
    raw = load_mls_l2(filepath, fields)

    q = raw["q"]
    precision = raw["precision"]
    pressure = raw["pressure"]

    # Profile-level hard fail
    keep_hard = hard_fail_mask_ml2h2o_v5(
        q=q,
        quality=raw["quality"],
        status=raw["status"],
        convergence=raw["convergence"],
    )

    # Level QC
    valid_level = level_valid_mask_ml2h2o(q, precision)

    # UT subset
    p_ut, q_ut, prec_ut, valid_ut = subset_ut(pressure, q, precision, valid_level)

    # Coverage rule, per profile
    keep_cov, uut_frac, lut_frac = apply_ut_layer_coverage(p_ut, valid_ut, min_frac=0.5)

    keep_final = keep_hard & keep_cov

    qc_report_ml2h2o(raw["quality"], raw["status"], raw["convergence"], keep_hard, keep_cov)

    # ln(q)
    lnq_ut = compute_lnq(q_ut, valid_ut)

    # Level-count summaries
    uut_idx = (p_ut >= P_UUT_MIN) & (p_ut <= P_UUT_MAX)
    lut_idx = (p_ut >= P_LUT_MIN) & (p_ut <= P_LUT_MAX)
    n_valid_ut  = np.sum(valid_ut, axis=1)
    n_valid_uut = np.sum(valid_ut[:, uut_idx], axis=1)
    n_valid_lut = np.sum(valid_ut[:, lut_idx], axis=1)

    # Output dataset; the per-profile valid mask is kept
    ds = xr.Dataset(
        data_vars={
            "q": (("profile", "plev"), q_ut),
            "lnq": (("profile", "plev"), lnq_ut),
            "precision": (("profile", "plev"), (prec_ut if prec_ut is not None else np.full_like(q_ut, np.nan))),
            "valid": (("profile", "plev"), valid_ut),
            "keep_hard": (("profile",), keep_hard),
            "keep_cov": (("profile",), keep_cov),
            "keep_final": (("profile",), keep_final),
            "uut_frac_valid": (("profile",), uut_frac),
            "lut_frac_valid": (("profile",), lut_frac),
            "n_valid_ut": (("profile",), n_valid_ut),
            "n_valid_uut": (("profile",), n_valid_uut),
            "n_valid_lut": (("profile",), n_valid_lut),
        },
        coords={
            "profile": np.arange(q_ut.shape[0]),
            "plev": p_ut,
        },
        attrs={
            "product": "ML2H2O v005",
            "hard_fail": "Quality>0.7, Convergence<2.0, Status even",
            "ut_range_hPa": f"{P_UT_MIN}-{P_UT_MAX}",
            "uut_range_hPa": f"{P_UUT_MIN}-{P_UUT_MAX}",
            "lut_range_hPa": f"{P_LUT_MIN}-{P_LUT_MAX}",
        }
    )

    # attach geolocation if present
    if raw.get("lat") is not None: ds["lat"] = (("profile",), raw["lat"])
    if raw.get("lon") is not None: ds["lon"] = (("profile",), raw["lon"])
    if raw.get("time") is not None: ds["time"] = (("profile",), raw["time"])

    # usually you carry forward only the passing profiles:
    ds_clean = ds.isel(profile=np.where(ds["keep_final"].values)[0])

    return ds_clean


# Process every granule and write one screened file each.

MLS_GLOB = r"data/mls_l2_h2o/*.he5"
OUT_DIR = r"data/mls_qc_processed"


def main(pattern: str = MLS_GLOB, out_dir: str = OUT_DIR) -> None:
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No MLS granules found. Looked for: {pattern}")

    os.makedirs(out_dir, exist_ok=True)

    for i, fp in enumerate(files, start=1):
        ds = process_ml2h2o_file(fp)
        out_name = os.path.basename(fp).replace(".he5", "_UT.nc")
        ds.to_netcdf(os.path.join(out_dir, out_name))
        print(f"[{i}/{len(files)}] wrote {out_name} ({ds.sizes.get('profile', 0)} profiles)", flush=True)
        ds.close()


if __name__ == "__main__":
    main()
