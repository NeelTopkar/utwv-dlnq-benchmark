"""Driver: rebuild the warm-pool products used in the paper.

Subsets to the warm pool and re-runs the benchmark, conditional-table and
bootstrap steps over the two binning experiments behind the reported
turning-point and robustness statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

# Settings

# Resolved, because helper scripts run as subprocesses with cwd set to code/;
# a relative root would make every derived path resolve under code/ instead.
ROOT = Path(os.environ.get("UTWV_DATA_ROOT", ".")).resolve()
CODE_DIR = Path(__file__).resolve().parent
AGGREGATE_DIR = ROOT / "data/aggregates/tropical_oceans"
OUTPUT_ROOT = ROOT / "results/warm_pool"
WARMPOOL_PROFILE_DIR = ROOT / "data/profiles/warm_pool"
MPLCONFIGDIR = OUTPUT_ROOT / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WARMPOOL_DATASETS_DIR = OUTPUT_ROOT / "aggregates"
BENCHMARK_DIR = OUTPUT_ROOT / "benchmarks"
BENCHMARK_PLOTS_DIR = OUTPUT_ROOT / "figures" / "benchmarks"
CONDITIONAL_OUT_DIR = OUTPUT_ROOT / "conditional"
INTERVAL_OUT_DIR = OUTPUT_ROOT / "bootstrap"
WEIGHTING_OUT_DIR = OUTPUT_ROOT / "weighting_analysis"
SUMMARY_PATH = OUTPUT_ROOT / "run_summary.txt"

LON_MIN, LON_MAX = 120.0, 180.0
LAT_MIN, LAT_MAX = -15.0, 15.0

MIN_SAMPLES_PER_GROUP = 10
N_REPLICATES = int(os.environ.get("WARMPOOL_N_REPLICATES", "1000"))
SEED = int(os.environ.get("WARMPOOL_SEED", "12345"))


@dataclass(frozen=True)
class Comparison:
    reference: tuple[float, float]
    warm: tuple[float, float]


@dataclass(frozen=True)
class Experiment:
    aggregate_folder: str
    tag: str
    comparisons: tuple[Comparison, ...]


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        aggregate_folder="sst_295-305_0.5K",
        tag="sst_295-305_0.5K",
        comparisons=(
            Comparison((299.5, 300.0), (300.0, 300.5)),
            Comparison((300.0, 300.5), (300.5, 301.0)),
            Comparison((300.5, 301.0), (301.5, 302.0)),
            Comparison((302.0, 302.5), (302.5, 303.0)),
            Comparison((302.5, 303.0), (303.0, 303.5)),
            Comparison((303.0, 303.5), (303.5, 304.0)),
        ),
    ),
    Experiment(
        aggregate_folder="sst_287-307_variable",
        tag="sst_287-307_variable",
        comparisons=(
            Comparison((299.0, 300.0), (300.0, 301.0)),
            Comparison((300.0, 301.0), (301.0, 302.0)),
            Comparison((301.0, 302.0), (302.0, 303.0)),
            Comparison((302.0, 303.0), (303.0, 305.0)),
        ),
    ),
)

SUBSET_FILES = (
    ("aggregated_gridcells.nc", "sample"),
    ("profiles.nc", "profile"),
)

BENCHMARK_EXPECTED_FIGURES = (
    "fig1_main_profiles_by_sst.png",
    "fig1b_lnq_profiles_by_sst.png",
    "fig2_profiles_by_sst_within_omega.png",
    "fig3_dlnq_vs_sst.png",
    "fig4_dlnq_heatmap_sst_omega.png",
    "fig4b_dlnq_anomaly_heatmap_sst_omega.png",
    "fig5_olr_mean_heatmap.png",
    "fig6_olr_true_fraction_panels.png",
    "fig6b_olr_dominant_fraction_panels.png",
    "fig7_sample_support_heatmap.png",
    "fig8_raw_profile_support_heatmap.png",
    "fig9_valid_q_fraction_by_pressure.png",
)

BENCHMARK_EXPECTED_TABLES = (
    "paper_key_results.csv",
    "olr_diagnostic_matrix.csv",
    "support_matrix.csv",
    "sample_support_matrix.csv",
    "valid_profile_support_table.csv",
)

CONDITIONAL_EXPECTED_FIGURES = (
    "fig_dlnq_vs_sst.png",
    "fig_dlnq_by_sst_and_omega.png",
    "fig_dlnq_heatmap_sst_omega.png",
    "fig_delta_dlnq_heatmap_sst_omega.png",
    "fig_uut_lut_by_sst.png",
    "fig_olr_by_sst_omega.png",
)

INTERVAL_EXPECTED_METHOD_FIGURES = (
    "fig_dlnq_ci_by_sst.png",
    "fig_dlnq_ci_by_sst_omega.png",
    "fig_dlnq_difference.png",
    "fig_profile_envelope.png",
    "fig_profile_difference.png",
)

INTERVAL_EXPECTED_COMPARISON_FIGURES = (
    "fig_compare_width_by_sst.png",
    "fig_compare_width_difference.png",
)


# Logging and safety helpers

SUMMARY_LINES: list[str] = []


def fmt_bounds(bounds: tuple[float, float]) -> str:
    lo, hi = bounds
    return f"{lo:g}-{hi:g}K"


# Directory component for one comparison: reference bin, then warm bin, no spaces.
def comparison_dir_name(comp: Comparison) -> str:
    return f"{fmt_bounds(comp.reference)[:-1]}_vs_{fmt_bounds(comp.warm)[:-1]}".replace(" ", "")


def comparison_dir_label(exp: Experiment, comp: Comparison) -> str:
    return f"{exp.tag}_{comparison_dir_name(comp)}"


def ensure_output_path(path: Path) -> Path:
    resolved_root = OUTPUT_ROOT.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise RuntimeError(f"Refusing to write outside results/warm_pool: {path}")
    return path


def mkdir(path: Path) -> Path:
    ensure_output_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_summary() -> None:
    mkdir(OUTPUT_ROOT)
    SUMMARY_PATH.write_text("\n".join(SUMMARY_LINES) + "\n", encoding="utf-8")


def require_packages() -> None:
    missing: list[str] = []
    for name in ("xarray", "netCDF4", "numpy", "pandas", "matplotlib"):
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing required Python packages: "
            + ", ".join(missing)
            + ". Create a fresh venv and install xarray netCDF4 numpy pandas matplotlib."
        )


def load_module_from_path(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


benchmark_mod = load_module_from_path("benchmark_module", CODE_DIR / "build_benchmarks.py")
import conditional_tables as conditional_mod  # noqa: E402
import bootstrap_intervals as interval_mod  # noqa: E402


# Warm-pool subsetting

def subset_dataset(ds: xr.Dataset, dim: str) -> tuple[xr.Dataset, int, int]:
    latc = ds["lat_bin_center"].values
    lonc = ds["lon_bin_center"].values
    mask = (
        (latc >= LAT_MIN)
        & (latc <= LAT_MAX)
        & (lonc >= LON_MIN)
        & (lonc <= LON_MAX)
    )
    idx = np.where(mask)[0]
    return ds.isel({dim: idx}), int(len(idx)), int(ds.sizes[dim])


# Fall back to the warm-pool per-profile file shipped with the archive when the
# tropical-ocean profiles.nc it would be subset from is not present.
def _shipped_warm_pool_profiles(exp: Experiment) -> Path | None:
    for name in ("profiles.nc", "profiles_reduced.nc"):
        candidate = WARMPOOL_PROFILE_DIR / exp.aggregate_folder / name
        if candidate.exists():
            return candidate
    return None


def subset_experiment(exp: Experiment) -> tuple[Path, Path]:
    src_dir = AGGREGATE_DIR / exp.aggregate_folder
    out_dir = mkdir(WARMPOOL_DATASETS_DIR / f"{exp.tag}")

    if not src_dir.exists():
        raise FileNotFoundError(f"Missing aggregated dataset folder: {src_dir}")

    screened = out_dir / "profiles.nc"
    for filename, dim in SUBSET_FILES:
        src = src_dir / filename
        if not src.exists():
            if filename == "profiles.nc":
                shipped = _shipped_warm_pool_profiles(exp)
                if shipped is None:
                    raise FileNotFoundError(
                        f"Neither {src} nor a warm-pool per-profile file in "
                        f"{WARMPOOL_PROFILE_DIR / exp.aggregate_folder} was found; available from the author on request")
                screened = shipped
                continue
            raise FileNotFoundError(f"Missing required aggregated file: {src}")
        dst = ensure_output_path(out_dir / filename)
        with xr.open_dataset(src) as ds:
            sub, kept, total = subset_dataset(ds, dim)
            sub.to_netcdf(dst)
            sub.close()

    agg = out_dir / "aggregated_gridcells.nc"
    return agg, screened


# Existing pipeline runners

def run_benchmark(exp: Experiment, agg_path: Path) -> Path:
    out_dir = mkdir(BENCHMARK_DIR / f"{exp.tag}")
    cfg = benchmark_mod.BenchmarkConfig(
        input_path=str(agg_path),
        output_dir=str(out_dir),
        min_samples_per_group=MIN_SAMPLES_PER_GROUP,
        overwrite_outputs=True,
    )
    benchmark_mod.run_benchmark(cfg)
    return out_dir


def run_benchmark_figures(exp: Experiment, bench_dir: Path) -> tuple[Path, Path]:
    fig_dir = mkdir(BENCHMARK_PLOTS_DIR / f"{exp.tag}" / "figures")
    table_dir = mkdir(BENCHMARK_PLOTS_DIR / f"{exp.tag}" / "tables")

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
    env["PYTHONIOENCODING"] = "utf-8"
    env["BENCHFIG_BENCHMARK_NC"] = str(bench_dir / "benchmark_profiles.nc")
    env["BENCHFIG_SUMMARY_CSV"] = str(bench_dir / "benchmark_summary.csv")
    env["BENCHFIG_COUNTS_CSV"] = str(bench_dir / "counts_summary.csv")
    env["BENCHFIG_FIG_DIR"] = str(fig_dir)
    env["BENCHFIG_TABLE_DIR"] = str(table_dir)

    script = CODE_DIR / "benchmark_figures.py"
    result = subprocess.run([sys.executable, str(script)], cwd=str(CODE_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"benchmark_figures.py failed for {exp.tag} with exit {result.returncode}")
    return fig_dir, table_dir


def run_conditional(exp: Experiment, comp: Comparison, agg_path: Path) -> Path:
    out_dir = mkdir(CONDITIONAL_OUT_DIR / f"{exp.tag}" / comparison_dir_name(comp))
    cfg = conditional_mod.ConditionalConfig(
        benchmark_name=f"warmpool_{exp.tag}",
        input_path=Path(agg_path),
        output_dir=out_dir,
        figures_dir=out_dir / "figures",
        reference_sst_bounds=comp.reference,
        warm_sst_bounds=comp.warm,
    )
    conditional_mod.run_conditional(cfg)
    return out_dir


def run_intervals(exp: Experiment, comp: Comparison, agg_path: Path) -> Path:
    out_dir = mkdir(INTERVAL_OUT_DIR / comparison_dir_label(exp, comp))
    cfg = interval_mod.IntervalConfig(
        benchmark_name=f"warmpool_{exp.tag}",
        input_path=Path(agg_path),
        output_dir=out_dir,
        figures_dir=out_dir / "figures",
        reference_sst_bounds=comp.reference,
        warm_sst_bounds=comp.warm,
        n_reps=N_REPLICATES,
        seed=SEED,
        spatial_block_deg=10.0,
    )
    interval_mod.run_comparison(cfg)
    return out_dir


# Custom weighting / decomposition / OLR analysis

def decode_values(values: Iterable) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def decode_profile_day(time_values: np.ndarray) -> np.ndarray:
    values = np.asarray(time_values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[D]")

    numeric = values.astype("float64")
    # MLS profile time is seconds since 1993-01-01 in these files.
    return (
        np.datetime64("1993-01-01T00:00:00")
        + np.round(numeric).astype("timedelta64[s]")
    ).astype("datetime64[D]")


def sst_center(label: str) -> float:
    import re

    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", str(label))]
    if len(nums) >= 2:
        return 0.5 * (nums[0] + nums[1])
    return np.nan


# Bin labels are strings in the full per-profile files and integer codes into a
# "categories" attribute in the reduced ones shipped with the archive.
def bin_labels(ds: xr.Dataset, name: str) -> list[str]:
    da = ds[name]
    categories = da.attrs.get("categories")
    if categories is not None:
        cats = categories.split("|")
        return [cats[int(i)] for i in da.values]
    return decode_values(da.values)


def load_screened_dataframe(screened_path: Path) -> pd.DataFrame:
    with xr.open_dataset(screened_path) as ds:
        df = pd.DataFrame(
            {
                "dlnq": ds["dlnq"].values,
                "q_uut": ds["q_uut_mean"].values,
                "q_lut": ds["q_lut_mean"].values,
                "sst_bin": bin_labels(ds, "sst_bin"),
                "omega_bin": bin_labels(ds, "omega_bin"),
                "lat": ds["lat_bin_center"].values,
                "lon": ds["lon_bin_center"].values,
                "day": decode_profile_day(ds["time"].values),
            }
        )
        if "olr_bin" in ds:
            df["olr_bin"] = bin_labels(ds, "olr_bin")
        else:
            df["olr_bin"] = "missing"

    df = df[
        np.isfinite(df["dlnq"])
        & np.isfinite(df["q_uut"])
        & np.isfinite(df["q_lut"])
        & (df["q_uut"] > 0)
        & (df["q_lut"] > 0)
    ].copy()
    df["cell"] = df["lat"].astype(str) + "|" + df["lon"].astype(str)
    df["sst_center"] = df["sst_bin"].map(sst_center)
    return df


def weighted_sst_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    profile = df.groupby("sst_bin", observed=True)[value_col].mean()
    perday = (
        df.groupby(["cell", "sst_bin", "omega_bin", "day"], observed=True)[value_col]
        .mean()
        .reset_index()
        .groupby("sst_bin", observed=True)[value_col]
        .mean()
    )
    cell = (
        df.groupby(["cell", "sst_bin", "omega_bin"], observed=True)[value_col]
        .mean()
        .reset_index()
        .groupby("sst_bin", observed=True)[value_col]
        .mean()
    )
    n_profiles = df.groupby("sst_bin", observed=True).size()

    out = pd.DataFrame(
        {
            "cell_weighted": cell,
            "per_day_weighted": perday,
            "profile_weighted": profile,
            "n_profiles": n_profiles,
        }
    )
    out["sst_center"] = [sst_center(x) for x in out.index]
    out = out.reset_index().sort_values(["sst_center", "sst_bin"]).reset_index(drop=True)
    return out


def plot_weighting(table: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(table["sst_center"], table["cell_weighted"], "-o", label="cell climatology")
    ax.plot(table["sst_center"], table["per_day_weighted"], "--s", label="per-day")
    ax.plot(table["sst_center"], table["profile_weighted"], ":^", label="profile")
    ax.set_xlabel("SST bin center (K)")
    ax.set_ylabel("mean dlnq")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    # Lower right is empty on the inverted axis, so the note cannot hide the minima near 303 K.
    ax.text(0.98, 0.04, "INVERTED Y-AXIS: more-negative dlnq plots higher",
            transform=ax.transAxes, ha="right", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.4"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_q_decomposition(q_table: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(q_table["sst_center"], q_table["q_uut_cell_weighted"], "-o", label="q_UUT cell")
    ax.plot(q_table["sst_center"], q_table["q_lut_cell_weighted"], "-s", label="q_LUT cell")
    ax.plot(q_table["sst_center"], q_table["q_uut_profile_weighted"], ":o", label="q_UUT profile")
    ax.plot(q_table["sst_center"], q_table["q_lut_profile_weighted"], ":s", label="q_LUT profile")
    ax.set_xlabel("SST bin center (K)")
    ax.set_ylabel("specific humidity / mixing ratio")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_olr_fractions(olr_table: pd.DataFrame, path: Path, title: str) -> None:
    labels = [c for c in olr_table.columns if c not in ("sst_bin", "sst_center", "n_profiles")]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bottom = np.zeros(len(olr_table))
    x = np.arange(len(olr_table))
    for label in labels:
        vals = olr_table[label].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(olr_table["sst_bin"], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("profile fraction")
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_weighting_analysis(exp: Experiment, screened_path: Path) -> Path:
    out_dir = mkdir(WEIGHTING_OUT_DIR / exp.tag)

    df = load_screened_dataframe(screened_path)

    dlnq_table = weighted_sst_table(df, "dlnq")
    dlnq_table.to_csv(out_dir / "weighting_dlnq_by_sst.csv", index=False)
    plot_weighting(
        dlnq_table,
        out_dir / "fig_weighting_dlnq_by_sst.png",
        f"{exp.tag}: dlnq weighting sensitivity",
    )

    q_uut = weighted_sst_table(df, "q_uut").rename(
        columns={
            "cell_weighted": "q_uut_cell_weighted",
            "per_day_weighted": "q_uut_per_day_weighted",
            "profile_weighted": "q_uut_profile_weighted",
            "n_profiles": "n_profiles_q_uut",
        }
    )
    q_lut = weighted_sst_table(df, "q_lut").rename(
        columns={
            "cell_weighted": "q_lut_cell_weighted",
            "per_day_weighted": "q_lut_per_day_weighted",
            "profile_weighted": "q_lut_profile_weighted",
            "n_profiles": "n_profiles_q_lut",
        }
    )
    # An outer merge re-sorts rows by the text label, which puts "sst_300.5_301" before
    # "sst_300_300.5"; restore numeric order so the plotted lines join adjacent bins.
    q_table = (pd.merge(q_uut, q_lut, on=["sst_bin", "sst_center"], how="outer")
               .sort_values("sst_center").reset_index(drop=True))
    q_table.to_csv(out_dir / "q_uut_lut_decomposition_by_sst.csv", index=False)
    plot_q_decomposition(
        q_table,
        out_dir / "fig_q_uut_lut_decomposition_by_sst.png",
        f"{exp.tag}: q_UUT / q_LUT decomposition",
    )

    olr_counts = (
        df.groupby(["sst_bin", "olr_bin"], observed=True)
        .size()
        .reset_index(name="count")
        .pivot(index="sst_bin", columns="olr_bin", values="count")
        .fillna(0)
    )
    olr_frac = olr_counts.div(olr_counts.sum(axis=1), axis=0)
    olr_frac["n_profiles"] = olr_counts.sum(axis=1).astype(int)
    olr_frac["sst_center"] = [sst_center(x) for x in olr_frac.index]
    olr_frac = olr_frac.reset_index().sort_values(["sst_center", "sst_bin"])
    olr_frac.to_csv(out_dir / "olr_fraction_by_sst.csv", index=False)
    plot_olr_fractions(
        olr_frac,
        out_dir / "fig_olr_fraction_by_sst.png",
        f"{exp.tag}: OLR class fractions by SST",
    )

    peak = dlnq_table.loc[dlnq_table["cell_weighted"].idxmin()]
    return out_dir


# Validation

def missing_files(folder: Path, filenames: Iterable[str]) -> list[str]:
    return [name for name in filenames if not (folder / name).exists()]


def validate_outputs(records: list[dict], skipped: list[str]) -> None:
    gaps: list[str] = []
    processed = [r["experiment"].tag for r in records]
    expected = [e.tag for e in EXPERIMENTS if e.tag not in skipped]
    if processed != expected:
        raise RuntimeError(f"Processed experiments mismatch. Expected {expected}; got {processed}")

    for record in records:
        exp = record["experiment"]
        # ensure_output_path is a write guard, so it applies only to what this driver
        # produces. screened_path can be an input read from the distributed archive.
        for key, path in record.items():
            if key in ("experiment", "screened_path") or isinstance(path, list):
                continue
            if isinstance(path, Path):
                ensure_output_path(path)

        def note(folder: Path, expected) -> None:
            for name in missing_files(folder, expected):
                gaps.append(f"{exp.tag}: {folder}/{name}")

        note(record["benchmark_figures"], BENCHMARK_EXPECTED_FIGURES)
        note(record["benchmark_tables"], BENCHMARK_EXPECTED_TABLES)

        for conditional_dir in record["conditional_dirs"]:
            note(conditional_dir / "figures", CONDITIONAL_EXPECTED_FIGURES)

        for interval_dir in record["interval_dirs"]:
            note(interval_dir / "figures", INTERVAL_EXPECTED_COMPARISON_FIGURES)
            for sub in ("iid", "block_spatial_10deg", "block_spatial_20deg"):
                note(interval_dir / sub / "figures", INTERVAL_EXPECTED_METHOD_FIGURES)

    if gaps:
        print(f"[validate] {len(gaps)} expected output(s) missing:")
        for g in gaps:
            print(f"    - {g}")
        raise RuntimeError(f"{len(gaps)} expected output(s) missing; see the list above")

    fine = next((r for r in records if "295-305" in r["experiment"].tag), None)
    if fine is not None:
        table = pd.read_csv(fine["weighting_dir"] / "weighting_dlnq_by_sst.csv")
        peak = table.loc[table["cell_weighted"].idxmin()]
        print(f"[validate] most negative cell-weighted dlnq at SST "
              f"{peak['sst_center']:.2f} K ({peak['cell_weighted']:+.4f})")


# Main

def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    require_packages()
    mkdir(OUTPUT_ROOT)
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    failures: list[str] = []
    skipped: list[str] = []

    for exp in EXPERIMENTS:
        tropics_profiles = AGGREGATE_DIR / exp.aggregate_folder / "profiles.nc"
        if not tropics_profiles.exists() and _shipped_warm_pool_profiles(exp) is None:
            skipped.append(exp.tag)
            print(f"[skip] {exp.tag}: no per-profile file (not distributed in the public "
                  f"archive; available from the author on request)")
            continue

        record: dict = {"experiment": exp, "conditional_dirs": [], "interval_dirs": []}
        try:
            agg_path, screened_path = subset_experiment(exp)
            record["agg_path"] = agg_path
            record["screened_path"] = screened_path

            bench_dir = run_benchmark(exp, agg_path)
            record["benchmark"] = bench_dir

            fig_dir, table_dir = run_benchmark_figures(exp, bench_dir)
            record["benchmark_figures"] = fig_dir
            record["benchmark_tables"] = table_dir

            for comp in exp.comparisons:
                record["conditional_dirs"].append(run_conditional(exp, comp, agg_path))
                record["interval_dirs"].append(run_intervals(exp, comp, agg_path))

            record["weighting_dir"] = run_weighting_analysis(exp, screened_path)
            records.append(record)
        except Exception as exc:
            failures.append(f"{exp.tag}: {exc}")
            traceback.print_exc()

    if records:
        validate_outputs(records, skipped)

    elapsed = (time.time() - t0) / 60.0
    print(f"\n[run] finished in {elapsed:.1f} min")
    if skipped:
        print("[run] skipped for missing per-profile data: " + ", ".join(skipped))
    write_summary()
    if failures:
        print(f"[run] {len(failures)} experiment(s) FAILED:")
        for failure in failures:
            print(f"    - {failure}")
        raise SystemExit(1)
    print("[run] all experiments completed with no failures")


if __name__ == "__main__":
    main()
