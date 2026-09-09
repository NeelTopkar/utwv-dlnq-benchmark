"""Driver: rebuild every warm-pool product from the aggregates.

Subsets to the warm pool and re-runs the benchmark, conditional-table and
bootstrap steps over all six binning experiments.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import xarray as xr


# SETTINGS

# Resolved, because helper scripts run as subprocesses with cwd set to code/;
# a relative root would make every derived path resolve under code/ instead.
ROOT = Path(os.environ.get("UTWV_DATA_ROOT", ".")).resolve()

CODE_DIR = Path(__file__).resolve().parent
AGGREGATE_DIR = ROOT / "data/aggregates/tropical_oceans"

# Everything this driver produces lives under here.
WARMPOOL_ROOT = ROOT / "results/warm_pool"
WARMPOOL_DATASETS_DIR = WARMPOOL_ROOT / "aggregates"
BENCHMARK_OUT_DIR = WARMPOOL_ROOT / "benchmarks"
BENCHMARK_PLOTS_DIR = WARMPOOL_ROOT / "figures" / "benchmarks"
CONDITIONAL_OUT_DIR = WARMPOOL_ROOT / "conditional"
INTERVAL_OUT_DIR = WARMPOOL_ROOT / "bootstrap"
WP_PLOTS_DIR = WARMPOOL_ROOT / "figures" / "warm_pool"
WP_INTERVAL_DIR = WARMPOOL_ROOT / "bootstrap" / "warm_pool_300-303_vs_303-305"

# Warm-pool box (Western Pacific warm pool).
LON_MIN, LON_MAX = 120.0, 180.0
LAT_MIN, LAT_MAX = -15.0, 15.0

# Bootstrap controls.
N_REPLICATES = 1000
SEED = 12345
# The paper's primary tile size; 24 tiles cover the warm pool. Passed through for the
# single-method path only: run_comparison() fixes its two block runs at 10 and 20 deg.
SPATIAL_BLOCK_DEG = 10.0

# Minimum samples per benchmark group (mirrors the tropics runs).
MIN_SAMPLES_PER_GROUP = 10

# Phase toggles.
RUN_PHASE_A = True   # subset the tropical-ocean aggregates to the warm pool
RUN_PHASE_B = True   # benchmark, conditional tables and bootstrap intervals, per experiment
RUN_PHASE_C = True   # warm-pool curves, figures and intervals (warm_pool_*.py)

# Which experiments take part in Phase C bespoke figures (need 300-303 + 0.5K).
PHASE_C_NEEDS = ("sst_297-307_coarse", "sst_295-305_0.5K")


# Each experiment: the source aggregate folder name, a short tag used to name all outputs, and the reference vs warm SST bins to compare. The bin bounds below match the comparisons already produced for the tropics; edit a pair if you want a different contrast for that experiment.
EXPERIMENTS = [
    {
        "aggregate_folder": "sst_297-307_coarse",
        "tag": "sst_297-307_coarse",
        "ref_bounds": (300.0, 303.0),
        "warm_bounds": (303.0, 305.0),   # <-- paper headline comparison
    },
    {
        "aggregate_folder": "sst_295-305_0.5K",
        "tag": "sst_295-305_0.5K",
        "ref_bounds": (302.5, 303.0),
        "warm_bounds": (303.0, 303.5),
    },
    {
        "aggregate_folder": "sst_299-303_0.5K",
        "tag": "sst_299-303_0.5K",
        "ref_bounds": (299.5, 300.0),
        "warm_bounds": (300.0, 300.5),
    },
    {
        "aggregate_folder": "sst_287-307_variable",
        "tag": "sst_287-307_variable",
        "ref_bounds": (302.0, 303.0),
        "warm_bounds": (303.0, 304.0),
    },
    {
        "aggregate_folder": "sst_303-307_0.5K",
        "tag": "sst_303-307_0.5K",
        "ref_bounds": (302.5, 303.0),
        "warm_bounds": (303.0, 303.5),
    },
    {
        "aggregate_folder": "sst_290-299_1K",
        "tag": "sst_290-299_1K",
        "ref_bounds": (297.0, 298.0),
        "warm_bounds": (298.0, 299.0),
    },
]


# LOGGING


# Import the existing pipeline modules

# Import a module from an explicit file path (handles spaces in names).
def _load_module_from_path(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Make conditional_tables.py and bootstrap_intervals.py importable by name.
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

benchmark_mod = _load_module_from_path("benchmark_module", CODE_DIR / "build_benchmarks.py")
import conditional_tables as conditional_mod  # noqa: E402
import bootstrap_intervals as interval_mod              # noqa: E402


# PHASE A: warm-pool subset

_SUBSET_FILES = [("aggregated_gridcells.nc", "sample"),
                 ("profiles.nc", "profile")]


def _subset_one(ds: xr.Dataset, dim: str):
    latc = ds["lat_bin_center"].values
    lonc = ds["lon_bin_center"].values
    mask = (latc >= LAT_MIN) & (latc <= LAT_MAX) & (lonc >= LON_MIN) & (lonc <= LON_MAX)
    idx = np.where(mask)[0]
    return ds.isel({dim: idx}), len(idx), int(ds.sizes[dim])


# Subset one experiment's aggregation outputs to the warm-pool box.
def subset_experiment(exp: dict) -> Path:
    src = AGGREGATE_DIR / exp["aggregate_folder"]
    out_dir = WARMPOOL_DATASETS_DIR / f"{exp['tag']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname, dim in _SUBSET_FILES:
        ip = src / fname
        if not ip.exists():
            continue
        ds = xr.open_dataset(ip)
        sub, kept, total = _subset_one(ds, dim)
        sub.to_netcdf(out_dir / fname)
        ds.close()
        sub.close()

    return out_dir / "aggregated_gridcells.nc"


# PHASE B: benchmark, figures, conditional tables and bootstrap intervals, per experiment

def run_benchmark_data(exp: dict, agg_path: Path) -> Path:
    out_dir = BENCHMARK_OUT_DIR / f"{exp['tag']}"
    cfg = benchmark_mod.BenchmarkConfig(
        input_path=str(agg_path),
        output_dir=str(out_dir),
        min_samples_per_group=MIN_SAMPLES_PER_GROUP,
        overwrite_outputs=True,
    )
    benchmark_mod.run_benchmark(cfg)
    return out_dir


# Drive benchmark_figures.py as an isolated subprocess via env vars.
def run_benchmark_figures(exp: dict, bench_dir: Path) -> None:
    fig_dir = BENCHMARK_PLOTS_DIR / f"{exp['tag']}" / "figures"
    table_dir = BENCHMARK_PLOTS_DIR / f"{exp['tag']}" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"  # headless-safe
    env["BENCHFIG_BENCHMARK_NC"] = str(bench_dir / "benchmark_profiles.nc")
    env["BENCHFIG_SUMMARY_CSV"] = str(bench_dir / "benchmark_summary.csv")
    env["BENCHFIG_COUNTS_CSV"] = str(bench_dir / "counts_summary.csv")
    env["BENCHFIG_FIG_DIR"] = str(fig_dir)
    env["BENCHFIG_TABLE_DIR"] = str(table_dir)

    import subprocess
    script = CODE_DIR / "benchmark_figures.py"
    res = subprocess.run([sys.executable, str(script)], env=env,
                         cwd=str(CODE_DIR))
    if res.returncode != 0:
        raise RuntimeError(f"benchmark_figures.py failed (exit {res.returncode})")


def run_conditional_analysis(exp: dict, agg_path: Path) -> None:
    out_dir = CONDITIONAL_OUT_DIR / f"{exp['tag']}"
    cfg = conditional_mod.ConditionalConfig(
        benchmark_name=f"warmpool_{exp['tag']}",
        input_path=Path(agg_path),
        output_dir=out_dir,
        figures_dir=out_dir / "figures",
        reference_sst_bounds=exp["ref_bounds"],
        warm_sst_bounds=exp["warm_bounds"],
    )
    conditional_mod.run_conditional(cfg)


def run_interval_bootstrap(exp: dict, agg_path: Path) -> None:
    ref = exp["ref_bounds"]
    warm = exp["warm_bounds"]
    cmp_name = (f"{exp['tag']}_{ref[0]:g}-{ref[1]:g}_vs_{warm[0]:g}-{warm[1]:g}")
    out_dir = INTERVAL_OUT_DIR / cmp_name
    cfg = interval_mod.IntervalConfig(
        benchmark_name=f"warmpool_{exp['tag']}",
        input_path=Path(agg_path),
        output_dir=out_dir,
        figures_dir=out_dir / "figures",
        reference_sst_bounds=ref,
        warm_sst_bounds=warm,
        n_reps=N_REPLICATES,
        seed=SEED,
        spatial_block_deg=SPATIAL_BLOCK_DEG,
    )
    interval_mod.run_comparison(cfg)


# PHASE C: bespoke warm-pool paper figures + headline bootstrap

# Run the three _wp_*.py scripts as subprocesses, pointed at results/warm_pool.
def run_phase_c() -> None:
    import subprocess

    agg3 = (WARMPOOL_DATASETS_DIR / "sst_297-307_coarse"
            / "aggregated_gridcells.nc")
    agg05_dir = WARMPOOL_DATASETS_DIR / "sst_295-305_0.5K"

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["WARMPOOL_DATASETS_DIR"] = str(WARMPOOL_DATASETS_DIR)
    env["WP_AGG3_DIR"] = str(agg3.parent)
    env["WP_AGG05_DIR"] = str(agg05_dir)
    env["WP_AGG3_NC"] = str(agg3)
    env["WP_PLOTS_DIR"] = str(WP_PLOTS_DIR)
    env["WP_INTERVAL_DIR"] = str(WP_INTERVAL_DIR)
    # Phase A writes the aggregate and the per-profile file into the same folder, so the
    # profile directory is normally the aggregate directory. When the tropical-ocean
    # profiles.nc it subsets is not distributed, Phase A produces no per-profile file and
    # the reduced warm-pool files shipped with the archive are used instead.
    def _profile_dir(exp_name: str, phase_a_dir: Path) -> Path:
        if any((phase_a_dir / n).exists() for n in ("profiles.nc", "profiles_reduced.nc")):
            return phase_a_dir
        shipped = ROOT / "data/profiles/warm_pool" / exp_name
        if any((shipped / n).exists() for n in ("profiles.nc", "profiles_reduced.nc")):
            return shipped
        return phase_a_dir

    prof3 = _profile_dir("sst_297-307_coarse", agg3.parent)
    prof05 = _profile_dir("sst_295-305_0.5K", agg05_dir)
    env["WARMPOOL_PROFILES_DIR"] = str(prof3.parent)
    env["WP_PROF3_DIR"] = str(prof3)
    env["WP_PROF05_DIR"] = str(prof05)
    WP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for script in ("warm_pool_curves.py", "warm_pool_figures.py", "warm_pool_intervals.py"):
        res = subprocess.run([sys.executable, str(CODE_DIR / script)],
                             env=env, cwd=str(CODE_DIR))
        # A failing helper must not be reported as a successful run.
        if res.returncode != 0:
            failed.append(f"{script} (exit {res.returncode})")
    if failed:
        raise RuntimeError("phase C helper(s) failed: " + ", ".join(failed))


# MAIN

def main() -> None:
    t_start = time.time()
    if not AGGREGATE_DIR.exists():
        raise FileNotFoundError(f"data/aggregates/tropical_oceans not found: {AGGREGATE_DIR}")

    agg_paths: dict[str, Path] = {}
    failures: list[str] = []

    # ---- Phase A: subset ----
    if RUN_PHASE_A:
        for exp in EXPERIMENTS:
            try:
                agg_paths[exp["tag"]] = subset_experiment(exp)
            except Exception as e:
                failures.append(f"A/{exp['tag']}: {e}")
                traceback.print_exc()
    else:
        for exp in EXPERIMENTS:
            agg_paths[exp["tag"]] = (WARMPOOL_DATASETS_DIR
                                     / f"{exp['tag']}"
                                     / "aggregated_gridcells.nc")

    # ---- Phase B: benchmark, conditional tables and intervals, per experiment ----
    if RUN_PHASE_B:
        for exp in EXPERIMENTS:
            tag = exp["tag"]
            agg = agg_paths.get(tag)
            if agg is None or not Path(agg).exists():
                failures.append(f"B/{tag}: missing aggregated file")
                continue

            t0 = time.time()
            bench_dir = BENCHMARK_OUT_DIR / f"{tag}"

            # Run in order; the benchmark figures depend on the benchmark data folder.
            steps = (
                ("benchmark data", lambda: run_benchmark_data(exp, agg)),
                ("benchmark figures", lambda: run_benchmark_figures(exp, bench_dir)),
                ("conditional tables", lambda: run_conditional_analysis(exp, agg)),
                ("bootstrap intervals", lambda: run_interval_bootstrap(exp, agg)),
            )
            for label, fn in steps:
                try:
                    fn()
                except Exception as e:
                    failures.append(f"B/{tag}/{label}: {e}")
                    traceback.print_exc()
            print(f"[run] phase B {tag}: {time.time() - t0:.0f}s", flush=True)

    # ---- Phase C: bespoke warm-pool paper figures ----
    if RUN_PHASE_C:
        missing = [n for n in PHASE_C_NEEDS
                   if not (WARMPOOL_DATASETS_DIR / f"{n}"
                           / "aggregated_gridcells.nc").exists()]
        if missing:
            failures.append(f"C: missing {missing}")
        else:
            try:
                run_phase_c()
            except Exception as e:
                failures.append(f"C: {e}")
                traceback.print_exc()

    print(f"\n[run] finished in {time.time() - t_start:.0f}s")
    if failures:
        print(f"[run] {len(failures)} step(s) FAILED:")
        for f in failures:
            print(f"    - {f}")
        raise SystemExit(1)
    print("[run] all phases completed with no failures")


if __name__ == "__main__":
    main()
