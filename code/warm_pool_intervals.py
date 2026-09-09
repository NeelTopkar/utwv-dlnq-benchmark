"""Warm-pool bootstrap intervals under three resampling units.

Re-runs the bootstrap intervals for the warm pool comparing individual aggregates,
10 degree spatial tiles and 20 degree spatial tiles.
"""
import os
from pathlib import Path
import bootstrap_intervals as intervals

# Resolve all paths against a single data root.
ROOT = os.environ.get("UTWV_DATA_ROOT", ".")

# Reads the warm-pool subset; both input and output are overridable so the driver can repoint them.
INPUT = os.environ.get(
    "WP_AGG3_NC",
    os.path.join(ROOT, "data/aggregates/warm_pool/sst_297-307_coarse/aggregated_gridcells.nc"),
)
OUT = Path(os.environ.get(
    "WP_INTERVAL_DIR",
    os.path.join(ROOT, "results/bootstrap/warm_pool_300-303_vs_303-305"),
))

def main() -> None:
    cfg = intervals.IntervalConfig(
        benchmark_name="warmpool_300_303_vs_303_305",
        input_path=Path(INPUT),
        output_dir=OUT,
        figures_dir=OUT / "figures",
        reference_sst_bounds=(300.0, 303.0),
        warm_sst_bounds=(303.0, 305.0),
        n_reps=1000,
        seed=12345,
        spatial_block_deg=10.0,
    )
    intervals.run_comparison(cfg)


if __name__ == "__main__":
    main()
