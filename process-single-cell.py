"""
Profiling script for ICESat-2 ATL06 data processing workflow.

Generates timing reports and flamegraphs for each workflow section.

Usage:
    uv run process-single-cell.py

Outputs:
    - Console timing summary for each workflow section
    - flamegraph_*.html files for each profiled section
    - flamegraph_full.html for the complete workflow
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import earthaccess
import h5coro
import numpy as np
import pandas as pd
from h5coro import s3driver
from mortie import clip2order, generate_morton_children, geo2mort, mort2healpix
from pyinstrument import Profiler


@dataclass
class TimingResult:
    """Container for section timing results."""

    name: str
    duration_s: float
    details: dict[str, Any] = field(default_factory=dict)


class SectionProfiler:
    """Context manager for profiling code sections with timing and flamegraphs."""

    def __init__(self, name: str, output_dir: Path | None = None):
        self.name = name
        self.output_dir = output_dir or Path(".")
        self.profiler = Profiler()
        self.start_time: float = 0
        self.duration: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.profiler.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop()
        self.duration = time.perf_counter() - self.start_time

        flamegraph_path = self.output_dir / f"flamegraph_{self.name}.html"
        with open(flamegraph_path, "w") as f:
            f.write(self.profiler.output_html())

        print(f"  [{self.name}] {self.duration:.3f}s - flamegraph: {flamegraph_path}")
        return False


def calculate_cell_statistics(
    df_cell: pd.DataFrame, value_col: str = "h_li", sigma_col: str = "s_li"
) -> dict:
    """Calculate summary statistics for a cell."""
    if len(df_cell) == 0:
        return {
            "count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean_weighted": np.nan,
            "sigma_mean": np.nan,
            "variance": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
        }

    values = df_cell[value_col].values
    sigmas = df_cell[sigma_col].values

    q = np.quantile(values, [0.25, 0.5, 0.75])
    weights = 1.0 / (sigmas**2)
    weighted_mean = np.sum(values * weights) / np.sum(weights)
    sigma_mean = 1.0 / np.sqrt(np.sum(weights))

    return {
        "count": len(df_cell),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "variance": float(np.var(values)),
        "q25": float(q[0]),
        "q50": float(q[1]),
        "q75": float(q[2]),
        "mean_weighted": float(weighted_mean),
        "sigma_mean": float(sigma_mean),
    }


def process_morton_cell_profiled(
    parent_morton: int,
    parent_order: int,
    child_order: int,
    granule_urls: list[str],
    s3_credentials: dict,
    output_dir: Path,
) -> tuple[dict[str, Any], list[TimingResult]]:
    """
    Process one parent morton cell with profiling for each section.

    Parameters
    ----------
    parent_morton : int
        Morton index of parent cell
    parent_order : int
        Order of parent morton cell (e.g., 6 or 7)
    child_order : int
        Order of child cells for statistics (typically 12)
    granule_urls : list
        List of S3 URLs to process (from pre-built catalog)
    s3_credentials : dict
        AWS S3 credentials for NSIDC access
    output_dir : Path
        Directory for flamegraph outputs

    Returns
    -------
    tuple
        (result dict, list of TimingResults)
    """
    timings: list[TimingResult] = []
    start_time = datetime.now()

    print(f"\nProcessing morton cell: {parent_morton}")
    print(f"  Granules: {len(granule_urls)}")

    if not granule_urls:
        return {
            "parent_morton": parent_morton,
            "cells_with_data": 0,
            "total_obs": 0,
            "parquet_path": None,
            "error": "No granules found",
        }, timings

    # ========================================================================
    # SECTION 1: READ FILES FROM S3 WITH SPATIAL SUBSETTING
    # ========================================================================

    print("\n[1/5] Reading files from S3...")

    credentials = {
        "aws_access_key_id": s3_credentials["accessKeyId"],
        "aws_secret_access_key": s3_credentials["secretAccessKey"],
        "aws_session_token": s3_credentials["sessionToken"],
    }

    all_dataframes = []
    files_processed = 0

    with SectionProfiler("s3_reading", output_dir) as profiler:
        for s3_url in granule_urls:
            try:
                resource_path = s3_url.replace("s3://", "")

                h5obj = h5coro.H5Coro(
                    resource_path,
                    s3driver.S3Driver,
                    credentials=credentials,
                    errorChecking=True,
                    verbose=False,
                )

                for g in ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]:
                    try:
                        coord_data = h5obj.readDatasets(
                            [
                                f"/{g}/land_ice_segments/latitude",
                                f"/{g}/land_ice_segments/longitude",
                            ]
                        )

                        lats = coord_data[f"/{g}/land_ice_segments/latitude"]
                        lons = coord_data[f"/{g}/land_ice_segments/longitude"]

                        if len(lats) == 0:
                            continue

                        midx18 = geo2mort(lats, lons, order=18)
                        midx_parent = clip2order(parent_order, midx18)
                        mask_spatial = midx_parent == parent_morton

                        if np.sum(mask_spatial) == 0:
                            continue

                        indices = np.where(mask_spatial)[0]
                        min_idx = int(indices[0])
                        max_idx = int(indices[-1]) + 1

                        data = h5obj.readDatasets(
                            [
                                {
                                    "dataset": f"/{g}/land_ice_segments/h_li",
                                    "hyperslice": [(min_idx, max_idx)],
                                },
                                {
                                    "dataset": f"/{g}/land_ice_segments/h_li_sigma",
                                    "hyperslice": [(min_idx, max_idx)],
                                },
                                {
                                    "dataset": f"/{g}/land_ice_segments/atl06_quality_summary",
                                    "hyperslice": [(min_idx, max_idx)],
                                },
                            ]
                        )

                        mask_sliced = mask_spatial[min_idx:max_idx]
                        h_li = data[f"/{g}/land_ice_segments/h_li"][mask_sliced]
                        s_li = data[f"/{g}/land_ice_segments/h_li_sigma"][mask_sliced]
                        q_flag = data[f"/{g}/land_ice_segments/atl06_quality_summary"][
                            mask_sliced
                        ]

                        quality_mask = q_flag == 0

                        if np.sum(quality_mask) == 0:
                            continue

                        midx_sliced = midx18[min_idx:max_idx][mask_sliced]
                        data_dict = {
                            "h_li": h_li[quality_mask],
                            "s_li": s_li[quality_mask],
                            "midx": midx_sliced[quality_mask],
                        }
                        all_dataframes.append(pd.DataFrame(data_dict))

                    except Exception:
                        continue

                files_processed += 1

            except Exception as e:
                print(f"  Warning: Error processing {s3_url}: {e}")
                continue

    timings.append(
        TimingResult(
            name="s3_reading",
            duration_s=profiler.duration,
            details={"files_processed": files_processed, "granules": len(granule_urls)},
        )
    )

    if not all_dataframes:
        return {
            "parent_morton": parent_morton,
            "cells_with_data": 0,
            "total_obs": 0,
            "parquet_path": None,
            "error": "No data after filtering",
        }, timings

    # ========================================================================
    # SECTION 2: CONCATENATE DATAFRAMES
    # ========================================================================

    print("\n[2/5] Concatenating dataframes...")

    with SectionProfiler("concatenation", output_dir) as profiler:
        df_all = pd.concat(all_dataframes, ignore_index=True)

    timings.append(
        TimingResult(
            name="concatenation",
            duration_s=profiler.duration,
            details={
                "total_observations": len(df_all),
                "dataframes": len(all_dataframes),
            },
        )
    )

    print(f"  Total observations: {len(df_all):,}")

    # ========================================================================
    # SECTION 3: CALCULATE STATISTICS
    # ========================================================================

    print("\n[3/5] Calculating statistics...")

    with SectionProfiler("statistics", output_dir) as profiler:
        children = generate_morton_children(parent_morton, child_order)
        df_all["m12"] = clip2order(child_order, df_all["midx"].values)

        n_cells = len(children)
        stats_arrays = {
            "count": np.zeros(n_cells, dtype=np.int32),
            "min": np.full(n_cells, np.nan, dtype=np.float32),
            "max": np.full(n_cells, np.nan, dtype=np.float32),
            "mean_weighted": np.full(n_cells, np.nan, dtype=np.float32),
            "sigma_mean": np.full(n_cells, np.nan, dtype=np.float32),
            "variance": np.full(n_cells, np.nan, dtype=np.float32),
            "q25": np.full(n_cells, np.nan, dtype=np.float32),
            "q50": np.full(n_cells, np.nan, dtype=np.float32),
            "q75": np.full(n_cells, np.nan, dtype=np.float32),
        }

        cells_with_data = 0
        for i, child_morton in enumerate(children):
            df_cell = df_all[df_all["m12"] == child_morton]
            if len(df_cell) > 0:
                cells_with_data += 1
            stats = calculate_cell_statistics(
                df_cell, value_col="h_li", sigma_col="s_li"
            )
            for key, value in stats.items():
                stats_arrays[key][i] = value

    timings.append(
        TimingResult(
            name="statistics",
            duration_s=profiler.duration,
            details={"n_cells": n_cells, "cells_with_data": cells_with_data},
        )
    )

    # ========================================================================
    # SECTION 4: HEALPIX CONVERSION AND DATAFRAME CREATION
    # ========================================================================

    print("\n[4/5] Creating output dataframe...")

    with SectionProfiler("dataframe_creation", output_dir) as profiler:
        child_cell_ids, _ = mort2healpix(children)

        df_out = pd.DataFrame(
            {
                "child_morton": children,
                "child_healpix": child_cell_ids,
                "count": stats_arrays["count"],
                "h_mean": stats_arrays["mean_weighted"],
                "h_sigma": stats_arrays["sigma_mean"],
                "h_min": stats_arrays["min"],
                "h_max": stats_arrays["max"],
                "h_variance": stats_arrays["variance"],
                "h_q25": stats_arrays["q25"],
                "h_q50": stats_arrays["q50"],
                "h_q75": stats_arrays["q75"],
            }
        )

    timings.append(
        TimingResult(
            name="dataframe_creation",
            duration_s=profiler.duration,
            details={"n_rows": len(df_out)},
        )
    )

    # ========================================================================
    # SECTION 5: WRITE PARQUET TO S3
    # ========================================================================

    print("\n[5/5] Writing parquet...")

    scratch = os.environ["SCRATCH_BUCKET"]
    parquet_path = f"{scratch}/{parent_morton}.parquet"

    with SectionProfiler("parquet_write", output_dir) as profiler:
        df_out.to_parquet(parquet_path, index=False, engine="pyarrow")

    timings.append(
        TimingResult(
            name="parquet_write",
            duration_s=profiler.duration,
            details={"path": str(parquet_path)},
        )
    )

    duration = (datetime.now() - start_time).total_seconds()

    return {
        "parent_morton": parent_morton,
        "cells_with_data": cells_with_data,
        "total_obs": int(stats_arrays["count"].sum()),
        "parquet_path": str(parquet_path),
        "error": None,
        "duration_s": duration,
        "files_processed": files_processed,
    }, timings


def print_timing_summary(timings: list[TimingResult]) -> None:
    """Print a summary table of timing results."""
    total = sum(t.duration_s for t in timings)

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"{'Section':<25} {'Time (s)':<12} {'% Total':<10} Details")
    print("-" * 70)

    for t in timings:
        pct = (t.duration_s / total) * 100 if total > 0 else 0
        details_str = ""
        if t.details:
            details_str = ", ".join(
                f"{k}={v}" for k, v in list(t.details.items())[:3]
            )
        print(f"{t.name:<25} {t.duration_s:<12.3f} {pct:<10.1f} {details_str}")

    print("-" * 70)
    print(f"{'TOTAL':<25} {total:<12.3f} {'100.0':<10}")
    print("=" * 70)


def load_catalog(catalog_path: str) -> dict:
    """Load granule catalog from JSON file."""
    import json

    with open(catalog_path) as f:
        data = json.load(f)
    return data


def main():
    """Main entry point for profiling."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile ICESat-2 processing workflow")
    parser.add_argument(
        "--catalog",
        default="data/granule_catalog_cycle22_order6.json",
        help="Path to granule catalog JSON",
    )
    parser.add_argument(
        "--cell-index",
        type=int,
        default=0,
        help="Index of cell to profile (default: first cell)",
    )
    parser.add_argument(
        "--child-order",
        type=int,
        default=12,
        help="Child cell order for statistics",
    )
    args = parser.parse_args()

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ICESat-2 Processing Profiler")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Started at: {datetime.now().isoformat()}")

    # Authenticate with earthaccess
    print("\nAuthenticating with NASA Earthdata...")
    auth = earthaccess.login()
    credentials = auth.get_s3_credentials(daac="NSIDC")

    s3_credentials = {
        "accessKeyId": credentials["accessKeyId"],
        "secretAccessKey": credentials["secretAccessKey"],
        "sessionToken": credentials["sessionToken"],
    }
    print("  Authentication successful")

    # Load catalog
    print(f"\nLoading catalog from {args.catalog}...")
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise(RuntimeError, f"Catalog not found at {catalog_path}")

    catalog_data = load_catalog(args.catalog)
    metadata = catalog_data["metadata"]
    catalog = catalog_data["catalog"]

    parent_order = metadata["parent_order"]
    print(f"  Cycle: {metadata['cycle']}")
    print(f"  Parent order: {parent_order}")
    print(f"  Total cells: {metadata['total_cells']}")
    print(f"  Total granules: {metadata['total_granules']}")

    # Select cell for profiling
    all_cells = list(catalog.keys())
    if args.cell_index >= len(all_cells):
        print(f"  Error: cell-index {args.cell_index} out of range (max {len(all_cells) - 1})")
        return

    sample_morton = int(all_cells[args.cell_index])
    granule_urls = catalog[all_cells[args.cell_index]]

    print(f"\nProfiling morton cell: {sample_morton}")
    print(f"  Granules to process: {len(granule_urls)}")

    # Run profiled processing
    result, timings = process_morton_cell_profiled(
        parent_morton=sample_morton,
        parent_order=parent_order,
        child_order=args.child_order,
        granule_urls=granule_urls,
        s3_credentials=s3_credentials,
        output_dir=output_dir,
    )

    # Print timing summary
    print_timing_summary(timings)

    # Print result summary
    print("\nPROCESSING RESULT")
    print("-" * 70)
    for key, value in result.items():
        print(f"  {key}: {value}")

    print(f"\nCompleted at: {datetime.now().isoformat()}")
    print(f"\nFlamegraph files written to: {output_dir.absolute()}")
    print("Open the HTML files in a browser to view interactive flamegraphs.")


if __name__ == "__main__":
    main()
