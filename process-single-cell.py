"""
Profiling script for ICESat-2 ATL06 data processing workflow.

Usage:
    uv run process-single-cell.py --profile time   # CPU time profiling (pyinstrument)
    uv run process-single-cell.py --profile memory # Memory profiling (memray)

Outputs:
    - flamegraph.html (time mode) or memray.bin + flamegraph.html (memory mode)
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import earthaccess
import h5coro
import numpy as np
import pandas as pd
from h5coro import s3driver
from mortie import clip2order, generate_morton_children, geo2mort, mort2healpix


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


def process_morton_cell(
    parent_morton: int,
    parent_order: int,
    child_order: int,
    granule_urls: list[str],
    s3_credentials: dict,
) -> dict[str, Any]:
    """
    Process one parent morton cell: read from S3, calculate stats.

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

    Returns
    -------
    dict
        Processing result
    """
    start_time = datetime.now()

    print(f"\nProcessing morton cell: {parent_morton}")
    print(f"  Granules: {len(granule_urls)}")

    if not granule_urls:
        return {
            "parent_morton": parent_morton,
            "cells_with_data": 0,
            "total_obs": 0,
            "error": "No granules found",
        }

    # Prepare credentials for h5coro S3Driver
    credentials = {
        "aws_access_key_id": s3_credentials["accessKeyId"],
        "aws_secret_access_key": s3_credentials["secretAccessKey"],
        "aws_session_token": s3_credentials["sessionToken"],
    }

    all_dataframes = []
    files_processed = 0

    print("\n  Reading files from S3...")
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

    print(f"  Processed {files_processed}/{len(granule_urls)} files")

    if not all_dataframes:
        return {
            "parent_morton": parent_morton,
            "cells_with_data": 0,
            "total_obs": 0,
            "error": "No data after filtering",
        }

    print("\n  Concatenating dataframes...")
    df_all = pd.concat(all_dataframes, ignore_index=True)
    print(f"  Total observations: {len(df_all):,}")

    print("\n  Calculating statistics...")
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
        stats = calculate_cell_statistics(df_cell, value_col="h_li", sigma_col="s_li")
        for key, value in stats.items():
            stats_arrays[key][i] = value

    print(f"  Cells with data: {cells_with_data}/{n_cells}")

    print("\n  Creating output dataframe...")
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

    print("\n  Writing parquet...")
    scratch = os.environ["SCRATCH_BUCKET"]
    parquet_path = f"{scratch}/{parent_morton}.parquet"
    df_out.to_parquet(parquet_path, index=False, engine="pyarrow")

    duration = (datetime.now() - start_time).total_seconds()

    return {
        "parent_morton": parent_morton,
        "cells_with_data": cells_with_data,
        "total_obs": int(stats_arrays["count"].sum()),
        "parquet_path": parquet_path,
        "error": None,
        "duration_s": duration,
        "files_processed": files_processed,
    }


def load_catalog(catalog_path: str) -> dict:
    """Load granule catalog from JSON file."""
    import json

    with open(catalog_path) as f:
        data = json.load(f)
    return data


def run_with_profiling(
    func,
    args: tuple,
    kwargs: dict,
    output_dir: Path,
    mode: Literal["time", "memory"],
    cell_id: int,
) -> Any:
    """Run a function with profiling enabled."""
    if mode == "time":
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()

        flamegraph_path = output_dir / f"pyinstrument_flamegraph_{cell_id}.html"
        with open(flamegraph_path, "w") as f:
            f.write(profiler.output_html())
        print(f"\nFlamegraph written to: {flamegraph_path}")

    else:  # memory
        import memray

        bin_path = output_dir / f"memray_{cell_id}.bin"
        with memray.Tracker(bin_path):
            result = func(*args, **kwargs)

        flamegraph_path = output_dir / f"memray_flamegraph_{cell_id}.html"
        subprocess.run(
            ["memray", "flamegraph", "-o", str(flamegraph_path), str(bin_path)],
            capture_output=True,
        )
        print(f"\nMemray data written to: {bin_path}")
        print(f"Flamegraph written to: {flamegraph_path}")

    return result


def main():
    """Main entry point for profiling."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile ICESat-2 processing workflow")
    parser.add_argument(
        "--catalog",
        default="data/granule_catalog_cycle22_order6.json",
        help="Path to granule catalog JSON",
    )
    # Default to cell index 127 (cell -6111121) which has 395 granules,
    # the most of any cell in the catalog for worst-case profiling
    parser.add_argument(
        "--cell-index",
        type=int,
        default=127,
        help="Index of cell to profile (default: 127, cell with most granules)",
    )
    parser.add_argument(
        "--child-order",
        type=int,
        default=12,
        help="Child cell order for statistics",
    )
    parser.add_argument(
        "--profile",
        choices=["time", "memory"],
        default="time",
        help="Profile mode: 'time' (pyinstrument) or 'memory' (memray)",
    )
    args = parser.parse_args()

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ICESat-2 Processing Profiler")
    print("=" * 70)
    print(f"Profile mode: {args.profile}")
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
        raise RuntimeError(f"Catalog not found at {catalog_path}")

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
        raise RuntimeError(
            f"cell-index {args.cell_index} out of range (max {len(all_cells) - 1})"
        )

    sample_morton = int(all_cells[args.cell_index])
    granule_urls = catalog[all_cells[args.cell_index]]

    print(f"\nProfiling morton cell: {sample_morton}")
    print(f"  Granules to process: {len(granule_urls)}")

    # Run with profiling
    start_time = time.perf_counter()

    result = run_with_profiling(
        process_morton_cell,
        args=(sample_morton, parent_order, args.child_order, granule_urls, s3_credentials),
        kwargs={},
        output_dir=output_dir,
        mode=args.profile,
        cell_id=sample_morton,
    )

    total_time = time.perf_counter() - start_time

    # Print result summary
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
