# Processing

The processing module contains the core aggregation pipeline: spatial filtering by morton cell, computing statistics, and writing results to Zarr.

## Granule Reader Protocol

::: magg.processing.GranuleReader

## Pipeline

::: magg.processing.process_morton_cell

## Statistics

::: magg.processing.calculate_cell_statistics

::: magg.processing.AGG_FUNCTIONS

## Zarr I/O

::: magg.processing.write_dataframe_to_zarr
