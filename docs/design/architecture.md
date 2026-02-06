# Architecture

## Core Axioms

1. **Data selection is declarative using STAC** --- query interfaces, not file paths
2. **Aggregation doesn't duplicate the at-rest data** --- we fetch files for aggregation but discard source data after processing

## The Challenge: Sparse Point Data

When casting point data (ATL06, OPR, etc) to a grid, we encounter two issues:

1. **Too dense** --- multiple points per cell
2. **Too sparse** --- no points for many cells

The sparsity is annoying but workable. The density is the real problem: xarray doesn't handle collisions natively, so we must define what to do with overlapping observations. Hence the "agg" in magg.

## Prior Art

Previously at the ICESat-2 project science office, we tackled this using hierarchical indexing: resharding ATL06 using healpix-based morton indexing to files in hive format. This kept data in columnar format (rasterized on-the-fly via vaex), but violated axiom #2 by requiring duplicated storage.

That approach also required high-memory nodes because we built the spatial tree root-to-leaves.

## Innovation: Building Leaves-to-Root

We invert the tree construction order, which:

- Enables parallel processing with small, commodity workers
- Avoids high-memory node requirements
- Allows on-the-fly processing without data duplication
- Outputs significantly compacted results via aggregation

## Vocabulary

- **Base Aggregation Cell**: Finest resolution of aggregation (leaf nodes). For ICESat-2: order 12 (~1.5km cells)
- **Shard**: Lowest level of chunking (not divisible). Contains explicit links to underlying raw granules. For our implementation: order 6 (~100km, containing 4096 base cells)

## Spatial Indexing

Uses HEALPix nested indexing via morton indices:

| Level | Order | Resolution | Purpose |
|-------|-------|------------|---------|
| Parent/Shard | 6 | ~100km | Processing unit, defines what granules to read |
| Child/Base Cell | 12 | ~1.5km | Output resolution, matches ICESat-2 beam pair spacing |

Target coverage: 1,872 cells covering Antarctic grounded ice drainage basins.

## End-to-End Flow

```
                        ┌───────────────────────────────┐
                        │   ICESat-2 ATL06 on S3        │
                        │   (NSIDC DAAC, ~2,000 HDF5    │
                        │    granules per cycle)         │
                        └───────────────┬───────────────┘
                                        │
            ┌───────────────────────────┐│┌───────────────────────────┐
            │                           │││                           │
            ▼                           ▼│▼                           │
┌───────────────────────┐  ┌────────────┴──────────────┐             │
│ 1. BUILD CATALOG      │  │ 2. AUTHENTICATE           │             │
│                       │  │                           │             │
│ catalog.py            │  │ auth.py                   │             │
│                       │  │                           │             │
│ Query CMR for cycle   │  │ earthaccess.login()       │             │
│         │             │  │         │                 │             │
│         ▼             │  │         ▼                 │             │
│ Extract S3 URLs +     │  │ Temporary S3 creds        │             │
│ polygon geometry      │  │ (valid ~1 hour)           │             │
│         │             │  └─────────────┬─────────────┘             │
│         ▼             │                │                           │
│ Densify polygons      │                │                           │
│ (pyproj EPSG:3031)    │                │                           │
│         │             │                │                           │
│         ▼             │                │                           │
│ geo2mort → clip2order │                │                           │
│ Map parent cells to   │                │                           │
│ granule S3 URLs       │                │                           │
│         │             │                │                           │
│         ▼             │                │                           │
│ catalog.json          │                │                           │
│ {morton: [urls]}      │                │                           │
└───────────┬───────────┘                │                           │
            │                            │                           │
            ▼                            ▼                           │
┌──────────────────────────────────────────────────────┐             │
│ 3. CREATE ZARR TEMPLATE                              │             │
│                                                      │             │
│ schema.py → xdggs_zarr_template()                    │             │
│                                                      │             │
│ CellStatsSchema metadata ──▶ GroupSpec ──▶ Zarr v3   │             │
│                                                      │             │
│ Shape:  12 × 4^child_order  (786,432 cells at O12)   │             │
│ Chunks: 4^(child - parent)  (4,096 cells at O12-O6)  │             │
│ Arrays: cell_ids, morton, count, h_min, h_max,       │             │
│         h_mean, h_sigma, h_variance, h_q25-75        │             │
│                                                      │             │
│ Written to: s3://bucket/prefix/12/                   │             │
└──────────────────────────┬───────────────────────────┘             │
                           │                                         │
                           ▼                                         │
┌──────────────────────────────────────────────────────────────────┐ │
│ 4. PARALLEL EXECUTION                                            │ │
│                                                                  │ │
│ invoke_lambda.py (orchestrator)                                  │ │
│ ThreadPoolExecutor(max_workers=1700)                             │ │
│                                                                  │ │
│ For each parent morton cell in catalog:                          │ │
│ ┌──────────────────────────────────────────────────────────────┐ │ │
│ │                    AWS Lambda Worker                         │ │ │
│ │                    (ARM64, 2GB, 15min)                       │ │ │
│ │                                                              │ │ │
│ │  lambda_handler.py                                           │ │ │
│ │       │                                                      │ │ │
│ │       ▼                                                      │ │ │
│ │  ┌──────────────────────────────────────────────────────┐   │ │ │
│ │  │  process_morton_cell()           processing.py       │   │◄┘ │
│ │  │                                                      │   │   │
│ │  │  For each granule URL:                               │   │   │
│ │  │    For each ground track (gt1l..gt3r):               │   │   │
│ │  │    ┌─────────────────────────────────────────────┐   │   │   │
│ │  │    │ READ: lat, lon via h5coro (S3 byte-range)  │◄──┼───┘
│ │  │    │                                             │   │
│ │  │    │ FILTER: geo2mort(lat,lon,O18)               │   │
│ │  │    │         clip2order(parent) == parent_morton  │   │
│ │  │    │                                             │   │
│ │  │    │ READ: h_li, h_li_sigma, quality_summary     │   │
│ │  │    │       (hyperslice on bounding indices)      │   │
│ │  │    │                                             │   │
│ │  │    │ QUALITY: keep only quality_summary == 0     │   │
│ │  │    │                                             │   │
│ │  │    │ OUTPUT: DataFrame(h_li, s_li, midx)         │   │
│ │  │    └─────────────────────────────────────────────┘   │
│ │  │                                                      │
│ │  │  Concatenate all track DataFrames                    │
│ │  │       │                                              │
│ │  │       ▼                                              │
│ │  │  clip2order(child_order=12, midx_18)                 │
│ │  │  generate_morton_children(parent, child_order)        │
│ │  │       │                                              │
│ │  │       ▼                                              │
│ │  │  For each of 4,096 child cells:                      │
│ │  │  ┌───────────────────────────────────────────────┐   │
│ │  │  │ calculate_cell_statistics()                   │   │
│ │  │  │                                               │   │
│ │  │  │ Schema-driven dispatch via AGG_FUNCTIONS:     │   │
│ │  │  │   count      → len(values)                    │   │
│ │  │  │   nanmin     → np.min(h_li)                   │   │
│ │  │  │   nanmax     → np.max(h_li)                   │   │
│ │  │  │   nanvar     → np.var(h_li)                   │   │
│ │  │  │   weighted   → Σ(h_li/σ²) / Σ(1/σ²)         │   │
│ │  │  │     _mean      using s_li as weights          │   │
│ │  │  │   weighted   → 1/√Σ(1/σ²)                    │   │
│ │  │  │     _sigma                                    │   │
│ │  │  │   quantile   → np.quantile(h_li, q)          │   │
│ │  │  │               for q ∈ {0.25, 0.50, 0.75}     │   │
│ │  │  └───────────────────────────────────────────────┘   │
│ │  │       │                                              │
│ │  │       ▼                                              │
│ │  │  mort2healpix(children) → cell_ids                   │
│ │  │  Assemble output DataFrame (11 columns × 4,096 rows) │
│ │  └──────────────────────────┬───────────────────────────┘
│ │                             │
│ │                             ▼
│ │  ┌──────────────────────────────────────────────────────┐
│ │  │  write_dataframe_to_zarr()         processing.py    │
│ │  │                                                      │
│ │  │  For each column:                                    │
│ │  │    open_array(store, "{child_order}/{col}")           │
│ │  │    array.set_block_selection(chunk_idx, values)       │
│ │  │                                                      │
│ │  │  One chunk per parent cell → concurrent-write-safe   │
│ │  └──────────────────────────────────────────────────────┘
│ └──────────────────────────────────────────────────────────┘
│                                                                  │
│ Results collected, retried on transient failures                 │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│ 5. CONSOLIDATE + ANALYZE                             │
│                                                      │
│ zarr.consolidate_metadata(store)                     │
│                                                      │
│ Output Zarr store:                                   │
│   s3://bucket/prefix/                                │
│   └── 12/                                            │
│       ├── cell_ids   (uint64, fill=0)                │
│       ├── morton     (int64,  fill=0)                │
│       ├── count      (int32,  fill=0)                │
│       ├── h_min      (float32, fill=NaN)             │
│       ├── h_max      (float32, fill=NaN)             │
│       ├── h_mean     (float32, fill=NaN)             │
│       ├── h_sigma    (float32, fill=NaN)             │
│       ├── h_variance (float32, fill=NaN)             │
│       ├── h_q25      (float32, fill=NaN)             │
│       ├── h_q50      (float32, fill=NaN)             │
│       └── h_q75      (float32, fill=NaN)             │
│                                                      │
│ Open with xarray + xdggs for visualization           │
└──────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
              ┌──────────────┐
              │  schema.py   │  Single source of truth
              │              │  CellStatsSchema, xdggs_zarr_template
              └──────┬───────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
  ┌──────────┐ ┌──────────┐ ┌───────────────────┐
  │ auth.py  │ │processing│ │ catalog.py        │
  │          │ │  .py     │ │                   │
  │ S3 creds │ │ AGG_FUNC │ │ CMR query,        │
  │          │ │ read/agg │ │ morton mapping     │
  │          │ │ write    │ │                   │
  └────┬─────┘ └────┬─────┘ └─────────┬─────────┘
       │             │                 │
       └──────┬──────┘                 │
              │                        │
              ▼                        │
  ┌───────────────────────┐            │
  │ lambda_handler.py     │            │
  │ (AWS-specific wrapper)│            │
  └───────────┬───────────┘            │
              │                        │
              ▼                        ▼
  ┌────────────────────────────────────────┐
  │ invoke_lambda.py                       │
  │ (orchestrator: catalog + auth +        │
  │  template + parallel Lambda dispatch)  │
  └────────────────────────────────────────┘
```

## Proposed: GranuleReader Protocol

!!! note "Strawman Proposal"
    This section describes a proposed refactoring, not the current implementation.
    The protocol and ATL06Reader skeleton exist in the codebase for discussion.
    See [`GranuleReader`][magg.processing.GranuleReader] and
    [`ATL06Reader`][magg.atl06.ATL06Reader] API docs.

`process_morton_cell` currently mixes two concerns: **data access** (ATL06 HDF5 structure, ground tracks, quality flags, h5coro credentials) and **generic aggregation** (spatial filtering, child cell grouping, schema-driven statistics). The [`GranuleReader`][magg.processing.GranuleReader] protocol separates these so the aggregation pipeline can be reused across datasets.

### Protocol boundary

```
process_morton_cell(reader, parent_morton, ...)
│
│  Generic: morton spatial filtering + aggregation
│  (processing.py --- unchanged across datasets)
│
│   For each granule_url:
│   ┌─────────────────────────────────────────────────────────┐
│   │  PHASE 1: Coordinates              ◄── reader protocol  │
│   │  groups = reader.read_coordinates(url)                   │
│   │  Returns [(lats, lons), ...] per observation group       │
│   └────────────────────────┬────────────────────────────────┘
│                            │
│   ┌────────────────────────▼────────────────────────────────┐
│   │  SPATIAL FILTER                     ◄── generic          │
│   │  For each group:                                         │
│   │    midx18 = geo2mort(lats, lons, order=18)               │
│   │    mask = clip2order(parent_order, midx18) == parent      │
│   │    row_slice = bounding range of mask                     │
│   └────────────────────────┬────────────────────────────────┘
│                            │
│   ┌────────────────────────▼────────────────────────────────┐
│   │  PHASE 2: Data + quality filter     ◄── reader protocol  │
│   │  df = reader.read_data(url, group_idx, row_slice, midx)  │
│   │  Returns DataFrame(value_col, weight_col, midx)          │
│   │  or None if nothing passes quality filter                │
│   └────────────────────────┬────────────────────────────────┘
│                            │
│   Concatenate all DataFrames
│                            │
│   ┌────────────────────────▼────────────────────────────────┐
│   │  GROUP + AGGREGATE                  ◄── generic          │
│   │  clip2order(child_order, midx_18)                        │
│   │  For each child cell:                                    │
│   │    calculate_cell_statistics()                            │
│   │    (driven by CellStatsSchema metadata)                  │
│   └────────────────────────┬────────────────────────────────┘
│                            │
│   ┌────────────────────────▼────────────────────────────────┐
│   │  OUTPUT                             ◄── generic          │
│   │  mort2healpix(children) → cell_ids                       │
│   │  DataFrame(cell_ids, morton, count, h_min, ...)          │
│   └─────────────────────────────────────────────────────────┘
```

### What changes per dataset

| Concern | Where it lives | ATL06 | A hypothetical ATL03 |
|---|---|---|---|
| **File format** | `GranuleReader` impl | HDF5 via h5coro | HDF5 via h5coro |
| **Observation groups** | `read_coordinates` | 6 ground tracks | 6 ground tracks |
| **Dataset paths** | `read_coordinates`, `read_data` | `land_ice_segments/h_li` | `heights/h_ph` |
| **Quality logic** | `read_data` | `quality_summary == 0` | `signal_conf_ph >= 3` |
| **Value/weight cols** | `read_data` return cols | `h_li`, `s_li` | `h_ph`, (none) |
| **Aggregation recipe** | `CellStatsSchema` | weighted mean + quantiles | count + quantiles |
| **Spatial indexing** | `process_morton_cell` | unchanged | unchanged |
| **Zarr output** | `write_dataframe_to_zarr` | unchanged | unchanged |

### What stays generic

Everything below the reader protocol is dataset-agnostic:

- **Spatial filtering** --- `geo2mort`, `clip2order`, bounding-box hyperslice optimization
- **Child cell grouping** --- `generate_morton_children`, group-by on clipped morton index
- **Aggregation dispatch** --- `CellStatsSchema` metadata drives `AGG_FUNCTIONS`
- **Zarr I/O** --- template creation from schema, chunk-aligned writes
- **Orchestration** --- catalog loading, credential management, parallel Lambda invocation

### Proposed process_morton_cell signature

```python
def process_morton_cell(
    reader: GranuleReader,    # ← injected, replaces h5coro_driver + credentials
    parent_morton: int,
    parent_order: int,
    child_order: int,
    granule_urls: list[str],
) -> tuple[pd.DataFrame, ProcessingMetadata]:
    ...
```

The caller (Lambda handler or local script) constructs the appropriate reader and passes it in:

```python
# ATL06 on AWS Lambda
reader = ATL06Reader(s3_credentials)
df, meta = process_morton_cell(reader, parent_morton, ...)

# Hypothetical ATL03
reader = ATL03Reader(s3_credentials)
df, meta = process_morton_cell(reader, parent_morton, ...)
```

## Optional: obspec-utils for Store Composition

!!! note "Optional Enhancement"
    This section describes how [obspec-utils](https://github.com/virtual-zarr/obspec-utils)
    could improve observability and I/O performance. It does not require changes to the
    aggregation pipeline itself.

The current pipeline uses h5coro's built-in S3 driver for byte-range access to HDF5 files. obspec-utils provides composable, protocol-based store wrappers that could sit *underneath* any `GranuleReader` implementation, adding caching, request tracing, and concurrent fetching without changing the reader's logic.

### Composable store stack

obspec-utils wrappers are transparent proxies --- each implements the same `ReadableStore` protocol and forwards to an inner store. They can be stacked in any order:

```
GranuleReader.read_coordinates() / .read_data()
        │
        ▼
┌─────────────────────────────────┐
│ h5py.File(reader)               │  ◄── file-like interface
│ or h5coro.H5Coro(driver)        │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│ EagerStoreReader                │  ◄── obspec-utils reader
│ Fetches full dataset via        │      (concurrent range requests)
│ concurrent get_ranges()         │
│ Default: 12 MB chunks × 18     │
│ parallel requests               │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│ CachingReadableStore            │  ◄── obspec-utils wrapper
│ LRU cache for repeated reads    │      (optional)
│ Thread-safe, configurable size  │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│ TracingReadableStore            │  ◄── obspec-utils wrapper
│ Records every byte-range        │      (development only)
│ request for profiling           │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│ obstore.S3Store                 │  ◄── concrete store
│ Rust-based object_store crate   │
└─────────────────────────────────┘
```

### Where each wrapper helps

| Wrapper | What it does | When it helps |
|---|---|---|
| `EagerStoreReader` | Fetches a file via parallel `get_ranges()` instead of sequential reads | HDF5 metadata parsing (h5py needs many small seeks; fetching the whole header region in parallel is faster) |
| `CachingReadableStore` | LRU cache of full objects, thread-safe | When the same granule appears in multiple parent cells (border cells share granules). Within a single Lambda this doesn't help, but in a local multi-threaded run it avoids redundant S3 fetches |
| `TracingReadableStore` | Logs every `get`, `get_range`, `get_ranges` call with path, offset, length, and duration | Profiling byte-range access patterns to identify I/O bottlenecks; understanding how many requests h5coro makes per granule |
| `SplittingReadableStore` | Splits large `get()` calls into concurrent `get_ranges()` | Large single-object downloads where the default `get()` is single-threaded |

### Example: profiling h5coro access patterns

```python
from obspec_utils.wrappers import TracingReadableStore, RequestTrace
from obstore.store import S3Store

trace = RequestTrace()
base = S3Store("nsidc-cumulus-prod-protected", region="us-west-2")
traced = TracingReadableStore(base, trace)

# Pass traced store to a GranuleReader that uses obstore instead of h5coro
reader = ATL06Reader(credentials, store=traced)
reader.read_coordinates("s3://nsidc-cumulus.../ATL06_...h5")

# Inspect access pattern
df = trace.to_dataframe()
print(df[["path", "offset", "length", "duration_ms"]])
print(trace.summary())  # total_requests, total_bytes, avg_latency
```

### Fit with the GranuleReader protocol

The store stack is an *implementation detail* of a `GranuleReader`. The protocol doesn't prescribe how bytes are fetched --- a reader could use h5coro, obstore, or fsspec internally. obspec-utils becomes relevant when building a reader backed by obstore:

```python
class ObstoreATL06Reader:
    """ATL06Reader using obstore + obspec-utils instead of h5coro."""

    def __init__(self, store: ReadableStore):
        self._store = store

    def read_coordinates(self, granule_url):
        reader = EagerStoreReader(self._store, granule_url)
        with h5py.File(reader, "r") as f:
            ...  # same ground-track logic as ATL06Reader
```

## Optional: VirtualiZarr for Pre-Computed References

!!! note "Optional Enhancement"
    This section describes how [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr)
    could eliminate per-invocation HDF5 metadata parsing by pre-computing byte-range
    references. This is a larger architectural change that replaces the catalog + h5coro
    read path.

### The current cost model

Each Lambda invocation pays two costs per granule:

1. **Metadata parsing** --- h5coro reads the HDF5 superblock and B-tree to locate datasets. This requires multiple small S3 range requests before any data is read.
2. **Data reading** --- h5coro fetches the actual dataset bytes via hyperslice.

For a parent cell that touches 20 granules × 6 ground tracks, that is 120 metadata-parse operations per Lambda, each requiring several S3 round trips. The data reads are unavoidable, but the metadata parsing is *identical every time the same granule is processed* --- it only depends on the file structure, not on which parent cell is being queried.

### How VirtualiZarr eliminates redundant parsing

VirtualiZarr parses each HDF5 file *once* and records the byte offsets of every chunk in a `ChunkManifest`. Subsequent access skips all HDF5 metadata parsing and reads data chunks directly by offset.

```
CURRENT PIPELINE (per Lambda):

  granule.h5 on S3
       │
       ▼
  h5coro: parse superblock ──── S3 range requests (metadata)
       │
       ▼
  h5coro: walk B-tree ───────── S3 range requests (metadata)
       │
       ▼
  h5coro: read dataset ──────── S3 range request  (data)


WITH VIRTUALIZARR (one-time setup + per Lambda):

  Step 0 (once):
  ┌──────────────────────────────────────────────────┐
  │ open_virtual_mfdataset(all 2,000 granules)       │
  │   ├─ Parse each HDF5 header (parallelizable)     │
  │   ├─ Record chunk byte offsets in ChunkManifest   │
  │   └─ Persist to Icechunk store                   │
  │                                                  │
  │ Output: {dataset_path: [(file, offset, len),...]} │
  │ Size:   ~100 MB of references for 2,000 granules  │
  └──────────────────────────────────────────────────┘

  Per Lambda:
  ┌──────────────────────────────────────────────────┐
  │ Open Icechunk store (instant, cached metadata)   │
  │       │                                          │
  │       ▼                                          │
  │ Look up chunk offsets for needed datasets         │
  │ (no HDF5 parsing, no S3 metadata round trips)    │
  │       │                                          │
  │       ▼                                          │
  │ Direct S3 range request to known offset ── data  │
  └──────────────────────────────────────────────────┘
```

### What changes

| Aspect | Current | With VirtualiZarr |
|---|---|---|
| **Catalog** | `{morton: [s3_urls]}` | `{morton: [s3_urls]}` + Icechunk reference store |
| **Per-Lambda HDF5 parsing** | 120 metadata-parse ops (20 granules × 6 tracks) | 0 --- offsets pre-computed |
| **Per-Lambda S3 round trips** | ~5 metadata + 1 data per track per granule | 1 data per track per granule |
| **One-time setup cost** | Catalog build (~30s) | Catalog build + virtualization (~10--30 min, parallelizable) |
| **Reader implementation** | h5coro with S3 driver | obstore with pre-computed offsets (or ManifestStore) |
| **Granule format changes** | Re-run pipeline | Re-run virtualization |

### Proposed integration

VirtualiZarr would add a new preparation step between catalog building and Lambda execution:

```
1. BUILD CATALOG  ──────────────────────── (unchanged)
        │
        ▼
2. VIRTUALIZE GRANULES  ──────────────────  (new step)
   open_virtual_mfdataset(all_granule_urls)
   Persist to Icechunk
        │
        ▼
3. CREATE ZARR TEMPLATE  ─────────────────  (unchanged)
        │
        ▼
4. PARALLEL EXECUTION  ───────────────────  (reader changes)
   Each Lambda:
     Open Icechunk reference store
     Look up byte offsets for needed datasets
     Read data via direct S3 range requests
        │
        ▼
5. CONSOLIDATE  ──────────────────────────  (unchanged)
```

The `GranuleReader` protocol accommodates this naturally --- a `VirtualATL06Reader` would look up chunk offsets from the Icechunk store instead of parsing HDF5 metadata:

```python
class VirtualATL06Reader:
    """ATL06Reader backed by pre-computed VirtualiZarr references."""

    def __init__(self, icechunk_store, s3_store):
        self._manifest = xr.open_zarr(icechunk_store)
        self._store = s3_store

    def read_coordinates(self, granule_url):
        # Read lat/lon via pre-computed byte offsets
        # No HDF5 metadata parsing needed
        ...

    def read_data(self, granule_url, group_index, row_slice, morton_indices):
        # Direct range request to known offset + length
        ...
```

### When virtualization is worth it

Virtualization adds a one-time setup cost but eliminates per-invocation overhead. The tradeoff depends on how many times the same granules are processed:

- **Single cycle, single run** --- marginal benefit. The 10--30 minute virtualization cost is comparable to the metadata overhead across 1,700 Lambdas.
- **Multiple runs on the same cycle** (parameter tuning, schema changes, debugging) --- clear win. Metadata parsing is done once, every subsequent run is faster.
- **Multi-cycle analysis** --- strong win. Virtualize each cycle once, re-aggregate as needed.
- **Adding new aggregation variables** --- strong win. The data is already referenced; only the `CellStatsSchema` and `AGG_FUNCTIONS` change.

## Key Design Decisions

**Why one chunk per parent cell?** Each Lambda writes to exactly one chunk of the Zarr store. Since chunks are the atomic unit of Zarr writes, 1,700+ workers can write concurrently without coordination or locking.

**Why h5coro?** Reading HDF5 from S3 normally requires downloading entire files. h5coro reads individual datasets via S3 byte-range requests, fetching only the data needed. A granule may be hundreds of MB, but we only read the few datasets we need for the tracks that intersect our cell.

**Why a pre-built catalog?** Without a catalog, each Lambda would need to query CMR independently to discover which granules intersect its cell. The catalog is built once (~30s) and passed to all workers, avoiding 1,700+ redundant CMR queries.

**Why morton indexing?** Morton (Z-order) curves preserve spatial locality --- nearby cells have nearby indices. This means a contiguous range of child indices maps to exactly one parent cell, enabling efficient `clip2order` operations and chunk-aligned writes.

## Output Format

Results are written to a [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) store following the [DGGS convention](https://github.com/zarr-conventions/dggs). The template is generated by [`xdggs_zarr_template`][magg.schema.xdggs_zarr_template] from the pandera schema, with one chunk per parent shard cell.

See [Schema](schema.md) for details on the output schema and aggregation dispatch.
