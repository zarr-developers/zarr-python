# Prospectus: Unified Chunk Grid Design for zarr-python

**Related:** 
- [#3750](https://github.com/zarr-developers/zarr-python/issues/3750) (single ChunkGrid proposal)
- [#3534](https://github.com/zarr-developers/zarr-python/pull/3534) (rectilinear implementation)
- [#3735](https://github.com/zarr-developers/zarr-python/pull/3735) (chunk grid module/registry)
- [ZEP0003](https://github.com/zarr-developers/zeps/blob/main/draft/ZEP0003.md) (variable chunking spec)
- [zarr-specs#370](https://github.com/zarr-developers/zarr-specs/pull/370) (sharding v1.1: non-divisible subchunks)
- [zarr-extensions#25](https://github.com/zarr-developers/zarr-extensions/pull/25) (rectilinear extension)
- [zarr-extensions#34](https://github.com/zarr-developers/zarr-extensions/issues/34) (sharding + rectilinear)

## Problem

The Zarr V3 spec defines `chunk_grid` as an extension point, suggesting chunk grids should be pluggable like codecs or data types. But chunk grids are fundamentally different:

- **Codecs are independent:** supporting `zstd` tells you nothing about `gzip`.
- **Chunk grids form a hierarchy:** the rectilinear chunk grid is strictly more general than the regular chunk grid (and zarrs' regular-bounded grid). Any regular grid is expressible as a rectilinear grid. Supporting rectilinear means you support all known grid types for free.

A registry-based plugin system adds complexity without clear benefit — there is no known chunk grid that is both (a) more general than rectilinear and (b) retains the tessellation properties that Zarr assumes. All known grids are special cases of the rectilinear grid:

| Grid type | Description | Rectilinear representation |
|---|---|---|
| Regular | All chunks same shape | All axes have a single repeated edge length |
| Regular-bounded (zarrs) | Regular, but boundary chunks trimmed to array extent | Last edge length per axis is `shape % chunk_size` |
| HPC boundary-padded | Regular interior, larger boundary chunks | First/last edge lengths differ from interior |
| Fully variable | Arbitrary per-chunk sizes | Direct representation |

If a future grid cannot be expressed as rectilinear (e.g., non-axis-aligned chunking, space-filling curves), it would require fundamentally different indexing and storage. Speculative generality today adds cost without benefit.

## Proposal

Replace the current multi-class chunk grid architecture with a single `ChunkGrid` implementation that handles both regular and rectilinear chunking, and drop user-defined chunk grids.

### Design principles

1. **One implementation, multiple serialization forms.** A single `ChunkGrid` class handles all chunking logic. It serializes to the simplest metadata — `"regular"` when all chunks are uniform, `"rectilinear"` otherwise.
2. **No chunk grid registry.** Remove the entrypoint-based registration system. A simple name-based dispatch in `parse_chunk_grid()` is sufficient.
3. **Fixed vs Varying per dimension.** Each axis is internally represented as either `FixedDimension(size)` (one integer — all chunks uniform) or `VaryingDimension(edges, cumulative)` (per-chunk edge lengths with precomputed prefix sums). This avoids expanding regular dimensions into lists of identical values.
4. **Shape-free grid.** The chunk grid describes a tiling pattern, not a bound region. It does not store the array shape. Methods that need the shape receive it as a parameter. This matches the Zarr V3 spec where `shape` and `chunk_grid` are independent fields.
5. **Transparent transitions.** Operations like `resize()` can move an array from regular to rectilinear chunking. This transition should be explicit and controllable.

### Internal representation

```python
@dataclass(frozen=True)
class FixedDimension:
    """All chunks on this axis have the same size."""
    size: int            # chunk edge length (> 0)

    def index_to_chunk(self, idx: int) -> int:
        return idx // self.size
    def chunk_offset(self, chunk_ix: int) -> int:
        return chunk_ix * self.size
    def chunk_size(self, chunk_ix: int, dim_len: int) -> int:
        return min(self.size, dim_len - chunk_ix * self.size)
    def indices_to_chunks(self, indices: NDArray) -> NDArray:
        return indices // self.size

@dataclass(frozen=True)
class VaryingDimension:
    """Chunks on this axis have explicit per-chunk sizes."""
    edges: tuple[int, ...]           # per-chunk edge lengths (all > 0)
    cumulative: tuple[int, ...]      # prefix sums for O(log n) lookup

    def index_to_chunk(self, idx: int) -> int:
        return bisect.bisect_right(self.cumulative, idx)
    def chunk_offset(self, chunk_ix: int) -> int:
        return self.cumulative[chunk_ix - 1] if chunk_ix > 0 else 0
    def chunk_size(self, chunk_ix: int, dim_len: int) -> int:
        return self.edges[chunk_ix]
    def indices_to_chunks(self, indices: NDArray) -> NDArray:
        return np.searchsorted(self.cumulative, indices, side='right')

@dataclass(frozen=True)
class ChunkGrid:
    dimensions: tuple[FixedDimension | VaryingDimension, ...]

    @property
    def is_regular(self) -> bool:
        return all(isinstance(d, FixedDimension) for d in self.dimensions)
```

`FixedDimension` and `VaryingDimension` share a common interface (`index_to_chunk`, `chunk_offset`, `chunk_size`, `indices_to_chunks`) used directly by the indexing pipeline. Memory usage scales with the number of *varying* dimensions and their chunk counts, not with the total number of chunks.

### API surface

#### Creating arrays

```python
# Regular chunks — serializes as {"name": "regular", ...}
arr = zarr.create_array(shape=(100, 200), chunks=(10, 20))

# Rectilinear chunks — serializes as {"name": "rectilinear", ...}
arr = zarr.create_array(shape=(60, 100), chunks=[[10, 20, 30], [25, 25, 25, 25]])

# RLE shorthand for rectilinear
arr = zarr.create_array(shape=(1000,), chunks=[[[100, 10]]])  # 10 chunks of size 100
```

#### Inspecting chunk grids

```python
arr.chunk_grid                  # ChunkGrid instance (always)
arr.chunk_grid.is_regular       # True if all dimensions are Fixed
arr.chunk_grid.chunk_shape      # (10, 20) — only when is_regular, else raises
arr.chunk_grid.ndim             # number of dimensions

# Per-chunk queries (array shape passed as parameter):
arr.chunk_grid.get_chunk_shape(arr.shape, chunk_coord=(0, 1))
arr.chunk_grid.get_chunk_origin(arr.shape, chunk_coord=(0, 1))
arr.chunk_grid.all_chunk_coords(arr.shape)
arr.chunk_grid.grid_shape(arr.shape)       # (10, 10) — chunks per dimension

# Out-of-bounds returns None:
arr.chunk_grid.get_chunk_shape(arr.shape, chunk_coord=(99, 99))  # None
```

#### `.chunks` property

`.chunks` is retained for regular grids, returning `tuple[int, ...]` as today. For rectilinear grids it raises `NotImplementedError`. `.chunk_grid` is the general-purpose API.

Three different chunk tuple conventions exist in the ecosystem:

| System | Type | Example |
|---|---|---|
| Zarr `arr.chunks` | `tuple[int, ...]` | `(256, 512)` |
| Dask `arr.chunks` | `tuple[tuple[int, ...], ...]` | `((256, 256, 64), (512, 512))` |
| xarray `.chunks` | `tuple[tuple[int, ...], ...]` | Same as dask |

Switching `.chunks` to dask-style tuples would be a breaking change and risks [expensive materialization for large regular grids](https://github.com/zarr-developers/zarr-python/pull/3534#discussion_r2457283002). The least disruptive path: keep `.chunks` for regular grids (no deprecation), add `.chunk_grid` alongside it, and let downstream libraries migrate at their own pace.

#### Serialization

```python
# Regular grid:
{"name": "regular", "configuration": {"chunk_shape": [10, 20]}}

# Rectilinear grid (with RLE compression):
{"name": "rectilinear", "configuration": {"chunk_shapes": [[10, 20, 30], [[25, 4]]]}}
```

Both names produce the same `ChunkGrid` class. Unknown names raise an error (chunk grids must always be understood).

#### Resize

```python
# Default: new region gets a single chunk spanning the growth
arr.resize((80, 100))  # becomes rectilinear if not evenly divisible

# Explicit: specify chunks for the new region
arr.resize((80, 100), chunks=[[10, 20, 30, 20], [25, 25, 25, 25]])

# Staying regular: if new shape is divisible by chunk size
arr.resize((70, 100))  # stays regular
```

### Indexing

The indexing pipeline is deeply coupled to regular grid assumptions. Every per-dimension indexer (`IntDimIndexer`, `SliceDimIndexer`, `BoolArrayDimIndexer`, `IntArrayDimIndexer`) takes a scalar `dim_chunk_len: int` and uses `//` and `*` for all arithmetic:

```python
dim_chunk_ix = self.dim_sel // self.dim_chunk_len          # IntDimIndexer
dim_offset = dim_chunk_ix * self.dim_chunk_len             # SliceDimIndexer
dim_sel_chunk = dim_sel // dim_chunk_len                   # IntArrayDimIndexer (vectorized)
```

For `VaryingDimension`, element-to-chunk mapping becomes a binary search and offset-to-chunk becomes a prefix sum lookup. The indexers must work with either representation.

**Recommended approach:** Replace `dim_chunk_len: int` with the dimension grid object (`FixedDimension | VaryingDimension`). The shared interface (`index_to_chunk`, `chunk_offset`, `chunk_size`, `indices_to_chunks`) means the indexer code structure stays the same — just replace `dim_sel // dim_chunk_len` with `dim_grid.index_to_chunk(dim_sel)`. This preserves O(1) arithmetic for regular dimensions and uses binary search only for varying ones.

Alternatives considered:
- **Precompute arrays** (offsets, sizes) at indexer creation and branch on scalar vs array — awkward, two code paths per indexer.
- **Always use `np.searchsorted`** for both types — uniform code but penalizes regular grids.

### Codec pipeline

Once the indexers determine *which* chunks to read or write, the codec pipeline needs to know *what shape* each chunk is. Today, `ArrayV3Metadata.get_chunk_spec()` ignores `chunk_coords` entirely — it returns the same `ArraySpec(shape=chunk_grid.chunk_shape)` for every chunk, because all chunks have the same shape in a regular grid.

For rectilinear grids, each chunk may have a different shape. `get_chunk_spec` must use the coordinates:

```python
def get_chunk_spec(self, chunk_coords, array_config, prototype) -> ArraySpec:
    chunk_shape = self.chunk_grid.get_chunk_shape(self.shape, chunk_coords)
    return ArraySpec(shape=chunk_shape, ...)
```

The codec pipeline uses `ArraySpec.shape` to allocate buffers, decode data, and validate output, so the per-chunk shape must be correct. This is a mechanical change — the `chunk_coords` parameter already exists (currently prefixed with `_` to signal it's unused) — but it touches every read/write path.

### Sharding

PR #3534 marks sharding as incompatible with rectilinear chunk grids. This constraint is unnecessary once the design is understood as three independent grid levels:

```
Level 1 — Outer chunk grid (shard boundaries)
    Can be regular or rectilinear.
    e.g., chunks = [[5000, 5980], [5000, 5980]]

Level 2 — Inner subchunk grid (within each shard)
    Always regular, but boundary subchunks may be clipped to shard shape.
    e.g., subchunk_shape = [512, 512]

Level 3 — Shard index
    ceil(shard_dim / subchunk_dim) entries per dimension, each (offset, size).
```

The `ShardingCodec` constructs a `ChunkGrid` per shard using the shard shape and subchunk shape. It doesn't need to know whether the outer grid is regular or rectilinear — each shard is self-contained.

[zarr-specs#370](https://github.com/zarr-developers/zarr-specs/pull/370) (sharding v1.1) lifts the requirement that subchunk shapes evenly divide the shard shape. With the proposed `ChunkGrid`, this requires one change: remove the `shard_shape % subchunk_shape == 0` validation. `FixedDimension` already handles boundary clipping.

These two features compose independently:

| Outer grid | Subchunk divisibility | Required change |
|---|---|---|
| Regular | Evenly divides (v1.0) | None (works today) |
| Regular | Non-divisible (v1.1) | Remove divisibility validation |
| Rectilinear | Evenly divides | Remove "sharding incompatible" guard |
| Rectilinear | Non-divisible | Both changes; no additional work |

### What this replaces

| Current design | Proposed design |
|---|---|
| `ChunkGrid` abstract base class | Single concrete `ChunkGrid` class |
| `RegularChunkGrid` subclass | `ChunkGrid` with `is_regular` property |
| `RectilinearChunkGrid` subclass (#3534) | Same `ChunkGrid` class |
| Chunk grid registry + entrypoints (#3735) | Removed — direct name dispatch |
| `arr.chunks` → `tuple[int, ...]` | Retained for regular grids; `arr.chunk_grid` for general use |

## Design decisions

### Why not store the array shape in ChunkGrid?

[#3736](https://github.com/zarr-developers/zarr-python/issues/3736) proposes adding `array_shape` to the chunk grid, motivated by the awkwardness of passing and re-validating `array_shape` on every method call in PR #3534. zarrs takes the same approach, storing the shape at construction. This prospectus diverges.

**For:** 

- Simpler method signatures (no repeated `array_shape` parameter). 
- Enables precomputing chunk count and boundary sizes. 
- Prevents callers from passing the wrong shape.
- Eliminates repeated validation.

**Against:** 

- The chunk grid is a tiling pattern, not a bound region. In the Zarr V3 spec, `chunk_grid` and `shape` are independent metadata fields. Storing the shape conflates "how to tile" with "what to tile over." Sharding exposes this — the same subchunk configuration produces different `ChunkGrid` instances for different shard shapes. `VaryingDimension` doesn't need the shape at all (edges fully define the grid). 
- TensorStore validates the separation in production, storing only `chunk_shape`.
- serialization becomes awkward — `to_dict()` would need to return the shape alongside the grid even though the spec doesn't couple them.

The repeated-validation problem from #3534 is real but has a simpler fix: validate once at `ArrayV3Metadata` construction (where both `shape` and `chunk_grid` are available), then trust that callers pass the correct shape downstream. For `VaryingDimension`, most methods don't use the shape at all — the edges and cumulative sums are self-contained. For `FixedDimension`, only boundary chunk size and grid extent need the shape, and these are computed with a single scalar per dimension, not the full tuple.

The cost of keeping them separate is one extra parameter on ~5 methods that are called O(1) times per operation. The benefit is a cleaner abstraction that's reusable across contexts (sharding, resize, serialization).

### Why not a chunk grid registry?

zarrs uses compile-time + runtime plugin registration. This makes sense for a library that explicitly supports user-defined extensions. For zarr-python, there is no known chunk grid outside the rectilinear family that retains the tessellation properties the codebase assumes. A simple `match` on the grid name in `parse_chunk_grid()` is sufficient and avoids entrypoint complexity.

### Why a single class instead of a Protocol?

zarrs uses independent types behind a shared trait. In Rust, the trait system enforces a uniform interface at zero runtime cost. In Python, a Protocol-based approach means every caller programs against an abstract interface, and adding a grid type requires implementing ~10 methods. Since all known grids are special cases of rectilinear, a single class is simpler while supporting the same metadata formats. If a genuinely novel grid type emerges, a Protocol can be extracted at that point.

## Prior art

### zarrs (Rust)

zarrs implements three independent chunk grid types (regular, regular-bounded, rectangular) behind a `ChunkGridTraits` trait. Key patterns adopted:

- **Fixed vs Varying per dimension** — rectangular grid distinguishes `Fixed(size)` vs `Varying(Vec<OffsetSize>)` per axis
- **Prefix sums + binary search** — precomputed offsets with `partition_point` for O(log n) lookup
- **None for out-of-bounds** — chunk queries return `Option<T>` instead of panicking
- **Non-zero chunk dimensions** — `NonZeroU64` makes zero-sized chunks unrepresentable
- **Sharding creates a separate grid** — `ShardingCodec` constructs an independent subchunk grid per shard

### TensorStore (C++)

TensorStore's `ChunkGridSpecification` stores only `chunk_shape`, not the array shape — validating the shape-free approach. It has both `RegularGridRef` and `IrregularGrid` internally (the latter with sorted breakpoints per dimension), but only the regular grid is used for Zarr V3. No chunk grid registry — the `"regular"` name is hardcoded.

## Migration

### Existing PRs

**#3735** (chunk grid module, +313/−65, approved by @maxrjones) splits `chunk_grids.py` into a `chunk_grids/` package (`__init__.py`, `common.py`, `regular.py`) and adds a chunk grid registry. The module layout is reusable. The registry (`register_chunk_grid` / `get_chunk_grid_class` in `registry.py`) is not — it should be replaced with direct name dispatch before merging.

**#3737** (chunk grid array shape, +514/−198, draft) implements #3736 by adding `array_shape` to `ChunkGrid`. Depends on #3735. The prospectus argues against storing the array shape in the grid (see Design decisions). This PR should be closed.

**#3534** (rectilinear implementation, +5716/−408, extensive review) introduces `RectilinearChunkGrid` as a separate subclass. The prospectus proposes a different architecture (single `ChunkGrid` with `FixedDimension`/`VaryingDimension`). Reusable components:

| #3534 component | Disposition |
|---|---|
| `_expand_run_length_encoding` / `_compress_run_length_encoding` | **Keep** as-is |
| `_normalize_rectilinear_chunks` / `_parse_chunk_shapes` | **Keep with modifications** — feed into `VaryingDimension` construction |
| `resolve_chunk_spec` / `ChunksLike` type alias | **Keep** — orthogonal to grid class design |
| `_validate_zarr_format_compatibility` | **Keep** — rectilinear is V3-only |
| `_validate_sharding_compatibility` | **Remove** — sharding is compatible with rectilinear |
| `_validate_data_compatibility` (`from_array` guard) | **Keep for now** — needs separate design work |
| `RectilinearChunkGrid` class / `ConfigurationDict` | **Replace** — single `ChunkGrid` class |
| `chunk_grid` property on `Array`/`AsyncArray` | **Keep** |
| `.chunks` raising for rectilinear | **Keep** |
| Tests | **Adapt** for single-class API |
| Indexing changes | **Insufficient** — `assert isinstance(chunk_grid, RegularChunkGrid)` guards remain |

Given the scope of architectural changes, a **fresh PR** is more practical than adapting #3534. Rebasing and reworking its core classes would touch nearly every line of a 5700-line diff while inheriting review history that no longer applies.

**#1483** (ZEP0003 POC, +346/−20, draft, V2) is @martindurant's original proof-of-concept for variable chunking on Zarr V2. It demonstrated feasibility but targets the V2 format and predates the V3 extension point design. Should be closed.

### Plan

1. **Amend and merge #3735.** Keep the `chunk_grids/` module layout. Replace the registry with direct name dispatch in `parse_chunk_grid()`. Remove `register_chunk_grid` / `get_chunk_grid_class` from `registry.py` and the entrypoint from `pyproject.toml`.

2. **Open a new PR** implementing the prospectus:
   - `FixedDimension` and `VaryingDimension` dataclasses with shared interface (`index_to_chunk`, `chunk_offset`, `chunk_size`, `indices_to_chunks`).
   - Single `ChunkGrid` class with `dimensions: tuple[FixedDimension | VaryingDimension, ...]` and `is_regular`.
   - `parse_chunk_grid()` recognizes `"regular"` and `"rectilinear"`.
   - Port RLE helpers, `resolve_chunk_spec`, `ChunksLike`, and validation functions from #3534.
   - Refactor per-dimension indexers to accept `FixedDimension | VaryingDimension` instead of `dim_chunk_len: int`.
   - Update `get_chunk_spec` to compute per-chunk shapes from coordinates.
   - Add `arr.chunk_grid` property. Keep `.chunks` for regular grids, raise for rectilinear.
   - Remove the "sharding incompatible with rectilinear" guard.
   - Adapt tests from #3534.

3. **Close trial PRs** with comments linking to the new PR and crediting contributions:
   - **Close #3534** — credit RLE helpers, validation logic, chunk spec resolution, test cases, and review discussion that shaped the design.
   - **Close #3737** — reference the shape-free design decision.
   - **Close #1483** — credit as the original POC that motivated the work; superseded by the V3 implementation.
   - **Close #3736** — respond with the shape-free design rationale.

4. **Sharding v1.1** (after zarr-specs#370 is accepted) — separate PR removing the `shard_shape % subchunk_shape == 0` validation in `ShardingCodec`.

### Downstream migration

Four active PRs/issues in the ecosystem depend on zarr-python's rectilinear chunk grid support. All currently track #3534 as their upstream dependency. The unified `ChunkGrid` design is a narrower API surface than the two-class hierarchy, so the net effect is less integration work per downstream — but each needs updates.

#### xarray ([pydata/xarray#10880](https://github.com/pydata/xarray/pull/10880))

Draft PR by @keewis (+26/−9 in `xarray/backends/zarr.py`) enabling variable-sized chunk writes and reads via the zarr backend. Currently imports `RectilinearChunkGrid` / `RegularChunkGrid` for feature detection and branches on `isinstance` checks.

**Required changes:**

- **Feature detection.** Replace class-existence checks (`hasattr(zarr, 'RectilinearChunkGrid')`) with a version check or try-import of the unified `ChunkGrid`. Since the prospectus exports a single class, detection simplifies to checking whether `ChunkGrid` accepts non-uniform dimensions (or just `zarr.__version__`).
- **Write path.** Currently constructs chunk info that `RectilinearChunkGrid` understands. The prospectus's `chunks=[[10, 20, 30], [25, 25, 25, 25]]` API for `create_array` is a more natural fit — the xarray write path may get simpler.
- **Read path.** Replace `isinstance(chunk_grid, RectilinearChunkGrid)` with `not chunk_grid.is_regular`. Per-dimension chunk sizes come from `chunk_grid.dimensions[i].edges` (for `VaryingDimension`) or are computed from `chunk_grid.dimensions[i].size` (for `FixedDimension`).
- **`validate_grid_chunks_alignment`.** Still needs work regardless of class hierarchy — the approach is the same either way.

**Effort:** ~1–2 days. The PR is small and the unified API is more ergonomic for xarray's use case.

#### VirtualiZarr ([zarr-developers/VirtualiZarr#877](https://github.com/zarr-developers/VirtualiZarr/pull/877))

Draft PR by @maxrjones adding rectilinear support to `ManifestArray`, with a `has_rectilinear_chunk_grid_support` feature flag and vendored `_is_nested_sequence` helper from #3534.

**Required changes:**

- **Drop vendored `_is_nested_sequence`.** The prospectus eliminates `RectilinearChunkGrid` as a separate class, so nested-sequence detection for choosing grid type is unnecessary — just construct `ChunkGrid` with appropriate dimension types.
- **`isinstance` → `.is_regular`.** All `isinstance(chunk_grid, RectilinearChunkGrid)` checks become `not chunk_grid.is_regular`.
- **`ManifestArray.chunks`.** Currently returns `chunk_grid.chunk_shapes` for rectilinear grids. Under the prospectus, chunk shapes come from iterating dimension edges. The dask-style `tuple[tuple[int, ...], ...]` format VirtualiZarr uses internally is unaffected.
- **`copy_and_replace_metadata`.** Simplifies: no need to detect nested sequences to pick a grid class.
- **Test environment.** Currently pins jhamman's zarr-python fork — would track whatever branch implements the prospectus.

**Effort:** ~1–2 days. Mostly mechanical type-check replacements plus dropping the vendored helper. Concat/stack logic is grid-type-agnostic once chunk shapes are available.

#### Icechunk ([earth-mover/icechunk#1338](https://github.com/earth-mover/icechunk/issues/1338))

Investigation issue for supporting rectilinear grids in the IC2 on-disk format. The `DimensionShape { dim_length, chunk_length }` struct needs extension to encode per-chunk sizes.

**Impact:** Minimal. Icechunk's format changes are driven by the *spec* (ZEP0003 / rectilinear extension), not zarr-python's class hierarchy. The unified `ChunkGrid` means Icechunk's Python-side metadata ingestion handles one type instead of two. The `shift_array` / `reindex` concerns raised in the discussion are orthogonal to this design.

**Effort:** No change to the work already scoped. May marginally simplify the Python integration layer.

#### cubed ([cubed-dev/cubed#876](https://github.com/cubed-dev/cubed/issues/876))

Draft by @TomNicholas using rectilinear intermediate stores to reduce rechunking stages (+142/−27 across storage adapter, blockwise, and ops).

**Required changes:**

- **Store creation.** `zarr_python_v3.py` currently creates `RectilinearChunkGrid` instances directly. Switch to constructing `ChunkGrid` via the prospectus's list-of-lists `chunks` API.
- **Chunk shape queries.** Any `isinstance` checks on grid type become `.is_regular` checks.
- The rechunking algorithm itself is independent of the class hierarchy — it operates on per-dimension chunk tuples internally.

**Effort:** <1 day. Changes are concentrated in the storage adapter layer, and the prospectus's API is a natural fit for cubed's internal representation.

#### Migration pattern

All four downstreams follow the same pattern. The migration from the two-class API to the unified API is mechanical:

| Two-class pattern | Unified pattern |
|---|---|
| `isinstance(cg, RegularChunkGrid)` | `cg.is_regular` |
| `isinstance(cg, RectilinearChunkGrid)` | `not cg.is_regular` |
| `cg.chunk_shape` (regular only) | `cg.chunk_shape` (raises if not regular) |
| `cg.chunk_shapes` (rectilinear) | `tuple(d.edges for d in cg.dimensions)` |
| `RegularChunkGrid(chunk_shape=(...))` | `ChunkGrid.from_regular((...))` or `chunks=(...)` in `create_array` |
| `RectilinearChunkGrid(chunk_shapes=(...))` | `ChunkGrid.from_rectilinear((...))` or `chunks=[[...], [...]]` in `create_array` |
| Feature detection via class import | Version check or `hasattr(ChunkGrid, 'is_regular')` |

## Open questions

1. **RLE in the Python API:** Should users pass RLE-encoded chunk specs directly, or only expanded lists? RLE is primarily a serialization concern, but for arrays with millions of chunks it matters at construction time too.
2. **Resize defaults:** When growing a regular array, should the default preserve regularity (extending the last chunk) or create a new chunk for the added region (transitioning to rectilinear)?
