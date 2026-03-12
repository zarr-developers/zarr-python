# Unified Chunk Grid

Version: 4

**Related:**

- [#3750](https://github.com/zarr-developers/zarr-python/issues/3750) (single ChunkGrid proposal)
- [#3534](https://github.com/zarr-developers/zarr-python/pull/3534) (rectilinear implementation)
- [#3735](https://github.com/zarr-developers/zarr-python/pull/3735) (chunk grid module/registry)
- [ZEP0003](https://github.com/zarr-developers/zeps/blob/main/draft/ZEP0003.md) (variable chunking spec)
- [zarr-specs#370](https://github.com/zarr-developers/zarr-specs/pull/370) (sharding v1.1: non-divisible subchunks)
- [zarr-extensions#25](https://github.com/zarr-developers/zarr-extensions/pull/25) (rectilinear extension)
- [zarr-extensions#34](https://github.com/zarr-developers/zarr-extensions/issues/34) (sharding + rectilinear)

## Problem

The Zarr V3 spec defines `chunk_grid` as an extension point, but chunk grids are fundamentally different from codecs. Codecs are independent — supporting `zstd` tells you nothing about `gzip`. Chunk grids form a hierarchy — the rectilinear grid is strictly more general than the regular grid. Any regular grid is expressible as a rectilinear grid.

There is no known chunk grid that is both (a) more general than rectilinear and (b) retains the axis-aligned tessellation properties Zarr assumes. All known grids are special cases:

| Grid type | Description |
|---|---|
| Regular | Uniform chunk size, boundary chunks padded with fill_value |
| Regular-bounded (zarrs) | Uniform chunk size, boundary chunks trimmed to array extent |
| HPC boundary-padded | Regular interior, larger boundary chunks |
| Fully variable | Arbitrary per-chunk sizes |

A registry-based plugin system adds complexity without clear benefit.

## Design

### Principles

1. **A chunk grid is a concrete arrangement of chunks.** Not an abstract tiling pattern — the specific partition of a specific array. The grid stores enough information to answer any question about any chunk without external parameters.
2. **One implementation, multiple serialization forms.** A single `ChunkGrid` class handles all chunking logic. The serialization format (`"regular"` vs `"rectilinear"`) is chosen by the metadata layer, not the grid.
3. **No chunk grid registry.** Simple name-based dispatch in `parse_chunk_grid()`.
4. **Fixed vs Varying per dimension.** `FixedDimension(size, extent)` for uniform chunks; `VaryingDimension(edges, extent)` for per-chunk edge lengths with precomputed prefix sums. Avoids expanding regular dimensions into lists of identical values.
5. **Transparent transitions.** Operations like `resize()` can move an array from regular to rectilinear chunking.

### Internal representation

```python
@dataclass(frozen=True)
class FixedDimension:
    """Uniform chunk size. Boundary chunks contain less data but are
    encoded at full size by the codec pipeline."""
    size: int            # chunk edge length (>= 0)
    extent: int          # array dimension length

    def __post_init__(self) -> None:
        # validates size >= 0 and extent >= 0

    @property
    def nchunks(self) -> int:
        if self.size == 0:
            return 1 if self.extent == 0 else 0
        return ceildiv(self.extent, self.size)

    def index_to_chunk(self, idx: int) -> int:
        return idx // self.size
    def chunk_offset(self, chunk_ix: int) -> int:
        return chunk_ix * self.size
    def chunk_size(self, chunk_ix: int) -> int:
        return self.size                                       # always uniform
    def data_size(self, chunk_ix: int) -> int:
        return max(0, min(self.size, self.extent - chunk_ix * self.size))
    def indices_to_chunks(self, indices: NDArray) -> NDArray:
        return indices // self.size
    def with_extent(self, new_extent: int) -> FixedDimension:
        return FixedDimension(size=self.size, extent=new_extent)
    def resize(self, new_extent: int) -> FixedDimension:
        return FixedDimension(size=self.size, extent=new_extent)

@dataclass(frozen=True)
class VaryingDimension:
    """Explicit per-chunk sizes. The last chunk may extend past the array
    extent, in which case data_size clips to the valid region while
    chunk_size returns the full edge length for codec processing."""
    edges: tuple[int, ...]           # per-chunk edge lengths (all > 0)
    cumulative: tuple[int, ...]      # prefix sums for O(log n) lookup
    extent: int                      # array dimension length (may be < sum(edges))

    def __init__(self, edges: Sequence[int], extent: int) -> None:
        # validates edges non-empty, all > 0, extent >= 0, extent <= sum(edges)
        # computes cumulative via itertools.accumulate
        # uses object.__setattr__ for frozen dataclass

    @property
    def nchunks(self) -> int:
        return len(self.edges)

    def index_to_chunk(self, idx: int) -> int:
        return bisect.bisect_right(self.cumulative, idx)
    def chunk_offset(self, chunk_ix: int) -> int:
        return self.cumulative[chunk_ix - 1] if chunk_ix > 0 else 0
    def chunk_size(self, chunk_ix: int) -> int:
        return self.edges[chunk_ix]
    def data_size(self, chunk_ix: int) -> int:
        offset = self.chunk_offset(chunk_ix)
        return max(0, min(self.edges[chunk_ix], self.extent - offset))
    def indices_to_chunks(self, indices: NDArray) -> NDArray:
        return np.searchsorted(self.cumulative, indices, side='right')
    def with_extent(self, new_extent: int) -> VaryingDimension:
        # validates edge_sum >= new_extent, re-binds extent
        return VaryingDimension(self.edges, extent=new_extent)
    def resize(self, new_extent: int) -> VaryingDimension:
        # grow: append chunk of size (new_extent - old_extent)
        # shrink: drop trailing chunks, keep those up to new_extent
```

Both types implement the `DimensionGrid` protocol: `nchunks`, `extent`, `index_to_chunk`, `chunk_offset`, `chunk_size`, `data_size`, `indices_to_chunks`, `with_extent`, `resize`. Memory usage scales with the number of *varying* dimensions, not total chunks.

The two size methods serve different consumers:

| Method | Returns | Consumer |
|---|---|---|
| `chunk_size` | Buffer size for codec processing | Codec pipeline (`ArraySpec.shape`) |
| `data_size` | Valid data region within the buffer | Indexing pipeline (`chunk_selection` slicing) |

For `FixedDimension`, these differ only at the boundary. For `VaryingDimension`, these differ only when the last chunk extends past the extent (i.e., `extent < sum(edges)`). This matches current zarr-python behavior: `get_chunk_spec` passes the full `chunk_shape` to the codec for all chunks, and the indexer generates a `chunk_selection` that clips the decoded buffer.

### DimensionGrid Protocol

```python
@runtime_checkable
class DimensionGrid(Protocol):
    """Structural interface shared by FixedDimension and VaryingDimension."""

    @property
    def nchunks(self) -> int: ...
    @property
    def extent(self) -> int: ...
    def index_to_chunk(self, idx: int) -> int: ...
    def chunk_offset(self, chunk_ix: int) -> int: ...
    def chunk_size(self, chunk_ix: int) -> int: ...
    def data_size(self, chunk_ix: int) -> int: ...
    def indices_to_chunks(self, indices: NDArray[np.intp]) -> NDArray[np.intp]: ...
    def with_extent(self, new_extent: int) -> DimensionGrid: ...
    def resize(self, new_extent: int) -> DimensionGrid: ...
```

The protocol is `@runtime_checkable`, enabling polymorphic handling of both dimension types without `isinstance` checks.

### ChunkSpec

```python
@dataclass(frozen=True)
class ChunkSpec:
    slices: tuple[slice, ...]        # valid data region in array coordinates
    codec_shape: tuple[int, ...]     # buffer shape for codec processing

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.slices)

    @property
    def is_boundary(self) -> bool:
        return self.shape != self.codec_shape
```

For interior chunks, `shape == codec_shape`. For boundary chunks of a regular grid, `codec_shape` is the full declared chunk size while `shape` is clipped. For rectilinear grids, `shape == codec_shape` unless the last chunk extends past the extent.

### API

```python
# Creating arrays
arr = zarr.create_array(shape=(100, 200), chunks=(10, 20))                          # regular
arr = zarr.create_array(shape=(60, 100), chunks=[[10, 20, 30], [25, 25, 25, 25]])   # rectilinear

# ChunkGrid as a collection
grid = arr.metadata.chunk_grid    # ChunkGrid instance
grid.shape                        # (10, 10) — number of chunks per dimension
grid.ndim                         # 2
grid.is_regular                   # True if all dimensions are Fixed

spec = grid[0, 1]                 # ChunkSpec for chunk at grid position (0, 1)
spec.slices                       # (slice(0, 10), slice(20, 40))
spec.shape                        # (10, 20) — data shape
spec.codec_shape                  # (10, 20) — same for interior chunks

boundary = grid[9, 0]             # boundary chunk (extent=100, size=10)
boundary.shape                    # (10, 20) — data shape
boundary.codec_shape              # (10, 20) — codec sees full buffer

grid[99, 99]                      # None — out of bounds

for spec in grid:                 # iterate all chunks
    ...

# .chunks property: retained for regular grids, raises NotImplementedError for rectilinear
arr.chunks                         # (10, 20)

# .chunk_sizes property: works for all grids (dask-style)
arr.chunk_sizes                    # ((10, 10, ..., 10), (20, 20, ..., 20))
```

`ChunkGrid.__getitem__` constructs `ChunkSpec` using `chunk_size` for `codec_shape` and `data_size` for `slices`:

```python
def __getitem__(self, coords: int | tuple[int, ...]) -> ChunkSpec | None:
    if isinstance(coords, int):
        coords = (coords,)
    slices = []
    codec_shape = []
    for dim, ix in zip(self.dimensions, coords):
        if ix < 0 or ix >= dim.nchunks:
            return None
        offset = dim.chunk_offset(ix)
        slices.append(slice(offset, offset + dim.data_size(ix)))
        codec_shape.append(dim.chunk_size(ix))
    return ChunkSpec(tuple(slices), tuple(codec_shape))
```

#### Construction

Both `from_regular` and `from_rectilinear` require `array_shape`, binding the extent per dimension at construction time. This is a core design choice: a chunk grid is a concrete arrangement for a specific array, not an abstract tiling pattern.

```python
# Regular grid — all FixedDimension
grid = ChunkGrid.from_regular(array_shape=(100, 200), chunk_shape=(10, 20))

# Rectilinear grid — extent = sum(edges) when shape matches
grid = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]], array_shape=(60, 100))

# Rectilinear grid with boundary clipping — last chunk extends past array extent
# e.g., shape=(55, 90) but edges sum to (60, 100): data_size clips at extent
grid = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]], array_shape=(55, 90))

# Direct construction
grid = ChunkGrid(dimensions=(FixedDimension(10, 100), VaryingDimension([10, 20, 30], 55)))
```

When `extent < sum(edges)`, the dimension is always stored as `VaryingDimension` (even if all edges are identical) to preserve the explicit edge count. The last chunk's `chunk_size` returns the full declared edge (codec buffer) while `data_size` clips to the extent. This mirrors how `FixedDimension` handles boundary chunks in regular grids.

#### Serialization

```python
# Regular grid:
{"name": "regular", "configuration": {"chunk_shape": [10, 20]}}

# Rectilinear grid (with RLE compression and "kind" field):
{"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": [[10, 20, 30], [[25, 4]]]}}
```

Both names deserialize to the same `ChunkGrid` class. The serialized form does not include the array extent — that comes from `shape` in array metadata and is passed to `parse_chunk_grid()` at construction time.

**The `ChunkGrid` does not serialize itself.** The format choice (`"regular"` vs `"rectilinear"`) belongs to `ArrayV3Metadata`, which stores the chunk grid's JSON `name` in the `chunk_grid_name` field. `serialize_chunk_grid(grid, name)` is called by `ArrayV3Metadata.to_dict()`. This gives round-trip fidelity — a store written as rectilinear with uniform edges stays rectilinear.

For `create_array`, the format is inferred from the `chunks` argument: a flat tuple produces `"regular"`, a nested list produces `"rectilinear"`. The `_is_rectilinear_chunks()` helper detects nested sequences like `[[10, 20], [5, 5]]`.

##### Rectilinear spec compliance

The rectilinear format requires `"kind": "inline"` (validated by `_validate_rectilinear_kind()`). Per the spec, each element of `chunk_shapes` can be:

- A bare integer `m`: repeated until `sum >= array_extent`
- A list of bare integers: explicit per-chunk sizes
- A mixed array of bare integers and `[value, count]` RLE pairs

RLE compression is used when serializing: runs of identical sizes become `[value, count]` pairs, singletons stay as bare integers.

```python
# _compress_rle([10, 10, 10, 5]) -> [[10, 3], 5]
# _expand_rle([[10, 3], 5])      -> [10, 10, 10, 5]
```

For `FixedDimension` serialized as rectilinear, `_serialize_fixed_dim()` produces a compact representation: bare integer when evenly divisible, `[size, last_data]` for two chunks, `[[size, n-1], last_data]` for more.

#### chunk_sizes

The `chunk_sizes` property provides universal access to per-dimension chunk data sizes, matching the dask `Array.chunks` convention. It works for both regular and rectilinear grids:

```python
>>> arr = zarr.create_array(store, shape=(100, 80), chunks=(30, 40))
>>> arr.chunk_sizes
((30, 30, 30, 10), (40, 40))

>>> arr = zarr.create_array(store, shape=(60, 100), chunks=[[10, 20, 30], [50, 50]])
>>> arr.chunk_sizes
((10, 20, 30), (50, 50))
```

#### Resize

```python
arr.resize((80, 100))       # re-binds extent; FixedDimension stays fixed
arr.resize((200, 100))      # VaryingDimension grows by appending a new chunk
arr.resize((30, 100))       # VaryingDimension shrinks by dropping trailing chunks
```

Resize uses `ChunkGrid.update_shape(new_shape)`, which delegates to each dimension's `.resize()` method:
- `FixedDimension.resize()`: simply re-binds the extent (identical to `with_extent`)
- `VaryingDimension.resize()`: grow appends a chunk of size `new_extent - old_extent`; shrink drops trailing chunks whose cumulative offset lies beyond the new extent

#### from_array

The `from_array()` function handles both regular and rectilinear source arrays:

```python
src = zarr.create_array(store, shape=(60, 100), chunks=[[10, 20, 30], [50, 50]])
new = zarr.from_array(data=src, store=new_store, chunks="keep")
# Preserves rectilinear structure: new.chunk_sizes == ((10, 20, 30), (50, 50))
```

When `chunks="keep"`, the logic checks `data.metadata.chunk_grid.is_regular`:
- Regular: extracts `data.chunks` (flat tuple) and preserves shards
- Rectilinear: extracts `data.chunk_sizes` (nested tuples) and forces shards to None

### Indexing

The indexing pipeline is coupled to regular grid assumptions — every per-dimension indexer takes a scalar `dim_chunk_len: int` and uses `//` and `*`:

```python
dim_chunk_ix = self.dim_sel // self.dim_chunk_len          # IntDimIndexer
dim_offset = dim_chunk_ix * self.dim_chunk_len             # SliceDimIndexer
```

Replace `dim_chunk_len: int` with the dimension object (`FixedDimension | VaryingDimension`). The shared interface means the indexer code structure stays the same — `dim_sel // dim_chunk_len` becomes `dim_grid.index_to_chunk(dim_sel)`. O(1) for regular, binary search for varying.

### Codec pipeline

Today, `get_chunk_spec()` returns the same `ArraySpec(shape=chunk_grid.chunk_shape)` for every chunk. For rectilinear grids, each chunk has a different codec shape:

```python
def get_chunk_spec(self, chunk_coords, array_config, prototype) -> ArraySpec:
    spec = self.chunk_grid[chunk_coords]
    return ArraySpec(shape=spec.codec_shape, ...)
```

Note `spec.codec_shape`, not `spec.shape`. For regular grids, `codec_shape` is uniform (preserving current behavior). The boundary clipping flow is unchanged:

```
Write: user data → pad to codec_shape with fill_value → encode → store
Read:  store → decode to codec_shape → slice via chunk_selection → user data
```

### Sharding

The `ShardingCodec` constructs a `ChunkGrid` per shard using the shard shape as extent and the subchunk shape as `FixedDimension`. Each shard is self-contained — it doesn't need to know whether the outer grid is regular or rectilinear. Rectilinear chunks with sharding currently raises `ValueError` pending further validation work.

```
Level 1 — Outer chunk grid (shard boundaries): regular or rectilinear
Level 2 — Inner subchunk grid (within each shard): always regular
Level 3 — Shard index: ceil(shard_dim / subchunk_dim) entries per dimension
```

[zarr-specs#370](https://github.com/zarr-developers/zarr-specs/pull/370) lifts the requirement that subchunk shapes evenly divide the shard shape. With the proposed `ChunkGrid`, this just means removing the `shard_shape % subchunk_shape == 0` validation — `FixedDimension` already handles boundary clipping via `data_size`.

| Outer grid | Subchunk divisibility | Required change |
|---|---|---|
| Regular | Evenly divides (v1.0) | None |
| Regular | Non-divisible (v1.1) | Remove divisibility validation |
| Rectilinear | Evenly divides | Remove "sharding incompatible" guard |
| Rectilinear | Non-divisible | Both changes |

### What this replaces

| Current | Proposed |
|---|---|
| `ChunkGrid` ABC + `RegularChunkGrid` subclass | Single concrete `ChunkGrid` with `is_regular` |
| `RectilinearChunkGrid` (#3534) | Same `ChunkGrid` class |
| Chunk grid registry + entrypoints (#3735) | Direct name dispatch |
| `arr.chunks` | Retained for regular; `arr.chunk_sizes` for general use |
| `get_chunk_shape(shape, coord)` | `grid[coord].codec_shape` or `grid[coord].shape` |

## Design decisions

### Why store the extent in ChunkGrid?

The chunk grid is a concrete arrangement, not an abstract tiling pattern. A finite collection naturally has an extent. Storing it enables `__getitem__`, eliminates `dim_len` parameters from every method, and makes the grid self-describing.

This does *not* mean `ArrayV3Metadata.shape` should delegate to the grid. The array shape remains an independent field in metadata. The extent is passed into the grid at construction time so it can answer boundary questions without external parameters. It is **not** serialized as part of the chunk grid JSON — it comes from the `shape` field in array metadata and is passed to `parse_chunk_grid()`.

### Why distinguish chunk_size from data_size?

A chunk in a regular grid has two sizes. `chunk_size` is the buffer size the codec processes — always `size` for `FixedDimension`, even at the boundary (padded with `fill_value`). `data_size` is the valid data region — clipped to `extent % size` at the boundary. The indexing layer uses `data_size` to generate `chunk_selection` slices.

This matches current zarr-python behavior and matters for:
1. **Backward compatibility.** Existing stores have boundary chunks encoded at full `chunk_shape`.
2. **Codec simplicity.** Codecs assume uniform input shapes for regular grids.
3. **Shard index correctness.** The index assumes `subchunk_dim`-sized entries.

For `VaryingDimension`, `chunk_size == data_size` when `extent == sum(edges)`. When `extent < sum(edges)` (e.g., after a resize that keeps the last chunk oversized), `data_size` clips the last chunk. This is the fundamental difference: `FixedDimension` has a declared size plus an extent that clips data; `VaryingDimension` has explicit sizes that normally *are* the extent but can also extend past it.

### Why not a chunk grid registry?

There is no known chunk grid outside the rectilinear family that retains the tessellation properties zarr-python assumes. A `match` on the grid name is sufficient.

### Why a single class instead of a Protocol?

All known grids are special cases of rectilinear. A Protocol-based approach means every caller programs against an abstract interface and adding a grid type requires implementing ~10 methods. A single class is simpler. If a genuinely novel grid type emerges, a Protocol can be extracted.

## Prior art

**zarrs (Rust):** Three independent grid types behind a `ChunkGridTraits` trait. Key patterns adopted: Fixed vs Varying per dimension, prefix sums + binary search, `Option<T>` for out-of-bounds, `NonZeroU64` for chunk dimensions, separate subchunk grid per shard, array shape at construction.

**TensorStore (C++):** Stores only `chunk_shape` — boundary clipping via `valid_data_bounds` at query time. Both `RegularGridRef` and `IrregularGrid` internally. No registry.

## Migration

### Plan

1. **Amend and merge #3735.** Keep the `chunk_grids/` module layout. Replace the registry with direct name dispatch. Remove `register_chunk_grid` / `get_chunk_grid_class` and the entrypoint.
2. **Open a new PR** implementing this prospectus:
   - `FixedDimension`, `VaryingDimension`, `DimensionGrid` protocol, `ChunkSpec`, and `ChunkGrid` classes.
   - `parse_chunk_grid(metadata, array_shape)` with `"regular"` and `"rectilinear"` dispatch.
   - Port RLE helpers, `resolve_chunk_spec`, `ChunksLike`, and validation functions from #3534.
   - Refactor per-dimension indexers to accept `FixedDimension | VaryingDimension`.
   - Update `get_chunk_spec` to use `grid[chunk_coords].codec_shape`.
   - Add `arr.chunk_sizes`. Keep `.chunks` for regular, raise for rectilinear.
   - Remove the "sharding incompatible with rectilinear" guard.
   - Adapt tests from #3534.
3. **Close trial PRs** with credits:
   - **#3534** — RLE helpers, validation logic, chunk spec resolution, test cases, review discussion.
   - **#3737** — extent-in-grid idea (adopted per-dimension).
   - **#1483** — original POC; superseded by V3 implementation.
   - **#3736** — resolved by storing extent per-dimension.
4. **Sharding v1.1** (separate PR, after zarr-specs#370) — remove `shard_shape % subchunk_shape == 0` validation.

### Reusable components from #3534

| Component | Disposition |
|---|---|
| RLE encode/decode helpers | **Keep** |
| `_normalize_rectilinear_chunks` / `_parse_chunk_shapes` | **Keep** — feed into `VaryingDimension` |
| `resolve_chunk_spec` / `ChunksLike` | **Keep** |
| `_validate_zarr_format_compatibility` | **Keep** — rectilinear is V3-only |
| `_validate_sharding_compatibility` | **Remove** — sharding is compatible |
| `RectilinearChunkGrid` class | **Replace** |
| Indexing changes | **Insufficient** — `isinstance` guards remain |

A **fresh PR** is more practical than adapting #3534's 5700-line diff.

### Downstream migration

All four downstream PRs/issues follow the same pattern:

| Two-class pattern | Unified pattern |
|---|---|
| `isinstance(cg, RegularChunkGrid)` | `cg.is_regular` |
| `isinstance(cg, RectilinearChunkGrid)` | `not cg.is_regular` |
| `cg.chunk_shape` | `cg.dimensions[i].size` or `cg[coord].shape` |
| `cg.chunk_shapes` | `tuple(d.edges for d in cg.dimensions)` |
| `RegularChunkGrid(chunk_shape=...)` | `ChunkGrid.from_regular(shape, chunks)` |
| `RectilinearChunkGrid(chunk_shapes=...)` | `ChunkGrid.from_rectilinear(edges, shape)` |
| Feature detection via class import | Version check or `hasattr(ChunkGrid, 'is_regular')` |

**[xarray#10880](https://github.com/pydata/xarray/pull/10880):** Replace `isinstance` checks with `.is_regular`. Write path simplifies with `chunks=[[...]]` API. ~1–2 days.

**[VirtualiZarr#877](https://github.com/zarr-developers/VirtualiZarr/pull/877):** Drop vendored `_is_nested_sequence`. Replace `isinstance` checks. ~1–2 days.

**[Icechunk#1338](https://github.com/earth-mover/icechunk/issues/1338):** Minimal impact — format changes driven by spec, not class hierarchy.

**[cubed#876](https://github.com/cubed-dev/cubed/issues/876):** Switch store creation to `ChunkGrid` API. <1 day.

## Open questions

1. **Resize defaults:** When growing a regular array, should the default preserve regularity or transition to rectilinear?
2. **`ChunkSpec` complexity:** `ChunkSpec` carries both `slices` and `codec_shape`. Should the grid expose separate methods for codec vs data queries instead?
3. **`__getitem__` with slices:** Should `grid[0, :]` or `grid[0:3, :]` return a sub-grid or an iterator of `ChunkSpec`s?
4. **Rectilinear + sharding:** The current POC raises `ValueError` for rectilinear chunks with sharding. When should this be relaxed?

## Proofs of concepts

- Zarr-Python:
    - branch - https://github.com/maxrjones/zarr-python/tree/poc/unified-chunk-grid
    - diff - https://github.com/zarr-developers/zarr-python/compare/main...maxrjones:zarr-python:poc/unified-chunk-grid?expand=1
- Xarray:
    - branch - https://github.com/maxrjones/xarray/tree/poc/unified-zarr-chunk-grid
    - diff - https://github.com/pydata/xarray/compare/main...maxrjones:xarray:poc/unified-zarr-chunk-grid?expand=1
- VirtualiZarr:
    - branch - https://github.com/maxrjones/VirtualiZarr/tree/poc/unified-chunk-grid
    - diff - https://github.com/zarr-developers/VirtualiZarr/compare/main...maxrjones:VirtualiZarr:poc/unified-chunk-grid?expand=1
- Virtual TIFF:
    - branch - https://github.com/virtual-zarr/virtual-tiff/tree/poc/unified-chunk-grid
    - diff - https://github.com/virtual-zarr/virtual-tiff/compare/main...poc/unified-chunk-grid?expand=1
- Microbenchmarks:
    - https://github.com/maxrjones/zarr-chunk-grid-tests/tree/unified-chunk-grid
## Breaking POC into reviewable PRs

### PR 1: Per-dimension grid types and `ChunkSpec` (pure additions)

**Files**: `chunk_grids.py` (new types only)
**Scope**: Add `FixedDimension`, `VaryingDimension`, `DimensionGrid` protocol, `ChunkSpec`, and RLE helpers (`_expand_rle`, `_compress_rle`). Unit tests for these types. No existing code changes — purely additive.

### PR 2: Unified `ChunkGrid` class (replaces old hierarchy)

**Files**: `chunk_grids.py` (new `ChunkGrid` class + `RegularChunkGrid` compat wrapper)
**Scope**: New `ChunkGrid` with `from_regular`, `from_rectilinear`, `__getitem__`, `all_chunk_coords()` (no shape arg), `is_regular`, `chunk_shape`, `chunk_sizes`. Keep `RegularChunkGrid` as backwards-compat subclass. Add `parse_chunk_grid()`, `serialize_chunk_grid()`, `_infer_chunk_grid_name()`. Tests for the grid class itself.

### PR 3: Indexing generalization

**Files**: `indexing.py`
**Scope**: Refactor `IntDimIndexer`, `SliceDimIndexer`, `BoolArrayDimIndexer`, `BasicIndexer`, `OrthogonalIndexer`, `CoordinateIndexer` to accept `DimensionGrid` instead of `dim_chunk_len: int`. Replace `get_chunk_shape()` calls with `_get_dim_grids()`. Tests for indexing with both regular and rectilinear grids.

### PR 4: Metadata and array integration

**Files**: `metadata/v3.py`, `metadata/v2.py`, `array.py`, `group.py`, `api/synchronous.py`
**Scope**: Wire the new `ChunkGrid` into `ArrayV3Metadata` (add `chunk_grid_name`, use `serialize_chunk_grid` in `to_dict`, use `parse_chunk_grid` in constructor). Update `init_array`/`create_array` to accept rectilinear chunks. Update `_resize` to guard against rectilinear grids.

### PR 5: Sharding codec compatibility

**Files**: `codecs/sharding.py`
**Scope**: Update `ShardingCodec.validate` to handle rectilinear outer grids (validate every chunk is divisible). Replace `RegularChunkGrid(chunk_shape=...)` calls with `ChunkGrid.from_regular(...)`.

### PR 6: End-to-end tests

**Files**: `tests/test_unified_chunk_grid.py`, updates to `tests/test_array.py`, `tests/test_indexing.py`
**Scope**: Full integration tests — round-trip create/read/write with rectilinear arrays, serialization fidelity, hypothesis strategies.

## Notes

- PRs 1–2 are purely additive and low-risk.
- PR 3 is the biggest behavioral change.
- PRs 4–5 wire things together.
- PR 6 adds comprehensive test coverage.
- Each PR builds on the previous but is independently reviewable.
