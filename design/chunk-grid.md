# Unified Chunk Grid

Version: 6

Design document for adding rectilinear (variable) chunk grid support to **zarr-python**, conforming to the [rectilinear chunk grid extension spec](https://github.com/zarr-developers/zarr-extensions/pull/25).

**Related:**

- [#3750](https://github.com/zarr-developers/zarr-python/issues/3750) (single ChunkGrid proposal)
- [#3534](https://github.com/zarr-developers/zarr-python/pull/3534) (rectilinear implementation)
- [#3735](https://github.com/zarr-developers/zarr-python/pull/3735) (chunk grid module/registry)
- [ZEP0003](https://github.com/zarr-developers/zeps/blob/main/draft/ZEP0003.md) (variable chunking spec)
- [zarr-specs#370](https://github.com/zarr-developers/zarr-specs/pull/370) (sharding v1.1: non-divisible subchunks)
- [zarr-extensions#25](https://github.com/zarr-developers/zarr-extensions/pull/25) (rectilinear extension)
- [zarr-extensions#34](https://github.com/zarr-developers/zarr-extensions/issues/34) (sharding + rectilinear)

## Problem

Chunk grids form a hierarchy — the rectilinear grid is strictly more general than the regular grid. Any regular grid is expressible as a rectilinear grid. There is no known chunk grid that is both (a) more general than rectilinear and (b) retains the axis-aligned tessellation properties Zarr assumes. All known grids are special cases:

| Grid type | Description | Example |
|---|---|---|
| Regular | Uniform chunk size, boundary chunks padded with fill_value | `[10, 10, 10, 10]` |
| Regular-bounded (zarrs) | Uniform chunk size, boundary chunks trimmed to array extent | `[10, 10, 10, 5]` |
| HPC boundary-padded | Regular interior, larger boundary chunks ([VirtualiZarr#217](https://github.com/zarr-developers/VirtualiZarr/issues/217)) | `[10, 8, 8, 8, 10]` |
| Fully variable | Arbitrary per-chunk sizes | `[5, 12, 3, 20]` |

Prior iterations on the chunk grid design were based on the Zarr V3 spec's definition of chunk grids as an extension point alongside codecs, dtypes, etc. Therefore, we started designing the chunk grid implementation following a similar registry-based approach. However, in practice chunk grids are fundamentally different than codecs. Codecs are independent; supporting `zstd` tells you nothing about `gzip`. Chunk grids are not: every regular grid is a valid rectilinear grid. A registry-based plugin system makes sense for codecs but adds complexity without clear benefit for chunk grids. Here we start from some basic goals and propose a more fitting design for supporting different chunk grids in zarr-python.

## Goals

1. **Follow the zarr extension proposal.** The implementation should conform to the [rectilinear chunk grid spec](https://github.com/zarr-developers/zarr-extensions/pull/25), not innovate on the metadata format.
2. **Minimize changes to the public API.** Users creating regular arrays should see no difference. Rectilinear is additive.
3. **Maintain backwards compatibility.** Existing code using `.chunks`, `isinstance` checks, or importing `RegularChunkGrid`/`RectilinearChunkGrid` from `zarr.core.chunk_grids` should continue to work (with deprecation warnings where appropriate).
4. **Design for future iteration.** The internal architecture should allow refactoring (e.g., metadata/array separation, new dimension types) without breaking the public API.
5. **Minimize downstream changes.** xarray, VirtualiZarr, Icechunk, Cubed, etc. should need minimal updates.
6. **Minimize time to stable release.** Ship behind a feature flag, stabilize through real-world usage, promote to stable API.
7. **The new API should be useful.** `read_chunk_sizes`/`write_chunk_sizes`, `ChunkGrid.__getitem__`, `is_regular` — these should solve real problems, not just expose internals.
8. **Extensible for other serialization structures.** The per-dimension design should support future encodings (tile, temporal) without changes to indexing or codecs.

## Design

### Design choices

1. **A chunk grid is a concrete arrangement of chunks.** Not an abstract tiling pattern. This means that the chunk grid is bound to specific array dimensions, which enables the chunk grid to answer any question about any chunk (offset, size, count) without external parameters.
2. **One implementation, multiple serialization forms.** A single `ChunkGrid` class handles all chunking logic. The serialization format (`"regular"` vs `"rectilinear"`) is chosen by the metadata layer, not the grid.
3. **No chunk grid registry.** Simple name-based dispatch in the metadata layer's `parse_chunk_grid()`.
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
            return 0
        return ceildiv(self.extent, self.size)

    def index_to_chunk(self, idx: int) -> int:
        return idx // self.size                                # raises IndexError if OOB
    def chunk_offset(self, chunk_ix: int) -> int:
        return chunk_ix * self.size                            # raises IndexError if OOB
    def chunk_size(self, chunk_ix: int) -> int:
        return self.size                                       # always uniform; raises IndexError if OOB
    def data_size(self, chunk_ix: int) -> int:
        return max(0, min(self.size, self.extent - chunk_ix * self.size))  # raises IndexError if OOB
    @property
    def unique_edge_lengths(self) -> Iterable[int]:
        return (self.size,)                                    # O(1)
    def indices_to_chunks(self, indices: NDArray) -> NDArray:
        return indices // self.size
    def with_extent(self, new_extent: int) -> FixedDimension:
        return FixedDimension(size=self.size, extent=new_extent)
    def resize(self, new_extent: int) -> FixedDimension:
        return FixedDimension(size=self.size, extent=new_extent)

@dataclass(frozen=True)
class VaryingDimension:
    """Explicit per-chunk sizes. The last chunk may extend past the array
    extent (extent < sum(edges)), in which case data_size clips to the
    valid region while chunk_size returns the full edge length for codec
    processing. This underflow is allowed to match how regular grids
    handle boundary chunks, and to support shrinking an array without
    rewriting chunk edges (the spec allows trailing edges beyond the extent)."""
    edges: tuple[int, ...]           # per-chunk edge lengths (all > 0)
    cumulative: tuple[int, ...]      # prefix sums for O(log n) lookup
    extent: int                      # array dimension length (may be < sum(edges))

    def __init__(self, edges: Sequence[int], extent: int) -> None:
        # validates edges non-empty, all > 0, extent >= 0, extent <= sum(edges)
        # computes cumulative via itertools.accumulate
        # uses object.__setattr__ for frozen dataclass

    @property
    def nchunks(self) -> int:
        # number of chunks that overlap [0, extent)
        if extent == 0:
            return 0
        return bisect.bisect_left(self.cumulative, extent) + 1

    @property
    def ngridcells(self) -> int:
        return len(self.edges)

    def index_to_chunk(self, idx: int) -> int:
        return bisect.bisect_right(self.cumulative, idx)       # raises IndexError if OOB
    def chunk_offset(self, chunk_ix: int) -> int:
        return self.cumulative[chunk_ix - 1] if chunk_ix > 0 else 0  # raises IndexError if OOB
    def chunk_size(self, chunk_ix: int) -> int:
        return self.edges[chunk_ix]                            # raises IndexError if OOB
    def data_size(self, chunk_ix: int) -> int:
        offset = self.chunk_offset(chunk_ix)
        return max(0, min(self.edges[chunk_ix], self.extent - offset))  # raises IndexError if OOB
    @property
    def unique_edge_lengths(self) -> Iterable[int]:
        # lazy generator: yields unseen values, short-circuits deduplication
    def indices_to_chunks(self, indices: NDArray) -> NDArray:
        return np.searchsorted(self.cumulative, indices, side='right')
    def with_extent(self, new_extent: int) -> VaryingDimension:
        # validates cumulative[-1] >= new_extent (O(1)), re-binds extent
        return VaryingDimension(self.edges, extent=new_extent)
    def resize(self, new_extent: int) -> VaryingDimension:
        # grow past edge sum: append chunk of size (new_extent - sum(edges))
        # shrink or grow within edge sum: preserve all edges, re-bind extent
```

Both types implement the `DimensionGrid` protocol: `nchunks`, `extent`, `index_to_chunk`, `chunk_offset`, `chunk_size`, `data_size`, `indices_to_chunks`, `unique_edge_lengths`, `with_extent`, `resize`. Memory usage scales with the number of *varying* dimensions, not total chunks.

All per-chunk methods (`chunk_offset`, `chunk_size`, `data_size`) raise `IndexError` for out-of-bounds chunk indices, providing consistent fail-fast behavior across both dimension types.

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
    def ngridcells(self) -> int: ...
    @property
    def extent(self) -> int: ...
    def index_to_chunk(self, idx: int) -> int: ...
    def chunk_offset(self, chunk_ix: int) -> int: ...     # raises IndexError if OOB
    def chunk_size(self, chunk_ix: int) -> int: ...        # raises IndexError if OOB
    def data_size(self, chunk_ix: int) -> int: ...         # raises IndexError if OOB
    def indices_to_chunks(self, indices: NDArray[np.intp]) -> NDArray[np.intp]: ...
    @property
    def unique_edge_lengths(self) -> Iterable[int]: ...
    def with_extent(self, new_extent: int) -> DimensionGrid: ...
    def resize(self, new_extent: int) -> DimensionGrid: ...
```

The protocol is `@runtime_checkable`, enabling polymorphic handling of both dimension types without `isinstance` checks.

`nchunks` and `ngridcells` differ when `extent < sum(edges)`: `nchunks` counts only chunks that overlap `[0, extent)`, while `ngridcells` counts total defined grid cells (i.e., `len(edges)`). For `FixedDimension`, both are equal. For `VaryingDimension`, they differ after a resize that shrinks the extent below the edge sum.

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
grid = arr._chunk_grid            # ChunkGrid (bound to array shape)
grid.grid_shape                   # (10, 10) — number of chunks per dimension
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

# .read_chunk_sizes / .write_chunk_sizes: works for all grids (dask-style)
arr.write_chunk_sizes              # ((10, 10, ..., 10), (20, 20, ..., 20))
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

`from_sizes` requires `array_shape`, binding the extent per dimension at construction time. This is a core design choice: a chunk grid is a concrete arrangement for a specific array, not an abstract tiling pattern.

```python
# Regular grid — all FixedDimension
grid = ChunkGrid.from_sizes(array_shape=(100, 200), chunk_sizes=(10, 20))

# Rectilinear grid — extent = sum(edges) when shape matches
grid = ChunkGrid.from_sizes(array_shape=(60, 100), chunk_sizes=[[10, 20, 30], [25, 25, 25, 25]])

# Rectilinear grid with boundary clipping — last chunk extends past array extent
# e.g., shape=(55, 90) but edges sum to (60, 100): data_size clips at extent
grid = ChunkGrid.from_sizes(array_shape=(55, 90), chunk_sizes=[[10, 20, 30], [25, 25, 25, 25]])

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

Both names deserialize to the same `ChunkGrid` class. The serialized form does not include the array extent — that comes from `shape` in array metadata and is combined with the chunk grid when constructing a `ChunkGrid` via `ChunkGrid.from_metadata()`.

**The `ChunkGrid` does not serialize itself.** The format choice (`"regular"` vs `"rectilinear"`) belongs to `ArrayV3Metadata`. Serialization and deserialization are handled by the metadata-layer chunk grid classes (`RegularChunkGridMetadata` and `RectilinearChunkGridMetadata` in `metadata/v3.py`), which provide `to_dict()` and `from_dict()` methods.

For `create_array`, the format is inferred from the `chunks` argument: a flat tuple produces `"regular"`, a nested list produces `"rectilinear"`. The `_is_rectilinear_chunks()` helper detects nested sequences like `[[10, 20], [5, 5]]`.

##### Rectilinear spec compliance

The rectilinear format requires `"kind": "inline"` (validated by `validate_rectilinear_kind()`). Per the spec, each element of `chunk_shapes` can be:

- A bare integer `m`: repeated until `sum >= array_extent`
- A list of bare integers: explicit per-chunk sizes
- A mixed array of bare integers and `[value, count]` RLE pairs

RLE compression is used when serializing: runs of identical sizes become `[value, count]` pairs, singletons stay as bare integers.

```python
# compress_rle([10, 10, 10, 5]) -> [[10, 3], 5]
# expand_rle([[10, 3], 5])      -> [10, 10, 10, 5]
```

For a single-element `chunk_shapes` tuple like `(10,)`, `RectilinearChunkGridMetadata.to_dict()` serializes it as a bare integer `10`. Per the rectilinear spec, a bare integer is repeated until the sum >= extent, preserving the full codec buffer size for boundary chunks.

**Zero-extent handling:** Regular grids serialize zero-extent dimensions without issue (the format encodes only `chunk_shape`, no edges). Rectilinear grids cannot represent zero-extent dimensions because the spec requires at least one positive-integer edge length per axis.

#### read_chunk_sizes / write_chunk_sizes

The `read_chunk_sizes` and `write_chunk_sizes` properties provide universal access to per-dimension chunk data sizes, matching the dask `Array.chunks` convention. They work for both regular and rectilinear grids:

- `write_chunk_sizes`: always returns outer (storage) chunk sizes
- `read_chunk_sizes`: returns inner chunk sizes when sharding is used, otherwise same as `write_chunk_sizes`

```python
>>> arr = zarr.create_array(store, shape=(100, 80), chunks=(30, 40))
>>> arr.write_chunk_sizes
((30, 30, 30, 10), (40, 40))

>>> arr = zarr.create_array(store, shape=(60, 100), chunks=[[10, 20, 30], [50, 50]])
>>> arr.write_chunk_sizes
((10, 20, 30), (50, 50))
```

The underlying `ChunkGrid.chunk_sizes` property (on the grid, not the array) returns the same as `write_chunk_sizes`.

#### Resize

```python
arr.resize((80, 100))       # re-binds extent; FixedDimension stays fixed
arr.resize((200, 100))      # VaryingDimension grows by appending a new chunk
arr.resize((30, 100))       # VaryingDimension shrinks: preserves all edges, re-binds extent
```

Resize uses `ChunkGrid.update_shape(new_shape)`, which delegates to each dimension's `.resize()` method:
- `FixedDimension.resize()`: simply re-binds the extent (identical to `with_extent`)
- `VaryingDimension.resize()`: grow past `sum(edges)` appends a chunk covering the gap; shrink or grow within `sum(edges)` preserves all edges and re-binds the extent (the spec allows trailing edges beyond the array extent)

**Known limitation (deferred):** When growing a `VaryingDimension`, the current implementation always appends a single chunk covering the new region. For example, `[10, 10, 10]` resized from 30 to 45 produces `[10, 10, 10, 15]` instead of the more natural `[10, 10, 10, 10, 10]`. A future improvement should add an optional `chunks` parameter to `resize()` that controls how the new region is partitioned, with a sane default (e.g., repeating the last chunk size). This is safely deferrable because:
- `FixedDimension` already handles resize correctly (regular grids stay regular)
- The single-chunk default produces valid state, just suboptimal chunk layout
- Rectilinear arrays are behind an experimental feature flag
- Adding an optional parameter is backwards-compatible

Open design questions for the `chunks` parameter:
- Does it describe the new region only, or the entire post-resize array?
- Must the overlapping portion agree with existing chunks (no rechunking)?
- What is the type? Same as `chunks` in `create_array`?

#### from_array

The `from_array()` function handles both regular and rectilinear source arrays:

```python
src = zarr.create_array(store, shape=(60, 100), chunks=[[10, 20, 30], [50, 50]])
new = zarr.from_array(data=src, store=new_store, chunks="keep")
# Preserves rectilinear structure: new.write_chunk_sizes == ((10, 20, 30), (50, 50))
```

When `chunks="keep"`, the logic checks `data._chunk_grid.is_regular`:
- Regular: extracts `data.chunks` (flat tuple) and preserves shards
- Rectilinear: extracts `data.write_chunk_sizes` (nested tuples) and forces shards to None

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
    spec = self._chunk_grid[chunk_coords]
    return ArraySpec(shape=spec.codec_shape, ...)
```

Note `spec.codec_shape`, not `spec.shape`. For regular grids, `codec_shape` is uniform (preserving current behavior). The boundary clipping flow is unchanged:

```
Write: user data → pad to codec_shape with fill_value → encode → store
Read:  store → decode to codec_shape → slice via chunk_selection → user data
```

### Sharding

The `ShardingCodec` constructs a `ChunkGrid` per shard using the shard shape as extent and the subchunk shape as `FixedDimension`. Each shard is self-contained — it doesn't need to know whether the outer grid is regular or rectilinear. Validation checks that every unique edge length per dimension is divisible by the inner chunk size, using `dim.unique_edge_lengths` for efficient polymorphic iteration (O(1) for fixed dimensions, lazy-deduplicated for varying).

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
| `arr.chunks` | Retained for regular; `arr.read_chunk_sizes`/`arr.write_chunk_sizes` for general use |
| `get_chunk_shape(shape, coord)` | `grid[coord].codec_shape` or `grid[coord].shape` |

## Design decisions

### Why store the extent in ChunkGrid?

The chunk grid is a concrete arrangement, not an abstract tiling pattern. A finite collection naturally has an extent. Storing it enables `__getitem__`, eliminates `dim_len` parameters from every method, and makes the grid self-describing.

This does *not* mean `ArrayV3Metadata.shape` should delegate to the grid. The array shape remains an independent field in metadata. The extent is passed into the grid at construction time so it can answer boundary questions without external parameters. It is **not** serialized as part of the chunk grid JSON — it comes from the `shape` field in array metadata and is combined with the chunk grid configuration in `ChunkGrid.from_metadata()`.

### Why distinguish chunk_size from data_size?

A chunk in a regular grid has two sizes. `chunk_size` is the buffer size the codec processes — always `size` for `FixedDimension`, even at the boundary (padded with `fill_value`). `data_size` is the valid data region — clipped to `extent % size` at the boundary. The indexing layer uses `data_size` to generate `chunk_selection` slices.

This matches current zarr-python behavior and matters for:
1. **Backward compatibility.** Existing stores have boundary chunks encoded at full `chunk_shape`.
2. **Codec simplicity.** Codecs assume uniform input shapes for regular grids.
3. **Shard index correctness.** The index assumes `subchunk_dim`-sized entries.

For `VaryingDimension`, `chunk_size == data_size` when `extent == sum(edges)`. When `extent < sum(edges)` (e.g., after a resize that keeps the last chunk oversized), `data_size` clips the last chunk. This is the fundamental difference: `FixedDimension` has a declared size plus an extent that clips data; `VaryingDimension` has explicit sizes that normally *are* the extent but can also extend past it.

### Why not a chunk grid registry?

There is no known chunk grid outside the rectilinear family that retains the tessellation properties zarr-python assumes. A `match` on the grid name is sufficient.

### Why a single ChunkGrid class instead of RegularChunkGrid + RectilinearChunkGrid?

[Discussed in #3534.](https://github.com/zarr-developers/zarr-python/pull/3534) @d-v-b argued that `RegularChunkGrid` is unnecessary since rectilinear is more general; @dcherian argued that downstream libraries need a fast way to detect regular grids without inspecting potentially millions of chunk edges (see [xarray#9808](https://github.com/pydata/xarray/pull/9808)).

The resolution: a single `ChunkGrid` class with an `is_regular` property (O(1), cached at construction). This gives downstream code the fast-path detection @dcherian needed without the class hierarchy complexity @d-v-b wanted to avoid. The metadata document's `name` field (`"regular"` vs `"rectilinear"`) is also available for clients who inspect JSON directly.

A backwards-compatibility shim in `chunk_grids.py` preserves the old `RegularChunkGrid` / `RectilinearChunkGrid` import paths with deprecation warnings — see [Backwards compatibility](#backwards-compatibility).

### Why is ChunkGrid a concrete class instead of a Protocol/ABC?

The old design had `ChunkGrid` as an ABC with `RegularChunkGrid` as its only subclass. #3534 added `RectilinearChunkGrid` as a second subclass. This branch makes `ChunkGrid` a single concrete class instead, with separate metadata DTOs (`RegularChunkGridMetadata` and `RectilinearChunkGridMetadata` in `metadata/v3.py`) for serialization.

All known grids are special cases of rectilinear, so there's no need for a class hierarchy at the grid level. A `ChunkGrid` Protocol/ABC would mean every caller programs against an abstract interface and adding a grid type requires implementing ~15 methods. A single class is simpler.

Note: the *dimension* types (`FixedDimension`, `VaryingDimension`) do use a `DimensionGrid` Protocol — that's where the polymorphism lives. The grid-level class is concrete; the dimension-level types are polymorphic. If a genuinely novel grid type emerges that can't be expressed as a combination of per-dimension types, a grid-level Protocol can be extracted.

### Why `.chunks` raises for rectilinear grids

[Debated in #3534.](https://github.com/zarr-developers/zarr-python/pull/3534) @d-v-b suggested making `.chunks` return `tuple[tuple[int, ...], ...]` (dask-style) for all grids. @dcherian strongly objected: every downstream consumer expects `tuple[int, ...]`, and silently returning a different type would be worse than raising. Materializing O(10M) chunk edges into a Python tuple is also a real performance risk ([xarray#8902](https://github.com/pydata/xarray/issues/8902#issuecomment-2546127373)).

The resolution:
- `.chunks` is retained for regular grids (returns `tuple[int, ...]` as before)
- `.chunks` raises `NotImplementedError` for rectilinear grids with a message pointing to `.read_chunk_sizes`/`.write_chunk_sizes`
- `.read_chunk_sizes` and `.write_chunk_sizes` return `tuple[tuple[int, ...], ...]` (dask convention) for all grids

@maxrjones noted in review that deprecating `.chunks` for regular grids was not desirable. The current branch does not deprecate it.

### User control over grid serialization format

@d-v-b raised in #3534 that users need a way to say "these chunks are regular, but serialize as rectilinear" (e.g., to allow future append/extend workflows without format changes). @jhamman initially made nested-list input always produce `RectilinearChunkGridMetadata`.

The current branch resolves this via the metadata-layer chunk grid classes. When metadata is deserialized, the original name (from `{"name": "regular"}` or `{"name": "rectilinear"}`) determines which metadata class is instantiated (`RegularChunkGridMetadata` or `RectilinearChunkGridMetadata`), and that class handles serialization via `to_dict()`. Current inference behavior for `create_array`:
- `chunks=(10, 20)` (flat tuple) → infers `"regular"`
- `chunks=[[10, 20], [5, 5]]` (nested lists with varying sizes) → infers `"rectilinear"`
- `chunks=[[10, 10], [20, 20]]` (nested lists with uniform sizes) → `from_sizes` collapses to `FixedDimension`, so `is_regular=True` and infers `"regular"`

**Open question:** Should uniform nested lists preserve `"rectilinear"` to support future append workflows without a format change? This could be addressed by checking the input form before collapsing, or by allowing users to pass `chunk_grid_name` explicitly through the `create_array` API.

### Deferred: Tiled/periodic chunk patterns

[#3750 discussion](https://github.com/zarr-developers/zarr-python/issues/3750) identified periodic chunk patterns as a use case not efficiently served by RLE alone. RLE compresses runs of identical values (`np.repeat`), but periodic patterns like days-per-month (`[31, 28, 31, 30, ...]` repeated 30 years) need a tile encoding (`np.tile`). Real-world examples include:

- **Oceanographic models** (ROMS): HPC boundary-padded chunks like `[10, 8, 8, 8, 10]` — handled by RLE
- **Temporal axes**: days-per-month, hours-per-day — need tile encoding for compact metadata
- **Temporal-aware grids**: date/time-aware chunk grids that layer over other axes (raised by @LDeakin)

A `TiledDimension` prototype was built ([commit 9c0f582](https://github.com/maxrjones/zarr-python/commit/9c0f582f)) demonstrating that the per-dimension design supports this without changes to indexing or the codec pipeline. However, it was intentionally excluded from this release because:

1. **Metadata format must come first.** Tile encoding requires a new `kind` value in the rectilinear spec (currently only `"inline"` is defined). This should go through [zarr-extensions#25](https://github.com/zarr-developers/zarr-extensions/pull/25), not zarr-python unilaterally.
2. **The per-dimension architecture doesn't preclude it.** A future `TiledDimension` can implement the `DimensionGrid` protocol alongside `FixedDimension` and `VaryingDimension` with no changes to indexing, codecs, or the `ChunkGrid` class.
3. **RLE covers the MVP.** Most real-world variable chunk patterns (HPC boundaries, irregular partitions) are efficiently encoded with RLE. Tile encoding is an optimization for a specific (temporal) subset.

### Metadata / Array separation (partially implemented)

An earlier design doc proposed decoupling `ChunkGrid` (runtime) from `ArrayV3Metadata` (serialization), so that metadata would store only a plain dict and the array layer would construct the `ChunkGrid`.

The current implementation partially realizes this separation:

- **Metadata DTOs** (`RegularChunkGridMetadata`, `RectilinearChunkGridMetadata` in `metadata/v3.py`): Pure data, frozen dataclasses, no array shape. These live on `ArrayV3Metadata.chunk_grid` and represent only what goes into `zarr.json`.
- **`ChunkGrid`** (`chunk_grids.py`): Shape-bound, supports indexing, iteration, and chunk specs. Lives on `AsyncArray._chunk_grid`, constructed from metadata + `shape` via `ChunkGrid.from_metadata()`.

This means `ArrayV3Metadata.chunk_grid` is now a `ChunkGridMetadata` (the DTO union type), **not** the runtime `ChunkGrid`. Code that previously accessed runtime methods on `metadata.chunk_grid` (e.g., `all_chunk_coords()`, `__getitem__`) must now use the grid from the array layer instead.

The name controls serialization format; each metadata DTO class provides its own `to_dict()` method for serialization. The `ChunkGrid` handles all runtime queries.

## Prior art

**zarrs (Rust):** Three independent grid types behind a `ChunkGridTraits` trait. Key patterns adopted: Fixed vs Varying per dimension, prefix sums + binary search, `Option<T>` for out-of-bounds, `NonZeroU64` for chunk dimensions, separate subchunk grid per shard, array shape at construction.

**TensorStore (C++):** Stores only `chunk_shape` — boundary clipping via `valid_data_bounds` at query time. Both `RegularGridRef` and `IrregularGrid` internally. No registry.

## Migration

### Public API compatibility

The user-facing API is fully backward-compatible. Existing code that creates, opens, reads, and writes zarr arrays continues to work without changes:

- `zarr.create_array`, `zarr.open`, `zarr.open_array`, `zarr.open_group` -- unchanged signatures. The `chunks` parameter type is *widened* (now also accepts nested sequences for rectilinear grids), but all existing call patterns still work.
- `arr.chunks` -- returns `tuple[int, ...]` for regular arrays, same as before.
- `arr.shape`, `arr.dtype`, `arr.ndim`, `arr.shards` -- unchanged.
- Top-level `zarr` exports -- unchanged.
- Rectilinear chunks are gated behind `zarr.config.set({'array.rectilinear_chunks': True})`, so they cannot be created accidentally.

New additions (purely additive): `arr.read_chunk_sizes`, `arr.write_chunk_sizes`, `zarr.experimental.ChunkGrid`, `zarr.experimental.ChunkSpec`.

The breaking changes discussed below are confined to **internal modules** (`zarr.core.chunk_grids`, `zarr.core.metadata.v3`, `zarr.core.indexing`) that downstream libraries like cubed and VirtualiZarr access directly.

### Internal API compatibility trade-off analysis

This section analyzes the internal breaking changes from the metadata/array separation and evaluates two strategies: (A) add backward-compatibility shims in zarr-python, vs. (B) require downstream packages to update. The baseline is **no shims at all**.

#### What breaks without any shims

Three API changes affect downstream code:

1. **`RegularChunkGrid` class removed from `zarr.core.chunk_grids`.** On `main`, `RegularChunkGrid` is defined in `chunk_grids.py` as a `Metadata` subclass. This branch replaces it with `RegularChunkGridMetadata` in `metadata/v3.py`. Without a shim, `from zarr.core.chunk_grids import RegularChunkGrid` raises `ImportError`.

2. **`RegularChunkGrid` no longer available from `zarr.core.metadata.v3`.** On `main`, `v3.py` imports `RegularChunkGrid` from `chunk_grids.py` for internal use. VirtualiZarr imports it from this location (`from zarr.core.metadata.v3 import RegularChunkGrid`). Without the internal import, this raises `ImportError`.

3. **`OrthogonalIndexer` constructor expects `ChunkGrid`, not `RegularChunkGrid`/`RegularChunkGridMetadata`.** Even if the import shims above resolve to `RegularChunkGridMetadata`, the indexer constructors access `chunk_grid._dimensions`, which only exists on the runtime `ChunkGrid` class. Cubed constructs `OrthogonalIndexer(selection, shape, RegularChunkGrid(chunk_shape=chunks))` directly.

#### Downstream impact without shims

**VirtualiZarr** (5 line changes across 2 files):

```python
# manifests/array.py (line 6): import
- from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGrid
+ from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata

# manifests/array.py (line 53): isinstance check
- if not isinstance(_metadata.chunk_grid, RegularChunkGrid):
+ if not isinstance(_metadata.chunk_grid, RegularChunkGridMetadata):

# parsers/zarr.py (line 16): import
- from zarr.core.chunk_grids import RegularChunkGrid
+ from zarr.core.metadata.v3 import RegularChunkGridMetadata

# parsers/zarr.py (line 270): isinstance check
- if not isinstance(array_v3_metadata.chunk_grid, RegularChunkGrid):
+ if not isinstance(array_v3_metadata.chunk_grid, RegularChunkGridMetadata):

# parsers/zarr.py (line 390): cast
- cast(RegularChunkGrid, metadata.chunk_grid).chunk_shape
+ cast(RegularChunkGridMetadata, metadata.chunk_grid).chunk_shape
```

The `manifests/array.py` import is from `zarr.core.metadata.v3` (never a documented export; VirtualiZarr relied on a transitive import). The `parsers/zarr.py` import is from `zarr.core.chunk_grids` (the canonical location on `main`). Both are straightforward renames. The `.chunk_shape` attribute is unchanged on the new class.

If VirtualiZarr needs to support both old and new zarr-python, a version-conditional import adds ~5 more lines.

**Cubed** (3 line changes in 1 file):

```python
# core/ops.py (lines 626-631)
def _create_zarr_indexer(selection, shape, chunks):
    if zarr.__version__[0] == "3":
-       from zarr.core.chunk_grids import RegularChunkGrid
+       from zarr.core.chunk_grids import ChunkGrid
        from zarr.core.indexing import OrthogonalIndexer
-       return OrthogonalIndexer(selection, shape, RegularChunkGrid(chunk_shape=chunks))
+       return OrthogonalIndexer(selection, shape, ChunkGrid.from_sizes(shape, chunks))
```

Note that `ChunkGrid` is *not* a renamed class. `RegularChunkGrid(chunk_shape=chunks)` took only chunk sizes; `ChunkGrid.from_sizes(shape, chunks)` also requires the array shape. The `shape` parameter is already available at this call site.

If cubed needs to support both old and new zarr-python:

```python
def _create_zarr_indexer(selection, shape, chunks):
    if zarr.__version__[0] == "3":
        from zarr.core.indexing import OrthogonalIndexer
        try:
            from zarr.core.chunk_grids import ChunkGrid
            return OrthogonalIndexer(selection, shape, ChunkGrid.from_sizes(shape, chunks))
        except ImportError:
            from zarr.core.chunk_grids import RegularChunkGrid
            return OrthogonalIndexer(selection, shape, RegularChunkGrid(chunk_shape=chunks))
    else:
        from zarr.indexing import OrthogonalIndexer
        return OrthogonalIndexer(selection, ZarrArrayIndexingAdaptor(shape, chunks))
```

#### What shims can cover

**Shim 1: `__getattr__` in `chunk_grids.py`** (~15 lines)

Maps `RegularChunkGrid` to `RegularChunkGridMetadata` with a deprecation warning. Covers:
- The `from zarr.core.chunk_grids import RegularChunkGrid` import pattern (used by cubed and VirtualiZarr's `parsers/zarr.py`)
- `isinstance(x, RegularChunkGrid)` checks (because the name resolves to the actual class)
- `RegularChunkGrid(chunk_shape=(...))` construction (because `RegularChunkGridMetadata` accepts the same arguments)

Does **not** cover: passing the result to `OrthogonalIndexer`, because `RegularChunkGridMetadata` lacks `._dimensions`.

**Shim 2: `__getattr__` in `metadata/v3.py`** (~12 lines)

Same pattern, covers VirtualiZarr's import from `zarr.core.metadata.v3`. Mirrors Shim 1 for a different import path.

**Shim 3: Auto-coerce `ChunkGridMetadata` in indexer constructors** (~30 lines)

A helper function + 1-line insertion in each of `BasicIndexer`, `OrthogonalIndexer`, `CoordinateIndexer`, and `MaskIndexer`:

```python
def _resolve_chunk_grid(chunk_grid, shape):
    """Coerce ChunkGridMetadata to runtime ChunkGrid if needed."""
    from zarr.core.chunk_grids import ChunkGrid as _ChunkGrid
    from zarr.core.metadata.v3 import ChunkGridMetadata
    if isinstance(chunk_grid, _ChunkGrid):
        return chunk_grid
    if isinstance(chunk_grid, ChunkGridMetadata):
        warnings.warn(
            "Passing ChunkGridMetadata to indexers is deprecated. "
            "Use ChunkGrid.from_sizes() instead.",
            DeprecationWarning, stacklevel=2,
        )
        if hasattr(chunk_grid, "chunk_shape"):
            return _ChunkGrid.from_sizes(shape, tuple(chunk_grid.chunk_shape))
        return _ChunkGrid.from_sizes(shape, chunk_grid.chunk_shapes)
    raise TypeError(f"Expected ChunkGrid or ChunkGridMetadata, got {type(chunk_grid)}")
```

This covers cubed's `OrthogonalIndexer(selection, shape, RegularChunkGrid(...))` pattern end-to-end (combined with Shim 1).

#### Comparison

| | No shims | Shims 1+2 only | Shims 1+2+3 |
|---|---|---|---|
| **zarr-python additions** | 0 lines | ~27 lines | ~57 lines |
| **VirtualiZarr changes** | 5 lines | 0 lines | 0 lines |
| **Cubed changes** | 3 lines | 3 lines | 0 lines |
| **Maintenance burden** | None | Low (deprecation shims are well-understood) | Medium (indexer coercion blurs metadata/runtime boundary) |
| **API clarity** | Clean (metadata DTOs and runtime types are distinct) | Good (old names redirect to new names) | Weaker (indexers implicitly accept two type families) |

With Shims 1+2 only, VirtualiZarr's `manifests/array.py` import from `zarr.core.metadata.v3` is covered by Shim 2, and the `parsers/zarr.py` import from `zarr.core.chunk_grids` is covered by Shim 1. The `isinstance` checks work because both shims resolve to `RegularChunkGridMetadata`. The `cast` works because `.chunk_shape` is unchanged. So VirtualiZarr needs 0 changes with Shims 1+2. The 3 lines for cubed remain because Shim 1 resolves the import but `OrthogonalIndexer` still needs a runtime `ChunkGrid`.

### Downstream migration

Migration from `main` (where only `RegularChunkGrid` and the abstract `ChunkGrid` ABC exist):

| Old pattern (on `main`) | New pattern |
|---|---|
| `from zarr.core.chunk_grids import RegularChunkGrid` | `from zarr.core.metadata.v3 import RegularChunkGridMetadata` |
| `from zarr.core.chunk_grids import ChunkGrid` (ABC) | `from zarr.core.chunk_grids import ChunkGrid` (concrete class, different API) |
| `isinstance(cg, RegularChunkGrid)` | `isinstance(cg, RegularChunkGridMetadata)` or `grid.is_regular` on the runtime `ChunkGrid` |
| `cg.chunk_shape` on `RegularChunkGrid` | `cg.chunk_shape` on `RegularChunkGridMetadata` (unchanged) |
| `ChunkGrid.from_dict(data)` | `parse_chunk_grid(data)` from `zarr.core.metadata.v3` |
| `chunk_grid.all_chunk_coords(array_shape)` | `chunk_grid.all_chunk_coords()` (shape now stored in grid) |
| `chunk_grid.get_nchunks(array_shape)` | `chunk_grid.get_nchunks()` (shape now stored in grid) |

During the earlier [#3534](https://github.com/zarr-developers/zarr-python/pull/3534) effort (which used separate `RegularChunkGrid`/`RectilinearChunkGrid` classes), downstream PRs and issues were opened to explore compatibility:

- xarray ([#10880](https://github.com/pydata/xarray/pull/10880)), VirtualiZarr ([#877](https://github.com/zarr-developers/VirtualiZarr/pull/877)), Icechunk ([#1338](https://github.com/earth-mover/icechunk/issues/1338)), cubed ([#876](https://github.com/cubed-dev/cubed/issues/876))

These target #3534's API, not this branch's unified `ChunkGrid` design. New downstream POC branches for this design are linked in [Proofs of concepts](#proofs-of-concepts).

### Credits

This implementation builds on prior work:

- **[#3534](https://github.com/zarr-developers/zarr-python/pull/3534)** (@jhamman) — RLE helpers, validation logic, test cases, and the review discussion that shaped the architecture.
- **[#3737](https://github.com/zarr-developers/zarr-python/pull/3737)** — extent-in-grid idea (adopted per-dimension).
- **[#1483](https://github.com/zarr-developers/zarr-python/pull/1483)** — original variable chunking POC.
- **[#3736](https://github.com/zarr-developers/zarr-python/pull/3736)** — resolved by storing extent per-dimension.


## Open questions

1. **Resize defaults (deferred):** When growing a rectilinear array, should `resize()` accept an optional `chunks` parameter? See the [Resize section](#resize) for details and open design questions. Regular arrays already stay regular on resize.
2. **`ChunkSpec` complexity:** `ChunkSpec` carries both `slices` and `codec_shape`. Should the grid expose separate methods for codec vs data queries instead?
3. **`__getitem__` with slices:** Should `grid[0, :]` or `grid[0:3, :]` return a sub-grid or an iterator of `ChunkSpec`s?
4. **Uniform nested lists:** Should `chunks=[[10, 10], [20, 20]]` serialize as `"rectilinear"` (preserving user intent for future append) or `"regular"` (current behavior, collapses uniform edges)? See [User control over grid serialization format](#user-control-over-grid-serialization-format).
5. **`zarr.open` with rectilinear:** @tomwhite noted in #3534 that `zarr.open(mode="w")` doesn't support rectilinear chunks directly. This could be addressed in a follow-up.

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
- Cubed:
    - branch - https://github.com/maxrjones/cubed/tree/poc/unified-chunk-grid
- Microbenchmarks:
    - https://github.com/maxrjones/zarr-chunk-grid-tests/tree/unified-chunk-grid
