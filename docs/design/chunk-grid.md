# Prospectus: Unified Chunk Grid Design for zarr-python

Version: 3

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
4. **Fixed vs Varying per dimension.** `FixedDimension(size, extent)` for uniform chunks; `VaryingDimension(edges, cumulative)` for per-chunk edge lengths with precomputed prefix sums. Avoids expanding regular dimensions into lists of identical values.
5. **Transparent transitions.** Operations like `resize()` can move an array from regular to rectilinear chunking.

### Internal representation

```python
@dataclass(frozen=True)
class FixedDimension:
    """Uniform chunk size. Boundary chunks contain less data but are
    encoded at full size by the codec pipeline."""
    size: int            # chunk edge length (> 0)
    extent: int          # array dimension length

    @property
    def nchunks(self) -> int:
        return ceildiv(self.extent, self.size)

    def index_to_chunk(self, idx: int) -> int:
        return idx // self.size
    def chunk_offset(self, chunk_ix: int) -> int:
        return chunk_ix * self.size
    def chunk_size(self, chunk_ix: int) -> int:
        return self.size                                       # always uniform
    def data_size(self, chunk_ix: int) -> int:
        return min(self.size, self.extent - chunk_ix * self.size)  # clipped at extent
    def indices_to_chunks(self, indices: NDArray) -> NDArray:
        return indices // self.size

@dataclass(frozen=True)
class VaryingDimension:
    """Explicit per-chunk sizes. No padding — each edge length is
    both the codec size and the data size."""
    edges: tuple[int, ...]           # per-chunk edge lengths (all > 0)
    cumulative: tuple[int, ...]      # prefix sums for O(log n) lookup

    @property
    def nchunks(self) -> int:
        return len(self.edges)
    @property
    def extent(self) -> int:
        return self.cumulative[-1]

    def index_to_chunk(self, idx: int) -> int:
        return bisect.bisect_right(self.cumulative, idx)
    def chunk_offset(self, chunk_ix: int) -> int:
        return self.cumulative[chunk_ix - 1] if chunk_ix > 0 else 0
    def chunk_size(self, chunk_ix: int) -> int:
        return self.edges[chunk_ix]
    def data_size(self, chunk_ix: int) -> int:
        return self.edges[chunk_ix]                            # same as chunk_size
    def indices_to_chunks(self, indices: NDArray) -> NDArray:
        return np.searchsorted(self.cumulative, indices, side='right')
```

Both types share a common interface: `nchunks`, `extent`, `index_to_chunk`, `chunk_offset`, `chunk_size`, `data_size`, `indices_to_chunks`. Memory usage scales with the number of *varying* dimensions, not total chunks.

The two size methods serve different consumers:

| Method | Returns | Consumer |
|---|---|---|
| `chunk_size` | Buffer size for codec processing | Codec pipeline (`ArraySpec.shape`) |
| `data_size` | Valid data region within the buffer | Indexing pipeline (`chunk_selection` slicing) |

For `FixedDimension`, these differ only at the boundary. For `VaryingDimension`, they are identical. This matches current zarr-python behavior: `get_chunk_spec` passes the full `chunk_shape` to the codec for all chunks, and the indexer generates a `chunk_selection` that clips the decoded buffer.

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

For interior chunks, `shape == codec_shape`. For boundary chunks of a regular grid, `codec_shape` is the full declared chunk size while `shape` is clipped. For rectilinear grids, `shape == codec_shape` always.

### API

```python
# Creating arrays
arr = zarr.create_array(shape=(100, 200), chunks=(10, 20))                          # regular
arr = zarr.create_array(shape=(60, 100), chunks=[[10, 20, 30], [25, 25, 25, 25]])   # rectilinear
arr = zarr.create_array(shape=(1000,), chunks=[[[100, 10]]])                        # RLE shorthand

# ChunkGrid as a collection
grid = arr.chunk_grid              # ChunkGrid instance
grid.shape                         # (10, 8) — number of chunks per dimension
grid.ndim                          # 2
grid.is_regular                    # True if all dimensions are Fixed

spec = grid[0, 1]                  # ChunkSpec for chunk at grid position (0, 1)
spec.slices                        # (slice(0, 10), slice(25, 50))
spec.shape                         # (10, 25) — data shape
spec.codec_shape                   # (10, 25) — same for interior chunks

boundary = grid[9, 0]             # boundary chunk (extent=95, size=10)
boundary.shape                     # (5, 25) — 5 elements of real data
boundary.codec_shape               # (10, 25) — codec sees full buffer

grid[99, 99]                       # None — out of bounds

for spec in grid:                  # iterate all chunks
    ...

# .chunks property: retained for regular grids, raises for rectilinear
arr.chunks                         # (10, 25)
```

`ChunkGrid.__getitem__` constructs `ChunkSpec` using `chunk_size` for `codec_shape` and `data_size` for `slices`:

```python
def __getitem__(self, coords: tuple[int, ...]) -> ChunkSpec | None:
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

#### Serialization

```python
# Regular grid:
{"name": "regular", "configuration": {"chunk_shape": [10, 20]}}

# Rectilinear grid (with RLE compression):
{"name": "rectilinear", "configuration": {"chunk_shapes": [[10, 20, 30], [[25, 4]]]}}
```

Both names deserialize to the same `ChunkGrid` class. The serialized form does not include the array extent — that comes from `shape` in array metadata and is passed to `parse_chunk_grid()` at construction time. For rectilinear grids, the extent is redundant (`sum(edges)`) and is validated for consistency.

**The `ChunkGrid` does not serialize itself.** The format choice (`"regular"` vs `"rectilinear"`) belongs to `ArrayV3Metadata`, which already knows how to produce its JSON document. The flow is always: metadata document → `ChunkGrid` (via `parse_chunk_grid`), never the reverse. The grid is a pure runtime computation object.

`ArrayV3Metadata` stores the chunk grid's JSON `name` from the original metadata document and uses it when serializing back. This gives round-trip fidelity for free — a store written as rectilinear with uniform edges stays rectilinear.

The only place where a user needs to choose the format is when creating new metadata. For `create_array`, the format is inferred from the `chunks` argument: a flat tuple produces `"regular"`, a nested list produces `"rectilinear"`. For `resize`, the format can be specified explicitly via `chunk_grid_metadata`:

```python
arr.resize(
    (80, 100),
    chunks=[[10, 20, 30, 20], [25, 25, 25, 25]],
    chunk_grid_metadata="rectilinear",
)
```

`chunk_grid_metadata` is typed as `str`, not a closed literal — the Zarr V3 spec allows any registered chunk grid name. zarr-python supports `"regular"` and `"rectilinear"` natively; other names (e.g., zarrs' `"regular_bounded"`) would raise unless a handler is registered. If omitted, the format is inferred: `"rectilinear"` when chunks are non-uniform or explicitly nested, `"regular"` when chunks are a flat tuple and evenly divide the shape. Specifying `"regular"` when the chunks are non-uniform raises an error.

#### Resize

```python
arr.resize((80, 100))                                                # inferred rectilinear if not evenly divisible
arr.resize((80, 100), chunks=[[10, 20, 30, 20], [25, 25, 25, 25]])  # explicit chunks
arr.resize((70, 100))                                                # stays regular if divisible
arr.resize((100, 100), chunk_grid_metadata="rectilinear")            # force rectilinear metadata
```

Resize creates new `ArrayV3Metadata` (and thus a new `ChunkGrid`). Since resize always creates new metadata, `chunk_grid_metadata` is the natural place to choose the serialization format.

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

PR #3534 marks sharding as incompatible with rectilinear grids. This is unnecessary — sharding has three independent grid levels:

```
Level 1 — Outer chunk grid (shard boundaries): regular or rectilinear
Level 2 — Inner subchunk grid (within each shard): always regular
Level 3 — Shard index: ceil(shard_dim / subchunk_dim) entries per dimension
```

The `ShardingCodec` constructs a `ChunkGrid` per shard using the shard shape as extent and the subchunk shape as `FixedDimension`. Each shard is self-contained — it doesn't need to know whether the outer grid is regular or rectilinear.

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
| `arr.chunks` | Retained for regular; `arr.chunk_grid` for general use |
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

For `VaryingDimension`, `chunk_size == data_size` — no padding. This is the fundamental difference: `FixedDimension` has a declared size plus an extent that clips data; `VaryingDimension` has explicit sizes that *are* the extent.

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
   - `FixedDimension`, `VaryingDimension`, `ChunkSpec`, and `ChunkGrid` classes.
   - `parse_chunk_grid(metadata, array_shape)` with `"regular"` and `"rectilinear"` dispatch.
   - Port RLE helpers, `resolve_chunk_spec`, `ChunksLike`, and validation functions from #3534.
   - Refactor per-dimension indexers to accept `FixedDimension | VaryingDimension`.
   - Update `get_chunk_spec` to use `grid[chunk_coords].codec_shape`.
   - Add `arr.chunk_grid`. Keep `.chunks` for regular, raise for rectilinear.
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
| `RectilinearChunkGrid(chunk_shapes=...)` | `ChunkGrid.from_rectilinear(edges)` |
| Feature detection via class import | Version check or `hasattr(ChunkGrid, 'is_regular')` |

**[xarray#10880](https://github.com/pydata/xarray/pull/10880):** Replace `isinstance` checks with `.is_regular`. Write path simplifies with `chunks=[[...]]` API. ~1–2 days.

**[VirtualiZarr#877](https://github.com/zarr-developers/VirtualiZarr/pull/877):** Drop vendored `_is_nested_sequence`. Replace `isinstance` checks. ~1–2 days.

**[Icechunk#1338](https://github.com/earth-mover/icechunk/issues/1338):** Minimal impact — format changes driven by spec, not class hierarchy.

**[cubed#876](https://github.com/cubed-dev/cubed/issues/876):** Switch store creation to `ChunkGrid` API. <1 day.

## Open questions

1. **RLE in the Python API:** Should users pass RLE-encoded chunk specs directly, or only expanded lists?
2. **Resize defaults:** When growing a regular array, should the default preserve regularity or transition to rectilinear?
3. **`ChunkSpec` complexity:** `ChunkSpec` carries both `slices` and `codec_shape`. Should the grid expose separate methods for codec vs data queries instead?
4. **`__getitem__` with slices:** Should `grid[0, :]` or `grid[0:3, :]` return a sub-grid or an iterator of `ChunkSpec`s?

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
**Scope**: New `ChunkGrid` with `from_regular`, `from_rectilinear`, `__getitem__`, `all_chunk_coords()` (no shape arg), `is_regular`, `chunk_shape`. Keep `RegularChunkGrid` as backwards-compat subclass. Add `parse_chunk_grid()`, `serialize_chunk_grid()`, `_infer_chunk_grid_name()`. Tests for the grid class itself.

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