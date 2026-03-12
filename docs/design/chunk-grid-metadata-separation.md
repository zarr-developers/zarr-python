# Chunk Grid: Metadata / Array Separation

**Related:**

- [chunk-grid.md](chunk-grid.md) (unified chunk grid design)

## Problem

`ArrayV3Metadata` stores `chunk_grid: ChunkGrid` — a behavioral object with index-to-chunk math, iteration, resize logic, and per-dimension grid types. Metadata should be a serializable data bag. The behavioral `ChunkGrid` carries runtime state (extent per dimension, prefix sums) that belongs on the array, not on the metadata record.

This coupling causes several issues:

1. **Metadata is not a simple DTO.** Constructing `ArrayV3Metadata` triggers `parse_chunk_grid()` which builds `FixedDimension`/`VaryingDimension` objects, computes prefix sums, and validates edge coverage. Metadata round-trips (`from_dict` → `to_dict`) pay this cost unnecessarily.
2. **Codec validation is misplaced.** `_validate_metadata()` calls `codec.validate(chunk_grid=...)`, passing the behavioral object. This conflates structural metadata validation with runtime array validation.
3. **`update_shape` lives on metadata.** Shape changes require constructing a new `ChunkGrid` with updated extents, serializing it back, and creating new metadata. The metadata layer shouldn't know about resize semantics.
4. **Redundant state.** `chunk_grid_name: str` exists solely to preserve serialization format because the `ChunkGrid` object doesn't carry its own name. With a plain dict, the name is just `chunk_grid["name"]`.

## Design

### Principles

1. **Metadata stores what's on disk.** `ArrayV3Metadata.chunk_grid` holds the JSON dict exactly as it appears in `zarr.json`. No parsing, no behavioral objects, no computed state.
2. **The array owns the behavioral grid.** `AsyncArray` constructs a `ChunkGrid` from the metadata dict + shape at init time. All chunk-related behavior (indexing, iteration, resize) goes through the array's grid.
3. **Codec validation happens at array construction.** The array has the full context (shape + grid + dtype) needed to validate codecs. Metadata validates only structural properties (correct keys, matching ndim).
4. **No signature changes to downstream consumers.** Indexers and codecs still receive `ChunkGrid`. Only the *source* of the grid changes — from `metadata.chunk_grid` to `array.chunk_grid`.

### Current architecture

```
ArrayV3Metadata
  ├── chunk_grid: ChunkGrid          ← behavioral object
  ├── chunk_grid_name: str           ← "regular" | "rectilinear"
  └── _validate_metadata()           → codec.validate(chunk_grid=...)
      get_chunk_spec()               → chunk_grid[coords]
      update_shape()                 → chunk_grid.update_shape()
      to_dict()                      → serialize_chunk_grid(chunk_grid, name)
      chunks                         → chunk_grid.chunk_shape
      shards                         → chunk_grid.chunk_shape

AsyncArray
  └── self.metadata.chunk_grid       ← delegates everything

Indexers
  └── __init__(chunk_grid: ChunkGrid)

Codec.validate()
  └── validate(shape, dtype, chunk_grid: ChunkGrid)
```

### Proposed architecture

```
ArrayV3Metadata
  ├── chunk_grid: dict[str, JSON]    ← plain serialized form
  └── _validate_metadata()           → structural checks only (ndim, required keys)
      to_dict()                      → return self.chunk_grid
      from_dict()                    → store dict as-is

AsyncArray
  ├── chunk_grid: ChunkGrid          ← behavioral object, constructed on init
  └── get_chunk_spec()               ← moved from metadata
      _validate_codecs()             ← moved from metadata
      update_shape()                 ← moved from metadata

Indexers                              ← unchanged
Codec.validate()                      ← unchanged
```

### Metadata changes

`ArrayV3Metadata` becomes simpler:

```python
@dataclass(frozen=True, kw_only=True)
class ArrayV3Metadata(Metadata):
    shape: tuple[int, ...]
    data_type: ZDType[TBaseDType, TBaseScalar]
    chunk_grid: dict[str, JSON]           # plain JSON dict
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: Any
    codecs: tuple[Codec, ...]
    # ...

    def __init__(self, *, chunk_grid, **kwargs):
        # Store the dict directly. Validate structure only:
        # - has "name" and "configuration" keys
        # - ndim matches shape
        # No parse_chunk_grid(), no ChunkGrid construction.
        object.__setattr__(self, "chunk_grid", chunk_grid)

    def _validate_metadata(self):
        # Structural: ndim from dict matches len(shape)
        # No codec validation — that moves to the array.
        name, config = parse_named_configuration(self.chunk_grid)
        if name == "regular":
            ndim = len(config["chunk_shape"])
        elif name == "rectilinear":
            ndim = len(config["chunk_shapes"])
        if ndim != len(self.shape):
            raise ValueError(...)

    @property
    def chunks(self) -> tuple[int, ...]:
        name, config = parse_named_configuration(self.chunk_grid)
        if name == "regular":
            return tuple(config["chunk_shape"])
        raise NotImplementedError(...)

    def to_dict(self):
        d = super().to_dict()
        # chunk_grid is already a dict — no serialize_chunk_grid() needed
        return d
```

The `chunk_grid_name` field is removed. Round-trip fidelity is preserved because the original dict is stored verbatim.

### Array changes

`AsyncArray` constructs the behavioral `ChunkGrid` and owns all chunk-related operations:

```python
@dataclass(frozen=True)
class AsyncArray:
    metadata: ArrayV2Metadata | ArrayV3Metadata
    # ...

    def __init__(self, metadata, store_path, config):
        # ... existing init ...
        chunk_grid = parse_chunk_grid(metadata.chunk_grid, metadata.shape)
        object.__setattr__(self, "_chunk_grid", chunk_grid)
        # Codec validation moves here:
        self._validate_codecs()

    @property
    def chunk_grid(self) -> ChunkGrid:
        return self._chunk_grid

    def get_chunk_spec(self, chunk_coords, array_config, prototype) -> ArraySpec:
        spec = self.chunk_grid[chunk_coords]
        if spec is None:
            raise ValueError(...)
        return ArraySpec(shape=spec.codec_shape, ...)

    def _validate_codecs(self) -> None:
        for codec in self.metadata.codecs:
            codec.validate(
                shape=self.metadata.shape,
                dtype=self.metadata.data_type,
                chunk_grid=self.chunk_grid,
            )
```

For resize, the array constructs a new `ChunkGrid` with the new shape. For regular grids, the metadata dict doesn't change on resize — only the extent changes, which is runtime state not serialized in the chunk grid JSON. The array rebuilds its `ChunkGrid` with the new shape:

```python
async def _resize(self, new_shape):
    new_grid = self.chunk_grid.update_shape(new_shape)
    # For regular grids, metadata.chunk_grid dict stays the same.
    # For rectilinear grids that grew/shrank, serialize back:
    new_chunk_grid_dict = serialize_chunk_grid(new_grid, self.metadata.chunk_grid["name"])
    new_metadata = replace(self.metadata, shape=new_shape, chunk_grid=new_chunk_grid_dict)
    # ...
```

### V2 metadata

`ArrayV2Metadata` already stores `chunks: tuple[int, ...]` as plain data. Its `chunk_grid` property (which constructs a `ChunkGrid`) is removed. The array handles construction for both V2 and V3:

```python
@property
def chunk_grid(self) -> ChunkGrid:
    if isinstance(self.metadata, ArrayV2Metadata):
        return ChunkGrid.from_regular(self.metadata.shape, self.metadata.chunks)
    return parse_chunk_grid(self.metadata.chunk_grid, self.metadata.shape)
```

### Call site migration

All `self.metadata.chunk_grid` references in `array.py` (~30 sites) change to `self.chunk_grid`:

```python
# Before
indexer = BasicIndexer(selection, self.shape, self.metadata.chunk_grid)

# After
indexer = BasicIndexer(selection, self.shape, self.chunk_grid)
```

Indexers, codecs, and the codec pipeline are unchanged — they still receive `ChunkGrid` as a parameter. Only the *source* changes.

### What does NOT change

| Component | Status |
|---|---|
| `ChunkGrid` class | Unchanged — all behavior stays |
| `FixedDimension`, `VaryingDimension` | Unchanged |
| `parse_chunk_grid()`, `serialize_chunk_grid()` | Unchanged — called from array instead of metadata |
| Indexer classes | Unchanged — still receive `ChunkGrid` |
| Codec `validate()` signature | Unchanged — still receives `ChunkGrid` |
| On-disk format | No spec change |

## Design decisions

### Why not keep ChunkGrid on metadata with a lazy property?

A lazy `@cached_property` on metadata would defer the cost but not fix the fundamental issue: metadata would still own behavioral state. Resize logic, codec validation, and `get_chunk_spec` would remain on metadata. The goal is a clean separation — metadata is data, the array is behavior.

### Why move codec validation to the array?

Codec validation needs the behavioral `ChunkGrid` (sharding checks divisibility, iterates chunk coords). With a plain dict on metadata, the array is the first place where a `ChunkGrid` exists. Validating there is natural — the array is the runtime boundary where all pieces (shape, dtype, grid, codecs) come together.

This means `ArrayV3Metadata` can be constructed with an invalid codec/grid combination without error. This is acceptable: metadata is a data transfer object. Validation at the array boundary catches errors at the same point users interact with the data.

### Why store the raw dict instead of a TypedDict or NamedTuple?

The dict is exactly what's in `zarr.json`. Storing it verbatim gives:
- Zero-cost round-trips (`to_dict` returns it as-is)
- No `chunk_grid_name` field needed (it's `chunk_grid["name"]`)
- Forward compatibility with unknown chunk grid types (metadata can store and round-trip grids it doesn't understand)

### How does resize work with a plain dict?

For regular grids, the chunk grid JSON doesn't change on resize — `{"name": "regular", "configuration": {"chunk_shape": [10, 20]}}` is the same regardless of array shape. The extent is runtime state derived from `shape`. The array rebuilds its `ChunkGrid` with the new shape.

For rectilinear grids, resize may add or remove chunks. The array resizes the `ChunkGrid`, serializes it back to a dict via `serialize_chunk_grid()`, and creates new metadata with the updated dict.

## Migration

### PR 1: Add `chunk_grid` property to `AsyncArray` (non-breaking)

**Files:** `array.py`
**Scope:** Add `chunk_grid` property that delegates to `self.metadata.chunk_grid`. Migrate all `self.metadata.chunk_grid` references in `array.py` to `self.chunk_grid`. Purely mechanical, no behavioral change.

### PR 2: Move codec validation to array

**Files:** `v3.py`, `array.py`
**Scope:** Remove `codec.validate()` calls from `ArrayV3Metadata._validate_metadata()`. Add `_validate_codecs()` to `AsyncArray.__init__`. Move `get_chunk_spec()` from metadata to array. The codec `validate()` signature is unchanged.

### PR 3: Make metadata chunk_grid a plain dict

**Files:** `v3.py`, `v2.py`, `array.py`
**Scope:** Replace `chunk_grid: ChunkGrid` with `chunk_grid: dict[str, JSON]` on `ArrayV3Metadata`. Remove `chunk_grid_name` field. Update `__init__`, `from_dict`, `to_dict`, `_validate_metadata`, `chunks`, `shards`. Construct `ChunkGrid` in `AsyncArray.__init__` via `parse_chunk_grid()`. Remove `chunk_grid` property from `ArrayV2Metadata`. Update `update_shape` / resize flow.

### PR 4: Update tests

**Files:** `tests/`
**Scope:** Update tests that construct `ArrayV3Metadata` directly and access `.chunk_grid` as a `ChunkGrid`. Tests that go through `Array` / `AsyncArray` should mostly work unchanged.

## Open questions

1. **Convenience properties on metadata.** Should `ArrayV3Metadata` expose `chunk_shape` parsed from the raw dict? Or should all chunk access go through the array? Exposing it avoids constructing a full `ChunkGrid` for simple queries, but adds dict-parsing logic to metadata.
2. **Downstream breakage.** Code that accesses `metadata.chunk_grid` as a `ChunkGrid` (e.g., xarray, VirtualiZarr) will break. Migration path: `array.chunk_grid` for behavioral access, `metadata.chunk_grid` for the raw dict. Downstream PRs needed for xarray, VirtualiZarr, Icechunk.
3. **Rectilinear resize serialization.** When a rectilinear array resizes, the array must serialize the updated `ChunkGrid` back to a dict. Should this use `serialize_chunk_grid()` (which applies RLE compression), or should the array manipulate the dict directly? The former is cleaner; the latter avoids a round-trip through `ChunkGrid`.
