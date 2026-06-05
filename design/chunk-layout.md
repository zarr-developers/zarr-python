# Public Chunk Layout

Version: 1

Design document for adding a public, typed chunk-structure introspection API to **zarr-python**: a `ChunkLayout` object that distills the chunk grid metadata and sharding codec configuration of an array.

**Related:**

- [#4035](https://github.com/zarr-developers/zarr-python/issues/4035) (public API for determining chunk grid type)
- [#4036](https://github.com/zarr-developers/zarr-python/issues/4036) (public API for determining if an array is sharded)
- [API-surface evaluation](https://hackmd.io/p9pU24IQR6uA9tg6c0wMhw) (gaps in the chunk/shard accessor family)
- [chunk-grid.md](./chunk-grid.md) (unified chunk grid design; this document builds on it)
- [zarr-extensions#34](https://github.com/zarr-developers/zarr-extensions/issues/34) (sharding + rectilinear)
- [pydata/xarray#11279](https://github.com/pydata/xarray/pull/11279) (downstream consumer needing grid-kind detection)

**Scope:** additive only. This design introduces a new introspection object and leaves `Array.chunks`, `Array.shards`, `Array.read_chunk_sizes`, and `Array.write_chunk_sizes` exactly as they are. A more ambitious variant — making `ChunkLayout` the canonical chunk-structure API and deprecating `.chunks`/`.shards` on a schedule — was evaluated separately and is deliberately out of scope; if there is appetite for it, it can be proposed as a later version of this document (see the discussion in [#4036](https://github.com/zarr-developers/zarr-python/issues/4036)).

## Problem

zarr-python cannot currently answer two basic questions about an array without exception handling or imports from private modules:

1. **Is this array sharded?** ([#4036](https://github.com/zarr-developers/zarr-python/issues/4036)) — `.shards is not None` works for regular chunk grids, but `.shards` raises `NotImplementedError` for rectilinear grids, so sharded-rectilinear arrays require `try/except`.
2. **What kind of chunk grid does this array use?** ([#4035](https://github.com/zarr-developers/zarr-python/issues/4035)) — requires either catching `NotImplementedError` from `.chunks` or `isinstance(array.metadata.chunk_grid, RegularChunkGridMetadata)` with an import from `zarr.core.metadata.v3`, which is private.

The [HackMD API-surface evaluation](https://hackmd.io/p9pU24IQR6uA9tg6c0wMhw) identifies further gaps that the same object can close: `.chunks` raises for rectilinear-sharded arrays even though their inner chunks are regular, and no accessor reports *declared* (unclipped) chunk sizes for round-trip preservation.

In [#4036](https://github.com/zarr-developers/zarr-python/issues/4036), @d-v-b proposed exposing the internal `ChunkLayout` ([`chunk_grids.py:55`](https://github.com/zarr-developers/zarr-python/blob/fe229107f9915f05817f7a664d3550695ff9ca44/src/zarr/core/chunk_grids.py#L55)) — "a distillation of the chunk grid + codecs for an array" — after polish. This design is that polish.

## Goals

1. **Non-raising predicates.** A public way to answer "is this array sharded?" and "what grid kind is this?" for every present and future grid/sharding combination, without exceptions or private imports.
2. **One typed object.** A single distillation of chunk grid metadata + sharding codec configuration, usable by downstream libraries (xarray, dask, cubed) as their introspection surface.
3. **Forward compatibility.** Layouts that do not exist yet — rectilinear inner chunks, rectilinear shards, nested sharding, and chunk grid kinds not expressible as per-dimension edge lists — must slot in without breaking the API.
4. **Third-party sharding codecs.** Per @d-v-b on #4036, the design must accommodate new codecs that implement sharding, not hard-code `ShardingCodec`.

## Non-goals

- No change or deprecation of any existing accessor (`.chunks`, `.shards`, `.read_chunk_sizes`, `.write_chunk_sizes`).
- No serialization (`to_dict`/`from_dict`) — `ChunkLayout` is derived from metadata, never stored.
- No extent-bound queries (clipped sizes, chunk counts, index-to-chunk mapping). Those remain the job of the internal `ChunkGrid` ([chunk-grid.md](./chunk-grid.md)), which stays private in this design.
- No creation-API changes (`create_array(chunks=..., shards=...)` is untouched).

## Design

### Core principle: declared, extent-free, recursive, typed

`ChunkLayout` describes the *declared* chunk structure of an array — exactly what the metadata and codec configuration say, never clipped to the array shape. It is recursive: each level describes the structure *within one cell of the level above* (the top level describes structure within the whole array). The grid *kind* is encoded as the Python type, so new kinds can be added as new subclasses without disturbing the existing surface.

```python
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ChunkLayout:
    """Declared chunk structure of an array.

    A distillation of the chunk grid metadata and sharding codec
    configuration. Extent-free: sizes are as declared, never clipped
    to the array shape. Abstract — only subclasses are instantiated.
    """

    inner: ChunkLayout | None = None
    """Structure within each chunk of this level, or None if chunks are opaque (no sharding)."""

    def __post_init__(self) -> None:
        if type(self) is ChunkLayout:
            raise TypeError("ChunkLayout is abstract; instantiate a subclass")
        if self.inner is not None and self.inner.ndim != self.ndim:
            raise ValueError(
                f"inner layout has {self.inner.ndim} dimensions, expected {self.ndim}"
            )

    @property
    def ndim(self) -> int:
        raise NotImplementedError  # implemented by subclasses

    @property
    def is_sharded(self) -> bool:
        """True if chunks at this level have internal sub-chunk structure."""
        return self.inner is not None

    @property
    def levels(self) -> tuple[ChunkLayout, ...]:
        """All nesting levels, outermost (storage granularity) to innermost."""
        return (self, *(self.inner.levels if self.inner is not None else ()))

    @property
    def innermost(self) -> ChunkLayout:
        """The innermost level: the smallest independently decodable unit."""
        return self.inner.innermost if self.inner is not None else self


@dataclass(frozen=True, kw_only=True)
class RegularChunkLayout(ChunkLayout):
    """All chunks at this level share one uniform shape."""

    chunk_shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.chunk_shape)


@dataclass(frozen=True, kw_only=True)
class RectilinearChunkLayout(ChunkLayout):
    """Per-dimension chunk specs: a bare int (uniform size) or explicit edge lengths.

    Mirrors ``RectilinearChunkGridMetadata.chunk_shapes``, including the
    bare-int shorthand, so the distillation round-trips faithfully.
    """

    chunk_shapes: tuple[int | tuple[int, ...], ...]

    @property
    def ndim(self) -> int:
        return len(self.chunk_shapes)
```

Notes on the field choices:

- Field names mirror the metadata classes exactly: `chunk_shape` (`RegularChunkGridMetadata`), `chunk_shapes` (`RectilinearChunkGridMetadata`). No new vocabulary.
- Per-dimension validation reuses the semantics of `_validate_chunk_shapes` in `zarr.core.metadata.v3` (each element a positive int or a non-empty tuple of positive ints) in each subclass's `__post_init__`.
- `kw_only=True` sidesteps the dataclass-inheritance default-ordering problem (`inner` has a default in the base; subclass fields do not) and matches how `RegularChunkGridMetadata`/`RectilinearChunkGridMetadata` are declared.
- The base class deliberately does **not** define a kind-independent `chunks` field. Any such field would commit to one universal representation (per-dimension edge lists), which a future non-separable grid kind could not satisfy. Kind-specific data lives on kind-specific types.

### Level semantics

- `layout` itself (level 0) describes **storage/write granularity**: the units in which data is stored, and the shard shape when sharding is active.
- `layout.innermost` describes **read granularity**: the smallest unit that can be decoded independently.
- When `inner is None`, those are the same object and there is exactly one level.
- The names `innermost`/`levels` are deliberately not `read_level`/`write_level`: if reading at intermediate levels of a nested-sharding hierarchy becomes possible, any level in `levels` is addressable by index, and no level is privileged by name.

### The four configurations (plus the future)

| Configuration | top-level type | `is_sharded` | `innermost` |
|---|---|---|---|
| regular, unsharded | `RegularChunkLayout(chunk_shape=chunks)` | `False` | itself |
| regular, sharded | `RegularChunkLayout(chunk_shape=shards, inner=...)` | `True` | `RegularChunkLayout(chunk_shape=chunks)` |
| rectilinear, unsharded | `RectilinearChunkLayout(chunk_shapes=edges)` | `False` | itself |
| rectilinear, sharded | `RectilinearChunkLayout(chunk_shapes=shard_edges, inner=...)` | `True` | `RegularChunkLayout(chunk_shape=chunks)` |
| *(future)* rectilinear shards + rectilinear chunks | `RectilinearChunkLayout(..., inner=RectilinearChunkLayout(...))` | `True` | `RectilinearChunkLayout` |
| *(future)* nested sharding | three or more entries in `levels` | `True` | innermost level |

The two future rows require **zero API change** — only the construction-time validation (below) would be relaxed when the codecs and spec support them.

Both issues are answered without exceptions, for every row:

```python
layout = arr.chunk_layout

# 4036: is this array sharded?
layout.is_sharded

# 4035: what kind of chunk grid?
isinstance(layout, RegularChunkLayout)
# or, dispatching over kinds:
match layout:
    case RegularChunkLayout(chunk_shape=shape):
        ...
    case RectilinearChunkLayout(chunk_shapes=edges):
        ...
    case _:  # a grid kind newer than this code
        ...
```

`isinstance` against a *public* class is exactly the check that is painful today only because `RegularChunkGridMetadata` is private.

### Construction

`ChunkLayout` instances are built from metadata; users normally never construct them directly (though the constructors are public and validated).

```python
@classmethod
def from_metadata(cls, metadata: ArrayV2Metadata | ArrayV3Metadata) -> ChunkLayout:
    if isinstance(metadata, ArrayV2Metadata):
        return RegularChunkLayout(chunk_shape=metadata.chunks)
    inner: ChunkLayout | None = None
    if len(metadata.codecs) == 1:
        inner = metadata.codecs[0].inner_chunk_layout()
    grid = metadata.chunk_grid
    if isinstance(grid, RegularChunkGridMetadata):
        return RegularChunkLayout(chunk_shape=grid.chunk_shape, inner=inner)
    if isinstance(grid, RectilinearChunkGridMetadata):
        return RectilinearChunkLayout(chunk_shapes=grid.chunk_shapes, inner=inner)
    raise TypeError(f"Cannot derive a ChunkLayout from chunk grid {type(grid).__name__}")
```

**Extensibility hook for sharding-like codecs.** Rather than the `isinstance(codecs[0], ShardingCodec)` checks scattered through the codebase today (`array.py`, `metadata/v3.py`), the codec ABC gains one method with a default:

```python
# on zarr.abc.codec.Codec
def inner_chunk_layout(self) -> ChunkLayout | None:
    """The chunk structure this codec creates inside each chunk it encodes.

    None (the default) means chunks are opaque. Codecs that subdivide
    chunks, such as ShardingCodec, override this.
    """
    return None

# on ShardingCodec
def inner_chunk_layout(self) -> ChunkLayout:
    inner = self.codecs[0].inner_chunk_layout() if len(self.codecs) == 1 else None
    return RegularChunkLayout(chunk_shape=self.chunk_shape, inner=inner)
```

Nested sharding falls out of the recursion. A third-party sharding-like codec overrides one method and participates fully; third-party codecs that do nothing get the default and keep working.

**Protocol compatibility.** The 4.0 codec proposal ([zarr-python-planning `codecs.md`](https://github.com/d-v-b/zarr-python-planning/blob/main/proposals/codecs.md)) moves codecs from base classes to structurally typed protocols with capability advertising. `inner_chunk_layout` is specified to survive that move unchanged: callers must treat it as an optional protocol member — `layout = codec.inner_chunk_layout() if hasattr(codec, "inner_chunk_layout") else None` is the normative consumption pattern, with the defaulted method on today's `Codec` ABC merely a convenience so current third-party subclasses need no change. A codec that does not implement the member means "chunks are opaque," exactly like returning `None`.

A symmetrical hook for **pluggable chunk grids** is possible (a `to_chunk_layout()` method on chunk grid metadata classes instead of the `isinstance` dispatch in `from_metadata`), and would be the natural follow-up when a third grid kind appears. This design keeps the dispatch in `from_metadata` for now to avoid touching the metadata classes, and flags the method as the upgrade path.

### Array surface

```python
class Array:  # and AsyncArray
    @property
    def chunk_layout(self) -> ChunkLayout:
        """Declared chunk structure: a distillation of the chunk grid and sharding codecs."""
        return ChunkLayout.from_metadata(self.metadata)

    @property
    def is_sharded(self) -> bool:
        """True if this array's chunks have internal sub-chunk structure (sharding)."""
        return self.chunk_layout.is_sharded
```

`Array.is_sharded` is a convenience answering #4036 in one hop; it is sugar for `chunk_layout.is_sharded` and trivially cheap to keep consistent. Construction from metadata is O(ndim) and allocation-light, so `chunk_layout` is computed on demand rather than cached at `__init__` like `_chunk_grid`; caching via the `object.__setattr__` pattern already used for `_chunk_grid` is a drop-in change if profiling ever warrants it.

### Naming collision with the internal `ChunkLayout`

`zarr.core.chunk_grids.ChunkLayout` (the creation-time `NamedTuple` holding normalized `ChunksTuple` arrays — see [chunk-grid.md § Creation-time chunk resolution](./chunk-grid.md#creation-time-chunk-resolution)) already uses this name. Resolution: the internal type is **renamed** to `_ResolvedChunks` (it is exactly the output of `resolve_outer_and_inner_chunks`). It is private and creation-time only, so the rename is invisible to users. Unifying it with the public type (having `resolve_outer_and_inner_chunks` return a public `ChunkLayout`) is plausible but not required here — the internal type carries normalized per-dimension `np.int64` edge arrays serving the grid-metadata construction path, while the public type preserves the declared int-vs-edges distinction. Keeping them separate keeps this change additive; unification should be revisited if the creation API is ever aligned with the public type.

### Public location

- Classes live in a new module `zarr/core/chunk_layouts.py` (avoiding both the name collision and a circular import with `zarr.abc.codec`), re-exported from the top-level `zarr` namespace: `zarr.ChunkLayout`, `zarr.RegularChunkLayout`, `zarr.RectilinearChunkLayout`.
- `Array.chunk_layout`, `Array.is_sharded`, `AsyncArray.chunk_layout`, `AsyncArray.is_sharded`.

### Validation and error handling

- Subclass `__post_init__` validates shapes (positive ints; non-empty edge tuples) with the same rules as the corresponding metadata classes, raising `ValueError`.
- The base `__post_init__` enforces `inner.ndim == ndim`, raising `ValueError`.
- **Codec-level restrictions are not type-level restrictions.** Today's rule that inner chunks must be regular (and divide every distinct outer edge) is enforced by `ShardingCodec` at creation/validation time, not by `ChunkLayout`. The layout type itself accepts `RectilinearChunkLayout` as `inner`, so relaxing the codec restriction later requires no API change. `from_metadata` only ever produces layouts that the codecs already validated.
- `from_metadata` raises `TypeError` for an unrecognized chunk grid metadata type — loud failure for grid kinds this version cannot distill, consistent with `ChunkGrid.from_metadata`.

## Design decisions

### Why typed subclasses instead of a union-typed field?

A single `chunks: tuple[int | tuple[int, ...], ...]` field (one per-dimension spec, int = regular) was considered and rejected: it bakes in two assumptions that are already known to be temporary — that inner levels are always regular (false once rectilinear inner chunks exist, see [zarr-extensions#34](https://github.com/zarr-developers/zarr-extensions/issues/34)), and that every grid kind is expressible as per-dimension edge lists (false once a non-separable pluggable grid kind exists). Kind-as-type carries neither assumption.

### Why not parallel `<kind>_chunks` accessors?

`regular_chunks`/`rectilinear_chunks`/… accessors, each `None` when not that kind, were considered: every new kind widens every existing object's surface with a `None`-returning accessor, mixed-dimension grids (one dim uniform, one explicit edges) do not partition between accessors, and `None`-probing chains are type dispatch without the type system's help. The typed-subclass form provides the same "new kinds slot in cleanly" property without those costs.

### Why not `typing.NewType` per kind?

`NewType` is static-only: no `isinstance`, no `match`, no fields, no methods. #4035 requires runtime dispatch on objects deserialized from `zarr.json`. (`NewType` remains the right tool for provenance-marking like `ChunksTuple`.)

### Why not just expose `ChunkGrid` plus predicates?

`ChunkGrid` is extent-bound and single-level — it cannot represent the grid+codec distillation, and it reports clipped/realized sizes where round-tripping needs declared sizes. It remains the right home for realized-size queries (`read_chunk_sizes`/`write_chunk_sizes` already cover those on `Array`). Predicates alone (`Array.is_sharded`, `Array.has_regular_chunk_grid`) would answer the two issues but leave inner-shape introspection, declared sizes, and nested sharding for a third issue.

### Why is `ChunkLayout` extent-free?

Two reasons. It makes the object a pure distillation of metadata — deterministic, hashable, comparable, with no dependence on the array it came from — and it makes declared sizes (which the extent-bound accessors clip) directly available, closing the round-trip-preservation gap from the HackMD evaluation. Extent-bound questions already have a home (`ChunkGrid`).

## Prior art

**TensorStore** has a public [`ChunkLayout`](https://google.github.io/tensorstore/python/api/tensorstore.ChunkLayout.html) distinguishing write/read/codec chunk granularities — the same storage-vs-decode split this design expresses as `levels[0]` vs `innermost`, without privileging any single level by name.

**zarrs** exposes typed chunk grid variants behind a trait, with the regular grid as a special case — kind-as-type dispatch, as here.

## Relationship to the 4.0 planning set ([d-v-b/zarr-python-planning](https://github.com/d-v-b/zarr-python-planning))

No proposal in the 4.0 planning set covers declared chunk-structure introspection — #4035/#4036 do not appear in the `missing-apis.md` audit — so this design fills a gap rather than duplicating one. Three proposals are adjacent:

- **`observability.md` Pillar 2 (stored-state introspection)** is the complement: it answers questions about *stored state* (`chunk_exists`, `chunk_byte_range`, `storage_size`, `read_block`) where this design answers questions about *declared structure*. Its proposed `array.nchunks` / `nshards` / `n_inner_chunks` disambiguation ([zarr#3296](https://github.com/zarr-developers/zarr-python/issues/3296)) should be derived from `chunk_layout.levels` so the two surfaces cannot disagree about what counts as a shard.
- **`functional-core.md`** places chunk-addressing types in the pure-data `zarr-metadata` package. `ChunkLayout` as specified here is pure data (frozen, extent-free, derived deterministically from metadata + codec config), so it is *compatible* with migrating into the functional core later — but per @d-v-b's comment on #4036 it is a zarr-python distillation, not spec metadata, so it ships in zarr-python and any package move is deferred to the functional-core work. Nothing in this design assumes either outcome.
- **`codecs.md`** (protocols over base classes): addressed by the protocol-compatibility paragraph in the construction section above.

## Testing

- Construction matrix: all four current configurations × v2/v3 metadata → expected type, `is_sharded`, `levels`, `innermost`.
- Property tests: `levels[0] is layout`, `levels[-1] is layout.innermost`, `len(levels) >= 2 ⟺ is_sharded`, `ndim` uniform across levels.
- Validation: ndim mismatch, non-positive sizes, empty edge tuples, direct instantiation of the abstract base → expected exceptions.
- Codec hook: a stub third-party codec overriding `inner_chunk_layout` is detected; nested `ShardingCodec` produces three levels.
- Equality/hash: layouts derived from equal metadata compare equal (frozen dataclass semantics) — needed by downstream caching.
- Doc snippets exercised under `pytest --doctest-modules` like the existing accessor examples.

## Open questions

1. Should `Array.is_sharded` ship, or is `arr.chunk_layout.is_sharded` enough? (This draft includes it; it is one property and directly answers the issue title.)
2. Should the rename of the internal `NamedTuple` to `_ResolvedChunks` happen in the same PR (proposed) or a precursor refactor PR?
3. Is `RectilinearChunkLayout` the right name, given the metadata class is `RectilinearChunkGridMetadata`? (Alternative: `VariableChunkLayout`, matching the "VLC" terminology from the 2026-02-11 dev meeting.)
4. Should `chunk_layout` live under `zarr.experimental` first (like `ChunkGrid`/`ChunkSpec` did) and graduate, or ship directly in the stable namespace given the small surface?
