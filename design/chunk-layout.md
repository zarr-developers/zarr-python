# Public Chunk Layout

Version: 2

Design document for adding a public, typed chunk-structure introspection API to **zarr-python**: a `ChunkLayout` object that distills the *declared* chunk structure of an array — its chunk grid metadata together with the chunk-structuring codecs in its pipeline — into the form consumers can feed back into `create_array`. (The *observed* chunk layout depends on the full codec pipeline and on store capabilities; that is the effective-granularity question, scoped as future work below.)

**Related:**

- [#4035](https://github.com/zarr-developers/zarr-python/issues/4035) (public API for determining chunk grid type)
- [#4036](https://github.com/zarr-developers/zarr-python/issues/4036) (public API for determining if an array is sharded)
- [#4040 review discussion](https://github.com/zarr-developers/zarr-python/pull/4040) (the v1 → v2 changes below)
- [API-surface evaluation](https://hackmd.io/p9pU24IQR6uA9tg6c0wMhw) (gaps in the chunk/shard accessor family)
- [chunk-grid.md](./chunk-grid.md) (unified chunk grid design; this document builds on it)
- [zarr-extensions#34](https://github.com/zarr-developers/zarr-extensions/issues/34) (sharding + rectilinear)
- [pydata/xarray#11279](https://github.com/pydata/xarray/pull/11279) (downstream consumer; the reconstruction use case)

**Changes from v1** (driven by review on #4040):

1. The kind-as-subclass hierarchy (`RegularChunkLayout` / `RectilinearChunkLayout`) is replaced by a **single class** with an `is_regular` property. Every regular grid is a rectilinear grid, so peer nominal types misrepresent the subset relation — the same argument that led [chunk-grid.md](./chunk-grid.md) to one internal `ChunkGrid` class ("one implementation, multiple serialization forms").
2. The layout is **canonical over metadata**: the same abstract grid yields the same `ChunkLayout` whether it was declared with `regular` or `rectilinear` metadata. The v1 goal of mirroring the metadata document faithfully is dropped — per review, consumers should not observe which metadata declared a grid, only what the grid is.
3. The problem statement is recast: the underlying downstream need is not "which grid kind is this?" but **non-raising predicates plus reconstruction** — enough public API to recreate an equivalent layout via `create_array`, without exception handling.
4. **Effective read/write granularities** (the partition sizes consumers need for access-pattern decisions, per the review discussion) are explicitly scoped as future work building on codec capabilities; this design provides the *declared* structure they would be computed from.
5. `Array.is_sharded` is derived from the codec hook rather than from `chunk_layout`, so it answers for grid kinds the layout cannot distill; a new stress-test section works two hypothetical grid extensions through the design to validate its extension seams.

**Scope:** additive only. This design introduces a new introspection object and leaves `Array.chunks`, `Array.shards`, `Array.read_chunk_sizes`, and `Array.write_chunk_sizes` exactly as they are (but see open question 4). A more ambitious variant — making `ChunkLayout` the canonical chunk-structure API and deprecating `.chunks`/`.shards` on a schedule — was evaluated separately and is deliberately out of scope.

## Problem

zarr-python's read-side APIs deliberately abstract over chunk grid metadata: `read_chunk_sizes` / `write_chunk_sizes` return the same size tuples whether the grid was declared with `regular` or `rectilinear` metadata. But the abstraction is asymmetric — it does not extend to writing or to the predicates consumers need:

1. **Is this array sharded?** ([#4036](https://github.com/zarr-developers/zarr-python/issues/4036)) — `.shards is not None` works for regular chunk grids, but `.shards` raises `NotImplementedError` for rectilinear grids, so sharded-rectilinear arrays require `try/except`.
2. **Is this grid regular?** ([#4035](https://github.com/zarr-developers/zarr-python/issues/4035)) — answerable today only by catching `NotImplementedError` from `.chunks` or by `isinstance(array.metadata.chunk_grid, RegularChunkGridMetadata)` with an import from `zarr.core.metadata.v3`, which is private. (`ChunkGrid.is_regular` exists internally but is reachable only through the private `Array._chunk_grid`.)
3. **Reconstruction.** `create_array` makes the caller choose the metadata through the shape of the argument: `chunks=(10, 10)` writes regular metadata, `chunks=[[10, 10], [10, 10]]` writes rectilinear metadata — and the choice has consequences (the rectilinear form requires the feature flag, does not exist in zarr format 2, and is unreadable by implementations without the extension). A consumer that reads an array and recreates its layout therefore has to recover the regularity the read APIs abstracted away, and it cannot do so reliably from size tuples alone, because those are boundary-clipped: `(10, 10, 10, 5)` on a length-35 axis *is* a regular grid of 10s.

The reconstruction case is not hypothetical — it is how xarray works. xarray reads an array's layout into an encoding dict and passes it back as `create_array(chunks=..., shards=...)` kwargs on write ([pydata/xarray#11279](https://github.com/pydata/xarray/pull/11279)). On current xarray `main`, [`open_store_variable`](https://github.com/pydata/xarray/blob/057f7039/xarray/backends/zarr.py#L914-L923) reads `.chunks` and `.shards` straight into that dict with no exception handling, so opening *any* rectilinear array through xarray raises `NotImplementedError` before the user touches a single chunk — a metadata extension became a breaking change for downstream code that never opted into it. And xarray's [`_determine_zarr_chunks`](https://github.com/pydata/xarray/blob/057f7039/xarray/backends/zarr.py#L336-L410) has always performed a regularity check on the chunks it is about to write, because that is what `create_array(chunks=int_per_dim)` demands; rectilinear support gives the non-regular branch a non-error answer but does not remove the question.

The [HackMD API-surface evaluation](https://hackmd.io/p9pU24IQR6uA9tg6c0wMhw) identifies further gaps the same object closes: `.chunks` raises for rectilinear-sharded arrays even though their inner chunks are regular, and no accessor reports *declared* (unclipped) chunk sizes for round-trip preservation.

In [#4036](https://github.com/zarr-developers/zarr-python/issues/4036), @d-v-b proposed exposing the internal `ChunkLayout` ([`chunk_grids.py:55`](https://github.com/zarr-developers/zarr-python/blob/fe229107f9915f05817f7a664d3550695ff9ca44/src/zarr/core/chunk_grids.py#L55)) — "a distillation of the chunk grid + codecs for an array" — after polish. This design is that polish.

## Goals

1. **Non-raising predicates.** A public way to answer "is this array sharded?" and "is this grid regular?" for every present and future grid/sharding combination, without exceptions or private imports.
2. **Reconstruction.** The declared chunk structure, unclipped, in the form `create_array(chunks=..., shards=...)` accepts — so a consumer can recreate an equivalent layout for any array it can read, without enumerating grid kinds.
3. **Forward compatibility.** Layouts that do not exist yet — rectilinear inner chunks, rectilinear shards, nested sharding — must slot in without breaking the API; grid kinds this version cannot distill must fail loudly, not silently.
4. **Third-party sharding codecs.** Per @d-v-b on #4036, the design must accommodate new codecs that implement sharding, not hard-code `ShardingCodec`.

## Non-goals

- No change or deprecation of any existing accessor (`.chunks`, `.shards`, `.read_chunk_sizes`, `.write_chunk_sizes`).
- No serialization (`to_dict`/`from_dict`) — `ChunkLayout` is derived from metadata, never stored.
- No extent-bound queries (clipped sizes, chunk counts, index-to-chunk mapping). Those remain the job of the internal `ChunkGrid` ([chunk-grid.md](./chunk-grid.md)), which stays private in this design; `read_chunk_sizes`/`write_chunk_sizes` already expose the realized sizes.
- No creation-API changes (`create_array(chunks=..., shards=...)` is untouched).
- **No effective read/write granularities.** The effective partitions for access depend on the full codec pipeline and on store capabilities (e.g. `[sharding, gzip]` makes the whole shard the effective read unit; a selection-preserving pipeline over a partial-write store makes the effective write unit a single element — see the [#4040 review discussion](https://github.com/zarr-developers/zarr-python/pull/4040)). Computing them requires codec capability advertising that does not exist yet. This design provides the *declared* structure those computations would consume, and the `inner_chunk_layout` codec hook below is the first such capability; see "Relationship to effective granularities".

## Design

### Core principle: declared, extent-free, recursive, canonical

`ChunkLayout` describes the *declared* chunk structure of an array — never clipped to the array shape. It is recursive: each level describes the structure *within one cell of the level above* (the top level describes structure within the whole array). It is **canonical**: equal abstract grids produce equal layouts, regardless of which metadata declared them. Regularity is a property (`is_regular`), not a type — mirroring the internal `ChunkGrid` design.

```python
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ChunkLayout:
    """Declared chunk structure of an array.

    A distillation of the chunk grid metadata and sharding codec
    configuration. Extent-free: sizes are as declared, never clipped
    to the array shape. Canonical: a dimension whose declared edge
    lengths are all equal is normalized to the bare uniform size, so
    layouts of the same abstract grid compare equal regardless of the
    metadata that declared them.
    """

    chunks: tuple[int | tuple[int, ...], ...]
    """Per-dimension chunk spec at this level: a bare int (uniform size)
    or explicit edge lengths — the same union ``create_array`` accepts."""

    inner: ChunkLayout | None = None
    """Structure within each chunk of this level, or None if chunks are opaque (no sharding)."""

    def __post_init__(self) -> None:
        # validation: positive ints; non-empty edge tuples (reuses the
        # semantics of _validate_chunk_shapes in zarr.core.metadata.v3)
        ...
        # canonicalization: a uniform edge tuple collapses to its int
        object.__setattr__(
            self,
            "chunks",
            tuple(c[0] if isinstance(c, tuple) and len(set(c)) == 1 else c for c in self.chunks),
        )
        if self.inner is not None and self.inner.ndim != self.ndim:
            raise ValueError(
                f"inner layout has {self.inner.ndim} dimensions, expected {self.ndim}"
            )

    @property
    def ndim(self) -> int:
        return len(self.chunks)

    @property
    def is_regular(self) -> bool:
        """True if every dimension at this level has one uniform chunk size."""
        return all(isinstance(c, int) for c in self.chunks)

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
        """The innermost level of declared subdivision.

        Whether this unit is independently decodable depends on the
        full codec pipeline, not on the declared structure alone.
        """
        return self.inner.innermost if self.inner is not None else self
```

Notes on the field choices:

- The `chunks` field uses **creation-parameter vocabulary**, not metadata vocabulary: `tuple[int | tuple[int, ...], ...]` is exactly the per-dimension union that `create_array(chunks=...)` accepts (and `ShardsLike` accepts the nested form for `shards=`). The point of the object is that consumers feed it back into `create_array`; the field is shaped for that.
- **Canonicalization is sound** because for any extent that valid metadata can cover, a dimension declared with `k` equal edges of length `n` and a dimension declared regular with size `n` produce identical chunk boundaries. Collapsing the former to the latter is the layout-level expression of the principle from [chunk-grid.md](./chunk-grid.md) that the serialization format is chosen by the metadata layer, not the grid — and it means downstream code that writes `layout.chunks` back through `create_array` automatically produces regular metadata wherever the grid permits it, maximizing format-2 and extension-less-reader compatibility with no special-casing.
- `is_regular` is per-level and `O(ndim)`; the whole hierarchy is regular iff `all(level.is_regular for level in layout.levels)`. It matches the internal `ChunkGrid.is_regular` for the same grid.
- `kw_only=True` sidesteps the dataclass default-ordering problem (`inner` has a default) and matches how the metadata classes are declared.

### Level semantics

- `layout` itself (level 0) describes **storage granularity**: the units in which data is stored — the shard shape when sharding is active, the chunk shape otherwise.
- `layout.innermost` describes the innermost *declared* subdivision; whether it is independently decodable, and whether it is the effective read unit, depends on the full codec pipeline (a bytes→bytes codec after a sharding codec makes the whole shard the decode unit — see "Relationship to effective granularities").
- When `inner is None`, those are the same object and there is exactly one level.
- The names `innermost`/`levels` are deliberately not `read_level`/`write_level`: the effective read and write granularities are not, in general, properties of the declared structure alone, and if reading at intermediate levels of a nested-sharding hierarchy becomes possible, any level in `levels` is addressable by index with no level privileged by name.

### The four configurations (plus the future)

| Configuration | `chunks` (level 0) | `is_regular` (level 0) | `is_sharded` | `innermost.chunks` |
|---|---|---|---|---|
| regular, unsharded | per-dim ints | `True` | `False` | itself |
| regular, sharded | shard shape (ints) | `True` | `True` | inner chunk shape (ints) |
| rectilinear, unsharded | per-dim int-or-edges | `False` | `False` | itself |
| rectilinear, sharded | shard edges | `False` | `True` | inner chunk shape (ints) |
| *(future)* rectilinear shards + rectilinear inner chunks | shard edges | `False` | `True` | int-or-edges |
| *(future)* nested sharding | — | — | `True` | three or more entries in `levels` |

The two future rows require **zero API change** — only the construction-time validation would be relaxed when the codecs and spec support them. (Today's rule that inner chunks must be regular is enforced by `ShardingCodec` at creation/validation time, not by `ChunkLayout`.)

Both issues are answered without exceptions, for every row:

```python
layout = arr.chunk_layout

# 4036: is this array sharded?
layout.is_sharded

# 4035: is this grid regular?
layout.is_regular
```

### The reconstruction contract

The guarantee this design makes to downstream libraries: **for any array the library can read, `chunk_layout` yields values that recreate an equivalent layout through `create_array`, with no exception handling and no grid-kind enumeration.** Concretely, for xarray's encoding extraction (which today wraps `.chunks` in `try/except NotImplementedError` and crashes on sharded-rectilinear arrays via `.shards`):

```python
layout = zarr_array.chunk_layout
if layout.is_sharded:
    encoding["shards"] = layout.chunks        # ShardsLike: ints or nested
    encoding["chunks"] = layout.inner.chunks  # inner chunk shape
else:
    encoding["shards"] = None
    encoding["chunks"] = layout.chunks
# later: create_array(chunks=encoding["chunks"], shards=encoding["shards"], ...)
```

This one branch covers all four current configurations. Because the layout is canonical, a uniform grid round-trips to regular metadata regardless of how it was declared — "equivalent layout" means equal chunk boundaries, not an identical metadata document, which is the strongest guarantee that is *desirable*: preserving a rectilinear declaration of a uniform grid would propagate the feature-flag and extension requirements to readers for no structural benefit.

Because the layout is extent-free, the values are declared sizes, never clipped — closing the round-trip gap where realized-size accessors report `(10, 10, 10, 5)` for a regular grid of 10s.

### Construction

`ChunkLayout` instances are built from metadata; users normally never construct them directly (though the constructor is public and validated).

```python
@classmethod
def from_metadata(cls, metadata: ArrayV2Metadata | ArrayV3Metadata) -> ChunkLayout:
    if isinstance(metadata, ArrayV2Metadata):
        return ChunkLayout(chunks=metadata.chunks)
    inner: ChunkLayout | None = None
    if metadata.codecs:
        inner = metadata.codecs[0].inner_chunk_layout()
    grid = metadata.chunk_grid
    if isinstance(grid, RegularChunkGridMetadata):
        return ChunkLayout(chunks=grid.chunk_shape, inner=inner)
    if isinstance(grid, RectilinearChunkGridMetadata):
        return ChunkLayout(chunks=grid.chunk_shapes, inner=inner)
    raise TypeError(f"Cannot derive a ChunkLayout from chunk grid {type(grid).__name__}")
```

(Canonicalization in `__post_init__` means the `RectilinearChunkGridMetadata` arm produces a layout with `is_regular == True` when the declared edges happen to be uniform — the metadata kind is deliberately not observable from the layout.)

**Why the *first* codec, rather than a sole codec?** The existing accessors gate on `len(codecs) == 1 and isinstance(codecs[0], ShardingCodec)`, so `.shards` reports `None` for a pipeline like `[sharding, gzip]` — an *effective*-granularity judgment (the trailing gzip makes the whole shard the decode unit, so reporting the inner shape could mislead) baked into what is otherwise a structural accessor. `ChunkLayout` is declared-only, so it draws the boundary structurally instead: trailing bytes→bytes codecs cannot *create* chunk structure — their effect on access belongs to the effective granularities — so `[sharding, gzip]` declares its subdivision (`is_sharded == True`, two levels), intentionally diverging from `.shards` (`None`) on such pipelines. Conversely, an array→array codec *before* a chunk-structuring codec changes the coordinate mapping between the codec's configured shapes and array dimensions, so a non-first chunk-structuring codec is treated as opaque (`inner is None`) until codec capabilities can express that mapping. The sole-codec gate is the intersection of both rules; the first-codec rule is each rule applied for its own structural reason.

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
    return ChunkLayout(chunks=self.chunk_shape, inner=inner)
```

Nested sharding falls out of the recursion. A third-party sharding-like codec overrides one method and participates fully; third-party codecs that do nothing get the default and keep working. This is the first instance of the codec-capability pattern from the [#4040 review discussion](https://github.com/zarr-developers/zarr-python/pull/4040) — codecs advertising what they do to chunk structure, instead of callers special-casing codec classes.

The `ChunkLayout | None` return reserves room for one known future refinement: `None` means "chunks are opaque," and a sharding revision with *per-shard variable* inner grids (see the stress tests below) would need a third state — "subdivided, but variably" — representable as a future marker layout rather than a signature change.

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
        codecs = getattr(self.metadata, "codecs", ())
        return bool(codecs) and codecs[0].inner_chunk_layout() is not None
```

`Array.is_sharded` answers #4036 in one hop. It is deliberately derived from the codec hook, *not* from `chunk_layout`: sharding is a codec property, independent of the grid, so `is_sharded` keeps answering correctly even for a future grid kind that `from_metadata` cannot distill (where `chunk_layout` raises `TypeError`). For every distillable configuration the two agree (see Testing). Construction from metadata is O(ndim) and allocation-light, so `chunk_layout` is computed on demand rather than cached at `__init__` like `_chunk_grid`; caching via the `object.__setattr__` pattern already used for `_chunk_grid` is a drop-in change if profiling ever warrants it.

### Relationship to effective granularities

The [#4040 review discussion](https://github.com/zarr-developers/zarr-python/pull/4040) identified four quantities consumers ultimately need:

| Quantity | Source | Status |
|---|---|---|
| size of each stored chunk (per-object partition) | `layout.chunks` / `layout.levels[0]` | this design |
| size of each subchunk (inside-an-object partitions, recursive) | `layout.levels[1:]` | this design |
| effective **write** granularity | bounded by `levels[0]`; refined by codec selection-preservation and store partial-write capability | future work |
| effective **read** granularity | bounded by `innermost`; refined by the codec pipeline (e.g. `[sharding, gzip]` makes the whole shard the effective read unit) | future work |

The declared structure and the effective granularities have different consumers, sometimes within one library. xarray needs all three rows it can get today, from different places: the reconstruction values from `chunk_layout` (this design), the dask `preferred_chunks` hint from the effective read granularity (today approximated by `.chunks`, which the `[sharding, gzip]` example shows can be misleading), and the parallel-write alignment unit from the effective write granularity (today hand-rolled as `shards or chunks`). This design deliberately ships the declared-structure rows now and leaves the effective rows to the codec-capabilities work, for two reasons: the effective values depend on store capabilities and so are not pure array properties, and the declared structure is the substrate any effective-granularity computation consumes — it has to exist first.

### Forward compatibility, stress-tested

Two hypothetical grid extensions, each permitted by the v3 core spec, used here as stress tests of the design's extension seams — not as predictions. (Spec background: the core spec requires only tessellation — no gaps, no overlaps — not per-dimension separability; chunk addressing is defined per grid extension; and `must_understand: false` is disallowed for chunk grids, so any new grid kind fails loudly in clients that predate it, by design.)

**A non-separable grid (e.g. a kd-tree/quadtree partition).** No per-dimension API can represent such a grid — `.chunks`, `read_chunk_sizes`, this design's `chunks` field, and the chunk models of dask and xarray are all per-dimension by construction — so this is not a gap `ChunkLayout` introduces, and downstream consumers could not consume such a grid through their existing chunk models regardless of what introspection returns. Behavior here: `from_metadata` raises `TypeError` (loud, correct), while `Array.is_sharded` still answers, because it is codec-derived. The extension path, if such a grid ships: widen the per-dimension union in `chunks` with a tree-node spec and canonicalize downward — a tree that is in fact rectilinear collapses to edge lists, edges that are uniform collapse to ints. That is the same subset hierarchy (regular ⊂ rectilinear ⊂ tree-partition) and the same canonicalization rule this design already uses; no subclasses are needed.

**Per-shard variable inner grids** (a sharding-codec revision where each shard's index carries its own inner grid; adjacent to [zarr-extensions#34](https://github.com/zarr-developers/zarr-extensions/issues/34)). This breaks two assumptions at once: that inner structure is uniform across cells of the level above, and that inner structure is *declared in metadata at all* — it becomes data, fetched from shard indexes. The declared/effective split then becomes load-bearing: `ChunkLayout` describes exactly the declared envelope — `levels[0]` remains fully specified (the storage/write-alignment grid downstream needs most), and the inner level is where "varies per chunk" would be represented via the codec hook's reserved third state. The per-shard *actual* partitions are stored state, not declared structure, and belong to stored-state introspection ([`observability.md`](https://github.com/d-v-b/zarr-python-planning)), not here.

Both stress tests resolve without changing this design's public surface, and they shaped two decisions above: `Array.is_sharded` is codec-derived, and the codec hook reserves its third state.

### Naming collision with the internal `ChunkLayout`

`zarr.core.chunk_grids.ChunkLayout` (the creation-time `NamedTuple` holding normalized `ChunksTuple` arrays — see [chunk-grid.md § Creation-time chunk resolution](./chunk-grid.md#creation-time-chunk-resolution)) already uses this name. Resolution: the internal type is **renamed** to `_ResolvedChunks` (it is exactly the output of `resolve_outer_and_inner_chunks`). It is private and creation-time only, so the rename is invisible to users. Unifying it with the public type (having `resolve_outer_and_inner_chunks` return a public `ChunkLayout`) is plausible but not required here — the internal type carries normalized per-dimension `np.int64` edge arrays serving the grid-metadata construction path, while the public type carries the canonical creation-parameter form. Keeping them separate keeps this change additive; unification should be revisited if the creation API is ever aligned with the public type.

### Public location

- The class lives in a new module `zarr/core/chunk_layouts.py` (avoiding both the name collision and a circular import with `zarr.abc.codec`), re-exported from the top-level `zarr` namespace: `zarr.ChunkLayout`.
- `Array.chunk_layout`, `Array.is_sharded`, `AsyncArray.chunk_layout`, `AsyncArray.is_sharded`.

### Validation and error handling

- `__post_init__` validates shapes (positive ints; non-empty edge tuples) with the same rules as the corresponding metadata classes, raising `ValueError`, then canonicalizes.
- `__post_init__` enforces `inner.ndim == ndim`, raising `ValueError`.
- **Codec-level restrictions are not type-level restrictions.** Today's rule that inner chunks must be regular (and divide every distinct outer edge) is enforced by `ShardingCodec` at creation/validation time, not by `ChunkLayout`. The layout type itself accepts a non-regular `inner`, so relaxing the codec restriction later requires no API change. `from_metadata` only ever produces layouts that the codecs already validated.
- `from_metadata` raises `TypeError` for an unrecognized chunk grid metadata type — loud failure for grid kinds this version cannot distill, consistent with `ChunkGrid.from_metadata`. A grid kind not expressible as per-dimension chunk specs would also need new `create_array` parameters, so extending `ChunkLayout` alongside is the expected (and acceptable) cost of such an extension.

## Design decisions

### Why one class with `is_regular` instead of kind subclasses? *(changed from v1)*

v1 proposed `RegularChunkLayout` / `RectilinearChunkLayout` as peer types, with kind-as-type dispatch. Review on #4040 made the case against it: every regular grid *is* a rectilinear grid, so peer nominal types misrepresent a subset relation as a partition — `isinstance(layout, RegularChunkLayout)` would be `False` for a rectilinear-declared grid with uniform edges, even though the grid is regular in every sense that matters to a consumer. It is the same argument that led [chunk-grid.md](./chunk-grid.md) to a single internal `ChunkGrid` class with `is_regular`, and the layout should not reintroduce at the public surface the split the internal design deliberately removed. The question consumers actually ask — "are all the chunks the same size?" — is answerable as a boolean for any grid kind.

### Why canonicalize instead of mirroring the metadata?

Because `{"name": "regular", ...}` and `{"name": "rectilinear", ...}` can declare the exact same abstract grid, a layout that mirrors the metadata makes equal grids compare unequal and leaks the declaration form to consumers who should not depend on it. Canonicalization makes the layout a value: deterministic, hashable, equal iff the grids are equal. It also makes reconstruction *normalizing* — `layout.chunks` fed back through `create_array` produces regular metadata wherever the grid permits, which minimizes feature-flag, format-2, and reader-compatibility exposure. Consumers who genuinely need the declaration form (none are known) have `array.metadata`.

### Why a `chunks` field in creation-parameter form?

The object exists so consumers can recreate layouts; the field is shaped as the argument they will pass. The alternative — kind-specific fields mirroring the metadata classes (`chunk_shape` vs `chunk_shapes`) — re-imports the metadata distinction the canonicalization just removed and forces consumers to branch to assemble `create_array` arguments.

### Why is `ChunkLayout` extent-free?

Two reasons. It makes the object a pure distillation of metadata — deterministic, hashable, comparable, with no dependence on the array it came from — and it makes declared sizes directly available where the extent-bound accessors clip (`(10, 10, 10, 5)` on a length-35 axis is a regular grid of 10s; only the declared form preserves that). Extent-bound questions already have homes: `read_chunk_sizes`/`write_chunk_sizes` on `Array`, and the internal `ChunkGrid`.

### Why not just expose `ChunkGrid` plus predicates?

Three reasons, none of them about what `ChunkGrid` stores (it keeps declared sizes internally — `FixedDimension.size` is the unclipped value — so declared-vs-clipped is an accessor gap, not a structural one):

1. **`ChunkGrid` is codec-blind, by correct layering.** `chunk_grids.py` cannot know about sharding, so the grid alone can never answer `is_sharded`, report inner chunk shapes, or represent nesting. The distillation inherently combines grid metadata with codec configuration — and that combined object already exists internally (the creation-time `ChunkLayout` NamedTuple, below) because array creation needs the same view. This design polishes the existing distillation rather than inventing a parallel one.
2. **A vocabulary trap.** For a sharded array, `Array._chunk_grid` is the *shard* grid. A public `arr.chunk_grid` would describe shards while `.chunks` describes inner chunks — a collision that is invisible while the grid is private and a footgun the moment it is not.
3. **Stability scope.** Publishing `ChunkGrid` stabilizes its whole operational surface (`index_to_chunk`, `ChunkSpec` slicing, prefix-sum internals) — and its per-dimension representation sits exactly at the known refactor point for non-separable grids ([chunk-grid.md](./chunk-grid.md)'s grid-level-Protocol escape hatch). `ChunkLayout` is a small frozen value, cheap to commit to. Notably, `ChunkGrid`'s *query* interface (coordinates in, chunk box out) is the one API shape that survives non-separability — an argument for an eventual public query-based chunk-geometry API as a separate, larger proposal, not for exposing the representation now.

Predicates alone (`Array.is_sharded`, `Array.has_regular_chunk_grid`) would answer the two issues but leave reconstruction unsolved: a consumer still could not obtain declared shard edges for a sharded-rectilinear array without `try/except`, nor inner chunk shapes without reaching into codec configuration.

## Prior art

**TensorStore** has a public [`ChunkLayout`](https://google.github.io/tensorstore/python/api/tensorstore.ChunkLayout.html) distinguishing write/read/codec chunk granularities — the same storage-vs-decode split this design expresses as `levels[0]` vs `innermost`, without privileging any single level by name.

**zarrs** exposes chunk grid variants behind a trait with the regular grid as a special case — regularity as a queryable property of one abstraction, as here.

## Relationship to the 4.0 planning set ([d-v-b/zarr-python-planning](https://github.com/d-v-b/zarr-python-planning))

No proposal in the 4.0 planning set covers declared chunk-structure introspection — #4035/#4036 do not appear in the `missing-apis.md` audit — so this design fills a gap rather than duplicating one. Three proposals are adjacent:

- **`observability.md` Pillar 2 (stored-state introspection)** is the complement: it answers questions about *stored state* (`chunk_exists`, `chunk_byte_range`, `storage_size`, `read_block`) where this design answers questions about *declared structure*. Its proposed `array.nchunks` / `nshards` / `n_inner_chunks` disambiguation ([zarr#3296](https://github.com/zarr-developers/zarr-python/issues/3296)) should be derived from `chunk_layout.levels` so the two surfaces cannot disagree about what counts as a shard.
- **`functional-core.md`** places chunk-addressing types in the pure-data `zarr-metadata` package. `ChunkLayout` as specified here is pure data (frozen, extent-free, canonical, derived deterministically from metadata + codec config), so it is *compatible* with migrating into the functional core later — but per @d-v-b's comment on #4036 it is a zarr-python distillation, not spec metadata, so it ships in zarr-python and any package move is deferred to the functional-core work. Nothing in this design assumes either outcome.
- **`codecs.md`** (protocols over base classes): addressed by the protocol-compatibility paragraph in the construction section above; `inner_chunk_layout` is an early instance of the capability-advertising pattern that proposal generalizes.

## Testing

- Construction matrix: all four current configurations × v2/v3 metadata → expected `chunks`, `is_regular`, `is_sharded`, `levels`, `innermost`.
- **Canonicalization:** a rectilinear-declared grid with uniform edges and the equivalent regular-declared grid produce equal layouts (`==`, same hash, `is_regular == True`).
- **Reconstruction round-trip (the contract test):** for every readable configuration, `create_array(chunks=..., shards=...)` from `chunk_layout`-derived values produces an array whose `chunk_layout` compares equal — including the normalizing case (rectilinear-declared uniform grid → regular metadata → equal layout).
- Property tests: `levels[0] is layout`, `levels[-1] is layout.innermost`, `len(levels) >= 2 ⟺ is_sharded`, `ndim` uniform across levels.
- Validation: ndim mismatch, non-positive sizes, empty edge tuples → expected exceptions.
- Codec hook: a stub third-party codec overriding `inner_chunk_layout` is detected; nested `ShardingCodec` produces three levels.
- `Array.is_sharded` agrees with `chunk_layout.is_sharded` for every distillable configuration, and still answers (rather than raises) for a stub unrecognized chunk grid metadata type where `chunk_layout` raises `TypeError`.
- First-codec rule: `[sharding, gzip]` yields `is_sharded == True` with two levels (documented divergence from `.shards`, which returns `None` there); a pipeline with an array→array codec before the sharding codec yields a single opaque level.
- Doc snippets exercised under `pytest --doctest-modules` like the existing accessor examples.

## Open questions

1. Should `Array.is_sharded` ship, or is `arr.chunk_layout.is_sharded` enough? (This draft includes it; it is one property and directly answers the issue title.)
2. Should the rename of the internal `NamedTuple` to `_ResolvedChunks` happen in the same PR (proposed) or a precursor refactor PR?
3. Should `chunk_layout` live under `zarr.experimental` first (like `ChunkGrid`/`ChunkSpec` did) and graduate, or ship directly in the stable namespace given the small surface?
4. Should the `metadata.chunks` check order be fixed alongside this work? `.chunks` currently raises for any non-regular grid *before* checking codecs ([`v3.py:569–581`](https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/metadata/v3.py)), but for a sharded array the inner chunk shape comes from the sharding codec and is a plain `tuple[int, ...]` regardless of the outer grid — reordering the checks makes `.chunks` answer correctly for sharded-rectilinear arrays with no type change. It is independent of `ChunkLayout` (and redundant once consumers migrate), but it is a one-line fix for a live downstream crash.
