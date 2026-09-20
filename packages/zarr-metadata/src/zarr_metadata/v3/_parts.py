"""How an array is divided, and what a codec chain does to it.

A codec pipeline encodes one chunk, but the same pipeline encodes *every*
chunk, so a rule about a pipeline is a statement about all of them at
once: a shard's inner chunk shape must divide every chunk it will ever be
handed, not some representative one. `ChunkGrid` is what makes that
statement expressible, and `ArrayParts` is what one codec hands the next.

Three pieces of metadata divide an array: a document's `chunk_grid`, a
`sharding_indexed` codec's `chunk_shape` (a regular grid over the chunk
that codec receives), and that codec's shard index, whose shape the spec
derives from the other two. Each grid entity builds its own, because what
a grid divides an array into is the grid's own knowledge.

Per dimension, and plurally
---------------------------
`extents` holds one entry per dimension: the set of lengths that
dimension's chunks take. A regular grid gives singletons; a rectilinear
grid gives `{30, 34}` on an axis whose chunks differ; `None` marks an axis
this package cannot read. `rank` survives even when no extent does,
because every chunk of an array has the array's rank whatever divides it.

Collapsing any of that loses real judgments. A rectilinear grid uniform on
one axis still pins that axis, and a shard is judged there while declining
on the others.

Prior art
---------
zarrs builds its grid from metadata *and* the array shape
(`ChunkGrid::create(metadata, array_shape)`) because neither determines a
grid alone, keeps `dimensionality()` total rather than optional, and
reports `chunk_edge_lengths(dimension)` per dimension for the reason
above. Its codec chain distinguishes "no global grid, but one per chunk"
(`ChunkGridMapped::ChunkLocal`) from "nothing known" (`::None`); an
`extents` entry of `None` beside a known `rank` is the per-dimension form
of that distinction.

- https://github.com/zarrs/zarrs/blob/main/zarrs_chunk_grid/src/lib.rs
- https://github.com/zarrs/zarrs/blob/main/zarrs_codec/src/lib.rs
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeAlias, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON


Extents: TypeAlias = "tuple[frozenset[int] | None, ...]"
"""One entry per dimension: the lengths that dimension's chunks take.

A singleton is a uniform axis. `None` is an axis whose lengths this
package cannot determine — distinct from an empty set, which would claim
the axis has no chunks at all.
"""


def _positive_int(value: object) -> int | None:
    """`value` as a chunk length, or None if it is not a usable one."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 1 else None


def _rank_of(array_shape: object) -> int | None:
    """The number of dimensions `array_shape` declares, if it declares any."""
    if not isinstance(array_shape, tuple):
        return None
    dimensions = cast("tuple[object, ...]", array_shape)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in dimensions):
        return None
    return len(dimensions)


def _uniform(lengths: Sequence[object]) -> Extents:
    """Extents for a grid whose chunks are the same everywhere."""
    return tuple(
        None if (length := _positive_int(value)) is None else frozenset({length})
        for value in lengths
    )


def _rectilinear_axis(spec: object) -> frozenset[int] | None:
    """The lengths one rectilinear dimension's chunks take.

    A bare integer is a regular step, so every chunk is that long. An
    explicit list names them, with `[size, count]` pairs standing for
    repeats; the distinct sizes are what any divisibility question needs.
    """
    step = _positive_int(spec)
    if step is not None:
        return frozenset({step})
    if not isinstance(spec, tuple):
        return None
    lengths: set[int] = set()
    for item in cast("tuple[object, ...]", spec):
        size = _positive_int(item)
        if size is None and isinstance(item, tuple):
            pair = cast("tuple[object, ...]", item)
            if len(pair) != 2 or _positive_int(pair[1]) is None:
                return None
            size = _positive_int(pair[0])
        if size is None:
            return None
        lengths.add(size)
    return frozenset(lengths) if len(lengths) != 0 else None


@dataclass(frozen=True, slots=True)
class ChunkGrid:
    """The division of an array into the parts a codec pipeline encodes.

    `metadata` is the grid as the document spells it, kept so that a rule
    for a grid this package does not model can still read its own
    configuration. It is absent for a grid this package derived rather
    than read — the regular grid a sharding codec imposes, or a transposed
    grid — so nothing may validate it or report a location into it.
    """

    rank: int | None
    extents: Extents | None
    metadata: ZarrV3MetadataFieldJSON | None = None

    @classmethod
    def unreadable(cls, array_shape: object) -> ChunkGrid:
        """A grid nothing is known about but the rank the array pins.

        Every third-party grid, and every modelled one whose own metadata
        could not be read: the array still has a rank, and a rule about
        rank is still answerable.
        """
        rank = _rank_of(array_shape)
        return cls(rank, None if rank is None else (None,) * rank)

    @classmethod
    def derived(cls, extents: Extents) -> ChunkGrid:
        """A grid this package computed rather than read from a document."""
        return cls(len(extents), extents)

    @classmethod
    def regular(cls, lengths: Sequence[object]) -> ChunkGrid:
        """The regular grid a sharding codec's `chunk_shape` imposes."""
        return cls.derived(_uniform(lengths))

    def permuted(self, order: Sequence[int]) -> ChunkGrid:
        """This grid with its dimensions reordered by `order`.

        A transposed grid is still a grid — permuting a regular one gives
        a regular one — but it is no longer the grid the document wrote,
        so the metadata does not survive the trip.

        Declines on anything that is not a permutation of this grid's rank.
        The caller checks that too and reports it, but an order is only
        shape-validated as a tuple of integers, so this must not be the
        thing that decides whether a validator raises `IndexError`.
        """
        if self.extents is None or sorted(order) != list(range(len(self.extents))):
            return ChunkGrid(self.rank, None)
        return ChunkGrid.derived(tuple(self.extents[axis] for axis in order))

    def axis(self, dimension: int) -> frozenset[int] | None:
        """The lengths `dimension`'s chunks take, or None if undetermined."""
        if self.extents is None or dimension >= len(self.extents):
            return None
        return self.extents[dimension]


UNKNOWN_GRID: ChunkGrid = ChunkGrid(None, None)
"""A grid nothing is known about — not even how many dimensions it has."""


def shard_index_grid(shard: ChunkGrid, inner: Sequence[object]) -> ChunkGrid:
    """The grid of a shard's index array.

    The spec derives it from the two shapes around it: "The index is an
    array with 64-bit unsigned integers with a shape that matches the
    chunks per shard tuple with an appended dimension of size 2." The
    index is one array rather than a divided one, so each axis holds a
    single length — except that under a rectilinear grid the shard itself
    varies, so the chunk count varies with it and the axis holds every
    value it takes.
    """
    inner_extents = _uniform(inner)
    trailing: frozenset[int] | None = frozenset({2})
    if shard.extents is None or len(shard.extents) != len(inner_extents):
        return ChunkGrid.derived((*(None,) * len(inner_extents), trailing))
    counts: list[frozenset[int] | None] = []
    for lengths, divisor in zip(shard.extents, inner_extents, strict=True):
        if lengths is None or divisor is None:
            counts.append(None)
            continue
        step = next(iter(divisor))
        quotients = {length // step for length in lengths if length % step == 0}
        counts.append(frozenset(quotients) if len(quotients) == len(lengths) else None)
    return ChunkGrid.derived((*counts, trailing))


@dataclass(frozen=True, slots=True)
class ArrayParts:
    """Every part of an array a codec will be handed, and their type.

    The parts an array is divided into, not the fields of its metadata.
    Plural deliberately: one pipeline encodes every chunk, so a rule about
    it quantifies over all of them — a shard's inner chunk shape must
    divide *every* chunk, which under a rectilinear grid is several
    different lengths.

    `data_type` is the metadata-field value verbatim, because rules compare
    it by name, and it is `None` where the element type is undetermined
    while the array itself is not. That happens inside a shard: the inner
    grid is the sharding codec's own `chunk_shape` whatever reached it, so
    an unreadable codec upstream costs the type and not the parts. `None`
    in place of the whole value means something else again — that there is
    no array here at all, past the array->bytes boundary or beyond a codec
    that could have changed anything.
    """

    grid: ChunkGrid
    data_type: ZarrV3MetadataFieldJSON | None

    def with_grid(self, grid: ChunkGrid) -> ArrayParts:
        return replace(self, grid=grid)

    def with_data_type(self, data_type: ZarrV3MetadataFieldJSON | None) -> ArrayParts:
        return replace(self, data_type=data_type)


__all__ = [
    "UNKNOWN_GRID",
    "ArrayParts",
    "ChunkGrid",
    "Extents",
    "shard_index_grid",
]
