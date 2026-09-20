"""
Rectilinear chunk grid (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired, Self, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    ChunkGridEntity,
    Loc,
    MemberTypes,
    ValueRoutine,
    is_integer,
    one_of,
    problem,
)
from zarr_metadata.v3._parts import ChunkGrid

if TYPE_CHECKING:
    from collections.abc import Sequence


RECTILINEAR_CHUNK_GRID_NAME: Final = "rectilinear"
"""The `name` field value of the rectilinear chunk grid."""

RECTILINEAR_CHUNK_GRID_KIND: Final = ("inline",)
"""The `kind` values the rectilinear grid defines.

Only `inline` so far: the extents are written into the metadata. The
member exists so a later kind can put them somewhere else.
"""

RectilinearChunkGridName = Literal["rectilinear"]
"""Literal type of the `name` field of the rectilinear chunk grid."""

RectilinearDimSpec = int | tuple[int | tuple[int, int], ...]
"""JSON shape for one dimension's rectilinear spec.

Either a bare integer (uniform shorthand for a regular dimension within
a rectilinear grid), or a tuple of integers and/or `[value, count]` RLE
pairs.
"""


class RectilinearChunkGridConfiguration(TypedDict, closed=True):
    """Configuration for the rectilinear chunk grid."""

    kind: Literal["inline"]
    chunk_shapes: tuple[RectilinearDimSpec, ...]


class RectilinearChunkGridObject(TypedDict, closed=True):
    """Rectilinear chunk grid metadata in object form."""

    name: RectilinearChunkGridName
    configuration: RectilinearChunkGridConfiguration
    must_understand: NotRequired[bool]


RectilinearChunkGridMetadata = RectilinearChunkGridObject
"""Permitted JSON shape for rectilinear chunk grid metadata.

`kind` and `chunk_shapes` are required, so only the object form is valid;
the short-hand-name form is not permitted by the spec for this grid.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md#L59-L62
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564
"""


def canonical_dim_spec(spec: RectilinearDimSpec) -> RectilinearDimSpec:
    """One dimension's chunk sizes in their simplest equivalent form.

    Runs of equal sizes collapse to `[size, count]` pairs, because that is
    the spelling that does not grow with the number of chunks: a million
    equal chunks is two numbers, not a million. A run of one stays a bare
    size, and `[size, 1]` collapses to one, since a pair says nothing extra
    there. Adjacent spellings of the same size merge, which is what makes
    this idempotent: `[[32, 2], 32]` and `[32, [32, 2]]` both become
    `[[32, 3]]`.

    A dimension-level bare integer is left alone. It is a *step* that
    repeats until it covers the extent, so it is not equivalent to any
    fixed list — expanding it would pin a grid that currently adapts, and
    the two would diverge the moment the array were resized. For the same
    reason a one-element list is never collapsed to a bare integer:
    `[32]` declares exactly one chunk and `32` declares as many as it takes.

    Assumes a spec the shape validator has already accepted.
    """
    if not isinstance(spec, tuple):
        return spec
    runs: list[tuple[int, int]] = []
    for entry in spec:
        size, count = entry if isinstance(entry, tuple) else (entry, 1)
        if len(runs) != 0 and runs[-1][0] == size:
            runs[-1] = (size, runs[-1][1] + count)
        else:
            runs.append((size, count))
    return tuple(size if count == 1 else (size, count) for size, count in runs)


def canonical_chunk_shapes(
    chunk_shapes: tuple[RectilinearDimSpec, ...],
) -> tuple[RectilinearDimSpec, ...]:
    """Every dimension's chunk sizes in their simplest equivalent form."""
    return tuple(canonical_dim_spec(spec) for spec in chunk_shapes)


__all__ = [
    "RECTILINEAR_CHUNK_GRID_KIND",
    "RECTILINEAR_CHUNK_GRID_NAME",
    "RectilinearChunkGrid",
    "RectilinearChunkGridConfiguration",
    "RectilinearChunkGridMetadata",
    "RectilinearChunkGridName",
    "RectilinearChunkGridObject",
    "RectilinearDimSpec",
    "canonical_chunk_shapes",
    "canonical_dim_spec",
]


def _is_dim_specs(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """One spec per dimension, each a bare extent or a list of entries.

    An entry is an extent or a `[size, count]` run. The nesting is why
    this is written out rather than composed from `sequence_of`.
    """
    if not isinstance(value, tuple):
        return problem(loc, f"expected an array of dimension specs, got {value!r}")
    specs = cast("tuple[object, ...]", value)
    found: list[ValidationProblem] = []
    for dim, spec in enumerate(specs):
        at: Loc = (*loc, dim)
        if is_integer(spec):
            continue
        if not isinstance(spec, tuple):
            found.extend(
                problem(
                    at,
                    "expected an integer or an array of integers / [value, count] pairs, "
                    f"got {spec!r}",
                )
            )
            continue
        for position, item in enumerate(cast("tuple[object, ...]", spec)):
            if is_integer(item):
                continue
            entries = cast("tuple[object, ...]", item) if isinstance(item, tuple) else ()
            if len(entries) == 2 and all(is_integer(part) for part in entries):
                continue
            found.extend(
                problem(
                    (*at, position), f"expected an integer or a [value, count] pair, got {item!r}"
                )
            )
    return tuple(found)


def _covered_extent(spec: tuple[int | tuple[int, int], ...]) -> int | None:
    """How much of a dimension an explicit spec covers, or None.

    None when any entry is non-positive: `problems` reports that, and a
    total computed from a nonsense entry would be nonsense too.
    """
    total = 0
    for item in spec:
        if isinstance(item, int):
            if item < 1:
                return None
            total += item
            continue
        size, count = item
        if size < 1 or count < 1:
            return None
        total += size * count
    return total


def _axis_lengths(spec: RectilinearDimSpec) -> frozenset[int] | None:
    """The lengths one dimension's chunks take, or None if undetermined.

    A bare integer is a regular step, so every chunk is that long. An
    explicit list names them, with `[size, count]` pairs standing for
    repeats; the distinct sizes are what any divisibility question needs.
    None for a non-positive length, which `problems` reports -- a grid
    that does not tile answers nothing about what divides it.
    """
    if isinstance(spec, int):
        return frozenset({spec}) if spec > 0 else None
    lengths: set[int] = set()
    for item in spec:
        size = item if isinstance(item, int) else item[0]
        if size < 1 or (not isinstance(item, int) and item[1] < 1):
            return None
        lengths.add(size)
    return frozenset(lengths) if len(lengths) != 0 else None


def _value_problems(
    **members: Unpack[RectilinearChunkGridConfiguration],
) -> tuple[ValidationProblem, ...]:
    """Every chunk extent, bare or run-length encoded, must be positive.

    A run's count must be positive too: a run of zero chunks is a way
    of writing nothing at all, and the empty spelling already exists.
    """
    found: list[ValidationProblem] = []
    for dim, spec in enumerate(members["chunk_shapes"]):
        loc: tuple[str | int, ...] = ("chunk_shapes", dim)
        if isinstance(spec, int):
            if spec < 1:
                found.extend(
                    problem(loc, f"expected a positive chunk extent, got {spec}", "invalid_value")
                )
            continue
        for position, item in enumerate(spec):
            if isinstance(item, int):
                if item < 1:
                    found.extend(
                        problem(
                            (*loc, position),
                            f"expected a positive chunk extent, got {item}",
                            "invalid_value",
                        )
                    )
            elif item[0] < 1 or item[1] < 1:
                found.extend(
                    problem(
                        (*loc, position),
                        f"expected a positive [size, count] pair, got {item!r}",
                        "invalid_value",
                    )
                )
    return tuple(found)


@dataclass(frozen=True)
class RectilinearChunkGrid(ChunkGridEntity):
    """The `rectilinear` chunk grid, coerced from its metadata."""

    kind: Literal["inline"]
    chunk_shapes: tuple[RectilinearDimSpec, ...]

    identifier: ClassVar[str] = RECTILINEAR_CHUNK_GRID_NAME

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {
        "kind": (True, one_of(RECTILINEAR_CHUNK_GRID_KIND)),
        "chunk_shapes": (True, _is_dim_specs),
    }

    value_problems: ClassVar[ValueRoutine] = staticmethod(_value_problems)

    def shape_problems(self, array_shape: object) -> tuple[ValidationProblem, ...]:
        """One spec per dimension, and explicit specs must cover it.

        A bare integer is uniform shorthand, so it covers whatever the
        dimension turns out to be and imposes no sum; an explicit list
        names every chunk, so the names have to add up.
        """
        if not isinstance(array_shape, (list, tuple)):
            return ()
        extents = tuple(cast("Sequence[object]", array_shape))
        if len(self.chunk_shapes) != len(extents):
            return problem(
                ("chunk_shapes",),
                f"chunk_shapes has {len(self.chunk_shapes)} entries but shape has "
                f"{len(extents)} dimensions",
                "invalid_value",
            )
        found: list[ValidationProblem] = []
        for dim, (spec, extent) in enumerate(zip(self.chunk_shapes, extents, strict=True)):
            if isinstance(spec, int) or not is_integer(extent):
                continue
            total = _covered_extent(spec)
            if total is not None and total < extent:
                found.extend(
                    problem(
                        ("chunk_shapes", dim),
                        f"chunk sizes sum to {total} but must cover shape[{dim}] extent {extent}",
                        "invalid_value",
                    )
                )
        return tuple(found)

    def grid(self, array_shape: object) -> ChunkGrid:
        """The distinct lengths each axis's chunks take.

        Plural per axis, which is the point of a rectilinear grid: an
        axis of `[30, 34]` gives `{30, 34}`, and anything asking about
        divisibility has to hold for both.
        """
        return ChunkGrid.derived(tuple(_axis_lengths(spec) for spec in self.chunk_shapes))

    def canonical(self) -> Self:
        """Run-length encoded, which is the spelling that does not grow.

        Two dimension specs listing the same extents describe the same
        grid, and the encoded one stays the same size as the array grows.
        """
        return replace(self, chunk_shapes=canonical_chunk_shapes(self.chunk_shapes))

    def to_json(self) -> RectilinearChunkGridObject:
        return cast("RectilinearChunkGridObject", super().to_json())
