"""Array specifications and how a codec chain transforms them.

Each array->array codec transforms the array it receives, so a codec's
configuration must be judged against the array that *reaches* it, not
against the document's top-level fields: `transpose` permutes the shape,
`cast_value` changes the data type, and a `sharding_indexed` codec that
follows either one sees the transformed array.

`ArrayParts` is every part of an array a codec will be handed, together
with their element type; `propagate` walks a chain handing each codec what
reaches it. There is no half-populated value: a codec receives `None` once
this package can no longer say what it operates on. An unknown codec might
change anything, so everything after one receives `None`, and so does
everything after the array->bytes boundary, where there is no array left.

Transitions are registered per array->array codec, next to that codec's
rules, via `spec_transition`. A modelled codec with no transition is
treated as unknown, so a forgotten transition fails closed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from zarr_metadata.rules._chunk_grid import ChunkGrid  # noqa: TC001
from zarr_metadata.v3._extension_points import CODECS, canonical_name
from zarr_metadata.v3._shape import entity_name
from zarr_metadata.v3.codec.kind import codec_kind_of_name

if TYPE_CHECKING:
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON


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


SpecTransition = Callable[[Mapping[str, object], "ArrayParts"], "ArrayParts | None"]
"""How one codec transforms what it receives.

Takes the codec's (shape-valid) configuration and the incoming parts, and
returns what the next codec sees, or `None` when this codec leaves nothing
determinable. A transition must never raise on the values the shape
validator admits.
"""

_TRANSITIONS: Final[dict[str, SpecTransition]] = {}


def spec_transition(codec: str) -> Callable[[SpecTransition], SpecTransition]:
    """Register how `codec` transforms the `ArrayParts` it receives.

    Only array->array codecs need one: array->bytes and bytes->bytes
    codecs end shape propagation by construction, so registering a
    transition for one is refused.
    """
    kind = codec_kind_of_name(codec)
    if kind != "array_array":
        msg = (
            f"spec transition registered for {codec!r}, which is "
            f"{kind or 'unknown'} rather than array_array; only array->array "
            "codecs transform the array spec"
        )
        raise ValueError(msg)

    def decorate(transition: SpecTransition) -> SpecTransition:
        _TRANSITIONS[canonical_name(CODECS, codec)] = transition
        return transition

    return decorate


def transitions_registered() -> frozenset[str]:
    """Every codec name with a registered spec transition."""
    return frozenset(_TRANSITIONS)


def propagate(
    codecs: Sequence[object],
    initial: ArrayParts | None,
    configuration_of: Callable[[object], Mapping[str, object] | None],
) -> Iterator[tuple[int, object, ArrayParts | None]]:
    """Yield `(index, codec, incoming)` for each codec in the chain.

    `incoming` is `None` once propagation has stopped: after an unknown
    codec, after a known codec whose configuration is not shape-valid,
    after a codec this package has no transition for, and after the
    array->bytes boundary, where there is no array to describe.
    `configuration_of` resolves a codec entry to its usable configuration
    (`entity_configuration` in practice; injected to keep this module free
    of the registry).
    """
    parts = initial
    for index, codec in enumerate(codecs):
        yield index, codec, parts
        if parts is None:
            continue
        name = entity_name(codec)
        kind = codec_kind_of_name(name) if name is not None else None
        if kind == "array_array":
            transition = _TRANSITIONS.get(canonical_name(CODECS, name or ""))
            configuration = configuration_of(codec)
            parts = (
                None
                if transition is None or configuration is None
                else transition(configuration, parts)
            )
        else:
            # An unknown codec may change anything; array->bytes consumes
            # the array; bytes->bytes never had one.
            parts = None


__all__ = [
    "ArrayParts",
    "SpecTransition",
    "propagate",
    "spec_transition",
    "transitions_registered",
]
