"""Array specifications and how a codec chain transforms them.

Each array->array codec transforms the array it receives, so a codec's
configuration must be judged against the array that *reaches* it, not
against the document's top-level fields: `transpose` permutes the shape,
`cast_value` changes the data type, and a `sharding_indexed` codec that
follows either one sees the transformed array.

`ArraySpec` is the array a codec receives; `propagate` walks a chain
handing each codec its incoming spec. A field is `None` once this package
can no longer determine it. An unknown codec might change anything, so
every codec after one receives `NOTHING_KNOWN` and rules that need a
field decline rather than guess. Shape stops at the array->bytes
boundary; the data type carries through.

Transitions are registered per array->array codec, next to that codec's
rules, via `spec_transition`. A modelled codec with no transition is
treated as unknown, so a forgotten transition fails closed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from zarr_metadata.v3._extension_points import CODECS, canonical_name
from zarr_metadata.v3._shape import entity_name
from zarr_metadata.v3.codec.kind import codec_kind_of_name

if TYPE_CHECKING:
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """The array a codec receives; a field is `None` when undetermined.

    `data_type` is the metadata-field value verbatim (a bare name or a
    name/configuration object) because rules compare it by name.
    """

    shape: tuple[int, ...] | None
    data_type: ZarrV3MetadataFieldJSON | None

    def with_shape(self, shape: tuple[int, ...] | None) -> ArraySpec:
        return replace(self, shape=shape)

    def with_data_type(self, data_type: ZarrV3MetadataFieldJSON | None) -> ArraySpec:
        return replace(self, data_type=data_type)


NOTHING_KNOWN: Final = ArraySpec(None, None)
"""The spec past a point where nothing about the array can be determined.

Compare by equality: a spec can arrive here field by field and is then
equal to this constant without being it.
"""


SpecTransition = Callable[[Mapping[str, object], ArraySpec], ArraySpec]
"""How one codec transforms the spec it receives.

Takes the codec's (shape-valid) configuration and the incoming spec, and
returns the outgoing one. A transition must never raise on the values the
shape validator admits; a field it cannot determine becomes `None`.
"""

_TRANSITIONS: Final[dict[str, SpecTransition]] = {}


def spec_transition(codec: str) -> Callable[[SpecTransition], SpecTransition]:
    """Register how `codec` transforms an incoming `ArraySpec`.

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
    initial: ArraySpec,
    configuration_of: Callable[[object], Mapping[str, object] | None],
) -> Iterator[tuple[int, object, ArraySpec]]:
    """Yield `(index, codec, incoming_spec)` for each codec in the chain.

    `incoming_spec` is `NOTHING_KNOWN` once propagation has stopped: after
    an unknown codec, after a known codec whose configuration is not
    shape-valid, or after a codec this package has no transition for.
    `configuration_of` resolves a codec entry to its usable configuration
    (`entity_configuration` in practice; injected to keep this module free
    of the registry).
    """
    spec = initial
    for index, codec in enumerate(codecs):
        yield index, codec, spec
        if spec == NOTHING_KNOWN:
            continue
        name = entity_name(codec)
        kind = codec_kind_of_name(name) if name is not None else None
        if kind is None:
            spec = NOTHING_KNOWN
        elif kind == "array_array":
            transition = _TRANSITIONS.get(canonical_name(CODECS, name or ""))
            configuration = configuration_of(codec)
            if transition is None or configuration is None:
                spec = NOTHING_KNOWN
            else:
                spec = transition(configuration, spec)
        else:
            # array->bytes: the array is gone; bytes->bytes: never had one.
            spec = spec.with_shape(None)


def initial_spec(document: Mapping[str, object], chunk_shape: tuple[int, ...] | None) -> ArraySpec:
    """The spec entering a document's top-level codec chain.

    The array a chunk pipeline encodes is one chunk, so the incoming shape
    is the chunk grid's chunk shape (`None` if the grid is not a regular
    grid this package can read). The data type is the document's own.
    """
    data_type = document.get("data_type")
    if not isinstance(data_type, (str, Mapping)):
        data_type = None
    return ArraySpec(chunk_shape, data_type)  # type: ignore[arg-type]


__all__ = [
    "NOTHING_KNOWN",
    "ArraySpec",
    "SpecTransition",
    "initial_spec",
    "propagate",
    "spec_transition",
    "transitions_registered",
]
