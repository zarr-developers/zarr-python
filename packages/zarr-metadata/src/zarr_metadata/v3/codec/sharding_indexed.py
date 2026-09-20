"""
Sharding-indexed codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired, Self, cast

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._chain import chain_problems
from zarr_metadata.v3._entity import (
    CODECS,
    CodecEntity,
    CodecKind,
    Loc,
    MemberTypes,
    Opaque,
    is_int,
    one_of,
    problem,
    sequence_of,
)
from zarr_metadata.v3._parts import (
    UNKNOWN_GRID,
    ArrayParts,
    ChunkGrid,
    shard_index_grid,
)
from zarr_metadata.v3.data_type.uint64 import Uint64DataType

if TYPE_CHECKING:
    from zarr_metadata.v3._registry import Context

from typing_extensions import TypedDict, Unpack

from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON

SHARDING_INDEXED_CODEC_NAME: Final = "sharding_indexed"
"""The `name` field value of the `sharding_indexed` codec."""

ShardingIndexedCodecName = Literal["sharding_indexed"]
"""Literal type of the `name` field of the `sharding_indexed` codec."""

ShardingIndexLocation = Literal["start", "end"]
"""Literal type of the position of the shard index within the encoded shard."""

SHARDING_INDEX_LOCATION: Final = ("start", "end")
"""Tuple of permitted values for the `index_location` field of the `sharding_indexed` codec."""


class ShardingIndexedCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `sharding_indexed` codec.

    `chunk_shape` is the shape of inner chunks along each dimension;
    it must evenly divide the shard shape.

    `codecs` is the codec pipeline applied to each inner chunk; exactly
    one array-to-bytes codec is required.

    `index_codecs` is the codec pipeline applied to the shard index;
    it must be deterministic (no variable-size compression).
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L147-L155

    `index_location` defaults to `"end"` per the spec.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L157-L161
    """

    chunk_shape: tuple[int, ...]
    codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    index_codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    index_location: NotRequired[ShardingIndexLocation]


class ShardingIndexedCodecObject(TypedDict, closed=True):
    """`sharding_indexed` codec metadata in object form."""

    name: ShardingIndexedCodecName
    configuration: ShardingIndexedCodecConfiguration
    must_understand: NotRequired[bool]


ShardingIndexedCodecMetadata = ShardingIndexedCodecObject
"""Permitted JSON shape for `sharding_indexed` codec metadata.

The configuration has multiple required keys (`chunk_shape`, `codecs`,
`index_codecs`), so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L141-L155 (required members)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""

__all__ = [
    "SHARDING_INDEXED_CODEC_NAME",
    "SHARDING_INDEX_LOCATION",
    "ShardingIndexLocation",
    "ShardingIndexedCodec",
    "ShardingIndexedCodecConfiguration",
    "ShardingIndexedCodecMetadata",
    "ShardingIndexedCodecName",
    "ShardingIndexedCodecObject",
]


def _is_field_tuple(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """An array of metadata fields -- their names are checked on recursion."""
    if not isinstance(value, tuple):
        return problem(loc, f"expected an array of codecs, got {value!r}")
    entries = cast("tuple[object, ...]", value)
    return tuple(
        found
        for index, entry in enumerate(entries)
        if not isinstance(entry, (str, Mapping))
        for found in problem((*loc, index), f"expected a metadata field, got {entry!r}")
    )


def _coerce_pipeline(
    entries: tuple[object, ...], context: "Context", loc: Loc
) -> tuple[tuple[CodecEntity | Opaque, ...], tuple[ValidationProblem, ...]]:
    """Every entry of one pipeline, read in `context`."""
    coerced: list[CodecEntity | Opaque] = []
    problems: list[ValidationProblem] = []
    for index, entry in enumerate(entries):
        codec, found = context.coerce(CODECS, entry, (*loc, index))
        coerced.append(codec)
        problems.extend(found)
    return tuple(coerced), tuple(problems)


def _canonical_pipeline(
    codecs: tuple[CodecEntity | Opaque, ...],
) -> tuple[CodecEntity | Opaque, ...]:
    """Each codec canonicalized; one out of scope is left as written."""
    return tuple(codec.canonical() if isinstance(codec, CodecEntity) else codec for codec in codecs)


class ShardingIndexedMembers(TypedDict):
    """A shard's members as the entity holds them.

    Not `ShardingIndexedCodecConfiguration`, which describes the JSON: by
    the time values are judged, `prepare` has read the two pipelines, so
    these are codecs rather than the metadata fields that named them.
    """

    chunk_shape: tuple[int, ...]
    codecs: tuple[CodecEntity | Opaque, ...]
    index_codecs: tuple[CodecEntity | Opaque, ...]
    index_location: NotRequired[ShardingIndexLocation]


@dataclass(frozen=True)
class ShardingIndexedCodec(CodecEntity):
    """The `sharding_indexed` codec, coerced from its metadata.

    Holds two codec pipelines, so it is one of the few entities that
    needs the scope it is being read in: an entry of either pipeline is
    itself an entity, read the same way this one was.
    """

    chunk_shape: tuple[int, ...]
    codecs: tuple[CodecEntity | Opaque, ...]
    index_codecs: tuple[CodecEntity | Opaque, ...]
    index_location: ShardingIndexLocation | UNSET = UNSET

    identifier: ClassVar[str] = SHARDING_INDEXED_CODEC_NAME
    variable_size: ClassVar[bool] = True
    kind: ClassVar[CodecKind] = "array_bytes"

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {
        "chunk_shape": (True, sequence_of(is_int)),
        "codecs": (True, _is_field_tuple),
        "index_codecs": (True, _is_field_tuple),
        "index_location": (False, one_of(SHARDING_INDEX_LOCATION)),
    }

    @staticmethod
    def value_problems(
        **members: Unpack[ShardingIndexedMembers],
    ) -> tuple[ValidationProblem, ...]:
        """Every inner chunk extent must be at least one element.

        Nothing about the two pipelines: their codecs are entities, and an
        entity exists only if its own values are allowed.
        """
        return tuple(
            ValidationProblem(
                ("chunk_shape", position),
                f"expected a positive chunk extent, got {extent}",
                "invalid_value",
            )
            for position, extent in enumerate(members["chunk_shape"])
            if extent < 1
        )

    @classmethod
    def prepare(
        cls, members: dict[str, object], context: "Context"
    ) -> tuple[dict[str, object], tuple[ValidationProblem, ...]]:
        """Both pipelines, read in this scope."""
        inner, from_inner = _coerce_pipeline(
            cast("tuple[object, ...]", members["codecs"]), context, ("configuration", "codecs")
        )
        index, from_index = _coerce_pipeline(
            cast("tuple[object, ...]", members["index_codecs"]),
            context,
            ("configuration", "index_codecs"),
        )
        return (
            {**members, "codecs": inner, "index_codecs": index},
            (*from_inner, *from_index),
        )

    def incoming_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """This shard against the array reaching it, and its two pipelines.

        One sharding configuration encodes every chunk, so its inner
        shape has to divide all of them. Under a rectilinear grid an axis
        has several lengths and the inner extent must divide each; an axis
        whose lengths are unknown declines while the others are judged.
        """
        found = list(self._inner_chunk_problems(incoming))
        # Both pipelines start from this codec's own configuration and
        # from the spec, so neither waits on what reached the codec. An
        # unreadable codec upstream costs the element type and the
        # enclosing extents; it does not make the inner chunk shape
        # unknown, and the index is a `uint64` array whatever precedes it.
        outer = incoming.grid if incoming is not None else UNKNOWN_GRID
        found.extend(
            chain_problems(
                self.codecs,
                ArrayParts(
                    ChunkGrid.regular(self.chunk_shape),
                    incoming.data_type if incoming is not None else None,
                ),
                ("codecs",),
            )
        )
        found.extend(
            chain_problems(
                self.index_codecs,
                ArrayParts(shard_index_grid(outer, self.chunk_shape), Uint64DataType()),
                ("index_codecs",),
            )
        )
        found.extend(
            ValidationProblem(
                ("index_codecs", index),
                f"{type(codec).identifier!r} produces variable-size output; "
                "index_codecs must be fixed-size",
                "invalid_value",
            )
            for index, codec in enumerate(self.index_codecs)
            if isinstance(codec, CodecEntity) and type(codec).variable_size
        )
        return tuple(found)

    def _inner_chunk_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """Whether the inner chunk divides every chunk this shard receives."""
        if incoming is None or incoming.grid.rank is None:
            return ()
        if len(self.chunk_shape) != incoming.grid.rank:
            return problem(
                ("chunk_shape",),
                f"chunk_shape has {len(self.chunk_shape)} entries but the incoming array "
                f"has {incoming.grid.rank} dimensions",
                "invalid_value",
            )
        found: list[ValidationProblem] = []
        for position, extent in enumerate(self.chunk_shape):
            lengths = incoming.grid.axis(position)
            if lengths is None or extent < 1:
                continue
            indivisible = sorted(length for length in lengths if length % extent != 0)
            if len(indivisible) != 0:
                found.extend(
                    problem(
                        ("chunk_shape", position),
                        f"inner chunk extent {extent} does not evenly divide the incoming "
                        f"extent {indivisible[0]}",
                        "invalid_value",
                    )
                )
        return tuple(found)

    def canonical(self) -> Self:
        """Each codec of each pipeline in its own canonical form."""
        return replace(
            self,
            codecs=_canonical_pipeline(self.codecs),
            index_codecs=_canonical_pipeline(self.index_codecs),
        )

    def configuration(self) -> dict[str, object]:
        """The two pipelines in their canonical spelling, entry by entry."""
        members = super().configuration()
        for member in ("codecs", "index_codecs"):
            members[member] = tuple(
                entry.to_json() if isinstance(entry, CodecEntity) else entry.json
                for entry in cast("tuple[CodecEntity | Opaque, ...]", members[member])
            )
        return members

    def to_json(self) -> ShardingIndexedCodecObject:
        return cast("ShardingIndexedCodecObject", super().to_json())
