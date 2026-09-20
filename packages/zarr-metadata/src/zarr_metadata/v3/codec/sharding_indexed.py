"""
Sharding-indexed codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired, Self, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    CODECS,
    CodecKind,
    Coerced,
    Loc,
    MemberTypes,
    MetadataEntity,
    is_int,
    one_of,
    problem,
    sequence_of,
)

if TYPE_CHECKING:
    from zarr_metadata.v3._registry import Context

from typing_extensions import TypedDict

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
) -> tuple[tuple[MetadataEntity | object, ...], tuple[ValidationProblem, ...]]:
    """Every entry of one pipeline, read in `context`."""
    coerced: list[MetadataEntity | object] = []
    problems: list[ValidationProblem] = []
    for index, entry in enumerate(entries):
        codec, found = context.coerce(CODECS, entry, (*loc, index))
        coerced.append(codec)
        problems.extend(found)
    return tuple(coerced), tuple(problems)


@dataclass(frozen=True)
class ShardingIndexedCodec(MetadataEntity):
    """The `sharding_indexed` codec, coerced from its metadata.

    Holds two codec pipelines, so it is one of the few entities that
    needs the scope it is being read in: an entry of either pipeline is
    itself an entity, read the same way this one was.
    """

    chunk_shape: tuple[int, ...] = ()
    codecs: tuple[MetadataEntity | object, ...] = ()
    index_codecs: tuple[MetadataEntity | object, ...] = ()
    index_location: ShardingIndexLocation | None = None

    identifier: ClassVar[str] = SHARDING_INDEXED_CODEC_NAME
    kind: ClassVar[CodecKind] = "array_bytes"

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {
        "chunk_shape": (True, sequence_of(is_int)),
        "codecs": (True, _is_field_tuple),
        "index_codecs": (True, _is_field_tuple),
        "index_location": (False, one_of(SHARDING_INDEX_LOCATION)),
    }

    @classmethod
    def coerce(cls, value: object, context: "Context") -> Coerced[Self]:
        shard, problems = super().coerce(value, context)
        if shard is None:
            return None, problems
        inner, from_inner = _coerce_pipeline(shard.codecs, context, ("codecs",))
        index, from_index = _coerce_pipeline(shard.index_codecs, context, ("index_codecs",))
        return (
            replace(shard, codecs=inner, index_codecs=index),
            (*problems, *from_inner, *from_index),
        )

    def problems(self) -> tuple[ValidationProblem, ...]:
        """This shard's own values, and those of the codecs it holds.

        Whether the two pipelines are well *formed* -- one array-to-bytes
        codec, in the right order -- spans the whole chain, so the rules
        layer asks that.
        """
        found: list[ValidationProblem] = [
            ValidationProblem(
                ("chunk_shape", position),
                f"expected a positive chunk extent, got {extent}",
                "invalid_value",
            )
            for position, extent in enumerate(self.chunk_shape)
            if extent < 1
        ]
        for member in ("codecs", "index_codecs"):
            for position, codec in enumerate(cast("tuple[object, ...]", getattr(self, member))):
                if isinstance(codec, MetadataEntity):
                    found.extend(
                        ValidationProblem((member, position, *entry.loc), entry.message, entry.kind)
                        for entry in codec.problems()
                    )
        return tuple(found)

    def configuration(self) -> dict[str, object]:
        """The two pipelines in their canonical spelling, entry by entry."""
        members = super().configuration()
        for member in ("codecs", "index_codecs"):
            members[member] = tuple(
                entry.to_json() if isinstance(entry, MetadataEntity) else entry
                for entry in cast("tuple[object, ...]", members[member])
            )
        return members

    def to_json(self) -> ShardingIndexedCodecObject:
        return cast("ShardingIndexedCodecObject", super().to_json())
