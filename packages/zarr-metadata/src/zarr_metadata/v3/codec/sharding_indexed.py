"""
Sharding-indexed codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
"""

from collections.abc import Iterator, Mapping
from typing import Annotated, Final, Literal, NotRequired, cast

from annotated_types import Ge
from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import (
    AcceptedField,
    Chunk,
    CodecDefinition,
    Lengths,
    Nested,
    ZarrV3CodecField,
    ZarrV3StaticCodecField,
)
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE, UINT64_DATA_TYPE_NAME

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

    `index_codecs` is the codec pipeline applied to the shard index. It may
    contain only codecs of static size, because a reader locates the index
    by a size that it knows before reading it. A compressor in this
    pipeline is therefore rejected.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L147-L155

    `index_location` defaults to `"end"` per the spec.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L157-L161
    """

    chunk_shape: tuple[Annotated[int, Ge(1)], ...]
    codecs: tuple[ZarrV3CodecField, ...]
    index_codecs: tuple[ZarrV3StaticCodecField, ...]
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
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L129-L138 (`chunk_shape`)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L141-L155 (`codecs` and `index_codecs`, the members the spec marks required)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""


def _chunk_rules(
    configuration: ShardingIndexedCodecConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """Check that `chunk_shape` has one inner chunk length per axis of the input shard, and that each length divides every shard length along its axis.

    "The length of the chunk_shape array must match the number of
    dimensions of the shard shape to which this sharding codec is applied,
    and the inner chunk shape along each dimension must evenly divide the
    size of the shard shape"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L131-L135).
    The shard is the chunk that the codec receives, after any earlier
    codecs have been applied. For example, a transposed shard is divided
    along its transposed axes.
    """
    chunk_shape = configuration["chunk_shape"]
    lengths = chunk.lengths
    if lengths is None:
        return
    if len(chunk_shape) != len(lengths):
        yield ValidationProblem(
            ("chunk_shape",),
            f"expected {len(lengths)} inner chunk lengths, one per axis of the chunk the codec "
            f"is handed, got {len(chunk_shape)}",
            "invalid_value",
        )
        return
    for axis, (inner, shard) in enumerate(zip(chunk_shape, lengths, strict=True)):
        undivided = [] if shard is None else [length for length in shard if length % inner != 0]
        if len(undivided) != 0:
            yield ValidationProblem(
                ("chunk_shape", axis),
                f"expected an inner chunk length that divides each length the shards take "
                f"along this axis, got {inner}, which does not divide {min(undivided)}",
                "invalid_value",
            )


_UINT64: Final = AcceptedField(
    json=UINT64_DATA_TYPE_NAME,
    name=UINT64_DATA_TYPE_NAME,
    definition=UINT64_DATA_TYPE,
    configuration={},
)
"""The data type of a shard index. The spec fixes it, regardless of what the scope holds."""


def _pipelines(
    configuration: ShardingIndexedCodecConfiguration, nested: Nested, chunk: Chunk
) -> Mapping[str, Chunk]:
    """Return the input chunk of each nested pipeline: the inner codecs receive the inner chunks, and the index codecs receive the shard index.

    An inner chunk is "a chunk within the shard". Its shape is
    `chunk_shape`, and its data type is the data type of the shard
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L168-L170).
    The index is "an array with 64-bit unsigned integers with a shape that
    matches the chunks per shard tuple with an appended dimension of size
    2"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L186-L187),
    where chunks per shard is "the element-wise division of the shard shape
    by the inner chunk shape"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L173-L174).
    The number of chunks per shard is unknown along an axis if the shard
    lengths there are unknown, or if the inner chunk length does not
    divide them. If the input shard has a different number of axes than
    `chunk_shape`, both pipelines receive chunks with unknown lengths,
    because it is not known which of the two is wrong.
    """
    chunk_shape = configuration["chunk_shape"]
    lengths = chunk.lengths
    if lengths is not None and len(lengths) != len(chunk_shape):
        return {"codecs": Chunk(None, chunk.data_type), "index_codecs": Chunk(None, _UINT64)}
    inner = Chunk(tuple(frozenset({length}) for length in chunk_shape), chunk.data_type)
    index = Chunk((*_per_shard(chunk_shape, lengths), frozenset({2})), _UINT64)
    return {"codecs": inner, "index_codecs": index}


def _per_shard(chunk_shape: tuple[int, ...], lengths: Lengths | None) -> Lengths:
    """Return the number of inner chunks a shard holds along each axis. An entry is None where that number is unknown."""
    if lengths is None:
        return (None,) * len(chunk_shape)
    return tuple(
        None
        if shard is None or any(length % inner != 0 for length in shard)
        else frozenset(length // inner for length in shard)
        for inner, shard in zip(chunk_shape, lengths, strict=True)
    )


def _canonical(
    configuration: ShardingIndexedCodecConfiguration,
) -> ShardingIndexedCodecConfiguration:
    """Remove an `index_location` of `end`, because an absent `index_location` means the same thing.

    "If the parameter is not present, the value defaults to `end`"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L157-L161),
    so the two spellings describe the same codec.
    """
    if configuration.get("index_location") != "end":
        return configuration
    return cast(
        "ShardingIndexedCodecConfiguration",
        {key: value for key, value in configuration.items() if key != "index_location"},
    )


SHARDING_INDEXED_CODEC: Final = CodecDefinition(
    name=SHARDING_INDEXED_CODEC_NAME,
    configuration=ShardingIndexedCodecConfiguration,
    canonical=_canonical,
    kind="array_bytes",
    size="dynamic",
    chunk_rules=_chunk_rules,
    pipelines=_pipelines,
)
"""The `sharding_indexed` codec.

Its two pipelines are nested fields. Each codec in them is read in the
same scope as the shard, and each pipeline is read as a pipeline: the
inner codecs receive the inner chunks, and the index codecs receive the
shard index.
"""


__all__ = [
    "SHARDING_INDEXED_CODEC",
    "SHARDING_INDEXED_CODEC_NAME",
    "SHARDING_INDEX_LOCATION",
    "ShardingIndexLocation",
    "ShardingIndexedCodecConfiguration",
    "ShardingIndexedCodecMetadata",
    "ShardingIndexedCodecName",
    "ShardingIndexedCodecObject",
]
