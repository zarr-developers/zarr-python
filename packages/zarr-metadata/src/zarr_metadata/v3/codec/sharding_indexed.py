"""
Sharding-indexed codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
"""

from collections.abc import Iterator, Mapping
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import (
    Chunk,
    CodecDefinition,
    CodecField,
    Lengths,
    Nested,
    Resolved,
    StaticCodecField,
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

    `index_codecs` is the codec pipeline applied to the shard index, of
    codecs of static size only: a reader finds the index by a size it knows
    before reading it, so a compressor there is refused.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L147-L155

    `index_location` defaults to `"end"` per the spec.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L157-L161
    """

    chunk_shape: tuple[int, ...]
    codecs: tuple[CodecField, ...]
    index_codecs: tuple[StaticCodecField, ...]
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


def _rules(
    configuration: ShardingIndexedCodecConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    """Every inner chunk extent is at least 1."""
    for index, extent in enumerate(configuration["chunk_shape"]):
        if extent < 1:
            yield ValidationProblem(
                ("chunk_shape", index), f"expected an integer >= 1, got {extent}", "invalid_value"
            )


def _chunk_rules(
    configuration: ShardingIndexedCodecConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """An inner chunk length for each axis of the shard it is handed, dividing each length the shards take along it.

    "The length of the chunk_shape array must match the number of
    dimensions of the shard shape to which this sharding codec is applied,
    and the inner chunk shape along each dimension must evenly divide the
    size of the shard shape"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L131-L135).
    The shard is the chunk the codec is handed, after any codec before it:
    a transposed shard is divided along its transposed axes.
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


_UINT64: Final = Resolved(UINT64_DATA_TYPE_NAME, "read", UINT64_DATA_TYPE, {})
"""The data type of a shard index, which the spec fixes whatever the scope holds."""


def _pipelines(
    configuration: ShardingIndexedCodecConfiguration, nested: Nested, chunk: Chunk
) -> Mapping[str, Chunk]:
    """The inner codecs are handed the inner chunks; the index codecs, the shard index.

    An inner chunk is "a chunk within the shard", of the shape
    `chunk_shape` gives, and so of the shard's data type
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L168-L170).
    The index is "an array with 64-bit unsigned integers with a shape that
    matches the chunks per shard tuple with an appended dimension of size
    2"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L186-L187),
    chunks per shard being "the element-wise division of the shard shape by
    the inner chunk shape"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L173-L174):
    unknown along an axis whose shard lengths are unknown, or that the
    inner chunk length does not divide. Handed a shard of another number
    of axes than `chunk_shape` has, it hands both pipelines chunks of
    lengths nothing is known of: which of the two is wrong is not known.
    """
    chunk_shape = configuration["chunk_shape"]
    lengths = chunk.lengths
    if lengths is not None and len(lengths) != len(chunk_shape):
        return {"codecs": Chunk(None, chunk.data_type), "index_codecs": Chunk(None, _UINT64)}
    inner = Chunk(tuple(frozenset({length}) for length in chunk_shape), chunk.data_type)
    index = Chunk((*_per_shard(chunk_shape, lengths), frozenset({2})), _UINT64)
    return {"codecs": inner, "index_codecs": index}


def _per_shard(chunk_shape: tuple[int, ...], lengths: Lengths | None) -> Lengths:
    """Along each axis, how many inner chunks a shard holds; None where that is unknown."""
    if lengths is None:
        return (None,) * len(chunk_shape)
    return tuple(
        None
        if shard is None or any(length % inner != 0 for length in shard)
        else frozenset(length // inner for length in shard)
        for inner, shard in zip(chunk_shape, lengths, strict=True)
    )


SHARDING_INDEXED_CODEC: Final = CodecDefinition(
    name=SHARDING_INDEXED_CODEC_NAME,
    configuration=ShardingIndexedCodecConfiguration,
    kind="array_bytes",
    size="dynamic",
    rules=_rules,
    chunk_rules=_chunk_rules,
    pipelines=_pipelines,
)
"""The `sharding_indexed` codec.

Its two pipelines are nested fields, each codec in them read in the
scope the shard is read in, and each read as a pipeline: the inner codecs
handed the inner chunks, the index codecs the shard index.
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
