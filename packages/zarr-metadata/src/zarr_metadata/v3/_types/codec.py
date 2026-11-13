"""The types of the v3 codecs.

For each codec: the value and `Literal` type of its name, its
configuration and object forms, the JSON shapes its metadata may take,
and the constants that spell its permitted values.

This module holds types only and imports no definition, so a definition
of any kind can import a codec's types at run time. Each codec's
submodule of `zarr_metadata.v3.codec` re-exports its names, and holds the
codec's definition and rules.
"""

from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Ge, Interval
from typing_extensions import TypedDict

from zarr_metadata._common import EmptyConfiguration, JSONValue
from zarr_metadata.v3._common import (
    ZarrV3CodecDeclaration,
    ZarrV3DataTypeDeclaration,
    ZarrV3StaticCodecDeclaration,
)

# blosc: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html

BLOSC_CODEC_NAME: Final = "blosc"
"""The `name` field value of the `blosc` codec."""

BloscCodecName = Literal["blosc"]
"""Literal type of the `name` field of the `blosc` codec."""

BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
"""Literal type of blosc shuffle mode names."""

BLOSC_SHUFFLE: Final = ("noshuffle", "shuffle", "bitshuffle")
"""Tuple of permitted values for the `shuffle` field of the `blosc` codec."""

BloscCName = Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
"""Literal type of blosc compressor identifiers."""

BLOSC_CNAME: Final = ("lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd")
"""Tuple of permitted values for the `cname` field of the `blosc` codec."""


class BloscCodecConfiguration(TypedDict, closed=True):
    """Configuration for the Zarr v3 `blosc` codec."""

    cname: BloscCName
    """The compressor blosc applies to each block, one of `BLOSC_CNAME`."""
    clevel: Annotated[int, Interval(ge=0, le=9)]
    """The compression level, an integer from 0 to 9.

    0 turns compression off; 1 is the fastest and compresses least, 9 the
    slowest and compresses most.
    """
    shuffle: BloscShuffle
    """The shuffling applied before compression.

    `"noshuffle"` for none, `"shuffle"` for byte-wise, `"bitshuffle"` for
    bit-wise.
    """
    blocksize: Annotated[int, Ge(0)]
    """The size in bytes of the blocks the buffer is divided into before compression.

    At least 0; 0 lets blosc choose the size.
    """
    typesize: NotRequired[int]
    """The stride in bytes over which shuffling is performed.

    Required, and at least 1, unless `shuffle` is `"noshuffle"`, under
    which the spec ignores it and `canonicalize` drops it.
    """


class BloscCodecObject(TypedDict, closed=True):
    """`blosc` codec metadata in object form."""

    name: BloscCodecName
    """The codec's name, `"blosc"`."""
    configuration: BloscCodecConfiguration
    """The codec's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


BloscCodecMetadata = BloscCodecObject
"""Permitted JSON shape for `blosc` codec metadata.

The configuration has multiple required keys (`cname`, `clevel`, `shuffle`,
`blocksize`), so only the object form is valid; the short-hand-name form
is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/blosc/index.rst#L57-L98 (configuration parameters)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""

BLOSC_NO_SHUFFLE: Final = "noshuffle"
"""The `shuffle` value under which `typesize` carries no information.

The spec requires `typesize` "unless `shuffle` is `"noshuffle"`, in which
case the value is ignored", so this is the one value that changes whether
another member is required.
"""


# bytes: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html

BYTES_CODEC_NAME: Final = "bytes"
"""The `name` field value of the `bytes` codec."""

BytesCodecName = Literal["bytes"]
"""Literal type of the `name` field of the `bytes` codec."""

Endianness = Literal["little", "big"]
"""Literal type of byte order of multi-byte numeric data."""

ENDIANNESS: Final = ("little", "big")
"""Tuple of permitted values for the `endian` field of the `bytes` codec."""


class BytesCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `bytes` codec.

    The `endian` field is required for multi-byte data types.
    """

    endian: NotRequired[Endianness]
    """The byte order in which each value is written.

    Required when the values of the data type the codec is handed take more
    than one byte; a single-byte type such as `uint8` or `bool` does not
    depend on it.
    """


class BytesCodecObject(TypedDict, closed=True):
    """`bytes` codec metadata in object form.

    `configuration` is itself optional — when no configuration fields are
    set, the entire `configuration` key may be omitted. This matches the
    bare-string short-hand form (`BytesCodecName`) at the canonical data
    level; both encodings describe a `bytes` codec with default settings.
    """

    name: BytesCodecName
    """The codec's name, `"bytes"`."""
    configuration: NotRequired[BytesCodecConfiguration]
    """The codec's parameters.

    May be absent when no parameter is set.
    """
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


BytesCodecMetadata = BytesCodecObject | BytesCodecName
"""Permitted JSON shapes for `bytes` codec metadata.

The configuration has no required keys (`endian` is conditionally required
at runtime based on data type), so the spec's short-hand-name form is
permitted in addition to the object form, and the object form may itself
omit `configuration` entirely.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/bytes/index.rst#L64-L69 ("endian: Required for data types for which endianness is applicable")
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564
"""


# cast_value: https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md

CAST_VALUE_CODEC_NAME: Final = "cast_value"
"""The `name` field value of the `cast_value` codec."""

CastValueCodecName = Literal["cast_value"]
"""Literal type of the `name` field of the `cast_value` codec."""

CastRoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]
"""Literal type of permitted values for the `rounding` configuration field.

Defaults to `"nearest-even"` if absent.
"""

CAST_ROUNDING_MODE: Final = (
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
)
"""Tuple of permitted values for the `rounding` field of the `cast_value` codec."""

CastOutOfRangeMode = Literal["clamp", "wrap"]
"""Literal type of permitted values for the `out_of_range` configuration field.

If absent, out-of-range values are an encoding/decoding error.
"""

CAST_OUT_OF_RANGE_MODE: Final = ("clamp", "wrap")
"""Tuple of permitted values for the `out_of_range` field of the `cast_value` codec."""

ScalarMapEntry = tuple[JSONValue, JSONValue]
"""A single `[input, output]` mapping in a `scalar_map` direction.

Each scalar is encoded as a fill value of its data type. For a float,
the infinities are `"Infinity"` and `"-Infinity"`, as the core encoding
writes them. The codec's own example writes `"+Infinity"` instead.
"""


class ScalarMap(TypedDict, closed=True):
    """Optional encode/decode scalar overrides for the cast_value codec."""

    encode: NotRequired[tuple[ScalarMapEntry, ...]]
    """The `[input, output]` pairs applied when encoding: input in the array's data type, output in `data_type`.

    Each scalar is written in its data type's fill value encoding. Where an
    input repeats, the first pair applies.
    """
    decode: NotRequired[tuple[ScalarMapEntry, ...]]
    """The `[input, output]` pairs applied when decoding: input in `data_type`, output in the array's data type.

    Each scalar is written in its data type's fill value encoding. Where an
    input repeats, the first pair applies.
    """


class CastValueCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `cast_value` codec.

    `data_type` is the target data type that input values are cast to. It
    is the same shape as the top-level array `data_type` field: either a
    bare-string primitive name or a `{name, configuration}` envelope.
    """

    data_type: ZarrV3DataTypeDeclaration
    """The data type values are cast to when encoding, and the one the decoder is handed.

    The codec is defined only for data types that model real numbers, the
    integer and floating-point types: a boolean, raw-bits, string, bytes,
    struct or complex type is refused, here and as the data type the codec
    is handed.
    """
    rounding: NotRequired[CastRoundingMode]
    """How a value `data_type` cannot represent exactly is rounded.

    Absent means `"nearest-even"`.
    """
    out_of_range: NotRequired[CastOutOfRangeMode]
    """How a value outside the range of `data_type` is brought into it, after rounding.

    `"clamp"` takes the nearest bound; `"wrap"` takes the value congruent
    modulo 2^N, N being the width in bits of `data_type`, and is permitted
    only when `data_type` is an integral type. Absent means an out-of-range
    value is an encoding or decoding error.
    """
    scalar_map: NotRequired[ScalarMap]
    """Scalars mapped explicitly, in place of `rounding` and `out_of_range`."""


class CastValueCodecObject(TypedDict, closed=True):
    """`cast_value` codec metadata in object form."""

    name: CastValueCodecName
    """The codec's name, `"cast_value"`."""
    configuration: CastValueCodecConfiguration
    """The codec's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


CastValueCodecMetadata = CastValueCodecObject
"""Permitted JSON shape for `cast_value` codec metadata.

`configuration.data_type` is required, so only the object form is valid;
the short-hand-name form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md#L33-L36 and #L46-L48 (required fields)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""


# crc32c: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html

CRC32C_CODEC_NAME: Final = "crc32c"
"""The `name` field value of the `crc32c` codec."""

Crc32cCodecName = Literal["crc32c"]
"""Literal type of the `name` field of the `crc32c` codec."""


class Crc32cCodecObject(TypedDict, closed=True):
    """`crc32c` codec metadata in object form.

    Per spec the codec has no configuration fields. `configuration` is
    optional and, if present, should be an empty mapping.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/crc32c/index.rst#L63-L66
    """

    name: Crc32cCodecName
    """The codec's name, `"crc32c"`."""
    configuration: NotRequired[EmptyConfiguration]
    """The codec's parameters, of which there are none.

    Absent, or an empty mapping.
    """
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


Crc32cCodecMetadata = Crc32cCodecObject | Crc32cCodecName
"""Permitted JSON shapes for `crc32c` codec metadata.

The spec's Extension definition allows extensions with no required
configuration to be encoded as a bare short-hand name. CRC32C has no
configuration, so both forms are valid.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564
"""


# gzip: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html

GZIP_CODEC_NAME: Final = "gzip"
"""The `name` field value of the `gzip` codec."""

GzipCodecName = Literal["gzip"]
"""Literal type of the `name` field of the `gzip` codec."""


class GzipCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `gzip` codec.

    `level` is an integer in the range 0-9; 0 disables compression and 9
    is slowest with the best compression ratio. The codec's compressed
    output depends on `level`, so metadata that omits it cannot
    reproducibly identify the chunk bytes produced by a writer — `level`
    is required for the metadata to fulfill its reproducibility role,
    even though the spec text does not mark it required with RFC 2119
    keywords.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/gzip/index.rst#L57-L66
    """

    level: Annotated[int, Interval(ge=0, le=9)]
    """The compression level, an integer from 0 to 9.

    0 turns compression off; 1 is the fastest and compresses least, 9 the
    slowest and compresses most.
    """


class GzipCodecObject(TypedDict, closed=True):
    """`gzip` codec metadata in object form."""

    name: GzipCodecName
    """The codec's name, `"gzip"`."""
    configuration: GzipCodecConfiguration
    """The codec's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


GzipCodecMetadata = GzipCodecObject
"""Permitted JSON shape for `gzip` codec metadata.

`configuration.level` is required (it determines the codec's output bytes
and is therefore part of the metadata's reproducibility contract), so
only the object form is valid; the short-hand-name form is not permitted.
"""


# scale_offset: https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md

SCALE_OFFSET_CODEC_NAME: Final = "scale_offset"
"""The `name` field value of the `scale_offset` codec."""

ScaleOffsetCodecName = Literal["scale_offset"]
"""Literal type of the `name` field of the `scale_offset` codec."""


class ScaleOffsetCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `scale_offset` codec.

    Both fields are optional. A missing `offset` is the additive identity
    (e.g. 0 for numeric types); a missing `scale` is the multiplicative
    identity (e.g. 1). Each scalar is encoded as a fill value of the input
    data type of the codec: an integer for an integer type, and for a
    float type a number, `"NaN"`, `"Infinity"`, `"-Infinity"` or a hex
    string.
    """

    offset: NotRequired[JSONValue]
    """The quantity subtracted from each value when encoding.

    Absent means the additive identity, e.g. 0. A fill value of the data
    type the codec is handed.
    """
    scale: NotRequired[JSONValue]
    """The quantity each value is multiplied by when encoding, after `offset` is subtracted.

    Absent means the multiplicative identity, e.g. 1. A fill value of the
    data type the codec is handed.
    """


class ScaleOffsetCodecObject(TypedDict, closed=True):
    """`scale_offset` codec metadata in object form.

    `configuration` is itself optional per spec — when both `offset` and
    `scale` are at their identity defaults, the codec is a no-op and the
    entire `configuration` field may be omitted.
      https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md#L18 and #L35
    """

    name: ScaleOffsetCodecName
    """The codec's name, `"scale_offset"`."""
    configuration: NotRequired[ScaleOffsetCodecConfiguration]
    """The codec's parameters.

    May be absent when neither `offset` nor `scale` is set, which makes the
    codec a no-op.
    """
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


ScaleOffsetCodecMetadata = ScaleOffsetCodecObject | ScaleOffsetCodecName
"""Permitted JSON shapes for `scale_offset` codec metadata.

The configuration has no required keys (both `offset` and `scale` are
optional, and the configuration itself is optional), so the short-hand-name
form is permitted in addition to the object form.
"""


# sharding_indexed: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html

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
    """The shape of the inner chunks a shard is divided into.

    One length, at least 1, per axis of the shard the codec is handed, each
    dividing the shard's length along that axis.
    """
    codecs: tuple[ZarrV3CodecDeclaration, ...]
    """The codec pipeline that encodes each inner chunk.

    Like an array's `codecs`, it holds exactly one array -> bytes codec.
    """
    index_codecs: tuple[ZarrV3StaticCodecDeclaration, ...]
    """The codec pipeline that encodes the shard index.

    It holds exactly one array -> bytes codec, and only codecs whose output
    size is fixed by the size of their input, so no compressor. The spec
    recommends a little-endian `bytes` codec followed by `crc32c`.
    """
    index_location: NotRequired[ShardingIndexLocation]
    """Whether the shard index sits at the `"start"` or the `"end"` of the encoded shard.

    Absent means `"end"`.
    """


class ShardingIndexedCodecObject(TypedDict, closed=True):
    """`sharding_indexed` codec metadata in object form."""

    name: ShardingIndexedCodecName
    """The codec's name, `"sharding_indexed"`."""
    configuration: ShardingIndexedCodecConfiguration
    """The codec's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


ShardingIndexedCodecMetadata = ShardingIndexedCodecObject
"""Permitted JSON shape for `sharding_indexed` codec metadata.

The configuration has multiple required keys (`chunk_shape`, `codecs`,
`index_codecs`), so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L129-L138 (`chunk_shape`)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L141-L155 (`codecs` and `index_codecs`, the members the spec marks required)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""


# transpose: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html

TRANSPOSE_CODEC_NAME: Final = "transpose"
"""The `name` field value of the `transpose` codec."""

TransposeCodecName = Literal["transpose"]
"""Literal type of the `name` field of the `transpose` codec."""


class TransposeCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `transpose` codec.

    `order` is a permutation of the dimension indices 0..n-1 that
    specifies the dimension reordering applied during encoding.
    """

    order: tuple[int, ...]
    """The reordering of axes applied when encoding: axis `i` of the encoded chunk is axis `order[i]` of the chunk the codec is handed.

    A permutation of `0, 1, ..., n-1`, `n` being the number of axes of the
    chunk the codec is handed.
    """


class TransposeCodecObject(TypedDict, closed=True):
    """`transpose` codec metadata in object form."""

    name: TransposeCodecName
    """The codec's name, `"transpose"`."""
    configuration: TransposeCodecConfiguration
    """The codec's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


TransposeCodecMetadata = TransposeCodecObject
"""Permitted JSON shape for `transpose` codec metadata.

`order` is required, so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/transpose/index.rst#L60-L66 ("order: Required")
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""


# vlen_bytes: https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-bytes/README.md

VLEN_BYTES_CODEC_NAME: Final = "vlen-bytes"
"""The `name` field value of the `vlen-bytes` codec."""

VlenBytesCodecName = Literal["vlen-bytes"]
"""Literal type of the `name` field of the `vlen-bytes` codec."""


class VlenBytesCodecObject(TypedDict, closed=True):
    """`vlen-bytes` codec metadata in object form. The codec has no configuration fields."""

    name: VlenBytesCodecName
    """The codec's name, `"vlen-bytes"`."""
    configuration: NotRequired[EmptyConfiguration]
    """The codec's parameters, of which there are none.

    Absent, or an empty mapping.
    """
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


VlenBytesCodecMetadata = VlenBytesCodecObject | VlenBytesCodecName
"""Permitted JSON shapes for `vlen-bytes` codec metadata: the object form, or the bare name, since the codec has no configuration."""


# vlen_utf8: https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/codecs/vlen-utf8/README.md

VLEN_UTF8_CODEC_NAME: Final = "vlen-utf8"
"""The `name` field value of the `vlen-utf8` codec."""

VlenUtf8CodecName = Literal["vlen-utf8"]
"""Literal type of the `name` field of the `vlen-utf8` codec."""


class VlenUtf8CodecObject(TypedDict, closed=True):
    """`vlen-utf8` codec metadata in object form. The codec has no configuration fields."""

    name: VlenUtf8CodecName
    """The codec's name, `"vlen-utf8"`."""
    configuration: NotRequired[EmptyConfiguration]
    """The codec's parameters, of which there are none.

    Absent, or an empty mapping.
    """
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


VlenUtf8CodecMetadata = VlenUtf8CodecObject | VlenUtf8CodecName
"""Permitted JSON shapes for `vlen-utf8` codec metadata: the object form, or the bare name, since the codec has no configuration."""


# zstd: https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md

ZSTD_CODEC_NAME: Final = "zstd"
"""The `name` field value of the `zstd` codec."""

ZstdCodecName = Literal["zstd"]
"""Literal type of the `name` field of the `zstd` codec."""

ZSTD_MIN_LEVEL: Final = -131072
"""The lowest `level` zstd accepts: ZSTD_minCLevel(), -(1 << 17)."""

ZSTD_MAX_LEVEL: Final = 22
"""The highest `level` zstd accepts: ZSTD_maxCLevel()."""


class ZstdCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `zstd` codec.

    `level` is required; `checksum` is optional ("Should be omitted if
    false").
      https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md#L9-L19
    """

    level: Annotated[int, Interval(ge=ZSTD_MIN_LEVEL, le=ZSTD_MAX_LEVEL)]
    """The compression level, an integer from `ZSTD_MIN_LEVEL` (-131072) to `ZSTD_MAX_LEVEL` (22).

    0 means zstd's default level; otherwise a higher level compresses more,
    more slowly. It has no effect on decoding.
    """
    checksum: NotRequired[bool]
    """Whether a checksum is stored when writing and verified when reading.

    Absent means `false`; the registry says to omit it rather than write
    `false`.
    """


class ZstdCodecObject(TypedDict, closed=True):
    """`zstd` codec metadata in object form."""

    name: ZstdCodecName
    """The codec's name, `"zstd"`."""
    configuration: ZstdCodecConfiguration
    """The codec's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this codec must refuse the document.

    Absent means `true`. The spec permits `false` for a codec
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1581),
    though `resolve` reports a `false` as a problem at every extension point.
    """


ZstdCodecMetadata = ZstdCodecObject
"""Permitted JSON shape for `zstd` codec metadata.

`level` is required, so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md#L9-L19
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""


__all__ = [
    "BLOSC_CNAME",
    "BLOSC_CODEC_NAME",
    "BLOSC_NO_SHUFFLE",
    "BLOSC_SHUFFLE",
    "BYTES_CODEC_NAME",
    "CAST_OUT_OF_RANGE_MODE",
    "CAST_ROUNDING_MODE",
    "CAST_VALUE_CODEC_NAME",
    "CRC32C_CODEC_NAME",
    "ENDIANNESS",
    "GZIP_CODEC_NAME",
    "SCALE_OFFSET_CODEC_NAME",
    "SHARDING_INDEXED_CODEC_NAME",
    "SHARDING_INDEX_LOCATION",
    "TRANSPOSE_CODEC_NAME",
    "VLEN_BYTES_CODEC_NAME",
    "VLEN_UTF8_CODEC_NAME",
    "ZSTD_CODEC_NAME",
    "ZSTD_MAX_LEVEL",
    "ZSTD_MIN_LEVEL",
    "BloscCName",
    "BloscCodecConfiguration",
    "BloscCodecMetadata",
    "BloscCodecName",
    "BloscCodecObject",
    "BloscShuffle",
    "BytesCodecConfiguration",
    "BytesCodecMetadata",
    "BytesCodecName",
    "BytesCodecObject",
    "CastOutOfRangeMode",
    "CastRoundingMode",
    "CastValueCodecConfiguration",
    "CastValueCodecMetadata",
    "CastValueCodecName",
    "CastValueCodecObject",
    "Crc32cCodecMetadata",
    "Crc32cCodecName",
    "Crc32cCodecObject",
    "Endianness",
    "GzipCodecConfiguration",
    "GzipCodecMetadata",
    "GzipCodecName",
    "GzipCodecObject",
    "ScalarMap",
    "ScalarMapEntry",
    "ScaleOffsetCodecConfiguration",
    "ScaleOffsetCodecMetadata",
    "ScaleOffsetCodecName",
    "ScaleOffsetCodecObject",
    "ShardingIndexLocation",
    "ShardingIndexedCodecConfiguration",
    "ShardingIndexedCodecMetadata",
    "ShardingIndexedCodecName",
    "ShardingIndexedCodecObject",
    "TransposeCodecConfiguration",
    "TransposeCodecMetadata",
    "TransposeCodecName",
    "TransposeCodecObject",
    "VlenBytesCodecMetadata",
    "VlenBytesCodecName",
    "VlenBytesCodecObject",
    "VlenUtf8CodecMetadata",
    "VlenUtf8CodecName",
    "VlenUtf8CodecObject",
    "ZstdCodecConfiguration",
    "ZstdCodecMetadata",
    "ZstdCodecName",
    "ZstdCodecObject",
]
